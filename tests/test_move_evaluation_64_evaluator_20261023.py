"""Synthetic complete64 and fail-closed routing; no models/native jobs."""
from copy import deepcopy
import pytest
from tools.move_evaluation_64_20261023 import evaluator as e
from tools.move_evaluation_64_20261023 import audit as a
from tools.external_strength_screen_20261018 import rules
from test_move_evaluation_64_audit_20261023 import final_pass, observation


def fixture(monkeypatch):
    state = final_pass(); observer = observation(state)
    world = a.reference.world_from_observer(observer)
    legal = rules.legal_actions(state)
    proposal = a.reference.derive_proposal(observer, world, [rules.serialize_action(x) for x in legal],
                                            [0.], [0.], {'sha256': 'a'*64})
    payload = {'observer': observer, 'fit': observer['context']['fit'], 'candidates': proposal['candidates'],
               'selection_schedule': e.prefix_schedule(proposal['schedule'])}
    locked = deepcopy({'roots': [{'payload': payload,
                                  'original_full_schedule_sha256': e.digest(proposal['schedule'])}]})
    monkeypatch.setattr(e.c, 'read', lambda _: deepcopy(locked))
    monkeypatch.setattr(e.q, 'score_actions', lambda state, legal: [0.]*len(legal))
    monkeypatch.setattr(e.q, 'proposal_for_root', lambda *args: deepcopy(proposal))
    return payload, proposal


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_actual_choose_computes64_and_preserves_full128_proposal(monkeypatch, arm):
    payload, proposal = fixture(monkeypatch); traces = []
    original_config = deepcopy(e.q.old.CONFIG)
    decision = e.choose(payload, None if arm == 'A' else object(), arm=arm, trace_sink=traces.append)
    rec = decision['record']
    assert rec['proposal'] == proposal and len(proposal['schedule']['world_seeds']) == 128
    assert rec['schedule'] == payload['selection_schedule'] and len(rec['worlds']) == len(traces) == 64
    assert rec['choice'] == 0 and rec['means'] == [-1.]
    assert all(row['turns'] == 1 and row['reason'] == 'completed' for row in traces)
    assert e.q.old.CONFIG == original_config


@pytest.mark.parametrize('damage', ['extra', 'count63', 'count65', 'reverse', 'boolfit', 'candidate', 'oldrules', 'hash'])
def test_production_reproduction_rejects_changed_payload_before_inference(monkeypatch, damage):
    payload, _ = fixture(monkeypatch)
    monkeypatch.setattr(e.q, 'score_actions', lambda *args: pytest.fail('bad payload reached model'))
    if damage == 'extra': payload['assessment_seeds'] = [1]
    elif damage.startswith('count'): payload['selection_schedule']['world_seeds'] = ['x']*int(damage[5:])
    elif damage == 'reverse': payload['selection_schedule']['world_seeds'].reverse()
    elif damage == 'boolfit': payload['fit'] = True
    elif damage == 'candidate': payload['candidates']['actions'][0]['noble_id'] = 'N01'
    elif damage == 'oldrules': payload['observer']['ruleset_id'] = 'historical'
    else: payload['observer']['catalogue_sha256'] = '0'*64
    with pytest.raises(ValueError): e.reproduce(payload)


@pytest.mark.parametrize('count', [0, 63, 64, 65, 127, 129])
def test_prefix_requires_original128_not_already_reduced(count):
    with pytest.raises(ValueError):
        e.prefix_schedule({'belief_sha256': 'a', 'candidate_nonce': 'b',
                           'world_seeds': [str(i) for i in range(count)]})


def test_prefix_does_not_alias_or_accept_duplicates():
    full = {'belief_sha256': 'a', 'candidate_nonce': 'b', 'world_seeds': [str(i) for i in range(128)]}
    out = e.prefix_schedule(full); out['world_seeds'][0] = 'changed'
    assert full['world_seeds'][0] == '0'
    full['world_seeds'][-1] = full['world_seeds'][0]
    with pytest.raises(ValueError): e.prefix_schedule(full)


def test_arm_cannot_silently_use_other_policy():
    with pytest.raises(ValueError): e.trajectory(None, None, object(), arm='A')
    with pytest.raises(ValueError): e.trajectory(None, None, None, arm='B')
    with pytest.raises(ValueError): e.arm_profile('a')


def test64_arithmetic_agrees_with_independent_integer_reduction_and_first_tie():
    worlds = [{'outcomes': [{'wdl': float((j+i)%3-1), 'reason': 'completed'} for j in range(4)]}
              for i in range(64)]
    selected = e.select64(worlds, 4)
    choice, means = a.select_outcomes64(worlds, 4)
    assert selected == {'choice': choice, 'means': means} and choice == 2
    assert means[0] == means[3] == -1/64 and means[2] == 1/64
    tie = [{'outcomes': [{'wdl': 0., 'reason': 'completed'} for _ in range(4)]} for _ in range(64)]
    assert e.select64(tie, 4)['choice'] == a.select_outcomes64(tie, 4)[0] == 0
    with pytest.raises(ValueError): e.select64(worlds[:63], 4)
    with pytest.raises(ValueError): a.select_outcomes64(worlds[:63], 4)
