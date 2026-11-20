"""No model calls: parent secrecy, fresh seeds, clocks, gating and evidence."""
from contextlib import nullcontext
from copy import deepcopy
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.move_evaluation_64_20261023 import assessment as a, common as c
from tools.move_evaluation_20261023 import audit as reference
from tools.external_strength_screen_20261018 import rules


def panel():
    prep = c.read(c.PREP/'selection_lock.json')
    roots, choices = [], []
    for index, source in enumerate(prep['roots']):
        row = deepcopy(source['payload'])
        row.update({k: source[k] for k in ('id', 'fit', 'game')})
        row['control_choice'] = 0
        row['control_action'] = row['candidates']['actions'][0]
        row['assessment_world_seeds_parent_only'] = [str(a.seed_for(row['id'], j)) for j in range(32)]
        alternative = int(index%3 == 0)
        choices.append({'id': row['id'], 'fit': row['fit'], 'control_choice': 0, 'alternative_choice': alternative,
                        'control_action': row['control_action'], 'alternative_action': row['candidates']['actions'][alternative]})
        roots.append(row)
    return {'roots': roots}, choices


def test_parent_imports_and_actual_seed_reconstruction_never_import_torch():
    code = '''import sys
from tools.move_evaluation_64_20261023 import assessment as a, analysis, common as c
from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics
from tools.move_evaluation_20261023.qualification import build_fixtures
source = c.read(c.PREP/'selection_lock.json')['roots'][0]
row = dict(source['payload'], id=source['id'])
row['assessment_world_seeds_parent_only'] = [str(a.seed_for(row['id'], j)) for j in range(32)]
state = a.initial_assessment_world(row, 0)
assert len(build_fixtures()) == 15
assert 'torch' not in sys.modules
print('parent_model_free')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=c.ROOT,
        env={**os.environ, 'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT)},
        capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'parent_model_free'


def test_new_world_sampler_and_independent_validator_agree():
    locked, _ = panel()
    row = locked['roots'][0]
    world = a.initial_assessment_world(row, 7)
    seed = a.seed_for(row['id'], 7)
    proof = a.audit_assessment_initial(row['observer'], seed, rules.serialize_snapshot(world))
    assert proof['world_index'] == 7
    assert proof['domain'] == a.DOMAIN
    old = int(c.digest(['move_evaluation_20261023_assessment_v1', row['id'], 7]), 16)
    with pytest.raises(ValueError, match='integer'):
        a.audit_assessment_initial(row['observer'], old, rules.serialize_snapshot(world))
    for invalid in (str(seed), True, -1):
        with pytest.raises(ValueError, match='integer'):
            a.audit_assessment_initial(row['observer'], invalid, rules.serialize_snapshot(world))
    broken = rules.serialize_snapshot(world)
    broken['state']['hidden_tier_decks']['1'].reverse()
    with pytest.raises(ValueError, match='initial world'):
        a.audit_assessment_initial(row['observer'], seed, broken)


@pytest.mark.parametrize('damage', ['historical_control', 'wrong_action', 'boolean', 'duplicate', 'order',
                                    'secret_seed', 'short_seed', 'extra', 'fit'])
def test_invalid_sealed_interventions_rejected(damage):
    locked, choices = panel()
    assert len(a.validate_choices(locked, choices)) == 6
    if damage == 'historical_control': locked['roots'][0]['control_choice'] = 1
    elif damage == 'wrong_action': choices[0]['alternative_action'] = choices[0]['control_action']
    elif damage == 'boolean': choices[0]['control_choice'] = False
    elif damage == 'duplicate': locked['roots'][0]['candidates']['actions'][1] = choices[0]['control_action']
    elif damage == 'order': choices[0], choices[1] = choices[1], choices[0]
    elif damage == 'secret_seed': locked['roots'][0]['assessment_world_seeds_parent_only'][0] = '123'
    elif damage == 'short_seed': locked['roots'][0]['assessment_world_seeds_parent_only'].pop()
    elif damage == 'extra': choices[0]['assessment_outcome'] = 1
    else: choices[0]['fit'] = 161002
    with pytest.raises(ValueError):
        a.validate_choices(locked, choices)


def test_all_sealed_choices_and_qualification_binding_required(monkeypatch, tmp_path):
    locked, choices = panel()
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    seal = c.save(tmp_path/'sealed/choices.json', choices)
    bound_lock = c.save(tmp_path/'sealed/evaluated_parent_lock.json', locked)
    c.save(tmp_path/'qualification_complete.json', {'status': 'passed', 'choices': seal, 'lock': bound_lock})
    assert len(a.sealed_admission(locked, choices, tmp_path)) == 6
    changed = deepcopy(choices)
    changed[1]['alternative_choice'] = 1
    changed[1]['alternative_action'] = locked['roots'][1]['candidates']['actions'][1]
    with pytest.raises(ValueError, match='immutable seal'):
        a.sealed_admission(locked, changed, tmp_path)
    changed_lock = deepcopy(locked)
    changed_lock['roots'][0]['unapproved_metadata'] = 'changed'
    with pytest.raises(ValueError, match='parent lock'):
        a.sealed_admission(changed_lock, choices, tmp_path)


def test_throughput_projection_is_outcome_blind_and_keeps_contingency():
    gate = a.throughput_gate(elapsed_seconds=10., added_bytes=100, current_bytes=1000, remaining_seconds=420.)
    assert gate['status'] == 'passed'
    assert gate['remaining_seconds_projection_35pct'] == 418.5
    assert gate['evidence_bytes_projection_35pct'] == 5185.
    assert gate['outcomes_inspected_for_gate'] is False
    assert a.throughput_gate(elapsed_seconds=10., added_bytes=100, current_bytes=1000,
                             remaining_seconds=418.5)['status'] == 'hold'
    assert a.throughput_gate(elapsed_seconds=1., added_bytes=c.GiB, current_bytes=c.GiB,
                             remaining_seconds=1000.)['status'] == 'hold'
    for invalid in (float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            a.throughput_gate(elapsed_seconds=invalid, added_bytes=0, current_bytes=0, remaining_seconds=100.)


def test_insufficient_breadth_launches_no_assessment_worker(monkeypatch, tmp_path):
    locked, choices = panel()
    for choice in choices:
        choice.update(alternative_choice=0, alternative_action=choice['control_action'])
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    seal = c.save(tmp_path/'sealed/choices.json', choices)
    bound_lock = c.save(tmp_path/'sealed/evaluated_parent_lock.json', locked)
    c.save(tmp_path/'qualification_complete.json', {'status': 'passed', 'choices': seal, 'lock': bound_lock})
    result = a.run_assessment(None, {}, locked, choices,
        lambda *args: pytest.fail('No worker may start'), lambda *args: pytest.fail('No model requests'), out=tmp_path)
    assert result['status'] == 'insufficient_intervention_breadth'
    assert result['assessment_started'] is False


@pytest.mark.parametrize('remaining_budget,expected_branches', [(1000., 384), (30., 12)])
def test_balanced_wave_finishes_before_outcome_blind_continuation_gate(monkeypatch, tmp_path, remaining_budget, expected_branches):
    locked, choices = panel()
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    seal = c.save(tmp_path/'sealed/choices.json', choices)
    bound_lock = c.save(tmp_path/'sealed/evaluated_parent_lock.json', locked)
    c.save(tmp_path/'qualification_complete.json', {'status': 'passed', 'choices': seal, 'lock': bound_lock})
    seen, workers = [], []
    monkeypatch.setattr(a, 'play_branch', lambda child, row, choice, j, arm, *args, **kwargs:
                        seen.append((row['id'], j, arm)))
    ticks = iter([10., 20., 20.])
    monkeypatch.setattr(a.time, 'monotonic', lambda: next(ticks))
    envelope = SimpleNamespace(collection_end=2000., effective_deadline=lambda deadline: remaining_budget,
                               guard=lambda d: None, admit=lambda d: None)
    def make_worker(*args):
        workers.append(args[1])
        return nullcontext(None)
    call = lambda: a.run_assessment(envelope, {}, locked, choices, make_worker,
                                     lambda *args: pytest.fail('No model calls'), bytes_used=lambda: 0)
    if expected_branches == 12:
        with pytest.raises(ValueError, match='throughput'):
            call()
    else:
        assert call()['branches'] == 384
    assert workers == ['assess']
    assert len(seen) == expected_branches
    assert len({root for root, _, _ in seen[:12]}) == 6
    assert all(j == 0 for _, j, _ in seen[:12])
    assert [arm for _, _, arm in seen[:12]] == ['A', 'B']*6
    assert c.read(tmp_path/'throughput_gate.json')['outcomes_inspected_for_gate'] is False


def test_assessor_qualification_binds_every_fixture_and_rejects_response_identity(monkeypatch, tmp_path):
    from tools.move_evaluation_20261023.qualification import build_fixtures
    from tools import analyze_external_strength_screen_20261018_02 as audit
    fixtures = build_fixtures()
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    c.save(tmp_path/'fixtures.json', fixtures)
    monkeypatch.setattr(audit, 'check_clock', lambda *args: None)
    monkeypatch.setattr(audit, 'check_diagnostics', lambda *args: {'synthetic_quota_check': 1})
    envelope = SimpleNamespace(qual_end=1000., guard=lambda d: None)
    child = SimpleNamespace(output_dir=tmp_path/'fake_worker')
    seen = []
    def request(worker, payload, seconds, deadline):
        assert set(payload) == {'observer', 'player'} and seconds == 1. and deadline == envelope.qual_end
        state = rules.state_from_observation(payload['observer'])
        legal = [rules.serialize_action(action) for action in rules.legal_actions(state)]
        raw = payload['player'] == 'raw035'
        result = {'action': legal[0], 'diagnostics': {'policy': 'raw035', 'scores_sha256': 'a'*64,
                  'selected_index': 0, 'legal_count': len(legal)} if raw else {},
                  'profile_sha256': None if raw else c.PROFILE_SHA, 'player': payload['player'],
                  'observer_sha256': c.digest(payload['observer']), 'legal_actions_sha256': c.digest(legal),
                  'legal_count': len(legal), 'ruleset': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}
        seen.append(payload)
        receipt = {'request_id': f'request-{len(seen):08d}', 'status': 'ok', 'result': result}
        c.save(child.output_dir/(receipt['request_id']+'.json'), receipt)
        return receipt
    a.qualify_assessor(envelope, {}, lambda *args: nullcontext(child), request)
    proof = c.read(tmp_path/'assessor_qualification.json')
    assert len(seen) == len(proof['records']) == 30
    assert {row['fixture'] for row in proof['records']} == {row['id'] for row in fixtures}
    assert proof['actual_puct_internal_edges_replayed'] is False
    def malformed(*args):
        response = request(*args)
        response['result']['catalogue_sha256'] = 'bad'
        return response
    with pytest.raises(ValueError, match='response identity'):
        a.qualify_assessor(envelope, {}, lambda *args: nullcontext(child), malformed)


def final_pass():
    state = rules.CorrectedEnv(seed=301).initial_state()
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    for tier in (1, 2, 3):
        for card in state.visible_tier_cards[tier]+state.hidden_tier_decks[tier]:
            player = state.players[0]
            player.purchased_cards.append(card)
            player.bonuses[card.bonus_color] += 1
            player.score += card.points
        state.visible_tier_cards[tier], state.hidden_tier_decks[tier], state.deck_counts[tier] = [], [], 0
    for seat, counts in enumerate(({'white': 4, 'blue': 4, 'green': 2}, {'green': 2, 'red': 4, 'black': 4})):
        for color, count in counts.items():
            state.players[seat].tokens[color] = count
            state.bank_tokens[color] -= count
    return state


def test_branch_uses_effective_clock_checks_durable_evidence_and_official_target(monkeypatch, tmp_path):
    locked, choices = panel()
    state = final_pass()
    choice = deepcopy(choices[0])
    choice['control_action'] = rules.serialize_action(rules.legal_actions(state)[0])
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(a, 'initial_assessment_world', lambda row, j: state)
    monkeypatch.setattr(a.time, 'monotonic', lambda: next(ticks))
    ticks = iter([10., 11., 12., 12.])
    deadlines = []
    envelope = SimpleNamespace(collection_end=1000., end=2000., effective_deadline=lambda d: 80.,
                               guard=lambda d: deadlines.append(d))
    branch = a.play_branch(None, locked['roots'][0], choice, 0, 'A', envelope,
                          lambda *args: pytest.fail('Forced ending has no model request'), out=tmp_path)
    assert branch['branch_deadline_monotonic'] == 80.
    assert set(deadlines) == {80.}
    assert branch['reason'] == 'completed' and branch['score'] == 0
    proof = reference.audit_branch({key: branch[key] for key in a.SCIENTIFIC_FIELDS})
    assert proof['official_terminal'] is True
    path = tmp_path/'assessment'/locked['roots'][0]['id'].replace('/', '_')/'w00_A/branch.json'
    a.validate_branch_files(path, branch, locked['roots'][0], choice, 0, 'A', envelope, set())
    changed = deepcopy(branch)
    changed['elapsed_seconds'] = 120.1
    with pytest.raises(ValueError, match='clock'):
        a.validate_branch_files(path, changed, locked['roots'][0], choice, 0, 'A', envelope, set())


def test_branch_failure_preserves_partial_evidence_without_imputed_outcome(monkeypatch, tmp_path):
    locked, choices = panel()
    state = rules.CorrectedEnv(seed=301).initial_state()
    state.current_player, state.turn_index = 1, 399
    choice = deepcopy(choices[0])
    choice['control_action'] = rules.serialize_action(rules.legal_actions(state)[0])
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(a, 'initial_assessment_world', lambda row, j: state)
    envelope = SimpleNamespace(collection_end=float('inf'), end=float('inf'),
                               effective_deadline=lambda d: d, guard=lambda d: None)
    with pytest.raises(ValueError, match='ceiling'):
        a.play_branch(None, locked['roots'][0], choice, 0, 'A', envelope,
                      lambda *args: pytest.fail('Turn ceiling before another request'), out=tmp_path)
    folder = tmp_path/'assessment'/locked['roots'][0]['id'].replace('/', '_')/'w00_A'
    assert (folder/'initial.json').exists() and (folder/'steps/0000.json').exists()
    assert not (folder/'branch.json').exists() and not (folder/'completion.json').exists()
