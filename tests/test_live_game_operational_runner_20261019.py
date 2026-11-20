"""Outcome-independent schedule, admission and failure-boundary checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import pytest

from tools.live_game_operational_20261019 import common as c, runner as r
from tools.external_strength_screen_20261018 import rules


def test_fixed_paired_schedule_and_disjoint_public_streams():
    rows = c.make_schedule()
    assert [x['game'] for x in rows] == [6,8,10,7,9,11,12,14,16,13,15,17,18,20,22,19,21,23]
    assert len({x['seed'] for x in rows}) == 9
    assert len({c.operational_context(x['game'], 0)['stream'] for x in rows}) == 18
    for pair in range(3, 12):
        fit = c.FITS[(pair-3)%3]
        two = [x for x in rows if x['pair'] == pair]
        assert {x['fit'] for x in two} == {fit}
        assert {x['local_seat'] for x in two} == {0, 1}
        assert len({x['seed'] for x in two}) == 1
        states = [rules.serialize_snapshot(rules.CorrectedEnv(seed=int(x['seed'])).initial_state()) for x in two]
        assert states[0] == states[1]
        for x in two:
            assert 'seed' not in c.operational_context(x['game'], 0)


def test_twelve_transfer_cases_and_hidden_variants_without_inference():
    cases = r.build_validation()
    assert len(cases) == 12
    assert [x['fixture'] for x in cases] == [0]*4+[6]*4+[12]*4
    assert {x['fit'] for x in cases} == set(c.FITS)
    for i in range(0, 12, 2):
        a, b = cases[i:i+2]
        assert a['snapshot'] != b['snapshot']
        assert a['observer'] == b['observer']
        assert a['kind'] == 'original' and b['kind'] == 'hidden_variant'
    assert [cases[i]['observer']['context']['game'] for i in (2, 6, 10)] == [6, 9, 10]


def test_transfer_projection_only_removes_declared_profile():
    decision = {'action': {'some': 'action'}, 'record': {'profile': 'new', 'choice': 2, 'worlds': [{'seed': 3}]}}
    expected = deepcopy(decision); del expected['record']['profile']
    assert r.scientific_projection(decision) == expected
    assert decision['record']['profile'] == 'new'
    for changed in ('choice', 'worlds'):
        altered = deepcopy(decision); altered['record'][changed] = None
        assert r.scientific_projection(altered) != expected


def test_unknown_audit_mode_rejected_before_reading_payload():
    with pytest.raises(ValueError, match='Unsupported audit mode'):
        c.check_audit_proof({}, {}, {}, 'accidentally_skip')


@pytest.mark.parametrize('status', ['timeout', 'worker_exit', 'malformed', 'wrong_request', 'illegal', 'unsolicited', 'write_error'])
def test_clock_forfeits_preserved_with_fresh_receipt(tmp_path, monkeypatch, status):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path)
    monkeypatch.setattr(r, 'worker_options', lambda *a, **k: {})
    monkeypatch.setattr(c, 'artifact', lambda p: {'path': str(p), 'sha256': c.file_hash(p), 'bytes': Path(p).stat().st_size})
    class FakeWorker:
        def __init__(self, module, folder, **options):
            self.folder = folder; folder.mkdir(parents=True)
            self.initialization = {'status': 'ok', 'result': {'pid': 1}}
            c.save_new(folder/'initialization.json', self.initialization)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def request(self, *args):
            value = {'status': status, 'accepted': False, 'receipt_elapsed_seconds': None}
            c.save_new(self.folder/'request-00000001.json', value); return value
    monkeypatch.setattr(r, 'PlannerWorker', FakeWorker)
    case = r.build_validation()[0]
    row = r.decision_attempt('fake', 'local', case['snapshot'], case['observer'], time.monotonic()+1000, 0, reserve=600)
    assert row['status'] == 'forfeit_'+status
    assert not row['correct'] and 'result_sha256' not in row
    assert c.read(tmp_path/'decision_attempts/fake.json') == row


def test_worker_error_is_hold_and_failed_attempt_survives(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path)
    monkeypatch.setattr(r, 'worker_options', lambda *a, **k: {})
    monkeypatch.setattr(c, 'artifact', lambda p: {'path': str(p), 'sha256': c.file_hash(p), 'bytes': Path(p).stat().st_size})
    class FakeWorker:
        def __init__(self, module, folder, **options):
            self.folder = folder; folder.mkdir(parents=True)
            self.initialization = {'status': 'ok'}; c.save_new(folder/'initialization.json', self.initialization)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def request(self, *args):
            value = {'status': 'worker_error', 'accepted': False}
            c.save_new(self.folder/'request-00000001.json', value); return value
    monkeypatch.setattr(r, 'PlannerWorker', FakeWorker)
    case = r.build_validation()[0]
    with pytest.raises(ValueError, match='Hard worker'):
        r.decision_attempt('fake', 'local', case['snapshot'], case['observer'], time.monotonic()+1000, 0, reserve=600)
    row = c.read(tmp_path/'decision_attempts/fake.json')
    assert row['status'] == 'incomplete' and row['failure_category'] == 'ValueError'


def test_move_reserve_blocks_before_initialization(monkeypatch):
    monkeypatch.setattr(r, 'PlannerWorker', lambda *a, **k: pytest.fail('worker must not start'))
    with pytest.raises((ValueError, c.ResourceHold), match='Insufficient move reserve'):
        r.decision_attempt('unused', 'local', {}, {}, time.monotonic()+100, 0, reserve=600)


def test_stress_strata_include_missing_cells_and_hash_selection_is_outcome_free():
    assert len(c.expected_stress_cells()) == 81
    case = r.build_validation()[2]
    obs = case['observer']; before = deepcopy(obs)
    cells = c.stress_cells(obs, 2)
    assert len(cells) == 2 and cells[0].startswith('ordinary:161001:0:0:')
    assert cells[1] == 'support:161001:2'
    assert obs == before
    identity = obs['context']['identity']
    expected = sorted(sorted(range(128), key=lambda i:
        (hashlib.sha256(json.dumps([c.AUDIT_DOMAIN, 'world', c.AUDIT_SEED, identity, i],
                                  sort_keys=True, separators=(',', ':')).encode()).hexdigest(), i))[:8])
    assert c.sample_worlds(identity) == expected and len(set(expected)) == 8


def test_global_resource_boundaries_leave_reserved_space(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'directory_bytes', lambda p: c.MAX_BYTES-c.EVIDENCE_RESERVE)
    monkeypatch.setattr(c, 'memory', lambda: {'available_physical': 2**34, 'available_page': 2**34})
    with pytest.raises(c.ResourceHold, match='evidence ceiling'):
        c.boundary(tmp_path, time.monotonic()+1000)


def test_game_forfeit_record_is_complete_but_progress_keeps_outcome_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path)
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'artifact', lambda p: {'path': Path(p).relative_to(tmp_path).as_posix(),
                                               'sha256': c.file_hash(p), 'bytes': Path(p).stat().st_size})
    spec = c.make_schedule()[1]
    def timeout(name, player, snapshot, observer, *args, **kwargs):
        assert player == 'opponent' and 'seed' not in observer
        return {'id': name, 'player': player, 'observer': observer,
                'status': 'forfeit_timeout', 'correct': False,
                'clock': {'status': 'timeout', 'receipt_elapsed_seconds': None}}
    monkeypatch.setattr(r, 'decision_attempt', timeout)
    game = r.play_game(spec, time.monotonic()+20000, 0)
    assert game['complete'] and game['reason'] == 'forfeit_timeout'
    assert game['winner'] == 1 and game['failing_player'] == 'opponent'
    assert not game['final']['state']['terminal']
    assert len(game['moves']) == 1
    progress = capsys.readouterr().out
    assert 'winner' not in progress and 'score' not in progress and 'outcomes_closed' in progress


def test_game_natural_transition_uses_dictionary_check_and_sealed_result(tmp_path, monkeypatch, capsys):
    original_env = rules.CorrectedEnv
    state = original_env(seed=812).initial_state()
    chosen = [card for card in rules.corrected_cards_by_id().values() if card.points == 5][:3]
    ids = {card.card_id for card in chosen}
    for tier in (1, 2, 3):
        for groups in (state.visible_tier_cards, state.hidden_tier_decks):
            groups[tier] = [card for card in groups[tier] if card.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop())
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    for card in chosen:
        state.players[0].purchased_cards.append(card)
        state.players[0].score += card.points
        state.players[0].bonuses[card.bonus_color] += 1
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    initial = rules.serialize_snapshot(state)
    class FixtureEnv(original_env):
        def initial_state(self): return rules.deserialize_snapshot(initial)
    monkeypatch.setattr(rules, 'CorrectedEnv', FixtureEnv)
    monkeypatch.setattr(c, 'OUTPUT', tmp_path)
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'artifact', lambda p: {'path': Path(p).relative_to(tmp_path).as_posix(),
                                               'sha256': c.file_hash(p), 'bytes': Path(p).stat().st_size})
    action = rules.serialize_action(original_env().legal_actions(state)[0])
    expected_after = rules.serialize_snapshot(original_env().step(state, rules.deserialize_action(action)))
    def timely(name, player, snapshot, observer, *args, **kwargs):
        assert player == 'local' and observer['context']['game'] == 8
        return {'id': name, 'player': player, 'observer': observer, 'status': 'completed', 'correct': True,
                'after_sha256': c.digest(expected_after), 'clock': {'receipt_elapsed_seconds': .01}}
    monkeypatch.setattr(r, 'decision_attempt', timely)
    monkeypatch.setattr(r, 'completed_result', lambda row: {'decision': {'action': action}})
    game = r.play_game(c.make_schedule()[1], time.monotonic()+20000, 0)
    assert game['complete'] and game['reason'] == 'natural' and game['final'] == expected_after
    assert game['final']['state']['terminal'] and len(game['moves']) == 1
    assert 'winner' not in capsys.readouterr().out


@pytest.mark.parametrize('damage', ['none', 'accepted_artifact', 'preserved_failure'])
def test_pilot_acceptance_artifacts_are_rechecked_before_refreezing(monkeypatch, damage):
    original = {'inputs': [{'path': 'old_input'}]}
    repair = {'inputs': [{'path': 'repair_input'}], 'preserved_failure': [{'path': 'failure_receipt'}],
              **{k: {'path': k} for k in ('original_analyzer','original_completion',
                   'original_final_input_check','original_freeze','original_launch')}}
    final = {'status':'accepted','outcomes_closed':True,'strength_analysis_games':0,
             'artifacts':[{'path':'accepted_artifact'}, {'path':'AGENT_CURRENT.md'}, {'path':'AGENT_HANDOFF.md'}]}
    accepted = {'status':'passed','outcomes_closed':True,'strength_analysis_games':0}
    mapping = {'freeze.json': original, 'analyzer_repair_02/freeze.json': repair,
               'independent_analysis_v2.json': accepted, 'finalization.json': final}
    monkeypatch.setattr(c,'read',lambda p: mapping[Path(p).relative_to(c.PRIOR_PILOT).as_posix()])
    monkeypatch.setattr(c,'file_hash',lambda p: r.PILOT_HASHES[Path(p).relative_to(c.PRIOR_PILOT).as_posix()])
    checked = []
    def verify(items):
        checked.extend(item['path'] for item in items)
        if damage == 'accepted_artifact' and any(item['path']=='accepted_artifact' for item in items):
            raise ValueError('Changed accepted artifact')
        if damage == 'preserved_failure' and any(item['path']=='failure_receipt' for item in items):
            raise ValueError('Changed preserved failure')
    monkeypatch.setattr(c,'verify',verify)
    if damage == 'none':
        assert r.authenticated_pilot() == (original,repair)
        assert 'accepted_artifact' in checked and 'failure_receipt' in checked
        assert 'AGENT_CURRENT.md' not in checked and 'AGENT_HANDOFF.md' not in checked
    else:
        with pytest.raises(ValueError, match='Changed'): r.authenticated_pilot()
