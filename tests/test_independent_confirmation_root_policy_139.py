from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from splendor_ai.eval import run_independent_confirmation_root_policy as audit


def source_panel():
    rows = []
    for cell in audit.CELLS:
        seat, phase = cell.split('|')
        for i in range(24):
            rows.append({'sample_index': len(rows), 'source_seed': len(rows),
                         'turn_index': i, 'actor_seat': int(seat[-1]), 'phase': phase,
                         'selection_cell': cell})
    return {'selected': rows}


def test_selection_disjoint_balanced_deterministic_and_outcome_blind():
    source = source_panel()
    rows = audit.select_states(source)
    assert len(rows) == len({r['source_seed'] for r in rows}) == 60
    assert not {r['source_seed'] for r in rows} & {r['source_seed'] for r in audit.base.select_states(source)}
    assert all(sum(r['selection_cell'] == c for r in rows) == 10 for c in audit.CELLS)
    altered = {'selected': [{**r, 'winner': 999, 'payoff': float('nan'), 'expanded_action': 'poison'}
                            for r in reversed(source['selected'])]}
    assert [r['source_seed'] for r in rows] == [r['source_seed'] for r in audit.select_states(altered)]


def test_seed_slots_unique_disjoint_from_test_and_138_but_checks_paired():
    check = {audit.continuation_seed(s, audit.CHECKS[0], b, w, r)
             for s in range(300) for b in range(8) for w in range(3) for r in range(4)}
    fresh = {audit.continuation_seed(s, audit.CHECKS[1], b, w, r)
             for s in range(300) for b in range(8) for w in range(3) for r in range(4)}
    test = {audit.continuation_seed(s, 'test', b, w, r)
            for s in range(300) for b in range(8) for w in range(3) for r in range(4)}
    assert len(check) == len(test) == 28_800 and check == fresh and not check & test
    assert min(check) > audit.base.continuation_seed(299, 3, 2, 3)
    assert max(test) < 1_399_000_000_000_000_000
    with pytest.raises(ValueError):
        audit.continuation_seed(0, 'test', 8, 0, 0)


def test_world_schedule_preserves_observer_and_separates_check_test_and_138():
    env = audit.base.SplendorEnv(seed=139)
    state = env.initial_state()
    sample = {'sample_index': 139, 'source_seed': 860139, 'turn_index': 0,
              'actor_seat': 0, 'state_snapshot': audit.base.serialize_state_snapshot(state)}
    world_hashes = lambda worlds: [audit.digest(audit.base.serialize_state_snapshot(w)) for w in worlds]
    original = world_hashes(audit.worlds_for(sample, audit.CHECKS[0], 0))
    assert original == world_hashes(audit.worlds_for(sample, audit.CHECKS[0], 7))
    assert original == world_hashes(audit.base.worlds_for(sample, 'original_worlds', 0))
    fresh, test = audit.worlds_for(sample, audit.CHECKS[1], 0), audit.worlds_for(sample, 'test', 0)
    all_hashes = original + world_hashes(fresh) + world_hashes(test)
    assert len(set(all_hashes)) == 9
    assert set(all_hashes).isdisjoint(world_hashes(audit.base.worlds_for(sample, 'fresh_worlds', 0)))
    key = audit.base.canonical_public_belief_key(state, 0)
    assert all(audit.base.canonical_public_belief_key(w, 0) == key for w in fresh+test)


def test_preflight_accepts_json_key_normalization_without_changing_game_state():
    env = audit.base.SplendorEnv(seed=139)
    state = env.initial_state()
    keys = [repr(audit.base.canonical_successor_action_key(state, a)) for a in env.legal_actions(state)[:2]]
    sample = {'sample_index': 139, 'source_seed': 860139, 'turn_index': 0, 'actor_seat': 0,
              'champion_action': keys[0], 'expanded_action': keys[1],
              'state_snapshot': json.loads(json.dumps(audit.base.serialize_state_snapshot(state)))}
    class OrderedScorer:
        def score_actions(self, state, actions):
            return tuple(SimpleNamespace(action=a, score=float(len(actions)-i), action_index=i)
                         for i, a in enumerate(actions))
    record = audit.verify_sample(sample, OrderedScorer())
    assert len(record['blocks']) == 24
    assert all(b['expected_continuations'] == 96 for b in record['blocks'])


def test_screen_requires_magnitude_and_consistency_and_distinct_proposal():
    assert audit.screen([.05]*8)['switch']
    assert not audit.screen([.024]*8)['switch']
    assert not audit.screen([.05]*8, different=False)['switch']
    assert not audit.screen([.4, -.3]*4)['switch']
    assert not audit.screen([0.]*8)['switch']
    with pytest.raises(ValueError):
        audit.screen([.1]*7)
    with pytest.raises(ValueError):
        audit.screen([float('nan')]*8)


def test_shared_reply_membership_ignores_order_but_rejects_changed_actions(monkeypatch):
    replies = {'a': ['x', 'y'], 'b': ['y', 'x'], 'c': ['x', 'y']}
    def one_world(worlds, keys, scorer):
        return [{'world': 0, 'root': ('root',), 'information_set': ('shared',),
                 'after': worlds[0], 'replies': [SimpleNamespace(action=r) for r in replies[worlds[0]]]}]
    monkeypatch.setattr(audit.base, 'prepare_branches', one_world)
    monkeypatch.setattr(audit.base, 'canonical_successor_action_key', lambda state, action: (action,))
    branches = audit.prepare_branches(('a', 'b', 'c'), (('root',),), None)
    assert [b['world'] for b in branches] == [0, 1, 2]
    assert [r.action for r in branches[1]['replies']] == ['y', 'x']
    replies['c'] = ['x', 'z']
    with pytest.raises(ValueError, match='membership'):
        audit.prepare_branches(('a', 'b', 'c'), (('root',),), None)


def check_blocks(same=.05, fresh=-.05):
    return [{'condition': c, 'block': b, 'gains': {'margin': same if c == audit.CHECKS[0] else fresh}}
            for c in audit.CHECKS for b in range(8)]


def test_decisions_cannot_include_test_returns_or_missing_duplicate_blocks():
    sample = {'champion_action': 'a', 'expanded_action': 'b'}
    blocks = check_blocks()
    decisions = audit.decisions_from_checks(sample, blocks)
    assert decisions['same_world']['switch'] and not decisions['independent']['switch']
    for broken in (blocks[:-1], blocks+[{'condition': 'test', 'block': 0}], blocks[:-1]+blocks[:1]):
        with pytest.raises(ValueError):
            audit.decisions_from_checks(sample, broken)


def test_test_outcomes_never_reselect_and_defaulting_cannot_create_gain():
    decisions = {'same_world': {'switch': True}, 'independent': {'switch': False}}
    assert audit.policy_gains(-.5, decisions) == {'champion': 0., 'unfiltered': -.5, 'same_world': -.5, 'independent': 0.}
    assert audit.policy_gains(.5, decisions)['independent'] == 0.
    decisions['independent']['switch'] = True
    assert audit.policy_gains(-.5, decisions)['independent'] == -.5


def test_test_worker_produces_no_decisions_and_only_test_blocks(monkeypatch):
    sample = {'sample_index': 0, 'schedule_index': 0}
    seen = []
    def fake(s, c, b, scorer):
        seen.append((c, b))
        return {'condition': c, 'block': b}
    monkeypatch.setattr(audit, 'evaluate_block', fake)
    row = audit.evaluate_stage(sample, 'test')
    assert 'decisions' not in row
    assert seen == [('test', i) for i in range(8)]


def test_sealing_rejects_incomplete_checks_and_binds_full_records():
    samples = [{'sample_index': i} for i in range(60)]
    rows = [{'sample_index': i, 'decisions': {'independent': {'switch': False}}} for i in range(60)]
    with pytest.raises(ValueError):
        audit.frozen_choices(rows[:-1], samples, {})
    before = audit.frozen_choices(rows, samples, {})
    rows[0]['decisions']['independent']['switch'] = True
    after = audit.frozen_choices(rows, samples, {})
    assert before['choices'][0]['check_row_sha256'] != after['choices'][0]['check_row_sha256']


def test_bootstrap_is_paired_game_weighted_and_order_independent():
    rows = [{'source_seed': i, 'selection_cell': audit.CELLS[i//10],
             'policy_gains': {'margin': {'champion': 0., 'unfiltered': i/100,
                                       'same_world': i/100-.01, 'independent': i/100+.02}}}
            for i in range(60)]
    result = audit.paired_summary(rows, 1000)
    assert result == audit.paired_summary(list(reversed(rows)), 1000)
    assert result['independent_minus_unfiltered']['mean'] == pytest.approx(.02)
    assert result['independent_minus_unfiltered']['ci95'] == pytest.approx([.02, .02])
    assert result['independent_minus_champion']['mean'] == pytest.approx(.315)


def test_success_requires_dose_default_gain_and_both_comparators():
    summary = {f'{a}_minus_{b}': {'mean': .03, 'ci95': [.01, .05]} for a, b in audit.CONTRASTS}
    assert audit.classify(summary, 8) == 'independent_confirmation_policy_improvement_supported'
    assert audit.classify(summary, 5) == 'insufficient_independent_switch_dose'
    assert audit.classify(summary, 8, False) == 'independent_confirmation_integrity_failed'
    summary['independent_minus_same_world']['ci95'][0] = -.01
    assert audit.classify(summary, 8) == 'default_improvement_supported_confirmation_benefit_unproven'
    summary['independent_minus_champion'] = {'mean': .005, 'ci95': [.001, .009]}
    assert audit.classify(summary, 8) == 'independent_confirmation_policy_inconclusive'
    summary['independent_minus_champion'] = {'mean': -.03, 'ci95': [-.05, -.01]}
    assert audit.classify(summary, 8) == 'no_positive_default_improvement_supported'


def saved_test_fixture():
    sample = {'sample_index': 0, 'champion_action': 'a', 'expanded_action': 'b'}
    blocks = [{'condition': 'test', 'block': i, 'world_hashes': [str(i)],
               'telemetry': {'continuations': 8, 'resolved_continuations': 8},
               'values': {'a': {m: 0. for m in audit.METRICS}, 'b': {m: .1 for m in audit.METRICS}},
               'gains': {m: .1 for m in audit.METRICS}} for i in range(8)]
    structure = {'blocks': [{'condition': 'test', 'block': i, 'world_hashes': [str(i)],
                             'expected_continuations': 8} for i in range(8)]}
    payload = {'identity': {'frozen_choices': '123'}, 'row': {'stage': 'test', 'blocks': blocks,
                                                          'sample_sha256': audit.digest(sample)}}
    return sample, structure, payload


def test_resume_rejects_changed_choices_worlds_accounting_and_gains():
    sample, structure, payload = saved_test_fixture()
    audit.validate_saved(payload, sample, payload['identity'], 'test', structure)
    with pytest.raises(ValueError, match='identity'):
        audit.validate_saved(payload, sample, {'frozen_choices': 'changed'}, 'test', structure)
    for field, value in [('world_hashes', ['changed']), ('gains', {m: .2 for m in audit.METRICS}),
                         ('telemetry', {'continuations': 7, 'resolved_continuations': 7})]:
        broken = deepcopy(payload)
        broken['row']['blocks'][0][field] = value
        with pytest.raises(ValueError):
            audit.validate_saved(broken, sample, payload['identity'], 'test', structure)


def test_fewer_replies_and_terminal_roots_count_exactly(monkeypatch):
    sample = {'sample_index': 0, 'actor_seat': 0, 'champion_action': "('a',)", 'expanded_action': "('a',)"}
    state = SimpleNamespace(terminal=False)
    monkeypatch.setattr(audit, 'worlds_for', lambda *args: [state]*3)
    monkeypatch.setattr(audit.base, 'serialize_state_snapshot', lambda s: {})
    monkeypatch.setattr(audit, 'prepare_branches', lambda *args: [
        {'world': w, 'root': ('a',), 'after': state, 'information_set': ('shared',),
         'replies': [SimpleNamespace(action='reply')]} for w in range(3)])
    monkeypatch.setattr(audit.base, 'canonical_successor_action_key', lambda *args: ('reply',))
    monkeypatch.setattr(audit.base, 'observe_rollout', lambda *args: {
        'value': .1, 'wdl': 1., 'resolved': True, 'loop_adjudicated': False,
        'fallback_triggers': 0, 'hard_cap_reached': False, 'termination_reason': 'completed'})
    result = audit.evaluate_block(sample, 'test', 0, None)
    assert result['telemetry']['continuations'] == result['telemetry']['expected_continuations'] == 12
    assert result['telemetry']['fewer_than_four_reply_branches'] == 3
    assert result['gains']['margin'] == 0.
    state.terminal = True
    monkeypatch.setattr(audit.base, 'terminal_record', lambda *args: {'wdl': 1., 'final_scores': [16, 10]})
    terminal = audit.evaluate_block(sample, 'test', 0, None)
    assert terminal['telemetry']['terminal_roots'] == 3
    assert terminal['telemetry'].get('continuations', 0) == 0
    assert terminal['values']["('a',)"]['margin'] == pytest.approx(.3)
