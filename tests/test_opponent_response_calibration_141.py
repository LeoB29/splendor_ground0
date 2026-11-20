from copy import deepcopy
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from splendor_ai.eval import run_opponent_response_calibration as audit


def load_verifier():
    spec = importlib.util.spec_from_file_location('verify141', audit.VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_panel():
    return {'selected': [{'sample_index': j*50+i, 'source_seed': j*50+i, 'turn_index': i,
                          'actor_seat': int(cell[4]), 'phase': cell.split('|')[1], 'selection_cell': cell}
                         for j, cell in enumerate(audit.CELLS) for i in range(50)]}


def test_outcome_blind_balanced_disjoint_selection_and_probes():
    source = source_panel()
    excluded = {r['source_seed'] for r in source['selected'] if r['source_seed'] % 50 < 27}
    panel, probes = audit.select_states(source, excluded)
    assert len(panel) == 120 and len(probes) == 6
    assert len({r['source_seed'] for r in panel+probes}) == 126
    assert not excluded & {r['source_seed'] for r in panel+probes}
    assert all(sum(r['selection_cell'] == c for r in panel) == 20 for c in audit.CELLS)
    poisoned = {'selected': [{**r, 'payoff': float('nan'), 'winner': 999, 'expanded_action': 'poison'} for r in reversed(source['selected'])]}
    p, q = audit.select_states(poisoned, excluded)
    assert [r['sample_index'] for r in panel+probes] == [r['sample_index'] for r in p+q]
    with pytest.raises(ValueError):
        audit.select_states(source, {r['source_seed'] for r in source['selected'][:50]})


def test_seed_namespaces_unique_and_disjoint_from_previous_runs():
    main = {audit.continuation_seed(s, b, w, r) for s in range(300) for b in range(16) for w in range(3) for r in range(4)}
    probe = {audit.continuation_seed(s, 0, w, r, True) for s in range(300) for w in range(3) for r in range(4)}
    assert len(main) == 57_600 and len(probe) == 3600 and not main & probe
    assert min(main) > 1_429_000_000_000_000_000 and max(probe) < 1_440_000_000_000_000_000
    for args in ((300, 0, 0, 0), (0, 16, 0, 0), (0, 0, 3, 0), (0, 0, 0, 4)):
        with pytest.raises(ValueError):
            audit.continuation_seed(*args)


def sample_for(state=None):
    state = state or audit.base.SplendorEnv(seed=141).initial_state()
    keys = list(dict.fromkeys(repr(audit.base.canonical_successor_action_key(state, a)) for a in audit.base.SplendorEnv().legal_actions(state)))
    return {'sample_index': 3, 'source_seed': 860004, 'schedule_index': 0,
            'turn_index': state.turn_index, 'actor_seat': state.current_player, 'phase': 'late', 'selection_cell': 'seat0|late',
            'state_snapshot': json.loads(json.dumps(audit.base.serialize_state_snapshot(state))),
            'champion_action': keys[0], 'expanded_action': keys[-1]}


class OrderedScorer:
    def score_actions(self, state, actions):
        return tuple(SimpleNamespace(action=a, score=float(len(actions)-i), action_index=i) for i, a in enumerate(actions))


def test_structures_preserve_information_inventory_and_reply_work():
    sample = sample_for()
    record = audit.verify_sample(sample, OrderedScorer())
    assert len(record['blocks']) == 16
    assert all(b['expected_continuations'] == {'sampled_min': 96, 'policy_reply': 24} for b in record['blocks'])
    hashes = [h for b in record['blocks'] for h in b['world_hashes']]
    assert len(set(hashes)) == 48
    probe = [audit.digest(audit.base.serialize_state_snapshot(w)) for w in audit.worlds_for(sample, 0, True)]
    assert not set(hashes) & set(probe)
    sample['expanded_action'] = sample['champion_action']
    record = audit.structure_block(sample, 0, OrderedScorer())
    assert record['expected_continuations'] == {'sampled_min': 48, 'policy_reply': 12}


def test_sampled_min_matches_139_with_matched_world_and_seed_schedule(monkeypatch):
    sample, scorer = sample_for(), OrderedScorer()
    def fake(state, action, player, seed, scorer):
        # Legal transitions determine a reproducible payoff without playing tails.
        text = repr(audit.base.canonical_successor_action_key(state, action))
        score = int(audit.digest([text, seed])[:5], 16) % 19
        return {'value': score/20., 'wdl': 1., 'winner': player,
                'final_scores': [score if i == player else 0 for i in range(2)],
                'final_card_counts': [0, 0], 'resolved': True, 'terminal': True,
                'loop_adjudicated': False, 'hard_cap_reached': False, 'fallback_triggers': 0,
                'termination_reason': 'completed', 'immediate_reply': text}
    monkeypatch.setattr(audit, 'observe', fake)
    monkeypatch.setattr(audit.base, 'observe_rollout', fake)
    monkeypatch.setattr(audit.prior139, 'worlds_for', lambda s, c, b: audit.worlds_for(s, b))
    monkeypatch.setattr(audit.prior139, 'continuation_seed', lambda s, c, b, w, r: audit.continuation_seed(s, b, w, r))
    old = audit.prior139.evaluate_block(sample, 'test', 0, scorer)
    new = audit.evaluate_block(sample, 0, scorer)
    for root in old['values']:
        assert new['values']['sampled_min'][root]['margin'] == pytest.approx(old['values'][root]['margin'], abs=1e-12)
        assert new['values']['sampled_min'][root]['wdl'] == pytest.approx(old['values'][root]['wdl_under_margin'], abs=1e-12)
    assert new['telemetry']['sampled_min']['continuations'] == 96
    assert new['telemetry']['policy_reply']['continuations'] == 24
    sample['expanded_action'] = sample['champion_action']
    same = audit.evaluate_block(sample, 1, scorer)
    assert same['gains'] == {c: {'margin': 0., 'wdl': 0.} for c in audit.CONDITIONS}
    assert same['telemetry']['sampled_min']['continuations'] == 48


@pytest.mark.parametrize(('ci', 'expected'), [
    ([.03, .05], 'material_response_rule_effect_supported'),
    ([-.05, -.03], 'material_response_rule_effect_supported'),
    ([.001, .04], 'response_rule_difference_detected_materiality_uncertain'),
    ([-.02, .02], 'response_rule_difference_bounded_below_material_size'),
    ([-.03, .04], 'response_rule_contrast_inconclusive'),
])
def test_classification_uses_interval_not_point_estimate_or_control_gate(ci, expected):
    summary = {'policy_minus_min': {'ci95': ci, 'mean': 999}, 'policy_reply_gain': {'ci95': [-1, 1], 'mean': 0}}
    assert audit.classify(summary) == expected
    assert audit.classify(summary, False) == 'opponent_response_calibration_integrity_or_dose_failed'


def test_bootstrap_uses_source_games_with_equal_cell_weights():
    rows = [{'source_seed': i, 'selection_cell': c, 'gains': {'sampled_min': {'margin': j/100}, 'policy_reply': {'margin': j/100+.02}}}
            for j, c in enumerate(audit.CELLS) for i in range(20)]
    result = audit.paired_summary(rows, samples=200)
    assert result['policy_minus_min']['mean'] == pytest.approx(.02)
    assert result['policy_minus_min']['ci95'] == pytest.approx([.02, .02])
    assert result == audit.paired_summary(list(reversed(rows)), samples=200)


def synthetic_block():
    sample = {'sample_index': 2, 'actor_seat': 0, 'turn_index': 5, 'champion_action': 'a', 'expanded_action': 'b'}
    block = {'block': 0, 'world_hashes': ['x', 'y', 'z'], 'leaves': [], 'policy_returns': [],
             'values': {'sampled_min': {}, 'policy_reply': {}}, 'telemetry': {}, 'gains': {}}
    structure = {'block': 0, 'world_hashes': block['world_hashes'], 'branches': [],
                 'expected_continuations': {'sampled_min': 48, 'policy_reply': 24}}
    # Shared information set: min(mean) differs from mean(min) for root b.
    table = {'a': [[4, 4], [4, 4], [4, 4]], 'b': [[0, 12], [12, 0], [6, 6]]}
    for root in ('a', 'b'):
        for w in range(3):
            structure['branches'].append({'world': w, 'root': root, 'info': root, 'replies': ['r0', 'r1'], 'terminal_root': False})
            for arm, field in zip(audit.CONDITIONS, ('leaves', 'policy_returns')):
                for reply in (range(2) if arm == 'sampled_min' else (None,)):
                    score = table[root][w][reply] if reply is not None else 8
                    returns = [{'final_scores': [score, 0], 'final_card_counts': [1, 2], 'winner': 0,
                                'wdl': 1., 'value': score/20., 'seed': audit.continuation_seed(2, 0, w, rep),
                                'terminal': True, 'resolved': True, 'hard_cap_reached': False, 'loop_adjudicated': False,
                                'fallback_triggers': 0, 'final_turn_index': 20, 'max_repetition_visits': 1,
                                'max_no_progress': 2, 'turns': 13 if arm == 'sampled_min' else 14,
                                'action_sequence_sha256': 'x'*64, 'immediate_reply': 'r0'} for rep in range(4)]
                    block[field].append({'world': w, 'root': root, 'information_set_sha256': root,
                                         'reply': f'r{reply}' if reply is not None else None,
                                         'returns': returns, 'margin': score/20., 'wdl': 1.})
        block['values']['sampled_min'][root] = {'margin': .2 if root == 'a' else .3, 'wdl': 1.}
        block['values']['policy_reply'][root] = {'margin': .4, 'wdl': 1.}
    for arm, n in (('sampled_min', 48), ('policy_reply', 24)):
        block['telemetry'][arm] = {'continuations': n, 'natural_continuations': n, 'resolved_continuations': n,
                                   'fewer_than_four_reply_branches': 6}
        block['gains'][arm] = {'margin': .1 if arm == 'sampled_min' else 0., 'wdl': 0.}
    return block, sample, structure


def test_independent_reconstruction_groups_before_min_and_detects_corruption():
    verify = load_verifier()
    block, sample, structure = synthetic_block()
    gain, _ = verify.reconstruct_block(block, sample, structure)
    assert gain['sampled_min']['margin'] == pytest.approx(.1)
    changed = deepcopy(block)
    changed['leaves'][0]['returns'][0]['final_scores'][0] += 1
    with pytest.raises(AssertionError):
        verify.reconstruct_block(changed, sample, structure)
    changed = deepcopy(block)
    changed['leaves'].append(deepcopy(changed['leaves'][0]))
    with pytest.raises(AssertionError):
        verify.reconstruct_block(changed, sample, structure)
    changed = deepcopy(block)
    changed['world_hashes'][0] = 'tampered'
    with pytest.raises(AssertionError):
        verify.reconstruct_block(changed, sample, structure)


def test_saved_resume_rejects_incomplete_or_foreign_records():
    with pytest.raises(ValueError, match='identity'):
        audit.validate_saved({'identity': 'old', 'row': {'sample_sha256': 'x'}}, {}, 'new', {})
    sample = {'champion_action': 'a', 'expanded_action': 'b'}
    with pytest.raises(ValueError, match='blocks'):
        audit.validate_saved({'identity': 'x', 'row': {**sample, 'sample_sha256': audit.digest(sample), 'blocks': []}}, sample, 'x', {})


def test_observe_rejects_unresolved_or_adjudicated_results(monkeypatch):
    sample = sample_for()
    state = audit.base.deserialize_state_snapshot(sample['state_snapshot'])
    action = audit.base.SplendorEnv().legal_actions(state)[0]
    def bad(env, scorer, state, action, *args):
        env.step(state, action)
        return SimpleNamespace(terminal=False, resolved=False, hard_cap_reached=True)
    monkeypatch.setattr(audit.base, 'rollout_score_margin_loop_safe', bad)
    with pytest.raises(ValueError, match='non-natural'):
        audit.observe(state, action, 0, 141, OrderedScorer())


def test_terminal_roots_are_exact_with_no_scorer_or_rollout_calls():
    state = audit.base.SplendorEnv(seed=141).initial_state()
    state.current_player = 1
    state.pending_round_end = True
    state.players[0].score = 15
    sample = sample_for(state)
    class ForbiddenScorer:
        def score_actions(self, *args):
            raise AssertionError('terminal root must not call a scorer')
    block = audit.evaluate_block(sample, 0, ForbiddenScorer())
    structure = audit.structure_block(sample, 0, ForbiddenScorer())
    assert structure['expected_continuations'] == {'sampled_min': 0, 'policy_reply': 0}
    assert all(block['telemetry'][c]['terminal_roots'] == 6 for c in audit.CONDITIONS)
    verify = load_verifier()
    gains, _ = verify.reconstruct_block(block, sample, structure)
    assert gains == {c: {'margin': 0., 'wdl': 0.} for c in audit.CONDITIONS}


def test_tracking_observe_and_140_agree_from_same_policy_rng():
    state = audit.base.SplendorEnv(seed=141).initial_state()
    state.pending_round_end = True
    state.players[0].score = 15
    scorer = OrderedScorer()
    action = audit.base.SplendorEnv().legal_actions(state)[0]
    root = audit.base.canonical_successor_action_key(state, action)
    result = audit.observe(state, action, 0, 141, scorer, retain_actions=True)
    bot = audit.prior140.ScorerPolicy(scorer, 141, True)
    end, trace, reason = audit.prior140.play_from_state(state, root, (bot, bot))
    assert reason == 'completed'
    assert result['actions'] == [t['action'] for t in trace] and len(trace) == 2
    assert result['final_scores'] == [p.score for p in end.players]
