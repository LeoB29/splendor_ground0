from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from splendor_ai.eval import match
from splendor_ai.eval import run_continuation_policy_crossover as audit
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    gameplay_belief_root_index, gameplay_continuation_seed,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.state import Card


def verifier():
    spec = importlib.util.spec_from_file_location('verify140', audit.VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FirstBot:
    def choose_action(self, env, state, legal_actions):
        return legal_actions[0]


class ForbiddenBot:
    def choose_action(self, *args):
        raise AssertionError('policy called after terminal forced root')


def sample_for(state=None):
    state = state or audit.base.SplendorEnv(seed=140).initial_state()
    actions = audit.base.SplendorEnv().legal_actions(state)
    keys = [repr(audit.base.canonical_successor_action_key(state, a)) for a in actions]
    return {'sample_index': 12, 'source_seed': 860012, 'turn_index': state.turn_index,
            'actor_seat': state.current_player, 'phase': 'late', 'selection_cell': f'seat{state.current_player}|late',
            'maximum_schedule_index': 0, 'state_snapshot': audit.base.serialize_state_snapshot(state),
            'champion_action': keys[0], 'expanded_action': keys[-1]}


def test_selection_ignores_outcomes_is_nested_balanced_and_disjoint():
    rows = [{'sample_index': j*50+i, 'source_seed': j*50+i, 'turn_index': i,
             'actor_seat': int(cell[4]), 'phase': cell.split('|')[1], 'selection_cell': cell}
            for j, cell in enumerate(audit.CELLS) for i in range(50)]
    maximum, probes, excluded = audit.select_candidates({'selected': rows})
    assert len(maximum) == 60 and len(probes) == 6 and len(excluded) == 120
    assert len({s['source_seed'] for s in maximum+probes}) == 66
    assert not set(excluded) & {s['source_seed'] for s in maximum+probes}
    poisoned = {'selected': [{**r, 'gain': float('nan'), 'winner': 888, 'expanded_action': 'poison'} for r in reversed(rows)]}
    changed, new_probes, _ = audit.select_candidates(poisoned)
    assert [s['source_seed'] for s in maximum+probes] == [s['source_seed'] for s in changed+new_probes]
    previous = set()
    for n in reversed(audit.SIZES):
        selected = audit.select_prefix(maximum, n)
        assert len(selected) == n
        assert all(sum(s['selection_cell'] == c for s in selected) == n//6 for c in audit.CELLS)
        ids = {s['source_seed'] for s in selected}
        assert previous <= ids
        previous = ids


def test_original_actions_bind_after_selection_and_reject_changed_source():
    sample = {**sample_for(), 'champion_root_keys': ['a', 'b'], 'expanded_root_keys': ['c', 'd']}
    old = {**sample, 'champion_selected_index': 1, 'expanded_selected_index': 0}
    result = {'rows': [old]}
    bound = audit.bind_samples([sample], result)[0]
    assert (bound['champion_action'], bound['expanded_action']) == ('b', 'c')
    old['source_seed'] += 1
    with pytest.raises(ValueError, match='binding'):
        audit.bind_samples([sample], result)


def test_exact_084_contract_and_seed_namespaces():
    cfg = audit.resolver_config()
    primary = {audit.policy_slot(s, w) for s in range(300) for w in range(3)}
    probes = {audit.policy_slot(s, 0, True) for s in range(300)}
    assert len(primary) == 900 and len(probes) == 300 and not primary & probes
    previous_high = 1_399_000_000_000_000_000
    for slot in (min(primary), max(primary), min(probes), max(probes)):
        low = gameplay_continuation_seed(cfg, cfg.smoke_seed+slot, 0, 0, 0, 0)
        high = gameplay_continuation_seed(cfg, cfg.smoke_seed+slot, 1, 399, 2, 3)
        assert previous_high < low <= high < cfg.continuation_seed_start+(slot+1)*1_000_000_000
        assert gameplay_belief_root_index(cfg, cfg.smoke_seed+slot, 1, 399) < cfg.belief_root_index_start+(slot+1)*1000
    bots = audit.make_bots('resolver_084', 7, object())
    assert all(type(b) is audit.FrozenPublicBeliefResolverBot for b in bots)
    assert all(b._continuation_workers == 1 for b in bots)
    assert bots[0] is not bots[1] and all(b.telemetry['decisions'] == 0 for b in bots)
    for b in bots:
        b.close()
    with pytest.raises(ValueError):
        audit.policy_slot(300, 0)


def test_stochastic_control_uses_unchanged_sampler_and_shared_paired_rng():
    scores = tuple(SimpleNamespace(action=i, score=-i*.1) for i in range(5))
    scorer = SimpleNamespace(score_actions=lambda *args: scores)
    bots = audit.make_bots('stochastic_035', 4, scorer)
    paired = audit.make_bots('stochastic_035', 4, scorer)
    assert bots[0] is bots[1]
    native_rng = random.Random(1_410_000_000_000_000_000+4*1_000_000)
    sequence = [bots[i % 2].choose_action(None, None, None) for i in range(200)]
    assert sequence == [audit._sample_scored_action(scores, native_rng, .1, .5, 3) for _ in range(200)]
    assert sequence == [paired[0].choose_action(None, None, None) for _ in range(200)]
    assert len(set(sequence)) > 1
    assert audit.make_bots('greedy_035', 4, scorer)[0].choose_action(None, None, None) == 0


def test_worlds_preserve_information_and_ignore_hidden_deck_order():
    sample = sample_for()
    sample['state_snapshot'] = json.loads(json.dumps(sample['state_snapshot']))
    structure = audit.verify_structure(sample)
    assert len(set(structure['world_hashes'])) == 3
    assert set(structure['world_hashes']).isdisjoint(audit.verify_structure(sample, True)['world_hashes'])
    earlier = audit.base.worlds_for(sample, 'fresh_worlds', 0)
    assert set(structure['world_hashes']).isdisjoint(audit.digest(audit.base.serialize_state_snapshot(s)) for s in earlier)


def deadlock_state():
    state = audit.base.SplendorEnv(seed=140).initial_state()
    state.bank_tokens.update({c: 0 for c in ALL_TOKEN_TYPES})
    for seat, player in enumerate(state.players):
        player.reserved_cards = [Card(card_id=f'test-{seat}-{i}', tier=1, bonus_color='white', points=0,
                                      cost={'blue': 10}) for i in range(3)]
    state.players[0].score = 2
    return state


@pytest.mark.parametrize('deadlock', [False, True])
def test_driver_matches_native_terminal_and_forced_pass_rules(monkeypatch, deadlock):
    state = deadlock_state() if deadlock else audit.base.SplendorEnv(seed=140).initial_state()
    if not deadlock:
        state.pending_round_end = True
        state.current_player = 1
        state.players[0].score = 15
    env = audit.base.SplendorEnv()
    key = audit.base.canonical_successor_action_key(state, env.legal_actions(state)[0])
    end, trace, reason = audit.play_from_state(state.copy_shallow(), key,
                         (FirstBot(), FirstBot()) if deadlock else (ForbiddenBot(), ForbiddenBot()))
    monkeypatch.setattr(match.SplendorEnv, 'initial_state', lambda self: state.copy_shallow())
    native = match.play_game(FirstBot(), FirstBot(), repetition_limit=4, no_progress_limit=60)
    assert native.termination_reason == reason and native.turns == end.turn_index
    assert native.final_scores == tuple(p.score for p in end.players)
    assert len(trace) == (2 if deadlock else 1) and trace[0]['forced']


@pytest.mark.parametrize('failure', ['max_turns', 'repetition_cutoff', 'no_progress_cutoff', 'stalled', 'illegal_continuation_action'])
def test_cutoffs_and_illegal_moves_are_failures_never_outcomes(monkeypatch, failure):
    state = audit.base.SplendorEnv(seed=140).initial_state()
    env = audit.base.SplendorEnv()
    action = env.legal_actions(state)[0]
    key = audit.base.canonical_successor_action_key(state, action)
    if failure == 'max_turns':
        state.turn_index = 400
    def step(s, a):
        after = s.copy_shallow()
        after.turn_index += 1
        after.current_player = 1-s.current_player
        return after
    fake = SimpleNamespace(legal_actions=lambda s: [] if failure == 'stalled' else [action], step=step)
    monkeypatch.setattr(audit, 'state_signature', lambda s: ('repeat',) if failure == 'repetition_cutoff' else (s.turn_index,))
    monkeypatch.setattr(audit, 'is_progress_transition', lambda *args: failure != 'no_progress_cutoff')
    bot = SimpleNamespace(choose_action=lambda *args: None) if failure == 'illegal_continuation_action' else FirstBot()
    with pytest.raises(audit.ContinuationIntegrityError, match=failure):
        audit.play_from_state(state, key, (bot, bot), env=fake)


def terminal_record(monkeypatch):
    state = audit.base.SplendorEnv(seed=140).initial_state()
    state.current_player = 1
    state.pending_round_end = True
    state.players[0].score = 15
    sample = sample_for(state)
    monkeypatch.setattr(audit, 'worlds_for', lambda *args: [state.copy_shallow()]*3)
    row = audit.evaluate_trajectory(sample, 0, 'resolver_084', sample['champion_action'])
    return sample, row


def test_saved_trace_replays_independently_and_tampering_is_rejected(monkeypatch):
    sample, row = terminal_record(monkeypatch)
    check = verifier()
    assert check.replay(row, 1)['wdl'] == -1
    identity = {'fixed': True}
    structure = {'world_hashes': [row['world_sha256']]*3}
    payload = {'identity': identity, 'row': row}
    audit.validate_saved(payload, identity, sample, 0, 'resolver_084', sample['champion_action'], structure)
    for field, value in [('margin', .8), ('world_sha256', 'changed'), ('condition', 'greedy_035')]:
        altered = deepcopy(payload)
        altered['row'][field] = value
        with pytest.raises(ValueError):
            audit.validate_saved(altered, identity, sample, 0, 'resolver_084', sample['champion_action'], structure)
    altered = deepcopy(row)
    altered['trace'][0]['after_sha256'] = 'tampered'
    with pytest.raises(AssertionError):
        check.replay(altered, 1)
    altered = deepcopy(payload)
    altered['row']['resolver_telemetry'][0]['completed_searches'] = 1
    with pytest.raises(ValueError, match='telemetry'):
        audit.validate_saved(altered, identity, sample, 0, 'resolver_084', sample['champion_action'], structure)


def test_probe_retains_no_outcomes_and_checks_from_zero_repeat(monkeypatch):
    sample = {'selection_cell': 'seat0|late', 'phase': 'late', 'sample_index': 1, 'champion_action': 'fixed'}
    calls = []
    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return {'elapsed_seconds': len(calls), 'cpu_seconds': len(calls), 'termination_reason': 'completed',
                'outer_turns': 2, 'resolver_telemetry': [], 'winner': 1, 'margin': -.5, 'trace': ['secret']}
    monkeypatch.setattr(audit, 'evaluate_trajectory', evaluate)
    result = audit.run_probe(sample)
    assert len(calls) == 4 and result['repeat']['identical'] and not result['outcomes_retained']
    assert all(word not in json.dumps(result) for word in ('winner', 'margin', 'secret', 'trace'))


def test_runtime_selection_obeys_frozen_boundary_and_never_uses_payoffs():
    probes = [{'phase': p, 'timings_and_integrity': {c: {'elapsed_seconds': 30.} for c in audit.CONDITIONS}}
              for p in ('early', 'mid', 'late') for seat in (0, 1)]
    assert audit.choose_panel_size(probes)[0] == 60
    assert audit.choose_panel_size(probes, ceiling=48_600)[0] == 60
    assert audit.choose_panel_size(probes, ceiling=48_599)[0] == 48
    assert audit.choose_panel_size(probes, ceiling=1)[0] is None


def test_bootstrap_paired_game_weighted_and_independent_reconstruction():
    rows = [{'source_seed': i, 'selection_cell': audit.CELLS[i//4],
             'gains': {c: {'margin': i/100+(0 if c == 'stochastic_035' else .01 if c == 'greedy_035' else -.03)}
                       for c in audit.CONDITIONS}} for i in range(24)]
    summary = audit.paired_summary(rows, samples=1000)
    assert summary == audit.paired_summary(rows[::-1], samples=1000)
    assert summary == verifier().reconstruct_summary(rows, 'margin', draws=1000)
    assert summary['strong_minus_stochastic']['ci95'] == pytest.approx([-.03, -.03])
    assert summary['stochastic_gain']['mean'] == pytest.approx(.115)


def test_identical_roots_share_trajectories_and_cannot_manufacture_gain():
    sample = sample_for()
    sample['expanded_action'] = sample['champion_action']
    rows = [{'world_index': w, 'condition': c, 'root_action': sample['champion_action'],
             'margin': w/10, 'wdl': 1., 'raw_score_margin': w*2}
            for w in range(3) for c in audit.CONDITIONS]
    summary = audit.summarize_state(sample, rows)
    assert all(v == 0 for gains in summary['gains'].values() for v in gains.values())
    with pytest.raises(ValueError, match='incomplete'):
        audit.summarize_state(sample, rows[:-1])
    with pytest.raises(ValueError, match='incomplete'):
        audit.summarize_state(sample, rows+rows[:1])


def test_conclusions_require_control_reproduction_and_material_contrast():
    summary = {'stochastic_gain': {'mean': .05, 'ci95': [.02, .08]},
               'strong_gain': {'mean': .0, 'ci95': [-.02, .02]},
               'strong_minus_stochastic': {'mean': -.05, 'ci95': [-.08, -.02]}}
    assert audit.classify(summary) == 'continuation_policy_mismatch_supported'
    assert audit.classify(summary, False) == 'continuation_policy_crossover_integrity_or_dose_failed'
    summary['stochastic_gain']['ci95'][0] = -.01
    assert audit.classify(summary) == 'stochastic_control_gain_uncertain'
    summary['stochastic_gain']['mean'] = -.01
    assert audit.classify(summary) == 'stochastic_control_gain_not_reproduced'
    summary['stochastic_gain'] = {'mean': .05, 'ci95': [.02, .08]}
    summary['strong_gain'] = {'mean': .05, 'ci95': [.02, .08]}
    summary['strong_minus_stochastic'] = {'mean': 0., 'ci95': [-.02, .02]}
    assert audit.classify(summary) == 'advantage_survives_stronger_continuations'
    summary['strong_minus_stochastic']['ci95'][0] = -.03
    assert audit.classify(summary) == 'continuation_policy_crossover_inconclusive'
