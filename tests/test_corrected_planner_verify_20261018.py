"""Independent verifier tamper fixtures; no checkpoint loading or game study."""
from copy import deepcopy
import gzip
import hashlib
import json
import random

import pytest

from tools.corrected_planner_qualification_20261018 import verify as v
from tools.external_strength_screen_20261018 import rules
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.encoding.public_belief import canonical_public_belief_key, canonical_successor_action_key


def observer(state, game=1):
    value = rules.observation_payload(state)
    pair = game // 2
    assignment = sorted(range(201), key=lambda p: v.digest(['proposal-breadth-arena-128-fit-v1', 'main', p]))
    return {**value, 'schema': 'corrected_planner_observer_v1', 'context': {
        'fit': v.FITS[assignment.index(pair) % 3], 'stream': v.digest(['proposal-breadth-arena-128-stream-v1', 'main', pair]),
        'identity': ['main', game, state.turn_index]}}


def final_pass():
    state = rules.CorrectedEnv(seed=301).initial_state()
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    for tier in (1, 2, 3):
        for card in state.visible_tier_cards[tier] + state.hidden_tier_decks[tier]:
            p = state.players[0]; p.purchased_cards.append(card); p.bonuses[card.bonus_color] += 1; p.score += card.points
        state.visible_tier_cards[tier], state.hidden_tier_decks[tier], state.deck_counts[tier] = [], [], 0
    for seat, counts in enumerate(({'white': 4, 'blue': 4, 'green': 2}, {'green': 2, 'red': 4, 'black': 4})):
        for color, count in counts.items():
            state.players[seat].tokens[color] = count; state.bank_tokens[color] -= count
    return state


def trace_for(world, actions, scorer):
    current = deepcopy(world); steps = []; hashes = hashlib.sha256()
    for index, action in enumerate(actions):
        state = rules.deserialize_snapshot(current); legal = rules.legal_actions(state)
        values = [rules.serialize_action(a) for a in legal]
        scores = None if index == 0 else list(map(float, scorer(state, legal)))
        steps.append({'state_hash': v.digest(current['state']), 'turn': state.turn_index, 'actor': state.current_player,
            'action': action, 'selected_index': values.index(action), 'legal_count': len(legal), 'forced': index == 0,
            'scores_sha256': None if scores is None else v.digest(scores)})
        hashes.update(v.digest(action).encode()); current = v.ledger.transition(current, action)
    return {'initial_state_sha256': v.digest(world['state']), 'steps': steps,
            'reason': 'completed', 'winner': current['state']['winner'], 'turns': len(steps),
            'trace_sha256': hashes.hexdigest(), 'final_state_sha256': v.digest(current['state'])}


def test_world_reconstruction_and_belief_ignore_actual_hidden_order():
    state = rules.CorrectedEnv(seed=302).initial_state()
    other = state.players[1]
    other.reserved_cards = [state.hidden_tier_decks[1].pop(), state.hidden_tier_decks[2].pop()]
    other.reserved_card_sources = ['deck', 'visible']
    state.deck_counts = {t: len(cards) for t, cards in state.hidden_tier_decks.items()}
    item = observer(state)
    assert v.belief_key(item) == canonical_public_belief_key(state, state.current_player)
    for seed in ('a', 'b'):
        expected = determinize_for_observer(state, state.current_player, random.Random(seed))
        assert v.world_from_observer(item, seed) == rules.serialize_snapshot(expected)
    altered = deepcopy(item); altered['observation']['players'][1]['reserves'][0]['card_id'] = other.reserved_cards[0].card_id
    with pytest.raises(ValueError, match='private identity'):
        v.world_from_observer(altered, 'a')


def test_group_keys_and_first_maximum_ties_are_independent():
    state = rules.CorrectedEnv(seed=303).initial_state(); snapshot = rules.serialize_snapshot(state)
    objects = rules.legal_actions(state); legal = [rules.serialize_action(a) for a in objects]
    assert [v.action_key(snapshot, a) for a in legal] == [repr(canonical_successor_action_key(state, a)) for a in objects]
    got = v.derive_proposal(observer(state), snapshot, legal, [0.] * len(legal), [0.] * len(legal), {'sha256': 'a' * 64})
    assert got['reference']['actions'][:2] == legal[:2]
    allowed = [k for k in got['group_keys'] if k not in got['reference']['keys'][:3]]
    assert got['candidates']['keys'][-1] == allowed[0]
    worlds = [{'outcomes': [{'wdl': 0}, {'wdl': 0}, {'wdl': -1}]} for _ in range(128)]
    assert v.select_outcomes(worlds, 3) == (0, [0., 0., -1.])
    worlds[0]['outcomes'][0]['wdl'] = True
    with pytest.raises(ValueError): v.select_outcomes(worlds, 3)


def two_step_terminal():
    state = rules.CorrectedEnv(seed=304).initial_state()
    cards = rules.corrected_cards_by_id()
    fives = [c for c in cards.values() if c.points == 5][:2]
    four = next(c for c in cards.values() if c.points == 4)
    owned = {c.card_id for c in fives + [four]} | {c.card_id for c in cards.values() if c.points == 0}
    p = state.players[0]
    for tier in (1, 2, 3):
        for group in (state.visible_tier_cards, state.hidden_tier_decks):
            keep = []
            for card in group[tier]:
                if card.card_id in owned:
                    p.purchased_cards.append(card); p.bonuses[card.bonus_color] += 1; p.score += card.points
                else: keep.append(card)
            group[tier] = keep
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    env = rules.CorrectedEnv()
    forced = next(a for a in env.legal_actions(state) if a.action_type.name == 'BUY_VISIBLE')
    after = env.step(state, forced)
    selected = env.legal_actions(after)[0]
    return rules.serialize_snapshot(state), [rules.serialize_action(a) for a in (forced, selected)]


def test_raw_policy_choices_and_digest_are_reexecuted_in_ledger_replay():
    world, actions = two_step_terminal()
    scorer = lambda state, legal: [0.] * len(legal)
    row = trace_for(world, actions, scorer)
    outcome, count = v.audit_trajectory(world, actions[0], row, scorer=scorer)
    assert count == 1 and outcome['turns'] == 2 and outcome['winner'] == 0
    row['steps'][1]['scores_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='score digest'):
        v.audit_trajectory(world, actions[0], row, scorer=scorer)


@pytest.mark.parametrize('damage', ['state', 'actor', 'count', 'selected', 'forced', 'trace_hash', 'winner', 'final', 'extra'])
def test_trajectory_rejects_corrupt_identity_transition_or_ending(damage):
    state = final_pass(); world = rules.serialize_snapshot(state)
    action = rules.serialize_action(rules.legal_actions(state)[0])
    row = trace_for(world, [action], None)
    assert v.audit_trajectory(world, action, row, scorer=None)[0]['winner'] == 0
    if damage == 'state': row['steps'][0]['state_hash'] = '0' * 64
    elif damage == 'actor': row['steps'][0]['actor'] = False
    elif damage == 'count': row['steps'][0]['legal_count'] += 1
    elif damage == 'selected': row['steps'][0]['selected_index'] = False
    elif damage == 'forced': row['steps'][0]['forced'] = False
    elif damage == 'trace_hash': row['trace_sha256'] = '0' * 64
    elif damage == 'winner': row['winner'] = 1
    elif damage == 'final': row['final_state_sha256'] = '0' * 64
    else: row['steps'].append(deepcopy(row['steps'][0]))
    with pytest.raises(ValueError): v.audit_trajectory(world, action, row, scorer=None)


def full_fixture(tmp_path, monkeypatch):
    state = final_pass(); item = observer(state); world = v.world_from_observer(item)
    legal = [rules.serialize_action(a) for a in rules.legal_actions(state)]
    proposal = v.derive_proposal(item, world, legal, [0.], [0.], {'sha256': 'a' * 64})
    monkeypatch.setattr(v, 'root_scores', lambda *_: ([0.], [0.], {'sha256': 'a' * 64}))
    outcome = trace_for(world, legal, None)
    rows = [{**deepcopy(outcome), 'world_index': i, 'candidate_index': 0, 'seed': seed}
            for i, seed in enumerate(proposal['schedule']['world_seeds'])]
    worlds = [{'seed': seed, 'state_sha256': v.digest(world['state']), 'outcomes': [
        {**{k: outcome[k] for k in v.OUTCOME_FIELDS}, 'wdl': -1.}]} for seed in proposal['schedule']['world_seeds']]
    record = {'root_identity': item['context']['identity'], 'planner': 'reference', 'privacy': {'own': 0, 'opponent': 0},
        'schedule': proposal['schedule'], 'candidates': proposal['candidates'], 'proposal': proposal, 'scores': [0.], 'worlds': worlds,
        'profile': v.LOCAL_PROFILE, 'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256, 'choice': 0, 'means': [-1.]}
    result = {'decision': {'action': legal[0], 'record': record}, 'trace': {}, 'ruleset_id': rules.RULESET_ID,
        'catalogue_sha256': rules.CATALOGUE_SHA256, 'profile_id': v.PROFILE_ID, 'observer_sha256': v.digest(item)}
    path = tmp_path / 'traces.gz'
    def write_trace():
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for row in rows: stream.write(json.dumps(row, sort_keys=True) + '\n')
        result['trace'] = {'sha256': v.file_hash(path), 'bytes': path.stat().st_size, 'records': 128}
    write_trace()
    return item, result, path, rows, write_trace


def test_exhaustive_full_decision_has_128_verified_continuations(tmp_path, monkeypatch):
    item, result, path, _, _ = full_fixture(tmp_path, monkeypatch)
    got = v.audit_decision(item, result, path)
    assert got['audited_worlds'] == got['audited_continuations'] == got['audited_transitions'] == 128
    assert got['raw035_choices_reexecuted'] == 0 and got['all_continuations_replayed'] is True


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'extra', 'wdl', 'mean', 'seed', 'proposal', 'profile', 'observer'])
def test_complete_decision_rejects_tampering(tmp_path, monkeypatch, damage):
    item, result, path, rows, write = full_fixture(tmp_path, monkeypatch)
    record = result['decision']['record']
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows[1] = deepcopy(rows[0])
    elif damage == 'extra': rows.append(deepcopy(rows[-1]))
    elif damage == 'wdl': record['worlds'][0]['outcomes'][0]['wdl'] = 1.
    elif damage == 'mean': record['means'] = [1.]
    elif damage == 'seed': record['worlds'][0]['seed'] = 'different'
    elif damage == 'proposal': record['proposal']['learned_logits'] = [1.]
    elif damage == 'profile': result['profile_id'] = 'historical'
    else: result['observer_sha256'] = '0' * 64
    write()
    with pytest.raises(ValueError): v.audit_decision(item, result, path)


def _recommendation_rows(maximum=95.):
    from tools import analyze_corrected_planner_qualification_20261018 as final
    entries = [{'fit': fit, 'actor': seat, 'phase': phase} for fit in final.FITS for seat in (0, 1) for phase in (0, 1, 2)]
    entries += [dict(entries[i]) for i in range(5)]
    rows = [{'case': case, 'fit': case['fit'], 'correct': case['allowance_seconds'] == 120,
             'clock': {'status': 'ok' if case['allowance_seconds'] == 120 else 'timeout',
                       'receipt_elapsed_seconds': maximum if case['allowance_seconds'] == 120 else None}}
            for case in final.cases_for(entries)]
    return final, rows


@pytest.mark.parametrize('maximum,margin', [(95., 'at_least_20_percent'), (96., 'at_least_20_percent'), (96.001, 'thin_margin'), (120., 'thin_margin')])
def test_budget_recommendation_requires_all_research_rows_and_preserves_margin_rule(maximum, margin):
    final, rows = _recommendation_rows(maximum)
    result = final.budget_recommendation(rows)
    assert result['recommended_research_seconds'] == 120 and result['observed_margin'] == margin
    assert result['one_second_qualified'] is False and result['other_budgets_inferred'] is False


def test_one_failed_repeat_blocks_research_budget_without_imputing_timeout_latency():
    final, rows = _recommendation_rows()
    rows[-1]['correct'] = False; rows[-1]['clock'] = {'status': 'timeout', 'receipt_elapsed_seconds': None}
    result = final.budget_recommendation(rows)
    assert result['recommended_research_seconds'] is None and result['research_completed_and_verified'] == 25
    assert result['observed_margin'] == 'unqualified' and result['maximum_accepted_research_receipt_seconds'] == 95.
    with pytest.raises(ValueError): final.budget_recommendation(rows[:-1])


@pytest.mark.parametrize('research_passed', [False, True])
def test_completed_negative_qualification_is_a_valid_audit_outcome(research_passed):
    final, rows = _recommendation_rows()
    if not research_passed:
        rows[-1]['correct'] = False
        rows[-1]['clock'] = {'status': 'timeout', 'receipt_elapsed_seconds': None}
    result = {'completed_without_referee_failure': True, 'complete_schedule': True, 'failure': None,
              'attempted': 49, 'prescribed': 49, 'strength_games': 0,
              'scientifically_accepted': research_passed, 'research_qualified_120s': research_passed}
    recommendation = final.budget_recommendation(rows)
    final.check_completion_flags(result)
    final.check_completion_flags(result, recommendation)
    result['scientifically_accepted'] = not research_passed
    with pytest.raises(ValueError, match='Reported research qualification'):
        final.check_completion_flags(result, recommendation)
    result['completed_without_referee_failure'] = False
    with pytest.raises(ValueError, match='Incomplete or failed'):
        final.check_completion_flags(result)
