from __future__ import annotations

import pytest

from splendor_ai.bots.rolling_horizon_bot import (
    BranchingMutationRollingHorizonBot,
    RollingHorizonConfig,
    TransitionEvents,
    event_value,
    transition_events,
    transition_value,
)
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, Noble, RESERVE_SOURCE_DECK


class CanonicalPrior:
    def score_actions(self, state, legal_actions):
        ordered = sorted(
            legal_actions,
            key=lambda action: repr(canonical_successor_action_key(state, action)),
        )
        return {action: float(len(ordered) - index) for index, action in enumerate(ordered)}


def _config(mode: str = "event_value_hc") -> RollingHorizonConfig:
    return RollingHorizonConfig(
        actor_horizon=2,
        sequence_evaluations=8,
        determinizations=2,
        max_actor_candidates=4,
        seed=17,
        determinization_seed=29,
        evaluator_mode=mode,
    )


def test_transition_events_extract_take_and_reserve() -> None:
    env = SplendorEnv(seed=3)
    state = env.initial_state()
    take = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.TAKE_TOKENS and len(action.take_tokens) == 3
    )
    after_take = env.step(state, take)
    assert transition_events(state, after_take, 0) == TransitionEvents(tokens_received=3)

    state = env.initial_state()
    reserve = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.RESERVE_DECK
    )
    after_reserve = env.step(state, reserve)
    assert transition_events(state, after_reserve, 0) == TransitionEvents(
        tokens_received=1,
        blind_reserves=1,
    )


def test_transition_events_extract_card_and_noble_prestige_events() -> None:
    env = SplendorEnv(seed=4)
    state = env.initial_state()
    card = Card(card_id="free-point", tier=1, bonus_color="white", points=1, cost={})
    noble = Noble(noble_id="free-noble", points=3, requirement={"white": 1})
    state.visible_tier_cards = {1: [card], 2: [], 3: []}
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}
    state.nobles = [noble]
    buy = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.BUY_VISIBLE and action.noble_id == noble.noble_id
    )
    after = env.step(state, buy)

    assert transition_events(state, after, 0) == TransitionEvents(
        nobles_received=1,
        prestige_events=2,
    )


def test_event_value_and_transition_value_are_root_relative() -> None:
    events = TransitionEvents(
        tokens_received=3,
        blind_reserves=1,
        visible_reserves=1,
        nobles_received=1,
        prestige_events=2,
    )
    assert event_value(events) == pytest.approx(1.4)

    env = SplendorEnv(seed=5)
    state = env.initial_state()
    action = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.TAKE_TOKENS and len(action.take_tokens) == 3
    )
    after = env.step(state, action)
    assert transition_value(state, after, 0, "event_value_hc") == pytest.approx(0.6)
    assert transition_value(state, after, 1, "event_value_hc") == pytest.approx(-0.6)


def test_planner_returns_legal_deterministic_action_and_exact_budget() -> None:
    env = SplendorEnv(seed=6)
    state = env.initial_state()
    legal = env.legal_actions(state)
    bot_a = BranchingMutationRollingHorizonBot(CanonicalPrior(), _config())
    bot_b = BranchingMutationRollingHorizonBot(CanonicalPrior(), _config())

    action_a = bot_a.choose_action(env, state, legal)
    action_b = bot_b.choose_action(env, state, legal)

    assert action_a in legal
    assert action_b == action_a
    assert bot_a.last_stats is not None
    assert bot_a.last_stats.sequence_evaluations == 8
    assert bot_a.last_stats.genome_evaluations == 4
    assert bot_a.last_stats.simulated_actions <= 24
    assert bot_a.total_planning_decisions == 1


def test_evaluator_variants_share_random_schedule() -> None:
    env = SplendorEnv(seed=7)
    state = env.initial_state()
    event_bot = BranchingMutationRollingHorizonBot(CanonicalPrior(), _config("event_value_hc"))
    score_bot = BranchingMutationRollingHorizonBot(CanonicalPrior(), _config("score_delta"))

    event_bot.choose_action(env, state)
    score_bot.choose_action(env, state)

    assert event_bot.last_stats is not None
    assert score_bot.last_stats is not None
    assert event_bot.last_stats.random_schedule == score_bot.last_stats.random_schedule
    assert event_bot.last_stats.sequence_evaluations == score_bot.last_stats.sequence_evaluations


def test_planner_is_invariant_to_hidden_order_and_private_reserve_identity() -> None:
    env = SplendorEnv(seed=8)
    state_a = env.initial_state()
    state_b = state_a.copy_shallow()
    state_b.hidden_tier_decks[1].reverse()
    hidden = state_b.hidden_tier_decks[1].pop()
    original = state_b.players[1].reserved_cards
    if not original:
        state_b.players[1].reserved_cards.append(hidden)
        state_b.players[1].reserved_card_sources.append(RESERVE_SOURCE_DECK)
        replacement = state_a.hidden_tier_decks[1][-1]
        state_a.players[1].reserved_cards.append(replacement)
        state_a.players[1].reserved_card_sources.append(RESERVE_SOURCE_DECK)
        state_a.hidden_tier_decks[1] = state_a.hidden_tier_decks[1][:-1]
    state_b.deck_counts[1] = len(state_b.hidden_tier_decks[1])
    state_a.deck_counts[1] = len(state_a.hidden_tier_decks[1])

    bot_a = BranchingMutationRollingHorizonBot(CanonicalPrior(), _config())
    bot_b = BranchingMutationRollingHorizonBot(CanonicalPrior(), _config())
    action_a = bot_a.choose_action(env, state_a)
    action_b = bot_b.choose_action(env, state_b)

    assert action_a is not None and action_b is not None
    assert canonical_successor_action_key(state_a, action_a) == canonical_successor_action_key(state_b, action_b)
