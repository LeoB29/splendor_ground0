from __future__ import annotations

from collections import Counter, defaultdict

from splendor_ai.alphazero.stratified_refill_search import (
    RootRefillTrackingSearch,
    StratifiedRootRefillSearch,
    _core_key,
    stratified_card_schedule,
)
from splendor_ai.alphazero.network import NetworkPrediction
from splendor_ai.alphazero.search import SearchConfig
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv


def test_schedule_is_deterministic_order_invariant_and_balanced() -> None:
    key = ("BUY_VISIBLE", "T1_01", None, (), None)
    cards = [f"T1_{index:02d}" for index in range(1, 11)]
    first = stratified_card_schedule(
        cards, visits=27, search_seed=137, action_key=key, tier=1
    )
    second = stratified_card_schedule(
        reversed(cards), visits=27, search_seed=137, action_key=key, tier=1
    )
    counts = Counter(first)
    assert first == second
    assert set(counts) == set(cards)
    assert max(counts.values()) - min(counts.values()) <= 1


def _bare_search(search_type, state):
    search = object.__new__(search_type)
    search._root_refill_pools = {
        tier: tuple(sorted(card.card_id for card in state.hidden_tier_decks[tier]))
        for tier in (1, 2, 3)
    }
    search._root_refill_counts = defaultdict(Counter)
    search._root_refill_action_meta = {}
    return search


def test_stratification_is_action_conditioned_legal_and_balanced() -> None:
    env = SplendorEnv(seed=137)
    state = env.initial_state()
    actions = [
        action for action in env.legal_actions(state)
        if action.action_type is ActionType.RESERVE_VISIBLE and action.tier == 1
    ][:2]
    search = _bare_search(StratifiedRootRefillSearch, state)
    visits = len(state.hidden_tier_decks[1]) + 7
    for action in actions:
        action_key = canonical_successor_action_key(state, action)
        for simulation in range(visits):
            sampled = state.copy_shallow()
            sampled.hidden_tier_decks[1].reverse()
            prepared = search._prepare_transition(
                sampled,
                action,
                action_key,
                depth=0,
                root_player=state.current_player,
                search_seed=137_100_000,
                simulation=simulation,
            )
            assert prepared.hidden_tier_decks[1][0].card_id in search._root_refill_pools[1]
    telemetry = search.refill_telemetry()
    assert len(telemetry["actions"]) == 2
    assert all(item["visits"] == visits for item in telemetry["actions"].values())
    assert all(item["max_minus_min_count"] <= 1 for item in telemetry["actions"].values())


def test_ordinary_tracking_records_without_changing_refill() -> None:
    env = SplendorEnv(seed=138)
    state = env.initial_state()
    action = next(
        action for action in env.legal_actions(state)
        if action.action_type is ActionType.RESERVE_VISIBLE and action.tier == 1
    )
    action_key = canonical_successor_action_key(state, action)
    search = _bare_search(RootRefillTrackingSearch, state)
    before = state.hidden_tier_decks[1][0].card_id
    prepared = search._prepare_transition(
        state,
        action,
        action_key,
        depth=0,
        root_player=state.current_player,
        search_seed=1,
        simulation=0,
    )
    assert prepared is state
    assert state.hidden_tier_decks[1][0].card_id == before
    assert next(iter(search.refill_telemetry()["actions"].values()))["counts"][before] == 1


def test_nonroot_and_nonrefill_transitions_are_unchanged() -> None:
    env = SplendorEnv(seed=139)
    state = env.initial_state()
    refill = next(
        action for action in env.legal_actions(state)
        if action.action_type is ActionType.RESERVE_VISIBLE
    )
    nonrefill = next(
        action for action in env.legal_actions(state)
        if action.action_type is ActionType.TAKE_TOKENS
    )
    search = _bare_search(StratifiedRootRefillSearch, state)
    original = {
        tier: tuple(card.card_id for card in state.hidden_tier_decks[tier])
        for tier in (1, 2, 3)
    }
    search._prepare_transition(
        state,
        refill,
        canonical_successor_action_key(state, refill),
        depth=1,
        root_player=state.current_player,
        search_seed=2,
        simulation=0,
    )
    search._prepare_transition(
        state,
        nonrefill,
        canonical_successor_action_key(state, nonrefill),
        depth=0,
        root_player=state.current_player,
        search_seed=2,
        simulation=1,
    )
    assert original == {
        tier: tuple(card.card_id for card in state.hidden_tier_decks[tier])
        for tier in (1, 2, 3)
    }
    assert not search.refill_telemetry()["actions"]


def test_core_key_ignores_payment_and_noble_but_not_target() -> None:
    first = ("BUY_VISIBLE", "T2_10", None, (("white", -2),), "N01")
    second = ("BUY_VISIBLE", "T2_10", None, (("white", -1), ("gold", -1)), "N02")
    other = ("BUY_VISIBLE", "T2_11", None, (("white", -2),), "N01")
    assert _core_key(first) == _core_key(second)
    assert _core_key(first) != _core_key(other)


class _ZeroValueRefillInference:
    def predict_one(self, _state, actions):
        return NetworkPrediction(
            logits=tuple(
                2.0 if action.action_type is ActionType.RESERVE_VISIBLE else 0.0
                for action in actions
            ),
            value=0.0,
        )


def test_full_stratified_search_is_reproducible_hidden_order_invariant_and_equal_budget() -> None:
    env = SplendorEnv(seed=140)
    state = env.initial_state()
    reversed_state = state.copy_shallow()
    for tier in (1, 2, 3):
        reversed_state.hidden_tier_decks[tier].reverse()
    config = SearchConfig(
        simulations=64,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        temperature=0.0,
        temperature_turns=0,
    )
    ordinary = RootRefillTrackingSearch(_ZeroValueRefillInference(), config)
    treatment_a = StratifiedRootRefillSearch(_ZeroValueRefillInference(), config)
    treatment_b = StratifiedRootRefillSearch(_ZeroValueRefillInference(), config)
    ordinary_result = ordinary.run(env, state, self_play=False, seed=137_100_140)
    result_a = treatment_a.run(env, state, self_play=False, seed=137_100_140)
    result_b = treatment_b.run(env, reversed_state, self_play=False, seed=137_100_140)
    assert sum(edge.visits for edge in ordinary_result.root_edges) == 64
    assert sum(edge.visits for edge in result_a.root_edges) == 64
    assert [
        (edge.action_key, edge.visits, edge.q) for edge in result_a.root_edges
    ] == [
        (edge.action_key, edge.visits, edge.q) for edge in result_b.root_edges
    ]
    assert treatment_a.refill_telemetry() == treatment_b.refill_telemetry()
    assert treatment_a.refill_telemetry()["actions"]
    assert all(
        item["max_minus_min_count"] <= 1
        for item in treatment_a.refill_telemetry()["actions"].values()
    )
