from __future__ import annotations

import math
import subprocess
import sys

from splendor_ai.bots import ShallowSearchBot
from splendor_ai.bots.value_reserve_search_bot import (
    ValueReserveLeafEvaluator,
    canonical_action_identity,
    reserved_option_value,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, PlayerState, RESERVE_SOURCE_VISIBLE
from splendor_ai.eval.run_value_reserve_leaf_search_teacher import (
    _bootstrap,
    _mechanics_gates,
)


def test_reserved_option_value_is_funded_fraction_of_prestige() -> None:
    player = PlayerState(player_id=0)
    player.tokens["red"] = 1
    player.tokens["gold"] = 1
    target = Card("target", 3, "blue", 4, {"red": 7})
    below_filter = Card("weak", 2, "blue", 1, {"red": 3})

    assert math.isclose(reserved_option_value(target, player), 4_000 * 2 / 7)
    assert reserved_option_value(below_filter, player) == 0.0


def test_leaf_bonus_uses_best_target_not_sum() -> None:
    env = SplendorEnv(seed=4)
    state = env.initial_state()
    player = state.players[0]
    player.tokens["red"] = 1
    player.tokens["gold"] = 1
    first = Card("first", 3, "blue", 4, {"red": 7})
    second = Card("second", 3, "green", 5, {"red": 7, "blue": 3})
    player.reserve_card(first, RESERVE_SOURCE_VISIBLE)
    player.reserve_card(second, RESERVE_SOURCE_VISIBLE)

    values = [reserved_option_value(card, player) for card in (first, second)]
    assert ValueReserveLeafEvaluator.strategic_bonus(state, 0) == max(values)
    assert max(values) < sum(values)


def test_search_keeps_all_twelve_visible_reserve_targets() -> None:
    env = SplendorEnv(seed=77)
    state = env.initial_state()
    legal = env.legal_actions(state)
    bot = ShallowSearchBot(
        depth=1,
        max_branching=10_000,
        max_buy_actions=10_000,
        max_reserve_actions=10_000,
        max_take_actions=10_000,
        max_pass_actions=10_000,
        seed=0,
    )
    candidates = bot._ordered_candidates(  # noqa: SLF001 - contract regression test
        env, state, legal, root_player=state.current_player
    )
    visible_reserves = [
        candidate
        for candidate in candidates
        if candidate.action.action_type == ActionType.RESERVE_VISIBLE
    ]

    assert len(visible_reserves) == 12
    assert len(
        {
            (candidate.action.tier, candidate.action.market_index)
            for candidate in visible_reserves
        }
    ) == 12


def test_canonical_buy_identity_ignores_payment_alias() -> None:
    env = SplendorEnv(seed=11)
    state = env.initial_state()
    first = Action(
        ActionType.BUY_VISIBLE,
        tier=1,
        market_index=0,
        spend_tokens=("red", "gold"),
    )
    second = Action(
        ActionType.BUY_VISIBLE,
        tier=1,
        market_index=0,
        spend_tokens=("red", "red"),
    )
    assert canonical_action_identity(state, first) == canonical_action_identity(
        state, second
    )


def test_mechanics_gate_is_conjunctive() -> None:
    passing = {
        "movement_from_raw035": 0.15,
        "movement_from_search_ablation": 0.15,
        "qualifying_visible_reserve_rate": 0.05,
        "qualifying_visible_reserve_conversion_rate": 0.50,
        "mean_purchases_per_candidate_game": 5.0,
    }
    assert all(_mechanics_gates(passing).values())
    failing = dict(passing)
    failing["movement_from_search_ablation"] = 0.149
    assert not all(_mechanics_gates(failing).values())


def test_bootstrap_is_deterministic_and_clustered() -> None:
    values = [0.0, 0.5, 1.0, 1.0]
    assert _bootstrap(values, 100, 77) == _bootstrap(values, 100, 77)


def test_root_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "run_value_reserve_leaf_search_teacher_077.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Value-reserve leaf search" in result.stdout
