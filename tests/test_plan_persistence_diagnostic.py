from dataclasses import replace

import pytest

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_plan_persistence_diagnostic import (
    PlanPersistenceConfig,
    _anchors_for_game,
    build_parser,
    card_progress_for_take,
    group_assignment,
    plan_persistence_gates,
    remaining_card_cost,
    sanitize_snapshot,
)


def _take(*colors: str) -> Action:
    return Action(action_type=ActionType.TAKE_TOKENS, take_tokens=tuple(colors))


def test_card_progress_respects_tokens_discounts_and_gold() -> None:
    state = SplendorEnv(seed=3).initial_state()
    player = state.players[0]
    card = state.visible_tier_cards[2][0]
    player.bonuses.update({color: 0 for color in player.bonuses})
    player.tokens.update({color: 0 for color in player.tokens})
    needed_color = next(color for color, cost in card.cost.items() if cost > 0)
    before = remaining_card_cost(state, card)
    progress = card_progress_for_take(state, card, _take(needed_color))

    assert before > 0
    assert progress == 1

    player.tokens["gold"] = 1
    assert remaining_card_cost(state, card) == before - 1


def test_sanitized_snapshot_removes_private_hidden_information() -> None:
    state = SplendorEnv(seed=7).initial_state()
    state.players[1].reserve_card(state.hidden_tier_decks[1][0], "deck")
    snapshot = sanitize_snapshot(serialize_state_snapshot(state), observer=0)

    assert all(not cards for cards in snapshot["hidden_tier_decks"].values())
    assert snapshot["players"][1]["reserved_card_ids"] == []
    assert snapshot["players"][0]["reserved_card_ids"] == []
    assert "winner" not in snapshot


def test_group_assignment_balances_each_split() -> None:
    cells = {"discovery": set(), "confirmation": set()}
    counts = {"discovery": {}, "confirmation": {}}
    for game_index in range(256):
        opponent, seat, split = group_assignment(256, game_index)
        cells[split].add((opponent, seat))
        counts[split][(opponent, seat)] = counts[split].get((opponent, seat), 0) + 1

    assert len(cells["discovery"]) == 8
    assert len(cells["confirmation"]) == 8
    assert set(counts["discovery"].values()) == {16}
    assert set(counts["confirmation"].values()) == {16}


def test_anchor_compares_actual_take_with_same_state_alternatives() -> None:
    state = SplendorEnv(seed=11).initial_state()
    env = SplendorEnv()
    target = next(
        card
        for cards in state.visible_tier_cards.values()
        for card in cards
        if any(cost > 0 for cost in card.cost.values())
    )
    useful = next(
        action
        for action in env._legal_take_actions(state.players[0], state)
        if card_progress_for_take(state, target, action) > 0
    )
    target_tier = target.tier
    target_index = state.visible_tier_cards[target_tier].index(target)
    future = state.copy_shallow()
    future.turn_index = 2
    rows = [
        {
            "game_index": 0,
            "environment_seed": 390000,
            "split": "discovery",
            "opponent": "035",
            "actor_seat": 0,
            "turn_index": 0,
            "action_payload": {
                "action_type": "TAKE_TOKENS",
                "take_tokens": list(useful.take_tokens),
                "return_tokens": list(useful.return_tokens),
            },
            "public_state_snapshot": sanitize_snapshot(
                serialize_state_snapshot(state), 0
            ),
        },
        {
            "game_index": 0,
            "environment_seed": 390000,
            "split": "discovery",
            "opponent": "035",
            "actor_seat": 0,
            "turn_index": 2,
            "action_payload": {
                "action_type": "RESERVE_VISIBLE",
                "tier": target_tier,
                "market_index": target_index,
            },
            "public_state_snapshot": sanitize_snapshot(
                serialize_state_snapshot(future), 0
            ),
        },
    ]

    anchors = _anchors_for_game(rows, horizon_own_turns=4)

    assert len(anchors) == 1
    assert anchors[0]["target_card_id"] == target.card_id
    assert anchors[0]["actual_progress"] > 0
    assert anchors[0]["legal_take_count"] > 1


def test_frozen_gate_requires_both_split_replication(tmp_path) -> None:
    config = PlanPersistenceConfig(
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        policy_checkpoint=tmp_path / "035.pt",
        checkpoint_016=tmp_path / "016.pt",
        checkpoint_023=tmp_path / "023.pt",
        minimum_anchors_per_split=2,
        minimum_multi_take_episodes_per_split=1,
        minimum_cell_anchors=1,
        minimum_phase_anchors=1,
    )
    good = {
        "anchors": 2,
        "progress_advantage_ci": {"mean": 0.2, "ci95_low": 0.1},
        "progress_rank_percentile_ci": {"mean": 0.7, "ci95_low": 0.6},
    }
    multi = {
        "episodes": 1,
        "progress_advantage_ci": {"mean": 0.2, "ci95_low": 0.1},
    }
    cell = {"anchors": 1, "mean_progress_advantage": 0.1}
    analysis = {
        "splits": {"discovery": dict(good), "confirmation": dict(good)},
        "multi_take_splits": {
            "discovery": dict(multi),
            "confirmation": dict(multi),
        },
        "breakdowns": {
            "opponent_seat": {
                f"{opponent}:seat{seat}": dict(cell)
                for opponent in ("035", "016", "shallow", "023")
                for seat in (0, 1)
            },
            "phase": {
                "0-14": dict(cell),
                "15-29": dict(cell),
                "30-44": dict(cell),
            },
        },
    }
    assert all(plan_persistence_gates(analysis, config).values())

    analysis["splits"]["confirmation"]["progress_advantage_ci"]["ci95_low"] = -0.01
    assert not plan_persistence_gates(analysis, config)[
        "confirmation_progress_advantage"
    ]


def test_parser_preregisters_fresh_cpu_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.games == 256
    assert args.seed_start == 390_000
    assert args.horizon_own_turns == 4
    assert args.workers == 4
    assert args.device == "cpu"
