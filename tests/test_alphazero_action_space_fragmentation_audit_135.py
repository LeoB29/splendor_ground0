from __future__ import annotations

import pytest

from splendor_ai.alphazero.action_fragmentation_audit import (
    EXPECTED_VISITS,
    analyze_row,
    intent_key,
    phase,
)


def _payload(
    action_type: str,
    *,
    card_id: int = 0,
    tier: int | None = None,
    take: tuple[str, ...] = (),
    returned: tuple[str, ...] = (),
    spent: tuple[str, ...] = (),
    noble: str | None = None,
) -> tuple[dict[str, object], int]:
    return (
        {
            "action_type": action_type,
            "market_index": 0 if "VISIBLE" in action_type else None,
            "metadata": [],
            "noble_id": noble,
            "reserved_index": 0 if action_type == "BUY_RESERVED" else None,
            "return_tokens": list(returned),
            "spend_tokens": list(spent),
            "take_tokens": list(take),
            "tier": tier,
        },
        card_id,
    )


def _row(actions: list[tuple[dict[str, object], int]]) -> dict[str, object]:
    count = len(actions)
    visits = [0] * count
    visits[0] = EXPECTED_VISITS
    return {
        "game_seed": 1,
        "turn_index": 7,
        "player_id": 0,
        "action_payloads": [payload for payload, _card_id in actions],
        "action_card_ids": [card_id for _payload, card_id in actions],
        "action_noble_ids": [0] * count,
        "root_network_priors": [1.0 / count] * count,
        "root_visits": visits,
        "root_q_values": [0.0] * count,
        "raw_policy_top_row": 0,
        "visit_argmax_row": 0,
    }


def test_frozen_intent_grouping_merges_only_implementation_details() -> None:
    buy_a, card_a = _payload("BUY_VISIBLE", card_id=17, spent=("red", "gold"), noble="N1")
    buy_b, card_b = _payload("BUY_VISIBLE", card_id=17, spent=("red", "red"), noble="N2")
    buy_other, other_id = _payload("BUY_VISIBLE", card_id=18, spent=("red", "gold"), noble="N1")
    assert intent_key(buy_a, card_a, preserve_noble=False) == intent_key(
        buy_b, card_b, preserve_noble=False
    )
    assert intent_key(buy_a, card_a, preserve_noble=True) != intent_key(
        buy_b, card_b, preserve_noble=True
    )
    assert intent_key(buy_a, card_a, preserve_noble=False) != intent_key(
        buy_other, other_id, preserve_noble=False
    )


def test_take_intent_keeps_taken_colors_but_ignores_returns() -> None:
    first, _ = _payload(
        "TAKE_TOKENS", take=("red", "blue", "green"), returned=("white",)
    )
    alias, _ = _payload(
        "TAKE_TOKENS", take=("green", "red", "blue"), returned=("black",)
    )
    other, _ = _payload("TAKE_TOKENS", take=("red", "blue"), returned=("white",))
    assert intent_key(first, 0, preserve_noble=False) == intent_key(
        alias, 0, preserve_noble=False
    )
    assert intent_key(first, 0, preserve_noble=False) != intent_key(
        other, 0, preserve_noble=False
    )


def test_row_analysis_detects_exact_disagreement_rescued_by_intent() -> None:
    actions = [
        _payload("BUY_VISIBLE", card_id=17, spent=("red", "gold")),
        _payload("BUY_VISIBLE", card_id=17, spent=("red", "red")),
        _payload("BUY_VISIBLE", card_id=18, spent=("blue",)),
    ]
    row = _row(actions)
    row["root_network_priors"] = [0.45, 0.10, 0.45]
    row["root_visits"] = [400, 624, 0]
    row["root_q_values"] = [0.1, 0.11, -0.2]
    row["raw_policy_top_row"] = 0
    row["visit_argmax_row"] = 1
    result = analyze_row(row)
    assert not result["exact_agreement"]
    assert result["noble_intent_agreement"]
    assert result["core_intent_agreement"]
    assert result["core_intent_count"] == 2
    assert result["core_multi_visit_mass"] == 1.0
    assert result["winning_core_concentration"] == pytest.approx(624 / 1024)
    assert result["q_ranges"] == pytest.approx([0.01])


def test_row_analysis_rejects_broken_visit_accounting() -> None:
    row = _row([_payload("PASS")])
    row["root_visits"] = [EXPECTED_VISITS - 1]
    with pytest.raises(ValueError, match="Invalid root visit vector"):
        analyze_row(row)


def test_phase_boundaries_are_frozen() -> None:
    assert phase(0) == "early"
    assert phase(19) == "early"
    assert phase(20) == "middle"
    assert phase(49) == "middle"
    assert phase(50) == "late"

