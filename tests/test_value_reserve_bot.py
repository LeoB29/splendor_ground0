"""Contract tests for the scripted value-reserve teacher (`072`)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from splendor_ai.bots import (
    EFFICIENCY_FILTER,
    ValueReserveTeacherBot,
    ValueReserveTelemetry,
    effective_cost,
    effective_value,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.constants import (
    BANK_GEMS_PER_COLOR_2P,
    GOLD_TOKENS,
    TOKEN_COLORS,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card
from splendor_ai.eval.run_scripted_value_teacher import (
    CHECKPOINT_035_SHA256,
    PAIRS,
    SEED_START,
)


CARDS_CSV = Path("docs/splendor_base_cards.csv")


def _card(cost: dict[str, int], points: int, tier: int = 3) -> Card:
    return Card(
        card_id="X",
        tier=tier,
        bonus_color="white",
        points=points,
        cost={color: cost.get(color, 0) for color in TOKEN_COLORS},
    )


def test_protocol_constants_match_preregistration() -> None:
    assert SEED_START == 750_000
    assert PAIRS == 200
    assert EFFICIENCY_FILTER == pytest.approx(0.40)
    text = Path("docs/experiments/scripted_value_teacher_072.md").read_text(
        encoding="utf-8"
    )
    assert "`750000`--`750199`" in text
    assert "200 fresh paired seeds" in text
    assert CHECKPOINT_035_SHA256 in text


def test_effective_cost_and_value_use_bonuses_only() -> None:
    card = _card({"black": 7}, points=4)
    assert effective_cost(card, {color: 0 for color in TOKEN_COLORS}) == 7
    assert effective_value(card, {color: 0 for color in TOKEN_COLORS}) == pytest.approx(
        4 / 7
    )
    bonuses = {color: 0 for color in TOKEN_COLORS}
    bonuses["black"] = 4
    assert effective_cost(card, bonuses) == 3
    assert effective_value(card, bonuses) == pytest.approx(4 / 3)
    bonuses["black"] = 7
    assert effective_cost(card, bonuses) == 0
    assert effective_value(card, bonuses) == float("inf")


@pytest.mark.skipif(not CARDS_CSV.exists(), reason="card table not present")
def test_efficiency_filter_admits_exactly_eighteen_cards_at_zero_bonus() -> None:
    """The protocol's headline claim about the filter must hold on the real deck."""
    zero = {color: 0 for color in TOKEN_COLORS}
    admitted = []
    for row in csv.DictReader(CARDS_CSV.open(encoding="utf-8")):
        card = Card(
            card_id=row["card_id"],
            tier=int(row["tier"]),
            bonus_color=row["bonus_color"],
            points=int(row["prestige_points"]),
            cost={color: int(row[f"cost_{color}"]) for color in TOKEN_COLORS},
        )
        if effective_value(card, zero) >= EFFICIENCY_FILTER:
            admitted.append(card)
    assert len(admitted) == 18
    by_tier: dict[int, int] = {}
    for card in admitted:
        by_tier[card.tier] = by_tier.get(card.tier, 0) + 1
    assert by_tier == {2: 8, 3: 10}


@pytest.mark.skipif(not CARDS_CSV.exists(), reason="card table not present")
def test_mono_colour_seven_cost_is_unpayable_without_gold_or_bonuses() -> None:
    """The mechanical premise of the protocol: 7X needs gold or discounts."""
    assert BANK_GEMS_PER_COLOR_2P == 4
    assert GOLD_TOKENS == 5
    mono_seven = [
        row
        for row in csv.DictReader(CARDS_CSV.open(encoding="utf-8"))
        if int(row["total_cost_tokens"]) == 7
        and sum(1 for color in TOKEN_COLORS if int(row[f"cost_{color}"]) > 0) == 1
    ]
    assert len(mono_seven) == 5
    for row in mono_seven:
        peak = max(int(row[f"cost_{color}"]) for color in TOKEN_COLORS)
        assert peak == 7
        assert peak > BANK_GEMS_PER_COLOR_2P


def test_telemetry_merge_and_reset() -> None:
    left = ValueReserveTelemetry()
    left.decisions = 2
    left.reserves = 1
    left.rule_counts = {"r7_reserve_high_value": 1}
    left.reserves_by_tier = {3: 1}
    right = ValueReserveTelemetry()
    right.decisions = 3
    right.reserves = 2
    right.rule_counts = {"r7_reserve_high_value": 2, "r8_take_tokens": 1}
    right.reserves_by_tier = {2: 1, 3: 1}
    left.merge(right.as_dict())
    assert left.decisions == 5
    assert left.reserves == 3
    assert left.rule_counts == {"r7_reserve_high_value": 3, "r8_take_tokens": 1}
    assert left.reserves_by_tier == {2: 1, 3: 2}
    left.reset()
    assert left.decisions == 0 and left.rule_counts == {}


def test_teacher_never_reserves_from_the_deck_and_always_acts() -> None:
    bot = ValueReserveTeacherBot()
    other = ValueReserveTeacherBot()
    env = SplendorEnv(seed=750_000)
    state = env.initial_state()
    for _ in range(400):
        if state.terminal:
            break
        actions = env.legal_actions(state)
        if not actions:
            break
        actor = bot if state.current_player == 0 else other
        chosen = actor.choose_action(env, state, actions)
        assert chosen is not None
        assert chosen in actions
        state = env.step(state, chosen)
    assert bot.telemetry.deck_reserves == 0
    assert other.telemetry.deck_reserves == 0
    assert bot.telemetry.decisions > 0


def test_teacher_is_deterministic() -> None:
    def run() -> tuple[list[str], int]:
        left = ValueReserveTeacherBot()
        right = ValueReserveTeacherBot()
        env = SplendorEnv(seed=750_001)
        state = env.initial_state()
        trace: list[str] = []
        for _ in range(400):
            if state.terminal:
                break
            actions = env.legal_actions(state)
            if not actions:
                break
            actor = left if state.current_player == 0 else right
            chosen = actor.choose_action(env, state, actions)
            trace.append(repr(chosen))
            state = env.step(state, chosen)
        return trace, state.turn_index

    assert run() == run()


def test_teacher_holds_at_most_three_reserves() -> None:
    bot = ValueReserveTeacherBot()
    other = ValueReserveTeacherBot()
    env = SplendorEnv(seed=750_002)
    state = env.initial_state()
    for _ in range(200):
        if state.terminal:
            break
        actions = env.legal_actions(state)
        if not actions:
            break
        actor = bot if state.current_player == 0 else other
        chosen = actor.choose_action(env, state, actions)
        state = env.step(state, chosen)
        for player in state.players:
            assert len(player.reserved_cards) <= 3


def test_073_rules_default_off_so_072_stays_reproducible() -> None:
    from splendor_ai.bots.value_reserve_bot import ValueReserveRules

    rules = ValueReserveRules()
    assert rules.protect_gold is False
    assert rules.reserve_and_clear is False
    assert ValueReserveTeacherBot().rules == rules


def test_073_protocol_constants_and_terminality() -> None:
    from run_scripted_value_teacher_073 import (
        BOOTSTRAP_SEED_073,
        SEED_START_073,
    )

    assert SEED_START_073 == 760_000
    assert BOOTSTRAP_SEED_073 == 760_073
    text = Path("docs/experiments/scripted_value_teacher_073.md").read_text(
        encoding="utf-8"
    )
    assert "`760000`--`760199`" in text
    # Terminality must be stated in the protocol, not only in the runner.
    assert "terminal for this line" in text
    assert "do not re-elicit again" in text


def test_073_reserve_and_clear_keeps_one_live_target() -> None:
    from splendor_ai.bots.value_reserve_bot import ValueReserveRules

    rules = ValueReserveRules(protect_gold=True, reserve_and_clear=True)
    left = ValueReserveTeacherBot(rules)
    right = ValueReserveTeacherBot(rules)
    env = SplendorEnv(seed=760_000)
    state = env.initial_state()
    for _ in range(400):
        if state.terminal:
            break
        actions = env.legal_actions(state)
        if not actions:
            break
        actor = left if state.current_player == 0 else right
        chosen = actor.choose_action(env, state, actions)
        state = env.step(state, chosen)
    assert left.telemetry.multi_target_hand_decisions == 0
    assert right.telemetry.multi_target_hand_decisions == 0


def test_073_gold_discipline_binds_at_least_once() -> None:
    from splendor_ai.bots.value_reserve_bot import ValueReserveRules

    rules = ValueReserveRules(protect_gold=True, reserve_and_clear=True)
    total = 0
    for seed in (760_000, 760_001, 760_002):
        left = ValueReserveTeacherBot(rules)
        right = ValueReserveTeacherBot(rules)
        env = SplendorEnv(seed=seed)
        state = env.initial_state()
        for _ in range(400):
            if state.terminal:
                break
            actions = env.legal_actions(state)
            if not actions:
                break
            actor = left if state.current_player == 0 else right
            state = env.step(state, actor.choose_action(env, state, actions))
        total += left.telemetry.gold_discipline_skips
        total += right.telemetry.gold_discipline_skips
    assert total > 0


def test_win_now_rule_outranks_everything() -> None:
    """A purchase reaching 15 must be taken as soon as it is legal."""
    bot = ValueReserveTeacherBot()
    other = ValueReserveTeacherBot()
    env = SplendorEnv(seed=750_003)
    state = env.initial_state()
    for _ in range(400):
        if state.terminal:
            break
        actions = env.legal_actions(state)
        if not actions:
            break
        actor = bot if state.current_player == 0 else other
        player = state.players[state.current_player]
        winning = [
            action
            for action in actions
            if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
            and player.score
            + (
                state.visible_tier_cards[action.tier][action.market_index].points
                if action.action_type is ActionType.BUY_VISIBLE
                else player.reserved_cards[action.reserved_index].points
            )
            >= 15
        ]
        chosen = actor.choose_action(env, state, actions)
        if winning:
            assert chosen in winning
        state = env.step(state, chosen)
