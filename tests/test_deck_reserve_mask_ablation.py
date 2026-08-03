"""Contract tests for the deck-reserve mask ablation (`071`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from splendor_ai.bots import (
    DeckReserveMaskTelemetry,
    DeckReserveMaskedAttentionScorerBot,
    LegalActionAttentionScorerBot,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_deck_reserve_mask_ablation import (
    CHECKPOINT_035_SHA256,
    MINIMUM_DOSE,
    PAIRS,
    SEED_START,
    sha256,
)


CHECKPOINT = Path(
    "outputs/legal_action_scorer_035_legal_observation/"
    "legal_action_attention_scorer_best.pt"
)


requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="frozen 035 checkpoint is not present",
)


def test_protocol_constants_match_preregistration() -> None:
    """The runner must not drift from the frozen protocol document."""
    assert SEED_START == 740_000
    assert PAIRS == 1_000
    assert MINIMUM_DOSE == pytest.approx(0.010)
    text = Path("docs/experiments/deck_reserve_mask_ablation_071.md").read_text(
        encoding="utf-8"
    )
    assert "`740000`--`740999`" in text
    assert "1,000 fresh paired seeds" in text
    assert CHECKPOINT_035_SHA256 in text


@requires_checkpoint
def test_checkpoint_hash_is_frozen() -> None:
    assert sha256(CHECKPOINT) == CHECKPOINT_035_SHA256


def test_telemetry_merge_is_additive() -> None:
    left = DeckReserveMaskTelemetry()
    left.decisions = 3
    left.deck_reserve_legal = 2
    left.unmasked_choice_was_deck_reserve = 1
    left.switched = 1
    left.deck_only_fallback = 0
    left.switched_replacement_types = {"TAKE_TOKENS": 1}

    right = DeckReserveMaskTelemetry()
    right.decisions = 5
    right.deck_reserve_legal = 1
    right.unmasked_choice_was_deck_reserve = 2
    right.switched = 2
    right.deck_only_fallback = 1
    right.switched_replacement_types = {"TAKE_TOKENS": 1, "RESERVE_VISIBLE": 1}

    left.merge(right.as_dict())
    assert left.decisions == 8
    assert left.deck_reserve_legal == 3
    assert left.unmasked_choice_was_deck_reserve == 3
    assert left.switched == 3
    assert left.deck_only_fallback == 1
    assert left.switched_replacement_types == {
        "TAKE_TOKENS": 2,
        "RESERVE_VISIBLE": 1,
    }
    assert left.switch_fraction == pytest.approx(3 / 8)


def test_telemetry_reset_clears_every_counter() -> None:
    telemetry = DeckReserveMaskTelemetry()
    telemetry.decisions = 4
    telemetry.switched = 2
    telemetry.switched_replacement_types = {"TAKE_TOKENS": 2}
    telemetry.reset()
    assert telemetry.as_dict() == {
        "decisions": 0,
        "deck_reserve_legal": 0,
        "unmasked_choice_was_deck_reserve": 0,
        "switched": 0,
        "deck_only_fallback": 0,
        "switch_fraction": 0.0,
        "switched_replacement_types": {},
    }


@requires_checkpoint
def test_masked_bot_never_returns_a_deck_reserve_when_alternatives_exist() -> None:
    bot = DeckReserveMaskedAttentionScorerBot(CHECKPOINT)
    env = SplendorEnv(seed=740_000)
    state = env.initial_state()
    for _ in range(60):
        if state.terminal:
            break
        actions = env.legal_actions(state)
        if not actions:
            break
        chosen = bot.choose_action(env, state, actions)
        assert chosen is not None
        non_deck = [
            action
            for action in actions
            if action.action_type is not ActionType.RESERVE_DECK
        ]
        if non_deck:
            assert chosen.action_type is not ActionType.RESERVE_DECK
        state = env.step(state, chosen)
    assert bot.telemetry.decisions > 0


@requires_checkpoint
def test_mask_only_changes_deck_reserve_decisions() -> None:
    """Where the parent's greedy action is legal under the mask, both agree."""
    masked = DeckReserveMaskedAttentionScorerBot(CHECKPOINT)
    parent = LegalActionAttentionScorerBot(CHECKPOINT)
    env = SplendorEnv(seed=740_001)
    state = env.initial_state()
    compared = 0
    for _ in range(60):
        if state.terminal:
            break
        actions = env.legal_actions(state)
        if not actions:
            break
        parent_choice = parent.choose_action(env, state, actions)
        masked_choice = masked.choose_action(env, state, actions)
        assert parent_choice is not None and masked_choice is not None
        if parent_choice.action_type is not ActionType.RESERVE_DECK:
            assert masked_choice == parent_choice
        compared += 1
        state = env.step(state, parent_choice)
    assert compared > 0
    assert masked.telemetry.switched == masked.telemetry.unmasked_choice_was_deck_reserve


@requires_checkpoint
def test_masked_bot_is_deterministic() -> None:
    first = DeckReserveMaskedAttentionScorerBot(CHECKPOINT)
    second = DeckReserveMaskedAttentionScorerBot(CHECKPOINT)
    env = SplendorEnv(seed=740_002)
    state = env.initial_state()
    for _ in range(30):
        if state.terminal:
            break
        actions = env.legal_actions(state)
        if not actions:
            break
        left = first.choose_action(env, state, actions)
        right = second.choose_action(env, state, actions)
        assert left == right
        state = env.step(state, left)
    assert first.telemetry.as_dict() == second.telemetry.as_dict()
