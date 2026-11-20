"""Fast target/action controller for experiment 081."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.bots.value_reserve_search_bot import canonical_action_identity
from splendor_ai.training.card_option_q import (
    OptionActionPair,
    collate_card_option_batch,
    enumerate_option_action_pairs,
    load_card_option_model,
)


@dataclass(slots=True)
class CardOptionTelemetry:
    decisions: int = 0
    moved_from_raw035: int = 0
    target_decisions: int = 0
    target_selections: int = 0
    target_keeps: int = 0
    target_switches: int = 0
    target_clears: int = 0
    target_completions: int = 0
    target_losses: int = 0
    purchases: int = 0
    deck_reserves: int = 0
    gold_spent: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    reserves_by_tier: dict[int, int] = field(default_factory=dict)
    purchases_by_tier: dict[int, int] = field(default_factory=dict)
    phase_decisions: dict[str, int] = field(default_factory=dict)
    phase_movement: dict[str, int] = field(default_factory=dict)
    inference_latencies_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        scalars = (
            "decisions", "moved_from_raw035", "target_decisions",
            "target_selections", "target_keeps", "target_switches",
            "target_clears", "target_completions", "target_losses", "purchases",
            "deck_reserves", "gold_spent",
        )
        result: dict[str, Any] = {name: getattr(self, name) for name in scalars}
        for name in (
            "action_counts", "reserves_by_tier", "purchases_by_tier",
            "phase_decisions", "phase_movement",
        ):
            result[name] = {str(k): v for k, v in sorted(getattr(self, name).items())}
        result["inference_latencies_ms"] = list(self.inference_latencies_ms)
        return result

    def merge(self, payload: dict[str, Any]) -> None:
        for name in (
            "decisions", "moved_from_raw035", "target_decisions",
            "target_selections", "target_keeps", "target_switches",
            "target_clears", "target_completions", "target_losses", "purchases",
            "deck_reserves", "gold_spent",
        ):
            setattr(self, name, getattr(self, name) + int(payload[name]))
        for name in (
            "action_counts", "reserves_by_tier", "purchases_by_tier",
            "phase_decisions", "phase_movement",
        ):
            destination = getattr(self, name)
            for key, value in dict(payload[name]).items():
                normalized: Any = int(key) if name in ("reserves_by_tier", "purchases_by_tier") else str(key)
                destination[normalized] = destination.get(normalized, 0) + int(value)
        self.inference_latencies_ms.extend(float(v) for v in payload["inference_latencies_ms"])


class CardOptionQBot:
    def __init__(
        self,
        checkpoint_path: str | Path,
        raw035_bot: Any | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = load_card_option_model(self.checkpoint_path, device)
        self.raw035 = raw035_bot
        self.telemetry = CardOptionTelemetry()
        self.current_target_id: str | None = None
        self.player_id: int | None = None

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def start_game(self, seed: int) -> None:  # noqa: ARG002
        self.telemetry = CardOptionTelemetry()
        self.current_target_id = None
        self.player_id = None

    def score_pairs(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> tuple[list[OptionActionPair], torch.Tensor]:
        pairs = enumerate_option_action_pairs(state, legal_actions)
        batch = collate_card_option_batch([(state, pairs)], device=self.device)
        with torch.no_grad():
            scores = self.model(**batch)[0, : len(pairs)].detach().cpu()
        return pairs, scores

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        self.player_id = state.current_player
        began = time.perf_counter()
        pairs, scores = self.score_pairs(state, actions)
        selected = pairs[int(torch.argmax(scores).item())]
        self.telemetry.inference_latencies_ms.append((time.perf_counter() - began) * 1000.0)
        self._record_target_choice(selected)
        raw = self.raw035.choose_action(env, state, actions) if self.raw035 is not None else None
        self._record_action(state, selected.action, raw)
        return selected.action

    def _record_target_choice(self, selected: OptionActionPair) -> None:
        new_target = None if selected.target.card is None else selected.target.card.card_id
        if new_target is not None:
            self.telemetry.target_decisions += 1
        if new_target == self.current_target_id:
            if new_target is not None:
                self.telemetry.target_keeps += 1
            return
        if self.current_target_id is not None:
            if new_target is None:
                self.telemetry.target_clears += 1
            else:
                self.telemetry.target_switches += 1
        if new_target is not None:
            self.telemetry.target_selections += 1
        self.current_target_id = new_target

    def _record_action(
        self,
        state: SplendorState,
        action: Action,
        raw: Action | None,
    ) -> None:
        telemetry = self.telemetry
        telemetry.decisions += 1
        name = action.action_type.name
        telemetry.action_counts[name] = telemetry.action_counts.get(name, 0) + 1
        phase = "early" if state.turn_index < 20 else "mid" if state.turn_index < 40 else "late"
        telemetry.phase_decisions[phase] = telemetry.phase_decisions.get(phase, 0) + 1
        if raw is not None and canonical_action_identity(state, action) != canonical_action_identity(state, raw):
            telemetry.moved_from_raw035 += 1
            telemetry.phase_movement[phase] = telemetry.phase_movement.get(phase, 0) + 1
        telemetry.gold_spent += sum(token == "gold" for token in (action.spend_tokens or ()))
        if action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK):
            if action.tier is not None:
                telemetry.reserves_by_tier[action.tier] = telemetry.reserves_by_tier.get(action.tier, 0) + 1
            if action.action_type is ActionType.RESERVE_DECK:
                telemetry.deck_reserves += 1
        if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
            telemetry.purchases += 1
            card = (
                state.visible_tier_cards[action.tier][action.market_index]
                if action.action_type is ActionType.BUY_VISIBLE
                else state.players[state.current_player].reserved_cards[action.reserved_index]
            )
            telemetry.purchases_by_tier[card.tier] = telemetry.purchases_by_tier.get(card.tier, 0) + 1

    def observe_action(
        self,
        before: SplendorState,
        action: Action,
        after: SplendorState,
    ) -> None:  # noqa: ARG002
        if self.current_target_id is None or self.player_id is None:
            return
        target = self.current_target_id
        player = after.players[self.player_id]
        if any(card.card_id == target for card in player.purchased_cards):
            self.telemetry.target_completions += 1
            self.current_target_id = None
            return
        reserved = any(card.card_id == target for card in player.reserved_cards)
        visible = any(
            card.card_id == target
            for cards in after.visible_tier_cards.values()
            for card in cards
        )
        if not reserved and not visible:
            self.telemetry.target_losses += 1
            self.current_target_id = None
