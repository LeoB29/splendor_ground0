"""Persistent executor-conditioned target controller for experiment 083."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch

from splendor_ai.bots.card_option_q_bot import CardOptionTelemetry
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.bots.value_reserve_search_bot import canonical_action_identity
from splendor_ai.card_target_executor import target_executor_action
from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, SplendorState
from splendor_ai.training.card_option_q import enumerate_option_targets
from splendor_ai.training.executor_conditioned_target import (
    ExecutorTargetCandidate,
    collate_executor_target_batch,
    load_executor_conditioned_model,
)


@dataclass(frozen=True, slots=True)
class _ExecutorScore:
    action: Action
    action_index: int
    score: float


class ExecutorConditionedTargetBot:
    def __init__(
        self,
        checkpoint_path: str | Path,
        checkpoint_035: str | Path,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_035 = Path(checkpoint_035)
        self.device = device
        self.model = load_executor_conditioned_model(self.checkpoint_path, device)
        self.raw_scorer = LegalActionAttentionScorerBot(self.checkpoint_035, device)
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

    def score_candidates(
        self,
        state: SplendorState,
        candidates: list[ExecutorTargetCandidate],
    ) -> torch.Tensor:
        batch = collate_executor_target_batch(
            [(state, candidates)], device=self.device
        )
        with torch.no_grad():
            return self.model(**batch)[0, : len(candidates)].detach().cpu()

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        supplied = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not supplied:
            return None
        actions = list(actions_by_canonical_successor_key(state, supplied).values())
        self.player_id = state.current_player
        codec = ActionCodec()
        values = self.raw_scorer.score_actions(state, actions)
        raw_scores = tuple(
            sorted(
                (
                    _ExecutorScore(action, codec.encode(state, action), score)
                    for action, score in zip(actions, values, strict=True)
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        raw_action = raw_scores[0].action
        target = self._live_target(state)
        if self.current_target_id is not None and target is None:
            self.telemetry.target_losses += 1
            self.current_target_id = None
        if target is None:
            targets = enumerate_option_targets(state, state.current_player)
            candidates = []
            for option in targets:
                action = (
                    raw_action
                    if option.card is None
                    else target_executor_action(state, actions, raw_scores, option.card)
                )
                candidates.append(
                    ExecutorTargetCandidate(option, action, codec.encode(state, action))
                )
            began = time.perf_counter()
            scores = self.score_candidates(state, candidates)
            selected = candidates[int(torch.argmax(scores).item())]
            self.telemetry.inference_latencies_ms.append(
                (time.perf_counter() - began) * 1000.0
            )
            if selected.target.card is None:
                action = raw_action
            else:
                self.current_target_id = selected.target.card.card_id
                self.telemetry.target_selections += 1
                action = selected.action
        else:
            self.telemetry.target_keeps += 1
            action = target_executor_action(state, actions, raw_scores, target)
        if self.current_target_id is not None:
            self.telemetry.target_decisions += 1
        self._record_action(state, action, raw_action)
        return action

    def _live_target(self, state: SplendorState) -> Card | None:
        if self.current_target_id is None or self.player_id is None:
            return None
        player = state.players[self.player_id]
        for card in player.reserved_cards:
            if card.card_id == self.current_target_id:
                return card
        for cards in state.visible_tier_cards.values():
            for card in cards:
                if card.card_id == self.current_target_id:
                    return card
        return None

    def _record_action(
        self,
        state: SplendorState,
        action: Action,
        raw: Action,
    ) -> None:
        telemetry = self.telemetry
        telemetry.decisions += 1
        name = action.action_type.name
        telemetry.action_counts[name] = telemetry.action_counts.get(name, 0) + 1
        phase = "early" if state.turn_index < 20 else "mid" if state.turn_index < 40 else "late"
        telemetry.phase_decisions[phase] = telemetry.phase_decisions.get(phase, 0) + 1
        if canonical_action_identity(state, action) != canonical_action_identity(state, raw):
            telemetry.moved_from_raw035 += 1
            telemetry.phase_movement[phase] = telemetry.phase_movement.get(phase, 0) + 1
        telemetry.gold_spent += sum(token == "gold" for token in (action.spend_tokens or ()))
        if action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK):
            if action.tier is not None:
                telemetry.reserves_by_tier[action.tier] = (
                    telemetry.reserves_by_tier.get(action.tier, 0) + 1
                )
            if action.action_type is ActionType.RESERVE_DECK:
                telemetry.deck_reserves += 1
        if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
            telemetry.purchases += 1
            card = (
                state.visible_tier_cards[action.tier][action.market_index]
                if action.action_type is ActionType.BUY_VISIBLE
                else state.players[state.current_player].reserved_cards[action.reserved_index]
            )
            telemetry.purchases_by_tier[card.tier] = (
                telemetry.purchases_by_tier.get(card.tier, 0) + 1
            )

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
