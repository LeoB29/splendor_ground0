"""Greedy recurrent policy bot with an information-legal public transcript."""

from __future__ import annotations

from pathlib import Path

import torch

from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.recurrent_policy import (
    PublicHistoryTracker,
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
)


class RecurrentLegalActionAttentionScorerBot:
    """Greedy recurrent scorer whose history is supplied by match hooks."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._model = load_recurrent_policy_checkpoint(
            self._checkpoint_path,
            device=device,
        )
        self._model.eval()
        self._history = PublicHistoryTracker()
        self._game_seed: int | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def public_history(self) -> tuple[tuple[object, ...], ...]:
        return self._history.events

    @property
    def game_seed(self) -> int | None:
        return self._game_seed

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def start_game(self, seed: int) -> None:
        self._game_seed = int(seed)
        self._history.reset()

    def observe_action(
        self,
        before: SplendorState,
        action: Action,
        after: SplendorState,
    ) -> None:
        del after
        self._history.record(before, action)

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        row = build_recurrent_policy_input(state, actions, self._history.events)
        batch = collate_recurrent_policy_inputs([row], self._device)
        with torch.no_grad():
            scores, _value = self._model(
                batch["observation"],
                batch["legal_action_features"],
                batch["legal_action_mask"],
                batch["history_action_type_ids"],
                batch["history_card_ids"],
                batch["history_noble_ids"],
                batch["history_numeric_features"],
                batch["history_mask"],
            )
            chosen = int(torch.argmax(scores[0, : len(actions)]).item())
        return actions[chosen]
