"""Inference-time action-class mask for `RESERVE_DECK`.

Experiment `071` tests one action class on the frozen champion without touching
weights, encoders, or features.  The bot scores the complete legal action set
exactly as its parent does and then takes the greedy choice over the actions
whose type is not :attr:`ActionType.RESERVE_DECK`.

The unmasked greedy choice is computed from the same forward pass, so the bot
also reports the *dose*: the fraction of decisions on which the mask actually
changed the action.  `071` treats that fraction as a primary reported quantity,
because an intervention that binds on a small share of decisions cannot be
distinguished from a null by an ordinary strength panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from splendor_ai.encoding import (
    encode_legal_action_features,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import masked_legal_action_scores

from .model_bot import LegalActionAttentionScorerBot


MASKED_ACTION_TYPE = ActionType.RESERVE_DECK


@dataclass(slots=True)
class DeckReserveMaskTelemetry:
    """Dose accounting for one measurement window."""

    decisions: int = 0
    deck_reserve_legal: int = 0
    unmasked_choice_was_deck_reserve: int = 0
    switched: int = 0
    deck_only_fallback: int = 0
    switched_replacement_types: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.decisions = 0
        self.deck_reserve_legal = 0
        self.unmasked_choice_was_deck_reserve = 0
        self.switched = 0
        self.deck_only_fallback = 0
        self.switched_replacement_types = {}

    @property
    def switch_fraction(self) -> float:
        return self.switched / self.decisions if self.decisions else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "decisions": self.decisions,
            "deck_reserve_legal": self.deck_reserve_legal,
            "unmasked_choice_was_deck_reserve": self.unmasked_choice_was_deck_reserve,
            "switched": self.switched,
            "deck_only_fallback": self.deck_only_fallback,
            "switch_fraction": self.switch_fraction,
            "switched_replacement_types": dict(self.switched_replacement_types),
        }

    def merge(self, other: dict[str, object]) -> None:
        self.decisions += int(other["decisions"])
        self.deck_reserve_legal += int(other["deck_reserve_legal"])
        self.unmasked_choice_was_deck_reserve += int(
            other["unmasked_choice_was_deck_reserve"]
        )
        self.switched += int(other["switched"])
        self.deck_only_fallback += int(other["deck_only_fallback"])
        for key, count in dict(other["switched_replacement_types"]).items():
            self.switched_replacement_types[key] = (
                self.switched_replacement_types.get(key, 0) + int(count)
            )


class DeckReserveMaskedAttentionScorerBot(LegalActionAttentionScorerBot):
    """Champion policy with `RESERVE_DECK` removed from the candidate set."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        super().__init__(checkpoint_path, device=device)
        self.telemetry = DeckReserveMaskTelemetry()

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        observation = encode_public_observation_tensor(state, state.current_player)
        observation_vector = fit_feature_vector(
            observation.vector,
            self._model.config.observation_size,
        )
        action_features = encode_legal_action_features(
            state=state,
            legal_actions=actions,
            player_id=state.current_player,
        )
        action_feature_vectors = [
            fit_feature_vector(features.vector, self._model.config.action_feature_size)
            for features in action_features
        ]
        observation_tensor = torch.tensor(
            observation_vector,
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)
        action_feature_tensor = torch.tensor(
            action_feature_vectors,
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)
        legal_action_mask = torch.ones(
            (1, len(actions)),
            dtype=torch.bool,
            device=self._device,
        )

        with torch.no_grad():
            action_scores, _value = self._model(
                observation_tensor,
                action_feature_tensor,
                legal_action_mask,
            )
            masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)
            unmasked_row = int(torch.argmax(masked_scores, dim=1).item())
            allowed = [
                index
                for index, action in enumerate(actions)
                if action.action_type is not MASKED_ACTION_TYPE
            ]
            if allowed:
                allowed_tensor = torch.tensor(
                    allowed,
                    dtype=torch.long,
                    device=self._device,
                )
                best_row = int(
                    allowed_tensor[
                        int(torch.argmax(masked_scores[0, allowed_tensor]).item())
                    ].item()
                )
            else:
                best_row = unmasked_row

        telemetry = self.telemetry
        telemetry.decisions += 1
        if len(allowed) != len(actions):
            telemetry.deck_reserve_legal += 1
        if not allowed:
            telemetry.deck_only_fallback += 1
        if actions[unmasked_row].action_type is MASKED_ACTION_TYPE:
            telemetry.unmasked_choice_was_deck_reserve += 1
        if best_row != unmasked_row:
            telemetry.switched += 1
            key = actions[best_row].action_type.name
            telemetry.switched_replacement_types[key] = (
                telemetry.switched_replacement_types.get(key, 0) + 1
            )

        return actions[best_row]
