"""Policy/value checkpoint bot with legal-action masking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import ActionCodec, encode_public_observation_tensor
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import PolicyValueMLP, PolicyValueModelConfig, masked_policy_logits


@dataclass(frozen=True, slots=True)
class LoopFallbackConfig:
    """Conservative inference-only escape hatch for repeated checkpoint loops."""

    enabled: bool = True
    min_state_visits: int = 2
    min_own_non_progress_actions: int = 6
    max_buy_logit_gap: float | None = 8.0


class CheckpointPolicyBot:
    """Greedy policy bot backed by a saved supervised checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        loop_fallback: LoopFallbackConfig | None = None,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._loop_fallback = loop_fallback or LoopFallbackConfig()
        checkpoint = torch.load(self._checkpoint_path, map_location=device, weights_only=True)
        config_payload = checkpoint.get("model_config", {})
        self._model = PolicyValueMLP(PolicyValueModelConfig(**config_payload))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()
        self._codec = ActionCodec()
        self._state_visit_counts: dict[tuple[object, ...], int] = {}
        self._own_non_progress_actions = 0
        self._last_turn_index: int | None = None
        self._loop_fallback_triggers = 0

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def loop_fallback_triggers(self) -> int:
        return self._loop_fallback_triggers

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        self._reset_tracking_if_new_game(state)
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        signature = state_signature(state)
        state_visit_count = self._state_visit_counts.get(signature, 0) + 1
        self._state_visit_counts[signature] = state_visit_count

        observation = encode_public_observation_tensor(state, state.current_player)
        observation_tensor = torch.tensor(
            observation.vector,
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)
        mask = torch.zeros(
            (1, self._codec.action_space_size),
            dtype=torch.bool,
            device=self._device,
        )
        action_to_index = {}
        for action in actions:
            index = self._codec.encode(state, action)
            mask[0, index] = True
            action_to_index[action] = index

        with torch.no_grad():
            policy_logits, _value = self._model(observation_tensor)
            masked_logits = masked_policy_logits(policy_logits, mask)
            best_index = int(torch.argmax(masked_logits, dim=1).item())

        chosen_action = self._action_for_index(action_to_index, best_index)
        fallback_action = self._loop_fallback_action(
            actions=actions,
            action_to_index=action_to_index,
            masked_logits=masked_logits,
            best_index=best_index,
            state_visit_count=state_visit_count,
        )
        if fallback_action is not None and fallback_action != chosen_action:
            chosen_action = fallback_action
            self._loop_fallback_triggers += 1

        self._record_selected_action(env, state, chosen_action)
        return chosen_action

    def _action_for_index(self, action_to_index: dict[Action, int], target_index: int) -> Action:
        for action, index in action_to_index.items():
            if index == target_index:
                return action
        raise RuntimeError("Best legal action index did not map back to a legal action.")

    def _loop_fallback_action(
        self,
        actions: list[Action],
        action_to_index: dict[Action, int],
        masked_logits: torch.Tensor,
        best_index: int,
        state_visit_count: int,
    ) -> Action | None:
        config = self._loop_fallback
        if not config.enabled:
            return None

        repeated_state = state_visit_count >= config.min_state_visits
        own_no_progress = self._own_non_progress_actions >= config.min_own_non_progress_actions
        if not repeated_state and not own_no_progress:
            return None

        buy_actions = [
            action
            for action in actions
            if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
        ]
        if not buy_actions:
            return None

        buy_action = max(
            buy_actions,
            key=lambda action: float(masked_logits[0, action_to_index[action]].item()),
        )
        if action_to_index[buy_action] == best_index:
            return None

        if config.max_buy_logit_gap is not None:
            best_logit = float(masked_logits[0, best_index].item())
            buy_logit = float(masked_logits[0, action_to_index[buy_action]].item())
            if best_logit - buy_logit > config.max_buy_logit_gap:
                return None

        return buy_action

    def _record_selected_action(self, env: SplendorEnv, state: SplendorState, action: Action) -> None:
        next_state = env.step(state, action)
        if is_progress_transition(state, next_state, action):
            self._own_non_progress_actions = 0
        else:
            self._own_non_progress_actions += 1
        self._last_turn_index = state.turn_index

    def _reset_tracking_if_new_game(self, state: SplendorState) -> None:
        if self._last_turn_index is None:
            return
        if state.turn_index < self._last_turn_index or (
            state.turn_index == 0 and self._last_turn_index != 0
        ):
            self._state_visit_counts.clear()
            self._own_non_progress_actions = 0
            self._loop_fallback_triggers = 0
