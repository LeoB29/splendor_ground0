"""Policy/value checkpoint bot with legal-action masking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import (
    ActionCodec,
    actions_by_canonical_successor_key,
    canonical_public_belief_key,
    collate_entity_action_batch,
    encode_legal_action_features,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    LegalActionScorerConfig,
    LegalActionScorerModel,
    LegalActionSetScorerConfig,
    LegalActionSetScorerModel,
    PolicyValueMLP,
    PolicyValueModelConfig,
    masked_legal_action_scores,
    masked_policy_logits,
)


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

    @property
    def loop_fallback_config(self) -> LoopFallbackConfig:
        return self._loop_fallback

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
        observation_vector = fit_feature_vector(
            observation.vector,
            self._model.config.observation_size,
        )
        observation_tensor = torch.tensor(
            observation_vector,
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


class LegalActionScorerBot:
    """Greedy policy bot backed by a legal-action scorer checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        checkpoint = torch.load(self._checkpoint_path, map_location=device, weights_only=True)
        config_payload = checkpoint.get("model_config", {})
        self._model = LegalActionScorerModel(LegalActionScorerConfig(**config_payload))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

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
            action_scores, _value = self._model(observation_tensor, action_feature_tensor)
            masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)
            best_row = int(torch.argmax(masked_scores, dim=1).item())

        return actions[best_row]


class CanonicalEntityActionScorerBot:
    """Greedy legal policy backed by a canonical entity/action checkpoint."""

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
        self._model = CanonicalEntityActionScorerModel(
            CanonicalEntityActionScorerConfig(**checkpoint.get("model_config", {}))
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()
        self._state_visit_counts: dict[tuple[object, ...], int] = {}
        self._successor_choice_counts: dict[
            tuple[object, ...], dict[tuple[object, ...], int]
        ] = {}
        self._own_non_progress_actions = 0
        self._last_turn_index: int | None = None
        self._loop_fallback_triggers = 0

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def loop_fallback_triggers(self) -> int:
        return self._loop_fallback_triggers

    @property
    def loop_fallback_config(self) -> LoopFallbackConfig:
        return self._loop_fallback

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        return self.choose_action_from_scores(
            env,
            state,
            self.score_actions(state, actions),
        )

    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> dict[Action, float]:
        """Score one representative for each canonical public successor."""

        if not legal_actions:
            return {}
        representatives = actions_by_canonical_successor_key(state, legal_actions)
        representative_actions = list(representatives.values())
        batch = collate_entity_action_batch(
            [(state, representative_actions)],
            device=self._device,
        )
        with torch.no_grad():
            scores, _value = self._model(**batch)
            masked_scores = masked_legal_action_scores(scores, batch["action_mask"])
        return {
            action: float(masked_scores[0, row].item())
            for row, action in enumerate(representative_actions)
        }

    def choose_action_from_scores(
        self,
        env: SplendorEnv,
        state: SplendorState,
        action_scores: dict[Action, float],
    ) -> Action | None:
        """Choose from already-computed scores while preserving loop safeguards."""

        if not action_scores:
            return None
        self._reset_tracking_if_new_game(state)

        signature = self._loop_state_signature(state)
        state_visit_count = self._state_visit_counts.get(signature, 0) + 1
        self._state_visit_counts[signature] = state_visit_count

        representatives = actions_by_canonical_successor_key(state, list(action_scores))
        representative_actions = list(representatives.values())
        masked_scores = torch.tensor(
            [[action_scores[action] for action in representative_actions]],
            dtype=torch.float32,
            device=self._device,
        )
        best_row = int(torch.argmax(masked_scores, dim=1).item())

        chosen_action = representative_actions[best_row]
        fallback_action = self._loop_fallback_action(
            successor_actions=representatives,
            representative_actions=representative_actions,
            masked_scores=masked_scores,
            best_row=best_row,
            state_visit_count=state_visit_count,
            state_signature=signature,
        )
        if fallback_action is not None and fallback_action != chosen_action:
            chosen_action = fallback_action
            self._loop_fallback_triggers += 1

        selected_successor_key = next(
            key for key, action in representatives.items() if action == chosen_action
        )
        successor_counts = self._successor_choice_counts.setdefault(signature, {})
        successor_counts[selected_successor_key] = successor_counts.get(selected_successor_key, 0) + 1
        self._record_selected_action(env, state, chosen_action)
        return chosen_action

    def record_external_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        action: Action,
    ) -> None:
        """Keep loop-tracking state coherent when a wrapper selects an action."""

        self._reset_tracking_if_new_game(state)
        self._record_selected_action(env, state, action)

    def _loop_fallback_action(
        self,
        successor_actions: dict[tuple[object, ...], Action],
        representative_actions: list[Action],
        masked_scores: torch.Tensor,
        best_row: int,
        state_visit_count: int,
        state_signature: tuple[object, ...],
    ) -> Action | None:
        """Choose at most one action per canonical successor when escaping a loop."""

        config = self._loop_fallback
        if not config.enabled:
            return None

        repeated_state = state_visit_count >= config.min_state_visits
        own_no_progress = self._own_non_progress_actions >= config.min_own_non_progress_actions
        if not repeated_state and not own_no_progress:
            return None

        row_by_action = {
            action: row for row, action in enumerate(representative_actions)
        }
        buy_actions = [
            action
            for action in successor_actions.values()
            if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
        ]
        if buy_actions:
            buy_action = max(
                buy_actions,
                key=lambda action: float(masked_scores[0, row_by_action[action]].item()),
            )
            buy_row = row_by_action[buy_action]
            close_enough = True
            if config.max_buy_logit_gap is not None:
                best_score = float(masked_scores[0, best_row].item())
                buy_score = float(masked_scores[0, buy_row].item())
                close_enough = best_score - buy_score <= config.max_buy_logit_gap
            if buy_row != best_row and close_enough:
                return buy_action

        if not repeated_state:
            return None

        successor_counts = self._successor_choice_counts.get(state_signature, {})
        keys = list(successor_actions)
        least_uses = min(successor_counts.get(key, 0) for key in keys)
        least_used_keys = [
            key for key in keys if successor_counts.get(key, 0) == least_uses
        ]
        fallback_key = max(
            least_used_keys,
            key=lambda key: float(
                masked_scores[0, row_by_action[successor_actions[key]]].item()
            ),
        )
        fallback_action = successor_actions[fallback_key]
        return fallback_action if row_by_action[fallback_action] != best_row else None

    @staticmethod
    def _loop_state_signature(state: SplendorState) -> tuple[object, ...]:
        """Canonical public state key with absolute time removed for cycle detection."""

        key = canonical_public_belief_key(state, state.current_player)
        return key[:3] + (state.turn_index % 2,) + key[4:]

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
            self._successor_choice_counts.clear()
            self._own_non_progress_actions = 0
            self._loop_fallback_triggers = 0


class LegalActionSetScorerBot:
    """Greedy policy bot backed by an action-set-aware legal-action scorer checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        checkpoint = torch.load(self._checkpoint_path, map_location=device, weights_only=True)
        config_payload = checkpoint.get("model_config", {})
        self._model = LegalActionSetScorerModel(LegalActionSetScorerConfig(**config_payload))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

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
            best_row = int(torch.argmax(masked_scores, dim=1).item())

        return actions[best_row]


class LegalActionAttentionScorerBot:
    """Greedy policy bot backed by an attention-based legal-action scorer checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        checkpoint = torch.load(self._checkpoint_path, map_location=device, weights_only=True)
        config_payload = checkpoint.get("model_config", {})
        self._model = LegalActionAttentionScorerModel(
            LegalActionAttentionScorerConfig(**config_payload)
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

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
            best_row = int(torch.argmax(masked_scores, dim=1).item())

        return actions[best_row]

    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> tuple[float, ...]:
        """Return legal-action scores in the supplied order without choosing."""

        if not legal_actions:
            return ()
        observation = encode_public_observation_tensor(state, state.current_player)
        observation_vector = fit_feature_vector(
            observation.vector,
            self._model.config.observation_size,
        )
        action_features = encode_legal_action_features(
            state=state,
            legal_actions=legal_actions,
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
            (1, len(legal_actions)),
            dtype=torch.bool,
            device=self._device,
        )
        with torch.no_grad():
            action_scores, _value = self._model(
                observation_tensor,
                action_feature_tensor,
                legal_action_mask,
            )
            masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)[0]
        return tuple(float(value) for value in masked_scores.detach().cpu().tolist())


class LegalActionAttentionSamplingBot(LegalActionAttentionScorerBot):
    """Top-k epsilon sampler backed by an attention legal-action scorer."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        seed: int = 0,
        epsilon: float = 0.2,
        temperature: float = 0.5,
        top_k: int = 3,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._sampling_seed = seed
        self._sampling_epsilon = epsilon
        self._sampling_temperature = temperature
        self._sampling_top_k = top_k
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._last_action_metadata: dict[str, object] | None = None

    @property
    def sampling_config(self) -> dict[str, object]:
        return {
            "seed": self._sampling_seed,
            "epsilon": self._sampling_epsilon,
            "temperature": self._sampling_temperature,
            "top_k": self._sampling_top_k,
        }

    @property
    def last_action_metadata(self) -> dict[str, object] | None:
        return self._last_action_metadata

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            self._last_action_metadata = None
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
            scores = masked_legal_action_scores(action_scores, legal_action_mask)[0]
            support_size = min(self._sampling_top_k, len(actions))
            support_rows = torch.topk(scores, k=support_size).indices
            support_probs = torch.softmax(
                scores[support_rows] / self._sampling_temperature,
                dim=0,
            )
            behavior_probs = torch.zeros_like(scores)
            behavior_probs[support_rows] = self._sampling_epsilon * support_probs
            greedy_row = int(torch.argmax(scores).item())
            behavior_probs[greedy_row] += 1.0 - self._sampling_epsilon
            chosen_row = int(
                torch.multinomial(
                    behavior_probs.detach().cpu(),
                    num_samples=1,
                    generator=self._generator,
                ).item()
            )

        probability = float(behavior_probs[chosen_row].item())
        self._last_action_metadata = {
            "behavior_probability": probability,
            "behavior_log_probability": math.log(probability),
            "behavior_greedy_action": chosen_row == greedy_row,
            "behavior_support_size": support_size,
        }
        return actions[chosen_row]
