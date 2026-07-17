"""Shallow adversarial search bot for stronger corpus generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from splendor_ai.encoding import (
    collate_canonical_state_batch,
    collate_entity_action_batch,
    encode_legal_action_features,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    CanonicalEntityValueConfig,
    CanonicalEntityDistributionalValueModel,
    CanonicalEntityValueModel,
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    LegalActionScorerConfig,
    LegalActionScorerModel,
    LegalActionSetScorerConfig,
    LegalActionSetScorerModel,
    ValueMLP,
    ValueModelConfig,
    masked_legal_action_scores,
)

from .heuristic_bot import GreedyHeuristicBot
from .hidden_information import determinize_for_observer


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    action: Action
    next_state: SplendorState
    heuristic_score: float
    loop_penalty: float
    strategic_signature: tuple[object, ...]
    policy_score: float = 0.0
    ordering_score: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchActionScore:
    action: Action
    value: float
    heuristic_score: float
    loop_penalty: float
    policy_score: float = 0.0


class StateValueEvaluator(Protocol):
    def evaluate_state(self, state: SplendorState, player_id: int) -> float:
        """Return a heuristic-scale value from player_id's perspective."""


class ActionPriorScorer(Protocol):
    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> dict[Action, float]:
        """Return model-side preference scores for legal actions."""


class CheckpointValueEvaluator:
    """Loads a value checkpoint and converts normalized predictions to heuristic scale."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        value_scale: float = 15_000.0,
        prediction_mode: str = "value",
        outcome_blend_weight: float = 0.5,
    ) -> None:
        if prediction_mode not in {"value", "outcome", "blend"}:
            raise ValueError("prediction_mode must be 'value', 'outcome', or 'blend'.")
        if not 0.0 <= outcome_blend_weight <= 1.0:
            raise ValueError("outcome_blend_weight must be in the range [0.0, 1.0].")
        self.checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._value_scale = value_scale
        self._prediction_mode = prediction_mode
        self._outcome_blend_weight = outcome_blend_weight
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        model_kind = checkpoint.get("model_kind")
        self._canonical_distributional = (
            model_kind == "canonical_entity_distributional_value"
        )
        self._canonical_entity = model_kind == "canonical_entity_value"
        if self._canonical_distributional:
            self._model = CanonicalEntityDistributionalValueModel(
                CanonicalEntityValueConfig(**checkpoint.get("model_config", {})),
                margin_bins=int(checkpoint.get("margin_bins", 41)),
            )
            self._checkpoint_blend_weight = float(
                checkpoint.get("search_wdl_blend_weight", outcome_blend_weight)
            )
        elif self._canonical_entity:
            if prediction_mode != "value":
                raise ValueError(
                    "canonical_entity_value checkpoints support prediction_mode='value' only."
                )
            self._model = CanonicalEntityValueModel(
                CanonicalEntityValueConfig(**checkpoint.get("model_config", {}))
            )
        else:
            self._model = ValueMLP(ValueModelConfig(**checkpoint.get("model_config", {})))
        self._model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self._model.to(device)
        self._model.eval()

    def evaluate_state(self, state: SplendorState, player_id: int) -> float:
        return self.evaluate_states([(state, player_id)])[0]

    def evaluate_states(
        self,
        states_and_players: list[tuple[SplendorState, int]],
    ) -> list[float]:
        if not states_and_players:
            return []
        if self._canonical_distributional:
            batch = collate_canonical_state_batch(
                states_and_players,
                device=self._device,
            )
            with torch.no_grad():
                margin_logits, wdl_logits = self._model(**batch)
                margin_values, wdl_values = self._model.expected_values(
                    margin_logits, wdl_logits
                )
                normalized_values = (
                    (1.0 - self._checkpoint_blend_weight) * margin_values
                    + self._checkpoint_blend_weight * wdl_values
                )
            return [
                float(value) * self._value_scale
                for value in normalized_values.detach().cpu().tolist()
            ]
        if self._canonical_entity:
            batch = collate_canonical_state_batch(
                states_and_players,
                device=self._device,
            )
            with torch.no_grad():
                normalized_values = self._model(**batch)
            return [
                float(value) * self._value_scale
                for value in normalized_values.detach().cpu().tolist()
            ]
        observation_vectors = [
            fit_feature_vector(
                encode_public_observation_tensor(state, player_id).vector,
                self._model.config.observation_size,
            )
            for state, player_id in states_and_players
        ]
        observation_tensor = torch.tensor(
            observation_vectors,
            dtype=torch.float32,
            device=self._device,
        )
        with torch.no_grad():
            values = self._model(observation_tensor)
            if self._prediction_mode == "value":
                normalized_values = values
            else:
                logits = self._model.outcome_logits(observation_tensor)
                probabilities = torch.softmax(logits, dim=1)
                outcome_values = probabilities[:, 2] - probabilities[:, 0]
                if self._prediction_mode == "outcome":
                    normalized_values = outcome_values
                else:
                    normalized_values = (
                        (1.0 - self._outcome_blend_weight) * values
                        + self._outcome_blend_weight * outcome_values
                    )
        return [
            float(value) * self._value_scale
            for value in normalized_values.detach().cpu().tolist()
        ]


class CheckpointLegalActionPriorScorer:
    """Loads a legal-action scorer checkpoint for action-ordering priors."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._device = device
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
        model_kind = checkpoint.get("model_kind", "legal_action_scorer")
        if model_kind == "legal_action_attention_scorer":
            self._model = LegalActionAttentionScorerModel(
                LegalActionAttentionScorerConfig(**checkpoint.get("model_config", {}))
            )
        elif model_kind == "legal_action_set_scorer":
            self._model = LegalActionSetScorerModel(
                LegalActionSetScorerConfig(**checkpoint.get("model_config", {}))
            )
        elif model_kind == "canonical_entity_action_scorer":
            self._model = CanonicalEntityActionScorerModel(
                CanonicalEntityActionScorerConfig(**checkpoint.get("model_config", {}))
            )
        else:
            self._model = LegalActionScorerModel(
                LegalActionScorerConfig(**checkpoint.get("model_config", {}))
            )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()

    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> dict[Action, float]:
        if not legal_actions:
            return {}
        return self.score_action_batches([(state, legal_actions)])[0]

    def score_action_batches(
        self,
        state_action_batches: list[tuple[SplendorState, list[Action]]],
    ) -> list[dict[Action, float]]:
        if not state_action_batches:
            return []
        if any(not legal_actions for _state, legal_actions in state_action_batches):
            raise ValueError("Each state in an action batch must have legal actions.")
        if isinstance(self._model, CanonicalEntityActionScorerModel):
            batch = collate_entity_action_batch(
                state_action_batches,
                device=self._device,
            )
            with torch.no_grad():
                action_scores, _value = self._model(**batch)
                masked_scores = masked_legal_action_scores(
                    action_scores,
                    batch["action_mask"],
                ).detach().cpu()
            return [
                {
                    action: float(masked_scores[batch_index, row_index].item())
                    for row_index, action in enumerate(legal_actions)
                }
                for batch_index, (_state, legal_actions) in enumerate(
                    state_action_batches
                )
            ]
        observation_vectors = []
        action_vectors_by_state = []
        for state, legal_actions in state_action_batches:
            observation = encode_public_observation_tensor(
                state,
                state.current_player,
            )
            observation_vectors.append(
                fit_feature_vector(
                    observation.vector,
                    self._model.config.observation_size,
                )
            )
            action_features = encode_legal_action_features(
                state=state,
                legal_actions=legal_actions,
                player_id=state.current_player,
            )
            action_vectors_by_state.append(
                [
                    fit_feature_vector(
                        features.vector,
                        self._model.config.action_feature_size,
                    )
                    for features in action_features
                ]
            )
        max_actions = max(len(vectors) for vectors in action_vectors_by_state)
        padded_action_vectors = []
        masks = []
        for action_vectors in action_vectors_by_state:
            padding_count = max_actions - len(action_vectors)
            padded_action_vectors.append(
                action_vectors
                + [
                    [0.0] * self._model.config.action_feature_size
                    for _ in range(padding_count)
                ]
            )
            masks.append(
                [True] * len(action_vectors) + [False] * padding_count
            )
        observation_tensor = torch.tensor(
            observation_vectors,
            dtype=torch.float32,
            device=self._device,
        )
        action_feature_tensor = torch.tensor(
            padded_action_vectors,
            dtype=torch.float32,
            device=self._device,
        )
        legal_action_mask = torch.tensor(
            masks,
            dtype=torch.bool,
            device=self._device,
        )
        with torch.no_grad():
            action_scores, _value = self._model(
                observation_tensor,
                action_feature_tensor,
                legal_action_mask,
            )
            masked_scores = masked_legal_action_scores(
                action_scores,
                legal_action_mask,
            ).detach().cpu()
        return [
            {
                action: float(masked_scores[batch_index, row_index].item())
                for row_index, action in enumerate(legal_actions)
            }
            for batch_index, (_state, legal_actions) in enumerate(
                state_action_batches
            )
        ]


class ShallowSearchBot:
    """Depth-limited minimax bot with heuristic leaf evaluation.

    This bot is intentionally modest:
    - deterministic exact lookahead using the current engine
    - heuristic leaf evaluation reused from `GreedyHeuristicBot`
    - action ordering plus root/node pruning for tractability
    """

    def __init__(
        self,
        depth: int = 2,
        max_branching: int = 10,
        max_buy_actions: int = 6,
        max_reserve_actions: int = 3,
        max_take_actions: int = 3,
        max_pass_actions: int = 1,
        repetition_penalty: float = 4_000.0,
        game_history_penalty: float = 3_000.0,
        no_progress_take_penalty: float = 900.0,
        return_token_penalty: float = 120.0,
        repeated_color_return_penalty: float = 250.0,
        seed: int | None = None,
        value_evaluator: StateValueEvaluator | None = None,
        value_weight: float = 0.0,
        value_min_turn_index: int | None = None,
        value_max_turn_index: int | None = None,
        action_prior: ActionPriorScorer | None = None,
        action_prior_weight: float = 0.0,
        information_legal: bool = False,
        determinizations: int = 1,
        determinization_seed: int = 0,
    ) -> None:
        if depth < 1:
            raise ValueError("Search depth must be at least 1.")
        if max_branching < 1:
            raise ValueError("max_branching must be at least 1.")
        if min(max_buy_actions, max_reserve_actions, max_take_actions, max_pass_actions) < 0:
            raise ValueError("Per-action-type branching caps must be non-negative.")
        if not 0.0 <= value_weight <= 1.0:
            raise ValueError("value_weight must be in the range [0.0, 1.0].")
        if value_min_turn_index is not None and value_min_turn_index < 0:
            raise ValueError("value_min_turn_index must be non-negative.")
        if value_max_turn_index is not None and value_max_turn_index < 0:
            raise ValueError("value_max_turn_index must be non-negative.")
        if (
            value_min_turn_index is not None
            and value_max_turn_index is not None
            and value_min_turn_index > value_max_turn_index
        ):
            raise ValueError("value_min_turn_index cannot exceed value_max_turn_index.")
        if action_prior_weight < 0.0:
            raise ValueError("action_prior_weight must be non-negative.")
        if determinizations <= 0:
            raise ValueError("determinizations must be positive.")
        self._depth = depth
        self._max_branching = max_branching
        self._max_buy_actions = max_buy_actions
        self._max_reserve_actions = max_reserve_actions
        self._max_take_actions = max_take_actions
        self._max_pass_actions = max_pass_actions
        self._repetition_penalty = repetition_penalty
        self._game_history_penalty = game_history_penalty
        self._no_progress_take_penalty = no_progress_take_penalty
        self._return_token_penalty = return_token_penalty
        self._repeated_color_return_penalty = repeated_color_return_penalty
        self._rng = random.Random(seed)
        self._heuristic = GreedyHeuristicBot(seed=seed)
        self._value_evaluator = value_evaluator
        self._value_weight = value_weight
        self._value_min_turn_index = value_min_turn_index
        self._value_max_turn_index = value_max_turn_index
        self._action_prior = action_prior
        self._action_prior_weight = action_prior_weight
        self._information_legal = information_legal
        self._determinizations = determinizations
        self._determinization_seed = determinization_seed
        self._transposition_cache: dict[tuple[int, int, tuple[object, ...]], float] = {}
        self._observed_state_counts: dict[tuple[object, ...], int] = {}
        self._post_action_state_counts: dict[tuple[object, ...], int] = {}
        self._last_turn_index: int | None = None

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        self._reset_game_tracking_if_needed(state)
        root_player = state.current_player
        if self._information_legal:
            return self._choose_determinized_action(
                env=env,
                state=state,
                actions=actions,
                root_player=root_player,
            )
        self._transposition_cache = {}
        root_signature = self._heuristic.state_signature(state)
        self._observed_state_counts[root_signature] = self._observed_state_counts.get(root_signature, 0) + 1
        candidates = self._ordered_candidates(
            env=env,
            state=state,
            actions=actions,
            root_player=root_player,
        )

        best_value = -math.inf
        best_actions: list[Action] = []
        alpha = -math.inf
        beta = math.inf

        best_candidates: list[_RankedCandidate] = []
        for candidate in candidates:
            value = self._minimax(
                env=env,
                state=candidate.next_state,
                depth=self._depth - 1,
                root_player=root_player,
                alpha=alpha,
                beta=beta,
                path_signatures={root_signature},
            )
            value -= candidate.loop_penalty
            if value > best_value:
                best_value = value
                best_actions = [candidate.action]
                best_candidates = [candidate]
            elif value == best_value:
                best_actions.append(candidate.action)
                best_candidates.append(candidate)
            alpha = max(alpha, best_value)

        if not best_candidates:
            chosen_candidate = candidates[0]
        else:
            chosen_candidate = self._rng.choice(best_candidates)
        next_signature = self._heuristic.state_signature(chosen_candidate.next_state)
        self._post_action_state_counts[next_signature] = self._post_action_state_counts.get(next_signature, 0) + 1
        self._last_turn_index = state.turn_index
        return chosen_candidate.action

    def _choose_determinized_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        actions: list[Action],
        root_player: int,
    ) -> Action:
        values_by_action: dict[Action, list[float]] = {
            action: [] for action in actions
        }
        for sample_index in range(self._determinizations):
            determinized = determinize_for_observer(
                state,
                observer_id=root_player,
                rng=random.Random(
                    self._determinization_seed
                    + state.turn_index * 1_000_003
                    + sample_index
                ),
            )
            for score in self.rank_actions(
                env,
                determinized,
                actions,
            ):
                values_by_action[score.action].append(score.value)
        complete_values = {
            action: sum(values) / len(values)
            for action, values in values_by_action.items()
            if len(values) == self._determinizations
        }
        if not complete_values:
            maximum_coverage = max(len(values) for values in values_by_action.values())
            complete_values = {
                action: sum(values) / len(values)
                for action, values in values_by_action.items()
                if len(values) == maximum_coverage and values
            }
        best_value = max(complete_values.values())
        best_actions = [
            action
            for action, value in complete_values.items()
            if value == best_value
        ]
        chosen_action = self._rng.choice(best_actions)
        root_signature = self._heuristic.state_signature(state)
        self._observed_state_counts[root_signature] = (
            self._observed_state_counts.get(root_signature, 0) + 1
        )
        next_state = env.step(state, chosen_action)
        next_signature = self._heuristic.state_signature(next_state)
        self._post_action_state_counts[next_signature] = (
            self._post_action_state_counts.get(next_signature, 0) + 1
        )
        self._last_turn_index = state.turn_index
        return chosen_action

    def rank_actions(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> tuple[SearchActionScore, ...]:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return ()

        root_player = state.current_player
        self._transposition_cache = {}
        root_signature = self._heuristic.state_signature(state)
        candidates = self._ordered_candidates(
            env=env,
            state=state,
            actions=actions,
            root_player=root_player,
        )
        scores = []
        for candidate in candidates:
            value = self._minimax(
                env=env,
                state=candidate.next_state,
                depth=self._depth - 1,
                root_player=root_player,
                alpha=-math.inf,
                beta=math.inf,
                path_signatures={root_signature},
            )
            value -= candidate.loop_penalty
            scores.append(
                SearchActionScore(
                    action=candidate.action,
                    value=value,
                    heuristic_score=candidate.heuristic_score,
                    loop_penalty=candidate.loop_penalty,
                    policy_score=candidate.policy_score,
                )
            )

        return tuple(
            sorted(
                scores,
                key=lambda item: (item.value, item.policy_score, item.heuristic_score),
                reverse=True,
            )
        )

    def _minimax(
        self,
        env: SplendorEnv,
        state: SplendorState,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
        path_signatures: set[tuple[object, ...]],
    ) -> float:
        state_signature = self._heuristic.state_signature(state)
        if state_signature in path_signatures:
            return self._evaluate_leaf(state, root_player) - self._repetition_penalty

        if depth <= 0 or state.terminal:
            return self._evaluate_leaf(state, root_player)

        cache_key = (depth, root_player, state_signature)
        cached = self._transposition_cache.get(cache_key)
        if cached is not None:
            return cached

        legal_actions = env.legal_actions(state)
        if not legal_actions:
            return self._evaluate_leaf(state, root_player)

        maximizing = state.current_player == root_player
        candidates = self._ordered_candidates(
            env=env,
            state=state,
            actions=legal_actions,
            root_player=root_player,
        )

        path_signatures.add(state_signature)
        if maximizing:
            value = -math.inf
            for candidate in candidates:
                value = max(
                    value,
                    self._minimax(
                        env=env,
                        state=candidate.next_state,
                        depth=depth - 1,
                        root_player=root_player,
                        alpha=alpha,
                        beta=beta,
                        path_signatures=path_signatures,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            path_signatures.remove(state_signature)
            self._transposition_cache[cache_key] = value
            return value

        value = math.inf
        for candidate in candidates:
            value = min(
                value,
                self._minimax(
                    env=env,
                    state=candidate.next_state,
                    depth=depth - 1,
                    root_player=root_player,
                    alpha=alpha,
                    beta=beta,
                    path_signatures=path_signatures,
                ),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break
        path_signatures.remove(state_signature)
        self._transposition_cache[cache_key] = value
        return value

    def _evaluate_leaf(self, state: SplendorState, root_player: int) -> float:
        heuristic_value = self._heuristic.evaluate_state(state, root_player)
        if self._value_evaluator is None or self._value_weight == 0.0 or state.terminal:
            return heuristic_value
        if self._value_min_turn_index is not None and state.turn_index < self._value_min_turn_index:
            return heuristic_value
        if self._value_max_turn_index is not None and state.turn_index > self._value_max_turn_index:
            return heuristic_value
        neural_value = self._value_evaluator.evaluate_state(state, root_player)
        return (1.0 - self._value_weight) * heuristic_value + self._value_weight * neural_value

    def _ordered_candidates(
        self,
        env: SplendorEnv,
        state: SplendorState,
        actions: list[Action],
        root_player: int,
    ) -> list[_RankedCandidate]:
        actor = state.current_player
        before_affordable = self._heuristic.affordable_card_count(state, root_player)
        before_best_missing = self._heuristic.best_purchase_missing(state, root_player)
        policy_scores = self._normalized_policy_scores(state, actions)
        candidates: list[_RankedCandidate] = []
        for action in actions:
            next_state = env.step(state, action)
            score = self._heuristic.score_transition(
                before=state,
                after=next_state,
                action=action,
                player_id=root_player,
                before_affordable=before_affordable,
                before_best_missing=before_best_missing,
            )
            loop_penalty = self._candidate_loop_penalty(
                before=state,
                after=next_state,
                action=action,
                player_id=root_player,
            )
            adjusted_score = score - loop_penalty
            policy_score = policy_scores.get(action, 0.0)
            prior_sign = 1.0 if actor == root_player else -1.0
            ordering_score = adjusted_score + prior_sign * self._action_prior_weight * policy_score
            candidates.append(
                _RankedCandidate(
                    action=action,
                    next_state=next_state,
                    heuristic_score=adjusted_score,
                    loop_penalty=loop_penalty,
                    strategic_signature=self._strategic_signature(next_state, actor, action),
                    policy_score=policy_score,
                    ordering_score=ordering_score,
                )
            )

        maximizing = state.current_player == root_player
        candidates.sort(
            key=lambda item: (item.ordering_score, item.heuristic_score, self._rng.random()),
            reverse=maximizing,
        )
        limited: list[_RankedCandidate] = []
        kept_by_type = {
            ActionType.BUY_VISIBLE: 0,
            ActionType.BUY_RESERVED: 0,
            ActionType.RESERVE_VISIBLE: 0,
            ActionType.RESERVE_DECK: 0,
            ActionType.TAKE_TOKENS: 0,
            ActionType.PASS: 0,
        }
        seen_signatures: set[tuple[object, ...]] = set()
        for candidate in candidates:
            action = candidate.action
            if len(limited) >= self._max_branching:
                break
            if candidate.strategic_signature in seen_signatures:
                continue
            if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
                if kept_by_type[ActionType.BUY_VISIBLE] + kept_by_type[ActionType.BUY_RESERVED] >= self._max_buy_actions:
                    continue
            elif action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK):
                if kept_by_type[ActionType.RESERVE_VISIBLE] + kept_by_type[ActionType.RESERVE_DECK] >= self._max_reserve_actions:
                    continue
            elif action.action_type == ActionType.TAKE_TOKENS:
                if kept_by_type[ActionType.TAKE_TOKENS] >= self._max_take_actions:
                    continue
            elif action.action_type == ActionType.PASS:
                if kept_by_type[ActionType.PASS] >= self._max_pass_actions:
                    continue

            limited.append(candidate)
            kept_by_type[action.action_type] += 1
            seen_signatures.add(candidate.strategic_signature)

        if not limited:
            limited = candidates[: self._max_branching]
        return limited

    def _normalized_policy_scores(
        self,
        state: SplendorState,
        actions: list[Action],
    ) -> dict[Action, float]:
        if self._action_prior is None or self._action_prior_weight == 0.0 or not actions:
            return {}
        raw_scores = self._action_prior.score_actions(state, actions)
        if not raw_scores:
            return {}
        ranked_actions = sorted(
            actions,
            key=lambda action: raw_scores.get(action, -math.inf),
            reverse=True,
        )
        if len(ranked_actions) == 1:
            return {ranked_actions[0]: 1.0}
        denominator = float(len(ranked_actions) - 1)
        return {
            action: 1.0 - (rank / denominator)
            for rank, action in enumerate(ranked_actions)
        }

    def _candidate_loop_penalty(
        self,
        before: SplendorState,
        after: SplendorState,
        action: Action,
        player_id: int,
    ) -> float:
        before_signature = self._heuristic.state_signature(before)
        after_signature = self._heuristic.state_signature(after)
        current_repeat_count = max(self._observed_state_counts.get(before_signature, 0) - 1, 0)
        after_repeat_count = self._post_action_state_counts.get(after_signature, 0)
        penalty = after_repeat_count * self._game_history_penalty

        if action.action_type != ActionType.TAKE_TOKENS:
            return penalty

        if action.return_tokens:
            penalty += self._return_token_penalty * len(action.return_tokens)
            repeated_colors = set(action.take_tokens) & set(action.return_tokens)
            penalty += self._repeated_color_return_penalty * len(repeated_colors)

        before_affordable = self._heuristic.affordable_card_count(before, player_id)
        after_affordable = self._heuristic.affordable_card_count(after, player_id)
        before_best_missing = self._heuristic.best_purchase_missing(before, player_id)
        after_best_missing = self._heuristic.best_purchase_missing(after, player_id)
        before_best_scoring_missing = self._heuristic.best_scoring_purchase_missing(before, player_id)
        after_best_scoring_missing = self._heuristic.best_scoring_purchase_missing(after, player_id)
        before_noble_gap = self._heuristic.best_noble_gap(before, player_id)
        after_noble_gap = self._heuristic.best_noble_gap(after, player_id)

        if not self._is_progress_transition(before, after, action):
            penalty += self._no_progress_take_penalty
            if current_repeat_count > 0:
                penalty += current_repeat_count * self._game_history_penalty
            if (
                after_affordable <= before_affordable
                and after_best_missing >= before_best_missing
                and after_best_scoring_missing >= before_best_scoring_missing
                and after_noble_gap >= before_noble_gap
            ):
                penalty += self._no_progress_take_penalty

        return penalty

    def _reset_game_tracking_if_needed(self, state: SplendorState) -> None:
        if state.turn_index == 0 or (self._last_turn_index is not None and state.turn_index <= self._last_turn_index):
            self._observed_state_counts = {}
            self._post_action_state_counts = {}
            self._last_turn_index = None

    def _is_progress_transition(
        self,
        before: SplendorState,
        after: SplendorState,
        action: Action,
    ) -> bool:
        current_before = before.players[before.current_player]
        current_after = after.players[before.current_player]
        opponent_before = before.players[1 - before.current_player]
        opponent_after = after.players[1 - before.current_player]

        if action.action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_DECK,
        ):
            return True
        if len(current_after.purchased_cards) != len(current_before.purchased_cards):
            return True
        if len(current_after.reserved_cards) != len(current_before.reserved_cards):
            return True
        if len(current_after.nobles) != len(current_before.nobles):
            return True
        if current_after.score != current_before.score:
            return True
        if current_after.bonuses != current_before.bonuses:
            return True
        if opponent_after.score != opponent_before.score:
            return True
        if opponent_after.bonuses != opponent_before.bonuses:
            return True
        return False

    def _strategic_signature(
        self,
        state: SplendorState,
        actor_id: int,
        action: Action,
    ) -> tuple[object, ...]:
        actor = state.players[actor_id]
        return (
            action.action_type,
            actor.score,
            len(actor.reserved_cards),
            len(actor.purchased_cards),
            len(actor.nobles),
            tuple(actor.tokens[color] for color in ALL_TOKEN_TYPES),
            tuple(actor.bonuses[color] for color in TOKEN_COLORS),
            tuple(state.bank_tokens[color] for color in ALL_TOKEN_TYPES),
            self._heuristic.affordable_card_count(state, actor_id),
            self._heuristic.best_purchase_missing(state, actor_id),
            self._heuristic.best_scoring_purchase_missing(state, actor_id),
            self._heuristic.best_noble_gap(state, actor_id),
        )


class PolicyGuidedSearchBot(ShallowSearchBot):
    """Shallow search with a legal-action scorer prior for action ordering."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        policy_prior_weight: float = 2_000.0,
        **search_kwargs: object,
    ) -> None:
        prior = CheckpointLegalActionPriorScorer(checkpoint_path=checkpoint_path, device=device)
        super().__init__(
            action_prior=prior,
            action_prior_weight=policy_prior_weight,
            **search_kwargs,
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.policy_prior_weight = policy_prior_weight
