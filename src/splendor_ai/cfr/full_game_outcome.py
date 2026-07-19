"""Finite full-game Splendor outcome-sampling trajectory collector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding.public_belief import actions_by_canonical_descriptor
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)

from .full_game import build_schema_row
from .outcome_sampling import OutcomeSamplingAdvantageRow, behavior_mixture


NATURAL_TERMINALS = ("completed", "forced_pass_deadlock")
FINITE_ADJUDICATIONS = (
    "repetition_adjudication",
    "no_progress_adjudication",
    "turn_limit_adjudication",
)


def finite_learning_terminal_reason(
    state: SplendorState,
    *,
    forced_pass_streak: int,
    repetition_count: int,
    no_progress_streak: int,
    repetition_limit: int = 4,
    no_progress_limit: int = 60,
    turn_limit: int = 200,
) -> str | None:
    """Apply the frozen 048 terminal priority to the declared learning game."""

    if state.terminal:
        return "completed"
    if forced_pass_streak >= 2:
        return "forced_pass_deadlock"
    if repetition_count >= repetition_limit:
        return "repetition_adjudication"
    if no_progress_streak >= no_progress_limit:
        return "no_progress_adjudication"
    if state.turn_index >= turn_limit:
        return "turn_limit_adjudication"
    return None


@dataclass(frozen=True, slots=True)
class _PendingFullGameDecision:
    schema: Any
    target_strategy: tuple[float, ...]
    behavior_strategy: tuple[float, ...]
    selected_index: int
    step_index: int
    log_opponent_prefix: float
    turn: int


class FrozenEntityTargetPolicy:
    """Softmax policy over canonical legal actions from a frozen entity model."""

    def __init__(self, checkpoint_path: str | Path, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.checkpoint_path = Path(checkpoint_path)
        self.temperature = temperature
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model_kind = checkpoint.get("model_kind")
        self.model_config = checkpoint.get("model_config", {})
        self.model = CanonicalEntityActionScorerModel(
            CanonicalEntityActionScorerConfig(**self.model_config)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def strategy(self, schema) -> tuple[float, ...]:
        state = schema.state
        actions = schema.actions
        entity_width = len(state.entity_type_ids)
        action_width = len(actions.action_type_ids)
        with torch.no_grad():
            scores, _value = self.model(
                entity_type_ids=torch.tensor([state.entity_type_ids], dtype=torch.long),
                entity_card_ids=torch.tensor([state.card_ids], dtype=torch.long),
                entity_noble_ids=torch.tensor([state.noble_ids], dtype=torch.long),
                entity_numeric_features=torch.tensor([state.numeric_features], dtype=torch.float32),
                entity_mask=torch.ones((1, entity_width), dtype=torch.bool),
                action_type_ids=torch.tensor([actions.action_type_ids], dtype=torch.long),
                action_card_ids=torch.tensor([actions.card_ids], dtype=torch.long),
                action_noble_ids=torch.tensor([actions.noble_ids], dtype=torch.long),
                action_numeric_features=torch.tensor([actions.numeric_features], dtype=torch.float32),
                action_mask=torch.ones((1, action_width), dtype=torch.bool),
            )
        scaled = scores[0].to(dtype=torch.float64) / self.temperature
        probabilities = torch.softmax(scaled, dim=0).tolist()
        floored = [max(float(value), 1e-12) for value in probabilities]
        total = sum(floored)
        return tuple(value / total for value in floored)


class FullGameOutcomeSampler:
    def __init__(
        self,
        target_policy: FrozenEntityTargetPolicy,
        *,
        exploration: float = 0.1,
        repetition_limit: int = 4,
        no_progress_limit: int = 60,
        turn_limit: int = 200,
    ) -> None:
        self.target_policy = target_policy
        self.exploration = exploration
        self.repetition_limit = repetition_limit
        self.no_progress_limit = no_progress_limit
        self.turn_limit = turn_limit

    def sample(self, seed: int, traverser: int) -> tuple[dict[str, Any], list[OutcomeSamplingAdvantageRow]]:
        if traverser not in (0, 1):
            raise ValueError("traverser must be 0 or 1.")
        env = SplendorEnv(seed=seed)
        rng = random.Random((seed * 10) + traverser)
        state = env.initial_state()
        history = ()
        visits = {state_signature(state): 1}
        no_progress_streak = 0
        forced_pass_streak = 0
        sampled_log_target_steps: list[float] = []
        pending: list[_PendingFullGameDecision] = []
        log_behavior_reach = 0.0
        log_opponent_prefix = 0.0
        illegal_actions = 0
        termination_reason = "completed"

        while True:
            if state.terminal:
                termination_reason = "completed"
                break
            legal_map = actions_by_canonical_descriptor(state, env.legal_actions(state))
            descriptors = sorted(legal_map, key=repr)
            ordered_actions = [legal_map[key] for key in descriptors]
            if not ordered_actions:
                raise RuntimeError("Nonterminal full-game state has no legal action.")
            schema = build_schema_row(state, history, ordered_actions)
            target = self.target_policy.strategy(schema)
            behavior = behavior_mixture(target, self.exploration)
            selected = _sample_index(behavior, rng)
            actor = state.current_player
            step_index = len(sampled_log_target_steps)
            if actor == traverser:
                pending.append(
                    _PendingFullGameDecision(
                        schema=schema,
                        target_strategy=target,
                        behavior_strategy=behavior,
                        selected_index=selected,
                        step_index=step_index,
                        log_opponent_prefix=log_opponent_prefix,
                        turn=state.turn_index,
                    )
                )
            selected_target = target[selected]
            log_target = math.log(selected_target)
            sampled_log_target_steps.append(log_target)
            log_behavior_reach += math.log(behavior[selected])
            if actor != traverser:
                log_opponent_prefix += log_target

            action = ordered_actions[selected]
            event = _public_action_event(state, action)
            try:
                next_state = env.step(state, action)
            except ValueError:
                illegal_actions += 1
                raise
            forced = len(ordered_actions) == 1 and action.action_type == ActionType.PASS
            forced_pass_streak = forced_pass_streak + 1 if forced else 0
            if is_progress_transition(state, next_state, action):
                no_progress_streak = 0
            else:
                no_progress_streak += 1
            history = history + (event,)
            state = next_state

            signature = state_signature(state)
            visits[signature] = visits.get(signature, 0) + 1
            reason = finite_learning_terminal_reason(
                state,
                forced_pass_streak=forced_pass_streak,
                repetition_count=visits[signature],
                no_progress_streak=no_progress_streak,
                repetition_limit=self.repetition_limit,
                no_progress_limit=self.no_progress_limit,
                turn_limit=self.turn_limit,
            )
            if reason is not None:
                termination_reason = reason
                break

        winner = state.winner if state.terminal else _adjudicate_scores(state)
        utility = 0.0 if winner is None else 1.0 if winner == traverser else -1.0
        suffix_logs = _exclusive_suffix_sums(sampled_log_target_steps)
        rows: list[OutcomeSamplingAdvantageRow] = []
        nonfinite_target_rows = 0
        log_weights: list[float] = []
        for record in pending:
            log_weight = (
                record.log_opponent_prefix
                + suffix_logs[record.step_index]
                - log_behavior_reach
            )
            log_weights.append(log_weight)
            if log_weight > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
                nonfinite_target_rows += 1
                continue
            magnitude = 0.0 if utility == 0.0 else math.copysign(math.exp(log_weight), utility)
            selected_probability = record.target_strategy[record.selected_index]
            regret_target = tuple(
                ((1.0 - selected_probability) if index == record.selected_index else -selected_probability)
                * magnitude
                for index in range(len(record.target_strategy))
            )
            if not all(math.isfinite(value) for value in regret_target):
                nonfinite_target_rows += 1
                continue
            rows.append(
                OutcomeSamplingAdvantageRow(
                    state=record.schema.state,
                    history=record.schema.history,
                    actions=record.schema.actions,
                    target_strategy=record.target_strategy,
                    behavior_strategy=record.behavior_strategy,
                    selected_index=record.selected_index,
                    regret_target=regret_target,
                    player=traverser,
                    turn=record.turn,
                    terminal_reason=termination_reason,
                    log_importance_weight=log_weight,
                )
            )
        row_hash = hashlib.sha256(
            json.dumps(
                [row.to_payload() for row in rows],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result = {
            "seed": seed,
            "traverser": traverser,
            "sampling_seed": (seed * 10) + traverser,
            "turns": state.turn_index,
            "winner": winner,
            "utility": utility,
            "final_scores": [state.players[0].score, state.players[1].score],
            "termination_reason": termination_reason,
            "natural_terminal": termination_reason in NATURAL_TERMINALS,
            "sampled_decision_steps": len(sampled_log_target_steps),
            "candidate_advantage_rows": len(pending),
            "emitted_advantage_rows": len(rows),
            "nonfinite_target_rows": nonfinite_target_rows,
            "illegal_actions": illegal_actions,
            "max_log_importance_weight": max(log_weights, default=None),
            "min_log_importance_weight": min(log_weights, default=None),
            "row_payload_sha256": row_hash,
        }
        return result, rows


def _sample_index(probabilities: tuple[float, ...], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def _exclusive_suffix_sums(values: list[float]) -> list[float]:
    suffix = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        suffix[index] = running
        running += values[index]
    return suffix


def _adjudicate_scores(state: SplendorState) -> int | None:
    best = max(player.score for player in state.players)
    contenders = [index for index, player in enumerate(state.players) if player.score == best]
    if len(contenders) == 1:
        return contenders[0]
    fewest = min(len(state.players[index].purchased_cards) for index in contenders)
    finalists = [
        index for index in contenders if len(state.players[index].purchased_cards) == fewest
    ]
    return finalists[0] if len(finalists) == 1 else None


def _public_action_event(state, action):
    from splendor_ai.encoding.information_state import public_action_event

    return public_action_event(state, action)
