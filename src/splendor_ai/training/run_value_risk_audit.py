"""Audit policy-vs-PGS disagreements with value/outcome metadata.

This diagnostic intentionally keeps the value model out of the primary
teacher. Plain policy-guided search remains the reference; the value/outcome
model is recorded as metadata and as a separate "would outcome-PGS change this"
warning signal.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from splendor_ai.bots import (
    CheckpointLegalActionPriorScorer,
    CheckpointValueEvaluator,
    ShallowSearchBot,
)
from splendor_ai.encoding import (
    ActionCodec,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import ValueMLP, ValueModelConfig
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_policy_search_disagreement import (
    DisagreementConfig,
    ModelActionScore,
    _load_policy_scorer,
)
from splendor_ai.training.run_search_value_diff import _phase_for_turn, _score_margin_bucket


@dataclass(frozen=True, slots=True)
class ValueRiskAuditConfig:
    policy_checkpoint: Path
    policy_model_kind: str
    replay_paths: tuple[Path, ...]
    output_path: Path
    policy_device: str = "cpu"
    policy_prior_checkpoint_path: Path | None = None
    policy_prior_device: str = "cpu"
    policy_prior_weight: float = 3_000.0
    value_checkpoint_path: Path | None = None
    value_device: str = "cpu"
    value_weight: float = 0.05
    value_scale: float = 15_000.0
    value_prediction_mode: str = "outcome"
    value_outcome_blend_weight: float = 0.5
    value_min_turn_index: int | None = 40
    value_max_turn_index: int | None = None
    search_depth: int = 2
    search_max_branching: int = 10
    search_buy_branching: int = 6
    search_reserve_branching: int = 3
    search_take_branching: int = 3
    seed: int = 0
    max_rows: int | None = None
    log_every: int = 500
    player_id_filter: tuple[int, ...] | None = None
    phase_filter: tuple[str, ...] | None = None
    min_turn_index: int | None = None
    rows_output_path: Path | None = None
    examples_limit: int = 30


class _ValueProbe:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        self._device = device
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._model = ValueMLP(ValueModelConfig(**checkpoint.get("model_config", {})))
        self._model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self._model.to(device)
        self._model.eval()

    def evaluate(self, state: SplendorState, player_id: int) -> dict[str, float]:
        observation = encode_public_observation_tensor(state, player_id)
        observation_vector = fit_feature_vector(
            observation.vector,
            self._model.config.observation_size,
        )
        observation_tensor = torch.tensor(
            [observation_vector],
            dtype=torch.float32,
            device=self._device,
        )
        with torch.no_grad():
            scalar_value = float(self._model(observation_tensor).detach().cpu().item())
            logits = self._model.outcome_logits(observation_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        loss_probability = float(probabilities[0].item())
        draw_probability = float(probabilities[1].item())
        win_probability = float(probabilities[2].item())
        return {
            "scalar_value": scalar_value,
            "loss_probability": loss_probability,
            "draw_probability": draw_probability,
            "win_probability": win_probability,
            "outcome_value": win_probability - loss_probability,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a neural policy top action with plain PGS and record value/outcome "
            "metadata without using value as the primary teacher."
        )
    )
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument(
        "--policy-model-kind",
        choices=(
            "flat_policy",
            "legal_action_scorer",
            "legal_action_attention_scorer",
            "legal_action_set_scorer",
        ),
        required=True,
    )
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--rows-output-path", default=None)
    parser.add_argument("--policy-prior-checkpoint", required=True)
    parser.add_argument("--policy-prior-device", default="cpu")
    parser.add_argument("--policy-prior-weight", type=float, default=3_000.0)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--value-device", default="cpu")
    parser.add_argument("--value-weight", type=float, default=0.05)
    parser.add_argument("--value-scale", type=float, default=15_000.0)
    parser.add_argument(
        "--value-prediction-mode",
        choices=("value", "outcome", "blend"),
        default="outcome",
    )
    parser.add_argument("--value-outcome-blend-weight", type=float, default=0.5)
    parser.add_argument("--value-min-turn-index", type=int, default=40)
    parser.add_argument("--value-max-turn-index", type=int, default=None)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--min-turn-index", type=int, default=None)
    parser.add_argument(
        "--player-id-filter",
        nargs="*",
        type=int,
        choices=(0, 1),
        default=None,
    )
    parser.add_argument(
        "--phase-filter",
        nargs="*",
        choices=("early", "mid", "late"),
        default=None,
    )
    parser.add_argument("--examples-limit", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_value_risk_audit(
        ValueRiskAuditConfig(
            policy_checkpoint=Path(args.policy_checkpoint),
            policy_model_kind=args.policy_model_kind,
            policy_device=args.policy_device,
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            rows_output_path=(
                Path(args.rows_output_path) if args.rows_output_path is not None else None
            ),
            policy_prior_checkpoint_path=Path(args.policy_prior_checkpoint),
            policy_prior_device=args.policy_prior_device,
            policy_prior_weight=args.policy_prior_weight,
            value_checkpoint_path=Path(args.value_checkpoint),
            value_device=args.value_device,
            value_weight=args.value_weight,
            value_scale=args.value_scale,
            value_prediction_mode=args.value_prediction_mode,
            value_outcome_blend_weight=args.value_outcome_blend_weight,
            value_min_turn_index=args.value_min_turn_index,
            value_max_turn_index=args.value_max_turn_index,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            seed=args.seed,
            max_rows=args.max_rows,
            log_every=args.log_every,
            min_turn_index=args.min_turn_index,
            player_id_filter=(
                tuple(args.player_id_filter)
                if args.player_id_filter is not None and len(args.player_id_filter) > 0
                else None
            ),
            phase_filter=(
                tuple(args.phase_filter)
                if args.phase_filter is not None and len(args.phase_filter) > 0
                else None
            ),
            examples_limit=args.examples_limit,
        )
    )
    print(
        "[value-risk-audit]"
        f" analyzed={summary['rows_analyzed']}"
        f" policy_plain_disagreements={summary['overall']['policy_plain_disagreements']}"
        f" policy_plain_disagreement_rate={summary['overall']['policy_plain_disagreement_rate']:.4f}"
        f" outcome_plain_changes={summary['overall']['outcome_plain_changes']}"
        f" output={args.output_path}"
    )


def run_value_risk_audit(config: ValueRiskAuditConfig) -> dict[str, Any]:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.policy_prior_checkpoint_path is None:
        raise ValueError("policy_prior_checkpoint_path is required.")
    if config.value_checkpoint_path is None:
        raise ValueError("value_checkpoint_path is required.")
    unknown_phases = set(config.phase_filter or ()) - {"early", "mid", "late"}
    if unknown_phases:
        raise ValueError(f"Unknown phases: {sorted(unknown_phases)}")
    if config.examples_limit < 0:
        raise ValueError("examples_limit must be non-negative.")

    env = SplendorEnv()
    codec = ActionCodec()
    policy_scorer = _load_policy_scorer(
        DisagreementConfig(
            checkpoint=config.policy_checkpoint,
            model_kind=config.policy_model_kind,
            replay_paths=config.replay_paths,
            output_path=config.output_path,
            device=config.policy_device,
        )
    )
    action_prior = CheckpointLegalActionPriorScorer(
        checkpoint_path=config.policy_prior_checkpoint_path,
        device=config.policy_prior_device,
    )
    value_evaluator = CheckpointValueEvaluator(
        checkpoint_path=config.value_checkpoint_path,
        device=config.value_device,
        value_scale=config.value_scale,
        prediction_mode=config.value_prediction_mode,
        outcome_blend_weight=config.value_outcome_blend_weight,
    )
    value_probe = _ValueProbe(config.value_checkpoint_path, config.value_device)

    counters = _empty_counters()
    examples: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    rows_seen = 0
    rows_skipped_by_player_id = 0
    rows_skipped_by_phase = 0
    rows_skipped_by_turn = 0
    rows_missing_snapshot = 0
    rows_without_actions = 0

    for replay_path in config.replay_paths:
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if config.max_rows is not None and rows_seen >= config.max_rows:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                rows_seen += 1
                payload = json.loads(stripped)
                player_id = int(payload.get("player_id", -1))
                if config.player_id_filter is not None and player_id not in config.player_id_filter:
                    rows_skipped_by_player_id += 1
                    continue
                turn_index = int(payload.get("turn_index", 0))
                if config.min_turn_index is not None and turn_index < config.min_turn_index:
                    rows_skipped_by_turn += 1
                    continue
                phase = _phase_for_turn(turn_index)
                if config.phase_filter is not None and phase not in config.phase_filter:
                    rows_skipped_by_phase += 1
                    continue
                snapshot = payload.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    rows_missing_snapshot += 1
                    continue
                state = deserialize_state_snapshot(snapshot)
                legal_actions = env.legal_actions(state)
                if not legal_actions:
                    rows_without_actions += 1
                    continue

                plain_bot = _build_search_bot(
                    config=config,
                    row_offset=rows_seen,
                    action_prior=action_prior,
                )
                outcome_bot = _build_search_bot(
                    config=config,
                    row_offset=rows_seen,
                    action_prior=action_prior,
                    value_evaluator=value_evaluator,
                )
                policy_scores = policy_scorer.score_actions(state, legal_actions)
                plain_scores = plain_bot.rank_actions(env, state, legal_actions)
                outcome_scores = outcome_bot.rank_actions(env, state, legal_actions)
                if not policy_scores or not plain_scores or not outcome_scores:
                    rows_without_actions += 1
                    continue

                row_record = _record_row(
                    config=config,
                    counters=counters,
                    examples=examples,
                    codec=codec,
                    env=env,
                    value_probe=value_probe,
                    state=state,
                    payload=payload,
                    replay_path=replay_path,
                    policy_top=policy_scores[0],
                    plain_scores=plain_scores,
                    outcome_scores=outcome_scores,
                )
                if config.rows_output_path is not None:
                    row_records.append(row_record)

                if config.log_every > 0 and rows_seen % config.log_every == 0:
                    print(
                        f"[value-risk-audit] rows_seen={rows_seen}"
                        f" analyzed={counters['overall']['count']}"
                        f" policy_plain_disagreements={counters['overall']['policy_plain_disagreements']}"
                        f" outcome_plain_changes={counters['overall']['outcome_plain_changes']}"
                    )
        if config.max_rows is not None and rows_seen >= config.max_rows:
            break

    summary = {
        "rows_seen": rows_seen,
        "rows_analyzed": counters["overall"]["count"],
        "rows_skipped_by_player_id": rows_skipped_by_player_id,
        "rows_skipped_by_phase": rows_skipped_by_phase,
        "rows_skipped_by_turn": rows_skipped_by_turn,
        "rows_missing_snapshot": rows_missing_snapshot,
        "rows_without_actions": rows_without_actions,
        "overall": _finalize_counter(counters["overall"]),
        "by_seat": _finalize_buckets(counters["by_seat"]),
        "by_phase": _finalize_buckets(counters["by_phase"]),
        "by_seat_phase": _finalize_buckets(counters["by_seat_phase"]),
        "by_score_margin": _finalize_buckets(counters["by_score_margin"]),
        "by_policy_action_type": _finalize_buckets(counters["by_policy_action_type"]),
        "by_plain_action_type": _finalize_buckets(counters["by_plain_action_type"]),
        "by_outcome_action_type": _finalize_buckets(counters["by_outcome_action_type"]),
        "policy_plain_action_type_matrix": counters["policy_plain_action_type_matrix"],
        "plain_outcome_action_type_matrix": counters["plain_outcome_action_type_matrix"],
        "examples": examples,
        "config": {
            **asdict(config),
            "policy_checkpoint": str(config.policy_checkpoint),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
            "rows_output_path": (
                str(config.rows_output_path)
                if config.rows_output_path is not None
                else None
            ),
            "policy_prior_checkpoint_path": str(config.policy_prior_checkpoint_path),
            "value_checkpoint_path": str(config.value_checkpoint_path),
        },
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.rows_output_path is not None:
        config.rows_output_path.parent.mkdir(parents=True, exist_ok=True)
        with config.rows_output_path.open("w", encoding="utf-8") as handle:
            for row in row_records:
                handle.write(json.dumps(row) + "\n")
    return summary


def _build_search_bot(
    config: ValueRiskAuditConfig,
    row_offset: int,
    action_prior: CheckpointLegalActionPriorScorer,
    value_evaluator: CheckpointValueEvaluator | None = None,
) -> ShallowSearchBot:
    return ShallowSearchBot(
        depth=config.search_depth,
        max_branching=config.search_max_branching,
        max_buy_actions=config.search_buy_branching,
        max_reserve_actions=config.search_reserve_branching,
        max_take_actions=config.search_take_branching,
        seed=config.seed + row_offset,
        action_prior=action_prior,
        action_prior_weight=config.policy_prior_weight,
        value_evaluator=value_evaluator,
        value_weight=config.value_weight if value_evaluator is not None else 0.0,
        value_min_turn_index=config.value_min_turn_index,
        value_max_turn_index=config.value_max_turn_index,
    )


def _empty_counters() -> dict[str, Any]:
    return {
        "overall": _new_counter(),
        "by_seat": {},
        "by_phase": {},
        "by_seat_phase": {},
        "by_score_margin": {},
        "by_policy_action_type": {},
        "by_plain_action_type": {},
        "by_outcome_action_type": {},
        "policy_plain_action_type_matrix": {},
        "plain_outcome_action_type_matrix": {},
    }


def _new_counter() -> dict[str, Any]:
    return {
        "count": 0,
        "policy_plain_disagreements": 0,
        "outcome_plain_changes": 0,
        "policy_matches_outcome": 0,
        "policy_matches_outcome_on_plain_disagreement": 0,
        "policy_plain_regrets": [],
        "outcome_plain_regrets": [],
        "policy_outcome_value_advantages": [],
        "plain_outcome_value_advantages": [],
        "outcome_pgs_outcome_value_advantages": [],
    }


def _record_row(
    config: ValueRiskAuditConfig,
    counters: dict[str, Any],
    examples: list[dict[str, Any]],
    codec: ActionCodec,
    env: SplendorEnv,
    value_probe: _ValueProbe,
    state: SplendorState,
    payload: dict[str, Any],
    replay_path: Path,
    policy_top: ModelActionScore,
    plain_scores: tuple[Any, ...],
    outcome_scores: tuple[Any, ...],
) -> dict[str, Any]:
    actor_id = int(payload.get("player_id", state.current_player))
    plain_top = plain_scores[0]
    outcome_top = outcome_scores[0]
    plain_top_index = codec.encode(state, plain_top.action)
    outcome_top_index = codec.encode(state, outcome_top.action)
    policy_plain_disagrees = policy_top.action_index != plain_top_index
    outcome_plain_changes = outcome_top_index != plain_top_index
    policy_matches_outcome = policy_top.action_index == outcome_top_index

    plain_values = {codec.encode(state, score.action): float(score.value) for score in plain_scores}
    outcome_values = {codec.encode(state, score.action): float(score.value) for score in outcome_scores}
    policy_plain_regret = _difference(
        plain_values.get(plain_top_index),
        plain_values.get(policy_top.action_index),
    )
    outcome_plain_regret = _difference(
        plain_values.get(plain_top_index),
        plain_values.get(outcome_top_index),
    )
    outcome_regret_of_plain = _difference(
        outcome_values.get(outcome_top_index),
        outcome_values.get(plain_top_index),
    )

    current_value = value_probe.evaluate(state, actor_id)
    policy_after = _evaluate_after_action(env, value_probe, state, policy_top.action, actor_id)
    plain_after = _evaluate_after_action(env, value_probe, state, plain_top.action, actor_id)
    outcome_after = _evaluate_after_action(env, value_probe, state, outcome_top.action, actor_id)
    policy_value_advantage = policy_after["outcome_value"] - current_value["outcome_value"]
    plain_value_advantage = plain_after["outcome_value"] - current_value["outcome_value"]
    outcome_value_advantage = outcome_after["outcome_value"] - current_value["outcome_value"]

    turn_index = int(payload.get("turn_index", 0))
    phase = _phase_for_turn(turn_index)
    seat = f"seat{actor_id}"
    score_margin = state.players[actor_id].score - state.players[1 - actor_id].score
    score_margin_bucket = _score_margin_bucket(score_margin)
    policy_type = policy_top.action.action_type.name
    plain_type = plain_top.action.action_type.name
    outcome_type = outcome_top.action.action_type.name

    for counter in (
        counters["overall"],
        _bucket(counters["by_seat"], seat),
        _bucket(counters["by_phase"], phase),
        _bucket(counters["by_seat_phase"], f"{seat}:{phase}"),
        _bucket(counters["by_score_margin"], score_margin_bucket),
        _bucket(counters["by_policy_action_type"], policy_type),
        _bucket(counters["by_plain_action_type"], plain_type),
        _bucket(counters["by_outcome_action_type"], outcome_type),
    ):
        _record_counter(
            counter=counter,
            policy_plain_disagrees=policy_plain_disagrees,
            outcome_plain_changes=outcome_plain_changes,
            policy_matches_outcome=policy_matches_outcome,
            policy_plain_regret=policy_plain_regret,
            outcome_plain_regret=outcome_plain_regret,
            policy_value_advantage=policy_value_advantage,
            plain_value_advantage=plain_value_advantage,
            outcome_value_advantage=outcome_value_advantage,
        )
    _increment_matrix(counters["policy_plain_action_type_matrix"], policy_type, plain_type)
    _increment_matrix(counters["plain_outcome_action_type_matrix"], plain_type, outcome_type)

    row_record = {
        "replay_path": str(replay_path),
        "game_seed": payload.get("game_seed"),
        "turn_index": turn_index,
        "phase": phase,
        "player_id": actor_id,
        "seat": seat,
        "scores": [state.players[0].score, state.players[1].score],
        "score_margin": score_margin,
        "score_margin_bucket": score_margin_bucket,
        "policy_plain_disagrees": policy_plain_disagrees,
        "outcome_plain_changes": outcome_plain_changes,
        "policy_matches_outcome": policy_matches_outcome,
        "policy_plain_regret": policy_plain_regret,
        "outcome_plain_regret": outcome_plain_regret,
        "outcome_regret_of_plain_action": outcome_regret_of_plain,
        "current_value": current_value,
        "policy_after_value": policy_after,
        "plain_after_value": plain_after,
        "outcome_after_value": outcome_after,
        "policy_outcome_value_advantage": policy_value_advantage,
        "plain_outcome_value_advantage": plain_value_advantage,
        "outcome_pgs_outcome_value_advantage": outcome_value_advantage,
        "policy_action_index": policy_top.action_index,
        "policy_action_type": policy_type,
        "policy_score": float(policy_top.score),
        "policy_action_payload": serialize_action(policy_top.action),
        "plain_action_index": plain_top_index,
        "plain_action_type": plain_type,
        "plain_search_value": float(plain_top.value),
        "plain_heuristic_score": float(plain_top.heuristic_score),
        "plain_policy_score": float(plain_top.policy_score),
        "plain_action_payload": serialize_action(plain_top.action),
        "outcome_action_index": outcome_top_index,
        "outcome_action_type": outcome_type,
        "outcome_search_value": float(outcome_top.value),
        "outcome_heuristic_score": float(outcome_top.heuristic_score),
        "outcome_policy_score": float(outcome_top.policy_score),
        "outcome_action_payload": serialize_action(outcome_top.action),
        "replay_action_index": payload.get("action_index"),
        "replay_action_payload": payload.get("action_payload"),
    }
    if policy_plain_disagrees:
        _store_example(examples, config.examples_limit, row_record)
    return row_record


def _evaluate_after_action(
    env: SplendorEnv,
    value_probe: _ValueProbe,
    state: SplendorState,
    action: Action,
    actor_id: int,
) -> dict[str, float]:
    next_state = env.step(state, action)
    return value_probe.evaluate(next_state, actor_id)


def _bucket(buckets: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    return buckets.setdefault(key, _new_counter())


def _record_counter(
    counter: dict[str, Any],
    policy_plain_disagrees: bool,
    outcome_plain_changes: bool,
    policy_matches_outcome: bool,
    policy_plain_regret: float | None,
    outcome_plain_regret: float | None,
    policy_value_advantage: float,
    plain_value_advantage: float,
    outcome_value_advantage: float,
) -> None:
    counter["count"] += 1
    if policy_plain_disagrees:
        counter["policy_plain_disagreements"] += 1
        if policy_matches_outcome:
            counter["policy_matches_outcome_on_plain_disagreement"] += 1
        if policy_plain_regret is not None and math.isfinite(policy_plain_regret):
            counter["policy_plain_regrets"].append(float(policy_plain_regret))
    if outcome_plain_changes:
        counter["outcome_plain_changes"] += 1
        if outcome_plain_regret is not None and math.isfinite(outcome_plain_regret):
            counter["outcome_plain_regrets"].append(float(outcome_plain_regret))
    if policy_matches_outcome:
        counter["policy_matches_outcome"] += 1
    counter["policy_outcome_value_advantages"].append(float(policy_value_advantage))
    counter["plain_outcome_value_advantages"].append(float(plain_value_advantage))
    counter["outcome_pgs_outcome_value_advantages"].append(float(outcome_value_advantage))


def _increment_matrix(matrix: dict[str, dict[str, int]], row: str, column: str) -> None:
    matrix.setdefault(row, {})
    matrix[row][column] = matrix[row].get(column, 0) + 1


def _store_example(
    examples: list[dict[str, Any]],
    examples_limit: int,
    row_record: dict[str, Any],
) -> None:
    if examples_limit <= 0:
        return
    examples.append(row_record)
    examples.sort(
        key=lambda item: _example_sort_key(item),
        reverse=True,
    )
    del examples[examples_limit:]


def _example_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    regret = item.get("policy_plain_regret")
    outcome_shift = (
        float(item["policy_outcome_value_advantage"])
        - float(item["plain_outcome_value_advantage"])
    )
    return (
        float(regret) if regret is not None and math.isfinite(float(regret)) else -math.inf,
        1.0 if item.get("policy_matches_outcome") else 0.0,
        outcome_shift,
    )


def _finalize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: _finalize_counter(buckets[key]) for key in sorted(buckets)}


def _finalize_counter(counter: dict[str, Any]) -> dict[str, Any]:
    count = counter["count"]
    return {
        "count": count,
        "policy_plain_disagreements": counter["policy_plain_disagreements"],
        "policy_plain_disagreement_rate": _rate(counter["policy_plain_disagreements"], count),
        "outcome_plain_changes": counter["outcome_plain_changes"],
        "outcome_plain_change_rate": _rate(counter["outcome_plain_changes"], count),
        "policy_matches_outcome": counter["policy_matches_outcome"],
        "policy_matches_outcome_rate": _rate(counter["policy_matches_outcome"], count),
        "policy_matches_outcome_on_plain_disagreement": counter[
            "policy_matches_outcome_on_plain_disagreement"
        ],
        "policy_matches_outcome_on_plain_disagreement_rate": _rate(
            counter["policy_matches_outcome_on_plain_disagreement"],
            counter["policy_plain_disagreements"],
        ),
        "policy_plain_regret_mean": _mean(counter["policy_plain_regrets"]),
        "policy_plain_regret_max": _max_or_none(counter["policy_plain_regrets"]),
        "outcome_plain_regret_mean": _mean(counter["outcome_plain_regrets"]),
        "outcome_plain_regret_max": _max_or_none(counter["outcome_plain_regrets"]),
        "policy_outcome_value_advantage_mean": _mean(
            counter["policy_outcome_value_advantages"]
        ),
        "plain_outcome_value_advantage_mean": _mean(
            counter["plain_outcome_value_advantages"]
        ),
        "outcome_pgs_outcome_value_advantage_mean": _mean(
            counter["outcome_pgs_outcome_value_advantages"]
        ),
    }


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


if __name__ == "__main__":
    main()
