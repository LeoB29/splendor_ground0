"""Attribute late PUCT errors using legal matched-continuation root returns."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_leaf_quality_audit import _phase
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer
from splendor_ai.training.run_rollout_action_credit import (
    rollout_action_return,
    select_root_candidates,
)


@dataclass(frozen=True, slots=True)
class ResidualErrorAuditConfig:
    replay_paths: tuple[Path, ...]
    policy_checkpoint: Path
    output_path: Path
    exclude_audit_path: Path | None = None
    max_states: int = 90
    determinizations: int = 3
    rollouts_per_determinization: int = 6
    min_turn_index: int = 40
    policy_top_k: int = 4
    max_policy_candidates: int = 6
    min_return_range: float = 0.05
    confidence_z: float = 1.96
    rollout_epsilon: float = 0.1
    rollout_temperature: float = 0.5
    rollout_top_k: int = 3
    rollout_max_additional_turns: int = 120
    outcome_weight: float = 0.75
    score_margin_scale: float = 5.0
    bootstrap_samples: int = 5_000
    seed: int = 71
    device: str = "cpu"
    log_every_states: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute resolved late PUCT root errors to policy-candidate, prior, "
            "backup/search, or information-set-variance causes."
        )
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--exclude-audit-path",
        default=None,
        help="Optional prior residual-audit JSON whose sampled states must be excluded.",
    )
    parser.add_argument("--max-states", type=int, default=90)
    parser.add_argument("--determinizations", type=int, default=3)
    parser.add_argument("--rollouts-per-determinization", type=int, default=6)
    parser.add_argument("--min-turn-index", type=int, default=40)
    parser.add_argument("--policy-top-k", type=int, default=4)
    parser.add_argument("--max-policy-candidates", type=int, default=6)
    parser.add_argument("--min-return-range", type=float, default=0.05)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--rollout-epsilon", type=float, default=0.1)
    parser.add_argument("--rollout-temperature", type=float, default=0.5)
    parser.add_argument("--rollout-top-k", type=int, default=3)
    parser.add_argument("--rollout-max-additional-turns", type=int, default=120)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-states", type=int, default=10)
    return parser


def _state_key(payload: dict[str, Any]) -> tuple[int, int, int]:
    return int(payload["game_seed"]), int(payload["turn_index"]), int(payload["player_id"])


def _excluded_state_keys(path: Path | None) -> set[tuple[int, int, int]]:
    if path is None:
        return set()
    audit = json.loads(path.read_text(encoding="utf-8"))
    return {
        _state_key(row)
        for row in audit.get("states", [])
        if all(key in row for key in ("game_seed", "turn_index", "player_id"))
    }


def _select_rows(config: ResidualErrorAuditConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    rows_excluded = 0
    excluded_keys = _excluded_state_keys(config.exclude_audit_path)
    for path in config.replay_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_read += 1
                payload = json.loads(line)
                if (
                    isinstance(payload.get("state_snapshot"), dict)
                    and int(payload.get("turn_index", 0)) >= config.min_turn_index
                    and isinstance(payload.get("root_visit_distribution"), list)
                ):
                    if _state_key(payload) in excluded_keys:
                        rows_excluded += 1
                        continue
                    by_phase[_phase(int(payload["turn_index"]))].append(payload)
    rng = random.Random(config.seed)
    selected: list[dict[str, Any]] = []
    selected_by_phase: dict[str, int] = {}
    phase_names = ("late_40_49", "endgame_50_59", "endgame_60_plus")
    base, remainder = divmod(config.max_states, len(phase_names))
    for position, phase in enumerate(phase_names):
        phase_rows = by_phase[phase]
        rng.shuffle(phase_rows)
        count = min(len(phase_rows), base + (1 if position < remainder else 0))
        selected.extend(phase_rows[:count])
        selected_by_phase[phase] = count
    rng.shuffle(selected)
    return selected, {
        "rows_read": rows_read,
        "rows_excluded": rows_excluded,
        "excluded_state_count": len(excluded_keys),
        "eligible_rows_by_phase": {phase: len(rows) for phase, rows in by_phase.items()},
        "states_selected": len(selected),
        "states_selected_by_phase": selected_by_phase,
    }


def _paired_margin(values: list[float], comparator: list[float]) -> tuple[float, float]:
    differences = [left - right for left, right in zip(values, comparator, strict=True)]
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return mean, 0.0
    return mean, statistics.stdev(differences) / math.sqrt(len(differences))


def _top_action_index(items: list[dict[str, Any]], value_key: str) -> int:
    return min(items, key=lambda item: (-float(item[value_key]), int(item["action_index"])))["action_index"]


def _root_visit_choice(payload: dict[str, Any], legal_indices: set[int]) -> int | None:
    visits = [
        item
        for item in payload.get("root_visit_distribution", [])
        if isinstance(item, dict) and int(item.get("action_index", -1)) in legal_indices
    ]
    if not visits:
        return None
    return min(
        visits,
        key=lambda item: (-float(item.get("visits", item.get("probability", 0.0))), int(item["action_index"])),
    )["action_index"]


def _classify_state(
    *,
    resolved: bool,
    puct_choice: int | None,
    oracle_action: int,
    prior_action: int,
) -> str:
    if not resolved:
        return "unresolved"
    if puct_choice == oracle_action:
        return "puct_matches_oracle"
    if puct_choice == prior_action and prior_action != oracle_action:
        return "prior_miss"
    return "backup_search_miss"


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = statistics.fmean(values)
    if len(values) == 1 or samples <= 0:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": estimates[int(0.025 * (samples - 1))],
        "ci95_high": estimates[int(0.975 * (samples - 1))],
    }


def _summarize(rows: list[dict[str, Any]], config: ResidualErrorAuditConfig) -> dict[str, Any]:
    counts = Counter(row["attribution"] for row in rows)
    resolved = [row for row in rows if row["resolved"]]
    resolved_counts = Counter(row["attribution"] for row in resolved)
    by_phase: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_phase[row["phase"]][row["attribution"]] += 1
    return {
        "states_evaluated": len(rows),
        "resolved_states": len(resolved),
        "attribution_counts": dict(sorted(counts.items())),
        "resolved_attribution_counts": dict(sorted(resolved_counts.items())),
        "resolved_attribution_share": {
            label: _bootstrap_mean_ci(
                [float(row["attribution"] == label) for row in resolved],
                config.seed + position,
                config.bootstrap_samples,
            )
            for position, label in enumerate(sorted(resolved_counts))
        },
        "policy_candidate_oracle_miss_rate": _bootstrap_mean_ci(
            [float(not row["oracle_in_policy_candidates"]) for row in resolved],
            config.seed + 101,
            config.bootstrap_samples,
        ),
        "puct_oracle_match_rate": _bootstrap_mean_ci(
            [float(row["puct_choice"] == row["oracle_action_index"]) for row in resolved],
            config.seed + 102,
            config.bootstrap_samples,
        ),
        "prior_oracle_match_rate": _bootstrap_mean_ci(
            [float(row["prior_action_index"] == row["oracle_action_index"]) for row in resolved],
            config.seed + 103,
            config.bootstrap_samples,
        ),
        "mean_return_range": statistics.fmean(row["return_range"] for row in rows) if rows else 0.0,
        "mean_determinization_best_count": statistics.fmean(
            row["determinization_best_action_count"] for row in resolved
        ) if resolved else 0.0,
        "determinization_best_action_disagreement_rate": _bootstrap_mean_ci(
            [float(row["determinization_best_action_count"] > 1) for row in resolved],
            config.seed + 104,
            config.bootstrap_samples,
        ),
        "by_phase": {phase: dict(sorted(counts.items())) for phase, counts in sorted(by_phase.items())},
    }


def run_residual_error_audit(config: ResidualErrorAuditConfig) -> dict[str, Any]:
    _validate_config(config)
    rows, selection = _select_rows(config)
    env = SplendorEnv()
    codec = ActionCodec()
    scorer = _LegalActionScorer(config.policy_checkpoint, config.device, "legal_action_attention_scorer")
    output_rows: list[dict[str, Any]] = []
    for state_position, payload in enumerate(rows):
        root_state = deserialize_state_snapshot(payload["state_snapshot"])
        root_player = root_state.current_player
        root_legal_actions = env.legal_actions(root_state)
        parent_scores = scorer.score_actions(root_state, root_legal_actions)
        policy_candidates = select_root_candidates(
            parent_scores,
            policy_top_k=config.policy_top_k,
            max_candidates=config.max_policy_candidates,
        )
        action_data: dict[int, dict[str, Any]] = {
            codec.encode(root_state, action): {
                "action": action,
                "returns": [],
                "returns_by_determinization": [],
            }
            for action in root_legal_actions
        }
        for determinization_index in range(config.determinizations):
            determinized = determinize_for_observer(
                root_state,
                root_player,
                random.Random(config.seed + state_position * 1_000_003 + determinization_index),
            )
            determinized_actions = env.legal_actions(determinized)
            action_by_index = {
                action_index: action
                for action_index, action in zip(
                    codec.legal_action_indices(determinized, determinized_actions),
                    determinized_actions,
                    strict=True,
                )
            }
            for action_index, data in action_data.items():
                action = action_by_index.get(action_index)
                if action is None:
                    continue
                returns = [
                    rollout_action_return(
                        env=env,
                        scorer=scorer,
                        state=determinized,
                        root_action=action,
                        root_player=root_player,
                        seed=(
                            config.seed
                            + state_position * 10_000_019
                            + determinization_index * 100_003
                            + rollout_index
                        ),
                        epsilon=config.rollout_epsilon,
                        temperature=config.rollout_temperature,
                        top_k=config.rollout_top_k,
                        max_additional_turns=config.rollout_max_additional_turns,
                        outcome_weight=config.outcome_weight,
                        score_margin_scale=config.score_margin_scale,
                    )
                    for rollout_index in range(config.rollouts_per_determinization)
                ]
                data["returns"].extend(returns)
                data["returns_by_determinization"].append(returns)
        actions = [
            {
                "action_index": action_index,
                "action_payload": serialize_action(data["action"]),
                "rollout_return": statistics.fmean(data["returns"]),
                "rollout_standard_error": (
                    statistics.stdev(data["returns"]) / math.sqrt(len(data["returns"]))
                    if len(data["returns"]) > 1
                    else 0.0
                ),
                "determinization_means": [statistics.fmean(values) for values in data["returns_by_determinization"]],
            }
            for action_index, data in action_data.items()
            if data["returns"]
        ]
        if len(actions) < 2:
            continue
        scores_by_index = {item.action_index: item.score for item in parent_scores}
        for action in actions:
            action["prior_score"] = scores_by_index[action["action_index"]]
        actions.sort(key=lambda item: int(item["action_index"]))
        oracle_action = _top_action_index(actions, "rollout_return")
        prior_action = _top_action_index(actions, "prior_score")
        ordered_by_return = sorted(actions, key=lambda item: (-float(item["rollout_return"]), int(item["action_index"])))
        best, runner_up = ordered_by_return[:2]
        oracle_returns = action_data[oracle_action]["returns"]
        runner_up_returns = action_data[runner_up["action_index"]]["returns"]
        best_margin, best_margin_se = _paired_margin(oracle_returns, runner_up_returns)
        resolved = (
            best_margin >= config.min_return_range
            and best_margin > config.confidence_z * best_margin_se
        )
        determinization_best_actions = {
            _top_action_index(
                [
                    {
                        "action_index": action["action_index"],
                        "determinization_value": action["determinization_means"][index],
                    }
                    for action in actions
                    if len(action["determinization_means"]) > index
                ],
                "determinization_value",
            )
            for index in range(config.determinizations)
        }
        legal_indices = {int(item["action_index"]) for item in actions}
        puct_choice = _root_visit_choice(payload, legal_indices)
        policy_candidate_indices = {item.action_index for item in policy_candidates}
        attribution = _classify_state(
            resolved=resolved,
            puct_choice=puct_choice,
            oracle_action=oracle_action,
            prior_action=prior_action,
        )
        output_rows.append(
            {
                "game_seed": payload.get("game_seed"),
                "turn_index": payload.get("turn_index"),
                "player_id": root_player,
                "phase": _phase(int(payload["turn_index"])),
                "legal_action_count": len(actions),
                "policy_candidate_indices": sorted(policy_candidate_indices),
                "oracle_action_index": oracle_action,
                "prior_action_index": prior_action,
                "puct_choice": puct_choice,
                "oracle_in_policy_candidates": oracle_action in policy_candidate_indices,
                "return_range": float(best["rollout_return"]) - float(ordered_by_return[-1]["rollout_return"]),
                "best_margin_vs_runner_up": best_margin,
                "best_margin_standard_error": best_margin_se,
                "resolved": resolved,
                "determinization_best_actions": sorted(determinization_best_actions),
                "determinization_best_action_count": len(determinization_best_actions),
                "attribution": attribution,
                "actions": actions,
            }
        )
        if config.log_every_states > 0 and (state_position + 1) % config.log_every_states == 0:
            print(f"[residual-error-audit] evaluated={state_position + 1}/{len(rows)} kept={len(output_rows)}")
    output = {
        "audit": "puct_residual_error_attribution",
        "config": {
            **asdict(config),
            "replay_paths": [str(path) for path in config.replay_paths],
            "policy_checkpoint": str(config.policy_checkpoint),
            "output_path": str(config.output_path),
            "exclude_audit_path": (
                str(config.exclude_audit_path)
                if config.exclude_audit_path is not None
                else None
            ),
        },
        "selection": selection,
        "summary": _summarize(output_rows, config),
        "states": output_rows,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _validate_config(config: ResidualErrorAuditConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    for name, value in (
        ("max_states", config.max_states),
        ("determinizations", config.determinizations),
        ("rollouts_per_determinization", config.rollouts_per_determinization),
        ("policy_top_k", config.policy_top_k),
        ("max_policy_candidates", config.max_policy_candidates),
        ("rollout_top_k", config.rollout_top_k),
        ("rollout_max_additional_turns", config.rollout_max_additional_turns),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.min_turn_index < 0 or config.min_return_range < 0.0 or config.confidence_z < 0.0:
        raise ValueError("Turn, return-range, and confidence settings must be non-negative.")
    if not 0.0 <= config.rollout_epsilon <= 1.0 or not 0.0 <= config.outcome_weight <= 1.0:
        raise ValueError("Rollout epsilon and outcome weight must be in [0, 1].")
    if config.rollout_temperature <= 0.0 or config.score_margin_scale <= 0.0 or config.bootstrap_samples < 0:
        raise ValueError("Rollout temperature, margin scale, and bootstrap samples are invalid.")


def main() -> None:
    args = build_parser().parse_args()
    output = run_residual_error_audit(
        ResidualErrorAuditConfig(
            replay_paths=tuple(Path(path) for path in args.replay_path),
            policy_checkpoint=Path(args.policy_checkpoint),
            output_path=Path(args.output_path),
            exclude_audit_path=(
                Path(args.exclude_audit_path)
                if args.exclude_audit_path is not None
                else None
            ),
            max_states=args.max_states,
            determinizations=args.determinizations,
            rollouts_per_determinization=args.rollouts_per_determinization,
            min_turn_index=args.min_turn_index,
            policy_top_k=args.policy_top_k,
            max_policy_candidates=args.max_policy_candidates,
            min_return_range=args.min_return_range,
            confidence_z=args.confidence_z,
            rollout_epsilon=args.rollout_epsilon,
            rollout_temperature=args.rollout_temperature,
            rollout_top_k=args.rollout_top_k,
            rollout_max_additional_turns=args.rollout_max_additional_turns,
            outcome_weight=args.outcome_weight,
            score_margin_scale=args.score_margin_scale,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            device=args.device,
            log_every_states=args.log_every_states,
        )
    )
    print(
        "[residual-error-audit]"
        f" evaluated={output['summary']['states_evaluated']}"
        f" resolved={output['summary']['resolved_states']}"
        f" output={args.output_path}"
    )


if __name__ == "__main__":
    main()
