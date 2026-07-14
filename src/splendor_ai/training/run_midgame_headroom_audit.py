"""Two-stage all-legal counterfactual audit for mid-game policy headroom."""

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
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer
from splendor_ai.training.run_residual_error_audit import (
    _bootstrap_mean_ci,
    _paired_margin,
    _root_visit_choice,
    _top_action_index,
)
from splendor_ai.training.run_rollout_action_credit import (
    rollout_action_return,
    select_root_candidates,
)


MIDGAME_BANDS = (
    ("mid_15_24", 15, 24),
    ("mid_25_32", 25, 32),
    ("mid_33_40", 33, 40),
)


@dataclass(frozen=True, slots=True)
class MidgameHeadroomAuditConfig:
    replay_paths: tuple[Path, ...]
    policy_checkpoint: Path
    output_path: Path
    checkpoint_path: Path | None = None
    screening_states_per_band: int = 20
    refined_states_per_band: int = 10
    screening_determinizations: int = 2
    screening_rollouts_per_determinization: int = 2
    refinement_determinizations: int = 3
    refinement_rollouts_per_determinization: int = 6
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
    seed: int = 73
    device: str = "cpu"
    log_every_states: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a phase-stratified two-stage all-legal mid-game matched-return "
            "audit for policy/search headroom."
        )
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional resumable progress JSON; defaults beside --output-path.",
    )
    parser.add_argument("--screening-states-per-band", type=int, default=20)
    parser.add_argument("--refined-states-per-band", type=int, default=10)
    parser.add_argument("--screening-determinizations", type=int, default=2)
    parser.add_argument("--screening-rollouts-per-determinization", type=int, default=2)
    parser.add_argument("--refinement-determinizations", type=int, default=3)
    parser.add_argument("--refinement-rollouts-per-determinization", type=int, default=6)
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
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-states", type=int, default=10)
    return parser


def _band(turn_index: int) -> str | None:
    for name, start, end in MIDGAME_BANDS:
        if start <= turn_index <= end:
            return name
    return None


def _state_key(payload: dict[str, Any]) -> tuple[int, int, int]:
    return int(payload["game_seed"]), int(payload["turn_index"]), int(payload["player_id"])


def _select_screening_rows(config: MidgameHeadroomAuditConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    for path in config.replay_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_read += 1
                payload = json.loads(line)
                band = _band(int(payload.get("turn_index", -1)))
                if (
                    band is not None
                    and isinstance(payload.get("state_snapshot"), dict)
                    and isinstance(payload.get("root_visit_distribution"), list)
                ):
                    by_band[band].append(payload)
    rng = random.Random(config.seed)
    selected: list[dict[str, Any]] = []
    selected_by_band: dict[str, int] = {}
    for name, _start, _end in MIDGAME_BANDS:
        rows = by_band[name]
        rng.shuffle(rows)
        chosen = rows[: config.screening_states_per_band]
        selected.extend(chosen)
        selected_by_band[name] = len(chosen)
    rng.shuffle(selected)
    return selected, {
        "rows_read": rows_read,
        "eligible_rows_by_band": {name: len(rows) for name, rows in by_band.items()},
        "screening_states_selected": len(selected),
        "screening_states_selected_by_band": selected_by_band,
    }


def _evaluate_state(
    *,
    payload: dict[str, Any],
    scorer: _LegalActionScorer,
    env: SplendorEnv,
    codec: ActionCodec,
    determinizations: int,
    rollouts_per_determinization: int,
    seed: int,
    config: MidgameHeadroomAuditConfig,
) -> dict[str, Any] | None:
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
        codec.encode(root_state, action): {"action": action, "returns": [], "returns_by_determinization": []}
        for action in root_legal_actions
    }
    for determinization_index in range(determinizations):
        determinized = determinize_for_observer(
            root_state,
            root_player,
            random.Random(seed + determinization_index),
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
                    seed=seed + determinization_index * 100_003 + rollout_index,
                    epsilon=config.rollout_epsilon,
                    temperature=config.rollout_temperature,
                    top_k=config.rollout_top_k,
                    max_additional_turns=config.rollout_max_additional_turns,
                    outcome_weight=config.outcome_weight,
                    score_margin_scale=config.score_margin_scale,
                )
                for rollout_index in range(rollouts_per_determinization)
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
        return None
    scores_by_index = {item.action_index: item.score for item in parent_scores}
    for action in actions:
        action["prior_score"] = scores_by_index[action["action_index"]]
    actions.sort(key=lambda item: int(item["action_index"]))
    oracle_action = _top_action_index(actions, "rollout_return")
    prior_action = _top_action_index(actions, "prior_score")
    ordered = sorted(actions, key=lambda item: (-float(item["rollout_return"]), int(item["action_index"])))
    best, runner_up = ordered[:2]
    best_margin, best_margin_standard_error = _paired_margin(
        action_data[oracle_action]["returns"],
        action_data[runner_up["action_index"]]["returns"],
    )
    resolved = (
        best_margin >= config.min_return_range
        and best_margin > config.confidence_z * best_margin_standard_error
    )
    determinization_best_actions = {
        _top_action_index(
            [
                {"action_index": action["action_index"], "value": action["determinization_means"][index]}
                for action in actions
                if len(action["determinization_means"]) > index
            ],
            "value",
        )
        for index in range(determinizations)
    }
    legal_indices = {int(action["action_index"]) for action in actions}
    return {
        "game_seed": payload["game_seed"],
        "turn_index": payload["turn_index"],
        "player_id": root_player,
        "phase": _band(int(payload["turn_index"])),
        "legal_action_count": len(actions),
        "policy_candidate_indices": sorted(item.action_index for item in policy_candidates),
        "oracle_action_index": oracle_action,
        "prior_action_index": prior_action,
        "puct_choice": _root_visit_choice(payload, legal_indices),
        "return_range": float(best["rollout_return"]) - float(ordered[-1]["rollout_return"]),
        "best_margin_vs_runner_up": best_margin,
        "best_margin_standard_error": best_margin_standard_error,
        "resolved": resolved,
        "determinization_best_actions": sorted(determinization_best_actions),
        "determinization_best_action_count": len(determinization_best_actions),
        "actions": actions,
    }


def _summarize(rows: list[dict[str, Any]], config: MidgameHeadroomAuditConfig) -> dict[str, Any]:
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_band[str(row["phase"])].append(row)

    def summarize_band(items: list[dict[str, Any]], seed_offset: int) -> dict[str, Any]:
        resolved = [item for item in items if item["resolved"]]
        return {
            "states": len(items),
            "resolved_states": len(resolved),
            "mean_return_range": statistics.fmean(item["return_range"] for item in items) if items else 0.0,
            "prior_oracle_match_rate": _bootstrap_mean_ci(
                [float(item["prior_action_index"] == item["oracle_action_index"]) for item in resolved],
                config.seed + seed_offset,
                config.bootstrap_samples,
            ),
            "puct_oracle_match_rate": _bootstrap_mean_ci(
                [float(item["puct_choice"] == item["oracle_action_index"]) for item in resolved],
                config.seed + seed_offset + 1,
                config.bootstrap_samples,
            ),
        }

    return {
        "states": len(rows),
        "resolved_states": sum(row["resolved"] for row in rows),
        "by_band": {
            name: summarize_band(items, position * 10)
            for position, (name, items) in enumerate(sorted(by_band.items()))
        },
    }


def _write_checkpoint(
    path: Path,
    *,
    selection: dict[str, Any],
    screening: list[dict[str, Any]],
    refined: list[dict[str, Any]],
    complete: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "audit": "midgame_headroom_two_stage_progress",
                "selection": selection,
                "screening_states": screening,
                "states": refined,
                "complete": complete,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_midgame_headroom_audit(config: MidgameHeadroomAuditConfig) -> dict[str, Any]:
    _validate_config(config)
    screening_payloads, selection = _select_screening_rows(config)
    env = SplendorEnv()
    codec = ActionCodec()
    scorer = _LegalActionScorer(config.policy_checkpoint, config.device, "legal_action_attention_scorer")
    checkpoint_path = config.checkpoint_path or config.output_path.with_suffix(".checkpoint.json")
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {}
    )
    screening: list[dict[str, Any]] = list(checkpoint.get("screening_states", []))
    refined: list[dict[str, Any]] = list(checkpoint.get("states", []))
    screening_keys = {_state_key(row) for row in screening}
    refined_keys = {_state_key(row) for row in refined}
    payload_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    for position, payload in enumerate(screening_payloads):
        payload_by_key[_state_key(payload)] = payload
        if _state_key(payload) in screening_keys:
            continue
        row = _evaluate_state(
            payload=payload,
            scorer=scorer,
            env=env,
            codec=codec,
            determinizations=config.screening_determinizations,
            rollouts_per_determinization=config.screening_rollouts_per_determinization,
            seed=config.seed + position * 10_000_019,
            config=config,
        )
        if row is not None:
            screening.append(row)
            screening_keys.add(_state_key(row))
            _write_checkpoint(
                checkpoint_path,
                selection=selection,
                screening=screening,
                refined=refined,
            )
        if config.log_every_states > 0 and (position + 1) % config.log_every_states == 0:
            print(f"[midgame-headroom] screening={position + 1}/{len(screening_payloads)} kept={len(screening)}")
    selected_for_refinement: list[dict[str, Any]] = []
    refined_selection: dict[str, int] = {}
    screening_by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in screening:
        if row["return_range"] >= config.min_return_range:
            screening_by_band[str(row["phase"])].append(row)
    for name, _start, _end in MIDGAME_BANDS:
        candidates = sorted(
            screening_by_band[name],
            key=lambda item: (-float(item["return_range"]), _state_key(item)),
        )[: config.refined_states_per_band]
        selected_for_refinement.extend(candidates)
        refined_selection[name] = len(candidates)
    for position, screening_row in enumerate(selected_for_refinement):
        if _state_key(screening_row) in refined_keys:
            continue
        payload = payload_by_key[_state_key(screening_row)]
        row = _evaluate_state(
            payload=payload,
            scorer=scorer,
            env=env,
            codec=codec,
            determinizations=config.refinement_determinizations,
            rollouts_per_determinization=config.refinement_rollouts_per_determinization,
            seed=config.seed + 5_000_000_000 + position * 10_000_019,
            config=config,
        )
        if row is not None:
            row["screening_return_range"] = screening_row["return_range"]
            refined.append(row)
            refined_keys.add(_state_key(row))
            _write_checkpoint(
                checkpoint_path,
                selection=selection | {"refined_states_selected_by_band": refined_selection},
                screening=screening,
                refined=refined,
            )
        if config.log_every_states > 0 and (position + 1) % config.log_every_states == 0:
            print(f"[midgame-headroom] refined={position + 1}/{len(selected_for_refinement)} kept={len(refined)}")
    output = {
        "audit": "midgame_headroom_two_stage",
        "config": {
            **asdict(config),
            "replay_paths": [str(path) for path in config.replay_paths],
            "policy_checkpoint": str(config.policy_checkpoint),
            "output_path": str(config.output_path),
            "checkpoint_path": str(checkpoint_path),
            "bands": [{"name": name, "start": start, "end": end} for name, start, end in MIDGAME_BANDS],
        },
        "selection": selection | {"refined_states_selected_by_band": refined_selection},
        "screening_summary": _summarize(screening, config),
        "refined_summary": _summarize(refined, config),
        "screening_states": screening,
        "states": refined,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_checkpoint(
        checkpoint_path,
        selection=output["selection"],
        screening=screening,
        refined=refined,
        complete=True,
    )
    return output


def _validate_config(config: MidgameHeadroomAuditConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    for name, value in (
        ("screening_states_per_band", config.screening_states_per_band),
        ("refined_states_per_band", config.refined_states_per_band),
        ("screening_determinizations", config.screening_determinizations),
        ("screening_rollouts_per_determinization", config.screening_rollouts_per_determinization),
        ("refinement_determinizations", config.refinement_determinizations),
        ("refinement_rollouts_per_determinization", config.refinement_rollouts_per_determinization),
        ("policy_top_k", config.policy_top_k),
        ("max_policy_candidates", config.max_policy_candidates),
        ("rollout_top_k", config.rollout_top_k),
        ("rollout_max_additional_turns", config.rollout_max_additional_turns),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.min_return_range < 0.0 or config.confidence_z < 0.0:
        raise ValueError("Return-range and confidence settings must be non-negative.")
    if not 0.0 <= config.rollout_epsilon <= 1.0 or not 0.0 <= config.outcome_weight <= 1.0:
        raise ValueError("Rollout epsilon and outcome weight must be in [0, 1].")
    if config.rollout_temperature <= 0.0 or config.score_margin_scale <= 0.0 or config.bootstrap_samples < 0:
        raise ValueError("Rollout temperature, margin scale, and bootstrap samples are invalid.")


def main() -> None:
    args = build_parser().parse_args()
    output = run_midgame_headroom_audit(
        MidgameHeadroomAuditConfig(
            replay_paths=tuple(Path(path) for path in args.replay_path),
            policy_checkpoint=Path(args.policy_checkpoint),
            output_path=Path(args.output_path),
            checkpoint_path=(Path(args.checkpoint_path) if args.checkpoint_path is not None else None),
            screening_states_per_band=args.screening_states_per_band,
            refined_states_per_band=args.refined_states_per_band,
            screening_determinizations=args.screening_determinizations,
            screening_rollouts_per_determinization=args.screening_rollouts_per_determinization,
            refinement_determinizations=args.refinement_determinizations,
            refinement_rollouts_per_determinization=args.refinement_rollouts_per_determinization,
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
        "[midgame-headroom]"
        f" screening={output['screening_summary']['states']}"
        f" refined={output['refined_summary']['states']}"
        f" refined_resolved={output['refined_summary']['resolved_states']}"
        f" output={args.output_path}"
    )


if __name__ == "__main__":
    main()
