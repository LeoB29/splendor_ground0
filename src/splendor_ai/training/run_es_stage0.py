"""Corrected antithetic-ES Stage 0 for the frozen legal ``035`` policy.

The experiment does not select a promising individual direction.  It estimates
one aggregate antithetic gradient from paired game fitness, materializes a
single normalized update, then evaluates that update on fresh paired seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
from typing import Any

import torch

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.parallel_paired import PairedCheckpointConfig, run_paired_checkpoint_benchmark
from splendor_ai.training.es_adapter import RankAdapterSpec, adapter_parameter_count, normalized_direction, write_adapter_checkpoint
from splendor_ai.training.replay import deserialize_state_snapshot


@dataclass(frozen=True, slots=True)
class EsStage0Config:
    base_checkpoint: Path
    replay_paths: tuple[Path, ...]
    output_dir: Path
    output_path: Path
    directions: int = 16
    rank: int = 4
    adapter_seed: int = 61
    direction_seed: int = 62
    calibration_states: int = 256
    calibration_sigmas: tuple[float, ...] = (0.25, 0.75)
    target_action_change_rate: float = 0.10
    fitness_pairs_per_sign: int = 50
    fresh_evaluation_pairs: int = 100
    seed_start: int = 71000
    workers: int = 4
    bootstrap_samples: int = 20_000
    score_margin_scale: float = 5.0
    device: str = "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the aggregate-gradient antithetic ES Stage 0 experiment."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--adapter-seed", type=int, default=61)
    parser.add_argument("--direction-seed", type=int, default=62)
    parser.add_argument("--calibration-states", type=int, default=256)
    parser.add_argument("--calibration-sigma", action="append", type=float, default=None)
    parser.add_argument("--target-action-change-rate", type=float, default=0.10)
    parser.add_argument("--fitness-pairs-per-sign", type=int, default=50)
    parser.add_argument("--fresh-evaluation-pairs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=71000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> EsStage0Config:
    return EsStage0Config(
        base_checkpoint=Path(args.base_checkpoint),
        replay_paths=tuple(Path(path) for path in args.replay_path),
        output_dir=Path(args.output_dir),
        output_path=Path(args.output_path),
        directions=args.directions,
        rank=args.rank,
        adapter_seed=args.adapter_seed,
        direction_seed=args.direction_seed,
        calibration_states=args.calibration_states,
        calibration_sigmas=tuple(args.calibration_sigma or (0.25, 0.75)),
        target_action_change_rate=args.target_action_change_rate,
        fitness_pairs_per_sign=args.fitness_pairs_per_sign,
        fresh_evaluation_pairs=args.fresh_evaluation_pairs,
        seed_start=args.seed_start,
        workers=args.workers,
        bootstrap_samples=args.bootstrap_samples,
        score_margin_scale=args.score_margin_scale,
        device=args.device,
    )


def run_es_stage0(config: EsStage0Config) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(config.base_checkpoint, map_location="cpu", weights_only=True)
    spec = RankAdapterSpec(rank=config.rank, seed=config.adapter_seed)
    parameter_count = adapter_parameter_count(checkpoint["model_state_dict"], spec)
    directions = [
        normalized_direction(parameter_count, config.direction_seed + index)
        for index in range(config.directions)
    ]
    snapshots = _sample_snapshots(config.replay_paths, config.calibration_states, config.direction_seed)
    calibration = _calibrate_sigma(config, spec, directions, snapshots)
    if not calibration["passed"]:
        payload = {
            "format": "antithetic_es_stage0_v1",
            "config": _config_payload(config),
            "adapter": {"parameter_count": parameter_count, **asdict(spec)},
            "calibration": calibration,
            "decision_rule": {
                "pass": False,
                "rule": (
                    "calibration must produce at least the minimum action-change rate "
                    "before any paired-game fitness evaluation"
                ),
            },
        }
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("[es-stage0] calibration did not move behavior; no fitness games were run.")
        return payload
    selected_sigma = float(calibration["selected_sigma"])

    candidates: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(directions):
        for sign in (1, -1):
            theta = direction * (sign * selected_sigma)
            path = config.output_dir / "candidates" / f"direction_{direction_index:02d}_{'plus' if sign > 0 else 'minus'}.pt"
            write_adapter_checkpoint(
                config.base_checkpoint,
                path,
                theta,
                spec,
                provenance={
                    "stage": "stage0_antithetic_fitness",
                    "direction_index": direction_index,
                    "sign": sign,
                    "sigma": selected_sigma,
                },
            )
            benchmark = run_paired_checkpoint_benchmark(
                _benchmark_config(
                    config,
                    candidate=path,
                    baseline=config.base_checkpoint,
                    output_path=config.output_dir / "fitness" / f"direction_{direction_index:02d}_{sign:+d}.json",
                    pairs=config.fitness_pairs_per_sign,
                    seed_start=config.seed_start + direction_index * 10_000,
                )
            )
            candidates.append(
                {
                    "direction_index": direction_index,
                    "sign": sign,
                    "checkpoint": str(path),
                    "fitness": benchmark["paired_statistics"]["bounded_utility"],
                    "score_share": benchmark["paired_statistics"]["score_share"],
                }
            )

    update = _aggregate_update(directions, candidates, selected_sigma)
    update_path = config.output_dir / "aggregate_update.pt"
    write_adapter_checkpoint(
        config.base_checkpoint,
        update_path,
        update,
        spec,
        provenance={
            "stage": "stage0_aggregate_update",
            "sigma": selected_sigma,
            "directions": config.directions,
        },
    )
    fresh = run_paired_checkpoint_benchmark(
        _benchmark_config(
            config,
            candidate=update_path,
            baseline=config.base_checkpoint,
            output_path=config.output_dir / "fresh_aggregate_update_vs_base.json",
            pairs=config.fresh_evaluation_pairs,
            seed_start=config.seed_start + 1_000_000,
        )
    )
    payload = {
        "format": "antithetic_es_stage0_v1",
        "config": _config_payload(config),
        "adapter": {"parameter_count": parameter_count, **asdict(spec)},
        "calibration": calibration,
        "fitness_candidates": candidates,
        "aggregate_update": {
            "checkpoint": str(update_path),
            "theta_l2_norm": float(update.norm().item()),
            "fresh_paired_evaluation": fresh,
        },
        "decision_rule": {
            "pass": (
                fresh["paired_statistics"]["bounded_utility"]["ci95_low"] > 0.0
                and fresh["timed_out_games"] == 0
                and fresh["stalled_games"] == 0
            ),
            "rule": "fresh bounded-utility lower CI > 0 with no timed-out or stalled games",
        },
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[es-stage0] wrote results: {config.output_path}")
    return payload


def _calibrate_sigma(
    config: EsStage0Config,
    spec: RankAdapterSpec,
    directions: list[torch.Tensor],
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = LegalActionAttentionScorerBot(config.base_checkpoint, device=config.device)
    rows: list[dict[str, Any]] = []
    # Calibration is behavioral only.  It never sees game outcomes and merely
    # prevents an arbitrarily tiny or destructive perturbation scale.
    for sigma in config.calibration_sigmas:
        rates: list[float] = []
        for direction_index, direction in enumerate(directions):
            path = config.output_dir / "calibration" / f"sigma_{sigma:g}_direction_{direction_index:02d}.pt"
            write_adapter_checkpoint(
                config.base_checkpoint,
                path,
                direction * sigma,
                spec,
                provenance={"stage": "behavior_calibration", "sigma": sigma, "direction_index": direction_index},
            )
            candidate = LegalActionAttentionScorerBot(path, device=config.device)
            rates.append(_action_change_rate(reference, candidate, snapshots))
        rows.append({"sigma": sigma, "mean_action_change_rate": statistics.fmean(rates), "per_direction": rates})
    selected = min(rows, key=lambda row: abs(float(row["mean_action_change_rate"]) - config.target_action_change_rate))
    minimum_change_rate = min(0.01, config.target_action_change_rate)
    return {
        "target_action_change_rate": config.target_action_change_rate,
        "minimum_action_change_rate": minimum_change_rate,
        "rows": rows,
        "selected_sigma": selected["sigma"],
        "passed": float(selected["mean_action_change_rate"]) >= minimum_change_rate,
    }


def _action_change_rate(
    reference: LegalActionAttentionScorerBot,
    candidate: LegalActionAttentionScorerBot,
    snapshots: list[dict[str, Any]],
) -> float:
    env = SplendorEnv(seed=0)
    changes = 0
    for snapshot in snapshots:
        state = deserialize_state_snapshot(snapshot)
        legal = env.legal_actions(state)
        if reference.choose_action(env, state, legal) != candidate.choose_action(env, state, legal):
            changes += 1
    return changes / len(snapshots) if snapshots else 0.0


def _sample_snapshots(paths: tuple[Path, ...], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                snapshot = payload.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    continue
                seen += 1
                if len(reservoir) < count:
                    reservoir.append(snapshot)
                else:
                    index = rng.randrange(seen)
                    if index < count:
                        reservoir[index] = snapshot
    if not reservoir:
        raise ValueError("No replay state snapshots were found for ES calibration.")
    return reservoir


def _aggregate_update(
    directions: list[torch.Tensor],
    candidates: list[dict[str, Any]],
    sigma: float,
) -> torch.Tensor:
    by_direction: dict[int, dict[int, float]] = {}
    for row in candidates:
        by_direction.setdefault(int(row["direction_index"]), {})[int(row["sign"])] = float(row["fitness"]["mean"])
    gradient = torch.zeros_like(directions[0])
    for index, direction in enumerate(directions):
        signs = by_direction.get(index, {})
        if set(signs) != {1, -1}:
            raise RuntimeError(f"Incomplete antithetic fitness pair for direction {index}.")
        gradient += ((signs[1] - signs[-1]) / (2.0 * sigma)) * direction
    gradient /= len(directions)
    if float(gradient.norm().item()) == 0.0:
        return gradient
    # A normalized one-step update has the same adapter norm as the calibrated
    # perturbations, so the fresh gate tests signal direction, not step size.
    return sigma * gradient / gradient.norm()


def _benchmark_config(
    config: EsStage0Config,
    candidate: Path,
    baseline: Path,
    output_path: Path,
    pairs: int,
    seed_start: int,
) -> PairedCheckpointConfig:
    return PairedCheckpointConfig(
        candidate_checkpoint=candidate,
        baseline_checkpoint=baseline,
        output_path=output_path,
        checkpoint_path=output_path.with_suffix(output_path.suffix + ".progress.json"),
        pairs=pairs,
        seed_start=seed_start,
        workers=config.workers,
        bootstrap_samples=config.bootstrap_samples,
        score_margin_scale=config.score_margin_scale,
        device=config.device,
    )


def _config_payload(config: EsStage0Config) -> dict[str, Any]:
    payload = asdict(config)
    payload["base_checkpoint"] = str(config.base_checkpoint)
    payload["replay_paths"] = [str(path) for path in config.replay_paths]
    payload["output_dir"] = str(config.output_dir)
    payload["output_path"] = str(config.output_path)
    return payload


def _validate_config(config: EsStage0Config) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(config.base_checkpoint)
    if config.directions <= 0 or config.rank <= 0 or config.calibration_states <= 0:
        raise ValueError("directions, rank, and calibration_states must be positive.")
    if config.fitness_pairs_per_sign <= 0 or config.fresh_evaluation_pairs <= 0:
        raise ValueError("ES game counts must be positive paired-seed counts.")
    if any(sigma <= 0.0 for sigma in config.calibration_sigmas):
        raise ValueError("Calibration sigmas must be positive.")
    if not 0.0 <= config.target_action_change_rate <= 1.0:
        raise ValueError("target_action_change_rate must be in [0, 1].")


def main() -> None:
    run_es_stage0(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()
