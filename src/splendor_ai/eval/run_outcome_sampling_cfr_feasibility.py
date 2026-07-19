"""Run preregistered experiment 048: outcome-sampling CFR feasibility."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any

from splendor_ai.cfr import (
    ExactCFRSolver,
    FrozenEntityTargetPolicy,
    FullGameOutcomeSampler,
    OutcomeSamplingMCCFRSolver,
    ReducedSplendorCFRGame,
    ReservoirMemory,
    exact_fixed_policy_regret_increments,
    profile_exploitability,
    sample_reduced_outcome_updates,
)
from splendor_ai.eval.run_full_game_deep_cfr_feasibility import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    run_privacy_and_invariance_audits,
)


EXACT_HASH_044 = "b8184ec7c9dbfa3611d014f98c674efbeae27728429b2fa90c7939e1bbe043ab"
ADAPTIVE_SEEDS = (480046, 480047, 480048)
FULL_GAME_SEEDS = tuple(range(480000, 480256))


def run_experiment(
    *,
    uniform_samples: int = 200_000,
    adaptive_iterations: int = 100_000,
    full_game_seeds: tuple[int, ...] = FULL_GAME_SEEDS,
    checkpoint: Path = CHECKPOINT,
    output_dir: Path = Path("outputs/outcome_sampling_cfr_feasibility_048"),
    enforce_frozen_counts: bool = True,
) -> dict[str, Any]:
    print("048 exact reduced reference and fixed-policy estimator audit", flush=True)
    exact_game = ReducedSplendorCFRGame()
    exact_solver = ExactCFRSolver(exact_game)
    exact_payload = exact_solver.run(10_000, snapshot_iterations=(100,))
    exact_hash = _canonical_hash(exact_payload)
    exact_keys = set(exact_solver.nodes)

    uniform_results = [
        _uniform_estimator_audit(traverser, uniform_samples, seed=480048)
        for traverser in (0, 1)
    ]
    uniform_gate = all(
        row["eligible_information_set_coverage"] >= 0.95
        and row["mean_absolute_component_error"] <= 0.010
        and row["max_absolute_component_error"] <= 0.050
        and row["root_band_max_component_error"] <= 0.010
        and row["illegal_actions"] == 0
        and row["finite"]
        for row in uniform_results
    )

    adaptive_results: list[dict[str, Any]] = []
    first_payload: dict[str, Any] | None = None
    for index, seed in enumerate(ADAPTIVE_SEEDS, start=1):
        print(f"048 adaptive reduced seed {seed} ({index}/3)", flush=True)
        game = ReducedSplendorCFRGame()
        solver = OutcomeSamplingMCCFRSolver(game, seed=seed, exploration=0.1)
        solver_payload = solver.run(adaptive_iterations, snapshot_iterations=(100,))
        if first_payload is None:
            first_payload = solver_payload
        analysis = _compact_exploitability(
            profile_exploitability(game, solver.current_policy(), solver.current_policy())
        )
        adaptive_results.append(
            {
                "seed": seed,
                "payload_hash": _canonical_hash(solver_payload),
                "information_sets": len(solver.nodes),
                "exact_information_set_coverage": len(exact_keys & solver.nodes.keys()) / len(exact_keys),
                "normalized_positive_regret": solver_payload["normalized_positive_regret"],
                "max_log_importance_weight": solver_payload["max_log_importance_weight"],
                "sampled_trajectories": solver_payload["sampled_trajectories"],
                "sampled_updates": solver_payload["sampled_updates"],
                "illegal_actions": solver_payload["illegal_actions"],
                "unstable_action_sets": solver_payload["unstable_action_sets"],
                "current_profile": analysis,
                "finite": _finite(solver_payload) and _finite(analysis),
            }
        )
    print("048 deterministic repeat of adaptive seed 480046", flush=True)
    repeat_game = ReducedSplendorCFRGame()
    repeat_solver = OutcomeSamplingMCCFRSolver(repeat_game, seed=ADAPTIVE_SEEDS[0], exploration=0.1)
    repeat_payload = repeat_solver.run(adaptive_iterations, snapshot_iterations=(100,))
    if first_payload is None:
        raise AssertionError("No adaptive payload was produced.")
    first_repeat_identical = first_payload == repeat_payload
    adaptive_exploitabilities = [row["current_profile"]["exploitability"] for row in adaptive_results]
    adaptive_gate = (
        first_repeat_identical
        and all(
            row["illegal_actions"] == 0
            and row["unstable_action_sets"] == 0
            and row["finite"]
            and row["exact_information_set_coverage"] >= 0.95
            and all(value < 0.05 for value in row["normalized_positive_regret"].values())
            and row["current_profile"]["exploitability"] <= 0.030
            and row["current_profile"]["best_response_integrity"]
            for row in adaptive_results
        )
        and median(adaptive_exploitabilities) <= 0.015
    )

    print("048 full-game outcome trajectories", flush=True)
    target_policy = FrozenEntityTargetPolicy(checkpoint, temperature=1.0)
    sampler = FullGameOutcomeSampler(
        target_policy,
        exploration=0.1,
        repetition_limit=4,
        no_progress_limit=60,
        turn_limit=200,
    )
    primary_reservoir = ReservoirMemory(capacity=512, seed=480048)
    repeated_reservoir = ReservoirMemory(capacity=512, seed=480048)
    trajectory_results: list[dict[str, Any]] = []
    all_log_weights: list[float] = []
    first_half_logs: list[float] = []
    second_half_logs: list[float] = []
    phase_logs: dict[str, list[float]] = defaultdict(list)
    phase_counts: Counter[str] = Counter()
    action_counts: list[int] = []
    total_rows = 0
    privileged_rows = 0
    provenance_errors = 0
    for index, seed in enumerate(full_game_seeds):
        if index % 32 == 0:
            print(f"048 full-game trajectory {index + 1}/{len(full_game_seeds)}", flush=True)
        traverser = seed % 2
        trajectory, rows = sampler.sample(seed, traverser)
        trajectory_results.append(trajectory)
        total_rows += len(rows)
        for row in rows:
            primary_reservoir.add(row)
            repeated_reservoir.add(row)
            log_weight = row.log_importance_weight
            all_log_weights.append(log_weight)
            (first_half_logs if index < 128 else second_half_logs).append(log_weight)
            phase = _phase(row.turn)
            phase_counts[phase] += 1
            phase_logs[phase].append(log_weight)
            action_counts.append(len(row.actions.action_type_ids))
            privileged_rows += any(value >= 13 for value in row.state.entity_type_ids)
            provenance_errors += row.provenance != "outcome_sampling_048"

    print("048 deterministic repeat of first 16 full-game trajectories", flush=True)
    repeated_first = [sampler.sample(seed, seed % 2)[0] for seed in full_game_seeds[:16]]
    trajectory_repeat_identical = trajectory_results[:16] == repeated_first

    reservoir_payload = primary_reservoir.payload()
    repeated_reservoir_payload = repeated_reservoir.payload()
    reservoir_bytes = json.dumps(
        reservoir_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    reservoir_roundtrip = json.loads(reservoir_bytes.decode("utf-8")) == reservoir_payload
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "advantage_reservoir.json").write_bytes(reservoir_bytes)

    termination_counts = Counter(row["termination_reason"] for row in trajectory_results)
    natural_count = sum(row["natural_terminal"] for row in trajectory_results)
    natural_fraction = natural_count / max(len(trajectory_results), 1)
    overall_ess_ratio = _ess_ratio(all_log_weights)
    phase_ess_ratios = {phase: _ess_ratio(logs) for phase, logs in sorted(phase_logs.items())}
    max_log_weight = max(all_log_weights, default=math.inf)
    p99_log_weight = _percentile(all_log_weights, 0.99)
    split_median_delta = abs(median(first_half_logs) - median(second_half_logs))
    split_p95_delta = abs(
        _percentile(first_half_logs, 0.95) - _percentile(second_half_logs, 0.95)
    )
    privacy = run_privacy_and_invariance_audits()
    checkpoint_hash = _sha256_file(checkpoint)

    frozen_count_check = (
        len(full_game_seeds) == 256
        and full_game_seeds == FULL_GAME_SEEDS
        and uniform_samples == 200_000
        and adaptive_iterations == 100_000
    ) if enforce_frozen_counts else True
    trajectory_integrity = (
        len(trajectory_results) == len(full_game_seeds)
        and Counter(row["traverser"] for row in trajectory_results) == {0: 128, 1: 128}
        and all(row["illegal_actions"] == 0 for row in trajectory_results)
        and all(row["nonfinite_target_rows"] == 0 for row in trajectory_results)
        and all(row["turns"] <= 200 for row in trajectory_results)
        and total_rows == sum(row["candidate_advantage_rows"] for row in trajectory_results)
    )
    row_gate = (
        total_rows >= 2_048
        and all(phase_counts[phase] >= 128 for phase in ("early", "mid", "late"))
        and bool(action_counts)
        and min(action_counts) >= 1
        and max(action_counts) <= 600
    )
    variance_gate = (
        overall_ess_ratio >= 0.01
        and all(phase_ess_ratios.get(phase, 0.0) >= 0.01 for phase in ("early", "mid", "late"))
        and max_log_weight <= math.log(1e12)
        and p99_log_weight <= math.log(1e6)
        and split_median_delta <= 1.0
        and split_p95_delta <= 2.0
    )
    reproducibility_memory_gate = (
        trajectory_repeat_identical
        and reservoir_payload == repeated_reservoir_payload
        and reservoir_roundtrip
        and len(reservoir_bytes) <= 64 * 1024 * 1024
        and privileged_rows == 0
        and provenance_errors == 0
        and privacy["passed"]
    )
    gates = {
        "frozen_configuration": frozen_count_check,
        "exact_reference_hash": exact_hash == EXACT_HASH_044,
        "reduced_fixed_policy_estimator": uniform_gate,
        "reduced_adaptive_outcome_sampling": adaptive_gate,
        "checkpoint_and_privacy": checkpoint_hash == CHECKPOINT_SHA256 and privacy["passed"],
        "full_game_trajectory_integrity": trajectory_integrity,
        "full_game_natural_terminal_rate": natural_fraction >= 0.95,
        "full_game_row_count_phase_and_action_bounds": row_gate,
        "full_game_importance_variance_and_ess": variance_gate,
        "full_game_reproducibility_and_memory": reproducibility_memory_gate,
    }
    passed = all(gates.values())
    result = {
        "experiment": "048_outcome_sampling_cfr_feasibility",
        "protocol": "docs/experiments/outcome_sampling_cfr_feasibility_048.md",
        "configuration": {
            "uniform_samples_per_traverser": uniform_samples,
            "adaptive_iterations_per_seed": adaptive_iterations,
            "adaptive_seeds": list(ADAPTIVE_SEEDS),
            "full_game_seeds": [full_game_seeds[0], full_game_seeds[-1]],
            "full_game_trajectories": len(full_game_seeds),
            "exploration": 0.1,
            "target_temperature": 1.0,
            "repetition_limit": 4,
            "no_progress_limit": 60,
            "turn_limit": 200,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
        },
        "exact_reference_hash": exact_hash,
        "expected_exact_reference_hash": EXACT_HASH_044,
        "reduced_fixed_uniform": uniform_results,
        "reduced_adaptive": {
            "seeds": adaptive_results,
            "first_seed_repeat_identical": first_repeat_identical,
            "median_current_exploitability": median(adaptive_exploitabilities),
        },
        "full_game": {
            "trajectories": trajectory_results,
            "termination_counts": dict(termination_counts),
            "natural_terminal_fraction": natural_fraction,
            "total_advantage_rows": total_rows,
            "phase_counts": dict(phase_counts),
            "min_legal_actions": min(action_counts, default=0),
            "max_legal_actions": max(action_counts, default=0),
            "mean_legal_actions": sum(action_counts) / max(len(action_counts), 1),
            "importance": {
                "overall_ess_ratio": overall_ess_ratio,
                "phase_ess_ratios": phase_ess_ratios,
                "max_log_weight": max_log_weight,
                "p99_log_weight": p99_log_weight,
                "p99_absolute_weight": (
                    math.exp(p99_log_weight) if p99_log_weight <= 700.0 else None
                ),
                "split_half_median_log_delta": split_median_delta,
                "split_half_p95_log_delta": split_p95_delta,
            },
            "repeat_first_16_identical": trajectory_repeat_identical,
            "privacy": privacy,
            "privileged_rows": privileged_rows,
            "provenance_errors": provenance_errors,
            "reservoir": {
                "capacity": primary_reservoir.capacity,
                "seen": primary_reservoir.seen,
                "rows": len(primary_reservoir.rows),
                "repeat_identical": reservoir_payload == repeated_reservoir_payload,
                "roundtrip_lossless": reservoir_roundtrip,
                "serialized_bytes": len(reservoir_bytes),
                "sha256": hashlib.sha256(reservoir_bytes).hexdigest(),
            },
        },
        "gates": gates,
        "passed": passed,
        "neural_fit_authorized": passed,
        "gpu_rental_authorized": passed,
        "decision": (
            "authorize_separately_preregistered_neural_outcome_sampling_fit"
            if passed
            else "do_not_train_or_rent_gpu"
        ),
    }
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"048 result: {'PASS' if passed else 'FAIL'} -> {output_path}", flush=True)
    return result


def _uniform_estimator_audit(traverser: int, samples: int, seed: int) -> dict[str, Any]:
    game = ReducedSplendorCFRGame()
    exact = exact_fixed_policy_regret_increments(game, traverser)
    lookup = ExactCFRSolver(game)
    rng = random.Random((seed * 10) + traverser)
    sums: dict[Any, list[float]] = {}
    observations: Counter[Any] = Counter()
    illegal_actions = 0
    for _ in range(samples):
        updates, _utility, _logs = sample_reduced_outcome_updates(
            game,
            traverser,
            rng,
            lookup._node,
            lambda _state, node: tuple([1.0 / len(node.actions)] * len(node.actions)),
            exploration=0.1,
        )
        for update in updates:
            values = sums.setdefault(update.information_key, [0.0] * len(update.actions))
            for index, value in enumerate(update.regret_estimate):
                values[index] += value
            observations[update.information_key] += 1
    eligible = {
        key: values for key, values in exact.items() if max(abs(value) for value in values) >= 1e-4
    }
    errors: list[float] = []
    error_by_key: dict[Any, float] = {}
    for key, exact_values in eligible.items():
        estimated = [value / samples for value in sums.get(key, [0.0] * len(exact_values))]
        components = [
            abs(left - right) for left, right in zip(estimated, exact_values, strict=True)
        ]
        errors.extend(components)
        error_by_key[key] = max(components)
    minimum_history = min(len(key[-1]) for key in eligible)
    root_band_error = max(
        error for key, error in error_by_key.items() if len(key[-1]) == minimum_history
    )
    return {
        "traverser": traverser,
        "samples": samples,
        "exact_information_sets": len(exact),
        "eligible_information_sets": len(eligible),
        "sampled_eligible_information_sets": sum(observations[key] > 0 for key in eligible),
        "eligible_information_set_coverage": sum(observations[key] > 0 for key in eligible)
        / len(eligible),
        "mean_absolute_component_error": sum(errors) / len(errors),
        "max_absolute_component_error": max(errors),
        "root_band_history_length": minimum_history,
        "root_band_max_component_error": root_band_error,
        "min_observations_over_eligible": min(observations[key] for key in eligible),
        "illegal_actions": illegal_actions + lookup.illegal_actions,
        "unstable_action_sets": lookup.unstable_action_sets,
        "finite": all(math.isfinite(value) for value in errors),
    }


def _compact_exploitability(analysis: dict[str, Any]) -> dict[str, Any]:
    integrity = True
    for response in analysis["best_responses"].values():
        integrity = integrity and (
            response["selected_information_sets"] == response["expected_information_sets"]
            and response["missing_selections"] == 0
            and response["unstable_action_sets"] == 0
        )
    return {key: value for key, value in analysis.items() if key != "best_responses"} | {
        "best_response_integrity": integrity
    }


def _ess_ratio(log_weights: list[float]) -> float:
    if not log_weights:
        return 0.0
    maximum = max(log_weights)
    scaled = [math.exp(value - maximum) for value in log_weights]
    return (sum(scaled) ** 2) / (len(scaled) * sum(value * value for value in scaled))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = max(0, min(math.ceil(quantile * len(ordered)) - 1, len(ordered) - 1))
    return ordered[index]


def _phase(turn: int) -> str:
    return "early" if turn <= 19 else "mid" if turn <= 49 else "late"


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/outcome_sampling_cfr_feasibility_048"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
