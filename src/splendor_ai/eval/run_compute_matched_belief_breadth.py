"""Run experiment 059: compute-matched belief-breadth audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace
from typing import Any

import torch

from splendor_ai.cfr import ResolvedLeaf, deterministic_belief_range, resolve_two_ply
from splendor_ai.encoding import (
    canonical_public_belief_key,
    canonical_successor_action_key,
)
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    ResolverIneligible,
    _distinct_canonical_scores,
)
from splendor_ai.eval.run_resolver_budget_convergence_audit import (
    _all_numeric_finite,
    _atomic_write,
    _zero_counters,
)
from splendor_ai.eval.run_resolver_target_repeatability_audit import (
    _cluster_bootstrap,
    _comparison_row,
    _comparison_summary,
    _quantile,
    validate_selected_root,
)
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


RESULT_058_SHA256 = "9369c7a6ace8cacd0907fa4dd6d2dd72a48495aae3b09725cd5664d0ffa74a64"
SELECTION_055_SHA256 = "4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a"
VALIDATION_052_SHA256 = "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
CONFIGURATIONS: dict[str, tuple[int, int]] = {
    "3x4": (3, 4),
    "6x2": (6, 2),
    "12x1": (12, 1),
}
SIDES = ("left", "right")
_WORKER_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class BeliefBreadthConfig:
    output_dir: Path = Path("outputs/compute_matched_belief_breadth_059")
    result_058: Path = Path(
        "outputs/early_resolver_variance_decomposition_058/result.json"
    )
    selection_055: Path = Path(
        "outputs/resolver_target_repeatability_audit_055/selection.json"
    )
    validation_replay: Path = Path(
        "outputs/resolver_selfplay_distillation_052/validation_targets.jsonl"
    )
    parent_checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    root_actions: int = 4
    opponent_actions: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_base: int = 590_000_000
    continuation_seed_base: int = 59_000_000_000_000_000
    workers: int = 10
    bootstrap_samples: int = 20_000
    equivalence_regret: float = 0.025
    loose_regret: float = 0.05
    minimum_completion_fraction: float = 0.90
    minimum_early_exact: float = 0.55
    minimum_early_equivalence: float = 0.60
    minimum_early_spearman: float = 0.50
    maximum_early_mean_regret: float = 0.035
    maximum_early_p90_regret: float = 0.075
    minimum_early_exact_gain: float = 0.15
    minimum_early_equivalence_gain: float = 0.10
    minimum_gain_ci_lower: float = 0.0
    maximum_early_regret_ratio: float = 0.70
    maximum_regret_change_ci_upper: float = 0.0
    maximum_guardrail_agreement_deficit: float = 0.10
    maximum_guardrail_regret_excess: float = 0.01
    maximum_loop_adjudicated_fraction: float = 0.001
    device: str = "cpu"


def run_experiment(
    config: BeliefBreadthConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BeliefBreadthConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 059 result does not match frozen identity.")
        return result

    selected = select_breadth_states(cfg)
    _atomic_write(
        cfg.output_dir / "selection.json",
        {
            "selected_count": len(selected),
            "source_games": len({row["game_seed"] for row in selected}),
            "cell_counts": _selection_counts(selected),
            "selected": selected,
        },
    )
    started = time.perf_counter()
    rows = _evaluate_parallel(cfg, selected)
    evaluation_wall_seconds = time.perf_counter() - started
    scorer = _LegalActionScorer(
        cfg.parent_checkpoint,
        cfg.device,
        "legal_action_attention_scorer",
    )
    repeats = {
        name: _evaluate_estimate(selected[0], cfg, name, "left", scorer)
        for name in CONFIGURATIONS
    }
    repeat_identical = {
        name: (
            _deterministic_projection(rows[0]["estimates"][f"{name}_left"])
            == _deterministic_projection(repeats[name])
        )
        for name in CONFIGURATIONS
    }
    summary = summarize_breadth(
        rows,
        cfg,
        evaluation_wall_seconds=evaluation_wall_seconds,
    )
    integrity = _integrity(
        cfg,
        selected,
        rows,
        summary,
        repeat_identical,
    )
    integrity_passed = all(integrity.values())
    candidate_gates = {
        name: _candidate_gates(cfg, summary, name)
        for name in ("12x1", "6x2")
    }
    if not integrity_passed:
        decision = "belief_breadth_audit_invalid_no_design_decision"
    elif all(candidate_gates["12x1"].values()):
        decision = (
            "twelve_by_one_belief_breadth_stable_"
            "authorize_separate_resolver_pilot"
        )
    elif all(candidate_gates["6x2"].values()):
        decision = (
            "six_by_two_belief_breadth_stable_"
            "authorize_separate_resolver_pilot"
        )
    else:
        decision = (
            "belief_breadth_reallocation_insufficient_"
            "move_to_public_belief_visit_search"
        )
    payload = {
        "experiment": "059_compute_matched_belief_breadth",
        "protocol": "docs/experiments/compute_matched_belief_breadth_059.md",
        "identity": _identity(cfg),
        "selection": {
            "selected_count": len(selected),
            "unique_states": len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            ),
            "source_games": len({row["game_seed"] for row in selected}),
            "cell_counts": _selection_counts(selected),
            "outcome_blind": True,
        },
        "summary": summary,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "candidate_gates": candidate_gates,
        "first_state_left_repeat_identical": repeat_identical,
        "rows": rows,
        "decision": decision,
        "resolver_pilot_authorized": (
            integrity_passed and "authorize_separate_resolver_pilot" in decision
        ),
        "collection_authorized": False,
        "training_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(result_path, payload)
    print(f"059 decision={decision} integrity={integrity_passed}", flush=True)
    return payload


def select_breadth_states(cfg: BeliefBreadthConfig) -> list[dict[str, Any]]:
    selection = json.loads(cfg.selection_055.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    selected = []
    for row in selection["selected"]:
        cell = (
            f"{row['phase']}|seat{row['actor_seat']}|{row['gap_stratum']}"
        )
        limit = 4 if str(row["phase"]) == "early" else 2
        if counts[cell] >= limit:
            continue
        selected.append(row)
        counts[cell] += 1
    if len(selected) != 48 or len(counts) != 18:
        raise ValueError("059 did not find the frozen 48-state balance.")
    expected = {
        4 if cell.startswith("early|") else 2 for cell in counts
    }
    if expected != {2, 4} or any(
        count != (4 if cell.startswith("early|") else 2)
        for cell, count in counts.items()
    ):
        raise ValueError("059 phase/cell counts changed.")
    if len({(row["game_seed"], row["turn_index"]) for row in selected}) != 48:
        raise ValueError("059 selection contains duplicate states.")
    if len({row["game_seed"] for row in selected}) != 17:
        raise ValueError("059 selection must span exactly 17 games.")
    return selected


def _selection_counts(selected: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{row['phase']}|seat{row['actor_seat']}|{row['gap_stratum']}"
                for row in selected
            ).items()
        )
    )


def _evaluate_parallel(
    cfg: BeliefBreadthConfig,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    identity = _identity(cfg)
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 059 progress does not match identity.")
            completed[int(payload["sample_index"])] = payload["result"]
    missing = [
        sample
        for sample in selected
        if int(sample["sample_index"]) not in completed
    ]
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(cfg.workers, len(missing)),
            initializer=_initialize_worker,
            initargs=(str(cfg.parent_checkpoint), cfg.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_state_worker, sample, cfg): int(
                    sample["sample_index"]
                )
                for sample in missing
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    sample_index = futures[future]
                    result = future.result()
                    completed[sample_index] = result
                    handle.write(
                        json.dumps(
                            {
                                "identity": identity,
                                "sample_index": sample_index,
                                "result": result,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                    print(
                        f"[resolver059] states={len(completed)}/{len(selected)}",
                        flush=True,
                    )
    return [completed[int(sample["sample_index"])] for sample in selected]


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(
        Path(checkpoint), device, "legal_action_attention_scorer"
    )


def _evaluate_state_worker(
    sample: dict[str, Any],
    cfg: BeliefBreadthConfig,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("059 worker scorer was not initialized.")
    return _evaluate_state(sample, cfg, _WORKER_SCORER)


def _evaluate_state(
    sample: dict[str, Any],
    cfg: BeliefBreadthConfig,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    estimates = {}
    for name in CONFIGURATIONS:
        for side in SIDES:
            estimates[f"{name}_{side}"] = _evaluate_estimate(
                sample, cfg, name, side, scorer
            )
    return {
        "sample_index": int(sample["sample_index"]),
        "game_seed": int(sample["game_seed"]),
        "turn_index": int(sample["turn_index"]),
        "actor_seat": int(sample["actor_seat"]),
        "phase": str(sample["phase"]),
        "gap_stratum": str(sample["gap_stratum"]),
        "selection_cell": str(sample["selection_cell"]),
        "estimates": estimates,
    }


def _evaluate_estimate(
    sample: dict[str, Any],
    cfg: BeliefBreadthConfig,
    configuration: str,
    side: str,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    determinizations, continuations_per_leaf = CONFIGURATIONS[configuration]
    prepared = validate_selected_root(sample, scorer)
    state = prepared["state"]
    root_keys = tuple(
        canonical_successor_action_key(state, row.action)
        for row in prepared["ranked"][: cfg.root_actions]
    )
    counters: Counter[str] = Counter()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        branches = _prepare_branches(
            sample,
            cfg,
            state,
            root_keys,
            side,
            determinizations,
            scorer,
        )
    except ResolverIneligible as exc:
        return {
            "completed": False,
            "fallback_reason": exc.reason,
            "root_keys": prepared["root_keys"],
            "action_values": None,
            "best_index": None,
            "counters": _zero_counters(),
            "wall_seconds": time.perf_counter() - wall_started,
            "process_cpu_seconds": time.process_time() - cpu_started,
        }
    env = SplendorEnv()
    leaves: list[ResolvedLeaf] = []
    for branch in branches:
        after_root = branch["after_root"]
        for response in branch["responses"]:
            response_key = canonical_successor_action_key(
                after_root, response.action
            )
            values = []
            for continuation in range(continuations_per_leaf):
                seed = breadth_continuation_seed(
                    cfg,
                    int(sample["sample_index"]),
                    side,
                    int(branch["determinization"]),
                    continuations_per_leaf,
                    continuation,
                )
                outcome = rollout_score_margin_loop_safe(
                    env,
                    scorer,
                    after_root,
                    response.action,
                    int(sample["actor_seat"]),
                    seed,
                    cfg.continuation_epsilon,
                    cfg.continuation_temperature,
                    cfg.continuation_top_k,
                    cfg.continuation_max_turns,
                    cfg.score_margin_scale,
                    cfg.max_loop_fallback_triggers,
                )
                if not math.isfinite(outcome.value):
                    raise RuntimeError("059 produced a nonfinite leaf value.")
                values.append(float(outcome.value))
                counters["continuations"] += 1
                counters["resolved_continuations"] += int(outcome.resolved)
                counters["hard_caps"] += int(outcome.hard_cap_reached)
                counters["loop_adjudications"] += int(
                    outcome.loop_adjudicated
                )
                counters["rollout_fallback_triggers"] += int(
                    outcome.fallback_triggers
                )
            mean = statistics.fmean(values)
            leaves.append(
                ResolvedLeaf(
                    determinization=int(branch["determinization"]),
                    root_action=branch["root_key"],
                    opponent_information_set=branch["information_set"],
                    opponent_action=response_key,
                    discovery_payoff=mean,
                    evaluation_payoff=mean,
                )
            )
    resolution = resolve_two_ply(
        leaves,
        root_keys,
        determinizations,
        payoff="discovery_payoff",
    )
    root_key_payload = [repr(key) for key in resolution.root_actions]
    if root_key_payload != prepared["root_keys"]:
        raise RuntimeError("059 root keys changed after resolution.")
    values = [float(value) for value in resolution.action_values]
    return {
        "completed": True,
        "fallback_reason": None,
        "root_keys": root_key_payload,
        "action_values": values,
        "best_index": int(resolution.best_index),
        "counters": _counter_payload(counters),
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": time.process_time() - cpu_started,
    }


def _prepare_branches(
    sample: dict[str, Any],
    cfg: BeliefBreadthConfig,
    state: Any,
    root_keys: tuple[tuple[object, ...], ...],
    side: str,
    determinization_count: int,
    scorer: _LegalActionScorer,
) -> list[dict[str, Any]]:
    determinizations = deterministic_belief_range(
        state,
        state.current_player,
        root_index=breadth_belief_root_index(
            cfg, int(sample["sample_index"]), side
        ),
        count=determinization_count,
    )
    env = SplendorEnv()
    branches = []
    action_sets: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
    for det_index, determinized in enumerate(determinizations):
        root_map = actions_by_canonical_successor_key(
            determinized, env.legal_actions(determinized)
        )
        if any(key not in root_map for key in root_keys):
            raise ResolverIneligible("root_mapping_changed")
        for root_key in root_keys:
            after_root = env.step(determinized, root_map[root_key])
            if after_root.terminal:
                raise ResolverIneligible("root_action_terminal")
            ranked = _distinct_canonical_scores(
                after_root,
                scorer.score_actions(
                    after_root, env.legal_actions(after_root)
                ),
            )
            if len(ranked) < cfg.opponent_actions:
                raise ResolverIneligible("fewer_than_four_opponent_actions")
            responses = ranked[: cfg.opponent_actions]
            response_keys = tuple(
                canonical_successor_action_key(after_root, row.action)
                for row in responses
            )
            information_set = (
                root_key,
                canonical_public_belief_key(
                    after_root, after_root.current_player
                ),
            )
            previous = action_sets.setdefault(information_set, response_keys)
            if previous != response_keys:
                raise ResolverIneligible("opponent_action_set_unstable")
            branches.append(
                {
                    "determinization": det_index,
                    "root_key": root_key,
                    "after_root": after_root,
                    "information_set": information_set,
                    "responses": responses,
                }
            )
    return branches


def breadth_belief_root_index(
    cfg: BeliefBreadthConfig,
    sample_index: int,
    side: str,
) -> int:
    side_index = SIDES.index(side)
    return cfg.belief_root_base + sample_index * 10 + side_index


def breadth_continuation_seed(
    cfg: BeliefBreadthConfig,
    sample_index: int,
    side: str,
    determinization: int,
    continuations_per_leaf: int,
    continuation: int,
) -> int:
    side_index = SIDES.index(side)
    slot = determinization * continuations_per_leaf + continuation
    return (
        cfg.continuation_seed_base
        + sample_index * 1_000_000_000
        + side_index * 100_000_000
        + slot
    )


def summarize_breadth(
    rows: list[dict[str, Any]],
    cfg: BeliefBreadthConfig,
    *,
    evaluation_wall_seconds: float,
) -> dict[str, Any]:
    configurations = {}
    raw_comparisons: dict[str, list[dict[str, Any]]] = {}
    for offset, name in enumerate(CONFIGURATIONS):
        comparisons = []
        for row in rows:
            left = row["estimates"][f"{name}_left"]
            right = row["estimates"][f"{name}_right"]
            if left["completed"] and right["completed"]:
                comparisons.append(_comparison_row(row, left, right))
        raw_comparisons[name] = comparisons
        helper = _metric_helper_config(cfg, 590_060 + offset)
        metrics = _comparison_summary(comparisons, helper)
        metrics["bootstrap"] = _cluster_bootstrap(comparisons, helper)
        metrics["structural"] = _structural_summary(rows, name)
        metrics["performance"] = _configuration_performance(rows, name)
        configurations[name] = metrics
    paired = {
        candidate: _paired_candidate_summary(
            rows,
            raw_comparisons["3x4"],
            raw_comparisons[candidate],
            cfg,
            seed=590_160 + offset,
        )
        for offset, candidate in enumerate(("6x2", "12x1"))
    }
    attempts = [
        estimate
        for row in rows
        for estimate in row["estimates"].values()
    ]
    performance = {
        "evaluation_wall_seconds_this_run": evaluation_wall_seconds,
        "aggregate_estimate_wall_seconds": sum(
            float(row["wall_seconds"]) for row in attempts
        ),
        "aggregate_process_cpu_seconds": sum(
            float(row["process_cpu_seconds"]) for row in attempts
        ),
        "completed_estimates": sum(row["completed"] for row in attempts),
        "attempts": len(attempts),
    }
    return {
        "configurations": configurations,
        "paired_vs_3x4": paired,
        "performance": performance,
    }


def _structural_summary(
    rows: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    phase = {}
    for phase_name in ("early", "mid", "late"):
        phase_rows = [row for row in rows if row["phase"] == phase_name]
        both = sum(
            row["estimates"][f"{name}_left"]["completed"]
            and row["estimates"][f"{name}_right"]["completed"]
            for row in phase_rows
        )
        phase[phase_name] = {
            "states": len(phase_rows),
            "both_completed": both,
            "both_completed_fraction": both / len(phase_rows),
        }
    attempts = [
        row["estimates"][f"{name}_{side}"]
        for row in rows
        for side in SIDES
    ]
    both = sum(
        row["estimates"][f"{name}_left"]["completed"]
        and row["estimates"][f"{name}_right"]["completed"]
        for row in rows
    )
    return {
        "states": len(rows),
        "attempts": len(attempts),
        "completed_estimates": sum(row["completed"] for row in attempts),
        "both_completed": both,
        "both_completed_fraction": both / len(rows),
        "fallback_reasons": dict(
            sorted(
                Counter(
                    row["fallback_reason"]
                    for row in attempts
                    if not row["completed"]
                ).items()
            )
        ),
        "phase": phase,
    }


def _configuration_performance(
    rows: list[dict[str, Any]],
    name: str,
) -> dict[str, float]:
    attempts = [
        row["estimates"][f"{name}_{side}"]
        for row in rows
        for side in SIDES
    ]
    return {
        "aggregate_wall_seconds": sum(
            float(row["wall_seconds"]) for row in attempts
        ),
        "aggregate_process_cpu_seconds": sum(
            float(row["process_cpu_seconds"]) for row in attempts
        ),
        "mean_wall_seconds": statistics.fmean(
            float(row["wall_seconds"]) for row in attempts
        ),
        "mean_process_cpu_seconds": statistics.fmean(
            float(row["process_cpu_seconds"]) for row in attempts
        ),
    }


def _paired_candidate_summary(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    cfg: BeliefBreadthConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    baseline = {
        (int(row["game_seed"]), int(row["turn_index"])): row
        for row in baseline_rows
    }
    candidate = {
        (int(row["game_seed"]), int(row["turn_index"])): row
        for row in candidate_rows
    }
    selected = []
    metadata = {
        (int(row["game_seed"]), int(row["turn_index"])): row for row in rows
    }
    for key in sorted(set(baseline) & set(candidate)):
        base = baseline[key]
        cand = candidate[key]
        meta = metadata[key]
        selected.append(
            {
                "game_seed": key[0],
                "turn_index": key[1],
                "phase": str(meta["phase"]),
                "baseline_exact": float(base["exact_best_agreement"]),
                "candidate_exact": float(cand["exact_best_agreement"]),
                "baseline_equivalence": float(base["value_equivalent"]),
                "candidate_equivalence": float(cand["value_equivalent"]),
                "baseline_regret": float(base["symmetric_regret"]),
                "candidate_regret": float(cand["symmetric_regret"]),
            }
        )
    by_phase = {
        phase: _paired_phase_summary(
            [row for row in selected if row["phase"] == phase]
        )
        for phase in ("early", "mid", "late")
    }
    early_rows = [row for row in selected if row["phase"] == "early"]
    return {
        "matched_states": len(selected),
        "by_phase": by_phase,
        "early_bootstrap": _paired_bootstrap(
            early_rows, cfg.bootstrap_samples, seed
        ),
        "rows": selected,
    }


def _paired_phase_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "states": 0,
            "baseline_exact": 0.0,
            "candidate_exact": 0.0,
            "exact_gain": 0.0,
            "baseline_equivalence": 0.0,
            "candidate_equivalence": 0.0,
            "equivalence_gain": 0.0,
            "baseline_mean_regret": math.inf,
            "candidate_mean_regret": math.inf,
            "regret_change": math.inf,
            "regret_ratio": math.inf,
        }
    baseline_regret = statistics.fmean(row["baseline_regret"] for row in rows)
    candidate_regret = statistics.fmean(row["candidate_regret"] for row in rows)
    baseline_exact = statistics.fmean(row["baseline_exact"] for row in rows)
    candidate_exact = statistics.fmean(row["candidate_exact"] for row in rows)
    baseline_equivalence = statistics.fmean(
        row["baseline_equivalence"] for row in rows
    )
    candidate_equivalence = statistics.fmean(
        row["candidate_equivalence"] for row in rows
    )
    return {
        "states": len(rows),
        "baseline_exact": baseline_exact,
        "candidate_exact": candidate_exact,
        "exact_gain": candidate_exact - baseline_exact,
        "baseline_equivalence": baseline_equivalence,
        "candidate_equivalence": candidate_equivalence,
        "equivalence_gain": candidate_equivalence - baseline_equivalence,
        "baseline_mean_regret": baseline_regret,
        "candidate_mean_regret": candidate_regret,
        "regret_change": candidate_regret - baseline_regret,
        "regret_ratio": (
            candidate_regret / baseline_regret
            if baseline_regret > 0.0
            else math.inf
        ),
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[int(row["game_seed"])].append(row)
    cluster_rows = list(clusters.values())
    rng = random.Random(seed)
    draws = {
        "exact_gain": [],
        "equivalence_gain": [],
        "regret_change": [],
    }
    for _ in range(samples):
        sampled = [
            row
            for _cluster in cluster_rows
            for row in cluster_rows[rng.randrange(len(cluster_rows))]
        ]
        phase = _paired_phase_summary(sampled)
        for metric in draws:
            draws[metric].append(float(phase[metric]))
    return {
        metric: {
            "lower": _quantile(sorted(values), 0.025),
            "upper": _quantile(sorted(values), 0.975),
        }
        for metric, values in draws.items()
    }


def _candidate_gates(
    cfg: BeliefBreadthConfig,
    summary: dict[str, Any],
    candidate: str,
) -> dict[str, bool]:
    metrics = summary["configurations"][candidate]
    structural = metrics["structural"]
    early = metrics["breakdowns"]["phase"]["early"]
    paired = summary["paired_vs_3x4"][candidate]
    early_paired = paired["by_phase"]["early"]
    bootstrap = paired["early_bootstrap"]
    guardrails = paired["by_phase"]
    return {
        "overall_completion": (
            structural["both_completed_fraction"]
            >= cfg.minimum_completion_fraction
        ),
        "phase_completion": all(
            row["both_completed_fraction"] >= cfg.minimum_completion_fraction
            for row in structural["phase"].values()
        ),
        "early_exact": (
            early["exact_best_agreement"] >= cfg.minimum_early_exact
        ),
        "early_equivalence": (
            early["value_equivalent_agreement"]
            >= cfg.minimum_early_equivalence
        ),
        "early_spearman": (
            early["mean_spearman"] >= cfg.minimum_early_spearman
        ),
        "early_mean_regret": (
            early["mean_symmetric_cross_regret"]
            <= cfg.maximum_early_mean_regret
        ),
        "early_p90_regret": (
            _phase_p90(summary, candidate, "early")
            <= cfg.maximum_early_p90_regret
        ),
        "early_exact_gain": (
            early_paired["exact_gain"] >= cfg.minimum_early_exact_gain
        ),
        "early_exact_gain_ci": (
            bootstrap["exact_gain"]["lower"] >= cfg.minimum_gain_ci_lower
        ),
        "early_equivalence_gain": (
            early_paired["equivalence_gain"]
            >= cfg.minimum_early_equivalence_gain
        ),
        "early_equivalence_gain_ci": (
            bootstrap["equivalence_gain"]["lower"]
            >= cfg.minimum_gain_ci_lower
        ),
        "early_regret_ratio": (
            early_paired["regret_ratio"]
            <= cfg.maximum_early_regret_ratio
        ),
        "early_regret_change_ci": (
            bootstrap["regret_change"]["upper"]
            <= cfg.maximum_regret_change_ci_upper
        ),
        "mid_late_exact_guardrail": all(
            guardrails[phase]["candidate_exact"]
            >= guardrails[phase]["baseline_exact"]
            - cfg.maximum_guardrail_agreement_deficit
            for phase in ("mid", "late")
        ),
        "mid_late_equivalence_guardrail": all(
            guardrails[phase]["candidate_equivalence"]
            >= guardrails[phase]["baseline_equivalence"]
            - cfg.maximum_guardrail_agreement_deficit
            for phase in ("mid", "late")
        ),
        "mid_late_regret_guardrail": all(
            guardrails[phase]["candidate_mean_regret"]
            <= guardrails[phase]["baseline_mean_regret"]
            + cfg.maximum_guardrail_regret_excess
            for phase in ("mid", "late")
        ),
    }


def _phase_p90(
    summary: dict[str, Any],
    candidate: str,
    phase: str,
) -> float:
    rows = [
        row
        for row in summary["paired_vs_3x4"][candidate]["rows"]
        if row["phase"] == phase
    ]
    return (
        _quantile(sorted(row["candidate_regret"] for row in rows), 0.90)
        if rows
        else math.inf
    )


def _integrity(
    cfg: BeliefBreadthConfig,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    repeat_identical: dict[str, bool],
) -> dict[str, bool]:
    attempts = [
        estimate
        for row in rows
        for estimate in row["estimates"].values()
    ]
    completed = [row for row in attempts if row["completed"]]
    incomplete = [row for row in attempts if not row["completed"]]
    continuations = sum(
        int(row["counters"]["continuations"]) for row in attempts
    )
    resolved = sum(
        int(row["counters"]["resolved_continuations"]) for row in attempts
    )
    hard_caps = sum(int(row["counters"]["hard_caps"]) for row in attempts)
    adjudications = sum(
        int(row["counters"]["loop_adjudications"]) for row in attempts
    )
    counts = _selection_counts(selected)
    expected_counts = {
        cell: 4 if cell.startswith("early|") else 2 for cell in counts
    }
    namespace_contract = all(
        _namespace_contract_for_state(cfg, int(row["sample_index"]))
        for row in selected
    )
    finite = all(
        (
            not estimate["completed"]
            or all(
                math.isfinite(float(value))
                for value in estimate["action_values"]
            )
        )
        for estimate in attempts
    ) and _all_numeric_finite(summary)
    return {
        "frozen_prerequisites_and_configuration": True,
        "exact_48_unique_outcome_blind_states": (
            len(selected) == 48
            and len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            )
            == 48
        ),
        "exact_phase_cell_balance": counts == expected_counts,
        "exact_17_source_games": (
            len({row["game_seed"] for row in selected}) == 17
        ),
        "outcome_blind_selection": True,
        "root_and_legal_alignment": True,
        "exact_288_estimate_attempts": len(attempts) == 288,
        "exact_completed_continuation_accounting": (
            all(
                int(row["counters"]["continuations"]) == 192
                for row in completed
            )
            and continuations == 192 * len(completed)
            and continuations <= 55_296
        ),
        "ineligible_before_continuation": all(
            int(row["counters"]["continuations"]) == 0 for row in incomplete
        ),
        "all_attempted_continuations_resolved": resolved == continuations,
        "zero_hard_caps": hard_caps == 0,
        "bounded_loop_adjudication": (
            adjudications / continuations if continuations else math.inf
        )
        <= cfg.maximum_loop_adjudicated_fraction,
        "opponent_action_sets_stable_when_completed": True,
        "nested_belief_roots_and_paired_slots": namespace_contract,
        "finite_values_and_metrics": finite,
        "first_state_left_repeats_identical": all(
            repeat_identical.values()
        ),
    }


def _metric_helper_config(
    cfg: BeliefBreadthConfig,
    seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        equivalence_regret=cfg.equivalence_regret,
        loose_regret=cfg.loose_regret,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=seed,
    )


def _namespace_contract_for_state(
    cfg: BeliefBreadthConfig,
    sample_index: int,
) -> bool:
    seed_sets = {}
    for side in SIDES:
        seed_sets[side] = {}
        for name, (determinizations, continuations) in CONFIGURATIONS.items():
            seed_sets[side][name] = {
                breadth_continuation_seed(
                    cfg,
                    sample_index,
                    side,
                    det,
                    continuations,
                    continuation,
                )
                for det in range(determinizations)
                for continuation in range(continuations)
            }
    left_sets = list(seed_sets["left"].values())
    right_sets = list(seed_sets["right"].values())
    return (
        len(
            {
                breadth_belief_root_index(cfg, sample_index, side)
                for side in SIDES
            }
        )
        == 2
        and all(values == left_sets[0] for values in left_sets)
        and all(values == right_sets[0] for values in right_sets)
        and len(left_sets[0]) == 12
        and len(right_sets[0]) == 12
        and left_sets[0].isdisjoint(right_sets[0])
    )


def _counter_payload(counters: Counter[str]) -> dict[str, int]:
    return {
        key: int(counters[key])
        for key in (
            "continuations",
            "resolved_continuations",
            "hard_caps",
            "loop_adjudications",
            "rollout_fallback_triggers",
        )
    }


def _deterministic_projection(estimate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in estimate.items()
        if key not in {"wall_seconds", "process_cpu_seconds"}
    }


def _validate_config(cfg: BeliefBreadthConfig) -> None:
    frozen = {
        cfg.result_058: RESULT_058_SHA256,
        cfg.selection_055: SELECTION_055_SHA256,
        cfg.validation_replay: VALIDATION_052_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 059 prerequisite mismatch: {path}")
    result_058 = json.loads(cfg.result_058.read_text(encoding="utf-8"))
    if (
        not result_058.get("integrity_passed")
        or result_058.get("decision")
        != "belief_sampling_primary_design_broader_public_belief_search"
    ):
        raise ValueError("059 requires the frozen integrity-valid 058 decision.")
    if cfg != BeliefBreadthConfig():
        raise ValueError("059 frozen configuration changed.")
    if any(dets * 4 * 4 * continuations != 192 for dets, continuations in CONFIGURATIONS.values()):
        raise ValueError("059 candidate compute is not exactly matched.")


def _identity(cfg: BeliefBreadthConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "sha256": {
            "result_058": _sha256_file(cfg.result_058),
            "selection_055": _sha256_file(cfg.selection_055),
            "validation_052": _sha256_file(cfg.validation_replay),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
        },
        "candidate_configurations": {
            name: {
                "determinizations": values[0],
                "continuations_per_leaf": values[1],
                "continuations_per_estimate": values[0] * 4 * 4 * values[1],
            }
            for name, values in CONFIGURATIONS.items()
        },
        "paired_namespace_contract": {
            "belief_root_excludes_configuration": True,
            "continuation_seed_excludes_configuration_root_response": True,
            "left_right_disjoint": True,
        },
    }


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
