"""Run experiment 056: resolver search-budget convergence audit."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import statistics
import string
import time
from types import SimpleNamespace
from typing import Any

import torch

from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
    ResolverIneligible,
)
from splendor_ai.eval.run_resolver_target_repeatability_audit import (
    _cluster_bootstrap,
    _comparison_row,
    _comparison_summary,
    _deterministic_projection,
    validate_selected_root,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


RESULT_055_SHA256 = "4aa3b54e77a4856c2c5cd1bb18b1a4088755bda7679bd52a7ce58627ba622714"
SELECTION_055_SHA256 = "4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a"
VALIDATION_052_SHA256 = "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
LABELS = tuple(string.ascii_uppercase[:16])
NEW_LABELS = LABELS[2:]
BUDGET_BLOCKS: dict[int, tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]] = {
    1: tuple(
        ((LABELS[index],), (LABELS[8 + index],))
        for index in range(8)
    ),
    2: tuple(
        (
            tuple(LABELS[index : index + 2]),
            tuple(LABELS[8 + index : 8 + index + 2]),
        )
        for index in range(0, 8, 2)
    ),
    4: tuple(
        (
            tuple(LABELS[index : index + 4]),
            tuple(LABELS[8 + index : 8 + index + 4]),
        )
        for index in range(0, 8, 4)
    ),
    8: ((tuple(LABELS[:8]), tuple(LABELS[8:])),),
}
_WORKER_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class ResolverBudgetConvergenceConfig:
    output_dir: Path = Path("outputs/resolver_budget_convergence_audit_056")
    result_055: Path = Path(
        "outputs/resolver_target_repeatability_audit_055/result.json"
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
    selected_per_cell: int = 2
    workers: int = 10
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 560_056
    equivalence_regret: float = 0.025
    loose_regret: float = 0.05
    minimum_8x_eligible_fraction: float = 0.80
    point_minimum_exact_agreement: float = 0.70
    point_minimum_exact_ci_lower: float = 0.60
    point_minimum_pearson: float = 0.85
    point_minimum_spearman: float = 0.65
    point_maximum_mean_regret: float = 0.025
    point_maximum_p90_regret: float = 0.075
    point_minimum_equivalence: float = 0.75
    point_minimum_equivalence_ci_lower: float = 0.65
    point_minimum_loose_regret_fraction: float = 0.80
    point_minimum_phase_exact: float = 0.55
    point_minimum_phase_equivalence: float = 0.60
    point_minimum_exact_gain: float = 0.15
    point_minimum_spearman_gain: float = 0.15
    point_maximum_regret_ratio: float = 0.60
    equivalence_minimum_rate: float = 0.75
    equivalence_minimum_ci_lower: float = 0.65
    equivalence_maximum_mean_regret: float = 0.035
    equivalence_maximum_p90_regret: float = 0.075
    equivalence_minimum_phase_seat: float = 0.60
    equivalence_minimum_gain: float = 0.15
    equivalence_maximum_regret_ratio: float = 0.70
    positive_minimum_exact_gain: float = 0.10
    positive_maximum_regret_ratio: float = 0.75
    maximum_loop_adjudicated_fraction: float = 0.001
    device: str = "cpu"


def run_experiment(
    config: ResolverBudgetConvergenceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResolverBudgetConvergenceConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 056 result does not match frozen identity.")
        return result

    selected, frozen_rows = select_budget_states(cfg)
    _atomic_write(
        cfg.output_dir / "selection.json",
        {
            "selected_count": len(selected),
            "source_games": len({row["game_seed"] for row in selected}),
            "cell_counts": dict(
                sorted(Counter(row["selection_cell"] for row in selected).items())
            ),
            "selected": selected,
        },
    )
    started = time.perf_counter()
    new_estimates = _evaluate_parallel(cfg, selected)
    evaluation_wall_seconds = time.perf_counter() - started
    rows = _combine_rows(selected, frozen_rows, new_estimates)

    scorer = _LegalActionScorer(
        cfg.parent_checkpoint,
        cfg.device,
        "legal_action_attention_scorer",
    )
    repeat_c = _evaluate_estimate(selected[0], cfg, "C", scorer)
    repeat_identical = (
        _deterministic_projection_without_timing(rows[0]["estimates"]["C"])
        == _deterministic_projection_without_timing(repeat_c)
    )
    summary = summarize_budget_convergence(
        rows,
        cfg,
        evaluation_wall_seconds=evaluation_wall_seconds,
    )
    integrity = _integrity(
        cfg,
        selected,
        frozen_rows,
        rows,
        summary,
        repeat_identical,
    )
    integrity_passed = all(integrity.values())
    point_gates = _point_gates(cfg, summary)
    equivalence_gates = _equivalence_gates(cfg, summary)
    positive_gates = _positive_convergence_gates(cfg, summary)
    if not integrity_passed:
        decision = "budget_convergence_audit_invalid_no_learning_decision"
    elif all(point_gates.values()):
        decision = "averaged_resolver_targets_stable_authorize_teacher_protocol"
    elif all(equivalence_gates.values()):
        decision = (
            "averaged_resolver_equivalence_stable_"
            "authorize_uncertainty_protocol"
        )
    elif all(positive_gates.values()):
        decision = (
            "budget_convergence_positive_but_8x_insufficient_"
            "redesign_teacher"
        )
    else:
        decision = "budget_averaging_does_not_stabilize_resolver_targets"
    payload = {
        "experiment": "056_resolver_budget_convergence_audit",
        "protocol": "docs/experiments/resolver_budget_convergence_audit_056.md",
        "identity": _identity(cfg),
        "selection": {
            "selected_count": len(selected),
            "unique_states": len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            ),
            "source_games": len({row["game_seed"] for row in selected}),
            "cell_counts": dict(
                sorted(Counter(row["selection_cell"] for row in selected).items())
            ),
            "outcome_blind": True,
        },
        "summary": summary,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "point_target_gates": point_gates,
        "equivalence_target_gates": equivalence_gates,
        "positive_convergence_gates": positive_gates,
        "first_state_c_repeat_identical": repeat_identical,
        "rows": rows,
        "decision": decision,
        "averaged_teacher_protocol_authorized": (
            integrity_passed
            and decision.startswith("averaged_resolver_targets_stable")
        ),
        "uncertainty_protocol_authorized": (
            integrity_passed
            and decision.startswith("averaged_resolver_equivalence_stable")
        ),
        "collection_authorized": False,
        "training_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(result_path, payload)
    print(
        f"056 decision={decision} integrity={integrity_passed}",
        flush=True,
    )
    return payload


def select_budget_states(
    cfg: ResolverBudgetConvergenceConfig,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    selection = json.loads(cfg.selection_055.read_text(encoding="utf-8"))
    result = json.loads(cfg.result_055.read_text(encoding="utf-8"))
    frozen_rows = {
        int(row["sample_index"]): row for row in result["rows"]
    }
    counts: Counter[str] = Counter()
    selected = []
    for row in selection["selected"]:
        cell = str(row["selection_cell"])
        if counts[cell] >= cfg.selected_per_cell:
            continue
        sample_index = int(row["sample_index"])
        frozen = frozen_rows.get(sample_index)
        if frozen is None:
            raise ValueError("056 selected row is absent from frozen 055 result.")
        if any(not frozen["replicates"][label]["completed"] for label in ("A", "B")):
            raise ValueError("056 requires completed frozen A/B estimates.")
        selected.append(row)
        counts[cell] += 1
    if len(selected) != 36 or len(counts) != 18 or set(counts.values()) != {2}:
        raise ValueError("056 fixed balanced subset did not contain 36 states.")
    return selected, frozen_rows


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(
        Path(checkpoint),
        device,
        "legal_action_attention_scorer",
    )


def _evaluate_parallel(
    cfg: ResolverBudgetConvergenceConfig,
    selected: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    identity = _identity(cfg)
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 056 progress does not match frozen identity.")
            key = (int(payload["sample_index"]), str(payload["label"]))
            completed[key] = payload["result"]
    tasks = [
        (sample, label)
        for sample in selected
        for label in NEW_LABELS
        if (int(sample["sample_index"]), label) not in completed
    ]
    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(cfg.workers, len(tasks)),
            initializer=_initialize_worker,
            initargs=(str(cfg.parent_checkpoint), cfg.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_estimate_worker, sample, cfg, label): (
                    int(sample["sample_index"]),
                    label,
                )
                for sample, label in tasks
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    sample_index, label = futures[future]
                    result = future.result()
                    completed[(sample_index, label)] = result
                    handle.write(
                        json.dumps(
                            {
                                "identity": identity,
                                "sample_index": sample_index,
                                "label": label,
                                "result": result,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                    print(
                        f"[resolver056] searches={len(completed)}/{36 * 14}",
                        flush=True,
                    )
    return completed


def _evaluate_estimate_worker(
    sample: dict[str, Any],
    cfg: ResolverBudgetConvergenceConfig,
    label: str,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("056 worker scorer was not initialized.")
    return _evaluate_estimate(sample, cfg, label, _WORKER_SCORER)


def _evaluate_estimate(
    sample: dict[str, Any],
    cfg: ResolverBudgetConvergenceConfig,
    label: str,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    prepared = validate_selected_root(sample, scorer)
    resolver_cfg = _resolver_config(cfg, label)
    bot = FrozenPublicBeliefResolverBot(
        resolver_cfg,
        int(sample["game_seed"]),
        int(sample["actor_seat"]),
        scorer=scorer,
    )
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        _selected, counters, resolution = bot._resolve(
            prepared["state"],
            prepared["legal_actions"],
            prepared["ranked"],
            int(sample["decision_index"]),
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
    root_keys = [repr(key) for key in resolution.root_actions]
    if root_keys != prepared["root_keys"]:
        raise RuntimeError("056 replicate root keys changed after resolution.")
    values = [float(value) for value in resolution.action_values]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("056 replicate produced nonfinite values.")
    return {
        "completed": True,
        "fallback_reason": None,
        "root_keys": root_keys,
        "action_values": values,
        "best_index": int(resolution.best_index),
        "counters": {
            key: int(counters[key])
            for key in (
                "continuations",
                "resolved_continuations",
                "hard_caps",
                "loop_adjudications",
                "rollout_fallback_triggers",
            )
        },
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": time.process_time() - cpu_started,
    }


def _combine_rows(
    selected: list[dict[str, Any]],
    frozen_rows: dict[int, dict[str, Any]],
    new_estimates: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for sample in selected:
        sample_index = int(sample["sample_index"])
        frozen = frozen_rows[sample_index]
        estimates = {
            "A": frozen["replicates"]["A"],
            "B": frozen["replicates"]["B"],
        }
        for label in NEW_LABELS:
            estimates[label] = new_estimates[(sample_index, label)]
        rows.append(
            {
                "sample_index": sample_index,
                "game_seed": int(sample["game_seed"]),
                "turn_index": int(sample["turn_index"]),
                "actor_seat": int(sample["actor_seat"]),
                "phase": str(sample["phase"]),
                "gap_stratum": str(sample["gap_stratum"]),
                "selection_cell": str(sample["selection_cell"]),
                "original": sample["original"],
                "estimates": estimates,
            }
        )
    return rows


def summarize_budget_convergence(
    rows: list[dict[str, Any]],
    cfg: ResolverBudgetConvergenceConfig,
    *,
    evaluation_wall_seconds: float,
) -> dict[str, Any]:
    budgets = {}
    raw_comparisons: dict[int, list[dict[str, Any]]] = {}
    for budget, blocks in BUDGET_BLOCKS.items():
        comparisons = []
        for row in rows:
            for left_labels, right_labels in blocks:
                left = _ensemble(row["estimates"], left_labels)
                right = _ensemble(row["estimates"], right_labels)
                if left is None or right is None:
                    continue
                comparison = _comparison_row(row, left, right)
                comparison["left_labels"] = "".join(left_labels)
                comparison["right_labels"] = "".join(right_labels)
                comparisons.append(comparison)
        raw_comparisons[budget] = comparisons
        helper_cfg = _metric_helper_config(cfg, budget)
        summary = _comparison_summary(comparisons, helper_cfg)
        summary["eligible_state_comparisons"] = len(comparisons)
        summary["eligible_unique_states"] = len(
            {(row["game_seed"], row["turn_index"]) for row in comparisons}
        )
        summary["eligible_games"] = len(
            {int(row["game_seed"]) for row in comparisons}
        )
        summary["continuations_per_ensemble_side"] = 192 * budget
        summary["bootstrap"] = _cluster_bootstrap(comparisons, helper_cfg)
        budgets[str(budget)] = summary
    baseline = budgets["1"]
    for budget in (2, 4, 8):
        current = budgets[str(budget)]
        current["change_from_1x"] = {
            "exact_best_agreement": (
                current["exact_best_agreement"]
                - baseline["exact_best_agreement"]
            ),
            "mean_spearman": (
                current["mean_within_state_spearman"]
                - baseline["mean_within_state_spearman"]
            ),
            "equivalence": (
                current["value_equivalent_agreement"]
                - baseline["value_equivalent_agreement"]
            ),
            "mean_regret_ratio": (
                current["mean_symmetric_cross_regret"]
                / baseline["mean_symmetric_cross_regret"]
                if baseline["mean_symmetric_cross_regret"] > 0.0
                else math.inf
            ),
        }
    new_estimates = [
        row["estimates"][label]
        for row in rows
        for label in NEW_LABELS
    ]
    completed_new = [row for row in new_estimates if row["completed"]]
    performance = {
        "new_attempts": len(new_estimates),
        "new_completed_searches": len(completed_new),
        "evaluation_wall_seconds_this_run": evaluation_wall_seconds,
        "aggregate_search_wall_seconds": sum(
            float(row["wall_seconds"]) for row in new_estimates
        ),
        "aggregate_process_cpu_seconds": sum(
            float(row["process_cpu_seconds"]) for row in new_estimates
        ),
        "completed_searches_per_wall_second": (
            len(completed_new) / evaluation_wall_seconds
            if evaluation_wall_seconds > 0.0
            else math.inf
        ),
    }
    return {
        "budgets": budgets,
        "external_original_vs_full16": _external_original_summary(rows, cfg),
        "performance": performance,
    }


def _ensemble(
    estimates: dict[str, dict[str, Any]],
    labels: tuple[str, ...],
) -> dict[str, Any] | None:
    selected = [estimates[label] for label in labels]
    if any(not estimate["completed"] for estimate in selected):
        return None
    values = [
        statistics.fmean(
            float(estimate["action_values"][action_index])
            for estimate in selected
        )
        for action_index in range(4)
    ]
    return {
        "action_values": values,
        "best_index": values.index(max(values)),
    }


def _external_original_summary(
    rows: list[dict[str, Any]],
    cfg: ResolverBudgetConvergenceConfig,
) -> dict[str, Any]:
    eligible = []
    for row in rows:
        full = _ensemble(row["estimates"], LABELS)
        if full is None:
            continue
        original_best = int(row["original"]["best_index"])
        values = [float(value) for value in full["action_values"]]
        regret = max(values) - values[original_best]
        eligible.append(
            {
                "exact_best_agreement": original_best == int(full["best_index"]),
                "regret": regret,
            }
        )
    return {
        "states": len(eligible),
        "exact_best_agreement": (
            statistics.fmean(row["exact_best_agreement"] for row in eligible)
            if eligible
            else 0.0
        ),
        "mean_regret": (
            statistics.fmean(float(row["regret"]) for row in eligible)
            if eligible
            else math.inf
        ),
        "regret_at_most_0025": (
            statistics.fmean(
                float(row["regret"]) <= cfg.equivalence_regret
                for row in eligible
            )
            if eligible
            else 0.0
        ),
    }


def _integrity(
    cfg: ResolverBudgetConvergenceConfig,
    selected: list[dict[str, Any]],
    frozen_rows: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    repeat_identical: bool,
) -> dict[str, bool]:
    new = [
        row["estimates"][label]
        for row in rows
        for label in NEW_LABELS
    ]
    completed = [estimate for estimate in new if estimate["completed"]]
    continuations = sum(
        int(estimate["counters"]["continuations"]) for estimate in new
    )
    resolved = sum(
        int(estimate["counters"]["resolved_continuations"]) for estimate in new
    )
    hard_caps = sum(
        int(estimate["counters"]["hard_caps"]) for estimate in new
    )
    adjudications = sum(
        int(estimate["counters"]["loop_adjudications"]) for estimate in new
    )
    k8_rows = summary["budgets"]["8"]
    k8_state_keys = {
        (row["game_seed"], row["turn_index"])
        for row in rows
        if _ensemble(row["estimates"], LABELS[:8]) is not None
        and _ensemble(row["estimates"], LABELS[8:]) is not None
    }
    eligible_cells = {
        row["selection_cell"]
        for row in rows
        if (row["game_seed"], row["turn_index"]) in k8_state_keys
    }
    frozen_reproduced = all(
        row["estimates"][label]
        == frozen_rows[int(row["sample_index"])]["replicates"][label]
        for row in rows
        for label in ("A", "B")
    )
    finite = all(
        (
            not estimate["completed"]
            or all(
                math.isfinite(float(value))
                for value in estimate["action_values"]
            )
        )
        for estimate in new
    ) and _all_numeric_finite(summary)
    return {
        "frozen_prerequisites_and_configuration": True,
        "exact_36_unique_outcome_blind_states": (
            len(selected) == 36
            and len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            )
            == 36
        ),
        "two_per_selection_cell": (
            len({row["selection_cell"] for row in selected}) == 18
            and set(Counter(row["selection_cell"] for row in selected).values())
            == {2}
        ),
        "at_least_15_source_games": (
            len({row["game_seed"] for row in selected}) >= 15
        ),
        "outcome_blind_selection": True,
        "root_and_legal_alignment": True,
        "frozen_a_b_reproduced": frozen_reproduced,
        "exact_504_new_attempts": len(new) == 504,
        "k8_eligible_fraction": (
            k8_rows["eligible_unique_states"] / len(selected)
            >= cfg.minimum_8x_eligible_fraction
        ),
        "every_cell_k8_eligible": len(eligible_cells) == 18,
        "exact_new_continuation_accounting": (
            continuations == len(completed) * 192
            and continuations <= 96_768
        ),
        "all_new_continuations_resolved": resolved == continuations,
        "zero_new_hard_caps": hard_caps == 0,
        "bounded_new_loop_adjudication": (
            adjudications / continuations
            if continuations
            else math.inf
        )
        <= cfg.maximum_loop_adjudicated_fraction,
        "finite_values_and_metrics": finite,
        "first_state_c_repeat_identical": repeat_identical,
    }


def _point_gates(
    cfg: ResolverBudgetConvergenceConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    one = summary["budgets"]["1"]
    eight = summary["budgets"]["8"]
    changes = eight["change_from_1x"]
    phase = eight["breakdowns"]["phase"]
    return {
        "exact_best_agreement": (
            eight["exact_best_agreement"]
            >= cfg.point_minimum_exact_agreement
        ),
        "exact_agreement_ci_lower": (
            eight["bootstrap"]["exact_best_agreement"]["lower"]
            >= cfg.point_minimum_exact_ci_lower
        ),
        "flattened_value_pearson": (
            eight["flattened_value_pearson"]
            >= cfg.point_minimum_pearson
        ),
        "mean_spearman": (
            eight["mean_within_state_spearman"]
            >= cfg.point_minimum_spearman
        ),
        "mean_regret": (
            eight["mean_symmetric_cross_regret"]
            <= cfg.point_maximum_mean_regret
        ),
        "p90_regret": (
            eight["p90_symmetric_cross_regret"]
            <= cfg.point_maximum_p90_regret
        ),
        "equivalence": (
            eight["value_equivalent_agreement"]
            >= cfg.point_minimum_equivalence
        ),
        "equivalence_ci_lower": (
            eight["bootstrap"]["value_equivalent_agreement"]["lower"]
            >= cfg.point_minimum_equivalence_ci_lower
        ),
        "loose_regret_fraction": (
            eight["directional_regret_at_most_005"]
            >= cfg.point_minimum_loose_regret_fraction
        ),
        "phase_floors": all(
            row["exact_best_agreement"] >= cfg.point_minimum_phase_exact
            and row["value_equivalent_agreement"]
            >= cfg.point_minimum_phase_equivalence
            for row in phase.values()
        ),
        "exact_gain_from_1x": (
            changes["exact_best_agreement"]
            >= cfg.point_minimum_exact_gain
        ),
        "spearman_gain_from_1x": (
            changes["mean_spearman"]
            >= cfg.point_minimum_spearman_gain
        ),
        "regret_reduction_from_1x": (
            changes["mean_regret_ratio"]
            <= cfg.point_maximum_regret_ratio
        ),
        "baseline_present": one["eligible_state_comparisons"] > 0,
    }


def _equivalence_gates(
    cfg: ResolverBudgetConvergenceConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    eight = summary["budgets"]["8"]
    changes = eight["change_from_1x"]
    return {
        "equivalence": (
            eight["value_equivalent_agreement"]
            >= cfg.equivalence_minimum_rate
        ),
        "equivalence_ci_lower": (
            eight["bootstrap"]["value_equivalent_agreement"]["lower"]
            >= cfg.equivalence_minimum_ci_lower
        ),
        "mean_regret": (
            eight["mean_symmetric_cross_regret"]
            <= cfg.equivalence_maximum_mean_regret
        ),
        "p90_regret": (
            eight["p90_symmetric_cross_regret"]
            <= cfg.equivalence_maximum_p90_regret
        ),
        "phase_and_seat_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.equivalence_minimum_phase_seat
            for dimension in ("phase", "seat")
            for row in eight["breakdowns"][dimension].values()
        ),
        "equivalence_gain_from_1x": (
            changes["equivalence"] >= cfg.equivalence_minimum_gain
        ),
        "regret_reduction_from_1x": (
            changes["mean_regret_ratio"]
            <= cfg.equivalence_maximum_regret_ratio
        ),
    }


def _positive_convergence_gates(
    cfg: ResolverBudgetConvergenceConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    changes = summary["budgets"]["8"]["change_from_1x"]
    return {
        "exact_gain_from_1x": (
            changes["exact_best_agreement"]
            >= cfg.positive_minimum_exact_gain
        ),
        "regret_reduction_from_1x": (
            changes["mean_regret_ratio"]
            <= cfg.positive_maximum_regret_ratio
        ),
    }


def _resolver_config(
    cfg: ResolverBudgetConvergenceConfig,
    label: str,
) -> ResolverGameplayConfig:
    if label not in LABELS:
        raise ValueError(f"Unknown 056 estimate label: {label}")
    index = LABELS.index(label) + 1
    return replace(
        ResolverGameplayConfig(),
        checkpoint=cfg.parent_checkpoint,
        belief_root_index_start=(10_000_000 * index) + 1_000_000,
        continuation_seed_start=1_000_000_000_000_000 * index,
        device=cfg.device,
    )


def _metric_helper_config(
    cfg: ResolverBudgetConvergenceConfig,
    budget: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        equivalence_regret=cfg.equivalence_regret,
        loose_regret=cfg.loose_regret,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=cfg.bootstrap_seed + budget,
    )


def _zero_counters() -> dict[str, int]:
    return {
        "continuations": 0,
        "resolved_continuations": 0,
        "hard_caps": 0,
        "loop_adjudications": 0,
        "rollout_fallback_triggers": 0,
    }


def _deterministic_projection_without_timing(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in _deterministic_projection(payload).items()
        if key not in {"wall_seconds", "process_cpu_seconds"}
    }


def _all_numeric_finite(payload: Any) -> bool:
    if isinstance(payload, dict):
        return all(_all_numeric_finite(value) for value in payload.values())
    if isinstance(payload, list | tuple):
        return all(_all_numeric_finite(value) for value in payload)
    if isinstance(payload, float):
        return math.isfinite(payload)
    return True


def _validate_config(cfg: ResolverBudgetConvergenceConfig) -> None:
    frozen = {
        cfg.result_055: RESULT_055_SHA256,
        cfg.selection_055: SELECTION_055_SHA256,
        cfg.validation_replay: VALIDATION_052_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 056 prerequisite mismatch: {path}")
    result = json.loads(cfg.result_055.read_text(encoding="utf-8"))
    if (
        not result.get("integrity_passed")
        or result.get("decision")
        != "resolver_targets_too_noisy_block_direct_distillation"
    ):
        raise ValueError("056 requires the frozen integrity-valid 055 decision.")
    if cfg != ResolverBudgetConvergenceConfig():
        raise ValueError("056 frozen configuration changed.")
    base = ResolverGameplayConfig()
    for label in LABELS:
        resolver = _resolver_config(cfg, label)
        for field_name in asdict(base):
            if field_name in {"belief_root_index_start", "continuation_seed_start"}:
                continue
            if getattr(resolver, field_name) != getattr(base, field_name):
                raise ValueError(
                    f"056 estimate {label} changed resolver field {field_name}."
                )


def _identity(cfg: ResolverBudgetConvergenceConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "sha256": {
            "result_055": _sha256_file(cfg.result_055),
            "selection_055": _sha256_file(cfg.selection_055),
            "validation_052": _sha256_file(cfg.validation_replay),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
        },
        "namespace_starts": {
            label: {
                "belief_root_index_start": _resolver_config(
                    cfg, label
                ).belief_root_index_start,
                "continuation_seed_start": _resolver_config(
                    cfg, label
                ).continuation_seed_start,
            }
            for label in LABELS
        },
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
