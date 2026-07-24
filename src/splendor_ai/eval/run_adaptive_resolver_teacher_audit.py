"""Run experiment 057: adaptive resolver teacher audit."""

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
from splendor_ai.eval.run_resolver_budget_convergence_audit import (
    _all_numeric_finite,
    _atomic_write,
    _deterministic_projection_without_timing,
    _ensemble,
    _zero_counters,
)
from splendor_ai.eval.run_resolver_target_repeatability_audit import (
    _cluster_bootstrap,
    _comparison_row,
    _comparison_summary,
    validate_selected_root,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


RESULT_056_SHA256 = "bc20cde5436d904a4e63a381a8173b000fcc5b66462d71194134bdab77dac081"
RESULT_055_SHA256 = "4aa3b54e77a4856c2c5cd1bb18b1a4088755bda7679bd52a7ce58627ba622714"
SELECTION_055_SHA256 = "4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
LABELS = tuple(string.ascii_uppercase[:16])
NEW_LABELS = LABELS[2:]
LEFT_LABELS = LABELS[:8]
RIGHT_LABELS = LABELS[8:]
NEW_NAMESPACE_INDICES = {
    label: 17 + offset for offset, label in enumerate(NEW_LABELS)
}
T_CRITICAL = {
    1: 3.078,
    2: 1.886,
    3: 1.638,
    4: 1.533,
    5: 1.476,
    6: 1.440,
    7: 1.415,
}
_WORKER_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveResolverTeacherConfig:
    output_dir: Path = Path("outputs/adaptive_resolver_teacher_audit_057")
    result_056: Path = Path(
        "outputs/resolver_budget_convergence_audit_056/result.json"
    )
    result_055: Path = Path(
        "outputs/resolver_target_repeatability_audit_055/result.json"
    )
    selection_055: Path = Path(
        "outputs/resolver_target_repeatability_audit_055/selection.json"
    )
    parent_checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    selected_per_cell: int = 2
    skipped_per_cell: int = 2
    workers: int = 10
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 570_057
    equivalence_regret: float = 0.025
    loose_regret: float = 0.05
    minimum_initial_samples: int = 2
    maximum_samples_per_action: int = 8
    stable_allocation_steps: int = 4
    maximum_active_radius: float = 0.075
    diagnostic_temperature: float = 0.05
    point_minimum_exact_agreement: float = 0.70
    point_minimum_exact_ci_lower: float = 0.60
    point_minimum_pearson: float = 0.85
    point_minimum_spearman: float = 0.65
    maximum_mean_regret: float = 0.025
    maximum_p90_regret: float = 0.075
    minimum_equivalence: float = 0.75
    minimum_equivalence_ci_lower: float = 0.65
    minimum_phase_exact: float = 0.55
    minimum_phase_seat_equivalence: float = 0.60
    minimum_set_jaccard: float = 0.70
    maximum_distribution_jsd: float = 0.15
    maximum_compute_ratio: float = 0.90
    maximum_exact_deficit_vs_uniform: float = 0.03
    maximum_equivalence_deficit_vs_uniform: float = 0.03
    maximum_regret_excess_vs_uniform: float = 0.005
    maximum_loop_adjudicated_fraction: float = 0.001
    device: str = "cpu"


def run_experiment(
    config: AdaptiveResolverTeacherConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AdaptiveResolverTeacherConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 057 result does not match frozen identity.")
        return result

    selected, frozen_rows = select_adaptive_states(cfg)
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
    summary = summarize_adaptive_teacher(
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
    uniform_point_gates = _uniform_point_gates(cfg, summary)
    uniform_equivalence_gates = _uniform_equivalence_gates(cfg, summary)
    if not integrity_passed:
        decision = "adaptive_teacher_audit_invalid_no_learning_decision"
    elif all(point_gates.values()):
        decision = "adaptive_point_teacher_stable_authorize_separate_corpus_pilot"
    elif all(equivalence_gates.values()):
        decision = (
            "adaptive_equivalence_teacher_stable_"
            "authorize_separate_uncertainty_corpus_pilot"
        )
    elif all(uniform_point_gates.values()) or all(
        uniform_equivalence_gates.values()
    ):
        decision = "uniform_8x_stable_but_adaptive_failed_consider_fixed_budget_teacher"
    else:
        decision = "adaptive_allocation_failed_redesign_search_teacher"
    payload = {
        "experiment": "057_adaptive_resolver_teacher_audit",
        "protocol": "docs/experiments/adaptive_resolver_teacher_audit_057.md",
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
            "overlap_with_056": 0,
        },
        "summary": summary,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "point_teacher_gates": point_gates,
        "equivalence_teacher_gates": equivalence_gates,
        "uniform_point_gates": uniform_point_gates,
        "uniform_equivalence_gates": uniform_equivalence_gates,
        "first_state_c_repeat_identical": repeat_identical,
        "rows": rows,
        "decision": decision,
        "point_teacher_corpus_pilot_authorized": (
            integrity_passed and decision.startswith("adaptive_point_teacher")
        ),
        "uncertainty_corpus_pilot_authorized": (
            integrity_passed
            and decision.startswith("adaptive_equivalence_teacher")
        ),
        "collection_authorized": False,
        "training_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(result_path, payload)
    print(f"057 decision={decision} integrity={integrity_passed}", flush=True)
    return payload


def select_adaptive_states(
    cfg: AdaptiveResolverTeacherConfig,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    selection = json.loads(cfg.selection_055.read_text(encoding="utf-8"))
    result_055 = json.loads(cfg.result_055.read_text(encoding="utf-8"))
    result_056 = json.loads(cfg.result_056.read_text(encoding="utf-8"))
    frozen_rows = {
        int(row["sample_index"]): row for row in result_055["rows"]
    }
    used_056 = {
        (int(row["game_seed"]), int(row["turn_index"]))
        for row in result_056["rows"]
    }
    seen: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in selection["selected"]:
        cell = str(row["selection_cell"])
        position = seen[cell]
        seen[cell] += 1
        if position < cfg.skipped_per_cell:
            continue
        if position >= cfg.skipped_per_cell + cfg.selected_per_cell:
            continue
        key = (int(row["game_seed"]), int(row["turn_index"]))
        if key in used_056:
            raise ValueError("057 fixed remainder overlaps 056.")
        frozen = frozen_rows.get(int(row["sample_index"]))
        if frozen is None or any(
            not frozen["replicates"][label]["completed"] for label in ("A", "B")
        ):
            raise ValueError("057 requires completed frozen A/B estimates.")
        selected.append(row)
    counts = Counter(str(row["selection_cell"]) for row in selected)
    if len(selected) != 36 or len(counts) != 18 or set(counts.values()) != {2}:
        raise ValueError("057 fixed balanced remainder did not contain 36 states.")
    return selected, frozen_rows


def adaptive_replay(
    estimates: dict[str, dict[str, Any]],
    labels: tuple[str, ...],
    cfg: AdaptiveResolverTeacherConfig,
) -> dict[str, Any]:
    if len(labels) != cfg.maximum_samples_per_action:
        raise ValueError("057 adaptive stream must contain exactly eight labels.")
    if any(not estimates[label]["completed"] for label in labels):
        raise ValueError("057 adaptive replay requires complete estimates.")
    streams = [
        [float(estimates[label]["action_values"][action]) for label in labels]
        for action in range(4)
    ]
    counts = [cfg.minimum_initial_samples] * 4
    active_history: list[tuple[int, ...]] = []
    allocation_order: list[int] = []
    stop_reason = "uniform_8x_ceiling"
    while True:
        stats = _adaptive_statistics(streams, counts, cfg)
        active = tuple(stats["active_set"])
        active_history.append(active)
        stable = (
            len(active_history) >= cfg.stable_allocation_steps + 1
            and len(
                set(
                    active_history[-(cfg.stable_allocation_steps + 1) :]
                )
            )
            == 1
        )
        active_ready = all(
            counts[index] >= cfg.minimum_initial_samples + 1
            for index in active
        )
        radius_ready = max(
            float(stats["actions"][index]["radius"]) for index in active
        ) <= cfg.maximum_active_radius
        if stable and active_ready and radius_ready:
            stop_reason = "stable_confidence_set"
            break
        candidates = [
            index
            for index in active
            if counts[index] < cfg.maximum_samples_per_action
        ]
        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda index: (
                -float(stats["actions"][index]["radius"]),
                counts[index],
                index,
            ),
        )
        counts[selected] += 1
        allocation_order.append(selected)
    final = _adaptive_statistics(streams, counts, cfg)
    means = [float(row["mean"]) for row in final["actions"]]
    best_index = max(range(4), key=lambda index: (means[index], -index))
    distribution = _diagnostic_distribution(
        means,
        final["active_set"],
        cfg.diagnostic_temperature,
    )
    return {
        "action_values": means,
        "best_index": best_index,
        "active_set": final["active_set"],
        "actions": final["actions"],
        "sample_counts": counts,
        "allocation_order": allocation_order,
        "root_units": sum(counts),
        "continuations": 48 * sum(counts),
        "compute_ratio_vs_uniform_8x": sum(counts) / 32.0,
        "diagnostic_distribution": distribution,
        "stop_reason": stop_reason,
    }


def _adaptive_statistics(
    streams: list[list[float]],
    counts: list[int],
    cfg: AdaptiveResolverTeacherConfig,
) -> dict[str, Any]:
    actions = []
    for action, count in enumerate(counts):
        values = streams[action][:count]
        mean = statistics.fmean(values)
        standard_deviation = statistics.stdev(values)
        radius = T_CRITICAL[count - 1] * standard_deviation / math.sqrt(count)
        actions.append(
            {
                "action_index": action,
                "samples": count,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "radius": radius,
                "lower": max(-1.0, mean - radius),
                "upper": min(1.0, mean + radius),
            }
        )
    best_lower = max(float(row["lower"]) for row in actions)
    active = [
        int(row["action_index"])
        for row in actions
        if float(row["upper"]) >= best_lower - cfg.equivalence_regret
    ]
    return {"actions": actions, "active_set": active}


def _diagnostic_distribution(
    means: list[float],
    active_set: list[int],
    temperature: float,
) -> list[float]:
    maximum = max(means[index] for index in active_set)
    weights = {
        index: math.exp((means[index] - maximum) / temperature)
        for index in active_set
    }
    total = sum(weights.values())
    return [weights.get(index, 0.0) / total for index in range(4)]


def summarize_adaptive_teacher(
    rows: list[dict[str, Any]],
    cfg: AdaptiveResolverTeacherConfig,
    *,
    evaluation_wall_seconds: float,
) -> dict[str, Any]:
    adaptive_rows = []
    uniform_rows = []
    adaptive_payloads = []
    for row in rows:
        left = adaptive_replay(row["estimates"], LEFT_LABELS, cfg)
        right = adaptive_replay(row["estimates"], RIGHT_LABELS, cfg)
        adaptive_comparison = _comparison_row(row, left, right)
        adaptive_comparison.update(_adaptive_pair_fields(left, right))
        adaptive_rows.append(adaptive_comparison)
        adaptive_payloads.append(
            {
                "sample_index": row["sample_index"],
                "game_seed": row["game_seed"],
                "turn_index": row["turn_index"],
                "phase": row["phase"],
                "seat": row["actor_seat"],
                "gap_stratum": row["gap_stratum"],
                "left": left,
                "right": right,
            }
        )
        uniform_left = _ensemble(row["estimates"], LEFT_LABELS)
        uniform_right = _ensemble(row["estimates"], RIGHT_LABELS)
        if uniform_left is None or uniform_right is None:
            continue
        uniform_rows.append(_comparison_row(row, uniform_left, uniform_right))
    helper = _metric_helper_config(cfg)
    adaptive = _comparison_summary(adaptive_rows, helper)
    adaptive["bootstrap"] = _cluster_bootstrap(adaptive_rows, helper)
    adaptive.update(_adaptive_summary(adaptive_rows, adaptive_payloads))
    uniform = _comparison_summary(uniform_rows, helper)
    uniform["bootstrap"] = _cluster_bootstrap(uniform_rows, helper)
    paired = _paired_changes(adaptive_rows, uniform_rows)
    new = [
        row["estimates"][label] for row in rows for label in NEW_LABELS
    ]
    performance = {
        "new_attempts": len(new),
        "new_completed_searches": sum(row["completed"] for row in new),
        "evaluation_wall_seconds_this_run": evaluation_wall_seconds,
        "aggregate_search_wall_seconds": sum(
            float(row["wall_seconds"]) for row in new
        ),
        "aggregate_process_cpu_seconds": sum(
            float(row["process_cpu_seconds"]) for row in new
        ),
        "completed_searches_per_wall_second": (
            sum(row["completed"] for row in new) / evaluation_wall_seconds
            if evaluation_wall_seconds > 0.0
            else math.inf
        ),
    }
    return {
        "adaptive": adaptive,
        "uniform_8x": uniform,
        "paired_adaptive_minus_uniform": paired,
        "adaptive_rows": adaptive_payloads,
        "performance": performance,
    }


def _adaptive_pair_fields(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_set = set(int(index) for index in left["active_set"])
    right_set = set(int(index) for index in right["active_set"])
    union = left_set | right_set
    return {
        "exact_set_agreement": left_set == right_set,
        "set_jaccard": len(left_set & right_set) / len(union),
        "distribution_jsd": _jensen_shannon(
            left["diagnostic_distribution"],
            right["diagnostic_distribution"],
        ),
        "left_root_units": int(left["root_units"]),
        "right_root_units": int(right["root_units"]),
    }


def _jensen_shannon(left: list[float], right: list[float]) -> float:
    middle = [(a + b) / 2.0 for a, b in zip(left, right, strict=True)]

    def divergence(source: list[float]) -> float:
        return sum(
            value * math.log(value / target, 2)
            for value, target in zip(source, middle, strict=True)
            if value > 0.0
        )

    return (divergence(left) + divergence(right)) / 2.0


def _adaptive_summary(
    comparisons: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    root_units = [
        int(arm["root_units"])
        for row in payloads
        for arm in (row["left"], row["right"])
    ]
    stop_reasons = Counter(
        str(arm["stop_reason"])
        for row in payloads
        for arm in (row["left"], row["right"])
    )
    compute_breakdowns: dict[str, dict[str, float]] = {}
    for key in ("phase", "seat", "gap_stratum"):
        values: dict[str, list[int]] = {}
        for row in payloads:
            group = str(row[key])
            values.setdefault(group, []).extend(
                [int(row["left"]["root_units"]), int(row["right"]["root_units"])]
            )
        compute_breakdowns[key] = {
            group: statistics.fmean(units) for group, units in values.items()
        }
    return {
        "exact_set_agreement": statistics.fmean(
            bool(row["exact_set_agreement"]) for row in comparisons
        ),
        "mean_set_jaccard": statistics.fmean(
            float(row["set_jaccard"]) for row in comparisons
        ),
        "mean_distribution_jsd": statistics.fmean(
            float(row["distribution_jsd"]) for row in comparisons
        ),
        "mean_root_units_per_arm": statistics.fmean(root_units),
        "median_root_units_per_arm": statistics.median(root_units),
        "minimum_root_units_per_arm": min(root_units),
        "maximum_root_units_per_arm": max(root_units),
        "mean_continuations_per_arm": 48 * statistics.fmean(root_units),
        "mean_compute_ratio_vs_uniform_8x": statistics.fmean(root_units) / 32.0,
        "mean_compute_saved_fraction": 1.0 - statistics.fmean(root_units) / 32.0,
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "mean_root_units_breakdowns": compute_breakdowns,
    }


def _paired_changes(
    adaptive: list[dict[str, Any]],
    uniform: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(adaptive) != len(uniform):
        raise ValueError("057 adaptive/uniform rows are not aligned.")
    rows = [
        {
            "game_seed": int(a["game_seed"]),
            "turn_index": int(a["turn_index"]),
            "phase": str(a["phase"]),
            "seat": int(a["seat"]),
            "gap_stratum": str(a["gap_stratum"]),
            "exact_best_agreement": (
                float(a["exact_best_agreement"])
                - float(u["exact_best_agreement"])
            ),
            "value_equivalent_agreement": (
                float(a["value_equivalent"])
                - float(u["value_equivalent"])
            ),
            "symmetric_cross_regret": (
                float(a["symmetric_regret"]) - float(u["symmetric_regret"])
            ),
            "spearman": float(a["spearman"]) - float(u["spearman"]),
        }
        for a, u in zip(adaptive, uniform, strict=True)
    ]
    return {
        "mean_changes": {
            "exact_best_agreement": statistics.fmean(
                row["exact_best_agreement"] for row in rows
            ),
            "value_equivalent_agreement": statistics.fmean(
                row["value_equivalent_agreement"] for row in rows
            ),
            "mean_symmetric_cross_regret": statistics.fmean(
                row["symmetric_cross_regret"] for row in rows
            ),
            "mean_spearman": statistics.fmean(
                row["spearman"] for row in rows
            ),
        },
        "rows": rows,
    }


def _point_gates(
    cfg: AdaptiveResolverTeacherConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    adaptive = summary["adaptive"]
    uniform = summary["uniform_8x"]
    return {
        "exact_agreement": (
            adaptive["exact_best_agreement"] >= cfg.point_minimum_exact_agreement
        ),
        "exact_ci_lower": (
            adaptive["bootstrap"]["exact_best_agreement"]["lower"]
            >= cfg.point_minimum_exact_ci_lower
        ),
        "pearson": (
            adaptive["flattened_value_pearson"] >= cfg.point_minimum_pearson
        ),
        "spearman": (
            adaptive["mean_within_state_spearman"]
            >= cfg.point_minimum_spearman
        ),
        "mean_regret": (
            adaptive["mean_symmetric_cross_regret"] <= cfg.maximum_mean_regret
        ),
        "p90_regret": (
            adaptive["p90_symmetric_cross_regret"] <= cfg.maximum_p90_regret
        ),
        "equivalence": (
            adaptive["value_equivalent_agreement"] >= cfg.minimum_equivalence
        ),
        "equivalence_ci_lower": (
            adaptive["bootstrap"]["value_equivalent_agreement"]["lower"]
            >= cfg.minimum_equivalence_ci_lower
        ),
        "phase_exact_floor": all(
            row["exact_best_agreement"] >= cfg.minimum_phase_exact
            for row in adaptive["breakdowns"]["phase"].values()
        ),
        "phase_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in adaptive["breakdowns"]["phase"].values()
        ),
        "seat_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in adaptive["breakdowns"]["seat"].values()
        ),
        "compute_saving": (
            adaptive["mean_compute_ratio_vs_uniform_8x"]
            <= cfg.maximum_compute_ratio
        ),
        "exact_noninferior_to_uniform": (
            adaptive["exact_best_agreement"]
            >= uniform["exact_best_agreement"]
            - cfg.maximum_exact_deficit_vs_uniform
        ),
        "regret_noninferior_to_uniform": (
            adaptive["mean_symmetric_cross_regret"]
            <= uniform["mean_symmetric_cross_regret"]
            + cfg.maximum_regret_excess_vs_uniform
        ),
    }


def _equivalence_gates(
    cfg: AdaptiveResolverTeacherConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    adaptive = summary["adaptive"]
    uniform = summary["uniform_8x"]
    return {
        "equivalence": (
            adaptive["value_equivalent_agreement"] >= cfg.minimum_equivalence
        ),
        "equivalence_ci_lower": (
            adaptive["bootstrap"]["value_equivalent_agreement"]["lower"]
            >= cfg.minimum_equivalence_ci_lower
        ),
        "mean_regret": (
            adaptive["mean_symmetric_cross_regret"] <= cfg.maximum_mean_regret
        ),
        "p90_regret": (
            adaptive["p90_symmetric_cross_regret"] <= cfg.maximum_p90_regret
        ),
        "phase_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in adaptive["breakdowns"]["phase"].values()
        ),
        "seat_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in adaptive["breakdowns"]["seat"].values()
        ),
        "set_jaccard": (
            adaptive["mean_set_jaccard"] >= cfg.minimum_set_jaccard
        ),
        "distribution_jsd": (
            adaptive["mean_distribution_jsd"] <= cfg.maximum_distribution_jsd
        ),
        "compute_saving": (
            adaptive["mean_compute_ratio_vs_uniform_8x"]
            <= cfg.maximum_compute_ratio
        ),
        "equivalence_noninferior_to_uniform": (
            adaptive["value_equivalent_agreement"]
            >= uniform["value_equivalent_agreement"]
            - cfg.maximum_equivalence_deficit_vs_uniform
        ),
        "regret_noninferior_to_uniform": (
            adaptive["mean_symmetric_cross_regret"]
            <= uniform["mean_symmetric_cross_regret"]
            + cfg.maximum_regret_excess_vs_uniform
        ),
    }


def _uniform_point_gates(
    cfg: AdaptiveResolverTeacherConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    uniform = summary["uniform_8x"]
    return {
        "exact_agreement": (
            uniform["exact_best_agreement"] >= cfg.point_minimum_exact_agreement
        ),
        "exact_ci_lower": (
            uniform["bootstrap"]["exact_best_agreement"]["lower"]
            >= cfg.point_minimum_exact_ci_lower
        ),
        "pearson": uniform["flattened_value_pearson"] >= cfg.point_minimum_pearson,
        "spearman": (
            uniform["mean_within_state_spearman"] >= cfg.point_minimum_spearman
        ),
        "mean_regret": (
            uniform["mean_symmetric_cross_regret"] <= cfg.maximum_mean_regret
        ),
        "p90_regret": (
            uniform["p90_symmetric_cross_regret"] <= cfg.maximum_p90_regret
        ),
        "equivalence": (
            uniform["value_equivalent_agreement"] >= cfg.minimum_equivalence
        ),
        "equivalence_ci_lower": (
            uniform["bootstrap"]["value_equivalent_agreement"]["lower"]
            >= cfg.minimum_equivalence_ci_lower
        ),
        "phase_exact_floor": all(
            row["exact_best_agreement"] >= cfg.minimum_phase_exact
            for row in uniform["breakdowns"]["phase"].values()
        ),
        "phase_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in uniform["breakdowns"]["phase"].values()
        ),
        "seat_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in uniform["breakdowns"]["seat"].values()
        ),
    }


def _uniform_equivalence_gates(
    cfg: AdaptiveResolverTeacherConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    uniform = summary["uniform_8x"]
    return {
        "equivalence": (
            uniform["value_equivalent_agreement"] >= cfg.minimum_equivalence
        ),
        "equivalence_ci_lower": (
            uniform["bootstrap"]["value_equivalent_agreement"]["lower"]
            >= cfg.minimum_equivalence_ci_lower
        ),
        "mean_regret": (
            uniform["mean_symmetric_cross_regret"] <= cfg.maximum_mean_regret
        ),
        "p90_regret": (
            uniform["p90_symmetric_cross_regret"] <= cfg.maximum_p90_regret
        ),
        "phase_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in uniform["breakdowns"]["phase"].values()
        ),
        "seat_equivalence_floor": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_phase_seat_equivalence
            for row in uniform["breakdowns"]["seat"].values()
        ),
    }


def _combine_rows(
    selected: list[dict[str, Any]],
    frozen_rows: dict[int, dict[str, Any]],
    new_estimates: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for sample in selected:
        sample_index = int(sample["sample_index"])
        estimates = {
            "A": frozen_rows[sample_index]["replicates"]["A"],
            "B": frozen_rows[sample_index]["replicates"]["B"],
        }
        estimates.update(
            {
                label: new_estimates[(sample_index, label)]
                for label in NEW_LABELS
            }
        )
        rows.append(
            {
                "sample_index": sample_index,
                "game_seed": int(sample["game_seed"]),
                "turn_index": int(sample["turn_index"]),
                "actor_seat": int(sample["actor_seat"]),
                "phase": str(sample["phase"]),
                "gap_stratum": str(sample["gap_stratum"]),
                "selection_cell": str(sample["selection_cell"]),
                "estimates": estimates,
            }
        )
    return rows


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(
        Path(checkpoint), device, "legal_action_attention_scorer"
    )


def _evaluate_parallel(
    cfg: AdaptiveResolverTeacherConfig,
    selected: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    identity = _identity(cfg)
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 057 progress does not match identity.")
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
                        f"[resolver057] searches={len(completed)}/{36 * 14}",
                        flush=True,
                    )
    return completed


def _evaluate_estimate_worker(
    sample: dict[str, Any],
    cfg: AdaptiveResolverTeacherConfig,
    label: str,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("057 worker scorer was not initialized.")
    return _evaluate_estimate(sample, cfg, label, _WORKER_SCORER)


def _evaluate_estimate(
    sample: dict[str, Any],
    cfg: AdaptiveResolverTeacherConfig,
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
        raise RuntimeError("057 replicate root keys changed after resolution.")
    values = [float(value) for value in resolution.action_values]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("057 replicate produced nonfinite values.")
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


def _integrity(
    cfg: AdaptiveResolverTeacherConfig,
    selected: list[dict[str, Any]],
    frozen_rows: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    repeat_identical: bool,
) -> dict[str, bool]:
    new = [row["estimates"][label] for row in rows for label in NEW_LABELS]
    completed = [estimate for estimate in new if estimate["completed"]]
    continuations = sum(int(row["counters"]["continuations"]) for row in new)
    resolved = sum(
        int(row["counters"]["resolved_continuations"]) for row in new
    )
    hard_caps = sum(int(row["counters"]["hard_caps"]) for row in new)
    adjudications = sum(
        int(row["counters"]["loop_adjudications"]) for row in new
    )
    adaptive_arms = [
        arm
        for row in summary["adaptive_rows"]
        for arm in (row["left"], row["right"])
    ]
    used_056 = {
        (int(row["game_seed"]), int(row["turn_index"]))
        for row in json.loads(cfg.result_056.read_text(encoding="utf-8"))["rows"]
    }
    selected_keys = {
        (int(row["game_seed"]), int(row["turn_index"])) for row in selected
    }
    frozen_reproduced = all(
        row["estimates"][label]
        == frozen_rows[int(row["sample_index"])]["replicates"][label]
        for row in rows
        for label in ("A", "B")
    )
    prefix_safe = all(
        len(arm["sample_counts"]) == 4
        and all(
            cfg.minimum_initial_samples
            <= int(count)
            <= cfg.maximum_samples_per_action
            for count in arm["sample_counts"]
        )
        and 8 <= int(arm["root_units"]) <= 32
        and int(arm["root_units"]) == sum(arm["sample_counts"])
        and len(arm["allocation_order"]) == int(arm["root_units"]) - 8
        for arm in adaptive_arms
    )
    return {
        "frozen_prerequisites_and_configuration": True,
        "exact_36_unique_outcome_blind_states": (
            len(selected) == 36 and len(selected_keys) == 36
        ),
        "two_per_selection_cell": (
            len({row["selection_cell"] for row in selected}) == 18
            and set(
                Counter(row["selection_cell"] for row in selected).values()
            )
            == {2}
        ),
        "at_least_15_source_games": (
            len({row["game_seed"] for row in selected}) >= 15
        ),
        "no_overlap_with_056": selected_keys.isdisjoint(used_056),
        "outcome_blind_selection": True,
        "root_and_legal_alignment": True,
        "frozen_a_b_reproduced": frozen_reproduced,
        "exact_504_new_attempts": len(new) == 504,
        "all_states_both_streams_eligible": (
            len(summary["adaptive_rows"]) == 36
            and len(completed) == len(new)
        ),
        "exact_new_continuation_accounting": (
            continuations == len(completed) * 192
            and continuations <= 96_768
        ),
        "all_new_continuations_resolved": resolved == continuations,
        "zero_new_hard_caps": hard_caps == 0,
        "bounded_new_loop_adjudication": (
            adjudications / continuations if continuations else math.inf
        )
        <= cfg.maximum_loop_adjudicated_fraction,
        "adaptive_prefix_and_budget_safe": prefix_safe,
        "finite_values_and_metrics": _all_numeric_finite(summary),
        "first_state_c_repeat_identical": repeat_identical,
    }


def _metric_helper_config(
    cfg: AdaptiveResolverTeacherConfig,
) -> SimpleNamespace:
    return SimpleNamespace(
        equivalence_regret=cfg.equivalence_regret,
        loose_regret=cfg.loose_regret,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=cfg.bootstrap_seed,
    )


def _resolver_config(
    cfg: AdaptiveResolverTeacherConfig,
    label: str,
) -> ResolverGameplayConfig:
    if label not in NEW_NAMESPACE_INDICES:
        raise ValueError(f"057 has no fresh namespace for label {label}.")
    index = NEW_NAMESPACE_INDICES[label]
    return replace(
        ResolverGameplayConfig(),
        output_dir=cfg.output_dir,
        checkpoint=cfg.parent_checkpoint,
        belief_root_index_start=(10_000_000 * index) + 1_000_000,
        continuation_seed_start=1_000_000_000_000_000 * index,
        workers=cfg.workers,
        device=cfg.device,
    )


def _validate_config(cfg: AdaptiveResolverTeacherConfig) -> None:
    frozen = {
        cfg.result_056: RESULT_056_SHA256,
        cfg.result_055: RESULT_055_SHA256,
        cfg.selection_055: SELECTION_055_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 057 prerequisite mismatch: {path}")
    result_056 = json.loads(cfg.result_056.read_text(encoding="utf-8"))
    if (
        not result_056.get("integrity_passed")
        or result_056.get("decision")
        != "budget_convergence_positive_but_8x_insufficient_redesign_teacher"
    ):
        raise ValueError("057 requires the frozen integrity-valid 056 decision.")
    if cfg != AdaptiveResolverTeacherConfig():
        raise ValueError("057 frozen configuration changed.")
    base = ResolverGameplayConfig()
    for label in NEW_LABELS:
        resolver = _resolver_config(cfg, label)
        for field_name in asdict(base):
            if field_name in {
                "output_dir",
                "checkpoint",
                "belief_root_index_start",
                "continuation_seed_start",
                "workers",
                "device",
            }:
                continue
            if getattr(resolver, field_name) != getattr(base, field_name):
                raise ValueError(
                    f"057 estimate {label} changed resolver field {field_name}."
                )


def _identity(cfg: AdaptiveResolverTeacherConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "sha256": {
            "result_056": _sha256_file(cfg.result_056),
            "result_055": _sha256_file(cfg.result_055),
            "selection_055": _sha256_file(cfg.selection_055),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
        },
        "new_namespace_indices": NEW_NAMESPACE_INDICES,
        "namespace_starts": {
            label: {
                "belief_root_index_start": _resolver_config(
                    cfg, label
                ).belief_root_index_start,
                "continuation_seed_start": _resolver_config(
                    cfg, label
                ).continuation_seed_start,
            }
            for label in NEW_LABELS
        },
    }


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
