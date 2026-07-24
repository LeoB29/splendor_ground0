"""Run experiment 055: resolver target repeatability and noise-ceiling audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

import torch

from splendor_ai.encoding import ActionCodec, canonical_successor_action_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
    ResolverIneligible,
    _distinct_canonical_scores,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


RESULT_051_SHA256 = "1a731e5edf3c63cffb2b46d8adaebba9fc50b81e490367bb0c8448ef6a885ed4"
VALIDATION_052_SHA256 = "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052"
RESULT_054_SHA256 = "2e5c062e3ed12688ba66a031c1aff368f34cf62d8f9cb21b7c9ed59cdd1a7945"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
PHASES = ("early", "mid", "late")
GAP_STRATA = ("close", "medium", "clear")
REPLICATE_NAMES = ("A", "B")
_WORKER_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class ResolverTargetRepeatabilityConfig:
    output_dir: Path = Path("outputs/resolver_target_repeatability_audit_055")
    result_051: Path = Path(
        "outputs/broad_public_belief_resolver_confirmation_051/result.json"
    )
    validation_replay: Path = Path(
        "outputs/resolver_selfplay_distillation_052/validation_targets.jsonl"
    )
    result_054: Path = Path(
        "outputs/resolver_public_history_representation_054/result.json"
    )
    parent_checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    selected_per_cell: int = 4
    close_gap_maximum: float = 0.025
    medium_gap_maximum: float = 0.075
    selection_seed: int = 55
    replicate_a_belief_root_start: int = 11_000_000
    replicate_b_belief_root_start: int = 21_000_000
    replicate_a_continuation_start: int = 1_000_000_000_000_000
    replicate_b_continuation_start: int = 2_000_000_000_000_000
    workers: int = 10
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 550_055
    equivalence_regret: float = 0.025
    loose_regret: float = 0.05
    point_minimum_best_agreement: float = 0.70
    point_minimum_best_ci_lower: float = 0.60
    point_minimum_both_completed: float = 0.80
    point_minimum_pearson: float = 0.70
    point_minimum_spearman: float = 0.60
    point_maximum_mean_regret: float = 0.025
    point_minimum_loose_regret_fraction: float = 0.80
    point_minimum_clear_agreement: float = 0.80
    point_minimum_phase_seat_agreement: float = 0.55
    equivalence_minimum_rate: float = 0.75
    equivalence_minimum_ci_lower: float = 0.65
    equivalence_minimum_both_completed: float = 0.70
    equivalence_maximum_mean_regret: float = 0.035
    equivalence_maximum_p90_regret: float = 0.075
    equivalence_minimum_phase_seat_rate: float = 0.60
    maximum_loop_adjudicated_fraction: float = 0.001
    device: str = "cpu"


def run_experiment(
    config: ResolverTargetRepeatabilityConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResolverTargetRepeatabilityConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 055 result does not match frozen identity.")
        return result

    selection = select_outcome_blind_states(cfg)
    _atomic_write(cfg.output_dir / "selection.json", selection)
    rows = _evaluate_parallel(cfg, selection["selected"])
    scorer = _LegalActionScorer(
        cfg.parent_checkpoint,
        cfg.device,
        "legal_action_attention_scorer",
    )
    repeat = _evaluate_replicate(
        selection["selected"][0],
        cfg,
        "A",
        scorer,
    )
    deterministic_repeat = (
        _deterministic_projection(rows[0]["replicates"]["A"])
        == _deterministic_projection(repeat)
    )
    summary = summarize_repeatability(rows, cfg)
    integrity = _integrity(
        cfg,
        selection,
        rows,
        deterministic_repeat,
    )
    integrity_passed = all(integrity.values())
    point_gates = _point_target_gates(cfg, summary)
    equivalence_gates = _equivalence_target_gates(cfg, summary)
    if not integrity_passed:
        decision = "repeatability_audit_invalid_no_learning_decision"
    elif all(point_gates.values()):
        decision = (
            "resolver_point_targets_repeatable_"
            "authorize_richer_representation_protocol"
        )
    elif all(equivalence_gates.values()):
        decision = (
            "resolver_equivalence_targets_"
            "authorize_uncertainty_aware_protocol"
        )
    else:
        decision = "resolver_targets_too_noisy_block_direct_distillation"
    payload = {
        "experiment": "055_resolver_target_repeatability_audit",
        "protocol": "docs/experiments/resolver_target_repeatability_audit_055.md",
        "identity": _identity(cfg),
        "selection": {
            key: value for key, value in selection.items() if key != "selected"
        },
        "summary": summary,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "point_target_gates": point_gates,
        "equivalence_target_gates": equivalence_gates,
        "first_state_replicate_a_repeat_identical": deterministic_repeat,
        "rows": rows,
        "decision": decision,
        "richer_representation_protocol_authorized": (
            integrity_passed
            and decision.startswith("resolver_point_targets_repeatable")
        ),
        "uncertainty_aware_protocol_authorized": (
            integrity_passed
            and decision.startswith("resolver_equivalence_targets")
        ),
        "fresh_collection_authorized": False,
        "training_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(result_path, payload)
    print(
        f"055 decision={decision} integrity={integrity_passed}",
        flush=True,
    )
    return payload


def select_outcome_blind_states(
    cfg: ResolverTargetRepeatabilityConfig,
) -> dict[str, Any]:
    rows = [
        row
        for row in _read_jsonl(cfg.validation_replay)
        if bool(row.get("target_available", False))
    ]
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_groups = row.get("resolver_action_values", [])
        if not isinstance(raw_groups, list) or len(raw_groups) != 4:
            raise ValueError("055 requires four values in every resolved row.")
        values = [float(group["value"]) for group in raw_groups]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("055 selection found a nonfinite original value.")
        gap = _best_gap(values)
        gap_stratum = _gap_stratum(cfg, gap)
        seat = int(row["player_id"])
        phase = str(row["phase"])
        cell = f"seat:{seat}|phase:{phase}|gap:{gap_stratum}"
        cells[cell].append(
            {
                "game_seed": int(row["game_seed"]),
                "turn_index": int(row["turn_index"]),
                "decision_index": int(row["decision_index"]),
                "actor_seat": seat,
                "phase": phase,
                "gap_stratum": gap_stratum,
                "original_gap": gap,
                "state_snapshot": row["state_snapshot"],
                "original": {
                    "root_keys": [
                        str(group["canonical_successor"]) for group in raw_groups
                    ],
                    "action_values": values,
                    "best_index": _best_index(values),
                },
            }
        )
    selected: list[dict[str, Any]] = []
    available_counts = {}
    for seat in (0, 1):
        for phase in PHASES:
            for gap_stratum in GAP_STRATA:
                cell = f"seat:{seat}|phase:{phase}|gap:{gap_stratum}"
                candidates = cells[cell]
                available_counts[cell] = len(candidates)
                if len(candidates) < cfg.selected_per_cell:
                    raise ValueError(f"055 selection cell is undersupplied: {cell}")
                ordered = sorted(
                    candidates,
                    key=lambda row: _selection_hash(
                        cfg,
                        int(row["game_seed"]),
                        int(row["turn_index"]),
                    ),
                )
                selected.extend(ordered[: cfg.selected_per_cell])
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index
        row["selection_cell"] = (
            f"seat:{row['actor_seat']}|phase:{row['phase']}|"
            f"gap:{row['gap_stratum']}"
        )
    cell_counts = Counter(row["selection_cell"] for row in selected)
    return {
        "eligible_rows": len(rows),
        "selected_count": len(selected),
        "unique_states": len(
            {
                (int(row["game_seed"]), int(row["turn_index"]))
                for row in selected
            }
        ),
        "source_games": len({int(row["game_seed"]) for row in selected}),
        "available_cell_counts": dict(sorted(available_counts.items())),
        "selected_cell_counts": dict(sorted(cell_counts.items())),
        "selection_used_outcomes": False,
        "selected": selected,
    }


def validate_selected_root(
    sample: dict[str, Any],
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(sample["state_snapshot"])
    if (
        state.current_player != int(sample["actor_seat"])
        or state.turn_index != int(sample["turn_index"])
    ):
        raise ValueError("055 selected state observer/turn mismatch.")
    env = SplendorEnv()
    legal = env.legal_actions(state)
    codec = ActionCodec()
    action_indices = [codec.encode(state, action) for action in legal]
    if len(action_indices) != len(set(action_indices)):
        raise ValueError("055 selected state has duplicate legal action indices.")
    ranked = _distinct_canonical_scores(state, scorer.score_actions(state, legal))
    if len(ranked) < 4:
        raise ValueError("055 selected state has fewer than four canonical actions.")
    root_keys = [
        repr(canonical_successor_action_key(state, row.action))
        for row in ranked[:4]
    ]
    if root_keys != list(sample["original"]["root_keys"]):
        raise ValueError("055 frozen root ranking does not match recorded 052 keys.")
    return {
        "state": state,
        "legal_actions": legal,
        "ranked": ranked,
        "root_keys": root_keys,
        "unique_legal_action_indices": True,
    }


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(
        Path(checkpoint),
        device,
        "legal_action_attention_scorer",
    )


def _evaluate_parallel(
    cfg: ResolverTargetRepeatabilityConfig,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    identity = _identity(cfg)
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 055 progress does not match frozen identity.")
            completed[int(payload["sample_index"])] = payload["result"]
    missing = [
        sample
        for sample in samples
        if int(sample["sample_index"]) not in completed
    ]
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(cfg.workers, len(missing)),
            initializer=_initialize_worker,
            initargs=(str(cfg.parent_checkpoint), cfg.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_sample_worker, sample, cfg): int(
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
                        f"[resolver055] states={len(completed)}/{len(samples)}",
                        flush=True,
                    )
    return [completed[int(sample["sample_index"])] for sample in samples]


def _evaluate_sample_worker(
    sample: dict[str, Any],
    cfg: ResolverTargetRepeatabilityConfig,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("055 worker scorer was not initialized.")
    replicates = {
        name: _evaluate_replicate(sample, cfg, name, _WORKER_SCORER)
        for name in REPLICATE_NAMES
    }
    return {
        "sample_index": int(sample["sample_index"]),
        "game_seed": int(sample["game_seed"]),
        "turn_index": int(sample["turn_index"]),
        "decision_index": int(sample["decision_index"]),
        "actor_seat": int(sample["actor_seat"]),
        "phase": str(sample["phase"]),
        "gap_stratum": str(sample["gap_stratum"]),
        "selection_cell": str(sample["selection_cell"]),
        "original_gap": float(sample["original_gap"]),
        "original": sample["original"],
        "replicates": replicates,
    }


def _evaluate_replicate(
    sample: dict[str, Any],
    cfg: ResolverTargetRepeatabilityConfig,
    replicate: str,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    prepared = validate_selected_root(sample, scorer)
    resolver_cfg = _resolver_config(cfg, replicate)
    bot = FrozenPublicBeliefResolverBot(
        resolver_cfg,
        int(sample["game_seed"]),
        int(sample["actor_seat"]),
        scorer=scorer,
    )
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
            "counters": {
                "continuations": 0,
                "resolved_continuations": 0,
                "hard_caps": 0,
                "loop_adjudications": 0,
                "rollout_fallback_triggers": 0,
            },
        }
    root_keys = [repr(key) for key in resolution.root_actions]
    if root_keys != prepared["root_keys"]:
        raise RuntimeError("055 replicate root keys changed after resolution.")
    values = [float(value) for value in resolution.action_values]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("055 replicate produced nonfinite action values.")
    counter_payload = {
        key: int(counters[key])
        for key in (
            "continuations",
            "resolved_continuations",
            "hard_caps",
            "loop_adjudications",
            "rollout_fallback_triggers",
        )
    }
    return {
        "completed": True,
        "fallback_reason": None,
        "root_keys": root_keys,
        "action_values": values,
        "best_index": int(resolution.best_index),
        "counters": counter_payload,
    }


def summarize_repeatability(
    rows: list[dict[str, Any]],
    cfg: ResolverTargetRepeatabilityConfig,
) -> dict[str, Any]:
    structural = _structural_summary(rows)
    a_b_rows = [
        _comparison_row(
            row,
            row["replicates"]["A"],
            row["replicates"]["B"],
        )
        for row in rows
        if row["replicates"]["A"]["completed"]
        and row["replicates"]["B"]["completed"]
    ]
    original_a_rows = [
        _comparison_row(row, row["original"], row["replicates"]["A"])
        for row in rows
        if row["replicates"]["A"]["completed"]
    ]
    original_b_rows = [
        _comparison_row(row, row["original"], row["replicates"]["B"])
        for row in rows
        if row["replicates"]["B"]["completed"]
    ]
    a_b = _comparison_summary(a_b_rows, cfg)
    a_b["bootstrap"] = _cluster_bootstrap(a_b_rows, cfg)
    return {
        "structural": structural,
        "fresh_a_vs_b": a_b,
        "original_vs_a": _comparison_summary(original_a_rows, cfg),
        "original_vs_b": _comparison_summary(original_b_rows, cfg),
        "all_three_consensus": _all_three_summary(rows, cfg),
    }


def _structural_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = len(rows) * 2
    completed_a = sum(row["replicates"]["A"]["completed"] for row in rows)
    completed_b = sum(row["replicates"]["B"]["completed"] for row in rows)
    both = sum(
        row["replicates"]["A"]["completed"]
        and row["replicates"]["B"]["completed"]
        for row in rows
    )
    completion_agreement = sum(
        bool(row["replicates"]["A"]["completed"])
        == bool(row["replicates"]["B"]["completed"])
        for row in rows
    )
    fallback_reasons = {
        name: dict(
            sorted(
                Counter(
                    row["replicates"][name]["fallback_reason"]
                    for row in rows
                    if not row["replicates"][name]["completed"]
                ).items()
            )
        )
        for name in REPLICATE_NAMES
    }
    cells = {}
    for cell in sorted({row["selection_cell"] for row in rows}):
        group = [row for row in rows if row["selection_cell"] == cell]
        cells[cell] = {
            "states": len(group),
            "a_completed": sum(row["replicates"]["A"]["completed"] for row in group),
            "b_completed": sum(row["replicates"]["B"]["completed"] for row in group),
            "both_completed": sum(
                row["replicates"]["A"]["completed"]
                and row["replicates"]["B"]["completed"]
                for row in group
            ),
        }
    return {
        "states": len(rows),
        "attempts": attempts,
        "a_completed": completed_a,
        "b_completed": completed_b,
        "completed_searches": completed_a + completed_b,
        "a_completion_rate": completed_a / len(rows),
        "b_completion_rate": completed_b / len(rows),
        "both_completed": both,
        "both_completed_fraction": both / len(rows),
        "completion_status_agreement": completion_agreement / len(rows),
        "fallback_reasons": fallback_reasons,
        "cells": cells,
    }


def _comparison_row(
    row: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_values = [float(value) for value in left["action_values"]]
    right_values = [float(value) for value in right["action_values"]]
    left_best = int(left["best_index"])
    right_best = int(right["best_index"])
    left_to_right = max(right_values) - right_values[left_best]
    right_to_left = max(left_values) - left_values[right_best]
    return {
        "game_seed": int(row["game_seed"]),
        "turn_index": int(row["turn_index"]),
        "phase": str(row["phase"]),
        "seat": int(row["actor_seat"]),
        "gap_stratum": str(row["gap_stratum"]),
        "left_values": left_values,
        "right_values": right_values,
        "exact_best_agreement": left_best == right_best,
        "left_to_right_regret": left_to_right,
        "right_to_left_regret": right_to_left,
        "symmetric_regret": (left_to_right + right_to_left) / 2.0,
        "value_equivalent": (
            left_to_right <= 0.025 and right_to_left <= 0.025
        ),
        "spearman": _spearman(left_values, right_values),
    }


def _comparison_summary(
    rows: list[dict[str, Any]],
    cfg: ResolverTargetRepeatabilityConfig,
) -> dict[str, Any]:
    left_flat = [value for row in rows for value in row["left_values"]]
    right_flat = [value for row in rows for value in row["right_values"]]
    directional_regrets = [
        value
        for row in rows
        for value in (
            float(row["left_to_right_regret"]),
            float(row["right_to_left_regret"]),
        )
    ]
    symmetric = [float(row["symmetric_regret"]) for row in rows]
    return {
        "states": len(rows),
        "flattened_value_pearson": _pearson(left_flat, right_flat),
        "flattened_value_mae": (
            statistics.fmean(
                abs(left - right)
                for left, right in zip(left_flat, right_flat, strict=True)
            )
            if left_flat
            else math.inf
        ),
        "mean_within_state_spearman": (
            statistics.fmean(float(row["spearman"]) for row in rows)
            if rows
            else 0.0
        ),
        "exact_best_agreement": _mean_bool(rows, "exact_best_agreement"),
        "mean_symmetric_cross_regret": (
            statistics.fmean(symmetric) if symmetric else math.inf
        ),
        "p90_symmetric_cross_regret": (
            _quantile(sorted(symmetric), 0.90) if symmetric else math.inf
        ),
        "directional_regret_at_most_0025": (
            statistics.fmean(value <= cfg.equivalence_regret for value in directional_regrets)
            if directional_regrets
            else 0.0
        ),
        "directional_regret_at_most_005": (
            statistics.fmean(value <= cfg.loose_regret for value in directional_regrets)
            if directional_regrets
            else 0.0
        ),
        "value_equivalent_agreement": _mean_bool(rows, "value_equivalent"),
        "breakdowns": {
            "phase": _comparison_breakdown(rows, "phase"),
            "seat": _comparison_breakdown(rows, "seat"),
            "gap_stratum": _comparison_breakdown(rows, "gap_stratum"),
        },
    }


def _comparison_breakdown(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    for value in sorted({str(row[key]) for row in rows}):
        group = [row for row in rows if str(row[key]) == value]
        result[value] = {
            "states": len(group),
            "exact_best_agreement": _mean_bool(group, "exact_best_agreement"),
            "value_equivalent_agreement": _mean_bool(group, "value_equivalent"),
            "mean_symmetric_cross_regret": statistics.fmean(
                float(row["symmetric_regret"]) for row in group
            ),
            "mean_spearman": statistics.fmean(
                float(row["spearman"]) for row in group
            ),
        }
    return result


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    cfg: ResolverTargetRepeatabilityConfig,
) -> dict[str, dict[str, float]]:
    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[int(row["game_seed"])].append(row)
    if not clusters:
        return {
            metric: {"lower": -math.inf, "upper": math.inf}
            for metric in ("exact_best_agreement", "value_equivalent_agreement", "mean_regret")
        }
    cluster_rows = list(clusters.values())
    rng = random.Random(cfg.bootstrap_seed)
    draws = {
        "exact_best_agreement": [],
        "value_equivalent_agreement": [],
        "mean_regret": [],
    }
    for _ in range(cfg.bootstrap_samples):
        sampled = [
            row
            for _cluster in cluster_rows
            for row in cluster_rows[rng.randrange(len(cluster_rows))]
        ]
        draws["exact_best_agreement"].append(
            _mean_bool(sampled, "exact_best_agreement")
        )
        draws["value_equivalent_agreement"].append(
            _mean_bool(sampled, "value_equivalent")
        )
        draws["mean_regret"].append(
            statistics.fmean(float(row["symmetric_regret"]) for row in sampled)
        )
    return {
        key: {
            "lower": _quantile(sorted(values), 0.025),
            "upper": _quantile(sorted(values), 0.975),
        }
        for key, values in draws.items()
    }


def _all_three_summary(
    rows: list[dict[str, Any]],
    cfg: ResolverTargetRepeatabilityConfig,
) -> dict[str, Any]:
    completed = [
        row
        for row in rows
        if row["replicates"]["A"]["completed"]
        and row["replicates"]["B"]["completed"]
    ]
    exact = 0
    majority = 0
    all_equivalent = 0
    standard_deviations = []
    for row in completed:
        estimates = [
            row["original"],
            row["replicates"]["A"],
            row["replicates"]["B"],
        ]
        bests = [int(estimate["best_index"]) for estimate in estimates]
        exact += int(len(set(bests)) == 1)
        majority += int(max(Counter(bests).values()) >= 2)
        cross_regrets = []
        for chooser in estimates:
            chosen = int(chooser["best_index"])
            for evaluator in estimates:
                values = [float(value) for value in evaluator["action_values"]]
                cross_regrets.append(max(values) - values[chosen])
        all_equivalent += int(max(cross_regrets) <= cfg.equivalence_regret)
        for action_index in range(4):
            standard_deviations.append(
                statistics.pstdev(
                    float(estimate["action_values"][action_index])
                    for estimate in estimates
                )
            )
    return {
        "states": len(completed),
        "exact_unanimous_best_fraction": exact / len(completed) if completed else 0.0,
        "at_least_two_best_fraction": majority / len(completed) if completed else 0.0,
        "all_three_value_equivalent_fraction": (
            all_equivalent / len(completed) if completed else 0.0
        ),
        "mean_per_action_value_standard_deviation": (
            statistics.fmean(standard_deviations)
            if standard_deviations
            else math.inf
        ),
    }


def _integrity(
    cfg: ResolverTargetRepeatabilityConfig,
    selection: dict[str, Any],
    rows: list[dict[str, Any]],
    deterministic_repeat: bool,
) -> dict[str, bool]:
    structural = _structural_summary(rows)
    counters = [
        replicate["counters"]
        for row in rows
        for replicate in row["replicates"].values()
    ]
    completed = structural["completed_searches"]
    continuations = sum(int(counter["continuations"]) for counter in counters)
    resolved = sum(int(counter["resolved_continuations"]) for counter in counters)
    hard_caps = sum(int(counter["hard_caps"]) for counter in counters)
    adjudications = sum(int(counter["loop_adjudications"]) for counter in counters)
    finite = all(
        (
            not replicate["completed"]
            or all(math.isfinite(float(value)) for value in replicate["action_values"])
        )
        for row in rows
        for replicate in row["replicates"].values()
    )
    return {
        "frozen_prerequisites_and_configuration": True,
        "exact_72_unique_selected_states": (
            selection["selected_count"] == 72
            and selection["unique_states"] == 72
        ),
        "four_per_selection_cell": (
            len(selection["selected_cell_counts"]) == 18
            and all(
                int(value) == cfg.selected_per_cell
                for value in selection["selected_cell_counts"].values()
            )
        ),
        "at_least_18_source_games": selection["source_games"] >= 18,
        "outcome_blind_selection": not selection["selection_used_outcomes"],
        "exact_144_attempts": structural["attempts"] == 144,
        "every_cell_in_value_metrics": all(
            int(cell["both_completed"]) >= 1
            for cell in structural["cells"].values()
        ),
        "exact_continuation_accounting": (
            continuations == completed * 192 and continuations <= 27_648
        ),
        "all_continuations_resolved": resolved == continuations,
        "zero_hard_caps": hard_caps == 0,
        "bounded_loop_adjudication": (
            adjudications / continuations
            if continuations
            else math.inf
        )
        <= cfg.maximum_loop_adjudicated_fraction,
        "finite_values": finite,
        "first_state_replicate_a_repeat_identical": deterministic_repeat,
    }


def _point_target_gates(
    cfg: ResolverTargetRepeatabilityConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    comparison = summary["fresh_a_vs_b"]
    structural = summary["structural"]
    return {
        "both_completed_fraction": (
            structural["both_completed_fraction"]
            >= cfg.point_minimum_both_completed
        ),
        "exact_best_agreement": (
            comparison["exact_best_agreement"]
            >= cfg.point_minimum_best_agreement
        ),
        "best_agreement_ci_lower": (
            comparison["bootstrap"]["exact_best_agreement"]["lower"]
            >= cfg.point_minimum_best_ci_lower
        ),
        "flattened_value_pearson": (
            comparison["flattened_value_pearson"]
            >= cfg.point_minimum_pearson
        ),
        "mean_within_state_spearman": (
            comparison["mean_within_state_spearman"]
            >= cfg.point_minimum_spearman
        ),
        "mean_symmetric_cross_regret": (
            comparison["mean_symmetric_cross_regret"]
            <= cfg.point_maximum_mean_regret
        ),
        "loose_regret_fraction": (
            comparison["directional_regret_at_most_005"]
            >= cfg.point_minimum_loose_regret_fraction
        ),
        "clear_gap_best_agreement": (
            comparison["breakdowns"]["gap_stratum"]["clear"][
                "exact_best_agreement"
            ]
            >= cfg.point_minimum_clear_agreement
        ),
        "phase_and_seat_floor": _breakdown_floor(
            comparison,
            "exact_best_agreement",
            cfg.point_minimum_phase_seat_agreement,
        ),
    }


def _equivalence_target_gates(
    cfg: ResolverTargetRepeatabilityConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    comparison = summary["fresh_a_vs_b"]
    structural = summary["structural"]
    return {
        "both_completed_fraction": (
            structural["both_completed_fraction"]
            >= cfg.equivalence_minimum_both_completed
        ),
        "value_equivalent_agreement": (
            comparison["value_equivalent_agreement"]
            >= cfg.equivalence_minimum_rate
        ),
        "equivalence_ci_lower": (
            comparison["bootstrap"]["value_equivalent_agreement"]["lower"]
            >= cfg.equivalence_minimum_ci_lower
        ),
        "mean_symmetric_cross_regret": (
            comparison["mean_symmetric_cross_regret"]
            <= cfg.equivalence_maximum_mean_regret
        ),
        "p90_symmetric_cross_regret": (
            comparison["p90_symmetric_cross_regret"]
            <= cfg.equivalence_maximum_p90_regret
        ),
        "phase_and_seat_floor": _breakdown_floor(
            comparison,
            "value_equivalent_agreement",
            cfg.equivalence_minimum_phase_seat_rate,
        ),
    }


def _breakdown_floor(
    comparison: dict[str, Any],
    metric: str,
    floor: float,
) -> bool:
    return all(
        float(row[metric]) >= floor
        for dimension in ("phase", "seat")
        for row in comparison["breakdowns"][dimension].values()
    )


def _resolver_config(
    cfg: ResolverTargetRepeatabilityConfig,
    replicate: str,
) -> ResolverGameplayConfig:
    if replicate == "A":
        belief_start = cfg.replicate_a_belief_root_start
        continuation_start = cfg.replicate_a_continuation_start
    elif replicate == "B":
        belief_start = cfg.replicate_b_belief_root_start
        continuation_start = cfg.replicate_b_continuation_start
    else:
        raise ValueError(f"Unknown 055 replicate: {replicate}")
    return replace(
        ResolverGameplayConfig(),
        checkpoint=cfg.parent_checkpoint,
        belief_root_index_start=belief_start,
        continuation_seed_start=continuation_start,
        device=cfg.device,
    )


def _gap_stratum(
    cfg: ResolverTargetRepeatabilityConfig,
    gap: float,
) -> str:
    if gap <= cfg.close_gap_maximum:
        return "close"
    if gap <= cfg.medium_gap_maximum:
        return "medium"
    return "clear"


def _best_gap(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1]


def _best_index(values: list[float]) -> int:
    maximum = max(values)
    return values.index(maximum)


def _selection_hash(
    cfg: ResolverTargetRepeatabilityConfig,
    game_seed: int,
    turn_index: int,
) -> str:
    return hashlib.sha256(
        f"{cfg.selection_seed}:{game_seed}:{turn_index}".encode("utf-8")
    ).hexdigest()


def _deterministic_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


def _validate_config(cfg: ResolverTargetRepeatabilityConfig) -> None:
    frozen = {
        cfg.result_051: RESULT_051_SHA256,
        cfg.validation_replay: VALIDATION_052_SHA256,
        cfg.result_054: RESULT_054_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 055 prerequisite mismatch: {path}")
    result_051 = json.loads(cfg.result_051.read_text(encoding="utf-8"))
    if not result_051.get("passed"):
        raise ValueError("055 requires the passed broad resolver result 051.")
    result_054 = json.loads(cfg.result_054.read_text(encoding="utf-8"))
    if (
        not result_054.get("integrity_passed")
        or result_054.get("decision")
        != "resolver_targets_not_learnable_with_current_frozen_features"
    ):
        raise ValueError("055 requires the frozen integrity-valid 054 decision.")
    expected = ResolverTargetRepeatabilityConfig()
    if cfg != expected:
        raise ValueError("055 frozen configuration changed.")
    resolver_a = _resolver_config(cfg, "A")
    resolver_b = _resolver_config(cfg, "B")
    base = ResolverGameplayConfig()
    allowed = {"belief_root_index_start", "continuation_seed_start"}
    for field_name in asdict(base):
        if field_name in allowed:
            continue
        if getattr(resolver_a, field_name) != getattr(base, field_name):
            raise ValueError(f"055 replicate A changed resolver field {field_name}.")
        if getattr(resolver_b, field_name) != getattr(base, field_name):
            raise ValueError(f"055 replicate B changed resolver field {field_name}.")


def _identity(cfg: ResolverTargetRepeatabilityConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "sha256": {
            "result_051": _sha256_file(cfg.result_051),
            "validation_052": _sha256_file(cfg.validation_replay),
            "result_054": _sha256_file(cfg.result_054),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
        },
        "resolver_replicates": {
            name: {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(_resolver_config(cfg, name)).items()
            }
            for name in REPLICATE_NAMES
        },
    }


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - mean_left) ** 2 for value in left)
        * sum((value - mean_right) ** 2 for value in right)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(bool(row[key]) for row in rows) if rows else 0.0


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.inf
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
