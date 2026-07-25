"""Run experiment 058: early resolver variance decomposition."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
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
)
from splendor_ai.eval.run_resolver_target_repeatability_audit import (
    _cluster_bootstrap,
    _comparison_row,
    _comparison_summary,
    validate_selected_root,
)
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


RESULT_057_SHA256 = "72d935d5b3e4430a362e365b4e782bcb04aca41c0c3c2ad84549f82e859a4754"
SELECTION_055_SHA256 = "4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a"
VALIDATION_052_SHA256 = "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
_WORKER_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class EarlyVarianceConfig:
    output_dir: Path = Path("outputs/early_resolver_variance_decomposition_058")
    result_057: Path = Path(
        "outputs/adaptive_resolver_teacher_audit_057/result.json"
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
    belief_batches: int = 4
    continuation_replicates: int = 4
    determinizations: int = 3
    root_actions: int = 4
    opponent_actions: int = 4
    continuations_per_leaf: int = 2
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_base: int = 580_000_000
    continuation_seed_base: int = 58_000_000_000_000_000
    smooth_min_temperature: float = 0.05
    workers: int = 10
    bootstrap_samples: int = 20_000
    belief_bootstrap_seed: int = 580_057
    continuation_bootstrap_seed: int = 580_058
    equivalence_regret: float = 0.025
    loose_regret: float = 0.05
    maximum_loop_adjudicated_fraction: float = 0.001
    minimum_minimizer_switch_rate: float = 0.20
    minimum_smooth_exact_gain: float = 0.10
    maximum_smooth_regret_ratio: float = 0.75
    minimum_primary_fraction: float = 0.45
    minimum_fraction_lead: float = 0.15
    minimum_split_regret_ratio: float = 1.25
    device: str = "cpu"


def run_experiment(config: EarlyVarianceConfig | None = None) -> dict[str, Any]:
    cfg = config or EarlyVarianceConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 058 result does not match frozen identity.")
        return result

    selected = select_early_states(cfg)
    _atomic_write(
        cfg.output_dir / "selection.json",
        {
            "selected_count": len(selected),
            "source_games": len({row["game_seed"] for row in selected}),
            "cell_counts": dict(
                sorted(
                    Counter(
                        f"seat{row['actor_seat']}|{row['gap_stratum']}"
                        for row in selected
                    ).items()
                )
            ),
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
    repeat = _evaluate_state(selected[0], cfg, scorer)
    repeat_identical = (
        _deterministic_projection(rows[0])
        == _deterministic_projection(repeat)
    )
    summary = summarize_variance(
        rows,
        cfg,
        evaluation_wall_seconds=evaluation_wall_seconds,
    )
    integrity = _integrity(cfg, selected, rows, summary, repeat_identical)
    integrity_passed = all(integrity.values())
    attribution = _attribution_gates(cfg, summary)
    if not integrity_passed:
        decision = "early_variance_decomposition_invalid_no_design_decision"
    elif all(attribution["minimizer_primary"].values()):
        decision = (
            "hard_min_response_switching_primary_"
            "design_soft_or_visit_response_teacher"
        )
    elif all(attribution["belief_primary"].values()):
        decision = "belief_sampling_primary_design_broader_public_belief_search"
    elif all(attribution["continuation_primary"].values()):
        decision = "continuation_noise_primary_improve_rollout_evaluator"
    else:
        decision = "mixed_early_variance_design_joint_public_belief_visit_search"
    payload = {
        "experiment": "058_early_resolver_variance_decomposition",
        "protocol": "docs/experiments/early_resolver_variance_decomposition_058.md",
        "identity": _identity(cfg),
        "selection": {
            "selected_count": len(selected),
            "unique_states": len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            ),
            "source_games": len({row["game_seed"] for row in selected}),
            "cell_counts": dict(
                sorted(
                    Counter(
                        f"seat{row['actor_seat']}|{row['gap_stratum']}"
                        for row in selected
                    ).items()
                )
            ),
            "outcome_blind": True,
        },
        "summary": summary,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "attribution_gates": attribution,
        "first_state_repeat_identical": repeat_identical,
        "rows": rows,
        "decision": decision,
        "collection_authorized": False,
        "training_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(result_path, payload)
    print(f"058 decision={decision} integrity={integrity_passed}", flush=True)
    return payload


def select_early_states(cfg: EarlyVarianceConfig) -> list[dict[str, Any]]:
    selection = json.loads(cfg.selection_055.read_text(encoding="utf-8"))
    selected = [
        row for row in selection["selected"] if str(row["phase"]) == "early"
    ]
    counts = Counter(
        f"seat{row['actor_seat']}|{row['gap_stratum']}" for row in selected
    )
    if len(selected) != 24 or len(counts) != 6 or set(counts.values()) != {4}:
        raise ValueError("058 did not find the fixed 24-state early balance.")
    if len({(row["game_seed"], row["turn_index"]) for row in selected}) != 24:
        raise ValueError("058 early selection contains duplicate states.")
    if len({row["game_seed"] for row in selected}) < 12:
        raise ValueError("058 early selection spans fewer than 12 games.")
    return selected


def _evaluate_parallel(
    cfg: EarlyVarianceConfig,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    identity = _identity(cfg)
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 058 progress does not match identity.")
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
                        f"[resolver058] states={len(completed)}/{len(selected)}",
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
    cfg: EarlyVarianceConfig,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("058 worker scorer was not initialized.")
    return _evaluate_state(sample, cfg, _WORKER_SCORER)


def _evaluate_state(
    sample: dict[str, Any],
    cfg: EarlyVarianceConfig,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    prepared = validate_selected_root(sample, scorer)
    state = prepared["state"]
    root_keys = tuple(
        canonical_successor_action_key(state, row.action)
        for row in prepared["ranked"][: cfg.root_actions]
    )
    counters: Counter[str] = Counter()
    cells: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for belief_batch in range(cfg.belief_batches):
            branches = _prepare_belief_branches(
                sample,
                state,
                root_keys,
                belief_batch,
                cfg,
                scorer,
            )
            for continuation_replicate in range(
                cfg.continuation_replicates
            ):
                cells.append(
                    _evaluate_cell(
                        sample,
                        root_keys,
                        branches,
                        belief_batch,
                        continuation_replicate,
                        cfg,
                        scorer,
                        counters,
                    )
                )
    except ResolverIneligible as exc:
        return {
            "sample_index": int(sample["sample_index"]),
            "game_seed": int(sample["game_seed"]),
            "turn_index": int(sample["turn_index"]),
            "actor_seat": int(sample["actor_seat"]),
            "phase": str(sample["phase"]),
            "gap_stratum": str(sample["gap_stratum"]),
            "selection_cell": str(sample["selection_cell"]),
            "completed": False,
            "fallback_reason": exc.reason,
            "root_keys": prepared["root_keys"],
            "cells": cells,
            "counters": _counter_payload(counters),
            "wall_seconds": time.perf_counter() - wall_started,
            "process_cpu_seconds": time.process_time() - cpu_started,
        }
    return {
        "sample_index": int(sample["sample_index"]),
        "game_seed": int(sample["game_seed"]),
        "turn_index": int(sample["turn_index"]),
        "actor_seat": int(sample["actor_seat"]),
        "phase": str(sample["phase"]),
        "gap_stratum": str(sample["gap_stratum"]),
        "selection_cell": str(sample["selection_cell"]),
        "completed": True,
        "fallback_reason": None,
        "root_keys": [repr(key) for key in root_keys],
        "cells": cells,
        "counters": _counter_payload(counters),
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": time.process_time() - cpu_started,
    }


def _prepare_belief_branches(
    sample: dict[str, Any],
    state: Any,
    root_keys: tuple[tuple[object, ...], ...],
    belief_batch: int,
    cfg: EarlyVarianceConfig,
    scorer: _LegalActionScorer,
) -> list[dict[str, Any]]:
    determinizations = deterministic_belief_range(
        state,
        state.current_player,
        root_index=belief_root_index(
            cfg, int(sample["sample_index"]), belief_batch
        ),
        count=cfg.determinizations,
    )
    env = SplendorEnv()
    branches: list[dict[str, Any]] = []
    action_sets: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
    for det_index, determinized in enumerate(determinizations):
        root_action_map = actions_by_canonical_successor_key(
            determinized, env.legal_actions(determinized)
        )
        if any(key not in root_action_map for key in root_keys):
            raise ResolverIneligible("root_mapping_changed")
        for root_index, root_key in enumerate(root_keys):
            after_root = env.step(determinized, root_action_map[root_key])
            if after_root.terminal:
                raise ResolverIneligible("root_action_terminal")
            opponent_ranked = _distinct_canonical_scores(
                after_root,
                scorer.score_actions(
                    after_root, env.legal_actions(after_root)
                ),
            )
            if len(opponent_ranked) < cfg.opponent_actions:
                raise ResolverIneligible("fewer_than_four_opponent_actions")
            responses = opponent_ranked[: cfg.opponent_actions]
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
                    "root_index": root_index,
                    "root_key": root_key,
                    "after_root": after_root,
                    "information_set": information_set,
                    "responses": responses,
                }
            )
    return branches


def _evaluate_cell(
    sample: dict[str, Any],
    root_keys: tuple[tuple[object, ...], ...],
    branches: list[dict[str, Any]],
    belief_batch: int,
    continuation_replicate: int,
    cfg: EarlyVarianceConfig,
    scorer: _LegalActionScorer,
    counters: Counter[str],
) -> dict[str, Any]:
    env = SplendorEnv()
    leaves: list[ResolvedLeaf] = []
    leaf_payloads = []
    for branch in branches:
        after_root = branch["after_root"]
        for response_index, response in enumerate(branch["responses"]):
            response_key = canonical_successor_action_key(
                after_root, response.action
            )
            values = []
            for continuation_index in range(cfg.continuations_per_leaf):
                seed = continuation_seed(
                    cfg,
                    int(sample["sample_index"]),
                    continuation_replicate,
                    int(branch["determinization"]),
                    continuation_index,
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
                    raise RuntimeError("058 produced a nonfinite leaf value.")
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
            leaf_payloads.append(
                {
                    "determinization": int(branch["determinization"]),
                    "root_index": int(branch["root_index"]),
                    "response_index": response_index,
                    "continuation_values": values,
                    "leaf_mean": mean,
                }
            )
    hard = resolve_two_ply(
        leaves,
        root_keys,
        cfg.determinizations,
        payoff="discovery_payoff",
    )
    smooth_values = smooth_min_root_values(
        leaves,
        root_keys,
        cfg.determinizations,
        cfg.smooth_min_temperature,
    )
    smooth_best = max(
        range(len(smooth_values)),
        key=lambda index: (smooth_values[index], -index),
    )
    return {
        "belief_batch": belief_batch,
        "continuation_replicate": continuation_replicate,
        "hard_action_values": [float(value) for value in hard.action_values],
        "hard_best_index": int(hard.best_index),
        "smooth_action_values": smooth_values,
        "smooth_best_index": smooth_best,
        "opponent_signatures": opponent_response_signatures(hard, root_keys),
        "leaves": leaf_payloads,
    }


def smooth_min_root_values(
    leaves: list[ResolvedLeaf],
    root_actions: tuple[tuple[object, ...], ...],
    determinization_count: int,
    temperature: float,
) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("Smooth-min temperature must be positive.")
    values = []
    for root_action in root_actions:
        root_leaves = [leaf for leaf in leaves if leaf.root_action == root_action]
        groups: dict[tuple[object, ...], list[ResolvedLeaf]] = {}
        for leaf in root_leaves:
            groups.setdefault(leaf.opponent_information_set, []).append(leaf)
        root_total = 0.0
        for group in groups.values():
            determinizations = sorted(
                {leaf.determinization for leaf in group}
            )
            actions = sorted(
                {leaf.opponent_action for leaf in group}, key=repr
            )
            action_means = [
                statistics.fmean(
                    float(leaf.discovery_payoff)
                    for leaf in group
                    if leaf.opponent_action == action
                )
                for action in actions
            ]
            minimum = min(action_means)
            soft = minimum - temperature * math.log(
                statistics.fmean(
                    math.exp(-(value - minimum) / temperature)
                    for value in action_means
                )
            )
            root_total += (
                len(determinizations) / determinization_count
            ) * soft
        values.append(root_total)
    return values


def opponent_response_signatures(
    resolution: Any,
    root_actions: tuple[tuple[object, ...], ...],
) -> list[list[str]]:
    result = []
    for root in root_actions:
        root_repr = repr(root)
        selected = sorted(
            str(choice["selected_action"])
            for choice in resolution.opponent_choices
            if str(choice["root_action"]) == root_repr
        )
        result.append(selected)
    return result


def belief_root_index(
    cfg: EarlyVarianceConfig,
    sample_index: int,
    belief_batch: int,
) -> int:
    return cfg.belief_root_base + sample_index * 100 + belief_batch


def continuation_seed(
    cfg: EarlyVarianceConfig,
    sample_index: int,
    continuation_replicate: int,
    determinization: int,
    continuation: int,
) -> int:
    return (
        cfg.continuation_seed_base
        + sample_index * 1_000_000_000
        + continuation_replicate * 1_000_000
        + determinization * 10_000
        + continuation
    )


def summarize_variance(
    rows: list[dict[str, Any]],
    cfg: EarlyVarianceConfig,
    *,
    evaluation_wall_seconds: float,
) -> dict[str, Any]:
    completed = [row for row in rows if row["completed"]]
    splits = {}
    raw_split_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for aggregation in ("hard", "smooth"):
        splits[aggregation] = {}
        for axis in ("belief", "continuation"):
            comparisons = [
                _split_comparison_row(row, aggregation, axis)
                for row in completed
            ]
            helper = _metric_helper_config(cfg, axis)
            split = _comparison_summary(comparisons, helper)
            split["bootstrap"] = _cluster_bootstrap(comparisons, helper)
            splits[aggregation][axis] = split
            raw_split_rows[(aggregation, axis)] = comparisons

    anova = {}
    for aggregation in ("hard", "smooth"):
        components = [
            {
                "seat": row["actor_seat"],
                "gap_stratum": row["gap_stratum"],
                "action": action,
                **two_way_components(
                    _value_matrix(row, aggregation, action)
                ),
            }
            for row in completed
            for action in range(cfg.root_actions)
        ]
        anova[aggregation] = _anova_summary(components)

    switching = _switching_summary(completed, cfg)
    changes = _smooth_changes(splits)
    performance = {
        "states_attempted": len(rows),
        "states_completed": len(completed),
        "evaluation_wall_seconds_this_run": evaluation_wall_seconds,
        "aggregate_state_wall_seconds": sum(
            float(row["wall_seconds"]) for row in rows
        ),
        "aggregate_process_cpu_seconds": sum(
            float(row["process_cpu_seconds"]) for row in rows
        ),
        "completed_states_per_wall_second": (
            len(completed) / evaluation_wall_seconds
            if evaluation_wall_seconds > 0.0
            else math.inf
        ),
    }
    return {
        "splits": splits,
        "anova": anova,
        "opponent_switching": switching,
        "smooth_minus_hard": changes,
        "performance": performance,
    }


def _split_comparison_row(
    row: dict[str, Any],
    aggregation: str,
    axis: str,
) -> dict[str, Any]:
    value_key = f"{aggregation}_action_values"
    if axis == "belief":
        left_cells = [
            cell for cell in row["cells"] if cell["belief_batch"] < 2
        ]
        right_cells = [
            cell for cell in row["cells"] if cell["belief_batch"] >= 2
        ]
    elif axis == "continuation":
        left_cells = [
            cell
            for cell in row["cells"]
            if cell["continuation_replicate"] < 2
        ]
        right_cells = [
            cell
            for cell in row["cells"]
            if cell["continuation_replicate"] >= 2
        ]
    else:
        raise ValueError(f"Unknown 058 split axis: {axis}")
    left_values = [
        statistics.fmean(cell[value_key][action] for cell in left_cells)
        for action in range(4)
    ]
    right_values = [
        statistics.fmean(cell[value_key][action] for cell in right_cells)
        for action in range(4)
    ]
    return _comparison_row(
        row,
        {
            "action_values": left_values,
            "best_index": max(
                range(4), key=lambda index: (left_values[index], -index)
            ),
        },
        {
            "action_values": right_values,
            "best_index": max(
                range(4), key=lambda index: (right_values[index], -index)
            ),
        },
    )


def _value_matrix(
    row: dict[str, Any],
    aggregation: str,
    action: int,
) -> list[list[float]]:
    key = f"{aggregation}_action_values"
    matrix = [[0.0] * 4 for _ in range(4)]
    for cell in row["cells"]:
        matrix[int(cell["belief_batch"])][
            int(cell["continuation_replicate"])
        ] = float(cell[key][action])
    return matrix


def two_way_components(matrix: list[list[float]]) -> dict[str, float]:
    belief_count = len(matrix)
    continuation_count = len(matrix[0])
    grand = statistics.fmean(value for row in matrix for value in row)
    belief_means = [statistics.fmean(row) for row in matrix]
    continuation_means = [
        statistics.fmean(matrix[b][r] for b in range(belief_count))
        for r in range(continuation_count)
    ]
    belief_ss = continuation_count * sum(
        (value - grand) ** 2 for value in belief_means
    )
    continuation_ss = belief_count * sum(
        (value - grand) ** 2 for value in continuation_means
    )
    interaction_ss = sum(
        (
            matrix[b][r]
            - belief_means[b]
            - continuation_means[r]
            + grand
        )
        ** 2
        for b in range(belief_count)
        for r in range(continuation_count)
    )
    return {
        "belief_ss": belief_ss,
        "continuation_ss": continuation_ss,
        "interaction_ss": interaction_ss,
        "total_ss": belief_ss + continuation_ss + interaction_ss,
    }


def _anova_summary(components: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(group: list[dict[str, Any]]) -> dict[str, float]:
        belief = sum(float(row["belief_ss"]) for row in group)
        continuation = sum(float(row["continuation_ss"]) for row in group)
        interaction = sum(float(row["interaction_ss"]) for row in group)
        total = belief + continuation + interaction
        return {
            "state_actions": len(group),
            "belief_ss": belief,
            "continuation_ss": continuation,
            "interaction_ss": interaction,
            "total_ss": total,
            "belief_fraction": belief / total if total else 0.0,
            "continuation_fraction": continuation / total if total else 0.0,
            "interaction_fraction": interaction / total if total else 0.0,
        }

    return {
        "aggregate": aggregate(components),
        "breakdowns": {
            "seat": {
                str(value): aggregate(
                    [row for row in components if str(row["seat"]) == value]
                )
                for value in sorted({str(row["seat"]) for row in components})
            },
            "gap_stratum": {
                value: aggregate(
                    [
                        row
                        for row in components
                        if str(row["gap_stratum"]) == value
                    ]
                )
                for value in sorted(
                    {str(row["gap_stratum"]) for row in components}
                )
            },
        },
    }


def _switching_summary(
    rows: list[dict[str, Any]],
    cfg: EarlyVarianceConfig,
) -> dict[str, Any]:
    records = []
    for row in rows:
        for root_index in range(cfg.root_actions):
            signatures = [
                json.dumps(
                    cell["opponent_signatures"][root_index],
                    sort_keys=True,
                )
                for cell in row["cells"]
            ]
            modal = max(Counter(signatures).values())
            records.append(
                {
                    "sample_index": row["sample_index"],
                    "seat": row["actor_seat"],
                    "gap_stratum": row["gap_stratum"],
                    "root_index": root_index,
                    "modal_deviation": 1.0 - modal / len(signatures),
                    "unique_signatures": len(set(signatures)),
                }
            )
    return {
        "state_roots": len(records),
        "mean_modal_response_deviation": statistics.fmean(
            row["modal_deviation"] for row in records
        ),
        "state_root_fraction_with_any_switch": statistics.fmean(
            row["unique_signatures"] > 1 for row in records
        ),
        "mean_unique_signatures": statistics.fmean(
            row["unique_signatures"] for row in records
        ),
        "records": records,
    }


def _smooth_changes(splits: dict[str, Any]) -> dict[str, Any]:
    changes = {}
    for axis in ("belief", "continuation"):
        hard = splits["hard"][axis]
        smooth = splits["smooth"][axis]
        changes[axis] = {
            "exact_best_agreement": (
                smooth["exact_best_agreement"]
                - hard["exact_best_agreement"]
            ),
            "value_equivalent_agreement": (
                smooth["value_equivalent_agreement"]
                - hard["value_equivalent_agreement"]
            ),
            "mean_spearman": (
                smooth["mean_within_state_spearman"]
                - hard["mean_within_state_spearman"]
            ),
            "mean_regret": (
                smooth["mean_symmetric_cross_regret"]
                - hard["mean_symmetric_cross_regret"]
            ),
        }
    hard_exact = statistics.fmean(
        splits["hard"][axis]["exact_best_agreement"]
        for axis in ("belief", "continuation")
    )
    smooth_exact = statistics.fmean(
        splits["smooth"][axis]["exact_best_agreement"]
        for axis in ("belief", "continuation")
    )
    hard_regret = statistics.fmean(
        splits["hard"][axis]["mean_symmetric_cross_regret"]
        for axis in ("belief", "continuation")
    )
    smooth_regret = statistics.fmean(
        splits["smooth"][axis]["mean_symmetric_cross_regret"]
        for axis in ("belief", "continuation")
    )
    changes["average"] = {
        "hard_exact_best_agreement": hard_exact,
        "smooth_exact_best_agreement": smooth_exact,
        "exact_best_agreement_gain": smooth_exact - hard_exact,
        "hard_mean_regret": hard_regret,
        "smooth_mean_regret": smooth_regret,
        "regret_ratio": (
            smooth_regret / hard_regret if hard_regret > 0.0 else math.inf
        ),
    }
    return changes


def _attribution_gates(
    cfg: EarlyVarianceConfig,
    summary: dict[str, Any],
) -> dict[str, dict[str, bool]]:
    fractions = summary["anova"]["hard"]["aggregate"]
    belief = float(fractions["belief_fraction"])
    continuation = float(fractions["continuation_fraction"])
    hard_belief_regret = float(
        summary["splits"]["hard"]["belief"][
            "mean_symmetric_cross_regret"
        ]
    )
    hard_continuation_regret = float(
        summary["splits"]["hard"]["continuation"][
            "mean_symmetric_cross_regret"
        ]
    )
    smooth = summary["smooth_minus_hard"]["average"]
    switching = summary["opponent_switching"]
    return {
        "minimizer_primary": {
            "switch_rate": (
                switching["mean_modal_response_deviation"]
                >= cfg.minimum_minimizer_switch_rate
            ),
            "smooth_exact_gain": (
                smooth["exact_best_agreement_gain"]
                >= cfg.minimum_smooth_exact_gain
            ),
            "smooth_regret_reduction": (
                smooth["regret_ratio"] <= cfg.maximum_smooth_regret_ratio
            ),
        },
        "belief_primary": {
            "fraction": belief >= cfg.minimum_primary_fraction,
            "fraction_lead": (
                belief >= continuation + cfg.minimum_fraction_lead
            ),
            "split_regret": (
                hard_belief_regret
                >= cfg.minimum_split_regret_ratio
                * hard_continuation_regret
            ),
        },
        "continuation_primary": {
            "fraction": continuation >= cfg.minimum_primary_fraction,
            "fraction_lead": (
                continuation >= belief + cfg.minimum_fraction_lead
            ),
            "split_regret": (
                hard_continuation_regret
                >= cfg.minimum_split_regret_ratio * hard_belief_regret
            ),
        },
    }


def _integrity(
    cfg: EarlyVarianceConfig,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    repeat_identical: bool,
) -> dict[str, bool]:
    completed = [row for row in rows if row["completed"]]
    continuations = sum(
        int(row["counters"]["continuations"]) for row in rows
    )
    resolved = sum(
        int(row["counters"]["resolved_continuations"]) for row in rows
    )
    hard_caps = sum(int(row["counters"]["hard_caps"]) for row in rows)
    adjudications = sum(
        int(row["counters"]["loop_adjudications"]) for row in rows
    )
    counts = Counter(
        f"seat{row['actor_seat']}|{row['gap_stratum']}" for row in selected
    )
    exact_cells = all(
        len(row["cells"]) == 16
        and {
            (
                int(cell["belief_batch"]),
                int(cell["continuation_replicate"]),
            )
            for cell in row["cells"]
        }
        == {(b, r) for b in range(4) for r in range(4)}
        for row in completed
    )
    return {
        "frozen_prerequisites_and_configuration": True,
        "exact_24_unique_outcome_blind_early_states": (
            len(selected) == 24
            and len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            )
            == 24
            and all(str(row["phase"]) == "early" for row in selected)
        ),
        "four_per_seat_gap_cell": (
            len(counts) == 6 and set(counts.values()) == {4}
        ),
        "at_least_12_source_games": (
            len({row["game_seed"] for row in selected}) >= 12
        ),
        "outcome_blind_selection": True,
        "root_and_legal_alignment": True,
        "all_states_structurally_complete": (
            len(completed) == 24 and exact_cells
        ),
        "exact_continuation_accounting": continuations == 36_864,
        "all_continuations_resolved": resolved == continuations,
        "zero_hard_caps": hard_caps == 0,
        "bounded_loop_adjudication": (
            adjudications / continuations if continuations else math.inf
        )
        <= cfg.maximum_loop_adjudicated_fraction,
        "opponent_action_sets_stable": len(completed) == 24,
        "finite_leaf_root_summary_values": (
            _all_numeric_finite(rows) and _all_numeric_finite(summary)
        ),
        "first_state_repeat_identical": repeat_identical,
    }


def _metric_helper_config(
    cfg: EarlyVarianceConfig,
    axis: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        equivalence_regret=cfg.equivalence_regret,
        loose_regret=cfg.loose_regret,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=(
            cfg.belief_bootstrap_seed
            if axis == "belief"
            else cfg.continuation_bootstrap_seed
        ),
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


def _deterministic_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"wall_seconds", "process_cpu_seconds"}
    }


def _validate_config(cfg: EarlyVarianceConfig) -> None:
    frozen = {
        cfg.result_057: RESULT_057_SHA256,
        cfg.selection_055: SELECTION_055_SHA256,
        cfg.validation_replay: VALIDATION_052_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 058 prerequisite mismatch: {path}")
    result_057 = json.loads(cfg.result_057.read_text(encoding="utf-8"))
    if (
        not result_057.get("integrity_passed")
        or result_057.get("decision")
        != "adaptive_allocation_failed_redesign_search_teacher"
    ):
        raise ValueError("058 requires the frozen integrity-valid 057 decision.")
    if cfg != EarlyVarianceConfig():
        raise ValueError("058 frozen configuration changed.")


def _identity(cfg: EarlyVarianceConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "sha256": {
            "result_057": _sha256_file(cfg.result_057),
            "selection_055": _sha256_file(cfg.selection_055),
            "validation_052": _sha256_file(cfg.validation_replay),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
        },
        "continuation_seed_pairing_excludes": [
            "belief_batch",
            "root_action",
            "opponent_response",
        ],
    }


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
