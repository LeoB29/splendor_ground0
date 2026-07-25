"""Run experiment 060: public-belief visit-search diagnostic."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from typing import Any

import torch

from splendor_ai.cfr import deterministic_belief_range
from splendor_ai.encoding import (
    canonical_public_belief_key,
    canonical_successor_action_key,
)
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval.run_compute_matched_belief_breadth import (
    select_breadth_states,
)
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
    _spearman,
    validate_selected_root,
)
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


RESULT_059_SHA256 = "14dad95d3e41cf2b2f108a44177fe237fa12d84deddad90e58ae6acaf7cc0d47"
SELECTION_055_SHA256 = "4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a"
VALIDATION_052_SHA256 = "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
SIDES = ("left", "right")
PHASES = ("early", "mid", "late")
_WORKER_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class VisitSearchConfig:
    output_dir: Path = Path("outputs/public_belief_visit_search_060")
    result_059: Path = Path(
        "outputs/compute_matched_belief_breadth_059/result.json"
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
    determinizations: int = 12
    simulations: int = 192
    halfway_simulations: int = 96
    root_actions: int = 4
    opponent_actions: int = 4
    prior_temperature: float = 1.0
    prior_uniform_mix: float = 0.05
    root_puct_constant: float = 1.5
    opponent_puct_constant: float = 1.5
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_base: int = 600_000_000
    continuation_seed_base: int = 60_000_000_000_000_000
    workers: int = 10
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 600_060
    equivalence_regret: float = 0.025
    loose_regret: float = 0.05
    minimum_completion_fraction: float = 0.90
    maximum_raw_jsd: float = 0.15
    maximum_early_raw_jsd: float = 0.18
    maximum_collapsed_jsd: float = 0.10
    maximum_early_collapsed_jsd: float = 0.12
    maximum_halfway_final_jsd: float = 0.10
    minimum_early_top_exact: float = 0.55
    minimum_early_equivalence: float = 0.65
    minimum_early_q_spearman: float = 0.50
    maximum_early_mean_regret: float = 0.030
    maximum_early_p90_regret: float = 0.075
    minimum_guardrail_equivalence: float = 0.65
    maximum_guardrail_mean_regret: float = 0.035
    minimum_prior_total_variation: float = 0.10
    minimum_prior_top_change_fraction: float = 0.10
    minimum_prior_jsd_margin: float = 0.01
    maximum_loop_adjudicated_fraction: float = 0.001
    device: str = "cpu"


@dataclass(slots=True)
class Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class OpponentNode:
    root_index: int
    action_keys: tuple[tuple[object, ...], ...]
    priors: tuple[float, ...]
    edges: list[Edge]


def run_experiment(
    config: VisitSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VisitSearchConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 060 result does not match frozen identity.")
        return result

    selected = select_visit_states(cfg)
    _atomic_write(
        cfg.output_dir / "selection.json",
        {
            "selected_count": len(selected),
            "source_games": len({row["game_seed"] for row in selected}),
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
        side: _run_search(selected[0], cfg, side, scorer) for side in SIDES
    }
    repeat_identical = {
        side: (
            _deterministic_projection(rows[0]["searches"][side])
            == _deterministic_projection(repeats[side])
        )
        for side in SIDES
    }
    summary = summarize_visit_search(
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
    feasibility_gates = _feasibility_gates(cfg, summary)
    if not integrity_passed:
        decision = "public_belief_visit_search_invalid_no_design_decision"
    elif all(feasibility_gates.values()):
        decision = (
            "public_belief_visit_targets_stable_"
            "authorize_separate_full_search_pilot"
        )
    else:
        decision = (
            "public_belief_visit_targets_unstable_"
            "pause_search_teacher_branch"
        )
    payload = {
        "experiment": "060_public_belief_visit_search",
        "protocol": "docs/experiments/public_belief_visit_search_060.md",
        "identity": _identity(cfg),
        "selection": {
            "selected_count": len(selected),
            "unique_states": len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            ),
            "source_games": len({row["game_seed"] for row in selected}),
            "outcome_blind": True,
        },
        "summary": summary,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "feasibility_gates": feasibility_gates,
        "first_state_repeats_identical": repeat_identical,
        "rows": rows,
        "decision": decision,
        "full_search_pilot_authorized": (
            integrity_passed and all(feasibility_gates.values())
        ),
        "collection_authorized": False,
        "training_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(result_path, payload)
    print(f"060 decision={decision} integrity={integrity_passed}", flush=True)
    return payload


def select_visit_states(cfg: VisitSearchConfig) -> list[dict[str, Any]]:
    proxy = SimpleNamespace(selection_055=cfg.selection_055)
    return select_breadth_states(proxy)


def _evaluate_parallel(
    cfg: VisitSearchConfig,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    identity = _identity(cfg)
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 060 progress does not match identity.")
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
                        f"[resolver060] states={len(completed)}/{len(selected)}",
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
    cfg: VisitSearchConfig,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("060 worker scorer was not initialized.")
    return _evaluate_state(sample, cfg, _WORKER_SCORER)


def _evaluate_state(
    sample: dict[str, Any],
    cfg: VisitSearchConfig,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    return {
        "sample_index": int(sample["sample_index"]),
        "game_seed": int(sample["game_seed"]),
        "turn_index": int(sample["turn_index"]),
        "actor_seat": int(sample["actor_seat"]),
        "phase": str(sample["phase"]),
        "gap_stratum": str(sample["gap_stratum"]),
        "selection_cell": str(sample["selection_cell"]),
        "searches": {
            side: _run_search(sample, cfg, side, scorer) for side in SIDES
        },
    }


def _run_search(
    sample: dict[str, Any],
    cfg: VisitSearchConfig,
    side: str,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    prepared = validate_selected_root(sample, scorer)
    state = prepared["state"]
    root_ranked = prepared["ranked"][: cfg.root_actions]
    root_keys = tuple(
        canonical_successor_action_key(state, row.action)
        for row in root_ranked
    )
    root_priors = score_priors(
        [float(row.score) for row in root_ranked],
        cfg.prior_temperature,
        cfg.prior_uniform_mix,
    )
    root_edges = [Edge(prior=prior) for prior in root_priors]
    particles = deterministic_belief_range(
        state,
        state.current_player,
        root_index=visit_belief_root_index(
            cfg, int(sample["sample_index"]), side
        ),
        count=cfg.determinizations,
    )
    particle_counts = [0] * cfg.determinizations
    opponent_nodes: dict[tuple[object, ...], OpponentNode] = {}
    counters: Counter[str] = Counter()
    halfway: dict[str, Any] | None = None
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    env = SplendorEnv()
    try:
        for simulation in range(cfg.simulations):
            particle_index = simulation % cfg.determinizations
            particle_counts[particle_index] += 1
            particle = particles[particle_index]
            root_map = actions_by_canonical_successor_key(
                particle, env.legal_actions(particle)
            )
            if any(key not in root_map for key in root_keys):
                raise ResolverIneligible("root_mapping_changed")
            root_index = select_puct_edge(
                root_edges,
                cfg.root_puct_constant,
                minimizing=False,
            )
            after_root = env.step(particle, root_map[root_keys[root_index]])
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
            priors = tuple(
                score_priors(
                    [float(row.score) for row in responses],
                    cfg.prior_temperature,
                    cfg.prior_uniform_mix,
                )
            )
            information_key = (
                root_keys[root_index],
                canonical_public_belief_key(
                    after_root, after_root.current_player
                ),
            )
            node = opponent_nodes.get(information_key)
            if node is None:
                node = OpponentNode(
                    root_index=root_index,
                    action_keys=response_keys,
                    priors=priors,
                    edges=[Edge(prior=prior) for prior in priors],
                )
                opponent_nodes[information_key] = node
            elif node.action_keys != response_keys or any(
                abs(left - right) > 1e-12
                for left, right in zip(node.priors, priors, strict=True)
            ):
                raise ResolverIneligible(
                    "opponent_action_set_or_prior_unstable"
                )
            response_index = select_puct_edge(
                node.edges,
                cfg.opponent_puct_constant,
                minimizing=True,
            )
            response_map = actions_by_canonical_successor_key(
                after_root, env.legal_actions(after_root)
            )
            response_key = node.action_keys[response_index]
            if response_key not in response_map:
                raise ResolverIneligible("opponent_response_mapping_changed")
            outcome = rollout_score_margin_loop_safe(
                env,
                scorer,
                after_root,
                response_map[response_key],
                int(sample["actor_seat"]),
                visit_continuation_seed(
                    cfg, int(sample["sample_index"]), side, simulation
                ),
                cfg.continuation_epsilon,
                cfg.continuation_temperature,
                cfg.continuation_top_k,
                cfg.continuation_max_turns,
                cfg.score_margin_scale,
                cfg.max_loop_fallback_triggers,
            )
            if not math.isfinite(outcome.value):
                raise RuntimeError("060 produced a nonfinite rollout value.")
            root_edges[root_index].visits += 1
            root_edges[root_index].value_sum += float(outcome.value)
            node.edges[response_index].visits += 1
            node.edges[response_index].value_sum += float(outcome.value)
            counters["continuations"] += 1
            counters["resolved_continuations"] += int(outcome.resolved)
            counters["hard_caps"] += int(outcome.hard_cap_reached)
            counters["loop_adjudications"] += int(
                outcome.loop_adjudicated
            )
            counters["rollout_fallback_triggers"] += int(
                outcome.fallback_triggers
            )
            if simulation + 1 == cfg.halfway_simulations:
                halfway = root_snapshot(root_edges)
    except ResolverIneligible as exc:
        return {
            "completed": False,
            "fallback_reason": exc.reason,
            "root_keys": prepared["root_keys"],
            "root_priors": list(root_priors),
            "root_visits": [edge.visits for edge in root_edges],
            "root_probabilities": _visit_probabilities(root_edges),
            "root_q_values": [edge.q for edge in root_edges],
            "top_visit_index": None,
            "halfway": halfway,
            "particle_counts": particle_counts,
            "opponent_nodes": _opponent_node_payload(opponent_nodes),
            "counters": _counter_payload(counters),
            "wall_seconds": time.perf_counter() - wall_started,
            "process_cpu_seconds": time.process_time() - cpu_started,
        }
    snapshot = root_snapshot(root_edges)
    top = max(
        range(cfg.root_actions),
        key=lambda index: (
            root_edges[index].visits,
            root_edges[index].q,
            root_edges[index].prior,
            -index,
        ),
    )
    return {
        "completed": True,
        "fallback_reason": None,
        "root_keys": [repr(key) for key in root_keys],
        "root_priors": list(root_priors),
        "root_visits": snapshot["visits"],
        "root_probabilities": snapshot["probabilities"],
        "root_q_values": snapshot["q_values"],
        "top_visit_index": top,
        "halfway": halfway,
        "particle_counts": particle_counts,
        "opponent_nodes": _opponent_node_payload(opponent_nodes),
        "counters": _counter_payload(counters),
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": time.process_time() - cpu_started,
    }


def score_priors(
    scores: list[float],
    temperature: float,
    uniform_mix: float,
) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("Prior temperature must be positive.")
    if not 0.0 <= uniform_mix < 1.0:
        raise ValueError("Prior uniform mix must be in [0,1).")
    maximum = max(scores)
    weights = [math.exp((score - maximum) / temperature) for score in scores]
    total = sum(weights)
    softmax = [weight / total for weight in weights]
    uniform = 1.0 / len(scores)
    return [
        (1.0 - uniform_mix) * probability + uniform_mix * uniform
        for probability in softmax
    ]


def select_puct_edge(
    edges: list[Edge],
    constant: float,
    *,
    minimizing: bool,
) -> int:
    unvisited = [index for index, edge in enumerate(edges) if edge.visits == 0]
    if unvisited:
        return max(unvisited, key=lambda index: (edges[index].prior, -index))
    total = sum(edge.visits for edge in edges)
    return max(
        range(len(edges)),
        key=lambda index: (
            (-edges[index].q if minimizing else edges[index].q)
            + constant
            * edges[index].prior
            * math.sqrt(total)
            / (1 + edges[index].visits),
            edges[index].prior,
            -index,
        ),
    )


def root_snapshot(edges: list[Edge]) -> dict[str, Any]:
    return {
        "visits": [edge.visits for edge in edges],
        "probabilities": _visit_probabilities(edges),
        "q_values": [edge.q for edge in edges],
    }


def _visit_probabilities(edges: list[Edge]) -> list[float]:
    total = sum(edge.visits for edge in edges)
    return [
        edge.visits / total if total else 0.0
        for edge in edges
    ]


def _opponent_node_payload(
    nodes: dict[tuple[object, ...], OpponentNode],
) -> list[dict[str, Any]]:
    payload = []
    for key, node in sorted(nodes.items(), key=lambda item: repr(item[0])):
        payload.append(
            {
                "information_set_sha256": hashlib.sha256(
                    repr(key).encode("utf-8")
                ).hexdigest(),
                "root_index": node.root_index,
                "action_keys": [repr(action) for action in node.action_keys],
                "priors": list(node.priors),
                "visits": [edge.visits for edge in node.edges],
                "q_values": [edge.q for edge in node.edges],
            }
        )
    return payload


def visit_belief_root_index(
    cfg: VisitSearchConfig,
    sample_index: int,
    side: str,
) -> int:
    return cfg.belief_root_base + sample_index * 10 + SIDES.index(side)


def visit_continuation_seed(
    cfg: VisitSearchConfig,
    sample_index: int,
    side: str,
    simulation: int,
) -> int:
    return (
        cfg.continuation_seed_base
        + sample_index * 1_000_000_000
        + SIDES.index(side) * 100_000_000
        + simulation
    )


def summarize_visit_search(
    rows: list[dict[str, Any]],
    cfg: VisitSearchConfig,
    *,
    evaluation_wall_seconds: float,
) -> dict[str, Any]:
    comparisons = []
    for row in rows:
        left = row["searches"]["left"]
        right = row["searches"]["right"]
        if not left["completed"] or not right["completed"]:
            continue
        comparison = _comparison_row(
            row,
            {
                "action_values": left["root_q_values"],
                "best_index": left["top_visit_index"],
            },
            {
                "action_values": right["root_q_values"],
                "best_index": right["top_visit_index"],
            },
        )
        comparison.update(_distribution_fields(left, right, cfg))
        comparisons.append(comparison)
    helper = _metric_helper_config(cfg)
    metrics = _comparison_summary(comparisons, helper)
    metrics["bootstrap"] = _cluster_bootstrap(comparisons, helper)
    metrics["distribution"] = _distribution_summary(comparisons)
    metrics["structural"] = _structural_summary(rows)
    metrics["prior_movement"] = _prior_movement_summary(rows)
    metrics["halfway"] = _halfway_summary(rows)
    metrics["frozen_059_12x1_comparison"] = _frozen_comparison(cfg, metrics)
    searches = [
        search for row in rows for search in row["searches"].values()
    ]
    metrics["performance"] = {
        "evaluation_wall_seconds_this_run": evaluation_wall_seconds,
        "aggregate_search_wall_seconds": sum(
            float(row["wall_seconds"]) for row in searches
        ),
        "aggregate_process_cpu_seconds": sum(
            float(row["process_cpu_seconds"]) for row in searches
        ),
        "attempts": len(searches),
        "completed_searches": sum(row["completed"] for row in searches),
    }
    metrics["comparison_rows"] = comparisons
    return metrics


def _distribution_fields(
    left: dict[str, Any],
    right: dict[str, Any],
    cfg: VisitSearchConfig,
) -> dict[str, float]:
    left_probability = [float(value) for value in left["root_probabilities"]]
    right_probability = [float(value) for value in right["root_probabilities"]]
    mutual = [
        index
        for index in range(cfg.root_actions)
        if max(left["root_q_values"]) - left["root_q_values"][index]
        <= cfg.equivalence_regret
        and max(right["root_q_values"]) - right["root_q_values"][index]
        <= cfg.equivalence_regret
    ]
    left_collapsed, right_collapsed = collapse_equivalence_group(
        left_probability, right_probability, mutual
    )
    return {
        "raw_jsd": jensen_shannon(left_probability, right_probability),
        "raw_total_variation": total_variation(
            left_probability, right_probability
        ),
        "collapsed_jsd": jensen_shannon(
            left_collapsed, right_collapsed
        ),
        "visit_rank_spearman": _spearman(
            left_probability, right_probability
        ),
        "mutual_equivalence_group_size": len(mutual),
    }


def collapse_equivalence_group(
    left: list[float],
    right: list[float],
    group: list[int],
) -> tuple[list[float], list[float]]:
    group_set = set(group)
    indices = [index for index in range(len(left)) if index not in group_set]
    if group:
        return (
            [sum(left[index] for index in group)] + [left[index] for index in indices],
            [sum(right[index] for index in group)] + [right[index] for index in indices],
        )
    return list(left), list(right)


def jensen_shannon(left: list[float], right: list[float]) -> float:
    middle = [(a + b) / 2.0 for a, b in zip(left, right, strict=True)]

    def divergence(source: list[float]) -> float:
        return sum(
            value * math.log(value / target, 2)
            for value, target in zip(source, middle, strict=True)
            if value > 0.0
        )

    return (divergence(left) + divergence(right)) / 2.0


def total_variation(left: list[float], right: list[float]) -> float:
    return sum(
        abs(a - b) for a, b in zip(left, right, strict=True)
    ) / 2.0


def _distribution_summary(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "mean_raw_jsd": statistics.fmean(row["raw_jsd"] for row in comparisons),
        "p90_raw_jsd": _quantile(
            sorted(row["raw_jsd"] for row in comparisons), 0.90
        ),
        "mean_raw_total_variation": statistics.fmean(
            row["raw_total_variation"] for row in comparisons
        ),
        "mean_collapsed_jsd": statistics.fmean(
            row["collapsed_jsd"] for row in comparisons
        ),
        "mean_visit_rank_spearman": statistics.fmean(
            row["visit_rank_spearman"] for row in comparisons
        ),
        "mean_mutual_equivalence_group_size": statistics.fmean(
            row["mutual_equivalence_group_size"] for row in comparisons
        ),
    }
    result["breakdowns"] = {}
    for key in ("phase", "seat", "gap_stratum"):
        result["breakdowns"][key] = {}
        for value in sorted({str(row[key]) for row in comparisons}):
            group = [row for row in comparisons if str(row[key]) == value]
            result["breakdowns"][key][value] = {
                "states": len(group),
                "mean_raw_jsd": statistics.fmean(
                    row["raw_jsd"] for row in group
                ),
                "mean_collapsed_jsd": statistics.fmean(
                    row["collapsed_jsd"] for row in group
                ),
                "mean_raw_total_variation": statistics.fmean(
                    row["raw_total_variation"] for row in group
                ),
                "mean_visit_rank_spearman": statistics.fmean(
                    row["visit_rank_spearman"] for row in group
                ),
            }
    return result


def _prior_movement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = [
        {
            "phase": row["phase"],
            "prior": search["root_priors"],
            "visit": search["root_probabilities"],
            "top_changed": (
                int(search["top_visit_index"])
                != max(
                    range(4),
                    key=lambda index: (search["root_priors"][index], -index),
                )
            ),
        }
        for row in rows
        for search in row["searches"].values()
        if search["completed"]
    ]
    jsd = [jensen_shannon(row["prior"], row["visit"]) for row in arms]
    tv = [total_variation(row["prior"], row["visit"]) for row in arms]
    entropies = [_entropy(row["visit"]) for row in arms]
    state_change = [
        any(
            search["completed"]
            and int(search["top_visit_index"])
            != max(
                range(4),
                key=lambda index: (
                    search["root_priors"][index],
                    -index,
                ),
            )
            for search in row["searches"].values()
        )
        for row in rows
    ]
    return {
        "arms": len(arms),
        "mean_prior_to_search_jsd": statistics.fmean(jsd),
        "mean_prior_to_search_total_variation": statistics.fmean(tv),
        "arm_top_change_fraction": statistics.fmean(
            row["top_changed"] for row in arms
        ),
        "state_any_top_change_fraction": statistics.fmean(state_change),
        "mean_visit_entropy": statistics.fmean(entropies),
        "mean_visit_effective_actions": statistics.fmean(
            math.exp(value) for value in entropies
        ),
    }


def _halfway_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    by_phase: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for search in row["searches"].values():
            if not search["completed"] or search["halfway"] is None:
                continue
            value = jensen_shannon(
                search["halfway"]["probabilities"],
                search["root_probabilities"],
            )
            values.append(value)
            by_phase[row["phase"]].append(value)
    return {
        "arms": len(values),
        "mean_halfway_to_final_jsd": statistics.fmean(values),
        "p90_halfway_to_final_jsd": _quantile(sorted(values), 0.90),
        "phase": {
            phase: {
                "arms": len(group),
                "mean_halfway_to_final_jsd": statistics.fmean(group),
            }
            for phase, group in sorted(by_phase.items())
        },
    }


def _structural_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phase = {}
    for phase_name in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase_name]
        both = sum(
            all(search["completed"] for search in row["searches"].values())
            for row in phase_rows
        )
        phase[phase_name] = {
            "states": len(phase_rows),
            "both_completed": both,
            "both_completed_fraction": both / len(phase_rows),
        }
    searches = [
        search for row in rows for search in row["searches"].values()
    ]
    both = sum(
        all(search["completed"] for search in row["searches"].values())
        for row in rows
    )
    return {
        "states": len(rows),
        "attempts": len(searches),
        "completed_searches": sum(row["completed"] for row in searches),
        "both_completed": both,
        "both_completed_fraction": both / len(rows),
        "fallback_reasons": dict(
            sorted(
                Counter(
                    row["fallback_reason"]
                    for row in searches
                    if not row["completed"]
                ).items()
            )
        ),
        "phase": phase,
    }


def _frozen_comparison(
    cfg: VisitSearchConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    frozen = json.loads(cfg.result_059.read_text(encoding="utf-8"))
    baseline = frozen["summary"]["configurations"]["12x1"]
    result = {
        "overall": {
            "exact_change": (
                metrics["exact_best_agreement"]
                - baseline["exact_best_agreement"]
            ),
            "equivalence_change": (
                metrics["value_equivalent_agreement"]
                - baseline["value_equivalent_agreement"]
            ),
            "mean_regret_change": (
                metrics["mean_symmetric_cross_regret"]
                - baseline["mean_symmetric_cross_regret"]
            ),
        },
        "phase": {},
    }
    for phase in PHASES:
        current = metrics["breakdowns"]["phase"][phase]
        previous = baseline["breakdowns"]["phase"][phase]
        result["phase"][phase] = {
            "exact_change": (
                current["exact_best_agreement"]
                - previous["exact_best_agreement"]
            ),
            "equivalence_change": (
                current["value_equivalent_agreement"]
                - previous["value_equivalent_agreement"]
            ),
            "mean_regret_change": (
                current["mean_symmetric_cross_regret"]
                - previous["mean_symmetric_cross_regret"]
            ),
        }
    return result


def _feasibility_gates(
    cfg: VisitSearchConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    structural = summary["structural"]
    distribution = summary["distribution"]
    early_distribution = distribution["breakdowns"]["phase"]["early"]
    early = summary["breakdowns"]["phase"]["early"]
    mid = summary["breakdowns"]["phase"]["mid"]
    late = summary["breakdowns"]["phase"]["late"]
    early_rows = [
        row for row in summary["comparison_rows"] if row["phase"] == "early"
    ]
    early_p90 = _quantile(
        sorted(row["symmetric_regret"] for row in early_rows), 0.90
    )
    movement = summary["prior_movement"]
    return {
        "overall_completion": (
            structural["both_completed_fraction"]
            >= cfg.minimum_completion_fraction
        ),
        "phase_completion": all(
            row["both_completed_fraction"] >= cfg.minimum_completion_fraction
            for row in structural["phase"].values()
        ),
        "raw_jsd": distribution["mean_raw_jsd"] <= cfg.maximum_raw_jsd,
        "early_raw_jsd": (
            early_distribution["mean_raw_jsd"]
            <= cfg.maximum_early_raw_jsd
        ),
        "collapsed_jsd": (
            distribution["mean_collapsed_jsd"]
            <= cfg.maximum_collapsed_jsd
        ),
        "early_collapsed_jsd": (
            early_distribution["mean_collapsed_jsd"]
            <= cfg.maximum_early_collapsed_jsd
        ),
        "halfway_convergence": (
            summary["halfway"]["mean_halfway_to_final_jsd"]
            <= cfg.maximum_halfway_final_jsd
        ),
        "early_top_exact": (
            early["exact_best_agreement"] >= cfg.minimum_early_top_exact
        ),
        "early_equivalence": (
            early["value_equivalent_agreement"]
            >= cfg.minimum_early_equivalence
        ),
        "early_q_spearman": (
            early["mean_spearman"] >= cfg.minimum_early_q_spearman
        ),
        "early_mean_regret": (
            early["mean_symmetric_cross_regret"]
            <= cfg.maximum_early_mean_regret
        ),
        "early_p90_regret": early_p90 <= cfg.maximum_early_p90_regret,
        "guardrail_equivalence": all(
            row["value_equivalent_agreement"]
            >= cfg.minimum_guardrail_equivalence
            for row in (mid, late)
        ),
        "guardrail_regret": all(
            row["mean_symmetric_cross_regret"]
            <= cfg.maximum_guardrail_mean_regret
            for row in (mid, late)
        ),
        "prior_total_variation": (
            movement["mean_prior_to_search_total_variation"]
            >= cfg.minimum_prior_total_variation
        ),
        "prior_top_movement": (
            movement["state_any_top_change_fraction"]
            >= cfg.minimum_prior_top_change_fraction
        ),
        "independent_agreement_beyond_prior": (
            distribution["mean_raw_jsd"]
            <= movement["mean_prior_to_search_jsd"]
            - cfg.minimum_prior_jsd_margin
        ),
    }


def _integrity(
    cfg: VisitSearchConfig,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    repeat_identical: dict[str, bool],
) -> dict[str, bool]:
    searches = [
        search for row in rows for search in row["searches"].values()
    ]
    completed = [row for row in searches if row["completed"]]
    continuations = sum(
        int(row["counters"]["continuations"]) for row in searches
    )
    resolved = sum(
        int(row["counters"]["resolved_continuations"]) for row in searches
    )
    hard_caps = sum(int(row["counters"]["hard_caps"]) for row in searches)
    adjudications = sum(
        int(row["counters"]["loop_adjudications"]) for row in searches
    )
    return {
        "frozen_prerequisites_and_configuration": True,
        "exact_frozen_48_state_balance": (
            len(selected) == 48
            and len(
                {(row["game_seed"], row["turn_index"]) for row in selected}
            )
            == 48
        ),
        "exact_17_source_games": (
            len({row["game_seed"] for row in selected}) == 17
        ),
        "outcome_blind_selection": True,
        "exact_96_search_attempts": len(searches) == 96,
        "root_legal_and_information_set_alignment": True,
        "completed_root_visit_accounting": all(
            sum(row["root_visits"]) == cfg.simulations
            and int(row["counters"]["continuations"]) == cfg.simulations
            for row in completed
        ),
        "bounded_attempted_continuations": continuations <= 18_432,
        "all_attempted_continuations_resolved": resolved == continuations,
        "zero_hard_caps": hard_caps == 0,
        "bounded_loop_adjudication": (
            adjudications / continuations if continuations else math.inf
        )
        <= cfg.maximum_loop_adjudicated_fraction,
        "stable_revisited_opponent_nodes_when_completed": True,
        "exact_particle_schedule_when_completed": all(
            row["particle_counts"] == [16] * 12 for row in completed
        ),
        "finite_values_and_metrics": (
            _all_numeric_finite(rows) and _all_numeric_finite(summary)
        ),
        "first_state_repeats_identical": all(repeat_identical.values()),
    }


def _metric_helper_config(cfg: VisitSearchConfig) -> SimpleNamespace:
    return SimpleNamespace(
        equivalence_regret=cfg.equivalence_regret,
        loose_regret=cfg.loose_regret,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=cfg.bootstrap_seed,
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


def _entropy(probabilities: list[float]) -> float:
    return -sum(
        value * math.log(value) for value in probabilities if value > 0.0
    )


def _deterministic_projection(search: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in search.items()
        if key not in {"wall_seconds", "process_cpu_seconds"}
    }


def _validate_config(cfg: VisitSearchConfig) -> None:
    frozen = {
        cfg.result_059: RESULT_059_SHA256,
        cfg.selection_055: SELECTION_055_SHA256,
        cfg.validation_replay: VALIDATION_052_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 060 prerequisite mismatch: {path}")
    result_059 = json.loads(cfg.result_059.read_text(encoding="utf-8"))
    if (
        not result_059.get("integrity_passed")
        or result_059.get("decision")
        != (
            "belief_breadth_reallocation_insufficient_"
            "move_to_public_belief_visit_search"
        )
    ):
        raise ValueError("060 requires the frozen integrity-valid 059 decision.")
    if cfg != VisitSearchConfig():
        raise ValueError("060 frozen configuration changed.")


def _identity(cfg: VisitSearchConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "sha256": {
            "result_059": _sha256_file(cfg.result_059),
            "selection_055": _sha256_file(cfg.selection_055),
            "validation_052": _sha256_file(cfg.validation_replay),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
        },
        "information_set_contract": {
            "root_shared_across_particles": True,
            "opponent_keyed_by_public_belief": True,
            "determinization_identity_excluded": True,
            "complete_root_visit_distribution": True,
        },
    }


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
