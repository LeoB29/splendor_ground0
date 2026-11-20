"""Experiment 133: progressive public-belief planner feasibility."""

from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from splendor_ai.cfr import deterministic_belief_range
from splendor_ai.encoding import (
    canonical_public_belief_key,
    canonical_successor_action_key,
)
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval import run_expanded_root_resolver_headroom as audit085
from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval.run_public_belief_visit_search import (
    Edge,
    OpponentNode,
    collapse_equivalence_group,
    jensen_shannon,
    score_priors,
    select_puct_edge,
    total_variation,
)
from splendor_ai.eval.run_resolver_budget_convergence_audit import (
    _all_numeric_finite,
    _atomic_write,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


PROTOCOL = Path(
    "docs/engineering/progressive_public_belief_planner_feasibility_133.md"
)
PROTOCOL_SHA256 = "6da421705d1c845854dda1f271a5d96d7d4f2f45e0d7cb55b9a140829ac71902"
RESULT_086_SHA256 = "8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5"
SELECTION_086_SHA256 = "7bebc07fede835be023c986701fd813e4134374f1a4559fdcaf9439910d7e46f"
CHECKPOINT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
RESULT_132_SHA256 = "347a8098494a882e72eff03b53d535734206efa0894cb31ccb2472ceae33ba45"
ROADMAP_SHA256 = "414f1749fb146c5882a7774aa6ac5d5223efe57b74303a80c42e79612a34df79"
PHASES = ("early", "mid", "late")
CELLS = tuple(f"seat{seat}|{phase}" for seat in (0, 1) for phase in PHASES)
SIDES = ("left", "right")
_WORKER_SCORER: _LegalActionScorer | None = None


class PlannerIneligible(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ProgressivePlannerConfig:
    output_dir: Path = Path("outputs/progressive_public_belief_planner_feasibility_133")
    result_086: Path = Path(
        "outputs/expanded_root_regret_stability_replication_086/result.json"
    )
    selection_086: Path = Path(
        "outputs/expanded_root_regret_stability_replication_086/selection.json"
    )
    result_132: Path = Path(
        "outputs/alphazero_deadlock_consistent_repeated_return_value_ab_132/result.json"
    )
    roadmap: Path = Path(
        "docs/strategy/superhuman_information_set_search_roadmap_after_132.md"
    )
    checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    states_per_cell: int = 6
    champion_roots: int = 4
    determinizations: int = 12
    opponent_actions: int = 4
    simulations: int = 1536
    root_prior_temperature: float = 2.0
    root_prior_uniform_mix: float = 0.25
    opponent_prior_temperature: float = 1.0
    opponent_prior_uniform_mix: float = 0.05
    root_puct_constant: float = 1.5
    opponent_puct_constant: float = 1.5
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_base: int = 1_330_000_000
    continuation_seed_base: int = 1_330_000_000_000_000_000
    workers: int = 10
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 1_330_133
    equivalence_regret: float = 0.025
    maximum_loop_adjudicated_fraction: float = 0.001
    minimum_completion_fraction: float = 0.90
    minimum_exact_agreement: float = 0.60
    maximum_raw_jsd: float = 0.15
    maximum_early_raw_jsd: float = 0.18
    maximum_collapsed_jsd: float = 0.10
    maximum_early_collapsed_jsd: float = 0.12
    maximum_mean_cross_regret: float = 0.025
    maximum_p90_cross_regret: float = 0.075
    minimum_oracle_equivalence: float = 0.65
    maximum_mean_oracle_regret: float = 0.035
    minimum_q_spearman: float = 0.50
    minimum_prior_total_variation: float = 0.15
    minimum_prior_top_change_fraction: float = 0.15
    minimum_prior_control_margin: float = 0.02
    minimum_mean_gain: float = 0.025
    minimum_expanded_selection_fraction: float = 0.15
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class PreparedBranch:
    after_root: Any
    information_key: tuple[object, ...]
    response_keys: tuple[tuple[object, ...], ...]
    response_priors: tuple[float, ...]


def run_experiment(config: ProgressivePlannerConfig | None = None) -> dict[str, Any]:
    cfg = config or ProgressivePlannerConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    identity = _identity(cfg)
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("Existing 133 result has a different identity.")
        return payload

    selected = select_states(cfg)
    _atomic_write(
        cfg.output_dir / "selection.json",
        {
            "identity": identity,
            "outcome_blind": True,
            "states": [_selection_projection(row) for row in selected],
        },
    )
    started = time.perf_counter()
    rows = _evaluate_parallel(cfg, selected)
    wall_seconds = time.perf_counter() - started

    scorer = _LegalActionScorer(cfg.checkpoint, cfg.device, "legal_action_attention_scorer")
    repeats = {
        side: _run_search(selected[0], cfg, side, scorer) for side in SIDES
    }
    repeat_identical = {
        side: _deterministic_projection(rows[0]["searches"][side])
        == _deterministic_projection(repeats[side])
        for side in SIDES
    }
    privacy = _privacy_audit(selected[0], cfg, scorer)
    summary = summarize(rows, cfg, wall_seconds)
    integrity = integrity_gates(
        cfg, selected, rows, summary, repeat_identical, privacy
    )
    feasibility = feasibility_gates(cfg, summary)
    integrity_passed = all(integrity.values())
    feasibility_passed = integrity_passed and all(feasibility.values())
    if not integrity_passed:
        decision = "progressive_public_belief_planner_133_invalid_no_search_decision"
    elif feasibility_passed:
        decision = (
            "progressive_public_belief_planner_133_pass_"
            "freeze_fresh_confirmation_design"
        )
    else:
        decision = (
            "progressive_public_belief_planner_133_failed_"
            "review_search_mechanism"
        )
    payload = {
        "experiment": "133_progressive_public_belief_planner_feasibility",
        "status": "complete",
        "protocol": str(PROTOCOL),
        "identity": identity,
        "selection": {
            "states": len(selected),
            "cells": {cell: sum(row["selection_cell"] == cell for row in selected) for cell in CELLS},
            "outcome_blind": True,
            "development_evidence": True,
        },
        "summary": summary,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "feasibility_gates": feasibility,
        "feasibility_passed": feasibility_passed,
        "first_state_repeats_identical": repeat_identical,
        "privacy_audit": privacy,
        "rows": rows,
        "decision": decision,
        "fresh_confirmation_design_authorized": feasibility_passed,
        "gameplay_authorized": False,
        "training_authorized": False,
        "collection_authorized": False,
        "installation_authorized": False,
        "promotion_authorized": False,
        "automatic_continuation": False,
    }
    _atomic_write(result_path, payload)
    return payload


def preflight(config: ProgressivePlannerConfig | None = None) -> dict[str, Any]:
    cfg = config or ProgressivePlannerConfig()
    _validate_config(cfg)
    selected = select_states(cfg)
    scorer = _LegalActionScorer(cfg.checkpoint, cfg.device, "legal_action_attention_scorer")
    smoke = _run_search(selected[0], cfg, "left", scorer, simulation_limit=32)
    checks = {
        "input_hashes_and_decisions": _provenance_matches(cfg),
        "exact_balanced_selection": len(selected) == len(CELLS) * cfg.states_per_cell
        and all(sum(row["selection_cell"] == cell for row in selected) == cfg.states_per_cell for cell in CELLS),
        "exact_root_value_alignment": all(_row_alignment(row) for row in selected),
        "forced_coverage_fits_budget": all(
            int(row["expanded_root_count"])
            * cfg.determinizations
            * cfg.opponent_actions
            <= cfg.simulations
            for row in selected
        ),
        "smoke_completed_requested_simulations": bool(smoke["completed"])
        and int(smoke["counters"]["continuations"]) == 32,
        "smoke_all_continuations_resolved": int(smoke["counters"]["resolved_continuations"]) == 32,
        "smoke_no_hard_cap": int(smoke["counters"]["hard_caps"]) == 0,
        "no_training_or_gameplay_authority": True,
    }
    return {
        "experiment": "133_progressive_public_belief_planner_feasibility_preflight",
        "identity": _identity(cfg),
        "checks": checks,
        "passed": all(checks.values()),
        "durable_evidence_written": False,
        "training_performed": False,
        "gameplay_performed": False,
    }


def select_states(cfg: ProgressivePlannerConfig) -> list[dict[str, Any]]:
    selection_payload = json.loads(cfg.selection_086.read_text(encoding="utf-8"))
    result_payload = json.loads(cfg.result_086.read_text(encoding="utf-8"))
    oracle_by_index = {
        int(row["sample_index"]): row for row in result_payload["rows"]
    }
    selected: list[dict[str, Any]] = []
    for cell in CELLS:
        candidates = sorted(
            (
                row
                for row in selection_payload["selected"]
                if row["selection_cell"] == cell
            ),
            key=lambda row: int(row["sample_index"]),
        )
        if len(candidates) < cfg.states_per_cell:
            raise ValueError(f"086 selection lacks enough states for {cell}.")
        for source in candidates[: cfg.states_per_cell]:
            row = dict(source)
            oracle = oracle_by_index[int(source["sample_index"])]
            row["oracle_pooled_values"] = list(
                oracle["expanded_root_values"]["pooled"]
            )
            row["oracle_champion_selected_index"] = int(
                oracle["champion_selected_index"]
            )
            row["oracle_best_index"] = int(oracle["pooled_evaluation_best_index"])
            if not _row_alignment(row):
                raise ValueError("086 root/value alignment failed during selection.")
            selected.append(row)
    return sorted(selected, key=lambda row: int(row["sample_index"]))


def _selection_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": int(row["sample_index"]),
        "source_seed": int(row["source_seed"]),
        "turn_index": int(row["turn_index"]),
        "actor_seat": int(row["actor_seat"]),
        "phase": str(row["phase"]),
        "selection_cell": str(row["selection_cell"]),
        "expanded_root_count": int(row["expanded_root_count"]),
    }


def _row_alignment(row: dict[str, Any]) -> bool:
    count = int(row["expanded_root_count"])
    return (
        len(row["expanded_root_keys"]) == count
        and len(row["oracle_pooled_values"]) == count
        and 0 <= int(row["oracle_champion_selected_index"]) < min(4, count)
        and 0 <= int(row["oracle_best_index"]) < count
        and all(math.isfinite(float(value)) for value in row["oracle_pooled_values"])
    )


def _evaluate_parallel(
    cfg: ProgressivePlannerConfig,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    identity = _identity(cfg)
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 133 progress has a different identity.")
            completed[int(payload["sample_index"])] = payload["result"]
    pending = [row for row in selected if int(row["sample_index"]) not in completed]
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(cfg.workers, len(pending)),
            initializer=_initialize_worker,
            initargs=(str(cfg.checkpoint), cfg.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_state_worker, row, cfg): int(row["sample_index"])
                for row in pending
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
                    print(f"[planner133] states={len(completed)}/{len(selected)}", flush=True)
    return [completed[int(row["sample_index"])] for row in selected]


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(Path(checkpoint), device, "legal_action_attention_scorer")


def _evaluate_state_worker(
    sample: dict[str, Any],
    cfg: ProgressivePlannerConfig,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("133 worker scorer was not initialized.")
    return _evaluate_state(sample, cfg, _WORKER_SCORER)


def _evaluate_state(
    sample: dict[str, Any],
    cfg: ProgressivePlannerConfig,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    return {
        "sample_index": int(sample["sample_index"]),
        "source_seed": int(sample["source_seed"]),
        "turn_index": int(sample["turn_index"]),
        "actor_seat": int(sample["actor_seat"]),
        "phase": str(sample["phase"]),
        "selection_cell": str(sample["selection_cell"]),
        "expanded_root_count": int(sample["expanded_root_count"]),
        "oracle_pooled_values": list(sample["oracle_pooled_values"]),
        "oracle_champion_selected_index": int(sample["oracle_champion_selected_index"]),
        "oracle_best_index": int(sample["oracle_best_index"]),
        "searches": {
            side: _run_search(sample, cfg, side, scorer) for side in SIDES
        },
    }


def _run_search(
    sample: dict[str, Any],
    cfg: ProgressivePlannerConfig,
    side: str,
    scorer: _LegalActionScorer,
    *,
    simulation_limit: int | None = None,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(sample["state_snapshot"])
    env = SplendorEnv()
    ranked = audit085._distinct_canonical_scores(
        state, scorer.score_actions(state, env.legal_actions(state))
    )
    score_by_key = {
        canonical_successor_action_key(state, row.action): float(row.score)
        for row in ranked
    }
    root_keys = tuple(ast.literal_eval(value) for value in sample["expanded_root_keys"])
    if any(key not in score_by_key for key in root_keys):
        raise PlannerIneligible("expanded_root_missing_from_ranked_actions")
    root_priors = score_priors(
        [score_by_key[key] for key in root_keys],
        cfg.root_prior_temperature,
        cfg.root_prior_uniform_mix,
    )
    root_edges = [Edge(prior=value) for value in root_priors]
    particles = deterministic_belief_range(
        state,
        state.current_player,
        root_index=belief_root_index(cfg, int(sample["sample_index"]), side),
        count=cfg.determinizations,
    )
    opponent_nodes: dict[tuple[object, ...], OpponentNode] = {}
    try:
        branches = _prepare_branches(
            particles, root_keys, cfg, scorer, opponent_nodes
        )
    except PlannerIneligible as exc:
        return _ineligible_search(exc.reason, root_keys, root_priors)

    full_forced = len(root_keys) * cfg.determinizations * cfg.opponent_actions
    limit = cfg.simulations if simulation_limit is None else int(simulation_limit)
    if limit <= 0 or limit > cfg.simulations:
        raise ValueError("Invalid 133 simulation limit.")
    counters: Counter[str] = Counter()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()

    for simulation in range(limit):
        if simulation < full_forced:
            unit = cfg.determinizations * cfg.opponent_actions
            root_index = simulation // unit
            within = simulation % unit
            particle_index = within // cfg.opponent_actions
            response_index = within % cfg.opponent_actions
            counters["forced_simulations"] += 1
        else:
            root_index = select_puct_edge(
                root_edges, cfg.root_puct_constant, minimizing=False
            )
            particle_index = (simulation - full_forced) % cfg.determinizations
            branch = branches[(root_index, particle_index)]
            node = opponent_nodes[branch.information_key]
            response_index = select_puct_edge(
                node.edges, cfg.opponent_puct_constant, minimizing=True
            )
            counters["adaptive_simulations"] += 1

        branch = branches[(root_index, particle_index)]
        node = opponent_nodes[branch.information_key]
        response_key = branch.response_keys[response_index]
        response_map = actions_by_canonical_successor_key(
            branch.after_root, env.legal_actions(branch.after_root)
        )
        if response_key not in response_map:
            return _ineligible_search(
                "opponent_response_mapping_changed", root_keys, root_priors, counters
            )
        outcome = rollout_score_margin_loop_safe(
            env,
            scorer,
            branch.after_root,
            response_map[response_key],
            int(sample["actor_seat"]),
            continuation_seed(cfg, int(sample["sample_index"]), side, simulation),
            cfg.continuation_epsilon,
            cfg.continuation_temperature,
            cfg.continuation_top_k,
            cfg.continuation_max_turns,
            cfg.score_margin_scale,
            cfg.max_loop_fallback_triggers,
        )
        if not math.isfinite(float(outcome.value)):
            raise RuntimeError("133 produced a nonfinite continuation payoff.")
        root_edges[root_index].visits += 1
        root_edges[root_index].value_sum += float(outcome.value)
        node.edges[response_index].visits += 1
        node.edges[response_index].value_sum += float(outcome.value)
        counters["continuations"] += 1
        counters["resolved_continuations"] += int(outcome.resolved)
        counters["hard_caps"] += int(outcome.hard_cap_reached)
        counters["loop_adjudications"] += int(outcome.loop_adjudicated)
        counters["rollout_fallback_triggers"] += int(outcome.fallback_triggers)

    selected_index = max(
        range(len(root_edges)),
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
        "root_visits": [edge.visits for edge in root_edges],
        "root_probabilities": _visit_probabilities(root_edges),
        "root_q_values": [edge.q for edge in root_edges],
        "selected_index": selected_index,
        "prior_top_index": max(range(len(root_priors)), key=lambda index: (root_priors[index], -index)),
        "forced_coverage_required": full_forced,
        "counters": _counter_payload(counters),
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": time.process_time() - cpu_started,
    }


def _prepare_branches(
    particles: tuple[Any, ...],
    root_keys: tuple[tuple[object, ...], ...],
    cfg: ProgressivePlannerConfig,
    scorer: _LegalActionScorer,
    opponent_nodes: dict[tuple[object, ...], OpponentNode],
) -> dict[tuple[int, int], PreparedBranch]:
    env = SplendorEnv()
    branches: dict[tuple[int, int], PreparedBranch] = {}
    for root_index, root_key in enumerate(root_keys):
        for particle_index, particle in enumerate(particles):
            root_map = actions_by_canonical_successor_key(
                particle, env.legal_actions(particle)
            )
            if root_key not in root_map:
                raise PlannerIneligible("root_mapping_changed")
            after_root = env.step(particle, root_map[root_key])
            if after_root.terminal:
                raise PlannerIneligible("root_action_terminal")
            responses = audit085._distinct_canonical_scores(
                after_root,
                scorer.score_actions(after_root, env.legal_actions(after_root)),
            )
            if len(responses) < cfg.opponent_actions:
                raise PlannerIneligible("fewer_than_four_opponent_actions")
            responses = responses[: cfg.opponent_actions]
            response_keys = tuple(
                canonical_successor_action_key(after_root, row.action)
                for row in responses
            )
            response_priors = tuple(
                score_priors(
                    [float(row.score) for row in responses],
                    cfg.opponent_prior_temperature,
                    cfg.opponent_prior_uniform_mix,
                )
            )
            information_key = (
                root_key,
                canonical_public_belief_key(after_root, after_root.current_player),
            )
            node = opponent_nodes.get(information_key)
            if node is None:
                opponent_nodes[information_key] = OpponentNode(
                    root_index=root_index,
                    action_keys=response_keys,
                    priors=response_priors,
                    edges=[Edge(prior=value) for value in response_priors],
                )
            elif node.action_keys != response_keys or any(
                abs(left - right) > 1e-12
                for left, right in zip(node.priors, response_priors, strict=True)
            ):
                raise PlannerIneligible("opponent_action_set_or_prior_unstable")
            branches[(root_index, particle_index)] = PreparedBranch(
                after_root=after_root,
                information_key=information_key,
                response_keys=response_keys,
                response_priors=response_priors,
            )
    return branches


def _ineligible_search(
    reason: str,
    root_keys: tuple[tuple[object, ...], ...],
    root_priors: list[float],
    counters: Counter[str] | None = None,
) -> dict[str, Any]:
    return {
        "completed": False,
        "fallback_reason": reason,
        "root_keys": [repr(key) for key in root_keys],
        "root_priors": list(root_priors),
        "root_visits": [0] * len(root_keys),
        "root_probabilities": [0.0] * len(root_keys),
        "root_q_values": [0.0] * len(root_keys),
        "selected_index": None,
        "prior_top_index": max(range(len(root_priors)), key=lambda index: (root_priors[index], -index)),
        "forced_coverage_required": len(root_keys),
        "counters": _counter_payload(counters or Counter()),
        "wall_seconds": 0.0,
        "process_cpu_seconds": 0.0,
    }


def belief_root_index(
    cfg: ProgressivePlannerConfig, sample_index: int, side: str
) -> int:
    return cfg.belief_root_base + sample_index * 10 + SIDES.index(side)


def continuation_seed(
    cfg: ProgressivePlannerConfig,
    sample_index: int,
    side: str,
    simulation: int,
) -> int:
    return (
        cfg.continuation_seed_base
        + sample_index * 10_000_000
        + SIDES.index(side) * 2_000_000
        + simulation
    )


def summarize(
    rows: list[dict[str, Any]],
    cfg: ProgressivePlannerConfig,
    wall_seconds: float,
) -> dict[str, Any]:
    comparisons = [_comparison(row, cfg) for row in rows if _both_completed(row)]
    phase = {
        name: _comparison_summary(
            [item for item in comparisons if item["phase"] == name]
        )
        for name in PHASES
    }
    all_summary = _comparison_summary(comparisons)
    state_gains = [float(item["mean_gain"]) for item in comparisons]
    gain_ci = audit085._bootstrap_mean_ci(
        state_gains, cfg.bootstrap_samples, cfg.bootstrap_seed
    ) if state_gains else (math.nan, math.nan)
    counters = Counter()
    completed_by_phase = Counter()
    for row in rows:
        if _both_completed(row):
            completed_by_phase[row["phase"]] += 1
        for side in SIDES:
            counters.update(
                {
                    key: int(value)
                    for key, value in row["searches"][side]["counters"].items()
                }
            )
    all_summary.update(
        {
            "states": len(rows),
            "both_completed_states": len(comparisons),
            "both_completed_fraction": len(comparisons) / len(rows),
            "phase_completed_fraction": {
                name: completed_by_phase[name]
                / sum(row["phase"] == name for row in rows)
                for name in PHASES
            },
            "phase": phase,
            "gain_bootstrap_ci95": list(gain_ci),
            "counters": dict(sorted(counters.items())),
            "loop_adjudication_fraction": counters["loop_adjudications"]
            / counters["continuations"]
            if counters["continuations"]
            else 1.0,
            "wall_seconds": wall_seconds,
            "aggregate_process_cpu_seconds": sum(
                float(row["searches"][side]["process_cpu_seconds"])
                for row in rows
                for side in SIDES
            ),
        }
    )
    return all_summary


def _comparison(row: dict[str, Any], cfg: ProgressivePlannerConfig) -> dict[str, Any]:
    left = row["searches"]["left"]
    right = row["searches"]["right"]
    values = [float(value) for value in row["oracle_pooled_values"]]
    best = max(values)
    equivalent = [index for index, value in enumerate(values) if best - value <= cfg.equivalence_regret]
    left_p, right_p = collapse_equivalence_group(
        list(left["root_probabilities"]), list(right["root_probabilities"]), equivalent
    )
    left_index = int(left["selected_index"])
    right_index = int(right["selected_index"])
    champion_index = int(row["oracle_champion_selected_index"])
    left_gain = values[left_index] - values[champion_index]
    right_gain = values[right_index] - values[champion_index]
    left_regret = best - values[left_index]
    right_regret = best - values[right_index]
    cross_regret = abs(values[left_index] - values[right_index]) / 2.0
    raw_jsd = jensen_shannon(
        list(left["root_probabilities"]), list(right["root_probabilities"])
    )
    prior_search_jsds = [
        jensen_shannon(list(search["root_priors"]), list(search["root_probabilities"]))
        for search in (left, right)
    ]
    return {
        "sample_index": int(row["sample_index"]),
        "phase": str(row["phase"]),
        "exact_agreement": left_index == right_index,
        "raw_jsd": raw_jsd,
        "collapsed_jsd": jensen_shannon(left_p, right_p),
        "cross_regret": cross_regret,
        "left_regret": left_regret,
        "right_regret": right_regret,
        "left_equivalent": left_regret <= cfg.equivalence_regret,
        "right_equivalent": right_regret <= cfg.equivalence_regret,
        "mean_oracle_regret": statistics.fmean((left_regret, right_regret)),
        "left_gain": left_gain,
        "right_gain": right_gain,
        "mean_gain": statistics.fmean((left_gain, right_gain)),
        "expanded_selected": left_index >= cfg.champion_roots or right_index >= cfg.champion_roots,
        "left_expanded": left_index >= cfg.champion_roots,
        "right_expanded": right_index >= cfg.champion_roots,
        "prior_tv": statistics.fmean(
            total_variation(list(search["root_priors"]), list(search["root_probabilities"]))
            for search in (left, right)
        ),
        "prior_search_jsd": statistics.fmean(prior_search_jsds),
        "prior_control_margin": statistics.fmean(prior_search_jsds) - raw_jsd,
        "prior_top_changed": any(
            int(search["selected_index"]) != int(search["prior_top_index"])
            for search in (left, right)
        ),
        "champion_changed": left_index != champion_index or right_index != champion_index,
        "q_pearson": audit085._pearson(left["root_q_values"], right["root_q_values"]),
        "q_spearman": _spearman(left["root_q_values"], right["root_q_values"]),
    }


def _comparison_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "states": 0,
            "exact_agreement": math.nan,
            "mean_raw_jsd": math.nan,
            "mean_collapsed_jsd": math.nan,
            "mean_cross_regret": math.nan,
            "p90_cross_regret": math.nan,
            "left_equivalent_fraction": math.nan,
            "right_equivalent_fraction": math.nan,
            "mean_oracle_regret": math.nan,
            "mean_gain": math.nan,
            "expanded_selection_fraction": math.nan,
            "mean_prior_tv": math.nan,
            "prior_top_change_fraction": math.nan,
            "mean_prior_search_jsd": math.nan,
            "mean_prior_control_margin": math.nan,
            "mean_q_pearson": math.nan,
            "mean_q_spearman": math.nan,
        }
    cross = sorted(float(item["cross_regret"]) for item in items)
    p90 = cross[min(math.ceil(0.9 * len(cross)) - 1, len(cross) - 1)]
    return {
        "states": len(items),
        "exact_agreement": statistics.fmean(bool(item["exact_agreement"]) for item in items),
        "mean_raw_jsd": statistics.fmean(float(item["raw_jsd"]) for item in items),
        "mean_collapsed_jsd": statistics.fmean(float(item["collapsed_jsd"]) for item in items),
        "mean_cross_regret": statistics.fmean(float(item["cross_regret"]) for item in items),
        "p90_cross_regret": p90,
        "left_equivalent_fraction": statistics.fmean(bool(item["left_equivalent"]) for item in items),
        "right_equivalent_fraction": statistics.fmean(bool(item["right_equivalent"]) for item in items),
        "mean_oracle_regret": statistics.fmean(float(item["mean_oracle_regret"]) for item in items),
        "mean_gain": statistics.fmean(float(item["mean_gain"]) for item in items),
        "expanded_selection_fraction": statistics.fmean(bool(item["expanded_selected"]) for item in items),
        "left_expanded_fraction": statistics.fmean(bool(item["left_expanded"]) for item in items),
        "right_expanded_fraction": statistics.fmean(bool(item["right_expanded"]) for item in items),
        "mean_prior_tv": statistics.fmean(float(item["prior_tv"]) for item in items),
        "prior_top_change_fraction": statistics.fmean(bool(item["prior_top_changed"]) for item in items),
        "champion_change_fraction": statistics.fmean(bool(item["champion_changed"]) for item in items),
        "mean_prior_search_jsd": statistics.fmean(float(item["prior_search_jsd"]) for item in items),
        "mean_prior_control_margin": statistics.fmean(float(item["prior_control_margin"]) for item in items),
        "mean_q_pearson": statistics.fmean(float(item["q_pearson"]) for item in items),
        "mean_q_spearman": statistics.fmean(float(item["q_spearman"]) for item in items),
    }


def integrity_gates(
    cfg: ProgressivePlannerConfig,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    repeats: dict[str, bool],
    privacy: dict[str, Any],
) -> dict[str, bool]:
    completed = [row for row in rows if _both_completed(row)]
    return {
        "frozen_inputs_and_decisions": _provenance_matches(cfg),
        "exact_balanced_outcome_blind_selection": len(selected) == 36
        and all(sum(row["selection_cell"] == cell for row in selected) == 6 for cell in CELLS),
        "exact_root_value_alignment": all(_row_alignment(row) for row in selected),
        "completion_floor_overall_and_phase": float(summary["both_completed_fraction"]) >= cfg.minimum_completion_fraction
        and all(float(summary["phase_completed_fraction"][phase]) >= cfg.minimum_completion_fraction for phase in PHASES),
        "exact_simulation_accounting": all(
            int(row["searches"][side]["counters"]["continuations"]) == cfg.simulations
            for row in completed for side in SIDES
        ),
        "exact_forced_coverage": all(
            int(row["searches"][side]["counters"]["forced_simulations"])
            == int(row["expanded_root_count"]) * cfg.determinizations * cfg.opponent_actions
            for row in completed for side in SIDES
        ),
        "all_continuations_resolved": int(summary["counters"].get("resolved_continuations", 0))
        == int(summary["counters"].get("continuations", 0)),
        "zero_hard_caps": int(summary["counters"].get("hard_caps", 0)) == 0,
        "bounded_loop_adjudication": float(summary["loop_adjudication_fraction"]) <= cfg.maximum_loop_adjudicated_fraction,
        "stable_information_sets": all(
            row["searches"][side]["fallback_reason"] != "opponent_action_set_or_prior_unstable"
            for row in rows for side in SIDES
        ),
        "privacy_invariance": bool(privacy.get("passed")),
        "first_state_repeats_identical": all(repeats.values()),
        "all_metrics_finite": _all_numeric_finite(summary),
        "no_training_gameplay_human_or_partial_087_path": True,
    }


def feasibility_gates(
    cfg: ProgressivePlannerConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_agreement_at_least_60pct": float(summary["exact_agreement"]) >= cfg.minimum_exact_agreement,
        "raw_jsd_bounded": float(summary["mean_raw_jsd"]) <= cfg.maximum_raw_jsd
        and float(summary["phase"]["early"]["mean_raw_jsd"]) <= cfg.maximum_early_raw_jsd,
        "collapsed_jsd_bounded": float(summary["mean_collapsed_jsd"]) <= cfg.maximum_collapsed_jsd
        and float(summary["phase"]["early"]["mean_collapsed_jsd"]) <= cfg.maximum_early_collapsed_jsd,
        "cross_regret_bounded": float(summary["mean_cross_regret"]) <= cfg.maximum_mean_cross_regret
        and float(summary["p90_cross_regret"]) <= cfg.maximum_p90_cross_regret,
        "both_arms_oracle_equivalence": float(summary["left_equivalent_fraction"]) >= cfg.minimum_oracle_equivalence
        and float(summary["right_equivalent_fraction"]) >= cfg.minimum_oracle_equivalence,
        "mean_oracle_regret_bounded": float(summary["mean_oracle_regret"]) <= cfg.maximum_mean_oracle_regret,
        "q_spearman_at_least_0p50": float(summary["mean_q_spearman"]) >= cfg.minimum_q_spearman,
        "material_prior_movement": float(summary["mean_prior_tv"]) >= cfg.minimum_prior_total_variation
        and float(summary["prior_top_change_fraction"]) >= cfg.minimum_prior_top_change_fraction,
        "prior_control_margin_at_least_0p02": float(summary["mean_prior_control_margin"]) >= cfg.minimum_prior_control_margin,
        "mean_gain_at_least_0p025": float(summary["mean_gain"]) >= cfg.minimum_mean_gain,
        "gain_ci95_lower_above_zero": float(summary["gain_bootstrap_ci95"][0]) > 0.0,
        "positive_gain_every_phase": all(float(summary["phase"][phase]["mean_gain"]) > 0.0 for phase in PHASES),
        "expanded_selection_material_every_phase": float(summary["expanded_selection_fraction"]) >= cfg.minimum_expanded_selection_fraction
        and all(float(summary["phase"][phase]["expanded_selection_fraction"]) > 0.0 for phase in PHASES),
    }


def _privacy_audit(
    sample: dict[str, Any],
    cfg: ProgressivePlannerConfig,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(sample["state_snapshot"])
    reversed_state = state.copy_shallow()
    for tier in (1, 2, 3):
        reversed_state.hidden_tier_decks[tier] = list(
            reversed(reversed_state.hidden_tier_decks[tier])
        )
    left = _public_root_projection(state, sample, cfg, scorer)
    right = _public_root_projection(reversed_state, sample, cfg, scorer)
    return {"passed": left == right, "original": left, "reversed": right}


def _public_root_projection(
    state: Any,
    sample: dict[str, Any],
    cfg: ProgressivePlannerConfig,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    env = SplendorEnv()
    ranked = audit085._distinct_canonical_scores(
        state, scorer.score_actions(state, env.legal_actions(state))
    )
    scores = {
        canonical_successor_action_key(state, row.action): float(row.score)
        for row in ranked
    }
    root_keys = tuple(ast.literal_eval(value) for value in sample["expanded_root_keys"])
    return {
        "public_key": repr(canonical_public_belief_key(state, state.current_player)),
        "root_keys": [repr(key) for key in root_keys],
        "priors": score_priors(
            [scores[key] for key in root_keys],
            cfg.root_prior_temperature,
            cfg.root_prior_uniform_mix,
        ),
    }


def _both_completed(row: dict[str, Any]) -> bool:
    return all(bool(row["searches"][side]["completed"]) for side in SIDES)


def _visit_probabilities(edges: list[Edge]) -> list[float]:
    total = sum(edge.visits for edge in edges)
    return [edge.visits / total if total else 0.0 for edge in edges]


def _counter_payload(counters: Counter[str]) -> dict[str, int]:
    return {
        key: int(counters[key])
        for key in (
            "continuations",
            "resolved_continuations",
            "forced_simulations",
            "adaptive_simulations",
            "hard_caps",
            "loop_adjudications",
            "rollout_fallback_triggers",
        )
    }


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    return audit085._pearson(left_ranks, right_ranks)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
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


def _deterministic_projection(search: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in search.items()
        if key not in ("wall_seconds", "process_cpu_seconds")
    }


def _provenance_matches(cfg: ProgressivePlannerConfig) -> bool:
    if cfg != ProgressivePlannerConfig():
        return False
    if any(not path.exists() for path in (PROTOCOL, cfg.result_086, cfg.selection_086, cfg.result_132, cfg.roadmap, cfg.checkpoint)):
        return False
    if (
        _sha256_file(PROTOCOL) != PROTOCOL_SHA256
        or _sha256_file(cfg.result_086) != RESULT_086_SHA256
        or _sha256_file(cfg.selection_086) != SELECTION_086_SHA256
        or _sha256_file(cfg.checkpoint) != CHECKPOINT_SHA256
        or _sha256_file(cfg.result_132) != RESULT_132_SHA256
        or _sha256_file(cfg.roadmap) != ROADMAP_SHA256
    ):
        return False
    result_086 = json.loads(cfg.result_086.read_text(encoding="utf-8"))
    result_132 = json.loads(cfg.result_132.read_text(encoding="utf-8"))
    return (
        result_086.get("decision")
        == "expanded_root_086_replication_pass_authorize_gameplay_preregistration"
        and result_132.get("decision")
        == "repeated_return_value_target_not_causally_better_no_gameplay"
    )


def _identity(cfg: ProgressivePlannerConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "protocol_sha256": _sha256_file(PROTOCOL),
        "result_086_sha256": _sha256_file(cfg.result_086),
        "selection_086_sha256": _sha256_file(cfg.selection_086),
        "result_132_sha256": _sha256_file(cfg.result_132),
        "roadmap_sha256": _sha256_file(cfg.roadmap),
        "checkpoint_sha256": _sha256_file(cfg.checkpoint),
        "selection_contract": "first6_per_086_seat_phase_cell_v1",
        "search_contract": "progressive_full_coverage_then_public_belief_puct_v1",
    }


def _validate_config(cfg: ProgressivePlannerConfig) -> None:
    if cfg != ProgressivePlannerConfig():
        raise ValueError("Experiment 133 permits only its frozen default configuration.")
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 133 frozen provenance does not reproduce.")
    if cfg.simulations != 1536 or cfg.determinizations != 12 or cfg.opponent_actions != 4:
        raise ValueError("Experiment 133 search dose changed.")
    if cfg.continuation_seed_base + 36 * 10_000_000 >= 2**63:
        raise ValueError("Experiment 133 continuation namespace exceeds int64.")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()

