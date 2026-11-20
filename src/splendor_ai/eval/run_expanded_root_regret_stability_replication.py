"""Run preregistered experiment 086: expanded-root regret-stability replication."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from splendor_ai.cfr import ResolvedLeaf, resolve_two_ply
from splendor_ai.eval import run_expanded_root_resolver_headroom as audit085
from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    _sha256_file,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


PROTOCOL = Path("docs/experiments/expanded_root_regret_stability_replication_086.md")
PROTOCOL_SHA256 = "4521d8252c762d5678c81bf354e70e30b2f39afeedd3f831211394703b046fe2"
PHASES = audit085.PHASES
CELLS = audit085.CELLS


@dataclass(frozen=True, slots=True)
class ExpandedRootReplicationConfig:
    output_dir: Path = Path("outputs/expanded_root_regret_stability_replication_086")
    checkpoint: Path = CHECKPOINT
    source_seed_start: int = 860_000
    source_games: int = 600
    states_per_cell: int = 50
    champion_roots: int = 4
    determinizations: int = 3
    opponent_actions: int = 4
    continuations_per_leaf: int = 16
    discovery_continuations: int = 4
    evaluation_halves: int = 3
    continuations_per_evaluation_half: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_base: int = 860_000_000
    continuation_seed_base: int = 8_600_000_000_000_000_000
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 860_086
    minimum_mean_gain: float = 0.05
    minimum_change_fraction: float = 0.15
    phase_collapse_boundary: float = -0.05
    maximum_equivalent_regret: float = 0.025
    minimum_equivalent_fraction: float = 0.60
    maximum_mean_regret: float = 0.025
    maximum_p90_regret: float = 0.075
    minimum_leaf_pearson: float = 0.60
    maximum_leaf_mae: float = 0.10
    minimum_leaf_sign_agreement: float = 0.70
    minimum_root_pearson: float = 0.60
    maximum_loop_adjudicated_fraction: float = 0.001
    maximum_evaluation_wall_seconds: float = 43_200.0
    device: str = "cpu"


def run_experiment(
    config: ExpandedRootReplicationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ExpandedRootReplicationConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 086 result does not match frozen identity.")
        return result

    selection = audit085._collect_and_select(cfg, identity)
    selection_gates = audit085._selection_gates(cfg, selection)
    if not all(selection_gates.values()):
        result = _invalid_result(identity, selection, selection_gates, "source_or_selection_invalid")
        audit085._atomic_write(result_path, result)
        return result

    selected = list(selection["selected"])
    started = time.perf_counter()
    rows = _evaluate_parallel(cfg, identity, selected)
    evaluation_wall_seconds = time.perf_counter() - started
    if len(rows) != len(selected):
        result = _invalid_result(
            identity,
            selection,
            selection_gates,
            "incomplete_evaluation",
            completed_rows=len(rows),
        )
        audit085._atomic_write(result_path, result)
        return result

    scorer = _LegalActionScorer(cfg.checkpoint, cfg.device, "legal_action_attention_scorer")
    repeated = _evaluate_state(selected[0], cfg, scorer)
    repeat_identical = rows[0] == repeated
    privacy = audit085._privacy_audit(selected[0], cfg, scorer)
    summary = _summarize(rows, cfg, evaluation_wall_seconds)
    integrity = _integrity_gates(
        cfg,
        rows,
        summary,
        privacy,
        repeat_identical,
        evaluation_wall_seconds,
    )
    primary = _primary_gates(cfg, summary)
    argmax = _argmax_gates(cfg, summary)
    phase_guardrails = {
        phase: float(summary["phase_breakdown"][phase]["gain_ci95_high"])
        > cfg.phase_collapse_boundary
        for phase in PHASES
    }
    valid = all(selection_gates.values()) and all(integrity.values())
    headroom_passed = all(primary.values()) and all(phase_guardrails.values())
    argmax_passed = all(argmax.values())
    passed = valid and headroom_passed and argmax_passed
    if not valid:
        decision = "expanded_root_086_invalid_no_replication_decision"
    elif not headroom_passed and not argmax_passed:
        decision = "expanded_root_086_headroom_and_argmax_failed_close_exact_expansion"
    elif not argmax_passed:
        decision = "expanded_root_086_argmax_unreliable_close_exact_expansion"
    elif not headroom_passed:
        decision = "expanded_root_086_headroom_not_replicated_close_exact_expansion"
    else:
        decision = "expanded_root_086_replication_pass_authorize_gameplay_preregistration"

    result = {
        "experiment": "086_expanded_root_regret_stability_replication",
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "identity": identity,
        "selection": {key: value for key, value in selection.items() if key != "selected"},
        "selection_gates": selection_gates,
        "summary": summary,
        "integrity_gates": integrity,
        "primary_gates": primary,
        "argmax_gates": argmax,
        "phase_guardrails": phase_guardrails,
        "privacy": privacy,
        "first_state_repeat_identical": repeat_identical,
        "valid": valid,
        "passed": passed,
        "decision": decision,
        "gameplay_preregistration_authorized": passed,
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "promotion_authorized": False,
        "fast_champion": "035",
        "research_champion": "public_belief_resolver_050_051",
        "rows": rows,
    }
    audit085._atomic_write(result_path, result)
    print(f"086 decision={decision}", flush=True)
    return result


def _invalid_result(
    identity: dict[str, Any],
    selection: dict[str, Any],
    selection_gates: dict[str, bool],
    reason: str,
    *,
    completed_rows: int = 0,
) -> dict[str, Any]:
    return {
        "experiment": "086_expanded_root_regret_stability_replication",
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "identity": identity,
        "selection": {key: value for key, value in selection.items() if key != "selected"},
        "selection_gates": selection_gates,
        "summary": None,
        "integrity_gates": None,
        "primary_gates": None,
        "argmax_gates": None,
        "phase_guardrails": None,
        "valid": False,
        "passed": False,
        "decision": "expanded_root_086_invalid_no_replication_decision",
        "invalid_reason": reason,
        "completed_rows": completed_rows,
        "gameplay_preregistration_authorized": False,
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "promotion_authorized": False,
    }


def continuation_seed(
    cfg: ExpandedRootReplicationConfig,
    sample_index: int,
    determinization: int,
    continuation: int,
) -> int:
    return (
        cfg.continuation_seed_base
        + sample_index * 1_000_000_000
        + determinization * 10_000
        + continuation
    )


def _evaluate_parallel(
    cfg: ExpandedRootReplicationConfig,
    identity: dict[str, Any],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 086 evaluation progress identity mismatch.")
            completed[int(payload["sample_index"])] = payload["result"]
    pending = [row for row in selected if int(row["sample_index"]) not in completed]
    print(
        f"[resolver086:evaluate] states={len(selected)} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(
            max_workers=cfg.workers,
            initializer=audit085._initialize_worker,
            initargs=(cfg.checkpoint, cfg.device),
        ) as executor:
            futures = {executor.submit(_evaluate_state_worker, row, cfg): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                sample = futures[future]
                result = future.result()
                sample_index = int(sample["sample_index"])
                completed[sample_index] = result
                audit085._append_jsonl(
                    progress_path,
                    {"identity": identity, "sample_index": sample_index, "result": result},
                )
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver086:evaluate] complete={len(completed)}/{len(selected)} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )
    return [completed[index] for index in sorted(completed)]


def _evaluate_state_worker(
    sample: dict[str, Any],
    cfg: ExpandedRootReplicationConfig,
) -> dict[str, Any]:
    scorer = audit085._WORKER_SCORER
    if scorer is None:
        raise RuntimeError("086 evaluation worker scorer was not initialized.")
    return _evaluate_state(sample, cfg, scorer)


def _evaluate_state(
    sample: dict[str, Any],
    cfg: ExpandedRootReplicationConfig,
    scorer: Any,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(sample["state_snapshot"])
    structure = audit085._prepare_structure(
        state,
        cfg,
        scorer,
        int(sample["source_seed"]),
        int(sample["turn_index"]),
    )
    if not structure["champion_eligible"]:
        raise RuntimeError("086 selected champion root structure no longer reproduces.")
    if structure["champion_root_keys"] != sample["champion_root_keys"]:
        raise RuntimeError("086 champion roots changed after selection.")
    if structure["expanded_root_keys"] != sample["expanded_root_keys"]:
        raise RuntimeError("086 expanded roots changed after selection.")

    table_names = ("discovery", "eval0", "eval1", "eval2", "pooled")
    leaves: dict[str, list[ResolvedLeaf]] = {name: [] for name in table_names}
    leaf_values: dict[str, list[float]] = {name: [] for name in table_names}
    counters: Counter[str] = Counter()
    env = audit085.SplendorEnv()
    for branch in structure["branches"]:
        for response in branch["responses"]:
            values = []
            for continuation in range(cfg.continuations_per_leaf):
                outcome = rollout_score_margin_loop_safe(
                    env,
                    scorer,
                    branch["after_root"],
                    response.action,
                    int(sample["actor_seat"]),
                    continuation_seed(
                        cfg,
                        int(sample["sample_index"]),
                        int(branch["determinization"]),
                        continuation,
                    ),
                    cfg.continuation_epsilon,
                    cfg.continuation_temperature,
                    cfg.continuation_top_k,
                    cfg.continuation_max_turns,
                    cfg.score_margin_scale,
                    cfg.max_loop_fallback_triggers,
                )
                if not math.isfinite(outcome.value):
                    raise RuntimeError("086 produced a nonfinite continuation value.")
                values.append(float(outcome.value))
                counters["continuations"] += 1
                counters["resolved_continuations"] += int(outcome.resolved)
                counters["hard_caps"] += int(outcome.hard_cap_reached)
                counters["loop_adjudications"] += int(outcome.loop_adjudicated)
                counters["rollout_fallback_triggers"] += int(outcome.fallback_triggers)
            table_means = {
                "discovery": statistics.fmean(values[0:4]),
                "eval0": statistics.fmean(values[4:8]),
                "eval1": statistics.fmean(values[8:12]),
                "eval2": statistics.fmean(values[12:16]),
                "pooled": statistics.fmean(values[4:16]),
            }
            response_key = audit085.canonical_successor_action_key(
                branch["after_root"], response.action
            )
            for name, value in table_means.items():
                leaf_values[name].append(value)
                leaves[name].append(
                    ResolvedLeaf(
                        determinization=int(branch["determinization"]),
                        root_action=branch["root_key"],
                        opponent_information_set=branch["information_set"],
                        opponent_action=response_key,
                        discovery_payoff=value,
                        evaluation_payoff=value,
                    )
                )

    root_keys = structure["root_keys"]
    champion_keys = root_keys[: cfg.champion_roots]
    resolutions: dict[str, dict[str, Any]] = {}
    for name in table_names:
        expanded = resolve_two_ply(
            leaves[name], root_keys, cfg.determinizations, payoff="discovery_payoff"
        )
        champion = resolve_two_ply(
            [leaf for leaf in leaves[name] if leaf.root_action in champion_keys],
            champion_keys,
            cfg.determinizations,
            payoff="discovery_payoff",
        )
        resolutions[name] = {"expanded": expanded, "champion": champion}

    expanded_selected = int(resolutions["discovery"]["expanded"].best_index)
    champion_selected = int(resolutions["discovery"]["champion"].best_index)
    pooled_expanded = resolutions["pooled"]["expanded"]
    pooled_values = list(pooled_expanded.action_values)
    regret = float(max(pooled_values) - pooled_values[expanded_selected])
    half_gains = [
        float(resolutions[f"eval{index}"]["expanded"].action_values[expanded_selected])
        - float(resolutions[f"eval{index}"]["expanded"].action_values[champion_selected])
        for index in range(cfg.evaluation_halves)
    ]
    pooled_gain = (
        float(pooled_values[expanded_selected]) - float(pooled_values[champion_selected])
    )
    expected_continuations = (
        len(root_keys)
        * cfg.determinizations
        * cfg.opponent_actions
        * cfg.continuations_per_leaf
    )
    return {
        "sample_index": int(sample["sample_index"]),
        "source_seed": int(sample["source_seed"]),
        "turn_index": int(sample["turn_index"]),
        "actor_seat": int(sample["actor_seat"]),
        "phase": str(sample["phase"]),
        "selection_cell": str(sample["selection_cell"]),
        "champion_root_count": cfg.champion_roots,
        "expanded_root_count": len(root_keys),
        "added_root_count": len(root_keys) - cfg.champion_roots,
        "excluded_root_reasons": structure["excluded_root_reasons"],
        "champion_root_keys": structure["champion_root_keys"],
        "expanded_root_keys": structure["expanded_root_keys"],
        "expanded_selected_index": expanded_selected,
        "champion_selected_index": champion_selected,
        "pooled_evaluation_best_index": int(pooled_expanded.best_index),
        "exact_best_agreement": expanded_selected == int(pooled_expanded.best_index),
        "changed": expanded_selected >= cfg.champion_roots,
        "pooled_evaluation_gain": pooled_gain,
        "evaluation_half_gains": half_gains,
        "pooled_evaluation_regret": max(regret, 0.0),
        "leaf_values": leaf_values,
        "expanded_root_values": {
            name: list(resolutions[name]["expanded"].action_values) for name in table_names
        },
        "counters": {
            "continuations": counters["continuations"],
            "expected_continuations": expected_continuations,
            "resolved_continuations": counters["resolved_continuations"],
            "hard_caps": counters["hard_caps"],
            "loop_adjudications": counters["loop_adjudications"],
            "rollout_fallback_triggers": counters["rollout_fallback_triggers"],
        },
    }


def _summarize(
    rows: list[dict[str, Any]],
    cfg: ExpandedRootReplicationConfig,
    evaluation_wall_seconds: float,
) -> dict[str, Any]:
    gains = [float(row["pooled_evaluation_gain"]) for row in rows]
    gain_ci = audit085._bootstrap_mean_ci(gains, cfg.bootstrap_samples, cfg.bootstrap_seed)
    regrets = [float(row["pooled_evaluation_regret"]) for row in rows]
    sorted_regrets = sorted(regrets)
    p90_regret = sorted_regrets[min(math.ceil(0.90 * len(sorted_regrets)) - 1, len(sorted_regrets) - 1)]
    exact = sum(bool(row["exact_best_agreement"]) for row in rows)
    exact_ci = _wilson_interval(exact, len(rows))
    reliability = {}
    discovery_leaf = [value for row in rows for value in row["leaf_values"]["discovery"]]
    discovery_roots = [
        value for row in rows for value in row["expanded_root_values"]["discovery"]
    ]
    for index in range(cfg.evaluation_halves):
        name = f"eval{index}"
        eval_leaf = [value for row in rows for value in row["leaf_values"][name]]
        eval_roots = [value for row in rows for value in row["expanded_root_values"][name]]
        eligible = [
            position
            for position, (left, right) in enumerate(zip(discovery_leaf, eval_leaf, strict=True))
            if abs(left) >= 0.025 and abs(right) >= 0.025
        ]
        reliability[name] = {
            "leaf_pearson": audit085._pearson(discovery_leaf, eval_leaf),
            "leaf_mae": statistics.fmean(
                abs(left - right)
                for left, right in zip(discovery_leaf, eval_leaf, strict=True)
            ),
            "leaf_sign_eligible": len(eligible),
            "leaf_sign_agreement": (
                sum(discovery_leaf[pos] * eval_leaf[pos] > 0.0 for pos in eligible)
                / len(eligible)
                if eligible
                else 0.0
            ),
            "expanded_root_pearson": audit085._pearson(discovery_roots, eval_roots),
            "mean_selected_gain": statistics.fmean(
                float(row["evaluation_half_gains"][index]) for row in rows
            ),
        }
    pooled_roots = [
        value for row in rows for value in row["expanded_root_values"]["pooled"]
    ]
    phase_breakdown = {}
    for phase_index, phase in enumerate(PHASES):
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_gains = [float(row["pooled_evaluation_gain"]) for row in phase_rows]
        phase_ci = audit085._bootstrap_mean_ci(
            phase_gains,
            cfg.bootstrap_samples,
            cfg.bootstrap_seed + phase_index + 1,
        )
        phase_breakdown[phase] = {
            "states": len(phase_rows),
            "changed": sum(bool(row["changed"]) for row in phase_rows),
            "change_fraction": sum(bool(row["changed"]) for row in phase_rows)
            / len(phase_rows),
            "mean_gain": statistics.fmean(phase_gains),
            "gain_ci95_low": phase_ci[0],
            "gain_ci95_high": phase_ci[1],
        }
    counters = Counter()
    excluded = Counter()
    for row in rows:
        counters.update({key: int(value) for key, value in row["counters"].items()})
        excluded.update(row["excluded_root_reasons"])
    continuations = counters["continuations"]
    return {
        "states": len(rows),
        "mean_pooled_evaluation_gain": statistics.fmean(gains),
        "pooled_evaluation_gain_ci95_low": gain_ci[0],
        "pooled_evaluation_gain_ci95_high": gain_ci[1],
        "evaluation_half_mean_gains": [
            statistics.fmean(float(row["evaluation_half_gains"][index]) for row in rows)
            for index in range(cfg.evaluation_halves)
        ],
        "changed_states": sum(bool(row["changed"]) for row in rows),
        "change_fraction": sum(bool(row["changed"]) for row in rows) / len(rows),
        "mean_expanded_root_count": statistics.fmean(int(row["expanded_root_count"]) for row in rows),
        "minimum_expanded_root_count": min(int(row["expanded_root_count"]) for row in rows),
        "maximum_expanded_root_count": max(int(row["expanded_root_count"]) for row in rows),
        "excluded_root_reasons": dict(sorted(excluded.items())),
        "exact_best_agreement_count": exact,
        "exact_best_agreement_fraction": exact / len(rows),
        "exact_best_agreement_wilson95_low": exact_ci[0],
        "exact_best_agreement_wilson95_high": exact_ci[1],
        "equivalent_regret_states": sum(regret <= cfg.maximum_equivalent_regret for regret in regrets),
        "equivalent_regret_fraction": sum(regret <= cfg.maximum_equivalent_regret for regret in regrets) / len(regrets),
        "mean_pooled_evaluation_regret": statistics.fmean(regrets),
        "p90_pooled_evaluation_regret": p90_regret,
        "pooled_expanded_root_value_pearson": audit085._pearson(discovery_roots, pooled_roots),
        "evaluation_half_reliability": reliability,
        "phase_breakdown": phase_breakdown,
        "continuations": continuations,
        "expected_continuations": counters["expected_continuations"],
        "resolved_continuations": counters["resolved_continuations"],
        "hard_caps": counters["hard_caps"],
        "loop_adjudications": counters["loop_adjudications"],
        "loop_adjudication_fraction": counters["loop_adjudications"] / continuations if continuations else 1.0,
        "rollout_fallback_triggers": counters["rollout_fallback_triggers"],
        "evaluation_wall_seconds": evaluation_wall_seconds,
    }


def _integrity_gates(
    cfg: ExpandedRootReplicationConfig,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    privacy: dict[str, Any],
    repeat_identical: bool,
    evaluation_wall_seconds: float,
) -> dict[str, bool]:
    half_reliability = summary["evaluation_half_reliability"]
    return {
        "frozen_protocol_checkpoint_and_configuration": _provenance_matches(cfg),
        "exact_selected_state_coverage": len(rows) == cfg.states_per_cell * len(CELLS)
        and {int(row["sample_index"]) for row in rows}
        == set(range(cfg.states_per_cell * len(CELLS))),
        "champion_and_expanded_root_contracts_reproduce": all(
            int(row["champion_root_count"]) == cfg.champion_roots
            and int(row["expanded_root_count"]) >= cfg.champion_roots for row in rows
        ),
        "exact_variable_branch_accounting": int(summary["continuations"])
        == int(summary["expected_continuations"])
        and all(
            int(row["counters"]["continuations"])
            == int(row["counters"]["expected_continuations"]) for row in rows
        ),
        "all_continuations_resolved": int(summary["resolved_continuations"])
        == int(summary["continuations"]),
        "zero_hard_caps": int(summary["hard_caps"]) == 0,
        "zero_rollout_fallback_triggers": int(summary["rollout_fallback_triggers"]) == 0,
        "bounded_loop_adjudication": float(summary["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction,
        "all_three_evaluation_halves_reliable": all(
            float(half_reliability[f"eval{index}"]["leaf_pearson"])
            >= cfg.minimum_leaf_pearson
            and float(half_reliability[f"eval{index}"]["leaf_mae"])
            <= cfg.maximum_leaf_mae
            and float(half_reliability[f"eval{index}"]["leaf_sign_agreement"])
            >= cfg.minimum_leaf_sign_agreement
            for index in range(cfg.evaluation_halves)
        ),
        "privacy_invariance": bool(privacy.get("passed")),
        "first_state_repeat_identical": repeat_identical,
        "within_twelve_hour_operational_ceiling": evaluation_wall_seconds
        <= cfg.maximum_evaluation_wall_seconds,
        "no_human_input": "human" not in json.dumps(_config_payload(cfg)).lower(),
    }


def _primary_gates(
    cfg: ExpandedRootReplicationConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    return {
        "mean_gain_at_least_0p05": float(summary["mean_pooled_evaluation_gain"])
        >= cfg.minimum_mean_gain,
        "gain_ci95_lower_above_zero": float(summary["pooled_evaluation_gain_ci95_low"]) > 0.0,
        "change_fraction_at_least_15pct": float(summary["change_fraction"])
        >= cfg.minimum_change_fraction,
        "at_least_one_change_every_phase": all(
            int(summary["phase_breakdown"][phase]["changed"]) > 0 for phase in PHASES
        ),
        "all_three_evaluation_halves_positive": all(
            float(value) > 0.0 for value in summary["evaluation_half_mean_gains"]
        ),
    }


def _argmax_gates(
    cfg: ExpandedRootReplicationConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    half_reliability = summary["evaluation_half_reliability"]
    return {
        "equivalent_regret_fraction_at_least_60pct": float(summary["equivalent_regret_fraction"])
        >= cfg.minimum_equivalent_fraction,
        "mean_regret_at_most_0p025": float(summary["mean_pooled_evaluation_regret"])
        <= cfg.maximum_mean_regret,
        "p90_regret_at_most_0p075": float(summary["p90_pooled_evaluation_regret"])
        <= cfg.maximum_p90_regret,
        "pooled_expanded_root_pearson_at_least_0p60": float(summary["pooled_expanded_root_value_pearson"])
        >= cfg.minimum_root_pearson,
        "each_half_expanded_root_pearson_at_least_0p60": all(
            float(half_reliability[f"eval{index}"]["expanded_root_pearson"])
            >= cfg.minimum_root_pearson for index in range(cfg.evaluation_halves)
        ),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total.")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - spread, center + spread


def _provenance_matches(cfg: ExpandedRootReplicationConfig) -> bool:
    return (
        cfg == ExpandedRootReplicationConfig()
        and _sha256_file(PROTOCOL) == PROTOCOL_SHA256
        and _sha256_file(cfg.checkpoint) == CHECKPOINT_SHA256
    )


def _identity(cfg: ExpandedRootReplicationConfig) -> dict[str, Any]:
    return {
        "configuration": _config_payload(cfg),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "checkpoint_sha256": _sha256_file(cfg.checkpoint),
        "root_contract": "top4_plus_all_stable_non_deck_public_roots_v1",
        "continuation_contract": "4_discovery_plus_3x4_evaluation_v1",
        "experiment_085_pooled": False,
    }


def _validate_config(cfg: ExpandedRootReplicationConfig) -> None:
    if cfg != ExpandedRootReplicationConfig():
        raise ValueError("Experiment 086 permits only its preregistered default configuration.")
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(cfg.checkpoint)
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 086 protocol, checkpoint, or configuration mismatch.")
    expected = cfg.discovery_continuations + cfg.evaluation_halves * cfg.continuations_per_evaluation_half
    if expected != cfg.continuations_per_leaf:
        raise ValueError("086 continuation halves do not sum to the frozen total.")
    if cfg.continuation_seed_base + cfg.states_per_cell * len(CELLS) * 1_000_000_000 >= 2**63:
        raise ValueError("086 continuation namespace exceeds signed 64-bit range.")


def _config_payload(cfg: ExpandedRootReplicationConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
