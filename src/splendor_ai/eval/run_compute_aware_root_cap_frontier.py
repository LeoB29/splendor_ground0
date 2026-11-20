"""Run preregistered experiment 088: compute-aware root-cap frontier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval import run_expanded_root_resolver_headroom as audit085


PROTOCOL = Path("docs/experiments/compute_aware_root_cap_frontier_088.md")
PROTOCOL_SHA256 = "f3e473564fff91acaa298efba8dac9b2729e8517190efcc097fb32033ad042a4"
RESULT_085_SHA256 = "74d2e1e968d95634e2c73061e0bc479489026c805120de6d2ca2847c3b1f9654"
RESULT_086_SHA256 = "8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5"
PHASES = ("early", "mid", "late")


@dataclass(frozen=True, slots=True)
class ComputeAwareRootCapConfig:
    output_dir: Path = Path("outputs/compute_aware_root_cap_frontier_088")
    result_085: Path = Path("outputs/expanded_root_resolver_headroom_085/result.json")
    result_086: Path = Path("outputs/expanded_root_regret_stability_replication_086/result.json")
    caps: tuple[int, ...] = (6, 8, 12, 16, 24, 32)
    bootstrap_samples: int = 20_000
    discovery_bootstrap_base: int = 880_085
    confirmation_bootstrap_base: int = 880_086
    minimum_discovery_gain: float = 0.045
    minimum_discovery_preservation: float = 0.80
    minimum_confirmation_gain: float = 0.040
    minimum_change_fraction: float = 0.15
    maximum_equivalent_regret: float = 0.025
    minimum_equivalent_fraction: float = 0.60
    maximum_mean_regret: float = 0.025
    maximum_p90_regret: float = 0.075
    maximum_root_ratio: float = 0.70
    phase_collapse_boundary: float = -0.05


def run_audit(config: ComputeAwareRootCapConfig | None = None) -> dict[str, Any]:
    cfg = config or ComputeAwareRootCapConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    identity = _identity(cfg)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 088 result does not match frozen identity.")
        return result

    payload085 = json.loads(cfg.result_085.read_text(encoding="utf-8"))
    payload086 = json.loads(cfg.result_086.read_text(encoding="utf-8"))
    rows085, integrity085 = _validate_input(payload085, "085")
    rows086, integrity086 = _validate_input(payload086, "086")
    integrity_gates = {
        "frozen_protocol_and_input_hashes": _provenance_matches(cfg),
        "085_rows_and_uncapped_summary_reproduce": all(integrity085.values()),
        "086_rows_and_uncapped_summary_reproduce": all(integrity086.values()),
        "cap_frontier_is_strictly_ordered_and_above_champion": cfg.caps
        == tuple(sorted(set(cfg.caps)))
        and min(cfg.caps) > 4,
    }
    if not all(integrity_gates.values()):
        result = {
            "experiment": "088_compute_aware_root_cap_frontier",
            "identity": identity,
            "valid": False,
            "passed": False,
            "integrity_gates": integrity_gates,
            "input_integrity": {"085": integrity085, "086": integrity086},
            "decision": "compute_aware_cap_088_invalid_no_decision",
            "research_champion": "public_belief_resolver_050_051",
            "fast_champion": "035",
            "gameplay_authorized": False,
            "training_authorized": False,
            "corpus_generation_authorized": False,
        }
        _atomic_write(result_path, result)
        return result

    uncapped085 = float(payload085["summary"]["mean_evaluation_gain"])
    discovery_caps: dict[str, Any] = {}
    selected_cap: int | None = None
    for cap in cfg.caps:
        cap_rows = _replay_cap(rows085, cap, "085")
        summary = _summarize_cap(
            cap_rows,
            cap,
            cfg.bootstrap_samples,
            cfg.discovery_bootstrap_base + cap,
            evaluation_halves=0,
        )
        repeated = _summarize_cap(
            cap_rows,
            cap,
            cfg.bootstrap_samples,
            cfg.discovery_bootstrap_base + cap,
            evaluation_halves=0,
        )
        repeat_identical = summary == repeated
        gates = _discovery_gates(cfg, summary, uncapped085)
        gates["bootstrap_repeat_identical"] = repeat_identical
        eligible = all(gates.values())
        discovery_caps[str(cap)] = {
            "summary": summary,
            "gates": gates,
            "eligible": eligible,
        }
        if selected_cap is None and eligible:
            selected_cap = cap

    if selected_cap is None:
        result = {
            "experiment": "088_compute_aware_root_cap_frontier",
            "identity": identity,
            "valid": True,
            "passed": False,
            "integrity_gates": integrity_gates,
            "input_integrity": {"085": integrity085, "086": integrity086},
            "discovery_caps": discovery_caps,
            "selected_cap": None,
            "confirmation": None,
            "confirmation_opened": False,
            "decision": "no_compute_aware_cap_selected",
            "research_champion": "public_belief_resolver_050_051",
            "fast_champion": "035",
            "gameplay_authorized": False,
            "training_authorized": False,
            "corpus_generation_authorized": False,
        }
        _atomic_write(result_path, result)
        return result

    confirmation_rows = _replay_cap(rows086, selected_cap, "086")
    confirmation_summary = _summarize_cap(
        confirmation_rows,
        selected_cap,
        cfg.bootstrap_samples,
        cfg.confirmation_bootstrap_base + selected_cap,
        evaluation_halves=3,
    )
    confirmation_repeated = _summarize_cap(
        confirmation_rows,
        selected_cap,
        cfg.bootstrap_samples,
        cfg.confirmation_bootstrap_base + selected_cap,
        evaluation_halves=3,
    )
    confirmation_gates = _confirmation_gates(cfg, confirmation_summary)
    confirmation_gates["bootstrap_repeat_identical"] = (
        confirmation_summary == confirmation_repeated
    )
    passed = all(confirmation_gates.values())
    decision = (
        "compute_aware_root_cap_confirmed_for_future_fresh_evaluation"
        if passed
        else "selected_root_cap_not_confirmed_close_static_cap_frontier"
    )
    result = {
        "experiment": "088_compute_aware_root_cap_frontier",
        "identity": identity,
        "valid": True,
        "passed": passed,
        "integrity_gates": integrity_gates,
        "input_integrity": {"085": integrity085, "086": integrity086},
        "discovery_caps": discovery_caps,
        "selected_cap": selected_cap,
        "confirmation_opened": True,
        "confirmation": {
            "summary": confirmation_summary,
            "gates": confirmation_gates,
            "rows": confirmation_rows,
        },
        "decision": decision,
        "research_champion": "public_belief_resolver_050_051",
        "fast_champion": "035",
        "gameplay_authorized": False,
        "future_fresh_evaluation_design_authorized": passed,
        "training_authorized": False,
        "corpus_generation_authorized": False,
    }
    _atomic_write(result_path, result)
    print(f"088 result: {'PASS' if passed else 'FAIL'} cap={selected_cap}", flush=True)
    return result


def _validate_input(
    payload: dict[str, Any],
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows = list(payload.get("rows", []))
    basic = (
        len(rows) == 300
        and {int(row.get("sample_index", -1)) for row in rows} == set(range(300))
        and all(str(row.get("phase")) in PHASES for row in rows)
        and all(int(row.get("champion_root_count", -1)) == 4 for row in rows)
    )
    tables_valid = basic
    selections_reproduce = basic
    gains_reproduce = basic
    finite = basic
    for row in rows if basic else []:
        discovery, evaluation, halves = _tables(row, dataset)
        count = int(row["expanded_root_count"])
        tables_valid = tables_valid and count == len(discovery) == len(evaluation) and count >= 4
        tables_valid = tables_valid and all(len(values) == count for values in halves)
        finite = finite and all(
            math.isfinite(float(value))
            for values in (discovery, evaluation, *halves)
            for value in values
        )
        selected = _best_index(discovery)
        stored_selected = int(
            row["expanded_discovery_best_index"]
            if dataset == "085"
            else row["expanded_selected_index"]
        )
        champion = int(
            row["champion_discovery_best_index"]
            if dataset == "085"
            else row["champion_selected_index"]
        )
        selections_reproduce = selections_reproduce and selected == stored_selected
        gain = float(evaluation[selected]) - float(evaluation[champion])
        stored_gain = float(
            row["evaluation_gain"] if dataset == "085" else row["pooled_evaluation_gain"]
        )
        gains_reproduce = gains_reproduce and math.isclose(gain, stored_gain, abs_tol=1e-12)
        if dataset == "086":
            gains_reproduce = gains_reproduce and all(
                math.isclose(
                    float(values[selected]) - float(values[champion]),
                    float(stored),
                    abs_tol=1e-12,
                )
                for values, stored in zip(halves, row["evaluation_half_gains"], strict=True)
            )
    summary_key = "mean_evaluation_gain" if dataset == "085" else "mean_pooled_evaluation_gain"
    row_gain_key = "evaluation_gain" if dataset == "085" else "pooled_evaluation_gain"
    summary_reproduces = bool(rows) and math.isclose(
        statistics.fmean(float(row[row_gain_key]) for row in rows),
        float(payload.get("summary", {}).get(summary_key, math.nan)),
        abs_tol=1e-12,
    )
    return rows, {
        "exact_300_unique_ordered_rows": basic,
        "root_table_lengths_and_champion_prefix_valid": tables_valid,
        "all_root_values_finite": finite,
        "uncapped_discovery_selections_reproduce": selections_reproduce,
        "uncapped_gains_and_halves_reproduce": gains_reproduce,
        "uncapped_summary_mean_reproduces": summary_reproduces,
    }


def _tables(
    row: dict[str, Any],
    dataset: str,
) -> tuple[list[float], list[float], list[list[float]]]:
    if dataset == "085":
        return (
            list(row["expanded_discovery_values"]),
            list(row["expanded_evaluation_values"]),
            [],
        )
    values = row["expanded_root_values"]
    return (
        list(values["discovery"]),
        list(values["pooled"]),
        [list(values[f"eval{index}"]) for index in range(3)],
    )


def _best_index(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("Cannot select from an empty root-value table.")
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


def _replay_cap(
    rows: list[dict[str, Any]],
    cap: int,
    dataset: str,
) -> list[dict[str, Any]]:
    replayed = []
    for row in rows:
        discovery, evaluation, halves = _tables(row, dataset)
        retained = min(cap, len(discovery))
        selected = _best_index(discovery[:retained])
        champion = int(
            row["champion_discovery_best_index"]
            if dataset == "085"
            else row["champion_selected_index"]
        )
        full_selected = _best_index(discovery)
        gain = float(evaluation[selected]) - float(evaluation[champion])
        regret = max(float(value) for value in evaluation) - float(evaluation[selected])
        replayed.append(
            {
                "sample_index": int(row["sample_index"]),
                "phase": str(row["phase"]),
                "uncapped_roots": len(discovery),
                "retained_roots": retained,
                "selected_index": selected,
                "champion_selected_index": champion,
                "full_discovery_selected_index": full_selected,
                "changed": selected != champion,
                "full_discovery_action_agreement": selected == full_selected,
                "evaluation_gain": gain,
                "full_frontier_evaluation_regret": max(regret, 0.0),
                "evaluation_half_gains": [
                    float(values[selected]) - float(values[champion]) for values in halves
                ],
            }
        )
    return replayed


def _summarize_cap(
    rows: list[dict[str, Any]],
    cap: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    *,
    evaluation_halves: int,
) -> dict[str, Any]:
    gains = [float(row["evaluation_gain"]) for row in rows]
    gain_ci = audit085._bootstrap_mean_ci(gains, bootstrap_samples, bootstrap_seed)
    regrets = sorted(float(row["full_frontier_evaluation_regret"]) for row in rows)
    retained = sorted(int(row["retained_roots"]) for row in rows)
    uncapped = [int(row["uncapped_roots"]) for row in rows]
    phase_breakdown = {}
    for phase_index, phase in enumerate(PHASES):
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_gains = [float(row["evaluation_gain"]) for row in phase_rows]
        phase_ci = audit085._bootstrap_mean_ci(
            phase_gains,
            bootstrap_samples,
            bootstrap_seed + 1000 + phase_index,
        )
        phase_breakdown[phase] = {
            "states": len(phase_rows),
            "changed": sum(bool(row["changed"]) for row in phase_rows),
            "mean_gain": statistics.fmean(phase_gains),
            "gain_ci95_low": phase_ci[0],
            "gain_ci95_high": phase_ci[1],
        }
    return {
        "cap": cap,
        "states": len(rows),
        "mean_evaluation_gain": statistics.fmean(gains),
        "evaluation_gain_ci95_low": gain_ci[0],
        "evaluation_gain_ci95_high": gain_ci[1],
        "changed_states": sum(bool(row["changed"]) for row in rows),
        "change_fraction": sum(bool(row["changed"]) for row in rows) / len(rows),
        "changed_every_phase": all(
            any(bool(row["changed"]) for row in rows if row["phase"] == phase)
            for phase in PHASES
        ),
        "full_discovery_action_agreement_fraction": sum(
            bool(row["full_discovery_action_agreement"]) for row in rows
        )
        / len(rows),
        "equivalent_regret_fraction": sum(regret <= 0.025 for regret in regrets)
        / len(regrets),
        "mean_full_frontier_regret": statistics.fmean(regrets),
        "p90_full_frontier_regret": _nearest_rank(regrets, 0.90),
        "mean_retained_roots": statistics.fmean(retained),
        "mean_uncapped_roots": statistics.fmean(uncapped),
        "mean_root_ratio": statistics.fmean(retained) / statistics.fmean(uncapped),
        "retained_root_quantiles": {
            "p50": _nearest_rank(retained, 0.50),
            "p90": _nearest_rank(retained, 0.90),
            "p99": _nearest_rank(retained, 0.99),
            "maximum": max(retained),
        },
        "theoretical_maximum_branches_per_search": 48 * cap,
        "evaluation_half_mean_gains": [
            statistics.fmean(float(row["evaluation_half_gains"][index]) for row in rows)
            for index in range(evaluation_halves)
        ],
        "phase_breakdown": phase_breakdown,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _nearest_rank(sorted_values: Sequence[float | int], quantile: float) -> float | int:
    if not sorted_values:
        raise ValueError("Quantile requires at least one value.")
    index = min(math.ceil(quantile * len(sorted_values)) - 1, len(sorted_values) - 1)
    return sorted_values[max(index, 0)]


def _discovery_gates(
    cfg: ComputeAwareRootCapConfig,
    summary: dict[str, Any],
    uncapped_mean_gain: float,
) -> dict[str, bool]:
    return {
        "mean_gain_at_least_0p045": float(summary["mean_evaluation_gain"])
        >= cfg.minimum_discovery_gain,
        "preserves_at_least_80pct_uncapped_mean_gain": float(summary["mean_evaluation_gain"])
        >= cfg.minimum_discovery_preservation * uncapped_mean_gain,
        "gain_ci95_lower_above_zero": float(summary["evaluation_gain_ci95_low"]) > 0.0,
        "change_fraction_at_least_15pct": float(summary["change_fraction"])
        >= cfg.minimum_change_fraction,
        "changed_every_phase": bool(summary["changed_every_phase"]),
        "positive_mean_gain_every_phase": all(
            float(summary["phase_breakdown"][phase]["mean_gain"]) > 0.0
            for phase in PHASES
        ),
        "equivalent_regret_fraction_at_least_60pct": float(
            summary["equivalent_regret_fraction"]
        )
        >= cfg.minimum_equivalent_fraction,
        "mean_regret_at_most_0p025": float(summary["mean_full_frontier_regret"])
        <= cfg.maximum_mean_regret,
        "p90_regret_at_most_0p075": float(summary["p90_full_frontier_regret"])
        <= cfg.maximum_p90_regret,
        "mean_root_ratio_at_most_0p70": float(summary["mean_root_ratio"])
        <= cfg.maximum_root_ratio,
    }


def _confirmation_gates(
    cfg: ComputeAwareRootCapConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    return {
        "mean_gain_at_least_0p040": float(summary["mean_evaluation_gain"])
        >= cfg.minimum_confirmation_gain,
        "gain_ci95_lower_above_zero": float(summary["evaluation_gain_ci95_low"]) > 0.0,
        "all_three_evaluation_halves_positive": len(summary["evaluation_half_mean_gains"])
        == 3
        and all(float(value) > 0.0 for value in summary["evaluation_half_mean_gains"]),
        "change_fraction_at_least_15pct": float(summary["change_fraction"])
        >= cfg.minimum_change_fraction,
        "changed_every_phase": bool(summary["changed_every_phase"]),
        "positive_mean_gain_every_phase": all(
            float(summary["phase_breakdown"][phase]["mean_gain"]) > 0.0
            for phase in PHASES
        ),
        "no_phase_collapse": all(
            float(summary["phase_breakdown"][phase]["gain_ci95_high"])
            > cfg.phase_collapse_boundary
            for phase in PHASES
        ),
        "equivalent_regret_fraction_at_least_60pct": float(
            summary["equivalent_regret_fraction"]
        )
        >= cfg.minimum_equivalent_fraction,
        "mean_regret_at_most_0p025": float(summary["mean_full_frontier_regret"])
        <= cfg.maximum_mean_regret,
        "p90_regret_at_most_0p075": float(summary["p90_full_frontier_regret"])
        <= cfg.maximum_p90_regret,
        "mean_root_ratio_at_most_0p70": float(summary["mean_root_ratio"])
        <= cfg.maximum_root_ratio,
    }


def _provenance_matches(cfg: ComputeAwareRootCapConfig) -> bool:
    return (
        cfg == ComputeAwareRootCapConfig()
        and _sha256_file(PROTOCOL) == PROTOCOL_SHA256
        and _sha256_file(cfg.result_085) == RESULT_085_SHA256
        and _sha256_file(cfg.result_086) == RESULT_086_SHA256
    )


def _identity(cfg: ComputeAwareRootCapConfig) -> dict[str, Any]:
    return {
        "configuration": _config_payload(cfg),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "result_085_sha256": _sha256_file(cfg.result_085),
        "result_086_sha256": _sha256_file(cfg.result_086),
        "selection_contract": "smallest_eligible_prefix_cap_on_085_v1",
        "confirmation_contract": "one_locked_selected_cap_on_086_v1",
    }


def _validate_config(cfg: ComputeAwareRootCapConfig) -> None:
    if cfg != ComputeAwareRootCapConfig():
        raise ValueError("Experiment 088 permits only its preregistered default configuration.")
    for path in (PROTOCOL, cfg.result_085, cfg.result_086):
        if not path.exists():
            raise FileNotFoundError(path)
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 088 frozen protocol or input provenance mismatch.")


def _config_payload(cfg: ComputeAwareRootCapConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        for key, value in asdict(cfg).items()
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()
