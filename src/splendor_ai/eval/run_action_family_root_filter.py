"""Run preregistered experiment 089: action-family expanded-root filter."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval import run_expanded_root_resolver_headroom as audit085
from splendor_ai.eval.run_compute_aware_root_cap_frontier import (
    RESULT_085_SHA256,
    RESULT_086_SHA256,
    _best_index,
    _confirmation_gates,
    _discovery_gates,
    _nearest_rank,
    _tables,
    _validate_input,
)


PROTOCOL = Path("docs/experiments/action_family_root_filter_089.md")
PROTOCOL_SHA256 = "437ebefc10b382fc181e6068898302dd6219fbb86f02107780a33d759b11dce9"
PHASES = ("early", "mid", "late")
BUY = frozenset(("BUY_VISIBLE", "BUY_RESERVED"))
PUBLIC_OPTIONAL_FAMILIES = frozenset((*BUY, "RESERVE_VISIBLE", "TAKE_TOKENS"))
FILTERS: tuple[tuple[str, frozenset[str]], ...] = (
    ("buy_only", BUY),
    ("reserve_visible_only", frozenset(("RESERVE_VISIBLE",))),
    ("take_tokens_only", frozenset(("TAKE_TOKENS",))),
    ("card_actions", frozenset((*BUY, "RESERVE_VISIBLE"))),
    ("buy_or_take", frozenset((*BUY, "TAKE_TOKENS"))),
    ("reserve_or_take", frozenset(("RESERVE_VISIBLE", "TAKE_TOKENS"))),
    ("all_public_non_deck", PUBLIC_OPTIONAL_FAMILIES),
)
ALL_ACTION_NAMES = frozenset(
    ("PASS", "TAKE_TOKENS", "RESERVE_VISIBLE", "RESERVE_DECK", "BUY_VISIBLE", "BUY_RESERVED")
)


@dataclass(frozen=True, slots=True)
class ActionFamilyRootFilterConfig:
    output_dir: Path = Path("outputs/action_family_root_filter_089")
    result_085: Path = Path("outputs/expanded_root_resolver_headroom_085/result.json")
    result_086: Path = Path("outputs/expanded_root_regret_stability_replication_086/result.json")
    bootstrap_samples: int = 20_000
    discovery_bootstrap_base: int = 890_085
    confirmation_bootstrap_base: int = 890_086
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


def run_audit(config: ActionFamilyRootFilterConfig | None = None) -> dict[str, Any]:
    cfg = config or ActionFamilyRootFilterConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    identity = _identity(cfg)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 089 result does not match frozen identity.")
        return result

    payload085 = json.loads(cfg.result_085.read_text(encoding="utf-8"))
    payload086 = json.loads(cfg.result_086.read_text(encoding="utf-8"))
    rows085, base_integrity085 = _validate_input(payload085, "085")
    rows086, base_integrity086 = _validate_input(payload086, "086")
    family_integrity085 = _family_integrity(rows085)
    family_integrity086 = _family_integrity(rows086)
    integrity_gates = {
        "frozen_protocol_and_input_hashes": _provenance_matches(cfg),
        "085_uncapped_input_reproduces": all(base_integrity085.values()),
        "086_uncapped_input_reproduces": all(base_integrity086.values()),
        "085_action_families_parse_and_optional_roots_are_public_non_deck": all(
            family_integrity085.values()
        ),
        "086_action_families_parse_and_optional_roots_are_public_non_deck": all(
            family_integrity086.values()
        ),
        "frontier_names_and_order_are_frozen": tuple(name for name, _ in FILTERS)
        == (
            "buy_only",
            "reserve_visible_only",
            "take_tokens_only",
            "card_actions",
            "buy_or_take",
            "reserve_or_take",
            "all_public_non_deck",
        ),
    }
    if not all(integrity_gates.values()):
        result = _base_result(identity, integrity_gates, base_integrity085, base_integrity086)
        result.update(
            {
                "valid": False,
                "passed": False,
                "family_integrity": {"085": family_integrity085, "086": family_integrity086},
                "decision": "action_family_filter_089_invalid_no_decision",
            }
        )
        _atomic_write(result_path, result)
        return result

    uncapped085 = float(payload085["summary"]["mean_evaluation_gain"])
    discovery_filters: dict[str, Any] = {}
    eligible: list[tuple[float, int, str]] = []
    for frontier_index, (name, families) in enumerate(FILTERS, start=1):
        filtered_rows = _replay_filter(rows085, name, families, "085")
        summary = _summarize_filter(
            filtered_rows,
            name,
            cfg.bootstrap_samples,
            cfg.discovery_bootstrap_base + frontier_index,
            evaluation_halves=0,
        )
        repeated = _summarize_filter(
            filtered_rows,
            name,
            cfg.bootstrap_samples,
            cfg.discovery_bootstrap_base + frontier_index,
            evaluation_halves=0,
        )
        gates = _discovery_gates(cfg, summary, uncapped085)
        gates["bootstrap_repeat_identical"] = summary == repeated
        is_eligible = all(gates.values())
        discovery_filters[name] = {
            "frontier_index": frontier_index,
            "families": sorted(families),
            "summary": summary,
            "gates": gates,
            "eligible": is_eligible,
        }
        if is_eligible:
            eligible.append((float(summary["mean_root_ratio"]), frontier_index, name))

    full_control = discovery_filters["all_public_non_deck"]["summary"]
    control_reproduces = math.isclose(
        float(full_control["mean_evaluation_gain"]), uncapped085, abs_tol=1e-12
    ) and float(full_control["mean_root_ratio"]) == 1.0
    if not control_reproduces:
        result = _base_result(identity, integrity_gates, base_integrity085, base_integrity086)
        result.update(
            {
                "valid": False,
                "passed": False,
                "family_integrity": {"085": family_integrity085, "086": family_integrity086},
                "discovery_filters": discovery_filters,
                "uncapped_control_reproduces": False,
                "decision": "action_family_filter_089_invalid_no_decision",
            }
        )
        _atomic_write(result_path, result)
        return result

    selected_name = min(eligible)[2] if eligible else None
    if selected_name is None:
        result = _base_result(identity, integrity_gates, base_integrity085, base_integrity086)
        result.update(
            {
                "valid": True,
                "passed": False,
                "family_integrity": {"085": family_integrity085, "086": family_integrity086},
                "uncapped_control_reproduces": True,
                "discovery_filters": discovery_filters,
                "selected_filter": None,
                "confirmation_opened": False,
                "confirmation": None,
                "decision": "no_action_family_root_filter_selected",
            }
        )
        _atomic_write(result_path, result)
        return result

    frontier_index = next(index for index, (name, _) in enumerate(FILTERS, start=1) if name == selected_name)
    selected_families = dict(FILTERS)[selected_name]
    confirmation_rows = _replay_filter(rows086, selected_name, selected_families, "086")
    confirmation_summary = _summarize_filter(
        confirmation_rows,
        selected_name,
        cfg.bootstrap_samples,
        cfg.confirmation_bootstrap_base + frontier_index,
        evaluation_halves=3,
    )
    confirmation_repeated = _summarize_filter(
        confirmation_rows,
        selected_name,
        cfg.bootstrap_samples,
        cfg.confirmation_bootstrap_base + frontier_index,
        evaluation_halves=3,
    )
    confirmation_gates = _confirmation_gates(cfg, confirmation_summary)
    confirmation_gates["bootstrap_repeat_identical"] = (
        confirmation_summary == confirmation_repeated
    )
    passed = all(confirmation_gates.values())
    decision = (
        "action_family_root_filter_confirmed_for_future_fresh_evaluation"
        if passed
        else "selected_action_family_filter_not_confirmed_close_family_frontier"
    )
    result = _base_result(identity, integrity_gates, base_integrity085, base_integrity086)
    result.update(
        {
            "valid": True,
            "passed": passed,
            "family_integrity": {"085": family_integrity085, "086": family_integrity086},
            "uncapped_control_reproduces": True,
            "discovery_filters": discovery_filters,
            "selected_filter": selected_name,
            "selected_families": sorted(selected_families),
            "confirmation_opened": True,
            "confirmation": {
                "summary": confirmation_summary,
                "gates": confirmation_gates,
                "rows": confirmation_rows,
            },
            "decision": decision,
            "future_fresh_evaluation_design_authorized": passed,
        }
    )
    _atomic_write(result_path, result)
    print(f"089 result: {'PASS' if passed else 'FAIL'} filter={selected_name}", flush=True)
    return result


def _base_result(
    identity: dict[str, Any],
    integrity_gates: dict[str, bool],
    integrity085: dict[str, bool],
    integrity086: dict[str, bool],
) -> dict[str, Any]:
    return {
        "experiment": "089_action_family_root_filter",
        "identity": identity,
        "integrity_gates": integrity_gates,
        "input_integrity": {"085": integrity085, "086": integrity086},
        "research_champion": "public_belief_resolver_050_051",
        "fast_champion": "035",
        "gameplay_authorized": False,
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "running_087_changed": False,
    }


def _parse_action_name(representation: str) -> str:
    key = ast.literal_eval(representation)
    if not isinstance(key, tuple) or len(key) != 5 or not isinstance(key[0], str):
        raise ValueError("Stored canonical root key has an unexpected schema.")
    if key[0] not in ALL_ACTION_NAMES:
        raise ValueError("Stored canonical root key has an unknown action family.")
    return key[0]


def _family_integrity(rows: list[dict[str, Any]]) -> dict[str, bool]:
    parseable = True
    lengths_match = True
    optional_public_non_deck = True
    for row in rows:
        keys = list(row.get("expanded_root_keys", []))
        lengths_match = lengths_match and len(keys) == int(row.get("expanded_root_count", -1))
        try:
            names = [_parse_action_name(value) for value in keys]
        except (ValueError, SyntaxError):
            parseable = False
            names = []
        optional_public_non_deck = optional_public_non_deck and all(
            name in PUBLIC_OPTIONAL_FAMILIES for name in names[4:]
        )
    return {
        "canonical_root_keys_parse": parseable,
        "root_key_lengths_match_tables": lengths_match,
        "every_optional_root_is_supported_public_non_deck_family": optional_public_non_deck,
    }


def _replay_filter(
    rows: list[dict[str, Any]],
    filter_name: str,
    families: frozenset[str],
    dataset: str,
) -> list[dict[str, Any]]:
    replayed = []
    for row in rows:
        discovery, evaluation, halves = _tables(row, dataset)
        names = [_parse_action_name(value) for value in row["expanded_root_keys"]]
        retained_indices = [
            index for index, name in enumerate(names) if index < 4 or name in families
        ]
        selected = max(
            retained_indices,
            key=lambda index: (float(discovery[index]), -index),
        )
        champion = int(
            row["champion_discovery_best_index"]
            if dataset == "085"
            else row["champion_selected_index"]
        )
        full_selected = _best_index(discovery)
        regret = max(float(value) for value in evaluation) - float(evaluation[selected])
        retained_families = Counter(names[index] for index in retained_indices[4:])
        replayed.append(
            {
                "sample_index": int(row["sample_index"]),
                "phase": str(row["phase"]),
                "filter": filter_name,
                "uncapped_roots": len(discovery),
                "retained_roots": len(retained_indices),
                "retained_optional_family_counts": dict(sorted(retained_families.items())),
                "selected_index": selected,
                "selected_action_family": names[selected],
                "champion_selected_index": champion,
                "full_discovery_selected_index": full_selected,
                "changed": selected != champion,
                "full_discovery_action_agreement": selected == full_selected,
                "evaluation_gain": float(evaluation[selected]) - float(evaluation[champion]),
                "full_frontier_evaluation_regret": max(regret, 0.0),
                "evaluation_half_gains": [
                    float(values[selected]) - float(values[champion]) for values in halves
                ],
            }
        )
    return replayed


def _summarize_filter(
    rows: list[dict[str, Any]],
    filter_name: str,
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
    retained_family_counts: Counter[str] = Counter()
    selected_family_counts: Counter[str] = Counter()
    for row in rows:
        retained_family_counts.update(row["retained_optional_family_counts"])
        selected_family_counts[str(row["selected_action_family"])] += 1
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
    mean_retained = statistics.fmean(retained)
    mean_uncapped = statistics.fmean(uncapped)
    return {
        "filter": filter_name,
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
        "mean_retained_roots": mean_retained,
        "mean_uncapped_roots": mean_uncapped,
        "mean_root_ratio": mean_retained / mean_uncapped,
        "retained_root_quantiles": {
            "p50": _nearest_rank(retained, 0.50),
            "p90": _nearest_rank(retained, 0.90),
            "p99": _nearest_rank(retained, 0.99),
            "maximum": max(retained),
        },
        "mean_observed_branches_per_search": 48 * mean_retained,
        "maximum_observed_branches_per_search": 48 * max(retained),
        "retained_optional_family_counts": dict(sorted(retained_family_counts.items())),
        "selected_action_family_counts": dict(sorted(selected_family_counts.items())),
        "evaluation_half_mean_gains": [
            statistics.fmean(float(row["evaluation_half_gains"][index]) for row in rows)
            for index in range(evaluation_halves)
        ],
        "phase_breakdown": phase_breakdown,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _provenance_matches(cfg: ActionFamilyRootFilterConfig) -> bool:
    return (
        cfg == ActionFamilyRootFilterConfig()
        and _sha256_file(PROTOCOL) == PROTOCOL_SHA256
        and _sha256_file(cfg.result_085) == RESULT_085_SHA256
        and _sha256_file(cfg.result_086) == RESULT_086_SHA256
    )


def _identity(cfg: ActionFamilyRootFilterConfig) -> dict[str, Any]:
    return {
        "configuration": _config_payload(cfg),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "result_085_sha256": _sha256_file(cfg.result_085),
        "result_086_sha256": _sha256_file(cfg.result_086),
        "frontier": [
            {"index": index, "name": name, "families": sorted(families)}
            for index, (name, families) in enumerate(FILTERS, start=1)
        ],
        "selection_contract": "lowest_root_ratio_eligible_then_frozen_order_on_085_v1",
        "confirmation_contract": "one_locked_selected_family_filter_on_086_v1",
    }


def _validate_config(cfg: ActionFamilyRootFilterConfig) -> None:
    if cfg != ActionFamilyRootFilterConfig():
        raise ValueError("Experiment 089 permits only its preregistered default configuration.")
    for path in (PROTOCOL, cfg.result_085, cfg.result_086):
        if not path.exists():
            raise FileNotFoundError(path)
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 089 frozen protocol or input provenance mismatch.")


def _config_payload(cfg: ActionFamilyRootFilterConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
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
