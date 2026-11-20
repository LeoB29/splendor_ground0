"""Offline action-space fragmentation audit for experiment 135."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable


PROTOCOL = Path("docs/engineering/alphazero_action_space_fragmentation_audit_135.md")
PROTOCOL_SHA256 = "0a4f24d673fd1f88033da6c5178b685d01aace8964745baf1746e4333944def7"
OUTPUT_DIR = Path("outputs/alphazero_action_space_fragmentation_audit_135")
RESULT_PATH = OUTPUT_DIR / "result.json"
BOOTSTRAP_SEED = 1_350_135
BOOTSTRAP_SAMPLES = 5_000
EXPECTED_VISITS = 1_024
EXPECTED_GAMES = 256
EXPECTED_ROWS = 15_916


@dataclass(frozen=True, slots=True)
class SourceSpec:
    path: Path
    rows: int
    size: int
    sha256: str


SOURCES = (
    SourceSpec(
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_0/sanitized.jsonl"),
        3_956,
        74_919_121,
        "f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b",
    ),
    SourceSpec(
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_1/sanitized.jsonl"),
        4_004,
        76_570_863,
        "c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60",
    ),
    SourceSpec(
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_2/sanitized.jsonl"),
        3_956,
        76_478_700,
        "00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8",
    ),
    SourceSpec(
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_3/sanitized.jsonl"),
        4_000,
        76_509_765,
        "7bd6b03227aa6e4e5ce1bb902adda6021d438b5e927d1da2b9f56793424dd95e",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "middle"
    return "late"


def _sorted_tokens(payload: dict[str, Any], field: str) -> tuple[str, ...]:
    return tuple(sorted(str(token) for token in payload.get(field, [])))


def intent_key(
    payload: dict[str, Any],
    action_card_id: int,
    *,
    preserve_noble: bool,
) -> tuple[object, ...]:
    """Return the frozen strategic grouping key for one stored action."""

    action_type = str(payload.get("action_type"))
    if action_type == "TAKE_TOKENS":
        return (action_type, _sorted_tokens(payload, "take_tokens"))
    if action_type == "RESERVE_VISIBLE":
        return (action_type, int(action_card_id))
    if action_type == "RESERVE_DECK":
        return (action_type, int(payload.get("tier")))
    if action_type in ("BUY_VISIBLE", "BUY_RESERVED"):
        base: tuple[object, ...] = (action_type, int(action_card_id))
        if preserve_noble:
            return (*base, payload.get("noble_id"))
        return base
    if action_type == "PASS":
        return (action_type,)
    raise ValueError(f"Unexpected action type: {action_type!r}")


def _groups(keys: list[tuple[object, ...]]) -> dict[tuple[object, ...], list[int]]:
    result: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        result[key].append(index)
    return dict(result)


def _group_argmax(
    groups: dict[tuple[object, ...], list[int]],
    values: list[float] | list[int],
) -> tuple[object, ...]:
    return max(
        groups,
        key=lambda key: (
            sum(float(values[index]) for index in groups[key]),
            repr(key),
        ),
    )


def _validate_row(row: dict[str, Any]) -> None:
    required = (
        "game_seed",
        "turn_index",
        "player_id",
        "action_payloads",
        "action_card_ids",
        "action_noble_ids",
        "root_network_priors",
        "root_visits",
        "root_q_values",
        "raw_policy_top_row",
        "visit_argmax_row",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"Replay row is missing fields: {missing}")

    lengths = {
        len(row[field])
        for field in (
            "action_payloads",
            "action_card_ids",
            "action_noble_ids",
            "root_network_priors",
            "root_visits",
            "root_q_values",
        )
    }
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError(f"Action-aligned vectors disagree: {sorted(lengths)}")
    count = next(iter(lengths))
    for field in ("raw_policy_top_row", "visit_argmax_row"):
        index = int(row[field])
        if index < 0 or index >= count:
            raise ValueError(f"{field} is out of range: {index} / {count}")

    visits = [int(value) for value in row["root_visits"]]
    if any(value < 0 for value in visits) or sum(visits) != EXPECTED_VISITS:
        raise ValueError(f"Invalid root visit vector: sum={sum(visits)}")
    priors = [float(value) for value in row["root_network_priors"]]
    q_values = [float(value) for value in row["root_q_values"]]
    if any(not math.isfinite(value) or value < 0.0 for value in priors):
        raise ValueError("Root network priors must be finite and nonnegative.")
    if any(not math.isfinite(value) for value in q_values):
        raise ValueError("Root Q values must be finite.")
    if not math.isclose(sum(priors), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"Root network priors do not sum to one: {sum(priors)}")


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    _validate_row(row)
    payloads = [dict(payload) for payload in row["action_payloads"]]
    card_ids = [int(value) for value in row["action_card_ids"]]
    visits = [int(value) for value in row["root_visits"]]
    priors = [float(value) for value in row["root_network_priors"]]
    q_values = [float(value) for value in row["root_q_values"]]
    exact_count = len(payloads)

    noble_keys = [
        intent_key(payload, card_id, preserve_noble=True)
        for payload, card_id in zip(payloads, card_ids, strict=True)
    ]
    core_keys = [
        intent_key(payload, card_id, preserve_noble=False)
        for payload, card_id in zip(payloads, card_ids, strict=True)
    ]
    noble_groups = _groups(noble_keys)
    core_groups = _groups(core_keys)
    raw_top = int(row["raw_policy_top_row"])
    visit_top = int(row["visit_argmax_row"])

    noble_prior_top = _group_argmax(noble_groups, priors)
    noble_visit_top = _group_argmax(noble_groups, visits)
    core_prior_top = _group_argmax(core_groups, priors)
    core_visit_top = _group_argmax(core_groups, visits)

    core_multi_indices = {
        index
        for indices in core_groups.values()
        if len(indices) > 1
        for index in indices
    }
    winning_indices = core_groups[core_visit_top]
    winning_mass = sum(visits[index] for index in winning_indices)
    winning_concentration = max(visits[index] for index in winning_indices) / winning_mass

    q_ranges: list[float] = []
    variation_counts: Counter[str] = Counter()
    for indices in core_groups.values():
        if len(indices) <= 1:
            continue
        varying: list[str] = []
        for field in ("return_tokens", "spend_tokens"):
            values = {_sorted_tokens(payloads[index], field) for index in indices}
            if len(values) > 1:
                varying.append(field)
        if len({payloads[index].get("noble_id") for index in indices}) > 1:
            varying.append("noble_id")
        variation_counts["+".join(varying) if varying else "other"] += 1
        visited = [index for index in indices if visits[index] > 0]
        if len(visited) >= 2:
            values = [q_values[index] for index in visited]
            q_ranges.append(max(values) - min(values))

    return {
        "game_seed": int(row["game_seed"]),
        "turn_index": int(row["turn_index"]),
        "phase": phase(int(row["turn_index"])),
        "exact_count": exact_count,
        "noble_intent_count": len(noble_groups),
        "core_intent_count": len(core_groups),
        "noble_ratio": len(noble_groups) / exact_count,
        "core_ratio": len(core_groups) / exact_count,
        "has_core_multi_variant": bool(core_multi_indices),
        "core_multi_visit_mass": sum(visits[index] for index in core_multi_indices)
        / EXPECTED_VISITS,
        "exact_agreement": raw_top == visit_top,
        "noble_intent_agreement": noble_keys[raw_top] == noble_keys[visit_top],
        "core_intent_agreement": core_keys[raw_top] == core_keys[visit_top],
        "aggregate_noble_agreement": noble_prior_top == noble_visit_top,
        "aggregate_core_agreement": core_prior_top == core_visit_top,
        "aggregate_noble_changes_visit_winner": noble_visit_top != noble_keys[visit_top],
        "aggregate_core_changes_visit_winner": core_visit_top != core_keys[visit_top],
        "winning_core_concentration": winning_concentration,
        "q_ranges": q_ranges,
        "variation_counts": dict(variation_counts),
        "winning_core_family": str(core_visit_top[0]),
    }


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"count": 0}
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "median": _quantile(materialized, 0.5),
        "p90": _quantile(materialized, 0.9),
        "maximum": max(materialized),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty audit slice.")
    variation_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    q_ranges: list[float] = []
    for row in rows:
        variation_counts.update(row["variation_counts"])
        family_counts[row["winning_core_family"]] += 1
        q_ranges.extend(row["q_ranges"])

    q_summary = _distribution(q_ranges)
    if q_ranges:
        q_summary["fraction_at_most_0_025"] = sum(value <= 0.025 for value in q_ranges) / len(q_ranges)
        q_summary["fraction_at_most_0_05"] = sum(value <= 0.05 for value in q_ranges) / len(q_ranges)

    exact_agreement = statistics.fmean(float(row["exact_agreement"]) for row in rows)
    noble_agreement = statistics.fmean(float(row["noble_intent_agreement"]) for row in rows)
    core_agreement = statistics.fmean(float(row["core_intent_agreement"]) for row in rows)
    aggregate_noble = statistics.fmean(float(row["aggregate_noble_agreement"]) for row in rows)
    aggregate_core = statistics.fmean(float(row["aggregate_core_agreement"]) for row in rows)
    return {
        "positions": len(rows),
        "games": len({row["game_seed"] for row in rows}),
        "action_counts": {
            "exact": _distribution(row["exact_count"] for row in rows),
            "noble_preserving_intent": _distribution(row["noble_intent_count"] for row in rows),
            "core_intent": _distribution(row["core_intent_count"] for row in rows),
        },
        "compression": {
            "mean_noble_intent_to_exact_ratio": statistics.fmean(row["noble_ratio"] for row in rows),
            "mean_core_intent_to_exact_ratio": statistics.fmean(row["core_ratio"] for row in rows),
            "positions_with_core_multi_variant": statistics.fmean(
                float(row["has_core_multi_variant"]) for row in rows
            ),
            "core_multi_variant_visit_mass": statistics.fmean(
                row["core_multi_visit_mass"] for row in rows
            ),
        },
        "agreement": {
            "exact_raw_vs_visit_argmax": exact_agreement,
            "noble_intent_of_exact_argmaxes": noble_agreement,
            "core_intent_of_exact_argmaxes": core_agreement,
            "noble_intent_gain_over_exact": noble_agreement - exact_agreement,
            "core_intent_gain_over_exact": core_agreement - exact_agreement,
            "aggregate_noble_prior_vs_visit": aggregate_noble,
            "aggregate_core_prior_vs_visit": aggregate_core,
            "aggregate_noble_gain_over_exact": aggregate_noble - exact_agreement,
            "aggregate_core_gain_over_exact": aggregate_core - exact_agreement,
            "aggregate_noble_changes_single_visit_winner": statistics.fmean(
                float(row["aggregate_noble_changes_visit_winner"]) for row in rows
            ),
            "aggregate_core_changes_single_visit_winner": statistics.fmean(
                float(row["aggregate_core_changes_visit_winner"]) for row in rows
            ),
        },
        "winning_core_variant_concentration": _distribution(
            row["winning_core_concentration"] for row in rows
        ),
        "within_core_group_q_range": q_summary,
        "multi_variant_group_causes": dict(sorted(variation_counts.items())),
        "winning_core_family_mix": {
            key: {"count": value, "fraction": value / len(rows)}
            for key, value in sorted(family_counts.items())
        },
    }


def _bootstrap(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | list[float]]]:
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[int(row["game_seed"])].append(row)
    seeds = sorted(by_game)
    if len(seeds) != EXPECTED_GAMES:
        raise ValueError(f"Expected {EXPECTED_GAMES} games, found {len(seeds)}")
    game_stats: dict[int, tuple[int, int, int, int, int, int, int]] = {}
    for seed, game_rows in by_game.items():
        game_stats[seed] = (
            len(game_rows),
            sum(int(row["exact_agreement"]) for row in game_rows),
            sum(int(row["noble_intent_agreement"]) for row in game_rows),
            sum(int(row["core_intent_agreement"]) for row in game_rows),
            sum(int(row["aggregate_core_agreement"]) for row in game_rows),
            sum(int(row["core_intent_count"]) for row in game_rows),
            sum(int(row["exact_count"]) for row in game_rows),
        )

    samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_SAMPLES):
        totals = [0] * 7
        for _game in seeds:
            picked = game_stats[seeds[rng.randrange(len(seeds))]]
            for index, value in enumerate(picked):
                totals[index] += value
        row_count, exact, noble, core, aggregate_core, core_count, exact_count = totals
        samples["noble_intent_gain_over_exact"].append((noble - exact) / row_count)
        samples["core_intent_gain_over_exact"].append((core - exact) / row_count)
        samples["aggregate_core_gain_over_exact"].append((aggregate_core - exact) / row_count)
        samples["core_intent_to_exact_count_ratio"].append(core_count / exact_count)

    point = {
        "noble_intent_gain_over_exact": statistics.fmean(
            float(row["noble_intent_agreement"]) - float(row["exact_agreement"])
            for row in rows
        ),
        "core_intent_gain_over_exact": statistics.fmean(
            float(row["core_intent_agreement"]) - float(row["exact_agreement"])
            for row in rows
        ),
        "aggregate_core_gain_over_exact": statistics.fmean(
            float(row["aggregate_core_agreement"]) - float(row["exact_agreement"])
            for row in rows
        ),
        "core_intent_to_exact_count_ratio": sum(row["core_intent_count"] for row in rows)
        / sum(row["exact_count"] for row in rows),
    }
    return {
        name: {
            "point": point[name],
            "ci95": [_quantile(values, 0.025), _quantile(values, 0.975)],
        }
        for name, values in samples.items()
    }


def preflight() -> dict[str, Any]:
    checks: dict[str, bool] = {
        "protocol_hash": PROTOCOL.exists() and sha256_file(PROTOCOL) == PROTOCOL_SHA256,
    }
    source_rows: list[dict[str, Any]] = []
    for spec in SOURCES:
        exists = spec.path.exists()
        size_matches = exists and spec.path.stat().st_size == spec.size
        hash_matches = exists and sha256_file(spec.path) == spec.sha256
        checks[f"{spec.path.parent.name}_exists"] = exists
        checks[f"{spec.path.parent.name}_size"] = size_matches
        checks[f"{spec.path.parent.name}_hash"] = hash_matches
        source_rows.append(
            {
                "path": str(spec.path),
                "expected_rows": spec.rows,
                "bytes": spec.size,
                "sha256": spec.sha256,
            }
        )
    return {
        "experiment": 135,
        "kind": "alphazero_action_space_fragmentation_audit",
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": source_rows,
        "checks": checks,
        "passed": all(checks.values()),
        "no_training": True,
        "no_search": True,
        "no_gameplay": True,
        "reads_experiment_134": False,
    }


def run_audit() -> dict[str, Any]:
    started = time.perf_counter()
    manifest = preflight()
    if not manifest["passed"]:
        raise RuntimeError("Experiment 135 preflight failed.")

    analyzed: list[dict[str, Any]] = []
    seen_positions: set[tuple[int, int, int]] = set()
    realized_rows: dict[str, int] = {}
    for spec in SOURCES:
        row_count = 0
        with spec.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                item = analyze_row(row)
                identity = (item["game_seed"], item["turn_index"], int(row["player_id"]))
                if identity in seen_positions:
                    raise ValueError(f"Duplicate replay position: {identity}")
                seen_positions.add(identity)
                analyzed.append(item)
                row_count += 1
        if row_count != spec.rows:
            raise ValueError(f"Unexpected row count for {spec.path}: {row_count} != {spec.rows}")
        realized_rows[str(spec.path)] = row_count

    if len(analyzed) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(analyzed)}")
    games = {row["game_seed"] for row in analyzed}
    if len(games) != EXPECTED_GAMES:
        raise ValueError(f"Expected {EXPECTED_GAMES} games, found {len(games)}")

    overall = summarize(analyzed)
    phases = {
        name: summarize([row for row in analyzed if row["phase"] == name])
        for name in ("early", "middle", "late")
    }
    bootstrap = _bootstrap(analyzed)
    compression = overall["compression"]
    agreement = overall["agreement"]
    structural = (
        compression["mean_core_intent_to_exact_ratio"] <= 0.80
        and compression["core_multi_variant_visit_mass"] >= 0.20
    )
    discrepancy = (
        agreement["core_intent_gain_over_exact"] >= 0.05
        and bootstrap["core_intent_gain_over_exact"]["ci95"][0] > 0.02
    )
    if structural and discrepancy:
        decision = "material_action_fragmentation_supports_hierarchical_design_audit"
    elif structural:
        decision = "structural_action_fragmentation_without_policy_rescue_review_afterstate"
    else:
        decision = "action_fragmentation_not_material_keep_current_encoding"

    result = {
        **manifest,
        "status": "complete",
        "passed": True,
        "source_accounting": {
            "realized_rows": realized_rows,
            "positions": len(analyzed),
            "games": len(games),
            "unique_position_identities": len(seen_positions),
        },
        "overall": overall,
        "phase_breakdown": phases,
        "game_cluster_bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "metrics": bootstrap,
        },
        "gates": {
            "material_structural_compression": structural,
            "material_detail_only_policy_discrepancy": discrepancy,
        },
        "decision": decision,
        "hierarchical_training_authorized": False,
        "afterstate_training_authorized": False,
        "gameplay_authorized": False,
        "automatic_continuation": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

