"""Replay-only blind-reserve and refill-coverage audit for experiment 136."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from splendor_ai.alphazero.search import actions_by_canonical_successor_key
from splendor_ai.encoding import (
    encode_canonical_action_entities,
    encode_canonical_state_entities,
)
from splendor_ai.encoding.entity_encoding import (
    ENTITY_BOARD_CARD,
    ENTITY_DECK,
    ENTITY_OPPONENT_BLIND_RESERVED,
    ENTITY_OPPONENT_PURCHASED,
    ENTITY_OPPONENT_VISIBLE_RESERVED,
    ENTITY_SELF_PURCHASED,
    ENTITY_SELF_RESERVED,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.data import build_base_deck_by_tier
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE, SplendorState
from splendor_ai.training.replay import serialize_action


PROTOCOL = Path("docs/engineering/alphazero_blind_reserve_refill_coverage_audit_136.md")
PROTOCOL_SHA256 = "001692a92e5889b38133e24032601f2d0fbc8ec10fe939895d53deea960e08ce"
OUTPUT_DIR = Path("outputs/alphazero_blind_reserve_refill_coverage_audit_136")
RESULT_PATH = OUTPUT_DIR / "result.json"
DECK_NORMALIZERS = {1: 36.0, 2: 26.0, 3: 16.0}
VISIBLE_REFILL_TYPES = {"BUY_VISIBLE", "RESERVE_VISIBLE"}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    label: str
    generation: int
    simulations: int
    path: Path
    rows: int
    size: int
    sha256: str


SOURCES = (
    SourceSpec(
        "generation_008", 8, 256,
        Path("outputs/alphazero_continued_scaling_104/selfplay/generation_008_sanitized.jsonl"),
        16_030, 307_848_775,
        "96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10",
    ),
    SourceSpec(
        "generation_009", 9, 256,
        Path("outputs/alphazero_continued_scaling_104/selfplay/generation_009_sanitized.jsonl"),
        16_000, 306_518_964,
        "1b13ef038ccc6b3f695d7c7860a2284272ae18c8e75d44028e4cbb6f69914f0c",
    ),
    SourceSpec(
        "generation_010", 10, 256,
        Path("outputs/alphazero_continued_scaling_104/selfplay/generation_010_sanitized.jsonl"),
        15_936, 307_318_318,
        "09279d9593acfd944a32571716a6527c0be40e41055b6fa1f5bc25c88f43a60c",
    ),
    SourceSpec(
        "generation_011_shard_0", 11, 1_024,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_0/sanitized.jsonl"),
        3_956, 74_919_121,
        "f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b",
    ),
    SourceSpec(
        "generation_011_shard_1", 11, 1_024,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_1/sanitized.jsonl"),
        4_004, 76_570_863,
        "c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60",
    ),
    SourceSpec(
        "generation_011_shard_2", 11, 1_024,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_2/sanitized.jsonl"),
        3_956, 76_478_700,
        "00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8",
    ),
    SourceSpec(
        "generation_011_shard_3", 11, 1_024,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_3/sanitized.jsonl"),
        4_000, 76_509_765,
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


def expected_identity_coverage(visits: int, identities: int) -> tuple[float, float]:
    """Return expected covered fraction and single-identity miss probability."""

    if visits < 0 or identities <= 0:
        raise ValueError("visits must be nonnegative and identities positive")
    miss = (1.0 - 1.0 / identities) ** visits
    return 1.0 - miss, miss


def _argmax(values: list[float] | list[int]) -> int:
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
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
        "p10": _quantile(materialized, 0.1),
        "p90": _quantile(materialized, 0.9),
        "minimum": min(materialized),
        "maximum": max(materialized),
    }


def _card_maps() -> tuple[dict[int, set[int]], dict[str, int]]:
    cards = sorted(
        (card for tier_cards in build_base_deck_by_tier().values() for card in tier_cards),
        key=lambda card: card.card_id,
    )
    by_name = {card.card_id: index for index, card in enumerate(cards, start=1)}
    by_tier = {
        tier: {by_name[card.card_id] for card in build_base_deck_by_tier()[tier]}
        for tier in (1, 2, 3)
    }
    return by_tier, by_name


BASE_IDS_BY_TIER, CARD_ID_TO_INDEX = _card_maps()


def possible_refill_sets_from_state(state: SplendorState, observer: int) -> dict[int, set[int]]:
    opponent = state.players[1 - observer]
    result: dict[int, set[int]] = {}
    for tier in (1, 2, 3):
        identities = {CARD_ID_TO_INDEX[card.card_id] for card in state.hidden_tier_decks[tier]}
        identities.update(
            CARD_ID_TO_INDEX[card.card_id]
            for index, card in enumerate(opponent.reserved_cards)
            if card.tier == tier
            and opponent.reserved_card_source(index) != RESERVE_SOURCE_VISIBLE
        )
        result[tier] = identities
    return result


def possible_refill_sets_from_encoding(
    entity_type_ids: list[int] | tuple[int, ...],
    entity_card_ids: list[int] | tuple[int, ...],
) -> dict[int, set[int]]:
    exposed_types = {
        ENTITY_BOARD_CARD,
        ENTITY_SELF_RESERVED,
        ENTITY_OPPONENT_VISIBLE_RESERVED,
        ENTITY_SELF_PURCHASED,
        ENTITY_OPPONENT_PURCHASED,
    }
    exposed = {
        int(card_id)
        for entity_type, card_id in zip(entity_type_ids, entity_card_ids, strict=True)
        if int(entity_type) in exposed_types and int(card_id) > 0
    }
    return {tier: set(ids) - exposed for tier, ids in BASE_IDS_BY_TIER.items()}


def _encoding_key(row: dict[str, Any]) -> str:
    payload = {
        "type": row["entity_type_ids"],
        "card": row["entity_card_ids"],
        "noble": row["entity_noble_ids"],
        "numeric": row["entity_numeric_features"],
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _validate_probabilities(values: list[float], name: str) -> None:
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{name} does not normalize: {sum(values)}")


def _validate_and_bind_actions(
    row: dict[str, Any], state: SplendorState, env: SplendorEnv, spec: SourceSpec
) -> list[Any]:
    required = (
        "generation", "game_seed", "turn_index", "player_id",
        "entity_type_ids", "entity_card_ids", "entity_noble_ids",
        "entity_numeric_features", "action_type_ids", "action_card_ids",
        "action_noble_ids", "action_numeric_features", "action_payloads",
        "root_network_priors", "root_search_priors", "root_visits",
        "root_q_values", "selected_action_row", "raw_policy_top_row",
        "visit_argmax_row", "checkpoint_generation", "search_config",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"Replay row missing fields: {missing}")
    if int(row["generation"]) != spec.generation:
        raise ValueError("Replay generation disagrees with source contract")
    if int(row["checkpoint_generation"]) != spec.generation:
        raise ValueError("Checkpoint generation disagrees with source contract")
    if int(row["search_config"]["simulations"]) != spec.simulations:
        raise ValueError("Search simulations disagree with source contract")

    fields = (
        "action_type_ids", "action_card_ids", "action_noble_ids",
        "action_numeric_features", "action_payloads", "root_network_priors",
        "root_search_priors", "root_visits", "root_q_values",
    )
    lengths = {len(row[field]) for field in fields}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError(f"Action-aligned vector lengths disagree: {sorted(lengths)}")
    count = next(iter(lengths))
    for field in ("selected_action_row", "raw_policy_top_row", "visit_argmax_row"):
        index = int(row[field])
        if not 0 <= index < count:
            raise ValueError(f"{field} outside legal action range")

    network = [float(value) for value in row["root_network_priors"]]
    search = [float(value) for value in row["root_search_priors"]]
    visits = [int(value) for value in row["root_visits"]]
    q_values = [float(value) for value in row["root_q_values"]]
    _validate_probabilities(network, "network priors")
    _validate_probabilities(search, "search priors")
    if any(value < 0 for value in visits) or sum(visits) != spec.simulations:
        raise ValueError("Root visits disagree with source contract")
    if any(not math.isfinite(value) for value in q_values):
        raise ValueError("Root Q values must be finite")

    encoded_state = encode_canonical_state_entities(state, state.current_player)
    if (
        tuple(int(value) for value in row["entity_type_ids"]) != encoded_state.entity_type_ids
        or tuple(int(value) for value in row["entity_card_ids"]) != encoded_state.card_ids
        or tuple(int(value) for value in row["entity_noble_ids"]) != encoded_state.noble_ids
        or tuple(tuple(float(x) for x in values) for values in row["entity_numeric_features"])
        != encoded_state.numeric_features
    ):
        raise ValueError("Stored public state encoding does not match reconstruction")

    legal = list(actions_by_canonical_successor_key(state, env.legal_actions(state)).values())
    by_payload: dict[str, list[Any]] = {}
    for action in legal:
        key = json.dumps(serialize_action(action), sort_keys=True)
        by_payload.setdefault(key, []).append(action)
    actions: list[Any] = []
    for payload in row["action_payloads"]:
        matches = by_payload.get(json.dumps(payload, sort_keys=True), [])
        if len(matches) != 1:
            raise ValueError("Stored action payload does not bind uniquely to canonical legality")
        actions.append(matches[0])
    encoded_actions = encode_canonical_action_entities(state, actions, state.current_player)
    if (
        tuple(int(value) for value in row["action_type_ids"]) != encoded_actions.action_type_ids
        or tuple(int(value) for value in row["action_card_ids"]) != encoded_actions.card_ids
        or tuple(int(value) for value in row["action_noble_ids"]) != encoded_actions.noble_ids
        or tuple(tuple(float(x) for x in values) for values in row["action_numeric_features"])
        != encoded_actions.numeric_features
    ):
        raise ValueError("Stored action encoding does not match reconstruction")
    return actions


def _representation_check(
    row: dict[str, Any], state: SplendorState, encoding_sets: dict[str, tuple[tuple[int, ...], ...]]
) -> None:
    observer = state.current_player
    actual = possible_refill_sets_from_state(state, observer)
    encoded = possible_refill_sets_from_encoding(row["entity_type_ids"], row["entity_card_ids"])
    if actual != encoded:
        raise ValueError("Public encoding cannot reconstruct the possible refill identities")
    deck_counts: dict[int, int] = {}
    blind_counts: Counter[int] = Counter()
    for entity_type, numeric in zip(
        row["entity_type_ids"], row["entity_numeric_features"], strict=True
    ):
        entity_type = int(entity_type)
        if entity_type == ENTITY_DECK:
            tier = int(round(float(numeric[29]) * 3.0))
            deck_counts[tier] = int(round(float(numeric[30]) * DECK_NORMALIZERS[tier]))
        elif entity_type == ENTITY_OPPONENT_BLIND_RESERVED:
            tier = int(round(float(numeric[29]) * 3.0))
            if tier in (1, 2, 3):
                blind_counts[tier] += 1
    if set(deck_counts) != {1, 2, 3}:
        raise ValueError("Public encoding lacks a tier deck-count entity")
    for tier in (1, 2, 3):
        if len(actual[tier]) != deck_counts[tier] + blind_counts[tier]:
            raise ValueError("Possible refill set size disagrees with deck plus blind count")
    signature = tuple(tuple(sorted(actual[tier])) for tier in (1, 2, 3))
    key = _encoding_key(row)
    previous = encoding_sets.setdefault(key, signature)
    if previous != signature:
        raise ValueError("Identical public encodings map to different refill sets")


def _analyze_row(
    row: dict[str, Any], state: SplendorState, actions: list[Any], spec: SourceSpec
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads = [dict(payload) for payload in row["action_payloads"]]
    network = [float(value) for value in row["root_network_priors"]]
    search = [float(value) for value in row["root_search_priors"]]
    visits = [int(value) for value in row["root_visits"]]
    q_values = [float(value) for value in row["root_q_values"]]
    families = [str(payload["action_type"]) for payload in payloads]
    reserve_indices = [index for index, family in enumerate(families) if family == "RESERVE_DECK"]
    nonreserve_indices = [index for index in range(len(families)) if index not in reserve_indices]
    raw_top = int(row["raw_policy_top_row"])
    search_top = _argmax(search)
    visit_top = int(row["visit_argmax_row"])
    selected = int(row["selected_action_row"])
    visited_reserve = [index for index in reserve_indices if visits[index] > 0]
    visited_nonreserve = [index for index in nonreserve_indices if visits[index] > 0]
    q_gap = None
    if visited_reserve and visited_nonreserve:
        q_gap = max(q_values[index] for index in visited_reserve) - max(
            q_values[index] for index in visited_nonreserve
        )
    blind = {
        "generation": spec.generation,
        "game_seed": int(row["game_seed"]),
        "phase": phase(int(row["turn_index"])),
        "reserve_legal": bool(reserve_indices),
        "network_mass": sum(network[index] for index in reserve_indices),
        "search_mass": sum(search[index] for index in reserve_indices),
        "visit_mass": sum(visits[index] for index in reserve_indices) / spec.simulations,
        "network_top": families[raw_top] == "RESERVE_DECK",
        "search_top": families[search_top] == "RESERVE_DECK",
        "visit_top": families[visit_top] == "RESERVE_DECK",
        "selected": families[selected] == "RESERVE_DECK",
        "raw_to_visit_promoted": families[raw_top] != "RESERVE_DECK" and families[visit_top] == "RESERVE_DECK",
        "raw_to_visit_demoted": families[raw_top] == "RESERVE_DECK" and families[visit_top] != "RESERVE_DECK",
        "search_to_visit_promoted": families[search_top] != "RESERVE_DECK" and families[visit_top] == "RESERVE_DECK",
        "search_to_visit_demoted": families[search_top] == "RESERVE_DECK" and families[visit_top] != "RESERVE_DECK",
        "best_visited_q_gap": q_gap,
    }

    possible = possible_refill_sets_from_state(state, state.current_player)
    refill: list[dict[str, Any]] = []
    for index, (payload, action) in enumerate(zip(payloads, actions, strict=True)):
        family = str(payload["action_type"])
        tier = int(action.tier or 0)
        if family not in VISIBLE_REFILL_TYPES or tier not in (1, 2, 3):
            continue
        if state.deck_counts[tier] <= 0 or visits[index] <= 0:
            continue
        identities = len(possible[tier])
        coverage, miss = expected_identity_coverage(visits[index], identities)
        refill.append(
            {
                "generation": spec.generation,
                "game_seed": int(row["game_seed"]),
                "turn_index": int(row["turn_index"]),
                "phase": phase(int(row["turn_index"])),
                "tier": tier,
                "family": family,
                "visits": visits[index],
                "possible_identities": identities,
                "visits_per_identity": visits[index] / identities,
                "expected_distinct_identities": identities * coverage,
                "expected_coverage": coverage,
                "single_identity_miss_probability": miss,
                "priority": visits[index] / spec.simulations >= 0.05,
                "visit_winner": index == visit_top,
                "selected": index == selected,
            }
        )
    return blind, refill


def summarize_blind(rows: list[dict[str, Any]]) -> dict[str, Any]:
    legal = [row for row in rows if row["reserve_legal"]]
    q_rows = [row for row in legal if row["best_visited_q_gap"] is not None]
    denominator = len(legal)
    mean = lambda field: statistics.fmean(float(row[field]) for row in legal) if legal else 0.0
    return {
        "positions": len(rows),
        "reserve_legal_positions": len(legal),
        "reserve_legal_fraction": len(legal) / len(rows) if rows else 0.0,
        "conditional_when_legal": {
            "mean_network_prior_mass": mean("network_mass"),
            "mean_post_noise_search_prior_mass": mean("search_mass"),
            "mean_visit_mass": mean("visit_mass"),
            "mean_search_minus_network_mass": (
                statistics.fmean(row["search_mass"] - row["network_mass"] for row in legal)
                if legal else 0.0
            ),
            "mean_visit_minus_search_mass": (
                statistics.fmean(row["visit_mass"] - row["search_mass"] for row in legal)
                if legal else 0.0
            ),
            "network_top_fraction": mean("network_top"),
            "post_noise_search_top_fraction": mean("search_top"),
            "visit_top_fraction": mean("visit_top"),
            "selected_fraction": mean("selected"),
            "raw_to_visit_promoted": sum(row["raw_to_visit_promoted"] for row in legal),
            "raw_to_visit_demoted": sum(row["raw_to_visit_demoted"] for row in legal),
            "search_to_visit_promoted": sum(row["search_to_visit_promoted"] for row in legal),
            "search_to_visit_demoted": sum(row["search_to_visit_demoted"] for row in legal),
            "best_visited_reserve_q_minus_nonreserve_q": _distribution(
                row["best_visited_q_gap"] for row in q_rows
            ),
        },
        "unconditional_selected_fraction": (
            sum(row["selected"] for row in rows) / len(rows) if rows else 0.0
        ),
        "unconditional_visit_top_fraction": (
            sum(row["visit_top"] for row in rows) / len(rows) if rows else 0.0
        ),
        "accounting_denominator": denominator,
    }


def summarize_refill(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"edges": 0, "positions": 0, "total_visits": 0}
    total_visits = sum(int(row["visits"]) for row in rows)
    weighted_miss = sum(
        float(row["single_identity_miss_probability"]) * int(row["visits"])
        for row in rows
    ) / total_visits
    return {
        "edges": len(rows),
        "positions": len({(row["generation"], row["game_seed"], row["turn_index"]) for row in rows}),
        "total_visits": total_visits,
        "possible_identities": _distribution(row["possible_identities"] for row in rows),
        "visits": _distribution(row["visits"] for row in rows),
        "visits_per_identity": _distribution(row["visits_per_identity"] for row in rows),
        "expected_coverage": _distribution(row["expected_coverage"] for row in rows),
        "mean_single_identity_miss_probability": statistics.fmean(
            row["single_identity_miss_probability"] for row in rows
        ),
        "visit_weighted_mean_single_identity_miss_probability": weighted_miss,
        "visit_weighted_expected_coverage": 1.0 - weighted_miss,
        "fraction_expected_coverage_below_0_80": sum(
            row["expected_coverage"] < 0.80 for row in rows
        ) / len(rows),
        "fraction_expected_coverage_below_0_90": sum(
            row["expected_coverage"] < 0.90 for row in rows
        ) / len(rows),
    }


def _refill_category_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_visited": summarize_refill(rows),
        "priority": summarize_refill([row for row in rows if row["priority"]]),
        "visit_winner": summarize_refill([row for row in rows if row["visit_winner"]]),
        "selected": summarize_refill([row for row in rows if row["selected"]]),
    }


def decision_from_metrics(
    *, representation_passed: bool, generation_11_priority: dict[str, Any]
) -> tuple[str, bool]:
    if not representation_passed:
        return "public_refill_representation_defect_fix_before_search_pilot", False
    pressure = (
        int(generation_11_priority.get("edges", 0)) > 0
        and float(generation_11_priority["fraction_expected_coverage_below_0_80"]) >= 0.25
        and float(generation_11_priority["visit_weighted_mean_single_identity_miss_probability"]) > 0.10
    )
    if pressure:
        return "refill_coverage_pressure_supports_stratified_chance_search_pilot", True
    return "no_broad_refill_identity_coverage_pressure_test_bootstrap_or_targeted_chance", False


def preflight() -> dict[str, Any]:
    checks: dict[str, bool] = {
        "protocol_hash": PROTOCOL.exists() and sha256_file(PROTOCOL) == PROTOCOL_SHA256,
    }
    sources: list[dict[str, Any]] = []
    for spec in SOURCES:
        exists = spec.path.exists()
        checks[f"{spec.label}_exists"] = exists
        checks[f"{spec.label}_size"] = exists and spec.path.stat().st_size == spec.size
        checks[f"{spec.label}_hash"] = exists and sha256_file(spec.path) == spec.sha256
        sources.append(
            {
                "label": spec.label,
                "generation": spec.generation,
                "simulations": spec.simulations,
                "path": str(spec.path),
                "rows": spec.rows,
                "bytes": spec.size,
                "sha256": spec.sha256,
            }
        )
    return {
        "experiment": 136,
        "kind": "alphazero_blind_reserve_refill_coverage_audit",
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources,
        "checks": checks,
        "passed": all(checks.values()),
        "no_model_inference": True,
        "no_search": True,
        "no_training": True,
        "no_gameplay": True,
        "reads_experiment_134": False,
    }


def run_audit() -> dict[str, Any]:
    started = time.perf_counter()
    manifest = preflight()
    if not manifest["passed"]:
        raise RuntimeError("Experiment 136 preflight failed")

    blind_rows: list[dict[str, Any]] = []
    refill_rows: list[dict[str, Any]] = []
    encoding_sets: dict[str, tuple[tuple[int, ...], ...]] = {}
    seen_positions: set[tuple[int, int, int, int]] = set()
    source_accounting: dict[str, Any] = {}

    for spec in SOURCES:
        current_seed: int | None = None
        env: SplendorEnv | None = None
        state: SplendorState | None = None
        finished_seeds: set[int] = set()
        rows_read = 0
        games = 0
        with spec.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                seed = int(row["game_seed"])
                if seed != current_seed:
                    if current_seed is not None:
                        if state is None or not state.terminal:
                            raise ValueError(f"Source game {current_seed} did not end terminally")
                        finished_seeds.add(current_seed)
                    if seed in finished_seeds:
                        raise ValueError(f"Source game {seed} is not contiguous")
                    current_seed = seed
                    env = SplendorEnv(seed=seed)
                    state = env.initial_state()
                    games += 1
                assert env is not None and state is not None
                if state.turn_index != int(row["turn_index"]) or state.current_player != int(row["player_id"]):
                    raise ValueError(f"Replay reconstruction diverged at generation/seed/turn {spec.generation}/{seed}/{row['turn_index']}")
                identity = (spec.generation, seed, state.turn_index, state.current_player)
                if identity in seen_positions:
                    raise ValueError(f"Duplicate replay position {identity}")
                seen_positions.add(identity)
                actions = _validate_and_bind_actions(row, state, env, spec)
                _representation_check(row, state, encoding_sets)
                blind, refill = _analyze_row(row, state, actions, spec)
                blind_rows.append(blind)
                refill_rows.extend(refill)
                selected = int(row["selected_action_row"])
                state = env.step(state, actions[selected])
                rows_read += 1
        if current_seed is not None and (state is None or not state.terminal):
            raise ValueError(f"Final source game {current_seed} did not end terminally")
        if rows_read != spec.rows:
            raise ValueError(f"Unexpected row count for {spec.path}: {rows_read} != {spec.rows}")
        source_accounting[spec.label] = {"rows": rows_read, "games": games}

    by_generation_blind = {
        str(generation): summarize_blind(
            [row for row in blind_rows if row["generation"] == generation]
        )
        for generation in (8, 9, 10, 11)
    }
    by_phase_blind = {
        name: summarize_blind([row for row in blind_rows if row["phase"] == name])
        for name in ("early", "middle", "late")
    }
    refill_by_generation = {
        str(generation): _refill_category_summaries(
            [row for row in refill_rows if row["generation"] == generation]
        )
        for generation in (8, 9, 10, 11)
    }
    refill_by_phase = {
        name: _refill_category_summaries([row for row in refill_rows if row["phase"] == name])
        for name in ("early", "middle", "late")
    }
    refill_by_tier = {
        str(tier): _refill_category_summaries([row for row in refill_rows if row["tier"] == tier])
        for tier in (1, 2, 3)
    }
    gen11 = by_generation_blind["11"]["conditional_when_legal"]
    blind_pressure = (
        gen11["mean_visit_minus_search_mass"] >= 0.02
        and gen11["visit_top_fraction"] - gen11["network_top_fraction"] >= 0.02
    )
    decision, refill_pressure = decision_from_metrics(
        representation_passed=True,
        generation_11_priority=refill_by_generation["11"]["priority"],
    )
    result = {
        **manifest,
        "status": "complete",
        "passed": True,
        "source_accounting": source_accounting,
        "positions": len(blind_rows),
        "unique_positions": len(seen_positions),
        "representation": {
            "passed": True,
            "rows_verified": len(blind_rows),
            "distinct_public_encoding_hashes": len(encoding_sets),
            "possible_refill_set_reconstructed_exactly": True,
            "identical_encoding_collision_count": 0,
        },
        "blind_reserve": {
            "by_generation": by_generation_blind,
            "by_phase": by_phase_blind,
            "generation_sequence_is_descriptive_not_causal": True,
            "generation_11_search_pressure_flag": blind_pressure,
        },
        "refill_coverage": {
            "model": "iid_uniform_root_determinization_identity_coverage",
            "overall": _refill_category_summaries(refill_rows),
            "by_generation": refill_by_generation,
            "by_phase": refill_by_phase,
            "by_tier": refill_by_tier,
            "generation_11_material_pressure": refill_pressure,
        },
        "decision": decision,
        "training_authorized": False,
        "gameplay_authorized": False,
        "checkpoint_change_authorized": False,
        "experiment_134_changed": False,
        "automatic_continuation": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

