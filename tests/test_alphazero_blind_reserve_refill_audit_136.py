from __future__ import annotations

import math

from splendor_ai.alphazero.blind_reserve_refill_audit import (
    decision_from_metrics,
    expected_identity_coverage,
    possible_refill_sets_from_encoding,
    possible_refill_sets_from_state,
    summarize_blind,
    summarize_refill,
)
from splendor_ai.encoding import encode_canonical_state_entities
from splendor_ai.engine.env import SplendorEnv


def test_expected_identity_coverage_matches_closed_form() -> None:
    coverage, miss = expected_identity_coverage(10, 20)
    assert math.isclose(miss, 0.95**10)
    assert math.isclose(coverage, 1.0 - 0.95**10)


def test_initial_public_encoding_reconstructs_refill_sets() -> None:
    state = SplendorEnv(seed=136).initial_state()
    encoded = encode_canonical_state_entities(state, 0)
    assert possible_refill_sets_from_encoding(encoded.entity_type_ids, encoded.card_ids) == possible_refill_sets_from_state(state, 0)


def test_blind_summary_separates_network_noise_and_search() -> None:
    row = {
        "reserve_legal": True,
        "network_mass": 0.10,
        "search_mass": 0.20,
        "visit_mass": 0.35,
        "network_top": False,
        "search_top": False,
        "visit_top": True,
        "selected": True,
        "raw_to_visit_promoted": True,
        "raw_to_visit_demoted": False,
        "search_to_visit_promoted": True,
        "search_to_visit_demoted": False,
        "best_visited_q_gap": 0.1,
    }
    summary = summarize_blind([row])
    conditional = summary["conditional_when_legal"]
    assert math.isclose(conditional["mean_search_minus_network_mass"], 0.10)
    assert math.isclose(conditional["mean_visit_minus_search_mass"], 0.15)
    assert conditional["raw_to_visit_promoted"] == 1


def test_refill_summary_uses_visit_weighted_miss_probability() -> None:
    rows = [
        {
            "generation": 11, "game_seed": 1, "turn_index": 2,
            "possible_identities": 20, "visits": 10, "visits_per_identity": 0.5,
            "expected_coverage": 0.5, "single_identity_miss_probability": 0.5,
        },
        {
            "generation": 11, "game_seed": 2, "turn_index": 2,
            "possible_identities": 20, "visits": 30, "visits_per_identity": 1.5,
            "expected_coverage": 0.9, "single_identity_miss_probability": 0.1,
        },
    ]
    summary = summarize_refill(rows)
    assert math.isclose(summary["visit_weighted_mean_single_identity_miss_probability"], 0.2)


def test_decision_requires_both_refill_pressure_gates() -> None:
    pressured = {
        "edges": 10,
        "fraction_expected_coverage_below_0_80": 0.30,
        "visit_weighted_mean_single_identity_miss_probability": 0.11,
    }
    decision, flag = decision_from_metrics(
        representation_passed=True, generation_11_priority=pressured
    )
    assert flag
    assert decision == "refill_coverage_pressure_supports_stratified_chance_search_pilot"
    pressured["visit_weighted_mean_single_identity_miss_probability"] = 0.09
    decision, flag = decision_from_metrics(
        representation_passed=True, generation_11_priority=pressured
    )
    assert not flag
    assert decision.startswith("no_broad_refill_identity_coverage_pressure")
