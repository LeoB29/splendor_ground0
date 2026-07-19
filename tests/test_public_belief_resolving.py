from __future__ import annotations

from collections import Counter

import pytest

from splendor_ai.cfr import (
    ResolvedLeaf,
    deterministic_belief_range,
    resolve_two_ply,
)
from splendor_ai.diagnostics import state_signature
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_bounded_public_belief_resolving import assigned_stratum
from splendor_ai.eval.run_bounded_public_belief_resolving_replication import (
    GATE_THRESHOLDS,
    SOURCE_BANDS,
    ReplicationConfig,
    _frozen_resolver_contract_matches_049,
    _source_collection_gate,
)


def test_resolver_shares_opponent_response_across_indistinguishable_histories() -> None:
    root = ("root",)
    info = ("same-opponent-information",)
    action_x = ("x",)
    action_y = ("y",)
    leaves = [
        ResolvedLeaf(0, root, info, action_x, 0.0, 1.0),
        ResolvedLeaf(0, root, info, action_y, 4.0, 3.0),
        ResolvedLeaf(1, root, info, action_x, 10.0, 9.0),
        ResolvedLeaf(1, root, info, action_y, 4.0, 3.0),
    ]

    resolution = resolve_two_ply(
        leaves,
        (root,),
        determinization_count=2,
        payoff="discovery_payoff",
    )

    # A hidden-state-aware opponent would choose x in det0 and y in det1 for
    # value 2.0. The legal shared response is y in both histories, value 4.0.
    assert resolution.action_values == pytest.approx((4.0,))
    assert resolution.opponent_choices[0]["selected_action"] == repr(action_y)


def test_resolver_rejects_unstable_actions_inside_opponent_information_set() -> None:
    root = ("root",)
    info = ("same",)
    leaves = [
        ResolvedLeaf(0, root, info, ("x",), 0.0, 0.0),
        ResolvedLeaf(1, root, info, ("y",), 0.0, 0.0),
    ]

    with pytest.raises(ValueError, match="unstable semantic actions"):
        resolve_two_ply(leaves, (root,), 2, payoff="discovery_payoff")


def test_belief_range_ignores_incidental_hidden_deck_order() -> None:
    state = SplendorEnv(seed=49).initial_state()
    permuted = state.copy_shallow()
    for tier in (1, 2, 3):
        permuted.hidden_tier_decks[tier].reverse()

    first = deterministic_belief_range(state, 0, root_index=7)
    second = deterministic_belief_range(permuted, 0, root_index=7)

    assert [state_signature(item) for item in first] == [state_signature(item) for item in second]


def test_source_assignment_is_exactly_balanced_before_collection() -> None:
    counts = Counter(assigned_stratum(index) for index in range(192))

    assert len(counts) == 6
    assert set(counts.values()) == {32}


def test_replication_source_bands_are_disjoint_and_preregistered() -> None:
    seeds = [set(range(start, end + 1)) for start, end in SOURCE_BANDS.values()]

    assert len(SOURCE_BANDS) == 6
    assert all(len(band) == 1_000 for band in seeds)
    assert all(not left.intersection(right) for index, left in enumerate(seeds) for right in seeds[index + 1 :])
    assert min(set.union(*seeds)) == 491_000
    assert max(set.union(*seeds)) == 496_999


def test_replication_freezes_049_resolver_and_leaf_contract() -> None:
    cfg = ReplicationConfig()

    assert _frozen_resolver_contract_matches_049(cfg)
    assert cfg.continuation_seed_start == 52_000_000
    assert GATE_THRESHOLDS == {
        "max_loop_adjudication_fraction": 0.001,
        "min_leaf_pearson": 0.60,
        "max_leaf_mae": 0.10,
        "min_leaf_sign_agreement": 0.70,
        "min_action_value_pearson": 0.60,
        "min_best_action_agreement": 0.60,
        "min_effect_sign_eligible": 24,
        "min_effect_sign_agreement": 0.70,
        "min_changed_roots": 8,
        "min_resolved_fraction": 0.25,
        "min_stratum_changed": 1,
        "min_stratum_mean_gain": -0.025,
    }


def test_replication_source_gate_accepts_recorded_skips_but_not_missing_admissions() -> None:
    cfg = ReplicationConfig()
    keys = [f"seat{seat}_{phase}" for seat in (0, 1) for phase in ("early", "mid", "late")]
    source = {
        "collection_complete": True,
        "admitted_games": 192,
        "invalid_admitted_games": 0,
        "admitted_stratum_counts": {key: 32 for key in keys},
        "invalid_attempted_games": 2,
    }
    selection = {
        "selected_count": 48,
        "unique_source_games": 48,
        "stratum_counts": {key: 8 for key in keys},
    }

    assert _source_collection_gate(cfg, source, selection)
    source["admitted_stratum_counts"]["seat1_late"] = 31
    assert not _source_collection_gate(cfg, source, selection)
