from __future__ import annotations

import pytest

from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    ResolverGameplayConfig,
)
from splendor_ai.eval.run_resolver_target_repeatability_audit import (
    ResolverTargetRepeatabilityConfig,
    _cluster_bootstrap,
    _comparison_row,
    _comparison_summary,
    _equivalence_target_gates,
    _point_target_gates,
    _resolver_config,
    _validate_config,
    select_outcome_blind_states,
    validate_selected_root,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


def _row() -> dict[str, object]:
    return {
        "game_seed": 1,
        "turn_index": 10,
        "phase": "early",
        "actor_seat": 0,
        "gap_stratum": "clear",
    }


def _estimate(values: list[float], best: int) -> dict[str, object]:
    return {
        "action_values": values,
        "best_index": best,
    }


def test_cross_regret_distinguishes_exact_and_value_equivalent_actions() -> None:
    row = _row()
    left = _estimate([0.10, 0.09, -0.2, -0.3], 0)
    right = _estimate([0.09, 0.10, -0.2, -0.3], 1)
    comparison = _comparison_row(row, left, right)

    assert not comparison["exact_best_agreement"]
    assert comparison["value_equivalent"]
    assert comparison["left_to_right_regret"] == pytest.approx(0.01)
    assert comparison["right_to_left_regret"] == pytest.approx(0.01)
    assert comparison["symmetric_regret"] == pytest.approx(0.01)


def test_comparison_metrics_and_cluster_bootstrap_are_deterministic() -> None:
    cfg = ResolverTargetRepeatabilityConfig()
    rows = []
    for game_seed, exact in ((1, True), (1, False), (2, True)):
        row = {**_row(), "game_seed": game_seed}
        left = _estimate([0.2, 0.1, 0.0, -0.1], 0)
        right = (
            _estimate([0.21, 0.11, 0.0, -0.1], 0)
            if exact
            else _estimate([0.19, 0.20, 0.0, -0.1], 1)
        )
        rows.append(_comparison_row(row, left, right))

    summary = _comparison_summary(rows, cfg)
    first = _cluster_bootstrap(rows, cfg)
    second = _cluster_bootstrap(rows, cfg)

    assert summary["states"] == 3
    assert summary["exact_best_agreement"] == pytest.approx(2.0 / 3.0)
    assert first == second
    assert first["exact_best_agreement"]["lower"] <= first[
        "exact_best_agreement"
    ]["upper"]


def test_point_and_equivalence_gates_are_separate() -> None:
    cfg = ResolverTargetRepeatabilityConfig()
    breakdown = {
        "phase": {
            phase: {
                "exact_best_agreement": 0.75,
                "value_equivalent_agreement": 0.85,
            }
            for phase in ("early", "mid", "late")
        },
        "seat": {
            str(seat): {
                "exact_best_agreement": 0.75,
                "value_equivalent_agreement": 0.85,
            }
            for seat in (0, 1)
        },
        "gap_stratum": {
            "close": {
                "exact_best_agreement": 0.65,
                "value_equivalent_agreement": 0.90,
            },
            "medium": {
                "exact_best_agreement": 0.70,
                "value_equivalent_agreement": 0.85,
            },
            "clear": {
                "exact_best_agreement": 0.85,
                "value_equivalent_agreement": 0.90,
            },
        },
    }
    summary = {
        "structural": {"both_completed_fraction": 0.90},
        "fresh_a_vs_b": {
            "exact_best_agreement": 0.75,
            "flattened_value_pearson": 0.80,
            "mean_within_state_spearman": 0.70,
            "mean_symmetric_cross_regret": 0.02,
            "p90_symmetric_cross_regret": 0.05,
            "directional_regret_at_most_005": 0.90,
            "value_equivalent_agreement": 0.85,
            "breakdowns": breakdown,
            "bootstrap": {
                "exact_best_agreement": {"lower": 0.62, "upper": 0.85},
                "value_equivalent_agreement": {"lower": 0.72, "upper": 0.93},
                "mean_regret": {"lower": 0.01, "upper": 0.03},
            },
        },
    }

    assert all(_point_target_gates(cfg, summary).values())
    assert all(_equivalence_target_gates(cfg, summary).values())
    summary["fresh_a_vs_b"]["exact_best_agreement"] = 0.60
    assert not all(_point_target_gates(cfg, summary).values())
    assert all(_equivalence_target_gates(cfg, summary).values())


def test_replicates_change_only_frozen_random_namespaces() -> None:
    cfg = ResolverTargetRepeatabilityConfig()
    base = ResolverGameplayConfig()
    replicate_a = _resolver_config(cfg, "A")
    replicate_b = _resolver_config(cfg, "B")

    assert replicate_a.belief_root_index_start != base.belief_root_index_start
    assert replicate_b.belief_root_index_start != replicate_a.belief_root_index_start
    assert replicate_a.continuation_seed_start != base.continuation_seed_start
    assert replicate_b.continuation_seed_start != replicate_a.continuation_seed_start
    for field in base.__dataclass_fields__:
        if field in {"belief_root_index_start", "continuation_seed_start"}:
            continue
        assert getattr(replicate_a, field) == getattr(base, field)
        assert getattr(replicate_b, field) == getattr(base, field)


def test_055_frozen_selection_is_balanced_outcome_blind_and_root_aligned() -> None:
    cfg = ResolverTargetRepeatabilityConfig()
    _validate_config(cfg)
    selection = select_outcome_blind_states(cfg)

    assert selection["selected_count"] == 72
    assert selection["unique_states"] == 72
    assert selection["source_games"] >= 18
    assert not selection["selection_used_outcomes"]
    assert len(selection["selected_cell_counts"]) == 18
    assert set(selection["selected_cell_counts"].values()) == {4}
    assert all(
        "winner" not in sample and "outcome" not in sample
        for sample in selection["selected"]
    )

    scorer = _LegalActionScorer(
        cfg.parent_checkpoint,
        cfg.device,
        "legal_action_attention_scorer",
    )
    prepared = validate_selected_root(selection["selected"][0], scorer)

    assert prepared["unique_legal_action_indices"]
    assert prepared["root_keys"] == selection["selected"][0]["original"][
        "root_keys"
    ]
