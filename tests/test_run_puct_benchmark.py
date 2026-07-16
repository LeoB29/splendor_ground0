import pytest

from splendor_ai.eval.run_puct_benchmark import (
    _paired_statistics,
    _summarize_search_stats,
    build_parser,
)
from splendor_ai.bots.puct_bot import PuctSearchStats


def test_search_summary_measures_variance_within_each_action() -> None:
    stats = PuctSearchStats(
        simulations=4,
        value_calls=3,
        n_step_leaf_evaluations=5,
        n_step_leaf_cache_hits=7,
        n_step_leaf_clipped_evaluations=4,
        n_step_leaf_absolute_delta_sum=1.2,
        n_step_leaf_applied_absolute_delta_sum=0.8,
        rollout_leaf_evaluations=2,
        elapsed_seconds=2.0,
        root_action_values_by_determinization={
            10: [0.1, 0.3],
            20: [0.8, 1.0],
        },
        root_prior_entropy=1.0,
        root_prior_normalized_entropy=0.5,
        root_prior_top1_probability=0.7,
        root_prior_effective_actions=2.7,
        root_visit_entropy=0.6,
        root_visit_normalized_entropy=0.3,
        root_visit_effective_actions=1.8,
        root_visited_actions=3,
    )

    summary = _summarize_search_stats([stats])

    assert summary["simulations_per_second"] == 2.0
    assert summary["value_calls"] == 3
    assert summary["n_step_leaf_evaluations"] == 5
    assert summary["n_step_leaf_cache_hits"] == 7
    assert summary["n_step_leaf_clipped_evaluations"] == 4
    assert summary["n_step_leaf_clipped_rate"] == 0.8
    assert summary["n_step_leaf_absolute_delta_sum"] == 1.2
    assert summary["n_step_leaf_applied_absolute_delta_sum"] == 0.8
    assert summary["mean_n_step_leaf_absolute_delta"] == pytest.approx(0.24)
    assert summary["mean_n_step_leaf_applied_absolute_delta"] == pytest.approx(0.16)
    assert summary["rollout_leaf_evaluations"] == 2
    assert summary["mean_root_action_determinization_std"] == pytest.approx(0.1)
    assert summary["root_actions_with_variance_samples"] == 2
    assert summary["mean_root_prior_entropy"] == 1.0
    assert summary["mean_root_prior_normalized_entropy"] == 0.5
    assert summary["mean_root_prior_top1_probability"] == 0.7
    assert summary["mean_root_visit_effective_actions"] == 1.8
    assert summary["mean_root_visited_actions"] == 3


def test_paired_statistics_use_both_seat_assignments() -> None:
    details = [
        {
            "pair_index": 0,
            "puct_seat": 0,
            "winner": 0,
            "final_scores": [15, 10],
        },
        {
            "pair_index": 0,
            "puct_seat": 1,
            "winner": 0,
            "final_scores": [16, 14],
        },
        {
            "pair_index": 1,
            "puct_seat": 0,
            "winner": None,
            "final_scores": [15, 15],
        },
        {
            "pair_index": 1,
            "puct_seat": 1,
            "winner": 1,
            "final_scores": [12, 15],
        },
    ]

    summary = _paired_statistics(
        details,
        bootstrap_seed=3,
        bootstrap_samples=1_000,
    )

    assert summary["complete_pairs"] == 2
    assert summary["mean_score_share"] == pytest.approx(0.625)
    assert summary["mean_final_score_margin"] == pytest.approx(1.5)
    assert summary["mean_score_share_ci95"][0] <= 0.625
    assert summary["mean_score_share_ci95"][1] >= 0.625


def test_parser_accepts_plain_puct_opponent_flag() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint", "policy.pt",
            "--opponent", "puct",
            "--opponent-checkpoint", "opponent.pt",
            "--output-path", "benchmark.json",
            "--opponent-plain-puct",
        ]
    )

    assert args.opponent_plain_puct is True


def test_parser_accepts_canonical_entity_policy_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint", "policy.pt",
            "--opponent", "policy",
            "--opponent-checkpoint", "parent.pt",
            "--opponent-model-kind", "canonical_entity_action_scorer",
            "--output-path", "benchmark.json",
        ]
    )

    assert args.opponent_model_kind == "canonical_entity_action_scorer"
