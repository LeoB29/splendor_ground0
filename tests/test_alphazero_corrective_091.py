from __future__ import annotations

from pathlib import Path

import pytest
import torch

from splendor_ai.alphazero.arena import (
    alpha_zero_raw_factory,
    alpha_zero_search_factory,
)
from splendor_ai.alphazero.corrective import (
    candidate_arena_acceptance,
    candidate_offline_eligibility,
    search_confirmation_acceptance,
    search_discovery_eligibility,
)
from splendor_ai.alphazero.network import create_initial_checkpoint
from splendor_ai.alphazero.search import SearchConfig
from splendor_ai.alphazero.training import TrainingConfig, alpha_zero_loss


def _training_result(
    *,
    objective_delta: float = -0.1,
    policy_delta: float = -0.02,
    value_delta: float = 0.01,
    parent_top: float = 0.4,
    child_top: float = 0.4,
) -> dict[str, object]:
    return {
        "validation_loss_delta_child_minus_parent": {
            "objective_loss": objective_delta,
            "policy_loss": policy_delta,
            "value_loss": value_delta,
        },
        "parent_validation_metrics": {
            "policy_top_target_agreement": parent_top,
        },
        "child_validation_metrics": {
            "policy_top_target_agreement": child_top,
        },
    }


def _arena(
    name: str,
    *,
    score_share: float = 0.5,
    mean_margin: float = 0.0,
    completed: int = 16,
) -> dict[str, object]:
    return {
        "name": name,
        "games": 16,
        "score_share": score_share,
        "mean_margin": mean_margin,
        "termination_reasons": {"completed": completed},
    }


def test_offline_candidate_requires_every_frozen_condition() -> None:
    assert candidate_offline_eligibility(_training_result())["eligible"] is True
    assert (
        candidate_offline_eligibility(
            _training_result(value_delta=0.0250001)
        )["eligible"]
        is False
    )
    assert (
        candidate_offline_eligibility(
            _training_result(parent_top=0.4, child_top=0.3899)
        )["eligible"]
        is False
    )


def test_candidate_arena_gate_accepts_boundaries_and_rejects_cutoff() -> None:
    passing = [_arena("raw"), _arena("search")]
    assert candidate_arena_acceptance(passing)["accepted"] is True
    cutoff = [_arena("raw"), _arena("search", completed=15)]
    assert candidate_arena_acceptance(cutoff)["accepted"] is False
    negative = [_arena("raw"), _arena("search", mean_margin=-0.01)]
    assert candidate_arena_acceptance(negative)["accepted"] is False


def test_search_gates_enforce_discovery_and_confirmation_thresholds() -> None:
    assert search_discovery_eligibility(
        _arena("discovery", score_share=0.625, mean_margin=0.01)
    )["eligible"]
    assert not search_discovery_eligibility(
        _arena("discovery", score_share=0.6249, mean_margin=1.0)
    )["eligible"]
    assert search_confirmation_acceptance(
        _arena("confirmation", score_share=0.5625, mean_margin=0.01)
    )["accepted"]
    assert not search_confirmation_acceptance(
        _arena("confirmation", score_share=0.5625, mean_margin=0.0)
    )["accepted"]


def test_weighted_objective_preserves_unweighted_diagnostics() -> None:
    losses = alpha_zero_loss(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([[True, True]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        value_weight=0.25,
    )
    assert losses["policy_loss"].item() == pytest.approx(0.693147, rel=1e-5)
    assert losses["value_loss"].item() == pytest.approx(1.0)
    assert losses["total_loss"].item() == pytest.approx(1.693147, rel=1e-5)
    assert losses["objective_loss"].item() == pytest.approx(0.943147, rel=1e-5)


def test_corrective_training_config_validation() -> None:
    TrainingConfig(
        value_loss_weight=0.25,
        early_stopping_patience=2,
        early_stopping_min_delta=0.002,
        bucket_batches=True,
    ).validate()
    with pytest.raises(ValueError):
        TrainingConfig(value_loss_weight=-0.1).validate()
    with pytest.raises(ValueError):
        TrainingConfig(early_stopping_patience=0).validate()


def test_arena_factories_share_inference_but_search_state_is_fresh(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "generation_000.pt"
    create_initial_checkpoint(checkpoint, seed=91)
    raw_factory = alpha_zero_raw_factory(checkpoint, "cpu")
    raw_a = raw_factory(1, 0)
    raw_b = raw_factory(2, 1)
    assert raw_a.inference is raw_b.inference

    search_factory = alpha_zero_search_factory(
        checkpoint,
        "cpu",
        SearchConfig(simulations=2, mode="debug"),
    )
    search_a = search_factory(1, 0)
    search_b = search_factory(2, 1)
    assert search_a.inference is search_b.inference
    assert search_a.search is not search_b.search
