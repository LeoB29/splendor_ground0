import torch
import pytest

from splendor_ai.training.run_action_value_training import _spearman, compute_action_value_losses


def test_action_value_loss_prefers_aligned_returns_and_ordering() -> None:
    targets = torch.tensor([[0.8, -0.3, 0.0]])
    mask = torch.tensor([[True, True, True]])
    aligned = compute_action_value_losses(
        action_values=targets.clone(),
        target_values=targets,
        target_mask=mask,
        pairwise_loss_weight=0.5,
        minimum_target_gap=0.05,
    )
    reversed_values = compute_action_value_losses(
        action_values=-targets,
        target_values=targets,
        target_mask=mask,
        pairwise_loss_weight=0.5,
        minimum_target_gap=0.05,
    )

    assert aligned["q_loss"] < reversed_values["q_loss"]
    assert aligned["pairwise_loss"] < reversed_values["pairwise_loss"]


def test_spearman_handles_tied_action_values() -> None:
    assert _spearman([0.4, 0.4, -0.2], [0.9, 0.9, -0.1]) == pytest.approx(1.0)
    assert _spearman([0.4, 0.4, -0.2], [-0.1, -0.1, 0.9]) == pytest.approx(-1.0)
