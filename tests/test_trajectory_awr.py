from __future__ import annotations

import torch

from splendor_ai.training.run_trajectory_awr import compute_awr_loss
from splendor_ai.training.run_trajectory_policy_gradient import _exploration_probabilities


def test_awr_loss_upweights_positive_advantage_trajectory() -> None:
    scores = torch.tensor([[2.0, 0.0], [2.0, 0.0]], requires_grad=True)
    reference = scores.detach().clone()
    mask = torch.ones((2, 2), dtype=torch.bool)
    targets = torch.tensor([0, 1])
    _, behavior = _exploration_probabilities(
        reference, reference, mask, 0.5, 1.0, 2
    )
    behavior_logs = behavior[torch.arange(2), targets].log()

    favorable_second = compute_awr_loss(
        action_scores=scores,
        reference_scores=reference,
        legal_action_mask=mask,
        target_action_row=targets,
        advantages=torch.tensor([-0.5, 0.5]),
        sample_weight=torch.ones(2),
        behavior_log_probability=behavior_logs,
        advantage_temperature=0.25,
        max_advantage_weight=10.0,
        kl_weight=1.0,
        entropy_weight=0.0,
        sampling_epsilon=0.5,
        sampling_temperature=1.0,
        sampling_top_k=2,
    )
    favorable_first = compute_awr_loss(
        action_scores=scores,
        reference_scores=reference,
        legal_action_mask=mask,
        target_action_row=targets,
        advantages=torch.tensor([0.5, -0.5]),
        sample_weight=torch.ones(2),
        behavior_log_probability=behavior_logs,
        advantage_temperature=0.25,
        max_advantage_weight=10.0,
        kl_weight=1.0,
        entropy_weight=0.0,
        sampling_epsilon=0.5,
        sampling_temperature=1.0,
        sampling_top_k=2,
    )

    assert favorable_second["policy_loss"] > favorable_first["policy_loss"]
    assert torch.isclose(favorable_second["behavior_log_prob_mae"], torch.tensor(0.0))
    favorable_second["total_loss"].backward()
    assert torch.isfinite(scores.grad).all()

