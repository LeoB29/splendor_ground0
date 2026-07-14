import torch

from splendor_ai.training.run_root_visit_correction import (
    compute_root_visit_correction_loss,
)


def test_root_visit_correction_uses_soft_target_and_zero_anchor_kl_for_parent_scores() -> None:
    target_mask = torch.tensor([[True, True, True]])
    root_target = torch.tensor([[0.2, 0.3, 0.5]])
    target_scores = torch.log(root_target)
    anchor_mask = torch.tensor([[True, True]])
    anchor_scores = torch.tensor([[0.3, -0.2]])

    losses = compute_root_visit_correction_loss(
        target_scores=target_scores,
        target_mask=target_mask,
        root_visit_target=root_target,
        root_visit_mask=torch.tensor([True]),
        anchor_scores=anchor_scores,
        parent_anchor_scores=anchor_scores.detach().clone(),
        anchor_mask=anchor_mask,
        anchor_kl_weight=2.0,
    )

    expected_entropy = -(root_target * root_target.log()).sum()
    assert torch.isclose(losses["root_visit_loss"], expected_entropy, atol=1e-6)
    assert torch.isclose(losses["anchor_kl"], torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(losses["total_loss"], expected_entropy, atol=1e-6)
