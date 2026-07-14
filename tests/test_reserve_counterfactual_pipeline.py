import json
from pathlib import Path

import torch

from splendor_ai.training.run_reserve_counterfactual_corpus import _split_rows
from splendor_ai.training.run_reserve_counterfactual_training import (
    _ranks,
    _spearman,
    compute_counterfactual_hard_top_loss,
    compute_counterfactual_soft_loss,
)


def test_game_split_is_balanced_and_disjoint() -> None:
    rows = []
    for band_index, band in enumerate(("0-4", "5-9", "10-14", "15-19")):
        for seed in range(10):
            rows.append({"game_seed": band_index * 100 + seed, "turn_band": band})
    train, heldout, split = _split_rows(rows, heldout_games_per_band=2, seed=7)
    assert len(train) == 32
    assert len(heldout) == 8
    assert split["heldout_by_band"] == {"0-4": 2, "5-9": 2, "10-14": 2, "15-19": 2}
    assert {row["game_seed"] for row in train}.isdisjoint({row["game_seed"] for row in heldout})


def test_counterfactual_soft_loss_prefers_return_aligned_scores() -> None:
    candidate_mask = torch.tensor([[True, True, False]])
    returns = torch.tensor([[1.0, 0.0, 0.0]])
    aligned = compute_counterfactual_soft_loss(
        torch.tensor([[2.0, -2.0, 0.0]]), candidate_mask, returns, 0.35
    )
    reversed_scores = compute_counterfactual_soft_loss(
        torch.tensor([[-2.0, 2.0, 0.0]]), candidate_mask, returns, 0.35
    )
    assert aligned < reversed_scores


def test_hard_top_loss_prefers_highest_return_action() -> None:
    candidate_mask = torch.tensor([[True, True, False]])
    returns = torch.tensor([[1.0, 0.0, 0.0]])
    aligned = compute_counterfactual_hard_top_loss(
        torch.tensor([[2.0, -2.0, 0.0]]), candidate_mask, returns
    )
    reversed_scores = compute_counterfactual_hard_top_loss(
        torch.tensor([[-2.0, 2.0, 0.0]]), candidate_mask, returns
    )
    assert aligned < reversed_scores


def test_rank_helpers_handle_order_and_ties() -> None:
    assert _spearman([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == 1.0
    assert _spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert _ranks([1.0, 1.0, 2.0]) == [0.5, 0.5, 2.0]
