import math
import random

import pytest
import torch

from splendor_ai.training.run_from_scratch_resolver_bc_pilot import (
    normalized_masked_entropy,
    shuffled_train_game_prefixes,
    summarize_records,
)


def test_normalized_masked_entropy_is_width_normalized() -> None:
    scores = torch.zeros((3, 4), dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    entropy = normalized_masked_entropy(scores, mask)
    assert entropy.tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_normalized_masked_entropy_preserves_gradients() -> None:
    scores = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    entropy = normalized_masked_entropy(
        scores, torch.tensor([[True, True, False]])
    ).mean()
    entropy.backward()
    assert scores.grad is not None
    assert all(math.isfinite(value) for value in scores.grad.tolist()[0])
    assert scores.grad[0, 2] == 0.0


def test_training_game_prefixes_are_frozen_nested_and_not_sorted() -> None:
    prefixes = shuffled_train_game_prefixes(range(610_100, 610_180))
    expected = list(range(610_100, 610_180))
    random.Random(75_075).shuffle(expected)
    assert prefixes[80] == tuple(expected)
    assert prefixes[20] == prefixes[40][:20]
    assert prefixes[40] == prefixes[80][:40]
    assert list(prefixes[80]) != sorted(prefixes[80])


def test_training_game_prefixes_require_exact_source_game_count() -> None:
    with pytest.raises(ValueError, match="exactly 80"):
        shuffled_train_game_prefixes(range(79))


def test_summary_uses_game_weighted_agreement_and_reports_dose() -> None:
    records = [
        {
            "game_seed": 1,
            "phase": "early",
            "seat": 0,
            "outcome": "winner",
            "candidate_correct": True,
            "parent_correct": False,
            "moved_from_parent": True,
            "nll": 0.2,
            "normalized_entropy": 0.4,
        },
        {
            "game_seed": 1,
            "phase": "early",
            "seat": 0,
            "outcome": "winner",
            "candidate_correct": False,
            "parent_correct": False,
            "moved_from_parent": False,
            "nll": 0.4,
            "normalized_entropy": 0.6,
        },
        {
            "game_seed": 2,
            "phase": "late",
            "seat": 1,
            "outcome": "loser",
            "candidate_correct": True,
            "parent_correct": True,
            "moved_from_parent": False,
            "nll": 0.1,
            "normalized_entropy": 0.8,
        },
    ]
    summary = summarize_records(records)
    assert summary["overall"]["agreement"] == pytest.approx(2 / 3)
    assert summary["overall"]["game_weighted_agreement"] == pytest.approx(0.75)
    assert summary["overall"]["parent_game_weighted_agreement"] == pytest.approx(0.5)
    assert summary["overall"]["game_weighted_agreement_delta"] == pytest.approx(0.25)
    assert summary["overall"]["movement_from_parent"] == pytest.approx(1 / 3)
    assert set(summary["by_phase"]) == {"early", "late"}
