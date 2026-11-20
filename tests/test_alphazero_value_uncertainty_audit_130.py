from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from splendor_ai.alphazero.value_uncertainty_audit import (
    bootstrap_uncertainty,
    select_audit_states,
    target_noise_decomposition,
    two_way_sum_squares,
    value_uncertainty_decision,
)


HOLDOUT = Path("outputs/alphazero_high_budget_distillation_114/partition/holdout_32.jsonl")


def test_select_audit_states_uses_exact_balanced_first_24_holdout_games() -> None:
    states = select_audit_states(HOLDOUT)
    assert len(states) == 24
    assert [row["identity"]["source_game_ordinal"] for row in states] == list(range(24))
    assert Counter(row["identity"]["phase"] for row in states) == Counter(
        {"early": 8, "middle": 8, "late": 8}
    )
    assert all(not row["identity"]["pending_round_end"] for row in states)


def test_two_way_decomposition_attributes_pure_world_effect() -> None:
    result = two_way_sum_squares([[-1.0, -1.0], [1.0, 1.0]])
    assert result["possible_world_fraction"] == pytest.approx(1.0)
    assert result["policy_seed_fraction"] == pytest.approx(0.0)
    assert result["interaction_fraction"] == pytest.approx(0.0)


def test_two_way_decomposition_attributes_pure_policy_effect() -> None:
    result = two_way_sum_squares([[-1.0, 1.0], [-1.0, 1.0]])
    assert result["possible_world_fraction"] == pytest.approx(0.0)
    assert result["policy_seed_fraction"] == pytest.approx(1.0)


def test_target_noise_decomposition_separates_within_and_between_state_signal() -> None:
    noisy = target_noise_decomposition([[-1.0, 1.0], [-1.0, 1.0]])
    separated = target_noise_decomposition([[-1.0, -1.0], [1.0, 1.0]])
    assert noisy["within_state_target_noise_fraction"] == pytest.approx(1.0)
    assert separated["within_state_target_noise_fraction"] == pytest.approx(0.0)


def test_bootstrap_uncertainty_is_deterministic_and_finite() -> None:
    groups = [[-1.0, 1.0] * 8, [-1.0] * 8 + [1.0] * 8]
    first = bootstrap_uncertainty(groups, samples=200, seed=7)
    second = bootstrap_uncertainty(groups, samples=200, seed=7)
    assert first == second
    assert first["mean_within_state_wdl_variance_ci95"] == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    ("integrity", "material", "world", "expected"),
    [
        (False, True, True, "value_uncertainty_integrity_failed_no_model_conclusion"),
        (True, False, True, "scalar_terminal_value_uncertainty_not_primary_close_compact_alphazero"),
        (True, True, False, "trajectory_policy_value_uncertainty_material_review_joint_target"),
        (True, True, True, "possible_world_value_distribution_material_freeze_distributional_pilot"),
    ],
)
def test_decision_ladder(integrity: bool, material: bool, world: bool, expected: str) -> None:
    assert value_uncertainty_decision(
        integrity=integrity, material_noise=material, possible_world_primary=world
    ) == expected
