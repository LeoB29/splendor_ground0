from __future__ import annotations

import pytest

from splendor_ai.alphazero.replay_consolidation import (
    NEWEST_OBJECTIVE_TOLERANCE,
    consolidation_decision,
    consolidation_training_config,
    retention_assessment,
)


def _training(deltas: dict[int, float], aggregate: float = -0.01) -> dict[str, object]:
    return {
        "validation_games": 205,
        "validation_rows": 12_500,
        "validation_loss_delta_child_minus_parent": {"objective_loss": aggregate},
        "validation_slices": {
            f"replay_generation_{generation}": {
                "rows": 3_000,
                "parent": {"objective_loss": 2.5},
                "child": {"objective_loss": 2.5 + deltas[generation]},
            }
            for generation in range(7, 11)
        },
    }


def test_107_training_changes_only_window_and_seed_from_proven_recipe() -> None:
    config = consolidation_training_config()
    assert config.epochs == 12
    assert config.batch_size == 256
    assert config.learning_rate == 1e-4
    assert config.value_loss_weight == 0.5
    assert config.early_stopping_patience == 3
    assert config.early_stopping_min_delta == 0.001
    assert config.bucket_batches is True
    assert config.policy_divergence_weight == config.visit_argmax_aux_weight == 0.0


def test_retention_gate_requires_old_recovery_and_newest_preservation() -> None:
    passed = retention_assessment(
        _training({7: -0.02, 8: -0.01, 9: -0.01, 10: NEWEST_OBJECTIVE_TOLERANCE})
    )
    assert passed["passed"]
    old_failure = retention_assessment(
        _training({7: 0.001, 8: -0.01, 9: -0.01, 10: -0.01})
    )
    assert not old_failure["passed"]
    newest_failure = retention_assessment(
        _training({7: -0.01, 8: -0.01, 9: -0.01, 10: 0.006})
    )
    assert not newest_failure["passed"]


def test_retention_gate_rejects_missing_or_empty_corpus() -> None:
    missing = _training({7: -0.01, 8: -0.01, 9: -0.01, 10: -0.01})
    del missing["validation_slices"]["replay_generation_8"]  # type: ignore[index]
    with pytest.raises(ValueError, match="lack validation"):
        retention_assessment(missing)
    empty = _training({7: -0.01, 8: -0.01, 9: -0.01, 10: -0.01})
    empty["validation_slices"]["replay_generation_8"]["rows"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="no held-out rows"):
        retention_assessment(empty)


def test_consolidation_decision_preserves_stage_gates() -> None:
    assert "retention_gate_failed" in consolidation_decision(
        retention_passed=False, internal_passed=None, external_progress=None
    )
    assert "not_internally_stronger" in consolidation_decision(
        retention_passed=True, internal_passed=False, external_progress=None
    )
    assert "confirmed" in consolidation_decision(
        retention_passed=True,
        internal_passed=True,
        external_progress={"ci95_low": 0.01, "mean_difference": 0.03},
    )
    assert "directional" in consolidation_decision(
        retention_passed=True,
        internal_passed=True,
        external_progress={"ci95_low": -0.01, "mean_difference": 0.03},
    )
    assert "no_external_progress" in consolidation_decision(
        retention_passed=True,
        internal_passed=True,
        external_progress={"ci95_low": -0.03, "mean_difference": 0.0},
    )


def test_internal_pass_requires_external_comparison_for_final_decision() -> None:
    with pytest.raises(ValueError, match="requires an external"):
        consolidation_decision(
            retention_passed=True, internal_passed=True, external_progress=None
        )
