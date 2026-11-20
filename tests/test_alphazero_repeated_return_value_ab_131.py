from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from splendor_ai.alphazero.network import load_checkpoint
from splendor_ai.alphazero.repeated_return_value_ab import (
    protected_qualification,
    repeated_return_decision,
    select_repeated_return_states,
    split_for_block_index,
    value_only_scope,
)


SOURCES = (
    Path("outputs/alphazero_scaled_loop_099/selfplay/generation_002.jsonl"),
    Path("outputs/alphazero_scaled_continuation_103/selfplay/generation_005_sanitized.jsonl"),
    Path("outputs/alphazero_continued_scaling_104/selfplay/generation_008_sanitized.jsonl"),
    Path("outputs/alphazero_high_budget_distillation_114/partition/train_224.jsonl"),
)
PARENT = Path("outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt")


def test_state_schedule_is_exact_balanced_and_disjoint() -> None:
    states, summaries = select_repeated_return_states(SOURCES)
    assert len(states) == 300
    assert len(summaries) == 4
    assert Counter(row["identity"]["split"] for row in states) == Counter(
        {"fitting": 192, "development": 48, "protected": 60}
    )
    assert Counter(row["identity"]["phase"] for row in states) == Counter(
        {"early": 100, "middle": 100, "late": 100}
    )
    assert all(row["identity"]["source_game_ordinal"] >= 112 for row in states)
    assert len({(row["identity"]["source_index"], row["identity"]["source_game_ordinal"]) for row in states}) == 300
    by_index = {row["identity"]["state_index"]: row["identity"] for row in states}
    assert by_index[48] == {
        "state_index": 48,
        "source_index": 0,
        "source_generation": 2,
        "source_game_ordinal": 160,
        "source_game_seed": 1_700_160,
        "source_turn_index": 16,
        "root_player": 0,
        "phase": "early",
        "split": "development",
        "source_progress": pytest.approx(0.24615384615384617),
        "pending_round_end": False,
    }
    assert by_index[294] == {
        "state_index": 294,
        "source_index": 3,
        "source_generation": 11,
        "source_game_ordinal": 181,
        "source_game_seed": 2_120_013,
        "source_turn_index": 16,
        "root_player": 0,
        "phase": "early",
        "split": "protected",
        "source_progress": pytest.approx(0.24615384615384617),
        "pending_round_end": False,
    }


def test_cell_split_is_sixteen_fit_four_development() -> None:
    assert [split_for_block_index(index) for index in range(0, 60, 3)].count("fitting") == 16
    assert [split_for_block_index(index) for index in range(0, 60, 3)].count("development") == 4


def test_value_only_scope_is_exact() -> None:
    model, _checkpoint = load_checkpoint(PARENT, device="cpu")
    assert value_only_scope(model) == {
        "total_parameters": 356_450,
        "trainable_parameters": 4_705,
        "frozen_parameters": 351_745,
    }
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
        "value_head.0.weight",
        "value_head.0.bias",
        "value_head.2.weight",
        "value_head.2.bias",
    }


@pytest.mark.parametrize(
    ("offline", "control", "parent", "expected"),
    [
        (False, None, None, "repeated_return_value_target_not_causally_better_no_gameplay"),
        (True, False, None, "repeated_return_value_offline_gain_not_search_causal"),
        (True, True, False, "repeated_return_value_gain_not_parent_robust"),
        (True, True, True, "repeated_return_value_causal_authorize_joint_reanalysis_design_no_install"),
    ],
)
def test_decision_ladder(offline: bool, control: bool | None, parent: bool | None, expected: str) -> None:
    assert repeated_return_decision(
        offline_passed=offline,
        control_gameplay_passed=control,
        parent_gameplay_passed=parent,
    ) == expected


def test_protected_gate_requires_causal_mse_and_bootstrap_gain() -> None:
    metrics = {
        "generation_11": {"mse": 0.30, "mae": 0.40, "pearson": 0.2},
        "single_outcome_control": {
            "mse": 0.25,
            "mae": 0.35,
            "pearson": 0.3,
            "phase_breakdown": {phase: {"mse": 0.25} for phase in ("early", "middle", "late")},
        },
        "repeated_mean_treatment": {
            "mse": 0.20,
            "mae": 0.30,
            "pearson": 0.4,
            "phase_breakdown": {phase: {"mse": 0.20} for phase in ("early", "middle", "late")},
        },
    }
    training = {
        "single_outcome_control": {"selected_epoch": 1},
        "repeated_mean_treatment": {"selected_epoch": 1},
    }
    bootstrap = {
        "treatment_minus_control_squared_error_ci95": [-0.10, -0.01],
        "treatment_minus_parent_squared_error_ci95": [-0.15, -0.02],
    }
    assert protected_qualification(
        training=training,
        metrics=metrics,
        bootstrap=bootstrap,
        integrity={"passed": True},
    )["qualified"]
