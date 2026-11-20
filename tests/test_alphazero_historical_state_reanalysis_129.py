from __future__ import annotations

import pytest

from splendor_ai.alphazero.historical_state_reanalysis import (
    CONTROL_ROWS,
    PHASES,
    _arena_from_rows,
    _combine_historical_shards,
    nearest_eligible_index,
    phase_for_slot,
    reanalysis_decision,
    reanalysis_offline_qualification,
    reanalysis_training_config,
    select_anchor_game_count,
)


def test_phase_schedule_is_exactly_balanced_per_source() -> None:
    phases = [phase_for_slot(slot) for slot in range(64)]
    assert phases.count("early") == 22
    assert phases.count("middle") == 21
    assert phases.count("late") == 21
    assert tuple(phases[:3]) == PHASES


def test_frozen_source_ranges_fit_both_224_and_256_game_sources() -> None:
    assert 111 < 224 < 256


def test_nearest_eligible_state_avoids_round_end_and_ties_early() -> None:
    pending = [False] * 9
    pending[2] = True
    assert nearest_eligible_index(pending, "early") == 1
    assert nearest_eligible_index(pending, "middle") == 4
    assert nearest_eligible_index(pending, "late") == 6


def test_anchor_selection_matches_control_with_whole_games() -> None:
    selection = select_anchor_game_count([60] * 224, historical_rows=8_000)
    assert selection["selected_games"] == 99
    assert selection["combined_rows"] == 13_940
    assert selection["absolute_row_difference"] <= selection["maximum_selected_game_rows"]
    assert selection["control_rows"] == CONTROL_ROWS


def test_training_recipe_preserves_standard_objective() -> None:
    config = reanalysis_training_config()
    assert config.seed == 1_129_000
    assert config.epochs == 12
    assert config.batch_size == 256
    assert config.policy_loss_weight == 1.0
    assert config.value_loss_weight == 0.5
    assert config.validation_fraction == 0.2


def test_offline_gate_requires_parent_gain_and_control_retention() -> None:
    metrics = {
        "generation_11": {"policy_loss": 2.0, "value_loss": 0.90, "objective_loss": 2.45},
        "terminal_control_114": {"policy_loss": 1.96, "value_loss": 0.89, "objective_loss": 2.405},
        "historical_reanalysis_candidate": {"policy_loss": 1.965, "value_loss": 0.90, "objective_loss": 2.415},
    }
    assert reanalysis_offline_qualification(training={"selected_epoch": 1}, metrics=metrics)["qualified"]
    bad = {name: dict(values) for name, values in metrics.items()}
    bad["historical_reanalysis_candidate"]["policy_loss"] = 1.971
    assert not reanalysis_offline_qualification(training={"selected_epoch": 1}, metrics=bad)["qualified"]


def _historical_shards():
    shards = {}
    for shard in range(4):
        rows = []
        for local in range(16):
            panel_index = shard * 16 + local
            for seat in (0, 1):
                rows.append({
                    "seed": 2_710_000 + panel_index,
                    "candidate_seat": seat,
                    "candidate_points": 1.0 if seat == 0 else 0.0,
                    "candidate_margin": 1 if seat == 0 else -1,
                    "termination_reason": "completed",
                })
        shards[shard] = rows
    return shards


def test_historical_panel_combiner_binds_64_paired_states() -> None:
    result = _combine_historical_shards(_historical_shards())
    assert result["paired_seeds"] == 64
    assert result["games"] == 128
    assert result["score_share"] == 0.5
    missing = _historical_shards()
    del missing[3]
    with pytest.raises(ValueError, match="four exact"):
        _combine_historical_shards(missing)


def test_decision_tree_never_installs() -> None:
    cases = [
        (False, None, None, "offline_failed"),
        (True, False, None, "failed_diverse_state"),
        (True, True, False, "not_ordinary_robust"),
        (True, True, True, "authorize_population_loop"),
    ]
    for offline, historical, ordinary, fragment in cases:
        decision = reanalysis_decision(
            offline_passed=offline,
            historical_passed=historical,
            ordinary_passed=ordinary,
        )
        assert fragment in decision
        assert decision.endswith("no_install")
