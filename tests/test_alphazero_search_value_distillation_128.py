from __future__ import annotations

from dataclasses import replace

import pytest

from splendor_ai.alphazero.replay import AlphaZeroReplayRow
from splendor_ai.alphazero.search_value_distillation import (
    BLEND_WEIGHT,
    PANEL_SPECS,
    _combine_panel,
    blended_value_target,
    search_value_decision,
    search_value_offline_qualification,
    search_value_training_config,
)


def _row() -> AlphaZeroReplayRow:
    return AlphaZeroReplayRow(
        generation=11,
        game_seed=1,
        turn_index=0,
        player_id=0,
        entity_type_ids=(1,),
        entity_card_ids=(0,),
        entity_noble_ids=(0,),
        entity_numeric_features=((0.0,) * 14,),
        action_type_ids=(1,),
        action_card_ids=(0,),
        action_noble_ids=(0,),
        action_numeric_features=((0.0,) * 12,),
        action_payloads=({},),
        root_visits=(1024,),
        policy_target=(1.0,),
        root_network_priors=(1.0,),
        root_search_priors=(1.0,),
        root_q_values=(0.2,),
        selected_action_row=0,
        raw_policy_top_row=0,
        search_changed_raw_action=False,
        visit_argmax_row=0,
        visit_argmax_changed_raw_action=False,
        raw_network_value=0.1,
        search_root_value=0.2,
        value_target=1.0,
        winner=0,
        final_scores=(15, 10),
        termination_reason="completed",
        checkpoint_generation=11,
        checkpoint_path="generation_011.pt",
        search_config={"simulations": 1024},
        search_telemetry={"simulations": 1024},
    )


def test_fixed_blend_is_exact_bounded_and_not_selectable() -> None:
    assert BLEND_WEIGHT == 0.5
    assert blended_value_target(_row()) == pytest.approx(0.6)
    assert blended_value_target(replace(_row(), value_target=-1.0, search_root_value=1.0)) == 0.0


def test_training_recipe_changes_only_seed_and_value_target_adapter() -> None:
    config = search_value_training_config()
    assert config.epochs == 12
    assert config.batch_size == 256
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-4
    assert config.policy_loss_weight == 1.0
    assert config.value_loss_weight == 0.5
    assert config.validation_fraction == 0.2
    assert config.early_stopping_patience == 3
    assert config.early_stopping_min_delta == 0.001
    assert config.seed == 1_128_000


def test_offline_gate_requires_control_advantage_and_terminal_safety() -> None:
    holdout = {
        "generation_11": {
            "policy_ce": 2.0, "search_value_mse": 0.4, "blended_value_mse": 0.5,
            "terminal_value_mse": 0.9,
        },
        "terminal_control_114": {
            "policy_ce": 1.98, "search_value_mse": 0.39, "blended_value_mse": 0.45,
            "terminal_value_mse": 0.89,
        },
        "search_value_candidate": {
            "policy_ce": 1.979, "search_value_mse": 0.35, "blended_value_mse": 0.40,
            "terminal_value_mse": 0.90,
        },
    }
    assert search_value_offline_qualification(
        training={"selected_epoch": 1}, holdout=holdout
    )["qualified"]
    bad = {name: dict(metrics) for name, metrics in holdout.items()}
    bad["search_value_candidate"]["blended_value_mse"] = 0.46
    assert not search_value_offline_qualification(
        training={"selected_epoch": 1}, holdout=bad
    )["qualified"]


def _shards(panel: str) -> dict[int, dict[str, object]]:
    records = {}
    for shard in range(4):
        start = PANEL_SPECS[panel]["seed_start"] + shard * 50
        rows = [
            {
                "seed": start + offset,
                "candidate_seat": seat,
                "candidate_points": 1.0 if seat == 0 else 0.0,
                "candidate_margin": 1 if seat == 0 else -1,
                "termination_reason": "completed",
            }
            for offset in range(50)
            for seat in (0, 1)
        ]
        records[shard] = {"rows": rows}
    return records


@pytest.mark.parametrize("panel", list(PANEL_SPECS))
def test_panel_combiner_binds_fresh_exact_schedule(panel: str) -> None:
    combined = _combine_panel(panel, _shards(panel))
    assert combined["games"] == 400
    assert combined["paired_seeds"] == 200
    assert combined["score_share"] == 0.5
    missing = _shards(panel)
    del missing[3]
    with pytest.raises(ValueError, match="four exact"):
        _combine_panel(panel, missing)  # type: ignore[arg-type]


def test_decision_tree_never_installs() -> None:
    cases = [
        (False, None, None, None, "signal_audit_failed"),
        (True, False, None, None, "offline_failed"),
        (True, True, False, None, "failed_direct_control"),
        (True, True, True, False, "not_parent_robust"),
        (True, True, True, True, "authorize_population_pilot"),
    ]
    for audit, offline, primary, secondary, fragment in cases:
        decision = search_value_decision(
            audit_passed=audit,
            offline_passed=offline,
            primary_passed=primary,
            secondary_passed=secondary,
        )
        assert fragment in decision
        assert decision.endswith("no_install")
