from __future__ import annotations

import torch

from splendor_ai.alphazero.decisive_objective import offline_selection
from splendor_ai.alphazero.replay import AlphaZeroReplayRow
from splendor_ai.alphazero.training import alpha_zero_loss, collate_replay_rows


def _row(*, decisive: bool = True) -> AlphaZeroReplayRow:
    return AlphaZeroReplayRow(
        generation=2, game_seed=1, turn_index=10, player_id=0,
        entity_type_ids=(1,), entity_card_ids=(0,), entity_noble_ids=(0,),
        entity_numeric_features=((0.0,) * 34,), action_type_ids=(1, 2),
        action_card_ids=(0, 0), action_noble_ids=(0, 0),
        action_numeric_features=((0.0,) * 34, (0.0,) * 34),
        action_payloads=({}, {}), root_visits=(1, 3), policy_target=(0.25, 0.75),
        root_network_priors=(0.8, 0.2), root_search_priors=(0.8, 0.2),
        root_q_values=(0.0, 0.0), selected_action_row=1, raw_policy_top_row=0,
        search_changed_raw_action=decisive, visit_argmax_row=1,
        visit_argmax_changed_raw_action=decisive, raw_network_value=0.0,
        search_root_value=0.0, value_target=1.0, winner=0, final_scores=(15, 10),
        termination_reason="completed", checkpoint_generation=2, checkpoint_path="parent.pt",
        search_config={}, search_telemetry={},
    )


def test_corrective_collation_is_opt_in_and_weights_divergence() -> None:
    row = _row()
    ordinary = collate_replay_rows([row])
    assert "policy_row_weight" not in ordinary
    corrective = collate_replay_rows(
        [row], policy_divergence_weight=2.0, include_corrective_fields=True
    )
    assert corrective["decisive_mask"].tolist() == [True]
    assert corrective["visit_argmax_target"].tolist() == [1]
    assert torch.isclose(corrective["policy_row_weight"][0], torch.tensor(2.1))


def test_visit_argmax_auxiliary_loss_only_uses_decisive_rows() -> None:
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    result = alpha_zero_loss(
        logits, torch.zeros(2), torch.ones_like(logits, dtype=torch.bool), target,
        torch.zeros(2), visit_argmax_target=torch.tensor([1, 1]),
        decisive_mask=torch.tensor([True, False]), visit_argmax_aux_weight=0.5,
    )
    expected_aux = -torch.log_softmax(logits[0], dim=0)[1]
    assert torch.allclose(result["decisive_aux_loss"], expected_aux)
    assert result["objective_loss"] > result["total_loss"]


def _training(agreement: float, probability: float) -> dict:
    metrics = {
        "policy_loss": 1.0, "value_loss": 0.5,
        "policy_top_target_agreement": agreement,
        "policy_target_top_probability": probability,
    }
    parent = {**metrics, "policy_loss": 1.1, "value_loss": 0.6,
              "policy_top_target_agreement": 0.5}
    phases = {
        name: {"delta_child_minus_parent": {"value_loss": 0.0}}
        for name in ("phase_early_lt20", "phase_middle_20_49", "phase_late_ge50")
    }
    phases["decisive_search_change"] = {"child": metrics}
    return {
        "parent_validation_metrics": parent, "child_validation_metrics": metrics,
        "validation_slices": phases,
    }


def test_offline_selection_requires_material_gain_over_control() -> None:
    candidates = {
        "standard_control": {"training": _training(0.50, 0.20)},
        "divergence_weighted": {"training": _training(0.53, 0.22)},
        "visit_argmax_aux": {"training": _training(0.51, 0.25)},
    }
    selection = offline_selection(candidates)
    assert selection["selected"] == "divergence_weighted"
    assert selection["eligibility"]["divergence_weighted"]["eligible"]
    assert not selection["eligibility"]["visit_argmax_aux"]["eligible"]
