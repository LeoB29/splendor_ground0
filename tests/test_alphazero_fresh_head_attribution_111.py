from __future__ import annotations

from types import SimpleNamespace

import pytest

from splendor_ai.alphazero.fresh_head_attribution import (
    _phase,
    _panel_identity,
    _roc_auc,
    _score_tempo_features,
    _shared_parameters,
    head_attribution_decision,
)
from splendor_ai.alphazero.network import default_model_config
from splendor_ai.training.model import CanonicalEntityActionScorerModel


def test_phase_boundaries_are_frozen() -> None:
    assert _phase(0) == "early_lt20"
    assert _phase(19) == "early_lt20"
    assert _phase(20) == "middle_20_49"
    assert _phase(49) == "middle_20_49"
    assert _phase(50) == "late_ge50"


def test_roc_auc_handles_order_and_ties() -> None:
    assert _roc_auc([0.8, 0.5], [-0.2, 0.1]) == 1.0
    assert _roc_auc([0.0], [0.0]) == 0.5
    assert _roc_auc([], [0.0]) is None


def test_attribution_decision_uses_strict_paired_interval() -> None:
    positive = {"ci95_low": 0.001}
    zero = {"ci95_low": 0.0}
    assert "value_update_primary" in head_attribution_decision(
        value_restore_difference=positive, policy_restore_difference=zero
    )
    assert "policy_update_primary" in head_attribution_decision(
        value_restore_difference=zero, policy_restore_difference=positive
    )
    assert "both_high_dose" in head_attribution_decision(
        value_restore_difference=positive, policy_restore_difference=positive
    )
    assert "no_clear" in head_attribution_decision(
        value_restore_difference=zero, policy_restore_difference=zero
    )


def test_shared_gradient_parameters_exclude_both_heads_and_action_only_tower() -> None:
    model = CanonicalEntityActionScorerModel(default_model_config())
    selected_ids = {id(parameter) for parameter in _shared_parameters(model)}
    named = {name: id(parameter) in selected_ids for name, parameter in model.named_parameters()}
    assert named["entity_encoder.layers.0.self_attn.in_proj_weight"]
    assert named["card_embedding.weight"]
    assert not named["score_head.0.weight"]
    assert not named["value_head.0.weight"]
    assert not named["action_encoder.layers.0.self_attn.in_proj_weight"]


def test_shared_gradient_parameter_selection_is_nonempty_and_unique() -> None:
    model = CanonicalEntityActionScorerModel(default_model_config())
    selected = _shared_parameters(model)
    assert len(selected) == len({id(parameter) for parameter in selected})
    assert sum(parameter.numel() for parameter in selected) > 0


def test_score_tempo_features_decode_canonical_player_rows() -> None:
    own = [0.0] * 34
    opponent = [0.0] * 34
    own[31], opponent[31] = 0.5, 0.25
    own[33], opponent[33] = 0.4, 0.2
    own[26:31] = [2 / 7] * 5
    opponent[26:31] = [1 / 7] * 5
    row = SimpleNamespace(
        entity_type_ids=(4, 5),
        entity_numeric_features=(tuple(own), tuple(opponent)),
        turn_index=50,
    )
    assert _score_tempo_features(row) == pytest.approx(
        [1.0, 5 / 15, 4 / 15, 5 / 15, 0.5]
    )


def test_panel_identity_binds_functional_policy_and_value_roles() -> None:
    base = {"max_turns": 200}
    full = _panel_identity({**base, "variant": "full_high_dose"})
    value = _panel_identity({**base, "variant": "restore_parent_value"})
    policy = _panel_identity({**base, "variant": "restore_parent_policy"})
    assert full["policy_checkpoint_sha256"] == full["value_checkpoint_sha256"]
    assert value["policy_checkpoint_sha256"] != value["value_checkpoint_sha256"]
    assert policy["policy_checkpoint_sha256"] != policy["value_checkpoint_sha256"]
    assert value["policy_checkpoint_sha256"] == full["policy_checkpoint_sha256"]
    assert policy["value_checkpoint_sha256"] == full["value_checkpoint_sha256"]
