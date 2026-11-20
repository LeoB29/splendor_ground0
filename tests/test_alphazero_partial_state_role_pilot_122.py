from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from run_alphazero_partial_state_role_pilot_122 import build_parser
from splendor_ai.alphazero.network import AlphaZeroInference, create_initial_checkpoint
from splendor_ai.alphazero.role_state_network import (
    create_identity_role_state_model,
    load_role_state_checkpoint,
    role_state_parameter_counts,
    role_state_trainable_names,
    save_role_state_checkpoint,
)
from splendor_ai.alphazero.start_role_state_pilot import PROTOCOL_SHA256
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_122_protocol_hash_and_launch_guard() -> None:
    protocol = Path(
        "docs/engineering/alphazero_partial_state_role_representation_pilot_122.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert build_parser().parse_args([]).launch is False
    assert build_parser().parse_args(["--launch"]).launch is True


def test_partial_state_role_identity_scope_and_roundtrip(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    create_initial_checkpoint(parent_path, seed=122)
    model, identity = create_identity_role_state_model(parent_path, device="cpu")
    counts = role_state_parameter_counts(model)
    assert counts == {
        "parent": 356_450,
        "frozen_parent": 281_666,
        "final_entity_block": 74_784,
        "role_input": 96,
        "total": 356_546,
        "trainable": 74_880,
    }
    assert role_state_trainable_names(model)[0] == "start_role_direction"
    assert all(
        name.startswith("parent.entity_encoder.layers.1.")
        for name in role_state_trainable_names(model)[1:]
    )

    state = SplendorEnv(seed=122).initial_state()
    flipped = state.copy_shallow()
    flipped.start_player = 1
    actions = SplendorEnv(seed=122).legal_actions(state)
    batch = collate_entity_action_batch([(state, actions), (flipped, actions)])
    roles = torch.tensor([1, 0], dtype=torch.long)
    with torch.no_grad():
        parent_logits, parent_values = model.parent(**batch)
        logits, values = model(**batch, observer_is_start_player=roles)
    torch.testing.assert_close(logits, parent_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values, parent_values, rtol=0.0, atol=0.0)

    checkpoint = tmp_path / "role_state.pt"
    save_role_state_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={"parent_generation": identity.generation},
    )
    restored, restored_identity = load_role_state_checkpoint(checkpoint, device="cpu")
    assert restored_identity.generation == 12
    with torch.no_grad():
        restored_logits, restored_values = restored(
            **batch, observer_is_start_player=roles
        )
    torch.testing.assert_close(logits, restored_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values, restored_values, rtol=0.0, atol=0.0)


def test_partial_state_role_joint_gradients_and_inference(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    create_initial_checkpoint(parent_path, seed=123)
    model, _identity = create_identity_role_state_model(parent_path, device="cpu")
    state = SplendorEnv(seed=123).initial_state()
    actions = SplendorEnv(seed=123).legal_actions(state)
    batch = collate_entity_action_batch([(state, actions)])

    logits, values = model(
        **batch, observer_is_start_player=torch.tensor([1], dtype=torch.long)
    )
    (logits.sum() + values.sum()).backward()
    assert model.start_role_direction.grad is not None
    assert float(model.start_role_direction.grad.abs().sum()) > 0.0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.parent.entity_encoder.layers[-1].parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.parent.named_parameters()
        if not name.startswith("entity_encoder.layers.1.")
    )

    with torch.no_grad():
        model.start_role_direction.fill_(0.02)
    checkpoint = tmp_path / "role_state.pt"
    save_role_state_checkpoint(checkpoint, model=model, generation=12)
    inference = AlphaZeroInference(checkpoint)
    nonstart = state.copy_shallow()
    nonstart.start_player = 1
    first = inference.predict_one(state, actions)
    second = inference.predict_one(nonstart, actions)
    assert first.logits != second.logits
    assert first.value != second.value
