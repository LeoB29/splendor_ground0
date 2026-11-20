from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from run_alphazero_split_readout_role_pilot_120 import build_parser
from splendor_ai.alphazero.network import AlphaZeroInference, create_initial_checkpoint
from splendor_ai.alphazero.role_readout_network import (
    create_identity_readout_role_model,
    load_readout_role_checkpoint,
    readout_role_parameter_counts,
    save_readout_role_checkpoint,
)
from splendor_ai.alphazero.start_role_readout_pilot import PROTOCOL_SHA256
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_120_protocol_hash_and_launch_guard() -> None:
    protocol = Path("docs/engineering/alphazero_split_readout_role_pilot_120.md")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert build_parser().parse_args([]).launch is False
    assert build_parser().parse_args(["--launch"]).launch is True


def test_split_readout_is_exact_identity_and_round_trips(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    create_initial_checkpoint(parent_path, seed=120)
    model, identity = create_identity_readout_role_model(parent_path, device="cpu")
    counts = readout_role_parameter_counts(model)
    assert counts["policy_role"] == counts["value_role"] == 96
    assert counts["role_total"] == counts["trainable"] == 192
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())

    first = SplendorEnv(seed=120).initial_state()
    second = first.copy_shallow()
    second.start_player = 1
    actions = SplendorEnv(seed=120).legal_actions(first)
    batch = collate_entity_action_batch([(first, actions), (second, actions)])
    roles = torch.tensor([1, 0], dtype=torch.long)
    with torch.no_grad():
        parent_logits, parent_values = model.parent(**batch)
        logits, values = model(**batch, observer_is_start_player=roles)
    torch.testing.assert_close(logits, parent_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values, parent_values, rtol=0.0, atol=0.0)

    checkpoint = tmp_path / "role.pt"
    save_readout_role_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={"parent_generation": identity.generation},
    )
    restored, restored_identity = load_readout_role_checkpoint(checkpoint, device="cpu")
    assert restored_identity.generation == 12
    with torch.no_grad():
        restored_logits, restored_values = restored(
            **batch, observer_is_start_player=roles
        )
    torch.testing.assert_close(logits, restored_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values, restored_values, rtol=0.0, atol=0.0)


def test_split_readout_gradients_and_inference_are_endpoint_specific(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    create_initial_checkpoint(parent_path, seed=121)
    model, _identity = create_identity_readout_role_model(parent_path, device="cpu")
    state = SplendorEnv(seed=121).initial_state()
    actions = SplendorEnv(seed=121).legal_actions(state)
    batch = collate_entity_action_batch([(state, actions)])

    logits, values = model(
        **batch,
        observer_is_start_player=torch.tensor([1]),
    )
    logits.sum().backward()
    assert model.policy_role.grad is not None
    assert float(model.policy_role.grad.abs().sum()) > 0.0
    assert model.value_role.grad is None
    model.zero_grad(set_to_none=True)
    _logits, values = model(
        **batch,
        observer_is_start_player=torch.tensor([1]),
    )
    values.sum().backward()
    assert model.value_role.grad is not None
    assert float(model.value_role.grad.abs().sum()) > 0.0
    assert model.policy_role.grad is None

    with torch.no_grad():
        model.policy_role.fill_(0.05)
        model.value_role.fill_(-0.05)
    checkpoint = tmp_path / "role.pt"
    save_readout_role_checkpoint(checkpoint, model=model, generation=12)
    inference = AlphaZeroInference(checkpoint)
    first = state.copy_shallow()
    second = state.copy_shallow()
    second.start_player = 1
    first_prediction = inference.predict_one(first, actions)
    second_prediction = inference.predict_one(second, actions)
    assert first_prediction.logits != second_prediction.logits
    assert first_prediction.value != second_prediction.value
