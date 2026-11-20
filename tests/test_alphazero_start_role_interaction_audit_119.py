from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from run_alphazero_start_role_interaction_audit_119 import build_parser
from splendor_ai.alphazero.network import create_initial_checkpoint, load_checkpoint
from splendor_ai.alphazero.start_role_interaction_audit import (
    PROTOCOL_SHA256,
    GlobalAntisymmetricRoleChannel,
    SplitReadoutAdditiveRole,
    SplitReadoutFiLMRole,
    _contrast_losses,
)
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_119_protocol_hash_and_no_launch_switch() -> None:
    protocol = Path("docs/engineering/alphazero_start_role_interaction_audit_119.md")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert not hasattr(build_parser().parse_args([]), "launch")


def test_119_prototypes_are_exact_identity_with_frozen_parent(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    create_initial_checkpoint(checkpoint, seed=119)
    state = SplendorEnv(seed=119).initial_state()
    flipped = state.copy_shallow()
    flipped.start_player = 1
    actions = SplendorEnv(seed=119).legal_actions(state)
    batch = collate_entity_action_batch([(state, actions), (flipped, actions)])
    roles = torch.tensor([1, 0], dtype=torch.long)

    expected_counts = {
        GlobalAntisymmetricRoleChannel: 96,
        SplitReadoutAdditiveRole: 192,
        SplitReadoutFiLMRole: 384,
    }
    for prototype, expected_count in expected_counts.items():
        parent, _identity = load_checkpoint(checkpoint, device="cpu")
        model = prototype(parent)
        with torch.no_grad():
            parent_logits, parent_values = parent(**batch)
            logits, values = model(**batch, observer_is_start_player=roles)
        torch.testing.assert_close(logits, parent_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(values, parent_values, rtol=0.0, atol=0.0)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert sum(parameter.numel() for parameter in trainable) == expected_count
        assert all(float(parameter.abs().max()) == 0.0 for parameter in trainable)
        assert all(not parameter.requires_grad for parameter in model.parent.parameters())


def test_split_additive_contrast_gradients_are_endpoint_isolated(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    create_initial_checkpoint(checkpoint, seed=120)
    parent, _identity = load_checkpoint(checkpoint, device="cpu")
    model = SplitReadoutAdditiveRole(parent)
    state = SplendorEnv(seed=120).initial_state()
    actions = SplendorEnv(seed=120).legal_actions(state)
    inputs = collate_entity_action_batch([(state, actions), (state, actions)])
    inputs["observer_is_start_player"] = torch.tensor([0, 1], dtype=torch.long)
    width = len(actions)
    target_policy = torch.zeros((2, width), dtype=torch.float32)
    target_policy[0, 0] = 1.0
    target_policy[1, 1] = 1.0
    batch = {
        "inputs": inputs,
        "target_policy": target_policy,
        "target_value": torch.tensor([-0.5, 0.5]),
    }

    policy_loss, _value_loss = _contrast_losses(model, batch)
    policy_loss.backward()
    assert model.policy_role.grad is not None
    assert float(model.policy_role.grad.abs().sum()) > 0.0
    assert model.value_role.grad is None

    model.zero_grad(set_to_none=True)
    _policy_loss, value_loss = _contrast_losses(model, batch)
    value_loss.backward()
    assert model.value_role.grad is not None
    assert float(model.value_role.grad.abs().sum()) > 0.0
    assert model.policy_role.grad is None
    assert float(model.policy_role.abs().max()) == 0.0
    assert float(model.value_role.abs().max()) == 0.0
