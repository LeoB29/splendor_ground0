from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from run_alphazero_scale_native_policy_contrast_audit_124 import build_parser
from splendor_ai.alphazero.network import create_initial_checkpoint
from splendor_ai.alphazero.role_state_network import create_identity_role_state_model
from splendor_ai.alphazero.start_role_policy_contrast_audit import (
    ARMS,
    PROTOCOL_SHA256,
    _retained_value_coefficient_matches,
    _variant_loss,
)
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_124_protocol_hash_and_no_protected_or_launch_paths() -> None:
    protocol = Path(
        "docs/engineering/alphazero_scale_native_policy_contrast_audit_124.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    args = build_parser().parse_args([])
    assert not hasattr(args, "launch")
    assert not hasattr(args, "result_117")
    assert not hasattr(args, "holdout_replay")


def test_retained_value_coefficient_accepts_only_serialization_roundoff() -> None:
    assert _retained_value_coefficient_matches(39.42415274786513)
    assert not _retained_value_coefficient_matches(39.4241528)


def _synthetic_pair(tmp_path: Path):
    checkpoint = tmp_path / "parent.pt"
    create_initial_checkpoint(checkpoint, seed=124)
    model, _identity = create_identity_role_state_model(checkpoint, device="cpu")
    state = SplendorEnv(seed=124).initial_state()
    actions = SplendorEnv(seed=124).legal_actions(state)
    inputs = collate_entity_action_batch([(state, actions), (state, actions)])
    inputs["observer_is_start_player"] = torch.tensor([0, 1], dtype=torch.long)
    targets = torch.zeros((2, len(actions)), dtype=torch.float32)
    targets[0, 0] = 1.0
    targets[1, 1] = 1.0
    return model, {"inputs": inputs, "target_policy": targets}


def test_scale_native_losses_have_declared_units(tmp_path: Path) -> None:
    model, batch = _synthetic_pair(tmp_path)
    summed = _variant_loss(model, batch, ARMS[0], energy_floor=1.0)
    normalized = _variant_loss(model, batch, ARMS[1], energy_floor=1.0)
    torch.testing.assert_close(summed, torch.tensor(2.0), rtol=0.0, atol=1e-7)
    torch.testing.assert_close(normalized, torch.tensor(1.0), rtol=0.0, atol=1e-7)


def test_scale_native_contrast_reaches_role_then_block(tmp_path: Path) -> None:
    model, batch = _synthetic_pair(tmp_path)
    loss = _variant_loss(model, batch, ARMS[0], energy_floor=1.0)
    loss.backward()
    role = model.start_role_direction.grad
    block = torch.cat(
        [
            parameter.grad.reshape(-1)
            for parameter in model.parent.entity_encoder.layers[-1].parameters()
        ]
    )
    role_rms = float(torch.linalg.vector_norm(role)) / math.sqrt(role.numel())
    block_rms = float(torch.linalg.vector_norm(block)) / math.sqrt(block.numel())
    assert role_rms > 0.0
    assert block_rms <= 0.001 * role_rms

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.start_role_direction.fill_(0.02)
    loss = _variant_loss(model, batch, ARMS[0], energy_floor=1.0)
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.parent.entity_encoder.layers[-1].parameters()
    )
