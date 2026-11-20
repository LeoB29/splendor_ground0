from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from run_alphazero_paired_role_objective_audit_123 import build_parser
from splendor_ai.alphazero.network import create_initial_checkpoint
from splendor_ai.alphazero.role_state_network import create_identity_role_state_model
from splendor_ai.alphazero.start_role_objective_audit import (
    PROTOCOL_SHA256,
    _future_development_gate,
    _objective_losses,
)
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_123_protocol_hash_and_protected_paths_absent() -> None:
    protocol = Path("docs/engineering/alphazero_paired_role_objective_audit_123.md")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    args = build_parser().parse_args([])
    assert not hasattr(args, "launch")
    assert not hasattr(args, "result_117")
    assert not hasattr(args, "holdout_replay")


def test_contrast_first_reaches_role_then_shared_block(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    create_initial_checkpoint(checkpoint, seed=123)
    model, _identity = create_identity_role_state_model(checkpoint, device="cpu")
    state = SplendorEnv(seed=123).initial_state()
    actions = SplendorEnv(seed=123).legal_actions(state)
    inputs = collate_entity_action_batch([(state, actions), (state, actions)])
    inputs["observer_is_start_player"] = torch.tensor([0, 1], dtype=torch.long)
    target_policy = torch.zeros((2, len(actions)), dtype=torch.float32)
    target_policy[0, 0] = 1.0
    target_policy[1, 1] = 1.0
    batch = {
        "inputs": inputs,
        "target_policy": target_policy,
        "target_value": torch.tensor([-0.5, 0.5], dtype=torch.float32),
    }

    losses = _objective_losses(model, batch)
    losses["policy_contrast"].backward()
    assert model.start_role_direction.grad is not None
    assert float(model.start_role_direction.grad.abs().sum()) > 0.0
    block = torch.cat(
        [
            parameter.grad.reshape(-1)
            for parameter in model.parent.entity_encoder.layers[-1].parameters()
        ]
    )
    role_rms = float(torch.linalg.vector_norm(model.start_role_direction.grad)) / math.sqrt(
        model.start_role_direction.numel()
    )
    block_rms = float(torch.linalg.vector_norm(block)) / math.sqrt(block.numel())
    assert block_rms <= 0.001 * role_rms

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.start_role_direction.fill_(0.02)
    losses = _objective_losses(model, batch)
    losses["policy_contrast"].backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.parent.entity_encoder.layers[-1].parameters()
    )


def test_future_development_gate_rejects_general_fit_without_role_uptake() -> None:
    identity = {
        "ordinary_objective": 2.0,
        "policy_contrast": 0.04,
        "value_contrast": 0.20,
    }
    candidate = {
        "ordinary_objective": 1.9,
        "policy_contrast": 0.039,
        "value_contrast": 0.195,
        "mean_raw_policy_tv": 0.001,
        "mean_raw_abs_value_delta": 0.004,
    }
    gate = _future_development_gate(identity, candidate)
    assert gate["checks"]["aggregate_relative_improvement_at_least_0_5pct"]
    assert not gate["checks"]["policy_contrast_relative_reduction_at_least_10pct"]
    assert not gate["checks"]["raw_policy_tv_at_least_0_025"]
    assert gate["rejected"]
