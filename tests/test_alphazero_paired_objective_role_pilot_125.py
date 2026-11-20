from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from run_alphazero_paired_objective_role_pilot_125 import build_parser
from splendor_ai.alphazero.network import create_initial_checkpoint
from splendor_ai.alphazero.role_state_network import create_identity_role_state_model
from splendor_ai.alphazero.start_role_paired_objective_pilot import (
    LAMBDA_POLICY,
    LAMBDA_VALUE,
    PROTOCOL_SHA256,
    _joint_losses,
    _pair_contract,
)
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_125_protocol_hash_and_explicit_launch_guard() -> None:
    protocol = Path(
        "docs/engineering/alphazero_paired_objective_role_pilot_125.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert LAMBDA_POLICY == 702.476696552303
    assert LAMBDA_VALUE == 39.4241527478651
    assert build_parser().parse_args([]).launch is False
    assert build_parser().parse_args(["--launch"]).launch is True


def test_125_pair_contract_requires_intact_opposite_role_pairs() -> None:
    actions = ("take-white", "reserve-card")
    pairs = [
        (
            SimpleNamespace(
                game_seed=seed,
                observer_is_start_player=False,
                actions=actions,
            ),
            SimpleNamespace(
                game_seed=seed,
                observer_is_start_player=True,
                actions=actions,
            ),
        )
        for seed in range(448)
    ]
    contract = _pair_contract(pairs)
    assert all(contract["checks"].values())
    assert (contract["pairs"], contract["oriented_rows"]) == (448, 896)


def test_125_joint_loss_uses_summed_policy_contrast_and_frozen_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "parent.pt"
    create_initial_checkpoint(checkpoint, seed=125)
    model, _identity = create_identity_role_state_model(checkpoint, device="cpu")
    state = SplendorEnv(seed=125).initial_state()
    actions = SplendorEnv(seed=125).legal_actions(state)
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

    losses = _joint_losses(model, batch)

    torch.testing.assert_close(
        losses["summed_policy_contrast"], torch.tensor(2.0), rtol=0.0, atol=1e-7
    )
    torch.testing.assert_close(
        losses["canonical_policy_contrast"],
        torch.tensor(2.0 / len(actions)),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        losses["value_contrast"], torch.tensor(1.0), rtol=0.0, atol=1e-7
    )
    expected = (
        losses["ordinary_objective"]
        + LAMBDA_POLICY * losses["summed_policy_contrast"]
        + 0.5 * LAMBDA_VALUE * losses["value_contrast"]
    )
    torch.testing.assert_close(losses["joint_objective"], expected)
    assert float(losses["mean_raw_policy_tv"]) <= 1e-6
    assert float(losses["mean_raw_abs_value_delta"]) <= 1e-6

    losses["joint_objective"].backward()
    assert model.start_role_direction.grad is not None
    assert float(model.start_role_direction.grad.abs().sum()) > 0.0
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
