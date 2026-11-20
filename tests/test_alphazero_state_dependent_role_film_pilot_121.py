from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from run_alphazero_state_dependent_role_film_pilot_121 import build_parser
from splendor_ai.alphazero.network import AlphaZeroInference, create_initial_checkpoint
from splendor_ai.alphazero.role_film_network import (
    create_identity_role_film_model,
    load_role_film_checkpoint,
    role_film_parameter_counts,
    save_role_film_checkpoint,
)
from splendor_ai.alphazero.start_role_film_pilot import PROTOCOL_SHA256
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_121_protocol_hash_and_launch_guard() -> None:
    protocol = Path("docs/engineering/alphazero_state_dependent_role_film_pilot_121.md")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert build_parser().parse_args([]).launch is False
    assert build_parser().parse_args(["--launch"]).launch is True


def test_role_film_is_exact_identity_and_round_trips(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    create_initial_checkpoint(parent_path, seed=121)
    model, identity = create_identity_role_film_model(parent_path, device="cpu")
    counts = role_film_parameter_counts(model)
    assert counts["role_total"] == counts["trainable"] == 384
    assert counts["policy_gain"] == counts["policy_bias"] == 96
    assert counts["value_gain"] == counts["value_bias"] == 96
    state = SplendorEnv(seed=121).initial_state()
    flipped = state.copy_shallow()
    flipped.start_player = 1
    actions = SplendorEnv(seed=121).legal_actions(state)
    batch = collate_entity_action_batch([(state, actions), (flipped, actions)])
    roles = torch.tensor([1, 0], dtype=torch.long)
    with torch.no_grad():
        parent_logits, parent_values = model.parent(**batch)
        logits, values = model(**batch, observer_is_start_player=roles)
    torch.testing.assert_close(logits, parent_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values, parent_values, rtol=0.0, atol=0.0)
    checkpoint = tmp_path / "film.pt"
    save_role_film_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={"parent_generation": identity.generation},
    )
    restored, restored_identity = load_role_film_checkpoint(checkpoint, device="cpu")
    assert restored_identity.generation == 12
    with torch.no_grad():
        restored_logits, restored_values = restored(
            **batch, observer_is_start_player=roles
        )
    torch.testing.assert_close(logits, restored_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values, restored_values, rtol=0.0, atol=0.0)


def test_role_film_endpoint_separation_and_inference(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    create_initial_checkpoint(parent_path, seed=122)
    model, _identity = create_identity_role_film_model(parent_path, device="cpu")
    state = SplendorEnv(seed=122).initial_state()
    actions = SplendorEnv(seed=122).legal_actions(state)
    batch = collate_entity_action_batch([(state, actions)])
    logits, _values = model(**batch, observer_is_start_player=torch.tensor([1]))
    logits.sum().backward()
    assert model.policy_role.grad is not None
    assert float(model.policy_role.grad.abs().sum()) > 0.0
    assert model.value_role.grad is None
    model.zero_grad(set_to_none=True)
    _logits, values = model(**batch, observer_is_start_player=torch.tensor([1]))
    values.sum().backward()
    assert model.value_role.grad is not None
    assert float(model.value_role.grad.abs().sum()) > 0.0
    assert model.policy_role.grad is None

    with torch.no_grad():
        model.policy_role[0].fill_(0.02)
        model.policy_role[1].fill_(0.03)
        model.value_role[0].fill_(-0.02)
        model.value_role[1].fill_(0.01)
    checkpoint = tmp_path / "film.pt"
    save_role_film_checkpoint(checkpoint, model=model, generation=12)
    inference = AlphaZeroInference(checkpoint)
    first = state.copy_shallow()
    second = state.copy_shallow()
    second.start_player = 1
    first_prediction = inference.predict_one(first, actions)
    second_prediction = inference.predict_one(second, actions)
    assert first_prediction.logits != second_prediction.logits
    assert first_prediction.value != second_prediction.value
