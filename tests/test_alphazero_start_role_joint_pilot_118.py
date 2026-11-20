from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from run_alphazero_start_role_joint_pilot_118 import build_parser
from splendor_ai.alphazero.network import AlphaZeroInference, create_initial_checkpoint
from splendor_ai.alphazero.role_network import (
    create_identity_role_model,
    load_role_checkpoint,
    role_parameter_counts,
    save_role_checkpoint,
)
from splendor_ai.alphazero.start_role_pilot import (
    PROTOCOL_SHA256,
    SIMULATIONS,
    TRAIN_PAIRS,
    TRAIN_SPLIT_GAMES,
    TRAIN_SPLIT_PAIRS,
    VALIDATION_SPLIT_GAMES,
    VALIDATION_SPLIT_PAIRS,
    _action_key,
    _ordered_actions,
    split_pair_games,
    summarize_target_records,
)
from splendor_ai.encoding import actions_by_canonical_successor_key, collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def test_protocol_hash_and_frozen_counts() -> None:
    protocol = Path("docs/engineering/alphazero_start_role_joint_pilot_118.md")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert SIMULATIONS == 1024
    assert (TRAIN_PAIRS, TRAIN_SPLIT_PAIRS, VALIDATION_SPLIT_PAIRS) == (448, 358, 90)
    assert (TRAIN_SPLIT_GAMES, VALIDATION_SPLIT_GAMES) == (179, 45)


def test_zero_role_bridge_is_exact_and_round_trips(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    create_initial_checkpoint(parent, seed=118)
    model, identity = create_identity_role_model(parent, device="cpu")
    counts = role_parameter_counts(model)
    assert counts["role_embedding"] == 2 * model.parent.config.entity_hidden_size
    assert counts["trainable"] == counts["role_embedding"]
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())

    state = SplendorEnv(seed=118).initial_state()
    flipped = state.copy_shallow()
    flipped.start_player = 1
    env = SplendorEnv(seed=118)
    actions = env.legal_actions(state)
    batch = collate_entity_action_batch([(state, actions), (flipped, actions)])
    with torch.no_grad():
        parent_logits, parent_values = model.parent(**batch)
        role_logits, role_values = model(
            **batch,
            observer_is_start_player=torch.tensor([1, 0]),
        )
    torch.testing.assert_close(parent_logits, role_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(parent_values, role_values, rtol=0.0, atol=0.0)

    checkpoint = tmp_path / "role.pt"
    save_role_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={"parent_generation": identity.generation, "selected_epoch": 0},
    )
    restored, restored_identity = load_role_checkpoint(checkpoint, device="cpu")
    assert restored_identity.generation == 12
    assert restored_identity.metadata["selected_epoch"] == 0
    with torch.no_grad():
        restored_logits, restored_values = restored(
            **batch,
            observer_is_start_player=torch.tensor([1, 0]),
        )
    torch.testing.assert_close(parent_logits, restored_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(parent_values, restored_values, rtol=0.0, atol=0.0)


def test_role_checkpoint_inference_receives_explicit_start_role(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    create_initial_checkpoint(parent, seed=119)
    model, _identity = create_identity_role_model(parent, device="cpu")
    with torch.no_grad():
        model.start_role_embedding.weight[1].fill_(0.25)
    checkpoint = tmp_path / "role.pt"
    save_role_checkpoint(checkpoint, model=model, generation=12)

    first = SplendorEnv(seed=9).initial_state()
    second = first.copy_shallow()
    second.start_player = 1
    actions = SplendorEnv(seed=9).legal_actions(first)
    inference = AlphaZeroInference(checkpoint)
    first_prediction = inference.predict_one(first, actions)
    second_prediction = inference.predict_one(second, actions)
    assert first_prediction.value != second_prediction.value or first_prediction.logits != second_prediction.logits


def test_role_gradient_reaches_only_embedding(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    create_initial_checkpoint(parent, seed=120)
    model, _identity = create_identity_role_model(parent, device="cpu")
    state = SplendorEnv(seed=10).initial_state()
    actions = SplendorEnv(seed=10).legal_actions(state)
    batch = collate_entity_action_batch([(state, actions)])
    logits, values = model(
        **batch,
        observer_is_start_player=torch.tensor([1]),
    )
    (logits.sum() + values.sum()).backward()
    assert model.start_role_embedding.weight.grad is not None
    assert float(model.start_role_embedding.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.parent.parameters())


def test_whole_game_split_keeps_pair_orientations_together() -> None:
    state = SplendorEnv(seed=1).initial_state()
    states = [
        (seed, state.copy_shallow())
        for seed in range(224)
        for _local in range(2)
    ]
    split = split_pair_games(states)
    assert len(split["train_games"]) == 179
    assert len(split["validation_games"]) == 45
    assert split["train_pairs"] == 358
    assert split["validation_pairs"] == 90
    assert not set(split["train_games"]) & set(split["validation_games"])


def test_target_summary_and_cli_launch_guard() -> None:
    records = [
        {
            "identity": {"game_seed": 1},
            "original": {
                "policy_target": [0.75, 0.25],
                "search_root_value": 0.4,
                "visit_argmax_action_key": "a",
                "simulations": 1024,
            },
            "flipped": {
                "policy_target": [0.25, 0.75],
                "search_root_value": -0.2,
                "visit_argmax_action_key": "b",
                "simulations": 1024,
            },
        }
    ]
    summary = summarize_target_records(records)
    assert summary["mean_visit_tv"] == 0.5
    assert summary["top_action_change_rate"] == 1.0
    assert abs(summary["mean_abs_root_value_delta"] - 0.6) < 1e-12
    assert summary["all_exact_1024"]
    assert build_parser().parse_args([]).launch is False
    assert build_parser().parse_args(["--launch"]).launch is True


def test_teacher_action_order_uses_canonical_successor_representatives() -> None:
    env = SplendorEnv(seed=1)
    state = env.initial_state()
    colors = ["white", "blue", "green", "red", "black"]
    for index in range(8):
        color = colors[index % len(colors)]
        state.players[0].tokens[color] += 1
        state.bank_tokens[color] -= 1

    legal = env.legal_actions(state)
    canonical = actions_by_canonical_successor_key(state, legal)
    ordered = _ordered_actions(env, state)
    assert len(canonical) < len(legal)
    assert [_action_key(action) for action in ordered] == sorted(
        _action_key(action) for action in canonical.values()
    )
