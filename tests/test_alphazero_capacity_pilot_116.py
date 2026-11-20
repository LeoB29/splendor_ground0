from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from splendor_ai.alphazero.capacity_network import (
    CAPACITY_MODEL_KIND,
    capacity_parameter_counts,
    create_identity_capacity_model,
    load_capacity_checkpoint,
    save_capacity_checkpoint,
)
from splendor_ai.alphazero.capacity_pilot import (
    ARENA_SHARDS,
    PRIMARY_PAIRED_SEEDS,
    PRIMARY_SEED_START,
    PROTOCOL_SHA256,
    SHARD_PAIRED_SEEDS,
    _arena_identity,
    capacity_decision,
    combine_primary_shards,
    gameplay_assessment,
    offline_qualification,
    residual_model_config,
)
from splendor_ai.alphazero.network import AlphaZeroInference, create_initial_checkpoint
from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.env import SplendorEnv


def _rows(seed_start: int, paired_seeds: int) -> list[dict[str, object]]:
    return [
        {
            "seed": seed_start + offset,
            "candidate_seat": seat,
            "candidate_points": 1.0 if seat == 0 else 0.0,
            "candidate_margin": 1 if seat == 0 else -1,
            "termination_reason": "completed",
        }
        for offset in range(paired_seeds)
        for seat in (0, 1)
    ]


def _shards() -> dict[int, dict[str, object]]:
    return {
        index: {
            "paired_seeds": SHARD_PAIRED_SEEDS,
            "games": 2 * SHARD_PAIRED_SEEDS,
            "rows": _rows(
                PRIMARY_SEED_START + index * SHARD_PAIRED_SEEDS,
                SHARD_PAIRED_SEEDS,
            ),
        }
        for index in range(ARENA_SHARDS)
    }


def test_protocol_hash_and_medium_configuration_are_frozen() -> None:
    protocol = Path("docs/engineering/alphazero_capacity_pilot_116.md")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    config = residual_model_config()
    assert (config.entity_hidden_size, config.action_hidden_size) == (128, 128)
    assert config.scorer_hidden_size == 192
    assert (config.entity_layers, config.action_layers) == (2, 2)


def test_capacity_bridge_is_exact_at_epoch_zero_and_round_trips(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    create_initial_checkpoint(parent_path, seed=116)
    model, parent_identity = create_identity_capacity_model(
        parent_path,
        residual_config=residual_model_config(),
        device="cpu",
    )
    env = SplendorEnv(seed=116)
    state = env.initial_state()
    actions = env.legal_actions(state)
    batch = collate_entity_action_batch([(state, actions)])
    with torch.no_grad():
        parent_logits, parent_value = model.parent(**batch)
        candidate_logits, candidate_value = model(**batch)
    assert torch.equal(parent_logits, candidate_logits)
    assert torch.equal(parent_value, candidate_value)
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())
    assert all(parameter.requires_grad for parameter in model.residual.parameters())
    counts = capacity_parameter_counts(model)
    assert counts == {
        "parent": 356_450,
        "residual": 635_378,
        "total": 991_828,
        "trainable": 635_378,
    }

    checkpoint = tmp_path / "capacity.pt"
    save_capacity_checkpoint(
        checkpoint,
        model=model,
        generation=parent_identity.generation + 1,
        metadata={"parent_generation": parent_identity.generation, "selected_epoch": 0},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["model_kind"] == CAPACITY_MODEL_KIND
    restored, identity = load_capacity_checkpoint(checkpoint, device="cpu")
    assert identity.generation == 1
    with torch.no_grad():
        restored_logits, restored_value = restored(**batch)
    assert torch.equal(candidate_logits, restored_logits)
    assert torch.equal(candidate_value, restored_value)
    prediction = AlphaZeroInference(checkpoint).predict_one(state, actions)
    assert prediction.logits == pytest.approx(candidate_logits[0].tolist())
    assert prediction.value == pytest.approx(float(candidate_value[0]))


def test_offline_gate_requires_identity_capacity_and_real_holdout_gain() -> None:
    training = {
        "identity_audit": {"max_logit_abs_diff": 0.0, "max_value_abs_diff": 0.0},
        "parameter_counts": {"parent": 100, "total": 300},
        "parent_frozen": True,
        "selected_epoch": 1,
    }
    parent = {"policy_loss": 1.0, "value_loss": 1.0, "objective_loss": 1.5}
    candidate = {"policy_loss": 0.9, "value_loss": 1.01, "objective_loss": 1.405}
    assert offline_qualification(
        training=training, parent=parent, candidate=candidate
    )["qualified"]
    training["selected_epoch"] = 0
    assert not offline_qualification(
        training=training, parent=parent, candidate=candidate
    )["qualified"]


def test_primary_identity_and_combiner_bind_exact_schedule() -> None:
    identities = [_arena_identity(index, "candidate", 200) for index in range(4)]
    assert [identity["seed_start"] for identity in identities] == [
        2_160_000,
        2_160_050,
        2_160_100,
        2_160_150,
    ]
    assert all(identity["candidate_search_config"]["simulations"] == 256 for identity in identities)
    assert all(
        identity["candidate_search_config"] == identity["baseline_search_config"]
        for identity in identities
    )
    arena = combine_primary_shards(_shards())
    assert arena["paired_seeds"] == PRIMARY_PAIRED_SEEDS
    assert arena["games"] == 400
    assert arena["score_share"] == 0.5
    broken = _shards()
    broken[0]["rows"][0]["seed"] = -1  # type: ignore[index]
    with pytest.raises(ValueError, match="exact schedule"):
        combine_primary_shards(broken)


def test_gameplay_gate_and_decisions_never_install() -> None:
    arena = {
        "games": 400,
        "score_share": 0.56,
        "mean_margin": 0.5,
        "termination_reasons": {"completed": 400},
        "rows": [
            {"candidate_seat": seat, "candidate_points": 0.56}
            for seat in (0, 1)
            for _ in range(200)
        ],
    }
    assert not gameplay_assessment(arena, {"ci95_low": 0.5})["passed"]
    assert gameplay_assessment(arena, {"ci95_low": 0.501})["passed"]
    decisions = {
        capacity_decision(qualified=False, gameplay_passed=None),
        capacity_decision(qualified=True, gameplay_passed=False),
        capacity_decision(qualified=True, gameplay_passed=True),
    }
    assert all(decision.endswith("no_install") for decision in decisions)
