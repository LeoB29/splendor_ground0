from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from splendor_ai.bots import CanonicalEntityActionScorerBot
from splendor_ai.encoding import (
    encode_canonical_state_entities,
    encode_privileged_state_entities,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.entity_actor_critic import (
    ActorCriticConfig,
    full_support_probabilities,
    load_entity_actor,
    train_actor_critic_generation,
)
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    PrivilegedEntityCriticConfig,
)


def _checkpoint(tmp_path):
    model = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=16,
            action_hidden_size=16,
            scorer_hidden_size=16,
            entity_layers=1,
            action_layers=1,
            attention_heads=2,
            attention_feedforward_size=32,
            id_embedding_size=4,
        )
    )
    path = tmp_path / "actor.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        path,
    )
    return path


def test_privileged_encoding_sees_hidden_deck_order_public_actor_cannot() -> None:
    state_a = SplendorEnv(seed=3).initial_state()
    state_b = state_a.copy_shallow()
    state_b.hidden_tier_decks[1].reverse()

    assert encode_canonical_state_entities(state_a, 0) == encode_canonical_state_entities(state_b, 0)
    assert encode_privileged_state_entities(state_a, 0) != encode_privileged_state_entities(state_b, 0)


def test_full_support_distribution_is_positive_and_normalized() -> None:
    scores = torch.tensor([[4.0, 1.0, -3.0, 0.0]])
    mask = torch.tensor([[True, True, True, False]])

    probabilities = full_support_probabilities(scores, mask, temperature=1.0, uniform_mix=0.05)

    assert float(probabilities[0, :3].min()) > 0.0
    assert probabilities[0, 3] == 0.0
    assert float(probabilities.sum()) == pytest.approx(1.0)


def test_actor_checkpoint_compatibility_and_smoke_update(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    actor = load_entity_actor(checkpoint)
    assert actor.entity_type_embedding.num_embeddings == 13
    config = ActorCriticConfig(
        rollout_games=4,
        seed_start=900,
        max_turns=16,
        repetition_limit=0,
        no_progress_limit=0,
        actor_epochs=3,
        critic_warmup_epochs=1,
        critic_epochs=1,
        batch_size=16,
        actor_learning_rate=1e-3,
        target_kl=1e-12,
    )

    artifacts = train_actor_critic_generation(
        checkpoint,
        [
            ("parent_a", lambda: CanonicalEntityActionScorerBot(checkpoint)),
            ("parent_b", lambda: CanonicalEntityActionScorerBot(checkpoint)),
        ],
        config,
        critic_config=PrivilegedEntityCriticConfig(
            hidden_size=16,
            layers=1,
            attention_heads=2,
            attention_feedforward_size=32,
            id_embedding_size=4,
        ),
    )

    assert artifacts.rollout.steps > 0
    assert artifacts.rollout.minimum_behavior_probability > 0.0
    assert artifacts.rollout.minimum_stratum_games == 1
    assert artifacts.rollout.maximum_stratum_games == 1
    assert artifacts.rollout.opponent_seats == {
        "parent_a": {"seat0": 1, "seat1": 1},
        "parent_b": {"seat0": 1, "seat1": 1},
    }
    assert artifacts.update.behavior_log_prob_max_error < 1e-5
    assert artifacts.update.max_ratio > 0.0
    assert artifacts.update.actor_epochs_completed == 1
    assert artifacts.update.stopped_for_target_kl
    assert len(artifacts.update.epoch_mean_kls) == 1
    assert artifacts.update.first_batch_gradient_norm > 0.0
    assert artifacts.update.parameter_delta_l2 > 0.0
    assert artifacts.update.relative_parameter_delta > 0.0
    assert artifacts.update.mean_centered_logit_absolute_delta > 0.0
    assert artifacts.update.mean_probability_total_variation > 0.0
    assert 0.0 <= artifacts.update.argmax_change_fraction <= 1.0
    assert artifacts.update.action_type_advantage_means
