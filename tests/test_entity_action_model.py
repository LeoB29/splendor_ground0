from __future__ import annotations

from dataclasses import asdict

import torch

from splendor_ai.bots import CanonicalEntityActionScorerBot
from splendor_ai.bots.search_bot import CheckpointLegalActionPriorScorer
from splendor_ai.encoding import (
    canonical_legal_action_descriptor,
    collate_entity_action_batch,
    encode_canonical_state_entities,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, RESERVE_SOURCE_DECK
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)


def _model() -> CanonicalEntityActionScorerModel:
    torch.manual_seed(4)
    return CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=32,
            action_hidden_size=32,
            scorer_hidden_size=32,
            entity_layers=1,
            action_layers=1,
            attention_heads=4,
            attention_feedforward_size=64,
            id_embedding_size=8,
            dropout=0.0,
        )
    ).eval()


def test_entity_encoding_hides_blind_reserve_identity_but_not_tier() -> None:
    state_a = SplendorEnv(seed=0).initial_state()
    state_b = SplendorEnv(seed=0).initial_state()
    state_c = SplendorEnv(seed=0).initial_state()
    state_a.players[1].reserve_card(Card("private-a", 1, "white", 0, {}), RESERVE_SOURCE_DECK)
    state_b.players[1].reserve_card(Card("private-b", 1, "black", 5, {}), RESERVE_SOURCE_DECK)
    state_c.players[1].reserve_card(Card("private-c", 3, "black", 5, {}), RESERVE_SOURCE_DECK)

    assert encode_canonical_state_entities(state_a, 0) == encode_canonical_state_entities(state_b, 0)
    assert encode_canonical_state_entities(state_a, 0) != encode_canonical_state_entities(state_c, 0)


def test_entity_model_is_equivariant_to_legal_action_order() -> None:
    env = SplendorEnv(seed=5)
    state = env.initial_state()
    actions = env.legal_actions(state)
    permuted_actions = list(reversed(actions))
    model = _model()

    with torch.no_grad():
        scores, value = model(**collate_entity_action_batch([(state, actions)]))
        permuted_scores, permuted_value = model(
            **collate_entity_action_batch([(state, permuted_actions)])
        )

    torch.testing.assert_close(value, permuted_value)
    torch.testing.assert_close(scores[0], torch.flip(permuted_scores[0], dims=(0,)))


def test_entity_model_is_invariant_to_market_order_for_semantic_actions() -> None:
    env = SplendorEnv(seed=6)
    state_a = env.initial_state()
    state_b = state_a.copy_shallow()
    state_b.visible_tier_cards[1].reverse()
    actions_a = env.legal_actions(state_a)
    actions_b = env.legal_actions(state_b)
    model = _model()

    with torch.no_grad():
        scores_a, _ = model(**collate_entity_action_batch([(state_a, actions_a)]))
        scores_b, _ = model(**collate_entity_action_batch([(state_b, actions_b)]))
    by_action_a = {
        canonical_legal_action_descriptor(state_a, action): scores_a[0, index]
        for index, action in enumerate(actions_a)
    }
    by_action_b = {
        canonical_legal_action_descriptor(state_b, action): scores_b[0, index]
        for index, action in enumerate(actions_b)
    }

    assert by_action_a.keys() == by_action_b.keys()
    for key in by_action_a:
        torch.testing.assert_close(by_action_a[key], by_action_b[key])


def test_checkpoint_scorer_loads_canonical_entity_model(tmp_path) -> None:
    model = _model()
    checkpoint_path = tmp_path / "entity.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    env = SplendorEnv(seed=7)
    state = env.initial_state()
    actions = env.legal_actions(state)

    scores = CheckpointLegalActionPriorScorer(checkpoint_path).score_actions(state, actions)

    assert set(scores) == set(actions)


def test_entity_checkpoint_bot_returns_a_legal_action(tmp_path) -> None:
    model = _model()
    checkpoint_path = tmp_path / "entity_bot.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    env = SplendorEnv(seed=8)
    state = env.initial_state()
    actions = env.legal_actions(state)

    chosen = CanonicalEntityActionScorerBot(checkpoint_path).choose_action(env, state, actions)

    assert chosen in actions
