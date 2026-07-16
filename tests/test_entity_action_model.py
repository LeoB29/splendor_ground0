from __future__ import annotations

from dataclasses import asdict

import torch

from splendor_ai.bots import CanonicalEntityActionScorerBot, LoopFallbackConfig
from splendor_ai.bots.search_bot import CheckpointLegalActionPriorScorer
from splendor_ai.encoding import (
    actions_by_canonical_successor_key,
    canonical_legal_action_descriptor,
    collate_entity_action_batch,
    encode_canonical_state_entities,
)
from splendor_ai.engine.actions import ActionType
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


class _ActionTypeScores:
    def __init__(self, *, take_score: float, buy_score: float) -> None:
        self._take_score = take_score
        self._buy_score = buy_score

    def __call__(self, *, action_type_ids, action_mask, **_):
        scores = torch.zeros_like(action_type_ids, dtype=torch.float32)
        take_id = list(ActionType).index(ActionType.TAKE_TOKENS) + 1
        buy_visible_id = list(ActionType).index(ActionType.BUY_VISIBLE) + 1
        buy_reserved_id = list(ActionType).index(ActionType.BUY_RESERVED) + 1
        scores[action_type_ids == take_id] = self._take_score
        scores[(action_type_ids == buy_visible_id) | (action_type_ids == buy_reserved_id)] = (
            self._buy_score
        )
        return scores, torch.zeros(action_type_ids.shape[0])


def _write_entity_checkpoint(tmp_path):
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
    return checkpoint_path


def _state_with_legal_buy():
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[state.current_player]
    for card in state.visible_tier_cards[1][:2]:
        for color, cost in card.cost.items():
            player.tokens[color] = max(player.tokens[color], cost)
    return env, state


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
    checkpoint_path = _write_entity_checkpoint(tmp_path)
    env = SplendorEnv(seed=8)
    state = env.initial_state()
    actions = env.legal_actions(state)

    chosen = CanonicalEntityActionScorerBot(checkpoint_path).choose_action(env, state, actions)

    assert chosen in actions


def test_entity_loop_fallback_uses_best_canonical_buy_after_repeated_public_state(
    tmp_path,
) -> None:
    env, state = _state_with_legal_buy()
    checkpoint_path = _write_entity_checkpoint(tmp_path)
    bot = CanonicalEntityActionScorerBot(checkpoint_path)
    bot._model = _ActionTypeScores(take_score=5.0, buy_score=4.5)

    first_actions = env.legal_actions(state)
    first = bot.choose_action(env, state, first_actions)
    assert first.action_type == ActionType.TAKE_TOKENS
    assert bot.loop_fallback_triggers == 0

    reordered_state = state.copy_shallow()
    reordered_state.visible_tier_cards[1].reverse()
    reordered_actions = env.legal_actions(reordered_state)
    chosen = bot.choose_action(env, reordered_state, reordered_actions)

    assert chosen.action_type == ActionType.BUY_VISIBLE
    assert chosen in actions_by_canonical_successor_key(
        reordered_state,
        reordered_actions,
    ).values()
    assert bot.loop_fallback_triggers == 1


def test_entity_loop_fallback_respects_score_gap(tmp_path) -> None:
    env, state = _state_with_legal_buy()
    checkpoint_path = _write_entity_checkpoint(tmp_path)
    bot = CanonicalEntityActionScorerBot(
        checkpoint_path,
        loop_fallback=LoopFallbackConfig(max_buy_logit_gap=8.0),
    )
    bot._model = _ActionTypeScores(take_score=20.0, buy_score=1.0)
    actions = env.legal_actions(state)

    first = bot.choose_action(env, state, actions)
    second = bot.choose_action(env, state, actions)

    assert first.action_type == ActionType.TAKE_TOKENS
    assert second.action_type == ActionType.TAKE_TOKENS
    assert second != first
    assert bot.loop_fallback_triggers == 1
