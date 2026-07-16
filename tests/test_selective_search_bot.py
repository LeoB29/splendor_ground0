from __future__ import annotations

from dataclasses import asdict

import torch

from splendor_ai.bots import SelectiveSearchHybridBot
from splendor_ai.encoding import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)


def _checkpoint(tmp_path):
    config = CanonicalEntityActionScorerConfig(
        entity_hidden_size=16,
        action_hidden_size=16,
        scorer_hidden_size=16,
        entity_layers=1,
        action_layers=1,
        attention_heads=4,
        attention_feedforward_size=32,
        id_embedding_size=4,
        dropout=0.0,
    )
    model = CanonicalEntityActionScorerModel(config)
    path = tmp_path / "entity.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


class _FixedSearch:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def choose_action(self, _env, _state, _actions):
        self.calls += 1
        return self.action


def _scores(state, actions, gap):
    representatives = list(actions_by_canonical_successor_key(state, actions).values())
    return {
        action: (gap if index == 0 else 0.0)
        for index, action in enumerate(representatives)
    }


def test_low_logit_gap_routes_to_search_and_records_telemetry(tmp_path) -> None:
    env = SplendorEnv(seed=2)
    state = env.initial_state()
    actions = env.legal_actions(state)
    bot = SelectiveSearchHybridBot(_checkpoint(tmp_path))
    search = _FixedSearch(actions[-1])
    bot._search = search
    bot._policy.score_actions = lambda state, actions: _scores(state, actions, 0.4)

    chosen = bot.choose_action(env, state, actions)

    assert chosen == actions[-1]
    assert search.calls == 1
    assert bot.last_decision is not None
    assert bot.last_decision.uncertainty_triggered is True
    assert bot.telemetry["search_fraction"] == 1.0


def test_high_logit_gap_uses_policy_without_search(tmp_path) -> None:
    env = SplendorEnv(seed=3)
    state = env.initial_state()
    actions = env.legal_actions(state)
    bot = SelectiveSearchHybridBot(_checkpoint(tmp_path))
    search = _FixedSearch(actions[-1])
    bot._search = search
    scores = _scores(state, actions, 0.6)
    bot._policy.score_actions = lambda _state, _actions: scores

    chosen = bot.choose_action(env, state, actions)

    assert chosen == next(iter(scores))
    assert search.calls == 0
    assert bot.last_decision is not None
    assert bot.last_decision.used_search is False


def test_score_threshold_routes_high_confidence_tactical_state(tmp_path) -> None:
    env = SplendorEnv(seed=4)
    state = env.initial_state()
    state.players[state.current_player].score = 12
    actions = env.legal_actions(state)
    bot = SelectiveSearchHybridBot(_checkpoint(tmp_path))
    search = _FixedSearch(actions[-1])
    bot._search = search
    bot._policy.score_actions = lambda state, actions: _scores(state, actions, 5.0)

    chosen = bot.choose_action(env, state, actions)

    assert chosen == actions[-1]
    assert bot.last_decision is not None
    assert bot.last_decision.tactical_triggered is True
    assert bot.last_decision.uncertainty_triggered is False
