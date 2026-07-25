from __future__ import annotations

import math

import torch

from splendor_ai.bots import GreedyHeuristicBot
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.recurrent_league import (
    RecurrentLeagueConfig,
    RecurrentLeagueStep,
    behavior_probabilities,
    collect_recurrent_league,
    forward_recurrent,
    train_recurrent_generation,
)
from splendor_ai.training.recurrent_policy import (
    PublicHistoryTracker,
    RecurrentLegalActionAttentionScorerModel,
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
)


def _model() -> RecurrentLegalActionAttentionScorerModel:
    parent = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(
            state_hidden_size=32,
            action_hidden_size=16,
            scorer_hidden_size=24,
            state_depth=1,
            action_depth=1,
            attention_layers=1,
            attention_heads=2,
            attention_feedforward_size=32,
            dropout=0.0,
        )
    )
    return RecurrentLegalActionAttentionScorerModel(parent).eval()


def test_behavior_distribution_has_full_support_and_normalizes() -> None:
    scores = torch.tensor([[2.0, 0.0, -4.0, 9.0]])
    mask = torch.tensor([[True, True, True, False]])
    probabilities = behavior_probabilities(
        scores,
        mask,
        temperature=1.0,
        uniform_mix=0.05,
    )

    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert torch.all(probabilities[0, :3] > 0.0)
    assert probabilities[0, 3] == 0.0


def test_small_collection_balances_cells_and_game_weights() -> None:
    config = RecurrentLeagueConfig(
        games=4,
        maximum_attempts=8,
        validation_divisor=2,
        minimum_validation_games_per_cell=1,
        max_turns=400,
    )
    games, steps, summary = collect_recurrent_league(
        _model(),
        [("greedy", lambda: GreedyHeuristicBot(seed=7))],
        generation=1,
        seed_start=710_000,
        config=config,
    )

    assert len(games) == summary.eligible_games == 4
    assert summary.opponent_seat_games == {
        "greedy": {"seat0": 2, "seat1": 2}
    }
    assert summary.split_opponent_seat_games["train"]["greedy"] == {
        "seat0": 1,
        "seat1": 1,
    }
    assert summary.split_opponent_seat_games["validation"]["greedy"] == {
        "seat0": 1,
        "seat1": 1,
    }
    assert all(step.split in {"train", "validation"} for step in steps)
    for game in games:
        assert math.isclose(
            sum(step.game_weight for step in game.steps),
            1.0,
            abs_tol=1e-9,
        )


def _synthetic_steps(model) -> list[RecurrentLeagueStep]:
    env = SplendorEnv(seed=72)
    state = env.initial_state()
    history = PublicHistoryTracker()
    rows = []
    for _ in range(4):
        actions = env.legal_actions(state)
        policy_input = build_recurrent_policy_input(
            state,
            actions,
            history.events,
        )
        batch = collate_recurrent_policy_inputs([policy_input])
        with torch.no_grad():
            scores, _value = forward_recurrent(model, batch)
            probabilities = behavior_probabilities(
                scores,
                batch["legal_action_mask"],
                temperature=1.0,
                uniform_mix=0.05,
            )[0, : len(actions)]
        action_row = len(rows) % min(len(actions), 2)
        probability_values = tuple(float(value) for value in probabilities)
        rows.append(
            RecurrentLeagueStep(
                policy_input=policy_input,
                action_row=action_row,
                behavior_scores=tuple(float(value) for value in scores[0, : len(actions)]),
                behavior_probabilities=probability_values,
                behavior_log_probability=math.log(
                    probability_values[action_row]
                ),
                game_id=len(rows),
                opponent_label="test",
                actor_seat=len(rows) % 2,
                action_type=actions[action_row].action_type.name,
                split="train" if len(rows) < 2 else "validation",
                advantage=1.0 if len(rows) % 2 == 0 else -1.0,
                game_weight=1.0,
            )
        )
        action = actions[0]
        history.record(state, action)
        state = env.step(state, action)
    return rows


def test_recurrent_update_moves_only_trainable_residual() -> None:
    parent = _model()
    parent_state = {
        name: parameter.detach().clone()
        for name, parameter in parent.parent.named_parameters()
    }
    config = RecurrentLeagueConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        target_validation_kl=1.0,
    )
    candidate, summary = train_recurrent_generation(
        parent,
        _synthetic_steps(parent),
        generation=1,
        config=config,
    )

    assert summary.behavior_reconstruction_max_error <= 1e-5
    assert summary.first_gradient_norm > 0.0
    assert math.isfinite(summary.validation_raw_kl)
    assert all(
        torch.equal(parent_state[name], parameter)
        for name, parameter in candidate.parent.named_parameters()
    )
    assert any(
        not torch.equal(
            dict(parent.named_parameters())[name],
            parameter,
        )
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    )
