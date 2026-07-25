from __future__ import annotations

import copy
import math

import torch

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.recurrent_league import (
    RecurrentLeagueStep,
    behavior_probabilities,
    forward_recurrent,
)
from splendor_ai.training.run_recurrent_generation2_diagnosis import (
    _bootstrap_mean,
    _empty_history_steps,
    _parameter_direction,
    _summarize_records,
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


def _synthetic_steps(model) -> list[RecurrentLeagueStep]:
    env = SplendorEnv(seed=73)
    state = env.initial_state()
    history = PublicHistoryTracker()
    steps = []
    for index in range(2):
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
        probability_values = tuple(float(value) for value in probabilities)
        steps.append(
            RecurrentLeagueStep(
                policy_input=policy_input,
                action_row=0,
                behavior_scores=tuple(float(value) for value in scores[0, : len(actions)]),
                behavior_probabilities=probability_values,
                behavior_log_probability=math.log(probability_values[0]),
                game_id=index,
                opponent_label="test",
                actor_seat=index % 2,
                action_type=actions[0].action_type.name,
                advantage=float(index * 2 - 1),
                game_weight=1.0,
            )
        )
        history.record(state, actions[0])
        state = env.step(state, actions[0])
    return steps


def test_record_summary_is_game_weighted() -> None:
    records = [
        {
            "game_id": 0,
            "weight": 0.5,
            "contribution": 0.1,
            "kl_numerator": 0.01,
            "tv_numerator": 0.02,
            "argmax_changed": False,
        },
        {
            "game_id": 0,
            "weight": 0.5,
            "contribution": 0.2,
            "kl_numerator": 0.02,
            "tv_numerator": 0.03,
            "argmax_changed": True,
        },
        {
            "game_id": 1,
            "weight": 1.0,
            "contribution": -0.1,
            "kl_numerator": 0.01,
            "tv_numerator": 0.01,
            "argmax_changed": False,
        },
    ]

    summary = _summarize_records(records)

    assert summary["games"] == 2
    assert summary["weight"] == 2.0
    assert abs(float(summary["surrogate"]) - 0.1) < 1e-12
    assert abs(float(summary["argmax_movement"]) - 0.25) < 1e-12
    assert summary["positive_game_fraction"] == 0.5


def test_bootstrap_is_deterministic() -> None:
    first = _bootstrap_mean([1.0, -1.0, 0.5], samples=500, seed=63)
    second = _bootstrap_mean([1.0, -1.0, 0.5], samples=500, seed=63)

    assert first == second
    assert first["lower"] <= first["mean"] <= first["upper"]


def test_empty_history_ablation_preserves_other_step_fields() -> None:
    model = _model()
    steps = _synthetic_steps(model)
    empty = _empty_history_steps(steps)

    assert all(not step.policy_input.history.action_type_ids for step in empty)
    assert [step.action_row for step in empty] == [
        step.action_row for step in steps
    ]
    assert [step.advantage for step in empty] == [
        step.advantage for step in steps
    ]


def test_parameter_direction_detects_reversal() -> None:
    epoch0 = _model()
    generation1 = copy.deepcopy(epoch0)
    generation2 = copy.deepcopy(epoch0)
    for name, parameter in generation1.named_parameters():
        if parameter.requires_grad:
            parameter.data.add_(0.01)
    for name, parameter in generation2.named_parameters():
        if parameter.requires_grad:
            parameter.data.sub_(0.01)

    direction = _parameter_direction(epoch0, generation1, generation2)

    assert direction["all"]["cosine"] < -0.99
    assert direction["history"]["cosine"] < -0.99
    assert direction["residual"]["cosine"] < -0.99
