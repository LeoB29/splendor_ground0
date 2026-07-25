from __future__ import annotations

from dataclasses import replace
import math

import torch

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.recurrent_league import (
    RecurrentLeagueConfig,
    RecurrentLeagueGame,
    RecurrentLeagueStep,
    _assign_returns_and_advantages,
    forward_recurrent,
)
from splendor_ai.training.recurrent_policy import (
    PublicHistoryTracker,
    RecurrentLegalActionAttentionScorerModel,
    RecurrentPolicyConfig,
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
    save_recurrent_policy_checkpoint,
)
from splendor_ai.training.run_history_required_recurrent_repair import (
    _rotated_history_steps,
)


def _parent() -> LegalActionAttentionScorerModel:
    return LegalActionAttentionScorerModel(
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
    ).eval()


def _rows(seed: int = 640, count: int = 4):
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    history = PublicHistoryTracker()
    rows = []
    for _ in range(count):
        actions = env.legal_actions(state)
        rows.append(build_recurrent_policy_input(state, actions, history.events))
        action = actions[0]
        history.record(state, action)
        state = env.step(state, action)
    return rows


def _row_after_action(seed: int, action_row: int):
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    history = PublicHistoryTracker()
    actions = env.legal_actions(state)
    action = actions[action_row]
    history.record(state, action)
    state = env.step(state, action)
    return build_recurrent_policy_input(
        state,
        env.legal_actions(state),
        history.events,
    )


def _history_required_model():
    return RecurrentLegalActionAttentionScorerModel(
        _parent(),
        RecurrentPolicyConfig(residual_mode="history_required_interaction"),
    ).eval()


def test_history_required_residual_is_exact_at_epoch_zero() -> None:
    model = _history_required_model()
    batch = collate_recurrent_policy_inputs(_rows())

    with torch.no_grad():
        scores, value = forward_recurrent(model, batch)
        parent_scores, parent_value = model.parent(
            batch["observation"],
            batch["legal_action_features"],
            batch["legal_action_mask"],
        )

    assert torch.equal(scores, parent_scores)
    assert torch.equal(value, parent_value)
    assert torch.count_nonzero(model.required_output.weight) == 0
    assert model.required_history_projection.bias is None
    assert model.required_output.bias is None


def test_empty_history_fallback_remains_exact_after_residual_moves() -> None:
    rows = _rows()
    model = _history_required_model()
    with torch.no_grad():
        model.required_output.weight.fill_(0.1)

    nonempty = collate_recurrent_policy_inputs(rows[1:])
    empty_history = rows[0].history
    emptied = collate_recurrent_policy_inputs(
        [replace(row, history=empty_history) for row in rows[1:]]
    )
    with torch.no_grad():
        moved_scores, _ = forward_recurrent(model, nonempty)
        moved_parent, _ = model.parent(
            nonempty["observation"],
            nonempty["legal_action_features"],
            nonempty["legal_action_mask"],
        )
        empty_scores, empty_value = forward_recurrent(model, emptied)
        empty_parent, empty_parent_value = model.parent(
            emptied["observation"],
            emptied["legal_action_features"],
            emptied["legal_action_mask"],
        )

    assert not torch.equal(moved_scores, moved_parent)
    assert torch.equal(empty_scores, empty_parent)
    assert torch.equal(empty_value, empty_parent_value)


def test_history_required_checkpoint_round_trip(tmp_path) -> None:
    model = _history_required_model()
    with torch.no_grad():
        model.required_output.weight.fill_(0.05)
    batch = collate_recurrent_policy_inputs(_rows())
    path = tmp_path / "history_required.pt"
    save_recurrent_policy_checkpoint(path, model, {"experiment": "test"})
    reloaded = load_recurrent_policy_checkpoint(path).eval()

    with torch.no_grad():
        expected = forward_recurrent(model, batch)
        actual = forward_recurrent(reloaded, batch)

    assert reloaded.config.residual_mode == "history_required_interaction"
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def _step(row, *, game_id: int, turn: int) -> RecurrentLeagueStep:
    row = replace(row, turn=turn)
    return RecurrentLeagueStep(
        policy_input=row,
        action_row=0,
        behavior_scores=(0.0,),
        behavior_probabilities=(1.0,),
        behavior_log_probability=0.0,
        game_id=game_id,
        opponent_label="test",
        actor_seat=0,
        action_type="PASS",
        split="train",
    )


def test_phase_weights_reduce_early_credit_and_sum_per_game() -> None:
    base = _rows(count=1)[0]
    steps = [
        _step(base, game_id=0, turn=0),
        _step(base, game_id=0, turn=20),
        _step(base, game_id=0, turn=50),
    ]
    game = RecurrentLeagueGame(
        game_id=0,
        seed=1,
        opponent_label="test",
        actor_seat=0,
        cell_ordinal=0,
        termination_reason="completed",
        turns=51,
        winner=0,
        final_scores=(15, 10),
        steps=steps,
        split="train",
    )
    config = RecurrentLeagueConfig(
        early_turn_weight=0.25,
        mid_turn_weight=1.0,
        late_turn_weight=1.0,
    )

    _assign_returns_and_advantages([game], config)

    assert math.isclose(sum(step.game_weight for step in steps), 1.0)
    assert math.isclose(steps[0].game_weight, 1.0 / 9.0)
    assert math.isclose(steps[1].game_weight, 4.0 / 9.0)
    assert math.isclose(steps[2].game_weight, 4.0 / 9.0)


def test_history_rotation_stays_within_exact_stratum() -> None:
    row_a = _row_after_action(seed=641, action_row=0)
    row_b = _row_after_action(seed=642, action_row=-1)
    steps = [
        _step(row_a, game_id=0, turn=1),
        _step(row_b, game_id=1, turn=1),
    ]

    rotated, changed = _rotated_history_steps(steps)

    assert changed == 2
    assert rotated[0].policy_input.history == row_b.history
    assert rotated[1].policy_input.history == row_a.history
    assert rotated[0].opponent_label == steps[0].opponent_label
    assert rotated[0].actor_seat == steps[0].actor_seat
    assert rotated[0].policy_input.turn == steps[0].policy_input.turn
    assert len(rotated[0].policy_input.history.action_type_ids) == len(
        steps[0].policy_input.history.action_type_ids
    )
