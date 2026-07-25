from __future__ import annotations

from dataclasses import asdict

import torch

from splendor_ai.cfr.full_game import encode_public_history
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.match import play_game
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.recurrent_policy import (
    PublicHistoryTracker,
    RecurrentLegalActionAttentionScorerModel,
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
    save_recurrent_policy_checkpoint,
)


def _small_parent() -> LegalActionAttentionScorerModel:
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


def _trajectory_rows(count: int = 4):
    env = SplendorEnv(seed=61)
    state = env.initial_state()
    history = PublicHistoryTracker()
    rows = []
    for _ in range(count):
        actions = env.legal_actions(state)
        rows.append(build_recurrent_policy_input(state, actions, history.events))
        action = actions[0]
        history.record(state, action)
        state = env.step(state, action)
    return rows, history


def _forward(model, batch):
    return model(
        batch["observation"],
        batch["legal_action_features"],
        batch["legal_action_mask"],
        batch["history_action_type_ids"],
        batch["history_card_ids"],
        batch["history_noble_ids"],
        batch["history_numeric_features"],
        batch["history_mask"],
    )


def test_zero_residual_exactly_reproduces_parent_and_freezes_it() -> None:
    rows, _history = _trajectory_rows()
    batch = collate_recurrent_policy_inputs(rows)
    parent = _small_parent()
    recurrent = RecurrentLegalActionAttentionScorerModel(parent).eval()

    with torch.no_grad():
        parent_scores, parent_value = parent(
            batch["observation"],
            batch["legal_action_features"],
            batch["legal_action_mask"],
        )
        recurrent_scores, recurrent_value = _forward(recurrent, batch)

    assert torch.equal(recurrent_scores, parent_scores)
    assert torch.equal(recurrent_value, parent_value)
    assert torch.count_nonzero(recurrent.residual_output.weight) == 0
    assert torch.count_nonzero(recurrent.residual_output.bias) == 0
    assert all(not parameter.requires_grad for parameter in recurrent.parent.parameters())
    assert any(
        parameter.requires_grad
        for name, parameter in recurrent.named_parameters()
        if not name.startswith("parent.")
    )


def test_full_and_stepwise_public_history_contexts_match() -> None:
    _rows, tracker = _trajectory_rows()
    encoding = encode_public_history(tracker.events, observer=0)
    row, _ = _trajectory_rows(count=1)
    row_with_history = type(row[0])(
        observation=row[0].observation,
        legal_action_features=row[0].legal_action_features,
        history=encoding,
        player=0,
        turn=len(tracker.events),
    )
    batch = collate_recurrent_policy_inputs([row_with_history])
    model = RecurrentLegalActionAttentionScorerModel(_small_parent()).eval()

    with torch.no_grad():
        full = model.encode_history(
            batch["history_action_type_ids"],
            batch["history_card_ids"],
            batch["history_noble_ids"],
            batch["history_numeric_features"],
            batch["history_mask"],
        )
        hidden = None
        stepped = None
        for index in range(len(encoding.action_type_ids)):
            stepped, hidden = model.step_history(
                batch["history_action_type_ids"][:, index],
                batch["history_card_ids"][:, index],
                batch["history_noble_ids"][:, index],
                batch["history_numeric_features"][:, index],
                hidden,
            )

    assert stepped is not None
    assert torch.allclose(full, stepped, atol=1e-6, rtol=0.0)


def test_recurrent_checkpoint_round_trip_is_exact(tmp_path) -> None:
    rows, _history = _trajectory_rows()
    batch = collate_recurrent_policy_inputs(rows)
    model = RecurrentLegalActionAttentionScorerModel(_small_parent()).eval()
    checkpoint = tmp_path / "recurrent.pt"
    provenance = {"parent_path": "parent.pt", "parent_sha256": "abc"}
    save_recurrent_policy_checkpoint(checkpoint, model, provenance)
    reloaded = load_recurrent_policy_checkpoint(checkpoint).eval()

    with torch.no_grad():
        expected = _forward(model, batch)
        actual = _forward(reloaded, batch)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["provenance"] == provenance
    assert payload["parent_model_config"] == asdict(model.parent_config)


def test_tracker_reset_and_state_round_trip() -> None:
    _rows, tracker = _trajectory_rows(count=2)
    payload = tracker.state_dict()
    restored = PublicHistoryTracker()
    restored.load_state_dict(payload)
    assert restored.events == tracker.events
    restored.reset()
    assert restored.events == ()


class _LifecycleBot:
    def __init__(self) -> None:
        self.starts: list[int] = []
        self.events = []

    def start_game(self, seed: int) -> None:
        self.starts.append(seed)
        self.events.clear()

    def observe_action(self, before, action, after) -> None:
        self.events.append((before.turn_index, action, after.turn_index))

    def choose_action(self, env, state, legal_actions=None):
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        return actions[0]


def test_match_lifecycle_hooks_are_called_once_for_shared_bot() -> None:
    bot = _LifecycleBot()
    result = play_game(bot, bot, seed=610, max_turns=3)

    assert result.termination_reason == "max_turns"
    assert bot.starts == [610]
    assert len(bot.events) == 3
    assert [event[0] for event in bot.events] == [0, 1, 2]
    assert [event[2] for event in bot.events] == [1, 2, 3]
