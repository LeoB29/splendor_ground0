from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.training.replay import serialize_action, serialize_state_snapshot
from splendor_ai.training.run_action_preference_corpus import (
    _split_quality,
    assigned_phase_for_game,
    export_resolved_preference_rows,
)
from splendor_ai.training.run_action_preference_training import (
    evaluate_preference_model,
)
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
    select_canonical_loop_fallback,
)
from splendor_ai.training.run_action_preference_linear_probe import (
    binary_auc,
    binary_metrics,
    stratified_fold_indices,
)
from splendor_ai.training.run_policy_search_disagreement import ModelActionScore


def test_phase_assignment_cycles_inside_each_opponent_seat_cell() -> None:
    assert {assigned_phase_for_game(index) for index in range(8)} == {"early"}
    assert {assigned_phase_for_game(index) for index in range(8, 16)} == {"mid"}
    assert {assigned_phase_for_game(index) for index in range(16, 24)} == {"late"}
    assert assigned_phase_for_game(24) == "early"


def test_split_quality_counts_only_resolved_preference_directions() -> None:
    summary = _split_quality(
        [
            {"resolved": True, "candidate_minus_parent_full": 0.2},
            {"resolved": True, "candidate_minus_parent_full": -0.1},
            {"resolved": False, "candidate_minus_parent_full": 0.5},
        ]
    )
    assert summary["resolved_states"] == 2
    assert summary["challenger_preferred_fraction"] == 0.5


def test_resolved_export_encodes_winner_as_target_and_loser_as_competitor(tmp_path) -> None:
    env = SplendorEnv(seed=17)
    codec = ActionCodec()
    state = env.initial_state()
    actions = env.legal_actions(state)
    first = codec.encode(state, actions[0])
    second = codec.encode(state, actions[1])
    selection = {
        "source_game_seed": 17,
        "turn_index": 0,
        "actor_seat": 0,
        "candidate_action_index": second,
        "parent_action_index": first,
        "state_snapshot": serialize_state_snapshot(state),
        "split": "train",
        "opponent": "035",
        "phase": "early",
    }
    config = SimpleNamespace(data_dir=tmp_path)
    paths = export_resolved_preference_rows(
        config,
        [selection],
        [{"resolved": True, "candidate_minus_parent_full": 0.2}],
    )
    payload = json.loads(paths["train"].read_text(encoding="utf-8"))
    assert payload["action_index"] == second
    assert payload["model_top_action_index"] == first
    assert payload["preference_challenger_better"] is True
    assert payload["action_payload"] == serialize_action(actions[1])
    assert paths["validation"].read_text(encoding="utf-8") == ""


class _Scores(torch.nn.Module):
    def __init__(self, rows: list[list[float]]) -> None:
        super().__init__()
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.float32))

    def forward(self, observation, features, mask):
        return self.rows[: observation.shape[0]], torch.zeros(observation.shape[0])


def test_preference_metrics_report_challenger_and_parent_accuracy() -> None:
    batch = {
        "observation": torch.zeros((2, 1)),
        "legal_action_features": torch.zeros((2, 2, 1)),
        "legal_action_mask": torch.ones((2, 2), dtype=torch.bool),
        "target_action_row": torch.tensor([1, 0]),
        "preference_competitor_row": torch.tensor([0, 1]),
    }
    parent = _Scores([[2.0, 1.0], [2.0, 1.0]])
    candidate = _Scores([[1.0, 2.0], [2.0, 1.0]])
    metrics = evaluate_preference_model(
        candidate, parent, [batch], "cpu", challenger_flags=[True, False]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["challenger_best_accuracy"] == 1.0
    assert metrics["parent_best_accuracy"] == 1.0
    assert metrics["raw_kl"] > 0.0


def test_canonical_loop_fallback_prioritizes_buy(monkeypatch) -> None:
    take = Action(ActionType.TAKE_TOKENS, metadata=(("key", "take"),))
    buy = Action(ActionType.BUY_VISIBLE, tier=1, market_index=0, metadata=(("key", "buy"),))
    monkeypatch.setattr(
        "splendor_ai.training.run_action_preference_repair.canonical_successor_action_key",
        lambda _state, action: action.metadata,
    )
    scores = (
        ModelActionScore(take, 1, 10.0),
        ModelActionScore(buy, 2, 1.0),
    )
    assert select_canonical_loop_fallback(None, scores, {}) == buy  # type: ignore[arg-type]


def test_canonical_loop_fallback_uses_least_used_successor(monkeypatch) -> None:
    first = Action(ActionType.TAKE_TOKENS, metadata=(("key", "first"),))
    second = Action(ActionType.RESERVE_DECK, tier=1, metadata=(("key", "second"),))
    monkeypatch.setattr(
        "splendor_ai.training.run_action_preference_repair.canonical_successor_action_key",
        lambda _state, action: action.metadata,
    )
    scores = (
        ModelActionScore(first, 1, 10.0),
        ModelActionScore(second, 2, 2.0),
    )
    assert select_canonical_loop_fallback(
        None, scores, {first.metadata: 3, second.metadata: 0}  # type: ignore[arg-type]
    ) == second


@dataclass(frozen=True)
class _FakePlayer:
    score: int


@dataclass(frozen=True)
class _FakeState:
    turn_index: int
    terminal: bool = False
    players: tuple[_FakePlayer, _FakePlayer] = (_FakePlayer(2), _FakePlayer(0))


class _CyclingEnv:
    def __init__(self, action: Action) -> None:
        self.action = action

    def step(self, state, action):
        return _FakeState(state.turn_index + 1)

    def legal_actions(self, state):
        return [self.action]


class _CyclingScorer:
    def __init__(self, action: Action) -> None:
        self.action = action

    def score_actions(self, state, actions):
        return (ModelActionScore(self.action, 1, 1.0),)


def test_loop_safe_rollout_adjudicates_after_bounded_fallback(monkeypatch) -> None:
    action = Action(ActionType.PASS)
    monkeypatch.setattr(
        "splendor_ai.training.run_action_preference_repair.state_signature",
        lambda state: ("cycle",),
    )
    monkeypatch.setattr(
        "splendor_ai.training.run_action_preference_repair.is_progress_transition",
        lambda state, next_state, selected: False,
    )
    monkeypatch.setattr(
        "splendor_ai.training.run_action_preference_repair.canonical_successor_action_key",
        lambda state, selected: ("same",),
    )
    result = rollout_score_margin_loop_safe(
        _CyclingEnv(action),  # type: ignore[arg-type]
        _CyclingScorer(action),
        _FakeState(0),  # type: ignore[arg-type]
        action,
        root_player=0,
        seed=1,
        epsilon=0.1,
        temperature=0.5,
        top_k=3,
        max_additional_turns=40,
        score_margin_scale=20.0,
        max_loop_fallback_triggers=2,
    )
    assert result.resolved
    assert result.loop_adjudicated
    assert result.fallback_triggers == 2
    assert not result.hard_cap_reached


def test_binary_probe_metrics_recognize_perfect_ranking() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(labels, probabilities)
    assert metrics["auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert binary_auc(labels.bool(), 1.0 - probabilities) == 0.0


def test_probe_folds_cover_rows_once_and_stratify_cells() -> None:
    metadata = []
    for label in (False, True):
        for phase in ("early", "mid"):
            for offset in range(10):
                metadata.append(
                    {
                        "preference_challenger_better": label,
                        "phase": phase,
                        "opponent_label": "035",
                        "player_id": offset % 2,
                    }
                )
    folds = stratified_fold_indices(metadata, folds=5, seed=3)
    assert sorted(index for fold in folds for index in fold) == list(range(len(metadata)))
    assert max(len(fold) for fold in folds) - min(len(fold) for fold in folds) <= 1
