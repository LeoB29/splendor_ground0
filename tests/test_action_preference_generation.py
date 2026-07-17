from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import serialize_action, serialize_state_snapshot
from splendor_ai.training.run_action_preference_corpus import (
    _split_quality,
    assigned_phase_for_game,
    export_resolved_preference_rows,
)
from splendor_ai.training.run_action_preference_training import (
    evaluate_preference_model,
)


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
