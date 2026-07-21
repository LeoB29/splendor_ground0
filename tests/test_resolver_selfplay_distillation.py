from __future__ import annotations

import json

import pytest
import torch

from splendor_ai.cfr import TwoPlyResolution
from splendor_ai.encoding import ActionCodec, canonical_successor_action_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.dataset import LegalActionReplayDataset
from splendor_ai.training.legal_action_supervised import (
    compute_legal_action_supervised_losses,
)
from splendor_ai.training.run_policy_search_disagreement import ModelActionScore
from splendor_ai.training.run_resolver_selfplay_distillation import (
    ResolverDistillationConfig,
    _validate_config,
    build_recording_row,
)


def _state_scores():
    env = SplendorEnv(seed=610_100)
    state = env.initial_state()
    actions = env.legal_actions(state)
    scores = tuple(
        ModelActionScore(action=action, action_index=index, score=-index / 10.0)
        for index, action in enumerate(actions)
    )
    return state, actions, scores


def _resolution(state, actions) -> TwoPlyResolution:
    root_actions = tuple(canonical_successor_action_key(state, action) for action in actions[:4])
    return TwoPlyResolution(
        root_actions=root_actions,
        action_values=(0.30, 0.10, -0.05, -0.10),
        regrets_from_best=(0.0, -0.20, -0.35, -0.40),
        best_index=0,
        opponent_choices=(),
    )


def test_resolved_recording_target_covers_every_legal_action() -> None:
    cfg = ResolverDistillationConfig()
    state, actions, scores = _state_scores()
    row = build_recording_row(
        cfg,
        ActionCodec(),
        610_100,
        state,
        actions,
        scores,
        actions[0],
        actions[0],
        0,
        resolution=_resolution(state, actions),
        fallback_reason=None,
    )

    distribution = row["resolver_policy_distribution"]
    assert row["target_available"]
    assert row["game_seed"] == 610_100
    assert len(distribution) == len(actions)
    assert len(row["resolver_action_values"]) == 4
    assert sum(item["probability"] for item in distribution) == pytest.approx(1.0)
    assert all(item["probability"] > 0.0 for item in distribution)


def test_fallback_recording_is_exact_parent_anchor() -> None:
    cfg = ResolverDistillationConfig()
    state, actions, scores = _state_scores()
    row = build_recording_row(
        cfg,
        ActionCodec(),
        610_100,
        state,
        actions,
        scores,
        actions[0],
        actions[0],
        0,
        resolution=None,
        fallback_reason="fewer_than_four_root_actions",
    )

    assert not row["target_available"]
    assert row["training_weight"] == 0.25
    assert row["resolver_action_values"] == []
    assert all(
        item["probability"] == pytest.approx(item["parent_probability"])
        for item in row["resolver_policy_distribution"]
    )


def test_dataset_loads_resolver_policy_distribution(tmp_path) -> None:
    cfg = ResolverDistillationConfig()
    state, actions, scores = _state_scores()
    row = build_recording_row(
        cfg,
        ActionCodec(),
        610_100,
        state,
        actions,
        scores,
        actions[0],
        actions[0],
        0,
        resolution=_resolution(state, actions),
        fallback_reason=None,
    )
    row["final_value"] = 1.0
    path = tmp_path / "targets.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    sample = LegalActionReplayDataset(path)[0]
    assert sample.root_visit_mask
    assert float(sample.root_visit_target.sum()) == pytest.approx(1.0)


def test_distribution_only_loss_can_disable_hard_policy_and_value() -> None:
    scores = torch.tensor([[1.0, 0.0]], requires_grad=True)
    policy, value, total = compute_legal_action_supervised_losses(
        action_scores=scores,
        value_pred=torch.tensor([0.5], requires_grad=True),
        target_action_row=torch.tensor([0]),
        legal_action_mask=torch.tensor([[True, True]]),
        value_target=torch.tensor([1.0]),
        value_loss_weight=0.0,
        root_visit_target=torch.tensor([[0.2, 0.8]]),
        root_visit_mask=torch.tensor([True]),
        root_visit_loss_weight=1.0,
        policy_loss_weight=0.0,
    )

    assert float(policy) > 0.0
    assert float(value) > 0.0
    assert float(total) > 0.0
    total.backward()
    assert scores.grad is not None


def test_052_prerequisites_and_defaults_are_frozen() -> None:
    cfg = ResolverDistillationConfig()
    _validate_config(cfg)
    assert cfg.resolver_temperature == 0.10
    assert cfg.resolver_blend == 0.75
    assert cfg.epochs == 4
    assert cfg.device == "cpu"
