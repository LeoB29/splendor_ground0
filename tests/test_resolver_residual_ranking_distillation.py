from __future__ import annotations

import pytest
import torch

from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.resolver_residual import (
    FrozenParentResidualScorerBot,
    FrozenParentResidualScorerModel,
    ResolverResidualConfig,
    load_frozen_parent_residual_model,
    save_frozen_parent_residual_checkpoint,
    trainable_parameter_count,
)
from splendor_ai.training.run_resolver_residual_ranking_distillation import (
    ResolverResidualReplayDataset,
    ResolverResidualExperimentConfig,
    _bootstrap_gain,
    _offline_gates,
    _validate_config,
    collate_resolver_residual_rows,
    compute_residual_ranking_losses,
)


def _small_parent() -> LegalActionAttentionScorerModel:
    torch.manual_seed(53)
    model = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(
            observation_size=7,
            action_feature_size=5,
            state_hidden_size=8,
            action_hidden_size=8,
            scorer_hidden_size=8,
            state_depth=1,
            action_depth=1,
            attention_layers=1,
            attention_heads=2,
            attention_feedforward_size=16,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def test_zero_initialized_residual_exactly_reproduces_frozen_parent() -> None:
    parent = _small_parent()
    model = FrozenParentResidualScorerModel(
        parent,
        ResolverResidualConfig(hidden_size=4),
    )
    observation = torch.randn(3, 7)
    actions = torch.randn(3, 5, 5)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )

    with torch.inference_mode():
        expected_scores, expected_value = parent(observation, actions, mask)
        scores, value, parent_scores, residual = model.forward_with_residual(
            observation,
            actions,
            mask,
        )

    assert torch.equal(scores, expected_scores)
    assert torch.equal(parent_scores, expected_scores)
    assert torch.equal(value, expected_value)
    assert torch.count_nonzero(residual) == 0
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())
    assert trainable_parameter_count(model) == 77


def test_residual_is_centered_over_legal_actions() -> None:
    model = FrozenParentResidualScorerModel(
        _small_parent(),
        ResolverResidualConfig(hidden_size=4),
    )
    with torch.no_grad():
        final = model.residual_head[-1]
        final.weight.fill_(0.2)
        final.bias.fill_(0.3)
    observation = torch.randn(2, 7)
    actions = torch.randn(2, 4, 5)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )

    _scores, _value, _parent, residual = model.forward_with_residual(
        observation,
        actions,
        mask,
    )

    assert float(residual[0, :3].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(residual[1].mean()) == pytest.approx(0.0, abs=1e-6)
    assert residual[0, 3] == 0.0


def test_pairwise_objective_rewards_crossing_resolver_over_parent() -> None:
    parent_scores = torch.tensor([[0.5, 0.0], [0.2, -0.1]])
    mask = torch.ones_like(parent_scores, dtype=torch.bool)
    common = {
        "parent_scores": parent_scores,
        "residual": torch.zeros_like(parent_scores),
        "legal_mask": mask,
        "resolver_rows": torch.tensor([1, 0]),
        "parent_rows": torch.tensor([0, 0]),
        "target_available": torch.tensor([True, False]),
        "ranking_margin": 0.25,
        "anchor_kl_weight": 5.0,
        "residual_mse_weight": 0.01,
    }
    baseline = compute_residual_ranking_losses(
        student_scores=parent_scores.clone().requires_grad_(True),
        **common,
    )
    improved_scores = parent_scores.clone()
    improved_scores[0, 1] = 1.0
    improved = compute_residual_ranking_losses(
        student_scores=improved_scores.requires_grad_(True),
        **common,
    )

    assert improved["ranking"] < baseline["ranking"]
    assert float(baseline["anchor_kl"].detach()) == pytest.approx(0.0)
    baseline["total"].backward()


def _passing_evaluation() -> dict[str, object]:
    return {
        "resolved": {
            "resolver_agreement_gain": 0.08,
            "target_kl_ratio": 0.9,
        },
        "disagreement": {
            "capture": 0.2,
            "directional_movement": 0.9,
        },
        "agreement_anchor": {"retention": 0.99},
        "fallback": {"retention": 0.99, "mean_kl": 0.01},
        "action_movement": 0.12,
        "strata": {
            "phase:early": {"rows": 100, "resolver_agreement_gain": 0.01},
            "phase:mid": {"rows": 100, "resolver_agreement_gain": 0.02},
            "phase:late": {"rows": 100, "resolver_agreement_gain": 0.03},
            "seat:0": {"rows": 100, "resolver_agreement_gain": 0.02},
            "seat:1": {"rows": 100, "resolver_agreement_gain": 0.02},
            "outcome:winner": {"rows": 100, "resolver_agreement_gain": 0.02},
            "outcome:loser": {"rows": 100, "resolver_agreement_gain": 0.02},
        },
        "integrity": {"finite": True, "parent_argmax_reproduced": True},
        "bootstrap_resolver_agreement_gain_ci95": {
            "lower": 0.01,
            "upper": 0.15,
        },
    }


def test_offline_gate_requires_fresh_confirmation_bootstrap() -> None:
    cfg = ResolverResidualExperimentConfig()
    evaluation = _passing_evaluation()
    train = {"disagreement": {"capture": 0.25}}

    validation = _offline_gates(
        cfg,
        evaluation,
        train,
        require_bootstrap=False,
    )
    confirmation = _offline_gates(
        cfg,
        evaluation,
        train,
        require_bootstrap=True,
    )

    assert all(validation.values())
    assert all(confirmation.values())
    evaluation["bootstrap_resolver_agreement_gain_ci95"]["lower"] = 0.0
    failed = _offline_gates(
        cfg,
        evaluation,
        train,
        require_bootstrap=True,
    )
    assert not failed["bootstrap_gain_lower_above_zero"]


def test_game_cluster_bootstrap_is_deterministic() -> None:
    rows = [
        {
            "game_seed": 1,
            "student_agrees_resolver": True,
            "parent_agrees_resolver": False,
        },
        {
            "game_seed": 1,
            "student_agrees_resolver": False,
            "parent_agrees_resolver": False,
        },
        {
            "game_seed": 2,
            "student_agrees_resolver": True,
            "parent_agrees_resolver": True,
        },
    ]

    first = _bootstrap_gain(rows, 200, 53)
    second = _bootstrap_gain(rows, 200, 53)

    assert first == second
    assert first["lower"] <= first["upper"]


def test_residual_checkpoint_is_self_contained_and_bot_loadable(tmp_path) -> None:
    model = FrozenParentResidualScorerModel(
        _small_parent(),
        ResolverResidualConfig(hidden_size=4),
    )
    path = tmp_path / "residual.pt"
    save_frozen_parent_residual_checkpoint(path, model, {"parent_sha256": "test"})

    payload = torch.load(path, map_location="cpu", weights_only=True)
    bot = FrozenParentResidualScorerBot(path)

    assert payload["model_kind"] == "frozen_parent_resolver_residual_scorer"
    assert payload["residual_model_config"] == {"hidden_size": 4}
    assert bot.checkpoint_path == path


def test_053_frozen_artifacts_reconstruct_residual_batch() -> None:
    cfg = ResolverResidualExperimentConfig()
    _validate_config(cfg)
    model = load_frozen_parent_residual_model(
        cfg.parent_checkpoint,
        cfg.device,
        ResolverResidualConfig(hidden_size=cfg.residual_hidden_size),
    )
    dataset = ResolverResidualReplayDataset(cfg.train_replay, model)
    batch = collate_resolver_residual_rows([dataset[0], dataset[1]])

    scores, _value, parent_scores, residual = model.forward_with_residual(
        batch["observation"],
        batch["legal_action_features"],
        batch["legal_action_mask"],
    )

    assert len(dataset) == 4_940
    assert trainable_parameter_count(model) == 24_769
    assert torch.equal(scores, parent_scores)
    assert torch.count_nonzero(residual) == 0
    assert batch["parent_action_row"].shape == (2,)
