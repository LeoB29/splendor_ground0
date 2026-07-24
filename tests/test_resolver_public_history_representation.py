from __future__ import annotations

import copy

import pytest
import torch

from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)
from splendor_ai.training.resolver_history_value import (
    FrozenParentResolverValueModel,
    ResolverHistoryValueConfig,
)
from splendor_ai.training.run_resolver_public_history_representation import (
    ResolverPublicHistoryDataset,
    ResolverPublicHistoryExperimentConfig,
    _decision,
    _history_help_gates,
    _learnability_gates,
    _validate_config,
    aggregate_canonical_group_values,
    collate_resolver_public_history_rows,
    compute_resolver_value_loss,
    paired_game_cluster_bootstrap,
)


def _small_parent() -> LegalActionAttentionScorerModel:
    torch.manual_seed(54)
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


def _history_inputs(batch: int = 2, width: int = 3) -> dict[str, torch.Tensor]:
    return {
        "history_action_type_ids": torch.tensor(
            [[1, 2, 0], [3, 0, 0]],
            dtype=torch.long,
        )[:batch, :width],
        "history_card_ids": torch.tensor(
            [[1, 2, 0], [3, 0, 0]],
            dtype=torch.long,
        )[:batch, :width],
        "history_noble_ids": torch.tensor(
            [[1, 0, 0], [2, 0, 0]],
            dtype=torch.long,
        )[:batch, :width],
        "history_numeric_features": torch.randn(batch, width, 34),
        "history_mask": torch.tensor(
            [[True, True, False], [True, False, False]],
            dtype=torch.bool,
        )[:batch, :width],
    }


def test_snapshot_arm_never_uses_public_history() -> None:
    model = FrozenParentResolverValueModel(
        _small_parent(),
        use_history=False,
        config=ResolverHistoryValueConfig(
            history_hidden_size=6,
            value_hidden_size=7,
        ),
    )
    observation = torch.randn(2, 7)
    actions = torch.randn(2, 4, 5)
    legal_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    first = _history_inputs()
    second = {
        key: torch.zeros_like(value)
        for key, value in first.items()
    }

    with torch.inference_mode():
        first_values, first_parent = model(
            observation,
            actions,
            legal_mask,
            **first,
        )
        second_values, second_parent = model(
            observation,
            actions,
            legal_mask,
            **second,
        )

    assert torch.equal(first_values, second_values)
    assert torch.equal(first_parent, second_parent)
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())


def test_history_encoder_maps_empty_to_zero_and_nonempty_to_context() -> None:
    model = FrozenParentResolverValueModel(
        _small_parent(),
        use_history=True,
        config=ResolverHistoryValueConfig(
            history_hidden_size=6,
            value_hidden_size=7,
        ),
    )
    inputs = _history_inputs()
    inputs["history_mask"][1] = False

    context = model.encode_history(
        inputs["history_action_type_ids"],
        inputs["history_card_ids"],
        inputs["history_noble_ids"],
        inputs["history_numeric_features"],
        inputs["history_mask"],
    )

    assert context.shape == (2, 6)
    assert torch.count_nonzero(context[0]) > 0
    assert torch.count_nonzero(context[1]) == 0


def test_matched_arms_have_identical_added_initialization() -> None:
    parent = _small_parent()
    second_parent = copy.deepcopy(parent)
    torch.manual_seed(540)
    snapshot = FrozenParentResolverValueModel(
        parent,
        use_history=False,
    )
    torch.manual_seed(540)
    history = FrozenParentResolverValueModel(
        second_parent,
        use_history=True,
    )

    snapshot_added = {
        name: value
        for name, value in snapshot.state_dict().items()
        if not name.startswith("parent.")
    }
    history_added = {
        name: value
        for name, value in history.state_dict().items()
        if not name.startswith("parent.")
    }

    assert snapshot_added.keys() == history_added.keys()
    assert all(
        torch.equal(snapshot_added[name], history_added[name])
        for name in snapshot_added
    )


def test_group_aliases_are_meaned_and_ranking_loss_rewards_order() -> None:
    action_values = torch.tensor([[1.0, 3.0, -1.0, 0.0, 2.0]])
    membership = torch.zeros((1, 4, 5), dtype=torch.bool)
    membership[0, 0, [0, 1]] = True
    membership[0, 1, 2] = True
    membership[0, 2, 3] = True
    membership[0, 3, 4] = True
    predictions = aggregate_canonical_group_values(action_values, membership)
    targets = torch.tensor([[2.0, -1.0, 0.0, 2.0]])
    bad = predictions.flip(dims=[1]).requires_grad_(True)
    good = predictions.clone().requires_grad_(True)

    good_loss = compute_resolver_value_loss(
        good,
        targets,
        beta=0.1,
        ranking_weight=0.5,
    )
    bad_loss = compute_resolver_value_loss(
        bad,
        targets,
        beta=0.1,
        ranking_weight=0.5,
    )

    assert predictions.tolist() == [[2.0, -1.0, 0.0, 2.0]]
    assert good_loss["total"] < bad_loss["total"]
    good_loss["total"].backward()


def test_paired_game_bootstrap_is_clustered_and_deterministic() -> None:
    snapshot = [
        {"game_seed": 1, "turn_index": 0, "best_correct": False},
        {"game_seed": 1, "turn_index": 1, "best_correct": True},
        {"game_seed": 2, "turn_index": 0, "best_correct": False},
    ]
    history = [
        {"game_seed": 1, "turn_index": 0, "best_correct": True},
        {"game_seed": 1, "turn_index": 1, "best_correct": True},
        {"game_seed": 2, "turn_index": 0, "best_correct": False},
    ]

    first = paired_game_cluster_bootstrap(snapshot, history, samples=200, seed=54)
    second = paired_game_cluster_bootstrap(snapshot, history, samples=200, seed=54)

    assert first == second
    assert first["mean"] == pytest.approx(1.0 / 3.0)
    assert first["lower"] <= first["upper"]


def _metrics(accuracy: float) -> dict[str, object]:
    return {
        "best_group_accuracy": accuracy,
        "mean_within_state_spearman": 0.40,
        "pairwise_sign_accuracy": 0.70,
        "resolver_parent_disagreement": {"capture": accuracy},
        "strata": {
            name: {"rows": 100, "best_group_accuracy": accuracy}
            for name in (
                "phase:early",
                "phase:mid",
                "phase:late",
                "seat:0",
                "seat:1",
                "outcome:winner",
                "outcome:loser",
            )
        },
        "integrity": {
            "finite": True,
            "four_values_per_row": True,
            "non_tied_pairs_present": True,
        },
    }


def test_fixed_decision_priority_prefers_material_history() -> None:
    cfg = ResolverPublicHistoryExperimentConfig()
    snapshot_train = _metrics(0.65)
    history_train = _metrics(0.70)
    snapshot_validation = _metrics(0.42)
    history_validation = _metrics(0.50)
    history_validation["mean_within_state_spearman"] = 0.46
    arms = {
        "snapshot": {"train": snapshot_train, "validation": snapshot_validation},
        "history": {"train": history_train, "validation": history_validation},
    }
    learnability = {
        "snapshot": _learnability_gates(
            cfg,
            snapshot_train,
            snapshot_validation,
        ),
        "history": _learnability_gates(
            cfg,
            history_train,
            history_validation,
        ),
    }
    history_help = _history_help_gates(
        cfg,
        snapshot_validation,
        history_validation,
        learnability["history"],
        {"mean": 0.08, "lower": 0.01, "upper": 0.15},
    )

    assert all(learnability["history"].values())
    assert all(history_help.values())
    assert _decision(cfg, arms, learnability, history_help).startswith(
        "history_representation_promising"
    )


def test_054_frozen_artifacts_reconstruct_information_legal_rows() -> None:
    cfg = ResolverPublicHistoryExperimentConfig()
    _validate_config(cfg)
    from splendor_ai.training.resolver_history_value import (
        load_frozen_parent_resolver_value_model,
    )

    model = load_frozen_parent_resolver_value_model(
        cfg.parent_checkpoint,
        use_history=False,
    )
    dataset = ResolverPublicHistoryDataset(cfg.train_replay, model)
    batch = collate_resolver_public_history_rows([dataset[0], dataset[1]])

    assert len(dataset) == 4_397
    assert all(dataset.integrity.values())
    assert batch["group_values"].shape == (2, 4)
    assert batch["group_membership"].shape[1] == 4
    assert int(batch["history_mask"][0].sum()) == dataset[0].metadata["turn_index"]
