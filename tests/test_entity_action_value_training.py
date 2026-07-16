from __future__ import annotations

import json
from dataclasses import asdict

import torch

from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_entity_action_value_training import (
    EntityQConfig,
    EntityQDataset,
    _ranking_gate,
    build_parser as build_q_parser,
    collate_entity_q_samples,
    train_entity_action_value,
)
from splendor_ai.training.run_entity_late_counterfactual_corpus import (
    build_parser as build_corpus_parser,
)
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)


def test_entity_q_dataset_aligns_sparse_counterfactual_actions() -> None:
    state = SplendorEnv(seed=3).initial_state()
    legal_actions = tuple(
        actions_by_canonical_successor_key(
            state, SplendorEnv().legal_actions(state)
        ).values()
    )
    codec = ActionCodec()
    evaluated = legal_actions[:3]
    dataset = EntityQDataset(
        [
            {
                "state_snapshot": serialize_state_snapshot(state),
                "actions": [
                    {
                        "action_index": codec.encode(state, action),
                        "full_mean": 0.2 - index * 0.1,
                        "prior_score": 1.0 - index,
                    }
                    for index, action in enumerate(evaluated)
                ],
            }
        ]
    )

    sample = dataset[0]
    batch = collate_entity_q_samples([sample])

    assert int(sample.target_mask.sum()) == 3
    assert int(batch["target_mask"].sum()) == 3
    assert batch["target_values"].shape == batch["action_mask"].shape
    assert batch["prior_scores"].shape == batch["action_mask"].shape


def test_entity_q_gate_requires_both_regret_and_ranking_confidence() -> None:
    metrics = {
        "q_minus_parent_regret_ci95": [-0.2, -0.01],
        "q_minus_parent_spearman_ci95": [0.01, 0.4],
        "q_minus_parent_top1": 0.1,
    }

    assert _ranking_gate(metrics)["passed"]

    metrics["q_minus_parent_regret_ci95"] = [-0.2, 0.01]
    assert not _ranking_gate(metrics)["passed"]


def test_late_corpus_and_q_defaults_are_preregistered() -> None:
    corpus_args = build_corpus_parser().parse_args(
        [
            "--parent-checkpoint",
            "parent.pt",
            "--checkpoint-016",
            "016.pt",
            "--checkpoint-035",
            "035.pt",
            "--output-path",
            "corpus.json",
        ]
    )
    q_args = build_q_parser().parse_args(
        [
            "--corpus-path",
            "corpus.json",
            "--parent-checkpoint",
            "parent.pt",
            "--output-dir",
            "q-output",
        ]
    )

    assert corpus_args.rollout_games == 240
    assert corpus_args.late_states == 240
    assert (corpus_args.train_states, corpus_args.validation_states) == (160, 40)
    assert corpus_args.confirmation_states == 40
    assert corpus_args.determinizations == 3
    assert corpus_args.continuations_per_determinization == 8
    assert corpus_args.workers == 4
    assert q_args.epochs == 30
    assert q_args.pairwise_loss_weight == 0.5
    assert q_args.minimum_target_gap == 0.05


def test_entity_q_training_smoke_uses_untouched_confirmation(tmp_path) -> None:
    model = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=16,
            action_hidden_size=16,
            scorer_hidden_size=16,
            entity_layers=1,
            action_layers=1,
            attention_heads=2,
            attention_feedforward_size=32,
            id_embedding_size=4,
        )
    )
    parent = tmp_path / "parent.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        parent,
    )
    rows = []
    for seed, split in enumerate(("train", "validation", "confirmation"), start=10):
        state = SplendorEnv(seed=seed).initial_state()
        legal_actions = tuple(
            actions_by_canonical_successor_key(
                state, SplendorEnv().legal_actions(state)
            ).values()
        )
        codec = ActionCodec()
        rows.append(
            {
                "split": split,
                "state_snapshot": serialize_state_snapshot(state),
                "actions": [
                    {
                        "action_index": codec.encode(state, action),
                        "full_mean": 0.2 - index * 0.1,
                        "prior_score": 1.0 - index,
                    }
                    for index, action in enumerate(legal_actions[:3])
                ],
            }
        )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps({"all_gates_passed": True, "states": rows}), encoding="utf-8"
    )

    result = train_entity_action_value(
        EntityQConfig(
            corpus_path=corpus,
            parent_checkpoint=parent,
            output_dir=tmp_path / "q",
            epochs=1,
            batch_size=1,
            bootstrap_samples=20,
        )
    )

    assert result["split_counts"] == {"train": 1, "validation": 1, "confirmation": 1}
    assert result["best_epoch"] == 1
    assert result["confirmation"]["states"] == 1
    assert (tmp_path / "q" / "canonical_entity_action_value_scorer.pt").exists()
