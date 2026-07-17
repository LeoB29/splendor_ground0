from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from splendor_ai.bots.search_bot import CheckpointValueEvaluator
from splendor_ai.encoding import collate_canonical_state_batch
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    CanonicalEntityDistributionalValueModel,
    CanonicalEntityValueConfig,
)
from splendor_ai.training.run_scaled_public_belief_value import (
    ExactlyBalancedPhaseDataset,
    _selected_group_ids,
    initialize_from_actor,
    offline_gates,
)


def _config() -> CanonicalEntityValueConfig:
    return CanonicalEntityValueConfig(
        hidden_size=16,
        layers=1,
        attention_heads=1,
        attention_feedforward_size=16,
        id_embedding_size=4,
    )


def _model() -> CanonicalEntityDistributionalValueModel:
    return CanonicalEntityDistributionalValueModel(_config(), margin_bins=41)


def test_distributional_public_value_ignores_hidden_deck_order() -> None:
    state_a = SplendorEnv(seed=9).initial_state()
    state_b = state_a.copy_shallow()
    state_b.hidden_tier_decks[1].reverse()
    model = _model().eval()
    with torch.no_grad():
        margin_a, wdl_a = model(**collate_canonical_state_batch([(state_a, 0)]))
        margin_b, wdl_b = model(**collate_canonical_state_batch([(state_b, 0)]))
    torch.testing.assert_close(margin_a, margin_b)
    torch.testing.assert_close(wdl_a, wdl_b)


def test_distributional_value_checkpoint_is_deployable(tmp_path) -> None:
    model = _model().eval()
    path = tmp_path / "value.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_distributional_value",
            "model_config": asdict(_config()),
            "margin_bins": 41,
            "search_wdl_blend_weight": 0.5,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    state = SplendorEnv(seed=2).initial_state()
    evaluator = CheckpointValueEvaluator(path, value_scale=1.0, prediction_mode="blend")
    value = evaluator.evaluate_state(state, 0)
    assert -1.0 <= value <= 1.0


def test_encoder_warm_start_excludes_distribution_heads() -> None:
    actor = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=16,
            action_hidden_size=16,
            scorer_hidden_size=16,
            entity_layers=1,
            action_layers=1,
            attention_heads=1,
            attention_feedforward_size=16,
            id_embedding_size=4,
        )
    )
    model = _model()
    margin_before = {name: value.clone() for name, value in model.margin_head.state_dict().items()}
    transferred = initialize_from_actor(model, actor.state_dict())
    assert transferred
    assert all("margin_head" not in name and "wdl_head" not in name for name in transferred)
    for name, value in model.margin_head.state_dict().items():
        torch.testing.assert_close(value, margin_before[name])
    torch.testing.assert_close(model.entity_type_embedding.weight, actor.entity_type_embedding.weight)


def test_frozen_group_splits_preserve_original_032_033_boundaries() -> None:
    assert list(_selected_group_ids("train")) == list(range(448))
    assert list(_selected_group_ids("validation")) == list(range(448, 512))
    assert set(_selected_group_ids("train")).isdisjoint(_selected_group_ids("validation"))


def test_phase_dataset_exactly_balances_all_64_strata() -> None:
    rows = []
    for source in range(2):
        for opponent in range(4):
            for seat in range(2):
                for phase in range(4):
                    rows.extend([(source, opponent, seat, phase)] * (1 + phase))
    count = len(rows)
    payload = {
        "entity_type_ids": torch.zeros(count, 1, dtype=torch.int16),
        "entity_card_ids": torch.zeros(count, 1, dtype=torch.int16),
        "entity_noble_ids": torch.zeros(count, 1, dtype=torch.int16),
        "entity_numeric_features": torch.zeros(count, 1, 34, dtype=torch.float16),
        "entity_mask": torch.ones(count, 1, dtype=torch.bool),
        "margin_class": torch.zeros(count, dtype=torch.int8),
        "wdl_class": torch.zeros(count, dtype=torch.int8),
        "source_id": torch.tensor([row[0] for row in rows], dtype=torch.int8),
        "opponent_id": torch.tensor([row[1] for row in rows], dtype=torch.int8),
        "actor_seat": torch.tensor([row[2] for row in rows], dtype=torch.int8),
        "phase_id": torch.tensor([row[3] for row in rows], dtype=torch.int8),
    }
    dataset = ExactlyBalancedPhaseDataset(payload)
    selected = [int(index) for index in dataset.indices]
    counts = {}
    for index in selected:
        key = rows[index]
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 64
    assert set(counts.values()) == {4}


def test_offline_gate_requires_midgame_calibration_and_successor_consistency() -> None:
    candidate_breakdowns = {}
    heuristic_breakdowns = {}
    for label in [*(f"profile:{name}" for name in ("035", "016", "shallow", "023")), *(f"phase:{name}" for name in ("0-14", "15-29", "30-44", "45+"))]:
        candidate_breakdowns[label] = {"mae": 0.20}
        heuristic_breakdowns[label] = {"mae": 0.30}
    result = {
        "candidate": {"macro_profile_seat_mae": 0.20},
        "heuristic": {"macro_profile_seat_mae": 0.30},
        "candidate_ordering": {"concordance": 0.65, "macro_stratum_correlation": 0.5},
        "heuristic_ordering": {"concordance": 0.55, "macro_stratum_correlation": 0.3},
        "candidate_ordering_by_phase": {phase: {"concordance": 0.64} for phase in ("0-14", "15-29", "30-44", "45+")},
        "heuristic_ordering_by_phase": {phase: {"concordance": 0.55} for phase in ("0-14", "15-29", "30-44", "45+")},
        "candidate_breakdowns": candidate_breakdowns,
        "heuristic_breakdowns": heuristic_breakdowns,
        "paired_game_error": {"ci95_high": -0.01},
        "calibration": {"wdl_ece10": 0.05},
    }
    consistency = {
        "mean_spearman": 0.1,
        "mean_policy_top_value_percentile": 0.6,
        "value_best_in_policy_top3_rate": 0.55,
    }
    assert all(offline_gates(result, consistency).values())
    consistency["mean_spearman"] = -0.01
    assert not offline_gates(result, consistency)["successor_policy_value_positive_rank_correlation"]
