from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from splendor_ai.bots.puct_bot import legal_heuristic_value
from splendor_ai.bots.search_bot import CheckpointValueEvaluator
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    CanonicalEntityDistributionalValueModel,
    CanonicalEntityValueConfig,
)
from splendor_ai.training.run_reliability_blend_value import (
    CANDIDATE_GRID,
    BlendSpec,
    _blend_values,
)


def _model_config():
    return CanonicalEntityValueConfig(
        hidden_size=16,
        layers=1,
        attention_heads=1,
        attention_feedforward_size=16,
        id_embedding_size=4,
    )


def test_candidate_grid_is_frozen_small_and_unique() -> None:
    assert len(CANDIDATE_GRID) == 12
    assert len({spec.name for spec in CANDIDATE_GRID}) == 12
    assert {spec.wdl_confidence_floor for spec in CANDIDATE_GRID} == {0.0, 0.45, 0.55}
    assert all(spec.phase_learned_weights[0] <= 0.25 for spec in CANDIDATE_GRID)


def test_blend_uses_phase_weights_and_confidence_fallback() -> None:
    payload = {
        "phase_id": torch.tensor([0, 1, 2, 3], dtype=torch.int8),
        "heuristic": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }
    learned = torch.tensor([0.9, 0.8, 0.7, 0.6])
    confidence = torch.tensor([0.9, 0.4, 0.9, 0.9])
    spec = BlendSpec("probe", (0.0, 0.5, 0.75, 1.0), 0.45)

    values = _blend_values(payload, learned, confidence, spec)

    assert values.tolist() == pytest.approx([0.1, 0.2, 0.6, 0.6])


def test_deployable_zero_weight_blend_matches_handcrafted_heuristic(tmp_path) -> None:
    config = _model_config()
    model = CanonicalEntityDistributionalValueModel(config, margin_bins=41)
    path = tmp_path / "blend.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_reliability_blend",
            "model_config": asdict(config),
            "margin_bins": 41,
            "search_wdl_blend_weight": 0.5,
            "phase_learned_weights": [0.0, 0.0, 0.0, 0.0],
            "wdl_confidence_floor": 0.0,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    state = SplendorEnv(seed=11).initial_state()
    evaluator = CheckpointValueEvaluator(
        path, value_scale=1.0, prediction_mode="blend"
    )

    assert evaluator.evaluate_state(state, 0) == pytest.approx(
        legal_heuristic_value(state, 0, scale=4_000.0)
    )


def test_reliability_checkpoint_rejects_invalid_phase_weights(tmp_path) -> None:
    config = _model_config()
    model = CanonicalEntityDistributionalValueModel(config, margin_bins=41)
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_reliability_blend",
            "model_config": asdict(config),
            "margin_bins": 41,
            "phase_learned_weights": [0.0, 1.0],
            "wdl_confidence_floor": 0.0,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="four phase weights"):
        CheckpointValueEvaluator(path)
