from pathlib import Path

import torch

from splendor_ai.training.es_adapter import RankAdapterSpec, adapter_parameter_count, normalized_direction, write_adapter_checkpoint
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel


def _checkpoint(path: Path) -> Path:
    model = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig())
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_config": {
                "observation_size": 295,
                "action_feature_size": 79,
                "state_hidden_size": 256,
                "action_hidden_size": 128,
                "scorer_hidden_size": 256,
                "state_depth": 3,
                "action_depth": 2,
                "attention_layers": 2,
                "attention_heads": 4,
                "attention_feedforward_size": 256,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_rank_adapter_materializes_deterministic_nonzero_update(tmp_path) -> None:
    base = _checkpoint(tmp_path / "base.pt")
    source = torch.load(base, map_location="cpu", weights_only=True)
    spec = RankAdapterSpec(rank=4, seed=9)
    count = adapter_parameter_count(source["model_state_dict"], spec)
    theta = normalized_direction(count, seed=10) * 0.5

    output = write_adapter_checkpoint(base, tmp_path / "adapted.pt", theta, spec)
    adapted = torch.load(output, map_location="cpu", weights_only=True)

    key = spec.target_weight_keys[0]
    assert count == 3584
    assert not torch.equal(source["model_state_dict"][key], adapted["model_state_dict"][key])
    assert adapted["adapter_provenance"]["parameter_count"] == count
