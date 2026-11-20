"""Identity-preserving residual policy/value network for capacity pilot 116."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)

from .network import AlphaZeroCheckpoint, load_checkpoint


CAPACITY_MODEL_KIND = "alphazero_frozen_parent_residual_policy_value_v2"


@dataclass(frozen=True, slots=True)
class CapacityCheckpoint:
    path: Path
    generation: int
    parent_model_config: CanonicalEntityActionScorerConfig
    residual_model_config: CanonicalEntityActionScorerConfig
    metadata: dict[str, Any]


class FrozenParentResidualPolicyValueModel(nn.Module):
    """Exact frozen parent plus a trainable zero-initialized correction tower."""

    def __init__(
        self,
        parent: CanonicalEntityActionScorerModel,
        residual_config: CanonicalEntityActionScorerConfig,
        *,
        initialize_zero: bool,
    ) -> None:
        super().__init__()
        self.parent = parent
        self.residual_config = residual_config
        self.residual = CanonicalEntityActionScorerModel(residual_config)
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        if initialize_zero:
            self.zero_correction_heads()

    def zero_correction_heads(self) -> None:
        """Make the composite function exactly equal to the frozen parent."""

        for head in (self.residual.score_head, self.residual.value_head):
            final = head[-1] if isinstance(head[-1], nn.Linear) else head[-2]
            if not isinstance(final, nn.Linear):
                raise TypeError("Residual correction head has no final linear layer.")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, **inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            parent_logits, parent_value = self.parent(**inputs)
        residual_logits, residual_value = self.residual(**inputs)
        return parent_logits + residual_logits, torch.clamp(
            parent_value + residual_value, min=-1.0, max=1.0
        )


def create_identity_capacity_model(
    parent_checkpoint: str | Path,
    *,
    residual_config: CanonicalEntityActionScorerConfig,
    device: str | torch.device,
) -> tuple[FrozenParentResidualPolicyValueModel, AlphaZeroCheckpoint]:
    parent_model, parent_identity = load_checkpoint(parent_checkpoint, device=device)
    model = FrozenParentResidualPolicyValueModel(
        parent_model,
        residual_config,
        initialize_zero=True,
    ).to(device)
    model.eval()
    return model, parent_identity


def save_capacity_checkpoint(
    path: str | Path,
    *,
    model: FrozenParentResidualPolicyValueModel,
    generation: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": CAPACITY_MODEL_KIND,
            "format_version": 1,
            "generation": int(generation),
            "parent_model_config": asdict(model.parent.config),
            "residual_model_config": asdict(model.residual_config),
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def load_capacity_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[FrozenParentResidualPolicyValueModel, CapacityCheckpoint]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("model_kind") != CAPACITY_MODEL_KIND:
        raise ValueError(f"Unsupported capacity checkpoint: {payload.get('model_kind')!r}")
    parent_config = CanonicalEntityActionScorerConfig(**payload["parent_model_config"])
    residual_config = CanonicalEntityActionScorerConfig(**payload["residual_model_config"])
    model = FrozenParentResidualPolicyValueModel(
        CanonicalEntityActionScorerModel(parent_config),
        residual_config,
        initialize_zero=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    identity = CapacityCheckpoint(
        path=checkpoint_path,
        generation=int(payload.get("generation", 0)),
        parent_model_config=parent_config,
        residual_model_config=residual_config,
        metadata=dict(payload.get("metadata", {})),
    )
    return model, identity


def capacity_parameter_counts(
    model: FrozenParentResidualPolicyValueModel,
) -> dict[str, int]:
    return {
        "parent": sum(parameter.numel() for parameter in model.parent.parameters()),
        "residual": sum(parameter.numel() for parameter in model.residual.parameters()),
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
