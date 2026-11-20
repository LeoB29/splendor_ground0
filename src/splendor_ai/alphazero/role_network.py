"""Identity-preserving start-role joint policy/value bridge for pilot 118."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from splendor_ai.encoding.entity_encoding import ENTITY_GLOBAL
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)

from .network import AlphaZeroCheckpoint, load_checkpoint


ROLE_MODEL_KIND = "alphazero_frozen_parent_start_role_joint_v3"


@dataclass(frozen=True, slots=True)
class RoleCheckpoint:
    path: Path
    generation: int
    parent_model_config: CanonicalEntityActionScorerConfig
    metadata: dict[str, Any]


class FrozenParentStartRoleJointModel(nn.Module):
    """Frozen v1 joint model with one trainable global start-role embedding."""

    def __init__(
        self,
        parent: CanonicalEntityActionScorerModel,
        *,
        initialize_zero: bool,
    ) -> None:
        super().__init__()
        self.parent = parent
        self.start_role_embedding = nn.Embedding(2, parent.config.entity_hidden_size)
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        if initialize_zero:
            nn.init.zeros_(self.start_role_embedding.weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(
        self,
        entity_type_ids: torch.Tensor,
        entity_card_ids: torch.Tensor,
        entity_noble_ids: torch.Tensor,
        entity_numeric_features: torch.Tensor,
        entity_mask: torch.Tensor,
        action_type_ids: torch.Tensor,
        action_card_ids: torch.Tensor,
        action_noble_ids: torch.Tensor,
        action_numeric_features: torch.Tensor,
        action_mask: torch.Tensor,
        observer_is_start_player: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.parent._validate_inputs(
            entity_type_ids,
            entity_card_ids,
            entity_noble_ids,
            entity_numeric_features,
            entity_mask,
            action_type_ids,
            action_card_ids,
            action_noble_ids,
            action_numeric_features,
            action_mask,
        )
        if observer_is_start_player.ndim != 1:
            raise ValueError("observer_is_start_player must have shape [batch].")
        if observer_is_start_player.shape[0] != entity_type_ids.shape[0]:
            raise ValueError("Start-role batch size must match entity tensors.")
        if not bool((entity_type_ids[:, 0] == ENTITY_GLOBAL).all()):
            raise ValueError("Start-role bridge requires the canonical global entity first.")

        entity_input = torch.cat(
            [
                entity_numeric_features,
                self.parent.entity_type_embedding(entity_type_ids),
                self.parent.card_embedding(entity_card_ids),
                self.parent.noble_embedding(entity_noble_ids),
            ],
            dim=-1,
        )
        transformer_input = self.parent.entity_input(entity_input)
        role_delta = self.start_role_embedding(observer_is_start_player.long())
        global_mask = torch.zeros_like(transformer_input)
        global_mask[:, 0, :] = role_delta
        entity_hidden = self.parent.entity_encoder(
            transformer_input + global_mask,
            src_key_padding_mask=~entity_mask,
        )
        entity_hidden = entity_hidden.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        entity_count = (
            entity_mask.sum(dim=1, keepdim=True).clamp_min(1).to(entity_hidden.dtype)
        )
        state_hidden = entity_hidden.sum(dim=1) / entity_count

        action_input = torch.cat(
            [
                action_numeric_features,
                self.parent.action_type_embedding(action_type_ids),
                self.parent.card_embedding(action_card_ids),
                self.parent.noble_embedding(action_noble_ids),
            ],
            dim=-1,
        )
        action_hidden = self.parent.action_input(action_input)
        action_hidden = self.parent.action_encoder(
            action_hidden
            + self.parent.state_to_action_context(state_hidden).unsqueeze(1),
            src_key_padding_mask=~action_mask,
        )
        action_hidden = self.parent.action_layer_norm(action_hidden)
        action_hidden = action_hidden.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        expanded_state = state_hidden.unsqueeze(1).expand(
            -1, action_hidden.shape[1], -1
        )
        scores = self.parent.score_head(
            torch.cat([expanded_state, action_hidden], dim=-1)
        ).squeeze(-1)
        value = self.parent.value_head(state_hidden).squeeze(-1)
        return scores, value


def create_identity_role_model(
    parent_checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> tuple[FrozenParentStartRoleJointModel, AlphaZeroCheckpoint]:
    parent_model, parent_identity = load_checkpoint(parent_checkpoint, device=device)
    model = FrozenParentStartRoleJointModel(
        parent_model,
        initialize_zero=True,
    ).to(device)
    model.eval()
    return model, parent_identity


def save_role_checkpoint(
    path: str | Path,
    *,
    model: FrozenParentStartRoleJointModel,
    generation: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": ROLE_MODEL_KIND,
            "format_version": 1,
            "generation": int(generation),
            "parent_model_config": asdict(model.parent.config),
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def load_role_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[FrozenParentStartRoleJointModel, RoleCheckpoint]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("model_kind") != ROLE_MODEL_KIND:
        raise ValueError(f"Unsupported start-role checkpoint: {payload.get('model_kind')!r}")
    parent_config = CanonicalEntityActionScorerConfig(**payload["parent_model_config"])
    model = FrozenParentStartRoleJointModel(
        CanonicalEntityActionScorerModel(parent_config),
        initialize_zero=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    identity = RoleCheckpoint(
        path=checkpoint_path,
        generation=int(payload.get("generation", 0)),
        parent_model_config=parent_config,
        metadata=dict(payload.get("metadata", {})),
    )
    return model, identity


def role_parameter_counts(model: FrozenParentStartRoleJointModel) -> dict[str, int]:
    return {
        "parent": sum(parameter.numel() for parameter in model.parent.parameters()),
        "role_embedding": model.start_role_embedding.weight.numel(),
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
