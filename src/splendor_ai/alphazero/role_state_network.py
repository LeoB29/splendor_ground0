"""Partially trainable start-role-aware state representation for pilot 122."""

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


ROLE_STATE_MODEL_KIND = "alphazero_partial_state_role_representation_v6"


@dataclass(frozen=True, slots=True)
class RoleStateCheckpoint:
    path: Path
    generation: int
    parent_model_config: CanonicalEntityActionScorerConfig
    metadata: dict[str, Any]


class PartialStateRoleRepresentationModel(nn.Module):
    """Exact parent whose signed role input and final entity block can learn."""

    def __init__(
        self,
        parent: CanonicalEntityActionScorerModel,
        *,
        initialize_zero: bool,
    ) -> None:
        super().__init__()
        if len(parent.entity_encoder.layers) != 2:
            raise ValueError("Pilot 122 requires the exact two-block entity encoder.")
        self.parent = parent
        self.start_role_direction = nn.Parameter(
            torch.empty(parent.config.entity_hidden_size)
        )
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        for parameter in self.parent.entity_encoder.layers[-1].parameters():
            parameter.requires_grad_(True)
        self.parent.eval()
        if initialize_zero:
            nn.init.zeros_(self.start_role_direction)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen modules stay in inference mode. The selected block has zero
        # dropout in the frozen architecture, but its mode remains explicit.
        self.parent.eval()
        self.parent.entity_encoder.layers[-1].train(mode)
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
        if not bool(((observer_is_start_player == 0) | (observer_is_start_player == 1)).all()):
            raise ValueError("observer_is_start_player must contain only zero or one.")
        if not bool((entity_type_ids[:, 0] == ENTITY_GLOBAL).all()):
            raise ValueError("Pilot 122 requires the canonical global entity first.")

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
        signed_role = (
            observer_is_start_player.to(transformer_input.dtype).mul(2.0).sub(1.0)
        )
        role_delta = signed_role.unsqueeze(-1) * self.start_role_direction.unsqueeze(0)
        global_delta = torch.zeros_like(transformer_input)
        global_delta[:, 0, :] = role_delta
        entity_hidden = self.parent.entity_encoder(
            transformer_input + global_delta,
            src_key_padding_mask=~entity_mask,
        )
        entity_hidden = entity_hidden.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        entity_count = entity_mask.sum(dim=1, keepdim=True).clamp_min(1).to(entity_hidden.dtype)
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
            action_hidden + self.parent.state_to_action_context(state_hidden).unsqueeze(1),
            src_key_padding_mask=~action_mask,
        )
        action_hidden = self.parent.action_layer_norm(action_hidden)
        action_hidden = action_hidden.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        expanded_state = state_hidden.unsqueeze(1).expand(-1, action_hidden.shape[1], -1)
        logits = self.parent.score_head(
            torch.cat([expanded_state, action_hidden], dim=-1)
        ).squeeze(-1)
        value = self.parent.value_head(state_hidden).squeeze(-1)
        return logits, value


def create_identity_role_state_model(
    parent_checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> tuple[PartialStateRoleRepresentationModel, AlphaZeroCheckpoint]:
    parent, identity = load_checkpoint(parent_checkpoint, device=device)
    model = PartialStateRoleRepresentationModel(parent, initialize_zero=True).to(device)
    model.eval()
    return model, identity


def save_role_state_checkpoint(
    path: str | Path,
    *,
    model: PartialStateRoleRepresentationModel,
    generation: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": ROLE_STATE_MODEL_KIND,
            "format_version": 1,
            "generation": int(generation),
            "parent_model_config": asdict(model.parent.config),
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def load_role_state_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[PartialStateRoleRepresentationModel, RoleStateCheckpoint]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("model_kind") != ROLE_STATE_MODEL_KIND:
        raise ValueError(
            f"Unsupported partial role-state checkpoint: {payload.get('model_kind')!r}"
        )
    config = CanonicalEntityActionScorerConfig(**payload["parent_model_config"])
    model = PartialStateRoleRepresentationModel(
        CanonicalEntityActionScorerModel(config), initialize_zero=False
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    identity = RoleStateCheckpoint(
        path=checkpoint_path,
        generation=int(payload.get("generation", 0)),
        parent_model_config=config,
        metadata=dict(payload.get("metadata", {})),
    )
    return model, identity


def role_state_parameter_counts(
    model: PartialStateRoleRepresentationModel,
) -> dict[str, int]:
    parent = sum(parameter.numel() for parameter in model.parent.parameters())
    final_block = sum(
        parameter.numel()
        for parameter in model.parent.entity_encoder.layers[-1].parameters()
    )
    role = model.start_role_direction.numel()
    return {
        "parent": parent,
        "frozen_parent": parent - final_block,
        "final_entity_block": final_block,
        "role_input": role,
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def role_state_trainable_names(
    model: PartialStateRoleRepresentationModel,
) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
