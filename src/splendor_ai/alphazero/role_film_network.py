"""Identity-preserving state-dependent start-role FiLM model for pilot 121."""

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


ROLE_FILM_MODEL_KIND = "alphazero_frozen_parent_split_role_film_v5"


@dataclass(frozen=True, slots=True)
class RoleFiLMCheckpoint:
    path: Path
    generation: int
    parent_model_config: CanonicalEntityActionScorerConfig
    metadata: dict[str, Any]


class FrozenParentSplitRoleFiLMModel(nn.Module):
    """Frozen parent with separate signed gain/bias modulation at each head."""

    def __init__(
        self,
        parent: CanonicalEntityActionScorerModel,
        *,
        initialize_zero: bool,
    ) -> None:
        super().__init__()
        self.parent = parent
        width = parent.config.entity_hidden_size
        # Row 0 is gain; row 1 is bias. Keeping one parameter per endpoint
        # makes the policy/value separation contract explicit.
        self.policy_role = nn.Parameter(torch.empty(2, width))
        self.value_role = nn.Parameter(torch.empty(2, width))
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        if initialize_zero:
            nn.init.zeros_(self.policy_role)
            nn.init.zeros_(self.value_role)

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
        entity_input = torch.cat(
            [
                entity_numeric_features,
                self.parent.entity_type_embedding(entity_type_ids),
                self.parent.card_embedding(entity_card_ids),
                self.parent.noble_embedding(entity_noble_ids),
            ],
            dim=-1,
        )
        entity_hidden = self.parent.entity_encoder(
            self.parent.entity_input(entity_input),
            src_key_padding_mask=~entity_mask,
        )
        entity_hidden = entity_hidden.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        counts = entity_mask.sum(dim=1, keepdim=True).clamp_min(1).to(entity_hidden.dtype)
        state = entity_hidden.sum(dim=1) / counts
        signed = observer_is_start_player.to(state.dtype).mul(2.0).sub(1.0).unsqueeze(-1)
        policy_state = (
            state * (1.0 + signed * self.policy_role[0])
            + signed * self.policy_role[1]
        )
        value_state = (
            state * (1.0 + signed * self.value_role[0])
            + signed * self.value_role[1]
        )

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
            + self.parent.state_to_action_context(policy_state).unsqueeze(1),
            src_key_padding_mask=~action_mask,
        )
        action_hidden = self.parent.action_layer_norm(action_hidden)
        action_hidden = action_hidden.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        expanded = policy_state.unsqueeze(1).expand(-1, action_hidden.shape[1], -1)
        logits = self.parent.score_head(
            torch.cat([expanded, action_hidden], dim=-1)
        ).squeeze(-1)
        value = self.parent.value_head(value_state).squeeze(-1)
        return logits, value


def create_identity_role_film_model(
    parent_checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> tuple[FrozenParentSplitRoleFiLMModel, AlphaZeroCheckpoint]:
    parent, identity = load_checkpoint(parent_checkpoint, device=device)
    model = FrozenParentSplitRoleFiLMModel(parent, initialize_zero=True).to(device)
    model.eval()
    return model, identity


def save_role_film_checkpoint(
    path: str | Path,
    *,
    model: FrozenParentSplitRoleFiLMModel,
    generation: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": ROLE_FILM_MODEL_KIND,
            "format_version": 1,
            "generation": int(generation),
            "parent_model_config": asdict(model.parent.config),
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def load_role_film_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[FrozenParentSplitRoleFiLMModel, RoleFiLMCheckpoint]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("model_kind") != ROLE_FILM_MODEL_KIND:
        raise ValueError(f"Unsupported role FiLM checkpoint: {payload.get('model_kind')!r}")
    config = CanonicalEntityActionScorerConfig(**payload["parent_model_config"])
    model = FrozenParentSplitRoleFiLMModel(
        CanonicalEntityActionScorerModel(config),
        initialize_zero=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    identity = RoleFiLMCheckpoint(
        path=checkpoint_path,
        generation=int(payload.get("generation", 0)),
        parent_model_config=config,
        metadata=dict(payload.get("metadata", {})),
    )
    return model, identity


def role_film_parameter_counts(model: FrozenParentSplitRoleFiLMModel) -> dict[str, int]:
    return {
        "parent": sum(parameter.numel() for parameter in model.parent.parameters()),
        "policy_gain": model.policy_role[0].numel(),
        "policy_bias": model.policy_role[1].numel(),
        "value_gain": model.value_role[0].numel(),
        "value_bias": model.value_role[1].numel(),
        "role_total": model.policy_role.numel() + model.value_role.numel(),
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
