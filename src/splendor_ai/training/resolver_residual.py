"""Frozen-parent action-specific residual scorer used by experiment 053."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from splendor_ai.encoding import (
    encode_legal_action_features,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    masked_legal_action_scores,
)


@dataclass(frozen=True, slots=True)
class ResolverResidualConfig:
    hidden_size: int = 64


class FrozenParentResidualScorerModel(nn.Module):
    """Add a centered learned residual to a permanently frozen attention scorer."""

    def __init__(
        self,
        parent: LegalActionAttentionScorerModel,
        config: ResolverResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.parent = parent
        self.config = config or ResolverResidualConfig()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        input_size = (
            self.parent.config.state_hidden_size
            + self.parent.config.action_hidden_size
            + 1
        )
        self.residual_head = nn.Sequential(
            nn.Linear(input_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, 1),
        )
        final = self.residual_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def parent_config(self) -> LegalActionAttentionScorerConfig:
        return self.parent.config

    def train(self, mode: bool = True) -> FrozenParentResidualScorerModel:
        super().train(mode)
        self.parent.eval()
        return self

    def forward(
        self,
        observation: torch.Tensor,
        legal_action_features: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores, value, _parent_scores, _residual = self.forward_with_residual(
            observation,
            legal_action_features,
            legal_action_mask,
        )
        return scores, value

    def forward_with_residual(
        self,
        observation: torch.Tensor,
        legal_action_features: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, observation_size].")
        if legal_action_features.ndim != 3:
            raise ValueError(
                "legal_action_features must have shape "
                "[batch, legal_actions, action_feature_size]."
            )
        batch_size, legal_count, _ = legal_action_features.shape
        if legal_action_mask is None:
            legal_action_mask = torch.ones(
                (batch_size, legal_count),
                dtype=torch.bool,
                device=legal_action_features.device,
            )
        if legal_action_mask.shape != (batch_size, legal_count):
            raise ValueError("legal_action_mask must match the legal-action rows.")

        with torch.no_grad():
            state_hidden = self.parent.state_encoder(observation)
            flat_actions = legal_action_features.reshape(
                batch_size * legal_count,
                self.parent.config.action_feature_size,
            )
            action_hidden = self.parent.action_encoder(flat_actions).reshape(
                batch_size,
                legal_count,
                self.parent.config.action_hidden_size,
            )
            contextual = self.parent.action_set_encoder(
                action_hidden
                + self.parent.state_to_action_context(state_hidden).unsqueeze(1),
                src_key_padding_mask=~legal_action_mask,
            )
            contextual = self.parent.action_set_layer_norm(contextual)
            contextual = contextual.masked_fill(
                ~legal_action_mask.unsqueeze(-1),
                0.0,
            )
            expanded_state = state_hidden.unsqueeze(1).expand(
                batch_size,
                legal_count,
                self.parent.config.state_hidden_size,
            )
            parent_scores = self.parent.score_head(
                torch.cat([expanded_state, contextual], dim=-1)
            ).squeeze(-1)
            value = self.parent.value_head(state_hidden).squeeze(-1)

        residual_input = torch.cat(
            [expanded_state.detach(), contextual.detach(), parent_scores.detach().unsqueeze(-1)],
            dim=-1,
        )
        residual = self.residual_head(residual_input).squeeze(-1)
        mask_float = legal_action_mask.to(dtype=residual.dtype)
        residual_mean = (residual * mask_float).sum(dim=1, keepdim=True) / (
            mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        residual = (residual - residual_mean).masked_fill(~legal_action_mask, 0.0)
        return parent_scores + residual, value, parent_scores, residual


class FrozenParentResidualScorerBot:
    """Greedy information-legal bot backed by a self-contained residual checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        checkpoint = torch.load(
            self._checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if checkpoint.get("model_kind") != "frozen_parent_resolver_residual_scorer":
            raise ValueError("Unsupported residual scorer checkpoint.")
        parent = LegalActionAttentionScorerModel(
            LegalActionAttentionScorerConfig(**checkpoint["parent_model_config"])
        )
        model = FrozenParentResidualScorerModel(
            parent,
            ResolverResidualConfig(**checkpoint["residual_model_config"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self._model = model.to(device)
        self._model.eval()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        config = self._model.parent_config
        observation = encode_public_observation_tensor(state, state.current_player)
        observation_vector = fit_feature_vector(
            observation.vector,
            config.observation_size,
        )
        action_features = encode_legal_action_features(
            state=state,
            legal_actions=actions,
            player_id=state.current_player,
        )
        action_vectors = [
            fit_feature_vector(features.vector, config.action_feature_size)
            for features in action_features
        ]
        observation_tensor = torch.tensor(
            [observation_vector],
            dtype=torch.float32,
            device=self._device,
        )
        action_tensor = torch.tensor(
            [action_vectors],
            dtype=torch.float32,
            device=self._device,
        )
        mask = torch.ones(
            (1, len(actions)),
            dtype=torch.bool,
            device=self._device,
        )
        with torch.inference_mode():
            scores, _value = self._model(observation_tensor, action_tensor, mask)
            best = int(masked_legal_action_scores(scores, mask).argmax(dim=1).item())
        return actions[best]


def load_frozen_parent_residual_model(
    parent_checkpoint: Path,
    device: str,
    config: ResolverResidualConfig | None = None,
) -> FrozenParentResidualScorerModel:
    checkpoint = torch.load(parent_checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError("Residual parent must be a legal-action attention scorer.")
    parent = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(**checkpoint.get("model_config", {}))
    )
    parent.load_state_dict(checkpoint["model_state_dict"])
    model = FrozenParentResidualScorerModel(parent, config)
    return model.to(device)


def save_frozen_parent_residual_checkpoint(
    path: Path,
    model: FrozenParentResidualScorerModel,
    provenance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "frozen_parent_resolver_residual_scorer",
            "model_state_dict": model.state_dict(),
            "parent_model_config": asdict(model.parent_config),
            "residual_model_config": asdict(model.config),
            "provenance": provenance,
        },
        path,
    )


def trainable_parameter_count(model: FrozenParentResidualScorerModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
