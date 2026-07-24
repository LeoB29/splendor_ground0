"""Frozen-parent action-value model with an optional public-history encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from splendor_ai.encoding.entity_encoding import (
    ACTION_NUMERIC_SIZE,
    ACTION_TYPE_COUNT,
    card_vocabulary_size,
    noble_vocabulary_size,
)
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)


@dataclass(frozen=True, slots=True)
class ResolverHistoryValueConfig:
    action_type_embedding_size: int = 8
    card_embedding_size: int = 16
    noble_embedding_size: int = 8
    history_numeric_hidden_size: int = 32
    history_hidden_size: int = 64
    value_hidden_size: int = 128


class FrozenParentResolverValueModel(nn.Module):
    """Predict action values from frozen 035 features and optional public history."""

    def __init__(
        self,
        parent: LegalActionAttentionScorerModel,
        *,
        use_history: bool,
        config: ResolverHistoryValueConfig | None = None,
    ) -> None:
        super().__init__()
        self.parent = parent
        self.use_history = bool(use_history)
        self.config = config or ResolverHistoryValueConfig()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()

        self.history_action_type_embedding = nn.Embedding(
            ACTION_TYPE_COUNT,
            self.config.action_type_embedding_size,
            padding_idx=0,
        )
        self.history_card_embedding = nn.Embedding(
            card_vocabulary_size(),
            self.config.card_embedding_size,
            padding_idx=0,
        )
        self.history_noble_embedding = nn.Embedding(
            noble_vocabulary_size(),
            self.config.noble_embedding_size,
            padding_idx=0,
        )
        self.history_numeric_projection = nn.Linear(
            ACTION_NUMERIC_SIZE,
            self.config.history_numeric_hidden_size,
        )
        event_width = (
            self.config.action_type_embedding_size
            + self.config.card_embedding_size
            + self.config.noble_embedding_size
            + self.config.history_numeric_hidden_size
        )
        self.history_event_projection = nn.Sequential(
            nn.Linear(event_width, self.config.history_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.history_hidden_size),
        )
        self.history_gru = nn.GRU(
            input_size=self.config.history_hidden_size,
            hidden_size=self.config.history_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        value_input_size = (
            self.parent.config.state_hidden_size
            + self.parent.config.action_hidden_size
            + 1
            + self.config.history_hidden_size
        )
        self.action_value_head = nn.Sequential(
            nn.Linear(value_input_size, self.config.value_hidden_size),
            nn.GELU(),
            nn.Linear(self.config.value_hidden_size, 1),
        )

    @property
    def parent_config(self) -> LegalActionAttentionScorerConfig:
        return self.parent.config

    def train(self, mode: bool = True) -> FrozenParentResolverValueModel:
        super().train(mode)
        self.parent.eval()
        return self

    def encode_history(
        self,
        action_type_ids: torch.Tensor,
        card_ids: torch.Tensor,
        noble_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final chronological GRU state, with exact zero for empties."""

        if action_type_ids.ndim != 2:
            raise ValueError("history action ids must have shape [batch, events].")
        if numeric_features.ndim != 3:
            raise ValueError("history numeric features must have shape [batch, events, 34].")
        if numeric_features.shape[:2] != action_type_ids.shape:
            raise ValueError("history numeric rows must align with history ids.")
        if numeric_features.shape[2] != ACTION_NUMERIC_SIZE:
            raise ValueError("Unexpected public-history numeric feature size.")
        if any(
            tensor.shape != action_type_ids.shape
            for tensor in (card_ids, noble_ids, history_mask)
        ):
            raise ValueError("All public-history id tensors and mask must align.")

        lengths = history_mask.to(dtype=torch.long).sum(dim=1)
        context = numeric_features.new_zeros(
            (numeric_features.shape[0], self.config.history_hidden_size)
        )
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
        if nonempty.numel() == 0:
            return context

        event_input = torch.cat(
            [
                self.history_action_type_embedding(action_type_ids),
                self.history_card_embedding(card_ids),
                self.history_noble_embedding(noble_ids),
                self.history_numeric_projection(numeric_features),
            ],
            dim=-1,
        )
        events = self.history_event_projection(event_input)
        chosen_events = events.index_select(0, nonempty)
        chosen_lengths = lengths.index_select(0, nonempty)
        packed = pack_padded_sequence(
            chosen_events,
            chosen_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _packed_output, hidden = self.history_gru(packed)
        context.index_copy_(0, nonempty, hidden[-1])
        return context

    def forward(
        self,
        observation: torch.Tensor,
        legal_action_features: torch.Tensor,
        legal_action_mask: torch.Tensor,
        history_action_type_ids: torch.Tensor,
        history_card_ids: torch.Tensor,
        history_noble_ids: torch.Tensor,
        history_numeric_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-legal-action value predictions and frozen parent scores."""

        if observation.ndim != 2 or legal_action_features.ndim != 3:
            raise ValueError("Observation and legal-action tensors have invalid rank.")
        batch_size, legal_count, _ = legal_action_features.shape
        if legal_action_mask.shape != (batch_size, legal_count):
            raise ValueError("legal_action_mask must align with legal actions.")

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

        if self.use_history:
            history_context = self.encode_history(
                history_action_type_ids,
                history_card_ids,
                history_noble_ids,
                history_numeric_features,
                history_mask,
            )
        else:
            history_context = observation.new_zeros(
                (batch_size, self.config.history_hidden_size)
            )
        expanded_history = history_context.unsqueeze(1).expand(
            batch_size,
            legal_count,
            self.config.history_hidden_size,
        )
        value_input = torch.cat(
            [
                expanded_state.detach(),
                contextual.detach(),
                parent_scores.detach().unsqueeze(-1),
                expanded_history,
            ],
            dim=-1,
        )
        values = self.action_value_head(value_input).squeeze(-1)
        values = values.masked_fill(~legal_action_mask, 0.0)
        return values, parent_scores


def load_frozen_parent_resolver_value_model(
    parent_checkpoint: Path,
    *,
    use_history: bool,
    device: str = "cpu",
    config: ResolverHistoryValueConfig | None = None,
) -> FrozenParentResolverValueModel:
    checkpoint = torch.load(parent_checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError("Resolver-value parent must be a legal-action attention scorer.")
    parent = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(**checkpoint.get("model_config", {}))
    )
    parent.load_state_dict(checkpoint["model_state_dict"])
    return FrozenParentResolverValueModel(
        parent,
        use_history=use_history,
        config=config,
    ).to(device)


def save_frozen_parent_resolver_value_checkpoint(
    path: Path,
    model: FrozenParentResolverValueModel,
    provenance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "frozen_parent_resolver_public_history_value",
            "model_state_dict": model.state_dict(),
            "parent_model_config": asdict(model.parent_config),
            "resolver_history_value_config": asdict(model.config),
            "use_history": model.use_history,
            "provenance": provenance,
        },
        path,
    )


def trainable_parameter_count(model: FrozenParentResolverValueModel) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
