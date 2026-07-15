"""Policy/value models for Splendor training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from splendor_ai.encoding import ACTION_FEATURE_SIZE, OBSERVATION_SIZE, ActionCodec
from splendor_ai.encoding.entity_encoding import (
    ACTION_NUMERIC_SIZE,
    ACTION_TYPE_COUNT,
    ENTITY_NUMERIC_SIZE,
    ENTITY_TYPE_COUNT,
    PRIVILEGED_ENTITY_TYPE_COUNT,
    card_vocabulary_size,
    noble_vocabulary_size,
)


@dataclass(frozen=True, slots=True)
class PolicyValueModelConfig:
    observation_size: int = OBSERVATION_SIZE
    action_space_size: int = ActionCodec().action_space_size
    trunk_hidden_size: int = 256
    trunk_depth: int = 3
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class LegalActionScorerConfig:
    observation_size: int = OBSERVATION_SIZE
    action_feature_size: int = ACTION_FEATURE_SIZE
    state_hidden_size: int = 256
    action_hidden_size: int = 128
    scorer_hidden_size: int = 256
    state_depth: int = 3
    action_depth: int = 2
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class LegalActionSetScorerConfig:
    observation_size: int = OBSERVATION_SIZE
    action_feature_size: int = ACTION_FEATURE_SIZE
    state_hidden_size: int = 256
    action_hidden_size: int = 128
    set_context_hidden_size: int = 128
    scorer_hidden_size: int = 256
    state_depth: int = 3
    action_depth: int = 2
    set_context_depth: int = 2
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class LegalActionAttentionScorerConfig:
    observation_size: int = OBSERVATION_SIZE
    action_feature_size: int = ACTION_FEATURE_SIZE
    state_hidden_size: int = 256
    action_hidden_size: int = 128
    scorer_hidden_size: int = 256
    state_depth: int = 3
    action_depth: int = 2
    attention_layers: int = 2
    attention_heads: int = 4
    attention_feedforward_size: int = 256
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class CanonicalEntityActionScorerConfig:
    """Configuration for the clean variable-length entity/action scorer."""

    entity_hidden_size: int = 192
    action_hidden_size: int = 192
    scorer_hidden_size: int = 256
    entity_layers: int = 3
    action_layers: int = 2
    attention_heads: int = 4
    attention_feedforward_size: int = 384
    id_embedding_size: int = 32
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class PrivilegedEntityCriticConfig:
    hidden_size: int = 128
    layers: int = 2
    attention_heads: int = 4
    attention_feedforward_size: int = 256
    id_embedding_size: int = 16
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class ValueModelConfig:
    observation_size: int = OBSERVATION_SIZE
    hidden_size: int = 256
    depth: int = 3
    dropout: float = 0.0


class PolicyValueMLP(nn.Module):
    """First supervised baseline network for policy/value prediction."""

    def __init__(self, config: PolicyValueModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyValueModelConfig()

        layers: list[nn.Module] = []
        input_size = self.config.observation_size
        for _ in range(self.config.trunk_depth):
            layers.append(nn.Linear(input_size, self.config.trunk_hidden_size))
            layers.append(nn.LayerNorm(self.config.trunk_hidden_size))
            layers.append(nn.GELU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            input_size = self.config.trunk_hidden_size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(self.config.trunk_hidden_size, self.config.action_space_size)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.trunk_hidden_size, self.config.trunk_hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.config.trunk_hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observation)
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return policy_logits, value


class LegalActionScorerModel(nn.Module):
    """Policy/value network that scores the legal actions supplied per state."""

    def __init__(self, config: LegalActionScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LegalActionScorerConfig()

        self.state_encoder = _make_mlp(
            input_size=self.config.observation_size,
            hidden_size=self.config.state_hidden_size,
            output_size=self.config.state_hidden_size,
            depth=self.config.state_depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.action_encoder = _make_mlp(
            input_size=self.config.action_feature_size,
            hidden_size=self.config.action_hidden_size,
            output_size=self.config.action_hidden_size,
            depth=self.config.action_depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.score_head = nn.Sequential(
            nn.Linear(
                self.config.state_hidden_size + self.config.action_hidden_size,
                self.config.scorer_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(self.config.scorer_hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.state_hidden_size, self.config.state_hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.config.state_hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        observation: torch.Tensor,
        legal_action_features: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, observation_size].")
        if legal_action_features.ndim != 3:
            raise ValueError(
                "legal_action_features must have shape [batch, legal_actions, action_feature_size]."
            )
        if observation.shape[0] != legal_action_features.shape[0]:
            raise ValueError("observation and legal_action_features batch sizes must match.")
        if observation.shape[1] != self.config.observation_size:
            raise ValueError("Unexpected observation feature size.")
        if legal_action_features.shape[2] != self.config.action_feature_size:
            raise ValueError("Unexpected action feature size.")

        batch_size, legal_action_count, _ = legal_action_features.shape
        state_hidden = self.state_encoder(observation)
        flat_action_features = legal_action_features.reshape(
            batch_size * legal_action_count,
            self.config.action_feature_size,
        )
        action_hidden = self.action_encoder(flat_action_features).reshape(
            batch_size,
            legal_action_count,
            self.config.action_hidden_size,
        )
        expanded_state = state_hidden.unsqueeze(1).expand(
            batch_size,
            legal_action_count,
            self.config.state_hidden_size,
        )
        score_input = torch.cat([expanded_state, action_hidden], dim=-1)
        action_scores = self.score_head(score_input).squeeze(-1)
        value = self.value_head(state_hidden).squeeze(-1)
        return action_scores, value


class LegalActionSetScorerModel(nn.Module):
    """Legal-action scorer with pooled context from the full legal action set."""

    def __init__(self, config: LegalActionSetScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LegalActionSetScorerConfig()

        self.state_encoder = _make_mlp(
            input_size=self.config.observation_size,
            hidden_size=self.config.state_hidden_size,
            output_size=self.config.state_hidden_size,
            depth=self.config.state_depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.action_encoder = _make_mlp(
            input_size=self.config.action_feature_size,
            hidden_size=self.config.action_hidden_size,
            output_size=self.config.action_hidden_size,
            depth=self.config.action_depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.set_context_encoder = _make_mlp(
            input_size=(2 * self.config.action_hidden_size) + 1,
            hidden_size=self.config.set_context_hidden_size,
            output_size=self.config.set_context_hidden_size,
            depth=self.config.set_context_depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.score_head = nn.Sequential(
            nn.Linear(
                self.config.state_hidden_size
                + self.config.action_hidden_size
                + self.config.set_context_hidden_size,
                self.config.scorer_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(self.config.scorer_hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.state_hidden_size, self.config.state_hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.config.state_hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        observation: torch.Tensor,
        legal_action_features: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, observation_size].")
        if legal_action_features.ndim != 3:
            raise ValueError(
                "legal_action_features must have shape [batch, legal_actions, action_feature_size]."
            )
        if observation.shape[0] != legal_action_features.shape[0]:
            raise ValueError("observation and legal_action_features batch sizes must match.")
        if observation.shape[1] != self.config.observation_size:
            raise ValueError("Unexpected observation feature size.")
        if legal_action_features.shape[2] != self.config.action_feature_size:
            raise ValueError("Unexpected action feature size.")

        batch_size, legal_action_count, _ = legal_action_features.shape
        if legal_action_mask is None:
            legal_action_mask = torch.ones(
                (batch_size, legal_action_count),
                dtype=torch.bool,
                device=legal_action_features.device,
            )
        if legal_action_mask.shape != (batch_size, legal_action_count):
            raise ValueError("legal_action_mask must have shape [batch, legal_actions].")

        state_hidden = self.state_encoder(observation)
        flat_action_features = legal_action_features.reshape(
            batch_size * legal_action_count,
            self.config.action_feature_size,
        )
        action_hidden = self.action_encoder(flat_action_features).reshape(
            batch_size,
            legal_action_count,
            self.config.action_hidden_size,
        )
        set_context = self._encode_set_context(action_hidden, legal_action_mask)

        expanded_state = state_hidden.unsqueeze(1).expand(
            batch_size,
            legal_action_count,
            self.config.state_hidden_size,
        )
        expanded_set_context = set_context.unsqueeze(1).expand(
            batch_size,
            legal_action_count,
            self.config.set_context_hidden_size,
        )
        score_input = torch.cat([expanded_state, action_hidden, expanded_set_context], dim=-1)
        action_scores = self.score_head(score_input).squeeze(-1)
        value = self.value_head(state_hidden).squeeze(-1)
        return action_scores, value

    def _encode_set_context(
        self,
        action_hidden: torch.Tensor,
        legal_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = legal_action_mask.unsqueeze(-1)
        counts = mask.sum(dim=1).clamp_min(1).to(dtype=action_hidden.dtype)
        mean_hidden = (action_hidden * mask.to(dtype=action_hidden.dtype)).sum(dim=1) / counts
        masked_for_max = action_hidden.masked_fill(~mask, torch.finfo(action_hidden.dtype).min)
        max_hidden = masked_for_max.max(dim=1).values
        max_hidden = torch.where(torch.isfinite(max_hidden), max_hidden, torch.zeros_like(max_hidden))
        count_feature = torch.log1p(counts) / math.log(65.0)
        context_input = torch.cat([mean_hidden, max_hidden, count_feature], dim=-1)
        return self.set_context_encoder(context_input)


class LegalActionAttentionScorerModel(nn.Module):
    """Legal-action scorer that contextualizes actions with masked self-attention."""

    def __init__(self, config: LegalActionAttentionScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LegalActionAttentionScorerConfig()
        if self.config.action_hidden_size % self.config.attention_heads != 0:
            raise ValueError("action_hidden_size must be divisible by attention_heads.")

        self.state_encoder = _make_mlp(
            input_size=self.config.observation_size,
            hidden_size=self.config.state_hidden_size,
            output_size=self.config.state_hidden_size,
            depth=self.config.state_depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.action_encoder = _make_mlp(
            input_size=self.config.action_feature_size,
            hidden_size=self.config.action_hidden_size,
            output_size=self.config.action_hidden_size,
            depth=self.config.action_depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.state_to_action_context = nn.Linear(
            self.config.state_hidden_size,
            self.config.action_hidden_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.action_hidden_size,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.attention_feedforward_size,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.action_set_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.attention_layers,
            enable_nested_tensor=False,
        )
        self.action_set_layer_norm = nn.LayerNorm(self.config.action_hidden_size)
        self.score_head = nn.Sequential(
            nn.Linear(
                self.config.state_hidden_size + self.config.action_hidden_size,
                self.config.scorer_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(self.config.scorer_hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.state_hidden_size, self.config.state_hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.config.state_hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        observation: torch.Tensor,
        legal_action_features: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, observation_size].")
        if legal_action_features.ndim != 3:
            raise ValueError(
                "legal_action_features must have shape [batch, legal_actions, action_feature_size]."
            )
        if observation.shape[0] != legal_action_features.shape[0]:
            raise ValueError("observation and legal_action_features batch sizes must match.")
        if observation.shape[1] != self.config.observation_size:
            raise ValueError("Unexpected observation feature size.")
        if legal_action_features.shape[2] != self.config.action_feature_size:
            raise ValueError("Unexpected action feature size.")

        batch_size, legal_action_count, _ = legal_action_features.shape
        if legal_action_mask is None:
            legal_action_mask = torch.ones(
                (batch_size, legal_action_count),
                dtype=torch.bool,
                device=legal_action_features.device,
            )
        if legal_action_mask.shape != (batch_size, legal_action_count):
            raise ValueError("legal_action_mask must have shape [batch, legal_actions].")

        state_hidden = self.state_encoder(observation)
        flat_action_features = legal_action_features.reshape(
            batch_size * legal_action_count,
            self.config.action_feature_size,
        )
        action_hidden = self.action_encoder(flat_action_features).reshape(
            batch_size,
            legal_action_count,
            self.config.action_hidden_size,
        )
        state_context = self.state_to_action_context(state_hidden).unsqueeze(1)
        contextual_input = action_hidden + state_context
        attention_mask = ~legal_action_mask
        contextual_actions = self.action_set_encoder(
            contextual_input,
            src_key_padding_mask=attention_mask,
        )
        contextual_actions = self.action_set_layer_norm(contextual_actions)
        contextual_actions = contextual_actions.masked_fill(~legal_action_mask.unsqueeze(-1), 0.0)

        expanded_state = state_hidden.unsqueeze(1).expand(
            batch_size,
            legal_action_count,
            self.config.state_hidden_size,
        )
        score_input = torch.cat([expanded_state, contextual_actions], dim=-1)
        action_scores = self.score_head(score_input).squeeze(-1)
        value = self.value_head(state_hidden).squeeze(-1)
        return action_scores, value


class CanonicalEntityActionScorerModel(nn.Module):
    """Permutation-equivariant scorer over legal state and action entities.

    Unlike historical scorers this model receives no slot-indexed observation
    vector.  Entity collections have no positional encoding, so reordering
    visible cards, reserves, nobles, or legal actions cannot change the score
    associated with a semantic action.
    """

    def __init__(self, config: CanonicalEntityActionScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or CanonicalEntityActionScorerConfig()
        if self.config.entity_hidden_size % self.config.attention_heads != 0:
            raise ValueError("entity_hidden_size must be divisible by attention_heads.")
        if self.config.action_hidden_size % self.config.attention_heads != 0:
            raise ValueError("action_hidden_size must be divisible by attention_heads.")
        id_size = self.config.id_embedding_size
        self.entity_type_embedding = nn.Embedding(ENTITY_TYPE_COUNT, id_size, padding_idx=0)
        self.card_embedding = nn.Embedding(card_vocabulary_size(), id_size, padding_idx=0)
        self.noble_embedding = nn.Embedding(noble_vocabulary_size(), id_size, padding_idx=0)
        self.entity_input = nn.Sequential(
            nn.Linear(ENTITY_NUMERIC_SIZE + (3 * id_size), self.config.entity_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.entity_hidden_size),
        )
        self.entity_encoder = nn.TransformerEncoder(
            _transformer_encoder_layer(
                self.config.entity_hidden_size,
                self.config.attention_heads,
                self.config.attention_feedforward_size,
                self.config.dropout,
            ),
            num_layers=self.config.entity_layers,
            enable_nested_tensor=False,
        )
        self.action_type_embedding = nn.Embedding(ACTION_TYPE_COUNT, id_size, padding_idx=0)
        self.action_input = nn.Sequential(
            nn.Linear(ACTION_NUMERIC_SIZE + (3 * id_size), self.config.action_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.action_hidden_size),
        )
        self.state_to_action_context = nn.Linear(
            self.config.entity_hidden_size,
            self.config.action_hidden_size,
        )
        self.action_encoder = nn.TransformerEncoder(
            _transformer_encoder_layer(
                self.config.action_hidden_size,
                self.config.attention_heads,
                self.config.attention_feedforward_size,
                self.config.dropout,
            ),
            num_layers=self.config.action_layers,
            enable_nested_tensor=False,
        )
        self.action_layer_norm = nn.LayerNorm(self.config.action_hidden_size)
        self.score_head = nn.Sequential(
            nn.Linear(
                self.config.entity_hidden_size + self.config.action_hidden_size,
                self.config.scorer_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(self.config.scorer_hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.entity_hidden_size, self.config.entity_hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.config.entity_hidden_size // 2, 1),
            nn.Tanh(),
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
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
        entity_input = torch.cat(
            [
                entity_numeric_features,
                self.entity_type_embedding(entity_type_ids),
                self.card_embedding(entity_card_ids),
                self.noble_embedding(entity_noble_ids),
            ],
            dim=-1,
        )
        entity_hidden = self.entity_encoder(
            self.entity_input(entity_input),
            src_key_padding_mask=~entity_mask,
        )
        entity_hidden = entity_hidden.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        entity_count = entity_mask.sum(dim=1, keepdim=True).clamp_min(1).to(entity_hidden.dtype)
        state_hidden = entity_hidden.sum(dim=1) / entity_count

        action_input = torch.cat(
            [
                action_numeric_features,
                self.action_type_embedding(action_type_ids),
                self.card_embedding(action_card_ids),
                self.noble_embedding(action_noble_ids),
            ],
            dim=-1,
        )
        action_hidden = self.action_input(action_input)
        action_hidden = self.action_encoder(
            action_hidden + self.state_to_action_context(state_hidden).unsqueeze(1),
            src_key_padding_mask=~action_mask,
        )
        action_hidden = self.action_layer_norm(action_hidden)
        action_hidden = action_hidden.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        expanded_state = state_hidden.unsqueeze(1).expand(-1, action_hidden.shape[1], -1)
        scores = self.score_head(torch.cat([expanded_state, action_hidden], dim=-1)).squeeze(-1)
        return scores, self.value_head(state_hidden).squeeze(-1)

    @staticmethod
    def _validate_inputs(
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
    ) -> None:
        if entity_type_ids.ndim != 2 or action_type_ids.ndim != 2:
            raise ValueError("Entity and action ids must have shape [batch, items].")
        if entity_numeric_features.shape != (*entity_type_ids.shape, ENTITY_NUMERIC_SIZE):
            raise ValueError("Unexpected entity numeric feature shape.")
        if action_numeric_features.shape != (*action_type_ids.shape, ACTION_NUMERIC_SIZE):
            raise ValueError("Unexpected action numeric feature shape.")
        if entity_mask.shape != entity_type_ids.shape or action_mask.shape != action_type_ids.shape:
            raise ValueError("Entity/action masks must match their id tensors.")
        if entity_card_ids.shape != entity_type_ids.shape or entity_noble_ids.shape != entity_type_ids.shape:
            raise ValueError("Entity id tensors must share a shape.")
        if action_card_ids.shape != action_type_ids.shape or action_noble_ids.shape != action_type_ids.shape:
            raise ValueError("Action id tensors must share a shape.")


class PrivilegedEntityCritic(nn.Module):
    """Centralized full-state value critic used only while training an actor."""

    def __init__(self, config: PrivilegedEntityCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrivilegedEntityCriticConfig()
        if self.config.hidden_size % self.config.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads.")
        id_size = self.config.id_embedding_size
        self.entity_type_embedding = nn.Embedding(PRIVILEGED_ENTITY_TYPE_COUNT, id_size, padding_idx=0)
        self.card_embedding = nn.Embedding(card_vocabulary_size(), id_size, padding_idx=0)
        self.noble_embedding = nn.Embedding(noble_vocabulary_size(), id_size, padding_idx=0)
        self.input = nn.Sequential(
            nn.Linear(ENTITY_NUMERIC_SIZE + 3 * id_size, self.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_size),
        )
        self.encoder = nn.TransformerEncoder(
            _transformer_encoder_layer(
                self.config.hidden_size,
                self.config.attention_heads,
                self.config.attention_feedforward_size,
                self.config.dropout,
            ),
            num_layers=self.config.layers,
            enable_nested_tensor=False,
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        entity_type_ids: torch.Tensor,
        entity_card_ids: torch.Tensor,
        entity_noble_ids: torch.Tensor,
        entity_numeric_features: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = torch.cat(
            [
                entity_numeric_features,
                self.entity_type_embedding(entity_type_ids),
                self.card_embedding(entity_card_ids),
                self.noble_embedding(entity_noble_ids),
            ],
            dim=-1,
        )
        hidden = self.encoder(self.input(embedded), src_key_padding_mask=~entity_mask)
        hidden = hidden.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        counts = entity_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        pooled = hidden.sum(dim=1) / counts
        return self.value_head(pooled).squeeze(-1)


def _transformer_encoder_layer(
    hidden_size: int,
    attention_heads: int,
    feedforward_size: int,
    dropout: float,
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=attention_heads,
        dim_feedforward=feedforward_size,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )


class ValueMLP(nn.Module):
    """State-value model for evaluating a position from the acting player's view."""

    def __init__(self, config: ValueModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ValueModelConfig()
        self.trunk = _make_mlp(
            input_size=self.config.observation_size,
            hidden_size=self.config.hidden_size,
            output_size=self.config.hidden_size,
            depth=self.config.depth,
            dropout=self.config.dropout,
            final_activation=True,
            final_layer_norm=True,
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_size // 2, 1),
            nn.Tanh(),
        )
        self.outcome_head = nn.Linear(self.config.hidden_size, 3)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(observation)
        return self.value_head(hidden).squeeze(-1)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, observation_size].")
        if observation.shape[1] != self.config.observation_size:
            raise ValueError("Unexpected observation feature size.")
        return self.trunk(observation)

    def outcome_logits(self, observation: torch.Tensor) -> torch.Tensor:
        return self.outcome_head(self.encode(observation))


def masked_policy_logits(policy_logits: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != legal_action_mask.shape:
        raise ValueError("Policy logits and legal action mask must have the same shape.")
    return policy_logits.masked_fill(~legal_action_mask, torch.finfo(policy_logits.dtype).min)


def masked_legal_action_scores(
    action_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> torch.Tensor:
    if action_scores.shape != legal_action_mask.shape:
        raise ValueError("Action scores and legal action mask must have the same shape.")
    return action_scores.masked_fill(~legal_action_mask, torch.finfo(action_scores.dtype).min)


def _make_mlp(
    input_size: int,
    hidden_size: int,
    output_size: int,
    depth: int,
    dropout: float,
    final_activation: bool,
    final_layer_norm: bool,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("MLP depth must be at least 1.")

    layers: list[nn.Module] = []
    current_size = input_size
    for layer_index in range(depth):
        is_final = layer_index == depth - 1
        next_size = output_size if is_final else hidden_size
        layers.append(nn.Linear(current_size, next_size))
        if not is_final or final_layer_norm:
            layers.append(nn.LayerNorm(next_size))
        if not is_final or final_activation:
            layers.append(nn.GELU())
        if dropout > 0.0 and (not is_final or final_activation):
            layers.append(nn.Dropout(dropout))
        current_size = next_size
    return nn.Sequential(*layers)
