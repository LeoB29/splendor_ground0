"""Policy/value models for Splendor training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from splendor_ai.encoding import ACTION_FEATURE_SIZE, ActionCodec


@dataclass(frozen=True, slots=True)
class PolicyValueModelConfig:
    observation_size: int = 256
    action_space_size: int = ActionCodec().action_space_size
    trunk_hidden_size: int = 256
    trunk_depth: int = 3
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class LegalActionScorerConfig:
    observation_size: int = 256
    action_feature_size: int = ACTION_FEATURE_SIZE
    state_hidden_size: int = 256
    action_hidden_size: int = 128
    scorer_hidden_size: int = 256
    state_depth: int = 3
    action_depth: int = 2
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class ValueModelConfig:
    observation_size: int = 256
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

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, observation_size].")
        if observation.shape[1] != self.config.observation_size:
            raise ValueError("Unexpected observation feature size.")
        return self.value_head(self.trunk(observation)).squeeze(-1)


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
