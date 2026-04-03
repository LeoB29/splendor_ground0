"""Policy/value models for Splendor training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from splendor_ai.encoding import ActionCodec


@dataclass(frozen=True, slots=True)
class PolicyValueModelConfig:
    observation_size: int = 256
    action_space_size: int = ActionCodec().action_space_size
    trunk_hidden_size: int = 256
    trunk_depth: int = 3
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


def masked_policy_logits(policy_logits: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != legal_action_mask.shape:
        raise ValueError("Policy logits and legal action mask must have the same shape.")
    return policy_logits.masked_fill(~legal_action_mask, torch.finfo(policy_logits.dtype).min)
