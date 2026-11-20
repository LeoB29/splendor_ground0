"""Observer-legal target-only value model used by experiment 082."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from splendor_ai.encoding.entity_encoding import (
    ENTITY_NUMERIC_SIZE,
    ENTITY_TYPE_COUNT,
    _card_id,
    _card_row,
    card_vocabulary_size,
    collate_canonical_state_batch,
    noble_vocabulary_size,
)
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.card_option_q import (
    TARGET_TYPE_COUNT,
    OptionTarget,
)
from splendor_ai.training.model import _transformer_encoder_layer


@dataclass(frozen=True, slots=True)
class CardTargetValueConfig:
    entity_hidden_size: int = 128
    target_hidden_size: int = 128
    scorer_hidden_size: int = 192
    entity_layers: int = 2
    attention_heads: int = 4
    attention_feedforward_size: int = 256
    id_embedding_size: int = 16
    dropout: float = 0.0


def collate_card_target_batch(
    rows: list[tuple[SplendorState, list[OptionTarget]]],
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("rows must not be empty.")
    if any(not targets for _state, targets in rows):
        raise ValueError("Every state must have at least one target.")
    batch = collate_canonical_state_batch(
        [(state, state.current_player) for state, _targets in rows],
        device=device,
    )
    width = max(len(targets) for _state, targets in rows)
    target_device = None if device is None else torch.device(device)
    type_ids = torch.zeros((len(rows), width), dtype=torch.long, device=target_device)
    card_ids = torch.zeros((len(rows), width), dtype=torch.long, device=target_device)
    numeric = torch.zeros(
        (len(rows), width, ENTITY_NUMERIC_SIZE),
        dtype=torch.float32,
        device=target_device,
    )
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=target_device)
    for row_index, (_state, targets) in enumerate(rows):
        for target_index, target in enumerate(targets):
            type_ids[row_index, target_index] = target.type_id
            card_ids[row_index, target_index] = _card_id(target.card)
            numeric[row_index, target_index] = torch.tensor(
                _card_row(target.card), dtype=torch.float32, device=target_device
            )
            mask[row_index, target_index] = True
    batch.update(
        {
            "target_type_ids": type_ids,
            "target_card_ids": card_ids,
            "target_numeric_features": numeric,
            "target_mask": mask,
        }
    )
    return batch


class CardTargetValueModel(nn.Module):
    """Permutation-invariant public-state encoder with independent target scores."""

    def __init__(self, config: CardTargetValueConfig | None = None) -> None:
        super().__init__()
        self.config = config or CardTargetValueConfig()
        if self.config.entity_hidden_size % self.config.attention_heads:
            raise ValueError("entity_hidden_size must be divisible by attention_heads.")
        size = self.config.id_embedding_size
        self.entity_type_embedding = nn.Embedding(ENTITY_TYPE_COUNT, size, padding_idx=0)
        self.card_embedding = nn.Embedding(card_vocabulary_size(), size, padding_idx=0)
        self.noble_embedding = nn.Embedding(noble_vocabulary_size(), size, padding_idx=0)
        self.target_type_embedding = nn.Embedding(TARGET_TYPE_COUNT, size, padding_idx=0)
        self.entity_input = nn.Sequential(
            nn.Linear(ENTITY_NUMERIC_SIZE + 3 * size, self.config.entity_hidden_size),
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
        self.target_input = nn.Sequential(
            nn.Linear(ENTITY_NUMERIC_SIZE + 2 * size, self.config.target_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.target_hidden_size),
        )
        self.score_head = nn.Sequential(
            nn.Linear(
                self.config.entity_hidden_size + self.config.target_hidden_size,
                self.config.scorer_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(self.config.scorer_hidden_size, 1),
        )

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        entity_input = torch.cat(
            [
                batch["entity_numeric_features"],
                self.entity_type_embedding(batch["entity_type_ids"]),
                self.card_embedding(batch["entity_card_ids"]),
                self.noble_embedding(batch["entity_noble_ids"]),
            ],
            dim=-1,
        )
        entity_mask = batch["entity_mask"]
        hidden = self.entity_encoder(
            self.entity_input(entity_input), src_key_padding_mask=~entity_mask
        )
        hidden = hidden.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        count = entity_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        state_hidden = hidden.sum(dim=1) / count
        target_input = torch.cat(
            [
                batch["target_numeric_features"],
                self.target_type_embedding(batch["target_type_ids"]),
                self.card_embedding(batch["target_card_ids"]),
            ],
            dim=-1,
        )
        target_hidden = self.target_input(target_input)
        expanded_state = state_hidden.unsqueeze(1).expand(-1, target_hidden.shape[1], -1)
        scores = self.score_head(torch.cat([expanded_state, target_hidden], dim=-1)).squeeze(-1)
        return scores.masked_fill(~batch["target_mask"], -torch.inf)


def save_card_target_checkpoint(
    path: Path,
    model: CardTargetValueModel,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "card_target_value_selector_082",
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            **metadata,
        },
        path,
    )


def load_card_target_model(
    path: str | Path,
    device: str = "cpu",
) -> CardTargetValueModel:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("model_kind") != "card_target_value_selector_082":
        raise ValueError("Checkpoint is not an 082 card-target value selector.")
    model = CardTargetValueModel(CardTargetValueConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
