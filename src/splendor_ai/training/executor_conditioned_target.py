"""Executor-conditioned target-advantage model for experiment 083."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.encoding.entity_encoding import (
    ACTION_NUMERIC_SIZE,
    ACTION_TYPE_COUNT,
    ENTITY_NUMERIC_SIZE,
    ENTITY_TYPE_COUNT,
    _card_id,
    _card_row,
    card_vocabulary_size,
    collate_entity_action_batch,
    noble_vocabulary_size,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.card_option_q import TARGET_TYPE_COUNT, OptionTarget
from splendor_ai.training.model import _transformer_encoder_layer


@dataclass(frozen=True, slots=True)
class ExecutorTargetCandidate:
    target: OptionTarget
    action: Action
    action_index: int


@dataclass(frozen=True, slots=True)
class ExecutorConditionedTargetConfig:
    entity_hidden_size: int = 128
    item_hidden_size: int = 128
    scorer_hidden_size: int = 192
    entity_layers: int = 2
    attention_heads: int = 4
    attention_feedforward_size: int = 256
    id_embedding_size: int = 16
    dropout: float = 0.0


def reconstruct_executor_candidates(
    state: SplendorState,
    target_rows: list[dict[str, Any]],
    legal_actions: list[Action],
) -> list[ExecutorTargetCandidate]:
    from splendor_ai.training.card_option_q import enumerate_option_targets

    targets = enumerate_option_targets(state, state.current_player)
    if len(targets) != len(target_rows):
        raise ValueError("Stored and reconstructed target counts differ.")
    codec = ActionCodec()
    representatives = list(
        actions_by_canonical_successor_key(state, legal_actions).values()
    )
    action_by_index = {codec.encode(state, action): action for action in representatives}
    candidates = []
    for target, row in zip(targets, target_rows, strict=True):
        if target.key != row["target_key"] or target.source != row["target_source"]:
            raise ValueError("Stored target identity does not match reconstruction.")
        action_index = int(row["executor_action_index"])
        action = action_by_index.get(action_index)
        if action is None:
            raise ValueError("Stored executor action is not a canonical legal action.")
        candidates.append(ExecutorTargetCandidate(target, action, action_index))
    if candidates[0].target.key != "CLEAR":
        raise ValueError("CLEAR must be the first executor-conditioned candidate.")
    return candidates


def collate_executor_target_batch(
    rows: list[tuple[SplendorState, list[ExecutorTargetCandidate]]],
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("rows must not be empty.")
    if any(not candidates for _state, candidates in rows):
        raise ValueError("Every state must have at least one candidate.")
    batch = collate_entity_action_batch(
        [(state, [candidate.action for candidate in candidates]) for state, candidates in rows],
        device=device,
    )
    width = int(batch["action_mask"].shape[1])
    target_device = None if device is None else torch.device(device)
    type_ids = torch.zeros((len(rows), width), dtype=torch.long, device=target_device)
    card_ids = torch.zeros((len(rows), width), dtype=torch.long, device=target_device)
    numeric = torch.zeros(
        (len(rows), width, ENTITY_NUMERIC_SIZE),
        dtype=torch.float32,
        device=target_device,
    )
    for row_index, (_state, candidates) in enumerate(rows):
        for candidate_index, candidate in enumerate(candidates):
            type_ids[row_index, candidate_index] = candidate.target.type_id
            card_ids[row_index, candidate_index] = _card_id(candidate.target.card)
            numeric[row_index, candidate_index] = torch.tensor(
                _card_row(candidate.target.card),
                dtype=torch.float32,
                device=target_device,
            )
    batch.update(
        {
            "target_type_ids": type_ids,
            "target_card_ids": card_ids,
            "target_numeric_features": numeric,
            "target_mask": batch["action_mask"],
        }
    )
    return batch


class ExecutorConditionedTargetModel(nn.Module):
    """Target/action queries cross-attend to observer-legal state entities."""

    def __init__(self, config: ExecutorConditionedTargetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExecutorConditionedTargetConfig()
        if self.config.entity_hidden_size % self.config.attention_heads:
            raise ValueError("entity_hidden_size must be divisible by attention_heads.")
        size = self.config.id_embedding_size
        self.entity_type_embedding = nn.Embedding(ENTITY_TYPE_COUNT, size, padding_idx=0)
        self.card_embedding = nn.Embedding(card_vocabulary_size(), size, padding_idx=0)
        self.noble_embedding = nn.Embedding(noble_vocabulary_size(), size, padding_idx=0)
        self.action_type_embedding = nn.Embedding(ACTION_TYPE_COUNT, size, padding_idx=0)
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
            nn.Linear(ENTITY_NUMERIC_SIZE + 2 * size, self.config.item_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.item_hidden_size),
        )
        self.action_input = nn.Sequential(
            nn.Linear(ACTION_NUMERIC_SIZE + 3 * size, self.config.item_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.item_hidden_size),
        )
        self.query_input = nn.Sequential(
            nn.Linear(2 * self.config.item_hidden_size, self.config.entity_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.entity_hidden_size),
        )
        self.target_cross_attention = nn.MultiheadAttention(
            self.config.entity_hidden_size,
            self.config.attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.Linear(
                self.config.entity_hidden_size + 2 * self.config.item_hidden_size,
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
        entity_hidden = self.entity_encoder(
            self.entity_input(entity_input), src_key_padding_mask=~entity_mask
        )
        target_input = torch.cat(
            [
                batch["target_numeric_features"],
                self.target_type_embedding(batch["target_type_ids"]),
                self.card_embedding(batch["target_card_ids"]),
            ],
            dim=-1,
        )
        action_input = torch.cat(
            [
                batch["action_numeric_features"],
                self.action_type_embedding(batch["action_type_ids"]),
                self.card_embedding(batch["action_card_ids"]),
                self.noble_embedding(batch["action_noble_ids"]),
            ],
            dim=-1,
        )
        target_hidden = self.target_input(target_input)
        action_hidden = self.action_input(action_input)
        query = self.query_input(torch.cat([target_hidden, action_hidden], dim=-1))
        context, _weights = self.target_cross_attention(
            query,
            entity_hidden,
            entity_hidden,
            key_padding_mask=~entity_mask,
            need_weights=False,
        )
        raw_scores = self.score_head(
            torch.cat([context, target_hidden, action_hidden], dim=-1)
        ).squeeze(-1)
        advantages = raw_scores - raw_scores[:, :1]
        return advantages.masked_fill(~batch["target_mask"], -torch.inf)


def save_executor_conditioned_checkpoint(
    path: Path,
    model: ExecutorConditionedTargetModel,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "executor_conditioned_target_selector_083",
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            **metadata,
        },
        path,
    )


def load_executor_conditioned_model(
    path: str | Path,
    device: str = "cpu",
) -> ExecutorConditionedTargetModel:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("model_kind") != "executor_conditioned_target_selector_083":
        raise ValueError("Checkpoint is not an executor-conditioned 083 selector.")
    model = ExecutorConditionedTargetModel(
        ExecutorConditionedTargetConfig(**payload["model_config"])
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
