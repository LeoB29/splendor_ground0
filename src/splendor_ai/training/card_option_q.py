"""Canonical target/action scorer used by experiment 081."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from splendor_ai.encoding import actions_by_canonical_successor_key
from splendor_ai.encoding.action_codec import ActionCodec
from splendor_ai.encoding.entity_encoding import (
    ACTION_NUMERIC_SIZE,
    ACTION_TYPE_COUNT,
    ENTITY_NUMERIC_SIZE,
    ENTITY_TYPE_COUNT,
    _action_target,
    _card_id,
    _card_row,
    card_vocabulary_size,
    collate_entity_action_batch,
    noble_vocabulary_size,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.state import Card, SplendorState
from splendor_ai.training.model import _transformer_encoder_layer


TARGET_PAD = 0
TARGET_CLEAR = 1
TARGET_VISIBLE = 2
TARGET_RESERVED = 3
TARGET_TYPE_COUNT = 4
OPTION_RELATION_SIZE = 6


@dataclass(frozen=True, slots=True)
class OptionTarget:
    """One observer-visible card commitment, or the explicit clear option."""

    source: str
    card: Card | None = None

    @property
    def key(self) -> str:
        return "CLEAR" if self.card is None else self.card.card_id

    @property
    def type_id(self) -> int:
        return {
            "clear": TARGET_CLEAR,
            "visible": TARGET_VISIBLE,
            "reserved": TARGET_RESERVED,
        }[self.source]


@dataclass(frozen=True, slots=True)
class OptionActionPair:
    target: OptionTarget
    action: Action
    action_index: int


@dataclass(frozen=True, slots=True)
class CardOptionQConfig:
    entity_hidden_size: int = 128
    item_hidden_size: int = 128
    scorer_hidden_size: int = 192
    entity_layers: int = 2
    attention_heads: int = 4
    attention_feedforward_size: int = 256
    id_embedding_size: int = 16
    dropout: float = 0.0


def enumerate_option_targets(state: SplendorState, player_id: int) -> list[OptionTarget]:
    targets = [OptionTarget("clear")]
    visible = sorted(
        (card for cards in state.visible_tier_cards.values() for card in cards),
        key=lambda card: card.card_id,
    )
    reserved = sorted(
        state.players[player_id].reserved_cards,
        key=lambda card: card.card_id,
    )
    targets.extend(OptionTarget("visible", card) for card in visible)
    targets.extend(OptionTarget("reserved", card) for card in reserved)
    return targets


def enumerate_option_action_pairs(
    state: SplendorState,
    legal_actions: Iterable[Action],
) -> list[OptionActionPair]:
    codec = ActionCodec()
    representatives = list(
        actions_by_canonical_successor_key(state, list(legal_actions)).values()
    )
    representatives.sort(key=lambda action: codec.encode(state, action))
    targets = enumerate_option_targets(state, state.current_player)
    return [
        OptionActionPair(target, action, codec.encode(state, action))
        for target in targets
        for action in representatives
    ]


def target_from_key(state: SplendorState, player_id: int, key: str) -> OptionTarget:
    for target in enumerate_option_targets(state, player_id):
        if target.key == key:
            return target
    raise KeyError(f"Target {key!r} is not public in this state.")


def collate_card_option_batch(
    rows: list[tuple[SplendorState, list[OptionActionPair]]],
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("rows must not be empty.")
    if any(not pairs for _state, pairs in rows):
        raise ValueError("Every state must have at least one target/action pair.")
    batch = collate_entity_action_batch(
        [(state, [pair.action for pair in pairs]) for state, pairs in rows],
        device=device,
    )
    width = int(batch["action_mask"].shape[1])
    target_device = None if device is None else torch.device(device)
    target_type_ids = torch.zeros((len(rows), width), dtype=torch.long, device=target_device)
    target_card_ids = torch.zeros((len(rows), width), dtype=torch.long, device=target_device)
    target_numeric = torch.zeros(
        (len(rows), width, ENTITY_NUMERIC_SIZE), dtype=torch.float32, device=target_device
    )
    relations = torch.zeros(
        (len(rows), width, OPTION_RELATION_SIZE), dtype=torch.float32, device=target_device
    )
    for row_index, (state, pairs) in enumerate(rows):
        for pair_index, pair in enumerate(pairs):
            target_type_ids[row_index, pair_index] = pair.target.type_id
            target_card_ids[row_index, pair_index] = _card_id(pair.target.card)
            target_numeric[row_index, pair_index] = torch.tensor(
                _card_row(pair.target.card), dtype=torch.float32, device=target_device
            )
            relations[row_index, pair_index] = torch.tensor(
                _option_relation(state, pair), dtype=torch.float32, device=target_device
            )
    batch["target_type_ids"] = target_type_ids
    batch["target_card_ids"] = target_card_ids
    batch["target_numeric_features"] = target_numeric
    batch["option_relation_features"] = relations
    return batch


def _option_relation(state: SplendorState, pair: OptionActionPair) -> tuple[float, ...]:
    target = pair.target.card
    addressed = _action_target(state, pair.action, state.current_player)
    same = target is not None and addressed is not None and target.card_id == addressed.card_id
    return (
        float(pair.target.source == "clear"),
        float(pair.target.source == "visible"),
        float(pair.target.source == "reserved"),
        float(same),
        float(same and pair.action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)),
        float(same and pair.action.action_type is ActionType.RESERVE_VISIBLE),
    )


class CardOptionQModel(nn.Module):
    """Permutation-invariant public-state encoder with independent pair scoring."""

    def __init__(self, config: CardOptionQConfig | None = None) -> None:
        super().__init__()
        self.config = config or CardOptionQConfig()
        if self.config.entity_hidden_size % self.config.attention_heads:
            raise ValueError("entity_hidden_size must be divisible by attention_heads.")
        size = self.config.id_embedding_size
        self.entity_type_embedding = nn.Embedding(ENTITY_TYPE_COUNT, size, padding_idx=0)
        self.card_embedding = nn.Embedding(card_vocabulary_size(), size, padding_idx=0)
        self.noble_embedding = nn.Embedding(noble_vocabulary_size(), size, padding_idx=0)
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
        self.action_type_embedding = nn.Embedding(ACTION_TYPE_COUNT, size, padding_idx=0)
        self.target_type_embedding = nn.Embedding(TARGET_TYPE_COUNT, size, padding_idx=0)
        self.action_input = nn.Sequential(
            nn.Linear(ACTION_NUMERIC_SIZE + 3 * size, self.config.item_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.item_hidden_size),
        )
        self.target_input = nn.Sequential(
            nn.Linear(ENTITY_NUMERIC_SIZE + 2 * size, self.config.item_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.item_hidden_size),
        )
        score_size = self.config.entity_hidden_size + 2 * self.config.item_hidden_size + OPTION_RELATION_SIZE
        self.score_head = nn.Sequential(
            nn.Linear(score_size, self.config.scorer_hidden_size),
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
        mask = batch["entity_mask"]
        hidden = self.entity_encoder(
            self.entity_input(entity_input), src_key_padding_mask=~mask
        )
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        state_hidden = hidden.sum(dim=1) / count

        action_input = torch.cat(
            [
                batch["action_numeric_features"],
                self.action_type_embedding(batch["action_type_ids"]),
                self.card_embedding(batch["action_card_ids"]),
                self.noble_embedding(batch["action_noble_ids"]),
            ],
            dim=-1,
        )
        target_input = torch.cat(
            [
                batch["target_numeric_features"],
                self.target_type_embedding(batch["target_type_ids"]),
                self.card_embedding(batch["target_card_ids"]),
            ],
            dim=-1,
        )
        action_hidden = self.action_input(action_input)
        target_hidden = self.target_input(target_input)
        expanded_state = state_hidden.unsqueeze(1).expand(-1, action_hidden.shape[1], -1)
        scores = self.score_head(
            torch.cat(
                [
                    expanded_state,
                    target_hidden,
                    action_hidden,
                    batch["option_relation_features"],
                ],
                dim=-1,
            )
        ).squeeze(-1)
        return scores.masked_fill(~batch["action_mask"], -torch.inf)


def save_card_option_checkpoint(
    path: Path,
    model: CardOptionQModel,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "card_option_q_controller_081",
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            **metadata,
        },
        path,
    )


def load_card_option_model(path: str | Path, device: str = "cpu") -> CardOptionQModel:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("model_kind") != "card_option_q_controller_081":
        raise ValueError("Checkpoint is not a card-option Q controller.")
    model = CardOptionQModel(CardOptionQConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
