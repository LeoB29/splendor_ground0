"""History-aware residual policy built on the frozen legal 035 scorer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from splendor_ai.cfr.full_game import PublicHistoryEncoding, encode_public_history
from splendor_ai.encoding import (
    ACTION_FEATURE_SIZE,
    OBSERVATION_SIZE,
    encode_legal_action_features,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.encoding.entity_encoding import (
    ACTION_NUMERIC_SIZE,
    ACTION_TYPE_COUNT,
    card_vocabulary_size,
    noble_vocabulary_size,
)
from splendor_ai.encoding.information_state import PublicActionEvent, public_action_event
from splendor_ai.engine.actions import Action
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
)


MODEL_KIND = "recurrent_legal_action_attention_residual"


@dataclass(frozen=True, slots=True)
class RecurrentPolicyConfig:
    action_type_embedding_size: int = 8
    card_embedding_size: int = 16
    noble_embedding_size: int = 8
    history_numeric_hidden_size: int = 32
    history_hidden_size: int = 64
    residual_hidden_size: int = 128


@dataclass(frozen=True, slots=True)
class RecurrentPolicyInput:
    """One information-legal decision row suitable for padded batching."""

    observation: tuple[float, ...]
    legal_action_features: tuple[tuple[float, ...], ...]
    history: PublicHistoryEncoding
    player: int
    turn: int


class PublicHistoryTracker:
    """Per-game public transcript with explicit reset and serialization."""

    def __init__(self, events: Iterable[PublicActionEvent] = ()) -> None:
        self._events = list(events)

    @property
    def events(self) -> tuple[PublicActionEvent, ...]:
        return tuple(self._events)

    def reset(self) -> None:
        self._events.clear()

    def record(self, before: SplendorState, action: Action) -> PublicActionEvent:
        event = public_action_event(before, action)
        self._events.append(event)
        return event

    def encoding(self, observer: int) -> PublicHistoryEncoding:
        return encode_public_history(self._events, observer)

    def state_dict(self) -> dict[str, object]:
        return {"schema": "public-history-tracker-v1", "events": self.events}

    def load_state_dict(self, payload: dict[str, object]) -> None:
        if payload.get("schema") != "public-history-tracker-v1":
            raise ValueError("Unsupported public-history tracker schema.")
        events = payload.get("events")
        if not isinstance(events, (list, tuple)):
            raise ValueError("Public-history tracker events must be a sequence.")
        for event in events:
            if (
                not isinstance(event, (list, tuple))
                or len(event) != 10
                or event[0] != "splendor-public-action-v1"
            ):
                raise ValueError("Invalid public-history tracker event.")
        self._events = [tuple(event) for event in events]


class RecurrentLegalActionAttentionScorerModel(nn.Module):
    """Frozen 035 policy plus a trainable, zero-initialized history residual."""

    def __init__(
        self,
        parent: LegalActionAttentionScorerModel,
        config: RecurrentPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        self.parent = parent
        self.config = config or RecurrentPolicyConfig()
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
        residual_width = (
            self.parent.config.state_hidden_size
            + self.parent.config.action_hidden_size
            + 1
            + self.config.history_hidden_size
        )
        self.residual_hidden = nn.Sequential(
            nn.Linear(residual_width, self.config.residual_hidden_size),
            nn.GELU(),
        )
        self.residual_output = nn.Linear(self.config.residual_hidden_size, 1)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    @property
    def parent_config(self) -> LegalActionAttentionScorerConfig:
        return self.parent.config

    def train(
        self,
        mode: bool = True,
    ) -> RecurrentLegalActionAttentionScorerModel:
        super().train(mode)
        self.parent.eval()
        return self

    def _event_features(
        self,
        action_type_ids: torch.Tensor,
        card_ids: torch.Tensor,
        noble_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.history_event_projection(
            torch.cat(
                [
                    self.history_action_type_embedding(action_type_ids),
                    self.history_card_embedding(card_ids),
                    self.history_noble_embedding(noble_ids),
                    self.history_numeric_projection(numeric_features),
                ],
                dim=-1,
            )
        )

    def encode_history(
        self,
        action_type_ids: torch.Tensor,
        card_ids: torch.Tensor,
        noble_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode padded chronological histories, returning zero for empties."""

        if action_type_ids.ndim != 2:
            raise ValueError("history action ids must have shape [batch, events].")
        if numeric_features.ndim != 3:
            raise ValueError("history numeric features must have rank three.")
        if numeric_features.shape[:2] != action_type_ids.shape:
            raise ValueError("history numeric rows must align with history ids.")
        if numeric_features.shape[2] != ACTION_NUMERIC_SIZE:
            raise ValueError("Unexpected public-history numeric feature size.")
        if any(
            tensor.shape != action_type_ids.shape
            for tensor in (card_ids, noble_ids, history_mask)
        ):
            raise ValueError("All public-history tensors must align.")

        lengths = history_mask.to(dtype=torch.long).sum(dim=1)
        context = numeric_features.new_zeros(
            (numeric_features.shape[0], self.config.history_hidden_size)
        )
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
        if nonempty.numel() == 0:
            return context

        events = self._event_features(
            action_type_ids,
            card_ids,
            noble_ids,
            numeric_features,
        )
        packed = pack_padded_sequence(
            events.index_select(0, nonempty),
            lengths.index_select(0, nonempty).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _output, hidden = self.history_gru(packed)
        context.index_copy_(0, nonempty, hidden[-1])
        return context

    def step_history(
        self,
        action_type_ids: torch.Tensor,
        card_ids: torch.Tensor,
        noble_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one public event for online recurrent-state maintenance."""

        if action_type_ids.ndim != 1:
            raise ValueError("step history ids must have shape [batch].")
        batch_size = action_type_ids.shape[0]
        if any(
            tensor.shape != (batch_size,)
            for tensor in (card_ids, noble_ids)
        ):
            raise ValueError("step history ids must align.")
        if numeric_features.shape != (batch_size, ACTION_NUMERIC_SIZE):
            raise ValueError("step history numeric features have the wrong shape.")
        event = self._event_features(
            action_type_ids,
            card_ids,
            noble_ids,
            numeric_features,
        ).unsqueeze(1)
        _output, next_hidden = self.history_gru(event, hidden)
        return next_hidden[-1], next_hidden

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
            parent_value = self.parent.value_head(state_hidden).squeeze(-1)

        history_context = self.encode_history(
            history_action_type_ids,
            history_card_ids,
            history_noble_ids,
            history_numeric_features,
            history_mask,
        )
        expanded_history = history_context.unsqueeze(1).expand(
            batch_size,
            legal_count,
            self.config.history_hidden_size,
        )
        residual_input = torch.cat(
            [
                expanded_state.detach(),
                contextual.detach(),
                parent_scores.detach().unsqueeze(-1),
                expanded_history,
            ],
            dim=-1,
        )
        residual = self.residual_output(self.residual_hidden(residual_input)).squeeze(-1)
        residual = residual.masked_fill(~legal_action_mask, 0.0)
        return parent_scores + residual, parent_value


def build_recurrent_policy_input(
    state: SplendorState,
    legal_actions: list[Action],
    public_history: Iterable[PublicActionEvent],
) -> RecurrentPolicyInput:
    if not legal_actions:
        raise ValueError("A recurrent policy input requires at least one legal action.")
    observation = fit_feature_vector(
        encode_public_observation_tensor(state, state.current_player).vector,
        OBSERVATION_SIZE,
    )
    action_rows = tuple(
        fit_feature_vector(features.vector, ACTION_FEATURE_SIZE)
        for features in encode_legal_action_features(
            state,
            legal_actions,
            player_id=state.current_player,
        )
    )
    return RecurrentPolicyInput(
        observation=observation,
        legal_action_features=action_rows,
        history=encode_public_history(public_history, state.current_player),
        player=state.current_player,
        turn=state.turn_index,
    )


def collate_recurrent_policy_inputs(
    rows: list[RecurrentPolicyInput],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("rows must not be empty.")
    target = torch.device(device)
    action_width = max(len(row.legal_action_features) for row in rows)
    history_width = max(max(len(row.history.action_type_ids), 1) for row in rows)
    observation_size = len(rows[0].observation)
    action_size = len(rows[0].legal_action_features[0])
    if any(len(row.observation) != observation_size for row in rows):
        raise ValueError("Observation widths must be constant.")
    if any(
        len(features) != action_size
        for row in rows
        for features in row.legal_action_features
    ):
        raise ValueError("Legal-action feature widths must be constant.")

    batch_size = len(rows)
    observations = torch.tensor(
        [row.observation for row in rows],
        dtype=torch.float32,
        device=target,
    )
    actions = torch.zeros(
        (batch_size, action_width, action_size),
        dtype=torch.float32,
        device=target,
    )
    action_mask = torch.zeros(
        (batch_size, action_width),
        dtype=torch.bool,
        device=target,
    )
    history_action = torch.zeros(
        (batch_size, history_width),
        dtype=torch.long,
        device=target,
    )
    history_card = torch.zeros_like(history_action)
    history_noble = torch.zeros_like(history_action)
    history_numeric = torch.zeros(
        (batch_size, history_width, ACTION_NUMERIC_SIZE),
        dtype=torch.float32,
        device=target,
    )
    history_mask = torch.zeros(
        (batch_size, history_width),
        dtype=torch.bool,
        device=target,
    )
    for index, row in enumerate(rows):
        action_count = len(row.legal_action_features)
        actions[index, :action_count] = torch.tensor(
            row.legal_action_features,
            dtype=torch.float32,
            device=target,
        )
        action_mask[index, :action_count] = True
        history_count = len(row.history.action_type_ids)
        if history_count:
            history_action[index, :history_count] = torch.tensor(
                row.history.action_type_ids,
                dtype=torch.long,
                device=target,
            )
            history_card[index, :history_count] = torch.tensor(
                row.history.card_ids,
                dtype=torch.long,
                device=target,
            )
            history_noble[index, :history_count] = torch.tensor(
                row.history.noble_ids,
                dtype=torch.long,
                device=target,
            )
            history_numeric[index, :history_count] = torch.tensor(
                row.history.numeric_features,
                dtype=torch.float32,
                device=target,
            )
            history_mask[index, :history_count] = True
    return {
        "observation": observations,
        "legal_action_features": actions,
        "legal_action_mask": action_mask,
        "history_action_type_ids": history_action,
        "history_card_ids": history_card,
        "history_noble_ids": history_noble,
        "history_numeric_features": history_numeric,
        "history_mask": history_mask,
        "players": torch.tensor(
            [row.player for row in rows],
            dtype=torch.long,
            device=target,
        ),
        "turns": torch.tensor(
            [row.turn for row in rows],
            dtype=torch.long,
            device=target,
        ),
    }


def load_recurrent_policy_from_parent(
    parent_checkpoint: str | Path,
    *,
    device: str = "cpu",
    config: RecurrentPolicyConfig | None = None,
) -> RecurrentLegalActionAttentionScorerModel:
    checkpoint = torch.load(parent_checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError("Recurrent parent must be a legal-action attention scorer.")
    parent = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(**checkpoint.get("model_config", {}))
    )
    parent.load_state_dict(checkpoint["model_state_dict"])
    model = RecurrentLegalActionAttentionScorerModel(parent, config=config)
    return model.to(device)


def save_recurrent_policy_checkpoint(
    path: str | Path,
    model: RecurrentLegalActionAttentionScorerModel,
    provenance: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": MODEL_KIND,
            "model_state_dict": model.state_dict(),
            "parent_model_config": asdict(model.parent_config),
            "recurrent_policy_config": asdict(model.config),
            "provenance": provenance,
        },
        target,
    )


def load_recurrent_policy_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> RecurrentLegalActionAttentionScorerModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != MODEL_KIND:
        raise ValueError("Unsupported recurrent policy checkpoint kind.")
    parent = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(**checkpoint["parent_model_config"])
    )
    model = RecurrentLegalActionAttentionScorerModel(
        parent,
        config=RecurrentPolicyConfig(**checkpoint["recurrent_policy_config"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)
