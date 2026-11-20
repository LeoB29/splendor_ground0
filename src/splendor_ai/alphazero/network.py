"""Joint policy/value checkpoint and inference utilities for programme 090."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from splendor_ai.encoding import collate_entity_action_batch
from splendor_ai.engine.actions import Action
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)


MODEL_KIND = "alphazero_canonical_entity_policy_value_v1"


@dataclass(frozen=True, slots=True)
class AlphaZeroCheckpoint:
    path: Path
    generation: int
    model_config: CanonicalEntityActionScorerConfig
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NetworkPrediction:
    logits: tuple[float, ...]
    value: float


def default_model_config() -> CanonicalEntityActionScorerConfig:
    """A compact CPU-friendly model for mechanism validation."""

    return CanonicalEntityActionScorerConfig(
        entity_hidden_size=96,
        action_hidden_size=96,
        scorer_hidden_size=128,
        entity_layers=2,
        action_layers=2,
        attention_heads=4,
        attention_feedforward_size=192,
        id_embedding_size=16,
        dropout=0.0,
    )


def create_initial_checkpoint(
    path: str | Path,
    *,
    seed: int,
    model_config: CanonicalEntityActionScorerConfig | None = None,
) -> AlphaZeroCheckpoint:
    """Create a reproducible generation-zero checkpoint from random weights."""

    torch.manual_seed(seed)
    config = model_config or default_model_config()
    model = CanonicalEntityActionScorerModel(config)
    save_checkpoint(
        path,
        model=model,
        generation=0,
        metadata={"initialization_seed": int(seed), "parent_checkpoint": None},
    )
    return load_checkpoint(path, device="cpu")[1]


def save_checkpoint(
    path: str | Path,
    *,
    model: CanonicalEntityActionScorerModel,
    generation: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": MODEL_KIND,
            "format_version": 1,
            "generation": int(generation),
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output_path,
    )
    return output_path


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[CanonicalEntityActionScorerModel, AlphaZeroCheckpoint]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("model_kind") != MODEL_KIND:
        raise ValueError(
            f"Unsupported AlphaZero checkpoint kind: {payload.get('model_kind')!r}"
        )
    config = CanonicalEntityActionScorerConfig(**payload.get("model_config", {}))
    model = CanonicalEntityActionScorerModel(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    identity = AlphaZeroCheckpoint(
        path=checkpoint_path,
        generation=int(payload.get("generation", 0)),
        model_config=config,
        metadata=dict(payload.get("metadata", {})),
    )
    return model, identity


class AlphaZeroInference:
    """Batched information-legal inference from one joint checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.role_aware = False
        if payload.get("model_kind") == MODEL_KIND:
            self.model, self.checkpoint = load_checkpoint(
                checkpoint_path,
                device=self.device,
            )
        else:
            # Local import avoids a module cycle: the capacity loader reuses the
            # stable v1 loader for construction from exact parent checkpoints.
            from .capacity_network import CAPACITY_MODEL_KIND, load_capacity_checkpoint
            from .role_readout_network import (
                READOUT_ROLE_MODEL_KIND,
                load_readout_role_checkpoint,
            )
            from .role_film_network import ROLE_FILM_MODEL_KIND, load_role_film_checkpoint
            from .role_network import ROLE_MODEL_KIND, load_role_checkpoint
            from .role_state_network import (
                ROLE_STATE_MODEL_KIND,
                load_role_state_checkpoint,
            )

            if payload.get("model_kind") == ROLE_MODEL_KIND:
                self.model, self.checkpoint = load_role_checkpoint(
                    checkpoint_path,
                    device=self.device,
                )
                self.role_aware = True
            elif payload.get("model_kind") == READOUT_ROLE_MODEL_KIND:
                self.model, self.checkpoint = load_readout_role_checkpoint(
                    checkpoint_path,
                    device=self.device,
                )
                self.role_aware = True
            elif payload.get("model_kind") == ROLE_FILM_MODEL_KIND:
                self.model, self.checkpoint = load_role_film_checkpoint(
                    checkpoint_path,
                    device=self.device,
                )
                self.role_aware = True
            elif payload.get("model_kind") == ROLE_STATE_MODEL_KIND:
                self.model, self.checkpoint = load_role_state_checkpoint(
                    checkpoint_path,
                    device=self.device,
                )
                self.role_aware = True
            elif payload.get("model_kind") == CAPACITY_MODEL_KIND:
                self.model, self.checkpoint = load_capacity_checkpoint(
                    checkpoint_path,
                    device=self.device,
                )
            else:
                raise ValueError(
                    f"Unsupported AlphaZero inference checkpoint: {payload.get('model_kind')!r}"
                )

    def predict(
        self,
        batches: Iterable[tuple[SplendorState, list[Action]]],
    ) -> list[NetworkPrediction]:
        items = list(batches)
        if not items:
            return []
        if any(not actions for _state, actions in items):
            raise ValueError("Every inference state must contain a legal action.")
        encoded = collate_entity_action_batch(items, device=self.device)
        if self.role_aware:
            encoded["observer_is_start_player"] = torch.tensor(
                [state.current_player == state.start_player for state, _actions in items],
                dtype=torch.long,
                device=self.device,
            )
        with torch.no_grad():
            logits, values = self.model(**encoded)
        predictions: list[NetworkPrediction] = []
        for row, (_state, actions) in enumerate(items):
            count = len(actions)
            predictions.append(
                NetworkPrediction(
                    logits=tuple(float(value) for value in logits[row, :count].tolist()),
                    value=float(values[row].item()),
                )
            )
        return predictions

    def predict_one(
        self,
        state: SplendorState,
        actions: list[Action],
    ) -> NetworkPrediction:
        return self.predict([(state, actions)])[0]


class HybridAlphaZeroInference:
    """Inference whose policy and value heads come from separate checkpoints."""

    def __init__(
        self,
        policy_checkpoint_path: str | Path,
        value_checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.policy_model, self.policy_checkpoint = load_checkpoint(
            policy_checkpoint_path, device=self.device
        )
        self.value_model, self.value_checkpoint = load_checkpoint(
            value_checkpoint_path, device=self.device
        )
        if self.policy_checkpoint.model_config != self.value_checkpoint.model_config:
            raise ValueError("Hybrid policy and value checkpoints use different models.")
        # Bot adapters use this identity only for provenance/bookkeeping. The
        # hybrid's operative policy identity is the least surprising generic
        # checkpoint identity; the value identity remains separately exposed.
        self.checkpoint = self.policy_checkpoint

    def predict(
        self,
        batches: Iterable[tuple[SplendorState, list[Action]]],
    ) -> list[NetworkPrediction]:
        items = list(batches)
        if not items:
            return []
        if any(not actions for _state, actions in items):
            raise ValueError("Every inference state must contain a legal action.")
        encoded = collate_entity_action_batch(items, device=self.device)
        with torch.no_grad():
            policy_logits, _policy_values = self.policy_model(**encoded)
            _value_logits, values = self.value_model(**encoded)
        return [
            NetworkPrediction(
                logits=tuple(
                    float(value)
                    for value in policy_logits[row, : len(actions)].tolist()
                ),
                value=float(values[row].item()),
            )
            for row, (_state, actions) in enumerate(items)
        ]

    def predict_one(
        self,
        state: SplendorState,
        actions: list[Action],
    ) -> NetworkPrediction:
        return self.predict([(state, actions)])[0]
