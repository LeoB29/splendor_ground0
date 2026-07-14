"""Low-dimensional, reproducible adapters for antithetic ES experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch


TARGET_WEIGHT_KEYS = (
    "action_set_encoder.layers.1.self_attn.in_proj_weight",
    "action_set_encoder.layers.1.self_attn.out_proj.weight",
    "action_set_encoder.layers.1.linear1.weight",
    "action_set_encoder.layers.1.linear2.weight",
)


@dataclass(frozen=True, slots=True)
class RankAdapterSpec:
    """Fixed-A, trainable-B rank adapter over the final attention block."""

    rank: int = 4
    seed: int = 0
    target_weight_keys: tuple[str, ...] = TARGET_WEIGHT_KEYS


def adapter_parameter_count(state_dict: dict[str, torch.Tensor], spec: RankAdapterSpec) -> int:
    _validate_targets(state_dict, spec)
    return sum(int(state_dict[key].shape[0]) * spec.rank for key in spec.target_weight_keys)


def normalized_direction(size: int, seed: int) -> torch.Tensor:
    if size <= 0:
        raise ValueError("Direction size must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(size, generator=generator)
    return direction / direction.norm().clamp_min(torch.finfo(direction.dtype).eps)


def write_adapter_checkpoint(
    base_checkpoint_path: str | Path,
    output_path: str | Path,
    theta: torch.Tensor,
    spec: RankAdapterSpec,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Materialize ``base + B @ A`` as a standard inference checkpoint.

    The saved file is deliberately ordinary: benchmark and corpus code do not
    need adapter-aware model classes, and every candidate remains auditable.
    """

    base_path = Path(base_checkpoint_path)
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError("RankAdapterSpec only supports legal_action_attention_scorer checkpoints.")
    state_dict = checkpoint["model_state_dict"]
    parameter_count = adapter_parameter_count(state_dict, spec)
    theta = theta.detach().cpu().to(dtype=torch.float32).flatten()
    if theta.numel() != parameter_count:
        raise ValueError(f"Expected {parameter_count} adapter parameters, got {theta.numel()}.")

    adapted = {key: value.detach().clone() for key, value in state_dict.items()}
    cursor = 0
    for key in spec.target_weight_keys:
        weight = adapted[key]
        output_size, input_size = weight.shape
        segment_size = output_size * spec.rank
        b_matrix = theta[cursor : cursor + segment_size].reshape(output_size, spec.rank)
        a_matrix = _fixed_a_matrix(key, spec, input_size)
        adapted[key] = weight + (b_matrix @ a_matrix).to(dtype=weight.dtype)
        cursor += segment_size

    output = dict(checkpoint)
    output["model_state_dict"] = adapted
    output["adapter_provenance"] = {
        "family": "rank4_final_attention_es",
        "rank": spec.rank,
        "seed": spec.seed,
        "target_weight_keys": list(spec.target_weight_keys),
        "parameter_count": parameter_count,
        "theta_l2_norm": float(theta.norm().item()),
        "base_checkpoint": str(base_path),
        **(provenance or {}),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    return path


def _fixed_a_matrix(key: str, spec: RankAdapterSpec, input_size: int) -> torch.Tensor:
    # hashlib keeps this independent of Python's randomized hash seed.
    digest = hashlib.sha256(f"{spec.seed}:{key}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((spec.rank, input_size), generator=generator) / (input_size**0.5)


def _validate_targets(state_dict: dict[str, torch.Tensor], spec: RankAdapterSpec) -> None:
    if spec.rank <= 0:
        raise ValueError("Adapter rank must be positive.")
    if not spec.target_weight_keys:
        raise ValueError("Adapter must target at least one weight matrix.")
    for key in spec.target_weight_keys:
        tensor = state_dict.get(key)
        if tensor is None or tensor.ndim != 2:
            raise ValueError(f"Adapter target is missing or not matrix-shaped: {key}")
