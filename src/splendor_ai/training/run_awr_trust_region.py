"""Project an AWR proposal toward its parent under a validation KL constraint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.model import masked_legal_action_scores
from splendor_ai.training.run_trajectory_awr import compute_awr_loss
from splendor_ai.training.run_trajectory_policy_gradient import _load_attention_model


@dataclass(frozen=True, slots=True)
class AwrTrustRegionConfig:
    replay_path: Path
    parent_checkpoint: Path
    proposal_checkpoint: Path
    output_dir: Path
    validation_seed_min: int
    target_raw_kl: float = 0.008
    bisection_steps: int = 12
    batch_size: int = 256
    advantage_temperature: float = 0.25
    max_advantage_weight: float = 10.0
    kl_weight: float = 1.0
    entropy_weight: float = 0.001
    sampling_epsilon: float = 0.5
    sampling_temperature: float = 1.0
    sampling_top_k: int = 3
    device: str = "cpu"


def interpolate_state_dict(
    parent: dict[str, torch.Tensor],
    proposal: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Interpolation alpha must be in [0, 1].")
    if parent.keys() != proposal.keys():
        raise ValueError("Parent and proposal state dictionaries must have identical keys.")
    result = {}
    for key in parent:
        parent_value = parent[key]
        proposal_value = proposal[key]
        if parent_value.shape != proposal_value.shape:
            raise ValueError(f"State shape mismatch for {key}.")
        if parent_value.is_floating_point() or parent_value.is_complex():
            result[key] = torch.lerp(parent_value, proposal_value, alpha)
        else:
            result[key] = parent_value.clone()
    return result


def project_awr_proposal(config: AwrTrustRegionConfig) -> dict[str, Any]:
    _validate(config)
    parent_model, parent_checkpoint = _load_attention_model(
        config.parent_checkpoint, config.device
    )
    proposal_model, proposal_checkpoint = _load_attention_model(
        config.proposal_checkpoint, config.device
    )
    candidate_model, _ = _load_attention_model(config.parent_checkpoint, config.device)
    parent_model.eval()
    proposal_model.eval()
    candidate_model.eval()
    cached_batches = _cache_validation_batches(config, parent_model)
    attach_parent_scores(cached_batches, parent_model, config.device)
    parent_state = parent_model.state_dict()
    proposal_state = proposal_model.state_dict()

    parent_metrics = _evaluate_model(parent_model, cached_batches, config)
    proposal_metrics = _evaluate_model(proposal_model, cached_batches, config)
    trace = [
        {"alpha": 0.0, "raw_kl": parent_metrics["raw_kl"]},
        {"alpha": 1.0, "raw_kl": proposal_metrics["raw_kl"]},
    ]
    if float(proposal_metrics["raw_kl"]) <= config.target_raw_kl:
        selected_alpha = 1.0
    else:
        low = 0.0
        high = 1.0
        for _ in range(config.bisection_steps):
            midpoint = (low + high) / 2.0
            candidate_model.load_state_dict(
                interpolate_state_dict(parent_state, proposal_state, midpoint)
            )
            metrics = _evaluate_model(candidate_model, cached_batches, config, loss=False)
            trace.append({"alpha": midpoint, "raw_kl": metrics["raw_kl"]})
            if float(metrics["raw_kl"]) <= config.target_raw_kl:
                low = midpoint
            else:
                high = midpoint
        selected_alpha = low

    projected_state = interpolate_state_dict(
        parent_state, proposal_state, selected_alpha
    )
    candidate_model.load_state_dict(projected_state)
    projected_metrics = _evaluate_model(candidate_model, cached_batches, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "legal_action_attention_scorer_projected.pt"
    metrics_path = config.output_dir / "trust_region_projection.json"
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": projected_state,
            "model_config": parent_checkpoint["model_config"],
            "train_config": {
                "projection": _config_payload(config),
                "selected_alpha": selected_alpha,
                "parent_checkpoint": str(config.parent_checkpoint),
                "proposal_checkpoint": str(config.proposal_checkpoint),
                "proposal_train_config": proposal_checkpoint.get("train_config", {}),
            },
        },
        checkpoint_path,
    )
    payload = {
        "format": "awr_parameter_trust_region_v1",
        "config": _config_payload(config),
        "validation_rows": sum(int(batch["target_action_row"].numel()) for batch in cached_batches),
        "selected_alpha": selected_alpha,
        "parent": parent_metrics,
        "proposal": proposal_metrics,
        "projected": projected_metrics,
        "trace": trace,
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _cache_validation_batches(config, parent_model):
    dataset = LegalActionReplayDataset(
        config.replay_path,
        include_stalled_games=False,
        include_timed_out_games=False,
        observation_size=parent_model.config.observation_size,
        action_feature_size=parent_model.config.action_feature_size,
    )
    indices = [
        index
        for index in range(len(dataset))
        if dataset.replay_step_metadata_for_index(index).game_seed
        >= config.validation_seed_min
    ]
    if not indices:
        raise ValueError("Trust-region projection requires validation rows.")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_legal_action_replay_samples,
    )
    return [{key: value for key, value in batch.items()} for batch in loader]


def _evaluate_model(model, cached_batches, config, *, loss=True):
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    with torch.inference_mode():
        for batch in cached_batches:
            observation = batch["observation"].to(config.device)
            action_features = batch["legal_action_features"].to(config.device)
            legal_mask = batch["legal_action_mask"].to(config.device)
            scores, _ = model(observation, action_features, legal_mask)
            reference_scores = batch.get("_reference_scores")
            if reference_scores is None:
                raise ValueError("Cached validation batch lacks parent scores.")
            reference_scores = reference_scores.to(config.device)
            reference_logits = masked_legal_action_scores(reference_scores, legal_mask)
            current_logits = masked_legal_action_scores(scores, legal_mask)
            reference_probs = torch.softmax(reference_logits, dim=1)
            per_row_kl = (
                reference_probs
                * (
                    torch.log_softmax(reference_logits, dim=1)
                    - torch.log_softmax(current_logits, dim=1)
                )
            ).sum(dim=1)
            batch_size = int(per_row_kl.numel())
            samples += batch_size
            totals["raw_kl"] = totals.get("raw_kl", 0.0) + float(per_row_kl.sum().cpu())
            if loss:
                metrics = compute_awr_loss(
                    action_scores=scores,
                    reference_scores=reference_scores,
                    legal_action_mask=legal_mask,
                    target_action_row=batch["target_action_row"].to(config.device),
                    advantages=batch["trajectory_advantage"].to(config.device),
                    sample_weight=batch["sample_weight"].to(config.device),
                    behavior_log_probability=batch["behavior_log_probability"].to(config.device),
                    advantage_temperature=config.advantage_temperature,
                    max_advantage_weight=config.max_advantage_weight,
                    kl_weight=config.kl_weight,
                    entropy_weight=config.entropy_weight,
                    sampling_epsilon=config.sampling_epsilon,
                    sampling_temperature=config.sampling_temperature,
                    sampling_top_k=config.sampling_top_k,
                )
                for key in ("policy_loss", "total_loss", "entropy"):
                    totals[key] = totals.get(key, 0.0) + float(metrics[key].cpu()) * batch_size
    return {key: value / samples for key, value in totals.items()} | {"samples": samples}


def attach_parent_scores(cached_batches, parent_model, device):
    parent_model.eval()
    with torch.inference_mode():
        for batch in cached_batches:
            scores, _ = parent_model(
                batch["observation"].to(device),
                batch["legal_action_features"].to(device),
                batch["legal_action_mask"].to(device),
            )
            batch["_reference_scores"] = scores.cpu()


def _config_payload(config):
    payload = asdict(config)
    for key in ("replay_path", "parent_checkpoint", "proposal_checkpoint", "output_dir"):
        payload[key] = str(payload[key])
    return payload


def _validate(config):
    for path in (config.replay_path, config.parent_checkpoint, config.proposal_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.0 < config.target_raw_kl < 1.0:
        raise ValueError("target_raw_kl must be in (0, 1).")
    if config.bisection_steps <= 0 or config.batch_size <= 0:
        raise ValueError("Projection steps and batch size must be positive.")
