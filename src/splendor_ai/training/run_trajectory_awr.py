"""Conservative advantage-weighted regression over grouped full trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import Subset

from splendor_ai.training.dataset import LegalActionReplayDataset
from splendor_ai.training.legal_action_supervised import create_legal_action_replay_dataloader
from splendor_ai.training.model import masked_legal_action_scores
from splendor_ai.training.run_trajectory_policy_gradient import (
    _exploration_probabilities,
    _load_attention_model,
)


@dataclass(frozen=True, slots=True)
class TrajectoryAwrConfig:
    replay_path: Path
    init_checkpoint: Path
    output_dir: Path
    train_seed_max: int
    validation_seed_min: int
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    advantage_temperature: float = 0.25
    max_advantage_weight: float = 10.0
    kl_weight: float = 1.0
    entropy_weight: float = 0.001
    sampling_epsilon: float = 0.5
    sampling_temperature: float = 1.0
    sampling_top_k: int = 3
    max_grad_norm: float = 1.0
    seed: int = 32
    device: str = "cpu"


def compute_awr_loss(
    *,
    action_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    target_action_row: torch.Tensor,
    advantages: torch.Tensor,
    sample_weight: torch.Tensor,
    behavior_log_probability: torch.Tensor,
    advantage_temperature: float,
    max_advantage_weight: float,
    kl_weight: float,
    entropy_weight: float,
    sampling_epsilon: float,
    sampling_temperature: float,
    sampling_top_k: int,
) -> dict[str, torch.Tensor]:
    masked_current = masked_legal_action_scores(action_scores, legal_action_mask)
    masked_reference = masked_legal_action_scores(reference_scores, legal_action_mask)
    current_log_probs = torch.log_softmax(masked_current, dim=1)
    reference_log_probs = torch.log_softmax(masked_reference, dim=1)
    current_probs = current_log_probs.exp()
    reference_probs = reference_log_probs.exp()
    rows = torch.arange(action_scores.shape[0], device=action_scores.device)

    advantage_weights = torch.exp(
        (advantages / advantage_temperature).clamp(max=math.log(max_advantage_weight))
    ).clamp(max=max_advantage_weight)
    combined_weights = advantage_weights * sample_weight
    normalized_weights = combined_weights / combined_weights.mean().clamp_min(1e-12)
    chosen_log_probs = current_log_probs[rows, target_action_row]
    policy_loss = -(normalized_weights * chosen_log_probs).mean()

    per_row_kl = (
        reference_probs * (reference_log_probs - current_log_probs)
    ).sum(dim=1)
    kl_loss = per_row_kl.mean()
    per_row_entropy = -(current_probs * current_log_probs).sum(dim=1)
    entropy = per_row_entropy.mean()
    total_loss = policy_loss + kl_weight * kl_loss - entropy_weight * entropy

    current_behavior, reference_behavior = _exploration_probabilities(
        action_scores,
        reference_scores,
        legal_action_mask,
        sampling_epsilon,
        sampling_temperature,
        sampling_top_k,
    )
    current_chosen = current_behavior[rows, target_action_row].clamp_min(1e-12)
    reference_chosen = reference_behavior[rows, target_action_row].clamp_min(1e-12)
    ratios = current_chosen / behavior_log_probability.exp().clamp_min(1e-12)
    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "entropy": entropy,
        "advantage_weight_mean": advantage_weights.mean(),
        "advantage_weight_max": advantage_weights.max(),
        "behavior_log_prob_mae": (
            reference_chosen.log() - behavior_log_probability
        ).abs().mean(),
        "approx_ratio": ratios.mean(),
        "clip_fraction": (ratios.sub(1.0).abs() > 0.1).float().mean(),
    }


def train_trajectory_awr(config: TrajectoryAwrConfig) -> dict[str, Any]:
    _validate(config)
    torch.manual_seed(config.seed)
    model, checkpoint = _load_attention_model(config.init_checkpoint, config.device)
    reference, _ = _load_attention_model(config.init_checkpoint, config.device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    dataset = LegalActionReplayDataset(
        config.replay_path,
        include_stalled_games=False,
        include_timed_out_games=False,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    train_indices = []
    validation_indices = []
    for index in range(len(dataset)):
        metadata = dataset.replay_step_metadata_for_index(index)
        if metadata.game_seed <= config.train_seed_max:
            train_indices.append(index)
        elif metadata.game_seed >= config.validation_seed_min:
            validation_indices.append(index)
    if not train_indices or not validation_indices:
        raise ValueError("AWR requires non-empty game-disjoint train and validation rows.")
    train_loader = create_legal_action_replay_dataloader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = create_legal_action_replay_dataloader(
        Subset(dataset, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.eval()
    with torch.no_grad():
        baseline_validation = _run_loader(
            model,
            reference,
            validation_loader,
            config,
            optimizer=None,
        )
    epoch_summaries = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_metrics = _run_loader(model, reference, train_loader, config, optimizer)
        model.eval()
        with torch.no_grad():
            validation_metrics = _run_loader(
                model,
                reference,
                validation_loader,
                config,
                optimizer=None,
            )
        epoch_summaries.append(
            {
                "epoch": epoch,
                **train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"[trajectory-awr] epoch={epoch}/{config.epochs}"
            f" samples={train_metrics['samples']}"
            f" policy={train_metrics['policy_loss']:.6f}"
            f" kl={train_metrics['kl_loss']:.6f}"
            f" validation={validation_metrics['total_loss']:.6f}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "legal_action_attention_scorer.pt"
    metrics_path = config.output_dir / "trajectory_awr_metrics.json"
    config_payload = _config_payload(config)
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": checkpoint["model_config"],
            "train_config": config_payload,
        },
        checkpoint_path,
    )
    payload = {
        "format": "trajectory_awr_v1",
        "config": config_payload,
        "dataset_samples": len(dataset),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "baseline_validation": baseline_validation,
        "checkpoint_path": str(checkpoint_path),
        "epochs": epoch_summaries,
        "training_seconds": time.perf_counter() - started,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_loader(model, reference, loader, config, optimizer=None) -> dict[str, float | int]:
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        observation = batch["observation"].to(config.device)
        action_features = batch["legal_action_features"].to(config.device)
        legal_mask = batch["legal_action_mask"].to(config.device)
        target_rows = batch["target_action_row"].to(config.device)
        advantages = batch["trajectory_advantage"].to(config.device)
        behavior_log_probs = batch["behavior_log_probability"].to(config.device)
        sample_weights = batch["sample_weight"].to(config.device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        scores, _ = model(observation, action_features, legal_mask)
        with torch.no_grad():
            reference_scores, _ = reference(observation, action_features, legal_mask)
        metrics = compute_awr_loss(
            action_scores=scores,
            reference_scores=reference_scores,
            legal_action_mask=legal_mask,
            target_action_row=target_rows,
            advantages=advantages,
            sample_weight=sample_weights,
            behavior_log_probability=behavior_log_probs,
            advantage_temperature=config.advantage_temperature,
            max_advantage_weight=config.max_advantage_weight,
            kl_weight=config.kl_weight,
            entropy_weight=config.entropy_weight,
            sampling_epsilon=config.sampling_epsilon,
            sampling_temperature=config.sampling_temperature,
            sampling_top_k=config.sampling_top_k,
        )
        if optimizer is not None:
            metrics["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        batch_size = int(target_rows.numel())
        samples += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    return {key: value / samples for key, value in totals.items()} | {"samples": samples}


def _config_payload(config: TrajectoryAwrConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("replay_path", "init_checkpoint", "output_dir"):
        payload[key] = str(payload[key])
    return payload


def _validate(config: TrajectoryAwrConfig) -> None:
    if not config.replay_path.exists() or not config.init_checkpoint.exists():
        raise FileNotFoundError("AWR replay and parent checkpoint must exist.")
    if config.train_seed_max >= config.validation_seed_min:
        raise ValueError("Training and validation seed ranges must be disjoint.")
    if min(config.epochs, config.batch_size) <= 0 or config.learning_rate <= 0.0:
        raise ValueError("AWR optimizer settings must be positive.")
    if config.advantage_temperature <= 0.0 or config.max_advantage_weight < 1.0:
        raise ValueError("AWR weighting settings are invalid.")
    if config.kl_weight < 0.0 or config.entropy_weight < 0.0:
        raise ValueError("AWR regularization weights must be non-negative.")
