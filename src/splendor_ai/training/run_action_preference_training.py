"""Train and gate a 035-initialized direct pairwise action-preference update."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import statistics
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.run_action_value_training import _bootstrap_mean_ci
from splendor_ai.training.run_awr_trust_region import interpolate_state_dict
from splendor_ai.training.run_trajectory_policy_gradient import _load_attention_model


@dataclass(frozen=True, slots=True)
class ActionPreferenceTrainingConfig:
    parent_checkpoint: Path
    train_replay: Path
    validation_replay: Path
    confirmation_replay: Path
    output_dir: Path
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    kl_weight: float = 1.0
    max_grad_norm: float = 1.0
    target_validation_kl: float = 0.004
    bisection_steps: int = 12
    minimum_relative_loss_improvement: float = 0.03
    minimum_challenger_accuracy: float = 0.20
    minimum_parent_accuracy: float = 0.90
    minimum_confirmation_accuracy_gain: float = 0.05
    bootstrap_samples: int = 20_000
    seed: int = 35
    device: str = "cpu"


def train_and_gate_action_preferences(config: ActionPreferenceTrainingConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.output_dir / "offline_preference_gate.json"
    candidate_path = config.output_dir / "legal_action_attention_scorer_preference.pt"
    if result_path.exists() and candidate_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    torch.manual_seed(config.seed)
    parent, parent_checkpoint = _load_attention_model(config.parent_checkpoint, config.device)
    model, _ = _load_attention_model(config.parent_checkpoint, config.device)
    parent.eval()
    train_loader = _loader(config.train_replay, parent, config, shuffle=True)
    validation_loader = _loader(config.validation_replay, parent, config, shuffle=False)
    confirmation_loader = _loader(config.confirmation_replay, parent, config, shuffle=False)
    parent_validation = evaluate_preference_model(parent, parent, validation_loader, config.device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    epochs: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        training = _train_epoch(model, parent, train_loader, optimizer, config)
        validation = evaluate_preference_model(model, parent, validation_loader, config.device)
        row = {"epoch": epoch, "training": training, "validation": validation}
        epochs.append(row)
        print(
            f"[preference-train] epoch={epoch}/{config.epochs} "
            f"train_loss={training['pairwise_loss']:.6f} "
            f"val_loss={validation['pairwise_loss']:.6f} "
            f"val_kl={validation['raw_kl']:.6f}",
            flush=True,
        )
        if float(validation["pairwise_loss"]) < best_loss:
            best_loss = float(validation["pairwise_loss"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None

    proposal_path = config.output_dir / "legal_action_attention_scorer_preference_proposal.pt"
    _save_checkpoint(
        proposal_path, best_state, parent_checkpoint, config, {"best_epoch": best_epoch, "epochs": epochs}
    )
    parent_state = {key: value.detach().cpu() for key, value in parent.state_dict().items()}
    selected_alpha, trace = _select_projection_alpha(
        model, parent, parent_state, best_state, validation_loader, config
    )
    projected_state = interpolate_state_dict(parent_state, best_state, selected_alpha)
    model.load_state_dict(projected_state)
    model.to(config.device)
    validation = evaluate_preference_model(model, parent, validation_loader, config.device)
    _save_checkpoint(
        candidate_path,
        projected_state,
        parent_checkpoint,
        config,
        {"best_epoch": best_epoch, "selected_alpha": selected_alpha, "epochs": epochs},
    )

    # Confirmation is first touched here, after epoch and interpolation selection are frozen.
    flags = _load_challenger_flags(config.confirmation_replay)
    parent_confirmation = evaluate_preference_model(
        parent, parent, confirmation_loader, config.device, flags
    )
    confirmation = evaluate_preference_model(
        model, parent, confirmation_loader, config.device, flags
    )
    deltas = [
        candidate - baseline
        for candidate, baseline in zip(
            confirmation["row_pairwise_losses"],
            parent_confirmation["row_pairwise_losses"],
            strict=True,
        )
    ]
    delta_ci = _bootstrap_mean_ci(deltas, config.seed + 1, config.bootstrap_samples)
    gates = {
        "nonzero_projection": selected_alpha > 0.0,
        "validation_kl_ceiling": float(validation["raw_kl"])
        <= config.target_validation_kl + 1e-6,
        "validation_loss_improves": _relative_improvement(
            parent_validation["pairwise_loss"], validation["pairwise_loss"]
        )
        >= config.minimum_relative_loss_improvement,
        "confirmation_loss_improves": _relative_improvement(
            parent_confirmation["pairwise_loss"], confirmation["pairwise_loss"]
        )
        >= config.minimum_relative_loss_improvement,
        "confirmation_loss_ci_below_zero": float(delta_ci[1]) < 0.0,
        "confirmation_accuracy_gain": float(confirmation["accuracy"])
        - float(parent_confirmation["accuracy"])
        >= config.minimum_confirmation_accuracy_gain,
        "learns_challenger_wins": float(confirmation["challenger_best_accuracy"])
        >= config.minimum_challenger_accuracy,
        "retains_parent_wins": float(confirmation["parent_best_accuracy"])
        >= config.minimum_parent_accuracy,
    }
    payload = {
        "format": "action_preference_offline_gate_v1",
        "candidate_checkpoint": str(candidate_path),
        "proposal_checkpoint": str(proposal_path),
        "best_epoch": best_epoch,
        "selected_alpha": selected_alpha,
        "projection_trace": trace,
        "parent_validation": _without_rows(parent_validation),
        "validation": _without_rows(validation),
        "parent_confirmation": _without_rows(parent_confirmation),
        "confirmation": _without_rows(confirmation),
        "confirmation_loss_delta_mean": statistics.fmean(deltas) if deltas else 0.0,
        "confirmation_loss_delta_ci95": delta_ci,
        "epochs": epochs,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_write(result_path, payload)
    return payload


def evaluate_preference_model(
    model: torch.nn.Module,
    parent: torch.nn.Module,
    loader: DataLoader,
    device: str,
    challenger_flags: list[bool] | None = None,
) -> dict[str, Any]:
    model.eval()
    parent.eval()
    losses: list[float] = []
    kls: list[float] = []
    correct: list[bool] = []
    margins: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            observation = batch["observation"].to(device)
            features = batch["legal_action_features"].to(device)
            mask = batch["legal_action_mask"].to(device)
            preferred = batch["target_action_row"].to(device)
            rejected = batch["preference_competitor_row"].to(device)
            scores, _ = model(observation, features, mask)
            parent_scores, _ = parent(observation, features, mask)
            rows = torch.arange(scores.shape[0], device=device)
            margin = scores[rows, preferred] - scores[rows, rejected]
            losses.extend(torch.nn.functional.softplus(-margin).cpu().tolist())
            margins.extend(margin.cpu().tolist())
            correct.extend((margin > 0.0).cpu().tolist())
            parent_logits = parent_scores.masked_fill(~mask, -torch.inf)
            logits = scores.masked_fill(~mask, -torch.inf)
            parent_probs = torch.softmax(parent_logits, dim=1)
            parent_log = torch.log_softmax(parent_logits, dim=1).masked_fill(~mask, 0.0)
            current_log = torch.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
            kls.extend((parent_probs * (parent_log - current_log)).sum(dim=1).cpu().tolist())
    payload: dict[str, Any] = {
        "rows": len(losses),
        "pairwise_loss": statistics.fmean(losses),
        "accuracy": sum(correct) / len(correct),
        "mean_margin": statistics.fmean(margins),
        "raw_kl": statistics.fmean(kls),
        "row_pairwise_losses": losses,
    }
    if challenger_flags is not None:
        if len(challenger_flags) != len(correct):
            raise ValueError("Confirmation metadata does not match dataset rows.")
        challenger_rows = [ok for ok, flag in zip(correct, challenger_flags, strict=True) if flag]
        parent_rows = [ok for ok, flag in zip(correct, challenger_flags, strict=True) if not flag]
        payload["challenger_best_accuracy"] = sum(challenger_rows) / len(challenger_rows)
        payload["parent_best_accuracy"] = sum(parent_rows) / len(parent_rows)
    return payload


def _train_epoch(model, parent, loader, optimizer, config) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    pairwise_losses: list[float] = []
    kls: list[float] = []
    grad_norms: list[float] = []
    for batch in loader:
        observation = batch["observation"].to(config.device)
        features = batch["legal_action_features"].to(config.device)
        mask = batch["legal_action_mask"].to(config.device)
        preferred = batch["target_action_row"].to(config.device)
        rejected = batch["preference_competitor_row"].to(config.device)
        with torch.no_grad():
            parent_scores, _ = parent(observation, features, mask)
            parent_logits = parent_scores.masked_fill(~mask, -torch.inf)
            parent_probs = torch.softmax(parent_logits, dim=1)
            parent_log = torch.log_softmax(parent_logits, dim=1).masked_fill(~mask, 0.0)
        scores, _ = model(observation, features, mask)
        rows = torch.arange(scores.shape[0], device=config.device)
        pairwise = torch.nn.functional.softplus(
            scores[rows, rejected] - scores[rows, preferred]
        ).mean()
        logits = scores.masked_fill(~mask, -torch.inf)
        current_log = torch.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
        kl = (parent_probs * (parent_log - current_log)).sum(dim=1).mean()
        loss = pairwise + config.kl_weight * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        pairwise_losses.append(float(pairwise.detach().cpu()))
        kls.append(float(kl.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu()))
    return {
        "loss": statistics.fmean(losses),
        "pairwise_loss": statistics.fmean(pairwise_losses),
        "raw_kl": statistics.fmean(kls),
        "grad_norm": statistics.fmean(grad_norms),
    }


def _select_projection_alpha(model, parent, parent_state, proposal_state, loader, config):
    trace: list[dict[str, float]] = []
    model.load_state_dict(proposal_state)
    proposal = evaluate_preference_model(model, parent, loader, config.device)
    trace.append({"alpha": 1.0, "raw_kl": float(proposal["raw_kl"])})
    if float(proposal["raw_kl"]) <= config.target_validation_kl:
        return 1.0, trace
    low, high = 0.0, 1.0
    for _ in range(config.bisection_steps):
        midpoint = (low + high) / 2.0
        model.load_state_dict(interpolate_state_dict(parent_state, proposal_state, midpoint))
        metrics = evaluate_preference_model(model, parent, loader, config.device)
        trace.append({"alpha": midpoint, "raw_kl": float(metrics["raw_kl"])})
        if float(metrics["raw_kl"]) <= config.target_validation_kl:
            low = midpoint
        else:
            high = midpoint
    return low, trace


def _loader(path, model, config, shuffle):
    dataset = LegalActionReplayDataset(
        path,
        preference_rows_policy_weight=0.0,
        preference_rows_distill_weight=0.0,
        preference_rows_preference_weight=1.0,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_legal_action_replay_samples,
        generator=torch.Generator().manual_seed(config.seed) if shuffle else None,
    )


def _load_challenger_flags(path: Path) -> list[bool]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            bool(json.loads(line)["preference_challenger_better"])
            for line in handle
            if line.strip()
        ]


def _save_checkpoint(path, state, parent_checkpoint, config, training):
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": state,
            "model_config": parent_checkpoint["model_config"],
            "train_config": {
                "method": "direct_pairwise_action_preference_v1",
                "parent_checkpoint": str(config.parent_checkpoint),
                "training": training,
            },
        },
        path,
    )


def _relative_improvement(parent: float, candidate: float) -> float:
    return (float(parent) - float(candidate)) / float(parent)


def _without_rows(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "row_pairwise_losses"}


def _validate(config):
    for path in (
        config.parent_checkpoint,
        config.train_replay,
        config.validation_replay,
        config.confirmation_replay,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0.0:
        raise ValueError("Epochs, batch size, and learning rate must be positive.")
    if not 0.0 < config.target_validation_kl < 0.01:
        raise ValueError("Validation KL target must be in (0, 0.01).")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
