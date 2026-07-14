"""Train an action-conditioned Q scorer from counterfactual rollout returns."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from splendor_ai.bots.puct_bot import legal_heuristic_value
from splendor_ai.bots.search_bot import CheckpointLegalActionPriorScorer
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv

from .dataset import LegalActionReplayDataset, collate_legal_action_replay_samples
from .legal_action_supervised import split_legal_action_replay_dataset
from .model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel


@dataclass(frozen=True, slots=True)
class ActionValueTrainingConfig:
    replay_path: Path
    init_checkpoint: Path
    output_dir: Path
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    pairwise_loss_weight: float = 0.5
    minimum_target_gap: float = 0.05
    validation_fraction: float = 0.2
    validation_seed: int = 65
    max_grad_norm: float = 1.0
    device: str = "cpu"
    log_every_batches: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an action-conditioned Q scorer from counterfactual action returns."
    )
    parser.add_argument("--replay-path", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.5)
    parser.add_argument("--minimum-target-gap", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=65)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-batches", type=int, default=10)
    return parser


def compute_action_value_losses(
    action_values: torch.Tensor,
    target_values: torch.Tensor,
    target_mask: torch.Tensor,
    pairwise_loss_weight: float,
    minimum_target_gap: float,
) -> dict[str, torch.Tensor]:
    if action_values.shape != target_values.shape or action_values.shape != target_mask.shape:
        raise ValueError("Action values, targets, and masks must have matching shape.")
    if minimum_target_gap < 0.0 or pairwise_loss_weight < 0.0:
        raise ValueError("Pairwise settings must be non-negative.")
    valid_count = target_mask.sum()
    if int(valid_count.item()) == 0:
        raise ValueError("Action-value batch has no counterfactual targets.")
    q_loss = F.smooth_l1_loss(action_values[target_mask], target_values[target_mask])
    value_differences = target_values.unsqueeze(2) - target_values.unsqueeze(1)
    prediction_differences = action_values.unsqueeze(2) - action_values.unsqueeze(1)
    pair_mask = target_mask.unsqueeze(2) & target_mask.unsqueeze(1)
    pair_mask &= value_differences.abs() >= minimum_target_gap
    pair_mask &= value_differences > 0.0
    if bool(pair_mask.any().item()):
        pairwise_loss = F.softplus(-prediction_differences[pair_mask]).mean()
    else:
        pairwise_loss = action_values.sum() * 0.0
    return {
        "q_loss": q_loss,
        "pairwise_loss": pairwise_loss,
        "total_loss": q_loss + pairwise_loss_weight * pairwise_loss,
    }


def _load_q_model(path: Path, device: torch.device) -> tuple[LegalActionAttentionScorerModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError("Action-value training requires a legal-action attention checkpoint.")
    config_keys = {field.name for field in fields(LegalActionAttentionScorerConfig)}
    config = LegalActionAttentionScorerConfig(
        **{key: value for key, value in checkpoint.get("model_config", {}).items() if key in config_keys}
    )
    model = LegalActionAttentionScorerModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    for module in model.score_head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return model, checkpoint


def _loader(dataset: Subset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_legal_action_replay_samples)


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int = 5_000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = statistics.fmean(values)
    if len(values) == 1:
        return [mean, mean]
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def _average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted((value, index) for index, value in enumerate(values))
    start = 0
    while start < len(ordered):
        value = ordered[start][0]
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][0] == value:
            end += 1
        for position in range(start, end + 1):
            ranks[ordered[position][1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _evaluate_ranking(
    model: LegalActionAttentionScorerModel,
    dataset: LegalActionReplayDataset,
    indices: list[int],
    parent_checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    env = SplendorEnv()
    codec = ActionCodec()
    parent = CheckpointLegalActionPriorScorer(parent_checkpoint, device=str(device))
    q_hits: list[float] = []
    heuristic_hits: list[float] = []
    q_regrets: list[float] = []
    heuristic_regrets: list[float] = []
    q_spearmans: list[float] = []
    heuristic_spearmans: list[float] = []
    q_parent_disagreements: list[float] = []
    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            valid_mask = sample.search_value_mask
            if int(valid_mask.sum().item()) < 2:
                continue
            observation = sample.observation.unsqueeze(0).to(device)
            action_features = sample.legal_action_features.unsqueeze(0).to(device)
            legal_mask = torch.ones((1, sample.legal_action_features.shape[0]), dtype=torch.bool, device=device)
            predictions, _ = model(observation, action_features, legal_mask)
            predicted_values = predictions[0].detach().cpu()
            target_values = sample.search_value_target
            candidate_rows = valid_mask.nonzero(as_tuple=False).squeeze(1)
            target_candidate_values = target_values[candidate_rows]
            best_target = float(target_candidate_values.max().item())
            q_row = int(candidate_rows[torch.argmax(predicted_values[candidate_rows])].item())

            state = dataset.state_for_index(index)
            legal_actions = env.legal_actions(state)
            legal_indices = tuple(codec.legal_action_indices(state, legal_actions))
            heuristic_values = []
            for row in candidate_rows.tolist():
                child = env.step(state, legal_actions[row])
                heuristic_values.append(legal_heuristic_value(child, state.current_player, scale=4_000.0))
            heuristic_row = int(candidate_rows[max(range(len(heuristic_values)), key=lambda row: heuristic_values[row])].item())
            parent_scores = parent.score_actions(state, legal_actions)
            parent_row = max(
                range(len(legal_actions)),
                key=lambda row: parent_scores[legal_actions[row]],
            )
            q_hits.append(float(target_values[q_row].item() == best_target))
            heuristic_hits.append(float(target_values[heuristic_row].item() == best_target))
            q_regrets.append(best_target - float(target_values[q_row].item()))
            heuristic_regrets.append(best_target - float(target_values[heuristic_row].item()))
            q_spearmans.append(
                _spearman(
                    [float(predicted_values[row].item()) for row in candidate_rows.tolist()],
                    [float(target_values[row].item()) for row in candidate_rows.tolist()],
                )
            )
            heuristic_spearmans.append(_spearman(heuristic_values, [float(value.item()) for value in target_candidate_values]))
            q_parent_disagreements.append(float(q_row != parent_row))
    return {
        "states": len(q_hits),
        "q_top1_rate": statistics.fmean(q_hits) if q_hits else 0.0,
        "heuristic_top1_rate": statistics.fmean(heuristic_hits) if heuristic_hits else 0.0,
        "q_minus_heuristic_top1": statistics.fmean(q - h for q, h in zip(q_hits, heuristic_hits, strict=True)) if q_hits else 0.0,
        "q_minus_heuristic_top1_ci95": _bootstrap_mean_ci([q - h for q, h in zip(q_hits, heuristic_hits, strict=True)], 91),
        "q_mean_regret": statistics.fmean(q_regrets) if q_regrets else 0.0,
        "heuristic_mean_regret": statistics.fmean(heuristic_regrets) if heuristic_regrets else 0.0,
        "q_minus_heuristic_regret": statistics.fmean(q - h for q, h in zip(q_regrets, heuristic_regrets, strict=True)) if q_regrets else 0.0,
        "q_minus_heuristic_regret_ci95": _bootstrap_mean_ci([q - h for q, h in zip(q_regrets, heuristic_regrets, strict=True)], 92),
        "q_mean_spearman": statistics.fmean(q_spearmans) if q_spearmans else 0.0,
        "heuristic_mean_spearman": statistics.fmean(heuristic_spearmans) if heuristic_spearmans else 0.0,
        "q_minus_heuristic_spearman": statistics.fmean(q - h for q, h in zip(q_spearmans, heuristic_spearmans, strict=True)) if q_spearmans else 0.0,
        "q_minus_heuristic_spearman_ci95": _bootstrap_mean_ci([q - h for q, h in zip(q_spearmans, heuristic_spearmans, strict=True)], 94),
        "q_parent_disagreement_rate": statistics.fmean(q_parent_disagreements) if q_parent_disagreements else 0.0,
        "q_parent_disagreement_ci95": _bootstrap_mean_ci(q_parent_disagreements, 93),
    }


def _offline_gate(validation: dict[str, Any]) -> dict[str, Any]:
    regret_upper = float(validation["q_minus_heuristic_regret_ci95"][1])
    spearman_lower = float(validation["q_minus_heuristic_spearman_ci95"][0])
    return {
        "passed": regret_upper < 0.0 and spearman_lower > 0.0,
        "criteria": {
            "q_regret_ci95_upper_lt_zero": regret_upper < 0.0,
            "q_spearman_ci95_lower_gt_zero": spearman_lower > 0.0,
        },
        "reason": (
            "Q improves held-out regret and ranking correlation over the heuristic."
            if regret_upper < 0.0 and spearman_lower > 0.0
            else "Do not integrate into PUCT; Q has not demonstrated superior held-out action ranking."
        ),
    }


def train_action_value_model(config: ActionValueTrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    device = torch.device(config.device)
    torch.manual_seed(config.validation_seed)
    model, parent_checkpoint = _load_q_model(config.init_checkpoint, device)
    dataset = LegalActionReplayDataset(
        config.replay_path,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    train_split, validation_split = split_legal_action_replay_dataset(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=config.validation_seed,
        validation_split_unit="game",
    )
    if not isinstance(train_split, Subset) or not isinstance(validation_split, Subset):
        raise ValueError("Action-value training requires a non-empty game-level validation split.")
    train_loader = _loader(train_split, config.batch_size, True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_epoch = None
    best_regret = math.inf
    epochs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"q_loss": 0.0, "pairwise_loss": 0.0, "total_loss": 0.0}
        rows = 0
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            observation = batch["observation"].to(device)
            action_features = batch["legal_action_features"].to(device)
            legal_mask = batch["legal_action_mask"].to(device)
            target_values = batch["search_value_target"].to(device)
            target_mask = batch["search_value_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(observation, action_features, legal_mask)
            losses = compute_action_value_losses(
                predictions,
                target_values,
                target_mask,
                config.pairwise_loss_weight,
                config.minimum_target_gap,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            batch_rows = int(observation.shape[0])
            rows += batch_rows
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_rows
            if batch_index == 1 or batch_index == len(train_loader) or (config.log_every_batches > 0 and batch_index % config.log_every_batches == 0):
                print(
                    f"[action-value-train] epoch={epoch}/{config.epochs} batch={batch_index}/{len(train_loader)}"
                    f" q={totals['q_loss'] / rows:.5f} rank={totals['pairwise_loss'] / rows:.5f}"
                    f" elapsed={time.perf_counter() - epoch_started:.1f}s"
                )
        validation = _evaluate_ranking(model, dataset, list(validation_split.indices), config.init_checkpoint, device)
        train_metrics = {key: value / rows for key, value in totals.items()} | {"rows": rows}
        epochs.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        if validation["q_mean_regret"] < best_regret:
            best_regret = validation["q_mean_regret"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_epoch is None:
        raise RuntimeError("Action-value training did not produce a validation checkpoint.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.output_dir / "action_value_attention_scorer.pt"
    best_path = config.output_dir / "action_value_attention_scorer_best.pt"
    metrics_path = config.output_dir / "action_value_metrics.json"
    config_payload = asdict(config) | {
        "replay_path": str(config.replay_path),
        "init_checkpoint": str(config.init_checkpoint),
        "output_dir": str(config.output_dir),
    }
    checkpoint_payload = {
        "model_kind": "action_value_attention_scorer",
        "model_config": parent_checkpoint["model_config"],
        "train_config": config_payload,
    }
    torch.save(checkpoint_payload | {"model_state_dict": model.state_dict()}, final_path)
    torch.save(checkpoint_payload | {"model_state_dict": best_state}, best_path)
    best_validation = next(item["validation"] for item in epochs if item["epoch"] == best_epoch)
    summary = {
        "config": config_payload,
        "dataset_rows": len(dataset),
        "train_rows": len(train_split),
        "validation_rows": len(validation_split),
        "validation_games": len(
            {
                dataset.replay_step_metadata_for_index(index).game_seed
                for index in validation_split.indices
            }
        ),
        "best_validation_epoch": best_epoch,
        "best_validation": best_validation,
        "offline_gate": _offline_gate(best_validation),
        "epochs": epochs,
        "training_seconds": time.perf_counter() - started,
        "final_checkpoint": str(final_path),
        "best_checkpoint": str(best_path),
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _validate_config(config: ActionValueTrainingConfig) -> None:
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    if config.pairwise_loss_weight < 0.0 or config.minimum_target_gap < 0.0:
        raise ValueError("Pairwise settings must be non-negative.")
    if not 0.0 < config.validation_fraction < 1.0 or config.max_grad_norm <= 0.0:
        raise ValueError("Validation fraction and gradient norm are invalid.")


def main() -> None:
    args = build_parser().parse_args()
    summary = train_action_value_model(
        ActionValueTrainingConfig(
            replay_path=Path(args.replay_path),
            init_checkpoint=Path(args.init_checkpoint),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            pairwise_loss_weight=args.pairwise_loss_weight,
            minimum_target_gap=args.minimum_target_gap,
            validation_fraction=args.validation_fraction,
            validation_seed=args.validation_seed,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            log_every_batches=args.log_every_batches,
        )
    )
    best = summary["epochs"][summary["best_validation_epoch"] - 1]["validation"]
    print(
        "[action-value-train]"
        f" best_epoch={summary['best_validation_epoch']}"
        f" q_regret={best['q_mean_regret']:.4f}"
        f" heuristic_regret={best['heuristic_mean_regret']:.4f}"
        f" disagreement={best['q_parent_disagreement_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
