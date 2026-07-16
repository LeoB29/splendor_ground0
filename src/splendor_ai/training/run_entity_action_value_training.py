"""Train and gate an auxiliary canonical-entity action-value scorer."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.encoding import (
    ActionCodec,
    actions_by_canonical_successor_key,
    collate_entity_action_batch,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.entity_actor_critic import load_entity_actor
from splendor_ai.training.model import CanonicalEntityActionScorerModel
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_action_value_training import compute_action_value_losses, _spearman


@dataclass(frozen=True, slots=True)
class EntityQConfig:
    corpus_path: Path
    parent_checkpoint: Path
    output_dir: Path
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    pairwise_loss_weight: float = 0.5
    minimum_target_gap: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 211_000
    bootstrap_samples: int = 5_000
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class EntityQSample:
    state: Any
    legal_actions: tuple[Any, ...]
    target_values: torch.Tensor
    target_mask: torch.Tensor
    prior_scores: torch.Tensor


class EntityQDataset(Dataset[EntityQSample]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.env = SplendorEnv()
        self.codec = ActionCodec()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EntityQSample:
        row = self.rows[index]
        state = deserialize_state_snapshot(row["state_snapshot"])
        legal_actions = tuple(
            actions_by_canonical_successor_key(
                state, self.env.legal_actions(state)
            ).values()
        )
        action_rows = {
            self.codec.encode(state, action): action_row
            for action_row, action in enumerate(legal_actions)
        }
        targets = torch.zeros(len(legal_actions), dtype=torch.float32)
        mask = torch.zeros(len(legal_actions), dtype=torch.bool)
        prior = torch.full((len(legal_actions),), -torch.inf, dtype=torch.float32)
        for action in row["actions"]:
            action_row = action_rows[int(action["action_index"])]
            targets[action_row] = float(action["full_mean"])
            prior[action_row] = float(action["prior_score"])
            mask[action_row] = True
        return EntityQSample(state, legal_actions, targets, mask, prior)


def collate_entity_q_samples(samples: list[EntityQSample]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate zero entity Q samples.")
    batch = collate_entity_action_batch(
        [(sample.state, list(sample.legal_actions)) for sample in samples]
    )
    width = batch["action_mask"].shape[1]
    targets = torch.zeros((len(samples), width), dtype=torch.float32)
    target_mask = torch.zeros((len(samples), width), dtype=torch.bool)
    prior_scores = torch.full((len(samples), width), -torch.inf, dtype=torch.float32)
    for row, sample in enumerate(samples):
        count = len(sample.legal_actions)
        targets[row, :count] = sample.target_values
        target_mask[row, :count] = sample.target_mask
        prior_scores[row, :count] = sample.prior_scores
    batch["target_values"] = targets
    batch["target_mask"] = target_mask
    batch["prior_scores"] = prior_scores
    return batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.5)
    parser.add_argument("--minimum-target-gap", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=211_000)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_entity_action_value(
        EntityQConfig(
            corpus_path=Path(args.corpus_path),
            parent_checkpoint=Path(args.parent_checkpoint),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            pairwise_loss_weight=args.pairwise_loss_weight,
            minimum_target_gap=args.minimum_target_gap,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            device=args.device,
        )
    )
    print(
        json.dumps(
            {"gates": result["gates"], "all_gates_passed": result["all_gates_passed"]},
            indent=2,
        )
    )


def train_entity_action_value(config: EntityQConfig) -> dict[str, Any]:
    _validate_config(config)
    corpus = json.loads(config.corpus_path.read_text(encoding="utf-8"))
    if not corpus.get("all_gates_passed", False):
        raise ValueError("Counterfactual corpus did not pass its preregistered quality gates.")
    rows_by_split = {
        split: [row for row in corpus["states"] if row["split"] == split]
        for split in ("train", "validation", "confirmation")
    }
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError(
            "Corpus must contain non-empty train, validation, and confirmation splits."
        )
    datasets = {split: EntityQDataset(rows) for split, rows in rows_by_split.items()}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            collate_fn=collate_entity_q_samples,
            generator=torch.Generator().manual_seed(config.seed),
        )
        for split, dataset in datasets.items()
    }
    torch.manual_seed(config.seed)
    model = load_entity_actor(config.parent_checkpoint, config.device)
    for module in model.score_head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    best_state = None
    best_epoch = None
    best_validation_regret = math.inf
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(model, loaders["train"], optimizer, config)
        validation = evaluate_entity_q(
            model,
            loaders["validation"],
            config.device,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + epoch,
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        if validation["q_mean_regret"] < best_validation_regret:
            best_validation_regret = validation["q_mean_regret"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"[entity-q] epoch={epoch}/{config.epochs} "
            f"loss={train_metrics['total_loss']:.5f} "
            f"val_q_regret={validation['q_mean_regret']:.5f} "
            f"val_parent_regret={validation['parent_mean_regret']:.5f}",
            flush=True,
        )
    if best_state is None or best_epoch is None:
        raise RuntimeError("Entity Q training did not produce a best epoch.")
    model.load_state_dict(best_state)
    validation = evaluate_entity_q(
        model,
        loaders["validation"],
        config.device,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed + 10_001,
    )
    confirmation = evaluate_entity_q(
        model,
        loaders["confirmation"],
        config.device,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed + 20_001,
    )
    validation_gate = _ranking_gate(validation)
    confirmation_gate = _ranking_gate(confirmation)
    gates = {
        "validation_q_beats_parent": validation_gate["passed"],
        "confirmation_q_beats_parent": confirmation_gate["passed"],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "canonical_entity_action_value_scorer.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_value_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "parent_checkpoint": str(config.parent_checkpoint),
            "training_only": True,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    result = {
        "training": "canonical_entity_action_value_q_v1",
        "config": _config_payload(config),
        "corpus_path": str(config.corpus_path),
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "best_epoch": best_epoch,
        "history": history,
        "validation": validation,
        "confirmation": confirmation,
        "validation_gate": validation_gate,
        "confirmation_gate": confirmation_gate,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "checkpoint_path": str(checkpoint_path),
        "decision": (
            "Auxiliary Q generalizes on validation and untouched confirmation; "
            "a matched anchored actor update is eligible."
            if all(gates.values())
            else "Do not update the actor: auxiliary Q did not beat the frozen parent "
            "on both held-out splits."
        ),
    }
    (config.output_dir / "entity_action_value_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _train_epoch(
    model: CanonicalEntityActionScorerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: EntityQConfig,
) -> dict[str, float]:
    model.train()
    totals = {"q_loss": 0.0, "pairwise_loss": 0.0, "total_loss": 0.0}
    states = 0
    for batch in loader:
        batch = {name: value.to(config.device) for name, value in batch.items()}
        predictions, _ = model(**_model_batch(batch))
        losses = compute_action_value_losses(
            predictions,
            batch["target_values"],
            batch["target_mask"],
            config.pairwise_loss_weight,
            config.minimum_target_gap,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        batch_states = int(batch["target_values"].shape[0])
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_states
        states += batch_states
    return {key: value / states for key, value in totals.items()}


def evaluate_entity_q(
    model: CanonicalEntityActionScorerModel,
    loader: DataLoader,
    device: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    q_regrets = []
    parent_regrets = []
    q_spearmans = []
    parent_spearmans = []
    q_hits = []
    parent_hits = []
    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            predictions, _ = model(**_model_batch(batch))
            for row in range(predictions.shape[0]):
                candidate_rows = batch["target_mask"][row].nonzero(as_tuple=False).squeeze(1)
                targets = batch["target_values"][row, candidate_rows].cpu().tolist()
                q_values = predictions[row, candidate_rows].cpu().tolist()
                parent_values = batch["prior_scores"][row, candidate_rows].cpu().tolist()
                best = max(targets)
                q_row = max(range(len(targets)), key=lambda index: (q_values[index], -index))
                parent_row = max(
                    range(len(targets)), key=lambda index: (parent_values[index], -index)
                )
                q_regrets.append(best - targets[q_row])
                parent_regrets.append(best - targets[parent_row])
                q_spearmans.append(_spearman(q_values, targets))
                parent_spearmans.append(_spearman(parent_values, targets))
                q_hits.append(float(targets[q_row] == best))
                parent_hits.append(float(targets[parent_row] == best))
    regret_deltas = [q - parent for q, parent in zip(q_regrets, parent_regrets, strict=True)]
    spearman_deltas = [
        q - parent for q, parent in zip(q_spearmans, parent_spearmans, strict=True)
    ]
    top1_deltas = [q - parent for q, parent in zip(q_hits, parent_hits, strict=True)]
    return {
        "states": len(q_regrets),
        "q_top1_rate": statistics.fmean(q_hits),
        "parent_top1_rate": statistics.fmean(parent_hits),
        "q_minus_parent_top1": statistics.fmean(top1_deltas),
        "q_minus_parent_top1_ci95": _bootstrap_mean_ci(
            top1_deltas, bootstrap_samples, seed
        ),
        "q_mean_regret": statistics.fmean(q_regrets),
        "parent_mean_regret": statistics.fmean(parent_regrets),
        "q_minus_parent_regret": statistics.fmean(regret_deltas),
        "q_minus_parent_regret_ci95": _bootstrap_mean_ci(
            regret_deltas, bootstrap_samples, seed + 1
        ),
        "q_mean_spearman": statistics.fmean(q_spearmans),
        "parent_mean_spearman": statistics.fmean(parent_spearmans),
        "q_minus_parent_spearman": statistics.fmean(spearman_deltas),
        "q_minus_parent_spearman_ci95": _bootstrap_mean_ci(
            spearman_deltas, bootstrap_samples, seed + 2
        ),
    }


def _ranking_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    criteria = {
        "regret_ci95_upper_below_zero": metrics["q_minus_parent_regret_ci95"][1] < 0.0,
        "spearman_ci95_lower_above_zero": (
            metrics["q_minus_parent_spearman_ci95"][0] > 0.0
        ),
        "top1_direction_positive": metrics["q_minus_parent_top1"] > 0.0,
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def _model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in (
            "entity_type_ids",
            "entity_card_ids",
            "entity_noble_ids",
            "entity_numeric_features",
            "entity_mask",
            "action_type_ids",
            "action_card_ids",
            "action_noble_ids",
            "action_numeric_features",
            "action_mask",
        )
    }


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def _validate_config(config: EntityQConfig) -> None:
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if config.learning_rate <= 0.0 or config.max_grad_norm <= 0.0:
        raise ValueError("learning_rate and max_grad_norm must be positive.")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")


def _config_payload(config: EntityQConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("corpus_path", "parent_checkpoint", "output_dir"):
        payload[key] = str(payload[key])
    return payload


if __name__ == "__main__":
    main()
