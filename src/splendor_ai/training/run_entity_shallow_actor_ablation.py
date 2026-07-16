"""Train and gate a shallow-actor-only entity imitation ablation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from splendor_ai.training.entity_action_supervised import (
    EntityActionReplayDataset,
    EntityTrainMetrics,
    create_entity_action_replay_dataloader,
    fit_entity_action_scorer,
)
from splendor_ai.training.model import CanonicalEntityActionScorerModel
from splendor_ai.training.run_entity_supervised_scale import (
    EntityScaleConfig,
    _candidate_configs,
    _load_checkpoint_model,
    _offline_gates,
    _paired_loss_comparison,
    evaluate_detailed,
    split_game_heldout_indices,
)


@dataclass(frozen=True, slots=True)
class ShallowActorAblationConfig:
    source_dir: Path
    output_dir: Path
    parent_checkpoint: Path
    split_seed: int = 2_026_0716
    init_seed: int = 2_026_0716
    train_games: int = 400
    validation_games: int = 50
    confirmation_games: int = 50
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bootstrap_samples: int = 10_000
    device: str = "cpu"
    actor_label: str = "ShallowSearchBot"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--split-seed", type=int, default=2_026_0716)
    parser.add_argument("--init-seed", type=int, default=2_026_0716)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> ShallowActorAblationConfig:
    return ShallowActorAblationConfig(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        parent_checkpoint=Path(args.parent_checkpoint),
        split_seed=args.split_seed,
        init_seed=args.init_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )


def filter_indices_by_actor(
    dataset: EntityActionReplayDataset,
    indices: list[int],
    actor_label: str,
) -> list[int]:
    return [
        index
        for index in indices
        if dataset.actor_label_for_index(index) == actor_label
    ]


def _scale_gate_config(config: ShallowActorAblationConfig) -> EntityScaleConfig:
    return EntityScaleConfig(
        output_dir=config.output_dir,
        parent_checkpoint=config.parent_checkpoint,
        games=config.train_games + config.validation_games + config.confirmation_games,
        train_games=config.train_games,
        validation_games=config.validation_games,
        confirmation_games=config.confirmation_games,
        split_seed=config.split_seed,
        init_seed=config.init_seed,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        bootstrap_samples=config.bootstrap_samples,
        device=config.device,
    )


def _validate(config: ShallowActorAblationConfig) -> Path:
    replay_path = config.source_dir / "replays.jsonl"
    split_path = config.source_dir / "game_splits.json"
    if not replay_path.exists() or not split_path.exists():
        raise FileNotFoundError("Source directory must contain replays.jsonl and game_splits.json.")
    if not config.parent_checkpoint.exists():
        raise FileNotFoundError(config.parent_checkpoint)
    if config.epochs <= 0 or config.batch_size <= 0 or config.bootstrap_samples < 0:
        raise ValueError("epochs and batch_size must be positive; bootstrap_samples must be non-negative.")
    return replay_path


def run_shallow_actor_ablation(config: ShallowActorAblationConfig) -> dict[str, Any]:
    replay_path = _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EntityActionReplayDataset(replay_path)
    train, validation, confirmation, split_seeds = split_game_heldout_indices(
        dataset,
        config.train_games,
        config.validation_games,
        config.confirmation_games,
        config.split_seed,
    )
    persisted_splits = json.loads(
        (config.source_dir / "game_splits.json").read_text(encoding="utf-8")
    )
    if split_seeds != persisted_splits:
        raise ValueError("Recomputed split does not match the frozen source split.")
    filtered = {
        "train": filter_indices_by_actor(dataset, train, config.actor_label),
        "validation": filter_indices_by_actor(dataset, validation, config.actor_label),
        "confirmation": filter_indices_by_actor(dataset, confirmation, config.actor_label),
    }
    if any(not indices for indices in filtered.values()):
        raise RuntimeError("Actor filtering produced an empty split.")
    subsets = {name: Subset(dataset, indices) for name, indices in filtered.items()}
    model_config = _candidate_configs()["width64"]
    torch.manual_seed(config.init_seed)
    model = CanonicalEntityActionScorerModel(model_config)
    started = time.perf_counter()

    def report_epoch(
        epoch: int,
        train_metric: EntityTrainMetrics,
        validation_metric: EntityTrainMetrics | None,
    ) -> None:
        validation_text = (
            f" validation_loss={validation_metric.policy_loss:.4f}"
            if validation_metric is not None
            else ""
        )
        print(
            f"[shallow-ablation] epoch={epoch}/{config.epochs} "
            f"train_loss={train_metric.policy_loss:.4f}{validation_text}",
            flush=True,
        )

    artifacts = fit_entity_action_scorer(
        create_entity_action_replay_dataloader(subsets["train"], config.batch_size, shuffle=True),
        create_entity_action_replay_dataloader(subsets["validation"], config.batch_size, shuffle=False),
        model=model,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        selection_metric="policy_loss",
        progress_callback=report_epoch,
    )
    if artifacts.best_validation_state_dict is None:
        raise RuntimeError("Training did not produce a validation-selected checkpoint.")
    model.load_state_dict(artifacts.best_validation_state_dict)
    model.to(config.device).eval()
    checkpoint_path = config.output_dir / "canonical_entity_action_scorer_shallow_actor_width64_best.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": artifacts.best_validation_state_dict,
            "model_config": asdict(model_config),
            "train_config": _config_payload(config),
        },
        checkpoint_path,
    )
    parent = _load_checkpoint_model(config.parent_checkpoint, config.device)
    gate_config = _scale_gate_config(config)
    evaluations: dict[str, Any] = {}
    for offset, split in enumerate(("validation", "confirmation")):
        parent_metrics = evaluate_detailed(parent, subsets[split], config.batch_size, config.device)
        candidate_metrics = evaluate_detailed(model, subsets[split], config.batch_size, config.device)
        comparison = _paired_loss_comparison(
            candidate_metrics,
            parent_metrics,
            config.bootstrap_samples,
            config.split_seed + offset,
        )
        evaluations[split] = {
            "parent": parent_metrics,
            "candidate": candidate_metrics,
            "comparison": comparison,
            "gates": _offline_gates(gate_config, candidate_metrics, parent_metrics, comparison),
        }
    confirmation_gates = evaluations["confirmation"]["gates"]
    payload = {
        "experiment": "canonical_entity_shallow_actor_ablation_v1",
        "config": _config_payload(config),
        "source_replay": str(replay_path),
        "split_seeds": split_seeds,
        "source_positions": {
            "train": len(train), "validation": len(validation), "confirmation": len(confirmation)
        },
        "filtered_positions": {name: len(indices) for name, indices in filtered.items()},
        "model_config": asdict(model_config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": artifacts.best_epoch_index,
        "elapsed_seconds": time.perf_counter() - started,
        "train_history": [asdict(metric) for metric in artifacts.train_metrics],
        "validation_history": [asdict(metric) for metric in artifacts.validation_metrics],
        "evaluations": evaluations,
        "checkpoint": str(checkpoint_path),
        "confirmation_gates": confirmation_gates,
        "offline_gate_passed": all(confirmation_gates.values()),
        "match_gate_eligible": all(confirmation_gates.values()),
    }
    (config.output_dir / "shallow_actor_ablation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "filtered_positions": payload["filtered_positions"],
                "best_epoch": payload["best_epoch"],
                "confirmation_gates": confirmation_gates,
                "match_gate_eligible": payload["match_gate_eligible"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def _config_payload(config: ShallowActorAblationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["source_dir"] = str(config.source_dir)
    payload["output_dir"] = str(config.output_dir)
    payload["parent_checkpoint"] = str(config.parent_checkpoint)
    return payload


def main() -> None:
    run_shallow_actor_ablation(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()
