"""CLI entry point for supervised legal-action scorer training."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

from .dataset import LegalActionReplayDataset
from .legal_action_supervised import (
    create_legal_action_replay_dataloader,
    fit_legal_action_scorer_dataloaders_with_artifacts,
    split_legal_action_replay_dataset,
)
from .model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    LegalActionScorerConfig,
    LegalActionScorerModel,
    LegalActionSetScorerConfig,
    LegalActionSetScorerModel,
)
from .supervised import BatchProgress, SupervisedTrainConfig


MODEL_ARCHITECTURE_CHOICES = ("independent", "set", "attention")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Splendor legal-action scorer from snapshot-enabled replay JSONL data."
    )
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Snapshot-enabled replay JSONL path. Repeat to mix multiple corpora.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--search-distill-loss-weight",
        type=float,
        default=0.0,
        help="Optional loss weight for soft legal-action targets derived from stored search scores.",
    )
    parser.add_argument(
        "--search-distill-temperature",
        type=float,
        default=1_000.0,
        help="Temperature used to convert stored search values into soft action targets.",
    )
    parser.add_argument(
        "--root-visit-loss-weight",
        type=float,
        default=0.0,
        help="Optional loss weight for stored normalized PUCT root-visit targets.",
    )
    parser.add_argument(
        "--preference-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional pairwise loss weight for rows with model_top_action_index. "
            "The replay action is scored above the stored model-top competitor."
        ),
    )
    parser.add_argument(
        "--preference-margin",
        type=float,
        default=1.0,
        help="Margin used by the pairwise preference loss.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--architecture",
        choices=MODEL_ARCHITECTURE_CHOICES,
        default="independent",
        help="Use the original independent action scorer or the action-set-aware scorer.",
    )
    parser.add_argument("--state-hidden-size", type=int, default=256)
    parser.add_argument("--action-hidden-size", type=int, default=128)
    parser.add_argument("--set-context-hidden-size", type=int, default=128)
    parser.add_argument("--scorer-hidden-size", type=int, default=256)
    parser.add_argument("--state-depth", type=int, default=3)
    parser.add_argument("--action-depth", type=int, default=2)
    parser.add_argument("--set-context-depth", type=int, default=2)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-feedforward-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument(
        "--validation-split-unit",
        choices=("row", "game", "source"),
        default="game",
        help="Validation split granularity. Use 'game' to avoid positions from the same game leaking across splits.",
    )
    parser.add_argument(
        "--policy-target-filter",
        choices=("all", "winner"),
        default="all",
        help="Which replay actions to imitate. 'winner' keeps only actions by the eventual game winner.",
    )
    parser.add_argument(
        "--winner-action-weight",
        type=float,
        default=1.0,
        help="Multiplicative sample weight for actions by the eventual winner.",
    )
    parser.add_argument(
        "--loser-action-weight",
        type=float,
        default=1.0,
        help="Multiplicative sample weight for actions by the eventual loser.",
    )
    parser.add_argument(
        "--draw-action-weight",
        type=float,
        default=1.0,
        help="Multiplicative sample weight for drawn-game actions.",
    )
    parser.add_argument(
        "--preference-rows-policy-weight",
        type=float,
        default=1.0,
        help=(
            "Additional multiplier for CE policy loss on rows that contain "
            "a valid model_top_action_index preference pair. Use 0 for "
            "preference-only correction rows."
        ),
    )
    parser.add_argument(
        "--preference-rows-distill-weight",
        type=float,
        default=1.0,
        help=(
            "Additional multiplier for search-distillation loss on rows that "
            "contain a valid model_top_action_index preference pair."
        ),
    )
    parser.add_argument(
        "--preference-rows-preference-weight",
        type=float,
        default=1.0,
        help=(
            "Additional multiplier for pairwise preference loss on rows that "
            "contain a valid model_top_action_index preference pair."
        ),
    )
    parser.add_argument("--exclude-stalled-games", action="store_true")
    parser.add_argument("--exclude-timeout-games", action="store_true")
    parser.add_argument("--max-game-model-loop-fallback-triggers", type=int, default=None)
    return parser


def _build_model(
    args: argparse.Namespace,
) -> LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel:
    if args.init_checkpoint is None:
        return _new_model_from_args(args)

    checkpoint_path = Path(args.init_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = _new_model_from_checkpoint_config(args, checkpoint.get("model_config", {}))
    checkpoint_model_kind = str(checkpoint.get("model_kind", "legal_action_scorer"))
    target_model_kind = _model_kind_for_architecture(args.architecture)
    excluded_prefixes = ("score_head.",) if checkpoint_model_kind != target_model_kind else ()
    loaded, skipped = _load_compatible_state_dict(
        model,
        checkpoint["model_state_dict"],
        excluded_prefixes=excluded_prefixes,
    )
    print(
        f"[legal-train] initialized model from checkpoint: {checkpoint_path}"
        f" compatible_tensors={loaded} skipped_tensors={skipped}"
    )
    return model


def _new_model_from_args(
    args: argparse.Namespace,
) -> LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel:
    if args.architecture == "attention":
        return LegalActionAttentionScorerModel(
            LegalActionAttentionScorerConfig(
                state_hidden_size=args.state_hidden_size,
                action_hidden_size=args.action_hidden_size,
                scorer_hidden_size=args.scorer_hidden_size,
                state_depth=args.state_depth,
                action_depth=args.action_depth,
                attention_layers=args.attention_layers,
                attention_heads=args.attention_heads,
                attention_feedforward_size=args.attention_feedforward_size,
                dropout=args.dropout,
            )
        )
    if args.architecture == "set":
        return LegalActionSetScorerModel(
            LegalActionSetScorerConfig(
                state_hidden_size=args.state_hidden_size,
                action_hidden_size=args.action_hidden_size,
                set_context_hidden_size=args.set_context_hidden_size,
                scorer_hidden_size=args.scorer_hidden_size,
                state_depth=args.state_depth,
                action_depth=args.action_depth,
                set_context_depth=args.set_context_depth,
                dropout=args.dropout,
            )
        )
    return LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=args.state_hidden_size,
            action_hidden_size=args.action_hidden_size,
            scorer_hidden_size=args.scorer_hidden_size,
            state_depth=args.state_depth,
            action_depth=args.action_depth,
            dropout=args.dropout,
        )
    )


def _new_model_from_checkpoint_config(
    args: argparse.Namespace,
    checkpoint_config: dict[str, object],
) -> LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel:
    if args.architecture == "attention":
        payload = _filtered_config_payload(
            LegalActionAttentionScorerConfig,
            checkpoint_config,
            {
                "attention_layers": args.attention_layers,
                "attention_heads": args.attention_heads,
                "attention_feedforward_size": args.attention_feedforward_size,
            },
        )
        return LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(**payload))
    if args.architecture == "set":
        payload = _filtered_config_payload(
            LegalActionSetScorerConfig,
            checkpoint_config,
            {
                "set_context_hidden_size": args.set_context_hidden_size,
                "set_context_depth": args.set_context_depth,
            },
        )
        return LegalActionSetScorerModel(LegalActionSetScorerConfig(**payload))
    payload = _filtered_config_payload(LegalActionScorerConfig, checkpoint_config, {})
    return LegalActionScorerModel(LegalActionScorerConfig(**payload))


def _filtered_config_payload(
    config_type: (
        type[LegalActionScorerConfig]
        | type[LegalActionSetScorerConfig]
        | type[LegalActionAttentionScorerConfig]
    ),
    checkpoint_config: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    allowed = {field.name for field in fields(config_type)}
    payload = {
        key: value
        for key, value in checkpoint_config.items()
        if key in allowed
    }
    payload.update(overrides)
    return payload


def _load_compatible_state_dict(
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel,
    checkpoint_state_dict: dict[str, torch.Tensor],
    excluded_prefixes: tuple[str, ...] = (),
) -> tuple[int, int]:
    model_state_dict = model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in checkpoint_state_dict.items()
        if name in model_state_dict
        and model_state_dict[name].shape == tensor.shape
        and not name.startswith(excluded_prefixes)
    }
    model_state_dict.update(compatible)
    model.load_state_dict(model_state_dict)
    return len(compatible), len(checkpoint_state_dict) - len(compatible)


def _model_kind_for_architecture(architecture: str) -> str:
    if architecture == "attention":
        return "legal_action_attention_scorer"
    if architecture == "set":
        return "legal_action_set_scorer"
    return "legal_action_scorer"


def _checkpoint_filenames(architecture: str) -> tuple[str, str, str]:
    if architecture == "attention":
        return (
            "legal_action_attention_scorer.pt",
            "legal_action_attention_scorer_best.pt",
            "legal_action_attention_scorer_metrics.json",
        )
    if architecture == "set":
        return (
            "legal_action_set_scorer.pt",
            "legal_action_set_scorer_best.pt",
            "legal_action_set_scorer_metrics.json",
        )
    return (
        "legal_action_scorer.pt",
        "legal_action_scorer_best.pt",
        "legal_action_scorer_metrics.json",
    )


def _save_checkpoint(
    path: Path,
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel,
    train_config_payload: dict[str, object],
    model_kind: str,
) -> None:
    torch.save(
        {
            "model_kind": model_kind,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "train_config": train_config_payload,
        },
        path,
    )


def _print_batch_progress(progress: BatchProgress) -> None:
    rate = progress.samples / progress.elapsed_seconds if progress.elapsed_seconds > 0 else 0.0
    eta = (
        ((progress.total_batches - progress.batch_index) * (progress.elapsed_seconds / progress.batch_index))
        if progress.batch_index > 0
        else math.inf
    )
    eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
    print(
        f"[{progress.phase}] epoch={progress.epoch_index}/{progress.total_epochs}"
        f" batch={progress.batch_index}/{progress.total_batches}"
        f" elapsed={progress.elapsed_seconds:.1f}s"
        f" rate={rate:.1f} samples/s"
        f" eta={eta_text}"
        f" loss={progress.total_loss:.4f}"
        f" policy_acc={progress.policy_accuracy:.4f}"
    )


def _serialize_metric(metric: object | None) -> dict[str, object] | None:
    return asdict(metric) if metric is not None else None


def _train_config_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "replay_paths": [str(Path(path)) for path in args.replay_path],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "value_loss_weight": args.value_loss_weight,
        "search_distill_loss_weight": args.search_distill_loss_weight,
        "search_distill_temperature": args.search_distill_temperature,
        "root_visit_loss_weight": args.root_visit_loss_weight,
        "preference_loss_weight": args.preference_loss_weight,
        "preference_margin": args.preference_margin,
        "epochs": args.epochs,
        "device": args.device,
        "init_checkpoint": args.init_checkpoint,
        "architecture": args.architecture,
        "attention_layers": args.attention_layers,
        "attention_heads": args.attention_heads,
        "attention_feedforward_size": args.attention_feedforward_size,
        "include_stalled_games": not args.exclude_stalled_games,
        "include_timed_out_games": not args.exclude_timeout_games,
        "max_model_loop_fallback_triggers": args.max_game_model_loop_fallback_triggers,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "validation_split_unit": args.validation_split_unit,
        "policy_target_filter": args.policy_target_filter,
        "winner_action_weight": args.winner_action_weight,
        "loser_action_weight": args.loser_action_weight,
        "draw_action_weight": args.draw_action_weight,
        "preference_rows_policy_weight": args.preference_rows_policy_weight,
        "preference_rows_distill_weight": args.preference_rows_distill_weight,
        "preference_rows_preference_weight": args.preference_rows_preference_weight,
        "log_every_batches": args.log_every_batches,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = tuple(Path(path) for path in args.replay_path)

    print(
        f"[legal-train] replay_paths={', '.join(str(path) for path in replay_paths)}"
        f" output_dir={output_dir}"
        f" device={args.device}"
        f" epochs={args.epochs}"
        f" batch_size={args.batch_size}"
        f" architecture={args.architecture}"
        f" search_distill_loss_weight={args.search_distill_loss_weight}"
        f" root_visit_loss_weight={args.root_visit_loss_weight}"
        f" preference_loss_weight={args.preference_loss_weight}"
        f" policy_target_filter={args.policy_target_filter}"
        f" validation_split_unit={args.validation_split_unit}"
    )
    model = _build_model(args)
    train_config = SupervisedTrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        epochs=args.epochs,
        device=args.device,
        include_stalled_games=not args.exclude_stalled_games,
        include_timed_out_games=not args.exclude_timeout_games,
        max_model_loop_fallback_triggers=args.max_game_model_loop_fallback_triggers,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )

    print("[legal-train] loading snapshot replay dataset...")
    dataset = LegalActionReplayDataset(
        replay_paths,
        include_stalled_games=train_config.include_stalled_games,
        include_timed_out_games=train_config.include_timed_out_games,
        max_model_loop_fallback_triggers=train_config.max_model_loop_fallback_triggers,
        policy_target_filter=args.policy_target_filter,
        winner_action_weight=args.winner_action_weight,
        loser_action_weight=args.loser_action_weight,
        draw_action_weight=args.draw_action_weight,
        preference_rows_policy_weight=args.preference_rows_policy_weight,
        preference_rows_distill_weight=args.preference_rows_distill_weight,
        preference_rows_preference_weight=args.preference_rows_preference_weight,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    train_dataset, validation_dataset = split_legal_action_replay_dataset(
        dataset,
        validation_fraction=train_config.validation_fraction,
        seed=train_config.validation_seed,
        validation_split_unit=args.validation_split_unit,
    )
    print(
        f"[legal-train] dataset_samples={len(dataset)}"
        f" train_samples={len(train_dataset)}"
        f" validation_samples={len(validation_dataset) if validation_dataset is not None else 0}"
    )
    train_dataloader = create_legal_action_replay_dataloader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = create_legal_action_replay_dataloader(
            validation_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
        )

    print("[legal-train] starting optimization...")
    start_time = time.perf_counter()
    artifacts = fit_legal_action_scorer_dataloaders_with_artifacts(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=train_config,
        model=model,
        log_every_batches=args.log_every_batches,
        batch_progress_callback=_print_batch_progress,
        search_distill_loss_weight=args.search_distill_loss_weight,
        search_distill_temperature=args.search_distill_temperature,
        root_visit_loss_weight=args.root_visit_loss_weight,
        preference_loss_weight=args.preference_loss_weight,
        preference_margin=args.preference_margin,
    )
    elapsed = time.perf_counter() - start_time

    model_kind = _model_kind_for_architecture(args.architecture)
    checkpoint_name, best_checkpoint_name, metrics_name = _checkpoint_filenames(args.architecture)
    checkpoint_path = output_dir / checkpoint_name
    best_checkpoint_path = output_dir / best_checkpoint_name
    metrics_path = output_dir / metrics_name
    train_payload = _train_config_payload(args)
    _save_checkpoint(checkpoint_path, artifacts.model, train_payload, model_kind)
    print(f"[legal-train] wrote final checkpoint: {checkpoint_path}")

    best_checkpoint_written = False
    if artifacts.best_model_state_dict is not None and artifacts.best_epoch_index is not None:
        best_model = artifacts.model.__class__(artifacts.model.config)
        best_model.load_state_dict(artifacts.best_model_state_dict)
        _save_checkpoint(best_checkpoint_path, best_model, train_payload, model_kind)
        best_checkpoint_written = True
        print(
            f"[legal-train] wrote best validation checkpoint: {best_checkpoint_path}"
            f" (epoch {artifacts.best_epoch_index})"
        )

    metrics_payload = {
        "dataset": {
            "replay_path": str(replay_paths[0]) if len(replay_paths) == 1 else None,
            "replay_paths": [str(path) for path in replay_paths],
            "total_samples": len(dataset),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "validation_split_unit": args.validation_split_unit,
            "policy_target_filter": args.policy_target_filter,
            "winner_action_weight": args.winner_action_weight,
            "loser_action_weight": args.loser_action_weight,
            "draw_action_weight": args.draw_action_weight,
            "excluded_stalled_games": args.exclude_stalled_games,
            "excluded_timeout_games": args.exclude_timeout_games,
            "max_game_model_loop_fallback_triggers": args.max_game_model_loop_fallback_triggers,
            "search_distill_loss_weight": args.search_distill_loss_weight,
            "search_distill_temperature": args.search_distill_temperature,
            "root_visit_loss_weight": args.root_visit_loss_weight,
            "preference_loss_weight": args.preference_loss_weight,
            "preference_margin": args.preference_margin,
            "architecture": args.architecture,
            "model_kind": model_kind,
            "preference_rows_policy_weight": args.preference_rows_policy_weight,
            "preference_rows_distill_weight": args.preference_rows_distill_weight,
            "preference_rows_preference_weight": args.preference_rows_preference_weight,
        },
        "checkpoints": {
            "final_checkpoint_path": checkpoint_path.name,
            "best_validation_checkpoint_path": best_checkpoint_path.name if best_checkpoint_written else None,
            "best_validation_epoch": artifacts.best_epoch_index,
        },
        "epochs": [
            {
                "train": _serialize_metric(metric.train),
                "validation": _serialize_metric(metric.validation),
                "is_best_validation": epoch_index == artifacts.best_epoch_index,
            }
            for epoch_index, metric in enumerate(artifacts.metrics, start=1)
        ],
        "timing": {
            "training_seconds": elapsed,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[legal-train] wrote metrics: {metrics_path}")

    final_epoch = artifacts.metrics[-1]
    print(
        "Legal-action scorer training complete:"
        f" train_loss={final_epoch.train.total_loss:.4f}"
        f" train_acc={final_epoch.train.policy_accuracy:.4f}"
        f" train_samples={final_epoch.train.samples}"
        f" training_seconds={elapsed:.1f}"
    )
    if final_epoch.validation is not None:
        print(
            "Validation:"
            f" loss={final_epoch.validation.total_loss:.4f}"
            f" acc={final_epoch.validation.policy_accuracy:.4f}"
            f" samples={final_epoch.validation.samples}"
            f" best_epoch={artifacts.best_epoch_index}"
        )


if __name__ == "__main__":
    main()
