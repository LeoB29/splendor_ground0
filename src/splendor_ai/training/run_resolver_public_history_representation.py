"""Run experiment 054: matched snapshot versus public-history resolver values."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from splendor_ai.cfr.full_game import PublicHistoryEncoding, encode_public_history
from splendor_ai.encoding import ActionCodec
from splendor_ai.encoding.entity_encoding import ACTION_NUMERIC_SIZE
from splendor_ai.encoding.information_state import PublicActionEvent, public_action_event
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    LegalActionReplaySample,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.resolver_history_value import (
    FrozenParentResolverValueModel,
    ResolverHistoryValueConfig,
    load_frozen_parent_resolver_value_model,
    save_frozen_parent_resolver_value_checkpoint,
    trainable_parameter_count,
)


CFR_047_SHA256 = "7e59f4b8d91c2b2c3be2c593820706cc4167987aa482469b93570abd9e8dfd44"
RESULT_053_SHA256 = "7610c56789cf3491cc889e4f9689957bf06a769fb906f5314731903c933339ef"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
TRAIN_SHA256 = "6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5"
VALIDATION_SHA256 = "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052"
REQUIRED_STRATA = (
    "phase:early",
    "phase:mid",
    "phase:late",
    "seat:0",
    "seat:1",
    "outcome:winner",
    "outcome:loser",
)


@dataclass(frozen=True, slots=True)
class ResolverPublicHistoryExperimentConfig:
    output_dir: Path = Path("outputs/resolver_public_history_representation_054")
    result_047: Path = Path(
        "outputs/full_game_deep_cfr_feasibility_047/result.json"
    )
    result_053: Path = Path(
        "outputs/resolver_residual_ranking_distillation_053/result.json"
    )
    train_replay: Path = Path(
        "outputs/resolver_selfplay_distillation_052/train_targets.jsonl"
    )
    validation_replay: Path = Path(
        "outputs/resolver_selfplay_distillation_052/validation_targets.jsonl"
    )
    parent_checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 20
    seed: int = 54
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 540_054
    minimum_train_best_accuracy: float = 0.60
    minimum_validation_best_accuracy: float = 0.40
    minimum_validation_spearman: float = 0.25
    minimum_validation_pairwise_accuracy: float = 0.60
    maximum_generalization_gap: float = 0.20
    minimum_stratum_rows: int = 50
    minimum_stratum_accuracy: float = 0.30
    minimum_history_accuracy_gain: float = 0.05
    minimum_history_spearman_gain: float = 0.05
    minimum_history_capture_gain: float = 0.05
    maximum_history_stratum_regression: float = 0.02
    data_scale_train_accuracy: float = 0.70
    ranking_weight: float = 0.5
    smooth_l1_beta: float = 0.1
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ResolverPublicHistorySample:
    base: LegalActionReplaySample
    history: PublicHistoryEncoding
    group_action_rows: tuple[tuple[int, ...], ...]
    group_values: tuple[float, ...]
    parent_group: int
    metadata: dict[str, Any]


class ResolverPublicHistoryDataset(Dataset[ResolverPublicHistorySample]):
    """Reconstruct information-legal histories and four resolver value groups."""

    def __init__(
        self,
        path: Path,
        model: FrozenParentResolverValueModel,
    ) -> None:
        rows = _read_jsonl(path)
        base = LegalActionReplayDataset(
            path,
            observation_size=model.parent_config.observation_size,
            action_feature_size=model.parent_config.action_feature_size,
        )
        if len(rows) != len(base):
            raise ValueError("054 replay metadata and base rows differ.")

        env = SplendorEnv()
        codec = ActionCodec()
        indices_by_seed: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            indices_by_seed[int(row["game_seed"])].append(index)

        histories: dict[int, PublicHistoryEncoding] = {}
        unique_action_matches = True
        chronological = True
        observer_matches = True
        for seed in sorted(indices_by_seed):
            indices = sorted(
                indices_by_seed[seed],
                key=lambda index: int(rows[index]["turn_index"]),
            )
            turns = [int(rows[index]["turn_index"]) for index in indices]
            if turns != list(range(len(indices))):
                chronological = False
                raise ValueError(f"Game {seed} does not contain contiguous turns from zero.")
            public_history: list[PublicActionEvent] = []
            for index in indices:
                row = rows[index]
                state = deserialize_state_snapshot(row["state_snapshot"])
                if (
                    int(row["player_id"]) != state.current_player
                    or int(row["turn_index"]) != state.turn_index
                ):
                    observer_matches = False
                    raise ValueError("Stored observer or turn disagrees with replay state.")
                legal_actions = env.legal_actions(state)
                matches = [
                    action
                    for action in legal_actions
                    if codec.encode(state, action) == int(row["action_index"])
                ]
                if len(matches) != 1:
                    unique_action_matches = False
                    raise ValueError("Recorded action did not regenerate uniquely as legal.")
                histories[index] = encode_public_history(
                    public_history,
                    observer=state.current_player,
                )
                if len(histories[index].action_type_ids) != state.turn_index:
                    raise ValueError("Encoded history length does not equal turn index.")
                public_history.append(public_action_event(state, matches[0]))

        self._samples: list[ResolverPublicHistorySample] = []
        groups_complete = True
        parent_group_present = True
        finite_targets = True
        for index, row in enumerate(rows):
            if not bool(row.get("target_available", False)):
                continue
            base_sample = base[index]
            raw_groups = row.get("resolver_action_values", [])
            if not isinstance(raw_groups, list) or len(raw_groups) != 4:
                groups_complete = False
                raise ValueError("Every resolved 054 row must contain exactly four groups.")
            group_rows: list[tuple[int, ...]] = []
            group_values: list[float] = []
            seen_action_rows: set[int] = set()
            for group in raw_groups:
                action_indices = tuple(int(item) for item in group["action_indices"])
                if not action_indices:
                    raise ValueError("Resolver canonical group has no concrete action.")
                try:
                    action_rows = tuple(
                        base_sample.legal_action_indices.index(action_index)
                        for action_index in action_indices
                    )
                except ValueError as exc:
                    raise ValueError("Resolver group action is absent from legal rows.") from exc
                if seen_action_rows.intersection(action_rows):
                    raise ValueError("Resolver canonical groups overlap.")
                seen_action_rows.update(action_rows)
                value = float(group["value"])
                if not math.isfinite(value):
                    finite_targets = False
                    raise ValueError("Resolver group target must be finite.")
                group_rows.append(action_rows)
                group_values.append(value)
            parent_index = int(row["raw_parent_action_index"])
            parent_groups = [
                group_index
                for group_index, group in enumerate(raw_groups)
                if parent_index in [int(item) for item in group["action_indices"]]
            ]
            if len(parent_groups) != 1:
                parent_group_present = False
                raise ValueError("Raw parent action must occur in exactly one resolver group.")
            self._samples.append(
                ResolverPublicHistorySample(
                    base=base_sample,
                    history=histories[index],
                    group_action_rows=tuple(group_rows),
                    group_values=tuple(group_values),
                    parent_group=parent_groups[0],
                    metadata={
                        "game_seed": int(row["game_seed"]),
                        "turn_index": int(row["turn_index"]),
                        "phase": str(row["phase"]),
                        "seat": int(row["player_id"]),
                        "outcome": str(row["outcome_role"]),
                    },
                )
            )
        self.integrity = {
            "all_games_chronological_and_contiguous": chronological,
            "recorded_actions_uniquely_regenerated": unique_action_matches,
            "observer_and_turn_match_state": observer_matches,
            "history_length_equals_turn": all(
                len(sample.history.action_type_ids)
                == int(sample.metadata["turn_index"])
                for sample in self._samples
            ),
            "exactly_four_nonoverlapping_groups": groups_complete,
            "parent_group_present": parent_group_present,
            "finite_targets": finite_targets,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ResolverPublicHistorySample:
        return self._samples[index]


def collate_resolver_public_history_rows(
    rows: Sequence[ResolverPublicHistorySample],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot collate an empty 054 batch.")
    batch = collate_legal_action_replay_samples([row.base for row in rows])
    history_width = max(max(len(row.history.action_type_ids), 1) for row in rows)
    batch_size = len(rows)
    history_action_type_ids = torch.zeros((batch_size, history_width), dtype=torch.long)
    history_card_ids = torch.zeros((batch_size, history_width), dtype=torch.long)
    history_noble_ids = torch.zeros((batch_size, history_width), dtype=torch.long)
    history_numeric_features = torch.zeros(
        (batch_size, history_width, ACTION_NUMERIC_SIZE),
        dtype=torch.float32,
    )
    history_mask = torch.zeros((batch_size, history_width), dtype=torch.bool)
    legal_width = batch["legal_action_mask"].shape[1]
    group_membership = torch.zeros(
        (batch_size, 4, legal_width),
        dtype=torch.bool,
    )
    group_values = torch.zeros((batch_size, 4), dtype=torch.float32)
    best_group_mask = torch.zeros((batch_size, 4), dtype=torch.bool)
    parent_groups = torch.zeros(batch_size, dtype=torch.long)
    for row_index, row in enumerate(rows):
        length = len(row.history.action_type_ids)
        if length:
            history_action_type_ids[row_index, :length] = torch.tensor(
                row.history.action_type_ids,
                dtype=torch.long,
            )
            history_card_ids[row_index, :length] = torch.tensor(
                row.history.card_ids,
                dtype=torch.long,
            )
            history_noble_ids[row_index, :length] = torch.tensor(
                row.history.noble_ids,
                dtype=torch.long,
            )
            history_numeric_features[row_index, :length] = torch.tensor(
                row.history.numeric_features,
                dtype=torch.float32,
            )
            history_mask[row_index, :length] = True
        for group_index, action_rows in enumerate(row.group_action_rows):
            group_membership[row_index, group_index, list(action_rows)] = True
        values = torch.tensor(row.group_values, dtype=torch.float32)
        group_values[row_index] = values
        best_group_mask[row_index] = values == values.max()
        parent_groups[row_index] = row.parent_group
    batch.update(
        {
            "history_action_type_ids": history_action_type_ids,
            "history_card_ids": history_card_ids,
            "history_noble_ids": history_noble_ids,
            "history_numeric_features": history_numeric_features,
            "history_mask": history_mask,
            "group_membership": group_membership,
            "group_values": group_values,
            "best_group_mask": best_group_mask,
            "parent_group": parent_groups,
            "metadata": [row.metadata for row in rows],
        }
    )
    return batch


def aggregate_canonical_group_values(
    action_values: torch.Tensor,
    group_membership: torch.Tensor,
) -> torch.Tensor:
    membership = group_membership.to(dtype=action_values.dtype)
    counts = membership.sum(dim=2)
    if bool((counts <= 0).any()):
        raise ValueError("Every canonical successor group must have an action.")
    return torch.einsum("ba,bga->bg", action_values, membership) / counts


def compute_resolver_value_loss(
    group_predictions: torch.Tensor,
    group_targets: torch.Tensor,
    *,
    beta: float,
    ranking_weight: float,
) -> dict[str, torch.Tensor]:
    value_loss = nn.functional.smooth_l1_loss(
        group_predictions,
        group_targets,
        beta=beta,
    )
    pair_losses = []
    for left in range(group_targets.shape[1]):
        for right in range(left + 1, group_targets.shape[1]):
            target_delta = group_targets[:, left] - group_targets[:, right]
            non_tied = target_delta != 0.0
            if bool(non_tied.any()):
                prediction_delta = (
                    group_predictions[:, left] - group_predictions[:, right]
                )
                pair_losses.append(
                    nn.functional.softplus(
                        -target_delta[non_tied].sign()
                        * prediction_delta[non_tied]
                    )
                )
    ranking_loss = (
        torch.cat(pair_losses).mean()
        if pair_losses
        else group_predictions.sum() * 0.0
    )
    return {
        "total": value_loss + ranking_weight * ranking_loss,
        "value": value_loss,
        "ranking": ranking_loss,
    }


def run_experiment(
    config: ResolverPublicHistoryExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResolverPublicHistoryExperimentConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 054 result does not match frozen identity.")
        return result

    torch.set_num_threads(1)
    train_reference = _new_model(cfg, use_history=False)
    train_dataset = ResolverPublicHistoryDataset(cfg.train_replay, train_reference)
    if len(train_dataset) != 4_397:
        raise ValueError(f"054 expected 4,397 resolved train rows, got {len(train_dataset)}.")

    arms: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, FrozenParentResolverValueModel] = {}
    for name, use_history in (("snapshot", False), ("history", True)):
        model, training = _train_arm(cfg, train_dataset, name, use_history)
        trained_models[name] = model
        arms[name] = {"training": training}
    for name in ("snapshot", "history"):
        train_metrics, _records = _evaluate_model(
            cfg,
            trained_models[name],
            train_dataset,
            split=f"train_{name}",
        )
        arms[name]["train"] = train_metrics

    validation_dataset = ResolverPublicHistoryDataset(
        cfg.validation_replay,
        trained_models["snapshot"],
    )
    if len(validation_dataset) != 1_135:
        raise ValueError(
            f"054 expected 1,135 resolved validation rows, got {len(validation_dataset)}."
        )
    validation_records: dict[str, list[dict[str, Any]]] = {}
    for name in ("snapshot", "history"):
        metrics, records = _evaluate_model(
            cfg,
            trained_models[name],
            validation_dataset,
            split=f"validation_{name}",
        )
        arms[name]["validation"] = metrics
        validation_records[name] = records

    bootstrap = paired_game_cluster_bootstrap(
        validation_records["snapshot"],
        validation_records["history"],
        samples=cfg.bootstrap_samples,
        seed=cfg.bootstrap_seed,
    )
    learnability = {
        name: _learnability_gates(
            cfg,
            arms[name]["train"],
            arms[name]["validation"],
        )
        for name in ("snapshot", "history")
    }
    history_help = _history_help_gates(
        cfg,
        arms["snapshot"]["validation"],
        arms["history"]["validation"],
        learnability["history"],
        bootstrap,
    )
    decision = _decision(
        cfg,
        arms,
        learnability,
        history_help,
    )
    authorized = decision.endswith("authorize_scaled_corpus_protocol")
    integrity = {
        "train": train_dataset.integrity,
        "validation": validation_dataset.integrity,
        "validated_047_privacy_and_invariance": True,
        "models_finite": all(
            arms[name]["train"]["integrity"]["finite"]
            and arms[name]["validation"]["integrity"]["finite"]
            for name in ("snapshot", "history")
        ),
        "matched_train_rows": (
            arms["snapshot"]["train"]["rows"] == arms["history"]["train"]["rows"]
        ),
        "matched_validation_rows": (
            arms["snapshot"]["validation"]["rows"]
            == arms["history"]["validation"]["rows"]
        ),
    }
    integrity_passed = (
        all(train_dataset.integrity.values())
        and all(validation_dataset.integrity.values())
        and all(
            value
            for key, value in integrity.items()
            if key not in {"train", "validation"}
        )
    )
    payload = {
        "experiment": "054_resolver_public_history_representation",
        "protocol": (
            "docs/experiments/resolver_public_history_representation_054.md"
        ),
        "identity": _identity(cfg),
        "recipe": _recipe(cfg),
        "arms": arms,
        "paired_history_minus_snapshot_accuracy_ci95": bootstrap,
        "learnability_gates": learnability,
        "history_material_help_gates": history_help,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "decision": decision,
        "scaled_corpus_protocol_authorized": authorized and integrity_passed,
        "fresh_collection_authorized": False,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(result_path, payload)
    print(
        f"054 decision={decision} integrity={integrity_passed}",
        flush=True,
    )
    return payload


def _new_model(
    cfg: ResolverPublicHistoryExperimentConfig,
    *,
    use_history: bool,
) -> FrozenParentResolverValueModel:
    torch.manual_seed(cfg.seed)
    return load_frozen_parent_resolver_value_model(
        cfg.parent_checkpoint,
        use_history=use_history,
        device=cfg.device,
        config=ResolverHistoryValueConfig(),
    )


def _train_arm(
    cfg: ResolverPublicHistoryExperimentConfig,
    dataset: ResolverPublicHistoryDataset,
    name: str,
    use_history: bool,
) -> tuple[FrozenParentResolverValueModel, dict[str, Any]]:
    model = _new_model(cfg, use_history=use_history)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_resolver_public_history_rows,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    history = []
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals = {"total": 0.0, "value": 0.0, "ranking": 0.0}
        batches = 0
        for batch in loader:
            action_values, _parent_scores = _forward_batch(cfg, model, batch)
            group_predictions = aggregate_canonical_group_values(
                action_values,
                batch["group_membership"].to(cfg.device),
            )
            losses = compute_resolver_value_loss(
                group_predictions,
                batch["group_values"].to(cfg.device),
                beta=cfg.smooth_l1_beta,
                ranking_weight=cfg.ranking_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("054 training loader produced zero batches.")
        epoch_losses = {key: value / batches for key, value in totals.items()}
        history.append({"epoch": epoch, "losses": epoch_losses})
        print(
            f"[resolver054:{name}] epoch={epoch}/{cfg.epochs}"
            f" loss={epoch_losses['total']:.6f}",
            flush=True,
        )
    checkpoint = cfg.output_dir / f"{name}_epoch20.pt"
    save_frozen_parent_resolver_value_checkpoint(
        checkpoint,
        model,
        {
            "experiment": "054_resolver_public_history_representation",
            "arm": name,
            "epoch": cfg.epochs,
            "parent_sha256": PARENT_SHA256,
            "train_sha256": TRAIN_SHA256,
        },
    )
    return model, {
        "epochs": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "trainable_parameters": trainable_parameter_count(model),
        "seconds": time.perf_counter() - started,
    }


def _forward_batch(
    cfg: ResolverPublicHistoryExperimentConfig,
    model: FrozenParentResolverValueModel,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        batch["observation"].to(cfg.device),
        batch["legal_action_features"].to(cfg.device),
        batch["legal_action_mask"].to(cfg.device),
        batch["history_action_type_ids"].to(cfg.device),
        batch["history_card_ids"].to(cfg.device),
        batch["history_noble_ids"].to(cfg.device),
        batch["history_numeric_features"].to(cfg.device),
        batch["history_mask"].to(cfg.device),
    )


def _evaluate_model(
    cfg: ResolverPublicHistoryExperimentConfig,
    model: FrozenParentResolverValueModel,
    dataset: ResolverPublicHistoryDataset,
    *,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_resolver_public_history_rows,
    )
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            action_values, _parent_scores = _forward_batch(cfg, model, batch)
            predictions = aggregate_canonical_group_values(
                action_values,
                batch["group_membership"].to(cfg.device),
            ).cpu()
            targets = batch["group_values"]
            best_masks = batch["best_group_mask"]
            predicted_best = predictions.argmax(dim=1)
            for index, metadata in enumerate(batch["metadata"]):
                predicted = [float(value) for value in predictions[index]]
                target = [float(value) for value in targets[index]]
                predicted_group = int(predicted_best[index])
                parent_group = int(batch["parent_group"][index])
                best_mask = [bool(value) for value in best_masks[index]]
                correct = best_mask[predicted_group]
                parent_correct = best_mask[parent_group]
                pairwise_correct = 0
                pairwise_rows = 0
                for left in range(4):
                    for right in range(left + 1, 4):
                        target_delta = target[left] - target[right]
                        if target_delta == 0.0:
                            continue
                        pairwise_rows += 1
                        prediction_delta = predicted[left] - predicted[right]
                        pairwise_correct += int(
                            (prediction_delta > 0.0) == (target_delta > 0.0)
                        )
                records.append(
                    {
                        **metadata,
                        "mae_sum": sum(
                            abs(predicted_value - target_value)
                            for predicted_value, target_value in zip(
                                predicted,
                                target,
                                strict=True,
                            )
                        ),
                        "value_count": 4,
                        "best_correct": correct,
                        "parent_best_correct": parent_correct,
                        "disagreement": not parent_correct,
                        "disagreement_captured": correct and not parent_correct,
                        "spearman": _spearman(predicted, target),
                        "pairwise_correct": pairwise_correct,
                        "pairwise_rows": pairwise_rows,
                        "finite": all(math.isfinite(value) for value in predicted),
                    }
                )
    return _summarize_records(records, split), records


def _summarize_records(
    records: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    disagreements = [row for row in records if row["disagreement"]]
    strata = {
        **{
            f"phase:{phase}": _stratum(
                [row for row in records if row["phase"] == phase]
            )
            for phase in ("early", "mid", "late")
        },
        **{
            f"seat:{seat}": _stratum(
                [row for row in records if row["seat"] == seat]
            )
            for seat in (0, 1)
        },
        **{
            f"outcome:{outcome}": _stratum(
                [row for row in records if row["outcome"] == outcome]
            )
            for outcome in ("winner", "loser")
        },
    }
    pairwise_rows = sum(int(row["pairwise_rows"]) for row in records)
    return {
        "split": split,
        "rows": len(records),
        "games": len({int(row["game_seed"]) for row in records}),
        "four_value_mae": (
            sum(float(row["mae_sum"]) for row in records)
            / sum(int(row["value_count"]) for row in records)
        ),
        "best_group_accuracy": _mean_bool(records, "best_correct"),
        "mean_within_state_spearman": statistics.fmean(
            float(row["spearman"]) for row in records
        ),
        "pairwise_sign_accuracy": (
            sum(int(row["pairwise_correct"]) for row in records) / pairwise_rows
        ),
        "resolver_parent_disagreement": {
            "rows": len(disagreements),
            "capture": _mean_bool(disagreements, "disagreement_captured"),
        },
        "strata": strata,
        "integrity": {
            "finite": all(bool(row["finite"]) for row in records),
            "four_values_per_row": all(
                int(row["value_count"]) == 4 for row in records
            ),
            "non_tied_pairs_present": pairwise_rows > 0,
        },
    }


def _stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "best_group_accuracy": _mean_bool(rows, "best_correct"),
        "mean_spearman": (
            statistics.fmean(float(row["spearman"]) for row in rows)
            if rows
            else 0.0
        ),
    }


def _learnability_gates(
    cfg: ResolverPublicHistoryExperimentConfig,
    train: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, bool]:
    return {
        "training_best_accuracy": (
            float(train["best_group_accuracy"])
            >= cfg.minimum_train_best_accuracy
        ),
        "validation_best_accuracy": (
            float(validation["best_group_accuracy"])
            >= cfg.minimum_validation_best_accuracy
        ),
        "validation_spearman": (
            float(validation["mean_within_state_spearman"])
            >= cfg.minimum_validation_spearman
        ),
        "validation_pairwise_accuracy": (
            float(validation["pairwise_sign_accuracy"])
            >= cfg.minimum_validation_pairwise_accuracy
        ),
        "bounded_generalization_gap": (
            float(train["best_group_accuracy"])
            - float(validation["best_group_accuracy"])
            <= cfg.maximum_generalization_gap
        ),
        "all_required_strata": all(
            int(validation["strata"][name]["rows"]) >= cfg.minimum_stratum_rows
            and float(validation["strata"][name]["best_group_accuracy"])
            >= cfg.minimum_stratum_accuracy
            for name in REQUIRED_STRATA
        ),
        "integrity": (
            all(train["integrity"].values())
            and all(validation["integrity"].values())
        ),
    }


def _history_help_gates(
    cfg: ResolverPublicHistoryExperimentConfig,
    snapshot: dict[str, Any],
    history: dict[str, Any],
    history_learnability: dict[str, bool],
    bootstrap: dict[str, float],
) -> dict[str, bool]:
    return {
        "history_learnable": all(history_learnability.values()),
        "best_accuracy_gain": (
            float(history["best_group_accuracy"])
            - float(snapshot["best_group_accuracy"])
            >= cfg.minimum_history_accuracy_gain
        ),
        "bootstrap_lower_above_zero": bootstrap["lower"] > 0.0,
        "spearman_gain": (
            float(history["mean_within_state_spearman"])
            - float(snapshot["mean_within_state_spearman"])
            >= cfg.minimum_history_spearman_gain
        ),
        "disagreement_capture_gain": (
            float(history["resolver_parent_disagreement"]["capture"])
            - float(snapshot["resolver_parent_disagreement"]["capture"])
            >= cfg.minimum_history_capture_gain
        ),
        "no_stratum_regression": all(
            float(history["strata"][name]["best_group_accuracy"])
            >= (
                float(snapshot["strata"][name]["best_group_accuracy"])
                - cfg.maximum_history_stratum_regression
            )
            for name in REQUIRED_STRATA
        ),
    }


def _decision(
    cfg: ResolverPublicHistoryExperimentConfig,
    arms: dict[str, dict[str, Any]],
    learnability: dict[str, dict[str, bool]],
    history_help: dict[str, bool],
) -> str:
    if all(history_help.values()):
        return "history_representation_promising_authorize_scaled_corpus_protocol"
    if all(learnability["snapshot"].values()):
        return "snapshot_representation_sufficient_authorize_scaled_corpus_protocol"
    if all(learnability["history"].values()):
        return "history_representation_promising_authorize_scaled_corpus_protocol"
    for name in ("snapshot", "history"):
        train_accuracy = float(arms[name]["train"]["best_group_accuracy"])
        validation_accuracy = float(arms[name]["validation"]["best_group_accuracy"])
        if (
            train_accuracy >= cfg.data_scale_train_accuracy
            and train_accuracy - validation_accuracy
            > cfg.maximum_generalization_gap
        ):
            return "data_scale_primary_authorize_scaled_corpus_protocol"
    return "resolver_targets_not_learnable_with_current_frozen_features"


def paired_game_cluster_bootstrap(
    snapshot: list[dict[str, Any]],
    history: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    snapshot_map = {
        (int(row["game_seed"]), int(row["turn_index"])): row for row in snapshot
    }
    history_map = {
        (int(row["game_seed"]), int(row["turn_index"])): row for row in history
    }
    if snapshot_map.keys() != history_map.keys() or not snapshot_map:
        raise ValueError("Snapshot and history validation records must align.")
    groups: dict[int, list[float]] = defaultdict(list)
    for key in sorted(snapshot_map):
        groups[key[0]].append(
            float(history_map[key]["best_correct"])
            - float(snapshot_map[key]["best_correct"])
        )
    clusters = [(len(values), sum(values)) for values in groups.values()]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draws.append(
            sum(item[1] for item in selected) / sum(item[0] for item in selected)
        )
    draws.sort()
    return {
        "mean": statistics.fmean(
            float(history_map[key]["best_correct"])
            - float(snapshot_map[key]["best_correct"])
            for key in snapshot_map
        ),
        "lower": _quantile(draws, 0.025),
        "upper": _quantile(draws, 0.975),
    }


def _spearman(left: list[float], right: list[float]) -> float:
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_rank, right_rank, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left_rank))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right_rank))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _validate_config(cfg: ResolverPublicHistoryExperimentConfig) -> None:
    frozen = {
        cfg.result_047: CFR_047_SHA256,
        cfg.result_053: RESULT_053_SHA256,
        cfg.train_replay: TRAIN_SHA256,
        cfg.validation_replay: VALIDATION_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 054 prerequisite mismatch: {path}")
    result_047 = json.loads(cfg.result_047.read_text(encoding="utf-8"))
    if (
        not result_047["gates"]["privacy_and_invariance"]
        or not result_047["gates"]["schema_corpus"]
    ):
        raise ValueError("054 requires the validated 047 representation gates.")
    result_053 = json.loads(cfg.result_053.read_text(encoding="utf-8"))
    if result_053.get("decision") != "validation_residual_gate_failed":
        raise ValueError("054 requires the frozen failed 053 validation decision.")
    frozen_contract = {
        "learning_rate": (cfg.learning_rate, 1e-3),
        "weight_decay": (cfg.weight_decay, 1e-4),
        "batch_size": (cfg.batch_size, 128),
        "epochs": (cfg.epochs, 20),
        "seed": (cfg.seed, 54),
        "bootstrap_samples": (cfg.bootstrap_samples, 20_000),
        "bootstrap_seed": (cfg.bootstrap_seed, 540_054),
        "ranking_weight": (cfg.ranking_weight, 0.5),
        "smooth_l1_beta": (cfg.smooth_l1_beta, 0.1),
        "device": (cfg.device, "cpu"),
    }
    changed = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in frozen_contract.items()
        if observed != expected
    }
    if changed:
        raise ValueError(f"054 frozen configuration changed: {changed}")


def _recipe(cfg: ResolverPublicHistoryExperimentConfig) -> dict[str, Any]:
    return {
        "parent_frozen": True,
        "history_model": asdict(ResolverHistoryValueConfig()),
        "value_head": "Linear(449,128),GELU,Linear(128,1)",
        "loss": {
            "smooth_l1_beta": cfg.smooth_l1_beta,
            "pairwise_softplus_weight": cfg.ranking_weight,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "arm_order": ["snapshot", "history"],
        "device": cfg.device,
    }


def _identity(cfg: ResolverPublicHistoryExperimentConfig) -> dict[str, Any]:
    return {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "sha256": {
            "result_047": _sha256_file(cfg.result_047),
            "result_053": _sha256_file(cfg.result_053),
            "train_052": _sha256_file(cfg.train_replay),
            "validation_052": _sha256_file(cfg.validation_replay),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
        },
    }


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(bool(row[key]) for row in rows) if rows else 0.0


def _quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
