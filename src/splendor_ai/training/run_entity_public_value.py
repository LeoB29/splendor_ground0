"""Train and gate one public-belief canonical entity score-margin value model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.bots.puct_bot import legal_heuristic_value
from splendor_ai.encoding import collate_canonical_state_batch
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityValueConfig,
    CanonicalEntityValueModel,
)
from splendor_ai.training.replay import deserialize_state_snapshot


PROFILES = ("parent", "shallow", "016", "035")
PHASES = ("0-14", "15-29", "30-44", "45+")


@dataclass(frozen=True, slots=True)
class PublicValueConfig:
    source_dir: Path
    parent_checkpoint: Path
    output_dir: Path
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    score_margin_scale: float = 20.0
    heuristic_scale: float = 4_000.0
    seed: int = 2_029_0716
    bootstrap_samples: int = 20_000
    device: str = "cpu"
    minimum_macro_mae_improvement: float = 0.10
    minimum_ordering_concordance: float = 0.55
    minimum_ordering_improvement: float = 0.03
    maximum_breakdown_mae_regression: float = 0.10


@dataclass(frozen=True, slots=True)
class PublicValueSample:
    state: SplendorState
    target: float
    game_seed: int
    profile: str
    actor_seat: int
    turn_index: int
    phase: str


class PublicValueReplayDataset(Dataset[PublicValueSample]):
    """026 replay rows reduced to public-value fields and decoded lazily."""

    def __init__(self, replay_path: str | Path, score_margin_scale: float = 20.0) -> None:
        if score_margin_scale <= 0.0:
            raise ValueError("score_margin_scale must be positive.")
        self._scale = score_margin_scale
        self._rows: list[dict[str, Any]] = []
        with Path(replay_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                snapshot = row.get("state_snapshot")
                scores = row.get("game_final_scores")
                profile = row.get("rollout_profile")
                actor = int(row.get("player_id", -1))
                if not isinstance(snapshot, dict):
                    raise ValueError("Public value training requires state snapshots.")
                if not isinstance(scores, list) or len(scores) != 2:
                    raise ValueError("Public value training requires two final scores.")
                if profile not in PROFILES:
                    raise ValueError(f"Unexpected rollout profile: {profile!r}")
                if actor not in (0, 1):
                    raise ValueError(f"Unexpected acting seat: {actor}")
                self._rows.append(
                    {
                        "snapshot": snapshot,
                        "scores": (float(scores[0]), float(scores[1])),
                        "game_seed": int(row["game_seed"]),
                        "profile": str(profile),
                        "actor": actor,
                        "turn_index": int(row["turn_index"]),
                    }
                )
        if not self._rows:
            raise ValueError("Public value replay dataset is empty.")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> PublicValueSample:
        row = self._rows[index]
        actor = int(row["actor"])
        scores = row["scores"]
        margin = float(scores[actor]) - float(scores[1 - actor])
        target = max(-1.0, min(1.0, margin / self._scale))
        turn_index = int(row["turn_index"])
        return PublicValueSample(
            state=deserialize_state_snapshot(row["snapshot"]),
            target=target,
            game_seed=int(row["game_seed"]),
            profile=str(row["profile"]),
            actor_seat=actor,
            turn_index=turn_index,
            phase=phase_for_turn(turn_index),
        )

    def game_seed_for_index(self, index: int) -> int:
        return int(self._rows[index]["game_seed"])

    def stratum_for_index(self, index: int) -> tuple[str, int]:
        row = self._rows[index]
        return str(row["profile"]), int(row["actor"])

    def profile_for_seed(self, seed: int) -> str:
        profiles = {
            str(row["profile"])
            for row in self._rows
            if int(row["game_seed"]) == seed
        }
        if len(profiles) != 1:
            raise ValueError(f"Seed {seed} does not map to exactly one profile.")
        return next(iter(profiles))


class BalancedValueStrataDataset(Dataset[PublicValueSample]):
    """Cycle profile/actor strata with exactly equal frequency per epoch."""

    def __init__(self, source: PublicValueReplayDataset, indices: Iterable[int]) -> None:
        self.source = source
        groups: dict[tuple[str, int], list[int]] = {}
        for index in indices:
            groups.setdefault(source.stratum_for_index(index), []).append(index)
        expected = {(profile, seat) for profile in PROFILES for seat in (0, 1)}
        if set(groups) != expected or any(not values for values in groups.values()):
            raise ValueError("Every profile/actor training stratum must be non-empty.")
        self.strata = tuple(sorted(groups))
        self.groups = groups
        self.per_stratum = max(len(values) for values in groups.values())

    def __len__(self) -> int:
        return len(self.strata) * self.per_stratum

    def __getitem__(self, index: int) -> PublicValueSample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        stratum = self.strata[index % len(self.strata)]
        offset = index // len(self.strata)
        values = self.groups[stratum]
        return self.source[values[offset % len(values)]]


def phase_for_turn(turn_index: int) -> str:
    if turn_index < 15:
        return "0-14"
    if turn_index < 30:
        return "15-29"
    if turn_index < 45:
        return "30-44"
    return "45+"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-margin-scale", type=float, default=20.0)
    parser.add_argument("--heuristic-scale", type=float, default=4_000.0)
    parser.add_argument("--seed", type=int, default=2_029_0716)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> PublicValueConfig:
    return PublicValueConfig(
        source_dir=Path(args.source_dir),
        parent_checkpoint=Path(args.parent_checkpoint),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        score_margin_scale=args.score_margin_scale,
        heuristic_scale=args.heuristic_scale,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )


def _collate(samples: list[PublicValueSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty public value batch.")
    batch: dict[str, Any] = collate_canonical_state_batch(
        [(sample.state, sample.actor_seat) for sample in samples]
    )
    batch["target"] = torch.tensor(
        [sample.target for sample in samples],
        dtype=torch.float32,
    )
    batch["samples"] = samples
    return batch


def _indices_for_splits(
    dataset: PublicValueReplayDataset,
    splits: dict[str, list[int]],
) -> dict[str, list[int]]:
    seed_sets = {name: set(values) for name, values in splits.items()}
    indices = {
        name: [
            index
            for index in range(len(dataset))
            if dataset.game_seed_for_index(index) in seeds
        ]
        for name, seeds in seed_sets.items()
    }
    if any(not values for values in indices.values()):
        raise ValueError("Every frozen public-value split must contain positions.")
    return indices


def _validate_splits(
    dataset: PublicValueReplayDataset,
    splits: dict[str, list[int]],
) -> dict[str, Any]:
    sets = {name: set(values) for name, values in splits.items()}
    if set(sets) != {"train", "validation", "confirmation"}:
        raise ValueError("Expected train/validation/confirmation split names.")
    if any(left & right for name, left in sets.items() for other, right in sets.items() if name < other):
        raise ValueError("Frozen game splits overlap.")
    all_seeds = {dataset.game_seed_for_index(index) for index in range(len(dataset))}
    if set().union(*sets.values()) != all_seeds:
        raise ValueError("Frozen game splits do not cover the replay corpus.")
    profile_games: dict[str, dict[str, int]] = {}
    for name, seeds in splits.items():
        counts = {profile: 0 for profile in PROFILES}
        for seed in seeds:
            counts[dataset.profile_for_seed(seed)] += 1
        profile_games[name] = counts
    expected = {
        "train": {profile: 48 for profile in PROFILES},
        "validation": {profile: 6 for profile in PROFILES},
        "confirmation": {profile: 6 for profile in PROFILES},
    }
    if profile_games != expected:
        raise ValueError("026 split profile balance differs from its frozen contract.")
    return {
        "games": {name: len(values) for name, values in splits.items()},
        "profile_games": profile_games,
    }


def _model_config_from_parent(checkpoint: dict[str, Any]) -> CanonicalEntityValueConfig:
    parent = checkpoint.get("model_config", {})
    return CanonicalEntityValueConfig(
        hidden_size=int(parent["entity_hidden_size"]),
        layers=int(parent["entity_layers"]),
        attention_heads=int(parent["attention_heads"]),
        attention_feedforward_size=int(parent["attention_feedforward_size"]),
        id_embedding_size=int(parent["id_embedding_size"]),
        dropout=0.0,
    )


def initialize_from_public_actor_encoder(
    model: CanonicalEntityValueModel,
    parent_state_dict: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    target = model.state_dict()
    prefixes = (
        "entity_type_embedding.",
        "card_embedding.",
        "noble_embedding.",
        "entity_input.",
        "entity_encoder.",
    )
    transferred = {
        name: value
        for name, value in parent_state_dict.items()
        if name.startswith(prefixes)
        and name in target
        and target[name].shape == value.shape
    }
    expected = {name for name in target if name.startswith(prefixes)}
    if set(transferred) != expected:
        missing = sorted(expected - set(transferred))
        raise ValueError(f"Parent encoder is incompatible with public value model: {missing}")
    model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


def _move_tensor_batch(batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device)
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _train_epoch(
    model: CanonicalEntityValueModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: str,
) -> dict[str, float]:
    model.train()
    squared_error = 0.0
    absolute_error = 0.0
    samples = 0
    for batch in dataloader:
        tensors = _move_tensor_batch(batch, device)
        predictions = model(
            entity_type_ids=tensors["entity_type_ids"],
            entity_card_ids=tensors["entity_card_ids"],
            entity_noble_ids=tensors["entity_noble_ids"],
            entity_numeric_features=tensors["entity_numeric_features"],
            entity_mask=tensors["entity_mask"],
        )
        errors = predictions - tensors["target"]
        loss = errors.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        squared_error += float(errors.detach().square().sum().cpu())
        absolute_error += float(errors.detach().abs().sum().cpu())
        samples += int(errors.numel())
    return {
        "mse": squared_error / samples,
        "mae": absolute_error / samples,
        "samples": samples,
    }


def _correlation(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) < 2:
        return 0.0
    prediction_mean = statistics.fmean(predictions)
    target_mean = statistics.fmean(targets)
    left = [value - prediction_mean for value in predictions]
    right = [value - target_mean for value in targets]
    denominator = math.sqrt(
        sum(value * value for value in left)
        * sum(value * value for value in right)
    )
    return (
        sum(a * b for a, b in zip(left, right, strict=True)) / denominator
        if denominator > 0.0
        else 0.0
    )


def _basic_metrics(records: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    predictions = [float(record[key]) for record in records]
    targets = [float(record["target"]) for record in records]
    errors = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
    nonzero = [
        (prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
        if abs(target) > 1e-12
    ]
    return {
        "mse": statistics.fmean(error * error for error in errors),
        "mae": statistics.fmean(abs(error) for error in errors),
        "correlation": _correlation(predictions, targets),
        "sign_accuracy": (
            sum((prediction > 0.0) == (target > 0.0) for prediction, target in nonzero)
            / len(nonzero)
            if nonzero
            else 1.0
        ),
        "samples": len(records),
    }


def _breakdowns(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        labels = (
            f"profile:{record['profile']}",
            f"phase:{record['phase']}",
            f"seat:{record['actor_seat']}",
            f"profile_seat:{record['profile']}:seat{record['actor_seat']}",
        )
        for label in labels:
            groups.setdefault(label, []).append(record)
    return {label: _basic_metrics(values, key) for label, values in sorted(groups.items())}


def _paired_game_error(
    records: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    games: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        games.setdefault(int(record["game_seed"]), []).append(record)
    differences = []
    for rows in games.values():
        candidate = statistics.fmean(abs(float(row["candidate"]) - float(row["target"])) for row in rows)
        heuristic = statistics.fmean(abs(float(row["heuristic"]) - float(row["target"])) for row in rows)
        differences.append(candidate - heuristic)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(bootstrap_samples)
    )
    return {
        "games": len(differences),
        "mean_candidate_minus_heuristic_mae": statistics.fmean(differences),
        "ci95_low": means[int(0.025 * (len(means) - 1))],
        "ci95_high": means[int(0.975 * (len(means) - 1))],
        "candidate_better_games": sum(value < 0.0 for value in differences),
        "candidate_better_fraction": sum(value < 0.0 for value in differences) / len(differences),
    }


def _ordering_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    units: dict[tuple[int, str, int, str], list[dict[str, Any]]] = {}
    for record in records:
        unit = (
            int(record["game_seed"]),
            str(record["profile"]),
            int(record["actor_seat"]),
            str(record["phase"]),
        )
        units.setdefault(unit, []).append(record)
    compact = [
        {
            "game_seed": unit[0],
            "profile": unit[1],
            "actor_seat": unit[2],
            "phase": unit[3],
            "prediction": statistics.fmean(float(row[key]) for row in rows),
            "target": statistics.fmean(float(row["target"]) for row in rows),
        }
        for unit, rows in units.items()
    ]
    strata: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in compact:
        strata.setdefault(
            (str(row["profile"]), int(row["actor_seat"]), str(row["phase"])),
            [],
        ).append(row)
    correct = 0.0
    comparable = 0
    correlations = []
    for rows in strata.values():
        targets = [float(row["target"]) for row in rows]
        predictions = [float(row["prediction"]) for row in rows]
        if len(set(targets)) > 1:
            correlations.append(_correlation(predictions, targets))
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                target_delta = targets[left_index] - targets[right_index]
                if abs(target_delta) <= 1e-12:
                    continue
                prediction_delta = predictions[left_index] - predictions[right_index]
                comparable += 1
                if abs(prediction_delta) <= 1e-12:
                    correct += 0.5
                elif prediction_delta * target_delta > 0.0:
                    correct += 1.0
    return {
        "game_phase_units": len(compact),
        "comparable_pairs": comparable,
        "concordance": correct / comparable if comparable else 0.0,
        "macro_stratum_correlation": statistics.fmean(correlations) if correlations else 0.0,
        "correlation_strata": len(correlations),
    }


def evaluate_value_model(
    model: CanonicalEntityValueModel,
    dataset: PublicValueReplayDataset,
    indices: list[int],
    config: PublicValueConfig,
    *,
    with_bootstrap: bool,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(0, len(indices), config.batch_size):
            samples = [dataset[index] for index in indices[offset : offset + config.batch_size]]
            batch = _collate(samples)
            tensors = _move_tensor_batch(batch, config.device)
            predictions = model(
                entity_type_ids=tensors["entity_type_ids"],
                entity_card_ids=tensors["entity_card_ids"],
                entity_noble_ids=tensors["entity_noble_ids"],
                entity_numeric_features=tensors["entity_numeric_features"],
                entity_mask=tensors["entity_mask"],
            ).cpu().tolist()
            for sample, prediction in zip(samples, predictions, strict=True):
                records.append(
                    {
                        "candidate": float(prediction),
                        "heuristic": legal_heuristic_value(
                            sample.state,
                            player_id=sample.actor_seat,
                            scale=config.heuristic_scale,
                        ),
                        "target": sample.target,
                        "game_seed": sample.game_seed,
                        "profile": sample.profile,
                        "actor_seat": sample.actor_seat,
                        "turn_index": sample.turn_index,
                        "phase": sample.phase,
                    }
                )
    candidate = _basic_metrics(records, "candidate")
    heuristic = _basic_metrics(records, "heuristic")
    candidate_breakdowns = _breakdowns(records, "candidate")
    heuristic_breakdowns = _breakdowns(records, "heuristic")
    profile_seat_labels = [
        f"profile_seat:{profile}:seat{seat}"
        for profile in PROFILES
        for seat in (0, 1)
    ]
    candidate["macro_profile_seat_mae"] = statistics.fmean(
        float(candidate_breakdowns[label]["mae"]) for label in profile_seat_labels
    )
    heuristic["macro_profile_seat_mae"] = statistics.fmean(
        float(heuristic_breakdowns[label]["mae"]) for label in profile_seat_labels
    )
    payload = {
        "candidate": candidate,
        "heuristic": heuristic,
        "candidate_breakdowns": candidate_breakdowns,
        "heuristic_breakdowns": heuristic_breakdowns,
        "candidate_ordering": _ordering_summary(records, "candidate"),
        "heuristic_ordering": _ordering_summary(records, "heuristic"),
    }
    if with_bootstrap:
        payload["paired_game_error"] = _paired_game_error(
            records,
            config.bootstrap_samples,
            seed,
        )
    return payload


def _offline_gates(config: PublicValueConfig, result: dict[str, Any]) -> dict[str, bool]:
    candidate = result["candidate"]
    heuristic = result["heuristic"]
    candidate_ordering = result["candidate_ordering"]
    heuristic_ordering = result["heuristic_ordering"]
    relative_mae = 1.0 - (
        float(candidate["macro_profile_seat_mae"])
        / float(heuristic["macro_profile_seat_mae"])
    )
    comparison_labels = [
        *(f"profile:{profile}" for profile in PROFILES),
        *(f"phase:{phase}" for phase in PHASES),
    ]
    no_breakdown_regression = all(
        float(result["candidate_breakdowns"][label]["mae"])
        <= (1.0 + config.maximum_breakdown_mae_regression)
        * float(result["heuristic_breakdowns"][label]["mae"])
        for label in comparison_labels
    )
    return {
        "macro_mae_improves_at_least_10pct": (
            relative_mae >= config.minimum_macro_mae_improvement
        ),
        "paired_game_mae_ci_below_zero": (
            float(result["paired_game_error"]["ci95_high"]) < 0.0
        ),
        "ordering_concordance_at_least_55pct": (
            float(candidate_ordering["concordance"])
            >= config.minimum_ordering_concordance
        ),
        "ordering_concordance_improves_at_least_3pt": (
            float(candidate_ordering["concordance"])
            - float(heuristic_ordering["concordance"])
            >= config.minimum_ordering_improvement
        ),
        "macro_ordering_correlation_improves": (
            float(candidate_ordering["macro_stratum_correlation"])
            > float(heuristic_ordering["macro_stratum_correlation"])
        ),
        "no_profile_or_phase_mae_regression_over_10pct": no_breakdown_regression,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_and_gate(config: PublicValueConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = config.source_dir / "replays.jsonl"
    split_path = config.source_dir / "game_splits.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    dataset = PublicValueReplayDataset(replay_path, config.score_margin_scale)
    split_integrity = _validate_splits(dataset, splits)
    split_indices = _indices_for_splits(dataset, splits)
    balanced_train = BalancedValueStrataDataset(dataset, split_indices["train"])

    torch.manual_seed(config.seed)
    parent = torch.load(config.parent_checkpoint, map_location="cpu", weights_only=True)
    if parent.get("model_kind") != "canonical_entity_action_scorer":
        raise ValueError("Parent must be a canonical entity action scorer checkpoint.")
    model_config = _model_config_from_parent(parent)
    model = CanonicalEntityValueModel(model_config)
    transferred = initialize_from_public_actor_encoder(model, parent["model_state_dict"])
    model.to(config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        balanced_train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_collate,
    )
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = None
    best_validation_mae = math.inf
    for epoch in range(config.epochs):
        train_metrics = _train_epoch(model, train_loader, optimizer, config.device)
        validation = evaluate_value_model(
            model,
            dataset,
            split_indices["validation"],
            config,
            with_bootstrap=False,
            seed=config.seed + epoch,
        )
        selection = float(validation["candidate"]["macro_profile_seat_mae"])
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation,
                "selection_macro_profile_seat_mae": selection,
            }
        )
        print(
            "[public-value]"
            f" epoch={epoch + 1}/{config.epochs}"
            f" train_mae={train_metrics['mae']:.6f}"
            f" validation_macro_mae={selection:.6f}"
        )
        if selection < best_validation_mae:
            best_validation_mae = selection
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None or best_epoch is None:
        raise RuntimeError("Value training produced no selected checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    validation_selected = evaluate_value_model(
        model,
        dataset,
        split_indices["validation"],
        config,
        with_bootstrap=True,
        seed=config.seed + 100,
    )
    confirmation = evaluate_value_model(
        model,
        dataset,
        split_indices["confirmation"],
        config,
        with_bootstrap=True,
        seed=config.seed + 200,
    )
    gates = _offline_gates(config, confirmation)
    checkpoint_path = config.output_dir / "canonical_entity_public_value_width64_best.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_value",
            "input_contract": "canonical_public_belief_v1",
            "target": "clip((acting_player_final_score-opponent_final_score)/20,-1,1)",
            "model_state_dict": best_state,
            "model_config": asdict(model_config),
            "parent_checkpoint": str(config.parent_checkpoint),
            "parent_checkpoint_sha256": _sha256(config.parent_checkpoint),
            "selected_epoch": best_epoch,
            "train_config": _json_config(config),
        },
        checkpoint_path,
    )
    payload = {
        "experiment": "canonical_public_belief_value_029",
        "config": _json_config(config),
        "source_replay": str(replay_path),
        "source_splits": str(split_path),
        "split_integrity": split_integrity,
        "split_positions": {name: len(values) for name, values in split_indices.items()},
        "balanced_train": {
            "samples_per_epoch": len(balanced_train),
            "positions_per_stratum": balanced_train.per_stratum,
            "strata": [f"{profile}:seat{seat}" for profile, seat in balanced_train.strata],
        },
        "model_config": asdict(model_config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "transferred_encoder_tensors": list(transferred),
        "selected_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "history": history,
        "selected_validation": validation_selected,
        "confirmation": confirmation,
        "offline_gates": gates,
        "match_gate_eligible": all(gates.values()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path = config.output_dir / "public_value_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_epoch": best_epoch,
                "offline_gates": gates,
                "match_gate_eligible": all(gates.values()),
                "metrics": str(metrics_path),
            },
            indent=2,
        )
    )
    return payload


def _json_config(config: PublicValueConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "source_dir": str(config.source_dir),
        "parent_checkpoint": str(config.parent_checkpoint),
        "output_dir": str(config.output_dir),
    }


def main() -> None:
    train_and_gate(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()
