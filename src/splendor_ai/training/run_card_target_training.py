"""Extract historical rows, train, and gate experiment 082's target selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.card_target_executor import target_executor_action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.card_option_q import OptionTarget, enumerate_option_targets
from splendor_ai.training.card_target_value import (
    CardTargetValueConfig,
    CardTargetValueModel,
    collate_card_target_batch,
    save_card_target_checkpoint,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


HISTORICAL_CORPUS_SHA256 = (
    "f368b462b71bf02154ed00ce524606a8142e238ad55b0d2689026d9d8abb373e"
)


@dataclass(frozen=True, slots=True)
class CardTargetTrainingConfig:
    historical_corpus_path: Path
    fresh_corpus_path: Path
    checkpoint_035: Path
    output_dir: Path
    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    listwise_temperature: float = 0.10
    centered_mse_weight: float = 0.25
    max_grad_norm: float = 1.0
    seed: int = 782_082
    bootstrap_samples: int = 20_000
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class CardTargetSample:
    state: Any
    targets: tuple[OptionTarget, ...]
    target_values: torch.Tensor
    executor_action_indices: torch.Tensor
    baseline_index: int


class CardTargetDataset(Dataset[CardTargetSample]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CardTargetSample:
        row = self.rows[index]
        state = deserialize_state_snapshot(row["state_snapshot"])
        targets = enumerate_option_targets(state, state.current_player)
        stored = row["targets"]
        if len(targets) != len(stored):
            raise ValueError("Stored and reconstructed target counts differ.")
        values = []
        action_indices = []
        baseline = []
        for target_index, (target, payload) in enumerate(zip(targets, stored, strict=True)):
            if target.key != payload["target_key"] or target.source != payload["target_source"]:
                raise ValueError("Stored target order does not match reconstruction.")
            values.append(float(payload["full_mean"]))
            action_indices.append(int(payload["executor_action_index"]))
            if payload["baseline"]:
                baseline.append(target_index)
        if baseline != [0] or targets[0].key != "CLEAR":
            raise ValueError("Each state must have CLEAR as its unique first baseline.")
        return CardTargetSample(
            state=state,
            targets=tuple(targets),
            target_values=torch.tensor(values, dtype=torch.float32),
            executor_action_indices=torch.tensor(action_indices, dtype=torch.long),
            baseline_index=0,
        )


def collate_card_target_samples(samples: list[CardTargetSample]) -> dict[str, Any]:
    batch = collate_card_target_batch(
        [(sample.state, list(sample.targets)) for sample in samples]
    )
    width = int(batch["target_mask"].shape[1])
    values = torch.zeros((len(samples), width), dtype=torch.float32)
    actions = torch.zeros((len(samples), width), dtype=torch.long)
    baseline = torch.zeros(len(samples), dtype=torch.long)
    target_objects: list[tuple[OptionTarget, ...]] = []
    for row_index, sample in enumerate(samples):
        count = len(sample.targets)
        values[row_index, :count] = sample.target_values
        actions[row_index, :count] = sample.executor_action_indices
        baseline[row_index] = sample.baseline_index
        target_objects.append(sample.targets)
    batch.update(
        {
            "target_values": values,
            "executor_action_indices": actions,
            "baseline_indices": baseline,
            "target_objects": target_objects,
        }
    )
    return batch


def train_card_target_value(config: CardTargetTrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    historical_rows, historical_summary = extract_historical_target_rows(config)
    fresh = json.loads(config.fresh_corpus_path.read_text(encoding="utf-8"))
    if not fresh.get("all_gates_passed", False):
        raise ValueError("082 fresh corpus did not pass every frozen gate.")
    fresh_by_split = {
        split: [row for row in fresh["states"] if row["split"] == split]
        for split in ("validation", "confirmation")
    }
    if {key: len(value) for key, value in fresh_by_split.items()} != {
        "validation": 200,
        "confirmation": 200,
    }:
        raise ValueError("082 requires exact fresh 200/200 validation/confirmation splits.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    historical_path = config.output_dir / "historical_target_rows.json"
    historical_payload = {
        "corpus": "card_target_historical_training_082_v1",
        "source_path": str(config.historical_corpus_path),
        "source_sha256": HISTORICAL_CORPUS_SHA256,
        "states": historical_rows,
        "summary": historical_summary,
        "human_data_used": False,
    }
    historical_path.write_text(
        json.dumps(historical_payload, indent=2) + "\n", encoding="utf-8"
    )
    datasets = {
        "train": CardTargetDataset(historical_rows),
        "validation": CardTargetDataset(fresh_by_split["validation"]),
        "confirmation": CardTargetDataset(fresh_by_split["confirmation"]),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            collate_fn=collate_card_target_samples,
            generator=torch.Generator().manual_seed(config.seed),
        )
        for split, dataset in datasets.items()
    }
    torch.manual_seed(config.seed)
    model = CardTargetValueModel(CardTargetValueConfig()).to(config.device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    best_epoch = None
    best_regret = math.inf
    best_state = None
    for epoch in range(1, config.epochs + 1):
        training = _train_epoch(model, loaders["train"], optimizer, config)
        validation = evaluate_card_target_value(
            model, loaders["validation"], config.device, 0, config.seed + epoch
        )
        history.append({"epoch": epoch, "training": training, "validation": validation})
        if validation["model_mean_regret"] < best_regret:
            best_regret = validation["model_mean_regret"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"[082:train] epoch={epoch}/{config.epochs} "
            f"loss={training['total_loss']:.5f} "
            f"val_uplift={validation['selected_minus_baseline_mean']:.4f} "
            f"val_regret={validation['model_mean_regret']:.4f}",
            flush=True,
        )
    if best_state is None or best_epoch is None:
        raise RuntimeError("No best 082 epoch was selected.")
    model.load_state_dict(best_state)
    validation = evaluate_card_target_value(
        model,
        loaders["validation"],
        config.device,
        config.bootstrap_samples,
        config.seed + 10_001,
    )
    confirmation = evaluate_card_target_value(
        model,
        loaders["confirmation"],
        config.device,
        config.bootstrap_samples,
        config.seed + 20_001,
    )
    gates = {
        "validation_uplift_positive": validation["selected_minus_baseline_mean"] > 0.0,
        "confirmation_uplift_at_least_005": (
            confirmation["selected_minus_baseline_mean"] >= 0.05
        ),
        "confirmation_uplift_lower95_positive": (
            confirmation["selected_minus_baseline_ci95"][0] > 0.0
        ),
        "confirmation_regret_ratio_at_most_080": confirmation["regret_ratio"] <= 0.80,
        "confirmation_action_movement_at_least_015": (
            confirmation["action_movement"] >= 0.15
        ),
        "confirmation_nonclear_target_at_least_020": (
            confirmation["nonclear_target_rate"] >= 0.20
        ),
    }
    checkpoint = config.output_dir / "card_target_value_082.pt"
    save_card_target_checkpoint(
        checkpoint,
        model,
        {
            "training_seed": config.seed,
            "best_epoch": best_epoch,
            "historical_corpus_sha256": HISTORICAL_CORPUS_SHA256,
            "fresh_corpus_path": str(config.fresh_corpus_path),
            "human_data_used": False,
        },
    )
    result = {
        "training": "card_target_value_selector_082_v1",
        "configuration": _config_payload(config),
        "model_configuration": asdict(model.config),
        "split_counts": {key: len(value) for key, value in datasets.items()},
        "historical": {
            **historical_summary,
            "path": str(historical_path),
            "sha256": _sha(historical_path),
        },
        "best_epoch": best_epoch,
        "history": history,
        "validation": validation,
        "confirmation": confirmation,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "checkpoint_path": str(checkpoint),
        "human_data_used": False,
    }
    (config.output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def extract_historical_target_rows(
    config: CardTargetTrainingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha = _sha(config.historical_corpus_path)
    if actual_sha != HISTORICAL_CORPUS_SHA256:
        raise ValueError("Historical 081 corpus hash does not match the frozen 082 contract.")
    corpus = json.loads(config.historical_corpus_path.read_text(encoding="utf-8"))
    if len(corpus.get("states", [])) != 1_000 or not corpus.get("all_gates_passed"):
        raise ValueError("Historical 081 corpus is incomplete or failed its gates.")
    scorer = _LegalActionScorer(
        config.checkpoint_035, config.device, "legal_action_attention_scorer"
    )
    env = SplendorEnv()
    codec = ActionCodec()
    extracted = []
    total_targets = 0
    for row in corpus["states"]:
        state = deserialize_state_snapshot(row["state_snapshot"])
        actions = list(
            actions_by_canonical_successor_key(state, env.legal_actions(state)).values()
        )
        scores = scorer.score_actions(state, actions)
        stored: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for pair in row["pairs"]:
            stored.setdefault(
                (str(pair["target_key"]), int(pair["action_index"])), []
            ).append(pair)
        targets = enumerate_option_targets(state, state.current_player)
        target_rows = []
        for target in targets:
            action = (
                scores[0].action
                if target.card is None
                else target_executor_action(state, actions, scores, target.card)
            )
            action_index = codec.encode(state, action)
            matches = stored.get((target.key, action_index), [])
            if len(matches) != 1:
                raise ValueError("Historical target/executor action did not match uniquely.")
            pair = matches[0]
            if int(pair["total_continuations"]) != 8:
                raise ValueError("Historical target label lacks exactly eight continuations.")
            if target.card is None and not bool(pair["baseline"]):
                raise ValueError("Historical CLEAR executor action is not the stored baseline.")
            target_rows.append(
                {
                    "target_key": target.key,
                    "target_source": target.source,
                    "executor_action_index": action_index,
                    "first_half_mean": float(pair["first_half_mean"]),
                    "second_half_mean": float(pair["second_half_mean"]),
                    "full_mean": float(pair["full_mean"]),
                    "terminal_continuations": int(pair["terminal_continuations"]),
                    "total_continuations": 8,
                    "baseline": target.card is None,
                }
            )
        total_targets += len(target_rows)
        extracted.append(
            {
                "source_game_seed": row["source_game_seed"],
                "state_snapshot": row["state_snapshot"],
                "target_count": len(target_rows),
                "targets": target_rows,
            }
        )
    return extracted, {
        "states": len(extracted),
        "targets": total_targets,
        "continuations": total_targets * 8,
        "exact_unique_matches": True,
        "source_sha256": actual_sha,
        "old_splits_used_as_training_only": True,
    }


def _train_epoch(
    model: CardTargetValueModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    config: CardTargetTrainingConfig,
) -> dict[str, float]:
    model.train()
    totals = {"listwise_loss": 0.0, "centered_mse_loss": 0.0, "total_loss": 0.0}
    states = 0
    for batch in loader:
        batch.pop("target_objects")
        batch = {key: value.to(config.device) for key, value in batch.items()}
        scores = model(**batch)
        listwise_terms = []
        mse_terms = []
        for row_index in range(scores.shape[0]):
            valid = batch["target_mask"][row_index]
            labels = batch["target_values"][row_index, valid]
            predictions = scores[row_index, valid]
            centered_labels = labels - labels.mean()
            centered_predictions = predictions - predictions.mean()
            target_distribution = torch.softmax(
                centered_labels / config.listwise_temperature, dim=0
            )
            listwise_terms.append(
                -(target_distribution * torch.log_softmax(predictions, dim=0)).sum()
            )
            mse_terms.append(torch.mean((centered_predictions - centered_labels) ** 2))
        listwise = torch.stack(listwise_terms).mean()
        centered_mse = torch.stack(mse_terms).mean()
        total = listwise + config.centered_mse_weight * centered_mse
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        count = int(scores.shape[0])
        totals["listwise_loss"] += float(listwise.detach().cpu()) * count
        totals["centered_mse_loss"] += float(centered_mse.detach().cpu()) * count
        totals["total_loss"] += float(total.detach().cpu()) * count
        states += count
    return {key: value / states for key, value in totals.items()}


def evaluate_card_target_value(
    model: CardTargetValueModel,
    loader: DataLoader[Any],
    device: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    deltas: list[float] = []
    model_regrets: list[float] = []
    baseline_regrets: list[float] = []
    moved: list[float] = []
    nonclear: list[float] = []
    selected_sources: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            target_objects = batch.pop("target_objects")
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = model(**batch)
            for row_index, targets in enumerate(target_objects):
                count = len(targets)
                labels = batch["target_values"][row_index, :count].cpu()
                selected = int(torch.argmax(scores[row_index, :count]).item())
                baseline = int(batch["baseline_indices"][row_index].item())
                best = float(labels.max().item())
                selected_value = float(labels[selected].item())
                baseline_value = float(labels[baseline].item())
                deltas.append(selected_value - baseline_value)
                model_regrets.append(best - selected_value)
                baseline_regrets.append(best - baseline_value)
                selected_action = int(
                    batch["executor_action_indices"][row_index, selected].item()
                )
                baseline_action = int(
                    batch["executor_action_indices"][row_index, baseline].item()
                )
                moved.append(float(selected_action != baseline_action))
                nonclear.append(float(targets[selected].key != "CLEAR"))
                source = targets[selected].source
                selected_sources[source] = selected_sources.get(source, 0) + 1
    baseline_mean_regret = statistics.fmean(baseline_regrets)
    return {
        "states": len(deltas),
        "selected_minus_baseline_mean": statistics.fmean(deltas),
        "selected_minus_baseline_ci95": (
            _bootstrap_ci(deltas, bootstrap_samples, seed)
            if bootstrap_samples
            else [statistics.fmean(deltas), statistics.fmean(deltas)]
        ),
        "model_mean_regret": statistics.fmean(model_regrets),
        "baseline_mean_regret": baseline_mean_regret,
        "regret_ratio": (
            statistics.fmean(model_regrets) / baseline_mean_regret
            if baseline_mean_regret
            else 0.0
        ),
        "action_movement": statistics.fmean(moved),
        "nonclear_target_rate": statistics.fmean(nonclear),
        "selected_target_sources": selected_sources,
    }


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_payload(config: CardTargetTrainingConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _validate_config(config: CardTargetTrainingConfig) -> None:
    if (config.epochs, config.batch_size, config.seed) != (25, 16, 782_082):
        raise ValueError("082 training epochs, batch size, and seed are frozen.")
    if (
        config.learning_rate,
        config.weight_decay,
        config.listwise_temperature,
        config.centered_mse_weight,
        config.max_grad_norm,
    ) != (1e-4, 1e-4, 0.10, 0.25, 1.0):
        raise ValueError("082 optimizer and listwise-loss contract is frozen.")
