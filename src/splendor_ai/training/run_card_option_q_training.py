"""Train and gate the single from-scratch card-option Q model for 081."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.card_option_q import (
    CardOptionQConfig,
    CardOptionQModel,
    OptionActionPair,
    collate_card_option_batch,
    enumerate_option_action_pairs,
    save_card_option_checkpoint,
)
from splendor_ai.training.replay import deserialize_state_snapshot


@dataclass(frozen=True, slots=True)
class CardOptionTrainingConfig:
    corpus_path: Path
    output_dir: Path
    epochs: int = 25
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    ranking_weight: float = 0.5
    minimum_ranking_gap: float = 0.10
    max_grad_norm: float = 1.0
    seed: int = 781_081
    bootstrap_samples: int = 20_000
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class CardOptionSample:
    state: Any
    pairs: tuple[OptionActionPair, ...]
    target_values: torch.Tensor
    baseline_index: int


class CardOptionDataset(Dataset[CardOptionSample]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.env = SplendorEnv()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CardOptionSample:
        row = self.rows[index]
        state = deserialize_state_snapshot(row["state_snapshot"])
        pairs = enumerate_option_action_pairs(state, self.env.legal_actions(state))
        stored = row["pairs"]
        if len(pairs) != len(stored):
            raise ValueError("Stored and reconstructed option-pair counts differ.")
        baseline = []
        values = []
        for pair, payload in zip(pairs, stored, strict=True):
            if pair.target.key != payload["target_key"] or pair.action_index != int(payload["action_index"]):
                raise ValueError("Stored option-pair order does not match reconstruction.")
            values.append(float(payload["full_mean"]))
            if payload["baseline"]:
                baseline.append(len(values) - 1)
        if len(baseline) != 1:
            raise ValueError("Each state must contain exactly one raw-035 CLEAR baseline pair.")
        return CardOptionSample(
            state=state,
            pairs=tuple(pairs),
            target_values=torch.tensor(values, dtype=torch.float32),
            baseline_index=baseline[0],
        )


def collate_card_option_samples(samples: list[CardOptionSample]) -> dict[str, Any]:
    batch = collate_card_option_batch(
        [(sample.state, list(sample.pairs)) for sample in samples]
    )
    width = int(batch["action_mask"].shape[1])
    targets = torch.zeros((len(samples), width), dtype=torch.float32)
    baseline = torch.zeros(len(samples), dtype=torch.long)
    pair_objects: list[tuple[OptionActionPair, ...]] = []
    for row, sample in enumerate(samples):
        targets[row, : len(sample.pairs)] = sample.target_values
        baseline[row] = sample.baseline_index
        pair_objects.append(sample.pairs)
    batch["target_values"] = targets
    batch["baseline_indices"] = baseline
    batch["pair_objects"] = pair_objects
    return batch


def train_card_option_q(config: CardOptionTrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    corpus = json.loads(config.corpus_path.read_text(encoding="utf-8"))
    if not corpus.get("all_gates_passed", False):
        raise ValueError("081 corpus did not pass every frozen gate.")
    rows_by_split = {
        split: [row for row in corpus["states"] if row["split"] == split]
        for split in ("train", "validation", "confirmation")
    }
    if {key: len(value) for key, value in rows_by_split.items()} != {
        "train": 600, "validation": 200, "confirmation": 200
    }:
        raise ValueError("081 requires exact 600/200/200 state splits.")
    datasets = {key: CardOptionDataset(value) for key, value in rows_by_split.items()}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            collate_fn=collate_card_option_samples,
            generator=torch.Generator().manual_seed(config.seed),
        )
        for split, dataset in datasets.items()
    }
    torch.manual_seed(config.seed)
    model = CardOptionQModel(CardOptionQConfig()).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    best_epoch = None
    best_regret = math.inf
    best_state = None
    for epoch in range(1, config.epochs + 1):
        training = _train_epoch(model, loaders["train"], optimizer, config)
        validation = evaluate_card_option_q(model, loaders["validation"], config.device, 0, config.seed + epoch)
        history.append({"epoch": epoch, "training": training, "validation": validation})
        if validation["model_mean_regret"] < best_regret:
            best_regret = validation["model_mean_regret"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            f"[081:train] epoch={epoch}/{config.epochs} loss={training['total_loss']:.5f} "
            f"val_uplift={validation['selected_minus_baseline_mean']:.4f} "
            f"val_regret={validation['model_mean_regret']:.4f}",
            flush=True,
        )
    if best_state is None or best_epoch is None:
        raise RuntimeError("No best 081 epoch was selected.")
    model.load_state_dict(best_state)
    validation = evaluate_card_option_q(
        model, loaders["validation"], config.device, config.bootstrap_samples, config.seed + 10_001
    )
    confirmation = evaluate_card_option_q(
        model, loaders["confirmation"], config.device, config.bootstrap_samples, config.seed + 20_001
    )
    gates = {
        "validation_uplift_positive": validation["selected_minus_baseline_mean"] > 0.0,
        "confirmation_uplift_at_least_005": confirmation["selected_minus_baseline_mean"] >= 0.05,
        "confirmation_uplift_lower95_positive": confirmation["selected_minus_baseline_ci95"][0] > 0.0,
        "confirmation_regret_ratio_at_most_080": confirmation["regret_ratio"] <= 0.80,
        "confirmation_action_movement_at_least_015": confirmation["action_movement"] >= 0.15,
        "confirmation_nonclear_target_at_least_020": confirmation["nonclear_target_rate"] >= 0.20,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "card_option_q_081.pt"
    save_card_option_checkpoint(
        checkpoint,
        model,
        {
            "training_seed": config.seed,
            "best_epoch": best_epoch,
            "corpus_path": str(config.corpus_path),
            "human_data_used": False,
        },
    )
    result = {
        "training": "card_option_q_controller_081_v1",
        "configuration": _config_payload(config),
        "model_configuration": asdict(model.config),
        "split_counts": {key: len(value) for key, value in rows_by_split.items()},
        "best_epoch": best_epoch,
        "history": history,
        "validation": validation,
        "confirmation": confirmation,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "checkpoint_path": str(checkpoint),
        "human_data_used": False,
    }
    (config.output_dir / "training_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _train_epoch(
    model: CardOptionQModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    config: CardOptionTrainingConfig,
) -> dict[str, float]:
    model.train()
    totals = {"mse_loss": 0.0, "ranking_loss": 0.0, "total_loss": 0.0}
    states = 0
    for batch in loader:
        pair_objects = batch.pop("pair_objects")
        del pair_objects
        batch = {key: value.to(config.device) for key, value in batch.items()}
        scores = model(**batch)
        mask = batch["action_mask"]
        targets = batch["target_values"]
        mse = ((scores.masked_fill(~mask, 0.0) - targets) ** 2)[mask].mean()
        ranking_terms = []
        for row in range(scores.shape[0]):
            valid = mask[row]
            labels = targets[row, valid]
            predictions = scores[row, valid]
            best_row = int(torch.argmax(labels).item())
            worse = labels <= labels[best_row] - config.minimum_ranking_gap
            if bool(worse.any()):
                ranking_terms.append(torch.nn.functional.softplus(-(predictions[best_row] - predictions[worse])).mean())
        ranking = torch.stack(ranking_terms).mean() if ranking_terms else mse * 0.0
        total = mse + config.ranking_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        count = int(scores.shape[0])
        totals["mse_loss"] += float(mse.detach().cpu()) * count
        totals["ranking_loss"] += float(ranking.detach().cpu()) * count
        totals["total_loss"] += float(total.detach().cpu()) * count
        states += count
    return {key: value / states for key, value in totals.items()}


def evaluate_card_option_q(
    model: CardOptionQModel,
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
    with torch.no_grad():
        for batch in loader:
            pair_objects = batch.pop("pair_objects")
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = model(**batch)
            for row, pairs in enumerate(pair_objects):
                count = len(pairs)
                labels = batch["target_values"][row, :count].cpu()
                selected = int(torch.argmax(scores[row, :count]).item())
                baseline = int(batch["baseline_indices"][row].item())
                best = float(labels.max().item())
                selected_value = float(labels[selected].item())
                baseline_value = float(labels[baseline].item())
                deltas.append(selected_value - baseline_value)
                model_regrets.append(best - selected_value)
                baseline_regrets.append(best - baseline_value)
                moved.append(float(_strategic_key(pairs[selected]) != _strategic_key(pairs[baseline])))
                nonclear.append(float(pairs[selected].target.key != "CLEAR"))
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
        "regret_ratio": statistics.fmean(model_regrets) / baseline_mean_regret if baseline_mean_regret else 0.0,
        "action_movement": statistics.fmean(moved),
        "nonclear_target_rate": statistics.fmean(nonclear),
    }


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def _strategic_key(pair: OptionActionPair) -> tuple[Any, ...]:
    action = pair.action
    return (
        action.action_type.name,
        action.tier,
        action.market_index,
        action.reserved_index,
        tuple(sorted(action.take_tokens)),
        tuple(sorted(action.return_tokens)),
        action.noble_id,
    )


def _config_payload(config: CardOptionTrainingConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _validate_config(config: CardOptionTrainingConfig) -> None:
    if (config.epochs, config.batch_size, config.seed) != (25, 8, 781_081):
        raise ValueError("081 training epochs, batch size, and seed are frozen.")
    if (config.learning_rate, config.weight_decay, config.ranking_weight, config.minimum_ranking_gap) != (1e-4, 1e-4, 0.5, 0.10):
        raise ValueError("081 optimizer and loss contract is frozen.")
