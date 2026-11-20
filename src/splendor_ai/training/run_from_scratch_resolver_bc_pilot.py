"""Frozen from-scratch resolver behaviour-cloning pilot (experiment 075)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from splendor_ai.encoding import (
    ActionCodec,
    actions_by_canonical_successor_key,
    canonical_successor_action_key,
    collate_entity_action_batch,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    masked_legal_action_scores,
)
from splendor_ai.training.replay import deserialize_state_snapshot


EXPERIMENT_SEED = 75_075
SOURCE_SHA256 = {
    "result": "60458c1bea4328b64f33ad8238529d074a25d1034fa2845e0b339deee3827153",
    "train": "6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5",
    "validation": "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052",
    "confirmation": "8625d9049957ce266fa544896f0262cb37ccea6d168ed0c60a847ef5325cce8f",
    "parent_035": "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd",
}
EXPECTED_ROWS = {"train": 4_397, "validation": 1_135, "confirmation": 1_093}
EXPECTED_GAMES = {"train": 80, "validation": 20, "confirmation": 20}
TRAIN_GAME_PREFIXES = (20, 40, 80)


@dataclass(frozen=True, slots=True)
class ResolverBCConfig:
    source_dir: Path = Path("outputs/resolver_selfplay_distillation_052")
    parent_checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    output_dir: Path = Path("outputs/from_scratch_resolver_bc_pilot_075")
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    entropy_weight: float = 0.01
    bootstrap_samples: int = 20_000
    seed: int = EXPERIMENT_SEED
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ResolverBCSample:
    state: SplendorState
    legal_actions: tuple[Action, ...]
    target_action_row: int
    parent_action_row: int
    game_seed: int
    phase: str
    seat: int
    outcome: str


class ResolverBCDataset(Dataset[ResolverBCSample]):
    def __init__(self, samples: Sequence[ResolverBCSample]) -> None:
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ResolverBCSample:
        return self.samples[index]


def model_config_075() -> CanonicalEntityActionScorerConfig:
    return CanonicalEntityActionScorerConfig(
        entity_hidden_size=64,
        action_hidden_size=64,
        scorer_hidden_size=96,
        entity_layers=1,
        action_layers=1,
        attention_heads=4,
        attention_feedforward_size=128,
        id_embedding_size=16,
        dropout=0.0,
    )


def normalized_masked_entropy(
    scores: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """Per-row entropy divided by log legal count; one-action rows are zero."""

    masked = masked_legal_action_scores(scores, action_mask)
    probabilities = torch.softmax(masked, dim=1).masked_fill(~action_mask, 0.0)
    log_probabilities = torch.log_softmax(masked, dim=1).masked_fill(~action_mask, 0.0)
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    legal_count = action_mask.sum(dim=1)
    denominator = torch.log(legal_count.clamp_min(2).to(scores.dtype))
    return torch.where(legal_count > 1, entropy / denominator, torch.zeros_like(entropy))


def shuffled_train_game_prefixes(
    game_seeds: Iterable[int], seed: int = EXPERIMENT_SEED
) -> dict[int, tuple[int, ...]]:
    games = sorted(set(int(value) for value in game_seeds))
    if len(games) != 80:
        raise ValueError(f"075 requires exactly 80 training games, observed {len(games)}.")
    random.Random(seed).shuffle(games)
    return {count: tuple(games[:count]) for count in TRAIN_GAME_PREFIXES}


def collate_resolver_bc_samples(
    samples: list[ResolverBCSample],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty resolver-BC batch.")
    batch = collate_entity_action_batch(
        [(sample.state, list(sample.legal_actions)) for sample in samples]
    )
    batch.update(
        {
            "target_action_row": torch.tensor(
                [sample.target_action_row for sample in samples], dtype=torch.long
            ),
            "parent_action_row": torch.tensor(
                [sample.parent_action_row for sample in samples], dtype=torch.long
            ),
            "sample_index": torch.tensor(
                [sample.game_seed for sample in samples], dtype=torch.long
            ),
        }
    )
    return batch


def _forward(
    model: CanonicalEntityActionScorerModel, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    scores, _ = model(
        entity_type_ids=batch["entity_type_ids"],
        entity_card_ids=batch["entity_card_ids"],
        entity_noble_ids=batch["entity_noble_ids"],
        entity_numeric_features=batch["entity_numeric_features"],
        entity_mask=batch["entity_mask"],
        action_type_ids=batch["action_type_ids"],
        action_card_ids=batch["action_card_ids"],
        action_noble_ids=batch["action_noble_ids"],
        action_numeric_features=batch["action_numeric_features"],
        action_mask=batch["action_mask"],
    )
    return scores


def train_final_epoch_model(
    dataset: ResolverBCDataset,
    indices: Sequence[int],
    config: ResolverBCConfig,
) -> tuple[CanonicalEntityActionScorerModel, list[dict[str, float | int]]]:
    torch.manual_seed(config.seed)
    model = CanonicalEntityActionScorerModel(model_config_075()).to(config.device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    dataloader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_resolver_bc_samples,
        num_workers=0,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"ce": 0.0, "entropy": 0.0, "loss": 0.0, "correct": 0, "rows": 0}
        for raw_batch in dataloader:
            batch = {name: value.to(config.device) for name, value in raw_batch.items()}
            scores = _forward(model, batch)
            masked = masked_legal_action_scores(scores, batch["action_mask"])
            cross_entropy = F.cross_entropy(masked, batch["target_action_row"])
            entropy = normalized_masked_entropy(scores, batch["action_mask"]).mean()
            loss = cross_entropy - config.entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = int(batch["target_action_row"].numel())
            predictions = masked.argmax(dim=1)
            totals["ce"] += float(cross_entropy.detach().cpu()) * rows
            totals["entropy"] += float(entropy.detach().cpu()) * rows
            totals["loss"] += float(loss.detach().cpu()) * rows
            totals["correct"] += int(
                (predictions == batch["target_action_row"]).sum().item()
            )
            totals["rows"] += rows
        count = int(totals["rows"])
        if count == 0:
            raise ValueError("Training prefix contains no rows.")
        history.append(
            {
                "epoch": epoch,
                "rows": count,
                "cross_entropy": float(totals["ce"]) / count,
                "normalized_entropy": float(totals["entropy"]) / count,
                "total_loss": float(totals["loss"]) / count,
                "accuracy": int(totals["correct"]) / count,
            }
        )
    return model.eval(), history


def evaluate_model(
    model: CanonicalEntityActionScorerModel,
    dataset: ResolverBCDataset,
    config: ResolverBCConfig,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_resolver_bc_samples,
        num_workers=0,
    )
    records: list[dict[str, Any]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = {name: value.to(config.device) for name, value in raw_batch.items()}
            scores = _forward(model, batch)
            masked = masked_legal_action_scores(scores, batch["action_mask"])
            nll = F.cross_entropy(
                masked, batch["target_action_row"], reduction="none"
            )
            entropy = normalized_masked_entropy(scores, batch["action_mask"])
            predictions = masked.argmax(dim=1)
            for row_index in range(len(nll)):
                sample = dataset[offset + row_index]
                candidate = int(predictions[row_index].item())
                records.append(
                    {
                        "game_seed": sample.game_seed,
                        "phase": sample.phase,
                        "seat": sample.seat,
                        "outcome": sample.outcome,
                        "candidate_correct": candidate == sample.target_action_row,
                        "parent_correct": sample.parent_action_row
                        == sample.target_action_row,
                        "moved_from_parent": candidate != sample.parent_action_row,
                        "nll": float(nll[row_index].item()),
                        "normalized_entropy": float(entropy[row_index].item()),
                    }
                )
            offset += len(nll)
    if offset != len(dataset):
        raise RuntimeError("Evaluation row accounting mismatch.")
    return summarize_records(records)


def _metric_block(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty record group.")
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_game[int(record["game_seed"])].append(record)
    candidate_game = [
        statistics.fmean(float(row["candidate_correct"]) for row in rows)
        for rows in by_game.values()
    ]
    parent_game = [
        statistics.fmean(float(row["parent_correct"]) for row in rows)
        for rows in by_game.values()
    ]
    return {
        "rows": len(records),
        "games": len(by_game),
        "agreement": statistics.fmean(
            float(row["candidate_correct"]) for row in records
        ),
        "game_weighted_agreement": statistics.fmean(candidate_game),
        "parent_agreement": statistics.fmean(
            float(row["parent_correct"]) for row in records
        ),
        "parent_game_weighted_agreement": statistics.fmean(parent_game),
        "game_weighted_agreement_delta": statistics.fmean(
            candidate - parent
            for candidate, parent in zip(candidate_game, parent_game, strict=True)
        ),
        "nll": statistics.fmean(float(row["nll"]) for row in records),
        "normalized_entropy": statistics.fmean(
            float(row["normalized_entropy"]) for row in records
        ),
        "movement_from_parent": statistics.fmean(
            float(row["moved_from_parent"]) for row in records
        ),
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "phase": ("early", "mid", "late"),
        "seat": (0, 1),
        "outcome": ("winner", "loser"),
    }
    summary: dict[str, Any] = {"overall": _metric_block(records)}
    for name, values in dimensions.items():
        groups = {}
        for value in values:
            selected = [record for record in records if record[name] == value]
            if selected:
                groups[str(value)] = _metric_block(selected)
        summary[f"by_{name}"] = groups
    return summary


def paired_game_bootstrap(
    candidate_summary: dict[str, Any],
    dataset: ResolverBCDataset,
    model: CanonicalEntityActionScorerModel,
    config: ResolverBCConfig,
) -> dict[str, Any]:
    # Re-evaluate lightweight correctness records to preserve exact per-game pairing.
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_resolver_bc_samples,
        num_workers=0,
    )
    by_game: dict[int, list[float]] = defaultdict(list)
    offset = 0
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = {name: value.to(config.device) for name, value in raw_batch.items()}
            predictions = masked_legal_action_scores(
                _forward(model, batch), batch["action_mask"]
            ).argmax(dim=1)
            for row_index in range(len(predictions)):
                sample = dataset[offset + row_index]
                delta = float(int(predictions[row_index]) == sample.target_action_row) - float(
                    sample.parent_action_row == sample.target_action_row
                )
                by_game[sample.game_seed].append(delta)
            offset += len(predictions)
    differences = [statistics.fmean(by_game[seed]) for seed in sorted(by_game)]
    if len(differences) != 40:
        raise ValueError(f"Primary endpoint requires 40 games, observed {len(differences)}.")
    estimate = statistics.fmean(differences)
    rng = random.Random(config.seed)
    draws = sorted(
        statistics.fmean(
            differences[rng.randrange(len(differences))] for _ in differences
        )
        for _ in range(config.bootstrap_samples)
    )
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    if not math.isclose(
        estimate,
        float(candidate_summary["overall"]["game_weighted_agreement_delta"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Primary endpoint disagrees with evaluation summary.")
    return {
        "games": len(differences),
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.seed,
        "candidate_minus_parent": estimate,
        "ci95_low": low,
        "ci95_high": high,
    }


def run_experiment(config: ResolverBCConfig = ResolverBCConfig()) -> dict[str, Any]:
    _validate_frozen_config(config)
    identity = _validate_source_identity(config)
    source_samples, source_integrity = _load_source_samples(config)
    train_dataset = ResolverBCDataset(source_samples["train"])
    evaluation_dataset = ResolverBCDataset(
        source_samples["validation"] + source_samples["confirmation"]
    )
    prefixes = shuffled_train_game_prefixes(
        sample.game_seed for sample in train_dataset.samples
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fractions: dict[str, Any] = {}
    full_model: CanonicalEntityActionScorerModel | None = None
    full_reload_deterministic = False
    finite_training = True
    for game_count in TRAIN_GAME_PREFIXES:
        label = f"{game_count}_games"
        selected_games = set(prefixes[game_count])
        indices = [
            index
            for index, sample in enumerate(train_dataset.samples)
            if sample.game_seed in selected_games
        ]
        print(
            f"[075:{label}] training rows={len(indices)} epochs={config.epochs}",
            flush=True,
        )
        started = time.perf_counter()
        model, history = train_final_epoch_model(train_dataset, indices, config)
        finite_training &= all(
            math.isfinite(float(value))
            for epoch in history
            for key, value in epoch.items()
            if key not in {"epoch", "rows"}
        )
        checkpoint_path = config.output_dir / f"resolver_bc_{label}_final.pt"
        torch.save(
            {
                "model_kind": "canonical_entity_action_scorer",
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "model_config": asdict(model_config_075()),
                "train_config": {
                    "experiment": "075",
                    "games": game_count,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "entropy_weight": config.entropy_weight,
                    "seed": config.seed,
                },
            },
            checkpoint_path,
        )
        evaluation = evaluate_model(model, evaluation_dataset, config)
        fractions[label] = {
            "training_games": game_count,
            "training_rows": len(indices),
            "training_game_seeds": list(prefixes[game_count]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds": time.perf_counter() - started,
            "train_history": history,
            "evaluation": evaluation,
        }
        if game_count == 80:
            full_model = model
            checkpoint = torch.load(
                checkpoint_path, map_location=config.device, weights_only=True
            )
            reloaded = CanonicalEntityActionScorerModel(
                CanonicalEntityActionScorerConfig(**checkpoint["model_config"])
            ).to(config.device)
            reloaded.load_state_dict(checkpoint["model_state_dict"])
            reloaded.eval()
            evaluation_a = evaluate_model(reloaded, evaluation_dataset, config)
            evaluation_b = evaluate_model(reloaded, evaluation_dataset, config)
            bytes_a = json.dumps(
                evaluation_a, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            bytes_b = json.dumps(
                evaluation_b, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            full_reload_deterministic = (
                bytes_a == bytes_b
                and evaluation == evaluation_a
                and all(
                    torch.equal(model.state_dict()[name].cpu(), value.cpu())
                    for name, value in reloaded.state_dict().items()
                )
            )
    if full_model is None:
        raise RuntimeError("Full 80-game model was not trained.")
    primary = paired_game_bootstrap(
        fractions["80_games"]["evaluation"],
        evaluation_dataset,
        full_model,
        config,
    )
    full_evaluation = fractions["80_games"]["evaluation"]
    phase_deltas = {
        phase: float(block["game_weighted_agreement_delta"])
        for phase, block in full_evaluation["by_phase"].items()
    }
    phase_entropies = {
        phase: float(block["normalized_entropy"])
        for phase, block in full_evaluation["by_phase"].items()
    }
    integrity_checks = {
        **source_integrity,
        "finite_training_and_evaluation": finite_training
        and _all_finite(fractions),
        "deterministic_reload_and_repeated_evaluation": full_reload_deterministic,
        "human_data_not_configured": "human" not in str(config.source_dir).lower()
        and "human" not in str(config.parent_checkpoint).lower(),
        "nested_training_prefixes": set(prefixes[20]) < set(prefixes[40])
        and set(prefixes[40]) < set(prefixes[80]),
    }
    integrity_passed = all(integrity_checks.values())
    gates = {
        "integrity": integrity_passed,
        "primary_ci95_low_above_plus_3_points": primary["ci95_low"] > 0.03,
        "overall_normalized_entropy_at_least_0_30": float(
            full_evaluation["overall"]["normalized_entropy"]
        )
        >= 0.30,
        "each_phase_normalized_entropy_at_least_0_20": all(
            value >= 0.20 for value in phase_entropies.values()
        )
        and set(phase_entropies) == {"early", "mid", "late"},
        "no_phase_agreement_delta_below_minus_5_points": all(
            value >= -0.05 for value in phase_deltas.values()
        )
        and set(phase_deltas) == {"early", "mid", "late"},
    }
    passed = all(gates.values())
    payload = {
        "experiment": "075_from_scratch_resolver_bc_pilot",
        "protocol": "docs/experiments/from_scratch_resolver_bc_pilot_075.md",
        "configuration": _json_config(config),
        "identity": identity,
        "source_integrity": source_integrity,
        "training_game_order": list(prefixes[80]),
        "fractions": fractions,
        "primary_endpoint": primary,
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "gates": gates,
        "passed": passed,
        "decision": (
            "offline_learnability_pass_preregister_fresh_gameplay_experiment"
            if passed
            else "from_scratch_width64_hard_label_resolver_bc_pilot_rejected"
        ),
        "fresh_data_authorized": passed,
        "gameplay_authorized": False,
        "promotion_authorized": False,
        "human_data_used": False,
        "gpu_rental_authorized": False,
    }
    _atomic_write(config.output_dir / "result.json", payload)
    print(
        json.dumps(
            {
                "passed": passed,
                "decision": payload["decision"],
                "primary_endpoint": primary,
                "gates": gates,
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def _validate_frozen_config(config: ResolverBCConfig) -> None:
    expected = ResolverBCConfig(output_dir=config.output_dir)
    for field in (
        "source_dir",
        "parent_checkpoint",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "entropy_weight",
        "bootstrap_samples",
        "seed",
        "device",
    ):
        if getattr(config, field) != getattr(expected, field):
            raise ValueError(f"075 frozen configuration mismatch for {field}.")


def _validate_source_identity(config: ResolverBCConfig) -> dict[str, Any]:
    paths = {
        "result": config.source_dir / "result.json",
        "train": config.source_dir / "train_targets.jsonl",
        "validation": config.source_dir / "validation_targets.jsonl",
        "confirmation": config.source_dir / "confirmation_targets.jsonl",
        "parent_035": config.parent_checkpoint,
    }
    observed = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256_file(path)
        if observed[name] != SOURCE_SHA256[name]:
            raise ValueError(
                f"075 source identity mismatch for {name}: expected "
                f"{SOURCE_SHA256[name]}, observed {observed[name]}."
            )
    source_result = json.loads(paths["result"].read_text(encoding="utf-8"))
    collection_passed = all(
        bool(source_result.get(split, {}).get("passed"))
        and int(source_result[split]["integrity"]["invalid_games"]) == 0
        and int(source_result[split]["integrity"]["stalled_games"]) == 0
        and int(source_result[split]["integrity"]["timed_out_games"]) == 0
        for split in ("train", "validation", "confirmation")
    )
    if not collection_passed:
        raise ValueError("Frozen 052 collection-integrity gates did not pass.")
    return {
        "expected_sha256": SOURCE_SHA256,
        "observed_sha256": observed,
        "collection_integrity_passed": collection_passed,
    }


def _load_source_samples(
    config: ResolverBCConfig,
) -> tuple[dict[str, list[ResolverBCSample]], dict[str, bool]]:
    env = SplendorEnv()
    codec = ActionCodec()
    samples: dict[str, list[ResolverBCSample]] = {}
    checks = {
        "exact_source_rows_and_games": True,
        "legal_canonical_targets": True,
        "train_eval_games_disjoint": True,
        "allowed_source_files_only": True,
    }
    games_by_split: dict[str, set[int]] = {}
    for split in ("train", "validation", "confirmation"):
        split_samples: list[ResolverBCSample] = []
        path = config.source_dir / f"{split}_targets.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not bool(row.get("target_available")):
                    continue
                try:
                    split_samples.append(_sample_from_row(row, env, codec))
                except (KeyError, TypeError, ValueError):
                    checks["legal_canonical_targets"] = False
                    raise
        games = {sample.game_seed for sample in split_samples}
        games_by_split[split] = games
        checks["exact_source_rows_and_games"] &= (
            len(split_samples) == EXPECTED_ROWS[split]
            and len(games) == EXPECTED_GAMES[split]
        )
        samples[split] = split_samples
    checks["train_eval_games_disjoint"] = not (
        games_by_split["train"]
        & (games_by_split["validation"] | games_by_split["confirmation"])
    ) and not (games_by_split["validation"] & games_by_split["confirmation"])
    if not all(checks.values()):
        raise ValueError(f"075 source integrity failed: {checks}")
    return samples, checks


def _sample_from_row(
    row: dict[str, Any], env: SplendorEnv, codec: ActionCodec
) -> ResolverBCSample:
    state = deserialize_state_snapshot(row["state_snapshot"])
    raw_actions = env.legal_actions(state)
    by_index = {codec.encode(state, action): action for action in raw_actions}
    target_index = int(row["action_index"])
    parent_index = int(row["raw_parent_action_index"])
    if target_index not in by_index or parent_index not in by_index:
        raise ValueError("Resolver or parent action is illegal in reconstructed state.")
    representatives = actions_by_canonical_successor_key(state, raw_actions)
    keys = list(representatives)
    target_key = canonical_successor_action_key(state, by_index[target_index])
    parent_key = canonical_successor_action_key(state, by_index[parent_index])
    if target_key not in representatives or parent_key not in representatives:
        raise ValueError("Canonical target is absent from reconstructed legal actions.")
    return ResolverBCSample(
        state=state,
        legal_actions=tuple(representatives.values()),
        target_action_row=keys.index(target_key),
        parent_action_row=keys.index(parent_key),
        game_seed=int(row["game_seed"]),
        phase=str(row["phase"]),
        seat=int(row["player_id"]),
        outcome=str(row["outcome_role"]),
    )


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _json_config(config: ResolverBCConfig) -> dict[str, Any]:
    payload = asdict(config)
    for name in ("source_dir", "parent_checkpoint", "output_dir"):
        payload[name] = str(payload[name])
    payload["model_config"] = asdict(model_config_075())
    payload["training_game_prefixes"] = list(TRAIN_GAME_PREFIXES)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/from_scratch_resolver_bc_pilot_075"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(ResolverBCConfig(output_dir=Path(args.output_dir)))


if __name__ == "__main__":
    main()
