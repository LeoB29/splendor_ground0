"""Train and gate a score-head-only reserve counterfactual policy correction."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from splendor_ai.encoding import ActionCodec, encode_legal_action_features, encode_public_observation_tensor, fit_feature_vector
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.dataset import LegalActionReplayDataset, collate_legal_action_replay_samples
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel, masked_legal_action_scores
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_root_visit_correction import compute_parent_policy_kl


@dataclass(frozen=True, slots=True)
class ReserveCounterfactualTrainingConfig:
    train_path: Path
    heldout_path: Path
    external_path: Path
    anchor_path: Path
    init_checkpoint: Path
    output_dir: Path
    epochs: int = 4
    batch_size: int = 32
    anchor_batch_size: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    anchor_kl_weight: float = 2.0
    target_temperature: float = 0.35
    target_objective: str = "soft_return"
    train_scope: str = "score_head"
    max_grad_norm: float = 1.0
    bootstrap_samples: int = 2_000
    seed: int = 142
    device: str = "cpu"
    log_every_batches: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one score-head-only policy035 reserve correction on soft matched-return rankings, "
            "with broad parent-KL anchoring and fixed held-out gates."
        )
    )
    for argument in ("train", "heldout", "external", "anchor"):
        parser.add_argument(f"--{argument}-path", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--anchor-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-kl-weight", type=float, default=2.0)
    parser.add_argument("--target-temperature", type=float, default=0.35)
    parser.add_argument(
        "--target-objective",
        choices=("soft_return", "hard_top"),
        default="soft_return",
    )
    parser.add_argument(
        "--train-scope",
        choices=("score_head", "full_model"),
        default="score_head",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-batches", type=int, default=10)
    return parser


@dataclass(frozen=True, slots=True)
class _CounterfactualSample:
    observation: torch.Tensor
    legal_action_features: torch.Tensor
    candidate_rows: torch.Tensor
    candidate_returns: torch.Tensor
    game_seed: int
    turn_index: int
    turn_band: str


class _CounterfactualDataset(Dataset[_CounterfactualSample]):
    def __init__(self, path: Path, *, observation_size: int, action_feature_size: int) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"No counterfactual rows in {path}.")
        self.path = path
        self._env = SplendorEnv()
        self._codec = ActionCodec()
        self._observation_size = observation_size
        self._action_feature_size = action_feature_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> _CounterfactualSample:
        row = self.rows[index]
        state = deserialize_state_snapshot(row["state_snapshot"])
        player_id = int(row["player_id"])
        legal_actions = self._env.legal_actions(state)
        legal_indices = tuple(self._codec.legal_action_indices(state, legal_actions))
        candidate_map = {
            int(action["action_index"]): float(action["rollout_return"])
            for action in row["counterfactual_actions"]
        }
        candidate_rows = []
        candidate_returns = []
        for action_index, value in candidate_map.items():
            try:
                action_row = legal_indices.index(action_index)
            except ValueError as exc:
                raise ValueError(
                    f"Candidate action {action_index} is not legal for state {row['game_seed']}/{row['turn_index']}."
                ) from exc
            candidate_rows.append(action_row)
            candidate_returns.append(value)
        if len(candidate_rows) < 2:
            raise ValueError("Counterfactual row needs at least two legal candidate actions.")
        observation = encode_public_observation_tensor(state, player_id)
        action_features = [
            torch.tensor(
                fit_feature_vector(features.vector, self._action_feature_size), dtype=torch.float32
            )
            for features in encode_legal_action_features(state, legal_actions, player_id)
        ]
        return _CounterfactualSample(
            observation=torch.tensor(
                fit_feature_vector(observation.vector, self._observation_size), dtype=torch.float32
            ),
            legal_action_features=torch.stack(action_features),
            candidate_rows=torch.tensor(candidate_rows, dtype=torch.long),
            candidate_returns=torch.tensor(candidate_returns, dtype=torch.float32),
            game_seed=int(row["game_seed"]),
            turn_index=int(row["turn_index"]),
            turn_band=str(row["turn_band"]),
        )


def _collate_counterfactual(samples: list[_CounterfactualSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate zero counterfactual samples.")
    max_actions = max(sample.legal_action_features.shape[0] for sample in samples)
    feature_size = samples[0].legal_action_features.shape[1]
    features = torch.zeros((len(samples), max_actions, feature_size), dtype=torch.float32)
    legal_mask = torch.zeros((len(samples), max_actions), dtype=torch.bool)
    candidate_mask = torch.zeros((len(samples), max_actions), dtype=torch.bool)
    candidate_returns = torch.zeros((len(samples), max_actions), dtype=torch.float32)
    for row, sample in enumerate(samples):
        action_count = sample.legal_action_features.shape[0]
        features[row, :action_count] = sample.legal_action_features
        legal_mask[row, :action_count] = True
        candidate_mask[row, sample.candidate_rows] = True
        candidate_returns[row, sample.candidate_rows] = sample.candidate_returns
    return {
        "observation": torch.stack([sample.observation for sample in samples]),
        "legal_action_features": features,
        "legal_action_mask": legal_mask,
        "candidate_mask": candidate_mask,
        "candidate_returns": candidate_returns,
        "game_seed": [sample.game_seed for sample in samples],
        "turn_index": [sample.turn_index for sample in samples],
        "turn_band": [sample.turn_band for sample in samples],
    }


def _load_model(path: Path, device: torch.device) -> tuple[LegalActionAttentionScorerModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError("Reserve correction requires a legal action attention scorer checkpoint.")
    config_keys = {field.name for field in fields(LegalActionAttentionScorerConfig)}
    config = LegalActionAttentionScorerConfig(
        **{key: value for key, value in checkpoint["model_config"].items() if key in config_keys}
    )
    model = LegalActionAttentionScorerModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def compute_counterfactual_soft_loss(
    action_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_returns: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("target temperature must be positive.")
    logits = action_scores.masked_fill(~candidate_mask, -torch.inf)
    target_logits = (candidate_returns / temperature).masked_fill(~candidate_mask, -torch.inf)
    target = torch.softmax(target_logits, dim=1).detach()
    log_probs = torch.log_softmax(logits, dim=1).masked_fill(~candidate_mask, 0.0)
    return (-(target * log_probs).sum(dim=1)).mean()


def compute_counterfactual_hard_top_loss(
    action_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_returns: torch.Tensor,
) -> torch.Tensor:
    logits = action_scores.masked_fill(~candidate_mask, -torch.inf)
    target_rows = candidate_returns.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1)
    return torch.nn.functional.cross_entropy(logits, target_rows)


def _counterfactual_loss(
    action_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_returns: torch.Tensor,
    config: ReserveCounterfactualTrainingConfig,
) -> torch.Tensor:
    if config.target_objective == "soft_return":
        return compute_counterfactual_soft_loss(
            action_scores, candidate_mask, candidate_returns, config.target_temperature
        )
    if config.target_objective == "hard_top":
        return compute_counterfactual_hard_top_loss(
            action_scores, candidate_mask, candidate_returns
        )
    raise ValueError(f"Unsupported target objective: {config.target_objective}")


def _freeze_train_scope(
    model: LegalActionAttentionScorerModel,
    scope: str,
) -> list[str]:
    trainable = []
    for name, parameter in model.named_parameters():
        train = scope == "full_model" or name.startswith("score_head.")
        parameter.requires_grad_(train)
        if train:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("No parameters selected for training.")
    return trainable


def train_reserve_counterfactual_correction(
    config: ReserveCounterfactualTrainingConfig,
) -> dict[str, Any]:
    _validate(config)
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    model, checkpoint = _load_model(config.init_checkpoint, device)
    parent, _ = _load_model(config.init_checkpoint, device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    trainable_names = _freeze_train_scope(model, config.train_scope)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    train_dataset = _CounterfactualDataset(
        config.train_path,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    anchor_dataset = LegalActionReplayDataset(
        config.anchor_path,
        include_stalled_games=False,
        include_timed_out_games=False,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=_collate_counterfactual)
    anchor_loader = DataLoader(anchor_dataset, batch_size=config.anchor_batch_size, shuffle=True, collate_fn=collate_legal_action_replay_samples)
    optimizer = AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    started = time.perf_counter()
    epochs = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        anchor_iterator = iter(anchor_loader)
        totals = {"counterfactual_loss": 0.0, "anchor_kl": 0.0, "total_loss": 0.0}
        rows = 0
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            try:
                anchor_batch = next(anchor_iterator)
            except StopIteration:
                anchor_iterator = iter(anchor_loader)
                anchor_batch = next(anchor_iterator)
            observation = batch["observation"].to(device)
            features = batch["legal_action_features"].to(device)
            legal_mask = batch["legal_action_mask"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            candidate_returns = batch["candidate_returns"].to(device)
            anchor_observation = anchor_batch["observation"].to(device)
            anchor_features = anchor_batch["legal_action_features"].to(device)
            anchor_mask = anchor_batch["legal_action_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            scores, _ = model(observation, features, legal_mask)
            anchor_scores, _ = model(anchor_observation, anchor_features, anchor_mask)
            with torch.no_grad():
                parent_scores, _ = parent(anchor_observation, anchor_features, anchor_mask)
            counterfactual_loss = _counterfactual_loss(
                scores, candidate_mask, candidate_returns, config
            )
            anchor_kl = compute_parent_policy_kl(anchor_scores, parent_scores, anchor_mask)
            total_loss = counterfactual_loss + config.anchor_kl_weight * anchor_kl
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
            optimizer.step()
            count = int(observation.shape[0])
            rows += count
            totals["counterfactual_loss"] += float(counterfactual_loss.detach()) * count
            totals["anchor_kl"] += float(anchor_kl.detach()) * count
            totals["total_loss"] += float(total_loss.detach()) * count
            if batch_index == 1 or batch_index == len(train_loader) or (
                config.log_every_batches > 0 and batch_index % config.log_every_batches == 0
            ):
                print(
                    f"[reserve-counterfactual] epoch={epoch}/{config.epochs} batch={batch_index}/{len(train_loader)}"
                    f" soft_ce={totals['counterfactual_loss'] / rows:.5f}"
                    f" anchor_kl={totals['anchor_kl'] / rows:.6f}"
                    f" elapsed={time.perf_counter() - epoch_started:.1f}s"
                )
        epochs.append({"epoch": epoch, **{key: value / rows for key, value in totals.items()}, "rows": rows})
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "legal_action_attention_scorer.pt"
    metrics_path = config.output_dir / "reserve_counterfactual_training_metrics.json"
    config_payload = _config_payload(config)
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_config": checkpoint["model_config"],
            "model_state_dict": model.state_dict(),
            "train_config": config_payload,
        },
        checkpoint_path,
    )
    summary = {
        "format": "reserve_counterfactual_training_v1",
        "config": config_payload,
        "train_rows": len(train_dataset),
        "anchor_rows": len(anchor_dataset),
        "trainable_parameters": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "epochs": epochs,
        "training_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def audit_reserve_counterfactual_correction(
    config: ReserveCounterfactualTrainingConfig,
    candidate_checkpoint: Path,
    output_path: Path,
) -> dict[str, Any]:
    device = torch.device(config.device)
    baseline, _ = _load_model(config.init_checkpoint, device)
    candidate, _ = _load_model(candidate_checkpoint, device)
    baseline.eval()
    candidate.eval()
    heldout = _CounterfactualDataset(config.heldout_path, observation_size=baseline.config.observation_size, action_feature_size=baseline.config.action_feature_size)
    external = _CounterfactualDataset(config.external_path, observation_size=baseline.config.observation_size, action_feature_size=baseline.config.action_feature_size)
    train = _CounterfactualDataset(config.train_path, observation_size=baseline.config.observation_size, action_feature_size=baseline.config.action_feature_size)
    anchor = LegalActionReplayDataset(config.anchor_path, include_stalled_games=False, include_timed_out_games=False, observation_size=baseline.config.observation_size, action_feature_size=baseline.config.action_feature_size)
    heldout_metrics = _counterfactual_metrics(heldout, baseline, candidate, config)
    external_metrics = _counterfactual_metrics(external, baseline, candidate, config)
    broad = _broad_drift_metrics(anchor, baseline, candidate, config)
    gates = {
        "heldout_soft_ce_improves": heldout_metrics["soft_ce_delta_candidate_minus_baseline"]["ci95_high"] < -0.01,
        "heldout_best_action_improves": heldout_metrics["best_action_accuracy_delta"]["ci95_low"] > 0.0,
        "heldout_decision_novelty": 0.02 <= heldout_metrics["full_top_action_change_rate"] <= 0.20,
        "external_soft_ce_nonregression": external_metrics["soft_ce_delta_candidate_minus_baseline"]["ci95_high"] < 0.0,
        "broad_change_ceiling": broad["top_action_change_rate"] <= 0.10,
        "broad_kl_ceiling": broad["mean_parent_kl"] <= 0.01,
    }
    payload = {
        "format": "reserve_counterfactual_uptake_audit_v1",
        "config": _config_payload(config) | {"candidate_checkpoint": str(candidate_checkpoint)},
        "train": _counterfactual_metrics(train, baseline, candidate, config),
        "heldout": heldout_metrics,
        "external": external_metrics,
        "broad_drift": broad,
        "gates": gates,
        "pass": all(gates.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _counterfactual_metrics(
    dataset: _CounterfactualDataset,
    baseline: LegalActionAttentionScorerModel,
    candidate: LegalActionAttentionScorerModel,
    config: ReserveCounterfactualTrainingConfig,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_counterfactual)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            observation = batch["observation"].to(config.device)
            features = batch["legal_action_features"].to(config.device)
            legal_mask = batch["legal_action_mask"].to(config.device)
            candidate_mask = batch["candidate_mask"].to(config.device)
            returns = batch["candidate_returns"].to(config.device)
            baseline_scores, _ = baseline(observation, features, legal_mask)
            candidate_scores, _ = candidate(observation, features, legal_mask)
            target_logits = (returns / config.target_temperature).masked_fill(~candidate_mask, -torch.inf)
            target = torch.softmax(target_logits, dim=1)
            base_log_probs = torch.log_softmax(baseline_scores.masked_fill(~candidate_mask, -torch.inf), dim=1).masked_fill(~candidate_mask, 0.0)
            cand_log_probs = torch.log_softmax(candidate_scores.masked_fill(~candidate_mask, -torch.inf), dim=1).masked_fill(~candidate_mask, 0.0)
            base_probs = base_log_probs.exp().masked_fill(~candidate_mask, 0.0)
            cand_probs = cand_log_probs.exp().masked_fill(~candidate_mask, 0.0)
            best = returns.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1)
            base_top = baseline_scores.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1)
            cand_top = candidate_scores.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1)
            base_full_top = baseline_scores.masked_fill(~legal_mask, -torch.inf).argmax(dim=1)
            cand_full_top = candidate_scores.masked_fill(~legal_mask, -torch.inf).argmax(dim=1)
            for index in range(observation.shape[0]):
                mask = candidate_mask[index]
                rows.append(
                    {
                        "game_seed": int(batch["game_seed"][index]),
                        "turn_band": batch["turn_band"][index],
                        "baseline_soft_ce": float((-(target[index] * base_log_probs[index]).sum()).item()),
                        "candidate_soft_ce": float((-(target[index] * cand_log_probs[index]).sum()).item()),
                        "baseline_best_action": float(base_top[index] == best[index]),
                        "candidate_best_action": float(cand_top[index] == best[index]),
                        "baseline_spearman": _spearman(
                            baseline_scores[index][mask].tolist(), returns[index][mask].tolist()
                        ),
                        "candidate_spearman": _spearman(
                            candidate_scores[index][mask].tolist(), returns[index][mask].tolist()
                        ),
                        "full_top_action_changed": float(base_full_top[index] != cand_full_top[index]),
                        "candidate_best_probability": float(cand_probs[index, best[index]].item()),
                        "baseline_best_probability": float(base_probs[index, best[index]].item()),
                    }
                )
    return _summarize_counterfactual_rows(rows, config.bootstrap_samples, config.seed)


def _broad_drift_metrics(
    dataset: LegalActionReplayDataset,
    baseline: LegalActionAttentionScorerModel,
    candidate: LegalActionAttentionScorerModel,
    config: ReserveCounterfactualTrainingConfig,
) -> dict[str, float]:
    count = min(2_000, len(dataset))
    rng = random.Random(config.seed + 97)
    indices = sorted(rng.sample(range(len(dataset)), count)) if count < len(dataset) else list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=config.anchor_batch_size, shuffle=False, collate_fn=collate_legal_action_replay_samples)
    changes = 0
    kls = []
    with torch.inference_mode():
        for batch in loader:
            observation = batch["observation"].to(config.device)
            features = batch["legal_action_features"].to(config.device)
            mask = batch["legal_action_mask"].to(config.device)
            parent_scores, _ = baseline(observation, features, mask)
            candidate_scores, _ = candidate(observation, features, mask)
            parent_logits = masked_legal_action_scores(parent_scores, mask)
            candidate_logits = masked_legal_action_scores(candidate_scores, mask)
            parent_probs = torch.softmax(parent_logits, dim=1)
            candidate_probs = torch.softmax(candidate_logits, dim=1)
            kls.extend((parent_probs * (parent_probs.clamp_min(1e-12).log() - candidate_probs.clamp_min(1e-12).log())).sum(dim=1).cpu().tolist())
            changes += int((parent_logits.argmax(dim=1) != candidate_logits.argmax(dim=1)).sum().item())
    return {
        "rows": count,
        "top_action_changes": changes,
        "top_action_change_rate": changes / count,
        "mean_parent_kl": statistics.fmean(kls),
        "p95_parent_kl": _percentile(sorted(kls), 0.95),
    }


def _summarize_counterfactual_rows(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Counterfactual audit has no rows.")
    def paired(name: str, left: str, right: str) -> dict[str, float]:
        values = [float(row[left]) - float(row[right]) for row in rows]
        return _mean_ci(values, bootstrap_samples, seed + sum(ord(char) for char in name))
    return {
        "rows": len(rows),
        "games": len({int(row["game_seed"]) for row in rows}),
        "baseline_soft_ce": _mean_ci([float(row["baseline_soft_ce"]) for row in rows], bootstrap_samples, seed + 1),
        "candidate_soft_ce": _mean_ci([float(row["candidate_soft_ce"]) for row in rows], bootstrap_samples, seed + 2),
        "soft_ce_delta_candidate_minus_baseline": paired("ce", "candidate_soft_ce", "baseline_soft_ce"),
        "baseline_best_action_accuracy": _mean_ci([float(row["baseline_best_action"]) for row in rows], bootstrap_samples, seed + 3),
        "candidate_best_action_accuracy": _mean_ci([float(row["candidate_best_action"]) for row in rows], bootstrap_samples, seed + 4),
        "best_action_accuracy_delta": paired("best", "candidate_best_action", "baseline_best_action"),
        "spearman_delta_candidate_minus_baseline": paired("spearman", "candidate_spearman", "baseline_spearman"),
        "best_probability_delta_candidate_minus_baseline": paired("prob", "candidate_best_probability", "baseline_best_probability"),
        "full_top_action_change_rate": statistics.fmean(float(row["full_top_action_changed"]) for row in rows),
        "by_turn_band": {
            band: _band_metrics([row for row in rows if row["turn_band"] == band])
            for band in sorted({str(row["turn_band"]) for row in rows})
        },
    }


def _band_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "rows": len(rows),
        "soft_ce_delta_candidate_minus_baseline": statistics.fmean(float(row["candidate_soft_ce"]) - float(row["baseline_soft_ce"]) for row in rows),
        "best_action_accuracy_delta": statistics.fmean(float(row["candidate_best_action"]) - float(row["baseline_best_action"]) for row in rows),
    }


def _mean_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    mean = statistics.fmean(values)
    if len(values) < 2 or samples <= 0:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return {"mean": mean, "ci95_low": _percentile(means, 0.025), "ci95_high": _percentile(means, 0.975)}


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def _spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left_ranks = _ranks(list(left))
    right_ranks = _ranks(list(right))
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True))
    left_var = sum((a - left_mean) ** 2 for a in left_ranks)
    right_var = sum((b - right_mean) ** 2 for b in right_ranks)
    if left_var <= 0.0 or right_var <= 0.0:
        return 0.0
    return numerator / math.sqrt(left_var * right_var)


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


def _config_payload(config: ReserveCounterfactualTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("train_path", "heldout_path", "external_path", "anchor_path", "init_checkpoint", "output_dir"):
        payload[key] = str(payload[key])
    return payload


def _validate(config: ReserveCounterfactualTrainingConfig) -> None:
    for path in (config.train_path, config.heldout_path, config.external_path, config.anchor_path, config.init_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    for name, value in (("epochs", config.epochs), ("batch_size", config.batch_size), ("anchor_batch_size", config.anchor_batch_size), ("bootstrap_samples", config.bootstrap_samples)):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.learning_rate <= 0.0 or config.target_temperature <= 0.0 or config.max_grad_norm <= 0.0:
        raise ValueError("Learning rate, target temperature, and grad norm must be positive.")
    if config.anchor_kl_weight < 0.0 or config.weight_decay < 0.0:
        raise ValueError("Anchor KL and weight decay must be non-negative.")
    if config.target_objective not in {"soft_return", "hard_top"}:
        raise ValueError("target_objective must be 'soft_return' or 'hard_top'.")
    if config.train_scope not in {"score_head", "full_model"}:
        raise ValueError("train_scope must be 'score_head' or 'full_model'.")


def main() -> None:
    args = build_parser().parse_args()
    config = ReserveCounterfactualTrainingConfig(
        train_path=Path(args.train_path), heldout_path=Path(args.heldout_path), external_path=Path(args.external_path), anchor_path=Path(args.anchor_path), init_checkpoint=Path(args.init_checkpoint), output_dir=Path(args.output_dir), epochs=args.epochs, batch_size=args.batch_size, anchor_batch_size=args.anchor_batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, anchor_kl_weight=args.anchor_kl_weight, target_temperature=args.target_temperature, target_objective=args.target_objective, train_scope=args.train_scope, max_grad_norm=args.max_grad_norm, bootstrap_samples=args.bootstrap_samples, seed=args.seed, device=args.device, log_every_batches=args.log_every_batches,
    )
    summary = train_reserve_counterfactual_correction(config)
    candidate = Path(summary["checkpoint"])
    audit = audit_reserve_counterfactual_correction(config, candidate, config.output_dir / "reserve_counterfactual_uptake_audit.json")
    print(
        "[reserve-counterfactual]"
        f" checkpoint={candidate}"
        f" heldout_ce_delta={audit['heldout']['soft_ce_delta_candidate_minus_baseline']['mean']:+.5f}"
        f" pass={audit['pass']}"
    )


if __name__ == "__main__":
    main()
