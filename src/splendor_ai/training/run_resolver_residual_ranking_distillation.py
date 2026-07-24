"""Run preregistered experiment 053: frozen-parent residual ranking distillation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable, Literal, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_bounded_public_belief_resolving import _sha256_file
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    _integrity,
    _seat_breakdown,
)
from splendor_ai.eval.run_shallow_operational_gate import (
    SHALLOW_CONTRACT,
    summarize_pair_rows,
)
from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    LegalActionReplaySample,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.resolver_residual import (
    FrozenParentResidualScorerBot,
    FrozenParentResidualScorerModel,
    ResolverResidualConfig,
    load_frozen_parent_residual_model,
    save_frozen_parent_residual_checkpoint,
    trainable_parameter_count,
)
from splendor_ai.training.run_resolver_selfplay_distillation import (
    CHECKPOINT_016_SHA256,
    CHECKPOINT_023_SHA256,
    ResolverDistillationConfig,
    _collect_split,
)


AUDIT_052A_SHA256 = "226d28a4cd9667ef27eacfc1afccf8cac628d07b30006e9dc179ca282b2eba57"
PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
TRAIN_SHA256 = "6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5"
VALIDATION_SHA256 = "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052"
Opponent = Literal["035", "shallow", "016", "023"]
BROAD_BLOCKS: tuple[tuple[Opponent, int], ...] = (
    ("shallow", 620_700),
    ("016", 620_800),
    ("023", 620_900),
)


@dataclass(frozen=True, slots=True)
class ResolverResidualExperimentConfig:
    output_dir: Path = Path("outputs/resolver_residual_ranking_distillation_053")
    audit_052a: Path = Path(
        "outputs/resolver_distillation_disagreement_audit_052a/result.json"
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
    checkpoint_016: Path = Path(
        "outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt"
    )
    checkpoint_023: Path = Path(
        "outputs/canonical_entity_coherent_023/"
        "canonical_entity_action_scorer_coherent_width64_best.pt"
    )
    residual_hidden_size: int = 64
    ranking_margin: float = 0.25
    anchor_kl_weight: float = 5.0
    residual_mse_weight: float = 0.01
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 12
    shuffle_seed: int = 53
    fresh_confirmation_seed_start: int = 620_300
    fresh_confirmation_games: int = 20
    direct_seed_start: int = 620_500
    direct_pairs: int = 100
    broad_pairs: int = 50
    workers: int = 10
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 530_053
    minimum_resolver_agreement_gain: float = 0.05
    minimum_disagreement_capture: float = 0.15
    minimum_directional_movement: float = 0.80
    minimum_anchor_retention: float = 0.98
    maximum_fallback_kl: float = 0.02
    minimum_action_movement: float = 0.05
    maximum_action_movement: float = 0.25
    maximum_target_kl_ratio: float = 1.0
    maximum_train_validation_capture_gap: float = 0.10
    minimum_stratum_rows: int = 50
    minimum_score_share: float = 0.55
    minimum_seat_score_share: float = 0.50
    device: str = "cpu"


class ResolverResidualReplayDataset(Dataset[tuple[LegalActionReplaySample, dict[str, Any]]]):
    """Expose 052 rows with the raw parent action needed by the ranking loss."""

    def __init__(self, path: Path, model: FrozenParentResidualScorerModel) -> None:
        self._metadata = _read_jsonl(path)
        self._base = LegalActionReplayDataset(
            path,
            observation_size=model.parent_config.observation_size,
            action_feature_size=model.parent_config.action_feature_size,
        )
        if len(self._metadata) != len(self._base):
            raise ValueError("Residual metadata and reconstructed replay rows differ.")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[LegalActionReplaySample, dict[str, Any]]:
        return self._base[index], self._metadata[index]


def collate_resolver_residual_rows(
    rows: Sequence[tuple[LegalActionReplaySample, dict[str, Any]]],
) -> dict[str, Any]:
    samples = [row[0] for row in rows]
    metadata = [row[1] for row in rows]
    batch = collate_legal_action_replay_samples(samples)
    parent_rows = []
    target_available = []
    for sample, payload in rows:
        parent_index = int(payload["raw_parent_action_index"])
        if parent_index not in sample.legal_action_indices:
            raise ValueError("Stored parent action is absent from legal replay actions.")
        parent_rows.append(sample.legal_action_indices.index(parent_index))
        target_available.append(bool(payload["target_available"]))
    batch["parent_action_row"] = torch.tensor(parent_rows, dtype=torch.long)
    batch["target_available"] = torch.tensor(target_available, dtype=torch.bool)
    batch["metadata"] = metadata
    return batch


def compute_residual_ranking_losses(
    *,
    student_scores: torch.Tensor,
    parent_scores: torch.Tensor,
    residual: torch.Tensor,
    legal_mask: torch.Tensor,
    resolver_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    target_available: torch.Tensor,
    ranking_margin: float,
    anchor_kl_weight: float,
    residual_mse_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute the one frozen 053 objective."""

    device = student_scores.device
    legal_mask = legal_mask.to(device=device, dtype=torch.bool)
    resolver_rows = resolver_rows.to(device)
    parent_rows = parent_rows.to(device)
    target_available = target_available.to(device)
    row_indices = torch.arange(student_scores.shape[0], device=device)
    ranking_mask = target_available & (resolver_rows != parent_rows)
    anchor_mask = ~ranking_mask
    gaps = (
        student_scores[row_indices, resolver_rows]
        - student_scores[row_indices, parent_rows]
    )
    ranking_loss = (
        nn.functional.softplus(ranking_margin - gaps[ranking_mask]).mean()
        if ranking_mask.any()
        else student_scores.sum() * 0.0
    )
    parent_logits = parent_scores.masked_fill(~legal_mask, -torch.inf)
    student_logits = student_scores.masked_fill(~legal_mask, -torch.inf)
    parent_probs = nn.functional.softmax(parent_logits, dim=1)
    parent_log_probs = nn.functional.log_softmax(parent_logits, dim=1)
    student_log_probs = nn.functional.log_softmax(student_logits, dim=1)
    per_row_kl = (
        parent_probs
        * (parent_log_probs - student_log_probs).masked_fill(~legal_mask, 0.0)
    ).sum(dim=1)
    anchor_kl = (
        per_row_kl[anchor_mask].mean()
        if anchor_mask.any()
        else student_scores.sum() * 0.0
    )
    residual_mse = (
        residual.square().masked_fill(~legal_mask, 0.0).sum()
        / legal_mask.sum().clamp_min(1)
    )
    total = (
        ranking_loss
        + anchor_kl_weight * anchor_kl
        + residual_mse_weight * residual_mse
    )
    return {
        "total": total,
        "ranking": ranking_loss,
        "anchor_kl": anchor_kl,
        "residual_mse": residual_mse,
        "ranking_rows": ranking_mask.sum(),
        "anchor_rows": anchor_mask.sum(),
    }


def run_experiment(
    config: ResolverResidualExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResolverResidualExperimentConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != _identity(cfg):
            raise ValueError("Existing 053 result does not match frozen identity.")
        return result

    training = _train_and_select(cfg)
    if not training["validation_passed"]:
        return _finish(
            cfg,
            decision="validation_residual_gate_failed",
            training=training,
        )

    teacher_cfg = ResolverDistillationConfig(
        output_dir=cfg.output_dir,
        confirmation_seed_start=cfg.fresh_confirmation_seed_start,
        confirmation_games=cfg.fresh_confirmation_games,
        workers=cfg.workers,
    )
    confirmation_collection = _collect_split(
        teacher_cfg,
        "fresh_confirmation",
        range(
            cfg.fresh_confirmation_seed_start,
            cfg.fresh_confirmation_seed_start + cfg.fresh_confirmation_games,
        ),
        1_000,
    )
    if not confirmation_collection["passed"]:
        return _finish(
            cfg,
            decision="fresh_confirmation_collection_failed",
            training=training,
            confirmation_collection=confirmation_collection,
        )

    confirmation = _evaluate_checkpoint(
        cfg,
        cfg.output_dir / "fresh_confirmation_targets.jsonl",
        cfg.output_dir / "frozen_parent_residual_scorer_best.pt",
        split_name="confirmation",
    )
    confirmation_gates = _offline_gates(
        cfg,
        confirmation,
        training["selected_train"],
        require_bootstrap=True,
    )
    if not all(confirmation_gates.values()):
        return _finish(
            cfg,
            decision="fresh_confirmation_residual_gate_failed",
            training=training,
            confirmation_collection=confirmation_collection,
            confirmation=confirmation,
            confirmation_gates=confirmation_gates,
        )

    direct = _run_gameplay_block(
        cfg,
        "direct_vs_035",
        "035",
        cfg.direct_seed_start,
        cfg.direct_pairs,
    )
    direct_gates = _gameplay_gates(cfg, direct)
    if not all(direct_gates.values()):
        return _finish(
            cfg,
            decision="paired_parent_gate_failed",
            training=training,
            confirmation_collection=confirmation_collection,
            confirmation=confirmation,
            confirmation_gates=confirmation_gates,
            direct=direct,
            direct_gates=direct_gates,
        )

    broad: dict[str, Any] = {}
    broad_gates: dict[str, dict[str, bool]] = {}
    for opponent, seed_start in BROAD_BLOCKS:
        block = _run_gameplay_block(
            cfg,
            f"broad_vs_{opponent}",
            opponent,
            seed_start,
            cfg.broad_pairs,
        )
        broad[opponent] = block
        broad_gates[opponent] = _gameplay_gates(cfg, block)
    passed = all(all(gates.values()) for gates in broad_gates.values())
    return _finish(
        cfg,
        passed=passed,
        decision=(
            "promote_residual_authorize_preregistered_next_generation"
            if passed
            else "broad_promotion_gate_failed"
        ),
        training=training,
        confirmation_collection=confirmation_collection,
        confirmation=confirmation,
        confirmation_gates=confirmation_gates,
        direct=direct,
        direct_gates=direct_gates,
        broad=broad,
        broad_gates=broad_gates,
    )


def _train_and_select(cfg: ResolverResidualExperimentConfig) -> dict[str, Any]:
    path = cfg.output_dir / "training.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    torch.manual_seed(cfg.shuffle_seed)
    torch.set_num_threads(1)
    model = load_frozen_parent_residual_model(
        cfg.parent_checkpoint,
        cfg.device,
        ResolverResidualConfig(hidden_size=cfg.residual_hidden_size),
    )
    if trainable_parameter_count(model) != 24_769:
        raise ValueError("Frozen 053 residual parameter count changed.")
    train_dataset = ResolverResidualReplayDataset(cfg.train_replay, model)
    validation_dataset = ResolverResidualReplayDataset(cfg.validation_replay, model)
    generator = torch.Generator().manual_seed(cfg.shuffle_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_resolver_residual_rows,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_resolver_residual_rows,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_resolver_residual_rows,
    )
    baseline_validation = _evaluate_model(
        cfg,
        model,
        validation_loader,
        "validation_epoch_0",
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    checkpoint_dir = cfg.output_dir / "checkpoints"
    history = []
    started = time.perf_counter()
    provenance = _checkpoint_provenance(cfg)
    for epoch in range(1, cfg.epochs + 1):
        train_losses = _train_epoch(cfg, model, train_loader, optimizer)
        validation = _evaluate_model(
            cfg,
            model,
            validation_loader,
            f"validation_epoch_{epoch}",
        )
        checkpoint = checkpoint_dir / f"epoch_{epoch:02d}.pt"
        save_frozen_parent_residual_checkpoint(
            checkpoint,
            model,
            {**provenance, "epoch": epoch},
        )
        eligible = _selection_eligible(cfg, validation)
        history.append(
            {
                "epoch": epoch,
                "train_losses": train_losses,
                "validation": validation,
                "selection_eligible": eligible,
                "checkpoint": str(checkpoint),
            }
        )
        print(
            f"[resolver053:train] epoch={epoch}/{cfg.epochs}"
            f" gain={validation['resolved']['resolver_agreement_gain']:.4f}"
            f" capture={validation['disagreement']['capture']:.4f}"
            f" fallback_kl={validation['fallback']['mean_kl']:.6f}"
            f" eligible={eligible}",
            flush=True,
        )
    eligible_rows = [row for row in history if row["selection_eligible"]]
    selected = (
        max(
            eligible_rows,
            key=lambda row: (
                float(row["validation"]["resolved"]["resolver_agreement_gain"]),
                -float(row["validation"]["disagreement"]["ranking_loss"]),
                -int(row["epoch"]),
            ),
        )
        if eligible_rows
        else None
    )
    if selected is None:
        payload = {
            "recipe": _recipe(cfg),
            "trainable_parameters": trainable_parameter_count(model),
            "baseline_validation": baseline_validation,
            "history": history,
            "selected_epoch": None,
            "selection_eligible_epochs": [],
            "validation_gates": {"eligible_epoch_exists": False},
            "validation_passed": False,
            "training_seconds": time.perf_counter() - started,
        }
        _atomic_write(path, payload)
        return payload

    selected_checkpoint = Path(selected["checkpoint"])
    selected_model = load_frozen_parent_residual_model(
        cfg.parent_checkpoint,
        cfg.device,
        ResolverResidualConfig(hidden_size=cfg.residual_hidden_size),
    )
    selected_payload = torch.load(
        selected_checkpoint,
        map_location=cfg.device,
        weights_only=True,
    )
    selected_model.load_state_dict(selected_payload["model_state_dict"])
    selected_model.eval()
    selected_train = _evaluate_model(
        cfg,
        selected_model,
        train_eval_loader,
        "selected_train",
    )
    selected_validation = selected["validation"]
    validation_gates = _offline_gates(
        cfg,
        selected_validation,
        selected_train,
        require_bootstrap=False,
    )
    best_path = cfg.output_dir / "frozen_parent_residual_scorer_best.pt"
    save_frozen_parent_residual_checkpoint(
        best_path,
        selected_model,
        {**provenance, "epoch": int(selected["epoch"]), "selected": True},
    )
    payload = {
        "recipe": _recipe(cfg),
        "trainable_parameters": trainable_parameter_count(model),
        "baseline_validation": baseline_validation,
        "history": history,
        "selected_epoch": int(selected["epoch"]),
        "selection_eligible_epochs": [int(row["epoch"]) for row in eligible_rows],
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256": _sha256_file(best_path),
        "selected_train": selected_train,
        "selected_validation": selected_validation,
        "validation_gates": validation_gates,
        "validation_passed": all(validation_gates.values()),
        "training_seconds": time.perf_counter() - started,
    }
    _atomic_write(path, payload)
    return payload


def _train_epoch(
    cfg: ResolverResidualExperimentConfig,
    model: FrozenParentResidualScorerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "ranking": 0.0, "anchor_kl": 0.0, "residual_mse": 0.0}
    batches = 0
    for batch in loader:
        observation = batch["observation"].to(cfg.device)
        actions = batch["legal_action_features"].to(cfg.device)
        mask = batch["legal_action_mask"].to(cfg.device)
        scores, _value, parent_scores, residual = model.forward_with_residual(
            observation,
            actions,
            mask,
        )
        losses = compute_residual_ranking_losses(
            student_scores=scores,
            parent_scores=parent_scores,
            residual=residual,
            legal_mask=mask,
            resolver_rows=batch["target_action_row"],
            parent_rows=batch["parent_action_row"],
            target_available=batch["target_available"],
            ranking_margin=cfg.ranking_margin,
            anchor_kl_weight=cfg.anchor_kl_weight,
            residual_mse_weight=cfg.residual_mse_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        batches += 1
    if not batches:
        raise ValueError("Residual training loader produced zero batches.")
    return {key: value / batches for key, value in totals.items()}


def _evaluate_checkpoint(
    cfg: ResolverResidualExperimentConfig,
    replay: Path,
    checkpoint: Path,
    *,
    split_name: str,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=cfg.device, weights_only=True)
    model = load_frozen_parent_residual_model(
        cfg.parent_checkpoint,
        cfg.device,
        ResolverResidualConfig(hidden_size=cfg.residual_hidden_size),
    )
    model.load_state_dict(payload["model_state_dict"])
    dataset = ResolverResidualReplayDataset(replay, model)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_resolver_residual_rows,
    )
    return _evaluate_model(cfg, model, loader, split_name)


def _evaluate_model(
    cfg: ResolverResidualExperimentConfig,
    model: FrozenParentResidualScorerModel,
    loader: DataLoader,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            observation = batch["observation"].to(cfg.device)
            actions = batch["legal_action_features"].to(cfg.device)
            mask = batch["legal_action_mask"].to(cfg.device)
            scores, _value, parent_scores, residual = model.forward_with_residual(
                observation,
                actions,
                mask,
            )
            masked_parent = parent_scores.masked_fill(~mask, -torch.inf)
            masked_student = scores.masked_fill(~mask, -torch.inf)
            parent_top = masked_parent.argmax(dim=1)
            student_top = masked_student.argmax(dim=1)
            resolver_rows = batch["target_action_row"].to(cfg.device)
            parent_rows = batch["parent_action_row"].to(cfg.device)
            available = batch["target_available"].to(cfg.device)
            row_indices = torch.arange(scores.shape[0], device=cfg.device)
            parent_gap = (
                parent_scores[row_indices, resolver_rows]
                - parent_scores[row_indices, parent_rows]
            )
            student_gap = (
                scores[row_indices, resolver_rows]
                - scores[row_indices, parent_rows]
            )
            parent_log_probs = nn.functional.log_softmax(masked_parent, dim=1)
            student_log_probs = nn.functional.log_softmax(masked_student, dim=1)
            parent_probs = parent_log_probs.exp()
            target = batch["root_visit_target"].to(cfg.device)
            target_log = target.clamp_min(1e-30).log()
            target_kl_parent = (
                target * (target_log - parent_log_probs).masked_fill(~mask, 0.0)
            ).sum(dim=1)
            target_kl_student = (
                target * (target_log - student_log_probs).masked_fill(~mask, 0.0)
            ).sum(dim=1)
            fallback_kl = (
                parent_probs
                * (parent_log_probs - student_log_probs).masked_fill(~mask, 0.0)
            ).sum(dim=1)
            finite = (
                torch.isfinite(scores.masked_fill(~mask, 0.0)).all(dim=1)
                & torch.isfinite(residual).all(dim=1)
                & torch.isfinite(target_kl_student)
            )
            for index, metadata in enumerate(batch["metadata"]):
                resolved = bool(available[index].item())
                ranking = resolved and int(resolver_rows[index]) != int(parent_rows[index])
                records.append(
                    {
                        "game_seed": int(metadata["game_seed"]),
                        "phase": str(metadata["phase"]),
                        "seat": int(metadata["player_id"]),
                        "outcome": str(metadata["outcome_role"]),
                        "resolved": resolved,
                        "ranking": ranking,
                        "parent_top_matches_stored": (
                            int(parent_top[index]) == int(parent_rows[index])
                        ),
                        "parent_agrees_resolver": (
                            int(parent_top[index]) == int(resolver_rows[index])
                        ),
                        "student_agrees_resolver": (
                            int(student_top[index]) == int(resolver_rows[index])
                        ),
                        "student_retains_parent": (
                            int(student_top[index]) == int(parent_rows[index])
                        ),
                        "moved": int(student_top[index]) != int(parent_top[index]),
                        "parent_gap": float(parent_gap[index]),
                        "student_gap": float(student_gap[index]),
                        "directional": float(student_gap[index]) > float(parent_gap[index]),
                        "target_kl_parent": float(target_kl_parent[index]),
                        "target_kl_student": float(target_kl_student[index]),
                        "fallback_kl": float(fallback_kl[index]),
                        "finite": bool(finite[index]),
                    }
                )
    return _summarize_evaluation(
        records,
        split_name=split_name,
        ranking_margin=cfg.ranking_margin,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=cfg.bootstrap_seed,
    )


def _summarize_evaluation(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    ranking_margin: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    resolved = [row for row in records if row["resolved"]]
    disagreement = [row for row in resolved if row["ranking"]]
    agreement = [row for row in resolved if not row["ranking"]]
    fallback = [row for row in records if not row["resolved"]]
    parent_agreement = _mean_bool(resolved, "parent_agrees_resolver")
    student_agreement = _mean_bool(resolved, "student_agrees_resolver")
    parent_kl = _mean_float(resolved, "target_kl_parent")
    student_kl = _mean_float(resolved, "target_kl_student")
    summary = {
        "split": split_name,
        "rows": len(records),
        "resolved": {
            "rows": len(resolved),
            "parent_resolver_agreement": parent_agreement,
            "student_resolver_agreement": student_agreement,
            "resolver_agreement_gain": student_agreement - parent_agreement,
            "parent_target_kl": parent_kl,
            "student_target_kl": student_kl,
            "target_kl_ratio": (
                student_kl / parent_kl if parent_kl > 0.0 else math.inf
            ),
        },
        "disagreement": {
            "rows": len(disagreement),
            "capture": _mean_bool(disagreement, "student_agrees_resolver"),
            "directional_movement": _mean_bool(disagreement, "directional"),
            "mean_parent_gap": _mean_float(disagreement, "parent_gap"),
            "mean_student_gap": _mean_float(disagreement, "student_gap"),
            "mean_gap_delta": (
                statistics.fmean(
                    row["student_gap"] - row["parent_gap"] for row in disagreement
                )
                if disagreement
                else 0.0
            ),
            "ranking_loss": (
                statistics.fmean(
                    math.log1p(math.exp(ranking_margin - row["student_gap"]))
                    for row in disagreement
                )
                if disagreement
                else math.inf
            ),
        },
        "agreement_anchor": {
            "rows": len(agreement),
            "retention": _mean_bool(agreement, "student_retains_parent"),
        },
        "fallback": {
            "rows": len(fallback),
            "retention": _mean_bool(fallback, "student_retains_parent"),
            "mean_kl": _mean_float(fallback, "fallback_kl"),
        },
        "action_movement": _mean_bool(records, "moved"),
        "strata": {
            **{
                f"phase:{phase}": _resolver_stratum(
                    [row for row in resolved if row["phase"] == phase]
                )
                for phase in ("early", "mid", "late")
            },
            **{
                f"seat:{seat}": _resolver_stratum(
                    [row for row in resolved if row["seat"] == seat]
                )
                for seat in (0, 1)
            },
            **{
                f"outcome:{outcome}": _resolver_stratum(
                    [row for row in resolved if row["outcome"] == outcome]
                )
                for outcome in ("winner", "loser")
            },
        },
        "bootstrap_resolver_agreement_gain_ci95": _bootstrap_gain(
            resolved,
            bootstrap_samples,
            bootstrap_seed,
        ),
        "integrity": {
            "parent_argmax_reproduced": all(
                row["parent_top_matches_stored"] for row in records
            ),
            "finite": all(row["finite"] for row in records),
        },
    }
    return summary


def _resolver_stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent = _mean_bool(rows, "parent_agrees_resolver")
    student = _mean_bool(rows, "student_agrees_resolver")
    return {
        "rows": len(rows),
        "parent_resolver_agreement": parent,
        "student_resolver_agreement": student,
        "resolver_agreement_gain": student - parent,
    }


def _selection_eligible(
    cfg: ResolverResidualExperimentConfig,
    validation: dict[str, Any],
) -> bool:
    return (
        validation["fallback"]["rows"] > 0
        and validation["fallback"]["mean_kl"] <= cfg.maximum_fallback_kl
        and validation["fallback"]["retention"] >= cfg.minimum_anchor_retention
        and validation["agreement_anchor"]["retention"]
        >= cfg.minimum_anchor_retention
        and all(validation["integrity"].values())
    )


def _offline_gates(
    cfg: ResolverResidualExperimentConfig,
    evaluation: dict[str, Any],
    train: dict[str, Any],
    *,
    require_bootstrap: bool,
) -> dict[str, bool]:
    stratum_values = list(evaluation["strata"].values())
    gates = {
        "resolver_agreement_gain": (
            evaluation["resolved"]["resolver_agreement_gain"]
            >= cfg.minimum_resolver_agreement_gain
        ),
        "disagreement_capture": (
            evaluation["disagreement"]["capture"]
            >= cfg.minimum_disagreement_capture
        ),
        "directional_movement": (
            evaluation["disagreement"]["directional_movement"]
            >= cfg.minimum_directional_movement
        ),
        "agreement_anchor_retention": (
            evaluation["agreement_anchor"]["retention"]
            >= cfg.minimum_anchor_retention
        ),
        "fallback_anchor_retention": (
            evaluation["fallback"]["retention"] >= cfg.minimum_anchor_retention
        ),
        "fallback_anchor_kl": (
            evaluation["fallback"]["mean_kl"] <= cfg.maximum_fallback_kl
        ),
        "bounded_action_movement": (
            cfg.minimum_action_movement
            <= evaluation["action_movement"]
            <= cfg.maximum_action_movement
        ),
        "complete_target_kl_not_worse": (
            evaluation["resolved"]["target_kl_ratio"]
            <= cfg.maximum_target_kl_ratio
        ),
        "no_stratum_collapse": all(
            int(row["rows"]) >= cfg.minimum_stratum_rows
            and float(row["resolver_agreement_gain"]) >= 0.0
            for row in stratum_values
        ),
        "bounded_train_validation_gap": (
            train["disagreement"]["capture"]
            - evaluation["disagreement"]["capture"]
            <= cfg.maximum_train_validation_capture_gap
        ),
        "integrity": all(evaluation["integrity"].values()),
    }
    if require_bootstrap:
        gates["bootstrap_gain_lower_above_zero"] = (
            evaluation["bootstrap_resolver_agreement_gain_ci95"]["lower"] > 0.0
        )
    return gates


def _bootstrap_gain(
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not rows:
        return {"lower": -math.inf, "upper": math.inf}
    groups: dict[int, list[float]] = {}
    for row in rows:
        groups.setdefault(int(row["game_seed"]), []).append(
            float(row["student_agrees_resolver"])
            - float(row["parent_agrees_resolver"])
        )
    clusters = [(len(values), sum(values)) for values in groups.values()]
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values.append(
            sum(item[1] for item in chosen) / sum(item[0] for item in chosen)
        )
    values.sort()
    return {
        "lower": _quantile(values, 0.025),
        "upper": _quantile(values, 0.975),
    }


def _run_gameplay_block(
    cfg: ResolverResidualExperimentConfig,
    name: str,
    opponent: Opponent,
    seed_start: int,
    pairs: int,
) -> dict[str, Any]:
    path = cfg.output_dir / f"{name}.json"
    progress_path = cfg.output_dir / f"{name}.progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = {int(row["seed"]): row for row in progress.get("pairs", [])}
    seeds = list(range(seed_start, seed_start + pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(f"[resolver053:{name}] pairs={pairs} complete={len(completed)}", flush=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {
                executor.submit(_play_residual_pair, cfg, opponent, seed): seed
                for seed in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                completed[seed] = future.result()
                _atomic_write(
                    progress_path,
                    {
                        "completed": False,
                        "pairs": [completed[key] for key in sorted(completed)],
                    },
                )
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver053:{name}] complete={len(completed)}/{pairs}"
                    f" eta_seconds={eta:.0f}",
                    flush=True,
                )
    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, cfg.bootstrap_samples, seed_start)
    summary["seat_breakdown"] = _seat_breakdown(rows)
    payload = {
        "name": name,
        "opponent": opponent,
        "seed_start": seed_start,
        "pairs_count": pairs,
        "summary": summary,
        "integrity": _integrity(rows),
        "pairs": rows,
    }
    _atomic_write(path, payload)
    _atomic_write(progress_path, {"completed": True, "pairs": rows})
    return payload


def _play_residual_pair(
    cfg: ResolverResidualExperimentConfig,
    opponent: Opponent,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    games = []
    for candidate_seat in (0, 1):
        candidate = FrozenParentResidualScorerBot(
            cfg.output_dir / "frozen_parent_residual_scorer_best.pt",
            cfg.device,
        )
        baseline = _gameplay_opponent(cfg, opponent)
        bots = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
        games.append(
            asdict(
                play_game(
                    bots[0],
                    bots[1],
                    seed=seed,
                    max_turns=400,
                    repetition_limit=4,
                    no_progress_limit=60,
                )
            )
        )
    return {"seed": seed, "candidate_seats": [0, 1], "games": games}


def _gameplay_opponent(
    cfg: ResolverResidualExperimentConfig,
    opponent: Opponent,
) -> Any:
    if opponent == "035":
        return LegalActionAttentionScorerBot(cfg.parent_checkpoint, cfg.device)
    if opponent == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    if opponent == "016":
        return LegalActionScorerBot(cfg.checkpoint_016, cfg.device)
    if opponent == "023":
        return CanonicalEntityActionScorerBot(cfg.checkpoint_023, cfg.device)
    raise ValueError(opponent)


def _gameplay_gates(
    cfg: ResolverResidualExperimentConfig,
    block: dict[str, Any],
) -> dict[str, bool]:
    summary = block["summary"]
    integrity = block["integrity"]
    return {
        "exact_clean_games": (
            integrity["games"] == int(block["pairs_count"]) * 2
            and integrity["invalid_games"] == 0
        ),
        "zero_timeouts_and_stalls": (
            integrity["timed_out_games"] == 0 and integrity["stalled_games"] == 0
        ),
        "score_share_at_least_55pct": (
            float(summary["score_share"]["mean"]) >= cfg.minimum_score_share
        ),
        "score_share_ci95_lower_above_50pct": (
            float(summary["score_share"]["ci95_low"]) > 0.50
        ),
        "positive_mean_margin": (
            float(summary["final_score_margin"]["mean"]) > 0.0
        ),
        "no_seat_collapse": all(
            float(item["score_share"]) >= cfg.minimum_seat_score_share
            for item in summary["seat_breakdown"].values()
        ),
    }


def _validate_config(cfg: ResolverResidualExperimentConfig) -> None:
    frozen = {
        cfg.audit_052a: AUDIT_052A_SHA256,
        cfg.train_replay: TRAIN_SHA256,
        cfg.validation_replay: VALIDATION_SHA256,
        cfg.parent_checkpoint: PARENT_SHA256,
        cfg.checkpoint_016: CHECKPOINT_016_SHA256,
        cfg.checkpoint_023: CHECKPOINT_023_SHA256,
    }
    for path, expected in frozen.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Frozen 053 prerequisite mismatch: {path}")
    audit = json.loads(cfg.audit_052a.read_text(encoding="utf-8"))
    if not audit.get("integrity_passed"):
        raise ValueError("053 requires the integrity-valid 052a diagnostic.")
    frozen_contract = {
        "residual_hidden_size": (cfg.residual_hidden_size, 64),
        "ranking_margin": (cfg.ranking_margin, 0.25),
        "anchor_kl_weight": (cfg.anchor_kl_weight, 5.0),
        "residual_mse_weight": (cfg.residual_mse_weight, 0.01),
        "learning_rate": (cfg.learning_rate, 3e-4),
        "weight_decay": (cfg.weight_decay, 1e-4),
        "batch_size": (cfg.batch_size, 128),
        "epochs": (cfg.epochs, 12),
        "shuffle_seed": (cfg.shuffle_seed, 53),
        "fresh_confirmation_seed_start": (
            cfg.fresh_confirmation_seed_start,
            620_300,
        ),
        "fresh_confirmation_games": (cfg.fresh_confirmation_games, 20),
        "direct_seed_start": (cfg.direct_seed_start, 620_500),
        "direct_pairs": (cfg.direct_pairs, 100),
        "broad_pairs": (cfg.broad_pairs, 50),
        "bootstrap_samples": (cfg.bootstrap_samples, 20_000),
        "bootstrap_seed": (cfg.bootstrap_seed, 530_053),
        "device": (cfg.device, "cpu"),
    }
    changed = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in frozen_contract.items()
        if observed != expected
    }
    if changed:
        raise ValueError(f"053 frozen configuration changed: {changed}")


def _identity(cfg: ResolverResidualExperimentConfig) -> dict[str, Any]:
    return {
        "configuration": _config_payload(cfg),
        "sha256": {
            "audit_052a": _sha256_file(cfg.audit_052a),
            "train_replay": _sha256_file(cfg.train_replay),
            "validation_replay": _sha256_file(cfg.validation_replay),
            "parent_035": _sha256_file(cfg.parent_checkpoint),
            "checkpoint_016": _sha256_file(cfg.checkpoint_016),
            "checkpoint_023": _sha256_file(cfg.checkpoint_023),
        },
    }


def _checkpoint_provenance(
    cfg: ResolverResidualExperimentConfig,
) -> dict[str, Any]:
    return {
        "experiment": "053_frozen_parent_resolver_residual_ranking",
        "protocol": "docs/experiments/resolver_residual_ranking_distillation_053.md",
        "parent_sha256": PARENT_SHA256,
        "train_sha256": TRAIN_SHA256,
        "validation_sha256": VALIDATION_SHA256,
        "recipe": _recipe(cfg),
    }


def _recipe(cfg: ResolverResidualExperimentConfig) -> dict[str, Any]:
    return {
        "architecture": "zero_initialized_centered_385x64x1_residual",
        "parent_frozen": True,
        "ranking_margin": cfg.ranking_margin,
        "anchor_kl_weight": cfg.anchor_kl_weight,
        "residual_mse_weight": cfg.residual_mse_weight,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "shuffle_seed": cfg.shuffle_seed,
        "device": cfg.device,
    }


def _finish(
    cfg: ResolverResidualExperimentConfig,
    *,
    decision: str,
    passed: bool = False,
    **stages: Any,
) -> dict[str, Any]:
    payload = {
        "experiment": "053_frozen_parent_resolver_residual_ranking",
        "protocol": "docs/experiments/resolver_residual_ranking_distillation_053.md",
        "identity": _identity(cfg),
        **stages,
        "passed": passed,
        "decision": decision,
        "gpu_rental_authorized": False,
        "next_generation_authorized": passed,
    }
    _atomic_write(cfg.output_dir / "result.json", payload)
    print(
        f"053 result: {'PASS' if passed else 'FAIL'} decision={decision}",
        flush=True,
    )
    return payload


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(bool(row[key]) for row in rows) if rows else 0.0


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


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


def _config_payload(cfg: ResolverResidualExperimentConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }


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
