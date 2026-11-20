"""Identity-preserving medium-capacity distillation pilot 116."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from splendor_ai.training.model import CanonicalEntityActionScorerConfig

from .arena import ArenaResult, alpha_zero_search_factory, run_paired_arena
from .capacity_network import (
    capacity_parameter_counts,
    create_identity_capacity_model,
    load_capacity_checkpoint,
    save_capacity_checkpoint,
)
from .endpoint_evaluation import practical_assessment
from .indexed_training import (
    IndexedReplayDataset,
    index_replay_paths,
    make_indexed_loader,
    split_index_by_game,
)
from .network import load_checkpoint
from .scaled_continuation import _sha256, _write_json
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .training import TrainingConfig, _compare_policies, _run_epoch


@dataclass(frozen=True, slots=True)
class CapacityPilotConfig:
    output_dir: Path
    result_114: Path
    result_115: Path
    partition_manifest: Path
    train_replay: Path
    untouched_holdout: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None
    arena_workers: int = 4


PROTOCOL_SHA256 = "827ad3eabe8557740f79cad177af12522948509d94e3a278f36f7b97af88e0e1"
EXPECTED_HASHES = {
    "result_114": "dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5",
    "result_115": "02d4d1d2557c57c6778b92950fb4c06d02e4112dc83d45d20791f638df0db993",
    "partition_manifest": "61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc",
    "train_replay": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "untouched_holdout": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
}
PRIMARY_SEED_START = 2_160_000
PRIMARY_PAIRED_SEEDS = 200
ARENA_SHARDS = 4
SHARD_PAIRED_SEEDS = 50
SIMULATIONS = 256
BOOTSTRAP_SAMPLES = 50_000


def residual_model_config() -> CanonicalEntityActionScorerConfig:
    return CanonicalEntityActionScorerConfig(
        entity_hidden_size=128,
        action_hidden_size=128,
        scorer_hidden_size=192,
        entity_layers=2,
        action_layers=2,
        attention_heads=4,
        attention_feedforward_size=256,
        id_embedding_size=24,
        dropout=0.0,
    )


def capacity_training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=12,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=1_116_000,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        bucket_batches=True,
        policy_divergence_weight=0.0,
        visit_argmax_aux_weight=0.0,
    )


def offline_qualification(
    *,
    training: dict[str, Any],
    parent: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, Any]:
    deltas = {
        key: float(candidate[key]) - float(parent[key])
        for key in ("policy_loss", "value_loss", "objective_loss")
    }
    identity = training["identity_audit"]
    counts = training["parameter_counts"]
    checks = {
        "epoch_zero_exact_logits": float(identity["max_logit_abs_diff"]) <= 1e-7,
        "epoch_zero_exact_values": float(identity["max_value_abs_diff"]) <= 1e-7,
        "parent_frozen": bool(training["parent_frozen"]),
        "capacity_at_least_2_5x_parent": float(counts["total"]) >= 2.5 * float(counts["parent"]),
        "nonzero_epoch_selected": int(training.get("selected_epoch", 0)) > 0,
        "holdout_policy_improved": deltas["policy_loss"] < 0.0,
        "holdout_objective_improved": deltas["objective_loss"] < 0.0,
        "holdout_value_regression_at_most_0_025": deltas["value_loss"] <= 0.025,
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for metrics in (parent, candidate)
            for value in metrics.values()
            if isinstance(value, (int, float))
        ),
    }
    return {"qualified": all(checks.values()), "checks": checks, "deltas": deltas}


def gameplay_assessment(
    arena: dict[str, Any], bootstrap: dict[str, float | int]
) -> dict[str, Any]:
    practical = practical_assessment(arena)
    checks = {
        **practical["checks"],
        "paired_ci95_low_strictly_above_half": float(bootstrap["ci95_low"]) > 0.5,
    }
    return {
        "passed": all(checks.values()),
        "category": "medium_capacity_compression_confirmed"
        if all(checks.values())
        else "medium_capacity_compression_not_confirmed",
        "checks": checks,
        "seat_score_shares": practical["seat_score_shares"],
    }


def capacity_decision(*, qualified: bool, gameplay_passed: bool | None) -> str:
    if not qualified:
        return "identity_capacity_offline_gate_failed_review_objective_or_bridge_no_install"
    if gameplay_passed is None:
        raise ValueError("Qualified capacity candidate requires ordinary-budget gameplay.")
    if gameplay_passed:
        return "identity_capacity_compression_confirmed_review_short_continuation_no_install"
    return "identity_capacity_offline_gain_not_converted_review_representation_design_no_install"


def run_capacity_pilot_116(config: CapacityPilotConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed = config.output_dir / "result.json"
    if sealed.exists():
        payload = json.loads(sealed.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 116 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_identity_capacity_pilot_116",
        "status": "running",
        "active_stage": "identity_preserving_capacity_training",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "training": None,
        "untouched_holdout": None,
        "offline_qualified": False,
        "primary_256": None,
        "selfplay_generated": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(config.output_dir, result, started)
    training = _run_or_load_training(config)
    result["training"] = training
    result["active_stage"] = "untouched_holdout_gate"
    _write_progress(config.output_dir, result, started)
    audit = _run_or_load_holdout(config, training)
    result["untouched_holdout"] = audit
    result["offline_qualified"] = bool(audit["qualification"]["qualified"])
    if not result["offline_qualified"]:
        result["decision"] = capacity_decision(qualified=False, gameplay_passed=None)
        return _seal(config.output_dir, result, started)
    result["active_stage"] = "four_shard_primary_256"
    _write_progress(config.output_dir, result, started)
    arena = combine_primary_shards(_run_or_load_parallel_arena(config, training))
    bootstrap = paired_score_share_bootstrap(
        arena, samples=BOOTSTRAP_SAMPLES, seed=1_116_100
    )
    assessment = gameplay_assessment(arena, bootstrap)
    result["primary_256"] = {
        "arena": arena,
        "paired_bootstrap": bootstrap,
        "assessment": assessment,
    }
    result["decision"] = capacity_decision(
        qualified=True, gameplay_passed=bool(assessment["passed"])
    )
    return _seal(config.output_dir, result, started)


def _run_or_load_training(config: CapacityPilotConfig) -> dict[str, Any]:
    stage = config.output_dir / "training" / "capacity.stage.json"
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("parent_checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
            or payload.get("train_replay_sha256") != EXPECTED_HASHES["train_replay"]
            or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
            or payload.get("metrics_sha256") != _sha256(payload["metrics_path"])
        ):
            raise ValueError("116 training stage identity mismatch.")
        metrics = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
        return {**payload, "training": metrics}

    cfg = capacity_training_config()
    cfg.validate()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    paths, entries = index_replay_paths([config.train_replay])
    train_entries, validation_entries = split_index_by_game(
        entries, validation_fraction=cfg.validation_fraction, seed=cfg.seed
    )
    train_loader = make_indexed_loader(paths, train_entries, config=cfg, shuffle=True)
    validation_loader = make_indexed_loader(
        paths, validation_entries, config=cfg, shuffle=False
    )
    model, parent_identity = create_identity_capacity_model(
        config.generation_11_checkpoint,
        residual_config=residual_model_config(),
        device=config.device,
    )
    parent = model.parent
    parent.eval()
    identity = _identity_audit(parent, model, validation_loader, config.device)
    counts = capacity_parameter_counts(model)
    parent_train_metrics = _run_epoch(parent, train_loader, config.device, None, cfg)
    parent_validation_metrics = _run_epoch(parent, validation_loader, config.device, None, cfg)
    best_state = deepcopy(model.residual.state_dict())
    best_validation = float(parent_validation_metrics["objective_loss"])
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    optimizer = torch.optim.AdamW(
        model.residual.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    stopped_early = False
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_metrics = _run_epoch(model, train_loader, config.device, optimizer, cfg)
        model.eval()
        validation_metrics = _run_epoch(model, validation_loader, config.device, None, cfg)
        if validation_metrics["objective_loss"] < best_validation - cfg.early_stopping_min_delta:
            best_validation = float(validation_metrics["objective_loss"])
            best_state = deepcopy(model.residual.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if (
            cfg.early_stopping_patience is not None
            and epochs_without_improvement >= cfg.early_stopping_patience
        ):
            stopped_early = True
            break
    model.residual.load_state_dict(best_state)
    model.eval()
    child_train_metrics = _run_epoch(model, train_loader, config.device, None, cfg)
    child_validation_metrics = _run_epoch(model, validation_loader, config.device, None, cfg)
    comparison = _compare_policies(parent, model, validation_loader, config.device)
    train_games = sorted({f"{entry.generation}:{entry.game_seed}" for entry in train_entries})
    validation_games = sorted(
        {f"{entry.generation}:{entry.game_seed}" for entry in validation_entries}
    )
    checkpoint = config.output_dir / "checkpoints" / "capacity_generation_012.pt"
    save_capacity_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "parent_generation": parent_identity.generation,
            "replay_path": str(config.train_replay),
            "replay_sha256": EXPECTED_HASHES["train_replay"],
            "training_config": asdict(cfg),
            "identity_preserving_initialization": True,
            "frozen_parent": True,
            "parent_safe_selection": True,
            "selected_epoch": best_epoch,
            "train_games": train_games,
            "validation_games": validation_games,
        },
    )
    roundtrip, checkpoint_identity = load_capacity_checkpoint(checkpoint, device=config.device)
    if (
        checkpoint_identity.generation != 12
        or checkpoint_identity.metadata.get("parent_generation") != 11
        or checkpoint_identity.metadata.get("selected_epoch") != best_epoch
        or any(parameter.requires_grad for parameter in roundtrip.parent.parameters())
    ):
        raise RuntimeError("116 capacity checkpoint failed round-trip lineage checks.")
    metrics = {
        "parent_checkpoint": str(config.generation_11_checkpoint),
        "parent_generation": parent_identity.generation,
        "child_generation": 12,
        "rows": len(entries),
        "train_rows": len(train_entries),
        "validation_rows": len(validation_entries),
        "train_games": len(train_games),
        "validation_games": len(validation_games),
        "identity_audit": identity,
        "parameter_counts": counts,
        "parent_frozen": all(not parameter.requires_grad for parameter in model.parent.parameters()),
        "best_validation_loss": best_validation,
        "selection_metric": "objective_loss_with_exact_epoch_zero_parent_eligible",
        "selected_epoch": best_epoch,
        "selected_parent_baseline": best_epoch == 0,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "parent_train_metrics": parent_train_metrics,
        "parent_validation_metrics": parent_validation_metrics,
        "child_train_metrics": child_train_metrics,
        "child_validation_metrics": child_validation_metrics,
        "validation_loss_delta_child_minus_parent": {
            key: child_validation_metrics[key] - parent_validation_metrics[key]
            for key in ("policy_loss", "value_loss", "total_loss", "objective_loss")
        },
        "child_parent_policy_comparison": comparison,
        "history": history,
        "config": asdict(cfg),
        "indexed_loading": True,
        "parent_safe_selection": True,
    }
    metrics_path = config.output_dir / "training" / "capacity.json"
    _write_json(metrics_path, metrics)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "train_replay_sha256": EXPECTED_HASHES["train_replay"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
    }
    _write_json(stage, payload)
    _close_loaders(train_loader, validation_loader)
    return {**payload, "training": metrics}


def _run_or_load_holdout(
    config: CapacityPilotConfig, training: dict[str, Any]
) -> dict[str, Any]:
    stage = config.output_dir / "audit" / "untouched_holdout.stage.json"
    hashes = {
        "parent": EXPECTED_HASHES["generation_11_checkpoint"],
        "candidate": training["checkpoint_sha256"],
    }
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("holdout_sha256") != EXPECTED_HASHES["untouched_holdout"]
            or payload.get("checkpoint_hashes") != hashes
        ):
            raise ValueError("116 untouched holdout stage identity mismatch.")
        return payload
    cfg = capacity_training_config()
    paths, entries = index_replay_paths([config.untouched_holdout])
    loader = make_indexed_loader(paths, entries, config=cfg, shuffle=False)
    parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)[0]
    candidate = load_capacity_checkpoint(training["checkpoint"], device=config.device)[0]
    parent_metrics = _run_epoch(parent, loader, config.device, None, cfg)
    candidate_metrics = _run_epoch(candidate, loader, config.device, None, cfg)
    qualification = offline_qualification(
        training=training["training"],
        parent=parent_metrics,
        candidate=candidate_metrics,
    )
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "holdout_path": str(config.untouched_holdout),
        "holdout_sha256": EXPECTED_HASHES["untouched_holdout"],
        "rows": len(entries),
        "checkpoint_hashes": hashes,
        "parent": parent_metrics,
        "candidate": candidate_metrics,
        "qualification": qualification,
    }
    _write_json(stage, payload)
    _close_loaders(loader)
    return payload


def _identity_audit(parent, candidate, loader, device: str) -> dict[str, float | int]:
    rows = 0
    max_logit = 0.0
    max_value = 0.0
    parent.eval()
    candidate.eval()
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            inputs = _model_inputs(moved)
            parent_logits, parent_values = parent(**inputs)
            candidate_logits, candidate_values = candidate(**inputs)
            mask = moved["action_mask"]
            max_logit = max(
                max_logit,
                float(torch.abs(parent_logits[mask] - candidate_logits[mask]).max().item()),
            )
            max_value = max(
                max_value,
                float(torch.abs(parent_values - candidate_values).max().item()),
            )
            rows += int(mask.shape[0])
    return {
        "rows": rows,
        "max_logit_abs_diff": max_logit,
        "max_value_abs_diff": max_value,
    }


def _run_or_load_parallel_arena(
    config: CapacityPilotConfig, training: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    stage = config.output_dir / "arenas" / "primary_256.stage.json"
    candidate_hash = training["checkpoint_sha256"]
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("candidate_checkpoint_sha256") != candidate_hash
            or payload.get("baseline_checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
            or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
        ):
            raise ValueError("116 primary arena stage identity mismatch.")
        shards = {int(index): arena for index, arena in payload["shards"].items()}
        combine_primary_shards(shards)
        return shards
    tasks = [
        {
            "index": index,
            "output_dir": str(config.output_dir / "arenas"),
            "candidate_checkpoint": training["checkpoint"],
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
        }
        for index in range(ARENA_SHARDS)
    ]
    shards: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
        futures = {executor.submit(_arena_worker, task): task for task in tasks}
        for future in as_completed(futures):
            index, arena = future.result()
            shards[index] = arena
    combine_primary_shards(shards)
    _write_json(
        stage,
        {
            "protocol_sha256": PROTOCOL_SHA256,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "arena_workers": config.arena_workers,
            "shards": {str(index): shards[index] for index in sorted(shards)},
        },
    )
    return shards


def _arena_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    _limit_worker_threads()
    index = int(task["index"])
    search = _search_config()
    candidate = alpha_zero_search_factory(task["candidate_checkpoint"], task["device"], search)
    baseline = alpha_zero_search_factory(task["baseline_checkpoint"], task["device"], search)
    seed_start = PRIMARY_SEED_START + index * SHARD_PAIRED_SEEDS
    arena_dir = Path(task["output_dir"])
    progress = arena_dir / f"primary_256_shard_{index}.json"
    sidecar = arena_dir / f"primary_256_shard_{index}.identity.json"
    identity = _arena_identity(index, task["candidate_checkpoint_sha256"], int(task["max_turns"]))
    if sidecar.exists():
        if json.loads(sidecar.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"116 arena shard {index} identity mismatch.")
    else:
        if progress.exists():
            raise ValueError(f"116 arena shard {index} progress exists without identity.")
        _write_json(sidecar, identity)
    arena = run_paired_arena(
        name=f"identity_capacity_primary_256_shard_{index}",
        candidate_factory=candidate,
        baseline_factory=baseline,
        seed_start=seed_start,
        paired_seeds=SHARD_PAIRED_SEEDS,
        max_turns=int(task["max_turns"]),
        progress_path=progress,
        resume=True,
    )
    return index, _arena_dict(arena)


def combine_primary_shards(shards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(shards) != set(range(ARENA_SHARDS)):
        raise ValueError("116 requires all four exact arena shards.")
    rows: list[dict[str, object]] = []
    for index in range(ARENA_SHARDS):
        arena = shards[index]
        start = PRIMARY_SEED_START + index * SHARD_PAIRED_SEEDS
        expected = [
            (start + offset, seat)
            for offset in range(SHARD_PAIRED_SEEDS)
            for seat in (0, 1)
        ]
        observed = [
            (int(row["seed"]), int(row["candidate_seat"]))
            for row in arena.get("rows", [])
        ]
        if (
            arena.get("paired_seeds") != SHARD_PAIRED_SEEDS
            or arena.get("games") != 2 * SHARD_PAIRED_SEEDS
            or observed != expected
        ):
            raise ValueError(f"116 arena shard {index} is not its exact schedule.")
        rows.extend(arena["rows"])
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    terminations: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        terminations[reason] = terminations.get(reason, 0) + 1
    return _arena_dict(
        ArenaResult(
            name="identity_capacity_primary_256",
            paired_seeds=PRIMARY_PAIRED_SEEDS,
            games=len(rows),
            candidate_wins=wins,
            candidate_losses=losses,
            draws=draws,
            score_share=(wins + 0.5 * draws) / len(rows),
            mean_margin=sum(float(row["candidate_margin"]) for row in rows) / len(rows),
            termination_reasons=dict(sorted(terminations.items())),
            rows=tuple(rows),
        )
    )


def _search_config() -> SearchConfig:
    return SearchConfig(
        simulations=SIMULATIONS,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_116_100,
    )


def _arena_identity(index: int, candidate_hash: str, max_turns: int) -> dict[str, Any]:
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard_index": index,
        "candidate_checkpoint_sha256": candidate_hash,
        "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "candidate_search_config": asdict(_search_config()),
        "baseline_search_config": asdict(_search_config()),
        "operative_search_seed": "2 * game_seed + seat",
        "seed_start": PRIMARY_SEED_START + index * SHARD_PAIRED_SEEDS,
        "paired_seeds": SHARD_PAIRED_SEEDS,
        "max_turns": max_turns,
    }


def _verify_inputs(config: CapacityPilotConfig) -> None:
    paths = {
        "result_114": config.result_114,
        "result_115": config.result_115,
        "partition_manifest": config.partition_manifest,
        "train_replay": config.train_replay,
        "untouched_holdout": config.untouched_holdout,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 116 input hash mismatch for {name}: {actual}")
    result_115 = json.loads(config.result_115.read_text(encoding="utf-8"))
    pooled = result_115.get("pooled_primary", {})
    if (
        result_115.get("status") != "complete"
        or result_115.get("decision")
        != "pooled_high_budget_distillation_not_confirmed_proceed_medium_capacity_pilot_no_install"
        or pooled.get("assessment", {}).get("passed") is not False
        or (pooled.get("arena", {}).get("candidate_wins"), pooled.get("arena", {}).get("draws"), pooled.get("arena", {}).get("candidate_losses"))
        != (208, 1, 191)
        or result_115.get("promotion_authorized") is not False
    ):
        raise ValueError("Frozen 115 result does not authorize exact capacity pilot 116.")
    manifest = json.loads(config.partition_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("train_sha256") != EXPECTED_HASHES["train_replay"]
        or manifest.get("holdout_sha256") != EXPECTED_HASHES["untouched_holdout"]
        or len(manifest.get("train_seeds", [])) != 224
        or len(manifest.get("holdout_seeds", [])) != 32
        or set(manifest["train_seeds"]) & set(manifest["holdout_seeds"])
    ):
        raise ValueError("Frozen 114 corpus partition is not exact and disjoint.")
    _, parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    if parent.generation != 11 or parent.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen 116 parent is not exact generation 11.")
    if config.max_turns != 200 or config.torch_threads != 1 or config.arena_workers != 4:
        raise ValueError("116 requires the frozen 200-turn, one-thread, four-worker execution.")


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def _close_loaders(*loaders) -> None:
    for loader in loaders:
        if loader is not None and isinstance(loader.dataset, IndexedReplayDataset):
            loader.dataset.close()


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config: CapacityPilotConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )


def _seal(output: Path, result: dict[str, Any], started: float) -> dict[str, Any]:
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result
