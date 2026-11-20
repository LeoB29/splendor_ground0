"""One-generation high-budget self-play distillation pilot 114."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import torch

from .arena import ArenaResult, alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .indexed_training import (
    IndexedReplayDataset,
    index_replay_paths,
    make_indexed_loader,
    train_parent_safe_indexed_successor,
)
from .network import load_checkpoint
from .replay import _row_from_payload
from .scaled_continuation import _sha256, _write_json
from .scaled_loop import training_validity
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .selfplay import SelfPlayConfig, generate_self_play
from .training import TrainingConfig, _compare_policies, _run_epoch


@dataclass(frozen=True, slots=True)
class HighBudgetDistillationConfig:
    output_dir: Path
    result_113: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None
    shard_workers: int = 4
    arena_workers: int = 4


PROTOCOL_SHA256 = "a630009335be37d1293b8400220b8b5e6e350cb4aea289abf3556344f7bf0e49"
EXPECTED_HASHES = {
    "result_113": "aea9e5a69fa91669ff3ee05d66d9d912b4d5920c90ca2944f5de7003dab18f8d",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
}
SHARD_SEED_STARTS = (2_090_000, 2_100_000, 2_110_000, 2_120_000)
GAMES_PER_SHARD = 64
TRAIN_GAMES_PER_SHARD = 56
HOLDOUT_GAMES_PER_SHARD = 8
MAX_REPLACEMENTS_PER_SHARD = 16
SELFPLAY_SIMULATIONS = 1_024
SELFPLAY_SEARCH_SEED = 1_114_000
PRIMARY_SEED_START = 2_130_000
PRIMARY_PAIRED_SEEDS = 200
DIAGNOSTIC_SEED_START = 2_140_000
DIAGNOSTIC_PAIRED_SEEDS = 100
ARENA_SHARDS = 4
BOOTSTRAP_SAMPLES = 50_000
_GAME_SEED_PATTERN = re.compile(r'"game_seed"\s*:\s*(\d+)')


def distillation_training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=12,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=1_114_000,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        bucket_batches=True,
        policy_divergence_weight=0.0,
        visit_argmax_aux_weight=0.0,
    )


def distillation_qualification(
    *, training: dict[str, Any], parent: dict[str, float], candidate: dict[str, float]
) -> dict[str, Any]:
    deltas = {
        key: float(candidate[key]) - float(parent[key])
        for key in ("policy_loss", "value_loss", "objective_loss")
    }
    checks = {
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


def compression_assessment(
    arena: dict[str, Any], bootstrap: dict[str, float | int]
) -> dict[str, Any]:
    practical = practical_assessment(arena)
    checks = {
        **practical["checks"],
        "paired_ci95_low_strictly_above_half": float(bootstrap["ci95_low"]) > 0.5,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "category": "compressed_teacher_strength_pass" if passed else "compression_not_established",
        "checks": checks,
        "seat_score_shares": practical["seat_score_shares"],
    }


def high_budget_distillation_decision(
    *, qualified: bool, primary_passed: bool | None, diagnostic_passed: bool | None
) -> str:
    if not qualified:
        return "high_budget_distillation_offline_gate_failed_proceed_capacity_optimization_diagnosis_no_install"
    if primary_passed is None or diagnostic_passed is None:
        raise ValueError("Qualified 114 requires both frozen gameplay panels.")
    if primary_passed and diagnostic_passed:
        return "high_budget_teacher_absorbed_at_both_budgets_review_short_continuation_no_install"
    if primary_passed:
        return "high_budget_teacher_compressed_primary_pass_high_budget_interaction_unresolved_no_install"
    if diagnostic_passed:
        return "high_budget_gain_requires_expensive_search_review_representation_capacity_no_install"
    return "high_budget_teacher_not_converted_to_gameplay_proceed_capacity_pilot_no_install"


def partition_seed_schedule(shard_seed_lists: list[list[int]]) -> dict[str, list[int]]:
    if len(shard_seed_lists) != 4:
        raise ValueError("114 partition requires exactly four shards.")
    train: list[int] = []
    holdout: list[int] = []
    for seeds in shard_seed_lists:
        if len(seeds) != GAMES_PER_SHARD or len(set(seeds)) != GAMES_PER_SHARD:
            raise ValueError("Every 114 shard must contain 64 unique clean games.")
        train.extend(seeds[:TRAIN_GAMES_PER_SHARD])
        holdout.extend(seeds[-HOLDOUT_GAMES_PER_SHARD:])
    if len(set(train)) != 224 or len(set(holdout)) != 32 or set(train) & set(holdout):
        raise RuntimeError("114 train/holdout partition is not exact and disjoint.")
    return {"train": sorted(train), "holdout": sorted(holdout)}


def run_high_budget_distillation_114(config: HighBudgetDistillationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed = output / "result.json"
    if sealed.exists():
        payload = json.loads(sealed.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 114 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_high_budget_distillation_114",
        "status": "running",
        "active_stage": "parallel_high_budget_selfplay",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "selfplay": None,
        "partition": None,
        "training": None,
        "holdout_audit": None,
        "offline_qualified": False,
        "panels": {},
        "selfplay_generated": True,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    result["selfplay"] = _run_or_load_parallel_selfplay(config)
    result["active_stage"] = "partition_high_budget_replay"
    _write_progress(output, result, started)
    partition = _run_or_load_partition(config, result["selfplay"])
    result["partition"] = partition
    result["active_stage"] = "parent_safe_distillation"
    _write_progress(output, result, started)
    training = _run_or_load_training(config, Path(partition["train_path"]))
    result["training"] = training
    result["checkpoint_trained"] = True
    result["active_stage"] = "untouched_holdout_audit"
    _write_progress(output, result, started)
    audit = _run_or_load_holdout_audit(config, partition, training)
    result["holdout_audit"] = audit
    result["offline_qualified"] = bool(audit["qualification"]["qualified"])
    _write_progress(output, result, started)
    if result["offline_qualified"]:
        candidate = Path(training["checkpoint"])
        candidate_hash = str(training["checkpoint_sha256"])
        result["active_stage"] = "primary_256_simulation_gameplay"
        _write_progress(output, result, started)
        primary = _run_or_load_panel(
            config,
            panel="primary_256",
            candidate_checkpoint=candidate,
            candidate_hash=candidate_hash,
        )
        primary_bootstrap = paired_score_share_bootstrap(
            primary, samples=BOOTSTRAP_SAMPLES, seed=1_114_300
        )
        result["panels"]["primary_256"] = {
            "arena": primary,
            "paired_bootstrap": primary_bootstrap,
            "assessment": compression_assessment(primary, primary_bootstrap),
        }
        _write_progress(output, result, started)
        result["active_stage"] = "diagnostic_1024_simulation_gameplay"
        _write_progress(output, result, started)
        diagnostic = _run_or_load_panel(
            config,
            panel="diagnostic_1024",
            candidate_checkpoint=candidate,
            candidate_hash=candidate_hash,
        )
        result["panels"]["diagnostic_1024"] = {
            "arena": diagnostic,
            "paired_bootstrap": paired_score_share_bootstrap(
                diagnostic, samples=BOOTSTRAP_SAMPLES, seed=1_114_400
            ),
            "assessment": practical_assessment(diagnostic),
        }
    primary_assessment = result["panels"].get("primary_256", {}).get("assessment")
    diagnostic_assessment = result["panels"].get("diagnostic_1024", {}).get("assessment")
    result["decision"] = high_budget_distillation_decision(
        qualified=bool(result["offline_qualified"]),
        primary_passed=(bool(primary_assessment["passed"]) if primary_assessment else None),
        diagnostic_passed=(
            bool(diagnostic_assessment["passed"]) if diagnostic_assessment else None
        ),
    )
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed, result)
    return result


def _run_or_load_parallel_selfplay(config: HighBudgetDistillationConfig) -> dict[str, Any]:
    manifest = config.output_dir / "selfplay" / "parallel_selfplay.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_parallel_selfplay_manifest(config, payload)
        return payload
    tasks = [
        {
            "output_dir": str(config.output_dir),
            "checkpoint": str(config.generation_11_checkpoint),
            "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "device": config.device,
            "max_turns": config.max_turns,
            "shard_index": index,
            "seed_start": seed_start,
        }
        for index, seed_start in enumerate(SHARD_SEED_STARTS)
    ]
    records: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=config.shard_workers) as executor:
        futures = {executor.submit(_generate_clean_shard_worker, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records[int(record["shard_index"])] = record
    shards = [records[index] for index in range(4)]
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "shard_workers": config.shard_workers,
        "shards": shards,
        "games": sum(int(shard["summary"]["games"]) for shard in shards),
        "positions": sum(int(shard["summary"]["positions"]) for shard in shards),
        "termination_reasons": {"completed": sum(int(shard["summary"]["games"]) for shard in shards)},
    }
    if payload["games"] != 256:
        raise RuntimeError("114 self-play did not produce exactly 256 clean games.")
    _write_json(manifest, payload)
    return payload


def _generate_clean_shard_worker(task: dict[str, Any]) -> dict[str, Any]:
    _limit_worker_threads()
    index = int(task["shard_index"])
    seed_start = int(task["seed_start"])
    shard_dir = Path(task["output_dir"]) / "selfplay" / f"shard_{index}"
    manifest = shard_dir / "clean_shard.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_clean_shard_manifest(task, payload)
        return payload
    records: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + GAMES_PER_SHARD):
        record = _run_or_load_single_selfplay_game(task, shard_dir, seed, replacement=False)
        records.append(record)
        if record["termination_reason"] == "completed":
            clean.append(record)
    for attempt in range(MAX_REPLACEMENTS_PER_SHARD):
        if len(clean) >= GAMES_PER_SHARD:
            break
        record = _run_or_load_single_selfplay_game(
            task,
            shard_dir,
            seed_start + GAMES_PER_SHARD + attempt,
            replacement=True,
        )
        records.append(record)
        if record["termination_reason"] == "completed":
            clean.append(record)
    if len(clean) < GAMES_PER_SHARD:
        raise RuntimeError(f"114 shard {index} could not obtain 64 clean games.")
    selected = clean[:GAMES_PER_SHARD]
    sanitized = shard_dir / "sanitized.jsonl"
    temporary = sanitized.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as destination:
        for record in selected:
            with Path(record["path"]).open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(sanitized)
    summary = _stream_clean_summary(
        sanitized,
        expected_seeds=[int(record["seed"]) for record in selected],
        expected_checkpoint_path=Path(task["checkpoint"]),
    )
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": task["checkpoint_sha256"],
        "shard_index": index,
        "seed_start": seed_start,
        "original_games": GAMES_PER_SHARD,
        "attempted_games": len(records),
        "abnormal_original_games": [
            record for record in records[:GAMES_PER_SHARD] if record["termination_reason"] != "completed"
        ],
        "replacement_games": records[GAMES_PER_SHARD:],
        "selected_clean_seeds": [int(record["seed"]) for record in selected],
        "sanitized_path": str(sanitized),
        "sanitized_sha256": _sha256(sanitized),
        "summary": summary,
    }
    _write_json(manifest, payload)
    return payload


def _run_or_load_single_selfplay_game(
    task: dict[str, Any], shard_dir: Path, seed: int, *, replacement: bool
) -> dict[str, Any]:
    label = "replacement" if replacement else "original"
    path = shard_dir / "games" / f"{label}_{seed}.jsonl"
    game = generate_self_play(
        checkpoint_path=task["checkpoint"],
        device=task["device"],
        config=SelfPlayConfig(
            generation=11,
            game_count=1,
            seed_start=seed,
            search=SearchConfig(
                simulations=SELFPLAY_SIMULATIONS,
                c_puct=2.25,
                mode="real",
                root_noise_fraction=0.25,
                seed=SELFPLAY_SEARCH_SEED,
            ),
            max_turns=int(task["max_turns"]),
        ),
        replay_path=path,
        progress_path=shard_dir / "games" / f"{label}_{seed}.progress.json",
        resume=True,
    )
    summary = game.games[0]
    record = {
        "seed": seed,
        "path": str(path),
        "sha256": _sha256(path),
        "termination_reason": summary.termination_reason,
        "winner": summary.winner,
        "final_scores": list(summary.final_scores),
        "turns": summary.turns,
        "positions": summary.positions,
    }
    del game
    gc.collect()
    return record


def _stream_clean_summary(
    path: Path, *, expected_seeds: list[int], expected_checkpoint_path: Path | None = None
) -> dict[str, Any]:
    expected = set(expected_seeds)
    order: list[int] = []
    next_turn: dict[int, int] = {}
    positions = 0
    selected: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = _row_from_payload(json.loads(line))
            if row.game_seed not in expected:
                raise ValueError("114 clean replay contains an unexpected seed.")
            if row.game_seed not in next_turn:
                next_turn[row.game_seed] = 0
                order.append(row.game_seed)
            if row.turn_index != next_turn[row.game_seed]:
                raise ValueError("114 clean replay turns are not contiguous.")
            next_turn[row.game_seed] += 1
            if (
                row.generation != 11
                or row.checkpoint_generation != 11
                or (expected_checkpoint_path is not None and Path(row.checkpoint_path) != expected_checkpoint_path)
                or row.value_target is None
                or row.termination_reason != "completed"
                or int(row.search_telemetry.get("simulations", -1)) != SELFPLAY_SIMULATIONS
                or int(row.search_config.get("simulations", -1)) != SELFPLAY_SIMULATIONS
                or float(row.search_config.get("c_puct", -1.0)) != 2.25
                or row.search_config.get("mode") != "real"
                or float(row.search_config.get("root_noise_fraction", -1.0)) != 0.25
                or int(row.search_config.get("seed", -1)) != SELFPLAY_SEARCH_SEED
            ):
                raise ValueError("114 clean replay row violates its frozen contract.")
            selected[str(row.action_payloads[row.selected_action_row]["action_type"])] += 1
            positions += 1
    if order != expected_seeds or set(next_turn) != expected or any(value <= 0 for value in next_turn.values()):
        raise ValueError("114 clean replay seed ordering or coverage mismatch.")
    return {
        "valid": True,
        "games": len(expected_seeds),
        "positions": positions,
        "termination_reasons": {"completed": len(expected_seeds)},
        "mean_turns": positions / len(expected_seeds),
        "selected_action_family_counts": dict(sorted(selected.items())),
    }


def _validate_clean_shard_manifest(task: dict[str, Any], payload: dict[str, Any]) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("checkpoint_sha256") != task["checkpoint_sha256"]
        or payload.get("shard_index") != task["shard_index"]
        or payload.get("seed_start") != task["seed_start"]
        or payload.get("sanitized_sha256") != _sha256(payload["sanitized_path"])
    ):
        raise ValueError("114 clean shard manifest identity mismatch.")
    summary = _stream_clean_summary(
        Path(payload["sanitized_path"]),
        expected_seeds=[int(seed) for seed in payload["selected_clean_seeds"]],
        expected_checkpoint_path=Path(task["checkpoint"]),
    )
    if summary != payload["summary"]:
        raise ValueError("114 clean shard streaming summary mismatch.")


def _validate_parallel_selfplay_manifest(
    config: HighBudgetDistillationConfig, payload: dict[str, Any]
) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("games") != 256
        or payload.get("shard_workers") != 4
        or len(payload.get("shards", [])) != 4
    ):
        raise ValueError("114 parallel self-play manifest mismatch.")
    for shard in payload["shards"]:
        index = int(shard["shard_index"])
        _validate_clean_shard_manifest(
            {
                "checkpoint": str(config.generation_11_checkpoint),
                "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
                "shard_index": index,
                "seed_start": SHARD_SEED_STARTS[index],
            },
            shard,
        )


def _run_or_load_partition(
    config: HighBudgetDistillationConfig, selfplay: dict[str, Any]
) -> dict[str, Any]:
    manifest = config.output_dir / "partition" / "train_holdout.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_partition_manifest(payload)
        return payload
    schedule = partition_seed_schedule(
        [[int(seed) for seed in shard["selected_clean_seeds"]] for shard in selfplay["shards"]]
    )
    seed_sets = {name: set(seeds) for name, seeds in schedule.items()}
    paths = {
        "train": config.output_dir / "partition" / "train_224.jsonl",
        "holdout": config.output_dir / "partition" / "holdout_32.jsonl",
    }
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    for path in temporary.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = {name: path.open("w", encoding="utf-8", newline="") for name, path in temporary.items()}
    try:
        for shard in selfplay["shards"]:
            with Path(shard["sanitized_path"]).open("r", encoding="utf-8") as source:
                for line in source:
                    match = _GAME_SEED_PATTERN.search(line)
                    if match is None:
                        raise ValueError("114 partition source row lacks game seed.")
                    seed = int(match.group(1))
                    matched = [name for name, seeds in seed_sets.items() if seed in seeds]
                    if len(matched) != 1:
                        raise ValueError("114 source row does not map uniquely to a partition.")
                    handles[matched[0]].write(line)
    finally:
        for handle in handles.values():
            handle.close()
    for name, path in paths.items():
        temporary[name].replace(path)
    summaries = {
        name: _stream_clean_summary(path, expected_seeds=schedule[name])
        for name, path in paths.items()
    }
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "train_path": str(paths["train"]),
        "train_sha256": _sha256(paths["train"]),
        "train_seeds": schedule["train"],
        "holdout_path": str(paths["holdout"]),
        "holdout_sha256": _sha256(paths["holdout"]),
        "holdout_seeds": schedule["holdout"],
        "summaries": summaries,
    }
    _write_json(manifest, payload)
    return payload


def _validate_partition_manifest(payload: dict[str, Any]) -> None:
    if payload.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("114 partition protocol mismatch.")
    for name, count in (("train", 224), ("holdout", 32)):
        if (
            len(payload.get(f"{name}_seeds", [])) != count
            or payload.get(f"{name}_sha256") != _sha256(payload[f"{name}_path"])
        ):
            raise ValueError(f"114 {name} partition identity mismatch.")
        summary = _stream_clean_summary(
            Path(payload[f"{name}_path"]),
            expected_seeds=[int(seed) for seed in payload[f"{name}_seeds"]],
        )
        if summary != payload["summaries"][name]:
            raise ValueError(f"114 {name} partition summary mismatch.")
    if set(payload["train_seeds"]) & set(payload["holdout_seeds"]):
        raise ValueError("114 training and holdout games overlap.")


def _run_or_load_training(
    config: HighBudgetDistillationConfig, replay_path: Path
) -> dict[str, Any]:
    manifest = config.output_dir / "training" / "distillation.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_training_manifest(config, replay_path, payload)
        training = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
        return {**payload, "training": training, "training_validity": training_validity(training, expected_parent=11)}
    checkpoint = config.output_dir / "checkpoints" / "high_budget_generation_012.pt"
    metrics = config.output_dir / "training" / "distillation.json"
    training = train_parent_safe_indexed_successor(
        parent_checkpoint=config.generation_11_checkpoint,
        replay_paths=[replay_path],
        output_checkpoint=checkpoint,
        metrics_path=metrics,
        device=config.device,
        config=distillation_training_config(),
    )
    validity = training_validity(training, expected_parent=11)
    _, identity = load_checkpoint(checkpoint, device=config.device)
    validity["checks"]["checkpoint_roundtrip_generation"] = identity.generation == 12
    validity["checks"]["parent_safe_metadata"] = (
        identity.metadata.get("parent_safe_selection") is True
        and identity.metadata.get("selected_epoch") == training["selected_epoch"]
    )
    validity["valid"] = all(validity["checks"].values())
    if not validity["valid"]:
        raise RuntimeError("114 indexed training failed validity checks.")
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "replay_path": str(replay_path),
        "replay_sha256": _sha256(replay_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
    }
    _write_json(manifest, payload)
    return {**payload, "training": training, "training_validity": validity}


def _validate_training_manifest(
    config: HighBudgetDistillationConfig, replay_path: Path, payload: dict[str, Any]
) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("parent_checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("replay_sha256") != _sha256(replay_path)
        or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        or payload.get("metrics_sha256") != _sha256(payload["metrics_path"])
    ):
        raise ValueError("114 training manifest mismatch.")
    training = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
    validity = training_validity(training, expected_parent=11)
    _, identity = load_checkpoint(payload["checkpoint"], device=config.device)
    if (
        not validity["valid"]
        or identity.generation != 12
        or identity.metadata.get("parent_safe_selection") is not True
        or identity.metadata.get("selected_epoch") != training.get("selected_epoch")
    ):
        raise ValueError("114 training lineage or parent-safe selection mismatch.")


def _run_or_load_holdout_audit(
    config: HighBudgetDistillationConfig,
    partition: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    path = config.output_dir / "audit" / "untouched_holdout.stage.json"
    checkpoint_hashes = {
        "parent": EXPECTED_HASHES["generation_11_checkpoint"],
        "candidate": training["checkpoint_sha256"],
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("holdout_sha256") != partition["holdout_sha256"]
            or payload.get("checkpoint_hashes") != checkpoint_hashes
        ):
            raise ValueError("114 holdout audit identity mismatch.")
        return payload
    replay_paths, entries = index_replay_paths([partition["holdout_path"]])
    loader = make_indexed_loader(
        replay_paths, entries, config=distillation_training_config(), shuffle=False
    )
    models = {
        "parent": load_checkpoint(config.generation_11_checkpoint, device=config.device)[0],
        "candidate": load_checkpoint(training["checkpoint"], device=config.device)[0],
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.eval()
        metrics[name] = _run_epoch(model, loader, config.device, None, distillation_training_config())
    qualification = distillation_qualification(
        training=training["training"], parent=metrics["parent"], candidate=metrics["candidate"]
    )
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "holdout_sha256": partition["holdout_sha256"],
        "checkpoint_hashes": checkpoint_hashes,
        "rows": len(entries),
        "games": 32,
        "metrics": metrics,
        "policy_drift_vs_parent": _compare_policies(
            models["parent"], models["candidate"], loader, config.device
        ),
        "qualification": qualification,
    }
    _write_json(path, payload)
    if isinstance(loader.dataset, IndexedReplayDataset):
        loader.dataset.close()
    return payload


def _panel_spec(panel: str) -> dict[str, int]:
    if panel == "primary_256":
        return {"seed_start": PRIMARY_SEED_START, "paired_seeds": PRIMARY_PAIRED_SEEDS, "simulations": 256, "paired_per_shard": 50}
    if panel == "diagnostic_1024":
        return {"seed_start": DIAGNOSTIC_SEED_START, "paired_seeds": DIAGNOSTIC_PAIRED_SEEDS, "simulations": 1_024, "paired_per_shard": 25}
    raise ValueError(f"Unknown 114 panel: {panel}")


def _run_or_load_panel(
    config: HighBudgetDistillationConfig,
    *,
    panel: str,
    candidate_checkpoint: Path,
    candidate_hash: str,
) -> dict[str, Any]:
    manifest = config.output_dir / "arenas" / f"{panel}.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_panel_manifest(payload, panel, candidate_hash)
        shards = {int(index): arena for index, arena in payload["shards"].items()}
        return combine_arena_shards(panel, shards)
    tasks = [
        {
            "panel": panel,
            "shard_index": index,
            "output_dir": str(config.output_dir / "arenas"),
            "candidate_checkpoint": str(candidate_checkpoint),
            "candidate_hash": candidate_hash,
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
        }
        for index in range(ARENA_SHARDS)
    ]
    shards: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
        futures = {executor.submit(_arena_shard_worker, task): task for task in tasks}
        for future in as_completed(futures):
            index, arena = future.result()
            shards[index] = arena
    combined = combine_arena_shards(panel, shards)
    _write_json(
        manifest,
        {
            "protocol_sha256": PROTOCOL_SHA256,
            "panel": panel,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "arena_workers": config.arena_workers,
            "shards": {str(index): shards[index] for index in sorted(shards)},
        },
    )
    return combined


def combine_arena_shards(panel: str, shards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    spec = _panel_spec(panel)
    if set(shards) != set(range(ARENA_SHARDS)):
        raise ValueError(f"114 {panel} requires all four exact shards.")
    rows: list[dict[str, object]] = []
    per_shard = spec["paired_per_shard"]
    for index in range(ARENA_SHARDS):
        arena = shards[index]
        start = spec["seed_start"] + index * per_shard
        expected = [(start + offset, seat) for offset in range(per_shard) for seat in (0, 1)]
        shard_rows = list(arena.get("rows", []))
        observed = [(int(row["seed"]), int(row["candidate_seat"])) for row in shard_rows]
        if arena.get("paired_seeds") != per_shard or arena.get("games") != 2 * per_shard or observed != expected:
            raise ValueError(f"114 {panel} shard {index} is not its exact schedule.")
        rows.extend(shard_rows)
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    terminations: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        terminations[reason] = terminations.get(reason, 0) + 1
    return _arena_dict(
        ArenaResult(
            name=f"high_budget_distilled_candidate_vs_generation_11_{panel}",
            paired_seeds=spec["paired_seeds"],
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


def _arena_shard_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    _limit_worker_threads()
    panel = str(task["panel"])
    index = int(task["shard_index"])
    spec = _panel_spec(panel)
    search = _arena_search(spec["simulations"])
    candidate = alpha_zero_search_factory(task["candidate_checkpoint"], task["device"], search)
    baseline = alpha_zero_search_factory(task["parent_checkpoint"], task["device"], search)
    seed_start = spec["seed_start"] + index * spec["paired_per_shard"]
    arena_dir = Path(task["output_dir"])
    progress = arena_dir / f"{panel}_shard_{index}.json"
    identity_path = arena_dir / f"{panel}_shard_{index}.identity.json"
    identity = _arena_shard_identity(panel, index, str(task["candidate_hash"]), int(task["max_turns"]))
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"114 {panel} shard {index} identity mismatch.")
    else:
        if progress.exists():
            raise ValueError(f"114 {panel} shard {index} progress lacks an identity.")
        _write_json(identity_path, identity)
    arena = _arena_dict(
        run_paired_arena(
            name=f"{panel}_shard_{index}",
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=seed_start,
            paired_seeds=spec["paired_per_shard"],
            max_turns=int(task["max_turns"]),
            progress_path=progress,
            resume=True,
        )
    )
    return index, arena


def _arena_search(simulations: int) -> SearchConfig:
    return SearchConfig(
        simulations=simulations,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_114_200,
    )


def _arena_shard_identity(panel: str, index: int, candidate_hash: str, max_turns: int) -> dict[str, Any]:
    spec = _panel_spec(panel)
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "panel": panel,
        "shard_index": index,
        "candidate_checkpoint_sha256": candidate_hash,
        "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "candidate_search_config": asdict(_arena_search(spec["simulations"])),
        "baseline_search_config": asdict(_arena_search(spec["simulations"])),
        "operative_search_seed": "2 * game_seed + seat",
        "seed_start": spec["seed_start"] + index * spec["paired_per_shard"],
        "paired_seeds": spec["paired_per_shard"],
        "max_turns": max_turns,
    }


def _validate_panel_manifest(payload: dict[str, Any], panel: str, candidate_hash: str) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("panel") != panel
        or payload.get("candidate_checkpoint_sha256") != candidate_hash
        or payload.get("baseline_checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("arena_workers") != 4
        or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
    ):
        raise ValueError(f"114 {panel} manifest identity mismatch.")


def _verify_inputs(config: HighBudgetDistillationConfig) -> None:
    paths = {
        "result_113": config.result_113,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 114 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_113.read_text(encoding="utf-8"))
    arena = prior.get("arena", {})
    if (
        prior.get("status") != "complete"
        or prior.get("decision")
        != "deeper_search_teacher_headroom_confirmed_review_high_budget_distillation_no_install"
        or prior.get("assessment", {}).get("passed") is not True
        or (arena.get("candidate_wins"), arena.get("draws"), arena.get("candidate_losses"))
        != (229, 1, 170)
        or prior.get("paired_bootstrap", {}).get("ci95_low") != 0.525
        or arena.get("termination_reasons") != {"completed": 400}
        or prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]
    ):
        raise ValueError("Frozen 113 result lacks its exact completed pass contract.")
    _, identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    if identity.generation != 11 or identity.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen 114 parent is not exact generation 11.")
    if (
        config.max_turns != 200
        or config.torch_threads != 1
        or config.shard_workers != 4
        or config.arena_workers != 4
    ):
        raise ValueError("114 requires the frozen 200-turn, one-thread, four-worker execution.")


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config: HighBudgetDistillationConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})
