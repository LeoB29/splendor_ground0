"""Parallel fresh-self-play dose experiment 110."""

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

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .head_attribution import paired_arena_difference_bootstrap
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
class FreshDoseConfig:
    output_dir: Path
    result_109: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None
    shard_workers: int = 4
    arena_workers: int = 2


PROTOCOL_SHA256 = "b1284d08e0f414180e0d44bcff0d6f9bb43d71552742911aad2040432e750f49"
EXPECTED_HASHES = {
    "result_109": "ba4318190b84775a097fbc3a656b823447f05edd45618fde5be6800e3d0e81a1",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
}
SHARD_SEED_STARTS = (2_010_000, 2_020_000, 2_030_000, 2_040_000)
GAMES_PER_SHARD = 256
MAX_REPLACEMENTS_PER_SHARD = 32
LOW_GAMES_PER_SHARD = 64
HOLDOUT_GAMES_PER_SHARD = 32
SELFPLAY_SEARCH_SEED = 1_100_110
ARENA_SEED_START = 2_050_000
PAIRED_SEEDS = 100
BOOTSTRAP_SAMPLES = 50_000
_GAME_SEED_PATTERN = re.compile(r'"game_seed"\s*:\s*(\d+)')


def dose_training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=12,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=1_100_110,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        bucket_batches=True,
        policy_divergence_weight=0.0,
        visit_argmax_aux_weight=0.0,
    )


def qualification_assessment(
    *, training: dict[str, Any], parent_metrics: dict[str, float], candidate_metrics: dict[str, float]
) -> dict[str, Any]:
    objective_delta = float(candidate_metrics["objective_loss"]) - float(
        parent_metrics["objective_loss"]
    )
    checks = {
        "nonzero_epoch_selected": int(training.get("selected_epoch", 0)) > 0,
        "common_holdout_objective_improved": objective_delta < 0.0,
        "all_common_metrics_finite": all(
            math.isfinite(float(value))
            for metrics in (parent_metrics, candidate_metrics)
            for value in metrics.values()
            if isinstance(value, (int, float))
        ),
    }
    return {
        "qualified": all(checks.values()),
        "checks": checks,
        "common_holdout_delta_candidate_minus_parent": {
            key: float(candidate_metrics[key]) - float(parent_metrics[key])
            for key in ("policy_loss", "value_loss", "objective_loss")
        },
    }


def fresh_dose_decision(
    *,
    high_qualified: bool,
    high_assessment: dict[str, Any] | None,
    low_assessment: dict[str, Any] | None,
) -> str:
    if not high_qualified:
        return "high_dose_failed_offline_gate_review_fresh_data_and_capacity_no_install"
    if high_assessment is None:
        raise ValueError("A qualified high-dose candidate requires gameplay assessment.")
    if not high_assessment["passed"]:
        return "high_dose_fresh_selfplay_failed_internal_gate_review_capacity_no_install"
    if low_assessment is not None and float(high_assessment["score_share"]) <= float(
        low_assessment["score_share"]
    ):
        return "fresh_selfplay_progress_not_dose_specific_no_install"
    return "high_dose_fresh_selfplay_internal_pass_review_next_generation_no_install"


def partition_seed_schedule(
    shard_seed_lists: list[list[int]],
) -> dict[str, list[int]]:
    if len(shard_seed_lists) != len(SHARD_SEED_STARTS):
        raise ValueError("The frozen partition requires exactly four shards.")
    low: set[int] = set()
    high: set[int] = set()
    holdout: set[int] = set()
    for seeds in shard_seed_lists:
        if len(seeds) != GAMES_PER_SHARD or len(set(seeds)) != GAMES_PER_SHARD:
            raise ValueError("Every clean shard must contain exactly 256 unique games.")
        low.update(seeds[:LOW_GAMES_PER_SHARD])
        holdout.update(seeds[-HOLDOUT_GAMES_PER_SHARD:])
        high.update(seeds[:-HOLDOUT_GAMES_PER_SHARD])
    if not low < high or high & holdout:
        raise RuntimeError("Fresh replay partition does not satisfy nesting/isolation.")
    if (len(low), len(high), len(holdout)) != (256, 896, 128):
        raise RuntimeError("Fresh replay partition has an invalid game count.")
    return {
        "low": sorted(low),
        "high": sorted(high),
        "holdout": sorted(holdout),
    }


def run_fresh_dose_110(config: FreshDoseConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 110 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_fresh_gen11_dose_110",
        "status": "running",
        "active_stage": "parallel_fresh_selfplay",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "selfplay": None,
        "partition": None,
        "training": {},
        "common_holdout_audit": None,
        "qualified_arms": [],
        "panels": {},
        "dose_comparisons": {},
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
    result["active_stage"] = "partition_fresh_replay"
    _write_progress(output, result, started)
    partition = _run_or_load_partition(config, result["selfplay"])
    result["partition"] = partition
    _write_progress(output, result, started)

    for arm in ("low", "high"):
        result["active_stage"] = f"train_{arm}_dose"
        _write_progress(output, result, started)
        result["training"][arm] = _run_or_load_training_arm(
            config,
            arm=arm,
            replay_path=Path(partition[f"{arm}_path"]),
        )
        result["checkpoint_trained"] = True
        _write_progress(output, result, started)

    result["active_stage"] = "common_holdout_audit"
    _write_progress(output, result, started)
    audit = _run_or_load_holdout_audit(config, partition, result["training"])
    result["common_holdout_audit"] = audit
    result["qualified_arms"] = [
        arm for arm in ("low", "high") if audit["qualification"][arm]["qualified"]
    ]
    _write_progress(output, result, started)

    if result["qualified_arms"]:
        result["active_stage"] = "raw_gameplay"
        _write_progress(output, result, started)
        for index, arm in enumerate(result["qualified_arms"]):
            checkpoint = Path(result["training"][arm]["checkpoint"])
            name = f"raw_{arm}_dose_vs_raw_generation_11"
            result["panels"][name] = _run_panel(
                output=output / "arenas",
                name=name,
                candidate=alpha_zero_raw_factory(checkpoint, config.device),
                baseline=alpha_zero_raw_factory(config.generation_11_checkpoint, config.device),
                bootstrap_seed=1_100_300 + index,
                max_turns=config.max_turns,
            )
            _write_progress(output, result, started)

        result["active_stage"] = "parallel_searched_gameplay"
        _write_progress(output, result, started)
        searched = _run_parallel_searched_panels(config, result["qualified_arms"], result["training"])
        for index, arm in enumerate(result["qualified_arms"]):
            name = f"searched_{arm}_dose_vs_searched_generation_11"
            arena = searched[arm]
            result["panels"][name] = {
                "arena": arena,
                "paired_bootstrap": paired_score_share_bootstrap(
                    arena, samples=BOOTSTRAP_SAMPLES, seed=1_100_400 + index
                ),
                "assessment": practical_assessment(arena),
            }
            _write_progress(output, result, started)

    if set(result["qualified_arms"]) == {"low", "high"}:
        result["dose_comparisons"] = {
            "raw_high_minus_low": paired_arena_difference_bootstrap(
                result["panels"]["raw_high_dose_vs_raw_generation_11"]["arena"],
                result["panels"]["raw_low_dose_vs_raw_generation_11"]["arena"],
                samples=BOOTSTRAP_SAMPLES,
                seed=1_100_500,
            ),
            "searched_high_minus_low": paired_arena_difference_bootstrap(
                result["panels"]["searched_high_dose_vs_searched_generation_11"]["arena"],
                result["panels"]["searched_low_dose_vs_searched_generation_11"]["arena"],
                samples=BOOTSTRAP_SAMPLES,
                seed=1_100_501,
            ),
        }
    high_panel = result["panels"].get("searched_high_dose_vs_searched_generation_11")
    low_panel = result["panels"].get("searched_low_dose_vs_searched_generation_11")
    high_assessment = None
    if high_panel is not None:
        high_assessment = {
            **high_panel["assessment"],
            "score_share": high_panel["arena"]["score_share"],
        }
    low_assessment = None
    if low_panel is not None:
        low_assessment = {
            **low_panel["assessment"],
            "score_share": low_panel["arena"]["score_share"],
        }
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = fresh_dose_decision(
        high_qualified="high" in result["qualified_arms"],
        high_assessment=high_assessment,
        low_assessment=low_assessment,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def _run_or_load_parallel_selfplay(config: FreshDoseConfig) -> dict[str, Any]:
    output = config.output_dir
    manifest = output / "selfplay" / "parallel_selfplay.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_parallel_selfplay_manifest(config, payload)
        return payload
    tasks = [
        {
            "output_dir": str(output),
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
    with ProcessPoolExecutor(max_workers=min(config.shard_workers, len(tasks))) as executor:
        futures = {executor.submit(_generate_clean_shard_worker, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records[int(record["shard_index"])] = record
    shards = [records[index] for index in range(len(tasks))]
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "shard_workers": min(config.shard_workers, len(tasks)),
        "shards": shards,
        "games": sum(int(shard["summary"]["games"]) for shard in shards),
        "positions": sum(int(shard["summary"]["positions"]) for shard in shards),
        "termination_reasons": {"completed": sum(int(shard["summary"]["games"]) for shard in shards)},
    }
    if payload["games"] != 1024:
        raise RuntimeError("Parallel self-play did not produce exactly 1,024 clean games.")
    _write_json(manifest, payload)
    return payload


def _generate_clean_shard_worker(task: dict[str, Any]) -> dict[str, Any]:
    _limit_worker_threads()
    output = Path(task["output_dir"])
    shard_index = int(task["shard_index"])
    seed_start = int(task["seed_start"])
    shard_dir = output / "selfplay" / f"shard_{shard_index}"
    manifest = shard_dir / "clean_shard.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_clean_shard_manifest(task, payload)
        return payload
    schedule = list(range(seed_start, seed_start + GAMES_PER_SHARD))
    records: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    for seed in schedule:
        record = _run_or_load_single_selfplay_game(task, shard_dir, seed, replacement=False)
        records.append(record)
        if record["termination_reason"] == "completed":
            clean.append(record)
    for attempt in range(MAX_REPLACEMENTS_PER_SHARD):
        if len(clean) >= GAMES_PER_SHARD:
            break
        seed = seed_start + GAMES_PER_SHARD + attempt
        record = _run_or_load_single_selfplay_game(task, shard_dir, seed, replacement=True)
        records.append(record)
        if record["termination_reason"] == "completed":
            clean.append(record)
    if len(clean) < GAMES_PER_SHARD:
        raise RuntimeError(f"Shard {shard_index} could not obtain 256 clean games.")
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
        expected_generation=11,
        expected_checkpoint_path=Path(task["checkpoint"]),
    )
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": task["checkpoint_sha256"],
        "shard_index": shard_index,
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
    progress = shard_dir / "games" / f"{label}_{seed}.progress.json"
    game = generate_self_play(
        checkpoint_path=task["checkpoint"],
        device=task["device"],
        config=SelfPlayConfig(
            generation=11,
            game_count=1,
            seed_start=seed,
            search=SearchConfig(
                simulations=256,
                c_puct=2.25,
                mode="real",
                root_noise_fraction=0.25,
                seed=SELFPLAY_SEARCH_SEED,
            ),
            max_turns=int(task["max_turns"]),
        ),
        replay_path=path,
        progress_path=progress,
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


def _validate_clean_shard_manifest(task: dict[str, Any], payload: dict[str, Any]) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("checkpoint_sha256") != task["checkpoint_sha256"]
        or payload.get("shard_index") != task["shard_index"]
        or payload.get("seed_start") != task["seed_start"]
        or payload.get("sanitized_sha256") != _sha256(payload["sanitized_path"])
    ):
        raise ValueError("Clean shard manifest identity mismatch.")
    summary = _stream_clean_summary(
        Path(payload["sanitized_path"]),
        expected_seeds=[int(seed) for seed in payload["selected_clean_seeds"]],
        expected_generation=11,
        expected_checkpoint_path=Path(task["checkpoint"]),
    )
    if summary != payload["summary"]:
        raise ValueError("Clean shard streaming summary mismatch.")


def _validate_parallel_selfplay_manifest(config: FreshDoseConfig, payload: dict[str, Any]) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("games") != 1024
        or len(payload.get("shards", [])) != 4
    ):
        raise ValueError("Parallel self-play manifest mismatch.")
    for shard in payload["shards"]:
        task = {
            "checkpoint": str(config.generation_11_checkpoint),
            "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "shard_index": shard["shard_index"],
            "seed_start": SHARD_SEED_STARTS[int(shard["shard_index"])],
        }
        _validate_clean_shard_manifest(task, shard)


def _stream_clean_summary(
    path: Path,
    *,
    expected_seeds: list[int],
    expected_generation: int,
    expected_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    expected = set(expected_seeds)
    seen_order: list[int] = []
    next_turn: dict[int, int] = {}
    positions = 0
    selected: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = _row_from_payload(json.loads(line))
            if row.game_seed not in expected:
                raise ValueError("Clean shard contains an unexpected seed.")
            if row.game_seed not in next_turn:
                next_turn[row.game_seed] = 0
                seen_order.append(row.game_seed)
            if row.turn_index != next_turn[row.game_seed]:
                raise ValueError("Clean shard game turns are not contiguous.")
            next_turn[row.game_seed] += 1
            if (
                row.generation != expected_generation
                or row.checkpoint_generation != expected_generation
                or (
                    expected_checkpoint_path is not None
                    and Path(row.checkpoint_path) != expected_checkpoint_path
                )
                or row.value_target is None
                or row.termination_reason != "completed"
                or int(row.search_telemetry.get("simulations", -1)) != 256
                or int(row.search_config.get("simulations", -1)) != 256
                or float(row.search_config.get("c_puct", -1.0)) != 2.25
                or row.search_config.get("mode") != "real"
                or float(row.search_config.get("root_noise_fraction", -1.0)) != 0.25
                or int(row.search_config.get("seed", -1)) != SELFPLAY_SEARCH_SEED
            ):
                raise ValueError("Clean shard row violates the frozen replay contract.")
            selected[str(row.action_payloads[row.selected_action_row]["action_type"])] += 1
            positions += 1
    if seen_order != expected_seeds or set(next_turn) != expected or any(turns <= 0 for turns in next_turn.values()):
        raise ValueError("Clean shard seed ordering or coverage mismatch.")
    return {
        "valid": True,
        "games": len(expected_seeds),
        "positions": positions,
        "termination_reasons": {"completed": len(expected_seeds)},
        "mean_turns": positions / len(expected_seeds),
        "selected_action_family_counts": dict(sorted(selected.items())),
    }


def _run_or_load_partition(config: FreshDoseConfig, selfplay: dict[str, Any]) -> dict[str, Any]:
    output = config.output_dir
    manifest = output / "partition" / "fresh_pools.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_partition_manifest(payload)
        return payload
    seed_partition = partition_seed_schedule(
        [
            [int(seed) for seed in shard["selected_clean_seeds"]]
            for shard in selfplay["shards"]
        ]
    )
    low_seeds = set(seed_partition["low"])
    high_seeds = set(seed_partition["high"])
    holdout_seeds = set(seed_partition["holdout"])
    paths = {
        "low": output / "partition" / "low_256.jsonl",
        "high": output / "partition" / "high_896.jsonl",
        "holdout": output / "partition" / "holdout_128.jsonl",
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
                        raise ValueError("Partition source row lacks game seed.")
                    seed = int(match.group(1))
                    if seed in high_seeds:
                        handles["high"].write(line)
                    if seed in low_seeds:
                        handles["low"].write(line)
                    if seed in holdout_seeds:
                        handles["holdout"].write(line)
    finally:
        for handle in handles.values():
            handle.close()
    for name, path in paths.items():
        temporary[name].replace(path)
    summaries = {
        "low": _stream_clean_summary(paths["low"], expected_seeds=sorted(low_seeds), expected_generation=11),
        "high": _stream_clean_summary(paths["high"], expected_seeds=sorted(high_seeds), expected_generation=11),
        "holdout": _stream_clean_summary(paths["holdout"], expected_seeds=sorted(holdout_seeds), expected_generation=11),
    }
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "low_path": str(paths["low"]),
        "low_sha256": _sha256(paths["low"]),
        "low_seeds": sorted(low_seeds),
        "high_path": str(paths["high"]),
        "high_sha256": _sha256(paths["high"]),
        "high_seeds": sorted(high_seeds),
        "holdout_path": str(paths["holdout"]),
        "holdout_sha256": _sha256(paths["holdout"]),
        "holdout_seeds": sorted(holdout_seeds),
        "summaries": summaries,
    }
    _write_json(manifest, payload)
    return payload


def _validate_partition_manifest(payload: dict[str, Any]) -> None:
    if payload.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Fresh partition protocol mismatch.")
    expected_counts = {"low": 256, "high": 896, "holdout": 128}
    for name, count in expected_counts.items():
        if (
            len(payload.get(f"{name}_seeds", [])) != count
            or payload.get(f"{name}_sha256") != _sha256(payload[f"{name}_path"])
        ):
            raise ValueError(f"Fresh {name} partition identity mismatch.")
        summary = _stream_clean_summary(
            Path(payload[f"{name}_path"]),
            expected_seeds=[int(seed) for seed in payload[f"{name}_seeds"]],
            expected_generation=11,
        )
        if summary != payload["summaries"][name]:
            raise ValueError(f"Fresh {name} partition summary mismatch.")
    if not set(payload["low_seeds"]) < set(payload["high_seeds"]):
        raise ValueError("Low-dose pool is not a strict high-dose subset.")
    if set(payload["high_seeds"]) & set(payload["holdout_seeds"]):
        raise ValueError("Training and common holdout partitions overlap.")


def _run_or_load_training_arm(
    config: FreshDoseConfig, *, arm: str, replay_path: Path
) -> dict[str, Any]:
    output = config.output_dir
    manifest = output / "training" / f"{arm}_dose.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_training_manifest(config, arm, replay_path, payload)
        payload["training"] = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
        payload["training_validity"] = training_validity(payload["training"], expected_parent=11)
        return payload
    checkpoint = output / "checkpoints" / f"{arm}_dose_generation_012.pt"
    metrics = output / "training" / f"{arm}_dose.json"
    training = train_parent_safe_indexed_successor(
        parent_checkpoint=config.generation_11_checkpoint,
        replay_paths=[replay_path],
        output_checkpoint=checkpoint,
        metrics_path=metrics,
        device=config.device,
        config=dose_training_config(),
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
        raise RuntimeError(f"{arm} indexed training failed validity checks.")
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "arm": arm,
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
    config: FreshDoseConfig, arm: str, replay_path: Path, payload: dict[str, Any]
) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("arm") != arm
        or payload.get("parent_checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("replay_sha256") != _sha256(replay_path)
        or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        or payload.get("metrics_sha256") != _sha256(payload["metrics_path"])
    ):
        raise ValueError(f"{arm} training manifest mismatch.")
    training = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
    validity = training_validity(training, expected_parent=11)
    _, identity = load_checkpoint(payload["checkpoint"], device=config.device)
    if (
        not validity["valid"]
        or identity.generation != 12
        or identity.metadata.get("parent_safe_selection") is not True
        or identity.metadata.get("selected_epoch") != training.get("selected_epoch")
    ):
        raise ValueError(f"{arm} training lineage or parent-safe selection mismatch.")


def _run_or_load_holdout_audit(
    config: FreshDoseConfig,
    partition: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    path = config.output_dir / "audit" / "common_holdout.stage.json"
    checkpoint_hashes = {
        "parent": EXPECTED_HASHES["generation_11_checkpoint"],
        "low": training["low"]["checkpoint_sha256"],
        "high": training["high"]["checkpoint_sha256"],
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("holdout_sha256") != partition["holdout_sha256"]
            or payload.get("checkpoint_hashes") != checkpoint_hashes
        ):
            raise ValueError("Common holdout audit identity mismatch.")
        return payload
    replay_paths, entries = index_replay_paths([partition["holdout_path"]])
    loader = make_indexed_loader(
        replay_paths, entries, config=dose_training_config(), shuffle=False
    )
    checkpoint_paths = {
        "parent": config.generation_11_checkpoint,
        "low": Path(training["low"]["checkpoint"]),
        "high": Path(training["high"]["checkpoint"]),
    }
    models = {
        name: load_checkpoint(checkpoint, device=config.device)[0]
        for name, checkpoint in checkpoint_paths.items()
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.eval()
        metrics[name] = _run_epoch(model, loader, config.device, None, dose_training_config())
    drift = {
        arm: _compare_policies(models["parent"], models[arm], loader, config.device)
        for arm in ("low", "high")
    }
    qualification = {
        arm: qualification_assessment(
            training=training[arm]["training"],
            parent_metrics=metrics["parent"],
            candidate_metrics=metrics[arm],
        )
        for arm in ("low", "high")
    }
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "holdout_sha256": partition["holdout_sha256"],
        "checkpoint_hashes": checkpoint_hashes,
        "rows": len(entries),
        "games": 128,
        "metrics": metrics,
        "policy_drift_vs_parent": drift,
        "qualification": qualification,
    }
    _write_json(path, payload)
    if isinstance(loader.dataset, IndexedReplayDataset):
        loader.dataset.close()
    return payload


def _run_parallel_searched_panels(
    config: FreshDoseConfig,
    arms: list[str],
    training: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    tasks = [
        {
            "arm": arm,
            "candidate_checkpoint": training[arm]["checkpoint"],
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
            "output_dir": str(config.output_dir / "arenas"),
        }
        for arm in arms
    ]
    records: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(config.arena_workers, len(tasks))) as executor:
        futures = {executor.submit(_searched_panel_worker, task): task for task in tasks}
        for future in as_completed(futures):
            arm, arena = future.result()
            records[arm] = arena
    return records


def _searched_panel_worker(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _limit_worker_threads()
    arm = str(task["arm"])
    search = _evaluation_search()
    name = f"searched_{arm}_dose_vs_searched_generation_11"
    arena = _arena_dict(
        run_paired_arena(
            name=name,
            candidate_factory=alpha_zero_search_factory(
                task["candidate_checkpoint"], task["device"], search
            ),
            baseline_factory=alpha_zero_search_factory(
                task["parent_checkpoint"], task["device"], search
            ),
            seed_start=ARENA_SEED_START,
            paired_seeds=PAIRED_SEEDS,
            max_turns=int(task["max_turns"]),
            progress_path=Path(task["output_dir"]) / f"{name}.json",
            resume=True,
        )
    )
    return arm, arena


def _run_panel(
    *,
    output: Path,
    name: str,
    candidate: Any,
    baseline: Any,
    bootstrap_seed: int,
    max_turns: int,
) -> dict[str, Any]:
    arena = _arena_dict(
        run_paired_arena(
            name=name,
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=ARENA_SEED_START,
            paired_seeds=PAIRED_SEEDS,
            max_turns=max_turns,
            progress_path=output / f"{name}.json",
            resume=True,
        )
    )
    return {
        "arena": arena,
        "paired_bootstrap": paired_score_share_bootstrap(
            arena, samples=BOOTSTRAP_SAMPLES, seed=bootstrap_seed
        ),
        "assessment": practical_assessment(arena),
    }


def _evaluation_search() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_100_110,
    )


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _verify_inputs(config: FreshDoseConfig) -> None:
    paths = {
        "result_109": config.result_109,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 110 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_109.read_text(encoding="utf-8"))
    searched = prior.get("panels", {}).get(
        "searched_hybrid_vs_searched_generation_11", {}
    ).get("arena", {})
    if (
        prior.get("status") != "complete"
        or prior.get("decision") != "hybrid_internal_transfer_failed_close_consolidation_branch_no_install"
        or (searched.get("candidate_wins"), searched.get("draws"), searched.get("candidate_losses"))
        != (102, 2, 96)
        or prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]
    ):
        raise ValueError("Frozen 109 result lacks its exact branch-closing result.")
    _, identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    if identity.generation != 11 or identity.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen 110 parent is not exact generation 11.")
    if not 1 <= config.shard_workers <= 4 or not 1 <= config.arena_workers <= 2:
        raise ValueError("110 worker counts exceed the frozen concurrency bounds.")


def _jsonable_config(config: FreshDoseConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )
