"""Resumable deep-search teacher-refresh continuation 127."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import torch

from .arena import ArenaResult, alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .high_budget_distillation import _stream_clean_summary as _stream_114_summary
from .indexed_training import train_parent_safe_indexed_successor
from .network import load_checkpoint
from .replay import _row_from_payload
from .scaled_continuation import _sha256, _write_json
from .scaled_loop import training_validity
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .selfplay import SelfPlayConfig, generate_self_play
from .training import TrainingConfig


PROTOCOL_SHA256 = "a98987804328c2ba4bd32716b588ba553773ef1ecffec4cdf1f90e958dba2780"
EXPECTED_HASHES = {
    "result_114": "dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5",
    "result_115": "02d4d1d2557c57c6778b92950fb4c06d02e4112dc83d45d20791f638df0db993",
    "result_126": "f2043a8b86925081dca058b0643c1906ce9ac69fcda48c181364713e0b71136e",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "generation_12_checkpoint": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
}
EXPECTED_INITIAL_REPLAYS = (
    "f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b",
    "c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60",
    "00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8",
    "7bd6b03227aa6e4e5ce1bb902adda6021d438b5e927d1da2b9f56793424dd95e",
)
INITIAL_SEED_STARTS = (2_090_000, 2_100_000, 2_110_000, 2_120_000)
SELFPLAY_SEED_STARTS = {
    12: (2_300_000, 2_310_000, 2_320_000, 2_330_000),
    13: (2_400_000, 2_410_000, 2_420_000, 2_430_000),
}
GAMES_PER_SHARD = 64
MAX_REPLACEMENTS = 16
SELFPLAY_SIMULATIONS = 1_024
ENDPOINT_SEED_START = 2_500_000
ENDPOINT_PAIRED_SEEDS = 200
WORKERS = 4
BOOTSTRAP_SAMPLES = 50_000


@dataclass(frozen=True, slots=True)
class DeepSearchTeacherRefreshConfig:
    output_dir: Path
    result_114: Path
    result_115: Path
    result_126: Path
    generation_11_checkpoint: Path
    generation_12_checkpoint: Path
    installed_checkpoint: Path
    initial_replays: tuple[Path, Path, Path, Path]
    device: str = "cpu"
    torch_threads: int = 1
    shard_workers: int = 4
    arena_workers: int = 4
    max_turns: int = 200


def refresh_training_config(child_generation: int) -> TrainingConfig:
    if child_generation not in (13, 14):
        raise ValueError("127 only trains generations 13 and 14.")
    return TrainingConfig(
        epochs=12,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=1_127_000 + child_generation,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        bucket_batches=True,
        policy_divergence_weight=0.0,
        visit_argmax_aux_weight=0.0,
    )


def endpoint_assessment(arena: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    practical = practical_assessment(arena)
    checks = {
        **practical["checks"],
        "paired_ci95_low_strictly_above_half": float(bootstrap["ci95_low"]) > 0.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "seat_score_shares": practical["seat_score_shares"],
    }


def preflight_deep_search_teacher_refresh_127(config) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = _verify_inputs(config)
    summaries = [
        _stream_114_summary(
            path,
            expected_seeds=list(range(INITIAL_SEED_STARTS[index], INITIAL_SEED_STARTS[index] + 64)),
            expected_checkpoint_path=config.generation_11_checkpoint,
        )
        for index, path in enumerate(config.initial_replays)
    ]
    if sum(item["games"] for item in summaries) != 256:
        raise ValueError("127 starting replay does not contain exactly 256 games.")
    smoke = _run_smoke(config)
    result = {
        "kind": "alphazero_deep_search_teacher_refresh_127_preflight",
        "status": "preflight_complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": hashes,
        "starting_replay_summaries": summaries,
        "starting_replay_games": 256,
        "generation_schedule": [
            {"actor": 12, "child": 13},
            {"actor": 13, "child": 14},
        ],
        "selfplay_seed_starts": {str(key): list(value) for key, value in SELFPLAY_SEED_STARTS.items()},
        "endpoint_seed_start": ENDPOINT_SEED_START,
        "endpoint_paired_seeds": ENDPOINT_PAIRED_SEEDS,
        "training_configs": {str(child): asdict(refresh_training_config(child)) for child in (13, 14)},
        "smoke": smoke,
        "durable_selfplay_written": False,
        "training_performed": False,
        "arena_performed": False,
        "promotion_authorized": False,
        "launch_required": True,
    }
    _write_json(config.output_dir / "preflight.json", result)
    return result


def run_deep_search_teacher_refresh_127(config, *, launch: bool = False) -> dict[str, Any]:
    preflight = preflight_deep_search_teacher_refresh_127(config)
    if not launch:
        return preflight
    started = time.perf_counter()
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 127 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_deep_search_teacher_refresh_127",
        "status": "running",
        "active_stage": "selfplay_generation_12_actor",
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight": preflight,
        "generations": [],
        "endpoint": None,
        "installed_checkpoint_unchanged": True,
        "promotion_authorized": False,
        "automatic_continuation": False,
    }
    _write_progress(config.output_dir, result, started)
    parent = config.generation_12_checkpoint
    previous_replays = list(config.initial_replays)
    for actor_generation in (12, 13):
        child_generation = actor_generation + 1
        result["active_stage"] = f"selfplay_generation_{actor_generation}_actor"
        _write_progress(config.output_dir, result, started)
        selfplay = _run_or_load_parallel_selfplay(config, parent, actor_generation)
        current_replays = [Path(item["sanitized_path"]) for item in selfplay["shards"]]
        result["active_stage"] = f"train_generation_{child_generation}"
        _write_progress(config.output_dir, result, started)
        training = _run_or_load_training(
            config,
            parent=parent,
            parent_generation=actor_generation,
            replay_paths=previous_replays + current_replays,
        )
        record = {
            "actor_generation": actor_generation,
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": _sha256(parent),
            "selfplay": selfplay,
            "training": training,
            "child_generation": child_generation,
            "child_checkpoint": training["child_checkpoint"],
            "child_checkpoint_sha256": training["child_checkpoint_sha256"],
        }
        record_path = config.output_dir / "generation_records" / f"generation_{child_generation:03d}.json"
        _write_json(record_path, record)
        result["generations"].append(record)
        parent = Path(training["child_checkpoint"])
        previous_replays = current_replays
        result["active_stage"] = f"generation_{child_generation}_complete"
        _write_progress(config.output_dir, result, started)

    result["active_stage"] = "generation_14_vs_generation_11_endpoint"
    _write_progress(config.output_dir, result, started)
    arena = _run_or_load_endpoint(config, parent)
    bootstrap = paired_score_share_bootstrap(
        arena, samples=BOOTSTRAP_SAMPLES, seed=1_127_300
    )
    assessment = endpoint_assessment(arena, bootstrap)
    result["endpoint"] = {
        "arena": arena,
        "paired_bootstrap": bootstrap,
        "assessment": assessment,
    }
    result["decision"] = (
        "deep_search_refresh_endpoint_compounded_review_035_benchmark_no_install"
        if assessment["passed"]
        else "deep_search_refresh_endpoint_failed_close_current_distillation_recipe_no_install"
    )
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["installed_checkpoint_unchanged"] = (
        _sha256(config.installed_checkpoint) == EXPECTED_HASHES["installed_checkpoint"]
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(result_path, result)
    _write_progress(config.output_dir, result, started)
    return result


def _run_smoke(config) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="splendor_127_smoke_") as temporary:
        base = Path(temporary)
        search = SearchConfig(
            simulations=16,
            c_puct=2.25,
            mode="real",
            root_noise_fraction=0.25,
            seed=1_127_999,
        )
        generated = generate_self_play(
            checkpoint_path=config.generation_12_checkpoint,
            device=config.device,
            config=SelfPlayConfig(
                generation=12,
                game_count=1,
                seed_start=2_299_999,
                search=search,
                max_turns=config.max_turns,
            ),
            replay_path=base / "smoke.jsonl",
            progress_path=base / "smoke.progress.json",
            resume=True,
        )
        game = generated.games[0]
        if game.termination_reason != "completed" or game.positions <= 0:
            raise ValueError("127 preflight smoke did not complete normally.")
        return {
            "seed": 2_299_999,
            "simulations": 16,
            "turns": game.turns,
            "positions": game.positions,
            "termination_reason": game.termination_reason,
            "temporary_artifacts_removed": True,
        }


def _run_or_load_parallel_selfplay(config, parent: Path, generation: int) -> dict[str, Any]:
    manifest = config.output_dir / "selfplay" / f"generation_{generation:03d}.stage.json"
    parent_hash = _sha256(parent)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_parallel_manifest(config, payload, parent, generation)
        return payload
    tasks = [
        {
            "output_dir": str(config.output_dir),
            "checkpoint": str(parent),
            "checkpoint_sha256": parent_hash,
            "device": config.device,
            "max_turns": config.max_turns,
            "generation": generation,
            "shard": shard,
            "seed_start": SELFPLAY_SEED_STARTS[generation][shard],
        }
        for shard in range(WORKERS)
    ]
    records: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=config.shard_workers) as executor:
        futures = {executor.submit(_selfplay_shard_worker, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records[int(record["shard"])] = record
    shards = [records[index] for index in range(WORKERS)]
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "generation": generation,
        "checkpoint_sha256": parent_hash,
        "shard_workers": config.shard_workers,
        "shards": shards,
        "games": sum(item["summary"]["games"] for item in shards),
        "positions": sum(item["summary"]["positions"] for item in shards),
    }
    if payload["games"] != 256:
        raise RuntimeError("127 self-play did not produce exactly 256 clean games.")
    _write_json(manifest, payload)
    return payload


def _selfplay_shard_worker(task: dict[str, Any]) -> dict[str, Any]:
    _limit_worker_threads()
    generation = int(task["generation"])
    shard = int(task["shard"])
    seed_start = int(task["seed_start"])
    shard_dir = Path(task["output_dir"]) / "selfplay" / f"generation_{generation:03d}" / f"shard_{shard}"
    manifest = shard_dir / "clean_shard.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_shard_manifest(task, payload)
        return payload
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "generation": generation,
        "shard": shard,
        "seed_start": seed_start,
        "checkpoint_sha256": task["checkpoint_sha256"],
        "simulations": SELFPLAY_SIMULATIONS,
        "c_puct": 2.25,
        "root_noise_fraction": 0.25,
        "search_seed": 1_127_000 + generation,
        "max_turns": int(task["max_turns"]),
    }
    identity_path = shard_dir / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("127 self-play shard identity mismatch.")
    else:
        _write_json(identity_path, identity)
    attempted = []
    clean = []
    for seed in range(seed_start, seed_start + GAMES_PER_SHARD):
        record = _run_or_load_game(task, shard_dir, seed, replacement=False)
        attempted.append(record)
        if record["termination_reason"] == "completed":
            clean.append(record)
    for offset in range(MAX_REPLACEMENTS):
        if len(clean) >= GAMES_PER_SHARD:
            break
        seed = seed_start + GAMES_PER_SHARD + offset
        record = _run_or_load_game(task, shard_dir, seed, replacement=True)
        attempted.append(record)
        if record["termination_reason"] == "completed":
            clean.append(record)
    if len(clean) < GAMES_PER_SHARD:
        raise RuntimeError("127 shard could not obtain 64 clean games.")
    selected = clean[:GAMES_PER_SHARD]
    sanitized = shard_dir / "sanitized.jsonl"
    temporary = sanitized.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as destination:
        for record in selected:
            with Path(record["path"]).open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(sanitized)
    selected_seeds = [int(record["seed"]) for record in selected]
    summary = _stream_refresh_summary(
        sanitized,
        generation=generation,
        expected_seeds=selected_seeds,
        expected_checkpoint=Path(task["checkpoint"]),
    )
    payload = {
        **identity,
        "attempted_games": len(attempted),
        "abnormal_original_games": [
            record for record in attempted[:GAMES_PER_SHARD]
            if record["termination_reason"] != "completed"
        ],
        "replacement_games": attempted[GAMES_PER_SHARD:],
        "selected_clean_seeds": selected_seeds,
        "sanitized_path": str(sanitized),
        "sanitized_sha256": _sha256(sanitized),
        "summary": summary,
    }
    _write_json(manifest, payload)
    return payload


def _run_or_load_game(task, shard_dir: Path, seed: int, *, replacement: bool) -> dict[str, Any]:
    label = "replacement" if replacement else "original"
    path = shard_dir / "games" / f"{label}_{seed}.jsonl"
    game = generate_self_play(
        checkpoint_path=task["checkpoint"],
        device=task["device"],
        config=SelfPlayConfig(
            generation=int(task["generation"]),
            game_count=1,
            seed_start=seed,
            search=SearchConfig(
                simulations=SELFPLAY_SIMULATIONS,
                c_puct=2.25,
                mode="real",
                root_noise_fraction=0.25,
                seed=1_127_000 + int(task["generation"]),
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


def _stream_refresh_summary(path: Path, *, generation: int, expected_seeds: list[int], expected_checkpoint: Path) -> dict[str, Any]:
    expected = set(expected_seeds)
    order = []
    next_turn: dict[int, int] = {}
    positions = 0
    selected: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = _row_from_payload(json.loads(line))
            if row.game_seed not in expected:
                raise ValueError("127 replay contains an unexpected seed.")
            if row.game_seed not in next_turn:
                next_turn[row.game_seed] = 0
                order.append(row.game_seed)
            if row.turn_index != next_turn[row.game_seed]:
                raise ValueError("127 replay turns are not contiguous.")
            next_turn[row.game_seed] += 1
            if (
                row.generation != generation
                or row.checkpoint_generation != generation
                or Path(row.checkpoint_path) != expected_checkpoint
                or row.value_target is None
                or row.termination_reason != "completed"
                or int(row.search_telemetry.get("simulations", -1)) != SELFPLAY_SIMULATIONS
                or int(row.search_config.get("simulations", -1)) != SELFPLAY_SIMULATIONS
                or float(row.search_config.get("c_puct", -1)) != 2.25
                or row.search_config.get("mode") != "real"
                or float(row.search_config.get("root_noise_fraction", -1)) != 0.25
                or int(row.search_config.get("seed", -1)) != 1_127_000 + generation
            ):
                raise ValueError("127 replay row violates its frozen contract.")
            selected[str(row.action_payloads[row.selected_action_row]["action_type"])] += 1
            positions += 1
    if order != expected_seeds or set(next_turn) != expected:
        raise ValueError("127 replay seed order or coverage mismatch.")
    return {
        "valid": True,
        "games": len(expected_seeds),
        "positions": positions,
        "mean_turns": positions / len(expected_seeds),
        "termination_reasons": {"completed": len(expected_seeds)},
        "selected_action_family_counts": dict(sorted(selected.items())),
    }


def _validate_shard_manifest(task, payload) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("generation") != task["generation"]
        or payload.get("shard") != task["shard"]
        or payload.get("seed_start") != task["seed_start"]
        or payload.get("checkpoint_sha256") != task["checkpoint_sha256"]
        or payload.get("sanitized_sha256") != _sha256(payload["sanitized_path"])
    ):
        raise ValueError("127 clean shard manifest mismatch.")
    summary = _stream_refresh_summary(
        Path(payload["sanitized_path"]),
        generation=int(task["generation"]),
        expected_seeds=[int(seed) for seed in payload["selected_clean_seeds"]],
        expected_checkpoint=Path(task["checkpoint"]),
    )
    if summary != payload["summary"]:
        raise ValueError("127 clean shard summary mismatch.")


def _validate_parallel_manifest(config, payload, parent: Path, generation: int) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("generation") != generation
        or payload.get("checkpoint_sha256") != _sha256(parent)
        or payload.get("games") != 256
        or payload.get("shard_workers") != 4
        or len(payload.get("shards", [])) != 4
    ):
        raise ValueError("127 parallel self-play manifest mismatch.")
    for item in payload["shards"]:
        shard = int(item["shard"])
        _validate_shard_manifest(
            {
                "checkpoint": str(parent),
                "checkpoint_sha256": _sha256(parent),
                "generation": generation,
                "shard": shard,
                "seed_start": SELFPLAY_SEED_STARTS[generation][shard],
            },
            item,
        )


def _run_or_load_training(config, *, parent: Path, parent_generation: int, replay_paths: list[Path]) -> dict[str, Any]:
    child_generation = parent_generation + 1
    stage = config.output_dir / "training" / f"generation_{child_generation:03d}.stage.json"
    expected_replays = [_sha256(path) for path in replay_paths]
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        _validate_training_stage(config, payload, parent, parent_generation, expected_replays)
        payload["training"] = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
        payload["training_validity"] = training_validity(payload["training"], expected_parent=parent_generation)
        return payload
    child = config.output_dir / "checkpoints" / f"generation_{child_generation:03d}.pt"
    metrics = config.output_dir / "training" / f"generation_{child_generation:03d}.json"
    training = train_parent_safe_indexed_successor(
        parent_checkpoint=parent,
        replay_paths=replay_paths,
        output_checkpoint=child,
        metrics_path=metrics,
        device=config.device,
        config=refresh_training_config(child_generation),
    )
    validity = training_validity(training, expected_parent=parent_generation)
    _, identity = load_checkpoint(child, device=config.device)
    validity["checks"]["checkpoint_roundtrip_generation"] = identity.generation == child_generation
    validity["checks"]["parent_safe_metadata"] = (
        identity.metadata.get("parent_safe_selection") is True
        and identity.metadata.get("selected_epoch") == training["selected_epoch"]
    )
    validity["valid"] = all(validity["checks"].values())
    if not validity["valid"]:
        raise RuntimeError("127 training failed validity checks.")
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": _sha256(parent),
        "parent_generation": parent_generation,
        "replay_paths": [str(path) for path in replay_paths],
        "replay_sha256": expected_replays,
        "child_checkpoint": str(child),
        "child_checkpoint_sha256": _sha256(child),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
        "training": training,
        "training_validity": validity,
    }
    _write_json(stage, {key: value for key, value in payload.items() if key not in {"training", "training_validity"}})
    return payload


def _validate_training_stage(config, payload, parent, parent_generation, expected_replays) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("parent_checkpoint_sha256") != _sha256(parent)
        or payload.get("parent_generation") != parent_generation
        or payload.get("replay_sha256") != expected_replays
        or payload.get("child_checkpoint_sha256") != _sha256(payload["child_checkpoint"])
        or payload.get("metrics_sha256") != _sha256(payload["metrics_path"])
    ):
        raise ValueError("127 training stage mismatch.")
    training = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
    validity = training_validity(training, expected_parent=parent_generation)
    _, identity = load_checkpoint(payload["child_checkpoint"], device=config.device)
    if not validity["valid"] or identity.generation != parent_generation + 1:
        raise ValueError("127 training stage lineage invalid.")


def _run_or_load_endpoint(config, candidate: Path) -> dict[str, Any]:
    endpoint_dir = config.output_dir / "endpoint"
    manifest = endpoint_dir / "endpoint.stage.json"
    candidate_hash = _sha256(candidate)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("candidate_checkpoint_sha256") != candidate_hash
            or payload.get("baseline_checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
            or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
        ):
            raise ValueError("127 endpoint manifest mismatch.")
        return _combine_endpoint_shards({int(key): value for key, value in payload["shards"].items()})
    tasks = [
        {
            "shard": shard,
            "output_dir": str(endpoint_dir),
            "candidate": str(candidate),
            "candidate_hash": candidate_hash,
            "baseline": str(config.generation_11_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
        }
        for shard in range(WORKERS)
    ]
    shards = {}
    with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
        futures = {executor.submit(_endpoint_shard_worker, task): task for task in tasks}
        for future in as_completed(futures):
            shard, arena = future.result()
            shards[shard] = arena
    combined = _combine_endpoint_shards(shards)
    _write_json(
        manifest,
        {
            "protocol_sha256": PROTOCOL_SHA256,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "shards": {str(key): shards[key] for key in sorted(shards)},
        },
    )
    return combined


def _endpoint_shard_worker(task) -> tuple[int, dict[str, Any]]:
    _limit_worker_threads()
    shard = int(task["shard"])
    paired = ENDPOINT_PAIRED_SEEDS // WORKERS
    start = ENDPOINT_SEED_START + shard * paired
    output = Path(task["output_dir"])
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard": shard,
        "candidate_checkpoint_sha256": task["candidate_hash"],
        "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "seed_start": start,
        "paired_seeds": paired,
        "search": asdict(_endpoint_search()),
        "max_turns": int(task["max_turns"]),
    }
    identity_path = output / f"shard_{shard}.identity.json"
    progress = output / f"shard_{shard}.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("127 endpoint shard identity mismatch.")
    else:
        if progress.exists():
            raise ValueError("127 endpoint progress lacks identity.")
        _write_json(identity_path, identity)
    candidate = alpha_zero_search_factory(task["candidate"], task["device"], _endpoint_search())
    baseline = alpha_zero_search_factory(task["baseline"], task["device"], _endpoint_search())
    arena = _arena_dict(
        run_paired_arena(
            name=f"deep_search_refresh_endpoint_shard_{shard}",
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=start,
            paired_seeds=paired,
            max_turns=int(task["max_turns"]),
            progress_path=progress,
            resume=True,
        )
    )
    return shard, arena


def _combine_endpoint_shards(shards) -> dict[str, Any]:
    if set(shards) != set(range(WORKERS)):
        raise ValueError("127 endpoint requires four shards.")
    rows = []
    paired = ENDPOINT_PAIRED_SEEDS // WORKERS
    for shard in range(WORKERS):
        arena = shards[shard]
        start = ENDPOINT_SEED_START + shard * paired
        expected = [(start + offset, seat) for offset in range(paired) for seat in (0, 1)]
        shard_rows = list(arena["rows"])
        if [(int(row["seed"]), int(row["candidate_seat"])) for row in shard_rows] != expected:
            raise ValueError("127 endpoint shard schedule mismatch.")
        rows.extend(shard_rows)
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    return _arena_dict(
        ArenaResult(
            name="searched_generation_14_vs_searched_generation_11",
            paired_seeds=ENDPOINT_PAIRED_SEEDS,
            games=len(rows),
            candidate_wins=wins,
            candidate_losses=losses,
            draws=draws,
            score_share=(wins + 0.5 * draws) / len(rows),
            mean_margin=sum(float(row["candidate_margin"]) for row in rows) / len(rows),
            termination_reasons=dict(sorted(reasons.items())),
            rows=tuple(rows),
        )
    )


def _endpoint_search() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_127_200,
    )


def _verify_inputs(config) -> dict[str, Any]:
    observed = {
        "result_114": _sha256(config.result_114),
        "result_115": _sha256(config.result_115),
        "result_126": _sha256(config.result_126),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "generation_12_checkpoint": _sha256(config.generation_12_checkpoint),
        "installed_checkpoint": _sha256(config.installed_checkpoint),
    }
    replay_hashes = tuple(_sha256(path) for path in config.initial_replays)
    if observed != EXPECTED_HASHES or replay_hashes != EXPECTED_INITIAL_REPLAYS:
        raise ValueError(f"127 frozen input mismatch: {observed}, {replay_hashes}")
    result_114 = json.loads(config.result_114.read_text(encoding="utf-8"))
    result_115 = json.loads(config.result_115.read_text(encoding="utf-8"))
    result_126 = json.loads(config.result_126.read_text(encoding="utf-8"))
    if (
        result_114.get("status") != "complete"
        or result_114.get("training", {}).get("checkpoint_sha256") != EXPECTED_HASHES["generation_12_checkpoint"]
        or result_114.get("training", {}).get("training_validity", {}).get("valid") is not True
        or result_115.get("decision") != "pooled_high_budget_distillation_not_confirmed_proceed_medium_capacity_pilot_no_install"
        or result_126.get("decision") != "start_role_not_operationally_material_close_representation_branch"
        or result_126.get("promotion_authorized") is not False
    ):
        raise ValueError("127 does not reproduce finalized 114/115/126.")
    _, parent_identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    _, candidate_identity = load_checkpoint(config.generation_12_checkpoint, device=config.device)
    if (
        parent_identity.generation != 11
        or candidate_identity.generation != 12
        or candidate_identity.metadata.get("parent_generation") != 11
        or config.torch_threads != 1
        or config.shard_workers != 4
        or config.arena_workers != 4
        or config.max_turns != 200
    ):
        raise ValueError("127 lineage or execution settings mismatch.")
    return {**observed, "initial_replays": list(replay_hashes)}


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config) -> dict[str, Any]:
    payload = asdict(config)
    return {
        key: ([str(item) for item in value] if key == "initial_replays" else str(value) if isinstance(value, Path) else value)
        for key, value in payload.items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})
