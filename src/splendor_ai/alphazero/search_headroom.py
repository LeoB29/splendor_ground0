"""Direct 1,024-versus-256 simulation search-teacher headroom duel 113."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from .arena import ArenaResult, alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .network import load_checkpoint
from .scaled_continuation import _sha256, _write_json
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap


@dataclass(frozen=True, slots=True)
class SearchHeadroomConfig:
    output_dir: Path
    result_112: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None
    arena_workers: int = 4


PROTOCOL_SHA256 = "b1b87ad5ef402b850e95febbad7f9f270165e9cc62d7afc1ab29d7a2148272c0"
EXPECTED_HASHES = {
    "result_112": "bb42a38826990f94ecedd75b6f05649ac7461183a56bc1c6871502c33ed13251",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
}
SEED_START = 2_080_000
PAIRED_SEEDS = 200
SHARD_PAIRED_SEEDS = 50
SHARD_COUNT = 4
BOOTSTRAP_SAMPLES = 50_000
CANDIDATE_SIMULATIONS = 1_024
BASELINE_SIMULATIONS = 256


def search_headroom_assessment(
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
        "category": "teacher_headroom_pass" if passed else "teacher_headroom_not_established",
        "checks": checks,
        "seat_score_shares": practical["seat_score_shares"],
    }


def search_headroom_decision(assessment: dict[str, Any]) -> str:
    if bool(assessment["passed"]):
        return "deeper_search_teacher_headroom_confirmed_review_high_budget_distillation_no_install"
    return "deeper_search_teacher_headroom_not_confirmed_proceed_capacity_representation_pilot_no_install"


def run_search_headroom_113(config: SearchHeadroomConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 113 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_search_headroom_113",
        "status": "running",
        "active_stage": "four_shard_search_headroom_arena",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "contrast": {
            "candidate_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "candidate_simulations": CANDIDATE_SIMULATIONS,
            "baseline_simulations": BASELINE_SIMULATIONS,
            "simulation_multiplier": CANDIDATE_SIMULATIONS / BASELINE_SIMULATIONS,
            "network_held_fixed": True,
            "c_puct_held_fixed": True,
        },
        "arena": None,
        "paired_bootstrap": None,
        "assessment": None,
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    arena = combine_shard_arenas(_run_or_load_parallel_shards(config))
    bootstrap = paired_score_share_bootstrap(
        arena, samples=BOOTSTRAP_SAMPLES, seed=1_113_000
    )
    assessment = search_headroom_assessment(arena, bootstrap)
    result["arena"] = arena
    result["paired_bootstrap"] = bootstrap
    result["assessment"] = assessment
    result["decision"] = search_headroom_decision(assessment)
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def combine_shard_arenas(shards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(shards) != set(range(SHARD_COUNT)):
        raise ValueError("113 requires all four exact arena shards.")
    rows: list[dict[str, object]] = []
    for shard_index in range(SHARD_COUNT):
        arena = shards[shard_index]
        start = SEED_START + shard_index * SHARD_PAIRED_SEEDS
        expected = [
            (start + offset, seat)
            for offset in range(SHARD_PAIRED_SEEDS)
            for seat in (0, 1)
        ]
        shard_rows = list(arena.get("rows", []))
        observed = [
            (int(row["seed"]), int(row["candidate_seat"])) for row in shard_rows
        ]
        if (
            arena.get("paired_seeds") != SHARD_PAIRED_SEEDS
            or arena.get("games") != 2 * SHARD_PAIRED_SEEDS
            or observed != expected
        ):
            raise ValueError(f"113 shard {shard_index} is not its exact completed schedule.")
        rows.extend(shard_rows)
    full_schedule = [
        (SEED_START + offset, seat) for offset in range(PAIRED_SEEDS) for seat in (0, 1)
    ]
    if [(int(row["seed"]), int(row["candidate_seat"])) for row in rows] != full_schedule:
        raise ValueError("113 combined arena is not the exact frozen schedule.")
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    terminations: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        terminations[reason] = terminations.get(reason, 0) + 1
    return _arena_dict(
        ArenaResult(
            name="generation_11_search_1024_vs_256",
            paired_seeds=PAIRED_SEEDS,
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


def _run_or_load_parallel_shards(config: SearchHeadroomConfig) -> dict[int, dict[str, Any]]:
    manifest = config.output_dir / "arenas" / "parallel_shards.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_manifest(payload)
        shards = {int(index): arena for index, arena in payload["shards"].items()}
        combine_shard_arenas(shards)
        return shards
    tasks = [
        {
            "shard_index": index,
            "output_dir": str(config.output_dir / "arenas"),
            "generation_11_checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
        }
        for index in range(SHARD_COUNT)
    ]
    shards: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
        futures = {executor.submit(_shard_worker, task): task for task in tasks}
        for future in as_completed(futures):
            index, arena = future.result()
            shards[index] = arena
    combine_shard_arenas(shards)
    _write_json(
        manifest,
        {
            "protocol_sha256": PROTOCOL_SHA256,
            "stage": "parallel_shards",
            "input_hashes": dict(EXPECTED_HASHES),
            "arena_workers": config.arena_workers,
            "shards": {str(index): shards[index] for index in sorted(shards)},
        },
    )
    return shards


def _shard_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    _limit_worker_threads()
    index = int(task["shard_index"])
    checkpoint = Path(task["generation_11_checkpoint"])
    candidate = alpha_zero_search_factory(
        checkpoint, str(task["device"]), _search_config(CANDIDATE_SIMULATIONS)
    )
    baseline = alpha_zero_search_factory(
        checkpoint, str(task["device"]), _search_config(BASELINE_SIMULATIONS)
    )
    seed_start = SEED_START + index * SHARD_PAIRED_SEEDS
    arena_dir = Path(task["output_dir"])
    progress_path = arena_dir / f"shard_{index}.json"
    identity_path = arena_dir / f"shard_{index}.identity.json"
    identity = _shard_identity(index, int(task["max_turns"]))
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"113 shard {index} sidecar identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError(f"113 shard {index} progress exists without an identity sidecar.")
        _write_json(identity_path, identity)
    arena = _arena_dict(
        run_paired_arena(
            name=f"generation_11_search_1024_vs_256_shard_{index}",
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=seed_start,
            paired_seeds=SHARD_PAIRED_SEEDS,
            max_turns=int(task["max_turns"]),
            progress_path=progress_path,
            resume=True,
        )
    )
    return index, arena


def _search_config(simulations: int) -> SearchConfig:
    return SearchConfig(
        simulations=simulations,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_113_000,
    )


def _shard_identity(index: int, max_turns: int) -> dict[str, Any]:
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard_index": index,
        "candidate_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "candidate_search_config": asdict(_search_config(CANDIDATE_SIMULATIONS)),
        "baseline_search_config": asdict(_search_config(BASELINE_SIMULATIONS)),
        "operative_search_seed": "2 * game_seed + seat",
        "seed_start": SEED_START + index * SHARD_PAIRED_SEEDS,
        "paired_seeds": SHARD_PAIRED_SEEDS,
        "max_turns": max_turns,
    }


def _verify_inputs(config: SearchHeadroomConfig) -> None:
    paths = {
        "result_112": config.result_112,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 113 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_112.read_text(encoding="utf-8"))
    arena = prior.get("arena", {})
    if (
        prior.get("status") != "complete"
        or prior.get("decision")
        != "high_dose_value_path_causal_gameplay_not_confirmed_no_architecture_authorization_no_install"
        or prior.get("assessment", {}).get("passed") is not False
        or arena.get("candidate_wins") != 194
        or arena.get("draws") != 7
        or arena.get("candidate_losses") != 199
        or arena.get("termination_reasons") != {"completed": 400}
        or prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]
    ):
        raise ValueError("Frozen 112 result lacks its exact completed negative contract.")
    _, identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    if identity.generation != 11 or identity.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen 113 checkpoint is not exact generation 11.")
    if config.max_turns != 200 or config.torch_threads != 1 or config.arena_workers != 4:
        raise ValueError("113 requires the frozen 200-turn, one-thread, four-worker execution.")


def _validate_manifest(payload: dict[str, Any]) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("stage") != "parallel_shards"
        or payload.get("input_hashes") != EXPECTED_HASHES
        or payload.get("arena_workers") != 4
        or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
    ):
        raise ValueError("113 parallel shard manifest identity mismatch.")


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config: SearchHeadroomConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})
