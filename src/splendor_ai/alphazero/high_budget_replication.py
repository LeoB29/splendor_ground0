"""Fresh and prospectively pooled high-budget replication 115."""

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
class HighBudgetReplicationConfig:
    output_dir: Path
    result_114: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    candidate_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None
    arena_workers: int = 4


PROTOCOL_SHA256 = "174b65e51184051444d3acb84ceca98f609aa6ffabaf5f5746348c0e0332e6e9"
EXPECTED_HASHES = {
    "result_114": "dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "candidate_checkpoint": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
}
PRIOR_SEED_START = 2_140_000
FRESH_SEED_START = 2_150_000
PAIRED_SEEDS = 100
SHARD_COUNT = 4
SHARD_PAIRED_SEEDS = 25
SIMULATIONS = 1_024
BOOTSTRAP_SAMPLES = 50_000


def confirmation_assessment(
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
        "category": "pooled_high_budget_absorption_confirmed" if passed else "pooled_high_budget_absorption_not_confirmed",
        "checks": checks,
        "seat_score_shares": practical["seat_score_shares"],
    }


def high_budget_replication_decision(assessment: dict[str, Any]) -> str:
    if bool(assessment["passed"]):
        return "pooled_high_budget_distillation_confirmed_review_short_1024_continuation_no_install"
    return "pooled_high_budget_distillation_not_confirmed_proceed_medium_capacity_pilot_no_install"


def run_high_budget_replication_115(config: HighBudgetReplicationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prior = _verify_inputs(config)
    sealed = output / "result.json"
    if sealed.exists():
        payload = json.loads(sealed.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 115 result is not a valid completed result.")
    anchor = prior["panels"]["diagnostic_1024"]["arena"]
    result: dict[str, Any] = {
        "kind": "alphazero_high_budget_replication_115",
        "status": "running",
        "active_stage": "four_shard_fresh_replication",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "prior_anchor": {
            "source": str(config.result_114),
            "arena": anchor,
            "paired_bootstrap": prior["panels"]["diagnostic_1024"]["paired_bootstrap"],
        },
        "fresh_replication": None,
        "pooled_primary": None,
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    fresh = combine_fresh_shards(_run_or_load_parallel_shards(config))
    fresh_bootstrap = paired_score_share_bootstrap(
        fresh, samples=BOOTSTRAP_SAMPLES, seed=1_115_000
    )
    result["fresh_replication"] = {
        "arena": fresh,
        "paired_bootstrap": fresh_bootstrap,
        "assessment": practical_assessment(fresh),
    }
    result["active_stage"] = "prospectively_pooled_primary"
    _write_progress(output, result, started)
    pooled = pool_prior_and_fresh(anchor, fresh)
    pooled_bootstrap = paired_score_share_bootstrap(
        pooled, samples=BOOTSTRAP_SAMPLES, seed=1_115_100
    )
    assessment = confirmation_assessment(pooled, pooled_bootstrap)
    result["pooled_primary"] = {
        "arena": pooled,
        "paired_bootstrap": pooled_bootstrap,
        "assessment": assessment,
    }
    result["decision"] = high_budget_replication_decision(assessment)
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed, result)
    return result


def combine_fresh_shards(shards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(shards) != set(range(SHARD_COUNT)):
        raise ValueError("115 requires all four exact fresh shards.")
    rows: list[dict[str, object]] = []
    for index in range(SHARD_COUNT):
        arena = shards[index]
        start = FRESH_SEED_START + index * SHARD_PAIRED_SEEDS
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
            raise ValueError(f"115 shard {index} is not its exact completed schedule.")
        rows.extend(shard_rows)
    return _summarize_rows("fresh_high_budget_replication", rows, PAIRED_SEEDS)


def pool_prior_and_fresh(
    prior: dict[str, Any], fresh: dict[str, Any]
) -> dict[str, Any]:
    _validate_arena_schedule(prior, PRIOR_SEED_START, PAIRED_SEEDS, "114 anchor")
    _validate_arena_schedule(fresh, FRESH_SEED_START, PAIRED_SEEDS, "115 fresh")
    rows = [*prior["rows"], *fresh["rows"]]
    keys = [(int(row["seed"]), int(row["candidate_seat"])) for row in rows]
    if len(keys) != 400 or len(set(keys)) != 400:
        raise ValueError("115 pooled primary does not contain 400 unique scheduled games.")
    return _summarize_rows("pooled_114_115_high_budget", rows, 2 * PAIRED_SEEDS)


def _summarize_rows(
    name: str, rows: list[dict[str, object]], paired_seeds: int
) -> dict[str, Any]:
    if len(rows) != 2 * paired_seeds:
        raise ValueError("115 arena summary requires its complete paired schedule.")
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    terminations: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        terminations[reason] = terminations.get(reason, 0) + 1
    return _arena_dict(
        ArenaResult(
            name=name,
            paired_seeds=paired_seeds,
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


def _validate_arena_schedule(
    arena: dict[str, Any], seed_start: int, paired_seeds: int, label: str
) -> None:
    expected = [
        (seed_start + offset, seat) for offset in range(paired_seeds) for seat in (0, 1)
    ]
    observed = [
        (int(row["seed"]), int(row["candidate_seat"])) for row in arena.get("rows", [])
    ]
    if (
        arena.get("paired_seeds") != paired_seeds
        or arena.get("games") != 2 * paired_seeds
        or observed != expected
    ):
        raise ValueError(f"115 {label} is not its exact frozen schedule.")


def _run_or_load_parallel_shards(
    config: HighBudgetReplicationConfig,
) -> dict[int, dict[str, Any]]:
    manifest = config.output_dir / "arenas" / "parallel_shards.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_manifest(payload)
        shards = {int(index): arena for index, arena in payload["shards"].items()}
        combine_fresh_shards(shards)
        return shards
    tasks = [
        {
            "shard_index": index,
            "output_dir": str(config.output_dir / "arenas"),
            "candidate_checkpoint": str(config.candidate_checkpoint),
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
    combine_fresh_shards(shards)
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
    search = _search_config()
    candidate = alpha_zero_search_factory(task["candidate_checkpoint"], task["device"], search)
    baseline = alpha_zero_search_factory(task["generation_11_checkpoint"], task["device"], search)
    seed_start = FRESH_SEED_START + index * SHARD_PAIRED_SEEDS
    arena_dir = Path(task["output_dir"])
    progress = arena_dir / f"shard_{index}.json"
    identity_path = arena_dir / f"shard_{index}.identity.json"
    identity = _shard_identity(index, int(task["max_turns"]))
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"115 shard {index} sidecar identity mismatch.")
    else:
        if progress.exists():
            raise ValueError(f"115 shard {index} progress exists without identity.")
        _write_json(identity_path, identity)
    arena = _arena_dict(
        run_paired_arena(
            name=f"fresh_high_budget_replication_shard_{index}",
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=seed_start,
            paired_seeds=SHARD_PAIRED_SEEDS,
            max_turns=int(task["max_turns"]),
            progress_path=progress,
            resume=True,
        )
    )
    return index, arena


def _search_config() -> SearchConfig:
    return SearchConfig(
        simulations=SIMULATIONS,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_115_000,
    )


def _shard_identity(index: int, max_turns: int) -> dict[str, Any]:
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard_index": index,
        "candidate_checkpoint_sha256": EXPECTED_HASHES["candidate_checkpoint"],
        "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "candidate_search_config": asdict(_search_config()),
        "baseline_search_config": asdict(_search_config()),
        "operative_search_seed": "2 * game_seed + seat",
        "seed_start": FRESH_SEED_START + index * SHARD_PAIRED_SEEDS,
        "paired_seeds": SHARD_PAIRED_SEEDS,
        "max_turns": max_turns,
    }


def _verify_inputs(config: HighBudgetReplicationConfig) -> dict[str, Any]:
    paths = {
        "result_114": config.result_114,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "candidate_checkpoint": config.candidate_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 115 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_114.read_text(encoding="utf-8"))
    primary = prior.get("panels", {}).get("primary_256", {}).get("arena", {})
    diagnostic = prior.get("panels", {}).get("diagnostic_1024", {}).get("arena", {})
    if (
        prior.get("status") != "complete"
        or prior.get("decision")
        != "high_budget_gain_requires_expensive_search_review_representation_capacity_no_install"
        or prior.get("offline_qualified") is not True
        or prior.get("training", {}).get("checkpoint_sha256") != EXPECTED_HASHES["candidate_checkpoint"]
        or prior.get("training", {}).get("training", {}).get("selected_epoch") != 1
        or (primary.get("candidate_wins"), primary.get("draws"), primary.get("candidate_losses"))
        != (207, 3, 190)
        or (diagnostic.get("candidate_wins"), diagnostic.get("draws"), diagnostic.get("candidate_losses"))
        != (111, 1, 88)
        or diagnostic.get("termination_reasons") != {"completed": 200}
        or prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]
    ):
        raise ValueError("Frozen 114 result lacks its exact completed contract.")
    _validate_arena_schedule(diagnostic, PRIOR_SEED_START, PAIRED_SEEDS, "114 anchor")
    _, parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    _, candidate = load_checkpoint(config.candidate_checkpoint, device=config.device)
    if parent.generation != 11 or parent.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen 115 baseline is not exact generation 11.")
    if (
        candidate.generation != 12
        or candidate.metadata.get("parent_generation") != 11
        or candidate.metadata.get("selected_epoch") != 1
        or candidate.metadata.get("parent_safe_selection") is not True
    ):
        raise ValueError("Frozen 115 candidate has invalid lineage.")
    if config.max_turns != 200 or config.torch_threads != 1 or config.arena_workers != 4:
        raise ValueError("115 requires the frozen 200-turn, one-thread, four-worker execution.")
    return prior


def _validate_manifest(payload: dict[str, Any]) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("stage") != "parallel_shards"
        or payload.get("input_hashes") != EXPECTED_HASHES
        or payload.get("arena_workers") != 4
        or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
    ):
        raise ValueError("115 parallel shard manifest identity mismatch.")


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config: HighBudgetReplicationConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})
