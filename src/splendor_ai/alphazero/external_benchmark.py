"""Generation-11 external benchmark against raw legal 035 for programme 106."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

from .arena import (
    alpha_zero_raw_factory,
    alpha_zero_search_factory,
    policy_035_factory,
    run_paired_arena,
)
from .head_attribution import paired_arena_difference_bootstrap
from .network import load_checkpoint
from .scaled_continuation import _sha256, _write_json
from .search import SearchConfig
from .search_primary import (
    _arena_dict,
    _percentile,
    _seat_score_shares,
    paired_score_share_bootstrap,
)


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkConfig:
    output_dir: Path
    result_105: Path
    result_103: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    checkpoint_035: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "8fef3d7b1ad081da9b735046e01c262af1288f1353cd1d816aa9eea7137805d2"
EXPECTED_HASHES = {
    "result_105": "71c2cb3564468059b7d82906d5e18f6fd8f4c144475c4e353dcc66156616a401",
    "result_103": "0152c113dee56bd156483b0ab6d3a55d48e35b70cd3bcc3f5f9c0b5c2c0f7f96",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "checkpoint_035": "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd",
}
SEED_START = 1_940_000
PAIRED_SEEDS = 100
BOOTSTRAP_SAMPLES = 50_000


def independent_paired_difference_bootstrap(
    current: dict[str, Any],
    anchor: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    current_clusters = list(_arena_clusters(current).values())
    anchor_clusters = list(_arena_clusters(anchor).values())
    if set(int(row["seed"]) for row in current["rows"]) & set(
        int(row["seed"]) for row in anchor["rows"]
    ):
        raise ValueError("Independent comparison requires disjoint seed panels.")
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(current_clusters) for _ in current_clusters)
        / len(current_clusters)
        - sum(rng.choice(anchor_clusters) for _ in anchor_clusters)
        / len(anchor_clusters)
        for _ in range(samples)
    )
    current_mean = sum(current_clusters) / len(current_clusters)
    anchor_mean = sum(anchor_clusters) / len(anchor_clusters)
    return {
        "current_paired_seeds": len(current_clusters),
        "anchor_paired_seeds": len(anchor_clusters),
        "samples": samples,
        "seed": seed,
        "current_mean": current_mean,
        "anchor_mean": anchor_mean,
        "mean_difference": current_mean - anchor_mean,
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def external_parity_assessment(arena: dict[str, Any]) -> dict[str, Any]:
    seats = _seat_score_shares(arena)
    checks = {
        "score_share_at_least_50_percent": arena["score_share"] >= 0.5,
        "positive_margin": arena["mean_margin"] > 0.0,
        "seat_zero_nonlosing": seats[0] >= 0.5,
        "seat_one_nonlosing": seats[1] >= 0.5,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "seat_score_shares": {str(key): value for key, value in seats.items()},
    }


def external_decision(
    parity: dict[str, Any], searched_progress: dict[str, Any]
) -> str:
    if parity["passed"]:
        return "generation_11_reached_035_practical_parity_review_no_install"
    if float(searched_progress["ci95_low"]) > 0.0:
        return "generation_11_external_progress_confirmed_below_035_no_install"
    if float(searched_progress["mean_difference"]) > 0.0:
        return "generation_11_external_progress_directional_below_035_no_install"
    return "generation_11_no_external_progress_detected_below_035_no_install"


def run_external_benchmark_106(config: ExternalBenchmarkConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 106 result is not a valid completed result.")

    search = SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_060_106,
    )
    result: dict[str, Any] = {
        "kind": "alphazero_generation_11_external_benchmark_106",
        "status": "running",
        "active_stage": "searched_generation_11_vs_raw_035",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "confirmed_search": asdict(search),
        "panels": {},
        "comparisons": {},
        "external_035_teacher_used": False,
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    specs = (
        (
            "searched_generation_11_vs_raw_035",
            alpha_zero_search_factory(config.generation_11_checkpoint, config.device, search),
        ),
        (
            "raw_generation_11_vs_raw_035",
            alpha_zero_raw_factory(config.generation_11_checkpoint, config.device),
        ),
    )
    for index, (name, candidate) in enumerate(specs):
        result["active_stage"] = name
        _write_progress(output, result, started)
        arena = _arena_dict(
            run_paired_arena(
                name=name,
                candidate_factory=candidate,
                baseline_factory=policy_035_factory(config.checkpoint_035, config.device),
                seed_start=SEED_START,
                paired_seeds=PAIRED_SEEDS,
                max_turns=config.max_turns,
                progress_path=output / "arenas" / f"{name}.json",
                resume=True,
            )
        )
        result["panels"][name] = {
            "arena": arena,
            "paired_bootstrap": paired_score_share_bootstrap(
                arena, samples=BOOTSTRAP_SAMPLES, seed=1_060_300 + index
            ),
            "parity_assessment": external_parity_assessment(arena),
        }
        _write_progress(output, result, started)

    prior = json.loads(config.result_103.read_text(encoding="utf-8"))
    searched = result["panels"]["searched_generation_11_vs_raw_035"]["arena"]
    raw = result["panels"]["raw_generation_11_vs_raw_035"]["arena"]
    searched_anchor = prior["benchmark_035"]["searched_generation_8_vs_raw_035"]["arena"]
    raw_anchor = prior["benchmark_035"]["raw_generation_8_vs_raw_035"]["arena"]
    searched_progress = independent_paired_difference_bootstrap(
        searched,
        searched_anchor,
        samples=BOOTSTRAP_SAMPLES,
        seed=1_060_400,
    )
    result["comparisons"] = {
        "searched_generation_11_minus_searched_generation_8_vs_035": searched_progress,
        "raw_generation_11_minus_raw_generation_8_vs_035": independent_paired_difference_bootstrap(
            raw,
            raw_anchor,
            samples=BOOTSTRAP_SAMPLES,
            seed=1_060_401,
        ),
        "generation_11_search_minus_raw_vs_035": paired_arena_difference_bootstrap(
            searched,
            raw,
            samples=BOOTSTRAP_SAMPLES,
            seed=1_060_402,
        ),
    }
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = external_decision(
        result["panels"]["searched_generation_11_vs_raw_035"]["parity_assessment"],
        searched_progress,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def _arena_clusters(arena: dict[str, Any]) -> dict[int, float]:
    clusters: dict[int, list[float]] = {}
    for row in arena["rows"]:
        clusters.setdefault(int(row["seed"]), []).append(float(row["candidate_points"]))
    if not clusters or any(len(points) != 2 for points in clusters.values()):
        raise ValueError("Arena requires exactly two candidate seats per seed.")
    return {seed: sum(points) / 2.0 for seed, points in clusters.items()}


def _verify_inputs(config: ExternalBenchmarkConfig) -> None:
    paths = {
        "result_105": config.result_105,
        "result_103": config.result_103,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "checkpoint_035": config.checkpoint_035,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 106 input hash mismatch for {name}: {actual}")
    result_105 = json.loads(config.result_105.read_text(encoding="utf-8"))
    if result_105.get("status") != "complete" or result_105.get("decision") != "pooled_compounding_confirmed_review_035_no_install":
        raise ValueError("Frozen 105 result has the wrong controlling decision.")
    pooled = result_105["pooled_assessment"]
    if (
        not pooled["assessment"]["passed"]
        or pooled["arena"]["candidate_wins"] != 220
        or pooled["arena"]["draws"] != 1
        or pooled["arena"]["candidate_losses"] != 179
    ):
        raise ValueError("Frozen 105 result lacks the exact pooled pass.")
    result_103 = json.loads(config.result_103.read_text(encoding="utf-8"))
    searched_anchor = result_103["benchmark_035"]["searched_generation_8_vs_raw_035"]["arena"]
    raw_anchor = result_103["benchmark_035"]["raw_generation_8_vs_raw_035"]["arena"]
    if (
        (searched_anchor["candidate_wins"], searched_anchor["draws"], searched_anchor["candidate_losses"])
        != (53, 0, 147)
        or (raw_anchor["candidate_wins"], raw_anchor["draws"], raw_anchor["candidate_losses"])
        != (24, 1, 175)
    ):
        raise ValueError("Frozen 103 result lacks the exact gen8 external anchors.")
    _, identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    if identity.generation != 11 or identity.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen candidate is not exact generation 11.")
    if result_105.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]:
        raise ValueError("Frozen 105 result did not retain exact generation 2.")


def _jsonable_config(config: ExternalBenchmarkConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})

