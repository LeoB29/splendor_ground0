"""Three-panel rolling-lineage endpoint evaluation for programme 102."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .search import SearchConfig
from .search_primary import _arena_dict, _seat_score_shares, paired_score_share_bootstrap


@dataclass(frozen=True, slots=True)
class EndpointEvaluationConfig:
    output_dir: Path
    result_101: Path
    generation_2_checkpoint: Path
    generation_5_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "b3146dc52462278070c6e7332e99704487e80c550e1ba6b67322888fe8bd59f0"
EXPECTED_HASHES = {
    "result_101": "4e29b92f13232f31f082fd0a3c5eb00bf2bd1068a18956832b048cfb7bb04a05",
    "generation_2_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_5_checkpoint": "201e34128c9588722ce78edc85e8672675b1075dc0696651484718ddd7fa2444",
}
SEED_START = 1_800_000
PAIRED_SEEDS = 100
BOOTSTRAP_SAMPLES = 50_000


def practical_assessment(arena: dict[str, Any]) -> dict[str, Any]:
    seats = _seat_score_shares(arena)
    checks = {
        "score_share_at_least_55_percent": arena["score_share"] >= 0.55,
        "positive_margin": arena["mean_margin"] > 0.0,
        "seat_zero_nonlosing": seats[0] >= 0.5,
        "seat_one_nonlosing": seats[1] >= 0.5,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    passed = all(checks.values())
    if passed:
        category = "practical_strength_pass"
    elif (
        0.5 <= arena["score_share"] < 0.55
        and arena["mean_margin"] >= 0.0
        and checks["normal_terminations"]
        and checks["seat_zero_nonlosing"]
        and checks["seat_one_nonlosing"]
    ):
        category = "practically_flat"
    else:
        category = "failed_or_mixed"
    return {
        "passed": passed,
        "category": category,
        "checks": checks,
        "seat_score_shares": {str(key): value for key, value in seats.items()},
    }


def endpoint_decision(
    primary: dict[str, Any], raw: dict[str, Any], search_health: dict[str, Any]
) -> str:
    if primary["passed"]:
        if raw["passed"] and search_health["passed"]:
            return "endpoint_compounding_with_raw_transfer_and_healthy_search"
        if search_health["passed"]:
            return "endpoint_compounding_with_incomplete_raw_transfer"
        if raw["passed"]:
            return "endpoint_compounding_with_weak_own_search_lift"
        return "endpoint_compounding_with_mixed_mechanism_diagnostics"
    if primary["category"] == "practically_flat":
        return "endpoint_practically_flat_diagnose_before_scaling"
    if not search_health["passed"]:
        return "endpoint_failed_search_health_is_primary_suspect"
    if raw["passed"]:
        return "endpoint_failed_despite_raw_transfer_search_compatibility_suspect"
    return "endpoint_failed_with_incomplete_network_transfer"


def run_endpoint_evaluation_102(config: EndpointEvaluationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    search = SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_020_102,
    )
    result: dict[str, Any] = {
        "kind": "alphazero_endpoint_evaluation_102",
        "status": "running",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "confirmed_search": asdict(search),
        "schedule": {
            "seed_start": SEED_START,
            "paired_seeds": PAIRED_SEEDS,
            "games_per_panel": PAIRED_SEEDS * 2,
            "total_games": PAIRED_SEEDS * 2 * 3,
        },
        "panels": {},
        "endpoint_strength_claim": False,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.generation_2_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["generation_2_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)

    panel_specs = (
        (
            "searched_generation_5_vs_searched_generation_2",
            alpha_zero_search_factory(config.generation_5_checkpoint, config.device, search),
            alpha_zero_search_factory(config.generation_2_checkpoint, config.device, search),
        ),
        (
            "raw_generation_5_vs_raw_generation_2",
            alpha_zero_raw_factory(config.generation_5_checkpoint, config.device),
            alpha_zero_raw_factory(config.generation_2_checkpoint, config.device),
        ),
        (
            "generation_5_search_vs_generation_5_raw",
            alpha_zero_search_factory(config.generation_5_checkpoint, config.device, search),
            alpha_zero_raw_factory(config.generation_5_checkpoint, config.device),
        ),
    )
    for panel_index, (name, candidate_factory, baseline_factory) in enumerate(panel_specs):
        result["active_panel"] = name
        _write_progress(output, result, started)
        arena = _arena_dict(
            run_paired_arena(
                name=name,
                candidate_factory=candidate_factory,
                baseline_factory=baseline_factory,
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
                arena,
                samples=BOOTSTRAP_SAMPLES,
                seed=1_020_102 + panel_index,
            ),
            "assessment": practical_assessment(arena),
        }
        _write_progress(output, result, started)

    primary = result["panels"]["searched_generation_5_vs_searched_generation_2"]["assessment"]
    raw = result["panels"]["raw_generation_5_vs_raw_generation_2"]["assessment"]
    search_health = result["panels"]["generation_5_search_vs_generation_5_raw"]["assessment"]
    result["endpoint_strength_claim"] = primary["passed"]
    result["decision"] = endpoint_decision(primary, raw, search_health)
    result["status"] = "complete"
    result["active_panel"] = None
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result


def _verify_inputs(config: EndpointEvaluationConfig) -> None:
    paths = {
        "result_101": config.result_101,
        "generation_2_checkpoint": config.generation_2_checkpoint,
        "generation_5_checkpoint": config.generation_5_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 102 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_101.read_text(encoding="utf-8"))
    if prior.get("status") != "complete":
        raise ValueError("Frozen 101 result is not complete.")
    if prior.get("decision") != "clean_continuation_complete_review_trajectory":
        raise ValueError("Frozen 101 result has the wrong controlling decision.")
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["generation_2_checkpoint"]:
        raise ValueError("Frozen 101 result did not retain exact generation 2.")
    if prior.get("final_sandbox_checkpoint_sha256") != EXPECTED_HASHES["generation_5_checkpoint"]:
        raise ValueError("Frozen 101 result does not end at exact generation 5.")


def _jsonable_config(config: EndpointEvaluationConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
