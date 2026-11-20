"""Adaptive exact-candidate searched replication for programme 098."""

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
class AdaptiveReplicationConfig:
    output_dir: Path
    result_097: Path
    incumbent_checkpoint: Path
    candidate_checkpoint: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "01e3ba323568f9e31d398e2bbfe8cade7fd59c1ae209bb0a77f00c9e352d21f6"
EXPECTED_HASHES = {
    "result_097": "d8d8de1521065389bdfa003de0bd09f77b9371587ae2e821e941894c1796d5cf",
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "candidate_checkpoint": "59b17517ae0f0e1f996f05bca7c66acaeecd710862066f22c2eec99da983bdfa",
}


def adaptive_replication_acceptance(
    arena: dict[str, Any], bootstrap: dict[str, float | int]
) -> dict[str, Any]:
    seats = _seat_score_shares(arena)
    checks = {
        "score_share": arena["score_share"] >= 0.55,
        "bootstrap_ci95_low_above_half": float(bootstrap["ci95_low"]) > 0.5,
        "positive_margin": arena["mean_margin"] > 0.0,
        "seat_zero_nonlosing": seats[0] >= 0.5,
        "seat_one_nonlosing": seats[1] >= 0.5,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "seat_score_shares": {str(key): value for key, value in seats.items()},
    }


def run_adaptive_replication_098(config: AdaptiveReplicationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    search = SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=980_098,
    )
    result: dict[str, Any] = {
        "kind": "alphazero_adaptive_replication_098",
        "status": "running",
        "strength_claim": False,
        "adaptive_followup": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "candidate_name": "visit_argmax_aux_097_exact",
        "confirmed_search": asdict(search),
        "searched_replication": None,
        "raw_diagnostic": None,
        "installed_checkpoint": str(config.incumbent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.incumbent_checkpoint),
        "external_035_arena": None,
        "decision": "running",
    }
    _write_progress(output, result, started)

    searched = _arena_dict(
        run_paired_arena(
            name="visit_argmax_aux_097_search_vs_generation_2_adaptive_replication",
            candidate_factory=alpha_zero_search_factory(
                config.candidate_checkpoint, config.device, search
            ),
            baseline_factory=alpha_zero_search_factory(
                config.incumbent_checkpoint, config.device, search
            ),
            seed_start=1_690_000,
            paired_seeds=100,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_replication.json",
            resume=True,
        )
    )
    bootstrap = paired_score_share_bootstrap(
        searched, samples=50_000, seed=980_098
    )
    acceptance = adaptive_replication_acceptance(searched, bootstrap)
    result["searched_replication"] = {
        "arena": searched,
        "paired_bootstrap": bootstrap,
        "acceptance": acceptance,
    }
    _write_progress(output, result, started)
    if not acceptance["accepted"]:
        if (
            acceptance["checks"]["score_share"]
            and acceptance["checks"]["bootstrap_ci95_low_above_half"]
            and acceptance["checks"]["positive_margin"]
            and acceptance["checks"]["normal_terminations"]
        ):
            decision = "replication_overall_pass_seat_failure_keep_generation_2"
        else:
            decision = "replication_overall_failed_keep_generation_2"
        return _finish(output, result, started, decision)

    result["installed_checkpoint"] = str(config.candidate_checkpoint)
    result["installed_checkpoint_sha256"] = _sha256(config.candidate_checkpoint)
    _write_progress(output, result, started)
    raw = _arena_dict(
        run_paired_arena(
            name="visit_argmax_aux_097_raw_vs_generation_2_raw_diagnostic",
            candidate_factory=alpha_zero_raw_factory(
                config.candidate_checkpoint, config.device
            ),
            baseline_factory=alpha_zero_raw_factory(
                config.incumbent_checkpoint, config.device
            ),
            seed_start=1_690_000,
            paired_seeds=100,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "raw_diagnostic.json",
            resume=True,
        )
    )
    result["raw_diagnostic"] = raw
    return _finish(
        output,
        result,
        started,
        "generation_3_adaptive_replication_confirmed_and_installed",
    )


def _verify_inputs(config: AdaptiveReplicationConfig) -> None:
    paths = {
        "result_097": config.result_097,
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "candidate_checkpoint": config.candidate_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 098 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_097.read_text(encoding="utf-8"))
    development = (prior.get("searched_development") or {}).get("arena") or {}
    if prior.get("decision") != "searched_development_failed_keep_generation_2":
        raise ValueError("Frozen 097 result has the wrong controlling decision.")
    if (prior.get("offline_selection") or {}).get("selected") != "visit_argmax_aux":
        raise ValueError("Frozen 097 result selected the wrong candidate.")
    if development.get("candidate_wins") != 19 or development.get("candidate_losses") != 13:
        raise ValueError("Frozen 097 result does not contain the expected 19-13 arena.")
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["incumbent_checkpoint"]:
        raise ValueError("Frozen 097 result did not retain exact generation 2.")


def _jsonable_config(config: AdaptiveReplicationConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _finish(
    output: Path,
    result: dict[str, Any],
    started: float,
    decision: str,
) -> dict[str, Any]:
    result["status"] = "complete"
    result["decision"] = decision
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result


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
