"""Hash-locked scaled AlphaZero continuation for programme 103."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .arena import (
    alpha_zero_raw_factory,
    alpha_zero_search_factory,
    policy_035_factory,
    run_paired_arena,
)
from .clean_continuation import clean_replay_summary, sanitize_replay
from .endpoint_evaluation import endpoint_decision, practical_assessment
from .network import load_checkpoint
from .scaled_loop import _diagnostics, _training_config, training_validity
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play
from .training import train_successor


@dataclass(frozen=True, slots=True)
class ScaledContinuationConfig:
    output_dir: Path
    installed_checkpoint: Path
    generation_5_checkpoint: Path
    replay_generation_4: Path
    result_101: Path
    result_102: Path
    checkpoint_035: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "6a40915f9999d5cfab7c7b29be4f532a2080334656f5176e0e7dfb3c0fc036d6"
EXPECTED_HASHES = {
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_5_checkpoint": "201e34128c9588722ce78edc85e8672675b1075dc0696651484718ddd7fa2444",
    "replay_generation_4": "7e15847d2cdf51ecd467a2bb6f38f13e1901bd7337971bbe8f4817ace1dec244",
    "result_101": "4e29b92f13232f31f082fd0a3c5eb00bf2bd1068a18956832b048cfb7bb04a05",
    "result_102": "82e515abb4f1edbf2333f0cae93b02388db96b58d10defb7b92195810ce6b3be",
    "checkpoint_035": "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd",
}
SELFPLAY_GAMES = 256
PAIRED_SEEDS = 100
BOOTSTRAP_SAMPLES = 50_000


def rolling_window(previous_clean: Path, current_clean: Path) -> list[Path]:
    if previous_clean == current_clean:
        raise ValueError("Rolling replay paths must be distinct.")
    return [previous_clean, current_clean]


def final_decision(*, internal_primary_passed: bool, benchmark_passed: bool | None) -> str:
    if not internal_primary_passed:
        return "generation_8_endpoint_not_compounded_035_skipped"
    if benchmark_passed:
        return "generation_8_compounded_and_035_practical_gate_passed_review_no_install"
    return "generation_8_compounded_035_benchmark_complete_no_install"


def run_scaled_continuation_103(config: ScaledContinuationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    if (output / "result.json").exists():
        sealed = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 103 result is not a valid completed result.")

    result: dict[str, Any] = {
        "kind": "alphazero_scaled_continuation_103",
        "status": "running",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "generations": [],
        "internal_endpoint": {},
        "benchmark_035": None,
        "active_stage": "initializing",
        "final_sandbox_checkpoint": None,
        "final_sandbox_checkpoint_sha256": None,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "promotion_authorized": False,
        "decision": "running",
    }
    _write_progress(output, result, started)

    parent = config.generation_5_checkpoint
    previous_clean = config.replay_generation_4
    for parent_generation in (5, 6, 7):
        child_generation = parent_generation + 1
        record_path = output / "generation_records" / f"generation_{child_generation:03d}.json"
        if record_path.exists():
            generation_record = _load_generation_record(
                record_path,
                expected_parent=parent_generation,
                parent=parent,
                device=config.device,
            )
        else:
            result["active_stage"] = f"selfplay_generation_{parent_generation}"
            _write_progress(output, result, started)
            sanitation = _run_or_load_clean_selfplay(
                config=config,
                parent=parent,
                generation=parent_generation,
            )
            current_clean = Path(sanitation["sanitized_path"])
            replay_window = rolling_window(previous_clean, current_clean)

            result["active_stage"] = f"train_generation_{child_generation}"
            result["active_replay_window"] = [str(path) for path in replay_window]
            _write_progress(output, result, started)
            training_stage = _run_or_load_training(
                config=config,
                parent=parent,
                parent_generation=parent_generation,
                replay_window=replay_window,
            )
            child = Path(training_stage["child_checkpoint"])

            result["active_stage"] = f"diagnostics_generation_{child_generation}"
            _write_progress(output, result, started)
            diagnostics = _diagnostics(
                output=output,
                config=config,  # type: ignore[arg-type]
                iteration=child_generation - 3,
                parent=parent,
                child=child,
            )
            generation_record = {
                "parent_generation": parent_generation,
                "parent_checkpoint": str(parent),
                "parent_checkpoint_sha256": _sha256(parent),
                "selfplay": sanitation["summary"],
                "sanitation": sanitation,
                "replay_path": str(current_clean),
                "replay_sha256": _sha256(current_clean),
                "replay_window": [str(path) for path in replay_window],
                "child_generation": child_generation,
                "child_checkpoint": str(child),
                "child_checkpoint_sha256": _sha256(child),
                "training": training_stage["training"],
                "training_validity": training_stage["training_validity"],
                "diagnostics": diagnostics,
            }
            _write_json(record_path, generation_record)

        result["generations"].append(generation_record)
        parent = Path(generation_record["child_checkpoint"])
        previous_clean = Path(generation_record["replay_path"])
        result["active_stage"] = f"generation_{child_generation}_complete"
        _write_progress(output, result, started)

    result["final_sandbox_checkpoint"] = str(parent)
    result["final_sandbox_checkpoint_sha256"] = _sha256(parent)
    search = _evaluation_search()
    internal_specs = (
        (
            "searched_generation_8_vs_searched_generation_5",
            alpha_zero_search_factory(parent, config.device, search),
            alpha_zero_search_factory(config.generation_5_checkpoint, config.device, search),
        ),
        (
            "raw_generation_8_vs_raw_generation_5",
            alpha_zero_raw_factory(parent, config.device),
            alpha_zero_raw_factory(config.generation_5_checkpoint, config.device),
        ),
        (
            "generation_8_search_vs_generation_8_raw",
            alpha_zero_search_factory(parent, config.device, search),
            alpha_zero_raw_factory(parent, config.device),
        ),
    )
    for panel_index, (name, candidate, baseline) in enumerate(internal_specs):
        result["active_stage"] = name
        _write_progress(output, result, started)
        result["internal_endpoint"][name] = _run_panel(
            output=output / "endpoint_arenas",
            name=name,
            candidate=candidate,
            baseline=baseline,
            seed_start=1_850_000,
            bootstrap_seed=1_030_300 + panel_index,
            max_turns=config.max_turns,
        )
        _write_progress(output, result, started)

    primary_name = "searched_generation_8_vs_searched_generation_5"
    raw_name = "raw_generation_8_vs_raw_generation_5"
    health_name = "generation_8_search_vs_generation_8_raw"
    primary = result["internal_endpoint"][primary_name]["assessment"]
    raw = result["internal_endpoint"][raw_name]["assessment"]
    health = result["internal_endpoint"][health_name]["assessment"]
    result["internal_endpoint_decision"] = endpoint_decision(primary, raw, health)

    benchmark_passed: bool | None = None
    if primary["passed"]:
        result["benchmark_035"] = {}
        benchmark_specs = (
            (
                "searched_generation_8_vs_raw_035",
                alpha_zero_search_factory(parent, config.device, search),
                policy_035_factory(config.checkpoint_035, config.device),
            ),
            (
                "raw_generation_8_vs_raw_035",
                alpha_zero_raw_factory(parent, config.device),
                policy_035_factory(config.checkpoint_035, config.device),
            ),
        )
        for panel_index, (name, candidate, baseline) in enumerate(benchmark_specs):
            result["active_stage"] = name
            _write_progress(output, result, started)
            result["benchmark_035"][name] = _run_panel(
                output=output / "benchmark_035_arenas",
                name=name,
                candidate=candidate,
                baseline=baseline,
                seed_start=1_860_000,
                bootstrap_seed=1_030_400 + panel_index,
                max_turns=config.max_turns,
            )
            _write_progress(output, result, started)
        benchmark_passed = result["benchmark_035"][
            "searched_generation_8_vs_raw_035"
        ]["assessment"]["passed"]
    else:
        result["benchmark_035"] = {
            "status": "not_opened",
            "reason": "searched_generation_8_vs_searched_generation_5_practical_gate_failed",
        }

    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = final_decision(
        internal_primary_passed=primary["passed"],
        benchmark_passed=benchmark_passed,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result


def _run_or_load_clean_selfplay(
    *, config: ScaledContinuationConfig, parent: Path, generation: int
) -> dict[str, Any]:
    output = config.output_dir
    manifest = output / "sanitation" / f"generation_{generation:03d}.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("generation") != generation:
            raise ValueError("Sanitation manifest has the wrong generation.")
        if payload.get("checkpoint_sha256") != _sha256(parent):
            raise ValueError("Sanitation manifest has the wrong checkpoint.")
        if payload.get("source_sha256") != _sha256(payload["source_path"]):
            raise ValueError("Sanitation source hash mismatch.")
        if payload.get("sanitized_sha256") != _sha256(payload["sanitized_path"]):
            raise ValueError("Sanitized replay hash mismatch.")
        rows_summary = clean_replay_summary_for_path(
            Path(payload["sanitized_path"]), generation
        )
        if not rows_summary["valid"] or rows_summary != payload["summary"]:
            raise ValueError("Sanitation manifest summary mismatch.")
        return payload

    offset = generation - 5
    seed_start = 1_820_000 + offset * 10_000
    raw_path = output / "selfplay" / f"generation_{generation:03d}_raw.jsonl"
    selfplay = generate_self_play(
        checkpoint_path=parent,
        device=config.device,
        config=SelfPlayConfig(
            generation=generation,
            game_count=SELFPLAY_GAMES,
            seed_start=seed_start,
            search=SearchConfig(
                simulations=256,
                c_puct=2.25,
                mode="real",
                root_noise_fraction=0.25,
                seed=990_102 + offset,
            ),
            max_turns=config.max_turns,
        ),
        replay_path=raw_path,
        progress_path=output / "selfplay" / f"generation_{generation:03d}_raw.progress.json",
        resume=True,
    )
    raw_summary = summarize_self_play(selfplay)
    del selfplay
    sanitized_path = output / "selfplay" / f"generation_{generation:03d}_sanitized.jsonl"
    sanitation = sanitize_replay(
        source_path=raw_path,
        output_path=sanitized_path,
        checkpoint=parent,
        generation=generation,
        original_seed_start=seed_start,
        search_seed=990_102 + offset,
        config=config,  # type: ignore[arg-type]
    )
    sanitation["checkpoint_sha256"] = _sha256(parent)
    sanitation["raw_summary"] = raw_summary
    _write_json(manifest, sanitation)
    return sanitation


def clean_replay_summary_for_path(path: Path, generation: int) -> dict[str, Any]:
    from .replay import read_replay_jsonl

    return clean_replay_summary(
        read_replay_jsonl(path), expected_generation=generation
    )


def _run_or_load_training(
    *,
    config: ScaledContinuationConfig,
    parent: Path,
    parent_generation: int,
    replay_window: list[Path],
) -> dict[str, Any]:
    child_generation = parent_generation + 1
    output = config.output_dir
    manifest = output / "training" / f"generation_{child_generation:03d}.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected_replays = [_sha256(path) for path in replay_window]
        if payload.get("parent_checkpoint_sha256") != _sha256(parent):
            raise ValueError("Training manifest has the wrong parent checkpoint.")
        if payload.get("replay_sha256") != expected_replays:
            raise ValueError("Training manifest has the wrong replay window.")
        if payload.get("child_checkpoint_sha256") != _sha256(payload["child_checkpoint"]):
            raise ValueError("Training child checkpoint hash mismatch.")
        if payload.get("metrics_sha256") != _sha256(payload["metrics_path"]):
            raise ValueError("Training metrics hash mismatch.")
        training = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
        validity = training_validity(training, expected_parent=parent_generation)
        _, identity = load_checkpoint(payload["child_checkpoint"], device=config.device)
        if not validity["valid"] or identity.generation != child_generation:
            raise ValueError("Training manifest failed lineage validation.")
        payload["training"] = training
        payload["training_validity"] = validity
        return payload

    child = output / "checkpoints" / f"generation_{child_generation:03d}.pt"
    metrics = output / "training" / f"generation_{child_generation:03d}.json"
    training = train_successor(
        parent_checkpoint=parent,
        replay_paths=replay_window,
        output_checkpoint=child,
        metrics_path=metrics,
        device=config.device,
        config=_training_config(parent_generation - 2),
    )
    validity = training_validity(training, expected_parent=parent_generation)
    _, identity = load_checkpoint(child, device=config.device)
    validity["checks"]["checkpoint_roundtrip_generation"] = identity.generation == child_generation
    validity["valid"] = bool(validity["valid"] and validity["checks"]["checkpoint_roundtrip_generation"])
    if not validity["valid"]:
        raise RuntimeError(f"Generation-{child_generation} training failed validity checks.")
    payload = {
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": _sha256(parent),
        "replay_paths": [str(path) for path in replay_window],
        "replay_sha256": [_sha256(path) for path in replay_window],
        "child_checkpoint": str(child),
        "child_checkpoint_sha256": _sha256(child),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
        "training": training,
        "training_validity": validity,
    }
    _write_json(manifest, {key: value for key, value in payload.items() if key not in {"training", "training_validity"}})
    return payload


def _load_generation_record(
    path: Path,
    *,
    expected_parent: int,
    parent: Path,
    device: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("parent_generation") != expected_parent:
        raise ValueError("Generation record has the wrong parent generation.")
    if payload.get("parent_checkpoint_sha256") != _sha256(parent):
        raise ValueError("Generation record has the wrong parent checkpoint.")
    if payload.get("replay_sha256") != _sha256(payload["replay_path"]):
        raise ValueError("Generation record replay hash mismatch.")
    if payload.get("child_checkpoint_sha256") != _sha256(payload["child_checkpoint"]):
        raise ValueError("Generation record child hash mismatch.")
    _, identity = load_checkpoint(payload["child_checkpoint"], device=device)
    if identity.generation != expected_parent + 1:
        raise ValueError("Generation record child lineage mismatch.")
    if not (payload.get("training_validity") or {}).get("valid"):
        raise ValueError("Generation record contains invalid training.")
    diagnostics = payload.get("diagnostics") or []
    if len(diagnostics) != 3 or any(arena.get("games") != 16 for arena in diagnostics):
        raise ValueError("Generation record lacks the complete health diagnostics.")
    return payload


def _run_panel(
    *, output: Path, name: str, candidate: Any, baseline: Any,
    seed_start: int, bootstrap_seed: int, max_turns: int,
) -> dict[str, Any]:
    arena = _arena_dict(
        run_paired_arena(
            name=name,
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=seed_start,
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
        seed=1_030_103,
    )


def _verify_inputs(config: ScaledContinuationConfig) -> None:
    paths = {
        "installed_checkpoint": config.installed_checkpoint,
        "generation_5_checkpoint": config.generation_5_checkpoint,
        "replay_generation_4": config.replay_generation_4,
        "result_101": config.result_101,
        "result_102": config.result_102,
        "checkpoint_035": config.checkpoint_035,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 103 input hash mismatch for {name}: {actual}")
    prior_101 = json.loads(config.result_101.read_text(encoding="utf-8"))
    prior_102 = json.loads(config.result_102.read_text(encoding="utf-8"))
    if prior_101.get("decision") != "clean_continuation_complete_review_trajectory":
        raise ValueError("Frozen 101 result has the wrong decision.")
    if prior_101.get("final_sandbox_checkpoint_sha256") != EXPECTED_HASHES["generation_5_checkpoint"]:
        raise ValueError("Frozen 101 result does not end at exact generation 5.")
    if prior_102.get("decision") != "endpoint_compounding_with_raw_transfer_and_healthy_search":
        raise ValueError("Frozen 102 result did not establish endpoint compounding.")
    if prior_102.get("endpoint_strength_claim") is not True:
        raise ValueError("Frozen 102 result lacks its endpoint strength claim.")
    if prior_102.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]:
        raise ValueError("Frozen 102 result did not retain exact generation 2.")


def _jsonable_config(config: ScaledContinuationConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})


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
