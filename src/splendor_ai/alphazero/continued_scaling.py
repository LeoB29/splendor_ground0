"""Third unchanged AlphaZero scaling block for programme 104."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

from .arena import (
    alpha_zero_raw_factory,
    alpha_zero_search_factory,
    policy_035_factory,
)
from .clean_continuation import sanitize_replay
from .endpoint_evaluation import endpoint_decision
from .scaled_continuation import (
    _load_generation_record,
    _run_or_load_training,
    _run_panel,
    _sha256,
    _write_json,
    clean_replay_summary_for_path,
    rolling_window,
)
from .scaled_loop import _diagnostics
from .search import SearchConfig
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play


@dataclass(frozen=True, slots=True)
class ContinuedScalingConfig:
    output_dir: Path
    installed_checkpoint: Path
    generation_8_checkpoint: Path
    replay_generation_7: Path
    result_103: Path
    checkpoint_035: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "c144e663507bff22dfd524820a7775c3c14a41fb86eb4419e649153f94d05a72"
EXPECTED_HASHES = {
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_8_checkpoint": "beea26beddf91c7b0ab8f531ba6bc2bfa104d4e635e12af8002d04f7fb9c9281",
    "replay_generation_7": "06ee156543da64de3126eed00cb363ed5f2d4c0e49e093550024fffc2ace5a1c",
    "result_103": "0152c113dee56bd156483b0ab6d3a55d48e35b70cd3bcc3f5f9c0b5c2c0f7f96",
    "checkpoint_035": "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd",
}
SELFPLAY_GAMES = 256


def selfplay_seed_start(generation: int) -> int:
    if generation not in (8, 9, 10):
        raise ValueError("104 self-play generation must be 8, 9, or 10.")
    return 1_880_000 + (generation - 8) * 10_000


def final_decision_104(
    *, internal_primary_passed: bool, benchmark_passed: bool | None
) -> str:
    if not internal_primary_passed:
        return "generation_11_endpoint_not_compounded_035_skipped"
    if benchmark_passed:
        return "generation_11_compounded_and_035_practical_gate_passed_review_no_install"
    return "generation_11_compounded_035_benchmark_complete_no_install"


def run_continued_scaling_104(config: ContinuedScalingConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    if (output / "result.json").exists():
        sealed = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 104 result is not a valid completed result.")

    result: dict[str, Any] = {
        "kind": "alphazero_continued_scaling_104",
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

    parent = config.generation_8_checkpoint
    previous_clean = config.replay_generation_7
    for parent_generation in (8, 9, 10):
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
            sanitation = _run_or_load_clean_selfplay_104(
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
                config=config,  # type: ignore[arg-type]
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
    search = _evaluation_search_104()
    internal_specs = (
        (
            "searched_generation_11_vs_searched_generation_8",
            alpha_zero_search_factory(parent, config.device, search),
            alpha_zero_search_factory(config.generation_8_checkpoint, config.device, search),
        ),
        (
            "raw_generation_11_vs_raw_generation_8",
            alpha_zero_raw_factory(parent, config.device),
            alpha_zero_raw_factory(config.generation_8_checkpoint, config.device),
        ),
        (
            "generation_11_search_vs_generation_11_raw",
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
            seed_start=1_910_000,
            bootstrap_seed=1_040_300 + panel_index,
            max_turns=config.max_turns,
        )
        _write_progress(output, result, started)

    primary_name = "searched_generation_11_vs_searched_generation_8"
    raw_name = "raw_generation_11_vs_raw_generation_8"
    health_name = "generation_11_search_vs_generation_11_raw"
    primary = result["internal_endpoint"][primary_name]["assessment"]
    raw = result["internal_endpoint"][raw_name]["assessment"]
    health = result["internal_endpoint"][health_name]["assessment"]
    result["internal_endpoint_decision"] = endpoint_decision(primary, raw, health)

    benchmark_passed: bool | None = None
    if primary["passed"]:
        result["benchmark_035"] = {}
        benchmark_specs = (
            (
                "searched_generation_11_vs_raw_035",
                alpha_zero_search_factory(parent, config.device, search),
                policy_035_factory(config.checkpoint_035, config.device),
            ),
            (
                "raw_generation_11_vs_raw_035",
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
                seed_start=1_920_000,
                bootstrap_seed=1_040_400 + panel_index,
                max_turns=config.max_turns,
            )
            _write_progress(output, result, started)
        benchmark_passed = result["benchmark_035"][
            "searched_generation_11_vs_raw_035"
        ]["assessment"]["passed"]
    else:
        result["benchmark_035"] = {
            "status": "not_opened",
            "reason": "searched_generation_11_vs_searched_generation_8_practical_gate_failed",
        }

    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = final_decision_104(
        internal_primary_passed=primary["passed"],
        benchmark_passed=benchmark_passed,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result


def _run_or_load_clean_selfplay_104(
    *, config: ContinuedScalingConfig, parent: Path, generation: int
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
        summary = clean_replay_summary_for_path(
            Path(payload["sanitized_path"]), generation
        )
        if not summary["valid"] or summary != payload["summary"]:
            raise ValueError("Sanitation manifest summary mismatch.")
        return payload

    offset = generation - 8
    seed_start = selfplay_seed_start(generation)
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
                seed=990_105 + offset,
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
        search_seed=990_105 + offset,
        config=config,  # type: ignore[arg-type]
    )
    sanitation["checkpoint_sha256"] = _sha256(parent)
    sanitation["raw_summary"] = raw_summary
    _write_json(manifest, sanitation)
    return sanitation


def _evaluation_search_104() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_040_104,
    )


def _verify_inputs(config: ContinuedScalingConfig) -> None:
    paths = {
        "installed_checkpoint": config.installed_checkpoint,
        "generation_8_checkpoint": config.generation_8_checkpoint,
        "replay_generation_7": config.replay_generation_7,
        "result_103": config.result_103,
        "checkpoint_035": config.checkpoint_035,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 104 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_103.read_text(encoding="utf-8"))
    if prior.get("status") != "complete":
        raise ValueError("Frozen 103 result is not complete.")
    if prior.get("decision") != "generation_8_compounded_035_benchmark_complete_no_install":
        raise ValueError("Frozen 103 result has the wrong decision.")
    if prior.get("internal_endpoint_decision") != "endpoint_compounding_with_incomplete_raw_transfer":
        raise ValueError("Frozen 103 result has the wrong internal endpoint decision.")
    if prior.get("final_sandbox_checkpoint_sha256") != EXPECTED_HASHES["generation_8_checkpoint"]:
        raise ValueError("Frozen 103 result does not end at exact generation 8.")
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]:
        raise ValueError("Frozen 103 result did not retain exact generation 2.")
    primary = prior["internal_endpoint"]["searched_generation_8_vs_searched_generation_5"]
    if not primary["assessment"]["passed"] or primary["arena"]["candidate_wins"] != 118:
        raise ValueError("Frozen 103 result lacks the exact successful primary endpoint.")


def _jsonable_config(config: ContinuedScalingConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )
