"""Scaled rolling multi-generation AlphaZero sandbox for programme 099."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .network import load_checkpoint
from .search import SearchConfig
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play
from .training import TrainingConfig, train_successor


@dataclass(frozen=True, slots=True)
class ScaledLoopConfig:
    output_dir: Path
    incumbent_checkpoint: Path
    bootstrap_replay: Path
    result_098: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "d61ae82f801fb8167340cfece54aee1bd3231c3d931cd3f5b715ef004f717181"
EXPECTED_HASHES = {
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "bootstrap_replay": "c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47",
    "result_098": "79bbb9f1730d1167767205ec55001daf44af04b9ed1e58fefa3a533ac73d4772",
}
ITERATIONS = 3
SELFPLAY_GAMES = 256
SEARCH_SIMULATIONS = 256
SEARCH_C_PUCT = 2.25
DIAGNOSTIC_PAIRS = 8


def selfplay_corpus_validity(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "exact_game_count": summary["games"] == SELFPLAY_GAMES,
        "all_normal_terminations": summary["termination_reasons"]
        == {"completed": SELFPLAY_GAMES},
        "positive_position_count": summary["positions"] > 0,
        "exact_simulation_dose": summary["simulations"]
        == summary["positions"] * SEARCH_SIMULATIONS,
    }
    return {"valid": all(checks.values()), "checks": checks}


def rolling_replay_paths(
    bootstrap: Path, generated: list[Path]
) -> list[Path]:
    if not generated:
        raise ValueError("At least one fresh replay is required.")
    if len(generated) == 1:
        return [bootstrap, generated[0]]
    return generated[-2:]


def training_validity(training: dict[str, Any], *, expected_parent: int) -> dict[str, Any]:
    metric_payload = {
        "parent_train_metrics": training.get("parent_train_metrics"),
        "parent_validation_metrics": training.get("parent_validation_metrics"),
        "child_train_metrics": training.get("child_train_metrics"),
        "child_validation_metrics": training.get("child_validation_metrics"),
        "history": training.get("history"),
    }
    checks = {
        "exact_parent_generation": training.get("parent_generation") == expected_parent,
        "exact_child_generation": training.get("child_generation") == expected_parent + 1,
        "nonempty_validation": int(training.get("validation_rows", 0)) > 0,
        "at_least_one_epoch": int(training.get("epochs_completed", 0)) > 0,
        "all_metrics_finite": _all_numbers_finite(metric_payload),
    }
    return {"valid": all(checks.values()), "checks": checks}


def run_scaled_loop_099(config: ScaledLoopConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    result: dict[str, Any] = {
        "kind": "alphazero_scaled_rolling_loop_099",
        "status": "running",
        "strength_claim": False,
        "sandbox_lineage": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "generations": [],
        "active_iteration": 0,
        "active_parent_generation": 2,
        "active_stage": "initializing",
        "final_sandbox_checkpoint": None,
        "final_sandbox_checkpoint_sha256": None,
        "installed_checkpoint": str(config.incumbent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.incumbent_checkpoint),
        "external_035_arena": None,
        "decision": "running",
    }
    _write_progress(output, result, started)

    parent = config.incumbent_checkpoint
    generated_replays: list[Path] = []
    for iteration in range(ITERATIONS):
        parent_generation = 2 + iteration
        child_generation = parent_generation + 1
        result["active_iteration"] = iteration + 1
        result["active_parent_generation"] = parent_generation
        result["active_stage"] = "selfplay"
        _write_progress(output, result, started)

        replay_path = output / "selfplay" / f"generation_{parent_generation:03d}.jsonl"
        selfplay = generate_self_play(
            checkpoint_path=parent,
            device=config.device,
            config=SelfPlayConfig(
                generation=parent_generation,
                game_count=SELFPLAY_GAMES,
                seed_start=1_700_000 + iteration * 10_000,
                search=SearchConfig(
                    simulations=SEARCH_SIMULATIONS,
                    c_puct=SEARCH_C_PUCT,
                    mode="real",
                    root_noise_fraction=0.25,
                    seed=990_099 + iteration,
                ),
                max_turns=config.max_turns,
            ),
            replay_path=replay_path,
            progress_path=output
            / "selfplay"
            / f"generation_{parent_generation:03d}.progress.json",
            resume=True,
        )
        selfplay_summary = summarize_self_play(selfplay)
        corpus_validity = selfplay_corpus_validity(selfplay_summary)
        del selfplay
        generated_replays.append(replay_path)
        if not corpus_validity["valid"]:
            result["active_stage"] = "stopped_invalid_selfplay"
            result["failed_generation"] = parent_generation
            result["failed_selfplay"] = selfplay_summary
            result["failed_selfplay_validity"] = corpus_validity
            return _finish(
                output,
                result,
                started,
                "invalid_selfplay_stopped_keep_generation_2",
            )

        buffer_paths = rolling_replay_paths(config.bootstrap_replay, generated_replays)
        result["active_stage"] = "training"
        result["active_replay_window"] = [str(path) for path in buffer_paths]
        _write_progress(output, result, started)
        child = output / "checkpoints" / f"generation_{child_generation:03d}.pt"
        training = train_successor(
            parent_checkpoint=parent,
            replay_paths=buffer_paths,
            output_checkpoint=child,
            metrics_path=output
            / "training"
            / f"generation_{child_generation:03d}.json",
            device=config.device,
            config=_training_config(iteration),
        )
        validity = training_validity(training, expected_parent=parent_generation)
        _, identity = load_checkpoint(child, device=config.device)
        checkpoint_generation_valid = identity.generation == child_generation
        validity["checks"]["checkpoint_roundtrip_generation"] = checkpoint_generation_valid
        validity["valid"] = bool(validity["valid"] and checkpoint_generation_valid)
        if not validity["valid"]:
            result["active_stage"] = "stopped_invalid_training"
            result["failed_generation"] = child_generation
            result["failed_training"] = training
            result["failed_training_validity"] = validity
            return _finish(
                output,
                result,
                started,
                "invalid_training_stopped_keep_generation_2",
            )

        result["active_stage"] = "diagnostics"
        _write_progress(output, result, started)
        diagnostics = _diagnostics(
            output=output,
            config=config,
            iteration=iteration,
            parent=parent,
            child=child,
        )
        generation_result = {
            "iteration": iteration + 1,
            "parent_generation": parent_generation,
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": _sha256(parent),
            "selfplay": selfplay_summary,
            "selfplay_validity": corpus_validity,
            "replay_path": str(replay_path),
            "replay_sha256": _sha256(replay_path),
            "replay_window": [str(path) for path in buffer_paths],
            "child_generation": child_generation,
            "child_checkpoint": str(child),
            "child_checkpoint_sha256": _sha256(child),
            "training": training,
            "training_validity": validity,
            "diagnostics": diagnostics,
        }
        result["generations"].append(generation_result)
        parent = child
        result["active_stage"] = "generation_complete"
        _write_progress(output, result, started)

    result["active_stage"] = "complete"
    result["final_sandbox_checkpoint"] = str(parent)
    result["final_sandbox_checkpoint_sha256"] = _sha256(parent)
    result["trajectory"] = _trajectory(result["generations"])
    return _finish(
        output,
        result,
        started,
        "scaled_three_generation_sandbox_complete_review_trajectory",
    )


def _training_config(iteration: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=12,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=990_099 + iteration,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        bucket_batches=True,
        policy_divergence_weight=0.0,
        visit_argmax_aux_weight=0.0,
    )


def _evaluation_search() -> SearchConfig:
    return SearchConfig(
        simulations=SEARCH_SIMULATIONS,
        c_puct=SEARCH_C_PUCT,
        mode="real",
        root_noise_fraction=0.0,
        seed=990_099,
    )


def _diagnostics(
    *,
    output: Path,
    config: ScaledLoopConfig,
    iteration: int,
    parent: Path,
    child: Path,
) -> list[dict[str, Any]]:
    label = f"generation_{3 + iteration:03d}"
    search = _evaluation_search()
    specs = (
        (
            "raw_child_vs_raw_parent",
            alpha_zero_raw_factory(child, config.device),
            alpha_zero_raw_factory(parent, config.device),
            1_780_000,
        ),
        (
            "searched_child_vs_searched_parent",
            alpha_zero_search_factory(child, config.device, search),
            alpha_zero_search_factory(parent, config.device, search),
            1_780_100,
        ),
        (
            "child_search_vs_child_raw",
            alpha_zero_search_factory(child, config.device, search),
            alpha_zero_raw_factory(child, config.device),
            1_780_200,
        ),
    )
    results: list[dict[str, Any]] = []
    for name, candidate, baseline, seed_start in specs:
        arena = run_paired_arena(
            name=f"{label}_{name}",
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=seed_start,
            paired_seeds=DIAGNOSTIC_PAIRS,
            max_turns=config.max_turns,
            progress_path=output / "diagnostics" / label / f"{name}.json",
            resume=True,
        )
        results.append(asdict(arena))
    return results


def _trajectory(generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generation in generations:
        rows.append(
            {
                "child_generation": generation["child_generation"],
                "diagnostics": [
                    {
                        "name": arena["name"],
                        "score_share": arena["score_share"],
                        "mean_margin": arena["mean_margin"],
                    }
                    for arena in generation["diagnostics"]
                ],
                "selfplay_visit_argmax_changed_raw_fraction": generation["selfplay"][
                    "visit_argmax_changed_raw_fraction"
                ],
                "child_validation_policy_loss": generation["training"][
                    "child_validation_metrics"
                ]["policy_loss"],
                "child_validation_value_loss": generation["training"][
                    "child_validation_metrics"
                ]["value_loss"],
            }
        )
    return rows


def _verify_inputs(config: ScaledLoopConfig) -> None:
    paths = {
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "bootstrap_replay": config.bootstrap_replay,
        "result_098": config.result_098,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 099 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_098.read_text(encoding="utf-8"))
    arena = ((prior.get("searched_replication") or {}).get("arena") or {})
    if prior.get("decision") != "replication_overall_failed_keep_generation_2":
        raise ValueError("Frozen 098 result has the wrong controlling decision.")
    if (
        arena.get("candidate_wins") != 107
        or arena.get("draws") != 2
        or arena.get("candidate_losses") != 91
    ):
        raise ValueError("Frozen 098 result does not contain the exact 107-2-91 panel.")
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["incumbent_checkpoint"]:
        raise ValueError("Frozen 098 result did not retain exact generation 2.")


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _jsonable_config(config: ScaledLoopConfig) -> dict[str, Any]:
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
