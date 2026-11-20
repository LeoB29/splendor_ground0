"""Controlled successor learning from the confirmed 094 operator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .search import SearchConfig
from .search_primary import (
    _arena_dict,
    _seat_score_shares,
    paired_score_share_bootstrap,
)
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play
from .training import TrainingConfig, train_successor


@dataclass(frozen=True, slots=True)
class ConfirmedOperatorSuccessorConfig:
    output_dir: Path
    incumbent_checkpoint: Path
    replay_generation_2: Path
    result_094: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "9ac6d35f5f8e53bcf7efdfd2abb33527fa461480d4438b818d9fb39e09f884bc"
EXPECTED_HASHES = {
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "replay_generation_2": "33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b",
    "result_094": "fd386489640d0b6bdff806c47effedd7f388d1ae678219d97c91957ec9ad2e79",
}
SEARCH_SIMULATIONS = 256
SEARCH_C_PUCT = 2.25


def selfplay_corpus_eligibility(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "exact_game_count": summary["games"] == 64,
        "all_normal_terminations": summary["termination_reasons"]
        == {"completed": 64},
        "positive_position_count": summary["positions"] > 0,
        "exact_simulation_dose": summary["simulations"]
        == summary["positions"] * SEARCH_SIMULATIONS,
    }
    return {"eligible": all(checks.values()), "checks": checks}


def offline_successor_eligibility(training: dict[str, Any]) -> dict[str, Any]:
    delta = training["validation_loss_delta_child_minus_parent"]
    parent = training["parent_validation_metrics"]
    child = training["child_validation_metrics"]
    phase_checks = {
        name: row["delta_child_minus_parent"]["value_loss"] <= 0.025
        for name, row in training["validation_slices"].items()
        if name.startswith("phase_")
    }
    checks = {
        "objective_improved": delta["objective_loss"] < 0.0,
        "policy_not_worse": delta["policy_loss"] <= 0.0,
        "value_not_worse": delta["value_loss"] <= 0.0,
        "top_agreement_preserved": child["policy_top_target_agreement"]
        >= parent["policy_top_target_agreement"] - 0.01,
        "phase_value_degradation_bounded": bool(phase_checks)
        and all(phase_checks.values()),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "phase_value_checks": phase_checks,
    }


def searched_development_acceptance(arena: dict[str, Any]) -> dict[str, Any]:
    seats = _seat_score_shares(arena)
    checks = {
        "score_share": arena["score_share"] >= 0.5625,
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


def searched_confirmation_acceptance(
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


def run_confirmed_operator_successor_095(
    config: ConfirmedOperatorSuccessorConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    result_094 = json.loads(config.result_094.read_text(encoding="utf-8"))
    _verify_094_decision(result_094)
    result: dict[str, Any] = {
        "kind": "alphazero_confirmed_operator_successor_095",
        "status": "running",
        "strength_claim": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "confirmed_search": asdict(_evaluation_search()),
        "selfplay": None,
        "selfplay_eligibility": None,
        "fresh_replay_path": None,
        "fresh_replay_sha256": None,
        "child_checkpoint": None,
        "child_checkpoint_sha256": None,
        "training": None,
        "offline_eligibility": None,
        "searched_development": None,
        "searched_confirmation": None,
        "raw_confirmation_diagnostic": None,
        "installed_checkpoint": str(config.incumbent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.incumbent_checkpoint),
        "external_035_arena": None,
        "decision": "running",
    }
    _write_progress(output, result, started)

    replay_path = output / "selfplay" / "generation_002_high_dose.jsonl"
    selfplay = generate_self_play(
        checkpoint_path=config.incumbent_checkpoint,
        device=config.device,
        config=SelfPlayConfig(
            generation=2,
            game_count=64,
            seed_start=970_000,
            search=replace(
                _evaluation_search(),
                root_noise_fraction=0.25,
                seed=950_095,
            ),
            max_turns=config.max_turns,
        ),
        replay_path=replay_path,
        progress_path=output / "selfplay" / "progress.json",
        resume=True,
    )
    selfplay_summary = summarize_self_play(selfplay)
    corpus_eligibility = selfplay_corpus_eligibility(selfplay_summary)
    result["selfplay"] = selfplay_summary
    result["selfplay_eligibility"] = corpus_eligibility
    result["fresh_replay_path"] = str(replay_path)
    result["fresh_replay_sha256"] = _sha256(replay_path)
    _write_progress(output, result, started)
    if not corpus_eligibility["eligible"]:
        return _finish(
            output,
            result,
            started,
            decision="selfplay_corpus_ineligible_keep_generation_2",
        )

    child = output / "candidate" / "generation_003.pt"
    training = train_successor(
        parent_checkpoint=config.incumbent_checkpoint,
        replay_paths=[config.replay_generation_2, replay_path],
        output_checkpoint=child,
        metrics_path=output / "candidate" / "training.json",
        device=config.device,
        config=_training_config(),
    )
    offline = offline_successor_eligibility(training)
    result["child_checkpoint"] = str(child)
    result["child_checkpoint_sha256"] = _sha256(child)
    result["training"] = training
    result["offline_eligibility"] = offline
    _write_progress(output, result, started)
    if not offline["eligible"]:
        return _finish(
            output,
            result,
            started,
            decision="offline_successor_ineligible_keep_generation_2",
        )

    evaluation_search = _evaluation_search()
    development = _arena_dict(
        run_paired_arena(
            name="generation_3_search_vs_generation_2_search_development",
            candidate_factory=alpha_zero_search_factory(
                child, config.device, evaluation_search
            ),
            baseline_factory=alpha_zero_search_factory(
                config.incumbent_checkpoint, config.device, evaluation_search
            ),
            seed_start=1_660_000,
            paired_seeds=16,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_development.json",
            resume=True,
        )
    )
    development_acceptance = searched_development_acceptance(development)
    result["searched_development"] = {
        "arena": development,
        "acceptance": development_acceptance,
    }
    _write_progress(output, result, started)
    if not development_acceptance["accepted"]:
        return _finish(
            output,
            result,
            started,
            decision="searched_development_failed_keep_generation_2",
        )

    confirmation = _arena_dict(
        run_paired_arena(
            name="generation_3_search_vs_generation_2_search_confirmation",
            candidate_factory=alpha_zero_search_factory(
                child, config.device, evaluation_search
            ),
            baseline_factory=alpha_zero_search_factory(
                config.incumbent_checkpoint, config.device, evaluation_search
            ),
            seed_start=1_661_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_confirmation.json",
            resume=True,
        )
    )
    bootstrap = paired_score_share_bootstrap(
        confirmation, samples=50_000, seed=950_095
    )
    confirmation_acceptance = searched_confirmation_acceptance(
        confirmation, bootstrap
    )
    result["searched_confirmation"] = {
        "arena": confirmation,
        "paired_bootstrap": bootstrap,
        "acceptance": confirmation_acceptance,
    }
    _write_progress(output, result, started)

    raw = _arena_dict(
        run_paired_arena(
            name="generation_3_raw_vs_generation_2_raw_diagnostic",
            candidate_factory=alpha_zero_raw_factory(child, config.device),
            baseline_factory=alpha_zero_raw_factory(
                config.incumbent_checkpoint, config.device
            ),
            seed_start=1_661_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "raw_confirmation_diagnostic.json",
            resume=True,
        )
    )
    result["raw_confirmation_diagnostic"] = raw
    if confirmation_acceptance["accepted"]:
        result["installed_checkpoint"] = str(child)
        result["installed_checkpoint_sha256"] = _sha256(child)
        decision = "generation_3_confirmed_and_installed"
    else:
        decision = "searched_confirmation_failed_keep_generation_2"
    return _finish(output, result, started, decision=decision)


def _evaluation_search() -> SearchConfig:
    return SearchConfig(
        simulations=SEARCH_SIMULATIONS,
        c_puct=SEARCH_C_PUCT,
        mode="real",
        root_noise_fraction=0.0,
        seed=950_095,
    )


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=6,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=950_095,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=2,
        early_stopping_min_delta=0.002,
        bucket_batches=True,
    )


def _verify_094_decision(result: dict[str, Any]) -> None:
    selection = result.get("search_selection") or {}
    spec = selection.get("spec") or {}
    if result.get("decision") != "search_operator_confirmed":
        raise ValueError("Hash-locked 094 result did not confirm its operator.")
    if int(spec.get("simulations", -1)) != SEARCH_SIMULATIONS or float(
        spec.get("c_puct", -1.0)
    ) != SEARCH_C_PUCT:
        raise ValueError("Hash-locked 094 result has the wrong search selection.")


def _verify_inputs(config: ConfirmedOperatorSuccessorConfig) -> None:
    paths = {
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "replay_generation_2": config.replay_generation_2,
        "result_094": config.result_094,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 095 input hash mismatch for {name}: {actual}")


def _jsonable_config(config: ConfirmedOperatorSuccessorConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "incumbent_checkpoint": str(config.incumbent_checkpoint),
        "replay_generation_2": str(config.replay_generation_2),
        "result_094": str(config.result_094),
    }


def _finish(
    output: Path,
    result: dict[str, Any],
    started: float,
    *,
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
