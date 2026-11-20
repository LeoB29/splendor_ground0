"""Gated corrective loop for AlphaZero engineering programme 091."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .arena import (
    ArenaResult,
    alpha_zero_raw_factory,
    alpha_zero_search_factory,
    policy_035_factory,
    run_paired_arena,
)
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play
from .search import SearchConfig
from .training import TrainingConfig, train_successor


@dataclass(frozen=True, slots=True)
class CorrectiveConfig:
    output_dir: Path
    incumbent_checkpoint: Path
    replay_generation_1: Path
    replay_generation_2: Path
    checkpoint_035: Path | None = None
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    learning_rate: float
    value_loss_weight: float


@dataclass(frozen=True, slots=True)
class SearchSpec:
    name: str
    simulations: int
    c_puct: float


CANDIDATES = (
    CandidateSpec("balanced_half", 1e-4, 0.5),
    CandidateSpec("value_quarter", 1e-4, 0.25),
)
SEARCH_SPECS = (
    SearchSpec("baseline_64", 64, 1.5),
    SearchSpec("focused_96", 96, 1.0),
    SearchSpec("exploratory_96", 96, 2.0),
)
EXPECTED_INPUT_HASHES = {
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "replay_generation_1": "500b9e6a68e8e2332767c03eaecede595a63ee30b38b52a5efd001f6619d925a",
    "replay_generation_2": "33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b",
}


def candidate_offline_eligibility(training: dict[str, Any]) -> dict[str, Any]:
    delta = training["validation_loss_delta_child_minus_parent"]
    parent = training["parent_validation_metrics"]
    child = training["child_validation_metrics"]
    checks = {
        "objective_improved": delta["objective_loss"] < 0.0,
        "policy_not_worse": delta["policy_loss"] <= 0.0,
        "value_degradation_bounded": delta["value_loss"] <= 0.025,
        "top_agreement_preserved": (
            child["policy_top_target_agreement"]
            >= parent["policy_top_target_agreement"] - 0.01
        ),
    }
    return {"eligible": all(checks.values()), "checks": checks}


def candidate_arena_acceptance(arenas: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, dict[str, bool]] = {}
    for arena in arenas:
        checks[arena["name"]] = {
            "score_share": arena["score_share"] >= 0.5,
            "mean_margin": arena["mean_margin"] >= 0.0,
            "normal_terminations": arena["termination_reasons"]
            == {"completed": arena["games"]},
        }
    return {
        "accepted": len(checks) == 2
        and all(all(values.values()) for values in checks.values()),
        "checks": checks,
    }


def search_discovery_eligibility(arena: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "score_share": arena["score_share"] >= 0.625,
        "positive_margin": arena["mean_margin"] > 0.0,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {"eligible": all(checks.values()), "checks": checks}


def search_confirmation_acceptance(arena: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "score_share": arena["score_share"] >= 0.5625,
        "positive_margin": arena["mean_margin"] > 0.0,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {"accepted": all(checks.values()), "checks": checks}


def run_corrective_091(config: CorrectiveConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    result: dict[str, Any] = {
        "kind": "alphazero_corrective_091",
        "status": "running",
        "strength_claim": False,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_INPUT_HASHES),
        "candidate_screen": [],
        "candidate_selection": None,
        "candidate_arenas": [],
        "candidate_acceptance": None,
        "search_discovery": [],
        "search_selection": None,
        "search_confirmation": None,
        "conditional_continuation": None,
        "external_035_arena": None,
        "installed_checkpoint": str(config.incumbent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.incumbent_checkpoint),
        "decision": "running",
    }
    _write_progress(output, result, started)

    for order, spec in enumerate(CANDIDATES):
        candidate_dir = output / "candidates" / spec.name
        candidate_checkpoint = candidate_dir / "generation_003.pt"
        training = train_successor(
            parent_checkpoint=config.incumbent_checkpoint,
            replay_paths=[config.replay_generation_1, config.replay_generation_2],
            output_checkpoint=candidate_checkpoint,
            metrics_path=candidate_dir / "training.json",
            device=config.device,
            config=_training_config(spec),
        )
        eligibility = candidate_offline_eligibility(training)
        result["candidate_screen"].append(
            {
                "order": order,
                "spec": asdict(spec),
                "checkpoint": str(candidate_checkpoint),
                "checkpoint_sha256": _sha256(candidate_checkpoint),
                "training": training,
                "offline_eligibility": eligibility,
            }
        )
        _write_progress(output, result, started)

    eligible = [
        row
        for row in result["candidate_screen"]
        if row["offline_eligibility"]["eligible"]
    ]
    if not eligible:
        return _finish(
            output,
            result,
            started,
            decision="rollback_no_offline_eligible_candidate",
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["training"]["child_validation_metrics"]["objective_loss"],
            row["training"]["child_validation_metrics"]["policy_loss"],
            row["order"],
        ),
    )
    selected_checkpoint = Path(selected["checkpoint"])
    result["candidate_selection"] = {
        "name": selected["spec"]["name"],
        "checkpoint": str(selected_checkpoint),
        "selection_basis": "offline_validation_only",
    }
    _write_progress(output, result, started)

    baseline_search = SearchConfig(
        simulations=64,
        c_puct=1.5,
        mode="real",
        root_noise_fraction=0.0,
        seed=910091,
    )
    candidate_arenas = [
        _arena_dict(
            run_paired_arena(
                name="child_raw_vs_incumbent_raw",
                candidate_factory=alpha_zero_raw_factory(
                    selected_checkpoint, config.device
                ),
                baseline_factory=alpha_zero_raw_factory(
                    config.incumbent_checkpoint, config.device
                ),
                seed_start=1_610_000,
                paired_seeds=8,
                max_turns=config.max_turns,
            )
        ),
        _arena_dict(
            run_paired_arena(
                name="child_search_vs_incumbent_search",
                candidate_factory=alpha_zero_search_factory(
                    selected_checkpoint, config.device, baseline_search
                ),
                baseline_factory=alpha_zero_search_factory(
                    config.incumbent_checkpoint, config.device, baseline_search
                ),
                seed_start=1_610_100,
                paired_seeds=8,
                max_turns=config.max_turns,
            )
        ),
    ]
    result["candidate_arenas"] = candidate_arenas
    acceptance = candidate_arena_acceptance(candidate_arenas)
    result["candidate_acceptance"] = acceptance
    _write_progress(output, result, started)
    if not acceptance["accepted"]:
        return _finish(
            output,
            result,
            started,
            decision="rollback_candidate_failed_parent_arenas",
        )

    result["installed_checkpoint"] = str(selected_checkpoint)
    result["installed_checkpoint_sha256"] = _sha256(selected_checkpoint)
    for order, spec in enumerate(SEARCH_SPECS):
        search_config = _search_config(spec)
        arena = _arena_dict(
            run_paired_arena(
                name=f"{spec.name}_search_vs_raw_discovery",
                candidate_factory=alpha_zero_search_factory(
                    selected_checkpoint, config.device, search_config
                ),
                baseline_factory=alpha_zero_raw_factory(
                    selected_checkpoint, config.device
                ),
                seed_start=1_611_000 + order * 100,
                paired_seeds=4,
                max_turns=config.max_turns,
            )
        )
        result["search_discovery"].append(
            {
                "order": order,
                "spec": asdict(spec),
                "arena": arena,
                "eligibility": search_discovery_eligibility(arena),
            }
        )
        _write_progress(output, result, started)
    eligible_searches = [
        row
        for row in result["search_discovery"]
        if row["eligibility"]["eligible"]
    ]
    if not eligible_searches:
        return _finish(
            output,
            result,
            started,
            decision="accepted_child_but_no_search_improvement_candidate",
        )
    selected_search = min(
        eligible_searches,
        key=lambda row: (
            -row["arena"]["score_share"],
            -row["arena"]["mean_margin"],
            row["spec"]["simulations"],
            row["order"],
        ),
    )
    selected_search_spec = SearchSpec(**selected_search["spec"])
    result["search_selection"] = {
        "spec": asdict(selected_search_spec),
        "selection_basis": "frozen_discovery_order",
    }
    confirmation_arena = _arena_dict(
        run_paired_arena(
            name=f"{selected_search_spec.name}_search_vs_raw_confirmation",
            candidate_factory=alpha_zero_search_factory(
                selected_checkpoint,
                config.device,
                _search_config(selected_search_spec),
            ),
            baseline_factory=alpha_zero_raw_factory(
                selected_checkpoint, config.device
            ),
            seed_start=1_612_000,
            paired_seeds=8,
            max_turns=config.max_turns,
        )
    )
    result["search_confirmation"] = {
        "arena": confirmation_arena,
        "acceptance": search_confirmation_acceptance(confirmation_arena),
    }
    _write_progress(output, result, started)
    if not result["search_confirmation"]["acceptance"]["accepted"]:
        return _finish(
            output,
            result,
            started,
            decision="accepted_child_search_improvement_not_confirmed",
        )

    continuation = _conditional_continuation(
        config=config,
        output=output,
        parent_checkpoint=selected_checkpoint,
        selected_candidate=CandidateSpec(**selected["spec"]),
        selected_search=selected_search_spec,
    )
    result["conditional_continuation"] = continuation
    if continuation["acceptance"]["accepted"]:
        result["installed_checkpoint"] = continuation["child_checkpoint"]
        result["installed_checkpoint_sha256"] = continuation[
            "child_checkpoint_sha256"
        ]
        decision = "successor_accepted"
    else:
        decision = "successor_rejected_rollback_to_first_accepted_child"
    if config.checkpoint_035 is not None:
        final_checkpoint = Path(result["installed_checkpoint"])
        result["external_035_arena"] = _arena_dict(
            run_paired_arena(
                name="installed_raw_vs_035_external",
                candidate_factory=alpha_zero_raw_factory(
                    final_checkpoint, config.device
                ),
                baseline_factory=policy_035_factory(
                    config.checkpoint_035, config.device
                ),
                seed_start=1_614_000,
                paired_seeds=8,
                max_turns=config.max_turns,
            )
        )
    return _finish(output, result, started, decision=decision)


def _conditional_continuation(
    *,
    config: CorrectiveConfig,
    output: Path,
    parent_checkpoint: Path,
    selected_candidate: CandidateSpec,
    selected_search: SearchSpec,
) -> dict[str, Any]:
    continuation_dir = output / "conditional_continuation"
    replay_path = continuation_dir / "generation_003_selfplay.jsonl"
    selfplay = generate_self_play(
        checkpoint_path=parent_checkpoint,
        device=config.device,
        config=SelfPlayConfig(
            generation=3,
            game_count=32,
            seed_start=930_000,
            search=replace(
                _search_config(selected_search),
                root_noise_fraction=0.25,
                seed=930_091,
            ),
            max_turns=config.max_turns,
        ),
        replay_path=replay_path,
        progress_path=continuation_dir / "generation_003_selfplay.progress.json",
    )
    child = continuation_dir / "generation_004.pt"
    training = train_successor(
        parent_checkpoint=parent_checkpoint,
        replay_paths=[config.replay_generation_2, replay_path],
        output_checkpoint=child,
        metrics_path=continuation_dir / "generation_004_training.json",
        device=config.device,
        config=_training_config(selected_candidate),
    )
    offline = candidate_offline_eligibility(training)
    arenas: list[dict[str, Any]] = []
    if offline["eligible"]:
        search_config = _search_config(selected_search)
        arenas = [
            _arena_dict(
                run_paired_arena(
                    name="successor_raw_vs_parent_raw",
                    candidate_factory=alpha_zero_raw_factory(child, config.device),
                    baseline_factory=alpha_zero_raw_factory(
                        parent_checkpoint, config.device
                    ),
                    seed_start=1_613_000,
                    paired_seeds=8,
                    max_turns=config.max_turns,
                )
            ),
            _arena_dict(
                run_paired_arena(
                    name="successor_search_vs_parent_search",
                    candidate_factory=alpha_zero_search_factory(
                        child, config.device, search_config
                    ),
                    baseline_factory=alpha_zero_search_factory(
                        parent_checkpoint, config.device, search_config
                    ),
                    seed_start=1_613_100,
                    paired_seeds=8,
                    max_turns=config.max_turns,
                )
            ),
        ]
        acceptance = candidate_arena_acceptance(arenas)
    else:
        acceptance = {
            "accepted": False,
            "checks": {"offline_eligibility": offline["checks"]},
        }
    return {
        "selfplay": summarize_self_play(selfplay),
        "replay_path": str(replay_path),
        "replay_sha256": _sha256(replay_path),
        "child_checkpoint": str(child),
        "child_checkpoint_sha256": _sha256(child),
        "training": training,
        "offline_eligibility": offline,
        "arenas": arenas,
        "acceptance": acceptance,
    }


def _training_config(spec: CandidateSpec) -> TrainingConfig:
    return TrainingConfig(
        epochs=6,
        batch_size=256,
        learning_rate=spec.learning_rate,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=910_091,
        policy_loss_weight=1.0,
        value_loss_weight=spec.value_loss_weight,
        early_stopping_patience=2,
        early_stopping_min_delta=0.002,
        bucket_batches=True,
    )


def _search_config(spec: SearchSpec) -> SearchConfig:
    return SearchConfig(
        simulations=spec.simulations,
        c_puct=spec.c_puct,
        mode="real",
        root_noise_fraction=0.0,
        seed=910_091,
    )


def _arena_dict(result: ArenaResult) -> dict[str, Any]:
    return asdict(result)


def _verify_inputs(config: CorrectiveConfig) -> None:
    paths = {
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "replay_generation_1": config.replay_generation_1,
        "replay_generation_2": config.replay_generation_2,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_INPUT_HASHES[name]:
            raise ValueError(
                f"Frozen 091 input hash mismatch for {name}: {actual}"
            )


def _jsonable_config(config: CorrectiveConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "incumbent_checkpoint": str(config.incumbent_checkpoint),
        "replay_generation_1": str(config.replay_generation_1),
        "replay_generation_2": str(config.replay_generation_2),
        "checkpoint_035": (
            str(config.checkpoint_035) if config.checkpoint_035 is not None else None
        ),
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


def _write_progress(
    output: Path,
    result: dict[str, Any],
    started: float,
) -> None:
    payload = {**result, "elapsed_seconds": time.perf_counter() - started}
    _write_json(output / "progress.json", payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
