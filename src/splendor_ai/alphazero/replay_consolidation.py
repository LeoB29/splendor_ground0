"""Four-corpus replay consolidation experiment 107."""

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
    run_paired_arena,
)
from .endpoint_evaluation import practical_assessment
from .external_benchmark import (
    external_parity_assessment,
    independent_paired_difference_bootstrap,
)
from .head_attribution import paired_arena_difference_bootstrap
from .network import load_checkpoint
from .scaled_continuation import _sha256, _write_json
from .scaled_loop import training_validity
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .training import TrainingConfig, train_successor


@dataclass(frozen=True, slots=True)
class ReplayConsolidationConfig:
    output_dir: Path
    result_105: Path
    result_106: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    replay_7: Path
    replay_8: Path
    replay_9: Path
    replay_10: Path
    checkpoint_035: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "5e5d42ba283e8ebf177b5fd5d897753bc32ea387703ea84c96d07d5042d7e99f"
EXPECTED_HASHES = {
    "result_105": "71c2cb3564468059b7d82906d5e18f6fd8f4c144475c4e353dcc66156616a401",
    "result_106": "7e4bb4c7d53f13bcf29258f5856ab0b5d6fbe56844c6c1d9d742426e780c0ca1",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "replay_7": "06ee156543da64de3126eed00cb363ed5f2d4c0e49e093550024fffc2ace5a1c",
    "replay_8": "96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10",
    "replay_9": "1b13ef038ccc6b3f695d7c7860a2284272ae18c8e75d44028e4cbb6f69914f0c",
    "replay_10": "09279d9593acfd944a32571716a6527c0be40e41055b6fa1f5bc25c88f43a60c",
    "checkpoint_035": "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd",
}
INTERNAL_SEED_START = 1_950_000
EXTERNAL_SEED_START = 1_960_000
PAIRED_SEEDS = 100
BOOTSTRAP_SAMPLES = 50_000
NEWEST_OBJECTIVE_TOLERANCE = 0.005


def consolidation_training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=12,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=1_070_107,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        bucket_batches=True,
        policy_divergence_weight=0.0,
        visit_argmax_aux_weight=0.0,
    )


def retention_assessment(training: dict[str, Any]) -> dict[str, Any]:
    """Assess parent/child losses on the exact whole-game validation split."""
    slices = training.get("validation_slices") or {}
    required = {f"replay_generation_{generation}" for generation in range(7, 11)}
    if not required.issubset(slices):
        missing = sorted(required - set(slices))
        raise ValueError(f"Training metrics lack validation replay slices: {missing}")
    corpus: dict[str, Any] = {}
    for generation in range(7, 11):
        entry = slices[f"replay_generation_{generation}"]
        rows = int(entry.get("rows", 0))
        parent = float(entry["parent"]["objective_loss"])
        child = float(entry["child"]["objective_loss"])
        if rows <= 0:
            raise ValueError(f"Replay generation {generation} has no held-out rows.")
        corpus[str(generation)] = {
            "rows": rows,
            "parent_objective_loss": parent,
            "candidate_objective_loss": child,
            "delta_candidate_minus_parent": child - parent,
        }
    aggregate_delta = float(
        training["validation_loss_delta_child_minus_parent"]["objective_loss"]
    )
    checks = {
        "aggregate_objective_improved": aggregate_delta < 0.0,
        "replay_7_objective_improved": corpus["7"]["delta_candidate_minus_parent"] < 0.0,
        "replay_8_objective_improved": corpus["8"]["delta_candidate_minus_parent"] < 0.0,
        "replay_10_not_meaningfully_regressed": (
            corpus["10"]["delta_candidate_minus_parent"]
            <= NEWEST_OBJECTIVE_TOLERANCE
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "aggregate_objective_delta_candidate_minus_parent": aggregate_delta,
        "newest_objective_tolerance": NEWEST_OBJECTIVE_TOLERANCE,
        "held_out_corpus_metrics": corpus,
        "validation_games": int(training["validation_games"]),
        "validation_rows": int(training["validation_rows"]),
    }


def consolidation_decision(
    *,
    retention_passed: bool,
    internal_passed: bool | None,
    external_progress: dict[str, Any] | None,
) -> str:
    if not retention_passed:
        return "consolidation_retention_gate_failed_no_gameplay_no_install"
    if internal_passed is not True:
        return "consolidation_retained_but_not_internally_stronger_external_skipped_no_install"
    if external_progress is None:
        raise ValueError("An internal pass requires an external comparison.")
    if float(external_progress["ci95_low"]) > 0.0:
        return "consolidation_internal_pass_external_progress_confirmed_no_install"
    if float(external_progress["mean_difference"]) > 0.0:
        return "consolidation_internal_pass_external_progress_directional_no_install"
    return "consolidation_internal_pass_no_external_progress_detected_no_install"


def run_replay_consolidation_107(config: ReplayConsolidationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 107 result is not a valid completed result.")

    result: dict[str, Any] = {
        "kind": "alphazero_generation_11_replay_consolidation_107",
        "status": "running",
        "active_stage": "consolidated_training",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "training": None,
        "retention_assessment": None,
        "internal_panels": {},
        "external_panels": {},
        "comparisons": {},
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_teacher_used": False,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    training_stage = _run_or_load_training(config)
    result["training"] = training_stage
    result["checkpoint_trained"] = True
    result["retention_assessment"] = retention_assessment(training_stage["training"])
    _write_progress(output, result, started)

    if not result["retention_assessment"]["passed"]:
        result["status"] = "complete"
        result["active_stage"] = "complete"
        result["decision"] = consolidation_decision(
            retention_passed=False, internal_passed=None, external_progress=None
        )
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(sealed_path, result)
        return result

    candidate = Path(training_stage["child_checkpoint"])
    search = _evaluation_search()
    internal_specs = (
        (
            "raw_consolidated_candidate_vs_raw_generation_11",
            alpha_zero_raw_factory(candidate, config.device),
            alpha_zero_raw_factory(config.generation_11_checkpoint, config.device),
        ),
        (
            "searched_consolidated_candidate_vs_searched_generation_11",
            alpha_zero_search_factory(candidate, config.device, search),
            alpha_zero_search_factory(config.generation_11_checkpoint, config.device, search),
        ),
    )
    for index, (name, candidate_factory, baseline_factory) in enumerate(internal_specs):
        result["active_stage"] = name
        _write_progress(output, result, started)
        result["internal_panels"][name] = _run_panel(
            output=output / "arenas",
            name=name,
            candidate=candidate_factory,
            baseline=baseline_factory,
            seed_start=INTERNAL_SEED_START,
            bootstrap_seed=1_070_300 + index,
            max_turns=config.max_turns,
            include_practical=True,
        )
        _write_progress(output, result, started)

    searched_internal = result["internal_panels"][
        "searched_consolidated_candidate_vs_searched_generation_11"
    ]
    internal_passed = bool(searched_internal["assessment"]["passed"])
    if not internal_passed:
        result["status"] = "complete"
        result["active_stage"] = "complete"
        result["decision"] = consolidation_decision(
            retention_passed=True, internal_passed=False, external_progress=None
        )
        result["elapsed_seconds"] = time.perf_counter() - started
        _write_json(sealed_path, result)
        return result

    external_specs = (
        (
            "searched_consolidated_candidate_vs_raw_035",
            alpha_zero_search_factory(candidate, config.device, search),
        ),
        (
            "raw_consolidated_candidate_vs_raw_035",
            alpha_zero_raw_factory(candidate, config.device),
        ),
    )
    for index, (name, candidate_factory) in enumerate(external_specs):
        result["active_stage"] = name
        _write_progress(output, result, started)
        result["external_panels"][name] = _run_panel(
            output=output / "arenas",
            name=name,
            candidate=candidate_factory,
            baseline=policy_035_factory(config.checkpoint_035, config.device),
            seed_start=EXTERNAL_SEED_START,
            bootstrap_seed=1_070_400 + index,
            max_turns=config.max_turns,
            include_practical=False,
        )
        result["external_panels"][name]["parity_assessment"] = external_parity_assessment(
            result["external_panels"][name]["arena"]
        )
        _write_progress(output, result, started)

    prior = json.loads(config.result_106.read_text(encoding="utf-8"))
    searched = result["external_panels"][
        "searched_consolidated_candidate_vs_raw_035"
    ]["arena"]
    raw = result["external_panels"]["raw_consolidated_candidate_vs_raw_035"]["arena"]
    searched_progress = independent_paired_difference_bootstrap(
        searched,
        prior["panels"]["searched_generation_11_vs_raw_035"]["arena"],
        samples=BOOTSTRAP_SAMPLES,
        seed=1_070_500,
    )
    result["comparisons"] = {
        "searched_consolidated_minus_generation_11_vs_035": searched_progress,
        "raw_consolidated_minus_generation_11_vs_035": independent_paired_difference_bootstrap(
            raw,
            prior["panels"]["raw_generation_11_vs_raw_035"]["arena"],
            samples=BOOTSTRAP_SAMPLES,
            seed=1_070_501,
        ),
        "consolidated_search_minus_raw_vs_035": paired_arena_difference_bootstrap(
            searched, raw, samples=BOOTSTRAP_SAMPLES, seed=1_070_502
        ),
    }
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = consolidation_decision(
        retention_passed=True,
        internal_passed=True,
        external_progress=searched_progress,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def _run_or_load_training(config: ReplayConsolidationConfig) -> dict[str, Any]:
    output = config.output_dir
    manifest = output / "training" / "consolidated_generation_012.stage.json"
    replay_paths = [config.replay_7, config.replay_8, config.replay_9, config.replay_10]
    expected_replays = [_sha256(path) for path in replay_paths]
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("Training manifest protocol hash mismatch.")
        if payload.get("parent_checkpoint_sha256") != _sha256(config.generation_11_checkpoint):
            raise ValueError("Training manifest has the wrong parent checkpoint.")
        if payload.get("replay_sha256") != expected_replays:
            raise ValueError("Training manifest has the wrong replay window.")
        if payload.get("child_checkpoint_sha256") != _sha256(payload["child_checkpoint"]):
            raise ValueError("Training child checkpoint hash mismatch.")
        if payload.get("metrics_sha256") != _sha256(payload["metrics_path"]):
            raise ValueError("Training metrics hash mismatch.")
        training = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
        validity = training_validity(training, expected_parent=11)
        _, identity = load_checkpoint(payload["child_checkpoint"], device=config.device)
        if not validity["valid"] or identity.generation != 12:
            raise ValueError("Training manifest failed exact lineage validation.")
        return {**payload, "training": training, "training_validity": validity}

    child = output / "checkpoints" / "consolidated_generation_012.pt"
    metrics = output / "training" / "consolidated_generation_012.json"
    training = train_successor(
        parent_checkpoint=config.generation_11_checkpoint,
        replay_paths=replay_paths,
        output_checkpoint=child,
        metrics_path=metrics,
        device=config.device,
        config=consolidation_training_config(),
    )
    validity = training_validity(training, expected_parent=11)
    _, identity = load_checkpoint(child, device=config.device)
    validity["checks"]["checkpoint_roundtrip_generation"] = identity.generation == 12
    validity["valid"] = bool(
        validity["valid"] and validity["checks"]["checkpoint_roundtrip_generation"]
    )
    if not validity["valid"]:
        raise RuntimeError("Consolidated generation-12 training failed validity checks.")
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_checkpoint": str(config.generation_11_checkpoint),
        "parent_checkpoint_sha256": _sha256(config.generation_11_checkpoint),
        "replay_paths": [str(path) for path in replay_paths],
        "replay_sha256": expected_replays,
        "child_checkpoint": str(child),
        "child_checkpoint_sha256": _sha256(child),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
    }
    _write_json(manifest, payload)
    return {**payload, "training": training, "training_validity": validity}


def _run_panel(
    *,
    output: Path,
    name: str,
    candidate: Any,
    baseline: Any,
    seed_start: int,
    bootstrap_seed: int,
    max_turns: int,
    include_practical: bool,
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
    result = {
        "arena": arena,
        "paired_bootstrap": paired_score_share_bootstrap(
            arena, samples=BOOTSTRAP_SAMPLES, seed=bootstrap_seed
        ),
    }
    if include_practical:
        result["assessment"] = practical_assessment(arena)
    return result


def _evaluation_search() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_070_107,
    )


def _verify_inputs(config: ReplayConsolidationConfig) -> None:
    paths = {
        "result_105": config.result_105,
        "result_106": config.result_106,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "replay_7": config.replay_7,
        "replay_8": config.replay_8,
        "replay_9": config.replay_9,
        "replay_10": config.replay_10,
        "checkpoint_035": config.checkpoint_035,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 107 input hash mismatch for {name}: {actual}")
    prior_105 = json.loads(config.result_105.read_text(encoding="utf-8"))
    if (
        prior_105.get("status") != "complete"
        or prior_105.get("decision") != "pooled_compounding_confirmed_review_035_no_install"
        or not prior_105["pooled_assessment"]["assessment"]["passed"]
    ):
        raise ValueError("Frozen 105 result lacks its exact pooled pass.")
    prior_106 = json.loads(config.result_106.read_text(encoding="utf-8"))
    searched = prior_106.get("panels", {}).get("searched_generation_11_vs_raw_035", {}).get("arena", {})
    raw = prior_106.get("panels", {}).get("raw_generation_11_vs_raw_035", {}).get("arena", {})
    if (
        prior_106.get("status") != "complete"
        or prior_106.get("decision") != "generation_11_no_external_progress_detected_below_035_no_install"
        or (searched.get("candidate_wins"), searched.get("draws"), searched.get("candidate_losses")) != (53, 0, 147)
        or (raw.get("candidate_wins"), raw.get("draws"), raw.get("candidate_losses")) != (25, 1, 174)
    ):
        raise ValueError("Frozen 106 result lacks its exact external anchors.")
    _, identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    if identity.generation != 11 or identity.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen parent is not exact generation 11.")
    if prior_106.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]:
        raise ValueError("Frozen 106 result did not retain exact generation 2.")


def _jsonable_config(config: ReplayConsolidationConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )
