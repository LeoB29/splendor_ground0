"""Frozen decisive-policy objective screen for programme 097."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .confirmed_operator_successor import (
    searched_confirmation_acceptance,
    searched_development_acceptance,
)
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .training import TrainingConfig, train_successor


@dataclass(frozen=True, slots=True)
class DecisiveObjectiveConfig:
    output_dir: Path
    parent_checkpoint: Path
    replay_generation_2: Path
    replay_095: Path
    result_095: Path
    result_096: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "86ecf5196a6dfb93ea11162b78217c5a347b90856758e10d7ab4da862ba4b6e6"
EXPECTED_HASHES = {
    "parent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "replay_generation_2": "33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b",
    "replay_095": "c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47",
    "result_095": "6d2ec319982ced65855d526e1fe2c692c1f4bda49d8e3f60c473b693b4714655",
    "result_096": "11cc6d1fcc72a6744fa87646dd31ddac89a505c0d86a6e1396287a1dd48259e6",
}
CANDIDATES = (
    ("standard_control", 0.0, 0.0),
    ("divergence_weighted", 2.0, 0.0),
    ("visit_argmax_aux", 0.0, 0.5),
)


def run_decisive_objective_097(config: DecisiveObjectiveConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    result: dict[str, Any] = {
        "kind": "alphazero_decisive_objective_097",
        "status": "running",
        "strength_claim": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "candidates": {},
        "offline_selection": None,
        "searched_development": None,
        "searched_confirmation": None,
        "raw_confirmation_diagnostic": None,
        "installed_checkpoint": str(config.parent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.parent_checkpoint),
        "external_035_arena": None,
        "decision": "running",
    }
    _write_progress(output, result, started)
    replay_paths = [config.replay_generation_2, config.replay_095]
    for name, divergence_weight, aux_weight in CANDIDATES:
        checkpoint = output / "candidates" / name / "generation_003.pt"
        training = train_successor(
            parent_checkpoint=config.parent_checkpoint,
            replay_paths=replay_paths,
            output_checkpoint=checkpoint,
            metrics_path=output / "candidates" / name / "training.json",
            device=config.device,
            config=_training_config(divergence_weight, aux_weight),
        )
        result["candidates"][name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "training": training,
        }
        _write_progress(output, result, started)

    selection = offline_selection(result["candidates"])
    result["offline_selection"] = selection
    _write_progress(output, result, started)
    if selection["selected"] is None:
        return _finish(output, result, started, "no_corrective_offline_candidate_keep_generation_2")

    selected_name = str(selection["selected"])
    selected = Path(result["candidates"][selected_name]["checkpoint"])
    search = _evaluation_search()
    development = _arena_dict(
        run_paired_arena(
            name=f"{selected_name}_search_vs_generation_2_search_development",
            candidate_factory=alpha_zero_search_factory(selected, config.device, search),
            baseline_factory=alpha_zero_search_factory(config.parent_checkpoint, config.device, search),
            seed_start=1_680_000,
            paired_seeds=16,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_development.json",
            resume=True,
        )
    )
    dev_acceptance = searched_development_acceptance(development)
    result["searched_development"] = {"arena": development, "acceptance": dev_acceptance}
    _write_progress(output, result, started)
    if not dev_acceptance["accepted"]:
        return _finish(output, result, started, "searched_development_failed_keep_generation_2")

    confirmation = _arena_dict(
        run_paired_arena(
            name=f"{selected_name}_search_vs_generation_2_search_confirmation",
            candidate_factory=alpha_zero_search_factory(selected, config.device, search),
            baseline_factory=alpha_zero_search_factory(config.parent_checkpoint, config.device, search),
            seed_start=1_681_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_confirmation.json",
            resume=True,
        )
    )
    bootstrap = paired_score_share_bootstrap(confirmation, samples=50_000, seed=970_097)
    confirm_acceptance = searched_confirmation_acceptance(confirmation, bootstrap)
    result["searched_confirmation"] = {
        "arena": confirmation,
        "paired_bootstrap": bootstrap,
        "acceptance": confirm_acceptance,
    }
    _write_progress(output, result, started)
    raw = _arena_dict(
        run_paired_arena(
            name=f"{selected_name}_raw_vs_generation_2_raw_diagnostic",
            candidate_factory=alpha_zero_raw_factory(selected, config.device),
            baseline_factory=alpha_zero_raw_factory(config.parent_checkpoint, config.device),
            seed_start=1_681_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "raw_confirmation_diagnostic.json",
            resume=True,
        )
    )
    result["raw_confirmation_diagnostic"] = raw
    if confirm_acceptance["accepted"]:
        result["installed_checkpoint"] = str(selected)
        result["installed_checkpoint_sha256"] = _sha256(selected)
        decision = "generation_3_corrective_confirmed_and_installed"
    else:
        decision = "searched_confirmation_failed_keep_generation_2"
    return _finish(output, result, started, decision)


def offline_selection(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control = candidates["standard_control"]["training"]
    control_decisive = control["validation_slices"]["decisive_search_change"]["child"]
    eligibility: dict[str, Any] = {}
    for name in ("divergence_weighted", "visit_argmax_aux"):
        training = candidates[name]["training"]
        parent = training["parent_validation_metrics"]
        child = training["child_validation_metrics"]
        decisive = training["validation_slices"]["decisive_search_change"]["child"]
        phase_checks = {
            slice_name: row["delta_child_minus_parent"]["value_loss"] <= 0.025
            for slice_name, row in training["validation_slices"].items()
            if slice_name.startswith("phase_")
        }
        checks = {
            "ordinary_policy_not_worse_than_parent": child["policy_loss"] <= parent["policy_loss"],
            "value_not_worse_than_parent": child["value_loss"] <= parent["value_loss"],
            "overall_top_agreement_not_worse_than_parent": child["policy_top_target_agreement"] >= parent["policy_top_target_agreement"],
            "phase_value_degradation_bounded": bool(phase_checks) and all(phase_checks.values()),
            "decisive_agreement_gain_vs_control": decisive["policy_top_target_agreement"] >= control_decisive["policy_top_target_agreement"] + 0.02,
            "decisive_probability_gain_vs_control": decisive["policy_target_top_probability"] >= control_decisive["policy_target_top_probability"] + 0.01,
        }
        eligibility[name] = {
            "eligible": all(checks.values()),
            "checks": checks,
            "phase_value_checks": phase_checks,
            "decisive_metrics": decisive,
        }
    eligible = [name for name, row in eligibility.items() if row["eligible"]]
    order = {"divergence_weighted": 1, "visit_argmax_aux": 0}
    eligible.sort(
        key=lambda name: (
            eligibility[name]["decisive_metrics"]["policy_top_target_agreement"],
            eligibility[name]["decisive_metrics"]["policy_target_top_probability"],
            -eligibility[name]["decisive_metrics"]["policy_loss"],
            order[name],
        ),
        reverse=True,
    )
    return {
        "selected": eligible[0] if eligible else None,
        "eligible_candidates": eligible,
        "control_decisive_metrics": control_decisive,
        "eligibility": eligibility,
        "selection_order": "decisive agreement, decisive target probability, lower ordinary decisive CE, frozen candidate order",
    }


def _training_config(divergence_weight: float, aux_weight: float) -> TrainingConfig:
    return TrainingConfig(
        epochs=6, batch_size=256, learning_rate=1e-4, weight_decay=1e-4,
        validation_fraction=0.2, seed=970_097, policy_loss_weight=1.0,
        value_loss_weight=0.5, early_stopping_patience=2,
        early_stopping_min_delta=0.002, bucket_batches=True,
        policy_divergence_weight=divergence_weight,
        visit_argmax_aux_weight=aux_weight,
    )


def _evaluation_search() -> SearchConfig:
    return SearchConfig(simulations=256, c_puct=2.25, mode="real", root_noise_fraction=0.0, seed=970_097)


def _verify_inputs(config: DecisiveObjectiveConfig) -> None:
    paths = {name: getattr(config, name) for name in EXPECTED_HASHES}
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 097 input hash mismatch for {name}: {actual}")
    result_095 = json.loads(config.result_095.read_text(encoding="utf-8"))
    result_096 = json.loads(config.result_096.read_text(encoding="utf-8"))
    if result_095.get("decision") != "searched_confirmation_failed_keep_generation_2":
        raise ValueError("Frozen 095 result has the wrong controlling decision.")
    if result_096.get("decision") != "no_clear_gameplay_head_attribution":
        raise ValueError("Frozen 096 result has the wrong controlling decision.")


def _jsonable_config(config: DecisiveObjectiveConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _finish(output: Path, result: dict[str, Any], started: float, decision: str) -> dict[str, Any]:
    result["status"] = "complete"
    result["decision"] = decision
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
