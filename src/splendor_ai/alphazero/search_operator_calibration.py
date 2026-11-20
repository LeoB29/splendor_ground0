"""Value audit and bounded PUCT calibration for programme 094."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .replay import AlphaZeroReplayRow, read_replay_jsonl
from .search import SearchConfig
from .search_primary import (
    SearchSpec,
    _arena_dict,
    _seat_score_shares,
    paired_score_share_bootstrap,
)


@dataclass(frozen=True, slots=True)
class SearchOperatorCalibrationConfig:
    output_dir: Path
    incumbent_checkpoint: Path
    replay_generation_2: Path
    result_093: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


EXPECTED_HASHES = {
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "replay_generation_2": "33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b",
    "result_093": "3fcba198fbcdcf48b4909c84eae6e8402dfb8fee941231791efa98fe0423f54d",
}

SEARCH_GRID = tuple(
    SearchSpec(
        name=f"puct_{simulations}_c{str(c_puct).replace('.', '_')}",
        simulations=simulations,
        c_puct=c_puct,
    )
    for simulations in (64, 128, 256)
    for c_puct in (0.75, 1.5, 2.25)
)


def summarize_value_predictions(
    values: Iterable[tuple[float, float]],
) -> dict[str, Any]:
    pairs = [(float(prediction), float(target)) for prediction, target in values]
    if not pairs:
        return {"count": 0}
    errors = [prediction - target for prediction, target in pairs]
    probabilities = [
        ((max(-1.0, min(1.0, prediction)) + 1.0) / 2.0, (target + 1.0) / 2.0)
        for prediction, target in pairs
    ]
    bins: list[dict[str, Any]] = []
    for index in range(10):
        members = [
            (predicted, actual)
            for predicted, actual in probabilities
            if min(9, int(predicted * 10.0)) == index
        ]
        if members:
            mean_predicted = sum(row[0] for row in members) / len(members)
            mean_actual = sum(row[1] for row in members) / len(members)
            bins.append(
                {
                    "bin": index,
                    "lower": index / 10.0,
                    "upper": (index + 1) / 10.0,
                    "count": len(members),
                    "mean_predicted_probability": mean_predicted,
                    "mean_actual_score": mean_actual,
                    "calibration_gap": mean_predicted - mean_actual,
                }
            )
    return {
        "count": len(pairs),
        "mean_prediction": sum(row[0] for row in pairs) / len(pairs),
        "mean_target": sum(row[1] for row in pairs) / len(pairs),
        "bias": sum(errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mse": sum(error * error for error in errors) / len(errors),
        "sign_accuracy": sum(
            _sign(prediction) == _sign(target) for prediction, target in pairs
        )
        / len(pairs),
        "brier_score": sum(
            (predicted - actual) ** 2 for predicted, actual in probabilities
        )
        / len(probabilities),
        "calibration_bins": bins,
    }


def value_calibration_audit(rows: Iterable[AlphaZeroReplayRow]) -> dict[str, Any]:
    complete = [row for row in rows if row.value_target is not None]
    if not complete:
        raise ValueError("Value audit requires complete replay outcomes.")

    def summarize(selected: list[AlphaZeroReplayRow], field: str) -> dict[str, Any]:
        return summarize_value_predictions(
            (float(getattr(row, field)), float(row.value_target)) for row in selected
        )

    phase_rows = {
        "early": [row for row in complete if row.turn_index < 20],
        "mid": [row for row in complete if 20 <= row.turn_index < 40],
        "late": [row for row in complete if row.turn_index >= 40],
    }
    outcome_rows = {
        "loss": [row for row in complete if float(row.value_target) < 0.0],
        "draw": [row for row in complete if float(row.value_target) == 0.0],
        "win": [row for row in complete if float(row.value_target) > 0.0],
    }

    def report(field: str) -> dict[str, Any]:
        return {
            "overall": summarize(complete, field),
            "by_phase": {
                name: summarize(selected, field) for name, selected in phase_rows.items()
            },
            "by_outcome": {
                name: summarize(selected, field)
                for name, selected in outcome_rows.items()
            },
        }

    return {
        "replay_rows": len(complete),
        "games": len({row.game_seed for row in complete}),
        "phase_definition": {"early": "0-19", "mid": "20-39", "late": "40+"},
        "raw_network_value": report("raw_network_value"),
        "stored_search_root_value": report("search_root_value"),
    }


def discovery_eligibility(arena: dict[str, Any]) -> dict[str, Any]:
    seats = _seat_score_shares(arena)
    checks = {
        "score_share": arena["score_share"] >= 0.55,
        "positive_margin": arena["mean_margin"] > 0.0,
        "seat_zero_at_least_45pct": seats[0] >= 0.45,
        "seat_one_at_least_45pct": seats[1] >= 0.45,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "seat_score_shares": {str(key): value for key, value in seats.items()},
    }


def select_search_configuration(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["eligibility"]["eligible"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -row["arena"]["score_share"],
            -row["arena"]["mean_margin"],
            row["spec"]["simulations"],
            abs(row["spec"]["c_puct"] - 1.5),
            row["order"],
        ),
    )


def confirmation_acceptance(
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


def run_search_operator_calibration_094(
    config: SearchOperatorCalibrationConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    controlling = json.loads(config.result_093.read_text(encoding="utf-8"))
    if controlling.get("decision") != "large_confirmation_failed_keep_generation_2":
        raise ValueError("The hash-locked 093 result does not retain generation 2.")
    result: dict[str, Any] = {
        "kind": "alphazero_search_operator_calibration_094",
        "status": "running",
        "strength_claim": False,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "value_audit": None,
        "search_discovery": [],
        "search_selection": None,
        "search_confirmation": None,
        "installed_checkpoint": str(config.incumbent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.incumbent_checkpoint),
        "selfplay_generated": False,
        "child_checkpoint_generated": False,
        "external_035_arena": None,
        "decision": "running",
    }
    _write_progress(output, result, started)
    result["value_audit"] = value_calibration_audit(
        read_replay_jsonl(config.replay_generation_2)
    )
    _write_progress(output, result, started)

    for order, spec in enumerate(SEARCH_GRID):
        arena = _arena_dict(
            run_paired_arena(
                name=f"{spec.name}_search_vs_generation_2_raw_discovery",
                candidate_factory=alpha_zero_search_factory(
                    config.incumbent_checkpoint,
                    config.device,
                    _search_config(spec),
                ),
                baseline_factory=alpha_zero_raw_factory(
                    config.incumbent_checkpoint, config.device
                ),
                seed_start=1_640_000 + order * 100,
                paired_seeds=10,
                max_turns=config.max_turns,
                progress_path=output / "arenas" / f"discovery_{spec.name}.json",
                resume=True,
            )
        )
        result["search_discovery"].append(
            {
                "order": order,
                "spec": asdict(spec),
                "arena": arena,
                "eligibility": discovery_eligibility(arena),
            }
        )
        _write_progress(output, result, started)

    selected = select_search_configuration(result["search_discovery"])
    if selected is None:
        return _finish(
            output, result, started, decision="no_search_configuration_discovered"
        )
    spec = SearchSpec(**selected["spec"])
    result["search_selection"] = {
        "spec": asdict(spec),
        "discovery_order": selected["order"],
        "selection_basis": (
            "share_then_margin_then_lower_simulations_then_c_puct_distance_from_1_5"
        ),
    }
    _write_progress(output, result, started)
    arena = _arena_dict(
        run_paired_arena(
            name=f"{spec.name}_search_vs_generation_2_raw_confirmation",
            candidate_factory=alpha_zero_search_factory(
                config.incumbent_checkpoint, config.device, _search_config(spec)
            ),
            baseline_factory=alpha_zero_raw_factory(
                config.incumbent_checkpoint, config.device
            ),
            seed_start=1_650_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "search_confirmation.json",
            resume=True,
        )
    )
    bootstrap = paired_score_share_bootstrap(arena, samples=50_000, seed=940_094)
    acceptance = confirmation_acceptance(arena, bootstrap)
    result["search_confirmation"] = {
        "arena": arena,
        "paired_bootstrap": bootstrap,
        "acceptance": acceptance,
    }
    return _finish(
        output,
        result,
        started,
        decision=(
            "search_operator_confirmed"
            if acceptance["accepted"]
            else "search_operator_not_confirmed"
        ),
    )


def _search_config(spec: SearchSpec) -> SearchConfig:
    return SearchConfig(
        simulations=spec.simulations,
        c_puct=spec.c_puct,
        mode="real",
        root_noise_fraction=0.0,
        seed=940_094,
    )


def _verify_inputs(config: SearchOperatorCalibrationConfig) -> None:
    paths = {
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "replay_generation_2": config.replay_generation_2,
        "result_093": config.result_093,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 094 input hash mismatch for {name}: {actual}")


def _jsonable_config(config: SearchOperatorCalibrationConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "incumbent_checkpoint": str(config.incumbent_checkpoint),
        "replay_generation_2": str(config.replay_generation_2),
        "result_093": str(config.result_093),
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


def _sign(value: float) -> int:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return 0
    return 1 if value > 0.0 else -1
