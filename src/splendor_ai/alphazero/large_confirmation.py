"""Exact-candidate large searched confirmation for programme 093."""

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
from .search_primary import (
    SEARCH_SPECS,
    SearchSpec,
    _arena_dict,
    _conditional_continuation,
    _search_config,
    _seat_score_shares,
    paired_score_share_bootstrap,
)
from .training import TrainingConfig


@dataclass(frozen=True, slots=True)
class LargeConfirmationConfig:
    output_dir: Path
    result_092: Path
    incumbent_checkpoint: Path
    candidate_checkpoint: Path
    replay_generation_2: Path
    checkpoint_035: Path | None = None
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


EXPECTED_HASHES = {
    "result_092": "b3a5c2b85174bf73c119b4b10b5b821b6fb0391173c04b920fa523460061c3e1",
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "candidate_checkpoint": "87e7d591bb779df5a378b456abebd6cae94ec32f88420ce7830e5f5a41d0ea2f",
    "replay_generation_2": "33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b",
}


def large_confirmation_acceptance(
    arena: dict[str, Any],
    bootstrap: dict[str, float | int],
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


def run_large_confirmation_093(
    config: LargeConfirmationConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    result_092 = json.loads(config.result_092.read_text(encoding="utf-8"))
    selection = result_092["candidate_selection"]
    if selection["name"] != "value_quarter":
        raise ValueError("Hash-locked 092 selection is not value_quarter.")
    training_config = TrainingConfig(**selection["training_config"])
    comparison_search = _search_config(SearchSpec("comparison_64", 64, 1.5))
    result: dict[str, Any] = {
        "kind": "alphazero_large_confirmation_093",
        "status": "running",
        "strength_claim": False,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "candidate_name": "value_quarter",
        "candidate_training_config": asdict(training_config),
        "searched_confirmation": None,
        "raw_confirmation_diagnostic": None,
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

    searched = _arena_dict(
        run_paired_arena(
            name="value_quarter_search_vs_generation_2_large_confirmation",
            candidate_factory=alpha_zero_search_factory(
                config.candidate_checkpoint, config.device, comparison_search
            ),
            baseline_factory=alpha_zero_search_factory(
                config.incumbent_checkpoint, config.device, comparison_search
            ),
            seed_start=1_630_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_large_confirmation.json",
            resume=True,
        )
    )
    bootstrap = paired_score_share_bootstrap(
        searched,
        samples=50_000,
        seed=930_093,
    )
    acceptance = large_confirmation_acceptance(searched, bootstrap)
    result["searched_confirmation"] = {
        "arena": searched,
        "paired_bootstrap": bootstrap,
        "acceptance": acceptance,
    }
    raw = _arena_dict(
        run_paired_arena(
            name="value_quarter_raw_vs_generation_2_raw_diagnostic",
            candidate_factory=alpha_zero_raw_factory(
                config.candidate_checkpoint, config.device
            ),
            baseline_factory=alpha_zero_raw_factory(
                config.incumbent_checkpoint, config.device
            ),
            seed_start=1_630_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "raw_large_diagnostic.json",
            resume=True,
        )
    )
    result["raw_confirmation_diagnostic"] = raw
    _write_progress(output, result, started)
    if not acceptance["accepted"]:
        return _finish(
            output,
            result,
            started,
            decision="large_confirmation_failed_keep_generation_2",
        )

    result["installed_checkpoint"] = str(config.candidate_checkpoint)
    result["installed_checkpoint_sha256"] = _sha256(config.candidate_checkpoint)
    for order, spec in enumerate(SEARCH_SPECS):
        arena = _arena_dict(
            run_paired_arena(
                name=f"{spec.name}_search_vs_generation_3_raw_discovery",
                candidate_factory=alpha_zero_search_factory(
                    config.candidate_checkpoint, config.device, _search_config(spec)
                ),
                baseline_factory=alpha_zero_raw_factory(
                    config.candidate_checkpoint, config.device
                ),
                seed_start=1_631_000 + order * 100,
                paired_seeds=4,
                max_turns=config.max_turns,
                progress_path=output
                / "arenas"
                / f"search_discovery_{spec.name}.json",
                resume=True,
            )
        )
        checks = {
            "score_share": arena["score_share"] >= 0.625,
            "positive_margin": arena["mean_margin"] > 0.0,
            "normal_terminations": arena["termination_reasons"]
            == {"completed": arena["games"]},
        }
        result["search_discovery"].append(
            {
                "order": order,
                "spec": asdict(spec),
                "arena": arena,
                "eligibility": {
                    "eligible": all(checks.values()),
                    "checks": checks,
                },
            }
        )
        _write_progress(output, result, started)
    eligible = [
        row
        for row in result["search_discovery"]
        if row["eligibility"]["eligible"]
    ]
    if not eligible:
        return _finish(
            output,
            result,
            started,
            decision="generation_3_confirmed_no_search_improvement_candidate",
        )
    selected = min(
        eligible,
        key=lambda row: (
            -row["arena"]["score_share"],
            -row["arena"]["mean_margin"],
            row["spec"]["simulations"],
            row["order"],
        ),
    )
    search_spec = SearchSpec(**selected["spec"])
    result["search_selection"] = {"spec": asdict(search_spec)}
    search_confirmation = _arena_dict(
        run_paired_arena(
            name="selected_search_vs_generation_3_raw_confirmation",
            candidate_factory=alpha_zero_search_factory(
                config.candidate_checkpoint,
                config.device,
                _search_config(search_spec),
            ),
            baseline_factory=alpha_zero_raw_factory(
                config.candidate_checkpoint, config.device
            ),
            seed_start=1_632_000,
            paired_seeds=8,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "search_confirmation.json",
            resume=True,
        )
    )
    search_checks = {
        "score_share": search_confirmation["score_share"] >= 0.5625,
        "positive_margin": search_confirmation["mean_margin"] > 0.0,
        "normal_terminations": search_confirmation["termination_reasons"]
        == {"completed": search_confirmation["games"]},
    }
    result["search_confirmation"] = {
        "arena": search_confirmation,
        "acceptance": {
            "accepted": all(search_checks.values()),
            "checks": search_checks,
        },
    }
    _write_progress(output, result, started)
    if not result["search_confirmation"]["acceptance"]["accepted"]:
        return _finish(
            output,
            result,
            started,
            decision="generation_3_confirmed_search_improvement_not_confirmed",
        )

    continuation = _conditional_continuation(
        config=config,  # type: ignore[arg-type]
        output=output,
        parent_checkpoint=config.candidate_checkpoint,
        training_config=training_config,
        search_spec=search_spec,
        selfplay_seed_start=950_000,
        arena_seed_start=1_633_000,
    )
    result["conditional_continuation"] = continuation
    if continuation["acceptance"]["accepted"]:
        result["installed_checkpoint"] = continuation["child_checkpoint"]
        result["installed_checkpoint_sha256"] = continuation[
            "child_checkpoint_sha256"
        ]
        decision = "generation_4_accepted"
    else:
        decision = "generation_4_rejected_keep_confirmed_generation_3"
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
                seed_start=1_634_000,
                paired_seeds=8,
                max_turns=config.max_turns,
                progress_path=output / "arenas" / "external_035.json",
                resume=True,
            )
        )
    return _finish(output, result, started, decision=decision)


def _verify_inputs(config: LargeConfirmationConfig) -> None:
    paths = {
        "result_092": config.result_092,
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "candidate_checkpoint": config.candidate_checkpoint,
        "replay_generation_2": config.replay_generation_2,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 093 input hash mismatch for {name}: {actual}")


def _jsonable_config(config: LargeConfirmationConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "result_092": str(config.result_092),
        "incumbent_checkpoint": str(config.incumbent_checkpoint),
        "candidate_checkpoint": str(config.candidate_checkpoint),
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
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
