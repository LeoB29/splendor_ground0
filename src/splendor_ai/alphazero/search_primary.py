"""Search-primary candidate confirmation and continuation for programme 092."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from .arena import (
    ArenaResult,
    alpha_zero_raw_factory,
    alpha_zero_search_factory,
    policy_035_factory,
    run_paired_arena,
)
from .corrective import candidate_offline_eligibility
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play
from .search import SearchConfig
from .training import TrainingConfig, train_successor


@dataclass(frozen=True, slots=True)
class SearchPrimaryConfig:
    output_dir: Path
    result_091: Path
    incumbent_checkpoint: Path
    balanced_half_checkpoint: Path
    value_quarter_checkpoint: Path
    replay_generation_2: Path
    checkpoint_035: Path | None = None
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


@dataclass(frozen=True, slots=True)
class SearchSpec:
    name: str
    simulations: int
    c_puct: float


EXPECTED_HASHES = {
    "result_091": "b3dc44707e8d5948c3f18da5799bfc27c6bd99411e0e9b799ec84cbaeaa3335b",
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "balanced_half_checkpoint": "cf930d481696f11aecb0cb556f98ec51ef044bab8c2c24838a9e12bdb8300b31",
    "value_quarter_checkpoint": "87e7d591bb779df5a378b456abebd6cae94ec32f88420ce7830e5f5a41d0ea2f",
    "replay_generation_2": "33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b",
}
SEARCH_SPECS = (
    SearchSpec("baseline_64", 64, 1.5),
    SearchSpec("focused_96", 96, 1.0),
    SearchSpec("exploratory_96", 96, 2.0),
)


def paired_score_share_bootstrap(
    arena: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    clusters: dict[int, list[float]] = {}
    for row in arena["rows"]:
        clusters.setdefault(int(row["seed"]), []).append(
            float(row["candidate_points"])
        )
    if not clusters or any(len(values) != 2 for values in clusters.values()):
        raise ValueError("Paired bootstrap requires exactly two games per seed.")
    cluster_means = [sum(clusters[key]) / 2.0 for key in sorted(clusters)]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(
            sum(rng.choice(cluster_means) for _ in cluster_means)
            / len(cluster_means)
        )
    draws.sort()
    return {
        "paired_seeds": len(cluster_means),
        "samples": samples,
        "seed": seed,
        "mean": sum(cluster_means) / len(cluster_means),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def candidate_discovery_eligibility(arena: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "score_share": arena["score_share"] >= 0.5,
        "nonnegative_margin": arena["mean_margin"] >= 0.0,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {"eligible": all(checks.values()), "checks": checks}


def select_discovery_candidate(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["eligibility"]["eligible"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -row["arena"]["score_share"],
            -row["arena"]["mean_margin"],
            row["unweighted_validation_total_loss"],
            row["order"],
        ),
    )


def searched_confirmation_acceptance(
    arena: dict[str, Any],
    bootstrap: dict[str, float | int],
) -> dict[str, Any]:
    seat_shares = _seat_score_shares(arena)
    checks = {
        "score_share": arena["score_share"] >= 0.5625,
        "bootstrap_ci95_low_above_half": float(bootstrap["ci95_low"]) > 0.5,
        "positive_margin": arena["mean_margin"] > 0.0,
        "seat_zero_nonlosing": seat_shares[0] >= 0.5,
        "seat_one_nonlosing": seat_shares[1] >= 0.5,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "seat_score_shares": {str(key): value for key, value in seat_shares.items()},
    }


def run_search_primary_092(config: SearchPrimaryConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    result_091 = json.loads(config.result_091.read_text(encoding="utf-8"))
    candidates = _candidate_inputs(config, result_091)
    result: dict[str, Any] = {
        "kind": "alphazero_search_primary_092",
        "status": "running",
        "strength_claim": False,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "candidate_discovery": [],
        "candidate_selection": None,
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
    comparison_search = _search_config(SearchSpec("comparison_64", 64, 1.5))
    for order, candidate in enumerate(candidates):
        arena = _arena_dict(
            run_paired_arena(
                name=f"{candidate['name']}_search_vs_incumbent_search_discovery",
                candidate_factory=alpha_zero_search_factory(
                    candidate["checkpoint"], config.device, comparison_search
                ),
                baseline_factory=alpha_zero_search_factory(
                    config.incumbent_checkpoint, config.device, comparison_search
                ),
                seed_start=1_620_000,
                paired_seeds=8,
                max_turns=config.max_turns,
                progress_path=output
                / "arenas"
                / f"candidate_discovery_{candidate['name']}.json",
                resume=True,
            )
        )
        result["candidate_discovery"].append(
            {
                **candidate,
                "order": order,
                "arena": arena,
                "eligibility": candidate_discovery_eligibility(arena),
            }
        )
        _write_progress(output, result, started)
    selected = select_discovery_candidate(result["candidate_discovery"])
    if selected is None:
        return _finish(
            output,
            result,
            started,
            decision="no_candidate_discovery_eligible",
        )
    selected_checkpoint = Path(selected["checkpoint"])
    result["candidate_selection"] = {
        "name": selected["name"],
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "training_config": selected["training_config"],
        "selection_basis": "searched_discovery_then_comparable_unweighted_loss",
    }
    _write_progress(output, result, started)

    confirmation = _arena_dict(
        run_paired_arena(
            name="selected_search_vs_incumbent_search_confirmation",
            candidate_factory=alpha_zero_search_factory(
                selected_checkpoint, config.device, comparison_search
            ),
            baseline_factory=alpha_zero_search_factory(
                config.incumbent_checkpoint, config.device, comparison_search
            ),
            seed_start=1_621_000,
            paired_seeds=16,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_confirmation.json",
            resume=True,
        )
    )
    bootstrap = paired_score_share_bootstrap(
        confirmation,
        samples=20_000,
        seed=920_092,
    )
    acceptance = searched_confirmation_acceptance(confirmation, bootstrap)
    result["searched_confirmation"] = {
        "arena": confirmation,
        "paired_bootstrap": bootstrap,
        "acceptance": acceptance,
    }
    raw_diagnostic = _arena_dict(
        run_paired_arena(
            name="selected_raw_vs_incumbent_raw_diagnostic",
            candidate_factory=alpha_zero_raw_factory(
                selected_checkpoint, config.device
            ),
            baseline_factory=alpha_zero_raw_factory(
                config.incumbent_checkpoint, config.device
            ),
            seed_start=1_621_000,
            paired_seeds=16,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "raw_confirmation_diagnostic.json",
            resume=True,
        )
    )
    result["raw_confirmation_diagnostic"] = raw_diagnostic
    _write_progress(output, result, started)
    if not acceptance["accepted"]:
        return _finish(
            output,
            result,
            started,
            decision="searched_candidate_not_confirmed_keep_generation_2",
        )

    result["installed_checkpoint"] = str(selected_checkpoint)
    result["installed_checkpoint_sha256"] = selected["checkpoint_sha256"]
    for order, spec in enumerate(SEARCH_SPECS):
        arena = _arena_dict(
            run_paired_arena(
                name=f"{spec.name}_search_vs_selected_raw_discovery",
                candidate_factory=alpha_zero_search_factory(
                    selected_checkpoint, config.device, _search_config(spec)
                ),
                baseline_factory=alpha_zero_raw_factory(
                    selected_checkpoint, config.device
                ),
                seed_start=1_622_000 + order * 100,
                paired_seeds=4,
                max_turns=config.max_turns,
                progress_path=output
                / "arenas"
                / f"search_discovery_{spec.name}.json",
                resume=True,
            )
        )
        eligibility = candidate_discovery_eligibility(arena)
        eligibility["checks"]["score_share"] = arena["score_share"] >= 0.625
        eligibility["checks"]["positive_margin"] = arena["mean_margin"] > 0.0
        eligibility["checks"].pop("nonnegative_margin")
        eligibility["eligible"] = all(eligibility["checks"].values())
        result["search_discovery"].append(
            {
                "order": order,
                "spec": asdict(spec),
                "arena": arena,
                "eligibility": eligibility,
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
            decision="candidate_accepted_no_search_improvement_candidate",
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
    search_spec = SearchSpec(**selected_search["spec"])
    result["search_selection"] = {"spec": asdict(search_spec)}
    search_confirmation = _arena_dict(
        run_paired_arena(
            name="selected_search_vs_selected_raw_confirmation",
            candidate_factory=alpha_zero_search_factory(
                selected_checkpoint, config.device, _search_config(search_spec)
            ),
            baseline_factory=alpha_zero_raw_factory(
                selected_checkpoint, config.device
            ),
            seed_start=1_623_000,
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
            decision="candidate_accepted_search_improvement_not_confirmed",
        )

    continuation = _conditional_continuation(
        config=config,
        output=output,
        parent_checkpoint=selected_checkpoint,
        training_config=TrainingConfig(**selected["training_config"]),
        search_spec=search_spec,
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
                seed_start=1_625_000,
                paired_seeds=8,
                max_turns=config.max_turns,
                progress_path=output / "arenas" / "external_035.json",
                resume=True,
            )
        )
    return _finish(output, result, started, decision=decision)


def _conditional_continuation(
    *,
    config: SearchPrimaryConfig,
    output: Path,
    parent_checkpoint: Path,
    training_config: TrainingConfig,
    search_spec: SearchSpec,
    selfplay_seed_start: int = 940_000,
    arena_seed_start: int = 1_624_000,
) -> dict[str, Any]:
    directory = output / "conditional_continuation"
    replay = directory / "generation_003_selfplay.jsonl"
    search = replace(
        _search_config(search_spec),
        root_noise_fraction=0.25,
        seed=940_092,
    )
    selfplay = generate_self_play(
        checkpoint_path=parent_checkpoint,
        device=config.device,
        config=SelfPlayConfig(
            generation=3,
            game_count=32,
            seed_start=selfplay_seed_start,
            search=search,
            max_turns=config.max_turns,
        ),
        replay_path=replay,
        progress_path=directory / "generation_003_selfplay.progress.json",
    )
    child = directory / "generation_004.pt"
    training = train_successor(
        parent_checkpoint=parent_checkpoint,
        replay_paths=[config.replay_generation_2, replay],
        output_checkpoint=child,
        metrics_path=directory / "generation_004_training.json",
        device=config.device,
        config=training_config,
    )
    offline = candidate_offline_eligibility(training)
    searched_arena = None
    raw_arena = None
    if offline["eligible"]:
        evaluation_search = _search_config(search_spec)
        searched_arena = _arena_dict(
            run_paired_arena(
                name="generation_4_search_vs_generation_3_search",
                candidate_factory=alpha_zero_search_factory(
                    child, config.device, evaluation_search
                ),
                baseline_factory=alpha_zero_search_factory(
                    parent_checkpoint, config.device, evaluation_search
                ),
                seed_start=arena_seed_start,
                paired_seeds=8,
                max_turns=config.max_turns,
                progress_path=output / "arenas" / "generation_4_search_gate.json",
                resume=True,
            )
        )
        raw_arena = _arena_dict(
            run_paired_arena(
                name="generation_4_raw_vs_generation_3_raw_diagnostic",
                candidate_factory=alpha_zero_raw_factory(child, config.device),
                baseline_factory=alpha_zero_raw_factory(
                    parent_checkpoint, config.device
                ),
                seed_start=arena_seed_start,
                paired_seeds=8,
                max_turns=config.max_turns,
                progress_path=output / "arenas" / "generation_4_raw_diagnostic.json",
                resume=True,
            )
        )
        checks = {
            "offline_eligible": True,
            "score_share": searched_arena["score_share"] >= 0.5625,
            "positive_margin": searched_arena["mean_margin"] > 0.0,
            "normal_terminations": searched_arena["termination_reasons"]
            == {"completed": searched_arena["games"]},
        }
    else:
        checks = {
            "offline_eligible": False,
            "score_share": False,
            "positive_margin": False,
            "normal_terminations": False,
        }
    return {
        "selfplay": summarize_self_play(selfplay),
        "replay_path": str(replay),
        "replay_sha256": _sha256(replay),
        "child_checkpoint": str(child),
        "child_checkpoint_sha256": _sha256(child),
        "training": training,
        "offline_eligibility": offline,
        "searched_arena": searched_arena,
        "raw_arena_diagnostic": raw_arena,
        "acceptance": {"accepted": all(checks.values()), "checks": checks},
    }


def _candidate_inputs(
    config: SearchPrimaryConfig,
    result_091: dict[str, Any],
) -> list[dict[str, Any]]:
    checkpoint_by_name = {
        "balanced_half": config.balanced_half_checkpoint,
        "value_quarter": config.value_quarter_checkpoint,
    }
    rows = []
    for candidate in result_091["candidate_screen"]:
        name = candidate["spec"]["name"]
        if name not in checkpoint_by_name:
            continue
        rows.append(
            {
                "name": name,
                "checkpoint": str(checkpoint_by_name[name]),
                "checkpoint_sha256": _sha256(checkpoint_by_name[name]),
                "unweighted_validation_total_loss": candidate["training"][
                    "child_validation_metrics"
                ]["total_loss"],
                "training_config": candidate["training"]["config"],
            }
        )
    if [row["name"] for row in rows] != ["balanced_half", "value_quarter"]:
        raise ValueError("091 result does not contain the two frozen candidates.")
    return rows


def _seat_score_shares(arena: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for seat in (0, 1):
        points = [
            float(row["candidate_points"])
            for row in arena["rows"]
            if int(row["candidate_seat"]) == seat
        ]
        if not points:
            raise ValueError(f"Arena has no candidate games from seat {seat}.")
        result[seat] = sum(points) / len(points)
    return result


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of no values.")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _search_config(spec: SearchSpec) -> SearchConfig:
    return SearchConfig(
        simulations=spec.simulations,
        c_puct=spec.c_puct,
        mode="real",
        root_noise_fraction=0.0,
        seed=920_092,
    )


def _verify_inputs(config: SearchPrimaryConfig) -> None:
    paths = {
        "result_091": config.result_091,
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "balanced_half_checkpoint": config.balanced_half_checkpoint,
        "value_quarter_checkpoint": config.value_quarter_checkpoint,
        "replay_generation_2": config.replay_generation_2,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 092 input hash mismatch for {name}: {actual}")


def _arena_dict(result: ArenaResult) -> dict[str, Any]:
    return asdict(result)


def _jsonable_config(config: SearchPrimaryConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "result_091": str(config.result_091),
        "incumbent_checkpoint": str(config.incumbent_checkpoint),
        "balanced_half_checkpoint": str(config.balanced_half_checkpoint),
        "value_quarter_checkpoint": str(config.value_quarter_checkpoint),
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
