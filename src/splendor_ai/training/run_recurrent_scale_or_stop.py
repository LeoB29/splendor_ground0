"""Run the preregistered recurrent scale-or-stop experiment (069)."""

from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.training.recurrent_league import (
    RecurrentLeagueConfig,
    VALID_TERMINATIONS,
    collect_recurrent_league,
    policy_diagnostics,
    recurrent_snapshot_factory,
    run_parent_gate,
    run_static_population_gate,
    sha256,
    static_opponent_factories,
    train_recurrent_generation,
    verify_checkpoint_reload,
)
from splendor_ai.training.recurrent_policy import (
    load_recurrent_policy_checkpoint,
    save_recurrent_policy_checkpoint,
)
from splendor_ai.training.run_history_required_recurrent_repair import (
    _empty_history_fallback_exact,
    _history_rotation_tv,
)
from splendor_ai.training.run_recurrent_broad_failure_diagnosis import (
    OPPONENT_ORDER,
    _paired_breakdown,
    _paired_seed_records,
)


PROTOCOL_COMMIT = "887f40e"
RESULT_068_SHA256 = (
    "d4d366e6e77de3af9dba78f51b2341442198ca3752dd464361b1888e7d81d9b1"
)
RESULT_064_SHA256 = (
    "7a3094d37e6d37961b44f74029330c7083f5da3af7a9a03f8050b62bc59cf2e6"
)
GENERATION1_SHA256 = (
    "0c42c04df97114bf79147dfee80e8f296485b15742d6248c8f37ff7cee78756e"
)
EPOCH0_SHA256 = (
    "3dd945ba0c525e936603042040b2c33ab70ad8552e007b6c8f2dc4619813290a"
)
GENERATION2_SHA256 = (
    "3ed0bd7e2dcda21646c7ac4f68e2d0ffcfe0bdacf0c151c453c52ea6a08fe09e"
)
CHECKPOINT_035_SHA256 = (
    "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
)
CHECKPOINT_016_SHA256 = (
    "876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056"
)
CHECKPOINT_023_SHA256 = (
    "29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070"
)

COLLECTION_SEEDS = {1: 690_000, 2: 700_000, 3: 710_000}
COLLECTION_GAMES = {1: 2_880, 2: 3_360, 3: 3_840}
COLLECTION_ATTEMPTS = {1: 3_600, 2: 4_200, 3: 4_800}
PARENT_GATE_SEEDS = {1: 720_000, 2: 721_000, 3: 722_000}
FINAL_SEEDS = {
    "035": 730_000,
    "016": 731_000,
    "023": 732_000,
    "shallow": 733_000,
}
GAMES_PER_CELL = 240
TRAIN_GAMES_PER_CELL = 192
VALIDATION_GAMES_PER_CELL = 48
PARENT_GATE_PAIRS = 50
FINAL_PAIRS = 100


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, model, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_recurrent_policy_checkpoint(temporary, model, metadata)
    os.replace(temporary, path)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return False


def _expected_paths(
    result_068: Path,
    result_064: Path,
    generation1: Path,
    epoch0: Path,
    generation2: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, tuple[Path, str]]:
    return {
        "result_068": (result_068, RESULT_068_SHA256),
        "result_064": (result_064, RESULT_064_SHA256),
        "generation1": (generation1, GENERATION1_SHA256),
        "epoch0": (epoch0, EPOCH0_SHA256),
        "generation2": (generation2, GENERATION2_SHA256),
        "035": (checkpoint_035, CHECKPOINT_035_SHA256),
        "016": (checkpoint_016, CHECKPOINT_016_SHA256),
        "023": (checkpoint_023, CHECKPOINT_023_SHA256),
    }


def _prerequisites(expected: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for label, (path, digest) in expected.items():
        actual = sha256(path)
        payload[label] = {
            "path": str(path).replace("\\", "/"),
            "sha256": actual,
            "expected_sha256": digest,
            "passed": actual == digest,
        }
    result_068 = json.loads(expected["result_068"][0].read_text(encoding="utf-8"))
    failed_strength = sorted(
        key for key, passed in result_068.get("strength_gates", {}).items()
        if not passed
    )
    authorization = {
        "068_decision_exact": result_068.get("decision")
        == "recurrent_seat0_confirmation_failed_no_scale",
        "068_all_integrity_gates": bool(result_068.get("integrity_gates"))
        and all(result_068["integrity_gates"].values()),
        "068_only_primary_ci_gate_failed": failed_strength
        == ["aggregate_score_ci_proves_noninferiority"],
        "068_not_passed": result_068.get("passed") is False,
    }
    result_064 = json.loads(expected["result_064"][0].read_text(encoding="utf-8"))
    authorization.update(
        {
            "064_decision_exact": result_064.get("decision")
            == "history_required_repair_failed_diagnose_before_scale",
            "064_two_generations_accepted": result_064.get(
                "generation_count_accepted"
            )
            == 2,
        }
    )
    payload["authorization"] = authorization
    payload["all_passed"] = (
        all(row["passed"] for key, row in payload.items() if key != "authorization")
        and all(authorization.values())
    )
    return payload


def _hashes_unchanged(expected: dict[str, tuple[Path, str]]) -> bool:
    return all(sha256(path) == digest for path, digest in expected.values())


def _config(scale_generation: int) -> RecurrentLeagueConfig:
    return RecurrentLeagueConfig(
        games=COLLECTION_GAMES[scale_generation],
        maximum_attempts=COLLECTION_ATTEMPTS[scale_generation],
        minimum_validation_games_per_cell=VALIDATION_GAMES_PER_CELL,
        validation_divisor=5,
        early_turn_weight=0.25,
        mid_turn_weight=1.0,
        late_turn_weight=1.0,
    )


def _balance_exact(collection, labels: list[str]) -> bool:
    if set(collection.opponent_seat_games) != set(labels):
        return False
    for label in labels:
        if collection.opponent_seat_games[label] != {
            "seat0": GAMES_PER_CELL,
            "seat1": GAMES_PER_CELL,
        }:
            return False
        if collection.split_opponent_seat_games["train"][label] != {
            "seat0": TRAIN_GAMES_PER_CELL,
            "seat1": TRAIN_GAMES_PER_CELL,
        }:
            return False
        if collection.split_opponent_seat_games["validation"][label] != {
            "seat0": VALIDATION_GAMES_PER_CELL,
            "seat1": VALIDATION_GAMES_PER_CELL,
        }:
            return False
    return True


def _generation_gates(
    *,
    collection,
    games: list[dict[str, Any]],
    update,
    diagnostics: dict[str, Any],
    midlate: dict[str, Any],
    empty_exact: bool,
    history_use: dict[str, Any],
    parent_gate: dict[str, Any],
    model,
    labels: list[str],
    scale_generation: int,
    hashes_unchanged: bool,
) -> dict[str, bool]:
    residual_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("parent.")
    ]
    numeric = [
        update.behavior_reconstruction_max_error,
        update.first_gradient_norm,
        update.validation_raw_kl,
        update.validation_argmax_movement,
        update.validation_surrogate,
        update.validation_normalized_entropy,
        diagnostics["raw_kl"],
        diagnostics["argmax_movement"],
        diagnostics["surrogate"],
        midlate["surrogate"],
        history_use["mean_total_variation"],
        parent_gate["candidate_score_share"],
        parent_gate["mean_paired_score_margin"],
    ]
    lower = COLLECTION_SEEDS[scale_generation]
    upper = lower + COLLECTION_ATTEMPTS[scale_generation]
    seeds = [int(game["seed"]) for game in games]
    return {
        "exact_collection": collection.eligible_games
        == COLLECTION_GAMES[scale_generation]
        and collection.attempted_games <= COLLECTION_ATTEMPTS[scale_generation]
        and len(games) == COLLECTION_GAMES[scale_generation],
        "exact_balance_and_split": _balance_exact(collection, labels),
        "exact_unique_reserved_seeds": len(seeds) == len(set(seeds))
        and all(lower <= seed < upper for seed in seeds),
        "eligible_only": sum(collection.eligible_termination_counts.values())
        == collection.eligible_games
        and set(collection.eligible_termination_counts).issubset(
            VALID_TERMINATIONS
        ),
        "zero_illegal_actions": collection.termination_counts.get(
            "illegal_action", 0
        )
        == 0,
        "behavior_reconstruction": update.behavior_reconstruction_max_error
        <= 1e-5,
        "finite_metrics": all(math.isfinite(float(value)) for value in numeric),
        "gradient_present": update.first_gradient_norm > 1e-8,
        "parent_frozen": all(
            not parameter.requires_grad for parameter in model.parent.parameters()
        ),
        "history_residual_trainable": bool(residual_parameters)
        and all(parameter.requires_grad for parameter in residual_parameters),
        "checkpoint_reload_exact": update.checkpoint_reload_exact,
        "validation_kl_band": 0.0001 <= diagnostics["raw_kl"] <= 0.0100,
        "validation_argmax_band": 0.005
        <= diagnostics["argmax_movement"]
        <= 0.12,
        "validation_surrogate_positive": diagnostics["surrogate"] > 0.0,
        "midlate_surrogate_positive": midlate["surrogate"] > 0.0,
        "validation_entropy": update.validation_normalized_entropy >= 0.30,
        "empty_history_exact_035": empty_exact,
        "history_content_used": history_use["histories_changed"] > 0
        and history_use["mean_total_variation"] >= 0.002,
        "parent_gate_valid": bool(parent_gate["valid"])
        and parent_gate["games"] == 2 * PARENT_GATE_PAIRS,
        "parent_score_share": parent_gate["candidate_score_share"] >= 0.45,
        "parent_score_margin": parent_gate["mean_paired_score_margin"] >= -1.5,
        "prerequisite_hashes_unchanged": hashes_unchanged,
    }


def _completed_generation(
    output_dir: Path,
    scale_generation: int,
    parent_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    record_path = output_dir / f"scale_generation_{scale_generation}.json"
    candidate_path = output_dir / f"scale_generation_{scale_generation}.pt"
    if not record_path.exists():
        return None, None
    if not candidate_path.exists():
        return None, "completed_generation_checkpoint_missing"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "experiment": "069",
        "protocol_commit": PROTOCOL_COMMIT,
        "scale_generation": scale_generation,
        "parent_checkpoint": str(parent_path).replace("\\", "/"),
        "parent_sha256": sha256(parent_path),
        "candidate_checkpoint": str(candidate_path).replace("\\", "/"),
        "candidate_sha256": sha256(candidate_path),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None, "completed_generation_contract_mismatch"
    if payload.get("accepted") != all(payload.get("gates", {}).values()):
        return None, "completed_generation_acceptance_mismatch"
    return payload, None


def _final_integrity(
    final_population: dict[str, Any],
    records: list[dict[str, Any]],
    inference: dict[str, Any],
    *,
    hashes_unchanged: bool,
) -> dict[str, bool]:
    exact_opponents = set(final_population["opponents"]) == set(OPPONENT_ORDER)
    exact_rows = exact_opponents and all(
        len(final_population["opponents"][label]["rows"]) == FINAL_PAIRS
        for label in OPPONENT_ORDER
    )
    exact_seeds = exact_rows and all(
        [
            int(row["seed"])
            for row in final_population["opponents"][label]["rows"]
        ]
        == list(range(FINAL_SEEDS[label], FINAL_SEEDS[label] + FINAL_PAIRS))
        for label in OPPONENT_ORDER
    )
    endings = [
        result
        for label in OPPONENT_ORDER
        for row in final_population["opponents"][label]["rows"]
        for model in ("candidate", "control")
        for result in row[model]
    ] if exact_rows else []
    score_mean = sum(float(row["score_share_delta"]) for row in records) / len(
        records
    )
    margin_mean = sum(float(row["margin_delta"]) for row in records) / len(records)
    recomputed = math.isclose(
        score_mean,
        float(inference["aggregate"]["score_share_delta"]["mean"]),
        abs_tol=1e-12,
    ) and math.isclose(
        margin_mean,
        float(inference["aggregate"]["margin_delta"]["mean"]),
        abs_tol=1e-12,
    )
    return {
        "exact_four_opponents": exact_opponents,
        "exact_100_clusters_per_opponent": exact_rows,
        "exact_fresh_seed_bands": exact_seeds,
        "exact_1600_games": final_population["games"] == 1_600
        and len(endings) == 1_600
        and len(records) == 400,
        "all_endings_valid": bool(final_population["valid"])
        and all(
            result["termination_reason"] in VALID_TERMINATIONS
            for result in endings
        ),
        "zero_illegal_actions": all(
            result["termination_reason"] != "illegal_action" for result in endings
        ),
        "finite_metrics": _finite(final_population) and _finite(inference),
        "paired_metrics_recomputed_exact": recomputed,
        "prerequisite_hashes_unchanged": hashes_unchanged,
        "no_training_selection_or_adaptation_in_final_gate": True,
    }


def _final_strength(
    inference: dict[str, Any],
    final_policy_diagnostics: dict[str, Any],
) -> dict[str, bool]:
    aggregate_score = inference["aggregate"]["score_share_delta"]
    aggregate_margin = inference["aggregate"]["margin_delta"]
    opponent_intervals = [
        inference["by_opponent"][opponent]["score_share_delta"]
        for opponent in OPPONENT_ORDER
    ]
    opponent_means = [float(interval["mean"]) for interval in opponent_intervals]
    seat_means = [
        float(inference["by_seat"][f"seat{seat}"]["score_share_delta"])
        for seat in (0, 1)
    ]
    half_means = [
        float(inference["split_halves"][half]["score_share_delta"])
        for half in ("even", "odd")
    ]
    base = final_policy_diagnostics["versus_base_generation2"]
    epoch0 = final_policy_diagnostics["versus_epoch0"]
    return {
        "meaningful_aggregate_score_lift": float(aggregate_score["mean"])
        >= 0.025 - 1e-12,
        "aggregate_score_guardrail": float(aggregate_score["lower"])
        >= -0.025 - 1e-12,
        "aggregate_margin_positive": float(aggregate_margin["mean"]) > 0.0,
        "aggregate_margin_guardrail": float(aggregate_margin["lower"])
        >= -0.50 - 1e-12,
        "three_opponents_nonnegative": sum(value >= -1e-12 for value in opponent_means)
        >= 3,
        "no_opponent_point_collapse": min(opponent_means) >= -0.05 - 1e-12,
        "no_systematic_opponent_collapse": all(
            float(interval["upper"]) >= -1e-12 for interval in opponent_intervals
        ),
        "both_seats_guardrail": min(seat_means) >= -0.025 - 1e-12,
        "both_split_halves_guardrail": min(half_means) >= -0.025 - 1e-12,
        "base_generation2_movement_band": 0.01
        <= float(base["argmax_movement"])
        <= 0.25,
        "base_generation2_kl_ceiling": float(base["raw_kl"]) <= 0.04,
        "epoch0_movement_ceiling": float(epoch0["argmax_movement"]) <= 0.30,
        "epoch0_kl_ceiling": float(epoch0["raw_kl"]) <= 0.05,
    }


def _decision(
    *,
    prerequisites_passed: bool,
    resume_error: str | None,
    generation_payloads: list[dict[str, Any]],
    final_integrity: dict[str, bool],
    final_strength: dict[str, bool],
    inference: dict[str, Any] | None,
) -> str:
    if not prerequisites_passed or resume_error is not None:
        return "recurrent_scale_or_stop_invalid_no_learning_decision"
    if len(generation_payloads) != 3 or any(
        not payload.get("accepted", False) for payload in generation_payloads
    ):
        return "recurrent_scale_learning_contract_failed_stop_outcome_only_branch"
    if not final_integrity or not all(final_integrity.values()):
        return "recurrent_scale_or_stop_invalid_no_learning_decision"
    if all(final_strength.values()):
        return (
            "recurrent_scale_learning_signal_"
            "authorize_confirmation_preregistration"
        )
    assert inference is not None
    if (
        float(inference["aggregate"]["score_share_delta"]["upper"]) < -0.025
        or float(inference["aggregate"]["margin_delta"]["upper"]) < -0.50
    ):
        return "recurrent_scale_systematic_regression_stop_outcome_only_branch"
    return "recurrent_scale_remains_neutral_stop_outcome_only_branch"


def run(
    *,
    output_dir: Path,
    result_068: Path,
    result_064: Path,
    generation1_path: Path,
    epoch0_path: Path,
    generation2_path: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    expected = _expected_paths(
        result_068,
        result_064,
        generation1_path,
        epoch0_path,
        generation2_path,
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    prerequisites = _prerequisites(expected)
    if not prerequisites["all_passed"]:
        payload = {
            "experiment": "069",
            "protocol_commit": PROTOCOL_COMMIT,
            "prerequisites": prerequisites,
            "decision": "recurrent_scale_or_stop_invalid_no_learning_decision",
        }
        _atomic_json(output_dir / "result.json", payload)
        return payload

    static = static_opponent_factories(
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    archived_generation1 = load_recurrent_policy_checkpoint(generation1_path).eval()
    base_generation2 = load_recurrent_policy_checkpoint(generation2_path).eval()
    epoch0 = load_recurrent_policy_checkpoint(epoch0_path).eval()
    parent = base_generation2
    parent_path = generation2_path
    accepted_scaled_models = []
    generation_payloads: list[dict[str, Any]] = []
    resume_error = None
    stopped_generation = None

    for scale_generation in (1, 2, 3):
        completed, error = _completed_generation(
            output_dir,
            scale_generation,
            parent_path,
        )
        if error is not None:
            resume_error = error
            stopped_generation = scale_generation
            break
        if completed is not None:
            generation_payloads.append(completed)
            if not completed["accepted"]:
                stopped_generation = scale_generation
                break
            candidate_path = output_dir / f"scale_generation_{scale_generation}.pt"
            candidate = load_recurrent_policy_checkpoint(candidate_path).eval()
            accepted_scaled_models.append(candidate)
            parent = candidate
            parent_path = candidate_path
            print(
                json.dumps(
                    {"event": "generation_resumed", "generation": scale_generation}
                ),
                flush=True,
            )
            continue

        config = _config(scale_generation)
        league = list(static)
        league.extend(
            [
                (
                    "archived_064_generation1",
                    recurrent_snapshot_factory(archived_generation1),
                ),
                (
                    "base_064_generation2",
                    recurrent_snapshot_factory(base_generation2),
                ),
            ]
        )
        league.extend(
            (
                f"scale_generation{index}",
                recurrent_snapshot_factory(model),
            )
            for index, model in enumerate(accepted_scaled_models, start=1)
        )
        labels = [label for label, _factory in league]
        print(
            json.dumps(
                {
                    "event": "collection_started",
                    "generation": scale_generation,
                    "games": config.games,
                    "opponents": len(league),
                }
            ),
            flush=True,
        )
        games, steps, collection = collect_recurrent_league(
            parent,
            league,
            generation=2 + scale_generation,
            seed_start=COLLECTION_SEEDS[scale_generation],
            config=config,
        )
        game_summaries = [game.summary() for game in games]
        print(
            json.dumps(
                {
                    "event": "collection_completed",
                    "generation": scale_generation,
                    "eligible_games": collection.eligible_games,
                    "steps": collection.steps,
                }
            ),
            flush=True,
        )
        torch.manual_seed(690_069 + scale_generation)
        candidate, update = train_recurrent_generation(
            parent,
            steps,
            generation=2 + scale_generation,
            config=config,
        )
        candidate_path = output_dir / f"scale_generation_{scale_generation}.pt"
        _atomic_checkpoint(
            candidate_path,
            candidate,
            {
                "experiment": "069",
                "protocol_commit": PROTOCOL_COMMIT,
                "scale_generation": scale_generation,
                "global_generation": 2 + scale_generation,
                "parent_checkpoint": str(parent_path).replace("\\", "/"),
                "parent_sha256": sha256(parent_path),
                "collection_seed_start": COLLECTION_SEEDS[scale_generation],
                "league": labels,
                "train_config": asdict(config),
            },
        )
        validation_steps = [step for step in steps if step.split == "validation"]
        update = replace(
            update,
            checkpoint_reload_exact=verify_checkpoint_reload(
                candidate_path,
                candidate,
                validation_steps,
                config,
            ),
        )
        diagnostics = policy_diagnostics(parent, candidate, validation_steps, config)
        midlate = policy_diagnostics(
            parent,
            candidate,
            [step for step in validation_steps if step.policy_input.turn >= 20],
            config,
        )
        empty_exact = _empty_history_fallback_exact(
            candidate,
            validation_steps,
            config,
        )
        history_use = _history_rotation_tv(candidate, validation_steps, config)
        parent_gate = run_parent_gate(
            candidate,
            parent,
            seed_start=PARENT_GATE_SEEDS[scale_generation],
            pairs=PARENT_GATE_PAIRS,
            config=config,
        )
        final_policy_diagnostics = None
        if scale_generation == 3:
            final_policy_diagnostics = {
                "versus_base_generation2": policy_diagnostics(
                    base_generation2,
                    candidate,
                    validation_steps,
                    config,
                ),
                "versus_epoch0": policy_diagnostics(
                    epoch0,
                    candidate,
                    validation_steps,
                    config,
                ),
            }
        gates = _generation_gates(
            collection=collection,
            games=game_summaries,
            update=update,
            diagnostics=diagnostics,
            midlate=midlate,
            empty_exact=empty_exact,
            history_use=history_use,
            parent_gate=parent_gate,
            model=candidate,
            labels=labels,
            scale_generation=scale_generation,
            hashes_unchanged=_hashes_unchanged(expected),
        )
        accepted = all(gates.values())
        generation_payload = {
            "experiment": "069",
            "protocol_commit": PROTOCOL_COMMIT,
            "scale_generation": scale_generation,
            "global_generation": 2 + scale_generation,
            "parent_checkpoint": str(parent_path).replace("\\", "/"),
            "parent_sha256": sha256(parent_path),
            "candidate_checkpoint": str(candidate_path).replace("\\", "/"),
            "candidate_sha256": sha256(candidate_path),
            "league": labels,
            "configuration": asdict(config),
            "collection": asdict(collection),
            "games": game_summaries,
            "update": asdict(update),
            "validation_diagnostics": diagnostics,
            "midlate_validation_diagnostics": midlate,
            "empty_history_exact_035": empty_exact,
            "history_rotation": history_use,
            "parent_gate": parent_gate,
            "final_policy_diagnostics": final_policy_diagnostics,
            "gates": gates,
            "accepted": accepted,
        }
        _atomic_json(
            output_dir / f"scale_generation_{scale_generation}.json",
            generation_payload,
        )
        generation_payloads.append(generation_payload)
        _atomic_json(
            output_dir / "progress.json",
            {
                "experiment": "069",
                "protocol_commit": PROTOCOL_COMMIT,
                "completed_generations": len(generation_payloads),
                "accepted_generations": sum(
                    payload["accepted"] for payload in generation_payloads
                ),
                "last_generation": scale_generation,
                "last_generation_accepted": accepted,
            },
        )
        print(
            json.dumps(
                {
                    "event": "generation_completed",
                    "generation": scale_generation,
                    "accepted": accepted,
                    "validation_kl": diagnostics["raw_kl"],
                    "movement": diagnostics["argmax_movement"],
                    "parent_score_share": parent_gate["candidate_score_share"],
                }
            ),
            flush=True,
        )
        if not accepted:
            stopped_generation = scale_generation
            break
        accepted_scaled_models.append(candidate)
        parent = candidate
        parent_path = candidate_path
        # The next collection is constructed before tuple assignment replaces
        # these locals. Release the completed generation explicitly so two
        # full trajectory corpora cannot coexist at the generation boundary.
        del games, steps, validation_steps
        gc.collect()

    final_population = None
    paired_records = None
    paired_inference = None
    final_integrity: dict[str, bool] = {}
    final_strength: dict[str, bool] = {}
    final_policy_diagnostics = None
    if (
        resume_error is None
        and len(generation_payloads) == 3
        and all(payload["accepted"] for payload in generation_payloads)
    ):
        final_policy_diagnostics = generation_payloads[-1][
            "final_policy_diagnostics"
        ]
        if final_policy_diagnostics is None:
            resume_error = "final_policy_diagnostics_missing_from_generation3"
        else:
            final_config = _config(3)
            print(json.dumps({"event": "final_gate_started", "games": 1600}), flush=True)
            final_population = run_static_population_gate(
                accepted_scaled_models[-1],
                epoch0,
                static,
                seed_starts=FINAL_SEEDS,
                pairs=FINAL_PAIRS,
                config=final_config,
            )
            paired_records = _paired_seed_records(
                {"final_population_gate": final_population}
            )
            paired_inference = _paired_breakdown(
                paired_records,
                aggregate_bootstrap_seed=690_069,
                opponent_bootstrap_seed=690_100,
                score_sign_flip_seed=690_169,
                margin_sign_flip_seed=690_170,
            )
            final_integrity = _final_integrity(
                final_population,
                paired_records,
                paired_inference,
                hashes_unchanged=_hashes_unchanged(expected),
            )
            final_strength = _final_strength(
                paired_inference,
                final_policy_diagnostics,
            )

    decision = _decision(
        prerequisites_passed=bool(prerequisites["all_passed"]),
        resume_error=resume_error,
        generation_payloads=generation_payloads,
        final_integrity=final_integrity,
        final_strength=final_strength,
        inference=paired_inference,
    )
    passed = decision == (
        "recurrent_scale_learning_signal_authorize_confirmation_preregistration"
    )
    payload = {
        "experiment": "069",
        "protocol": "docs/experiments/recurrent_scale_or_stop_069.md",
        "protocol_commit": PROTOCOL_COMMIT,
        "prerequisites": prerequisites,
        "configuration": {
            "collection_seeds": COLLECTION_SEEDS,
            "collection_games": COLLECTION_GAMES,
            "collection_attempt_limits": COLLECTION_ATTEMPTS,
            "total_eligible_collection_games": sum(COLLECTION_GAMES.values()),
            "games_per_opponent_seat_cell": GAMES_PER_CELL,
            "parent_gate_seeds": PARENT_GATE_SEEDS,
            "parent_gate_pairs": PARENT_GATE_PAIRS,
            "final_seeds": FINAL_SEEDS,
            "final_pairs_per_opponent": FINAL_PAIRS,
            "training": True,
        },
        "resume_error": resume_error,
        "generation_count_completed": len(generation_payloads),
        "generation_count_accepted": sum(
            payload["accepted"] for payload in generation_payloads
        ),
        "stopped_generation": stopped_generation,
        "generations": generation_payloads,
        "final_population_gate": final_population,
        "paired_seed_records": paired_records,
        "paired_inference": paired_inference,
        "final_policy_diagnostics": final_policy_diagnostics,
        "final_integrity_gates": final_integrity,
        "final_strength_gates": final_strength,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": passed,
        "decision": decision,
    }
    _atomic_json(output_dir / "result.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/recurrent_scale_or_stop_069"),
    )
    parser.add_argument(
        "--result-068",
        type=Path,
        default=Path("outputs/recurrent_seat0_confirmation_068/result.json"),
    )
    parser.add_argument(
        "--result-064",
        type=Path,
        default=Path("outputs/history_required_recurrent_repair_064/result.json"),
    )
    parser.add_argument(
        "--generation1",
        type=Path,
        default=Path("outputs/history_required_recurrent_repair_064/generation_1.pt"),
    )
    parser.add_argument(
        "--epoch0",
        type=Path,
        default=Path(
            "outputs/history_required_recurrent_repair_064/history_required_epoch0.pt"
        ),
    )
    parser.add_argument(
        "--generation2",
        type=Path,
        default=Path("outputs/history_required_recurrent_repair_064/generation_2.pt"),
    )
    parser.add_argument(
        "--checkpoint-035",
        type=Path,
        default=Path(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-016",
        type=Path,
        default=Path(
            "outputs/legal_action_scorer_016_policy_guided_mix/"
            "legal_action_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-023",
        type=Path,
        default=Path(
            "outputs/canonical_entity_coherent_023/"
            "canonical_entity_action_scorer_coherent_width64_best.pt"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        output_dir=args.output_dir,
        result_068=args.result_068,
        result_064=args.result_064,
        generation1_path=args.generation1,
        epoch0_path=args.epoch0,
        generation2_path=args.generation2,
        checkpoint_035=args.checkpoint_035,
        checkpoint_016=args.checkpoint_016,
        checkpoint_023=args.checkpoint_023,
    )
    print(
        json.dumps(
            {
                "accepted_generations": result.get("generation_count_accepted", 0),
                "decision": result["decision"],
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
