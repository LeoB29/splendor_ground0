"""Run the fresh history-required recurrent succession repair (064)."""

from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.cfr.full_game import PublicHistoryEncoding
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.recurrent_league import (
    RecurrentLeagueConfig,
    RecurrentLeagueStep,
    collect_recurrent_league,
    forward_recurrent,
    policy_diagnostics,
    recurrent_snapshot_factory,
    run_parent_gate,
    run_static_population_gate,
    save_generation_checkpoint,
    sha256,
    static_opponent_factories,
    train_recurrent_generation,
    verify_checkpoint_reload,
)
from splendor_ai.training.recurrent_policy import (
    PublicHistoryTracker,
    RecurrentPolicyConfig,
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
    load_recurrent_policy_from_parent,
    save_recurrent_policy_checkpoint,
)


RESULT_063_SHA256 = (
    "ee6b5275ff012081b7bce97637ccd64fd1bae25ae74e07c434f71915c6882a73"
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
COLLECTION_SEEDS = {1: 640_000, 2: 641_000}
PARENT_GATE_SEEDS = {1: 642_100, 2: 642_200}
FINAL_SEEDS = {
    "035": 643_000,
    "016": 643_100,
    "023": 643_200,
    "shallow": 643_300,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prerequisites(
    result_063: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    expected = {
        "result_063": (result_063, RESULT_063_SHA256),
        "035": (checkpoint_035, CHECKPOINT_035_SHA256),
        "016": (checkpoint_016, CHECKPOINT_016_SHA256),
        "023": (checkpoint_023, CHECKPOINT_023_SHA256),
    }
    payload = {}
    for label, (path, digest) in expected.items():
        actual = sha256(path)
        payload[label] = {
            "path": str(path).replace("\\", "/"),
            "sha256": actual,
            "expected_sha256": digest,
            "passed": actual == digest,
        }
    diagnosis = json.loads(result_063.read_text(encoding="utf-8"))
    payload["result_063"]["decision"] = diagnosis.get("decision")
    payload["result_063"]["decision_passed"] = diagnosis.get("decision") == (
        "generation2_failure_diagnostic_complete_no_scale_authorized"
    )
    payload["all_passed"] = all(
        row["passed"] for row in payload.values()
    ) and payload["result_063"]["decision_passed"]
    return payload


def _epoch0_contract(model, checkpoint_path: Path) -> dict[str, Any]:
    env = SplendorEnv(seed=640_064)
    state = env.initial_state()
    history = PublicHistoryTracker()
    scores_exact = True
    values_exact = True
    choices_exact = True
    rows = []
    while len(rows) < 32 and not state.terminal:
        actions = env.legal_actions(state)
        row = build_recurrent_policy_input(state, actions, history.events)
        batch = collate_recurrent_policy_inputs([row])
        with torch.no_grad():
            scores, value = forward_recurrent(model, batch)
            parent_scores, parent_value = model.parent(
                batch["observation"],
                batch["legal_action_features"],
                batch["legal_action_mask"],
            )
        scores_exact &= torch.equal(scores, parent_scores)
        values_exact &= torch.equal(value, parent_value)
        candidate_row = int(torch.argmax(scores[0, : len(actions)]).item())
        parent_row = int(torch.argmax(parent_scores[0, : len(actions)]).item())
        choices_exact &= candidate_row == parent_row
        chosen = actions[parent_row]
        rows.append(row)
        history.record(state, chosen)
        state = env.step(state, chosen)
    reloaded = load_recurrent_policy_checkpoint(checkpoint_path).eval()
    reload_exact = True
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = collate_recurrent_policy_inputs(rows[start : start + 16])
            expected = forward_recurrent(model, batch)
            actual = forward_recurrent(reloaded, batch)
            reload_exact &= torch.equal(expected[0], actual[0])
            reload_exact &= torch.equal(expected[1], actual[1])
    return {
        "decisions": len(rows),
        "scores_exact": scores_exact,
        "values_exact": values_exact,
        "greedy_choices_exact": choices_exact,
        "checkpoint_reload_exact": reload_exact,
        "passed": (
            len(rows) == 32
            and scores_exact
            and values_exact
            and choices_exact
            and reload_exact
        ),
    }


def _empty_history_steps(
    steps: list[RecurrentLeagueStep],
) -> list[RecurrentLeagueStep]:
    empty = PublicHistoryEncoding((), (), (), ())
    return [
        replace(
            step,
            policy_input=replace(step.policy_input, history=empty),
        )
        for step in steps
    ]


def _empty_history_fallback_exact(
    model,
    steps: list[RecurrentLeagueStep],
    config: RecurrentLeagueConfig,
) -> bool:
    empty_steps = _empty_history_steps(steps)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(empty_steps), config.batch_size):
            chosen = empty_steps[start : start + config.batch_size]
            batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in chosen],
                config.device,
            )
            scores, value = forward_recurrent(model, batch)
            parent_scores, parent_value = model.parent(
                batch["observation"],
                batch["legal_action_features"],
                batch["legal_action_mask"],
            )
            if not torch.equal(scores, parent_scores):
                return False
            if not torch.equal(value, parent_value):
                return False
    return True


def _rotated_history_steps(
    steps: list[RecurrentLeagueStep],
) -> tuple[list[RecurrentLeagueStep], int]:
    groups: dict[tuple[str, int, int], list[int]] = {}
    for index, step in enumerate(steps):
        key = (
            step.opponent_label,
            step.actor_seat,
            step.policy_input.turn,
        )
        groups.setdefault(key, []).append(index)
    histories = [step.policy_input.history for step in steps]
    rotated = list(histories)
    changed = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for offset, index in enumerate(indices):
            source = indices[(offset + 1) % len(indices)]
            rotated[index] = histories[source]
            changed += rotated[index] != histories[index]
    return (
        [
            replace(
                step,
                policy_input=replace(
                    step.policy_input,
                    history=rotated[index],
                ),
            )
            for index, step in enumerate(steps)
        ],
        changed,
    )


def _history_rotation_tv(
    model,
    steps: list[RecurrentLeagueStep],
    config: RecurrentLeagueConfig,
) -> dict[str, Any]:
    rotated, changed = _rotated_history_steps(steps)
    total_weight = 0.0
    tv_sum = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(steps), config.batch_size):
            real_rows = steps[start : start + config.batch_size]
            rotated_rows = rotated[start : start + config.batch_size]
            real_batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in real_rows],
                config.device,
            )
            rotated_batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in rotated_rows],
                config.device,
            )
            real_scores, _ = forward_recurrent(model, real_batch)
            rotated_scores, _ = forward_recurrent(model, rotated_batch)
            mask = real_batch["legal_action_mask"]
            real_probabilities = torch.softmax(
                real_scores.masked_fill(~mask, -torch.inf),
                dim=1,
            )
            rotated_probabilities = torch.softmax(
                rotated_scores.masked_fill(~mask, -torch.inf),
                dim=1,
            )
            for row, step in enumerate(real_rows):
                width = len(step.policy_input.legal_action_features)
                weight = step.game_weight
                tv = float(
                    (
                        0.5
                        * torch.abs(
                            real_probabilities[row, :width]
                            - rotated_probabilities[row, :width]
                        ).sum()
                    ).cpu()
                )
                total_weight += weight
                tv_sum += weight * tv
    return {
        "decisions": len(steps),
        "histories_changed": changed,
        "mean_total_variation": tv_sum / max(total_weight, 1e-12),
    }


def _balance_exact(
    collection,
    *,
    opponent_count: int,
    games: int,
) -> bool:
    expected = games // (2 * opponent_count)
    for label, seats in collection.opponent_seat_games.items():
        if any(value != expected for value in seats.values()):
            return False
        for seat in ("seat0", "seat1"):
            if collection.split_opponent_seat_games["train"][label][seat] != 32:
                return False
            if (
                collection.split_opponent_seat_games["validation"][label][seat]
                != 8
            ):
                return False
    return True


def _generation_gates(
    collection,
    update,
    diagnostics,
    midlate_diagnostics,
    history_use,
    empty_exact,
    parent_gate,
    *,
    model,
    opponent_count: int,
    games: int,
    maximum_attempts: int,
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
        midlate_diagnostics["surrogate"],
        history_use["mean_total_variation"],
    ]
    return {
        "exact_collection": collection.eligible_games == games
        and collection.attempted_games <= maximum_attempts,
        "exact_balance_and_split": _balance_exact(
            collection,
            opponent_count=opponent_count,
            games=games,
        ),
        "eligible_only": (
            sum(collection.eligible_termination_counts.values())
            == collection.eligible_games
            and set(collection.eligible_termination_counts).issubset(
                {"completed", "forced_pass_deadlock"}
            )
        ),
        "zero_illegal_actions": collection.termination_counts.get(
            "illegal_action",
            0,
        )
        == 0,
        "behavior_reconstruction": (
            update.behavior_reconstruction_max_error <= 1e-5
        ),
        "finite_metrics": all(math.isfinite(value) for value in numeric),
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
        "midlate_surrogate_positive": midlate_diagnostics["surrogate"] > 0.0,
        "validation_entropy": update.validation_normalized_entropy >= 0.30,
        "empty_history_exact_035": empty_exact,
        "history_content_used": (
            history_use["histories_changed"] > 0
            and history_use["mean_total_variation"] >= 0.002
        ),
        "parent_gate_valid": bool(parent_gate["valid"]),
        "parent_score_share": parent_gate["candidate_score_share"] >= 0.45,
        "parent_score_margin": parent_gate["mean_paired_score_margin"] >= -1.5,
    }


def run(
    *,
    output_dir: Path,
    result_063: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    prerequisites = _prerequisites(
        result_063,
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    if not prerequisites["all_passed"]:
        payload = {
            "experiment": "064",
            "prerequisites": prerequisites,
            "decision": "history_required_repair_invalid_no_design_decision",
        }
        _atomic_json(output_dir / "result.json", payload)
        return payload

    torch.manual_seed(640_064)
    epoch0 = load_recurrent_policy_from_parent(
        checkpoint_035,
        config=RecurrentPolicyConfig(
            residual_mode="history_required_interaction",
        ),
    ).eval()
    epoch0_path = output_dir / "history_required_epoch0.pt"
    save_recurrent_policy_checkpoint(
        epoch0_path,
        epoch0,
        {
            "experiment": "064",
            "generation": 0,
            "parent_path": str(checkpoint_035).replace("\\", "/"),
            "parent_sha256": sha256(checkpoint_035),
            "training_steps": 0,
        },
    )
    epoch0_contract = _epoch0_contract(epoch0, epoch0_path)
    if not epoch0_contract["passed"]:
        payload = {
            "experiment": "064",
            "prerequisites": prerequisites,
            "epoch0_contract": epoch0_contract,
            "decision": "history_required_repair_invalid_no_design_decision",
        }
        _atomic_json(output_dir / "result.json", payload)
        return payload

    static = static_opponent_factories(
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    accepted_models = []
    parent = epoch0
    parent_path = epoch0_path
    generation_payloads = []
    stopped_generation = None
    for generation in (1, 2):
        games_target = 320 if generation == 1 else 400
        attempt_limit = 400 if generation == 1 else 480
        config = RecurrentLeagueConfig(
            games=games_target,
            maximum_attempts=attempt_limit,
            minimum_validation_games_per_cell=8,
            validation_divisor=5,
            early_turn_weight=0.25,
            mid_turn_weight=1.0,
            late_turn_weight=1.0,
        )
        league = list(static)
        if generation == 2:
            league.append(
                ("snapshot_g1", recurrent_snapshot_factory(accepted_models[0]))
            )
        games, steps, collection = collect_recurrent_league(
            parent,
            league,
            generation=generation,
            seed_start=COLLECTION_SEEDS[generation],
            config=config,
        )
        candidate, update = train_recurrent_generation(
            parent,
            steps,
            generation=generation,
            config=config,
        )
        candidate_path = output_dir / f"generation_{generation}.pt"
        save_generation_checkpoint(
            candidate_path,
            candidate,
            generation=generation,
            parent_checkpoint=parent_path,
            provenance={
                "architecture": "history_required_interaction",
                "collection_seed_start": COLLECTION_SEEDS[generation],
                "league": [label for label, _ in league],
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
        diagnostics = policy_diagnostics(
            parent,
            candidate,
            validation_steps,
            config,
        )
        midlate = policy_diagnostics(
            parent,
            candidate,
            [
                step
                for step in validation_steps
                if step.policy_input.turn >= 20
            ],
            config,
        )
        empty_exact = _empty_history_fallback_exact(
            candidate,
            validation_steps,
            config,
        )
        history_use = _history_rotation_tv(
            candidate,
            validation_steps,
            config,
        )
        parent_gate = run_parent_gate(
            candidate,
            parent,
            seed_start=PARENT_GATE_SEEDS[generation],
            pairs=40,
            config=config,
        )
        gates = _generation_gates(
            collection,
            update,
            diagnostics,
            midlate,
            history_use,
            empty_exact,
            parent_gate,
            model=candidate,
            opponent_count=len(league),
            games=games_target,
            maximum_attempts=attempt_limit,
        )
        accepted = all(gates.values())
        generation_payload = {
            "generation": generation,
            "parent_checkpoint": str(parent_path).replace("\\", "/"),
            "candidate_checkpoint": str(candidate_path).replace("\\", "/"),
            "candidate_sha256": sha256(candidate_path),
            "league": [label for label, _ in league],
            "collection": asdict(collection),
            "games": [game.summary() for game in games],
            "update": asdict(update),
            "validation_diagnostics": diagnostics,
            "midlate_validation_diagnostics": midlate,
            "empty_history_exact_035": empty_exact,
            "history_rotation": history_use,
            "parent_gate": parent_gate,
            "gates": gates,
            "accepted": accepted,
        }
        generation_payloads.append(generation_payload)
        _atomic_json(
            output_dir / f"generation_{generation}.json",
            generation_payload,
        )
        if not accepted:
            stopped_generation = generation
            break
        accepted_models.append(candidate)
        parent = candidate
        parent_path = candidate_path

    final_population = None
    final_gates = {}
    if len(accepted_models) == 2:
        final_config = RecurrentLeagueConfig(
            early_turn_weight=0.25,
            mid_turn_weight=1.0,
            late_turn_weight=1.0,
        )
        final_population = run_static_population_gate(
            accepted_models[-1],
            epoch0,
            static,
            seed_starts=FINAL_SEEDS,
            pairs=10,
            config=final_config,
        )
        deltas = [
            row["delta"]
            for row in final_population["opponents"].values()
        ]
        final_gates = {
            "all_160_games_valid": final_population["valid"]
            and final_population["games"] == 160,
            "aggregate_delta_positive": final_population["aggregate_delta"]
            > 0.0,
            "two_opponents_nonnegative": sum(delta >= 0.0 for delta in deltas)
            >= 2,
            "no_opponent_collapse": min(deltas) >= -0.10,
        }
    passed = len(accepted_models) == 2 and all(final_gates.values())
    decision = (
        "history_required_repair_composes_authorize_stage_c_preregistration"
        if passed
        else "history_required_repair_failed_diagnose_before_scale"
    )
    payload = {
        "experiment": "064",
        "protocol": "docs/experiments/history_required_recurrent_repair_064.md",
        "prerequisites": prerequisites,
        "epoch0_checkpoint": {
            "path": str(epoch0_path).replace("\\", "/"),
            "sha256": sha256(epoch0_path),
        },
        "epoch0_contract": epoch0_contract,
        "generation_count_attempted": len(generation_payloads),
        "generation_count_accepted": len(accepted_models),
        "stopped_generation": stopped_generation,
        "generations": generation_payloads,
        "final_population_gate": final_population,
        "final_gates": final_gates,
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
        default=Path("outputs/history_required_recurrent_repair_064"),
    )
    parser.add_argument(
        "--result-063",
        type=Path,
        default=Path(
            "outputs/recurrent_generation2_diagnosis_063/result.json"
        ),
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
        result_063=args.result_063,
        checkpoint_035=args.checkpoint_035,
        checkpoint_016=args.checkpoint_016,
        checkpoint_023=args.checkpoint_023,
    )
    print(
        json.dumps(
            {
                "accepted_generations": result.get(
                    "generation_count_accepted",
                    0,
                ),
                "decision": result["decision"],
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
