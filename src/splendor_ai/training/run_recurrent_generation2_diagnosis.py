"""Diagnose the rejected recurrent generation-2 update without retraining."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable

import torch

from splendor_ai.cfr.full_game import PublicHistoryEncoding
from splendor_ai.training.recurrent_league import (
    RecurrentLeagueConfig,
    RecurrentLeagueStep,
    _behavior_reconstruction_error,
    collect_recurrent_league,
    forward_recurrent,
    recurrent_snapshot_factory,
    sha256,
    static_opponent_factories,
)
from splendor_ai.training.recurrent_policy import (
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
)


RESULT_062_SHA256 = (
    "2c4d245eb683d2008f84a3ac705c1fa684817e0fac3b7b000b07d7112d7eeb5d"
)
EPOCH0_SHA256 = (
    "2d807231666420bb221049aa7e2a4a28358ef62618cf1c652ede67dd7249acef"
)
GENERATION1_SHA256 = (
    "e1805b4ff76cbe43f8ff19f3d38e94ef2ced87713109a451c9c249006945c485"
)
GENERATION2_SHA256 = (
    "b2d4f85b126b469eb70931a92cb9c808c44ceb743eb640803f97495530373d10"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _phase(turn: int) -> str:
    if turn < 20:
        return "early"
    if turn < 50:
        return "mid"
    return "late"


def _raw_policy_records(
    parent,
    candidate,
    steps: list[RecurrentLeagueStep],
    config: RecurrentLeagueConfig,
) -> list[dict[str, Any]]:
    records = []
    parent.eval()
    candidate.eval()
    with torch.no_grad():
        for start in range(0, len(steps), config.batch_size):
            chosen = steps[start : start + config.batch_size]
            batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in chosen],
                config.device,
            )
            parent_scores, _ = forward_recurrent(parent, batch)
            candidate_scores, _ = forward_recurrent(candidate, batch)
            mask = batch["legal_action_mask"]
            old_probabilities = torch.softmax(
                parent_scores.masked_fill(~mask, -torch.inf),
                dim=1,
            )
            new_probabilities = torch.softmax(
                candidate_scores.masked_fill(~mask, -torch.inf),
                dim=1,
            )
            for row, step in enumerate(chosen):
                width = len(step.policy_input.legal_action_features)
                old = old_probabilities[row, :width]
                new = new_probabilities[row, :width]
                log_ratio = float(
                    (
                        new[step.action_row].clamp_min(1e-12).log()
                        - old[step.action_row].clamp_min(1e-12).log()
                    ).cpu()
                )
                kl = float(
                    (
                        old
                        * (
                            old.clamp_min(1e-12).log()
                            - new.clamp_min(1e-12).log()
                        )
                    ).sum().cpu()
                )
                tv = float((0.5 * torch.abs(old - new).sum()).cpu())
                records.append(
                    {
                        "game_id": step.game_id,
                        "opponent": step.opponent_label,
                        "seat": step.actor_seat,
                        "cell": f"{step.opponent_label}:seat{step.actor_seat}",
                        "turn": step.policy_input.turn,
                        "phase": _phase(step.policy_input.turn),
                        "action_type": step.action_type,
                        "split": step.split,
                        "advantage": step.advantage,
                        "weight": step.game_weight,
                        "log_ratio": log_ratio,
                        "contribution": (
                            step.game_weight * step.advantage * log_ratio
                        ),
                        "kl_numerator": step.game_weight * kl,
                        "tv_numerator": step.game_weight * tv,
                        "argmax_changed": int(torch.argmax(old).item())
                        != int(torch.argmax(new).item()),
                    }
                )
    return records


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "decisions": 0,
            "games": 0,
            "weight": 0.0,
            "surrogate": 0.0,
            "raw_kl": 0.0,
            "mean_total_variation": 0.0,
            "argmax_movement": 0.0,
            "positive_game_fraction": 0.0,
        }
    weight = sum(float(row["weight"]) for row in records)
    game_contributions: dict[int, float] = defaultdict(float)
    game_weights: dict[int, float] = defaultdict(float)
    changed = 0.0
    for row in records:
        game_id = int(row["game_id"])
        game_contributions[game_id] += float(row["contribution"])
        game_weights[game_id] += float(row["weight"])
        changed += float(row["weight"]) * bool(row["argmax_changed"])
    game_values = [
        game_contributions[game_id] / game_weights[game_id]
        for game_id in game_contributions
    ]
    return {
        "decisions": len(records),
        "games": len(game_values),
        "weight": weight,
        "surrogate": sum(float(row["contribution"]) for row in records)
        / max(weight, 1e-12),
        "raw_kl": sum(float(row["kl_numerator"]) for row in records)
        / max(weight, 1e-12),
        "mean_total_variation": sum(
            float(row["tv_numerator"]) for row in records
        )
        / max(weight, 1e-12),
        "argmax_movement": changed / max(weight, 1e-12),
        "positive_game_fraction": sum(value > 0.0 for value in game_values)
        / len(game_values),
    }


def _breakdown(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {
        label: _summarize_records(rows)
        for label, rows in sorted(groups.items())
    }


def _game_values(records: list[dict[str, Any]]) -> dict[int, float]:
    numerators: dict[int, float] = defaultdict(float)
    denominators: dict[int, float] = defaultdict(float)
    for row in records:
        game_id = int(row["game_id"])
        numerators[game_id] += float(row["contribution"])
        denominators[game_id] += float(row["weight"])
    return {
        game_id: numerators[game_id] / denominators[game_id]
        for game_id in numerators
    }


def _bootstrap_mean(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty.")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(
            sum(values[rng.randrange(len(values))] for _ in values)
            / len(values)
        )
    draws.sort()
    return {
        "mean": sum(values) / len(values),
        "lower": draws[int(0.025 * samples)],
        "upper": draws[min(int(0.975 * samples), samples - 1)],
        "samples": samples,
    }


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs must be nonempty and aligned.")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


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


def _parameter_direction(epoch0, generation1, generation2) -> dict[str, Any]:
    payload = {}
    names = [
        name
        for name, parameter in generation1.named_parameters()
        if parameter.requires_grad
    ]
    for label, predicate in (
        ("all", lambda _name: True),
        ("history", lambda name: name.startswith("history_")),
        ("residual", lambda name: name.startswith("residual_")),
    ):
        selected = [name for name in names if predicate(name)]
        delta1 = torch.cat(
            [
                (
                    dict(generation1.named_parameters())[name].detach().cpu()
                    - dict(epoch0.named_parameters())[name].detach().cpu()
                ).reshape(-1)
                for name in selected
            ]
        ).double()
        delta2 = torch.cat(
            [
                (
                    dict(generation2.named_parameters())[name].detach().cpu()
                    - dict(generation1.named_parameters())[name].detach().cpu()
                ).reshape(-1)
                for name in selected
            ]
        ).double()
        norm1 = float(torch.linalg.vector_norm(delta1))
        norm2 = float(torch.linalg.vector_norm(delta2))
        denominator = norm1 * norm2
        payload[label] = {
            "parameter_tensors": len(selected),
            "generation1_delta_norm": norm1,
            "generation2_delta_norm": norm2,
            "cosine": (
                float(torch.dot(delta1, delta2)) / denominator
                if denominator > 0.0
                else 0.0
            ),
        }
    return payload


def _logit_direction(
    epoch0,
    generation1,
    generation2,
    steps: list[RecurrentLeagueStep],
    config: RecurrentLeagueConfig,
) -> dict[str, Any]:
    cosines: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(steps), config.batch_size):
            chosen = steps[start : start + config.batch_size]
            batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in chosen],
                config.device,
            )
            scores0, _ = forward_recurrent(epoch0, batch)
            scores1, _ = forward_recurrent(generation1, batch)
            scores2, _ = forward_recurrent(generation2, batch)
            for row, step in enumerate(chosen):
                width = len(step.policy_input.legal_action_features)
                delta1 = scores1[row, :width] - scores0[row, :width]
                delta2 = scores2[row, :width] - scores1[row, :width]
                delta1 = delta1 - delta1.mean()
                delta2 = delta2 - delta2.mean()
                denominator = float(
                    torch.linalg.vector_norm(delta1)
                    * torch.linalg.vector_norm(delta2)
                )
                cosine = (
                    float(torch.dot(delta1, delta2)) / denominator
                    if denominator > 0.0
                    else 0.0
                )
                cosines["overall"].append(cosine)
                cosines[_phase(step.policy_input.turn)].append(cosine)
    return {
        label: {
            "decisions": len(values),
            "mean_cosine": sum(values) / len(values),
            "negative_fraction": sum(value < 0.0 for value in values)
            / len(values),
        }
        for label, values in sorted(cosines.items())
    }


def _parent_gate_uncertainty(
    parent_gate: dict[str, Any],
) -> dict[str, Any]:
    pair_points = []
    pair_margins = []
    seat_points = {0: [], 1: []}
    for row in parent_gate["rows"]:
        first = row["first"]
        second = row["second"]
        first_point = (
            0.5
            if first["winner"] is None
            else 1.0
            if int(first["winner"]) == 0
            else 0.0
        )
        second_point = (
            0.5
            if second["winner"] is None
            else 1.0
            if int(second["winner"]) == 1
            else 0.0
        )
        pair_points.append((first_point + second_point) / 2.0)
        seat_points[0].append(first_point)
        seat_points[1].append(second_point)
        pair_margins.append(
            (
                int(first["final_scores"][0])
                - int(first["final_scores"][1])
                + int(second["final_scores"][1])
                - int(second["final_scores"][0])
            )
            / 2.0
        )
    return {
        "score_share": _bootstrap_mean(
            pair_points,
            samples=20_000,
            seed=630_064,
        ),
        "paired_margin": _bootstrap_mean(
            pair_margins,
            samples=20_000,
            seed=630_065,
        ),
        "seat0_score_share": sum(seat_points[0]) / len(seat_points[0]),
        "seat1_score_share": sum(seat_points[1]) / len(seat_points[1]),
    }


def _split_half(
    records: list[dict[str, Any]],
    game_metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    halves = {"even": [], "odd": []}
    for row in records:
        ordinal = int(game_metadata[int(row["game_id"])]["cell_ordinal"])
        halves["even" if ordinal % 2 == 0 else "odd"].append(row)
    return {
        label: _summarize_records(rows)
        for label, rows in halves.items()
    }


def _leave_one_cell_out(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    cells = sorted({str(row["cell"]) for row in records})
    return {
        cell: _summarize_records(
            [row for row in records if str(row["cell"]) != cell]
        )
        for cell in cells
    }


def run(
    *,
    output_path: Path,
    result_062_path: Path,
    epoch0_path: Path,
    generation1_path: Path,
    generation2_path: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    prerequisite_paths = {
        "result_062": (result_062_path, RESULT_062_SHA256),
        "epoch0": (epoch0_path, EPOCH0_SHA256),
        "generation1": (generation1_path, GENERATION1_SHA256),
        "generation2": (generation2_path, GENERATION2_SHA256),
    }
    prerequisites = {
        label: {
            "path": str(path).replace("\\", "/"),
            "sha256": sha256(path),
            "expected_sha256": expected,
            "passed": sha256(path) == expected,
        }
        for label, (path, expected) in prerequisite_paths.items()
    }
    if not all(row["passed"] for row in prerequisites.values()):
        payload = {
            "experiment": "063",
            "prerequisites": prerequisites,
            "decision": "generation2_diagnosis_invalid_corpus_not_reproduced",
        }
        _atomic_json(output_path, payload)
        return payload

    original = json.loads(result_062_path.read_text(encoding="utf-8"))
    config = RecurrentLeagueConfig(**original["config"])
    epoch0 = load_recurrent_policy_checkpoint(epoch0_path).eval()
    generation1 = load_recurrent_policy_checkpoint(generation1_path).eval()
    generation2 = load_recurrent_policy_checkpoint(generation2_path).eval()
    static = static_opponent_factories(
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    league = [
        *static,
        ("snapshot_g1", recurrent_snapshot_factory(generation1)),
    ]
    games, steps, collection = collect_recurrent_league(
        generation1,
        league,
        generation=2,
        seed_start=621_000,
        config=config,
    )
    original_generation = original["generations"][1]
    collection_exact = _canonical(asdict(collection)) == _canonical(
        original_generation["collection"]
    )
    games_exact = _canonical([game.summary() for game in games]) == _canonical(
        original_generation["games"]
    )
    behavior_error = _behavior_reconstruction_error(
        generation1,
        steps,
        config,
    )
    integrity = {
        "prerequisites": all(row["passed"] for row in prerequisites.values()),
        "collection_summary_exact": collection_exact,
        "game_summaries_exact": games_exact,
        "eligible_games_exact": len(games) == 240,
        "behavior_reconstruction": behavior_error <= 1e-5,
        "no_training": True,
    }
    if not all(integrity.values()):
        payload = {
            "experiment": "063",
            "prerequisites": prerequisites,
            "integrity": integrity,
            "behavior_reconstruction_max_error": behavior_error,
            "decision": "generation2_diagnosis_invalid_corpus_not_reproduced",
        }
        _atomic_json(output_path, payload)
        return payload

    train_steps = [step for step in steps if step.split == "train"]
    validation_steps = [step for step in steps if step.split == "validation"]
    train_records = _raw_policy_records(
        generation1,
        generation2,
        train_steps,
        config,
    )
    validation_records = _raw_policy_records(
        generation1,
        generation2,
        validation_steps,
        config,
    )
    game_metadata = {
        game.game_id: game.summary()
        for game in games
    }
    validation_game_values = list(_game_values(validation_records).values())
    localization = {
        "train": _summarize_records(train_records),
        "validation": _summarize_records(validation_records),
        "validation_bootstrap": _bootstrap_mean(
            validation_game_values,
            samples=20_000,
            seed=630_063,
        ),
        "validation_split_half": _split_half(
            validation_records,
            game_metadata,
        ),
        "validation_leave_one_cell_out": _leave_one_cell_out(
            validation_records
        ),
        "validation_by_opponent": _breakdown(
            validation_records,
            "opponent",
        ),
        "validation_by_seat": _breakdown(validation_records, "seat"),
        "validation_by_cell": _breakdown(validation_records, "cell"),
        "validation_by_phase": _breakdown(validation_records, "phase"),
        "validation_by_action_type": _breakdown(
            validation_records,
            "action_type",
        ),
    }

    previous_direction_records = _raw_policy_records(
        epoch0,
        generation1,
        validation_steps,
        config,
    )
    previous_direction = {
        "overall": _summarize_records(previous_direction_records),
        "by_phase": _breakdown(previous_direction_records, "phase"),
        "by_cell": _breakdown(previous_direction_records, "cell"),
    }
    parameter_direction = _parameter_direction(
        epoch0,
        generation1,
        generation2,
    )
    logit_direction = _logit_direction(
        epoch0,
        generation1,
        generation2,
        validation_steps,
        config,
    )

    empty_steps = _empty_history_steps(validation_steps)
    empty_records = _raw_policy_records(
        generation1,
        generation2,
        empty_steps,
        config,
    )
    real_summary = _summarize_records(validation_records)
    empty_summary = _summarize_records(empty_records)
    real_ratios = [float(row["log_ratio"]) for row in validation_records]
    empty_ratios = [float(row["log_ratio"]) for row in empty_records]
    ratio_correlation = _pearson(real_ratios, empty_ratios)
    ratio_mae = sum(
        abs(real - empty)
        for real, empty in zip(real_ratios, empty_ratios)
    ) / len(real_ratios)
    history_ablation = {
        "real_history": real_summary,
        "empty_history": empty_summary,
        "chosen_log_ratio_correlation": ratio_correlation,
        "chosen_log_ratio_mae": ratio_mae,
        "movement_removed_fraction": (
            1.0
            - float(empty_summary["raw_kl"])
            / max(float(real_summary["raw_kl"]), 1e-12)
        ),
    }

    gate_uncertainty = _parent_gate_uncertainty(
        original_generation["parent_gate"]
    )
    validation_ci = localization["validation_bootstrap"]
    score_ci = gate_uncertainty["score_share"]
    flags = {
        "train_validation_reversal": (
            float(localization["train"]["surrogate"]) > 0.0
            and float(localization["validation"]["surrogate"]) <= 0.0
        ),
        "validation_sign_uncertain": (
            float(validation_ci["lower"]) <= 0.0
            <= float(validation_ci["upper"])
        ),
        "validation_systematically_negative": float(validation_ci["upper"])
        < 0.0,
        "direction_reversal": (
            float(parameter_direction["all"]["cosine"]) < 0.0
            or float(logit_direction["overall"]["mean_cosine"]) < 0.0
        ),
        "history_material": (
            float(history_ablation["movement_removed_fraction"]) >= 0.25
            or ratio_correlation < 0.80
        ),
        "history_mostly_unused": (
            float(empty_summary["raw_kl"])
            >= 0.80 * float(real_summary["raw_kl"])
            and ratio_correlation >= 0.90
        ),
        "parent_gate_uncertain": (
            float(score_ci["lower"]) <= 0.50 <= float(score_ci["upper"])
        ),
        "parent_gate_systematically_worse": float(score_ci["upper"]) < 0.50,
    }
    payload = {
        "experiment": "063",
        "protocol": (
            "docs/experiments/recurrent_generation2_diagnosis_063.md"
        ),
        "prerequisites": prerequisites,
        "integrity": integrity,
        "behavior_reconstruction_max_error": behavior_error,
        "collection": asdict(collection),
        "localization": localization,
        "previous_direction_on_generation2_validation": previous_direction,
        "parameter_direction": parameter_direction,
        "logit_direction": logit_direction,
        "history_ablation": history_ablation,
        "existing_parent_gate_uncertainty": gate_uncertainty,
        "interpretation_flags": flags,
        "decision": "generation2_failure_diagnostic_complete_no_scale_authorized",
    }
    _atomic_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "outputs/recurrent_generation2_diagnosis_063/result.json"
        ),
    )
    parser.add_argument(
        "--result-062",
        type=Path,
        default=Path("outputs/recurrent_league_stage_b_062/result.json"),
    )
    parser.add_argument(
        "--epoch0",
        type=Path,
        default=Path(
            "outputs/recurrent_league_stage_a_061/recurrent_policy_epoch0.pt"
        ),
    )
    parser.add_argument(
        "--generation1",
        type=Path,
        default=Path(
            "outputs/recurrent_league_stage_b_062/generation_1.pt"
        ),
    )
    parser.add_argument(
        "--generation2",
        type=Path,
        default=Path(
            "outputs/recurrent_league_stage_b_062/generation_2.pt"
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
        output_path=args.output_path,
        result_062_path=args.result_062,
        epoch0_path=args.epoch0,
        generation1_path=args.generation1,
        generation2_path=args.generation2,
        checkpoint_035=args.checkpoint_035,
        checkpoint_016=args.checkpoint_016,
        checkpoint_023=args.checkpoint_023,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "integrity": result.get("integrity"),
                "flags": result.get("interpretation_flags"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
