"""Diagnose the failed 064 broad gate without training or fresh seeds."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable

import torch

from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import GameResult, play_game
from splendor_ai.training.recurrent_league import (
    InMemoryRecurrentBot,
    RecurrentLeagueConfig,
    VALID_TERMINATIONS,
    forward_recurrent,
    sha256,
    static_opponent_factories,
)
from splendor_ai.training.recurrent_policy import (
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
)


RESULT_064_SHA256 = (
    "7a3094d37e6d37961b44f74029330c7083f5da3af7a9a03f8050b62bc59cf2e6"
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
SEED_STARTS = {
    "035": 643_000,
    "016": 643_100,
    "023": 643_200,
    "shallow": 643_300,
}
OPPONENT_ORDER = ("035", "016", "023", "shallow")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _phase(turn: int) -> str:
    if turn < 20:
        return "early"
    if turn < 50:
        return "mid"
    return "late"


def _point(result: dict[str, Any], seat: int) -> float:
    winner = result["winner"]
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == seat else 0.0


def _margin(result: dict[str, Any], seat: int) -> float:
    scores = result["final_scores"]
    return float(int(scores[seat]) - int(scores[1 - seat]))


def _normalize_result(result: GameResult | dict[str, Any]) -> dict[str, Any]:
    payload = asdict(result) if isinstance(result, GameResult) else dict(result)
    payload["bot_seats"] = [
        "InMemoryRecurrentBot" if name == "TracingRecurrentBot" else name
        for name in payload["bot_seats"]
    ]
    payload["final_scores"] = list(payload["final_scores"])
    payload["loop_fallback_triggers_by_seat"] = list(
        payload["loop_fallback_triggers_by_seat"]
    )
    return payload


def _percentile(sorted_values: list[float], probability: float) -> float:
    index = min(int(probability * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def _bootstrap_mean(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    rng = random.Random(seed)
    draws = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    ]
    draws.sort()
    return {
        "mean": sum(values) / len(values),
        "lower": _percentile(draws, 0.025),
        "upper": _percentile(draws, 0.975),
        "samples": samples,
    }


def _stratified_bootstrap(
    values_by_stratum: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if not values_by_stratum or any(not rows for rows in values_by_stratum.values()):
        raise ValueError("Every bootstrap stratum must be nonempty")
    labels = sorted(values_by_stratum)
    total = sum(len(values_by_stratum[label]) for label in labels)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        value = 0.0
        for label in labels:
            rows = values_by_stratum[label]
            value += sum(rows[rng.randrange(len(rows))] for _ in rows)
        draws.append(value / total)
    draws.sort()
    observed = sum(sum(values_by_stratum[label]) for label in labels) / total
    return {
        "mean": observed,
        "lower": _percentile(draws, 0.025),
        "upper": _percentile(draws, 0.975),
        "samples": samples,
        "strata": len(labels),
    }


def _sign_flip_test(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("sign-flip values must not be empty")
    observed = sum(values) / len(values)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        permuted = sum(
            value if rng.randrange(2) else -value
            for value in values
        ) / len(values)
        extreme += abs(permuted) >= abs(observed) - 1e-15
    return {
        "mean": observed,
        "two_sided_p": (extreme + 1) / (samples + 1),
        "samples": samples,
    }


def _paired_seed_records(result_064: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    opponents = result_064["final_population_gate"]["opponents"]
    for opponent in OPPONENT_ORDER:
        rows = opponents[opponent]["rows"]
        for ordinal, row in enumerate(rows):
            candidate = row["candidate"]
            control = row["control"]
            candidate_points = [_point(candidate[seat], seat) for seat in (0, 1)]
            control_points = [_point(control[seat], seat) for seat in (0, 1)]
            candidate_margins = [_margin(candidate[seat], seat) for seat in (0, 1)]
            control_margins = [_margin(control[seat], seat) for seat in (0, 1)]
            records.append(
                {
                    "opponent": opponent,
                    "seed": int(row["seed"]),
                    "seed_ordinal": ordinal,
                    "candidate_score_share": sum(candidate_points) / 2.0,
                    "control_score_share": sum(control_points) / 2.0,
                    "score_share_delta": (
                        sum(candidate_points) - sum(control_points)
                    ) / 2.0,
                    "candidate_mean_margin": sum(candidate_margins) / 2.0,
                    "control_mean_margin": sum(control_margins) / 2.0,
                    "margin_delta": (
                        sum(candidate_margins) - sum(control_margins)
                    ) / 2.0,
                    "seat0_score_delta": candidate_points[0] - control_points[0],
                    "seat1_score_delta": candidate_points[1] - control_points[1],
                    "seat0_margin_delta": candidate_margins[0] - control_margins[0],
                    "seat1_margin_delta": candidate_margins[1] - control_margins[1],
                }
            )
    return records


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _paired_breakdown(
    records: list[dict[str, Any]],
    *,
    aggregate_bootstrap_seed: int = 650_065,
    opponent_bootstrap_seed: int = 650_100,
    score_sign_flip_seed: int = 650_165,
    margin_sign_flip_seed: int = 650_166,
) -> dict[str, Any]:
    by_opponent = {
        opponent: [row for row in records if row["opponent"] == opponent]
        for opponent in OPPONENT_ORDER
    }
    opponent_summary = {}
    for index, opponent in enumerate(OPPONENT_ORDER):
        rows = by_opponent[opponent]
        opponent_summary[opponent] = {
            "seeds": len(rows),
            "score_share_delta": _bootstrap_mean(
                [float(row["score_share_delta"]) for row in rows],
                samples=50_000,
                seed=opponent_bootstrap_seed + index,
            ),
            "margin_delta": _bootstrap_mean(
                [float(row["margin_delta"]) for row in rows],
                samples=50_000,
                seed=opponent_bootstrap_seed + index,
            ),
        }
    score_by_opponent = {
        label: [float(row["score_share_delta"]) for row in rows]
        for label, rows in by_opponent.items()
    }
    margin_by_opponent = {
        label: [float(row["margin_delta"]) for row in rows]
        for label, rows in by_opponent.items()
    }
    split_halves = {}
    for label, parity in (("even", 0), ("odd", 1)):
        rows = [row for row in records if int(row["seed_ordinal"]) % 2 == parity]
        split_halves[label] = {
            "seeds": len(rows),
            "score_share_delta": _mean(rows, "score_share_delta"),
            "margin_delta": _mean(rows, "margin_delta"),
        }
    leave_one_opponent_out = {}
    for opponent in OPPONENT_ORDER:
        rows = [row for row in records if row["opponent"] != opponent]
        leave_one_opponent_out[opponent] = {
            "seeds": len(rows),
            "score_share_delta": _mean(rows, "score_share_delta"),
            "margin_delta": _mean(rows, "margin_delta"),
        }
    leave_one_seed_score = []
    leave_one_seed_margin = []
    for removed in records:
        rows = [row for row in records if row is not removed]
        leave_one_seed_score.append(_mean(rows, "score_share_delta"))
        leave_one_seed_margin.append(_mean(rows, "margin_delta"))
    return {
        "aggregate": {
            "seeds": len(records),
            "score_share_delta": _stratified_bootstrap(
                score_by_opponent,
                samples=100_000,
                seed=aggregate_bootstrap_seed,
            ),
            "margin_delta": _stratified_bootstrap(
                margin_by_opponent,
                samples=100_000,
                seed=aggregate_bootstrap_seed,
            ),
            "score_sign_flip": _sign_flip_test(
                [float(row["score_share_delta"]) for row in records],
                samples=100_000,
                seed=score_sign_flip_seed,
            ),
            "margin_sign_flip": _sign_flip_test(
                [float(row["margin_delta"]) for row in records],
                samples=100_000,
                seed=margin_sign_flip_seed,
            ),
        },
        "by_opponent": opponent_summary,
        "by_seat": {
            f"seat{seat}": {
                "score_share_delta": _mean(records, f"seat{seat}_score_delta"),
                "margin_delta": _mean(records, f"seat{seat}_margin_delta"),
            }
            for seat in (0, 1)
        },
        "split_halves": split_halves,
        "leave_one_opponent_out": leave_one_opponent_out,
        "leave_one_seed_out": {
            "score_share_delta_min": min(leave_one_seed_score),
            "score_share_delta_max": max(leave_one_seed_score),
            "margin_delta_min": min(leave_one_seed_margin),
            "margin_delta_max": max(leave_one_seed_margin),
        },
    }


class TracingRecurrentBot(InMemoryRecurrentBot):
    """Exact recurrent actor that also compares generation 2 with epoch zero."""

    def __init__(
        self,
        model,
        *,
        candidate,
        control,
        acting_label: str,
        opponent: str,
        recurrent_seat: int,
        seed: int,
        records: list[dict[str, Any]],
    ) -> None:
        super().__init__(model)
        self.candidate = candidate
        self.control = control
        self.acting_label = acting_label
        self.opponent = opponent
        self.recurrent_seat = recurrent_seat
        self.seed = seed
        self.records = records

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        policy_input = build_recurrent_policy_input(
            state,
            actions,
            self.history.events,
        )
        batch = collate_recurrent_policy_inputs([policy_input])
        with torch.no_grad():
            candidate_scores, _ = forward_recurrent(self.candidate, batch)
            control_scores, _ = forward_recurrent(self.control, batch)
        width = len(actions)
        candidate_row = int(torch.argmax(candidate_scores[0, :width]).item())
        control_row = int(torch.argmax(control_scores[0, :width]).item())
        candidate_probability = torch.softmax(candidate_scores[0, :width], dim=0)
        control_probability = torch.softmax(control_scores[0, :width], dim=0)
        total_variation = float(
            (0.5 * torch.abs(candidate_probability - control_probability).sum()).cpu()
        )
        acting_row = candidate_row if self.acting_label == "candidate" else control_row
        self.records.append(
            {
                "opponent": self.opponent,
                "seed": self.seed,
                "seed_ordinal": self.seed - SEED_STARTS[self.opponent],
                "trajectory_source": self.acting_label,
                "recurrent_seat": self.recurrent_seat,
                "game_key": (
                    f"{self.opponent}:{self.seed}:{self.acting_label}:"
                    f"seat{self.recurrent_seat}"
                ),
                "turn": state.turn_index,
                "phase": _phase(state.turn_index),
                "history_length": len(policy_input.history.action_type_ids),
                "legal_actions": width,
                "candidate_action_row": candidate_row,
                "control_action_row": control_row,
                "candidate_action_type": actions[candidate_row].action_type.name,
                "control_action_type": actions[control_row].action_type.name,
                "acting_action_row": acting_row,
                "greedy_disagreement": candidate_row != control_row,
                "total_variation": total_variation,
            }
        )
        return actions[acting_row]


def _trace_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "decisions": 0,
            "games": 0,
            "disagreements": 0,
            "decision_disagreement_rate": 0.0,
            "game_balanced_disagreement_rate": 0.0,
            "decision_mean_total_variation": 0.0,
            "game_balanced_mean_total_variation": 0.0,
        }
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_game[str(row["game_key"])].append(row)
    game_disagreement = [
        sum(bool(row["greedy_disagreement"]) for row in rows) / len(rows)
        for rows in by_game.values()
    ]
    game_tv = [
        sum(float(row["total_variation"]) for row in rows) / len(rows)
        for rows in by_game.values()
    ]
    return {
        "decisions": len(records),
        "games": len(by_game),
        "disagreements": sum(bool(row["greedy_disagreement"]) for row in records),
        "decision_disagreement_rate": sum(
            bool(row["greedy_disagreement"]) for row in records
        ) / len(records),
        "game_balanced_disagreement_rate": sum(game_disagreement)
        / len(game_disagreement),
        "decision_mean_total_variation": sum(
            float(row["total_variation"]) for row in records
        ) / len(records),
        "game_balanced_mean_total_variation": sum(game_tv) / len(game_tv),
    }


def _trace_breakdown(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {label: _trace_summary(rows) for label, rows in sorted(groups.items())}


def _transitions(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        if row["greedy_disagreement"]:
            key = f"{row['control_action_type']}->{row['candidate_action_type']}"
            counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _movement_concentrated(
    breakdown: dict[str, dict[str, float | int]],
) -> bool:
    values = [
        float(row["game_balanced_disagreement_rate"])
        for row in breakdown.values()
    ]
    if len(values) < 2:
        return False
    minimum = min(values)
    maximum = max(values)
    return maximum >= 2.0 * minimum and maximum - minimum >= 0.02


def _replay(
    result_064: dict[str, Any],
    candidate,
    control,
    opponent_factories: list[tuple[str, Callable[[], Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    factories = dict(opponent_factories)
    trace_records: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    mismatches = []
    valid = True
    config = RecurrentLeagueConfig()
    stored_opponents = result_064["final_population_gate"]["opponents"]
    for opponent in OPPONENT_ORDER:
        stored_rows = stored_opponents[opponent]["rows"]
        for ordinal, stored in enumerate(stored_rows):
            seed = SEED_STARTS[opponent] + ordinal
            if int(stored["seed"]) != seed:
                mismatches.append(f"{opponent}:{seed}:stored_seed")
            actual: dict[str, list[dict[str, Any]]] = {}
            for acting_label, model in (("candidate", candidate), ("control", control)):
                first_bot = TracingRecurrentBot(
                    model,
                    candidate=candidate,
                    control=control,
                    acting_label=acting_label,
                    opponent=opponent,
                    recurrent_seat=0,
                    seed=seed,
                    records=trace_records,
                )
                first = play_game(
                    first_bot,
                    factories[opponent](),
                    seed=seed,
                    max_turns=config.max_turns,
                    repetition_limit=config.repetition_limit,
                    no_progress_limit=config.no_progress_limit,
                )
                second_bot = TracingRecurrentBot(
                    model,
                    candidate=candidate,
                    control=control,
                    acting_label=acting_label,
                    opponent=opponent,
                    recurrent_seat=1,
                    seed=seed,
                    records=trace_records,
                )
                second = play_game(
                    factories[opponent](),
                    second_bot,
                    seed=seed,
                    max_turns=config.max_turns,
                    repetition_limit=config.repetition_limit,
                    no_progress_limit=config.no_progress_limit,
                )
                normalized = [_normalize_result(first), _normalize_result(second)]
                actual[acting_label] = normalized
                for result in normalized:
                    valid &= result["termination_reason"] in VALID_TERMINATIONS
                if normalized != stored[acting_label]:
                    mismatches.append(f"{opponent}:{seed}:{acting_label}")
            replay_rows.append(
                {
                    "opponent": opponent,
                    "seed": seed,
                    "candidate": actual["candidate"],
                    "control": actual["control"],
                }
            )
    return replay_rows, trace_records, {
        "games": 4 * len(replay_rows),
        "valid": valid,
        "outcomes_exact": not mismatches,
        "mismatches": mismatches,
    }


def _recomputed_population(replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    opponents = {}
    total_candidate = 0.0
    total_control = 0.0
    for opponent in OPPONENT_ORDER:
        rows = [row for row in replay_rows if row["opponent"] == opponent]
        candidate = sum(
            _point(row["candidate"][seat], seat)
            for row in rows
            for seat in (0, 1)
        )
        control = sum(
            _point(row["control"][seat], seat)
            for row in rows
            for seat in (0, 1)
        )
        games = 2 * len(rows)
        opponents[opponent] = {
            "candidate_score_share": candidate / games,
            "control_score_share": control / games,
            "delta": (candidate - control) / games,
        }
        total_candidate += candidate
        total_control += control
    total_games_per_model = 2 * len(replay_rows)
    return {
        "games": 2 * total_games_per_model,
        "aggregate_candidate_score_share": total_candidate / total_games_per_model,
        "aggregate_control_score_share": total_control / total_games_per_model,
        "aggregate_delta": (total_candidate - total_control) / total_games_per_model,
        "opponents": opponents,
    }


def _population_matches(
    recomputed: dict[str, Any],
    stored: dict[str, Any],
) -> bool:
    if recomputed["games"] != stored["games"]:
        return False
    for key in (
        "aggregate_candidate_score_share",
        "aggregate_control_score_share",
        "aggregate_delta",
    ):
        if not math.isclose(float(recomputed[key]), float(stored[key]), abs_tol=1e-12):
            return False
    for opponent in OPPONENT_ORDER:
        for key in ("candidate_score_share", "control_score_share", "delta"):
            if not math.isclose(
                float(recomputed["opponents"][opponent][key]),
                float(stored["opponents"][opponent][key]),
                abs_tol=1e-12,
            ):
                return False
    return True


def _interpretation(
    uncertainty: dict[str, Any],
    localization: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    aggregate_score = uncertainty["aggregate"]["score_share_delta"]
    aggregate_margin = uncertainty["aggregate"]["margin_delta"]
    by_opponent = uncertainty["by_opponent"]
    flags = {
        "aggregate_score_uncertain": (
            float(aggregate_score["lower"]) <= 0.0 <= float(aggregate_score["upper"])
        ),
        "aggregate_score_systematically_negative": float(aggregate_score["upper"])
        < 0.0,
        "aggregate_margin_systematically_negative": float(aggregate_margin["upper"])
        < 0.0,
        "016_loss_systematic": float(
            by_opponent["016"]["score_share_delta"]["upper"]
        ) < 0.0,
        "023_loss_systematic": float(
            by_opponent["023"]["score_share_delta"]["upper"]
        ) < 0.0,
        "policy_movement_opponent_concentrated": _movement_concentrated(
            localization["by_opponent"]
        ),
        "policy_movement_phase_concentrated": _movement_concentrated(
            localization["by_phase"]
        ),
    }
    flags["losses_localized"] = bool(
        flags["aggregate_score_uncertain"]
        and (flags["016_loss_systematic"] or flags["023_loss_systematic"])
    )
    if (
        flags["aggregate_score_systematically_negative"]
        or flags["aggregate_margin_systematically_negative"]
    ):
        decision = "recurrent_broad_regression_systematic_change_objective"
    elif flags["losses_localized"]:
        decision = "recurrent_broad_regression_localized_change_objective"
    elif flags["aggregate_score_uncertain"]:
        decision = (
            "recurrent_broad_result_inconclusive_"
            "authorize_confirmation_preregistration"
        )
    else:
        decision = "recurrent_broad_diagnosis_complete_no_scale_authorized"
    return flags, decision


def run(
    *,
    output_path: Path,
    result_064_path: Path,
    epoch0_path: Path,
    generation2_path: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    expected = {
        "result_064": (result_064_path, RESULT_064_SHA256),
        "epoch0": (epoch0_path, EPOCH0_SHA256),
        "generation2": (generation2_path, GENERATION2_SHA256),
        "035": (checkpoint_035, CHECKPOINT_035_SHA256),
        "016": (checkpoint_016, CHECKPOINT_016_SHA256),
        "023": (checkpoint_023, CHECKPOINT_023_SHA256),
    }
    prerequisites = {}
    for label, (path, digest) in expected.items():
        actual = sha256(path)
        prerequisites[label] = {
            "path": str(path).replace("\\", "/"),
            "sha256": actual,
            "expected_sha256": digest,
            "passed": actual == digest,
        }
    if not all(row["passed"] for row in prerequisites.values()):
        payload = {
            "experiment": "065",
            "prerequisites": prerequisites,
            "decision": "recurrent_broad_diagnosis_invalid_replay_mismatch",
        }
        _atomic_json(output_path, payload)
        return payload

    result_064 = json.loads(result_064_path.read_text(encoding="utf-8"))
    false_final_gates = sorted(
        name for name, passed in result_064["final_gates"].items() if not passed
    )
    result_contract = {
        "decision_exact": result_064.get("decision")
        == "history_required_repair_failed_diagnose_before_scale",
        "two_generations_accepted": result_064.get("generation_count_accepted") == 2,
        "final_population_valid": bool(result_064["final_population_gate"]["valid"]),
        "final_population_games": result_064["final_population_gate"]["games"] == 160,
        "only_aggregate_gate_failed": false_final_gates
        == ["aggregate_delta_positive"],
    }
    if not all(result_contract.values()):
        payload = {
            "experiment": "065",
            "prerequisites": prerequisites,
            "result_contract": result_contract,
            "decision": "recurrent_broad_diagnosis_invalid_replay_mismatch",
        }
        _atomic_json(output_path, payload)
        return payload

    candidate = load_recurrent_policy_checkpoint(generation2_path).eval()
    control = load_recurrent_policy_checkpoint(epoch0_path).eval()
    opponents = static_opponent_factories(
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    replay_rows, trace_records, replay_integrity = _replay(
        result_064,
        candidate,
        control,
        opponents,
    )
    recomputed = _recomputed_population(replay_rows)
    replay_integrity["population_metrics_exact"] = _population_matches(
        recomputed,
        result_064["final_population_gate"],
    )
    replay_integrity["no_training"] = True
    replay_integrity["no_new_seeds"] = all(
        int(row["seed"]) == SEED_STARTS[str(row["opponent"])] + ordinal
        for opponent in OPPONENT_ORDER
        for ordinal, row in enumerate(
            [item for item in replay_rows if item["opponent"] == opponent]
        )
    )
    replay_passed = (
        replay_integrity["games"] == 160
        and replay_integrity["valid"]
        and replay_integrity["outcomes_exact"]
        and replay_integrity["population_metrics_exact"]
        and replay_integrity["no_training"]
        and replay_integrity["no_new_seeds"]
        and not replay_integrity["mismatches"]
    )
    replay_integrity["passed"] = replay_passed
    if not replay_passed:
        payload = {
            "experiment": "065",
            "protocol": "docs/experiments/recurrent_broad_failure_diagnosis_065.md",
            "prerequisites": prerequisites,
            "result_contract": result_contract,
            "replay_integrity": replay_integrity,
            "recomputed_population": recomputed,
            "decision": "recurrent_broad_diagnosis_invalid_replay_mismatch",
        }
        _atomic_json(output_path, payload)
        return payload

    paired_records = _paired_seed_records(result_064)
    paired_by_key = {
        (str(row["opponent"]), int(row["seed"])): row
        for row in paired_records
    }
    for row in trace_records:
        paired = paired_by_key[(str(row["opponent"]), int(row["seed"]))]
        score_delta = float(paired["score_share_delta"])
        row["paired_score_delta"] = score_delta
        row["paired_margin_delta"] = float(paired["margin_delta"])
        row["paired_outcome_sign"] = (
            "negative" if score_delta < 0.0 else "positive" if score_delta > 0.0 else "zero"
        )
    uncertainty = _paired_breakdown(paired_records)
    localization = {
        "overall": _trace_summary(trace_records),
        "by_opponent": _trace_breakdown(trace_records, "opponent"),
        "by_seat": _trace_breakdown(trace_records, "recurrent_seat"),
        "by_phase": _trace_breakdown(trace_records, "phase"),
        "by_trajectory_source": _trace_breakdown(
            trace_records,
            "trajectory_source",
        ),
        "by_paired_outcome_sign": _trace_breakdown(
            trace_records,
            "paired_outcome_sign",
        ),
        "all_disagreement_transitions": _transitions(trace_records),
        "loss_opponent_transitions": {
            opponent: _transitions(
                [row for row in trace_records if row["opponent"] == opponent]
            )
            for opponent in ("016", "023")
        },
    }
    flags, decision = _interpretation(uncertainty, localization)
    payload = {
        "experiment": "065",
        "protocol": "docs/experiments/recurrent_broad_failure_diagnosis_065.md",
        "prerequisites": prerequisites,
        "result_contract": result_contract,
        "replay_integrity": replay_integrity,
        "recomputed_population": recomputed,
        "paired_seed_records": paired_records,
        "paired_uncertainty": uncertainty,
        "policy_change_localization": localization,
        "interpretation_flags": flags,
        "decision": decision,
    }
    _atomic_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/recurrent_broad_failure_diagnosis_065/result.json"),
    )
    parser.add_argument(
        "--result-064",
        type=Path,
        default=Path("outputs/history_required_recurrent_repair_064/result.json"),
    )
    parser.add_argument(
        "--epoch0",
        type=Path,
        default=Path(
            "outputs/history_required_recurrent_repair_064/"
            "history_required_epoch0.pt"
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
        output_path=args.output_path,
        result_064_path=args.result_064,
        epoch0_path=args.epoch0,
        generation2_path=args.generation2,
        checkpoint_035=args.checkpoint_035,
        checkpoint_016=args.checkpoint_016,
        checkpoint_023=args.checkpoint_023,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "replay_games": result.get("replay_integrity", {}).get("games", 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
