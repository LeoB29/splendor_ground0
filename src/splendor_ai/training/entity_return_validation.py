"""Empirical validation for denser canonical-entity actor return targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

from splendor_ai.training.entity_actor_critic import ActorStep


@dataclass(frozen=True, slots=True)
class GameReturnRecord:
    game_index: int
    opponent_label: str
    actor_seat: int
    outcome: float
    score_margin: float
    normalized_score_margin: float


@dataclass(frozen=True, slots=True)
class ReturnTargetSummary:
    games: int
    outcome_unique_count: int
    score_margin_unique_count: int
    normalized_score_margin_unique_count: int
    outcome_score_margin_correlation: float
    outcome_sign_alignment: float
    zero_margin_non_draws: int
    saturation_fraction: float
    score_margin_mean: float
    score_margin_std: float
    normalized_score_margin_mean: float
    normalized_score_margin_std: float
    all_loss_strata: int
    all_loss_strata_with_margin_variation: int
    strata: dict[str, dict[str, float | int | bool]]


def game_return_records(steps: list[ActorStep] | tuple[ActorStep, ...]) -> list[GameReturnRecord]:
    records: list[GameReturnRecord] = []
    seen_games: set[int] = set()
    for step in steps:
        if step.game_index in seen_games:
            continue
        seen_games.add(step.game_index)
        records.append(
            GameReturnRecord(
                game_index=step.game_index,
                opponent_label=step.opponent_label,
                actor_seat=step.actor_seat,
                outcome=step.final_return,
                score_margin=step.final_score_margin,
                normalized_score_margin=step.score_margin_return,
            )
        )
    return records


def summarize_return_targets(
    records: list[GameReturnRecord],
    *,
    minimum_all_loss_margin_unique: int = 3,
    minimum_all_loss_normalized_std: float = 0.03,
) -> ReturnTargetSummary:
    if not records:
        raise ValueError("At least one game return record is required.")
    outcomes = [record.outcome for record in records]
    margins = [record.score_margin for record in records]
    normalized_margins = [record.normalized_score_margin for record in records]
    non_draws = [record for record in records if record.outcome != 0.0]
    aligned = sum(
        (record.outcome > 0.0 and record.score_margin > 0.0)
        or (record.outcome < 0.0 and record.score_margin < 0.0)
        for record in non_draws
    )
    zero_margin_non_draws = sum(record.score_margin == 0.0 for record in non_draws)
    saturation_fraction = sum(abs(value) >= 1.0 - 1e-12 for value in normalized_margins) / len(
        normalized_margins
    )

    by_stratum: dict[tuple[str, int], list[GameReturnRecord]] = {}
    for record in records:
        by_stratum.setdefault((record.opponent_label, record.actor_seat), []).append(record)
    strata: dict[str, dict[str, float | int | bool]] = {}
    all_loss_strata = 0
    varying_all_loss_strata = 0
    for (opponent_label, actor_seat), stratum_records in sorted(by_stratum.items()):
        stratum_outcomes = [record.outcome for record in stratum_records]
        stratum_margins = [record.score_margin for record in stratum_records]
        stratum_normalized = [record.normalized_score_margin for record in stratum_records]
        opponent_wins = sum(value < 0.0 for value in stratum_outcomes)
        all_losses = opponent_wins == len(stratum_records)
        normalized_std = _population_std(stratum_normalized)
        margin_unique_count = len(set(stratum_margins))
        margin_varies = (
            margin_unique_count >= minimum_all_loss_margin_unique
            and normalized_std >= minimum_all_loss_normalized_std
        )
        if all_losses:
            all_loss_strata += 1
            varying_all_loss_strata += margin_varies
        strata[f"{opponent_label}:seat{actor_seat}"] = {
            "games": len(stratum_records),
            "actor_wins": sum(value > 0.0 for value in stratum_outcomes),
            "opponent_wins": opponent_wins,
            "draws": sum(value == 0.0 for value in stratum_outcomes),
            "outcome_unique_count": len(set(stratum_outcomes)),
            "score_margin_unique_count": margin_unique_count,
            "score_margin_mean": sum(stratum_margins) / len(stratum_margins),
            "score_margin_std": _population_std(stratum_margins),
            "score_margin_min": min(stratum_margins),
            "score_margin_max": max(stratum_margins),
            "normalized_score_margin_std": normalized_std,
            "all_losses": all_losses,
            "all_loss_margin_varies": all_losses and margin_varies,
        }
    return ReturnTargetSummary(
        games=len(records),
        outcome_unique_count=len(set(outcomes)),
        score_margin_unique_count=len(set(margins)),
        normalized_score_margin_unique_count=len(set(normalized_margins)),
        outcome_score_margin_correlation=_correlation(outcomes, margins),
        outcome_sign_alignment=aligned / len(non_draws) if non_draws else 1.0,
        zero_margin_non_draws=zero_margin_non_draws,
        saturation_fraction=saturation_fraction,
        score_margin_mean=sum(margins) / len(margins),
        score_margin_std=_population_std(margins),
        normalized_score_margin_mean=sum(normalized_margins) / len(normalized_margins),
        normalized_score_margin_std=_population_std(normalized_margins),
        all_loss_strata=all_loss_strata,
        all_loss_strata_with_margin_variation=varying_all_loss_strata,
        strata=strata,
    )


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator
