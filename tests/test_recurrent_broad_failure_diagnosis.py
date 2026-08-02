from __future__ import annotations

from splendor_ai.eval.match import GameResult
from splendor_ai.training.run_recurrent_broad_failure_diagnosis import (
    OPPONENT_ORDER,
    _interpretation,
    _movement_concentrated,
    _normalize_result,
    _paired_seed_records,
    _stratified_bootstrap,
    _trace_summary,
)


def _game(winner, scores=(15, 10)):
    return {
        "winner": winner,
        "final_scores": list(scores),
    }


def test_paired_seed_records_keep_both_seats_clustered() -> None:
    opponents = {}
    for index, opponent in enumerate(OPPONENT_ORDER):
        opponents[opponent] = {
            "rows": [
                {
                    "seed": 100 + index,
                    "candidate": [_game(0), _game(1, (10, 15))],
                    "control": [_game(1), _game(0, (15, 10))],
                }
            ]
        }
    result = {"final_population_gate": {"opponents": opponents}}

    rows = _paired_seed_records(result)

    assert len(rows) == 4
    assert all(row["score_share_delta"] == 1.0 for row in rows)
    assert all(row["seat0_score_delta"] == 1.0 for row in rows)
    assert all(row["seat1_score_delta"] == 1.0 for row in rows)


def test_stratified_bootstrap_is_deterministic_and_preserves_mean() -> None:
    values = {"a": [1.0, -1.0], "b": [0.5, 0.5]}

    first = _stratified_bootstrap(values, samples=1_000, seed=65)
    second = _stratified_bootstrap(values, samples=1_000, seed=65)

    assert first == second
    assert first["mean"] == 0.25
    assert first["lower"] <= first["mean"] <= first["upper"]


def test_trace_summary_reports_decision_and_game_balanced_rates() -> None:
    rows = [
        {"game_key": "a", "greedy_disagreement": True, "total_variation": 0.2},
        {"game_key": "a", "greedy_disagreement": False, "total_variation": 0.0},
        {"game_key": "b", "greedy_disagreement": False, "total_variation": 0.1},
    ]

    summary = _trace_summary(rows)

    assert summary["decisions"] == 3
    assert summary["games"] == 2
    assert summary["decision_disagreement_rate"] == 1.0 / 3.0
    assert summary["game_balanced_disagreement_rate"] == 0.25
    assert abs(summary["game_balanced_mean_total_variation"] - 0.1) < 1e-12


def test_normalize_result_changes_only_tracing_class_identity() -> None:
    result = GameResult(
        seed=1,
        turns=2,
        winner=0,
        final_scores=(15, 10),
        bot_seats=("TracingRecurrentBot", "GreedyHeuristicBot"),
    )

    normalized = _normalize_result(result)

    assert normalized["bot_seats"] == [
        "InMemoryRecurrentBot",
        "GreedyHeuristicBot",
    ]
    assert normalized["final_scores"] == [15, 10]


def _uncertainty(score=(-0.1, -0.3, 0.2), margin=(-1.0, -3.0, 1.0)):
    def interval(values):
        return {"mean": values[0], "lower": values[1], "upper": values[2]}

    return {
        "aggregate": {
            "score_share_delta": interval(score),
            "margin_delta": interval(margin),
        },
        "by_opponent": {
            opponent: {"score_share_delta": interval((0.0, -0.5, 0.5))}
            for opponent in OPPONENT_ORDER
        },
    }


def _localization():
    quiet = {"game_balanced_disagreement_rate": 0.03}
    return {
        "by_opponent": {opponent: dict(quiet) for opponent in OPPONENT_ORDER},
        "by_phase": {phase: dict(quiet) for phase in ("early", "mid", "late")},
    }


def test_interpretation_authorizes_only_preregistering_confirmation_when_uncertain() -> None:
    flags, decision = _interpretation(_uncertainty(), _localization())

    assert flags["aggregate_score_uncertain"]
    assert not flags["aggregate_score_systematically_negative"]
    assert decision == (
        "recurrent_broad_result_inconclusive_"
        "authorize_confirmation_preregistration"
    )


def test_interpretation_prioritizes_systematic_aggregate_regression() -> None:
    uncertainty = _uncertainty(score=(-0.2, -0.4, -0.01))

    flags, decision = _interpretation(uncertainty, _localization())

    assert flags["aggregate_score_systematically_negative"]
    assert decision == "recurrent_broad_regression_systematic_change_objective"


def test_movement_concentration_requires_ratio_and_absolute_gap() -> None:
    concentrated = {
        "a": {"game_balanced_disagreement_rate": 0.02},
        "b": {"game_balanced_disagreement_rate": 0.05},
    }
    close = {
        "a": {"game_balanced_disagreement_rate": 0.02},
        "b": {"game_balanced_disagreement_rate": 0.03},
    }

    assert _movement_concentrated(concentrated)
    assert not _movement_concentrated(close)
