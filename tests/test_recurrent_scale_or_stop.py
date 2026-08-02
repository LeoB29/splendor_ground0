from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from splendor_ai.training.run_recurrent_scale_or_stop import (
    COLLECTION_GAMES,
    GAMES_PER_CELL,
    TRAIN_GAMES_PER_CELL,
    VALIDATION_GAMES_PER_CELL,
    _balance_exact,
    _completed_generation,
    _config,
    _decision,
    _final_strength,
)


def test_scaled_generation_sizes_sum_to_exact_protocol_total() -> None:
    assert COLLECTION_GAMES == {1: 2_880, 2: 3_360, 3: 3_840}
    assert sum(COLLECTION_GAMES.values()) == 10_080
    assert [_config(index).games for index in (1, 2, 3)] == [
        2_880,
        3_360,
        3_840,
    ]


def test_balance_requires_exact_192_48_split_in_every_seat_cell() -> None:
    labels = ["a", "b"]
    collection = SimpleNamespace(
        opponent_seat_games={
            label: {"seat0": GAMES_PER_CELL, "seat1": GAMES_PER_CELL}
            for label in labels
        },
        split_opponent_seat_games={
            "train": {
                label: {
                    "seat0": TRAIN_GAMES_PER_CELL,
                    "seat1": TRAIN_GAMES_PER_CELL,
                }
                for label in labels
            },
            "validation": {
                label: {
                    "seat0": VALIDATION_GAMES_PER_CELL,
                    "seat1": VALIDATION_GAMES_PER_CELL,
                }
                for label in labels
            },
        },
    )

    assert _balance_exact(collection, labels)
    collection.split_opponent_seat_games["validation"]["b"]["seat1"] -= 1
    assert not _balance_exact(collection, labels)


def _interval(mean: float, lower: float, upper: float) -> dict[str, float]:
    return {"mean": mean, "lower": lower, "upper": upper}


def _passing_inference() -> dict:
    return {
        "aggregate": {
            "score_share_delta": _interval(0.025, -0.025, 0.075),
            "margin_delta": _interval(0.5, -0.5, 1.5),
        },
        "by_opponent": {
            opponent: {
                "score_share_delta": _interval(0.0, -0.1, 0.1),
            }
            for opponent in ("035", "016", "023", "shallow")
        },
        "by_seat": {
            "seat0": {"score_share_delta": -0.025},
            "seat1": {"score_share_delta": 0.075},
        },
        "split_halves": {
            "even": {"score_share_delta": -0.025},
            "odd": {"score_share_delta": 0.075},
        },
    }


def _passing_policy_diagnostics() -> dict:
    return {
        "versus_base_generation2": {
            "argmax_movement": 0.01,
            "raw_kl": 0.04,
        },
        "versus_epoch0": {
            "argmax_movement": 0.30,
            "raw_kl": 0.05,
        },
    }


def test_final_strength_accepts_every_inclusive_boundary() -> None:
    gates = _final_strength(_passing_inference(), _passing_policy_diagnostics())

    assert all(gates.values())


def test_final_strength_rejects_neutral_score_after_scale() -> None:
    inference = _passing_inference()
    inference["aggregate"]["score_share_delta"]["mean"] = 0.0249

    gates = _final_strength(inference, _passing_policy_diagnostics())

    assert not gates["meaningful_aggregate_score_lift"]
    assert gates["aggregate_score_guardrail"]


def test_decision_keeps_neutral_and_systematic_failures_distinct() -> None:
    generations = [{"accepted": True} for _ in range(3)]
    inference = _passing_inference()
    strength = _final_strength(inference, _passing_policy_diagnostics())
    strength["meaningful_aggregate_score_lift"] = False

    assert _decision(
        prerequisites_passed=True,
        resume_error=None,
        generation_payloads=generations,
        final_integrity={"all": True},
        final_strength=strength,
        inference=inference,
    ) == "recurrent_scale_remains_neutral_stop_outcome_only_branch"

    inference["aggregate"]["score_share_delta"]["upper"] = -0.026
    assert _decision(
        prerequisites_passed=True,
        resume_error=None,
        generation_payloads=generations,
        final_integrity={"all": True},
        final_strength=strength,
        inference=inference,
    ) == "recurrent_scale_systematic_regression_stop_outcome_only_branch"


def test_decision_never_interprets_invalid_or_incomplete_evidence() -> None:
    assert _decision(
        prerequisites_passed=False,
        resume_error=None,
        generation_payloads=[],
        final_integrity={},
        final_strength={},
        inference=None,
    ) == "recurrent_scale_or_stop_invalid_no_learning_decision"
    assert _decision(
        prerequisites_passed=True,
        resume_error=None,
        generation_payloads=[{"accepted": False}],
        final_integrity={},
        final_strength={},
        inference=None,
    ) == (
        "recurrent_scale_learning_contract_failed_stop_outcome_only_branch"
    )


def test_completed_generation_resume_is_bound_to_parent_and_candidate(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.pt"
    candidate = tmp_path / "scale_generation_1.pt"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    from splendor_ai.training.recurrent_league import sha256
    from splendor_ai.training.run_recurrent_scale_or_stop import PROTOCOL_COMMIT

    record = {
        "experiment": "069",
        "protocol_commit": PROTOCOL_COMMIT,
        "scale_generation": 1,
        "parent_checkpoint": str(parent).replace("\\", "/"),
        "parent_sha256": sha256(parent),
        "candidate_checkpoint": str(candidate).replace("\\", "/"),
        "candidate_sha256": sha256(candidate),
        "gates": {"all": True},
        "accepted": True,
    }
    (tmp_path / "scale_generation_1.json").write_text(json.dumps(record))

    loaded, error = _completed_generation(tmp_path, 1, parent)
    assert error is None
    assert loaded == record

    candidate.write_bytes(b"mutated")
    loaded, error = _completed_generation(tmp_path, 1, parent)
    assert loaded is None
    assert error == "completed_generation_contract_mismatch"
