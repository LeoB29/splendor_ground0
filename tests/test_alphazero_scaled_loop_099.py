from __future__ import annotations

from pathlib import Path

from splendor_ai.alphazero.scaled_loop import (
    rolling_replay_paths,
    selfplay_corpus_validity,
    training_validity,
)


def test_rolling_window_bootstraps_then_keeps_two_fresh_corpora() -> None:
    bootstrap = Path("bootstrap.jsonl")
    fresh = [Path(f"generation_{generation}.jsonl") for generation in (2, 3, 4)]
    assert rolling_replay_paths(bootstrap, fresh[:1]) == [bootstrap, fresh[0]]
    assert rolling_replay_paths(bootstrap, fresh[:2]) == fresh[:2]
    assert rolling_replay_paths(bootstrap, fresh) == fresh[1:]


def test_selfplay_validity_requires_exact_clean_256_simulation_dose() -> None:
    summary = {
        "games": 256,
        "positions": 1000,
        "simulations": 256_000,
        "termination_reasons": {"completed": 256},
    }
    assert selfplay_corpus_validity(summary)["valid"] is True
    summary["simulations"] -= 1
    assert selfplay_corpus_validity(summary)["valid"] is False
    summary["simulations"] += 1
    summary["termination_reasons"] = {"completed": 255, "max_turns": 1}
    assert selfplay_corpus_validity(summary)["valid"] is False


def _training() -> dict:
    metrics = {"policy_loss": 1.0, "value_loss": 0.5}
    return {
        "parent_generation": 2,
        "child_generation": 3,
        "validation_rows": 10,
        "epochs_completed": 2,
        "parent_train_metrics": metrics,
        "parent_validation_metrics": metrics,
        "child_train_metrics": metrics,
        "child_validation_metrics": metrics,
        "history": [{"train": metrics, "validation": metrics}],
    }


def test_training_validity_is_engineering_only_and_rejects_nonfinite_metrics() -> None:
    valid = training_validity(_training(), expected_parent=2)
    assert valid["valid"] is True
    degraded_but_finite = _training()
    degraded_but_finite["child_validation_metrics"] = {
        "policy_loss": 99.0,
        "value_loss": 99.0,
    }
    assert training_validity(degraded_but_finite, expected_parent=2)["valid"] is True
    nonfinite = _training()
    nonfinite["history"][0]["validation"] = {"objective_loss": float("nan")}
    assert training_validity(nonfinite, expected_parent=2)["valid"] is False


def test_training_validity_requires_exact_lineage_and_nonempty_work() -> None:
    wrong = _training()
    wrong["child_generation"] = 4
    assert training_validity(wrong, expected_parent=2)["valid"] is False
    empty = _training()
    empty["validation_rows"] = 0
    assert training_validity(empty, expected_parent=2)["valid"] is False
