from __future__ import annotations

import pytest

from splendor_ai.alphazero.high_budget_distillation import (
    HOLDOUT_GAMES_PER_SHARD,
    SHARD_SEED_STARTS,
    TRAIN_GAMES_PER_SHARD,
    _arena_shard_identity,
    _panel_spec,
    combine_arena_shards,
    compression_assessment,
    distillation_qualification,
    distillation_training_config,
    high_budget_distillation_decision,
    partition_seed_schedule,
)


def test_partition_is_exact_whole_game_and_disjoint() -> None:
    partition = partition_seed_schedule(
        [list(range(start, start + 64)) for start in SHARD_SEED_STARTS]
    )
    assert len(partition["train"]) == 4 * TRAIN_GAMES_PER_SHARD == 224
    assert len(partition["holdout"]) == 4 * HOLDOUT_GAMES_PER_SHARD == 32
    assert set(partition["train"]).isdisjoint(partition["holdout"])


def test_training_recipe_preserves_parent_safe_proven_weights() -> None:
    config = distillation_training_config()
    assert config.epochs == 12
    assert config.batch_size == 256
    assert config.policy_loss_weight == 1.0
    assert config.value_loss_weight == 0.5
    assert config.early_stopping_patience == 3
    assert config.validation_fraction == 0.2


def test_offline_gate_requires_policy_objective_and_bounded_value() -> None:
    parent = {"policy_loss": 2.0, "value_loss": 0.9, "objective_loss": 2.45}
    good = {"policy_loss": 1.98, "value_loss": 0.91, "objective_loss": 2.435}
    assert distillation_qualification(
        training={"selected_epoch": 1}, parent=parent, candidate=good
    )["qualified"]
    assert not distillation_qualification(
        training={"selected_epoch": 0}, parent=parent, candidate=good
    )["qualified"]
    assert not distillation_qualification(
        training={"selected_epoch": 1},
        parent=parent,
        candidate={**good, "value_loss": 0.926, "objective_loss": 2.443},
    )["qualified"]


def test_primary_and_diagnostic_specs_are_fresh_and_compute_matched() -> None:
    primary = _panel_spec("primary_256")
    diagnostic = _panel_spec("diagnostic_1024")
    assert primary == {
        "seed_start": 2_130_000,
        "paired_seeds": 200,
        "simulations": 256,
        "paired_per_shard": 50,
    }
    assert diagnostic == {
        "seed_start": 2_140_000,
        "paired_seeds": 100,
        "simulations": 1_024,
        "paired_per_shard": 25,
    }
    identity = _arena_shard_identity("diagnostic_1024", 0, "candidate", 200)
    assert identity["candidate_search_config"] == identity["baseline_search_config"]
    assert identity["candidate_search_config"]["simulations"] == 1_024


def _shards(panel: str) -> dict[int, dict[str, object]]:
    spec = _panel_spec(panel)
    records: dict[int, dict[str, object]] = {}
    for index in range(4):
        start = spec["seed_start"] + index * spec["paired_per_shard"]
        rows = [
            {
                "seed": start + offset,
                "candidate_seat": seat,
                "candidate_points": 1.0 if seat == 0 else 0.0,
                "candidate_margin": 1 if seat == 0 else -1,
                "termination_reason": "completed",
            }
            for offset in range(spec["paired_per_shard"])
            for seat in (0, 1)
        ]
        records[index] = {
            "paired_seeds": spec["paired_per_shard"],
            "games": 2 * spec["paired_per_shard"],
            "rows": rows,
        }
    return records


@pytest.mark.parametrize("panel,games", [("primary_256", 400), ("diagnostic_1024", 200)])
def test_combiner_requires_and_summarizes_exact_panel_shards(panel: str, games: int) -> None:
    arena = combine_arena_shards(panel, _shards(panel))
    assert arena["games"] == games
    assert arena["score_share"] == 0.5
    missing = _shards(panel)
    del missing[3]
    with pytest.raises(ValueError, match="four exact"):
        combine_arena_shards(panel, missing)  # type: ignore[arg-type]


def test_primary_gate_adds_strict_confidence_requirement() -> None:
    arena = {
        "games": 400,
        "score_share": 0.56,
        "mean_margin": 0.5,
        "termination_reasons": {"completed": 400},
        "rows": [
            {"candidate_seat": seat, "candidate_points": 0.56}
            for seat in (0, 1)
            for _ in range(200)
        ],
    }
    assert not compression_assessment(arena, {"ci95_low": 0.5})["passed"]
    assert compression_assessment(arena, {"ci95_low": 0.501})["passed"]


def test_decision_matrix_is_frozen_and_never_installs() -> None:
    cases = [
        (False, None, None, "offline_gate_failed"),
        (True, True, True, "absorbed_at_both"),
        (True, True, False, "compressed_primary_pass"),
        (True, False, True, "requires_expensive_search"),
        (True, False, False, "not_converted"),
    ]
    for qualified, primary, diagnostic, fragment in cases:
        decision = high_budget_distillation_decision(
            qualified=qualified,
            primary_passed=primary,
            diagnostic_passed=diagnostic,
        )
        assert fragment in decision
        assert decision.endswith("no_install")
