from __future__ import annotations

from collections import Counter

from splendor_ai.eval.run_compute_matched_belief_breadth import (
    CONFIGURATIONS,
    BeliefBreadthConfig,
    _candidate_gates,
    _paired_bootstrap,
    _validate_config,
    breadth_belief_root_index,
    breadth_continuation_seed,
    select_breadth_states,
)


def test_all_breadth_candidates_use_exactly_192_continuations() -> None:
    assert tuple(CONFIGURATIONS) == ("3x4", "6x2", "12x1")
    assert all(
        determinizations * 4 * 4 * continuations == 192
        for determinizations, continuations in CONFIGURATIONS.values()
    )


def test_059_selects_fixed_early_primary_and_guardrails() -> None:
    cfg = BeliefBreadthConfig()
    _validate_config(cfg)
    selected = select_breadth_states(cfg)

    assert len(selected) == 48
    assert len({row["game_seed"] for row in selected}) == 17
    counts = Counter(
        f"{row['phase']}|seat{row['actor_seat']}|{row['gap_stratum']}"
        for row in selected
    )
    assert len(counts) == 18
    assert all(
        count == (4 if cell.startswith("early|") else 2)
        for cell, count in counts.items()
    )


def test_belief_roots_are_nested_and_sides_are_disjoint() -> None:
    cfg = BeliefBreadthConfig()

    left = breadth_belief_root_index(cfg, 9, "left")
    right = breadth_belief_root_index(cfg, 9, "right")

    assert left == cfg.belief_root_base + 90
    assert right == left + 1


def test_all_configurations_use_same_twelve_continuation_slots() -> None:
    cfg = BeliefBreadthConfig()
    seed_sets = {}
    for name, (determinizations, continuations) in CONFIGURATIONS.items():
        seed_sets[name] = {
            breadth_continuation_seed(
                cfg, 7, "left", det, continuations, continuation
            )
            for det in range(determinizations)
            for continuation in range(continuations)
        }

    assert seed_sets["3x4"] == seed_sets["6x2"] == seed_sets["12x1"]
    assert len(seed_sets["3x4"]) == 12
    assert seed_sets["3x4"].isdisjoint(
        {
            breadth_continuation_seed(
                cfg, 7, "right", det, 1, 0
            )
            for det in range(12)
        }
    )


def test_paired_bootstrap_is_deterministic_and_clustered() -> None:
    rows = [
        {
            "game_seed": game,
            "baseline_exact": 0.0,
            "candidate_exact": 1.0,
            "baseline_equivalence": 0.0,
            "candidate_equivalence": 1.0,
            "baseline_regret": 0.10,
            "candidate_regret": 0.02,
        }
        for game in (1, 1, 2, 2)
    ]

    first = _paired_bootstrap(rows, 100, 123)
    second = _paired_bootstrap(rows, 100, 123)

    assert first == second
    assert first["exact_gain"] == {"lower": 1.0, "upper": 1.0}
    assert first["equivalence_gain"] == {"lower": 1.0, "upper": 1.0}
    assert first["regret_change"]["upper"] < 0.0


def _summary(
    *,
    exact: float = 0.65,
    equivalence: float = 0.70,
    spearman: float = 0.60,
    mean_regret: float = 0.02,
    p90_regret: float = 0.05,
    exact_gain: float = 0.20,
    equivalence_gain: float = 0.15,
    regret_ratio: float = 0.60,
) -> dict[str, object]:
    phase_breakdown = {
        phase: {
            "exact_best_agreement": exact,
            "value_equivalent_agreement": equivalence,
            "mean_spearman": spearman,
            "mean_symmetric_cross_regret": mean_regret,
        }
        for phase in ("early", "mid", "late")
    }
    structural = {
        "both_completed_fraction": 1.0,
        "phase": {
            phase: {"both_completed_fraction": 1.0}
            for phase in ("early", "mid", "late")
        },
    }
    paired_phase = {
        "early": {
            "baseline_exact": exact - exact_gain,
            "candidate_exact": exact,
            "exact_gain": exact_gain,
            "baseline_equivalence": equivalence - equivalence_gain,
            "candidate_equivalence": equivalence,
            "equivalence_gain": equivalence_gain,
            "baseline_mean_regret": mean_regret / regret_ratio,
            "candidate_mean_regret": mean_regret,
            "regret_change": mean_regret - mean_regret / regret_ratio,
            "regret_ratio": regret_ratio,
        },
        "mid": {
            "baseline_exact": exact,
            "candidate_exact": exact,
            "baseline_equivalence": equivalence,
            "candidate_equivalence": equivalence,
            "baseline_mean_regret": mean_regret,
            "candidate_mean_regret": mean_regret,
        },
        "late": {
            "baseline_exact": exact,
            "candidate_exact": exact,
            "baseline_equivalence": equivalence,
            "candidate_equivalence": equivalence,
            "baseline_mean_regret": mean_regret,
            "candidate_mean_regret": mean_regret,
        },
    }
    candidate_rows = [
        {
            "phase": "early",
            "candidate_regret": p90_regret,
        }
        for _ in range(24)
    ]
    return {
        "configurations": {
            "12x1": {
                "breakdowns": {"phase": phase_breakdown},
                "structural": structural,
            }
        },
        "paired_vs_3x4": {
            "12x1": {
                "by_phase": paired_phase,
                "early_bootstrap": {
                    "exact_gain": {"lower": 0.01, "upper": 0.30},
                    "equivalence_gain": {"lower": 0.01, "upper": 0.30},
                    "regret_change": {"lower": -0.03, "upper": -0.001},
                },
                "rows": candidate_rows,
            }
        },
    }


def test_candidate_gates_require_breadth_gain_and_guardrails() -> None:
    cfg = BeliefBreadthConfig()
    summary = _summary()

    assert all(_candidate_gates(cfg, summary, "12x1").values())

    summary["paired_vs_3x4"]["12x1"]["by_phase"]["early"][
        "exact_gain"
    ] = 0.10
    assert not all(_candidate_gates(cfg, summary, "12x1").values())
