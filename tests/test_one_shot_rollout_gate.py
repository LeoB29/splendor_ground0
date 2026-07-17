from __future__ import annotations

import pytest

from splendor_ai.eval.run_one_shot_rollout_gate import assigned_phase, operational_rates


def test_assigned_phase_cycles_deterministically_by_seed() -> None:
    assert [assigned_phase(seed) for seed in range(6)] == [
        "early",
        "mid",
        "late",
        "early",
        "mid",
        "late",
    ]


def test_operational_rates_project_trigger_switches_over_all_turns() -> None:
    rates = operational_rates(
        eligible_triggers=80,
        actor_turns=2_000,
        trigger_switch_fraction=0.25,
    )
    assert rates["search_turn_fraction"] == pytest.approx(0.04)
    assert rates["projected_switch_turn_fraction"] == pytest.approx(0.01)


def test_operational_rates_handle_empty_turn_stream() -> None:
    assert operational_rates(0, 0, 0.5) == {
        "search_turn_fraction": 0.0,
        "projected_switch_turn_fraction": 0.0,
    }
