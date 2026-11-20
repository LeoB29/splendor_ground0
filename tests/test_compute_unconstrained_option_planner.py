from __future__ import annotations

import subprocess
import sys

from splendor_ai.eval.run_compute_unconstrained_option_planner import strength_decision


def test_strength_decision_is_frozen() -> None:
    integrity = {"all": True}
    assert strength_decision(
        integrity, {"mean": 0.55, "ci95_low": 0.505, "ci95_high": 0.60}
    ).endswith("stronger_promote_research_champion")
    assert strength_decision(
        integrity, {"mean": 0.54, "ci95_low": 0.49, "ci95_high": 0.59}
    ).endswith("not_decisively_stronger_close_architecture")
    assert strength_decision(
        {"all": False}, {"mean": 0.60, "ci95_low": 0.55, "ci95_high": 0.65}
    ).endswith("invalid_no_strength_decision")


def test_root_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "run_compute_unconstrained_option_planner_080.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Compute-unconstrained persistent option planner" in result.stdout
