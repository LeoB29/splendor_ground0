from __future__ import annotations

import hashlib
from pathlib import Path

from run_alphazero_start_role_operational_audit_126 import build_parser
from splendor_ai.alphazero.start_role_operational_audit import (
    BUDGETS,
    PROTOCOL_SHA256,
    _materiality,
    _octile_indices,
)


def test_126_protocol_hash_and_no_training_launch_path() -> None:
    protocol = Path(
        "docs/engineering/alphazero_start_role_operational_importance_audit_126.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    args = build_parser().parse_args([])
    assert BUDGETS == (64, 256)
    assert not hasattr(args, "launch")
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "learning_rate")


def test_126_octile_midpoints_are_unique_and_span_game() -> None:
    assert _octile_indices(56) == (3, 10, 17, 24, 31, 38, 45, 52)
    assert _octile_indices(68) == (4, 12, 21, 29, 38, 46, 55, 63)


def test_126_materiality_uses_frozen_policy_and_value_rules() -> None:
    summary = {
        "mean_visit_tv": 0.05,
        "top_action_change_rate": 0.10,
        "mean_abs_root_value_delta": 0.049,
        "bootstrap": {
            "visit_tv": {"ci95_low": 0.011},
            "abs_root_value_delta": {"ci95_low": 0.02},
        },
    }
    gate = _materiality(summary)
    assert gate["policy_material"] is True
    assert gate["value_material"] is False
    assert gate["material"] is True

    summary["top_action_change_rate"] = 0.099
    summary["mean_abs_root_value_delta"] = 0.05
    gate = _materiality(summary)
    assert gate["policy_material"] is False
    assert gate["value_material"] is True
    assert gate["material"] is True
