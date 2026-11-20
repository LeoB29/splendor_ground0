from __future__ import annotations

import hashlib
from pathlib import Path

from run_alphazero_deadlock_consistent_repeated_return_value_ab_132 import (
    PROTOCOL_SHA256,
    RESULT_131_SHA256,
    build_parser,
)
from splendor_ai.alphazero.repeated_return_value_ab import (
    repeated_gameplay_assessment,
)
from splendor_ai.termination import VALID_NATURAL_TERMINATIONS


PROTOCOL = Path(
    "docs/engineering/alphazero_deadlock_consistent_repeated_return_value_ab_132.md"
)
RESULT_131 = Path("outputs/alphazero_repeated_return_value_ab_131/result.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_protocol_and_failed_predecessor_hashes_are_exact() -> None:
    assert _sha256(PROTOCOL) == PROTOCOL_SHA256
    assert _sha256(RESULT_131) == RESULT_131_SHA256


def test_runner_defaults_to_fresh_132_output_without_launch() -> None:
    args = build_parser().parse_args([])
    assert args.output_dir == "outputs/alphazero_deadlock_consistent_repeated_return_value_ab_132"
    assert args.launch is False


def test_corrected_gameplay_gate_accepts_only_natural_terminations() -> None:
    rows = [
        {"candidate_seat": seat, "candidate_points": 1.0}
        for _seed in range(2)
        for seat in (0, 1)
    ]
    arena = {
        "games": 4,
        "score_share": 1.0,
        "mean_margin": 1.0,
        "termination_reasons": {"completed": 3, "forced_pass_deadlock": 1},
        "rows": rows,
    }
    bootstrap = {"ci95_low": 0.75}

    corrected = repeated_gameplay_assessment(
        arena,
        bootstrap,
        VALID_NATURAL_TERMINATIONS,
    )
    old_contract = repeated_gameplay_assessment(
        arena,
        bootstrap,
        ("completed",),
    )

    assert corrected["passed"]
    assert not old_contract["passed"]
    assert not old_contract["checks"]["all_terminations_valid"]
