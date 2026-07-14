from pathlib import Path

from splendor_ai.eval.parallel_paired import (
    PairedCheckpointConfig,
    _pair_metrics,
    _progress_identity,
    build_parser,
)


def _config() -> PairedCheckpointConfig:
    return PairedCheckpointConfig(
        candidate_checkpoint=Path("candidate.pt"),
        baseline_checkpoint=Path("baseline.pt"),
        output_path=Path("result.json"),
        checkpoint_path=Path("progress.json"),
        pairs=3,
    )


def test_pair_metrics_cancel_seat_order_and_include_bounded_margin() -> None:
    row = {
        "seed": 12,
        "candidate_seats": [0, 1],
        "games": [
            {"winner": 0, "final_scores": [15, 10]},
            {"winner": 0, "final_scores": [14, 15]},
        ],
    }

    metrics = _pair_metrics(row, _config())

    assert metrics["score_share"] == 0.5
    assert metrics["outcome"] == 0.0
    assert metrics["final_score_margin"] == 3.0
    assert 0.0 < metrics["bounded_utility"] < 1.0


def test_progress_identity_excludes_output_locations() -> None:
    config = _config()

    identity = _progress_identity(config)

    assert "output_path" not in identity
    assert "checkpoint_path" not in identity
    assert identity["pairs"] == 3


def test_parallel_paired_parser_accepts_resumable_gate_options() -> None:
    args = build_parser().parse_args(
        [
            "--candidate-checkpoint", "candidate.pt",
            "--baseline-checkpoint", "baseline.pt",
            "--output-path", "result.json",
            "--pairs", "50",
            "--workers", "3",
        ]
    )

    assert args.pairs == 50
    assert args.workers == 3
