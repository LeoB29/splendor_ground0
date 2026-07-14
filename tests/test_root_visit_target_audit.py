import pytest

from splendor_ai.training.run_root_visit_target_audit import _mean_ci, _phase_bucket, build_parser


def test_phase_bucket_tracks_the_late_n_step_boundary() -> None:
    assert _phase_bucket(0) == "early_0_19"
    assert _phase_bucket(19) == "early_0_19"
    assert _phase_bucket(20) == "middle_20_39"
    assert _phase_bucket(39) == "middle_20_39"
    assert _phase_bucket(40) == "late_40_59"
    assert _phase_bucket(59) == "late_40_59"
    assert _phase_bucket(60) == "endgame_60_plus"


def test_mean_ci_without_bootstrap_returns_the_exact_mean() -> None:
    summary = _mean_ci([0.2, 0.4, 0.6], bootstrap_samples=0, seed=0)

    assert summary == pytest.approx({"mean": 0.4, "ci95_low": 0.4, "ci95_high": 0.4})


def test_parser_accepts_train_split() -> None:
    args = build_parser().parse_args(
        [
            "--replay-path", "replay.jsonl",
            "--target-source-path", "puct.jsonl",
            "--baseline-checkpoint", "baseline.pt",
            "--candidate-checkpoint", "candidate.pt",
            "--output-path", "audit.json",
            "--split", "train",
        ]
    )

    assert args.split == "train"
