import pytest

from splendor_ai.training.run_leaf_quality_audit import _pearson, _phase, _spearman, build_parser


def test_leaf_audit_phase_buckets_respect_late_boundary() -> None:
    assert _phase(40) == "late_40_49"
    assert _phase(49) == "late_40_49"
    assert _phase(50) == "endgame_50_59"
    assert _phase(60) == "endgame_60_plus"


def test_rank_correlation_identifies_aligned_and_reversed_scores() -> None:
    values = [0.1, 0.4, 0.2]
    assert _pearson(values, values) == pytest.approx(1.0)
    assert _spearman(values, values) == pytest.approx(1.0)
    assert _spearman(values, [-value for value in values]) == pytest.approx(-1.0)


def test_parser_accepts_optional_corpus_output_path() -> None:
    args = build_parser().parse_args(
        [
            "--replay-path", "replay.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--value009-checkpoint", "value009.pt",
            "--value010-checkpoint", "value010.pt",
            "--output-path", "audit.json",
            "--corpus-output-path", "corpus.jsonl",
        ]
    )

    assert args.corpus_output_path == "corpus.jsonl"
