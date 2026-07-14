from splendor_ai.training.run_midgame_headroom_audit import _band, build_parser


def test_midgame_bands_are_contiguous_and_exclusive() -> None:
    assert _band(14) is None
    assert _band(15) == "mid_15_24"
    assert _band(24) == "mid_15_24"
    assert _band(25) == "mid_25_32"
    assert _band(32) == "mid_25_32"
    assert _band(33) == "mid_33_40"
    assert _band(40) == "mid_33_40"
    assert _band(41) is None


def test_midgame_parser_keeps_two_stage_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--replay-path", "replay.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--output-path", "audit.json",
        ]
    )

    assert args.screening_states_per_band == 20
    assert args.refined_states_per_band == 10


def test_midgame_parser_accepts_resumable_checkpoint_path() -> None:
    args = build_parser().parse_args(
        [
            "--replay-path", "replay.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--output-path", "audit.json",
            "--checkpoint-path", "progress.json",
        ]
    )

    assert args.checkpoint_path == "progress.json"
