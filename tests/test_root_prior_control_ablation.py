from splendor_ai.training.run_root_prior_control_ablation import (
    RootPriorControlAblationConfig,
    _puct_config,
    _variants,
    _variant_summary,
    build_parser,
)


def test_root_prior_control_parser_uses_frozen_search_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--audit-path", "audit.json",
            "--replay-path", "replay.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--output-path", "output.json",
        ]
    )

    assert args.simulations == 64
    assert args.n_step_leaf_delta_clip == 0.20


def test_root_prior_control_only_changes_root_mix(tmp_path) -> None:
    config = RootPriorControlAblationConfig(
        audit_path=tmp_path / "audit.json",
        replay_paths=(tmp_path / "replay.jsonl",),
        policy_checkpoint=tmp_path / "policy.pt",
        output_path=tmp_path / "output.json",
    )

    frozen = _puct_config(config, 1.0)
    uniform = _puct_config(config, 0.0)

    assert frozen.root_prior_mix == 1.0
    assert uniform.root_prior_mix == 0.0
    assert frozen.n_step_leaf_delta_clip == uniform.n_step_leaf_delta_clip
    assert frozen.n_step_leaf_depth == uniform.n_step_leaf_depth


def test_variant_summary_reads_resolution_from_parent_rows(tmp_path) -> None:
    config = RootPriorControlAblationConfig(
        audit_path=tmp_path / "audit.json",
        replay_paths=(tmp_path / "replay.jsonl",),
        policy_checkpoint=tmp_path / "policy.pt",
        output_path=tmp_path / "output.json",
    )
    rows = [
        {
            "resolved": True,
            "source_puct_choice": 1,
            "variants": {
                "frozen": {
                    "selected_action_index": 1,
                    "regret": 0.2,
                    "oracle_match": False,
                    "root_effective_actions": 1.0,
                    "root_visited_actions": 1,
                }
            },
        }
    ]

    summary = _variant_summary(rows, "frozen", config)

    assert summary["resolved_states"] == 1
    assert summary["mean_regret"] == 0.2
    assert summary["source_replay_choice_match_rate"] == 1.0


def test_confirmation_variants_exclude_the_exploratory_mixed_control(tmp_path) -> None:
    config = RootPriorControlAblationConfig(
        audit_path=tmp_path / "audit.json",
        replay_paths=(tmp_path / "replay.jsonl",),
        policy_checkpoint=tmp_path / "policy.pt",
        output_path=tmp_path / "output.json",
    )

    assert _variants(config) == (("frozen", 1.0), ("uniform", 0.0))
