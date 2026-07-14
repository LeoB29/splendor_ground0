from splendor_ai.training.run_puct_budget_ablation import (
    PuctBudgetAblationConfig,
    _puct_config,
    build_parser,
)


def test_budget_ablation_parser_uses_cpu_and_fixed_budget_variants() -> None:
    args = build_parser().parse_args(
        [
            "--audit-path", "audit.json",
            "--replay-path", "replay.jsonl",
            "--policy-checkpoint", "policy.pt",
            "--output-path", "output.json",
        ]
    )

    assert args.device == "cpu"
    assert args.n_step_leaf_delta_clip == 0.20


def test_budget_ablation_preserves_frozen_root_prior(tmp_path) -> None:
    config = PuctBudgetAblationConfig(
        audit_path=tmp_path / "audit.json",
        replay_paths=(tmp_path / "replay.jsonl",),
        policy_checkpoint=tmp_path / "policy.pt",
        output_path=tmp_path / "output.json",
    )

    puct = _puct_config(config, 512)

    assert puct.simulations == 512
    assert puct.root_prior_mix == 1.0
    assert puct.n_step_leaf_depth == 2
