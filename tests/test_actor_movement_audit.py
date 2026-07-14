from pathlib import Path

from splendor_ai.training.run_actor_movement_audit import (
    ActorMovementAuditConfig,
    _evaluate_gates,
    build_parser,
)


def _config(tmp_path: Path) -> ActorMovementAuditConfig:
    return ActorMovementAuditConfig(
        parent_checkpoint=tmp_path / "parent.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        training_metrics_path=tmp_path / "metrics.json",
        on_policy_replay_paths=(tmp_path / "on_policy.jsonl",),
        broad_replay_paths=(tmp_path / "broad.jsonl",),
        output_path=tmp_path / "audit.json",
    )


def test_actor_movement_gates_require_bounded_nonzero_movement(tmp_path: Path) -> None:
    config = _config(tmp_path)
    on_policy = {
        "top_action_change_rate": 0.02,
        "exploration_policy_kl": {"mean": 0.002},
    }
    broad = {
        "top_action_change_rate": 0.01,
        "exploration_policy_kl": {"mean": 0.001},
    }
    epoch = {
        "behavior_log_prob_mae": 1e-7,
        "clip_fraction": 0.01,
        "approx_ratio": 1.0,
    }

    assert all(_evaluate_gates(config, on_policy, broad, epoch).values())
    broad["top_action_change_rate"] = 0.0
    assert not _evaluate_gates(config, on_policy, broad, epoch)["broad_movement_floor"]


def test_actor_movement_parser_accepts_repeated_stream_paths() -> None:
    args = build_parser().parse_args(
        [
            "--parent-checkpoint", "parent.pt",
            "--candidate-checkpoint", "candidate.pt",
            "--training-metrics-path", "metrics.json",
            "--on-policy-replay-path", "on1.jsonl",
            "--on-policy-replay-path", "on2.jsonl",
            "--broad-replay-path", "broad.jsonl",
            "--output-path", "audit.json",
        ]
    )

    assert args.on_policy_replay_path == ["on1.jsonl", "on2.jsonl"]
