import json
import math

from splendor_ai.training.run_group_relative_signal_audit import (
    GroupRelativeSignalAuditConfig,
    audit_group_relative_signal,
    build_parser,
)


def test_signal_audit_parser_uses_four_way_leave_one_out_contract() -> None:
    args = build_parser().parse_args(
        [
            "--trajectory-path", "trajectory.jsonl",
            "--corpus-summary-path", "summary.json",
            "--parent-checkpoint", "parent.pt",
            "--output-path", "audit.json",
        ]
    )

    assert args.expected_group_size == 4
    assert args.min_informative_group_rate == 0.75


def test_signal_audit_passes_valid_informative_group(monkeypatch, tmp_path) -> None:
    trajectory_path = tmp_path / "trajectory.jsonl"
    rows = []
    returns = (1.0, -1.0, 0.5, -0.5)
    for index, value in enumerate(returns):
        probability = 0.2 if index % 2 else 0.8
        rows.append(
            {
                "game_seed": 100,
                "player_id": 0,
                "behavior_probability": probability,
                "behavior_log_probability": math.log(probability),
                "behavior_greedy_action": index % 2 == 0,
                "trajectory_metadata": {
                    "source_replay_path": f"repeat_{index}.jsonl",
                    "raw_return": value,
                    "group_key": [100, 0, "sampling", "baseline"],
                },
            }
        )
    trajectory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "groups_total": 1,
                "groups_informative": 1,
                "config": {"baseline_mode": "leave_one_out"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "splendor_ai.training.run_group_relative_signal_audit._reconstruct_behavior_log_probs",
        lambda _config: {"rows": 4, "mae": 1e-8, "max_error": 2e-8},
    )

    result = audit_group_relative_signal(
        GroupRelativeSignalAuditConfig(
            trajectory_path=trajectory_path,
            corpus_summary_path=summary_path,
            parent_checkpoint=tmp_path / "parent.pt",
            output_path=tmp_path / "audit.json",
            min_groups=1,
            min_nongreedy_row_rate=0.1,
        )
    )

    assert result["pass"] is True
    assert result["corpus"]["observed_group_size_min"] == 4
