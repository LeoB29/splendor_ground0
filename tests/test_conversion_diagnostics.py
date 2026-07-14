import json
import subprocess
import sys

from splendor_ai.bots import RandomLegalBot
from splendor_ai.training import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.run_conversion_diagnostics import (
    ConversionDiagnosticsConfig,
    run_conversion_diagnostics,
)


def _write_snapshot_replay(tmp_path, seed: int = 3, max_turns: int = 10):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=seed,
        max_turns=max_turns,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / f"snapshot_replay_{seed}.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    return replay_path


def _write_replay_with_changed_first_action(tmp_path, source_path):
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["action_payload"] = {
        "action_type": "PASS",
        "tier": None,
        "market_index": None,
        "reserved_index": None,
        "take_tokens": [],
        "return_tokens": [],
        "spend_tokens": [],
        "noble_id": None,
        "metadata": [],
    }
    changed_path = tmp_path / "snapshot_replay_changed.jsonl"
    changed_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return changed_path


def test_conversion_diagnostics_writes_summary_and_pair_attribution(tmp_path) -> None:
    before_path = _write_snapshot_replay(tmp_path)
    after_path = _write_replay_with_changed_first_action(tmp_path, before_path)
    output_path = tmp_path / "conversion.json"

    summary = run_conversion_diagnostics(
        ConversionDiagnosticsConfig(
            replay_paths=(before_path, after_path),
            labels=("before", "after"),
            output_path=output_path,
            early_turn_max=4,
            window_turns=(4, 8),
            log_every=0,
        )
    )

    assert output_path.exists()
    assert summary["rows_seen"] > 0
    assert summary["player_games"] > 0
    assert summary["by_label"]["before"]["overall"]["count"] > 0
    assert "early_action_buckets" in summary["by_label"]["before"]
    paired = summary["paired_first_two_labels"]
    assert paired["paired_player_games"] > 0
    assert paired["changed_early_action_type_buckets"]


def test_run_conversion_diagnostics_cli_writes_summary(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path, seed=5, max_turns=6)
    output_path = tmp_path / "analysis.json"

    result = subprocess.run(
        [
            sys.executable,
            "run_conversion_diagnostics.py",
            "--replay-path",
            str(replay_path),
            "--label",
            "candidate",
            "--output-path",
            str(output_path),
            "--early-turn-max",
            "3",
            "--window-turns",
            "3",
            "6",
            "--log-every",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "[conversion]" in result.stdout
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["labels"] == ["candidate"]
    assert summary["by_label"]["candidate"]["by_window"]
