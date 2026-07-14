from dataclasses import asdict
import json
import subprocess
import sys

import torch

from splendor_ai.bots import RandomLegalBot
from splendor_ai.training import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.model import (
    LegalActionScorerConfig,
    LegalActionScorerModel,
    ValueMLP,
    ValueModelConfig,
)
from splendor_ai.training.run_search_value_diff import (
    SearchValueDiffConfig,
    compare_search_value_diff,
)


def _write_snapshot_replay(tmp_path):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=6,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "replays.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    return replay_path


def _write_prior_checkpoint(tmp_path):
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=12,
            state_depth=1,
            action_depth=1,
        )
    )
    checkpoint_path = tmp_path / "prior.pt"
    torch.save(
        {
            "model_kind": "legal_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_value_checkpoint(tmp_path):
    model = ValueMLP(ValueModelConfig(hidden_size=16, depth=1))
    checkpoint_path = tmp_path / "value.pt"
    torch.save(
        {
            "model_kind": "value_mlp",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_compare_search_value_diff_writes_summary(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    prior_path = _write_prior_checkpoint(tmp_path)
    value_path = _write_value_checkpoint(tmp_path)
    output_path = tmp_path / "summary.json"

    summary = compare_search_value_diff(
        SearchValueDiffConfig(
            replay_paths=(replay_path,),
            output_path=output_path,
            policy_prior_checkpoint_path=prior_path,
            value_checkpoint_path=value_path,
            search_depth=1,
            search_max_branching=4,
            search_buy_branching=2,
            search_reserve_branching=1,
            search_take_branching=1,
            max_rows=3,
            log_every=0,
        )
    )

    assert output_path.exists()
    assert summary["rows_analyzed"] > 0
    assert "changed_rate" in summary["overall"]
    assert "by_seat" in summary
    assert "action_type_matrix" in summary


def test_run_search_value_diff_cli_writes_summary(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    prior_path = _write_prior_checkpoint(tmp_path)
    value_path = _write_value_checkpoint(tmp_path)
    output_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "run_search_value_diff.py",
            "--replay-path",
            str(replay_path),
            "--output-path",
            str(output_path),
            "--policy-prior-checkpoint",
            str(prior_path),
            "--value-checkpoint",
            str(value_path),
            "--search-depth",
            "1",
            "--search-max-branching",
            "4",
            "--search-buy-branching",
            "2",
            "--search-reserve-branching",
            "1",
            "--search-take-branching",
            "1",
            "--max-rows",
            "2",
            "--log-every",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "[search-value-diff]" in result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["rows_seen"] == 2
    assert payload["rows_analyzed"] > 0
