import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from splendor_ai.bots import RandomLegalBot, ShallowSearchBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.training import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.model import (
    LegalActionScorerConfig,
    LegalActionScorerModel,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_policy_search_disagreement import (
    ModelActionScore,
    _empty_counters,
    _record_row,
)


def _write_snapshot_replay(tmp_path, max_turns: int = 4):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=max_turns,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "snapshot_replay.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    return replay_path


def _write_legal_action_scorer_checkpoint(tmp_path):
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=12,
            state_depth=1,
            action_depth=1,
        )
    )
    checkpoint_path = tmp_path / "legal_action_scorer.pt"
    torch.save(
        {
            "model_kind": "legal_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_record_row_counts_action_type_disagreement(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    payload = json.loads(replay_path.read_text(encoding="utf-8").splitlines()[0])
    state = deserialize_state_snapshot(payload["state_snapshot"])
    from splendor_ai.engine.env import SplendorEnv

    env = SplendorEnv()
    legal_actions = env.legal_actions(state)
    if len(legal_actions) < 2:
        pytest.skip("Need at least two legal actions for disagreement test.")
    codec = ActionCodec()
    search_scores = ShallowSearchBot(seed=0).rank_actions(env, state, legal_actions)
    if len(search_scores) < 2:
        pytest.skip("Need at least two searched actions for disagreement test.")
    model_action = next(
        action
        for action in legal_actions
        if codec.encode(state, action) != codec.encode(state, search_scores[0].action)
    )
    counters = _empty_counters()
    examples = []

    _record_row(
        counters=counters,
        examples=examples,
        examples_limit=3,
        codec=codec,
        state=state,
        payload=payload,
        replay_path=replay_path,
        model_scores=(
            ModelActionScore(
                action=model_action,
                action_index=codec.encode(state, model_action),
                score=1.0,
            ),
        ),
        search_scores=search_scores,
    )

    assert counters["rows_analyzed"] == 1
    assert counters["disagreements"] == 1
    assert counters["by_model_action_type"]
    assert counters["by_search_action_type"]
    assert counters["action_type_matrix"]


def test_run_policy_search_disagreement_cli_writes_summary(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path, max_turns=5)
    checkpoint_path = _write_legal_action_scorer_checkpoint(tmp_path)
    output_path = tmp_path / "analysis.json"

    result = subprocess.run(
        [
            sys.executable,
            "run_policy_search_disagreement.py",
            "--checkpoint",
            str(checkpoint_path),
            "--model-kind",
            "legal_action_scorer",
            "--replay-path",
            str(replay_path),
            "--output-path",
            str(output_path),
            "--max-rows",
            "2",
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
            "--log-every",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Policy/search disagreement" in result.stdout
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["rows_seen"] == 2
    assert summary["rows_analyzed"] > 0
    assert "agreement_rate" in summary["overall"]
    assert "action_type_matrix" in summary
