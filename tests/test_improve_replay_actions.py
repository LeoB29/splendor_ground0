import json
from types import SimpleNamespace

from splendor_ai.bots import RandomLegalBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.improve_replay_actions import (
    ImproveReplayConfig,
    improve_replay_actions,
)
from splendor_ai.training.replay import deserialize_state_snapshot


def test_improve_replay_actions_writes_search_relabels(tmp_path) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=8,
        include_state_snapshots=True,
    )
    input_path = tmp_path / "replays.jsonl"
    output_path = tmp_path / "improved.jsonl"
    summary_path = tmp_path / "summary.json"
    export_replay_games_jsonl(input_path, [replay])

    summary = improve_replay_actions(
        ImproveReplayConfig(
            input_paths=(input_path,),
            output_path=output_path,
            summary_path=summary_path,
            search_depth=1,
            search_max_branching=4,
            search_buy_branching=2,
            search_reserve_branching=1,
            search_take_branching=1,
            write_unchanged=True,
            max_rows=5,
        )
    )

    assert summary.rows_read == 5
    assert summary.rows_written == 5
    assert summary.rows_changed_written + summary.rows_unchanged_written == 5
    assert summary.rows_missing_snapshot == 0
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    assert "original_action_index" in rows[0]
    assert "improvement_metadata" in rows[0]
    assert rows[0]["improvement_metadata"]["source"] == "shallow_search"
    assert summary_path.exists()


def test_improve_replay_actions_skips_rows_without_snapshots(tmp_path) -> None:
    input_path = tmp_path / "plain.jsonl"
    output_path = tmp_path / "improved.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps({"action_index": 1}) + "\n", encoding="utf-8")

    summary = improve_replay_actions(
        ImproveReplayConfig(
            input_paths=(input_path,),
            output_path=output_path,
            summary_path=summary_path,
        )
    )

    assert summary.rows_read == 1
    assert summary.rows_written == 0
    assert summary.rows_missing_snapshot == 1


def test_improve_replay_actions_applies_search_margin_gate(tmp_path, monkeypatch) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=10),
        bot_seat_1=RandomLegalBot(seed=11),
        seed=12,
        max_turns=4,
        include_state_snapshots=True,
    )
    input_path = tmp_path / "replays.jsonl"
    skipped_output_path = tmp_path / "skipped.jsonl"
    skipped_summary_path = tmp_path / "skipped_summary.json"
    accepted_output_path = tmp_path / "accepted.jsonl"
    accepted_summary_path = tmp_path / "accepted_summary.json"
    export_replay_games_jsonl(input_path, [replay])
    first_row = json.loads(input_path.read_text(encoding="utf-8").splitlines()[0])
    original_index = int(first_row["action_index"])
    codec = ActionCodec()

    class FakeSearchBot:
        def rank_actions(self, env, state, legal_actions):
            original_action = next(
                action for action in legal_actions if codec.encode(state, action) == original_index
            )
            replacement_action = next(
                action for action in legal_actions if codec.encode(state, action) != original_index
            )
            return (
                SimpleNamespace(
                    action=replacement_action,
                    value=10.0,
                    heuristic_score=10.0,
                    loop_penalty=0.0,
                ),
                SimpleNamespace(
                    action=original_action,
                    value=8.0,
                    heuristic_score=8.0,
                    loop_penalty=0.0,
                ),
            )

    monkeypatch.setattr(
        "splendor_ai.training.improve_replay_actions._build_search_bot",
        lambda _config, row_offset: FakeSearchBot(),
    )

    skipped = improve_replay_actions(
        ImproveReplayConfig(
            input_paths=(input_path,),
            output_path=skipped_output_path,
            summary_path=skipped_summary_path,
            min_search_margin=5.0,
            max_rows=1,
        )
    )
    accepted = improve_replay_actions(
        ImproveReplayConfig(
            input_paths=(input_path,),
            output_path=accepted_output_path,
            summary_path=accepted_summary_path,
            min_search_margin=1.0,
            max_rows=1,
        )
    )

    assert skipped.rows_changed == 1
    assert skipped.rows_written == 0
    assert skipped.rows_filtered_by_margin == 1
    assert accepted.rows_changed == 1
    assert accepted.rows_changed_written == 1
    row = json.loads(accepted_output_path.read_text(encoding="utf-8").strip())
    assert row["improvement_metadata"]["search_margin"] == 2.0


def test_improve_replay_actions_can_filter_changed_action_types(tmp_path, monkeypatch) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=20),
        bot_seat_1=RandomLegalBot(seed=21),
        seed=22,
        max_turns=4,
        include_state_snapshots=True,
    )
    input_path = tmp_path / "replays.jsonl"
    output_path = tmp_path / "filtered.jsonl"
    summary_path = tmp_path / "summary.json"
    export_replay_games_jsonl(input_path, [replay])
    first_row = json.loads(input_path.read_text(encoding="utf-8").splitlines()[0])
    state = deserialize_state_snapshot(first_row["state_snapshot"])
    legal_actions = SplendorEnv().legal_actions(state)
    codec = ActionCodec()
    original_index = int(first_row["action_index"])
    replacement_action = next(
        action for action in legal_actions if codec.encode(state, action) != original_index
    )

    class FakeSearchBot:
        def rank_actions(self, env, state, legal_actions):
            original_action = next(
                action for action in legal_actions if codec.encode(state, action) == original_index
            )
            return (
                SimpleNamespace(
                    action=replacement_action,
                    value=10.0,
                    heuristic_score=10.0,
                    loop_penalty=0.0,
                ),
                SimpleNamespace(
                    action=original_action,
                    value=1.0,
                    heuristic_score=1.0,
                    loop_penalty=0.0,
                ),
            )

    monkeypatch.setattr(
        "splendor_ai.training.improve_replay_actions._build_search_bot",
        lambda _config, row_offset: FakeSearchBot(),
    )

    summary = improve_replay_actions(
        ImproveReplayConfig(
            input_paths=(input_path,),
            output_path=output_path,
            summary_path=summary_path,
            exclude_changed_action_types=(replacement_action.action_type.name,),
            max_rows=1,
        )
    )

    assert summary.rows_changed == 1
    assert summary.rows_written == 0
    assert summary.rows_filtered_by_action_type == 1
