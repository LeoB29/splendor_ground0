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
            changed_action_weight=2.0,
            unchanged_action_weight=0.5,
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
    assert "search_action_scores" in rows[0]
    assert rows[0]["search_action_scores"]
    assert rows[0]["search_action_scores"][0]["rank"] == 1
    assert "action_index" in rows[0]["search_action_scores"][0]
    assert "value" in rows[0]["search_action_scores"][0]
    assert "improvement_metadata" in rows[0]
    assert rows[0]["improvement_metadata"]["source"] == "shallow_search"
    assert rows[0]["training_weight"] in (0.5, 2.0)
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
        lambda _config, row_offset, action_prior=None, value_evaluator=None: FakeSearchBot(),
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
            changed_action_weight=2.5,
        )
    )

    assert skipped.rows_changed == 1
    assert skipped.rows_written == 0
    assert skipped.rows_filtered_by_margin == 1
    assert accepted.rows_changed == 1
    assert accepted.rows_changed_written == 1
    row = json.loads(accepted_output_path.read_text(encoding="utf-8").strip())
    assert row["improvement_metadata"]["search_margin"] == 2.0
    assert row["training_weight"] == 2.5


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
        lambda _config, row_offset, action_prior=None, value_evaluator=None: FakeSearchBot(),
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


def test_improve_replay_actions_can_filter_by_player_id(tmp_path) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=30),
        bot_seat_1=RandomLegalBot(seed=31),
        seed=32,
        max_turns=6,
        include_state_snapshots=True,
    )
    input_path = tmp_path / "replays.jsonl"
    output_path = tmp_path / "seat1.jsonl"
    summary_path = tmp_path / "summary.json"
    export_replay_games_jsonl(input_path, [replay])
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
    ]
    seat1_rows = sum(1 for row in rows if int(row["player_id"]) == 1)

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
            player_id_filter=(1,),
        )
    )

    assert summary.rows_written == seat1_rows
    assert summary.rows_filtered_by_player_id == len(rows) - seat1_rows
    written_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written_rows
    assert {int(row["player_id"]) for row in written_rows} == {1}


def test_improve_replay_actions_can_filter_by_turn_and_action_type(tmp_path) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=35),
        bot_seat_1=RandomLegalBot(seed=36),
        seed=37,
        max_turns=8,
        include_state_snapshots=True,
    )
    input_path = tmp_path / "replays.jsonl"
    output_path = tmp_path / "filtered.jsonl"
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
            max_turn_index=3,
            action_type_filter=("TAKE_TOKENS", "RESERVE_VISIBLE", "RESERVE_DECK"),
            action_type_filter_scope="either",
        )
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary.rows_written == len(rows)
    assert summary.rows_filtered_by_turn > 0
    assert rows
    for row in rows:
        original_action_type = row["original_action_payload"]["action_type"]
        selected_action_type = row["action_payload"]["action_type"]
        assert int(row["turn_index"]) <= 3
        assert (
            original_action_type in {"TAKE_TOKENS", "RESERVE_VISIBLE", "RESERVE_DECK"}
            or selected_action_type in {"TAKE_TOKENS", "RESERVE_VISIBLE", "RESERVE_DECK"}
        )


def test_improve_replay_actions_reuses_policy_prior(monkeypatch, tmp_path) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=40),
        bot_seat_1=RandomLegalBot(seed=41),
        seed=42,
        max_turns=6,
        include_state_snapshots=True,
    )
    input_path = tmp_path / "replays.jsonl"
    output_path = tmp_path / "pgs.jsonl"
    summary_path = tmp_path / "summary.json"
    checkpoint_path = tmp_path / "prior.pt"
    checkpoint_path.write_text("unused", encoding="utf-8")
    export_replay_games_jsonl(input_path, [replay])
    constructed = []
    seen_prior_ids = []
    codec = ActionCodec()

    class FakePrior:
        def __init__(self, checkpoint_path, device):
            constructed.append((checkpoint_path, device))

    class FakeSearchBot:
        def __init__(self, action_prior):
            seen_prior_ids.append(id(action_prior))

        def rank_actions(self, env, state, legal_actions):
            action = legal_actions[0]
            return (
                SimpleNamespace(
                    action=action,
                    value=1.0,
                    heuristic_score=1.0,
                    loop_penalty=0.0,
                ),
            )

    def fake_build_search_bot(config, row_offset, action_prior=None, value_evaluator=None):
        return FakeSearchBot(action_prior)

    monkeypatch.setattr(
        "splendor_ai.training.improve_replay_actions.CheckpointLegalActionPriorScorer",
        FakePrior,
    )
    monkeypatch.setattr(
        "splendor_ai.training.improve_replay_actions._build_search_bot",
        fake_build_search_bot,
    )

    summary = improve_replay_actions(
        ImproveReplayConfig(
            input_paths=(input_path,),
            output_path=output_path,
            summary_path=summary_path,
            policy_prior_checkpoint_path=checkpoint_path,
            policy_prior_device="cpu",
            policy_prior_weight=3000.0,
            write_unchanged=True,
            max_rows=3,
        )
    )

    assert summary.rows_read == 3
    assert summary.rows_written == 3
    assert constructed == [(checkpoint_path, "cpu")]
    assert len(set(seen_prior_ids)) == 1
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["improvement_metadata"]["source"] == "policy_guided_search"
    assert rows[0]["improvement_metadata"]["policy_prior_weight"] == 3000.0
