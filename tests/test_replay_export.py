import json

from splendor_ai.bots import GreedyHeuristicBot, RandomLegalBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training import (
    collect_game_replay,
    deserialize_state_snapshot,
    export_replay_games_jsonl,
)


class TokenOnlyBot:
    def choose_action(self, env, state, legal_actions=None):
        legal = legal_actions or env.legal_actions(state)
        for action in legal:
            if action.action_type.name == "TAKE_TOKENS":
                return action
        return legal[0] if legal else None


def test_collect_game_replay_records_legal_training_examples() -> None:
    codec = ActionCodec()
    replay = collect_game_replay(
        bot_seat_0=GreedyHeuristicBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=4,
        max_turns=400,
        codec=codec,
    )

    assert replay.turns > 0
    assert len(replay.steps) > 0
    assert len(replay.bot_seats) == 2
    assert "bank_tokens" in replay.final_state_snapshot
    assert replay.timed_out is False
    assert replay.loop_fallback_triggers_by_seat == (0, 0)
    assert replay.bot_metadata[0]["bot_class"] == "GreedyHeuristicBot"
    for step in replay.steps:
        assert len(step.observation_vector) == 256
        assert step.action_index in step.legal_action_indices
        assert step.action_payload["action_type"]
        assert step.final_value in (-1.0, 0.0, 1.0)
        if replay.winner is not None:
            expected = 1.0 if replay.winner == step.player_id else -1.0
            assert step.final_value == expected


def test_export_replay_games_jsonl_writes_one_line_per_step(tmp_path) -> None:
    replay = collect_game_replay(
        bot_seat_0=GreedyHeuristicBot(seed=3),
        bot_seat_1=RandomLegalBot(seed=4),
        seed=5,
        max_turns=400,
    )
    output_path = tmp_path / "replays.jsonl"

    export_replay_games_jsonl(output_path, [replay])

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(replay.steps)

    first_payload = json.loads(lines[0])
    assert first_payload["game_seed"] == replay.seed
    assert first_payload["game_bot_seats"] == list(replay.bot_seats)
    assert first_payload["game_bot_metadata"] == list(replay.bot_metadata)
    assert first_payload["game_loop_fallback_triggers_by_seat"] == [0, 0]
    assert first_payload["game_model_loop_fallback_triggers"] == 0
    assert first_payload["game_timed_out"] is False
    assert first_payload["game_termination_reason"] == "completed"
    assert len(first_payload["observation_vector"]) == 256
    assert first_payload["action_index"] in first_payload["legal_action_indices"]
    assert "action_payload" in first_payload
    assert "state_snapshot" not in first_payload


def test_collect_game_replay_can_record_reconstructable_state_snapshots(tmp_path) -> None:
    codec = ActionCodec()
    replay = collect_game_replay(
        bot_seat_0=GreedyHeuristicBot(seed=5),
        bot_seat_1=RandomLegalBot(seed=6),
        seed=7,
        max_turns=20,
        codec=codec,
        include_state_snapshots=True,
    )

    first_step = replay.steps[0]
    assert first_step.state_snapshot is not None
    reconstructed_state = deserialize_state_snapshot(first_step.state_snapshot)
    legal_actions = SplendorEnv().legal_actions(reconstructed_state)
    legal_indices = codec.legal_action_indices(reconstructed_state, legal_actions)

    assert reconstructed_state.turn_index == first_step.turn_index
    assert reconstructed_state.current_player == first_step.player_id
    assert first_step.action_index in legal_indices
    assert tuple(legal_indices) == first_step.legal_action_indices

    output_path = tmp_path / "replays.jsonl"
    export_replay_games_jsonl(output_path, [replay])
    first_payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_payload["state_snapshot"]["turn_index"] == first_step.turn_index


def test_collect_game_replay_applies_no_progress_cutoff() -> None:
    replay = collect_game_replay(
        bot_seat_0=TokenOnlyBot(),
        bot_seat_1=TokenOnlyBot(),
        seed=0,
        max_turns=400,
        no_progress_limit=6,
        repetition_limit=0,
    )

    assert replay.timed_out is True
    assert replay.termination_reason == "no_progress_cutoff"
    assert replay.turns == 6
