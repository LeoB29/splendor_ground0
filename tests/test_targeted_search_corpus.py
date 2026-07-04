import json

from splendor_ai.bots import RandomLegalBot
from splendor_ai.bots.search_bot import SearchActionScore
from splendor_ai.encoding import ActionCodec, OBSERVATION_SIZE
from splendor_ai.training.replay import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.run_policy_search_disagreement import ModelActionScore
from splendor_ai.training.run_targeted_search_corpus import (
    TargetedSearchCorpusConfig,
    build_targeted_search_corpus,
)


class FakePolicyScorer:
    def __init__(self) -> None:
        self.codec = ActionCodec()

    def score_actions(self, state, legal_actions):
        return (
            ModelActionScore(
                action=legal_actions[0],
                action_index=self.codec.encode(state, legal_actions[0]),
                score=10.0,
            ),
            ModelActionScore(
                action=legal_actions[1],
                action_index=self.codec.encode(state, legal_actions[1]),
                score=0.0,
            ),
        )


class FakeSearchBot:
    def rank_actions(self, env, state, legal_actions):
        return (
            SearchActionScore(
                action=legal_actions[1],
                value=500.0,
                heuristic_score=500.0,
                loop_penalty=0.0,
            ),
            SearchActionScore(
                action=legal_actions[0],
                value=0.0,
                heuristic_score=0.0,
                loop_penalty=0.0,
            ),
        )


def test_targeted_search_corpus_writes_search_relabel_rows(tmp_path, monkeypatch) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=2,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "replay.jsonl"
    output_path = tmp_path / "targeted.jsonl"
    summary_path = tmp_path / "summary.json"
    export_replay_games_jsonl(replay_path, [replay])

    monkeypatch.setattr(
        "splendor_ai.training.run_targeted_search_corpus._load_policy_scorer",
        lambda config: FakePolicyScorer(),
    )
    monkeypatch.setattr(
        "splendor_ai.training.run_targeted_search_corpus._build_search_bot",
        lambda config, row_offset: FakeSearchBot(),
    )

    summary = build_targeted_search_corpus(
        TargetedSearchCorpusConfig(
            checkpoint=tmp_path / "unused.pt",
            model_kind="legal_action_scorer",
            replay_paths=(replay_path,),
            output_path=output_path,
            summary_path=summary_path,
            min_search_margin=100.0,
            target_action_types=("TAKE_TOKENS", "RESERVE_VISIBLE", "RESERVE_DECK", "BUY_VISIBLE", "BUY_RESERVED"),
            late_buy_turn_min=0,
        )
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary["rows_written"] == 2
    assert len(rows) == 2
    assert rows[0]["target_source"] == "targeted_search_disagreement"
    assert rows[0]["action_index"] != rows[0]["original_action_index"]
    assert rows[0]["search_margin"] == 500.0
    assert len(rows[0]["observation_vector"]) == OBSERVATION_SIZE
    assert len(rows[0]["search_action_scores"]) == 2
    assert json.loads(summary_path.read_text(encoding="utf-8"))["rows_written"] == 2
