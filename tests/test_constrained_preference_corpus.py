import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from splendor_ai.bots import RandomLegalBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import (
    collect_game_replay,
    deserialize_state_snapshot,
    export_replay_games_jsonl,
    serialize_action,
)
from splendor_ai.training.run_constrained_preference_corpus import (
    ConstrainedPreferenceCorpusConfig,
    _conversion_quality_reason,
    build_constrained_preference_corpus,
)


class _FakeScorer:
    def __init__(self, top_action_index: int) -> None:
        self.top_action_index = top_action_index

    def score_actions(self, state, legal_actions):
        codec = ActionCodec()
        top_action = next(
            action
            for action in legal_actions
            if codec.encode(state, action) == self.top_action_index
        )
        return (
            SimpleNamespace(
                action=top_action,
                action_index=self.top_action_index,
                score=1.0,
            ),
        )


def test_builds_scoring_buy_preference_and_champion_anchors(tmp_path, monkeypatch) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=31),
        bot_seat_1=RandomLegalBot(seed=32),
        seed=33,
        max_turns=70,
        include_state_snapshots=True,
    )
    env = SplendorEnv()
    codec = ActionCodec()
    source_row = None
    selected_action = None
    model_action = None
    for candidate in replay.steps:
        candidate_payload = asdict(candidate)
        state = deserialize_state_snapshot(candidate_payload["state_snapshot"])
        legal_actions = env.legal_actions(state)
        scoring_buys = [
            action
            for action in legal_actions
            if action.action_type in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}
            and _points_for_buy(state, action) > 0
        ]
        alternatives = [
            action
            for action in legal_actions
            if action not in scoring_buys
        ]
        if scoring_buys and alternatives:
            source_row = candidate_payload
            selected_action = scoring_buys[0]
            model_action = alternatives[0]
            break
    assert source_row is not None
    assert selected_action is not None
    assert model_action is not None

    selected_index = codec.encode(
        deserialize_state_snapshot(source_row["state_snapshot"]),
        selected_action,
    )
    model_index = codec.encode(
        deserialize_state_snapshot(source_row["state_snapshot"]),
        model_action,
    )
    source_row.update(
        {
            "action_index": selected_index,
            "action_payload": serialize_action(selected_action),
            "original_action_index": model_index,
            "original_action_payload": serialize_action(model_action),
            "search_action_scores": [
                {"action_index": selected_index, "value": 100.0},
                {"action_index": model_index, "value": 0.0},
            ],
        }
    )
    pgs_path = tmp_path / "pgs.jsonl"
    pgs_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")

    anchor_row = dict(source_row)
    anchor_row["game_seed"] = 44
    anchor_row["player_id"] = 0
    anchor_row["game_bot_metadata"] = [
        {"checkpoint_path": "outputs/legal_action_scorer_022_attention_distill/model.pt"},
        {},
    ]
    regression_path = tmp_path / "regression.jsonl"
    regression_path.write_text(json.dumps(anchor_row) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "splendor_ai.training.run_constrained_preference_corpus._load_policy_scorer",
        lambda config: _FakeScorer(model_index),
    )
    preference_path = tmp_path / "preferences.jsonl"
    anchor_path = tmp_path / "anchors.jsonl"
    summary = build_constrained_preference_corpus(
        ConstrainedPreferenceCorpusConfig(
            checkpoint=Path("unused.pt"),
            pgs_replay_paths=(pgs_path,),
            regression_replay_paths=(regression_path,),
            preference_output_path=preference_path,
            anchor_output_path=anchor_path,
            summary_path=tmp_path / "summary.json",
            min_search_margin=50.0,
            regression_game_seeds=(44,),
        )
    )

    preference = json.loads(preference_path.read_text(encoding="utf-8"))
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    assert summary["preferences"]["rows_written"] == 1
    assert preference["model_top_action_index"] == model_index
    assert preference["search_margin"] == 100.0
    assert preference["target_source"] == "constrained_pgs_preference"
    assert summary["anchors"]["rows_written"] == 1
    assert anchor["target_source"] == "champion_anti_regression_anchor"
    assert anchor["training_weight"] == 2.0
    assert "model_top_action_index" not in anchor


def _points_for_buy(state, action) -> int:
    if action.action_type == ActionType.BUY_VISIBLE:
        return state.visible_tier_cards[action.tier][action.market_index].points
    return state.players[state.current_player].reserved_cards[action.reserved_index].points


def test_token_correction_never_replaces_a_buy() -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=51),
        bot_seat_1=RandomLegalBot(seed=52),
        seed=53,
        max_turns=70,
        include_state_snapshots=True,
    )
    env = SplendorEnv()
    for step in replay.steps:
        state = deserialize_state_snapshot(step.state_snapshot)
        legal_actions = env.legal_actions(state)
        token_action = next(
            (action for action in legal_actions if action.action_type == ActionType.TAKE_TOKENS),
            None,
        )
        buy_action = next(
            (
                action
                for action in legal_actions
                if action.action_type in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}
            ),
            None,
        )
        if token_action is not None and buy_action is not None:
            assert (
                _conversion_quality_reason(env, state, token_action, buy_action)
                is None
            )
            return
    raise AssertionError("Expected a replay state with both token and buy actions.")
