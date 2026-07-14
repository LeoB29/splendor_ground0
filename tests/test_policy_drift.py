from types import SimpleNamespace

from splendor_ai.bots import RandomLegalBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.training.replay import collect_game_replay, export_replay_games_jsonl
from splendor_ai.training.run_policy_drift import PolicyDriftConfig, measure_policy_drift


class _IndexedScorer:
    def __init__(self, offset: int) -> None:
        self.offset = offset

    def score_actions(self, state, legal_actions):
        codec = ActionCodec()
        action = legal_actions[min(self.offset, len(legal_actions) - 1)]
        return (
            SimpleNamespace(
                action=action,
                action_index=codec.encode(state, action),
                score=1.0,
            ),
        )


def test_policy_drift_reports_top_action_changes(tmp_path, monkeypatch) -> None:
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=41),
        bot_seat_1=RandomLegalBot(seed=42),
        seed=43,
        max_turns=4,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "replay.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    scorers = iter((_IndexedScorer(0), _IndexedScorer(1)))
    monkeypatch.setattr(
        "splendor_ai.training.run_policy_drift._load_policy_scorer",
        lambda config: next(scorers),
    )

    summary = measure_policy_drift(
        PolicyDriftConfig(
            before_checkpoint=tmp_path / "before.pt",
            after_checkpoint=tmp_path / "after.pt",
            replay_paths=(replay_path,),
            output_path=tmp_path / "drift.json",
            log_every=0,
        )
    )

    assert summary["overall"]["rows_compared"] == 4
    assert summary["overall"]["top_action_agreement_rate"] == 0.0
    assert summary["overall"]["top_action_change_rate"] == 1.0
    assert sum(summary["transitions"].values()) == 4
