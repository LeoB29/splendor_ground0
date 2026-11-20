from __future__ import annotations

from splendor_ai.gui import _GuiGameRecorder


class _ResolverLikeBot:
    replay_metadata = {
        "checkpoint_sha256": "frozen",
        "continuation_workers": 10,
    }
    telemetry = {"decisions": 7, "switches": 3}


def test_gui_recorder_carries_resolver_identity_for_replay_rows() -> None:
    messages = []
    recorder = _GuiGameRecorder(messages.append)
    bot = _ResolverLikeBot()

    recorder.start_game(
        seed=800_000,
        human_seat=0,
        opponent_kind="resolver",
        opponent_bot=bot,
        model_path="",
    )

    opponent = recorder._game_info["bot_metadata"][1]
    assert opponent["gui_opponent_kind"] == "resolver"
    assert opponent["checkpoint_sha256"] == "frozen"
    assert opponent["continuation_workers"] == 10
    assert recorder._opponent_bot is bot
