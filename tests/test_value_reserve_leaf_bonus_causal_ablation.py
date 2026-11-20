from __future__ import annotations

import subprocess
import sys

from splendor_ai.bots.value_reserve_search_bot import (
    FrozenFullActionSearchBot,
    ValueReserveLeafSearchBot,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_value_reserve_leaf_bonus_causal_ablation import causal_decision


class _FirstActionSearch:
    def __init__(self, captured: list[tuple[int, bool]], seed: int, with_bonus: bool):
        captured.append((seed, with_bonus))

    def choose_action(self, env, state, actions):  # noqa: ANN001, ARG002
        return actions[0]


def test_pure_ablation_uses_exact_frozen_decision_seed(monkeypatch) -> None:
    captured: list[tuple[int, bool]] = []

    def fake_search(seed: int, with_bonus: bool):
        return _FirstActionSearch(captured, seed, with_bonus)

    monkeypatch.setattr(ValueReserveLeafSearchBot, "_search", staticmethod(fake_search))
    env = SplendorEnv(seed=5)
    state = env.initial_state()
    bot = FrozenFullActionSearchBot(with_bonus=False)
    bot.start_game(772_000)
    action = bot.choose_action(env, state)

    assert action is not None
    assert captured == [(772_000 * 1_000_003 + 77, False)]


def test_bonus_dose_bot_needs_no_raw035_comparator(monkeypatch) -> None:
    class FakeSearch:
        def __init__(self, with_bonus: bool):
            self.with_bonus = with_bonus

        def choose_action(self, env, state, actions):  # noqa: ANN001, ARG002
            return actions[0] if self.with_bonus else actions[-1]

    monkeypatch.setattr(
        ValueReserveLeafSearchBot,
        "_search",
        staticmethod(lambda seed, with_bonus: FakeSearch(with_bonus)),
    )
    env = SplendorEnv(seed=8)
    state = env.initial_state()
    bot = ValueReserveLeafSearchBot(raw035_bot=None)
    bot.start_game(772_000)
    assert bot.choose_action(env, state) is not None
    assert bot.telemetry.decisions == 1
    assert bot.telemetry.moved_from_search_ablation == 1
    assert bot.telemetry.moved_from_raw035 == 0


def test_causal_decision_is_frozen_conjunction() -> None:
    integrity = {"all": True}
    assert causal_decision(
        integrity, {"mean": 0.55, "ci95_low": 0.505, "ci95_high": 0.60}
    ).endswith("causally_stronger_supports_principle")
    assert causal_decision(
        integrity, {"mean": 0.54, "ci95_low": 0.50, "ci95_high": 0.58}
    ).endswith("not_decisive_close_principle")
    assert causal_decision(
        integrity, {"mean": 0.44, "ci95_low": 0.39, "ci95_high": 0.49}
    ).endswith("causally_harmful_close_principle")
    assert causal_decision(
        {"all": False}, {"mean": 0.60, "ci95_low": 0.55, "ci95_high": 0.65}
    ).endswith("invalid_no_causal_decision")


def test_root_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "run_value_reserve_leaf_bonus_causal_ablation_078.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Value-reserve leaf bonus ablation" in result.stdout
