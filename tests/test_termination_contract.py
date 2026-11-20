from __future__ import annotations

from pathlib import Path

import pytest

from splendor_ai.alphazero.network import AlphaZeroInference, create_initial_checkpoint
from splendor_ai.alphazero.repeated_return_value_ab import (
    _play_continuation,
    search_config,
)
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig
from splendor_ai.diagnostics import state_signature
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card
from splendor_ai.termination import (
    adjudicate_scores,
    forced_pass_deadlock_successor,
    is_forced_pass_turn,
)


def _forced_pass_state():
    state = SplendorEnv(seed=0).initial_state()
    state.bank_tokens.update({color: 0 for color in ALL_TOKEN_TYPES})
    state.players[0].score = 2
    for player_id, player in enumerate(state.players):
        player.reserved_cards = [
            Card(
                card_id=f"forced-{player_id}-{index}",
                tier=1,
                bonus_color="white",
                points=0,
                cost={"blue": 10},
            )
            for index in range(3)
        ]
    return state


def test_shared_forced_pass_contract_is_a_two_pass_fixed_point() -> None:
    env = SplendorEnv()
    state = _forced_pass_state()
    legal = env.legal_actions(state)
    assert is_forced_pass_turn(legal)

    after_first = env.step(state, legal[0])
    after_second = forced_pass_deadlock_successor(env, after_first)

    assert after_second is not None
    assert state_signature(after_second) == state_signature(state)
    assert adjudicate_scores(after_second) == 0


def test_alphazero_search_backs_up_adjudicated_deadlock_value(tmp_path: Path) -> None:
    checkpoint = tmp_path / "generation_000.pt"
    create_initial_checkpoint(checkpoint, seed=132)
    env = SplendorEnv()
    state = _forced_pass_state()

    result = AlphaZeroSearch(
        AlphaZeroInference(checkpoint),
        SearchConfig(
            simulations=8,
            max_depth=8,
            mode="debug",
            root_noise_fraction=0.0,
        ),
    ).run(env, state, self_play=False)

    assert len(result.root_edges) == 1
    assert result.root_edges[0].visits == 8
    assert result.root_edges[0].q == pytest.approx(1.0)
    assert result.search_root_value == pytest.approx(1.0)
    assert result.max_depth_reached == 0


def test_repeated_return_continuation_records_adjudicated_deadlock(tmp_path: Path) -> None:
    checkpoint = tmp_path / "generation_000.pt"
    create_initial_checkpoint(checkpoint, seed=133)
    inference = AlphaZeroInference(checkpoint)
    env = SplendorEnv()
    state = _forced_pass_state()
    action = env.legal_actions(state)[0]

    result = _play_continuation(
        initial_state=state,
        root_player=0,
        first_action_key=canonical_successor_action_key(state, action),
        inference=inference,
        state_index=0,
        world_slot=0,
        policy_slot=0,
        max_turns=20,
    )

    assert result["termination_reason"] == "forced_pass_deadlock"
    assert result["turns"] == 2
    assert result["winner"] == 0
    assert result["wdl"] == 1.0
    assert result["final_scores"] == [2, 0]
