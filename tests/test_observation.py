from splendor_ai.encoding import encode_public_observation
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import (
    Card,
    RESERVE_SOURCE_DECK,
    RESERVE_SOURCE_VISIBLE,
)


def test_public_observation_shows_opponent_visible_reserved_card_identity() -> None:
    env = SplendorEnv()
    state = env.initial_state()
    state.players[1].reserve_card(
        Card(
            card_id="hidden-card",
            tier=1,
            bonus_color="blue",
            points=0,
            cost={"white": 1},
        ),
        RESERVE_SOURCE_VISIBLE,
    )

    observation = encode_public_observation(state, player_id=0)

    assert observation["opponent"]["reserved_count"] == 1
    assert observation["opponent"]["reserved_cards"] == ["hidden-card"]


def test_public_observation_hides_opponent_deck_reserved_card_identity() -> None:
    env = SplendorEnv()
    state = env.initial_state()
    state.players[1].reserve_card(
        Card(
            card_id="private-card",
            tier=1,
            bonus_color="blue",
            points=0,
            cost={"white": 1},
        ),
        RESERVE_SOURCE_DECK,
    )

    observation = encode_public_observation(state, player_id=0)

    assert observation["opponent"]["reserved_count"] == 1
    assert observation["opponent"]["reserved_cards"] == [None]
    assert observation["opponent"]["reserved_sources"] == ["deck"]


def test_public_observation_shows_own_reserved_card_identity() -> None:
    env = SplendorEnv()
    state = env.initial_state()
    state.players[0].reserve_card(
        Card(
            card_id="own-card",
            tier=1,
            bonus_color="white",
            points=0,
            cost={"blue": 1},
        ),
        RESERVE_SOURCE_DECK,
    )

    observation = encode_public_observation(state, player_id=0)

    assert observation["self"]["reserved_cards"] == ["own-card"]
