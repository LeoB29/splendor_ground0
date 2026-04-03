from splendor_ai.encoding import encode_public_observation
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card


def test_public_observation_hides_opponent_reserved_card_identity() -> None:
    env = SplendorEnv()
    state = env.initial_state()
    state.players[1].reserved_cards.append(
        Card(
            card_id="hidden-card",
            tier=1,
            bonus_color="blue",
            points=0,
            cost={"white": 1},
        )
    )

    observation = encode_public_observation(state, player_id=0)

    assert observation["opponent"]["reserved_count"] == 1
    assert "reserved_cards" not in observation["opponent"]


def test_public_observation_shows_own_reserved_card_identity() -> None:
    env = SplendorEnv()
    state = env.initial_state()
    state.players[0].reserved_cards.append(
        Card(
            card_id="own-card",
            tier=1,
            bonus_color="white",
            points=0,
            cost={"blue": 1},
        )
    )

    observation = encode_public_observation(state, player_id=0)

    assert observation["self"]["reserved_cards"] == ["own-card"]
