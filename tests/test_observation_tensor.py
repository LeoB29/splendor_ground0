from splendor_ai.encoding import encode_public_observation_tensor
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card


def test_observation_tensor_has_stable_length_and_sections() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()

    observation = encode_public_observation_tensor(state, player_id=0)

    assert len(observation.vector) == 256
    assert observation.sections == {
        "global": (0, 3),
        "bank": (3, 9),
        "decks": (9, 12),
        "nobles": (12, 33),
        "board": (33, 189),
        "self_summary": (189, 203),
        "self_reserved": (203, 242),
        "opponent_summary": (242, 256),
    }


def test_observation_tensor_hides_opponent_reserved_identity() -> None:
    env = SplendorEnv(seed=0)
    state_a = env.initial_state()
    state_b = env.initial_state()

    state_a.players[1].reserved_cards.append(
        Card(card_id="opp-a", tier=1, bonus_color="white", points=0, cost={"blue": 1})
    )
    state_b.players[1].reserved_cards.append(
        Card(card_id="opp-b", tier=3, bonus_color="black", points=5, cost={"white": 7})
    )

    observation_a = encode_public_observation_tensor(state_a, player_id=0)
    observation_b = encode_public_observation_tensor(state_b, player_id=0)

    assert observation_a.vector == observation_b.vector


def test_observation_tensor_exposes_own_reserved_identity_features() -> None:
    env = SplendorEnv(seed=0)
    state_a = env.initial_state()
    state_b = env.initial_state()

    state_a.players[0].reserved_cards.append(
        Card(card_id="own-a", tier=1, bonus_color="white", points=0, cost={"blue": 1})
    )
    state_b.players[0].reserved_cards.append(
        Card(card_id="own-b", tier=3, bonus_color="black", points=5, cost={"white": 7})
    )

    observation_a = encode_public_observation_tensor(state_a, player_id=0)
    observation_b = encode_public_observation_tensor(state_b, player_id=0)

    assert observation_a.vector != observation_b.vector
