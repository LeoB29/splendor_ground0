from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, Noble


def test_action_codec_has_stable_positive_action_space_size() -> None:
    codec = ActionCodec()

    assert codec.action_space_size > 0
    assert codec.action_space_size == 183961


def test_action_codec_round_trips_initial_state_legal_actions() -> None:
    env = SplendorEnv(seed=0)
    codec = ActionCodec()
    state = env.initial_state()
    legal_actions = env.legal_actions(state)

    indices = codec.legal_action_indices(state, legal_actions)

    assert len(indices) == len(legal_actions)
    assert len(set(indices)) == len(indices)
    decoded = [codec.decode(state, index) for index in indices]
    assert set(decoded) == set(legal_actions)


def test_action_mask_marks_exactly_the_legal_indices() -> None:
    env = SplendorEnv(seed=0)
    codec = ActionCodec()
    state = env.initial_state()
    legal_actions = env.legal_actions(state)

    indices = codec.legal_action_indices(state, legal_actions)
    mask = codec.legal_action_mask(state, legal_actions)

    assert len(mask) == codec.action_space_size
    assert sum(mask) == len(legal_actions)
    assert all(mask[index] for index in indices)


def test_codec_round_trips_buy_action_with_noble_slot_choice() -> None:
    env = SplendorEnv(seed=0)
    codec = ActionCodec()
    state = env.initial_state()
    player = state.players[0]
    player.bonuses.update({"white": 3, "blue": 3, "green": 2, "red": 0, "black": 0})
    player.tokens["red"] = 1
    state.visible_tier_cards = {
        1: [Card(card_id="noble-trigger", tier=1, bonus_color="green", points=0, cost={"red": 1})],
        2: [],
        3: [],
    }
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}
    state.nobles = [
        Noble(noble_id="custom-a", points=3, requirement={"white": 3, "blue": 3, "green": 3}),
        Noble(noble_id="custom-b", points=3, requirement={"blue": 3, "green": 3}),
    ]

    legal_actions = env.legal_actions(state)
    action = next(
        candidate
        for candidate in legal_actions
        if candidate.action_type == ActionType.BUY_VISIBLE and candidate.noble_id == "custom-b"
    )

    index = codec.encode(state, action)
    decoded = codec.decode(state, index)

    assert decoded == action


def test_codec_round_trips_buy_reserved_action() -> None:
    env = SplendorEnv(seed=0)
    codec = ActionCodec()
    state = env.initial_state()
    player = state.players[0]
    player.reserved_cards.append(
        Card(card_id="reserved-buy", tier=2, bonus_color="black", points=1, cost={"red": 2})
    )
    player.tokens["red"] = 1
    player.tokens["gold"] = 1

    action = next(
        candidate
        for candidate in env.legal_actions(state)
        if candidate.action_type == ActionType.BUY_RESERVED
    )

    index = codec.encode(state, action)
    decoded = codec.decode(state, index)

    assert decoded == action


def test_codec_round_trips_limited_distinct_take_actions() -> None:
    env = SplendorEnv(seed=0)
    codec = ActionCodec()
    state = env.initial_state()
    state.bank_tokens.update(
        {
            "white": 1,
            "blue": 0,
            "green": 1,
            "red": 0,
            "black": 0,
        }
    )

    legal_actions = env.legal_actions(state)
    action = next(
        candidate
        for candidate in legal_actions
        if candidate.action_type == ActionType.TAKE_TOKENS and candidate.take_tokens == ("white", "green")
    )

    index = codec.encode(state, action)
    decoded = codec.decode(state, index)

    assert decoded == action


def test_codec_round_trips_pass_action() -> None:
    env = SplendorEnv(seed=0)
    codec = ActionCodec()
    state = env.initial_state()
    state.bank_tokens.update(
        {
            "white": 0,
            "blue": 0,
            "green": 0,
            "red": 0,
            "black": 0,
        }
    )
    state.players[0].reserved_cards = [
        Card(card_id="r1", tier=1, bonus_color="white", points=0, cost={"blue": 10}),
        Card(card_id="r2", tier=1, bonus_color="blue", points=0, cost={"green": 10}),
        Card(card_id="r3", tier=1, bonus_color="green", points=0, cost={"red": 10}),
    ]

    action = next(
        candidate
        for candidate in env.legal_actions(state)
        if candidate.action_type == ActionType.PASS
    )

    index = codec.encode(state, action)
    decoded = codec.decode(state, index)

    assert decoded == action
