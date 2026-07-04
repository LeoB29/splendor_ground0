from splendor_ai.encoding import (
    ACTION_FEATURE_SIZE,
    ActionFeatureTensor,
    encode_action_features,
    encode_legal_action_features,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, Noble


def _section(features: ActionFeatureTensor, section_name: str) -> tuple[float, ...]:
    start, end = features.sections[section_name]
    return features.vector[start:end]


def test_action_features_have_stable_length_and_sections() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    action = env.legal_actions(state)[0]

    features = encode_action_features(state, action)

    assert len(features.vector) == ACTION_FEATURE_SIZE
    assert ACTION_FEATURE_SIZE == 79
    assert features.sections == {
        "action_type": (0, 6),
        "location": (6, 16),
        "target_card": (16, 29),
        "chosen_noble": (29, 36),
        "token_flow": (36, 60),
        "derived": (60, 64),
        "opponent_context": (64, 79),
    }


def test_legal_action_features_preserve_legal_action_order() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)

    encoded_actions = encode_legal_action_features(state, legal_actions)

    assert len(encoded_actions) == len(legal_actions)
    assert all(len(features.vector) == ACTION_FEATURE_SIZE for features in encoded_actions)


def test_take_action_features_encode_token_flow() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    action = next(
        candidate
        for candidate in env.legal_actions(state)
        if candidate.action_type == ActionType.TAKE_TOKENS
        and candidate.take_tokens == ("white", "blue", "green")
    )

    features = encode_action_features(state, action)
    action_type = _section(features, "action_type")
    token_flow = _section(features, "token_flow")

    assert action_type == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert token_flow[:6] == (0.25, 0.25, 0.25, 0.0, 0.0, 0.0)
    assert token_flow[6:12] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert token_flow[18:24] == (0.25, 0.25, 0.25, 0.0, 0.0, 0.0)


def test_visible_card_action_features_include_target_card_and_location() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    action = next(
        candidate
        for candidate in env.legal_actions(state)
        if candidate.action_type == ActionType.RESERVE_VISIBLE
        and candidate.tier == 2
        and candidate.market_index == 1
    )
    target_card = state.visible_tier_cards[2][1]

    features = encode_action_features(state, action)

    assert _section(features, "action_type") == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert _section(features, "location") == (
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert _section(features, "target_card") == (
        1.0,
        target_card.tier / 3.0,
        target_card.points / 5.0,
        *(1.0 if target_card.bonus_color == color else 0.0 for color in ("white", "blue", "green", "red", "black")),
        *(target_card.cost.get(color, 0) / 7.0 for color in ("white", "blue", "green", "red", "black")),
    )


def test_reserve_deck_features_do_not_expose_hidden_card_identity() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    action = next(
        candidate
        for candidate in env.legal_actions(state)
        if candidate.action_type == ActionType.RESERVE_DECK and candidate.tier == 3
    )

    features = encode_action_features(state, action)

    assert _section(features, "action_type") == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert _section(features, "target_card") == (0.0,) * 13
    assert _section(features, "derived")[2] == state.deck_counts[3] / 16.0


def test_buy_reserved_action_features_include_own_reserved_target() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    reserved_card = Card(
        card_id="reserved-buy",
        tier=2,
        bonus_color="black",
        points=1,
        cost={"red": 2},
    )
    state.players[0].reserved_cards.append(reserved_card)
    state.players[0].tokens["red"] = 1
    state.players[0].tokens["gold"] = 1
    action = next(
        candidate
        for candidate in env.legal_actions(state)
        if candidate.action_type == ActionType.BUY_RESERVED
    )

    features = encode_action_features(state, action)

    assert _section(features, "action_type") == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert _section(features, "location")[-3:] == (1.0, 0.0, 0.0)
    assert _section(features, "target_card")[:3] == (1.0, 2.0 / 3.0, 1.0 / 5.0)
    assert _section(features, "derived")[:2] == (2.0 / 7.0, 1.0 / 8.0)


def test_buy_action_features_include_chosen_noble() -> None:
    env = SplendorEnv(seed=0)
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

    action = next(
        candidate
        for candidate in env.legal_actions(state)
        if candidate.action_type == ActionType.BUY_VISIBLE
        and candidate.noble_id == "custom-b"
    )
    features = encode_action_features(state, action)

    assert _section(features, "chosen_noble") == (
        1.0,
        1.0,
        0.0,
        3.0 / 4.0,
        3.0 / 4.0,
        0.0,
        0.0,
    )
    assert _section(features, "derived")[1] == 3.0 / 8.0
