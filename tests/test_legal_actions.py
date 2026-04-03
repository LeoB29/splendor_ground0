from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card


def _actions_of_type(actions, action_type: ActionType):
    return [action for action in actions if action.action_type == action_type]


def test_initial_state_has_expected_legal_action_count() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()

    actions = env.legal_actions(state)

    assert len(actions) == 30
    assert len(_actions_of_type(actions, ActionType.TAKE_TOKENS)) == 15
    assert len(_actions_of_type(actions, ActionType.RESERVE_VISIBLE)) == 12
    assert len(_actions_of_type(actions, ActionType.RESERVE_DECK)) == 3
    assert len(_actions_of_type(actions, ActionType.BUY_VISIBLE)) == 0
    assert len(_actions_of_type(actions, ActionType.BUY_RESERVED)) == 0


def test_take_two_same_color_requires_four_tokens_in_bank() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    state.bank_tokens["white"] = 3

    actions = env.legal_actions(state)
    take_actions = _actions_of_type(actions, ActionType.TAKE_TOKENS)

    assert all(action.take_tokens != ("white", "white") for action in take_actions)


def test_take_distinct_can_fall_back_to_two_colors_when_only_two_are_available() -> None:
    env = SplendorEnv(seed=0)
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

    take_actions = _actions_of_type(env.legal_actions(state), ActionType.TAKE_TOKENS)

    assert [action.take_tokens for action in take_actions] == [("white", "green")]


def test_take_distinct_can_fall_back_to_one_color_when_only_one_is_available() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    state.bank_tokens.update(
        {
            "white": 0,
            "blue": 0,
            "green": 0,
            "red": 1,
            "black": 0,
        }
    )

    take_actions = _actions_of_type(env.legal_actions(state), ActionType.TAKE_TOKENS)

    assert [action.take_tokens for action in take_actions] == [("red",)]


def test_take_actions_include_all_legal_return_options_when_over_token_limit() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[state.current_player]

    player.tokens.update(
        {
            "white": 8,
            "blue": 0,
            "green": 0,
            "red": 0,
            "black": 0,
            "gold": 0,
        }
    )
    state.bank_tokens.update(
        {
            "white": 0,
            "blue": 1,
            "green": 1,
            "red": 1,
            "black": 0,
            "gold": 5,
        }
    )

    actions = env.legal_actions(state)
    relevant = [
        action
        for action in _actions_of_type(actions, ActionType.TAKE_TOKENS)
        if action.take_tokens == ("blue", "green", "red")
    ]

    assert {action.return_tokens for action in relevant} == {
        ("white",),
        ("blue",),
        ("green",),
        ("red",),
    }


def test_reserve_actions_are_blocked_when_player_has_three_reserved_cards() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[state.current_player]
    player.reserved_cards = [
        Card(card_id="r1", tier=1, bonus_color="white", points=0, cost={"blue": 1}),
        Card(card_id="r2", tier=1, bonus_color="blue", points=0, cost={"green": 1}),
        Card(card_id="r3", tier=1, bonus_color="green", points=0, cost={"red": 1}),
    ]

    actions = env.legal_actions(state)

    assert _actions_of_type(actions, ActionType.RESERVE_VISIBLE) == []
    assert _actions_of_type(actions, ActionType.RESERVE_DECK) == []


def test_reserve_gold_gain_can_force_token_return_choices() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[state.current_player]

    player.tokens.update(
        {
            "white": 10,
            "blue": 0,
            "green": 0,
            "red": 0,
            "black": 0,
            "gold": 0,
        }
    )
    state.visible_tier_cards = {1: [state.visible_tier_cards[1][0]], 2: [], 3: []}
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}

    actions = env.legal_actions(state)
    reserve_actions = _actions_of_type(actions, ActionType.RESERVE_VISIBLE)

    assert len(reserve_actions) == 2
    assert {action.return_tokens for action in reserve_actions} == {
        ("white",),
        ("gold",),
    }


def test_buy_visible_uses_bonuses_and_gold_to_cover_missing_cost() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[state.current_player]

    state.visible_tier_cards = {
        1: [
            Card(
                card_id="buy-me",
                tier=1,
                bonus_color="red",
                points=0,
                cost={"white": 2, "blue": 1, "green": 1},
            )
        ],
        2: [],
        3: [],
    }
    player.tokens.update(
        {
            "white": 1,
            "blue": 1,
            "green": 0,
            "red": 0,
            "black": 0,
            "gold": 1,
        }
    )
    player.bonuses["green"] = 1

    actions = env.legal_actions(state)
    buy_actions = _actions_of_type(actions, ActionType.BUY_VISIBLE)

    assert len(buy_actions) == 1
    assert buy_actions[0].tier == 1
    assert buy_actions[0].market_index == 0
    assert buy_actions[0].spend_tokens == ("white", "blue", "gold")


def test_buy_reserved_is_generated_for_affordable_reserved_card() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[state.current_player]

    player.reserved_cards.append(
        Card(
            card_id="reserved-buy",
            tier=2,
            bonus_color="black",
            points=1,
            cost={"red": 2},
        )
    )
    player.tokens["red"] = 1
    player.tokens["gold"] = 1

    actions = env.legal_actions(state)
    buy_actions = _actions_of_type(actions, ActionType.BUY_RESERVED)

    assert len(buy_actions) == 1
    assert buy_actions[0].reserved_index == 0
    assert buy_actions[0].spend_tokens == ("red", "gold")


def test_pass_is_generated_when_no_other_actions_are_legal() -> None:
    env = SplendorEnv(seed=0)
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
    acting_player = state.players[state.current_player]
    acting_player.reserved_cards = [
        Card(card_id="r1", tier=1, bonus_color="white", points=0, cost={"blue": 10}),
        Card(card_id="r2", tier=1, bonus_color="blue", points=0, cost={"green": 10}),
        Card(card_id="r3", tier=1, bonus_color="green", points=0, cost={"red": 10}),
    ]

    actions = env.legal_actions(state)

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.PASS
