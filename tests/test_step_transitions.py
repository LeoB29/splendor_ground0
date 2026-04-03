from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, Noble


def _first_action(actions, action_type: ActionType):
    return next(action for action in actions if action.action_type == action_type)


def test_step_take_tokens_updates_bank_player_and_turn() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    action = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.TAKE_TOKENS
        and action.take_tokens == ("white", "blue", "green")
    )

    next_state = env.step(state, action)

    assert next_state.current_player == 1
    assert next_state.turn_index == 1
    assert next_state.players[0].tokens["white"] == 1
    assert next_state.players[0].tokens["blue"] == 1
    assert next_state.players[0].tokens["green"] == 1
    assert next_state.bank_tokens["white"] == 3
    assert next_state.bank_tokens["blue"] == 3
    assert next_state.bank_tokens["green"] == 3


def test_step_take_single_available_color_updates_bank_and_player() -> None:
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
    action = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.TAKE_TOKENS and action.take_tokens == ("red",)
    )

    next_state = env.step(state, action)

    assert next_state.current_player == 1
    assert next_state.turn_index == 1
    assert next_state.players[0].tokens["red"] == 1
    assert next_state.bank_tokens["red"] == 0


def test_step_pass_advances_turn_without_changing_resources() -> None:
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
    state.players[0].reserved_cards = [
        Card(card_id="r1", tier=1, bonus_color="white", points=0, cost={"blue": 10}),
        Card(card_id="r2", tier=1, bonus_color="blue", points=0, cost={"green": 10}),
        Card(card_id="r3", tier=1, bonus_color="green", points=0, cost={"red": 10}),
    ]

    action = _first_action(env.legal_actions(state), ActionType.PASS)
    next_state = env.step(state, action)

    assert next_state.current_player == 1
    assert next_state.turn_index == 1
    assert next_state.bank_tokens == state.bank_tokens
    assert next_state.players[0].tokens == state.players[0].tokens


def test_step_reserve_visible_moves_card_to_reserved_and_refills_market() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    original_market = list(state.visible_tier_cards[1])
    original_hidden_top = state.hidden_tier_decks[1][0]
    action = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.RESERVE_VISIBLE
        and action.tier == 1
        and action.market_index == 0
    )

    next_state = env.step(state, action)

    assert next_state.players[0].reserved_cards[0].card_id == original_market[0].card_id
    assert len(next_state.visible_tier_cards[1]) == 4
    assert next_state.visible_tier_cards[1][-1].card_id == original_hidden_top.card_id
    assert next_state.players[0].tokens["gold"] == 1
    assert next_state.bank_tokens["gold"] == 4
    assert next_state.deck_counts[1] == len(state.hidden_tier_decks[1]) - 1


def test_step_reserve_deck_takes_hidden_card_without_changing_visible_market() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    visible_before = [card.card_id for card in state.visible_tier_cards[2]]
    hidden_top = state.hidden_tier_decks[2][0]
    action = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.RESERVE_DECK and action.tier == 2
    )

    next_state = env.step(state, action)

    assert [card.card_id for card in next_state.visible_tier_cards[2]] == visible_before
    assert next_state.players[0].reserved_cards[0].card_id == hidden_top.card_id
    assert next_state.deck_counts[2] == len(state.hidden_tier_decks[2]) - 1


def test_step_buy_visible_spends_tokens_awards_bonus_and_refills_market() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    card = Card(
        card_id="buy-visible",
        tier=1,
        bonus_color="red",
        points=1,
        cost={"white": 1, "blue": 1},
    )
    replacement = Card(
        card_id="replacement",
        tier=1,
        bonus_color="green",
        points=0,
        cost={"red": 1},
    )
    state.visible_tier_cards = {1: [card], 2: [], 3: []}
    state.hidden_tier_decks = {1: [replacement], 2: [], 3: []}
    state.deck_counts = {1: 1, 2: 0, 3: 0}
    player = state.players[0]
    player.tokens["white"] = 1
    player.tokens["blue"] = 1

    action = _first_action(env.legal_actions(state), ActionType.BUY_VISIBLE)
    next_state = env.step(state, action)

    assert next_state.players[0].score == 1
    assert next_state.players[0].bonuses["red"] == 1
    assert next_state.players[0].tokens["white"] == 0
    assert next_state.players[0].tokens["blue"] == 0
    assert next_state.bank_tokens["white"] == 5
    assert next_state.bank_tokens["blue"] == 5
    assert next_state.visible_tier_cards[1][0].card_id == "replacement"


def test_step_buy_reserved_removes_reserved_card_and_awards_points() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    reserved_card = Card(
        card_id="reserved-card",
        tier=2,
        bonus_color="black",
        points=2,
        cost={"red": 2},
    )
    player = state.players[0]
    player.reserved_cards.append(reserved_card)
    player.tokens["red"] = 1
    player.tokens["gold"] = 1

    action = _first_action(env.legal_actions(state), ActionType.BUY_RESERVED)
    next_state = env.step(state, action)

    assert next_state.players[0].reserved_cards == []
    assert next_state.players[0].score == 2
    assert next_state.players[0].bonuses["black"] == 1


def test_buy_action_branches_on_multiple_claimable_nobles_and_step_applies_choice() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[0]
    player.bonuses.update({"white": 3, "blue": 3, "green": 2, "red": 0, "black": 0})
    player.tokens["red"] = 1
    buy_card = Card(
        card_id="noble-trigger",
        tier=1,
        bonus_color="green",
        points=0,
        cost={"red": 1},
    )
    state.visible_tier_cards = {1: [buy_card], 2: [], 3: []}
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}
    state.nobles = [
        Noble(noble_id="n1", points=3, requirement={"white": 3, "blue": 3, "green": 3}),
        Noble(noble_id="n2", points=3, requirement={"blue": 3, "green": 3}),
    ]

    actions = [action for action in env.legal_actions(state) if action.action_type == ActionType.BUY_VISIBLE]

    assert {action.noble_id for action in actions} == {"n1", "n2"}

    chosen_action = next(action for action in actions if action.noble_id == "n2")
    next_state = env.step(state, chosen_action)

    assert [noble.noble_id for noble in next_state.players[0].nobles] == ["n2"]
    assert [noble.noble_id for noble in next_state.nobles] == ["n1"]
    assert next_state.players[0].score == 3


def test_reaching_fifteen_starts_final_round_and_finishes_when_turn_returns_to_start_player() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player0 = state.players[0]
    player0.score = 14
    state.visible_tier_cards = {
        1: [Card(card_id="game-end", tier=1, bonus_color="white", points=1, cost={})],
        2: [],
        3: [],
    }
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}

    buy_action = _first_action(env.legal_actions(state), ActionType.BUY_VISIBLE)
    state_after_buy = env.step(state, buy_action)

    assert state_after_buy.pending_round_end is True
    assert state_after_buy.terminal is False
    assert state_after_buy.current_player == 1

    final_action = _first_action(env.legal_actions(state_after_buy), ActionType.TAKE_TOKENS)
    terminal_state = env.step(state_after_buy, final_action)

    assert terminal_state.terminal is True
    assert terminal_state.winner == 0


def test_terminal_tie_break_uses_fewest_purchased_cards() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    state.current_player = 1
    state.pending_round_end = True
    state.players[0].score = 15
    state.players[1].score = 15
    state.players[0].purchased_cards = [
        Card(card_id=f"p0-{i}", tier=1, bonus_color="white", points=0, cost={})
        for i in range(4)
    ]
    state.players[1].purchased_cards = [
        Card(card_id=f"p1-{i}", tier=1, bonus_color="blue", points=0, cost={})
        for i in range(5)
    ]

    final_action = _first_action(env.legal_actions(state), ActionType.TAKE_TOKENS)
    terminal_state = env.step(state, final_action)

    assert terminal_state.terminal is True
    assert terminal_state.winner == 0


def test_terminal_exact_post_tiebreak_tie_remains_unresolved() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    state.current_player = 1
    state.pending_round_end = True
    state.players[0].score = 15
    state.players[1].score = 15
    shared_cards = [
        Card(card_id=f"shared-{i}", tier=1, bonus_color="white", points=0, cost={})
        for i in range(4)
    ]
    state.players[0].purchased_cards = list(shared_cards)
    state.players[1].purchased_cards = list(shared_cards)

    final_action = _first_action(env.legal_actions(state), ActionType.TAKE_TOKENS)
    terminal_state = env.step(state, final_action)

    assert terminal_state.terminal is True
    assert terminal_state.winner is None
