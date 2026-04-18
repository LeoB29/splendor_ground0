from splendor_ai.bots import GreedyHeuristicBot, RandomLegalBot, ShallowSearchBot
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.state import Card
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval import MatchConfig, play_game, play_match


def test_greedy_bot_prefers_immediate_scoring_buy() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[0]
    state.visible_tier_cards = {
        1: [Card(card_id="score-now", tier=1, bonus_color="red", points=1, cost={})],
        2: [],
        3: [],
    }
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}

    bot = GreedyHeuristicBot(seed=0)
    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert action.action_type == ActionType.BUY_VISIBLE
    assert action.market_index == 0


def test_greedy_bot_avoids_no_op_take_and_return_loop() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    state.visible_tier_cards = {
        1: [Card(card_id="hard-1", tier=1, bonus_color="white", points=0, cost={"blue": 10})],
        2: [],
        3: [],
    }
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}
    state.bank_tokens.update(
        {
            "white": 0,
            "blue": 0,
            "green": 0,
            "red": 2,
            "black": 0,
            "gold": 5,
        }
    )
    player = state.players[0]
    player.tokens.update(
        {
            "white": 1,
            "blue": 3,
            "green": 3,
            "red": 0,
            "black": 1,
            "gold": 0,
        }
    )
    player.reserved_cards = [
        Card(card_id="r1", tier=1, bonus_color="white", points=0, cost={"blue": 10}),
        Card(card_id="r2", tier=1, bonus_color="blue", points=0, cost={"green": 10}),
        Card(card_id="r3", tier=1, bonus_color="green", points=0, cost={"red": 10}),
    ]
    state.players[1].tokens.update(
        {
            "white": 0,
            "blue": 0,
            "green": 0,
            "red": 0,
            "black": 0,
            "gold": 0,
        }
    )

    bot = GreedyHeuristicBot(seed=0)
    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert action.action_type == ActionType.TAKE_TOKENS
    assert action.take_tokens == ("red",)
    assert action.return_tokens != ("red",)


def test_play_game_returns_terminal_result() -> None:
    result = play_game(
        bot_seat_0=GreedyHeuristicBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=400,
    )

    assert result.turns > 0
    assert result.turns <= 400
    assert len(result.final_scores) == 2


def test_search_bot_blocks_opponent_immediate_winning_buy() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    state.visible_tier_cards = {
        1: [
            Card(card_id="opp-win", tier=1, bonus_color="blue", points=1, cost={"white": 1}),
        ],
        2: [],
        3: [],
    }
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}
    state.players[0].score = 0
    state.players[1].score = 14
    state.players[0].reserved_cards = [
        Card(card_id="self-point", tier=1, bonus_color="white", points=1, cost={}),
    ]
    state.players[1].tokens["white"] = 1

    greedy_bot = GreedyHeuristicBot(seed=0)
    search_bot = ShallowSearchBot(depth=2, max_branching=12, seed=0)

    greedy_action = greedy_bot.choose_action(env, state, env.legal_actions(state))
    search_action = search_bot.choose_action(env, state, env.legal_actions(state))

    assert greedy_action is not None
    assert search_action is not None
    assert greedy_action.action_type == ActionType.BUY_RESERVED
    assert greedy_action.reserved_index == 0
    assert search_action.action_type == ActionType.RESERVE_VISIBLE
    assert search_action.market_index == 0


def test_search_bot_penalizes_repeated_take_return_loop_action() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    state.turn_index = 8
    state.visible_tier_cards = {
        1: [Card(card_id="hard-1", tier=1, bonus_color="white", points=0, cost={"blue": 10})],
        2: [],
        3: [],
    }
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}
    state.bank_tokens.update(
        {
            "white": 0,
            "blue": 0,
            "green": 0,
            "red": 2,
            "black": 0,
            "gold": 5,
        }
    )
    player = state.players[0]
    player.tokens.update(
        {
            "white": 2,
            "blue": 3,
            "green": 3,
            "red": 0,
            "black": 2,
            "gold": 0,
        }
    )
    player.reserved_cards = [
        Card(card_id="r1", tier=1, bonus_color="white", points=0, cost={"blue": 10}),
        Card(card_id="r2", tier=1, bonus_color="blue", points=0, cost={"green": 10}),
        Card(card_id="r3", tier=1, bonus_color="green", points=0, cost={"red": 10}),
    ]
    state.players[1].tokens.update(
        {
            "white": 0,
            "blue": 0,
            "green": 0,
            "red": 0,
            "black": 0,
            "gold": 0,
        }
    )

    legal_actions = env.legal_actions(state)
    loop_action = next(
        action
        for action in legal_actions
        if action.action_type == ActionType.TAKE_TOKENS
        and action.take_tokens == ("red",)
        and action.return_tokens == ("red",)
    )
    cleaner_action = next(
        action
        for action in legal_actions
        if action.action_type == ActionType.TAKE_TOKENS
        and action.take_tokens == ("red",)
        and action.return_tokens != ("red",)
    )

    search_bot = ShallowSearchBot(depth=2, max_branching=12, seed=0)
    root_signature = search_bot._heuristic.state_signature(state)
    search_bot._observed_state_counts[root_signature] = 1
    search_bot._last_turn_index = state.turn_index - 2
    loop_next_state = env.step(state, loop_action)
    search_bot._post_action_state_counts[search_bot._heuristic.state_signature(loop_next_state)] = 1

    loop_penalty = search_bot._candidate_loop_penalty(
        before=state,
        after=loop_next_state,
        action=loop_action,
        player_id=state.current_player,
    )
    clean_penalty = search_bot._candidate_loop_penalty(
        before=state,
        after=env.step(state, cleaner_action),
        action=cleaner_action,
        player_id=state.current_player,
    )

    assert loop_penalty > clean_penalty
    chosen_action = search_bot.choose_action(env, state, legal_actions)
    assert chosen_action is not None
    assert not (
        chosen_action.action_type == ActionType.TAKE_TOKENS
        and chosen_action.take_tokens == ("red",)
        and chosen_action.return_tokens == ("red",)
    )


def test_search_bot_resets_loop_history_on_new_game() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    search_bot = ShallowSearchBot(seed=0)
    search_bot._observed_state_counts[("stale",)] = 3
    search_bot._post_action_state_counts[("stale-next",)] = 2
    search_bot._last_turn_index = 12

    action = search_bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert ("stale",) not in search_bot._observed_state_counts
    assert ("stale-next",) not in search_bot._post_action_state_counts


def test_play_match_swaps_seats_and_collects_results() -> None:
    result = play_match(
        bot_a_factory=lambda: GreedyHeuristicBot(seed=1),
        bot_b_factory=lambda: RandomLegalBot(seed=2),
        config=MatchConfig(games=4, swap_seats=True, max_turns_per_game=400),
        seed_start=10,
    )

    assert result.total_games == 4
    assert result.games[0].bot_seats != result.games[1].bot_seats
    assert sum(result.wins_by_seat) + result.draws == 4
