from dataclasses import asdict

import torch

from splendor_ai.bots import (
    CheckpointValueEvaluator,
    GreedyHeuristicBot,
    RandomLegalBot,
    ShallowSearchBot,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.state import Card
from splendor_ai.engine.state import RESERVE_SOURCE_DECK
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval import MatchConfig, play_game, play_match
from splendor_ai.training.model import ValueMLP, ValueModelConfig


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


def test_search_bot_can_blend_value_evaluator_at_leaf() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()

    class ConstantValueEvaluator:
        def evaluate_state(self, state, player_id):
            return 100.0

    heuristic_bot = ShallowSearchBot(seed=0)
    value_bot = ShallowSearchBot(
        seed=0,
        value_evaluator=ConstantValueEvaluator(),
        value_weight=0.25,
    )

    heuristic_value = heuristic_bot._heuristic.evaluate_state(state, state.current_player)
    blended_value = value_bot._evaluate_leaf(state, state.current_player)

    assert blended_value == 0.75 * heuristic_value + 25.0


def test_checkpoint_value_evaluator_can_use_outcome_head(tmp_path) -> None:
    model = ValueMLP(ValueModelConfig(hidden_size=8, depth=1))
    for parameter in model.parameters():
        parameter.data.zero_()
    model.outcome_head.bias.data = torch.tensor([-1.0, 0.0, 1.0])
    checkpoint_path = tmp_path / "value.pt"
    torch.save(
        {
            "model_kind": "value_mlp",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    state = SplendorEnv(seed=0).initial_state()
    evaluator = CheckpointValueEvaluator(
        checkpoint_path=checkpoint_path,
        prediction_mode="outcome",
        value_scale=100.0,
    )

    value = evaluator.evaluate_state(state, player_id=0)
    probabilities = torch.softmax(torch.tensor([-1.0, 0.0, 1.0]), dim=0)
    expected = float((probabilities[2] - probabilities[0]).item()) * 100.0

    assert abs(value - expected) < 1e-5


def test_search_bot_can_gate_value_evaluator_by_turn_index() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()

    class ConstantValueEvaluator:
        def evaluate_state(self, state, player_id):
            return 100.0

    gated_bot = ShallowSearchBot(
        seed=0,
        value_evaluator=ConstantValueEvaluator(),
        value_weight=0.25,
        value_min_turn_index=40,
    )

    heuristic_value = gated_bot._heuristic.evaluate_state(state, state.current_player)
    assert gated_bot._evaluate_leaf(state, state.current_player) == heuristic_value

    state.turn_index = 40
    assert gated_bot._evaluate_leaf(state, state.current_player) == 0.75 * heuristic_value + 25.0


def test_information_legal_search_ignores_true_deck_order() -> None:
    env = SplendorEnv(seed=12)
    state_a = env.initial_state()
    state_b = state_a.copy_shallow()
    for tier in (1, 2, 3):
        state_b.hidden_tier_decks[tier].reverse()
    bot_a = ShallowSearchBot(
        seed=13,
        information_legal=True,
        determinizations=3,
        determinization_seed=14,
    )
    bot_b = ShallowSearchBot(
        seed=13,
        information_legal=True,
        determinizations=3,
        determinization_seed=14,
    )

    action_a = bot_a.choose_action(env, state_a, env.legal_actions(state_a))
    action_b = bot_b.choose_action(env, state_b, env.legal_actions(state_b))

    assert action_a == action_b


def test_information_legal_search_ignores_opponent_private_reserve_identity() -> None:
    env = SplendorEnv(seed=15)
    state_a = env.initial_state()
    private_card = state_a.hidden_tier_decks[1].pop()
    state_a.deck_counts[1] -= 1
    state_a.players[1].reserve_card(private_card, RESERVE_SOURCE_DECK)
    state_b = state_a.copy_shallow()
    replacement = state_b.hidden_tier_decks[1][0]
    state_b.hidden_tier_decks[1][0] = state_b.players[1].reserved_cards[0]
    state_b.players[1].reserved_cards[0] = replacement
    bot_a = ShallowSearchBot(
        seed=16,
        information_legal=True,
        determinizations=3,
        determinization_seed=17,
    )
    bot_b = ShallowSearchBot(
        seed=16,
        information_legal=True,
        determinizations=3,
        determinization_seed=17,
    )

    action_a = bot_a.choose_action(env, state_a, env.legal_actions(state_a))
    action_b = bot_b.choose_action(env, state_b, env.legal_actions(state_b))

    assert action_a == action_b


def test_search_bot_can_use_action_prior_for_ordering() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    preferred_action = next(
        action
        for action in legal_actions
        if action.action_type == ActionType.RESERVE_VISIBLE
    )

    class PreferredActionPrior:
        def score_actions(self, state, legal_actions):
            return {
                action: 1.0 if action == preferred_action else 0.0
                for action in legal_actions
            }

    search_bot = ShallowSearchBot(
        seed=0,
        max_branching=5,
        action_prior=PreferredActionPrior(),
        action_prior_weight=1_000_000.0,
    )

    candidates = search_bot._ordered_candidates(
        env=env,
        state=state,
        actions=legal_actions,
        root_player=state.current_player,
    )

    assert candidates[0].action == preferred_action
    assert candidates[0].policy_score == 1.0


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
