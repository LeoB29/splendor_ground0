from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card


def _pick_action(actions: list[Action], **criteria) -> Action:
    for action in actions:
        if all(getattr(action, key) == value for key, value in criteria.items()):
            return action
    raise AssertionError(f"No action found for criteria: {criteria}")


def test_replay_seeded_opening_sequence_from_initial_state() -> None:
    env = SplendorEnv(seed=0)
    state0 = env.initial_state()

    action0 = _pick_action(
        env.legal_actions(state0),
        action_type=ActionType.TAKE_TOKENS,
        take_tokens=("white", "blue", "green"),
        return_tokens=(),
    )
    state1 = env.step(state0, action0)

    assert state1.current_player == 1
    assert state1.turn_index == 1
    assert state1.players[0].token_count == 3

    action1 = _pick_action(
        env.legal_actions(state1),
        action_type=ActionType.RESERVE_VISIBLE,
        tier=1,
        market_index=0,
        take_tokens=("gold",),
        return_tokens=(),
    )
    reserved_card_id = state1.visible_tier_cards[1][0].card_id
    hidden_replacement_id = state1.hidden_tier_decks[1][0].card_id
    state2 = env.step(state1, action1)

    assert state2.current_player == 0
    assert state2.turn_index == 2
    assert state2.players[1].reserved_cards[0].card_id == reserved_card_id
    assert state2.players[1].tokens["gold"] == 1
    assert len(state2.visible_tier_cards[1]) == 4
    assert state2.visible_tier_cards[1][-1].card_id == hidden_replacement_id

    action2 = _pick_action(
        env.legal_actions(state2),
        action_type=ActionType.RESERVE_DECK,
        tier=2,
        take_tokens=("gold",),
        return_tokens=(),
    )
    hidden_reserved_id = state2.hidden_tier_decks[2][0].card_id
    state3 = env.step(state2, action2)

    assert state3.current_player == 1
    assert state3.turn_index == 3
    assert state3.players[0].reserved_cards[0].card_id == hidden_reserved_id
    assert state3.players[0].tokens["gold"] == 1
    assert state3.deck_counts[2] == 25

    action3 = _pick_action(
        env.legal_actions(state3),
        action_type=ActionType.TAKE_TOKENS,
        take_tokens=("red", "red"),
        return_tokens=(),
    )
    state4 = env.step(state3, action3)

    assert state4.current_player == 0
    assert state4.turn_index == 4
    assert state4.players[1].tokens["red"] == 2
    assert state4.bank_tokens["red"] == 2


def test_replay_reserve_then_buy_reserved_card() -> None:
    env = SplendorEnv(seed=0)
    state0 = env.initial_state()
    reserve_target = Card(
        card_id="reserve-target",
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
    state0.visible_tier_cards = {1: [reserve_target], 2: [], 3: []}
    state0.hidden_tier_decks = {1: [replacement], 2: [], 3: []}
    state0.deck_counts = {1: 1, 2: 0, 3: 0}
    state0.players[0].tokens["white"] = 1
    state0.bank_tokens["white"] = 3

    action0 = _pick_action(
        env.legal_actions(state0),
        action_type=ActionType.RESERVE_VISIBLE,
        tier=1,
        market_index=0,
        take_tokens=("gold",),
        return_tokens=(),
    )
    state1 = env.step(state0, action0)

    assert state1.players[0].reserved_cards[0].card_id == "reserve-target"
    assert state1.players[0].tokens["gold"] == 1

    action1 = _pick_action(
        env.legal_actions(state1),
        action_type=ActionType.TAKE_TOKENS,
        take_tokens=("white", "blue", "green"),
        return_tokens=(),
    )
    state2 = env.step(state1, action1)

    action2 = _pick_action(
        env.legal_actions(state2),
        action_type=ActionType.BUY_RESERVED,
        reserved_index=0,
        spend_tokens=("white", "gold"),
    )
    state3 = env.step(state2, action2)

    assert state3.current_player == 1
    assert state3.players[0].reserved_cards == []
    assert state3.players[0].score == 1
    assert state3.players[0].bonuses["red"] == 1
    assert state3.bank_tokens["white"] == 3
    assert state3.bank_tokens["gold"] == 5


def test_replay_final_opponent_turn_can_change_winner() -> None:
    env = SplendorEnv(seed=0)
    state0 = env.initial_state()
    state0.visible_tier_cards = {
        1: [Card(card_id="p0-win-trigger", tier=1, bonus_color="white", points=1, cost={})],
        2: [Card(card_id="p1-answer", tier=2, bonus_color="blue", points=1, cost={})],
        3: [],
    }
    state0.hidden_tier_decks = {1: [], 2: [], 3: []}
    state0.deck_counts = {1: 0, 2: 0, 3: 0}
    state0.players[0].score = 14
    state0.players[1].score = 14
    state0.players[0].purchased_cards = [
        Card(card_id=f"p0-{i}", tier=1, bonus_color="white", points=0, cost={})
        for i in range(5)
    ]
    state0.players[1].purchased_cards = [
        Card(card_id=f"p1-{i}", tier=1, bonus_color="blue", points=0, cost={})
        for i in range(3)
    ]

    action0 = _pick_action(
        env.legal_actions(state0),
        action_type=ActionType.BUY_VISIBLE,
        tier=1,
        market_index=0,
        spend_tokens=(),
        noble_id=None,
    )
    state1 = env.step(state0, action0)

    assert state1.pending_round_end is True
    assert state1.terminal is False
    assert state1.current_player == 1
    assert state1.players[0].score == 15

    action1 = _pick_action(
        env.legal_actions(state1),
        action_type=ActionType.BUY_VISIBLE,
        tier=2,
        market_index=0,
        spend_tokens=(),
        noble_id=None,
    )
    state2 = env.step(state1, action1)

    assert state2.terminal is True
    assert state2.players[1].score == 15
    assert state2.winner == 1
