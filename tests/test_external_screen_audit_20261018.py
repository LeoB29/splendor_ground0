"""Independent ledger checks on synthetic fixtures and 32 deterministic plies."""
from copy import deepcopy
from collections import Counter

import pytest

from splendor_ai.engine.actions import ActionType
from tools.external_strength_screen_20261018 import audit, rules


def _step(state, action):
    before = rules.serialize_snapshot(state)
    after_state = rules.CorrectedEnv().step(state, action)
    after = rules.serialize_snapshot(after_state)
    encoded = rules.serialize_action(action)
    assert audit.transition(before, encoded) == after
    assert len(audit.check_step(before, encoded, after)) == 64
    assert len(audit.check_observer(after, rules.observation_payload(after_state))) == 64
    return after_state


def _remove(state, card_id):
    for group in (state.visible_tier_cards, state.hidden_tier_decks):
        for tier, values in group.items():
            index = next((i for i, c in enumerate(values) if c.card_id == card_id), None)
            if index is not None:
                card = values.pop(index)
                if group is state.visible_tier_cards and state.hidden_tier_decks[tier]:
                    values.append(state.hidden_tier_decks[tier].pop(0))
                state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
                return card
    raise AssertionError("Card absent from fixture")


def _grant(state, seat, card_id):
    card = _remove(state, card_id)
    p = state.players[seat]
    p.purchased_cards.append(card)
    p.bonuses[card.bonus_color] += 1
    p.score += card.points


def _gold_purchase():
    state = rules.CorrectedEnv(seed=82018).initial_state()
    card = _remove(state, "T2_14")
    # Put the corrected white-six card in own reserves, preserving all 90 cards.
    state.players[0].reserve_card(card, "visible")
    state.players[0].tokens.update(white=4, gold=2)
    state.bank_tokens.update(white=0, gold=3)
    action = next(a for a in rules.legal_actions(state) if a.action_type == ActionType.BUY_RESERVED)
    return state, action


def _noble_threshold(actor, start):
    state = rules.CorrectedEnv(seed=82018).initial_state()
    state.start_player, state.current_player = start, actor
    state.turn_index = int(actor != start)
    cards = rules.corrected_cards_by_id()
    for color in ("white", "black"):
        for c in [c for c in cards.values() if c.bonus_color == color and c.points == 0][:4]:
            _grant(state, actor, c.card_id)
    for color in ("white", "blue", "black"):
        _grant(state, actor, next(c.card_id for c in cards.values() if c.bonus_color == color and c.points == 4))
    nobles = rules.corrected_nobles_by_id()
    state.nobles = [nobles[n] for n in ("N03", "N04", "N10")]
    assert state.players[actor].score == 12
    rules.validate_state(state)
    return state


def test_independent_ledger_agrees_over_bounded_deterministic_sequence():
    state = rules.CorrectedEnv(seed=82018).initial_state()
    for _ in range(32):
        assert not state.terminal
        actions = rules.legal_actions(state)
        buys = [a for a in actions if a.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)]
        action = buys[0] if buys else actions[0]
        state = _step(state, action)


@pytest.mark.parametrize("kind", [ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK])
def test_independent_reservation_replacement_and_private_observer(kind):
    state = rules.CorrectedEnv(seed=82018).initial_state()
    action = next(a for a in rules.legal_actions(state) if a.action_type == kind and a.tier == 2)
    next_state = _step(state, action)
    payload = rules.observation_payload(next_state)
    reserve = payload["observation"]["players"][0]["reserves"][0]
    assert (reserve["card_id"] is None) == (kind == ActionType.RESERVE_DECK)
    corrupted = deepcopy(payload)
    if kind == ActionType.RESERVE_DECK:
        corrupted["observation"]["players"][0]["reserves"][0]["card_id"] = next_state.players[0].reserved_cards[0].card_id
    else:
        corrupted["observation"]["players"][0]["reserves"][0]["card_id"] = None
    with pytest.raises(ValueError, match="observer"):
        audit.check_observer(rules.serialize_snapshot(next_state), corrupted)


def test_independent_gold_payment_and_corrupted_payment_rejection():
    state, action = _gold_purchase()
    assert action.spend_tokens.count("white") == 4 and action.spend_tokens.count("gold") == 2
    _step(state, action)
    encoded = rules.serialize_action(action)
    corrupted = deepcopy(encoded)
    corrupted["spend_tokens"].remove("gold")
    with pytest.raises(ValueError, match="gold payment"):
        audit.transition(rules.serialize_snapshot(state), corrupted)
    corrupted = deepcopy(encoded)
    corrupted["spend_tokens"].insert(0, "white")
    with pytest.raises(ValueError, match="colored payment"):
        audit.transition(rules.serialize_snapshot(state), corrupted)


@pytest.mark.parametrize("start,actor", [(0, 0), (0, 1), (1, 1), (1, 0)])
def test_independent_nonbuy_noble_threshold_and_terminal(start, actor):
    state = _noble_threshold(actor, start)
    action = next(a for a in rules.legal_actions(state) if a.action_type == ActionType.TAKE_TOKENS and a.noble_id == "N03")
    missing = rules.serialize_action(action)
    missing["noble_id"] = None
    with pytest.raises(ValueError, match="mandatory"):
        audit.transition(rules.serialize_snapshot(state), missing)
    state = _step(state, action)
    assert state.players[actor].score == 15
    if actor == start:
        assert not state.terminal
        state = _step(state, rules.legal_actions(state)[0])
    assert state.terminal and state.winner == actor
    corrupted = rules.serialize_snapshot(state)
    corrupted["state"]["winner"] = 1 - actor
    with pytest.raises(ValueError, match="winner"):
        audit.validate_ledger(corrupted)


@pytest.mark.parametrize("corruption", ["schema", "winner", "parity", "bank", "bonus", "deck", "noble"])
def test_ledger_rejects_corrupted_state(corruption):
    saved = rules.serialize_snapshot(rules.CorrectedEnv(seed=82018).initial_state())
    if corruption == "schema":
        saved["schema"] = "wrong"
    elif corruption == "winner":
        saved["state"]["winner"] = 0
    elif corruption == "parity":
        saved["state"]["current_player"] = 1
    elif corruption == "bank":
        saved["state"]["bank_tokens"]["blue"] += 1
    elif corruption == "bonus":
        saved["state"]["players"][0]["bonuses"]["white"] += 1
    elif corruption == "deck":
        saved["state"]["deck_counts"]["1"] += 1
    else:
        saved["state"]["nobles"][1] = saved["state"]["nobles"][0]
    with pytest.raises(ValueError):
        audit.validate_ledger(saved)


@pytest.mark.parametrize("field,value", [("tier", 1), ("market_index", 0), ("reserved_index", 0),
                                         ("noble_id", "N03"), ("metadata", [["ignored", "value"]])])
def test_ignored_action_fields_are_rejected(field, value):
    state = rules.CorrectedEnv(seed=82018).initial_state()
    action = next(a for a in rules.legal_actions(state) if a.action_type == ActionType.TAKE_TOKENS)
    encoded = rules.serialize_action(action)
    encoded[field] = value
    with pytest.raises(ValueError):
        audit.transition(rules.serialize_snapshot(state), encoded)


@pytest.mark.parametrize("index", [-1, True, 4])
def test_bad_market_indices_are_not_python_list_aliases(index):
    state = rules.CorrectedEnv(seed=82018).initial_state()
    action = next(a for a in rules.legal_actions(state) if a.action_type == ActionType.RESERVE_VISIBLE)
    encoded = rules.serialize_action(action)
    encoded["market_index"] = index
    with pytest.raises(ValueError, match="market index"):
        audit.transition(rules.serialize_snapshot(state), encoded)


def test_after_state_and_observer_corruptions_fail_comparison():
    state = rules.CorrectedEnv(seed=82018).initial_state()
    action = rules.legal_actions(state)[0]
    after_state = rules.CorrectedEnv().step(state, action)
    after = rules.serialize_snapshot(after_state)
    after["state"]["turn_index"] += 2
    with pytest.raises(ValueError, match="differs"):
        audit.check_step(rules.serialize_snapshot(state), rules.serialize_action(action), after)
    payload = rules.observation_payload(state)
    payload["observation"]["seed"] = 82018
    with pytest.raises(ValueError, match="observer"):
        audit.check_observer(rules.serialize_snapshot(state), payload)


@pytest.mark.parametrize("first,second,target,winner", [
    ([("white", 5), ("blue", 5), ("black", 5)],
     [("red", 5), ("green", 5), ("white", 4)], ("blue", 1), 0),
    ([("white", 5), ("blue", 5), ("black", 4), ("red", 1)],
     [("black", 5), ("green", 5), ("red", 4)], ("white", 1), None),
    ([("white", 5), ("blue", 4), ("black", 3), ("red", 2), ("green", 1)],
     [("black", 5), ("green", 5), ("white", 4)], ("red", 1), 1),
])
def test_independent_terminal_tiebreak_and_exact_draw(first, second, target, winner):
    state = rules.CorrectedEnv(seed=82018).initial_state()
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    cards = rules.corrected_cards_by_id()

    def pick(spec):
        return next(c for c in cards.values() if (c.bonus_color, c.points) == spec)

    for seat, specifications in enumerate((first, second)):
        for specification in specifications:
            _grant(state, seat, pick(specification).card_id)
    card = _remove(state, pick(target).card_id)
    state.players[1].reserve_card(card, "visible")
    for color in audit.COLORS:
        required = max(0, card.cost.get(color, 0) - state.players[1].bonuses[color])
        state.players[1].tokens[color] = required
        state.bank_tokens[color] -= required
    assert [p.score for p in state.players] == [15, 14]
    action = next(a for a in rules.legal_actions(state) if a.action_type == ActionType.BUY_RESERVED)
    following = _step(state, action)
    assert following.terminal and following.winner == winner
    assert [p.score for p in following.players] == [15, 15]


@pytest.mark.parametrize("seat", [0, 1])
def test_boolean_seat_is_not_an_integer_identity(seat):
    snapshot = rules.serialize_snapshot(rules.CorrectedEnv(seed=82018).initial_state())
    audit.validate_ledger(snapshot)
    snapshot["state"]["players"][seat]["player_id"] = bool(seat)
    with pytest.raises(ValueError, match="seat identity"):
        audit.validate_ledger(snapshot)


@pytest.mark.parametrize("remaining", [0, 1])
def test_boolean_deck_count_cannot_alias_zero_or_one(remaining):
    snapshot = rules.serialize_snapshot(rules.CorrectedEnv(seed=82018).initial_state())
    state = snapshot["state"]
    removed = state["hidden_tier_decks"]["1"][remaining:]
    state["hidden_tier_decks"]["1"] = state["hidden_tier_decks"]["1"][:remaining]
    state["deck_counts"]["1"] = remaining
    player = state["players"][0]
    player["purchased_card_ids"] = removed
    cards = rules.corrected_cards_by_id()
    bonuses = Counter(cards[key].bonus_color for key in removed)
    player["bonuses"] = {color: bonuses[color] for color in audit.COLORS}
    player["score"] = sum(cards[key].points for key in removed)
    assert player["score"] < 15
    audit.validate_ledger(snapshot)
    state["deck_counts"]["1"] = bool(remaining)
    with pytest.raises(ValueError, match="deck count"):
        audit.validate_ledger(snapshot)


@pytest.mark.parametrize("flow", ["take_tokens", "return_tokens", "spend_tokens"])
def test_token_multisets_require_engine_canonical_order(flow):
    if flow == "spend_tokens":
        state, action = _gold_purchase()
    else:
        state = rules.CorrectedEnv(seed=82018).initial_state()
        if flow == "return_tokens":
            for color, count in {"white": 4, "blue": 4, "green": 2}.items():
                state.players[0].tokens[color] = count
                state.bank_tokens[color] -= count
        action = next(a for a in rules.legal_actions(state) if a.action_type == ActionType.TAKE_TOKENS
                      and len(set(getattr(a, flow))) > 1)
    encoded = rules.serialize_action(action)
    snapshot = rules.serialize_snapshot(state)
    audit.transition(snapshot, encoded)
    encoded[flow].reverse()
    assert encoded[flow] != list(getattr(action, flow))
    with pytest.raises(ValueError, match="noncanonical token ordering"):
        audit.transition(snapshot, encoded)
