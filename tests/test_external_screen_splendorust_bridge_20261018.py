"""Finite catalogue, observer and action fixtures; no learned inference/games."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import bridge

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "outputs/external_opponent_search_20261018_02/splendorust"


def initial(seat=0):
    state = rules.CorrectedEnv(seed=38016).initial_state()
    state.current_player = state.start_player = seat
    return state


def tokens(state, seat, values):
    for color, amount in values.items():
        state.players[seat].tokens[color] += amount
        state.bank_tokens[color] -= amount


def payload(state):
    return rules.observation_payload(state)


def fixture_payloads():
    saved = json.loads((ROOT / "outputs/external_strength_screen_20261018_01/fixtures.json").read_text())
    return [rules.observation_payload(rules.deserialize_snapshot(item["snapshot"])) for item in saved]


def multiple_noble_payload():
    """Ledger-valid structural fixture with two compulsory noble alternatives."""
    state = initial()
    needed = {c: max(n.requirement.get(c, 0) for n in state.nobles[:2]) for c in bridge.COLORS}
    cards = rules.corrected_cards_by_id()
    owned = []
    for color, amount in needed.items():
        owned.extend(sorted((c for c in cards.values() if c.bonus_color == color and c.points == 0),
                            key=lambda c: c.card_id)[:amount])
    ids = {c.card_id for c in owned}
    for tier in (1, 2, 3):
        state.visible_tier_cards[tier] = [c for c in state.visible_tier_cards[tier] if c.card_id not in ids]
        state.hidden_tier_decks[tier] = [c for c in state.hidden_tier_decks[tier] if c.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop())
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    state.players[0].purchased_cards = owned
    for card in owned:
        state.players[0].bonuses[card.bonus_color] += 1
        state.players[0].score += card.points
    return payload(state)


def independent_main(observation):
    """Direct pinned core rule enumeration, without atomic action projection."""
    colors, out = bridge.COLORS, []
    actor = observation["players"][observation["current"]]
    bank = observation["bank"]
    count = min(sum(n > 0 for n in bank[:5]), 3)
    if count:
        for mask in range(1, 32):
            take = [(mask >> i) & 1 for i in range(5)]
            if sum(take) == count and all(n <= bank[i] for i, n in enumerate(take)):
                out.append({"kind": "Take", "tokens": take})
    for i in range(5):
        if bank[i] >= 4:
            take = [0] * 5
            take[i] = 2
            out.append({"kind": "Take", "tokens": take})
    cards = rules.corrected_cards_by_id()
    def affordable(index):
        card = cards[bridge.RUST_CARD_IDS[index]]
        return sum(max(card.cost.get(c, 0) - actor["bonuses"][i] - actor["tokens"][i], 0)
                   for i, c in enumerate(colors)) <= actor["tokens"][5]
    can_reserve = observation["reserved_counts"][observation["current"]] < 3
    for slot, card in enumerate(observation["market"]):
        if card == bridge.NONE:
            continue
        if affordable(card):
            out.append({"kind": "BuyVisible", "index": slot})
        if can_reserve:
            out.append({"kind": "ReserveVisible", "index": slot})
    for i in range(3):
        card = actor["reserved"][i]["card"]
        if card != bridge.NONE and affordable(card):
            out.append({"kind": "BuyReserved", "index": i})
        if can_reserve and observation["remaining"][i] > 0:
            out.append({"kind": "ReserveDeck", "tier": i})
    return out


def test_exact_catalogue_csv_rust_constants_and_local_bijection():
    rows = list(csv.DictReader((PINNED / "data/cards.csv").open()))
    source = (PINNED / "crates/splendor-core/src/data.rs").read_text()
    cards = re.findall(r"Card\s*\{\s*tier:\s*(\d+),\s*bonus:\s*(\d+),\s*points:\s*(\d+),\s*cost:\s*\[([^]]+)\],\s*\}", source)
    assert len(cards) == len(rows) == len(bridge.RUST_CARD_IDS) == len(set(bridge.RUST_CARD_IDS)) == 90
    local = rules.corrected_cards_by_id()
    for index, (row, values) in enumerate(zip(rows, cards)):
        tier, bonus, points, cost = values
        rust = (int(tier) + 1, bridge.COLORS[int(bonus)], int(points), tuple(map(int, cost.split(","))))
        csv_face = (int(row["tier"]), row["bonus"], int(row["points"]), tuple(int(row[c]) for c in bridge.COLORS))
        card = local[bridge.RUST_CARD_IDS[index]]
        assert rust == csv_face == (card.tier, card.bonus_color, card.points, tuple(card.cost.get(c, 0) for c in bridge.COLORS))
        assert int(row["id"]) == index
    block = re.search(r"pub const NOBLES[^=]*=\s*\[(.*?)\];", source, re.S).group(1)
    nobles = [tuple(map(int, row.split(","))) for row in re.findall(r"\[([^]]+)\]", block)]
    rows = list(csv.DictReader((PINNED / "data/nobles.csv").open()))
    assert len(nobles) == len(rows) == len(set(bridge.RUST_NOBLE_IDS)) == 10
    for index, (cost, row) in enumerate(zip(nobles, rows)):
        noble = rules.corrected_nobles_by_id()[bridge.RUST_NOBLE_IDS[index]]
        assert cost == tuple(int(row[c]) for c in bridge.COLORS) == tuple(noble.requirement.get(c, 0) for c in bridge.COLORS)
        assert noble.points == int(row["points"]) == 3


@pytest.mark.parametrize("start", [0, 1])
def test_starting_seat_normalization_preserves_turn_and_terminal_order(start):
    state = initial(start)
    state.turn_index, state.current_player = 5, 1 - start
    encoded = bridge.encode_observation(payload(state))
    assert encoded["current"] == encoded["viewer"] == 1 == encoded["turns"] % 2
    assert encoded["players"][2:] == [bridge._empty_player(), bridge._empty_player()]


def test_private_hidden_permutation_cannot_change_any_bridge_input_or_action():
    state = initial()
    for seat, tier in ((0, 1), (1, 2)):
        card = state.hidden_tier_decks[tier].pop()
        state.deck_counts[tier] -= 1
        state.players[seat].reserved_cards.append(card)
        state.players[seat].reserved_card_sources.append("deck")
    changed = state.copy_shallow()
    for cards in changed.hidden_tier_decks.values():
        cards.reverse()
    private = changed.players[1].reserved_cards[0]
    changed.players[1].reserved_cards[0], changed.hidden_tier_decks[2][0] = changed.hidden_tier_decks[2][0], private
    a, b = payload(state), payload(changed)
    assert a == b and bridge.encode_observation(a) == bridge.encode_observation(b)
    encoded = bridge.encode_observation(a)
    assert encoded["reserved_counts"] == [1, 1, 0, 0]
    assert encoded["players"][0]["reserved"][0]["card"] != bridge.NONE
    assert encoded["players"][1]["reserved"][0] == {"card": 255, "tier": 1, "public": False}
    for choice in bridge.main_actions(a):
        assert bridge.resolve_compound(a, choice, lambda o, values: values[0]) == bridge.resolve_compound(b, choice, lambda o, values: values[0])


def test_main_action_order_matches_independent_pinned_rules_on_saved_panel():
    for item in fixture_payloads() + [payload(initial()), payload(initial(1)), multiple_noble_payload()]:
        expected = independent_main(bridge.encode_observation(item))
        actual = bridge.main_actions(item)
        assert actual == (expected or [{"kind": "Pass"}])


def test_all_main_families_resolve_to_legal_atomic_actions_on_saved_panel():
    phases, kinds = set(), set()
    for item in fixture_payloads() + [multiple_noble_payload()]:
        legal = rules.legal_actions(rules.state_from_observation(item))
        for main in bridge.main_actions(item):
            kinds.add(main["kind"])
            for index in (0, -1):
                def choose(observation, values):
                    phases.add(observation["phase"]["kind"])
                    return values[index]
                assert bridge.resolve_compound(item, main, choose) in legal
    assert {"Take", "ReserveVisible", "ReserveDeck", "BuyVisible", "BuyReserved"} <= kinds
    assert {"Return", "Payment", "Noble"} <= phases


def test_every_atomic_payment_return_and_noble_choice_is_representable():
    count = 0
    for item in fixture_payloads() + [multiple_noble_payload()]:
        legal = rules.legal_actions(rules.state_from_observation(item))
        for expected in legal:
            def choose(observation, values):
                phase = observation["phase"]["kind"]
                if phase == "Payment":
                    wanted = {"kind": "Pay", "tokens": bridge._vector(expected.spend_tokens, bridge.COLORS)}
                elif phase == "Return":
                    wanted = {"kind": "Return", "tokens": bridge._vector(expected.return_tokens)}
                else:
                    assert phase == "Noble"
                    wanted = {"kind": "Noble", "index": bridge.NOBLE_TO_RUST[expected.noble_id]}
                assert wanted in values
                return wanted
            actual = bridge.resolve_compound(item, bridge._main(expected), choose)
            assert actual == expected
            count += 1
    assert count == 981


def test_buy_noble_projection_awards_known_card_without_revealing_refill():
    item = multiple_noble_payload()
    original = bridge.encode_observation(item)
    main = next(a for a in bridge.main_actions(item) if a["kind"] == "BuyVisible")
    card_id = original["market"][main["index"]]
    calls = []
    def choose(observation, values):
        calls.append(observation["phase"]["kind"])
        if observation["phase"]["kind"] == "Noble":
            assert observation["market"][main["index"]] == bridge.NONE
            assert observation["remaining"] == original["remaining"]
            assert card_id in observation["players"][observation["current"]]["owned"]
            assert observation["turns"] == original["turns"]
            assert observation["current"] == original["current"]
        return values[-1]
    action = bridge.resolve_compound(item, main, choose)
    assert "Noble" in calls and action.noble_id is not None


@pytest.mark.parametrize("main", [{"kind": "ReserveVisible", "index": 0}, {"kind": "ReserveDeck", "tier": 0}])
def test_reserve_auxiliary_projection_contains_no_future_refill_or_blind_card(main):
    state = initial()
    tokens(state, 0, {"white": 3, "blue": 3, "green": 3, "gold": 1})
    calls = []
    def choose(observation, values):
        calls.append(deepcopy(observation))
        assert observation["phase"] == {"kind": "Return"}
        assert sum(observation["players"][0]["tokens"]) == 11
        assert observation["reserved_counts"][0] == 1
        if main["kind"] == "ReserveVisible":
            assert observation["market"][0] == bridge.NONE
            assert observation["players"][0]["reserved"][0]["card"] == bridge.CARD_TO_RUST[state.visible_tier_cards[1][0].card_id]
        else:
            assert observation["players"][0]["reserved"][0]["card"] == bridge.NONE
            assert observation["remaining"][0] == state.deck_counts[1] - 1
        return values[-1]
    action = bridge.resolve_compound(payload(state), main, choose)
    assert calls and action in rules.legal_actions(state)


def test_invalid_or_unforced_choice_fails_closed():
    item = payload(initial())
    for main in ({"kind": "Pass"}, {"kind": "ReserveVisible", "index": True}, {"kind": "ReserveDeck", "tier": 5}):
        with pytest.raises(ValueError, match="invalid Main"):
            bridge.resolve_compound(item, main, lambda o, values: values[0])
    state = initial()
    tokens(state, 0, {"white": 3, "blue": 3, "green": 3, "gold": 1})
    with pytest.raises(ValueError, match="invalid auxiliary"):
        bridge.resolve_compound(payload(state), {"kind": "ReserveDeck", "tier": 0}, lambda o, values: {"kind": "Pass"})


def test_explicit_forced_pass_only_matches_authoritative_pass():
    # A structural state with no remaining cards, no tokens to take and actor1
    # completing the final round; no played/reachable-history claim is made.
    state = initial()
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    for tier in (1, 2, 3):
        for card in state.visible_tier_cards[tier] + state.hidden_tier_decks[tier]:
            p = state.players[0]
            p.purchased_cards.append(card)
            p.bonuses[card.bonus_color] += 1
            p.score += card.points
        state.visible_tier_cards[tier], state.hidden_tier_decks[tier], state.deck_counts[tier] = [], [], 0
    tokens(state, 0, {"white": 4, "blue": 4, "green": 2})
    tokens(state, 1, {"green": 2, "red": 4, "black": 4})
    item = payload(state)
    assert independent_main(bridge.encode_observation(item)) == []
    assert bridge.main_actions(item) == [{"kind": "Pass"}]
    assert bridge.resolve_compound(item, {"kind": "Pass"}, lambda o, values: values[0]) in rules.legal_actions(state)
