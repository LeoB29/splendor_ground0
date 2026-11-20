"""Small component and transition fixtures; no model inference or arena games."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles
from splendor_ai.engine.env import SplendorEnv
from tools.external_strength_screen_20261018 import rules


EXPECTED = {
    "T1_05": {"blue": 2, "black": 2},
    "T1_07": {"white": 3, "blue": 1, "black": 1},
    "T2_10": {"white": 2, "blue": 3, "red": 3},
    "T2_12": {"red": 5},
    "T2_13": {"red": 5, "black": 3},
    "T2_14": {"white": 6},
}


def _cards(state):
    return [c for groups in (state.visible_tier_cards, state.hidden_tier_decks)
            for values in groups.values() for c in values] + [
                c for p in state.players for c in p.reserved_cards + p.purchased_cards]


def _purchase_fixture(card_id):
    state = rules.CorrectedEnv(seed=41018).initial_state()
    card = rules.corrected_cards_by_id()[card_id]
    market = state.visible_tier_cards[card.tier]
    for group in (state.visible_tier_cards, state.hidden_tier_decks):
        for values in group.values():
            for i, value in enumerate(values):
                if value.card_id == card_id:
                    values[i], market[0] = market[0], values[i]
                    break
            else:
                continue
            break
        else:
            continue
        break
    required = {color: min(card.cost.get(color, 0), 4) for color in ALL_TOKEN_TYPES}
    required["gold"] = sum(max(count - 4, 0) for count in card.cost.values())
    for color, count in required.items():
        state.players[0].tokens[color] = count
        state.bank_tokens[color] -= count
    rules.validate_state(state)
    spend = tuple(color for color in ALL_TOKEN_TYPES for _ in range(required[color]))
    return state, card, spend


def test_catalogue_has_exactly_six_changes_and_does_not_mutate_legacy():
    old = {c.card_id: c for values in build_base_deck_by_tier().values() for c in values}
    before = {key: deepcopy(c.cost) for key, c in old.items()}
    new = rules.corrected_cards_by_id()
    assert len(new) == 90 and len(rules.corrected_nobles_by_id()) == 10
    assert Counter(c.tier for c in new.values()) == {1: 40, 2: 30, 3: 20}
    assert {key for key in new if old[key] != new[key]} == set(EXPECTED)
    for key, cost in EXPECTED.items():
        assert new[key].cost == cost
        assert (new[key].card_id, new[key].tier, new[key].points, new[key].bonus_color) == (
            old[key].card_id, old[key].tier, old[key].points, old[key].bonus_color)
    assert list(rules.corrected_nobles_by_id().values()) == list(build_base_nobles())
    new["T1_05"].cost["blue"] = 99
    assert rules.corrected_cards_by_id()["T1_05"].cost["blue"] == 2
    assert {key: c.cost for key, c in old.items()} == before
    canonical = json.dumps(rules.catalogue_document(), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == rules.CATALOGUE_SHA256


@pytest.mark.parametrize("card_id", list(EXPECTED))
def test_each_corrected_cost_controls_legality_payment_and_replay(card_id):
    state, card, spend = _purchase_fixture(card_id)
    env = rules.CorrectedEnv()
    before = rules.serialize_snapshot(state)
    buys = [a for a in env.legal_actions(state) if a.action_type == ActionType.BUY_VISIBLE
            and a.tier == card.tier and a.market_index == 0 and a.spend_tokens == spend]
    assert len(buys) == 1
    action = buys[0]
    next_state = env.step(state, action)
    assert next_state.players[0].tokens == dict.fromkeys(ALL_TOKEN_TYPES, 0)
    assert next_state.bank_tokens == {c: 5 if c == "gold" else 4 for c in ALL_TOKEN_TYPES}
    assert next_state.players[0].purchased_cards == [card]
    assert next_state.players[0].bonuses["white"] == 1
    assert next_state.players[0].score == card.points
    assert next_state.turn_index == 1 and next_state.current_player == 1
    assert len(next_state.visible_tier_cards[card.tier]) == 4
    assert next_state.deck_counts[card.tier] == state.deck_counts[card.tier] - 1
    rules.validate_state(next_state)
    assert rules.serialize_snapshot(state) == before, "Transition must preserve its input"
    replayed = env.step(rules.deserialize_snapshot(before), rules.deserialize_action(rules.serialize_action(action)))
    assert rules.serialize_snapshot(replayed) == rules.serialize_snapshot(next_state)
    # The same exact payment is not legal for the historical disputed face.
    legacy = state.copy_shallow()
    originals = {c.card_id: c for values in build_base_deck_by_tier().values() for c in values}
    legacy.visible_tier_cards[card.tier][0] = originals[card_id]
    assert action not in SplendorEnv().legal_actions(legacy)


def test_new_initial_state_preserves_original_seed_order_and_other_components():
    old = SplendorEnv(seed=41018).initial_state()
    new = rules.CorrectedEnv(seed=41018).initial_state()
    assert [c.card_id for c in _cards(old)] == [c.card_id for c in _cards(new)]
    assert old.bank_tokens == new.bank_tokens and old.nobles == new.nobles
    assert rules.validate_state(new)


def _privacy_fixture():
    state = rules.CorrectedEnv(seed=41018).initial_state()
    for seat, tier in ((0, 1), (1, 2)):
        state.players[seat].reserve_card(state.hidden_tier_decks[tier].pop(), "deck")
        state.deck_counts[tier] -= 1
    visible = state.visible_tier_cards[3].pop(0)
    state.players[1].reserve_card(visible, "visible")
    state.visible_tier_cards[3].append(state.hidden_tier_decks[3].pop())
    state.deck_counts[3] -= 1
    rules.validate_state(state)
    return state


def test_observation_privacy_and_canonical_reconstruction():
    state = _privacy_fixture()
    payload = rules.observation_payload(state)
    view = payload["observation"]
    assert "hidden_tier_decks" not in view and "seed" not in view
    assert view["players"][0]["reserves"][0]["card_id"] == state.players[0].reserved_cards[0].card_id
    assert view["players"][1]["reserves"][0] == {"card_id": None, "tier": 2, "source": "deck"}
    assert view["players"][1]["reserves"][1]["card_id"] == state.players[1].reserved_cards[1].card_id
    variant = state.copy_shallow()
    variant.players[1].reserved_cards[0], variant.hidden_tier_decks[2][0] = (
        variant.hidden_tier_decks[2][0], variant.players[1].reserved_cards[0])
    for values in variant.hidden_tier_decks.values():
        values.reverse()
    assert rules.observation_payload(variant) == payload
    restored = rules.state_from_observation(payload)
    assert rules.observation_payload(restored) == payload
    assert rules.legal_actions(restored) == rules.legal_actions(state)
    assert rules.validate_state(restored)


def test_snapshot_roundtrip_and_explicit_counterfactual_conversion():
    state = _privacy_fixture()
    saved = rules.serialize_snapshot(state)
    assert rules.serialize_snapshot(rules.deserialize_snapshot(saved)) == saved
    # Input is legacy-shaped IDs, explicitly labelled as a new-rules fixture.
    converted = rules.convert_legacy_snapshot(saved["state"])
    assert converted == saved
    assert converted["ruleset_id"] == rules.RULESET_ID
    assert converted["catalogue_sha256"] == rules.CATALOGUE_SHA256
    restored = rules.deserialize_snapshot(converted)
    lookup = {c.card_id: c for c in _cards(restored)}
    assert {key: lookup[key].cost for key in EXPECTED} == EXPECTED
    with pytest.raises(ValueError, match="noncorrected"):
        rules.serialize_snapshot(SplendorEnv(seed=41018).initial_state())


@pytest.mark.parametrize("field", ["ruleset_id", "catalogue_sha256", "schema"])
def test_snapshot_identity_mismatch_fails_closed(field):
    saved = rules.serialize_snapshot(rules.CorrectedEnv(seed=41018).initial_state())
    saved[field] = "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        rules.deserialize_snapshot(saved)


def test_malformed_observations_cannot_inject_hidden_information():
    original = rules.observation_payload(_privacy_fixture())
    hidden = deepcopy(original)
    hidden["observation"]["hidden_tier_decks"] = {}
    revealed = deepcopy(original)
    revealed["observation"]["players"][1]["reserves"][0]["card_id"] = "T2_10"
    count = deepcopy(original)
    count["observation"]["deck_counts"]["1"] += 1
    duplicate = deepcopy(original)
    duplicate["observation"]["visible_tier_cards"]["1"][1] = duplicate["observation"]["visible_tier_cards"]["1"][0]
    for payload in (hidden, revealed, count, duplicate):
        with pytest.raises(ValueError):
            rules.state_from_observation(payload)


@pytest.mark.parametrize("kind", ["tokens", "bonuses", "score", "duplicate"])
def test_state_accounting_rejects_corruption(kind):
    state = rules.CorrectedEnv(seed=41018).initial_state()
    if kind == "tokens":
        state.bank_tokens["blue"] += 1
    elif kind == "bonuses":
        state.players[0].bonuses["white"] += 1
    elif kind == "score":
        state.players[0].score += 1
    else:
        state.hidden_tier_decks[1][0] = state.visible_tier_cards[1][0]
    with pytest.raises(ValueError):
        rules.validate_state(state)


def test_blind_reserve_transition_keeps_source_and_conservation():
    env = rules.CorrectedEnv(seed=41018)
    state = env.initial_state()
    action = next(a for a in env.legal_actions(state) if a.action_type == ActionType.RESERVE_DECK and a.tier == 2)
    following = env.step(state, action)
    assert following.players[0].reserved_card_sources == ["deck"]
    assert following.players[0].tokens["gold"] == 1 and following.bank_tokens["gold"] == 4
    assert following.deck_counts[2] == state.deck_counts[2] - 1
    assert following.visible_tier_cards == state.visible_tier_cards
    assert rules.validate_state(following)


def _grant_card(state, seat, card):
    for group in (state.visible_tier_cards, state.hidden_tier_decks):
        for tier, values in group.items():
            index = next((i for i, value in enumerate(values) if value.card_id == card.card_id), None)
            if index is not None:
                values.pop(index)
                if group is state.visible_tier_cards and state.hidden_tier_decks[tier]:
                    values.append(state.hidden_tier_decks[tier].pop(0))
                state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
                player = state.players[seat]
                player.purchased_cards.append(card)
                player.bonuses[card.bonus_color] += 1
                player.score += card.points
                return
    raise AssertionError("Fixture card already assigned")


def _noble_fixture(actor=0, start=0, threshold=False):
    state = rules.CorrectedEnv(seed=41018).initial_state()
    state.current_player = actor
    state.start_player = start
    state.turn_index = int(actor != start)
    cards = rules.corrected_cards_by_id()
    nobles = rules.corrected_nobles_by_id()
    state.nobles = [nobles[k] for k in ("N03", "N04", "N10")]
    counts = {"white": 4 if threshold else 3, "black": 4, "blue": 0 if threshold else 4}
    for color, count in counts.items():
        selected = [c for c in cards.values() if c.bonus_color == color and c.points == 0 and c.card_id != "T1_05"][:count]
        for card in selected:
            _grant_card(state, actor, card)
    if threshold:
        for color in ("white", "black", "blue"):
            card = next(c for c in cards.values() if c.bonus_color == color and c.points == 4)
            _grant_card(state, actor, card)
    rules.validate_state(state)
    return state


def test_two_eligible_nobles_choose_one_then_nonbuy_claims_remaining():
    env = rules.CorrectedEnv()
    state = _noble_fixture()
    card = rules.corrected_cards_by_id()["T1_05"]
    market = state.visible_tier_cards[1]
    for group in (state.visible_tier_cards, state.hidden_tier_decks):
        values = group[1]
        position = next((i for i, c in enumerate(values) if c.card_id == card.card_id), None)
        if position is not None:
            values[position], market[0] = market[0], values[position]
            break
    buys = [a for a in env.legal_actions(state) if a.action_type == ActionType.BUY_VISIBLE and a.tier == 1 and a.market_index == 0]
    assert {a.noble_id for a in buys} == {"N03", "N04"}
    state = env.step(state, next(a for a in buys if a.noble_id == "N03"))
    assert [n.noble_id for n in state.players[0].nobles] == ["N03"]
    state = env.step(state, next(a for a in env.legal_actions(state) if a.action_type == ActionType.TAKE_TOKENS))
    takes = [a for a in env.legal_actions(state) if a.action_type == ActionType.TAKE_TOKENS]
    assert takes and {a.noble_id for a in takes} == {"N04"}
    state = env.step(state, takes[0])
    assert [n.noble_id for n in state.players[0].nobles] == ["N03", "N04"]
    assert state.players[0].score == 6 and rules.validate_state(state)


@pytest.mark.parametrize("start,actor", [(0, 0), (0, 1), (1, 1), (1, 0)])
def test_nonbuy_noble_crossing_15_respects_both_seats_and_end_of_round(start, actor):
    env = rules.CorrectedEnv()
    state = _noble_fixture(actor, start, threshold=True)
    assert state.players[actor].score == 12
    action = next(a for a in env.legal_actions(state) if a.action_type == ActionType.TAKE_TOKENS and a.noble_id == "N03")
    following = env.step(state, action)
    assert following.players[actor].score == 15 and following.pending_round_end
    assert following.terminal == (actor != start)
    if actor == start:
        following = env.step(following, env.legal_actions(following)[0])
    assert following.terminal and following.winner == actor
    assert rules.validate_state(following)


def test_forced_pass_still_awards_eligible_noble():
    env = rules.CorrectedEnv()
    state = _noble_fixture(threshold=True)
    actor = state.players[0]
    actor.tokens.update(white=4, blue=4, green=2)
    state.players[1].tokens.update(green=2, red=4, black=4)
    for color in ("white", "blue", "green", "red", "black"):
        state.bank_tokens[color] = 0
    purchased = {c.card_id for p in state.players for c in p.purchased_cards}
    available = {k: c for k, c in rules.corrected_cards_by_id().items() if k not in purchased}
    reserve_ids = [c.card_id for c in available.values() if c.tier == 3 and
                   any(c.cost.get(color, 0) > actor.bonuses[color] + actor.tokens[color]
                       for color in ("white", "blue", "green", "red", "black"))][:3]
    actor.reserved_cards = [available.pop(k) for k in reserve_ids]
    actor.reserved_card_sources = ["deck"] * 3
    for tier in (1, 2, 3):
        choices = [c for c in available.values() if c.tier == tier and
                   any(c.cost.get(color, 0) > actor.bonuses[color] + actor.tokens[color]
                       for color in ("white", "blue", "green", "red", "black"))]
        assert len(choices) >= 4
        state.visible_tier_cards[tier] = choices[:4]
        for card in choices[:4]:
            available.pop(card.card_id)
        state.hidden_tier_decks[tier] = [c for c in available.values() if c.tier == tier]
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    rules.validate_state(state)
    actions = env.legal_actions(state)
    assert len(actions) == 1 and actions[0].action_type == ActionType.PASS and actions[0].noble_id == "N03"
    following = env.step(state, actions[0])
    assert following.players[0].score == 15 and following.pending_round_end and not following.terminal
    assert rules.validate_state(following)
