"""Corrected component faces without changing the frozen historical engine.

The six corrections are independently adjudicated from the complete public BGA
component database. Historical IDs/order remain stable; every new record binds
the new catalogue. This module contains no model loading or game runner.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, Noble, PlayerState, SplendorState


RULESET_ID = "splendor_base_qualified_rules_20261018_v1"
SNAPSHOT_SCHEMA = "external_screen_snapshot_v1"
OBSERVATION_SCHEMA = "external_screen_observation_v1"
CORRECTED_COSTS = {
    "T1_05": {"blue": 2, "black": 2},
    "T1_07": {"white": 3, "blue": 1, "black": 1},
    "T2_10": {"white": 2, "blue": 3, "red": 3},
    "T2_12": {"red": 5},
    "T2_13": {"red": 5, "black": 3},
    "T2_14": {"white": 6},
}
_HISTORICAL_COSTS = {
    "T1_05": {"blue": 2, "green": 2},
    "T1_07": {"blue": 1, "black": 1},
    "T2_10": {"white": 2, "blue": 3, "red": 3, "green": 4},
    "T2_12": {"red": 5, "green": 4},
    "T2_13": {"red": 5, "black": 3, "green": 4},
    "T2_14": {"white": 6, "green": 4},
}
_COMMON_FIELDS = {"current_player", "turn_index", "start_player", "pending_round_end",
                  "terminal", "winner", "bank_tokens", "visible_tier_cards", "deck_counts",
                  "nobles", "players"}
_PLAYER_FIELDS = {"player_id", "tokens", "bonuses", "score", "purchased_card_ids", "noble_ids"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def corrected_cards_by_id():
    """Fresh card/cost objects; never mutate the historical cached catalogue."""
    result = {}
    for cards in build_base_deck_by_tier().values():
        for card in cards:
            if card.card_id in CORRECTED_COSTS:
                _require(card.cost == _HISTORICAL_COSTS[card.card_id], "Historical correction source changed")
            result[card.card_id] = Card(card.card_id, card.tier, card.bonus_color, card.points,
                                       dict(CORRECTED_COSTS.get(card.card_id, card.cost)))
    _require(len(result) == 90 and set(CORRECTED_COSTS) <= set(result), "Incorrect card catalogue")
    return result


def corrected_nobles_by_id():
    return {n.noble_id: Noble(n.noble_id, n.points, dict(n.requirement)) for n in build_base_nobles()}


def catalogue_document():
    return {"cards": [{"card_id": c.card_id, "tier": c.tier, "bonus_color": c.bonus_color,
                       "points": c.points, "cost": {k: c.cost.get(k, 0) for k in TOKEN_COLORS}}
                      for _, c in sorted(corrected_cards_by_id().items())],
            "nobles": [{"noble_id": n.noble_id, "points": n.points,
                        "requirement": {k: n.requirement.get(k, 0) for k in TOKEN_COLORS}}
                       for _, n in sorted(corrected_nobles_by_id().items())]}


CATALOGUE_SHA256 = hashlib.sha256(json.dumps(catalogue_document(), sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()


def _replace_cards(state):
    cards, nobles = corrected_cards_by_id(), corrected_nobles_by_id()
    state.visible_tier_cards = {tier: [cards[c.card_id] for c in values]
                                for tier, values in state.visible_tier_cards.items()}
    state.hidden_tier_decks = {tier: [cards[c.card_id] for c in values]
                              for tier, values in state.hidden_tier_decks.items()}
    state.nobles = [nobles[n.noble_id] for n in state.nobles]
    for player in state.players:
        player.reserved_cards = [cards[c.card_id] for c in player.reserved_cards]
        player.purchased_cards = [cards[c.card_id] for c in player.purchased_cards]
        player.nobles = [nobles[n.noble_id] for n in player.nobles]
    return state


class CorrectedEnv(SplendorEnv):
    """Versioned components and compulsory noble visits after any action.

    Historical buy transitions already resolve one noble. The isolated wrapper
    adds remaining eligible nobles to non-buy choices and accounts for their
    points before deciding whether the round/game has finished.
    """
    def initial_state(self):
        return _replace_cards(super().initial_state())

    def legal_actions(self, state):
        actions = super().legal_actions(state)
        if not actions:
            return actions
        player = state.players[state.current_player]
        eligible = [n.noble_id for n in state.nobles
                    if all(player.bonuses.get(c, 0) >= count for c, count in n.requirement.items())]
        if not eligible:
            return actions
        return [variant for action in actions for variant in (
            [action] if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
            else [replace(action, noble_id=noble_id) for noble_id in eligible])]

    def step(self, state, action):
        following = super().step(state, action)
        if action.action_type not in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED) and action.noble_id is not None:
            noble = next(n for n in following.nobles if n.noble_id == action.noble_id)
            following.nobles.remove(noble)
            player = following.players[state.current_player]
            player.nobles.append(noble)
            player.score += noble.points
            following.pending_round_end = following.pending_round_end or player.score >= 15
            if following.pending_round_end and following.current_player == following.start_player:
                self._finalize_terminal_state(following)
        return following


def _envelope(schema, view, key):
    return {"schema": schema, "ruleset_id": RULESET_ID, "catalogue_sha256": CATALOGUE_SHA256, key: view}


def _unwrap(value, schema, key):
    _require(isinstance(value, dict) and set(value) == {"schema", "ruleset_id", "catalogue_sha256", key},
             "Unexpected ruleset envelope fields")
    _require(value["schema"] == schema and value["ruleset_id"] == RULESET_ID and
             value["catalogue_sha256"] == CATALOGUE_SHA256, "Ruleset/catalogue identity mismatch")
    return deepcopy(value[key])


def _snapshot_view(state):
    cards = corrected_cards_by_id()
    nobles = corrected_nobles_by_id()
    for values in list(state.visible_tier_cards.values()) + list(state.hidden_tier_decks.values()) + [
            p.reserved_cards + p.purchased_cards for p in state.players]:
        _require(all(c.card_id in cards and c == cards[c.card_id] for c in values), "State has noncorrected card face")
    _require(all(n.noble_id in nobles and n == nobles[n.noble_id]
                 for n in state.nobles + [n for p in state.players for n in p.nobles]), "State has invalid noble face")
    return {"current_player": state.current_player, "turn_index": state.turn_index,
            "start_player": state.start_player, "pending_round_end": state.pending_round_end,
            "terminal": state.terminal, "winner": state.winner, "bank_tokens": dict(state.bank_tokens),
            "visible_tier_cards": {str(t): [c.card_id for c in v] for t, v in state.visible_tier_cards.items()},
            "hidden_tier_decks": {str(t): [c.card_id for c in v] for t, v in state.hidden_tier_decks.items()},
            "deck_counts": {str(t): count for t, count in state.deck_counts.items()},
            "nobles": [n.noble_id for n in state.nobles],
            "players": [{"player_id": p.player_id, "tokens": dict(p.tokens), "bonuses": dict(p.bonuses),
                         "score": p.score, "purchased_card_ids": [c.card_id for c in p.purchased_cards],
                         "noble_ids": [n.noble_id for n in p.nobles],
                         "reserved_card_ids": [c.card_id for c in p.reserved_cards],
                         "reserved_card_sources": [p.reserved_card_source(i) for i in range(len(p.reserved_cards))]}
                        for p in state.players]}


def serialize_snapshot(state):
    validate_state(state)
    return _envelope(SNAPSHOT_SCHEMA, _snapshot_view(state), "state")


def _counts(value, keys):
    _require(isinstance(value, dict) and set(value) == set(keys) and
             all(type(n) is int and n >= 0 for n in value.values()), "Invalid count map")


def _state_from_view(view):
    _require(isinstance(view, dict) and set(view) == _COMMON_FIELDS | {"hidden_tier_decks"}, "Invalid snapshot fields")
    _require(type(view["current_player"]) is int and view["current_player"] in (0, 1) and
             type(view["start_player"]) is int and view["start_player"] in (0, 1) and
             type(view["turn_index"]) is int and view["turn_index"] >= 0, "Invalid turn metadata")
    _require(type(view["pending_round_end"]) is bool and type(view["terminal"]) is bool and
             (view["winner"] is None or (type(view["winner"]) is int and view["winner"] in (0, 1))), "Invalid terminal metadata")
    _counts(view["bank_tokens"], ALL_TOKEN_TYPES)
    _counts(view["deck_counts"], ("1", "2", "3"))
    cards, nobles = corrected_cards_by_id(), corrected_nobles_by_id()
    used, used_nobles = set(), set()

    def card_list(values, tier=None):
        _require(isinstance(values, list), "Invalid card list")
        result = []
        for key in values:
            _require(isinstance(key, str) and key in cards and key not in used, "Unknown/duplicate card")
            _require(tier is None or cards[key].tier == tier, "Card tier mismatch")
            used.add(key)
            result.append(cards[key])
        return result

    def noble_list(values):
        _require(isinstance(values, list), "Invalid noble list")
        result = []
        for key in values:
            _require(isinstance(key, str) and key in nobles and key not in used_nobles, "Unknown/duplicate noble")
            used_nobles.add(key)
            result.append(nobles[key])
        return result

    groups = []
    for name in ("visible_tier_cards", "hidden_tier_decks"):
        _require(isinstance(view[name], dict) and set(view[name]) == {"1", "2", "3"}, "Invalid tier map")
        groups.append({t: card_list(view[name][str(t)], t) for t in (1, 2, 3)})
    visible, hidden = groups
    _require(all(len(visible[t]) <= 4 and len(hidden[t]) == view["deck_counts"][str(t)] for t in hidden), "Invalid deck size")
    available_nobles = noble_list(view["nobles"])
    _require(isinstance(view["players"], list) and len(view["players"]) == 2, "Invalid players")
    players = []
    for seat, p in enumerate(view["players"]):
        _require(isinstance(p, dict) and set(p) == _PLAYER_FIELDS | {"reserved_card_ids", "reserved_card_sources"}, "Invalid player fields")
        _require(type(p["player_id"]) is int and p["player_id"] == seat and type(p["score"]) is int and p["score"] >= 0, "Invalid player metadata")
        _counts(p["tokens"], ALL_TOKEN_TYPES)
        _counts(p["bonuses"], TOKEN_COLORS)
        reserved = card_list(p["reserved_card_ids"])
        sources = p["reserved_card_sources"]
        _require(isinstance(sources, list) and len(sources) == len(reserved) <= 3 and
                 all(s in ("visible", "deck") for s in sources), "Invalid reserve sources")
        players.append(PlayerState(player_id=seat, tokens=dict(p["tokens"]), bonuses=dict(p["bonuses"]),
                                   score=p["score"], reserved_cards=reserved, reserved_card_sources=list(sources),
                                   purchased_cards=card_list(p["purchased_card_ids"]), nobles=noble_list(p["noble_ids"])))
    _require(len(used) == 90 and len(used_nobles) == 3, "Incomplete component allocation")
    return SplendorState(current_player=view["current_player"], bank_tokens=dict(view["bank_tokens"]),
        players=players, visible_tier_cards=visible, hidden_tier_decks=hidden,
        deck_counts={int(t): n for t, n in view["deck_counts"].items()}, nobles=available_nobles,
        turn_index=view["turn_index"], start_player=view["start_player"], pending_round_end=view["pending_round_end"],
        terminal=view["terminal"], winner=view["winner"])


def deserialize_snapshot(payload):
    state = _state_from_view(_unwrap(payload, SNAPSHOT_SCHEMA, "state"))
    validate_state(state)
    return state


def validate_state(state):
    """Check component allocation and public accounting, not historical reachability."""
    _state_from_view(_snapshot_view(state))
    _require(state.current_player == (state.start_player + state.turn_index) % 2, "Turn/actor parity mismatch")
    for color in ALL_TOKEN_TYPES:
        expected = 5 if color == "gold" else 4
        _require(state.bank_tokens[color] + sum(p.tokens[color] for p in state.players) == expected,
                 "Token conservation failure")
    for p in state.players:
        _require(sum(p.tokens.values()) <= 10, "Player token cap exceeded")
        expected = {c: sum(card.bonus_color == c for card in p.purchased_cards) for c in TOKEN_COLORS}
        _require(p.bonuses == expected, "Purchased cards/bonus counts disagree")
        _require(p.score == sum(c.points for c in p.purchased_cards) + sum(n.points for n in p.nobles),
                 "Purchased cards/nobles/score disagree")
    return True


def convert_legacy_snapshot(snapshot):
    """Explicit counterfactual fixture conversion; this does not replay old games.

    Old snapshots store IDs only. Resolving those IDs here replaces six faces
    while preserving all recorded allocations/accounting. Historical actions
    may no longer be legal, so callers must label these as corrected-cost fixtures.
    """
    view = deepcopy(snapshot)
    _require(isinstance(view, dict), "Invalid legacy snapshot")
    for name in ("visible_tier_cards", "hidden_tier_decks", "deck_counts"):
        _require(isinstance(view.get(name), dict), "Missing legacy tier map")
        view[name] = {str(k): v for k, v in view[name].items()}
    return serialize_snapshot(_state_from_view(view))


def legal_actions(state):
    return CorrectedEnv().legal_actions(state)


def observation_payload(state):
    """Only the acting player's legal information, with no game seed or hidden order."""
    view = serialize_snapshot(state)["state"]
    del view["hidden_tier_decks"]
    for seat, p in enumerate(view["players"]):
        ids, sources = p.pop("reserved_card_ids"), p.pop("reserved_card_sources")
        p["reserves"] = [{"card_id": key if seat == state.current_player or source == "visible" else None,
                          "tier": state.players[seat].reserved_cards[i].tier, "source": source}
                         for i, (key, source) in enumerate(zip(ids, sources))]
    return _envelope(OBSERVATION_SCHEMA, view, "observation")


def state_from_observation(payload):
    """Canonical dummy hidden allocations are inferred only from lawful information."""
    view = _unwrap(payload, OBSERVATION_SCHEMA, "observation")
    _require(isinstance(view, dict) and set(view) == _COMMON_FIELDS, "Invalid observation fields")
    _require(type(view["current_player"]) is int and view["current_player"] in (0, 1), "Invalid observer")
    _require(isinstance(view["players"], list) and len(view["players"]) == 2, "Invalid players")
    cards = corrected_cards_by_id()
    seen, unknown = set(), {t: [] for t in (1, 2, 3)}

    def known(key, tier=None):
        _require(isinstance(key, str) and key in cards and key not in seen and
                 (tier is None or cards[key].tier == tier), "Unknown/duplicate public card or wrong tier")
        seen.add(key)
        return key

    _require(isinstance(view["visible_tier_cards"], dict) and set(view["visible_tier_cards"]) == {"1", "2", "3"}, "Invalid market")
    _counts(view["deck_counts"], ("1", "2", "3"))
    for t in (1, 2, 3):
        _require(isinstance(view["visible_tier_cards"][str(t)], list), "Invalid market list")
        for key in view["visible_tier_cards"][str(t)]:
            known(key, t)
    for seat, p in enumerate(view["players"]):
        _require(isinstance(p, dict) and set(p) == _PLAYER_FIELDS | {"reserves"}, "Invalid observer player")
        _require(isinstance(p["purchased_card_ids"], list), "Invalid purchases")
        for key in p["purchased_card_ids"]:
            known(key)
        reserves = p.pop("reserves")
        _require(isinstance(reserves, list) and len(reserves) <= 3, "Invalid reserves")
        p["reserved_card_ids"], p["reserved_card_sources"] = [], []
        for index, reserve in enumerate(reserves):
            _require(isinstance(reserve, dict) and set(reserve) == {"card_id", "tier", "source"} and
                     type(reserve["tier"]) is int and reserve["tier"] in (1, 2, 3) and
                     reserve["source"] in ("visible", "deck"), "Invalid reserve descriptor")
            private = seat != view["current_player"] and reserve["source"] == "deck"
            if private:
                _require(reserve["card_id"] is None, "Opponent private reserve identity disclosed")
                unknown[reserve["tier"]].append((seat, index))
                p["reserved_card_ids"].append(None)
            else:
                p["reserved_card_ids"].append(known(reserve["card_id"], reserve["tier"]))
            p["reserved_card_sources"].append(reserve["source"])
    view["hidden_tier_decks"] = {}
    for tier in (1, 2, 3):
        unseen = sorted(key for key, c in cards.items() if c.tier == tier and key not in seen)
        _require(len(unseen) == view["deck_counts"][str(tier)] + len(unknown[tier]), "Unseen card count mismatch")
        for seat, index in unknown[tier]:
            view["players"][seat]["reserved_card_ids"][index] = unseen.pop()
        view["hidden_tier_decks"][str(tier)] = unseen
    state = _state_from_view(view)
    validate_state(state)
    return state


def serialize_action(action):
    return {"action_type": action.action_type.name, "tier": action.tier, "market_index": action.market_index,
            "reserved_index": action.reserved_index, "take_tokens": list(action.take_tokens),
            "return_tokens": list(action.return_tokens), "spend_tokens": list(action.spend_tokens),
            "noble_id": action.noble_id, "metadata": [list(item) for item in action.metadata]}


def deserialize_action(value):
    _require(isinstance(value, dict) and set(value) == {"action_type", "tier", "market_index", "reserved_index",
             "take_tokens", "return_tokens", "spend_tokens", "noble_id", "metadata"}, "Invalid action fields")
    _require(isinstance(value["action_type"], str) and value["action_type"] in ActionType.__members__, "Invalid action type")
    for key, values in (("tier", (1, 2, 3)), ("market_index", range(4)), ("reserved_index", range(3))):
        _require(value[key] is None or (type(value[key]) is int and value[key] in values), "Invalid action location")
    for key in ("take_tokens", "return_tokens", "spend_tokens"):
        _require(isinstance(value[key], list) and all(c in ALL_TOKEN_TYPES for c in value[key]), "Invalid token flow")
    _require(value["noble_id"] is None or value["noble_id"] in corrected_nobles_by_id(), "Invalid action noble")
    _require(isinstance(value["metadata"], list) and all(isinstance(pair, list) and len(pair) == 2 and
             all(isinstance(x, str) for x in pair) for pair in value["metadata"]), "Invalid action metadata")
    return Action(action_type=ActionType[value["action_type"]], tier=value["tier"], market_index=value["market_index"],
                  reserved_index=value["reserved_index"], take_tokens=tuple(value["take_tokens"]),
                  return_tokens=tuple(value["return_tokens"]), spend_tokens=tuple(value["spend_tokens"]),
                  noble_id=value["noble_id"], metadata=tuple(tuple(x) for x in value["metadata"]))


make_observer_payload = observation_payload
state_from_observer_payload = state_from_observation
