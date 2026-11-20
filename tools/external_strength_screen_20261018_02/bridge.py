"""Lawful observer and atomic-turn bridge to the pinned SplendoRust engine.

Auxiliary decisions use explicit pre-reveal projections, never referee draws.
Those projections are inputs to the upstream non-main heuristic only: they are
not complete GameState observations and must not be determinized or fed to NN.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

from tools.external_strength_screen_20261018 import rules

COLORS = ("white", "blue", "green", "red", "black")
TOKENS = COLORS + ("gold",)
NONE = 255
CATALOGUE_SHA256 = "db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312"
RUST_CARD_IDS = (
    "T1_37 T1_42 T1_43 T1_40 T1_38 T1_39 T1_41 T1_44 "
    "T1_21 T1_20 T1_24 T1_22 T1_19 T1_25 T1_23 T1_26 "
    "T1_03 T1_06 T1_01 T1_07 T1_02 T1_05 T1_04 T1_08 "
    "T1_76 T1_77 T1_78 T1_75 T1_73 T1_74 T1_79 T1_80 "
    "T1_60 T1_59 T1_58 T1_56 T1_57 T1_61 T1_55 T1_62 "
    "T2_45 T2_46 T2_47 T2_49 T2_48 T2_50 "
    "T2_27 T2_28 T2_29 T2_31 T2_30 T2_32 "
    "T2_09 T2_10 T2_11 T2_13 T2_12 T2_14 "
    "T2_81 T2_82 T2_83 T2_85 T2_84 T2_86 "
    "T2_64 T2_63 T2_65 T2_66 T2_67 T2_68 "
    "T3_51 T3_52 T3_53 T3_54 T3_33 T3_34 T3_35 T3_36 "
    "T3_15 T3_16 T3_17 T3_18 T3_87 T3_89 T3_88 T3_90 T3_69 T3_70 T3_71 T3_72"
).split()
RUST_NOBLE_IDS = "N04 N05 N10 N09 N03 N06 N01 N08 N07 N02".split()
CARD_TO_RUST = {name: index for index, name in enumerate(RUST_CARD_IDS)}
NOBLE_TO_RUST = {name: index for index, name in enumerate(RUST_NOBLE_IDS)}


def require(condition, message):
    if not condition:
        raise ValueError("SplendoRust bridge: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _empty_reserve():
    return {"card": NONE, "tier": 0, "public": False}


def _empty_player():
    return {"tokens": [0] * 6, "bonuses": [0] * 5, "score": 0,
            "owned": [], "nobles": [], "reserved": [_empty_reserve() for _ in range(3)]}


def _state_and_legal(payload):
    require(rules.CATALOGUE_SHA256 == CATALOGUE_SHA256, "corrected catalogue changed")
    state = rules.state_from_observation(payload)
    require(not state.terminal, "terminal state has no decision")
    legal = rules.legal_actions(state)
    require(bool(legal), "authoritative legal set is empty")
    return state, legal


def encode_observation(payload):
    """Encode root public data only; normalize the starting seat to Rust seat0."""
    _state_and_legal(payload)
    view = payload["observation"]
    start, actor = view["start_player"], view["current_player"]
    players, counts = [_empty_player() for _ in range(4)], [0] * 4
    for seat, source in enumerate(view["players"]):
        index = (seat - start) % 2
        reserves = [{"card": NONE if r["card_id"] is None else CARD_TO_RUST[r["card_id"]],
                     "tier": r["tier"] - 1, "public": r["source"] == "visible"}
                    for r in source["reserves"]]
        counts[index] = len(reserves)
        players[index] = {"tokens": [source["tokens"][c] for c in TOKENS],
                          "bonuses": [source["bonuses"][c] for c in COLORS],
                          "score": source["score"],
                          "owned": sorted(CARD_TO_RUST[c] for c in source["purchased_card_ids"]),
                          "nobles": sorted(NOBLE_TO_RUST[n] for n in source["noble_ids"]),
                          "reserved": reserves + [_empty_reserve() for _ in range(3 - len(reserves))]}
    return {"count": 2, "viewer": (actor - start) % 2, "current": (actor - start) % 2,
            "bank": [view["bank_tokens"][c] for c in TOKENS],
            "market": [value for tier in ("1", "2", "3")
                       for value in [CARD_TO_RUST[c] for c in view["visible_tier_cards"][tier]] +
                       [NONE] * (4 - len(view["visible_tier_cards"][tier]))],
            "remaining": [view["deck_counts"][str(t)] for t in (1, 2, 3)],
            "nobles": sorted(NOBLE_TO_RUST[n] for n in view["nobles"]),
            "players": players, "reserved_counts": counts, "turns": view["turn_index"],
            "phase": {"kind": "Main"}, "final_round": view["pending_round_end"]}


def _vector(tokens, colors=TOKENS):
    counts = Counter(tokens)
    return [counts[c] for c in colors]


def _main(action):
    name = action.action_type.name
    if name == "TAKE_TOKENS":
        return {"kind": "Take", "tokens": _vector(action.take_tokens, COLORS)}
    if name in ("BUY_VISIBLE", "RESERVE_VISIBLE"):
        return {"kind": "BuyVisible" if name == "BUY_VISIBLE" else "ReserveVisible",
                "index": 4 * (action.tier - 1) + action.market_index}
    if name == "BUY_RESERVED":
        return {"kind": "BuyReserved", "index": action.reserved_index}
    if name == "RESERVE_DECK":
        return {"kind": "ReserveDeck", "tier": action.tier - 1}
    require(name == "PASS", "unknown authoritative action")
    return {"kind": "Pass"}


def _main_order(action):
    kind = action["kind"]
    if kind == "Take":
        tokens = action["tokens"]
        if 2 in tokens:
            return 1, tokens.index(2), 0
        return 0, sum((1 << i) for i, n in enumerate(tokens) if n), 0
    if kind in ("BuyVisible", "ReserveVisible"):
        return 2, action["index"], int(kind == "ReserveVisible")
    if kind in ("BuyReserved", "ReserveDeck"):
        return 3, action.get("index", action.get("tier")), int(kind == "ReserveDeck")
    return 4, 0, 0


def main_actions(payload):
    """Deduplicate atomic choices in exact upstream Main enumeration order."""
    _, legal = _state_and_legal(payload)
    values = {canonical(_main(a)): _main(a) for a in legal}
    result = sorted(values.values(), key=_main_order)
    if any(a["kind"] == "Pass" for a in result):
        require(result == [{"kind": "Pass"}], "PASS permitted only when forced")
    return result


def _choose_aux(observation, values, select_aux):
    require(bool(values), "empty auxiliary choice set")
    if len(values) == 1:
        return values[0]
    selected = select_aux(deepcopy(observation), deepcopy(values))
    require(canonical(selected) in {canonical(a) for a in values}, "invalid auxiliary choice")
    return deepcopy(selected)


def resolve_compound(payload, selected_main, select_aux):
    """Return one authoritative Action, resolving subchoices without future draws.

    select_aux receives a pre-reveal phase projection and ordered legal Rust
    actions. It must use only the unchanged upstream non-main heuristic, never
    NN/search/determinization. All calls belong to the same external turn clock.
    """
    _, legal = _state_and_legal(payload)
    require(canonical(selected_main) in {canonical(a) for a in main_actions(payload)}, "invalid Main choice")
    candidates = [a for a in legal if canonical(_main(a)) == canonical(selected_main)]
    observation = encode_observation(payload)
    actor = observation["current"]
    player = observation["players"][actor]
    kind = selected_main["kind"]
    card_id = None
    if kind in ("BuyVisible", "BuyReserved"):
        source_kind = "Market" if kind == "BuyVisible" else "Reserved"
        index = selected_main["index"]
        observation["phase"] = {"kind": "Payment", "source": {"kind": source_kind, "index": index}}
        payments = sorted({tuple(_vector(a.spend_tokens, COLORS)) for a in candidates}, reverse=True)
        chosen = _choose_aux(observation, [{"kind": "Pay", "tokens": list(p)} for p in payments], select_aux)
        candidates = [a for a in candidates if _vector(a.spend_tokens, COLORS) == chosen["tokens"]]
        paid = _vector(candidates[0].spend_tokens)
        for color, amount in enumerate(paid):
            player["tokens"][color] -= amount
            observation["bank"][color] += amount
        if kind == "BuyVisible":
            card_id = observation["market"][index]
            observation["market"][index] = NONE  # no future refill is supplied
        else:
            card_id = player["reserved"].pop(index)["card"]
            player["reserved"].append(_empty_reserve())
            observation["reserved_counts"][actor] -= 1
        card = rules.corrected_cards_by_id()[RUST_CARD_IDS[card_id]]
        player["bonuses"][COLORS.index(card.bonus_color)] += 1
        player["score"] += card.points
        player["owned"] = sorted(player["owned"] + [card_id])
    elif kind in ("Take", "ReserveVisible", "ReserveDeck"):
        gained = _vector(candidates[0].take_tokens)
        for color, amount in enumerate(gained):
            player["tokens"][color] += amount
            observation["bank"][color] -= amount
        if kind != "Take":
            if kind == "ReserveVisible":
                index = selected_main["index"]
                card_id, tier, public = observation["market"][index], index // 4, True
                observation["market"][index] = NONE
            else:
                card_id, tier, public = NONE, selected_main["tier"], False
                observation["remaining"][tier] -= 1
            slot = observation["reserved_counts"][actor]
            player["reserved"][slot] = {"card": card_id, "tier": tier, "public": public}
            observation["reserved_counts"][actor] += 1
        returns = sorted({tuple(_vector(a.return_tokens)) for a in candidates})
        if any(returns[0]):
            observation["phase"] = {"kind": "Return"}
            chosen = _choose_aux(observation, [{"kind": "Return", "tokens": list(r)} for r in returns], select_aux)
            candidates = [a for a in candidates if _vector(a.return_tokens) == chosen["tokens"]]
            for color, amount in enumerate(chosen["tokens"]):
                player["tokens"][color] -= amount
                observation["bank"][color] += amount
    else:
        require(kind == "Pass" and all(a.action_type.name == "PASS" for a in legal), "unforced PASS")

    noble_ids = {a.noble_id for a in candidates}
    if noble_ids != {None}:
        require(None not in noble_ids, "inconsistent compulsory noble alternatives")
        observation["phase"] = {"kind": "Noble"}
        values = [{"kind": "Noble", "index": n} for n in sorted(NOBLE_TO_RUST[n] for n in noble_ids)]
        chosen = _choose_aux(observation, values, select_aux)
        candidates = [a for a in candidates if NOBLE_TO_RUST[a.noble_id] == chosen["index"]]
    require(len(candidates) == 1 and candidates[0] in legal, "compound choice is ambiguous or illegal")
    return candidates[0]
