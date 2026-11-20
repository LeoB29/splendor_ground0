"""Explicit adapted pinned PPO on the separately corrected referee rules.

No upstream engine/import side effects, training, hidden state, or game runner.
Qualification alone evaluates selected authenticated upstream definitions in an
isolated namespace, with dependencies supplied explicitly and no upstream loader.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from collections import Counter
from collections.abc import ValuesView
import copy
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
import hashlib
from itertools import chain, combinations, combinations_with_replacement, repeat, starmap
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TypeVar, cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools.external_strength_screen_20261018 import rules

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/external_strength_preparation_20261018_01"
UP = "upstream/roeey777/src/splendor/"
QUAL = "qualification_sources/roeey777/src/splendor/"
CHECKPOINT = BASE / UP / "agents/our_agents/ppo/ppo_model.pth"
CHECKPOINT_SHA256 = "7fdb58b46d00305af42fad818b05549692e27ff1243cb63a76c6a858ba348df4"
SOURCE_HASHES = {
    UP + "splendor/features.py": "115a3e5bf1efe65ebadcba725386939195f7d6a6df742a95b2ab9602c418437b",
    UP + "splendor/splendor_model.py": "3a61c7c25d201da44cdb6d7c1404ba7737c4110eb7da7152c284bb4676af781c",
    QUAL + "splendor/gym/envs/actions.py": "dcd94a45844ed895ceeb7083657e3ec957f63cb2882d64aa4b27ea918f1c559f",
    QUAL + "agents/our_agents/ppo/input_norm.py": "0c2a318c540f054bfb4b8e307c6e5e669caea47757dc8a3d559331de3102b336",
    QUAL + "agents/our_agents/ppo/ppo_base.py": "9fad7e8ecb65eb46b133ceff55b75194613bab2801aac78d1114d585fc5807b0",
    QUAL + "agents/our_agents/ppo/network.py": "5a1c5a2e92b9a2fd13266519a5d046a466f606765994d3d57ae06c0c75715f66",
}
COLORS = ("black", "red", "yellow", "green", "blue", "white")
NORMAL = tuple(c for c in COLORS if c != "yellow")
ALPHABETICAL = tuple(sorted(COLORS))
COLOURS = dict(zip(("B", "r", "y", "g", "b", "w"), COLORS))
INPUT_SIZE, OUTPUT_SIZE = 265, 3510
ORIGINAL_PROFILE = "published_support_v1"
BUY_PROFILE = "official_rules_buy_support_v1"
ALIAS_PROFILE = "official_rules_buy_and_token_alias_v1"
DEFAULT_PROFILE = ALIAS_PROFILE
PROFILES = (ORIGINAL_PROFILE, BUY_PROFILE, ALIAS_PROFILE)
_runtime = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def _counts(values):
    if values is None:
        return None
    return tuple(sorted((c, n) for c, n in values.items() if n))


def _key(kind, collected=None, returned=None, position=None, noble=None):
    return kind, _counts(collected), _counts(returned), position, noble


def action_vocabulary():
    """Independently reproduce the pinned 3510 positional action indices."""
    values = [_key("PASS")] + [_key("PASS", noble=n) for n in range(5)]
    for tier in range(3):
        for index in range(4):
            position = (tier, index, -1)
            for noble in (None, *range(5)):
                values.append(_key("RESERVE", {}, {}, position, noble))
                values.append(_key("RESERVE", {"yellow": 1}, {}, position, noble))
                values.extend(_key("RESERVE", {"yellow": 1}, {c: 1}, position, noble) for c in NORMAL)
    for color in NORMAL:
        for noble in (None, *range(5)):
            values.append(_key("COLLECT_SAME", {color: 2}, {}, noble=noble))
            values.extend(_key("COLLECT_SAME", {color: 2}, {a: 1, b: 1}, noble=noble)
                          for a, b in combinations([c for c in COLORS if c != color], 2))
            for count in (1, 2):
                values.extend(_key("COLLECT_SAME", {color: 2}, {c: count}, noble=noble) for c in COLORS if c != color)
    for length in (1, 2, 3):
        for selected in combinations(NORMAL, length):
            for noble in (None, *range(5)):
                take = dict.fromkeys(selected, 1)
                values.append(_key("COLLECT_DIFF", take, {}, noble=noble))
                for count in range(1, length + 1):
                    values.extend(_key("COLLECT_DIFF", take, dict(Counter(returned)), noble=noble)
                                  for returned in combinations_with_replacement([c for c in COLORS if c not in selected], count))
    for reserved in range(3):
        values.extend(_key("BUY_RESERVE", position=(-1, -1, reserved), noble=n) for n in (None, *range(5)))
    for tier in range(3):
        for index in range(4):
            values.extend(_key("BUY_AVAILABLE", position=(tier, index, -1), noble=n) for n in (None, *range(5)))
    require(len(values) == len(set(values)) == OUTPUT_SIZE, "Pinned action vocabulary mismatch")
    return tuple(values)


ACTION_KEYS = action_vocabulary()
ACTION_INDEX = {key: index for index, key in enumerate(ACTION_KEYS)}


def _external_color(color):
    return "yellow" if color == "gold" else color


def _token_counts(tokens):
    return dict(Counter(_external_color(c) for c in tokens))


def _serialize(action):
    return json.loads(json.dumps(rules.serialize_action(action)))


def _native_signature(state, action):
    """Lossless semantic signature for intersecting authoritative legal actions."""
    kind = action.action_type.name
    noble = next((i for i, n in enumerate(state.nobles) if n.noble_id == action.noble_id), None)
    if kind in ("BUY_VISIBLE", "RESERVE_VISIBLE"):
        position = (action.tier - 1, action.market_index, -1)
    elif kind == "BUY_RESERVED":
        position = (-1, -1, action.reserved_index)
    else:
        position = None
    name = {"BUY_VISIBLE": "BUY_AVAILABLE", "BUY_RESERVED": "BUY_RESERVE", "RESERVE_VISIBLE": "RESERVE",
            "PASS": "PASS", "RESERVE_DECK": "UNSUPPORTED_BLIND"}.get(kind)
    take, returned = _token_counts(action.take_tokens), _token_counts(action.return_tokens)
    if kind == "TAKE_TOKENS":
        name = "COLLECT_SAME" if len(take) == 1 and sum(take.values()) == 2 else "COLLECT_DIFF"
    if kind.startswith("BUY"):
        return name, None, _counts(_token_counts(action.spend_tokens)), position, noble
    if kind == "PASS":
        return name, None, None, position, noble
    return name, _counts(take), _counts(returned), position, noble


def _token_effect(take, returned, noble):
    take, returned = dict(take or ()), dict(returned or ())
    return tuple(take.get(c, 0) - returned.get(c, 0) for c in COLORS), noble


def _source_token_key_allowed(state, key):
    """Literal published take/return rules for an existing checkpoint index."""
    kind, take, returned, position, noble = key
    if kind not in ("COLLECT_DIFF", "COLLECT_SAME") or position is not None:
        return False
    player = state.players[state.current_player]
    eligible = [i for i, n in enumerate(state.nobles)
                if all(player.bonuses[c] >= amount for c, amount in n.requirement.items())]
    if noble not in (eligible or [None]):
        return False
    take, returned = dict(take or ()), dict(returned or ())
    count = sum(take.values())
    available = sum(state.bank_tokens[c] > 0 for c in NORMAL)
    total = sum(player.tokens.values())
    minimum = min(3 if total <= 7 else 2 if total == 8 else 1, available)
    if kind == "COLLECT_DIFF":
        if not (minimum <= count <= min(3, available) and count > 0 and all(n == 1 and c in NORMAL for c, n in take.items())):
            return False
    elif len(take) != 1 or count != 2 or any(c not in NORMAL or state.bank_tokens[c] < 4 for c in take):
        return False
    if any(state.bank_tokens[c] < amount for c, amount in take.items()):
        return False
    if sum(returned.values()) != max(total + count - 10, 0):
        return False
    return all(c not in take and amount <= player.tokens["gold" if c == "yellow" else c] for c, amount in returned.items())


def legal_mapping(state, profile=DEFAULT_PROFILE, *, require_nonempty=True):
    """Pinned support restrictions intersected with the corrected referee's legal set."""
    require(profile in PROFILES, "Unknown PPO support profile")
    legal = rules.CorrectedEnv().legal_actions(state)
    player = state.players[state.current_player]
    mapping, excluded = {}, Counter()
    for action in legal:
        kind = action.action_type.name
        if kind == "RESERVE_DECK":
            excluded["blind_reserve_not_in_checkpoint_vocabulary"] += 1
            continue
        signature = _native_signature(state, action)
        name, take, returned, position, noble = signature
        if kind.startswith("BUY"):
            card = state.visible_tier_cards[action.tier][action.market_index] if kind == "BUY_VISIBLE" else player.reserved_cards[action.reserved_index]
            if profile == ORIGINAL_PROFILE and player.bonuses[card.bonus_color] == 7:
                excluded["pinned_seven_card_pruning"] += 1
                continue
            cost = {color: max(card.cost.get(color, 0) - player.bonuses[color], 0) for color in NORMAL}
            expected = {color: min(cost[color], player.tokens[color]) for color in NORMAL}
            expected["gold"] = sum(cost[color] - expected[color] for color in NORMAL)
            expected = {color: count for color, count in expected.items() if count}
            if dict(Counter(action.spend_tokens)) != expected:
                excluded["noncanonical_payment"] += 1
                continue
            key = _key(name, position=position, noble=noble)
        else:
            key = signature
        if key not in ACTION_INDEX:
            excluded["pinned_return_or_action_pruning"] += 1
            continue
        index = ACTION_INDEX[key]
        payload = _serialize(action)
        require(index not in mapping or mapping[index] == payload, "Ambiguous canonical action mapping")
        mapping[index] = payload
    aliases = []
    if profile == ALIAS_PROFILE:
        representatives = {}
        for action in legal:
            if action.action_type.name != "TAKE_TOKENS":
                continue
            _, take, returned, _, noble = _native_signature(state, action)
            effect = _token_effect(take, returned, noble)
            payload = _serialize(action)
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if effect not in representatives or canonical < representatives[effect][0]:
                representatives[effect] = canonical, payload
        for index, key in enumerate(ACTION_KEYS):
            if index in mapping or not _source_token_key_allowed(state, key):
                continue
            effect = _token_effect(key[1], key[2], key[4])
            if effect in representatives:
                mapping[index] = representatives[effect][1]
                aliases.append(index)
    require(bool(mapping) or not require_nonempty, "Empty translated PPO legal mask")
    mask = np.zeros(OUTPUT_SIZE, dtype=np.float64)
    mask[list(mapping)] = 1
    return mask, mapping, {"profile": profile, "authoritative_legal_actions": len(legal), "support": len(mapping), "excluded": dict(excluded),
                           "token_alias_indices": aliases}


def _missing(card, power):
    return [amount - power[color] for color, amount in card.cost.items() if amount > power[color]]


def extract_features(state):
    """Exact pinned raw feature order, before its deliberate float32 rounding."""
    player = state.players[state.current_player]
    power = {color: player.tokens[color] + player.bonuses[color] for color in NORMAL}
    distances = {color: [] for color in NORMAL}
    gems, turns, reserved_gems, reserved_turns = [], [], [0] * 3, [0] * 3
    market = [c for tier in (1, 2, 3) for c in (state.visible_tier_cards[tier] + [None] * (4 - len(state.visible_tier_cards[tier])))]
    for card in market:
        missing = _missing(card, power) if card else []
        gems.append(sum(missing))
        turns.append(max(np.ceil(sum(missing) / 3), *missing) if missing else 0)
        if card:
            distances[card.bonus_color].append(sum(missing))
    for index, card in enumerate(player.reserved_cards):
        missing = _missing(card, power)
        reserved_gems[index] = sum(missing)
        reserved_turns[index] = max(np.ceil(sum(missing) / 3), *missing) if missing else 0
        distances[card.bonus_color].append(sum(missing))
    for values in distances.values():
        values.sort()
    noble_gems, noble_cards = [0] * 5, [0] * 5
    for index, noble in enumerate(state.nobles):
        missing = {color: count - player.bonuses[color] for color, count in noble.requirement.items() if count > player.bonuses[color]}
        noble_cards[index] = sum(missing.values())
        noble_gems[index] = sum(sum(distances[color][:count]) + 14 * max(count - len(distances[color]), 0)
                                 for color, count in missing.items())
    before, after = [0] * 3, [0] * 3
    for seat, rival in enumerate(state.players):
        if seat < state.current_player:
            before[seat] = rival.score
        elif seat > state.current_player:
            after[seat - 1] = rival.score
    own_turns = state.turn_index // 2 + int(state.turn_index % 2 == 1 and state.current_player == state.start_player)
    metrics = [1, own_turns, player.score, int(player.score >= 15), len(player.purchased_cards), len(player.reserved_cards),
               np.var(list(power.values())), player.tokens["gold"], sum(player.tokens.values())]
    metrics += [player.bonuses[c] for c in NORMAL] + list(power.values()) + [np.log(1 + float(v)) for v in power.values()]
    metrics += before + after + gems + turns + reserved_gems + reserved_turns + noble_gems + noble_cards
    vectors = []
    for card in market + player.reserved_cards + [None] * (3 - len(player.reserved_cards)):
        vectors.extend(([int(card.bonus_color == c) for c in ALPHABETICAL] +
                        [card.cost.get(c, 0) for c in ALPHABETICAL if c != "yellow"] + [card.tier - 1, card.points])
                       if card else [0] * 13)
    result = np.asarray(metrics + vectors, dtype=np.float64)
    require(result.shape == (INPUT_SIZE,) and np.isfinite(result).all(), "Invalid PPO features")
    return result


class InputNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1, INPUT_SIZE))
        self.register_buffer("running_var", torch.ones(1, INPUT_SIZE))

    def forward(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)


class AdaptedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = InputNormalization()
        layers = []
        for size in (INPUT_SIZE, 128, 128, 128):
            layers += [nn.Linear(size, 128), nn.LayerNorm(128), nn.Dropout(.2), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.actor, self.critic = nn.Linear(128, OUTPUT_SIZE), nn.Linear(128, 1)

    def forward(self, x, mask):
        hidden = self.net(self.input_norm(x.unsqueeze(0) if x.ndim == 1 else x))
        logits = self.actor(hidden)
        return F.softmax(torch.where(mask == 0, -1e8, logits), dim=1), self.critic(hidden), logits


def safe_checkpoint():
    require(file_hash(CHECKPOINT) == CHECKPOINT_SHA256, "Pinned PPO checkpoint changed")
    checkpoint = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    require(isinstance(checkpoint, dict) and set(checkpoint) == {"model_state_dict", "running_mean", "running_var"}, "Unexpected checkpoint keys")
    state = checkpoint["model_state_dict"]
    template = AdaptedNetwork().double()
    require(set(state) == set(template.state_dict()), "Unexpected parameter/buffer keys")
    for key, expected in template.state_dict().items():
        tensor = state[key]
        require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and tensor.dtype == torch.float64
                and tensor.shape == expected.shape and torch.isfinite(tensor).all().item(), f"Invalid tensor {key}")
    for key in ("running_mean", "running_var"):
        tensor = checkpoint[key]
        require(type(tensor) is torch.Tensor and tensor.shape == (1, INPUT_SIZE) and tensor.dtype == torch.float64
                and tensor.device.type == "cpu" and torch.isfinite(tensor).all().item(), f"Invalid {key}")
    require((checkpoint["running_var"] >= 0).all().item(), "Invalid input variance")
    return checkpoint


def _load_network(model, checkpoint):
    model.double().cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.input_norm.running_mean = checkpoint["running_mean"].squeeze(0).clone()
    model.input_norm.running_var = checkpoint["running_var"].squeeze(0).clone()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class PPOAdapter:
    def __init__(self, profile=DEFAULT_PROFILE):
        require(profile in PROFILES, "Unknown PPO support profile")
        self.profile = profile
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        checkpoint = safe_checkpoint()
        self.model = _load_network(AdaptedNetwork(), checkpoint)
        self.receipt = {"checkpoint_sha256": CHECKPOINT_SHA256, "safe_weights_only": True, "input_size": INPUT_SIZE,
                        "output_size": OUTPUT_SIZE, "dtype": "float64_after_float32_features", "device": "cpu",
                        "eval_mode": True, "repair": "Pinned agent unpacked two outputs although network returns three; adapter explicitly selects probability output.",
                        "profile": profile,
                        "policy": "Explicit adapted policy on corrected rules; original token-index semantics with exact successor aliases only in alias profile; coloured-first minimum-gold payment; first argmax",
                        "seven_card_buy_pruning_removed": profile != ORIGINAL_PROFILE,
                        "exact_token_effect_aliases": profile == ALIAS_PROFILE}

    @torch.inference_mode()
    def evaluate(self, payload):
        state = rules.state_from_observation(payload)
        require(not state.terminal, "No PPO decision on terminal state")
        features = extract_features(state).astype(np.float32).astype(np.float64)
        mask, mapping, info = legal_mapping(state, self.profile)
        probabilities, value, logits = self.model(torch.from_numpy(features), torch.from_numpy(mask))
        require(torch.isfinite(probabilities).all().item() and torch.isfinite(value).all().item()
                and torch.isfinite(logits).all().item(), "Nonfinite PPO output")
        index = int(probabilities.argmax().item())
        require(index in mapping and mask[index] == 1, "PPO selected a masked action")
        return {"state": state, "features": features, "mask": mask, "mapping": mapping, "info": info,
                "probabilities": probabilities[0].numpy(), "logits": logits[0].numpy(), "value": value.numpy(), "index": index}

    def choose(self, payload):
        result = self.evaluate(payload)
        return {"action": result["mapping"][result["index"]], "diagnostics": {
            "policy": "adapted_pinned_ppo_v1", "checkpoint_sha256": CHECKPOINT_SHA256, "action_index": result["index"],
            "profile": self.profile,
            "token_alias_count": len(result["info"]["token_alias_indices"]),
            "selected_token_alias": result["index"] in result["info"]["token_alias_indices"],
            "support": result["info"]["support"], "authoritative_legal_actions": result["info"]["authoritative_legal_actions"],
            "features_sha256": array_hash(result["features"]), "mask_sha256": array_hash(result["mask"]),
            "output_sha256": array_hash(result["probabilities"])}}


def initialize():
    global _runtime
    _runtime = PPOAdapter()
    return _runtime


def choose(payload):
    require(_runtime is not None, "PPO not initialized")
    return _runtime.choose(payload)


def _exec_definitions(relative, namespace, names=None):
    """Qualification-only pinned reference definitions, without imports or loaders."""
    path = BASE / relative
    require(file_hash(path) == SOURCE_HASHES[relative], "Pinned reference source changed")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            if names is None or getattr(node, "name", None) in names:
                body.append(node)
    code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec")
    exec(code, namespace)


def pinned_reference():
    from sklearn.preprocessing import OneHotEncoder
    namespace = {"__name__": __name__, "np": np, "torch": torch, "nn": nn, "F": F, "ABC": ABC,
        "abstractmethod": abstractmethod, "override": lambda f: f, "cache": cache, "chain": chain, "repeat": repeat,
        "starmap": starmap, "ValuesView": ValuesView, "OneHotEncoder": OneHotEncoder, "Color": str,
        "COLOURS": COLOURS, "NORMAL_COLORS": list(NORMAL), "WILDCARD": "yellow", "RESERVED": "yellow",
        "NUMBER_OF_TIERS": 3, "MAX_TIER_CARDS": 4, "MAX_RESERVED": 3, "MAX_NOBLES": 5, "MAX_RIVALS": 3,
        "MAX_SCORE": 22, "MAX_GEMS": 10, "MAX_WILDCARDS": 5, "ROUNDS_LIMIT": 100, "WINNING_SCORE_TRESHOLD": 15,
        "MAX_CARD_GEMS_DIST_1": 5, "MAX_CARD_GEMS_DIST_2": 8, "MAX_CARD_GEMS_DIST_3": 14,
        "MAX_CARD_TURNS_DIST_1": 4, "MAX_CARD_TURNS_DIST_2": 6, "MAX_CARD_TURNS_DIST_3": 7,
        "MAX_NOBLE_CARDS_DISTANCE": 9, "MAX_NOBLE_GEMS_DISTANCE": 126,
        "VERY_SMALL_EPSILON": 1e-8, "HIDDEN_DIMS": [128] * 4, "DROPOUT": .2, "HUGE_NEG": -1e8,
        "dataclass": dataclass, "Enum": Enum, "auto": auto, "TypeVar": TypeVar, "cast": cast,
        "combinations": combinations, "combinations_with_replacement": combinations_with_replacement, "Counter": Counter,
        "splendor_utils": SimpleNamespace(COLOURS=COLOURS), "ActionType": dict, "BuyAction": dict, "GemsCount": dict,
        "itertools": itertools, "copy": copy, "GameRule": object}
    _exec_definitions(UP + "splendor/features.py", namespace)
    _exec_definitions(QUAL + "splendor/gym/envs/actions.py", namespace)
    _exec_definitions(UP + "splendor/splendor_model.py", namespace, {"SplendorGameRule"})
    _exec_definitions(QUAL + "agents/our_agents/ppo/input_norm.py", namespace)
    _exec_definitions(QUAL + "agents/our_agents/ppo/ppo_base.py", namespace, {"PPOBase", "unused"})
    _exec_definitions(QUAL + "agents/our_agents/ppo/network.py", namespace, {"PPO"})
    namespace["reference_model"] = _load_network(namespace["PPO"](INPUT_SIZE, OUTPUT_SIZE), safe_checkpoint())
    return namespace


def _reference_state(state):
    def card(c):
        return SimpleNamespace(colour=c.bonus_color, code=c.card_id, cost=dict(c.cost), deck_id=c.tier - 1, points=c.points)
    agents = []
    for seat, player in enumerate(state.players):
        cards = {color: [] for color in COLORS}
        for owned in player.purchased_cards:
            cards[owned.bonus_color].append(card(owned))
        cards["yellow"] = [card(c) for c in player.reserved_cards] if seat == state.current_player else [None] * len(player.reserved_cards)
        turns = state.turn_index // 2 + int(state.turn_index % 2 and seat == state.start_player)
        agents.append(SimpleNamespace(id=seat, gems={c: player.tokens["gold" if c == "yellow" else c] for c in COLORS},
                      cards=cards, score=player.score, agent_trace=SimpleNamespace(action_reward=[None] * turns)))
    board = SimpleNamespace(dealt=[[card(c) for c in state.visible_tier_cards[t]] + [None] * (4 - len(state.visible_tier_cards[t])) for t in (1, 2, 3)],
                            nobles=[(n.noble_id, dict(n.requirement)) for n in state.nobles],
                            gems={c: state.bank_tokens["gold" if c == "yellow" else c] for c in COLORS})
    return SimpleNamespace(agents=agents, board=board, agent_to_move=state.current_player)


def _reference_action_signature(action, reference_state, seat):
    position = None
    if action.get("card") is not None:
        card = action["card"]
        if action["type"] == "buy_reserve":
            position = (-1, -1, reference_state.agents[seat].cards["yellow"].index(card))
        else:
            position = (card.deck_id, reference_state.board.dealt[card.deck_id].index(card), -1)
    noble = reference_state.board.nobles.index(action["noble"]) if action.get("noble") else None
    return (action["type"].upper(), _counts(action.get("collected_gems")), _counts(action.get("returned_gems")), position, noble)


def _reference_token_successor(state, source_action):
    """Qualification-only independent source token effect and official turn end."""
    following = copy.deepcopy(state)
    player = following.players[state.current_player]
    for name, sign in (("collected_gems", 1), ("returned_gems", -1)):
        for color, amount in source_action.get(name, {}).items():
            local_color = "gold" if color == "yellow" else color
            player.tokens[local_color] += sign * amount
            following.bank_tokens[local_color] -= sign * amount
    if source_action.get("noble"):
        noble_id = source_action["noble"][0]
        noble = next(n for n in following.nobles if n.noble_id == noble_id)
        following.nobles.remove(noble)
        player.nobles.append(noble)
        player.score += noble.points
    following.turn_index += 1
    following.current_player = 1 - state.current_player
    following.pending_round_end = following.pending_round_end or player.score >= 15
    if following.pending_round_end and following.current_player == following.start_player:
        following.terminal = True
        rank = [(p.score, -len(p.purchased_cards)) for p in following.players]
        following.winner = None if rank[0] == rank[1] else int(rank[1] > rank[0])
    return following


def qualify_payloads(payloads, destination=None, *, profile=DEFAULT_PROFILE):
    """Independent pinned-source features, masks and network comparison; no games."""
    runtime, reference = PPOAdapter(profile), pinned_reference()
    vocabulary = reference["ALL_ACTIONS"]
    external_keys = [(_a.type_enum.name, _counts(_a.collected_gems), _counts(_a.returned_gems),
                      None if _a.position is None else (_a.position.tier, _a.position.card_index, _a.position.reserved_index), _a.noble_index)
                     for _a in vocabulary]
    require(tuple(external_keys) == ACTION_KEYS, "Independent pinned action index order differs")
    rows = []
    for name, payload in payloads:
        result = runtime.evaluate(payload)
        state, proxy = result["state"], _reference_state(result["state"])
        features = reference["extract_metrics_with_cards"](proxy, state.current_player).astype(np.float32).astype(np.float64)
        require(np.array_equal(features, result["features"]), f"Pinned features differ: {name}")
        native = {_native_signature(state, a): _serialize(a) for a in rules.CorrectedEnv().legal_actions(state)}
        upstream_legal = object.__new__(reference["SplendorGameRule"]).getLegalActions(proxy, state.current_player)
        independent_mapping = {}
        for action in upstream_legal:
            signature = _reference_action_signature(action, proxy, state.current_player)
            if signature not in native:
                continue
            element = reference["Action"].to_action_element(action, proxy, state.current_player)
            index = vocabulary.index(element)
            require(index not in independent_mapping or independent_mapping[index] == native[signature], "Pinned action alias ambiguity")
            independent_mapping[index] = native[signature]
        original_mapping = dict(independent_mapping)
        if profile != ORIGINAL_PROFILE:
            # The independent reference starts with the executed published mask,
            # then adds only official purchases suppressed by its literal ==7 cap.
            for action in rules.CorrectedEnv().legal_actions(state):
                if action.action_type.name not in ("BUY_VISIBLE", "BUY_RESERVED"):
                    continue
                player = state.players[state.current_player]
                card = state.visible_tier_cards[action.tier][action.market_index] if action.action_type.name == "BUY_VISIBLE" else player.reserved_cards[action.reserved_index]
                if player.bonuses[card.bonus_color] != 7:
                    continue
                paid = Counter(action.spend_tokens)
                coloured = {c: min(max(card.cost.get(c, 0) - player.bonuses[c], 0), player.tokens[c]) for c in NORMAL}
                if any(paid[c] != coloured[c] for c in NORMAL):
                    continue
                signature = _native_signature(state, action)
                key = (signature[0], None, None, signature[3], signature[4])
                if key in external_keys:
                    independent_mapping[external_keys.index(key)] = _serialize(action)
        added_buys = sorted(set(independent_mapping) - set(original_mapping))
        alias_checks = 0
        independent_aliases = []
        if profile == ALIAS_PROFILE:
            for action in upstream_legal:
                if action["type"] not in ("collect_diff", "collect_same"):
                    continue
                element = reference["Action"].to_action_element(action, proxy, state.current_player)
                index = vocabulary.index(element)
                if index in independent_mapping:
                    continue
                expected = _reference_token_successor(state, action)
                expected_snapshot = rules.serialize_snapshot(expected)
                target = expected.players[state.current_player].tokens
                candidates = []
                for local in rules.CorrectedEnv().legal_actions(state):
                    if local.action_type.name != "TAKE_TOKENS" or local.noble_id != (action["noble"][0] if action.get("noble") else None):
                        continue
                    tokens = Counter(state.players[state.current_player].tokens)
                    tokens.update(local.take_tokens)
                    tokens.subtract(local.return_tokens)
                    if dict(tokens) != target:
                        continue
                    following = rules.CorrectedEnv().step(state, local)
                    require(rules.serialize_snapshot(following) == expected_snapshot, f"Token alias changes successor: {name}/{index}")
                    alias_checks += 1
                    candidates.append(_serialize(local))
                if candidates:
                    independent_mapping[index] = min(candidates, key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
                    independent_aliases.append(index)
        require(independent_mapping == result["mapping"], f"Independently adapted legal subset differs: {name}")
        require(independent_aliases == result["info"]["token_alias_indices"] or sorted(independent_aliases) == result["info"]["token_alias_indices"], "Independent token alias indices differ")
        mask = np.zeros(OUTPUT_SIZE)
        mask[list(independent_mapping)] = 1
        require(np.array_equal(mask, result["mask"]), "Pinned legal mask differs")
        logits = []
        hook = reference["reference_model"].actor.register_forward_hook(lambda _m, _a, output: logits.append(output.detach()))
        with torch.inference_mode():
            probabilities, value, third = reference["reference_model"](torch.from_numpy(features), torch.from_numpy(mask))
        hook.remove()
        require(third is None and np.array_equal(logits[0][0].numpy(), result["logits"])
                and np.array_equal(probabilities[0].numpy(), result["probabilities"])
                and np.array_equal(value.numpy(), result["value"]), f"Pinned neural outputs differ: {name}")
        require(int(probabilities.argmax().item()) == result["index"], "Pinned first argmax differs")
        repeated = runtime.choose(payload)
        require(repeated == runtime.choose(copy.deepcopy(payload)), "Deterministic repeated decision differs")
        rows.append({"fixture": name, "observation_sha256": digest(payload), **repeated["diagnostics"],
                     "original_mask_retained": all(independent_mapping.get(i) == a for i, a in original_mapping.items()),
                     "original_support": len(original_mapping), "added_buy_indices": added_buys,
                     "token_alias_indices": sorted(independent_aliases), "token_alias_successor_checks": alias_checks,
                     "exact_features_mask_logits_probabilities_value_and_action": True})
    receipt = {"checkpoint": runtime.receipt, "ruleset_id": rules.RULESET_ID, "catalogue_sha256": rules.CATALOGUE_SHA256,
               "fixture_count": len(rows), "fixtures": rows, "action_vocabulary_exact": True,
               "profile": profile, "checkpoint_sha256": CHECKPOINT_SHA256,
               "token_alias_successor_checks": sum(r["token_alias_successor_checks"] for r in rows),
               "qualification": "Pinned features and network arithmetic under independently verified explicit profile mask; three-output unpack repair; no games",
               "reference_source_hashes": SOURCE_HASHES, "training_updates": 0, "arena_games": 0}
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    return receipt
