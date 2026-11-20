"""Observer-only adapter for the unchanged 4/8-candidate, 128-world planners.

The parent owns clocks, process isolation, launch authorization and artifact
writes. This module has no production collector, output path, retry or fallback.
The worker receives public state and its own private reserves only. A canonical
dummy hidden state recreates the same unseen pools before frozen determinization.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random

import torch

from tools.proposal_breadth_arena_128 import arena
from tools.local_player_qualification_128.selection import (
    BINDING, FITS, ROOT, SOURCE, SOURCE_HASHES, file_hash, require,
)
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles


old = arena.old
SCHEMA = "local_player_observer_v1"
PUBLIC_FIELDS = {"current_player", "turn_index", "start_player", "pending_round_end", "terminal",
                 "winner", "bank_tokens", "visible_tier_cards", "deck_counts", "nobles", "players"}
PLAYER_FIELDS = {"player_id", "tokens", "bonuses", "score", "reserves", "purchased_card_ids", "noble_ids"}
RESERVE_SOURCES = {"visible", "deck", "legacy_unknown"}
_INITIALIZED = False


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


@lru_cache(maxsize=201)
def _public_fit_and_stream(pair):
    """Reproduce public salts without constructing an environment-seed-bearing game spec."""
    order = sorted(range(201), key=lambda p: old.digest([arena.CONFIG["assignment_tag"], "main", p]))
    return FITS[order.index(pair) % 3], old.digest([arena.CONFIG["stream_tag"], "main", pair])


def _context(value):
    require(isinstance(value, dict) and set(value) == {"fit", "stream", "identity"}, "Invalid observer context fields")
    identity = arena.root_identity(value["identity"])
    require(identity[0] == "main" and type(value["fit"]) is int and value["fit"] in FITS, "Invalid saved root context")
    fit, stream = _public_fit_and_stream(identity[1] // 2)
    require(value["fit"] == fit and value["stream"] == stream, "Saved public stream/fit mismatch")
    return {"fit": value["fit"], "stream": value["stream"], "identity": identity}


def make_observer_payload(snapshot, context):
    """Trusted-parent conversion: never transmit hidden deck order or opponent blind IDs."""
    state = old.deserialize_state_snapshot(snapshot)
    clean = old.serialize_state_snapshot(state)
    observation = {key: deepcopy(clean[key]) for key in PUBLIC_FIELDS - {"players"}}
    observation["deck_counts"] = {str(k): v for k, v in clean["deck_counts"].items()}
    observation["players"] = []
    for seat, player in enumerate(state.players):
        item = clean["players"][seat]
        public = {key: deepcopy(item[key]) for key in PLAYER_FIELDS - {"reserves"}}
        public["reserves"] = [
            {"tier": card.tier, "source": player.reserved_card_source(i),
             "card_id": card.card_id if seat == state.current_player or player.reserved_card_source(i) == "visible" else None}
            for i, card in enumerate(player.reserved_cards)
        ]
        observation["players"].append(public)
    payload = {"schema": SCHEMA, "context": _context(context), "observation": observation}
    state_from_observer_payload(payload)
    return _json_copy(payload)


def _counts(value, keys):
    require(isinstance(value, dict) and set(value) == set(keys)
            and all(type(n) is int and n >= 0 for n in value.values()), "Invalid public count map")


def state_from_observer_payload(payload):
    """Build dummy hidden assignments from the known catalogue, never hidden evidence."""
    require(isinstance(payload, dict) and set(payload) == {"schema", "context", "observation"}
            and payload["schema"] == SCHEMA, "Unexpected observer payload fields")
    context = _context(payload["context"])
    view = payload["observation"]
    require(isinstance(view, dict) and set(view) == PUBLIC_FIELDS, "Unexpected public state fields")
    require(type(view["current_player"]) is int and view["current_player"] in (0, 1)
            and type(view["start_player"]) is int and view["start_player"] in (0, 1)
            and type(view["turn_index"]) is int and view["turn_index"] == context["identity"][2], "Invalid observer/turn")
    require(type(view["pending_round_end"]) is bool and view["terminal"] is False and view["winner"] is None,
            "Payload must be a live decision")
    _counts(view["bank_tokens"], ALL_TOKEN_TYPES)
    _counts(view["deck_counts"], ("1", "2", "3"))
    require(isinstance(view["visible_tier_cards"], dict) and set(view["visible_tier_cards"]) == {"1", "2", "3"},
            "Invalid public market")
    require(isinstance(view["players"], list) and len(view["players"]) == 2, "Invalid public players")
    by_tier = build_base_deck_by_tier()
    cards = {c.card_id: c for values in by_tier.values() for c in values}
    nobles = {n.noble_id for n in build_base_nobles()}
    seen, seen_nobles = set(), set()

    def known(card_id, tier=None):
        require(type(card_id) is str and card_id in cards and card_id not in seen, "Unknown or duplicate known card")
        require(tier is None or cards[card_id].tier == tier, "Known card tier mismatch")
        seen.add(card_id)
        return card_id

    def noble_ids(values):
        require(isinstance(values, list), "Invalid noble list")
        for value in values:
            require(type(value) is str and value in nobles and value not in seen_nobles, "Unknown or duplicate noble")
            seen_nobles.add(value)

    noble_ids(view["nobles"])
    for tier in (1, 2, 3):
        values = view["visible_tier_cards"][str(tier)]
        require(isinstance(values, list) and len(values) <= 4, "Invalid visible card list")
        for value in values:
            known(value, tier)
    snapshot = deepcopy(view)
    unknown = {tier: [] for tier in (1, 2, 3)}
    for seat, player in enumerate(view["players"]):
        require(isinstance(player, dict) and set(player) == PLAYER_FIELDS and type(player["player_id"]) is int
                and player["player_id"] == seat and type(player["score"]) is int and player["score"] >= 0,
                "Invalid public player fields")
        _counts(player["tokens"], ALL_TOKEN_TYPES)
        _counts(player["bonuses"], TOKEN_COLORS)
        noble_ids(player["noble_ids"])
        require(isinstance(player["purchased_card_ids"], list), "Invalid purchased cards")
        for value in player["purchased_card_ids"]:
            known(value)
        reserves = player["reserves"]
        require(isinstance(reserves, list) and len(reserves) <= 3, "Invalid reserve list")
        target = snapshot["players"][seat]
        del target["reserves"]
        target["reserved_card_ids"], target["reserved_card_sources"] = [], []
        for index, reserve in enumerate(reserves):
            require(isinstance(reserve, dict) and set(reserve) == {"tier", "source", "card_id"}
                    and type(reserve["tier"]) is int and reserve["tier"] in (1, 2, 3)
                    and reserve["source"] in RESERVE_SOURCES, "Invalid reserve descriptor")
            private = seat != view["current_player"] and reserve["source"] != "visible"
            if private:
                require(reserve["card_id"] is None, "Opponent private card identity was disclosed")
                unknown[reserve["tier"]].append((seat, index))
                target["reserved_card_ids"].append(None)
            else:
                target["reserved_card_ids"].append(known(reserve["card_id"], reserve["tier"]))
            target["reserved_card_sources"].append(reserve["source"])
    snapshot["hidden_tier_decks"] = {}
    for tier in (1, 2, 3):
        unseen = sorted(c.card_id for c in by_tier[tier] if c.card_id not in seen)
        require(len(unseen) == view["deck_counts"][str(tier)] + len(unknown[tier]), "Unseen catalogue/card count mismatch")
        for seat, index in unknown[tier]:
            snapshot["players"][seat]["reserved_card_ids"][index] = unseen.pop()
        snapshot["hidden_tier_decks"][str(tier)] = unseen
    return old.deserialize_state_snapshot(snapshot)


def hidden_variant(snapshot):
    """Trusted-parent privacy fixture: alter hidden allocation/order, preserving the view."""
    state = old.deserialize_state_snapshot(snapshot)
    other = state.players[1 - state.current_player]
    for tier in (1, 2, 3):
        hidden = state.hidden_tier_decks[tier]
        for index, card in enumerate(other.reserved_cards):
            if card.tier == tier and other.reserved_card_source(index) != "visible" and hidden:
                other.reserved_cards[index], hidden[0] = hidden[0], card
        hidden.reverse()
    return old.serialize_state_snapshot(state)


def initialize(registry=None, repo_root=ROOT):
    """Load authenticated unchanged models once; initialization is outside move timing."""
    global _INITIALIZED
    require(not _INITIALIZED, "Worker already initialized")
    repo_root = Path(repo_root)
    path = repo_root / SOURCE / "models_lock.json"
    require(file_hash(path) == SOURCE_HASHES["models_lock.json"], "Changed model registry")
    sealed = json.loads(path.read_text(encoding="utf-8"))
    require(registry is None or registry == sealed, "Provided registry differs from frozen registry")
    registry = sealed
    require(registry["manifest"] == BINDING and set(registry["models"]) == {f"anchor_{fit}" for fit in FITS},
            "Unexpected model registry")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    loaded, artifacts = {}, {}
    for model_id, item in registry["models"].items():
        require(item["source_fit"] == int(model_id.split("_")[1]) and item["unchanged_from_anchor"] is True,
                "Unexpected proposal checkpoint")
        model_path = repo_root / item["path"]
        require(file_hash(model_path) == item["sha256"], "Changed proposal checkpoint")
        loaded[model_id] = arena.StrongPolicyInference(model_path).model
        artifacts[model_id] = deepcopy(item)
    continuation = registry["continuation_model"]
    raw_path = repo_root / continuation["path"]
    require(file_hash(raw_path) == continuation["sha256"], "Changed continuation checkpoint")
    old._RAW = old.LegalActionAttentionScorerBot(raw_path)
    arena._MODELS.clear()
    arena._MODELS.update(loaded)
    arena._MODEL_ARTIFACTS.clear()
    arena._MODEL_ARTIFACTS.update(artifacts)
    _INITIALIZED = True
    return {"models": deepcopy(artifacts), "continuation_model": deepcopy(continuation),
            "backend": "unchanged_serial", "worlds": 128, "torch_threads": torch.get_num_threads()}


def choose(payload, breadth, boundary=lambda: None):
    """One exact decision; no cooperative deadline fallback or incomplete-world selection."""
    require(_INITIALIZED, "Worker models not initialized")
    require(type(breadth) is bool, "Invalid breadth role")
    boundary()
    state = state_from_observer_payload(payload)
    context = payload["context"]
    proposal = arena.proposal_for_root(state, context["stream"], f'anchor_{context["fit"]}', context["identity"], breadth)
    proposal.pop("timing", None)
    legal = old.SplendorEnv().legal_actions(state)
    record = {"root_identity": list(context["identity"]), "planner": "candidate" if breadth else "reference",
              "privacy": arena.baseline.privacy(state), "schedule": proposal["schedule"],
              "candidates": proposal["candidates"], "proposal": proposal,
              "scores": list(map(float, old._RAW.score_actions(state, legal))), "worlds": []}
    actions = [old.decode_action(action) for action in proposal["candidates"]["actions"]]
    for seed in proposal["schedule"]["world_seeds"]:
        boundary()
        world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
        row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": []}
        legal_world = old.SplendorEnv().legal_actions(world)
        for action in actions:
            require(action in legal_world, "Illegal forced proposal")
            value = old.trajectory(world, action)
            require(value["reason"] in old.VALID_NATURAL_TERMINATIONS, "Invalid continuation ending")
            value["wdl"] = old.outcome_value(value["winner"], state.current_player)
            row["outcomes"].append(value)
        record["worlds"].append(row)
    record.update(arena.select(record["worlds"], len(actions)))
    return _json_copy({"action": old.serialize_action(actions[record["choice"]]), "record": record})


def expected_from_saved(root, breadth):
    """Project saved 8x128 evidence to either exact policy, without replay or timings.

For 4x128 the original four-action prefix is assessed with the same saved worlds;
the winner is recomputed among that prefix, never inherited from the eight-way
winner. The parent should compare this projection independently as well.
"""
    require(type(breadth) is bool and root["planner"] == "candidate" and "failure" not in root,
            "Expected a complete saved candidate-side root")
    proposal = deepcopy(root["proposal"])
    proposal.pop("timing", None)
    require(proposal["breadth"] is True and proposal["candidates"] == root["candidates"], "Saved breadth mismatch")
    original = deepcopy(proposal["anchor_candidates"])
    support = len(original["keys"])
    require(1 <= support <= 4 and original["keys"] == root["candidates"]["keys"][:support]
            and original["actions"] == root["candidates"]["actions"][:support], "Saved anchor is not the original prefix")
    worlds = deepcopy(root["worlds"])
    if not breadth:
        proposal.update(candidates=original, breadth=False, added_indices={"ranked": [], "random": []})
        for world in worlds:
            world["outcomes"] = world["outcomes"][:support]
    record = {"root_identity": deepcopy(root["root_identity"]), "planner": "candidate" if breadth else "reference",
              "privacy": deepcopy(root["privacy"]), "schedule": deepcopy(root["schedule"]),
              "candidates": deepcopy(proposal["candidates"]), "proposal": proposal,
              "scores": deepcopy(root["scores"]), "worlds": worlds}
    record.update(arena.select(worlds, len(record["candidates"]["keys"])))
    if breadth:
        require(record["choice"] == root["choice"] and record["means"] == root["means"], "Saved eight-way choice mismatch")
    return _json_copy({"action": record["candidates"]["actions"][record["choice"]], "record": record})
