"""Synthetic observer, selection and exact-prefix tests; no fitted model is run."""
from copy import deepcopy
import random

import pytest

from tools.local_player_qualification_128 import local, selection
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.training.model import LegalActionAttentionScorerConfig


def fixture_state(seat=0, privacy=True):
    state = local.old.SplendorEnv(seed=37491).initial_state()
    state.current_player = seat
    if privacy:
        for player_id, tier in ((seat, 1), (1 - seat, 1), (1 - seat, 2)):
            state.players[player_id].reserved_cards.append(state.hidden_tier_decks[tier].pop())
            state.players[player_id].reserved_card_sources.append("deck")
            state.deck_counts[tier] -= 1
        public = state.visible_tier_cards[3].pop()
        state.players[1 - seat].reserved_cards.append(public)
        state.players[1 - seat].reserved_card_sources.append("visible")
    spec = local.arena.specs("main")[seat]
    context = {"fit": spec["fit"], "stream": spec["search_stream"], "identity": ["main", seat, 0]}
    return state, context


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("privacy", [False, True])
def test_observer_reconstructs_exact_sampled_worlds_without_private_ids(seat, privacy):
    state, context = fixture_state(seat, privacy)
    snapshot = local.old.serialize_state_snapshot(state)
    payload = local.make_observer_payload(snapshot, context)
    dummy = local.state_from_observer_payload(payload)
    assert canonical_public_belief_key(dummy, seat) == canonical_public_belief_key(state, seat)
    assert local.old.SplendorEnv().legal_actions(dummy) == local.old.SplendorEnv().legal_actions(state)
    original_row, original_keys, original_actions = local.arena.encode_position(state, LegalActionAttentionScorerConfig())
    dummy_row, dummy_keys, dummy_actions = local.arena.encode_position(dummy, LegalActionAttentionScorerConfig())
    assert original_keys == dummy_keys and original_actions == dummy_actions
    for key in ("observation", "features", "groups"):
        assert local.torch.equal(original_row[key], dummy_row[key])
    assert "hidden_tier_decks" not in payload["observation"]
    for row in payload["observation"]["players"][1 - seat]["reserves"]:
        if row["source"] != "visible":
            assert row["card_id"] is None
    for seed in (0, 115, "exact-frozen-world-seed"):
        original = local.old.determinize_for_observer(state, seat, random.Random(seed))
        reconstructed = local.old.determinize_for_observer(dummy, seat, random.Random(seed))
        assert local.old.serialize_state_snapshot(original) == local.old.serialize_state_snapshot(reconstructed)


@pytest.mark.parametrize("seat", [0, 1])
def test_hidden_reallocation_preserves_exact_observer_input(seat):
    state, context = fixture_state(seat)
    snapshot = local.old.serialize_state_snapshot(state)
    variant = local.hidden_variant(snapshot)
    assert variant != snapshot
    assert local.make_observer_payload(snapshot, context) == local.make_observer_payload(variant, context)
    assert snapshot == local.old.serialize_state_snapshot(state)


@pytest.mark.parametrize("mutation", ["top_secret", "hidden_decks", "private_id", "count", "duplicate", "context_seed", "bool_seat"])
def test_observer_rejects_disclosure_or_inconsistent_input(mutation):
    state, context = fixture_state()
    payload = local.make_observer_payload(local.old.serialize_state_snapshot(state), context)
    view = payload["observation"]
    if mutation == "top_secret":
        payload["environment_seed"] = 100
    elif mutation == "hidden_decks":
        view["hidden_tier_decks"] = {}
    elif mutation == "private_id":
        view["players"][1]["reserves"][0]["card_id"] = state.players[1].reserved_cards[0].card_id
    elif mutation == "count":
        view["deck_counts"]["1"] += 1
    elif mutation == "duplicate":
        view["visible_tier_cards"]["1"][1] = view["visible_tier_cards"]["1"][0]
    elif mutation == "context_seed":
        payload["context"]["environment_seed"] = 15
    else:
        view["current_player"] = False
    with pytest.raises(ValueError):
        local.state_from_observer_payload(payload)


def metadata(index, turn, fit, support=19, privacy=(0, 0), arm="candidate"):
    actor = index % 2 if arm == "candidate" else 1 - index % 2
    return {"root_id": f"{index:04d}:{turn:03d}", "index": index, "pair": index // 2,
            "fit": fit, "turn": turn, "arm": arm, "actor": actor,
            "phase": 0 if turn < 20 else 1 if turn < 40 else 2,
            "privacy": {"own": privacy[0], "opponent": privacy[1]},
            "privacy_category": int(bool(privacy[0])) + 2 * int(bool(privacy[1])),
            "support": min(support, 8 if arm == "candidate" else 4), "canonical_support": support,
            "worlds": 128, "sha256": selection.digest([index, turn])}


def inventory():
    rows = []
    for fit_i, fit in enumerate(selection.FITS):
        for seat in (0, 1):
            for phase, turn in enumerate((0, 20, 40)):
                rows.append(metadata(2 * (fit_i * 3 + phase) + seat, turn, fit))
    return rows


def test_metadata_selection_balanced_order_invariant_and_source_bound():
    rows = inventory() + [metadata(70, 52, selection.FITS[0], 2, (1, 1)),
                          metadata(72, 54, selection.FITS[0], 6, (1, 1)),
                          metadata(74, 56, selection.FITS[0], 10, (1, 1)),
                          metadata(76, 58, selection.FITS[0], 10, (1, 1)),
                          metadata(78, 60, selection.FITS[0], 10, (1, 1)),
                          metadata(80, 60, selection.FITS[0], 2, (1, 1), "reference")]
    a = selection.select_inventory(rows)
    b = selection.select_inventory(list(reversed(rows)))
    assert a == b
    assert len(a["core"]) == 18
    assert [(r["fit"], r["actor"], r["phase"]) for r in a["core"]] == list(selection.CELLS)
    selected = a["core"] + a["edges"]
    assert len({r["root_id"] for r in selected}) == len(selected)
    assert all(r["arm"] == "candidate" for r in selected)
    assert all(r["sha256"] == next(v["sha256"] for v in rows if v["root_id"] == r["root_id"]) for r in selected)
    assert all(r["path"].endswith(f'game_{r["index"]:04d}/turn_{r["turn"]:03d}.json') for r in selected)
    assert a["selection_reads_outcomes_or_timings"] is False
    assert a["synthetic_edges"] == ["ordered_first_maximum_tie"]


def test_absent_metadata_edges_are_reported_without_replacement():
    plan = selection.select_inventory(inventory())
    assert plan["edges"] == []
    assert len(plan["absent_edge_strata"]) == 5


@pytest.mark.parametrize("mutation", ["missing_cell", "duplicate", "outcomes", "timing", "bool_index", "wrong_phase"])
def test_selection_rejects_bad_or_outcome_bearing_metadata(mutation):
    rows = inventory()
    if mutation == "missing_cell":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation in ("outcomes", "timing"):
        rows[0][mutation] = 0
    elif mutation == "bool_index":
        rows[0]["index"] = False
    else:
        rows[0]["phase"] = 2
    with pytest.raises(ValueError):
        selection.select_inventory(rows)


def synthetic_root():
    keys = list("abcdefgh")
    actions = [{"synthetic": k} for k in keys]
    candidates = {"keys": keys, "actions": actions, "canonical_support": 8, "legal_support": 8}
    anchor = {**candidates, "keys": keys[:4], "actions": actions[:4]}
    proposal = {"candidates": candidates, "anchor_candidates": anchor, "breadth": True,
                "added_indices": {"ranked": [4, 5], "random": [6, 7]}, "timing": {"wall": 99}}
    worlds = [{"seed": str(i), "state_sha256": "0" * 64,
               "outcomes": [{"reason": "completed", "winner": None, "wdl": 0} for _ in keys]} for i in range(128)]
    worlds[0]["outcomes"][4].update(winner=0, wdl=1)
    return {"planner": "candidate", "root_identity": ["main", 0, 0], "privacy": {"own": 0, "opponent": 0},
            "proposal": proposal, "candidates": candidates, "worlds": worlds, "scores": [0] * 8,
            "schedule": {"world_seeds": [str(i) for i in range(128)]},
            **local.arena.select(worlds, 8), "snapshot": {"hidden": "never copied"}, "timing": {"wall": 1}}


def test_expected_four_prefix_reselects_ordered_tie_without_eight_way_winner():
    root = synthetic_root()
    prior = deepcopy(root)
    four, eight = local.expected_from_saved(root, False), local.expected_from_saved(root, True)
    assert four["record"]["choice"] == 0
    assert eight["record"]["choice"] == 4
    assert four["action"] == {"synthetic": "a"}
    assert eight["action"] == {"synthetic": "e"}
    assert four["record"]["means"] == [0] * 4
    assert "snapshot" not in four["record"] and "timing" not in four["record"]["proposal"]
    assert all(len(row["outcomes"]) == 4 for row in four["record"]["worlds"])
    assert root == prior


def test_all_equal_eight_way_tie_selects_first_offered():
    root = synthetic_root()
    for row in root["worlds"]:
        for outcome in row["outcomes"]:
            outcome.update(wdl=0, winner=None)
    assert local.arena.select(root["worlds"], 8) == {"choice": 0, "means": [0] * 8}


def test_choose_uses_only_observer_reconstruction_and_frozen_primitives(monkeypatch):
    state, context = fixture_state()
    payload = local.make_observer_payload(local.old.serialize_state_snapshot(state), context)
    actions = local.old.SplendorEnv().legal_actions(state)[:2]
    seeds = [f"synthetic-{i}" for i in range(128)]
    proposal = {"schedule": {"world_seeds": seeds},
                "candidates": {"keys": ["a", "b"], "actions": [local.old.serialize_action(a) for a in actions]},
                "timing": {"wall": 123}}
    monkeypatch.setattr(local, "_INITIALIZED", True)
    monkeypatch.setattr(local.arena, "proposal_for_root", lambda *args: deepcopy(proposal))
    class FakeRaw:
        def score_actions(self, state, legal):
            return [0] * len(legal)
    monkeypatch.setattr(local.old, "_RAW", FakeRaw())
    calls = []
    def trajectory(world, action):
        calls.append(action)
        return {"reason": "completed", "winner": None, "turns": 1,
                "trace_sha256": "0" * 64, "final_state_sha256": "1" * 64}
    monkeypatch.setattr(local.old, "trajectory", trajectory)
    def forbidden(*args, **kwargs):
        pytest.fail("Adapter invoked a production write or budget guard")
    monkeypatch.setattr(local.old, "save_new", forbidden)
    monkeypatch.setattr(local.arena.c, "budget_boundary", forbidden)
    boundaries = []
    result = local.choose(payload, False, lambda: boundaries.append(1))
    assert len(calls) == 256 and len(boundaries) == 129
    assert result["record"]["means"] == [0.0, 0.0]
    assert result["record"]["choice"] == 0
    assert len(result["record"]["worlds"]) == 128
    assert result["record"]["planner"] == "reference"
    assert "snapshot" not in result["record"]
    assert "timing" not in result["record"]["proposal"]


def test_choose_requires_explicit_model_initialization(monkeypatch):
    monkeypatch.setattr(local, "_INITIALIZED", False)
    with pytest.raises(ValueError, match="not initialized"):
        local.choose({}, False)
