"""Synthetic sampling and source-authentication regressions; no policy inference."""
import copy
import hashlib
import json
from collections import Counter

import pytest
import torch

from . import common as c, source as s


def population():
    games = []
    for pair in range(201):
        fit = c.FITS[pair//67]
        for seat in (0, 1):
            index = 2*pair+seat
            games.append({"spec": {"index": index, "pair": pair, "fit": fit, "candidate_seat": seat,
                                    "environment_seed": 900000+pair, "search_stream": f"fixture-{pair}"},
                          "trace": [{"actor": seat, "turn": turn, "state_sha256": f"state-{index}-{turn}",
                                     "root_sha256": f"root-{index}-{turn}"} for turn in range(seat, 60, 2)]})
    return games


def split_population(games):
    return s.original_splits([g["spec"] for g in games])


def independent_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def test_all_validation_pairs_and_exact_identity_train_selection():
    games = population()
    split = split_population(games)
    schedule = s.allocations(games, split)
    assert s.validate_schedule(schedule)
    assert len(schedule["main"]) == 336 and len(schedule["probe"]) == 18
    for fit in c.FITS:
        offered = {g["spec"]["pair"] for g in games if g["spec"]["fit"] == fit}
        valid = {pair for pair in offered if split[pair] == "validation"}
        train = sorted(offered-valid, key=lambda pair: (independent_digest([s.TRAIN_TAG, fit, pair]), pair))[:14]
        assert {r["pair"] for r in schedule["main"] if r["fit"] == fit and r["split"] == "validation"} == valid
        assert {r["pair"] for r in schedule["main"] if r["fit"] == fit and r["split"] == "train"} == set(train)
    assert Counter(r["phase"] for r in schedule["main"]) == {0: 112, 1: 112, 2: 112}
    assert {r["pair"] for r in schedule["probe"]}.isdisjoint(r["pair"] for r in schedule["main"])
    assert {r["split"] for r in schedule["probe"]} == {"train"}
    for fit in c.FITS:
        train = Counter((r["seat"], r["phase"]) for r in schedule["main"] if r["fit"] == fit and r["split"] == "train")
        valid = Counter((r["seat"], r["phase"]) for r in schedule["main"] if r["fit"] == fit and r["split"] == "validation")
        assert train == valid


def test_sampling_is_independent_of_input_order_outcomes_and_policy_outputs():
    games = population(); splits = split_population(games)
    expected = s.allocations(games, splits)
    changed = copy.deepcopy(games)
    changed.reverse()
    for game in changed:
        game.update(winner=1, value=999, model_choice="adversarial", score=-999)
        game["trace"].reverse()
        for row in game["trace"]: row.update(wdl=1, choice=999, gap=-999)
    assert s.allocations(changed, splits) == expected


def test_root_selected_by_prescribed_identity_hash_within_each_phase():
    games = population(); schedule = s.allocations(games, split_population(games))
    for stage, roots in schedule.items():
        for selected in roots:
            available = [r for r in games[selected["game_index"]]["trace"] if min(r["turn"]//20, 2) == selected["phase"]]
            expected = min(available, key=lambda r: (independent_digest([s.ROOT_TAG, stage, selected["game_index"], selected["phase"], r["turn"]]), r["turn"]))
            assert selected["turn"] == expected["turn"]


def test_missing_preassigned_phase_holds_without_substituting_a_pair():
    games = population(); split = split_population(games)
    selected = s.allocations(games, split)["main"][0]
    game = games[selected["game_index"]]
    game["trace"] = [r for r in game["trace"] if min(r["turn"]//20, 2) != selected["phase"]]
    with pytest.raises(ValueError, match="unavailable"):
        s.allocations(games, split)


@pytest.mark.parametrize("change", ["split", "missing_game", "duplicate_game", "seat", "pair_seed"])
def test_bad_source_population_is_rejected(change):
    games = population(); split = split_population(games)
    if change == "split": split[0] = "train" if split[0] == "validation" else "validation"
    if change == "missing_game": games.pop()
    if change == "duplicate_game": games[-1] = copy.deepcopy(games[0])
    if change == "seat": games[0]["spec"]["candidate_seat"] = 1
    if change == "pair_seed": games[1]["spec"]["environment_seed"] += 1
    with pytest.raises(ValueError): s.allocations(games, split)


@pytest.mark.parametrize("change", ["root_count", "index", "duplicate_row", "seat", "pair", "split", "rank", "omission", "phase", "stream", "probe_validation"])
def test_schedule_rejects_broken_pairs_matching_or_probe_contract(change):
    games = population(); schedule = s.allocations(games, split_population(games))
    row = schedule["main"][0]
    if change == "root_count": schedule["main"].pop()
    if change == "index": row["index"] = 9
    if change == "duplicate_row": row["source_rowid"] = schedule["main"][1]["source_rowid"]
    if change == "seat": row["seat"] = 1-row["seat"]
    if change == "pair": row["pair"] += 1
    if change == "split": row["split"] = "validation"
    if change == "rank": row["pair_rank"] = 15
    if change == "omission": row["omitted_phase"] = (row["omitted_phase"]+1)%3
    if change == "phase": row["phase"] = (row["phase"]+1)%3
    if change == "stream": row["search_stream"] = "changed"
    if change == "probe_validation": schedule["probe"][0]["split"] = "validation"
    with pytest.raises(ValueError): s.validate_schedule(schedule)


def provenance_fixture(tmp_path):
    """A legal initial position plus synthetic closed provenance, without rollouts."""
    env = c.old.SplendorEnv(seed=912345)
    state = env.initial_state()
    seat = state.current_player
    game_index = seat
    legal = env.legal_actions(state)
    groups = c.old.actions_by_canonical_successor_key(state, legal)
    keys = [repr(key) for key in groups]
    representatives = list(groups.values())[:8]
    offered = {"keys": keys[:8], "actions": [c.old.serialize_action(a) for a in representatives]}
    snapshot = c.old.serialize_state_snapshot(state)
    privacy = c.baseline.privacy(state)
    root = {"manifest": c.SOURCE_BINDING, "planner": "candidate", "root_identity": ["main", game_index, 0],
            "snapshot": snapshot, "privacy": privacy, "candidates": offered, "choice": 0,
            "proposal": {"legal_actions": [c.old.serialize_action(a) for a in legal], "group_keys": keys}}
    def save(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value), encoding="utf-8")
        return c.artifact(path)
    root_ref = save("root.json", root)
    gs = {"index": game_index, "pair": 0, "fit": 161001, "candidate_seat": seat,
          "environment_seed": 912345, "search_stream": "synthetic-source"}
    trace = {"turn": 0, "actor": seat, "root_sha256": root_ref["sha256"],
             "state_sha256": c.old.digest(snapshot), "action": offered["actions"][0]}
    game = {"spec": gs, "manifest": c.SOURCE_BINDING, "reason": "completed", "trace": [trace]}
    game_ref = save("game.json", game)
    checked = {"turn": 0, "sha256": root_ref["sha256"], "index": game_index, "seat": seat,
               "assigned_fit": 161001, "planner": "candidate", "canonical_support": len(keys)}
    audit = {"manifest": c.SOURCE_BINDING, "passed": True, "game_artifact": game_ref, "roots": [checked]}
    audit_ref = save("audit.json", audit)
    membership = [keys.index(repr(c.old.canonical_successor_action_key(state, a))) for a in legal]
    row = {"id": f"breadth128:{game_index}:0", "pair": "breadth128:0", "split": "train", "seat": seat,
           "phase": 0, "privacy": privacy, "teacher": 0, "narrow_support": 4, "group_count": len(keys),
           "groups": torch.tensor(membership), "features": torch.zeros((len(legal), 1)),
           "members": [torch.tensor([i for i, group in enumerate(membership) if group == j]) for j in range(8)]}
    export_path = tmp_path/"export.pt"; torch.save([row], export_path)
    export_ref = c.artifact(export_path, rows=1)
    ea = {"manifest": c.LEARNING_BINDING, "passed": True, "actual_game_replayed": True,
          "independent_targets": True, "every_row_hidden_input_invariance": True, "export": export_ref,
          "source": {"game": game_ref, "audit": audit_ref, "spec": gs, "parent_binding": c.SOURCE_BINDING},
          "source_roots": [{**root_ref, "turn": 0}]}
    ea_ref = save("export_audit.json", ea)
    spec = {"index": 0, "pair": 0, "game_index": game_index, "fit": 161001, "seat": seat, "phase": 0,
            "split": "train", "turn": 0, "source_arm": "candidate", "snapshot_sha256": c.old.digest(snapshot),
            "canonical_support": len(keys), "privacy": privacy, "source_rowid": row["id"],
            "environment_seed": gs["environment_seed"], "search_stream": gs["search_stream"],
            "source_root": root_ref, "source_game": game_ref, "source_audit": audit_ref,
            "source_export": export_ref, "source_export_audit": ea_ref}
    return spec, root, row


def test_authenticated_root_links_actual_snapshot_export_and_legal_membership(tmp_path):
    spec, root, _ = provenance_fixture(tmp_path)
    assert c.old.digest(s.load_root(spec)) == c.old.digest(root)


@pytest.mark.parametrize("change", ["snapshot", "split", "rowid", "privacy", "support", "fit", "seat", "turn", "stream", "seed", "wrong_planner"])
def test_load_rejects_changed_source_identity(tmp_path, change):
    spec, _, _ = provenance_fixture(tmp_path)
    if change == "snapshot": spec["snapshot_sha256"] = "changed"
    if change == "split": spec["split"] = "validation"
    if change == "rowid": spec["source_rowid"] = "breadth128:0:99"
    if change == "privacy": spec["privacy"] = {"own": 1, "opponent": 1}
    if change == "support": spec["canonical_support"] += 1
    if change == "fit": spec["fit"] = 161002
    if change == "seat": spec["seat"] = 1-spec["seat"]
    if change == "turn": spec["turn"] = 2
    if change == "stream": spec["search_stream"] = "changed"
    if change == "seed": spec["environment_seed"] += 1
    if change == "wrong_planner": spec["source_arm"] = "reference"
    with pytest.raises(ValueError): s.load_root(spec)


@pytest.mark.parametrize("which", ["source_game", "source_root", "source_audit", "source_export", "source_export_audit"])
def test_load_authenticates_every_selected_input_hash(tmp_path, which):
    spec, _, _ = provenance_fixture(tmp_path)
    path = c.verify_hash(spec[which])
    with path.open("ab") as handle: handle.write(b" ")
    with pytest.raises(ValueError, match="Changed artifact"):
        s.load_root(spec)


@pytest.mark.parametrize("field", ["groups", "members", "group_count", "features", "teacher", "privacy"])
def test_tensor_membership_check_rejects_structural_corruption(tmp_path, field):
    spec, root, row = provenance_fixture(tmp_path)
    if field == "groups": row["groups"] = row["groups"]+1
    if field == "members": row["members"][0] = torch.tensor([999])
    if field == "group_count": row["group_count"] += 1
    if field == "features": row["features"] = row["features"][:-1]
    if field == "teacher": row["teacher"] = 1
    if field == "privacy": row["privacy"] = {"own": 1, "opponent": 1}
    with pytest.raises(ValueError): s._verify_export(spec, root, row)


def test_load_obeys_outer_resource_budget_before_reading(tmp_path, monkeypatch):
    spec, _, _ = provenance_fixture(tmp_path)
    def stop(base): raise RuntimeError("synthetic budget hold")
    monkeypatch.setattr(c, "budget_boundary", stop)
    with pytest.raises(RuntimeError, match="synthetic budget"):
        s.load_root(spec, tmp_path)
