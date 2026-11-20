"""Outcome-blind whole-pair sampling and authenticated JSON snapshot regression tests."""
import copy
from collections import Counter

import pytest

from tools.planner_state_transfer_128 import source as p


def metadata_games():
    result = []
    for pair in range(201):
        fit = p.c.FITS[pair//67]
        for leg in range(2):
            index = pair*2+leg
            result.append({"spec": {"index": index, "pair": pair, "fit": fit, "candidate_seat": leg,
                                     "environment_seed": 2170000000+pair, "search_stream": f"pair-{pair}"},
                           "trace": [{"actor": turn%2, "turn": turn, "state_sha256": f"state-{index}-{turn}",
                                      "root_sha256": f"root-{index}-{turn}"} for turn in range(70)]})
    return result


def test_sampling_is_order_and_outcome_blind_and_exactly_balanced():
    games = metadata_games()
    expected = p.allocations(games)
    changed = copy.deepcopy(list(reversed(games)))
    for game in changed:
        game.update(winner=1, reason="untrusted-outcome", final_score=100)
        for move in game["trace"]:
            move.update(chosen_probability=1., score=-100)
    assert p.allocations(changed) == expected
    assert p.validate_schedule(expected)
    assert {k: len(v) for k, v in expected.items()} == {"probe": 18, "main": 360}
    main = expected["main"]
    assert len({s["pair"] for s in main}) == 180
    assert Counter((s["fit"], s["phase"], s["source_arm"]) for s in main) == Counter({
        (fit, phase, arm): 20 for fit in p.c.FITS for phase in range(3) for arm in ("candidate", "reference")})


def test_missing_fixed_cell_holds_instead_of_substituting_another_pair():
    games = metadata_games()
    selected = p.allocations(games)["main"][0]
    victim = games[selected["game_index"]]
    victim["trace"] = [r for r in victim["trace"] if p.phase(r["turn"]) != selected["phase"]]
    with pytest.raises(ValueError, match="unavailable"):
        p.allocations(games)


@pytest.mark.parametrize("change", ["duplicate_game", "wrong_fit", "broken_seed_pair", "broken_stream_pair", "broken_seat_pair"])
def test_invalid_pair_population_is_rejected(change):
    games = metadata_games()
    selected = p.allocations(games)["main"][0]
    pair = selected["pair"]
    left, right = games[2*pair], games[2*pair+1]
    if change == "duplicate_game": right["spec"]["index"] = left["spec"]["index"]
    if change == "wrong_fit": left["spec"]["fit"] = -1
    if change == "broken_seed_pair": right["spec"]["environment_seed"] += 1
    if change == "broken_stream_pair": right["spec"]["search_stream"] = "changed"
    if change == "broken_seat_pair": right["spec"]["candidate_seat"] = left["spec"]["candidate_seat"]
    with pytest.raises(ValueError):
        p.allocations(games)


@pytest.mark.parametrize("change", ["extra_stage", "missing_root", "duplicate_game", "wrong_pair", "wrong_phase", "wrong_arm", "broken_seats"])
def test_schedule_validator_rejects_cell_changes(change):
    panel = p.allocations(metadata_games())
    s = panel["main"][0]
    if change == "extra_stage": panel["extra"] = []
    if change == "missing_root": panel["main"].pop()
    if change == "duplicate_game": s["game_index"] = panel["main"][1]["game_index"]
    if change == "wrong_pair": s["pair"] += 1
    if change == "wrong_phase": s["phase"] = (s["phase"]+1)%3
    if change == "wrong_arm": s["source_arm"] = "broad"
    if change == "broken_seats": s["seat"] = panel["main"][1]["seat"]
    with pytest.raises(ValueError):
        p.validate_schedule(panel)


def test_all_authenticated_historical_game_seeds_are_included_and_duplicates_rejected(monkeypatch):
    ref = {"path": "training.json", "sha256": "fixture"}
    refs = [{"path": f"fixture/main/games/game_{i:04d}.json", "sha256": str(i)} for i in range(1204)]
    data = {"manifest": p.c.LEARNING_BINDING, "source_evidence": refs}
    calls = []
    def verified(item):
        calls.append(item)
        if item == ref: return data
        assert item in refs
        return {"spec": {"environment_seed": 100+int(item["sha256"])//2}}
    monkeypatch.setattr(p.c, "verified", verified)
    result = p.training_seeds(ref)
    assert result["games"] == 1204 and result["environment_seeds"] == list(range(100, 702))
    assert len(calls) == 1205 and result["game_refs_sha256"] == p.c.old.digest(refs)
    data["source_evidence"] = refs[:-1]+[refs[0]]
    with pytest.raises(ValueError, match="population"):
        p.training_seeds(ref)


def snapshot_fixture(tmp_path):
    state = p.c.old.SplendorEnv(seed=2189000000).initial_state()
    snapshot = p.c.old.serialize_state_snapshot(state)
    source_spec = {"index": 0, "pair": 0, "fit": 161001, "candidate_seat": 0,
                   "environment_seed": 2189000000, "search_stream": "fixture-source"}
    root = {"manifest": p.arena.ORIGINAL_BINDING, "root_identity": ["main", 0, 0],
            "snapshot": snapshot, "planner": "candidate"}
    root_path = tmp_path/"root.json"
    p.c.old.save_new(root_path, root)
    root_ref = p.c.artifact(root_path)
    game = {"manifest": p.arena.ORIGINAL_BINDING, "spec": source_spec, "reason": "completed",
            "trace": [{"turn": 0, "actor": 0, "root_sha256": root_ref["sha256"],
                       "state_sha256": p.c.old.digest(snapshot)}]}
    game_path = tmp_path/"game.json"
    p.c.old.save_new(game_path, game)
    game_ref = p.c.artifact(game_path, index=0)
    audit = {"manifest": p.arena.ORIGINAL_BINDING, "passed": True, "game_artifact": game_ref,
             "roots": [{"turn": 0, "index": 0, "seat": 0, "assigned_fit": 161001,
                        "planner": "candidate", "sha256": root_ref["sha256"]}]}
    audit_path = tmp_path/"audit.json"
    p.c.old.save_new(audit_path, audit)
    return {"index": 0, "pair": 0, "game_index": 0, "fit": 161001, "seat": 0, "phase": 0,
            "source_arm": "candidate", "turn": 0, "search_stream": source_spec["search_stream"],
            "environment_seed": source_spec["environment_seed"], "snapshot_sha256": p.c.old.digest(snapshot),
            "source_game": game_ref, "source_root": root_ref, "source_audit": p.c.artifact(audit_path)}, snapshot


def test_saved_json_snapshot_uses_canonical_identity(tmp_path):
    spec, live_snapshot = snapshot_fixture(tmp_path)
    loaded = p.load_snapshot(spec, tmp_path)
    assert p.c.old.digest(loaded) == p.c.old.digest(live_snapshot)
    assert loaded["deck_counts"] != live_snapshot["deck_counts"]
    assert p.c.old.deserialize_state_snapshot(loaded).turn_index == 0


@pytest.mark.parametrize("key,value", [("environment_seed", -1), ("fit", 161002), ("seat", 1),
                                     ("source_arm", "broad"), ("source_arm", "reference"), ("phase", 1),
                                     ("turn", -1), ("snapshot_sha256", "changed")])
def test_changed_snapshot_spec_rejected(tmp_path, key, value):
    spec, _ = snapshot_fixture(tmp_path)
    spec[key] = value
    with pytest.raises(ValueError):
        p.load_snapshot(spec, tmp_path)


def test_saved_root_bytes_cannot_change_behind_a_valid_receipt(tmp_path):
    spec, _ = snapshot_fixture(tmp_path)
    path = p.c.verify_hash(spec["source_root"])
    path.write_text(path.read_text()+" ", encoding="utf-8")
    with pytest.raises(ValueError):
        p.load_snapshot(spec, tmp_path)
