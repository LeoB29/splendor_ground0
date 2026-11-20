"""New label budgets, arithmetic, durable blocks, and actual replay fixtures."""
import copy
import json
import random

import pytest

from . import labels as p


def fixture_spec(index=0):
    env = p.c.old.SplendorEnv(seed=2319800000)
    state = env.initial_state()
    for _ in range(2):
        reserve = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, reserve)
    take = next(a for a in env.legal_actions(state) if a.action_type.name == "TAKE_TOKENS")
    state = env.step(state, take)
    assert state.current_player == 1 and state.turn_index == 3
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 10 for color in player.bonuses}
    action = next(a for a in env.legal_actions(state) if a.action_type.name == "BUY_VISIBLE" and a.tier == 3)
    snap = p.c.old.serialize_state_snapshot(state)
    old_worlds = []
    for seed in range(2319900000, 2319900128):
        world = p.c.old.determinize_for_observer(state, 1, random.Random(seed))
        outcome = p.c.old.trajectory(world, action)
        assert outcome["reason"] == "completed" and outcome["turns"] == 1
        outcome["wdl"] = p.c.old.outcome_value(outcome["winner"], 1)
        old_worlds.append({"seed": seed, "state_sha256": p.c.old.digest(p.c.old.serialize_state_snapshot(world)), "outcomes": [outcome]})
    total = sum(int(w["outcomes"][0]["wdl"]) for w in old_worlds)
    source = {"snapshot": snap,
              "union": {"keys": [repr(p.c.old.canonical_successor_action_key(state, action))],
                        "actions": [p.c.old.serialize_action(action)]},
              "old_block": {"worlds": old_worlds, "world_seeds": [w["seed"] for w in old_worlds]},
              "old_sums_wdl": [total],
              **{name: {"path": f"fixture_{name}", "sha256": "0"*64} for name in ("old_selection", "source_root", "source_audit")}}
    return json.loads(json.dumps({"index": index, "row_index": 11950+index, "row_id": f"fixture:{index}:3", "fit": 161001,
            "seat": 1, "phase": 0, "turn": 3, "privacy": {"own": 1, "opponent": 1}, "support": 1,
            "old_sums_wdl": [total], "snapshot_sha256": p.c.old.digest(snap), "fixture_payload": source}))


def test_worlds_are_exact384_unique_and_separate_from_fresh_evaluation():
    seen = set()
    for index in range(720):
        for block in (1, 2, 3):
            seeds = p.world_seeds(index, block)
            assert seeds == list(range(2310000000+index*1024+block*128, 2310000000+index*1024+(block+1)*128))
            assert len(seeds) == 128 and seen.isdisjoint(seeds)
            seen.update(seeds)
            sample = p.ordinary_indices(index, block)
            assert len(set(sample)) == 8 and sample == sorted(sample) and all(0 <= n < 128 for n in sample)
    assert len(seen) == 720*384
    assert min(seen) > 2308000000


def test_integer_sums_and_unknowns_are_not_draws():
    outcomes = [{"wdl": v, "reason": "completed", "winner": None} for v in [-1., 0., 1.]]
    worlds = [{"outcomes": copy.deepcopy(outcomes)} for _ in range(128)]
    assert p.sums(worlds, 3) == ([-128, 0, 128], [])
    assert p.independent_sums(worlds, 3) == [-128, 0, 128]
    worlds[17]["outcomes"][1] = {"wdl": None, "reason": "repetition_cutoff", "winner": None}
    assert p.sums(worlds, 3, require_resolved=False) == ([-128, 0, 128], [[17, 1]])
    with pytest.raises(p.UnresolvedTarget): p.sums(worlds, 3)
    with pytest.raises(AssertionError): p.independent_sums(worlds, 3)
    worlds[17]["outcomes"][1]["reason"] = "max_turns"
    with pytest.raises((ValueError, AssertionError)): p.sums(worlds, 3, require_resolved=False)


@pytest.fixture(scope="module")
def short_spec():
    return fixture_spec()


def test_actual_collection_exact512_and_full_audit(tmp_path, short_spec):
    item = p.collect(short_spec, "fixture", tmp_path)
    root = p.validate_root(short_spec, "fixture", tmp_path, independent=True)
    assert root["sums_wdl512"] == [512]
    assert root["old_sums_wdl"] == [128] and root["worlds"] == 512
    assert len(root["blocks"]) == 3 and root["recorded_continuations"] == 384
    audit = p.c.verified(p.audit(short_spec, "fixture", tmp_path, full=True))
    assert audit["root"] == {k: item[k] for k in ("path", "sha256")}
    assert audit["recorded_continuations"] == audit["replayed_continuations"] == 384
    assert audit["full_inner_replay"] and audit["all512_sums_independently_checked"]
    assert p.audit(short_spec, "fixture", tmp_path, full=True)["sha256"] == p.c.artifact(p.audit_path(0, tmp_path))["sha256"]


def test_partial_failure_preserves_completed_block_and_recovery(tmp_path, short_spec, monkeypatch):
    original = p.c.old.trajectory
    calls = 0
    def interrupt(state, action):
        nonlocal calls
        calls += 1
        if calls == 129: raise RuntimeError("deliberate isolated interruption")
        return original(state, action)
    monkeypatch.setattr(p.c.old, "trajectory", interrupt)
    with pytest.raises(RuntimeError): p.collect(short_spec, "fixture", tmp_path)
    block = p.block_path(0, 1, tmp_path)
    before = p.c.old.file_hash(block)
    assert block.exists() and not p.root_path(0, tmp_path).exists()
    assert len(list(block.parent.glob("block_2_partial_*.json"))) == 1
    monkeypatch.setattr(p.c.old, "trajectory", original)
    item = p.collect(short_spec, "fixture", tmp_path)
    assert p.c.old.file_hash(block) == before
    assert p.c.verified(item)["sums_wdl512"] == [512]
    receipt = p.c.verified(p.audit(short_spec, "fixture", tmp_path))
    assert receipt["replayed_continuations"] == 24 and not receipt["full_inner_replay"]


def test_tampered_block_rejected_before_reuse(tmp_path, short_spec):
    p.collect(short_spec, "fixture", tmp_path)
    path = p.block_path(0, 2, tmp_path)
    text = path.read_text()
    path.write_text(text.replace('"sums_wdl": [\n    128', '"sums_wdl": [\n    127'))
    # Hash alteration alone must be rejected, even if the particular formatting
    # did not match; appending whitespace changes bytes without changing values.
    path.write_text(path.read_text()+"\n")
    with pytest.raises((ValueError, AssertionError)):
        p.validate_root(short_spec, "fixture", tmp_path, independent=True)


def test_crossfit_known_gain_ties_empty_support_and_excluded_block():
    # Block0 alone prefers action3; the three other blocks prefer4. Only fold3
    # uses block0 as its small teacher, and its independent outcome favors4.
    matrix = [[0, 0, 0, 100, 0]]+[[0, 0, 0, 0, 100] for _ in range(3)]
    result = p.crossfit_row(matrix)
    assert result["choice_disagreements"] == 1
    assert result["sum_heldout_wdl_differences"] == 100
    assert result["mean_score_gain"] == 100/1024
    assert p.crossfit_row([[0, 0, 0]]*4)["mean_score_gain"] == 0
    assert all(f["choice128"] == f["choice384"] == 3 for f in p.crossfit_row([[0]*5]*4)["folds"])
    changed = copy.deepcopy(matrix)
    changed[2] = [0, 0, 0, -128, 128]
    before, after = result["folds"][2], p.crossfit_row(changed)["folds"][2]
    assert (before["choice128"], before["choice384"]) == (after["choice128"], after["choice384"])


def test_unknown_branch_is_replayed_and_holds_without_target(tmp_path, short_spec, monkeypatch):
    unknown = {"reason": "repetition_cutoff", "winner": None, "turns": 8,
               "trace_sha256": "1"*64, "final_state_sha256": "2"*64}
    monkeypatch.setattr(p.c.old, "trajectory", lambda state, action: dict(unknown))
    calls = []
    def replay(world, action, actor):
        calls.append(actor)
        return {**unknown, "wdl": None}
    monkeypatch.setattr(p.continuation_audit, "replay_branch", replay)
    with pytest.raises(p.UnresolvedTarget):
        p.collect(short_spec, "fixture", tmp_path)
    assert len(calls) == 128
    assert p.block_path(0, 1, tmp_path).exists() and not p.block_path(0, 2, tmp_path).exists()
    assert not p.root_path(0, tmp_path).exists()
    record = p.c.old.read_json(tmp_path/"labels/unresolved/state_0000_block_1.json")
    assert record["all_unknown_branches_replayed"] and record["imputed"] is False


def test_independent_sample_detects_forged_outcome_even_with_updated_hashes(tmp_path, short_spec):
    p.collect(short_spec, "fixture", tmp_path)
    path = p.block_path(0, 1, tmp_path)
    block = p.c.old.read_json(path)
    wi = p.ordinary_indices(0, 1)[0]
    block["worlds"][wi]["outcomes"][0]["final_state_sha256"] = "a"*64
    path.write_text(json.dumps(block))
    rootpath = p.root_path(0, tmp_path)
    root = p.c.old.read_json(rootpath)
    receipt_path = p.block_receipt_path(0, 1, tmp_path)
    receipt = p.c.old.read_json(receipt_path)
    receipt["artifact"] = p.c.artifact(path)
    receipt_path.write_text(json.dumps(receipt))
    root["blocks"][0] = p.c.artifact(path, block=1, completion=p.c.artifact(receipt_path))
    rootpath.write_text(json.dumps(root))
    # Structural arithmetic passes but independent trajectory evidence disagrees.
    p.validate_root(short_spec, "fixture", tmp_path, independent=True)
    with pytest.raises(AssertionError):
        p.audit(short_spec, "fixture", tmp_path)


def test_inherited_world_seed_overlap_rejected_before_collection(tmp_path, short_spec):
    spec = copy.deepcopy(short_spec)
    spec["fixture_payload"]["old_block"]["world_seeds"][0] = p.world_seeds(0, 1)[0]
    with pytest.raises((ValueError, AssertionError)):
        p.collect(spec, "fixture", tmp_path)
    assert not p.block_path(0, 1, tmp_path).exists()


@pytest.mark.parametrize("damage", ["coherent_outcomes", "unsealed"])
def test_partial_root_recovery_requires_original_block_hash(tmp_path, short_spec, monkeypatch, damage):
    original, calls = p.c.old.trajectory, 0
    def interrupt(state, action):
        nonlocal calls
        calls += 1
        if calls == 129: raise RuntimeError("isolated interruption before root seal")
        return original(state, action)
    monkeypatch.setattr(p.c.old, "trajectory", interrupt)
    with pytest.raises(RuntimeError): p.collect(short_spec, "fixture", tmp_path)
    monkeypatch.setattr(p.c.old, "trajectory", original)
    block_path, receipt_path = p.block_path(0, 1, tmp_path), p.block_receipt_path(0, 1, tmp_path)
    assert block_path.exists() and receipt_path.exists() and not p.root_path(0, tmp_path).exists()
    if damage == "coherent_outcomes":
        record = p.c.old.read_json(block_path)
        for world in record["worlds"]:
            world["outcomes"][0].update(winner=0, wdl=-1.)
        record["sums_wdl"] = [-128]
        block_path.write_text(json.dumps(record))
    else:
        receipt_path.unlink()
    original_bytes = block_path.read_bytes()
    with pytest.raises((ValueError, AssertionError, p.c.BudgetExceeded)):
        p.collect(short_spec, "fixture", tmp_path)
    assert block_path.read_bytes() == original_bytes and not p.root_path(0, tmp_path).exists()
