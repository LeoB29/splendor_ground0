"""Whole-source replay, identity selection, source shortage and fixture safety."""
import copy
from collections import Counter
import time

import pytest

from tools.structured_proposer_128 import source as p
from tools.structured_proposer_128.test_engine import fixture_registry,private_state


def fixture_learning_seal(base,*,passed=True,underfit=None,underfit_arm="relations"):
    registry = fixture_registry()
    gate = {"manifest":"fixture","passed":passed,"thresholds":{"added":.95,"old":.90},
            "fits":{f"{arm}_{f}":{"passed":True,"added_capture":.97,"old_capture":.92}
                    for arm in p.c.LEARNING_ARMS for f in p.c.FITS}}
    if underfit is not None: gate["fits"][f"{underfit_arm}_{underfit}"]["added_capture"] = .94
    p.old.save_new(base/"models_lock.json",registry)
    p.old.save_new(base/"acquisition_gate.json",gate)
    p.old.save_new(base/"learning_complete.json",{"manifest":"fixture",
        "models":p.c.artifact(base/"models_lock.json"),"acquisition_gate":p.c.artifact(base/"acquisition_gate.json")})
    return registry,gate


@pytest.fixture(scope="module",autouse=True)
def models(tmp_path_factory):
    base = tmp_path_factory.mktemp("structured_sources")
    p.initialize_models(base,fixture_registry(base))


def test_source_schedule_balanced_prospective_and_disjoint():
    seeds,streams = set(),set()
    for stage,n in (("main",120),("probe",3),("test",1)):
        specs = p.schedule(stage)
        assert Counter((s["fit"],s["seat"],s["phase"]) for s in specs) == {cell:n for cell in p.CELLS}
        assert all(s == p.independent_spec(stage,s["index"]) for s in specs)
        for s in specs:
            assert s["environment_seed"] not in seeds and s["search_stream"] not in streams
            seeds.add(s["environment_seed"]);streams.add(s["search_stream"])
    assert len(seeds) == len(streams) == 2232


def test_source_real_private_game_all_move_audit_and_replay(tmp_path):
    state = private_state()
    spec = next(s for s in p.schedule("test") if s["seat"] == state.current_player and s["phase"] == 0)
    item = p.collect(spec,"fixture",tmp_path,p.old.serialize_state_snapshot(state))
    checked = p.audit(spec,item,"fixture",tmp_path)
    game,receipt = p.c.verified(item),p.c.verified(checked)
    assert receipt["passed"] and receipt["every_move_hidden_input_invariance"]
    assert receipt["coverage"]["moves"] == len(game["trace"]) >= 1
    first = game["trace"][0]
    selected = p._selected_spec(0,"test",(spec,game,item,checked),first)
    assert p.old.digest(p.replay_selected_state(selected,tmp_path)) == p.old.digest(first["snapshot"])
    original = p.old.file_hash(p.game_path("test",spec["index"],tmp_path))
    with pytest.raises(ValueError,match="One-shot"):
        p.collect(spec,"fixture",tmp_path)
    assert p.old.file_hash(p.game_path("test",spec["index"],tmp_path)) == original
    changed = copy.deepcopy(selected);changed["turn"] += 1
    with pytest.raises(ValueError):
        p.replay_selected_state(changed,tmp_path)


@pytest.mark.parametrize("active_hours", [0, 40])
def test_actual_source_to_sparse_assessment_and_independent_audit(tmp_path,active_hours):
    from tools.structured_proposer_128 import engine,verify
    p.old.save_new(tmp_path/"launch.json",{"manifest":"fixture","epoch_seconds":time.time()-active_hours*3600})
    p.old.save_new(tmp_path/"launch_seal.json",{"manifest":"fixture","receipt":p.c.artifact(tmp_path/"launch.json")})
    registry = fixture_registry()
    p.old.save_new(tmp_path/"models_lock.json",registry)
    engine.initialize_models(tmp_path)
    state = private_state()
    source_spec = next(s for s in p.schedule("test") if s["seat"] == state.current_player and s["phase"] == 0)
    item = p.collect(source_spec,"fixture",tmp_path,p.old.serialize_state_snapshot(state))
    checked = p.audit(source_spec,item,"fixture",tmp_path)
    game = p.c.verified(item)
    spec = p._selected_spec(0,"test",(source_spec,game,item,checked),game["trace"][0])
    assert "fixture_snapshot" not in spec
    p.old.save_new(tmp_path/"test/audit_sample.json",verify.sample_plan([spec],"test","fixture"))
    root = engine.collect("test",spec,"fixture",tmp_path)
    receipt = p.c.verified(verify.audit("test",spec,root,"fixture",tmp_path))
    record = p.c.verified(root)
    assert receipt["passed"] and receipt["full_inner_replay"]
    assert receipt["recorded_continuations"] == root["continuations"] == receipt["replayed_continuations"]
    assert receipt["row"] == engine.statistics_row(record)
    assert p.old.digest(p.old.serialize_state_snapshot(p.load_state(spec,"fixture",tmp_path))) == spec["snapshot_sha256"]
    with pytest.raises(ValueError,match="binding"):
        p.load_state(spec,"wrong",tmp_path)


def test_source_audit_catches_rehashed_model_action_corruption(tmp_path):
    state = private_state()
    spec = p.schedule("test")[0]
    item = p.collect(spec,"fixture",tmp_path,p.old.serialize_state_snapshot(state))
    path = p.game_path("test",0,tmp_path)
    game = p.c.verified(item)
    game["trace"][0]["evidence"]["choice"] = (game["trace"][0]["evidence"]["choice"]+1)%4
    game["trace_sha256"] = p.old.digest(game["trace"])
    p.old.atomic_write(path,game)
    with pytest.raises((AssertionError,ValueError)):
        p.audit(spec,p.c.artifact(path,index=0,stage="test"),"fixture",tmp_path)
    assert not p.audit_path("test",0,tmp_path).exists()


def fake_inventory(stages=("main","probe")):
    by_stage,items,audits = {},{},{}
    for stage in stages:
        entries,source_items,audit_items = [],[],[]
        for spec in p.schedule(stage):
            index = spec["index"]
            item = {"path":f"fixture/{stage}/game_{index}.json","sha256":f"game{index}","index":index,"stage":stage}
            audit = {"path":f"fixture/{stage}/audit_{index}.json","sha256":f"audit{index}","index":index,"stage":stage}
            turns = [20*spec["phase"]+spec["seat"],20*spec["phase"]+spec["seat"]+2]
            game = {"winner":index%2,"trace":[{"turn":turn,"actor":spec["seat"],"phase":spec["phase"],
                    "state_sha256":f"state{index}:{turn}","evidence":{"privacy":{"own":index%2,"opponent":0},
                    "candidates":{"canonical_support":1+index%8}}} for turn in turns]}
            entries.append((spec,game,item,audit));source_items.append(item);audit_items.append(audit)
        by_stage[stage],items[stage],audits[stage] = entries,source_items,audit_items
    return by_stage,items,audits


def test_selection_full_pool_identity_only_and_independently_checked(tmp_path,monkeypatch):
    records,items,audits = fake_inventory()
    monkeypatch.setattr(p,"_inventory",lambda stage,*args:records[stage])
    fixture_learning_seal(tmp_path/"first")
    first = p.c.verified(p.prepare_selection(items,audits,"fixture",tmp_path/"first"))
    for stage,n in (("main",40),("probe",1)):
        assert Counter((s["fit"],s["seat"],s["phase"]) for s in first[stage]) == {cell:n for cell in p.CELLS}
        assert len({s["game_index"] for s in first[stage]}) == 18*n
        assert [s["index"] for s in first[stage]] == list(range(18*n))
        for _,game,_,_ in records[stage]:
            game["winner"] = 1-game["winner"]
    fixture_learning_seal(tmp_path/"second")
    second = p.c.verified(p.prepare_selection(items,audits,"fixture",tmp_path/"second"))
    assert first == second
    changed = copy.deepcopy(first);changed["main"][0]["turn"] += 2
    with pytest.raises(ValueError,match="selection mismatch"):
        p.validate_selection(changed,"fixture",tmp_path/"first")


def test_short_source_cell_holds_without_extension(tmp_path,monkeypatch):
    fixture_learning_seal(tmp_path)
    records,items,audits = fake_inventory()
    cell = p.CELLS[-1]
    for spec,game,_,_ in records["main"]:
        if (spec["fit"],spec["seat"],spec["phase"]) == cell:
            game["trace"] = []
    monkeypatch.setattr(p,"_inventory",lambda stage,*args:records[stage])
    with pytest.raises(p.c.BudgetExceeded,match="no extension"):
        p.prepare_selection(items,audits,"fixture",tmp_path)
    assert not (tmp_path/"selection.json").exists()


def test_source_fixture_cannot_target_production_or_override_natural_source(tmp_path):
    with pytest.raises(ValueError,match="production"):
        p.collect(p.schedule("test")[0],"fixture",p.OUTPUT)
    with pytest.raises(ValueError,match="Synthetic"):
        p.collect(p.schedule("main")[0],"fixture",tmp_path,p.old.serialize_state_snapshot(private_state()))
    assert not p.game_path("main",0,tmp_path).exists()


def test_source_loads_only_anchors_and_rejects_unsealed_registry(tmp_path):
    assert set(p._MODELS) == set(p._MODEL_ARTIFACTS) == {f"anchor_{fit}" for fit in p.c.FITS}
    sealed = dict(p._MODEL_ARTIFACTS)
    damaged = fixture_registry()
    damaged["models"].pop("relations_161003")
    with pytest.raises(ValueError,match="Incomplete"):
        p.initialize_models(tmp_path,damaged)
    assert p._MODEL_ARTIFACTS == sealed


def test_fresh_source_namespaces_exclude_inherited_coverage():
    from tools.large_corpus_optimization_128 import source as previous
    for stage in ("main","probe","test"):
        fresh = p.schedule(stage)
        inherited = previous.schedule(stage)
        assert {s["environment_seed"] for s in fresh}.isdisjoint(s["environment_seed"] for s in inherited)
        assert {s["search_stream"] for s in fresh}.isdisjoint(s["search_stream"] for s in inherited)
        assert p.CONFIG["source_assignment_tag"] != previous.CONFIG["source_assignment_tag"]
        assert p.CONFIG["selection_tag"] != previous.CONFIG["selection_tag"]




def test_schedule_cache_does_not_expose_mutable_identity_and_collection_uses_cached_rows(tmp_path,monkeypatch):
    original = p.independent_spec("test",0)
    altered = p.schedule("test");altered[0]["environment_seed"] = -1
    assert p.schedule("test")[0] == original
    monkeypatch.setattr(p,"schedule",lambda *a:pytest.fail("Per-game validation rebuilt full source pool"))
    item = p.collect(original,"fixture",tmp_path,p.old.serialize_state_snapshot(private_state()))
    assert p.c.verified(item)["spec"] == original


@pytest.mark.parametrize("damage",["missing","failed","one_underfit","gate_hash","models_hash","redirect","binding"])
def test_no_fresh_source_before_authenticated_acquisition_and_model_seal(tmp_path,damage):
    if damage != "missing":
        fixture_learning_seal(tmp_path,passed=damage != "failed",underfit=p.c.FITS[1] if damage == "one_underfit" else None)
        if damage in ("gate_hash","models_hash"):
            path = tmp_path/("acquisition_gate.json" if damage == "gate_hash" else "models_lock.json")
            path.write_text(path.read_text()+"\n")
        if damage in ("redirect","binding"):
            complete = p.old.read_json(tmp_path/"learning_complete.json")
            if damage == "binding": complete["manifest"] = "wrong"
            else:
                p.old.save_new(tmp_path/"other_gate.json",p.old.read_json(tmp_path/"acquisition_gate.json"))
                complete["acquisition_gate"] = p.c.artifact(tmp_path/"other_gate.json")
            p.old.atomic_write(tmp_path/"learning_complete.json",complete)
    spec = p.schedule("main")[0]
    with pytest.raises((ValueError,FileNotFoundError)):
        p.collect(spec,"fixture",tmp_path)
    assert not p.game_path("main",0,tmp_path).exists()


def test_valid_gate_is_required_even_for_selection_only(tmp_path,monkeypatch):
    fixture_learning_seal(tmp_path)
    assert set(p.evaluation_ready("fixture",tmp_path)["models"]) == {f"{arm}_{fit}" for arm in p.c.MODEL_ARMS for fit in p.c.FITS}
    with pytest.raises((KeyError,ValueError)):
        p.schedule("train")


@pytest.mark.parametrize("arm",["raw","relations"])
@pytest.mark.parametrize("pool",["added","old"])
def test_each_arm_and_each_full_training_pool_must_pass_acquisition(tmp_path,arm,pool):
    fixture_learning_seal(tmp_path)
    gate = p.old.read_json(tmp_path/"acquisition_gate.json")
    gate["fits"][f"{arm}_{p.c.FITS[2]}"][f"{pool}_capture"] = {"added":.949,"old":.899}[pool]
    p.old.atomic_write(tmp_path/"acquisition_gate.json",gate)
    complete = p.old.read_json(tmp_path/"learning_complete.json")
    complete["acquisition_gate"] = p.c.artifact(tmp_path/"acquisition_gate.json")
    p.old.atomic_write(tmp_path/"learning_complete.json",complete)
    with pytest.raises(ValueError,match="failed"):
        p.evaluation_ready("fixture",tmp_path)
