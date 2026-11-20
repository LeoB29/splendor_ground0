"""Whole-source replay, identity selection, source shortage and fixture safety."""
import copy
from collections import Counter

import pytest

from tools.equal_budget_breadth_128 import source as p
from tools.equal_budget_breadth_128.test_engine import fixture_registry,private_state


@pytest.fixture(scope="module",autouse=True)
def models(tmp_path_factory):
    p.initialize_models(tmp_path_factory.mktemp("equal_budget_breadth_sources"),fixture_registry())


def test_source_schedule_balanced_prospective_and_disjoint():
    seeds,streams = set(),set()
    for stage,n in (("main",60),("probe",3),("test",1)):
        specs = p.schedule(stage)
        assert Counter((s["fit"],s["seat"],s["phase"]) for s in specs) == {cell:n for cell in p.CELLS}
        assert all(s == p.independent_spec(stage,s["index"]) for s in specs)
        for s in specs:
            assert s["environment_seed"] not in seeds and s["search_stream"] not in streams
            seeds.add(s["environment_seed"]);streams.add(s["search_stream"])
    assert len(seeds) == len(streams) == 1152
    from tools.proposal_breadth_128 import source as predecessor
    prior = [s for stage in ("main","probe","test") for s in predecessor.schedule(stage)]
    assert not seeds.intersection(s["environment_seed"] for s in prior)
    assert not streams.intersection(s["search_stream"] for s in prior)


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


def test_actual_source_to_sparse_assessment_and_independent_audit(tmp_path):
    from tools.equal_budget_breadth_128 import engine,verify
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


def fake_inventory():
    by_stage,items,audits = {},{},{}
    for stage in ("main","probe"):
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
    first = p.c.verified(p.prepare_selection(items,audits,"fixture",tmp_path/"first"))
    for stage,n in (("main",20),("probe",1)):
        assert Counter((s["fit"],s["seat"],s["phase"]) for s in first[stage]) == {cell:n for cell in p.CELLS}
        assert len({s["game_index"] for s in first[stage]}) == 18*n
        assert [s["index"] for s in first[stage]] == list(range(18*n))
        for _,game,_,_ in records[stage]:
            game["winner"] = 1-game["winner"]
    second = p.c.verified(p.prepare_selection(items,audits,"fixture",tmp_path/"second"))
    assert first == second
    changed = copy.deepcopy(first);changed["main"][0]["turn"] += 2
    with pytest.raises(ValueError,match="selection mismatch"):
        p.validate_selection(changed,"fixture",tmp_path)


def test_short_source_cell_holds_without_extension(tmp_path,monkeypatch):
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


def test_source_registry_rejects_extra_models_and_wrong_fit(tmp_path):
    for kind in ("role","fit"):
        registry = fixture_registry()
        if kind == "role":
            registry["models"]["ranking_161001"] = copy.deepcopy(registry["models"]["anchor_161001"])
        else:
            registry["models"]["anchor_161001"]["source_fit"] = 161002
        with pytest.raises(ValueError):
            p.initialize_models(tmp_path,registry)
