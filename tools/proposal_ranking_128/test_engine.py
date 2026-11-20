"""Focused tests for source-private proposer roles and sparse independent worlds."""
import copy
from pathlib import Path

import pytest

from tools.proposal_ranking_128 import engine as p
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit


def fixture_registry():
    old = p.old.read_json(p.c.SOURCE/"models_lock.json")["models"]
    return {"manifest":"fixture", "models":{
        f"{arm}_{fit}":dict(old[f"{source}_{fit}"])
        for arm,source in (("anchor","anchor"),("ranking","treatment"),("ce","replay")) for fit in p.c.FITS}}


@pytest.fixture(scope="module",autouse=True)
def models(tmp_path_factory):
    p.initialize_models(tmp_path_factory.mktemp("ranking_engine"),fixture_registry())


def private_state():
    env = p.old.SplendorEnv(seed=2099800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state,action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color:5 for color in player.bonuses}
    return state


def fixture_spec(index=0):
    state = private_state()
    snapshot = p.old.serialize_state_snapshot(state)
    return {"index":index,"pair":index,"game_index":index,"fit":161001,"seat":state.current_player,
            "phase":0,"turn":state.turn_index,"source_stage":"test","search_stream":"ranking-isolated-source",
            "snapshot_sha256":p.old.digest(snapshot),"fixture_snapshot":snapshot,
            "privacy":p.baseline.privacy(state),
            "canonical_support":len(p.old.actions_by_canonical_successor_key(state,p.old.SplendorEnv().legal_actions(state)))}


def synthetic_union(size,fourths=None):
    keys = list("abcdef")[:size]
    reps = {k:{"action":k} for k in keys}
    reference = {"keys":keys[:4],"actions":[reps[k] for k in keys[:4]]}
    fourths = fourths or (keys[min(size-1,3)],)*3
    proposals = {}
    for i,arm in enumerate(p.ARMS):
        offered = keys[:3]+([fourths[i]] if size >= 4 else [])
        proposals[arm] = {"candidates":{"keys":offered,"actions":[reps[k] for k in offered]}}
    return reference,proposals,keys,reps


def test_all_world_and_source_namespaces_are_disjoint():
    from tools.proposal_ranking_128 import source
    from tools.proposal_diagnostic_128 import engine as prior
    seen = set()
    for stage,count in (("main",360),("probe",18),("test",18)):
        for index in range(count):
            for block in ("selection","assessment"):
                seeds = p.world_seeds(stage,index,block)
                assert len(seeds) == len(set(seeds)) == 128 and not seen.intersection(seeds)
                seen.update(seeds)
    source_seeds = {s["environment_seed"] for stage in ("main","probe","test") for s in source.schedule(stage)}
    assert len(source_seeds) == 1152 and not seen.intersection(source_seeds)
    earlier = {seed for stage,count in (("main",201),("probe",12),("test",12))
               for index in range(count) for block in ("selection","assessment")
               for seed in prior.world_seeds(stage,index,block)}
    assert not seen.intersection(earlier) and not source_seeds.intersection(earlier)


@pytest.mark.parametrize("size",[1,2,3,4])
def test_short_support_is_retained(size):
    value = p.assemble_union(*synthetic_union(size))
    assert value["union"]["keys"] == list("abcd")[:size]
    assert all(value["candidates"][a]["keys"] == value["union"]["keys"] for a in p.ARMS)
    assert all(v == (3 if size == 4 else None) for v in value["proposed_indices"].values())


def test_three_proposer_union_deduplication_and_missing_candidate():
    value = p.assemble_union(*synthetic_union(6,("d","e","f")))
    assert value["union"]["keys"] == list("abcdef")
    assert value["proposed_indices"] == {"anchor":3,"ranking":4,"ce":5}
    damaged = synthetic_union(6,("d","e","f"))
    damaged[1]["ranking"]["candidates"]["keys"][0] = "f"
    with pytest.raises(ValueError):
        p.assemble_union(*damaged)


@pytest.mark.parametrize("fit",[161001,161002,161003])
def test_roles_are_independent_of_hidden_input_and_do_not_mutate_ancestor_registry(fit):
    from tools.proposal_iteration_128 import arena as ancestor
    from tools.proposal_ranking_128 import verify
    before_models,before_artifacts = dict(ancestor._MODELS),dict(ancestor._MODEL_ARTIFACTS)
    state = private_state()
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    first = p.proposal_union(state,"hidden-invariant-ranking",fit)
    other = p.proposal_union(input_audit.hidden_alternative(state),"hidden-invariant-ranking",fit)
    for arm in p.ARMS:
        a = {k:v for k,v in first["proposals"][arm].items() if k != "timing"}
        b = {k:v for k,v in other["proposals"][arm].items() if k != "timing"}
        input_audit.compare(a,b)
        input_audit.compare(a,verify.independent_proposal(state,"hidden-invariant-ranking",f"{arm}_{fit}"))
        assert a["model_id"] == f"{arm}_{fit}"
        assert a["model_sha256"] == p._MODEL_ARTIFACTS[f"{arm}_{fit}"]["sha256"]
    assert first["union"] == other["union"]
    assert before == p.old.digest(p.old.serialize_state_snapshot(state))
    assert ancestor._MODELS == before_models and ancestor._MODEL_ARTIFACTS == before_artifacts


def worlds(values):
    return [{"outcomes":[{"wdl":v} for v in values]} for _ in range(128)]


def test_sparse_assessment_maps_union_indices_and_preserves_ties():
    candidates = {"anchor":{"indices":[2,0,1]},"ranking":{"indices":[0,1,2]},"ce":{"indices":[1,0,2]}}
    values,choices = p.choose(worlds([1.,0.,1.]),candidates,3)
    assert values == [1.,0.,1.] and choices == {"anchor":2,"ranking":0,"ce":0}
    result = p.summarize(worlds([-1.,0.]),[0,2],choices,{a:None for a in p.ARMS},3)
    assert result["arm_scores"] == {"anchor":.5,"ranking":0.,"ce":0.}
    assert result["assessment_indices"] == [0,2]
    with pytest.raises(ValueError,match="locked choices"):
        p.summarize(worlds([-1.,0.,1.]),[0,1,2],choices,{a:None for a in p.ARMS},3)
    damaged = worlds([0.,1.]); damaged[-1]["outcomes"].pop()
    with pytest.raises(ValueError,match="Missing"):
        p.summarize(damaged,[0,2],choices,{a:None for a in p.ARMS},3)


def test_real_private_root_locks_before_sparse_assessment(tmp_path,monkeypatch):
    spec = fixture_spec()
    original,calls = p.old.trajectory,0
    support = len(p.proposal_union(private_state(),spec["search_stream"],spec["fit"])["union"]["keys"])
    def observed(world,action):
        nonlocal calls
        if calls == 128*support:
            seal = p.old.read_json(p.selection_path("test",0,tmp_path))
            assert seal["selection"]["candidate_indices"] == list(range(support))
            assert seal["selection_lock"]["assessment_indices"] == sorted(set(seal["selection_lock"]["choices"].values()))
            assert "assessment" not in seal
        calls += 1
        return original(world,action)
    monkeypatch.setattr(p.old,"trajectory",observed)
    item = p.collect("test",spec,"fixture",tmp_path)
    record = p.c.verified(item)
    count = len(set(record["selection_lock"]["choices"].values()))
    assert calls == item["continuations"] == 128*(support+count)
    assert len(record["assessment"]["worlds"]) == 128
    assert all(len(w["outcomes"]) == count for w in record["assessment"]["worlds"])
    assert record["privacy"] == {"own":1,"opponent":1}
    assert record["selection_lock"]["created_at_ns"] <= p.selection_path("test",0,tmp_path).stat().st_mtime_ns <= record["assessment_started_at_ns"]
    row = p.statistics_row(record)
    assert row["arm_scores"] == record["summary"]["arm_scores"] and row["privacy"] == record["privacy"]
    changed = copy.deepcopy(record); changed["summary"]["arm_scores"]["ranking"] += .125
    with pytest.raises(ValueError,match="summary"):
        p.statistics_row(changed)
    before = p.old.file_hash(p.root_path("test",0,tmp_path))
    with pytest.raises(ValueError,match="One-shot"):
        p.collect("test",spec,"fixture",tmp_path)
    assert before == p.old.file_hash(p.root_path("test",0,tmp_path))


def test_registry_and_snapshot_guards_preserve_failed_evidence(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="production"):
        p.initialize_models(p.OUTPUT,fixture_registry())
    with pytest.raises(ValueError,match="production"):
        p.collect("test",fixture_spec(),"fixture",p.OUTPUT)
    with pytest.raises(ValueError,match="Synthetic"):
        p.collect("main",fixture_spec(),"fixture",tmp_path)
    assert "failure" in p.old.read_json(p.root_path("main",0,tmp_path))
    original = p.run_block
    def fail_assessment(state,actions,seeds,block,base):
        if seeds[0] == p.world_seeds("test",0,"assessment")[0]:
            assert p.selection_path("test",0,base).exists()
            raise RuntimeError("assessment fixture stop")
        return original(state,actions,seeds,block,base)
    monkeypatch.setattr(p,"run_block",fail_assessment)
    with pytest.raises(RuntimeError,match="fixture stop"):
        p.collect("test",fixture_spec(),"fixture",tmp_path/"failure")
    assert p.old.read_json(p.selection_path("test",0,tmp_path/"failure"))["selection_lock"]["choices"]
    assert "failure" in p.old.read_json(p.root_path("test",0,tmp_path/"failure"))
