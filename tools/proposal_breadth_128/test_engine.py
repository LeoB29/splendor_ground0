"""Budget equality, stable expansion, independent choices and private inputs."""
import copy

import pytest

from tools.proposal_breadth_128 import engine as p
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit


def fixture_registry():
    prior = p.old.read_json(p.c.SOURCE/"models_lock.json")["models"]
    return {"manifest":"fixture", "models":{f"anchor_{fit}":dict(prior[f"anchor_{fit}"]) for fit in p.c.FITS}}


@pytest.fixture(scope="module",autouse=True)
def models(tmp_path_factory):
    p.initialize_models(tmp_path_factory.mktemp("breadth_engine"),fixture_registry())


def private_state():
    env = p.old.SplendorEnv(seed=2109800000)
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
            "phase":0,"turn":state.turn_index,"source_stage":"test","search_stream":"breadth-isolated-source",
            "snapshot_sha256":p.old.digest(snapshot),"fixture_snapshot":snapshot,
            "privacy":p.baseline.privacy(state),
            "canonical_support":len(p.old.actions_by_canonical_successor_key(state,p.old.SplendorEnv().legal_actions(state)))}


def synthetic_proposal(size, base_keys=None, scores=None):
    keys = [f"key{i}" for i in range(size)]
    reps = {key:{"representative":key} for key in keys}
    base_keys = keys[:min(size,4)] if base_keys is None else base_keys
    # Incumbent representatives deliberately differ from canonical defaults.
    proposal = {"group_keys":keys, "learned_group_scores":list(range(size)) if scores is None else scores,
                "candidates":{"keys":base_keys, "actions":[{"incumbent":key} for key in base_keys],
                              "canonical_support":size}}
    return proposal,reps


def worlds(values, count=128):
    return [{"outcomes":[{"wdl":v} for v in values]} for _ in range(count)]


def block(values, count=128):
    return {"candidate_indices":list(range(len(values))), "worlds":worlds(values,count)}


@pytest.mark.parametrize("size",range(1,12))
def test_exact_budgets_and_short_supports(size):
    value = p.assemble_union(*synthetic_proposal(size),("test",0))
    budget = value["budgets"]
    m,n = min(size,4),min(size,8)
    assert budget["arm_continuations"]["breadth"] == budget["arm_continuations"]["precision"] == n*128
    assert budget["precision_worlds"] == 128*n//m
    assert value["union"]["keys"][:m] == [f"key{i}" for i in range(m)]
    assert len(value["union"]["keys"]) == n
    assert len(value["added_indices"]["ranked"]) == min(2,n-m)
    assert len(value["added_indices"]["random"]) == max(0,n-m-2)
    if size <= 4:
        assert value["candidates"]["anchor"] == value["candidates"]["breadth"] == value["candidates"]["precision"]
        assert budget["extension_worlds"] == 0
        values,choices = p.choose(block([0.]*n),block([0.]*m,0),value["candidates"],budget)
        assert all(v == 0 for v in choices.values()) and len(set(map(tuple,values.values()))) == 1


def test_rank_ties_uniform_order_no_replacement_and_incumbent_representatives():
    proposal,reps = synthetic_proposal(10,["key7","key2","key4","key0"],[1.]*10)
    value = p.assemble_union(proposal,reps,("test",3))
    assert value["union"]["keys"][:6] == ["key7","key2","key4","key0","key1","key3"]
    remaining = ["key5","key6","key8","key9"]
    expected = sorted(remaining,key=lambda key:(p.old.digest([p.CONFIG["random_tag"],"test",3,key]),proposal["group_keys"].index(key)))[:2]
    assert value["union"]["keys"][6:] == expected
    assert len(set(value["union"]["keys"])) == 8
    assert value["union"]["actions"][:4] == proposal["candidates"]["actions"]
    assert value["union"]["actions"][4:] == [reps[key] for key in value["union"]["keys"][4:]]
    assert p.assemble_union(proposal,reps,("test",3)) == value


@pytest.mark.parametrize("damage",["missing","duplicate","nan","bad_base"])
def test_expansion_rejects_incomplete_or_ambiguous_support(damage):
    proposal,reps = synthetic_proposal(9)
    if damage == "missing":
        reps.pop("key8")
    elif damage == "duplicate":
        proposal["group_keys"][-1] = "key7"
    elif damage == "nan":
        proposal["learned_group_scores"][-1] = float("nan")
    else:
        proposal["candidates"]["keys"][0] = "key8"
        proposal["candidates"]["keys"][1] = "key8"
    with pytest.raises(ValueError):
        p.assemble_union(proposal,reps,("test",0))


@pytest.mark.parametrize("size",[5,6,7,8])
def test_precision_combines_integer_sums_with_ragged_extension(size):
    value = p.assemble_union(*synthetic_proposal(size),("test",0))
    selection = block([0.]*size)
    for i,w in enumerate(selection["worlds"]):
        w["outcomes"][0]["wdl"] = 1. if i < 32 else 0.
        w["outcomes"][1]["wdl"] = 1. if i < 24 else 0.
        w["outcomes"][4]["wdl"] = 1. if i < 64 else 0.
    extra = value["budgets"]["extension_worlds"]
    extension = block([-1.,0.,-1.,-1.],extra)
    means,choices = p.choose(selection,extension,value["candidates"],value["budgets"])
    assert choices == {"anchor":0,"breadth":4,"precision":1}
    assert means["precision"][0] == (32-extra)/(128+extra)
    assert means["precision"][1] == 24/(128+extra)
    result = p.summarize(worlds([-1.,0.,1.]),[0,1,4],choices,value["proposed_indices"],size)
    assert result["arm_scores"] == {"anchor":0.,"breadth":1.,"precision":.5}
    # Assessment must value only the already locked choices, never choose anew.
    with pytest.raises(ValueError,match="locked choices"):
        p.summarize(worlds([-1.,0.]),[0,4],choices,value["proposed_indices"],size)


@pytest.mark.parametrize("damage",["missing_world","missing_candidate","reordered_indices","wrong_budget","empty_extension"])
def test_incomplete_extension_or_changed_budget_is_rejected(damage):
    value = p.assemble_union(*synthetic_proposal(5),("test",0))
    selection,extension = block([0.]*5),block([0.]*4,32)
    if damage == "missing_world":
        extension["worlds"].pop()
    elif damage == "missing_candidate":
        extension["worlds"][-1]["outcomes"].pop()
    elif damage == "reordered_indices":
        extension["candidate_indices"] = [1,0,2,3]
    elif damage == "wrong_budget":
        value["budgets"]["precision_worlds"] = 256
    else:
        extension["worlds"] = []
    with pytest.raises(ValueError):
        p.choose(selection,extension,value["candidates"],value["budgets"])


def test_all_world_and_source_namespaces_are_disjoint():
    from tools.proposal_breadth_128 import source
    from tools.proposal_ranking_128 import engine as prior
    seen = set()
    for stage,count in (("main",360),("probe",18),("test",18)):
        for index in range(count):
            for name in ("selection","extension","assessment"):
                seeds = p.world_seeds(stage,index,name)
                assert len(seeds) == len(set(seeds)) == 128 and not seen.intersection(seeds)
                seen.update(seeds)
    source_seeds = {s["environment_seed"] for stage in ("main","probe","test") for s in source.schedule(stage)}
    assert len(source_seeds) == 1152 and not seen.intersection(source_seeds)
    earlier = {seed for stage,count in (("main",360),("probe",18),("test",18))
               for index in range(count) for name in ("selection","assessment") for seed in prior.world_seeds(stage,index,name)}
    assert not seen.intersection(earlier) and not source_seeds.intersection(earlier)


@pytest.mark.parametrize("fit",[161001,161002,161003])
def test_private_inputs_preserve_anchor_and_never_mutate_ancestor_registry(fit):
    from tools.proposal_ranking_128 import engine as ancestor
    before_models,before_artifacts = dict(ancestor._MODELS),dict(ancestor._MODEL_ARTIFACTS)
    state = private_state()
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    first = p.proposal_union(state,"hidden-invariant-breadth",fit,("test",0))
    other = p.proposal_union(input_audit.hidden_alternative(state),"hidden-invariant-breadth",fit,("test",0))
    for value in (first,other):
        anchor = value["proposals"]["anchor"]
        assert anchor["model_id"] == f"anchor_{fit}"
        assert anchor["model_sha256"] == p._MODEL_ARTIFACTS[f"anchor_{fit}"]["sha256"]
        assert value["candidates"]["anchor"]["keys"][:3] == anchor["reference"]["keys"][:3]
    a = {k:v for k,v in first["proposals"]["anchor"].items() if k != "timing"}
    b = {k:v for k,v in other["proposals"]["anchor"].items() if k != "timing"}
    input_audit.compare(a,b)
    assert first["union"] == other["union"] and first["budgets"] == other["budgets"]
    assert before == p.old.digest(p.old.serialize_state_snapshot(state))
    assert ancestor._MODELS == before_models and ancestor._MODEL_ARTIFACTS == before_artifacts


def test_real_private_root_locks_both_selection_blocks_before_assessment(tmp_path,monkeypatch):
    spec = fixture_spec()
    proposed = p.proposal_union(private_state(),spec["search_stream"],spec["fit"],("test",0))
    budget = proposed["budgets"]
    selection_calls = 128*budget["breadth_candidates"]+budget["extension_worlds"]*budget["anchor_candidates"]
    original,calls = p.old.trajectory,0
    def observed(world,action):
        nonlocal calls
        if calls == selection_calls:
            seal = p.old.read_json(p.selection_path("test",0,tmp_path))
            assert len(seal["selection"]["worlds"]) == 128
            assert len(seal["extension"]["worlds"]) == budget["extension_worlds"]
            assert seal["selection_lock"]["assessment_indices"] == sorted(set(seal["selection_lock"]["choices"].values()))
            assert "assessment" not in seal
        calls += 1
        return original(world,action)
    monkeypatch.setattr(p.old,"trajectory",observed)
    item = p.collect("test",spec,"fixture",tmp_path)
    record = p.c.verified(item)
    count = len(set(record["selection_lock"]["choices"].values()))
    assert calls == item["continuations"] == selection_calls+128*count
    assert all(len(w["outcomes"]) == count for w in record["assessment"]["worlds"])
    assert record["privacy"] == {"own":1,"opponent":1}
    assert record["selection_lock"]["created_at_ns"] <= p.selection_path("test",0,tmp_path).stat().st_mtime_ns <= record["assessment_started_at_ns"]
    row = p.statistics_row(record)
    assert row["arm_scores"] == record["summary"]["arm_scores"] and row["budgets"] == budget
    changed = copy.deepcopy(record); changed["summary"]["arm_scores"]["breadth"] += .125
    with pytest.raises(ValueError,match="summary"):
        p.statistics_row(changed)
    changed = copy.deepcopy(record); changed["extension"]["worlds"].pop()
    with pytest.raises(ValueError,match="schedule"):
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
        if seeds and seeds[0] == p.world_seeds("test",0,"assessment")[0]:
            assert p.selection_path("test",0,base).exists()
            raise RuntimeError("assessment fixture stop")
        return original(state,actions,seeds,block,base)
    monkeypatch.setattr(p,"run_block",fail_assessment)
    with pytest.raises(RuntimeError,match="fixture stop"):
        p.collect("test",fixture_spec(),"fixture",tmp_path/"failure")
    assert p.old.read_json(p.selection_path("test",0,tmp_path/"failure"))["selection_lock"]["choices"]
    assert "failure" in p.old.read_json(p.root_path("test",0,tmp_path/"failure"))
