"""Allocation, exact means, selection access and durable independent assessment."""
import copy
from fractions import Fraction

import pytest

from tools.equal_budget_breadth_128 import engine as p
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit


def fixture_registry():
    prior = p.old.read_json(p.c.ROOT/"outputs/proposal_iteration_128/models_lock.json")["models"]
    return {"manifest":"fixture", "models":{f"anchor_{fit}":dict(prior[f"anchor_{fit}"]) for fit in p.c.FITS}}


@pytest.fixture(scope="module",autouse=True)
def models(tmp_path_factory):
    p.initialize_models(tmp_path_factory.mktemp("equal_budget_breadth_engine"),fixture_registry())


def private_state():
    env = p.old.SplendorEnv(seed=2159800000)
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
            "phase":0,"turn":state.turn_index,"source_stage":"test","search_stream":"equal-budget-breadth-isolated-source",
            "snapshot_sha256":p.old.digest(snapshot),"fixture_snapshot":snapshot,
            "privacy":p.baseline.privacy(state),
            "canonical_support":len(p.old.actions_by_canonical_successor_key(state,p.old.SplendorEnv().legal_actions(state)))}


def synthetic_proposal(size, base_keys=None, scores=None):
    keys = [f"key{i}" for i in range(size)]
    reps = {key:{"representative":key} for key in keys}
    base_keys = keys[:min(size,4)] if base_keys is None else base_keys
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
    budget, allocation = value["budgets"], value["uniform_allocation"]
    m,n = min(size,4),min(size,8)
    q,r = divmod(128*m,n)
    assert budget["arm_continuations"] == {"anchor":128*m,"uniform":128*m,"teacher":128*n}
    assert budget["uniform_base_worlds"] == q and budget["uniform_remainder_count"] == r
    assert allocation["world_counts"] == [q+int(j in allocation["remainder_indices"]) for j in range(n)]
    assert value["union"]["keys"][:m] == [f"key{i}" for i in range(m)]
    assert len(value["union"]["keys"]) == n
    assert len(value["added_indices"]["ranked"]) == min(2,n-m)
    assert len(value["added_indices"]["random"]) == max(0,n-m-2)
    arithmetic,choices = p.choose(block([0.]*n),value["candidates"],budget,allocation)
    assert choices == dict.fromkeys(p.ARMS,0)
    for arm in p.ARMS:
        assert sum(arithmetic["counts"][arm]) == budget["arm_continuations"][arm]
        assert arithmetic["world_indices"][arm] == [list(range(count)) for count in arithmetic["counts"][arm]]
    if size <= 4:
        assert value["candidates"]["anchor"] == value["candidates"]["uniform"] == value["candidates"]["teacher"]
        assert all(counts == [128]*n for counts in arithmetic["counts"].values())
        assert len(set(map(tuple,arithmetic["means"].values()))) == 1


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
        proposal["candidates"]["keys"][:2] = ["key8","key8"]
    with pytest.raises(ValueError):
        p.assemble_union(proposal,reps,("test",0))


@pytest.mark.parametrize("size",[5,6,7,8])
def test_uniform_exact_prefix_denominators_and_first_ties(size):
    value = p.assemble_union(*synthetic_proposal(size),("test",0))
    selection = block([1.]*size)
    arithmetic,choices = p.choose(selection,value["candidates"],value["budgets"],value["uniform_allocation"])
    assert choices == dict.fromkeys(p.ARMS,0)
    assert arithmetic["means"]["uniform"] == [1.]*size
    for wi,world in enumerate(selection["worlds"]):
        for j,outcome in enumerate(world["outcomes"]):
            outcome["wdl"] = (-1,0,1)[(wi*7+j*11)%3]
    arithmetic,choices = p.choose(selection,value["candidates"],value["budgets"],value["uniform_allocation"])
    for arm in p.ARMS:
        scores=[]
        for j,count in enumerate(arithmetic["counts"][arm]):
            total=sum(selection["worlds"][wi]["outcomes"][j]["wdl"] for wi in range(count))
            assert arithmetic["sums_wdl"][arm][j] == total
            assert arithmetic["means"][arm][j] == total/count
            scores.append(Fraction(total,count))
        assert choices[arm] == scores.index(max(scores))


def test_unallocated_teacher_outcomes_cannot_change_uniform_choice():
    value=p.assemble_union(*synthetic_proposal(8),("test",0))
    selection=block([1.,-1.,-1.,-1.,0.,-1.,-1.,-1.])
    first,choices=p.choose(selection,value["candidates"],value["budgets"],value["uniform_allocation"])
    assert choices == dict.fromkeys(p.ARMS,0)
    for world in selection["worlds"][64:]:
        world["outcomes"][0]["wdl"]=-1.
        world["outcomes"][4]["wdl"]=1.
    other,new_choices=p.choose(selection,value["candidates"],value["budgets"],value["uniform_allocation"])
    assert new_choices == {"anchor":0,"uniform":0,"teacher":4}
    assert all(first[field]["uniform"] == other[field]["uniform"] for field in first)


@pytest.mark.parametrize("damage",["missing_world","missing_candidate","reordered_indices","wrong_budget","counts","remainder","order","wdl"])
def test_incomplete_selection_or_changed_allocation_is_rejected(damage):
    value=p.assemble_union(*synthetic_proposal(5),("test",0))
    selection=block([0.]*5)
    if damage == "missing_world": selection["worlds"].pop()
    elif damage == "missing_candidate": selection["worlds"][-1]["outcomes"].pop()
    elif damage == "reordered_indices": selection["candidate_indices"]=[1,0,2,3,4]
    elif damage == "wrong_budget": value["budgets"]["uniform_base_worlds"] += 1
    elif damage == "counts": value["uniform_allocation"]["world_counts"][0] += 1
    elif damage == "remainder": value["uniform_allocation"]["remainder_indices"] = []
    elif damage == "order": value["uniform_allocation"]["remainder_order"] = [0]*5
    else: selection["worlds"][0]["outcomes"][0]["wdl"] = .5
    with pytest.raises(ValueError):
        p.choose(selection,value["candidates"],value["budgets"],value["uniform_allocation"])


def test_sparse_assessment_maps_distinct_locked_choices():
    choices={"anchor":0,"uniform":4,"teacher":5}
    result=p.summarize(worlds([-1.,0.,1.]),[0,4,5],choices,dict.fromkeys(p.ARMS,3),8)
    assert result["arm_scores"] == {"anchor":0.,"uniform":.5,"teacher":1.}
    with pytest.raises(ValueError,match="locked choices"):
        p.summarize(worlds([-1.,0.]),[0,4],choices,dict.fromkeys(p.ARMS,3),8)


def test_all_world_and_source_namespaces_are_disjoint():
    from tools.equal_budget_breadth_128 import source
    from tools.proposal_breadth_128 import engine as breadth
    from tools.proposal_specialist_128 import engine as specialist
    from tools.proposal_breadth_learning_128 import engine as learning
    seen=set()
    for stage,count in (("main",360),("probe",18),("test",18)):
        for index in range(count):
            for name in ("selection","assessment"):
                seeds=p.world_seeds(stage,index,name)
                assert len(seeds)==len(set(seeds))==128 and not seen.intersection(seeds)
                seen.update(seeds)
    source_seeds={s["environment_seed"] for stage in ("main","probe","test") for s in source.schedule(stage)}
    assert len(source_seeds)==1152 and seen.isdisjoint(source_seeds)
    for prior in (breadth,specialist,learning):
        blocks=("selection","extension","assessment") if prior is breadth else ("selection","assessment")
        earlier={seed for stage,count in (("main",360),("probe",18),("test",18)) for index in range(count)
                 for name in blocks for seed in prior.world_seeds(stage,index,name)}
        assert seen.isdisjoint(earlier) and source_seeds.isdisjoint(earlier)


@pytest.mark.parametrize("fit",[161001,161002,161003])
def test_private_inputs_preserve_anchor_and_never_mutate_ancestor_registry(fit):
    from tools.proposal_breadth_128 import engine as ancestor
    before_models,before_artifacts=dict(ancestor._MODELS),dict(ancestor._MODEL_ARTIFACTS)
    state=private_state()
    before=p.old.digest(p.old.serialize_state_snapshot(state))
    first=p.proposal_union(state,"hidden-invariant-equal-budget-breadth",fit,("test",0))
    other=p.proposal_union(input_audit.hidden_alternative(state),"hidden-invariant-equal-budget-breadth",fit,("test",0))
    for value in (first,other):
        anchor=value["proposals"]["anchor"]
        assert anchor["model_id"] == f"anchor_{fit}"
        assert anchor["model_sha256"] == p._MODEL_ARTIFACTS[f"anchor_{fit}"]["sha256"]
        assert value["candidates"]["anchor"]["keys"][:3] == anchor["reference"]["keys"][:3]
        assert anchor["schedule"] == p.world128.root_schedule(state,"hidden-invariant-equal-budget-breadth",128)
    input_audit.compare({k:v for k,v in first["proposals"]["anchor"].items() if k!="timing"},
                        {k:v for k,v in other["proposals"]["anchor"].items() if k!="timing"})
    assert all(first[k] == other[k] for k in ("union","budgets","uniform_allocation"))
    assert before == p.old.digest(p.old.serialize_state_snapshot(state))
    assert ancestor._MODELS == before_models and ancestor._MODEL_ARTIFACTS == before_artifacts


def test_real_private_root_locks_accesses_and_all_choices_before_assessment(tmp_path,monkeypatch):
    spec=fixture_spec()
    proposed=p.proposal_union(private_state(),spec["search_stream"],spec["fit"],("test",0))
    budget=proposed["budgets"]
    selection_calls=128*budget["breadth_candidates"]
    original,calls=p.old.trajectory,0
    def observed(world,action):
        nonlocal calls
        if calls == selection_calls:
            seal=p.old.read_json(p.selection_path("test",0,tmp_path))
            assert len(seal["selection"]["worlds"])==128
            assert set(seal["selection_lock"]["arithmetic"]["world_indices"])==set(p.ARMS)
            assert seal["selection_lock"]["assessment_indices"]==sorted(set(seal["selection_lock"]["choices"].values()))
            assert "assessment" not in seal and "extension" not in seal
        calls+=1
        return original(world,action)
    monkeypatch.setattr(p.old,"trajectory",observed)
    item=p.collect("test",spec,"fixture",tmp_path)
    record=p.c.verified(item)
    count=len(set(record["selection_lock"]["choices"].values()))
    assert calls==item["continuations"]==selection_calls+128*count
    assert all(len(w["outcomes"])==count for w in record["assessment"]["worlds"])
    assert set(record["work"])=={"selection","assessment"}
    assert record["privacy"]=={"own":1,"opponent":1}
    assert record["selection_lock"]["created_at_ns"]<=p.selection_path("test",0,tmp_path).stat().st_mtime_ns<=record["assessment_started_at_ns"]
    row=p.statistics_row(record)
    assert row["arm_scores"]==record["summary"]["arm_scores"] and row["budgets"]==budget
    for damage in ("summary","access","allocation","schedule"):
        changed=copy.deepcopy(record)
        if damage=="summary": changed["summary"]["arm_scores"]["uniform"]+=.125
        elif damage=="access": changed["selection_lock"]["arithmetic"]["world_indices"]["uniform"][0].append(127)
        elif damage=="allocation": changed["uniform_allocation"]["world_counts"][0]+=1
        else: changed["selection"]["world_seeds"][0]+=1
        with pytest.raises(ValueError): p.statistics_row(changed)
    before=p.old.file_hash(p.root_path("test",0,tmp_path))
    with pytest.raises(ValueError,match="One-shot"): p.collect("test",spec,"fixture",tmp_path)
    assert before==p.old.file_hash(p.root_path("test",0,tmp_path))


def test_registry_and_snapshot_guards_preserve_failed_evidence(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="production"):
        p.initialize_models(p.OUTPUT,fixture_registry())
    with pytest.raises(ValueError,match="production"):
        p.collect("test",fixture_spec(),"fixture",p.OUTPUT)
    with pytest.raises(ValueError,match="Synthetic"):
        p.collect("main",fixture_spec(),"fixture",tmp_path)
    assert "failure" in p.old.read_json(p.root_path("main",0,tmp_path))
    original=p.run_block
    def fail_assessment(state,actions,seeds,block,base):
        if seeds and seeds[0]==p.world_seeds("test",0,"assessment")[0]:
            assert p.selection_path("test",0,base).exists()
            raise RuntimeError("assessment fixture stop")
        return original(state,actions,seeds,block,base)
    monkeypatch.setattr(p,"run_block",fail_assessment)
    with pytest.raises(RuntimeError,match="fixture stop"):
        p.collect("test",fixture_spec(),"fixture",tmp_path/"failure")
    assert p.old.read_json(p.selection_path("test",0,tmp_path/"failure"))["selection_lock"]["choices"]
    assert "failure" in p.old.read_json(p.root_path("test",0,tmp_path/"failure"))
