"""Independent ragged-budget arithmetic, sealed choices and real private replay."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from tools.equal_budget_breadth_128 import verify as v, engine


def synthetic_specs(count=360):
    specs = []
    for i in range(count):
        fit, seat, phase = v.CELLS[i % 18]
        specs.append({"index": i, "pair": 3*i, "game_index": 3*i, "fit": fit,
                      "seat": seat, "phase": phase,
                      "privacy": {"own": i % 2, "opponent": (i//2) % 2},
                      "canonical_support": 1+i % 11})
    return specs


def test_metadata_only_sample_plan_full_probe_and_ragged_stress():
    specs = synthetic_specs()
    plan = v.sample_plan(specs, "main", "binding")
    assert plan == v.sample_plan(copy.deepcopy(specs), "main", "binding")
    stress = {row["index"] for row in plan["selections"] if row["stress"]}
    assert 18 <= len(stress) <= 48 and not plan["coverage"]["missing_cells"]
    assert all(row["selected_index"] in stress for row in plan["stress_cells"])
    for spec, row in zip(specs, plan["selections"], strict=True):
        for block, length in v.block_lengths(spec["canonical_support"]).items():
            assert len(row["ordinary"][block]) == (length+15)//16
            assert len(row["replay"][block]) == (length if row["stress"] else (length+15)//16)
    altered = [{**spec, "irrelevant_outcome": i} for i, spec in enumerate(specs)]
    other = v.sample_plan(altered, "main", "other")
    assert other["selections"] == plan["selections"] and other["stress_cells"] == plan["stress_cells"]
    probe = v.sample_plan(synthetic_specs(18), "probe", "binding")
    assert all(row["stress"] for row in probe["selections"])
    assert probe["coverage"]["replayed_worlds"] == probe["coverage"]["recorded_worlds"]


def test_world_namespaces_disjoint_selection_and_assessment():
    seen = set()
    for stage, count in (("probe",18),("main",360),("test",18)):
        for index in range(count):
            for block in v.BLOCKS:
                seeds = v._world_seeds(stage,index,block)
                assert seeds == engine.world_seeds(stage,index,block)
                assert not set(seeds) & seen
                seen.update(seeds)


@pytest.mark.parametrize("support,q,r",[(1,128,0),(2,128,0),(3,128,0),(4,128,0),(5,102,2),(6,85,2),(7,73,1),(8,64,0),(50,64,0)])
def test_independent_budgets_and_identity_allocation(support,q,r):
    budget = v.independent_budgets(support)
    assert budget == engine.budgets(support)
    assert budget["uniform_base_worlds"] == q and budget["uniform_remainder_count"] == r
    assert budget["arm_continuations"]["uniform"] == budget["arm_continuations"]["anchor"]
    keys = [f"key_{j}" for j in range(min(support,8))]
    allocation = v.independent_allocation(keys,budget,"test",0)
    assert allocation == engine.uniform_allocation(keys,budget,("test",0))
    assert sum(allocation["world_counts"]) == min(support,4)*128


def worlds(values,count):
    return [{"outcomes":[{"wdl":value} for value in values]} for _ in range(count)]


def candidates(support):
    keys = list(range(min(support,8)))
    arms = {}
    for arm in v.ARMS:
        width = min(support,4) if arm == "anchor" else len(keys)
        arms[arm] = {"keys":keys[:width],"indices":list(range(width))}
    return keys,arms


@pytest.mark.parametrize("support",range(1,9))
def test_all_support_equal_rational_means_preserve_first_ties(support):
    keys,arms = candidates(support)
    budget = v.independent_budgets(support)
    allocation = v.independent_allocation(keys,budget,"test",0)
    selection = worlds([1]*support,128)
    arithmetic,choices = v._selection(selection,arms,keys,budget,allocation)
    assert choices == dict.fromkeys(v.ARMS,0)
    assert arithmetic["counts"]["uniform"] == allocation["world_counts"]
    assert all(sum(arithmetic["counts"][arm]) == budget["arm_continuations"][arm] for arm in v.ARMS)
    record={"worlds":selection,"candidate_indices":list(range(support))}
    assert (arithmetic,choices) == engine.choose(record,arms,budget,allocation)
    result = v._summary(selection,worlds([1],128),arms,keys,dict.fromkeys(v.ARMS,3 if support >= 4 else None),[0],budget,{},allocation)
    assert result["choices"] == choices and result["arm_scores"] == dict.fromkeys(v.ARMS,1.)


@pytest.mark.parametrize("support",[5,6,7,8])
def test_uniform_cannot_read_unallocated_suffix_even_when_teacher_changes(support):
    keys,arms = candidates(support)
    budget = v.independent_budgets(support)
    allocation = v.independent_allocation(keys,budget,"test",0)
    selection = worlds([0]*support,128)
    original,choices = v._selection(selection,arms,keys,budget,allocation)
    altered = copy.deepcopy(selection)
    for row in altered[allocation["world_counts"][-1]:]:
        row["outcomes"][-1]["wdl"] = 1
    changed,other = v._selection(altered,arms,keys,budget,allocation)
    assert other["uniform"] == choices["uniform"] == 0 and other["teacher"] == support-1
    for field in ("means","sums_wdl","counts","world_indices"):
        assert changed[field]["uniform"] == original[field]["uniform"]


def test_sparse_assessment_maps_three_locked_choices_without_reselection():
    keys,arms = candidates(8)
    budget = v.independent_budgets(8)
    allocation = v.independent_allocation(keys,budget,"test",0)
    selection = worlds([-1,-1,-1,0,-1,-1,-1,-1],128)
    for i,row in enumerate(selection):
        row["outcomes"][3]["wdl"] = 1 if i < 32 else 0
        row["outcomes"][5]["wdl"] = 1 if i < 64 else -1
        row["outcomes"][6]["wdl"] = 1 if i < 32 or i >= 64 else 0
    arithmetic,choices = v._selection(selection,arms,keys,budget,allocation)
    assert choices == {"anchor":3,"uniform":5,"teacher":6}
    proposed = dict.fromkeys(v.ARMS,3)
    result = v._summary(selection,worlds([-1,0,1],128),arms,keys,proposed,[3,5,6],budget,{},allocation)
    assert result["arm_scores"] == {"anchor":0.,"uniform":.5,"teacher":1.}
    with pytest.raises(AssertionError):
        v._summary(selection,worlds([-1,0,1],128),arms,keys,proposed,[2,3,5],budget,{},allocation)


def statistic_rows():
    result=[]
    for i,spec in enumerate(synthetic_specs()):
        anchor=100+i % 53
        values={"anchor":anchor,"uniform":anchor+(i//18) % 11-3,"teacher":anchor+(i//18) % 7-3}
        result.append({**spec,"arm_scores":{arm:value/256 for arm,value in values.items()}})
    return result


def test_bootstrap_matches_independent_indexed_resampling_and_rejects_bad_units():
    rows=statistic_rows();result=v.statistics(rows)
    matrices=[np.asarray([[r["arm_scores"][a]-r["arm_scores"][b] for _,a,b in v.CONTRASTS]
                          for r in rows if (r["fit"],r["seat"],r["phase"]) == cell]) for cell in v.CELLS]
    rng=np.random.default_rng(2153000000);samples=[]
    for _ in range(50):
        total=np.zeros((1000,3))
        for matrix in matrices:total += matrix[rng.integers(0,20,size=(1000,20))].sum(axis=1)
        samples.append(total/360)
    boot=np.concatenate(samples)
    for j,(name,_,_) in enumerate(v.CONTRASTS):
        assert result["contrasts"][name]["mean_score_difference"] == pytest.approx(sum(m[:,j].sum() for m in matrices)/360,abs=1e-12)
        assert result["contrasts"][name]["ci95"] == pytest.approx(np.quantile(boot[:,j],[.025,.975]),abs=1e-12)
    for damage in ("duplicate","stratum","score"):
        bad=copy.deepcopy(rows)
        if damage == "duplicate":bad[1]["game_index"]=bad[1]["pair"]=bad[0]["game_index"]
        elif damage == "stratum":bad[0]["seat"]=1
        else:bad[0]["arm_scores"]["anchor"]=.51
        with pytest.raises(AssertionError):v.statistics(bad)


@pytest.mark.parametrize("delta,teacher_delta,passed",[(2,0,True),(1,0,False),(2,2,True),(0,0,False),(-2,0,False),(3,4,True),(3,100,True)])
def test_gate_only_requires_practical_uniform_anchor_superiority(delta,teacher_delta,passed):
    rows=[{**r,"arm_scores":{"anchor":.5,"uniform":(128+delta)/256,"teacher":(128+teacher_delta)/256}}
          for r in statistic_rows()]
    value=v.statistics(rows)
    assert value["gate_pass"] is passed
    assert value["classification"] == ("positive_conditional_equal_budget_breadth" if passed else "practical_progress_not_established")


def private_state():
    env=v.old.SplendorEnv(seed=2159800000);state=env.initial_state()
    for _ in range(2):
        action=next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state=env.step(state,action)
    for player in state.players:
        player.score=14;player.bonuses=dict.fromkeys(player.bonuses,5)
    return state


def support_state(support):
    state=v.old.SplendorEnv(seed=2159810000+support).initial_state()
    cards=[card for tier in state.visible_tier_cards for card in state.visible_tier_cards[tier]+state.hidden_tier_decks[tier]]
    chosen=sorted([card for card in cards if card.points > 0],key=lambda card:card.card_id)[:support]
    state.visible_tier_cards={tier:[] for tier in state.visible_tier_cards}
    state.hidden_tier_decks={tier:[] for tier in state.hidden_tier_decks}
    state.deck_counts=dict.fromkeys(state.deck_counts,0);state.bank_tokens=dict.fromkeys(state.bank_tokens,0);state.nobles=[]
    for player in state.players:
        player.score=14;player.bonuses=dict.fromkeys(player.bonuses,10)
    reserved=chosen[:3]
    state.players[0].reserved_cards=reserved;state.players[0].reserved_card_sources=["deck"]*len(reserved)
    for card in chosen[3:]:state.visible_tier_cards[card.tier].append(card)
    state.players[1].purchased_cards=[card for card in cards if card not in chosen]
    return state


def install_models(base):
    parent=v.old.read_json(v.c.SOURCE/"models_lock.json")
    registry={"manifest":"fixture","models":{f"anchor_{fit}":parent["models"][f"anchor_{fit}"] for fit in v.FITS}}
    v.old.save_new(base/"models_lock.json",registry);engine.initialize_models(base)


def fixture_spec(state,index):
    snapshot=json.loads(json.dumps(v.old.serialize_state_snapshot(state)))
    legal=v.old.SplendorEnv().legal_actions(state)
    return {"index":index,"pair":index,"game_index":index,"fit":161001,"seat":state.current_player,
            "phase":min(state.turn_index//20,2),"turn":state.turn_index,"privacy":engine.baseline.privacy(state),
            "canonical_support":len({repr(v.old.canonical_successor_action_key(state,a)) for a in legal}),
            "source_stage":"test","search_stream":"independent-breadth-fixture",
            "snapshot_sha256":v.old.digest(snapshot),"fixture_snapshot":snapshot}


@pytest.fixture(scope="module")
def fixture_artifacts(tmp_path_factory):
    base=tmp_path_factory.mktemp("breadth-verifier");install_models(base)
    specs=[fixture_spec(private_state(),0)]+[fixture_spec(support_state(n),n) for n in range(1,8)]
    assert [s["canonical_support"] for s in specs[1:]] == list(range(1,8))
    sample=v.sample_plan(specs,"test","fixture");v.old.save_new(base/"test/audit_sample.json",sample)
    items=[engine.collect("test",spec,"fixture",base) for spec in specs]
    return base,specs,sample,items


def test_private_and_every_ragged_support_full_independent_replay(fixture_artifacts):
    base,specs,plan,items=fixture_artifacts
    for spec,item in zip(specs,items,strict=True):
        receipt=v.c.verified(v.audit("test",spec,item,"fixture",base))
        assert receipt["passed"] and receipt["row"] == engine.statistics_row(v.c.verified(item))
        assert receipt["full_inner_replay"] and receipt["recorded_continuations"] == receipt["replayed_continuations"]
        assert receipt["recorded_worlds"] == 256
        for block in v.BLOCKS:
            assert receipt[f"recorded_{block}_worlds"] == receipt[f"replayed_{block}_worlds"]
            assert receipt[f"recorded_{block}_continuations"] == receipt[f"replayed_{block}_continuations"]


@pytest.mark.parametrize("damage",["none","model","model_hash","summary","assessment_indices","selection_indices",
                                   "allocation_order","allocation_count","allocation_budget","access_suffix","ranked_order",
                                   "added_membership","random_order","world","outcome","sampled_trace","seal_time","sample_plan","work"])
def test_independent_audit_rejects_changed_evidence(fixture_artifacts,tmp_path,damage):
    base,specs,plan,items=fixture_artifacts
    engine.initialize_models(base)
    spec=specs[0];root=copy.deepcopy(v.c.verified(items[0]))
    v.old.save_new(tmp_path/"models_lock.json",v.old.read_json(base/"models_lock.json"))
    v.old.save_new(tmp_path/"test/audit_sample.json",plan)
    # Copy the actual independently recorded seal; adjust only filesystem times.
    seal_path=tmp_path/"test/selections/state_0000_selection.json"
    v.old.save_new(seal_path,v.c.verified(root["selection_artifact"]))
    root["selection_artifact"]=v.c.artifact(seal_path)
    root["assessment_started_at_ns"]=time.time_ns()
    # Both sample and seal need to precede the original sealed timestamp too.
    import os
    os.utime(tmp_path/"test/audit_sample.json",ns=(root["selection_lock"]["created_at_ns"]-1,)*2)
    if damage == "model":root["proposals"]["anchor"]["model_id"]="anchor_161002"
    elif damage == "model_hash":root["proposals"]["anchor"]["model_sha256"]="0"*64
    elif damage == "summary":root["summary"]["arm_scores"]["uniform"] += 1/256
    elif damage == "assessment_indices":root["assessment"]["candidate_indices"]=[]
    elif damage == "selection_indices":root["selection"]["candidate_indices"]=root["selection"]["candidate_indices"][1:]
    elif damage == "allocation_order":root["uniform_allocation"]["remainder_order"].reverse()
    elif damage == "allocation_count":root["uniform_allocation"]["world_counts"][0] += 1
    elif damage == "allocation_budget":root["budgets"]["uniform_base_worlds"] += 1
    elif damage == "access_suffix":root["selection_lock"]["arithmetic"]["world_indices"]["uniform"][0].append(127)
    elif damage == "ranked_order":root["union"]["keys"][4:6]=reversed(root["union"]["keys"][4:6])
    elif damage == "added_membership":root["added_indices"]["ranked"]=root["added_indices"]["random"]
    elif damage == "random_order":root["union"]["keys"][6:8]=reversed(root["union"]["keys"][6:8])
    elif damage == "world":root["assessment"]["worlds"][0]["state_sha256"]="0"*64
    elif damage == "outcome":root["assessment"]["worlds"][0]["outcomes"][0]["wdl"]=.5
    elif damage == "sampled_trace":root["assessment"]["worlds"][0]["outcomes"][0]["trace_sha256"]="0"*64
    elif damage == "seal_time":root["assessment_started_at_ns"]=root["selection_lock"]["created_at_ns"]-1
    elif damage == "sample_plan":
        value=copy.deepcopy(plan);value["selections"][0]["replay"]["selection"]=[]
        path=tmp_path/"test/audit_sample.json";path.write_text(json.dumps(value),encoding="utf-8")
    elif damage == "work":root["work"]["selection"]["continuations"] += 1
    path=tmp_path/"test/roots/state_0000.json";v.old.save_new(path,root)
    if damage == "none":assert v.c.verified(v.audit("test",spec,v.c.artifact(path),"fixture",tmp_path))["passed"]
    else:
        with pytest.raises((AssertionError,ValueError)):
            v.audit("test",spec,v.c.artifact(path),"fixture",tmp_path)


def test_fixture_override_isolated_from_production():
    with pytest.raises(AssertionError):v._source_state("test",{"fixture_snapshot":{}},v.OUTPUT)
    with pytest.raises(AssertionError):v._source_state("main",{"fixture_snapshot":{}},Path("elsewhere"))
