"""Independent ragged-budget arithmetic, sealed choices and real private replay."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from tools.proposal_breadth_128 import verify as v, engine


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


def test_world_namespaces_disjoint_including_extension_and_assessment():
    seen = set()
    for stage, count in (("probe", 18), ("main", 360), ("test", 18)):
        for index in range(count):
            for block in v.BLOCKS:
                seeds = v._world_seeds(stage, index, block)
                assert seeds == engine.world_seeds(stage, index, block)
                assert not set(seeds) & seen
                seen.update(seeds)
    for length in (0, 32, 64, 96, 128):
        assert v._world_seeds("main", 0, "extension", length) == engine.world_seeds("main", 0, "extension")[:length]


@pytest.mark.parametrize("support,precision", [(1,128),(2,128),(3,128),(4,128),(5,160),(6,192),(7,224),(8,256),(50,256)])
def test_independent_budgets_exact_even_on_short_support(support, precision):
    budget = v.independent_budgets(support)
    assert budget == engine.budgets(support)
    assert budget["precision_worlds"] == precision
    assert budget["arm_continuations"]["precision"] == budget["arm_continuations"]["breadth"]


def worlds(values, count):
    return [{"outcomes": [{"wdl": value} for value in values]} for _ in range(count)]


def candidates(support):
    keys = list(range(min(support, 8)))
    return keys, {arm: {"keys": keys[:(len(keys) if arm == "breadth" else min(support, 4))]} for arm in v.ARMS}


@pytest.mark.parametrize("support", [1,2,3,4])
def test_zero_extension_all_arms_equal_and_tie_preserves_first(support):
    keys, arms = candidates(support)
    budget = v.independent_budgets(support)
    result = v._summary(worlds([0]*support,128),[],worlds([1],128),arms,keys,
                        dict.fromkeys(v.ARMS,3 if support == 4 else None),[0],budget,{"ranked":[],"random":[]})
    assert result["choices"] == dict.fromkeys(v.ARMS,0)
    assert result["arm_scores"] == dict.fromkeys(v.ARMS,1.)


@pytest.mark.parametrize("support", [5,6,7,8])
def test_precision_uses_weighted_full_world_sum_not_block_mean(support):
    keys, arms = candidates(support)
    budget = v.independent_budgets(support)
    selection = worlds([0]*support,128)
    # A beats B by64 total units in the common block. Extension B gains32/64/96/128.
    for row in selection[:64]:
        row["outcomes"][0]["wdl"] = 1
    extension = worlds([0,1,0,0],budget["extension_worlds"])
    means, choices = v._selection(selection,extension,arms,keys,budget)
    expected = 1 if support >= 7 else 0
    assert choices == {"anchor":0,"breadth":0,"precision":expected}
    assert means["precision"][0] == 64/budget["precision_worlds"]
    assert means["precision"][1] == budget["extension_worlds"]/budget["precision_worlds"]
    assert choices["precision"] == (0 if means["precision"][0] >= means["precision"][1] else 1)


def test_sparse_assessment_maps_union_ids_and_cannot_choose_another_action():
    keys, arms = candidates(8)
    budget = v.independent_budgets(8)
    selection = worlds([-1,-1,-1,0,0,1,-1,-1],128)
    extension = worlds([-1,-1,1,-1],128)
    proposed = dict.fromkeys(v.ARMS,3)
    means, choices = v._selection(selection,extension,arms,keys,budget)
    assert choices == {"anchor":3,"breadth":5,"precision":2}
    assessment = worlds([-1,0,1],128)
    result = v._summary(selection,extension,assessment,arms,keys,proposed,[2,3,5],budget,{"ranked":[4,5],"random":[6,7]})
    assert result["arm_scores"] == {"anchor":.5,"breadth":1.,"precision":0.}
    with pytest.raises(AssertionError):
        v._summary(selection,extension,assessment,arms,keys,proposed,[0,3,5],budget,{})


def statistic_rows():
    result=[]
    for i,spec in enumerate(synthetic_specs()):
        anchor=100+i % 53
        values={"anchor":anchor,"breadth":anchor+(i//18) % 11-3,"precision":anchor+(i//18) % 7-3}
        result.append({**spec,"arm_scores":{arm:value/256 for arm,value in values.items()}})
    return result


def test_bootstrap_matches_independent_indexed_resampling_and_rejects_bad_units():
    rows=statistic_rows();result=v.statistics(rows)
    matrices=[np.asarray([[r["arm_scores"][a]-r["arm_scores"][b] for _,a,b in v.CONTRASTS]
                          for r in rows if (r["fit"],r["seat"],r["phase"]) == cell]) for cell in v.CELLS]
    rng=np.random.default_rng(2103000000);samples=[]
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


@pytest.mark.parametrize("delta,precision_delta,passed",[(2,0,True),(1,0,False),(2,2,False),(0,0,False),(-2,0,False),(3,4,False)])
def test_gate_needs_practical_anchor_gain_and_breadth_precision_superiority(delta,precision_delta,passed):
    rows=[{**r,"arm_scores":{"anchor":.5,"breadth":(128+delta)/256,"precision":(128+precision_delta)/256}}
          for r in statistic_rows()]
    value=v.statistics(rows)
    assert value["gate_pass"] is passed
    assert value["classification"] == ("positive_conditional_breadth_contribution" if passed else "practical_progress_not_established")


def private_state():
    env=v.old.SplendorEnv(seed=2109800000);state=env.initial_state()
    for _ in range(2):
        action=next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state=env.step(state,action)
    for player in state.players:
        player.score=14;player.bonuses=dict.fromkeys(player.bonuses,5)
    return state


def support_state(support):
    state=v.old.SplendorEnv(seed=2109810000+support).initial_state()
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
        assert receipt["recorded_worlds"] == 256+v.independent_budgets(spec["canonical_support"])["extension_worlds"]
        for block in v.BLOCKS:
            assert receipt[f"recorded_{block}_worlds"] == receipt[f"replayed_{block}_worlds"]
            assert receipt[f"recorded_{block}_continuations"] == receipt[f"replayed_{block}_continuations"]


@pytest.mark.parametrize("damage",["none","model","model_hash","summary","assessment_indices","selection_indices",
                                   "extension_indices","extension_seed","extension_length","precision_budget","ranked_order",
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
    elif damage == "summary":root["summary"]["arm_scores"]["breadth"] += 1/256
    elif damage == "assessment_indices":root["assessment"]["candidate_indices"]=[]
    elif damage == "selection_indices":root["selection"]["candidate_indices"]=root["selection"]["candidate_indices"][1:]
    elif damage == "extension_indices":root["extension"]["candidate_indices"]=[]
    elif damage == "extension_seed":root["extension"]["world_seeds"][0] += 1
    elif damage == "extension_length":root["extension"]["worlds"].pop()
    elif damage == "precision_budget":root["budgets"]["precision_worlds"] -= 1
    elif damage == "ranked_order":root["union"]["keys"][4:6]=reversed(root["union"]["keys"][4:6])
    elif damage == "added_membership":root["added_indices"]["ranked"]=root["added_indices"]["random"]
    elif damage == "random_order":root["union"]["keys"][6:8]=reversed(root["union"]["keys"][6:8])
    elif damage == "world":root["assessment"]["worlds"][0]["state_sha256"]="0"*64
    elif damage == "outcome":root["assessment"]["worlds"][0]["outcomes"][0]["wdl"]=.5
    elif damage == "sampled_trace":root["assessment"]["worlds"][0]["outcomes"][0]["trace_sha256"]="0"*64
    elif damage == "seal_time":root["assessment_started_at_ns"]=root["selection_lock"]["created_at_ns"]-1
    elif damage == "sample_plan":
        value=copy.deepcopy(plan);value["selections"][0]["replay"]["extension"]=[]
        path=tmp_path/"test/audit_sample.json";path.write_text(json.dumps(value),encoding="utf-8")
    elif damage == "work":root["work"]["extension"]["continuations"] += 1
    path=tmp_path/"test/roots/state_0000.json";v.old.save_new(path,root)
    if damage == "none":assert v.c.verified(v.audit("test",spec,v.c.artifact(path),"fixture",tmp_path))["passed"]
    else:
        with pytest.raises((AssertionError,ValueError)):
            v.audit("test",spec,v.c.artifact(path),"fixture",tmp_path)


def test_fixture_override_isolated_from_production():
    with pytest.raises(AssertionError):v._source_state("test",{"fixture_snapshot":{}},v.OUTPUT)
    with pytest.raises(AssertionError):v._source_state("main",{"fixture_snapshot":{}},Path("elsewhere"))
