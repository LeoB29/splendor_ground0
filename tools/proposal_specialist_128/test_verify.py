"""Independent sparse-assessment arithmetic, replay panels and corruptions."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from tools.proposal_specialist_128 import verify as v
from tools.proposal_specialist_128 import engine


def synthetic_specs(count=360):
    specs = []
    for i in range(count):
        fit, seat, phase = v.CELLS[i % 18]
        specs.append({"index": i, "pair": 3*i, "game_index": 3*i,
                      "fit": fit, "seat": seat, "phase": phase,
                      "privacy": {"own": i % 2, "opponent": (i//2) % 2},
                      "canonical_support": 1+i % 5})
    return specs


def test_sample_plan_identity_only_complete_cells_and_probe_exhaustive():
    specs = synthetic_specs()
    plan = v.sample_plan(specs, "main", "binding")
    assert plan == v.sample_plan(copy.deepcopy(specs), "main", "binding")
    stress = {row["index"] for row in plan["selections"] if row["stress"]}
    assert 18 <= len(stress) <= 39
    assert not plan["coverage"]["missing_cells"]
    assert all(row["selected_index"] in stress for row in plan["stress_cells"])
    for row in plan["selections"]:
        for block in v.BLOCKS:
            assert len(row["ordinary"][block]) == 8
            assert len(row["replay"][block]) == (128 if row["stress"] else 8)
    altered = copy.deepcopy(specs)
    for spec in altered:
        spec["irrelevant_outcome"] = 12345
    other = v.sample_plan(altered, "main", "other binding")
    assert other["selections"] == plan["selections"]
    assert other["stress_cells"] == plan["stress_cells"]
    probe = v.sample_plan(synthetic_specs(18), "probe", "binding")
    assert all(row["stress"] for row in probe["selections"])
    assert probe["coverage"]["replayed_worlds"] == 18*256


def test_independent_world_namespaces_global_disjoint():
    seen = set()
    for stage, count in (("probe", 18), ("main", 360), ("test", 18)):
        for index in range(count):
            for block in v.BLOCKS:
                seeds = v._world_seeds(stage, index, block)
                assert seeds == engine.world_seeds(stage, index, block)
                assert not (set(seeds) & seen)
                seen.update(seeds)


@pytest.mark.parametrize("size", [1, 2, 3])
def test_short_support_independent_arithmetic(size):
    keys = list("abc")[:size]
    candidates = {arm: {"keys": list(keys)} for arm in v.ARMS}
    selection = [{"outcomes": [{"wdl": 0} for _ in keys]} for _ in range(128)]
    assessment = [{"outcomes": [{"wdl": 1}]} for _ in range(128)]
    result = v._summary(selection, assessment, candidates, keys, dict.fromkeys(v.ARMS), [0])
    assert result["choices"] == dict.fromkeys(v.ARMS, 0)
    assert result["arm_scores"] == dict.fromkeys(v.ARMS, 1.)
    assert result["assessment_indices"] == [0]
    assert result["proposed_indices"] == dict.fromkeys(v.ARMS)


@pytest.mark.parametrize("chosen", [[3, 3, 3], [3, 4, 3], [3, 4, 5]])
def test_sparse_union_indices_independent_from_column_offsets(chosen):
    keys = list("abcdef")
    candidates = {arm: {"keys": list("abc")+[keys[j]]} for arm, j in zip(v.ARMS, chosen)}
    selection = [{"outcomes": [{"wdl": value} for value in [-1, -1, -1, 1, 1, 1]]} for _ in range(128)]
    indices = sorted(set(chosen))
    by_index = {3: -1, 4: 0, 5: 1}
    assessment = [{"outcomes": [{"wdl": by_index[j]} for j in indices]} for _ in range(128)]
    result = v._summary(selection, assessment, candidates, keys, dict(zip(v.ARMS, chosen)), indices)
    assert result["choices"] == dict(zip(v.ARMS, chosen))
    assert result["arm_scores"] == {arm: (1+by_index[j])/2 for arm,j in zip(v.ARMS, chosen)}
    with pytest.raises(AssertionError):
        v._summary(selection, assessment, candidates, keys, dict(zip(v.ARMS, chosen)), indices+[0])


def statistic_rows():
    rows = []
    for i, spec in enumerate(synthetic_specs()):
        anchor = 100+i % 53
        values = {"anchor": anchor, "specialist": anchor+(i//18) % 11-3, "control": anchor+(i//18) % 7-3}
        rows.append({**spec, "arm_scores": {arm: value/256 for arm, value in values.items()}})
    return rows


def test_bootstrap_matches_separate_indexed_resampling():
    rows = statistic_rows()
    result = v.statistics(rows)
    matrices = [np.asarray([[r["arm_scores"][a]-r["arm_scores"][b] for _,a,b in v.CONTRASTS]
                            for r in rows if (r["fit"],r["seat"],r["phase"]) == cell]) for cell in v.CELLS]
    rng = np.random.default_rng(2133000000)
    samples = []
    for _ in range(50):
        total = np.zeros((1000,len(v.CONTRASTS)))
        for matrix in matrices:
            total += matrix[rng.integers(0,20,size=(1000,20))].sum(axis=1)
        samples.append(total/360)
    boot = np.concatenate(samples)
    for j,(name,_,_) in enumerate(v.CONTRASTS):
        assert result["contrasts"][name]["mean_score_difference"] == pytest.approx(sum(m[:,j].sum() for m in matrices)/360, abs=1e-12)
        assert result["contrasts"][name]["ci95"] == pytest.approx(np.quantile(boot[:,j],[.025,.975]), abs=1e-12)
    assert result["gate_pass"] == (result["anchor_practical_gain"] and result["anchor_superiority"] and result["control_superiority"])
    for damage in ("duplicate", "stratum", "score"):
        bad = copy.deepcopy(rows)
        if damage == "duplicate":
            bad[1]["game_index"] = bad[1]["pair"] = bad[0]["game_index"]
        elif damage == "stratum":
            bad[0]["seat"] = 1
        else:
            bad[0]["arm_scores"]["anchor"] = .51
        with pytest.raises(AssertionError):
            v.statistics(bad)


@pytest.mark.parametrize("delta,ce_delta,passed", [(2,0,True),(1,0,False),(2,2,False),(0,0,False),(-2,0,False)])
def test_gate_requires_practical_anchor_gain_and_both_superiorities(delta, ce_delta, passed):
    rows = [{**r, "arm_scores": {"anchor": .5, "specialist": (128+delta)/256, "control": (128+ce_delta)/256}}
            for r in statistic_rows()]
    value = v.statistics(rows)
    assert value["gate_pass"] is passed
    assert value["classification"] == ("positive_conditional_specialist_contribution" if passed else "practical_progress_not_established")


def private_state():
    env = v.old.SplendorEnv(seed=2139800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = dict.fromkeys(player.bonuses, 5)
    return state


def short_state(support):
    state = v.old.SplendorEnv(seed=2139810000+support).initial_state()
    cards = [card for tier in state.visible_tier_cards for card in state.visible_tier_cards[tier]+state.hidden_tier_decks[tier]]
    reserved = sorted([card for card in cards if card.points > 0], key=lambda card: card.card_id)[:support]
    state.visible_tier_cards = {tier: [] for tier in state.visible_tier_cards}
    state.hidden_tier_decks = {tier: [] for tier in state.hidden_tier_decks}
    state.deck_counts = dict.fromkeys(state.deck_counts, 0)
    state.bank_tokens = dict.fromkeys(state.bank_tokens, 0)
    state.nobles = []
    for player in state.players:
        player.score = 14
        player.bonuses = dict.fromkeys(player.bonuses, 10)
    state.players[0].reserved_cards = reserved
    state.players[0].reserved_card_sources = ["deck"]*support
    state.players[1].purchased_cards = [card for card in cards if card not in reserved]
    return state


def install_models(base):
    parent = v.old.read_json(v.c.ROOT/"outputs/proposal_iteration_128/models_lock.json")
    mapped = {"anchor":"anchor", "specialist":"treatment", "control":"replay"}
    registry = {"manifest":"fixture", "models": {f"{arm}_{fit}": parent["models"][f"{mapped[arm]}_{fit}"]
                                                  for arm in v.ARMS for fit in v.FITS}}
    v.old.save_new(base/"models_lock.json", registry)
    engine.initialize_models(base)


def fixture_spec(state, index):
    snapshot = json.loads(json.dumps(v.old.serialize_state_snapshot(state)))
    legal = v.old.SplendorEnv().legal_actions(state)
    return {"index":index,"pair":index,"game_index":index,"fit":161001,"seat":state.current_player,
            "phase":min(state.turn_index//20,2),"turn":state.turn_index,"privacy":engine.baseline.privacy(state),
            "canonical_support":len({repr(v.old.canonical_successor_action_key(state,a)) for a in legal}),
            "source_stage":"test","search_stream":"independent-specialist-fixture",
            "snapshot_sha256":v.old.digest(snapshot),"fixture_snapshot":snapshot}


@pytest.fixture(scope="module")
def fixture_artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("specialist-verifier")
    install_models(base)
    specs = [fixture_spec(private_state(),i) for i in range(2)]
    specs += [fixture_spec(short_state(n),n+1) for n in (1,2,3)]
    assert [spec["canonical_support"] for spec in specs[2:]] == [1,2,3]
    sample = v.sample_plan(specs,"test","fixture")
    v.old.save_new(base/"test"/"audit_sample.json",sample)
    items = [engine.collect("test",spec,"fixture",base) for spec in specs]
    return base,specs,sample,items


def test_private_and_short_support_fixtures_replay_exact_sparse_budgets(fixture_artifacts):
    base,specs,plan,items = fixture_artifacts
    assert any(not r["stress"] for r in plan["selections"])
    for spec,sample,item in zip(specs,plan["selections"],items,strict=True):
        receipt = v.c.verified(v.audit("test",spec,item,"fixture",base))
        assert receipt["passed"] and receipt["row"] == engine.statistics_row(v.c.verified(item))
        assert receipt["recorded_worlds"] == 256
        assert receipt["replayed_worlds"] == (256 if sample["stress"] else 16)
        assert receipt["recorded_continuations"] == 128*(receipt["support_selection"]+receipt["support_assessment"])
        assert receipt["recorded_continuations"] == receipt["recorded_selection_continuations"]+receipt["recorded_assessment_continuations"]
        assert receipt["replayed_continuations"] == receipt["replayed_selection_continuations"]+receipt["replayed_assessment_continuations"]
        if spec["canonical_support"] <= 3:
            assert receipt["support_assessment"] == 1
            assert receipt["row"]["proposed_indices"] == dict.fromkeys(v.ARMS)


def clone_fixture(fixture_artifacts, tmp_path):
    base,specs,plan,items = fixture_artifacts
    index = next(row["index"] for row in plan["selections"] if not row["stress"])
    spec,root = specs[index],copy.deepcopy(v.c.verified(items[index]))
    v.old.save_new(tmp_path/"models_lock.json",v.old.read_json(base/"models_lock.json"))
    v.old.save_new(tmp_path/"test"/"audit_sample.json",plan)
    seal_path = tmp_path/"test"/"selections"/f"state_{index:04d}_selection.json"
    seal = copy.deepcopy(v.c.verified(root["selection_artifact"]))
    seal["selection_lock"]["created_at_ns"] = time.time_ns()
    root["selection_lock"] = seal["selection_lock"]
    v.old.save_new(seal_path,seal)
    root["selection_artifact"] = v.c.artifact(seal_path)
    root["assessment_started_at_ns"] = time.time_ns()
    return spec,root


@pytest.mark.parametrize("damage", ["none","model","model_hash","summary","assessment_indices","selection_indices","schedule","world","outcome","sampled_trace","seal_time","sample_plan"])
def test_rehashed_corruptions_and_pristine_clone(fixture_artifacts,tmp_path,damage):
    spec,root = clone_fixture(fixture_artifacts,tmp_path)
    index,block = spec["index"],"assessment"
    wi = v.ordinary_indices("test",index,block)[0]
    if damage == "model":
        root["proposals"]["specialist"]["model_id"] = "control_161001"
    elif damage == "model_hash":
        root["proposals"]["specialist"]["model_sha256"] = "0"*64
    elif damage == "summary":
        root["summary"]["arm_scores"]["specialist"] += 1/256
    elif damage == "assessment_indices":
        root[block]["candidate_indices"] = []
    elif damage == "selection_indices":
        root["selection"]["candidate_indices"] = root["selection"]["candidate_indices"][1:]
    elif damage == "schedule":
        root[block]["world_seeds"][0] += 1
    elif damage == "world":
        root[block]["worlds"][0]["state_sha256"] = "0"*64
    elif damage == "outcome":
        wi = next(i for i in range(128) if i not in v.ordinary_indices("test",index,block))
        root[block]["worlds"][wi]["outcomes"][0]["wdl"] = .5
    elif damage == "sampled_trace":
        root[block]["worlds"][wi]["outcomes"][0]["trace_sha256"] = "0"*64
    elif damage == "seal_time":
        root["assessment_started_at_ns"] = root["selection_lock"]["created_at_ns"]-1
    elif damage == "sample_plan":
        path = tmp_path/"test"/"audit_sample.json"
        value = v.old.read_json(path)
        value["selections"][index]["replay"][block] = []
        path.write_text(json.dumps(value),encoding="utf-8")
    path = tmp_path/"test"/"roots"/f"state_{index:04d}.json"
    v.old.save_new(path,root)
    if damage == "none":
        assert v.c.verified(v.audit("test",spec,v.c.artifact(path),"fixture",tmp_path))["passed"]
    else:
        with pytest.raises((AssertionError,ValueError)):
            v.audit("test",spec,v.c.artifact(path),"fixture",tmp_path)


def test_fixture_override_isolated_from_production():
    with pytest.raises(AssertionError):
        v._source_state("test",{"fixture_snapshot":{}},v.OUTPUT)
    with pytest.raises(AssertionError):
        v._source_state("main",{"fixture_snapshot":{}},Path("elsewhere"))
