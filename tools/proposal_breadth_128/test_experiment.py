"""Independent statistics, ragged resource accounting and strict stage gates."""
from contextlib import contextmanager
import copy
from types import SimpleNamespace

import pytest

from tools.proposal_breadth_128 import common as c, experiment as p, verify


def rows():
    result = []
    for fit in c.FITS:
        for seat in (0,1):
            for phase in (0,1,2):
                for k in range(20):
                    i = len(result)
                    result.append({"index":i,"pair":i,"game_index":i,"fit":fit,"seat":seat,"phase":phase,
                        "arm_scores":{"anchor":.5+(k%3-1)/256,
                                      "breadth":.5+(k%9-2)/256,
                                      "precision":.5+(k%4-1)/256}})
    return result


def test_nonconstant_quantized_statistics_agree_with_independent_verifier():
    values = rows()
    expected,actual = verify.statistics(values),p.independent_statistics(values)
    for name in p.CONTRASTS:
        assert actual["contrasts"][name]["mean"] == expected["contrasts"][name]["mean_score_difference"]
        assert actual["contrasts"][name]["ci95"] == pytest.approx(expected["contrasts"][name]["ci95"],abs=1e-12)
    for key in ("anchor_practical_gain","anchor_superiority","precision_superiority","gate_pass","classification"):
        assert actual[key] == expected[key]
    # Raw-root order is irrelevant to the independently stratified reconstruction.
    assert p.independent_statistics(list(reversed(values))) == actual


@pytest.mark.parametrize("breadth,precision,expected",[
    (0,0,(False,False,False,False)),
    (1,0,(False,True,True,False)),
    (2,2,(True,True,False,False)),
    (2,1,(True,True,True,True)),
    (2,3,(True,True,False,False)),
])
def test_practical_and_precision_gates_are_both_required(breadth,precision,expected):
    values = rows()
    for row in values:
        row["arm_scores"] = {"anchor":.5,"breadth":.5+breadth/256,"precision":.5+precision/256}
    result = p.independent_statistics(values)
    assert tuple(result[k] for k in ("anchor_practical_gain","anchor_superiority","precision_superiority","gate_pass")) == expected
    assert result["classification"] == ("positive_conditional_breadth_contribution" if expected[-1] else "practical_progress_not_established")


@pytest.mark.parametrize("extra,passes",[(100,False),(101,True)])
def test_half_percentage_point_gate_uses_exact_aggregate_units(extra,passes):
    values = rows()
    for row in values:
        gain = 1+int(row["index"] < extra)
        row["arm_scores"] = {"anchor":.5,"breadth":.5+gain/256,"precision":.5}
    result = p.independent_statistics(values)
    assert result["contrasts"]["breadth_minus_anchor"]["mean"] == (360+extra)/(360*256)
    assert result["anchor_practical_gain"] is passes
    assert result["anchor_superiority"] and result["precision_superiority"]


@pytest.mark.parametrize("damage",["duplicate_game","missing_row","changed_cell","wrong_units","extra_arm"])
def test_statistics_reject_invalid_population_or_score_scale(damage):
    values = rows()
    if damage == "duplicate_game":
        values[0]["game_index"] = values[1]["game_index"]
    elif damage == "missing_row":
        values.pop()
    elif damage == "changed_cell":
        values[0]["phase"] = 1
    elif damage == "wrong_units":
        values[0]["arm_scores"]["breadth"] = .501
    else:
        values[0]["arm_scores"]["random"] = .5
    with pytest.raises(ValueError):
        p.independent_statistics(values)


def cost_arguments():
    counts = {"selection":18432,"extension":9216,"assessment":6912}
    return {"collection":{"selection":184.32,"extension":184.32,"assessment":207.36},
            "replay":{"selection":147.456,"extension":147.456,"assessment":165.888},
            "probe_counts":counts,
            "main_counts":{"selection":368640,"extension":184320,"assessment":138240},
            "main_replays":{"selection":69120,"extension":34560,"assessment":25920},
            "collection_structural":100.,"audit_structural":200.,
            "size_bytes":20*1024**2,"free_bytes":100*1024**3,"elapsed_seconds":1800.}


def test_forecast_accounts_for_all_blocks_structure_and_total_elapsed_execution():
    args = cost_arguments()
    value = p.cost_projection(**args)
    assert value["reasonable"]
    expected_collection = (368640*.01+184320*.02+138240*.03+20*100)/8
    expected_replay = (69120*.008+34560*.016+25920*.024+20*200)/8
    assert value["collection_seconds_before_allowance"] == pytest.approx(expected_collection)
    assert value["audit_seconds_before_allowance"] == pytest.approx(expected_replay)
    assert value["remaining_seconds"] == pytest.approx(1.35*(expected_collection+expected_replay)+300)
    for block in p.BLOCKS:
        changed = copy.deepcopy(args)
        changed["collection"][block] *= 2
        assert p.cost_projection(**changed)["remaining_seconds"] > value["remaining_seconds"]
        changed = copy.deepcopy(args)
        changed["replay"][block] *= 2
        assert p.cost_projection(**changed)["remaining_seconds"] > value["remaining_seconds"]
    changed = dict(args,elapsed_seconds=8*3600-1)
    assert not p.cost_projection(**changed)["reasonable"]
    changed = dict(args,free_bytes=value["projected_bytes"]+c.CONFIG["disk_reserve_bytes"]-1)
    assert not p.cost_projection(**changed)["reasonable"]
    changed["free_bytes"] += 1
    assert p.cost_projection(**changed)["reasonable"]


def test_zero_extension_is_valid_only_if_main_also_requires_none():
    args = cost_arguments()
    for key in ("collection","replay","probe_counts","main_counts","main_replays"):
        args[key]["extension"] = 0
    value = p.cost_projection(**args)
    assert value["reasonable"] and value["collection_seconds_per_continuation"]["extension"] == 0
    args["main_counts"]["extension"] = 4*32
    held = p.cost_projection(**args)
    assert held["reasonable"] is False and held["missing_blocks"] == ["extension"]
    assert held["remaining_seconds"] is None


def test_ragged_extension_projection_uses_actual_continuation_counts():
    args = cost_arguments()
    # Three measured short-support roots require32,64,96 extra worlds,4 actions each.
    probe_extension = 4*(32+64+96)
    args["probe_counts"]["extension"] = probe_extension
    args["collection"]["extension"] = probe_extension*.02
    args["replay"]["extension"] = probe_extension*.016
    args["main_counts"]["extension"] = probe_extension*20
    args["main_replays"]["extension"] = 4*(2+4+6)*20
    value = p.cost_projection(**args)
    assert value["collection_seconds_per_continuation"]["extension"] == pytest.approx(.02)
    assert value["replay_seconds_per_continuation"]["extension"] == pytest.approx(.016)
    assert value["main_continuations_upper_bound"]["extension"] == 15360
    assert value["main_replay_continuations_upper_bound"]["extension"] == 960


@pytest.mark.parametrize("damage",["negative","nan","missing_block"])
def test_malformed_probe_cost_evidence_cannot_authorize_main(damage):
    args = cost_arguments()
    if damage == "negative":
        args["audit_structural"] = -1.
    elif damage == "nan":
        args["collection"]["extension"] = float("nan")
    else:
        args["replay"].pop("extension")
    with pytest.raises(ValueError):
        p.cost_projection(**args)


def coverage_fixture(support=5,stress=False):
    m,n = min(support,4),min(support,8)
    lengths = {"selection":128,"extension":128*n//m-128,"assessment":128}
    supports = {"selection":n,"extension":m,"assessment":min(n,3)}
    spec = {"index":0,"canonical_support":support}
    ordinary = {b:verify.ordinary_indices("test",0,b,lengths[b]) for b in p.BLOCKS}
    replay = {b:list(range(lengths[b])) if stress else ordinary[b] for b in p.BLOCKS}
    sample = {"stress":stress,"ordinary":ordinary,"replay":replay}
    artifact = {"path":"fixture/sample.json","sha256":"sample"}
    root = {"path":"fixture/root.json","sha256":"root","index":0,
            "continuations":sum(lengths[b]*supports[b] for b in p.BLOCKS)}
    receipt = {"manifest":"fixture","passed":True,"spec":spec,
               "root":{k:root[k] for k in ("path","sha256")},"sample_artifact":artifact,
               "stress":stress,"ordinary_world_indices":ordinary,"replay_world_indices":replay,
               "full_inner_replay":stress}
    for b in p.BLOCKS:
        receipt[f"support_{b}"] = supports[b]
        for kind,count in (("recorded",lengths[b]),("replayed",len(replay[b]))):
            receipt[f"{kind}_{b}_worlds"] = count
            receipt[f"{kind}_{b}_continuations"] = count*supports[b]
    for kind in ("recorded","replayed"):
        for unit in ("worlds","continuations"):
            receipt[f"{kind}_{unit}"] = sum(receipt[f"{kind}_{b}_{unit}"] for b in p.BLOCKS)
    return receipt,spec,root,sample,artifact,"fixture"


@pytest.mark.parametrize("support",[1,2,3,4,5,6,7,8,15])
def test_geometry_and_receipt_reconciliation_cover_short_and_ragged_support(support):
    lengths,counts = p.geometry(support)
    assert counts["extension"]*(128+lengths["extension"]) == counts["selection"]*128
    for stress in (False,True):
        p.validate_coverage(*coverage_fixture(support,stress))


@pytest.mark.parametrize("damage",["link","sample","block_worlds","block_work","support","total","root_work","replay_disclosure"])
def test_corrupt_receipt_cannot_publish_stage_completion(damage):
    args = copy.deepcopy(coverage_fixture(5))
    receipt = args[0]
    if damage == "link":
        receipt["root"]["sha256"] = "wrong"
    elif damage == "sample":
        receipt["replay_world_indices"] = {**receipt["replay_world_indices"],"extension":[0]}
    elif damage == "block_worlds":
        receipt["recorded_extension_worlds"] -= 1
    elif damage == "block_work":
        receipt["replayed_extension_continuations"] += 4
    elif damage == "support":
        receipt["support_assessment"] = 4
    elif damage == "total":
        receipt["recorded_continuations"] += 1
    elif damage == "root_work":
        args[2]["continuations"] += 1
    else:
        receipt["full_inner_replay"] = True
    with pytest.raises(ValueError):
        p.validate_coverage(*args)


def fake_run(tmp_path,monkeypatch,gate=True):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setattr(p,"REPORTER",None)
    monkeypatch.setattr(p.windows,"assert_windowless_worker",lambda *a:{})
    monkeypatch.setattr(p,"validate_freeze",lambda:"fixture")
    monkeypatch.setattr(p,"budget_boundary",lambda *a:None)
    @contextmanager
    def awake():
        yield {"fixture":True}
    monkeypatch.setattr(p.parent.parent,"inhibit_system_sleep",awake)
    calls = []
    monkeypatch.setattr(p.models,"prepare",lambda *a:calls.append("models"))
    monkeypatch.setattr(p,"build_sources",lambda *a:calls.append("sources"))
    monkeypatch.setattr(p,"build_stage",lambda stage,*a:calls.append(stage))
    monkeypatch.setattr(p,"operational_decision",lambda *a:gate)
    monkeypatch.setattr(p,"finish",lambda *a:calls.append("finish"))
    return calls


@pytest.mark.parametrize("gate,expected",[(True,["models","sources","probe","main","finish"]),
                                         (False,["models","sources","probe"])])
def test_pipeline_respects_operational_gate_and_has_no_training_or_successor(tmp_path,monkeypatch,gate,expected):
    calls = fake_run(tmp_path,monkeypatch,gate)
    p.run()
    assert calls == expected and not (tmp_path/"failure.json").exists()
    with pytest.raises(ValueError,match="One-shot"):
        p.run()


def test_resource_shortage_preserves_hold_and_stops_before_probe(tmp_path,monkeypatch):
    calls = fake_run(tmp_path,monkeypatch)
    def shortage(*a):
        p.old.save_new(tmp_path/"retained_partial.json",{"preserved":True})
        raise c.BudgetExceeded("Fixed source pool short; no extension")
    monkeypatch.setattr(p,"build_sources",shortage)
    p.run()
    assert calls == ["models"] and (tmp_path/"retained_partial.json").exists()
    assert "source pool short" in p.old.read_json(tmp_path/"operational_hold.json")["reason"]
    assert not (tmp_path/"failure.json").exists() and not (tmp_path/"final_status.json").exists()


def test_budget_boundary_enforces_elapsed_time_and_disk_reserve(tmp_path,monkeypatch):
    c.old.save_new(tmp_path/"launch.json",{"epoch_seconds":1.})
    monkeypatch.setattr(c.time,"time",lambda:8*3600+2.)
    with pytest.raises(c.BudgetExceeded,match="Eight-hour"):
        c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time,"time",lambda:2.)
    monkeypatch.setattr(c.shutil,"disk_usage",lambda *a:SimpleNamespace(free=c.CONFIG["disk_reserve_bytes"]-1))
    with pytest.raises(c.BudgetExceeded,match="disk reserve"):
        c.budget_boundary(tmp_path)


def test_status_permission_error_is_advisory_but_scientific_write_remains_strict(tmp_path,monkeypatch):
    from tools.proposal_ranking_128_recovery_01 import reporting
    reporter = p.StatusReporter(tmp_path)
    monkeypatch.setattr(p,"REPORTER",reporter)
    def locked(*a,**kw):
        raise PermissionError("fixture Windows replacement lock")
    monkeypatch.setattr(reporting.os,"replace",locked)
    monkeypatch.setattr(reporting.time,"sleep",lambda *a:None)
    value = p.event("fixture_progress",completed=1)
    assert value["stage"] == "fixture_progress"
    assert any(e["operation"] == "progress_replace" for e in reporter.last_errors)
    assert list(tmp_path.glob("status_events/*.json"))
    with pytest.raises(PermissionError,match="Windows replacement lock"):
        p.old.save_new(tmp_path/"scientific_artifact.json",{"must_be_durable":True})
    assert not (tmp_path/"scientific_artifact.json").exists()
