"""Cross-module finalization checks with an independently specified score law."""
import copy

import pytest

from . import common as c, engine, experiment as p, verify


def _outcome(value):
    return {"wdl":value,"reason":"completed","winner":0 if value>0 else 1 if value<0 else None}


def _root(index, control_sum, continued_sum):
    fit,seat,phase = verify.CELLS[index//40]
    spec = {"index":index,"game_index":index,"pair":index,"fit":fit,"seat":seat,"phase":phase}
    choices = dict(anchor=3,control=4,continued=5,teacher=9)
    candidates = {arm:{"indices":[0,1,2,j]} for arm,j in choices.items() if arm != "teacher"}
    candidates["teacher"] = {"indices":[0,1,2,3,6,7,8,9]}
    record = {"stage":"main","spec":spec,"privacy":{"own":0,"opponent":0},"snapshot":{},
              "proposals":dict.fromkeys(c.MODEL_ARMS,{}),"candidates":candidates,
              "union":{"keys":[str(i) for i in range(10)]},"proposed_indices":choices,
              "teacher_added_indices":{"ranked":[6,7],"random":[8,9]}}
    seeds = engine.world_seeds("main",index,"selection")
    outcomes = [_outcome(v) for v in [-1,-1,-1,0,1,1,-1,-1,-1,1]]
    record["selection"] = {"world_seeds":seeds,"candidate_indices":list(range(10)),
        "worlds":[{"seed":seed,"outcomes":outcomes} for seed in seeds]}
    selected = engine.choose(record["selection"]["worlds"],candidates,10)
    assert selected["choices"] == choices
    seeds = engine.world_seeds("main",index,"assessment")
    sums = [0,control_sum,continued_sum,8]
    worlds = [{"seed":seed,"outcomes":[_outcome((1 if total>0 else -1) if i<abs(total) else 0)
               for total in sums]} for i,seed in enumerate(seeds)]
    indices = [3,4,5,9]
    record["assessment"] = {"world_seeds":seeds,"candidate_indices":indices,"worlds":worlds}
    record["selection_lock"] = {"selection_statistics":selected,"choices":choices,
        "identity_digest":c.old.digest(engine.proposal_identity(record)),
        "selection_sha256":c.old.digest(record["selection"]),"proofs_sha256":c.old.digest([]),
        "assessment_indices":indices}
    record["summary"] = engine.summarize(worlds,indices,choices,choices,10)
    return record


@pytest.mark.parametrize("control_sum,continued_sum,passes",[(2,4,True),(6,4,False),(0,1,False)])
def test_finalization_requires_both_practical_gain_and_improvement_over_saved_model(
        tmp_path,monkeypatch,control_sum,continued_sum,passes):
    """Real reductions, independent bootstraps and publication must agree.

    Only upstream file authentication/collection are substituted. The score
    outcomes, root-to-row reductions and both independent statistics engines run.
    The last two cases reject a better anchor score when the saved model is
    stronger, and a certain but insufficiently large improvement respectively.
    """
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setitem(c.CONFIG,"bootstrap_samples",211)
    monkeypatch.setitem(c.CONFIG,"bootstrap_chunk",47)
    monkeypatch.setattr(p,"validate_freeze",lambda **kw:"fixture")
    monkeypatch.setattr(p.learning,"validate_models",lambda *a:None)
    monkeypatch.setattr(p,"event",lambda *a,**kw:None)
    specs,rows = [],[]
    for index in range(720):
        root = _root(index,control_sum,continued_sum)
        specs.append(root["spec"])
        rows.append(engine.statistics_row(root))
    summary = {"manifest":"fixture","rows":rows,"coverage":{"states":720},"full_inner_replay":False}
    monkeypatch.setattr(p,"validate_selection",lambda *a:{"main":specs})
    monkeypatch.setattr(p,"stage_lock",lambda stage,kind,binding:{
        "files":[{"index":i} for i in range(720)] if kind=="collection" else []})
    monkeypatch.setattr(p,"verified",lambda ref:_root(ref["index"],control_sum,continued_sum))
    monkeypatch.setattr(p.verify,"summarize",lambda *a,**kw:copy.deepcopy(summary))
    for name in ("learning_complete.json","selection_seal.json","probe_complete.json","main_complete.json"):
        c.old.save_new(tmp_path/name,{"manifest":"fixture","complete":True})
    p.finish("fixture")
    status = c.old.read_json(tmp_path/"final_status.json")
    result,audit = c.verified(status["result"]),c.verified(status["audit"])
    c.verify_hash(status["report"])
    actual = result["statistics"]
    assert status["complete"] and status["all_prescribed_audits_pass"] and audit["passed"]
    assert actual["gate_pass"] is passes
    assert actual["anchor_practical_gain"] is (continued_sum >= 2)
    assert actual["anchor_superiority"] is True
    assert actual["control_superiority"] is (continued_sum>control_sum)
    assert actual["dose_attribution"] is (continued_sum>control_sum)
    assert actual["contrasts"]["continued_minus_anchor"]["mean_score_difference"] == continued_sum/256
    assert actual["contrasts"]["continued_minus_control"]["mean_score_difference"] == (continued_sum-control_sum)/256
    assert set(actual["contrasts"]) == set(c.CONTRASTS)
    assert "coverage_attribution" not in actual and "precision_attribution" not in actual
    assert status["automatic_successor"] is False
    before = c.artifact(tmp_path/"result.json")
    with pytest.raises(FileExistsError):p.finish("fixture")
    c.verify_hash(before)
