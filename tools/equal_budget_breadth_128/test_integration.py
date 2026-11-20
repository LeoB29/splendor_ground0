"""Two-block stage receipts and independent raw/receipt final statistics."""
import copy

import pytest

from tools.equal_budget_breadth_128 import common as c, experiment as p, engine, verify
from tools.equal_budget_breadth_128.test_verify import install_models, fixture_spec, support_state, synthetic_specs


@pytest.mark.parametrize("support,total", [(3,512),(5,768)])
def test_actual_ragged_engine_and_audit_pass_orchestrator_accounting(tmp_path, monkeypatch, support, total):
    install_models(tmp_path)
    spec = fixture_spec(support_state(support), 0)
    plan = verify.sample_plan([spec], "test", "fixture")
    c.old.save_new(tmp_path / "test/audit_sample.json", plan)
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_selection", lambda *args: {"test": [spec]})
    monkeypatch.setattr(p.models, "validate", lambda *args: None)
    calls = []

    def pool(stage, kind, arguments, binding, upstream):
        assert stage == "test" and binding == "fixture"
        calls.append(kind)
        if kind == "collection":
            items = [engine.collect(stage, spec, binding, tmp_path) for spec in arguments]
        else:
            items = [verify.audit(stage, spec, item, binding, tmp_path) for spec, item in arguments]
        c.old.save_new(tmp_path / f"{stage}_{kind}_lock.json", {"manifest": binding, "files": items})
        return items

    monkeypatch.setattr(p, "run_pool", pool)
    p.build_stage("test", "fixture")
    complete = c.old.read_json(tmp_path / "test_complete.json")
    coverage = complete["coverage"]
    assert calls == ["collection", "audits"] and complete["all_prescribed_audits_pass"]
    assert coverage["recorded_selection_worlds"] == coverage["replayed_selection_worlds"] == 128
    assert coverage["recorded_assessment_worlds"] == coverage["replayed_assessment_worlds"] == 128
    assert coverage["recorded_selection_continuations"] == support*128
    assert coverage["recorded_assessment_continuations"] == 128
    assert coverage["recorded_continuations"] == coverage["replayed_continuations"] == total
    assert coverage["recorded_worlds"] == coverage["replayed_worlds"] == 256


def synthetic_record(spec):
    """Hand-built complete outcomes, never production data or simulations."""
    index, support = spec["index"], spec["canonical_support"]
    budget = engine.budgets(support)
    m,n = budget["anchor_candidates"],budget["breadth_candidates"]
    keys = [f"action_{i}" for i in range(n)]
    actions = [{"synthetic_action":i} for i in range(n)]
    candidates = {arm:{"keys":keys[:m if arm == "anchor" else n],
                       "actions":actions[:m if arm == "anchor" else n],
                       "indices":list(range(m if arm == "anchor" else n))} for arm in engine.ARMS}
    allocation = engine.uniform_allocation(keys,budget,("main",index))
    proposed = dict.fromkeys(engine.ARMS,3 if m == 4 else None)
    added = {"ranked":list(range(m,min(m+2,n))),"random":list(range(min(m+2,n),n))}
    record = {"manifest":"binding","stage":"main","spec":spec,"snapshot":{"synthetic":index},
              "privacy":spec["privacy"],"proposals":{"anchor":{"synthetic":True}},
              "candidates":candidates,"union":{"keys":keys,"actions":actions},
              "proposed_indices":proposed,"added_indices":added,"budgets":budget,"uniform_allocation":allocation}
    def block(name,indices,outcome):
        seeds = engine.world_seeds("main",index,name)
        return {"candidate_indices":indices,"world_seeds":seeds,
                "worlds":[{"seed":seed,"outcomes":[{"wdl":outcome(w,i)} for i in indices]}
                          for w,seed in enumerate(seeds)],"timing":{"wall":1.,"cpu":1.}}
    L = max(allocation["world_counts"])
    def selection_value(world,action):
        if action == 0: return 0
        if n > 4 and action == 1: return 1 if world < 8 else -1 if world >= L else 0
        if n > 4 and action == n-1: return -1 if world < 8 else 1 if world >= L else 0
        return -1
    record["selection"] = block("selection",list(range(n)),selection_value)
    arithmetic,choices = engine.choose(record["selection"],candidates,budget,allocation)
    assert choices["anchor"] == 0
    if n > 4: assert choices["uniform"] == 1 and choices["teacher"] == n-1
    else: assert set(choices.values()) == {0}
    indices = sorted(set(choices.values()))
    anchor = 100+index%53
    scores = {"anchor":anchor,"uniform":anchor+(index//18)%11-3 if n>4 else anchor,
              "teacher":anchor+(index//18)%7-3 if n>4 else anchor}
    targets = {choice:scores[arm]-128 for arm,choice in choices.items()}
    def outcome(world,action):
        target = targets[action]
        return (1 if target>0 else -1) if world < abs(target) else 0
    record["assessment"] = block("assessment",indices,outcome)
    record["selection_lock"] = {"means":arithmetic["means"],"arithmetic":arithmetic,"choices":choices,
          "assessment_indices":indices,"identity_digest":c.old.digest(engine.proposal_identity(record)),
          "selection_sha256":c.old.digest(record["selection"])}
    record["summary"] = engine.summarize(record["assessment"]["worlds"],indices,choices,proposed,n)
    assert record["summary"]["arm_scores"] == {arm:value/256 for arm,value in scores.items()}
    return record


def final_fixture(tmp_path, monkeypatch, *, receipt_damage=False):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "REPORTER", None)
    monkeypatch.setattr(p, "validate_freeze", lambda: "binding")
    monkeypatch.setattr(p.models, "validate", lambda *args: None)
    specs = synthetic_specs()
    monkeypatch.setattr(p, "validate_selection", lambda *args: {"main": specs, "probe": []})
    plan = verify.sample_plan(specs, "main", "binding")
    c.old.save_new(tmp_path / "main/audit_sample.json", plan)
    sample_item = c.artifact(tmp_path / "main/audit_sample.json")
    roots, audits, expected_rows = [], [], []
    for spec,sample in zip(specs,plan["selections"],strict=True):
        index = spec["index"]
        record = synthetic_record(spec)
        selection_path = tmp_path / "main/selections" / f"state_{index:04d}_selection.json"
        c.old.save_new(selection_path, {"manifest": "binding", "stage": "main", "index": index,
                     "identity": engine.proposal_identity(record), "selection": record["selection"],
                     "selection_lock": record["selection_lock"]})
        record["selection_artifact"] = c.artifact(selection_path)
        path = tmp_path / "main/roots" / f"state_{index:04d}.json"
        c.old.save_new(path, record)
        row = engine.statistics_row(record)
        expected_rows.append(copy.deepcopy(row))
        if receipt_damage and index == 0:
            row["arm_scores"]["uniform"] += 1/256
        lengths,supports = p.geometry(spec["canonical_support"])
        supports["assessment"] = len(set(row["choices"].values()))
        counts = {}
        for kind in ("recorded", "replayed"):
            for block in p.BLOCKS:
                count = lengths[block] if kind == "recorded" else len(sample["replay"][block])
                counts[f"{kind}_{block}_worlds"] = count
                counts[f"{kind}_{block}_continuations"] = count*supports[block]
            for unit in ("worlds", "continuations"):
                counts[f"{kind}_{unit}"] = sum(counts[f"{kind}_{block}_{unit}"] for block in p.BLOCKS)
        item = c.artifact(path, index=index, continuations=counts["recorded_continuations"])
        roots.append(item)
        audit_path = tmp_path / "main/audits" / f"state_{index:04d}.json"
        c.old.save_new(audit_path, {"manifest": "binding", "passed": True, "spec": spec,
                     "root": {key:item[key] for key in ("path", "sha256")},
                     "selection_artifact": record["selection_artifact"], "sample_artifact": sample_item,
                     "row": row, "stress": sample["stress"], **counts})
        audits.append(c.artifact(audit_path,index=index))

    def locks(stage, kind, binding):
        assert binding == "binding"
        return {"files": [] if stage != "main" else roots if kind == "collection" else audits}

    monkeypatch.setattr(p, "stage_lock", locks)
    for name in ("models_lock.json", "selection_seal.json", "probe_complete.json", "main_complete.json"):
        c.old.save_new(tmp_path / name, {"synthetic_fixture": True})
    return expected_rows


def test_finish_reconciles_real_raw_world_statistics_with_receipt_statistics(tmp_path, monkeypatch):
    rows = final_fixture(tmp_path, monkeypatch)
    p.finish("binding")
    final = c.old.read_json(tmp_path / "final_status.json")
    result,audit = c.verified(final["result"]),c.verified(final["audit"])
    assert final["complete"] and final["all_prescribed_audits_pass"] and audit["passed"]
    assert not final["automatic_successor"] and result["rows"] == rows
    assert result["coverage"]["states"] == 360
    for name,left,right in verify.CONTRASTS:
        expected = sum(row["arm_scores"][left]-row["arm_scores"][right] for row in rows)/360
        assert result["statistics"]["contrasts"][name]["mean_score_difference"] == pytest.approx(expected,abs=1e-12)
        assert audit["independent_statistics"]["contrasts"][name]["ci95"] == pytest.approx(result["statistics"]["contrasts"][name]["ci95"],abs=1e-12)
    assert final["classification"] == result["statistics"]["classification"]


def test_rehashed_receipt_score_disagreement_blocks_final_result(tmp_path, monkeypatch):
    final_fixture(tmp_path, monkeypatch, receipt_damage=True)
    with pytest.raises(ValueError, match="Independent statistics disagree"):
        p.finish("binding")
    assert not (tmp_path / "result.json").exists() and not (tmp_path / "final_status.json").exists()


def test_late_budget_expiry_preserves_partial_result_without_completion(tmp_path, monkeypatch):
    final_fixture(tmp_path, monkeypatch)
    def boundary(*args):
        if (tmp_path / "report.md").exists():
            raise c.BudgetExceeded("Finalization exceeded its original budget")
    monkeypatch.setattr(p, "budget_boundary", boundary)
    with pytest.raises(c.BudgetExceeded):
        p.finish("binding")
    assert (tmp_path / "result.json").exists() and (tmp_path / "final_audit.json").exists()
    assert (tmp_path / "report.md").exists() and not (tmp_path / "final_status.json").exists()
