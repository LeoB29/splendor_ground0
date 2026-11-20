"""Cross-component receipt and finalization tests without production work."""
import copy
import torch

import pytest

from tools.proposal_ranking_128 import common as c, experiment as p, engine, verify
from tools.proposal_ranking_128.test_verify import install_models, fixture_spec, short_state
from tools.proposal_ranking_128.test_experiment import rows


def test_actual_sparse_engine_receipt_passes_orchestrator_accounting(tmp_path, monkeypatch):
    install_models(tmp_path)
    spec = fixture_spec(short_state(3), 0)
    plan = verify.sample_plan([spec], "test", "fixture")
    c.old.save_new(tmp_path/"test/audit_sample.json", plan)
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_selection", lambda *a: {"test": [spec]})
    monkeypatch.setattr(p.learning, "validate_models", lambda *a: None)
    calls = []
    def pool(stage, kind, arguments, binding, upstream):
        assert stage == "test" and binding == "fixture"
        calls.append(kind)
        if kind == "collection":
            result = [engine.collect(stage, s, binding, tmp_path) for s in arguments]
        else:
            result = [verify.audit(stage, s, item, binding, tmp_path) for s, item in arguments]
        c.old.save_new(tmp_path/f"{stage}_{kind}_lock.json", {"manifest": binding, "files": result})
        return result
    monkeypatch.setattr(p, "run_pool", pool)
    p.build_stage("test", "fixture")
    value = c.old.read_json(tmp_path/"test_complete.json")
    assert calls == ["collection", "audits"] and value["all_prescribed_audits_pass"]
    assert value["coverage"]["recorded_selection_continuations"] == 3*128
    assert value["coverage"]["recorded_assessment_continuations"] == 128
    assert value["coverage"]["recorded_continuations"] == value["coverage"]["replayed_continuations"] == 512


def final_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_freeze", lambda: "binding")
    monkeypatch.setattr(p.learning, "validate_models", lambda *a: None)
    monkeypatch.setattr(p, "validate_selection", lambda *a: {"main": [], "probe": []})
    values = rows()
    roots, audits = [], []
    c.old.save_new(tmp_path/"choice.json", {"sealed": True})
    for row in values:
        path = tmp_path/"fixture_roots"/f"{row['index']:04d}.json"
        c.old.save_new(path, {"row": row, "selection_artifact": c.artifact(tmp_path/"choice.json")})
        item = c.artifact(path, index=row["index"], continuations=512)
        roots.append(item)
        ap = tmp_path/"fixture_audits"/path.name
        c.old.save_new(ap, {"manifest": "binding", "passed": True,
                           "root": {k: item[k] for k in ("path", "sha256")}})
        audits.append(c.artifact(ap, index=row["index"]))
    def locks(stage, kind, binding):
        assert binding == "binding"
        return {"files": [] if stage != "main" else roots if kind == "collection" else audits}
    monkeypatch.setattr(p, "stage_lock", locks)
    monkeypatch.setattr(p.engine, "statistics_row", lambda record: record["row"])
    stats = verify.statistics(values)
    monkeypatch.setattr(p.verify, "summarize", lambda *a: {"statistics": copy.deepcopy(stats), "rows": values})
    for name in ("learning_complete.json", "selection_seal.json", "probe_complete.json", "main_complete.json"):
        c.old.save_new(tmp_path/name, {"fixture": True})
    return stats


def test_finalization_actual_statistics_schema_and_artifact_metadata(tmp_path, monkeypatch):
    stats = final_fixture(tmp_path, monkeypatch)
    p.finish("binding")
    final = c.old.read_json(tmp_path/"final_status.json")
    assert final["complete"] and final["classification"] == stats["classification"]
    assert not final["automatic_successor"]
    assert c.verified(final["audit"])["passed"]
    assert c.verified(final["result"])["statistics"] == stats


def test_expiry_during_finalization_retains_report_but_never_claims_completion(tmp_path, monkeypatch):
    final_fixture(tmp_path, monkeypatch)
    def boundary(*a):
        if (tmp_path/"report.md").exists():
            raise c.BudgetExceeded("Finalization exceeded budget")
    monkeypatch.setattr(p, "budget_boundary", boundary)
    with pytest.raises(c.BudgetExceeded):
        p.finish("binding")
    assert (tmp_path/"result.json").exists() and (tmp_path/"report.md").exists()
    assert not (tmp_path/"final_status.json").exists()


@pytest.mark.parametrize("damage", [False, True])
def test_no_update_closure_checks_real_tensors_and_records_skipped_evaluation(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_freeze", lambda: "binding")
    models = {"all_ranking_unchanged": True, "models": {}}
    for fit in c.FITS:
        for arm in ("anchor", "ranking"):
            path = tmp_path/f"{arm}_{fit}.pt"
            value = 2 if damage and arm == "ranking" and fit == c.FITS[-1] else 1
            torch.save({"model_kind": "fixture", "model_config": {"width": 1},
                        "model_state_dict": {"weight": torch.tensor([value])}}, path)
            models["models"][f"{arm}_{fit}"] = c.artifact(path)
    monkeypatch.setattr(p.learning, "validate_models", lambda *a: models)
    c.old.save_new(tmp_path/"learning_complete.json", {"checked": True})
    if damage:
        with pytest.raises(ValueError, match="tensors"):
            p.finish_if_identical("binding")
        assert not (tmp_path/"final_status.json").exists()
    else:
        assert p.finish_if_identical("binding")
        final = c.old.read_json(tmp_path/"final_status.json")
        result = c.verified(final["result"])
        assert final["complete"] and not final["evaluation_run"]
        assert result["statistics"] is None and result["ranking_minus_anchor_exact_difference"] == 0.
