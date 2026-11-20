from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import pytest
from tools.higher_teacher_curve_128 import common as c, experiment as exp


def test_production_receipt_budget_and_preserved_abort(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    with pytest.raises(ValueError): c.budget_boundary(tmp_path)
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()})
    monkeypatch.setattr(c.shutil, "disk_usage", lambda _: SimpleNamespace(free=10*1024**3))
    c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time, "time", lambda: c.old.read_json(tmp_path/"launch.json")["epoch_seconds"]+1801)
    with pytest.raises(c.BudgetExceeded): c.budget_boundary(tmp_path)
    exp.abort(ValueError("first"), tmp_path)
    ref = c.artifact(tmp_path/"abort.json")
    exp.abort(ValueError("second"), tmp_path)
    c.verify_hash(ref)
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "first"


def test_storage_reserve(tmp_path, monkeypatch):
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()})
    monkeypatch.setattr(c.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    with pytest.raises(c.BudgetExceeded): c.budget_boundary(tmp_path)


def test_failed_audit_cannot_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "validate_freeze", lambda: ("fixture", {}))
    c.old.save_new(tmp_path/"assessment.json", {"manifest": "fixture"})
    c.old.save_new(tmp_path/"audit.json", {"manifest": "fixture", "passed": False})
    with pytest.raises(ValueError):
        exp.finish("fixture", {}, c.artifact(tmp_path/"assessment.json"), c.artifact(tmp_path/"audit.json"))
    assert not (tmp_path/"final_status.json").exists()


def test_report_keeps_training_probe_and_full_validation_distinct(tmp_path):
    from tools.higher_teacher_curve_128 import reporting
    metrics = {"rows":168, "first_winner_accuracy":.4, "any_old_max_accuracy":.42,
               "mean_loss":1.2, "mean_old_score_regret":.05}
    probe = {"metrics":{"all_rows":metrics}, "regret_capture_vs_initializer":.1,
             "diagnostic_scope":"original_168_train_probe"}
    full = {"metrics":{"all_rows":{**metrics,"rows":9461}}, "regret_capture_vs_initializer":.6,
            "diagnostic_scope":"full_train", "metrics_by_slice":{"original_168_train":probe}}
    assert reporting._training_probe(full) == reporting._training_probe(probe) == probe
    broken = {**probe,"metrics":{"all_rows":{**metrics,"rows":9461}}}
    with pytest.raises(ValueError): reporting._training_probe(broken)
    parent = {"training":{}}
    for fit in c.FITS:
        parent["training"][str(fit)] = {}
        for u in c.CHECKPOINTS:
            path=tmp_path/f"training_{fit}_{u}.json"
            c.old.save_new(path,full if u in (0,1784) else probe)
            parent["training"][str(fit)][str(u)]=c.artifact(path)
    low=tmp_path/"old_validation.json"
    c.old.save_new(low,{"families":{"lowrate":{"metrics":{"all_rows":metrics}}}})
    parent["validation_old_targets"]=c.artifact(low)
    names = [*c.CONFIG["primary_contrasts"],"lowrate_outsider_minus_anchor_outsider","lowrate_supported_minus_anchor"]
    row={"mean":0.,"mean_bounds":[0.,0.],"ci95":[-.01,.01]}
    stats={"checkpoints":list(c.CHECKPOINTS),"curves":{
        str(u):{"groups":{split:{"contrasts":{n:row for n in names}} for split in ("train","validation")}}
        for u in c.CHECKPOINTS},"simultaneous_validation":{"groups":{
        str(u):{n:row for n in c.CONFIG["primary_contrasts"]} for u in c.CHECKPOINTS[1:]}}}
    result={"statistics":stats,"full_validation_old_labels":{
        str(u):{"states":2489,"matched_source_fit":True,"metrics":{"all_rows":metrics}} for u in c.CHECKPOINTS}}
    refs=reporting.write_report("fixture",result,parent,tmp_path)
    for key in ("report","figure","figure_pdf","figure_data","render_receipt"):
        c.verify_hash(refs[key])
    text=(tmp_path/"report.md").read_text(encoding="utf-8")
    assert "72 noninitial" in text and "2,489" in text and "original 168-position probe" in text
    assert "Versus fixed final higher rate" in text and "Full training endpoints" in text
    assert not c.old.read_json(tmp_path/"render_complete.json")["torch_imported"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hidden-process integration")
def test_actual_hidden_model_loading_process(tmp_path):
    registry = c.old.read_json(c.PARENT/"models_lock.json")
    ref = registry["models"]["focused_161001"]
    c.old.save_new(tmp_path/"fixture_model.json", ref)
    launch = c.windows.detached_launch([sys.executable, "-u", "-m", "tools.higher_teacher_curve_128.worker_fixture", tmp_path],
                                       tmp_path/"stdout.txt", tmp_path/"stderr.txt", c.ROOT)
    assert Path(launch["command"][0]).name.lower() == "pythonw.exe"
    assert launch["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+40
    while not (tmp_path/"complete.json").exists():
        if (tmp_path/"failure.json").exists(): pytest.fail(c.old.read_json(tmp_path/"failure.json")["traceback"])
        if time.monotonic() > deadline: pytest.fail(f"Hidden integration timeout pid={launch['pid']}")
        time.sleep(.05)
    result = c.old.read_json(tmp_path/"complete.json")
    assert result["health"]["pid"] == launch["pid"]
    assert result["health"]["console_window"] == result["health"]["console_process_count"] == 0
    assert result["health"]["console_process_ids"] == []
    assert result["model"] == ref and result["parameters"] > 0
    assert result["eval"] and result["deterministic"] and result["threads"] == 1
    assert (tmp_path/"stderr.txt").stat().st_size == 0
