"""Isolated real hidden-worker regression beyond every predecessor time limit."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from tools.independent_game_coverage_128_recovery_01.test_learning import row
from . import common as c, engine, experiment, labels, learning, source, verify
from .test_engine import fixture_registry, private_state
from .test_labels import fixture_spec as label_fixture_spec


PRIOR_ACTIVE_SECONDS = 96*3600
TASKS = (("learning", "fits", "normal"), ("learning", "fit_audits", "audit"),
         ("source_train_probe", "collection", "source_train"),
         ("source_main", "collection", "source_eval"),
         ("labels_probe", "collection", "labels"), ("main", "audits", "evaluation"))


def _diagnostic_input():
    torch.manual_seed(719)
    model = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(
        observation_size=6, action_feature_size=4, state_hidden_size=8,
        action_hidden_size=8, scorer_hidden_size=8, state_depth=1, action_depth=1,
        attention_layers=1, attention_heads=2, attention_feedforward_size=16))
    return model, [row(i, support=n) for i, n in enumerate((3, 4, 5, 8, 6))]


def _diagnostics(base, audit):
    model, rows = _diagnostic_input()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    report = learning.training_report(model, rows, base, {**learning.CONFIG, "batch_size":2}, audit=audit)
    assert all(torch.equal(value, before[key]) for key,value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    if audit:
        learning._compare(c.old.read_json(base/"diagnostic_normal.json"), report)
    else:
        c.old.save_new(base/"diagnostic_normal.json", report)
    return {"audit":audit, "rows":len(report["rows"]),
            "independent_matches_normal":audit, "parameters_unchanged":True}


def _source(base, training):
    assert set(source._MODELS) == {f"anchor_{fit}" for fit in c.FITS}
    if training:
        assert not (base/"models_lock.json").exists(), "TRAIN initialization must need only its three anchors"
    else:
        assert len(c.old.read_json(base/"models_lock.json")["models"]) == 15
    state = private_state()
    available = [s for s in source.schedule("test") if s["seat"] == state.current_player and s["phase"] == 0]
    spec = available[0 if training else 1]
    item = source.collect(spec, "fixture", base, c.old.serialize_state_snapshot(state))
    checked = experiment.run_job("source_test", "audits", (spec,item), "fixture", base)
    game, receipt = c.verified(item), c.verified(checked)
    assert receipt["passed"] and receipt["every_move_hidden_input_invariance"]
    if not training:
        selected = source._selected_spec(0, "test", (spec,game,item,checked), game["trace"][0])
        c.old.save_new(base/"selected_fixture.json", selected)
    return {"source":item, "source_audit":checked, "moves":len(game["trace"]),
            "training_registry_only":training, "all_move_audit":True}


def _labels(base):
    spec = label_fixture_spec()
    item = experiment.run_job("labels_probe", "collection", spec, "fixture", base)
    checked = experiment.run_job("labels_probe", "audits", (spec,item), "fixture", base)
    receipt = c.verified(checked)
    assert receipt["passed"] and receipt["full_inner_replay"]
    assert receipt["replayed_continuations"] == 128
    assert receipt["independent_public_private_invariant_encoding"]
    return {"label":item, "label_audit":checked, "full_inner_replay":True,
            "replayed_continuations":receipt["replayed_continuations"]}


def _evaluation(base):
    spec = c.old.read_json(base/"selected_fixture.json")
    c.old.save_new(base/"test/audit_sample.json", verify.sample_plan([spec], "test", "fixture"))
    item = experiment.run_job("test", "collection", spec, "fixture", base)
    checked = experiment.run_job("test", "audits", (spec,item), "fixture", base)
    receipt = c.verified(checked)
    assert receipt["passed"] and receipt["full_inner_replay"]
    assert receipt["recorded_continuations"] == item["continuations"] == receipt["replayed_continuations"]
    assert receipt["row"] == engine.statistics_row(c.verified(item))
    return {"evaluation":item, "evaluation_audit":checked, "full_inner_replay":True,
            "recorded_continuations":item["continuations"], "identical_statistics_row":True}


def _guard_controls(base):
    launch = c.old.read_json(base/"launch.json")
    model, rows = _diagnostic_input()
    checks = (("common", lambda:c.budget_boundary(base)),
              ("orchestrator", lambda:experiment.budget_boundary(base)),
              ("normal_report", lambda:learning.training_report(model,rows,base,learning.CONFIG)),
              ("audit_report", lambda:learning.training_report(model,rows,base,learning.CONFIG,audit=True)))
    original_clock = c.time.time
    exceptions = []
    try:
        c.time.time = lambda:launch["epoch_seconds"]+168*3600-launch["prior_active_seconds"]+1
        for name,boundary in checks:
            try: boundary()
            except c.BudgetExceeded as exc:
                assert type(exc) is c.BudgetExceeded and "Seven-day" in str(exc)
                exceptions.append({"mode":"168h", "path":name, "exception":type(exc).__module__+"."+type(exc).__name__})
            else: raise AssertionError("Current168h ceiling not enforced in hidden worker: "+name)
    finally:
        c.time.time = original_clock
    c.budget_boundary(base)
    experiment.abort(RuntimeError("Isolated96h hidden-worker cooperative hold"), base)
    before = c.artifact(base/"abort.json")
    for name,boundary in checks:
        try: boundary()
        except c.BudgetExceeded as exc:
            assert type(exc) is c.BudgetExceeded and "Cooperative abort" in str(exc)
            exceptions.append({"mode":"abort", "path":name, "exception":type(exc).__module__+"."+type(exc).__name__})
        else: raise AssertionError("Current cooperative abort not enforced in hidden worker: "+name)
    experiment.abort(RuntimeError("Must preserve first cooperative hold"), base)
    c.verify_hash(before)
    return {"ceiling_checked":True, "abort_checked":True, "exceptions":exceptions, "abort":before}


def inspect_worker(task, study_base):
    base = Path(study_base)
    assert c.budget_boundary is experiment.budget_boundary
    c.budget_boundary(base)
    active = c.active_seconds(base)
    assert PRIOR_ACTIVE_SECONDS <= active < PRIOR_ACTIVE_SECONDS+900
    # This mutates only the disposable child's memory and proves no ancestor
    # timer is reached by any current diagnostic, source, label, or eval path.
    def stale_guard(*args, **kwargs): raise AssertionError("Inherited8h/48h execution guard reached")
    learning.prior.c.budget_boundary = stale_guard
    learning.checkpoint_helpers.c.budget_boundary = stale_guard
    if task in ("normal", "audit"):
        result = _diagnostics(base, task == "audit")
    elif task in ("source_train", "source_eval"):
        result = _source(base, task == "source_train")
    elif task == "labels":
        result = _labels(base)
    else:
        assert task == "evaluation"
        result = _evaluation(base)
    c.budget_boundary(base)
    if task == "evaluation": result["guard_controls"] = _guard_controls(base)
    print(f"aged scaling fixture task={task} pid={os.getpid()}", flush=True)
    return {"task":task, "result":result, "health":c.windows.assert_windowless_worker(),
            "active_seconds":active, "threads":torch.get_num_threads(),
            "deterministic":torch.are_deterministic_algorithms_enabled()}


def main(destination):
    base = Path(destination).resolve()
    assert not base.is_relative_to(c.OUTPUT.resolve())
    study = base/"study"
    study.mkdir()
    supervisor = c.windows.assert_windowless_worker(base/"supervisor_stdout.txt", base/"supervisor_stderr.txt")
    c.old.save_new(study/"launch.json", {"manifest":"fixture", "prior_active_seconds":PRIOR_ACTIVE_SECONDS,
                   "epoch_seconds":time.time(), "fixture":True})
    c.old.save_new(study/"launch_seal.json", {"manifest":"fixture", "receipt":c.artifact(study/"launch.json")})
    registry = fixture_registry(study)
    c.old.save_new(study/"source_models_lock.json", {"manifest":"fixture", "source_only":True,
                   "models":{key:value for key,value in registry["models"].items() if key.startswith("anchor_")}})
    seals = [c.artifact(study/name) for name in ("launch.json", "launch_seal.json", "source_models_lock.json")]
    stages = []
    for stage,kind,task in TASKS:
        if task == "source_eval":
            c.old.save_new(study/"models_lock.json", registry)
            seals.append(c.artifact(study/"models_lock.json"))
        runtime = c.windows.configure_spawn()
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                initializer=experiment.worker_initialize, initargs=(stage,kind,"fixture",study)) as pool:
            result = pool.submit(inspect_worker,task,study).result(timeout=180)
        stages.append({"stage":stage, "kind":kind, "runtime":runtime, "result":result})
    for seal in seals: c.verify_hash(seal)
    assert (study/"abort.json").exists()
    c.old.save_new(base/"worker_fixture_complete.json", {"supervisor":supervisor, "stages":stages,
                   "original_seals":seals, "original_receipts_unchanged":True,
                   "prior_active_seconds":PRIOR_ACTIVE_SECONDS})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try: main(destination)
    except BaseException:
        c.old.save_new(destination/"worker_fixture_failure.json", {"traceback":traceback.format_exc()})
        raise
