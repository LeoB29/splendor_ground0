"""Isolated hidden-process regression for an aged recovery and downstream work.

Only a tiny diagnostic network and a synthetic nearly terminal game are used.
No production training, source game, evaluation, or runtime file is touched.
"""
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
from tools.teacher_precision_512 import common as c, engine, experiment, learning, source, verify
from tools.teacher_precision_512.test_engine import fixture_registry, private_state
from . import recovery


PRIOR_ACTIVE_SECONDS = 13 * 3600 + 12 * 60
TASKS = (("learning", "fits", "normal"), ("learning", "fit_audits", "audit"),
         ("source_main", "collection", "source"), ("main", "audits", "evaluation"))


def _diagnostics(base, audit):
    torch.manual_seed(719)
    model = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(
        observation_size=6, action_feature_size=4, state_hidden_size=8,
        action_hidden_size=8, scorer_hidden_size=8, state_depth=1, action_depth=1,
        attention_layers=1, attention_heads=2, attention_feedforward_size=16))
    rows = [row(i, support=n) for i, n in enumerate((3, 4, 5, 8, 6))]
    for value in rows:
        sums = [4*v for v in value["old_sums_wdl"]]
        if len(sums) > 4:
            sums[3], sums[4] = 401, 79
        value.update(teacher_sums_wdl=sums, teacher_worlds=512,
                     teacher_outsider=max(range(3, len(sums)), key=lambda j: sums[j]) if len(sums) > 3 else -1)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    reports = {str(precision): learning.training_report(
        model, rows, base, {**learning.CONFIG, "batch_size": 2}, audit=audit, precision=precision)
        for precision in (False, True)}
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert reports["True"]["target_worlds"] == 512
    assert any(value["old_regret_units"] % 4 for value in reports["True"]["rows"])
    if audit:
        normal = c.old.read_json(base/"diagnostic_normal.json")
        for precision, report in reports.items():
            learning._compare(normal[precision], report)
    else:
        c.old.save_new(base/"diagnostic_normal.json", reports)
    return {"audit": audit, "old_rows": len(reports["False"]["rows"]),
            "precision_rows": len(reports["True"]["rows"]),
            "precision_worlds": reports["True"]["target_worlds"],
            "independent_matches_normal": audit, "parameters_unchanged": True}


def _source(base):
    state = private_state()
    spec = next(s for s in source.schedule("test") if s["seat"] == state.current_player and s["phase"] == 0)
    item = source.collect(spec, "fixture", base, c.old.serialize_state_snapshot(state))
    checked = experiment.run_job("source_test", "audits", (spec, item), "fixture", base)
    game, receipt = c.verified(item), c.verified(checked)
    selected = source._selected_spec(0, "test", (spec, game, item, checked), game["trace"][0])
    c.old.save_new(base/"selected_fixture.json", selected)
    assert "fixture_snapshot" not in selected
    assert receipt["passed"] and receipt["every_move_hidden_input_invariance"]
    return {"source": item, "source_audit": checked, "moves": len(game["trace"]),
            "all_move_audit": receipt["every_move_hidden_input_invariance"]}


def _evaluation(base):
    spec = c.old.read_json(base/"selected_fixture.json")
    c.old.save_new(base/"test/audit_sample.json", verify.sample_plan([spec], "test", "fixture"))
    item = experiment.run_job("test", "collection", spec, "fixture", base)
    checked = experiment.run_job("test", "audits", (spec, item), "fixture", base)
    receipt = c.verified(checked)
    assert receipt["passed"] and receipt["full_inner_replay"]
    assert receipt["recorded_continuations"] == item["continuations"] == receipt["replayed_continuations"]
    assert receipt["row"] == engine.statistics_row(c.verified(item))
    return {"evaluation": item, "evaluation_audit": checked, "full_inner_replay": True,
            "recorded_continuations": item["continuations"], "identical_statistics_row": True}


def _guard_controls(base, output):
    launch = c.old.read_json(output/"launch.json")
    original_clock = recovery.time.time
    exceptions = []
    try:
        recovery.time.time = lambda: launch["epoch_seconds"] + 48*3600 - launch["prior_active_seconds"] + 1
        for boundary in (c.budget_boundary, experiment.budget_boundary):
            try:
                boundary(base)
            except c.BudgetExceeded as exc:
                assert type(exc) is c.BudgetExceeded and "Forty-eight-hour" in str(exc)
                exceptions.append(type(exc).__module__+"."+type(exc).__name__)
            else:
                raise AssertionError("Cumulative48h ceiling not enforced in spawned worker")
    finally:
        recovery.time.time = original_clock
    c.budget_boundary(base)
    before = (base/"abort.json").read_bytes()
    experiment.abort(RuntimeError("Isolated hidden fixture cooperative hold"), base)
    for boundary in (c.budget_boundary, experiment.budget_boundary):
        try:
            boundary(base)
        except c.BudgetExceeded as exc:
            assert type(exc) is c.BudgetExceeded and "cooperative abort" in str(exc)
            exceptions.append(type(exc).__module__+"."+type(exc).__name__)
        else:
            raise AssertionError("New cooperative abort not enforced in spawned worker")
    assert (base/"abort.json").read_bytes() == before
    return {"ceiling_checked": True, "new_abort_checked": True,
            "exception_classes": exceptions, "old_abort_unchanged": True,
            "new_abort": c.artifact(output/"abort.json")}


def inspect_worker(task, study_base, recovery_base):
    base, output = Path(study_base), Path(recovery_base)
    assert c.old.read_json(base/"models_lock.json")["manifest"] == "fixture"
    assert c.budget_boundary is experiment.budget_boundary is recovery.budget_boundary
    assert c.active_seconds is recovery.active_seconds
    c.budget_boundary(base)
    active = c.active_seconds(base)
    assert PRIOR_ACTIVE_SECONDS <= active < PRIOR_ACTIVE_SECONDS + 600
    assert (base/"abort.json").exists()
    if task in ("normal", "audit"):
        result = _diagnostics(base, task == "audit")
    elif task == "source":
        result = _source(base)
    else:
        assert task == "evaluation"
        result = _evaluation(base)
    c.budget_boundary(base)
    if task == "evaluation":
        result["guard_controls"] = _guard_controls(base, output)
    print(f"aged recovery fixture task={task} pid={os.getpid()}", flush=True)
    return {"task": task, "result": result, "health": c.windows.assert_windowless_worker(),
            "active_seconds": active, "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "recovery_base": str(output), "report_module": learning.training_report.__module__}


def main(base):
    base = Path(base).resolve()
    assert not base.is_relative_to(c.OUTPUT.resolve())
    assert not base.is_relative_to(recovery.RECOVERY_OUTPUT.resolve())
    study, output = base/"study", base/"recovery"
    study.mkdir(); output.mkdir()
    supervisor = c.windows.assert_windowless_worker(base/"supervisor_stdout.txt", base/"supervisor_stderr.txt")
    epoch = time.time()
    c.old.save_new(study/"launch.json", {"epoch_seconds": epoch - 80*3600, "fixture": True})
    c.old.save_new(study/"abort.json", {"error": "Eight-hour execution ceiling reached; preserve evidence", "fixture": True})
    c.old.save_new(study/"models_lock.json", fixture_registry(study))
    seals = [c.artifact(study/name) for name in ("launch.json", "abort.json", "models_lock.json")]
    c.old.save_new(output/"launch.json", {"study_binding": "fixture", "prior_active_seconds": PRIOR_ACTIVE_SECONDS,
                   "epoch_seconds": epoch, "resolved_abort": seals[1]})
    c.old.save_new(output/"launch_seal.json", {"study_binding": "fixture", "launch": c.artifact(output/"launch.json")})
    recovery.install_runtime(study, output, "fixture", fixture=True)
    c.budget_boundary(study)
    stages = []
    for stage, kind, task in TASKS:
        runtime = c.windows.configure_spawn()
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                initializer=recovery.worker_initialize,
                initargs=(stage, kind, "fixture", study, output)) as pool:
            result = pool.submit(inspect_worker, task, study, output).result(timeout=120)
        stages.append({"stage": stage, "kind": kind, "runtime": runtime, "result": result})
    for seal in seals:
        c.verify_hash(seal)
    assert not (study/"workers").exists()
    assert (output/"abort.json").exists()
    c.old.save_new(base/"worker_fixture_complete.json", {"supervisor": supervisor, "stages": stages,
                   "original_seals": seals, "original_runtime_untouched": True,
                   "prior_active_seconds": PRIOR_ACTIVE_SECONDS})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try:
        main(destination)
    except BaseException:
        c.old.save_new(destination/"worker_fixture_failure.json", {"traceback": traceback.format_exc()})
        raise
