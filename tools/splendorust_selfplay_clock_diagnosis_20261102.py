"""Saved-only reproduction of a mixed-clock learned-response checker failure.

No models, games, training, acknowledgement publication, or recovery are run.
Counterfactual timestamps demonstrate the defect; they do not recover the
unrecorded parent observation or accept any historical failed decision.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/splendorust_selfplay_pilot_qualification_20261102_01"
OUT = ROOT / "outputs/splendorust_selfplay_pilot_clock_diagnosis_20261102_01"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    begun = time.perf_counter()
    from tools.splendorust_selfplay_pilot_20261102 import qualification_control as control
    assert "torch" not in sys.modules
    assert not OUT.exists(), "Separate diagnostic output must be fresh"
    OUT.mkdir()
    files = [x for x in SOURCE.rglob("*") if x.is_file()]
    before = {x.relative_to(SOURCE).as_posix(): (x.stat().st_size, x.stat().st_mtime_ns) for x in files}
    folder = SOURCE / "fit"
    learned = folder / "learned_entrypoint"
    start_path = learned / "request_002.started.json"
    response_path = learned / "request_002.completed.json"
    start, response = read(start_path), read(response_path)
    freeze, execution = read(SOURCE / "freeze.json"), read(SOURCE / "fit_execution.json")
    expected_start = control.c.artifact(start_path)
    structural = {
        "schema": response["schema"] == "selfplay_learned_response_v1",
        "request_index": response["request_index"] == 2,
        "start_artifact": response["start"] == expected_start,
        "case_id": response["case_id"] == start["case_id"],
        "mode": response["mode"] == start["mode"],
    }
    assert all(structural.values())
    env = SimpleNamespace(
        deadline=read(folder / "owner.json")["monotonic_deadline"],
        wall_deadline=read(folder / "owner.json")["unix_deadline"],
    )
    tests = []

    def exercise(name, observed_mono, observed_wall, expected_error=None, case_ids=None):
        observed = list(execution["learned_response_receipts"])
        local_freeze = dict(freeze)
        if case_ids is not None:
            local_freeze["learned_entrypoint"] = dict(freeze["learned_entrypoint"], case_ids=case_ids)
        fake_time = SimpleNamespace(monotonic=lambda: observed_mono, time=lambda: observed_wall)
        with patch.object(control, "time", fake_time), patch.object(control.c, "atomic_save") as publish:
            error = None
            try:
                control.observe_learned_requests(folder, execution["identity"], env, local_freeze, observed)
            except ValueError as exc:
                error = str(exc)
            if expected_error is None:
                assert error is None and len(observed) == 3 and publish.call_count == 1
            else:
                assert error and expected_error in error and len(observed) == 2 and publish.call_count == 0
        tests.append({"name": name, "passed": True, "expected_error": expected_error,
                      "actual_error": error, "synthetic_parent_monotonic": observed_mono,
                      "synthetic_parent_unix": observed_wall,
                      "historical_receipt_written": False})

    end, wall = response["finished_monotonic"], response["finished_unix"]
    exercise("mixed_timer_ordering_reproduces_original_error", end - .010, wall + .005,
             "Owning complete response identity and ordering")
    exercise("same_domain_ordering_passes_in_memory_only", end + .005, wall + .005)
    exercise("late_monotonic_receipt_still_rejected", start["monotonic_deadline"], wall + .005,
             "Full learned response arrived after original external10s deadline")
    exercise("late_wall_receipt_still_rejected", end + .005, start["unix_deadline"],
             "Full learned response arrived after original external10s deadline")
    exercise("future_wall_finish_still_rejected", end + .005, wall - .005,
             "Owning complete response identity and ordering")
    ids = list(freeze["learned_entrypoint"]["case_ids"])
    ids[2] = "wrong_fixture"
    exercise("fixture_identity_still_required", end + .005, wall + .005,
             "Unchanged learned9/10-second clocks", case_ids=ids)

    samples = []
    for _ in range(2000):
        earlier = time.perf_counter()
        later = time.monotonic()
        samples.append(earlier - later)
    import psutil
    identities = [execution["identity"]]
    for scope, name, pid_key, created_key in [
        (ROOT / "outputs/splendorust_selfplay_pilot_workflow_20261102_01", "claim.json", "pid", "created"),
        (SOURCE, "guardian_claim.json", "guardian_pid", "guardian_created"),
        (SOURCE, "watchdog.json", "watchdog_pid", "watchdog_created"),
        (SOURCE, "launch.json", "pid", "created"),
    ]:
        item = read(scope / name)
        identities.append({"pid": item[pid_key], "created": item[created_key]})
    for item in identities:
        try:
            assert psutil.Process(item["pid"]).create_time() != item["created"], "Original owner still alive"
        except psutil.NoSuchProcess:
            pass
    after = {x.relative_to(SOURCE).as_posix(): (x.stat().st_size, x.stat().st_mtime_ns)
             for x in SOURCE.rglob("*") if x.is_file()}
    assert before == after
    assert not (learned / "request_002.parent_accepted.json").exists()
    assert "torch" not in sys.modules
    assert time.perf_counter() - begun < 60
    result = {
        "schema": "selfplay_saved_clock_diagnosis_v1", "status": "diagnosis_reproduced",
        "script_sha256": digest(Path(__file__)), "python": sys.version,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in
                   [start_path, response_path, SOURCE / "fit_execution.json", SOURCE / "freeze.json",
                    ROOT / "tools/splendorust_selfplay_pilot_20261102/qualification_control.py",
                    ROOT / "tools/splendorust_selfplay_pilot_20261102/training.py"]],
        "clocks": {n: vars(time.get_clock_info(n)) for n in ("monotonic", "perf_counter", "time")},
        "live_clock_probe": {"samples": len(samples), "later_monotonic_before_earlier_qpc": sum(x > 0 for x in samples),
                             "min_difference_seconds": min(samples), "max_difference_seconds": max(samples)},
        "saved_response_structural_checks": structural, "saved_response_elapsed_seconds": response["elapsed_seconds"],
        "tests": tests, "models": 0, "games": 0, "training_steps": 0,
        "original_evidence_inventory_unchanged": before == after, "inventory_files": len(before),
        "exact_original_owner_identities_absent": identities,
        "original_qualification_status": "incomplete", "full_pilot_launched": False,
        "historical_failed_parent_observation_recorded": False,
        "limitation": "The failed parent read times were not persisted. Synthetic readings reproduce the demonstrated clock-domain defect; they do not establish the exact failed reading or retroactively accept it.",
        "remedy": "Use an explicit consistent request clock in a separate version; keep whole-job/stage boundaries in their own declared original clock domain. Audit trainer environment deadlines and owner-start comparisons as well as response ordering. Record branch-level failure observations. Test cross-process clock handoff before fresh separately frozen qualification.",
        "elapsed_seconds": time.perf_counter() - begun,
    }
    path = OUT / "diagnosis.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "tests_passed": len(tests),
                      "elapsed_seconds": result["elapsed_seconds"], "models": 0, "games": 0,
                      "evidence_preserved": True, "report": str(path)}))


if __name__ == "__main__":
    main()
