"""Owner-authorized saved-data screen; no simulation/model imports or successor."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.saved_equal_budget_breadth import common as c


def check():
    c.require(not (c.OUTPUT / "freeze.json").exists(), "Study already frozen")
    for number in range(1, 100):
        folder = c.OUTPUT / f"preflight_{number:02d}"
        if not folder.exists():
            folder.mkdir(parents=True)
            break
    else:
        raise ValueError("No unused preflight directory")
    identity = c.code_identity()
    started = time.time()
    command = [sys.executable, "-m", "pytest", "-q", "tools/saved_equal_budget_breadth",
               "-p", "no:cacheprovider", "--basetemp", str(folder / "temp")]
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        completed = subprocess.run(command, cwd=c.ROOT, stdout=stdout, stderr=stderr,
                                   timeout=300, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(identity == c.code_identity(), "Files changed during preflight")
    result = {"time": c.now(), "passed": completed.returncode == 0, "returncode": completed.returncode,
              "identity": identity, "command": command, "elapsed_seconds": time.time() - started,
              "stdout": c.reference(folder / "stdout.log"), "stderr": c.reference(folder / "stderr.log")}
    c.save(folder / "checks.json", result)
    print((folder / "stdout.log").read_text(encoding="utf-8", errors="replace"))
    c.require(result["passed"], f"Preflight failed: {folder}")
    print(c.relative(folder / "checks.json"))


def freeze(checks_path):
    c.require(not (c.OUTPUT / "freeze.json").exists(), "Study already frozen")
    check_path = c.local(checks_path)
    checks = c.read(check_path)
    c.require(checks["passed"] is True and checks["returncode"] == 0, "Preflight not passed")
    identity = c.code_identity()
    c.require(checks["identity"] == identity, "Preflight identity is stale")
    for key in ("stdout", "stderr"):
        c.authenticate(checks[key])
    inputs = c.collect_source()
    c.require(c.code_identity() == identity, "Preparation changed study files")
    manifest = {"identity": identity, "inputs": inputs, "checks": c.reference(check_path),
                "owner_authorized": True, "authorization": "User accepted the saved equal-budget breadth screen on 2026-10-03.",
                "scope": "Arithmetic-only reused development data; no new inference, fit, rollout or automatic successor."}
    c.save(c.OUTPUT / "manifest.json", manifest)
    binding = c.digest(manifest)
    c.save(c.OUTPUT / "freeze.json", {"time": c.now(), "binding": binding,
                                      "manifest": c.reference(c.OUTPUT / "manifest.json"), "owner_authorized": True})
    print({"binding": binding, "source_files": len(inputs["inventory"]), "roots": inputs["root_count"]})


def frozen(check_sources=True):
    seal = c.read(c.OUTPUT / "freeze.json")
    c.require(seal["owner_authorized"] is True, "Missing owner authorization")
    manifest = c.read(c.authenticate(seal["manifest"]))
    c.require(c.digest(manifest) == seal["binding"], "Frozen binding mismatch")
    c.require(manifest["identity"] == c.code_identity(), "Frozen code/config/runtime changed")
    checks = c.read(c.authenticate(manifest["checks"]))
    c.require(checks["passed"] is True and checks["identity"] == manifest["identity"], "Invalid frozen preflight")
    for key in ("stdout", "stderr"):
        c.authenticate(checks[key])
    if check_sources:
        c.check_inventory(manifest["inputs"]["inventory"])
    return seal["binding"], manifest


def launch():
    binding, _ = frozen()
    c.require(not (c.OUTPUT / "launch.json").exists(), "Already launched; no automatic retry")
    c.require(shutil.disk_usage(c.OUTPUT).free >= c.CONFIG["disk_reserve_bytes"], "Insufficient disk reserve")
    runtime = c.OUTPUT / "runtime"
    runtime.mkdir()
    python = Path(sys.executable).with_name("pythonw.exe") if os.name == "nt" else Path(sys.executable)
    c.require(python.is_file(), "Hidden Python runtime unavailable")
    epoch = time.time()
    receipt = {"time": c.now(), "manifest": binding, "epoch": epoch,
               "deadline_epoch": epoch + c.CONFIG["execution_cap_seconds"], "maximum_seconds": c.CONFIG["execution_cap_seconds"],
               "hidden": True, "automatic_successor": False}
    c.save(c.OUTPUT / "launch.json", receipt)
    command = [str(python), str(c.ROOT / "run_saved_equal_budget_breadth.py"), "execute"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (runtime / "stdout.log").open("xb") as stdout, (runtime / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, cwd=c.ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 env=env, close_fds=True,
                                 creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0,
                                 start_new_session=os.name != "nt")
    c.save(c.OUTPUT / "launch_ack.json", {"time": c.now(), "pid": child.pid,
                                         "manifest": binding, "launch": c.reference(c.OUTPUT / "launch.json"), "command": command})
    print({"pid": child.pid, "hidden": True, "output": c.relative(c.OUTPUT), "deadline_epoch": receipt["deadline_epoch"]})


def budget(launch_info):
    c.require(time.time() <= launch_info["deadline_epoch"], "One-hour execution ceiling reached")
    c.require(shutil.disk_usage(c.OUTPUT).free >= c.CONFIG["disk_reserve_bytes"], "Disk reserve reached")


def execute():
    from tools.saved_equal_budget_breadth.analysis import analyze_root, summarize
    from tools.saved_equal_budget_breadth.verify import verify_root, verify_statistics

    started = time.time()
    launch_info = c.read(c.OUTPUT / "launch.json")
    binding, manifest = frozen()
    c.require(binding == launch_info["manifest"], "Launch binding differs")
    c.save(c.OUTPUT / "run.lock", {"pid": os.getpid(), "manifest": binding, "time": c.now()})
    # Keep the machine awake only while this bounded task is executing; display sleep is allowed.
    kernel = None
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        c.require(bool(kernel.SetThreadExecutionState(0x80000001)), "Could not inhibit system sleep")
    try:
        rows, links = [], []
        config = manifest["identity"]["config"]
        inputs = manifest["inputs"]
        c.status({"stage": "analysis", "done": 0, "total": 360, "manifest": binding, "pid": os.getpid()})
        for i, ref in enumerate(inputs["roots"]):
            budget(launch_info)
            record = c.read(c.authenticate(ref))
            c.validate_record(record, inputs["specs"][i])
            row = analyze_root(record, config)
            path = c.OUTPUT / "analysis" / f"state_{i:04d}.json"
            c.save(path, {"manifest": binding, "source": {k: ref[k] for k in ("path", "sha256")}, "result": row})
            rows.append(row)
            links.append(c.reference(path))
            if (i + 1) % 10 == 0:
                c.status({"stage": "analysis", "done": i + 1, "total": 360, "manifest": binding, "pid": os.getpid()})
        budget(launch_info)
        statistics = summarize(rows, config)
        c.save(c.OUTPUT / "analysis_complete.json", {"time": c.now(), "manifest": binding, "roots": links,
                                                      "statistics": statistics, "elapsed_seconds": time.time() - started,
                                                      "verification_pending": True})
        audit_links = []
        c.status({"stage": "verification", "done": 0, "total": 360, "manifest": binding, "pid": os.getpid()})
        for i, (source, link) in enumerate(zip(inputs["roots"], links)):
            budget(launch_info)
            record = c.read(c.authenticate(source))
            saved = c.read(c.authenticate(link))
            c.require(saved["manifest"] == binding and saved["source"] == {k: source[k] for k in ("path", "sha256")}, "Analysis provenance changed")
            receipt = verify_root(record, saved["result"], config)
            c.require(receipt["passed"] is True, "Independent root reconstruction failed")
            path = c.OUTPUT / "verification" / f"state_{i:04d}.json"
            c.save(path, {"time": c.now(), "manifest": binding, "source": saved["source"],
                          "analysis": link, "receipt": receipt})
            audit_links.append(c.reference(path))
            if (i + 1) % 10 == 0:
                c.status({"stage": "verification", "done": i + 1, "total": 360, "manifest": binding, "pid": os.getpid()})
        budget(launch_info)
        statistics_audit = verify_statistics(rows, statistics, config)
        c.require(statistics_audit["passed"] is True, "Independent statistics failed")
        c.status({"stage": "closure", "done": 360, "total": 360, "manifest": binding, "pid": os.getpid()})
        # All inputs and every output receipt are checked again before final acceptance.
        frozen()
        for ref in links + audit_links:
            c.authenticate(ref)
        budget(launch_info)
        result = {"time": c.now(), "manifest": binding, "classification": statistics["classification"],
                  "statistics": statistics, "analysis": c.reference(c.OUTPUT / "analysis_complete.json"),
                  "source_binding": inputs["source_binding"], "scope": "Reused-development conditional decision score; not arena win rate.",
                  "roots": 360, "partitions_per_root": 64, "all_prescribed_checks_pass": True,
                  "automatic_successor": False, "new_simulations": 0, "new_training": 0, "full_inner_replay": False,
                  "elapsed_seconds": time.time() - launch_info["epoch"]}
        c.save(c.OUTPUT / "result.json", result)
        c.save(c.OUTPUT / "final_audit.json", {"time": c.now(), "manifest": binding, "passed": True,
                                               "result": c.reference(c.OUTPUT / "result.json"), "root_receipts": audit_links,
                                               "independent_statistics": statistics_audit,
                                               "source_inventory_count": len(inputs["inventory"]),
                                               "inherited_coverage": inputs["inherited_coverage"],
                                               "new_trajectory_replays": 0, "full_inner_replay": False})
        mean = statistics["score_percentage_points"]
        low, high = statistics["ci95_percentage_points"]
        report = ("# Saved equal-budget breadth screen\n\n"
                  f"Classification: {statistics['classification']}. All prescribed arithmetic and provenance checks passed.\n\n"
                  f"Broader uniform minus original four-candidate planner: **{mean:+.6f} percentage points**, "
                  f"paired95% interval **[{low:+.6f}, {high:+.6f}]**.\n\n"
                  "Both spend512 continuations at full support: eight by64 versus four by128. Short supports follow the frozen exact-budget rules. "
                  "All360 roots and64 partitions per root are retained; uncertainty resamples roots in18 cells. "
                  "Partitions are not independent replications. Scores are conditional terminal outcomes under035 continuations, not arena win rates.\n\n"
                  f"Practical >=+0.5pp hurdle: {statistics['practical_gain']}; positive95% lower endpoint: {statistics['superiority']}.\n\n"
                  "This is reused development evidence. Selection and assessment are disjoint within each partition; prior use of these data remains. "
                  "A pass supports proposing fresh confirmation. Equal continuation counts do not establish equal wall time or equivalence to eight by128.\n\n"
                  "No new inference, training, simulated game, continuation or trajectory replay was performed. Original sampled replay evidence was authenticated. "
                  "Every new partition and final statistic was independently reconstructed. No successor, promotion or monitoring was launched.\n")
        with (c.OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
        c.save(c.OUTPUT / "final_status.json", {"time": c.now(), "manifest": binding, "complete": True,
                                                "classification": statistics["classification"], "all_prescribed_checks_pass": True,
                                                "result": c.reference(c.OUTPUT / "result.json"),
                                                "audit": c.reference(c.OUTPUT / "final_audit.json"),
                                                "report": c.reference(c.OUTPUT / "report.md"),
                                                "elapsed_seconds": time.time() - launch_info["epoch"], "automatic_successor": False})
        c.status({"stage": "complete", "done": 360, "total": 360, "manifest": binding,
                  "classification": statistics["classification"], "pid": os.getpid()})
    finally:
        if kernel is not None:
            kernel.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "freeze", "launch", "execute"])
    parser.add_argument("--checks")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "freeze":
        c.require(args.checks is not None, "Specify the exact passed preflight receipt")
        freeze(args.checks)
    elif args.command == "launch":
        launch()
    else:
        try:
            execute()
        except BaseException as error:
            if not (c.OUTPUT / "failure.json").exists():
                c.save(c.OUTPUT / "failure.json", {"time": c.now(), "error": repr(error), "traceback": traceback.format_exc(),
                                                   "automatic_retry": False, "automatic_successor": False})
            c.status({"stage": "failed", "error": repr(error), "pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()
