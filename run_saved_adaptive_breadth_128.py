"""Owner-authorized saved-data screen; no simulation/model imports or successor."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.saved_adaptive_breadth_128 import common as c


def render_report(statistics):
    labels = {"adaptive_minus_uniform": "Adaptive minus uniform",
              "adaptive_minus_anchor": "Adaptive minus anchor",
              "uniform_minus_anchor": "Uniform minus anchor"}
    lines = ["# Saved adaptive allocation screen", "",
             f"Classification: {statistics['classification']}. All prescribed arithmetic and provenance checks passed.",
             "", "| Contrast | Gain in percentage points | Paired 95% interval |",
             "| --- | ---: | ---: |"]
    for contrast, label in labels.items():
        row = statistics["contrasts"][contrast]
        low, high = row["ci95_percentage_points"]
        lines.append(f"| {label} | {row['score_percentage_points']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ["", "These are reused-development conditional decision scores under greedy035 continuations, not arena win rates.",
              "", "All 360 roots, three original fits and 64 fixed partitions per root are retained. "
              "Partitions are averaged within roots; the bootstrap resamples roots within 18 fixed cells.",
              "", "At full eight-candidate support, adaptive allocation uses 32 worlds for all eight, "
              "then 64 additional worlds for four survivors and selects using their combined 96 outcomes. "
              "Uniform uses eight by64, anchor four by128. Every arm spends512 continuations. "
              "For smaller supports adaptive exactly equals uniform. Every tie uses original candidate order.",
              "", "The fixed gate requires adaptive-minus-uniform mean at least +0.25pp and adaptive-minus-anchor "
              "mean at least +0.5pp, with both paired 95% lower endpoints above zero.",
              f"Gate pass: {statistics['gate_pass']}.",
              "", "All arms use a common independent assessment suffix within each partition. "
              "The old uniform/anchor choices and permutations are preserved, but their scores are newly assessed "
              "on32 rather than64 worlds at full support. Earlier results remain unchanged. "
              "Reusing positions and overlapping partitions does not provide fresh confirmation.",
              "", "No new inference, training, simulation or trajectory replay was performed. "
              "Original prescribed sampled trajectory-audit evidence was authenticated; every new partition "
              "and final statistic was independently reconstructed. Equal continuation counts do not establish equal wall time.",
              "", "A pass supports a separately proposed fresh adaptive/uniform/anchor confirmation with the same gates. "
              "A miss closes this schedule; the previously confirmed uniform rule remains available for a game test. "
              "No tuning sweep, fresh experiment, arena, promotion, successor or monitoring starts automatically.", ""]
    return "\n".join(lines)


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
    command = [sys.executable, "-m", "pytest", "-q", "tools/saved_adaptive_breadth_128",
               "-p", "no:cacheprovider", "--basetemp", str(folder / "temp")]
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        try:
            completed = subprocess.run(command, cwd=c.ROOT, stdout=stdout, stderr=stderr,
                                       timeout=300, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            returncode, timeout = completed.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timeout = -1, True
    c.require(identity == c.code_identity(), "Files changed during preflight")
    result = {"time": c.now(), "passed": returncode == 0, "returncode": returncode, "timeout": timeout,
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
                "owner_authorized": True, "authorization": "User accepted the single saved adaptive allocation screen on 2026-10-03.",
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
    c.require(checks["passed"] is True and checks["returncode"] == 0
              and checks["identity"] == manifest["identity"], "Invalid frozen preflight")
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
    command = [str(python), str(c.ROOT / "run_saved_adaptive_breadth_128.py"), "supervise"]
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


def validate_launch(info, binding):
    c.require(info["manifest"] == binding and info["hidden"] is True
              and info["automatic_successor"] is False, "Launch contract differs")
    for name in ("epoch", "deadline_epoch"):
        c.require(type(info[name]) in (float, int) and math.isfinite(info[name]) and info[name] > 0,
                  "Invalid launch clock")
    c.require(info["maximum_seconds"] == c.CONFIG["execution_cap_seconds"]
              and info["deadline_epoch"] == info["epoch"] + c.CONFIG["execution_cap_seconds"],
              "Launch deadline differs from frozen ceiling")
    c.require(info["epoch"] <= time.time(), "Launch epoch is in the future")
    return True


def validate_receipt_envelope(envelope, analysis_ref, source_ref, prior_ref, index, config):
    c.require(envelope["analysis"] == analysis_ref and envelope["source"] == source_ref
              and envelope["prior_analysis"] == prior_ref, "Verification receipt links differ")
    receipt, prior = envelope["receipt"], envelope["prior_choices"]
    c.require(receipt["passed"] is True and type(receipt["index"]) is int and receipt["index"] == index
              and type(receipt["partitions"]) is int
              and receipt["partitions"] == config["partitions"], "Incomplete verification receipt")
    for field in ("independent_selection_and_assessment_reconstruction", "whole_world_vectors_preserved",
                  "exact_rational_comparisons", "equal_per_arm_selection_budget"):
        c.require(receipt[field] is True, f"Verification assurance missing: {field}")
    c.require(prior["passed"] is True and prior["prior_choices_preserved"] is True
              and prior["prior_scores_reused"] is False
              and type(prior["partitions"]) is int
              and prior["partitions"] == config["partitions"], "Prior choices receipt differs")
    return receipt["partitions"]


def supervise():
    """A hidden supervisor bounds the arithmetic child by the original deadline."""
    binding, _ = frozen(check_sources=False)
    info = c.read(c.OUTPUT / "launch.json")
    validate_launch(info, binding)
    budget(info)
    command = [sys.executable, str(c.ROOT / "run_saved_adaptive_breadth_128.py"), "execute"]
    with (c.OUTPUT / "runtime" / "worker_stdout.log").open("xb") as stdout, \
            (c.OUTPUT / "runtime" / "worker_stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, cwd=c.ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            c.save(c.OUTPUT / "worker_launch.json", {"time": c.now(), "manifest": binding, "pid": child.pid,
                                                    "launch": c.reference(c.OUTPUT / "launch.json"), "command": command})
            code = child.wait(timeout=max(.001, info["deadline_epoch"] - time.time()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
            raise TimeoutError("One-hour execution ceiling reached; arithmetic child stopped")
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
            raise
    c.require(code == 0, f"Arithmetic child failed with exit code {code}")
    final = c.read(c.OUTPUT / "final_status.json")
    c.require(final["complete"] is True and final["all_prescribed_checks_pass"] is True
              and final["manifest"] == binding, "Arithmetic child did not close successfully")


def execute():
    from tools.saved_adaptive_breadth_128.analysis import analyze_root, summarize
    from tools.saved_adaptive_breadth_128.verify import verify_root, verify_statistics

    started = time.time()
    launch_info = c.read(c.OUTPUT / "launch.json")
    validate_launch(launch_info, c.read(c.OUTPUT / "freeze.json")["binding"])
    budget(launch_info)
    binding, manifest = frozen()
    budget(launch_info)
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
            previous = c.read(c.authenticate(inputs["prior_roots"][i]))
            c.validate_preserved_choices(previous["result"], row)
            path = c.OUTPUT / "analysis" / f"state_{i:04d}.json"
            c.save(path, {"manifest": binding, "source": {k: ref[k] for k in ("path", "sha256")}, "result": row})
            rows.append(row)
            links.append(c.reference(path))
            if (i + 1) % 10 == 0:
                c.status({"stage": "analysis", "done": i + 1, "total": 360, "manifest": binding, "pid": os.getpid()})
        budget(launch_info)
        statistics = summarize(rows, config)
        budget(launch_info)
        c.save(c.OUTPUT / "analysis_complete.json", {"time": c.now(), "manifest": binding, "roots": links,
                                                      "statistics": statistics, "elapsed_seconds": time.time() - started,
                                                      "verification_pending": True})
        rows.clear()
        audit_links = []
        c.status({"stage": "verification", "done": 0, "total": 360, "manifest": binding, "pid": os.getpid()})
        for i, (source, link) in enumerate(zip(inputs["roots"], links)):
            budget(launch_info)
            record = c.read(c.authenticate(source))
            saved = c.read(c.authenticate(link))
            c.require(saved["manifest"] == binding and saved["source"] == {k: source[k] for k in ("path", "sha256")}, "Analysis provenance changed")
            receipt = verify_root(record, saved["result"], config)
            previous = c.read(c.authenticate(inputs["prior_roots"][i]))
            preserved = c.validate_preserved_choices(previous["result"], saved["result"])
            c.require(receipt["passed"] is True, "Independent root reconstruction failed")
            path = c.OUTPUT / "verification" / f"state_{i:04d}.json"
            c.save(path, {"time": c.now(), "manifest": binding, "source": saved["source"],
                          "analysis": link, "receipt": receipt, "prior_choices": preserved,
                          "prior_analysis": inputs["prior_roots"][i]})
            audit_links.append(c.reference(path))
            if (i + 1) % 10 == 0:
                c.status({"stage": "verification", "done": i + 1, "total": 360, "manifest": binding, "pid": os.getpid()})
        budget(launch_info)
        analysis_ref = c.reference(c.OUTPUT / "analysis_complete.json")
        persisted_analysis = c.read(c.authenticate(analysis_ref))
        c.require(persisted_analysis["manifest"] == binding and persisted_analysis["roots"] == links
                  and persisted_analysis["statistics"] == statistics, "Persisted analysis closure differs")
        verified_rows, partition_count = [], 0
        for i, (source, link, audit_link, prior_link) in enumerate(zip(inputs["roots"], links, audit_links,
                                                                      inputs["prior_roots"], strict=True)):
            budget(launch_info)
            saved, envelope = c.read(c.authenticate(link)), c.read(c.authenticate(audit_link))
            c.require(saved["manifest"] == envelope["manifest"] == binding, "Verification binding differs")
            source_ref = {k: source[k] for k in ("path", "sha256")}
            c.require(saved["source"] == source_ref and saved["result"]["spec"]["index"] == i,
                      "Persisted root order/source differs")
            partition_count += validate_receipt_envelope(envelope, link, source_ref, prior_link, i, config)
            c.require(envelope["receipt"]["mean_difference_fractions"] == saved["result"]["mean_difference_fractions"]
                      and envelope["receipt"]["mean_score_differences"] == saved["result"]["mean_score_differences"],
                      "Verified raw root statistics differ from receipt")
            verified_rows.append(saved["result"])
        c.require(len(verified_rows) == config["roots"]
                  and partition_count == config["roots"] * config["partitions"], "Incomplete verified panel")
        statistics_audit = verify_statistics(verified_rows, persisted_analysis["statistics"], config)
        budget(launch_info)
        c.require(statistics_audit["passed"] is True, "Independent statistics failed")
        c.status({"stage": "closure", "done": 360, "total": 360, "manifest": binding, "pid": os.getpid()})
        # All inputs and every output receipt are checked again before final acceptance.
        frozen()
        budget(launch_info)
        for ref in links + audit_links:
            c.authenticate(ref)
        budget(launch_info)
        result = {"time": c.now(), "manifest": binding, "classification": statistics["classification"],
                  "statistics": statistics, "analysis": analysis_ref,
                  "source_binding": inputs["source_binding"], "prior_binding": inputs["prior_binding"],
                  "prior_choices_preserved": True, "prior_scores_reused": False, "scope": "Reused-development conditional decision score; not arena win rate.",
                  "roots": 360, "partitions_per_root": 64, "all_prescribed_checks_pass": True,
                  "automatic_successor": False, "new_simulations": 0, "new_training": 0, "full_inner_replay": False,
                  "elapsed_seconds": time.time() - launch_info["epoch"]}
        c.save(c.OUTPUT / "result.json", result)
        c.save(c.OUTPUT / "final_audit.json", {"time": c.now(), "manifest": binding, "passed": True,
                                               "result": c.reference(c.OUTPUT / "result.json"), "root_receipts": audit_links,
                                               "independent_statistics": statistics_audit,
                                               "verified_roots": len(verified_rows), "verified_partitions": partition_count,
                                               "source_inventory_count": len(inputs["inventory"]),
                                               "inherited_coverage": inputs["inherited_coverage"],
                                               "new_trajectory_replays": 0, "full_inner_replay": False})
        report = render_report(statistics)
        with (c.OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
        budget(launch_info)
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
    parser.add_argument("command", choices=["check", "freeze", "launch", "execute", "supervise"])
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
            supervise() if args.command == "supervise" else execute()
        except BaseException as error:
            if not (c.OUTPUT / "failure.json").exists():
                c.save(c.OUTPUT / "failure.json", {"time": c.now(), "error": repr(error), "traceback": traceback.format_exc(),
                                                   "automatic_retry": False, "automatic_successor": False})
            c.status({"stage": "failed", "error": repr(error), "pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()
