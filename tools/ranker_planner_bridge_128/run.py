"""One-shot guarded execution of the saved-world planner diagnostic.

All scientific inputs are read-only. The worker authenticates them before
sealing an execution lock, then uses two independent arithmetic implementations.
The parent enforces resource bounds even while a worker is in a blocking call.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time
import traceback

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/ranker_planner_bridge_128"
PROTOCOL = ROOT / "docs/engineering/ranker_planner_bridge_128.md"
TESTS = ROOT / "tests/test_ranker_planner_bridge_128.py"
MAIN = ROOT / "outputs/structured_proposer_128"
ALT = ROOT / "outputs/alternative_ranker_128"
MAIN_BINDING = "57cd1786992c123e2e8fdedf6e4cf567196588efc8bd1ab9d2bbb24b7908f454"
ALT_BINDING = "c4cebd0e386787617404cf802eb3c4335801104449651f734dcd904ddf8e4a0a"
MAX_SECONDS = 1200
MAX_RSS = 1024**3
MAX_OUTPUT = 32 * 1024**2
RECEIPT_RESERVE = 64 * 1024
DISK_RESERVE = 5 * 1024**3
PINS = {
    "outputs/alternative_ranker_128/predictions.json": "02ffa2375cce830236b649c85990b74f791b2f8a18f033d857a287693d854569",
    "outputs/alternative_ranker_128/predictions_seal.json": "1c344116af9044d7183e1ef9ede9988e008b25f02ae1ff3f64e30e39b04a862f",
    "outputs/alternative_ranker_128/result.json": "ca16e70028b166a767ccea4ea2d36a9295b7d9be77ee365d566a71ffc80be821",
    "outputs/structured_proposer_128/final_status.json": "1c59c0f7b07bc4a37ca62ff519536932ca6c0afdcb90cfed729ad69fe5731e09",
    "outputs/structured_proposer_128/final_audit.json": "97ac6b861f81031ea92b5d3332c99a68c791fb3b3af991f4db26b4ea74394f2e",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def reference(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Reference outside repository")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def output_bytes():
    return sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file())


def new_json(path, value, *, control=False):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE), "Output outside diagnostic namespace")
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    limit = MAX_OUTPUT if control else MAX_OUTPUT - RECEIPT_RESERVE
    require(output_bytes() + len(encoded) <= limit, "Output size ceiling")
    with path.open("xb") as handle:
        handle.write(encoded)


def inventory():
    paths = list(Path(__file__).parent.glob("*.py")) + [PROTOCOL, TESTS]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}


def runtime():
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "psutil", "pytest", "threadpoolctl")}}


def configure():
    import psutil
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    return process


def resource_check(started, *, entry=False, process=None):
    from tools.alternative_ranker_128.common import resource_snapshot
    import psutil
    require(time.monotonic() - started <= MAX_SECONDS, "20-minute execution ceiling")
    snapshot = resource_snapshot()
    require(snapshot["available_bytes"] >= (768 if entry else 128) * 1024**2,
            "Available RAM reserve reached")
    require(snapshot["commit_headroom_bytes"] >= (2 if entry else 1) * 1024**3,
            "Commit headroom reserve reached")
    require(snapshot["disk_free_bytes"] >= DISK_RESERVE + (MAX_OUTPUT if entry else 0),
            "Disk reserve reached")
    process = process or psutil.Process()
    snapshot["rss_bytes"] = process.memory_info().rss
    require(snapshot["rss_bytes"] <= MAX_RSS, "Process RSS ceiling")
    require(output_bytes() <= MAX_OUTPUT - RECEIPT_RESERVE, "Output size ceiling")
    return snapshot


class Inputs:
    def __init__(self, boundary):
        self.files = {}
        self.boundary = boundary

    def add(self, ref):
        require(isinstance(ref, dict) and {"path", "sha256"} <= ref.keys(), "Invalid input reference")
        path = (ROOT / ref["path"]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Input path invalid")
        name = path.relative_to(ROOT).as_posix()
        if name in self.files:
            require(self.files[name] == ref["sha256"], "Conflicting input digests")
        else:
            self.boundary()
            require(sha(path) == ref["sha256"], "Changed input: " + name)
            self.files[name] = ref["sha256"]
        return path

    def metadata(self, path):
        return read(self.add(reference(path)))

    def refs_in(self, value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys():
                self.add(value)
            for child in value.values():
                self.refs_in(child)
        elif isinstance(value, list):
            for child in value:
                self.refs_in(child)

    def revalidate(self):
        for index, (name, expected) in enumerate(sorted(self.files.items())):
            self.boundary()
            require(sha(ROOT / name) == expected, "Input changed during execution: " + name)
            if index and index % 1000 == 0:
                print(f"Rechecked {index} input files", flush=True)


def authenticate(inputs):
    from tools.alternative_ranker_128 import common as c, evaluation as e
    frozen = c.validate_freeze()
    require(frozen["binding"] == ALT_BINDING, "Unexpected alternative freeze")
    for name, expected in PINS.items():
        inputs.add({"path": name, "sha256": expected})
    registration = inputs.metadata(ALT / "manifest.json")
    for name, expected in registration["own_sources"].items():
        inputs.add({"path": name, "sha256": expected})
    for name, expected in registration["source"]["dependencies"].items():
        inputs.add({"path": name, "sha256": expected})
    inputs.refs_in(registration)
    inputs.metadata(ALT / "freeze.json")
    predictions = e.validate_prediction_seal(ALT / "predictions.json", ALT / "predictions_seal.json",
                                           repo_root=ROOT, manifest=ALT_BINDING, study_binding=MAIN_BINDING)
    inputs.refs_in(predictions)
    for label in ("models_lock", "original_anchors"):
        metadata = read(inputs.add(predictions[label]))
        inputs.refs_in(metadata)
        if label == "models_lock":
            for dependency in ("learning", "dataset"):
                inputs.refs_in(read(inputs.add(metadata[dependency])))
    specs, roots, audits = e._authenticated_panel(MAIN, ROOT, MAIN_BINDING, predictions["selection"])
    for filename in ("freeze.json", "manifest.json", "final_status.json", "final_audit.json",
                     "selection_seal.json", "selection.json", "main_complete.json", "probe_complete.json",
                     "main_collection_lock.json", "main_audits_lock.json",
                     "probe_collection_lock.json", "probe_audits_lock.json"):
        inputs.refs_in(inputs.metadata(MAIN / filename))
    by_index = {row["index"]: row for row in predictions["rows"]}
    for spec, root_ref, audit_ref in zip(specs, roots, audits, strict=True):
        inputs.boundary()
        inputs.add(root_ref)
        receipt = read(inputs.add(audit_ref))
        require(receipt.get("manifest") == MAIN_BINDING and receipt.get("stage") == "main"
                and receipt.get("passed") is True and receipt["spec"] == spec
                and e._same_reference(receipt["root"], root_ref, ROOT), "Root audit mismatch")
        require(e._identity(by_index[spec["index"]]) == e._identity(spec), "Prediction identity mismatch")
        inputs.refs_in(spec)
    require(len(by_index) == len(specs) == len(roots) == len(audits) == 720, "Incomplete fixed panel")
    print(f"Authenticated {len(inputs.files)} source files and all 720 root/audit links", flush=True)
    return by_index, specs, roots


def validate_supervisor(supervisor):
    import psutil
    lease = read(BASE / "active.json")
    require(lease == supervisor and os.getppid() == supervisor["pid"], "Worker supervisor identity mismatch")
    parent = psutil.Process(supervisor["pid"])
    require(parent.is_running() and parent.create_time() == supervisor["created"],
            "Worker supervisor no longer alive")


def worker(execution, preparation, supervisor):
    from threadpoolctl import threadpool_info, threadpool_limits
    from . import analysis, independent
    started = time.monotonic()
    configure()
    validate_supervisor(supervisor)
    resource_check(started, entry=True)
    checks = read(preparation / "checks.json")
    require(checks.get("passed") is True and checks["inventory"] == inventory()
            and checks["runtime"] == runtime(), "Preparation no longer binds implementation")
    require(reference(preparation / "pytest.log") == checks["log"], "Preparation log changed")
    peak = 0

    def boundary():
        nonlocal peak
        validate_supervisor(supervisor)
        snapshot = resource_check(started)
        peak = max(peak, snapshot["rss_bytes"])

    with threadpool_limits(limits=1):
        require(all(item["num_threads"] == 1 for item in threadpool_info()), "Thread limit not applied")
        inputs = Inputs(boundary)
        inputs.add(reference(preparation / "checks.json"))
        inputs.add(reference(preparation / "pytest.log"))
        predictions, specs, roots = authenticate(inputs)
        sources = inventory()
        for name, expected in sources.items():
            inputs.add({"path": name, "sha256": expected})
        lock = {"schema": "ranker_planner_bridge_execution_lock_v1", "time": now(),
                "owner_authorized": True, "post_hoc_opened_panel": True,
                "confirmatory": False, "automatic_successor": False,
                "sources": sources, "input_files": inputs.files, "runtime": runtime(),
                "checks": reference(preparation / "checks.json"),
                "supervisor": supervisor,
                "main_binding": MAIN_BINDING, "alternative_binding": ALT_BINDING,
                "resources": {"seconds": MAX_SECONDS, "rss_bytes": MAX_RSS,
                              "output_bytes": MAX_OUTPUT, "threads": 1, "priority": "below_normal"},
                "analysis": {"roots": 720, "selection_worlds": 64, "assessment_worlds": 64,
                             "directions_per_root": 2, "bootstrap_samples": 50000,
                             "bootstrap_seed": 2813000000, "bootstrap_chunk": 1000}}
        new_json(execution / "execution_lock.json", lock)
        binding = digest(lock)
        print("Execution locked before calculating diagnostic metrics", flush=True)
        rows = []
        for spec, ref in zip(specs, roots, strict=True):
            boundary()
            path = ROOT / ref["path"]
            require(sha(path) == ref["sha256"], "Root changed after lock")
            root = read(path)
            require(root.get("manifest") == MAIN_BINDING and root.get("stage") == "main"
                    and root["spec"] == spec, "Wrong main root")
            rows.append(analysis.row_from_root(predictions[spec["index"]], root))
            del root
            if len(rows) % 120 == 0:
                print(f"Calculated {len(rows)}/720 roots", flush=True)
        new_json(execution / "rows.json", {"binding": binding, "rows": rows})
        summary = analysis.statistics(rows, boundary=boundary)
        new_json(execution / "result.json", {"binding": binding, "time": now(), "summary": summary,
                 "post_hoc": True, "confirmatory": False, "selector_worlds": 64,
                 "new_simulations": 0, "model_inference": False, "training": False,
                 "original_assessment_outcomes_used": False, "automatic_successor": False})
        print("Primary arithmetic complete; independently reconstructing all choices", flush=True)
        audit_rows = []
        for row, spec, ref in zip(rows, specs, roots, strict=True):
            boundary()
            path = ROOT / ref["path"]
            require(sha(path) == ref["sha256"], "Root changed before independent calculation")
            root = read(path)
            audited = independent.reconstruct(predictions[spec["index"]], root)
            require(audited == row, "Independent root arithmetic mismatch: " + str(spec["index"]))
            audit_rows.append(audited)
            del root
        verification = independent.verify_statistics(audit_rows, summary, boundary=boundary)
        new_json(execution / "independent_check.json", {"binding": binding, "time": now(),
                 "all_root_rows_exact": True, "roots": 720, "statistics": verification,
                 "new_trajectory_replay": False})
        require(inventory() == sources and runtime() == lock["runtime"], "Runtime or source changed")
        inputs.revalidate()
        boundary()
        new_json(execution / "completion.json", {"complete": True, "binding": binding, "time": now(),
                 "elapsed_seconds": time.monotonic() - started, "peak_sampled_rss_bytes": peak,
                 "all_input_hashes_rechecked": True, "input_files": len(inputs.files),
                 "frozen_sources_unchanged": True, "original_audits_reused": True,
                 "new_simulations": 0, "new_training": False, "automatic_successor": False,
                 "result": reference(execution / "result.json"),
                 "rows": reference(execution / "rows.json"),
                 "independent_check": reference(execution / "independent_check.json"),
                 "execution_lock": reference(execution / "execution_lock.json")})
        print("Diagnostic complete; independent arithmetic and source rechecks passed", flush=True)


def supervised(command, directory, *, scientific):
    import psutil
    started = time.monotonic()
    parent = configure()
    BASE.mkdir(parents=True, exist_ok=True)
    if scientific:
        require(not any(BASE.glob("*/execution_lock.json")), "A diagnostic execution was already locked; preserve it")
    lease_path = BASE / "active.json"
    lease = {"pid": parent.pid, "created": parent.create_time(), "token": secrets.token_hex(16)}
    # This lease covers both preparation and scientific execution, independent
    # of output directory names. A crashed parent's lease remains for review.
    new_json(lease_path, lease)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except BaseException:
        if read(lease_path) == lease:
            lease_path.unlink()
        raise
    log_path = directory / ("worker.log" if scientific else "pytest.log")
    peak = 0
    failure = None
    process, entry = None, None
    try:
        entry = resource_check(started, entry=True)
        if scientific:
            command = command + ["--supervisor-pid", str(lease["pid"]),
                                 "--supervisor-created", str(lease["created"]), "--worker-token", lease["token"]]
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=os.environ.copy(), creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            target = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    value = resource_check(started, process=target)
                    peak = max(peak, value["rss_bytes"])
                except psutil.NoSuchProcess:
                    if process.poll() is None:
                        raise
                time.sleep(.25)
    except BaseException as exc:
        failure = repr(exc)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    code = process.wait() if process is not None else None
    receipt = {"time": now(), "exit_code": code, "failure": failure,
               "elapsed_seconds": time.monotonic() - started, "peak_sampled_worker_rss_bytes": peak,
               "entry_resources": entry, "lease": lease,
               "log": reference(log_path) if log_path.exists() else None}
    try:
        new_json(directory / "supervisor.json", receipt, control=True)
        if failure or code != 0:
            new_json(directory / "stop.json", {**receipt, "complete": False,
                     "partial_evidence_preserved": True, "automatic_retry": False}, control=True)
            raise RuntimeError(f"Bounded process stopped: {failure or code}; see {log_path}")
    finally:
        require(read(lease_path) == lease, "Active lease changed")
        lease_path.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute", "worker"))
    parser.add_argument("--preparation", default="preparation_20261018_01")
    parser.add_argument("--execution", default="execution_20261018_01")
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--supervisor-created", type=float)
    parser.add_argument("--worker-token")
    args = parser.parse_args()
    require(Path(args.preparation).name == args.preparation and Path(args.execution).name == args.execution,
            "Directory names only")
    preparation, execution = BASE / args.preparation, BASE / args.execution
    if args.command == "prepare":
        sources = inventory()
        receipt = supervised([sys.executable, "-B", "-m", "pytest", str(TESTS), "-q", "-p", "no:cacheprovider"],
                             preparation, scientific=False)
        require(inventory() == sources, "Files changed during synthetic checks")
        new_json(preparation / "checks.json", {"passed": True, "time": now(), "inventory": sources,
                 "runtime": runtime(), "log": receipt["log"], "production_metrics_calculated": False})
        print(json.dumps({"preparation": str(preparation), "passed": True}))
    elif args.command == "execute":
        receipt = supervised([sys.executable, "-B", "-u", "-m", "tools.ranker_planner_bridge_128.run",
                              "worker", "--preparation", args.preparation, "--execution", args.execution],
                             execution, scientific=True)
        require(read(execution / "completion.json").get("complete") is True, "Missing worker completion")
        print(json.dumps({"execution": str(execution), "complete": True,
                          "elapsed_seconds": receipt["elapsed_seconds"]}))
    else:
        require(args.supervisor_pid is not None and args.supervisor_created is not None
                and args.worker_token is not None, "Worker requires an authenticated supervisor")
        supervisor = {"pid": args.supervisor_pid, "created": args.supervisor_created, "token": args.worker_token}
        validate_supervisor(supervisor)
        require(execution.is_dir() and not (execution / "execution_lock.json").exists(),
                "Worker requires fresh supervised directory")
        try:
            worker(execution, preparation, supervisor)
        except BaseException as exc:
            new_json(execution / "worker_failure.json", {"time": now(), "error": repr(exc),
                     "traceback": traceback.format_exc()[-16384:], "complete": False,
                     "automatic_retry": False}, control=True)
            raise


if __name__ == "__main__":
    main()
