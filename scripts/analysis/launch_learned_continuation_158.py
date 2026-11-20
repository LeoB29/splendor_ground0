"""Durable full-test evidence and one-shot158 launch; no automatic resume."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

from splendor_ai.alphazero import run_learned_continuation as p


def tests():
    before = p.identity()
    folder = p.OUTPUT/"preparation"
    folder.mkdir(parents=True, exist_ok=True)
    started, wall = p.old.now(), time.perf_counter()
    with (folder/"tests.txt").open("x", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], stdout=log, stderr=subprocess.STDOUT)
    assert p.identity() == before, "Identity changed during full tests"
    p.old.save_new(folder/"tests.json", {"identity": before, "full_repository": True,
                   "exit_code": result.returncode, "started": started, "finished": p.old.now(),
                   "elapsed_seconds": time.perf_counter()-wall, "log_sha256": p.old.file_hash(folder/"tests.txt"),
                   "wrapper_sha256": p.old.file_hash(Path(__file__))})
    print("\n".join((folder/"tests.txt").read_text(encoding="utf-8").splitlines()[-20:]), flush=True)
    return result.returncode


def launch():
    folder = p.OUTPUT/"preparation"
    with p.old.process_lock(folder/"launcher.lock"):
        assert not (folder/"process.json").exists(), "Launcher already used"
        assert not (p.OUTPUT/"manifest.json").exists(), "Existing run needs explicit review"
        p.verify_test_evidence()
        started, wall = p.old.now(), time.perf_counter()
        with (folder/"run.txt").open("x", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "run_learned_continuation_158.py", "start"],
                                       stdout=log, stderr=subprocess.STDOUT)
            p.old.save_new(folder/"process.json", {"wrapper_pid": os.getpid(), "pipeline_pid": process.pid,
                          "started": started, "launcher_sha256": p.old.file_hash(Path(__file__)),
                          "test_evidence_sha256": p.old.file_hash(folder/"tests.json")})
            code = process.wait()
        result = {"exit_code": code, "finished": p.old.now(), "elapsed_seconds": time.perf_counter()-wall,
                  "log_sha256": p.old.file_hash(folder/"run.txt")}
        if code == 0:
            result["manifest"] = p.validate_freeze()
            final = p.old.read_json(p.OUTPUT/"final_status.json")
            assert final["complete"] and final["integrity_pass"]
            assert final["result_sha256"] == p.old.file_hash(p.OUTPUT/"result.json")
            assert final["audit_sha256"] == p.old.file_hash(p.OUTPUT/"main_audit.json")
            result.update(final_status_sha256=p.old.file_hash(p.OUTPUT/"final_status.json"), full_audit_complete=True)
        p.old.save_new(folder/"exit.json", result)
        return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("tests", "launch"))
    args = parser.parse_args()
    raise SystemExit(tests() if args.command == "tests" else launch())
