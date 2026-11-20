"""Independent alternative-ranker preparation; never changes the running study."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"


def main():
    import argparse
    import json
    import subprocess
    import threading
    import time
    import xml.etree.ElementTree as ET
    from tools.alternative_ranker_128 import common as c
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "fit", "predict", "evaluate", "identity"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    c.configure_compute()
    if args.command == "preflight":
        folder = ROOT / "outputs" / ("alternative_ranker_128_preflight_" + time.strftime("%Y%m%d_%H%M%S"))
        folder.mkdir(parents=True, exist_ok=False)
        before = c.own_inventory()
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        xml = folder / "tests.xml"
        start = time.monotonic()
        command = [sys.executable, "-m", "pytest", "tools/alternative_ranker_128", "-q",
                   "-p", "no:cacheprovider", "--basetemp", str(folder / "tmp"), "--junitxml", str(xml)]
        with (folder / "stdout.txt").open("x", encoding="utf-8") as out, (folder / "stderr.txt").open("x", encoding="utf-8") as err:
            done = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, timeout=900,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        cases = ET.parse(xml).getroot().findall(".//testcase") if xml.exists() else []
        report = {"time": c.now(), "passed": done.returncode == 0 and bool(cases) and before == c.own_inventory(),
                  "tests": len(cases), "seconds": time.monotonic() - start,
                  "returncode": done.returncode, "source_inventory": before,
                  "stdout": c.artifact(folder / "stdout.txt"), "stderr": c.artifact(folder / "stderr.txt")}
        c.new_json(folder / "checks.json", report)
        print(json.dumps({"passed": report["passed"], "tests": len(cases), "checks": str(folder / "checks.json")}))
        return 0 if report["passed"] else 1
    if args.command == "freeze":
        c.require(args.checks is not None, "Supply successful exact-source preflight receipt")
        print(json.dumps(c.register(args.checks)))
        return 0
    freeze = c.validate_freeze()
    binding = freeze["binding"]
    if args.command == "identity":
        print(json.dumps({"binding": binding, "valid": True, "fresh_outcomes_read": False,
                          "resources": c.resource_snapshot()}))
        return 0
    if args.command == "fit":
        from threadpoolctl import threadpool_limits
        from tools.alternative_ranker_128 import learning
        c.require(not (c.OUTPUT / "fit_launch.json").exists(), "One-shot fitting already launched; preserve evidence")
        resources = c.resource_gate()
        c.ongoing_resource_gate()
        c.new_json(c.OUTPUT / "fit_launch.json", {"manifest": binding, "time": c.now(),
                   "pid": os.getpid(), "resources": resources, "fresh_outcomes_read": False})
        stopped = threading.Event()
        def timeout_guard():
            deadline = time.monotonic() + c.RESOURCE_CONFIG["maximum_seconds"]
            while not stopped.wait(c.RESOURCE_CONFIG["guard_interval_seconds"]):
                try:
                    c.require(time.monotonic() <= deadline, "Two-hour own fitting ceiling")
                    c.ongoing_resource_gate()
                except BaseException as exc:
                    c.new_json(c.OUTPUT / "operational_hold.json", {"manifest": binding, "time": c.now(),
                               "reason": str(exc), "current_study_untouched": True})
                    os._exit(75)
        threading.Thread(target=timeout_guard, daemon=True).start()
        try:
            manifest = c.verified(freeze["manifest"])
            with threadpool_limits(limits=1):
                dataset = learning.prepare_dataset(manifest["source"]["train_files"], c.OUTPUT / "training")
                result_ref = learning.fit_and_save(dataset, c.OUTPUT / "fitted")
                result = c.verified(result_ref)
            c.validate_freeze()
            c.new_json(c.OUTPUT / "models_lock.json", {"manifest": binding, "time": c.now(),
                "models": {name: entry["model"] for name, entry in result["models"].items()}, "learning": result_ref,
                "dataset": dataset, "fresh_outcomes_read": False,
                "both_recipes_preserved": True})
            print(json.dumps({"complete": True, "models_lock": str(c.OUTPUT / "models_lock.json")}))
        except BaseException as exc:
            c.new_json(c.OUTPUT / "fit_failure.json", {"manifest": binding, "time": c.now(),
                       "type": type(exc).__name__, "error": str(exc), "current_study_untouched": True})
            raise
        finally:
            stopped.set()
        return 0
    if args.command == "predict":
        from tools.alternative_ranker_128 import prediction
        print(json.dumps(prediction.create_predictions(repo_root=ROOT, base=c.OUTPUT,
            study_root=c.STUDY_ROOT, manifest=binding)))
        return 0
    if args.command == "evaluate":
        from tools.alternative_ranker_128.evaluation import evaluate_sealed
        result = evaluate_sealed(c.OUTPUT / "predictions.json", c.OUTPUT / "predictions_seal.json",
            repo_root=ROOT, study_root=c.STUDY_ROOT, manifest=binding, study_binding=c.STUDY_BINDING)
        c.new_json(c.OUTPUT / "result.json", result)
        print(json.dumps({"result": str(c.OUTPUT / "result.json")}))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
