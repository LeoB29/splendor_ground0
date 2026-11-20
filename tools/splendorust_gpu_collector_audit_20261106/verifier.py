"""One narrow private saved-audit correction; no collection, models or gameplay."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"outputs/splendorust_gpu_collector_audit_20261106_01"
PLAN = OUT/"plan.json"
ORIGINAL_OUT = ROOT/"outputs/splendorust_gpu_collector_20261105_01"
ORIGINAL_AUDITOR = ROOT/"tools/splendorust_gpu_collector_20261105/audit.py"
ORIGINAL_AUDITOR_SHA = "6a395b8a7b95dbe481ef5c6f1e2029d68de32ff6d2f672be9ef6f5dc248dfc1d"
ORIGINAL_PLAN_SHA = "45f5aaf9bd7d9a5dc485e1eaaa39c5f908edda887b332e83cc71811dd60b3563"
SCHEMA = "splendorust_gpu_collector_saved_audit_recovery_v1"
AUDIT_SECONDS = 3600.
ORIGINAL_SECONDS = 43200.
CONTROLLER_EXIT = dict(session_id=79670, exit_code=0, tool_chunk="72c5a5")
PUBLIC_MODULE = "tools.splendorust_gpu_collector_20261105.audit"


def require(value, message):
    if not value: raise ValueError("COLLECTOR_AUDIT_RECOVERY: "+message)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
def artifact(path):
    path = Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size, sha256=sha(path))
def resolve(item): return ROOT/Path(item["path"])
def verify(item):
    require(type(item) is dict and set(item) == {"path", "bytes", "sha256"}, "exact artifact binding")
    require(artifact(resolve(item)) == item, "frozen artifact unchanged: "+item["path"])
def nested_artifacts(value):
    if type(value) is dict:
        if set(value) == {"path", "bytes", "sha256"}: yield value
        else:
            for child in value.values(): yield from nested_artifacts(child)
    elif type(value) is list:
        for child in value: yield from nested_artifacts(child)
def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()) and not path.is_relative_to(ORIGINAL_OUT.resolve()),
            "write only separate recovery evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    return artifact(path)


def normalized_metrics(metrics):
    """Only integer histogram object keys acquire their JSON representation."""
    require(type(metrics) is dict and type(metrics.get("occupied_rows_histogram")) is dict,
            "original metrics histogram dictionary")
    histogram = metrics["occupied_rows_histogram"]
    require(all(type(key) is int and 1 <= key <= 8 and type(count) is int and count > 0
                for key, count in histogram.items()), "integer occupancy1..8 and positive integer counts; no booleans")
    result = dict(metrics)
    result["occupied_rows_histogram"] = {str(key): count for key, count in histogram.items()}
    return result


class AuditBudget:
    """One original process-start allowance, inside the original study deadline."""
    def __init__(self, monotonic_origin, unix_origin):
        self.monotonic_origin = monotonic_origin; self.unix_origin = unix_origin
    def guard(self, original_claim=None):
        require(0 <= time.monotonic()-self.monotonic_origin < AUDIT_SECONDS
                and 0 <= time.time()-self.unix_origin < AUDIT_SECONDS, "actual one-hour saved-audit dual clock; no reset")
        claim = read(ORIGINAL_OUT/"claim.json") if original_claim is None else original_claim
        require(claim.get("total_seconds", ORIGINAL_SECONDS) == ORIGINAL_SECONDS
                and 0 <= time.monotonic()-claim["monotonic_origin"] < ORIGINAL_SECONDS
                and 0 <= time.time()-claim["unix_origin"] < ORIGINAL_SECONDS,
                "unchanged original twelve-hour study dual clock")
    def evidence(self):
        return dict(monotonic_origin=self.monotonic_origin, unix_origin=self.unix_origin,
                    elapsed_monotonic_seconds=time.monotonic()-self.monotonic_origin,
                    elapsed_unix_seconds=time.time()-self.unix_origin,
                    maximum_audit_seconds=AUDIT_SECONDS, original_total_seconds=ORIGINAL_SECONDS,
                    clocks_reset=False)


def load_private_auditor(budget):
    """The original __file__ remains bound in every historical cost/proof check."""
    require(sha(ORIGINAL_AUDITOR) == ORIGINAL_AUDITOR_SHA, "exact original frozen auditor")
    public = sys.modules.get(PUBLIC_MODULE)
    before = None if public is None else (public.JournalIndex, public.guard_audit_clock, public.c)
    name = "tools.splendorust_gpu_collector_20261105._private_histogram_recovery_20261106"
    specification = importlib.util.spec_from_file_location(name, ORIGINAL_AUDITOR)
    module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)
    original_index, original_guard = module.JournalIndex, module.guard_audit_clock

    class JsonHistogramIndex(original_index):
        def metrics(self, prefix):
            return normalized_metrics(super().metrics(prefix))

    def original_and_recovery_guard(claim=None):
        budget.guard(claim); original_guard(claim)

    module.JournalIndex = JsonHistogramIndex
    module.guard_audit_clock = original_and_recovery_guard
    require(Path(module.__file__).resolve() == ORIGINAL_AUDITOR.resolve(), "preserved original auditor artifact identity")
    require(public is None or (public.JournalIndex, public.guard_audit_clock, public.c) == before,
            "public auditor module untouched")
    return module


def validate_recovery_plan(plan, expected_sha, budget):
    budget.guard()
    require(sha(PLAN) == expected_sha and plan["schema"] == SCHEMA, "exact separate prospective recovery plan")
    require(plan["maximum_audit_seconds"] == AUDIT_SECONDS and plan["original_total_seconds"] == ORIGINAL_SECONDS
            and plan["models"] == plan["games"] == 0, "saved-only original and recovery resource limits")
    require(plan["controller_exit"] == CONTROLLER_EXIT, "actual retained original controller completion")
    for item in nested_artifacts(plan):
        budget.guard(); verify(item)
    require(resolve(plan["original_plan"]).resolve() == ORIGINAL_OUT/"plan.json"
            and plan["original_plan"]["sha256"] == ORIGINAL_PLAN_SHA
            and resolve(plan["original_auditor"]).resolve() == ORIGINAL_AUDITOR
            and plan["original_auditor"]["sha256"] == ORIGINAL_AUDITOR_SHA, "unchanged original plan and auditor")
    controller = read(resolve(plan["controller_execution"]))
    require(all(controller[key] == value for key, value in CONTROLLER_EXIT.items())
            and controller["plan"] == plan["original_plan"], "actual retained exit receipt; not inferred from completion")
    for item in nested_artifacts(controller): verify(item)
    require(resolve(controller["execution"]).resolve() == ORIGINAL_OUT/"execution.json"
            and resolve(controller["completion"]).resolve() == ORIGINAL_OUT/"completion.json", "original actual owner receipts")
    failed = read(resolve(plan["failed_audit"]))
    require(failed["status"] == "saved_auditor_failed" and failed["actual_auditor_exit_code"] == 1
            and failed["final_acceptance"] is False and failed["result_file_exists"] is False
            and failed["auditor"]["sha256"] == ORIGINAL_AUDITOR_SHA
            and failed["expected_plan_sha256"] == ORIGINAL_PLAN_SHA
            and failed["frozen_sources_edited"] is False
            and all(failed["controller_observation"][key] == value for key, value in CONTROLLER_EXIT.items()),
            "original final audit failure remains preserved and unaccepted")
    for item in nested_artifacts(failed): verify(item)
    diagnosis = read(resolve(plan["diagnosis"]))
    require(diagnosis["status"] == "saved_first_cell_discrepancy_reproduced"
            and diagnosis["cell_id"] == "b1_cpu" and diagnosis["models"] == diagnosis["games"] == 0
            and diagnosis["raw_proof_equal"] is False and diagnosis["json_roundtrip_equal"] is True
            and diagnosis["source_unchanged"] == ORIGINAL_AUDITOR_SHA,
            "actual saved-only original mismatch reproduced")
    differences = diagnosis["differences"]
    require(len(differences) == 1 and differences[0]["path"] == "$.backend_metrics.occupied_rows_histogram"
            and differences[0]["actual_keys"] == [dict(type="int", value=1)]
            and differences[0]["saved_keys"] == [dict(type="str", value="1")],
            "only independently observed integer/string histogram-key discrepancy")
    for item in nested_artifacts(diagnosis): verify(item)
    require(read(resolve(diagnosis["minimal_diff"])) == differences
            and read(resolve(diagnosis["recomputed"])) == read(resolve(diagnosis["saved"])),
            "retained diagnostic artifacts reproduce the sole serialization difference")
    tests = read(resolve(plan["tests"]))
    require(tests["exit_code"] == 0 and tests["models"] == tests["games"] == 0,
            "passed source-bound model-free recovery tests")
    for item in nested_artifacts(tests): verify(item)
    require(plan["reviews"], "independent review before recovery")
    for item in plan["reviews"]:
        review = read(resolve(item))
        require(review["approved"] is True and not review["blocking_findings"], "approved independent recovery review")
        for source in review["inputs"]: verify(source)
    require(artifact(__file__) in plan["inputs"] and artifact(__file__) in tests["inputs"],
            "actual correction source is frozen and tested")
    budget.guard()
    return plan


def run_recovery(plan, expected_sha, budget):
    module = load_private_auditor(budget)
    original_plan = read(ORIGINAL_OUT/"plan.json")
    accepted = module.audit_study(original_plan, ORIGINAL_OUT, expected_plan_sha=ORIGINAL_PLAN_SHA,
        controller_session=CONTROLLER_EXIT["session_id"], controller_exit_code=CONTROLLER_EXIT["exit_code"],
        controller_chunk=CONTROLLER_EXIT["tool_chunk"])
    require(accepted["status"] == "saved_collector_comparison_verified"
            and accepted["auditor"] == plan["original_auditor"], "complete strict original acceptance after narrow key correction")
    require("torch" not in sys.modules, "saved-only verification imports no model runtime")
    budget.guard()
    for item in plan["inputs"]: verify(item)
    result = dict(accepted)
    result.update(status="recovered_saved_collector_comparison_verified",
        original_verifier_status=accepted["status"], original_final_audit_passed=False,
        corrected_verifier=artifact(__file__), recovery_plan=artifact(PLAN),
        preserved_original_failure=plan["failed_audit"], saved_diagnosis=plan["diagnosis"],
        correction_scope="Only original metrics occupancy histogram integer keys converted to their JSON string form; all values, proof equality and thresholds unchanged",
        recovery_execution=budget.evidence(), recovery_models_launched=0, recovery_games_launched=0,
        recovery_plan_sha256=expected_sha, original_cost_auditor_bindings_preserved=True)
    budget.guard()
    return result


def main():
    monotonic_origin, unix_origin = time.monotonic(), time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-plan", required=True); parser.add_argument("--recovery-plan-sha", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    destination = Path(args.output).resolve()
    require(Path(args.recovery_plan).resolve() == PLAN.resolve() and destination == OUT/"recovered_audit.json",
            "fixed separate recovery plan/output; no frozen evidence edits")
    budget = AuditBudget(monotonic_origin, unix_origin)
    stem = destination.with_suffix("")
    def receipt(kind): return stem.with_name(stem.name+"_"+kind+".json")
    for kind in ("launch", "failure", "cleanup", "exit", "runtime_sources"):
        require(not receipt(kind).exists(), "one exclusive recovery; no duplicate launch or retry")
    require(not destination.exists(), "no replacement recovered result")
    import psutil
    created = psutil.Process().create_time()
    save(receipt("launch"), dict(schema="collector_saved_audit_recovery_launch_v1", time=now(), pid=os.getpid(),
        created=created,
        recovery_plan_path=str(PLAN), expected_recovery_plan_sha256=args.recovery_plan_sha,
        output=str(destination), models=0, games=0, **budget.evidence()))
    status = "failed"; result = None; failure = None; sources = None
    try:
        from tools.future_training_rules_20261021 import verify_runtime_sources
        sources = verify_runtime_sources()  # Exactly once at this new saved evaluator's startup.
        budget.guard()
        save(receipt("runtime_sources"), dict(sources=sources, models=0, games=0))
        plan = validate_recovery_plan(read(PLAN), args.recovery_plan_sha, budget)
        require(read(ORIGINAL_OUT/"plan.json")["source_guard"] == sources, "original corrected runtime sources")
        result = run_recovery(plan, args.recovery_plan_sha, budget)
        budget.guard(); save(destination, result); budget.guard()
        status = "recovered_saved_collector_comparison_verified"
    except BaseException:
        failure = traceback.format_exc()
        save(receipt("failure"), dict(status="saved_recovery_failed", time=now(), error=failure,
            original_failure_preserved=True, result_file_exists=destination.exists(), models=0, games=0, **budget.evidence()))
    finally:
        save(receipt("cleanup"), dict(time=now(), success=failure is None, owned_model_processes=[], owned_game_processes=[],
            frozen_sources_edited=False, original_failure_preserved=True, models=0, games=0))
    code = 0 if failure is None else 1
    save(receipt("exit"), dict(time=now(), status=status, exit_code=code, pid=os.getpid(), created=created,
        result=None if result is None or not destination.exists() else artifact(destination),
        actual_process_exit_requires_parent_observation=True, models=0, games=0, **budget.evidence()))
    print(json.dumps(dict(status=status, exit_code=code, output=str(destination), models=0, games=0)))
    return code


if __name__ == "__main__": raise SystemExit(main())
