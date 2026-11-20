"""One-shot, bounded correctness/timing qualification with durable failure receipts."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from tools.local_player_qualification_128 import common as c
from tools.local_player_qualification_128.clock import PersistentWorker, WorkerInitializationError, processor_topology
from tools.local_player_qualification_128.plan import build_cases, summarize
from tools.local_player_qualification_128.selection import select_saved_roots
from tools.local_player_qualification_128.verify import compare, completed_work, expected

TESTS = ["tests/test_local_player_qualification_adapter.py",
         "tests/test_local_player_qualification_clock.py",
         "tests/test_local_player_qualification_plan.py",
         "tests/test_local_player_qualification_verify.py",
         "tests/test_external_strength_adapter.py"]


def prepare(base):
    base.mkdir(parents=True, exist_ok=False)
    try:
        original = c.frozen_source_check()
        c.save_new(base / "original_source_check.json", original)
        if original["failures"]:
            raise ValueError("Original experiment dependencies have changed")
        selection = select_saved_roots()
        # These files precede any saved decision/outcome read or model inference.
        c.save_new(base / "selection.json", selection)
        entries = selection["core"] + selection["edges"]
        c.save_new(base / "schedule.json", build_cases(entries))
        topology = processor_topology()
        if topology["api"] != "GetLogicalProcessorInformationEx(RelationProcessorCore)":
            raise ValueError("This execution requires verified Windows physical topology")
        if len(topology["distinct_efficiency_classes"]) < 2:
            raise ValueError("Cannot distinguish performance and efficiency cores")
        command = [sys.executable, "-B", "-m", "pytest", "-q", *TESTS, "-p", "no:cacheprovider"]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        start = time.monotonic()
        test = subprocess.run(command, cwd=c.ROOT, capture_output=True, text=True, env=env,
                              timeout=240, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        c.save_new(base / "preflight_tests.json", {"command": command, "returncode": test.returncode,
                   "stdout": test.stdout, "stderr": test.stderr, "seconds": time.monotonic() - start})
        if test.returncode:
            raise ValueError("Synthetic qualification tests failed")
        registry = c.read(c.ROOT / "outputs/proposal_breadth_arena_128/models_lock.json")
        declared = c.read(c.ROOT / "outputs/proposal_breadth_arena_128/manifest.json")["identity"]["dependencies"]
        paths = set(declared)
        paths.update(str(p.relative_to(c.ROOT)).replace("\\", "/") for p in
                     (c.ROOT / "tools/local_player_qualification_128").glob("*.py"))
        paths.update(TESTS)
        paths.update({"docs/engineering/local_player_qualification_20261018.md",
                      "outputs/proposal_breadth_arena_128/manifest.json",
                      "tools/external_strength_qualification/adapter.py",
                      "outputs/external_strength_preparation_20261018_01/qualification_catalog_check_01.json"})
        from tools.external_strength_qualification.adapter import UPSTREAM_CATALOG_PATH
        paths.add(UPSTREAM_CATALOG_PATH)
        paths.update(item["path"] for item in selection["source_artifacts"])
        paths.update(item["path"] for item in entries)
        paths.update(item["path"] for item in registry["models"].values())
        paths.add(registry["continuation_model"]["path"])
        paths.update(str((base / name).relative_to(c.ROOT)).replace("\\", "/") for name in
                     ("selection.json", "schedule.json", "preflight_tests.json", "original_source_check.json"))
        items = [c.artifact(c.ROOT / name) for name in sorted(paths)]
        c.save_new(base / "freeze.json", {"time": c.now(), "owner_authorized": True,
                   "inputs": items, "topology": topology, "python": sys.version,
                   "executable": sys.executable, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("torch", "numpy", "psutil", "pytest", "threadpoolctl")},
                   "limits": {"seconds": c.MAX_SECONDS, "bytes": c.MAX_BYTES,
                              "worker_rss_bytes": c.MAX_WORKER_RSS, "disk_reserve_bytes": c.MIN_FREE},
                   "cases": len(build_cases(entries)), "strength_games_authorized": 0,
                   "external_screen_conditional": True, "larger_data_deferred": True})
        print(c.canonical({"prepared": str(base), "cases": len(build_cases(entries)),
                           "core_positions": len(selection["core"]), "edge_positions": len(selection["edges"])}), flush=True)
    except BaseException:
        c.save_new(base / "preparation_failure.json", {"time": c.now(), "failure": traceback.format_exc()})
        raise


def external_check(base):
    from tools.external_strength_qualification.adapter import qualify_sources
    receipt = qualify_sources(c.ROOT)
    # Independently build the engine's live catalogue rather than reuse its AST parser.
    import ast
    from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles
    external_path = c.ROOT / receipt["sources"][1]["path"]
    tree = ast.parse(external_path.read_text(encoding="utf-8"))
    literal = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("CARDS", "NOBLES"):
                    literal[target.id] = ast.literal_eval(node.value)
    colors = ("white", "blue", "green", "red", "black")
    faces = {(tier, color, points, tuple(cost.get(k, 0) for k in colors))
             for color, cost, tier, points in literal["CARDS"].values()}
    missing = [card.card_id for cards in build_base_deck_by_tier().values() for card in cards
               if (card.tier, card.bonus_color, card.points, tuple(card.cost.get(k, 0) for k in colors)) not in faces]
    nobles = {tuple(cost.get(k, 0) for k in colors) for _, cost in literal["NOBLES"]}
    missing_nobles = [n.noble_id for n in build_base_nobles()
                      if tuple(n.requirement.get(k, 0) for k in colors) not in nobles]
    if set(missing) != set(receipt["unmatched_local_cards"]) or missing_nobles:
        raise ValueError("Independent component comparison disagrees")
    receipt["independent_live_engine_check"] = {"matched_cards": 90 - len(missing),
             "unmatched_card_ids": missing, "matched_nobles": 10 - len(missing_nobles), "agrees": True}
    receipt["screen_status"] = "blocked_catalog_mismatch" if missing else "requires_remaining_adapter_gates"
    c.save_new(base / "external_qualification.json", receipt)
    return receipt


def one_attempt(base, case, entry, root, payload, wanted, deadline, cpu):
    start = time.monotonic()
    row = {"case": case, "root_id": entry["root_id"], "fit": entry["fit"],
           "original_sha256": entry["sha256"], "time": c.now(),
           "observer_sha256": c.digest(payload), "expected_sha256": c.digest(wanted), "correct": False}
    folder = base / "attempts" / f'{case["index"]:03d}'
    try:
        row["resources_before"] = c.boundary(base, deadline)
        if deadline - time.monotonic() < 60 + case["allowance_seconds"] + 60:
            raise c.ResourceHold("Insufficient remaining time for complete attempt and finalization")
        env = {"PYTHONPATH": str(c.ROOT / "src") + os.pathsep + str(c.ROOT), "CUDA_VISIBLE_DEVICES": ""}
        with PersistentWorker("tools.local_player_qualification_128.worker", folder,
                affinity=[cpu], initialization_timeout=60, env=env,
                max_message_bytes=2 * 1024**2, max_log_bytes=2 * 1024**2,
                max_rss_bytes=c.MAX_WORKER_RSS, absolute_deadline=deadline - 60,
                resource_guard=lambda: c.boundary(base, deadline, memory_headroom=False)) as worker:
            row["initialization"] = {k: v for k, v in worker.initialization.items() if k != "result"}
            legal = wanted["record"]["proposal"]["legal_actions"]
            clock = worker.request({"observer": payload, "breadth": case["breadth"]},
                                   case["allowance_seconds"],
                                   validate=lambda r: isinstance(r, dict) and r.get("action") in legal)
            row["clock"] = {key: value for key, value in clock.items() if key != "result"}
            if clock["status"] == "ok":
                result = clock["result"]
                compare(result, wanted)
                row.update(correct=True, result_sha256=c.digest(result), work=completed_work(result))
            elif clock["status"] == "illegal":
                raise ValueError("Completed action violates original legal-action set")
            elif clock["status"] in ("rss_limit", "log_limit", "message_flood", "affinity_changed", "resource_guard"):
                raise c.ResourceHold("Worker resource boundary: " + clock["status"])
    except WorkerInitializationError as error:
        row["clock"] = {**{k: v for k, v in error.receipt.items() if k != "result"},
                        "status": "initialization_" + error.receipt["status"]}
        row["failure"] = str(error)
        if error.receipt["status"] in ("rss_limit", "log_limit", "message_flood", "affinity_changed", "resource_guard"):
            raise c.ResourceHold("Initialization resource boundary: " + error.receipt["status"])
    except BaseException:
        row["failure"] = traceback.format_exc()
        if "clock" not in row:
            row["clock"] = {"status": "referee_or_resource_failure", "accepted": False,
                            "receipt_elapsed_seconds": None}
        raise
    finally:
        row["total_seconds"] = time.monotonic() - start
        c.save_new(base / "rows" / f'{case["index"]:03d}.json', row)
    return row


def independent_receipt_check(base, entries):
    checked, completed, worlds, continuations = 0, 0, 0, 0
    schedule = c.read(base / "schedule.json")
    for path in sorted((base / "rows").glob("*.json")):
        row = c.read(path)
        case = row["case"]
        if case != schedule[checked] or path.stem != f'{checked:03d}':
            raise ValueError("Recorded case differs from the fixed schedule")
        entry = entries[case["root_index"]]
        if (row["root_id"] != entry["root_id"] or row["fit"] != entry["fit"]
                or row["original_sha256"] != entry["sha256"]):
            raise ValueError("Attempt/root/model linkage mismatch")
        checked += 1
        if not row.get("correct"):
            continue
        original = c.read(c.ROOT / entry["path"])
        saved = c.read(base / "attempts" / f'{case["index"]:03d}' / "request-00000001.json")
        if saved["status"] != "ok" or not saved["received_before_deadline"]:
            raise ValueError("Accepted result without timely trusted receipt")
        if not 0 <= saved["receipt_elapsed_seconds"] <= case["allowance_seconds"]:
            raise ValueError("Accepted receipt exceeds frozen allowance")
        # The adapter's separately implemented original-record projection must
        # also agree with the independently calculated referee expectation.
        from tools.local_player_qualification_128.local import expected_from_saved
        target = expected_from_saved(original, case["breadth"])
        compare(target, expected(original, case["breadth"]))
        compare(saved["result"], target)
        if c.digest(saved["result"]) != row["result_sha256"]:
            raise ValueError("Row/result linkage mismatch")
        work = completed_work(saved["result"])
        completed += 1
        worlds += work["worlds"]
        continuations += work["continuations"]
    return {"rows_checked": checked, "completed_decisions_checked": completed,
            "replayed_worlds": worlds, "replayed_continuations": continuations,
            "all_completed_decisions_match_originals": True,
            "shared_dependencies": "Frozen numerical models, game engine and trajectory primitives"}


def _run(base):
    lock = c.read(base / "freeze.json")
    c.verify(lock["inputs"])
    c.save_new(base / "launch.json", {"time": c.now(), "epoch_seconds": time.time(),
               "freeze": c.artifact(base / "freeze.json"), "pid": os.getpid(),
               "maximum_seconds": c.MAX_SECONDS})
    started = time.monotonic()
    deadline = started + c.MAX_SECONDS
    rows, failure, external, independent = [], None, None, None
    selection = c.read(base / "selection.json")
    entries = selection["core"] + selection["edges"]
    cases = c.read(base / "schedule.json")
    try:
        c.boundary(base, deadline)
        from tools.local_player_qualification_128 import local
        roots, payloads, variants = [], [], []
        privacy = []
        for entry in entries:
            if c.file_hash(c.ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("Selected original record changed")
            root = c.read(c.ROOT / entry["path"])
            roots.append(root)
            payload = local.make_observer_payload(root["snapshot"], entry["context"])
            variant = local.make_observer_payload(local.hidden_variant(root["snapshot"]), entry["context"])
            if c.canonical(payload) != c.canonical(variant):
                raise ValueError("Hidden true state leaked across observer boundary")
            payloads.append(payload)
            variants.append(variant)
            privacy.append({"root_id": entry["root_id"], "payload_sha256": c.digest(payload),
                            "hidden_variant_payload_sha256": c.digest(variant), "identical": True})
        c.save_new(base / "privacy_inputs.json", privacy)
        for case in cases:
            index = case["root_index"]
            wanted = expected(roots[index], case["breadth"])
            row = one_attempt(base, case, entries[index], roots[index],
                              variants[index] if case["hidden_variant"] else payloads[index],
                              wanted, deadline, lock["topology"]["selected_logical_processor"])
            if case["kind"] == "repeat" and row["correct"]:
                previous = next(r for r in rows if r["case"]["root_index"] == index
                                and r["case"]["breadth"] == case["breadth"]
                                and r["case"]["allowance_seconds"] == 120 and r["case"]["kind"] == "core")
                if previous["correct"] and row["result_sha256"] != previous["result_sha256"]:
                    raise ValueError("Identical observation/RNG repetition changed scientific output")
            rows.append(row)
            c.boundary(base, deadline, memory_headroom=False)
            print(c.canonical({"case": case["index"] + 1, "total": len(cases),
                  "root": row["root_id"], "fit": row["fit"], "candidates": 8 if case["breadth"] else 4,
                  "allowance": case["allowance_seconds"], "status": row["clock"]["status"],
                  "correct": row["correct"], "receipt_seconds": row["clock"]["receipt_elapsed_seconds"],
                  "elapsed_minutes": (time.monotonic() - started) / 60}), flush=True)
        independent = independent_receipt_check(base, entries)
        c.save_new(base / "independent_check.json", independent)
    except BaseException:
        failure = traceback.format_exc()
        c.save_new(base / "failure.json", {"time": c.now(), "failure": failure})
    finally:
        # Preserve every attempt, including the row written by a failed comparator.
        rows = [c.read(path) for path in sorted((base / "rows").glob("*.json"))]
        try:
            external = external_check(base)
            c.verify(lock["inputs"])
            preserved = c.frozen_source_check()
            c.save_new(base / "final_input_check.json", {"inputs_checked": len(lock["inputs"]),
                       "unchanged": True, "original": preserved})
            if preserved["failures"]:
                raise ValueError("Original dependency changed during qualification")
            c.boundary(base, deadline, memory_headroom=False)
        except BaseException:
            failure = (failure or "") + "\nFinalization failure:\n" + traceback.format_exc()
            c.save_new(base / "finalization_failure.json", {"failure": failure})
        summary = summarize(rows)
        result = {"time": c.now(), "complete_schedule": len(rows) == len(cases),
                  "scientifically_accepted": failure is None and independent is not None,
                  "attempted": len(rows), "prescribed": len(cases), "failure": failure,
                  "summary": summary,
                  "per_fit": {str(fit): summarize([r for r in rows if r["fit"] == fit])
                              for fit in (161001, 161002, 161003)},
                  "verification": independent, "elapsed_seconds": time.monotonic() - started,
                  "artifacts_bytes_before_result": c.directory_bytes(base),
                  "external_status": external["screen_status"] if external else "qualification_failed",
                  "external_games": 0, "larger_data_deferred": True,
                  "limits": "Finite saved-position timing/correctness only; no new strength evidence"}
        c.save_new(base / "result.json", result)
        c.save_new(base / "completion.json", {"time": c.now(), "result": c.artifact(base / "result.json"),
                   "status": "completed" if failure is None else "preserved_failure",
                   "worker_processes_closed": True})
        print(c.canonical({"finished": str(base), "attempted": len(rows), "failure": failure,
                           "external_status": result["external_status"]}), flush=True)
    return 0 if failure is None else 1


def run(base):
    with c.execution_lease():
        return _run(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=c.OUTPUT)
    args = parser.parse_args()
    base = args.output.resolve()
    if not base.is_relative_to((c.ROOT / "outputs").resolve()):
        raise ValueError("Qualification evidence must remain in the workspace outputs directory")
    if args.action == "prepare":
        prepare(base)
    else:
        raise SystemExit(run(base))
