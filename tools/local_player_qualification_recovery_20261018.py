"""Explicit memory-only supervisor recovery; frozen player and clock code stay intact."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from tools.local_player_qualification_128 import common as c
from tools.local_player_qualification_128.runner import (
    one_attempt, independent_receipt_check, external_check,
)
from tools.local_player_qualification_128.plan import summarize
from tools.local_player_qualification_128.verify import expected

PARENT = c.ROOT / "outputs/local_player_qualification_20261018_01"
OUTPUT = c.ROOT / "outputs/local_player_qualification_20261018_recovery_01"


def parent_check():
    lock = c.read(PARENT / "freeze.json")
    c.verify(lock["inputs"])
    result = c.read(PARENT / "result.json")
    if (result["scientifically_accepted"] or "Less than one GiB physical memory" not in result["failure"]
            or (PARENT / "attempts").exists()):
        raise ValueError("Recovery is restricted to the documented pre-worker memory hold")
    if len(list((PARENT / "rows").glob("*.json"))) != 1:
        raise ValueError("Unexpected parent attempt history")
    return lock


def prepare():
    lock = parent_check()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        selection = c.read(PARENT / "selection.json")
        cases = c.read(PARENT / "schedule.json")
        c.save_new(OUTPUT / "selection.json", selection)
        c.save_new(OUTPUT / "schedule.json", cases)
        # This short-lived preparation process exits before the timed supervisor
        # starts. Neither local.initialize nor any fitted policy is called.
        from tools.local_player_qualification_128 import local
        privacy = []
        for index, entry in enumerate(selection["core"] + selection["edges"]):
            root = c.read(c.ROOT / entry["path"])
            payload = local.make_observer_payload(root["snapshot"], entry["context"])
            changed = local.hidden_variant(root["snapshot"])
            variant = local.make_observer_payload(changed, entry["context"])
            if c.canonical(payload) != c.canonical(variant):
                raise ValueError("Hidden state changes the prepared lawful observation")
            c.save_new(OUTPUT / "observations" / f"{index:02d}.json", payload)
            c.save_new(OUTPUT / "variants" / f"{index:02d}.json", variant)
            privacy.append({"root_id": entry["root_id"], "payload_sha256": c.digest(payload),
                            "hidden_variant_payload_sha256": c.digest(variant), "identical": True,
                            "hidden_state_actually_changed": changed != root["snapshot"]})
        c.save_new(OUTPUT / "privacy_inputs.json", privacy)
        paths = [Path(__file__).resolve(), c.ROOT / "docs/engineering/local_player_qualification_recovery_20261018.md",
                 PARENT / "freeze.json", PARENT / "launch.json", PARENT / "completion.json",
                 PARENT / "result.json", PARENT / "rows/000.json"]
        paths += list(OUTPUT.rglob("*.json"))
        inputs = lock["inputs"] + [c.artifact(path) for path in paths]
        c.save_new(OUTPUT / "freeze.json", {"time": c.now(), "owner_authorized": True,
                   "inputs": inputs, "topology": lock["topology"], "parent": str(PARENT),
                   "original_launch_epoch": c.read(PARENT / "launch.json")["epoch_seconds"],
                   "maximum_seconds_from_original_launch": c.MAX_SECONDS,
                   "recovery_change": "Release observation-conversion Torch process before timed workers; stream original records one at a time",
                   "changed_player_or_clock": False, "cases": len(cases)})
        print(c.canonical({"prepared": str(OUTPUT), "observations": len(privacy),
                           "production_model_initializations": 0, "decisions": 0}), flush=True)
    except BaseException:
        c.save_new(OUTPUT / "preparation_failure.json", {"failure": traceback.format_exc()})
        raise


def _run():
    lock = c.read(OUTPUT / "freeze.json")
    c.verify(lock["inputs"])
    parent_check()
    if "torch" in sys.modules:
        raise ValueError("Timed supervisor must not import Torch before workers")
    elapsed_before = time.time() - lock["original_launch_epoch"]
    remaining = c.MAX_SECONDS - elapsed_before
    if remaining < 240:
        raise c.ResourceHold("Original four-hour window has insufficient time left")
    started = time.monotonic()
    deadline = started + remaining
    c.save_new(OUTPUT / "launch.json", {"time": c.now(), "epoch_seconds": time.time(), "pid": os.getpid(),
               "freeze": c.artifact(OUTPUT / "freeze.json"), "parent": str(PARENT),
               "remaining_seconds_in_original_four_hour_window": remaining,
               "torch_loaded_in_supervisor": False, "resources": c.boundary(OUTPUT, deadline)})
    entries_manifest = c.read(OUTPUT / "selection.json")
    entries = entries_manifest["core"] + entries_manifest["edges"]
    cases = c.read(OUTPUT / "schedule.json")
    rows, failure, independent, external = [], None, None, None
    try:
        for case in cases:
            if "torch" in sys.modules:
                raise ValueError("Torch entered the timed supervisor")
            index = case["root_index"]
            entry = entries[index]
            root = c.read(c.ROOT / entry["path"])
            wanted = expected(root, case["breadth"])
            location = "variants" if case["hidden_variant"] else "observations"
            payload = c.read(OUTPUT / location / f"{index:02d}.json")
            row = one_attempt(OUTPUT, case, entry, root, payload, wanted, deadline,
                              lock["topology"]["selected_logical_processor"])
            if case["kind"] == "repeat" and row["correct"]:
                previous = next(r for r in rows if r["case"]["root_index"] == index
                                and r["case"]["breadth"] == case["breadth"]
                                and r["case"]["allowance_seconds"] == 120 and r["case"]["kind"] == "core")
                if previous["correct"] and row["result_sha256"] != previous["result_sha256"]:
                    raise ValueError("Repeated lawful input changed scientific output")
            rows.append(row)
            c.boundary(OUTPUT, deadline, memory_headroom=False)
            print(c.canonical({"case": case["index"] + 1, "total": len(cases), "root": entry["root_id"],
                  "fit": entry["fit"], "candidates": 8 if case["breadth"] else 4,
                  "allowance": case["allowance_seconds"], "status": row["clock"]["status"],
                  "correct": row["correct"], "receipt_seconds": row["clock"]["receipt_elapsed_seconds"],
                  "elapsed_minutes": (time.monotonic() - started) / 60}), flush=True)
            del root, wanted, payload
        # Torch may load here for the independent original-record projection:
        # timed workers have all finished and no more player is launched.
        independent = independent_receipt_check(OUTPUT, entries)
        c.save_new(OUTPUT / "independent_check.json", independent)
    except BaseException:
        failure = traceback.format_exc()
        c.save_new(OUTPUT / "failure.json", {"time": c.now(), "failure": failure})
    finally:
        rows = [c.read(path) for path in sorted((OUTPUT / "rows").glob("*.json"))]
        try:
            external = external_check(OUTPUT)
            c.verify(lock["inputs"])
            preserved = c.frozen_source_check()
            if preserved["failures"]:
                raise ValueError("Frozen original dependencies changed")
            c.save_new(OUTPUT / "final_input_check.json", {"inputs_checked": len(lock["inputs"]),
                       "unchanged": True, "original": preserved})
            c.boundary(OUTPUT, deadline, memory_headroom=False)
        except BaseException:
            failure = (failure or "") + "\nFinalization failure:\n" + traceback.format_exc()
            c.save_new(OUTPUT / "finalization_failure.json", {"failure": failure})
        result = {"time": c.now(), "complete_schedule": len(rows) == len(cases),
                  "scientifically_accepted": failure is None and independent is not None,
                  "attempted": len(rows), "prescribed": len(cases), "failure": failure,
                  "summary": summarize(rows),
                  "per_fit": {str(fit): summarize([r for r in rows if r["fit"] == fit])
                              for fit in (161001, 161002, 161003)},
                  "verification": independent, "elapsed_seconds": time.monotonic() - started,
                  "elapsed_seconds_from_original_launch": time.time() - lock["original_launch_epoch"],
                  "artifacts_bytes_before_result": c.directory_bytes(OUTPUT),
                  "external_status": external["screen_status"] if external else "qualification_failed",
                  "external_games": 0, "larger_data_deferred": True,
                  "parent_failure_preserved": c.artifact(PARENT / "result.json"),
                  "limits": "Finite saved-position timing/correctness only; no new strength evidence"}
        c.save_new(OUTPUT / "result.json", result)
        c.save_new(OUTPUT / "completion.json", {"time": c.now(), "result": c.artifact(OUTPUT / "result.json"),
                   "status": "completed" if failure is None else "preserved_failure", "worker_processes_closed": True})
        print(c.canonical({"finished": str(OUTPUT), "attempted": len(rows), "failure": failure,
                           "external_status": result["external_status"]}), flush=True)
    return 0 if failure is None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        try:
            with c.execution_lease():
                result_code = _run()
        except BaseException:
            c.save_new(OUTPUT / "startup_failure.json", {"time": c.now(), "failure": traceback.format_exc()})
            raise
        raise SystemExit(result_code)
