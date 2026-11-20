"""Bounded RAM waiting before workers, with immutable per-admission receipts."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from tools.local_player_qualification_128.common import ResourceHold, memory, now, read, save_new

PHYSICAL_BYTES = 2 * 1024**3
COMMIT_BYTES = 4 * 1024**3
INTERVAL_SECONDS = 10
STABLE_SECONDS = 120
REQUIRED_SAMPLES = 13
MAXIMUM_WAIT_SECONDS = 7200
INITIALIZATION_SECONDS = 60
NAME = re.compile(r"(?:initial|analysis|(?:audits/)?(?:games/g[0-9]{3}/t[0-9]{4}|validation/v[0-9]{3}|g[0-9]{3}-t[0-9]{4}))\Z")


def guard(base, deadline, reserve, request_seconds):
    from tools.matched_strength_recovery_20261022 import common as c
    result = c.boundary(base, deadline, memory_headroom=False)
    if c.work_budget(result, reserve) < INITIALIZATION_SECONDS + request_seconds:
        raise ResourceHold("Insufficient global work reserve during memory waiting")
    return result


def emit(row):
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)


def previous_charge(base):
    seed = read(base/'prior_memory_state.json')
    if (seed['schema'] != 'matched_strength_prior_memory_v1' or type(seed['admission_count']) is not int or
            seed['admission_count'] < 1 or type(seed['charged_seconds']) not in (int, float) or
            not math.isfinite(seed['charged_seconds']) or seed['charged_seconds'] < 0):
        raise ResourceHold('Invalid frozen prior memory accounting')
    total = seed['charged_seconds']
    paths = sorted((base / "memory_admissions").rglob("*.json"))
    for path in paths:
        receipt = read(path)
        charge = receipt["charged_seconds"]
        if receipt["status"] != "passed" or type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0:
            raise ResourceHold("Prior memory admission is incomplete or failed")
        total += charge
    return seed['admission_count']+len(paths), total


def wait_for_memory(base, deadline, *, name, reserve, request_seconds, initial=False):
    """No worker exists here; active request clocks are never suspended or reset."""
    base = Path(base)
    from tools.matched_strength_recovery_20261022 import common as c
    cumulative_limit = c.config()['cumulative_memory_wait_seconds']
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        raise ResourceHold("Invalid memory admission name")
    if type(initial) is not bool or initial != (name == "initial"):
        raise ResourceHold("Initial memory admission identity mismatch")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (reserve, request_seconds)):
        raise ResourceHold("Invalid memory admission work reserve")
    path = base / "memory_admissions" / (name + ".json")
    log = path.with_suffix(".jsonl")
    if path.exists() or log.exists():
        raise ResourceHold("Memory admission already attempted; no retry")
    count, before = previous_charge(base)
    if (initial and count != read(base/'prior_memory_state.json')['admission_count']) or (not initial and not (base / "memory_admissions/initial.json").exists()):
        raise ResourceHold("Initial memory admission must precede all workers")
    started_mono, started_epoch = time.monotonic(), time.time()
    rows, streak, first_good = [], 0, None
    status, category, detail = "preserved_hold", None, None
    needs_stability = initial
    completed_mono = completed_epoch = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("x", encoding="utf-8", newline="\n") as stream:
            while True:
                charged = max(time.monotonic()-started_mono, time.time()-started_epoch, 0.)
                if charged >= MAXIMUM_WAIT_SECONDS or before+charged >= cumulative_limit:
                    raise ResourceHold("Bounded memory waiting allowance exhausted")
                guard_started = time.monotonic()
                resources = guard(base, deadline, reserve, request_seconds)
                reading = memory()
                stamp, epoch = time.monotonic(), time.time()
                charged = max(stamp-started_mono, epoch-started_epoch, 0.)
                if charged >= MAXIMUM_WAIT_SECONDS or before+charged >= cumulative_limit:
                    raise ResourceHold("Bounded memory waiting allowance exhausted")
                physical, commit = reading["available_physical"], reading["available_page"]
                if type(physical) is not int or type(commit) is not int or min(physical, commit) < 0:
                    raise ResourceHold("Invalid memory reading")
                good = physical >= PHYSICAL_BYTES and commit >= COMMIT_BYTES
                if not good:
                    needs_stability = True
                streak = streak+1 if good else 0
                first_good = (stamp if streak == 1 else first_good) if good else None
                row = {"epoch_seconds": epoch, "monotonic_seconds": stamp,
                    "elapsed_seconds": stamp-started_mono, "available_physical_bytes": physical,
                    "available_commit_bytes": commit, "admissible": good,
                    "consecutive_admissible_samples": streak,
                    "remaining_seconds": resources["remaining_seconds"]}
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                stream.flush()
                if needs_stability:
                    emit({"stage": "memory_wait", "name": name, **row, "outcomes_closed": True})
                if good and (not needs_stability or
                             streak >= REQUIRED_SAMPLES and stamp-first_good >= STABLE_SECONDS):
                    break
                # Every wait remains inside both global clocks and the protected work reserve.
                left = min(MAXIMUM_WAIT_SECONDS-charged, cumulative_limit-before-charged,
                           deadline-time.monotonic()-reserve-INITIALIZATION_SECONDS-request_seconds,
                           c.work_budget(resources, reserve)-(time.monotonic()-guard_started)-
                           INITIALIZATION_SECONDS-request_seconds)
                if left <= 0:
                    raise ResourceHold("Insufficient global work reserve during memory waiting")
                time.sleep(min(INTERVAL_SECONDS, left))
            os.fsync(stream.fileno())
        guard(base, deadline, reserve, request_seconds)
        completed_mono, completed_epoch = time.monotonic(), time.time()
        charged = max(completed_mono-started_mono, completed_epoch-started_epoch, 0.)
        if charged >= MAXIMUM_WAIT_SECONDS or before+charged >= cumulative_limit:
            raise ResourceHold("Bounded memory waiting allowance exhausted during final admission check")
        status = "passed"
    except BaseException as error:
        status = "preserved_hold"
        category, detail = type(error).__name__, str(error)
        raise
    finally:
        finished_mono, finished_epoch = ((completed_mono, completed_epoch) if status == "passed"
                                         else (time.monotonic(), time.time()))
        charged = max(finished_mono-started_mono, finished_epoch-started_epoch, 0.)
        log_bytes = log.read_bytes() if log.exists() else b""
        receipt = {
            "schema": "operational_memory_admission_v2", "time": now(), "name": name,
            "status": status, "failure_category": category, "failure_detail": detail,
            "initial": initial, "stability_required": needs_stability,
            "started_epoch_seconds": started_epoch, "started_monotonic": started_mono,
            "finished_epoch_seconds": finished_epoch, "finished_monotonic": finished_mono,
            "elapsed_seconds": finished_mono-started_mono, "charged_seconds": charged,
            "prior_admission_count": count, "cumulative_before_seconds": before,
            "cumulative_after_seconds": before+charged,
            "maximum_wait_seconds": MAXIMUM_WAIT_SECONDS,
            "maximum_cumulative_wait_seconds": cumulative_limit,
            "minimum_available_physical_bytes": PHYSICAL_BYTES,
            "minimum_available_commit_bytes": COMMIT_BYTES,
            "sample_interval_seconds": INTERVAL_SECONDS, "stable_seconds": STABLE_SECONDS,
            "required_consecutive_samples": REQUIRED_SAMPLES,
            "reserve_seconds": reserve, "request_seconds": request_seconds,
            "initialization_seconds": INITIALIZATION_SECONDS,
            "readings": rows,
            "readings_log": {"path": log.relative_to(base).as_posix(),
                             "sha256": hashlib.sha256(log_bytes).hexdigest(), "bytes": len(log_bytes)},
            "no_worker_during_admission": True, "outcomes_closed": True}
        save_new(path, receipt)
    return receipt
