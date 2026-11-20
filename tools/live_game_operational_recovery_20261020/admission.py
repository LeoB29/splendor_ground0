"""Bounded, recorded RAM admission before any new recovery player request."""
from __future__ import annotations

from pathlib import Path
import time

from tools.local_player_qualification_128.common import ResourceHold, memory, now, save_new

PHYSICAL_BYTES = 2 * 1024**3
COMMIT_BYTES = 4 * 1024**3
INTERVAL_SECONDS = 10
REQUIRED_SAMPLES = 4
MAXIMUM_WAIT_SECONDS = 600


def guard(base, deadline):
    from tools.live_game_operational_recovery_20261020 import common as c
    # There is no player during admission; low RAM causes bounded waiting here.
    # Disk, evidence and both global clocks remain enforced on every sample.
    receipt = c.boundary(base, deadline, memory_headroom=False)
    if receipt["remaining_seconds"] < c.POST_GAME_RESERVE + c.INIT_SECONDS + c.MOVE_SECONDS:
        raise ResourceHold("Insufficient recovery work reserve during RAM admission")


def emit(value):
    import json
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def wait_for_admission(base, deadline):
    base = Path(base)
    path = base / "admission.json"
    if path.exists():
        raise ResourceHold("Recovery RAM admission already attempted")
    started = time.monotonic()
    end = min(started + MAXIMUM_WAIT_SECONDS, deadline)
    readings, streak, first_good = [], 0, None
    status, failure = "preserved_hold", None
    try:
        while True:
            if time.monotonic() >= end:
                raise ResourceHold("Recovery RAM admission wait exhausted")
            guard(base, deadline)
            value = memory()
            stamp = time.monotonic()
            physical = value["available_physical"]
            commit = value["available_page"]
            if type(physical) is not int or type(commit) is not int or min(physical, commit) < 0:
                raise ResourceHold("Invalid recovery RAM reading")
            good = physical >= PHYSICAL_BYTES and commit >= COMMIT_BYTES
            streak = streak + 1 if good else 0
            first_good = (stamp if streak == 1 else first_good) if good else None
            row = {
                "elapsed_seconds": stamp - started,
                "available_physical_bytes": physical,
                "available_commit_bytes": commit,
                "admissible": good,
                "consecutive_admissible_samples": streak,
            }
            readings.append(row)
            emit({"stage": "recovery_memory_admission", **row, "outcomes_closed": True})
            if (
                streak >= REQUIRED_SAMPLES
                and stamp - first_good >= INTERVAL_SECONDS * (REQUIRED_SAMPLES - 1)
                and stamp < end
            ):
                status = "passed"
                break
            left = end - time.monotonic()
            if left <= 0:
                raise ResourceHold("Recovery RAM admission wait exhausted")
            time.sleep(min(INTERVAL_SECONDS, left))
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        receipt = {
            "schema": "operational_recovery_memory_admission_v1",
            "time": now(),
            "status": status,
            "failure_category": failure,
            "minimum_available_physical_bytes": PHYSICAL_BYTES,
            "minimum_available_commit_bytes": COMMIT_BYTES,
            "required_consecutive_samples": REQUIRED_SAMPLES,
            "sample_interval_seconds": INTERVAL_SECONDS,
            "maximum_wait_seconds": MAXIMUM_WAIT_SECONDS,
            "elapsed_seconds": time.monotonic() - started,
            "readings": readings,
            "outcomes_closed": True,
            "no_player_requests_during_admission": True,
        }
        save_new(path, receipt)
    return receipt
