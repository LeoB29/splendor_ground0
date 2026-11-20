"""Reproduce the complete saved160 label-count review; no inference or rollouts.

Run from the repository root. This only reads existing evidence and prints JSON.
The saved2026-09-16 report records the already completed98.025879-second in-memory
execution of this counting loop. It was written afterward without rerunning the
scan. Reproduction should match counts and rates, not runtime or recording time.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import time


BASE = Path("outputs/world_budget_160")
LOCK_SHA256 = "ac89ccc151af8149e2791b7d6fcd989c7ca9416100ca5cf14e270e72cf0872b9"
RESULT_SHA256 = "2e3efa2c524f8d8c68284fad0f3400365bcb08f4b598aecf121039a4c45b19e5"


def analyze():
    lock_bytes = (BASE / "main_collect_lock.json").read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == LOCK_SHA256
    assert hashlib.sha256((BASE / "result.json").read_bytes()).hexdigest() == RESULT_SHA256
    lock = json.loads(lock_bytes)
    count = changed = raw32 = raw128 = 0
    phases = collections.defaultdict(lambda: [0, 0])
    start = time.perf_counter()
    for index, sha in lock["files"].items():
        game_path = BASE / "main" / "games" / f"game_{int(index):04d}.json"
        game_bytes = game_path.read_bytes()
        assert hashlib.sha256(game_bytes).hexdigest() == sha
        game = json.loads(game_bytes)
        for move in game["trace"]:
            if move["actor"] != game["spec"]["candidate_seat"]:
                continue
            root_path = (BASE / "main" / "roots" / f"game_{int(index):04d}"
                         / f"turn_{move['turn']:03d}.json")
            root_bytes = root_path.read_bytes()
            assert hashlib.sha256(root_bytes).hexdigest() == move["root_sha256"]
            root = json.loads(root_bytes)
            assert len(root["worlds"]) == 128
            support = len(root["candidates"]["actions"])
            sums = [sum(world["outcomes"][j]["wdl"] for world in root["worlds"][:32])
                    for j in range(support)]
            choice32 = max(range(support), key=lambda j: sums[j])
            choice128 = root["choice"]
            count += 1
            changed += choice32 != choice128
            raw32 += choice32 == 0
            raw128 += choice128 == 0
            phase = str(min(move["turn"] // 20, 2))
            phases[phase][0] += 1
            phases[phase][1] += choice32 != choice128
    return {
        "roots": count,
        "labels_changed": changed,
        "change_rate": changed / count,
        "raw32": raw32 / count,
        "raw128": raw128 / count,
        "phase_counts_changes": dict(phases),
        "elapsed": time.perf_counter() - start,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, sort_keys=True))
