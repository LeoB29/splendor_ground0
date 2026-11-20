"""Budget cases are fixed before any attempt, including independent repetitions."""
from __future__ import annotations


def build_cases(entries):
    if len(entries) < 18 or len(entries) > 23:
        raise ValueError("Expected 18 core and at most five distinct edge roots")
    cases = []
    for allowance in (1, 120):
        for index, _ in enumerate(entries):
            for breadth in ((False, True) if index % 2 == 0 else (True, False)):
                cases.append({"index": len(cases), "root_index": index,
                              "breadth": breadth, "allowance_seconds": allowance,
                              "kind": "core" if index < 18 else "edge",
                              "hidden_variant": False})
    seen = set()
    for index, entry in enumerate(entries[:18]):
        fit = entry["fit"]
        if fit in seen:
            continue
        seen.add(fit)
        for breadth in (False, True):
            cases.append({"index": len(cases), "root_index": index,
                          "breadth": breadth, "allowance_seconds": 120,
                          "kind": "repeat", "hidden_variant": True})
    if seen != {161001, 161002, 161003}:
        raise ValueError("All three frozen fits must be present")
    return cases


def summarize(rows):
    """Completion timings are never imputed for timed-out attempts."""
    import statistics
    result = []
    for kind in ("core", "edge", "repeat"):
        for breadth in (False, True):
            for allowance in (1, 120):
                group = [r for r in rows if r["case"]["kind"] == kind
                         and r["case"]["breadth"] == breadth
                         and r["case"]["allowance_seconds"] == allowance]
                completed = [r for r in group if r.get("correct") is True]
                times = sorted(r["clock"]["receipt_elapsed_seconds"] for r in completed)
                statuses = {}
                for row in group:
                    status = row["clock"]["status"]
                    statuses[status] = statuses.get(status, 0) + 1
                result.append({"kind": kind, "candidates": 8 if breadth else 4,
                               "allowance_seconds": allowance, "attempted": len(group),
                               "verified_completions": len(completed), "statuses": statuses,
                               "completed_within_one_second": sum(t <= 1 for t in times),
                               "completion_mean_seconds": statistics.mean(times) if times else None,
                               "completion_median_seconds": statistics.median(times) if times else None,
                               "completion_max_seconds": max(times) if times else None})
    return result
