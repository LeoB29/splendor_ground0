"""Independent, read-only final audit of the local-player deadline qualification.

Standard library only: this never imports the player, clock, frozen reporting
code or numerical libraries. Run only after completion.json exists. An optional
report destination is created exclusively, never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/local_player_qualification_20261018_recovery_01"
DEFAULT_ORIGINAL = ROOT / "outputs/local_player_qualification_20261018_01"
FITS = (161001, 161002, 161003)
NATURAL = {"completed", "forced_pass_deadlock"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_constant)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_path(item, repo_root):
    path = (repo_root / item["path"]).resolve()
    require(path.is_relative_to(repo_root.resolve()), "Artifact outside repository")
    if "bytes" in item:
        require(type(item["bytes"]) is int and item["bytes"] >= 0
                and path.stat().st_size == item["bytes"], f"Artifact size changed: {item['path']}")
    require(file_hash(path) == item["sha256"], f"Artifact changed: {item['path']}")
    return path


def verify_frozen_inputs(freeze, repo_root):
    """Stream every bound input independently, including original parent inputs."""
    items = freeze["inputs"]
    require(isinstance(items, list) and items, "Empty frozen input registry")
    seen, byte_count = {}, 0
    for item in items:
        require(isinstance(item, dict) and {"path", "sha256", "bytes"}.issubset(item),
                "Frozen input has no complete file identity")
        identity = (item["sha256"], item["bytes"])
        require(item["path"] not in seen or seen[item["path"]] == identity,
                "Conflicting frozen identities for one path")
        source_path(item, repo_root)
        seen[item["path"]] = identity
        byte_count += item["bytes"]
    return {"all_hashes_and_sizes_match": True, "input_entries_checked": len(items),
            "unique_paths_checked": len(seen), "bytes_hashed_including_repeated_entries": byte_count,
            "verification": "independent streaming SHA256 and byte-size check after completion"}


def verify_observer_digest(row, payload):
    require(row["observer_sha256"] == digest(payload), "Attempt observer hash differs from sealed input")


def number(value, nonnegative=True):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0 or not nonnegative)


def quantile(values, probability):
    """Linear interpolation at (n-1)*p, explicitly including singleton panels."""
    require(0 <= probability <= 1, "Invalid quantile")
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    values = list(values)
    require(all(number(v) for v in values), "Invalid nonnegative measurement")
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95_linear": quantile(values, .95), "max": max(values) if values else None}


def independent_cases(selection):
    core, edges = selection["core"], selection["edges"]
    expected_cells = [(fit, seat, phase) for fit in FITS for seat in (0, 1) for phase in (0, 1, 2)]
    require([(r["fit"], r["actor"], r["phase"]) for r in core] == expected_cells,
            "Core panel is not the prescribed 18 cells")
    require(len(edges) <= 5, "Too many metadata edge roots")
    entries = core + edges
    require(len({r["root_id"] for r in entries}) == len(entries), "Duplicate selected root")
    require(all(r["arm"] == "candidate" for r in entries), "Non-candidate saved root")
    result = []
    for allowance in (1, 120):
        for index in range(len(entries)):
            for breadth in ([False, True] if index % 2 == 0 else [True, False]):
                result.append({"index": len(result), "root_index": index, "breadth": breadth,
                               "allowance_seconds": allowance, "kind": "core" if index < 18 else "edge",
                               "hidden_variant": False})
    for fit in FITS:
        index = next(i for i, item in enumerate(core) if item["fit"] == fit)
        for breadth in (False, True):
            result.append({"index": len(result), "root_index": index, "breadth": breadth,
                           "allowance_seconds": 120, "kind": "repeat", "hidden_variant": True})
    return result


def expected_result(root, breadth):
    """Reconstruct exact four-prefix/eight-way evidence with independent arithmetic."""
    require(root["planner"] == "candidate" and "failure" not in root, "Incomplete saved candidate root")
    record = {key: deepcopy(root[key]) for key in
              ("root_identity", "planner", "privacy", "schedule", "candidates", "proposal", "scores", "worlds")}
    record["proposal"].pop("timing", None)
    proposal, candidates = record["proposal"], record["candidates"]
    prefix = proposal["anchor_candidates"]
    n = len(prefix["keys"])
    require(1 <= n <= 4 and len(prefix["actions"]) == n, "Invalid saved four-prefix")
    require(candidates == proposal["candidates"] and proposal["breadth"] is True, "Saved proposal mismatch")
    require(prefix["keys"] == candidates["keys"][:n] and prefix["actions"] == candidates["actions"][:n],
            "Saved four-prefix was not preserved")
    require(len(set(candidates["keys"])) == len(candidates["keys"]) == len(candidates["actions"]),
            "Duplicate/incomplete candidate support")
    count = len(candidates["keys"]) if breadth else n
    require(1 <= count <= (8 if breadth else 4), "Invalid candidate count")
    if not breadth:
        record["planner"] = "reference"
        record["candidates"] = deepcopy(prefix)
        proposal.update(candidates=deepcopy(prefix), breadth=False, added_indices={"ranked": [], "random": []})
    seeds = record["schedule"]["world_seeds"]
    require(len(seeds) == len(set(seeds)) == len(record["worlds"]) == 128, "Incomplete world schedule")
    totals = [0] * count
    for seed, world in zip(seeds, record["worlds"]):
        require(world["seed"] == seed, "Wrong world order")
        require(len(world["outcomes"]) == len(candidates["keys"]), "Incomplete eight-way outcomes")
        world["outcomes"] = world["outcomes"][:count]
        for index, outcome in enumerate(world["outcomes"]):
            require(outcome["reason"] in NATURAL and type(outcome["wdl"]) in (int, float)
                    and outcome["wdl"] in (-1, 0, 1), "Invalid natural WDL outcome")
            totals[index] += int(outcome["wdl"])
    # List.index independently enforces the first maximum in offered order.
    choice = totals.index(max(totals))
    means = [value / 128 for value in totals]
    require(not breadth or (root["choice"] == choice and root["means"] == means), "Saved eight-way arithmetic mismatch")
    record.update(choice=choice, means=means)
    return {"action": deepcopy(record["candidates"]["actions"][choice]), "record": record}


def compare_scientific(actual, wanted, path=(), tolerant_count=None):
    """Discrete scientific evidence is exact; continuous proposal values use frozen tolerance."""
    if tolerant_count is None:
        tolerant_count = [0]
    if isinstance(wanted, dict):
        require(isinstance(actual, dict) and set(actual) == set(wanted), f"Different object fields at {path}")
        for key in wanted:
            compare_scientific(actual[key], wanted[key], path + (key,), tolerant_count)
    elif isinstance(wanted, list):
        require(isinstance(actual, list) and len(actual) == len(wanted), f"Different list at {path}")
        for index, (left, right) in enumerate(zip(actual, wanted)):
            compare_scientific(left, right, path + (index,), tolerant_count)
    elif type(wanted) in (int, float) and type(wanted) is not bool:
        require(number(actual, nonnegative=False), f"Invalid number at {path}")
        # Only pre-search model-derived quantities may use 1e-6. World evidence,
        # means, ordered choices, card/action identities and schedules stay exact.
        model_value = type(wanted) is float and (path[:2] == ("record", "scores") or (
            path[:2] == ("record", "proposal") and any(
                key in path for key in ("scores", "learned_logits", "learned_group_scores"))))
        if model_value:
            require(math.isclose(actual, wanted, rel_tol=1e-6, abs_tol=1e-6), f"Model value differs at {path}")
            tolerant_count[0] += int(actual != wanted)
        else:
            require(actual == wanted, f"Discrete/scientific number differs at {path}")
    else:
        require(type(actual) is type(wanted) and actual == wanted, f"Value differs at {path}")
    return tolerant_count[0]


def progress_lower_bound(path):
    if not path.exists():
        return {"reported_completed_worlds_lower_bound": 0, "progress_messages": 0}
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # Unfinished stderr lines are not completed progress receipts.
        if isinstance(row, dict) and row.get("event") == "progress":
            value = row.get("completed_worlds")
            require(type(value) is int and 0 <= value <= 128 and row.get("worlds") == 128, "Invalid progress counter")
            values.append(value)
    require(values == sorted(values), "Decreasing worker progress")
    return {"reported_completed_worlds_lower_bound": max(values, default=0), "progress_messages": len(values)}


def verify_rows(base, cases, entries, repo_root):
    rows, notes, tolerance_differences = [], [], 0
    paths = sorted((base / "rows").glob("*.json"))
    require(len(paths) <= len(cases), "Too many attempt rows")
    for index, path in enumerate(paths):
        require(path.name == f"{index:03d}.json", "Attempt rows are not a contiguous schedule prefix")
        row = read(path)
        require(row["case"] == cases[index], "Attempt differs from frozen schedule")
        case = cases[index]
        entry = entries[case["root_index"]]
        require(row["root_id"] == entry["root_id"] and row["fit"] == entry["fit"]
                and row["original_sha256"] == entry["sha256"], "Attempt/source/fit linkage mismatch")
        require(type(row["correct"]) is bool, "Invalid correctness field")
        input_folder = "variants" if case["hidden_variant"] else "observations"
        sealed_input = read(base / input_folder / f'{case["root_index"]:02d}.json')
        verify_observer_digest(row, sealed_input)
        attempt = base / "attempts" / f"{index:03d}"
        receipt_path = attempt / "request-00000001.json"
        receipt = read(receipt_path) if receipt_path.exists() else None
        if receipt is not None:
            require(row["clock"] == {k: v for k, v in receipt.items() if k != "result"}, "Row and trusted receipt differ")
        else:
            require(not row["correct"], "Completed decision has no trusted receipt")
        if row.get("initialization") is not None:
            initial = read(attempt / "initialization.json")
            require(all(initial.get(k) == v for k, v in row["initialization"].items()), "Initialization receipt mismatch")
            require(initial["status"] == "ok" and initial["accepted"] is True, "Move began before initialization succeeded")
            require(len(initial["affinity"]) == 1 and all(v == "1" for v in initial["thread_environment"].values()),
                    "Unexpected numerical thread/affinity configuration")
            telemetry = row["clock"].get("telemetry", {})
            require(telemetry.get("affinity", initial["affinity"]) == initial["affinity"], "Changed request affinity")
        if row["correct"]:
            clock = row["clock"]
            require(clock["status"] == "ok" and clock["accepted"] is True and clock["received_before_deadline"] is True,
                    "Correct decision lacks timely accepted receipt")
            elapsed = clock["receipt_elapsed_seconds"]
            require(number(elapsed) and elapsed <= clock["deadline_seconds"] <= case["allowance_seconds"],
                    "Completed receipt exceeds deadline")
            root = read(source_path(entry, repo_root))
            target = expected_result(root, case["breadth"])
            require(digest(target) == row["expected_sha256"], "Independently reconstructed expectation hash differs")
            result = receipt["result"]
            tolerance_differences += compare_scientific(result, target)
            require(digest(result) == row["result_sha256"], "Row/result hash mismatch")
            work = {"worlds": len(result["record"]["worlds"]),
                    "continuations": sum(len(w["outcomes"]) for w in result["record"]["worlds"])}
            require(work == row["work"] and work["worlds"] == 128
                    and work["continuations"] == 128 * min(entry["canonical_support"], 8 if case["breadth"] else 4),
                    "Completed work is not exact fixed-budget work")
        elif row["clock"]["status"] == "ok":
            notes.append({"case": index, "issue": "Timely response failed scientific comparison", "failure": row.get("failure")})
        progress = progress_lower_bound(attempt / "stderr.log")
        row = deepcopy(row)
        row["analysis_progress"] = progress
        rows.append(row)
    return rows, {"completed_scientific_records_checked": sum(r["correct"] for r in rows),
                  "continuous_values_differing_only_within_frozen_tolerance": tolerance_differences,
                  "scientific_comparison_failures": notes}


def group_summary(cases, rows):
    ids = {case["index"] for case in cases}
    observed = [row for row in rows if row["case"]["index"] in ids]
    completed = [row for row in observed if row["correct"]]
    statuses = dict(sorted(Counter(row["clock"]["status"] for row in observed).items()))
    cpu = [row["clock"].get("telemetry", {}).get("request_cpu_seconds") for row in observed]
    rss = [row["clock"].get("telemetry", {}).get("peak_sampled_rss_bytes") for row in observed]
    initial = [row.get("initialization", {}).get("receipt_elapsed_seconds") for row in observed]
    total_work = {key: sum(row.get("work", {}).get(key, 0) for row in completed) for key in ("worlds", "continuations")}
    unfinished_progress = sum(row.get("analysis_progress", {}).get("reported_completed_worlds_lower_bound", 0)
                              for row in observed if not row["correct"])
    return {"prescribed": len(cases), "attempted": len(observed), "not_attempted": len(cases) - len(observed),
            "verified_completions": len(completed), "operational_or_correctness_failures": len(observed) - len(completed),
            "statuses": statuses,
            "completion_seconds": distribution(row["clock"]["receipt_elapsed_seconds"] for row in completed),
            "completed_within_one_second": sum(row["clock"]["receipt_elapsed_seconds"] <= 1 for row in completed),
            "request_cpu_seconds_all_attempts_with_telemetry": distribution(v for v in cpu if v is not None),
            "total_sampled_request_cpu_seconds": sum(v for v in cpu if v is not None),
            "peak_sampled_worker_rss_bytes": distribution(v for v in rss if v is not None),
            "initialization_seconds": distribution(v for v in initial if v is not None),
            "exact_work_in_verified_completions": total_work,
            "unfinished_attempt_progress_worlds_lower_bound": unfinished_progress,
            "timeouts_are_not_completion_latency_samples": True}


def repeats_summary(cases, rows):
    by_index = {row["case"]["index"]: row for row in rows}
    pairs = []
    for case in cases:
        if case["kind"] != "repeat":
            continue
        original_case = next(c for c in cases if c["kind"] == "core" and c["root_index"] == case["root_index"]
                             and c["breadth"] == case["breadth"] and c["allowance_seconds"] == 120)
        a, b = by_index.get(original_case["index"]), by_index.get(case["index"])
        both = a is not None and b is not None and a["correct"] and b["correct"]
        same_input = a is not None and b is not None and a["observer_sha256"] == b["observer_sha256"]
        same_result = both and a["result_sha256"] == b["result_sha256"]
        require(not both or (same_input and same_result), "Completed privacy repetition disagrees with its baseline")
        pairs.append({"root_index": case["root_index"], "candidates": 8 if case["breadth"] else 4,
                      "original_case": original_case["index"], "repeat_case": case["index"],
                      "both_verified_complete": bool(both), "same_observer_payload": bool(same_input),
                      "same_scientific_result": bool(same_result)})
    return {"prescribed_pairs": len(pairs), "verified_identical_pairs": sum(p["same_scientific_result"] for p in pairs),
            "pairs": pairs}


def qualifications(cases, rows, final_integrity_accepted):
    result = []
    by_index = {row["case"]["index"]: row for row in rows}
    for breadth in (False, True):
        for allowance in (1, 120):
            required = [c for c in cases if c["breadth"] == breadth and c["allowance_seconds"] == allowance]
            missing = [c["index"] for c in required if c["index"] not in by_index]
            failed = [c["index"] for c in required if c["index"] in by_index and not by_index[c["index"]]["correct"]]
            all_pass = bool(required) and not missing and not failed and final_integrity_accepted
            result.append({"candidates": 8 if breadth else 4, "allowance_seconds": allowance,
                           "required_cases": len(required), "missing_cases": missing, "failed_cases": failed,
                           "meets_entire_prescribed_sampled_panel": all_pass,
                           "status": "met_sampled_panel" if all_pass else "missed_sampled_panel" if failed else "not_established",
                           "scope": "All prescribed core/edge cases; 120s also requires all prescribed repeats. No universal timing guarantee."})
    return result


def inspect_original(base, repo_root):
    result, completion = read(base / "result.json"), read(base / "completion.json")
    require(source_path(completion["result"], repo_root) == (base / "result.json").resolve(), "Original completion result mismatch")
    rows = [read(path) for path in sorted((base / "rows").glob("*.json"))]
    worker_initializations = len(list((base / "attempts").glob("*/initialization.json")))
    return {"path": str(base), "completion_status": completion["status"],
            "attempt_rows": len(rows), "worker_initializations": worker_initializations,
            "verified_completions": sum(r.get("correct") is True for r in rows),
            "statuses": dict(Counter(r["clock"]["status"] for r in rows)), "failure": result["failure"],
            "scientifically_accepted": result["scientifically_accepted"],
            "result_sha256": file_hash(base / "result.json"),
            "separate_from_recovery_timing_samples": True}


def analyze(run=DEFAULT_RUN, original=DEFAULT_ORIGINAL, repo_root=ROOT):
    run, original, repo_root = Path(run).resolve(), Path(original).resolve(), Path(repo_root).resolve()
    require((run / "completion.json").is_file(), "Run is not complete; do not audit a moving output directory")
    completion, saved = read(run / "completion.json"), read(run / "result.json")
    require(source_path(completion["result"], repo_root) == run / "result.json", "Completion result reference mismatch")
    selection, schedule = read(run / "selection.json"), read(run / "schedule.json")
    entries = selection["core"] + selection["edges"]
    cases = independent_cases(selection)
    require(schedule == cases, "Frozen schedule differs from independent prescribed schedule")
    require(selection == read(original / "selection.json") and schedule == read(original / "schedule.json"),
            "Recovery changed the original frozen panel or schedule")
    launch, freeze = read(run / "launch.json"), read(run / "freeze.json")
    require(source_path(launch["freeze"], repo_root) == run / "freeze.json", "Launch freeze reference mismatch")
    require(freeze["changed_player_or_clock"] is False, "Recovery changed the numerical player or clock")
    all_inputs = verify_frozen_inputs(freeze, repo_root)
    parent = inspect_original(original, repo_root)
    require(parent["worker_initializations"] == parent["verified_completions"] == 0
            and "Less than one GiB physical memory" in (parent["failure"] or ""), "Recovery parent differs from declared pre-worker hold")
    require(saved["parent_failure_preserved"]["sha256"] == parent["result_sha256"], "Original failure reference mismatch")
    frozen = {item["path"]: item["sha256"] for item in freeze["inputs"]}
    for filename in ("selection.json", "schedule.json"):
        relative = (run / filename).relative_to(repo_root).as_posix()
        require(frozen.get(relative) == file_hash(run / filename), "Selection/schedule not bound to freeze")
    for item in entries:
        require(frozen.get(item["path"]) == item["sha256"], "Selected root not bound to freeze")
        source_path(item, repo_root)
    rows, independent = verify_rows(run, cases, entries, repo_root)
    require(saved["attempted"] == len(rows) and saved["prescribed"] == len(cases)
            and saved["complete_schedule"] == (len(rows) == len(cases)), "Saved aggregate case count mismatch")
    privacy = read(run / "privacy_inputs.json")
    require(len(privacy) == len(entries), "Incomplete privacy input checks")
    privacy_relative = (run / "privacy_inputs.json").relative_to(repo_root).as_posix()
    require(frozen.get(privacy_relative) == file_hash(run / "privacy_inputs.json"), "Privacy receipt not bound to freeze")
    for index, (entry, receipt) in enumerate(zip(entries, privacy)):
        for folder in ("observations", "variants"):
            path = run / folder / f"{index:02d}.json"
            require(frozen.get(path.relative_to(repo_root).as_posix()) == file_hash(path), "Observer input changed after freeze")
        a, b = read(run / "observations" / f"{index:02d}.json"), read(run / "variants" / f"{index:02d}.json")
        require(receipt["root_id"] == entry["root_id"] and a == b and receipt["identical"] is True
                and digest(a) == receipt["payload_sha256"] == receipt["hidden_variant_payload_sha256"], "Privacy input equality receipt differs")
    repeat = repeats_summary(cases, rows)
    final = read(run / "final_input_check.json") if (run / "final_input_check.json").exists() else None
    if final is not None:
        require(final["inputs_checked"] == all_inputs["input_entries_checked"], "Final frozen-input count differs")
    final_ok = final is not None and final.get("unchanged") is True and not final["original"]["failures"]
    in_window = number(saved.get("elapsed_seconds_from_original_launch")) and saved["elapsed_seconds_from_original_launch"] <= 14400
    accepted = saved["scientifically_accepted"] is True and not saved["failure"] and final_ok and in_window
    summary = group_summary(cases, rows)
    if saved["verification"] is not None:
        check = saved["verification"]
        require(check["rows_checked"] == len(rows) and check["completed_decisions_checked"] == summary["verified_completions"]
                and check["replayed_worlds"] == summary["exact_work_in_verified_completions"]["worlds"]
                and check["replayed_continuations"] == summary["exact_work_in_verified_completions"]["continuations"],
                "Saved independent work/case counts differ")
    dimensions = ("fit", "candidates", "allowance_seconds", "phase", "seat", "kind")
    groups = defaultdict(list)
    for case in cases:
        entry = entries[case["root_index"]]
        key = (entry["fit"], 8 if case["breadth"] else 4, case["allowance_seconds"], entry["phase"], entry["actor"], case["kind"])
        groups[key].append(case)
    detailed = [{**dict(zip(dimensions, key)), **group_summary(group, rows)} for key, group in sorted(groups.items())]
    role_clock = []
    for breadth in (False, True):
        for allowance in (1, 120):
            chosen = [c for c in cases if c["breadth"] == breadth and c["allowance_seconds"] == allowance]
            role_clock.append({"candidates": 8 if breadth else 4, "allowance_seconds": allowance,
                               **group_summary(chosen, rows)})
    external = read(run / "external_qualification.json") if (run / "external_qualification.json").exists() else None
    return {"schema": "independent_local_qualification_analysis_v1", "time": datetime.now(timezone.utc).isoformat(),
            "analyzer": {"path": str(Path(__file__).resolve()), "sha256": file_hash(Path(__file__))},
            "run": str(run), "audit_pass": True, "source_result_sha256": file_hash(run / "result.json"),
            "independent_verification": independent, "independent_frozen_input_check": all_inputs,
            "original_preserved_failure": parent,
            "recovery": {"completion_status": completion["status"], "failure": saved["failure"],
                         "complete_schedule": saved["complete_schedule"], "scientifically_accepted": saved["scientifically_accepted"],
                         "final_input_check_accepted": final_ok,
                         "elapsed_seconds_from_original_launch": saved.get("elapsed_seconds_from_original_launch"),
                         "within_original_four_hours": in_window},
            "summary": summary, "by_player_and_clock": role_clock, "by_fit_player_clock_phase_seat_kind": detailed,
            "repeats": repeat, "sampled_clock_qualification": qualifications(cases, rows, accepted),
            "privacy_input_checks": {"roots": len(privacy), "identical_lawful_inputs": len(privacy),
                                     "hidden_states_actually_changed": sum(p.get("hidden_state_actually_changed") is True for p in privacy)},
            "external": {"status": saved["external_status"], "games": saved["external_games"],
                         "independent_catalogue": external.get("independent_live_engine_check") if external else None},
            "automatic_promotion": False, "new_strength_evidence": False,
            "limitations": ["Finite predetermined saved-position panel; no claim of universal deadline compliance.",
                            "Completion quantiles exclude timeouts, initialization failures and incorrect responses; censored deadlines are never imputed.",
                            "Progress counts are lower bounds for unfinished work, not completed decisions.",
                            "Worker CPU and peak RSS are sampled telemetry; observed process thread counts do not imply multiple numerical threads.",
                            "Exact original action/world/outcome evidence checked; model values retain the predeclared 1e-6 tolerance.",
                            "Ordinary desktop load and full evidence serialization are part of measured receipt latency."]}


class ArithmeticTests(unittest.TestCase):
    def test_quantiles(self):
        self.assertEqual(quantile([], .95), None)
        self.assertEqual(quantile([7], .95), 7)
        self.assertAlmostEqual(quantile([0, 10], .95), 9.5)
        self.assertEqual(distribution([1, 3, 5])["median"], 3)

    def test_censored_time_is_not_completion_time(self):
        cases = [{"index": i} for i in range(3)]
        rows = [{"case": cases[0], "correct": True, "clock": {"status": "ok", "receipt_elapsed_seconds": 2},
                 "work": {"worlds": 128, "continuations": 512}},
                {"case": cases[1], "correct": False, "clock": {"status": "timeout", "receipt_elapsed_seconds": 120.1}}]
        result = group_summary(cases, rows)
        self.assertEqual(result["completion_seconds"]["n"], 1)
        self.assertEqual(result["completion_seconds"]["max"], 2)
        self.assertEqual(result["not_attempted"], 1)

    def test_later_completion_does_not_rescue_first_timeout(self):
        cases = [{"index": 0, "breadth": False, "allowance_seconds": 120, "kind": "core"},
                 {"index": 1, "breadth": False, "allowance_seconds": 120, "kind": "repeat"}]
        rows = [{"case": cases[0], "correct": False}, {"case": cases[1], "correct": True}]
        result = qualifications(cases, rows, True)[1]
        self.assertFalse(result["meets_entire_prescribed_sampled_panel"])
        self.assertEqual(result["failed_cases"], [0])

    def test_discrete_evidence_has_no_float_tolerance(self):
        with self.assertRaises(ValueError):
            compare_scientific({"record": {"means": [.50000001]}}, {"record": {"means": [.5]}})
        self.assertEqual(compare_scientific({"record": {"scores": [.50000001]}}, {"record": {"scores": [.5]}}), 1)

    def test_independent_prefix_reselection_includes_losses(self):
        candidates = {"keys": list("abcdefgh"), "actions": [{"id": i} for i in range(8)]}
        prefix = {"keys": candidates["keys"][:4], "actions": candidates["actions"][:4]}
        worlds = [{"seed": str(i), "outcomes": [{"reason": "completed", "wdl": v}
                   for v in (-1, 0, 0, -1, 1, 0, 0, 0)]} for i in range(128)]
        root = {"root_identity": ["main", 0, 0], "planner": "candidate", "privacy": {},
                "schedule": {"world_seeds": [str(i) for i in range(128)]}, "candidates": candidates,
                "proposal": {"candidates": candidates, "anchor_candidates": prefix, "breadth": True},
                "scores": [], "worlds": worlds, "choice": 4, "means": [-1, 0, 0, -1, 1, 0, 0, 0]}
        four, eight = expected_result(root, False), expected_result(root, True)
        self.assertEqual(four["record"]["choice"], 1)
        self.assertEqual(eight["record"]["choice"], 4)
        self.assertEqual(four["record"]["means"], [-1, 0, 0, -1])
        self.assertEqual(len(root["worlds"][0]["outcomes"]), 8)

    def test_attempt_observer_hash_cannot_disagree_with_sealed_payload(self):
        payload = {"schema": "synthetic", "context": {"fit": 161001}, "observation": {"turn": 20}}
        row = {"observer_sha256": digest(payload), "correct": False, "clock": {"status": "timeout"}}
        verify_observer_digest(row, payload)
        changed = deepcopy(payload)
        changed["observation"]["turn"] = 21
        with self.assertRaisesRegex(ValueError, "observer hash"):
            verify_observer_digest(row, changed)

    def test_every_frozen_input_is_checked_and_counted(self):
        from unittest.mock import patch
        inputs = [{"path": f"source/{i}.py", "sha256": str(i) * 64, "bytes": i + 1} for i in range(3)]
        with patch(__name__ + ".source_path") as check:
            receipt = verify_frozen_inputs({"inputs": inputs}, ROOT)
            self.assertEqual(check.call_count, 3)
            self.assertEqual([call.args[0] for call in check.call_args_list], inputs)
        self.assertEqual(receipt["input_entries_checked"], 3)
        self.assertEqual(receipt["bytes_hashed_including_repeated_entries"], 6)
        with patch(__name__ + ".source_path", side_effect=ValueError("Artifact changed")):
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                verify_frozen_inputs({"inputs": inputs}, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        test = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ArithmeticTests))
        return 0 if test.wasSuccessful() else 1
    result = analyze(args.run, args.original)
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"output": str(args.output.resolve()), "audit_pass": result["audit_pass"],
                          "attempted": result["summary"]["attempted"],
                          "verified_completions": result["summary"]["verified_completions"]}, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
