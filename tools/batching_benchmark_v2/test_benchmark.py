"""Preparation and orchestration checks only: stdlib, no neural inference."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.batching_benchmark_v2 import benchmark as b
from tools.batching_benchmark_v2 import runtime as r


def fixtures():
    rows = []
    for index, (phase, seat, planner, privacy, repetition) in enumerate(
            itertools.product(range(3), (0, 1), ("candidate", "reference"), (0, 1), range(3))):
        rows.append({"index": index, "turn": phase*20+seat, "seat": seat, "planner": planner,
                     "support": 4, "privacy": {"own": privacy, "opponent": 0}, "sha256": str(index),
                     "choice": 0, "moves": 9000, "timing": {"wall": 99}, "score": 1})
    return rows


class SelectionTests(unittest.TestCase):
    def test_exact_primary_cells_and_timing_balance(self):
        selected, coverage = b.select_roots(fixtures())
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({b.cell(row) for row in selected}), 24)
        self.assertEqual(len(coverage["timing_roots"]), 12)
        self.assertEqual(coverage["missing_primary_cells"], [])

    def test_outcomes_runtime_and_branch_length_do_not_affect_selection(self):
        rows = fixtures()
        before = b.select_roots(rows)
        for row in rows:
            row.update(choice=99, moves=-100, timing={"wall": 0}, score=-1, winner="changed", reasons={})
        self.assertEqual(before, b.select_roots(list(reversed(rows))))
        for row in before[0]:
            self.assertTrue({"choice", "moves", "timing", "score", "winner"}.isdisjoint(row))

    def test_duplicate_root_rejected(self):
        rows = fixtures()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            b.select_roots(rows + [rows[0]])

    def test_missing_cells_reported_without_replacement(self):
        selected, coverage = b.select_roots([row for row in fixtures() if row["seat"] == 0])
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(coverage["missing_primary_cells"]), 12)

    def test_natural_short_supports_retained_separately(self):
        rows = fixtures()
        for support in (1, 2, 3):
            rows.append({**rows[0], "index": 100+support, "support": support})
        selected, coverage = b.select_roots(rows)
        self.assertTrue({1, 2, 3}.issubset({row["support"] for row in selected}))
        self.assertLessEqual(len(selected), 27)
        self.assertEqual(coverage["absent_short_supports"], [])
        self.assertTrue(all(row["purpose"] == "primary" for row in selected if b.root_key(row) in coverage["timing_roots"]))

    def test_same_number_of_first_and_second_positions(self):
        schedule = b.timing_schedule(["a", "b", "c"])
        for key in ("a", "b", "c"):
            rows = [row for row in schedule if row["root"] == key]
            self.assertEqual([row["repetition"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(sum(row["order"][0] == "serial" for row in rows), 2)
        self.assertNotEqual(schedule[0]["order"], schedule[1]["order"])
        with self.assertRaises(ValueError):
            b.timing_schedule(["a", "a"])


class IntegrityTests(unittest.TestCase):
    def test_recycled_non_python_process_identity_is_safe(self):
        rows = [{"ProcessId": 10, "Name": "firefox.exe", "ExecutablePath": r"C:\Program Files\Mozilla Firefox\firefox.exe"},
                {"ProcessId": 11, "Name": "Discord.exe", "ExecutablePath": r"C:\Apps\Discord.exe"}]
        self.assertEqual(b.blocking_processes({10, 11}, rows, 99), [])

    def test_python_and_unknown_identity_still_block(self):
        rows = [{"ProcessId": 10, "Name": "pythonw.exe", "ExecutablePath": r"C:\Runtime\pythonw.exe"},
                {"ProcessId": 11, "Name": "firefox.exe", "ExecutablePath": None},
                {"ProcessId": 12, "Name": "firefox.exe", "ExecutablePath": r"C:\Apps\python.exe"},
                {"ProcessId": 13, "Name": "firefox.exe", "ExecutablePath": "firefox.exe"}]
        self.assertEqual(b.blocking_processes({10, 11, 12, 13}, rows, 99), [10, 11, 12, 13])

    def test_current_new_process_and_unrelated_ids_not_old_workers(self):
        rows = [{"ProcessId": 99, "Name": "pythonw.exe", "ExecutablePath": r"C:\Runtime\pythonw.exe"},
                {"ProcessId": 98, "Name": "python.exe", "ExecutablePath": None}]
        self.assertEqual(b.blocking_processes({99}, rows, 99), [])

    def test_no_model_or_runtime_dependency_imports(self):
        command = [sys.executable, "-B", "-c", "from tools.batching_benchmark_v2 import benchmark, runtime; import sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert not any(k.startswith('splendor_ai') for k in sys.modules)"]
        subprocess.run(command, cwd=b.ROOT, check=True, capture_output=True)

    def test_write_once_and_directory_boundaries(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(b, "ROOT", Path(directory)):
            target = b.path(b.OUTPUT) / "example.json"
            b.save_new(target, {"preserved": True})
            with self.assertRaises(FileExistsError):
                b.save_new(target, {"preserved": False})
            self.assertEqual(b.read(target), {"preserved": True})
            with self.assertRaisesRegex(ValueError, "outside"):
                b.save_new(Path(directory) / "old.json", {})
            with self.assertRaisesRegex(ValueError, "outside"):
                b.path("../escape.json")

    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(b, "ROOT", Path(directory)):
            target = Path(directory) / "old.json"
            target.write_text('{"a": 1}')
            item = b.artifact(target)
            self.assertEqual(b.verified(item), {"a": 1})
            target.write_text('{"a": 2}')
            with self.assertRaisesRegex(ValueError, "Changed"):
                b.verified(item)

    def test_execution_guard_requires_completed_audited_same_study(self):
        valid = {"complete": True, "all_prescribed_audits_pass": True, "full_inner_replay": True,
                 "manifest": b.ACTIVE_BINDING}
        b.validate_ready_status(valid)
        for key in valid:
            invalid = dict(valid)
            invalid[key] = False
            with self.assertRaises(ValueError):
                b.validate_ready_status(invalid)

    def test_unfinished_study_guard_precedes_models_and_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(b, "ROOT", Path(directory)):
            with self.assertRaisesRegex(ValueError, "has not finished"):
                r.execute("equivalence", "missing_plan", b.OUTPUT + "/not_created")
            self.assertFalse((Path(directory) / b.OUTPUT).exists())

    def test_trace_changes_and_missing_jobs_rejected(self):
        record = {"trace": [{"state_hash": "a", "action": {"type": "BUY"}}], "final_state": {"winner": 0},
                  "turns": 1, "reason": "completed"}
        r.compare_traces([record], [copy.deepcopy(record)])
        for field, value in (("final_state", {}), ("reason", "cutoff"), ("turns", 2), ("trace", [])):
            changed = copy.deepcopy(record)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "mismatch"):
                r.compare_traces([record], [changed])
        with self.assertRaisesRegex(ValueError, "Missing"):
            r.compare_traces([record], [])


class TimingTests(unittest.TestCase):
    def rows(self):
        schedule = b.timing_schedule(["one", "two"])
        rows = [{"root": block["root"], "repetition": block["repetition"], "mode": mode,
                 "collection_wall": 10 if mode == "serial" else 5, "audit_wall": 10,
                 "collection_cpu": 9 if mode == "serial" else 4, "audit_cpu": 9}
                for block in schedule for mode in block["order"]]
        return schedule, rows

    def test_audit_cost_included_in_speed_ratio(self):
        schedule, rows = self.rows()
        summary = r.timing_summary(rows, schedule)
        self.assertEqual(summary["collection_ratio"], 2)
        self.assertAlmostEqual(summary["root_collection_plus_audit_ratio"], 4/3)
        self.assertFalse(summary["full_pipeline_speedup_measured"])

    def test_missing_duplicate_and_nonfinite_timing_rejected(self):
        schedule, rows = self.rows()
        for invalid in (rows[:-1], rows + rows[:1], [{**rows[0], "collection_wall": float("nan")}] + rows[1:]):
            with self.assertRaises(ValueError):
                r.timing_summary(invalid, schedule)


if __name__ == "__main__":
    unittest.main()
