"""Model-free integrity and eligibility tests for the concurrency saved auditor."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.splendorust_gpu_concurrency_20261105 import audit


class EligibilityTests(unittest.TestCase):
    def rows(self):
        return [dict(completed=True, baseline_exact=True) for _ in range(8)]

    def test_full_complete_cell_rate(self):
        self.assertEqual(audit.qualified_rate("passed", self.rows(), 4.),
                         dict(eligible=True, completed_decisions_per_second=2.))

    def test_failed_cell_has_no_partial_rate_even_with_completed_peers(self):
        rows = self.rows()
        rows[-1]["completed"] = False
        self.assertIsNone(audit.qualified_rate("deadline_failed", rows, 2.)["completed_decisions_per_second"])

    def test_missing_baseline_cannot_qualify(self):
        rows = self.rows()
        rows[2]["baseline_exact"] = False
        self.assertFalse(audit.qualified_rate("passed", rows, 1.)["eligible"])

    def test_missing_eighth_case_cannot_qualify(self):
        self.assertFalse(audit.qualified_rate("passed", self.rows()[:7], 1.)["eligible"])

    def test_killed_c1_cannot_admit_eight_retained_outputs(self):
        retained = {f"g{i:03d}": dict(action="retained") for i in range(8)}
        actual = audit.admitted_baseline_results(dict(concurrency=1), "deadline_failed", dict(final_weights_verified=False), retained)
        self.assertEqual(actual, {})
        self.assertEqual(len(retained), 8)

    def test_selection_requires_cross_route_numeric_gate(self):
        cell = dict(cell_id="cpu_c1", route="cpu", concurrency=1, eligible=True, completed_decisions_per_second=2.)
        self.assertEqual(audit.select_eligible([cell], dict(passed=False)), dict(cpu=None, gpu=None))

    def test_selection_tie_prefers_lower_concurrency_and_ignores_failed(self):
        cells = [dict(cell_id="cpu_c"+str(n), route="cpu", concurrency=n, eligible=True,
                      completed_decisions_per_second=2.) for n in (4, 1, 2)]
        cells.append(dict(cell_id="gpu_c8", route="gpu", concurrency=8, eligible=False, completed_decisions_per_second=None))
        selected = audit.select_eligible(cells, dict(passed=True))
        self.assertEqual(selected["cpu"]["concurrency"], 1)
        self.assertIsNone(selected["gpu"])

    def test_matching_action_does_not_hide_changed_scientific_trace(self):
        with patch.object(audit, "root_output", return_value=[1.]):
            with self.assertRaisesRegex(ValueError, "full scientific"):
                audit.exact_baseline({}, dict(action="same", diagnostics="a"), dict(action="same", diagnostics="b"))

    def test_signed_zero_root_bits_are_not_silently_equal(self):
        with patch.object(audit, "root_output", side_effect=[[0.], [-0.]]):
            with self.assertRaisesRegex(ValueError, "float32 bits"):
                audit.exact_baseline({}, {}, {})


class JournalTests(unittest.TestCase):
    def event(self):
        row = dict(route="gpu", batch_id=1, native_batch=1, physical_batch=8, occupied_slots=[6], slot=6,
                   client_id="case_g006", request_id="case_g006:1", submitted_perf=1., dispatch_perf=1.25,
                   complete_readback_perf=1.5, deadline_perf=9., queue_seconds=.25, forward_roundtrip_seconds=.25,
                   input_sha256="input", output_sha256="output")
        return dict(kind="inference_batch", route="gpu", batch_id=1, physical_batch=8, requests=[row])

    def read_event(self, event):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"inference.jsonl").write_text(json.dumps(event)+"\n", encoding="utf-8")
            return audit.read_journal(folder, "gpu")

    def test_fixed_gpu_batch_and_high_slot_valid_at_concurrency_one(self):
        _, requests = self.read_event(self.event())
        self.assertEqual(len(audit.client_rows(dict(case_id="g006", index=6), requests)), 1)

    def test_changed_slot_rejected(self):
        _, requests = self.read_event(self.event())
        with self.assertRaisesRegex(ValueError, "original slot"):
            audit.client_rows(dict(case_id="g006", index=0), requests)

    def test_expired_completed_callback_rejected(self):
        event = self.event()
        event["requests"][0]["deadline_perf"] = 1.5
        with self.assertRaisesRegex(ValueError, "timing/deadline"):
            self.read_event(event)

    def test_missing_callback_prefix_rejected(self):
        event = self.event()
        event["requests"][0]["request_id"] = "case_g006:2"
        _, requests = self.read_event(event)
        with self.assertRaisesRegex(ValueError, "original order"):
            audit.client_rows(dict(case_id="g006", index=6), requests)

    def test_gpu_physical_batch_reduction_rejected(self):
        event = self.event()
        event["physical_batch"] = 1
        with self.assertRaisesRegex(ValueError, "fixed physical"):
            self.read_event(event)

    def test_returned_clock_arithmetic_tampering_rejected(self):
        clock = dict(internal_seconds=9., outer_seconds=10., no_retry=True, elapsed_seconds=1.,
                     started_perf_counter=10., ended_perf_counter=12., status="ok", completed=True)
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            audit.validate_clock(clock)


class FailureClassificationTests(unittest.TestCase):
    def wrapped(self, elapsed=9.5, message="Completed inference exceeded original callback deadline"):
        return dict(case_id="g000", status="cancelled", clock=dict(status="error", completed=False,
            internal_seconds=9., outer_seconds=10., started_perf_counter=0., ended_perf_counter=elapsed, elapsed_seconds=elapsed),
            error_chain=[dict(type="RuntimeError", message="Inference request failed: case_g000:1"),
                         dict(type="TimeoutError", message=message)])

    def test_wrapped_timeout_with_own_clock_expired_is_authenticated(self):
        initial = self.wrapped()
        proof = audit.wrapped_deadline_proof(initial)
        final = dict(initial, status="deadline_failed", initial_status="cancelled", deadline_proof=proof)
        self.assertEqual(audit.validate_group_failures([final], dict(g000=initial)), ["g000"])

    def test_wrapped_timeout_before_own_deadline_is_only_peer(self):
        self.assertIsNone(audit.wrapped_deadline_proof(self.wrapped(8.5)))

    def test_wrapped_outer_timeout_requires_original_ten_seconds(self):
        self.assertIsNone(audit.wrapped_deadline_proof(self.wrapped(9.5, "Whole completed operation exceeded dual-clock10s bound")))

    def test_worker_resolution_and_independent_audit_agree_on_wrapped_poison(self):
        from tools.splendorust_gpu_concurrency_20261105 import cell
        initial = self.wrapped()
        error = RuntimeError("Inference request failed: case_g000:1")
        error.__cause__ = TimeoutError("Completed inference exceeded original callback deadline")
        peer = dict(case_id="g001", status="integrity_failed", clock=deepcopy(initial["clock"]),
                    error_chain=[dict(type="RuntimeError", message="Evaluator aborted; no replacement or retry")])
        rows = {0: deepcopy(initial), 1: deepcopy(peer)}
        self.assertEqual(cell.resolve_group_outcomes(rows, {0: error, 1: RuntimeError("Evaluator aborted; no replacement or retry")}), "deadline_failed")
        self.assertEqual(audit.validate_group_failures(list(rows.values()), dict(g000=initial, g001=peer)), ["g000"])

    def group(self):
        clock = dict(status="error", completed=False, internal_seconds=9., outer_seconds=10.)
        direct = dict(case_id="g000", status="deadline_failed", clock=clock,
                      error_chain=[dict(type="TimeoutError", message="Completed inference exceeded original callback deadline")])
        poison = dict(case_id="g001", status="integrity_failed", clock=clock,
                      error_chain=[dict(type="RuntimeError", message="Evaluator aborted; no replacement or retry")])
        resolved = dict(poison, status="cancelled", initial_status="integrity_failed", cancellation_trigger_case_ids=["g000"])
        return [direct, resolved], dict(g000=direct, g001=poison)

    def test_actual_same_group_deadline_authorizes_abort_peer(self):
        self.assertEqual(audit.validate_group_failures(*self.group()), ["g000"])

    def test_fabricated_cancellation_trigger_is_rejected(self):
        final, initial = self.group()
        final[1]["cancellation_trigger_case_ids"] = ["g007"]
        with self.assertRaisesRegex(ValueError, "actual same-group"):
            audit.validate_group_failures(final, initial)

    def test_abort_without_same_group_direct_timeout_is_integrity_failure(self):
        final, initial = self.group()
        with self.assertRaisesRegex(ValueError, "same-group direct deadline"):
            audit.validate_group_failures(final[1:], dict(g001=initial["g001"]))

    def test_actual_decision_timeout_is_isolated(self):
        chain = [dict(type="TimeoutError", message="Completed inference exceeded original callback deadline")]
        self.assertEqual(audit.deadline_classification(chain), "deadline_failed")

    def test_poisoned_peer_is_distinct_from_independent_timeout(self):
        chain = [dict(type="RuntimeError", message="Inference service failed; no retry"),
                 dict(type="TimeoutError", message="Completed inference exceeded original callback deadline")]
        self.assertEqual(audit.deadline_classification(chain), "cancelled")

    def test_status_mutex_timeout_cannot_be_relabeled_decision_failure(self):
        chain = [dict(type="TimeoutError", message="Status mutex exceeded250ms acquisition bound")]
        self.assertEqual(audit.deadline_classification(chain), "integrity_failed")

    def test_real_permission_error_cannot_hide_behind_deadline_cause(self):
        chain = [dict(type="PermissionError", message="publication denied"),
                 dict(type="TimeoutError", message="Completed inference exceeded original callback deadline")]
        self.assertEqual(audit.deadline_classification(chain), "integrity_failed")


if __name__ == "__main__":
    unittest.main()
