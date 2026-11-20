"""Model-free integrity and eligibility tests for the concurrency saved auditor."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.splendorust_gpu_throughput_20261105 import audit


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
        with patch.object(audit, "root_output", return_value=[1.]), patch("tools.splendorust_gpu_throughput_20261105.scientific.normalize_result", side_effect=lambda value:value):
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
        clock = dict(internal_seconds=60., outer_seconds=75., execution_policy_id=audit.c.EXECUTION_POLICY_ID, no_retry=True, elapsed_seconds=1.,
                     started_perf_counter=10., ended_perf_counter=12., status="ok", completed=True)
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            audit.validate_clock(clock)


class RelaxedClockTests(unittest.TestCase):
    def clock(self, elapsed):
        return dict(internal_seconds=60., outer_seconds=75., execution_policy_id=audit.c.EXECUTION_POLICY_ID,
                    no_retry=True, elapsed_seconds=elapsed, started_perf_counter=0., ended_perf_counter=elapsed,
                    status="ok", completed=True)

    def test_late_relative_to_previous_clocks_remains_valid(self):
        self.assertTrue(audit.validate_clock(self.clock(10.1)))
        self.assertEqual(audit.descriptive_latency([.5, 9.5, 10.1])["exceeds"], {"1": 2, "9": 2, "10": 1})

    def test_old_policy_cannot_be_labeled_relaxed(self):
        clock = self.clock(5.)
        clock["internal_seconds"] = 9.
        with self.assertRaises(ValueError):
            audit.validate_clock(clock)

    def test_late_outer_success_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_clock(self.clock(75.))

    def test_policy_identity_required(self):
        clock = self.clock(5.)
        clock["execution_policy_id"] = "other"
        with self.assertRaises(ValueError):
            audit.validate_clock(clock)


class ProjectionTests(unittest.TestCase):
    def result(self):
        from tools.splendorust_gpu_throughput_20261105 import scientific as s
        value = {key: None for key in s.RESULT_FIELDS}
        value.update(simulations_quota=512, profile_id=s.original.profile_id(512, route="candidate"),
                     profile_sha256=s.original.profile_sha256(512, route="candidate"), wrapper_id=s.wrapper_id(512),
                     diagnostics={"trace": [1, 2]}, action={"unchanged": True})
        return value

    def test_only_wrapper_identity_is_normalized(self):
        from tools.splendorust_gpu_throughput_20261105 import scientific as s
        value = self.result(); before = deepcopy(value); projected = s.normalize_result(value)
        self.assertEqual(value, before)
        self.assertEqual({k:v for k,v in projected.items() if k != "wrapper_id"},
                         {k:v for k,v in value.items() if k != "wrapper_id"})
        self.assertEqual(projected["wrapper_id"], s.original.wrapper_id(512, route="candidate"))

    def test_old_wrapper_rejected_as_new_evidence(self):
        from tools.splendorust_gpu_throughput_20261105 import scientific as s
        value = s.normalize_result(self.result())
        with self.assertRaisesRegex(ValueError, "wrapper identity"):
            s.normalize_result(value)
        self.assertEqual(s.source_result(value, legacy=True), value)

    def test_actual_prior_c1_result_matches_only_via_explicit_legacy_path(self):
        from tools.splendorust_gpu_throughput_20261105 import scientific as s
        root=audit.c.ROOT/"outputs/splendorust_gpu_concurrency_20261105_01"
        case=audit.read(root/"plan.json")["cases"][0]
        old=audit.read(root/"cells/cpu_c1/worker/cell/g000/result.json")
        new=deepcopy(old);new["wrapper_id"]=s.wrapper_id(512)
        self.assertTrue(audit.exact_baseline(case,new,old,legacy=True))
        with self.assertRaisesRegex(ValueError,"wrapper identity"):
            audit.root_output(case,old)


class GamePrefixTests(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding="utf-8")

    def prepared(self,folder):
        from tools.splendorust_gpu_throughput_20261105 import games
        plan={"games":games.build_plan()}
        for spec in plan["games"]["specifications"]:
            self.write(folder/spec["game_id"]/"spec.json",spec)
            self.write(folder/spec["game_id"]/"initial.json",games.initial_snapshot(spec))
        requests={"warmup:"+str(i+1):dict(client_id="warmup") for i in range(5)}
        return plan,requests

    def attempt(self,folder,plan,requests,*,deadline=61.,input_hash=None):
        from tools.external_strength_screen_20261018 import rules
        from tools.future_training_rules_20261021 import decode_training_snapshot
        from tools.splendorust_policy_head_20261104.capture_adapter import public_root_tokens
        before=audit.read(folder/"g000/initial.json")
        observer=rules.observation_payload(decode_training_snapshot(before));request=dict(player="search",observer=observer)
        self.write(folder/"g000/started.json",dict(game_id="g000",slot=0,wave_index=0,started_perf_counter=.5))
        self.write(folder/"waves/000/start.json",dict(wave_index=0,game_indexes=[0],started_perf_counter=.5))
        self.write(folder/"g000/requests/000000.json",request)
        self.write(folder/"g000/attempts/000000.json",dict(game_id="g000",ply=0,client_id="game_g000",slot=0,
            wave_index=0,started_perf_counter=1.,before_sha256=audit.digest(before),request_sha256=audit.digest(request)))
        self.write(folder/"g000/clocks/000000.json",dict(status="error",completed=False,started_perf_counter=1.,
            ended_perf_counter=62.,elapsed_seconds=61.,internal_seconds=60.,outer_seconds=75.,no_retry=True,
            execution_policy_id=audit.c.EXECUTION_POLICY_ID))
        self.write(folder/"g000/service_calls/000000.json",[])
        requests["game_g000:1"]=dict(client_id="game_g000",request_id="game_g000:1",slot=0,deadline_perf=deadline,
            submitted_perf=1.1,complete_readback_perf=1.2,input_sha256=input_hash or audit.digest(public_root_tokens(observer)))

    def test_prepared_eight_games_are_zero_started_games(self):
        with tempfile.TemporaryDirectory() as value:
            folder=Path(value);plan,requests=self.prepared(folder)
            proof=audit.audit_game_callbacks(plan,"cpu",folder,requests)
            self.assertEqual(sum(row["started"] for row in proof["games"]),0)
            self.assertEqual(proof["attempted_decisions"],0)

    def test_failed_prefix_retained_without_completed_move(self):
        with tempfile.TemporaryDirectory() as value:
            folder=Path(value);plan,requests=self.prepared(folder);self.attempt(folder,plan,requests)
            proof=audit.audit_game_callbacks(plan,"cpu",folder,requests)
            self.assertEqual(proof["attempted_decisions"],1)
            self.assertEqual(proof["completed_moves"],0)
            self.assertEqual(proof["game_request_ids"],["game_g000:1"])

    def test_original_deadline_cannot_reset_or_keep_old_nine_seconds(self):
        with tempfile.TemporaryDirectory() as value:
            folder=Path(value);plan,requests=self.prepared(folder);self.attempt(folder,plan,requests,deadline=10.)
            with self.assertRaisesRegex(ValueError,"clock reset"):
                audit.audit_game_callbacks(plan,"cpu",folder,requests)

    def test_partial_callback_input_must_match_corrected_root(self):
        with tempfile.TemporaryDirectory() as value:
            folder=Path(value);plan,requests=self.prepared(folder);self.attempt(folder,plan,requests,input_hash="wrong")
            with self.assertRaisesRegex(ValueError,"corrected public root"):
                audit.audit_game_callbacks(plan,"cpu",folder,requests)


class FailureClassificationTests(unittest.TestCase):
    def wrapped(self, elapsed=60.5, message="Completed inference exceeded original callback deadline"):
        return dict(case_id="g000", status="cancelled", clock=dict(status="error", completed=False,
            internal_seconds=60., outer_seconds=75., started_perf_counter=0., ended_perf_counter=elapsed, elapsed_seconds=elapsed),
            error_chain=[dict(type="RuntimeError", message="Inference request failed: case_g000:1"),
                         dict(type="TimeoutError", message=message)])

    def test_wrapped_timeout_with_own_clock_expired_is_authenticated(self):
        initial = self.wrapped()
        proof = audit.wrapped_deadline_proof(initial)
        final = dict(initial, status="deadline_failed", initial_status="cancelled", deadline_proof=proof)
        self.assertEqual(audit.validate_group_failures([final], dict(g000=initial)), ["g000"])

    def test_wrapped_timeout_before_own_deadline_is_only_peer(self):
        self.assertIsNone(audit.wrapped_deadline_proof(self.wrapped(59.5)))

    def test_wrapped_outer_timeout_requires_original_seventy_five_seconds(self):
        self.assertIsNone(audit.wrapped_deadline_proof(self.wrapped(60.5, "Whole completed operation exceeded dual-clock75s bound")))

    def test_worker_resolution_and_independent_audit_agree_on_wrapped_poison(self):
        from tools.splendorust_gpu_throughput_20261105 import cell
        initial = self.wrapped()
        error = RuntimeError("Inference request failed: case_g000:1")
        error.__cause__ = TimeoutError("Completed inference exceeded original callback deadline")
        peer = dict(case_id="g001", status="integrity_failed", clock=deepcopy(initial["clock"]),
                    error_chain=[dict(type="RuntimeError", message="Evaluator aborted; no replacement or retry")])
        rows = {0: deepcopy(initial), 1: deepcopy(peer)}
        self.assertEqual(cell.resolve_group_outcomes(rows, {0: error, 1: RuntimeError("Evaluator aborted; no replacement or retry")}), "deadline_failed")
        self.assertEqual(audit.validate_group_failures(list(rows.values()), dict(g000=initial, g001=peer)), ["g000"])

    def group(self):
        clock = dict(status="error", completed=False, internal_seconds=60., outer_seconds=75.)
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
