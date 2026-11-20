"""Model-free exact-science coverage and narrow control continuation tests.

The already-qualified scientific/observer oracles are stubbed only for toy
snapshots. Their invocation counts and error propagation remain explicit.
Original trusted-clock validation and failure classification run unmodified.
"""
from contextlib import ExitStack
from copy import deepcopy
import unittest
from unittest.mock import patch

from tools.splendorust_supervisor_qualification_20261026 import assessment as a
from test_splendorust_supervisor_qualification_panel import make_plan,complete_rows,fail


class AssessmentTests(unittest.TestCase):
    def assess(self,rows,plan=None,**kwargs):
        with ExitStack() as stack:
            boundaries=stack.enter_context(patch.object(a.audit,'validate_boundary_row',return_value={}))
            results=stack.enter_context(patch.object(a.science,'validate_result',return_value={}))
            transitions=stack.enter_context(patch.object(a.science,'audit_decision',return_value={'transition_checked':True}))
            value=a.validate_timing_rows(rows,make_plan() if plan is None else plan,**kwargs)
            self.assertEqual(boundaries.call_count,len(rows))
            self.assertEqual(results.call_count,sum(row['completed'] for row in rows))
            self.assertEqual(transitions.call_count,sum(row['completed'] for row in rows))
            return value

    def test_complete_panel_has_48_two_isolated_cells_and_92_actual_checks(self):
        value=self.assess(complete_rows())
        self.assertTrue(value['complete']);self.assertEqual(value['completed'],192)
        self.assertEqual(value['fully_replicated_isolated_groups'],48)
        self.assertEqual(value['isolated_historical_comparisons'],46)
        self.assertEqual(value['control_historical_comparisons'],46)
        self.assertEqual(value['historical_comparisons'],92)

    def test_all_96_control_timeouts_leave_46_actual_historical_checks(self):
        rows=complete_rows()
        for row in rows:
            if row['placement_mode']=='original':fail(row)
        before=deepcopy(rows);value=self.assess(rows)
        self.assertEqual(rows,before);self.assertTrue(value['complete'])
        self.assertEqual(value['isolated_completed'],96);self.assertEqual(value['control_completed'],0)
        self.assertEqual(value['control_timeout_count'],96)
        self.assertEqual(value['fully_replicated_isolated_groups'],48)
        self.assertEqual(value['fully_replicated_groups'],0)
        self.assertEqual(value['historical_comparisons'],46)
        self.assertTrue(all(record['clock']['result'] is None for record in value['control_timeouts']))
        self.assertEqual([record['index'] for record in value['control_timeouts']],
            [row['index'] for row in rows if row['placement_mode']=='original'])

    def test_historical_count_follows_actual_successful_matching_quota(self):
        for index,expected in ((0,91),(1,92)):
            rows=complete_rows();fail(rows[index])
            value=self.assess(rows)
            self.assertEqual(value['historical_comparisons'],expected)
            self.assertEqual(value['isolated_historical_comparisons'],46)
            self.assertEqual(value['control_historical_comparisons'],expected-46)

    def test_original_timeouts_anywhere_do_not_stop_later_isolated_checks(self):
        rows=complete_rows()
        for index in (0,23,72,95,96,119,168,191):fail(rows[index])
        value=self.assess(rows)
        self.assertTrue(value['complete']);self.assertEqual(value['control_timeout_count'],8)
        self.assertEqual(value['isolated_completed'],96)

    def test_valid_prefix_must_end_in_first_isolated_failure(self):
        rows=complete_rows();fail(rows[0]);fail(rows[24],status='worker_error',detail='ONE_SECOND_INTERNAL_DEADLINE',received=.98)
        value=self.assess(rows[:25],allow_prefix=True)
        self.assertFalse(value['complete']);self.assertEqual(value['failure']['index'],24)
        self.assertEqual(value['failure']['placement_mode'],'isolated')
        self.assertEqual(value['control_timeout_count'],1)
        with self.assertRaisesRegex(ValueError,'continued after isolated'):self.assess(rows[:26],allow_prefix=True)
        with self.assertRaisesRegex(ValueError,'must end'):self.assess(rows[:24],allow_prefix=True)

    def test_failed_original_output_and_unclassified_control_failure_are_fatal(self):
        rows=complete_rows();fail(rows[0]);rows[0]['action']={'invented':True}
        with self.assertRaisesRegex(ValueError,'invent'):self.assess(rows)
        rows=complete_rows();fail(rows[0],status='worker_error',detail='PLAYER_ILLEGAL_ACTION',received=.2)
        with self.assertRaisesRegex(ValueError,'Non-timeout'):self.assess(rows)
        rows=complete_rows();fail(rows[0],status='resource_guard',detail='RAM')
        with self.assertRaises(ValueError):self.assess(rows)

    def test_every_successful_control_must_match_isolated_full_science(self):
        rows=complete_rows();row=rows[10]  # Original failed root has no historical target.
        row['action']={'different':True};row['clock']['result']['action']=deepcopy(row['action'])
        with self.assertRaisesRegex(ValueError,'Placement/repeat'):self.assess(rows)

    def test_second_isolated_success_must_match_first_even_with_both_controls_failed(self):
        rows=complete_rows()
        for row in rows:
            if row['fixture_index']==10 and row['quota']==64 and row['placement_mode']=='original':fail(row)
        row=next(row for row in rows if row['fixture_index']==10 and row['quota']==64 and
            row['placement_mode']=='isolated' and row['pass_index']==1)
        row['after']={'different':True}
        with self.assertRaisesRegex(ValueError,'Placement/repeat'):self.assess(rows)

    def test_historical_target_mismatch_and_missing_isolated_repeat_fail(self):
        plan=make_plan();plan['fixtures'][0]['historical']['after']={'different':True}
        with self.assertRaisesRegex(ValueError,'historical result'):self.assess(complete_rows(),plan)
        rows=complete_rows();rows.pop(24)
        with self.assertRaises(ValueError):self.assess(rows,allow_prefix=True)

    def test_scientific_or_observer_errors_propagate_instead_of_becoming_controls(self):
        rows=complete_rows();fail(rows[0]);rows[0]['placement_error']='corrupt placement'
        with self.assertRaisesRegex(ValueError,'evidence failure'):self.assess(rows)
        rows=complete_rows();fail(rows[0])
        with patch.object(a.audit,'validate_boundary_row',side_effect=ValueError('raw/profile mismatch')):
            with self.assertRaisesRegex(ValueError,'raw/profile mismatch'):a.validate_timing_rows(rows,make_plan())
        with patch.object(a.audit,'validate_boundary_row'),patch.object(a.science,'validate_result',side_effect=ValueError('scientific mismatch')):
            with self.assertRaisesRegex(ValueError,'scientific mismatch'):a.validate_timing_rows(complete_rows(),make_plan())


class ControlTimeoutTests(unittest.TestCase):
    def test_only_real_parent_timeout_or_internal_deadline_can_continue(self):
        for status,detail,received in (('timeout',None,None),('timeout',None,1.0235925),
            ('worker_error','TimeoutError: ONE_SECOND_INTERNAL_DEADLINE',.98)):
            row=fail(complete_rows()[0],status=status,detail=detail,received=received)
            self.assertEqual(a.control_timeout(row),status)

    def test_control_helper_rejects_success_isolated_and_protocol_failures(self):
        variants=[complete_rows()[0],fail(complete_rows()[24]),
            fail(complete_rows()[0],status='worker_error',detail='PLAYER_ILLEGAL_ACTION',received=.2),
            fail(complete_rows()[0],status='illegal',detail='ONE_SECOND_INTERNAL_DEADLINE',received=.2)]
        for row in variants:
            with self.subTest(row=row['clock']['status'],mode=row['placement_mode']):
                with self.assertRaises(ValueError):a.control_timeout(row)

    def test_outer_timeout_must_have_expired_without_a_timely_receipt(self):
        row=fail(complete_rows()[0],received=.98)
        with self.assertRaisesRegex(ValueError,'outer-timeout clock'):a.control_timeout(row)
        row=fail(complete_rows()[0]);row['clock']['elapsed_seconds']=.99
        with self.assertRaisesRegex(ValueError,'outer-timeout clock'):a.control_timeout(row)
        row=fail(complete_rows()[0],received=1.)
        with self.assertRaisesRegex(ValueError,'outer-timeout clock'):a.control_timeout(row)
        # A real internal guard error may legitimately arrive before one second.
        row=fail(complete_rows()[0],status='worker_error',detail='ONE_SECOND_INTERNAL_DEADLINE',received=.98)
        row['clock']['elapsed_seconds']=.99
        self.assertEqual(a.control_timeout(row),'worker_error')

    def test_control_helper_rejects_integrity_prerequisite_changed_clock_and_invented_result(self):
        variants=[]
        row=fail(complete_rows()[0]);row['profile_capture_error']='lost';variants.append(row)
        row=fail(complete_rows()[0]);row['clock']['detail']='resource unavailable';variants.append(row)
        row=fail(complete_rows()[0]);row['clock']['deadline_seconds']=.9;variants.append(row)
        row=fail(complete_rows()[0]);row['clock']['result']={'invented':True};variants.append(row)
        row=fail(complete_rows()[0]);row['clock']['accepted']=True;variants.append(row)
        for row in variants:
            with self.assertRaises(ValueError):a.control_timeout(row)


if __name__=='__main__':unittest.main()
