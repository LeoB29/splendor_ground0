"""Model-free prospective absolute gate and retained control outcomes."""
from copy import deepcopy
import unittest
from tools.splendorust_supervisor_qualification_20261026 import panel


def science_value(index,quota):
    action={'chosen':[index,quota]};trace={'work':quota}
    return dict(action=action,trace=trace,after={'fixture_successor':[index,quota]},
        result={'action':deepcopy(action),'diagnostics':deepcopy(trace)})


def make_plan():
    fixtures=[]
    for index,(stage,game,ply) in enumerate(panel.FOCUS+panel.CONTROLS):
        quota=64 if index%2==0 else 32
        kind='failed_saved_position' if index==10 else 'slow_completed' if index<10 else 'control'
        fixtures.append(dict(fixture_id=f'{stage}_g{game}_p{ply}',source_identity=[stage,game,ply],kind=kind,
            snapshot={'fixture':index},observer={'fixture_observer':index},original_quota=quota,
            historical=None if index==10 else dict(quota=quota,**science_value(index,quota))))
    return dict(schema='splendorust_supervisor_qualification_panel_v1',source_freeze_sha256=panel.SOURCE_FREEZE,
        fixtures=fixtures,schedule=panel.make_schedule(fixtures),isolated_historical_comparisons_required=46,
        control_historical_comparisons_maximum=46,agreement_cells_expected=48,isolated_agreements_per_cell=2,
        control_timeout_policy='retain_once_continue_next_unattempted_root',relative_comparison_is_admission_gate=False)


def complete_rows(plan=None):
    plan=make_plan() if plan is None else plan
    rows=[]
    for spec in plan['schedule']:
        fixture=plan['fixtures'][spec['fixture_index']];value=science_value(spec['fixture_index'],spec['quota'])
        duration=.4 if spec['quota']==64 else .25
        request_id=f"request-{spec['index']+1:08d}"
        rows.append(dict(**spec,arm='search',mode='candidate' if spec['quota']==64 else 'baseline',
            observation_mode='profiled',completed=True,before=deepcopy(fixture['snapshot']),observer=deepcopy(fixture['observer']),
            action=value['action'],trace=value['trace'],after=value['after'],
            worker_clock_file=f'owned/request-{spec["index"]+1:08d}.json',
            clock=dict(clock='perf_counter_ns',request_id=request_id,status='ok',accepted=True,
                received_before_deadline=True,deadline_seconds=1.,elapsed_seconds=duration+.02,
                receipt_elapsed_seconds=duration,result=value['result'],detail=None)))
    return rows


def fail(row,status='timeout',detail=None,received=None):
    row['completed']=False;row['action']=row['trace']=row['after']=None
    row['clock'].update(status=status,accepted=False,result=None,detail=detail,
        received_before_deadline=received is not None and 0<=received<=1.,
        elapsed_seconds=1.05 if received is None else max(1.05,received+.01),receipt_elapsed_seconds=received)
    return row


class PanelTests(unittest.TestCase):
    def test_fixed_roots_and_exact_eight_balanced_blocks_unchanged(self):
        plan=make_plan();panel.validate_plan(plan)
        self.assertEqual(len(set(panel.FOCUS+panel.CONTROLS)),24)
        self.assertEqual(panel.FOCUS[-1],('main',72,28))
        self.assertEqual(panel.CONTROLS[:11],[(s,g,p-1) for s,g,p in panel.FOCUS])
        self.assertEqual(len(plan['schedule']),192)
        for block in range(8):
            group=[row for row in plan['schedule'] if row['block_index']==block]
            self.assertEqual(len(group),24)
            self.assertEqual(len({(row['quota'],row['placement_mode'],row['pass_index']) for row in group}),1)
        self.assertEqual([row['fixture_index'] for row in plan['schedule'][:24]],list(range(24)))
        self.assertEqual([row['fixture_index'] for row in plan['schedule'][48:72]],list(reversed(range(24))))

    def test_complete_isolated_absolute_gate_passes(self):
        value=panel.timing_gate(complete_rows())
        self.assertTrue(value['qualified']);self.assertEqual(value['isolated_completed'],96)
        self.assertTrue(value['complete_inventory']);self.assertEqual(value['retained_control_timeouts'],0)

    def test_any_or_all_control_timeouts_retained_without_vetoing_isolated(self):
        for all_controls in (False,True):
            rows=complete_rows();controls=[row for row in rows if row['placement_mode']=='original']
            for row in controls if all_controls else controls[:1]:fail(row)
            before=deepcopy(rows);gate=panel.timing_gate(rows)
            self.assertTrue(gate['qualified']);self.assertEqual(rows,before)
            self.assertEqual(gate['retained_control_timeouts'],96 if all_controls else 1)
            self.assertEqual(gate['per_quota']['64']['relative_comparison']['status'],'not_estimable_with_control_timeouts')
            self.assertIsNone(gate['per_quota']['64']['relative_comparison']['mean_ratio'])
            if all_controls:
                self.assertEqual(gate['per_quota']['64']['distributions']['original']['count'],0)
                self.assertIsNone(gate['per_quota']['64']['distributions']['original']['mean'])

    def test_completed_controls_only_are_labelled_and_not_full_comparison(self):
        rows=complete_rows();fail(rows[0],status='worker_error',detail='ONE_SECOND_INTERNAL_DEADLINE',received=.98)
        value=panel.timing_gate(rows)['per_quota']['64']
        self.assertEqual(value['distributions']['original']['count'],47)
        self.assertEqual(value['distribution_scope']['original'],'completed_controls_only_conditional_on_success')
        self.assertEqual(value['original_attempts'],48)
        self.assertEqual(value['relative_comparison']['status'],'not_estimable_with_control_timeouts')
        self.assertFalse(value['relative_comparison']['is_admission_gate'])

    def test_relative_mean_and_p95_nonregression_are_only_descriptive(self):
        rows=complete_rows()
        for row in rows:
            if row['placement_mode']=='isolated':
                row['clock']['receipt_elapsed_seconds']=.6;row['clock']['elapsed_seconds']=.62
        gate=panel.timing_gate(rows)
        self.assertTrue(gate['qualified'])
        relative=gate['per_quota']['64']['relative_comparison']
        self.assertFalse(relative['mean_nonregression']);self.assertFalse(relative['p95_nonregression'])
        self.assertFalse(relative['is_admission_gate'])

    def test_700ms_margin_negative_keeps_valid_move_unchanged(self):
        rows=complete_rows();row=next(row for row in rows if row['placement_mode']=='isolated')
        row['clock']['receipt_elapsed_seconds']=.7;row['clock']['elapsed_seconds']=.72
        before=deepcopy(rows);self.assertFalse(panel.timing_gate(rows)['qualified'])
        self.assertEqual(rows,before);self.assertTrue(row['completed'])

    def test_isolated_failure_and_incomplete_prefix_cannot_pass(self):
        rows=complete_rows();fail(rows[24],status='worker_error',detail='ONE_SECOND_INTERNAL_DEADLINE',received=.98)
        self.assertFalse(panel.timing_gate(rows[:25])['qualified'])
        self.assertFalse(panel.timing_gate(complete_rows()[:-1])['qualified'])

    def test_order_and_duplicate_inventory_cannot_pass(self):
        rows=complete_rows();rows[0],rows[1]=rows[1],rows[0]
        self.assertFalse(panel.timing_gate(rows)['qualified'])
        rows=complete_rows();rows[-1]=deepcopy(rows[0])
        self.assertFalse(panel.timing_gate(rows)['qualified'])

    def test_non_timeout_control_and_changed_clock_are_rejected(self):
        rows=complete_rows();fail(rows[0],status='worker_error',detail='PLAYER_ILLEGAL_ACTION',received=.2)
        with self.assertRaisesRegex(ValueError,'Non-timeout'):panel.timing_gate(rows)
        rows=complete_rows();rows[0]['clock']['deadline_seconds']=1.1
        with self.assertRaises(ValueError):panel.timing_gate(rows)

    def test_no_nonfinite_or_negative_timing_imputation(self):
        for value in (float('nan'),float('inf'),-1):
            with self.assertRaisesRegex(ValueError,'Finite'):panel.distribution([value])
        self.assertEqual(panel.distribution([]),dict(count=0,mean=None,p95=None,max=None))


if __name__=='__main__':unittest.main()
