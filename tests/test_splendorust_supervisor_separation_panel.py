"""Model-free tests of fixed selection, comparison inventory and admission."""
from copy import deepcopy
import unittest
from tools.splendorust_supervisor_separation_20261025 import panel


def fixtures():
    return [dict(fixture_id=f'{s}_g{g}_p{p}') for s,g,p in panel.FOCUS+panel.CONTROLS]


def complete_rows():
    return [dict(**row,completed=True,clock=dict(status='ok',deadline_seconds=1.,
        receipt_elapsed_seconds=.4 if row['quota']==64 else .25))
        for row in panel.make_schedule(fixtures())]


class PanelTests(unittest.TestCase):
    def test_fixed_selection_and_distinct_controls(self):
        self.assertEqual(len(set(panel.FOCUS+panel.CONTROLS)),24)
        self.assertEqual(panel.FOCUS[-1],('main',72,28))
        self.assertEqual(panel.CONTROLS[:11],[(s,g,p-1) for s,g,p in panel.FOCUS])
        self.assertEqual(panel.CONTROLS[-2:],[('main',72,0),('main',71,52)])

    def test_eight_closed_worker_blocks_and_reversed_order(self):
        rows=panel.make_schedule(fixtures())
        self.assertEqual(len(rows),192)
        for block in range(8):
            group=[r for r in rows if r['block_index']==block]
            self.assertEqual(len(group),24)
            self.assertEqual(len({(r['quota'],r['placement_mode'],r['pass_index']) for r in group}),1)
        self.assertEqual([r['fixture_index'] for r in rows[:24]],list(range(24)))
        self.assertEqual([r['fixture_index'] for r in rows[48:72]],list(reversed(range(24))))
        for quota in panel.QUOTAS:
            for fixture in fixtures():
                cell=[r for r in rows if r['quota']==quota and r['fixture_id']==fixture['fixture_id']]
                self.assertEqual({(r['pass_index'],r['placement_mode']) for r in cell},
                    {(0,'original'),(0,'isolated'),(1,'original'),(1,'isolated')})

    def test_complete_nonregression_passes(self):
        self.assertTrue(panel.timing_gate(complete_rows())['qualified'])

    def test_original_exact_deadline_acceptance_matches_trusted_clock(self):
        rows=complete_rows()
        next(r for r in rows if r['placement_mode']=='original')['clock']['receipt_elapsed_seconds']=1.
        self.assertTrue(panel.timing_gate(rows)['qualified'])

    def test_margin_failure_does_not_relabel_completed_move(self):
        rows=complete_rows();row=next(r for r in rows if r['placement_mode']=='isolated')
        row['clock']['receipt_elapsed_seconds']=.75
        before=deepcopy(rows);result=panel.timing_gate(rows)
        self.assertFalse(result['qualified'])
        self.assertTrue(row['completed']);self.assertEqual(row['clock']['status'],'ok')
        self.assertEqual(rows,before)

    def test_missing_duplicate_failure_or_changed_deadline_cannot_pass(self):
        rows=complete_rows()
        variants=[rows[:-1],rows[:-1]+[deepcopy(rows[0])]]
        bad=deepcopy(rows);bad[0]['completed']=False;bad[0]['clock'].update(status='worker_error',receipt_elapsed_seconds=.98)
        variants.append(bad)
        bad=deepcopy(rows);bad[0]['clock']['deadline_seconds']=1.1;variants.append(bad)
        for variant in variants:
            self.assertFalse(panel.timing_gate(variant)['qualified'])

    def test_nonfinite_duration_rejected(self):
        for value in (float('nan'),float('inf'),-1):
            with self.assertRaisesRegex(ValueError,'Finite'):panel.distribution([value])

    def test_mean_and_p95_limits_are_independent(self):
        rows=complete_rows()
        for row in rows:
            if row['placement_mode']=='isolated' and row['quota']==64:
                row['clock']['receipt_elapsed_seconds']=.424
        result=panel.timing_gate(rows)['per_quota']['64']['conditions']
        self.assertFalse(result['isolated_mean_nonregression'])
        self.assertTrue(result['isolated_p95_nonregression'])
        rows=complete_rows();subset=[r for r in rows if r['placement_mode']=='isolated' and r['quota']==64]
        for row in subset[:4]:row['clock']['receipt_elapsed_seconds']=.6
        result=panel.timing_gate(rows)['per_quota']['64']['conditions']
        self.assertTrue(result['isolated_mean_nonregression'])
        self.assertFalse(result['isolated_p95_nonregression'])

    def test_exact_700ms_does_not_pass_strict_margin(self):
        rows=complete_rows();next(r for r in rows if r['placement_mode']=='isolated')['clock']['receipt_elapsed_seconds']=.7
        self.assertFalse(panel.timing_gate(rows)['qualified'])

    def test_linear_percentile(self):
        self.assertAlmostEqual(panel.distribution([0.,1.])['p95'],.95)


if __name__=='__main__':unittest.main()
