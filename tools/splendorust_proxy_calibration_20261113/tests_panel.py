"""Model-free panel selection, isolation and fixed-design counterexamples."""
from collections import Counter
from copy import deepcopy
import unittest

from . import panel


class PanelTests(unittest.TestCase):
    def roots(self, n=55, game=1024):
        return [dict(game_index=game, ply=i, split='validation', row_sha256='unused', root_input_sha256='unused') for i in range(n)]

    def fake_panel(self):
        return {'roots': [{'root_id': panel.root_id(g, p)} for p in (3, 17, 31, 49) for g in range(1024, 1152)]}

    def test_four_nonempty_equal_count_bands(self):
        for n in (4, 5, 7, 55, 123):
            selected = panel.choose_four(self.roots(n), 1024)
            self.assertEqual(len({x['ply'] for x in selected}), 4)
            for band, x in enumerate(selected):
                self.assertEqual(x['quantile_band'], band)
                self.assertTrue(band*n//4 <= x['ply'] < (band+1)*n//4)

    def test_outcomes_actions_values_timing_cannot_select(self):
        original = self.roots()
        contaminated = deepcopy(original)
        for i, x in enumerate(contaminated):
            x.update(outcome=i % 2, action={'arbitrary': i}, value=i / 55, elapsed_seconds=999-i)
        self.assertEqual(panel.choose_four(original, 1024), panel.choose_four(contaminated, 1024))

    def test_inventory_input_order_irrelevant(self):
        self.assertEqual(panel.choose_four(self.roots(), 1024), panel.choose_four(list(reversed(self.roots())), 1024))

    def test_training_games_rejected(self):
        with self.assertRaises(ValueError): panel.choose_four(self.roots(game=1), 1)
        rows=self.roots(); rows[0]['split']='train'
        with self.assertRaises(ValueError): panel.choose_four(rows, 1024)

    def test_short_or_duplicate_inventory_fails(self):
        for rows in (self.roots(3), self.roots(4)+self.roots(1)):
            with self.assertRaises(ValueError): panel.choose_four(rows, 1024)

    def test_late_or_wrong_game_rejected(self):
        for change in ({'ply':124}, {'game_index':1025}):
            rows=self.roots(); rows[0].update(change)
            with self.assertRaises(ValueError): panel.choose_four(rows, 1024)

    def test_exact_balanced_matrix(self):
        rows=panel.request_matrix(self.fake_panel())
        self.assertEqual(len(rows),5120)
        self.assertEqual(len({x['request_id'] for x in rows}),5120)
        self.assertEqual(Counter((x['quota'],x['seed_label']) for x in rows), {x[:2]:512 for x in panel.CONDITIONS})
        self.assertTrue(all(set(x)=={'request_id','root_id','quota','seed_label','role'} for x in rows))

    def test_condition_rotation_and_disjoint_reference_families(self):
        rows=panel.request_matrix(self.fake_panel())
        for i in range(10):
            self.assertEqual((rows[10*i]['quota'],rows[10*i]['seed_label'],rows[10*i]['role']),panel.CONDITIONS[i])
        self.assertEqual({x['seed_label'] for x in rows if x['role']=='control'},{'A','B'})
        self.assertEqual({x['seed_label'] for x in rows if x['role']=='reference'},{'R','S'})

    def test_reproductions_are_prospective_balanced_and_identical(self):
        doc=self.fake_panel(); rows=panel.reproduction_matrix(doc)
        self.assertEqual(len(rows),32)
        self.assertEqual(len({x['root_id'] for x in rows}),16)
        self.assertEqual(Counter(x['quota'] for x in rows),{64:4,128:4,256:4,512:4,2048:16})
        self.assertEqual(Counter(x['seed_label'] for x in rows),{'A':8,'B':8,'R':8,'S':8})
        original={x['request_id']:x for x in panel.request_matrix(doc)}
        for x in rows:
            expected=deepcopy(original[x['original_request_id']]); expected['request_id']=x['request_id'];expected['original_request_id']=x['original_request_id']
            self.assertEqual(x,expected)
        reversed_doc={'roots':list(reversed(doc['roots']))}
        self.assertEqual(rows,panel.reproduction_matrix(reversed_doc))


if __name__ == '__main__':
    unittest.main()
