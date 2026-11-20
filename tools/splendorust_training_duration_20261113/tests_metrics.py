"""Model-free diagnostic counterexamples; no new training or games."""
from copy import deepcopy
import math
import struct
import sys
import unittest

from . import offline as o


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def row(game, ply, split, logits=(0., 0., 0.), value=0.):
    output = [f32(x) for x in logits] + [0.] * 78 + [f32(value), f32(-value)]
    return dict(game_index=game, ply=ply, split=split, root_input_sha256='a' * 64,
                row_sha256='b' * 64, legal_mask81=[True] * 3 + [False] * 78,
                policy_target81=[.75, .25] + [0.] * 79,
                value_target2=[1., -1.], output83=output)


def label(record):
    return dict(game_index=record['game_index'], ply=record['ply'], split=record['split'],
                root_input_sha256=record['root_input_sha256'], row_sha256=record['row_sha256'],
                terminal_value=record['value_target2'], root_q_signed=.5, root_q_win_credit=.75)


def fixture():
    parent = {}
    labels = {}
    for index, split in enumerate(o.SPLITS):
        parent[split] = [row(100 + index * 10, 0, split), row(100 + index * 10, 1, split),
                         row(101 + index * 10, 0, split)]
        labels.update({o.previous.identity(r): label(r) for r in parent[split]})
    bundles = {'parent': parent}
    for name in o.MODEL_IDS[1:]:
        rep, passes = int(name[1]), int(name[-1])
        groups = deepcopy(parent)
        for split, records in groups.items():
            for item in records:
                item['output83'] = [f32(.15 * passes + .01 * rep), f32(-.05 * passes), f32(-.1 * passes)] + [0.] * 78 + [f32(.07 * passes), f32(-.07 * passes)]
        bundles[name] = groups
    return bundles, labels


class MetricCounterexamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundles, cls.labels = fixture()
        cls.analysis = o.analyze_predictions(cls.bundles, cls.labels, draws=31, seed=731, strict=False)

    def test_entropy_kl_and_equal_policy_limit(self):
        r = row(1, 0, 'train')
        expected = -.75 * math.log(.75) - .25 * math.log(.25)
        got = o.row_metrics(r, r, label(r))
        self.assertAlmostEqual(got['teacher_entropy'], expected)
        self.assertAlmostEqual(got['policy_ce'], math.log(3.))
        self.assertAlmostEqual(got['policy_kl'], math.log(3.) - expected)
        self.assertEqual(got['policy_choice_changed'], 0.)
        self.assertEqual(got['value_drift'], 0.)

    def test_illegal_logits_are_excluded(self):
        r = row(1, 0, 'train')
        original = o.row_metrics(r, r, label(r))
        r['output83'][4] = 1000.
        self.assertEqual(original, o.row_metrics(r, r, label(r)))

    def test_both_heads_and_actual_float32_mixed_target(self):
        r = row(1, 0, 'train', value=.25)
        m = o.row_metrics(r, r, label(r))
        self.assertAlmostEqual(m['outcome_mse'], .75 ** 2)
        self.assertAlmostEqual(m['search_value_mse'], .25 ** 2)
        mixed = f32((1 + .837 * .5) / 1.837)
        self.assertEqual(m['mixed_value_mse'], (.25 - mixed) ** 2)

    def test_ties_choose_lowest_legal_channel(self):
        r = row(1, 0, 'train')
        self.assertEqual(o.row_metrics(r, r, label(r))['policy_teacher_agreement'], 1.)

    def test_wrong_actor_search_sign_rejected(self):
        r = row(1, 0, 'train')
        wrong = label(r)
        wrong['root_q_signed'] = -.5
        with self.assertRaises(ValueError):
            o.row_metrics(r, r, wrong)

    def test_wrong_source_identity_rejected(self):
        r = row(1, 0, 'train')
        wrong = label(r)
        wrong['root_input_sha256'] = 'c' * 64
        with self.assertRaises(ValueError):
            o.row_metrics(r, r, wrong)

    def test_malformed_policy_and_nonfinite_rejected(self):
        r = row(1, 0, 'train')
        r['policy_target81'][0] = .8
        with self.assertRaises(ValueError):
            o.row_metrics(r, r, label(r))
        r = row(1, 0, 'train')
        r['output83'][0] = float('nan')
        with self.assertRaises(ValueError):
            o.row_metrics(r, r, label(r))

    def test_ratio_guard_and_mean_ratio(self):
        self.assertEqual(o.gap_closure(2., 1.5, 1.), .5)
        self.assertEqual(o.gap_closure(2., 2.5, 1.), -.5)
        self.assertIsNone(o.gap_closure(1., 1., 1.))
        self.assertIsNone(o.gap_closure(1. + 1e-10, 1., 1.))
        with self.assertRaises(ValueError):
            o.gap_closure(.5, .4, 1.)
        # The two game gaps are 1 and 3, closures 1 and 0: ratio of means is .25, not .5.
        self.assertEqual(o.gap_closure(3., 2.5, 1.), .25)

    def test_equal_game_weight_not_equal_root_weight(self):
        parent = [row(1, 0, 'train'), row(1, 1, 'train'), row(2, 0, 'train')]
        candidate = deepcopy(parent)
        candidate[2]['output83'][81:] = [1., -1.]
        labels = {o.previous.identity(r): label(r) for r in parent}
        game = o.game_metrics(candidate, parent, labels)
        self.assertEqual([g['roots'] for g in game], [2, 1])
        self.assertEqual(sum(g['outcome_mse'] for g in game) / 2., .5)
        self.assertNotEqual(sum(g['outcome_mse'] for g in game) / 2., 2. / 3.)

    def test_two_seed_average_and_all_contrasts(self):
        a = self.analysis
        for split in (*o.SPLITS, 'pooled'):
            expected = sum(a['estimates'][f'r{r}_pass2'][split]['policy_ce']['mean'] for r in (0, 1)) / 2
            self.assertAlmostEqual(a['estimates']['mean_pass2'][split]['policy_ce']['mean'], expected)
        self.assertEqual(len(a['contrasts']), 6 * 3 * 4 * len(o.METRICS))
        self.assertFalse(a['checkpoint_selection'])
        self.assertFalse(a['strength_improvement_established'])

    def test_independent_arithmetic_checks_all_saved_metrics(self):
        proof = o.verify_arithmetic(self.bundles, self.labels, self.analysis, strict=False)
        self.assertEqual(proof['status'], 'independent_duration_arithmetic_verified')
        self.assertEqual(proof['model_calls'], 0)

    def test_changed_metric_interval_and_ratio_fail(self):
        for field in ('mean', 'ci95'):
            changed = deepcopy(self.analysis)
            entry = changed['estimates']['mean_pass4']['pooled']['policy_ce']
            if field == 'mean':
                entry[field] += .01
            else:
                entry[field][0] += .01
            with self.assertRaises(ValueError):
                o.verify_arithmetic(self.bundles, self.labels, changed, strict=False)
        changed = deepcopy(self.analysis)
        changed['estimates']['mean_pass4']['pooled']['teacher_gap_closed']['mean'] += .01
        with self.assertRaises(ValueError):
            o.verify_arithmetic(self.bundles, self.labels, changed, strict=False)

    def test_missing_contrast_and_wrong_draw_stream_fail(self):
        changed = deepcopy(self.analysis)
        changed['contrasts'].pop()
        with self.assertRaises(ValueError):
            o.verify_arithmetic(self.bundles, self.labels, changed, strict=False)
        changed = deepcopy(self.analysis)
        changed['bootstrap']['strata']['validation']['index_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            o.verify_arithmetic(self.bundles, self.labels, changed, strict=False)

    def test_missing_or_duplicate_rows_rejected(self):
        changed = deepcopy(self.bundles)
        changed['r0_pass4']['offline'].pop()
        with self.assertRaises(ValueError):
            o.analyze_predictions(changed, self.labels, draws=3, seed=731, strict=False)
        changed = deepcopy(self.bundles)
        changed['r0_pass4']['offline'][1] = deepcopy(changed['r0_pass4']['offline'][0])
        with self.assertRaises(ValueError):
            o.analyze_predictions(changed, self.labels, draws=3, seed=731, strict=False)

    def test_small_gap_never_returns_nonfinite_or_selected_draws(self):
        bundles, labels = fixture()
        for groups in bundles.values():
            for rows in groups.values():
                for r in rows:
                    r['policy_target81'] = [.5, .5] + [0.] * 79
                    r['legal_mask81'] = [True, True] + [False] * 79
                    r['output83'][:81] = [0.] * 81
        a = o.analyze_predictions(bundles, labels, draws=7, seed=731, strict=False)
        for name in a['estimates']:
            entry = a['estimates'][name]['pooled']['teacher_gap_closed']
            self.assertIsNone(entry['mean'])
            self.assertIsNone(entry['ci95'])
            self.assertEqual(entry['valid_draws'], 0)
        o.verify_arithmetic(bundles, labels, a, strict=False)

    def test_model_free_import_and_analysis(self):
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
