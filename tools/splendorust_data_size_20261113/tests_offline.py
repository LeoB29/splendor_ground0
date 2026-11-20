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
        size_text, rep_text, endpoint = name.split('_')
        rep = int(rep_text[1:]); size = int(size_text[1:])
        passes = size / 256 + (1. if endpoint == 'matched' else 0.)
        groups = deepcopy(parent)
        for split, records in groups.items():
            for item in records:
                item['output83'] = [f32(.15 * passes + .01 * rep), f32(-.05 * passes), f32(-.1 * passes)] + [0.] * 78 + [f32(.07 * passes), f32(-.07 * passes)]
        bundles[name] = groups
    return bundles, labels


def selection():
    return dict(subsets={'256':dict(game_ids=[100]),'512':dict(game_ids=[100,101])})


class MetricCounterexamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundles, cls.labels = fixture()
        cls.analysis = o.analyze_predictions(cls.bundles, cls.labels, selection(), draws=31, seed=731, strict=False)

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
            expected = sum(a['estimates'][f'g256_r{r}_matched'][split]['policy_ce']['mean'] for r in (0, 1)) / 2
            self.assertAlmostEqual(a['estimates']['g256_mean_matched'][split]['policy_ce']['mean'], expected)
        self.assertEqual(len(a['contrasts']), 13 * 3 * 4 * len(o.METRICS))
        self.assertFalse(a['checkpoint_selection'])
        self.assertFalse(a['strength_improvement_established'])

    def test_independent_arithmetic_checks_all_saved_metrics(self):
        proof = o.verify_arithmetic(self.bundles, self.labels, selection(), self.analysis, strict=False)
        self.assertEqual(proof['status'], 'independent_data_size_arithmetic_verified')
        self.assertEqual(proof['model_calls'], 0)

    def test_changed_metric_interval_and_ratio_fail(self):
        for field in ('mean', 'ci95'):
            changed = deepcopy(self.analysis)
            entry = changed['estimates']['g512_mean_matched']['pooled']['policy_ce']
            if field == 'mean':
                entry[field] += .01
            else:
                entry[field][0] += .01
            with self.assertRaises(ValueError):
                o.verify_arithmetic(self.bundles, self.labels, selection(), changed, strict=False)
        changed = deepcopy(self.analysis)
        changed['estimates']['g512_mean_matched']['pooled']['teacher_gap_closed']['mean'] += .01
        with self.assertRaises(ValueError):
            o.verify_arithmetic(self.bundles, self.labels, selection(), changed, strict=False)

    def test_missing_contrast_and_wrong_draw_stream_fail(self):
        changed = deepcopy(self.analysis)
        changed['contrasts'].pop()
        with self.assertRaises(ValueError):
            o.verify_arithmetic(self.bundles, self.labels, selection(), changed, strict=False)
        changed = deepcopy(self.analysis)
        changed['bootstrap']['strata']['validation']['index_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            o.verify_arithmetic(self.bundles, self.labels, selection(), changed, strict=False)

    def test_missing_or_duplicate_rows_rejected(self):
        changed = deepcopy(self.bundles)
        changed['g512_r0_matched']['offline'].pop()
        with self.assertRaises(ValueError):
            o.analyze_predictions(changed, self.labels, selection(), draws=3, seed=731, strict=False)
        changed = deepcopy(self.bundles)
        changed['g512_r0_matched']['offline'][1] = deepcopy(changed['g512_r0_matched']['offline'][0])
        with self.assertRaises(ValueError):
            o.analyze_predictions(changed, self.labels, selection(), draws=3, seed=731, strict=False)

    def test_small_gap_never_returns_nonfinite_or_selected_draws(self):
        bundles, labels = fixture()
        for groups in bundles.values():
            for rows in groups.values():
                for r in rows:
                    r['policy_target81'] = [.5, .5] + [0.] * 79
                    r['legal_mask81'] = [True, True] + [False] * 79
                    r['output83'][:81] = [0.] * 81
        a = o.analyze_predictions(bundles, labels, selection(), draws=7, seed=731, strict=False)
        for name in a['estimates']:
            entry = a['estimates'][name]['pooled']['teacher_gap_closed']
            self.assertIsNone(entry['mean'])
            self.assertIsNone(entry['ci95'])
            self.assertEqual(entry['valid_draws'], 0)
        o.verify_arithmetic(bundles, labels, selection(), a, strict=False)

    def test_model_free_import_and_analysis(self):
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') for name in sys.modules))

    def test_onepass_and_matched_controls_have_correct_endpoints(self):
        mapping={(label,rep):(left,right) for label,rep,left,right in o._contrast_definitions()}
        self.assertEqual(mapping['onepass_256_to_512','mean'],('g256_mean_onepass','g512_mean_onepass'))
        self.assertEqual(mapping['matched_512_to_1024','r1'],('g512_r1_matched','g1024_r1_onepass'))
        self.assertEqual(mapping['extra_training_g256','r0'],('g256_r0_onepass','g256_r0_matched'))
        self.assertEqual(len(o.FRESH_IDS),8)
        self.assertEqual(len(o.MODEL_IDS),11)
        self.assertNotIn('g1024_r0_matched',o.MODEL_IDS)

    def test_membership_masks_compare_all_models_on_same_games(self):
        views=self.analysis['training_membership']
        selected=views['256']['selected'];outside=views['256']['outside']
        self.assertEqual(selected['game_ids'],[100]);self.assertEqual(outside['game_ids'],[101])
        self.assertEqual(set(selected['models']),set(o._definitions()))
        self.assertTrue(all(v['roots']==2 for v in selected['models'].values()))
        self.assertTrue(all(v['roots']==1 for v in outside['models'].values()))
        empty=views['512']['outside']
        self.assertEqual(empty['games'],0)
        self.assertTrue(all(v is None for row in empty['models'].values() for v in row['metrics'].values()))

    def test_membership_corruption_or_different_subset_is_rejected(self):
        altered=deepcopy(self.analysis)
        altered['training_membership']['256']['selected']['models']['g256_mean_matched']['metrics']['policy_ce']+=.1
        with self.assertRaises(ValueError):o.verify_arithmetic(self.bundles,self.labels,selection(),altered,strict=False)
        different=selection();different['subsets']['256']['game_ids']=[101]
        with self.assertRaises(ValueError):o.verify_arithmetic(self.bundles,self.labels,different,self.analysis,strict=False)

    def test_changed_training_interpretation_is_rejected(self):
        altered=deepcopy(self.analysis)
        altered['comparison_design']['onepass_is_not_matched_compute']=False
        with self.assertRaises(ValueError):o.verify_arithmetic(self.bundles,self.labels,selection(),altered,strict=False)


def test_committed_fit_endpoints_follow_actual_result_and_exit_files(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(o.c, 'OUT', tmp_path)
    committed = {'fits': []}; expected = {}
    for size in (256, 512):
        for rep in (0, 1):
            role = f'g{size}_r{rep}'; folder = tmp_path/'jobs'/role
            steps = {'onepass': 445 if size == 256 else 889, 'matched': 1782}
            endpoints = {endpoint: o.c.save(folder/'worker'/f'{endpoint}.json', {'fixture': endpoint})
                         for endpoint in steps}
            checkpoints = {str(steps[endpoint]): item for endpoint, item in endpoints.items()}
            result = o.c.save(folder/'worker/fit_result.json', dict(status='fit_complete', game_count=size,
                repetition=rep, endpoints=endpoints, checkpoints=checkpoints, endpoint_steps=steps))
            proof = o.c.save(tmp_path/'verified'/f'{role}.json', dict(status='data_size_fit_saved_verified',
                game_count=size, repetition=rep, optimizer_calls=1782, replay_optimizer_calls=2,
                result=result, endpoints=endpoints, checkpoints=checkpoints, endpoint_steps=steps))
            execution = o.c.save(folder/'actual_exit.json', dict(role=role, status='passed', exit_code=0,
                actual_exit_observed=True, exact_owned_absence=True, cleanup_errors=[], error=None))
            committed['fits'].append(dict(game_count=size, repetition=rep, verification=proof, execution=execution))
            expected.update({f'{role}_{endpoint}': item for endpoint, item in endpoints.items()})
    assert o._committed_fit_endpoints(committed) == expected
    changed = deepcopy(committed)
    proof = o.c.read(o.c.resolve(changed['fits'][0]['verification']))
    original_result = o.c.read(o.c.resolve(proof['result']))
    proof['result'] = o.c.save(tmp_path/'foreign/fit_result.json', original_result)
    changed['fits'][0]['verification'] = o.c.save(tmp_path/'foreign/verification.json', proof)
    with pytest.raises(ValueError, match='actual fit result'):
        o._committed_fit_endpoints(changed)
    changed = deepcopy(committed)
    changed['fits'][0]['execution'] = changed['fits'][1]['execution']
    with pytest.raises(ValueError, match='clean actual exit'):
        o._committed_fit_endpoints(changed)
    assert 'torch' not in sys.modules


if __name__ == '__main__':
    unittest.main()
