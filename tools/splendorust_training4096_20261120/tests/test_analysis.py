"""Synthetic-only arithmetic checks; no source targets or model calls."""
from copy import deepcopy
import hashlib
import math
import struct

import pytest

from tools.splendorust_training4096_20261120 import analysis as a


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def fixture_bundles():
    descriptions = {
        'train': [('prior_20261107', 0, 3), ('prior_20261107', 1, 1), ('fresh_20261115', 0, 2)],
        'validation': [('fresh_20261115', 3072, 1), ('fresh_20261115', 3073, 3)],
        'offline': [('fresh_20261115', 3200, 2), ('fresh_20261115', 3201, 1)],
    }
    base = {}
    for split, games in descriptions.items():
        rows = []
        for source, game, roots in games:
            for ply in range(roots):
                tag = f'{source}/{game}/{ply}/{split}'
                rows.append(dict(source=source, game_index=game, ply=ply, split=split,
                    row_sha256=hashlib.sha256(tag.encode()).hexdigest(),
                    root_input_sha256=hashlib.sha256(('input/' + tag).encode()).hexdigest(),
                    legal_mask81=[True, True] + [False] * 79,
                    policy_target81=[.75, .25] + [0.] * 79,
                    value_target2=[1., -1.] if game % 2 else [-1., 1.],
                    root_q_win_credit=.25 if game % 2 else .75,
                    output83=[0.] * 83))
        base[split] = rows
    gains = {'parent': 0., 'historical_r0': .08, 'historical_r1': .12,
             'larger_r0': .28, 'larger_r1': .34, 'control_r0': .02, 'control_r1': .04}
    result = {}
    for model, gain in gains.items():
        result[model] = deepcopy(base)
        for rows in result[model].values():
            for row in rows:
                offset = (row['game_index'] % 3 - 1) * .03 + row['ply'] * .01
                row['output83'] = [f32(.1 + gain + offset), f32(-.1 - gain)] + [f32(12.)] * 79 + [f32(gain), f32(-gain)]
    return result


def analyzed(bundles=None):
    bundles = fixture_bundles() if bundles is None else bundles
    return bundles, a.analyze_predictions(bundles, strict=False, draws=127, seed=45)


def test_saved_arithmetic_independent_and_all_fixed_comparisons(monkeypatch):
    bundles, report = analyzed()
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent verification called the analysis implementation')
    monkeypatch.setattr(a, 'analyze_predictions', forbidden)
    monkeypatch.setattr(a, '_row_metrics', forbidden)
    monkeypatch.setattr(a, '_analysis_bootstrap', forbidden)
    proof = a.verify_arithmetic(bundles, report, strict=False)
    assert proof['independent_count_weight_bootstrap']
    assert proof['independent_loss_formula']
    assert proof['fixed_models'] == 7
    assert len(report['contrasts']) == 3 * 3 * 4 * len(a.METRICS)
    assert report['primary']['conclusion'] == 'clearly_improves'
    assert report['primary']['seed_differences'].keys() == {'r0', 'r1'}
    assert report['counts']['pooled'] == {'games': 4, 'roots': 7}
    family = [r for r in report['contrasts'] if r['secondary_policy_family']]
    assert {r['contrast'] for r in family} == {'larger_minus_parent', 'larger_minus_historical'}
    assert len(family) == 2
    assert all(r['ci97_5'][0] <= r['ci95'][0] <= r['ci95'][1] <= r['ci97_5'][1] for r in family)


def test_unequal_root_counts_receive_equal_game_weights():
    bundles = fixture_bundles()
    for model in a.MODEL_IDS:
        for row in bundles[model]['validation']:
            row['policy_target81'] = [1., 0.] + [0.] * 79
            row['output83'] = [2. if row['game_index'] == 3072 else -2., 0.] + [100.] * 79 + [0., 0.]
    bundles, report = analyzed(bundles)
    good, bad = math.log1p(math.exp(-2)), math.log1p(math.exp(2))
    actual = report['estimates']['parent']['validation']['policy_ce']['mean']
    assert actual == pytest.approx((good + bad) / 2, abs=1e-13)
    assert abs(actual - (good + 3 * bad) / 4) > .4
    a.verify_arithmetic(bundles, report, strict=False)


def test_seed_mean_is_mean_loss_not_prediction_ensemble():
    bundles = fixture_bundles()
    for model in a.MODEL_IDS:
        for rows in bundles[model].values():
            for row in rows:
                row['policy_target81'] = [.5, .5] + [0.] * 79
                logits = [4., -4.] if model == 'larger_r0' else [-4., 4.] if model == 'larger_r1' else [0., 0.]
                row['output83'] = logits + [100.] * 79 + [0., 0.]
    bundles, report = analyzed(bundles)
    result = report['estimates']['larger_mean']['pooled']['policy_ce']['mean']
    assert result == pytest.approx(4 + math.log1p(math.exp(-8)))
    assert result > math.log(2) + 3
    assert result == pytest.approx(sum(report['estimates'][m]['pooled']['policy_ce']['mean']
                                       for m in ('larger_r0', 'larger_r1')) / 2)
    a.verify_arithmetic(bundles, report, strict=False)


def test_policy_mask_excludes_larger_illegal_logits():
    bundles = fixture_bundles()
    for model in a.MODEL_IDS:
        for rows in bundles[model].values():
            for row in rows:
                row['output83'] = [0., 0.] + [1000.] * 79 + [0., 0.]
    bundles, report = analyzed(bundles)
    assert report['estimates']['parent']['pooled']['policy_ce']['mean'] == pytest.approx(math.log(2))
    a.verify_arithmetic(bundles, report, strict=False)


def test_near_zero_teacher_gap_is_undefined():
    bundles = fixture_bundles()
    for model in a.MODEL_IDS:
        for rows in bundles[model].values():
            for row in rows:
                row['policy_target81'] = [.5, .5] + [0.] * 79
                row['output83'] = [0.] * 83
    bundles, report = analyzed(bundles)
    for model in report['estimates']:
        gap = report['estimates'][model]['pooled']['teacher_gap_closed']
        assert gap['mean'] is None and gap['ci95'] is None and gap['valid_draws'] == 0
    assert report['primary']['difference'] == 0.
    assert report['primary']['ci95'] == [0., 0.]
    assert report['primary']['conclusion'] == 'unresolved'
    a.verify_arithmetic(bundles, report, strict=False)


def test_namespace_preserves_overlapping_old_and_new_game_numbers():
    bundles, report = analyzed()
    games = {(r['source'], r['game_index']) for r in report['game_summaries']['parent']['train']}
    assert ('prior_20261107', 0) in games and ('fresh_20261115', 0) in games
    assert report['counts']['train']['games'] == 3
    assert report['training_membership']['prior_20261107']['games'] == 2
    assert report['training_membership']['fresh_20261115']['games'] == 1
    assert not report['training_membership']['fresh_20261115']['independent_assessment']


@pytest.mark.parametrize('mutation', [
    lambda r: r['estimates']['larger_mean']['pooled']['policy_ce'].__setitem__('mean', 8.),
    lambda r: r['estimates']['larger_mean']['pooled']['policy_ce'].__setitem__('ci95', [8., 9.]),
    lambda r: r['game_summaries']['parent']['train'][0].__setitem__('policy_ce', 8.),
    lambda r: r['primary'].__setitem__('difference', 8.),
    lambda r: r['primary'].__setitem__('conclusion', 'promote'),
    lambda r: r['bootstrap']['strata']['validation'].__setitem__('index_sha256', 'f' * 64),
    lambda r: r['training_membership']['fresh_20261115'].__setitem__('independent_assessment', True),
    lambda r: r['counts']['pooled'].__setitem__('games', 8),
    lambda r: next(x for x in r['contrasts'] if x['secondary_policy_family']).__setitem__('ci97_5', [8., 9.]),
    lambda r: r['recipe'].__setitem__('prediction_ensemble', True),
])
def test_saved_arithmetic_tampering_is_rejected(mutation):
    bundles, report = analyzed()
    mutation(report)
    with pytest.raises(ValueError):
        a.verify_arithmetic(bundles, report, strict=False)


@pytest.mark.parametrize('case', ['changed_target', 'illegal_policy', 'wrong_source', 'cross_split',
                                  'missing_model', 'duplicated_row', 'nonfinite_output', 'nonfloat32_output'])
def test_input_failures_rejected(case):
    bundles = fixture_bundles()
    row = bundles['larger_r0']['validation'][0]
    if case == 'changed_target':
        row['policy_target81'] = [.5, .5] + [0.] * 79
    elif case == 'illegal_policy':
        row['policy_target81'] = [.5, .25, .25] + [0.] * 78
    elif case == 'wrong_source':
        for model in a.MODEL_IDS:
            bundles[model]['validation'][0]['source'] = 'prior_20261107'
    elif case == 'cross_split':
        for model in a.MODEL_IDS:
            for item in bundles[model]['offline']:
                if item['game_index'] == 3200:
                    item['game_index'] = 3072
    elif case == 'missing_model':
        del bundles['historical_r1']
    elif case == 'duplicated_row':
        for model in a.MODEL_IDS:
            bundles[model]['validation'].append(deepcopy(bundles[model]['validation'][0]))
    elif case == 'nonfinite_output':
        row['output83'][0] = float('nan')
    elif case == 'nonfloat32_output':
        row['output83'][0] = .1
    with pytest.raises(ValueError):
        a.analyze_predictions(bundles, strict=False, draws=17)


def test_strict_production_rejects_small_synthetic_population_and_changed_recipe():
    bundles = fixture_bundles()
    with pytest.raises(ValueError, match='prospective source-game population'):
        a.analyze_predictions(bundles)
    with pytest.raises(ValueError, match='bootstrap recipe'):
        a.analyze_predictions(bundles, draws=10)


def test_bootstrap_is_stratified_repeatable_and_pairs_all_models():
    bundles = fixture_bundles()
    for model in a.MODEL_IDS:
        bundles[model] = deepcopy(bundles['parent'])
    _, one = analyzed(bundles)
    _, two = analyzed(bundles)
    assert one == two
    assert one['bootstrap']['stratified_by'] == ['validation', 'offline']
    for contrast in one['contrasts']:
        assert contrast['difference'] == 0.
        assert contrast['ci95'] is None if contrast['split'] == 'train' else contrast['ci95'] == [0., 0.]
    a.verify_arithmetic(bundles, one, strict=False)


def test_chart_outputs_are_bounded_exclusive_and_two_only(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(a.c, 'OUT', tmp_path)
    _, report = analyzed()
    items = a.render_charts(report, tmp_path / 'charts')
    assert len(items) == 2
    for item in items:
        with Image.open(a.c.resolve(item)) as picture:
            assert picture.size == (1400, 650)
    with pytest.raises(ValueError, match='Exclusive new policy chart'):
        a.render_charts(report, tmp_path / 'charts')
    with pytest.raises(ValueError, match='new study output'):
        a.render_charts(report, tmp_path.parent / 'outside')


def test_wrappers_bind_source_saved_arithmetic_and_both_charts(tmp_path, monkeypatch):
    from tools.splendorust_training4096_20261120 import offline
    monkeypatch.setattr(a.c, 'OUT', tmp_path)
    bundles = fixture_bundles()
    document = {'synthetic_fixture': True}
    a.c.save(tmp_path / 'jobs/offline/worker/evaluation_result.json', document)
    monkeypatch.setattr(offline, 'load_predictions', lambda **kwargs: (bundles, deepcopy(document)))
    analyze, verify = a.analyze_predictions, a.verify_arithmetic
    monkeypatch.setattr(a, 'analyze_predictions', lambda b, **kwargs: analyze(b, strict=False, draws=31, seed=42, **kwargs))
    monkeypatch.setattr(a, 'verify_arithmetic', lambda b, r, **kwargs: verify(b, r, strict=False, **kwargs))
    folder = tmp_path / 'jobs/analysis/worker'
    result = a.run_analysis(folder)
    proof = a.verify_analysis(folder)
    assert proof['analysis'] == result['analysis'] and len(proof['charts']) == 2
    a.c.resolve(result['charts'][0]).write_bytes(b'changed synthetic image')
    with pytest.raises(ValueError, match='fixed charts'):
        a.verify_analysis(folder)
