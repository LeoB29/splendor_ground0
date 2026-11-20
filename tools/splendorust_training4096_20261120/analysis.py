"""Saved, paired whole-game analysis for the fixed 4,096-game comparison.

This module never creates a model or reads a target corpus. The two arithmetic
routes deliberately use different row formulas, reductions and bootstrap
implementations. Production wrappers authenticate saved prediction receipts.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import sys

from . import common as c

MODEL_IDS = ('parent', 'historical_r0', 'historical_r1', 'larger_r0',
             'larger_r1', 'control_r0', 'control_r1')
SPLITS = ('train', 'validation', 'offline')
HELDOUT = ('validation', 'offline')
VIEWS = (*SPLITS, 'pooled')
SOURCES = ('prior_20261107', 'fresh_20261115')
METRICS = ('policy_ce', 'teacher_entropy', 'policy_kl',
           'policy_teacher_agreement', 'outcome_mse', 'search_value_mse',
           'mixed_value_mse', 'value_drift')
DRAW_COUNT, BOOTSTRAP_SEED, GAP_FLOOR = 20000, 202611201, 1e-8
SCHEMA = 'training4096_saved_prediction_analysis_v1'
require = c.require


def _definitions():
    definitions = {m: {m: 1.} for m in MODEL_IDS}
    definitions.update({arm + '_mean': {arm + '_r0': .5, arm + '_r1': .5}
                        for arm in ('historical', 'larger', 'control')})
    return definitions


def _contrasts():
    result = []
    for other in ('control', 'parent', 'historical'):
        for seed in ('r0', 'r1', 'mean'):
            result.append(('larger_minus_' + other, seed, 'larger_' + seed,
                           'parent' if other == 'parent' else other + '_' + seed))
    return result


def _recipe():
    return dict(training_games=4096, control_training_games=1024,
                presentations_per_new_fit=227203, updates_per_new_fit=7101,
                optimization_seeds=[202611071, 202611072], seed_loss_mean=True,
                prediction_ensemble=False, checkpoint_selection=False,
                game_weighting='mean_roots_within_game_then_equal_games',
                primary='larger_minus_control/mean/pooled/policy_ce',
                secondary_policy_family=['larger_minus_parent', 'larger_minus_historical'],
                secondary_family_interval=97.5, other_metrics_descriptive=True,
                training_panel_descriptive=True, strength_claim=False,
                model_calls=0, optimizer_calls=0)


def _identity(row):
    return row['source'], row['game_index'], row['ply'], row['split']


def _game(row):
    return row['source'], row['game_index']


def _validate(bundles, strict, boundary):
    # Imported only when data are supplied; this module remains model-free.
    from . import offline
    require(type(strict) is bool and type(bundles) is dict and set(bundles) == set(MODEL_IDS),
            'All seven fixed prediction models')
    games, identities, counts = {}, {}, {}
    for split in SPLITS:
        parent = bundles['parent'].get(split)
        require(type(parent) is list and parent, 'Nonempty fixed prediction split')
        identities[split] = [_identity(r) for r in parent]
        require(len(set(identities[split])) == len(parent), 'No duplicated namespaced root')
        grouped = defaultdict(int)
        for row in parent:
            require(row['split'] == split and row['source'] in SOURCES, 'Correct source and split')
            if split != 'train':
                require(row['source'] == SOURCES[1], 'Only new games in fresh assessment')
            grouped[_game(row)] += 1
        games[split] = sorted(grouped)
        counts[split] = dict(games=len(grouped), roots=len(parent))
        for model in MODEL_IDS:
            require(type(bundles[model]) is dict and set(bundles[model]) == set(SPLITS),
                    'Exact train/validation/offline prediction inventory')
            rows = bundles[model][split]
            require(type(rows) is list and len(rows) == len(parent), 'Identical row count for every model')
            for number, (row, old) in enumerate(zip(rows, parent, strict=True)):
                if number % 256 == 0:
                    boundary()
                offline.strict_prediction(row)
                require(_identity(row) == _identity(old) and
                        offline.target_identity(row) == offline.target_identity(old),
                        'Every model uses the identical ordered inputs and targets')
        if strict:
            expected = ([(SOURCES[0], i) for i in range(1024)] +
                        [(SOURCES[1], i) for i in range(3072)]) if split == 'train' else [
                            (SOURCES[1], i) for i in
                            (range(3072, 3200) if split == 'validation' else range(3200, 3328))]
            require(set(games[split]) == set(expected), 'Exact prospective source-game population')
            if split == 'train':
                require(len(parent) == 16384 and set(grouped.values()) == {4},
                        'Fixed four-root panel for all 4096 training games')
    require(not (set(games['train']) & set(games['validation']) or
                 set(games['train']) & set(games['offline']) or
                 set(games['validation']) & set(games['offline'])),
            'Disjoint namespaced training and assessment games')
    require({g[0] for g in games['train']} == set(SOURCES), 'Both old and new training panel sources')
    return games, counts


def _row_metrics(row, parent):
    """Analysis route: masked shifted log-sum-exp and vector reductions."""
    import numpy as np
    output = np.asarray(row['output83'], dtype=np.float64)
    old = np.asarray(parent['output83'], dtype=np.float64)
    mask = np.asarray(row['legal_mask81'], dtype=bool)
    indices = np.flatnonzero(mask)
    logits = output[:81][mask]
    target = np.asarray(row['policy_target81'], dtype=np.float64)[mask]
    peak = logits.max()
    log_probabilities = logits - peak - np.log(np.exp(logits - peak).sum())
    ce = float(-(target * log_probabilities).sum())
    nonzero = target > 0
    entropy = float(-(target[nonzero] * np.log(target[nonzero])).sum())
    z = np.asarray(row['value_target2'], dtype=np.float64)
    q = 2. * row['root_q_win_credit'] - 1.
    t = np.float32((z[0] + .837 * q) / 1.837)
    mixed = np.asarray([t, -t], dtype=np.float64)
    values = [ce, entropy, max(0., ce - entropy),
              float(indices[np.argmax(logits)] == indices[np.argmax(target)]),
              float(np.square(output[81:] - z).mean()),
              float(np.square(output[81:] - [q, -q]).mean()),
              float(np.square(output[81:] - mixed).mean()),
              float(np.abs(output[81:] - old[81:]).mean())]
    require(all(math.isfinite(v) for v in values), 'Finite computed row metrics')
    return values


def _analysis_matrices(bundles, games, boundary):
    import numpy as np
    matrices, summaries = {}, {m: {} for m in MODEL_IDS}
    for split in SPLITS:
        models = []
        for model in MODEL_IDS:
            grouped = defaultdict(list)
            for number, (row, parent) in enumerate(zip(bundles[model][split], bundles['parent'][split], strict=True)):
                if number % 256 == 0:
                    boundary()
                grouped[_game(row)].append(_row_metrics(row, parent))
            matrix = np.asarray([np.mean(grouped[g], axis=0) for g in games[split]])
            models.append(matrix)
            summaries[model][split] = [dict(source=g[0], game_index=g[1], roots=len(grouped[g]),
                                            **dict(zip(METRICS, map(float, vector), strict=True)))
                                        for g, vector in zip(games[split], matrix, strict=True)]
        matrices[split] = np.stack(models)
    matrices['pooled'] = np.concatenate([matrices[s] for s in HELDOUT], axis=1)
    return matrices, summaries


def _percentile(values, levels=(2.5, 97.5)):
    import numpy as np
    return np.percentile(values, levels, method='linear').tolist()


def _analysis_bootstrap(matrices, games, draws, seed, boundary):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled, receipts = {}, {}
    for split in HELDOUT:
        n = len(games[split])
        indices = rng.integers(0, n, size=(draws, n), dtype=np.int64)
        receipts[split] = dict(games=[list(g) for g in games[split]], game_count=n,
                               index_sha256=hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest())
        values = np.empty((len(MODEL_IDS), draws, len(METRICS)), dtype=np.float64)
        for start in range(0, draws, 128):
            boundary()
            values[:, start:start + 128] = matrices[split][:, indices[start:start + 128], :].mean(axis=2)
        sampled[split] = values
    total = sum(len(games[s]) for s in HELDOUT)
    sampled['pooled'] = sum(sampled[s] * (len(games[s]) / total) for s in HELDOUT)
    return sampled, receipts


def _gap(parent, point, parent_draws, draws, n_draws):
    import numpy as np
    denom = float(parent[0] - parent[1])
    mean = None if denom <= GAP_FLOOR else float((parent[0] - point[0]) / denom)
    interval, valid = None, None
    if draws is not None:
        denominators = parent_draws[:, 0] - parent_draws[:, 1]
        valid = int(np.count_nonzero(denominators > GAP_FLOOR))
        if denom > GAP_FLOOR and valid == n_draws:
            interval = _percentile((parent_draws[:, 0] - draws[:, 0]) / denominators)
    return dict(mean=mean, ci95=interval, baseline_gap=denom, denominator_floor=GAP_FLOOR,
                valid_draws=valid, ratio_of_equal_game_means=True)


def analyze_predictions(bundles, *, strict=True, draws=DRAW_COUNT,
                        seed=BOOTSTRAP_SEED, boundary=lambda: None):
    """Pure arithmetic; strict=False is solely for small synthetic fixtures."""
    import numpy as np
    require(type(draws) is int and draws > 0 and type(seed) is int, 'Valid bootstrap settings')
    if strict:
        require((draws, seed) == (DRAW_COUNT, BOOTSTRAP_SEED), 'Frozen production bootstrap recipe')
    games, counts = _validate(bundles, strict, boundary)
    matrices, summaries = _analysis_matrices(bundles, games, boundary)
    sampled, receipts = _analysis_bootstrap(matrices, games, draws, seed, boundary)
    definitions = _definitions()
    estimates, distributions = {}, {}
    for name, coefficients in definitions.items():
        estimates[name], distributions[name] = {}, {}
        for split in VIEWS:
            point = sum(weight * matrices[split][MODEL_IDS.index(model)].mean(axis=0)
                        for model, weight in coefficients.items())
            distribution = None if split == 'train' else sum(
                weight * sampled[split][MODEL_IDS.index(model)] for model, weight in coefficients.items())
            distributions[name][split] = distribution
            estimates[name][split] = {metric: dict(mean=float(point[i]), ci95=None if distribution is None else
                                                   _percentile(distribution[:, i]))
                                     for i, metric in enumerate(METRICS)}
            estimates[name][split]['teacher_gap_closed'] = _gap(
                matrices[split][0].mean(axis=0), point,
                None if split == 'train' else sampled[split][0], distribution, draws)
        boundary()
    contrasts = []
    for label, rep, left, right in _contrasts():
        for split in VIEWS:
            for i, metric in enumerate(METRICS):
                dist = None if split == 'train' else distributions[left][split][:, i] - distributions[right][split][:, i]
                family = rep == 'mean' and split == 'pooled' and metric == 'policy_ce' and label != 'larger_minus_control'
                contrasts.append(dict(contrast=label, seed=rep, split=split, metric=metric,
                    difference=estimates[left][split][metric]['mean'] - estimates[right][split][metric]['mean'],
                    ci95=None if dist is None else _percentile(dist),
                    ci97_5=_percentile(dist, (1.25, 98.75)) if family else None,
                    secondary_policy_family=family))
    lookup = {(r['contrast'], r['seed'], r['split'], r['metric']): r for r in contrasts}
    primary = dict(lookup['larger_minus_control', 'mean', 'pooled', 'policy_ce'])
    primary['conclusion'] = ('clearly_improves' if primary['ci95'][1] < 0 else
                             'clearly_worse' if primary['ci95'][0] > 0 else 'unresolved')
    primary['seed_differences'] = {rep: lookup['larger_minus_control', rep, 'pooled', 'policy_ce']['difference']
                                   for rep in ('r0', 'r1')}
    membership = {}
    for source in SOURCES:
        positions = [i for i, game in enumerate(games['train']) if game[0] == source]
        roots = sum(summaries['parent']['train'][i]['roots'] for i in positions)
        models = {}
        for name, coefficients in definitions.items():
            point = sum(weight * matrices['train'][MODEL_IDS.index(model), positions, :].mean(axis=0)
                        for model, weight in coefficients.items())
            models[name] = {metric: float(point[i]) for i, metric in enumerate(METRICS)}
        membership[source] = dict(games=len(positions), roots=roots,
            game_ids=[list(games['train'][i]) for i in positions], models=models,
            descriptive_only=True, independent_assessment=False)
    counts['pooled'] = {key: sum(counts[s][key] for s in HELDOUT) for key in ('games', 'roots')}
    return dict(schema=SCHEMA, study_id=c.STUDY, status='saved_predictions_analyzed',
                model_ids=list(MODEL_IDS), metrics=list(METRICS), recipe=_recipe(), counts=counts,
                game_summaries=summaries, estimates=estimates, contrasts=contrasts, primary=primary,
                training_membership=membership,
                bootstrap=dict(draws=draws, seed=seed, generator='numpy.PCG64', unit='whole_game',
                    paired_same_draws_all_models=True, stratified_by=list(HELDOUT), strata=receipts,
                    training_panel='descriptive_no_bootstrap', seed_uncertainty_estimated=False),
                **c.IDENTITY)


def _audit_matrices(bundles, games, reported, boundary):
    """Independent route: scalar fsum/logaddexp and explicit two-output losses."""
    import numpy as np
    matrices = {}
    for split in SPLITS:
        models = []
        for model in MODEL_IDS:
            accumulators = defaultdict(lambda: [[], 0])
            for number, (row, parent) in enumerate(zip(bundles[model][split], bundles['parent'][split], strict=True)):
                if number % 256 == 0:
                    boundary()
                output, old = row['output83'], parent['output83']
                legal = [i for i, valid in enumerate(row['legal_mask81']) if valid]
                normalizer = float(np.logaddexp.reduce([output[i] for i in legal]))
                target = row['policy_target81']
                ce = math.fsum(target[i] * (normalizer - output[i]) for i in legal)
                entropy = -math.fsum(target[i] * math.log(target[i]) for i in legal if target[i] > 0)
                q = row['root_q_win_credit'] * 2 - 1
                z = row['value_target2'][0]
                mixed = float(np.float32((z + q * .837) / 1.837))
                teacher = max(legal, key=lambda i: (target[i], -i))
                choice = max(legal, key=lambda i: (output[i], -i))
                vector = [ce, entropy, max(ce - entropy, 0.), float(teacher == choice),
                          ((output[81] - z)**2 + (output[82] + z)**2) / 2,
                          ((output[81] - q)**2 + (output[82] + q)**2) / 2,
                          ((output[81] - mixed)**2 + (output[82] + mixed)**2) / 2,
                          (abs(output[81] - old[81]) + abs(output[82] - old[82])) / 2]
                accumulators[_game(row)][0].append(vector)
                accumulators[_game(row)][1] += 1
            matrix = np.asarray([[math.fsum(v[k] for v in accumulators[g][0]) / accumulators[g][1]
                                  for k in range(len(METRICS))] for g in games[split]])
            saved = reported['game_summaries'][model][split]
            require(len(saved) == len(games[split]), 'Independent saved game count')
            for game, vector, record in zip(games[split], matrix, saved, strict=True):
                require(set(record) == {'source', 'game_index', 'roots', *METRICS} and
                        (record['source'], record['game_index']) == game and
                        record['roots'] == accumulators[game][1], 'Independent namespaced game reduction')
                require(np.allclose(vector, [record[k] for k in METRICS], rtol=1e-10, atol=1e-12),
                        'Independent row formula and equal-game means')
            models.append(matrix)
        matrices[split] = np.stack(models)
    matrices['pooled'] = np.concatenate((matrices['validation'], matrices['offline']), axis=1)
    return matrices


def _same(actual, expected, message):
    import numpy as np
    if expected is None:
        require(actual is None, message)
    else:
        require(actual is not None and np.asarray(actual).shape == np.asarray(expected).shape and
                np.allclose(actual, expected, rtol=1e-10, atol=1e-12), message)


def verify_arithmetic(bundles, reported, *, strict=True, boundary=lambda: None):
    """Independently recompute row losses and count-weight paired bootstrap."""
    import numpy as np
    games, counts = _validate(bundles, strict, boundary)
    keys = {'schema', 'study_id', 'status', 'model_ids', 'metrics', 'recipe', 'counts',
            'game_summaries', 'estimates', 'contrasts', 'primary', 'training_membership',
            'bootstrap', *c.IDENTITY}
    require(type(reported) is dict and set(reported) == keys and reported['schema'] == SCHEMA and
            reported['study_id'] == c.STUDY and reported['status'] == 'saved_predictions_analyzed' and
            all(reported[k] == v for k, v in c.IDENTITY.items()), 'Exact analysis identity/schema')
    require(reported['model_ids'] == list(MODEL_IDS) and reported['metrics'] == list(METRICS) and
            reported['recipe'] == _recipe(), 'Fixed analysis inventory and interpretation')
    require(set(reported['game_summaries']) == set(MODEL_IDS) and
            all(set(v) == set(SPLITS) for v in reported['game_summaries'].values()), 'Complete game summaries')
    counts['pooled'] = {key: sum(counts[s][key] for s in HELDOUT) for key in ('games', 'roots')}
    require(reported['counts'] == counts, 'Independent root/game populations')
    boot = reported['bootstrap']
    draws, seed = boot['draws'], boot['seed']
    require(type(draws) is int and draws > 0 and type(seed) is int, 'Valid saved bootstrap counts')
    if strict:
        require((draws, seed) == (DRAW_COUNT, BOOTSTRAP_SEED), 'Frozen independent bootstrap settings')
    require(set(boot) == {'draws', 'seed', 'generator', 'unit', 'paired_same_draws_all_models',
                         'stratified_by', 'strata', 'training_panel', 'seed_uncertainty_estimated'} and
            boot['generator'] == 'numpy.PCG64' and boot['unit'] == 'whole_game' and
            boot['paired_same_draws_all_models'] is True and boot['stratified_by'] == list(HELDOUT) and
            boot['training_panel'] == 'descriptive_no_bootstrap' and boot['seed_uncertainty_estimated'] is False and
            set(boot['strata']) == set(HELDOUT), 'Whole-game stratified paired recipe')
    matrices = _audit_matrices(bundles, games, reported, boundary)
    sampled = {}
    rng = np.random.Generator(np.random.PCG64(seed))
    for split in HELDOUT:
        n = len(games[split])
        # Generate each draw separately, then reduce multiplicity weights instead
        # of indexing sampled positions as the primary analysis does.
        ids = np.empty((draws, n), dtype=np.int64)
        for d in range(draws):
            if d % 500 == 0:
                boundary()
            ids[d] = rng.integers(0, n, size=n, dtype=np.int64)
        require(boot['strata'][split] == dict(games=[list(g) for g in games[split]], game_count=n,
                    index_sha256=hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest()),
                'Independent bootstrap game order and random stream')
        sample = np.empty((len(MODEL_IDS), draws, len(METRICS)))
        for start in range(0, draws, 128):
            weights = np.asarray([np.bincount(row, minlength=n) for row in ids[start:start + 128]], dtype=np.float64)
            weights /= n
            sample[:, start:start + 128] = np.einsum('dg,mgk->mdk', weights, matrices[split])
            boundary()
        sampled[split] = sample
    sampled['pooled'] = (sampled['validation'] * len(games['validation']) +
                          sampled['offline'] * len(games['offline'])) / counts['pooled']['games']
    definitions = _definitions()
    require(set(reported['estimates']) == set(definitions), 'All fixed seeds and mean-loss estimates')
    points, distributions = {}, {}
    for name, coefficients in definitions.items():
        points[name], distributions[name] = {}, {}
        require(set(reported['estimates'][name]) == set(VIEWS), 'All fixed estimate views')
        for split in VIEWS:
            point = np.zeros(len(METRICS))
            distribution = None if split == 'train' else np.zeros((draws, len(METRICS)))
            for model, weight in coefficients.items():
                point += weight * np.add.reduce(matrices[split][MODEL_IDS.index(model)], axis=0) / matrices[split].shape[1]
                if distribution is not None:
                    distribution += weight * sampled[split][MODEL_IDS.index(model)]
            points[name][split], distributions[name][split] = point, distribution
            records = reported['estimates'][name][split]
            require(set(records) == {*METRICS, 'teacher_gap_closed'}, 'Complete estimate metric inventory')
            for i, metric in enumerate(METRICS):
                record = records[metric]
                require(set(record) == {'mean', 'ci95'}, 'Exact scalar estimate schema')
                _same(record['mean'], point[i], 'Independent mean-loss estimate')
                _same(record['ci95'], None if distribution is None else np.percentile(distribution[:, i], [2.5, 97.5], method='linear'),
                      'Independent count-weight interval')
            parent = matrices[split][0].sum(axis=0) / matrices[split].shape[1]
            denominator = parent[0] - parent[1]
            gap = records['teacher_gap_closed']
            require(set(gap) == {'mean', 'ci95', 'baseline_gap', 'denominator_floor', 'valid_draws', 'ratio_of_equal_game_means'} and
                    gap['denominator_floor'] == GAP_FLOOR and gap['ratio_of_equal_game_means'] is True,
                    'Independent gap closure contract')
            _same(gap['baseline_gap'], denominator, 'Independent teacher-gap denominator')
            _same(gap['mean'], None if denominator <= GAP_FLOOR else (parent[0] - point[0]) / denominator,
                  'Independent ratio of equal-game means')
            valid, interval = None, None
            if distribution is not None:
                denominators = sampled[split][0, :, 0] - sampled[split][0, :, 1]
                valid = int(sum(denominators > GAP_FLOOR))
                if denominator > GAP_FLOOR and valid == draws:
                    interval = np.percentile((sampled[split][0, :, 0] - distribution[:, 0]) / denominators,
                                             [2.5, 97.5], method='linear')
            require(gap['valid_draws'] == valid, 'Independent undefined-gap draw count')
            _same(gap['ci95'], interval, 'Independent gap-closure interval')
        boundary()
    expected_keys = {(label, rep, split, metric) for label, rep, _, _ in _contrasts()
                     for split in VIEWS for metric in METRICS}
    require(len(reported['contrasts']) == len(expected_keys) and
            {(r['contrast'], r['seed'], r['split'], r['metric']) for r in reported['contrasts']} == expected_keys,
            'Complete prospective contrast inventory')
    contrast_models = {(label, rep): (left, right) for label, rep, left, right in _contrasts()}
    audited = {}
    for record in reported['contrasts']:
        require(set(record) == {'contrast', 'seed', 'split', 'metric', 'difference', 'ci95', 'ci97_5', 'secondary_policy_family'},
                'Exact contrast record schema')
        label, rep, split, metric = key = tuple(record[k] for k in ('contrast', 'seed', 'split', 'metric'))
        left, right = contrast_models[label, rep]
        i = METRICS.index(metric)
        difference = points[left][split][i] - points[right][split][i]
        dist = None if split == 'train' else distributions[left][split][:, i] - distributions[right][split][:, i]
        family = rep == 'mean' and split == 'pooled' and metric == 'policy_ce' and label != 'larger_minus_control'
        _same(record['difference'], difference, 'Independent paired contrast difference')
        _same(record['ci95'], None if dist is None else np.percentile(dist, [2.5, 97.5], method='linear'),
              'Independent paired contrast interval')
        _same(record['ci97_5'], np.percentile(dist, [1.25, 98.75], method='linear') if family else None,
              'Independent two-secondary-family interval')
        require(record['secondary_policy_family'] is family, 'Secondary family excludes the primary test')
        audited[key] = record
    primary = dict(audited['larger_minus_control', 'mean', 'pooled', 'policy_ce'])
    primary['conclusion'] = ('clearly_improves' if primary['ci95'][1] < 0 else
                             'clearly_worse' if primary['ci95'][0] > 0 else 'unresolved')
    primary['seed_differences'] = {rep: audited['larger_minus_control', rep, 'pooled', 'policy_ce']['difference'] for rep in ('r0', 'r1')}
    require(reported['primary'] == primary, 'Primary result and unselected seed directions')
    require(set(reported['training_membership']) == set(SOURCES), 'Both descriptive source masks')
    for source in SOURCES:
        positions = [i for i, game in enumerate(games['train']) if game[0] == source]
        saved = reported['training_membership'][source]
        require(set(saved) == {'games', 'roots', 'game_ids', 'models', 'descriptive_only', 'independent_assessment'} and
                saved['game_ids'] == [list(games['train'][i]) for i in positions] and saved['games'] == len(positions) and
                saved['roots'] == sum(reported['game_summaries']['parent']['train'][i]['roots'] for i in positions) and
                saved['descriptive_only'] is True and saved['independent_assessment'] is False and
                set(saved['models']) == set(definitions), 'Independent old/new descriptive membership')
        for name, coefficients in definitions.items():
            vector = sum(weight * matrices['train'][MODEL_IDS.index(model), positions].sum(axis=0) / len(positions)
                         for model, weight in coefficients.items())
            require(set(saved['models'][name]) == set(METRICS), 'Complete source-specific metrics')
            _same([saved['models'][name][m] for m in METRICS], vector, 'Independent old/new game-weighted means')
    return dict(status='independent_training4096_arithmetic_verified', model_calls=0, optimizer_calls=0,
                independent_loss_formula=True, independent_count_weight_bootstrap=True,
                seed_losses_averaged_not_predictions=True, namespaced_equal_game_weighting=True,
                same_draws_all_models_and_contrasts=True, draws=draws, seed=seed,
                fixed_models=len(MODEL_IDS), contrasts=len(expected_keys))


def render_charts(result, output, *, boundary=lambda: None):
    """Two fixed summaries; no selection of favourable seeds or metrics."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    folder = Path(output).resolve()
    require(folder.is_relative_to(c.OUT.resolve()), 'Charts stay in the new study output')
    folder.mkdir(parents=True, exist_ok=True)
    models = ('parent', 'historical_mean', 'control_mean', 'larger_mean')
    labels = ('Original EF8A', 'Historical\n1,024 one pass', 'Repeated\n1,024 control', 'Larger\n4,096 one pass')
    colors = ('#64748b', '#748ca8', '#c06832', '#007f85')
    artifacts = []

    def error_point(ax, x, row, color, scale=1.):
        if row['mean'] is None:
            ax.text(x, 0, 'Undefined', rotation=90, ha='center', va='bottom', fontsize=9)
            return
        ax.plot(x, row['mean'] * scale, 'o', color=color, markersize=7)
        if row['ci95'] is not None:
            ax.vlines(x, row['ci95'][0] * scale, row['ci95'][1] * scale, color=color, lw=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': (1.12, 1.)})
    for i, (model, color) in enumerate(zip(models, colors, strict=True)):
        error_point(axes[0], i, result['estimates'][model]['pooled']['teacher_gap_closed'], color, 100.)
        if model != 'parent':
            for rep in (0, 1):
                value = result['estimates'][model.replace('_mean', f'_r{rep}')]['pooled']['teacher_gap_closed']['mean']
                if value is not None:
                    axes[0].plot(i + (-.09 if rep == 0 else .09), value * 100, '.', color=color, alpha=.55)
    axes[0].set_xticks(range(4), labels, fontsize=10)
    axes[0].set_ylabel('Initial policy gap to teacher closed (%)')
    axes[0].set_title('Same fresh assessment for every model')
    axes[0].axhline(0, color='#94a3b8', lw=1, ls='--')
    contrast_labels = ('Larger − matched control\nPrimary', 'Larger − original EF8A\nSecondary',
                       'Larger − historical one pass\nSecondary')
    for i, label in enumerate(('larger_minus_control', 'larger_minus_parent', 'larger_minus_historical')):
        row = next(r for r in result['contrasts'] if (r['contrast'], r['seed'], r['split'], r['metric']) ==
                   (label, 'mean', 'pooled', 'policy_ce'))
        if row['ci97_5'] is not None:
            axes[1].hlines(i, *row['ci97_5'], color='#cbd5e1', lw=6)
        axes[1].hlines(i, *row['ci95'], color='#007f85' if i == 0 else '#526875', lw=2)
        axes[1].plot(row['difference'], i, 'o', color='#007f85' if i == 0 else '#526875')
    axes[1].set_yticks(range(3), contrast_labels, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='#94a3b8', ls='--', lw=1)
    axes[1].set_xlabel('Policy cross-entropy difference (lower is better)')
    axes[1].set_title('Paired differences in seed-mean losses')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.14)
    fig.suptitle('More independent games at matched training effort', fontsize=19)
    fig.text(.5, .045, '256 fresh games; equal game weights. Dots: fixed two-seed means; small dots: individual seeds.\n'
             'Intervals: paired whole-game 95%; pale secondary extensions: family-adjusted 97.5%. Offline results do not prove playing strength.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .13, 1, .92), w_pad=3)
    path = folder / 'policy_comparison.png'
    require(not path.exists(), 'Exclusive new policy chart')
    with path.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=100)
    plt.close(fig)
    artifacts.append(c.artifact(path)); boundary()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, metric, title in zip(axes, ('outcome_mse', 'mixed_value_mse'),
                                  ('Official-outcome value error', 'Mixed-target value error'), strict=True):
        for i, (model, color) in enumerate(zip(models, colors, strict=True)):
            error_point(ax, i, result['estimates'][model]['pooled'][metric], color)
            if model != 'parent':
                for rep in (0, 1):
                    value = result['estimates'][model.replace('_mean', f'_r{rep}')]['pooled'][metric]['mean']
                    ax.plot(i + (-.09 if rep == 0 else .09), value, '.', color=color, alpha=.55)
        ax.set_xticks(range(4), labels, fontsize=10)
        ax.set_title(title)
        ax.set_ylabel('Mean squared error (lower is better)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Value tradeoffs on the same fresh assessment', fontsize=19)
    fig.text(.5, .045, 'Fixed two-seed means and individual seeds; equal game weights; paired whole-game 95% descriptive intervals.\n'
             'Value metrics do not replace the prespecified primary policy comparison. No model is selected or promoted.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .13, 1, .92), w_pad=3)
    path = folder / 'value_tradeoffs.png'
    require(not path.exists(), 'Exclusive new value chart')
    with path.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=100)
    plt.close(fig)
    artifacts.append(c.artifact(path)); boundary()
    return artifacts


def _analysis_folder(folder):
    path = Path(folder).resolve()
    require(path == (c.OUT / 'jobs/analysis/worker').resolve(), 'Exact new analysis worker folder')
    require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
            'Fresh saved-only analysis worker')
    return path


def run_analysis(folder, *, boundary=lambda: None):
    from . import offline
    folder = _analysis_folder(folder)
    bundles, evaluation = offline.load_predictions(boundary=boundary)
    source = c.artifact(c.OUT / 'jobs/offline/worker/evaluation_result.json')
    require(c.read(c.resolve(source)) == evaluation, 'Exact authenticated evaluation document')
    result = analyze_predictions(bundles, boundary=boundary)
    proof = verify_arithmetic(bundles, result, boundary=boundary)
    analysis = c.save(folder / 'analysis.json', result)
    arithmetic = c.save(folder / 'arithmetic_verification.json', proof)
    charts = render_charts(result, folder, boundary=boundary)
    c.verify([source], boundary=boundary)
    receipt = dict(schema='training4096_analysis_result_v1', status='analysis_verified',
                   evaluation=source, analysis=analysis, arithmetic=arithmetic, charts=charts,
                   model_calls=0, optimizer_calls=0, **c.IDENTITY)
    item = c.save(folder / 'analysis_result.json', receipt)
    return dict(result=item, analysis=analysis, arithmetic=arithmetic, charts=charts)


def verify_analysis(folder, *, boundary=lambda: None):
    from . import offline
    folder = _analysis_folder(folder)
    item = c.artifact(folder / 'analysis_result.json')
    result = c.read(c.resolve(item))
    require(set(result) == {'schema', 'status', 'evaluation', 'analysis', 'arithmetic', 'charts',
                            'model_calls', 'optimizer_calls', *c.IDENTITY} and
            result['schema'] == 'training4096_analysis_result_v1' and result['status'] == 'analysis_verified' and
            result['model_calls'] == result['optimizer_calls'] == 0 and
            all(result[k] == v for k, v in c.IDENTITY.items()), 'Exact saved analysis result')
    expected = {'evaluation': c.OUT / 'jobs/offline/worker/evaluation_result.json',
                'analysis': folder / 'analysis.json', 'arithmetic': folder / 'arithmetic_verification.json'}
    for key, path in expected.items():
        require(result[key] == c.artifact(path), 'Exact bound analysis artifact ' + key)
    require(result['charts'] == [c.artifact(folder / name) for name in ('policy_comparison.png', 'value_tradeoffs.png')],
            'Both fixed charts preserved')
    bundles, evaluation = offline.load_predictions(boundary=boundary)
    require(evaluation == c.read(c.resolve(result['evaluation'])), 'Analysis binds authenticated saved evaluation')
    proof = verify_arithmetic(bundles, c.read(c.resolve(result['analysis'])), boundary=boundary)
    require(proof == c.read(c.resolve(result['arithmetic'])), 'Independent saved arithmetic proof unchanged')
    return dict(status='training4096_analysis_saved_verified', result=item, analysis=result['analysis'],
                arithmetic=result['arithmetic'], charts=result['charts'], model_calls=0, optimizer_calls=0)
