"""Saved-prediction analysis only; no network, optimizer, search or game calls.

Uncertainty resamples whole held-out games, using the same draws for every
checkpoint, training scope and sampling repetition. Source/loader correctness is
separately required by the study auditor; this reader does not replace it.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import math
import re
import struct
import sys

import numpy as np

from . import common as c

STEPS = (0, 96)
SPLITS = ('train', 'validation', 'offline')
HELDOUT = {'validation': list(range(192, 224)), 'offline': list(range(224, 256))}
HELDOUT_ROOTS = {'validation': 1802, 'offline': 1755}
METRICS = ('policy', 'value', 'combined')
ROW_KEYS = {'game_index', 'ply', 'split', 'root_input_sha256', 'row_sha256',
            'legal_mask81', 'policy_target81', 'value_target2', 'output83'}
BATCH_KEYS = {'fit_id', 'repetition', 'breadth', 'scope', 'checkpoint_step',
              'split', 'batch_index', 'selection_sha256', 'weights_sha256', 'checkpoint_manifest', 'rows'}
BOOTSTRAP_SEED = 20261106
BOOTSTRAP_DRAWS = 20000


def require(value, message):
    if not value:
        raise ValueError('LEARNING_ANALYSIS: ' + message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def f32_bytes(values):
    try:
        return struct.pack('<' + 'f' * len(values), *values)
    except (OverflowError, struct.error) as error:
        raise ValueError('LEARNING_ANALYSIS: finite float32 outputs required') from error


def prediction_metrics(row):
    """Independent stable float64 masked CE, official signed-value MSE."""
    require(set(row) == ROW_KEYS, 'exact prediction row schema')
    require(type(row['game_index']) is int and row['game_index'] >= 0 and
            type(row['ply']) is int and row['ply'] >= 0 and row['split'] in SPLITS,
            'canonical game, ply and split')
    require(all(isinstance(row[key], str) and re.fullmatch('[0-9a-f]{64}', row[key])
                for key in ('root_input_sha256', 'row_sha256')), 'source/input digests')
    mask, target, values, output = (row[key] for key in
        ('legal_mask81', 'policy_target81', 'value_target2', 'output83'))
    require(len(mask) == len(target) == 81 and all(type(x) is bool for x in mask)
            and any(mask), 'complete nonempty legal mask')
    require(all(number(x) and x >= 0 for x in target) and
            abs(math.fsum(target) - 1.) <= 1e-12 and
            all(ok or x == 0 for ok, x in zip(mask, target)), 'normalized legal visit target')
    require(len(values) == 2 and all(number(x) and x in (-1., 0., 1.) for x in values)
            and sum(values) == 0, 'official antisymmetric signed outcomes')
    require(len(output) == 83 and all(number(x) for x in output), 'finite output83')
    packed = f32_bytes(output)
    require(tuple(output) == struct.unpack('<83f', packed), 'actual float32 outputs')
    legal = [i for i, ok in enumerate(mask) if ok]
    maximum = max(output[i] for i in legal)
    log_sum = math.log(math.fsum(math.exp(output[i] - maximum) for i in legal))
    # Subtract before adding log_sum to avoid cancellation for large common logits.
    policy = math.fsum(t * ((maximum - x) + log_sum)
                       for t, x in zip(target, output[:81]) if t)
    value = math.fsum((x-y)**2 for x, y in zip(output[81:], values)) / 2.
    entropy = -math.fsum(t * math.log(t) for t in target if t)
    require(math.isfinite(policy) and policy >= -1e-12 and math.isfinite(value), 'finite losses')
    predicted = max(legal, key=lambda i: (output[i], -i))
    teacher = max(legal, key=lambda i: (target[i], -i))
    return dict(policy=max(0., policy), value=value, combined=max(0., policy)+value,
                teacher_entropy=entropy, policy_kl=max(0., policy)-entropy,
                teacher_top_action_agreement=float(predicted == teacher),
                policy_argmax=predicted)


def row_identity(row):
    return {key: value for key, value in row.items() if key != 'output83'}


def summarize_rows(rows, parent_rows=None, *, expected_games=None, policy_only=False):
    """Average roots within each game, then games, irrespective of game length."""
    require(isinstance(rows, list) and rows, 'nonempty complete prediction rows')
    parent_rows = rows if parent_rows is None else parent_rows
    require(len(rows) == len(parent_rows), 'complete paired root inventory')
    keys = [(row['game_index'], row['ply']) for row in rows]
    require(keys == sorted(set(keys)), 'canonical unique game/ply rows')
    by_game = defaultdict(list)
    for row, parent in zip(rows, parent_rows):
        metrics, baseline = prediction_metrics(row), prediction_metrics(parent)
        require(row_identity(row) == row_identity(parent), 'same source, mask and target across checkpoints')
        if policy_only:
            require(f32_bytes(row['output83'][81:]) == f32_bytes(parent['output83'][81:]),
                    'policy-only values preserve exact parent float32 bits')
        predicted = metrics.pop('policy_argmax')
        metrics.update(policy_argmax_changed=float(predicted != baseline['policy_argmax']),
                       signed_value_shift=row['output83'][81] - parent['output83'][81],
                       mean_absolute_value_shift=math.fsum(abs(a-b) for a, b in
                           zip(row['output83'][81:], parent['output83'][81:])) / 2.)
        by_game[row['game_index']].append(metrics)
    if expected_games is not None:
        require(sorted(by_game) == list(expected_games), 'all and only declared source games')
    names = tuple(next(iter(by_game.values()))[0])
    games = [dict(game_index=game, roots=len(values), **{
        key: math.fsum(value[key] for value in values)/len(values) for key in names})
        for game, values in sorted(by_game.items())]
    return dict(game_count=len(games), root_count=len(rows), games=games,
                equal_game_means={key: math.fsum(game[key] for game in games)/len(games) for key in names})


def pooled_summary(validation, offline):
    games = validation['games'] + offline['games']
    require(len({row['game_index'] for row in games}) == len(games), 'disjoint held-out games')
    names = tuple(validation['equal_game_means'])
    return dict(game_count=len(games), root_count=sum(x['roots'] for x in games), games=games,
                equal_game_means={key: math.fsum(row[key] for row in games)/len(games) for key in names})


def bootstrap_indices(game_count, rng, draws=BOOTSTRAP_DRAWS):
    require(type(game_count) is int and game_count >= 2 and type(draws) is int and draws > 0,
            'positive whole-game bootstrap dimensions')
    return rng.integers(0, game_count, size=(draws, game_count), dtype=np.int64)


def describe_difference(values, denominator, indices, *, boundary=lambda: None):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), 'finite paired game differences')
    require(indices.ndim == 2 and indices.shape[1] == len(values) and indices.dtype.kind in 'iu'
            and int(indices.min()) >= 0 and int(indices.max()) < len(values), 'paired whole-game draw indices')
    samples = []
    for start in range(0, len(indices), 512):
        boundary()
        samples.append(values[indices[start:start+512]].mean(axis=1))
    samples = np.concatenate(samples)
    delta = math.fsum(values.tolist())/len(values)
    interval = np.quantile(samples, (.025, .975), method='linear').tolist()
    baseline = math.fsum(denominator)/len(denominator) if denominator is not None else None
    require(baseline is None or (number(baseline) and baseline >= 0), 'nonnegative reference mean loss')
    return dict(absolute_difference=delta, nominal_ci95=interval, game_count=len(values),
                games_lower=int((values < 0).sum()), games_higher=int((values > 0).sum()),
                games_equal=int((values == 0).sum()), reference_equal_game_mean=baseline,
                relative_percent=None if baseline in (None, 0.) else 100.*delta/baseline,
                relative_percent_definition='ratio of equal-game means; null for zero or interaction reference',
                interval_method='percentile, linear quantiles, paired whole-game bootstrap',
                inferential_status='exploratory nominal interval, no multiplicity adjustment')


def _series(summaries, repetition, step, split, metric):
    key = f'r{repetition}_broad_joint'
    rows = summaries[key]['checkpoints'][str(step)][split]['games']
    return np.asarray([row[metric] for row in rows], dtype=np.float64)


def contrast_rows(summaries, split, indices, *, boundary=lambda: None):
    """Fixed step 96 minus original parent; repetitions share held-out games."""
    records = []
    for repetition in (0, 1, 'mean'):
        reps = (0, 1) if repetition == 'mean' else (repetition,)
        for metric in ('policy', 'value'):
            parent = sum((_series(summaries, rep, 0, split, metric) for rep in reps))/len(reps)
            candidate = sum((_series(summaries, rep, 96, split, metric) for rep in reps))/len(reps)
            values = describe_difference(candidate-parent, parent, indices, boundary=boundary)
            records.append(dict(kind='fixed_checkpoint_vs_parent', definition='step 96 minus pristine EF8A',
                breadth='broad', scope='joint', metric=metric, repetition=repetition,
                endpoint='primary', checkpoint_step=96, split=split,
                candidate_equal_game_mean=math.fsum(candidate.tolist())/len(candidate),
                interval_spans_zero=values['nominal_ci95'][0] <= 0 <= values['nominal_ci95'][1], **values))
    return records


def classify_direction(contrasts):
    """Prospective directional rule, deliberately separate from uncertainty."""
    rows = [row for row in contrasts if row['split'] == 'pooled' and row['repetition'] in (0, 1)]
    require(len(rows) == 4 and {(x['repetition'], x['metric']) for x in rows} ==
            {(rep, metric) for rep in (0, 1) for metric in ('policy', 'value')}, 'four fixed directional endpoints')
    deltas = [row['absolute_difference'] for row in rows]
    if all(value < 0 for value in deltas):
        label = 'consistent_promising_direction'
    elif any(value > 0 for value in deltas):
        label = 'tradeoff_or_degradation'
    else:
        label = 'no_consistent_strict_improvement'
    return dict(label=label, rule='Both losses below parent in both pooled repetitions; any higher pooled loss is tradeoff/degradation; otherwise no consistent strict improvement.',
                based_on='four pooled point differences, not confidence-interval selection',
                any_primary_repeat_interval_spans_zero=any(row['interval_spans_zero'] for row in rows),
                strength_improvement_established=False, checkpoint_selected=False, automatic_followup=False)


def analyze_bundle(bundles, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                   expected_heldout=HELDOUT, strict_counts=True, boundary=lambda: None):
    """Pure saved-row computation; tests may supply smaller declared cohorts."""
    require(len(bundles) == 2 and [bundle['spec'] for bundle in bundles] == c.fits(), 'fixed two broad joint fits')
    summaries, parent_heldout = {}, {}
    for bundle in bundles:
        boundary(); spec, evaluations = bundle['spec'], bundle['evaluations']
        require(set(evaluations) == {str(step) for step in STEPS}, 'fixed checkpoints only')
        checkpoints = {}
        for step in STEPS:
            require(set(evaluations[str(step)]) == set(SPLITS), 'all train and held-out splits')
            checkpoints[str(step)] = {}
            for split in SPLITS:
                boundary(); rows = evaluations[str(step)][split]; parent = evaluations['0'][split]
                games = sorted({row['game_index'] for row in parent}) if split == 'train' else expected_heldout[split]
                require(all(row['split'] == split for row in rows), 'batch and row split agree')
                if strict_counts:
                    if split == 'train':
                        require(spec['breadth'] == 'broad' and spec['scope'] == 'joint' and
                                len(rows) == 3072 and len(games) == 192
                                and all(sum(row['game_index'] == g for row in rows) == 16 for g in games),
                                'complete equal-count selected training roots')
                        require(all(0 <= game < 192 for game in games), 'training excludes held-out games')
                    else:
                        require(len(rows) == HELDOUT_ROOTS[split], 'all original held-out roots')
                if split != 'train':
                    if split not in parent_heldout:
                        parent_heldout[split] = parent
                    else:
                        expected = parent_heldout[split]
                        require(len(parent) == len(expected) and all(row_identity(a) == row_identity(b) and
                            f32_bytes(a['output83']) == f32_bytes(b['output83']) for a, b in zip(parent, expected)),
                            'one exact pristine-parent held-out baseline across all fits')
                checkpoints[str(step)][split] = summarize_rows(rows, parent, expected_games=games,
                    policy_only=spec['scope'] == 'policy')
            checkpoints[str(step)]['pooled'] = pooled_summary(checkpoints[str(step)]['validation'], checkpoints[str(step)]['offline'])
        summaries[spec['fit_id']] = dict(spec, checkpoints=checkpoints)
    rng = np.random.Generator(np.random.PCG64(seed)); contrasts, bootstrap = [], {}
    for split in ('validation', 'offline', 'pooled'):
        boundary(); ids = (expected_heldout['validation']+expected_heldout['offline']) if split == 'pooled' else expected_heldout[split]
        indices = bootstrap_indices(len(ids), rng, draws)
        bootstrap[split] = dict(game_ids=ids,draws=draws,index_sha256=hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest())
        contrasts.extend(contrast_rows(summaries, split, indices, boundary=boundary))
    return dict(schema='shorter_joint_saved_analysis_v1', status='saved_predictions_analyzed',
        summaries=summaries, contrasts=contrasts,
        directional_classification=classify_direction(contrasts),
        bootstrap=dict(seed=seed,draws=draws,generator='numpy.PCG64',split_order=['validation','offline','pooled'],
                       shared_indices_across_all_cells_checkpoints_repetitions=True,splits=bootstrap),
        original_heldout_games_already_opened=True,checkpoint_selection=False,reference_promoted=False,
        stronger_player_established=False,new_games=0,analysis_model_calls=0,analysis_optimizer_steps=0,
        limitations=['Exploratory comparison on already-opened games, not independent strength confirmation.',
                     'Two sampling repetitions share source games and are not independent new corpora.',
                     'Nominal intervals have no family-wide error control.',
                     'Conclusions apply to 3,072 records and 96 fixed updates: one complete pass.',
                     'Step96 is fixed prospectively; no intermediate checkpoint is evaluated or selected.',
                     'Previous 668-step findings are historical context, not an independent comparison arm.',
                     'No result establishes benefit from data beyond 192 games or need for a larger network.'])


def load_bundle(plan, spec, folder, *, boundary=lambda: None):
    """Authenticate explicit evaluator manifests, never discover a favorable subset."""
    folder = Path(folder).resolve(); result_path = folder/'diagnostics_result.json'
    boundary(); result = c.read(result_path)
    require(all(result[key] == spec[key] for key in spec), 'fixed fit identity in evaluator result')
    require(result['selection_sha256'] == c.digest(c.read(c.resolve(plan['selection']))), 'same frozen canonical selection digest')
    require(result['schema'] == 'learning_diagnostic_evaluation_v1' and result['status'] == 'diagnostics_complete'
            and result['optimizer_calls'] == 0 and result['opened_heldout_exploratory_only'] is True
            and result['checkpoint_selection'] is False and result['strength_claim'] is False,
            'complete fixed-checkpoint exploratory evaluator')
    require([row['checkpoint_step'] for row in result['checkpoints']] == list(STEPS), 'complete fixed checkpoint order')
    evaluations, used, diagnostics = {}, [c.artifact(result_path)], []
    for checkpoint in result['checkpoints']:
        boundary(); step = checkpoint['checkpoint_step']; descriptors = checkpoint['batches']
        c.verify([checkpoint['checkpoint_manifest']], boundary=boundary)
        manifest = c.read(c.resolve(checkpoint['checkpoint_manifest']))
        require(manifest['schema'] == 'learning_diagnostic_checkpoint_manifest_v1' and
                all(manifest['metadata'][key] == spec[key] for key in spec) and
                manifest['metadata']['checkpoint_step'] == step and
                manifest['metadata']['weights_sha256'] == checkpoint['weights_sha256'] and
                manifest['metadata']['selection_sha256'] == result['selection_sha256'],
                'saved checkpoint metadata matches evaluator and selection')
        used.append(checkpoint['checkpoint_manifest'])
        require(len({item['path'] for item in descriptors}) == len(descriptors), 'unique batch artifacts')
        evaluations[str(step)] = {split: [] for split in SPLITS}
        expected_paths = []
        for split in SPLITS:
            selected = [item for item in descriptors if c.resolve(item).parent == folder/'evaluation'/f'step_{step:03d}'/split]
            require(selected, 'every fixed split has prediction batches')
            for index, item in enumerate(selected):
                boundary(); path = folder/'evaluation'/f'step_{step:03d}'/split/f'{index:06d}.json'
                require(c.resolve(item) == path, 'canonical complete prediction batch order')
                c.verify([item], boundary=boundary); batch = c.read(path)
                require(set(batch) == BATCH_KEYS and all(batch[key] == spec[key] for key in spec)
                    and batch['checkpoint_step'] == step and batch['split'] == split and batch['batch_index'] == index
                    and batch['selection_sha256'] == result['selection_sha256']
                    and batch['weights_sha256'] == checkpoint['weights_sha256']
                    and batch['checkpoint_manifest'] == checkpoint['checkpoint_manifest'], 'bound batch/model/selection identity')
                require(isinstance(batch['rows'], list) and 0 < len(batch['rows']) <= 32 and
                        (index == len(selected)-1 or len(batch['rows']) == 32), 'complete original batch inventory')
                evaluations[str(step)][split].extend(batch['rows']); expected_paths.append(path);used.append(item)
        require({c.resolve(item) for item in descriptors} == set(expected_paths) and
                set((folder/'evaluation'/f'step_{step:03d}').rglob('*.json')) == set(expected_paths), 'no omitted or extra prediction batches')
        require(checkpoint['gradient_probes'] == [] and checkpoint['backward_probes'] == 0 and
                checkpoint['forward_calls'] == len(descriptors), 'no additional gradient probes or model calls')
        diagnostics.append(dict(checkpoint_step=step,group_displacement=checkpoint['group_displacement'],
                                gradient_probes=[]))
    require(not (folder/'probes').exists(), 'no probe evidence from unscheduled execution')
    require(result['backward_probes'] == 0 and
            result['forward_calls'] == sum(row['forward_calls'] for row in result['checkpoints']),
            'complete fixed evaluator call totals')
    return dict(spec=spec,evaluations=evaluations), dict(result=c.artifact(result_path),sources=used,
        parameter_and_gradient_diagnostics=diagnostics)


def run_analysis(plan, diagnostic_folders, output, *, boundary=lambda: None):
    require_no_torch()
    require(plan['fits'] == c.fits() and len(plan['fits']) == 2 and
            set(diagnostic_folders) == {x['fit_id'] for x in c.fits()}, 'both fixed evaluator folders')
    c.verify([plan['selection']], boundary=boundary)
    bundles, provenance = [], {}
    for spec in plan['fits']:
        bundle, evidence = load_bundle(plan, spec, diagnostic_folders[spec['fit_id']], boundary=boundary)
        bundles.append(bundle);provenance[spec['fit_id']] = evidence
    result = analyze_bundle(bundles, boundary=boundary)
    result['provenance'] = provenance
    result['source_binding_scope'] = 'Batch bytes and evaluator manifests checked here; source/loader/root identities require independent study audit.'
    output = Path(output); analysis_item = c.save(output/'analysis.json', result)
    del bundles
    report = render_report(result, output, analysis_item, boundary=boundary)
    require_no_torch()
    return dict(status='analysis_and_report_complete_pending_independent_acceptance',analysis=analysis_item,**report,
                new_games=0,models=0,optimizer_steps=0)


def require_no_torch():
    require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
            'Analysis and plotting require a fresh worker without Torch')


def loss_change_series(result, metric):
    """Common relative scale only when every plotted parent mean is positive."""
    values = {}
    for rep in (0, 1):
        for split in ('train', 'pooled'):
            steps = result['summaries'][f'r{rep}_broad_joint']['checkpoints']
            before = steps['0'][split]['equal_game_means'][metric]
            after = steps['96'][split]['equal_game_means'][metric]
            require(number(before) and number(after) and before >= 0 and after >= 0, 'finite plotted losses')
            values[(rep, split)] = (before, after)
    relative = all(before > 0 for before, after in values.values())
    return {key: 100*(after-before)/before if relative else after-before
            for key, (before, after) in values.items()}, ('Loss change from parent (%)' if relative else 'Absolute loss change (zero parent baseline)')


def render_report(result, output, analysis_item, *, boundary=lambda: None):
    """Fresh no-Torch renderer; two fixed charts with explicit exploratory scope."""
    require_no_torch()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    output = Path(output).resolve()
    require(output.is_relative_to(c.OUT), 'report stays inside new evidence root')
    output.mkdir(parents=True, exist_ok=True)
    chart1, chart2 = output/'fixed_endpoint.png', output/'training_transfer.png'
    document = output/'results.md'
    require(not any(p.exists() for p in (chart1, chart2, document)), 'exclusive report artifacts')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    boundary()
    rows = [(split, rep) for split in ('validation', 'offline', 'pooled') for rep in (0, 1, 'mean')]
    labels = [f'{split.title()} / '+('mean of repeats' if rep == 'mean' else f'repeat {rep+1}') for split, rep in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.3), sharey=True)
    for ax, metric, title in zip(axes, ('policy', 'value'), ('Policy: search-move prediction', 'Value: outcome prediction')):
        for y, (split, rep) in enumerate(rows):
            row = next(x for x in result['contrasts'] if x['metric'] == metric and x['split'] == split and x['repetition'] == rep)
            lo, hi = row['nominal_ci95']; value = row['absolute_difference']
            color = '#0f766e' if rep == 'mean' else '#8ba2ac'
            ax.plot([lo, hi], [y, y], color=color, lw=2.2)
            ax.scatter([value], [y], color=color, s=48 if rep == 'mean' else 25, zorder=3)
        ax.axvline(0, color='#475569', ls='--', lw=1)
        for y in (2.5, 5.5): ax.axhline(y, color='#e2e8f0', lw=.8)
        ax.set_title(title, pad=14)
        ax.set_xlabel('Loss after 96 updates minus original loss\nNegative means lower loss')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useOffset=False)
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Does one training pass improve saved-position predictions?', fontsize=17, y=.97)
    fig.text(.5, .035, 'Fixed step 96 versus pristine EF8A / two predetermined repetitions / 64 previously opened held-out games\nBars: nominal 95% paired whole-game bootstrap intervals. This is not a playing-strength test.', ha='center', fontsize=10)
    fig.tight_layout(rect=(.01, .12, .99, .92))
    fig.savefig(chart1, dpi=160); plt.close(fig)
    boundary()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.9))
    for ax, metric in zip(axes, ('policy', 'value')):
        changes, label = loss_change_series(result, metric)
        x = np.arange(2); width = .32
        ax.bar(x-width/2, [changes[(rep,'train')] for rep in (0,1)], width, color='#2563eb', label='Selected training roots')
        ax.bar(x+width/2, [changes[(rep,'pooled')] for rep in (0,1)], width, color='#0f766e', label='Held-out games')
        ax.axhline(0, color='#475569', ls='--', lw=1)
        ax.set_xticks(x, ['Repeat 1', 'Repeat 2']); ax.set_title(metric.title()+' loss')
        ax.set_ylabel(label); ax.grid(axis='y', alpha=.15)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .065), frameon=False)
    fig.suptitle('Does the training improvement transfer?', fontsize=17, y=.97)
    fig.text(.5, .025, 'After one fixed pass / equal-game losses / negative is lower loss\nAlready-opened held-out data; no checkpoint selection or strength claim.', ha='center', fontsize=10)
    fig.tight_layout(rect=(.015, .18, .99, .90)); fig.savefig(chart2, dpi=160); plt.close(fig)
    boundary(); require_no_torch()
    classification = result['directional_classification']
    lines = ['# One-pass joint-training diagnostic', '',
        'Two fresh joint fits each used the unchanged 3,072-root broad selection and the first 96 batches of its original fixed order. '+
        'The only evaluated endpoints were the pristine parent and the prospectively fixed step 96. These results require independent study acceptance.', '',
        '| Pooled endpoint, mean of repeats | Change from original | Absolute change | Nominal 95% interval |',
        '|---|---:|---:|---:|']
    for metric in ('policy', 'value'):
        row = next(x for x in result['contrasts'] if x['metric'] == metric and x['split'] == 'pooled' and x['repetition'] == 'mean')
        lo, hi = row['nominal_ci95']; relative = row['relative_percent']
        label = 'undefined (zero baseline)' if relative is None else f'{relative:+.3f}%'
        lines.append(f"| {metric.title()} | {label} | {row['absolute_difference']:+.6f} | {lo:+.6f} to {hi:+.6f} |")
    lines += ['', '**Fixed directional classification:** '+classification['label'].replace('_',' ')+'.',
        'This label uses the four pooled point estimates: both losses must improve in both repetitions. '+
        'It does not turn an interval spanning zero into evidence of a reliable improvement.', '',
        f'![Fixed endpoint]({chart1.as_posix()})', '', f'![Training transfer]({chart2.as_posix()})', '',
        'Both held-out splits and both repetitions are shown separately. Their mean resamples the same 64 games, '+
        'not 128 independent observations. The 20,000 bootstrap draws are shared across repetitions and endpoints; intervals are nominal and exploratory.', '',
        '## What this result establishes', '',
        'This isolates a prospectively fixed shorter update schedule on the existing corrected corpus. '+
        'It measures whether one pass improves saved-position policy and value predictions. '+
        'It cannot establish stronger play, a unique cause of earlier overfitting, or a benefit from collecting more games.', '',
        '## The next step', '',
        'Use the fixed classification, both repetitions and their uncertainty to recommend one further research question. '+
        'No checkpoint is promoted and no training or fresh-game assessment follows automatically.', '',
        '## The broader picture', '',
        'The wider goal is a stronger same-sized player trained from verified search guidance. '+
        'The earlier long-update diagnostic worsened held-out joint losses; this test checks one predetermined shorter schedule '+
        'before committing to more data. Any eventual playing-strength claim still needs independent fresh paired games. EF8A remains the reference.', '',
        f"Analysis SHA256: `{analysis_item['sha256']}`."]
    with document.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines)+'\n')
    return dict(charts=[c.artifact(chart1),c.artifact(chart2)], report=c.artifact(document))
