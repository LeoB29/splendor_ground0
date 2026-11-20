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

import numpy as np

from . import common as c

STEPS = (0, 334, 668)
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


def _series(summaries, repetition, breadth, scope, step, split, metric):
    key = f'r{repetition}_{breadth}_{scope}'
    rows = summaries[key]['checkpoints'][str(step)][split]['games']
    return np.asarray([row[metric] for row in rows], dtype=np.float64)


def contrast_rows(summaries, step, split, indices, *, boundary=lambda: None):
    records = []
    for repetition in (0, 1, 'mean'):
        reps = (0, 1) if repetition == 'mean' else (repetition,)
        def values(breadth, scope, metric):
            return sum((_series(summaries, rep, breadth, scope, step, split, metric) for rep in reps))/len(reps)
        for scope in ('joint', 'policy'):
            for metric in ('policy', 'value'):
                dense, broad = values('dense', scope, metric), values('broad', scope, metric)
                primary = metric == 'policy' or scope == 'joint'
                records.append(dict(kind='breadth', definition='broad minus dense', scope=scope,
                    metric=metric, repetition=repetition,
                    endpoint=('primary' if primary else 'correctness') if step == 668 else 'diagnostic',
                    checkpoint_step=step, split=split,
                    **describe_difference(broad-dense, dense, indices, boundary=boundary)))
        for breadth in ('dense', 'broad'):
            for metric in ('policy', 'value'):
                joint, policy = values(breadth, 'joint', metric), values(breadth, 'policy', metric)
                records.append(dict(kind='scope', definition='joint minus policy-only', breadth=breadth,
                    metric=metric, repetition=repetition, endpoint='secondary' if step == 668 else 'diagnostic', checkpoint_step=step, split=split,
                    **describe_difference(joint-policy, policy, indices, boundary=boundary)))
        for metric in ('policy', 'value'):
            interaction = (values('broad', 'joint', metric)-values('dense', 'joint', metric))-(
                           values('broad', 'policy', metric)-values('dense', 'policy', metric))
            records.append(dict(kind='interaction', definition='joint breadth effect minus policy-only breadth effect',
                metric=metric, repetition=repetition, endpoint='secondary' if step == 668 else 'diagnostic', checkpoint_step=step, split=split,
                **describe_difference(interaction, None, indices, boundary=boundary)))
    return records


def analyze_bundle(bundles, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                   expected_heldout=HELDOUT, strict_counts=True, boundary=lambda: None):
    """Pure saved-row computation; tests may supply smaller declared cohorts."""
    require(len(bundles) == 8 and [bundle['spec'] for bundle in bundles] == c.fits(), 'fixed eight-cell order')
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
                        per_game = 48 if spec['breadth'] == 'dense' else 16
                        require(len(rows) == 3072 and len(games) == (64 if per_game == 48 else 192)
                                and all(sum(row['game_index'] == g for row in rows) == per_game for g in games),
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
        for step in (334, 668):
            contrasts.extend(contrast_rows(summaries, step, split, indices, boundary=boundary))
    return dict(schema='learning_diagnostic_saved_analysis_v1', status='saved_predictions_analyzed',
        summaries=summaries, contrasts=contrasts,
        bootstrap=dict(seed=seed,draws=draws,generator='numpy.PCG64',split_order=['validation','offline','pooled'],
                       shared_indices_across_all_cells_checkpoints_repetitions=True,splits=bootstrap),
        original_heldout_games_already_opened=True,checkpoint_selection=False,reference_promoted=False,
        stronger_player_established=False,new_games=0,analysis_model_calls=0,analysis_optimizer_steps=0,
        limitations=['Exploratory comparison on already-opened games, not independent strength confirmation.',
                     'Two sampling repetitions share source games and are not independent new corpora.',
                     'Nominal intervals have no family-wide error control.',
                     'Conclusions apply to 3,072 records and 668 updates, about 6.96 passes.',
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
        probes=[]
        require(len(checkpoint['gradient_probes']) == 4, 'four predetermined gradient batches per checkpoint')
        for index,item in enumerate(checkpoint['gradient_probes']):
            c.verify([item],boundary=boundary);probe=c.read(c.resolve(item));used.append(item)
            require(c.resolve(item) == folder/'probes'/f'step_{step:03d}'/f'{index:02d}.json' and
                    all(probe[key] == spec[key] for key in spec) and probe['checkpoint_step'] == step and
                    probe['probe_index'] == index and probe['selection_sha256'] == result['selection_sha256'] and
                    probe['checkpoint_manifest'] == checkpoint['checkpoint_manifest'] and
                    probe['checkpoint_weights_sha256'] == checkpoint['weights_sha256'] and
                    probe['optimizer_calls'] == 0 and probe['forward_calls'] == 1 and
                    probe['backward_probes'] == (2 if spec['scope']=='joint' else 1) and
                    probe['weights_unchanged'] is probe['rng_unchanged'] is probe['zero_optimizer_calls'] is True,
                    'exact disposable no-update probe identity and inventory')
            for loss in ('policy','value') if spec['scope']=='joint' else ('policy',):
                require(all(number(probe[loss][key]) and probe[loss][key]>=0 for key in ('norm','shared_norm'))
                        and probe[loss]['finite'] is True, 'finite nonnegative recorded gradient norms')
            if spec['scope']=='joint':
                denominator=probe['policy']['shared_norm']*probe['value']['shared_norm']
                require(number(probe['shared_dot']) and probe['undefined_cosine'] == (denominator==0),
                        'explicit defined/undefined shared gradient alignment')
                if denominator:
                    require(number(probe['shared_cosine']) and -1.000000001<=probe['shared_cosine']<=1.000000001 and
                            math.isclose(probe['shared_cosine'],probe['shared_dot']/denominator,rel_tol=1e-12,abs_tol=1e-12),
                            'gradient alignment reconstructed from saved norms and dot')
                else:require(probe['shared_cosine'] is None,'no invented zero-norm cosine')
            probes.append(probe)
        require(checkpoint['backward_probes'] == 4*(2 if spec['scope']=='joint' else 1) and
                checkpoint['forward_calls'] == len(descriptors)+4,'declared evaluator model-call inventory')
        diagnostics.append(dict(checkpoint_step=step,group_displacement=checkpoint['group_displacement'],
                                gradient_probes=probes))
    require(result['backward_probes'] == 12*(2 if spec['scope']=='joint' else 1) and
            result['forward_calls'] == sum(row['forward_calls'] for row in result['checkpoints']),
            'complete fixed evaluator call totals')
    return dict(spec=spec,evaluations=evaluations), dict(result=c.artifact(result_path),sources=used,
        parameter_and_gradient_diagnostics=diagnostics)


def run_analysis(plan, diagnostic_folders, output, *, boundary=lambda: None):
    require(plan['fits'] == c.fits() and set(diagnostic_folders) == {x['fit_id'] for x in c.fits()}, 'all eight fixed evaluator folders')
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
    return dict(status='analysis_and_report_complete_pending_independent_acceptance',analysis=analysis_item,**report,
                new_games=0,models=0,optimizer_steps=0)


def trajectory_series(result, split, metric):
    """Use a common absolute scale if any compared parent loss is zero."""
    series={}
    for breadth,scope in (('dense','joint'),('broad','joint'),('dense','policy'),('broad','policy')):
        values=[math.fsum(result['summaries'][f'r{rep}_{breadth}_{scope}']['checkpoints'][str(step)][split]
                ['equal_game_means'][metric] for rep in (0,1))/2 for step in STEPS]
        require(all(number(value) and value>=0 for value in values),'finite nonnegative plotted losses')
        series[(breadth,scope)]=values
    percentage=all(values[0]>0 for values in series.values())
    return {key:[100*(value-values[0])/values[0] if percentage else value-values[0]
                 for value in values] for key,values in series.items()}, (
        'Change from parent (%)' if percentage else 'Absolute change from parent (zero baseline)')


def render_report(result, output, analysis_item, *, boundary=lambda: None):
    """At most two static scientific figures; no winner or promotion algorithm."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    output = Path(output).resolve()
    require(output.is_relative_to(c.OUT), 'report stays inside new evidence root')
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    chart1 = output/'breadth_contrasts.png';chart2 = output/'loss_trajectories.png';document=output/'results.md'
    require(not any(path.exists() for path in (chart1,chart2,document)), 'exclusive report artifacts')
    boundary();fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True)
    rows=[(split,rep) for split in ('validation','offline','pooled') for rep in (0,1,'mean')]
    labels=[f'{split.title()} · '+('mean of repeats' if rep=='mean' else f'repeat {rep+1}') for split,rep in rows]
    for ax,(scope,metric,title) in zip(axes,[('joint','policy','Joint: policy loss'),('policy','policy','Policy only: policy loss'),('joint','value','Joint: value loss')]):
        for i,(split,rep) in enumerate(rows):
            row=next(r for r in result['contrasts'] if r['kind']=='breadth' and r['checkpoint_step']==668 and
                     r['scope']==scope and r['metric']==metric and r['split']==split and r['repetition']==rep)
            lo,hi=row['nominal_ci95'];point=row['absolute_difference'];color='#164e63' if rep=='mean' else '#75969d'
            ax.plot([lo,hi],[i,i],color=color,lw=2);ax.scatter([point],[i],color=color,s=45 if rep=='mean' else 25,zorder=3)
        ax.axvline(0,color='#64748b',lw=1,ls='--');ax.set_title(title);ax.set_xlabel('Broad − dense loss (lower favors broad)');ax.grid(axis='x',alpha=.18)
    axes[0].set_yticks(range(len(rows)),labels);axes[0].invert_yaxis()
    fig.suptitle('Does broader game coverage improve transfer?',fontsize=18,y=.98)
    fig.text(.5,.025,'Step 668 · 64 already-opened held-out games · 20,000 paired whole-game resamples\nNominal 95% intervals are exploratory; this is not a playing-strength result.',ha='center',fontsize=10)
    fig.tight_layout(rect=(.01,.10,.995,.94));fig.savefig(chart1,dpi=160);plt.close(fig)
    boundary();fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    colors={('dense','joint'):'#1d4ed8',('broad','joint'):'#0f766e',('dense','policy'):'#b45309',('broad','policy'):'#a21caf'}
    for ax,(split,metric) in zip(axes.flat,[('train','policy'),('pooled','policy'),('train','value'),('pooled','value')]):
        trajectories,label=trajectory_series(result,split,metric)
        for (breadth,scope),color in colors.items():
            ax.plot(STEPS,trajectories[(breadth,scope)],marker='o',lw=2,color=color,label=f'{breadth.title()} / '+('joint' if scope=='joint' else 'policy only'))
        ax.axhline(0,color='#64748b',lw=.8,ls='--');ax.set_title(('Selected training' if split=='train' else 'Held-out, equal-game pooled')+f' · {metric}');ax.set_ylabel(label);ax.grid(alpha=.15)
    for ax in axes[1]:ax.set_xlabel('Fixed optimizer step');ax.set_xticks(STEPS)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.06))
    fig.suptitle('Fixed-checkpoint loss trajectories',fontsize=18,y=.98)
    fig.text(.5,.02,'Mean of two predetermined repetitions · negative is lower loss · training datasets differ between arms\nNo checkpoint selection, architecture change or strength claim.',ha='center',fontsize=10)
    fig.tight_layout(rect=(.01,.14,.995,.94));fig.savefig(chart2,dpi=160);plt.close(fig);boundary()
    lines=['# Saved-corpus learning diagnostic','',
        'Eight fixed fits compared game breadth and trainable scope on the existing corrected corpus. '+
        'These saved-prediction results require the separate independent study acceptance receipt.',
        '', '| Final pooled breadth comparison | Broad − dense loss | Nominal 95% interval |',
        '|---|---:|---:|']
    for scope,metric in [('joint','policy'),('policy','policy'),('joint','value')]:
        row=next(r for r in result['contrasts'] if r['kind']=='breadth' and r['checkpoint_step']==668 and
                 r['scope']==scope and r['metric']==metric and r['split']=='pooled' and r['repetition']=='mean')
        lo,hi=row['nominal_ci95'];lines.append(f"| {scope.title()} / {metric} | {row['absolute_difference']:+.6f} | {lo:+.6f} to {hi:+.6f} |")
    lines += ['',f'![Breadth comparisons]({chart1.as_posix()})','',f'![Loss trajectories]({chart2.as_posix()})','',
        'Validation and offline splits, both sampling repetitions and all secondary contrasts remain in analysis.json. '+
        'All intervals resample whole games with common indices across fits and checkpoints. No multiplicity-adjusted significance is claimed.',
        '', '## What this result establishes','',
        'The diagnostic measures transfer to already-opened saved positions at equal record and update budgets. '+
        'It does not establish stronger play, benefit beyond 192 source games, a causal gradient-conflict mechanism or a need for a larger network.',
        '', '## The next step','',
        'Review both repetitions and held-out splits before selecting one next research question. '+
        'No checkpoint is selected or promoted, and no new collection or 400-game assessment launches automatically.',
        '', '## The broader picture','',
        'The aim is to learn a stronger same-sized player from verified search guidance. '+
        'This study narrows the choice between broader game coverage and changes to training scope; '+
        'fresh independent paired games are still required to demonstrate a strength improvement. Original EF8A remains the reference.',
        '',f"Analysis SHA256: `{analysis_item['sha256']}`."]
    with document.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines)+'\n')
    return dict(charts=[c.artifact(chart1),c.artifact(chart2)],report=c.artifact(document))
