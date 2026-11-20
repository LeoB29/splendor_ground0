"""Saved online losses, clipping and final tensor displacement; no model loading."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

import numpy as np

from . import common as c


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def expected_shapes():
    result = {'identity': [31, 256], 'project.weight': [256, 48], 'project.bias': [256]}
    for i in range(6):
        for name, shape in {
            'self_attn.in_proj_weight': [768, 256], 'self_attn.in_proj_bias': [768],
            'self_attn.out_proj.weight': [256, 256], 'self_attn.out_proj.bias': [256],
            'linear1.weight': [1024, 256], 'linear1.bias': [1024],
            'linear2.weight': [256, 1024], 'linear2.bias': [256],
            'norm1.weight': [256], 'norm1.bias': [256],
            'norm2.weight': [256], 'norm2.bias': [256],
        }.items():
            result[f'blocks.layers.{i}.{name}'] = shape
    result.update({'norm.weight': [256], 'norm.bias': [256], 'policy.weight': [81, 256],
                   'policy.bias': [81], 'value.weight': [2, 256], 'value.bias': [2]})
    return result


def epoch_order(n, seed, epoch):
    return sorted(range(n), key=lambda i: (digest([seed, epoch, i]), i))


def numeric(value, label, *, nonnegative=True):
    c.require(type(value) in (int, float) and math.isfinite(value) and
              (not nonnegative or value >= 0), 'Finite '+label)
    return float(value)


def summarize_steps(rows, *, clip_cap):
    n = sum(r['batch_examples'] for r in rows)
    norms = [r['gradient_norm_before_clip'] for r in rows]
    scale = [min(1.0, clip_cap/(x+1e-6)) for x in norms]
    return {
        'step_first': rows[0]['step'], 'step_last': rows[-1]['step'],
        'steps': len(rows), 'example_presentations': n,
        'online_game_weighted_combined_loss': 32.0*math.fsum(r['loss'] for r in rows)/n,
        'online_root_weighted_policy_loss': math.fsum(r['policy_mean_unweighted']*r['batch_examples'] for r in rows)/n,
        'online_root_weighted_value_loss': math.fsum(r['value_mean_unweighted']*r['batch_examples'] for r in rows)/n,
        'gradient_norm_before_clip': {
            'minimum': min(norms), 'median': statistics.median(norms),
            'mean': math.fsum(norms)/len(norms), 'maximum': max(norms),
        },
        'clipped_steps': sum(x > clip_cap for x in norms),
        'clipped_fraction': sum(x > clip_cap for x in norms)/len(norms),
        'implied_uniform_gradient_scale': {
            'minimum': min(scale), 'median': statistics.median(scale), 'maximum': max(scale),
            'formula': 'min(1, clip_cap / (recorded_preclip_global_norm + 1e-6))',
            'not_parameter_update_scale': True,
        },
        'recorded_train_step_seconds': math.fsum(r['seconds'] for r in rows),
    }


def extract_steps(folder, plan, fit_result, *, read, check):
    """Validate all fixed orders and scalar records without replaying training."""
    recipe = plan['recipe']
    n = fit_result['training_roots']
    c.require(plan['epochs'] == recipe['epochs'] == 2 and recipe['batch_size'] == 32 and
              type(n) is int and n > 0, 'Fixed two-epoch batch32 plan')
    per_epoch = math.ceil(n/32)
    c.require(fit_result['optimizer_steps'] == 2*per_epoch, 'Complete fixed optimizer count')
    counts = fit_result['roots_per_game']
    c.require(len(counts) == fit_result['training_games'] == plan['training_game_count'] and
              all(type(x) is int and x > 0 for x in counts.values()) and sum(counts.values()) == n,
              'Complete nonempty training games and roots')
    c.require(sorted(p.name for p in (folder/'steps').glob('*.json')) ==
              [f'{i:06d}.json' for i in range(2*per_epoch)], 'Exact step inventory')
    c.require(sorted(p.name for p in (folder/'orders').glob('*.json')) == ['000.json', '001.json'],
              'Exact epoch inventory')
    compact, epochs, orders = [], [], []
    scalar_keys = ('loss', 'policy_mean_unweighted', 'value_mean_unweighted',
                   'gradient_norm_before_clip', 'seconds')
    for epoch in range(2):
        check()
        order = epoch_order(n, plan['seed'], epoch)
        order_hash = digest(order)
        c.require(read(folder/'orders'/f'{epoch:03d}.json') ==
                  {'epoch': epoch, 'indices': order, 'sha256': order_hash}, 'Exact deterministic epoch order')
        orders.append(order_hash)
        for offset in range(0, n, 32):
            check()
            step_id = len(compact)
            row = read(folder/'steps'/f'{step_id:06d}.json')
            indices = order[offset:offset+32]
            c.require(row['step'] == step_id and row['epoch'] == epoch and
                      row['example_indices'] == indices and row['batch_examples'] == len(indices) and
                      row['batch_denominator'] == 32 and row['forward_backward_calls'] == 1,
                      'Exact actual batch identity and denominator')
            c.require(len(row['outputs83']) == len(indices) and all(len(x) == 83 for x in row['outputs83']),
                      'Complete actual output inventory')
            c.require(all(type(x) in (float, int) and math.isfinite(x)
                          for output in row['outputs83'] for x in output), 'Finite saved actual outputs')
            c.require(isinstance(row['batch_source_sha256'], str) and
                      len(row['batch_source_sha256']) == 64 and
                      all(x in '0123456789abcdef' for x in row['batch_source_sha256']), 'Saved batch binding')
            record = {k: row[k] for k in ('step', 'epoch', 'batch_examples', 'batch_denominator', 'batch_source_sha256')}
            record.update({k: numeric(row[k], k) for k in scalar_keys})
            compact.append(record)
        epochs.append({'epoch': epoch, **summarize_steps(compact[epoch*per_epoch:], clip_cap=recipe['gradient_norm_cap'])})
    c.require(orders == fit_result['epoch_order_sha256'], 'Fit-result complete epoch binding')
    return {
        'optimizer_steps': len(compact), 'training_roots': n, 'training_games': len(counts),
        'epoch_orders': orders, 'epochs': epochs,
        'blocks_of_50_steps': [summarize_steps(compact[i:i+50], clip_cap=recipe['gradient_norm_cap'])
                               for i in range(0, len(compact), 50)],
        'overall': summarize_steps(compact, clip_cap=recipe['gradient_norm_cap']),
        'steps': compact,
        'loss_semantics': {
            'when': 'saved pre-update outputs at a changing checkpoint',
            'epoch_weighted_combined': 'sum(batch_loss * 32) / roots_per_epoch',
            'block_weighted_combined': 'sum(batch_loss * 32) / example_presentations_in_block',
            'policy_and_value': 'root-weighted means of saved unweighted batch means',
            'final_short_batch': 'four roots in the actual pilot; denominator stays32',
            'fixed_checkpoint_epoch_evaluations': False,
            'final_training_set_loss_available': False,
            'scalar_losses_reauthenticated_from_accepted_evidence': True,
            'new_forward_backward_or_optimizer_calls': 0,
        },
    }


def tensor_group(name):
    return 'policy' if name.startswith('policy.') else 'value' if name.startswith('value.') else 'shared'


def archive_arrays(path, entries, expected, *, check):
    """Only ZIP storages; data.pkl is never read or executed."""
    c.require([x['name'] for x in entries] == list(expected), 'Exact tensor names/order')
    c.require(len({x['zip_member'] for x in entries}) == len(entries), 'Distinct tensor storage inventory')
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        c.require(len(members) == len(set(members)), 'No duplicate ZIP members')
        roots = {x.split('/')[0] for x in members}
        c.require(len(roots) == 1, 'Single archive root')
        root = next(iter(roots))
        c.require(archive.read(root+'/byteorder') == b'little', 'Little-endian float32 checkpoint')
        c.require({x for x in members if x.startswith(root+'/data/')} == {x['zip_member'] for x in entries},
                  'No missing/extra checkpoint storages')
        for entry in entries:
            check()
            shape = expected[entry['name']]
            c.require(entry['shape'] == shape and entry['dtype'] == 'torch.float32' and
                      entry['numel'] == math.prod(shape) and entry['bytes'] == math.prod(shape)*4,
                      'Exact float32 tensor shape and length')
            c.require(archive.getinfo(entry['zip_member']).file_size == entry['bytes'], 'Bounded storage before read')
            raw = archive.read(entry['zip_member'])
            c.require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Authenticated tensor storage')
            array = np.frombuffer(raw, dtype='<f4')
            c.require(np.isfinite(array).all(), 'Finite checkpoint tensor')
            yield entry, raw, array


def parent_entries(path, expected, *, check):
    result = []
    with zipfile.ZipFile(path) as archive:
        roots = {x.split('/')[0] for x in archive.namelist()}
        c.require(len(roots) == 1, 'Single original archive root')
        root = next(iter(roots))
        for i, (name, shape) in enumerate(expected.items()):
            check()
            member = root+'/data/'+str(i)
            c.require(archive.getinfo(member).file_size == 4*math.prod(shape), 'Bounded original tensor extent')
            raw = archive.read(member)
            result.append(dict(name=name, shape=shape, numel=math.prod(shape), bytes=len(raw),
                               dtype='torch.float32', zip_member=member, sha256=hashlib.sha256(raw).hexdigest()))
    return result


def norms_finish(value):
    n = value['parameters']
    parent_l2 = math.sqrt(value.pop('parent_squares'))
    candidate_l2 = math.sqrt(value.pop('candidate_squares'))
    delta_l2 = math.sqrt(value.pop('delta_squares'))
    absolute_sum = value.pop('delta_absolute_sum')
    return {**value, 'parent_l2': parent_l2, 'candidate_l2': candidate_l2, 'delta_l2': delta_l2,
            'delta_over_parent_l2': delta_l2/parent_l2 if parent_l2 else None,
            'rms_delta': delta_l2/math.sqrt(n), 'mean_absolute_delta': absolute_sum/n,
            'changed_fraction': value['changed_parameters']/n}


def checkpoint_deltas(parent_path, candidate_path, candidate_entries, expected, before, after, *, check):
    parent = list(archive_arrays(parent_path, parent_entries(parent_path, expected, check=check), expected, check=check))
    candidate = list(archive_arrays(candidate_path, candidate_entries, expected, check=check))
    results, hashes = [], {}
    for label, tensors, actual in [('parent', parent, before), ('candidate', candidate, after)]:
        groups = {k: hashlib.sha256() for k in ('all', 'shared', 'policy', 'value')}
        for entry, raw, _ in sorted(tensors, key=lambda x: x[0]['name']):
            check()
            prefix = canonical([entry['name'], entry['shape'], entry['dtype']]).encode()+b'\0'
            for group in ('all', tensor_group(entry['name'])):
                groups[group].update(prefix); groups[group].update(raw)
        hashes[label] = {k: v.hexdigest() for k, v in groups.items()}
        c.require(hashes[label] == actual, 'Original/final complete and grouped tensor authentication')
    aggregates = {k: dict(parameters=0, changed_parameters=0, parent_squares=0., candidate_squares=0.,
                          delta_squares=0., delta_absolute_sum=0., maximum_absolute_delta=0.)
                  for k in ('all', 'shared', 'policy', 'value')}
    for (entry, _, p), (other, _, q) in zip(parent, candidate, strict=True):
        check()
        c.require(entry['name'] == other['name'], 'Same named tensor comparison')
        p64, q64 = p.astype(np.float64), q.astype(np.float64)
        delta = q64-p64
        values = dict(parameters=len(p), changed_parameters=int(np.count_nonzero(delta)),
                      parent_squares=float(np.dot(p64, p64)), candidate_squares=float(np.dot(q64, q64)),
                      delta_squares=float(np.dot(delta, delta)), delta_absolute_sum=float(np.abs(delta).sum()),
                      maximum_absolute_delta=float(np.abs(delta).max()))
        results.append({'name': entry['name'], 'shape': entry['shape'], 'group': tensor_group(entry['name']),
                        **norms_finish(dict(values))})
        for group in ('all', tensor_group(entry['name'])):
            for key, value in values.items():
                aggregates[group][key] = max(aggregates[group][key], value) if key == 'maximum_absolute_delta' else aggregates[group][key]+value
    return {'tensors': results, 'groups': {k: norms_finish(v) for k, v in aggregates.items()},
            'authenticated_group_hashes': hashes, 'raw_zip_storages_only': True,
            'model_loaded': False, 'pickle_read': False, 'accumulation_dtype': 'float64',
            'comparison': 'single final checkpoint minus original parent; no intermediate update norms',
            'zero_parent_norm_relative_delta': 'null rather than division by zero'}


def run(context=None):
    ctx = context or c.Context()
    folder = ctx.source/'fit'
    freeze = ctx.read(ctx.source/'freeze.json')
    plan = ctx.read(ctx.source/'training_plan.json')
    fit_result = ctx.read(folder/'fit_result.json')
    c.require(ctx.bind(ctx.source/'training_plan.json') == freeze['training_plan'], 'Frozen recipe file')
    source_path = c.ROOT/'tools/splendorust_selfplay_pilot_resume2_20261103/training.py'
    source_pin = next(x for x in freeze['inputs'] if x['path'] == source_path.relative_to(c.ROOT).as_posix())
    c.require(ctx.bind(source_path) == source_pin, 'Frozen actual training implementation')
    tree = ast.parse(source_path.read_text())
    recipe_node = next(x.value for x in tree.body if isinstance(x, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == 'RECIPE' for t in x.targets))
    c.require(ast.literal_eval(recipe_node) == plan['recipe'], 'Saved recipe equals actual source recipe')
    c.require(fit_result['training_plan_sha256'] == digest(plan) and fit_result['status'] == 'fixed_two_epoch_fit_passed' and
              not fit_result['qualification_only'] and fit_result['optimizer_steps'] == 668 and
              fit_result['training_roots'] == 10660 and fit_result['training_games'] == 192,
              'Accepted exact pilot fit identity')
    trajectory = extract_steps(folder, plan, fit_result, read=ctx.read, check=ctx.check)
    manifest = ctx.read(folder/'checkpoint_manifest.json')
    c.require(ctx.bind(folder/'checkpoint_manifest.json') == fit_result['checkpoint_manifest'], 'Fit selected checkpoint manifest')
    for key in ('checkpoint', 'tensor_manifest', 'model_definition'):
        c.require(ctx.bind(c.ROOT/manifest[key]['path']) == manifest[key], 'Candidate '+key)
    tensor_doc = ctx.read(c.ROOT/manifest['tensor_manifest']['path'])
    c.require(tensor_doc['checkpoint'] == manifest['checkpoint'] and tensor_doc['weights_sha256'] ==
              manifest['metadata']['weights_sha256'] == fit_result['candidate_weights_sha256'], 'Complete candidate identity')
    parent_item = freeze['historical_parent_checkpoint']
    c.require(ctx.bind(c.ROOT/parent_item['path']) == parent_item and
              manifest['metadata']['historical_warm_start_sha256'] == parent_item['sha256'], 'Original immutable parent')
    shapes = expected_shapes()
    c.require(sum(math.prod(s) for s in shapes.values()) == manifest['metadata']['parameter_count'] == 4780883,
              'Unchanged full parameter inventory')
    deltas = checkpoint_deltas(c.ROOT/parent_item['path'], c.ROOT/manifest['checkpoint']['path'],
                              tensor_doc['tensors'], shapes,
                              {'all': fit_result['parent_weights_sha256'], **fit_result['before_group_sha256']},
                              {'all': fit_result['candidate_weights_sha256'], **fit_result['after_group_sha256']}, check=ctx.check)
    return ctx.finish('trajectory.json', {
        'status': 'saved_trajectory_and_final_weight_delta_verified', 'recipe': plan,
        'trajectory': trajectory, 'checkpoint_displacement': deltas,
        'limitations': [
            'Online minibatch losses are measured before each update at changing checkpoints.',
            'There are no fixed-checkpoint training-set evaluations at epoch boundaries or at the final checkpoint.',
            'No per-head gradient vectors, gradient directions, optimizer moments or per-step parameter deltas were saved.',
            'Clipping frequency alone does not identify a bad learning rate or policy/value interference.',
            'Final weight displacement does not measure playing strength or guarantee output similarity.',
            'Short final batches retain the frozen32 denominator; summaries account for that explicitly.',
        ],
    })


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
