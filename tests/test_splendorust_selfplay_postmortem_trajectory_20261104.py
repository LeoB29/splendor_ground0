import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

from tools.splendorust_selfplay_postmortem_20261104 import trajectory as t


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


def step_fixture(tmp_path, n=36):
    plan = {'epochs': 2, 'seed': 123, 'training_game_count': 2,
            'recipe': {'epochs': 2, 'batch_size': 32, 'gradient_norm_cap': 1.}}
    result = {'training_roots': n, 'training_games': 2, 'roots_per_game': {'0': n//2, '1': n-n//2},
              'optimizer_steps': 2*math.ceil(n/32), 'epoch_order_sha256': []}
    step = 0
    for epoch in range(2):
        order = t.epoch_order(n, 123, epoch); sha = t.digest(order)
        result['epoch_order_sha256'].append(sha)
        write(tmp_path/'orders'/f'{epoch:03d}.json', {'epoch': epoch, 'indices': order, 'sha256': sha})
        for offset in range(0, n, 32):
            indices = order[offset:offset+32]
            write(tmp_path/'steps'/f'{step:06d}.json', {
                'step': step, 'epoch': epoch, 'example_indices': indices, 'batch_examples': len(indices),
                'batch_denominator': 32, 'forward_backward_calls': 1,
                'batch_source_sha256': 'a'*64, 'outputs83': [[0.]*83 for _ in indices],
                'loss': 3*len(indices)/32, 'policy_mean_unweighted': 1., 'value_mean_unweighted': 2.,
                'gradient_norm_before_clip': 2. if offset == 0 else .5, 'seconds': .2,
            }); step += 1
    return plan, result


def extract(tmp_path, plan, result):
    return t.extract_steps(tmp_path, plan, result, read=read, check=lambda: None)


def test_short_batch_uses_actual_fixed_denominator_and_root_weights(tmp_path):
    plan, result = step_fixture(tmp_path)
    out = extract(tmp_path, plan, result)
    assert out['optimizer_steps'] == 4
    assert len(out['steps']) == 4
    for epoch in out['epochs']:
        assert epoch['online_game_weighted_combined_loss'] == 3.
        assert epoch['online_root_weighted_policy_loss'] == 1.
        assert epoch['online_root_weighted_value_loss'] == 2.
        assert epoch['clipped_steps'] == 1
        assert epoch['example_presentations'] == 36
    assert out['loss_semantics']['fixed_checkpoint_epoch_evaluations'] is False
    assert out['loss_semantics']['new_forward_backward_or_optimizer_calls'] == 0


def test_fifty_step_blocks_preserve_every_step_and_short_batches(tmp_path):
    plan, result = step_fixture(tmp_path, n=1602)
    out = extract(tmp_path, plan, result)
    assert [(x['step_first'], x['step_last']) for x in out['blocks_of_50_steps']] == [(0, 49), (50, 99), (100, 101)]
    assert sum(x['example_presentations'] for x in out['blocks_of_50_steps']) == 3204
    assert all(x['online_game_weighted_combined_loss'] == 3 for x in out['blocks_of_50_steps'])


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'order', 'step', 'epoch', 'denominator', 'batch', 'indices',
                                    'outputs', 'nan', 'norm', 'source', 'forward_calls', 'epoch_digest'])
def test_incomplete_or_tampered_step_records_rejected(tmp_path, mutation):
    plan, result = step_fixture(tmp_path)
    p = tmp_path/'steps/000000.json'; row = read(p)
    if mutation == 'missing': p.unlink()
    elif mutation == 'extra': write(tmp_path/'steps/999999.json', row)
    elif mutation == 'order':
        p = tmp_path/'orders/000.json'; row = read(p); row['indices'].reverse(); write(p, row)
    elif mutation == 'epoch_digest': result['epoch_order_sha256'][0] = 'b'*64
    else:
        changes = {'step': ('step', 1), 'epoch': ('epoch', 1), 'denominator': ('batch_denominator', 4),
                   'batch': ('batch_examples', 31), 'indices': ('example_indices', [0]*32),
                   'outputs': ('outputs83', [[0.]*82]*32), 'nan': ('loss', float('nan')),
                   'norm': ('gradient_norm_before_clip', -1), 'source': ('batch_source_sha256', 'invalid'),
                   'forward_calls': ('forward_backward_calls', 2)}
        key, value = changes[mutation]; row[key] = value; write(p, row)
    with pytest.raises(ValueError): extract(tmp_path, plan, result)


def test_boundary_checked_for_each_saved_step(tmp_path):
    plan, result = step_fixture(tmp_path)
    count = 0
    def guard():
        nonlocal count
        count += 1
        if count == 4: raise TimeoutError('bounded')
    with pytest.raises(TimeoutError): t.extract_steps(tmp_path, plan, result, read=read, check=guard)


def archive(path, values, *, extras=None, endian=b'little'):
    entries = []
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('archive/byteorder', endian)
        # Intentionally invalid pickle; a reader must never unpickle this.
        z.writestr('archive/data.pkl', b'not-a-pickle-do-not-execute')
        for i, (name, vals) in enumerate(values.items()):
            raw = np.asarray(vals, dtype='<f4').tobytes()
            member = f'archive/data/{i}'
            z.writestr(member, raw)
            entries.append(dict(name=name, shape=[len(vals)], dtype='torch.float32', numel=len(vals),
                                bytes=len(raw), zip_member=member, sha256=hashlib.sha256(raw).hexdigest()))
        for key, value in (extras or {}).items(): z.writestr(key, value)
    return entries


def group_hashes(entries, values):
    hashes = {k: hashlib.sha256() for k in ('all', 'shared', 'policy', 'value')}
    for entry in sorted(entries, key=lambda x: x['name']):
        name = entry['name']; raw = np.asarray(values[name], dtype='<f4').tobytes()
        for group in ('all', t.tensor_group(name)):
            hashes[group].update(t.canonical([name, entry['shape'], 'torch.float32']).encode()+b'\0')
            hashes[group].update(raw)
    return {k: x.hexdigest() for k, x in hashes.items()}


def delta_fixture(tmp_path):
    p = {'identity': [3., 4.], 'policy.weight': [0., 2.], 'value.bias': [0., 0.]}
    q = {'identity': [6., 8.], 'policy.weight': [0., 3.], 'value.bias': [1., -1.]}
    parent, candidate = tmp_path/'parent.pt', tmp_path/'candidate.pt'
    pe, ce = archive(parent, p), archive(candidate, q)
    shapes = {k: [len(v)] for k, v in p.items()}
    return parent, candidate, ce, shapes, group_hashes(pe, p), group_hashes(ce, q), q


def test_final_delta_group_math_zero_parent_and_no_pickle_or_torch(tmp_path):
    parent, candidate, entries, shapes, before, after, _ = delta_fixture(tmp_path)
    out = t.checkpoint_deltas(parent, candidate, entries, shapes, before, after, check=lambda: None)
    assert out['groups']['shared']['delta_l2'] == 5
    assert out['groups']['shared']['delta_over_parent_l2'] == 1
    assert out['groups']['all']['delta_l2'] == pytest.approx(math.sqrt(28))
    assert out['groups']['all']['parameters'] == 6
    assert out['groups']['all']['changed_parameters'] == 5
    assert out['groups']['value']['delta_over_parent_l2'] is None
    assert out['groups']['value']['parent_l2'] == 0
    assert not out['pickle_read'] and not out['model_loaded']
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('mutation', ['storage_hash', 'shape', 'group_hash', 'extra_storage', 'big_endian', 'nonfinite'])
def test_tensor_manifest_or_bytes_mismatch_rejected(tmp_path, mutation):
    parent, candidate, entries, shapes, before, after, q = delta_fixture(tmp_path)
    if mutation == 'storage_hash': entries[0]['sha256'] = '0'*64
    elif mutation == 'shape': entries[0]['shape'] = [3]
    elif mutation == 'group_hash': after['shared'] = '0'*64
    elif mutation == 'extra_storage': archive(candidate, q, extras={'archive/data/999': b'1234'})
    elif mutation == 'big_endian': archive(candidate, q, endian=b'big')
    elif mutation == 'nonfinite':
        q['identity'][0] = float('inf'); entries = archive(candidate, q); after = group_hashes(entries, q)
    with pytest.raises(ValueError):
        t.checkpoint_deltas(parent, candidate, entries, shapes, before, after, check=lambda: None)


def test_duplicate_zip_members_rejected(tmp_path):
    parent, candidate, entries, shapes, _, _, _ = delta_fixture(tmp_path)
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(candidate, 'a') as z: z.writestr('archive/byteorder', b'little')
    with pytest.raises(ValueError, match='duplicate'):
        list(t.archive_arrays(candidate, entries, shapes, check=lambda: None))


def test_expected_full_architecture_inventory():
    shapes = t.expected_shapes()
    assert len(shapes) == 81
    assert sum(math.prod(x) for x in shapes.values()) == 4780883
    assert shapes['policy.weight'] == [81, 256]
    assert shapes['value.weight'] == [2, 256]


def test_clipping_summary_does_not_confuse_gradient_and_parameter_scales():
    rows = [dict(step=i, batch_examples=32, loss=1., policy_mean_unweighted=.5,
                 value_mean_unweighted=.5, gradient_norm_before_clip=x, seconds=1.)
            for i, x in enumerate([.5, 1., 2.])]
    out = t.summarize_steps(rows, clip_cap=1.)
    assert out['clipped_steps'] == 1
    assert out['implied_uniform_gradient_scale']['not_parameter_update_scale'] is True
    assert out['implied_uniform_gradient_scale']['minimum'] == pytest.approx(1/(2+1e-6))
