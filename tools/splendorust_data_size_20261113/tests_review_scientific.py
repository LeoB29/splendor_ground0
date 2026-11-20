"""Independent model-free review regressions for fit/evaluator provenance.

These tests exercise new code written by other agents; no network is imported
or executed. Artifact IO is isolated only for deliberate corruption fixtures.
"""
from copy import deepcopy
import math
import struct
import sys

import pytest
from . import fit, offline as o


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


@pytest.mark.parametrize('count', [1, 10, 31, 32])
def test_actual_partial_loss_uses_float32_targets_and_actual_denominator(count):
    examples = []
    outputs = []
    policy_losses = []
    value_losses = []
    for i in range(count):
        target = (1. + .837 * (.1 if i % 2 else -.3)) / 1.837
        logit = float(i % 3 - 1)
        prediction = _f32((i % 5 - 2) / 8)
        examples.append(dict(game_index=i, ply=0, split='train',
            root_input_sha256='1' * 64, row_sha256='2' * 64,
            mask=[True, True] + [False] * 79,
            policy=[.25, .75] + [0.] * 79, value=[target, -target]))
        outputs.append([logit, 0.] + [1000.] * 79 + [prediction, -prediction])
        policy_losses.append(math.log1p(math.exp(logit)) - .25 * logit)
        value_losses.append((prediction - _f32(target)) ** 2)
    actual = fit.saved_batch_statistics(examples, outputs)
    assert actual['policy_mean_unweighted'] == pytest.approx(math.fsum(policy_losses) / count, abs=1e-14)
    assert actual['value_mean_unweighted'] == math.fsum(value_losses) / count
    assert actual['loss'] == pytest.approx(math.fsum(a + b for a, b in zip(policy_losses, value_losses)) / count)
    assert 'torch' not in sys.modules


def _fixed_binding_fixture(monkeypatch):
    """Exercise production fixed_bindings with valid small IO-only stand-ins."""
    selected = dict(subsets={'256': {'one_pass_steps': 445}, '512': {'one_pass_steps': 889}})
    original = {'source': 'accepted-original-commitment'}
    study = {'source': 'new-commitment'}
    artifact = lambda path: {'path': str(path), 'bytes': 1, 'sha256': 'a' * 64}
    commit = dict(status='all_data_size_checkpoints_committed', checkpoint_selection=False,
        selection_sha256=o.c.digest(selected), source_selection=artifact(o.c.SOURCE_SELECTION),
        original_commitment=original, models=[])
    expected = {}
    endpoint_map = {}
    manifests = {}
    for size in (256, 512):
        one = selected['subsets'][str(size)]['one_pass_steps']
        batches = [[0] * 32 for _ in range(1782)]
        batches[one - 1] = [0] * 10
        for rep in (0, 1):
            expected[size, rep] = (dict(game_count=size, repetition=rep,
                selection_sha256=o.c.digest(selected), schedule_sha256=o.c.digest([size, rep]),
                value_target_sources_sha256=o.c.digest(['subset-targets', size]), qualification_only=False),
                dict(batches=batches))
    for name in o.FRESH_IDS:
        size_text, rep_text, endpoint = name.split('_')
        size, rep = int(size_text[1:]), int(rep_text[1:])
        step = selected['subsets'][str(size)]['one_pass_steps'] if endpoint == 'onepass' else 1782
        meta, schedule = expected[size, rep]
        item = artifact(name)
        endpoint_map[name] = item
        manifests[name] = dict(metadata={**meta, 'endpoint': endpoint, 'checkpoint_step': step,
            'record_presentations': sum(map(len, schedule['batches'][:step])), 'weights_sha256': 'b' * 64},
            checkpoint=artifact(name + '/weights'), tensor_manifest=artifact(name + '/tensor-manifest'))
        commit['models'].append(dict(model_id=name, game_count=size, repetition=rep, endpoint=endpoint,
            checkpoint_step=step, checkpoint=item, weights_sha256='b' * 64))
    def read(item, boundary=lambda: None):
        return commit if item == study else manifests[item['path']]
    monkeypatch.setattr(o, '_read', read)
    monkeypatch.setattr(o, 'source_selection', lambda: {})
    monkeypatch.setattr(o.previous_fit, 'validate_commitment', lambda *a, **k: {})
    monkeypatch.setattr(o.previous, 'prediction_bindings', lambda *a, **k: {old: {'reused': old} for old in o.OLD_IDS.values()})
    monkeypatch.setattr(o, '_expected_fit_bindings', lambda *a, **k: expected)
    monkeypatch.setattr(o, '_committed_fit_endpoints', lambda *a, **k: endpoint_map)
    monkeypatch.setattr(o.c, 'artifact', artifact)
    monkeypatch.setattr(o.c, 'verify', lambda *a, **k: None)
    # Metadata shape has its own full suite. Here isolate the cross-artifact
    # contract that previously accepted other well-formed selection identities.
    monkeypatch.setattr(fit, 'validate_checkpoint_metadata', lambda meta: meta)
    return selected, original, study, manifests, endpoint_map, commit


def test_all_eight_endpoints_bind_before_prediction_and_original_three_are_reused(monkeypatch):
    selected, original, study, _, _, _ = _fixed_binding_fixture(monkeypatch)
    result = o.fixed_bindings(selected, original, study)
    assert set(result) == set(o.MODEL_IDS)
    for name, old_name in o.OLD_IDS.items():
        assert result[name] == {'reused': old_name}
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('field', ['selection_sha256', 'schedule_sha256', 'value_target_sources_sha256'])
def test_well_formed_foreign_training_lineage_is_rejected(monkeypatch, field):
    selected, original, study, manifests, _, _ = _fixed_binding_fixture(monkeypatch)
    manifests['g512_r1_matched']['metadata'][field] = 'c' * 64
    with pytest.raises(ValueError, match='Exact selected subset'):
        o.fixed_bindings(selected, original, study)


def test_commitment_cannot_substitute_a_different_verified_endpoint(monkeypatch):
    selected, original, study, _, endpoint_map, _ = _fixed_binding_fixture(monkeypatch)
    endpoint_map['g256_r0_onepass'] = deepcopy(endpoint_map['g256_r0_matched'])
    with pytest.raises(ValueError, match='verified fit endpoint'):
        o.fixed_bindings(selected, original, study)


def test_commitment_rejects_padded_partial_exposure_count(monkeypatch):
    selected, original, study, manifests, _, _ = _fixed_binding_fixture(monkeypatch)
    manifests['g256_r0_onepass']['metadata']['record_presentations'] += 22
    with pytest.raises(ValueError, match='example presentations'):
        o.fixed_bindings(selected, original, study)


def test_commitment_requires_both_seeds_and_both_endpoints(monkeypatch):
    selected, original, study, _, _, commit = _fixed_binding_fixture(monkeypatch)
    commit['models'].pop()
    with pytest.raises(ValueError, match='Exact eight'):
        o.fixed_bindings(selected, original, study)


def test_independent_arithmetic_rejects_per_position_reweighting_of_training_panel():
    from .tests_offline import fixture, selection
    bundles, labels = fixture()
    # A one-root game differs strongly; the other game has two roots. Equal
    # positions and equal games give observably different descriptive means.
    for records in bundles['g256_r0_matched'].values():
        records[-1]['output83'][81:] = [1., -1.]
    report = o.analyze_predictions(bundles, labels, selection(), draws=11, seed=29, strict=False)
    rows = report['game_summaries']['g256_r0_matched']['train']
    equal_game = math.fsum(row['outcome_mse'] for row in rows) / len(rows)
    equal_root = math.fsum(row['roots'] * row['outcome_mse'] for row in rows) / sum(row['roots'] for row in rows)
    assert equal_game != equal_root
    report['estimates']['g256_r0_matched']['train']['outcome_mse']['mean'] = equal_root
    with pytest.raises(ValueError, match='independent metric mean'):
        o.verify_arithmetic(bundles, labels, selection(), report, strict=False)
