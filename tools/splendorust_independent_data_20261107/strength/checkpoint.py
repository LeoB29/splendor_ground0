"""Exact state-only import of the two committed broad final96 joint checkpoints.

Preparation is ZIP/JSON/byte work only: no Torch import, model construction,
optimizer, RNG restore or forward. Live workers construct the unchanged model
and copy those exact float32 bytes into its state. Original evidence is read-only.
"""
from __future__ import annotations

from array import array
import hashlib
import math
import os
from pathlib import Path
import sys
import zipfile

from . import common as c
from tools.splendorust_selfplay_pilot_resume2_20261103 import training as architecture
from tools.splendorust_selfplay_pilot_resume2_20261103.assessment import checkpoint as parent_export
from tools.future_training_rules_20261021 import RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT

MAGIC = b'SPENTY01'
SCHEMA = 'independent_data_strength_state_only_checkpoint_v1'
TENSOR_SCHEMA = 'independent_data_strength_state_only_tensors_v1'
MODEL_BIN = parent_export.MODEL_BIN
SOURCE_STUDY = 'splendorust_independent_data_20261107_v1'


def commit_descriptor():
    return c.artifact(c.STUDY_OUT / 'all_fits_committed.json')


def committed_fits(*, boundary=lambda: None):
    from .. import fit
    descriptor = commit_descriptor()
    c.verify([descriptor]); value = c.read(c.ROOT / descriptor['path'])
    c.require(value.get('schema') == 'independent_data_all_fits_committed_v1' and
              value.get('study_id') == SOURCE_STUDY, 'This new study commitment only')
    c.verify([value['freeze'], value['selection'], *c.parent.nested_artifacts(value['source_receipt'])], boundary=boundary)
    c.require(c.canonical(value['freeze']) == c.canonical(c.artifact(c.STUDY_OUT / 'freeze.json')) and
              c.canonical(value['selection']) == c.canonical(c.artifact(c.STUDY_OUT / 'selection.json')),
              'Commitment follows this original freeze and fixed source selection')
    selection = c.read(c.ROOT / value['selection']['path'])
    checked = fit.validate_commitment(selection, descriptor, boundary=boundary)
    c.require(c.canonical(checked) == c.canonical(value), 'Full committed final model inventory')
    for row in value['fits']:
        c.verify([row['actual_exit']], boundary=boundary)
        actual = c.read(c.ROOT / row['actual_exit']['path'])
        c.require(actual['worker_exit_code'] == 0 and actual['error'] is None and
                  actual['cleanup_errors'] == [], 'Every committed fitting owner actually exited cleanly')
    return value


def repetition(value=None):
    value = c.REPETITION if value is None else value
    c.require(type(value) is int and value in (0, 1), 'Exact fixed repetition integer')
    return value


def source_descriptor(rep=None, *, boundary=lambda: None):
    rep = repetition(rep)
    commit = committed_fits(boundary=boundary)
    rows = [row for row in commit['fits'] if row['spec']['fit_id'] == f'r{rep}_broad_joint']
    c.require(len(rows) == 1, 'One prospectively nominated broad final checkpoint')
    descriptor = rows[0]['final_checkpoint']
    path = c.STUDY_OUT / 'fits' / f'r{rep}_broad_joint' / 'worker/checkpoints/096/checkpoint_manifest.json'
    c.require(c.canonical(descriptor) == c.canonical(c.artifact(path)), 'Exact fixed final96 path and committed hash')
    return descriptor


def _identity(metadata, rep):
    c.require(type(metadata) is dict, 'Source metadata object')
    exact = {'ruleset_id': RULESET_ID, 'catalogue_sha256': CATALOGUE_SHA256,
             'turn_contract': TURN_CONTRACT, 'study_id': SOURCE_STUDY,
             'historical_warm_start_sha256': architecture.PARENT_SHA256,
             'model_source_sha256': architecture.MODEL_SOURCE_SHA256,
             'parameter_count': architecture.PARAMETERS, 'checkpoint_step': 96,
             'fixed_final_step': 96, 'complete_passes': 1, 'scope': 'joint',
             'breadth': 'broad', 'fit_id': f'r{rep}_broad_joint', 'repetition': rep,
             'historical_data_relabelled': False, 'checkpoint_selection': False,
             'qualification_only': False, 'diagnostic_gradient_probes': 0,
             'record_presentations': 3072, 'training_records': 3072,
             'source_training_games': 1024, 'roots_per_game': 3}
    for key, expected in exact.items():
        actual = metadata.get(key)
        c.require(type(actual) is type(expected) and actual == expected, 'Exact accepted source identity: ' + key)
    c.require(type(metadata.get('weights_sha256')) is str and len(metadata['weights_sha256']) == 64 and
              set(metadata.get('group_sha256', {})) == {'shared', 'policy', 'value'}, 'Complete model hashes')


def _state_entries(tree):
    c.require(type(tree) is dict and set(tree) == {'dict'} and type(tree['dict']) is list,
              'Exact accepted ordered state mapping tree')
    shapes = architecture.expected_state_shapes()
    c.require(len(tree['dict']) == len(shapes), 'Complete model tensor count')
    result = []
    for (name, shape), pair in zip(shapes.items(), tree['dict'], strict=True):
        c.require(type(pair) is list and len(pair) == 2 and pair[0] == {'type': 'str', 'value': name},
                  'Exact model state ordering and names')
        item = pair[1]
        c.require(type(item) is dict and set(item) == {'tensor', 'dtype', 'shape', 'bytes', 'sha256'} and
                  item['tensor'] is True and item['dtype'] == 'torch.float32' and
                  type(item['shape']) is list and all(type(n) is int for n in item['shape']) and
                  item['shape'] == shape and type(item['bytes']) is int and
                  item['bytes'] == 4 * math.prod(shape) and type(item['sha256']) is str and len(item['sha256']) == 64,
                  'Exact source tensor extent and float32 bytes')
        result.append({'name': name, 'shape': shape, 'dtype': 'torch.float32',
                       'numel': math.prod(shape), 'bytes': item['bytes'], 'sha256': item['sha256']})
    return result


def source(rep=None, *, boundary=lambda: None):
    """Hash-authenticate the complete newly committed checkpoint and all saved trees."""
    rep = repetition(rep); descriptor = source_descriptor(rep, boundary=boundary)
    boundary(); c.verify([descriptor])
    commit = c.read(c.ROOT / commit_descriptor()['path'])
    manifest = c.read(c.ROOT / descriptor['path'])
    c.require(manifest.get('schema') == 'learning_diagnostic_checkpoint_manifest_v1' and
              set(manifest) == {'schema', 'checkpoint', 'tensor_manifest', 'metadata', 'tree_sha256'},
              'Original diagnostic manifest, never relabelled')
    _identity(manifest['metadata'], rep)
    c.require(manifest['metadata']['selection_sha256'] == commit['selection_sha256'] and
              manifest['metadata']['source_receipt_sha256'] == commit['source_receipt_sha256'],
              'Checkpoint binds accepted corpus and preselected fresh examples')
    c.verify([manifest['checkpoint'], manifest['tensor_manifest']])
    tree = c.read(c.ROOT / manifest['tensor_manifest']['path'])
    c.require(set(tree) == {'state_dict', 'optimizer', 'rng'} and
              {key: c.digest(value) for key, value in tree.items()} == manifest['tree_sha256'],
              'Complete accepted state/optimizer/RNG tree digests')
    boundary()
    return manifest, _state_entries(tree['state_dict'])


def _finite(raw):
    c.require(len(raw) % 4 == 0, 'Whole float32 words')
    bits = array('I'); bits.frombytes(raw)
    c.require(bits.itemsize == 4 and sys.byteorder == 'little', 'Little-endian 32-bit byte reader')
    c.require(all((word & 0x7f800000) != 0x7f800000 for word in bits), 'Finite model tensor bytes')


def _hashes(entries, raw_by_name):
    result = {}
    for group in ('all', 'shared', 'policy', 'value'):
        sha = hashlib.sha256()
        for entry in sorted(entries, key=lambda row: row['name']):
            name = entry['name']
            selected = group == 'all' or (name.startswith(group + '.') if group in ('policy', 'value')
                                          else not name.startswith(('policy.', 'value.')))
            if selected:
                sha.update(c.canonical([name, entry['shape'], entry['dtype']]).encode())
                sha.update(b'\0'); sha.update(raw_by_name[name])
        result[group] = sha.hexdigest()
    return result


def source_model_bytes(rep=None, *, boundary=lambda: None):
    """Every chosen ZIP storage must match its independently accepted tensor hash."""
    manifest, entries = source(rep, boundary=boundary)
    raw_by_name = {}
    with zipfile.ZipFile(c.ROOT / manifest['checkpoint']['path']) as archive:
        names = archive.namelist()
        c.require(len(names) == len(set(names)), 'No duplicate source archive members')
        roots = {name.split('/')[0] for name in names}
        c.require(len(roots) == 1, 'One original checkpoint archive root')
        prefix = next(iter(roots))
        c.require(archive.read(prefix + '/byteorder') == b'little', 'Unchanged CPU little-endian source')
        # The frozen save_checkpoint writes state_dict first. Acceptance binds its
        # complete payload; here each selected storage independently matches the
        # accepted name/shape/byte hash. Extra optimizer/RNG storages are excluded.
        for index, entry in enumerate(entries):
            boundary(); member = prefix + '/data/' + str(index)
            c.require(archive.getinfo(member).file_size == entry['bytes'], 'Bounded model storage length')
            raw = archive.read(member)
            c.require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Named accepted model storage bytes')
            _finite(raw); raw_by_name[entry['name']] = raw
    observed = _hashes(entries, raw_by_name)
    c.require(observed['all'] == manifest['metadata']['weights_sha256'] and
              {k: observed[k] for k in ('shared', 'policy', 'value')} == manifest['metadata']['group_sha256'],
              'Exact complete accepted model and group hashes')
    return manifest, entries, raw_by_name


def _binary_bytes(path, entries, *, boundary=lambda: None):
    raw_by_name = {}; offset = len(MAGIC)
    with Path(path).open('rb') as stream:
        c.require(stream.read(len(MAGIC)) == MAGIC, 'Exact state-only/native header')
        for entry in entries:
            boundary()
            c.require(entry['offset'] == offset, 'Contiguous exact tensor byte offsets')
            raw = stream.read(entry['bytes'])
            c.require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'],
                      'Exact state-only tensor bytes')
            _finite(raw); raw_by_name[entry['name']] = raw; offset += entry['bytes']
        c.require(stream.read(1) == b'' and Path(path).stat().st_size == offset, 'No non-model or trailing bytes')
    return raw_by_name


def prepare_candidate(rep=None, *, boundary=lambda: None):
    """Exclusive model-free export; the root preparation owner calls once per rep."""
    rep = repetition(rep)
    c.require(rep == c.REPETITION, 'Exporter process bound to exactly one repetition')
    manifest, entries, raw = source_model_bytes(rep, boundary=boundary)
    folder = c.OUT / 'checkpoint'; folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'candidate_state.bin'
    offset = len(MAGIC); exported = []
    with path.open('xb') as stream:
        stream.write(MAGIC)
        for item in entries:
            boundary(); stream.write(raw[item['name']]); exported.append({**item, 'offset': offset}); offset += item['bytes']
        stream.flush(); os.fsync(stream.fileno())
    if getattr(c.parent, 'WRITE_OBSERVER', None) is not None: c.parent.WRITE_OBSERVER(path, path.stat().st_size)
    item = c.artifact(path)
    tensors = c.save(folder / 'state_tensors.json', {'schema': TENSOR_SCHEMA, 'checkpoint': item,
                      'tensors': exported, 'weights_sha256': manifest['metadata']['weights_sha256']})
    derived = {'schema': SCHEMA, 'checkpoint': item, 'tensor_manifest': tensors,
               'model_definition': c.artifact(architecture.MODEL_SOURCE), 'metadata': manifest['metadata'],
               'source_manifest': source_descriptor(rep), 'source_commitment': commit_descriptor(),
               'source_tensor_manifest': manifest['tensor_manifest'], 'repetition': rep,
               'only_model_tensor_bytes': True, 'optimizer_or_rng_in_live_state': False}
    descriptor = c.save(folder / 'checkpoint_manifest.json', derived)
    proof = verify_candidate_export(descriptor, item, boundary=boundary)
    c.save(folder / 'export_proof.json', proof)
    parent_proof = verify_parent_export(boundary=boundary)
    c.save(folder / 'parent_export_proof.json', parent_proof)
    parent_hashes = architecture.verify_parent_tensor_hashes(boundary=boundary)
    return {'candidate_manifest': descriptor,
            'checkpoint_bindings': {'candidate': {'checkpoint': item, 'weights_sha256': derived['metadata']['weights_sha256'],
                                                  'native_reference': item},
                                    'control': {'checkpoint': c.artifact(architecture.PARENT_CHECKPOINT),
                                                'weights_sha256': parent_hashes['all'], 'native_reference': c.artifact(MODEL_BIN)}},
            'export_proof': c.artifact(folder / 'export_proof.json'),
            'parent_export_proof': c.artifact(folder / 'parent_export_proof.json')}


def candidate_sources(manifest_descriptor, *, boundary=lambda: None):
    boundary(); c.verify([manifest_descriptor])
    manifest = c.read(c.ROOT / manifest_descriptor['path'])
    c.require(set(manifest) == {'schema', 'checkpoint', 'tensor_manifest', 'model_definition', 'metadata',
                               'source_manifest', 'source_commitment', 'source_tensor_manifest', 'repetition',
                               'only_model_tensor_bytes', 'optimizer_or_rng_in_live_state'} and
              manifest['schema'] == SCHEMA and type(manifest['repetition']) is int and
              manifest['repetition'] == c.REPETITION and manifest['only_model_tensor_bytes'] is True and
              manifest['optimizer_or_rng_in_live_state'] is False, 'Strict fixed-repetition derived state schema')
    original, source_entries = source(c.REPETITION, boundary=boundary)
    _identity(manifest['metadata'], c.REPETITION)
    actual_derivation = {key: manifest[key] for key in
                         ('source_manifest', 'source_commitment', 'source_tensor_manifest', 'metadata', 'model_definition')}
    expected_derivation = {'source_manifest': source_descriptor(), 'source_commitment': commit_descriptor(),
                           'source_tensor_manifest': original['tensor_manifest'], 'metadata': original['metadata'],
                           'model_definition': c.artifact(architecture.MODEL_SOURCE)}
    c.require(c.canonical(actual_derivation) == c.canonical(expected_derivation),
              'Exact accepted final96 derivation and JSON types')
    c.verify([manifest['checkpoint'], manifest['tensor_manifest'], manifest['model_definition']])
    doc = c.read(c.ROOT / manifest['tensor_manifest']['path'])
    c.require(set(doc) == {'schema', 'checkpoint', 'tensors', 'weights_sha256'} and doc['schema'] == TENSOR_SCHEMA and
              c.canonical(doc['checkpoint']) == c.canonical(manifest['checkpoint']) and
              doc['weights_sha256'] == manifest['metadata']['weights_sha256'],
              'State-only tensor manifest binding')
    c.require(type(doc['tensors']) is list and len(doc['tensors']) == len(source_entries), 'Complete live tensor inventory')
    offset = len(MAGIC)
    for actual, expected in zip(doc['tensors'], source_entries, strict=True):
        c.require(type(actual) is dict and set(actual) == set(expected) | {'offset'} and type(actual['offset']) is int and
                  c.canonical(actual) == c.canonical({**expected, 'offset': offset}),
                  'Every live tensor descriptor equals accepted source with exact JSON types')
        offset += expected['bytes']
    c.require(manifest['checkpoint']['bytes'] == offset, 'Only model tensor bytes in live file')
    raw = _binary_bytes(c.ROOT / manifest['checkpoint']['path'], doc['tensors'], boundary=boundary)
    observed = _hashes(doc['tensors'], raw)
    c.require(observed['all'] == manifest['metadata']['weights_sha256'] and
              {k: observed[k] for k in ('shared', 'policy', 'value')} == manifest['metadata']['group_sha256'],
              'Complete live state and every model group preserved exactly')
    return manifest, doc


def export_candidate(manifest_descriptor, *, boundary=lambda: None):
    """The exact state-only binary is already the native reference representation."""
    manifest, _ = candidate_sources(manifest_descriptor, boundary=boundary)
    verify_candidate_export(manifest_descriptor, manifest['checkpoint'], boundary=boundary)
    return manifest['checkpoint']


def verify_candidate_export(manifest_descriptor, item, *, boundary=lambda: None):
    manifest, doc = candidate_sources(manifest_descriptor, boundary=boundary)
    c.verify([item])
    c.require(c.canonical(item) == c.canonical(manifest['checkpoint']),
              'Live state bytes are identical native reference bytes and exact descriptor types')
    return {'status': 'passed', 'manifest': manifest_descriptor, 'reference': item, 'tensors': len(doc['tensors']),
            'weights_sha256': manifest['metadata']['weights_sha256'], 'group_sha256': manifest['metadata']['group_sha256'],
            'source_manifest': manifest['source_manifest'], 'repetition': c.REPETITION,
            'purpose': 'exact_state_only_live_and_native_reference', 'model_calls': 0, 'optimizer_calls': 0,
            'live_python_inference_unchanged': True, 'source_unchanged': True}


def verify_parent_export(*, boundary=lambda: None):
    return parent_export.verify_parent_export(boundary=boundary)


def load_parent():
    return architecture.load_parent()


def load_candidate(manifest_descriptor, *, boundary=lambda: None):
    """Runtime only: original architecture, strict exact float32 byte copies."""
    manifest, doc = candidate_sources(manifest_descriptor, boundary=boundary)
    import torch
    raw = _binary_bytes(c.ROOT / manifest['checkpoint']['path'], doc['tensors'], boundary=boundary)
    model = architecture._source_namespace()['EntityTransformer']()
    state = {}
    for entry in doc['tensors']:
        value = torch.frombuffer(bytearray(raw[entry['name']]), dtype=torch.float32).clone().reshape(entry['shape'])
        c.require(value.detach().numpy().tobytes() == raw[entry['name']], 'Exact float32 import without arithmetic')
        state[entry['name']] = value
    model.load_state_dict(state, strict=True); model.eval(); architecture.validate_architecture(model)
    c.require(architecture.state_digest(model) == manifest['metadata']['weights_sha256'] and
              architecture.group_digests(model) == manifest['metadata']['group_sha256'],
              'Live model exactly equals both source and derived state')
    boundary()
    return model, manifest
