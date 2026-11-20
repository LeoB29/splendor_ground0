"""Model-free regression coverage for final96 import and preserved live contracts."""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '0')
from tools.splendorust_independent_data_recovery_20261109.strength import checkpoint as cp
from tools.splendorust_independent_data_recovery_20261109.strength import adapters, graph, transport, worker
from tools.splendorust_independent_data_recovery_20261109.strength import common as c

OLD = c.ROOT / 'tools/splendorust_onepass_strength_20261106/assessment'
NEW = c.ROOT / 'tools/splendorust_independent_data_recovery_20261109/strength'


@pytest.fixture(autouse=True)
def saved_archive_fixture_only(monkeypatch):
    """Exercise exact byte importer against saved archive, with synthetic new identity.

    This deliberately bypasses FUTURE commitment existence only in these unit
    fixtures; actual commitment tests below exercise that chain independently.
    No checkpoint is exported, no historical target is relabelled on disk.
    """
    original_read = c.read
    fixture_commit = {'path': 'unit-fixture-commit.json', 'bytes': 1, 'sha256': 'f'*64}
    def descriptor(rep=None, **kwargs):
        rep = c.REPETITION if rep is None else rep
        return c.artifact(c.ROOT / f'outputs/splendorust_shorter_joint_20261106_01/fits/r{rep}_broad_joint/worker/checkpoints/096/checkpoint_manifest.json')
    def read(path):
        if Path(path).name == 'unit-fixture-commit.json':
            return {'selection_sha256': 'd'*64, 'source_receipt_sha256': 'e'*64}
        value = original_read(path)
        if 'splendorust_shorter_joint_20261106_01' in str(path) and Path(path).name == 'checkpoint_manifest.json':
            value = deepcopy(value)
            value['metadata'].update(study_id=cp.SOURCE_STUDY, source_training_games=1024, roots_per_game=3,
                selection_sha256='d'*64, source_receipt_sha256='e'*64)
        return value
    monkeypatch.setattr(cp, 'source_descriptor', descriptor)
    monkeypatch.setattr(cp, 'commit_descriptor', lambda: deepcopy(fixture_commit))
    monkeypatch.setattr(c, 'read', read)


@pytest.mark.parametrize('rep', [0, 1])
def test_actual_accepted_final96_zip_state_only(rep):
    before_torch = 'torch' in sys.modules
    manifest, entries, raw = cp.source_model_bytes(rep)
    assert len(entries) == 81 and sum(map(len, raw.values())) == 19_123_532
    assert sum(entry['numel'] for entry in entries) == 4_780_883
    assert cp._hashes(entries, raw)['all'] == manifest['metadata']['weights_sha256']
    assert ('torch' in sys.modules) == before_torch
    assert set(raw) == set(cp.architecture.expected_state_shapes())
    assert not any('optimizer' in name or 'rng' in name for name in raw)


@pytest.mark.parametrize('value', [True, False, 0.0, 1.0, '0', '1', -1, 2, None])
def test_repetition_strict_types(value):
    if value is None:
        assert cp.repetition(value) == c.REPETITION
    else:
        with pytest.raises(ValueError): cp.repetition(value)


@pytest.mark.parametrize('field,value', [
    ('repetition', True), ('checkpoint_step', 95), ('checkpoint_step', 96.0),
    ('fixed_final_step', 668), ('scope', 'policy'), ('qualification_only', True),
    ('checkpoint_selection', True), ('historical_data_relabelled', True),
    ('historical_warm_start_sha256', '0'*64), ('catalogue_sha256', '0'*64), ('source_training_games', 192), ('roots_per_game', 16),
    ('source_training_games', 1024.0), ('study_id', 'splendorust_shorter_joint_20261106_v1'),
])
def test_source_identity_rejects_wrong_checkpoint_and_legacy(field, value):
    rep = c.REPETITION
    manifest = c.read(c.ROOT / cp.source_descriptor(rep)['path'])
    metadata = deepcopy(manifest['metadata']); metadata[field] = value
    with pytest.raises(ValueError): cp._identity(metadata, rep)


def test_both_repeat_identities_are_distinct():
    manifests = [c.read(c.ROOT / cp.source_descriptor(rep)['path']) for rep in (0, 1)]
    assert manifests[0]['metadata']['weights_sha256'] != manifests[1]['metadata']['weights_sha256']
    with pytest.raises(ValueError): cp._identity(manifests[1]['metadata'], 0)
    with pytest.raises(ValueError): cp._identity(manifests[0]['metadata'], 1)


@pytest.mark.parametrize('mutation', ['metadata_repetition_bool', 'metadata_parameters_float',
                                     'source_manifest_bytes_float', 'source_commitment_bytes_float'])
def test_actual_derived_reader_rejects_python_equal_type_mutations(mutation):
    original = c.read(c.ROOT / cp.source_descriptor()['path'])
    derived = {'schema': cp.SCHEMA, 'checkpoint': original['checkpoint'],
               'tensor_manifest': original['tensor_manifest'],
               'model_definition': c.artifact(cp.architecture.MODEL_SOURCE),
               'metadata': deepcopy(original['metadata']), 'source_manifest': cp.source_descriptor(),
               'source_commitment': deepcopy(cp.commit_descriptor()),
               'source_tensor_manifest': original['tensor_manifest'], 'repetition': c.REPETITION,
               'only_model_tensor_bytes': True, 'optimizer_or_rng_in_live_state': False}
    if mutation == 'metadata_repetition_bool': derived['metadata']['repetition'] = bool(c.REPETITION)
    elif mutation == 'metadata_parameters_float': derived['metadata']['parameter_count'] = 4780883.0
    elif mutation == 'source_manifest_bytes_float': derived['source_manifest']['bytes'] = 2938.0
    elif mutation == 'source_commitment_bytes_float': derived['source_commitment']['bytes'] = 6046.0
    descriptor = {'path': 'unused-derived-manifest.json', 'bytes': 1, 'sha256': '0'*64}
    with patch.object(c, 'verify'), patch.object(c, 'read', return_value=derived), \
         patch.object(cp, 'source', return_value=(original, [])):
        with pytest.raises(ValueError): cp.candidate_sources(descriptor)


def test_actual_derived_reader_rejects_nested_checkpoint_descriptor_float():
    original = c.read(c.ROOT / cp.source_descriptor()['path'])
    metadata = deepcopy(original['metadata'])
    checkpoint = {'path': 'unused-state.bin', 'bytes': 19123540, 'sha256': 'a'*64}
    tensor_manifest = {'path': 'unused-tensors.json', 'bytes': 1, 'sha256': 'b'*64}
    descriptor = {'path': 'unused-derived.json', 'bytes': 1, 'sha256': 'c'*64}
    derived = {'schema': cp.SCHEMA, 'checkpoint': checkpoint, 'tensor_manifest': tensor_manifest,
               'model_definition': c.artifact(cp.architecture.MODEL_SOURCE), 'metadata': metadata,
               'source_manifest': cp.source_descriptor(), 'source_commitment': cp.commit_descriptor(),
               'source_tensor_manifest': original['tensor_manifest'], 'repetition': c.REPETITION,
               'only_model_tensor_bytes': True, 'optimizer_or_rng_in_live_state': False}
    doc = {'schema': cp.TENSOR_SCHEMA, 'checkpoint': {**checkpoint, 'bytes': float(checkpoint['bytes'])},
           'tensors': [], 'weights_sha256': metadata['weights_sha256']}
    def read(path): return doc if Path(path).name == 'unused-tensors.json' else derived
    with patch.object(c, 'verify'), patch.object(c, 'read', side_effect=read), \
         patch.object(cp, 'source', return_value=(original, [])):
        with pytest.raises(ValueError, match='State-only tensor manifest binding'):
            cp.candidate_sources(descriptor)


def test_export_verifier_rejects_python_equal_descriptor_types():
    item = {'path': 'unused-state.bin', 'bytes': 19123540, 'sha256': 'a'*64}
    with patch.object(cp, 'candidate_sources', return_value=({'checkpoint': item}, {})), patch.object(c, 'verify'):
        with pytest.raises(ValueError, match='exact descriptor types'):
            cp.verify_candidate_export({}, {**item, 'bytes': float(item['bytes'])})


@pytest.mark.parametrize('mutation', ['drop', 'reorder', 'dtype', 'shape_float', 'bytes_bool', 'extra'])
def test_nested_source_tensor_inventory_is_strict(mutation):
    manifest = c.read(c.ROOT / cp.source_descriptor()['path'])
    tree = deepcopy(c.read(c.ROOT / manifest['tensor_manifest']['path'])['state_dict'])
    if mutation == 'drop': tree['dict'].pop()
    elif mutation == 'reorder': tree['dict'][0], tree['dict'][1] = tree['dict'][1], tree['dict'][0]
    elif mutation == 'dtype': tree['dict'][0][1]['dtype'] = 'torch.float16'
    elif mutation == 'shape_float': tree['dict'][0][1]['shape'][0] = 31.0
    elif mutation == 'bytes_bool': tree['dict'][0][1]['bytes'] = True
    elif mutation == 'extra': tree['dict'][0][1]['zip_member'] = 'untrusted'
    with pytest.raises(ValueError): cp._state_entries(tree)


@pytest.mark.parametrize('words', [(0x7f800000,), (0xff800000,), (0x7fc00000,)])
def test_nonfinite_float32_bytes_rejected(words):
    with pytest.raises(ValueError): cp._finite(struct.pack('<I', *words))


def test_signed_zero_subnormal_and_finite_bits_preserved():
    raw = struct.pack('<5I', 0, 0x80000000, 1, 0x3f800000, 0xbf800000)
    cp._finite(raw)
    assert raw == struct.pack('<5I', 0, 0x80000000, 1, 0x3f800000, 0xbf800000)


@pytest.mark.parametrize('mutation', ['header', 'payload', 'truncated', 'trailing', 'offset'])
def test_binary_import_rejects_corruption(mutation):
    raw = struct.pack('<2f', 1., -0.)
    entry = {'name': 'toy', 'shape': [2], 'dtype': 'torch.float32', 'numel': 2,
             'bytes': 8, 'sha256': hashlib.sha256(raw).hexdigest(), 'offset': len(cp.MAGIC)}
    data = cp.MAGIC + raw
    if mutation == 'header': data = b'X' + data[1:]
    elif mutation == 'payload': data = data[:-1] + bytes([data[-1] ^ 1])
    elif mutation == 'truncated': data = data[:-1]
    elif mutation == 'trailing': data += b'X'
    elif mutation == 'offset': entry['offset'] += 1
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'state.bin'; path.write_bytes(data)
        with pytest.raises(ValueError): cp._binary_bytes(path, [entry])


def _function(path, name):
    return next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


@pytest.mark.parametrize('module,names', [
    ('adapters.py', ['BaselineOpponent']),
    ('worker.py', ['_ordinary', 'handle', 'serve', 'verify_graph_admission']),
    ('graph.py', ['_make_forward', '_build_graph', 'compare_outputs', 'validate_parity_fixtures']),
    ('transport.py', ['_PlacementConstructor', 'PlacedProfiledStudyWorker', 'request', 'validate_profiles']),
    ('qualification.py', ['verify_entrypoint', 'compare_control_parity']),
])
def test_qualified_model_search_clock_and_scientific_bodies_preserved(module, names):
    for name in names:
        assert ast.dump(_function(NEW / module, name), include_attributes=False) == ast.dump(
            _function(OLD / module, name), include_attributes=False)


@pytest.mark.parametrize('module', ['corpus.py', 'scientific.py', 'parent_profile.py', 'boundary_adapter.py'])
def test_entire_support_modules_preserved(module):
    assert (NEW / module).read_bytes().replace(b'\r\n', b'\n') == (OLD / module).read_bytes().replace(b'\r\n', b'\n')


def test_candidate_profile_and_runtime_namespace_pin_repetition():
    assert adapters.profile_id(128, route='candidate') == f'independent_data_r{c.REPETITION}_puct128_pool3_20261107_v1'
    assert adapters.profile_id(128, route='control') == 'strength_search_puct128_pool3_20261028_v1'
    assert transport.OBSERVER_MODULE == 'tools.splendorust_independent_data_recovery_20261109.strength.worker'
    initialization = ast.get_source_segment((NEW / 'worker.py').read_text(encoding='utf-8'), _function(NEW / 'worker.py', 'initialize'))
    assert "freeze['repetition'] == c.REPETITION" in initialization
    assert "os.environ.get('INDEPENDENT_DATA_REPETITION')" in initialization


def _segment_supervision_case():
    root_claim = {'path': 'root/claim.json', 'bytes': 10, 'sha256': 'a'*64}
    phase = {'path': 'root/main_start.json', 'bytes': 20, 'sha256': 'b'*64}
    start = {'repetition': c.REPETITION, 'phase': 'main_block', 'block_start': 10,
             'owner': {'pid': 123, 'created': 123.5}, 'root_claim': root_claim, 'root_phase_clock': phase}
    segment = {'repetition': c.REPETITION, 'mode': 'main_block', 'block_start': 10,
               'claim': root_claim, 'phase_clock': phase, 'freeze_sha256': 'c'*64}
    freeze = {'repetition': c.REPETITION, 'root_freeze': {'path': 'root/freeze.json', 'bytes': 30, 'sha256': 'c'*64}}
    supervisor = {'pid': 123, 'created': 123.5, 'unrelated_placement_field': True}
    return start, segment, supervisor, freeze


def test_fixed_segment_supervisor_matches_actual_owner():
    transport.validate_segment_supervisor(*_segment_supervision_case(), 'main_block', 10)


@pytest.mark.parametrize('mutation', ['wrong_pid', 'reused_pid', 'wrong_rep', 'bool_rep', 'wrong_mode',
                                     'wrong_block', 'float_block', 'wrong_freeze', 'wrong_phase', 'float_receipt_size'])
def test_saved_segment_supervisor_rejects_mismatch(mutation):
    start, segment, supervisor, freeze = deepcopy(_segment_supervision_case())
    if mutation == 'wrong_pid': supervisor['pid'] += 1
    elif mutation == 'reused_pid': supervisor['created'] += 1
    elif mutation == 'wrong_rep': segment['repetition'] = 1-c.REPETITION
    elif mutation == 'bool_rep': start['repetition'] = bool(c.REPETITION)
    elif mutation == 'wrong_mode': segment['mode'] = 'qualification'
    elif mutation == 'wrong_block': segment['block_start'] += 10
    elif mutation == 'float_block': segment['block_start'] = 10.0
    elif mutation == 'wrong_freeze': segment['freeze_sha256'] = 'd'*64
    elif mutation == 'wrong_phase': segment['phase_clock'] = {**segment['phase_clock'], 'sha256': 'd'*64}
    elif mutation == 'float_receipt_size': segment['claim'] = {**segment['claim'], 'bytes': 10.0}
    with pytest.raises(ValueError):
        transport.validate_segment_supervisor(start, segment, supervisor, freeze, 'main_block', 10)


def test_actual_placement_reader_uses_fixed_segment_not_old_single_launch():
    start, segment, supervisor, freeze = _segment_supervision_case()
    folder = c.OUT / 'workers/main_g010_candidate'
    claim_path = c.STUDY_OUT / 'segments' / f'main_block_r{c.REPETITION}_g010' / 'claim.json'
    def artifact(path):
        return {'path': Path(path).resolve().relative_to(c.ROOT).as_posix(), 'bytes': 10, 'sha256': 'a'*64}
    start['root_claim'] = artifact(c.STUDY_OUT/'claim.json')
    start['local_freeze'] = artifact(c.OUT/'freeze.json')
    start['segment_claim'] = artifact(claim_path)
    segment['claim'] = start['root_claim']
    packet = {'request_id': 'request-00000003'}
    files = {str(folder/'placement_requests/request-00000003.json'): packet,
             str(c.OUT/'segments/main_010/start.json'): start, str(claim_path): segment}
    seen = []
    def read(path):
        seen.append(str(path)); return files[str(path)]
    with patch.object(transport.placement, 'verify_worker_placement',
                      return_value={'player_pid': 999, 'supervisor': supervisor}), \
         patch.object(transport.placement, 'verify_saved_request'), patch.object(c, 'read', side_effect=read), \
         patch.object(c, 'verify'), patch.object(c, 'artifact', side_effect=artifact):
        assert transport.verify_placement_request(folder, 'request-00000003', freeze, {}) == packet
    assert not any(path.endswith('launch.json') for path in seen)


@pytest.mark.parametrize('raw', ['', 'true', 'False', '2', '-1', '0.0', ' 0'])
def test_worker_namespace_rejects_ambiguous_environment_before_models(raw):
    env = os.environ.copy(); env['INDEPENDENT_DATA_REPETITION'] = raw
    result = subprocess.run([sys.executable, '-B', '-c',
        'from tools.splendorust_independent_data_recovery_20261109.strength import common'],
        env=env, capture_output=True, timeout=10)
    assert result.returncode != 0 and b'INDEPENDENT_DATA_REPETITION' in result.stderr


def test_backend_import_does_not_import_torch_in_fresh_process():
    result = subprocess.run([sys.executable, '-B', '-c',
        'import sys; from tools.splendorust_independent_data_recovery_20261109.strength import checkpoint,adapters,worker,graph,qualification,scientific,corpus,transport; assert "torch" not in sys.modules'],
        env=os.environ.copy(), capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()


def test_parent_native_export_reproduces_accepted_binary():
    proof = cp.verify_parent_export()
    assert proof['status'] == 'passed' and proof['tensors'] == 81
    assert proof['model_bin']['sha256'] == '57f6e227f8ac0382b7fa67dba6b58ec6663d1f635c7b9f583937867cd6692deb'
