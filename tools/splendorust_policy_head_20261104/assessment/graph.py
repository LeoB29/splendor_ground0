"""One prospectively fixed graph of the existing float32 tensor forward.

Importing this module is model-free. Compilation, zero-input warmup and weight
fingerprinting are performed only by the supervised worker initialization.
Token validation and Python conversion remain outside the graph, in the same
order and with the same operations as the original Opponent._infer method.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import traceback
import warnings

from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256


BACKEND = 'torchscript_frozen_exact_forward_v1'
SCHEMA = 'splendorust_inference_graph_identity_v1'
PARITY_SCHEMA = 'splendorust_inference_graph_parity_v1'
SUPPORTED_BATCHES = (1, 2, 8)
TOKENS_PER_ITEM = 31 * 48
OUTPUTS_PER_ITEM = 83
PARITY_CASES = 11
MAX_GRAPH_BYTES = 256 * 1024
GRAPH_OPTIONS = {'trace_batch': 1, 'trace_check_trace': False, 'trace_strict': True,
                 'freeze_optimize_numerics': False, 'warm_batches': [1, 2, 8],
                 'trace_and_warm_inputs': 'architecture_zero_float32_only',
                 'automatic_fallback': False, 'alternative_candidates': 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _save_text(path, value):
    import os
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _weight_digest(model, torch):
    """Fingerprint unchanged float32 state; transient copy is one tensor only."""
    hasher = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.state_dict().items()):
        require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32,
                'Graph requires unchanged CPU float32 state')
        hasher.update(canonical([name, list(tensor.shape), str(tensor.dtype)]).encode())
        hasher.update(b'\0')
        hasher.update(tensor.detach().contiguous().numpy().tobytes(order='C'))
        count += tensor.numel()
    require(count == 4_780_883, 'Graph state parameter count changed')
    return hasher.hexdigest()


def _make_forward(torch, model):
    class ExactTensorForward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            h = self.model.norm(self.model.blocks(self.model.project(x) + self.model.identity)[:, 0])
            return torch.cat((self.model.policy(h), self.model.value(h).tanh()), dim=1)

    return ExactTensorForward().eval()


class FrozenForward:
    """Immutable selected callable; there is deliberately no eager fallback."""
    def __init__(self, compiled, identity):
        self.compiled = compiled
        self.identity = identity

    def infer(self, adapter, tokens, batch):
        import torch
        if batch not in (1, 2, 8) or len(tokens) != batch * 31 * 48:
            raise ValueError('Malformed public entity tokens')
        with torch.inference_mode():
            x = torch.tensor(tokens, dtype=torch.float32).reshape(batch, 31, 48)
            if not bool(torch.isfinite(x).all()):
                raise ValueError('Nonfinite model input')
            out = self.compiled(x)
            if not bool(torch.isfinite(out).all()):
                raise ValueError('Nonfinite model output')
            return out.tolist()


def _build_graph(model, folder, save_json, build_context, checkpoint_sha256):
    import torch
    import numpy
    build_context['phase'] = 'preconditions_and_weight_fingerprint'
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
            'Graph numerical thread limits changed')
    require(not model.has_history and all(not module.training for module in model.modules()),
            'Graph requires exact evaluation-mode architecture')
    before = _weight_digest(model, torch)
    with torch.inference_mode():
        forward = _make_forward(torch, model)
        example = torch.zeros((1, 31, 48), dtype=torch.float32, device='cpu')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                build_context['phase'] = 'trace_once'
                traced = torch.jit.trace(forward, example, check_trace=False, strict=True)
                build_context['phase'] = 'freeze_once_without_numerical_optimizations'
                compiled = torch.jit.freeze(traced, optimize_numerics=False)
            finally:
                build_context['trace_warnings'] = [
                    {'category': item.category.__name__, 'message': str(item.message)[:2048]}
                    for item in captured[:64]]
                build_context['trace_warning_count'] = len(captured)
        # The actual parity operation later uses nonzero frozen recorded
        # tensors. These warmups do not establish numerical correctness.
        for batch in SUPPORTED_BATCHES:
            build_context['phase'] = 'zero_warm_batch_' + str(batch)
            out = compiled(torch.zeros((batch, 31, 48), dtype=torch.float32, device='cpu'))
            require(tuple(out.shape) == (batch, OUTPUTS_PER_ITEM) and out.dtype == torch.float32
                    and out.device.type == 'cpu' and bool(torch.isfinite(out).all()),
                    'Frozen graph zero-warm shape/type/finiteness failed')
    build_context['phase'] = 'postbuild_weight_fingerprint_and_identity'
    after = _weight_digest(model, torch)
    require(before == after and all(not module.training for module in model.modules()),
            'Graph construction modified original weights or evaluation mode')
    graph_text = str(compiled.inlined_graph) + '\n'
    encoded = graph_text.encode('utf-8')
    require(len(encoded) <= MAX_GRAPH_BYTES, 'Frozen graph IR exceeds fixed evidence bound')
    _save_text(Path(folder) / 'graph_ir.txt', graph_text)
    identity = {'schema': SCHEMA, 'backend': BACKEND, 'options': GRAPH_OPTIONS,
        'checkpoint_sha256': checkpoint_sha256, 'weights_before_sha256': before,
        'weights_after_sha256': after, 'parameter_count': 4_780_883,
        'graph_ir_sha256': hashlib.sha256(encoded).hexdigest(), 'graph_ir_bytes': len(encoded),
        'graph_ir_artifact': 'graph_ir.txt', 'serialized_graph_saved': False,
        'supported_batches': list(SUPPORTED_BATCHES), 'input_shape': ['batch', 31, 48],
        'output_shape': ['batch', OUTPUTS_PER_ITEM], 'dtype': 'torch.float32', 'device': 'cpu',
        'torch_version': str(torch.__version__), 'torch_git_version': str(torch.version.git_version),
        'numpy_version': str(numpy.__version__),
        'torch_module': str(Path(torch.__file__).resolve()),
        'torch_core_module': str(Path(torch._C.__file__).resolve()),
        'torch_build_configuration': torch.__config__.show(),
        'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'mha_fastpath_enabled': torch.backends.mha.get_fastpath_enabled(),
        'mkldnn_enabled': torch.backends.mkldnn.enabled,
        'trace_warnings': build_context['trace_warnings'],
        'trace_warning_count': build_context['trace_warning_count'], 'warmup_is_parity_proof': False}
    save_json(Path(folder) / 'graph_identity.json', identity)
    return FrozenForward(compiled, identity)


def build_graph(model, folder, save_json, *, checkpoint_sha256):
    """Trace once, freeze once, warm three zero shapes; failure is terminal."""
    build_context = {'backend': BACKEND, 'options': GRAPH_OPTIONS, 'phase': 'runtime_import',
                     'trace_warnings': [], 'trace_warning_count': 0}
    try:
        return _build_graph(model, folder, save_json, build_context, checkpoint_sha256)
    except BaseException as error:
        save_json(Path(folder) / 'graph_build_error.json',
                  dict(build_context, error=repr(error), traceback=traceback.format_exc()))
        raise


def float32_bytes(outputs, batch):
    """Pack returned Python floats back to their original IEEE-754 float32 bits."""
    require(type(outputs) is list and len(outputs) == batch, 'Parity output batch shape')
    import math
    flat = []
    for row in outputs:
        require(type(row) is list and len(row) == OUTPUTS_PER_ITEM, 'Parity output width')
        require(all(type(value) is float and math.isfinite(value) for value in row),
                'Parity outputs must be finite exact floats')
        flat.extend(row)
    return struct.pack('<' + 'f' * len(flat), *flat)


def compare_outputs(fixture, eager, candidate):
    eager_bits = float32_bytes(eager, fixture['batch'])
    candidate_bits = float32_bytes(candidate, fixture['batch'])
    return {'fixture_id': fixture['fixture_id'], 'batch': fixture['batch'],
        'input_sha256': digest(fixture['tokens']), 'exact_float_equal': eager == candidate,
        'float32_bits_equal': eager_bits == candidate_bits,
        'eager_output_sha256': digest(eager), 'candidate_output_sha256': digest(candidate),
        'eager_float32_sha256': hashlib.sha256(eager_bits).hexdigest(),
        'candidate_float32_sha256': hashlib.sha256(candidate_bits).hexdigest()}


def validate_parity_fixtures(fixtures):
    import math
    require(type(fixtures) is list and len(fixtures) == PARITY_CASES, 'Expected fixed eleven parity cases')
    ids = set()
    batches = []
    for fixture in fixtures:
        require(type(fixture) is dict and set(fixture) == {'fixture_id', 'batch', 'tokens'},
                'Parity fixture allowlist')
        require(type(fixture['fixture_id']) is str and fixture['fixture_id'] and
                fixture['fixture_id'] not in ids, 'Parity fixture identity')
        ids.add(fixture['fixture_id'])
        batch, tokens = fixture['batch'], fixture['tokens']
        require(type(batch) is int and batch in SUPPORTED_BATCHES and type(tokens) is list and
                len(tokens) == batch * TOKENS_PER_ITEM, 'Parity tensor shape')
        require(all(type(value) in (int, float) and math.isfinite(value) for value in tokens),
                'Parity tensor values')
        batches.append(batch)
    require(sorted(batches) == [1] * 7 + [2] * 2 + [8] * 2, 'Parity eleven-case batch coverage')


class InferenceBinding:
    """Process-local base-method substitution; callback hashing stays untouched."""
    _active = False

    def __init__(self):
        self.owner = None
        self.original = None
        self.replacement = None

    def install(self, adapters, forward, *, owner=None):
        if owner is None:
            from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
            owner = Opponent
        require(not InferenceBinding._active and self.owner is None, 'Concurrent or repeated inference binding')
        require(bool(adapters) and len({id(adapter.model) for adapter in adapters.values()}) == 1,
                'Candidate adapters must share the exact eager model')
        allowed = {id(adapter) for adapter in adapters.values()}
        original = owner._infer
        def replacement(adapter, tokens, batch):
            require(id(adapter) in allowed, 'Unqualified adapter reached graph inference')
            return forward.infer(adapter, tokens, batch)
        self.owner, self.original, self.replacement = owner, original, replacement
        owner._infer = replacement
        InferenceBinding._active = True
        return self

    def close(self):
        if self.owner is not None:
            try:
                require(self.owner._infer is self.replacement, 'Graph inference binding changed unexpectedly')
            finally:
                self.owner._infer = self.original
                self.owner = self.original = self.replacement = None
                InferenceBinding._active = False

# Preserve the exact qualified callable owner and generated graph naming.
from tools.splendorust_inference_opt_20261025.graph import _make_forward
