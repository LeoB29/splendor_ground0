"""Symbolic eligibility and argument-order checks; no model or kernel executes."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tools.move_evaluation_inference_20261023 import inference as base
from tools.move_evaluation_inference_opt_20261023 import inference as opt


class Module:
    def __init__(self):
        self.training = False
        self._compiled_call_impl = None
        for name in opt.LOCAL_HOOKS:
            setattr(self, name, {})


class Parameter:
    def __init__(self, shape, torch):
        self.shape, self.dtype, self.layout = shape, torch.float32, torch.strided
        self.device, self.is_nested = SimpleNamespace(type='cpu'), False
        self.contiguous = True
    def is_contiguous(self):
        return self.contiguous
    def stride(self):
        return (self.shape[1], 1) if len(self.shape) == 2 else (1,)


class Attention(Module):
    pass


class Layer(Module):
    def __init__(self, torch):
        super().__init__()
        self.self_attn = Attention()
        self.self_attn.embed_dim, self.self_attn.num_heads = 256, 8
        self.self_attn.batch_first = self.self_attn._qkv_same_embed_dim = True
        self.activation_relu_or_gelu, self.norm_first = 2, True
        self.norm1, self.norm2 = Module(), Module()
        self.norm1.eps = self.norm2.eps = 1e-5
        self.linear1, self.linear2 = Module(), Module()
        self.self_attn.out_proj = Module()
        paths = {'in_proj_weight': (self.self_attn, 'in_proj_weight'),
                 'in_proj_bias': (self.self_attn, 'in_proj_bias')}
        for prefix, module in [('out_proj', self.self_attn.out_proj), ('norm1', self.norm1),
                               ('norm2', self.norm2), ('linear1', self.linear1), ('linear2', self.linear2)]:
            paths[prefix+'_weight'] = (module, 'weight')
            paths[prefix+'_bias'] = (module, 'bias')
        for name, (module, field) in paths.items():
            setattr(module, field, Parameter(opt.PARAMETER_SHAPES[name], torch))


class Encoder(Module):
    def __init__(self, torch):
        super().__init__()
        self.layers = [Layer(torch) for _ in range(6)]
        self.num_layers = 6
        self.norm = None
        self.enable_nested_tensor = self.use_nested_tensor = False


class Model(Module):
    def __init__(self, torch):
        super().__init__()
        self.blocks = Encoder(torch)
        self.has_history = False
    def modules(self):
        yield self
        yield self.blocks
        for layer in self.blocks.layers:
            yield from (layer, layer.self_attn, layer.self_attn.out_proj, layer.norm1,
                        layer.norm2, layer.linear1, layer.linear2)


class Kernel:
    __name__ = '_transformer_encoder_layer_fwd'
    __module__ = 'torch'
    def __init__(self):
        self.calls = []
    def __call__(self, h, *args):
        self.calls.append((h, args))
        return f'layer{len(self.calls)}'


class Runtime:
    __version__ = opt.TORCH_VERSION
    float32, strided = object(), object()
    def __init__(self):
        self.fastpath, self.autocast, self.cpu_autocast = True, False, False
        self.tracing, self.scripting, self.grad, self.inference = None, False, False, True
        self.override = False
        self.backends = SimpleNamespace(mha=SimpleNamespace(get_fastpath_enabled=lambda: self.fastpath))
        self._C = SimpleNamespace(_get_tracing_state=lambda: self.tracing)
        self.jit = SimpleNamespace(is_scripting=lambda: self.scripting)
        self.nn = SimpleNamespace(Parameter=Parameter, TransformerEncoder=Encoder,
                                  TransformerEncoderLayer=Layer, MultiheadAttention=Attention)
        self.overrides = SimpleNamespace(has_torch_function=lambda _: self.override)
        self._transformer_encoder_layer_fwd = Kernel()
    def is_autocast_enabled(self, device=None):
        return self.cpu_autocast if device == 'cpu' else self.autocast
    def is_grad_enabled(self):
        return self.grad
    def is_inference_mode_enabled(self):
        return self.inference


def setup():
    torch = Runtime()
    runtime = SimpleNamespace(**{name: {} for name in opt.GLOBAL_HOOKS})
    return torch, runtime, Model(torch)


def runtime_receipt(tmp_path):
    return [{'relative_path': relative, 'absolute_path': str((tmp_path/relative).resolve()),
             'sha256': sha, 'bytes': 100} for relative, sha in opt.RUNTIME_HASHES.items()]


def test_six_ordered_calls_use_exact_original_parameter_objects():
    torch, runtime, model = setup()
    kernel, bound, description = opt.bind_eligible_layers(model, torch, runtime)
    assert type(bound) is tuple and len(bound) == len(description) == 6
    for layer, args in zip(model.blocks.layers, bound):
        assert len(args) == 19
        assert args[2] is layer.self_attn.in_proj_weight
        assert args[3] is layer.self_attn.in_proj_bias
        assert args[4] is layer.self_attn.out_proj.weight
        assert args[5] is layer.self_attn.out_proj.bias
        assert args[9] is layer.norm1.weight and args[10] is layer.norm1.bias
        assert args[11] is layer.norm2.weight and args[12] is layer.norm2.bias
        assert args[13] is layer.linear1.weight and args[14] is layer.linear1.bias
        assert args[15] is layer.linear2.weight and args[16] is layer.linear2.bias
        assert args[:2] == (256, 8) and args[6:9] == (True, True, 1e-5) and args[17:] == (None, None)
    opponent = SimpleNamespace(_prebound_torch=torch, _module_runtime=runtime,
                               _prebound_kernel=kernel, _prebound_layers=bound)
    assert opt.infer_transformer(opponent, 'input') == 'layer6'
    assert [call[0] for call in kernel.calls] == ['input', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    for (_, observed), expected in zip(kernel.calls, bound):
        assert all(a is b for a, b in zip(observed, expected))


@pytest.mark.parametrize('field,value', [('fastpath', False), ('autocast', True), ('cpu_autocast', True),
    ('tracing', object()), ('scripting', True), ('override', True)])
def test_startup_global_and_tensor_conditions_fail_closed(field, value):
    torch, runtime, model = setup()
    setattr(torch, field, value)
    with pytest.raises(ValueError):
        opt.bind_eligible_layers(model, torch, runtime)
    assert torch._transformer_encoder_layer_fwd.calls == []


@pytest.mark.parametrize('hook', opt.GLOBAL_HOOKS)
def test_global_hooks_rejected(hook):
    torch, runtime, model = setup()
    setattr(runtime, hook, {1: object()})
    with pytest.raises(ValueError, match='Global module hooks'):
        opt.bind_eligible_layers(model, torch, runtime)


@pytest.mark.parametrize('hook', opt.LOCAL_HOOKS)
def test_local_hooks_rejected(hook):
    torch, runtime, model = setup()
    setattr(model.blocks.layers[3], hook, {1: object()})
    with pytest.raises(ValueError, match='Local module hooks'):
        opt.bind_eligible_layers(model, torch, runtime)


@pytest.mark.parametrize('change', ['history', 'count', 'norm', 'nested', 'training', 'compiled',
    'batch_first', 'qkv', 'gelu', 'norm_first', 'eps', 'heads', 'width', 'bias',
    'dtype', 'device', 'shape', 'layout', 'nested_parameter', 'contiguous', 'subclass'])
def test_static_layer_eligibility_counterexamples(change):
    torch, runtime, model = setup()
    layer = model.blocks.layers[0]
    parameter = layer.self_attn.in_proj_weight
    if change == 'history': model.has_history = True
    elif change == 'count': model.blocks.layers.pop()
    elif change == 'norm': model.blocks.norm = object()
    elif change == 'nested': model.blocks.use_nested_tensor = True
    elif change == 'training': layer.training = True
    elif change == 'compiled': layer._compiled_call_impl = object()
    elif change == 'batch_first': layer.self_attn.batch_first = False
    elif change == 'qkv': layer.self_attn._qkv_same_embed_dim = False
    elif change == 'gelu': layer.activation_relu_or_gelu = 1
    elif change == 'norm_first': layer.norm_first = False
    elif change == 'eps': layer.norm2.eps = 1e-4
    elif change == 'heads': layer.self_attn.num_heads = 4
    elif change == 'width': layer.self_attn.embed_dim = 128
    elif change == 'bias': layer.self_attn.in_proj_bias = None
    elif change == 'dtype': parameter.dtype = object()
    elif change == 'device': parameter.device.type = 'cuda'
    elif change == 'shape': parameter.shape = (256, 256)
    elif change == 'layout': parameter.layout = object()
    elif change == 'nested_parameter': parameter.is_nested = True
    elif change == 'contiguous': parameter.contiguous = False
    else:
        class DerivedLayer(Layer): pass
        model.blocks.layers[0] = DerivedLayer(torch)
    with pytest.raises(ValueError):
        opt.bind_eligible_layers(model, torch, runtime)


@pytest.mark.parametrize('change', ['grad', 'inference', 'kernel', 'global_hooks', 'autocast'])
def test_dynamic_changes_fail_before_any_native_call(change):
    torch, runtime, model = setup()
    kernel, bound, _ = opt.bind_eligible_layers(model, torch, runtime)
    opponent = SimpleNamespace(_prebound_torch=torch, _module_runtime=runtime,
                               _prebound_kernel=kernel, _prebound_layers=bound)
    if change == 'grad': torch.grad = True
    elif change == 'inference': torch.inference = False
    elif change == 'kernel': torch._transformer_encoder_layer_fwd = Kernel()
    elif change == 'global_hooks': runtime._global_forward_hooks[1] = object()
    else: torch.autocast = True
    with pytest.raises(ValueError): opt.infer_transformer(opponent, 'input')
    assert kernel.calls == []


def constructor_stubs(tmp_path, monkeypatch, *, eligibility_failure=False):
    torch, runtime, model = setup()
    events = []
    def initialize(self, *args, **kwargs):
        events.append(('inherited_initialize', args, kwargs))
        self.model = model
    monkeypatch.setattr(base.FactoredOpponent, '__init__', initialize)
    monkeypatch.setattr(base.FactoredOpponent, 'close', lambda self: events.append('closed'))
    monkeypatch.setattr(opt, 'authenticate_runtime', lambda module: runtime_receipt(tmp_path))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.nn.modules', SimpleNamespace(module=runtime))
    if eligibility_failure:
        model.blocks.layers[0].norm_first = False
    return torch, model, events


def test_constructor_receipt_without_model_or_native_execution(tmp_path, monkeypatch):
    torch, model, events = constructor_stubs(tmp_path, monkeypatch)
    opponent = opt.PreboundOpponent('synthetic-folder', boundary='synthetic-boundary')
    assert events == [('inherited_initialize', ('synthetic-folder',), {'boundary': 'synthetic-boundary'})]
    assert opponent.model is model and torch._transformer_encoder_layer_fwd.calls == []
    receipt = opponent.eligibility_receipt
    assert opt.validate_eligibility_receipt(receipt) is receipt
    assert receipt['implementation_id'] == opt.IMPLEMENTATION_ID
    assert receipt['original_parameter_references'] is True
    assert receipt['contract']['native_kernel_changed'] is False


def test_constructor_failure_closes_owned_native(tmp_path, monkeypatch):
    _, _, events = constructor_stubs(tmp_path, monkeypatch, eligibility_failure=True)
    with pytest.raises(ValueError): opt.PreboundOpponent('synthetic')
    assert events[-1] == 'closed'


@pytest.mark.parametrize('change', ['status', 'version', 'runtime_hash', 'layer_count', 'kernel',
                                   'order', 'parameter', 'scalar', 'contract', 'extra', 'bool_alias', 'float_alias'])
def test_receipt_rejects_contract_changes(tmp_path, monkeypatch, change):
    constructor_stubs(tmp_path, monkeypatch)
    receipt = deepcopy(opt.PreboundOpponent('synthetic').eligibility_receipt)
    if change == 'status': receipt['status'] = 'untested'
    elif change == 'version': receipt['torch_version'] = '2.5.0'
    elif change == 'runtime_hash': receipt['runtime_inputs'][0]['sha256'] = '0'*64
    elif change == 'layer_count': receipt['layer_count'] = 5
    elif change == 'kernel': receipt['kernel'] = 'other'
    elif change == 'order': receipt['argument_order'][0:2] = reversed(receipt['argument_order'][0:2])
    elif change == 'parameter': receipt['layer_arguments'][0]['parameters'][0]['dtype'] = 'float16'
    elif change == 'scalar': receipt['layer_arguments'][0]['scalars']['num_heads'] = 4
    elif change == 'contract': receipt['contract']['jit'] = True
    elif change == 'bool_alias': receipt['contract']['jit'] = 0
    elif change == 'float_alias': receipt['layer_arguments'][0]['scalars']['num_heads'] = 8.0
    else: receipt['extra'] = True
    with pytest.raises(ValueError): opt.validate_eligibility_receipt(receipt)


def test_runtime_authentication_binds_every_source_before_model_use(tmp_path, monkeypatch):
    import hashlib
    torch = Runtime()
    torch.__file__ = str(tmp_path/'__init__.py')
    expected = {}
    for relative in opt.RUNTIME_HASHES:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = relative.encode()
        path.write_bytes(data)
        expected[relative] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(opt, 'RUNTIME_HASHES', expected)
    assert [r['relative_path'] for r in opt.authenticate_runtime(torch)] == list(expected)
    (tmp_path/'nn/modules/transformer.py').write_text('changed')
    with pytest.raises(ValueError, match='runtime source changed'): opt.authenticate_runtime(torch)
    torch.__version__ = 'other'
    with pytest.raises(ValueError, match='Torch version'): opt.authenticate_runtime(torch)


def test_argument_tuple_matches_pinned_source_ast_exactly():
    import hashlib
    path = Path('C:/Users/leoba/anaconda3/Lib/site-packages/torch/nn/modules/transformer.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == opt.RUNTIME_HASHES['nn/modules/transformer.py']
    tree = ast.parse(path.read_text(encoding='utf-8'))
    layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TransformerEncoderLayer')
    call = next(n for n in ast.walk(layer) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == '_transformer_encoder_layer_fwd')
    assert len(call.args) == 20 and ast.dump(call.args[0]) == ast.dump(ast.Name(id='src', ctx=ast.Load()))
    class Replace(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'self': return ast.Name(id='layer', ctx=ast.Load())
            if node.id in ('merged_mask', 'mask_type'): return ast.Constant(value=None)
            return node
    expected = ast.Tuple(elts=[Replace().visit(deepcopy(n)) for n in call.args[1:]], ctx=ast.Load())
    actual = ast.parse(textwrap.dedent(inspect.getsource(opt._layer_arguments))).body[0].body[0].value
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def test_only_transformer_scope_and_argument_change_in_infer_ast():
    original = ast.parse(textwrap.dedent(inspect.getsource(base.FactoredOpponent._infer)))
    actual = ast.parse(textwrap.dedent(inspect.getsource(opt.PreboundOpponent._infer)))
    call = next(n for n in ast.walk(actual) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'infer_transformer')
    assert ast.dump(call.args[0]) == ast.dump(ast.Name(id='self', ctx=ast.Load()))
    call.args[0] = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='model', ctx=ast.Load())
    assert ast.dump(actual, include_attributes=False) == ast.dump(original, include_attributes=False)
    for label, target in opt.fine_targets().items():
        if label == 'infer_transformer': assert target is opt.infer_transformer
        else: assert target is base.fine_targets()[label]
    assert len({id(f.__code__) for f in opt.fine_targets().values()}) == 9
    for name in ('choose', 'exchange', 'inspect', 'native_reference'):
        assert getattr(opt.PreboundOpponent, name) is getattr(base.FactoredOpponent, name)
