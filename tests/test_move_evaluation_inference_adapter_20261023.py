"""Symbolic-only inference contract; no Torch kernels/models/native jobs."""
import ast
from copy import deepcopy
import inspect
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
from tools.move_evaluation_reuse_20261023.greedy_adapter import GreedyOpponent
from tools.move_evaluation_inference_20261023 import inference as new


class Symbol:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def reshape(self, *shape):
        self.owner.events.append(('reshape', self.name, shape))
        return Symbol(self.owner, 'x')

    def __add__(self, other):
        self.owner.events.append(('add', self.name, other.name))
        return Symbol(self.owner, 'project_plus_identity')

    def __getitem__(self, index):
        assert index == (slice(None), 0)
        self.owner.events.append(('pool', self.name, 'all', 0))
        return Symbol(self.owner, 'pooled')

    def tanh(self):
        self.owner.events.append(('tanh', self.name))
        return Symbol(self.owner, 'value_tanh')

    def tolist(self):
        self.owner.events.append(('tolist', self.name))
        return deepcopy(self.owner.output)


class Finite:
    def __init__(self, owner, value):
        self.owner, self.value = owner, value

    def all(self):
        self.owner.events.append(('finite_all', self.value.name))
        return Truth(self.owner, self.value.name)


class Truth:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def __bool__(self):
        self.owner.events.append(('finite_bool', self.name))
        return self.name != self.owner.nonfinite


class FakeTorch:
    float32 = object()

    def __init__(self, nonfinite=None, fail=None):
        self.events, self.nonfinite, self.fail = [], nonfinite, fail
        self.output = [[0.0] * 81 + [0.25, -0.25]]
        self.input_copy = None
        self.depth = 0

    def inference_mode(self):
        owner = self
        class Mode:
            def __enter__(self):
                owner.events.append(('enter_inference_mode',))
                owner.depth += 1
            def __exit__(self, kind, value, traceback):
                owner.events.append(('exit_inference_mode', None if kind is None else kind.__name__))
                owner.depth -= 1
        return Mode()

    def tensor(self, tokens, *, dtype):
        assert dtype is self.float32 and self.depth == 1
        self.events.append(('tensor', 'float32', tuple(tokens)))
        self.input_copy = tuple(tokens)
        return Symbol(self, 'tensor')

    def isfinite(self, value):
        assert self.depth == 1
        self.events.append(('isfinite', value.name))
        return Finite(self, value)

    def cat(self, values, *, dim):
        assert self.depth == 1
        self.events.append(('cat', tuple(v.name for v in values), dim))
        return Symbol(self, 'out')

    def model(self):
        def operation(name, output):
            def call(value):
                assert self.depth == 1
                self.events.append((name, value.name))
                if self.fail == name:
                    raise RuntimeError('synthetic ' + name)
                return Symbol(self, output)
            return call
        return SimpleNamespace(identity=Symbol(self, 'identity'),
            project=operation('project', 'projected'), blocks=operation('blocks', 'blocks'),
            norm=operation('norm', 'h'), policy=operation('policy', 'policy'),
            value=operation('value', 'value'))


def run(monkeypatch, method, tokens, batch, **kwargs):
    torch = FakeTorch(**kwargs)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    instance = SimpleNamespace(model=torch.model())
    try:
        result = method(instance, tokens, batch)
    except BaseException as error:
        return torch, None, error
    return torch, result, None


@pytest.mark.parametrize('batch', [1, 2, 8])
def test_original_and_factored_symbolic_operations_are_exact(monkeypatch, batch):
    tokens = [float(i % 7) / 7 for i in range(batch * 31 * 48)]
    prior = deepcopy(tokens)
    old, expected, error = run(monkeypatch, Opponent._infer, tokens, batch)
    actual, result, new_error = run(monkeypatch, new.FactoredOpponent._infer, tokens, batch)
    assert error is new_error is None
    assert result == expected and len(result[0]) == 83
    assert old.events == actual.events
    assert tokens == prior and old.input_copy == actual.input_copy == tuple(prior)
    assert actual.depth == old.depth == 0
    names = [event[0] for event in actual.events]
    assert names == ['enter_inference_mode', 'tensor', 'reshape', 'isfinite', 'finite_all', 'finite_bool',
                     'project', 'add', 'blocks', 'pool', 'norm', 'policy', 'value', 'tanh', 'cat',
                     'isfinite', 'finite_all', 'finite_bool', 'tolist', 'exit_inference_mode']


@pytest.mark.parametrize('tokens,batch', [([], 1), ([0.0]*1488, 0), ([0.0]*1488, 3),
                                        ([0.0]*1487, 1), ([0.0]*1489, 1), (None, 1)])
def test_invalid_shape_preserves_original_failure_before_tensor(monkeypatch, tokens, batch):
    old, _, expected = run(monkeypatch, Opponent._infer, tokens, batch)
    actual, _, error = run(monkeypatch, new.FactoredOpponent._infer, tokens, batch)
    assert type(error) is type(expected) and str(error) == str(expected)
    assert actual.events == old.events == []


@pytest.mark.parametrize('kwargs,message', [({'nonfinite': 'x'}, 'Nonfinite model input'),
    ({'nonfinite': 'out'}, 'Nonfinite model output'), ({'fail': 'blocks'}, 'synthetic blocks'),
    ({'fail': 'policy'}, 'synthetic policy'), ({'fail': 'value'}, 'synthetic value')])
def test_failed_checks_and_model_exceptions_preserve_order_and_cleanup(monkeypatch, kwargs, message):
    tokens = [0.0] * 1488
    old, _, expected = run(monkeypatch, Opponent._infer, tokens, 1, **kwargs)
    actual, _, error = run(monkeypatch, new.FactoredOpponent._infer, tokens, 1, **kwargs)
    assert type(error) is type(expected) and str(error) == str(expected) == message
    assert actual.events == old.events and actual.depth == old.depth == 0


def test_no_input_or_output_cache_or_mutable_alias(monkeypatch):
    tokens = [0.0] * 1488
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    instance = SimpleNamespace(model=torch.model())
    first = new.FactoredOpponent._infer(instance, tokens, 1)
    first[0][0] = 99
    tokens[0] = 23
    second = new.FactoredOpponent._infer(instance, tokens, 1)
    assert second == torch.output and second != first
    assert torch.input_copy[0] == 23
    assert len([e for e in torch.events if e[0] == 'tensor']) == 2
    assert set(vars(instance)) == {'model'}


def function_node(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]


def dump(node):
    return ast.dump(node, include_attributes=False)


class ModelName(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'self' and node.attr == 'model':
            return ast.copy_location(ast.Name(id='model', ctx=ast.Load()), node)
        return self.generic_visit(node)


def returned(fn):
    node = function_node(fn)
    assert len(node.body) == 1 and isinstance(node.body[0], ast.Return)
    return node.body[0].value


def replace_name(node, name, expression):
    class Replace(ast.NodeTransformer):
        def visit_Name(self, value):
            return deepcopy(expression) if value.id == name and isinstance(value.ctx, ast.Load) else value
    return Replace().visit(deepcopy(node))


def test_factored_expression_ast_is_original_arithmetic_and_checks():
    original = function_node(Opponent._infer)
    assert isinstance(original.body[0], ast.Import) and original.body[0].names[0].name == 'torch'
    assert dump(function_node(new.infer_input_shape).body[0]) == dump(original.body[1])
    body = original.body[2].body
    assert dump(returned(new.infer_tensor_input)) == dump(body[0].value)
    assert dump(function_node(new.infer_input_finite).body[0]) == dump(body[1])
    h = replace_name(returned(new.infer_transformer), 'h', returned(new.infer_project))
    h = replace_name(returned(new.infer_pool_norm), 'h', h)
    assert dump(h) == dump(ModelName().visit(deepcopy(body[2].value)))
    assert dump(returned(new.infer_heads)) == dump(ModelName().visit(deepcopy(body[3].value)))
    assert dump(function_node(new.infer_output_finite).body[0]) == dump(body[4])
    assert dump(returned(new.infer_output_list)) == dump(body[5].value)
    factored = function_node(new.FactoredOpponent._infer)
    scopes = [n for n in ast.walk(factored) if isinstance(n, ast.With)]
    assert len(scopes) == 1
    assert dump(scopes[0].items[0].context_expr) == dump(original.body[2].items[0].context_expr)
    calls = [n.func.id for n in ast.walk(factored) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls == list(new.fine_targets())


def test_distinct_timer_targets_and_only_infer_override():
    targets = new.fine_targets()
    expected = ('input_shape', 'tensor_input', 'input_finite', 'project', 'transformer',
                'pool_norm', 'heads', 'output_finite', 'output_list')
    assert tuple(targets) == tuple('infer_' + name for name in expected)
    assert len({id(fn.__code__) for fn in targets.values()}) == 9
    assert all(fn.__name__ == label for label, fn in targets.items())
    targets.clear()
    assert len(new.fine_targets()) == 9
    for name in ('__init__', 'choose', 'exchange', 'inspect', 'native_reference', 'close'):
        assert getattr(new.FactoredOpponent, name) is getattr(GreedyOpponent, name)
    tree = ast.parse(inspect.getsource(new))
    direct_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(direct_imports) == 1 and direct_imports[0].module.endswith('greedy_adapter')
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)] == ['_infer']


def test_selected_timer_reconciles_nine_helper_counts_without_models(monkeypatch):
    from tools.move_evaluation_timing_20261023.monitor import SelectedTimer
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    instance = SimpleNamespace(model=torch.model())
    targets = {'scientific_body': new.FactoredOpponent._infer, **new.fine_targets()}
    timer = SelectedTimer(targets)
    result, measurement = timer.run(new.FactoredOpponent._infer, instance, [0.0] * 1488, 1)
    assert result == torch.output and measurement['balanced']
    assert measurement['exclusive_sum_ns'] == measurement['root_region_ns']
    assert {r['label']: r['calls'] for r in measurement['rows']} == dict.fromkeys(targets, 1)
    assert all(row['returns'] == 1 and row['unwinds'] == 0 for row in measurement['rows'])
