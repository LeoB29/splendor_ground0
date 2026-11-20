"""Model-free qualification of the one fixed compilation option and refusal."""
import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_compiled_20261105 import backend as b, compilation as k, common as c


def expected_math():
    block = ["aten::layer_norm", "aten::linear", "aten::bmm", "aten::div", "aten::softmax",
             "aten::bmm", "aten::linear", "aten::add", "aten::layer_norm", "aten::linear",
             "aten::gelu", "aten::linear", "aten::add"]
    return ["aten::linear", "aten::add"]+block*6+["aten::layer_norm", "aten::linear", "aten::linear", "aten::tanh", "aten::cat"]


class Value:
    def __init__(self, typ="Tensor"): self.typ = typ
    def type(self): return self.typ


class Node:
    def __init__(self, kind, inputs, output=None, attrs=None):
        self.k = kind; self.ins = inputs; self.out = output or Value(); self.attrs = attrs or {}
    def kind(self): return self.k
    def inputs(self): return iter(self.ins)
    def outputs(self): return iter([self.out])
    def blocks(self): return iter([])
    def attributeNames(self): return list(self.attrs)
    def kindOf(self, name): return self.attrs[name][0]
    def s(self, name): return self.attrs[name][1]
    def i(self, name): return self.attrs[name][1]
    def f(self, name): return self.attrs[name][1]
    def t(self, name): return self.attrs[name][1]


class Graph:
    def __init__(self, kinds=None, *, mangle=0):
        self.root = Value("__torch__.example.___torch_mangle_"+str(mangle)+".Module")
        self.nodes_list = []; previous = self.root
        for kind in (expected_math() if kinds is None else kinds):
            node = Node(kind, [previous]); self.nodes_list.append(node); previous = node.out
        self.last = previous
    def inputs(self): return iter([self.root])
    def nodes(self): return iter(self.nodes_list)
    def outputs(self): return iter([self.last])
    def __str__(self): return "model-free graph: "+", ".join(node.kind() for node in self.nodes_list)


class FakeTensor:
    dtype = "torch.float32"; shape = (8,31,48); device = "cpu"
    def __init__(self, raw=b"original-f32-bytes"):
        self.raw = raw; self.moves = []
    def to(self, device, *, non_blocking):
        self.moves.append((str(device), non_blocking)); return self
    def detach(self): return self
    def contiguous(self): return self
    def numpy(self): return self
    def tobytes(self, *, order): assert order == "C"; return self.raw


def fake_torch(*, failure=None, failure_at="trace", frozen_kinds=None):
    calls = []; tensor = FakeTensor()
    def zeros(shape, *, dtype):
        calls.append(("zeros",shape,dtype)); return tensor
    def trace(module, example, **kwargs):
        calls.append(("trace",module,example,kwargs))
        if failure is not None and failure_at == "trace": raise failure
        return SimpleNamespace(inlined_graph=Graph(mangle=1))
    def freeze(traced, **kwargs):
        calls.append(("freeze",traced,kwargs))
        if failure is not None and failure_at == "freeze": raise failure
        return SimpleNamespace(inlined_graph=Graph(frozen_kinds,mangle=47))
    return SimpleNamespace(float32="torch.float32",zeros=zeros,jit=SimpleNamespace(trace=trace,freeze=freeze)),calls,tensor


def run_compile(tmp_path, monkeypatch, *, failure=None, failure_at="trace", frozen_kinds=None, closure_error=None):
    monkeypatch.setattr(c,"OUT",tmp_path)
    torch,calls,tensor=fake_torch(failure=failure,failure_at=failure_at,frozen_kinds=frozen_kinds)
    closure=[]; warnings=[]
    def verify():
        closure.append(True)
        if closure_error: raise closure_error
    folder=tmp_path/"candidate_compilation"
    def execute():
        return k.compile_fixed_b8(torch,"primitive","privateuseone:0",folder,
            checked=lambda fn:fn(),warning_records=warnings,weights_sha256="a"*64,verify_weights=verify)
    return execute,folder,calls,tensor,closure,warnings


def test_complete_source_equivalence_no_model_import():
    proof=b.source_equivalence()
    assert proof['status']=='source_equivalence_verified' and proof['models_loaded']==0
    assert proof['observer'] is False and len(proof['unchanged_ast_sha256'])==12
    assert 'complete_module_excluding_compilation_choice' in proof['unchanged_ast_sha256']


def test_one_fixed_option_and_artifact_binding(tmp_path,monkeypatch):
    execute,folder,calls,tensor,closure,_=run_compile(tmp_path,monkeypatch)
    frozen,returned=execute(); receipt=json.loads((folder/'compilation.json').read_text())
    assert [x[0] for x in calls]==['zeros','trace','freeze']
    assert calls[0][1:]==((8,31,48),'torch.float32')
    assert calls[1][-1]==dict(check_trace=False,strict=True)
    assert calls[2][-1]==dict(optimize_numerics=False)
    assert tensor.moves==[('privateuseone:0',False)] and closure==[True]
    assert receipt['compilation_fingerprint']==returned['compilation_fingerprint']
    assert receipt['binding']['options']==k.OPTIONS and receipt['original_weights_unchanged'] is True
    assert receipt['exact_outputs_qualified'] is False and receipt['timed_panel'] is False
    assert returned['receipt']==c.artifact(folder/'compilation.json')
    for item in receipt['graphs'].values(): c.verify([item])
    assert not (tmp_path/'candidate').exists()


def test_fingerprint_ignores_generated_names_but_binds_edges_and_attributes():
    one=k.graph_description(Graph(mangle=1)); two=k.graph_description(Graph(mangle=999))
    assert one==two
    changed=Graph(); changed.nodes_list[3].ins=[changed.root]
    assert k.graph_description(changed)['fingerprint']!=one['fingerprint']
    changed=Graph(); changed.nodes_list[0].attrs={'scale':('f',1.0)}
    assert k.graph_description(changed)['fingerprint']!=one['fingerprint']


def test_graph_fingerprint_binds_full_tensor_constant_bytes():
    graph=Graph(['prim::Constant']); graph.nodes_list[0].attrs={'value':('t',FakeTensor(b'left'))}
    first=k.graph_description(graph)
    graph.nodes_list[0].attrs={'value':('t',FakeTensor(b'right'))}
    assert first['fingerprint']!=k.graph_description(graph)['fingerprint']


def test_shape_constant_folding_allowed_without_math_change():
    traced=k.graph_description(Graph(['aten::size','prim::NumToTensor','aten::mul','aten::Int']+expected_math()))
    frozen=k.graph_description(Graph(expected_math()))
    assert k.validate_graph_pair(traced,frozen)


@pytest.mark.parametrize('kinds',[
    ['aten::_transformer_encoder_layer_fwd'],['prim::PythonOp'],['aten::matmul'],
])
def test_fused_or_different_operators_rejected(kinds):
    with pytest.raises(k.CompilationRejected,match='Different or unsupported'):
        k.graph_description(Graph(kinds))


def test_changed_math_order_or_missing_layer_rejected():
    original=k.graph_description(Graph()); changed=copy.deepcopy(original)
    changed['math_sequence'][0:2]=reversed(changed['math_sequence'][0:2])
    with pytest.raises(k.CompilationRejected,match='changed mathematical'):k.validate_graph_pair(original,changed)
    original['math_sequence'].pop()
    with pytest.raises(k.CompilationRejected,match='six-block'):k.validate_graph_pair(original,original)


@pytest.mark.parametrize('error',[
    NotImplementedError('operator privateuseone not implemented'),
    RuntimeError('Could not run aten::example with DirectML'),
    ValueError('unsupported GPU operation/CPU fallback: test'),
])
def test_known_candidate_rejection_preserves_original_cause_and_weights(tmp_path,monkeypatch,error):
    execute,folder,calls,_,closure,_=run_compile(tmp_path,monkeypatch,failure=error)
    with pytest.raises(k.CompilationRejected) as caught:execute()
    assert caught.value.__cause__ is error and [x[0] for x in calls]==['zeros','trace']
    saved=json.loads((folder/'compilation_failure.json').read_text())
    assert saved['message']==str(error) and saved['exception_type']==type(error).__name__
    assert saved['traceback'] and closure==[True]
    assert k.classify_saved_compile_failure(saved)=='candidate_rejected'
    assert not (folder/'compilation.json').exists()


@pytest.mark.parametrize('error',[
    MemoryError('memory exhausted'),OSError('disk full'),TimeoutError('deadline expired'),
    RuntimeError('DirectML unsupported operation: out of memory'),
    RuntimeError('device removed'),RuntimeError('unexpected unrelated internal error'),
    KeyboardInterrupt(),
])
def test_resource_and_unclassified_errors_remain_technical(tmp_path,monkeypatch,error):
    execute,folder,_,_,closure,_=run_compile(tmp_path,monkeypatch,failure=error)
    with pytest.raises(type(error)) as caught:execute()
    assert caught.value is error and closure==[True]
    saved=json.loads((folder/'compilation_failure.json').read_text())
    assert saved['classification']=='technical_failure'
    assert k.classify_saved_compile_failure(saved)=='technical_failure'


def test_failed_original_weight_closure_prevents_negative_result(tmp_path,monkeypatch):
    original=NotImplementedError('unsupported graph'); failure=RuntimeError('weight bytes changed')
    execute,folder,_,_,closure,_=run_compile(tmp_path,monkeypatch,failure=original,closure_error=failure)
    with pytest.raises(RuntimeError) as caught:execute()
    assert caught.value is failure and caught.value.__cause__ is original
    saved=json.loads((folder/'compilation_failure.json').read_text())
    assert saved['classification']=='technical_failure' and saved['original_weights_unchanged'] is False
    with pytest.raises(ValueError):k.classify_saved_compile_failure(saved)


def test_failed_graph_check_retains_both_graphs_and_dedicated_code(tmp_path,monkeypatch):
    execute,folder,_,_,closure,_=run_compile(tmp_path,monkeypatch,frozen_kinds=expected_math()[:-1])
    with pytest.raises(k.CompilationRejected):execute()
    saved=json.loads((folder/'compilation_failure.json').read_text())
    assert saved['phase']=='graph_check' and saved['exception_code']=='changed_math_sequence'
    assert set(saved['graphs'])=={'traced','frozen'} and closure==[True]
    assert k.classify_saved_compile_failure(saved)=='candidate_rejected'
    changed=copy.deepcopy(saved);changed['exception_code']='invented'
    with pytest.raises(ValueError):k.classify_saved_compile_failure(changed)


def test_same_directory_cannot_overwrite_compilation_or_retry(tmp_path,monkeypatch):
    execute,folder,calls,_,_,_=run_compile(tmp_path,monkeypatch)
    execute(); preserved=(folder/'compilation.json').read_bytes(); count=len(calls)
    with pytest.raises(FileExistsError):execute()
    assert (folder/'compilation.json').read_bytes()==preserved
    assert len(calls)==count and not (folder/'compilation_failure.json').exists()


def test_graph_inventory_and_refusal_do_not_hide_warning_evidence(tmp_path,monkeypatch):
    execute,folder,_,_,_,records=run_compile(tmp_path,monkeypatch)
    records.append({'category':'prior','message':'before compilation'})
    execute(); saved=json.loads((folder/'compilation.json').read_text())
    assert saved['warnings']==[]
    assert records==[{'category':'prior','message':'before compilation'}]


def test_actual_torch_ir_int_list_ivalue_is_typed_and_fingerprint_bound():
    import torch
    source='graph():\n %shape : int[] = prim::Constant[value=[256]]()\n return (%shape)'
    graph=torch._C.parse_ir(source);node=next(graph.nodes())
    assert node.kindOf('value')=='ival'
    actual=k.graph_description(graph)
    value=actual['body']['nodes'][0]['attributes']['value']
    assert value==dict(kind='ival',value=dict(type='list',items=[dict(type='int',value=256)]))
    node.ival_('value',[257])
    assert k.graph_description(graph)['fingerprint']!=actual['fingerprint']
    assert k.graph_description(torch._C.parse_ir(source))==actual


def test_actual_torch_ir_float_list_retains_signed_zero_and_all_float_bits():
    import torch
    graph=torch._C.parse_ir('graph():\n %values : float[] = prim::Constant[value=[0., -0., 1., -2.5]]()\n return (%values)')
    node=next(graph.nodes());assert node.kindOf('value')=='ival'
    values=[0.,-0.,1.,-2.5];description=k.graph_description(graph)
    items=description['body']['nodes'][0]['attributes']['value']['value']['items']
    assert items==[dict(type='float',ieee754_binary64_hex=struct.pack('<d',x).hex()) for x in values]
    assert items[0]!=items[1]
    # Actual Torch IValue setter exercises nonfinite literals without model math.
    nan=struct.unpack('<d',bytes.fromhex('010000000000f87f'))[0]
    node.ival_('value',[float('inf'),float('-inf'),nan])
    special=k.graph_description(graph)
    literal=special['body']['nodes'][0]['attributes']['value']['value']['items']
    assert literal==[dict(type='float',ieee754_binary64_hex=struct.pack('<d',x).hex()) for x in node.ival('value')]
    assert json.loads(json.dumps(special,allow_nan=False))==special
    assert special['fingerprint']!=description['fingerprint']


def test_ivalue_literal_types_remain_distinct_and_opaque_values_are_rejected():
    assert k._canonical_ivalue(None)==dict(type='none')
    assert k._canonical_ivalue(True)!=k._canonical_ivalue(1)
    assert k._canonical_ivalue(1)!=k._canonical_ivalue(1.)
    assert k._canonical_ivalue([1,'x'])!=k._canonical_ivalue((1,'x'))
    assert k._canonical_ivalue(([1],False))==dict(type='tuple',items=[dict(type='list',items=[dict(type='int',value=1)]),dict(type='bool',value=False)])
    for value in (object(),{'not_admitted':'mapping'},1+2j):
        with pytest.raises(k.CompilationRejected) as caught:k._canonical_ivalue(value)
        assert caught.value.code=='unsupported_graph_ivalue'


def test_actual_torch_ir_device_and_none_constants_keep_existing_routes():
    import torch
    device=torch._C.parse_ir('graph():\n %device : Device = prim::Constant[value="cpu"]()\n return (%device)')
    assert k.graph_description(device)['body']['nodes'][0]['attributes']['value']==dict(kind='s',value='cpu')
    none=torch._C.parse_ir('graph():\n %none : NoneType = prim::Constant()\n return (%none)')
    assert k.graph_description(none)['body']['nodes'][0]['attributes']=={}
