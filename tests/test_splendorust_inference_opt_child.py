"""Model-free contract tests for the one frozen inference candidate."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.splendorust_inference_opt_20261025 import graph, worker


def fixture(fixture_id='one', batch=1, value=0.):
    return {'fixture_id': fixture_id, 'batch': batch, 'tokens': [value] * (batch * 31 * 48)}


def fixtures():
    return [fixture('f' + str(index), batch, float(index))
            for index, batch in enumerate([1] * 7 + [2] * 2 + [8] * 2)]


def outputs(batch, value=0.):
    return [[float(value)] * 83 for _ in range(batch)]


def test_bitwise_gate_rejects_signed_zero_despite_equal_python_values():
    row = graph.compare_outputs(fixture(), outputs(1, 0.), outputs(1, -0.))
    assert row['exact_float_equal'] is True
    assert row['float32_bits_equal'] is False
    assert row['eager_float32_sha256'] != row['candidate_float32_sha256']


def test_bitwise_and_exact_float_checks_are_separate():
    # These float64 Python numbers round to the same float32 bit pattern.
    # Actual model tolist outputs should never introduce this discrepancy.
    row = graph.compare_outputs(fixture(), outputs(1, 1.), outputs(1, 1. + 1e-12))
    assert row['exact_float_equal'] is False
    assert row['float32_bits_equal'] is True


@pytest.mark.parametrize('bad', [[], [[0.] * 82], [[float('nan')] * 83], [[0] * 83]])
def test_output_shape_type_and_finiteness_are_not_silently_coerced(bad):
    with pytest.raises(ValueError):
        graph.float32_bytes(bad, 1)


def test_parity_requires_fixed_bounded_batch_coverage():
    rows = fixtures()
    graph.validate_parity_fixtures(rows)
    with pytest.raises(ValueError, match='eleven'):
        graph.validate_parity_fixtures(rows[:-1])
    rows[-1] = fixture('last', 1)
    with pytest.raises(ValueError, match='coverage'):
        graph.validate_parity_fixtures(rows)


def test_parity_rejects_duplicate_ids_nonfinite_values_and_unapproved_fields():
    for mutate in (
        lambda rows: rows[-1].update(fixture_id=rows[0]['fixture_id']),
        lambda rows: rows[0]['tokens'].__setitem__(0, float('inf')),
        lambda rows: rows[0].update(hidden_snapshot={}),
    ):
        rows = fixtures(); mutate(rows)
        with pytest.raises(ValueError): graph.validate_parity_fixtures(rows)


class Tensor:
    def __init__(self, values, shape, finite=True):
        self.values, self.shape, self.finite = values, shape, finite
        self.dtype, self.device = 'float32', SimpleNamespace(type='cpu')
    def reshape(self, *shape): self.shape = shape; return self
    def tolist(self): return self.values


def fake_torch():
    return SimpleNamespace(float32='float32', inference_mode=lambda:nullcontext(),
        tensor=lambda values,dtype:Tensor(values, (len(values),), all(abs(x) != float('inf') and x == x for x in values)),
        isfinite=lambda tensor:SimpleNamespace(all=lambda:tensor.finite))


def test_candidate_preserves_validation_and_conversion_outside_graph(monkeypatch):
    torch = fake_torch(); monkeypatch.setitem(sys.modules, 'torch', torch)
    called = []
    def compiled(x):
        called.append(x)
        return Tensor(outputs(x.shape[0], 0.25), (x.shape[0], 83))
    forward = graph.FrozenForward(compiled, {})
    for batch in (1, 2, 8):
        assert forward.infer(None, fixture(batch=batch)['tokens'], batch) == outputs(batch, .25)
        assert called[-1].shape == (batch, 31, 48) and called[-1].dtype == 'float32'
    with pytest.raises(ValueError, match='Malformed public entity tokens'):
        forward.infer(None, [], 1)
    bad = fixture()['tokens']; bad[0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite model input'):
        forward.infer(None, bad, 1)
    assert len(called) == 3


def test_graph_errors_and_nonfinite_output_never_fall_back(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch())
    def raises(x): raise RuntimeError('frozen graph failure')
    with pytest.raises(RuntimeError, match='frozen graph failure'):
        graph.FrozenForward(raises, {}).infer(None, fixture()['tokens'], 1)
    with pytest.raises(ValueError, match='Nonfinite model output'):
        graph.FrozenForward(lambda x:Tensor(outputs(1), (1,83), False), {}).infer(None, fixture()['tokens'], 1)


def graph_build_fixture(monkeypatch, tmp_path):
    calls, saved = [], []
    module = SimpleNamespace(training=False)
    model = SimpleNamespace(has_history=False, modules=lambda:[module])
    torch = fake_torch()
    torch.get_num_threads = lambda:1
    torch.get_num_interop_threads = lambda:1
    torch.zeros = lambda shape,**kwargs:Tensor([], shape)
    torch.__version__, torch.__file__ = 'mock-torch', str(tmp_path/'torch.py')
    torch.version = SimpleNamespace(git_version='mock-git')
    torch._C = SimpleNamespace(__file__=str(tmp_path/'torch_core.pyd'))
    torch.__config__ = SimpleNamespace(show=lambda:'mock build')
    torch.backends = SimpleNamespace(mha=SimpleNamespace(get_fastpath_enabled=lambda:True),
                                    mkldnn=SimpleNamespace(enabled=True))
    class Compiled:
        inlined_graph = 'exact mock graph'
        def __call__(self, x): calls.append(('warm',x.shape)); return Tensor([], (x.shape[0],83))
    def trace(forward, example, **kwargs):
        calls.append(('trace',example.shape,kwargs)); return 'traced'
    def freeze(traced, **kwargs):
        calls.append(('freeze',traced,kwargs)); return Compiled()
    torch.jit = SimpleNamespace(trace=trace,freeze=freeze)
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'numpy',SimpleNamespace(__version__='mock-numpy'))
    monkeypatch.setattr(graph,'_make_forward',lambda *args:'exact tensor wrapper')
    monkeypatch.setattr(graph,'_weight_digest',lambda *args:calls.append(('weights',)) or 'same-weights')
    return model,torch,calls,saved,lambda path,value:saved.append((path.name,deepcopy(value)))


def test_one_trace_one_freeze_false_only_zero_warms_and_no_serialized_weights(monkeypatch,tmp_path):
    model,torch,calls,saved,save = graph_build_fixture(monkeypatch,tmp_path)
    forward = graph.build_graph(model,tmp_path,save)
    assert calls == [('weights',),('trace',(1,31,48),{'check_trace':False,'strict':True}),
                     ('freeze','traced',{'optimize_numerics':False}),
                     ('warm',(1,31,48)),('warm',(2,31,48)),('warm',(8,31,48)),('weights',)]
    identity = saved[0][1]
    assert saved[0][0] == 'graph_identity.json'
    assert identity['weights_before_sha256'] == identity['weights_after_sha256']
    assert identity['serialized_graph_saved'] is False
    assert identity['numpy_version'] == 'mock-numpy'
    assert identity['graph_ir_sha256'] == hashlib.sha256((tmp_path/'graph_ir.txt').read_bytes()).hexdigest()
    assert forward.identity == identity and list(tmp_path.iterdir()) == [tmp_path/'graph_ir.txt']


def test_build_failure_preserves_phase_warning_and_does_not_try_another_graph(monkeypatch,tmp_path):
    import warnings
    model,torch,calls,saved,save = graph_build_fixture(monkeypatch,tmp_path)
    def failing_freeze(*args,**kwargs):
        warnings.warn('recorded frozen graph warning')
        raise RuntimeError('exact freeze failure')
    torch.jit.freeze = failing_freeze
    with pytest.raises(RuntimeError, match='exact freeze failure'):
        graph.build_graph(model,tmp_path,save)
    assert len([item for item in calls if item[0]=='trace']) == 1
    assert not any(item[0]=='warm' for item in calls)
    assert saved[0][0] == 'graph_build_error.json'
    assert saved[0][1]['phase'] == 'freeze_once_without_numerical_optimizations'
    assert saved[0][1]['trace_warnings'][0]['message'] == 'recorded frozen graph warning'


def test_weight_or_state_mutation_fails_closed(monkeypatch,tmp_path):
    model,torch,calls,saved,save = graph_build_fixture(monkeypatch,tmp_path)
    hashes = iter(['before','after'])
    monkeypatch.setattr(graph,'_weight_digest',lambda *args:next(hashes))
    with pytest.raises(ValueError, match='modified original weights'):
        graph.build_graph(model,tmp_path,save)
    assert saved[-1][0] == 'graph_build_error.json'


def test_base_binding_preserves_callback_wrapper_and_original_eager_for_parity():
    events = []
    class Base:
        def _infer(self,tokens,batch): return 'eager'
    class Search(Base):
        def _infer(self,tokens,batch):
            out = Base._infer(self,tokens,batch)
            events.append(('callback',out))
            return out
    model = object(); first,second = Search(),Search(); first.model=second.model=model
    original = Base._infer
    forward = SimpleNamespace(infer=lambda *args:'compiled')
    binding = graph.InferenceBinding().install({'search':first,'reference':second},forward,owner=Base)
    try:
        assert first._infer([],1) == 'compiled' and events == [('callback','compiled')]
        assert binding.original(first,[],1) == 'eager'
        with pytest.raises(ValueError,match='Concurrent or repeated'):
            graph.InferenceBinding().install({'search':first},forward,owner=Base)
        alien = Search(); alien.model=model
        with pytest.raises(ValueError,match='Unqualified adapter'):
            alien._infer([],1)
    finally:
        binding.close()
    assert Base._infer is original
    binding.close()


def context(tmp_path):
    return {'folder':tmp_path,'graph_parity_passed':False,'graph_parity_attempted':False,
        'adapters':{'search':object()},'inference_binding':SimpleNamespace(
            original=lambda adapter,tokens,batch:outputs(batch,.25)),
        'frozen_forward':SimpleNamespace(identity={'fixed':'identity'},
            infer=lambda adapter,tokens,batch:outputs(batch,.25))}


def test_parity_persists_full_values_before_admitting_decisions(monkeypatch,tmp_path):
    ctx = context(tmp_path); saved=[]
    def save(path,value):
        assert ctx['graph_parity_passed'] is False
        saved.append((path.name,deepcopy(value)))
    monkeypatch.setattr(worker.frozen,'_save',save)
    inputs=fixtures(); result=worker.handle(ctx,{'operation':'graph_parity','fixtures':inputs})
    assert result['passed'] is True and result['fixtures_count']==11
    assert result['fixture_payload_sha256']==graph.digest(inputs)
    assert saved[0][0]=='graph_parity.json' and saved[0][1]['summary']==result
    assert saved[0][1]['fixtures']==inputs and len(saved[0][1]['evidence'])==11
    assert saved[0][1]['evidence'][0]['eager_outputs']==outputs(1,.25)
    assert ctx['graph_parity_passed'] is True
    with pytest.raises(ValueError,match='cannot be retried'):
        worker.handle(ctx,{'operation':'graph_parity','fixtures':inputs})


def test_mismatched_bits_are_preserved_and_block_all_decisions(monkeypatch,tmp_path):
    ctx=context(tmp_path); saved=[]
    ctx['inference_binding'].original=lambda adapter,tokens,batch:outputs(batch,0.)
    ctx['frozen_forward'].infer=lambda adapter,tokens,batch:outputs(batch,-0.)
    monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append(deepcopy(value)))
    result=worker.handle(ctx,{'operation':'graph_parity','fixtures':fixtures()})
    assert result['passed'] is False and result['status']=='failed'
    assert saved[0]['evidence'][0]['comparison']['float32_bits_equal'] is False
    for request in ({'player':'search','observer':{}},{'operation':'entrypoint_checks'}):
        with pytest.raises(ValueError,match='parity first'): worker.handle(ctx,request)


def test_partial_parity_exception_preserves_eager_values_before_worker_failure(monkeypatch,tmp_path):
    ctx=context(tmp_path); saved=[]
    def failure(*args): raise RuntimeError('candidate exact failure')
    ctx['frozen_forward'].infer=failure
    monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append(deepcopy(value)))
    with pytest.raises(RuntimeError,match='candidate exact failure'):
        worker.handle(ctx,{'operation':'graph_parity','fixtures':fixtures()})
    assert saved[0]['summary']['status']=='error'
    assert saved[0]['evidence'][0]['eager_outputs']==outputs(1,.25)
    assert saved[0]['evidence'][0]['candidate_outputs'] is None
    assert ctx['graph_parity_attempted'] is True and ctx['graph_parity_passed'] is False


def test_candidate_decisions_and_all52_entrypoint_callbacks_use_same_original_science(monkeypatch,tmp_path):
    ctx=context(tmp_path); ctx['graph_parity_passed']=True; seen=[]
    previous=worker.frozen.handle
    def original(context,request):
        if request=={'operation':'entrypoint_checks'}:
            return [worker.frozen.handle(context,{'case':index}) for index in range(52)]
        seen.append(request); return request
    monkeypatch.setattr(worker,'_ORIGINAL_HANDLE',original)
    result=worker.handle(ctx,{'operation':'entrypoint_checks'})
    assert result==seen==[{'case':index} for index in range(52)]
    assert worker.frozen.handle is previous
    request={'player':'search','observer':{}}
    assert worker.handle(ctx,request) is request


def test_entrypoint_failure_restores_recursive_binding(monkeypatch,tmp_path):
    ctx=context(tmp_path); ctx['graph_parity_passed']=True
    previous=worker.frozen.handle
    def original(context,request):
        if request=={'operation':'entrypoint_checks'}:
            return worker.frozen.handle(context,{'case':0})
        raise RuntimeError('counterexample failure')
    monkeypatch.setattr(worker,'_ORIGINAL_HANDLE',original)
    with pytest.raises(RuntimeError,match='counterexample failure'):
        worker.handle(ctx,{'operation':'entrypoint_checks'})
    assert worker.frozen.handle is previous


def test_initialize_uses_corrected_initializer_once_and_adds_only_backend_ready_key(monkeypatch,tmp_path):
    calls=[]; saved=[]; model=object(); old={'scientific':'fixed'}
    ctx={'folder':tmp_path,'adapters':{'search':SimpleNamespace(model=model)},'metadata':old}
    monkeypatch.setattr(worker,'_ORIGINAL_INITIALIZE',lambda:calls.append('corrected') or ctx)
    monkeypatch.setattr(worker.graph,'build_graph',lambda *args:calls.append('graph') or SimpleNamespace(identity={}))
    monkeypatch.setattr(worker.graph.InferenceBinding,'install',lambda self,*args:self)
    monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append((path.name,value)))
    result=worker.initialize()
    assert calls==['corrected','graph'] and old=={'scientific':'fixed'}
    assert result['metadata']==dict(old,inference_backend=graph.BACKEND)
    assert saved[0][0]=='inference_execution_identity.json' and result['graph_parity_passed'] is False


def test_failed_graph_initialization_preserves_error_and_closes_both_adapters(monkeypatch,tmp_path):
    closed=[]; saved=[]; model=object()
    ctx={'folder':tmp_path,'adapters':{name:SimpleNamespace(model=model,close=lambda:closed.append(True))
                                    for name in ('search','reference')},'metadata':{}}
    monkeypatch.setattr(worker,'_ORIGINAL_INITIALIZE',lambda:ctx)
    def failure(*args): raise RuntimeError('graph trace failed')
    monkeypatch.setattr(worker.graph,'build_graph',failure)
    monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append((path.name,value)))
    with pytest.raises(RuntimeError,match='graph trace failed'): worker.initialize()
    assert closed==[True,True] and saved[0][0]=='graph_initialization_error.json'


def test_raw_request_origin_and_final_serialization_deadline_are_unchanged(monkeypatch,tmp_path):
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl,'threadpool_limits',lambda **kwargs:nullcontext())
    closed=[]; saved=[]; seen=[]
    ctx={'metadata':{},'folder':tmp_path,'adapters':{'search':SimpleNamespace(close=lambda:closed.append(True))}}
    packet={'type':'request','request_id':'r1','payload':{'player':'search','observer':{}}}
    stdout=io.StringIO()
    monkeypatch.setattr(worker,'initialize',lambda:ctx)
    monkeypatch.setattr(worker,'handle',lambda context,request:seen.append(context['request_started']) or {'exact':'science'})
    monkeypatch.setattr(worker.sys,'stdin',SimpleNamespace(buffer=io.BytesIO((json.dumps(packet)+'\n').encode())))
    monkeypatch.setattr(worker.sys,'stdout',stdout)
    monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append(value))
    times=iter([100.,100.901]); monkeypatch.setattr(worker.time,'perf_counter',lambda:next(times))
    assert worker.serve()==1
    assert seen==[100.] and ctx['request_started']==100.
    assert [json.loads(line)['type'] for line in stdout.getvalue().splitlines()]==['ready','error']
    assert '0.900s' in saved[0]['error'] and closed==[True]
