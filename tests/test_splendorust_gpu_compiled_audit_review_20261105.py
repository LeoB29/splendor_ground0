"""Independent model-free graph-evidence and initialization audit checks."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_compiled_20261105 import audit as a, common as c, compilation as k


def graph_fixture():
    block=['aten::layer_norm','aten::linear','aten::bmm','aten::div','aten::softmax',
        'aten::bmm','aten::linear','aten::add','aten::layer_norm','aten::linear','aten::gelu','aten::linear','aten::add']
    ops=['aten::linear','aten::add']+block*6+['aten::layer_norm','aten::linear','aten::linear','aten::tanh','aten::cat']
    nodes=[dict(kind=kind,inputs=[i],outputs=[dict(id=i+1,type='Tensor')],attributes={},blocks=[])
           for i,kind in enumerate(ops)]
    body=dict(inputs=[dict(id=0,type='Tensor')],nodes=nodes,outputs=[len(ops)])
    value=dict(body=body,operators=ops,operator_inventory=dict(Counter(ops)),math_sequence=ops,fingerprint=c.digest(body))
    text='graph(%0):\n'+''.join(f'  %{i+1} = {kind}(%{i})\n' for i,kind in enumerate(ops))
    return value,text


def prepared_graph(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(a.old,'load_saved_library',lambda _:SimpleNamespace(fallback=lambda text:'fallback' in text.lower()))
    folder=tmp_path/'worker/runtime';compiled=folder.with_name('runtime_compilation');compiled.mkdir(parents=True)
    descriptions,text=graph_fixture();graphs={}
    for name in ('traced','frozen'):
        path=compiled/(name+'_graph.txt');path.write_text(text,encoding='utf-8');graphs[name]=c.artifact(path)
        graphs[name+'_inventory']=c.save(compiled/(name+'_graph_inventory.json'),descriptions)
    binding=dict(options=k.OPTIONS,weights_sha256='a'*64,device='privateuseone:0',
        frozen_graph_fingerprint=descriptions['fingerprint'],mathematical_operator_sequence=descriptions['math_sequence'])
    receipt=dict(schema='fixed_b8_trace_freeze_v1',status='compiled_supported_pending_exact_qualification',
        implementation='compiled',retries=0,timed_panel=False,exact_outputs_qualified=False,original_weights_unchanged=True,
        binding=binding,compilation_fingerprint=c.digest(binding),graphs=graphs,warnings=[])
    artifact=c.save(compiled/'compilation.json',receipt)
    identity=dict(implementation='compiled',route='gpu',scheduler='row_fifo32',maximum_logical_clients=32,
        physical_batch_unchanged=8,weights_sha256='a'*64,device={'torch_device':'privateuseone:0'},
        compilation_fingerprint=receipt['compilation_fingerprint'],compilation_receipt=artifact)
    c.save(folder/'runtime_identity.json',identity)
    return folder,compiled,identity,receipt


def rewrite(path,value):
    Path(path).write_text(c.canonical(value)+'\n',encoding='utf-8')


def rebind_receipt(folder,compiled,identity,receipt):
    rewrite(compiled/'compilation.json',receipt)
    identity['compilation_receipt']=c.artifact(compiled/'compilation.json')
    rewrite(folder/'runtime_identity.json',identity)


def test_synthetic_complete_graph_text_inventory_and_fingerprint_acceptance(tmp_path,monkeypatch):
    folder,compiled,identity,receipt=prepared_graph(tmp_path,monkeypatch)
    proof=a.compilation_identity(folder,'compiled')
    assert proof['compilation_fingerprint']==identity['compilation_fingerprint']
    assert proof['compilation_receipt']==c.artifact(compiled/'compilation.json')


def test_tampered_canonical_graph_rejected_even_with_updated_file_binding(tmp_path,monkeypatch):
    folder,compiled,identity,receipt=prepared_graph(tmp_path,monkeypatch)
    path=compiled/'frozen_graph_inventory.json';value=c.read(path)
    value['body']['nodes'][2]['inputs']=[0];rewrite(path,value)
    receipt['graphs']['frozen_inventory']=c.artifact(path);rebind_receipt(folder,compiled,identity,receipt)
    with pytest.raises(ValueError,match='canonical graph topology'):a.compilation_identity(folder,'compiled')


def test_tampered_graph_text_rejected_even_with_updated_file_binding(tmp_path,monkeypatch):
    folder,compiled,identity,receipt=prepared_graph(tmp_path,monkeypatch)
    path=compiled/'frozen_graph.txt';path.write_text(path.read_text().replace('aten::linear','aten::bmm',1))
    receipt['graphs']['frozen']=c.artifact(path);rebind_receipt(folder,compiled,identity,receipt)
    with pytest.raises(ValueError,match='textual graph operation'):a.compilation_identity(folder,'compiled')


@pytest.mark.parametrize('change',[
    lambda identity,receipt:identity.update(compilation_fingerprint='0'*64),
    lambda identity,receipt:receipt['binding']['options'].update(freeze_optimize_numerics=True),
    lambda identity,receipt:receipt['warnings'].append({'message':'unexpected CPU fallback'}),
])
def test_fingerprint_options_or_fallback_cannot_claim_supported_identity(tmp_path,monkeypatch,change):
    folder,compiled,identity,receipt=prepared_graph(tmp_path,monkeypatch)
    # Do not mutate the frozen module OPTIONS through the synthetic fixture.
    import copy
    receipt=copy.deepcopy(receipt)
    change(identity,receipt);rebind_receipt(folder,compiled,identity,receipt)
    with pytest.raises(ValueError):a.compilation_identity(folder,'compiled')


def test_eager_control_cannot_carry_candidate_or_observer_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);folder=tmp_path/'runtime'
    value=dict(implementation='eager',route='gpu',scheduler='row_fifo32',maximum_logical_clients=32,
        physical_batch_unchanged=8,compilation_fingerprint=None,compilation_receipt=None)
    c.save(folder/'runtime_identity.json',value)
    assert a.compilation_identity(folder,'eager')==dict(implementation='eager',compilation_fingerprint=None)
    value['timing_observer_enabled']=False;rewrite(folder/'runtime_identity.json',value)
    with pytest.raises(ValueError,match='unobserved'):a.compilation_identity(folder,'eager')


def initialization_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);folder=tmp_path/'cell/worker'
    c.save(folder/'worker_identity.json',dict(started_unix=100.))
    c.save(folder.parent/'execution.json',dict(unix_end=110.))
    value=dict(schema='compiled_runtime_initialization_v1',implementation='compiled',
        runtime_folder=str((folder/'runtime').resolve()),status='completed',error=None,no_retry=True,inside_panel_timing=False,
        started_perf_counter=1.,ended_perf_counter=3.,elapsed_seconds=2.,started_unix=101.,ended_unix=103.)
    path=folder/'initialization/runtime.json';c.save(path,value)
    return folder,path,value


def test_preparation_interval_precedes_first_request_and_remains_inside_worker(tmp_path,monkeypatch):
    folder,path,value=initialization_fixture(tmp_path,monkeypatch)
    proof=a.audit_initialization(folder,'runtime','compiled',True,dict(started_perf_counter=4.))
    assert proof['elapsed_seconds']==2. and proof['artifact']==c.artifact(path)
    value.update(status='failed',error='Original compilation traceback');rewrite(path,value)
    assert a.audit_initialization(folder,'runtime','compiled',False)['status']=='failed'


@pytest.mark.parametrize('change',[
    lambda value:value.update(ended_unix=111.),
    lambda value:value.update(started_unix=99.),
    lambda value:value.update(elapsed_seconds=1.),
    lambda value:value.update(inside_panel_timing=True),
    lambda value:value.update(runtime_folder='C:/different/runtime'),
])
def test_invalid_preparation_boundary_or_ownership_rejected(tmp_path,monkeypatch,change):
    folder,path,value=initialization_fixture(tmp_path,monkeypatch);change(value);rewrite(path,value)
    with pytest.raises(ValueError):a.audit_initialization(folder,'runtime','compiled',True,dict(started_perf_counter=4.))


def test_support_cannot_start_before_compilation_finishes(tmp_path,monkeypatch):
    folder,_,_=initialization_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='excluded'):
        a.audit_initialization(folder,'runtime','compiled',True,dict(started_perf_counter=2.))
