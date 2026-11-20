"""Model-free child profiling semantics, bounded payload and actual-path routing."""
from contextlib import nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_component_timing_20261025 import child_profile as cp, worker


class Clock:
    def __init__(self, step): self.value=0; self.step=step
    def __call__(self): self.value+=self.step; return self.value


def profiler():
    return cp.ChildProfiler(wall_clock=Clock(10),cpu_clock=Clock(1),pid=123)


def test_nested_component_totals_are_exclusive_without_double_counting():
    p=profiler()
    outer=p.begin('inference'); inner=p.begin('callback_hash')
    p.end(inner); p.end(outer); p.finish()
    result=p.snapshot()
    assert result['totals']['inference']==[1,30,3,30,0,20,2]
    assert result['totals']['callback_hash']==[1,10,1,10,0,10,1]
    assert result['component_nested_calls']==1
    assert result['handle_end_ns']-result['handle_start_ns']==50
    assert result['handle_unattributed_wall_ns']==20
    assert result['handle_unattributed_cpu_ns']==2
    assert result['inclusive_labels_overlap'] is True
    assert result['exclusive_totals_exclude_nested_components'] is True


def test_fixed_payload_does_not_grow_with_inference_call_count():
    p=profiler()
    for _ in range(1000):
        frame=p.begin('inference'); p.end(frame)
    p.finish(); result=p.snapshot()
    assert result['totals']['inference'][0]==1000
    assert set(result['totals'])==set(cp.LABELS)
    assert not any('interval' in key for key in result)
    assert cp.validate_payload_size(result)<2048
    assert result['times_are_diagnostic_not_trusted_deadline_receipts'] is True


def test_profile_size_bound_includes_future_emit_marker():
    p=profiler(); p.finish(); value=p.snapshot()
    bound=cp.validate_payload_size(value)
    value['before_emit_ns']=2**63-1
    assert len(json.dumps(value,separators=(',',':')).encode())==bound
    value['unapproved_large_payload']='x'*8192
    with pytest.raises(ValueError,match='8192-byte'): cp.validate_payload_size(value)


def test_wrapper_returns_exact_object_and_preserves_original_exception():
    owner=cp.InstrumentedBindings(); marker=object(); observed=[]
    def original(*args,**kwargs): observed.append((args,kwargs)); return marker
    wrapped=owner.wrap('inference',original)
    assert wrapped(3,key=4) is marker
    assert observed==[((3,),{'key':4})]
    owner.active=profiler()
    assert wrapped(5) is marker
    error=LookupError('exact original error')
    def raises(): raise error
    with pytest.raises(LookupError) as caught: owner.wrap('callback_hash',raises)()
    assert caught.value is error
    p=owner.active; owner.active=None; p.finish()
    assert p.snapshot()['totals']['callback_hash'][4]==1


def fake_modules():
    class Opponent:
        def _infer(self,value): return value
    original=SimpleNamespace(Opponent=Opponent)
    callbacks=SimpleNamespace(digest=lambda value:value)
    bridge=SimpleNamespace(encode_observation=lambda value:value)
    frozen=SimpleNamespace(decode_observer=lambda value:value)
    stdin,stdout=io.StringIO(),io.StringIO('response\n')
    adapter=SimpleNamespace(proc=SimpleNamespace(stdin=stdin,stdout=stdout))
    return original,callbacks,bridge,frozen,adapter


def test_install_once_restores_exact_original_functions_and_streams():
    original,callbacks,bridge,frozen,adapter=fake_modules()
    old=(original.Opponent._infer,callbacks.digest,bridge.encode_observation,
         frozen.decode_observer,adapter.proc.stdin,adapter.proc.stdout)
    bindings=cp.InstrumentedBindings().install({'search':adapter},worker_module=frozen,
        adapter_module=callbacks,original_module=original,bridge_module=bridge)
    try:
        assert original.Opponent()._infer(old) is old
        p=profiler(); bindings.active=p
        assert original.Opponent()._infer(old) is old
        assert callbacks.digest(old) is old
        assert bridge.encode_observation(old) is old
        assert frozen.decode_observer(old) is old
        assert adapter.proc.stdin.write('exact')==5
        adapter.proc.stdin.flush()
        assert adapter.proc.stdout.readline()=='response\n'
        bindings.active=None; p.finish()
        assert all(row[0]==1 for row in p.snapshot()['totals'].values())
        assert old[4].getvalue()=='exact'
        with pytest.raises(RuntimeError,match='Concurrent or repeated'):
            bindings.install({})
    finally:
        bindings.close()
    assert (original.Opponent._infer,callbacks.digest,bridge.encode_observation,
            frozen.decode_observer,adapter.proc.stdin,adapter.proc.stdout)==old
    bindings.close()


def test_failed_install_restores_previous_patch_and_releases_lock():
    original,callbacks,bridge,frozen,adapter=fake_modules()
    old=original.Opponent._infer
    with pytest.raises(AttributeError):
        cp.InstrumentedBindings().install({'search':adapter},worker_module=frozen,
            adapter_module=SimpleNamespace(),original_module=original,bridge_module=bridge)
    assert original.Opponent._infer is old
    bindings=cp.InstrumentedBindings().install({'search':adapter},worker_module=frozen,
        adapter_module=callbacks,original_module=original,bridge_module=bridge)
    bindings.close()


def context(enabled=True):
    return {'component_profile_enabled':enabled,'component_bindings':cp.InstrumentedBindings(),
            'request_started':None,'request_read_complete_ns':None,'adapters':{}}


def test_baseline_handle_delegates_without_profile_or_scientific_copy(monkeypatch):
    result={'action':{'x':1},'diagnostics':{'unchanged':[1,2]}}
    monkeypatch.setattr(worker,'_ORIGINAL_HANDLE',lambda context,request:result)
    monkeypatch.setattr(worker,'ChildProfiler',lambda **kwargs:pytest.fail('Baseline must not profile'))
    assert worker.handle(context(False),{'ordinary':True}) is result
    assert 'timing_profile' not in result


def test_profile_adds_only_sidecar_and_restores_active_binding(monkeypatch):
    original={'action':{'x':1},'diagnostics':{'unchanged':[1,2]}}
    scientific=deepcopy(original); ctx=context()
    def handler(context,request):
        assert context['component_bindings'].active is not None
        return original
    monkeypatch.setattr(worker,'_ORIGINAL_HANDLE',handler)
    result=worker.handle(ctx,{'ordinary':True})
    profile=result.pop('timing_profile')
    assert result is original and result==scientific
    assert ctx['component_bindings'].active is None
    assert profile['schema']==cp.SCHEMA and profile['handle_failed'] is False
    assert cp.validate_payload_size(profile)<2048


def test_corrected_entrypoint_recursion_profiles_all52_and_returns_original_science(monkeypatch):
    ctx=context(); evidence=[]; ctx['evidence_callback']=evidence.append
    previous=worker.frozen.handle
    def handler(context,request):
        if request=={'operation':'entrypoint_checks'}:
            results=[worker.frozen.handle(context,{'case':i}) for i in range(52)]
            assert results==[{'scientific_case':i} for i in range(52)]
            return {'status':'entrypoint_counterexamples_passed','counts':{'complete_decisions':52}}
        assert context['component_bindings'].active is not None
        return {'scientific_case':request['case']}
    monkeypatch.setattr(worker,'_ORIGINAL_HANDLE',handler)
    result=worker.handle(ctx,{'operation':'entrypoint_checks'})
    assert result=={'status':'entrypoint_counterexamples_passed','counts':{'complete_decisions':52}}
    assert len(evidence)==52
    assert all(item['event']=='qualification_timing_profile' and
        item['profile']['before_emit_ns'] is None and item['profile']['request_read_complete_ns'] is None
        for item in evidence)
    assert worker.frozen.handle is previous


def test_failed_recursive_entrypoint_restores_binding_and_keeps_original_failure(monkeypatch):
    ctx=context(); ctx['evidence_callback']=lambda value:None
    previous=worker.frozen.handle; error=ValueError('actual counterexample failed')
    def handler(context,request):
        if request=={'operation':'entrypoint_checks'}: return worker.frozen.handle(context,{'case':0})
        raise error
    monkeypatch.setattr(worker,'_ORIGINAL_HANDLE',handler)
    with pytest.raises(ValueError) as caught: worker.handle(ctx,{'operation':'entrypoint_checks'})
    assert caught.value is error and worker.frozen.handle is previous
    assert ctx['component_bindings'].active is None
    assert ctx['_failed_component_profile']['handle_failed'] is True


@pytest.mark.parametrize('mode',['0','1'])
def test_initialize_delegates_corrected_guard_model_setup_once_and_keeps_old_metadata(monkeypatch,tmp_path,mode):
    calls=[]; saved=[]; oldmetadata={'scientific_identity':'unchanged'}
    original={'metadata':oldmetadata,'folder':tmp_path,'adapters':{}}
    monkeypatch.setenv('STUDY_COMPONENT_PROFILE',mode)
    monkeypatch.setattr(worker,'_ORIGINAL_INITIALIZE',lambda:calls.append('exact corrected initializer') or original)
    monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append((path.name,value)))
    monkeypatch.setattr(worker.InstrumentedBindings,'install',lambda self,adapters:self)
    result=worker.initialize()
    assert calls==['exact corrected initializer'] and oldmetadata=={'scientific_identity':'unchanged'}
    assert result['metadata']==dict(oldmetadata,component_timing_mode='profiled' if mode=='1' else 'baseline')
    assert saved[0][0]=='component_timing_identity.json'
    assert (result['component_bindings'] is not None)==(mode=='1')


def serve_fixture(monkeypatch,tmp_path,*,enabled,handle):
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl,'threadpool_limits',lambda **kwargs:nullcontext())
    closed=[]; saved=[]
    ctx={'metadata':{'unchanged':'ready'},'folder':tmp_path,'component_profile_enabled':enabled,
         'component_bindings':None,'adapters':{'search':SimpleNamespace(close=lambda:closed.append(True))}}
    packet={'type':'request','request_id':'r1','payload':{'player':'search','observer':{}}}
    stdout=io.StringIO()
    monkeypatch.setattr(worker,'initialize',lambda:ctx)
    monkeypatch.setattr(worker,'handle',handle)
    monkeypatch.setattr(worker.sys,'stdin',SimpleNamespace(buffer=io.BytesIO((json.dumps(packet)+'\n').encode())))
    monkeypatch.setattr(worker.sys,'stdout',stdout)
    monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append(value))
    return ctx,stdout,saved,closed


def test_raw_request_origin_is_not_reset_and_serialization_deadline_is_enforced(monkeypatch,tmp_path):
    observed=[]
    ctx,stdout,saved,closed=serve_fixture(monkeypatch,tmp_path,enabled=False,
        handle=lambda context,request:observed.append(context['request_started']) or {'action':'same'})
    ticks=iter([100.,100.901])
    monkeypatch.setattr(worker.time,'perf_counter',lambda:next(ticks))
    assert worker.serve()==1
    assert observed==[100.] and ctx['request_started']==100.
    packets=[json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [packet['type'] for packet in packets]==['ready','error']
    assert '0.900s' in saved[0]['error'] and closed==[True]


def test_before_emit_marker_precedes_final_response_serialization(monkeypatch,tmp_path):
    p=profiler(); p.finish(); profile=p.snapshot()
    ctx,stdout,saved,closed=serve_fixture(monkeypatch,tmp_path,enabled=True,
        handle=lambda context,request:{'action':'same','timing_profile':profile})
    monkeypatch.setattr(worker.time,'perf_counter',lambda:100.)
    ticks=iter([100000,200000])
    monkeypatch.setattr(worker.time,'perf_counter_ns',lambda:next(ticks))
    original_dumps=worker.json.dumps; observed=[]
    def dumps(value,*args,**kwargs):
        if value.get('type')=='response': observed.append(value['result']['timing_profile']['before_emit_ns'])
        return original_dumps(value,*args,**kwargs)
    monkeypatch.setattr(worker.json,'dumps',dumps)
    assert worker.serve()==0
    assert observed==[200000] and ctx['request_read_complete_ns']==100000
    assert json.loads(stdout.getvalue().splitlines()[-1])['result']['timing_profile']['before_emit_ns']==200000
    assert not saved and closed==[True]


def test_profile_envelope_time_cannot_extend_internal_deadline(monkeypatch):
    ctx=context(); ctx['request_started']=100.
    monkeypatch.setattr(worker,'_ORIGINAL_HANDLE',lambda *args:{'unchanged':True})
    monkeypatch.setattr(worker.time,'perf_counter',lambda:100.901)
    with pytest.raises(TimeoutError,match='component envelope'):
        worker.handle(ctx,{'ordinary':True})
    assert ctx['component_bindings'].active is None
    assert ctx['_failed_component_profile']['handle_failed'] is False
