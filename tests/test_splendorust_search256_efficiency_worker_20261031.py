"""Model-free tests of separate optional256 instrumentation and exact delegation."""
from contextlib import nullcontext
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_search256_efficiency_20261031 import worker as w


class Clock:
    def __init__(self):self.value=0
    def __call__(self):self.value+=10;return self.value


class Probe:
    def __init__(self,pid=123,values=(0,25)):
        self.pid=pid;self.created=10.;self.values=iter(values)
    def read(self):return next(self.values)


def profiler(values=(0,25)):
    return w.EfficiencyProfiler(Probe(values=values),wall_clock=Clock(),cpu_clock=Clock())


def test_nested_aggregates_have_an_exact_disjoint_partition():
    p=profiler();outer=p.begin('native_exchange');inner=p.begin('compiled_forward')
    p.end(inner);p.end(outer);p.finish();row=p.snapshot()
    assert row['native_cpu_ns']==25
    assert sum(x['exclusive_wall_ns'] for x in row['totals'].values())+row['handler_unattributed_wall_ns']==row['handler_wall_ns']
    assert sum(x['exclusive_thread_cpu_ns'] for x in row['totals'].values())+row['handler_unattributed_thread_cpu_ns']==row['handler_thread_cpu_ns']
    assert row['totals']['native_exchange']['inclusive_wall_ns']>row['totals']['native_exchange']['exclusive_wall_ns']


def test_decreasing_native_cpu_is_not_silently_clamped():
    p=profiler((100,99))
    with pytest.raises(ValueError,match='decreased'):p.finish()


def test_unavailable_native_cpu_is_not_treated_as_zero():
    probe=Probe();probe.read=lambda:(_ for _ in ()).throw(PermissionError('counter unavailable'))
    with pytest.raises(PermissionError):w.EfficiencyProfiler(probe)


def test_native_cpu_api_works_model_free_on_actual_current_process():
    probe=w.NativeCPUProbe(os.getpid());first=probe.read();second=probe.read()
    assert type(first) is int and second>=first and probe.created>0


def test_native_cpu_pid_reuse_is_rejected(monkeypatch):
    import psutil
    probe=object.__new__(w.NativeCPUProbe);probe.pid=123;probe.created=10.
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:11.))
    with pytest.raises(ValueError,match='identity changed'):probe.read()


@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf'),None])
def test_invalid_native_cpu_counter_rejected(monkeypatch,bad):
    import psutil
    probe=object.__new__(w.NativeCPUProbe);probe.pid=123;probe.created=10.
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:10.,cpu_times=lambda:SimpleNamespace(user=bad,system=0.)))
    with pytest.raises(ValueError,match='malformed'):probe.read()


class Forward:
    def __init__(self):self.compiled=lambda values:[x*2 for x in values]
    def infer(self,adapter,tokens,batch):return self.compiled(tokens)


class Adapter:
    def __init__(self,forward,pid):
        self.forward=forward
        self.proc=SimpleNamespace(pid=pid,stdin=io.StringIO(),stdout=io.StringIO('infer\nresult\n'))
    def _infer(self,tokens,batch):return self.forward.infer(self,tokens,batch)
    def exchange(self,request):
        self.proc.stdin.write('select\n');self.proc.stdin.flush();self.proc.stdout.readline()
        result=self._infer(request,1)
        self.proc.stdin.write('output\n');self.proc.stdin.flush();self.proc.stdout.readline()
        return result


def fixture_bindings():
    forward=Forward();instances={arm:Adapter(forward,pid) for arm,pid in (('reference',123),('search',124))}
    originals={arm:dict(instance.__dict__) for arm,instance in instances.items()}
    compiled=forward.compiled
    bindings=w.EfficiencyBindings(probe_factory=Probe).install(instances,forward)
    return forward,instances,bindings,originals,compiled


def test_instance_hooks_preserve_results_and_restore_exact_callables():
    forward,instances,bindings,originals,compiled=fixture_bindings()
    p=bindings.start('search');result=instances['search'].exchange([1.,2.]);bindings.finish(p)
    row=p.snapshot()
    assert result==[2.,4.]
    for label in ('compiled_forward','inference_bridge','callback_total','native_exchange'):
        assert row['totals'][label]['calls']==1
    for label in ('native_read_wait','native_write','native_flush'):assert row['totals'][label]['calls']==2
    bindings.close()
    assert forward.compiled is compiled and 'infer' not in vars(forward)
    assert all(vars(instances[arm])==originals[arm] for arm in instances)
    assert '_infer' not in vars(instances['search'])


def test_inactive_hooks_do_not_read_clocks_or_native_counters(monkeypatch):
    _,instances,bindings,_,_=fixture_bindings()
    monkeypatch.setattr(w,'EfficiencyProfiler',lambda *a,**k:pytest.fail('Inactive observer must not sample'))
    assert instances['search'].exchange([1])==[2]
    bindings.close()


def test_exception_identity_preserved_and_failed_span_recorded():
    bindings=w.EfficiencyBindings();p=profiler();bindings.active=p
    error=RuntimeError('scientific failure')
    wrapped=bindings.wrap('compiled_forward',lambda:(_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError) as caught:wrapped()
    assert caught.value is error
    p.finish(True);assert p.snapshot()['totals']['compiled_forward']['failures']==1


def test_original_exception_wins_if_instrumentation_also_fails():
    bindings=w.EfficiencyBindings();p=profiler();bindings.active=p
    p.end=lambda *a,**k:(_ for _ in ()).throw(ValueError('counter failure'))
    error=RuntimeError('scientific failure')
    with pytest.raises(RuntimeError) as caught:
        bindings.wrap('compiled_forward',lambda:(_ for _ in ()).throw(error))()
    assert caught.value is error and 'Efficiency span also failed' in error.__notes__[0]


def test_install_failure_restores_already_changed_instances():
    forward=Forward();instance=Adapter(forward,123);compiled=forward.compiled
    bindings=w.EfficiencyBindings(probe_factory=lambda pid:(_ for _ in ()).throw(PermissionError()))
    with pytest.raises(PermissionError):bindings.install({'search':instance},forward)
    assert forward.compiled is compiled and 'infer' not in vars(forward)
    assert not bindings.installed and bindings.restorations==[]


def test_cannot_restore_active_hooks():
    _,_,bindings,_,_=fixture_bindings();p=bindings.start('search')
    with pytest.raises(ValueError,match='active'):bindings.close()
    bindings.finish(p);bindings.close()


def test_ordinary_delegates_exact_frozen_function_and_object(monkeypatch):
    result=object();seen=[]
    monkeypatch.setattr(w.accepted,'_ordinary',lambda context,request:(seen.append((context,request)),result)[1])
    context={};request={}
    assert w._ordinary(context,request) is result and seen==[(context,request)]


@pytest.mark.parametrize('mode', ['off','on'])
def test_frozen_observer_mode_is_bound_to_exact_folder(mode):
    freeze={'efficiency_observers':{'observer_off':'off','observer_on':'on'}}
    assert w.observer_mode(freeze,Path('observer_'+mode),mode)==mode
    with pytest.raises(ValueError):w.observer_mode(freeze,Path('observer_'+mode),'on' if mode=='off' else 'off')


def context(tmp_path,mode='off'):
    return dict(folder=tmp_path,observer_mode=mode,quota=256,route='research',efficiency_profiles=[],component_profiles=[],
        graph_parity_passed=True,boundary_handle_active=False,request_started=None,efficiency_bindings=None,adapters={})


def test_off_handler_installs_no_profiler_and_preserves_result(tmp_path,monkeypatch):
    ctx=context(tmp_path);result={'simulations_quota':256,'unchanged':[1,2,3]}
    monkeypatch.setattr(w,'_ordinary',lambda *a:result)
    monkeypatch.setattr(w,'EfficiencyProfiler',lambda *a,**k:pytest.fail('OFF must not sample'))
    assert w.handle(ctx,{'player':'search','observer':{}}) is result
    assert ctx['efficiency_profiles']==[] and len(ctx['component_profiles'])==1


@pytest.mark.parametrize('payload',[{'player':'unknown','observer':{}},None,{'player':[],'observer':{}}])
def test_on_invalid_payload_preserves_the_original_error(tmp_path,monkeypatch,payload):
    _,_,bindings,_,_=fixture_bindings()
    ctx=context(tmp_path,'on');ctx['efficiency_bindings']=bindings
    error=ValueError('original request allowlist rejection')
    monkeypatch.setattr(w,'_ordinary',lambda *a:(_ for _ in ()).throw(error))
    with pytest.raises(ValueError) as caught:w.handle(ctx,payload)
    assert caught.value is error and bindings.active is None and not ctx['boundary_handle_active']
    assert len(ctx['efficiency_profiles'])==1 and ctx['efficiency_profiles'][0]['handler_failed'] is True
    bindings.close()


def test_counter_failure_after_scientific_success_is_preserved_as_failure(tmp_path,monkeypatch):
    _,_,bindings,_,_=fixture_bindings()
    ctx=context(tmp_path,'on');ctx['efficiency_bindings']=bindings
    bindings.probes['search']=Probe(values=(10,9))
    monkeypatch.setattr(w,'_ordinary',lambda *a:{'simulations_quota':256})
    with pytest.raises(ValueError,match='decreased'):w.handle(ctx,{'player':'search','observer':{}})
    assert bindings.active is None and not ctx['boundary_handle_active']
    assert ctx['component_profiles'][0]['handle_failed'] is True and ctx['efficiency_profiles']==[]
    bindings.close()


def test_original_exception_preserved_if_counter_finalization_also_fails(tmp_path,monkeypatch):
    _,_,bindings,_,_=fixture_bindings()
    ctx=context(tmp_path,'on');ctx['efficiency_bindings']=bindings
    bindings.probes['search']=Probe(values=(10,9));error=ValueError('original decoder rejection')
    monkeypatch.setattr(w,'_ordinary',lambda *a:(_ for _ in ()).throw(error))
    with pytest.raises(ValueError) as caught:w.handle(ctx,{'player':'search','observer':{}})
    assert caught.value is error and 'Efficiency finalization also failed' in error.__notes__[0]
    assert bindings.active is None and ctx['efficiency_profiles']==[]
    bindings.close()


def test_corrected_source_guard_precedes_initialization_and_model_loading(monkeypatch):
    calls=[];error=ValueError('guard stop')
    def guard():calls.append('guard');raise error
    monkeypatch.setattr(w,'verify_runtime_sources',guard)
    with pytest.raises(ValueError) as caught:w.initialize('research')
    assert caught.value is error and calls==['guard']


def boundary_evidence(payload=None):
    return dict(request_raw_sha256='a'*64,payload_sha256=w.digest(payload),result_sha256=None,
        serialized_packet_utf8_sha256='b'*64,serialized_packet_utf8_bytes=40,outcome='response_flushed',error=None,
        pre_write_ns=100,post_flush_ns=110,profiles=[])


def test_efficiency_file_is_fsynced_before_boundary_publication(tmp_path,monkeypatch):
    ctx=context(tmp_path);events=[]
    monkeypatch.setattr(w.os,'fsync',lambda fd:events.append('fsync'))
    def publish(folder,rid,evidence,**kw):
        assert json.loads((folder/(rid+'_efficiency_profile.json')).read_text())['observer_mode']=='off'
        assert events==['fsync'];events.append('boundary')
    monkeypatch.setattr(w,'publish_profile',publish)
    w.publish_evidence(ctx,'request-00000001',boundary_evidence())
    assert events==['fsync','boundary']


def test_publication_is_exclusive_and_keeps_old_evidence(tmp_path,monkeypatch):
    ctx=context(tmp_path);target=tmp_path/'request-00000001_efficiency_profile.json';target.write_text('original')
    monkeypatch.setattr(w,'publish_profile',lambda *a,**k:pytest.fail('No barrier after publication failure'))
    with pytest.raises(FileExistsError):w.publish_evidence(ctx,'request-00000001',boundary_evidence())
    assert target.read_text()=='original'


def test_failed_efficiency_fsync_preserves_fragment_without_barrier(tmp_path,monkeypatch):
    ctx=context(tmp_path)
    monkeypatch.setattr(w.os,'fsync',lambda fd:(_ for _ in ()).throw(OSError('disk')))
    monkeypatch.setattr(w,'publish_profile',lambda *a,**k:pytest.fail('No barrier after fsync failure'))
    with pytest.raises(OSError):w.publish_evidence(ctx,'request-00000001',boundary_evidence())
    assert (tmp_path/'request-00000001_efficiency_profile.json').is_file()


@pytest.mark.parametrize('rid',['../request-00000001','request-1','request-000000001',None])
def test_malformed_request_ids_cannot_publish(tmp_path,rid):
    with pytest.raises(ValueError):w.publish_evidence(context(tmp_path),rid,boundary_evidence())
    assert not list(tmp_path.iterdir())


def test_after_response_sidecar_failure_cannot_emit_second_packet(tmp_path,monkeypatch):
    import threadpoolctl
    ctx=context(tmp_path);ctx['metadata']={}
    payload={'player':'search','observer':{}}
    request=json.dumps(dict(type='request',request_id='request-00000001',payload=payload)).encode()+b'\n'
    stdout=io.StringIO()
    monkeypatch.setattr(w.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(request)))
    monkeypatch.setattr(w.sys,'stdout',stdout)
    monkeypatch.setattr(w,'initialize',lambda route:ctx)
    monkeypatch.setattr(w,'close_context',lambda context:None)
    monkeypatch.setattr(w,'handle',lambda *a:{'science':'unchanged'})
    monkeypatch.setattr(w,'publish_evidence',lambda *a,**k:(_ for _ in ()).throw(OSError('sidecar failed')))
    monkeypatch.setattr(w,'save_sidecar_error',lambda *a,**k:None)
    monkeypatch.setattr(threadpoolctl,'threadpool_limits',lambda **k:nullcontext())
    assert w.serve(route='research')==1
    packets=[json.loads(x) for x in stdout.getvalue().splitlines()]
    assert [x['type'] for x in packets]==['ready','response']
    assert packets[1]['result']=={'science':'unchanged'}


def test_off_profile_verifier_rejects_invented_measurements():
    child=boundary_evidence({});child['profiles']=[]
    profile=dict(schema=w.EFFICIENCY_SCHEMA,observer_mode='off',profile_enabled=False,pid=1,request_id='request-00000001',
        route='research',quota=256,scientific_response_fields_unchanged=True,success_sidecar_after_response_flush=True,
        boundary_sidecar_is_publication_barrier=True,times_are_diagnostic_not_trusted_deadline_receipts=True,**child)
    args=dict(mode='off',request_id='request-00000001',payload={},receipt={'request_id':'request-00000001'},
        parent_profile={'request_raw_sha256':'a'*64},child_profile=child,worker_pid=1)
    assert w.verify_efficiency_profile(profile,**args)['status']=='passed'
    profile['profiles']=[{}]
    with pytest.raises(ValueError,match='OFF'):w.verify_efficiency_profile(profile,**args)


def on_profile():
    p=profiler();p.finish();row=p.snapshot()
    payload={'player':'search','observer':{}}
    child=boundary_evidence(payload)
    child['profiles']=[dict(handle_start_ns=0,handle_end_ns=1000,handle_failed=False)]
    profile=dict(schema=w.EFFICIENCY_SCHEMA,observer_mode='on',profile_enabled=True,pid=1,request_id='request-00000001',
        route='research',quota=256,scientific_response_fields_unchanged=True,success_sidecar_after_response_flush=True,
        boundary_sidecar_is_publication_barrier=True,times_are_diagnostic_not_trusted_deadline_receipts=True,
        **{k:v for k,v in child.items() if k!='profiles'},profiles=[row])
    args=dict(mode='on',request_id='request-00000001',payload=payload,
        receipt={'request_id':'request-00000001','status':'ok','result':{'diagnostics':{'inference_calls':0,'phases':[]}}},
        parent_profile={'request_raw_sha256':'a'*64},child_profile=child,worker_pid=1)
    return profile,args


def test_on_profile_verifier_accepts_zero_inference_for_forced_pass():
    profile,args=on_profile()
    assert w.verify_efficiency_profile(profile,**args)['profiles']==1


@pytest.mark.parametrize('change',['owner','request','quota','payload','native_negative','thread_negative','sum','scope','count','boundary','failure'])
def test_mutated_instrumentation_evidence_rejected(change):
    profile,args=on_profile();row=profile['profiles'][0]
    if change=='owner':profile['pid']=2
    if change=='request':profile['request_id']='request-00000002'
    if change=='quota':profile['quota']=128
    if change=='payload':profile['payload_sha256']='f'*64
    if change=='native_negative':row['native_cpu_ns']=-1
    if change=='thread_negative':row['handler_thread_cpu_ns']=-1
    if change=='sum':row['handler_unattributed_wall_ns']+=1
    if change=='scope':row['wall_minus_cpu_is_unexplained_not_proven_scheduling']=False
    if change=='count':row['totals']['compiled_forward']['calls']=1
    if change=='boundary':row['handler_end_ns']=2000
    if change=='failure':row['handler_failed']=True
    with pytest.raises(ValueError):w.verify_efficiency_profile(profile,**args)
