"""Model-free gates, exact backend routing and failure-evidence preservation."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.splendorust_boundary_timing_20261025 import runner as r, common as c


class Envelope:
    def __init__(self): self.stage=None;self.calls=[]
    def begin(self,stage): self.stage=stage;self.calls.append(('begin',stage))
    def finish(self): self.calls.append(('finish',self.stage))
    def guard(self): pass
    def admit(self): pass
    def remaining(self): return 100.
    def timeout(self,value): return value
    def effective_deadline(self): return 9999.
    def register_worker(self,folder): pass
    def own_worker(self,child): pass
    def close_worker(self,*args): pass
    def guard_worker_sample(self,*args): pass


@pytest.mark.parametrize('passed',[True,False])
def test_all_four_entrypoints_precede_timing_and_overhead_gates_diagnostic(monkeypatch,passed):
    order=[];saved=[];envelope=Envelope()
    monkeypatch.setattr(r,'qualify',lambda *args:order.append('qualified4'))
    monkeypatch.setattr(r,'run_panel',lambda env,freeze,plan,stage:order.append(stage) or {'qualified':passed})
    monkeypatch.setattr(r,'final_checks',lambda *args:order.append('final'))
    monkeypatch.setattr(c,'save',lambda path,value:saved.append(value))
    result=r.run_stages(envelope,{}, {})
    assert order==['qualified4','overhead']+(['diagnostic'] if passed else [])+['final']
    assert [stage for event,stage in envelope.calls if event=='begin']==['qualification','overhead']+(['diagnostic'] if passed else [])+['final']
    assert ('diagnostic' in result) is passed and saved[-1]['fresh_games']==saved[-1]['strength_games']==0


def test_hard_failure_cannot_reach_followup_or_acceptance(monkeypatch):
    monkeypatch.setattr(r,'qualify',lambda *args:None)
    monkeypatch.setattr(r,'run_panel',lambda *args:(_ for _ in ()).throw(ValueError('saved timeout')))
    monkeypatch.setattr(r,'final_checks',lambda *args:pytest.fail('Incomplete run cannot be accepted'))
    with pytest.raises(ValueError,match='timeout'):r.run_stages(Envelope(),{}, {})


def test_qualification_closes_all_four_actual_workers_before_return(monkeypatch):
    events=[]
    class Worker:
        def __init__(self,name):self.name=name
        def __enter__(self):events.append(('enter',self.name));return self
        def __exit__(self,*args):events.append(('close',self.name))
    monkeypatch.setattr(r,'make_worker',lambda name,*args,**kwargs:Worker(name))
    monkeypatch.setattr(r,'actual_entrypoint',lambda child,env,label:events.append(('actual52',label)))
    monkeypatch.setattr(c,'save',lambda *args:None)
    result=r.qualify(Envelope(),{})
    assert result['workers']==4 and result['complete_decisions']==208
    assert [kind for kind,_ in events]==['enter','actual52','close']*4
    assert result['labels']==['64_baseline','64_profiled','32_baseline','32_profiled']


@pytest.mark.parametrize('mode,quota,module',[
    ('baseline',64,'tools.splendorust_inference_opt_20261025.worker'),
    ('baseline',32,'tools.splendorust_search_scale_20261024.worker'),
    ('profiled',64,'tools.splendorust_boundary_timing_20261025.worker'),
    ('profiled',32,'tools.splendorust_boundary_timing_20261025.worker')])
def test_worker_routes_original_graph64_eager32_without_graph32(monkeypatch,tmp_path,mode,quota,module):
    from tools.splendorust_search_scale_20261024 import adapters
    captured=[];parity=[]
    metadata={'search_simulations':quota}
    if quota==64:metadata['inference_backend']=r.BACKEND
    child=SimpleNamespace(process=SimpleNamespace(pid=91),initialization={'result':metadata},
        _sample=lambda:{'process_tree':[]},_failure=None,close=lambda:None)
    def make(mod,folder,**kwargs):captured.append((mod,kwargs));return child
    monkeypatch.setattr(c,'OUT',tmp_path/'out')
    monkeypatch.setattr(c,'BaselineStudyWorker',make);monkeypatch.setattr(c,'ProfiledStudyWorker',make)
    monkeypatch.setattr(adapters,'validate_ready_metadata',lambda actual,**kwargs:None)
    monkeypatch.setattr(r,'verify_instrumentation_identity',lambda *args:None)
    monkeypatch.setattr(r,'graph_parity',lambda *args:parity.append(True))
    freeze={'native_artifacts':[],'topology':{'selected_logical_processor':0}}
    assert r.make_worker('worker',Envelope(),freeze,mode=mode,quota=quota) is child
    assert captured[0][0]==module and captured[0][1]['env']['STUDY_SEARCH_SIMULATIONS']==str(quota)
    assert bool(parity) is (quota==64)
    assert child.runtime_mode==mode and child.challenger_quota==quota


def test_readiness_failure_closes_owned_worker(monkeypatch,tmp_path):
    from tools.splendorust_search_scale_20261024 import adapters
    closed=[];child=SimpleNamespace(process=SimpleNamespace(pid=1),initialization={'result':{}},
        _sample=lambda:{'process_tree':[]},_failure=None,close=lambda:closed.append(True))
    monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'BaselineStudyWorker',lambda *args,**kwargs:child)
    monkeypatch.setattr(adapters,'validate_ready_metadata',lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('bad readiness')))
    with pytest.raises(ValueError,match='readiness'):
        r.make_worker('one',Envelope(),{'native_artifacts':[],'topology':{'selected_logical_processor':0}},mode='baseline',quota=32)
    assert closed==[True]


@pytest.mark.parametrize('quota',[32,64])
def test_instrumentation_identity_binds_original_backend_without_changing_metadata(monkeypatch,quota):
    metadata={'search_simulations':quota};freeze={'unchanged_graph_backend_identity':{'graph_identity_sha256':'g'}}
    identity=dict(schema='splendorust_boundary_timing_child_v1',quota=quota,metadata=metadata,
        backend='qualified_graph64' if quota==64 else 'unchanged_eager32',
        scientific_response_fields_unchanged=True,clock='perf_counter_ns',graph_identity_sha256='g' if quota==64 else None)
    monkeypatch.setattr(c,'read',lambda path:identity)
    assert r.verify_instrumentation_identity('worker',freeze,quota,metadata)==identity
    identity['graph_identity_sha256']='different'
    with pytest.raises(ValueError,match='instrumentation identity'):
        r.verify_instrumentation_identity('worker',freeze,quota,metadata)


def test_profiles_are_separate_from_unchanged_clock(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    receipt={'request_id':'request-00000001','status':'ok'};original=deepcopy(receipt)
    folder=c.OUT/'workers/one';folder.mkdir(parents=True)
    child=SimpleNamespace(runtime_mode='profiled',last_parent_profile={'request_id':receipt['request_id'],'spans':[]},
        output_dir=folder,_closed=False)
    c.save(folder/(receipt['request_id']+'_child_profile.json'),{'request_id':receipt['request_id'],'scope':'sidecar'})
    result=r.collect_profiles(child,receipt,Envelope())
    assert receipt==original and result['child_profile']['scope']=='sidecar'
    assert c.read(c.ROOT/result['parent_profile_file'])==result['parent_profile']
    assert c.read(c.ROOT/result['child_profile_file'])==result['child_profile']


@pytest.mark.parametrize('status,raises',[('ok',True),('timeout',False)])
def test_missing_sidecar_on_closed_worker_only_allowed_for_failed_request(monkeypatch,tmp_path,status,raises):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    child=SimpleNamespace(runtime_mode='profiled',last_parent_profile={'request_id':'r'},output_dir=c.OUT/'workers/one',_closed=True)
    receipt={'request_id':'r','status':status}
    if raises:
        with pytest.raises(ValueError,match='sidecar'):r.collect_profiles(child,receipt,Envelope())
    else:
        result=r.collect_profiles(child,receipt,Envelope())
        assert result['parent_profile'] and result['child_profile'] is None


def test_no_partial_one_second_allowance_dispatched(monkeypatch,tmp_path):
    state=r.frozen_q.reference._initial();envelope=Envelope();envelope.timeout=lambda value:.999
    child=SimpleNamespace(challenger_quota=64,runtime_mode='baseline',_counter=0,output_dir=tmp_path,
        request=lambda *args,**kwargs:pytest.fail('Clamped request dispatched'))
    monkeypatch.setattr(c,'save',lambda *args:None)
    with pytest.raises(ValueError,match='stage-clamped'):r.decision(child,state,envelope)


def test_failed_panel_row_saved_before_stopping_without_second_request(monkeypatch):
    schedule=[dict(index=i,quota=64,mode='baseline') for i in range(48)];saved=[];calls=[]
    monkeypatch.setattr(r,'make_worker',lambda *args,**kwargs:nullcontext('worker'))
    monkeypatch.setattr(r.plan_rules,'snapshot_for_request',lambda *args:{})
    monkeypatch.setattr(r,'decode_training_snapshot',lambda value:value)
    def failed(*args):calls.append(True);return {'completed':False,'clock':{'status':'timeout'}}
    monkeypatch.setattr(r,'decision',failed)
    monkeypatch.setattr(c,'save',lambda path,row:saved.append(row))
    with pytest.raises(ValueError,match='failure preserved'):
        r.run_panel(Envelope(),{}, {'overhead_schedule':schedule},'overhead')
    assert len(calls)==1 and len(saved)==1 and saved[0]['clock']['status']=='timeout'


def test_stage_global_clocks_and_final_evidence_reserve(monkeypatch):
    now=SimpleNamespace(mono=1490.,wall=2490.)
    monkeypatch.setattr(r.time,'monotonic',lambda:now.mono);monkeypatch.setattr(r.time,'time',lambda:now.wall)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda root:SimpleNamespace(free=10*c.GiB));monkeypatch.setattr(c,'save',lambda *args:None)
    envelope=r.Envelope({'monotonic_origin':0.,'unix_origin':1000.})
    monkeypatch.setattr(envelope,'accounted_bytes',lambda:c.PREFINAL_EVIDENCE_BYTES-1)
    envelope.begin('diagnostic');assert envelope.remaining()==10 and envelope.deadline==1500
    monkeypatch.setattr(envelope,'accounted_bytes',lambda:c.PREFINAL_EVIDENCE_BYTES)
    with pytest.raises(ValueError,match='evidence'):envelope.guard()
    envelope.begin('final');assert envelope.deadline==1790
    now.wall=2800.
    with pytest.raises(ValueError,match='real-clock'):envelope.guard()


def test_constructor_aggregate_includes_new_second_worker():
    envelope=r.Envelope({'monotonic_origin':0.,'unix_origin':0.})
    root=SimpleNamespace(pid=1,create_time=lambda:1.,children=lambda recursive:[],memory_info=lambda:SimpleNamespace(rss=c.GiB))
    envelope.resident[(1,1.)]=root
    with pytest.raises(ValueError,match='Combined'):
        envelope.guard_worker_sample(SimpleNamespace(process=SimpleNamespace(pid=2)),{'rss_bytes':c.GiB+1})


def forecast():
    return dict(status='passed',model_calls=0,fresh_games=0,inventories=dict(workers=12,graph_workers=6,
        entrypoint_decisions=208,overhead_requests=48,diagnostic_requests=416),
        projected_stage_seconds={'qualification':100,'overhead':100,'diagnostic':500,'final':100},
        projected_total_seconds=800,projected_prefinal_bytes=80*c.MiB,projected_final_bytes=16*c.MiB)


@pytest.mark.parametrize('field,value',[
    ('model_calls',1),('projected_total_seconds',1800),('projected_total_seconds',799),
    ('projected_prefinal_bytes',96*1024**2),('projected_final_bytes',32*1024**2),
    ('projected_prefinal_bytes',float('nan'))])
def test_forecast_cannot_expand_any_allowance(field,value):
    record=forecast();record[field]=value
    with pytest.raises(ValueError):r.validate_resource_forecast(record)


def test_forecast_requires_full_inventory_and_each_stage():
    assert r.validate_resource_forecast(forecast())['status']=='passed'
    record=forecast();record['inventories']['workers']=8
    with pytest.raises(ValueError):r.validate_resource_forecast(record)
    record=forecast();record['projected_stage_seconds']['diagnostic']=600
    with pytest.raises(ValueError):r.validate_resource_forecast(record)


def test_real_finite_preparation_dependencies_are_canonical_and_current():
    accepted=r.predecessor_acceptance();prior=c.read(c.PRIOR/'freeze.json')
    plan=r.plan_rules.build_plan();parity=c.read(c.QUALIFIED/'graph_parity_plan.json')
    resource=r.validate_resource_forecast(forecast())
    inputs=r.preparation_dependencies(prior,accepted,plan,parity,resource)
    assert len(inputs)==len({item['path'] for item in inputs}) and all(not r.Path(item['path']).is_absolute() for item in inputs)
    assert 'torch' not in r.sys.modules
    c.verify(inputs)


def test_previous_negative_attempt_has_finite_verified_seal_and_cannot_be_promoted():
    result=r.previous_timing_acceptance()
    assert result['status']=='passed_preserved_negative_assessment'
    assert result['instrumentation_qualified'] is result['diagnostic_completed'] is False
    assert result['sealed_artifacts']==len(result['sealed_evidence'])==304
    assert result['full_strength_benchmark_seal_reaudit'] is False
    assert result['old_measurements_used_for_attribution'] is False
    assert result['freeze_sha256']==c.PREVIOUS_TIMING_FREEZE_SHA
    assert 'torch' not in r.sys.modules


@pytest.mark.parametrize('name,key,value',[
    ('completion.json','status','incomplete'),
    ('execution.json','exit_code',1),
    ('execution.json','cleanup_errors',['unconfirmed child']),
    ('diagnostic_outcome.json','instrumentation_qualified',True),
    ('diagnostic_outcome.json','diagnostic_completed',True),
    ('verification_summary.json','requests',47),
    ('all_graph_parity.json','workers',3),
    ('independent_overhead_review.json','status','incomplete')])
def test_previous_acceptance_rejects_altered_outcome_or_verification(monkeypatch,name,key,value):
    original=c.read
    def changed(path):
        result=original(path)
        if r.Path(path)==c.PREVIOUS_TIMING/name:
            result[key]=value
        return result
    monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError):r.previous_timing_acceptance()


def test_previous_sealed_evidence_mismatch_stops_before_new_work(monkeypatch):
    original=c.read
    def changed(path):
        result=original(path)
        if r.Path(path)==c.PREVIOUS_TIMING/'completed_evidence_seal.json':
            result['artifacts'][0]['sha256']='0'*64
        return result
    monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError,match='Changed frozen input'):r.previous_timing_acceptance()


@pytest.mark.parametrize('corruption',[None,'packet_hash','request_hash','request_incomplete',
    'parent_response_hash','parent_response_bytes','parent_response_overflow','response_result','startup_count'])
def test_final_raw_packet_binding_preserves_byte_order_and_startup_semantics(tmp_path,monkeypatch,corruption):
    import json,hashlib
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    folder=c.OUT/'workers'/'qualification_32_profiled';folder.mkdir(parents=True)
    request_id='request-00000001';metadata={'search_simulations':32}
    payload={'operation':'entrypoint_checks'};result={'z':1,'a':2}
    packet={'type':'response','request_id':request_id,'result':result}
    raw=(json.dumps(packet,separators=(',',':'))+'\n').encode()
    raw_input=b'{"type":"request","request_id":"request-00000001","payload":{"operation":"entrypoint_checks"}}\n'
    receipt={'status':'ok','request_id':request_id,'result':result}
    parent={'request_id':request_id,'request_raw_sha256':hashlib.sha256(raw_input).hexdigest(),
            'request_raw_bytes':len(raw_input),'request_raw_complete':True,'request_raw_overflow':False,
            'response_raw_sha256':hashlib.sha256(raw[:-1]+b'\r\n').hexdigest(),
            'response_raw_bytes':len(raw)+1,'response_capture_overflow':False}
    child={'request_id':request_id,'serialized_packet_utf8_sha256':hashlib.sha256(raw).hexdigest(),
           'serialized_packet_utf8_bytes':len(raw),'request_raw_sha256':parent['request_raw_sha256'],
           'payload_sha256':c.digest(payload),'result_sha256':c.digest(result)}
    if corruption=='packet_hash':child['serialized_packet_utf8_sha256']='0'*64
    if corruption=='request_hash':parent['request_raw_sha256']='0'*64
    if corruption=='request_incomplete':parent['request_raw_complete']=False
    if corruption=='parent_response_hash':parent['response_raw_sha256']='0'*64
    if corruption=='parent_response_bytes':parent['response_raw_bytes']+=1
    if corruption=='parent_response_overflow':parent['response_capture_overflow']=True
    if corruption=='response_result':receipt=deepcopy(receipt);receipt['result']['a']=3
    if corruption=='startup_count':folder=c.OUT/'workers'/'overhead_32_profiled';folder.mkdir(parents=True)
    c.save(folder/'cleanup.json',{'log_compression':{'stdout.log':{}}})
    c.save(folder/'initialization.json',{'result':metadata,'identity':{'pid':42}})
    c.save(folder/(request_id+'.json'),receipt)
    c.save(folder/'startup_inputs'/(request_id+'.json'),payload)
    c.save(folder/'parent_profiles'/(request_id+'.json'),parent)
    c.save(folder/(request_id+'_child_profile.json'),child)
    ready=(json.dumps({'type':'ready','metadata':metadata})+'\r\n').encode()
    wire=ready+raw[:-1]+b'\r\n';validated=[]
    def read_log(path,descriptor,**kwargs):
        for offset in range(0,len(wire),13):kwargs['sink'](wire[offset:offset+13])
        return {'verified':True}
    monkeypatch.setattr(r,'verify_log',read_log)
    monkeypatch.setattr(r.analyze,'validate_startup_profiles',lambda *args,**kwargs:validated.append(args))
    if corruption:
        with pytest.raises(ValueError):r.verify_raw_transport(folder,{},Envelope())
    else:
        checked=r.verify_raw_transport(folder,{},Envelope())
        assert checked['responses']==checked['startup_requests']==1 and checked['exact_bytes_checked']
        assert len(validated)==1 and validated[0][3]==payload
