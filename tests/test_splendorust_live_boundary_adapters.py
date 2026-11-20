"""Model-free live observer transport, preserved failure and original clock checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from tools.splendorust_live_boundary_20261025 import common as c, adapters as d, raw_transport as raw
from tools.splendorust_boundary_timing_20261025 import analyze as a


class Envelope:
    def remaining(self):return 100.
    def timeout(self,value):return value
    def guard(self):pass
    def admit(self):pass
    def effective_deadline(self):return None
    def register_worker(self,folder):pass
    def own_worker(self,child):pass
    def close_worker(self,folder,child):pass
    def guard_worker_sample(self,*args):pass


def profile_row(quota=64):
    start = 1_000_000_000
    def stamp(ms):
        return start+ms*1_000_000
    parent = dict(schema=a.PARENT_SCHEMA, request_id="request-00000001", pid=10, child_pid=42,
        clock="perf_counter_ns", clock_implementation="QueryPerformanceCounter()", request_start_ns=start,
        receipt_arrival_ns=stamp(400), receipt_reconstructed_arrival_ns=stamp(400), capture_snapshot_ns=stamp(460),
        raw_arrivals_ns=[stamp(400)], maximum_arrival_events=16, dropped_arrival_events=0,
        request_capture_maximum_bytes=1024*1024, request_raw_complete=True, request_raw_overflow=False,
        request_raw_bytes=100, request_raw_sha256="0"*64,
        response_raw_sha256="1"*64, response_raw_bytes=100, response_capture_overflow=False,
        response_capture_maximum_bytes=4*1024*1024,
        stdin_write_start_ns=stamp(10), stdin_write_end_ns=stamp(18), stdin_flush_start_ns=stamp(18),
        stdin_flush_end_ns=stamp(30), dispatch_complete_ns=stamp(30), dispatch_write_calls=1, dispatch_flush_calls=1,
        dispatch_write_failures=0, dispatch_flush_failures=0, pending_writes=0, pending_flushes=0,
        times_are_diagnostic_not_trusted_deadline_receipts=True, measurement_units="nanoseconds",
        raw_reader_semantics="unchanged raw FileIO.readline then immediate trusted perf_counter_ns",
        child_read_may_precede_dispatch_completion=True, child_post_flush_may_follow_trusted_arrival=True,
        hashes_calculated_after_original_request_returns=True, dispatch_markers_may_be_partial_if_pending=True)
    profile = dict(schema=a.CHILD_SCHEMA, pid=42, clock="perf_counter_ns", measurement_units="nanoseconds",
        request_read_complete_ns=stamp(20), handle_start_ns=stamp(25), handle_end_ns=stamp(225), handle_failed=False,
        times_are_diagnostic_not_trusted_deadline_receipts=True)
    observer, result = {"example": "corrected"}, {"fixed": True}
    child = dict(schema=a.CHILD_SCHEMA, request_id="request-00000001", pid=42, quota=quota,
        backend="qualified_graph64" if quota == 64 else "unchanged_eager32",
        graph_identity_sha256=a.GRAPH_IDENTITY if quota == 64 else None,
        clock="perf_counter_ns", clock_implementation="QueryPerformanceCounter()",
        request_read_complete_ns=stamp(20), request_started_seconds=1.019,
        pre_write_ns=stamp(230), post_flush_ns=stamp(410), outcome="response_flushed", error=None,
        profiles=[profile], request_raw_sha256="0"*64,
        payload_sha256=a._digest({"player": "search", "observer": observer}), result_sha256=a._digest(result),
        serialized_packet_utf8_sha256="1"*64, serialized_packet_utf8_bytes=100,
        times_are_diagnostic_not_trusted_deadline_receipts=True, success_sidecar_written_after_response_flush=True,
        post_flush_may_follow_parent_receipt=True, child_read_may_precede_parent_dispatch_flush_completion=True,
        killed_worker_sidecar_may_be_missing=True, text_transport_may_expand_newlines=True)
    clock = dict(request_id="request-00000001", status="ok", accepted=True, received_before_deadline=True,
        deadline_seconds=1., receipt_elapsed_seconds=.4, result=result,
        telemetry={"process_tree": [{"role": "python", "pid": 42}]})
    return dict(index=0, stage="overhead", fixture_id="saved/one.json", pass_index=0, mode="profiled", quota=quota,
        player_role="challenger" if quota == 64 else "opponent", completed=True, clock=clock, observer=observer,
        child_profile=child, parent_profile=parent)


def setup_root(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    monkeypatch.setattr(c,'WRITE_OBSERVER',None)


def raw_fixture(monkeypatch,tmp_path,kind='completed',corruption=None):
    setup_root(monkeypatch,tmp_path)
    folder=c.OUT/'workers'/'main_g0_opponent';folder.mkdir(parents=True)
    row=profile_row(32);rid=row['clock']['request_id'];parent=row['parent_profile'];child=row['child_profile'];receipt=row['clock']
    receipt.update(clock='perf_counter_ns',elapsed_seconds=.45,detail=None)
    payload={'player':'search','observer':row['observer']}
    wire_packet={'type':'response','request_id':rid,'result':deepcopy(receipt['result'])}
    raw_packet=(json.dumps(wire_packet,separators=(',',':'))+'\r\n').encode()
    if kind=='late':
        receipt.update(status='timeout',accepted=False,received_before_deadline=False,
            receipt_elapsed_seconds=1.1,elapsed_seconds=1.11,result=None)
        parent.update(receipt_arrival_ns=2_100_000_000,receipt_reconstructed_arrival_ns=2_100_000_000,
            capture_snapshot_ns=2_200_000_000,raw_arrivals_ns=[2_100_000_000])
    elif kind=='unobserved_late':
        receipt.update(status='timeout',accepted=False,received_before_deadline=False,
            receipt_elapsed_seconds=None,elapsed_seconds=1.02,result=None)
        parent.update(receipt_arrival_ns=None,receipt_reconstructed_arrival_ns=None,capture_snapshot_ns=2_050_000_000,
            raw_arrivals_ns=[2_010_000_000],response_raw_sha256=None,response_raw_bytes=None,response_capture_overflow=None)
    elif kind=='internal':
        detail="TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: post inference')"
        receipt.update(status='worker_error',accepted=False,receipt_elapsed_seconds=.975,elapsed_seconds=.99,result=None,detail=detail)
        parent.update(receipt_arrival_ns=1_975_000_000,receipt_reconstructed_arrival_ns=1_975_000_000,
            capture_snapshot_ns=2_000_000_000,raw_arrivals_ns=[1_975_000_000])
        child.update(outcome='failed_before_successful_emission',error=detail,pre_write_ns=None,post_flush_ns=None,
            serialized_packet_utf8_sha256=None,serialized_packet_utf8_bytes=None,result_sha256=None)
        child['profiles'][0].update(handle_failed=True,handle_end_ns=1_925_000_000)
        wire_packet={'type':'error','request_id':rid,'error':detail}
        raw_packet=(json.dumps(wire_packet,separators=(',',':'))+'\r\n').encode()
    elif kind in ('missing','truncated'):
        receipt.update(status='timeout' if kind=='missing' else 'malformed',accepted=False,received_before_deadline=False,
            receipt_elapsed_seconds=None,elapsed_seconds=1.02,result=None)
        parent.update(receipt_arrival_ns=None,receipt_reconstructed_arrival_ns=None,capture_snapshot_ns=2_050_000_000,
            raw_arrivals_ns=[],response_raw_sha256=None,response_raw_bytes=None,response_capture_overflow=None)
        child=None;row['child_profile']=None;raw_packet=b'' if kind=='missing' else b'{"type":'
    request=(json.dumps({'type':'request','request_id':rid,'payload':payload},separators=(',',':'))+'\n').encode()
    parent.update(request_raw_sha256=hashlib.sha256(request).hexdigest(),request_raw_bytes=len(request))
    if child is not None:
        child.update(request_raw_sha256=parent['request_raw_sha256'])
        if child['outcome']=='response_flushed':
            normalized=raw_packet[:-2]+b'\n'
            child.update(serialized_packet_utf8_sha256=hashlib.sha256(normalized).hexdigest(),serialized_packet_utf8_bytes=len(normalized))
    if kind not in ('missing','truncated','unobserved_late'):
        parent.update(response_raw_sha256=hashlib.sha256(raw_packet).hexdigest(),response_raw_bytes=len(raw_packet))
    if corruption=='packet_hash':child['serialized_packet_utf8_sha256']='2'*64
    if corruption=='request_hash':parent['request_raw_sha256']='2'*64
    if corruption=='response_hash':parent['response_raw_sha256']='2'*64
    if corruption=='result_hash':child['result_sha256']='2'*64
    if corruption=='receipt_result':receipt['result']={'changed':True}
    if corruption=='payload_hash':child['payload_sha256']='2'*64
    if corruption=='child_pid':child['pid']=99
    if corruption=='child_missing':child=None
    metadata={'search_simulations':32}
    c.save(folder/'cleanup.json',{'log_compression':{'stdout.log':{}}})
    c.save(folder/'initialization.json',{'result':metadata,'identity':{'pid':42}})
    c.save(folder/(rid+'.json'),receipt)
    c.save(folder/'inputs'/(rid+'.json'),dict(player='search',observer=row['observer'],before={}))
    c.save(folder/'parent_profiles'/(rid+'.json'),parent)
    if child is not None:c.save(folder/(rid+'_child_profile.json'),child)
    ready=(json.dumps({'type':'ready','metadata':metadata})+'\r\n').encode();wire=ready+raw_packet
    def log(path,descriptor,**kwargs):
        for offset in range(0,len(wire),17):kwargs['sink'](wire[offset:offset+17])
        return {'verified':True}
    monkeypatch.setattr(raw,'verify_log',log)
    return folder,row,receipt


@pytest.mark.parametrize('kind',['completed','late','internal','missing','truncated'])
def test_raw_transport_preserves_success_and_every_boundary_failure_kind(monkeypatch,tmp_path,kind):
    folder,row,receipt=raw_fixture(monkeypatch,tmp_path,kind)
    before=deepcopy(receipt);checked=raw.verify_raw_transport(folder,{},Envelope())
    assert checked['status']=='passed' and checked['requests']==1
    assert checked['failed_requests']==(kind!='completed')
    assert checked['original_trusted_receipts_unchanged'] and checked['failed_packets_remain_unaccepted']
    assert c.read(folder/(receipt['request_id']+'.json'))==before
    assert checked['requests_checked'][0]['child_sidecar_missing']==(kind in ('missing','truncated'))
    if kind=='late':
        assert checked['requests_checked'][0]['serialized_response_observed']
        assert checked['requests_checked'][0]['response_accepted'] is False
        assert before['result'] is None and before['status']=='timeout'
    if kind=='internal':assert checked['requests_checked'][0]['failed_child_markers_preserved']
    if kind=='truncated':assert checked['partial_failed_output']['complete_packet'] is False


@pytest.mark.parametrize('corruption',['packet_hash','request_hash','response_hash','result_hash',
    'receipt_result','payload_hash','child_pid','child_missing'])
def test_raw_transport_rejects_corrupted_completed_evidence(monkeypatch,tmp_path,corruption):
    folder,_,_=raw_fixture(monkeypatch,tmp_path,corruption=corruption)
    with pytest.raises(ValueError):raw.verify_raw_transport(folder,{},Envelope())


def test_worker_with_no_moves_after_other_players_immediate_failure_is_valid(monkeypatch,tmp_path):
    setup_root(monkeypatch,tmp_path);folder=c.OUT/'workers'/'main_g0_opponent';metadata={'search_simulations':32}
    c.save(folder/'cleanup.json',{'log_compression':{'stdout.log':{}}})
    c.save(folder/'initialization.json',{'result':metadata,'identity':{'pid':42}})
    wire=(json.dumps({'type':'ready','metadata':metadata})+'\n').encode()
    monkeypatch.setattr(raw,'verify_log',lambda path,descriptor,**kwargs:(kwargs['sink'](wire) or {'verified':True}))
    checked=raw.verify_raw_transport(folder,{},Envelope())
    assert checked['requests']==checked['startup_requests']==checked['failed_requests']==0


@pytest.mark.parametrize('mode,quota',[('candidate',64),('baseline',32)])
def test_both_roles_use_same_immutable_observer_and_original_bounds(monkeypatch,tmp_path,mode,quota):
    setup_root(monkeypatch,tmp_path);calls=[];closed=[];proof=[]
    metadata={'search_simulations':quota}
    if quota==64:metadata['inference_backend']=d.BACKEND
    child=SimpleNamespace(process=SimpleNamespace(pid=42),initialization={'result':metadata},_failure=None,
        _sample=lambda:{'process_tree':[]},close=lambda:closed.append(True))
    monkeypatch.setattr(d,'ProfiledStudyWorker',lambda *args,**kwargs:(calls.append((args,kwargs)) or child))
    from tools.splendorust_search_scale_20261024 import adapters as old
    monkeypatch.setattr(old,'validate_ready_metadata',lambda *args,**kwargs:True)
    monkeypatch.setattr(d,'verify_instrumentation_identity',lambda *args:proof.append('identity'))
    monkeypatch.setattr(d,'graph_parity',lambda *args:proof.append('graph'))
    freeze={'native_artifacts':[],'topology':{'selected_logical_processor':3}}
    result=d.make_worker('test_'+str(quota),Envelope(),freeze,mode=mode,quota=quota)
    args,kwargs=calls[0]
    assert result is child and args[0]=='tools.splendorust_boundary_timing_20261025.worker'
    assert child.runtime_mode==mode and child.challenger_quota==quota and child.observation_mode=='profiled'
    assert kwargs['env']['STUDY_BOUNDARY_PROFILE']=='1' and kwargs['env']['STUDY_SEARCH_SIMULATIONS']==str(quota)
    assert kwargs['max_message_bytes']==4*c.MiB and kwargs['max_request_bytes']==c.MiB and kwargs['max_log_bytes']==16*c.MiB
    assert kwargs['initialization_timeout']==60 and kwargs['affinity']==[3] and not closed
    assert proof==(['identity','graph'] if quota==64 else ['identity'])


@pytest.mark.parametrize('mode,quota',[('candidate',32),('baseline',64),('profiled',64),('candidate',True)])
def test_role_quota_mixing_rejected_before_worker(monkeypatch,mode,quota):
    monkeypatch.setattr(d,'ProfiledStudyWorker',lambda *a,**k:pytest.fail('invalid worker launch'))
    with pytest.raises(ValueError):d.make_worker('bad',Envelope(),{},mode=mode,quota=quota)


@pytest.mark.parametrize('status,present',[('ok',True),('worker_error',True),('timeout',False),('ok',False)])
def test_collect_preserves_partial_failures_without_waiting_on_dead_worker(monkeypatch,tmp_path,status,present):
    setup_root(monkeypatch,tmp_path);folder=c.OUT/'workers'/'main_g0_opponent';rid='request-00000001'
    parent={'request_id':rid,'unchanged':True};child_value={'request_id':rid,'available':True}
    if present:c.save(folder/(rid+'_child_profile.json'),child_value)
    child=SimpleNamespace(last_parent_profile=parent,output_dir=folder,_closed=True,runtime_mode='baseline')
    receipt={'status':status,'request_id':rid};before=deepcopy(receipt)
    monkeypatch.setattr(d.time,'sleep',lambda _:pytest.fail('must not wait for closed worker'))
    result=d.collect_profiles(child,receipt,Envelope())
    assert result['parent_profile']==parent and c.read(folder/'parent_profiles'/(rid+'.json'))==parent
    assert receipt==before and result['child_profile']==(child_value if present else None)
    if status=='ok' and not present:assert result['profile_capture_status']=='evidence_error' and 'profile_capture_error' in result
    else:assert 'profile_capture_error' not in result


def test_present_malformed_child_evidence_is_integrity_error_not_missing(monkeypatch,tmp_path):
    setup_root(monkeypatch,tmp_path);folder=c.OUT/'workers'/'x';folder.mkdir(parents=True);rid='request-00000001'
    (folder/(rid+'_child_profile.json')).write_text('{bad')
    child=SimpleNamespace(last_parent_profile={'request_id':rid},output_dir=folder,_closed=True)
    result=d.collect_profiles(child,{'request_id':rid,'status':'timeout'},Envelope())
    assert result['profile_capture_status']=='evidence_error' and 'profile_capture_error' in result
    assert (folder/(rid+'_child_profile.json')).read_text()=='{bad'


@pytest.mark.parametrize('status,received,detail',[('timeout',1.05,None),
    ('worker_error',.975,"TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: post inference')")])
def test_decision_keeps_failed_original_receipt_and_never_plays_late_action(monkeypatch,tmp_path,status,received,detail):
    setup_root(monkeypatch,tmp_path)
    state=d.rules.CorrectedEnv(seed=13).initial_state();folder=c.OUT/'workers'/'main_g0_challenger'
    receipt=dict(request_id='request-00000001',status=status,accepted=False,result=None,
        detail=detail,deadline_seconds=1.,receipt_elapsed_seconds=received,
        elapsed_seconds=received+.01,received_before_deadline=received<=1.,clock='perf_counter_ns')
    before=deepcopy(receipt);calls=[]
    def request(payload,allowance,validate):calls.append((payload,allowance,validate));return receipt
    child=SimpleNamespace(runtime_mode='candidate',challenger_quota=64,observation_mode='profiled',
        _counter=0,output_dir=folder,request=request,_closed=True,
        last_parent_profile={'request_id':'request-00000001'})
    row=d.decision(child,state,Envelope(),'challenger')
    assert row['clock'] is receipt and receipt==before and row['completed'] is False
    assert row['action'] is row['trace'] is row['after'] is None
    assert row['observation_mode']=='profiled' and row['mode']=='candidate' and row['quota']==64
    assert calls[0][1]==1. and calls[0][0]=={'player':'search','observer':row['observer']}
    assert row['profile_capture_status']=='missing_child_after_failed_request'
    assert c.read(folder/'inputs'/'request-00000001.json')['before']==row['before']


@pytest.mark.parametrize('remaining,allowance',[(.999,1.),(2.,.999)])
def test_shortened_stage_budget_never_dispatches_a_move(monkeypatch,tmp_path,remaining,allowance):
    setup_root(monkeypatch,tmp_path);state=d.rules.CorrectedEnv(seed=13).initial_state()
    child=SimpleNamespace(runtime_mode='baseline',challenger_quota=32,observation_mode='profiled',_counter=0,
        output_dir=c.OUT/'workers'/'main_g0_opponent',request=lambda *a,**k:pytest.fail('clamped move dispatched'))
    env=Envelope();env.remaining=lambda:remaining;env.timeout=lambda value:allowance
    with pytest.raises(ValueError):d.decision(child,state,env,'opponent')


def test_actual_entrypoint_calls_original52_suite_with300_bound_then_checks_profiles(monkeypatch,tmp_path):
    setup_root(monkeypatch,tmp_path);folder=c.OUT/'workers'/'qualification_32';calls=[]
    receipt={'status':'ok','request_id':'request-00000001','result':{'status':'entrypoint_counterexamples_passed',
        'challenger_quota':32,'counts':{'complete_decisions':52}}}
    child=SimpleNamespace(output_dir=folder,_counter=0,challenger_quota=32,process=SimpleNamespace(pid=42),
        request=lambda payload,timeout:(calls.append((payload,timeout)) or receipt))
    profiles={'parent_profile':{'parent':True},'child_profile':{'child':True}}
    monkeypatch.setattr(d,'collect_profiles',lambda *a,**k:profiles)
    validated=[];monkeypatch.setattr(d.observer_audit,'validate_startup_profiles',lambda *a,**k:validated.append((a,k)))
    assert d.actual_entrypoint(child,Envelope()) is receipt
    assert calls==[({'operation':'entrypoint_checks'},300)]
    assert validated[0][0][2] is receipt and validated[0][1]=={'worker_pid':42}
    assert c.read(folder/'startup_inputs'/'request-00000001.json')=={'operation':'entrypoint_checks'}
    assert not (c.OUT/'entrypoint_checks_32.json').exists()


@pytest.mark.parametrize('corrupt',[False,True])
def test_late_complete_packet_after_timeout_poll_requires_exact_child_hash_even_without_trusted_arrival(monkeypatch,tmp_path,corrupt):
    folder,_,receipt=raw_fixture(monkeypatch,tmp_path,'unobserved_late',corruption='packet_hash' if corrupt else None)
    assert receipt['receipt_elapsed_seconds'] is None and receipt['result'] is None
    if corrupt:
        with pytest.raises(ValueError,match='serialized packet differs'):raw.verify_raw_transport(folder,{},Envelope())
    else:
        checked=raw.verify_raw_transport(folder,{},Envelope())
        assert checked['requests_checked'][0]['serialized_response_observed']
        assert checked['requests_checked'][0]['response_accepted'] is False
        assert not checked['requests_checked'][0]['parent_trusted_packet_observed']
