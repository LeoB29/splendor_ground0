"""Model-free route, clock-origin, quota and failure-emission regressions."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_startup_logging_20261027_02 import worker as w


def context(route='frozen'):
    return dict(logging_contract=dict(route=route,arm_id=route+'_cpu0',specs=[]),
        boundary_handle_active=False,component_profiles=[],request_started=123.5,
        request_read_complete_ns=120,quota=64,graph_parity_passed=True,
        component_backend='qualified_graph64' if route=='frozen' else 'research_graph',
        component_backend_identity='0'*64,
        qualification_active=False,diagnostic_capture=False,logging_index=0,
        probe_recorders={},probe_active=None)


def test_frozen_search_reference_search_never_carries_reference_quota(monkeypatch):
    ctx=context();seen=[]
    def original(ctx,req):
        seen.append((req['player'],ctx['quota'],ctx['request_started']))
        return dict(simulations_quota=16 if req['player']=='reference' else 64)
    monkeypatch.setattr(w,'FROZEN_ORDINARY',original)
    for arm in ('search','reference','search'):w._ordinary_boundary(ctx,dict(player=arm,observer={}))
    assert seen==[('search',64,123.5),('reference',64,123.5),('search',64,123.5)]
    assert ctx['quota']==64 and len(ctx['component_profiles'])==3


def test_research_route_keeps_its_original_variable_qualification_quota(monkeypatch):
    ctx=context('research')
    def original(ctx,req):
        ctx['quota']=16 if req['player']=='reference' else 64
        return dict(simulations_quota=ctx['quota'])
    monkeypatch.setattr(w,'candidate_ordinary',original)
    for arm in ('reference','search'):
        result=w._ordinary_boundary(ctx,dict(player=arm,observer={}))
        assert ctx['quota']==result['simulations_quota']
    assert ctx['quota']==64


def test_failure_keeps_original_origin_and_failed_boundary(monkeypatch):
    ctx=context()
    def fail(*_):raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: fixture')
    monkeypatch.setattr(w,'FROZEN_ORDINARY',fail)
    with pytest.raises(TimeoutError):w._ordinary_boundary(ctx,dict(player='search',observer={}))
    assert ctx['request_started']==123.5 and ctx['quota']==64
    assert ctx['component_profiles'][0]['handle_failed'] is True and not ctx['boundary_handle_active']


def test_frozen_mode_selected_from_frozen_local_order_and_original_payload(monkeypatch):
    ctx=context();observer=dict(public='fixture')
    spec=dict(observer=observer,logging_timing_enabled=False,sequence=9,arm_id='frozen_cpu0',case_id='private_reserves')
    ctx['logging_contract']['specs']=[spec]
    rec=SimpleNamespace(entries=[],record=lambda event:None)
    ctx['probe_recorders']={False:rec}
    def original(c,request):
        assert request==dict(player='search',observer=observer)
        assert c['diagnostic_capture'] is True and c['request_started']==123.5
        assert c['evidence_callback']==rec.record
        return dict(simulations_quota=64)
    monkeypatch.setattr(w,'FROZEN_ORDINARY',original)
    w.handle(ctx,dict(player='search',observer=observer))
    assert ctx['logging_index']==1 and not ctx['diagnostic_capture'] and 'evidence_callback' not in ctx
    assert ctx['probe_active']['binding']['worker_local_index']==0
    with pytest.raises(ValueError,match='No extra'):w.handle(ctx,dict(player='search',observer=observer))


@pytest.mark.parametrize('bad',[dict(player='search',observer={},logging_timing_enabled=True),
                              dict(player='reference',observer={}),dict(player='search',observer={'wrong':1})])
def test_extra_controls_and_wrong_fixture_never_reach_player(bad):
    ctx=context();ctx['logging_contract']['specs']=[dict(observer={})]
    with pytest.raises(ValueError,match='Exact frozen'):w.handle(ctx,bad)
    assert ctx['logging_index']==0


def test_graph_gate_cannot_be_bypassed():
    ctx=context();ctx['graph_parity_passed']=False
    with pytest.raises(ValueError,match='graph parity'):w.handle(ctx,dict(player='search',observer={}))


def test_actual_suite_routes_rejections_through_same_handler_without_global_mutation(tmp_path,monkeypatch):
    ctx=context();ctx['folder']=tmp_path
    ctx['adapters']={arm:SimpleNamespace(admission_guard=None) for arm in ('search','reference')}
    original_global=w.original_worker.handle
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(available=4*1024**3))
    def ordinary(c,request):
        assert c['request_started'] is None and c['diagnostic_capture'] and c['qualification_active']
        if set(request)!={'player','observer'}:raise ValueError('fixture rejected')
        c['evidence_callback'](dict(event='complete_result_before_worker_validation'))
        return dict(simulations_quota=16 if request['player']=='reference' else 64)
    monkeypatch.setattr(w,'FROZEN_ORDINARY',ordinary)
    def suite(adapters,*,boundary,request_callback,evidence_callback):
        boundary()
        request_callback(dict(player='search',observer={}))
        request_callback(dict(player='reference',observer={}))
        request_callback(dict(player='search',observer={}))
        with pytest.raises(ValueError):request_callback(dict(player='search',observer={},hidden='forbidden'))
        return dict(status='mock52suite')
    monkeypatch.setattr(w.original_qualification,'run_entrypoint_checks',suite)
    assert w.handle(ctx,dict(operation='entrypoint_checks'))['status']=='mock52suite'
    assert ctx['quota']==64 and ctx['request_started']==123.5
    assert not ctx['qualification_active'] and not ctx['diagnostic_capture']
    assert len(ctx['component_profiles'])==4 and ctx['component_profiles'][-1]['handle_failed']
    assert w.original_worker.handle is original_global
    assert ctx['probe_active']['recorder'].closed


def test_new_entrypoint_has_no_imported_global_monkeypatch_or_deadline_reset_in_ordinary():
    tree=ast.parse(Path(w.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='handle')
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign):
            assert not any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and
                           t.value.id in ('original_worker','boundary_worker') for t in node.targets)
            assert not any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant) and
                           t.slice.value=='request_started' for t in node.targets)
    assert w.FROZEN_ORDINARY is w.original_worker.handle
    assert w.INTERNAL_SECONDS==.9


def test_error_probe_write_failure_cannot_emit_a_second_error_packet(tmp_path,monkeypatch):
    import threadpoolctl
    ctx=context();ctx['folder']=tmp_path;ctx['metadata']={}
    payload=json.dumps(dict(type='request',request_id='request-00000001',payload=dict(player='search',observer={}))).encode()+b'\n'
    monkeypatch.setattr(w.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(payload)))
    output=io.StringIO();monkeypatch.setattr(w.sys,'stdout',output)
    monkeypatch.setattr(threadpoolctl,'threadpool_limits',lambda **_:contextlib.nullcontext())
    monkeypatch.setattr(w,'initialize',lambda:ctx)
    def fail(*_):raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: fixture')
    monkeypatch.setattr(w,'handle',fail)
    monkeypatch.setattr(w,'_save',lambda *a,**k:None)
    monkeypatch.setattr(w,'publish_profile',lambda *a,**k:None)
    sidecar_errors=[];monkeypatch.setattr(w,'save_sidecar_error',lambda *a,**k:sidecar_errors.append((a,k)))
    probes=[]
    def failed_probe(*args):
        probes.append(args)
        assert [json.loads(line)['type'] for line in output.getvalue().splitlines()]==['ready']
        emission=args[-1]
        assert emission['kind']=='before_error' and emission['post_flush_ns'] is None
        raise OSError('probe persistence fixture failure')
    monkeypatch.setattr(w,'_publish_probe',failed_probe)
    monkeypatch.setattr(w,'close_context',lambda *_:None)
    assert w.serve()==1
    packets=[json.loads(line) for line in output.getvalue().splitlines()]
    assert [p['type'] for p in packets]==['ready','error'] and sidecar_errors and len(probes)==1


def test_success_probe_is_published_only_after_response_flush(tmp_path,monkeypatch):
    import threadpoolctl
    ctx=context();ctx['folder']=tmp_path;ctx['metadata']={}
    payload=json.dumps(dict(type='request',request_id='request-00000001',payload=dict(player='search',observer={}))).encode()+b'\n'
    monkeypatch.setattr(w.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(payload)))
    output=io.StringIO();monkeypatch.setattr(w.sys,'stdout',output)
    monkeypatch.setattr(threadpoolctl,'threadpool_limits',lambda **_:contextlib.nullcontext())
    monkeypatch.setattr(w,'initialize',lambda:ctx)
    monkeypatch.setattr(w,'handle',lambda *_:dict(scientific='unchanged'))
    monkeypatch.setattr(w,'publish_profile',lambda *a,**k:None)
    calls=[]
    def publish(*args):
        assert [json.loads(line)['type'] for line in output.getvalue().splitlines()]==['ready','response']
        assert args[-1]['kind']=='response' and args[-1]['post_flush_ns'] is not None
        calls.append(args)
    monkeypatch.setattr(w,'_publish_probe',publish)
    monkeypatch.setattr(w,'close_context',lambda *_:None)
    assert w.serve()==0 and len(calls)==1


def test_probe_close_error_cannot_skip_any_probe_or_original_player_cleanup(monkeypatch):
    ctx=context();calls=[]
    def first():calls.append('first');raise OSError('first probe close failure')
    def second():calls.append('second')
    ctx['probe_recorders']={False:SimpleNamespace(closed=False,close=first),
                            True:SimpleNamespace(closed=False,close=second)}
    def numerical(_):calls.append('numerical');raise RuntimeError('later cleanup failure')
    monkeypatch.setattr(w.boundary_worker,'close_context',numerical)
    with pytest.raises(OSError,match='first probe close failure'):w.close_context(ctx)
    assert calls==['first','second','numerical']
