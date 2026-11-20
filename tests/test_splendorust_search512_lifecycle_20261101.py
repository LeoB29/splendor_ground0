"""Actual saved-identity and original-clock regressions, without model calls."""
from copy import deepcopy
import threading
from types import SimpleNamespace
from pathlib import Path
import pytest
from tools.splendorust_search512_20261101 import common as c,control as r,independent_review as review

WORKSPACE=Path(__file__).resolve().parents[1]
SAVED=WORKSPACE/'outputs/splendorust_network_practical_20261031_01/workers/control/initialization.json'


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    folder=c.OUT/'workers/operational_g000_control';folder.mkdir(parents=True)
    saved=c.read(SAVED);(folder/'initialization.json').write_bytes(SAVED.read_bytes())
    identity=deepcopy(saved['identity'])
    extra={'command','identity','affinity','topology','thread_environment'}
    live={key:deepcopy(value) for key,value in saved.items() if key not in extra}
    assert 'identity' not in live
    return SimpleNamespace(_closed=False,_counter=2,_failure=None,_lock=threading.Lock(),
        process=SimpleNamespace(pid=identity['pid'],poll=lambda:None),identity=identity,
        _ps=SimpleNamespace(pid=identity['pid'],create_time=lambda:saved['identity']['created']),
        initialization=live,absolute_deadline=1900.,output_dir=folder)


def test_lifecycle_exact_routes_quotas_and_full_prefix():
    planned=r.lifecycle_inventory()
    assert len(planned)==814 and set(x['route'] for x in planned.values())==set(c.VARIANTS)
    for name,row in planned.items():
        assert row['quota']==c.ROUTE_QUOTAS[row['route']]
        assert row['startup_requests']==2 and row['graph_request_id']=='request-00000001'
        assert row['entrypoint_request_id']=='request-00000002' and row['ordinary_first_request_id']=='request-00000003'
    assert planned['main_g000_control']['ordinary_requests']==10*1000
    assert planned['main_g007_candidate']['ordinary_requests']==3*1000
    assert planned['main_g399_control']['ordinary_requests']==1000
    assert planned['operational_g000_control']['ordinary_requests']==6000
    assert planned['replay_candidate']['ordinary_requests']==96


def test_actual_saved_unaugmented_ready_shape_hands_off_only_idle_prefix2(tmp_path,monkeypatch):
    child=fixture(tmp_path,monkeypatch)
    envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    r.advance_idle_worker(child,envelope)
    receipt=c.read(c.OUT/'worker_phase_transitions/operational_g000_control.json')
    assert receipt['worker_identity']==child.identity
    assert receipt['completed_startup_requests']==2 and receipt['next_request_id']=='request-00000003'
    assert receipt['active_request'] is False and receipt['request_allowance_unchanged']==10.
    assert receipt['original_monotonic_origin']==100. and receipt['original_unix_origin']==1000.
    assert receipt['old_absolute_deadline']==1900. and child.absolute_deadline==5000.


@pytest.mark.parametrize('fault',['active','failed','prefix1','ordinary_started','closed','exited','cutoff','identity','ready_pid'])
def test_handoff_never_resets_active_failed_or_unowned_clock(tmp_path,monkeypatch,fault):
    child=fixture(tmp_path,monkeypatch);envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    if fault=='active':child._lock.acquire()
    elif fault=='failed':child._failure='timeout'
    elif fault=='prefix1':child._counter=1
    elif fault=='ordinary_started':child._counter=3
    elif fault=='closed':child._closed=True
    elif fault=='exited':child.process.poll=lambda:1
    elif fault=='cutoff':envelope.deadline=100.+c.COLLECTION_CUTOFF_SECONDS+1
    elif fault=='identity':child.identity['created']+=1
    else:child.initialization['result']['pid']+=1
    with pytest.raises(ValueError):r.advance_idle_worker(child,envelope)
    assert child.absolute_deadline==1900.
    assert not(c.OUT/'worker_phase_transitions/operational_g000_control.json').exists()


def test_handoff_publishes_before_mutating_deadline(tmp_path,monkeypatch):
    child=fixture(tmp_path,monkeypatch)
    def refused(*args):raise OSError('test full volume')
    monkeypatch.setattr(c,'save',refused)
    with pytest.raises(OSError):r.advance_idle_worker(child,SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.))
    assert child.absolute_deadline==1900.


@pytest.mark.parametrize('stage',['qualification','operational','main','replay','final'])
def test_late_stage_is_clipped_to_original_two_clocks(tmp_path,monkeypatch,stage):
    fixture(tmp_path,monkeypatch)
    start,wall=100.,1000.
    cutoff=c.JOB_SECONDS if stage=='final' else c.COLLECTION_CUTOFF_SECONDS
    now=start+cutoff-45.;wall_now=wall+cutoff-21.
    envelope=r.Envelope(dict(monotonic_origin=start,unix_origin=wall))
    monkeypatch.setattr(r.time,'monotonic',lambda:now)
    monkeypatch.setattr(r.time,'time',lambda:wall_now)
    monkeypatch.setattr(envelope,'guard',lambda:None)
    envelope.begin(stage)
    assert envelope.deadline==start+cutoff and envelope.wall_deadline==wall+cutoff
    assert envelope.remaining()==21.
    value=c.read(c.OUT/'stages'/f'{stage}_start.json')
    assert value['monotonic_deadline']==start+cutoff and value['unix_deadline']==wall+cutoff


def test_saved_only_final_requires_no_model_trees(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    envelope=r.Envelope(dict(monotonic_origin=r.time.monotonic(),unix_origin=r.time.time()))
    envelope.resident[(123,1.)]=object()
    with pytest.raises(ValueError,match='no resident'):envelope.begin('final')
    assert not(c.OUT/'stages/final_start.json').exists()


def test_reader_deadline_cannot_extend_stage_or_original_global_clock():
    claim=dict(monotonic_origin=100.,unix_origin=1000.)
    final=dict(monotonic_deadline=100.+c.JOB_SECONDS-5,unix_deadline=1000.+c.JOB_SECONDS-9)
    limits=review.review_deadlines(claim,final,100.+c.JOB_SECONDS-25,1000.+c.JOB_SECONDS-30)
    assert limits==(final['monotonic_deadline'],final['unix_deadline'])


@pytest.mark.parametrize('field,value',[('seconds_per_move',11.),('one_second_gate',True),
    ('game_replacements',1),('failed_decision_retries',1),('network_execution','modified_network'),
    ('main_games',200),('max_response_bytes',4*1024**2)])
def test_final_contract_rejects_silent_design_drift(field,value):
    frozen=dict(r.contract(),profiles={'control':{},'candidate':{}})
    assert r.validate_freeze_contract(frozen)
    frozen[field]=value
    with pytest.raises(ValueError):r.validate_freeze_contract(frozen)
