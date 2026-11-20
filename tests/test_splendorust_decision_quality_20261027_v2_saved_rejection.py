"""Model-free regression of the real rejected response, never a recovered run.

The integration fixture copies the finite 142-file preserved attempt into pytest
temporary storage. Only the copied research wrapper module label is projected
to this v2 reader; clocks, packets, sidecars, graph and entrypoint evidence stay
byte-identical. Missing stop/summary files are derived test artifacts. This does
not accept or change the historical incomplete attempt.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.splendorust_decision_quality_20261027_v2 import audit, common as c, runner as r
from tools.local_player_qualification_128 import clock as original_clock

ROOT = c.ROOT
OLD = ROOT/'outputs/splendorust_decision_quality_20261027_01'
MANIFEST = OLD/'postrun/attempt_preservation_inventory.json'
MANIFEST_SHA = 'f54e7767540d8143718ecc76680cad5f89c36c9d35bb67c2c3526bc65663c772'


def preserved_items():
    raw = MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA
    value = json.loads(raw)
    assert value['files'] == len(value['artifacts']) == 142
    return {item['path']: item for item in value['artifacts']}


def bound_bytes(path):
    item = preserved_items()[path.relative_to(ROOT).as_posix()]
    raw = path.read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    return raw


def saved_row(index=2):
    return json.loads(bound_bytes(OLD/'qualification_rows'/f'{index:04d}.json'))


def payload_for(row):
    return json.loads(bound_bytes(ROOT/row['worker_folder']/'inputs'/(row['clock']['request_id']+'.json')))


def test_actual_illegal_receipt_retains_result_without_accepting_it():
    row = saved_row(); before = deepcopy(row)
    audit.check_clock(row['clock'], limit=30.)
    assert row == before and row['completed'] is False
    assert row['action'] is row['trace'] is None
    assert row['clock']['status'] == 'illegal' and row['clock']['accepted'] is False
    assert row['clock']['receipt_elapsed_seconds'] == .4252405
    assert not r.deadline_failure(row['clock'], 'research')


@pytest.mark.parametrize('change',[
    {'accepted':True}, {'status':'timeout'}, {'status':'worker_error'},
    {'deadline_seconds':31.}, {'received_before_deadline':False},
    {'receipt_elapsed_seconds':30.1}, {'detail':None}, {'status':'invented_failure'},
])
def test_retained_result_cannot_be_relabelled_or_extend_the_clock(change):
    receipt = saved_row()['clock']; receipt.update(change)
    with pytest.raises(ValueError): audit.check_clock(receipt,limit=30.)


@pytest.mark.parametrize('index',[0,2])
def test_frozen_and_research_returned_rejections_keep_completed_worker_handle(index):
    row = saved_row(index)
    if index == 0:
        # Synthetic parent rejection of the actual completed frozen response.
        row['clock'].update(status='illegal',accepted=False,detail='synthetic validation rejection')
    before = deepcopy(row['clock'])
    result = r.validate_profiles(row['parent_profile'],row['child_profile'],row['clock'],
        payload_for(row),row['route'],worker_pid=row['child_profile']['pid'])
    assert row['clock']==before
    assert row['child_profile']['outcome']=='response_flushed'
    assert row['child_profile']['profiles'][0]['handle_failed'] is False
    if index == 2:
        assert result['status']=='passed'
        assert result['accepted_science'] is False and result['rejected_response_retained'] is True
    row['child_profile']['profiles'][0]['handle_failed']=True
    with pytest.raises(ValueError,match='completed worker handle'):
        r.validate_profiles(row['parent_profile'],row['child_profile'],row['clock'],payload_for(row),row['route'])


def test_missing_rejected_sidecar_is_disclosed_partial_not_complete_evidence():
    row=saved_row()
    result=r.validate_profiles(row['parent_profile'],None,row['clock'],payload_for(row),'research')
    assert result['status']=='preserved_failed_request_missing_child'
    assert not r.deadline_failure(row['clock'],'research')


def rejection_from_original_receive(row, validate):
    """Feed actual saved bytes into the unchanged trusted receive implementation."""
    raw_stream=gzip.decompress(bound_bytes(ROOT/row['worker_folder']/'stdout.log.gz'))
    matches=[line for line in raw_stream.splitlines(keepends=True)
             if json.loads(line).get('request_id')==row['clock']['request_id']]
    assert len(matches)==1
    telemetry=deepcopy(row['clock']['telemetry'])
    events=queue.Queue();events.put(('line',row['parent_profile']['receipt_arrival_ns'],matches[0]))
    proxy=SimpleNamespace(absolute_deadline=None,_failure=None,_failure_detail=None,
        _queue=events,_check_resource_guard=lambda:None,_sample=lambda:deepcopy(telemetry))
    start=row['parent_profile']['request_start_ns']
    end=start+round(row['clock']['elapsed_seconds']*1e9)
    with patch.object(original_clock.time,'perf_counter_ns',return_value=end):
        return original_clock.PersistentWorker._receive(proxy,start,30.,'response',row['clock']['request_id'],
            validate,before_cpu=telemetry['cpu_seconds']-telemetry['request_cpu_seconds'])


def test_actual_rejected_receive_to_row_closed_worker_and_final_prefix(tmp_path,monkeypatch):
    """One small saved-evidence readback, no process/model/game or source audit."""
    original_items=preserved_items()
    for relative,item in original_items.items():
        source=ROOT/relative;raw=bound_bytes(source);target=tmp_path/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    target_out=tmp_path/OLD.relative_to(ROOT)
    freeze=json.loads(bound_bytes(OLD/'freeze.json'))
    graph_item=freeze['graph_parity_plan'];raw=(ROOT/graph_item['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==graph_item['sha256'] and len(raw)==graph_item['bytes']
    target=tmp_path/graph_item['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',target_out)
    env=SimpleNamespace(admit=lambda:None,guard=lambda:None,remaining=lambda:999.,timeout=lambda n:n,
        close_worker=lambda *args:None,effective_deadline=lambda:None,placement_sequence=1,peak_combined_rss=0)
    row=saved_row();folder=tmp_path/row['worker_folder']
    # Only a derived test copy is changed to the v2 wrapper's expected name.
    identity=c.read(folder/'route_identity.json')
    identity['module']='tools.splendorust_decision_quality_20261027_v2.worker'
    (folder/'route_identity.json').write_bytes(c.canonical(identity).encode())
    # Exercise ordinary() writing precisely the same evidence anew in tmp only.
    (folder/'inputs'/f"{row['clock']['request_id']}.json").unlink()
    (tmp_path/row['parent_profile_file']).unlink()
    child=SimpleNamespace(route='research',_counter=3,output_dir=folder,_closed=True,
        placement_mode='isolated',last_placement_evidence=deepcopy(row['placement_evidence']),
        last_placement_error=None,last_parent_profile=deepcopy(row['parent_profile']))
    def receive(child,payload,allowance,validate=None):
        assert allowance==30. and payload==payload_for(row)
        value=rejection_from_original_receive(row,validate)
        assert value==row['clock']
        return value
    with patch.object(r,'request',side_effect=receive),\
         patch.object(audit,'profile_identity',return_value={}),\
         patch.object(audit,'validate_research_result',side_effect=ValueError(row['clock']['detail'])),\
         patch.object(r.subprocess,'Popen',side_effect=AssertionError('No subprocess permitted')):
        returned=r.ordinary(child,row['spec'],row['observer'],env,freeze)
        assert returned==row
        assert r.verify_saved_record(returned,freeze)['status']=='passed'
        closed=r.verify_closed_worker(folder,freeze,env)
        assert closed['status']=='passed' and closed['raw']['requests']==4
        # A rejected value is still exactly bound to its captured raw packet.
        # Even a matching edited child hash cannot authenticate different data.
        clock_path=folder/(row['clock']['request_id']+'.json')
        child_path=folder/(row['clock']['request_id']+'_child_profile.json')
        clock_raw,child_raw=clock_path.read_bytes(),child_path.read_bytes()
        tampered=deepcopy(row['clock']);tampered['result']['forensic_tamper']=True
        sidecar=deepcopy(row['child_profile']);sidecar['result_sha256']=c.digest(tampered['result'])
        try:
            clock_path.write_bytes(c.canonical(tampered).encode())
            child_path.write_bytes(c.canonical(sidecar).encode())
            with pytest.raises(ValueError,match='Child raw scientific response binding changed'):
                r.verify_raw_transport(folder,freeze,env)
        finally:
            clock_path.write_bytes(clock_raw);child_path.write_bytes(child_raw)
    # Final-prefix artifacts did not exist because the old checker failed; they
    # are synthesized only in this temporary harness from the exact old receipt.
    stop=dict(phase='qualification',index=2,clock=row['clock'],route='research',no_retry=True,
        diagnostic_deadline_failure=False)
    summary=dict(qualified=False,requests=3,failure=stop)
    c.save(target_out/'collection_stop.json',stop)
    c.save(target_out/'qualification_summary.json',summary)
    panel=json.loads((ROOT/freeze['panel']['path']).read_bytes())
    matrix=json.loads((ROOT/freeze['request_matrix']['path']).read_bytes())
    plan=json.loads(bound_bytes(OLD/'qualification_plan.json'))
    from tools.splendorust_search_scale_20261024 import qualification as old_qualification
    with patch.object(r,'verify_runtime') as runtime,\
         patch.object(old_qualification,'artifact',side_effect=c.artifact),\
         patch.object(r.subprocess,'Popen',side_effect=AssertionError('No subprocess permitted')):
        coverage=r.final_checks(env,freeze,panel,matrix,plan,{},summary)
    runtime.assert_called_once_with(freeze,env)
    assert coverage['complete'] is False
    final=c.read(target_out/'verification_summary.json')
    assert final['parent_requests']==7 and final['graph_cases']==22 and final['workers']==2
    assert not (target_out/'records').exists()
    assert c.read(target_out/'collection_stop.json')['clock']==row['clock']
    # Test the full run's disposition independently of the expensive runtime
    # audit: a verified rejected prefix still exits incomplete, never success.
    run_out=tmp_path/'derived_run';run_out.mkdir()
    (run_out/'freeze.json').write_bytes(bound_bytes(OLD/'freeze.json'))
    sha=c.file_hash(run_out/'freeze.json')
    claim=dict(freeze_sha256=sha,monotonic_origin=0.,unix_origin=0.)
    monkeypatch.setattr(c,'OUT',run_out)
    c.save(run_out/'watchdog_claim.json',claim)
    runenv=SimpleNamespace(stage=None,start=0.,wall_start=0.,resident={},
        enable_evidence_accounting=lambda:None,begin=lambda stage:setattr(runenv,'stage',stage),
        finish=lambda:None,admit=lambda:None)
    def stopped(*args):
        c.save(run_out/'collection_stop.json',stop)
        return summary
    with patch.object(r,'Envelope',return_value=runenv),patch.object(r,'verify_runtime'),\
         patch.object(r,'verify_runtime_sources',return_value={}),patch.object(r,'switch_placement'),\
         patch.object(r,'load_inputs',return_value=(panel,matrix,plan,{})),\
         patch.object(r,'qualify',side_effect=stopped),patch.object(r,'final_checks',return_value=coverage),\
         patch.object(r,'collect') as no_collection,\
         patch.object(r.subprocess,'Popen',side_effect=AssertionError('No subprocess permitted')):
        assert r.run(sha)==1
    no_collection.assert_not_called()
    completion=c.read(run_out/'completion.json')
    assert completion['status']=='incomplete' and completion['failure']
    assert completion['automatic_retries']==0 and completion['new_games']==0
    # The original finite evidence files remain byte-for-byte preserved.
    for relative in original_items:bound_bytes(ROOT/relative)


def test_qualification_saves_original_stop_before_secondary_audit_failure(tmp_path):
    row=saved_row();spec=row['spec'];env=SimpleNamespace()
    child=SimpleNamespace(route='research')
    # Keep the finite route loop but make the first route's parity inventory
    # empty; the synthetic startup receipt is irrelevant to this ordering test.
    plan={'parity_requests':[spec]}
    from tools.splendorust_decision_quality_20261027_v2 import adapters
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'make_worker',return_value=child),\
         patch.object(adapters,'actual_entrypoint',return_value={}),patch.object(r,'ordinary',return_value=row),\
         patch.object(r,'verify_saved_record',side_effect=ValueError('secondary evidence failure')),\
         patch.object(r,'close_and_audit'):
        with pytest.raises(ValueError,match='secondary evidence failure'):r.qualify(env,{},plan)
    assert c.read(tmp_path/'collection_stop.json')['clock']==row['clock']
    assert c.read(tmp_path/'qualification_rows/0002.json')==row
