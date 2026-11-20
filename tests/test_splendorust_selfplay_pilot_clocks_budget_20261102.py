"""Clock regression and real model-free six-response process handshake.

Historical timestamps are used only as labelled synthetic regression inputs.
No missing historical acknowledgement is recreated or retrospectively accepted.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_selfplay_pilot_budget_20261102 import clock_contract as clocks, training

ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / 'outputs/splendorust_selfplay_pilot_qualification_20261102_01/fit/learned_entrypoint'


def sample(qpc=100., owner=90., wall=1000.):
    return dict(request_qpc=qpc, owner_monotonic=owner, unix=wall)


def fixture():
    start = dict(schema='selfplay_learned_request_start_v2', request_index=2,
        case_id='gold_returns', mode='eager', candidate_manifest={'fixed':'candidate'},
        **clocks.start_fields(sample()))
    descriptor={'path':'synthetic/start','sha256':'a'*64,'bytes':1}
    response=dict(schema='selfplay_learned_response_v2',request_index=2,case_id='gold_returns',mode='eager',
        candidate_manifest=start['candidate_manifest'],start=descriptor,
        **clocks.completion_fields(start,sample(100.7,90.68,1000.7)))
    observed=sample(100.71,90.69,1000.71)
    ack=dict(schema='selfplay_learned_parent_receipt_v2',accepted=True,clock_contract=start['clock_contract'],
        request_index=2,start=descriptor,response={'path':'synthetic/response'},parent_observed_qpc=observed['request_qpc'],
        parent_observed_owner_monotonic=observed['owner_monotonic'],parent_observed_unix=observed['unix'],
        receipt_elapsed_seconds=observed['request_qpc']-start['request_qpc_start'],deadline_seconds=10.,
        response_complete_before_deadline=True,original_request_clock_preserved=True)
    return start,response,observed,ack


def received(start,response,observed):
    return clocks.validate_received(start,response,observed=observed,start_artifact={'path':'synthetic/start','sha256':'a'*64,'bytes':1},
        request_index=2,case_id='gold_returns',mode='eager',candidate_manifest={'fixed':'candidate'})


def test_runtime_contract_explicitly_names_each_independent_domain():
    value=clocks.runtime_contract()
    assert value['request_clock']['api']=='time.perf_counter'
    assert value['owner_clock']['api']=='time.monotonic' and value['wall_clock']['api']=='time.time'
    assert value['internal_seconds']==9. and value['complete_response_seconds']==10.
    assert value['no_clock_domain_conversion'] and value['added_tolerance_seconds']==0.
    assert value['request_clock']['implementation']==time.get_clock_info('perf_counter').implementation


def test_distinct_clock_orderings_are_never_compared_to_each_other():
    start,response,observed,ack=fixture()
    assert response['finished_qpc']>observed['owner_monotonic']
    assert received(start,response,observed)['status']=='passed'
    assert clocks.validate_ack(start,response,ack)['receipt_elapsed_seconds']==observed['request_qpc']-100.
    assert clocks.validate_admission(start,owner_start=80.,owner_unix_start=999.,owner_deadline=105.,unix_deadline=1011.)


@pytest.mark.parametrize('change,branch',[
    ('qpc_deadline','parent_request_qpc_deadline'),('wall_deadline','parent_wall_deadline'),
    ('future_qpc','request_qpc_order'),('future_owner','owner_clock_order'),('future_wall','wall_clock_order'),
    ('before_start','request_qpc_order'),('identity','complete_response_identity'),
    ('old_schema','complete_response_identity'),('bool_index','complete_response_identity'),
    ('mixed_contract','complete_response_clock_contract'),('elapsed','child_elapsed_qpc')])
def test_response_order_identity_and_exact_deadlines_fail_closed(change,branch):
    start,response,observed,_=fixture()
    if change=='qpc_deadline':observed['request_qpc']=start['request_qpc_deadline']
    elif change=='wall_deadline':observed['unix']=start['unix_deadline']
    elif change=='future_qpc':response['finished_qpc']=observed['request_qpc']+.001
    elif change=='future_owner':response['finished_owner_monotonic']=observed['owner_monotonic']+.001
    elif change=='future_wall':response['finished_unix']=observed['unix']+.001
    elif change=='before_start':response['finished_qpc']=start['request_qpc_start']-.001
    elif change=='identity':response['candidate_manifest']={'wrong':'parent'}
    elif change=='old_schema':response['schema']='selfplay_learned_response_v1'
    elif change=='bool_index':response['request_index']=True
    elif change=='mixed_contract':response['clock_contract']=dict(response['clock_contract'],request_clock=response['clock_contract']['owner_clock'])
    else:response['elapsed_seconds']+=.001
    with pytest.raises(clocks.ClockContractError) as caught:received(start,response,observed)
    assert caught.value.branch==branch


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),True,'100'])
@pytest.mark.parametrize('where',['start','response','observed'])
def test_nonfinite_and_wrong_type_clocks_rejected(value,where):
    start,response,observed,_=fixture()
    if where=='start':start['owner_monotonic_start']=value
    elif where=='response':response['finished_qpc']=value
    else:observed['request_qpc']=value
    with pytest.raises(clocks.ClockContractError):received(start,response,observed)


def test_owner_and_wall_limits_are_not_extended_or_reset():
    start,_,_,_=fixture()
    with pytest.raises(clocks.ClockContractError,match='original_owner_clock'):
        clocks.validate_admission(start,owner_start=80.,owner_unix_start=999.,owner_deadline=99.99,unix_deadline=1011.)
    with pytest.raises(clocks.ClockContractError,match='original_wall_clock'):
        clocks.validate_admission(start,owner_start=80.,owner_unix_start=999.,owner_deadline=101.,unix_deadline=1009.99)
    assert clocks.require_inherited_deadlines(91.,1001.,sample(qpc=1e12,owner=90.,wall=1000.))
    with pytest.raises(clocks.ClockContractError,match='original_owner_deadline'):
        clocks.require_inherited_deadlines(90.,1001.,sample(qpc=0.,owner=90.,wall=1000.))
    with pytest.raises(clocks.ClockContractError,match='original_wall_deadline'):
        clocks.require_inherited_deadlines(91.,1000.,sample(qpc=0.,owner=90.,wall=1000.))


@pytest.mark.parametrize('field',['request_qpc','owner_monotonic','unix'])
def test_pending_start_cannot_be_in_the_future_in_any_domain(field):
    start,_,_,_=fixture();observed=sample(100.1,90.1,1000.1)
    assert clocks.validate_pending(start,observed)
    observed[field]-=1.
    with pytest.raises(clocks.ClockContractError):clocks.validate_pending(start,observed)


@pytest.mark.parametrize('field',['request_qpc_start','owner_monotonic_start','unix_start'])
def test_sequential_acknowledgements_each_keep_own_domain(field):
    start,response,_,ack=fixture();clocks.validate_ack(start,response,ack)
    later=dict(start,**clocks.start_fields(sample(101.,91.,1001.)))
    assert clocks.validate_sequence(ack,later)
    key={'request_qpc_start':'parent_observed_qpc','owner_monotonic_start':'parent_observed_owner_monotonic','unix_start':'parent_observed_unix'}[field]
    later[field]=ack[key]-.001
    if field=='request_qpc_start':later['request_qpc_deadline']=later[field]+10.;later['internal_qpc_deadline']=later[field]+9.
    if field=='unix_start':later['unix_deadline']=later[field]+10.
    with pytest.raises(clocks.ClockContractError,match='sequential'):clocks.validate_sequence(ack,later)


def test_saved_failure_is_reproduced_synthetically_without_retroactive_acceptance():
    paths=[FAILED/'request_002.started.json',FAILED/'request_002.completed.json']
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    oldstart,oldresponse=[json.loads(p.read_text()) for p in paths]
    # The historical parent reading was not saved. These are expressly synthetic
    # domain-separated observations, not reconstructed historical receipts.
    q0=oldstart['monotonic_start'];q1=oldresponse['finished_monotonic'];w0=oldstart['unix_start'];w1=oldresponse['finished_unix']
    start,response,observed,_=fixture()
    start.update(clocks.start_fields(sample(q0,q0-100.,w0)))
    response.update(clocks.completion_fields(start,sample(q1,q1-100.,w1)))
    observed=sample(q1+.005,q1-100.+.005,w1+.005)
    assert q1>observed['owner_monotonic']  # Original mixed-domain inequality rejects.
    assert received(start,response,observed)['status']=='passed'
    observed['request_qpc']=start['request_qpc_deadline']
    with pytest.raises(clocks.ClockContractError,match='parent_request_qpc_deadline'):received(start,response,observed)
    assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert not (FAILED/'request_002.parent_accepted.json').exists()


@pytest.mark.parametrize('expired',[False,True])
def test_actual_fitter_main_inherits_owner_monotonic_not_qpc(monkeypatch,expired):
    monkeypatch.setattr(sys,'argv',['training','--mode','qualification','--manifest','unused','--plan','unused','--fixtures','unused','--out','unused'])
    monkeypatch.setenv('SELFPLAY_MONOTONIC_DEADLINE','100');monkeypatch.setenv('SELFPLAY_UNIX_DEADLINE','200')
    monkeypatch.setattr(clocks,'sample',lambda:sample(qpc=0. if expired else 1e12,owner=100. if expired else 99.,wall=199.))
    class StopBeforeAnyModel(Exception):pass
    monkeypatch.setattr(training,'startup',lambda:(_ for _ in ()).throw(StopBeforeAnyModel()))
    with pytest.raises(clocks.ClockContractError if expired else StopBeforeAnyModel):training.main()


def test_fitter_placement_records_separate_clocks_without_mutating_test_process(monkeypatch):
    import psutil
    fake=SimpleNamespace(pid=123,cpu_affinity=lambda x=None:[0],nice=lambda x=None:psutil.NORMAL_PRIORITY_CLASS,create_time=lambda:999.)
    monkeypatch.setattr(psutil,'Process',lambda:fake);monkeypatch.setattr(clocks,'sample',lambda:sample())
    saved=[];monkeypatch.setattr(training,'save_json',lambda path,value:saved.append(value))
    # This test module imports no Torch; other aggregate tests may have done so.
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    value=training.place_fitter(Path('unused'))
    assert value['owner_monotonic_time']==90. and value['request_qpc_time']==100.
    assert value['clock_contract']==clocks.runtime_contract() and 'monotonic_time' not in value


def test_real_six_response_cross_process_handshake_uses_production_observer(monkeypatch):
    from tools.splendorust_selfplay_pilot_budget_20261102 import qualification_control as control
    base=ROOT/'outputs/splendorust_selfplay_pilot_launch_preparation_20261102_01/prelaunch';base.mkdir(parents=True,exist_ok=True)
    script=r'''
import json,os,sys,time
from pathlib import Path
from tools.splendorust_selfplay_pilot_budget_20261102 import training as t,clock_contract as clocks
assert 'torch' not in sys.modules
folder=Path(sys.argv[1]);print('READY',flush=True);assert sys.stdin.readline().strip()=='GO'
owner=json.loads((folder/'owner.json').read_text());checkpoint=t.artifact(folder/'checkpoint_manifest.json')
ids=['face_T1_05','private_reserves','gold_returns']
for i in range(6):
 clocks.require_inherited_deadlines(owner['monotonic_deadline'],owner['unix_deadline'])
 start=dict(schema='selfplay_learned_request_start_v2',request_index=i,case_id=ids[i%3],mode='eager' if i<3 else 'compiled',candidate_manifest=checkpoint,**clocks.start_fields())
 item=t._atomic_json(folder/'learned_entrypoint'/f'request_{i:03d}.started.json',start)
 response=dict(schema='selfplay_learned_response_v2',request_index=i,case_id=start['case_id'],mode=start['mode'],candidate_manifest=checkpoint,start=item,**clocks.completion_fields(start))
 t._atomic_json(folder/'learned_entrypoint'/f'request_{i:03d}.completed.json',response)
 print('COMPLETE '+str(i),flush=True)
 assert sys.stdin.readline().strip()=='ACK'
 ack=json.loads((folder/'learned_entrypoint'/f'request_{i:03d}.parent_accepted.json').read_text())
 clocks.validate_ack(start,response,ack)
assert 'torch' not in sys.modules
print(json.dumps(dict(requests=6,models=0,torch_loaded=False)),flush=True)
'''
    with tempfile.TemporaryDirectory(prefix='clock_process_',dir=base) as name:
        monkeypatch.setattr(control.c,'OUT',Path(name))
        folder=Path(name)/'fit';(folder/'learned_entrypoint').mkdir(parents=True)
        training.save_json(folder/'checkpoint_manifest.json',{'synthetic_clock_fixture':True})
        process=subprocess.Popen([sys.executable,'-B','-u','-c',script,str(folder)],cwd=ROOT,
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT)},stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,text=True,creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        lines=queue.Queue()
        def consume():
            for line in process.stdout:lines.put(line)
            lines.put('')
        reader=threading.Thread(target=consume,daemon=True);reader.start()
        def line():return lines.get(timeout=10).strip()
        try:
            import psutil
            assert line()=='READY'
            origin=clocks.sample();envelope=SimpleNamespace(deadline=origin['owner_monotonic']+30.,wall_deadline=origin['unix']+30.)
            identity=dict(pid=process.pid,created=psutil.Process(process.pid).create_time())
            training.save_json(folder/'owner.json',dict(identity=identity,clock_contract=clocks.runtime_contract(),
                owner_monotonic_start=origin['owner_monotonic'],request_qpc_start=origin['request_qpc'],unix_time=origin['unix'],
                monotonic_deadline=envelope.deadline,unix_deadline=envelope.wall_deadline))
            process.stdin.write('GO\n');process.stdin.flush();observed=[]
            freeze=dict(clock_contract=clocks.runtime_contract(),learned_entrypoint=dict(case_ids=['face_T1_05','private_reserves','gold_returns']))
            for i in range(6):
                assert line()=='COMPLETE '+str(i)
                control.observe_learned_requests(folder,identity,envelope,freeze,observed)
                assert len(observed)==i+1 and observed[-1]['deadline_seconds']==10.
                process.stdin.write('ACK\n');process.stdin.flush()
            result=json.loads(line())
            assert result==dict(requests=6,models=0,torch_loaded=False)
            process.wait(timeout=10);assert process.returncode==0,process.stderr.read()
            assert all(x['parent_observed_qpc']>=json.loads((folder/'learned_entrypoint'/f'request_{i:03d}.completed.json').read_text())['finished_qpc'] for i,x in enumerate(observed))
            assert not list((folder/'learned_entrypoint').glob('*.parent_rejected.json'))
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
            reader.join(timeout=2)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()


def test_training_parameters_and_graphs_are_unchanged_by_clock_correction():
    from tools.splendorust_selfplay_pilot_20261102 import training as prior
    assert training.RECIPE==prior.RECIPE and training.qualification_plan()==prior.qualification_plan() and training.pilot_plan()==prior.pilot_plan()
    for name in ('tensor_forward','compile_inference','train_step','make_optimizer','expected_state_shapes','load_parent','load_candidate'):
        assert inspect.getsource(getattr(training,name))==inspect.getsource(getattr(prior,name))
