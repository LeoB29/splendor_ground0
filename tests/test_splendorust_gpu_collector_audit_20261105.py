"""Model-free streaming-accounting and scientific-summary counterexamples."""
from copy import deepcopy
import json
import pytest

from tools.splendorust_gpu_collector_20261105 import audit as a


def request(client,sequence,slot,submitted,ticket):
    return dict(client_id=client,request_id=f'{client}:{sequence}',slot=slot,
                submitted_perf=submitted,deadline_perf=70.,enqueue_sequence=ticket)


def gpu_records():
    one=request('a',1,0,10.,1);two=request('b',1,0,10.,2);three=request('c',1,1,10.,3)
    four=request('a',2,0,10.2,4);five=request('c',2,1,10.2,5)
    records=[]
    for index,ready,selected,remaining in ((1,[one,two,three],[one,three],[two]),
                                          (2,[two,four,five],[two,five],[four]),(3,[four],[four],[])):
        observed=10.+index*.2;dispatch=observed+.01;end=dispatch+.05
        records.append(dict(kind='scheduler_dispatch',scheduler='row_fifo32',scheduler_dispatch_index=index,
                            observed_perf=observed,ready=ready,selected_request_ids=[x['request_id'] for x in selected],
                            deferred_request_ids=[x['request_id'] for x in remaining]))
        occupied=sorted(x['slot'] for x in selected)
        rows=[dict({k:v for k,v in x.items() if k!='enqueue_sequence'},route='gpu',batch_id=index,native_batch=1,
                   physical_batch=8,occupied_slots=occupied,dispatch_perf=dispatch,complete_readback_perf=end,
                   queue_seconds=dispatch-x['submitted_perf'],forward_roundtrip_seconds=end-dispatch,
                   input_sha256='a'*64,output_sha256='b'*64) for x in selected]
        records.append(dict(kind='inference_batch',route='gpu',batch_id=index,physical_batch=8,requests=rows))
    # Second callbacks start only after their first readbacks.
    for event in records:
        source=event.get('ready',event.get('requests',[]))
        for row in source:
            if row['request_id'] in ('a:2','c:2'):
                row['submitted_perf']=10.3
                if 'dispatch_perf' in row:row['queue_seconds']=row['dispatch_perf']-row['submitted_perf']
    return records


def write_fixture(tmp_path,monkeypatch,records):
    monkeypatch.setattr(a.c,'ROOT',tmp_path);monkeypatch.setattr(a.c,'OUT',tmp_path/'study')
    path=tmp_path/'inference.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in records),encoding='utf-8')
    return path


def test_streamed_fifo_collisions_clocks_and_complete_unique_attribution(tmp_path,monkeypatch):
    path=write_fixture(tmp_path,monkeypatch,gpu_records());original=path.read_bytes()
    with a.JournalIndex(path,'gpu','row_fifo32') as index:
        assert len(index)==5 and index.batch_count==3 and index.enqueued==5
        for client,count in (('a',2),('b',1),('c',2)):
            rows=index.decision(client,10.,owner='game:'+client)
            assert len(rows)==count
        metrics=index.metrics('game:')
        assert metrics['completed_callback_prefixes']==5 and metrics['mean_occupied_rows']==5/3
        assert index.complete_attribution()['all_callbacks_attributed_once']
        with pytest.raises(ValueError,match='exactly one'):index.decision('a',10.,owner='retry')
    assert path.read_bytes()==original


@pytest.mark.parametrize('mutation',('fifo','duplicate','deadline','ticket','gap','orphan'))
def test_stream_rejects_fifo_accounting_and_original_clock_tampering(tmp_path,monkeypatch,mutation):
    rows=gpu_records()
    if mutation=='fifo':rows[0]['selected_request_ids'].reverse()
    if mutation=='duplicate':rows[-1]['batch_id']=1
    if mutation=='deadline':rows[1]['requests'][0]['complete_readback_perf']=70.
    if mutation=='ticket':rows[2]['ready'][-1]['enqueue_sequence']=99
    if mutation=='gap':rows=rows[:-1]
    path=write_fixture(tmp_path,monkeypatch,rows)
    with pytest.raises(ValueError):
        with a.JournalIndex(path,'gpu','row_fifo32') as index:index.complete_attribution()
    receipts=list((tmp_path/'study_audit_scratch_receipts').glob('*.json'))
    assert len(receipts)==1
    saved=json.loads(receipts[0].read_text());assert saved['scratch_index_was_not_source'] and saved['models']==saved['games']==0


def test_original_cpu_batches_may_be_journaled_out_of_batch_id_order(tmp_path,monkeypatch):
    rows=[]
    for index in (2,1):
        item=dict(client_id=f'c{index}',request_id=f'c{index}:1',slot=index-1,submitted_perf=10.,
            deadline_perf=70.,route='cpu',batch_id=index,native_batch=1,physical_batch=1,occupied_slots=[index-1],
            dispatch_perf=10.1,complete_readback_perf=10.2,queue_seconds=10.1-10.,forward_roundtrip_seconds=10.2-10.1)
        rows.append(dict(kind='inference_batch',route='cpu',batch_id=index,physical_batch=1,requests=[item]))
    path=write_fixture(tmp_path,monkeypatch,rows)
    with a.JournalIndex(path,'cpu','original8') as index:
        index.mark(['c1:1','c2:1'],'game:cpu');assert index.complete_attribution()['physical_batches']==2


def test_authenticated_incomplete_final_dispatch_is_preserved_without_a_rate(tmp_path,monkeypatch):
    rows=gpu_records();rows=rows[:-1];rows[-1]['observed_perf']=75.
    path=write_fixture(tmp_path,monkeypatch,rows)
    with a.JournalIndex(path,'gpu','row_fifo32',incomplete=True) as index:
        assert index.request_count==4 and index.dispatches==3 and index.batch_count==2
    with pytest.raises(ValueError):
        with a.JournalIndex(path,'gpu','row_fifo32'):pass


def test_independent_visits_and_actor_values_match_all_saved_actual_capture_fixtures():
    from tools.splendorust_gpu_collector_20261105 import data
    folder=a.c.ROOT/'outputs/splendorust_selfplay_pilot_20261102_02/capture_fixtures/capture_on'
    paths=sorted(folder.glob('*.json'));assert len(paths)==16
    eligible=0
    for path in paths:
        row=a.read(path);actual=a.independent_example(row)
        assert actual==data.decode_row(row)
        if actual['eligible']:
            eligible+=1;row=dict(row,game_id='engineering_saved_fixture',spec={'game_index':0},route='cpu')
            for winner in (None,0,1):
                example=a.independent_example(row,winner=winner,terminal=True);actor=example['actor']
                expected=0. if winner is None else 1. if winner==actor else -1.
                assert example['value']==[expected,-expected if expected else 0.]
            bad=deepcopy(row);bad['training_capture']['native_capture']['visits67'][0]+=1
            with pytest.raises(ValueError):a.independent_example(bad)
    assert eligible>6


def test_throughput_criterion_uses_both_verified_blocks_not_pooled_collection_only():
    proofs=[];costs={}
    for cell,route,block,seconds,verified in [('b1_cpu','cpu',1,100.,130.),('b1_gpu','gpu',1,90.,110.),
                                           ('b2_gpu','gpu',2,85.,120.),('b2_cpu','cpu',2,100.,124.)]:
        proofs.append(dict(cell_id=cell,route=route,block=block,status='passed',full_cohort_verified=True,games=16,
            seed_decimals=[str(block)],trajectory_hashes=['same'],wall_seconds=seconds,plies=800,eligible_roots=750,replay_seconds=10.))
        costs[cell]=dict(verified_seconds=verified,audit_seconds=5.)
    summary=a.comparison_summary(proofs,costs=costs)
    assert summary['eligible'] and summary['pooled_gpu_over_cpu_games_per_hour']>1.1
    assert not summary['useful_improvement'] and summary['pairs'][0]['passed_useful_threshold']
    assert not summary['pairs'][1]['passed_useful_threshold']
    assert not a.comparison_summary(proofs[:3])['eligible']


def test_gate_requires_complete_actual_two_route_capture_and_loader_evidence():
    route=dict(complete_decisions=116,entrypoint={'proof':{'actual_decisions':67}},parity_decisions=32,
        parity=[{'exact_full_result_and_after':True} for _ in range(32)],loader_decisions=16,official_continuations=1,
        loader_row_checks={'status':'actual_loader_counterexamples_passed'},
        full_loader_checks={'status':'actual_complete_fixture_loader_passed','eligible_roots':1},
        journal={'all_callbacks_attributed_once':True},runtime_closure={'final_weights_verified':True})
    proof=dict(status='passed',passed=True,routes={r:dict(deepcopy(route),route=r) for r in ('cpu','gpu')},
        complete_decisions=232,ordinary_games=0,training_updates=0,
        compilation_fingerprint='2a720c18d129310700dd850eb9b6ae8c99b4a6345bdbf437bc924fef6feb9dbb')
    assert a.qualification_gate(proof)['passed']
    for key,value in [('loader_decisions',15),('official_continuations',0),('complete_decisions',115)]:
        bad=deepcopy(proof);bad['routes']['gpu'][key]=value;assert not a.qualification_gate(bad)['passed']
    bad=deepcopy(proof);bad['compilation_fingerprint']='other';assert not a.qualification_gate(bad)['passed']
    assert not a.qualification_gate({'status':'failed','passed':False})['passed']


def test_saved_real_capture_through_full_decision_and_callback_auditor(tmp_path,monkeypatch):
    """Historical fixture + synthetic clocks test plumbing; never new game data."""
    from tools.splendorust_gpu_throughput_20261105 import scientific
    from tools.splendorust_gpu_collector_20261105 import data
    source=a.read(a.c.ROOT/'outputs/splendorust_selfplay_pilot_20261102_02/capture_fixtures/capture_on/008.json')
    result=deepcopy(source['clock']['result']);result['wrapper_id']=scientific.wrapper_id(512)
    scientific.validate_result(result,source['observer'],'search',expected_quota=512)
    callbacks=[x for phase in result['diagnostics']['phases'] for x in phase['inference']]
    events=[];receipts=[]
    for i,callback in enumerate(callbacks):
        start=10.1+i*.0001;end=start+.00005
        row=dict(client_id='worker_0',request_id=f'worker_0:{i+1}',slot=0,submitted_perf=start,deadline_perf=70.,
            route='cpu',batch_id=i+1,native_batch=1,physical_batch=1,occupied_slots=[0],dispatch_perf=start,
            complete_readback_perf=end,queue_seconds=0.,forward_roundtrip_seconds=end-start,
            input_sha256=callback['input_sha256'],output_sha256=callback['output_sha256'])
        events.append(dict(kind='inference_batch',route='cpu',batch_id=i+1,physical_batch=1,requests=[row]))
        receipts.append(dict(row,client_received_perf=end+.000001))
    path=write_fixture(tmp_path,monkeypatch,events);folder=tmp_path/'source';spec=dict(game_id='fixture',game_index=0,purpose='engineering_only')
    clock=dict(started_perf_counter=10.,ended_perf_counter=11.,elapsed_seconds=1.,internal_seconds=60.,outer_seconds=75.,
        no_retry=True,execution_policy_id=a.c.EXECUTION_POLICY_ID,status='ok',completed=True)
    row=dict(game_id='fixture',spec=spec,route='cpu',ply=0,actor=source['observer']['observation']['current_player'],
        client_id='worker_0',slot=0,before=source['before'],observer=source['observer'],result=result,
        action=result['action'],trace=result['diagnostics'],clock=clock,after=source['after'],training_capture=source['training_capture'])
    request=dict(player='search',observer=source['observer'])
    records={'moves':row,'clocks':clock,'requests':request,'captures':row['training_capture'],'service_calls':receipts,
        'raw_results':dict(observer=row['observer'],result=result,action=result['action'],diagnostics=result['diagnostics']),
        'attempts':dict(game_id='fixture',ply=0,client_id='worker_0',slot=0,started_perf_counter=10.,
            request_sha256=a.digest(request),before_sha256=a.digest(source['before']),no_retry=True)}
    for key,value in records.items():
        target=folder/key/'000000.json';target.parent.mkdir(parents=True);target.write_text(json.dumps(value),encoding='utf-8')
    with a.JournalIndex(path,'cpu','original8') as index:
        actual,example,count=a.audit_decision(folder,spec,0,index,before=source['before'],owner='game:fixture',client='worker_0',slot=0)
        assert actual==row and example==data.decode_row(row) and count==len(callbacks)
        assert index.complete_attribution()['requests']==len(callbacks)


def test_actual_verified_cost_binds_durable_audit_and_full_worker_wall(tmp_path,monkeypatch):
    monkeypatch.setattr(a.c,'ROOT',tmp_path);monkeypatch.setattr(a.c,'OUT',tmp_path/'study')
    container=a.c.OUT/'cells/b1_cpu';container.mkdir(parents=True)
    execution=dict(monotonic_origin=10.,monotonic_end=110.,unix_origin=1000.,unix_end=1100.)
    a.c.save(container/'execution.json',execution);a.c.save(container/'saved_audit.json',{'passed':True})
    a.c.save(a.c.OUT/'claim.json',dict(monotonic_origin=0.,unix_origin=990.))
    cost=dict(schema='collector_verified_cell_cost_v1',cell_id='b1_cpu',route='cpu',execution=a.c.artifact(container/'execution.json'),
        saved_audit=a.c.artifact(container/'saved_audit.json'),auditor=a.c.artifact(a.__file__),
        worker_started_monotonic=10.,worker_ended_monotonic=110.,worker_started_unix=1000.,worker_ended_unix=1100.,
        audit_started_monotonic=111.,audit_ended_monotonic=121.,verified_ended_monotonic=122.,
        audit_started_unix=1101.,audit_ended_unix=1111.,verified_ended_unix=1112.,
        worker_seconds=100.,saved_audit_seconds=10.,verified_seconds=112.)
    target=container/'cell_cost.json';a.c.save(target,cost)
    assert a.audit_cell_cost({'cell_id':'b1_cpu','route':'cpu'})['verified_seconds']==112.
    cost['verified_seconds']=90.;target.write_text(json.dumps(cost),encoding='utf-8')
    with pytest.raises(ValueError):a.audit_cell_cost({'cell_id':'b1_cpu','route':'cpu'})


def test_final_audit_cannot_restart_either_original_study_clock(monkeypatch):
    claim=dict(monotonic_origin=100.,unix_origin=1000.)
    monkeypatch.setattr(a.time,'monotonic',lambda:100.+a.c.TOTAL_SECONDS-1.)
    monkeypatch.setattr(a.time,'time',lambda:1000.+a.c.TOTAL_SECONDS-1.)
    a.guard_audit_clock(claim)
    monkeypatch.setattr(a.time,'monotonic',lambda:100.+a.c.TOTAL_SECONDS)
    with pytest.raises(ValueError,match='original overall'):a.guard_audit_clock(claim)
    monkeypatch.setattr(a.time,'monotonic',lambda:101.)
    monkeypatch.setattr(a.time,'time',lambda:1000.+a.c.TOTAL_SECONDS)
    with pytest.raises(ValueError,match='original overall'):a.guard_audit_clock(claim)

