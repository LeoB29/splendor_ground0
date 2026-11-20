"""Model-free falsification of the separately frozen saved-cost observer audit."""
import copy
import importlib
import sys
import time

import pytest

from tools.splendorust_gpu_costs_20261105 import audit as a,common as c
from tools.splendorust_gpu_costs_20261105.observer import LABELS,COLUMNS,MAX_EVENTS


def overhead(walls=(100.,103.,104.,100.)):
    identities=[('oh1_control',1,False),('oh1_observed',1,True),('oh2_observed',2,True),('oh2_control',2,False)]
    return [dict(cell_id=name,block=block,observed=observed,phase='overhead',configuration='gpu16',route='gpu',
                 concurrency=16,scheduler='row_fifo32',status='passed',eligible=True,wall_seconds=wall,
                 case_rows=[dict(completed=True,baseline_exact=True) for _ in range(32)])
            for (name,block,observed),wall in zip(identities,walls)]


def test_overhead_uses_pooled_original_wall_and_each_pair():
    good=a.overhead_gate(overhead());assert good['passed'] and good['pooled_observed_over_control_wall_ratio']==1.035
    assert not a.overhead_gate(overhead((100.,106.,106.,100.)))['passed']
    # Pooled cost passes, one reversed-block pair does not.
    bad=a.overhead_gate(overhead((100.,111.,95.,100.)))
    assert bad['pooled_observed_over_control_wall_ratio']==1.03 and not bad['passed']


def test_deadline_or_inexact_cell_cannot_earn_cost_eligibility():
    cells=overhead();cells[1].update(status='deadline_failed',eligible=False,wall_seconds=None)
    cells[1]['case_rows'][0]['completed']=False
    gate=a.overhead_gate(cells)
    assert not gate['passed'] and not gate['complete_four_cells'] and gate['pooled_observed_over_control_wall_ratio'] is None
    cells=overhead();cells[2]['case_rows'][0]['baseline_exact']=False
    assert not a.overhead_gate(cells)['passed']


@pytest.mark.parametrize('change',[lambda x:x.reverse(),lambda x:x[0].update(concurrency=8),lambda x:x[1].update(observed=False)])
def test_wrong_overhead_intervention_or_order_rejected(change):
    cells=overhead();change(cells)
    with pytest.raises(ValueError):a.overhead_gate(cells)


def payload(events,enabled=True):
    return dict(schema='gpu_cost_wall_spans_v1',enabled=enabled,columns=list(COLUMNS),labels=list(LABELS),
        maximum_events=MAX_EVENTS,session_id='a'*32,clock='time.perf_counter wall time',events=events,markers=[])


def event(i,parent,label,left,right,thread=1,owner='batch:1'):
    return [i,parent,LABELS.index(label),thread,owner,left,right,False]


def raw_fixture():
    rows=[event(1,None,'raw_forward_total',1.,7.)]
    for index,label in enumerate(('raw_input_validation','tensor_construction','gpu_roundtrip_synchronized','output_conversion','raw_output_validation'),2):
        rows.append(event(index,1,label,float(index-1)+.1,float(index)-.1))
    # Recorder saves completion order, not allocation order.
    return payload(rows[1:]+rows[:1])


def test_completed_order_not_allocation_order_and_explicit_nested_components():
    rows,_=a.validate_observer(raw_fixture(),True)
    result=a.raw_components(rows)
    assert len(result)==1 and result[0]['durations']['gpu_roundtrip_synchronized']==pytest.approx(.8)


@pytest.mark.parametrize('mutation',[
    lambda x:x['events'][0].__setitem__(1,999),
    lambda x:x['events'][0].__setitem__(5,.9),
    lambda x:x['events'][0].__setitem__(6,float('nan')),
    lambda x:x['events'][0].__setitem__(7,True),
    lambda x:x['events'][0].__setitem__(3,2),
    lambda x:x['events'].pop(0),
])
def test_missing_failed_nonfinite_or_orphan_observer_event_rejected(mutation):
    value=raw_fixture();mutation(value)
    with pytest.raises(ValueError):a.validate_observer(value,True)


def test_control_cannot_conceal_enabled_observer_work():
    value=raw_fixture();value['enabled']=False
    with pytest.raises(ValueError):a.validate_observer(value,False)
    rows,_=a.validate_observer(payload([],False),False);assert rows==[]


def test_raw_component_relabel_or_missing_call_rejected():
    value=raw_fixture();value['events'][0][2]=LABELS.index('pipe_readline')
    rows,_=a.validate_observer(value,True)
    with pytest.raises(ValueError):a.raw_components(rows)


def test_concurrent_spans_union_not_false_elapsed_sum():
    assert a.interval_union([(0.,3.),(1.,4.),(5.,6.)])==5.
    rows=[dict(kind_name='callback_total',start=0.,end=3.),dict(kind_name='callback_total',start=1.,end=4.)]
    result=a.summarize_spans(rows)['callback_total']
    assert result['summed_span_seconds']==6. and result['union_wall_seconds']==4.


def test_message_gaps_pair_only_adjacent_same_owner_and_thread():
    rows=[]
    for thread,owner,shift in ((1,'worker_0',0.),(2,'worker_1',.2)):
        for label,left,right in [('pipe_readline',0.,1.),('callback_total',1.2,3.),('pipe_write',3.3,3.5),('pipe_flush',3.5,3.6)]:
            rows.append(dict(thread_id=thread,owner=owner,kind_name=label,start=left+shift,end=right+shift))
    result=a.message_gaps(rows)
    assert result['callbacks']==result['input_gap_count']==result['output_gap_count']==2
    assert result['input_gap_sum_seconds']==pytest.approx(.4) and result['output_gap_sum_seconds']==pytest.approx(.6)


def test_private_checker_context_does_not_mutate_public_historical_module():
    public=importlib.import_module('tools.splendorust_gpu_batchfill_20261105.audit');original=public.c
    prior_sha=c.sha(a.LIBRARY);private=a.load_reused()
    assert private is not public and private.c is c and public.c is original
    assert c.sha(a.LIBRARY)==prior_sha==a.LIBRARY_SHA
    assert a.source_context_proof()['models']==a.source_context_proof()['games']==0


def test_exact_isolated_batch_rows_bound_to_prior_accepted_full_results():
    plan=c.read(c.ROOT/'outputs/splendorust_gpu_batchfill_20261105_01/plan.json')
    batches=a.expected_isolated_batches(plan)
    assert len(batches)==4 and all(len(x['input_tokens'])==11904 and len(x['expected_outputs'])==8 for x in batches)
    assert [i for x in batches for i in x['case_indexes']]==list(range(32))
    for batch in batches:
        assert batch['input_sha256']==a.digest(batch['input_tokens'])
        for slot,index in enumerate(batch['case_indexes']):
            assert plan['cases'][index]['slot']==slot and len(batch['expected_outputs'][slot])==83


def test_isolated_live_requires_all_eight_input_rows_for_matched_claim():
    plan=c.read(c.ROOT/'outputs/splendorust_gpu_batchfill_20261105_01/plan.json')
    hashes=[dict(slot=i,sha256=a.digest(plan['cases'][i]['expected_root_tokens'])) for i in range(8)]
    raw=[dict(batch_id=1,request_input_sha256=hashes,durations={'gpu_roundtrip_synchronized':.02}),
         dict(batch_id=2,request_input_sha256=hashes[:7],durations={'gpu_roundtrip_synchronized':.03})]
    cells=[dict(cell_id='diag1',phase='diagnostic',observer_coverage=dict(raw_batches=raw))]
    iso=[dict(cell_id='iso1_'+side,calls=[dict(phase='measured',batch_index=i,durations={'gpu_roundtrip_synchronized':.01}) for i in range(4)]) for side in ('pre','post')]
    result=a.isolate_live_comparison(plan,cells,iso)[0]
    assert result['exact_full_physical_input_match_count']==1 and result['exact_full_physical_input_matches'][0]['live_batch_id']==1
    assert result['all_live_gpu_roundtrip_seconds']['count']==2


def test_real_observer_service_and_journal_schema_integration_without_models(tmp_path):
    from tools.splendorust_gpu_costs_20261105 import backend as b
    from tools.splendorust_gpu_costs_20261105.observer import Recorder
    recorder=Recorder(True)
    def forward(tokens,batch):
        with recorder.span('raw_forward_total'):
            for name in ('raw_input_validation','tensor_construction','gpu_roundtrip_synchronized','output_conversion','raw_output_validation'):
                with recorder.span(name):pass
            return [[tokens[row*b.WIDTH]]*83 for row in range(batch)]
    runtime=b.Runtime({},'gpu',tmp_path/'runtime',{},forward,lambda:None,{},scheduler='row_fifo32',observer=recorder)
    try:
        runtime.service.register('warmup',0)
        runtime.service.infer('warmup',[1.]*b.WIDTH,deadline=time.perf_counter()+5.)
        runtime.service.unregister('warmup')
        runtime.service.register('native0',0)
        adapter=b.ObservedProxySearchOpponent.__new__(b.ObservedProxySearchOpponent)
        adapter.runtime=runtime;adapter.client_id='native0';adapter.active_deadline=time.perf_counter()+5.
        adapter.capture=True;adapter.callbacks=[];adapter.last_service_calls=[];adapter.boundary=lambda:None
        adapter._infer([2.]*b.WIDTH,1)
        runtime.service.unregister('native0')
    finally:runtime.close()
    # Round-trip JSON preserves the actual exported tuple representation.
    import json
    value=json.loads(json.dumps(recorder.snapshot()));rows,_=a.validate_observer(value,True)
    events,requests,_=a.load_reused().read_journal(tmp_path/'runtime','gpu')
    result=a.observer_service_coverage(rows,events,requests)
    assert result['dispatches']==2 and result['journal_spans']==4 and result['native_callbacks']==1
    broken=[row for row in rows if row['kind_name']!='tensor_construction']
    with pytest.raises(ValueError):a.observer_service_coverage(broken,events,requests)


def test_mandatory_recording_artifact_coverage_not_just_total_count():
    names=['attempt.json','raw_result.json','clock.json','result.json','root.json','after.json','service_calls.json','outcome.json']
    owners=['p000/'+name for name in names]+['group:0/start.json','group:0/summary.json']
    rows=[dict(kind_name='case_recording',owner=owner) for owner in owners]
    assert a.recording_coverage(rows,[dict(case_id='p000')],[[0]])['spans']==10
    rows[-1]['owner']='group:0/start.json'
    with pytest.raises(ValueError):a.recording_coverage(rows,[dict(case_id='p000')],[[0]])


def test_actual_isolated_record_writer_roundtrips_saved_auditor_without_models(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from tools.splendorust_gpu_costs_20261105 import cell,design
    from tools.splendorust_gpu_costs_20261105.observer import Recorder
    plan=c.read(c.ROOT/'outputs/splendorust_gpu_batchfill_20261105_01/plan.json')
    plan['isolated']=design.isolated();spec=plan['isolated'][0]
    batches=a.expected_isolated_batches(plan);by_input={x['input_sha256']:x for x in batches};recorder=Recorder(True)
    def physical_forward(tokens,*,deadline):
        assert time.perf_counter()<deadline
        with recorder.span('raw_forward_total'):
            for label in ('raw_input_validation','tensor_construction','gpu_roundtrip_synchronized','output_conversion','raw_output_validation'):
                with recorder.span(label):pass
            return by_input[a.digest(tokens)]['expected_outputs']
    runtime=SimpleNamespace(route='gpu',identity=dict(scheduler='row_fifo32'),observer=recorder,adapters={},
        service=SimpleNamespace(clients={},pending={},physical_forward=physical_forward),abort=lambda:None,
        operation=lambda *args,**kwargs:nullcontext())
    reused=a.load_reused();monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(cell,'validate_cases',lambda plan:plan['cases'])
    monkeypatch.setattr(reused,'audit_stage_execution',lambda *args,**kwargs:(dict(status='completed',worker_exit_code=0),[]))
    monkeypatch.setattr(reused,'audit_runtime',lambda *args,**kwargs:dict(service_failure=None))
    folder=tmp_path/'worker';folder.mkdir()
    result=cell.run_isolated(plan,spec,runtime,folder/'isolated')
    c.save(folder/'runtime_sources.json',dict(sources=plan['source_guard']))
    c.save(folder/'runtime/runtime_identity.json',dict(timing_observer_enabled=True,timing_observer_schema='gpu_cost_wall_spans_v1'))
    export=recorder.export(folder/'runtime/observer_timing.json',scope=dict(mode='isolated',cell_id=spec['cell_id'],observed=True,
        phase='isolated',exported_after_runtime_close=True,outside_measured_intervals=True))
    c.save(folder/'cleanup.json',dict(success=True))
    c.save(folder/'worker_result.json',dict(mode='isolated',cell_id=spec['cell_id'],exit_code=0,result=result,
        observer_exports=[export],reference_weights_unchanged=True,no_retry=True))
    proof=a.audit_isolated(plan,spec,folder)
    assert proof['status']=='saved_isolated_batches_verified' and len(proof['calls'])==148
    assert proof['phases']['measured']['calls']==128 and proof['phases']['warmup']['calls']==20
    path=folder/'isolated/measured/0000/result.json';bad=c.read(path);bad['outputs'][0][0]+=1.
    path.write_text(c.canonical(bad),encoding='utf-8')
    with pytest.raises(ValueError,match='isolated call exactly'):a.audit_isolated(plan,spec,folder)
