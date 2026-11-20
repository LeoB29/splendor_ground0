"""Model-free counterexamples for saved scheduling, clocks and rate acceptance."""
from copy import deepcopy
import json

import pytest

from tools.splendorust_gpu_batchfill_20261105 import audit as a


def item(client,ticket,slot,submitted=1.,deadline=61.):
    return dict(client_id=client,request_id=client+':1',slot=slot,enqueue_sequence=ticket,
                submitted_perf=submitted,deadline_perf=deadline)


def dispatch(index,ready,selected,deferred,observed=2.):
    return dict(kind='scheduler_dispatch',scheduler='row_fifo32',scheduler_dispatch_index=index,
                observed_perf=observed,ready=ready,selected_request_ids=selected,deferred_request_ids=deferred)


def physical(schedule):
    selected=[r for r in schedule['ready'] if r['request_id'] in schedule['selected_request_ids']]
    rows=[]; occupied=sorted(r['slot'] for r in selected)
    for r in selected:
        start=schedule['observed_perf']+.01; end=start+.01
        row={k:r[k] for k in ('client_id','request_id','slot','submitted_perf','deadline_perf')}
        row.update(route='gpu',batch_id=schedule['scheduler_dispatch_index'],native_batch=1,physical_batch=8,
            occupied_slots=occupied,dispatch_perf=start,complete_readback_perf=end,
            queue_seconds=start-r['submitted_perf'],forward_roundtrip_seconds=end-start,
            input_sha256='input',output_sha256='output')
        rows.append(row)
    return dict(kind='inference_batch',route='gpu',physical_batch=8,batch_id=schedule['scheduler_dispatch_index'],requests=rows)


def fixture():
    aa,bb,cc,dd=item('a',1,0),item('b',2,0),item('c',3,1),item('d',4,0,submitted=2.5)
    schedules=[dispatch(1,[aa,bb,cc],['a:1','c:1'],['b:1']),
               dispatch(2,[bb,dd],['b:1'],['d:1'],3.),dispatch(3,[dd],['d:1'],[],4.)]
    events=[physical(s) for s in schedules]
    requests={r['request_id']:r for e in events for r in e['requests']}
    return events,requests,schedules


def test_valid_oldest_per_row_and_deferred_prefix():
    e,r,s=fixture(); proof=a.audit_scheduler(e,r,s,'row_fifo32')
    assert proof['completed']==4 and proof['exact_oldest_per_row']


def test_newcomer_cannot_overtake_deferred_same_row():
    e,r,s=fixture(); s[1]['ready'].reverse()
    with pytest.raises(ValueError,match='deferred'): a.audit_scheduler(e,r,s,'row_fifo32')


def test_scheduler_cannot_skip_oldest_row_request():
    e,r,s=fixture(); s[0]['selected_request_ids']=['b:1','c:1']; s[0]['deferred_request_ids']=['a:1']
    with pytest.raises(ValueError,match='earliest'): a.audit_scheduler(e,r,s,'row_fifo32')


def test_missing_enqueue_ticket_rejected():
    e,r,s=fixture(); s[2]['ready'][0]['enqueue_sequence']=8
    with pytest.raises(ValueError): a.audit_scheduler(e,r,s,'row_fifo32')


def test_duplicate_physical_row_rejected(tmp_path):
    e,r,s=fixture(); e[0]['requests'][1]['slot']=0
    (tmp_path/'inference.jsonl').write_text(json.dumps(e[0])+'\n')
    with pytest.raises(ValueError,match='unique occupied'): a.read_journal(tmp_path,'gpu')


def test_selected_request_cannot_disappear():
    e,r,s=fixture(); del r['d:1']
    with pytest.raises(ValueError,match='none dropped'): a.audit_scheduler(e,r,s,'row_fifo32')


def test_batch_response_must_match_selected_identity():
    e,r,s=fixture(); e[0]['requests'][0]['client_id']='wrong'
    with pytest.raises(ValueError,match='identity'): a.audit_scheduler(e,r,s,'row_fifo32')


def test_old_control_rejects_candidate_dispatch_records():
    e,r,s=fixture()
    with pytest.raises(ValueError,match='control'): a.audit_scheduler(e,r,s,'original8')


def test_original_control_has_no_fifo_reinterpretation():
    e,r,s=fixture()
    assert not a.audit_scheduler(e,r,[],'original8')['fifo_applicable']


def test_terminal_uncompleted_expiry_is_preserved():
    pending=item('a',1,0,submitted=1.,deadline=2.)
    schedule=dispatch(1,[pending],['a:1'],[],observed=2.1)
    proof=a.audit_scheduler([],{},[schedule],'row_fifo32',incomplete=True)
    assert proof['completed']==0 and proof['enqueued']==1
    with pytest.raises(ValueError): a.audit_scheduler([],{},[schedule],'row_fifo32')


def test_completed_batch_never_gets_expired_ready_exception():
    e,r,s=fixture(); s[0]['ready'][0]['deadline_perf']=1.5
    with pytest.raises(ValueError,match='deadline'): a.audit_scheduler(e,r,s,'row_fifo32',incomplete=True)


def test_journal_deadline_and_full_raw_timing(tmp_path):
    e,r,s=fixture()
    text=''.join(json.dumps(x)+'\n' for pair in zip(s,e) for x in pair)
    (tmp_path/'inference.jsonl').write_text(text)
    ee,rr,ss=a.read_journal(tmp_path,'gpu')
    assert (ee,rr,ss)==(e,r,s)
    e[0]['requests'][0]['deadline_perf']=e[0]['requests'][0]['complete_readback_perf']
    (tmp_path/'inference.jsonl').write_text(json.dumps(e[0])+'\n')
    with pytest.raises(ValueError,match='clocks'): a.read_journal(tmp_path,'gpu')


def test_client_callback_gap_rejected():
    e,r,s=fixture(); one=deepcopy(r['a:1']); one['request_id']='a:2'
    with pytest.raises(ValueError,match='sequence'): a.validate_client_sequences({'a:2':one})


def test_overlapping_callbacks_rejected():
    e,r,s=fixture(); one=r['a:1']; two=dict(one,request_id='a:2')
    with pytest.raises(ValueError,match='one outstanding'): a.validate_client_sequences({'a:1':one,'a:2':two})


def test_case_clock_selects_reused_client_slice():
    first=dict(client_id='worker_0',request_id='worker_0:1',slot=0,deadline_perf=160.,submitted_perf=100.)
    later=dict(first,request_id='worker_0:2',deadline_perf=260.,submitted_perf=200.)
    client,rows=a.case_requests({'index':8,'slot':0},{'concurrency':8},{'a':first,'b':later},200.)
    assert client=='worker_0' and rows==[later]


def test_full32_only_rate_and_preserved_negative():
    rows=[dict(completed=True,baseline_exact=True) for _ in range(32)]
    assert a.qualified_rate('passed',rows,16.)['completed_decisions_per_second']==2.
    for status,count in [('deadline_failed',32),('passed',31)]:
        assert a.qualified_rate(status,rows[:count],16.)['completed_decisions_per_second'] is None
    rows[0]['baseline_exact']=False
    assert not a.qualified_rate('passed',rows,16.)['eligible']


def test_zero_or_nonfinite_duration_never_earns_rate():
    rows=[dict(completed=True,baseline_exact=True) for _ in range(32)]
    for value in (0.,float('nan'),float('inf'),-1.):
        with pytest.raises(ValueError): a.qualified_rate('passed',rows,value)


def cells():
    return [dict(configuration=name,block=block,eligible=True,wall_seconds=seconds,
                 completed_decisions_per_second=32/seconds) for block in (1,2)
            for name,seconds in (('cpu8',10.),('gpu8',20.),('gpu16',16.),('gpu32',15.))]


def test_aggregate_uses_total_work_over_total_time():
    values=cells(); values[4]['wall_seconds']=30.; values[4]['completed_decisions_per_second']=32/30.
    proof=a.aggregate_cells(values)
    assert proof['cpu8']['decisions_per_second']==64/40.
    assert proof['gpu16']['ratio_to_gpu8']==1.25


def test_one_failed_block_has_no_pooled_rate():
    values=cells(); values[-1].update(eligible=False,wall_seconds=None,completed_decisions_per_second=None)
    proof=a.aggregate_cells(values)
    assert proof['gpu32']['decisions_per_second'] is None
    assert proof['gpu32']['ratio_to_gpu8'] is None


def test_useful_signal_requires_10_percent_in_both_blocks():
    values=cells(); assert a.useful_candidates(values)['gpu16']['useful_scheduling_signal']
    values[-2]['completed_decisions_per_second']=values[-3]['completed_decisions_per_second']*1.09
    proof=a.useful_candidates(values)
    assert not proof['gpu16']['useful_scheduling_signal'] and proof['gpu32']['useful_scheduling_signal']


def test_failed_case_cannot_pass_useful_signal():
    values=cells(); values[-2]['eligible']=False
    assert not a.useful_candidates(values)['gpu16']['useful_scheduling_signal']


def test_source_cases_fixed_count_checked_before_access():
    with pytest.raises(ValueError,match='fixed32'): a.validate_cases([])
