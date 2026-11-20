"""Saved-only audit of observer overhead and GPU16 execution-cost evidence.

The accepted fixed-position checker is loaded privately with an explicit output
context binding. Historical modules/files are never mutated. No model, search or
game is created by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import importlib.util
import json
import math
import sys

from . import common as c
from tools.splendorust_gpu_throughput_20261105 import audit as old
from tools.splendorust_gpu_throughput_20261105 import qualification as q

LIBRARY = c.ROOT/'tools/splendorust_gpu_batchfill_20261105/audit.py'
LIBRARY_SHA = 'ef3e37d43fac0522cb44a903f37d35d3abe21b1b9f149040792e74f276f96c13'
_REUSED = None
read,digest,bits,timing = old.read,old.digest,old.bits,old.timing


def require(value,message):
    if not value: raise ValueError('GPU_COST_AUDIT: '+message)


def load_reused():
    """Authenticate a private checker copy; rebind only new evidence ownership."""
    global _REUSED
    require(c.sha(LIBRARY)==LIBRARY_SHA,'authenticated accepted fixed-position auditor')
    if _REUSED is None:
        name='tools.splendorust_gpu_batchfill_20261105._private_cost_audit_context'
        spec=importlib.util.spec_from_file_location(name,LIBRARY)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        old_common=module.c
        for key in ('ROOT','TOTAL_SECONDS','MODEL_SECONDS','INTERNAL_SECONDS','OUTER_SECONDS',
                    'EXECUTION_POLICY_ID','MAX_RSS','MAX_OUTPUT','MIN_DISK'):
            require(getattr(old_common,key)==getattr(c,key),'unchanged shared audit semantic constant '+key)
        public=sys.modules.get('tools.splendorust_gpu_batchfill_20261105.audit')
        prior_public_context=None if public is None else public.c
        module.c=c
        require(public is None or public.c is prior_public_context,'historical public module untouched')
        _REUSED=module
    require(_REUSED.c is c,'private checker uses only current evidence context')
    return _REUSED


def source_context_proof():
    load_reused()
    return dict(status='private_saved_checker_context_verified',library=c.artifact(LIBRARY),
                context=c.artifact(Path(c.__file__)),binding='private module.c only; no public module or historical file mutation',
                same_constants=['ROOT','TOTAL_SECONDS','MODEL_SECONDS','INTERNAL_SECONDS','OUTER_SECONDS',
                                'EXECUTION_POLICY_ID','MAX_RSS','MAX_OUTPUT','MIN_DISK'],models=0,games=0)


def overhead_gate(cells):
    order=['oh1_control','oh1_observed','oh2_observed','oh2_control']
    require([x['cell_id'] for x in cells]==order,'complete predetermined C,O,O,C overhead sequence')
    expected=[(1,False),(1,True),(2,True),(2,False)]
    for row,(block,observed) in zip(cells,expected):
        require(row['block']==block and row['observed'] is observed and row['phase']=='overhead' and
                row['configuration']=='gpu16' and row['route']=='gpu' and row['concurrency']==16 and row['scheduler']=='row_fifo32',
                'matched GPU16 overhead intervention only')
    complete=all(x['status']=='passed' and x['eligible'] and len(x['case_rows'])==32 and
                 all(r['completed'] and r['baseline_exact'] for r in x['case_rows']) for x in cells)
    pairs=[]; pooled=None
    if complete:
        for row in cells: require(math.isfinite(row['wall_seconds']) and row['wall_seconds']>0,'positive original overhead cell wall')
        for block,control,observed in ((1,cells[0],cells[1]),(2,cells[3],cells[2])):
            ratio=observed['wall_seconds']/control['wall_seconds']
            pairs.append(dict(block=block,control_cell=control['cell_id'],observed_cell=observed['cell_id'],
                              control_seconds=control['wall_seconds'],observed_seconds=observed['wall_seconds'],
                              observed_over_control_wall_ratio=ratio,passed=ratio<=1.10))
        pooled=(cells[1]['wall_seconds']+cells[2]['wall_seconds'])/(cells[0]['wall_seconds']+cells[3]['wall_seconds'])
    passed=complete and pooled<=1.05 and all(x['passed'] for x in pairs)
    return dict(status='observer_overhead_passed' if passed else 'observer_overhead_negative',passed=passed,
                complete_four_cells=complete,pooled_observed_over_control_wall_ratio=pooled,
                pooled_max_ratio=1.05,each_pair_max_ratio=1.10,pairs=pairs,
                decision_comparisons_exact=complete,no_retry=True,
                scope='Descriptive frozen observer admission on two matched blocks; no statistical hardware or strength claim')


def interval_union(intervals):
    """Union wall coverage, never sum overlapping concurrent spans as elapsed time."""
    values=sorted(intervals)
    if not values:return 0.
    total=0.;start,end=values[0]
    require(math.isfinite(start) and math.isfinite(end) and end>=start,'finite ordered interval')
    for left,right in values[1:]:
        require(math.isfinite(left) and math.isfinite(right) and right>=left,'finite ordered interval')
        if left>end:total+=end-start;start,end=left,right
        else:end=max(end,right)
    return total+end-start


def summarize_spans(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['kind_name']].append(row)
    return {name:dict(count=len(values),duration_seconds=timing([x['end']-x['start'] for x in values]),
                      summed_span_seconds=sum(x['end']-x['start'] for x in values),
                      union_wall_seconds=interval_union([(x['start'],x['end']) for x in values]),
                      scope='Host wall span durations; concurrent/nested kinds overlap and must not be added as panel wall shares')
            for name,values in sorted(groups.items())}


def validate_observer(value, enabled):
    from .observer import LABELS,COLUMNS,MAX_EVENTS
    require(value['schema']=='gpu_cost_wall_spans_v1' and value['enabled'] is enabled and
            value['columns']==list(COLUMNS) and value['labels']==list(LABELS) and
            value['maximum_events']==MAX_EVENTS and len(value['session_id'])==32,
            'frozen wall observer schema, labels and explicit mode')
    require(value['clock']=='time.perf_counter wall time' and len(value['events'])<=MAX_EVENTS,
            'bounded monotonic wall evidence')
    rows=[]; by_id={}
    for event in value['events']:
        require(type(event) is list and len(event)==len(COLUMNS),'complete observer tuple')
        row=dict(zip(COLUMNS,event)); ident=row['span_id']; kind=row['kind']
        require(type(ident) is int and ident>0 and ident not in by_id and type(kind) is int and 0<=kind<len(LABELS),
                'unique positive span and known kind')
        require(type(row['thread_id']) is int and type(row['owner']) is str and row['owner'] and row['failed'] is False and
                all(type(row[x]) in (int,float) and math.isfinite(row[x]) for x in ('start','end')) and
                0<=row['start']<=row['end'], 'successful finite ordered wall span')
        row['kind_name']=LABELS[kind];rows.append(row);by_id[ident]=row
    require(set(by_id)==set(range(1,len(rows)+1)),'every allocated completed observer span retained')
    require(enabled or not rows,'control observer performs no recorded spans')
    siblings=defaultdict(list)
    for row in rows:
        parent=row['parent_id']
        if parent is not None:
            require(type(parent) is int and parent in by_id and parent<row['span_id'],'explicit existing earlier parent')
            p=by_id[parent]
            require(p['thread_id']==row['thread_id'] and p['start']<=row['start']<=row['end']<=p['end'],
                    'same-thread child wholly inside parent')
        siblings[(row['thread_id'],parent)].append(row)
    for values in siblings.values():
        ordered=sorted(values,key=lambda x:x['start'])
        require(all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:])), 'same-thread sibling spans cannot overlap')
    previous_time=-1.;previous_index=0;markers={}
    for marker in value['markers']:
        require(set(marker)=={'label','event_index','perf_counter'} and type(marker['label']) is str and
                marker['label'] not in markers and type(marker['event_index']) is int and
                previous_index<=marker['event_index']<=len(rows) and math.isfinite(marker['perf_counter']) and
                marker['perf_counter']>=previous_time,'ordered unique explicit workload markers')
        previous_time=marker['perf_counter'];previous_index=marker['event_index'];markers[marker['label']]=marker
    return rows,markers


def children_of(rows):
    children=defaultdict(list)
    for row in rows:children[row['parent_id']].append(row)
    return children


def raw_components(rows):
    children=children_of(rows);result=[]
    expected=['raw_input_validation','tensor_construction','gpu_roundtrip_synchronized','output_conversion','raw_output_validation']
    for raw in (x for x in rows if x['kind_name']=='raw_forward_total'):
        values=sorted(children[raw['span_id']],key=lambda x:x['start'])
        require([x['kind_name'] for x in values]==expected,'every unchanged network call has all five ordered cost spans')
        result.append(dict(raw_span_id=raw['span_id'],owner=raw['owner'],start=raw['start'],end=raw['end'],
                           durations={x['kind_name']:x['end']-x['start'] for x in values}))
    require(sum(x['kind_name'] in expected for x in rows)==5*len(result),'no orphan raw network component')
    return result


def observer_service_coverage(rows, events, requests):
    children=children_of(rows);by_owner={}
    for row in rows:
        if row['kind_name']=='service_dispatch_total':
            require(row['owner'] not in by_owner,'one observed span per physical dispatch');by_owner[row['owner']]=row
    require(set(by_owner)=={'batch:'+str(e['batch_id']) for e in events},'every journaled physical dispatch observed exactly once')
    raw=raw_components(rows);raw_by_id={x['raw_span_id']:x for x in raw}
    expected=['service_input_packing','raw_forward_total','service_output_validation','service_receipts','journal_total','service_signal']
    batches=[]
    for event in events:
        dispatch=by_owner['batch:'+str(event['batch_id'])]
        values=sorted(children[dispatch['span_id']],key=lambda x:x['start'])
        require([x['kind_name'] for x in values]==expected,'all six ordered unchanged dispatch components')
        receipt=event['requests'][0];network=values[1]
        require(dispatch['start']<=receipt['dispatch_perf']<=network['start']<=network['end']<=receipt['complete_readback_perf']<=dispatch['end'],
                'independent journal and observed dispatch clocks agree')
        batches.append(dict(batch_id=event['batch_id'],occupied_rows=len(event['requests']),
                            request_input_sha256=[dict(slot=r['slot'],sha256=r['input_sha256']) for r in event['requests']],
                            **raw_by_id[network['span_id']]))
    require(len(raw)==len(events),'no additional unaccounted model calls in live service')
    journals=[x for x in rows if x['kind_name']=='journal_total']
    require(len(journals)==2*len(events),'one unchanged scheduler and one inference journal per batch')
    for row in journals:
        require([x['kind_name'] for x in sorted(children[row['span_id']],key=lambda x:x['start'])]==
                ['journal_serialization','journal_write_flush'],'journal serialization and write/flush coverage')
    callbacks=defaultdict(list);request_groups=defaultdict(list)
    for row in rows:
        if row['kind_name']=='callback_total':callbacks[row['owner']].append(row)
    for row in requests.values():
        if row['client_id']!='warmup':request_groups[row['client_id']].append(row)
    require(set(callbacks)==set(request_groups),'native callback owners exactly match service clients')
    for owner,values in callbacks.items():
        values.sort(key=lambda x:x['start']); bound=sorted(request_groups[owner],key=lambda x:x['submitted_perf'])
        require(len(values)==len(bound),'every native callback observed once')
        for span,request in zip(values,bound):
            parts=sorted(children[span['span_id']],key=lambda x:x['start'])
            require([x['kind_name'] for x in parts]==['callback_entry_checks','callback_service_wait','callback_bookkeeping'],
                    'complete callback preparation, service and bookkeeping spans')
            require(parts[1]['start']<=request['submitted_perf']<request['complete_readback_perf']<=parts[1]['end'] and
                    span['end']<request['deadline_perf'],'callback clock contains exact completed request under original deadline')
    return dict(dispatches=len(events),native_callbacks=sum(map(len,callbacks.values())),journal_spans=len(journals),raw_batches=batches)


def message_gaps(rows):
    groups=defaultdict(list)
    relevant={'callback_total','pipe_readline','pipe_read','pipe_read1','pipe_iteration_read','pipe_write','pipe_flush'}
    for row in rows:
        if row['kind_name'] in relevant:groups[(row['thread_id'],row['owner'])].append(row)
    inbound=[];outbound=[];callbacks=0
    for values in groups.values():
        values.sort(key=lambda x:x['start'])
        for index,row in enumerate(values):
            if row['kind_name']!='callback_total':continue
            callbacks+=1
            if index and values[index-1]['kind_name'] in {'pipe_readline','pipe_read','pipe_read1','pipe_iteration_read'}:
                gap=row['start']-values[index-1]['end'];require(gap>=0,'ordered input preparation gap');inbound.append(gap)
            if index+1<len(values) and values[index+1]['kind_name']=='pipe_write':
                gap=values[index+1]['start']-row['end'];require(gap>=0,'ordered output preparation gap');outbound.append(gap)
    return dict(callbacks=callbacks,input_gap_count=len(inbound),output_gap_count=len(outbound),
                read_end_to_callback_start_seconds=timing(inbound) if inbound else None,
                callback_end_to_write_start_seconds=timing(outbound) if outbound else None,
                input_gap_sum_seconds=sum(inbound),output_gap_sum_seconds=sum(outbound),
                scope='Consecutive same-thread same-client gaps: host message preparation/dispatch and scheduling; not pure parser/CPU time. Concurrent sums are nonadditive.')


def observer_file(folder, *, observed, mode, cell_id, phase):
    folder=Path(folder);value=read(folder/'observer_timing.json')
    require(value['scope']==dict(mode=mode,cell_id=cell_id,observed=observed,phase=phase,
                                exported_after_runtime_close=True,outside_measured_intervals=True), 'exact observer export scope')
    identity=read(folder/'runtime_identity.json')
    require(identity['timing_observer_enabled'] is observed and identity['timing_observer_schema']=='gpu_cost_wall_spans_v1',
            'loaded runtime observer matches assigned arm')
    rows,markers=validate_observer(value,observed)
    return rows,markers,dict(artifact=c.artifact(folder/'observer_timing.json'),session_id=value['session_id'],
                            observed=observed,span_count=len(rows),scope=value['scope'])


def recording_coverage(rows,cases,groups):
    wanted={case['case_id']+'/'+name for case in cases for name in
            ('attempt.json','raw_result.json','clock.json','result.json','root.json','after.json','service_calls.json','outcome.json')}
    wanted|={f'group:{index}/'+name for index in range(len(groups)) for name in ('start.json','summary.json')}
    recording=[x for x in rows if x['kind_name']=='case_recording']
    require(Counter(x['owner'] for x in recording)==Counter({name:1 for name in wanted}),
            'every actual mandatory case/group evidence artifact timed exactly once')
    return dict(spans=len(recording),case_files=8*len(cases),group_files=2*len(groups),
                scope='Original durable per-case/group serialization, write, flush and fsync; nested/concurrent durations nonadditive')


def audit_cell(plan,spec,worker_folder):
    folder=Path(worker_folder);reused=load_reused()
    proof=reused.audit_cell(plan,spec,folder)
    if proof['status']!='passed':
        require(proof['status']=='deadline_failed' and not proof['eligible'],'only independently established original clock failure is a valid negative cell')
        path=folder/'runtime/observer_timing.json'
        return dict(**proof,observed=spec['observed'],phase=spec['phase'],observer_coverage=None,panel_spans=None,
                    message_preparation_gaps=None,observer=dict(observed=spec['observed'],
                    preserved_partial_artifact=c.artifact(path) if path.exists() else None),
                    interval_scope='Original clock failure retained; partial observer evidence is not eligible for cost interpretation or completed-panel throughput.')
    summary=read(folder/'cell/cell_summary.json')
    require(summary['observed'] is spec['observed'] and summary['phase']==spec['phase'],'actual intervention and phase')
    rows,markers,observer=observer_file(folder/'runtime',observed=spec['observed'],mode='cell',cell_id=spec['cell_id'],phase=spec['phase'])
    require(set(markers)=={'warmup_start','warmup_end','panel_start','panel_end'} and
            markers['warmup_end']['perf_counter']<markers['panel_start']['perf_counter']<=summary['started_perf_counter'] and
            markers['panel_end']['perf_counter']>=summary['ended_perf_counter'],'warmup excluded and complete primary panel bracketed')
    panel=[x for x in rows if summary['started_perf_counter']<=x['start']<=x['end']<=summary['ended_perf_counter']]
    coverage=None
    if spec['observed']:
        events,requests,_=reused.read_journal(folder/'runtime','gpu')
        coverage=observer_service_coverage(rows,events,requests)
        coverage['raw_batches']=[x for x in coverage['raw_batches'] if summary['started_perf_counter']<=x['start']<=x['end']<=summary['ended_perf_counter']]
        require(len([x for x in panel if x['kind_name']=='callback_total'])==proof['callback_count'],'all native panel callback spans within primary clock')
        coverage['recording']=recording_coverage(panel,plan['cases'],spec['groups'])
    worker=read(folder/'worker_result.json')
    require(worker['observer_exports']==[observer['artifact']],'worker binds the one complete post-close observer export')
    return dict(**proof,observed=spec['observed'],phase=spec['phase'],observer=observer,observer_coverage=coverage,
                panel_spans=summarize_spans(panel),message_preparation_gaps=message_gaps(panel),
                interval_scope='Complete saved-position panel; no gameplay/new games. Host wall spans overlap across clients and kinds.')


def audit_qualification(plan,container):
    container=Path(container);folder=container/'worker';reused=load_reused()
    execution,absent=reused.audit_stage_execution(plan,container,mode='qualification')
    require(execution['status']=='completed' and execution['worker_exit_code']==0,'fresh GPU qualifier actual exit')
    require(read(folder/'runtime_sources.json')['sources']==plan['source_guard'],'actual GPU qualifier startup source guard')
    selected=plan['runtime']['qualification'];require(selected==q.build_plan(),'unchanged accepted mathematical qualification')
    c.verify(selected['reused_evidence']+[plan['reused_scheduler_qualification']])
    inherited=read(c.resolve(plan['reused_scheduler_qualification']))
    require(inherited['status']=='saved_batchfill_diagnostic_verified' and
            inherited['qualification']['collision_qualification']['callbacks']==96,
            'accepted prior packing, same-row collisions and saved full numerical qualification')
    path=folder/'qualification_gpu';proof=q.verify_qualification(plan['runtime'],path)
    require(proof==read(path/'saved_qualification_audit.json'),'fresh67 actual entrypoint proof independently reconstructed')
    events,requests,scheduling=reused.read_journal(folder/'gpu','gpu')
    closure=reused.audit_runtime(plan,folder/'gpu','gpu',events,requests,scheduling,'row_fifo32')
    require(closure['service_failure'] is None,'qualifier never poisoned')
    lib=old.load_saved_library(container);forensic,_=q.read_forensics(path/'entrypoint_forensics.jsonl')
    hashes={'search':[],'reference':[]};clocks=[e for e in forensic if e.get('operation') in ('outer_clock','outer_ordinary_clock')]
    for clock in clocks:old.validate_clock(clock)
    for event in forensic:
        if event.get('event')=='result' and event.get('operation') in ('ordinary_worker_request','reference_choose','direct_same_quota_choose','exchange'):
            hashes[event['arm']].extend(lib.result_callback_hashes(event['result']))
    for arm,slot in (('search',0),('reference',1)):
        values=lib.bind_client_callbacks('qualification_'+arm,hashes[arm],requests,slot=slot)
        for row in values:
            matching=[clock for clock in clocks if row['deadline_perf']==clock['started_perf_counter']+60. and
                      clock['started_perf_counter']<=row['submitted_perf'] and
                      row['complete_readback_perf']<=clock['started_perf_counter']+clock['elapsed_seconds']]
            require(len(matching)==1 and matching[0]['status']=='ok','each fresh qualification callback keeps original60s clock')
    require(sum(map(len,hashes.values()))==len(requests),'all fresh qualification callbacks accounted without extra searches')
    rows,markers,observer=observer_file(folder/'gpu',observed=True,mode='qualification',cell_id=None,phase='qualification')
    require(set(markers)=={'qualification_start','qualification_end'},'one whole actual qualification observer bracket')
    coverage=observer_service_coverage(rows,events,requests)
    require(all(markers['qualification_start']['perf_counter']<=x['start']<=x['end']<=markers['qualification_end']['perf_counter']
                for x in rows if x['kind_name'] in {'raw_forward_total','callback_total'}),'all qualifying work inside original observer bracket')
    worker=read(folder/'worker_result.json');result=worker['result']
    require(read(folder/'cleanup.json')['success'] is True and worker['mode']=='qualification' and worker['exit_code']==0 and
            worker['reference_weights_unchanged'] is True and worker['no_retry'] is True and worker['observer_exports']==[observer['artifact']] and
            result['status']=='qualification_passed' and result['observed'] is True and result['fresh_decisions']==67 and
            result['diagnostic_decisions']==result['new_games']==0 and result['reused_scheduler_qualification']==plan['reused_scheduler_qualification'] and
            result['gpu']==read(path/'qualification.json'),'exact qualifier outcome, unchanged weights, full observer and cleanup')
    return dict(status='saved_observed_gpu_entrypoint_verified',fresh_gpu=proof,callbacks={a:len(v) for a,v in hashes.items()},
                observer=observer,observer_coverage=dict(coverage,raw_batches=[]),runtime_closure=closure,
                reused_scheduler_qualification=plan['reused_scheduler_qualification'],
                reused_scope='Accepted physical packing,96 collision callbacks and source-bound cross-route numerics; no new CPU qualification',
                owned_processes_absent=absent,model_calls=0,new_games=0)


def expected_isolated_batches(plan):
    rows=[]
    for batch_index,start in enumerate(range(0,32,8)):
        indexes=list(range(start,start+8));cases=[plan['cases'][i] for i in indexes]
        tokens=[t for case in cases for t in case['expected_root_tokens']]
        outputs=[old.root_output(case,read(c.resolve(case['expected_results']['gpu']))['result']) for case in cases]
        rows.append(dict(batch_index=batch_index,case_indexes=indexes,case_ids=[x['case_id'] for x in cases],
                         input_tokens=tokens,input_sha256=digest(tokens),expected_outputs=outputs,
                         expected_source_results=[x['expected_results']['gpu'] for x in cases]))
    return rows


def audit_isolated(plan,spec,worker_folder):
    folder=Path(worker_folder);reused=load_reused()
    execution,absent=reused.audit_stage_execution(plan,folder.parent,mode='isolated',cell=spec)
    require(execution['status']=='completed' and execution['worker_exit_code']==0 and
            read(folder/'runtime_sources.json')['sources']==plan['source_guard'],'fresh isolated worker actual exit and startup guard')
    events,requests,scheduling=reused.read_journal(folder/'runtime','gpu')
    require(not events and not requests and not scheduling,'isolated physical calls have no logical clients or searches')
    closure=reused.audit_runtime(plan,folder/'runtime','gpu',events,requests,scheduling,'row_fifo32')
    require(closure['service_failure'] is None,'isolated runtime never poisoned')
    batches=expected_isolated_batches(plan);source=folder/'isolated'
    require(read(source/'batches.json')==batches,'all four exact same-source input batches and accepted GPU output rows')
    rows,markers,observer=observer_file(folder/'runtime',observed=True,mode='isolated',cell_id=spec['cell_id'],phase='isolated')
    components=raw_components(rows)
    require(len(components)==148 and len(rows)==148*6 and not any(x['kind_name'].startswith(('pipe_','callback_','journal_','service_')) for x in rows),
            'exactly148 physical forwards and their five components, no native/search/service calls')
    remaining={x['raw_span_id']:x for x in components};phase_proofs={};expected_markers=set()
    all_calls=[]
    for phase,cycles in (('warmup',5),('measured',32)):
        summary=read(source/phase/'summary.json');calls=[];expected_markers|={'isolated_'+phase+'_start','isolated_'+phase+'_end'}
        require(summary['phase']==phase and summary['cycles']==cycles and summary['calls']==4*cycles and
                summary['physical_rows']==32*cycles and summary['exact_saved_outputs'] is True and summary['no_retry'] is True and
                summary['loop_wall_seconds']==summary['ended_perf_counter']-summary['started_perf_counter']>0 and
                markers['isolated_'+phase+'_start']['perf_counter']<=summary['started_perf_counter'] and
                markers['isolated_'+phase+'_end']['perf_counter']>=summary['ended_perf_counter'],'fixed complete isolated phase and original loop interval')
        for index in range(4*cycles):
            location=source/phase/f'{index:04d}';result=read(location/'result.json');clock=read(location/'clock.json');old.validate_clock(clock)
            batch=batches[index%4]
            require(result['phase']==phase and result['index']==index and result['cycle']==index//4 and result['batch_index']==index%4 and
                    result['case_indexes']==batch['case_indexes'] and result['physical_batch']==8 and result['input_sha256']==batch['input_sha256'] and
                    result['clock']==clock and clock['status']=='ok' and clock['completed'] is True and clock['elapsed_seconds']<60. and
                    result['no_retry'] is True and result['outputs']==batch['expected_outputs'] and
                    bits([v for row in result['outputs'] for v in row])==bits([v for row in batch['expected_outputs'] for v in row]),
                    'every original60/75 isolated call exactly reproduces saved float32 output')
            start_label=f'isolated_{phase}_{index:04d}_start';end_label=f'isolated_{phase}_{index:04d}_end'
            expected_markers|={start_label,end_label}
            require(result['observer_start']==markers[start_label] and result['observer_end']==markers[end_label] and
                    summary['started_perf_counter']<=markers[start_label]['perf_counter']<=clock['started_perf_counter']<=clock['ended_perf_counter']<=
                    markers[end_label]['perf_counter']<=summary['ended_perf_counter'],'original isolated call and observer markers nested inside phase')
            matched=[x for x in remaining.values() if clock['started_perf_counter']<=x['start']<=x['end']<=clock['ended_perf_counter']]
            require(len(matched)==1,'one physical forward per original complete-call clock')
            raw=matched[0];remaining.pop(raw['raw_span_id'])
            calls.append(dict(phase=phase,index=index,batch_index=index%4,input_sha256=batch['input_sha256'],clock=clock,**raw))
        require(summary['elapsed_complete_call_seconds']==[x['clock']['elapsed_seconds'] for x in calls],'complete-call timing independently recomputed')
        require(all(a['clock']['ended_perf_counter']<=b['clock']['started_perf_counter'] for a,b in zip(calls,calls[1:])),
                'one serial isolated physical call at a time')
        phase_proofs[phase]=dict(calls=len(calls),loop_wall_seconds=summary['loop_wall_seconds'],
                                complete_call_seconds=timing([x['clock']['elapsed_seconds'] for x in calls]),
                                raw_forward_seconds=timing([x['end']-x['start'] for x in calls]),
                                components={name:timing([x['durations'][name] for x in calls]) for name in calls[0]['durations']})
        all_calls.extend(calls)
    require(not remaining and set(markers)==expected_markers,'complete isolated event and marker inventory')
    summary=read(source/'summary.json');worker=read(folder/'worker_result.json')
    require(summary['status']=='isolated_batches_passed' and summary['specification']==spec and summary['cell_id']==spec['cell_id'] and
            summary['route']=='gpu' and summary['scheduler']=='row_fifo32' and summary['observed'] is True and
            summary['warmup_calls']==20 and summary['measured_calls']==128 and summary['physical_batch']==8 and
            summary['all_outputs_exact'] is True and summary['batches_sha256']==digest(batches) and
            summary['ordinary_decisions']==summary['new_games']==summary['training_updates']==0 and summary['no_retry'] is True and
            summary['phases']=={p:read(source/p/'summary.json') for p in ('warmup','measured')},'isolated summary matches all saved raw calls')
    require(worker['mode']=='isolated' and worker['cell_id']==spec['cell_id'] and worker['exit_code']==0 and worker['result']==summary and
            worker['observer_exports']==[observer['artifact']] and worker['reference_weights_unchanged'] is True and worker['no_retry'] is True and
            read(folder/'cleanup.json')['success'] is True,'isolated worker exact outputs and cleanup')
    return dict(status='saved_isolated_batches_verified',cell_id=spec['cell_id'],observed=True,block=spec['block'],side=spec['side'],
                observer=observer,runtime_closure=closure,phases=phase_proofs,calls=all_calls,
                owned_processes_absent=absent,ordinary_decisions=0,model_calls=0,new_games=0,
                scope='Same-source fixed fullB8 inputs. Complete-call/forward wall includes CPU, transfer, device and readback; not kernel or gameplay timing.')


def validate_plan(plan):
    from . import design,backend
    require(plan['schema']=='splendorust_gpu_costs_v1' and plan['cells']==design.cells() and plan['isolated']==design.isolated(),
            'fixed unchanged GPU16 arms and bracketed isolated stages')
    require(plan['worker_limits']==dict(qualification_seconds=2700,cell_seconds=900,isolated_seconds=900) and
            plan['total_seconds']==c.TOTAL_SECONDS==14400 and plan['model_seconds']==c.MODEL_SECONDS==10800 and
            plan['internal_seconds']==60. and plan['outer_seconds']==75. and plan['execution_policy_id']==c.EXECUTION_POLICY_ID,
            'original total/model/stage/decision clocks')
    require(plan['maximum_worker_rss_bytes']==c.MAX_RSS and plan['maximum_output_bytes']==c.MAX_OUTPUT and
            plan['minimum_free_disk_bytes']==c.MIN_DISK and plan['available_ram_reserve_bytes']==0 and plan['forecast_admission_gate'] is False,
            'actual resource caps only')
    require(plan['worker_affinity']==list(range(19)) and plan['supervisor_affinity']==[19] and plan['group_gap_seconds']==0.,'unchanged placement and wave spacing')
    require(plan['overhead']==dict(order=['oh1_control','oh1_observed','oh2_observed','oh2_control'],pooled_max_ratio=1.05,
            each_pair_max_ratio=1.10,minimum_complete_cells=4,outcome_if_not_passed='valid negative observer qualification; no full diagnostic or retry') and
            plan['full_stage_order']==['iso1_pre','diag1','iso1_post','iso2_pre','diag2','iso2_post'],'prospective overhead gates and full-stage order')
    require(plan['maximum_diagnostic_decisions']==192 and plan['qualification_decisions_per_route']==67 and plan['qualification_routes']==['gpu'] and
            plan['retries']==plan['new_games']==plan['training_updates']==0 and plan['larger_data_deferred'] is True and
            plan['automatic_followup_games'] is False and plan['automatic_followup_training'] is False,'bounded saved-only scope')
    require(plan['late_completed_responses_accepted'] is True and plan['descriptive_latency_threshold_seconds']==[1.,9.,10.],
            'historical thresholds remain descriptive')
    require(plan['observer_source_equivalence']==backend.source_equivalence(),'independently reconstructed frozen source equivalence')
    for artifact,expected in ((plan['prior']['plan'],design.PLAN_SHA),(plan['prior']['audit'],design.AUDIT_SHA),(plan['prior']['closeout'],design.CLOSEOUT_SHA)):
        require(artifact['sha256']==expected,'authenticated accepted prior input');c.verify([artifact])
    previous=read(c.resolve(plan['prior']['plan']))
    require(plan['runtime']==previous['runtime'] and plan['cases']==previous['cases'] and plan['warmup']==previous['warmup'] and
            plan['source_guard']==previous['source_guard'] and plan['prior']['frozen_inputs']==previous['inputs'],
            'exact accepted source positions, warmup, model and search')
    require(plan['reused_scheduler_qualification']==plan['prior']['audit'],'reused physical/scheduler qualification binds accepted final audit')
    require(plan['runtime']['qualification']==q.build_plan(),'unchanged mathematical qualifier')
    load_reused().validate_cases(plan['cases']);source_context_proof()
    return True


def verify_gate(plan,output,proofs):
    expected=overhead_gate(proofs);value=read(output/'overhead_gate.json')
    artifacts=[c.artifact(output/'cells'/x['cell_id']/'saved_audit.json') for x in proofs]
    require(value==dict(expected,plan_sha256=c.sha(c.PLAN),source_cell_audits=artifacts),
            'prospective gate independently reconstructed from the four original audited cells')
    return value


def verify_full_binding(plan,spec,container,gate):
    owner=read(container/'owner.json');args=owner['args']
    path=c.OUT/'overhead_gate.json';expected=c.artifact(path)
    require(args.count('--gate-sha256')==1 and args[args.index('--gate-sha256')+1]==expected['sha256'],
            'actual full worker command binds original passed gate')
    require(read(container/'worker/overhead_gate_binding.json')==dict(gate=expected,source_cell_audits=gate['source_cell_audits'],
            passed=True,independently_recomputed=True,verified_before_model_creation=True),'full worker actually verified original overhead admission')


def isolate_live_comparison(plan,cells,isolated):
    by_id={x['cell_id']:x for x in isolated};reference=expected_isolated_batches(plan);out=[]
    expected_slot_hashes=[{i%8:digest(plan['cases'][i]['expected_root_tokens']) for i in b['case_indexes']} for b in reference]
    for cell in cells:
        if cell['phase']!='diagnostic':continue
        block=1 if cell['cell_id']=='diag1' else 2
        before,after=by_id[f'iso{block}_pre'],by_id[f'iso{block}_post']
        measured=[x for proof in (before,after) for x in proof['calls'] if x['phase']=='measured']
        live=cell['observer_coverage']['raw_batches'];matching=[]
        for event in live:
            actual={x['slot']:x['sha256'] for x in event['request_input_sha256']}
            indexes=[i for i,wanted in enumerate(expected_slot_hashes) if actual==wanted]
            if indexes:
                require(len(indexes)==1,'unambiguous exact full physical root-batch match')
                matching.append(dict(live_batch_id=event['batch_id'],isolated_batch_index=indexes[0],
                    live_gpu_roundtrip_seconds=event['durations']['gpu_roundtrip_synchronized'],
                    isolated_gpu_roundtrip_seconds=timing([x['durations']['gpu_roundtrip_synchronized'] for x in measured if x['batch_index']==indexes[0]])))
        out.append(dict(cell_id=cell['cell_id'],isolated_brackets=[before['cell_id'],after['cell_id']],
            all_live_gpu_roundtrip_seconds=timing([x['durations']['gpu_roundtrip_synchronized'] for x in live]),
            full_root_isolated_gpu_roundtrip_seconds=timing([x['durations']['gpu_roundtrip_synchronized'] for x in measured]),
            exact_full_physical_input_matches=matching,exact_full_physical_input_match_count=len(matching),
            scope='All-live versus isolated-root distributions differ in workload and host conditions; descriptive only. Only listed full8-row digest matches have identical physical inputs; no causal pressure/kernel claim.'))
    return out


def audit_study(plan,output,*,expected_plan_sha,controller_session,controller_exit_code,controller_chunk,allow_failure=False):
    output=Path(output);require(output.resolve()==c.OUT.resolve() and c.sha(c.PLAN)==expected_plan_sha,'exact frozen evidence ownership and plan')
    validate_plan(plan);c.verify(plan['inputs'])
    require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<plan['maximum_output_bytes'],'actual retained output cap')
    claim,execution=read(output/'claim.json'),read(output/'execution.json')
    require(claim['plan_sha256']==execution['plan_sha256']==expected_plan_sha and claim['total_seconds']==plan['total_seconds'] and
            claim['model_seconds']==execution['model_seconds']==plan['model_seconds'] and
            claim['monotonic_origin']==execution['monotonic_origin'] and claim['unix_origin']==execution['unix_origin'] and
            0<=execution['monotonic_end']-execution['monotonic_origin']<plan['total_seconds'] and
            0<=execution['unix_end']-execution['unix_origin']<plan['total_seconds'],'one unextended original global dual clock')
    require(execution['retries']==execution['new_games']==execution['game_routes_started']==execution['training_updates']==0 and
            not execution['cleanup_errors'] and execution['execution_policy_id']==c.EXECUTION_POLICY_ID,'scope and owned cleanup')
    absent=old.require_processes_absent(execution['observed_owned_identities']+[dict(pid=claim['pid'],created=claim['created'])])
    require(type(controller_session) is int and controller_session>0 and bool(controller_chunk),'actual retained controller exit attribution')
    base=dict(plan_sha256=expected_plan_sha,auditor=c.artifact(Path(__file__)),source_context=source_context_proof(),
        frozen_inputs_unchanged=len(plan['inputs']),exact_owned_processes_absent=absent,
        retained_controller_exit=dict(session_id=controller_session,exit_code=controller_exit_code,tool_chunk=controller_chunk,
                                     source='Actual retained execution supplied by root; exact owner absence independently checked'),
        auditor_models_launched=0,auditor_games_launched=0,new_games=0,training_updates=0)
    if execution['status']=='failed':
        require(allow_failure and controller_exit_code!=0 and read(output/'failure.json')==execution and execution['error'],
                'technical failure requires explicit saved failure scope')
        return dict(**base,status='preserved_failed_attempt_verified',eligible=False,failure=execution['error'],
                    scope='Preservation, clock and owner cleanup only; no completed diagnostic acceptance')
    require(controller_exit_code==0 and execution['error'] is None and not (output/'failure.json').exists(),'actual successful owner exit')
    qualification=audit_qualification(plan,output/'qualification')
    require(read(output/'qualification/saved_audit.json')==qualification,'reconstructed fresh qualifier matches saved receipt')
    proofs=[];artifacts=[];cells=[];isolated=[]
    for spec in plan['cells'][:4]:
        container=output/'cells'/spec['cell_id'];value=audit_cell(plan,spec,container/'worker')
        require(value==read(container/'saved_audit.json'),'raw overhead audit independently reproduced')
        proofs.append(value);cells.append(value);artifacts.append(c.artifact(container/'saved_audit.json'))
    gate=verify_gate(plan,output,proofs)
    stages=[dict(stage_id='qualification',status='completed',worker_exit_code=0)]
    for proof in proofs:
        actual=read(output/'cells'/proof['cell_id']/'execution.json')
        stages.append(dict(stage_id=proof['cell_id'],status=actual['status'],worker_exit_code=actual['worker_exit_code']))
    if gate['passed']:
        all_specs={x['cell_id']:x for x in plan['cells']+plan['isolated']}
        for stage_id in plan['full_stage_order']:
            spec=all_specs[stage_id];is_iso=stage_id.startswith('iso');container=output/('isolated' if is_iso else 'cells')/stage_id
            verify_full_binding(plan,spec,container,gate)
            value=(audit_isolated if is_iso else audit_cell)(plan,spec,container/'worker')
            require(read(container/'saved_audit.json')==value,'raw full diagnostic audit independently reproduced')
            (isolated if is_iso else cells).append(value);artifacts.append(c.artifact(container/'saved_audit.json'))
            stages.append(dict(stage_id=stage_id,status='completed',worker_exit_code=0))
        expected_status='completed_cost_diagnostic_pending_independent_acceptance'
    else:
        expected_status='observer_overhead_negative_pending_independent_acceptance'
        require(not (output/'isolated').exists() and all(not (output/'cells'/x['cell_id']).exists() for x in plan['cells'][4:]),
                'valid negative overhead stops all full-model followup')
    require(execution['status']==expected_status and execution['stages']==stages,'complete exact prescribed stage sequence and no hidden retry')
    completion=read(output/'completion.json')
    require(completion['status']==expected_status and completion['execution']==c.artifact(output/'execution.json') and
            completion['plan']==c.artifact(c.PLAN) and completion['qualified'] is True and completion['audited_stages']==len(artifacts) and
            completion['stage_audits']==artifacts and completion['new_games']==completion['training_updates']==0 and completion['inputs_unchanged'] is True,
            'completion binds every original independently accepted stage')
    sessions=[x['observer'].get('session_id') for x in cells+isolated]+[qualification['observer']['session_id']]
    sessions=[x for x in sessions if x is not None];require(len(sessions)==len(set(sessions)),'every observed runtime session fresh')
    source_numerics=load_reused().cross_route_sources(plan);c.verify(plan['inputs'])
    return dict(**base,status='saved_gpu_cost_diagnostic_verified' if gate['passed'] else 'saved_observer_overhead_negative_verified',
                qualification=qualification,overhead_gate=gate,cells=cells,isolated=isolated,cross_route_source_roots=source_numerics,
                isolated_live_comparisons=isolate_live_comparison(plan,cells,isolated) if gate['passed'] else [],
                attempted_decisions=sum(sum(x['attempted'] for x in cell['case_rows']) for cell in cells),
                completed_decisions=sum(sum(x['completed'] for x in cell['case_rows']) for cell in cells),
                scope='Saved-position GPU16 execution costs only. Overlapping host wall times are nonadditive; no pure GPU kernel, new game, strength, collector qualification or causal backend claim.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected-plan-sha',required=True);parser.add_argument('--controller-session',type=int,required=True)
    parser.add_argument('--controller-exit-code',type=int,required=True);parser.add_argument('--controller-chunk',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--allow-failure',action='store_true');args=parser.parse_args()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources();plan=read(c.PLAN);require(plan['source_guard']==sources,'saved corrected-source startup guard')
    result=audit_study(plan,c.OUT,expected_plan_sha=args.expected_plan_sha,controller_session=args.controller_session,
        controller_exit_code=args.controller_exit_code,controller_chunk=args.controller_chunk,allow_failure=args.allow_failure)
    destination=Path(args.output).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    require(not destination.is_relative_to(c.OUT),'final independent audit outside frozen experiment evidence')
    with destination.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(c.canonical(dict(status=result['status'],output=c.artifact(destination),models=0,games=0)))
    return 0


if __name__=='__main__':raise SystemExit(main())
