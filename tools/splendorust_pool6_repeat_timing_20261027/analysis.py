"""Descriptive repeated-position analysis; no strength or admission inference."""
import math
import statistics
from . import common as c


def distribution(values):
    values=sorted(values)
    if not values: return dict(n=0, minimum=None, median=None, p90=None, maximum=None, mean=None)
    return dict(n=len(values), minimum=values[0], median=statistics.median(values),
        p90=values[math.ceil(.9*len(values))-1], maximum=values[-1], mean=statistics.mean(values))


def measurements(row):
    clock=row['clock']; child=row.get('child_profile') or {}
    profiles=child.get('profiles',[])
    numeric=lambda x:type(x) in (int,float) and math.isfinite(x)
    handler=sum(p['handle_end_ns']-p['handle_start_ns'] for p in profiles)/1e9 if profiles and all(
        numeric(p.get(k)) for p in profiles for k in ('handle_end_ns','handle_start_ns')) else None
    emission=(child['post_flush_ns']-child['pre_write_ns'])/1e9 if all(
        numeric(child.get(key)) for key in ('post_flush_ns','pre_write_ns')) else None
    result=clock.get('result') or {}; trace=result.get('diagnostics') or {}
    return dict(sequence=row['spec']['sequence'], round=row['spec']['round'],
        status=clock['status'], receipt_seconds=clock.get('receipt_elapsed_seconds'),
        handler_seconds=handler, emission_seconds=emission,
        reply_bytes=(row.get('parent_profile') or {}).get('response_raw_bytes'),
        inference_calls=trace.get('inference_calls'),
        simulations=[p.get('simulations') for p in trace.get('phases',[])],
        margin_crossing=clock.get('receipt_elapsed_seconds') is not None and clock['receipt_elapsed_seconds']>=.7,
        failed_request=clock['status']!='ok',
        deadline_failed=clock['status']=='timeout' or any(marker in str(clock.get('detail','')) for marker in
            ('SAMPLING_INTERNAL_DEADLINE','ONE_SECOND_INTERNAL_DEADLINE')))


def analyze(records,panel,matrix,complete,boundary=lambda:None):
    c.require(type(complete) is bool,'Explicit completion label')
    c.require(len(records)<=144 and len(matrix['requests'])==144,'Fixed repeat inventory')
    if complete:c.require(len(records)==144 and all(r['clock']['status']=='ok' for r in records),'Complete matrix required')
    for row,spec in zip(records,matrix['requests']):
        boundary();c.require(row['spec']==spec,'Analysis ordered row binding')
    cells=[];paired=[]
    for pos in panel['positions']:
        by_route={}
        for route in ('frozen','research'):
            rows=[r for r in records if r['spec']['case_id']==pos['case_id'] and r['route']==route]
            values=[measurements(r) for r in rows];by_route[route]={v['round']:v for v in values}
            if complete:c.require(len(values)==12,'Twelve observations per cell')
            cells.append(dict(case_id=pos['case_id'],selection=pos['selection'],route=route,
                observations=values,attempted=len(rows),completed=sum(v['status']=='ok' for v in values),
                failed_requests=sum(v['failed_request'] for v in values),
                deadline_failures=sum(v['deadline_failed'] for v in values),
                margin_crossings=sum(v['margin_crossing'] for v in values),
                receipt=distribution([v['receipt_seconds'] for v in values if v['receipt_seconds'] is not None]),
                handler=distribution([v['handler_seconds'] for v in values if v['handler_seconds'] is not None]),
                emission=distribution([v['emission_seconds'] for v in values if v['emission_seconds'] is not None])))
        diffs=[]
        for repeat in range(12):
            a=by_route['frozen'].get(repeat);b=by_route['research'].get(repeat)
            if a is None or b is None:continue
            diffs.append(dict(round=repeat,both_completed=a['status']==b['status']=='ok',
                candidate_minus_frozen_receipt_seconds=(b['receipt_seconds']-a['receipt_seconds'])
                    if a['receipt_seconds'] is not None and b['receipt_seconds'] is not None else None,
                candidate_minus_frozen_handler_seconds=(b['handler_seconds']-a['handler_seconds'])
                    if a['handler_seconds'] is not None and b['handler_seconds'] is not None else None,
                candidate_minus_frozen_emission_seconds=(b['emission_seconds']-a['emission_seconds'])
                    if a['emission_seconds'] is not None and b['emission_seconds'] is not None else None))
        paired.append(dict(case_id=pos['case_id'],observations=diffs,
            candidate_minus_frozen_receipt=distribution([v['candidate_minus_frozen_receipt_seconds'] for v in diffs
                if v['candidate_minus_frozen_receipt_seconds'] is not None])))
    totals={}
    for route in ('frozen','research'):
        values=[measurements(r) for r in records if r['route']==route]
        totals[route]=dict(attempted=len(values),completed=sum(v['status']=='ok' for v in values),
            margin_crossings=sum(v['margin_crossing'] for v in values),
            failed_requests=sum(v['failed_request'] for v in values),
            deadline_failures=sum(v['deadline_failed'] for v in values),
            receipt=distribution([v['receipt_seconds'] for v in values if v['receipt_seconds'] is not None]),
            handler=distribution([v['handler_seconds'] for v in values if v['handler_seconds'] is not None]),
            emission=distribution([v['emission_seconds'] for v in values if v['emission_seconds'] is not None]))
    return dict(schema='pool6_repeat_timing_descriptive_v1',complete=complete,
        attempted=len(records),planned=144,cells=cells,paired_differences=paired,by_route=totals,
        distribution_definition='p90 nearest rank; no discarded observations or warmups',
        confidence_intervals=None,independent_trial_assumption=False,
        limitations=['Targets selected from previous slowness, controls fixed structurally',
            'Repeated requests are temporally dependent; no population timeout-rate estimate',
            'Both workers resident differs from prior serial admission lifecycle',
            'No controlled load intervention; handler/emission boundaries do not isolate neural/native/IPC costs',
            'No playing-strength, live-reliability, or admission claim'],
        prior_admission_still_negative=True,benchmark_authorized_to_launch=False,
        new_games=0,training=False,larger_data_deferred=True)
