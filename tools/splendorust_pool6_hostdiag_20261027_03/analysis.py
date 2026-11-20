"""Descriptive bounded host observations; never an admission or causal test."""
import math
import statistics
from collections import Counter
from . import common as c
from tools.splendorust_pool6_repeat_timing_20261027.analysis import measurements as timing_measurements


def distribution(values):
    values=sorted(values)
    if not values:return dict(n=0,minimum=None,median=None,p95=None,maximum=None,mean=None)
    return dict(n=len(values),minimum=values[0],median=statistics.median(values),
        p95=values[math.ceil(.95*len(values))-1],maximum=values[-1],mean=statistics.mean(values))


def measurements(row):
    value=timing_measurements(row);spec=row['spec']
    value.update(stage=spec['stage'],route=spec['route'],position_index=spec['position_index'],
        case_id=spec['case_id'],observer_enabled=spec['observer_enabled'],
        host_summary=host_summary(row),
        host_observation_file=row.get('host_observation_file'))
    return value


def span_availability(samples,whole_window=None):
    from .host_observer import host_cpu_availability
    availability=host_cpu_availability(samples)
    if whole_window is not None and whole_window['valid'] is False:
        return {**availability,'valid':False,'available':False,'busy_fraction':None,
            'reason':'whole_window_host_cpu_inconsistent',
            'whole_window_offending_pairs':whole_window['offending_pairs'],
            'local_interval_classification':{key:availability[key] for key in ('valid','reason','sample_count','offending_pairs')}}
    return availability


def counter_span(samples,route,*,whole_window=None):
    if len(samples)<2:return None
    availability=span_availability(samples,whole_window)
    first,last=samples[0],samples[-1]
    results=[]
    c.require(len(first['processes'])==len(last['processes']),'Counter span process coverage')
    for previous,current in zip(samples,samples[1:]):
        c.require(len(previous['processes'])==len(current['processes']),'Adjacent process coverage')
        for before,after in zip(previous['processes'],current['processes']):
            c.require(before['identity']==after['identity'],'Adjacent process identity')
            c.require(all(after[key]>=before[key] for key in ('kernel_100ns','user_100ns','page_faults')),
                      'Adjacent process counter reset')
    for before,after in zip(first['processes'],last['processes']):
        c.require(before['identity']==after['identity'],'Counter span identity')
        cpu=(after['kernel_100ns']+after['user_100ns']-before['kernel_100ns']-before['user_100ns'])/1e7
        faults=after['page_faults']-before['page_faults']
        c.require(cpu>=0 and faults>=0,'Counter span reset')
        results.append(dict(identity=before['identity'],cpu_seconds=cpu,page_faults=faults,
            maximum_working_set_bytes=max(s['processes'][len(results)]['working_set_bytes'] for s in samples),
            maximum_private_bytes=max(s['processes'][len(results)]['private_bytes'] for s in samples)))
    active=[x for x in results if x['identity']['role'].startswith(route+':')]
    c.require(bool(active),'Active player process scope')
    return dict(sample_count=len(samples),first_sample_start_ns=first['start_ns'],last_sample_end_ns=last['end_ns'],
        span_seconds=(last['end_ns']-first['start_ns'])/1e9,
        active_player_cpu_seconds=sum(x['cpu_seconds'] for x in active),
        active_player_page_faults=sum(x['page_faults'] for x in active),
        host_busy_fraction=availability['busy_fraction'],host_cpu_availability=availability,processes=results)


def host_summary(row):
    value=row.get('host_observation') or {}
    if not value.get('enabled') or value.get('status')!='observed':
        return dict(enabled=value.get('enabled'),status=value.get('status'),measurements_available=False)
    samples=value['samples'];parent=row.get('parent_profile') or {}
    from .host_observer import host_cpu_availability
    availability=host_cpu_availability(samples)
    c.require(value['host_cpu_availability']==availability,'Saved host CPU availability differs from raw adjacent intervals')
    start,end=parent.get('request_start_ns'),parent.get('receipt_arrival_ns')
    within=[s for s in samples if start is not None and end is not None and start<=s['start_ns']<=s['end_ns']<=end]
    inside_availability=span_availability(within,availability)
    contexts=value['context_switch_deltas']
    return dict(enabled=True,status=value['status'],measurements_available=True,sample_count=len(samples),
        complete_samples_inside_request=len(within),missed_intervals=value['missed_intervals'],
        maximum_sample_lateness_ns=max((s.get('lateness_ns') or 0 for s in samples),default=0),
        minimum_available_ram_bytes=min((s['system']['available_physical_bytes'] for s in samples),default=None),
        minimum_in_request_available_ram_bytes=min((s['system']['available_physical_bytes'] for s in within),default=None),
        minimum_commit_headroom_bytes=min((s['system']['commit_headroom_bytes'] for s in samples),default=None),
        host_cpu_availability=availability,in_request_host_cpu_availability=inside_availability,
        whole_observation_window=counter_span(samples,row['route']),
        wholly_in_request_sample_span=counter_span(within,row['route'],whole_window=availability),
        context_switch_window=dict(start_ns=value['context_before']['start_ns'],end_ns=value['context_after']['end_ns'],
            active_player_live_thread_aggregate_net_change=sum(x['net_change'] for x in contexts if x['identity']['role'].startswith(row['route']+':')),
            interval_total_context_switches=None,
            interval_total_context_switches_reason='not_identifiable_from_live_thread_aggregates',
            processes=contexts),
        scope='Whole-window and context changes extend outside request; inner sample span covers only part of request. An inconsistent host CPU interval makes host CPU unavailable throughout that window, including subspans; local interval consistency is classification only. Valid process counters and memory remain separate. Live-thread aggregate changes can be negative and never identify interval event totals or direct scheduling delay.')


def availability_summary(values):
    """Count missing optional values explicitly; disabled/failed windows are separate."""
    observed=[v for v in values if v['host_summary'].get('measurements_available') is True]
    result=dict(attempted_windows=len(values),enabled_windows=sum(v['observer_enabled'] for v in values),
        disabled_windows=sum(not v['observer_enabled'] for v in values),
        accepted_measurement_windows=len(observed),
        failed_measurement_windows=sum(v['observer_enabled'] and not v['host_summary'].get('measurements_available',False) for v in values),
        definitions='CPU availability denominators contain accepted enabled windows only; disabled and failed measurements are counted separately.')
    for name,key in (('whole_window','host_cpu_availability'),('in_request_span','in_request_host_cpu_availability')):
        unavailable=[v for v in observed if not v['host_summary'][key]['available']]
        reasons=Counter(v['host_summary'][key]['reason'] for v in unavailable)
        result[name]=dict(denominator=len(observed),available=len(observed)-len(unavailable),unavailable=len(unavailable),
            reasons=dict(sorted(reasons.items())),
            busy_fraction=distribution([v['host_summary'][key]['busy_fraction'] for v in observed if v['host_summary'][key]['available']]),
            unavailable_windows=[dict(sequence=v['sequence'],stage=v['stage'],route=v['route'],case_id=v['case_id'],
                availability=v['host_summary'][key]) for v in unavailable])
    return result


def summarize(values):
    return dict(attempted=len(values),completed=sum(x['status']=='ok' for x in values),
        margin_crossings=sum(x['margin_crossing'] for x in values),
        failed_requests=sum(x['failed_request'] for x in values),
        deadline_failures=sum(x['deadline_failed'] for x in values),
        receipt=distribution([x['receipt_seconds'] for x in values if x['receipt_seconds'] is not None]),
        handler=distribution([x['handler_seconds'] for x in values if x['handler_seconds'] is not None]),
        emission=distribution([x['emission_seconds'] for x in values if x['emission_seconds'] is not None]),
        host_cpu_availability=availability_summary(values))


def analyze(records,panel,matrix,complete,boundary=lambda:None):
    from .runner import overhead_gate
    c.require(type(complete) is bool and len(matrix['requests'])==216 and len(records)<=216,
        'Explicit bounded diagnostic completion')
    if complete:c.require(len(records)==216 and all(r['clock']['status']=='ok' for r in records),
        'Complete diagnostic inventory')
    values=[]
    for row,spec in zip(records,matrix['requests']):
        boundary();c.require(row['spec']==spec,'Analysis ordered specification')
        values.append(measurements(row))
    groups=[];cells=[];overhead_pairs=[]
    for stage in ('overhead','diagnostic'):
        for route in ('frozen','research'):
            modes=(False,True) if stage=='overhead' else (True,)
            for mode in modes:
                subset=[v for v in values if v['stage']==stage and v['route']==route and v['observer_enabled']==mode]
                groups.append(dict(stage=stage,route=route,observer_enabled=mode,**summarize(subset)))
    for position in panel['positions']:
        for route in ('frozen','research'):
            subset=[v for v in values if v['position_index']==position['position_index'] and v['route']==route]
            cells.append(dict(case_id=position['case_id'],selection=position['selection'],route=route,
                original_source_route=position['source_route'],historical_target_available=route==position['source_route'],
                observations=subset,**summarize(subset)))
            overhead=[v for v in subset if v['stage']=='overhead']
            if overhead:
                off=[v['receipt_seconds'] for v in overhead if not v['observer_enabled'] and v['receipt_seconds'] is not None]
                on=[v['receipt_seconds'] for v in overhead if v['observer_enabled'] and v['receipt_seconds'] is not None]
                overhead_pairs.append(dict(case_id=position['case_id'],route=route,off_count=len(off),on_count=len(on),
                    on_minus_off_mean_seconds=statistics.mean(on)-statistics.mean(off) if on and off else None,
                    on_over_off_mean=statistics.mean(on)/statistics.mean(off) if on and off and statistics.mean(off)>0 else None))
    return dict(schema='pool6_hostdiag_descriptive_analysis_v3',complete=complete,attempted=len(records),maximum_planned=216,
        overhead_gate=overhead_gate([r for r in records if r['spec']['stage']=='overhead']),
        host_cpu_availability=availability_summary(values),
        groups=groups,cells=cells,overhead_cell_pairs=overhead_pairs,
        distribution_definition='p95 nearest rank; no discarded warmups, trimming or replacements',
        independent_trial_assumption=False,confidence_intervals=None,
        limitations=['Selected past slow roots and dependent repeats do not estimate live-game timeout risk',
            'Four-GiB headroom is a disclosed diagnostic context, not a randomized host-load intervention',
            'Sampled process CPU includes native helpers when identified; window endpoints extend beyond the handler',
            'Inconsistent adjacent host-global CPU intervals remain raw evidence with busy fraction unavailable; availability counts disclose every omitted optional value',
            'Combined page faults cannot identify hard faults; signed live-thread context aggregate changes do not identify interval switch totals or runnable delay',
            'Control roots need not have identical auxiliary phase counts',
            'No playing-strength or admission claim; every previous negative attempt remains negative'],
        new_games=0,training=False,larger_data_deferred=True,external_testing='paused',benchmark_authorized_to_launch=False)
