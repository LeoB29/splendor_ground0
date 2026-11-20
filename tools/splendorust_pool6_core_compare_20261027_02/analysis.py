"""OFF-only fixed-core latency analysis; no additional host counters are collected."""
import math
import statistics
from . import common as c
from tools.splendorust_pool6_repeat_timing_20261027.analysis import measurements as timing_measurements


def distribution(values):
    values=sorted(values)
    if not values:return dict(n=0,minimum=None,median=None,p95=None,maximum=None,mean=None)
    return dict(n=len(values),minimum=values[0],median=statistics.median(values),
        p95=values[math.ceil(.95*len(values))-1],maximum=values[-1],mean=statistics.mean(values))


def host_summary(row):
    """Report intentional absence, never convert a disabled lifecycle into measurements."""
    from .host_observer import host_cpu_availability
    value=row['host_observation']
    c.require(row['spec']['observer_enabled'] is False and value['enabled'] is False,
              'Additional host sampler must be OFF')
    c.require(value['status'] in ('disabled','observer_failed'),'No enabled observation in lean comparison')
    c.require(value['samples']==[] and value['sample_count']==0 and value['missed_intervals']==0 and
        all(value.get(key) is None for key in ('context_before','context_after','context_switch_deltas','summary')),
        'Disabled lifecycle cannot contain host counters')
    c.require(type(value['errors']) is list and bool(value['errors'])==(value['status']=='observer_failed'),
              'Original disabled lifecycle status/error binding')
    if value['status']=='disabled':
        c.require(value['summary_absence_reason']=='disabled' and
            value['host_cpu_availability']==host_cpu_availability([]),'Explicit disabled counter absence')
    return dict(enabled=False,status=value['status'],measurements_available=False,sample_count=0,
        host_cpu_available=False,host_cpu_busy_fraction=None,
        absence_reason='extra_host_sampler_disabled',lifecycle_accepted=value['status']=='disabled',
        retained_measurements='Original response and boundary clocks plus existing resource/placement checks only')


def measurements(row):
    value=timing_measurements(row);spec=row['spec']
    value.update(stage=spec['stage'],route=spec['route'],position_index=spec['position_index'],
        arm_id=spec['arm_id'],logical_cpu=spec['logical_cpu'],core_pair_index=spec['core_pair_index'],
        case_id=spec['case_id'],observer_enabled=False,host_summary=host_summary(row),
        host_observation_file=row.get('host_observation_file'))
    return value


def availability_summary(values):
    return dict(attempted_windows=len(values),enabled_windows=0,disabled_windows=len(values),
        accepted_disabled_lifecycle_windows=sum(v['host_summary']['lifecycle_accepted'] for v in values),
        failed_disabled_lifecycle_windows=sum(not v['host_summary']['lifecycle_accepted'] for v in values),
        host_counter_samples=0,host_counter_measurements_available=False,
        host_cpu_busy_fraction=None,absence_reason='extra_host_sampler_disabled',
        definition='Disabled is an intentional absence of host measurements, not a zero utilization estimate')


def summarize(values):
    return dict(attempted=len(values),completed=sum(x['status']=='ok' for x in values),
        margin_crossings=sum(x['margin_crossing'] for x in values),
        failed_requests=sum(x['failed_request'] for x in values),deadline_failures=sum(x['deadline_failed'] for x in values),
        receipt=distribution([x['receipt_seconds'] for x in values if x['receipt_seconds'] is not None]),
        handler=distribution([x['handler_seconds'] for x in values if x['handler_seconds'] is not None]),
        emission=distribution([x['emission_seconds'] for x in values if x['emission_seconds'] is not None]),
        extra_host_sampler=availability_summary(values))


def engineering_gate(values,complete):
    """Fixed descriptive criterion; it never launches admission or changes a player."""
    from .design import engineering_contract
    contract=engineering_contract();full=[v for v in values if v['stage']=='diagnostic']
    ready=complete and len(full)==192 and all(v['status']=='ok' for v in full)
    groups={}
    for route in ('frozen','research'):
        by_core={core:[v['receipt_seconds'] for v in full if v['route']==route and v['logical_cpu']==core and v['status']=='ok'] for core in (0,2)}
        a,b=distribution(by_core[0]),distribution(by_core[2])
        paired_ready=ready and a['n']==b['n']==48
        nonregression=paired_ready and b['mean']<=contract['maximum_per_player_mean_ratio']*a['mean'] and b['p95']<=a['p95']+contract['maximum_per_player_p95_added_seconds']
        groups[route]=dict(cpu0=a,cpu2=b,complete=paired_ready,nonregression=bool(nonregression),
            cpu0_margin_crossings=sum(x>=.7 for x in by_core[0]),cpu2_margin_crossings=sum(x>=.7 for x in by_core[2]),
            cpu2_all_below_margin=paired_ready and all(x<.7 for x in by_core[2]),
            cpu2_p95_gain_seconds=a['p95']-b['p95'] if a['n'] and b['n'] else None)
    eligible=ready and all(g['nonregression'] and g['cpu2_all_below_margin'] for g in groups.values())
    advantage=eligible and (any(g['cpu0_margin_crossings']>0 for g in groups.values()) or
        any(g['cpu2_p95_gain_seconds']>=contract['minimum_p95_gain_seconds'] for g in groups.values()))
    return dict(complete=ready,eligible=bool(eligible) if ready else None,advantage=bool(advantage) if ready else None,per_player=groups,contract=contract,
        recommendation='prepare_fresh_admission_checks' if advantage else 'no_core_change_recommendation' if ready else 'not_evaluable_incomplete',
        automatic_promotion=False,new_games_authorized=False,statistical_significance_claim=False)


def analyze(records,panel,matrix,complete,boundary=lambda:None):
    from .runner import ARM_SPECS,arm_id
    c.require(type(complete) is bool and len(matrix['requests'])==192 and len(records)<=192,
              'Explicit bounded lean comparison completion')
    c.require(all(s['stage']=='diagnostic' and s['observer_enabled'] is False for s in matrix['requests']),
              'Exactly OFF-only diagnostic specifications')
    if complete:c.require(len(records)==192 and all(r['clock']['status']=='ok' for r in records),
                          'Complete diagnostic inventory')
    values=[]
    for row,spec in zip(records,matrix['requests']):
        boundary();c.require(row['spec']==spec and row['arm_id']==spec['arm_id']==arm_id(row['route'],row['logical_cpu']),
                            'Analysis ordered fixed-arm specification')
        values.append(measurements(row))
    if complete:c.require(all(v['host_summary']['lifecycle_accepted'] for v in values),
                          'Complete comparison requires verified disabled lifecycles')
    groups=[];cells=[]
    for route,core in ARM_SPECS:
        assigned=arm_id(route,core)
        subset=[v for v in values if v['arm_id']==assigned]
        groups.append(dict(stage='diagnostic',route=route,logical_cpu=core,arm_id=assigned,
                           observer_enabled=False,**summarize(subset)))
    for position in panel['positions']:
        for route,core in ARM_SPECS:
            assigned=arm_id(route,core)
            subset=[v for v in values if v['position_index']==position['position_index'] and v['arm_id']==assigned]
            cells.append(dict(case_id=position['case_id'],selection=position['selection'],route=route,logical_cpu=core,arm_id=assigned,
                historical_target_available=position['expectations'][route].get('expected_record') is not None,
                observations=subset,**summarize(subset)))
    pairs=[]
    for index in sorted({v['core_pair_index'] for v in values}):
        selected=[v for v in values if v['core_pair_index']==index]
        c.require(1<=len(selected)<=2 and len({v['logical_cpu'] for v in selected})==len(selected),'No duplicated core-pair observation')
        c.require(len({(v['route'],v['position_index'],v['stage'],v['round']) for v in selected})==1,'Core pair scientific cell differs')
        by_core={v['logical_cpu']:v for v in selected};paired=len(selected)==2 and all(v['status']=='ok' for v in selected)
        item=dict(core_pair_index=index,stage=selected[0]['stage'],route=selected[0]['route'],case_id=selected[0]['case_id'],round=selected[0]['round'],
            complete=paired,ordered_sequences=[v['sequence'] for v in selected],core_order=[v['logical_cpu'] for v in selected])
        for label,key in (('receipt','receipt_seconds'),('handler','handler_seconds'),('emission','emission_seconds')):
            item['cpu2_minus_cpu0_'+label+'_seconds']=by_core[2][key]-by_core[0][key] if paired and all(by_core[k][key] is not None for k in (0,2)) else None
        pairs.append(item)
    return dict(schema='pool6_core_compare_lean_descriptive_analysis_v1',complete=complete,attempted=len(records),maximum_planned=192,
        overhead_trial_performed=False,extra_host_sampler=availability_summary(values),
        engineering_gate=engineering_gate(values,complete),groups=groups,cells=cells,paired_core_observations=pairs,
        distribution_definition='p95 nearest rank; no discarded warmups, trimming or replacements',
        independent_trial_assumption=False,confidence_intervals=None,
        limitations=['Selected past slow roots and dependent repeats do not estimate live-game timeout risk',
            'Four-GiB headroom is a disclosed diagnostic context, not a randomized host-load intervention',
            'One persistent process instance per player/core; instance and core effects cannot be separated',
            'CPU affinity is not exclusive reservation against external applications',
            'Additional host counters are disabled; timing differences do not identify scheduling or paging causes',
            'New prospective OFF-only comparison leaves the prior negative overhead result unchanged',
            'No playing-strength or admission claim; all previous negative admissions remain negative'],
        new_games=0,training=False,larger_data_deferred=True,external_testing='paused',benchmark_authorized_to_launch=False)
