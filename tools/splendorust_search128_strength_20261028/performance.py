"""Descriptive trusted-arrival and existing-boundary reports; no model calls."""
from collections import Counter
import math
from tools.splendorust_boundary_timing_20261025 import analyze as boundaries

ROLES=('challenger','opponent')
QUOTAS={'challenger':128,'opponent':64}
MODES={'challenger':'research','opponent':'frozen'}
PLACEMENT_MODE='isolated'
COUNTER_DEFINITIONS={
    'completed_over_700ms':'strictly greater than0.700 seconds',
    'completed_over_900ms':'strictly greater than0.900 seconds',
    'completed_over_one_second':'strictly greater than1.000 seconds',
    'completed_at_or_above_one_second':'greater than or equal to1.000 seconds'}

def require(value,message):
    if not value:raise ValueError('SEARCH128_PERFORMANCE: '+message)

def _distribution(values):
    require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values),'Finite nonnegative timing')
    ordered=sorted(values)
    def quantile(q):
        if not ordered:return None
        p=(len(ordered)-1)*q;lo,hi=math.floor(p),math.ceil(p)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
    return dict(count=len(values),mean=sum(values)/len(values) if values else None,
        min=ordered[0] if ordered else None,median=quantile(.5),p95=quantile(.95),max=ordered[-1] if ordered else None)

def _thresholds(values):
    return dict(completed_over_700ms=sum(x>.7 for x in values),completed_over_900ms=sum(x>.9 for x in values),
        completed_over_one_second=sum(x>1. for x in values),completed_at_or_above_one_second=sum(x>=1. for x in values))

def _require_isolated(row):
    require(row.get('placement_mode')==PLACEMENT_MODE and not row.get('placement_error') and
            not row.get('profile_capture_error'),'Qualified placement and evidence capture')
    role=row.get('player_role')
    require(role in ROLES and row.get('arm')=='search' and row.get('mode')==MODES[role] and
            type(row.get('quota')) is int and row['quota']==QUOTAS[role],'Exact role/quota identity')
    require(type(row.get('seat')) is int and row['seat'] in (0,1),'Actual player seat')

def _clock(receipt):
    from .scientific import check_clock
    return check_clock(receipt)

def _player_summary(receipts,role):
    completed=[r['receipt_elapsed_seconds'] for r in receipts if r['status']=='ok']
    arrived=[r['receipt_elapsed_seconds'] for r in receipts if r['receipt_elapsed_seconds'] is not None and r['receipt_elapsed_seconds']>=0]
    fields=('request_cpu_seconds','rss_bytes','peak_sampled_rss_bytes')
    return dict(scientific_arm='search',quota=QUOTAS[role],world_pool=3,mode=MODES[role],
        attempted_decisions=len(receipts),completed_decisions=len(completed),unfinished_decisions=len(receipts)-len(completed),
        statuses=dict(Counter(r['status'] for r in receipts)),completed_arrival_seconds=_distribution(completed),
        all_observed_arrival_seconds=_distribution(arrived),
        parent_processing_through_validation_seconds=_distribution([r['elapsed_seconds'] for r in receipts if r['status']=='ok']),
        missing_arrival_timestamps=sum(r['receipt_elapsed_seconds'] is None for r in receipts),
        negative_arrival_timestamps=sum(r['receipt_elapsed_seconds'] is not None and r['receipt_elapsed_seconds']<0 for r in receipts),
        **_thresholds(completed),
        decision_cpu_seconds=_distribution([r['telemetry'][fields[0]] for r in receipts if fields[0] in r.get('telemetry',{})]),
        sampled_tree_rss_bytes=_distribution([r['telemetry'][fields[1]] for r in receipts if fields[1] in r.get('telemetry',{})]),
        cumulative_sampled_tree_peak_rss_bytes=_distribution([r['telemetry'][fields[2]] for r in receipts if fields[2] in r.get('telemetry',{})]),
        missing_telemetry={f:sum(f not in r.get('telemetry',{}) for r in receipts) for f in fields})

def performance(rows,games=None):
    games=[] if games is None else games
    require(all(r.get('stage') in ('operational','main') for r in rows) and
            all(g.get('stage') in ('operational','main') for g in games),'Live stages only')
    for row in rows:_require_isolated(row);_clock(row['clock'])
    players={role:_player_summary([r['clock'] for r in rows if r['player_role']==role],role) for role in ROLES}
    groups=[]
    for stage in ('operational','main'):
        for role in ROLES:
            for seat in (0,1):
                subset=[r['clock'] for r in rows if (r['stage'],r['player_role'],r['seat'])==(stage,role,seat)]
                groups.append(dict(stage=stage,player_role=role,seat=seat,**_player_summary(subset,role)))
    natural=[g for g in games if g['reason']=='natural']
    unfinished=[dict(**{k:r[k] for k in ('stage','pair_index','game_index','ply','player_role','seat','mode','quota')},
        request_id=r['clock']['request_id'],status=r['clock']['status'],receipt_elapsed_seconds=r['clock']['receipt_elapsed_seconds'],
        elapsed_seconds=r['clock']['elapsed_seconds'],completed=False) for r in rows if r['clock']['status']!='ok']
    return dict(schema='splendorust_search128_strength_performance_v1',players=players,by_stage_and_seat=groups,
        timing_clock='clock.receipt_elapsed_seconds: request start to trusted complete-line arrival before parent logging/decoding/validation',
        parent_processing_clock='clock.elapsed_seconds additionally includes parent processing through validation; later evidence writes are excluded',
        scope='collected_live_prefix_only',games=len(games),rows=len(rows),no_strength_claim=True,
        failure_rate_confidence_interval=None,placement_mode=PLACEMENT_MODE,
        descriptive_timing_threshold_seconds=1.,descriptive_timing_thresholds_seconds=[.7,.9,1.],
        descriptive_thresholds_change_completion_or_stopping=False,research_outer_seconds=10.,research_internal_seconds=9.,
        one_second_threshold_is_descriptive_only=True,one_second_changes_completion_or_stopping=False,
        timing_counter_definitions=COUNTER_DEFINITIONS,zero_failures_is_not_zero_risk=True,
        stage_counts=dict(Counter(g['stage'] for g in games)),natural_games=len(natural),
        natural_game_length_plies=_distribution([g['plies'] for g in natural]),
        natural_game_length_plies_by_challenger_seat={str(s):_distribution([g['plies'] for g in natural if g['challenger_seat']==s]) for s in (0,1)},
        forfeits=[{k:g.get(k) for k in ('stage','pair_index','game_index','challenger_seat','loser','winner','failed_role','failed_seat','failure','plies')} for g in games if g['reason']=='forfeit'],
        unfinished_requests=unfinished,memory_scope='sampled worker plus owned descendants; no continuous-peak guarantee',
        cpu_scope='sampled worker plus owned descendants; excludes supervising parent',
        log_integrity_scope='verified separately by the runner; not inferred from timing receipts')

summarise_performance=performance

def boundary_row(row):
    _require_isolated(row)
    require(row.get('observation_mode')=='profiled','Existing trusted boundary observer required')
    return {**row,'mode':'profiled'}

def validate_boundary_row(row):
    from .transport import validate_profiles
    boundary_row(row)
    return validate_profiles(row['parent_profile'],row['child_profile'],row['clock'],
        dict(player='search',observer=row['observer']),row['mode'])

def summarize_boundary_rows(rows,*,boundary=lambda:None):
    from .scientific import classify_failure
    grouped={};seat_groups={};failures=[]
    def group(mapping,key):
        return mapping.setdefault(key,dict(attempted=0,completed=0,statuses=Counter(),arrivals=[],metrics={},missing_parent=0,missing_child=0))
    def add(g,row,checked):
        g['attempted']+=1;g['statuses'][row['clock']['status']]+=1
        g['missing_parent']+=int(checked.get('missing_parent_profile',False));g['missing_child']+=int(checked.get('missing_child_profile',False))
        if not row['completed']:return
        g['completed']+=1;g['arrivals'].append(row['clock']['receipt_elapsed_seconds'])
        def append(prefix,value):
            for name,item in value.items():
                field=prefix+'.'+name if prefix else name
                if type(item) is dict:append(field,item)
                elif type(item) is int:g['metrics'].setdefault(field,[]).append(item)
        append('',checked)
    for row in rows:
        boundary();require(row.get('stage') in ('operational','main'),'Live boundary stage');_clock(row['clock'])
        checked=validate_boundary_row(row);role=row['player_role']
        require(row['completed'] is (row['clock']['status']=='ok'),'Completion and trusted receipt agree')
        add(group(grouped,(row['stage'],role)),row,checked)
        add(group(seat_groups,(row['stage'],role,row['seat'])),row,checked)
        if not row['completed']:
            failures.append(dict(**{k:row[k] for k in ('stage','pair_index','game_index','ply','player_role','seat','mode','quota')},
                request_id=row['clock']['request_id'],classification=classify_failure(row['clock']),
                receipt_elapsed_seconds=row['clock'].get('receipt_elapsed_seconds'),boundary_evidence=checked,
                partial_or_missing_evidence_not_promoted=True))
    def summarize(mapping):
        output=[]
        for key,g in sorted(mapping.items()):
            stage,role=key[:2]
            value=dict(stage=stage,player_role=role,quota=QUOTAS[role],world_pool=3,mode=MODES[role],
                observation_mode='profiled',attempted_decisions=g['attempted'],completed_decisions=g['completed'],
                statuses=dict(g['statuses']),missing_parent_profiles=g['missing_parent'],missing_child_profiles=g['missing_child'],
                completed_arrival_seconds=_distribution(g['arrivals']),**_thresholds(g['arrivals']),
                completed_boundary_metrics={name:dict(unit='nanoseconds' if name.endswith('_ns') else 'count',
                    distribution=boundaries._distribution(values)) for name,values in sorted(g['metrics'].items())})
            if len(key)==3:value['seat']=key[2]
            output.append(value)
        return output
    return dict(schema='splendorust_search128_strength_boundary_v1',rows=len(rows),groups=summarize(grouped),
        by_stage_and_seat=summarize(seat_groups),failures=failures,observed_faults=len(failures),
        all_collected_decisions_completed=not failures,complete_planned_collection_inferred=False,deadline_adjustment_seconds=0.,
        descriptive_timing_thresholds_seconds=[.7,.9,1.],timing_counter_definitions=COUNTER_DEFINITIONS,
        descriptive_thresholds_change_completion_or_stopping=False,causal_bottleneck_proven=False,no_strength_claim=True,
        interpretation=['Trusted complete-line arrival determines completion, not parent validation-finish elapsed time.',
            'Missing failure sidecars remain missing and never establish a completed measurement.',
            'Boundary intervals describe elapsed wall time, not inference, CPU or scheduling costs; signed cross-process gaps remain.',
            'Operational hard-guard faults stop; main administrative losses continue the complete fixed schedule.',
            'Late completed moves are valid; no counterfactual one-second forfeits or deployment qualification are inferred.'])
