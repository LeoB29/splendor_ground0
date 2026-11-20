"""Prospective absolute isolated qualification; retained original controls."""
from copy import deepcopy
import math
from pathlib import Path
import statistics

from . import common as c
from tools import future_training_rules_20261021 as corrected
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.splendorust_search_scale_20261024 import audit as science

SOURCE = c.ROOT / 'outputs/splendorust_live_boundary_20261025_01'
SOURCE_FREEZE = '50ca8363a364b06dbf452778b90aa27c5ea763d8f2d76d2c4efab8bca582ec6b'
QUOTAS = (64, 32)
PLACEMENTS = ('original', 'isolated')
FOCUS = [('operational',4,16), ('main',3,28), ('main',15,34),
    ('main',36,26), ('main',44,6), ('main',46,21),
    ('main',72,16), ('main',72,18), ('main',72,20), ('main',72,22), ('main',72,28)]
CONTROLS = [(stage, game, ply-1) for stage, game, ply in FOCUS] + [('main',72,0), ('main',71,52)]


def row_path(stage, game, ply):
    return SOURCE / f'games/{stage}/p{game//2:03d}_g{game}/moves/{ply:04d}.json'


def make_schedule(fixtures):
    c.require(len(fixtures)==24 and len({r['fixture_id'] for r in fixtures})==24,
              'Exactly24 unique saved positions required')
    result=[]
    block_index=0
    for quota in QUOTAS:
        for pass_index in range(2):
            indices=list(range(24))
            modes=list(PLACEMENTS)
            if pass_index:
                indices.reverse();modes.reverse()
            for placement in modes:
                for ordinal,index in enumerate(indices):
                    result.append(dict(index=len(result),block_index=block_index,
                        ordinal=ordinal,fixture_id=fixtures[index]['fixture_id'],fixture_index=index,
                        quota=quota,player_role='challenger' if quota==64 else 'opponent',
                        placement_mode=placement,pass_index=pass_index))
                block_index+=1
    return result


def validate_plan(plan):
    c.require(plan['schema']=='splendorust_supervisor_qualification_panel_v1', 'Panel schema')
    c.require(plan['source_freeze_sha256']==SOURCE_FREEZE, 'Source freeze identity')
    fixtures=plan['fixtures']
    c.require(len(fixtures)==24 and [tuple(x['source_identity']) for x in fixtures]==FOCUS+CONTROLS,
              'Fixed focus/control selection differs')
    c.require(sum(x['kind']=='failed_saved_position' for x in fixtures)==1 and
              sum(x['kind']=='slow_completed' for x in fixtures)==10 and
              sum(x['kind']=='control' for x in fixtures)==13, 'Fixed selection categories')
    c.require(plan['schedule']==make_schedule(fixtures), 'Balanced eight-block schedule differs')
    c.require(len(plan['schedule'])==192, 'Exactly192 timed requests')
    c.require(plan['isolated_historical_comparisons_required']==46 and
        plan['control_historical_comparisons_maximum']==46 and plan['agreement_cells_expected']==48 and
        plan['isolated_agreements_per_cell']==2 and plan['control_timeout_policy']=='retain_once_continue_next_unattempted_root' and
        plan['relative_comparison_is_admission_gate'] is False, 'Prospective qualification policy differs')
    c.require(sum(x['historical'] is not None for x in fixtures)==23 and
        all(x['historical'] is None if x['kind']=='failed_saved_position' else
            type(x['historical']) is dict and x['historical']['quota']==x['original_quota'] in QUOTAS
            for x in fixtures), 'Historical target inventory differs')
    return plan


def build_plan():
    runtime=corrected.verify_runtime_sources()
    c.require(c.file_hash(SOURCE/'freeze.json')==SOURCE_FREEZE, 'Original live freeze changed')
    completion=c.read(SOURCE/'completion.json');execution=c.read(SOURCE/'execution.json')
    verification=c.read(SOURCE/'verification_summary.json')
    c.require(completion['status']=='captured_fault_verified' and
        execution['status']=='completed' and execution['exit_code']==0 and not execution['cleanup_errors'] and
        verification['status']=='passed', 'Verified original failure prefix required')
    seal=c.read(SOURCE/'completed_evidence_seal.json')
    index={item['path']:item for item in seal['artifacts']}
    c.require(len(index)==len(seal['artifacts']), 'Duplicate original seal entries')
    sources={}
    for name in ('freeze.json','completion.json','execution.json','verification_summary.json',
                 'completed_evidence_seal.json','collection_stop.json'):
        item=c.artifact(SOURCE/name);sources[item['path']]=item
    fixtures=[]
    for ordinal,(stage,game,ply) in enumerate(FOCUS+CONTROLS):
        path=row_path(stage,game,ply);item=c.artifact(path)
        c.require(item==index[item['path']], 'Selected saved move differs from original seal')
        sources[item['path']]=item;row=c.read(path)
        c.require((row['stage'],row['game_index'],row['ply'])==(stage,game,ply), 'Saved identity')
        snapshot=row['before'];state=corrected.decode_training_snapshot(snapshot)
        observer=rules.observation_payload(state)
        c.require(observer==row['observer'], 'Strict corrected snapshot/observer mismatch')
        ledger.check_observer(snapshot,observer)
        for name in ('worker_clock_file','child_profile_file','parent_profile_file'):
            source=c.artifact(c.ROOT/row[name]);c.require(source==index[source['path']], 'Selected raw source differs')
            sources[source['path']]=source
        c.require(c.read(c.ROOT/row['worker_clock_file'])==row['clock'], 'Saved clock copy differs')
        history=None
        if row['completed']:
            c.require(row['clock']['status']=='ok' and row['clock']['receipt_elapsed_seconds']<=1.,
                'Historical accepted completion required')
            science.validate_result(row['clock']['result'],observer,'search',expected_quota=row['quota'])
            science.audit_decision(observer,row['action'],row['trace'],'search',
                before=snapshot,after=row['after'],expected_quota=row['quota'])
            history=dict(quota=row['quota'],result=deepcopy(row['clock']['result']),
                action=deepcopy(row['action']),trace=deepcopy(row['trace']),after=deepcopy(row['after']))
        else:
            c.require((stage,game,ply)==('main',72,28) and row['quota']==64 and
                row['clock']['status']=='worker_error' and row['clock']['result'] is None,
                'Only the disclosed failed saved position has no target')
        kind='control' if ordinal>=11 else 'failed_saved_position' if ordinal==10 else 'slow_completed'
        if kind=='slow_completed':
            c.require(row['completed'] and row['clock']['receipt_elapsed_seconds']>.7 and row['quota']==64,
                'Ten original slow completed moves')
        fixtures.append(dict(fixture_id=f'{stage}_g{game}_p{ply}',kind=kind,
            source_identity=[stage,game,ply],source=item,snapshot=deepcopy(snapshot),observer=observer,
            original_quota=row['quota'],historical=history))
    plan=dict(schema='splendorust_supervisor_qualification_panel_v1',
        source_freeze_sha256=SOURCE_FREEZE,fixtures=fixtures,schedule=make_schedule(fixtures),
        sources=sorted(sources.values(),key=lambda x:x['path']),source_guard=runtime,
        selection='All10slow completions plus failure; preceding11opposite-role controls; earlyg72p0 and latest64moveg71p52.',
        original_failed_attempt_unchanged=True,failed_saved_root_is_new_diagnostic_not_replacement=True,
        isolated_historical_comparisons_required=46,control_historical_comparisons_maximum=46,
        historical_comparisons_maximum=92,agreement_cells_expected=48,isolated_agreements_per_cell=2,
        control_timeout_policy='retain_once_continue_next_unattempted_root',
        relative_comparison_is_admission_gate=False,
        primary_question='Do both isolated players meet the absolute deadline/margin with exact science?',
        independent_population_sample=False,paired_order='Eight closed-worker blocks; reverse placement and root order on pass2')
    return validate_plan(plan)


def distribution(values):
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values),
              'Finite nonnegative timing values required')
    if not values:return dict(count=0,mean=None,p95=None,max=None)
    values=sorted(values);at=(len(values)-1)*.95;low=math.floor(at);high=math.ceil(at)
    return dict(count=len(values),mean=statistics.mean(values),p95=values[low]+(values[high]-values[low])*(at-low),max=max(values))


def timing_gate(rows):
    """Absolute admission; controls retain their failures without invented times.

    Full scientific/raw/placement validation is required independently by
    assessment and the runner. Relative timing is a secondary description only.
    """
    from .assessment import control_timeout
    fixtures=[dict(fixture_id=f'{s}_g{g}_p{p}') for s,g,p in FOCUS+CONTROLS]
    expected=make_schedule(fixtures)
    complete=(len(rows)==192 and all(all(type(row.get(key)) is type(value) and row[key]==value
        for key,value in spec.items()) for row,spec in zip(rows,expected,strict=False)))
    valid_clocks=True
    for row in rows:
        science.check_clock(row['clock'])
        c.require(type(row['completed']) is bool and row['completed']==(row['clock']['status']=='ok'),
            'Completed flag differs from original clock')
        valid_clocks &= row['clock']['deadline_seconds']==1.
        if not row['completed'] and row['placement_mode']=='original':control_timeout(row)
    per_quota={}
    for quota in QUOTAS:
        groups={mode:[r for r in rows if r['quota']==quota and r['placement_mode']==mode]
                for mode in PLACEMENTS}
        timings={mode:distribution([r['clock']['receipt_elapsed_seconds'] for r in group
                   if r['completed'] and r['clock']['status']=='ok']) for mode,group in groups.items()}
        isolated_full=len(groups['isolated'])==48 and timings['isolated']['count']==48
        controls_attempted=len(groups['original'])==48
        controls_complete=controls_attempted and timings['original']['count']==48
        timeout_count=sum(not row['completed'] for row in groups['original'])
        conditions=dict(isolated_complete_inventory=isolated_full,controls_all_attempted=controls_attempted,
            isolated_under_one_second=isolated_full and timings['isolated']['max']<=1.,
            isolated_under_700ms=isolated_full and timings['isolated']['max']<.7)
        if timeout_count:
            relative=dict(status='not_estimable_with_control_timeouts',mean_nonregression=None,
                p95_nonregression=None,mean_ratio=None,p95_difference_seconds=None)
        elif not controls_complete or not isolated_full:
            relative=dict(status='not_estimable_without_full_completed_panel',mean_nonregression=None,
                p95_nonregression=None,mean_ratio=None,p95_difference_seconds=None)
        else:
            relative=dict(status='complete_paired_descriptive_comparison',
                mean_nonregression=timings['isolated']['mean']<=1.05*timings['original']['mean'],
                p95_nonregression=timings['isolated']['p95']<=timings['original']['p95']+.025,
                mean_ratio=None if timings['original']['mean']==0 else timings['isolated']['mean']/timings['original']['mean'],
                p95_difference_seconds=timings['isolated']['p95']-timings['original']['p95'])
        relative.update(is_admission_gate=False,mean_ratio_descriptive_threshold=1.05,
            p95_added_seconds_descriptive_threshold=.025,no_timeout_time_imputation=True)
        per_quota[str(quota)]=dict(qualified=all(conditions.values()),conditions=conditions,distributions=timings,
            distribution_scope={'original':'completed_controls_only_conditional_on_success',
                                'isolated':'completed_isolated_requests_only'},
            original_attempts=len(groups['original']),control_timeout_count=timeout_count,
            isolated_attempts=len(groups['isolated']),isolated_failure_count=sum(not r['completed'] for r in groups['isolated']),
            relative_comparison=relative)
    qualified=complete and valid_clocks and all(x['qualified'] for x in per_quota.values())
    return dict(status='passed' if qualified else 'not_qualified',qualified=qualified,
        complete_inventory=complete,per_quota=per_quota,attempts=len(rows),
        isolated_completed=sum(r['completed'] and r['placement_mode']=='isolated' for r in rows),
        retained_control_timeouts=sum(q['control_timeout_count'] for q in per_quota.values()),
        margin_seconds=.7,seconds_per_move=1.,internal_seconds=.9,
        primary_gate='both_isolated_players_absolute_correctness_and_timing',
        relative_comparison_is_admission_gate=False,
        independent_scientific_and_raw_verification_required=True,
        engineering_margin_is_not_timeout=True,no_causal_fix_claim=True)
