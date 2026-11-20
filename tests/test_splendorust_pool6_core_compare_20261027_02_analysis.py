"""Lean OFF-only comparison and unchanged engineering boundary regressions."""
from copy import deepcopy
import pytest
from tools.splendorust_pool6_core_compare_20261027_02 import analysis,design,common as c,host_observer as h


def disabled_receipt():
    return dict(enabled=False,status='disabled',errors=[],samples=[],sample_count=0,missed_intervals=0,
        context_before=None,context_after=None,context_switch_deltas=None,summary=None,
        summary_absence_reason='disabled',host_cpu_availability=h.host_cpu_availability([]))


def fixture_rows():
    panel=dict(positions=[dict(position_index=i,case_id=f'case{i}',
        selection='observed_slow_target' if i<2 else 'fixed_structural_control',
        observer={},observer_sha256=c.digest({}),before_snapshot={},
        expectations={route:dict(expected_record={'path':'synthetic-only'}) for route in ('frozen','research')}) for i in range(4)])
    matrix=design.build_matrix(panel)
    rows=[dict(spec=deepcopy(spec),route=spec['route'],arm_id=spec['arm_id'],logical_cpu=spec['logical_cpu'],completed=True,
        parent_profile=dict(request_start_ns=0,receipt_arrival_ns=100),host_observation=disabled_receipt(),
        clock=dict(status='ok',receipt_elapsed_seconds=.4,result=dict(diagnostics=dict(inference_calls=65,phases=[]))))
        for spec in matrix['requests']]
    return panel,matrix,rows


def test_full_lean_counts_pairs_and_explicit_absence_of_host_measurements():
    panel,matrix,rows=fixture_rows();result=analysis.analyze(rows,panel,matrix,True)
    assert result['complete'] and result['attempted']==192 and result['overhead_trial_performed'] is False
    assert 'overhead_gate' not in result and len(result['cells'])==16 and len(result['groups'])==4
    assert len(result['paired_core_observations'])==96
    assert all(x['complete'] and x['cpu2_minus_cpu0_receipt_seconds']==0 for x in result['paired_core_observations'])
    assert {tuple(x['core_order']) for x in result['paired_core_observations']}=={(0,2),(2,0)}
    counts=result['extra_host_sampler']
    assert counts['enabled_windows']==0 and counts['disabled_windows']==192
    assert counts['host_counter_samples']==0 and counts['host_counter_measurements_available'] is False
    assert counts['host_cpu_busy_fraction'] is None and counts['accepted_disabled_lifecycle_windows']==192
    assert result['engineering_gate']['eligible'] is True and result['engineering_gate']['advantage'] is False
    assert all(x['cpu0']['n']==x['cpu2']['n']==48 for x in result['engineering_gate']['per_player'].values())
    assert not result['engineering_gate']['automatic_promotion'] and not result['benchmark_authorized_to_launch']


def values(cpu0=.4,cpu2=.4):
    return [dict(stage='diagnostic',route=route,logical_cpu=core,status='ok',receipt_seconds=cpu0 if core==0 else cpu2)
        for route in ('frozen','research') for core in (0,2) for _ in range(48)]


def test_exact_700ms_fails_engineering_but_is_not_a_timeout():
    result=analysis.engineering_gate(values(.7,.7),True)
    assert result['complete'] and result['eligible'] is False and result['advantage'] is False
    assert all(x['cpu2_margin_crossings']==48 for x in result['per_player'].values())


@pytest.mark.parametrize('kind',['mean','p95'])
def test_each_player_nonregression_is_required_even_when_other_player_improves(kind):
    rows=values(.5,.45)
    changed=[r for r in rows if r['route']=='research' and r['logical_cpu']==2]
    for row in changed:row['receipt_seconds']=.526 if kind=='mean' else .49
    if kind=='p95':
        for row in changed[:3]:row['receipt_seconds']=.526
    result=analysis.engineering_gate(rows,True)
    assert result['per_player']['frozen']['nonregression']
    assert not result['per_player']['research']['nonregression']
    assert result['eligible'] is False and result['advantage'] is False


def test_fixed_advantage_threshold_and_crossing_alternative():
    assert not analysis.engineering_gate(values(.5,.476),True)['advantage']
    assert analysis.engineering_gate(values(.5,.475),True)['advantage']
    rows=values(.5,.5);next(r for r in rows if r['logical_cpu']==0)['receipt_seconds']=.7
    result=analysis.engineering_gate(rows,True)
    assert result['eligible'] and result['advantage'] and result['recommendation']=='prepare_fresh_admission_checks'


@pytest.mark.parametrize('count,complete',[(0,False),(191,False),(191,True),(192,False)])
def test_partial_engineering_has_no_verdict(count,complete):
    result=analysis.engineering_gate(values()[:count],complete)
    assert result['eligible'] is result['advantage'] is None
    assert result['recommendation']=='not_evaluable_incomplete'


def test_timeout_prefix_preserves_missing_pair_and_has_no_engineering_verdict():
    panel,matrix,rows=fixture_rows();rows=rows[:3]
    rows[-1]['completed']=False;rows[-1]['clock'].update(status='timeout',receipt_elapsed_seconds=None)
    result=analysis.analyze(rows,panel,matrix,False)
    assert result['engineering_gate']['eligible'] is result['engineering_gate']['advantage'] is None
    assert result['attempted']==3 and sum(g['deadline_failures'] for g in result['groups'])==1
    pair=result['paired_core_observations'][-1]
    assert pair['complete'] is False and pair['cpu2_minus_cpu0_receipt_seconds'] is None
    assert result['extra_host_sampler']['disabled_windows']==3


def test_verified_failed_disabled_lifecycle_remains_incomplete_not_accepted_measurement():
    panel,matrix,rows=fixture_rows();rows=rows[:2]
    rows[-1]['host_observation'].update(status='observer_failed',errors=['placement read failed'],
        summary_absence_reason='observer_failed',host_cpu_availability=None)
    result=analysis.analyze(rows,panel,matrix,False)
    assert result['extra_host_sampler']['failed_disabled_lifecycle_windows']==1
    assert result['extra_host_sampler']['host_counter_measurements_available'] is False
    assert result['engineering_gate']['eligible'] is None


@pytest.mark.parametrize('field',['enabled','samples','sample_count','missed_intervals','context_before','context_after','context_switch_deltas',
    'summary','summary_absence_reason','host_cpu_availability','status','errors'])
def test_lean_analysis_rejects_counter_capture_or_false_disabled_metadata(field):
    panel,matrix,rows=fixture_rows();value=rows[0]['host_observation']
    value[field]={'enabled':True,'samples':[{}],'sample_count':1,'missed_intervals':1,'context_before':{},'context_after':{},
        'context_switch_deltas':[],'summary':{},'summary_absence_reason':'observed',
        'host_cpu_availability':{'available':True},'status':'observed','errors':['unreported']}[field]
    with pytest.raises(ValueError):analysis.analyze(rows,panel,matrix,True)


@pytest.mark.parametrize('kind',['enabled_spec','wrong_arm','incomplete_declared_complete'])
def test_analysis_requires_exact_off_only_arm_and_complete_inventory(kind):
    panel,matrix,rows=fixture_rows()
    if kind=='enabled_spec':matrix['requests'][0]['observer_enabled']=True
    elif kind=='wrong_arm':rows[0]['arm_id']='research_cpu2'
    else:rows.pop()
    with pytest.raises(ValueError):analysis.analyze(rows,panel,matrix,True)
