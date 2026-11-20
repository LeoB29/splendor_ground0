"""Fixed-core comparisons and explicitly unavailable optional host counters."""
from copy import deepcopy
import pytest
from tools.splendorust_pool6_core_compare_20261027 import analysis,design,common as c
from tools.splendorust_pool6_core_compare_20261027 import host_observer as h


ARMS=('frozen_cpu0','frozen_cpu2','research_cpu0','research_cpu2')


def sample(start,end,ticks,faults):
    return dict(start_ns=start,end_ns=end,system=dict(kernel_100ns=ticks*4,user_100ns=ticks,
        idle_100ns=ticks*2,available_physical_bytes=5*1024**3,commit_headroom_bytes=7*1024**3),
        processes=[dict(identity=dict(role=arm+':'+role,pid=100+i*3+j,created=1.),
            kernel_100ns=ticks*(i+1),user_100ns=0,page_faults=faults*(i+1),
            working_set_bytes=100*(i+1),private_bytes=200*(i+1))
            for i,arm in enumerate(ARMS) for j,role in enumerate(('python','search','reference'))])


def fixture_rows():
    panel=dict(positions=[dict(position_index=i,case_id=f'case{i}',
        selection='observed_slow_target' if i<2 else 'fixed_structural_control',
        observer={},observer_sha256=c.digest({}),before_snapshot={},
        expectations={route:dict(expected_record={'path':'synthetic-only'}) for route in ('frozen','research')}) for i in range(4)])
    matrix=design.build_matrix(panel)
    samples=[sample(0,10,10,1),sample(90,100,30,4)]
    contexts=[dict(identity=x['identity'],net_change=2,interval_total_context_switches=None,
        interval_total_context_switches_reason='not_identifiable_from_live_thread_aggregates') for x in samples[0]['processes']]
    rows=[dict(spec=deepcopy(spec),route=spec['route'],arm_id=spec['arm_id'],logical_cpu=spec['logical_cpu'],completed=True,
        parent_profile=dict(request_start_ns=0,receipt_arrival_ns=100),
        host_observation=dict(enabled=spec['observer_enabled'],status='observed' if spec['observer_enabled'] else 'disabled',
            errors=[],samples=deepcopy(samples) if spec['observer_enabled'] else [],
            host_cpu_availability=h.host_cpu_availability(samples if spec['observer_enabled'] else []),
            context_switch_deltas=deepcopy(contexts),missed_intervals=0,context_before=dict(start_ns=0),context_after=dict(end_ns=100)),
        clock=dict(status='ok',receipt_elapsed_seconds=.4,result=dict(diagnostics=dict(inference_calls=65,phases=[]))))
        for spec in matrix['requests']]
    return panel,matrix,rows


def test_full_inventory_per_arm_pairs_and_observer_denominators():
    panel,matrix,rows=fixture_rows();result=analysis.analyze(rows,panel,matrix,True)
    assert result['complete'] and result['attempted']==288 and result['overhead_gate']['passed']
    assert len(result['cells'])==16 and len(result['overhead_cell_pairs'])==8
    assert len(result['paired_core_observations'])==144
    assert all(x['complete'] and x['cpu2_minus_cpu0_receipt_seconds']==0 for x in result['paired_core_observations'])
    assert {tuple(x['core_order']) for x in result['paired_core_observations']}=={(0,2),(2,0)}
    counts=result['host_cpu_availability']
    assert counts['accepted_measurement_windows']==240 and counts['disabled_windows']==48
    assert counts['whole_window']['denominator']==240 and counts['whole_window']['available']==240
    assert result['engineering_gate']['eligible'] is True and result['engineering_gate']['advantage'] is False
    assert all(x['cpu0']['n']==x['cpu2']['n']==48 for x in result['engineering_gate']['per_player'].values())
    assert not result['engineering_gate']['automatic_promotion'] and not result['benchmark_authorized_to_launch']


def test_active_scope_excludes_same_player_other_core_and_keeps_memory():
    samples=[sample(0,10,10,1),sample(90,100,30,4)]
    result=analysis.counter_span(samples,'frozen_cpu2')
    assert result['active_player_cpu_seconds']==pytest.approx(120/1e7)
    assert result['active_player_page_faults']==18 and result['host_busy_fraction']==pytest.approx(.6)
    assert len(result['processes'])==12 and result['processes'][3]['maximum_private_bytes']==400
    with pytest.raises(ValueError,match='active player/core'):analysis.counter_span(samples,'frozen')


def test_active_context_and_sample_spans_bind_exact_core():
    _,_,rows=fixture_rows();row=next(r for r in rows if r['spec']['observer_enabled'] and r['arm_id']=='research_cpu2')
    samples=[sample(0,10,0,0),sample(20,30,10,1),sample(40,50,20,2),sample(60,80,30,3)]
    row['host_observation']['samples']=samples;row['host_observation']['host_cpu_availability']=h.host_cpu_availability(samples)
    row['parent_profile']=dict(request_start_ns=15,receipt_arrival_ns=70)
    summary=analysis.host_summary(row)
    assert summary['complete_samples_inside_request']==2
    assert summary['whole_observation_window']['active_player_page_faults']==36
    assert summary['wholly_in_request_sample_span']['active_player_page_faults']==12
    assert summary['context_switch_window']['active_player_live_thread_aggregate_net_change']==6
    assert summary['context_switch_window']['interval_total_context_switches'] is None


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


def test_overhead_negative_and_missing_pair_stay_explicit():
    panel,matrix,rows=fixture_rows()
    prefix=rows[:96]
    for row in prefix:
        if row['spec']['observer_enabled']:row['clock']['receipt_elapsed_seconds']=.5
    result=analysis.analyze(prefix,panel,matrix,False)
    assert result['overhead_gate']['complete'] and not result['overhead_gate']['passed']
    assert result['engineering_gate']['eligible'] is None
    partial=analysis.analyze(rows[:1],panel,matrix,False)['paired_core_observations'][0]
    assert partial['complete'] is False and partial['cpu2_minus_cpu0_receipt_seconds'] is None


def inconsistent_samples():
    samples=[sample(10*i,10*i+5,10+i*10,i) for i in range(4)]
    samples[1]['system']['kernel_100ns']=200
    return samples


def test_bad_whole_window_suppresses_clean_inner_span_without_losing_memory_or_process_cpu():
    _,_,rows=fixture_rows();row=next(r for r in rows if r['spec']['observer_enabled'])
    value=row['host_observation'];value['samples']=inconsistent_samples()
    value['host_cpu_availability']=h.host_cpu_availability(value['samples'])
    row['parent_profile']=dict(request_start_ns=20,receipt_arrival_ns=40)
    summary=analysis.host_summary(row)
    assert summary['host_cpu_availability']['valid'] is False
    assert summary['in_request_host_cpu_availability']['local_interval_classification']['valid'] is True
    assert summary['in_request_host_cpu_availability']['available'] is False
    assert summary['wholly_in_request_sample_span']['host_busy_fraction'] is None
    assert summary['whole_observation_window']['active_player_cpu_seconds']>0
    assert summary['minimum_available_ram_bytes']==5*1024**3


def test_unavailable_host_cpu_does_not_break_timing_gate_and_is_counted():
    panel,matrix,rows=fixture_rows();row=next(r for r in rows if r['spec']['observer_enabled'])
    value=row['host_observation'];value['samples']=inconsistent_samples()
    value['host_cpu_availability']=h.host_cpu_availability(value['samples'])
    result=analysis.analyze(rows,panel,matrix,True);counts=result['host_cpu_availability']
    assert result['overhead_gate']['passed'] and result['complete']
    assert counts['whole_window']['denominator']==240 and counts['whole_window']['available']==239
    assert counts['whole_window']['unavailable']==1 and counts['whole_window']['busy_fraction']['n']==239
    assert counts['whole_window']['unavailable_windows'][0]['arm_id']==row['arm_id']


@pytest.mark.parametrize('fault',['forged_availability','process_counter','process_identity','arm_binding'])
def test_optional_availability_never_weakens_mandatory_checks(fault):
    panel,matrix,rows=fixture_rows();row=next(r for r in rows if r['spec']['observer_enabled'])
    if fault=='forged_availability':row['host_observation']['samples']=inconsistent_samples()
    elif fault=='process_counter':row['host_observation']['samples'][0]['processes'][0]['page_faults']=99
    elif fault=='process_identity':row['host_observation']['samples'][0]['processes'][0]['identity']['pid']=9999
    else:row['arm_id']='frozen_cpu2' if row['arm_id']!='frozen_cpu2' else 'research_cpu0'
    with pytest.raises(ValueError):analysis.analyze(rows,panel,matrix,True)
