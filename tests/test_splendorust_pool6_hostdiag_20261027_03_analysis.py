from copy import deepcopy
import pytest
from tools.splendorust_pool6_hostdiag_20261027_03 import analysis
from tools.splendorust_pool6_hostdiag_20261027_03 import design,common as c
from tools.splendorust_pool6_hostdiag_20261027_03 import host_observer as h


def sample(start,end,ticks,faults):
    return dict(start_ns=start,end_ns=end,system=dict(kernel_100ns=ticks*4,user_100ns=ticks,
        idle_100ns=ticks*2,available_physical_bytes=5*1024**3,commit_headroom_bytes=7*1024**3),
        processes=[dict(identity=dict(role=role,pid=i,created=1.),kernel_100ns=ticks,user_100ns=0,
            page_faults=faults,working_set_bytes=100,private_bytes=200) for i,role in enumerate(
            ('frozen:python','frozen:search','frozen:reference','research:python','research:search','research:reference'))])


def test_counter_scope_includes_helpers_but_not_other_player():
    result=analysis.counter_span([sample(0,10,10,1),sample(90,100,30,4)],'frozen')
    assert result['active_player_cpu_seconds']==pytest.approx(60/1e7)
    assert result['active_player_page_faults']==9
    assert result['host_busy_fraction']==pytest.approx(.6)
    bad=sample(90,100,30,4);bad['processes'][0]['identity']['pid']=50
    with pytest.raises(ValueError,match='identity'):analysis.counter_span([sample(0,10,10,1),bad],'frozen')


def test_samples_crossing_request_edges_are_not_called_inside():
    samples=[sample(0,10,0,0),sample(20,30,10,1),sample(40,50,20,2),sample(60,80,30,3)]
    contexts=[dict(identity=x['identity'],net_change=2,interval_total_context_switches=None,
        interval_total_context_switches_reason='not_identifiable_from_live_thread_aggregates') for x in samples[0]['processes']]
    row=dict(route='frozen',parent_profile=dict(request_start_ns=15,receipt_arrival_ns=70),
        host_observation=dict(enabled=True,status='observed',samples=samples,context_switch_deltas=contexts,
            host_cpu_availability=h.host_cpu_availability(samples),
            missed_intervals=0,context_before=dict(start_ns=0),context_after=dict(end_ns=90)))
    result=analysis.host_summary(row)
    assert result['complete_samples_inside_request']==2
    assert result['whole_observation_window']['active_player_page_faults']==9
    assert result['wholly_in_request_sample_span']['active_player_page_faults']==3
    assert result['context_switch_window']['active_player_live_thread_aggregate_net_change']==6
    assert result['context_switch_window']['interval_total_context_switches'] is None
    assert analysis.host_summary(dict(host_observation=dict(enabled=False,status='disabled')))['measurements_available'] is False


@pytest.mark.parametrize('change',[-3,0,7])
def test_context_change_never_becomes_interval_total(change):
    samples=[sample(0,10,10,1),sample(90,100,30,4)]
    contexts=[dict(identity=x['identity'],net_change=change,interval_total_context_switches=None,
        interval_total_context_switches_reason='not_identifiable_from_live_thread_aggregates') for x in samples[0]['processes']]
    row=dict(route='frozen',parent_profile=dict(request_start_ns=15,receipt_arrival_ns=70),
        host_observation=dict(enabled=True,status='observed',samples=samples,context_switch_deltas=contexts,
            host_cpu_availability=h.host_cpu_availability(samples),
            missed_intervals=0,context_before=dict(start_ns=0),context_after=dict(end_ns=110)))
    result=analysis.host_summary(row)
    assert result['context_switch_window']['active_player_live_thread_aggregate_net_change']==3*change
    assert result['context_switch_window']['interval_total_context_switches'] is None
    assert result['whole_observation_window']['active_player_cpu_seconds']>0


def fixture_rows():
    panel=dict(positions=[dict(position_index=i,case_id=f'case{i}',
        selection='observed_slow_target' if i<9 else 'fixed_structural_control',source_route='frozen',
        observer={},observer_sha256=c.digest({}),before_snapshot={},
        expectations={route:dict(expected_first_observation=True) for route in ('frozen','research')}) for i in range(18)])
    matrix=design.build_matrix(panel)
    samples=[sample(0,10,10,1),sample(90,100,30,4)]
    contexts=[dict(identity=x['identity'],net_change=2,interval_total_context_switches=None,
        interval_total_context_switches_reason='not_identifiable_from_live_thread_aggregates') for x in samples[0]['processes']]
    rows=[dict(spec=spec,route=spec['route'],completed=True,parent_profile=dict(request_start_ns=0,receipt_arrival_ns=100),
        host_observation=dict(enabled=spec['observer_enabled'],
            status='observed' if spec['observer_enabled'] else 'disabled',errors=[],samples=deepcopy(samples) if spec['observer_enabled'] else [],
            host_cpu_availability=h.host_cpu_availability(samples if spec['observer_enabled'] else []),
            context_switch_deltas=deepcopy(contexts),missed_intervals=0,context_before=dict(start_ns=0),context_after=dict(end_ns=100)),
        clock=dict(status='ok',receipt_elapsed_seconds=.4,result=dict(diagnostics=dict(inference_calls=65,phases=[]))))
        for spec in matrix['requests']]
    return panel,matrix,rows


def test_full_216_row_analysis_uses_only_overhead_gate_prefix():
    panel,matrix,rows=fixture_rows()
    result=analysis.analyze(rows,panel,matrix,True)
    assert result['complete'] and result['attempted']==216
    assert result['overhead_gate']['passed'] and result['overhead_gate']['attempted']==72
    assert len(result['overhead_cell_pairs'])==18


def test_negative_overhead_and_interrupted_prefix_analysis_are_preserved():
    panel,matrix,rows=fixture_rows()
    rows=rows[:72]
    for row in rows:
        if row['spec']['observer_enabled']:row['clock']['receipt_elapsed_seconds']=.5
    result=analysis.analyze(rows,panel,matrix,False)
    assert not result['complete'] and not result['overhead_gate']['passed']
    assert result['overhead_gate']['complete'] and result['attempted']==72
    prefix=rows[:2];prefix[-1]['completed']=False;prefix[-1]['clock']['status']='timeout'
    prefix[-1]['clock']['receipt_elapsed_seconds']=None
    result=analysis.analyze(prefix,panel,matrix,False)
    assert not result['overhead_gate']['complete'] and result['attempted']==2


def inconsistent_samples():
    values=[sample(10*i,10*i+5,10+i*10,i) for i in range(4)]
    values[1]['system']['kernel_100ns']=200
    return values


def test_host_cpu_unavailable_even_when_endpoints_recover_process_counters_stay_valid():
    samples=inconsistent_samples()
    result=analysis.counter_span(samples,'frozen')
    assert result['host_busy_fraction'] is None and result['host_cpu_availability']['valid'] is False
    assert result['host_cpu_availability']['offending_pairs']
    assert result['active_player_cpu_seconds']==pytest.approx(90/1e7)
    assert result['active_player_page_faults']==9 and result['processes'][0]['maximum_private_bytes']==200
    assert samples[-1]['system']['kernel_100ns']>samples[0]['system']['kernel_100ns']
    inner=analysis.counter_span(samples[1:3],'frozen')
    assert inner['host_busy_fraction'] is None and not inner['host_cpu_availability']['available']


@pytest.mark.parametrize('field',['kernel_100ns','user_100ns','page_faults','identity'])
def test_process_counter_or_identity_failure_remains_fatal_even_with_optional_host_cpu(field):
    samples=inconsistent_samples()
    if field=='identity':samples[1]['processes'][0]['identity']['pid']=9000
    else:samples[1]['processes'][0][field]=999
    with pytest.raises(ValueError,match='process'):analysis.counter_span(samples,'frozen')


def test_actual_saved_v2_window193_is_only_a_labelled_analysis_policy_projection():
    source=c.ROOT/'outputs/splendorust_pool6_hostdiag_20261027_02/records/0193.json'
    source_sha=c.file_hash(source);original=c.read(source);unchanged=deepcopy(original)
    projected=deepcopy(original)
    projected['analysis_test_projection']=dict(source_sha256=source_sha,original_observer_status='observer_failed',
        original_errors=deepcopy(original['host_observation']['errors']),
        model_free=True,old_attempt_accepted=False,purpose='Evaluate v3 optional-field interpretation of unchanged saved samples only')
    value=projected['host_observation'];value['status']='observed'
    value['host_cpu_availability']=h.host_cpu_availability(value['samples'])
    value['context_switch_deltas']=h._context_delta(value['context_before'],value['context_after'],value['targets'])
    summary=analysis.host_summary(projected)
    assert not summary['host_cpu_availability']['available'] and summary['whole_observation_window']['host_busy_fraction'] is None
    assert summary['host_cpu_availability']['offending_pairs'][0]['before_index']==7
    assert summary['host_cpu_availability']['offending_pairs'][0]['after_index']==8
    assert summary['whole_observation_window']['active_player_cpu_seconds']>=0
    assert summary['minimum_available_ram_bytes']>0 and summary['minimum_commit_headroom_bytes']>0
    assert not summary['in_request_host_cpu_availability']['available']
    assert summary['in_request_host_cpu_availability']['reason']=='whole_window_host_cpu_inconsistent'
    assert summary['in_request_host_cpu_availability']['local_interval_classification']['valid']
    assert 'busy_fraction' not in summary['in_request_host_cpu_availability']['local_interval_classification']
    assert summary['wholly_in_request_sample_span']['host_busy_fraction'] is None
    assert original==unchanged and c.file_hash(source)==source_sha and original['host_observation']['status']=='observer_failed'


def test_analysis_rejects_forged_clean_host_cpu_availability():
    _,_,rows=fixture_rows();row=next(r for r in rows if r['spec']['observer_enabled'])
    row['host_observation']['samples']=inconsistent_samples()
    with pytest.raises(ValueError,match='availability differs'):analysis.host_summary(row)


def test_analysis_counts_unavailable_windows_without_rejecting_overhead_or_hiding_memory():
    panel,matrix,rows=fixture_rows();row=next(r for r in rows if r['spec']['observer_enabled'])
    value=row['host_observation'];value['samples']=inconsistent_samples()
    value['host_cpu_availability']=h.host_cpu_availability(value['samples'])
    result=analysis.analyze(rows,panel,matrix,True)
    assert result['complete'] and result['overhead_gate']['passed']
    counts=result['host_cpu_availability']
    assert counts['accepted_measurement_windows']==180 and counts['disabled_windows']==36
    assert counts['failed_measurement_windows']==0
    assert counts['whole_window']['denominator']==180 and counts['whole_window']['available']==179
    assert counts['whole_window']['unavailable']==1 and counts['whole_window']['busy_fraction']['n']==179
    assert counts['whole_window']['unavailable_windows'][0]['sequence']==row['spec']['sequence']
    assert sum(counts['whole_window']['reasons'].values())==1
    measured=analysis.host_summary(row)
    assert measured['measurements_available'] and measured['minimum_available_ram_bytes']==5*1024**3
