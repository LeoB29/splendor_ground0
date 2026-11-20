from copy import deepcopy
import pytest
from tools.splendorust_pool6_hostdiag_20261027 import analysis
from tools.splendorust_pool6_hostdiag_20261027 import design,common as c


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
    contexts=[dict(identity=x['identity'],total_context_switches=2) for x in samples[0]['processes']]
    row=dict(route='frozen',parent_profile=dict(request_start_ns=15,receipt_arrival_ns=70),
        host_observation=dict(enabled=True,status='observed',samples=samples,context_switch_deltas=contexts,
            missed_intervals=0,context_before=dict(start_ns=0),context_after=dict(end_ns=90)))
    result=analysis.host_summary(row)
    assert result['complete_samples_inside_request']==2
    assert result['whole_observation_window']['active_player_page_faults']==9
    assert result['wholly_in_request_sample_span']['active_player_page_faults']==3
    assert result['context_switch_window']['active_player_total']==6
    assert analysis.host_summary(dict(host_observation=dict(enabled=False,status='disabled')))['measurements_available'] is False


def fixture_rows():
    panel=dict(positions=[dict(position_index=i,case_id=f'case{i}',
        selection='observed_slow_target' if i<9 else 'fixed_structural_control',source_route='frozen',
        observer={},observer_sha256=c.digest({}),before_snapshot={},
        expectations={route:dict(expected_first_observation=True) for route in ('frozen','research')}) for i in range(18)])
    matrix=design.build_matrix(panel)
    rows=[dict(spec=spec,route=spec['route'],completed=True,
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
