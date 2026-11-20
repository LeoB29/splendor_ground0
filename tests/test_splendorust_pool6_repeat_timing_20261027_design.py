from collections import Counter
from copy import deepcopy
import pytest
from tools.splendorust_pool6_repeat_timing_20261027 import common as c,design,analysis


def test_fixed_panel_is_saved_and_controls_have_no_timing_selection():
    panel=design.build_panel()
    assert panel['controls']==['reserved_01','reserved_07','reserved_12']
    assert len(panel['positions'])==6
    assert panel['new_games']==panel['new_trajectories']==0
    for p in panel['positions']:
        assert c.digest(p['observer'])==p['observer_sha256']
        assert c.digest(p['before_snapshot'])==p['before_sha256']
        assert set(p['expectations'])=={'frozen','research'}
        for value in p['expectations'].values():
            c.verify([value.get('expected_record',value.get('expected_entrypoint'))])


def test_counterbalanced_schedule_and_no_missing_cells():
    matrix=design.build_matrix();rows=matrix['requests']
    assert len(rows)==144
    assert set(Counter((r['case_id'],r['route']) for r in rows).values())=={12}
    first=Counter()
    positions=Counter()
    for i in range(0,144,2):
        a,b=rows[i:i+2]
        assert a['round']==b['round'] and a['case_id']==b['case_id']
        assert {a['route'],b['route']}=={'frozen','research'}
        assert a['observer']==b['observer'] and a['before_snapshot']==b['before_snapshot']
        first[a['case_id'],a['route']]+=1
        positions[a['position_index'],(i//2)%6]+=1
    assert set(first.values())=={6}
    assert len(positions)==36 and set(positions.values())=={2}
    assert [r['sequence'] for r in rows]==list(range(144))


def test_plan_does_not_promote_or_change_players():
    plan=design.build_plan()
    assert plan['new_games']==0 and plan['promotion'] is False
    assert plan['player_changes'] is False and plan['observer_changes'] is False
    assert plan['seconds_per_move']==1 and plan['internal_seconds']==.9
    assert plan['margin_seconds']==.7 and 'descriptive' in plan['margin_role']
    assert plan['scientific_decisions']==248 and plan['parent_requests']==148


def test_analysis_retains_slow_and_failed_prefix_without_inventing_completion():
    panel=design.build_panel();matrix=design.build_matrix(panel)
    rows=[]
    for i in range(2):
        spec=matrix['requests'][i]
        rows.append(dict(spec=spec,route=spec['route'],clock=dict(status='ok' if i==0 else 'timeout',
            receipt_elapsed_seconds=.75 if i==0 else None)))
    value=analysis.analyze(rows,panel,matrix,False)
    assert value['attempted']==2 and value['complete'] is False
    assert sum(v['margin_crossings'] for v in value['by_route'].values())==1
    assert sum(v['deadline_failures'] for v in value['by_route'].values())==1
    assert value['benchmark_authorized_to_launch'] is False
    assert value['confidence_intervals'] is None
    with pytest.raises(ValueError):analysis.analyze(rows,panel,matrix,True)
    altered=deepcopy(rows);altered[0]['spec']['sequence']=55
    with pytest.raises(ValueError):analysis.analyze(altered,panel,matrix,False)


def test_nearest_rank_and_empty_distribution():
    assert analysis.distribution([])['n']==0
    assert analysis.distribution(range(12))['p90']==10


def test_incomplete_sidecar_remains_missing_and_error_is_not_timeout():
    row=dict(spec=dict(sequence=0,round=0),clock=dict(status='worker_error',detail='invalid action'),
        child_profile=dict(profiles=[dict(handle_start_ns=1,handle_end_ns=None)],
            post_flush_ns=None,pre_write_ns=2))
    value=analysis.measurements(row)
    assert value['handler_seconds'] is None and value['emission_seconds'] is None
    assert value['failed_request'] and not value['deadline_failed']
    row['clock']['detail']='SAMPLING_INTERNAL_DEADLINE'
    assert analysis.measurements(row)['deadline_failed']
    row['clock']['detail']='ONE_SECOND_INTERNAL_DEADLINE: .901'
    assert analysis.measurements(row)['deadline_failed']
