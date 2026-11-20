from collections import Counter
from copy import deepcopy
from tools.splendorust_pool6_hostdiag_20261027_02 import design,common as c


def test_fixed_control_selection_does_not_use_control_timing(monkeypatch):
    expected=design.build_panel()
    original=c.read
    targets={str(design.move_path(g,p)) for g,p in design.TARGETS}
    def changed(path):
        value=original(path)
        if str(path) not in targets and 'clock' in value and 'ply' in value:
            value=deepcopy(value)
            value['clock']['receipt_elapsed_seconds']=9999.
        return value
    monkeypatch.setattr(c,'read',changed)
    assert design.build_panel()==expected


def test_matrix_all_cells_balanced_and_exact_reference_scope():
    panel=design.build_panel();rows=design.build_matrix(panel)['requests']
    assert len(rows)==216
    assert [r['index'] for r in rows]==list(range(216))
    assert all(r['stage']=='overhead' for r in rows[:72])
    assert all(r['stage']=='diagnostic' and r['observer_enabled'] for r in rows[72:])
    for route in ('frozen','research'):
        assert Counter(r['observer_enabled'] for r in rows[:72] if r['route']==route)=={False:18,True:18}
    for position in panel['positions']:
        for route in ('frozen','research'):
            own=[r for r in rows if r['position_index']==position['position_index'] and r['route']==route]
            assert len(own)==(8 if position['selection']=='fixed_structural_control' else 4)
            assert all(('expected_record' in r)==(route==position['source_route']) for r in own)
            assert all(('expected_first_observation' in r)==(route!=position['source_route']) for r in own)
            if len(own)==8:assert [r['observer_enabled'] for r in own[:4]]==[False,True,True,False]
    assert all(r['quota']==64 and r['telemetry'] is False and r['seed_label']=='A' for r in rows)


def test_hard_bounds_and_no_automatic_promotion():
    plan=design.build_plan()
    assert plan['new_games']==0 and plan['promotion'] is False and plan['training'] is False
    assert plan['seconds_per_move']==1. and plan['internal_seconds']==.9
    assert plan['maximum_scientific_decisions']==320 and plan['maximum_parent_requests']==220
    assert plan['resources']['job_seconds']==7200
    gate=plan['overhead_contract']
    assert gate['maximum_mean_ratio']==1.05 and gate['maximum_mean_added_seconds']==.010
    assert gate['maximum_p95_added_seconds']==.025 and gate['failure_skips_full_diagnostic']


def test_same_panel_matrix_gates_and_resources_as_failed_attempt():
    panel=design.build_panel();matrix=design.build_matrix(panel)
    assert panel==c.read(design.PRESERVED_ATTEMPT/'saved_panel.json')
    assert matrix==c.read(design.PRESERVED_ATTEMPT/'request_matrix.json')
    old=c.read(design.PRESERVED_ATTEMPT/'study_plan.json')
    new=design.build_plan()
    assert new['overhead_contract']==old['overhead_contract']
    assert new['resources']==old['resources']
    for key in ('candidate','control','seconds_per_move','internal_seconds','margin_seconds'):
        assert new[key]==old[key]
