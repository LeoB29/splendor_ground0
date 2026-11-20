from collections import Counter
from copy import deepcopy
import pytest
from tools.splendorust_pool6_core_compare_20261027 import design, common as c, preparation


def test_panel_exact_both_player_targets_and_structural_controls(monkeypatch):
    panel=design.build_panel()
    assert panel['targets']==['g2_p36','g2_p45'] and panel['controls']==['g2_p34','g2_p37']
    original=c.read
    def changed(path):
        value=original(path)
        if 'records' in str(path) and str(path).endswith('.json') and int(path.stem) not in (80,170):
            value=deepcopy(value);value['clock']['receipt_elapsed_seconds']=9999.
        return value
    monkeypatch.setattr(c,'read',changed)
    assert design.build_panel()==panel
    for p in panel['positions']:
        assert set(p['expectations'])=={'frozen','research'}
        for route,item in p['expectations'].items():
            row=original(c.ROOT/item['expected_record']['path'])
            assert row['route']==route and row['observer']==p['observer']


def test_matrix_complete_balanced_adjacent_core_pairs():
    rows=design.build_matrix()['requests']
    assert len(rows)==288 and [r['index'] for r in rows]==list(range(288))
    assert Counter(r['stage'] for r in rows)=={'overhead':96,'diagnostic':192}
    for arm in design.ARMS:
        own=[r for r in rows if r['arm_id']==arm]
        assert Counter(r['observer_enabled'] for r in own if r['stage']=='overhead')=={False:12,True:12}
        assert len([r for r in own if r['stage']=='diagnostic'])==48
        for pos in range(4):
            assert len([r for r in own if r['stage']=='diagnostic' and r['position_index']==pos])==12
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['logical_cpu']!=b['logical_cpu'] and {a['logical_cpu'],b['logical_cpu']}=={0,2}
        for key in ('stage','route','observer','position_index','round','core_pair_index','observer_enabled','expected_record'):
            assert a[key]==b[key]
    for route in design.ROUTES:
        for pos in range(4):
            leads=[r['logical_cpu'] for r in rows[96::2] if r['route']==route and r['position_index']==pos]
            assert Counter(leads)=={0:6,2:6}
    assert all(r['quota']==64 and r['telemetry'] is False and r['seed_label']=='A' for r in rows)


def test_schedule_joint_leads_and_position_order_are_balanced():
    rows=design.build_matrix()['requests']
    for pos in range(4):
        blocks=[r for r in rows[96::4] if r['position_index']==pos]
        assert Counter((r['route'],r['logical_cpu']) for r in blocks)=={
            ('frozen',0):3,('frozen',2):3,('research',0):3,('research',2):3}
        for slot in range(4):
            at_slot=[rows[96+16*r+slot*4]['position_index'] for r in range(12)]
            assert Counter(at_slot)=={0:3,1:3,2:3,3:3}
    for r in range(12):
        assert rows[8*r]['position_index']==(2 if r%2==0 else 3)


def test_bounds_and_eligibility_are_not_live_admission():
    p=design.build_plan();f=preparation.contract()
    assert p['new_games']==0 and not p['promotion'] and not p['training']
    assert p['startup_complete_decisions']==f['qualification_decisions']==208
    assert p['maximum_parent_requests']==f['parent_requests']==296
    assert p['maximum_scientific_decisions']==f['scientific_decisions']==496
    assert p['graph_cases']==f['graph_cases']==44 and c.MAX_WORKERS==4
    assert p['seconds_per_move']==1. and p['internal_seconds']==.9 and p['margin_seconds']==.7
    assert p['resources']['job_seconds']==7200 and p['resources']['max_worker_rss']==2*c.GiB
    assert p['host_observer_contract']['maximum_compute_targets']==12
    e=p['engineering_contract']
    assert e['completed_verified_full_stage_required'] and e['cpu2_all_full_responses_strictly_below_margin']
    assert e['maximum_per_player_mean_ratio']==1.05 and e['maximum_per_player_p95_added_seconds']==.025
    assert e['statistical_significance_claim'] is False
    gate=p['overhead_contract']
    assert gate['per_arm_mode_count']==12 and gate['maximum_mean_ratio']==1.05
    assert gate['maximum_mean_added_seconds']==.010 and gate['maximum_p95_added_seconds']==.025
    assert gate['failure_skips_full_diagnostic']


def test_prior_consumed_paths_keep_original_spelling_and_reject_changes(tmp_path,monkeypatch):
    path=tmp_path/'nested'/'runtime.bin';path.parent.mkdir();path.write_bytes(b'original evidence')
    item=dict(path=str(path),bytes=path.stat().st_size,sha256=c.file_hash(path))
    preparation.verify_consumed([item])
    monkeypatch.setattr(c,'ROOT',tmp_path)
    relative={**item,'path':'nested\\runtime.bin'}
    preparation.verify_consumed([relative])
    path.write_bytes(b'altered evidence')
    with pytest.raises(ValueError,match='Prior consumed evidence changed'):
        preparation.verify_consumed([item])
