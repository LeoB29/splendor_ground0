from collections import Counter
from copy import deepcopy
import pytest
from tools.splendorust_pool6_core_compare_20261027_02 import design, common as c, preparation
from tools.splendorust_pool6_core_compare_20261027 import design as previous


def test_exact_same_panel_and_authenticated_targets():
    panel=design.build_panel()
    assert panel==previous.build_panel()==c.read(design.PRESERVED_ATTEMPT/'saved_panel.json')
    assert panel['targets']==['g2_p36','g2_p45'] and panel['controls']==['g2_p34','g2_p37']
    for p in panel['positions']:
        for route,item in p['expectations'].items():
            row=c.read(c.ROOT/item['expected_record']['path'])
            assert row['route']==route and row['observer']==p['observer']


def test_only_prospective_sampler_and_numbering_changes():
    panel=design.build_panel();rows=design.build_matrix(panel)['requests']
    old=[r for r in previous.build_matrix(panel)['requests'] if r['stage']=='diagnostic']
    assert len(rows)==len(old)==192
    for index,(new,prior) in enumerate(zip(rows,old,strict=True)):
        expected=deepcopy(prior)
        expected.update(index=index,sequence=index,core_pair_index=index//2,observer_enabled=False)
        assert new==expected
    assert Counter(r['stage'] for r in rows)=={'diagnostic':192}
    assert all(r['observer_enabled'] is False for r in rows)


def test_four_arm_balance_adjacent_pairs_and_all_joint_leads():
    rows=design.build_matrix()['requests']
    for arm in design.ARMS:
        own=[r for r in rows if r['arm_id']==arm]
        assert len(own)==48
        assert Counter(r['position_index'] for r in own)=={0:12,1:12,2:12,3:12}
    for a,b in zip(rows[::2],rows[1::2]):
        assert {a['logical_cpu'],b['logical_cpu']}=={0,2}
        for key in ('stage','route','observer','position_index','round','core_pair_index','observer_enabled','expected_record'):
            assert a[key]==b[key]
    for pos in range(4):
        blocks=[r for r in rows[::4] if r['position_index']==pos]
        assert Counter((r['route'],r['logical_cpu']) for r in blocks)=={
            ('frozen',0):3,('frozen',2):3,('research',0):3,('research',2):3}
        for route in design.ROUTES:
            assert Counter(r['logical_cpu'] for r in rows[::2] if r['route']==route and r['position_index']==pos)=={0:6,2:6}
    for slot in range(4):
        assert Counter(rows[16*r+slot*4]['position_index'] for r in range(12))=={0:3,1:3,2:3,3:3}


def test_same_engineering_criteria_and_hard_limits_no_overhead_stage():
    plan=design.build_plan();f=preparation.contract()
    assert design.engineering_contract()==previous.engineering_contract()
    assert plan['maximum_ordinary_requests']==f['total_requests']==192
    assert plan['startup_complete_decisions']==f['qualification_decisions']==208
    assert plan['maximum_parent_requests']==f['parent_requests']==200
    assert plan['maximum_scientific_decisions']==f['scientific_decisions']==400
    assert plan['graph_cases']==f['graph_cases']==44 and c.MAX_WORKERS==4
    assert plan['overhead_requests']==f['overhead_requests']==0
    assert 'overhead' not in c.STAGES and 'overhead_contract' not in plan
    assert plan['seconds_per_move']==1. and plan['internal_seconds']==.9 and plan['margin_seconds']==.7
    assert plan['resources']['job_seconds']==7200 and plan['resources']['max_worker_rss']==2*c.GiB
    assert plan['new_games']==0 and not plan['promotion'] and not plan['training']
    observer=plan['host_observer_contract']
    assert observer['enabled'] is False and observer['enabled_requests']==0
    assert observer['disabled_requests']==192 and observer['per_window_snapshots']==observer['per_window_context_reads']==0
    assert observer['disabled_receipts_required'] and observer['accidental_enabled_request_fatal']


def test_prior_consumed_paths_keep_original_spelling_and_reject_changes(tmp_path,monkeypatch):
    path=tmp_path/'nested'/'runtime.bin';path.parent.mkdir();path.write_bytes(b'original evidence')
    item=dict(path=str(path),bytes=path.stat().st_size,sha256=c.file_hash(path))
    preparation.verify_consumed([item]);monkeypatch.setattr(c,'ROOT',tmp_path)
    preparation.verify_consumed([{**item,'path':'nested\\runtime.bin'}])
    path.write_bytes(b'altered evidence')
    with pytest.raises(ValueError,match='Prior consumed evidence changed'):
        preparation.verify_consumed([item])
