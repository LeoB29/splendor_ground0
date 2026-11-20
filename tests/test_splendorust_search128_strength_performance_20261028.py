"""Model-free report arithmetic; scientific/transport validation is tested separately."""
from copy import deepcopy
import sys
from types import SimpleNamespace
import pytest
from tools.splendorust_search128_strength_20261028 import performance as report

def row(value,*,role='challenger',seat=0,stage='main',status='ok',elapsed=None):
    return dict(stage=stage,pair_index=0,game_index=0,ply=0,player_role=role,seat=seat,
        mode=report.MODES[role],route=report.MODES[role],quota=report.QUOTAS[role],arm='search',
        placement_mode='isolated',observation_mode='profiled',completed=status=='ok',
        clock=dict(status=status,request_id='request-00000002',receipt_elapsed_seconds=value,
            elapsed_seconds=value+.2 if elapsed is None and value is not None else 10.1 if elapsed is None else elapsed,
            deadline_seconds=10.,received_before_deadline=value is not None and value<=10.,accepted=status=='ok',
            telemetry={}),parent_profile={},child_profile={},observer={})

@pytest.fixture(autouse=True)
def isolate_report_math(monkeypatch):
    # No native process, model, referee transition or actual timing request.
    def check(clock):
        if clock['status']=='ok':
            assert clock['receipt_elapsed_seconds'] is not None and clock['receipt_elapsed_seconds']<=10
    monkeypatch.setattr(report,'_clock',check)
    monkeypatch.setattr(report,'validate_boundary_row',lambda r:dict(parent=dict(request_elapsed_ns=23),child=dict(post_write_gap_ns=-5)))
    monkeypatch.setitem(sys.modules,'tools.splendorust_search128_strength_20261028.scientific',
        SimpleNamespace(classify_failure=lambda receipt:'research_outer_hard_guard'))

def test_thresholds_use_arrival_not_parent_finish_and_exact_boundaries():
    rows=[row(x,elapsed=x+.5) for x in (.7,.70001,.9,.90001,1.,1.00001)]
    out=report.performance(rows)['players']['challenger']
    assert out['completed_over_700ms']==5
    assert out['completed_over_900ms']==3
    assert out['completed_over_one_second']==1
    assert out['completed_at_or_above_one_second']==2
    assert out['completed_arrival_seconds']['max']==1.00001
    assert out['parent_processing_through_validation_seconds']['max']==1.50001

def test_both_stages_and_seats_accounted_without_primary_strength_claim():
    rows=[row(.8,stage='operational'),row(1.2,seat=1),row(.5,role='opponent',seat=1)]
    out=report.performance(rows)
    assert out['rows']==3 and out['players']['challenger']['quota']==128
    assert out['players']['opponent']['quota']==64
    assert all(group['world_pool']==3 for group in out['by_stage_and_seat'])
    assert sum(group['attempted_decisions'] for group in out['by_stage_and_seat'])==3
    assert out['no_strength_claim'] and not out['descriptive_thresholds_change_completion_or_stopping']
    assert sum(g['completed_over_one_second'] for g in out['by_stage_and_seat'])==1

def test_timeout_preserved_separately_not_fabricated_completed_timing():
    rows=[row(.6),row(None,status='timeout')]
    out=report.performance(rows)
    player=out['players']['challenger']
    assert player['attempted_decisions']==2 and player['completed_decisions']==1
    assert player['unfinished_decisions']==1 and player['missing_arrival_timestamps']==1
    assert player['completed_arrival_seconds']['count']==1
    assert len(out['unfinished_requests'])==1
    assert out['unfinished_requests'][0]['receipt_elapsed_seconds'] is None

def test_boundary_group_and_seat_counts_match_once_not_duplicated():
    rows=[row(.8),row(1.1,seat=1),row(.5,role='opponent',stage='operational')]
    out=report.summarize_boundary_rows(rows)
    assert out['rows']==3 and out['observed_faults']==0
    assert sum(g['attempted_decisions'] for g in out['groups'])==3
    assert sum(g['attempted_decisions'] for g in out['by_stage_and_seat'])==3
    assert out['groups'][0]['completed_boundary_metrics']['child.post_write_gap_ns']['distribution']['mean']==-5
    assert out['complete_planned_collection_inferred'] is False

def test_boundary_failed_request_is_not_promoted_into_metrics():
    out=report.summarize_boundary_rows([row(.6),row(None,status='timeout')])
    assert out['observed_faults']==1 and not out['all_collected_decisions_completed']
    assert out['failures'][0]['partial_or_missing_evidence_not_promoted']
    assert out['groups'][0]['completed_arrival_seconds']['count']==1

@pytest.mark.parametrize('field,value',[('quota',64),('quota',True),('seat',True),('placement_mode','original'),('mode','frozen')])
def test_wrong_candidate_identity_rejected(field,value):
    value_row=row(.5);value_row[field]=value
    with pytest.raises(ValueError):report.performance([value_row])

def test_empty_groups_have_unavailable_quantiles_not_zero_latency():
    out=report.performance([])
    assert out['players']['challenger']['completed_arrival_seconds']['max'] is None
    assert out['players']['opponent']['completed_arrival_seconds']['mean'] is None

def test_completion_mismatch_rejected_in_boundary():
    value_row=row(.5);value_row['completed']=False
    with pytest.raises(ValueError,match='Completion'):report.summarize_boundary_rows([value_row])
