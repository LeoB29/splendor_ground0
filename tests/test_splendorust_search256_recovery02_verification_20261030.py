"""Three immutable execution layers, clean seam and unchanged retry sensitivity."""
from copy import deepcopy
from pathlib import Path
import pytest
from tools.splendorust_search256_recovery_20261030_02 import audit,analysis,verification as v,schedule,common as c


@pytest.mark.parametrize('game,ply,root',[(234,80,'old'),(235,29,'old'),(235,30,'previous'),(293,99,'previous'),
    (294,49,'previous'),(294,50,'new'),(295,0,'new'),(399,0,'new')])
def test_exact_three_layer_move_dispatch(game,ply,root):
    expected={'old':v.OLD_OUT,'previous':v.PREVIOUS_OUT,'new':c.OUT}[root]
    assert v.source_move_path(schedule.make_schedule('main')[game],ply).is_relative_to(expected)


def test_operational_evidence_always_comes_from_original_attempt():
    assert v.source_move_path(schedule.make_schedule('operational')[0],0).is_relative_to(v.OLD_OUT)


def seam():
    prefix=[dict(ply=i,after={'turn':i+1}) for i in range(50)]
    first=dict(ply=50,before={'turn':50},player_role='opponent',route='frozen',mode='frozen',quota=128,
        clock={'request_id':'request-00000003'})
    return prefix,deepcopy(prefix)+[first],schedule.make_schedule('main')[294]


def test_clean_seam_has_zero_retries():
    prefix,rows,spec=seam();result=audit.validate_segment_seam(rows,prefix,spec)
    assert result['retained_moves']==50 and result['retried_moves']==0
    assert result['previous_move_clock_started'] is False and result['historical_disclosed_retry_game_index']==235


@pytest.mark.parametrize('change',('prefix','count','ply','snapshot','role','quota','request','game'))
def test_mutated_clean_seam_cannot_pass(change):
    prefix,rows,spec=seam();spec=deepcopy(spec)
    if change=='prefix':rows[1]['after']={'tampered':True}
    if change=='count':rows.pop(0)
    if change=='ply':rows[50]['ply']=51
    if change=='snapshot':rows[50]['before']={}
    if change=='role':rows[50]['player_role']='challenger'
    if change=='quota':rows[50]['quota']=256
    if change=='request':rows[50]['clock']['request_id']='request-00000027'
    if change=='game':spec['game_index']=235
    with pytest.raises(ValueError):audit.validate_segment_seam(rows,prefix,spec)


def test_later_games_require_no_prefix_or_retry(monkeypatch):
    monkeypatch.setattr(audit,'audit_game',lambda *a,**k:{'ordinary':True})
    assert v.audit_merged_game({},[],{},schedule.make_schedule('main')[295],{}, {})=={'ordinary':True}


def test_clean_prefix_manifest_binds_closed_previous_workers():
    prefix=dict(original_initial=c.artifact(v.PREVIOUS_OUT/'games/main/research/p147_g294/initial.json'),
        move_artifacts=[c.artifact(v.PREVIOUS_OUT/'games/main/research/p147_g294/moves'/f'{i:04d}.json') for i in range(50)])
    result=v.expected_prefix_manifest(prefix,{})
    assert result['new_decision_retries']==0 and result['previous_next_request_started'] is False
    assert len(result['previous_moves'])==50 and len(result['previous_closed_workers'])==2
    assert result['previous_next_request_id']=='request-00000027'


def test_count_reconciliation_covers_all_three_layers_and48_reproductions():
    assert v.reconcile_execution_counts(20000,20000+872+96,48,260,864)


@pytest.mark.parametrize('field',range(5))
def test_missing_or_extra_execution_cannot_be_laundered(field):
    values=[20000,20000+872+96,48,260,864];values[field]+=1
    with pytest.raises(ValueError):v.reconcile_execution_counts(*values)


def test_same_scientific_analysis_and_old235_sensitivity_only(monkeypatch):
    monkeypatch.setattr(analysis,'REPLICATES',1000)
    games=[dict(s,complete=True,reason='natural',winner=s['challenger_seat']) for s in schedule.make_schedule('main')]
    original=deepcopy(games);official=analysis.analyze_main(games,True)
    sensitivity=analysis.analyze_recovery_sensitivity(games,official)
    assert games==original and sensitivity['disclosed_retry_game_index']==235
    assert sensitivity['recovery_conservative']['numerator']==798
    assert sensitivity['recovery_conservative']['pair_numerators'][117]==2
    assert sensitivity['recovery_conservative']['pair_numerators'][147]==4
    assert analysis.verify_recovery_sensitivity(games,official,sensitivity)['status']=='passed'
    broken=deepcopy(sensitivity);broken['disclosed_retry_game_index']=294
    with pytest.raises(ValueError):analysis.verify_recovery_sensitivity(games,official,broken)


def test_previous_saved_failure_cannot_be_relabelled_success(monkeypatch):
    from tools.splendorust_search256_recovery_20261030_02 import preparation_support as support
    monkeypatch.setattr(support,'previous_exit_provenance',lambda:{})
    monkeypatch.setattr(support,'ram_wait_accounting',lambda:{'charged_prior_ram_wait_seconds':c.PRIOR_RAM_WAIT_SECONDS})
    original=c.read
    def read(path):
        value=original(path)
        if Path(path)==v.PREVIOUS_OUT/'completion.json':value=dict(value,status='completed_verified')
        return value
    monkeypatch.setattr(c,'read',read)
    with pytest.raises(ValueError,match='incomplete59'):v.verify_previous_stop({'manifest':{}},None)


def test_no_recovery_private_output_path_can_alias_a_prior_frozen_root():
    assert len({c.OUT,v.PREVIOUS_OUT,v.OLD_OUT})==3
    assert not c.OUT.is_relative_to(v.PREVIOUS_OUT) and not c.OUT.is_relative_to(v.OLD_OUT)
