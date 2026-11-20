"""Layered audit fixtures: historical corrected transitions, synthetic routing.

The ledger fixture explicitly stubs only numerical/profile/clock validation. It
does not pretend historical decisions were produced by the new candidates.
Actual worker entrypoint qualification is owned by the separate worker suite.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.splendorust_exploration128_20261029_02 import audit, performance as timing, schedule


ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'outputs/splendorust_search128_strength_20261028_01/games/main/p016_g33'


@pytest.fixture(scope='module')
def saved_corrected_game():
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    return (read(SAVED/'initial.json')['initial'],
            [read(path) for path in sorted((SAVED/'moves').glob('*.json'))],
            read(SAVED/'result.json'))


def reroute_fixture(saved,candidate='high',stage='development'):
    initial,rows,ending=deepcopy(saved)
    spec=next(row for row in schedule.make_schedule(stage,candidate)
              if row['challenger_seat']==ending['challenger_seat'])
    ending.update(spec)
    for row in rows:
        for key in ('stage','candidate','pair_index','game_index'):row[key]=spec[key]
        row['mode']=row['route']=candidate if row['player_role']=='challenger' else 'frozen'
        row['quota']=128
    return initial,rows,ending,spec


def stub_outer_layers(monkeypatch):
    monkeypatch.setattr(audit,'profile_identity',lambda *args,**kwargs:{})
    monkeypatch.setattr(audit,'validate_result',lambda *args,**kwargs:None)
    monkeypatch.setattr(audit,'validate_boundary_row',lambda *args,**kwargs:{})
    monkeypatch.setattr(audit,'check_clock',lambda receipt:receipt['receipt_elapsed_seconds'])


@pytest.mark.parametrize('candidate,stage',[('low','development'),('high','development'),
                                          ('low','confirmation'),('high','confirmation')])
def test_actual_dictionary_ledger_replays_saved_corrected_transition_chain(saved_corrected_game,monkeypatch,candidate,stage):
    stub_outer_layers(monkeypatch)
    initial,rows,ending,spec=reroute_fixture(saved_corrected_game,candidate,stage)
    result=audit.audit_game(initial,rows,ending,spec,freeze={})
    assert result['status']=='passed' and result['transitions']==54
    assert result['scientific_arms']['challenger']=={'arm':'search','quota':128,'mode':candidate}
    assert result['scientific_arms']['opponent']=={'arm':'search','quota':128,'mode':'frozen'}


@pytest.mark.parametrize('mutation',['transition','candidate','route','quota','allowance','winner','omission'])
def test_fullgame_ledger_and_identity_reject_counterexamples(saved_corrected_game,monkeypatch,mutation):
    stub_outer_layers(monkeypatch)
    initial,rows,ending,spec=reroute_fixture(saved_corrected_game)
    if mutation=='transition':rows[0]['after']['state']['bank_tokens']['gold']+=1
    elif mutation=='candidate':rows[0]['candidate']='low'
    elif mutation=='route':rows[0]['mode']='low'
    elif mutation=='quota':rows[0]['quota']=64
    elif mutation=='allowance':rows[0]['clock']['deadline_seconds']=9.9
    elif mutation=='winner':ending['winner']=1-ending['winner']
    elif mutation=='omission':rows.pop()
    with pytest.raises(ValueError):audit.audit_game(initial,rows,ending,spec,freeze={})


def test_replay_cells_stable_outcome_blind_and_candidate_specific(saved_corrected_game):
    _,rows,_,_=reroute_fixture(saved_corrected_game)
    first=audit.select_replay_sample(iter(rows),stage='development',candidate='high')
    assert 0<len(first['decisions'])<=48
    assert first['selection_uses_outcomes'] is first['selection_uses_timing'] is False
    changed=deepcopy(rows)
    for row in changed:
        row['clock']['receipt_elapsed_seconds']=999
        row['winner']=1
        row['trace']={}
    second=audit.select_replay_sample(reversed(changed),stage='development',candidate='high')
    assert first==second
    assert len(first['decisions'])+len(first['empty_cells'])==48
    with pytest.raises(ValueError):audit.select_replay_sample(rows,stage='development',candidate='low')
    with pytest.raises(ValueError):audit.select_replay_sample(rows,maximum=49,stage='development',candidate='high')
    with pytest.raises(ValueError):audit.select_replay_sample(rows+rows[:1],stage='development',candidate='high')


def simple_row(candidate,role,elapsed,seat=0):
    return dict(stage='development',candidate=candidate,pair_index=0,game_index=0,ply=0,
                player_role=role,seat=seat,mode=candidate if role=='challenger' else 'frozen',
                arm='search',quota=128,placement_mode='isolated',observation_mode='profiled',
                completed=True,clock=dict(status='ok',receipt_elapsed_seconds=elapsed,
                                          elapsed_seconds=elapsed+.1,request_id=1,telemetry={}))


def test_scalar_timing_single_pass_routes_and_exact_descriptive_boundaries(monkeypatch):
    monkeypatch.setattr(timing,'_clock',lambda receipt:receipt['receipt_elapsed_seconds'])
    rows=[simple_row('low','challenger',.7),simple_row('high','challenger',.9),
          simple_row('low','opponent',1.),simple_row('high','opponent',1.001)]
    result=timing.performance(iter(rows))
    assert result['rows']==4 and set(result['players'])=={'low','high','frozen'}
    assert result['players']['low']['completed_over_700ms']==0
    assert result['players']['high']['completed_over_900ms']==0
    assert result['players']['frozen']['completed_over_one_second']==1
    assert result['players']['frozen']['completed_at_or_above_one_second']==2
    assert result['players']['frozen']['quota']==128 and result['players']['frozen']['cpuct']==.4
    groups=result['by_stage_candidate_and_seat']
    assert next(group for group in groups if (group['stage'],group['candidate'],group['player_role'],group['seat'])==
                ('development','low','opponent',0))['completed_decisions']==1
    assert result['descriptive_thresholds_change_completion_or_stopping'] is False


def test_boundary_summary_single_pass_separates_candidate_context(monkeypatch):
    monkeypatch.setattr(timing,'_clock',lambda receipt:receipt['receipt_elapsed_seconds'])
    monkeypatch.setattr(timing,'validate_boundary_row',lambda row:dict(whole_handler_ns=500))
    rows=[simple_row('low','opponent',1.),simple_row('high','opponent',1.001)]
    result=timing.summarize_boundary_rows(iter(rows))
    assert result['rows']==2 and len(result['groups'])==2
    assert {group['candidate'] for group in result['groups']}=={'low','high'}
    assert result['causal_bottleneck_proven'] is False


@pytest.mark.parametrize('field,value',[('candidate','unknown'),('mode','research'),('quota',64),
                                       ('placement_mode','shared'),('seat',True)])
def test_descriptive_timing_never_accepts_wrong_player_identity(monkeypatch,field,value):
    monkeypatch.setattr(timing,'_clock',lambda receipt:receipt['receipt_elapsed_seconds'])
    row=simple_row('high','challenger',.7);row[field]=value
    with pytest.raises(ValueError):timing.performance([row])


def test_phase_prefix_requires_selected_branch_and_keeps_operational_stops():
    specs=schedule.make_schedule()
    first=dict(specs[0],complete=True,reason='natural',plies=1)
    assert audit.validate_stopped_prefix(specs,[first])['status']=='verified_prefix'
    wrong=deepcopy(first);wrong['candidate']='high'
    with pytest.raises(ValueError):audit.validate_stopped_prefix(specs,[wrong])
    combined=specs+schedule.make_schedule('confirmation','low')
    assert audit.validate_stopped_prefix(combined,[],selected_candidate='low')['planned_games']==612
    with pytest.raises(ValueError):audit.validate_stopped_prefix(combined,[])

