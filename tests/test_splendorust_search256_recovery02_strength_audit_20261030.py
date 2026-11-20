"""Layered dictionary-ledger fixtures; numerical/profile/clock layers explicitly stubbed.

Historical corrected transitions are not claimed to be new256 model decisions.
Actual entrypoint/model qualification remains a separate worker requirement.
"""
from copy import deepcopy
import json
from itertools import product
from pathlib import Path
import pytest
from tools.splendorust_search256_recovery_20261030_02 import audit, schedule
ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'outputs/splendorust_search128_strength_20261028_01/games/main/p016_g33'


@pytest.fixture(scope='module')
def saved_corrected_game():
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    return (read(SAVED/'initial.json')['initial'],
            [read(path) for path in sorted((SAVED/'moves').glob('*.json'))],
            read(SAVED/'result.json'))


def reroute_fixture(saved,candidate='research',stage='main'):
    initial,rows,ending=deepcopy(saved)
    spec=next(row for row in schedule.make_schedule(stage,candidate)
              if row['challenger_seat']==ending['challenger_seat'])
    ending.update(spec)
    for row in rows:
        for key in ('stage','candidate','pair_index','game_index'):row[key]=spec[key]
        row['mode']=row['route']=candidate if row['player_role']=='challenger' else 'frozen'
        row['quota']=256 if row['player_role']=='challenger' else 128
    return initial,rows,ending,spec


def stub_outer_layers(monkeypatch):
    monkeypatch.setattr(audit,'profile_identity',lambda *args,**kwargs:{})
    monkeypatch.setattr(audit,'validate_result',lambda *args,**kwargs:None)
    monkeypatch.setattr(audit,'validate_boundary_row',lambda *args,**kwargs:{})
    monkeypatch.setattr(audit,'check_clock',lambda receipt:receipt['receipt_elapsed_seconds'])


@pytest.mark.parametrize('candidate,stage',[('research','operational'),('research','main')])
def test_actual_dictionary_ledger_replays_saved_corrected_transition_chain(saved_corrected_game,monkeypatch,candidate,stage):
    stub_outer_layers(monkeypatch)
    initial,rows,ending,spec=reroute_fixture(saved_corrected_game,candidate,stage)
    result=audit.audit_game(initial,rows,ending,spec,freeze={})
    assert result['status']=='passed' and result['transitions']==54
    assert result['scientific_arms']['challenger']=={'arm':'search','quota':256,'mode':candidate}
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
    first=audit.select_replay_sample(iter(rows),stage='main',candidate='research')
    assert 0<len(first['decisions'])<=48
    assert first['selection_uses_outcomes'] is first['selection_uses_timing'] is False
    changed=deepcopy(rows)
    for row in changed:
        row['clock']['receipt_elapsed_seconds']=999
        row['winner']=1
        row['trace']={}
    second=audit.select_replay_sample(reversed(changed),stage='main',candidate='research')
    assert first==second
    assert len(first['decisions'])+len(first['empty_cells'])==48
    with pytest.raises(ValueError):audit.select_replay_sample(rows,stage='main',candidate='unknown')
    with pytest.raises(ValueError):audit.select_replay_sample(rows,maximum=49,stage='main',candidate='research')
    with pytest.raises(ValueError):audit.select_replay_sample(rows+rows[:1],stage='main',candidate='research')


def test_fixed_prefix_keeps_operational_stops_and_no_conditional_branch():
    specs=schedule.make_schedule()
    first=dict(specs[0],complete=True,reason='natural',plies=1)
    assert audit.validate_stopped_prefix(specs,[first])['status']=='verified_prefix'
    assert audit.validate_stopped_prefix(specs,[])['planned_games']==406
    wrong=deepcopy(first);wrong['candidate']='high'
    with pytest.raises(ValueError):audit.validate_stopped_prefix(specs,[wrong])
    with pytest.raises(ValueError):audit.validate_stopped_prefix(specs[:-1],[])
    stopped=dict(first,reason='forfeit')
    with pytest.raises(ValueError):audit.validate_stopped_prefix(specs,[stopped,dict(specs[1],complete=True,reason='natural',plies=1)])
    allgames=[dict(spec,complete=True,reason='forfeit' if spec['stage']=='main' else 'natural',plies=1) for spec in specs]
    result=audit.validate_stopped_prefix(specs,allgames)
    assert result['complete_planned_workload'] and result['forfeits']==400
    assert result['continuation_after_forfeit']=='main_only'


def test_selector_fills_all48_frozen_cells_with_no_timing_or_outcome_choice():
    rows=[]
    for pair,(role,phase,seat,private) in enumerate(product(('challenger','opponent'),range(3),range(2),range(4))):
        desired=seat if role=='challenger' else 1-seat
        spec=next(s for s in schedule.make_schedule('main') if s['pair_index']==pair and s['challenger_seat']==desired)
        own,other=bool(private&1),bool(private&2)
        players=[dict(reserves=[]),dict(reserves=[])]
        if own:players[seat]['reserves']=[dict(source='deck')]
        if other:players[1-seat]['reserves']=[dict(source='deck')]
        rows.append(dict(stage='main',candidate='research',pair_index=pair,game_index=spec['game_index'],ply=phase*20,
            player_role=role,seat=seat,mode='research' if role=='challenger' else 'frozen',arm='search',
            quota=256 if role=='challenger' else 128,placement_mode='isolated',completed=True,
            observer=dict(observation=dict(current_player=seat,players=players))))
    selected=audit.select_replay_sample(iter(rows),stage='main',candidate='research')
    assert len(selected['decisions'])==48 and not selected['empty_cells']
    assert selected==audit.select_replay_sample(reversed(rows),stage='main',candidate='research')
    failed=deepcopy(rows[0]);failed['completed']=False
    assert selected==audit.select_replay_sample(rows+[failed],stage='main',candidate='research')
    with pytest.raises(ValueError):audit.select_replay_sample(rows,stage='operational',candidate='research')
