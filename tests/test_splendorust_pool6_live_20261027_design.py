"""Model-free regressions for phase separation and paired inference."""
from copy import deepcopy
import os
import subprocess
import sys
import pytest
from tools.splendorust_pool6_live_20261027 import schedule, analysis, design
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules


def results():
    games=[]
    for spec in schedule.stage_schedule('main'):
        pair=spec['pair_index'];seat=spec['challenger_seat']
        winner=seat if pair<60 else 1-seat
        games.append(dict(spec,winner=winner,reason='natural',loser=None,complete=True))
    return games


def test_schedule_fresh_phase_seat_pairs():
    ops,main=map(schedule.stage_schedule,('operational','main'))
    assert len(ops)==6 and len(main)==200
    assert set(x['seed_decimal'] for x in ops).isdisjoint(x['seed_decimal'] for x in main)
    assert len({x['seed_decimal'] for x in main})==100
    for a,b in zip(main[::2],main[1::2]):
        assert a['seed_decimal']==b['seed_decimal'] and a['challenger_seat']!=b['challenger_seat']
        assert a['challenger_quota']==a['opponent_quota']==64
        assert a['challenger_mode']=='research' and a['opponent_mode']=='frozen'


def test_assessment_startup_is_separate_process_phase():
    env=os.environ.copy();env['SPLENDOR_POOL6_PHASE']='assessment'
    code=("from tools.splendorust_pool6_live_20261027 import common as c,preparation,runner,schedule,adapters; "
          "runner.validate_freeze_contract(preparation.contract()); "
          "assert c.OUT==c.ASSESS_OUT and c.JOB_SECONDS==43200 and c.MAX_WORKERS==448; "
          "assert len(schedule.make_schedule())==200 and all(x['stage']=='main' for x in schedule.make_schedule()); "
          "assert adapters.profile_path(64,6).is_relative_to(c.QUAL_OUT)")
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('field,value',[('challenger_quota',32),('opponent_mode','baseline'),
    ('challenger_seat',True),('seed_decimal','0'),('pair_index',True)])
def test_schedule_rejects_identity_changes(field,value):
    spec=schedule.stage_schedule('main')[0];spec[field]=value
    with pytest.raises(ValueError):schedule.validate_spec(spec)


def test_no_partial_duplicate_or_operational_primary():
    games=results()
    with pytest.raises(ValueError):analysis.validated_games(games[:-1])
    changed=deepcopy(games);changed[1]=changed[0]
    with pytest.raises(ValueError):analysis.validated_games(changed)
    changed=deepcopy(games);changed[0].update(schedule.stage_schedule('operational')[0])
    with pytest.raises(ValueError):analysis.validated_games(changed)


def test_timeout_is_loss_and_cannot_relabel_winner():
    games=results();games[0].update(reason='forfeit',loser=games[0]['challenger_seat'],
                                 winner=1-games[0]['challenger_seat'])
    assert analysis.validated_games(games)[0]['search_score_halfpoints']==0
    games[0]['winner']=games[0]['loser']
    with pytest.raises(ValueError):analysis.validated_games(games)


def test_full_analysis_independent_replication_and_integrity_gate():
    games=results()
    with pytest.raises(ValueError):analysis.analyze_games(games,False)
    answer=analysis.analyze_games(games,True)
    assert answer['search_mean_score']==.6 and answer['reference_simulations']==64
    assert answer['challenger_world_pool']==6 and answer['reference_world_pool']==3
    assert answer['by_search_seat']=={'0':.6,'1':.6}
    assert analysis.verify_analysis(games,answer)['status']=='passed'
    answer['search_losses']-=1
    with pytest.raises(ValueError):analysis.verify_analysis(games,answer)


def test_reserved_panel_actual_corrected_decoder_and_support():
    panel=design.build_reserved_panel()
    assert len(panel['positions'])==18 and panel['model_calls']==0
    assert len({x['seed_decimal'] for x in panel['positions']})==18
    for row in panel['positions']:
        state=guard.decode_training_snapshot(row['before_snapshot'])
        assert state.current_player==row['seat'] and state.turn_index==row['target_ply']
        assert rules.observation_payload(state)==row['observer']
        assert rules.CorrectedEnv().legal_actions(state)
        with pytest.raises(ValueError):guard.decode_training_snapshot(row['before_snapshot']['state'])
