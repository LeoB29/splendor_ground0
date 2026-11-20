"""Same sample, two preserved segments and advisory admission; no models."""
from copy import deepcopy
from pathlib import Path
import pytest
from tools.splendorust_selfplay_pilot_resume2_20261103 import common as c,preparation as p,design as d
from tools.splendorust_selfplay_pilot_resume2_20261103 import continuation as x,forecast_preservation as f

ROOT=c.ROOT
OLD=ROOT/'outputs/splendorust_selfplay_pilot_20261102_02'
PREVIOUS=ROOT/'outputs/splendorust_selfplay_pilot_resume_20261103_01'


@pytest.fixture
def paths(monkeypatch):
    monkeypatch.setattr(x,'PREVIOUS_OUT',PREVIOUS,raising=False)
    monkeypatch.setattr(x,'previous_freeze',lambda:c.read(PREVIOUS/'freeze.json'),raising=False)
    monkeypatch.setattr(x,'prefix_game_path',lambda i:(OLD if i<32 else PREVIOUS)/'games'/f'g{i:03d}'/'game.json')


def test_entire_same_schedule_and_remaining_blocks_start_at_48():
    original=c.read(OLD/'freeze.json')
    assert d.schedule()==c.read(ROOT/original['game_schedule']['path'])
    blocks=d.remaining_collection_blocks()
    assert len(blocks)==26 and blocks[0]['game_indices']==list(range(48,56))
    assert blocks[-1]['game_indices']==list(range(248,256))
    assert len(d.collection_blocks())==32


def test_exact_authorized_extended_windows_and_no_new_origins():
    value=p.authorized_windows()
    assert value['authorized_clock_windows']==dict(job_seconds=120*3600,model_cutoff_seconds=108*3600,
        outer_seconds=124*3600,collection_seconds=72*3600,fit_seconds=8*3600,
        assessment_seconds=40*3600,reproduction_seconds=8*3600,final_seconds=12*3600)


@pytest.mark.parametrize('key,value',[('original_completed_games_preserved',32),('amended_clock_windows',False),
    ('actual_original_clock_origins_retained',False),('science_unchanged',False),('failed_decision_retries',1),
    ('job_seconds',121*3600),('model_cutoff_seconds',109*3600),('collection_seconds',73*3600)])
def test_changed_authorization_rejected(monkeypatch,key,value):
    original=c.read
    def read(path):
        result=original(path)
        if Path(path)==c.OUT/'prelaunch/user_authorization.json':result={**result,key:value}
        return result
    monkeypatch.setattr(c,'read',read)
    with pytest.raises(ValueError,match='authorized48-game'):p.authorized_windows()


def test_advisory_projection_uses_48_mixed_costs_and_7_successful_workers(paths,monkeypatch,tmp_path):
    stage=c.read(OLD/'stages/collection_start.json')
    stage={**stage,'monotonic_deadline':stage['monotonic']+72*3600,'unix_deadline':stage['unix_time']+72*3600}
    manifest=dict(original_monotonic_origin=904900.656,original_unix_origin=1793649048.764625,
        continuation_collection=stage,prior_evidence_bytes=123456,prior_ram_wait_seconds=60.093)
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(x,'MANIFEST',tmp_path/'manifest.json')
    original=c.read
    monkeypatch.setattr(c,'read',lambda path:deepcopy(manifest) if Path(path)==x.MANIFEST else original(path))
    captured=[]
    def project(prepared,snapshot,source_costs):
        captured.append((snapshot,source_costs))
        return dict(snapshot=snapshot,affordable=False,checks={'total':False,'collection':False})
    monkeypatch.setattr(d,'source_forecast',project)
    result=p.provisional_projection({})
    assert not result['affordable'] and result['forecast_advisory'] and not result['forecast_can_stop']
    snapshot,costs=captured[0]
    assert snapshot['completed_games']==48 and snapshot['qualified_source_workers']==7
    assert len(costs)==48 and costs[31]==original(OLD/'games/g031/cost.json') and costs[32]==original(PREVIOUS/'games/g032/cost.json')
    assert snapshot['stage_monotonic_deadline']==stage['monotonic_deadline']
    assert snapshot['original_monotonic_origin']==904900.656


def test_all_seven_preserved_forecasts_recomputed_without_new_gate(paths):
    result=f.evaluate_original_forecasts()
    assert result['approved'] and result['original_attempt_remains_incomplete'] and result['previous_attempt_remains_incomplete']
    assert len(result['forecasts'])==7
    assert [row['affordable'] for row in result['forecasts']]==[True,True,True,True,False,False,False]
    assert result['forecasts'][-1]['policy']=='advisory_only_in_previous_continuation'


@pytest.mark.parametrize('what',['arithmetic','cost_binding'])
def test_previous_forecast_corruption_rejected_even_when_advisory(paths,monkeypatch,what):
    original=c.read
    def changed(path):
        result=original(path)
        if Path(path)==PREVIOUS/'block_forecasts/005.json':
            result=deepcopy(result)
            if what=='arithmetic':result['projected_model_seconds']+=1
            else:result['cost_inputs'][32]=result['cost_inputs'][31]
        return result
    monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError,match='arithmetic|mixed-prefix'):f.evaluate_original_forecasts()


def test_preparation_refuses_both_preserved_output_trees(monkeypatch):
    for out in [OLD,PREVIOUS]:
        monkeypatch.setattr(c,'OUT',out)
        with pytest.raises(ValueError,match='New pilot continuation'):p.prepare()
