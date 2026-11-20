"""Saved-only continuation contract regressions; no model or game calls."""
import copy
from pathlib import Path
import pytest
from tools.splendorust_selfplay_pilot_resume_20261103 import continuation as x
from tools.splendorust_selfplay_pilot_resume_20261103 import common as c, design, preparation
from tools.splendorust_selfplay_pilot_resume_20261103 import forecast_preservation


@pytest.mark.parametrize('bad', [-1,32,True,1.0,'0',None])
def test_prefix_path_rejects_uncompleted_or_untyped_indices(bad):
    with pytest.raises(ValueError):x.prefix_game_path(bad)


def test_prefix_paths_and_source_worker_ownership_are_original():
    assert x.prefix_game_path(0)==x.OLD_OUT/'games/g000/game.json'
    assert x.prefix_game_path(31)==x.OLD_OUT/'games/g031/game.json'
    for name in x.PREFIX_NAMES:assert x.prefix_source_root(name)==x.OLD_OUT
    for bad in ('collect_004','replay_on','../capture_on'):
        with pytest.raises(ValueError):x.prefix_source_root(bad)


def test_actual_saved_resource_stop_has_complete_clean_prefix():
    value=x.verify_stop()
    assert value['original_completed_games']==32
    assert value['prior_source_lifecycles']==5
    assert value['original_monotonic_origin']==904900.656
    assert value['original_unix_origin']==1793649048.764625
    assert value['prior_ram_wait_seconds']==pytest.approx(60.078)


def test_actual_saved_schedule_seeds_and_source_declaration_unchanged(monkeypatch):
    monkeypatch.setattr(c,'SCOPE','pilot')
    monkeypatch.setattr(c,'MAX_SOURCE_GAMES',256)
    monkeypatch.setattr(c,'SOURCE_GAMES_PER_WORKER',8)
    old=x.old_freeze();declaration=x.source_continuation()
    assert design.schedule()==c.read(c.ROOT/old['game_schedule']['path'])
    assert declaration['original_schedule']==old['game_schedule']
    assert declaration['original_freeze']['sha256']==x.OLD_FREEZE_SHA
    assert len(declaration['games'])==32
    assert design.remaining_collection_blocks()[0]['game_indices']==list(range(32,40))
    assert design.remaining_collection_blocks()[-1]['game_indices']==list(range(248,256))
    assert len(design.remaining_collection_blocks())==28


def test_continuation_declines_wrong_original_status(monkeypatch):
    original=c.read
    def changed(path):
        result=original(path)
        if Path(path)==x.OLD_OUT/'completion.json':
            result=copy.deepcopy(result);result['status']='completed_verified'
        return result
    monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError,match='clean resource-forecast'):x.verify_stop()


def test_continuation_declines_missing_completed_prefix(monkeypatch):
    original=c.read
    def changed(path):
        result=original(path)
        if Path(path)==x.OLD_OUT/'completion.json':
            result=copy.deepcopy(result);result['completed_games']=31
        return result
    monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError,match='clean resource-forecast'):x.verify_stop()


def test_continuation_declines_unclean_owner_exit(monkeypatch):
    original=c.read
    def changed(path):
        result=original(path)
        if Path(path)==x.OLD_OUT/'execution.json':
            result=copy.deepcopy(result);result['cleanup_errors']=['unconfirmed identity']
        return result
    monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError,match='cleanup'):x.verify_stop()


def test_reused_pid_is_distinguished_from_exact_old_owner(monkeypatch):
    class P:
        def __init__(self,pid):pass
        def create_time(self):return 2.
    monkeypatch.setattr(x.psutil,'Process',P)
    x.require_absent([dict(pid=123,created=1.)])
    with pytest.raises(ValueError,match='remains alive'):x.require_absent([dict(pid=123,created=2.)])


def test_saved_guard_does_not_invent_new_global_clock(monkeypatch):
    monkeypatch.setattr(x.time,'monotonic',lambda:904900.656+72*3600+1)
    monkeypatch.setattr(x.time,'time',lambda:1793649048.764625+72*3600+1)
    guard=x.SavedGuard(1200)
    with pytest.raises(ValueError,match='original global deadline'):guard.guard()


def test_prepare_refuses_original_output_before_any_write(monkeypatch):
    monkeypatch.setattr(c,'SCOPE','pilot');monkeypatch.setattr(c,'OUT',x.OLD_OUT)
    with pytest.raises(ValueError,match='new pilot continuation output'):x.prepare_continuation()
    with pytest.raises(ValueError,match='New pilot continuation'):preparation.prepare()


def test_advisory_forecast_keeps_original_failed_result(monkeypatch):
    original=c.read
    saved=original(x.OLD_OUT/'block_forecasts/003.json')
    manifest={**x.verify_stop(),'prior_evidence_bytes':1686728956}
    def changed(path):
        if Path(path)==x.MANIFEST:return manifest
        if Path(path)==c.OUT/'prelaunch_resource_projection.json':return {'snapshot':saved['snapshot']}
        return original(path)
    monkeypatch.setattr(c,'read',changed)
    monkeypatch.setattr(c,'SCOPE','pilot')
    monkeypatch.setattr(c,'OUT',x.NEW_OUT)
    monkeypatch.setattr(Path,'exists',lambda self:True if self==c.OUT/'prelaunch_resource_projection.json' else Path.is_file(self))
    monkeypatch.setattr(design,'source_forecast',lambda prepared,snapshot,source_costs:{'affordable':False,'snapshot':snapshot,'checks':saved['checks']})
    result=preparation.provisional_projection({'resource_plan':saved['resource_plan']})
    assert result['affordable'] is False
    assert result['checks']['collection'] is False
    assert result['forecast_advisory'] is True and result['forecast_can_stop'] is False
    assert result['snapshot']==saved['snapshot']


def test_original_five_forecasts_recompute_without_laundering_failure():
    result=forecast_preservation.evaluate_original_forecasts()
    assert result['approved']
    assert len(result['forecasts'])==5
    assert [x['affordable'] for x in result['forecasts']]==[True,True,True,True,False]
    assert result['original_attempt_remains_incomplete']
    assert result['original_final_forecast_remains_unaffordable']


def test_tampered_old_forecast_is_rejected_even_when_advisory(monkeypatch):
    original=c.read
    def changed(path):
        result=original(path)
        if Path(path)==x.OLD_OUT/'block_forecasts/003.json':
            result=copy.deepcopy(result);result['affordable']=True
        return result
    monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError,match='forecast recomputation'):forecast_preservation.evaluate_original_forecasts()
