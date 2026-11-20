"""Saved-only second-continuation regressions; no Torch or player calls."""
import copy
from pathlib import Path
import pytest
from tools.splendorust_selfplay_pilot_resume2_20261103 import continuation as x


@pytest.mark.parametrize('index,root',[(0,x.OLD_OUT),(31,x.OLD_OUT),(32,x.PREVIOUS_OUT),(47,x.PREVIOUS_OUT)])
def test_exact_layered_game_paths(index,root):
    assert x.prefix_game_path(index)==root/'games'/f'g{index:03d}'/'game.json'


@pytest.mark.parametrize('index',[-1,48,True,32.0,'32'])
def test_no_future_or_coerced_prefix(index):
    with pytest.raises(ValueError):x.prefix_game_path(index)


def test_failed_worker_not_a_successful_prefix():
    assert len(x.PREFIX_NAMES)==7
    assert x.prefix_source_root('capture_on')==x.OLD_OUT
    assert x.prefix_source_root('collect_005')==x.PREVIOUS_OUT
    with pytest.raises(ValueError):x.prefix_source_root('collect_006')


def test_actual_declaration_has_both_exact_schedule_layers():
    value=x.source_continuation()
    assert set(value)=={'schema','completed_games','original_freeze','original_schedule','previous_freeze','previous_schedule','games'}
    assert value['schema']=='selfplay_saved_source_prefix_v2' and value['completed_games']==48
    assert len(value['games'])==48
    assert value['games'][31]['path'].startswith(x.OLD_OUT.relative_to(x.c.ROOT).as_posix())
    assert value['games'][32]['path'].startswith(x.PREVIOUS_OUT.relative_to(x.c.ROOT).as_posix())


def test_actual_stop_clocks_and_failed_lifetime_are_explicit():
    value=x.verify_stop()
    assert value['prior_source_lifecycles']==7 and value['preserved_failed_source_lifecycles']==1
    assert value['prior_actual_source_lifecycles']==8
    assert value['original_collection']['cap_seconds']==34*3600
    assert value['continuation_collection']['cap_seconds']==72*3600
    assert value['original_job_seconds']==72*3600 and value['amended_job_seconds']==120*3600
    assert value['original_model_seconds']==60*3600 and value['amended_model_seconds']==108*3600
    assert value['original_collection']['monotonic']==value['continuation_collection']['monotonic']
    assert value['original_collection']['unix_time']==value['continuation_collection']['unix_time']
    assert value['continuation_collection']['monotonic_deadline']==min(
        value['original_collection']['monotonic']+72*3600,value['original_monotonic_origin']+108*3600)
    assert value['prior_ram_wait_seconds']>=value['ram_wait_charge']['observed_cumulative_seconds']


def _alter_read(monkeypatch,path,alter):
    original=x.c.read
    def read(p):
        value=original(p)
        if Path(p)==path:
            value=copy.deepcopy(value);alter(value)
        return value
    monkeypatch.setattr(x.c,'read',read)


@pytest.mark.parametrize('mutation',[
    lambda v:v.update(accepted=True), lambda v:v.update(status='ok'),
    lambda v:v.update(result={'fake':'science'}),lambda v:v.update(request_id='request-00000003')])
def test_failed_startup_cannot_be_relabelled(monkeypatch,mutation):
    _alter_read(monkeypatch,x.PREVIOUS_OUT/'workers/collect_006/entrypoint_clock.json',mutation)
    with pytest.raises(ValueError,match='Preserved rejected'):x.verify_stop()


def test_actual_raw_failure_is_verified_but_never_accepted():
    value=x.verify_failed_startup(x.previous_freeze(),x.SavedGuard(60))
    assert value['accepted'] is False and value['parent_requests']==2 and value['graph_cases']==11
    assert value['scientific_qualification_decisions']==value['ordinary_decisions']==0


def test_raw_failure_rejects_scientific_profile(monkeypatch):
    path=x.PREVIOUS_OUT/'workers/collect_006/request-00000002_child_profile.json'
    _alter_read(monkeypatch,path,lambda v:v.update(profiles=[{'fake':'science'}]))
    with pytest.raises(ValueError,match='No scientific suite'):x.verify_failed_startup(x.previous_freeze(),x.SavedGuard(60))


def test_raw_failure_rejects_extra_ordinary_inventory(monkeypatch):
    original=Path.glob
    folder=x.PREVIOUS_OUT/'workers/collect_006/inputs'
    def glob(path,pattern):
        if path==folder and pattern=='*.json':return iter([folder/'request-00000003.json'])
        return original(path,pattern)
    monkeypatch.setattr(Path,'glob',glob)
    with pytest.raises(ValueError,match='Exactly graph'):x.verify_failed_startup(x.previous_freeze(),x.SavedGuard(60))
