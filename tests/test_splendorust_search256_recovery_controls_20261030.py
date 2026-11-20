"""Model-free checks for the authorized recovery's non-resettable boundaries."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.splendorust_search256_recovery_20261030 import common as c,design,guardian,preparation,adapters


def test_original_main_deadline_clamps_new_qualification_window():
    start=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE-100
    wall=c.ORIGINAL_MAIN_UNIX_DEADLINE-100
    assert guardian.qualification_clock_open(start,wall,start+99,wall+99,qualification_finished=False)
    assert not guardian.qualification_clock_open(start,wall,start+100,wall+99,qualification_finished=False)
    assert not guardian.qualification_clock_open(start,wall,start+99,wall+100,qualification_finished=False)


def test_recovery_qualification_two_hour_cap_not_historical_qualification_reset():
    start=c.PRIOR_ORIGIN+14*3600;wall=c.PRIOR_WALL_START+14*3600
    assert guardian.qualification_clock_open(start,wall,start+7199,wall+7199,qualification_finished=False)
    assert not guardian.qualification_clock_open(start,wall,start+7200,wall+7199,qualification_finished=False)
    assert not guardian.qualification_clock_open(start,wall,start+7199,wall+7200,qualification_finished=False)


def test_model_work_cannot_continue_beyond_original_model_cutoff():
    start,wall=c.PRIOR_ORIGIN,c.PRIOR_WALL_START
    assert not guardian.model_clock_open(start,wall,start+c.COLLECTION_CUTOFF_SECONDS,wall,final_started=False,owned_children=[])
    assert not guardian.model_clock_open(start,wall,start,wall+c.COLLECTION_CUTOFF_SECONDS,final_started=False,owned_children=[])
    with pytest.raises(ValueError):
        guardian.model_clock_open(start,wall,start,wall,final_started=True,owned_children=[123])


def test_only_remaining_games_can_create_game_workers():
    inventory=design.lifecycle_inventory()
    games={k:v for k,v in inventory.items() if k.startswith('main_')}
    assert len(games)==330
    assert not any(f'_g{i}_' in name for i in range(235) for name in games)
    for role,route in [('challenger','research'),('opponent','frozen')]:
        row=games[f'main_research_g235_{role}']
        assert row['route']==route
        assert row['entrypoint_request_id']=='request-00000002'
        assert row['ordinary_first_request_id']=='request-00000003'
    assert games['main_research_g236_challenger']['entrypoint_request_id'] is None
    # 96 permitted route alternatives are 48 actual slots, never96 actual trees.
    assert len(inventory)-len(games)==96
    assert c.MAX_WORKERS==330+48
    assert c.PRIOR_WORKER_TREES+c.MAX_WORKERS==866


def test_original_profile_bytes_and_paths_retained():
    old=c.read(c.OLD_OUT/'freeze.json')
    for route in ('research','frozen'):
        for quota,descriptor in old['profiles'][route].items():
            assert c.artifact(adapters.profile_path(int(quota),route=route))==descriptor
            assert c.read(c.ROOT/descriptor['path'])==adapters.profile(int(quota),route=route)


def test_resource_contract_charges_prior_work_without_weakening_original_limits():
    new=preparation.contract();old=c.read(c.OLD_OUT/'freeze.json')
    assert new['max_evidence_bytes']+c.PRIOR_EVIDENCE_BYTES==24*c.GiB
    assert new['prefinal_evidence_bytes']+c.PRIOR_EVIDENCE_BYTES==16*c.GiB
    assert new['prior_ram_wait_seconds']==195.54899999988265
    for key in ('job_seconds','collection_cutoff_seconds','max_rss_bytes',
        'initial_available_ram_bytes','minimum_available_ram_bytes','maximum_ram_wait_seconds',
        'seconds_per_move','internal_seconds','research_seconds_per_move','research_internal_seconds',
        'maximum_reproductions','main_games','initial_free_disk_bytes','minimum_free_disk_bytes'):
        assert new[key]==old[key]
    assert new['automatic_retries']==0 and new['authorized_interrupted_decision_retries']==1
    assert not new['one_second_gate'] and not new['seven_hundred_ms_gate']


def test_scientific_workers_are_namespace_only_copies():
    root=c.ROOT/'tools';old=root/'splendorust_search256_strength_20261029_02'
    new=root/'splendorust_search256_recovery_20261030'
    for name in ('worker.py','scientific.py','corpus.py','qualification.py','placement.py','schedule.py','performance.py'):
        expected=(old/name).read_bytes().replace(b'splendorust_search256_strength_20261029_02',b'splendorust_search256_recovery_20261030')
        assert (new/name).read_bytes()==expected,name


def test_manifest_rejects_retry_expansion_and_deadline_reset():
    path=c.OUT/'recovery_manifest.json'
    if not path.exists():
        pytest.skip('Prepared immutable manifest not yet written')
    original=c.read(path)
    assert preparation.validate_manifest(original)
    for key,value in [('authorized_retries',2),('continuation_ply',31),('completed_main_games',234),
                      ('main_unix_deadline',original['main_unix_deadline']+1),
                      ('prior_ram_wait_seconds',0),('maximum_combined_worker_trees',868)]:
        changed=deepcopy(original);changed[key]=value
        with pytest.raises(ValueError):preparation.validate_manifest(changed)
    changed=deepcopy(original);changed['retry']['old_request_id']='request-00000016'
    with pytest.raises(ValueError):preparation.validate_manifest(changed)
