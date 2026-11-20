"""Model-free preparation safety and genuinely fresh corrected positions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.splendorust_search128_strength_20261028 import common as c, design, preparation, schedule, runner
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules


def test_reserved_positions_are_corrected_disjoint_and_seat_balanced():
    panel=c.read(design.PANEL)
    seeds=[];cells={}
    for position in panel['positions']:
        state=guard.decode_training_snapshot(position['before_snapshot'])
        assert rules.observation_payload(state)==position['observer']
        assert state.current_player==position['seat']
        assert state.turn_index==position['target_ply'] and not state.terminal
        seeds.append(position['seed_decimal'])
        cell=(position['target_ply']//12,position['seat'])
        cells[cell]=cells.get(cell,0)+1
    game_seeds={row['seed_decimal'] for row in schedule.make_schedule()}
    assert len(seeds)==len(set(seeds))==18
    assert not set(seeds)&game_seeds
    assert len(cells)==6 and set(cells.values())=={3}


def test_preparation_contract_is_accepted_without_relaxing_review_budget():
    contract=preparation.contract()
    runner.validate_freeze_contract(contract)
    for wrong in (None,900,7200):
        damaged=copy.deepcopy(contract);damaged['independent_review_maximum_seconds']=wrong
        with pytest.raises(ValueError,match='Exact frozen'):
            runner.validate_freeze_contract(damaged)


def test_existing_prepared_artifact_cannot_be_replaced(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'ROOT',tmp_path)
    target=tmp_path/'plan.json'
    design.fixed(target,{'fixed':1})
    before=target.read_bytes()
    design.fixed(target,{'fixed':1})
    with pytest.raises(ValueError,match='must not change'):
        design.fixed(target,{'fixed':2})
    assert target.read_bytes()==before


def test_registered_seed_collision_is_rejected_without_replacement(tmp_path,monkeypatch):
    # A collision in another registered study is an error, never seed selection.
    panel=c.read(design.PANEL);frozen_schedule=schedule.make_schedule()
    root=tmp_path;out=root/'new_attempt';out.mkdir()
    previous=root/'old_schedule.json'
    previous.write_text(json.dumps({'seed_decimal':panel['positions'][0]['seed_decimal']}))
    panel_path=out/'reserved_panel.json';panel_path.write_text(json.dumps(panel))
    monkeypatch.setattr(c,'ROOT',root);monkeypatch.setattr(c,'OUT',out)
    monkeypatch.setattr(design,'PANEL',panel_path)
    monkeypatch.setattr(schedule,'stage_schedule',lambda stage:[x for x in frozen_schedule if x['stage']==stage])
    monkeypatch.setattr(preparation.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='old_schedule.json\n'))
    before=panel_path.read_bytes()
    with pytest.raises(ValueError,match='Fresh seed collision'):
        preparation.seed_inventory()
    assert panel_path.read_bytes()==before


def test_research_profiles_preserve_every_scientific_field_except_quota():
    from tools.splendorust_search128_strength_20261028 import adapters
    from tools.splendorust_search_scale_20261024 import common as original
    baseline=c.read(original.profile_path(64))
    for quota in (64,128):
        profile=adapters.profile(quota)
        for key,value in baseline.items():
            if key in ('profile_id','wrapper_id','internal_budget_seconds','iterations'):continue
            assert profile[key]==value
        assert profile['iterations']==quota and profile['world_pool']==3
        assert profile['internal_budget_seconds']==9.
        assert profile['execution_policy']['changed_scientific_fields']==(['iterations'] if quota==128 else [])
