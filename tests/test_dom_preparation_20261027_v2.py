"""Version isolation, accepted-prefix authentication and fresh-world admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from tools.dom_screen_20261027_v2 import preparation as p, common as c, schedule


def test_frozen_player_serialization_and_qualification_identity_are_unchanged():
    from tools.dom_screen_20261027_v2 import qualification
    old_profile=c.read(p.DOM_PRIOR/'opponent_profile.json')
    assert c.canonical(p.profile())==c.canonical(old_profile)
    assert qualification.build_plan()==c.read(p.DOM_PRIOR/'qualification_plan.json')
    assert c.OUT!=p.DOM_PRIOR and c.OUT.name=='dom_screen_20261027_02'


def test_accepted_prefix_authenticates_finite_prior_receipts():
    result=p.verify_previous_prefix()
    assert result['accepted_verified_operational_prefix'] is True
    assert result['coverage']['main_games']==0


def test_changed_prior_summary_is_rejected(monkeypatch):
    original=c.artifact
    def changed(path):
        item=original(path)
        if Path(path)==p.DOM_PRIOR/'operational_gate.json':item=dict(item,sha256='0'*64)
        return item
    monkeypatch.setattr(c,'artifact',changed)
    with pytest.raises(ValueError,match='summary changed: operational_gate'):
        p.verify_previous_prefix()


def test_prior_acceptance_hash_cannot_be_substituted(monkeypatch):
    original=c.file_hash
    monkeypatch.setattr(c,'file_hash',lambda path:'0'*64 if Path(path)==p.DOM_PRIOR/'independent_results_review.json' else original(path))
    with pytest.raises(ValueError,match='review identity'):
        p.verify_previous_prefix()


def test_prior_dom_world_collision_rejected_and_current_schedule_excluded(tmp_path,monkeypatch):
    old=tmp_path/'outputs/dom_screen_20261027_01';new=tmp_path/'outputs/dom_screen_20261027_02'
    old.mkdir(parents=True);new.mkdir()
    plan=schedule.generate_schedule('ab'*32)
    (old/'schedule.json').write_text(json.dumps(plan),encoding='utf-8')
    (new/'schedule.json').write_text(json.dumps(plan),encoding='utf-8')
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',new);monkeypatch.setattr(p,'DOM_PRIOR',old)
    with pytest.raises(ValueError,match='overlaps known schedule'):
        p.check_known_schedules(plan)
    fresh=schedule.generate_schedule('cd'*32)
    (new/'schedule.json').write_text(json.dumps(fresh),encoding='utf-8')
    result=p.check_known_schedules(fresh)
    assert result['known_unique_seeds']==33 and result['new_unique_worlds']==33
    assert len(result['schedules'])==1
    assert result['schedules'][0]['path']=='outputs/dom_screen_20261027_01/schedule.json'
