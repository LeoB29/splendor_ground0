"""Prospective independence and typed reservation; no games, models or full scan."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import pytest
from tools.splendorust_independent_data_20261107 import schedule as s


def test_fixed_counts_balance_independence_and_paired_order():
    collection, strength = s.collection_schedule(), s.strength_schedule()
    counts = s.validate(collection, strength)
    assert counts['all_independent_setups'] == 1360
    assert counts['auxiliary_seeds'] == 6
    assert len(s.narrow_ids()) == 192
    assert sum(i % 2 for i in s.narrow_ids()) == 96
    assert all(0 <= i < 1024 for i in s.narrow_ids())
    assert all(type(v) is int and 0 <= v <= 2**32-1 for v in s.FIT_SEEDS.values())
    assert {r['seed_decimal'] for r in s.qualification_schedule()}.isdisjoint(
        {r['seed_decimal'] for r in collection+strength})
    assert [r['repetition'] for r in strength[12:32:2]] == [0]*5+[1]*5


@pytest.mark.parametrize('where,field,value', [('collection','game_index',False),
    ('collection','quota',512.0),('collection','start_player',False),
    ('collection','split','offline'),('strength','candidate_simulations',128.0),
    ('strength','candidate_seat',False),('strength','seed_decimal','1'),
    ('strength','repetition',False)])
def test_exact_reservation_rejects_type_alias_and_substitution(where,field,value):
    a,b=s.collection_schedule(),s.strength_schedule()
    (a if where=='collection' else b)[0][field]=value
    with pytest.raises(ValueError): s.validate(a,b)


@pytest.mark.parametrize('bad',[False,True,0.0,'0',-1,1152])
def test_collection_exact_index(bad):
    with pytest.raises(ValueError): s.collection_spec(bad)


def test_corrected_initial_roundtrip_and_starting_seat():
    from tools.future_training_rules_20261021 import decode_training_snapshot
    for seat in (0,1):
        row=s.qualification_schedule()[seat]
        state=s._initial(row['seed_decimal'],seat)
        decoded=decode_training_snapshot(state)
        assert decoded.current_player==decoded.start_player==seat
        assert not decoded.terminal and decoded.turn_index==0


def test_inventory_rejects_collision_without_redraw(tmp_path,monkeypatch):
    target=s.collection_spec(0)['seed_decimal']
    (tmp_path/'outputs').mkdir()
    prior=tmp_path/'outputs/prior_schedule.json';prior.write_text('[{"seed_decimal":"'+target+'"}]')
    monkeypatch.setattr(s.c,'ROOT',tmp_path)
    monkeypatch.setattr(s.c,'OUT',tmp_path/'outputs/new')
    monkeypatch.setattr(s,'EXPLICIT_PRIOR',())
    monkeypatch.setattr(s.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='outputs/prior_schedule.json'))
    before=s.collection_schedule()
    with pytest.raises(ValueError,match='Registered seed collision'): s.registered_seed_inventory()
    assert s.collection_schedule()==before


def test_inventory_discloses_finite_coverage_and_skips_own(tmp_path,monkeypatch):
    (tmp_path/'outputs/new').mkdir(parents=True)
    (tmp_path/'outputs/old_schedule.json').write_text('[{"seed_decimal":"12"}]')
    (tmp_path/'outputs/new/schedule.json').write_text('[{"seed_decimal":"'+s.collection_spec(0)['seed_decimal']+'"}]')
    monkeypatch.setattr(s.c,'ROOT',tmp_path);monkeypatch.setattr(s.c,'OUT',tmp_path/'outputs/new')
    monkeypatch.setattr(s,'EXPLICIT_PRIOR',())
    monkeypatch.setattr(s.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='outputs/old_schedule.json\noutputs/new/schedule.json'))
    value=s.registered_seed_inventory()
    assert value['status']=='passed' and len(value['checked'])==1 and value['collisions']==[]
    assert value['setup_seeds']==1360 and 'Not all raw logs' in value['scope']
