"""Exact total, swapped seats, fresh namespaces and immutable schedule fields."""
from copy import deepcopy
import hashlib
import json
import pytest

from tools.splendorust_isolated_reliability_20261026 import schedule as s
from tools.splendorust_live_boundary_20261025 import schedule as previous
from tools.splendorust_supervisor_qualification_20261026 import preparation as qualification


def test_exact200_includes_six_operational_then194main():
    rows=s.make_schedule()
    assert len(rows)==s.TOTAL_GAMES==200
    assert [row['stage'] for row in rows]==['operational']*6+['main']*194
    assert rows[:6]==s.stage_schedule('operational')
    assert rows[6:]==s.stage_schedule('main')
    assert rows[-1]['game_index']==193 and rows[-1]['pair_index']==96
    assert s.validate_schedule(rows)==rows


def test_each_of100_pairs_has_same_fresh_seed_and_swapped_seats():
    rows=s.make_schedule();seeds=set()
    for first,second in zip(rows[::2],rows[1::2],strict=True):
        assert first['stage']==second['stage'] and first['pair_index']==second['pair_index']
        assert first['seed_decimal']==second['seed_decimal']
        assert first['challenger_seat']==1-second['challenger_seat']
        assert first['seed_decimal'] not in seeds
        seeds.add(first['seed_decimal'])
    assert len(seeds)==s.TOTAL_PAIRS==100
    prior={row['seed_decimal'] for row in previous.make_schedule()+qualification.operational_schedule()}
    assert seeds.isdisjoint(prior)
    assert s.DOMAIN=='splendorust_isolated_reliability_1s_20261026_v1'
    encoded=json.dumps([s.DOMAIN,'main',0],separators=(',',':'),ensure_ascii=True)
    assert rows[6]['seed_decimal']==str(int.from_bytes(hashlib.sha256(encoded.encode()).digest(),'big'))


@pytest.mark.parametrize('field,value',[('challenger_quota',32),('opponent_quota',64),
    ('challenger_mode','baseline'),('opponent_mode','candidate'),('challenger_seat',True),
    ('seed_decimal','7'),('pair_index',True),('game_index',True)])
def test_scientific_player_and_schedule_identity_are_fixed(field,value):
    row=deepcopy(s.make_schedule()[0]);row[field]=value
    with pytest.raises(ValueError):s.validate_spec(row)


def test_omission_duplication_reorder_extra_fields_and206_total_rejected():
    rows=s.make_schedule()
    variants=[rows[:-1],rows+[deepcopy(rows[0])],list(reversed(rows)),rows[:-1]+[deepcopy(rows[0])],
        s.stage_schedule('operational')+previous.stage_schedule('main')]
    extra=deepcopy(rows);extra[0]['unexpected']=True;variants.append(extra)
    for variant in variants:
        with pytest.raises(ValueError):s.validate_schedule(variant)


@pytest.mark.parametrize('stage,pair',[('main',97),('operational',3),('other',0),('main',-1),('main',True)])
def test_stage_pair_bounds_are_explicit(stage,pair):
    with pytest.raises(ValueError):s.expected_seed(stage,pair)
