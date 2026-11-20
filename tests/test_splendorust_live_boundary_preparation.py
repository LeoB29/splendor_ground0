from copy import deepcopy
from pathlib import Path
import pytest

from tools.splendorust_live_boundary_20261025 import common as c, preparation as p, schedule as s
from tools.splendorust_graph64_strength_20261025 import schedule as old


def valid_forecast():
    return dict(status='passed', model_calls=0, fresh_games=0,
        outcomes_used_for_selection_or_resource_estimation=False,
        projected_stage_seconds=dict(qualification=200,operational=400,main=11000,final=1300),
        projected_total_seconds=12900, projected_prefinal_bytes=1000*c.MiB,
        projected_final_bytes=200*c.MiB)


def test_schedule_fresh_paired_and_complete():
    rows=s.validate_schedule(s.make_schedule())
    assert len(rows)==206
    assert not ({r['seed_decimal'] for r in rows}&{r['seed_decimal'] for r in old.make_schedule()})
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['seed_decimal']==b['seed_decimal'] and a['challenger_seat']!=b['challenger_seat']
        assert a['challenger_mode']=='candidate' and a['challenger_quota']==64
        assert a['opponent_mode']=='baseline' and a['opponent_quota']==32


@pytest.mark.parametrize('change',['order','omit','duplicate','seed','quota','extra'])
def test_reject_changed_schedule(change):
    rows=deepcopy(s.make_schedule())
    if change=='order': rows[:2]=rows[1::-1]
    elif change=='omit': rows.pop()
    elif change=='duplicate': rows[1]=rows[0]
    elif change=='seed': rows[0]['seed_decimal']='1'
    elif change=='quota': rows[0]['opponent_quota']=64
    else: rows[0]['observer']='new'
    with pytest.raises(ValueError): s.validate_schedule(rows)


def test_forecast_fits_and_measured_forecast_is_model_free():
    assert p.validate_resource_forecast(valid_forecast())['status']=='passed'
    measured=p.resource_forecast()
    assert measured['projected_total_seconds']<c.JOB_SECONDS
    assert measured['model_calls']==measured['fresh_games']==0
    assert measured['inventories']['games_maximum']==206


@pytest.mark.parametrize('field,value',[
    ('model_calls',1),('fresh_games',1),('outcomes_used_for_selection_or_resource_estimation',True),
    ('projected_total_seconds',21600),('projected_total_seconds',float('nan')),
    ('projected_total_seconds',100),('projected_prefinal_bytes',c.PREFINAL_EVIDENCE_BYTES),
    ('projected_final_bytes',c.FINAL_RESERVE_BYTES),('projected_final_bytes',float('inf'))])
def test_forecast_rejects_limits_or_changed_scope(field,value):
    forecast=valid_forecast();forecast[field]=value
    with pytest.raises(ValueError):p.validate_resource_forecast(forecast)


@pytest.mark.parametrize('stage',list(c.STAGES))
def test_stage_cap_must_fit_strictly(stage):
    value=valid_forecast();value['projected_stage_seconds'][stage]=c.STAGES[stage]
    with pytest.raises(ValueError):p.validate_resource_forecast(value)


def test_watchdog_is_exact_qualified_version_except_new_runner():
    original=(c.ROOT/'tools/splendorust_boundary_timing_20261025/launch.py').read_text(encoding='utf-8')
    current=(c.ROOT/'tools/splendorust_live_boundary_20261025/launch.py').read_text(encoding='utf-8')
    assert current==original.replace('tools.splendorust_boundary_timing_20261025.runner',
        'tools.splendorust_live_boundary_20261025.runner')
    assert c.JOB_SECONDS==21600 and c.COLLECTION_CUTOFF_SECONDS==19800
    assert c.FINAL_RESERVE_BYTES==256*c.MiB and c.MAX_EVIDENCE_BYTES==2*c.GiB


def test_previous_sources_not_rewritten():
    frozen=c.read(c.PRIOR/'freeze.json')
    paths={item['path']:item for item in frozen['inputs']}
    selected=[item for path,item in paths.items()
              if path.startswith('tools/splendorust_boundary_timing_20261025/')]
    assert len(selected)>=7
    c.verify(selected)
