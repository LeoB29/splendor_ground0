"""Model-free schedule, immutable output and forecast requirements."""
from copy import deepcopy
import pytest
from tools.splendorust_supervisor_separation_20261025 import preparation as p, common as c


def test_six_fresh_paired_games_keep_players_and_swap_seats():
    rows=p.validate_schedule(p.operational_schedule())
    assert len(rows)==6 and len({r['seed_decimal'] for r in rows})==3
    for i in range(0,6,2):
        a,b=rows[i:i+2]
        assert a['seed_decimal']==b['seed_decimal'] and a['challenger_seat']!=b['challenger_seat']
        assert a['challenger_quota']==64 and a['opponent_quota']==32
    from tools.splendorust_live_boundary_20261025.schedule import make_schedule
    assert not ({r['seed_decimal'] for r in rows}&{r['seed_decimal'] for r in make_schedule()})


@pytest.mark.parametrize('key,value',[('seed_decimal','1'),('challenger_quota',32),('opponent_mode','candidate'),('stage','main'),('game_index',True)])
def test_operational_mutation_rejected(key,value):
    rows=p.operational_schedule();rows[0][key]=value
    with pytest.raises(ValueError):p.validate_schedule(rows)


def good_forecast():
    return dict(status='passed',model_calls=0,fresh_games=0,outcomes_used_for_selection_or_resource_estimation=False,
        projected_stage_seconds={k:100. for k in c.STAGES},projected_total_seconds=400.,
        projected_prefinal_bytes=1024,projected_final_bytes=1024,
        inventories={'workers_maximum':72,'graph_workers_maximum':60})


def test_forecast_hard_stage_global_inventory_and_evidence_limits():
    assert p.validate_resource_forecast(good_forecast())['status']=='passed'
    for key,value in [('projected_final_bytes',c.FINAL_RESERVE_BYTES),('projected_prefinal_bytes',c.PREFINAL_EVIDENCE_BYTES),
        ('projected_total_seconds',float('nan')),('model_calls',1)]:
        row=good_forecast();row[key]=value
        with pytest.raises(ValueError):p.validate_resource_forecast(row)
    row=good_forecast();row['projected_stage_seconds']['final']=600.
    with pytest.raises(ValueError):p.validate_resource_forecast(row)
    row=good_forecast();row['inventories']['workers_maximum']=24
    with pytest.raises(ValueError):p.validate_resource_forecast(row)


def test_control_publication_is_exclusive_complete_and_owned(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    target=tmp_path/'control'/'0001.json';c.atomic_save(target,{'value':1})
    assert c.read(target)=={'value':1} and not target.with_suffix('.json.pending').exists()
    with pytest.raises(ValueError):c.atomic_save(target,{'value':2})
    assert c.read(target)=={'value':1}


def test_placement_exchange_authenticates_both_owners_freeze_mode_and_bound(monkeypatch):
    from tools.splendorust_supervisor_separation_20261025 import placement
    seen=[]
    monkeypatch.setattr(placement,'verify_saved_runtime',lambda value,freeze,mode,**kw:seen.append((mode,kw)))
    watchdog=dict(freeze_sha256='f',watchdog_pid=10,watchdog_created=20.)
    identity=dict(pid=30,created=40.)
    request=dict(sequence=1,freeze_sha256='f',runner=identity,watchdog=dict(pid=10,created=20.),
        placement_mode='isolated',runner_receipt={'monotonic_ns':100})
    ack=dict(sequence=1,request_sha256=c.digest(request),freeze_sha256='f',placement_mode='isolated',
        watchdog_receipt={'monotonic_ns':200})
    assert c.validate_exchange(request,ack,{},watchdog,identity)
    assert seen[0][1]['identity']==identity and seen[1][1]['role']=='watchdog'
    for key,value in [('sequence',2),('freeze_sha256','other'),('request_sha256','bad'),('placement_mode','original')]:
        mutated=deepcopy(ack);mutated[key]=value
        with pytest.raises(ValueError):c.validate_exchange(request,mutated,{},watchdog,identity)
    for stamp in (99,5_000_000_100):
        mutated=deepcopy(ack);mutated['watchdog_receipt']['monotonic_ns']=stamp
        with pytest.raises(ValueError):c.validate_exchange(request,mutated,{},watchdog,identity)
    wrong=deepcopy(identity);wrong['created']=41.
    with pytest.raises(ValueError):c.validate_exchange(request,ack,{},watchdog,wrong)
