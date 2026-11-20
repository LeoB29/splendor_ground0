"""Resource/stage and fixed-schedule tests; no production model execution."""
from types import SimpleNamespace
import pytest
from tools.splendorust_one_second_20261024 import common as c, runner as r


def test_schedule_fixed_fresh_pairing():
    rows=c.make_schedule()
    assert len(rows)==206
    assert len({x['seed_decimal'] for x in rows})==103
    for stage,n in [('operational',3),('main',100)]:
        selected=[x for x in rows if x['stage']==stage]
        assert len(selected)==2*n
        for pair in range(n):
            a,b=selected[pair*2:pair*2+2]
            assert a['seed_decimal']==b['seed_decimal']
            assert (a['search_seat'],b['search_seat'])==(pair%2,1-pair%2)
            assert (a['game_index'],b['game_index'])==(2*pair,2*pair+1)


def test_exact_prepared_schedule_identity():
    prepared=c.read(c.PREP/'prospective_schedule.json')['games']
    assert c.make_schedule()==[{k:old[k] for k in ('stage','pair_index','game_index','seed_decimal')}|
        {'search_seat':old['puct16_seat']} for old in prepared]


def env(monkeypatch,mono=100,wall=1000):
    clocks={'mono':mono,'wall':wall}
    monkeypatch.setattr(r.time,'monotonic',lambda:clocks['mono'])
    monkeypatch.setattr(r.time,'time',lambda:clocks['wall'])
    monkeypatch.setattr(c,'save',lambda *a,**k:None)
    monkeypatch.setattr(r,'bytes_used',lambda:0)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda p:SimpleNamespace(free=10*c.GiB))
    return r.Envelope({'monotonic_origin':100,'unix_origin':1000}),clocks


def test_stage_cap_not_transferred(monkeypatch):
    e,t=env(monkeypatch); e.begin('qualification')
    assert e.deadline==3700
    t['mono']+=20; t['wall']+=20
    e.finish(); e.begin('operational')
    assert e.remaining()==1800
    t['mono']+=10; t['wall']+=10
    e.finish(); e.begin('main')
    assert e.remaining()==14400


def test_absolute_collection_cutoff(monkeypatch):
    e,t=env(monkeypatch); t['mono']+=19000; t['wall']+=19000
    e.begin('main'); assert e.remaining()==800
    assert e.timeout(1000)==800
    t['wall']+=800
    with pytest.raises(ValueError,match='deadline'): e.guard()


def test_utc_jump_cannot_extend(monkeypatch):
    e,t=env(monkeypatch); e.begin('qualification')
    t['wall']+=3600
    with pytest.raises(ValueError,match='deadline'): e.guard()


def test_final_reserve(monkeypatch):
    e,t=env(monkeypatch); e.begin('main')
    monkeypatch.setattr(r,'bytes_used',lambda:449*c.MiB)
    with pytest.raises(ValueError,match='evidence'): e.guard()
    e.begin('final'); e.guard()
    monkeypatch.setattr(r,'bytes_used',lambda:512*c.MiB)
    with pytest.raises(ValueError,match='evidence'): e.guard()


@pytest.mark.parametrize('status',['resource_guard','rss_limit','process_tree_limit','affinity_changed'])
def test_resource_failure_never_scores_loss(status):
    with pytest.raises(ValueError): r.classify_failure({'status':status,'detail':''})


def test_attributable_timeout_scores_loss():
    receipt=dict(status='timeout',detail=None,request_id='request-00000001',clock='perf_counter_ns',
        deadline_seconds=1.,elapsed_seconds=1.01,receipt_elapsed_seconds=None,received_before_deadline=False,accepted=False)
    assert r.classify_failure(receipt)=='timeout'


def test_real_player_deadline_not_engineering_margin():
    # A timely 0.8-second move is a valid main game response. No loss at 0.7.
    from tools.splendorust_one_second_20261024.audit import check_clock
    receipt=dict(request_id='request-00000001',clock='perf_counter_ns',deadline_seconds=1.,elapsed_seconds=.85,
                 receipt_elapsed_seconds=.8,received_before_deadline=True,accepted=True,status='ok')
    assert check_clock(receipt)==.8


def test_incremental_evidence_covers_parent_and_closed_worker(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'output')
    monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    c.OUT.mkdir()
    c.save(c.OUT/'freeze.json',{'fixed':True})
    e=r.Envelope({'monotonic_origin':r.time.monotonic(),'unix_origin':r.time.time()})
    e.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',{'new':1})
    folder=c.OUT/'workers'/'one'; folder.mkdir(parents=True)
    e.register_worker(folder)
    c.save(folder/'inputs'/'request.json',{'observer':1})
    (folder/'stdout.log').write_bytes(b'foreign_worker_output')
    assert e.accounted_bytes()==r.bytes_used()+65536
    e.close_worker(folder)
    c.save(c.OUT/'game.json',{'complete':True})
    (c.OUT/'runner_stdout.log').write_bytes(b'progress')
    assert e.accounted_bytes()==r.bytes_used()+65536
