"""Audit-only orchestration, provenance, abort and real hidden spawn checks."""
import copy
from contextlib import nullcontext
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from . import common as c, experiment as e


def test_historical_abort_does_not_cancel_new_audit_but_new_abort_does(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'AUDIT_OUTPUT',tmp_path)
    assert (c.DATA/'abort.json').exists()
    c.budget_boundary(c.DATA)
    (tmp_path/'abort.json').write_text('{}')
    with pytest.raises(c.RecoveryAborted):c.budget_boundary(c.DATA)


def test_original_deadline_and_disk_reserve_remain_enforced(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'AUDIT_OUTPUT',tmp_path)
    monkeypatch.setattr(c.time,'time',lambda:c.DEADLINE_EPOCH+1)
    with pytest.raises(c.BudgetExceeded,match='deadline'):c.budget_boundary(c.DATA)
    monkeypatch.setattr(c.time,'time',lambda:c.DEADLINE_EPOCH-1)
    monkeypatch.setattr(c.shutil,'disk_usage',lambda _:SimpleNamespace(free=c.CONFIG['disk_reserve_bytes']-1))
    with pytest.raises(c.BudgetExceeded,match='reserve'):c.budget_boundary(c.DATA)


def test_facade_forwards_provenance_without_changing_r2_module():
    assert c.resolve_game is c.predecessor.resolve_game
    assert c.resolve_root is c.predecessor.resolve_root
    assert c.validate_origin is c.predecessor.validate_origin
    assert c.recovery_binding(c.DATA)==c.DATA_BINDING
    assert c.OUTPUT==c.DATA and c.AUDIT_OUTPUT!=c.DATA
    assert c.budget_boundary is not c.predecessor.budget_boundary


def actual_saved():
    games=e.old.read_json(e.DATA/'main_games_lock.json')['files']
    retained=[c.artifact(p,index=e.old.read_json(p)['game']['index']) for p in sorted((e.DATA/'main/audits').glob('game_*.json'))]
    return {'games':games,'retained_audits':retained,'pending_indices':[i for i in range(402) if i not in {r['index'] for r in retained}]}


def test_measured_forecast_never_restarts_clock_or_drops_original_full_audit_estimate():
    saved=actual_saved()
    result=e.resource_decision(saved,now=c.ORIGINAL_LAUNCH_EPOCH+80*3600,free=100*1024**3)
    assert result['reasonable'] and result['remaining_games']==0 and result['remaining_audits']==391
    assert result['forecast_seconds']>=1.25*result['original_probe_audit_seconds']+4200
    assert result['forecast_seconds']>=1.25*result['observed_audit_seconds']+4200
    assert result['deadline_epoch']==c.DEADLINE_EPOCH
    assert not e.resource_decision(saved,now=c.DEADLINE_EPOCH-1,free=100*1024**3)['reasonable']
    assert not e.resource_decision(saved,now=c.ORIGINAL_LAUNCH_EPOCH,free=c.CONFIG['disk_reserve_bytes'])['reasonable']


@pytest.mark.parametrize('field,value',[('passed',False),('manifest','wrong'),('recovery_binding','wrong'),
                                     ('sample_sha256','wrong'),('game_artifact',{}),('full_inner_replay',True)])
def test_existing_passed_audit_must_match_frozen_game_sample_and_scope(field,value):
    receipt=e.old.read_json(e.DATA/'main/audits/game_0000.json')
    item=receipt['game_artifact'];sample=receipt['sample_sha256']
    e.validate_receipt(receipt,item,sample)
    receipt[field]=value
    with pytest.raises((ValueError,AssertionError)):e.validate_receipt(receipt,item,sample)


def test_original_passed_receipt_cannot_be_relabelled_as_new(tmp_path,monkeypatch):
    ref=c.artifact(e.DATA/'main/audits/game_0000.json',index=0)
    receipt=c.verified(ref);saved={'retained_audits':[ref]}
    e.receipt_generation(ref,receipt,saved,'newbinding')
    receipt['audit_recovery_binding']='newbinding'
    with pytest.raises(ValueError):e.receipt_generation(ref,receipt,saved,'newbinding')


def pool_fixture(tmp_path,monkeypatch,fail=False):
    monkeypatch.setattr(e,'OUTPUT',tmp_path)
    monkeypatch.setattr(e,'validate_freeze',lambda **_: 'binding')
    monkeypatch.setattr(e.c,'budget_boundary',lambda *_:None)
    monkeypatch.setattr(e,'event',lambda *_,**__:None)
    monkeypatch.setattr(e.c.windows,'configure_spawn',lambda:{})
    submitted=[];futures=[];closed=[]
    class Pool:
        def __init__(self,**kwargs):assert kwargs['max_workers']<=8
        def __enter__(self):return self
        def __exit__(self,*_):closed.append((tmp_path/'abort.json').exists())
        def submit(self,fn,arg,*_):
            assert fn is e.audit_job
            f=Future();submitted.append(arg['index']);futures.append(f);return f
    def completed(pending,**_):
        f=next(iter(pending))
        if fail:f.set_exception(RuntimeError('fixture failure'))
        else:f.set_result({'index':pending[f]})
        return {f},set(pending)-{f}
    monkeypatch.setattr(e,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(e,'wait',completed)
    return submitted,futures,closed


def test_only_audits_dispatched_with_bounded_eight_job_queue(tmp_path,monkeypatch):
    submitted,_,closed=pool_fixture(tmp_path,monkeypatch)
    files,_=e.run_pool([{'index':i} for i in range(19)],'binding','main_audits',13)
    assert submitted==[f['index'] for f in files]==list(range(19)) and closed==[False]


def test_audit_failure_signals_abort_before_pool_shutdown(tmp_path,monkeypatch):
    submitted,futures,closed=pool_fixture(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError):e.run_pool([{'index':i} for i in range(19)],'binding','main_audits',13)
    assert submitted==list(range(8)) and closed==[True]
    assert all(f.cancelled() for f in futures[1:])


@pytest.mark.parametrize('fail',[False,True])
def test_canaries_gate_bulk_automatically_and_all_402_receipts_are_merged(tmp_path,monkeypatch,fail):
    saved=actual_saved();e.old.save_new(tmp_path/'inputs.json',saved)
    monkeypatch.setattr(e,'OUTPUT',tmp_path)
    monkeypatch.setattr(e,'locks',lambda:nullcontext())
    monkeypatch.setattr(e,'validate_freeze',lambda **_:'binding')
    monkeypatch.setattr(e.c,'budget_boundary',lambda *_:None)
    monkeypatch.setattr(e.c.windows,'assert_windowless_worker',lambda *_:{})
    monkeypatch.setattr(e.prior.original.parent,'inhibit_system_sleep',lambda:nullcontext({}))
    monkeypatch.setattr(e,'resource_decision',lambda *_:{'reasonable':True})
    monkeypatch.setattr(e,'event',lambda *_,**__:None)
    calls=[];final=[]
    def pool(items,binding,stage,already):
        calls.append((stage,[x['index'] for x in items],already))
        if fail:raise RuntimeError('canary did not pass')
        return [{'index':x['index']} for x in items],{}
    monkeypatch.setattr(e,'run_pool',pool)
    monkeypatch.setattr(e,'finish',lambda binding,saved,refs:final.append(refs))
    if fail:
        with pytest.raises(RuntimeError):e.run()
        assert len(calls)==1 and not final and (tmp_path/'failure.json').exists()
    else:
        e.run()
        assert calls[0]==('canary_audits',[13,310],11)
        assert calls[1][0]=='main_audits' and calls[1][2]==13 and len(calls[1][1])==389
        assert sorted(x['index'] for x in final[0])==list(range(402))


def test_worker_failure_record_names_game_and_retains_traceback(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'OUTPUT',tmp_path);monkeypatch.setattr(e.c,'budget_boundary',lambda *_:None)
    def fail(*_):raise AssertionError('checkpoint mismatch')
    monkeypatch.setattr(e.verify,'audit_game',fail)
    item={'index':13,'sha256':'gamehash','path':'game'}
    with pytest.raises(AssertionError):e.audit_job(item,'binding')
    paths=list((tmp_path/'worker_failures').glob('*.json'))
    assert len(paths)==1
    record=e.old.read_json(paths[0]);assert record['index']==13 and record['game']==item
    assert 'checkpoint mismatch' in record['traceback']


@pytest.mark.skipif(sys.platform!='win32',reason='Real Windows hidden worker integration')
def test_actual_pythonw_worker_loads_models_with_no_console_or_inference(tmp_path):
    e.old.save_new(tmp_path/'freeze.json',{'binding':'audit-only-worker-fixture'})
    launched=c.windows.detached_launch([sys.executable,'-u','-m',
        'tools.equal_budget_breadth_arena_128_recovery_03.worker_fixture',tmp_path],
        tmp_path/'stdout.txt',tmp_path/'stderr.txt',c.ROOT)
    deadline=time.monotonic()+90
    while not (tmp_path/'complete.json').exists():
        if (tmp_path/'failure.json').exists():pytest.fail(e.old.read_json(tmp_path/'failure.json')['traceback'])
        if time.monotonic()>deadline:pytest.fail('Hidden worker fixture timed out')
        time.sleep(.05)
    result=e.old.read_json(tmp_path/'complete.json')
    assert result['supervisor']['pid']==launched['pid']
    assert result['worker']['health']['parent_pid']==launched['pid']
    for health in (result['supervisor'],result['worker']['health']):
        assert Path(health['executable']).name.lower()=='pythonw.exe'
        assert health['console_window']==health['console_process_count']==0
    assert result['worker']['threads']==1 and result['worker']['deterministic']
    assert result['worker']['all_cpu'] and result['worker']['all_eval']
    assert result['worker']['models']==['anchor_161001','anchor_161002','anchor_161003']
    assert all(p.stat().st_size==0 for p in tmp_path.rglob('*stderr.txt'))
    assert not (tmp_path/'main').exists()
