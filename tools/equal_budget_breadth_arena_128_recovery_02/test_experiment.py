"""Recovery orchestration/resource/accounting checks without production gameplay."""
import copy
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from . import experiment as e


def prior_gate():
    return {'reasonable':True,'outcome_blind':True,'manifest':e.c.ORIGINAL_BINDING,
            'main_game_seconds_before_allowance':40*3600,
            'main_ordinary_audit_seconds_before_allowance':3*3600,
            'main_stress_seconds_before_allowance':3600,'projected_main_bytes':10*1024**3}


def test_gate_keeps_original_clock_all_audits_and_only_subtracts_completed_games():
    result = e.resource_decision(prior_gate(),12*3600,30*1024**3)
    expected = 1.25*(40*3600*92/402+4*3600)+600+3600
    assert result['reasonable'] and result['remaining_forecast_seconds'] == expected
    assert result['deduct_completed_work'] is True and result['deduct_partial_work'] is False
    assert result['remaining_games'] == 92 and result['audit_games'] == 402
    assert result['all_main_audit_seconds_before_allowance'] == 4*3600
    assert result['projected_additional_bytes'] == 10*1024**3
    assert result['deadline_epoch'] == e.c.ORIGINAL_LAUNCH_EPOCH+120*3600
    assert not e.resource_decision(prior_gate(),120*3600-expected+1,30*1024**3)['reasonable']
    assert not e.resource_decision(prior_gate(),12*3600,14*1024**3)['reasonable']


@pytest.mark.parametrize('field,value',[
    ('main_game_seconds_before_allowance',float('nan')),('main_ordinary_audit_seconds_before_allowance',float('inf')),
    ('projected_main_bytes',-1),('main_stress_seconds_before_allowance',0),
    ('manifest','other'),('reasonable',False),('outcome_blind',False)])
def test_gate_rejects_invalid_evidence(field,value):
    prior = prior_gate();prior[field] = value
    with pytest.raises((ValueError,AssertionError)):
        e.resource_decision(prior,0,30*1024**3)


def coverage_fixture(unknown=True,extra=True):
    ordinary = list(range(8))
    cells = [[114,3]] if extra and unknown else []
    root = {'index':13,'turn':17,'planner':'candidate','fit':161002,'support':4,
            'sha256':'root','world_counts':[128]*4,'worlds':128,'continuations':512,
            'ordinary_world_indices':ordinary,'replay_world_indices':ordinary,'stress':False,
            'full_inner_replay':False,'full_fixture_replay':False,
            'replayed_branches':32+len(cells),'replayed_worlds':8+len(cells),
            'terminal_continuations':512-int(unknown),'unresolved_continuations':int(unknown),
            'certified_roots':int(unknown),'censored_replay_cells':int(unknown),
            'coupled_certified_roots':0,'equivalence_proof_cells':0,'coupled_unknown_groups':0,
            'extra_censored_cells':cells,'extra_censored_replay_cells':len(cells),
            'extra_censored_replay_world_indices':[114] if cells else []}
    receipt = {'passed':True,'full_inner_replay':False,'sample_sha256':'sample','roots':[root]}
    sample = {'inventory':[{'index':13,'turn':17,'root_id':'13:17','arm':'candidate',
                           'fit':161002,'sha256':'root','support':4,'world_counts':[128]*4,
                           'worlds':128,'continuations':512}],
              'selections':[{'root_id':'13:17','ordinary_world_indices':ordinary,
                             'replay_world_indices':ordinary,'stress':False}],
              'coverage':{'games':1,'roots':1,'recorded_worlds':128,'recorded_continuations':512,
                          'replayed_worlds':8,'replayed_continuations':32,'stress_roots':0}}
    return receipt,sample


@pytest.mark.parametrize('unknown,extra',[(False,False),(True,False),(True,True)])
def test_extra_replay_does_not_replace_prescribed_sample(unknown,extra):
    receipt,sample = coverage_fixture(unknown,extra)
    unchanged = copy.deepcopy(receipt)
    result = e.checked_coverage([receipt],sample,'sample')
    assert receipt == unchanged
    assert result['prescribed_replayed_continuations'] == 32
    assert result['replayed_continuations'] == 32+int(extra)
    assert result['terminal_continuations']+result['unresolved_continuations'] == 512
    assert result['certified_roots'] == int(unknown)


@pytest.mark.parametrize('field,value',[
    ('terminal_continuations',512),('censored_replay_cells',0),('certified_roots',0),
    ('extra_censored_cells',[[0,3]]),('extra_censored_cells',[[128,3]]),
    ('extra_censored_replay_world_indices',[]),('replayed_worlds',8),
    ('replayed_branches',32),('extra_censored_cells',[[114,3],[114,3]])])
def test_corrupt_extra_replay_accounting_is_rejected(field,value):
    receipt,sample = coverage_fixture()
    receipt['roots'][0][field] = value
    with pytest.raises((ValueError,AssertionError)):
        e.checked_coverage([receipt],sample,'sample')


def test_coupled_proof_accounting_is_separate_from_allocated_cells():
    receipt,sample=coverage_fixture()
    root=receipt['roots'][0]
    root.update(coupled_certified_roots=1,equivalence_proof_cells=1,coupled_unknown_groups=1)
    result=e.checked_coverage([receipt],sample,'sample')
    assert result['recorded_continuations']==512 and result['equivalence_proof_cells']==1
    assert result['coupled_certified_roots']==result['coupled_unknown_groups']==1
    root['equivalence_proof_cells']=2
    with pytest.raises((ValueError,AssertionError),match='coupled-proof'):
        e.checked_coverage([receipt],sample,'sample')


def pool_fixture(monkeypatch,tmp_path,fail=False):
    monkeypatch.setattr(e,'OUTPUT',tmp_path)
    monkeypatch.setattr(e,'validate_freeze',lambda **kw:'recovery')
    monkeypatch.setattr(e.c,'budget_boundary',lambda *a:None)
    monkeypatch.setattr(e,'event',lambda *a,**kw:None)
    monkeypatch.setattr(e.c.windows,'configure_spawn',lambda :{'fixture':True})
    submitted=[];futures=[];stopped=[]
    class Pool:
        def __init__(self,**kwargs):
            assert kwargs['max_workers'] == 8
        def __enter__(self):return self
        def __exit__(self,*exc):
            stopped.append((tmp_path/'abort.json').exists())
        def submit(self,fn,arg,*args):
            f=Future();submitted.append(arg['index']);futures.append(f)
            return f
    def complete_one(pending,**kwargs):
        future=next(iter(pending))
        if fail:
            future.set_exception(RuntimeError('first worker failure'))
        else:
            future.set_result({'index':pending[future],'path':str(pending[future])})
        return {future},set(pending)-{future}
    monkeypatch.setattr(e,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(e,'wait',complete_one)
    return submitted,futures,stopped


def test_pool_limits_queue_and_signals_abort_before_shutdown(monkeypatch,tmp_path):
    submitted,futures,stopped = pool_fixture(monkeypatch,tmp_path,fail=True)
    with pytest.raises(RuntimeError,match='first worker failure'):
        e.run_pool('games',[{'index':i} for i in range(30)],'recovery')
    assert submitted == list(range(8))
    assert all(f.cancelled() for f in futures[1:])
    assert stopped == [True]
    assert e.old.read_json(tmp_path/'abort.json')['reason'] == 'first worker failure'


def test_pool_success_collects_exact_schedule_without_abort(monkeypatch,tmp_path):
    submitted,futures,stopped = pool_fixture(monkeypatch,tmp_path)
    files,metadata=e.run_pool('games',[{'index':i} for i in range(20)],'recovery')
    assert submitted == [f['index'] for f in files] == list(range(20))
    assert stopped == [False] and 'elapsed_seconds' in metadata


def test_ambiguous_first_root_stops_before_dispatch(monkeypatch,tmp_path):
    monkeypatch.setattr(e,'OUTPUT',tmp_path)
    monkeypatch.setattr(e,'SOURCE',tmp_path/'original')
    monkeypatch.setattr(e,'validate_freeze',lambda **kw:'recovery')
    monkeypatch.setattr(e.old,'process_lock',lambda *a:nullcontext())
    monkeypatch.setattr(e.c.windows,'assert_windowless_worker',lambda *a:{})
    monkeypatch.setattr(e.original.parent,'inhibit_system_sleep',lambda :nullcontext({}))
    monkeypatch.setattr(e.c,'budget_boundary',lambda *a:None)
    monkeypatch.setattr(e,'resource_decision',lambda *a:{'reasonable':True})
    monkeypatch.setattr(e.models,'validate',lambda *a:None)
    monkeypatch.setattr(e.arena,'initialize_models',lambda **kw:None)
    monkeypatch.setattr(e.c,'load_inventory',lambda *a:{})
    monkeypatch.setattr(e,'event',lambda *a,**kw:None)
    monkeypatch.setattr(e,'REPORTER',None)
    monkeypatch.setattr(e,'StatusReporter',lambda *a:None)
    e.old.save_new(tmp_path/'original/operational_decision.json',{})
    def ambiguous(*a):
        raise e.arena.AmbiguousDecision({'choice':None})
    monkeypatch.setattr(e,'certify_failed_root',ambiguous)
    def no_pool(*a):
        pytest.fail('Game pool dispatched before certificate')
    monkeypatch.setattr(e,'run_pool',no_pool)
    e.run()
    assert (tmp_path/'abort.json').is_file()
    assert e.old.read_json(tmp_path/'operational_hold.json')['type'] == 'AmbiguousDecision'
    launch=e.old.read_json(tmp_path/'launch.json')
    assert launch['original_launch_epoch'] == e.c.ORIGINAL_LAUNCH_EPOCH
    assert 'epoch_seconds' not in launch
    assert not (tmp_path/'main_games_launch.json').exists()
