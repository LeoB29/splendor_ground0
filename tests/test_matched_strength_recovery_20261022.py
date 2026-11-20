"""Model-free recovery boundaries, unchanged routing and exact prefix reconstruction."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.matched_strength_recovery_20261022 import common as c, memory as m, runner as r, worker as w
from tools.matched_strength_recovery_20261022 import opponent_worker as ow
from tools.matched_strength_20261020 import runner as original_runner, worker as original_worker
from tools.splendorust_120_qualification_20261018 import worker as original_opponent


def artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'artifact',lambda p:{'path':Path(p).resolve().relative_to(tmp_path.resolve()).as_posix(),
                                             'sha256':c.file_hash(p),'bytes':Path(p).stat().st_size})


def test_unchanged_context_profiles_and_numerical_entrypoints():
    from tools.matched_strength_20261020 import common as old
    for name in ('operational_context','make_observer_payload','sample_worlds','stress_cells','check_wrapper','check_audit_proof'):
        assert getattr(c,name) is getattr(old,name)
    assert c.LOCAL_PROFILE_ID==old.LOCAL_PROFILE_ID and c.EXTERNAL_PROFILE_SHA256==old.EXTERNAL_PROFILE_SHA256
    assert c.COMPLETED_GAMES==old.game_order(30)[:-1] and len(c.COMPLETED_GAMES)==59
    assert c.COMPLETED_GAMES[-1]==57 and c.RETRY_IDENTITY=='games/g059/t0043'
    assert ow.handle is original_opponent.handle
    assert r.response_has_recognized_illegal_action is original_runner.response_has_recognized_illegal_action


def test_worker_handle_preserves_all_numerical_calls_and_trace_delivery():
    import inspect
    before=inspect.getsource(original_worker.handle)
    after=inspect.getsource(w.handle).replace("path = audit_trace_path(payload['traces_path'])","path = trace_path(payload['traces_path'])")
    assert ast.dump(ast.parse(before))==ast.dump(ast.parse(after))


def test_new_output_only_writes_but_old_traces_are_readable(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'new');monkeypatch.setattr(c,'SOURCE',tmp_path/'old')
    new=tmp_path/'new/workers/games/g059/t0043/continuations.jsonl.gz'
    old=tmp_path/'old/workers/games/g001/t0001/continuations.jsonl.gz'
    assert w.trace_path(new)==new.resolve() and w.audit_trace_path(old)==old.resolve()
    with pytest.raises(ValueError):w.trace_path(old)
    with pytest.raises(ValueError):w.audit_trace_path(tmp_path/'elsewhere/continuations.jsonl.gz')
    with pytest.raises(ValueError):w.audit_trace_path(tmp_path/'old/continuations.jsonl.gz')
    with pytest.raises(ValueError):w.audit_trace_path(tmp_path/'new/workers/../../outside/continuations.jsonl.gz')


@pytest.mark.parametrize('mode',['local','audit','opponent'])
def test_each_new_worker_authenticates_sources_once_before_initialization(tmp_path,monkeypatch,mode):
    from tools import future_training_rules_20261021 as guard
    import tools.matched_strength_20261020 as package
    calls=[]; sources={'synthetic':'a'*64}
    monkeypatch.setattr(guard,'verify_runtime_sources',lambda:calls.append('sources') or sources)
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    path=tmp_path/'workers/runtime_sources.json'
    monkeypatch.setenv('LIVE_RUNTIME_SOURCE_PATH',str(path));monkeypatch.setenv('LIVE_WORKER_MODE',mode)
    if mode=='opponent':
        monkeypatch.setattr(ow,'qualified_initialize',lambda:calls.append('qualified') or {'unchanged':True})
        assert ow.initialize()=={'unchanged':True}
    else:
        local=SimpleNamespace(initialize=lambda:calls.append('qualified') or {'unchanged':True})
        monkeypatch.setattr(package,'local',local,raising=False)
        assert w.initialize()['local'] is local
    assert calls==['sources','qualified']
    assert c.read(path)=={'schema':'corrected_runtime_sources_v1','ruleset_id':c.RULESET_ID,
                          'catalogue_sha256':c.CATALOGUE_SHA256,'sources':sources}
    with pytest.raises(FileExistsError):w.authenticate_runtime_sources()


def test_supervisor_authenticates_once_at_startup_before_source_reuse(tmp_path,monkeypatch):
    import inspect
    from tools import future_training_rules_20261021 as guard
    calls=[]; sources={'synthetic':'b'*64}
    monkeypatch.setattr(guard,'verify_runtime_sources',lambda:calls.append('authenticate') or sources)
    path=tmp_path/'runtime_sources.json';r.save_runtime_sources(path)
    assert calls==['authenticate'] and c.read(path)['sources']==sources
    tree=ast.parse(inspect.getsource(r.run))
    named=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    authentication=[node for node in named if node.func.id=='save_runtime_sources']
    reuse=[node for node in named if node.func.id in ('check_source_closed','verify_execution_inputs','check_interrupted_request')]
    assert len(authentication)==1 and all(authentication[0].lineno<node.lineno for node in reuse)


def clock_fixture(tmp_path,monkeypatch):
    source,new=tmp_path/'old',tmp_path/'new';monkeypatch.setattr(c,'SOURCE',source)
    c.save_new(source/'launch.json',{'maximum_seconds':259200,'epoch_seconds':1000.,'monotonic_start':100.})
    c.save_new(new/'launch.json',{'maximum_seconds':43200,'epoch_seconds':2000.,'monotonic_start':1100.})
    monkeypatch.setattr(c.time,'time',lambda:2001.);monkeypatch.setattr(c.time,'monotonic',lambda:1101.)
    monkeypatch.setattr(c.shutil,'disk_usage',lambda _:SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(c,'preserved_source_bytes',lambda:3*1024**3)
    monkeypatch.setattr(c,'directory_bytes',lambda _:1024)
    monkeypatch.setattr(c,'memory',lambda:{'available_physical':2*1024**3,'available_page':4*1024**3})
    monkeypatch.setattr(c,'config',lambda:{'evidence_cap_bytes':8*1024**3,'cumulative_memory_wait_seconds':21600})
    return new


@pytest.mark.parametrize('which',['old_wall','old_mono','new_wall','new_mono','clock_reset','disk','size','memory'])
def test_all_four_clocks_and_resource_limits_fail_closed(tmp_path,monkeypatch,which):
    new=clock_fixture(tmp_path,monkeypatch)
    if which=='old_wall':
        monkeypatch.setattr(c.time,'time',lambda:260200.)
    elif which=='old_mono':monkeypatch.setattr(c.time,'monotonic',lambda:259300.)
    elif which=='new_wall':monkeypatch.setattr(c.time,'time',lambda:45200.)
    elif which=='new_mono':monkeypatch.setattr(c.time,'monotonic',lambda:44300.)
    elif which=='clock_reset':monkeypatch.setattr(c.time,'monotonic',lambda:99.)
    elif which=='disk':monkeypatch.setattr(c.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    elif which=='size':monkeypatch.setattr(c,'directory_bytes',lambda _:5*1024**3)
    elif which=='memory':monkeypatch.setattr(c,'memory',lambda:{'available_physical':0,'available_page':4*1024**3})
    with pytest.raises((c.ResourceHold,ValueError)):c.boundary(new,44300.)


def test_different_original_and_recovery_game_reserves(monkeypatch):
    receipt={'original_remaining_seconds':64799.,'recovery_remaining_seconds':40000.}
    assert c.work_budget(receipt,21600)==-1
    assert c.work_budget(receipt,4200)==35800
    monkeypatch.setattr(c,'boundary',lambda *a,**kw:receipt)
    with pytest.raises(c.ResourceHold):m.guard(Path('unused'),1e9,21600,120)
    assert m.guard(Path('unused'),1e9,4200,600)==receipt


def test_worker_deadline_respects_old_reserve_even_when_recovery_has_time(monkeypatch):
    monkeypatch.setattr(r.sys,'modules',{k:v for k,v in sys.modules.items() if k!='torch'})
    monkeypatch.setattr(r.time,'monotonic',lambda:100.)
    value={'original_remaining_seconds':65000.,'recovery_remaining_seconds':40000.}
    monkeypatch.setattr(c,'boundary',lambda *a,**kw:value)
    result=r.worker_options(40100.,0,reserve=21600,mode='local',runtime=Path('runtime_sources.json'))
    assert result['absolute_deadline']==300.
    assert 'LIVE_RUNTIME_SOURCE_PATH' in result['env']
    value['original_remaining_seconds']=64800.
    with pytest.raises(c.ResourceHold):result['resource_guard']()
    with pytest.raises(c.ResourceHold):r.worker_options(40100.,0,reserve=21600,mode='local')


class FakeClock:
    value=0.
    def monotonic(self):return 100.+self.value
    def time(self):return 1000.+self.value
    def sleep(self,value):self.value+=value


def memory_fixture(tmp_path,monkeypatch,charge=100.):
    c.save_new(tmp_path/'prior_memory_state.json',{'schema':'matched_strength_prior_memory_v1',
                                                'admission_count':2000,'charged_seconds':charge,'inputs':[]})
    clock=FakeClock();monkeypatch.setattr(m.time,'time',clock.time);monkeypatch.setattr(m.time,'monotonic',clock.monotonic)
    monkeypatch.setattr(m.time,'sleep',clock.sleep);monkeypatch.setattr(m,'emit',lambda _:None)
    monkeypatch.setattr(c,'config',lambda:{'cumulative_memory_wait_seconds':21600})
    monkeypatch.setattr(m,'guard',lambda *a,**kw:{'remaining_seconds':40000,'recovery_remaining_seconds':40000,
                                                'original_remaining_seconds':100000})
    monkeypatch.setattr(m,'memory',lambda:{'available_physical':2*1024**3,'available_page':4*1024**3})
    return clock


def test_initial_ram_stability_and_total_charge_are_not_reset(tmp_path,monkeypatch):
    memory_fixture(tmp_path,monkeypatch)
    value=m.wait_for_memory(tmp_path,100000,name='initial',reserve=600,request_seconds=0,initial=True)
    assert value['charged_seconds']==120 and value['prior_admission_count']==2000
    assert value['cumulative_before_seconds']==100 and value['cumulative_after_seconds']==220
    assert len(value['readings'])==13
    next_value=m.wait_for_memory(tmp_path,100000,name='games/g059/t0043',reserve=21600,request_seconds=120)
    assert next_value['prior_admission_count']==2001 and next_value['cumulative_before_seconds']==220
    with pytest.raises(c.ResourceHold,match='already attempted'):
        m.wait_for_memory(tmp_path,100000,name='games/g059/t0043',reserve=21600,request_seconds=120)


def test_inherited_memory_limit_cannot_be_restarted(tmp_path,monkeypatch):
    memory_fixture(tmp_path,monkeypatch,21600.)
    with pytest.raises(c.ResourceHold):m.wait_for_memory(tmp_path,100000,name='initial',reserve=600,request_seconds=0,initial=True)
    assert c.read(tmp_path/'memory_admissions/initial.json')['status']=='preserved_hold'


def test_authentication_clock_callback_rejects_expiry(monkeypatch):
    monkeypatch.setattr(r.time,'monotonic',lambda:100.)
    monkeypatch.setattr(r.time,'time',lambda:50000.)
    with pytest.raises(c.ResourceHold):r.clock_guard({'monotonic_start':99.,'epoch_seconds':1.},
        {'monotonic_start':1.,'epoch_seconds':1.,'maximum_seconds':259200})


def test_preparation_exclusive_and_execution_never_restarts(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    with pytest.raises(FileExistsError):r.prepare()
    c.save_new(tmp_path/'freeze.json',{});c.save_new(tmp_path/'launch.json',{})
    with pytest.raises(ValueError,match='No second recovery'):r.run(c.file_hash(tmp_path/'freeze.json'))


def test_original_interrupted_request_with_later_response_cannot_be_retried(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'SOURCE',tmp_path)
    path=tmp_path/'workers/games/g059/t0043/request-00000001.json';c.save_new(path,{'synthetic':'receipt'})
    with pytest.raises(ValueError,match='replace accepted'):r.check_interrupted_request()


def synthetic_prefix(tmp_path,monkeypatch):
    artifacts(monkeypatch,tmp_path);source=tmp_path/'source';game=59
    spec={'game':game,'pair':29,'fit':161003,'local_seat':c.local_seat(game),
          'opponent_seat':1-c.local_seat(game),'seed':'814'}
    env=r.rules.CorrectedEnv(seed=814);state=env.initial_state();folder=source/'sealed/games/059'
    initial=r.rules.serialize_snapshot(state)
    c.save_new(folder/'initial.json',{**spec,'initial':initial,'initial_sha256':c.digest(initial)})
    monkeypatch.setattr(c,'check_wrapper',lambda value,obs,legal,path:value['decision']['action'] in legal)
    monkeypatch.setattr(r,'load_profile',lambda *a:{})
    monkeypatch.setattr(r,'check_opponent',lambda value,obs,profile,sha,snapshot:
        r.rules.serialize_snapshot(r.rules.CorrectedEnv().step(r.rules.deserialize_snapshot(snapshot),c.deserialize_action(value['action']))))
    for ply in range(3):
        before=r.rules.serialize_snapshot(state);actor=state.current_player
        player='local' if actor==spec['local_seat'] else 'opponent';name=f'games/g059/t{ply:04d}'
        observer=c.make_observer_payload(before,c.operational_context(59,ply)) if player=='local' else r.rules.observation_payload(state)
        action=c.serialize_action(r.rules.legal_actions(state)[0]);state=env.step(state,c.deserialize_action(action))
        value={'decision':{'action':action}} if player=='local' else {'action':action}
        receipt={'status':'ok','accepted':True,'result':value};rp=source/'workers'/name/'request-00000001.json';c.save_new(rp,receipt)
        row={'game':59,'pair':29,'fit':161003,'ply':ply,'actor':actor,'before_sha256':c.digest(before),'id':name,
             'player':player,'observer':observer,'receipt':c.artifact(rp),'clock':r.without_result(receipt),
             'status':'completed','correct':True,'result_sha256':c.digest(value),
             'after_sha256':c.digest(r.rules.serialize_snapshot(state))}
        if player=='local':
            trace=source/'workers'/name/'continuations.jsonl.gz';trace.write_bytes(b'synthetic');row['trace']=c.artifact(trace)
        c.save_new(folder/'moves'/f'{ply:04d}.json',row)
        c.save_new(source/'decision_attempts'/(name+'.json'),{k:v for k,v in row.items() if k not in ('game','pair','fit','ply','actor','before_sha256')})
    return spec,source,env,state


def test_prefix_reconstruction_preserves_referee_state_and_future_transitions(tmp_path,monkeypatch):
    spec,source,env,state=synthetic_prefix(tmp_path,monkeypatch)
    recovered=r.reconstruct_game(spec,source,complete=False,next_ply=3)
    assert recovered['snapshot']==r.rules.serialize_snapshot(state)
    action=r.rules.legal_actions(state)[0]
    assert r.rules.serialize_snapshot(env.step(state,action))==r.rules.serialize_snapshot(recovered['env'].step(recovered['state'],action))


@pytest.mark.parametrize('damage',['seed','missing_ply','state_hash','receipt'])
def test_prefix_damage_blocks_new_move(tmp_path,monkeypatch,damage):
    spec,source,env,state=synthetic_prefix(tmp_path,monkeypatch)
    if damage=='seed':spec={**spec,'seed':'815'}
    elif damage=='missing_ply':(source/'sealed/games/059/moves/0001.json').rename(source/'sealed/games/059/moves/0004.json')
    elif damage=='state_hash':
        path=source/'sealed/games/059/moves/0000.json';value=c.read(path);value['before_sha256']='0'*64;path.write_text(json.dumps(value))
    else:
        path=source/'workers/games/g059/t0000/request-00000001.json';path.write_text('{}')
    with pytest.raises(ValueError):r.reconstruct_game(spec,source,complete=False,next_ply=3)
