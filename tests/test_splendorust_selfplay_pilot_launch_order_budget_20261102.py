"""Actual owner/ack/readback orchestration with a deliberately fast fake fitter.

The fake Popen publishes placement before returning to the parent, then emits
six real-format clock records. No model, native process or game is invoked.
"""
import ast
import inspect
import json
from types import SimpleNamespace

import psutil
import pytest

from tools.splendorust_selfplay_pilot_budget_20261102 import common as c
from tools.splendorust_selfplay_pilot_budget_20261102 import qualification_control as owner
from tools.splendorust_selfplay_pilot_budget_20261102 import qualification_watchdog as watchdog
from tools.splendorust_selfplay_pilot_budget_20261102 import training


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def old_postspawn_sampling_function():
    tree=ast.parse(inspect.getsource(owner.run_fit))
    body=next(n for n in ast.walk(tree) if isinstance(n,ast.Try)).body
    assignment=next(n for n in body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='owner_reading' for t in n.targets))
    body.remove(assignment)
    spawn=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
        isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='Popen')
    body.insert(spawn+1,assignment)
    namespace=dict(vars(owner));exec(compile(ast.fix_missing_locations(tree),'<old-postspawn-order>','exec'),namespace)
    return namespace['run_fit']


def complete_fast_fitter(monkeypatch,tmp_path,*,legacy=False):
    from tools.local_player_qualification_128 import clock as process_clock
    import subprocess
    out=tmp_path/'scope';out.mkdir()
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    epoch,qpc_offset=1_800_000_000.,10_000_000.
    state={'mono':100.,'done':False};calls=[]
    def sample():return dict(owner_monotonic=state['mono'],request_qpc=qpc_offset+state['mono'],unix=epoch+state['mono'])
    monkeypatch.setattr(owner.clocks,'sample',sample)
    monkeypatch.setattr(owner,'time',SimpleNamespace(monotonic=lambda:state['mono'],time=lambda:epoch+state['mono'],sleep=lambda _:None))
    monkeypatch.setattr(owner,'fit_command_environment',lambda *args:(['fake-fitter'],{}))
    monkeypatch.setattr(process_clock,'_WindowsJob',lambda pid:None)
    monkeypatch.setattr(watchdog,'cleanup_owned',lambda *args,**kwargs:{'errors':[]})
    identity=dict(pid=54321,created=epoch+100.)
    contract=owner.clocks.runtime_contract()
    freeze=dict(clock_contract=contract,learned_entrypoint={'case_ids':['a','b','c']},
        fit_plan={'path':'fit_plan.json'},fit_parity_fixtures={'path':'parity.json'})
    write(tmp_path/'fit_plan.json',{});write(tmp_path/'parity.json',{'fixtures':[]});write(out/'corpus_manifest.json',{})
    write(out/'stages/fit_start.json',dict(monotonic=90.,unix_time=epoch+90.,monotonic_deadline=200.,unix_deadline=epoch+200.))
    def process(pid=None):
        if pid==identity['pid'] and state['done']:raise psutil.NoSuchProcess(pid)
        actual=identity if pid is not None else dict(pid=12345,created=epoch-20.)
        return SimpleNamespace(pid=actual['pid'],create_time=lambda:actual['created'],cpu_affinity=lambda:[0],
            nice=lambda:owner.placement.NORMAL_PRIORITY,children=lambda recursive=True:[],memory_info=lambda:SimpleNamespace(rss=100))
    monkeypatch.setattr(owner,'psutil',SimpleNamespace(Process=process,NoSuchProcess=psutil.NoSuchProcess))
    class FastChild:
        pid=identity['pid'];returncode=None
        def __init__(self,*args,**kwargs):
            # This event is permitted by Popen: the child runs before it returns.
            state['mono']=100.5;self.index=0
            folder=out/'fit';reading=sample()
            write(folder/'placement.json',dict(identity=identity,cpu_affinity=[0],priority='NORMAL_PRIORITY_CLASS',
                before_torch_import=True,clock_contract=contract,owner_monotonic_time=reading['owner_monotonic'],
                request_qpc_time=reading['request_qpc'],unix_time=reading['unix']))
            write(folder/'checkpoint_manifest.json',{'synthetic_checkpoint_identity':True})
            state['mono']=101.
        def poll(self):
            if self.index==6:
                state['mono']+=.1;state['done']=True;self.returncode=0;return 0
            i=self.index;self.index+=1;state['mono']+=.2
            directory=out/'fit/learned_entrypoint'
            start=dict(schema='selfplay_learned_request_start_v2',request_index=i,case_id=['a','b','c'][i%3],
                mode='eager' if i<3 else 'compiled',candidate_manifest=c.artifact(out/'fit/checkpoint_manifest.json'),
                **owner.clocks.start_fields())
            path=directory/f'request_{i:03d}.started.json';write(path,start)
            state['mono']+=.1
            response=dict(schema='selfplay_learned_response_v2',request_index=i,case_id=start['case_id'],mode=start['mode'],
                candidate_manifest=start['candidate_manifest'],start=c.artifact(path),**owner.clocks.completion_fields(start))
            write(directory/f'request_{i:03d}.completed.json',response);state['mono']+=.1
            return None
    monkeypatch.setattr(subprocess,'Popen',FastChild)
    monkeypatch.setattr(training,'verify_qualification',lambda *args,**kwargs:calls.append('science-verifier') or {'passed':True})
    env=SimpleNamespace(resident={},start=0.,wall_start=epoch,deadline=200.,wall_deadline=epoch+200.,admit=lambda:None,guard=lambda:None)
    (old_postspawn_sampling_function() if legacy else owner.run_fit)(env,freeze)
    return out,env,freeze,calls


def test_actual_owner_prespawn_sample_accepts_fast_child_and_all_six_acks(monkeypatch,tmp_path):
    out,env,freeze,calls=complete_fast_fitter(monkeypatch,tmp_path)
    launched=c.read(out/'fit/owner.json');placed=c.read(out/'fit/placement.json')
    assert launched['owner_monotonic_start']==100.<placed['owner_monotonic_time']==100.5
    assert launched['request_qpc_start']<placed['request_qpc_time']
    assert c.read(out/'fit_execution.json')['exit_code']==0
    assert len(list((out/'fit/learned_entrypoint').glob('*.parent_accepted.json')))==6
    assert owner.verify_fit(env,freeze)=={'passed':True} and calls==['science-verifier']


def test_old_postspawn_sampling_reproduces_saved_verifier_false_stop(monkeypatch,tmp_path):
    out,env,freeze,calls=complete_fast_fitter(monkeypatch,tmp_path,legacy=True)
    assert c.read(out/'fit/owner.json')['owner_monotonic_start']==101.
    assert c.read(out/'fit/placement.json')['owner_monotonic_time']==100.5
    with pytest.raises(ValueError,match='placement before Torch'):owner.verify_fit(env,freeze)
    assert calls==[]


@pytest.mark.parametrize('field',['owner_monotonic_start','request_qpc_start','monotonic_deadline','unix_deadline'])
def test_saved_reader_rejects_future_owner_or_reset_deadline(monkeypatch,tmp_path,field):
    out,env,freeze,calls=complete_fast_fitter(monkeypatch,tmp_path)
    path=out/'fit/owner.json';value=c.read(path);value[field]+=1000.;write(path,value)
    with pytest.raises(ValueError):owner.verify_fit(env,freeze)
    assert calls==[]
