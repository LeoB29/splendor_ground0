"""Narrow model-free launcher repair; unchanged science and original deadlines."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.move_evaluation_inference_recovery_20261023 import runner as r, launch as l


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def context(tmp_path,monkeypatch):
    c=r.c;root=tmp_path/'repo';out=root/'outputs/repair';source=root/'outputs/original'
    out.mkdir(parents=True);source.mkdir(parents=True)
    for key,value in {'ROOT':root,'OUT':out,'SOURCE':source}.items():monkeypatch.setattr(c,key,value)
    def artifact(path):
        path=Path(path).resolve();data=path.read_bytes()
        return {'path':path.relative_to(root).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(c,'artifact',artifact)
    clock=SimpleNamespace(mono=200.,wall=1100.,boot=900.)
    fake=SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=lambda s:None)
    monkeypatch.setattr(r,'time',fake);monkeypatch.setattr(l,'time',fake)
    monkeypatch.setattr(r.psutil,'boot_time',lambda:clock.boot)
    plan={**r.new_study_plan(),'monotonic_origin':100.,'unix_origin':1000.,'boot_time':900.}
    put(source/'study_plan.json',plan)
    claim={'monotonic_origin':110.,'unix_origin':1010.,'watchdog_pid':123,'watchdog_created':99.,'freeze_sha256':'a'*64}
    put(source/'watchdog_claim.json',claim)
    freeze={'study_plan':artifact(source/'study_plan.json'),'inputs':[artifact(source/'study_plan.json')],
        'job_seconds':900,'final_reserve_seconds':120,'original_claim':artifact(source/'watchdog_claim.json'),
        'original_stage_monotonic_origin':110.,'original_stage_unix_origin':1010.}
    put(out/'freeze.json',freeze)
    return c,clock,plan,claim,freeze


@pytest.mark.parametrize('difference',[-1.,-.3534994,0.,.288,1.])
def test_documented_psutil_estimate_fluctuation_is_accepted(context,difference):
    c,clock,plan,claim,freeze=context;clock.boot=plan['boot_time']+difference
    assert r.validate_study_plan(plan)==plan
    assert r.read_study_plan(freeze)==plan
    assert not (c.OUT/'study_plan.json').exists()


@pytest.mark.parametrize('difference',[-1.001,1.001,100.,float('nan')])
def test_larger_boot_difference_still_fails_closed(context,difference):
    c,clock,plan,claim,freeze=context;clock.boot=plan['boot_time']+difference
    with pytest.raises(ValueError,match='reboot'):r.validate_study_plan(plan)


def test_same_estimated_boot_cannot_hide_monotonic_restart(context):
    c,clock,plan,claim,freeze=context;clock.mono=99.
    with pytest.raises(ValueError,match='uptime'):r.validate_study_plan(plan)


@pytest.mark.parametrize('edge',['mono_work','wall_work'])
def test_original_stage_deadline_including_final_reserve_is_unchanged(context,edge):
    c,clock,plan,claim,freeze=context
    if edge=='mono_work':clock.mono=claim['monotonic_origin']+780
    else:clock.wall=claim['unix_origin']+780
    with pytest.raises(ValueError,match='Original Stage A'):r.original_stage_guard(claim)


@pytest.mark.parametrize('edge',['mono','wall'])
def test_original_three_hour_clock_is_not_reset(context,edge):
    c,clock,plan,claim,freeze=context
    if edge=='mono':clock.mono=plan['monotonic_origin']+10800
    else:clock.wall=plan['unix_origin']+10800
    with pytest.raises(ValueError,match='three-hour'):r.study_guard(plan)


def test_repair_launch_passes_original_origins_and_charges_repair_delay(context,monkeypatch):
    c,clock,plan,claim,freeze=context;events=[]
    sha=c.file_hash(c.OUT/'freeze.json')
    monkeypatch.setattr(l.sys,'argv',['launch','--freeze-sha256',sha])
    monkeypatch.setattr(l.psutil,'Process',lambda *args:SimpleNamespace(create_time=lambda:99.))
    class Child:
        pid=456;returncode=0
        def poll(self):return self.returncode
        def wait(self,timeout):events.append('wait')
        def kill(self):pytest.fail('completed child must not be killed directly')
    def popen(command,**kw):
        assert command[4]=='tools.move_evaluation_inference_recovery_20261023.runner'
        assert kw['env']['DIAGNOSTIC_START_MONOTONIC']=='110.0'
        assert kw['env']['DIAGNOSTIC_START_UNIX']=='1010.0'
        clock.mono+=10;clock.wall+=10
        return Child()
    monkeypatch.setattr(l.subprocess,'Popen',popen)
    class Job:
        def __init__(self,pid):assert pid==456
        def close(self):events.append('closed')
    monkeypatch.setattr(l,'_WindowsJob',Job)
    l.main()
    value=c.read(c.OUT/'execution.json')
    assert events==['closed','wait']
    assert value['charged_owned_job_seconds']==100 and value['remaining_owned_job_seconds']==4400
    assert value['monotonic_origin']==110 and value['dispatch_monotonic']==200
    assert value['study_elapsed_seconds']==110 and value['status']=='completed'
    with pytest.raises(ValueError,match='relaunch'):l.main()


def test_expired_original_stage_cannot_create_another_runner(context,monkeypatch):
    c,clock,plan,claim,freeze=context;clock.mono=890.
    monkeypatch.setattr(l.sys,'argv',['launch','--freeze-sha256',c.file_hash(c.OUT/'freeze.json')])
    monkeypatch.setattr(l.subprocess,'Popen',lambda *a,**k:pytest.fail('expired original stage'))
    with pytest.raises(ValueError,match='Original Stage A'):l.main()
    assert c.read(c.OUT/'execution.json')['exit_code'] is None


def test_direct_runner_cannot_claim_a_fresh_900_seconds(context,monkeypatch):
    c,clock,plan,claim,freeze=context
    monkeypatch.setenv('DIAGNOSTIC_START_MONOTONIC','200')
    monkeypatch.setenv('DIAGNOSTIC_START_UNIX','1100')
    with pytest.raises(ValueError,match='cannot reset'):r.run(c.file_hash(c.OUT/'freeze.json'))
    saved=c.read(c.OUT/'failure.json')
    assert saved['completed_requests']==[] and saved['stage_b_started'] is False


def test_all_scientific_worker_functions_are_identical_to_frozen_source():
    repo=Path(__file__).resolve().parents[1]
    old=ast.parse((repo/'tools/move_evaluation_inference_20261023/worker.py').read_text())
    new=ast.parse((repo/'tools/move_evaluation_inference_recovery_20261023/worker.py').read_text())
    functions=lambda tree:{node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    left,right=functions(old),functions(new)
    assert left.keys()==right.keys()
    for key in left:
        a,b=left[key],right[key]
        # Only resolve the immutable inference module explicitly from the facade.
        for node in ast.walk(b):
            if isinstance(node,ast.ImportFrom) and node.module=='tools.move_evaluation_inference_20261023.inference':
                node.module='inference';node.level=1
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),key


def test_psutil_explanation_source_is_bound_without_model_import():
    item=r.boot_estimate_runtime()
    assert Path(item['absolute_path']).name=='_pswindows.py'
    assert item['maximum_estimate_difference_seconds']==1 and len(item['sha256'])==64
