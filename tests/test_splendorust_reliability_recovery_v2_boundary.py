"""Model-free counterexamples for separate recovery clocks and immutable evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import psutil
import pytest
from tools.splendorust_reliability_recovery_v2_20261026 import common as c, preparation as p, runner, launch

@pytest.mark.parametrize('event,args',[
    ('subprocess.Popen',('x',)),('os.system',('x',)),('os.exec',('x',)),
    ('os.remove',('x',)),('os.rename',('x','y')),('os.truncate',('x',0)),
    ('open',('AGENT_CURRENT.md','w',os.O_WRONLY)),('open',(7,'w',os.O_WRONLY)),
    ('os.mkdir',('outside',)),('open',(str(c.ORIGINAL/'completion.json'),'w',os.O_WRONLY))])
def test_hook_rejects_external_writes_and_processes(event,args):
    with pytest.raises((RuntimeError,ValueError)):c.audit_event(event,args)

def test_hook_allows_only_new_output_write_and_original_read():
    c.audit_event('open',(str(c.OUT/'new.json'),'x',os.O_CREAT|os.O_WRONLY))
    c.audit_event('open',(str(c.ORIGINAL/'completion.json'),'r',os.O_RDONLY))

def test_actual_hook_child_blocks_write_and_spawn(tmp_path):
    # Test harness only; the real auditor is never allowed any child.
    script='''import sys, pathlib, subprocess
from tools.splendorust_reliability_recovery_v2_20261026 import common as c
c.OUT=pathlib.Path(sys.argv[1]);sys.addaudithook(c.audit_event)
blocked=0
for op in (lambda: pathlib.Path(sys.argv[2]).write_text('bad'), lambda: subprocess.Popen([sys.executable,'-c','pass'])):
 try:op()
 except (ValueError,RuntimeError):blocked+=1
assert blocked==2
print('blocked')
'''
    env=os.environ.copy();env['PYTHONPATH']=str(c.ROOT)+os.pathsep+str(c.ROOT/'src')
    result=subprocess.run([sys.executable,'-B','-c',script,str(tmp_path/'allowed'),str(tmp_path/'forbidden')],env=env,cwd=c.ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows Job bootstrap regression')
def test_actual_detached_bootstrap_in_job_has_strictly_zero_children(tmp_path,monkeypatch):
    """Only a tiny readiness/sleep fixture; never an auditor or player/model."""
    from tools.local_player_qualification_128.clock import _WindowsJob
    assert launch.AUDITOR_CREATION_FLAGS == subprocess.DETACHED_PROCESS
    assert not launch.AUDITOR_CREATION_FLAGS & subprocess.CREATE_NO_WINDOW
    ready=tmp_path/'ready.txt'
    script="import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('ready',encoding='ascii'); time.sleep(10)"
    process=job=None
    start,wall=time.monotonic(),time.time()
    identity=None
    # Exercise the real zero-descendant and affinity guard. Resource limits are
    # irrelevant to this native bootstrap fixture and remain unchanged in code.
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'MIN_AVAILABLE',0)
    monkeypatch.setattr(c,'MIN_DISK',0)
    monkeypatch.setattr(c,'MAX_OUTPUT',1024**2)
    monkeypatch.setattr(c,'MAX_RSS',1024**3)
    try:
        process=subprocess.Popen([sys.executable,'-B','-u','-c',script,str(ready)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=launch.AUDITOR_CREATION_FLAGS)
        observed=psutil.Process(process.pid)
        identity=dict(pid=process.pid,created=observed.create_time())
        observed.cpu_affinity(list(range(2,20)))
        job=_WindowsJob(process.pid)
        deadline=time.monotonic()+5
        while not ready.exists():
            assert process.poll() is None,'Detached fixture exited before readiness'
            assert time.monotonic()<deadline,'Detached fixture readiness timed out'
            time.sleep(.025)
        assert ready.read_text(encoding='ascii')=='ready'
        for _ in range(5):
            assert process.poll() is None
            assert observed.children(recursive=True)==[]
            launch.resource_guard(process)
            time.sleep(.05)
    finally:
        receipt=launch.cleanup(process,job,(),start,wall,identity)
        assert receipt['errors']==[] and receipt['unconfirmed']==[]
        if process is not None:assert process.poll() is not None


def test_expired_original_clock_rejects_new_v2_watchdog_without_writing(monkeypatch):
    monkeypatch.setattr(sys,'argv',['launch','--freeze-sha256','fixture'])
    monkeypatch.setattr(launch.time,'monotonic',lambda:c.SHARED_MONOTONIC_ORIGIN+c.JOB_SECONDS+.01)
    monkeypatch.setattr(launch.time,'time',lambda:c.SHARED_UNIX_ORIGIN+10)
    monkeypatch.setattr(c,'save',lambda *args:pytest.fail('Expired original clock attempted a new claim'))
    with pytest.raises(ValueError,match='Original shared recovery deadline'):
        launch.main()


def test_v2_origins_match_frozen_first_attempt_and_relaunch_is_disclosed():
    import ast
    import inspect
    parent=c.read(c.PARENT/'watchdog_claim.json')
    assert c.SHARED_MONOTONIC_ORIGIN==parent['monotonic_origin']
    assert c.SHARED_UNIX_ORIGIN==parent['unix_origin']
    nodes=ast.walk(ast.parse(inspect.getsource(p.prepare)))
    keywords={node.arg:node.value.value for node in nodes if isinstance(node,ast.keyword) and isinstance(node.value,ast.Constant)}
    assert keywords['bootstrap_relaunches']==1
    assert keywords['model_or_decision_retries']==0

@pytest.mark.parametrize('mono,wall',[(1900,1000),(1000,1900),(1900,1900)])
def test_shared_clock_expiration_cannot_be_reset_by_phase(monkeypatch,mono,wall):
    monkeypatch.setattr(c.time,'monotonic',lambda:mono);monkeypatch.setattr(c.time,'time',lambda:wall)
    e=c.Envelope(dict(monotonic_origin=100,unix_origin=100));e.phase='late';e.last_resource=mono
    with pytest.raises(ValueError,match='deadline'):e.guard()

def test_shared_clock_uses_smaller_remaining_allowance(monkeypatch):
    monkeypatch.setattr(c.time,'monotonic',lambda:150);monkeypatch.setattr(c.time,'time',lambda:200)
    e=c.Envelope(dict(monotonic_origin=100,unix_origin=100))
    assert e.remaining()==1700 and launch.remaining(100,100)==1700

def test_resource_guard_is_amortized_but_clock_not(monkeypatch):
    e=c.Envelope(dict(monotonic_origin=100,unix_origin=100))
    monkeypatch.setattr(c.time,'monotonic',lambda:150);monkeypatch.setattr(c.time,'time',lambda:150)
    e.last_resource=150
    e.process=SimpleNamespace(children=lambda **kw:pytest.fail('Repeated expensive resource scan'))
    e.guard()
    monkeypatch.setattr(c.time,'time',lambda:2000)
    with pytest.raises(ValueError,match='deadline'):e.guard()

def test_final_guard_forces_resource_sample(monkeypatch):
    e=c.Envelope(dict(monotonic_origin=100,unix_origin=100));e.last_resource=150
    seen=[]
    monkeypatch.setattr(e,'guard',lambda:seen.append(e.last_resource))
    e.final_guard()
    assert seen==[-1.]

def fixture_snapshot(tmp_path,monkeypatch):
    old=tmp_path/'original';old.mkdir();out=tmp_path/'recovery';out.mkdir()
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'ORIGINAL',old);monkeypatch.setattr(c,'OUT',out)
    (old/'completion.json').write_text('{"status":"incomplete"}')
    return old,out,[c.artifact(old/'completion.json')]

@pytest.mark.parametrize('mutation',['changed','added','deleted','escape','duplicate'])
def test_entire_snapshot_rejects_mutation(tmp_path,monkeypatch,mutation):
    old,out,items=fixture_snapshot(tmp_path,monkeypatch)
    if mutation=='changed':(old/'completion.json').write_text('{"status":"success"}')
    if mutation=='added':(old/'invented.json').write_text('{}')
    if mutation=='deleted':(old/'completion.json').unlink()
    if mutation=='escape':items[0]['path']='original/../../escape'
    if mutation=='duplicate':items+=items
    envelope=SimpleNamespace(guard=lambda:None)
    with pytest.raises((ValueError,FileNotFoundError)):runner.finish_snapshot(dict(original_snapshot=items,inputs=[]),envelope)

def test_preservation_readback_and_exclusive_output(tmp_path,monkeypatch):
    old,out,items=fixture_snapshot(tmp_path,monkeypatch)
    receipt=runner.finish_snapshot(dict(original_snapshot=items,inputs=[]),SimpleNamespace(guard=lambda:None))
    assert receipt['extra_missing_or_changed_original_files']==0
    c.save(out/'receipt.json',receipt)
    with pytest.raises(FileExistsError):c.save(out/'receipt.json',receipt)
    with pytest.raises(ValueError):c.save(old/'new.json',{})
    assert (old/'completion.json').read_text()=='{"status":"incomplete"}'

def test_cleanup_wait_clamped_to_one_job_deadline(monkeypatch):
    monkeypatch.setattr(launch.time,'monotonic',lambda:1899.5)
    monkeypatch.setattr(launch.time,'time',lambda:1899.8)
    waits=[]
    process=SimpleNamespace(poll=lambda:0,wait=lambda timeout:waits.append(timeout))
    receipt=launch.cleanup(process,None,(),100,100,{'pid':1,'created':1})
    assert receipt['errors']==[] and waits[0]==pytest.approx(.2)

def test_frozen_input_verify_covers_late_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    a=tmp_path/'a';b=tmp_path/'b';a.write_text('a');b.write_text('b')
    rows=[c.artifact(a),c.artifact(b)];b.write_text('changed')
    with pytest.raises(ValueError,match='Changed input'):c.verify(rows)

def test_pipeline_stops_if_runtime_verification_fails(tmp_path,monkeypatch):
    from tools.splendorust_reliability_recovery_v2_20261026 import runtime,review
    from tools import future_training_rules_20261021 as rules
    old,out,items=fixture_snapshot(tmp_path,monkeypatch);(old/'freeze.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(rules,'verify_runtime_sources',lambda:calls.append('source_guard') or {})
    def fail(*args):raise ValueError('runtime failure')
    monkeypatch.setattr(runtime,'verify_runtime_and_prior',fail)
    monkeypatch.setattr(review,'run_review',lambda *args:pytest.fail('Review after failed runtime'))
    envelope=SimpleNamespace(checkpoint=lambda *args:None)
    with pytest.raises(ValueError,match='runtime failure'):runner.execute({},envelope)
    assert calls==['source_guard'] and not (out/'verification_summary.json').exists()
