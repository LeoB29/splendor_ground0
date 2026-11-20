"""Model-free limits and preservation tests for the separate finalization audit."""
import os
from pathlib import Path
import subprocess
import sys
import time
import stat
from types import SimpleNamespace

import psutil
import pytest

from tools.splendorust_reliability_finalization_20261026 import common as c, launch, preservation as p


@pytest.mark.parametrize('event,args',[
    ('subprocess.Popen',('x',)),('os.system',('x',)),('os.exec',('x',)),
    ('os.remove',('x',)),('os.rename',('x','y')),('os.truncate',('x',0)),
    ('open',('AGENT_CURRENT.md','w',os.O_WRONLY)),('open',(7,'w',os.O_WRONLY)),
    ('os.mkdir',('outside',))])
def test_hook_rejects_external_writes_mutations_and_processes(event,args):
    with pytest.raises((RuntimeError,ValueError)):c.audit_event(event,args)


def test_hook_allows_original_read_and_only_new_output_write(tmp_path,monkeypatch):
    out=tmp_path/'finalization';monkeypatch.setattr(c,'OUT',out)
    for old in ('original','recovery_v1','recovery_v2'):
        path=tmp_path/old/'completion.json'
        c.audit_event('open',(str(path),'r',os.O_RDONLY))
        with pytest.raises((RuntimeError,ValueError)):
            c.audit_event('open',(str(path),'w',os.O_WRONLY))
    c.audit_event('open',(str(out/'new.json'),'x',os.O_CREAT|os.O_WRONLY))
    with pytest.raises((RuntimeError,ValueError)):
        c.audit_event('open',(str(out/'..'/'original'/'receipt.json'),'w',os.O_WRONLY))


def test_actual_hook_blocks_original_write_and_child_dispatch(tmp_path):
    # Only this short test harness may spawn; the real auditor forbids children.
    script='''import sys,pathlib,subprocess
from tools.splendorust_reliability_finalization_20261026 import common as c
c.OUT=pathlib.Path(sys.argv[1]);sys.addaudithook(c.audit_event)
blocked=0
for op in (lambda:pathlib.Path(sys.argv[2]).write_text('changed'),lambda:subprocess.Popen([sys.executable,'-c','pass'])):
 try:op()
 except (ValueError,RuntimeError):blocked+=1
assert blocked==2
'''
    original=tmp_path/'original.txt';original.write_text('unchanged')
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(c.ROOT)+os.pathsep+str(c.ROOT/'src')
    result=subprocess.run([sys.executable,'-B','-c',script,str(tmp_path/'out'),str(original)],
        env=environment,cwd=c.ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert original.read_text()=='unchanged'


@pytest.mark.skipif(sys.platform!='win32',reason='Actual Windows Job bootstrap regression')
def test_actual_detached_bootstrap_in_job_has_strictly_zero_children(tmp_path,monkeypatch):
    """Tiny readiness/sleep process only; never an auditor, model or player."""
    from tools.local_player_qualification_128.clock import _WindowsJob
    assert launch.AUDITOR_CREATION_FLAGS==subprocess.DETACHED_PROCESS
    assert not launch.AUDITOR_CREATION_FLAGS&subprocess.CREATE_NO_WINDOW
    ready=tmp_path/'ready.txt'
    script="import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text('ready',encoding='ascii');time.sleep(10)"
    process=job=None;identity=None
    start,wall=time.monotonic(),time.time()
    # Keep real zero-descendant and affinity checks. Resource test values apply
    # only inside this fixture; no frozen configuration is changed.
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'MIN_AVAILABLE',0);monkeypatch.setattr(c,'MIN_DISK',0)
    monkeypatch.setattr(c,'MAX_OUTPUT',1024**2);monkeypatch.setattr(c,'MAX_RSS',1024**3)
    try:
        process=subprocess.Popen([sys.executable,'-B','-u','-c',script,str(ready)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=launch.AUDITOR_CREATION_FLAGS)
        observed=psutil.Process(process.pid)
        identity=dict(pid=process.pid,created=observed.create_time())
        observed.cpu_affinity(list(range(2,20)));job=_WindowsJob(process.pid)
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


@pytest.mark.parametrize('mono,wall',[(1000,150),(150,1000),(1000,1000)])
def test_900_second_clock_expires_on_either_clock_without_phase_reset(monkeypatch,mono,wall):
    assert c.JOB_SECONDS==900
    monkeypatch.setattr(c.time,'monotonic',lambda:mono);monkeypatch.setattr(c.time,'time',lambda:wall)
    envelope=c.Envelope(dict(monotonic_origin=100,unix_origin=100))
    envelope.phase='final';envelope.last_resource=mono
    with pytest.raises(ValueError,match='deadline'):envelope.guard()


def test_clocks_use_smaller_remaining_allowance(monkeypatch):
    monkeypatch.setattr(c.time,'monotonic',lambda:150);monkeypatch.setattr(c.time,'time',lambda:200)
    envelope=c.Envelope(dict(monotonic_origin=100,unix_origin=100))
    assert envelope.remaining()==800 and launch.remaining(100,100)==800


def test_resources_are_amortized_but_every_guard_checks_real_clock(monkeypatch):
    envelope=c.Envelope(dict(monotonic_origin=100,unix_origin=100));envelope.last_resource=150
    monkeypatch.setattr(c.time,'monotonic',lambda:150);monkeypatch.setattr(c.time,'time',lambda:150)
    envelope.process=SimpleNamespace(children=lambda **kw:pytest.fail('Repeated expensive resource scan'))
    envelope.guard()
    monkeypatch.setattr(c.time,'time',lambda:1000)
    with pytest.raises(ValueError,match='deadline'):envelope.guard()


def test_final_guard_forces_a_fresh_resource_sample(monkeypatch):
    envelope=c.Envelope(dict(monotonic_origin=100,unix_origin=100));envelope.last_resource=150
    observed=[];monkeypatch.setattr(envelope,'guard',lambda:observed.append(envelope.last_resource))
    envelope.final_guard()
    assert observed==[-1.]


def test_cleanup_wait_is_clamped_to_both_job_clocks(monkeypatch):
    monkeypatch.setattr(launch.time,'monotonic',lambda:999.5)
    monkeypatch.setattr(launch.time,'time',lambda:999.8)
    waits=[];process=SimpleNamespace(poll=lambda:0,wait=lambda timeout:waits.append(timeout))
    receipt=launch.cleanup(process,None,(),100,100,{'pid':1,'created':1})
    assert receipt['errors']==[] and waits[0]==pytest.approx(.2)


def test_exclusive_receipts_cannot_overwrite_old_or_new_files(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);out=tmp_path/'new';out.mkdir()
    monkeypatch.setattr(c,'OUT',out)
    old=tmp_path/'original';old.mkdir();(old/'completion.json').write_text('incomplete')
    c.save(out/'receipt.json',{'status':'pending'})
    with pytest.raises(FileExistsError):c.save(out/'receipt.json',{'status':'replaced'})
    with pytest.raises(ValueError):c.save(old/'new.json',{})
    assert (old/'completion.json').read_text()=='incomplete'


def test_frozen_source_hash_check_detects_changed_later_input(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    first=tmp_path/'first';later=tmp_path/'later';first.write_text('a');later.write_text('b')
    inputs=[c.artifact(first),c.artifact(later)];later.write_text('changed')
    with pytest.raises(ValueError,match='Changed input'):c.verify(inputs)


def protected_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    trees={}
    for name in ('original','recovery_v1','recovery_v2'):
        directory=tmp_path/name;directory.mkdir()
        (directory/'completion.json').write_text('{"status":"incomplete"}')
        (directory/'nested').mkdir();(directory/'nested'/'evidence.bin').write_bytes(b'evidence')
        trees[name]=[c.artifact(directory/'completion.json'),c.artifact(directory/'nested'/'evidence.bin')]
    envelope=SimpleNamespace(guard=lambda:None,checkpoint=lambda *args:None)
    return trees,envelope


@pytest.mark.parametrize('tree',['original','recovery_v1','recovery_v2'])
@pytest.mark.parametrize('mutation',['changed','same_size','added','deleted','escape','duplicate','case_alias'])
def test_each_entire_protected_tree_rejects_changed_or_missing_inventory(tmp_path,monkeypatch,tree,mutation):
    trees,envelope=protected_fixture(tmp_path,monkeypatch);items=trees[tree]
    if mutation=='changed':(tmp_path/tree/'completion.json').write_text('{"status":"completed"}')
    elif mutation=='same_size':(tmp_path/tree/'nested'/'evidence.bin').write_bytes(b'EVIdence')
    elif mutation=='added':(tmp_path/tree/'invented.json').write_text('{}')
    elif mutation=='deleted':(tmp_path/tree/'completion.json').unlink()
    elif mutation=='escape':items[0]['path']=tree+'/../outside'
    elif mutation=='duplicate':items.append(dict(items[0]))
    elif mutation=='case_alias':items.append(dict(items[0],path=tree+'/COMPLETION.JSON'))
    with pytest.raises((ValueError,FileNotFoundError)):p.preserve_tree(tree,items,envelope)


def test_three_tree_preservation_accepts_all_original_bytes_without_science_calls(tmp_path,monkeypatch):
    trees,envelope=protected_fixture(tmp_path,monkeypatch)
    result=p.preserve_all(dict(preservation_trees=trees),envelope)
    assert result['status']=='passed_final_preservation' and result['total_files']==6
    assert result['total_bytes']==sum(row['bytes'] for items in trees.values() for row in items)
    assert result['scientific_replays_rerun']==result['models_or_games_run']==0
    assert all(row['full_byte_readback'] and row['final_inventory_and_metadata_unchanged'] for row in result['trees'])
    assert all((tmp_path/name/'completion.json').read_text()=='{"status":"incomplete"}' for name in trees)


@pytest.mark.parametrize('mutation',['add_file','remove_file','add_empty_directory'])
def test_post_hash_inventory_rejects_late_tree_mutation(tmp_path,monkeypatch,mutation):
    trees,envelope=protected_fixture(tmp_path,monkeypatch);original=p.inventory;calls=0
    def changing_inventory(relative,boundary=lambda:None):
        nonlocal calls
        calls+=1
        if calls==2:
            if mutation=='add_file':(tmp_path/relative/'late.json').write_text('{}')
            elif mutation=='remove_file':(tmp_path/relative/'completion.json').unlink()
            else:(tmp_path/relative/'late_empty').mkdir()
        return original(relative,boundary)
    monkeypatch.setattr(p,'inventory',changing_inventory)
    with pytest.raises(ValueError,match='inventory/identity changed'):
        p.preserve_tree('original',trees['original'],envelope)


@pytest.mark.parametrize('value',['.','..','a/../b','a//b','a/./b','C:/outside','/outside','a\\b'])
def test_manifest_paths_require_canonical_relative_components(value):
    with pytest.raises(ValueError):p.safe_relative(value)


def test_synthetic_windows_reparse_flag_is_rejected_even_for_regular_mode():
    apparent=SimpleNamespace(st_mode=stat.S_IFREG,st_file_attributes=p.REPARSE)
    with pytest.raises(ValueError,match='reparse'):p.no_redirect(apparent,'protected/file')


def test_actual_symlink_in_protected_tree_is_rejected_when_supported(tmp_path,monkeypatch):
    trees,envelope=protected_fixture(tmp_path,monkeypatch)
    target=tmp_path/'outside.txt';target.write_text('outside')
    link=tmp_path/'original'/'redirect'
    try:link.symlink_to(target)
    except (OSError,NotImplementedError):pytest.skip('Host cannot create test symlinks')
    with pytest.raises(ValueError,match='Symlink/reparse'):
        p.preserve_tree('original',trees['original'],envelope)


def test_tree_hashing_stops_on_shared_deadline(tmp_path,monkeypatch):
    trees,envelope=protected_fixture(tmp_path,monkeypatch)
    def expired():raise ValueError('shared deadline')
    envelope.guard=expired
    with pytest.raises(ValueError,match='shared deadline'):
        p.preserve_tree('original',trees['original'],envelope)


def test_open_handle_identity_check_rejects_change_during_readback(tmp_path,monkeypatch):
    trees,envelope=protected_fixture(tmp_path,monkeypatch);real_fstat=p.os.fstat;calls=0
    def changing_fstat(descriptor):
        nonlocal calls
        calls+=1;result=real_fstat(descriptor)
        if calls==1:(tmp_path/'original'/'completion.json').write_text('changed while open')
        return result
    monkeypatch.setattr(p.os,'fstat',changing_fstat)
    with pytest.raises(ValueError,match='changed while hashing'):
        p.preserve_tree('original',trees['original'],envelope)


def test_guard_rejects_unexpected_child_even_within_time_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(c.time,'monotonic',lambda:150);monkeypatch.setattr(c.time,'time',lambda:150)
    envelope=c.Envelope(dict(monotonic_origin=100,unix_origin=100))
    envelope.process=SimpleNamespace(children=lambda **kw:[SimpleNamespace(pid=123)])
    with pytest.raises(ValueError,match='Unexpected auditor child'):envelope.guard()


def test_windows_open_handle_ctime_can_differ_while_all_identity_and_byte_checks_pass(tmp_path,monkeypatch):
    trees,envelope=protected_fixture(tmp_path,monkeypatch);real_fstat=p.os.fstat
    def handle_semantics(descriptor):
        value=real_fstat(descriptor)
        fields={key:getattr(value,key) for key in ('st_mode','st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
        fields['st_file_attributes']=getattr(value,'st_file_attributes',0)
        fields['st_ctime_ns']+=123456
        return SimpleNamespace(**fields)
    monkeypatch.setattr(p.os,'fstat',handle_semantics)
    receipt=p.preserve_tree('original',trees['original'],envelope)
    assert receipt['status']=='passed' and receipt['files']==2


def test_open_handle_ctime_change_is_still_rejected_against_its_own_baseline(tmp_path,monkeypatch):
    trees,envelope=protected_fixture(tmp_path,monkeypatch);real_fstat=p.os.fstat;calls=0
    def changing_handle(descriptor):
        nonlocal calls
        calls+=1;value=real_fstat(descriptor)
        fields={key:getattr(value,key) for key in ('st_mode','st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
        fields['st_file_attributes']=getattr(value,'st_file_attributes',0)
        fields['st_ctime_ns']+=calls*123456
        return SimpleNamespace(**fields)
    monkeypatch.setattr(p.os,'fstat',changing_handle)
    with pytest.raises(ValueError,match='changed while hashing'):
        p.preserve_tree('original',trees['original'],envelope)
