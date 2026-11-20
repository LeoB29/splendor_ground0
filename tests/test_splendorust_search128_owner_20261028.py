"""Model-free owning-handle cleanup and shared acceptance-clock regressions."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_search128_strength_20261028 import launch_owner as owner, guardian, common as c


def fixture(tmp_path, *, fail_affinity=False, exit_code=0):
    class P:
        pid=1234
        returncode=exit_code
        waits=[]
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.waits.append(timeout);return self.returncode
    process=P()
    class Child:
        pid=1234
        def create_time(self):return 42.
        def cpu_affinity(self,value):
            if fail_affinity:raise OSError('affinity setup failed')
        def nice(self,value):pass
        def memory_info(self):return SimpleNamespace(rss=100)
        def children(self,recursive=False):return []
    child=Child();cleanups=[]
    def cleanup(process,job,streams,**kw):
        cleanups.append(dict(process=process,job=job,**kw))
        for stream in streams:stream.close()
        if process.returncode is None:process.returncode=1
        return dict(errors=[],confirmed=[{'pid':1234,'created':42.}],unconfirmed=[])
    freeze={'placement':{'role_masks':{'isolated':[2,3]}}}
    return process,child,cleanups,cleanup,freeze


def test_guardian_claim_publication_is_atomic_before_consumers():
    source=Path(guardian.__file__).read_text();tree=ast.parse(source)
    calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.args:
            if any(name in ast.unparse(node.args[0]) for name in ('guardian_claim.json','guardian_owner.json')) and node.func.attr in ('save','atomic_save'):
                calls.append(node.func.attr)
    assert calls==['atomic_save','atomic_save']


def test_acceptance_uses_original_final_and_global_limits_not_a_fresh_hour():
    claim=dict(freeze_sha256='x',monotonic_origin=100.,unix_origin=1000.)
    completion=dict(freeze_sha256='x',independent_review_maximum_seconds=3600,
        independent_acceptance_deadline_monotonic=85500.,independent_acceptance_deadline_unix=86400.)
    assert owner.acceptance_deadlines(claim,completion,85000.,85900.)==(85500.,86400.)
    completion['independent_acceptance_deadline_monotonic']=999999.
    completion['independent_acceptance_deadline_unix']=999999.
    assert owner.acceptance_deadlines(claim,completion,85000.,85900.)==(86500.,87400.)
    completion['freeze_sha256']='wrong'
    with pytest.raises(ValueError):owner.acceptance_deadlines(claim,completion,85000.,85900.)


def test_either_original_clock_expiration_controls_remaining():
    with patch.object(owner.time,'monotonic',return_value=20.),patch.object(owner.time,'time',return_value=100.):
        assert owner.deadline_remaining(19.,200.)==-1.
        assert owner.deadline_remaining(30.,99.)==-1.


def test_success_requires_actual_owned_wait_and_cleanup(tmp_path):
    process,child,cleanups,cleanup,freeze=fixture(tmp_path)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(owner.subprocess,'Popen',return_value=process) as spawn,\
        patch.object(owner.psutil,'Process',return_value=child),patch.object(owner,'_WindowsJob',return_value='job'),\
        patch.object(owner,'cleanup_owned',side_effect=cleanup),patch.object(owner,'deadline_remaining',return_value=10.):
        result=owner.observe(['python','-c','pass'],'guardian',freeze,'x',lambda:(10000.,20000.),permit_children=True)
    assert spawn.call_args.kwargs['creationflags']==getattr(owner.subprocess,'DETACHED_PROCESS',0)
    assert result['exit_code']==0 and result['cleanup_reason'] is None
    assert process.waits==[0] and len(cleanups)==1
    assert cleanups[0]['root_identity']==dict(pid=1234,created=42.)
    assert result['observed_by']=='owned Process handle'
    assert (tmp_path/'guardian_launch.json').is_file()


def test_failure_after_spawn_still_closes_job_and_retains_actual_exit(tmp_path):
    process,child,cleanups,cleanup,freeze=fixture(tmp_path,fail_affinity=True,exit_code=None)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(owner.subprocess,'Popen',return_value=process),\
        patch.object(owner.psutil,'Process',return_value=child),patch.object(owner,'_WindowsJob',return_value='job'),\
        patch.object(owner,'cleanup_owned',side_effect=cleanup),patch.object(owner,'deadline_remaining',return_value=10.):
        result=owner.observe(['python','-c','pass'],'guardian',freeze,'x',lambda:(10000.,20000.),permit_children=True)
    assert result['exit_code']==1 and 'affinity setup failed' in result['cleanup_reason']
    assert len(cleanups)==1 and cleanups[0]['process'] is process and cleanups[0]['job']=='job'


def test_deadline_supplier_failure_after_spawn_cannot_skip_cleanup(tmp_path):
    process,child,cleanups,cleanup,freeze=fixture(tmp_path,exit_code=None)
    calls=[]
    def deadlines():
        calls.append(1)
        if len(calls)>1:raise ValueError('malformed guardian clock claim')
        return 10000.,20000.
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(owner.subprocess,'Popen',return_value=process),\
        patch.object(owner.psutil,'Process',return_value=child),patch.object(owner,'_WindowsJob',return_value='job'),\
        patch.object(owner,'cleanup_owned',side_effect=cleanup),patch.object(owner,'deadline_remaining',return_value=10.):
        result=owner.observe(['python','-c','pass'],'guardian',freeze,'x',deadlines,permit_children=True)
    assert len(cleanups)==1 and cleanups[0]['process'] is process
    assert result['exit_code']==1 and 'malformed guardian clock claim' in result['cleanup_reason']


def test_no_spawn_when_original_deadline_already_expired(tmp_path):
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(owner.subprocess,'Popen') as popen,\
        patch.object(owner,'cleanup_owned',return_value={'errors':[]}),patch.object(owner,'deadline_remaining',return_value=-.01):
        result=owner.observe(['python','-c','pass'],'guardian',{'placement':{}},'x',lambda:(1.,1.))
    assert not popen.called and result['exit_code'] is None and result['cleanup_reason']



def test_post_spawn_job_assignment_failure_closes_actual_child(tmp_path):
    process,child,cleanups,cleanup,freeze=fixture(tmp_path,exit_code=None)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(owner.subprocess,'Popen',return_value=process),\
        patch.object(owner.psutil,'Process',return_value=child),patch.object(owner,'_WindowsJob',side_effect=OSError('job assignment failed')),\
        patch.object(owner,'cleanup_owned',side_effect=cleanup),patch.object(owner,'deadline_remaining',return_value=10.):
        result=owner.observe(['python','-c','pass'],'guardian',freeze,'x',lambda:(10000.,20000.),permit_children=True)
    assert len(cleanups)==1 and cleanups[0]['process'] is process and cleanups[0]['job'] is None
    assert cleanups[0]['root_identity']==dict(pid=1234,created=42.)
    assert result['exit_code']==1 and 'job assignment failed' in result['cleanup_reason']



@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),True])
def test_nonfinite_or_boolean_clock_cannot_spawn_or_extend_acceptance(tmp_path,invalid):
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(owner.subprocess,'Popen') as popen,\
        patch.object(owner,'cleanup_owned',return_value={'errors':[]}):
        result=owner.observe(['python','-c','pass'],'guardian',{'placement':{}},'x',lambda:(invalid,100.))
    assert not popen.called and result['cleanup_reason'] and result['exit_code'] is None
    claim=dict(freeze_sha256='x',monotonic_origin=invalid,unix_origin=1000.)
    completion=dict(freeze_sha256='x',independent_review_maximum_seconds=3600,
        independent_acceptance_deadline_monotonic=85500.,independent_acceptance_deadline_unix=86400.)
    with pytest.raises(ValueError):owner.acceptance_deadlines(claim,completion,85000.,85900.)
