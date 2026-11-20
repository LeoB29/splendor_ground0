"""Kernel membership and cleanup counterexamples without arbitrary process kills."""
from types import SimpleNamespace
from copy import deepcopy
import psutil
import pytest
from tools.splendorust_collection4096_resume_20261118_v2 import ownership as o


def ticks(seconds):return o.FILETIME_EPOCH+int(seconds*10000000)


class FakeKernel:
    def __init__(self):
        self.rows={10:dict(member=True,alive=True,created=100.,rss=20),
                   20:dict(member=False,alive=True,created=200.,rss=99999)}
        self.closed=[]
    def members(self,handle):assert handle==444;return [10,20]
    def open(self,pid):return pid
    def close(self,handle):self.closed.append(handle)
    def belongs(self,handle,job):assert job==444;return self.rows[handle]['member']
    def alive(self,handle):return self.rows[handle]['alive']
    def creation(self,handle):return ticks(self.rows[handle]['created'])
    def pid(self,handle):return handle
    def process(self,pid):
        row=self.rows[pid]
        if not row['alive']:raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda:row['created'],memory_info=lambda:SimpleNamespace(rss=row['rss']),
            children=lambda **kw:(_ for _ in ()).throw(AssertionError('Parent PID traversal forbidden')))


def test_pid_from_stale_list_is_not_ownership_and_never_counted():
    api=FakeKernel();job=SimpleNamespace(handle=444);observed={}
    assert o.observe_job(job,observed,api=api,process_factory=api.process)==20
    assert observed=={(10,100.):dict(pid=10,created=100.)}
    assert api.closed==[10,20]
    assert [x['identity'] for x in o.ownership_proofs(job)]==[dict(pid=10,created=100.)]


def test_dead_member_disappears_without_foreign_pid_substitution():
    api=FakeKernel();api.rows[10]['alive']=False;observed={}
    assert o.observe_job(SimpleNamespace(handle=444),observed,api=api,process_factory=api.process)==0
    assert not observed


def test_native_and_psutil_birth_must_agree():
    api=FakeKernel();observed={}
    def wrong(pid):return SimpleNamespace(create_time=lambda:101.,memory_info=lambda:SimpleNamespace(rss=10))
    with pytest.raises(ValueError,match='birth'):
        o.observe_job(SimpleNamespace(handle=444),observed,api=api,process_factory=wrong)
    assert not observed;assert api.closed==[10]


def test_pid_rebirth_during_sampling_is_rejected():
    api=FakeKernel();observed={};calls=[]
    def unstable(pid):
        calls.append(pid)
        return SimpleNamespace(create_time=lambda:100. if len(calls)==1 else 101.,memory_info=lambda:SimpleNamespace(rss=10))
    with pytest.raises(ValueError,match='rebirth'):
        o.observe_job(SimpleNamespace(handle=444),observed,api=api,process_factory=unstable)
    assert not observed;assert api.closed==[10]


def test_missing_job_handle_has_no_traversal_fallback():
    with pytest.raises(ValueError,match='handle'):
        o.observe_job(SimpleNamespace(handle=None),{},api=FakeKernel())


class FakePopen:
    pid=10;_handle=10
    def __init__(self,api):self.api=api;self.kills=0;self.waits=0
    def poll(self):return None if self.api.rows[10]['alive'] else 0
    def kill(self):self.kills+=1;self.api.rows[10]['alive']=False
    def wait(self,timeout):self.waits+=1;return 0


def test_fallback_kills_only_retained_original_handle_and_leaves_foreign_alive():
    api=FakeKernel();proc=FakePopen(api)
    o.finish_exact_process(proc,dict(pid=10,created=100.),api=api,process_factory=api.process)
    assert proc.kills==proc.waits==1
    assert api.rows[20]['alive']


@pytest.mark.parametrize('identity',[dict(pid=20,created=200.),dict(pid=10,created=101.),None])
def test_mismatched_fallback_identity_never_kills(identity):
    api=FakeKernel();proc=FakePopen(api)
    with pytest.raises(ValueError):o.finish_exact_process(proc,identity,api=api,process_factory=api.process)
    assert proc.kills==0 and all(x['alive'] for x in api.rows.values())


def test_foreign_job_candidate_is_never_opened_for_termination():
    api=FakeKernel();job=SimpleNamespace(handle=444);observed={}
    o.observe_job(job,observed,api=api,process_factory=api.process)
    assert set(observed)=={(10,100.)}
    assert api.rows[20]['alive'] and o.ownership_proofs(job)[0]['kernel_membership']


def receipt():
    api=FakeKernel();job=SimpleNamespace(handle=444);observed={}
    o.observe_job(job,observed,api=api,process_factory=api.process)
    return dict(status='passed',identity=dict(pid=10,created=100.),ownership_method=o.METHOD,
        observed_owned_identities=list(observed.values()),job_membership_observations=o.ownership_proofs(job))


def test_recorded_members_have_one_exact_kernel_proof_each():
    result=o.verify_observations(receipt())
    assert result['proof_count']==1 and result['verified_owned_identities']==[dict(pid=10,created=100.)]


def test_missing_kernel_proof_cannot_legitimize_parent_pid_observation():
    value=receipt();value['job_membership_observations']=[]
    with pytest.raises(ValueError,match='Exactly one'):o.verify_observations(value)


def test_extra_foreign_kernel_proof_rejected():
    value=receipt();extra=deepcopy(value['job_membership_observations'][0])
    extra['identity']=dict(pid=20,created=200.);extra['creation_filetime']=ticks(200.)
    value['job_membership_observations'].append(extra)
    with pytest.raises(ValueError,match='Exactly one'):o.verify_observations(value)


def test_foreign_root_cannot_borrow_a_valid_member_receipt():
    value=receipt();value['identity']=dict(pid=20,created=200.)
    with pytest.raises(ValueError,match='Successful launched root'):o.verify_observations(value)
