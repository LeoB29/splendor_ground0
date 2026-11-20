"""No-model regression tests for automatic admission and immutable ownership."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.splendorust_selfplay_pilot_20261102 import workflow as w, common as c
from tools.splendorust_selfplay_pilot_20261102 import qualification_control as control


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(c.canonical(value)+'\n',encoding='utf-8')


def workspace(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'QUALIFICATION_OUT',tmp_path/'qualification')
    monkeypatch.setattr(c,'PILOT_OUT',tmp_path/'pilot')
    monkeypatch.setattr(c,'OUT',c.QUALIFICATION_OUT)
    monkeypatch.setattr(w,'OUT',tmp_path/'workflow')


def acceptance_fixture(monkeypatch,tmp_path,scope='qualification'):
    workspace(monkeypatch,tmp_path);out=w.scope_out(scope)
    write(out/'freeze.json',dict(inputs=[],scope=scope,phase=scope))
    sha=c.file_hash(out/'freeze.json')
    coverage=dict(completed_games=4 if scope=='qualification' else 256,
                  qualification_only=scope=='qualification',pilot_data=scope=='pilot')
    review=dict(approved=True,freeze_sha256=sha,coverage=coverage,reviewer=dict(pid=902,created=1.))
    write(out/'independent_review.json',review)
    write(out/'completion.json',dict(status='completed_verified',freeze_sha256=sha,coverage=coverage,failure=None))
    for name in ('execution.json','watchdog_execution.json','independent_review_os_exit.json'):
        write(out/name,dict(status='completed',exit_code=0,freeze_sha256=sha,cleanup_errors=[],
            cleanup_receipt={'unconfirmed':[]},reviewer=review['reviewer']))
    for name in ('guardian_claim.json','guardian_owner.json','launch.json','fit_execution.json'):
        write(out/name,dict(pid=900,created=1.))
    for folder in (out/'workers/source',out/'assessment/workers/candidate'):
        write(folder/'initialization.json',dict(identity=dict(pid=901,created=1.)))
        write(folder/'cleanup.json',dict(pid=901,created=1.))
    for phase,count in [('operational',6),('main',0 if scope=='qualification' else 400)]:
        for index in range(count):write(out/'assessment/games'/phase/f'g{index:03d}/result.json',{})
    monkeypatch.setattr(w,'require_absent',lambda items:None)
    receipt=dict(status='completed',exit_code=0,cleanup_errors=[],scope=scope,
                 operation='guardian',freeze_sha256=sha,identity=dict(pid=903,created=1.))
    return out,sha,receipt


@pytest.mark.parametrize('scope',['qualification','pilot'])
def test_acceptance_requires_actual_exits_and_full_distinct_samples(monkeypatch,tmp_path,scope):
    out,sha,receipt=acceptance_fixture(monkeypatch,tmp_path,scope)
    value=w.accept_scope(scope,sha,receipt)
    assert value['completed_source_games']==(4 if scope=='qualification' else 256)
    assert value['independent_assessment_games']==(0 if scope=='qualification' else 400)
    assert value['reference_promoted'] is False
    assert len(value['exact_owner_identities_absent'])==4


@pytest.mark.parametrize('field,bad',[('exit_code',1),('status','running'),('cleanup_errors',['unconfirmed']),
    ('scope','pilot'),('operation','fitter'),('freeze_sha256','0'*64)])
def test_completion_file_cannot_replace_owning_handle(monkeypatch,tmp_path,field,bad):
    out,sha,receipt=acceptance_fixture(monkeypatch,tmp_path)
    receipt[field]=bad
    with pytest.raises(ValueError):w.accept_scope('qualification',sha,receipt)
    assert not(out/'qualification_acceptance.json').exists()


@pytest.mark.parametrize('failure',['review','reader_exit','sample','cleanup','live_owner'])
def test_no_acceptance_after_failed_check(monkeypatch,tmp_path,failure):
    out,sha,receipt=acceptance_fixture(monkeypatch,tmp_path)
    if failure=='review':
        p=out/'independent_review.json';v=c.read(p);v['approved']=False;write(p,v)
    elif failure=='reader_exit':
        p=out/'independent_review_os_exit.json';v=c.read(p);v['exit_code']=1;write(p,v)
    elif failure=='sample':(out/'assessment/games/operational/g005/result.json').unlink()
    elif failure=='cleanup':
        p=out/'execution.json';v=c.read(p);v['cleanup_receipt']['unconfirmed']=[{'pid':901}];write(p,v)
    else:monkeypatch.setattr(w,'require_absent',lambda items:(_ for _ in ()).throw(ValueError('still live')))
    with pytest.raises(ValueError):w.accept_scope('qualification',sha,receipt)
    assert not(out/'qualification_acceptance.json').exists()


def test_pid_reuse_does_not_misidentify_new_process(monkeypatch):
    monkeypatch.setattr(w.psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:2.))
    w.require_absent([dict(pid=123,created=1.)])
    with pytest.raises(ValueError):w.require_absent([dict(pid=123,created=2.)])


def test_scope_is_separate_environment_not_global_rebinding():
    before=c.SCOPE
    assert w.child_environment('pilot')['SELFPLAY_JOB_SCOPE']=='pilot'
    assert w.child_environment('qualification')['SELFPLAY_JOB_SCOPE']=='qualification'
    assert c.SCOPE==before
    with pytest.raises(ValueError):w.child_environment('recovery')


@pytest.mark.parametrize('failure',[None,'qualification_guardian','qualification_acceptance','pilot_freeze','pilot_admission'])
def test_automatic_sequence_only_advances_after_every_gate(monkeypatch,tmp_path,failure):
    workspace(monkeypatch,tmp_path);events=[]
    write(c.PILOT_OUT/'freeze.json',{'fixed':True})
    class Owner:
        def guard(self):pass
        def admit(self,scope):events.append(scope+'_resources')
        def process(self,scope,operation,args,maximum,**kwargs):
            label=scope+'_'+operation;events.append(label)
            if label==failure:raise ValueError('preserved failure')
            return {'fixed':True}
        def event(self,*args,**kwargs):pass
    def accept(scope,sha,receipt,**kwargs):
        label=scope+'_acceptance';events.append(label)
        if label==failure:raise ValueError('preserved failure')
        if scope=='pilot':write(c.PILOT_OUT/'final_closeout.json',{})
        return {'passed':True}
    monkeypatch.setattr(w,'accept_scope',accept)
    plan=dict(qualification_freeze={'sha256':'a'*64},tests={'path':'tests.json'},reviews=[{'path':'review.json'}])
    if failure:
        with pytest.raises(ValueError):w.run_sequence(plan,Owner())
        assert events[-1]==failure
        assert 'pilot_guardian' not in events
    else:
        assert w.run_sequence(plan,Owner())=={'passed':True}
        assert events==['qualification_resources','qualification_admission','qualification_guardian',
            'qualification_acceptance','pilot_freeze','pilot_resources','pilot_admission','pilot_guardian','pilot_acceptance']
    assert len(events)==len(set(events))


def test_root_accounting_owns_nested_assessment_workers_once(monkeypatch,tmp_path):
    workspace(monkeypatch,tmp_path)
    env=control.Envelope(dict(monotonic_origin=0.,unix_origin=0.))
    env.evidence={'baseline':0,'parent_bytes':0,'workers':{}}
    folder=c.OUT/'assessment/workers/candidate';env.worker_accounting={folder:object()}
    env.note_write(folder/'input.json',100)
    env.note_write(c.OUT/'assessment/games/operational/g000/result.json',200)
    assert env.evidence['parent_bytes']==200


def test_exact_one_gib_accounting_cap_does_not_relax_other_invariants(tmp_path):
    import threading
    from tools.splendorust_selfplay_pilot_20261102.accounting import WorkerAccounting,LOG_NAMES
    account=WorkerAccounting(tmp_path/'workers/candidate')
    child=SimpleNamespace(output_dir=account.folder,_log_lock=threading.Lock(),_logs={x:None for x in LOG_NAMES},max_log_bytes=c.GiB)
    account.bind_child(child)
    child.max_log_bytes=2*c.GiB
    with pytest.raises(ValueError):account.bind_child(child)


@pytest.mark.parametrize('failed',[False,True])
def test_actual_windows_owned_model_free_subprocess_exit(monkeypatch,tmp_path,failed):
    workspace(monkeypatch,tmp_path)
    write(w.OUT/'plan.json',{})
    owner=w.Owner({})
    monkeypatch.setattr(w,'PACKAGE','json')
    args=['tool','--not-a-valid-option' if failed else '--help']
    if failed:
        with pytest.raises(ValueError):owner.process('qualification','probe',args,10)
    else:
        result=owner.process('qualification','probe',args,10)
        assert result['exit_code']==0 and result['cleanup_errors']==[]
    receipt=c.read(w.OUT/'qualification_probe/exit.json')
    assert receipt['status']==('failed' if failed else 'completed')
    w.require_absent([receipt['identity']])


def test_acceptance_clock_expiry_cannot_publish_success(monkeypatch,tmp_path):
    out,sha,receipt=acceptance_fixture(monkeypatch,tmp_path)
    def expired():raise ValueError('outer clock expired')
    with pytest.raises(ValueError,match='clock expired'):
        w.accept_scope('qualification',sha,receipt,boundary=expired)
    assert not(out/'qualification_acceptance.json').exists()
