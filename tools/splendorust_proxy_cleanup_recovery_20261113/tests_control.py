"""Model-free regressions for saved-evidence ownership and deadline preservation."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from . import common as c,control,owner


def test_only_own_cpu6_affinity_is_changed(monkeypatch):
    seen=[];affinity=[]
    class Process:
        def __init__(self,*args):assert not args;seen.append('self')
        def cpu_affinity(self,value=None):
            if value is not None:affinity[:]=value
            return affinity[:]
        def nice(self,value):seen.append(value)
    monkeypatch.setattr(control.psutil,'Process',Process)
    control.set_own_cpu()
    assert affinity==[6] and seen[0]=='self'


def test_actual_limits_and_no_ram_forecast_admission():
    assert c.design()['available_ram_gate'] is False and c.design()['forecast_gate'] is False
    assert tuple(c.JOBS)==('analysis','acceptance') and c.MODEL_JOBS==()
    env=control.environment()
    assert env['CUDA_VISIBLE_DEVICES']=='' and 'KMP_DUPLICATE_LIB_OK' not in env
    assert all(env[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'))


def stage_fixture(monkeypatch):
    monkeypatch.setattr(control.time,'monotonic',lambda:1000.)
    monkeypatch.setattr(control.time,'time',lambda:5000.)
    monkeypatch.setattr(c,'CAPS',dict(analysis=500.,acceptance=500.))
    original=dict(monotonic_origin=400.,unix_origin=4400.,monotonic_deadline=1200.,unix_deadline=5200.)
    monkeypatch.setattr(c,'read',lambda p:dict(limits=original))
    return dict(monotonic_origin=900.,unix_origin=4900.,monotonic_deadline=1400.,unix_deadline=5400.)


def test_analysis_retains_both_original_stage_deadlines(monkeypatch):
    result=control.stage_limits(stage_fixture(monkeypatch),'analysis')
    assert result['monotonic_deadline']==1200 and result['unix_deadline']==5200


def test_acceptance_is_clipped_to_recovery_parent(monkeypatch):
    result=control.stage_limits(stage_fixture(monkeypatch),'acceptance')
    assert result['monotonic_deadline']==1400 and result['unix_deadline']==5400


def test_model_or_fresh_decision_stage_is_rejected(monkeypatch):
    parent=stage_fixture(monkeypatch)
    with pytest.raises(ValueError,match='Only two saved-evidence'):control.stage_limits(parent,'reproduction')
    monkeypatch.setattr(c,'MODEL_JOBS',('analysis',))
    with pytest.raises(ValueError,match='Only two saved-evidence'):control.stage_limits(parent,'analysis')


def test_both_clocks_expire_independently(monkeypatch):
    limits=control.stage_limits(stage_fixture(monkeypatch),'analysis')
    for key in ('monotonic_deadline','unix_deadline'):
        changed=deepcopy(limits);changed[key]=0
        with pytest.raises(ValueError,match='dual clock'):c.check_deadlines(changed)


def test_discovery_budget_does_not_restart_at_launch(monkeypatch):
    original=dict(monotonic_origin=100.,unix_origin=1000.,monotonic_deadline=10000.,unix_deadline=10900.)
    monkeypatch.setattr(c,'read',lambda p:dict(limits=original))
    monkeypatch.setattr(c,'DISCOVERED_UNIX',2000.);monkeypatch.setattr(c,'TOTAL_SECONDS',7200.)
    first=c.recovery_limits()
    monkeypatch.setattr(control.time,'time',lambda:8000.)
    assert c.recovery_limits()==first and first['unix_deadline']==9200 and first['monotonic_deadline']==8300
    assert first['unix_origin']==1000 and first['monotonic_origin']==100


def test_original_global_clock_can_be_tighter_than_recovery(monkeypatch):
    original=dict(monotonic_origin=100.,unix_origin=1000.,monotonic_deadline=4000.,unix_deadline=4900.)
    monkeypatch.setattr(c,'read',lambda p:dict(limits=original))
    monkeypatch.setattr(c,'DISCOVERED_UNIX',2000.);monkeypatch.setattr(c,'TOTAL_SECONDS',7200.)
    assert c.recovery_limits()==original


def test_cleanup_does_not_target_reused_pid(monkeypatch):
    ident=dict(pid=77,created=1.);killed=[]
    process=SimpleNamespace(returncode=0,poll=lambda:0,wait=lambda timeout:0)
    monkeypatch.setattr(c,'exact_alive',lambda value:False)
    monkeypatch.setattr(control.psutil,'Process',lambda pid:SimpleNamespace(kill=lambda:killed.append(pid)))
    errors=[];control.cleanup_owned(process,None,ident,{(77,1.):ident},errors)
    assert not errors and not killed


def test_cleanup_waits_for_delayed_owned_absence(monkeypatch):
    ident=dict(pid=77,created=1.);state={'checks':0,'sleeps':0,'killed':[]}
    process=SimpleNamespace(returncode=0,poll=lambda:0,wait=lambda timeout:0)
    def alive(value):
        state['checks']+=1
        return state['checks']<4
    monkeypatch.setattr(c,'exact_alive',alive)
    monkeypatch.setattr(control.time,'sleep',lambda seconds:state.update(sleeps=state['sleeps']+1))
    monkeypatch.setattr(control.psutil,'Process',lambda pid:SimpleNamespace(kill=lambda:state['killed'].append(pid)))
    errors=[];control.cleanup_owned(process,None,ident,{(77,1.):ident},errors)
    assert state['sleeps']>0 and not state['killed'] and not errors


def test_owner_handshake_binds_exact_identity_and_freeze(monkeypatch,tmp_path):
    ident,parent=dict(pid=1,created=1.),dict(pid=2,created=2.)
    record=dict(identity=ident,parent=parent,windows_job_assigned=True,freeze_sha256='a'*64,role='analysis')
    (tmp_path/'owner.json').write_text(c.canonical(record))
    monkeypatch.setattr(c,'identity',lambda:ident);monkeypatch.setattr(c,'exact_alive',lambda v:v==parent)
    assert control.handshake(tmp_path,'a'*64,'analysis')==record
    with pytest.raises(ValueError,match='Exact frozen owned role'):control.handshake(tmp_path,'b'*64,'analysis')
    monkeypatch.setattr(c,'exact_alive',lambda v:False)
    with pytest.raises(ValueError,match='Exact live owner'):control.handshake(tmp_path,'a'*64,'analysis')


def owner_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'PLAN',tmp_path/'freeze.json')
    limits=dict(monotonic_origin=0.,unix_origin=0.,monotonic_deadline=10**20,unix_deadline=10**20)
    c.save(c.PLAN,dict(design=c.design(),worker_inputs=[],data_inputs=[]))
    monkeypatch.setattr(control,'set_own_cpu',lambda:None)
    monkeypatch.setattr(control,'handshake',lambda *args:dict(limits=limits))
    monkeypatch.setattr(control,'bounds',lambda *args:None)
    monkeypatch.setattr(control,'stage_limits',lambda *args:limits)
    monkeypatch.setattr(c,'recovery_limits',lambda:limits)
    return c.sha(c.PLAN)


def test_failed_analysis_never_launches_acceptance(monkeypatch,tmp_path):
    expected=owner_fixture(monkeypatch,tmp_path);launched=[]
    def fail(command,folder,limits,metadata,rss):
        launched.append(metadata['role']);raise ValueError('saved evidence rejected')
    monkeypatch.setattr(control,'launch',fail)
    assert owner.run(expected)==1 and launched==['analysis']
    assert c.read(tmp_path/'failure.json')['status']=='incomplete_recovery'
    assert not(tmp_path/'completion.json').exists()


def test_outer_success_cannot_accept_unbound_result(monkeypatch,tmp_path):
    expected=owner_fixture(monkeypatch,tmp_path);launched=[]
    def unbound(command,folder,limits,metadata,rss):
        role=metadata['role'];launched.append(role)
        c.save(folder/'worker/result.json',dict(status='passed',role=role,error=None,freeze_sha256=expected,identity={'pid':99,'created':1.}))
        return c.save(folder/'actual_exit.json',dict(status='passed',exit_code=0,result_artifact={'wrong':'binding'},exact_owned_absence=True,cleanup_errors=[],identity={'pid':99,'created':1.}))
    monkeypatch.setattr(control,'launch',unbound)
    assert owner.run(expected)==1 and launched==['analysis']
    assert 'Exact clean saved-evidence worker exit' in c.read(tmp_path/'failure.json')['error']


@pytest.mark.parametrize('name',['stage.tmp','progress.tmp','owner.json.pending'])
def test_output_scan_accepts_only_declared_atomic_rename_races(monkeypatch,name):
    def vanished():raise FileNotFoundError(name)
    path=SimpleNamespace(name=name,is_file=lambda:True,stat=vanished)
    monkeypatch.setattr(c,'OUT',SimpleNamespace(rglob=lambda pattern:[path]))
    assert control.output_bytes()==0


def test_output_scan_rejects_vanishing_immutable_evidence(monkeypatch):
    def vanished():raise FileNotFoundError('result.json')
    path=SimpleNamespace(name='result.json',is_file=lambda:True,stat=vanished)
    monkeypatch.setattr(c,'OUT',SimpleNamespace(rglob=lambda pattern:[path]))
    with pytest.raises(ValueError,match='declared transient'):control.output_bytes()
