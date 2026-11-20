import copy,time
from pathlib import Path
import pytest
from tools.splendorust_training4096_20261120 import common as c,control,worker,placement

@pytest.fixture
def clocks(monkeypatch):
    limits=dict(monotonic_origin=10.,unix_origin=1000.,monotonic_deadline=10.+c.TOTAL_SECONDS,unix_deadline=1000.+c.TOTAL_SECONDS)
    launcher=dict(identity={'pid':1,'created':1.},limits=limits,freeze={'frozen':True})
    root=dict(identity={'pid':2,'created':2.},limits=limits)
    monkeypatch.setattr(c,'read',lambda path:launcher if Path(path).name=='launcher_claim.json' else root)
    monkeypatch.setattr(c,'artifact',lambda path:{'frozen':True})
    owner=dict(parent=root['identity'],limits=dict(monotonic_origin=20.,unix_origin=1010.,monotonic_deadline=30.,unix_deadline=1020.),rss_cap=c.MODEL_RSS)
    return owner,launcher,root

def test_fixed_original_clocks(clocks):control.validate_limits(clocks[0],'qualification')

@pytest.mark.parametrize('field,value',[('monotonic_deadline',200000.),('unix_deadline',200000.),('monotonic_origin',0.),('unix_origin',0.)])
def test_extended_or_backdated_worker_clocks_rejected(clocks,field,value):
    owner=copy.deepcopy(clocks[0]);owner['limits'][field]=value
    with pytest.raises(ValueError):control.validate_limits(owner,'qualification')

def test_model_cutoff_cannot_reset_late(clocks):
    owner=clocks[0]
    owner['limits'].update(monotonic_origin=10.+c.MODEL_SECONDS,monotonic_deadline=11.+c.MODEL_SECONDS,
        unix_origin=1000.+c.MODEL_SECONDS,unix_deadline=1001.+c.MODEL_SECONDS)
    with pytest.raises(ValueError,match='cutoff'):control.validate_limits(owner,'qualification')

def test_root_clock_is_original_launch(clocks):
    owner,launcher,_=clocks;owner.update(parent=launcher['identity'],limits=launcher['limits'],rss_cap=c.AUDIT_RSS+512*1024**2)
    control.validate_limits(owner,'root')

@pytest.mark.parametrize('mutation',['parent','rss','span'])
def test_original_owner_and_caps_required(clocks,mutation):
    owner,launcher,_=clocks
    if mutation=='parent':owner['parent']={'pid':99,'created':99.}
    if mutation=='rss':owner['rss_cap']+=1
    if mutation=='span':launcher['limits']['unix_deadline']+=1
    with pytest.raises(ValueError):control.validate_limits(owner,'qualification')

def test_clipped_stage_and_model_clocks(monkeypatch):
    monkeypatch.setattr(time,'monotonic',lambda:100.);monkeypatch.setattr(time,'time',lambda:1000.)
    parent=dict(monotonic_origin=0.,unix_origin=900.,monotonic_deadline=150.,unix_deadline=1050.)
    value=control.clipped_limits(parent,100.)
    assert value['monotonic_deadline']==150. and value['unix_deadline']==1050.

def test_both_clocks_enforced(monkeypatch):
    monkeypatch.setattr(time,'monotonic',lambda:10.);monkeypatch.setattr(time,'time',lambda:20.)
    c.check_deadlines(dict(monotonic_origin=0.,unix_origin=0.,monotonic_deadline=11.,unix_deadline=21.))
    for clock in ('unix','monotonic'):
        limits=dict(monotonic_origin=0.,unix_origin=0.,monotonic_deadline=11.,unix_deadline=21.)
        limits[clock+'_deadline']=1.
        with pytest.raises(ValueError):c.check_deadlines(limits)

def test_fixed_sequence_requires_all_previous_actual_exits(monkeypatch):
    seen=[]
    monkeypatch.setattr(control,'passed_job',lambda role,**kw:(seen.append(role) or role,{}))
    worker.dependencies('assessment_cache')
    assert seen==list(c.JOBS[:c.JOBS.index('assessment_cache')])
    assert seen[-1]=='commit' and all(role in seen for role in c.FIT_IDS)

def test_failed_predecessor_prevents_next_worker(monkeypatch):
    def fail(role,**kw):raise ValueError('preserved failure')
    monkeypatch.setattr(control,'passed_job',fail)
    with pytest.raises(ValueError):worker.dependencies('training_cache')

def test_no_ram_or_forecast_gate_and_single_thread_environment():
    env=control.environment()
    assert 'KMP_DUPLICATE_LIB_OK' not in env
    assert env['CUDA_VISIBLE_DEVICES']==''
    assert all(env[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS'))

def test_native_ownership_has_no_parent_pid_traversal():
    text=Path(control.__file__).read_text()
    assert '.children(' not in text and 'observe_job' in text and 'finish_exact_process' in text
