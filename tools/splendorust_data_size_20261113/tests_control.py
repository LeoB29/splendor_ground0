"""Model-free ownership, separate-core placement and original-clock regressions."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from . import common as c,control,placement


def topology():
    return dict(api='GetLogicalProcessorInformationEx(RelationProcessorCore)',
        cores=[dict(core_index=i,efficiency_class=1,logical_processors=[2*i,2*i+1]) for i in range(6)]+
              [dict(core_index=6,efficiency_class=0,logical_processors=[12])])


def test_physical_core_separation_rejects_smt_siblings(monkeypatch):
    assert placement.validate_topology(topology())['physical_cores_distinct']
    monkeypatch.setattr(c,'TRAIN_CPU',1)
    with pytest.raises(ValueError,match='physically distinct'):placement.validate_topology(topology())


def test_training_requires_same_performance_core_class(monkeypatch):
    monkeypatch.setattr(c,'TRAIN_CPU',12)
    with pytest.raises(ValueError,match='performance-core'):placement.validate_topology(topology())


def test_placement_only_modifies_own_process_and_rejects_active_cores(monkeypatch):
    import psutil
    seen=[];affinity=[]
    class Process:
        def __init__(self,*args):assert not args;seen.append('self')
        def cpu_affinity(self,value=None):
            if value is not None:affinity[:]=value
            return affinity[:]
        def nice(self,value):seen.append(value)
    monkeypatch.setattr(psutil,'Process',Process);monkeypatch.setattr(c,'identity',lambda:dict(pid=77,created=1.))
    for cpu in (0,1,2,3):
        with pytest.raises(ValueError,match='Only new-study'):placement.set_own(cpu)
    assert not seen
    assert placement.set_own(4)['affinity']==[4]


def test_both_original_clocks_and_model_cutoff(monkeypatch):
    monkeypatch.setattr(control.time,'monotonic',lambda:1000.);monkeypatch.setattr(control.time,'time',lambda:5000.)
    monkeypatch.setattr(c,'MODEL_SECONDS',200.)
    parent=dict(monotonic_origin=900.,unix_origin=4900.,monotonic_deadline=1800.,unix_deadline=5800.)
    limits=control.clipped_limits(parent,500,model=True)
    assert limits['monotonic_deadline']==1100 and limits['unix_deadline']==5100
    for key in ('monotonic_deadline','unix_deadline'):
        bad=deepcopy(limits);bad[key]=0
        with pytest.raises(ValueError,match='dual clock'):c.check_deadlines(bad)


def test_cleanup_never_targets_a_reused_pid(monkeypatch):
    ident=dict(pid=77,created=1.);killed=[]
    process=SimpleNamespace(returncode=0,poll=lambda:0,wait=lambda timeout:0)
    monkeypatch.setattr(c,'exact_alive',lambda value:False)
    monkeypatch.setattr(control.psutil,'Process',lambda pid:SimpleNamespace(kill=lambda:killed.append(pid)))
    errors=[];control.cleanup_owned(process,None,ident,{(77,1.):ident},errors)
    assert not errors and not killed


def test_active_preservation_verifies_only_frozen_inputs(monkeypatch):
    calls=[];plan=dict(worker_inputs=[{'source':'fixed'}],data_inputs=[{'data':'fixed'}])
    monkeypatch.setattr(c,'sha',lambda path:c.ACTIVE_PROXY_SHA)
    monkeypatch.setattr(c,'read',lambda path:plan)
    monkeypatch.setattr(c,'artifact',lambda path:{'path':str(path)})
    monkeypatch.setattr(c,'verify',lambda items,boundary:calls.extend(items))
    proof=placement.preserve_calibration()
    assert calls==[{'source':'fixed'},{'data':'fixed'}]
    assert not proof['processes_modified'] and not proof['scientific_results_inspected']


def test_handshake_binds_exact_process_freeze_and_live_owner(monkeypatch,tmp_path):
    ident,parent=dict(pid=1,created=1.),dict(pid=2,created=2.)
    owner=dict(identity=ident,parent=parent,windows_job_assigned=True,freeze_sha256='a'*64,role='g256_r0')
    (tmp_path/'owner.json').write_text(c.canonical(owner))
    monkeypatch.setattr(c,'identity',lambda:ident);monkeypatch.setattr(c,'exact_alive',lambda value:value==parent)
    assert control.handshake(tmp_path,'a'*64,'g256_r0')==owner
    with pytest.raises(ValueError,match='Exact frozen owned role'):control.handshake(tmp_path,'b'*64,'g256_r0')
    monkeypatch.setattr(c,'exact_alive',lambda value:False)
    with pytest.raises(ValueError,match='Exact live owner'):control.handshake(tmp_path,'a'*64,'g256_r0')


def test_no_available_ram_or_forecast_gate():
    assert c.design()['available_ram_gate'] is False and c.design()['forecast_gate'] is False
    env=control.environment()
    assert env['CUDA_VISIBLE_DEVICES']==''
    assert all(env[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'))
