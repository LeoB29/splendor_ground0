from pathlib import Path
import pytest
from tools.splendorust_training_duration_20261113 import common as c, control

def test_scope():
    d=c.design()
    assert d['passes']==[1,2,4] and d['seeds']==[0,1]
    assert d['new_games']==0 and not d['automatic_followup'] and not d['automatic_promotion']
    assert d['maximum_new_updates']==10692+6+6

def test_actual_bounds_no_forecast_gates():
    d=c.design()
    assert not d['available_ram_gate'] and not d['forecast_gate']
    assert c.MODEL_SECONDS<c.TOTAL_SECONDS and c.MIN_DISK==5*1024**3

@pytest.mark.parametrize('name',c.JOBS)
def test_fixed_jobs_have_caps(name):
    assert 0<c.CAPS[name]<=c.TOTAL_SECONDS

def test_json_reject_duplicate(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):c.read(p)

def test_json_reject_nan(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"a":NaN}')
    with pytest.raises(ValueError):c.read(p)

def test_writer_boundary(tmp_path):
    with pytest.raises(ValueError):c.save(tmp_path/'bad.json',{})

def test_writer_exclusive(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    c.save(tmp_path/'a.json',{'a':1})
    with pytest.raises(FileExistsError):c.save(tmp_path/'a.json',{'a':2})

def test_mutable_allowlist(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    with pytest.raises(ValueError):c.status(tmp_path/'checkpoint.json',{})

def test_hash_change_rejected(tmp_path):
    p=tmp_path/'a';p.write_text('first');item=c.artifact(p);p.write_text('second')
    with pytest.raises(ValueError):c.verify([item])

@pytest.mark.parametrize('clock',('mono','wall'))
def test_either_clock_expiry_fails(monkeypatch,clock):
    monkeypatch.setattr(c.time,'monotonic',lambda:11 if clock=='mono' else 1)
    monkeypatch.setattr(c.time,'time',lambda:11 if clock=='wall' else 1)
    with pytest.raises(ValueError):c.check_deadlines(dict(monotonic_origin=0,unix_origin=0,monotonic_deadline=10,unix_deadline=10))

def test_clipped_model_global_clock(monkeypatch):
    monkeypatch.setattr(control.time,'monotonic',lambda:100.)
    monkeypatch.setattr(control.time,'time',lambda:200.)
    parent=dict(monotonic_origin=0,unix_origin=100,monotonic_deadline=50000,unix_deadline=50100)
    limits=control.clipped_limits(parent,40000,True)
    assert limits['monotonic_deadline']==c.MODEL_SECONDS
    assert limits['unix_deadline']==100+c.MODEL_SECONDS

def test_environment_serial_cpu():
    env=control.environment()
    assert env['OMP_NUM_THREADS']=='1' and env['CUDA_VISIBLE_DEVICES']==''

def test_missing_process_absent():
    assert not c.exact_alive({'pid':2147483647,'created':0})

def test_origin_identity_matches_accepted():
    from tools.splendorust_full_corpus_20261110.sampling import IDENTITY
    assert c.IDENTITY==IDENTITY
