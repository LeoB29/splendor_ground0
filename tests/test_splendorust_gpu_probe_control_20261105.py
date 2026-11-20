"""Model-free supervisor identity and native-fallback checks."""
from types import SimpleNamespace
import pytest
from tools.splendorust_gpu_probe_20261105 import control

def test_native_fallback_is_detected(tmp_path):
    path=tmp_path/'stderr.log'
    path.write_text('normal initialization\nOperator aten::example is not supported on DirectML and will fall back to CPU\n')
    assert len(control.native_fallback_messages(path))==1

def test_native_normal_log_is_not_fallback(tmp_path):
    path=tmp_path/'stderr.log';path.write_text('normal initialization\n')
    assert control.native_fallback_messages(path)==[]

def test_fallback_cleanup_refuses_pid_reuse(monkeypatch):
    proc=SimpleNamespace(pid=7,poll=lambda:None)
    monkeypatch.setattr(control.psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:99))
    with pytest.raises(ValueError,match='Exact owned cleanup'):
        control.finish_owned_process(proc,{'pid':7,'created':98})

def test_fallback_cleanup_kills_only_owned_tree(monkeypatch):
    calls=[]
    child=SimpleNamespace(kill=lambda:calls.append('child'))
    actual=SimpleNamespace(create_time=lambda:98,children=lambda recursive:[child],kill=lambda:calls.append('parent'))
    proc=SimpleNamespace(pid=7,poll=lambda:None,wait=lambda timeout:calls.append('wait'))
    monkeypatch.setattr(control.psutil,'Process',lambda pid:actual)
    monkeypatch.setattr(control,'require_absent',lambda ident:calls.append('absent'))
    control.finish_owned_process(proc,{'pid':7,'created':98})
    assert calls==['child','parent','wait','absent']
