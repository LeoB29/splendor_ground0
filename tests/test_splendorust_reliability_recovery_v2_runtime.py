"""Model-free contracts for the saved-only runtime/prior recovery component."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import pytest
from tools.splendorust_reliability_recovery_v2_20261026 import runtime as r


class Envelope:
    placement_mode='isolated'
    def __init__(self):self.guards=0;self.checkpoints=[]
    def guard(self):self.guards+=1
    def checkpoint(self,phase,details=None):self.checkpoints.append((phase,details));self.guard()


def setup_pipeline(monkeypatch,*,runtime_fault=None,prior_fault=None,changed_acceptance=False,omitted_input=False):
    envelope=Envelope();events=[]
    freeze=dict(inputs=[],native_artifacts=[{'path':'native'}],torch_runtime=[{}],external_runtime=[{},{}],
        qualification_acceptance={'status':'expected'},placement_mode='isolated')
    artifact={'path':'original/freeze.json','bytes':1,'sha256':'pinned'}
    monkeypatch.setattr(r,'_original_inventory',lambda f,e:({},artifact))
    monkeypatch.setattr(r,'_prebind_python_sources',lambda i,e:[{'path':'frozen.py'}])
    monkeypatch.setattr(r,'_bind_validators',lambda *a:dict(status='passed',sources=[{'path':'frozen.py'}],functions=[]))
    monkeypatch.setattr(r.c,'artifact',lambda *a,**kw:artifact)
    def original_runtime(value,boundary):
        events.append(('runtime',value is freeze,boundary.placement_mode))
        for i in range(r.ORIGINAL_INPUTS+len(freeze['external_runtime'])-(1 if omitted_input else 0)):
            boundary.guard()
        if runtime_fault:raise ValueError(runtime_fault)
    def original_acceptance(*,boundary):
        events.append(('prior',));boundary()
        if prior_fault:raise ValueError(prior_fault)
        return {'status':'changed' if changed_acceptance else 'expected'}
    validators=(SimpleNamespace(verify_runtime=original_runtime),SimpleNamespace(acceptance=original_acceptance),None,None)
    monkeypatch.setattr(r,'_load_validators',lambda:validators)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('Runtime recovery must not spawn processes'))
    monkeypatch.setattr(os,'system',lambda *a,**k:pytest.fail('Runtime recovery must not use external commands'))
    monkeypatch.setattr(r.c,'save',lambda *a,**k:pytest.fail('Runtime helper writes only through the new Envelope checkpoint'))
    return freeze,envelope,events


def test_full_source_restart_then_exact_prior_acceptance_without_process_or_model_calls(monkeypatch):
    freeze,envelope,events=setup_pipeline(monkeypatch)
    result=r.verify_runtime_and_prior(freeze,envelope)
    assert events==[('runtime',True,'isolated'),('prior',)]
    assert result['status']=='passed_runtime_and_prior_recovery'
    assert result['original_inputs_rechecked_from_start']==9330 and not result['previous_partial_loop_reused']
    assert result['preceding_frozen_inputs']==4885 and result['preceding_sealed_artifacts']==4392
    assert result['model_calls']==result['workers_launched']==result['fresh_decisions']==result['fresh_games']==0
    assert not result['original_outputs_written'] and result['original_incomplete_status_unchanged']
    progress=[details for name,details in envelope.checkpoints if name=='original_runtime_input_progress']
    assert [x['inputs_verified_before_next_boundary'] for x in progress]==list(range(1024,9330,1024))
    assert all(not x['complete_runtime_check_accepted'] for x in progress)
    assert events[0][0]=='runtime' and events[-1][0]=='prior'


@pytest.mark.parametrize('kind',['runtime','prior','changed_acceptance','omitted_input'])
def test_failed_runtime_or_prior_checks_cannot_produce_recovered_pass(monkeypatch,kind):
    args={'runtime_fault':'original DLL changed'} if kind=='runtime' else {'prior_fault':'prior sealed evidence changed'} if kind=='prior' else {kind:True}
    freeze,envelope,events=setup_pipeline(monkeypatch,**args)
    with pytest.raises(ValueError):r.verify_runtime_and_prior(freeze,envelope)
    if kind in ('runtime','omitted_input'):assert events==[('runtime',True,'isolated')]
    assert not any(name=='preceding_qualification_acceptance_passed' for name,_ in envelope.checkpoints)


def test_guard_failure_stops_before_further_inputs_or_prior_acceptance(monkeypatch):
    freeze,envelope,events=setup_pipeline(monkeypatch)
    def limited_guard():
        envelope.guards+=1
        if envelope.guards>50:raise ValueError('recovery real-clock limit')
    envelope.guard=limited_guard
    with pytest.raises(ValueError,match='real-clock'):r.verify_runtime_and_prior(freeze,envelope)
    assert events==[('runtime',True,'isolated')]


def test_newly_imported_frozen_helper_is_bound_without_rejecting_valid_module_growth(monkeypatch):
    freeze,envelope,_=setup_pipeline(monkeypatch)
    bindings=iter([dict(status='passed',functions=[],sources=[{'path':'frozen.py'}]),
        dict(status='passed',functions=[],sources=[{'path':'frozen.py'},{'path':'late_helper.py'}])])
    monkeypatch.setattr(r,'_bind_validators',lambda *a:next(bindings))
    result=r.verify_runtime_and_prior(freeze,envelope)
    assert result['imported_validator_bindings']['sources'][-1]['path']=='late_helper.py'


def test_rebound_function_or_missing_source_after_checks_is_rejected(monkeypatch):
    freeze,envelope,_=setup_pipeline(monkeypatch)
    bindings=iter([dict(status='passed',functions=[],sources=[{'path':'frozen.py'}]),
        dict(status='passed',functions=[{'function':'replacement'}],sources=[{'path':'frozen.py'}])])
    monkeypatch.setattr(r,'_bind_validators',lambda *a:next(bindings))
    with pytest.raises(ValueError,match='ownership/source changed'):r.verify_runtime_and_prior(freeze,envelope)


def test_model_runtime_already_loaded_is_rejected_without_loading_original_validators(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    monkeypatch.setattr(r,'_load_validators',lambda:pytest.fail('Import after model-free precondition failure'))
    with pytest.raises(ValueError,match='model runtime'):r.verify_runtime_and_prior({},Envelope())


def toy_source(tmp_path,monkeypatch):
    monkeypatch.setattr(r.c,'ROOT',tmp_path)
    path=tmp_path/'frozen.py';source='def checked(value):\n    return value + 1\n'
    path.write_text(source,encoding='utf8')
    module=ModuleType('frozen_test_module');module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    index={'frozen.py':r.c.artifact(path)}
    return module,index,path


def test_actual_callable_and_code_bound_to_frozen_source_without_executing_module_again(monkeypatch,tmp_path):
    module,index,path=toy_source(tmp_path,monkeypatch)
    compiled={};sources={}
    result=r._bind_function(module.checked,module,index,compiled,sources,Envelope())
    assert result['function']=='checked' and sources==index and list(compiled)==['frozen.py']


@pytest.mark.parametrize('kind',['replacement_code','changed_file','wrong_owner','missing_freeze_binding'])
def test_validator_substitution_or_source_drift_is_rejected(monkeypatch,tmp_path,kind):
    module,index,path=toy_source(tmp_path,monkeypatch)
    if kind=='replacement_code':exec(compile('def checked(value):\n    return value + 2\n',str(path),'exec'),module.__dict__)
    elif kind=='changed_file':path.write_text('def checked(value):\n    return 0\n')
    elif kind=='wrong_owner':module.checked.__module__='different_owner'
    else:index={}
    with pytest.raises(ValueError):r._bind_function(module.checked,module,index,{}, {},Envelope())


def test_proxy_has_no_old_resource_envelope_and_reports_only_completed_preceding_inputs():
    envelope=Envelope();proxy=r._RuntimeBoundary(envelope)
    assert proxy.placement_mode=='isolated' and not hasattr(proxy,'resident')
    for _ in range(1025):proxy.guard()
    assert envelope.checkpoints==[('original_runtime_input_progress',dict(
        inputs_verified_before_next_boundary=1024,total=9330,complete_runtime_check_accepted=False))]
