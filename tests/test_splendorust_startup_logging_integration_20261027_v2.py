"""Forensic-finalization control flow; unrelated runtime gates are explicit stubs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tools.splendorust_startup_logging_20261027_02 import failure_audit as f
from tools.splendorust_startup_logging_20261027_02 import common as c,runner,qualification,runtime_support as support


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True),encoding='utf-8')


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No subprocess or model execution in saved-evidence tests')
    monkeypatch.setattr(subprocess,'Popen',forbidden)


@pytest.fixture(scope='module')
def actual_failure():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    verify_runtime_sources()
    return f.verify_failed_startup(f.PRIOR/'workers/qualification_frozen_cpu2',read(f.PRIOR/'freeze.json'))


def test_close_dispatches_to_explicit_new_source_forensics_without_owner_exit_requirement(tmp_path,monkeypatch):
    folder=tmp_path/'attempt/workers/qualification_frozen_cpu2'
    shutil.copytree(f.PRIOR/'workers/qualification_frozen_cpu2',folder)
    shutil.copyfile(f.PRIOR/'launch.json',folder.parent.parent/'launch.json')
    frozen=read(f.PRIOR/'freeze.json');prefix='tools.splendorust_startup_logging_20261027_02'
    for name in ('runner','placement','common','worker','logging_probe'):
        relative=prefix.replace('.','/')+'/'+name+'.py';path=f.ROOT/relative
        frozen['inputs'].append(dict(path=relative,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    identity=read(folder/'route_identity.json');identity['module']=prefix+'.worker';write(folder/'route_identity.json',identity)
    # The old failure predates logging-probe instrumentation. Only this unrelated
    # component is stubbed; the complete new-source failed-startup audit is real.
    monkeypatch.setattr(support,'verify_worker_probes',lambda folder,freeze:dict(status='test_only_probe_stub'))
    proof=support.close_proof(folder,frozen,SimpleNamespace(guard=lambda:None))
    assert proof['source_module_prefix']==prefix and proof['owner_exits'] is None
    assert proof['partial_science']['complete_decisions']==27 and proof['qualification_passed'] is False


@pytest.fixture
def finalization_fixture(tmp_path,monkeypatch,actual_failure):
    out=tmp_path/'attempt';out.mkdir()
    monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    proofs={};clocks={}
    for arm in ('frozen_cpu0','frozen_cpu2'):
        folder=out/'workers'/('qualification_'+arm);folder.mkdir(parents=True)
        clock=read(f.PRIOR/'workers'/folder.name/'entrypoint_clock.json');clocks[arm]=clock
        write(folder/'entrypoint_clock.json',clock);write(folder/(clock['request_id']+'.json'),clock)
        proof=copy.deepcopy(actual_failure) if arm=='frozen_cpu2' else dict(status='passed',worker=folder.name)
        proofs[folder.name]=proof;write(out/'closed_workers'/(folder.name+'.json'),proof)
        if arm=='frozen_cpu0':write(out/'entrypoint_checks'/(arm+'.json'),clock)
    calls=[]
    monkeypatch.setattr(support,'verify_replay',lambda:dict(passed=True))
    monkeypatch.setattr(support,'close_proof',lambda folder,freeze,envelope:copy.deepcopy(proofs[folder.name]))
    monkeypatch.setattr(qualification,'verify_entrypoint',lambda *args,**kwargs:calls.append('successful_suite'))
    monkeypatch.setattr(runner,'verify_runtime',lambda *args:calls.append('full_runtime'))
    def preservation(*args):
        path=out/'prior_preservation_after.json';c.save(path,dict(status='test_only_runtime_stub'));return c.artifact(path)
    monkeypatch.setattr(runner,'verify_prior_preservation',preservation)
    monkeypatch.setattr(runner,'verify_exchanges',lambda *args:dict(status='test_only_placement_stub'))
    return SimpleNamespace(out=out,proofs=proofs,clocks=clocks,calls=calls,envelope=SimpleNamespace(guard=lambda:None))


def finalize(value):
    return support.failed_startup_finalization(value.envelope,{},None,None,{})


def test_failed_finalization_runs_remaining_checks_and_creates_only_failure_seal(finalization_fixture):
    value=finalization_fixture;result=finalize(value)
    assert result['complete'] is False and result['failed_startup_verified'] is True
    assert result['startup_qualification_passed'] is False and result['attempt_accepted'] is False
    assert result['ordinary_requests']==0 and result['failed_workers'][0]['partial_science']['complete_decisions']==27
    assert value.calls==['successful_suite','full_runtime']
    seal=read(value.out/'failure_evidence_seal.json')
    assert seal['attempt_accepted'] is False and seal['status']=='failure_preservation_only'
    assert not (value.out/'completed_evidence_seal.json').exists()
    assert not (value.out/'completion.json').exists()  # Caller still owns failed/nonzero disposition.


@pytest.mark.parametrize('mutation',['missing_closed','false_qualification','wrong_top_clock','missing_successful_top',
                                      'extra_top','missing_worker_prefix','ordinary_after_failure'])
def test_failure_finalization_rejects_missing_or_promoted_evidence(finalization_fixture,mutation):
    v=finalization_fixture
    if mutation=='missing_closed':(v.out/'closed_workers/qualification_frozen_cpu2.json').unlink()
    elif mutation=='false_qualification':
        v.proofs['qualification_frozen_cpu2']['qualification_passed']=True
        write(v.out/'closed_workers/qualification_frozen_cpu2.json',v.proofs['qualification_frozen_cpu2'])
    elif mutation=='wrong_top_clock':
        clock=copy.deepcopy(v.clocks['frozen_cpu0']);clock['deadline_seconds']=1.
        write(v.out/'entrypoint_checks/frozen_cpu0.json',clock)
    elif mutation=='missing_successful_top':(v.out/'entrypoint_checks/frozen_cpu0.json').unlink()
    elif mutation=='extra_top':write(v.out/'entrypoint_checks/research_cpu0.json',v.clocks['frozen_cpu0'])
    elif mutation=='missing_worker_prefix':
        (v.out/'workers/qualification_frozen_cpu0').rename(v.out/'workers/not_a_planned_worker')
    elif mutation=='ordinary_after_failure':write(v.out/'records/0000.json',dict(completed=True))
    with pytest.raises((ValueError,FileNotFoundError)):
        finalize(v)
    assert not (v.out/'failure_evidence_seal.json').exists()
    assert not (v.out/'completed_evidence_seal.json').exists()


def test_failed_arm_optional_top_level_receipt_must_match_original(finalization_fixture):
    v=finalization_fixture;clock=copy.deepcopy(v.clocks['frozen_cpu2']);clock['detail']='forged failure'
    write(v.out/'entrypoint_checks/frozen_cpu2.json',clock)
    with pytest.raises(ValueError,match='Top-level original startup clock binding'):
        finalize(v)
