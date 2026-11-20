"""Saved-failure and unchanged-clock counterexamples; no model construction."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from tools.splendorust_independent_data_disk_recovery_20261109_02 import common as c,owner,recovery,audit,preflight


def test_actual_saved_disk_boundary_has_no_missing_decision():
    before='torch' in sys.modules
    value=recovery.check_source_boundary()
    assert len(value['passed_executions'])==35
    assert c.read(c.resolve(value['boundary']))['next_unplayed_ply']==26
    assert ('torch' in sys.modules)==before is False


def test_original_scientific_contract_is_identical():
    from tools.splendorust_independent_data_20261107.owner import contract
    assert owner.contract()==contract()


def test_actual_failed_metadata_attempt_has_zero_ordinary_decisions():
    value=recovery.check_failed_predecision()
    assert value['ordinary_decisions']==value['games']==0
    assert len(value['passed_executions'])==2
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('mode,rep,offset',[('resume_partial',0,None),('resume_partial',1,159),
    ('prepare',0,0),('main_block',1,159),('resume_partial',True,None),('prepare',2,None)])
def test_wrong_continuation_dispatch_rejected(mode,rep,offset):
    with pytest.raises(ValueError):owner.segment_id(mode,rep,offset)


def test_distinct_fresh_modes_and_fixed_stage_inventory():
    assert owner.segment_id('prepare',0)=='prepare_r0'
    assert owner.segment_id('resume_partial',1)=='resume_partial_r1'
    paths=audit.expected_executions()
    assert len(paths)==len(set(paths))==21
    assert not any('qualification' in str(p) or 'operational' in str(p) for p in paths)
    assert [p.name for p in recovery.expected_source_executions()]==['execution.json']*35


@pytest.fixture
def inventory_case(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'new');c.OUT.mkdir()
    source=tmp_path/'closed';source.mkdir();(source/'one.txt').write_bytes(b'preserve original failure\n')
    item=c.artifact(source/'one.txt')
    inventory=c.OUT/'saved.jsonl';inventory.write_text(c.canonical(item)+'\n',encoding='utf-8')
    return source,inventory,item


def verify_case(case):
    source,inventory,item=case
    return audit.verify_inventory(source,c.artifact(inventory),1,item['bytes'],lambda:None)


def test_exact_preservation_reader(inventory_case):
    assert verify_case(inventory_case)['all_original_bytes_preserved'] is True


@pytest.mark.parametrize('mutation',['changed','missing','extra','duplicate','wrong_size','outside','traversal'])
def test_preservation_changes_rejected(inventory_case,mutation):
    source,inventory,item=inventory_case
    if mutation=='changed':(source/'one.txt').write_bytes(b'x'*item['bytes'])
    elif mutation=='missing':(source/'one.txt').unlink()
    elif mutation=='extra':(source/'extra.txt').write_bytes(b'no')
    elif mutation=='duplicate':inventory.write_text((c.canonical(item)+'\n')*2,encoding='utf-8')
    else:
        item=deepcopy(item)
        if mutation=='wrong_size':item['bytes']+=1
        elif mutation=='outside':item['path']='elsewhere/one.txt'
        else:item['path']='closed/../closed/one.txt'
        inventory.write_text(c.canonical(item)+'\n',encoding='utf-8')
    with pytest.raises((ValueError,FileNotFoundError)):verify_case(inventory_case)


@pytest.mark.parametrize('key,value',[('tests_passed',526),('tests_collected',528),('failures',1),
    ('skipped',1),('models',1),('games',1),('tensor_runtime_loaded',True),('failures',False)])
def test_incomplete_or_model_loading_preflight_rejected(key,value):
    summary=dict(tests_collected=610,tests_passed=610,failures=0,skipped=0,models=0,games=0,tensor_runtime_loaded=False)
    summary[key]=value
    with pytest.raises(ValueError):preflight.complete_summary(summary)


def test_complete_baseline_preflight_is_accepted():
    assert preflight.complete_summary(dict(tests_collected=610,tests_passed=610,failures=0,
        skipped=0,models=0,games=0,tensor_runtime_loaded=False)) is True


@pytest.fixture
def claim_case(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    for attr,name in [('OUT','new'),('SOURCE_OUT','source'),('LIVE_SOURCE_OUT','live'),('FAILED_OUT','failed')]:
        monkeypatch.setattr(c,attr,tmp_path/name);(tmp_path/name).mkdir()
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json')
    def write(p,v):p.write_text(c.canonical(v)+'\n',encoding='utf-8')
    old=dict(pid=101,created=1.,monotonic_origin=10.,unix_origin=20.,total_seconds=432000,model_seconds=345600)
    write(c.SOURCE_OUT/'claim.json',old);write(c.LIVE_SOURCE_OUT/'claim.json',{**old,'pid':102,'created':2.})
    write(c.FAILED_OUT/'claim.json',{**old,'pid':104,'created':4.})
    write(c.PLAN,{})
    reuse=dict(previous_claim=c.artifact(c.LIVE_SOURCE_OUT/'claim.json'),failed_predecision=dict(claim=c.artifact(c.FAILED_OUT/'claim.json')))
    write(c.OUT/'reuse.json',reuse);monkeypatch.setattr(recovery,'validate_reuse',lambda value,**kw:value)
    claim=dict(**{k:old[k] for k in ('monotonic_origin','unix_origin','total_seconds','model_seconds')},
        pid=103,created=3.,reuse=c.artifact(c.OUT/'reuse.json'),original_claim=c.artifact(c.SOURCE_OUT/'claim.json'),
        previous_claim=reuse['previous_claim'],freeze=c.artifact(c.PLAN),plan_sha256=c.sha(c.PLAN),automatic_retries=0)
    claim['failed_predecision_claim']=reuse['failed_predecision']['claim']
    return claim


def test_inherited_claim_preserved(claim_case):assert recovery.validate_recovery_claim(claim_case)==claim_case


@pytest.mark.parametrize('key,value',[('monotonic_origin',11.),('unix_origin',21.),('total_seconds',432001),
    ('model_seconds',345601),('total_seconds',432000.),('automatic_retries',1)])
def test_clock_reset_extension_or_retry_rejected(claim_case,key,value):
    claim_case[key]=value
    with pytest.raises(ValueError):recovery.validate_recovery_claim(claim_case)


@pytest.mark.parametrize('pid,created',[(101,1.),(102,2.),(104,4.)])
def test_dead_owner_identity_not_reused(claim_case,pid,created):
    claim_case.update(pid=pid,created=created)
    with pytest.raises(ValueError):recovery.validate_recovery_claim(claim_case)


def test_expired_original_phase_rejected_before_save(claim_case,monkeypatch):
    old=dict(stage='strength',claim=c.artifact(c.SOURCE_OUT/'claim.json'),monotonic=10.,unix_time=20.,
        monotonic_deadline=50.,unix_deadline=60.,cap_seconds=144000)
    p=c.SOURCE_OUT/'stages';p.mkdir();(p/'strength_start.json').write_text(c.canonical(old),encoding='utf-8')
    (c.OUT/'claim.json').write_text(c.canonical(claim_case),encoding='utf-8')
    monkeypatch.setattr(owner.time,'monotonic',lambda:51.)
    monkeypatch.setattr(owner.time,'time',lambda:61.)
    with pytest.raises(ValueError):recovery.recover_phase(claim_case)
    assert not (c.OUT/'stages/strength_start.json').exists()
