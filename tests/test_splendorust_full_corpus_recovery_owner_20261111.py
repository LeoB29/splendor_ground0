"""Counterexamples for fixed samples, original clocks and immutable preservation."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tools.splendorust_full_corpus_recovery_20261111 import common as c, recovery as r, owner
from tools.splendorust_full_corpus_20261110 import schedule as original_schedule
from tools.splendorust_full_corpus_recovery_20261111 import schedule


def test_original_scientific_schedule_is_exact():
    assert schedule.strength_schedule()==original_schedule.strength_schedule()
    assert schedule.auxiliary_seeds()==original_schedule.auxiliary_seeds()


def test_fixed_remaining_blocks_and_partial_accounting():
    blocks=r.remaining_blocks()
    assert len(blocks)==98 and len(set(blocks))==98
    assert blocks[:2]==[('mixed_terminal',0,30),('mixed_terminal',1,30)]
    assert blocks[-1]==('mixed_terminal',1,190)
    assert ('terminal_parent',1,30) not in blocks
    assert 212+8+10*len(blocks)==1200


@pytest.mark.parametrize('matchup,rep,minimum',[
    ('mixed_parent',0,40),('mixed_parent',1,40),('terminal_parent',0,40),
    ('terminal_parent',1,40),('mixed_terminal',0,30),('mixed_terminal',1,30)])
def test_no_completed_block_or_interrupted_block_retried(matchup,rep,minimum):
    assert [b for m,s,b in r.remaining_blocks() if (m,s)==(matchup,rep)]==list(range(minimum,200,10))


def test_expected_live_roles_include_qualification_and_one_amendment():
    names=r.expected_live_segments()
    assert len(names)==111 and len(set(names))==111
    assert sum(n.startswith('prepare_') for n in names)==6
    assert sum(n.startswith('qualification_') for n in names)==6
    assert [n for n in names if n.startswith('resume_partial_')]==['resume_partial_terminal_parent_r1']


@pytest.fixture
def claim_context(monkeypatch,tmp_path):
    old={'pid':100,'created':1.5,'monotonic_origin':10.,'unix_origin':1000.,
         'total_seconds':432000,'model_seconds':345600}
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'new')
    monkeypatch.setattr(c,'ORIGINAL_OUT',tmp_path/'old')
    monkeypatch.setattr(c,'PLAN',tmp_path/'new/freeze.json')
    monkeypatch.setattr(c,'AUTHORIZATION',tmp_path/'new/authorization.json')
    def artifact(p): return {'path':str(p),'bytes':3,'sha256':'a'*64}
    monkeypatch.setattr(c,'artifact',artifact)
    monkeypatch.setattr(c,'sha',lambda p:'a'*64)
    monkeypatch.setattr(c,'read',lambda p:deepcopy(old))
    new=dict(old,pid=101,created=2.,original_claim=artifact(c.ORIGINAL_OUT/'claim.json'),
        reuse=artifact(c.OUT/'reuse.json'),authorization=artifact(c.AUTHORIZATION),freeze=artifact(c.PLAN),
        plan_sha256='a'*64,automatic_retries=0,authorized_retry_limit=1,retry_spec=deepcopy(c.RETRY_SPEC),
        amended_recovered_study=True)
    return old,new


def test_inherits_both_original_origins(claim_context):
    old,new=claim_context
    assert r.validate_recovery_claim(new)==new


@pytest.mark.parametrize('field',['monotonic_origin','unix_origin','total_seconds','model_seconds'])
def test_independent_clock_extension_rejected(claim_context,field):
    _,new=claim_context;new[field]+=1
    with pytest.raises(ValueError):r.validate_recovery_claim(new)


@pytest.mark.parametrize('field',['original_claim','reuse','authorization','freeze','plan_sha256'])
def test_changed_lineage_rejected(claim_context,field):
    _,new=claim_context;new[field]='wrong'
    with pytest.raises(ValueError):r.validate_recovery_claim(new)


@pytest.mark.parametrize('field,value',[
    ('automatic_retries',1),('automatic_retries',False),('authorized_retry_limit',2),
    ('authorized_retry_limit',True),('amended_recovered_study',False),('retry_spec',{})])
def test_extra_or_undisclosed_retry_rejected(claim_context,field,value):
    _,new=claim_context;new[field]=value
    with pytest.raises(ValueError):r.validate_recovery_claim(new)


def test_original_owner_identity_cannot_claim_recovery(claim_context):
    old,new=claim_context;new.update(pid=old['pid'],created=old['created'])
    with pytest.raises(ValueError):r.validate_recovery_claim(new)


def test_pid_reuse_with_new_creation_is_distinct(claim_context):
    old,new=claim_context;new['pid']=old['pid']
    assert r.validate_recovery_claim(new)==new


@pytest.fixture
def inventory(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'new');c.OUT.mkdir()
    monkeypatch.setattr(c,'status',lambda *a:None)
    old=tmp_path/'old';old.mkdir();(old/'a.json').write_text('original')
    descriptor=r.inventory_tree(old,c.OUT/'inventory.jsonl')
    return old,descriptor


def test_saved_inventory_accepts_unchanged_source(inventory):
    _,d=inventory
    assert r.verify_inventory(d)['status']=='preserved'


@pytest.mark.parametrize('operation',['edit','add','delete'])
def test_source_edit_add_or_delete_is_detected(inventory,operation):
    old,d=inventory
    if operation=='edit':(old/'a.json').write_text('modified')
    if operation=='add':(old/'b.json').write_text('extra')
    if operation=='delete':(old/'a.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):r.verify_inventory(d)


def test_inventory_cannot_write_into_preserved_source(inventory):
    old,_=inventory
    with pytest.raises(ValueError):r.inventory_tree(old,old/'inventory.jsonl')


def test_original_player_limits_and_no_fresh_training():
    value=owner.contract()
    assert (value['matched_simulations'],value['worlds'],value['cpuct'],value['depth'])==(128,3,.4,64)
    assert (value['internal_seconds'],value['complete_response_seconds'])==(9.,10.)
    assert value['new_optimizer_calls']==value['new_offline_model_evaluations']==0
    assert value['reused_fits']==4 and value['original_attempt_status']=='incomplete'
