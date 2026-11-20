"""Model-free regressions for accepted evidence reuse and prospective admission."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.splendorust_policy_head_20261104 import common as c, provenance as p, preparation as prep, design


@pytest.fixture
def saved_reuse(monkeypatch):
    proof=c.read(p.PROOF)
    freeze=dict(source_reuse_proof=c.artifact(p.PROOF), source_reuse=deepcopy(proof['declaration']),
                reused_corpus=proof['declaration']['corpus_manifest'], source_specs=proof['declaration']['source_specs'])
    original=c.read
    checked=[]
    monkeypatch.setattr(c,'verify',lambda entries:checked.extend(entries))
    monkeypatch.setattr(c,'read',lambda path:deepcopy(proof) if path==p.PROOF else original(path))
    return freeze,proof,checked


def test_exact_saved_reuse_preserves_original_specifications(saved_reuse):
    freeze,proof,checked=saved_reuse
    ticks=[]
    got=p.verify(freeze,lambda:ticks.append(1))
    assert set(got)=={'manifest','source_specs','proof'}
    assert got['source_specs']==design.schedule()
    assert got['manifest']['specs']==got['source_specs']
    assert len(checked)==len(ticks)==len(proof['inputs'])+7
    assert got['proof']['source_games_generated']==0


@pytest.mark.parametrize('field,value',[
    ('source_games_generated',1),('source_games_reused',255),
    ('previous_candidate_not_used',False),('status','incomplete'),('schema','old')])
def test_rejects_changed_reuse_contract(saved_reuse,field,value):
    freeze,proof,_=saved_reuse;proof[field]=value
    with pytest.raises(ValueError):p.verify(freeze)


@pytest.mark.parametrize('field',['reused_corpus','source_specs'])
def test_rejects_substituted_source_descriptor(saved_reuse,field):
    freeze,_,_=saved_reuse;freeze[field]={**freeze[field],'path':'substituted.json'}
    with pytest.raises(ValueError):p.verify(freeze)


def test_rejects_unaccepted_source_identity(saved_reuse):
    freeze,proof,_=saved_reuse
    proof['declaration']['accepted_source_freeze']['sha256']='0'*64
    freeze['source_reuse']=deepcopy(proof['declaration'])
    with pytest.raises(ValueError):p.verify(freeze)


def test_exact_old_owner_must_have_exited(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:12.))
    with pytest.raises(ValueError):p.require_absent([dict(pid=123,created=12.)])
    p.require_absent([dict(pid=123,created=11.)])


def test_authorization_binds_no_collection_and_no_promotion(monkeypatch):
    original=c.read(prep.AUTHORIZATION)
    assert original['automatic_launch_after_checks']
    prep.authorization()
    for key,value in [('new_source_games',1),('reference_promoted',True),('authorized',False)]:
        bad={**original,key:value}
        monkeypatch.setattr(c,'read',lambda path:bad)
        with pytest.raises(ValueError):prep.authorization()


def test_receipt_cannot_substitute_models_for_model_free_checks(monkeypatch):
    monkeypatch.setattr(c,'read',lambda _:dict(exit_code=0,models=1,games=0,inputs=[{}]))
    with pytest.raises(ValueError):prep.require_test_receipt('unused')


def test_review_rejects_unresolved_findings(monkeypatch):
    monkeypatch.setattr(c,'read',lambda _:dict(approved=True,blocking_findings=['unresolved'],inputs=[{}]))
    with pytest.raises(ValueError):prep.require_review('unused')


@pytest.mark.parametrize('key,bad_value',[('domain','different'),('bootstrap_seed',20261101128),
    ('new_setup_count',200),('prior_seed_reuse',True),('explicitly_disjoint_previous_pilot',{})])
def test_wrong_seed_study_rejected(monkeypatch,key,bad_value):
    from tools.splendorust_policy_head_20261104.assessment import design as ad
    good=dict(status='passed',collisions=[],domain=ad.DOMAIN,bootstrap_seed=ad.BOOTSTRAP_SEED,
        new_setup_count=203,prior_seed_reuse=False,explicitly_disjoint_previous_pilot=c.artifact(ad.PRIOR_ASSESSMENT_SCHEDULE),checked=[])
    monkeypatch.setattr(c,'verify',lambda _:None)
    prep.verify_seed_receipt(good)
    good[key]=bad_value
    with pytest.raises(ValueError):prep.verify_seed_receipt(good)
