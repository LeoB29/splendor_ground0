"""Prepare and exclusively freeze six new operational games, without model calls."""
import argparse
from copy import deepcopy
from pathlib import Path
from . import common as c

REUSED_NAMES=('panel.json','numerical_plan.json','qualification_plan.json')
REVIEWS=('independent_review_worker.json','independent_review_controller.json','independent_review_design.json')


def same_or_save(path,value):
    if path.exists():
        c.require(c.read(path)==value,'Never replace a preparation receipt: '+str(path))
        return c.artifact(path)
    return c.save(path,value)


def accepted_predecessor():
    c.require(c.file_hash(c.PRACTICAL_OUT/'freeze.json')==c.PRACTICAL_FREEZE_SHA and
        c.file_hash(c.PRACTICAL_CLOSEOUT)==c.PRACTICAL_CLOSEOUT_SHA and
        c.file_hash(c.PRACTICAL_ACCEPTANCE)==c.PRACTICAL_ACCEPTANCE_SHA,
        'Exact accepted practical predecessor')
    old=c.read(c.PRACTICAL_OUT/'freeze.json');closed=c.read(c.PRACTICAL_CLOSEOUT)
    acceptance=c.read(c.PRACTICAL_ACCEPTANCE)
    c.require(acceptance['approved'] is True and not acceptance['blocking_findings'] and
        acceptance['all_recorded_exact_identities_absent'] is True and
        all(v==0 for v in acceptance['actual_owner_exit_codes'].values()),'Accepted actual exits and cleanup')
    c.require(closed['useful_speedup'] is True and closed['candidate_numerical_admission_passed'] is True and
        closed['candidate_promoted'] is False and closed['full_numerical_outputs_exact'] is False and
        closed['strength_equivalence_established'] is False and closed['next_study_launched'] is False,
        'Predecessor establishes measured speedup only, with nonidentical numerical outputs')
    c.require(old['candidate_module']==c.artifact(c.CANDIDATE_SOURCE) and
        c.file_hash(c.CANDIDATE_SOURCE)==c.CANDIDATE_SOURCE_SHA,'Unchanged sole neural implementation')
    c.verify(old['inputs']);c.verify(acceptance['inputs'])
    seal=c.read(c.PRACTICAL_OUT/'completed_evidence_seal.json')['artifacts'];c.verify(seal)
    c.require(c.read(c.FAILED_OUT/'completion.json')['status']=='incomplete' and
        c.file_hash(c.FAILED_OUT/'freeze.json')==c.FAILED_FREEZE_SHA,'Preserved prior failed practical attempt')
    inventory=c.read(c.FAILED_INVENTORY)
    c.require({x['path'] for x in inventory['files']}==
        {p.relative_to(c.ROOT).as_posix() for p in c.FAILED_OUT.rglob('*') if p.is_file()},
        'Exact prior failed attempt inventory; no discarded or replaced evidence')
    c.verify(inventory['files'])
    return old,closed,acceptance,seal


def prepare():
    from . import design
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources()
    old,closed,acceptance,seal=accepted_predecessor()
    for name in REUSED_NAMES:
        source=c.PRACTICAL_OUT/name;target=c.OUT/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():c.require(target.read_bytes()==source.read_bytes(),'Repeated qualification bytes unchanged')
        else:
            with target.open('xb') as stream:stream.write(source.read_bytes())
    documents=design.prepare()
    seed_check=c.OUT/'seed_freshness.json'
    if seed_check.exists():
        registered=c.read(seed_check)
        c.require(registered['status']=='passed' and not registered['collisions'] and
            registered['domain']==design.DOMAIN,'Prior seed freshness receipt')
        c.verify(registered['checked'])
    else:
        c.save(seed_check,design.seed_inventory())
    documents['seed_freshness']=c.artifact(seed_check)
    value=dict(schema='network_live_preparation_v1',models=0,games=0,source_guard=sources,
        accepted_predecessor=c.artifact(c.PRACTICAL_OUT/'freeze.json'),
        accepted_predecessor_closeout=c.artifact(c.PRACTICAL_CLOSEOUT),
        accepted_predecessor_receipt=c.artifact(c.PRACTICAL_ACCEPTANCE),
        repeated_qualification_inputs=[c.artifact(c.OUT/name) for name in REUSED_NAMES],
        new_operational_inputs=documents,repeated_numerical_fixtures=True,
        candidate_implementation_unchanged=True,prior_failures_preserved=True,
        authorization='User: ok do it after six fresh operational games recommendation')
    return same_or_save(c.OUT/'preparation.json',value)


def require_tests(path):
    v=c.read(path)
    c.require(v['exit_code']==0 and v['models']==v['games']==0 and v['inputs'],
        'Passed actual model-free tests bound to current implementation')
    c.verify(v['inputs']+[v['log']]);return v


def require_review(path):
    v=c.read(path)
    c.require(v['approved'] is True and not v.get('blocking_findings',[]) and v['inputs'],
        'Independent current-source review must approve')
    c.verify(v['inputs']);return v


def freeze():
    from . import control,design
    c.require(not(c.OUT/'freeze.json').exists(),'One exclusive live qualification freeze')
    old,closed,acceptance,seal=accepted_predecessor();prepared=prepare()
    tests=c.OUT/'prelaunch/model_free_tests.json';require_tests(tests)
    reviews=[c.OUT/'prelaunch'/name for name in REVIEWS]
    for p in reviews:require_review(p)
    value=deepcopy(old)
    for key in ('evaluation_matrix','evaluation_requests','selection_rule','assessment_reuse',
        'decision_disagreement_stops_timing','timing_assessment','previous_practical_attempt',
        'previous_practical_closeout','previous_failed_inventory',
        'fresh_numerical_cases_per_worker','fresh_numerical_cases_maximum'):
        value.pop(key,None)
    value.update(control.contract(),time=c.now(),study_kind='held_network_implementation_live_operational_qualification',
        protocol=c.artifact(c.PROTOCOL),attempt_number=1,separate_user_authorized_attempt=True,
        preparation=prepared,accepted_predecessor=c.artifact(c.PRACTICAL_OUT/'freeze.json'),
        accepted_predecessor_closeout=c.artifact(c.PRACTICAL_CLOSEOUT),
        accepted_predecessor_receipt=c.artifact(c.PRACTICAL_ACCEPTANCE),
        model_free_tests=c.artifact(tests),implementation_reviews=[c.artifact(p) for p in reviews],
        numerical_assessment='repeat of accepted unchanged numerical qualification fixtures',
        operational_assessment='six fresh games; no strength-equivalence inference',
        original_failed_attempts_preserved=True,training=False,reference_promoted=False)
    for name in REUSED_NAMES:value[name[:-5]]=c.artifact(c.OUT/name)
    value.update(design.prepare())
    value['seed_freshness']=c.artifact(c.OUT/'seed_freshness.json')
    pins={}
    def add(item):
        c.require(pins.setdefault(item['path'],item)==item,'Conflicting frozen descriptor')
    for item in old['inputs']+seal+acceptance['inputs']:add(item)
    for item in c.read(c.OUT/'seed_freshness.json')['checked']:add(item)
    paths=list(Path(__file__).parent.glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('test_splendorust_network_live*_20261031.py'))
    paths+=[p for p in c.OUT.rglob('*') if p.is_file()]
    paths+=[c.PROTOCOL,c.PRACTICAL_OUT/'freeze.json',c.PRACTICAL_CLOSEOUT,c.PRACTICAL_ACCEPTANCE,
        c.PRACTICAL_OUT/'completed_evidence_seal.json']
    # New independent analysis uses only source-pinned existing helpers.
    for path in paths:add(c.artifact(path))
    value['inputs']=sorted(pins.values(),key=lambda x:x['path'])
    c.verify(value['inputs']);control.validate_freeze_contract(value)
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',value),
        source_inputs=len(pins),models=0,games=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze'])
    args=parser.parse_args();print(c.canonical({'prepare':prepare,'freeze':freeze}[args.action]()),flush=True)


if __name__=='__main__':main()
