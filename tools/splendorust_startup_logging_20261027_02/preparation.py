"""Freeze new sources and retain every previous failed attempt unchanged."""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil
from . import common as c,design
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS,nested_artifacts
from tools.future_training_rules_20261021 import RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT

def preserve_previous():
    previous=design.PRESERVED_ATTEMPT
    c.require(c.file_hash(previous/'freeze.json')==design.PRESERVED_FREEZE_SHA,'Previous freeze changed')
    review=previous/'postrun/independent_failure_review.json'
    c.require(c.file_hash(review)==design.PRESERVED_REVIEW_SHA,'Previous failure review changed')
    value=c.read(review)
    c.require(value['status']=='saved_replay_and_failure_evidence_verified_original_attempt_incomplete' and value['accepted'] is False and value['attempt_accepted'] is False,'Prior failure must remain incomplete')
    def authenticate(items):
        unique={item['path']:item for item in items}
        for item in unique.values():
            path=Path(item['path']);path=path if path.is_absolute() else c.ROOT/path
            c.require(path.is_file() and path.stat().st_size==item['bytes'] and c.file_hash(path)==item['sha256'],'Prior reviewed evidence changed: '+item['path'])
    authenticate(list(nested_artifacts(value)))
    for key in ('component_receipt','failure_evidence_inventory'):
        bound=value['preservation'][key]
        authenticate(list(nested_artifacts(c.read(c.ROOT/bound['path']))))
    c.require(c.read(previous/'completion.json')['status']=='incomplete','Prior completion changed')
    old=c.read(previous/'freeze.json');c.verify(old['inputs'])
    inventory=dict(prior_freeze=c.artifact(previous/'freeze.json'),review=c.artifact(review),
        source_input_count=len(old['inputs']),prior_status='incomplete',attempt_accepted=False,
        prior_attempt_artifacts=[c.artifact(p) for p in sorted(previous.rglob('*')) if p.is_file()],
        all_earlier_failures_preserved=True)
    design.fixed(c.OUT/'prelaunch/prior_attempt_preservation_before.json',inventory)
    return old,inventory

def contract():
    return dict(phase=c.PHASE,job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES,max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,max_rss_bytes=2*c.GiB,
        minimum_available_ram_bytes=2*c.GiB,initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        diagnostic_available_ram_bytes=c.DIAGNOSTIC_AVAILABLE_RAM_BYTES,host_observer_max_bytes=c.HOST_OBSERVER_MAX_BYTES,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
        seconds_per_move=1.,internal_seconds=.9,research_seconds_per_move=1.,research_internal_seconds=.9,
        research_log_max_bytes=128*c.MiB,frozen_log_max_bytes=16*c.MiB,research_response_max_bytes=4*c.MiB,
        frozen_response_max_bytes=256*1024,request_max_bytes=c.MiB,placement_mode='isolated',
        max_workers=4,max_graph_builds=4,qualification_decisions=208,total_requests=96,overhead_requests=64,diagnostic_requests=32,
        scientific_decisions=304,parent_requests=104,graph_cases=44,operational_games=0,main_games=0,new_games=0,
        maximum_reproductions=96,automatic_retries=0,failed_decision_retries=0,training=False)

def freeze():
    from . import adapters,qualification,runner,host_observer,placement
    c.require(not(c.OUT/'freeze.json').exists(),'Never replace frozen execution')
    design.prepare();old,preserved=preserve_previous()
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial disk reserve')
    tests=c.read(c.OUT/'prelaunch/model_free_tests_final_02.json');c.require(tests['passed'] is True and tests['exit_code']==0,'Tests must pass');c.verify(tests['inputs'])
    review=c.read(c.OUT/'prelaunch/implementation_review.json');c.require(review['approved'] is True,'Independent review required');c.verify(review['inputs'])
    profiles={f'{q}-{p}':c.artifact(adapters.profile_path(q,p)) for q,p in ((16,3),(64,6))}
    qitem=design.fixed(c.OUT/'qualification_plan.json',qualification.build_plan());pins={}
    def add(item):
        c.require(c.artifact(c.ROOT/item['path'])==item,'Input changed: '+item['path'])
        if item['path'] in pins:c.require(pins[item['path']]==item,'Conflicting source pin')
        pins[item['path']]=item
    for item in old['inputs']+preserved['prior_attempt_artifacts']+review['inputs']:add(item)
    for folder in (c.ROOT/'tools',c.ROOT/'src'):
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' not in path.parts:add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_startup_logging*.py')):add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file():add(c.artifact(path))
    for item in profiles.values():add(item)
    for item in nested_artifacts(design.build_panel()):add(item)
    add(c.artifact(c.PROTOCOL));add(c.artifact(c.ROOT/'docs/engineering/future_training_rules_20261021.md'))
    frozen={key:deepcopy(old[key]) for key in RUNTIME_KEYS};frozen.update(contract())
    frozen.update(schema='startup_logging_execution_v1',time=c.now(),placement=placement.plan(old['topology']),
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        protocol=c.artifact(c.PROTOCOL),study_plan=c.artifact(c.OUT/'study_plan.json'),panel=c.artifact(design.PANEL),
        request_matrix=c.artifact(design.MATRIX),qualification_plan=qitem,research_profiles=profiles,
        prior_accepted_attempt=dict(freeze=c.artifact(design.PRESERVED_ATTEMPT/'freeze.json'),
            acceptance=c.artifact(design.PRESERVED_ATTEMPT/'postrun/independent_failure_review.json'),
            preservation=c.artifact(c.OUT/'prelaunch/prior_attempt_preservation_before.json'),
            accepted_status='incomplete',attempt_accepted=False,completion_accepted=False,admission_passed=False),
        candidate_world_pool=6,opponent_world_pool=3,seed_label='A',extra_host_sampler_enabled=False,
        engineering_contract=design.engineering_contract(),host_observer_contract=design.host_observer_contract(),
        host_observer_runtime=host_observer.runtime_identity(),external_candidate_testing='paused',larger_data_deferred=True,
        native_recompiled=False,native_core_unchanged=True,players_unchanged=True,inputs=sorted(pins.values(),key=lambda x:x['path']))
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preserve','freeze']);args=parser.parse_args()
    if args.action=='prepare':print(c.canonical(design.prepare()))
    elif args.action=='preserve':
        _,value=preserve_previous();print(c.canonical(dict(preserved_files=len(value['prior_attempt_artifacts']))))
    else:print(c.canonical(freeze()))
