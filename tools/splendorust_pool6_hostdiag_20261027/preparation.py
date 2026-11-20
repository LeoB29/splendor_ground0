"""Preserve the verified negative attempt; freeze one finite diagnostic."""
from copy import deepcopy
import argparse
import shutil
from . import common as c,design
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS,nested_artifacts
from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT


def preserve_previous():
    previous=design.PREVIOUS
    c.require(c.file_hash(previous/'freeze.json')==design.PREVIOUS_FREEZE_SHA,'Prior freeze changed')
    review_path=previous/'postrun/independent_completion_review.json'
    c.require(c.file_hash(review_path)==design.PREVIOUS_REVIEW_SHA,'Prior independent negative review changed')
    review=c.read(review_path)
    c.require(review['accepted'] is True and review['admission_passed'] is False and
        review['benchmark_authorized_to_launch'] is False and review['freeze_sha256']==design.PREVIOUS_FREEZE_SHA,
        'Prior review accepts negative admission only')
    c.verify(review['consumed_inputs'])
    reviewed={x['path']:x for x in review['consumed_inputs']}
    critical=[]
    for name in ('completion.json','verification_summary.json','completed_evidence_seal.json',
                 'guardian_os_exit.json','execution.json','watchdog_execution.json'):
        item=c.artifact(previous/name)
        c.require(reviewed.get(item['path'])==item,'Prior result not authenticated: '+name)
        critical.append(item)
    c.require(c.read(previous/'completion.json')['status']=='not_admitted_verified','Prior status changed')
    old=c.read(previous/'freeze.json');c.verify(old['inputs'])
    c.verify(c.read(previous/'completed_evidence_seal.json')['artifacts'])
    inventory=dict(prior_freeze=c.artifact(previous/'freeze.json'),review=c.artifact(review_path),
        critical_reused_results=critical,source_input_count=len(old['inputs']),
        prior_attempt_artifacts=[c.artifact(p) for p in sorted(previous.rglob('*')) if p.is_file()],
        prior_status='not_admitted_verified',admission_passed=False,
        benchmark_authorized_to_launch=False,all_earlier_failures_preserved=True)
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
        max_workers=2,max_graph_builds=2,qualification_decisions=104,total_requests=216,overhead_requests=72,diagnostic_requests=144,
        scientific_decisions=320,parent_requests=220,graph_cases=22,
        operational_games=0,main_games=0,new_games=0,maximum_reproductions=216,
        automatic_retries=0,failed_decision_retries=0,training=False)


def freeze():
    from . import adapters,qualification,runner,host_observer
    c.require(not(c.OUT/'freeze.json').exists(),'Never replace a frozen execution')
    design.prepare();old,preserved=preserve_previous()
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial disk reserve')
    test=c.read(c.OUT/'prelaunch/model_free_tests.json')
    c.require(test['passed'] is True and test['exit_code']==0,'Model-free tests not passed')
    review=c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(review['approved'] is True,'Independent implementation review not passed');c.verify(review['inputs'])
    profiles={}
    for quota,pool in ((16,3),(64,6)):
        path=adapters.profile_path(quota,pool)
        c.require(c.read(path)==adapters.profile(quota,pool),'Frozen scientific profile changed')
        profiles[f'{quota}-{pool}']=c.artifact(path)
    qitem=design.fixed(c.OUT/'qualification_plan.json',qualification.build_plan())
    pins={}
    def add(item):
        c.require(c.artifact(c.ROOT/item['path'])==item,'Input changed: '+item['path'])
        if item['path'] in pins:c.require(pins[item['path']]==item,'Conflicting pin')
        pins[item['path']]=item
    for item in old['inputs']+preserved['prior_attempt_artifacts']+review['inputs']:add(item)
    for folder in (c.ROOT/'tools',c.ROOT/'src'):
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' not in path.parts:add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_pool6_hostdiag_20261027*.py')):add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file():add(c.artifact(path))
    for item in profiles.values():add(item)
    for item in nested_artifacts(design.build_panel()):add(item)
    add(c.artifact(c.PROTOCOL));add(c.artifact(c.ROOT/'docs/engineering/future_training_rules_20261021.md'))
    frozen={key:deepcopy(old[key]) for key in RUNTIME_KEYS}
    frozen.update(contract())
    frozen.update(schema='splendorust_pool6_hostdiag_execution_v1',time=c.now(),attempt=1,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        protocol=c.artifact(c.PROTOCOL),study_plan=c.artifact(c.OUT/'study_plan.json'),
        panel=c.artifact(design.PANEL),request_matrix=c.artifact(design.MATRIX),qualification_plan=qitem,
        research_profiles=profiles,
        prior_accepted_attempt=dict(freeze=c.artifact(design.PREVIOUS/'freeze.json'),
            acceptance=c.artifact(design.PREVIOUS/'postrun/independent_completion_review.json'),
            preservation=c.artifact(c.OUT/'prelaunch/prior_attempt_preservation_before.json'),
            accepted_status='not_admitted_verified',admission_passed=False),
        candidate_world_pool=6,opponent_world_pool=3,seed_label='A',
        overhead_contract=design.overhead_contract(),host_observer_contract=design.host_observer_contract(),
        host_observer_runtime=host_observer.runtime_identity(),
        external_candidate_testing='paused',larger_data_deferred=True,
        native_recompiled=False,native_core_unchanged=True,players_unchanged=True,
        inputs=sorted(pins.values(),key=lambda x:x['path']))
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preserve','freeze']);args=parser.parse_args()
    if args.action=='prepare':print(c.canonical(design.prepare()))
    elif args.action=='preserve':
        _,value=preserve_previous();print(c.canonical(dict(preserved_files=len(value['prior_attempt_artifacts']))))
    else:print(c.canonical(freeze()))
