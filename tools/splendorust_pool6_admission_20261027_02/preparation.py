"""Freeze one fresh admission, preserving negative and diagnostic predecessors."""
from copy import deepcopy
import argparse
import shutil
import subprocess
from . import common as c,design,schedule
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS,nested_artifacts
from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT

NEGATIVE=c.ROOT/'outputs/splendorust_pool6_live_20261027_qualification_01'
NEGATIVE_FREEZE='247b143d1f5256048b336231fb3ba3c544f74ceb7580e7087a1e69b967d480a0'
NEGATIVE_REVIEW='01405c2631723accafda75a642a64d481ead73c31fdd1307e9fde80295198feb'


def accepted_inventory(path,freeze_sha,review_sha,status):
    c.require(c.file_hash(path/'freeze.json')==freeze_sha,'Predecessor freeze changed')
    review_path=path/'postrun/independent_completion_review.json'
    c.require(c.file_hash(review_path)==review_sha,'Predecessor independent review changed')
    review=c.read(review_path)
    c.require(review['accepted'] is True and review['freeze_sha256']==freeze_sha and
        review['benchmark_authorized_to_launch'] is False,'Predecessor accepted scope changed')
    if status=='not_admitted_verified':c.require(review['admission_passed'] is False,'Prior negative cannot become admission')
    else:c.require(review['prior_admission_passed'] is False,'Repeat diagnostic cannot reverse negative admission')
    c.verify(review['consumed_inputs']);reviewed={x['path']:x for x in review['consumed_inputs']}
    critical=[]
    for name in ('completion.json','verification_summary.json','completed_evidence_seal.json',
                 'guardian_os_exit.json','execution.json','watchdog_execution.json'):
        item=c.artifact(path/name);c.require(reviewed.get(item['path'])==item,'Unauthenticated prior outcome: '+name);critical.append(item)
    c.require(c.read(path/'completion.json')['status']==status,'Predecessor completion changed')
    old=c.read(path/'freeze.json');c.verify(old['inputs']);c.verify(c.read(path/'completed_evidence_seal.json')['artifacts'])
    return old,dict(prior_freeze=c.artifact(path/'freeze.json'),review=c.artifact(review_path),
        critical_reused_results=critical,source_input_count=len(old['inputs']),prior_status=status,
        prior_attempt_artifacts=[c.artifact(p) for p in sorted(path.rglob('*')) if p.is_file()],
        admission_passed=False,benchmark_authorized_to_launch=False,all_earlier_failures_preserved=True)


def preserve_previous():
    old,value=accepted_inventory(design.PREVIOUS,design.PREVIOUS_FREEZE_SHA,design.PREVIOUS_REVIEW_SHA,'completed_verified')
    _,negative=accepted_inventory(NEGATIVE,NEGATIVE_FREEZE,NEGATIVE_REVIEW,'not_admitted_verified')
    value['additional_prior_attempts']=[negative]
    design.fixed(c.OUT/'prelaunch/prior_attempt_preservation_before.json',value)
    return old,value


def seed_inventory():
    from tools.splendorust_pool6_live_20261027 import schedule as old_schedule
    positions=c.read(design.PANEL)['positions']
    reserved={int(x['seed_decimal']) for x in positions}
    operations={int(x['seed_decimal']) for x in schedule.make_schedule()}
    old_reserved={int(x['seed_decimal']) for x in c.read(NEGATIVE/'reserved_panel.json')['positions']}
    old_ops={int(x['seed_decimal']) for x in old_schedule.stage_schedule('operational')}
    old_main={int(x['seed_decimal']) for x in old_schedule.stage_schedule('main')}
    targets=reserved|operations
    c.require(len(reserved)==18 and len(operations)==3 and len(targets)==21 and
        targets.isdisjoint(old_reserved|old_ops|old_main),'Fresh reserved/operational seeds collide or repeat')
    found=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json','-g','*schedule*.json',
        '-g','*seed_registry*.json','-g','*reserved_panel*.json'],cwd=c.ROOT,capture_output=True,text=True,check=True)
    checked=[];skipped=[];collisions=[]
    def visit(value,path):
        if type(value) is int or isinstance(value,str) and value.isdecimal():
            if int(value) in targets:collisions.append(path)
        elif isinstance(value,list):
            for child in value:visit(child,path)
        elif isinstance(value,dict):
            for child in value.values():visit(child,path)
    for name in sorted(found.stdout.splitlines()):
        path=(c.ROOT/name).resolve()
        if path.is_relative_to(c.OUT.resolve()):continue
        if path.stat().st_size>8*c.MiB:skipped.append(c.artifact(path));continue
        visit(c.read(path),path.relative_to(c.ROOT).as_posix());checked.append(c.artifact(path))
    c.require(not collisions,'Fresh registered seed collision; no replacement')
    return dict(status='passed',checked=checked,skipped_oversized=skipped,collisions=collisions,
        fresh_reserved_seeds=18,fresh_operational_seeds=3,new_assessment_seeds=0,
        direct_prior_namespace_exclusions=dict(reserved=18,operational=3,planned_assessment=100),
        scope='Registered manifests/schedules/seed registries/reserved panels up to8MiB plus direct prior namespaces; not exhaustive arbitrary logs')


def contract():
    return dict(phase=c.PHASE,job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES,max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,max_rss_bytes=2*c.GiB,
        minimum_available_ram_bytes=2*c.GiB,initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
        seconds_per_move=1.,internal_seconds=.9,research_seconds_per_move=1.,research_internal_seconds=.9,
        research_log_max_bytes=128*c.MiB,frozen_log_max_bytes=16*c.MiB,research_response_max_bytes=4*c.MiB,
        frozen_response_max_bytes=256*1024,request_max_bytes=c.MiB,placement_mode='isolated',
        max_workers=c.MAX_WORKERS,max_graph_builds=c.MAX_GRAPH_BUILDS,
        qualification_decisions=194,operational_games=6,main_games=0,maximum_reproductions=48,
        automatic_retries=0,failed_decision_retries=0,training=False)


def freeze():
    from . import adapters,qualification,runner
    c.require(not(c.OUT/'freeze.json').exists(),'Never replace execution freeze')
    design.prepare();source_guard=verify_runtime_sources();old,preserved=preserve_previous()
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial disk reserve')
    test=c.read(c.OUT/'prelaunch/model_free_tests.json')
    c.require(test['passed'] is True and test['exit_code']==0,'Model-free tests not passed')
    review=c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(review['approved'] is True,'Independent implementation review not passed');c.verify(review['inputs'])
    profiles={}
    for quota,pool in ((16,3),(64,6)):
        path=adapters.profile_path(quota,pool)
        c.require(c.read(path)==adapters.profile(quota,pool),'Original immutable profile changed')
        profiles[f'{quota}-{pool}']=c.artifact(path)
    qitem=design.fixed(c.OUT/'qualification_plan.json',qualification.build_plan())
    design.fixed(c.OUT/'runtime_sources.json',source_guard)
    design.fixed(c.OUT/'seed_inventory.json',seed_inventory())
    pins={}
    def add(item):
        c.require(c.artifact(c.ROOT/item['path'])==item,'Input changed: '+item['path'])
        if item['path'] in pins:c.require(pins[item['path']]==item,'Conflicting pin')
        pins[item['path']]=item
    for item in old['inputs']+preserved['prior_attempt_artifacts']+review['inputs']:add(item)
    for extra in preserved['additional_prior_attempts']:
        for item in extra['prior_attempt_artifacts']:add(item)
    for folder in (c.ROOT/'tools',c.ROOT/'src'):
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' not in path.parts:add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_pool6_admission_20261027_02*.py')):add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file():add(c.artifact(path))
    for item in profiles.values():add(item)
    for item in nested_artifacts(qualification.build_plan()):add(item)
    add(c.artifact(design.PANEL));add(c.artifact(c.PROTOCOL))
    add(c.artifact(c.ROOT/'docs/engineering/future_training_rules_20261021.md'))
    frozen={key:deepcopy(old[key]) for key in RUNTIME_KEYS}
    frozen.update(contract())
    frozen.update(schema='splendorust_pool6_fresh_admission_execution_v1',time=c.now(),attempt=2,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        protocol=c.artifact(c.PROTOCOL),study_plan=c.artifact(c.OUT/'study_plan.json'),
        schedule=c.artifact(c.OUT/'schedule.json'),qualification_plan=qitem,reserved_panel=c.artifact(design.PANEL),
        research_profiles=profiles,accepted_admission=None,
        prior_accepted_attempt=dict(freeze=c.artifact(design.PREVIOUS/'freeze.json'),
            acceptance=c.artifact(design.PREVIOUS/'postrun/independent_completion_review.json'),
            preservation=c.artifact(c.OUT/'prelaunch/prior_attempt_preservation_before.json')),
        candidate_world_pool=6,opponent_world_pool=3,seed_label='A',
        external_candidate_testing='paused',larger_data_deferred=True,
        native_recompiled=False,native_core_unchanged=True,players_unchanged=True,
        benchmark_authorized_to_launch=False,inputs=sorted(pins.values(),key=lambda x:x['path']))
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preserve','freeze']);args=parser.parse_args()
    if args.action=='prepare':print(c.canonical(design.prepare()))
    elif args.action=='preserve':
        _,value=preserve_previous();print(c.canonical(dict(preserved_files=len(value['prior_attempt_artifacts']),
            additional_prior_files=sum(len(x['prior_attempt_artifacts']) for x in value['additional_prior_attempts']))))
    else:print(c.canonical(freeze()))
