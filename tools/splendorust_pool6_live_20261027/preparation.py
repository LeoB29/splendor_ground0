"""Finite prior acceptance and separate immutable admission/assessment freezes."""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil
import subprocess
from . import common as c, design, schedule
from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT

RUNTIME_KEYS=('compression_runtime','external_runtime','native_artifacts','numpy_version','placement',
    'sampler_api_modules','source_guard','topology','torch_package_version','torch_runtime',
    'torch_runtime_version','python','python_executable_sha256','unchanged_graph_backend_identity',
    'graph_parity_plan','graph_parity_fixtures','research_native_artifacts','research_native_roles',
    'research_native_sha256','build_receipt')


def fixed(path,value):return design.fixed(path,value)


def nested_artifacts(value):
    if isinstance(value,dict):
        if {'path','sha256','bytes'}<=set(value):yield {k:value[k] for k in ('path','sha256','bytes')}
        for child in value.values():yield from nested_artifacts(child)
    elif isinstance(value,list):
        for child in value:yield from nested_artifacts(child)


def preserve_previous():
    previous=design.PREVIOUS
    c.require(c.file_hash(previous/'freeze.json')==design.PREVIOUS_FREEZE_SHA,'Previous freeze differs')
    review=previous/'postrun/independent_completion_review.json'
    c.require(c.file_hash(review)==design.PREVIOUS_REVIEW_SHA,'Previous accepted review differs')
    accepted=c.read(review)
    c.require(accepted['accepted'] is True and accepted['freeze_sha256']==design.PREVIOUS_FREEZE_SHA,
              'Previous review is not positive for exact freeze')
    c.verify(accepted['consumed_inputs'])
    reviewed={x['path']:x for x in accepted['consumed_inputs']}
    critical=[]
    for name in ('completion.json','analysis.json','completed_evidence_seal.json','verification_summary.json',
                 'guardian_os_exit.json','execution.json','watchdog_execution.json'):
        item=c.artifact(previous/name)
        c.require(reviewed.get(item['path'])==item,'Prior reviewed result changed: '+name)
        critical.append(item)
    old=c.read(previous/'freeze.json');c.verify(old['inputs'])
    c.require(c.read(previous/'completion.json')['status']=='completed_verified','Prior incomplete')
    inventory=dict(prior_freeze=c.artifact(previous/'freeze.json'),review=c.artifact(review),
        critical_reused_results=critical,source_input_count=len(old['inputs']),
        prior_attempt_artifacts=[c.artifact(p) for p in sorted(previous.rglob('*')) if p.is_file()],
        prior_status='completed_verified',all_earlier_failures_preserved=True)
    fixed(c.OUT/'prelaunch/prior_attempt_preservation_before.json',inventory)
    return old,inventory


def seed_inventory(rows):
    targets={int(row['seed_decimal']) for row in rows}
    reserved={int(x['seed_decimal']) for x in c.read(design.PANEL)['positions']}
    operational={int(x['seed_decimal']) for x in schedule.stage_schedule('operational')}
    main={int(x['seed_decimal']) for x in schedule.stage_schedule('main')}
    c.require(len(reserved)==18 and len(operational)==3 and len(main)==100 and
        reserved.isdisjoint(operational) and reserved.isdisjoint(main) and operational.isdisjoint(main),
        'Reserved, operational and assessment worlds must be disjoint')
    if c.PHASE=='qualification':targets.update(int(x['seed_decimal']) for x in c.read(design.PANEL)['positions'])
    found=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json',
        '-g','*schedule*.json','-g','*seed_registry*.json'],cwd=c.ROOT,capture_output=True,text=True,check=True)
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
    c.require(not collisions,'Fresh registered seed collision')
    return dict(status='passed',checked=checked,skipped_oversized=skipped,collisions=collisions,
        direct_new_namespace_disjointness=dict(reserved=18,operational=3,assessment=100,passed=True),
        scope='Registered manifests/schedules/seed registries up to8MiB; not exhaustive arbitrary logs')


def accepted_admission():
    review_path=c.QUAL_OUT/'postrun/independent_completion_review.json'
    review=c.read(review_path);frozen=c.read(c.QUAL_OUT/'freeze.json')
    c.require(review['accepted'] is True and review['freeze_sha256']==c.file_hash(c.QUAL_OUT/'freeze.json'),
              'Exact admission independently accepted before assessment')
    c.verify(review['consumed_inputs'])
    consumed={x['path']:x for x in review['consumed_inputs']}
    for name in ('completion.json','verification_summary.json','completed_evidence_seal.json',
                 'guardian_os_exit.json','execution.json','watchdog_execution.json','assessment_gate.json'):
        item=c.artifact(c.QUAL_OUT/name)
        c.require(consumed.get(item['path'])==item,'Admission result not bound by accepted review: '+name)
    c.require(c.read(c.QUAL_OUT/'completion.json')['status']=='qualified_verified','Admission incomplete or negative')
    gate=c.read(c.QUAL_OUT/'assessment_gate.json')
    c.require(gate['passed'] is True,'Admission/resource gate prevents assessment')
    c.verify(frozen['inputs'])
    seal=c.read(c.QUAL_OUT/'completed_evidence_seal.json');c.verify(seal['artifacts'])
    inventory=[c.artifact(p) for p in sorted(c.QUAL_OUT.rglob('*')) if p.is_file()]
    return dict(status='accepted',freeze=c.artifact(c.QUAL_OUT/'freeze.json'),review=c.artifact(review_path),
        inputs=frozen['inputs'],artifacts=inventory,gate=c.artifact(c.QUAL_OUT/'assessment_gate.json'))


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
        qualification_decisions=194 if c.PHASE=='qualification' else 0,
        operational_games=6 if c.PHASE=='qualification' else 0,main_games=200 if c.PHASE=='assessment' else 0,
        maximum_reproductions=48,automatic_retries=0,failed_decision_retries=0,training=False)


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
        if c.PHASE=='qualification':fixed(path,adapters.profile(quota,pool))
        else:c.require(c.read(path)==adapters.profile(quota,pool),'Accepted candidate profile differs')
        profiles[f'{quota}-{pool}']=c.artifact(path)
    qitem=None;acceptance=None
    if c.PHASE=='qualification':qitem=fixed(c.OUT/'qualification_plan.json',qualification.build_plan())
    else:acceptance=accepted_admission()
    fixed(c.OUT/'runtime_sources.json',source_guard)
    fixed(c.OUT/'seed_inventory.json',seed_inventory(schedule.make_schedule()))
    pins={}
    def add(item):
        c.require(c.artifact(c.ROOT/item['path'])==item,'Frozen input changed: '+item['path'])
        if item['path'] in pins:c.require(pins[item['path']]==item,'Conflicting frozen identity')
        pins[item['path']]=item
    for item in old['inputs']+preserved['prior_attempt_artifacts']:add(item)
    if acceptance:
        for item in acceptance['inputs']+acceptance['artifacts']:add(item)
    for item in review['inputs']:add(item)
    # Explicit finite full Python source inventory covers transitive imported helpers.
    # No model/data directory is recursively copied or modified.
    for folder in (c.ROOT/'tools',c.ROOT/'src'):
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' not in path.parts:add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_pool6_live_20261027*.py')):add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file():add(c.artifact(path))
    for item in profiles.values():add(item)
    for item in nested_artifacts(qualification.build_plan()):add(item)
    add(c.artifact(design.PANEL));add(c.artifact(c.PROTOCOL))
    add(c.artifact(c.ROOT/'docs/engineering/future_training_rules_20261021.md'))
    frozen={key:deepcopy(old[key]) for key in RUNTIME_KEYS}
    frozen.update(contract())
    frozen.update(schema='splendorust_pool6_live_execution_v1',time=c.now(),attempt=1,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        protocol=c.artifact(c.PROTOCOL),study_plan=c.artifact(c.OUT/'study_plan.json'),
        schedule=c.artifact(c.OUT/'schedule.json'),qualification_plan=qitem,
        research_profiles=profiles,accepted_admission=acceptance,
        prior_accepted_attempt=dict(freeze=c.artifact(design.PREVIOUS/'freeze.json'),
            acceptance=c.artifact(design.PREVIOUS/'postrun/independent_completion_review.json'),
            preservation=c.artifact(c.OUT/'prelaunch/prior_attempt_preservation_before.json')),
        candidate_world_pool=6,opponent_world_pool=3,seed_label='A',
        external_candidate_testing='paused',larger_data_deferred=True,
        native_recompiled=False,native_core_unchanged=True,
        inputs=sorted(pins.values(),key=lambda x:x['path']))
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preserve','freeze']);args=parser.parse_args()
    if args.action=='preserve':
        _,preserved=preserve_previous();print(c.canonical(dict(preserved_files=len(preserved['prior_attempt_artifacts']))))
    else:print(c.canonical(freeze()))
