"""Separate strength-first freezes; old one-second results remain unchanged."""
from copy import deepcopy
import argparse
import shutil
import subprocess
from . import common as c, design, schedule
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS,nested_artifacts
from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT

PRIOR_SPECS=(
 ('outputs/splendorust_pool6_admission_20261027_02',
  '09551c1f9e8782e0d717576aa51f96bff47403637b56cb087fd843ff8c58c0e8',
  'postrun/independent_completion_review.json',
  'e68f6dc7b34c49147a933303ebfc9bdcb97671951e15e143a71a0b4f990f615e','not_admitted_verified'),
 ('outputs/splendorust_sampling_stability_20261027_01',
  'cc550ef7f48597840ed5e41166b317574ef78d8ad88d3ae11b6ef5baca5b1eaa',
  'postrun/independent_completion_review.json',
  '7c1fbf2740a7acfd83bf98eb8ffe2a57266b80271f763362d092542bb9f0a952','completed_verified'),
 ('outputs/splendorust_minimal_logging_20261028_01',
  'a5f5e547a9d27cb4b81a521ca23d60ccc26d8110fb6a0a09bbd105700691a18b',
  'postrun/independent_review.json',
  '67aea43edb26f933f932215d2bd66939767a82ad368d52ce57375011611042c3','overhead_not_qualified_verified'))

def accepted_inventory(spec):
    name,fsha,rname,rsha,status=spec;folder=c.ROOT/name
    c.require(c.file_hash(folder/'freeze.json')==fsha,'Prior freeze changed')
    rpath=folder/rname;c.require(c.file_hash(rpath)==rsha,'Prior accepted review changed')
    review=c.read(rpath);old=c.read(folder/'freeze.json')
    c.require(review['accepted'] is True and review['freeze_sha256']==fsha,'Prior acceptance identity')
    c.require(c.read(folder/'completion.json')['status']==status,'Prior original status changed')
    if status=='not_admitted_verified':
        c.require(review['admission_passed'] is False and review['benchmark_authorized_to_launch'] is False,
                  'Prior negative admission cannot be relabelled')
    if status=='overhead_not_qualified_verified':
        c.require(review['timing_observer_qualified'] is False,'Prior observer remains unqualified')
    c.verify(review.get('consumed_inputs',review.get('inputs',[])));c.verify(old['inputs'])
    c.verify(c.read(folder/'completed_evidence_seal.json')['artifacts'])
    return old,dict(prior_freeze=c.artifact(folder/'freeze.json'),review=c.artifact(rpath),
        prior_status=status,source_input_count=len(old['inputs']),
        prior_attempt_artifacts=[c.artifact(p) for p in sorted(folder.rglob('*')) if p.is_file()],
        negative_admission_remains_negative=True,all_earlier_failures_preserved=True)

def preserve_previous():
    entries=[accepted_inventory(spec) for spec in PRIOR_SPECS]
    base,primary=entries[0];primary['additional_prior_attempts']=[v for _,v in entries[1:]]
    pins={}
    for old,_ in entries:
        for item in old['inputs']:
            if item['path'] in pins:c.require(pins[item['path']]==item,'Conflicting prior pin')
            pins[item['path']]=item
    primary['prior_source_union']=sorted(pins.values(),key=lambda x:x['path'])
    design.fixed(c.OUT/'prelaunch/prior_attempt_preservation_before.json',primary)
    return base,primary

def prepare_profiles():
    from . import adapters
    research,baseline={},{}
    for q,pool in ((16,3),(64,6)):
        path=adapters.profile_path(q,pool)
        if c.PHASE=='qualification':design.fixed(path,adapters.profile(q,pool))
        else:c.require(c.read(path)==adapters.profile(q,pool),'Accepted research profile changed')
        research[f'{q}-{pool}']=c.artifact(path)
    for q in (16,64):
        path=adapters.baseline_profile_path(q)
        if c.PHASE=='qualification':design.fixed(path,adapters.baseline_profile(q))
        else:c.require(c.read(path)==adapters.baseline_profile(q),'Accepted control profile changed')
        baseline[str(q)]=c.artifact(path)
    return research,baseline

def seed_inventory():
    reserved={int(x['seed_decimal']) for x in c.read(design.PANEL)['positions']}
    operations={int(x['seed_decimal']) for x in schedule.stage_schedule('operational')}
    main={int(x['seed_decimal']) for x in schedule.stage_schedule('main')}
    targets=reserved|operations|main
    c.require(len(reserved)==18 and len(operations)==3 and len(main)==200 and len(targets)==221,
              'Exact disjoint fresh world inventories')
    found=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json',
        '-g','*schedule*.json','-g','*seed_registry*.json','-g','*reserved_panel*.json'],
        cwd=c.ROOT,capture_output=True,text=True,check=True)
    checked,skipped,collisions=[],[],[]
    def visit(value,path):
        if type(value) is int or isinstance(value,str) and value.isdecimal():
            if int(value) in targets:collisions.append(path)
        elif isinstance(value,list):
            for item in value:visit(item,path)
        elif isinstance(value,dict):
            for item in value.values():visit(item,path)
    for name in sorted(found.stdout.splitlines()):
        path=(c.ROOT/name).resolve()
        if any(path.is_relative_to(out.resolve()) for out in (c.QUAL_OUT,c.ASSESS_OUT)):continue
        if path.stat().st_size>8*c.MiB:skipped.append(c.artifact(path));continue
        visit(c.read(path),path.relative_to(c.ROOT).as_posix());checked.append(c.artifact(path))
    from tools.splendorust_pool6_live_20261027 import schedule as previous
    prior={int(x['seed_decimal']) for x in previous.stage_schedule('main')}
    c.require(targets.isdisjoint(prior) and not collisions,'Fresh seed collision; no replacement')
    return dict(status='passed',checked=checked,skipped_oversized=skipped,collisions=collisions,
        direct_new_namespace_disjointness=dict(reserved=18,operational=3,assessment=200,passed=True),
        prior_unlaunched_assessment_namespace_excluded=True,
        scope='Registered metadata up to8MiB plus previous planned assessment; not arbitrary-log exhaustion')

def accepted_admission():
    path=c.QUAL_OUT/'postrun/independent_completion_review.json'
    review=c.read(path);frozen=c.read(c.QUAL_OUT/'freeze.json')
    c.require(review['accepted'] is True and review['freeze_sha256']==c.file_hash(c.QUAL_OUT/'freeze.json'),
              'New research qualification requires independent acceptance')
    c.require(frozen['seconds_per_move']==10. and frozen['internal_seconds']==9. and
              frozen['timing_thresholds_do_not_gate_admission'] is True,'Wrong accepted timing policy')
    c.verify(review['consumed_inputs']);consumed={x['path']:x for x in review['consumed_inputs']}
    for name in ('completion.json','verification_summary.json','completed_evidence_seal.json',
                 'guardian_os_exit.json','execution.json','watchdog_execution.json','assessment_gate.json'):
        item=c.artifact(c.QUAL_OUT/name);c.require(consumed.get(item['path'])==item,'Unbound qualification: '+name)
    c.require(c.read(c.QUAL_OUT/'completion.json')['status']=='qualified_verified','Research qualification incomplete')
    gate=c.read(c.QUAL_OUT/'assessment_gate.json');c.require(gate['passed'] is True,'Research resource gate failed')
    c.verify(frozen['inputs']);c.verify(c.read(c.QUAL_OUT/'completed_evidence_seal.json')['artifacts'])
    return dict(status='accepted_research_execution_only',freeze=c.artifact(c.QUAL_OUT/'freeze.json'),
        review=c.artifact(path),inputs=frozen['inputs'],
        artifacts=[c.artifact(p) for p in sorted(c.QUAL_OUT.rglob('*')) if p.is_file()],
        gate=c.artifact(c.QUAL_OUT/'assessment_gate.json'),one_second_admission=False)

def contract():
    return dict(phase=c.PHASE,job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES,max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,max_rss_bytes=c.MAX_RSS_BYTES,
        minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,
        startup_minimum_available_ram_bytes=c.STARTUP_MINIMUM_AVAILABLE_RAM_BYTES,
        initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
        seconds_per_move=c.SECONDS_PER_MOVE,internal_seconds=c.INTERNAL_SECONDS,
        research_seconds_per_move=c.SECONDS_PER_MOVE,research_internal_seconds=c.INTERNAL_SECONDS,
        research_log_max_bytes=128*c.MiB,frozen_log_max_bytes=16*c.MiB,research_response_max_bytes=4*c.MiB,
        frozen_response_max_bytes=256*1024,request_max_bytes=c.MiB,placement_mode='isolated',
        max_workers=c.MAX_WORKERS,max_graph_builds=c.MAX_GRAPH_BUILDS,
        qualification_decisions=194 if c.PHASE=='qualification' else 0,
        operational_games=6 if c.PHASE=='qualification' else 0,main_games=400 if c.PHASE=='assessment' else 0,
        maximum_reproductions=48,automatic_retries=0,failed_decision_retries=0,training=False,
        deployment_target_seconds=1.,timing_thresholds_do_not_gate_admission=True,
        late_completed_moves_accepted=True,
        main_hard_guard_failures='administrative_forfeit_loss_continue_fixed_schedule',
        scientific_infrastructure_resource_failures='stop_incomplete')

def freeze():
    from . import qualification,runner
    c.require(not(c.OUT/'freeze.json').exists(),'Never replace execution freeze')
    design.prepare();source_guard=verify_runtime_sources();old,preserved=preserve_previous()
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial disk reserve')
    test=c.read(c.OUT/'prelaunch/model_free_tests.json')
    c.require(test['passed'] is True and test['exit_code']==0,'Model-free tests not passed')
    review=c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(review['approved'] is True,'Independent implementation review not passed');c.verify(review['inputs'])
    research,baseline=prepare_profiles()
    qitem=design.fixed(c.OUT/'qualification_plan.json',qualification.build_plan()) if c.PHASE=='qualification' else None
    acceptance=accepted_admission() if c.PHASE=='assessment' else None
    design.fixed(c.OUT/'runtime_sources.json',source_guard)
    design.fixed(c.OUT/'seed_inventory.json',seed_inventory())
    pins={}
    def add(item):
        c.require(c.artifact(c.ROOT/item['path'])==item,'Changed frozen input: '+item['path'])
        if item['path'] in pins:c.require(pins[item['path']]==item,'Conflicting frozen identity')
        pins[item['path']]=item
    for item in preserved['prior_source_union']+review['inputs']:add(item)
    for prior in [preserved]+preserved['additional_prior_attempts']:
        for item in prior['prior_attempt_artifacts']:add(item)
    if acceptance:
        for item in acceptance['inputs']+acceptance['artifacts']:add(item)
    for folder in (c.ROOT/'tools',c.ROOT/'src'):
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' not in path.parts:add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_pool6_strengthfirst*20261028*.py')):add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file():add(c.artifact(path))
    for item in list(research.values())+list(baseline.values()):add(item)
    for item in nested_artifacts(qualification.build_plan()):add(item)
    for path in (design.PANEL,c.PROTOCOL,c.QUAL_OUT/'prospective_assessment_schedule.json',
                 c.ROOT/'docs/engineering/future_training_rules_20261021.md'):add(c.artifact(path))
    frozen={key:deepcopy(old[key]) for key in RUNTIME_KEYS};frozen.update(contract())
    frozen.update(schema='splendorust_pool6_strengthfirst_execution_v1',time=c.now(),attempt=1,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        protocol=c.artifact(c.PROTOCOL),study_plan=c.artifact(c.OUT/'study_plan.json'),
        schedule=c.artifact(c.OUT/'schedule.json'),qualification_plan=qitem,reserved_panel=c.artifact(design.PANEL),
        research_profiles=research,baseline_profiles=baseline,accepted_admission=acceptance,
        prior_accepted_attempt=dict(freeze=preserved['prior_freeze'],acceptance=preserved['review'],
            preservation=c.artifact(c.OUT/'prelaunch/prior_attempt_preservation_before.json'),
            acceptance_scope='Verified negative one-second admission; not a research launch gate'),
        candidate_world_pool=6,opponent_world_pool=3,seed_label='A',external_candidate_testing='paused',
        larger_data_deferred=True,native_recompiled=False,native_core_unchanged=True,
        numerical_policies_unchanged=True,execution_deadlines_changed_prospectively=True,original_attempts_unchanged=True,
        inputs=sorted(pins.values(),key=lambda x:x['path']))
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preserve','freeze']);args=parser.parse_args()
    if args.action=='prepare':
        result=design.prepare();research,baseline=prepare_profiles()
        print(c.canonical(dict(result,research_profiles=research,baseline_profiles=baseline)))
    elif args.action=='preserve':
        _,value=preserve_previous();print(c.canonical(dict(status='preserved',prior_sources=len(value['prior_source_union']),
            prior_files=sum(len(x['prior_attempt_artifacts']) for x in [value]+value['additional_prior_attempts']))))
    else:print(c.canonical(freeze()))
