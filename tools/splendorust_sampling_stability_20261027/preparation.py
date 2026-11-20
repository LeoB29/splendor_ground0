"""Bind one additive, bounded sampling attempt without running learned models."""
from copy import deepcopy
import argparse
import shutil
from pathlib import Path
from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
from . import common as c, design

PREVIOUS=design.PREVIOUS
BUILD=c.ROOT/'outputs/splendorust_sampling_stability_build_20261027_01'
RUNTIME_KEYS=('compression_runtime','external_runtime','native_artifacts','numpy_version','placement',
    'sampler_api_modules','source_guard','topology','torch_package_version','torch_runtime',
    'torch_runtime_version','python','python_executable_sha256','unchanged_graph_backend_identity')


def save_fixed(path,value):
    if Path(path).exists():c.require(c.read(path)==value,'Never replace prepared input: '+str(path))
    else:c.save(path,value)
    return c.artifact(path)


def nested_artifacts(value):
    if isinstance(value,dict):
        if {'path','sha256','bytes'}<=set(value):yield {k:value[k] for k in ('path','sha256','bytes')}
        for child in value.values():yield from nested_artifacts(child)
    elif isinstance(value,list):
        for child in value:yield from nested_artifacts(child)


def preserve_previous():
    c.require(c.file_hash(PREVIOUS/'freeze.json')==design.PREVIOUS_FREEZE_SHA,'Prior freeze changed')
    c.require(c.file_hash(PREVIOUS/'postrun/independent_completion_review.json')==design.PREVIOUS_REVIEW_SHA,
        'Prior independent acceptance changed')
    accepted=c.read(PREVIOUS/'postrun/independent_completion_review.json')
    c.require(accepted['accepted'] is True and accepted['freeze_sha256']==design.PREVIOUS_FREEZE_SHA,
        'Prior acceptance disposition/freeze differs')
    c.verify(accepted['consumed_inputs'])
    accepted_inputs={x['path']:x for x in accepted['consumed_inputs']}
    critical=[]
    for name in ('completion.json','analysis.json','completed_evidence_seal.json','verification_summary.json',
                 'guardian_os_exit.json','execution.json','watchdog_execution.json'):
        actual=c.artifact(PREVIOUS/name)
        c.require(accepted_inputs.get(actual['path'])==actual,'Prior result differs from accepted review: '+name)
        critical.append(actual)
    save_fixed(c.OUT/'prelaunch/prior_acceptance_binding.json',dict(
        review=c.artifact(PREVIOUS/'postrun/independent_completion_review.json'),
        verified_accepted_inputs=len(accepted_inputs),critical_reused_results=critical))
    old=c.read(PREVIOUS/'freeze.json')
    c.require(c.read(PREVIOUS/'completion.json')['status']=='completed_verified','Prior completion changed')
    c.verify(old['inputs'])
    path=c.OUT/'prelaunch/prior_attempt_preservation_before.json'
    if path.exists():
        value=c.read(path);c.verify(value['prior_attempt_artifacts'])
        c.require(value['prior_freeze']==c.artifact(PREVIOUS/'freeze.json'),'Prior preservation binding')
    else:
        value=dict(prior_freeze=c.artifact(PREVIOUS/'freeze.json'),source_input_count=len(old['inputs']),
            prior_attempt_artifacts=[c.artifact(p) for p in sorted(PREVIOUS.rglob('*')) if p.is_file()],
            prior_status='completed_verified',failed_attempt01_status='incomplete')
        c.save(path,value)
    c.require(c.read(c.ROOT/'outputs/splendorust_decision_quality_20261027_01/completion.json')['status']=='incomplete',
        'Earlier failed attempt must remain incomplete')
    return old,value


def load_design():
    plan,matrix=design.build_plan(),design.build_matrix()
    c.require(c.read(c.OUT/'study_plan.json')==plan and c.read(c.OUT/'prepared_request_matrix.json')==matrix,
        'Prepared design changed')
    return plan,matrix,c.read(design.PANEL)


def resource_forecast(matrix,qplan):
    previous=c.read(PREVIOUS/'analysis.json')
    times=previous['timing']['64-A']
    # No candidate measurement is inferred. These planning allowances are guarded
    # by real clocks and the fixed six-root measured gate before expansion.
    forecast=dict(schema='sampling_stability_model_free_forecast_v1',status='passed_projection_only',
        previous_analysis=c.artifact(PREVIOUS/'analysis.json'),
        previous_maximum64_seconds=times['maximum'],ordinary_requests=len(matrix['requests']),
        qualification_decisions=qplan['complete_decisions'],qualification_seconds=600,
        diagnostic_seconds=900,finalization_seconds=600,projected_prefinal_bytes=384*c.MiB,
        projected_final_bytes=256*c.MiB,projected_total_bytes=640*c.MiB,
        ordinary_request_deadline_upper_bound_seconds=len(matrix['requests']),
        diagnostic_startup_close_audit_allowance_seconds=576,
        candidate_performance_unqualified=True,no_models_run=True,
        builds_counted_separately_before_freeze=True,
        caveats=['Candidate timing is not measured by this forecast; actual one-second qualification is mandatory.',
            'Fixed six-root/72-request cost-only pilot gates all remaining228 requests.',
            'Real stage/global clocks and logical evidence guards override projections.'])
    c.require(forecast['qualification_seconds']<c.STAGES['qualification'] and
        forecast['diagnostic_seconds']<c.STAGES['diagnostic'] and
        forecast['finalization_seconds']<c.STAGES['finalization'] and
        sum(forecast[k] for k in ('qualification_seconds','diagnostic_seconds','finalization_seconds'))<c.JOB_SECONDS,
        'Planning projection does not fit bounded stages')
    return forecast


def freeze():
    from . import adapters,qualification,runner
    c.require(not(c.OUT/'freeze.json').exists(),'Never replace execution freeze')
    design.prepare();sources=verify_runtime_sources();old,preserved=preserve_previous()
    plan,matrix,panel=load_design()
    c.require(shutil.disk_usage(c.ROOT).free>=7*c.GiB,'Initial disk reserve')
    tests=c.read(c.OUT/'prelaunch/model_free_tests.json')
    c.require(tests['passed'] is True and tests['exit_code']==0,'Model-free test qualification missing')
    review=c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(review['approved'] is True,'Independent implementation review missing')
    c.verify(review['inputs'])
    profiles={}
    for quota,pool in ((16,3),(64,3),(64,6)):
        profiles[f'{quota}-{pool}']=save_fixed(adapters.profile_path(quota,pool),adapters.profile(quota,pool))
    qplan=qualification.build_plan();c.require(qplan['complete_decisions']==129,'Qualification total')
    qitem=save_fixed(c.OUT/'qualification_plan.json',qplan)
    save_fixed(c.OUT/'resource_forecast.json',resource_forecast(matrix,qplan))
    save_fixed(c.OUT/'runtime_sources.json',sources)
    build=c.read(BUILD/'build_receipt.json')
    c.require(build['status']=='compiled_not_model_qualified' and build['copied_core_agents_unchanged'] and
        build['original_sources_modified'] is False and build['model_inference_executed'] is False and
        build['search_executed'] is False,'Model-free additive bridge build identity')
    pins={}
    def add(item):
        c.require(c.artifact(c.ROOT/item['path'])==item,'Input changed: '+item['path'])
        if item['path'] in pins:c.require(pins[item['path']]==item,'Conflicting input identity')
        pins[item['path']]=item
    for item in old['inputs']+preserved['prior_attempt_artifacts']:add(item)
    postbuild=[];cache_changes=[]
    for item in c.read(c.ROOT/build['source_inventory']['path']):
        actual=c.artifact(c.ROOT/item['path'])
        if item!=actual:
            c.require(item['path']==(BUILD/'cargo_home/.global-cache').relative_to(c.ROOT).as_posix(),
                'Build source/dependency changed')
            cache_changes.append(dict(before=item,after=actual,reason='Cargo cache metadata, not executed source'))
        postbuild.append(actual);add(actual)
    save_fixed(c.OUT/'prelaunch/build_freeze_inventory.json',dict(artifacts=postbuild,cache_metadata_changes=cache_changes))
    for item in nested_artifacts(build):add(item)
    add(c.artifact(BUILD/'build_receipt.json'))
    for item in nested_artifacts(qplan):add(item)
    for folder in (c.ROOT/'tools/splendorust_sampling_stability_20261027',c.OUT/'prelaunch',c.OUT/'profiles'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:add(c.artifact(p))
    for p in sorted((c.ROOT/'tests').glob('test_splendorust_sampling_stability_20261027*.py')):add(c.artifact(p))
    for name in ('study_plan.json','prepared_request_matrix.json','qualification_plan.json','resource_forecast.json','runtime_sources.json'):
        add(c.artifact(c.OUT/name))
    for p in (c.PROTOCOL,c.ROOT/'docs/engineering/future_training_rules_20261021.md'):
        add(c.artifact(p))
    seal=save_fixed(c.OUT/'design_seal.json',dict(status='sealed_for_execution',execution_freeze=False,
        artifacts=[c.artifact(c.PROTOCOL),c.artifact(c.OUT/'study_plan.json'),
            c.artifact(c.OUT/'prepared_request_matrix.json'),c.artifact(design.PANEL)]))
    add(seal)
    frozen={k:deepcopy(old[k]) for k in RUNTIME_KEYS}
    frozen.update(schema='splendorust_sampling_stability_execution_v1',time=c.now(),attempt=1,
        prior_accepted_attempt=dict(freeze=c.artifact(PREVIOUS/'freeze.json'),
            acceptance=c.artifact(PREVIOUS/'postrun/independent_completion_review.json'),
            preservation=c.artifact(c.OUT/'prelaunch/prior_attempt_preservation_before.json'),
            preserved_files=len(preserved['prior_attempt_artifacts'])),
        protocol=c.artifact(c.PROTOCOL),prepared_design_seal=seal,
        panel=c.artifact(design.PANEL),request_matrix=c.artifact(c.OUT/'prepared_request_matrix.json'),
        study_plan=c.artifact(c.OUT/'study_plan.json'),qualification_plan=qitem,research_profiles=profiles,
        research_native_sha256=adapters.profile(64,3)['native_binary_sha256'],
        research_native_artifacts=list(build['natives'].values()),
        research_native_roles={item['path']:role for role,item in build['natives'].items()},
        build_receipt=c.artifact(BUILD/'build_receipt.json'),graph_parity_plan=old['graph_parity_plan'],
        graph_parity_fixtures=old['graph_parity_fixtures'],ruleset_id=RULESET_ID,
        catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        resource_forecast=c.artifact(c.OUT/'resource_forecast.json'),
        native_recompiled=True,native_core_unchanged=True,world_pools=[3,6],seed_labels=list(design.LABELS),
        external_candidate_testing='paused',larger_data_deferred=True,
        source_inventory_scope='All prior accepted input and attempt artifacts plus separate sampling implementation/build.',
        inputs=sorted(pins.values(),key=lambda x:x['path']))
    # Contract is declared here, independently checked by runner and review.
    frozen.update(job_seconds=3600,collection_cutoff_seconds=2700,
        stage_seconds=dict(qualification=900,diagnostic=1200,finalization=900),
        max_evidence_bytes=c.GiB,final_reserve_bytes=512*c.MiB,prefinal_evidence_bytes=512*c.MiB,
        max_rss_bytes=2*c.GiB,minimum_available_ram_bytes=2*c.GiB,initial_ram_stability_seconds=60,
        maximum_ram_wait_seconds=300,initial_free_disk_bytes=7*c.GiB,minimum_free_disk_bytes=5*c.GiB,
        seconds_per_move=1.,internal_seconds=.9,research_seconds_per_move=1.,research_internal_seconds=.9,
        research_log_max_bytes=128*c.MiB,frozen_log_max_bytes=16*c.MiB,research_response_max_bytes=4*c.MiB,
        frozen_response_max_bytes=256*1024,request_max_bytes=c.MiB,placement_mode='isolated',
        max_workers=4,max_graph_builds=4,total_requests=324,original_reproductions=24,research_requests=288,
        research_reproductions=12,qualification_decisions=129,automatic_retries=0,failed_decision_retries=0,
        new_games=0,training=False)
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    item=c.save(c.OUT/'freeze.json',frozen)
    return dict(status='frozen_not_launched',freeze=item,inputs=len(pins),model_execution=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preserve','freeze']);args=parser.parse_args()
    if args.action=='preserve':
        _,v=preserve_previous();print(c.canonical(dict(preserved_files=len(v['prior_attempt_artifacts']))))
    else:print(c.canonical(freeze()))
