"""Prospective fixed-corpus policy-head-only freeze and independent admission."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import shutil
import psutil
from . import common as c, design, provenance
from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
from tools.splendorust_search512_20261101.preparation import python_dependencies
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS, nested_artifacts

AUTHORIZATION = c.OUT/'prelaunch/user_authorization.json'
OWNER_SOURCE = c.ROOT/'tools/splendorust_policy_head_owner_20261104.py'

def require_absent(identities):
    return provenance.require_absent(identities)

def require_test_receipt(path):
    value=c.read(path)
    c.require(value['exit_code']==0 and value.get('models',0)==0 and value.get('games',0)==0 and value['inputs'], 'Passed exact-source model-free tests')
    c.verify(value['inputs']+[value['log']])
    return value

def require_review(path):
    value=c.read(path)
    c.require(value['approved'] is True and not value.get('blocking_findings',[]) and value['inputs'], 'Independent exact-source approval')
    c.verify(value['inputs'])
    return value

def authorization():
    value=c.read(AUTHORIZATION)
    c.require(value['authorized'] is True and value['study']=='policy_head_only' and value['new_source_games']==0 and
        value['reference_promoted'] is False and value['larger_data_deferred'] is True and value['automatic_launch_after_checks'] is True,
        'Explicit authorization of one policy-head-only study')
    return c.artifact(AUTHORIZATION)

def verify_reused_corpus(freeze,boundary=lambda:None):
    return provenance.verify(freeze,boundary)

def verify_seed_receipt(value):
    from .assessment import design as assessment_design
    c.require(value['status']=='passed' and not value['collisions'] and
        value['domain']==assessment_design.DOMAIN and value['bootstrap_seed']==assessment_design.BOOTSTRAP_SEED and
        value['new_setup_count']==203 and value['prior_seed_reuse'] is False and
        value['explicitly_disjoint_previous_pilot']==c.artifact(assessment_design.PRIOR_ASSESSMENT_SCHEDULE),
        'Exact fresh policy-head assessment seed domain and previous-pilot exclusion')
    c.verify(value['checked'])
    return value

def prepare():
    from . import qualification_control as control,training,build_capture
    from .assessment import control as assessment,design as assessment_design
    proof=provenance.prepare()
    source=c.read(c.SOURCE_OUT/'freeze.json')
    auth=authorization()
    source_plan=c.read(c.ROOT/proof['source_fit_plan']['path'])
    training_plan=training.pilot_plan()
    build=build_capture.validate_build(c.ROOT/source['native_capture_build']['path'])
    c.require(build['binary']==source['native_capture_binary'],'Same accepted search bridge')
    prepared={key:deepcopy(source[key]) for key in (
        'profiles','native_capture_build','native_capture_binary','native_artifacts_by_route','native_roles_by_route',
        'learned_entrypoint','historical_parent_checkpoint','unchanged_model_definition','external_runtime_inputs')}
    prepared.update(user_authorization=auth,source_reuse=proof['declaration'],
        reused_corpus=proof['declaration']['corpus_manifest'],source_specs=proof['declaration']['source_specs'],
        source_reuse_proof=c.artifact(provenance.PROOF),source_guard=verify_runtime_sources(),
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        original_checkpoint_reference=True,larger_data_deferred=True,already_opened_heldout_sets_are_descriptive=True)
    prepared['training_plan']=design.fixed(c.OUT/'training_plan.json',training_plan)
    prepared['qualification_plan']=design.fixed(c.OUT/'qualification_plan.json',control.build_plan())
    prepared['resource_plan']=design.fixed(c.OUT/'resource_plan.json',design.resource_plan())
    prepared['fit_plan']=design.fixed(c.OUT/'fit_plan.json',dict(training_plan=training_plan,
        source_specs=design.schedule(),learned_entrypoint=prepared['learned_entrypoint'],
        input_qualification=deepcopy(source_plan['input_qualification'])))
    prepared['fit_parity_fixtures']=design.fixed(c.OUT/'fit_parity_fixtures.json',
        c.read(c.ROOT/source['fit_parity_fixtures']['path']))
    prepared['fit_direct_forward_budget']=training.direct_forward_budget(training_plan)
    prepared['assessment_prepared']=assessment.prepare()
    prepared['assessment_plan']=prepared['assessment_prepared']['assessment_plan']
    seed_path=c.OUT/'assessment_seed_freshness.json'
    if not seed_path.exists():c.save(seed_path,assessment_design.seed_inventory())
    seed_proof=c.read(seed_path)
    verify_seed_receipt(seed_proof)
    prepared['assessment_seed_freshness']=c.artifact(seed_path)
    c.verify_external_runtime(prepared['external_runtime_inputs'])
    c.require('torch' not in sys.modules,'Preparation executes no model')
    return source,prepared

def freeze(test_path,review_paths):
    from . import qualification_control as control
    c.require(not (c.OUT/'freeze.json').exists(),'One separate prospective freeze')
    require_test_receipt(test_path)
    c.require(len(review_paths)>=3,'Independent training, assessment and owner reviews')
    for path in review_paths:require_review(path)
    source,prepared=prepare()
    value={key:deepcopy(source[key]) for key in RUNTIME_KEYS}
    value.update(control.contract());value.update(prepared)
    value.update(schema=c.SCHEMA,scope='pilot',phase='pilot',time=c.now(),protocol=c.artifact(c.PROTOCOL),
        purpose='policy_head_only_fixed_corpus_ablation',model_free_tests=c.artifact(test_path),
        implementation_reviews=[c.artifact(p) for p in review_paths],
        automatic_retries=0,scheduled_monitor=False,reference_promoted=False,
        fixed_precision='float32',original_architecture=True,original_weights_preserved=True,
        scientific_intervention='trainable_parameter_scope_only',new_games=406,
        conditional_workflow='corrected input and disposable optimizer-scope qualification -> fixed policy-head fit -> six operational games -> 400 fresh paired games -> all checks and independent acceptance')
    sources=python_dependencies((c.ROOT/'tools/splendorust_policy_head_20261104').rglob('*.py'))
    tests=sorted((c.ROOT/'tests').glob('test_splendorust_policy_head*20261104*.py'))
    entries=[c.artifact(p) for p in sources+tests+[OWNER_SOURCE]]
    entries+=list(nested_artifacts(value))
    entries+=list(nested_artifacts(c.read(c.ROOT/prepared['source_reuse_proof']['path'])))
    entries+=list(nested_artifacts(c.read(c.ROOT/prepared['assessment_plan']['path'])))
    entries+=list(nested_artifacts(c.read(c.ROOT/prepared['native_capture_build']['path'])))
    entries+=list(nested_artifacts(c.read(c.ROOT/prepared['assessment_seed_freshness']['path'])))
    for path in [test_path,*review_paths]:entries+=list(nested_artifacts(c.read(path)))
    unique={}
    for item in entries:
        if Path(item['path']).is_absolute():
            c.require(item in prepared['external_runtime_inputs'],'Only inherited external runtime descriptors')
            continue
        previous=unique.setdefault(item['path'],item)
        c.require(previous==item,'Conflicting frozen identity')
    value['inputs']=sorted(unique.values(),key=lambda x:x['path'])
    c.verify(value['inputs']);c.require('torch' not in sys.modules,'No model at freeze')
    return c.save(c.OUT/'freeze.json',value)

def validate_launch(expected_sha):
    from . import qualification_control as control,training,data
    from .assessment import control as assessment,design as assessment_design
    from tools.splendorust_search256_recovery_20261030_02 import placement
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact prospective study freeze')
    for name in ('guardian_claim.json','launch.json','failure.json','completion.json','prelaunch_review.json'):
        c.require(not (c.OUT/name).exists(),'No duplicate or implicit retry')
    value=c.read(c.OUT/'freeze.json');placement.configure_execution(value,'isolated')
    control.verify_runtime(value)
    c.require(value['user_authorization']==authorization(),'Exact authorized study')
    require_test_receipt(c.ROOT/value['model_free_tests']['path'])
    for item in value['implementation_reviews']:require_review(c.ROOT/item['path'])
    reused=verify_reused_corpus(value)
    data.validate_accepted_corpus_reuse(reused['manifest'],reused['source_specs'],value['source_reuse'])
    c.require(c.read(c.ROOT/value['training_plan']['path'])==training.pilot_plan(),'Fixed trainable-scope recipe')
    c.require(c.read(c.ROOT/value['qualification_plan']['path'])==control.build_plan(),'Fixed root qualification')
    c.require(c.read(c.ROOT/value['resource_plan']['path'])==design.resource_plan(),'Advisory resource plan')
    c.require(c.read(c.ROOT/value['assessment_plan']['path'])==assessment.build_plan(),'Fresh paired assessment plan fixed before training')
    c.require(c.read(c.ROOT/value['assessment_prepared']['assessment_game_schedule']['path'])==assessment.schedule(),'Fixed prospective assessment schedule')
    verify_seed_receipt(c.read(c.ROOT/value['assessment_seed_freshness']['path']))
    c.require(verify_runtime_sources()==value['source_guard'],'Corrected runtime identity')
    c.require(shutil.disk_usage(c.ROOT).free>=c.MINIMUM_FREE_DISK_BYTES,'Actual free disk floor')
    c.require('torch' not in sys.modules,'Independent admission contains no model')
    return c.save(c.OUT/'prelaunch_review.json',dict(approved=True,status='exact_scope_freeze_admitted',
        freeze_sha256=expected_sha,scope='pilot',time=c.now(),models=0,games=0,blocking_findings=[],
        inputs=value['inputs']+[c.artifact(c.OUT/'freeze.json')],no_retries=True,
        available_ram_bytes=psutil.virtual_memory().available,ram_reserve_policy='advisory_only',
        new_source_games=0,reused_corrected_source_games=256,original_reference=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','freeze','validate-launch'])
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*',default=[]);parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.operation=='prepare':result=prepare()[1]
    elif args.operation=='freeze':
        c.require(args.tests is not None,'Test receipt supplied')
        result=freeze(Path(args.tests),[Path(x) for x in args.reviews])
    else:result=validate_launch(args.freeze_sha256)
    print(c.canonical(result));return 0

if __name__=='__main__':raise SystemExit(main())
