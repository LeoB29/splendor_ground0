"""Model-free scoped preparation, exclusive freezes and conditional launch checks."""
import argparse
from copy import deepcopy
import sys
import shutil
import time
from pathlib import Path
import psutil
from . import common as c, design
from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
from tools.splendorust_search512_20261101.preparation import python_dependencies
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS, nested_artifacts

def accepted_parent():
    c.require(c.file_hash(c.ACCEPTED_OUT/'freeze.json') == c.ACCEPTED_FREEZE_SHA,
              'Exact accepted512 source freeze')
    c.require(c.file_hash(c.ACCEPTED_OUT/'independent_review.json') == c.ACCEPTED_REVIEW_SHA,
              'Accepted512 independent approval preserved')
    reader = c.read(c.ACCEPTED_OUT/'independent_review.json')
    closeout = c.read(c.ACCEPTED_CLOSEOUT)
    c.require(reader['approved'] and closeout['status'] == 'accepted_completed_verified'
              and closeout['guardian_exit_observation']['exit_code'] == 0,
              'Accepted512 full external completion')
    for name in ('execution.json','watchdog_execution.json','independent_review_os_exit.json'):
        value = c.read(c.ACCEPTED_OUT/name)
        c.require(value['status'] == 'completed' and value['exit_code'] == 0
                  and not value['cleanup_errors'], 'Actual successful predecessor owner exits')
    for item in closeout['exact_owner_identities_absent']:
        try:
            process = psutil.Process(item['pid'])
        except psutil.NoSuchProcess:
            continue
        c.require(abs(process.create_time()-item['created']) > .01, 'Predecessor exact owner still alive')
    return c.read(c.ACCEPTED_OUT/'freeze.json')


def require_absent(identities):
    for item in identities:
        try:
            process = psutil.Process(item['pid'])
        except psutil.NoSuchProcess:
            continue
        c.require(process.create_time() != item['created'], 'Previously owned exact process remains alive')


def accepted_engineering():
    """Recovered verification is provenance, not a silently passed old attempt."""
    c.require(c.file_hash(c.ENGINEERING_OUT/'freeze.json') == c.ENGINEERING_FREEZE_SHA
              and c.file_hash(c.RECOVERED_ACCEPTANCE) == c.RECOVERED_ACCEPTANCE_SHA
              and c.file_hash(c.RECOVERED_INDEPENDENT_ACCEPTANCE) == c.RECOVERED_INDEPENDENT_ACCEPTANCE_SHA,
              'Exact recovered engineering acceptance')
    closeout = c.read(c.RECOVERED_ACCEPTANCE)
    independent = c.read(c.RECOVERED_INDEPENDENT_ACCEPTANCE)
    c.require(closeout['status'] == 'accepted_recovered_saved_verification'
              and independent['approved'] and independent['actual_outer_exit_code'] == 0
              and closeout['original_attempt_status'] == 'incomplete'
              and closeout['first_audit_status'] == 'incomplete'
              and not closeout['original_final_stage_passed'] and not closeout['pilot_started']
              and closeout['coverage'] == independent['coverage'], 'Recovered engineering-only evidence')
    c.verify(independent['inputs'])
    require_absent(independent['exact_current_owner_and_readers_absent'])
    coverage = closeout['coverage']
    c.require(coverage['completed_games'] == 4 and coverage['exact_reproductions'] == 12
              and coverage['scientific_qualification_decisions'] == 201 and coverage['graph_cases'] == 33
              and coverage['qualification_only'] and not coverage['pilot_data']
              and coverage['fit']['learned_entrypoint']['same_checkpoint_exact_pairs'] == 3,
              'Qualified actual capture, fit, reload and player chain')
    return dict(original_freeze=c.artifact(c.ENGINEERING_OUT/'freeze.json'),
                recovered_acceptance=c.artifact(c.RECOVERED_ACCEPTANCE),
                independent_acceptance=c.artifact(c.RECOVERED_INDEPENDENT_ACCEPTANCE),
                original_attempt_remains_incomplete=True,
                new_clock_path_qualified=False,
                role='Historical scientific engineering provenance; not certification of the new clock path')


def failed_attempt_provenance():
    descriptor = c.artifact(c.FAILED_PRESERVATION)
    preservation = c.verify_failed_attempt_preservation(descriptor)
    require_absent(preservation['exact_owner_identities_absent'])
    c.require(c.file_hash(c.CLOCK_DIAGNOSIS) == c.CLOCK_DIAGNOSIS_SHA,
              'Exact preserved mixed-clock diagnosis')
    review = require_review(c.CLOCK_DIAGNOSIS_REVIEW)
    c.require(review['original_status'] == 'incomplete' and not review['qualification_accepted']
              and not review['pilot_started'] and not review['failed_parent_observation_available'],
              'Diagnosis never retrospectively accepts the failed response')
    return dict(inventory=descriptor, diagnosis=c.artifact(c.CLOCK_DIAGNOSIS),
                independent_review=c.artifact(c.CLOCK_DIAGNOSIS_REVIEW),
                original_attempt_remains_incomplete=True, training_examples_reused=0,
                checkpoints_reused=0)


def accepted_new_entrypoint():
    """Authenticate the original qualified science and separately verified prefix."""
    from . import continuation
    from tools.splendorust_selfplay_pilot_budget_20261102 import preparation as original
    c.require(c.SCOPE == 'pilot' and c.OUT == continuation.NEW_OUT, 'Separate continuation-only admission')
    qualified = original.accepted_new_entrypoint()
    c.require(qualified == continuation.old_freeze()['new_entrypoint_preflight'],
              'Unchanged accepted entrypoint; no failed qualification or replacement weights')
    return dict(original_qualified_entrypoint=qualified,
                continuation_manifest=c.artifact(continuation.MANIFEST),
                saved_prefix_review=c.artifact(continuation.PREFIX_REVIEW),
                qualification_games_reused=0, qualification_checkpoint_reused=False,
                preserved_pilot_games=48, remaining_source_games=208,
                new_continuation_requires_exact_source_review=True)

def authorized_windows():
    """Bind the explicit amended windows; available RAM and forecasts stay advisory."""
    path=c.OUT/'prelaunch/user_authorization.json';value=c.read(path)
    expected=dict(job_seconds=c.JOB_SECONDS,model_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        outer_seconds=124*3600,collection_seconds=c.STAGES['collection'],fit_seconds=c.STAGES['fit'],
        assessment_seconds=c.STAGES['assessment'],reproduction_seconds=c.STAGES['reproduction'],
        final_seconds=c.STAGES['final'])
    c.require(value['schema']=='selfplay_pilot_resume_authorization_v1' and value['resume_authorized'] is True and
        value['resource_forecasts_advisory'] is True and value['science_unchanged'] is True and
        value['actual_original_clock_origins_retained'] is True and value['amended_clock_windows'] is True and
        value['failed_decision_retries']==0 and value['original_completed_games_preserved']==48 and
        all(value.get(key)==span for key,span in expected.items()),'Exact authorized48-game continuation and amended windows')
    return dict(user_authorization=c.artifact(path),authorized_clock_windows=expected)


def require_test_receipt(path):
    value = c.read(path)
    c.require(value['exit_code'] == 0 and value['models'] == value['games'] == 0
              and value['inputs'], 'Passed model-free tests on exact source')
    c.verify(value['inputs']+[value['log']])
    return value

def require_review(path):
    value = c.read(path)
    c.require(value['approved'] is True and not value.get('blocking_findings', [])
              and value['inputs'], 'Independent exact-source review without blocking findings')
    c.verify(value['inputs'])
    return value

def prepare(native_build_path=None):
    from . import qualification_control as control, training, build_capture
    from tools.external_strength_screen_20261018 import rules
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from .assessment import control as assessment
    from . import continuation
    from . import forecast_preservation
    c.require(c.SCOPE == 'pilot' and c.OUT == continuation.NEW_OUT, 'New pilot continuation output only')
    authorization=authorized_windows()
    manifest = continuation.prepare_continuation()
    continuation.verify_prefix_review(c.artifact(continuation.PREFIX_REVIEW))
    execution_path=c.OUT/'prelaunch/saved_prefix_review_execution_01.json'
    execution=c.read(execution_path)
    c.require(execution['exit_code']==0 and execution['error'] is None and execution['source_unchanged'] is True and
              execution['models']==execution['new_games']==0 and execution['seconds']<=1210,
              'Actual successful bounded saved-prefix reviewer exit required')
    c.verify(execution['inputs']+[execution['log']])
    prior = accepted_parent()
    engineering = accepted_engineering()
    failure = failed_attempt_provenance()
    preflight = accepted_new_entrypoint() if c.SCOPE == 'pilot' else None
    sources = verify_runtime_sources()
    assessment_prepared = assessment.prepare()
    assessment_specs = assessment.schedule()
    prepared = design.prepare(extra_seeds=[x['seed_decimal'] for x in assessment_specs])
    prepared.update(authorization)
    prepared['assessment_prepared'] = assessment_prepared
    prepared['assessment_plan'] = assessment_prepared['assessment_plan']
    prepared['recovered_engineering_qualification'] = engineering
    prepared['new_entrypoint_preflight'] = preflight
    prepared['continuation'] = c.artifact(continuation.MANIFEST)
    prepared['source_continuation'] = manifest['source_continuation']
    prepared['saved_prefix_review'] = c.artifact(continuation.PREFIX_REVIEW)
    prepared['saved_prefix_review_execution'] = c.artifact(execution_path)
    prepared['original_forecasts_review'] = forecast_preservation.prepare_saved_forecasts()
    prepared['failed_attempt_preservation'] = failure['inventory']
    prepared['failed_attempt_provenance'] = failure
    prepared['qualification_plan'] = design.fixed(c.OUT/'qualification_plan.json', control.build_plan())
    training_plan = training.qualification_plan() if c.SCOPE == 'qualification' else training.pilot_plan()
    prepared['training_plan'] = design.fixed(c.OUT/'training_plan.json', training_plan)
    prepared['resource_plan'] = design.fixed(c.OUT/'resource_plan.json', design.resource_plan())
    profiles = deepcopy(prior['profiles']['candidate'])
    c.require(set(profiles) == {'16','512'}, 'Original512 and reference16 profiles')
    c.verify(list(profiles.values()))
    prepared['profiles'] = {'candidate': profiles}
    native_build_path = c.ENGINEERING_OUT/'native/build_receipt.json' if native_build_path is None else Path(native_build_path)
    build = build_capture.validate_build(native_build_path)
    c.require(build['binary']['sha256'] == '2fb73d91b75cfcf4fdfcbc3505a3954ea6a562f94a8363a269f2f170b0575218',
              'Exact already qualified capture bridge, no rebuild or search changes')
    prepared['native_capture_build'] = c.artifact(native_build_path)
    prepared['native_capture_binary'] = build['binary']
    reference = prior['native_artifacts_by_route']['candidate'][0]
    prepared['native_artifacts_by_route'] = {'candidate': [reference, build['binary']]}
    prepared['native_roles_by_route'] = {'candidate': {reference['path']:'reference', build['binary']['path']:'search'}}
    prepared['accepted_fixture_game_costs'] = [
        c.artifact(c.ACCEPTED_OUT/'games/operational'/f'g{i:03d}'/'cost.json') for i in range(6)]
    candidate_profile = c.ROOT/'outputs/splendorust_search128_strength_20261028_01/profiles/puct128.json'
    c.require(c.file_hash(candidate_profile) == '048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb',
              'Original128 search-settings template for learned-model qualification')
    case_ids = ['face_T1_05','private_reserves','gold_returns']
    parent_plan = c.read(c.ACCEPTED_OUT/'qualification_plan.json')['actual_suites']['512']
    cases = {x['id']:x for x in parent_plan['counterexamples']}
    prepared['learned_entrypoint'] = dict(case_ids=case_ids, profile=c.artifact(candidate_profile),
        native_artifact=build['binary'], observers=[dict(id=name,
            observer=rules.observation_payload(decode_training_snapshot(cases[name]['snapshot']))) for name in case_ids])
    prepared['fit_plan'] = design.fixed(c.OUT/'fit_plan.json', dict(
        training_plan=training_plan, source_specs=design.schedule(),
        learned_entrypoint=prepared['learned_entrypoint'],
        input_qualification=deepcopy(c.read(c.ROOT/continuation.old_freeze()['fit_plan']['path'])['input_qualification'])))
    prepared['fit_direct_forward_budget'] = training.direct_forward_budget(training_plan)
    prepared['fit_parity_fixtures'] = design.fixed(c.OUT/'fit_parity_fixtures.json',
        dict(fixtures=deepcopy(prior['graph_parity_fixtures'])))
    prepared['source_guard'] = sources
    prepared['historical_parent_checkpoint'] = c.artifact(training.PARENT_CHECKPOINT)
    prepared['unchanged_model_definition'] = c.artifact(training.MODEL_SOURCE)
    c.require(prepared['historical_parent_checkpoint']['sha256'] == c.PARENT_CHECKPOINT_SHA
              and prepared['unchanged_model_definition']['sha256'] == training.MODEL_SOURCE_SHA256,
              'Explicit original weights and architecture inputs')
    prepared['external_runtime_inputs'] = c.verify_external_runtime(
        [x for x in prior['inputs'] if Path(x['path']).is_absolute()])
    c.require('torch' not in sys.modules, 'Freeze preparation must remain model-free')
    prepared['prelaunch_resource_projection'] = design.fixed(c.OUT/'prelaunch_resource_projection.json',
        provisional_projection(prepared))
    return prior, prepared


def provisional_projection(prepared):
    from . import continuation
    manifest = c.read(continuation.MANIFEST)
    path = c.OUT/'prelaunch_resource_projection.json'
    if path.exists():
        snapshot = c.read(path)['snapshot']
    else:
        stage = manifest['continuation_collection']
        snapshot = dict(monotonic_now=time.monotonic(),unix_now=time.time(),
            original_monotonic_origin=manifest['original_monotonic_origin'],
            original_unix_origin=manifest['original_unix_origin'],
            stage_monotonic_deadline=stage['monotonic_deadline'],stage_unix_deadline=stage['unix_deadline'],
            accounted_bytes=manifest['prior_evidence_bytes']+128*c.MiB,
            cumulative_ram_wait_seconds=manifest['prior_ram_wait_seconds'],completed_games=48,qualified_source_workers=7)
    c.require(snapshot['original_monotonic_origin']==manifest['original_monotonic_origin'] and
              snapshot['original_unix_origin']==manifest['original_unix_origin'] and
              snapshot['stage_monotonic_deadline']==manifest['continuation_collection']['monotonic_deadline'] and
              snapshot['stage_unix_deadline']==manifest['continuation_collection']['unix_deadline'] and
              snapshot['completed_games']==48 and snapshot['qualified_source_workers']==7,
              'Continuation forecast retains original clocks and exact completed prefix')
    costs=[c.read(continuation.prefix_game_path(i).parent/'cost.json') for i in range(48)]
    value=design.source_forecast(prepared,snapshot,source_costs=costs)
    value.update(forecast_advisory=True,forecast_can_stop=False)
    return value

def freeze(native_build_path, test_path, review_paths):
    from . import qualification_control as control
    c.require(not (c.OUT/'freeze.json').exists(), 'Exclusive scoped freeze; never replace evidence')
    prior, prepared = prepare(native_build_path)
    require_test_receipt(test_path)
    c.require(len(review_paths) >= 3, 'Independent capture, data/training and owner reviews')
    for path in review_paths:
        require_review(path)
    value = {key:deepcopy(prior[key]) for key in RUNTIME_KEYS}
    value.update(control.contract())
    value.update(prepared)
    value.update(schema=c.SCHEMA, phase=c.PHASE, time=c.now(),
                 protocol=c.artifact(c.PROTOCOL), ruleset_id=RULESET_ID,
                 catalogue_sha256=CATALOGUE_SHA256, turn_contract=TURN_CONTRACT,
                 accepted_parent_freeze=c.artifact(c.ACCEPTED_OUT/'freeze.json'),
                 accepted_parent_review=c.artifact(c.ACCEPTED_OUT/'independent_review.json'),
                 accepted_parent_closeout=c.artifact(c.ACCEPTED_CLOSEOUT),
                 model_free_tests=c.artifact(test_path),
                 implementation_reviews=[c.artifact(p) for p in review_paths],
                 scope=c.SCOPE, purpose='engineering_qualification_only' if c.SCOPE == 'qualification' else 'single_generation_learning_pilot', teacher_simulations=512,
                 original_weights_preserved=True, original_architecture=True,
                 fixed_precision='float32', automatic_retries=0, scheduled_monitor=False,
                 new_source_games=c.MAX_SOURCE_GAMES-48, new_assessment_games=c.MAX_ASSESSMENT_GAMES,
                 new_games=c.MAX_TOTAL_GAMES-48, optimizer_step_maximum=c.MAX_OPTIMIZER_STEPS,
                 training_sample_reuse=False, reference_promoted=False,
                 conditional_workflow='verified original32 plus previous16-game prefix -> original schedule continuation -> complete original assessment and independent verification',
                 authorization='User explicitly requested advisory forecasts and immediate separate continuation',
                 forecast_advisory=True,forecast_can_stop=False,completed_source_games_preserved=48,clock_windows_amended=True,
                 original_attempt_remains_incomplete=True)
    sources = python_dependencies((c.ROOT/'tools/splendorust_selfplay_pilot_resume2_20261103').rglob('*.py'))
    tests = sorted((c.ROOT/'tests').glob('test_splendorust_selfplay*resume2*20261103*.py'))
    entries = [c.artifact(p) for p in sources+tests]
    entries += list(nested_artifacts(value))
    entries += list(nested_artifacts(c.read(c.ROOT/prepared['native_capture_build']['path'])))
    entries += c.read(c.FAILED_PRESERVATION)['inputs']
    entries += list(nested_artifacts(c.read(c.CLOCK_DIAGNOSIS_REVIEW)))
    for key in ('continuation','saved_prefix_review','saved_prefix_review_execution','original_forecasts_review'):
        entries += list(nested_artifacts(c.read(c.ROOT/prepared[key]['path'])))
    for receipt_path in [test_path, *review_paths]:
        entries += list(nested_artifacts(c.read(receipt_path)))
    unique = {}
    for item in entries:
        if Path(item['path']).is_absolute():
            c.require(item in prepared['external_runtime_inputs'],
                      'Absolute artifact must belong to authenticated runtime inventory')
            continue
        if item['path'] in unique:
            c.require(unique[item['path']] == item, 'Conflicting frozen input identity')
        unique[item['path']] = item
    value['inputs'] = sorted(unique.values(), key=lambda x:x['path'])
    c.verify(value['inputs'])
    c.require('torch' not in sys.modules, 'No learned-model execution during freeze')
    c.save(c.OUT/'freeze.json', value)
    return c.artifact(c.OUT/'freeze.json')


def validate_launch(expected_sha):
    """Independent model-free reader for the prospectively approved transition."""
    from . import qualification_control as control
    from .assessment import control as assessment
    from tools.splendorust_search256_recovery_20261030_02 import placement
    c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Exact supplied prospective freeze')
    value = c.read(c.OUT/'freeze.json')
    c.require(value['scope'] == c.SCOPE and value['phase'] == c.SCOPE and value['schema'] == c.SCHEMA,
              'Environment, scope and frozen entrypoint agree')
    c.require(not (c.OUT/'guardian_claim.json').exists() and not (c.OUT/'launch.json').exists(),
              'No duplicate or resumed job through automatic admission')
    placement.configure_execution(value, 'isolated')
    control.verify_runtime(value)
    require_test_receipt(c.ROOT/value['model_free_tests']['path'])
    for item in value['implementation_reviews']:
        require_review(c.ROOT/item['path'])
    accepted_parent(); accepted_engineering()
    c.require(failed_attempt_provenance() == value['failed_attempt_provenance'],
              'Same preserved failure, diagnosis and independent review')
    if c.SCOPE == 'pilot':
        c.require(accepted_new_entrypoint() == value['new_entrypoint_preflight'], 'Same accepted new-route preflight')
    c.require(all(value[key]==item for key,item in authorized_windows().items()),'Same explicit amended clock authorization')
    c.require(c.read(c.ROOT/value['resource_plan']['path']) == design.resource_plan(), 'Same prospectively measured prices')
    from . import continuation
    from . import forecast_preservation
    continuation.verify_continuation(value['continuation'])
    continuation.verify_prefix_review(value['saved_prefix_review'])
    forecast_preservation.verify_saved_forecasts(value['original_forecasts_review'])
    c.require(value['source_continuation']==continuation.source_continuation(), 'Exact preserved source-prefix declaration')
    projection = provisional_projection(value)
    c.require(projection == c.read(c.ROOT/value['prelaunch_resource_projection']['path']) and
              projection['forecast_advisory'] and not projection['forecast_can_stop'],
              'Authentic advisory forecast; hard live resource bounds remain enforced')
    c.require(c.read(c.ROOT/value['game_schedule']['path']) == design.schedule()
              and c.read(c.ROOT/value['initial_states']['path']) == design.initial_states(), 'Exact fresh source schedule/setup')
    c.require(c.read(c.ROOT/value['assessment_plan']['path']) == assessment.build_plan(),
              'Exact fixed learned-versus-parent assessment plan')
    c.require(c.read(c.ROOT/value['assessment_prepared']['assessment_game_schedule']['path']) == assessment.schedule(),
              'Exact separate paired assessment schedule')
    c.require(verify_runtime_sources() == value['source_guard'], 'Corrected runtime source identity')
    available = psutil.virtual_memory().available
    free = shutil.disk_usage(c.ROOT).free
    c.require(available >= c.INITIAL_AVAILABLE_RAM_BYTES and free >= c.INITIAL_FREE_DISK_BYTES,
              'Actual retained disk requirement; available host RAM is descriptive without reserve waits')
    c.require('torch' not in sys.modules, 'No model runtime in conditional launch validator')
    return c.save(c.OUT/'prelaunch_review.json', dict(approved=True, status='exact_scope_freeze_admitted',
        freeze_sha256=expected_sha, scope=c.SCOPE, time=c.now(), reviewer='independent model-free conditional-admission process',
        inputs=value['inputs']+[c.artifact(c.OUT/'freeze.json')], blocking_findings=[],
        available_ram_bytes=available, free_disk_bytes=free, resource_projection=projection,
        actual_new_entrypoint_required=c.SCOPE == 'pilot', models=0, games=0,
        scientific_changes_after_preflight=False, separately_reviewed_continuation_control=True,
        no_model_or_game_retries=True))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare','freeze','validate-launch'])
    parser.add_argument('--native-build')
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--tests')
    parser.add_argument('--reviews', nargs='*', default=[])
    args = parser.parse_args()
    if args.operation == 'prepare':
        result = prepare(None if args.native_build is None else Path(args.native_build))[1]
    elif args.operation == 'freeze':
        c.require(args.tests is not None, 'Frozen model-free test receipt required')
        result = freeze(None if args.native_build is None else Path(args.native_build), Path(args.tests), [Path(x) for x in args.reviews])
    else:
        c.require(args.freeze_sha256 is not None, 'Exact intended freeze SHA required')
        result = validate_launch(args.freeze_sha256)
    print(c.canonical(result))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
