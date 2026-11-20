"""Model-free scoped preparation, exclusive freezes and conditional launch checks."""
import argparse
from copy import deepcopy
import sys
import shutil
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
                original_attempt_remains_incomplete=True)


def accepted_new_entrypoint():
    """The same new source must finish its separately frozen preflight first."""
    path = c.QUALIFICATION_OUT / 'qualification_acceptance.json'
    value = c.read(path)
    c.require(value['status'] == 'accepted_completed_verified' and value['scope'] == 'qualification'
              and value['guardian_exit_observation']['exit_code'] == 0
              and value['completed_source_games'] == 4 and value['assessment_operational_games'] == 6
              and value['no_pilot_data'] is True, 'Actual new-entrypoint preflight accepted before pilot')
    c.verify([value['freeze'], value['independent_review']])
    qfreeze = c.read(c.ROOT/value['freeze']['path'])
    review = c.read(c.ROOT/value['independent_review']['path'])
    c.require(qfreeze['phase'] == 'qualification' and qfreeze['schema'] == c.SCHEMA
              and review['approved'] and review['freeze_sha256'] == value['freeze']['sha256'],
              'New preflight freeze and independent approval bind')
    c.verify(qfreeze['inputs'])
    for name in ('execution.json', 'watchdog_execution.json', 'independent_review_os_exit.json'):
        item = c.read(c.QUALIFICATION_OUT/name)
        c.require(item['exit_code'] == 0 and item['status'] == 'completed'
                  and not item['cleanup_errors'], 'Actual successful new preflight OS exits')
    require_absent(value['exact_owner_identities_absent'])
    return dict(acceptance=c.artifact(path), freeze=value['freeze'], independent_review=value['independent_review'],
                reused_training_games=0, reused_training_examples=0, new_source_required=True)

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
    prior = accepted_parent()
    engineering = accepted_engineering()
    preflight = accepted_new_entrypoint() if c.SCOPE == 'pilot' else None
    sources = verify_runtime_sources()
    assessment_prepared = assessment.prepare()
    assessment_specs = assessment.schedule()
    prepared = design.prepare(extra_seeds=[x['seed_decimal'] for x in assessment_specs])
    prepared['assessment_prepared'] = assessment_prepared
    prepared['assessment_plan'] = assessment_prepared['assessment_plan']
    prepared['recovered_engineering_qualification'] = engineering
    prepared['new_entrypoint_preflight'] = preflight
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
        input_qualification=dict(source='current_frozen_job', plan=prepared['qualification_plan'],
            row_paths=[(c.OUT/'capture_fixtures/capture_on'/f'{i:03d}.json').relative_to(c.ROOT).as_posix() for i in range(16)],
            proof_path=(c.OUT/'capture_qualification.json').relative_to(c.ROOT).as_posix())))
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
    c.require(c.read(c.ROOT/prepared['prelaunch_resource_projection']['path'])['affordable'],
              'Entire fixed source/fit/assessment/finalization pipeline prospectively affordable')
    return prior, prepared


def provisional_projection(prepared):
    # Qualification is rechecked on the actual clock before collection. This
    # model-free admission assumes six minutes for its established startup path.
    elapsed = 360.
    snapshot = dict(monotonic_now=elapsed, unix_now=elapsed, original_monotonic_origin=0.,
        original_unix_origin=0., stage_monotonic_deadline=elapsed+c.STAGES['collection'],
        stage_unix_deadline=elapsed+c.STAGES['collection'], accounted_bytes=128*c.MiB,
        cumulative_ram_wait_seconds=60., completed_games=0, qualified_source_workers=2)
    return design.source_forecast(prepared, snapshot)

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
                 new_source_games=c.MAX_SOURCE_GAMES, new_assessment_games=c.MAX_ASSESSMENT_GAMES,
                 new_games=c.MAX_TOTAL_GAMES, optimizer_step_maximum=c.MAX_OPTIMIZER_STEPS,
                 training_sample_reuse=False, reference_promoted=False,
                 conditional_workflow='new preflight accepted -> fixed pilot freeze -> independent exact admission -> launch; stop on failure',
                 authorization='User approved full512 teacher pilot with automatic launch if normal qualification/resources pass')
    sources = python_dependencies((c.ROOT/'tools/splendorust_selfplay_pilot_20261102').rglob('*.py'))
    tests = sorted((c.ROOT/'tests').glob('test_splendorust_selfplay_pilot*20261102*.py'))
    entries = [c.artifact(p) for p in sources+tests]
    entries += list(nested_artifacts(value))
    entries += list(nested_artifacts(c.read(c.ROOT/prepared['native_capture_build']['path'])))
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
    if c.SCOPE == 'pilot':
        c.require(accepted_new_entrypoint() == value['new_entrypoint_preflight'], 'Same accepted new-route preflight')
    c.require(c.read(c.ROOT/value['resource_plan']['path']) == design.resource_plan(), 'Same prospectively measured prices')
    projection = provisional_projection(value)
    c.require(projection['affordable'] and projection == c.read(c.ROOT/value['prelaunch_resource_projection']['path']),
              'Full pipeline prospectively affordable under unchanged prices')
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
              'Initial physical resources; real owner still requires60-second RAM stability')
    c.require('torch' not in sys.modules, 'No model runtime in conditional launch validator')
    return c.save(c.OUT/'prelaunch_review.json', dict(approved=True, status='exact_scope_freeze_admitted',
        freeze_sha256=expected_sha, scope=c.SCOPE, time=c.now(), reviewer='independent model-free conditional-admission process',
        inputs=value['inputs']+[c.artifact(c.OUT/'freeze.json')], blocking_findings=[],
        available_ram_bytes=available, free_disk_bytes=free, resource_projection=projection,
        actual_new_entrypoint_required=c.SCOPE == 'pilot', models=0, games=0,
        source_changes_after_preflight=False, no_model_or_game_retries=True))

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
