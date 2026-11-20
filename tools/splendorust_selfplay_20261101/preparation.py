"""Model-free preparation and exclusive engineering freeze."""
import argparse
from copy import deepcopy
import sys
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

def prepare(native_build_path):
    from . import qualification_control as control, training, build_capture
    from tools.external_strength_screen_20261018 import rules
    from tools.future_training_rules_20261021 import decode_training_snapshot
    prior = accepted_parent()
    sources = verify_runtime_sources()
    prepared = design.prepare()
    prepared['qualification_plan'] = design.fixed(c.OUT/'qualification_plan.json', control.build_plan())
    prepared['training_plan'] = design.fixed(c.OUT/'training_plan.json', training.qualification_plan())
    profiles = deepcopy(prior['profiles']['candidate'])
    c.require(set(profiles) == {'16','512'}, 'Original512 and reference16 profiles')
    c.verify(list(profiles.values()))
    prepared['profiles'] = {'candidate': profiles}
    build = build_capture.validate_build(native_build_path)
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
        training_plan=training.qualification_plan(), source_specs=design.schedule(),
        learned_entrypoint=prepared['learned_entrypoint']))
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
    return prior, prepared

def freeze(native_build_path, test_path, review_paths):
    from . import qualification_control as control
    c.require(not (c.OUT/'freeze.json').exists(), 'Exclusive new qualification freeze; never replace evidence')
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
                 purpose='engineering_qualification_only', teacher_simulations=512,
                 original_weights_preserved=True, original_architecture=True,
                 fixed_precision='float32', automatic_retries=0, scheduled_monitor=False,
                 new_games=4, optimizer_steps=1, training_sample_reuse=False,
                 no_strength_claim=True, reference_promoted=False,
                 authorization='User: ok let\'s do the next step you suggest; qualify pipeline then bounded pilot if affordable')
    sources = python_dependencies((c.ROOT/'tools/splendorust_selfplay_20261101').glob('*.py'))
    tests = sorted((c.ROOT/'tests').glob('test_splendorust_selfplay*_20261101.py'))
    entries = [c.artifact(p) for p in sources+tests]
    entries += list(nested_artifacts(value))
    entries += list(nested_artifacts(c.read(native_build_path)))
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare','freeze'])
    parser.add_argument('--native-build', required=True)
    parser.add_argument('--tests')
    parser.add_argument('--reviews', nargs='*', default=[])
    args = parser.parse_args()
    if args.operation == 'prepare':
        result = prepare(Path(args.native_build))[1]
    else:
        c.require(args.tests is not None, 'Frozen model-free test receipt required')
        result = freeze(Path(args.native_build), Path(args.tests), [Path(x) for x in args.reviews])
    print(c.canonical(result))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
