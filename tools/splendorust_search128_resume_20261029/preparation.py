"""Freeze a three-segment continuation without changing earlier evidence."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import time

from . import common as c
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime

MANIFEST_SCHEMA = 'search128_three_segment_resume_manifest_v1'
FREEZE_SCHEMA = 'splendorust_search128_three_segment_resume_v1'
MAX_TREE_FILES = 100000
PRESERVATION_SECONDS = 900.


def declared_worker_lifecycles():
    return {f'qualification_resume_{route}': dict(route=route, game_index=39,
        graph_request_id='request-00000001', entrypoint_request_id='request-00000002',
        ordinary_first_request_id='request-00000003') for route in ('research', 'frozen')}


def _exact_int(value, expected, label):
    c.require(type(value) is int and value == expected, label)


def validate_closure_disposition(original_completion, recovery_completion, closure,
                                 checker_exit, checker_launch):
    """Reconcile already-passed receipts; never rerun science or promote history."""
    c.require(original_completion['status'] == recovery_completion['status'] == 'incomplete',
              'Both prior attempts must remain incomplete')
    c.require(closure['status'] == 'qualified_saved_closure_checker_only' and
        closure['freeze_sha256'] == c.FIRST_RECOVERY_SHA and
        closure['original_attempt_status'] == 'incomplete' and
        closure['original_attempt_accepted'] is False and closure['benchmark_accepted'] is False,
        'Separate saved closure proof only')
    for key, expected in [('actual_entrypoint_decisions', 119), ('graph_cases', 22),
                          ('ordinary_saved_moves', 14), ('new_model_calls', 0), ('new_games', 0)]:
        _exact_int(closure[key], expected, 'Saved closure count: '+key)
    c.require(closure['original_files_unchanged'] is True and closure['original_clocks_rewritten'] is False,
              'Saved checker preserved all original evidence/clocks')
    game = closure['segmented_game']; proof = game['proof']
    c.require(game['status'] == 'passed_saved_game_only' and game['game_rerun'] is False and
        game['original_attempt_accepted'] is False and proof['status'] == 'passed' and
        proof['original_failed_receipt_rewritten'] is False and proof['recovered_move_in_ordinary_timing'] is False,
        'One independently verified game, no historical acceptance')
    for key, expected in [('new_ordinary_moves', 14), ('prior_accepted_moves', 39),
                          ('recovered_actions', 1), ('model_calls', 0)]:
        _exact_int(game[key], expected, 'Saved game count: '+key)
    for key, expected in [('transitions', 54), ('recovered_transitions', 1), ('failed_moves', 0)]:
        _exact_int(proof[key], expected, 'Saved ledger count: '+key)
    c.require(sum(map(len, proof['latencies'].values())) == 53, 'Bridge excluded from ordinary timing')
    suites = closure['actual_entrypoint_verification']
    c.require(len(suites) == 2 and {s['route'] for s in suites} == {'research', 'frozen'} and
        all(s['status'] == 'passed' and s['complete_decisions'] == (67 if s['route'] == 'research' else 52)
            for s in suites), 'Both actual corrected entrypoint proofs')
    workers = closure['closed_workers']
    c.require(len(workers) == 2 and {p['worker'] for p in workers} ==
        {'main_g38_challenger', 'main_g38_opponent'}, 'Exact two separately verified retained workers')
    for worker in workers:
        raw = worker['raw']
        c.require(worker['status'] == raw['status'] == 'passed' and raw['exact_bytes_checked'] is True and
            raw['requests'] == 9 and raw['startup_requests'] == 2 and raw['failed_requests'] == 0,
            'Full saved raw lifecycle proof')
    _exact_int(checker_exit['exit_code'], 0, 'Actual saved-checker process exit')
    c.require(checker_exit['observed_by'] == 'retained subprocess.Popen handle' and
        checker_exit['model_calls'] == checker_exit['new_games'] == 0 and
        checker_exit['checker_pid'] == checker_launch['pid'] == closure['reader']['pid'] and
        checker_exit['checker_created'] == checker_launch['created'] == closure['reader']['created'],
        'Actual saved checker identity and owned exit')
    c.require(0 <= closure['elapsed_seconds'] <= checker_exit['elapsed_seconds'] < 900 and
        checker_launch['maximum_seconds'] == 900 and
        checker_launch['original_global_deadline'] == c.ORIGINAL_UNIX+c.JOB_SECONDS,
        'Saved checker stayed within its original bounded allowance')
    c.require(closure['owner_cleanup']['status'] == 'passed_failed_owner_cleanup' and
        closure['owner_cleanup']['actual_exit_codes'] == [1, 1, 1, 1] and
        closure['owner_cleanup']['experiment_accepted'] is False, 'Failed recovery owner exits preserved')
    return dict(status='passed_saved_receipt_binding', completed_main_games=39,
        original_completed_main_games=38, continuation_game_index=39,
        original_attempt_status='incomplete', first_recovery_status='incomplete',
        historical_attempts_accepted=False, new_models_or_games=0)


def merge_inputs(*groups):
    """Reject contradictory same-path pins instead of silently replacing them."""
    result = {}
    for group in groups:
        for item in group:
            c.require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'Exact artifact descriptor')
            c.require(item['path'] not in result or result[item['path']] == item,
                      'Frozen input collision: '+item['path'])
            result[item['path']] = item
    return sorted(result.values(), key=lambda item: item['path'])


def _inventory(folder, reader):
    folder = Path(folder); paths = []
    reader._stat(folder, directory=True)
    for base, dirs, names in os.walk(folder, followlinks=False):
        reader.guard()
        for name in dirs: reader._stat(Path(base)/name, directory=True)
        for name in names:
            path = Path(base)/name; reader._stat(path)
            paths.append(path)
            c.require(len(paths) <= MAX_TREE_FILES, 'Bounded preserved-tree file inventory')
    # Existing frozen inputs are already authenticated in parallel; most pins are
    # reused only under the unchanged complete file identity in ReviewRuntime.
    items = [reader.pin(path) for path in sorted(paths)]
    reader.exact_tree(folder, {item['path'] for item in items})
    return items


def _old_owners(folder, reader):
    proof = []
    for name, key in [('owner_launch.json', None), ('guardian_claim.json', 'guardian'),
                      ('watchdog_claim.json', 'watchdog'), ('launch.json', None)]:
        value = reader.read_path(folder/name)
        identity = dict(pid=value['pid'] if key is None else value[key+'_pid'],
                        created=value['created'] if key is None else value[key+'_created'])
        proof.append(dict(record=reader.pin(folder/name), **reader.absent(identity)))
    return proof


def make_manifest(trees, bindings, owners, closure_binding):
    c.require(set(trees) == {'original', 'recovery1', 'closure'}, 'All three preserved trees required')
    value = dict(schema=MANIFEST_SCHEMA, time=c.now(), **bindings,
        **{prefix+'_tree': items for prefix, items in trees.items()},
        **{prefix+'_tree_files': len(items) for prefix, items in trees.items()},
        **{prefix+'_tree_bytes': sum(item['bytes'] for item in items) for prefix, items in trees.items()},
        owner_absence=owners, saved_closure_binding=closure_binding,
        completed_main_games=39, original_completed_main_games=38, continuation_game_index=39,
        original_attempt_status='incomplete', first_recovery_status='incomplete',
        recovered_ply=39, recovered_game_index=38, recovered_actions=1,
        original_failed_receipt_preserved=True, move_clock_reset=False, new_call_for_failed_move=False,
        failed_decision_retries=0, completed_game_retries=0, automatic_retries=0,
        original_monotonic_origin=c.ORIGINAL_MONOTONIC, original_unix_origin=c.ORIGINAL_UNIX,
        prior_final_seconds=c.PRIOR_FINAL_SECONDS, prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,
        prior_allowances_are_conservative_allocations=True,
        maximum_combined_worker_lifecycles=864, new_entrypoint_decisions=119,
        new_entrypoint_parent_requests=2, qualified_prior_decisions=322,
        combined_qualification_decisions=441, qualification_ordinary_requests=84,
        declared_worker_lifecycles=declared_worker_lifecycles())
    merge_inputs(*(value[prefix+'_tree'] for prefix in trees))
    validate_manifest(value)
    return value


def validate_manifest(manifest):
    c.require(manifest['schema'] == MANIFEST_SCHEMA, 'Exact three-segment manifest schema')
    for key, expected in [('completed_main_games', 39), ('original_completed_main_games', 38),
        ('continuation_game_index', 39), ('recovered_ply', 39), ('recovered_game_index', 38),
        ('recovered_actions', 1), ('failed_decision_retries', 0), ('completed_game_retries', 0),
        ('automatic_retries', 0), ('maximum_combined_worker_lifecycles', 864),
        ('new_entrypoint_decisions', 119), ('new_entrypoint_parent_requests', 2),
        ('qualified_prior_decisions', 322), ('combined_qualification_decisions', 441),
        ('qualification_ordinary_requests', 84)]:
        _exact_int(manifest[key], expected, 'Exact manifest field: '+key)
    c.require(manifest['original_attempt_status'] == manifest['first_recovery_status'] == 'incomplete' and
        manifest['original_failed_receipt_preserved'] is True and manifest['move_clock_reset'] is False and
        manifest['new_call_for_failed_move'] is False, 'Historical failures and clock remain unchanged')
    c.require((manifest['original_monotonic_origin'], manifest['original_unix_origin'],
               manifest['prior_final_seconds'], manifest['prior_ram_wait_seconds']) ==
              (c.ORIGINAL_MONOTONIC, c.ORIGINAL_UNIX, 190., 200.), 'Original clocks/used allowance carry')
    c.require(manifest['declared_worker_lifecycles'] == declared_worker_lifecycles(),
              'Exactly two declared startup-to-game39 lifecycles')
    all_pins = {}
    for prefix, folder in [('original', c.OLD_OUT), ('recovery1', c.FIRST_RECOVERY_OUT), ('closure', c.CLOSURE_OUT)]:
        items = manifest[prefix+'_tree']; names = set(); root_name = folder.relative_to(c.ROOT).as_posix()+'/'
        c.require(type(items) is list and 0 < len(items) <= MAX_TREE_FILES, 'Bounded nonempty complete tree')
        for item in items:
            c.require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'} and
                type(item['bytes']) is int and item['bytes'] >= 0 and type(item['sha256']) is str and
                len(item['sha256']) == 64 and set(item['sha256']) <= set('0123456789abcdef'), 'Artifact pin type')
            name = item['path']; path = PurePosixPath(name)
            c.require(type(name) is str and name.startswith(root_name) and '\\' not in name and
                path.as_posix() == name and not any(p in ('.', '..') or ':' in p for p in path.parts),
                'Canonical exact preserved-root path')
            c.require(name not in names, 'No duplicate prior-tree file'); names.add(name); all_pins[name] = item
        _exact_int(manifest[prefix+'_tree_files'], len(items), 'Tree file count')
        _exact_int(manifest[prefix+'_tree_bytes'], sum(x['bytes'] for x in items), 'Tree byte count')
    for key, path, sha in [
        ('old_freeze', c.OLD_OUT/'freeze.json', c.OLD_FREEZE_SHA),
        ('recovery1_freeze', c.FIRST_RECOVERY_OUT/'freeze.json', c.FIRST_RECOVERY_SHA),
        ('closure_check', c.CLOSURE_OUT/'closure_check.json', c.CLOSURE_SHA),
        ('original_claim', c.OLD_OUT/'guardian_claim.json', None),
        ('bridge', c.FIRST_RECOVERY_OUT/'recovered_bridge.json', None),
        ('checker_os_exit', c.CLOSURE_OUT/'checker_os_exit.json', None)]:
        pin = manifest[key]; name = path.relative_to(c.ROOT).as_posix()
        c.require(pin == all_pins.get(name) and (sha is None or pin['sha256'] == sha),
                  'Exact preserved manifest dependency: '+key)
    c.require(manifest['saved_closure_binding']['status'] == 'passed_saved_receipt_binding' and
        manifest['saved_closure_binding']['historical_attempts_accepted'] is False,
        'Preservation is not acceptance of prior attempts')
    return True


def preserve():
    """Bounded byte preservation only; the actual worker qualification is future."""
    reader = ReviewRuntime(c.ROOT, c.OUT, default_seconds=PRESERVATION_SECONDS)
    reader.shared_deadlines = (c.ORIGINAL_MONOTONIC+c.COLLECTION_CUTOFF_SECONDS,
                               c.ORIGINAL_UNIX+c.COLLECTION_CUTOFF_SECONDS)
    for folder, sha in [(c.OLD_OUT, c.OLD_FREEZE_SHA), (c.FIRST_RECOVERY_OUT, c.FIRST_RECOVERY_SHA)]:
        c.require(reader.pin(folder/'freeze.json')['sha256'] == sha, 'Exact prior freeze')
    c.require(reader.pin(c.CLOSURE_OUT/'closure_check.json')['sha256'] == c.CLOSURE_SHA,
              'Exact independently completed closure checker receipt')
    old = reader.read_path(c.OLD_OUT/'freeze.json')
    recovery1 = reader.read_path(c.FIRST_RECOVERY_OUT/'freeze.json')
    closure = reader.read_path(c.CLOSURE_OUT/'closure_check.json')
    checker_exit = reader.read_path(c.CLOSURE_OUT/'checker_os_exit.json')
    checker_launch = reader.read_path(c.CLOSURE_OUT/'checker_launch.json')
    binding = validate_closure_disposition(reader.read_path(c.OLD_OUT/'completion.json'),
        reader.read_path(c.FIRST_RECOVERY_OUT/'completion.json'), closure, checker_exit, checker_launch)
    source_items = merge_inputs(old['inputs'], recovery1['inputs'], closure['consumed_inputs'])
    reader.verify_many([(reader.physical(item['path']), item) for item in source_items])
    # These include pinned Python/torch/native files outside the project. They
    # remain external runtime descriptors, rather than counterfeit project pins.
    reader.verify_many([(Path(item['path']) if Path(item['path']).is_absolute() else reader.physical(item['path']), item)
                        for item in closure['imported_frozen_sources_and_runtime']])
    owners = dict(original=_old_owners(c.OLD_OUT, reader),
                  recovery1=_old_owners(c.FIRST_RECOVERY_OUT, reader),
                  checker=reader.absent(dict(pid=checker_exit['checker_pid'], created=checker_exit['checker_created'])))
    for folder in (c.OLD_OUT, c.FIRST_RECOVERY_OUT):
        c.require(not (folder/'postrun/independent_completion_review.json').exists(),
                  'Neither incomplete attempt may become accepted')
    claim = reader.read_path(c.OLD_OUT/'guardian_claim.json')
    c.require((claim['monotonic_origin'], claim['unix_origin']) ==
              (c.ORIGINAL_MONOTONIC, c.ORIGINAL_UNIX), 'Original real-clock identity')
    trees = {key: _inventory(folder, reader) for key, folder in
             [('original', c.OLD_OUT), ('recovery1', c.FIRST_RECOVERY_OUT), ('closure', c.CLOSURE_OUT)]}
    bindings = dict(old_freeze=reader.pin(c.OLD_OUT/'freeze.json'),
        recovery1_freeze=reader.pin(c.FIRST_RECOVERY_OUT/'freeze.json'),
        original_claim=reader.pin(c.OLD_OUT/'guardian_claim.json'),
        bridge=reader.pin(c.FIRST_RECOVERY_OUT/'recovered_bridge.json'),
        closure_check=reader.pin(c.CLOSURE_OUT/'closure_check.json'),
        checker_os_exit=reader.pin(c.CLOSURE_OUT/'checker_os_exit.json'))
    value = make_manifest(trees, bindings, owners, binding)
    value['preservation_identity_revalidation'] = reader.revalidate_identities()
    value['preservation_metrics'] = dict(reader.metrics)
    value['preservation_seconds'] = max(time.monotonic()-reader.start, time.time()-reader.wall)
    reader.guard()
    return c.save(c.OUT/'recovery_manifest.json', value)


def build_freeze(old, manifest_pin, manifest, inputs, protocol_pin):
    validate_manifest(manifest)
    c.require(manifest['schema'] == MANIFEST_SCHEMA and
        manifest['completed_main_games'] == manifest['continuation_game_index'] == 39 and
        manifest['original_completed_main_games'] == 38 and
        manifest['original_attempt_status'] == manifest['first_recovery_status'] == 'incomplete',
        'Exact three-segment continuation manifest')
    c.require((manifest['original_monotonic_origin'], manifest['original_unix_origin'],
               manifest['prior_final_seconds'], manifest['prior_ram_wait_seconds']) ==
              (c.ORIGINAL_MONOTONIC, c.ORIGINAL_UNIX, 190., 200.), 'Unreset clocks and charged prior allowances')
    c.require(manifest['declared_worker_lifecycles'] == declared_worker_lifecycles(),
              'Prospective exact startup/play lifecycle declaration')
    frozen = deepcopy(old)
    frozen.update(schema=FREEZE_SCHEMA, time=c.now(), attempt=2,
        protocol=protocol_pin, stage_seconds=c.STAGES, max_workers=864, max_graph_builds=864,
        resume_manifest=manifest_pin, original_freeze=manifest['old_freeze'],
        recovery1_freeze=manifest['recovery1_freeze'], closure_check=manifest['closure_check'],
        checker_os_exit=manifest['checker_os_exit'],
        original_monotonic_origin=c.ORIGINAL_MONOTONIC, original_unix_origin=c.ORIGINAL_UNIX,
        prior_final_seconds=190., prior_ram_wait_seconds=200.,
        source_attempts_remain_incomplete=True, continuation_game_index=39,
        completed_main_games=39, original_completed_main_games=38,
        new_entrypoint_decisions=119, new_entrypoint_parent_requests=2,
        qualified_prior_decisions=322, combined_qualification_decisions=441,
        declared_worker_lifecycles=declared_worker_lifecycles(),
        recovered_response_policy='retain_single_authenticated_saved_response_as_separate_bridge',
        original_failed_receipt_overwritten=False, failed_decision_retries=0,
        completed_game_retries=0, inputs=merge_inputs(inputs))
    return frozen


def freeze():
    c.require(not (c.OUT/'freeze.json').exists(), 'Never replace a resume freeze')
    c.require(shutil.disk_usage(c.ROOT).free >= c.INITIAL_FREE_DISK_BYTES, 'Original launch disk reserve')
    old = c.read(c.OLD_OUT/'freeze.json'); recovery1 = c.read(c.FIRST_RECOVERY_OUT/'freeze.json')
    c.require(c.file_hash(c.OLD_OUT/'freeze.json') == c.OLD_FREEZE_SHA and
              c.file_hash(c.FIRST_RECOVERY_OUT/'freeze.json') == c.FIRST_RECOVERY_SHA, 'Both exact historical freezes')
    manifest = c.read(c.OUT/'recovery_manifest.json')
    tests = c.read(c.OUT/'preparation/model_free_tests.json')
    review = c.read(c.OUT/'preparation/implementation_review.json')
    c.require(tests['passed'] is True and tests['exit_code'] == 0, 'Passed held model-free aggregate required')
    c.require(review['approved'] is True, 'Independent implementation approval required')
    c.verify(review['inputs'])
    closure = c.read(c.CLOSURE_OUT/'closure_check.json')
    c.require(c.file_hash(c.CLOSURE_OUT/'closure_check.json') == c.CLOSURE_SHA, 'Exact completed closure proof')
    checker_review = c.read(c.CLOSURE_OUT/'prelaunch/implementation_review.json')
    c.require(checker_review['approved'] is True, 'Independent saved-checker approval')
    items = merge_inputs(old['inputs'], recovery1['inputs'], manifest['original_tree'],
        manifest['recovery1_tree'], manifest['closure_tree'], closure['consumed_inputs'], checker_review['inputs'])
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((c.ROOT/'tests').glob('test_splendorust_search128_resume*20261029*.py'))
    paths += [path for path in c.OUT.rglob('*') if path.is_file()]
    paths += [c.PROTOCOL]
    source_guard = c.save(c.OUT/'runtime_sources.json', verify_runtime_sources())
    items = merge_inputs(items, [c.artifact(path) for path in paths], [source_guard])
    frozen = build_freeze(old, c.artifact(c.OUT/'recovery_manifest.json'), manifest, items, c.artifact(c.PROTOCOL))
    from .runner import validate_freeze_contract
    validate_freeze_contract(frozen)
    reader = ReviewRuntime(c.ROOT, c.OUT, default_seconds=PRESERVATION_SECONDS)
    reader.shared_deadlines = (c.ORIGINAL_MONOTONIC+c.COLLECTION_CUTOFF_SECONDS,
                               c.ORIGINAL_UNIX+c.COLLECTION_CUTOFF_SECONDS)
    reader.verify_many([(reader.physical(item['path']), item) for item in frozen['inputs']])
    for prefix, folder in [('original', c.OLD_OUT), ('recovery1', c.FIRST_RECOVERY_OUT), ('closure', c.CLOSURE_OUT)]:
        reader.exact_tree(folder, {item['path'] for item in manifest[prefix+'_tree']})
    reader.revalidate_identities(); reader.guard()
    return c.save(c.OUT/'freeze.json', frozen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['preserve', 'freeze'])
    args = parser.parse_args()
    print(c.canonical(preserve() if args.action == 'preserve' else freeze()))
