"""Outcome-free workload calibration, followed by a fixed 46-request gate."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.local_player_qualification_128 import common as io
from tools.local_player_qualification_128.clock import processor_topology, WorkerInitializationError
from tools.external_strength_screen_20261018 import rules, audit
from tools.external_strength_screen_20261018_02.runner import hidden_variant
from tools.analyze_external_strength_screen_20261018_02 import check_diagnostics
from tools.splendorust_120_qualification_20261018 import common as c

BASE = io.ROOT / 'outputs/splendorust_120_qualification_20261018_01'
PRIOR = io.ROOT / 'outputs/corrected_planner_qualification_20261018_01'
EXTERNAL = io.ROOT / 'outputs/external_strength_screen_20261018_02'
GRID = (256, 1024, 4096, 6400)
CALIBRATION = (0, 3, 18, 19, 20, 21, 22)
MOVE, INIT, MARGIN, CAP, FINAL_RESERVE = 120, 60, 96, 14400, 600
TESTS = ['tests/test_splendorust_120_adapter_20261018.py',
         'tests/test_splendorust_120_runner_20261018.py',
         'tests/test_splendorust_120_analyzer_20261018.py',
         'tests/test_external_screen_splendorust_bridge_20261018.py',
         'tests/test_external_screen_rules_20261018.py',
         'tests/test_external_screen_audit_20261018.py',
         'tests/test_external_screen_runner_20261018_02.py',
         'tests/test_local_player_qualification_clock.py']


def without_result(value):
    return {k: v for k, v in value.items() if k != 'result'}


def verify(items):
    for item in items:
        if io.artifact(io.ROOT / item['path']) != item:
            raise ValueError('Changed frozen artifact: ' + item['path'])


def prepare():
    BASE.mkdir(parents=True, exist_ok=False)
    try:
        original = io.read(PRIOR / 'freeze.json')
        if io.file_hash(PRIOR / 'freeze.json') != '75afcb4f4412cb2b8f106db93481c9d318734b3f55a1bec07cc3df01b363b180':
            raise ValueError('Prior freeze identity')
        verify(original['inputs'])
        if io.file_hash(PRIOR / 'independent_analysis_v2.json') != '4140061c31572602ea32227c3240a598ba33ff347c15574ae6495f9e3647dec2':
            raise ValueError('Prior accepted qualification identity')
        inventory = {v['path']: v for v in original['inputs']}
        extras = list(PRIOR.rglob('*')) + list(EXTERNAL.rglob('*'))
        extras += [io.ROOT / p for p in (
            'docs/engineering/corrected_planner_qualification_20261018_result.md',
            'tools/analyze_corrected_planner_qualification_20261018_v2.py',
            'tests/test_corrected_planner_analyzer_v2_20261018.py')]
        for path in extras:
            if path.is_file():
                item = io.artifact(path)
                if item['path'] in inventory and item != inventory[item['path']]:
                    raise ValueError('Prior inventory conflict')
                inventory[item['path']] = item
        io.save_new(BASE / 'prior_preservation.json', {'time': io.now(), 'inputs': list(inventory.values()),
                    'all_hashes_and_sizes_match': True})
        fixtures = io.read(EXTERNAL / 'fixtures.json')
        if len(fixtures) != 23 or [(fixtures[i]['source']['actor'], fixtures[i]['source']['phase']) for i in (0, 3)] != [(0, 0), (1, 0)]:
            raise ValueError('Fixed calibration metadata changed')
        variants, observers = [], []
        for item in fixtures:
            before = item['snapshot']; audit.validate_ledger(before)
            state = rules.deserialize_snapshot(before)
            variant = rules.serialize_snapshot(hidden_variant(state))
            if variant == before: raise ValueError('Hidden fixture did not change')
            original_obs = rules.observation_payload(state)
            alternate_obs = rules.observation_payload(rules.deserialize_snapshot(variant))
            audit.check_observer(before, original_obs); audit.check_observer(variant, alternate_obs)
            if original_obs != alternate_obs: raise ValueError('Hidden information crossed observer boundary')
            variants.append(variant); observers.append({'original': original_obs, 'hidden_variant': alternate_obs})
        io.save_new(BASE / 'fixtures.json', fixtures)
        io.save_new(BASE / 'variants.json', variants)
        io.save_new(BASE / 'observers.json', observers)
        for quota in GRID:
            io.save_new(BASE / 'profiles' / f'q{quota:05d}.json', c.make_profile(quota))
        schedule = [{'index': 2*i+j, 'fixture': i, 'kind': kind} for i in range(23)
                    for j, kind in enumerate(('original', 'hidden_variant'))]
        io.save_new(BASE / 'qualification_schedule.json', schedule)
        test = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *TESTS, '-p', 'no:cacheprovider'],
            cwd=io.ROOT, env={**os.environ, 'PYTHONPATH': str(io.ROOT / 'src')+os.pathsep+str(io.ROOT),
            'PYTHONDONTWRITEBYTECODE': '1'}, text=True, capture_output=True, timeout=300,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        io.save_new(BASE / 'preflight_tests.json', {'returncode': test.returncode, 'stdout': test.stdout, 'stderr': test.stderr})
        if test.returncode: raise ValueError('Preflight failed; preserve evidence')
        paths = set(inventory)
        paths.update(TESTS)
        paths.update(('tools/analyze_splendorust_120_qualification_20261018.py',
                      'docs/engineering/splendorust_120_qualification_20261018.md'))
        for folder in (io.ROOT / 'tools/splendorust_120_qualification_20261018', BASE):
            paths.update(p.relative_to(io.ROOT).as_posix() for p in folder.rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts)
        topology = processor_topology()
        if topology != original['topology']: raise ValueError('Hardware allocation/topology changed')
        io.save_new(BASE / 'freeze.json', {'time': io.now(), 'inputs': [io.artifact(io.ROOT/p) for p in sorted(paths)],
            'topology': topology, 'executable': sys.executable, 'ruleset_id': rules.RULESET_ID,
            'catalogue_sha256': rules.CATALOGUE_SHA256, 'grid': list(GRID), 'calibration_indices': list(CALIBRATION),
            'seconds_per_move': MOVE, 'initialization_seconds': INIT, 'internal_budget_seconds': 108,
            'calibration_margin_seconds': MARGIN, 'execution_cap_seconds': CAP,
            'finalization_reserve_seconds': FINAL_RESERVE, 'qualification_requests': 46,
            'maximum_requests': 74, 'max_evidence_bytes': 256*1024**2,
            'max_process_tree_rss_bytes': 2*1024**3, 'minimum_free_disk_bytes': 5*1024**3,
            'strength_games_authorized': 0, 'larger_data_deferred': True})
        print(io.canonical({'prepared': str(BASE), 'frozen_files': len(paths), 'maximum_requests': 74}), flush=True)
    except BaseException:
        io.save_new(BASE / 'preparation_failure.json', {'time': io.now(), 'failure': traceback.format_exc()})
        raise


def check_result(result, observer, profile, profile_sha, snapshot):
    state = rules.deserialize_snapshot(snapshot)
    legal = rules.legal_actions(state)
    expected_identity = {'player': 'opponent', 'observer_sha256': io.digest(observer),
        'legal_actions_sha256': c.legal_digest(legal), 'legal_count': len(legal),
        'ruleset': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256,
        'profile_sha256': profile_sha}
    if any(result.get(k) != v for k, v in expected_identity.items()):
        raise ValueError('Completed/exhausted observer and profile binding')
    if result['status'] == 'internal_budget_exhausted':
        expected = {**expected_identity, 'status': 'internal_budget_exhausted', 'internal_budget_seconds': 108,
            'opponent_id': 'splendorust', 'profile_id': profile['profile_id'],
            'reason': 'Internal fixed-work deadline exceeded'}
        if result != expected: raise ValueError('Internal exhaustion included unexpected partial result')
        return None
    if result['status'] != 'completed' or set(result) != set(expected_identity) | {'status', 'action', 'diagnostics'}:
        raise ValueError('Unknown result schema')
    action = result['action']
    serialized = [c.serialize_action(a) for a in legal]
    if action not in serialized: raise ValueError('Illegal completed action')
    check_diagnostics(result['diagnostics'], observer, action, serialized, profile)
    after = rules.serialize_snapshot(rules.CorrectedEnv().step(state, c.deserialize_action(action)))
    audit.check_step(snapshot, action, after)
    return after


def margin_passed(row):
    return row['correct'] is True and row['clock']['status'] == 'ok' and row['clock']['receipt_elapsed_seconds'] <= MARGIN


def one_attempt(stage, name, quota, fixture, kind, deadline, freeze):
    reserve = (46 * (INIT+MOVE) if stage == 'calibration' else 0) + INIT + MOVE + FINAL_RESERVE
    io.boundary(BASE, deadline)
    if deadline-time.monotonic() < reserve:
        raise io.ResourceHold('Insufficient declared request/finalization reserve')
    if 'torch' in sys.modules: raise ValueError('Supervisor must remain model-free')
    profile_path = BASE / 'profiles' / f'q{quota:05d}.json'
    profile = c.load_profile(profile_path, io.file_hash(profile_path))
    original = io.read(BASE / 'fixtures.json')[fixture]
    snapshot = original['snapshot'] if kind == 'original' else io.read(BASE / 'variants.json')[fixture]
    observer = io.read(BASE / 'observers.json')[fixture][kind]
    audit.check_observer(snapshot, observer)
    row = {'stage': stage, 'quota': quota, 'fixture': fixture, 'kind': kind,
        'root_id': original['source']['root_id'], 'observer_sha256': io.digest(observer),
        'profile': io.artifact(profile_path), 'correct': False}
    started = time.monotonic()
    folder = BASE / 'workers' / stage / name
    try:
        with c.ProcessTreeWorker('tools.splendorust_120_qualification_20261018.worker', folder,
            affinity=[freeze['topology']['selected_logical_processor']], initialization_timeout=INIT,
            absolute_deadline=deadline-FINAL_RESERVE,
            env={'PYTHONPATH': str(io.ROOT / 'src')+os.pathsep+str(io.ROOT), 'CUDA_VISIBLE_DEVICES': '',
                 'SPLENDOR_OPPONENT_PROFILE': str(profile_path.resolve()),
                 'SPLENDOR_OPPONENT_PROFILE_SHA256': row['profile']['sha256'],
                 'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'},
            max_message_bytes=256*1024, max_log_bytes=2*1024**2, max_rss_bytes=2*1024**3,
            resource_guard=lambda: io.boundary(BASE, deadline, memory_headroom=False)) as child:
            row['initialization'] = without_result(child.initialization)
            receipt = child.request({'player': 'opponent', 'observer': observer}, MOVE)
            row['clock'] = without_result(receipt)
            if receipt['status'] == 'ok':
                row['result'] = receipt['result']
                after = check_result(row['result'], observer, profile, row['profile']['sha256'], snapshot)
                row['result_sha256'] = io.digest(row['result'])
                if after is not None:
                    row.update(correct=True, post_snapshot=after, ledger_sha256=io.digest(after))
            elif receipt['status'] != 'timeout':
                raise ValueError('Non-timing worker failure: '+receipt['status'])
    except WorkerInitializationError as error:
        row['initialization'] = without_result(error.receipt)
        row['clock'] = {'status': 'initialization_'+error.receipt['status'], 'accepted': False,
                        'receipt_elapsed_seconds': None}
        row['failure'] = traceback.format_exc()
        raise
    except BaseException:
        row['failure'] = traceback.format_exc()
        raise
    finally:
        row['total_seconds'] = time.monotonic()-started
        io.save_new(BASE / 'rows' / stage / (name+'.json'), row)
    print(io.canonical({'stage': stage, 'case': name, 'quota': quota, 'fixture': fixture,
        'clock_status': row['clock']['status'], 'result_status': row.get('result', {}).get('status'),
        'correct': row['correct'], 'receipt_seconds': row['clock']['receipt_elapsed_seconds']}), flush=True)
    return row


def run():
    freeze = io.read(BASE / 'freeze.json'); verify(freeze['inputs'])
    if 'torch' in sys.modules: raise ValueError('Torch in supervisor')
    if (BASE / 'launch.json').exists(): raise ValueError('Never restart a launched qualification')
    started = time.monotonic(); deadline = started + CAP
    io.save_new(BASE / 'launch.json', {'time': io.now(), 'epoch_seconds': time.time(),
        'monotonic_start': started, 'maximum_seconds': CAP, 'freeze': io.artifact(BASE / 'freeze.json'),
        'torch_loaded_in_supervisor': False, 'pid': os.getpid()})
    selected, failure, calibration, final, pairs = None, None, [], [], 0
    try:
        attempts = []
        for quota in GRID:
            rows = []
            for ordinal, fixture in enumerate(CALIBRATION):
                row = one_attempt('calibration', f'q{quota:05d}_{ordinal:02d}', quota, fixture,
                                  'original', deadline, freeze)
                rows.append(row); calibration.append(row)
                if not margin_passed(row): break
            passed = len(rows) == len(CALIBRATION) and all(margin_passed(row) for row in rows)
            attempts.append({'quota': quota, 'passed_margin': passed, 'requests': len(rows),
                'profile': io.artifact(BASE / 'profiles' / f'q{quota:05d}.json'),
                'maximum_completed_receipt_seconds': max((r['clock']['receipt_elapsed_seconds'] for r in rows if r['correct']), default=None)})
            if not passed: break
            selected = quota
        selection = {'time': io.now(), 'grid': list(GRID), 'calibration_indices': list(CALIBRATION),
            'maximum_accepted_receipt_seconds': MARGIN, 'attempts': attempts, 'selected_quota': selected,
            'no_match_outcomes_used': True, 'profile': None}
        if selected is not None:
            io.save_new(BASE / 'selected_profile.json', io.read(BASE / 'profiles' / f'q{selected:05d}.json'))
            selection['profile'] = io.artifact(BASE / 'selected_profile.json')
        io.save_new(BASE / 'calibration_selection.json', selection)
        if selected is not None:
            schedule = io.read(BASE / 'qualification_schedule.json')
            for case in schedule:
                row = one_attempt('qualification', f'{case["index"]:03d}', selected, case['fixture'],
                                  case['kind'], deadline, freeze)
                final.append(row)
                if case['kind'] == 'hidden_variant' and final[-2]['correct'] and row['correct']:
                    if final[-2]['result'] != row['result']: raise ValueError('Hidden variant changed scientific result')
                    pairs += 1
    except BaseException:
        failure = traceback.format_exc()
        io.save_new(BASE / 'failure.json', {'time': io.now(), 'failure': failure})
    finally:
        calibration = [io.read(p) for p in sorted((BASE / 'rows/calibration').glob('*.json'))]
        final = [io.read(p) for p in sorted((BASE / 'rows/qualification').glob('*.json'))]
        try:
            verify(freeze['inputs'])
            io.boundary(BASE, deadline, memory_headroom=False)
            io.save_new(BASE / 'final_input_check.json', {'time': io.now(), 'inputs_checked': len(freeze['inputs']),
                        'all_hashes_and_sizes_match': True})
        except BaseException:
            failure = (failure or '')+'\nFinal check: '+traceback.format_exc()
            io.save_new(BASE / 'finalization_failure.json', {'time': io.now(), 'failure': failure})
        complete = failure is None and (selected is None or len(final) == 46)
        qualified = complete and selected is not None and len(final) == 46 and all(r['correct'] for r in final) and pairs == 23
        result = {'time': io.now(), 'complete_schedule': complete, 'qualified': qualified,
            'selected_quota': selected, 'calibration_attempted': len(calibration), 'qualification_attempted': len(final),
            'verified_decisions': sum(r['correct'] for r in final), 'privacy_pairs_matched': pairs,
            'failure': failure, 'elapsed_seconds': time.monotonic()-started,
            'strength_games': 0, 'larger_data_deferred': True}
        io.save_new(BASE / 'qualification_result.json', result)
        io.save_new(BASE / 'completion.json', {'time': io.now(), 'status': 'completed' if failure is None else 'preserved_failure',
            'worker_processes_closed': True, 'result': io.artifact(BASE / 'qualification_result.json')})
        print(io.canonical(result), flush=True)
    return 0 if failure is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        with io.execution_lease(): raise SystemExit(run())
