"""Fixed 49-case corrected-rules qualification, with immediate full replay audits."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from tools.corrected_planner_qualification_20261018 import common as c
from tools.local_player_qualification_128.clock import PersistentWorker, WorkerInitializationError, processor_topology
from tools.corrected_planner_qualification_20261018.clock import PlannerWorker

TESTS = ['tests/test_corrected_planner_adapter_20261018.py',
         'tests/test_corrected_planner_verify_20261018.py',
         'tests/test_corrected_planner_runner_20261018.py',
         'tests/test_local_player_qualification_clock.py',
         'tests/test_external_screen_rules_20261018.py',
         'tests/test_external_screen_audit_20261018.py']
HARD_FAILURES = {'rss_limit', 'log_limit', 'message_flood', 'affinity_changed', 'resource_guard',
                 'process_tree_limit', 'process_tree_unobservable'}


def preservation_inventory():
    prior = c.read(c.PRIOR_EXTERNAL / 'freeze.json')
    if c.file_hash(c.PRIOR_EXTERNAL / 'freeze.json') != '8e0c2565b2386947479346aa348cb6ed11ef548db1e3af592a943ecc1a2a1e5b':
        raise ValueError('Previous external freeze changed')
    c.verify_artifacts(prior['inputs'])
    old = c.read(c.PRIOR_LOCAL / 'freeze.json')
    c.verify_artifacts(old['inputs'])
    inventory = {v['path']: v for v in prior['inputs'] + old['inputs']}
    paths = list(c.PRIOR_EXTERNAL.rglob('*')) + list(c.PRIOR_LOCAL.rglob('*'))
    paths += [c.ROOT / 'docs/engineering/external_strength_screen_20261018_02_result.md',
              c.ROOT / 'docs/engineering/local_player_qualification_20261018_result.md']
    for p in paths:
        if p.is_file():
            item = c.artifact(p)
            if item['path'] in inventory and inventory[item['path']] != item:
                raise ValueError('Conflicting preservation identity')
            inventory[item['path']] = item
    return sorted(inventory.values(), key=lambda item: item['path'])


def prepare():
    base = c.OUTPUT
    base.mkdir(parents=True, exist_ok=False)
    try:
        prior_inputs = preservation_inventory()
        c.save_new(base / 'prior_preservation.json', {'time': c.now(), 'inputs': prior_inputs,
                   'files': len(prior_inputs), 'all_hashes_and_sizes_match': True})
        selection = c.read(c.PRIOR_LOCAL / 'selection.json')
        c.save_new(base / 'selection.json', selection)
        entries = selection['core'] + selection['edges']
        cases = c.build_cases(entries)
        c.save_new(base / 'schedule.json', cases)
        from tools.corrected_planner_qualification_20261018 import local
        from tools.external_strength_screen_20261018 import rules, audit
        privacy, fixtures = [], []
        for index, entry in enumerate(entries):
            if c.file_hash(c.ROOT / entry['path']) != entry['sha256']:
                raise ValueError('Historical selected record changed')
            original = c.read(c.ROOT / entry['path'])
            corrected = rules.convert_legacy_snapshot(original['snapshot'])
            state = rules.deserialize_snapshot(corrected)
            audit.validate_ledger(corrected)
            # Change genuine hidden allocations, before constructing either lawful input.
            variant_state = deepcopy(state)
            other = variant_state.players[1 - variant_state.current_player]
            for tier in (1, 2, 3):
                hidden = variant_state.hidden_tier_decks[tier]
                for slot, card in enumerate(other.reserved_cards):
                    if card.tier == tier and other.reserved_card_source(slot) != 'visible' and hidden:
                        other.reserved_cards[slot], hidden[0] = hidden[0], card
                hidden.reverse()
            changed = rules.serialize_snapshot(variant_state)
            observer = local.make_observer_payload(corrected, entry['context'])
            variant = local.make_observer_payload(changed, entry['context'])
            if observer != variant or changed == corrected:
                raise ValueError('Hidden-state permutation failed observer invariance')
            legal = [rules.serialize_action(a) for a in rules.CorrectedEnv().legal_actions(state)]
            c.save_new(base / 'observations' / f'{index:02d}.json', observer)
            c.save_new(base / 'variants' / f'{index:02d}.json', variant)
            c.save_new(base / 'fixtures' / f'{index:02d}.json', {'root_id': entry['root_id'],
                       'fit': entry['fit'], 'snapshot': corrected, 'hidden_variant': changed,
                       'legal_actions': legal, 'source_artifact': c.artifact(c.ROOT / entry['path']),
                       'fixture_semantics': 'Corrected-rules counterfactual; no historical reachability or unchanged outcome claim'})
            privacy.append({'root_id': entry['root_id'], 'observer_sha256': c.digest(observer),
                            'hidden_variant_observer_sha256': c.digest(variant),
                            'hidden_state_actually_changed': True, 'identical': True})
            fixtures.append({'root_id': entry['root_id'], 'fit': entry['fit'],
                             'corrected_legal_count': len(legal),
                             'legacy_legal_count': len(original['proposal']['legal_actions']),
                             'legal_support_changed': legal != original['proposal']['legal_actions']})
        c.save_new(base / 'privacy_inputs.json', privacy)
        c.save_new(base / 'fixture_conversion.json', fixtures)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', *TESTS, '-p', 'no:cacheprovider']
        env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
        started = time.monotonic()
        tests = subprocess.run(command, cwd=c.ROOT, env=env, capture_output=True, text=True, timeout=300,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        c.save_new(base / 'preflight_tests.json', {'command': command, 'returncode': tests.returncode,
                   'stdout': tests.stdout, 'stderr': tests.stderr, 'seconds': time.monotonic() - started})
        if tests.returncode:
            raise ValueError('Preflight tests failed')
        topology = processor_topology()
        if topology['api'] != 'GetLogicalProcessorInformationEx(RelationProcessorCore)' or len(topology['distinct_efficiency_classes']) < 2:
            raise ValueError('Cannot authenticate assigned performance core')
        paths = list((c.ROOT / 'tools/corrected_planner_qualification_20261018').glob('*.py'))
        paths += [c.ROOT / name for name in TESTS]
        paths += [c.ROOT / 'tools/analyze_corrected_planner_qualification_20261018.py',
                  c.ROOT / 'docs/engineering/corrected_planner_qualification_20261018.md',
                  c.ROOT / 'docs/engineering/verification_policy.md']
        paths += [p for p in (c.ROOT / 'outputs/corrected_planner_preparation_20261018_01').rglob('*') if p.is_file()]
        paths += list(base.rglob('*.json'))
        inventory = {item['path']: item for item in prior_inputs}
        for p in paths:
            item = c.artifact(p); inventory[item['path']] = item
        c.save_new(base / 'freeze.json', {'time': c.now(), 'profile_id': c.PROFILE_ID,
                   'ruleset_id': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
                   'inputs': sorted(inventory.values(), key=lambda item: item['path']),
                   'topology': topology, 'python': sys.version, 'executable': sys.executable,
                   'platform': platform.platform(),
                   'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'psutil', 'pytest')},
                   'cases': len(cases), 'fits': list(c.FITS), 'worlds': 128, 'maximum_candidates': 4,
                   'limits': {'seconds': c.MAX_SECONDS, 'bytes': c.MAX_BYTES,
                              'worker_rss_bytes': c.MAX_WORKER_RSS, 'disk_reserve_bytes': c.MIN_FREE,
                              'initialization_seconds': c.INITIALIZATION_SECONDS, 'audit_seconds_per_completed_root': c.AUDIT_SECONDS},
                   'strength_games_authorized': 0, 'larger_data_deferred': True,
                   'audit_mode': 'Every completed decision; all continuations, raw-policy reexecution, independent ledger transitions'})
        print(c.canonical({'prepared': str(base), 'cases': len(cases), 'fits': list(c.FITS),
                           'frozen_inputs': len(inventory), 'production_model_initializations': 0}), flush=True)
    except BaseException:
        c.save_new(base / 'preparation_failure.json', {'time': c.now(), 'failure': traceback.format_exc()})
        raise


def worker_options(base, deadline, cpu):
    return {'affinity': [cpu], 'initialization_timeout': c.INITIALIZATION_SECONDS,
            'env': {'PYTHONPATH': str(c.ROOT / 'src') + os.pathsep + str(c.ROOT), 'CUDA_VISIBLE_DEVICES': ''},
            'max_message_bytes': 4 * 1024**2, 'max_request_bytes': 4 * 1024**2,
            'max_log_bytes': 4 * 1024**2, 'max_rss_bytes': c.MAX_WORKER_RSS,
            'absolute_deadline': deadline - c.FINAL_RESERVE_SECONDS,
            'resource_guard': lambda: c.boundary(base, deadline, memory_headroom=False)}


def without_result(receipt):
    return {k: v for k, v in receipt.items() if k != 'result'}


def one_attempt(case, entry, observer, legal, deadline, cpu):
    base = c.OUTPUT; folder = base / 'attempts' / f'{case["index"]:03d}'
    trace_path = folder / 'continuations.jsonl.gz'
    started = time.monotonic()
    row = {'case': case, 'root_id': entry['root_id'], 'fit': entry['fit'],
           'original_sha256': entry['sha256'], 'time': c.now(),
           'observer_sha256': c.digest(observer), 'correct': False}
    try:
        row['resources_before'] = c.boundary(base, deadline)
        reserve = c.INITIALIZATION_SECONDS * 2 + case['allowance_seconds'] + c.AUDIT_SECONDS + c.FINAL_RESERVE_SECONDS
        if deadline - time.monotonic() < reserve:
            raise c.ResourceHold('Insufficient full attempt, full audit and finalization time')
        options = worker_options(base, deadline, cpu)
        options['env']['CORRECTED_PLANNER_TRACE_PATH'] = str(trace_path)
        with PlannerWorker('tools.corrected_planner_qualification_20261018.worker', folder, **options) as worker:
            row['initialization'] = without_result(worker.initialization)
            clock = worker.request({'observer': observer, 'breadth': False}, case['allowance_seconds'],
                                   validate=lambda r: isinstance(r, dict) and isinstance(r.get('decision'), dict)
                                   and r['decision'].get('action') in legal)
            row['clock'] = without_result(clock)
        if clock['status'] in HARD_FAILURES:
            raise c.ResourceHold('Production resource failure: ' + clock['status'])
        if clock['status'] == 'illegal':
            raise ValueError('Completed corrected-rules action illegal or malformed')
        if clock['status'] not in ('ok', 'timeout'):
            raise ValueError('Non-timeout player failure, stopping conservatively: ' + str(clock))
        if clock['status'] != 'ok':
            return row
        result = clock['result']
        c.check_wrapper(result, observer, legal, trace_path)
        row['result_sha256'] = c.digest(result)
        row['decision_sha256'] = c.digest(result['decision'])
        row['trace'] = c.artifact(trace_path)
        c.boundary(base, deadline)
        audit_folder = base / 'audits' / f'{case["index"]:03d}'
        print(c.canonical({'event': 'timely_decision_awaiting_audit', 'case': case['index'] + 1,
                           'fit': entry['fit'], 'receipt_seconds': clock['receipt_elapsed_seconds']}), flush=True)
        with PlannerWorker('tools.corrected_planner_qualification_20261018.audit_worker', audit_folder,
                              **worker_options(base, deadline, cpu)) as audit_worker:
            row['audit_initialization'] = without_result(audit_worker.initialization)
            audit_clock = audit_worker.request({'observer': observer, 'result': result,
                                                'traces_path': str(trace_path)}, c.AUDIT_SECONDS,
                                               validate=lambda r: isinstance(r, dict))
            row['audit_clock'] = without_result(audit_clock)
        if audit_clock['status'] != 'ok':
            raise ValueError('Full replay audit failed: ' + str(audit_clock))
        proof = audit_clock['result']
        c.check_audit_proof(proof, observer, result)
        row['verification'] = proof
        row['audit_result_sha256'] = c.digest(proof)
        row['work'] = {'worlds': 128, 'continuations': result['trace']['records']}
        row['correct'] = True
        return row
    except WorkerInitializationError as error:
        if 'clock' in row:
            row['audit_clock'] = {**without_result(error.receipt), 'status': 'initialization_' + error.receipt['status']}
            row['failure'] = str(error)
            raise
        row['clock'] = {**without_result(error.receipt), 'status': 'initialization_' + error.receipt['status']}
        row['failure'] = str(error)
        if error.receipt['status'] in HARD_FAILURES:
            raise c.ResourceHold('Initialization resource boundary') from error
        if error.receipt['status'] != 'timeout':
            raise
        return row
    except BaseException:
        row['failure'] = traceback.format_exc()
        if 'clock' not in row:
            row['clock'] = {'status': 'referee_or_resource_failure', 'accepted': False, 'receipt_elapsed_seconds': None}
        raise
    finally:
        row['total_seconds'] = time.monotonic() - started
        c.save_new(base / 'rows' / f'{case["index"]:03d}.json', row)


def _run():
    base = c.OUTPUT; lock = c.read(base / 'freeze.json')
    c.verify_artifacts(lock['inputs'])
    if 'torch' in sys.modules:
        raise ValueError('Torch must not load in timed supervisor')
    started = time.monotonic(); deadline = started + c.MAX_SECONDS
    c.save_new(base / 'launch.json', {'time': c.now(), 'epoch_seconds': time.time(), 'pid': os.getpid(),
               'freeze': c.artifact(base / 'freeze.json'), 'maximum_seconds': c.MAX_SECONDS,
               'torch_loaded_in_supervisor': False})
    selection = c.read(base / 'selection.json'); entries = selection['core'] + selection['edges']
    cases = c.read(base / 'schedule.json'); failure = None; rows = []
    try:
        for case in cases:
            if 'torch' in sys.modules:
                raise ValueError('Torch entered timed supervisor')
            index = case['root_index']; entry = entries[index]
            observer = c.read(base / ('variants' if case['hidden_variant'] else 'observations') / f'{index:02d}.json')
            legal = c.read(base / 'fixtures' / f'{index:02d}.json')['legal_actions']
            row = one_attempt(case, entry, observer, legal, deadline, lock['topology']['selected_logical_processor'])
            if case['kind'] == 'repeat' and row['correct']:
                prior = next(r for r in rows if r['case']['root_index'] == index
                             and r['case']['allowance_seconds'] == 120 and r['case']['kind'] == 'core')
                if prior['correct'] and (prior['decision_sha256'] != row['decision_sha256']
                                         or prior['trace']['sha256'] != row['trace']['sha256']):
                    raise ValueError('Repeat with identical lawful input changed scientific result or trace')
            rows.append(row)
            print(c.canonical({'case': case['index'] + 1, 'total': len(cases), 'fit': entry['fit'],
                  'root': entry['root_id'], 'allowance': case['allowance_seconds'], 'status': row['clock']['status'],
                  'correct': row['correct'], 'receipt_seconds': row['clock']['receipt_elapsed_seconds'],
                  'elapsed_minutes': (time.monotonic() - started) / 60}), flush=True)
    except BaseException:
        failure = traceback.format_exc()
        c.save_new(base / 'failure.json', {'time': c.now(), 'failure': failure})
    finally:
        rows = [c.read(p) for p in sorted((base / 'rows').glob('*.json'))]
        try:
            c.verify_artifacts(lock['inputs'])
            c.save_new(base / 'final_input_check.json', {'time': c.now(), 'inputs_checked': len(lock['inputs']),
                       'all_hashes_and_sizes_match': True})
            c.boundary(base, deadline, memory_headroom=False)
        except BaseException:
            failure = (failure or '') + '\nFinalization failure:\n' + traceback.format_exc()
            c.save_new(base / 'finalization_failure.json', {'failure': failure})
        research = [r for r in rows if r['case']['allowance_seconds'] == 120]
        research_qualified = failure is None and len(rows) == len(cases) and len(research) == 26 and all(r['correct'] for r in research)
        result = {'time': c.now(), 'complete_schedule': len(rows) == len(cases),
                  'completed_without_referee_failure': failure is None and len(rows) == len(cases),
                  'scientifically_accepted': research_qualified, 'research_qualified_120s': research_qualified,
                  'attempted': len(rows), 'prescribed': len(cases), 'failure': failure,
                  'verified_completed_decisions': sum(r['correct'] for r in rows),
                  'elapsed_seconds': time.monotonic() - started,
                  'artifacts_bytes_before_result': c.directory_bytes(base),
                  'strength_games': 0, 'larger_data_deferred': True,
                  'scope': 'Corrected-rules counterfactual saved-position qualification; no new playing-strength evidence'}
        c.save_new(base / 'result.json', result)
        c.save_new(base / 'completion.json', {'time': c.now(), 'result': c.artifact(base / 'result.json'),
                   'status': 'completed' if failure is None else 'preserved_failure', 'worker_processes_closed': True})
        print(c.canonical({'completed': len(rows), 'verified': result['verified_completed_decisions'], 'failure': failure}), flush=True)
    return 0 if failure is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        with c.execution_lease():
            raise SystemExit(_run())
