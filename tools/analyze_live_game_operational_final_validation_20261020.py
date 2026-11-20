"""Separate admission/provenance wrapper around the unchanged frozen numerical audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/live_game_operational_recovery_20261020_02'
REPAIR = ROOT/'outputs/live_game_operational_recovery_20261020_02_validation_02'
FROZEN = ROOT/'tools/analyze_live_game_operational_recovery_20261020_02.py'
FROZEN_SHA = '6661435f1a784bdc6492f1b4518a386d5409ff72f3033a52b2563f842b8ebb43'
ORIGINAL_FREEZE_SHA = '8540930acd731cb12b2574365e9d919e9d3ee508c56b9aa7e3d61b442a3e65ca'
CAP, RESERVE = 43200, 600
MAX_BYTES, MIN_FREE = 4*1024**3, 5*1024**3
PRIOR_GATES = 458


def require(value, label):
    if not value: raise ValueError('Final validation admission: '+label)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def finite(value): return type(value) in (int, float) and math.isfinite(value)
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}
def bound(item, expected=None):
    require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'} and
            type(item['path']) is str and type(item['bytes']) is int and item['bytes'] >= 0,
            'Artifact descriptor schema')
    path = (ROOT/item['path']).resolve()
    require(path.is_relative_to(ROOT) and (expected is None or path == Path(expected).resolve()) and
            artifact(path) == item, 'Authenticated artifact bytes')
    return path
def now(): return datetime.now(timezone.utc).isoformat()


def boundary(launch):
    wall = time.time()-launch['epoch_seconds']; mono = time.monotonic()-launch['monotonic_start']
    require(0 <= wall < CAP-RESERVE and 0 <= mono < CAP-RESERVE, 'Original dual-clock deadline and final reserve')


def verify_inventory(document, expected_base, launch):
    require(document['source_base'] == expected_base.relative_to(ROOT).as_posix() and
            document['outcomes_closed'] is True and type(document['files']) is list, 'Completed source inventory identity')
    items = {}; size = 0
    for item in document['files']:
        boundary(launch); path = bound(item)
        require(path.is_relative_to(expected_base) and path not in items, 'Unique source inventory member')
        items[path] = item; size += item['bytes']
    require({p.resolve() for p in expected_base.rglob('*') if p.is_file()} == set(items),
            'No omitted or added source evidence')
    return items, size


def previous_memory(source, items):
    paths = sorted((source/'memory_admissions').rglob('*.json'))
    receipts = []; charge = 0.
    for path in paths:
        descriptor = artifact(path)
        require(items.get(path.resolve()) == descriptor, 'Prior memory admission preserved')
        value = read(path)
        require(value['status'] == 'passed' and finite(value['charged_seconds']) and value['charged_seconds'] >= 0,
                'Prior admission passed and charged')
        charge += value['charged_seconds']; receipts.append(descriptor)
    return {'count': len(paths), 'charged_seconds': charge, 'receipts': receipts}


def check_supplemental(value, prior, launch):
    constants = {'schema': 'operational_memory_admission_v2', 'supplemental': True,
        'name': 'analysis_resume_02', 'status': 'passed', 'failure_category': None, 'failure_detail': None,
        'initial': False, 'stability_required': True, 'maximum_wait_seconds': 7200,
        'maximum_cumulative_wait_seconds': 14400, 'minimum_available_physical_bytes': 2*1024**3,
        'minimum_available_commit_bytes': 4*1024**3, 'sample_interval_seconds': 10, 'stable_seconds': 120,
        'required_consecutive_samples': 13, 'reserve_seconds': 600, 'request_seconds': 0,
        'initialization_seconds': 60, 'no_worker_during_admission': True, 'outcomes_closed': True}
    require(all(canonical(value.get(k)) == canonical(v) for k, v in constants.items()), 'Supplemental admission contract')
    require(type(value['prior_admission_count']) is int and value['prior_admission_count'] == prior['count'] and
            value['prior_gate_inputs'] == prior['receipts'] and
            value['cumulative_before_seconds'] == prior['charged_seconds'], 'No erased prior waiting or gates')
    keys = ('started_epoch_seconds', 'finished_epoch_seconds', 'started_monotonic', 'finished_monotonic',
            'elapsed_seconds', 'charged_seconds', 'cumulative_before_seconds', 'cumulative_after_seconds')
    require(all(finite(value[k]) for k in keys), 'Finite supplemental clocks')
    start, end = value['started_monotonic'], value['finished_monotonic']
    wall_start, wall_end = value['started_epoch_seconds'], value['finished_epoch_seconds']
    charge = max(end-start, wall_end-wall_start, 0.)
    require(value['elapsed_seconds'] == end-start and value['charged_seconds'] == charge and
            0 <= charge < 7200 and value['cumulative_after_seconds'] == prior['charged_seconds']+charge and
            0 <= value['cumulative_after_seconds'] < 14400 and
            0 <= start-launch['monotonic_start'] <= end-launch['monotonic_start'] < CAP-RESERVE and
            0 <= wall_start-launch['epoch_seconds'] < CAP-RESERVE and
            0 <= wall_end-launch['epoch_seconds'] < CAP-RESERVE, 'Unrefunded supplemental charge and original deadline')
    rows = value['readings']; require(type(rows) is list and 13 <= len(rows) <= 721, 'Complete stable-window readings')
    streak = 0; first_good = None; previous = None
    for index, row in enumerate(rows):
        require(set(row) == {'epoch_seconds', 'monotonic_seconds', 'elapsed_seconds', 'available_physical_bytes',
                'available_commit_bytes', 'admissible', 'consecutive_admissible_samples', 'remaining_seconds'},
                'Supplemental reading schema')
        mono, epoch = row['monotonic_seconds'], row['epoch_seconds']
        physical, commit = row['available_physical_bytes'], row['available_commit_bytes']
        require(all(finite(row[k]) for k in ('epoch_seconds', 'monotonic_seconds', 'elapsed_seconds', 'remaining_seconds')) and
                start <= mono <= end and 0 <= epoch-launch['epoch_seconds'] < CAP-RESERVE and
                row['elapsed_seconds'] == mono-start and (previous is None or mono-previous >= 9.95) and
                type(physical) is int and type(commit) is int and min(physical, commit) >= 0 and
                660 <= row['remaining_seconds'] <= CAP, 'Supplemental thresholds, spacing and retained work reserve')
        good = physical >= 2*1024**3 and commit >= 4*1024**3
        streak = streak+1 if good else 0
        first_good = (mono if streak == 1 else first_good) if good else None
        require(row['admissible'] is good and type(row['consecutive_admissible_samples']) is int and
                row['consecutive_admissible_samples'] == streak, 'Independent supplemental good streak')
        eligible = good and streak >= 13 and mono-first_good >= 120
        require(eligible is (index == len(rows)-1), 'First complete mandatory two-minute stable window')
        previous = mono
    return wall_end


def check_previous_dispatch(source, items):
    execution = read(bound(items[(source/'independent_analysis_execution_01.json').resolve()]))
    require(type(execution['exit_code']) is int and execution['exit_code'] == 1 and execution['error'] == 'Hold' and
            execution['cleanup_error'] is None and execution['analyzer_process_closed'] is True and
            execution['owned_descendants_closed'] is True and execution['output'] is None and
            execution['log']['bytes'] == 0 and execution['frozen_analyzer'] == artifact(FROZEN),
            'Preserved startup hold and closed processes')
    for name in ('coordinator_hold.json', 'sealed/coordinator_analyzer_failure_01.json', 'sealed/coordinator_failure_01.json'):
        bound(items[(source/name).resolve()])
    require(read(source/'coordinator_hold.json')['status'] == 'preserved_hold' and
            not (source/'independent_analysis.json').exists() and
            not (source/'sealed/independent_analysis_failure_01.json').exists(), 'Preserved prior failure without overwritten analysis')
    return execution


class Validation:
    def __init__(self, expected): self.expected = expected

    def authenticate(self):
        require(sha(REPAIR/'repair_freeze.json') == self.expected, 'Separate prospective repair freeze')
        self.repair_freeze = read(REPAIR/'repair_freeze.json')
        require(self.repair_freeze['schema'] == 'operational_final_validation_repair_freeze_v1' and
                self.repair_freeze['outcomes_closed'] is True and self.repair_freeze['larger_data_deferred'] is True,
                'Repair freeze scope')
        inputs = {}
        for item in self.repair_freeze['inputs']:
            path = bound(item); require(path not in inputs, 'Unique repair inputs'); inputs[path] = item
        self.repair_inputs = inputs
        for key, path in (('original_freeze', SOURCE/'freeze.json'), ('original_launch', SOURCE/'launch.json')):
            require(inputs.get(path.resolve()) == self.repair_freeze[key], 'Original control also in repair inputs')
            bound(self.repair_freeze[key], path)
        require(sha(SOURCE/'freeze.json') == ORIGINAL_FREEZE_SHA and sha(FROZEN) == FROZEN_SHA,
                'Exact original experiment and numerical verifier')
        self.launch = read(SOURCE/'launch.json')
        require(self.launch['maximum_seconds'] == CAP and self.launch['freeze'] == artifact(SOURCE/'freeze.json') and
                self.launch['outcomes_closed'] is True and all(finite(self.launch[k]) for k in ('epoch_seconds', 'monotonic_start')),
                'Unchanged original clock origin')
        boundary(self.launch)
        required = (Path(__file__).resolve(), ROOT/'tests/test_operational_final_validation_analyzer_20261020.py', FROZEN)
        require(all(path in inputs for path in required), 'Wrapper, regression tests and frozen algorithm authenticated')
        snapshot_path = bound(self.repair_freeze['source_inventory'])
        require(snapshot_path.is_relative_to(REPAIR) and inputs.get(snapshot_path) == self.repair_freeze['source_inventory'],
                'Completed evidence snapshot separately frozen')
        self.snapshot = read(snapshot_path)
        require(self.snapshot['schema'] == 'operational_final_validation_source_inventory_v1', 'Source inventory schema')
        self.source_items, self.source_bytes = verify_inventory(self.snapshot, SOURCE, self.launch)
        self.original_freeze = read(SOURCE/'freeze.json')
        self.original_inputs = {(ROOT/item['path']).resolve(): item for item in self.original_freeze['inputs']}
        require(len(self.original_inputs) == len(self.original_freeze['inputs']), 'Original input inventory unique')
        for name in ('analyze_live_game_operational_recovery_20261020_02.py',
                     'analyze_live_game_operational_recovery_20261020.py', 'analyze_live_game_operational_20261019.py'):
            bound(self.original_inputs[(ROOT/'tools'/name).resolve()], ROOT/'tools'/name)
        self.execution = check_previous_dispatch(SOURCE, self.source_items)
        self.prior = previous_memory(SOURCE, self.source_items)
        require(self.prior['count'] == PRIOR_GATES, 'Exact preserved admission count')
        return self

    def supplemental(self):
        value = read(REPAIR/'memory_admission.json')
        passed = check_supplemental(value, self.prior, self.launch)
        log = REPAIR/'memory_admission.jsonl'; raw = log.read_bytes()
        require(value['readings_log'] == {'path': log.relative_to(REPAIR).as_posix(), 'sha256': sha(log), 'bytes': len(raw)} and
                [json.loads(line) for line in raw.decode('utf-8').splitlines()] == value['readings'],
                'Complete immutable supplemental reading log')
        import psutil
        require(psutil.Process().create_time() >= passed, 'Second dispatch starts after fresh stable admission')
        self.admission = value
        self.admission_artifact, self.admission_log_artifact = artifact(REPAIR/'memory_admission.json'), artifact(log)

    def finish(self, result):
        boundary(self.launch)
        waiting = result['memory_waiting']
        require(result['status'] == 'passed' and result['outcomes_closed'] is True and result['strength_analysis_games'] == 0 and
                result['freeze_sha256'] == ORIGINAL_FREEZE_SHA and waiting['admissions'] == self.prior['count'] and
                math.isclose(waiting['charged_seconds'], self.prior['charged_seconds'], rel_tol=0., abs_tol=1e-6),
                'Unchanged numerical acceptance and prior memory reconciliation')
        _, size = verify_inventory(self.snapshot, SOURCE, self.launch)
        for item in self.original_freeze['inputs']:
            boundary(self.launch); bound(item)
        require(sha(REPAIR/'repair_freeze.json') == self.expected, 'Repair freeze unchanged during analysis')
        for item in self.repair_inputs.values():
            boundary(self.launch); bound(item)
        require(result['source_preservation']['files'] == len(self.original_inputs) and
                result['source_preservation']['all_hashes_and_sizes_match'] is True, 'Original full source coverage')
        require(artifact(REPAIR/'memory_admission.json') == self.admission_artifact and
                artifact(REPAIR/'memory_admission.jsonl') == self.admission_log_artifact,
                'Supplemental receipt unchanged during analysis')
        result['combined_memory_waiting'] = {'admissions': waiting['admissions']+1,
            'stability_required_admissions': waiting['stability_required_admissions']+1,
            'charged_seconds': self.admission['cumulative_after_seconds']}
        result['final_validation_repair'] = {'schema': 'operational_final_validation_repair_result_v1',
            'time': now(), 'repair_freeze': artifact(REPAIR/'repair_freeze.json'), 'wrapper': artifact(__file__),
            'frozen_numerical_algorithm_unchanged': True, 'original_coordinator_hold_retained': True,
            'previous_analysis_execution': artifact(SOURCE/'independent_analysis_execution_01.json'),
            'original_hold': artifact(SOURCE/'coordinator_hold.json'),
            'source_inventory': self.repair_freeze['source_inventory'],
            'source_evidence_files_rechecked': len(self.source_items), 'source_evidence_bytes_rechecked': size,
            'all_original_frozen_inputs_rechecked_after_analysis': len(self.original_inputs),
            'supplemental_admission': self.admission_artifact, 'supplemental_readings': self.admission_log_artifact,
            'original_deadline_retained': True, 'final_reserve_seconds': RESERVE,
            'elapsed_from_original_launch_seconds': time.time()-self.launch['epoch_seconds'],
            'new_games': 0, 'new_model_inference': False, 'outcomes_closed': True}
        return result

    def run(self):
        self.authenticate(); self.supplemental(); boundary(self.launch)
        frozen = importlib.import_module('tools.analyze_live_game_operational_recovery_20261020_02')
        result = frozen.Audit(SOURCE).run()
        return self.finish(result)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--repair-freeze-sha256', required=True)
    args = parser.parse_args(); output = REPAIR/'independent_analysis.json'
    require(not output.exists(), 'One exclusive repaired analysis output')
    try:
        validation = Validation(args.repair_freeze_sha256); result = validation.run()
        payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
        boundary(validation.launch)
        import shutil
        size = sum(p.stat().st_size for folder in (SOURCE, REPAIR) for p in folder.rglob('*') if p.is_file())
        require(size+len(payload.encode('utf-8')) < MAX_BYTES and shutil.disk_usage(REPAIR).free >= MIN_FREE,
                'Combined evidence and disk limits')
        with output.open('x', encoding='utf-8', newline='\n') as stream: stream.write(payload)
    except BaseException:
        folder = REPAIR/'sealed'; folder.mkdir(exist_ok=True)
        with (folder/'independent_analysis_failure_01.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump({'time': now(), 'traceback': traceback.format_exc(), 'outcomes_closed': True},
                      stream, sort_keys=True, indent=2)
        raise
    print(json.dumps({'status': 'passed', 'output': str(output), 'outcomes_closed': True,
                      'operational_games_checked': 18, 'strength_analysis_games': 0}), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        print(json.dumps({'status': 'failed', 'category': type(error).__name__, 'outcomes_closed': True}), file=sys.stderr)
        raise SystemExit(1)
