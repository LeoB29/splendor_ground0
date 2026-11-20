"""One authorized model-free validation dispatch, preserving the completed source attempt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASE = ROOT/'outputs/live_game_operational_recovery_20261020_02'
BASE = ROOT/'outputs/live_game_operational_recovery_20261020_02_validation_02'
SOURCE = Path(__file__).resolve()
ANALYZER = ROOT/'tools/analyze_live_game_operational_final_validation_20261020.py'
MEMORY_MODULE = 'tools.operational_validation_memory_20261020'
MEMORY_SOURCE = ROOT/'tools/operational_validation_memory_20261020.py'
FROZEN_ANALYZER = ROOT/'tools/analyze_live_game_operational_recovery_20261020_02.py'
ORIGINAL_FREEZE_SHA256 = '8540930acd731cb12b2574365e9d919e9d3ee508c56b9aa7e3d61b442a3e65ca'
CAP, RESERVE, CPU = 43200, 600, 2
MAX_BYTES, EVIDENCE_RESERVE = 4*1024**3, 32*1024**2
MIN_FREE, RSS_LIMIT, MAX_LOG = 5*1024**3, 2*1024**3, 2*1024**2
PRESERVED_NAMES = ('freeze.json', 'launch.json', 'completion.json', 'operational_progress.json',
    'final_input_check.json', 'runner_process.json', 'runner_execution.json', 'coordinator_plan.json',
    'coordinator_launch.json', 'coordinator_process.json', 'coordinator_execution.json', 'coordinator_hold.json',
    'independent_analysis_execution_01.json', 'independent_analysis_execution_01.log',
    'sealed/coordinator_analyzer_failure_01.json', 'sealed/coordinator_failure_01.json',
    'memory_admissions/analysis.json', 'memory_admissions/analysis.jsonl')
OUTPUT_NAMES = ('launch.json', 'execution.json', 'analyzer.log', 'hold.json', 'failure.json',
                'independent_analysis.json', 'memory_admission.json', 'memory_admission.jsonl')


class Hold(RuntimeError): pass
def require(value, label):
    if not value: raise Hold(label)
def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}
def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
def finite(value): return type(value) in (int, float) and math.isfinite(value)
def bound(item, expected=None):
    require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'}, 'artifact_schema')
    path = (ROOT/item['path']).resolve()
    require(path.is_relative_to(ROOT) and (expected is None or path == Path(expected).resolve()), 'artifact_path')
    require(artifact(path) == item, 'artifact_changed')
    return path
def identity(process): return {'pid': process.pid, 'created': process.create_time()}
def closed(value):
    if value is None: return False
    try: return psutil.Process(value['pid']).create_time() != value['created']
    except psutil.NoSuchProcess: return True
def remaining(plan):
    return min(plan['epoch_seconds']+CAP-time.time(), plan['monotonic_start']+CAP-time.monotonic())
def file_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())
def resources(plan, *, final=False):
    left = remaining(plan)
    require(left > (0 if final else RESERVE), 'original_deadline_or_reserve')
    sizes = file_bytes(SOURCE_BASE)+file_bytes(BASE)
    require(sizes < MAX_BYTES-(0 if final else EVIDENCE_RESERVE), 'combined_evidence_limit')
    free = shutil.disk_usage(BASE).free
    require(free >= MIN_FREE, 'disk_reserve')
    return {'remaining_seconds': left, 'combined_evidence_bytes': sizes, 'free_disk_bytes': free}


def memory_module():
    for path in (ROOT, ROOT/'src'):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    return importlib.import_module(MEMORY_MODULE)


def build_plan(repair_freeze_sha256, source_sha256, *, final=False):
    """Called by preparation before writing the separately authenticated plan."""
    require(sha(SOURCE) == source_sha256, 'launcher_source_identity')
    freeze_path = BASE/'repair_freeze.json'
    require(sha(freeze_path) == repair_freeze_sha256, 'repair_freeze_identity')
    frozen = read(freeze_path)
    require(frozen['schema'] == 'operational_final_validation_repair_freeze_v1' and
            frozen['source_base'] == SOURCE_BASE.relative_to(ROOT).as_posix() and
            frozen['repair_base'] == BASE.relative_to(ROOT).as_posix() and
            frozen['outcomes_closed'] is True and frozen['larger_data_deferred'] is True and
            frozen['maximum_seconds'] == CAP and frozen['final_reserve_seconds'] == RESERVE,
            'repair_scope')
    bound(frozen['original_freeze'], SOURCE_BASE/'freeze.json')
    require(frozen['original_freeze']['sha256'] == ORIGINAL_FREEZE_SHA256, 'original_frozen_identity')
    bound(frozen['original_launch'], SOURCE_BASE/'launch.json')
    bound(frozen['frozen_analyzer'], FROZEN_ANALYZER)
    inventory = bound(frozen['source_inventory'])
    require(inventory.is_relative_to(BASE), 'repair_inventory_location')
    inputs = {}
    for item in frozen['inputs']:
        path = bound(item); require(path not in inputs, 'duplicate_repair_input'); inputs[path] = item
    for path in (SOURCE, ANALYZER, MEMORY_SOURCE, BASE/'launcher_probe_01.json', inventory):
        require(inputs.get(path.resolve()) == artifact(path), 'required_repair_input')
    launch = read(SOURCE_BASE/'launch.json')
    require(launch['freeze'] == frozen['original_freeze'] and launch['maximum_seconds'] == CAP and
            launch['outcomes_closed'] is True and all(finite(launch[k]) for k in ('epoch_seconds','monotonic_start')),
            'original_launch_binding')
    completion = read(SOURCE_BASE/'completion.json'); execution = read(SOURCE_BASE/'runner_execution.json')
    require(completion['status'] == 'completed_pending_independent_analysis' and completion['failure'] is None and
            completion['worker_processes_closed'] is True and completion['completed_games'] == 18 and
            completion['outcomes_closed'] is True and completion['strength_analysis_games'] == 0 and
            type(execution['exit_code']) is int and execution['exit_code'] == 0,
            'completed_collection_required')
    runner = read(SOURCE_BASE/'runner_process.json')
    require(execution['runner_identity'] == runner['identity'] and closed(runner['identity']), 'runner_not_closed')
    coordinator = read(SOURCE_BASE/'coordinator_process.json')
    require(closed(coordinator['identity']), 'old_coordinator_not_closed')
    old_execution = read(SOURCE_BASE/'independent_analysis_execution_01.json')
    require(old_execution['error'] is not None and old_execution['analyzer_process_closed'] is True and
            old_execution['owned_descendants_closed'] is True and old_execution['output'] is None and
            closed(old_execution['analyzer_process']) and
            all(closed(v) for v in old_execution['owned_descendants']), 'old_analysis_not_closed')
    require(not (SOURCE_BASE/'independent_analysis.json').exists() and
            not (SOURCE_BASE/'finalization.json').exists() and
            read(SOURCE_BASE/'coordinator_hold.json')['status'] == 'preserved_hold', 'prior_hold_not_preserved')
    probe = read(BASE/'launcher_probe_01.json')
    require(probe['exit_code'] == 0 and probe['parent_closed'] is True and
            probe['models_or_games_executed'] is False and probe['cpu'] == CPU and
            probe['creationflags'] == 'CREATE_NO_WINDOW' and len(probe['descendants']) == 1, 'console_probe')
    helper = probe['descendants'][0]
    expected_exe = (Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve()
    require(Path(helper['executable']).resolve() == expected_exe and helper['name'].lower() == 'conhost.exe' and
            helper['ppid'] == probe['parent']['pid'] and helper['created'] >= probe['parent']['created'] and
            helper['closed_after_parent_exit'] is True and helper['affinity'] == [CPU] and
            sha(expected_exe) == helper['executable_sha256'], 'console_probe_identity')
    original_frozen = read(SOURCE_BASE/'freeze.json')
    python = str(Path(original_frozen['executable']).resolve())
    require(Path(sys.executable).resolve() == Path(python), 'same_python_executable')
    plan = {'schema': 'final_validation_launcher_plan_v1', 'source_base': SOURCE_BASE.relative_to(ROOT).as_posix(),
        'repair_base': BASE.relative_to(ROOT).as_posix(), 'source': artifact(SOURCE), 'analyzer': artifact(ANALYZER),
        'repair_freeze': artifact(freeze_path), 'original_freeze': frozen['original_freeze'],
        'original_launch': frozen['original_launch'], 'frozen_analyzer': frozen['frozen_analyzer'],
        'source_inventory': frozen['source_inventory'], 'probe': artifact(BASE/'launcher_probe_01.json'),
        'admission_source': artifact(MEMORY_SOURCE), 'admission_module': MEMORY_MODULE,
        'prior_memory_state': memory_module().prior_memory_state(SOURCE_BASE),
        'preserved_artifacts': [artifact(SOURCE_BASE/name) for name in PRESERVED_NAMES],
        'runner_identity': runner['identity'], 'old_coordinator_identity': coordinator['identity'],
        'old_analyzer_identity': old_execution['analyzer_process'],
        'epoch_seconds': launch['epoch_seconds'], 'monotonic_start': launch['monotonic_start'],
        'maximum_seconds': CAP, 'final_reserve_seconds': RESERVE, 'cpu': CPU,
        'evidence_cap_bytes': MAX_BYTES, 'evidence_reserve_bytes': EVIDENCE_RESERVE,
        'minimum_free_bytes': MIN_FREE, 'rss_limit_bytes': RSS_LIMIT, 'maximum_log_bytes': MAX_LOG,
        'console_helper': {'executable': str(expected_exe), 'name': 'conhost.exe',
                           'sha256': helper['executable_sha256'], 'maximum_distinct_helpers': 1},
        'analyzer_command': [python, '-B', str(ANALYZER.relative_to(ROOT)),
                             '--repair-freeze-sha256', repair_freeze_sha256],
        'analysis_attempts': 1, 'automatic_retries': 0, 'games_or_replays_rerun': False,
        'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True,
        'finalization_owned_by_root': True}
    resources(plan, final=final)
    return plan


class ProcessTree:
    """Authenticate a single Python parent and at most one direct OS console helper."""
    def __init__(self, owner, plan):
        self.owner, self.plan = owner, plan
        self.observed = {}
        self.helpers = set()
        self.peak_rss = 0

    def sample(self):
        process = psutil.Process(self.owner['pid'])
        require(process.create_time() == self.owner['created'], 'analyzer_identity_changed')
        require(Path(process.exe()).resolve() == Path(self.plan['analyzer_command'][0]).resolve() and
                process.cpu_affinity() == [CPU], 'analyzer_executable_or_affinity')
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                key = (child.pid, child.create_time())
                record = {'pid': child.pid, 'created': key[1], 'parent_pid': child.ppid(), 'name': child.name(),
                    'executable': str(Path(child.exe()).resolve()), 'affinity': child.cpu_affinity(),
                    'owned_parent': self.owner, 'first_observed_utc': now(), 'allowed_console_helper': False}
                # Preserve ownership before rejecting a child so failure cleanup remains scoped.
                require(key[0] not in (os.getpid(), self.plan['runner_identity']['pid'], self.owner['pid']) and
                        key[1] >= self.owner['created'], 'invalid_descendant_ownership')
                if key not in self.observed: self.observed[key] = record
                else:
                    prior = self.observed[key]
                    require(all(prior[k] == record[k] for k in ('pid','created','parent_pid','name','executable','affinity')),
                            'descendant_identity_changed')
                record = self.observed[key]
                expected = self.plan['console_helper']
                require(record['parent_pid'] == self.owner['pid'] and record['name'].lower() == expected['name'] and
                        Path(record['executable']).resolve() == Path(expected['executable']).resolve() and
                        record['affinity'] == [CPU], 'unexpected_analyzer_descendant')
                if 'executable_sha256' not in record: record['executable_sha256'] = sha(record['executable'])
                require(record['executable_sha256'] == expected['sha256'], 'console_helper_hash')
                self.helpers.add(key); require(len(self.helpers) <= 1, 'console_helper_replacement')
                record['allowed_console_helper'] = True
                rss = child.memory_info().rss; record['peak_sampled_rss_bytes'] = max(record.get('peak_sampled_rss_bytes',0), rss)
                total += rss
            except psutil.NoSuchProcess: continue
        self.peak_rss = max(self.peak_rss, total)
        require(total <= RSS_LIMIT, 'combined_analyzer_tree_rss')
        return total

    def cleanup(self, child):
        errors, terminated = [], []
        if child.poll() is None:
            try: self.sample()
            except BaseException as error: errors.append(type(error).__name__)
        for value in sorted(self.observed.values(), key=lambda v:v['created'], reverse=True):
            try:
                process = psutil.Process(value['pid'])
                if process.create_time() == value['created']:
                    process.kill(); terminated.append({'pid':value['pid'],'created':value['created']})
            except psutil.NoSuchProcess: pass
            except BaseException as error: errors.append(type(error).__name__)
        if child.poll() is None:
            try:
                if not closed(self.owner): child.kill()
            except BaseException as error: errors.append(type(error).__name__)
        try: child.wait(timeout=5)
        except BaseException as error: errors.append(type(error).__name__)
        for value in self.observed.values():
            try:
                process = psutil.Process(value['pid'])
                if process.create_time() == value['created']: process.wait(timeout=5)
            except psutil.NoSuchProcess: pass
            except BaseException as error: errors.append(type(error).__name__)
        return {'errors':errors, 'terminated_owned_identities':terminated,
                'parent_closed':closed(self.owner), 'descendants_closed':all(closed(v) for v in self.observed.values())}


class Launcher:
    def __init__(self, plan_path, plan_sha256, source_sha256):
        self.plan_path = Path(plan_path).resolve()
        self.plan_sha256, self.source_sha256 = plan_sha256, source_sha256
        self.claimed = False

    def authenticate(self, *, final=False):
        require(self.plan_path == (BASE/'launcher_plan.json').resolve() and sha(self.plan_path) == self.plan_sha256,
                'launcher_plan_identity')
        value = read(self.plan_path)
        require(value == build_plan(value['repair_freeze']['sha256'], self.source_sha256, final=final), 'prospective_plan_identity')
        return value

    def execute(self, plan):
        admission = memory_module().wait_for_memory(SOURCE_BASE, BASE, read(SOURCE_BASE/'launch.json'),
            plan['prior_memory_state'], resources=lambda:resources(plan))
        require(plan == self.authenticate(), 'post_admission_plan_identity'); resources(plan)
        require(read(BASE/'memory_admission.json') == admission and admission['status'] == 'passed', 'admission_readback')
        started = time.time(); started_mono = time.monotonic()
        child = tree = owner = None; error = diagnostic = None
        cleanup = {'errors':[], 'terminated_owned_identities':[], 'parent_closed':False, 'descendants_closed':False}
        log_path = BASE/'analyzer.log'
        with log_path.open('xb') as log:
            try:
                child = subprocess.Popen(plan['analyzer_command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'},
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                process = psutil.Process(child.pid); owner = identity(process); tree = ProcessTree(owner,plan)
                next_resource = 0.
                while child.poll() is None:
                    if time.monotonic() >= next_resource:
                        resources(plan); next_resource = time.monotonic()+1.
                    require(remaining(plan) > RESERVE, 'original_deadline_or_reserve')
                    require(log_path.stat().st_size <= MAX_LOG, 'analyzer_log_limit')
                    try: tree.sample()
                    except psutil.NoSuchProcess:
                        require(child.poll() is not None, 'live_analyzer_identity_unavailable'); break
                    try: child.wait(timeout=.1)
                    except subprocess.TimeoutExpired: pass
                resources(plan); require(log_path.stat().st_size <= MAX_LOG, 'analyzer_log_limit')
            except BaseException as exc:
                error = type(exc).__name__
                diagnostic = {'time':now(),'traceback':traceback.format_exc()[-16384:],'outcomes_closed':True}
            finally:
                if tree is not None: cleanup = tree.cleanup(child)
                elif child is not None:
                    # No authenticated identity means no PID-based action is authorized.
                    cleanup['errors'].append('analyzer_identity_never_authenticated')
        if error is None:
            try: resources(plan)
            except BaseException as exc:
                error = type(exc).__name__
                diagnostic = {'time':now(),'traceback':traceback.format_exc()[-16384:],'outcomes_closed':True}
        ended, ended_mono = time.time(), time.monotonic()
        if diagnostic: save(BASE/'failure.json', diagnostic)
        output = BASE/'independent_analysis.json'
        execution = {'schema':'final_validation_repair_execution_v1','time':now(),'command':plan['analyzer_command'],
            'started_epoch_seconds':started,'finished_epoch_seconds':ended,'elapsed_seconds':ended-started,
            'started_monotonic':started_mono,'finished_monotonic':ended_mono,
            'elapsed_from_original_launch_seconds':ended-plan['epoch_seconds'],'maximum_seconds':CAP,
            'final_reserve_seconds':RESERVE,'exit_code':child.returncode if child else None,'error':error,
            'analyzer_identity':owner,'analyzer_executable':plan['analyzer_command'][0],
            'observed_descendants':list(tree.observed.values()) if tree else [],
            'peak_sampled_combined_rss_bytes':tree.peak_rss if tree else None,'cleanup':cleanup,
            'plan':artifact(self.plan_path),'launch':artifact(BASE/'launch.json'),'analyzer':plan['analyzer'],
            'repair_freeze':plan['repair_freeze'],'memory_admission':artifact(BASE/'memory_admission.json'),
            'memory_readings':artifact(BASE/'memory_admission.jsonl'),'log':artifact(log_path),
            'output':artifact(output) if output.exists() else None,'outcomes_closed':True,
            'games_or_replays_rerun':False,'prior_hold_preserved':True,'additional_execution_authorized':False}
        save(BASE/'execution.json', execution)
        require(execution['exit_code'] == 0 and error is None and not cleanup['errors'] and
                cleanup['parent_closed'] and cleanup['descendants_closed'] and output.exists(), 'final_validation_failed')
        result = read(output)
        require(result['status'] == 'passed' and result['outcomes_closed'] is True and
                result['strength_analysis_games'] == 0 and result['freeze_sha256'] == ORIGINAL_FREEZE_SHA256,
                'analyzer_result_scope')
        require(plan == self.authenticate(final=True), 'final_plan_identity'); resources(plan,final=True)
        return execution

    def run(self):
        psutil.Process().cpu_affinity([CPU])
        plan = self.authenticate()
        require(all(not (BASE/name).exists() for name in OUTPUT_NAMES), 'repair_already_attempted')
        save(BASE/'launch.json', {'schema':'final_validation_repair_launch_v1','time':now(),'source':artifact(SOURCE),
            'plan':artifact(self.plan_path),'plan_sha256':self.plan_sha256,'expected_source_sha256':self.source_sha256,
            'identity':identity(psutil.Process()),'cpu':CPU,'original_epoch_seconds':plan['epoch_seconds'],
            'original_monotonic_start':plan['monotonic_start'],'maximum_seconds':CAP,'final_reserve_seconds':RESERVE,
            'outcomes_closed':True,'prior_hold_preserved':True,'automatic_retries':0})
        self.claimed = True
        return self.execute(plan)

    def preserve_hold(self):
        if self.claimed:
            save(BASE/'hold.json',{'status':'preserved_hold','time':now(),'launch':artifact(BASE/'launch.json'),
                'traceback':traceback.format_exc()[-16384:],'outcomes_closed':True,'automatic_retries':0,
                'additional_execution_authorized':False})


def validate_execution_for_finalization(expected_freeze_sha, expected_plan_sha, expected_execution_sha):
    """Read-only acceptance of the owned validation dispatch; no report or source mutations."""
    require(sha(BASE/'repair_freeze.json') == expected_freeze_sha and
            sha(BASE/'launcher_plan.json') == expected_plan_sha and
            sha(BASE/'execution.json') == expected_execution_sha, 'finalization_expected_hashes')
    plan = read(BASE/'launcher_plan.json')
    require(plan['repair_freeze']['sha256'] == expected_freeze_sha, 'finalization_repair_freeze')
    launcher = Launcher(BASE/'launcher_plan.json', expected_plan_sha, plan['source']['sha256'])
    require(plan == launcher.authenticate(final=True), 'finalization_plan_identity')
    value = read(BASE/'execution.json'); launch = read(BASE/'launch.json')
    require(not (BASE/'hold.json').exists() and not (BASE/'failure.json').exists(), 'repair_hold_prevents_acceptance')
    require(value['schema'] == 'final_validation_repair_execution_v1' and
            value['exit_code'] == 0 and type(value['exit_code']) is int and value['error'] is None and
            value['command'] == plan['analyzer_command'] and value['analyzer'] == plan['analyzer'] and
            value['repair_freeze'] == plan['repair_freeze'] and value['plan'] == artifact(BASE/'launcher_plan.json') and
            value['launch'] == artifact(BASE/'launch.json') and value['outcomes_closed'] is True and
            value['games_or_replays_rerun'] is False and value['prior_hold_preserved'] is True and
            value['additional_execution_authorized'] is False and value['maximum_seconds'] == CAP and
            value['final_reserve_seconds'] == RESERVE, 'successful_dispatch_binding')
    require(launch['schema'] == 'final_validation_repair_launch_v1' and launch['source'] == plan['source'] and
            launch['plan'] == value['plan'] and launch['plan_sha256'] == expected_plan_sha and
            launch['expected_source_sha256'] == plan['source']['sha256'] and launch['cpu'] == CPU and
            launch['original_epoch_seconds'] == plan['epoch_seconds'] and
            launch['original_monotonic_start'] == plan['monotonic_start'] and launch['maximum_seconds'] == CAP and
            launch['final_reserve_seconds'] == RESERVE and launch['outcomes_closed'] is True and
            launch['prior_hold_preserved'] is True and launch['automatic_retries'] == 0, 'dispatch_launch_binding')
    cleanup = value['cleanup']
    require(cleanup['errors'] == [] and cleanup['parent_closed'] is True and cleanup['descendants_closed'] is True,
            'dispatch_cleanup_incomplete')
    for key in ('started_epoch_seconds','finished_epoch_seconds','started_monotonic','finished_monotonic',
                'elapsed_seconds','elapsed_from_original_launch_seconds'):
        require(finite(value[key]), 'finite_dispatch_clocks')
    require(plan['epoch_seconds'] <= value['started_epoch_seconds'] <= value['finished_epoch_seconds'] < plan['epoch_seconds']+CAP-RESERVE and
            plan['monotonic_start'] <= value['started_monotonic'] <= value['finished_monotonic'] < plan['monotonic_start']+CAP-RESERVE and
            value['elapsed_seconds'] == value['finished_epoch_seconds']-value['started_epoch_seconds'] and
            value['elapsed_from_original_launch_seconds'] == value['finished_epoch_seconds']-plan['epoch_seconds'],
            'analysis_and_cleanup_retained_original_final_reserve')
    owner = value['analyzer_identity']
    require(type(owner['pid']) is int and owner['pid'] > 0 and finite(owner['created']) and
            owner != plan['runner_identity'] and owner != launch['identity'] and
            value['analyzer_executable'] == plan['analyzer_command'][0] and closed(owner) and closed(launch['identity']),
            'authenticated_dispatch_processes_not_closed')
    require(type(value['peak_sampled_combined_rss_bytes']) is int and
            0 <= value['peak_sampled_combined_rss_bytes'] <= RSS_LIMIT, 'dispatch_rss_bound')
    descendants = value['observed_descendants']
    require(type(descendants) is list and len(descendants) <= 1, 'one_console_helper_only')
    for child in descendants:
        expected = plan['console_helper']
        require(type(child['pid']) is int and child['pid'] > 0 and child['pid'] not in
                (owner['pid'],launch['identity']['pid'],plan['runner_identity']['pid']) and
                finite(child['created']) and child['created'] >= owner['created'] and
                child['owned_parent'] == owner and child['parent_pid'] == owner['pid'] and
                child['name'].lower() == expected['name'] and child['executable'] == expected['executable'] and
                child['executable_sha256'] == expected['sha256'] == sha(child['executable']) and
                child['affinity'] == [CPU] and child['allowed_console_helper'] is True and closed(child),
                'authenticated_console_helper_not_closed')
    allowed = [{'pid':v['pid'],'created':v['created']} for v in descendants]
    require(all(v in allowed for v in cleanup['terminated_owned_identities']), 'cleanup_identity_scope')
    for key, path in (('memory_admission',BASE/'memory_admission.json'), ('memory_readings',BASE/'memory_admission.jsonl'),
                      ('log',BASE/'analyzer.log'), ('output',BASE/'independent_analysis.json')):
        bound(value[key],path)
    require(value['log']['bytes'] <= MAX_LOG, 'final_log_bound')
    admission = read(BASE/'memory_admission.json')
    require(admission['status'] == 'passed' and owner['created'] >= admission['finished_epoch_seconds'],
            'dispatch_after_supplemental_admission')
    result = read(BASE/'independent_analysis.json')
    require(result['status'] == 'passed' and result['outcomes_closed'] is True and
            result['strength_analysis_games'] == 0 and result['freeze_sha256'] == ORIGINAL_FREEZE_SHA256,
            'final_analysis_scope')
    resources(plan,final=True)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--expected-source-sha256',required=True);args=parser.parse_args()
    launcher=Launcher(args.plan,args.plan_sha256,args.expected_source_sha256)
    try: launcher.run()
    except BaseException as error:
        launcher.preserve_hold()
        print(json.dumps({'status':'preserved_hold','category':type(error).__name__,'outcomes_closed':True}),flush=True)
        return 1
    print(json.dumps({'status':'validation_dispatch_completed','outcomes_closed':True,'finalization_pending':True}),flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
