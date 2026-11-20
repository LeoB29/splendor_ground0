"""One-shot, model-free orchestration of the already frozen operational audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/live_game_operational_20261019_01'
SOURCE = Path(__file__).resolve()
ANALYZER = ROOT/'tools/analyze_live_game_operational_20261019.py'
REPORT = ROOT/'docs/engineering/live_game_operational_20261019_result.md'
FREEZE_SHA256 = '21ca9cb617ca65ad9745eeed76a2f5f91153dd638981c7fbfd76f4e348a27c51'
RUNNER = {'pid': 63784, 'created': 1792434628.726858}
EPOCH = 1792434638.850369
MONOTONIC = 2197300.906
CAP, RESERVE, CPU, POLL = 86400, 600, 2, 30
MAX_BYTES, MIN_FREE, RSS_LIMIT = 4*1024**3, 5*1024**3, 2*1024**3
MAX_LOG = 2*1024**2
RECEIPT_DELIVERY_SECONDS = 5.
ALLOWED_STATUSES = {'completed', 'forfeit_timeout', 'forfeit_worker_exit', 'forfeit_malformed',
    'forfeit_wrong_request', 'forfeit_illegal', 'forfeit_unsolicited', 'forfeit_write_error', 'forfeit_internal_budget_exhausted'}


class Hold(RuntimeError): pass
def require(value, label):
    if not value: raise Hold(label)
def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def artifact(path):
    return {'path': path.resolve().relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
def alive(identity):
    try: return psutil.Process(identity['pid']).create_time() == identity['created']
    except psutil.NoSuchProcess: return False
def confirmed_closed(identity):
    try: return identity is not None and not alive(identity)
    except psutil.Error: return False
def remaining(): return min(EPOCH+CAP-time.time(), MONOTONIC+CAP-time.monotonic())


def remember_descendants(process, owner, identities):
    require(process.create_time() == owner['created'], 'analyzer_identity_changed')
    children = process.children(recursive=True)
    for child in children:
        try: identity = {'pid': child.pid, 'created': child.create_time()}
        except psutil.NoSuchProcess: continue
        require(identity['created'] >= owner['created'] and identity['pid'] not in (RUNNER['pid'], os.getpid(), owner['pid']),
                'invalid_analyzer_descendant_identity')
        identities[(identity['pid'], identity['created'])] = identity
    return children


def close_owned(child, owner, descendants):
    """Only the authenticated Popen process and its observed descendants are owned."""
    errors = []
    if owner is not None and child.poll() is None:
        try: remember_descendants(psutil.Process(owner['pid']), owner, descendants)
        except psutil.NoSuchProcess: pass
        except BaseException as error: errors.append(error)
    for identity in sorted(descendants.values(), key=lambda v: v['created'], reverse=True):
        try:
            process = psutil.Process(identity['pid'])
            if process.create_time() == identity['created']: process.kill()
        except psutil.NoSuchProcess: pass
        except BaseException as error: errors.append(error)
    code = None
    try:
        if child.poll() is None: child.kill()
        code = child.wait(timeout=30)
    except BaseException as error: errors.append(error)
    for identity in descendants.values():
        try:
            process = psutil.Process(identity['pid'])
            if process.create_time() == identity['created']: process.wait(timeout=5)
        except psutil.NoSuchProcess: pass
        except BaseException as error: errors.append(error)
    if errors: raise Hold('owned_analyzer_cleanup_incomplete') from errors[0]
    return code


def resources(*, reserve=RESERVE):
    left = remaining(); size = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
    free = shutil.disk_usage(BASE).free
    require(left > reserve, 'original_deadline_reserve')
    require(size < MAX_BYTES-32*1024**2, 'evidence_reserve')
    require(free >= MIN_FREE, 'disk_reserve')
    return {'remaining_seconds': left, 'evidence_bytes': size, 'evidence_cap_bytes': MAX_BYTES,
            'disk_free_bytes': free, 'minimum_free_bytes': MIN_FREE, 'final_reserve_seconds': RESERVE}


def analyzer_command(freeze):
    return [freeze['executable'], '-B', ANALYZER.relative_to(ROOT).as_posix(),
            '--base', BASE.relative_to(ROOT).as_posix(), '--output', (BASE/'independent_analysis.json').relative_to(ROOT).as_posix()]


def make_plan():
    """Parent may call this before launch and exclusively save coordinator_plan.json."""
    freeze = read(BASE/'freeze.json'); launch = read(BASE/'launch.json'); process = read(BASE/'runner_process.json')
    require(sha(BASE/'freeze.json') == FREEZE_SHA256, 'experiment_freeze_identity')
    require(launch['epoch_seconds'] == EPOCH and launch['monotonic_start'] == MONOTONIC and
            launch['maximum_seconds'] == CAP and launch['pid'] == RUNNER['pid'], 'original_launch_identity')
    expected_runner = [freeze['executable'], '-B', '-u', '-m', 'tools.live_game_operational_20261019.runner', 'run']
    require(process['identity'] == RUNNER and process['command'] == expected_runner and
            process['freeze_sha256'] == FREEZE_SHA256 and process['outcomes_closed'] is True, 'runner_process_identity')
    frozen_analyzer = [v for v in freeze['inputs'] if v['path'] == ANALYZER.relative_to(ROOT).as_posix()]
    require(frozen_analyzer == [artifact(ANALYZER)], 'frozen_analyzer_identity')
    return {'schema': 'operational_completion_plan_v1', 'source': artifact(SOURCE),
        'experiment_freeze': artifact(BASE/'freeze.json'), 'experiment_launch': artifact(BASE/'launch.json'),
        'runner_process': artifact(BASE/'runner_process.json'), 'runner_identity': RUNNER, 'runner_command': expected_runner,
        'analyzer': frozen_analyzer[0], 'analyzer_command': analyzer_command(freeze),
        'original_epoch_seconds': EPOCH, 'original_monotonic_start': MONOTONIC, 'maximum_seconds': CAP,
        'final_reserve_seconds': RESERVE, 'cpu': CPU, 'poll_seconds': POLL, 'evidence_cap_bytes': MAX_BYTES,
        'minimum_free_bytes': MIN_FREE, 'report': REPORT.relative_to(ROOT).as_posix(), 'outcomes_closed': True,
        'maximum_analysis_log_bytes': MAX_LOG,
        'analysis_attempts': 1, 'automatic_retries': 0, 'automatic_handoff_updates': False}


def check_runner(execution):
    process = read(BASE/'runner_process.json'); complete = read(BASE/'completion.json')
    require(execution['runner_identity'] == process['identity'] == RUNNER and
            execution['command'] == process['command'] and execution['outcomes_closed'] is True and
            type(execution['exit_code']) is int and execution['exit_code'] == 0, 'runner_execution_failed_or_mismatched')
    require(not alive(RUNNER), 'runner_not_closed')
    require(complete['status'] == 'completed_pending_independent_analysis' and complete['failure'] is None and
            complete['completed_games'] == 18 and complete['worker_processes_closed'] is True and
            complete['outcomes_closed'] is True, 'runner_not_complete')


def delivered_runner_execution():
    """Allow only the exclusive writer's short, incomplete-JSON delivery window."""
    deadline = time.monotonic()+RECEIPT_DELIVERY_SECONDS
    while True:
        try: return read(BASE/'runner_execution.json')
        except json.JSONDecodeError:
            resources()
            left = deadline-time.monotonic()
            require(left > 0, 'runner_execution_delivery_incomplete')
            time.sleep(min(.1, left, max(.001, remaining()-RESERVE)))


def operational_report(analysis):
    """Only the frozen analyzer's operational field allowlist enters this report."""
    require(analysis['status'] == 'passed' and analysis['outcomes_closed'] is True and
            type(analysis['strength_analysis_games']) is int and analysis['strength_analysis_games'] == 0 and
            analysis['operational_schedule']['games'] == 18 and analysis['operational_schedule']['pairs'] == 9 and
            analysis['larger_data_deferred'] is True,
            'accepted_analysis_contract')
    rows = []; failures = 0
    for key, label in (('games:local', 'Local planner'), ('games:opponent', 'SplendoRust')):
        row = analysis['per_stage'][key]; counts = row['statuses']; timing = row['completed_receipt_seconds']
        require(set(counts) <= ALLOWED_STATUSES and all(type(n) is int and n >= 0 for n in counts.values()), 'operational_status_contract')
        failed = sum(n for name, n in counts.items() if name != 'completed'); failures += failed
        fmt = lambda value: 'n/a' if value is None else f'{value:.3f}'
        rows.append(f"| {label} | {counts.get('completed', 0)} / {row['attempted']} | {fmt(timing['mean'])} | {fmt(timing['p95'])} | {fmt(timing['max'])} | {failed} |")
    coverage = analysis['actual_continuation_verification']; replay = coverage['replayed_union']; cost = analysis['resources']
    next_step = ('Investigate the recorded operational failures before committing to a fresh strength sample.' if failures else
                 'Use the measured operational costs to prepare a separately authorized fresh strength protocol, fixed before matchup totals are opened.')
    report = '\n'.join([
        '# Operational checks passed: remaining nine pairs', '',
        'Automatically generated from the accepted independent analysis; final human-facing synthesis may follow.', '',
        'Automatic finalization additionally requires `finalization_validation.json` to say `passed` and no `coordinator_hold.json` to exist.', '',
        'The fixed eighteen-game stage (nine swapped-seat pairs, three per fit) completed its operational checks. '
        'Matchup outcomes remain closed. These games contribute zero strength observations; the earlier six-game pilot remains a separate stage.', '',
        '| Player | Completed / attempted decisions | Mean seconds | 95th percentile | Maximum seconds | Recorded player failures |',
        '|---|---:|---:|---:|---:|---:|', *rows, '',
        f"All {analysis['validation']['requests']} transfer/new-context validation requests and {analysis['validation']['exact_hidden_state_repeat_pairs']} exact privacy pairs passed.", '',
        f"Game replay covered {coverage['roots']} local roots: {coverage['full_stress_roots']} full stress roots and {coverage['sample_only_roots']} sample-only roots. "
        f"Of {coverage['recorded_worlds']} recorded worlds and {coverage['recorded_continuations']} recorded continuations, "
        f"the unique replay union contains {replay['worlds']} worlds, {replay['continuations']} continuations and {replay['transitions']} transitions. "
        f"Stress coverage was {coverage['covered_cells']} of {coverage['cells']} prescribed cells; absent cells remain untested.", '',
        f"The runner took {cost['production_elapsed_seconds']/3600:.3f} hours. The analyzer reported {cost['including_final_analysis_seconds']/3600:.3f} hours from the original launch. "
        f"Peak sampled worker RSS was {cost['peak_sampled_combined_rss_bytes']/1024**2:.1f} MiB; evidence before analysis was {cost['evidence_bytes_before_this_receipt']/1024**3:.3f} GiB.", '',
        'The 120-second move allowance, original 24-hour cap and final reserve were retained. Finite, correlated game decisions do not establish future tail-latency reliability or playing strength. '
        'Unsampled continuations received structural/arithmetic checks, not full replay. Local replay shares frozen model, feature and support kernels. '
        'Native search edges and neural outputs retain the qualified trust boundary. Resource telemetry is sampled.', '',
        f"The frozen analyzer attests that all {analysis['source_preservation']['files']} bound inputs matched and its recorded worker identities were closed. "
        'The coordinator additionally checked runner and its own analyzer-process closure; it makes no broader process-closure claim.', '',
        next_step+' No additional games, training, larger-data study or outcome opening is authorized by this report.', '',
        'Evidence: `outputs/live_game_operational_20261019_01/independent_analysis.json`, `independent_analysis_execution_01.json`, '
        '`runner_execution.json`, `coordinator_plan.json`, `coordinator_launch.json`, `finalization.json` and `finalization_validation.json`.', ''])
    return report


class Coordinator:
    def __init__(self, expected_source, expected_freeze, expected_plan):
        self.expected_source, self.expected_freeze, self.expected_plan = expected_source, expected_freeze, expected_plan
        self.claimed = False

    def authenticate(self):
        require(sha(SOURCE) == self.expected_source and self.expected_freeze == FREEZE_SHA256 and
                sha(BASE/'freeze.json') == self.expected_freeze and sha(BASE/'coordinator_plan.json') == self.expected_plan,
                'coordinator_authentication')
        plan = read(BASE/'coordinator_plan.json'); require(plan == make_plan(), 'prospective_plan_identity')
        return plan

    def analyze(self, plan):
        paths = [BASE/name for name in ('independent_analysis.json', 'independent_analysis_execution_01.log',
            'independent_analysis_execution_01.json', 'sealed/independent_analysis_failure_01.json')]
        require(all(not p.exists() for p in paths), 'analysis_already_attempted')
        self.authenticate(); resources()
        started = time.time(); child = None; identity = None; code = None; error = None; closed = False; diagnostic = None
        descendants = {}; cleanup_error = None
        with paths[1].open('xb') as log:
            try:
                child = subprocess.Popen(plan['analyzer_command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONPATH': str(ROOT/'src')+os.pathsep+str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'},
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                identity = {'pid': child.pid, 'created': psutil.Process(child.pid).create_time()}
                while child.poll() is None:
                    resources()
                    require(paths[1].stat().st_size <= MAX_LOG, 'analysis_log_limit')
                    try:
                        process = psutil.Process(identity['pid'])
                        require(not remember_descendants(process, identity, descendants), 'unexpected_analyzer_descendants')
                        require(process.memory_info().rss <= RSS_LIMIT, 'analyzer_rss_limit')
                    except psutil.NoSuchProcess:
                        require(child.poll() is not None, 'analyzer_identity_unavailable_while_running')
                        break
                    try: child.wait(timeout=min(1., max(.001, remaining()-RESERVE)))
                    except subprocess.TimeoutExpired: pass
                code = child.returncode
                resources()
                require(paths[1].stat().st_size <= MAX_LOG, 'analysis_log_limit')
            except BaseException as exc:
                error = type(exc).__name__
                diagnostic = BASE/'sealed/coordinator_analyzer_failure_01.json'
                details = traceback.format_exc()
                try:
                    if child is not None: code = close_owned(child, identity, descendants)
                except BaseException as cleanup:
                    cleanup_error = type(cleanup).__name__; details += '\nCleanup failure:\n'+traceback.format_exc()
                finally: save(diagnostic, {'time': now(), 'traceback': details, 'outcomes_closed': True})
            finally:
                closed = confirmed_closed(identity)
        ended = time.time()
        execution = {'command': plan['analyzer_command'], 'started_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(),
            'finished_utc': now(), 'elapsed_seconds': ended-started, 'original_launch_epoch_seconds': EPOCH,
            'elapsed_from_original_launch_seconds': ended-EPOCH, 'final_reserve_seconds': RESERVE,
            'exit_code': code, 'error': error, 'analyzer_process': identity, 'analyzer_process_closed': closed,
            'owned_descendants': list(descendants.values()), 'owned_descendants_closed': all(confirmed_closed(v) for v in descendants.values()),
            'cleanup_error': cleanup_error,
            'diagnostic': artifact(diagnostic) if diagnostic else None,
            'frozen_analyzer': plan['analyzer'], 'log': artifact(paths[1]), 'output': artifact(paths[0]) if paths[0].exists() else None,
            'coordinator_launch': artifact(BASE/'coordinator_launch.json'), 'outcomes_closed': True}
        save(paths[2], execution)
        require(code == 0 and error is None and cleanup_error is None and closed and execution['owned_descendants_closed'] and paths[0].exists(),
                'independent_analysis_failed')
        return read(paths[0])

    def run(self):
        psutil.Process().cpu_affinity([CPU])
        plan = self.authenticate()
        require(not REPORT.exists() and not (BASE/'finalization.json').exists() and not (BASE/'finalization_validation.json').exists(),
                'finalization_already_exists')
        save(BASE/'coordinator_launch.json', {'time': now(), 'source': artifact(SOURCE), 'plan': artifact(BASE/'coordinator_plan.json'),
            'expected_source_sha256': self.expected_source, 'expected_freeze_sha256': self.expected_freeze,
            'expected_plan_sha256': self.expected_plan, 'argv': sys.argv,
            'identity': {'pid': os.getpid(), 'created': psutil.Process().create_time()}, 'cpu': CPU,
            'original_epoch_seconds': EPOCH, 'original_monotonic_start': MONOTONIC, 'maximum_seconds': CAP,
            'final_reserve_seconds': RESERVE, 'outcomes_closed': True})
        self.claimed = True
        while not (BASE/'runner_execution.json').exists():
            resources(); time.sleep(min(POLL, max(.001, remaining()-RESERVE)))
        resources(); check_runner(delivered_runner_execution())
        analysis = self.analyze(plan); report = operational_report(analysis)
        self.authenticate(); require(not alive(RUNNER), 'runner_reappeared'); check = resources(reserve=0)
        require(analysis['freeze_sha256'] == self.expected_freeze and analysis['source_preservation']['all_hashes_and_sizes_match'] is True,
                'accepted_source_attestation')
        with REPORT.open('x', encoding='utf-8', newline='\n') as stream: stream.write(report)
        check = resources(reserve=0)
        records = [BASE/name for name in ('independent_analysis.json', 'runner_execution.json', 'runner_process.json',
            'independent_analysis_execution_01.json', 'independent_analysis_execution_01.log', 'coordinator_plan.json', 'coordinator_launch.json')]
        records += [REPORT, SOURCE, ANALYZER, BASE/'freeze.json', BASE/'launch.json', BASE/'completion.json', BASE/'final_input_check.json']
        final = {'schema': 'operational_automatic_finalization_v1', 'status': 'operational_checks_passed_pending_finalization_validation',
            'time': now(), 'automatic_report': True, 'games': 18, 'pairs': 9, 'outcomes_closed': True, 'strength_analysis_games': 0,
            'larger_data_deferred': True, 'verified_actual_transitions': analysis['operational_schedule']['actual_dictionary_transitions'],
            'attempted_actual_decisions': sum(analysis['per_stage'][key]['attempted'] for key in ('games:local', 'games:opponent')),
            'acceptance_requires': 'finalization_validation.json status passed and coordinator_hold.json absent',
            'additional_execution_authorized': False, 'handoffs_rewritten': False, 'artifacts': [artifact(p) for p in records],
            'accepted_analysis_sha256': sha(BASE/'independent_analysis.json'), 'source_preservation_attestation': analysis['source_preservation'],
            'attestation_scope': 'Frozen independent analyzer checked bound input hashes and recorded worker closure; coordinator checked only runner and its owned analyzer.',
            'runner_closed': True, 'analyzer_closed': True, 'resources': check}
        save(BASE/'finalization.json', final)
        require(read(BASE/'finalization.json') == final and all(artifact(ROOT/item['path']) == item for item in final['artifacts']),
                'finalization_artifact_recheck')
        checked = resources(reserve=0)
        save(BASE/'finalization_validation.json', {'status': 'passed', 'time': now(), 'finalization': artifact(BASE/'finalization.json'),
            'bound_artifacts_rechecked': len(final['artifacts']), 'resources_including_finalization': checked, 'outcomes_closed': True,
            'acceptance_invalid_if_coordinator_hold_exists': True})
        resources(reserve=0)

    def preserve_hold(self):
        if not self.claimed: return
        failure = BASE/'sealed/coordinator_failure_01.json'
        save(failure, {'time': now(), 'traceback': traceback.format_exc(), 'outcomes_closed': True})
        save(BASE/'coordinator_hold.json', {'status': 'preserved_hold', 'time': now(), 'diagnostic': artifact(failure),
            'coordinator_launch': artifact(BASE/'coordinator_launch.json'), 'outcomes_closed': True, 'automatic_retries': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-source-sha256', required=True); parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--plan-sha256', required=True); args = parser.parse_args()
    coordinator = Coordinator(args.expected_source_sha256, args.expected_freeze_sha256, args.plan_sha256)
    try: coordinator.run()
    except BaseException as error:
        preserved = False
        try: coordinator.preserve_hold(); preserved = coordinator.claimed
        except BaseException: pass
        print(json.dumps({'status': 'preserved_hold', 'category': type(error).__name__, 'diagnostic_preserved': preserved,
                          'outcomes_closed': True}), flush=True)
        return 1
    print(json.dumps({'status': 'operational_checks_passed', 'outcomes_closed': True, 'games': 18, 'pairs': 9}), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
