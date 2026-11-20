"""One-shot, model-free completion of the separately authorized operational recovery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
BASE = ROOT/'outputs/live_game_operational_recovery_20261020_02'
PRIOR_BASE = ROOT/'outputs/live_game_operational_recovery_20261020_01'
ORIGINAL_BASE = ROOT/'outputs/live_game_operational_20261019_01'
SOURCE = Path(__file__).resolve()
ANALYZER = ROOT/'tools/analyze_live_game_operational_recovery_20261020_02.py'
REPORT = ROOT/'docs/engineering/live_game_operational_recovery_20261020_02_result.md'
ORIGINAL_FREEZE_SHA256 = '21ca9cb617ca65ad9745eeed76a2f5f91153dd638981c7fbfd76f4e348a27c51'
PRIOR_FREEZE_SHA256 = '3af87db0e61a9d014b4ec60e8b031f9a36a2b6f90214b559f2485f60b75ae716'
RUNNER_MODULE = 'tools.live_game_operational_recovery_20261020_02.runner'
MEMORY_MODULE = 'tools.live_game_operational_recovery_20261020_02.memory'
MEMORY_SOURCE = ROOT/'tools/live_game_operational_recovery_20261020_02/memory.py'
COMPLETED_GAMES = [6, 8, 10, 7, 9, 11, 12, 14, 16, 13, 15, 17, 18, 20, 22, 19]
CAP, RESERVE, CPU, POLL = 43200, 600, 2, 30
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
def remaining(plan):
    return min(plan['recovery_epoch_seconds']+CAP-time.time(),
               plan['recovery_monotonic_start']+CAP-time.monotonic())


def remember_descendants(process, owner, identities, runner):
    require(process.create_time() == owner['created'], 'analyzer_identity_changed')
    children = process.children(recursive=True)
    for child in children:
        try: identity = {'pid': child.pid, 'created': child.create_time()}
        except psutil.NoSuchProcess: continue
        require(identity['created'] >= owner['created'] and identity['pid'] not in (runner['pid'], os.getpid(), owner['pid']),
                'invalid_analyzer_descendant_identity')
        identities[(identity['pid'], identity['created'])] = identity
    return children


def close_owned(child, owner, descendants, runner):
    """Only the authenticated Popen process and its observed descendants are owned."""
    errors = []
    if owner is not None and child.poll() is None:
        try: remember_descendants(psutil.Process(owner['pid']), owner, descendants, runner)
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


def resources(plan, *, reserve=RESERVE):
    left = remaining(plan); size = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
    free = shutil.disk_usage(BASE).free
    require(left > reserve, 'recovery_deadline_reserve')
    require(size < MAX_BYTES-32*1024**2, 'evidence_reserve')
    require(free >= MIN_FREE, 'disk_reserve')
    return {'remaining_seconds': left, 'evidence_bytes': size, 'evidence_cap_bytes': MAX_BYTES,
            'disk_free_bytes': free, 'minimum_free_bytes': MIN_FREE, 'final_reserve_seconds': RESERVE}


def analyzer_command(freeze):
    return [freeze['executable'], '-B', ANALYZER.relative_to(ROOT).as_posix(),
            '--base', BASE.relative_to(ROOT).as_posix(), '--output', (BASE/'independent_analysis.json').relative_to(ROOT).as_posix()]


def identity(value):
    require(isinstance(value, dict) and set(value) == {'pid', 'created'} and
            type(value['pid']) is int and value['pid'] > 0 and
            type(value['created']) in (int, float) and math.isfinite(value['created']) and
            value['created'] > 0, 'process_identity_schema')
    return value


def make_plan(expected_freeze_sha256, expected_source_sha256):
    """Build only from the authenticated new freeze and its already-started runner."""
    require(sha(BASE/'freeze.json') == expected_freeze_sha256 and
            sha(SOURCE) == expected_source_sha256, 'experiment_freeze_or_source_identity')
    freeze = read(BASE/'freeze.json'); launch = read(BASE/'launch.json'); process = read(BASE/'runner_process.json')
    require(freeze['schema'] == 'live_operational_recovery02_freeze_v1', 'recovery_freeze_schema')
    limits = {'execution_cap_seconds': CAP, 'final_reserve_seconds': RESERVE,
              'evidence_cap_bytes': MAX_BYTES, 'minimum_free_bytes': MIN_FREE,
              'max_process_tree_rss_bytes': RSS_LIMIT, 'outcomes_closed': True, 'strength_analysis_games': 0}
    require(all(type(freeze.get(key)) is type(value) and freeze[key] == value for key, value in limits.items()),
            'recovery_freeze_resource_contract')
    inputs = {v['path']: v for v in freeze['inputs']}
    require(len(inputs) == len(freeze['inputs']), 'duplicate_frozen_path')

    def bound(path):
        value = artifact(path)
        require(inputs.get(value['path']) == value, 'frozen_recovery_input_identity')
        return value

    original_freeze = bound(ORIGINAL_BASE/'freeze.json')
    prior_freeze = bound(PRIOR_BASE/'freeze.json')
    require(original_freeze['sha256'] == ORIGINAL_FREEZE_SHA256 and
            prior_freeze['sha256'] == PRIOR_FREEZE_SHA256 and freeze['source_freeze'] == prior_freeze and
            freeze['original_freeze'] == original_freeze, 'preserved_three_segment_freeze_identity')
    recovery_plan = bound(BASE/'recovery_plan.json'); source_snapshot = bound(BASE/'source_snapshot.json')
    recovery = read(BASE/'recovery_plan.json')
    require(freeze['recovery_plan'] == recovery_plan and recovery['source_snapshot'] == source_snapshot and
            recovery['schema'] == 'live_operational_recovery02_plan_v1' and
            recovery['source_base'] == PRIOR_BASE.relative_to(ROOT).as_posix() and
            recovery['source_freeze'] == prior_freeze and recovery['original_freeze'] == original_freeze and
            recovery['completed_games'] == COMPLETED_GAMES and
            recovery['partial_game']['game'] == 21 and recovery['partial_game']['next_ply'] == 11 and
            recovery['validation_reused'] is True, 'authorized_recovery_plan_identity')
    original_launch = bound(ORIGINAL_BASE/'launch.json')
    original_complete = bound(ORIGINAL_BASE/'completion.json')
    original_execution = bound(ORIGINAL_BASE/'runner_execution.json')
    original_hold = bound(ORIGINAL_BASE/'coordinator_hold.json')
    bound(ORIGINAL_BASE/'final_input_check.json')
    old = read(ORIGINAL_BASE/'completion.json'); old_execution = read(ORIGINAL_BASE/'runner_execution.json')
    old_launch = read(ORIGINAL_BASE/'launch.json')
    require(old['status'] == 'preserved_hold' and old['completed_games'] == 5 and
            old['failure']['category'] == 'ResourceHold' and old['validation_passed'] is True and
            old['worker_processes_closed'] is True and old['outcomes_closed'] is True and
            old_execution['exit_code'] == 1 and old_execution['outcomes_closed'] is True and
            read(ORIGINAL_BASE/'coordinator_hold.json')['status'] == 'preserved_hold' and
            read(ORIGINAL_BASE/'final_input_check.json')['all_hashes_and_sizes_match'] is True,
            'original_attempt_not_preserved_closed_hold')
    require(not alive(identity(old_execution['runner_identity'])), 'original_runner_not_closed')
    prior_launch = bound(PRIOR_BASE/'launch.json')
    prior_complete = bound(PRIOR_BASE/'completion.json')
    prior_execution = bound(PRIOR_BASE/'runner_execution.json')
    prior_hold = bound(PRIOR_BASE/'coordinator_hold.json')
    bound(PRIOR_BASE/'final_input_check.json')
    previous = read(PRIOR_BASE/'completion.json'); previous_execution = read(PRIOR_BASE/'runner_execution.json')
    previous_launch = read(PRIOR_BASE/'launch.json')
    require(previous['status'] == 'preserved_hold' and previous['completed_games'] == 16 and
            previous['imported_completed_games'] == 5 and previous['validation_reused'] is True and
            previous['failure']['category'] == 'ResourceHold' and previous['worker_processes_closed'] is True and
            previous['outcomes_closed'] is True and type(previous_execution['exit_code']) is int and
            previous_execution['exit_code'] == 1 and previous_execution['outcomes_closed'] is True and
            read(PRIOR_BASE/'coordinator_hold.json')['status'] == 'preserved_hold' and
            read(PRIOR_BASE/'final_input_check.json')['all_hashes_and_sizes_match'] is True and
            not any((PRIOR_BASE/name).exists() for name in
                    ('audit_manifest.json', 'independent_analysis.json', 'independent_analysis_execution_01.json',
                     'finalization.json', 'finalization_validation.json')),
            'prior_recovery_not_preserved_unaudited_hold')
    require(not alive(identity(previous_execution['runner_identity'])), 'prior_recovery_runner_not_closed')
    runner = identity(process['identity'])
    require(all(type(launch[k]) in (int, float) and math.isfinite(launch[k]) and launch[k] > 0
                for k in ('epoch_seconds', 'monotonic_start')) and
            launch['maximum_seconds'] == CAP and launch['pid'] == runner['pid'] and
            runner['created'] <= launch['epoch_seconds'] and launch['outcomes_closed'] is True and
            launch['freeze'] == artifact(BASE/'freeze.json') and launch['recovery_plan'] == recovery_plan and
            launch['source_launch'] == prior_launch and launch['validation_reused'] is True and
            launch['imported_completed_games'] == 16, 'recovery_launch_identity')
    expected_runner = [freeze['executable'], '-B', '-u', '-m', RUNNER_MODULE, 'run']
    require(process['command'] == expected_runner and
            process['freeze_sha256'] == expected_freeze_sha256 and process['outcomes_closed'] is True,
            'runner_process_identity')
    frozen_source, frozen_analyzer = bound(SOURCE), bound(ANALYZER)
    frozen_memory = bound(MEMORY_SOURCE)
    return {'schema': 'operational_recovery02_completion_plan_v1', 'source': frozen_source,
        'experiment_freeze': artifact(BASE/'freeze.json'), 'experiment_launch': artifact(BASE/'launch.json'),
        'recovery_plan': recovery_plan, 'source_snapshot': source_snapshot,
        'runner_process': artifact(BASE/'runner_process.json'), 'runner_identity': runner, 'runner_command': expected_runner,
        'analyzer': frozen_analyzer, 'analyzer_command': analyzer_command(freeze),
        'preserved_original': {'freeze': original_freeze, 'launch': original_launch, 'completion': original_complete,
                              'execution': original_execution, 'coordinator_hold': original_hold},
        'preserved_prior_recovery': {'freeze': prior_freeze, 'launch': prior_launch, 'completion': prior_complete,
                                    'execution': prior_execution, 'coordinator_hold': prior_hold},
        'original_epoch_seconds': old_launch['epoch_seconds'],
        'original_production_elapsed_seconds': old['elapsed_seconds'],
        'first_recovery_epoch_seconds': previous_launch['epoch_seconds'],
        'first_recovery_production_elapsed_seconds': previous['elapsed_seconds'],
        'recovery_epoch_seconds': launch['epoch_seconds'], 'recovery_monotonic_start': launch['monotonic_start'],
        'maximum_seconds': CAP,
        'final_reserve_seconds': RESERVE, 'cpu': CPU, 'poll_seconds': POLL, 'evidence_cap_bytes': MAX_BYTES,
        'minimum_free_bytes': MIN_FREE, 'report': REPORT.relative_to(ROOT).as_posix(), 'outcomes_closed': True,
        'maximum_analysis_log_bytes': MAX_LOG,
        'analysis_memory_admission': {'module': MEMORY_MODULE, 'source': frozen_memory, 'name': 'analysis',
             'reserve': RESERVE, 'request_seconds': 0, 'maximum_episode_seconds': 7200,
             'maximum_cumulative_seconds': 14400, 'physical_bytes': 2*1024**3, 'commit_bytes': 4*1024**3,
             'stable_samples': 13, 'sample_interval_seconds': 10},
        'analysis_attempts': 1, 'automatic_retries': 0, 'automatic_handoff_updates': False}


def check_runner(execution, plan):
    process = read(BASE/'runner_process.json'); complete = read(BASE/'completion.json')
    require(execution['runner_identity'] == process['identity'] == plan['runner_identity'] and
            execution['command'] == process['command'] and execution['outcomes_closed'] is True and
            type(execution['exit_code']) is int and execution['exit_code'] == 0, 'runner_execution_failed_or_mismatched')
    require(not alive(plan['runner_identity']), 'runner_not_closed')
    require(complete['status'] == 'completed_pending_independent_analysis' and complete['failure'] is None and
            complete['completed_games'] == 18 and complete['worker_processes_closed'] is True and
            complete['outcomes_closed'] is True and complete['validation_reused'] is True and
            complete['imported_completed_games'] == 16, 'runner_not_complete')


def delivered_runner_execution(plan):
    """Allow only the exclusive writer's short, incomplete-JSON delivery window."""
    deadline = time.monotonic()+RECEIPT_DELIVERY_SECONDS
    while True:
        try: return read(BASE/'runner_execution.json')
        except json.JSONDecodeError:
            resources(plan)
            left = deadline-time.monotonic()
            require(left > 0, 'runner_execution_delivery_incomplete')
            time.sleep(min(.1, left, max(.001, remaining(plan)-RESERVE)))


def analyzer_admission(plan):
    """RAM waiting occurs before launching the analyzer, inside the same fresh cap."""
    from tools.live_game_operational_recovery_20261020_02.memory import wait_for_memory
    started = time.monotonic()
    deadline = min(plan['recovery_monotonic_start']+CAP, started+remaining(plan))
    value = wait_for_memory(BASE, deadline, name='analysis', reserve=RESERVE, request_seconds=0)
    path = BASE/'memory_admissions/analysis.json'
    log = BASE/'memory_admissions/analysis.jsonl'
    require(read(path) == value and value['schema'] == 'operational_memory_admission_v2' and
            value['status'] == 'passed' and value['name'] == 'analysis' and
            value['reserve_seconds'] == RESERVE and value['request_seconds'] == 0 and
            value['initial'] is False and value['no_worker_during_admission'] is True and
            value['outcomes_closed'] is True and
            value['readings_log'] == {'path': log.relative_to(BASE).as_posix(),
                                     'sha256': sha(log), 'bytes': log.stat().st_size},
            'analysis_memory_admission_identity')
    return {'receipt': artifact(path), 'readings_log': artifact(log),
            'charged_seconds': value['charged_seconds'], 'cumulative_after_seconds': value['cumulative_after_seconds']}


def operational_report(analysis, plan):
    """Only the frozen analyzer's operational field allowlist enters this report."""
    require(analysis['status'] == 'passed' and analysis['outcomes_closed'] is True and
            type(analysis['strength_analysis_games']) is int and analysis['strength_analysis_games'] == 0 and
            analysis['operational_schedule']['games'] == 18 and analysis['operational_schedule']['pairs'] == 9 and
            analysis['larger_data_deferred'] is True,
            'accepted_analysis_contract')
    require(analysis['per_stage']['games:local']['statuses'].get('forfeit_timeout', 0) >= 4,
            'preserved_original_timeouts_missing')
    require(analysis['validation']['requests'] == 12 and
            analysis['validation']['exact_hidden_state_repeat_pairs'] == 6 and
            analysis['validation']['reused_from_first_attempt'] is True and
            analysis['validation']['new_validation_requests'] == 0, 'reused_validation_contract')
    lineage = analysis['first_attempt_lineage']; prior = analysis['prior_recovery_lineage']
    require(lineage['original_attempt_status'] == 'preserved_hold' and
            lineage['original_source_and_evidence_unchanged'] is True and
            prior['completed_games_reused'] == COMPLETED_GAMES and prior['partial_game'] == 21 and
            prior['next_unstarted_ply'] == 11 and prior['prior_recovery_status'] == 'preserved_hold' and
            prior['prior_recovery_source_and_evidence_unchanged'] is True and
            analysis['scope']['original_timeout_forfeits_preserved'] == 4 and
            analysis['scope']['unstarted_hold_not_scored_as_player_failure'] is True, 'recovery_analysis_lineage')
    rows = []; failures = 0
    for key, label in (('games:local', 'Local planner'), ('games:opponent', 'SplendoRust')):
        row = analysis['per_stage'][key]; counts = row['statuses']; timing = row['completed_receipt_seconds']
        require(set(counts) <= ALLOWED_STATUSES and all(type(n) is int and n >= 0 for n in counts.values()), 'operational_status_contract')
        failed = sum(n for name, n in counts.items() if name != 'completed'); failures += failed
        fmt = lambda value: 'n/a' if value is None else f'{value:.3f}'
        rows.append(f"| {label} | {counts.get('completed', 0)} / {row['attempted']} | {fmt(timing['mean'])} | {fmt(timing['p95'])} | {fmt(timing['max'])} | {failed} |")
    coverage = analysis['actual_continuation_verification']; replay = coverage['replayed_union']; cost = analysis['resources']
    require(cost['original_attempt_elapsed_seconds'] == plan['original_production_elapsed_seconds'] and
            cost['first_recovery_elapsed_seconds'] == plan['first_recovery_production_elapsed_seconds'] and
            cost['sum_attempt_execution_seconds_including_analysis'] ==
            cost['original_attempt_elapsed_seconds']+cost['first_recovery_elapsed_seconds']+cost['including_final_analysis_seconds'] and
            cost['between_attempt_wall_gap_not_charged_to_fresh_cap'] is True, 'separate_execution_costs')
    waiting = analysis['memory_waiting']
    require(type(waiting['admissions']) is int and type(waiting['stability_required_admissions']) is int and
            1 <= waiting['stability_required_admissions'] <= waiting['admissions'] and
            type(waiting['charged_seconds']) in (int, float) and math.isfinite(waiting['charged_seconds']) and
            0 <= waiting['charged_seconds'] < 14400 and
            waiting['charged_seconds'] <= cost['including_final_analysis_seconds'], 'charged_memory_waiting')
    next_step = ('Investigate the recorded operational failures before committing to a fresh strength sample.' if failures else
                 'Use the measured operational costs to prepare a separately authorized fresh strength protocol, fixed before matchup totals are opened.')
    report = '\n'.join([
        '# Operational recovery completed with recorded timing failures', '',
        'Automatically generated from the accepted independent analysis; final human-facing synthesis may follow.', '',
        'Automatic finalization additionally requires `finalization_validation.json` to say `passed` and no `coordinator_hold.json` to exist.', '',
        'The fixed eighteen-game stage (nine swapped-seat pairs, three per fit) completed its operational checks across three separately recorded execution segments. '
        'The original attempt and first recovery stopped on host-memory admission safeguards. This second recovery preserved sixteen completed games and resumed the unstarted request in game 21 before the final game and combined audits. '
        'The four earlier player timeouts remain in the combined record; no completed game or forfeit was replaced. '
        'Matchup outcomes remain closed. These games contribute zero strength observations; the earlier six-game pilot remains a separate stage.', '',
        'Verification completion does not qualify either player as failure-free; the local planner exceeded 120 seconds at least four times.', '',
        '| Player | Completed / attempted decisions | Mean seconds | 95th percentile | Maximum seconds | Recorded player failures |',
        '|---|---:|---:|---:|---:|---:|', *rows, '',
        f"The original {analysis['validation']['requests']} passed transfer/new-context validation requests and "
        f"{analysis['validation']['exact_hidden_state_repeat_pairs']} exact privacy pairs were authenticated and reused; they were not rerun.", '',
        f"Game replay covered {coverage['roots']} local roots: {coverage['full_stress_roots']} full stress roots and {coverage['sample_only_roots']} sample-only roots. "
        f"Of {coverage['recorded_worlds']} recorded worlds and {coverage['recorded_continuations']} recorded continuations, "
        f"the unique replay union contains {replay['worlds']} worlds, {replay['continuations']} continuations and {replay['transitions']} transitions. "
        f"Stress coverage was {coverage['covered_cells']} of {coverage['cells']} prescribed cells; absent cells remain untested.", '',
        f"The preserved original production segment took {plan['original_production_elapsed_seconds']/3600:.3f} hours. "
        f"The preserved first recovery took {plan['first_recovery_production_elapsed_seconds']/3600:.3f} hours. "
        f"The second recovery production segment took {cost['production_elapsed_seconds']/3600:.3f} hours; "
        f"including its independent analysis, the second recovery took {cost['including_final_analysis_seconds']/3600:.3f} hours. "
        f"The sum of those active segments, excluding the gap between attempts, was "
        f"{cost['sum_attempt_execution_seconds_including_analysis']/3600:.3f} hours. "
        f"Peak sampled worker RSS was {cost['peak_sampled_combined_rss_bytes']/1024**2:.1f} MiB; evidence before analysis was {cost['evidence_bytes_before_this_receipt']/1024**3:.3f} GiB.", '',
        f"Memory admission used {waiting['admissions']} gates, of which {waiting['stability_required_admissions']} required a stable window. "
        f"Their total charged time was {waiting['charged_seconds']/3600:.3f} hours, already included in the second recovery's elapsed time and 12-hour cap. "
        'Waiting happened before workers started; active player clocks were never paused or retried.', '',
        'The 120-second move allowance was retained. This explicitly authorized second recovery had its own 12-hour cap and 600-second final reserve; '
        'both earlier attempts remain preserved holds under their original limits. No elapsed time was erased or reassigned. '
        'Finite, correlated game decisions do not establish future tail-latency reliability or playing strength. '
        'Unsampled continuations received structural/arithmetic checks, not full replay. Local replay shares frozen model, feature and support kernels. '
        'Native search edges and neural outputs retain the qualified trust boundary. Resource telemetry is sampled.', '',
        f"The frozen analyzer attests that all {analysis['source_preservation']['files']} bound inputs matched and its recorded worker identities were closed. "
        'The coordinator additionally checked runner and its own analyzer-process closure; it makes no broader process-closure claim.', '',
        next_step+' No additional games, training, larger-data study or outcome opening is authorized by this report.', '',
        'Evidence: `outputs/live_game_operational_recovery_20261020_02/independent_analysis.json`, `independent_analysis_execution_01.json`, '
        '`runner_execution.json`, `coordinator_plan.json`, `coordinator_launch.json`, `finalization.json` and `finalization_validation.json`.', ''])
    return report


class Coordinator:
    def __init__(self, expected_source, expected_freeze, expected_plan):
        self.expected_source, self.expected_freeze, self.expected_plan = expected_source, expected_freeze, expected_plan
        self.claimed = False

    def authenticate(self):
        require(sha(SOURCE) == self.expected_source and
                sha(BASE/'freeze.json') == self.expected_freeze and sha(BASE/'coordinator_plan.json') == self.expected_plan,
                'coordinator_authentication')
        plan = read(BASE/'coordinator_plan.json')
        require(plan == make_plan(self.expected_freeze, self.expected_source), 'prospective_plan_identity')
        return plan

    def analyze(self, plan):
        paths = [BASE/name for name in ('independent_analysis.json', 'independent_analysis_execution_01.log',
            'independent_analysis_execution_01.json', 'sealed/independent_analysis_failure_01.json')]
        require(all(not p.exists() for p in paths), 'analysis_already_attempted')
        require(plan == self.authenticate(), 'analysis_plan_identity'); resources(plan)
        admission = analyzer_admission(plan)
        require(plan == self.authenticate(), 'analysis_plan_identity'); resources(plan)
        started = time.time(); child = None; identity = None; code = None; error = None; closed = False; diagnostic = None
        descendants = {}; cleanup_error = None
        with paths[1].open('xb') as log:
            try:
                child = subprocess.Popen(plan['analyzer_command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONPATH': str(ROOT/'src')+os.pathsep+str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'},
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                identity = {'pid': child.pid, 'created': psutil.Process(child.pid).create_time()}
                while child.poll() is None:
                    resources(plan)
                    require(paths[1].stat().st_size <= MAX_LOG, 'analysis_log_limit')
                    try:
                        process = psutil.Process(identity['pid'])
                        require(not remember_descendants(process, identity, descendants, plan['runner_identity']),
                                'unexpected_analyzer_descendants')
                        require(process.memory_info().rss <= RSS_LIMIT, 'analyzer_rss_limit')
                    except psutil.NoSuchProcess:
                        require(child.poll() is not None, 'analyzer_identity_unavailable_while_running')
                        break
                    try: child.wait(timeout=min(1., max(.001, remaining(plan)-RESERVE)))
                    except subprocess.TimeoutExpired: pass
                code = child.returncode
                resources(plan)
                require(paths[1].stat().st_size <= MAX_LOG, 'analysis_log_limit')
            except BaseException as exc:
                error = type(exc).__name__
                diagnostic = BASE/'sealed/coordinator_analyzer_failure_01.json'
                details = traceback.format_exc()
                try:
                    if child is not None: code = close_owned(child, identity, descendants, plan['runner_identity'])
                except BaseException as cleanup:
                    cleanup_error = type(cleanup).__name__; details += '\nCleanup failure:\n'+traceback.format_exc()
                finally: save(diagnostic, {'time': now(), 'traceback': details, 'outcomes_closed': True})
            finally:
                closed = confirmed_closed(identity)
        ended = time.time()
        execution = {'command': plan['analyzer_command'], 'started_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(),
            'finished_utc': now(), 'elapsed_seconds': ended-started,
            'recovery_launch_epoch_seconds': plan['recovery_epoch_seconds'],
            'elapsed_from_recovery_launch_seconds': ended-plan['recovery_epoch_seconds'], 'final_reserve_seconds': RESERVE,
            'exit_code': code, 'error': error, 'analyzer_process': identity, 'analyzer_process_closed': closed,
            'owned_descendants': list(descendants.values()), 'owned_descendants_closed': all(confirmed_closed(v) for v in descendants.values()),
            'cleanup_error': cleanup_error,
            'memory_admission': admission,
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
        require(not REPORT.exists() and all(not (BASE/name).exists() for name in
                ('finalization.json', 'finalization_validation.json', 'coordinator_hold.json',
                 'independent_analysis.json', 'independent_analysis_execution_01.log',
                 'independent_analysis_execution_01.json', 'sealed/independent_analysis_failure_01.json')),
                'finalization_already_exists')
        save(BASE/'coordinator_launch.json', {'time': now(), 'source': artifact(SOURCE), 'plan': artifact(BASE/'coordinator_plan.json'),
            'expected_source_sha256': self.expected_source, 'expected_freeze_sha256': self.expected_freeze,
            'expected_plan_sha256': self.expected_plan, 'argv': sys.argv,
            'identity': {'pid': os.getpid(), 'created': psutil.Process().create_time()}, 'cpu': CPU,
            'recovery_epoch_seconds': plan['recovery_epoch_seconds'],
            'recovery_monotonic_start': plan['recovery_monotonic_start'], 'maximum_seconds': CAP,
            'preserved_original': plan['preserved_original'],
            'preserved_prior_recovery': plan['preserved_prior_recovery'],
            'final_reserve_seconds': RESERVE, 'outcomes_closed': True})
        self.claimed = True
        while not (BASE/'runner_execution.json').exists():
            resources(plan); time.sleep(min(POLL, max(.001, remaining(plan)-RESERVE)))
        resources(plan); check_runner(delivered_runner_execution(plan), plan)
        analysis = self.analyze(plan); report = operational_report(analysis, plan)
        self.authenticate(); require(not alive(plan['runner_identity']), 'runner_reappeared'); check = resources(plan, reserve=0)
        require(analysis['freeze_sha256'] == self.expected_freeze and analysis['source_preservation']['all_hashes_and_sizes_match'] is True,
                'accepted_source_attestation')
        with REPORT.open('x', encoding='utf-8', newline='\n') as stream: stream.write(report)
        check = resources(plan, reserve=0)
        records = [BASE/name for name in ('independent_analysis.json', 'runner_execution.json', 'runner_process.json',
            'independent_analysis_execution_01.json', 'independent_analysis_execution_01.log', 'coordinator_plan.json', 'coordinator_launch.json')]
        records += [REPORT, SOURCE, ANALYZER, BASE/'freeze.json', BASE/'launch.json', BASE/'completion.json', BASE/'final_input_check.json']
        records += [ROOT/value['path'] for value in plan['preserved_original'].values()]
        records += [ROOT/value['path'] for value in plan['preserved_prior_recovery'].values()]
        records += [MEMORY_SOURCE, BASE/'memory_admissions/analysis.json', BASE/'memory_admissions/analysis.jsonl']
        final = {'schema': 'operational_recovery02_automatic_finalization_v1', 'status': 'operational_checks_passed_pending_finalization_validation',
            'time': now(), 'automatic_report': True, 'games': 18, 'pairs': 9, 'outcomes_closed': True, 'strength_analysis_games': 0,
            'larger_data_deferred': True, 'verified_actual_transitions': analysis['operational_schedule']['actual_dictionary_transitions'],
            'preserved_original': plan['preserved_original'], 'preserved_prior_recovery': plan['preserved_prior_recovery'],
            'imported_completed_games': 16, 'validation_reused': True, 'preserved_player_timeouts': 4,
            'memory_waiting': analysis['memory_waiting'],
            'attempted_actual_decisions': sum(analysis['per_stage'][key]['attempted'] for key in ('games:local', 'games:opponent')),
            'acceptance_requires': 'finalization_validation.json status passed and coordinator_hold.json absent',
            'additional_execution_authorized': False, 'handoffs_rewritten': False, 'artifacts': [artifact(p) for p in records],
            'accepted_analysis_sha256': sha(BASE/'independent_analysis.json'), 'source_preservation_attestation': analysis['source_preservation'],
            'attestation_scope': 'Frozen independent analyzer checked bound input hashes and recorded worker closure; coordinator checked only runner and its owned analyzer.',
            'runner_closed': True, 'analyzer_closed': True, 'resources': check}
        save(BASE/'finalization.json', final)
        require(read(BASE/'finalization.json') == final and all(artifact(ROOT/item['path']) == item for item in final['artifacts']),
                'finalization_artifact_recheck')
        checked = resources(plan, reserve=0)
        save(BASE/'finalization_validation.json', {'status': 'passed', 'time': now(), 'finalization': artifact(BASE/'finalization.json'),
            'bound_artifacts_rechecked': len(final['artifacts']), 'resources_including_finalization': checked, 'outcomes_closed': True,
            'acceptance_invalid_if_coordinator_hold_exists': True})
        resources(plan, reserve=0)

    def preserve_hold(self):
        if not self.claimed: return
        failure = BASE/'sealed/coordinator_failure_01.json'
        save(failure, {'time': now(), 'traceback': traceback.format_exc()[-16384:], 'outcomes_closed': True})
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
