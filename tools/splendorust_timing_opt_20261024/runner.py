"""One bounded qualification of a separate Windows process-sampling optimization."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
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
from . import common as c
from . import qualification as plan_rules
from .sampler import BatchedStudyWorker, snapshot_api_modules
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit, qualification as frozen_q, common as frozen
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs

MODES = ('baseline', 'candidate')
DOMAIN = 'splendorust_sampler_optimization_ops_20261024_v1'
DIAGNOSTIC = c.ROOT/'outputs/splendorust_timing_20261024_01'
PROBE = c.ROOT/'outputs/splendorust_sampler_probe_20261024_01'

def bytes_used():
    return sum(path.stat().st_size for path in c.OUT.rglob('*') if path.is_file())


class Envelope:
    """One non-resettable job clock, aggregate resident-worker resource bound."""
    def __init__(self, claim):
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.end = self.start + c.JOB_SECONDS
        self.stage = None
        self.deadline, self.wall_deadline = self.end, self.wall_start + c.JOB_SECONDS
        self.waited, self.admitted, self.admissions = 0., False, 0
        self.evidence = None
        self.resident = {}
        self.peak_combined_rss = 0

    def enable_evidence_accounting(self):
        self.evidence = dict(baseline=bytes_used(), parent_bytes=0, workers={})
        self.log_baseline = {name: (c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                             for name in ('runner_stdout.log', 'runner_stderr.log')}
        c.WRITE_OBSERVER = self.note_write

    def note_write(self, path, size):
        if not path.resolve().is_relative_to((c.OUT/'workers').resolve()):
            self.evidence['parent_bytes'] += size

    def register_worker(self, folder):
        self.evidence['workers'][folder] = None

    def own_worker(self, child):
        proc = psutil.Process(child.process.pid)
        identity = (proc.pid, proc.create_time())
        c.require(identity not in self.resident, 'Duplicate resident worker identity')
        self.resident[identity] = proc

    def close_worker(self, folder, child=None):
        if child is not None:
            identities = [identity for identity in self.resident if identity[0] == child.process.pid]
            for identity in identities:
                self.resident.pop(identity)
        self.evidence['workers'][folder] = sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())

    def aggregate_worker_rss(self, exclude_pid=None, additional_rss=0):
        seen, total = set(), additional_rss
        for identity, root in self.resident.items():
            if identity[0] == exclude_pid:
                continue
            try:
                c.require(root.create_time() == identity[1], 'Resident root PID identity changed')
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                raise ValueError('Resident worker disappeared before owned cleanup')
            for process in processes:
                try:
                    child_identity = (process.pid, process.create_time())
                    if child_identity not in seen:
                        total += process.memory_info().rss
                        seen.add(child_identity)
                except psutil.NoSuchProcess:
                    # Helpers may exit between enumeration and observation; the
                    # owning StudyWorker independently checks required roles.
                    continue
        self.peak_combined_rss = max(self.peak_combined_rss, total)
        c.require(total <= 2*c.GiB, 'Combined resident workers/helpers RSS exceeded 2GiB')
        return total

    def guard_worker_sample(self, child, telemetry):
        """Include a new tree before its constructor has returned readiness."""
        self.aggregate_worker_rss(exclude_pid=child.process.pid, additional_rss=telemetry['rss_bytes'])

    def accounted_bytes(self):
        if self.evidence is None:
            return bytes_used()
        total = self.evidence['baseline'] + self.evidence['parent_bytes']
        for folder, closed in self.evidence['workers'].items():
            total += closed if closed is not None else sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
        for name, initial in self.log_baseline.items():
            path = c.OUT/name
            total += max(0, (path.stat().st_size if path.exists() else 0)-initial)
        return total + 65536

    def begin(self, stage):
        mono, wall = time.monotonic(), time.time()
        self.stage = stage
        cap = c.STAGES[stage]
        cutoff = c.JOB_SECONDS if stage == 'final' else c.COLLECTION_CUTOFF_SECONDS
        self.deadline = min(mono+cap, self.start+cutoff, self.end)
        self.wall_deadline = min(wall+cap, self.wall_start+cutoff, self.wall_start+c.JOB_SECONDS)
        self.guard()
        c.save(c.OUT/'stages'/f'{stage}_start.json', dict(time=c.now(), monotonic=mono, unix_time=wall,
            monotonic_deadline=self.deadline, unix_deadline=self.wall_deadline, cap_seconds=cap))

    def remaining(self):
        return min(self.deadline-time.monotonic(), self.wall_deadline-time.time(),
                   self.end-time.monotonic(), self.wall_start+c.JOB_SECONDS-time.time())

    def effective_deadline(self):
        return time.monotonic()+self.remaining()

    def guard(self):
        c.require(self.remaining() > 0, 'Stage/global real-clock deadline')
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        cap = (512 if self.stage == 'final' else 448)*c.MiB
        c.require(self.accounted_bytes() < cap, 'Study evidence ceiling')
        self.aggregate_worker_rss()

    def timeout(self, maximum):
        self.guard()
        return min(maximum, self.remaining())

    def admit(self):
        self.guard()
        begun = time.monotonic()
        available = psutil.virtual_memory().available
        stable = begun if available >= 2*c.GiB else None
        minimum = available
        fast = self.admitted and stable is not None
        while not fast:
            self.guard()
            current, available = time.monotonic(), psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 2*c.GiB:
                stable = None
            elif stable is None:
                stable = current
            c.require(self.waited+current-begun < 600, 'Cumulative RAM wait exceeded ten minutes')
            if stable is not None and (self.admitted or current-stable >= 60):
                break
            time.sleep(.5)
        elapsed = time.monotonic()-begun
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.require(self.waited < 600, 'Cumulative RAM admission bound')
        if not fast:
            c.save(c.OUT/'memory_admissions'/f'{self.admissions:06d}.json', dict(time=c.now(), wait_seconds=elapsed,
                cumulative_wait_seconds=self.waited, minimum_available_bytes=minimum,
                stable_seconds=60 if self.admissions == 1 else 0))

    def finish(self):
        self.guard()
        if self.evidence is not None:
            c.require(bytes_used() <= self.accounted_bytes(), 'Unaccounted study evidence')
        c.save(c.OUT/'stages'/f'{self.stage}_end.json', dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time()))


def verify_runtime(freeze, envelope=None):
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python runtime changed')
    c.require(processor_topology() == freeze['topology'], 'Processor topology changed')
    for item in freeze['torch_runtime']:
        path = Path(item['absolute_path'])
        c.require(path.stat().st_size == item['bytes'] and c.file_hash(path) == item['sha256'], 'Pinned Torch runtime changed')
    from tools.splendorust_search_scale_20261024.compression import runtime_contract
    c.require(runtime_contract() == freeze['compression_runtime'], 'Compression runtime changed')
    for item in freeze['external_runtime']:
        if envelope:
            envelope.guard()
        path = Path(item['absolute_path'])
        c.require(path.stat().st_size == item['bytes'] and c.file_hash(path) == item['sha256'],
                  'Pinned sampler runtime changed')
    c.require(snapshot_api_modules() == freeze['sampler_api_modules'],
              'Loaded snapshot API modules or actual function owners changed')


def operational_schedule():
    result = []
    for stage in ('ops32', 'ops64'):
        for pair in range(3):
            seed = int.from_bytes(hashlib.sha256(c.canonical([DOMAIN, stage, pair]).encode()).digest(), 'big')
            for game in range(2):
                result.append(dict(stage=stage, pair_index=pair, game_index=2*pair+game,
                    challenger_seat=(pair+game) % 2, seed_decimal=str(seed),
                    challenger_quota=32 if stage == 'ops32' else 64, opponent_quota=32,
                    challenger_mode='candidate', opponent_mode='baseline' if stage == 'ops32' else 'candidate'))
    return result


def seed_inventory(schedule):
    targets = {int(row['seed_decimal']) for row in schedule}
    listed = subprocess.run(['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json',
        '-g', '*schedule*.json', '-g', '*seed_registry*.json'], cwd=c.ROOT,
        capture_output=True, text=True, check=True)
    checked, skipped, collisions = [], [], []
    def visit(value, path):
        if type(value) is int or isinstance(value, str) and value.isdecimal():
            if int(value) in targets:
                collisions.append(path)
        elif isinstance(value, list):
            for item in value: visit(item, path)
        elif isinstance(value, dict):
            for item in value.values(): visit(item, path)
    for name in sorted(listed.stdout.splitlines()):
        path = (c.ROOT/name).resolve()
        if path.is_relative_to(c.OUT.resolve()):
            continue
        if path.stat().st_size > 8*c.MiB:
            skipped.append(c.artifact(path))
            continue
        visit(c.read(path), path.relative_to(c.ROOT).as_posix())
        checked.append(c.artifact(path))
    c.require(not collisions, 'Operational seeds overlap earlier registered evidence')
    return dict(status='passed', domain=DOMAIN, checked=checked, skipped_oversized=skipped,
                collisions=collisions, scope='registered JSON names up to8MiB; not exhaustive logs')


def loaded_windows_runtime(api_identity):
    """Bind actual loaded Windows API modules used by the snapshot/backend."""
    import ctypes
    c.require(os.name == 'nt', 'This sampler qualification requires Windows')
    import _ctypes
    paths = {Path(_ctypes.__file__).resolve()}
    paths.update(path.resolve() for path in Path(ctypes.__file__).parent.rglob('*.py') if path.is_file())
    paths.update(Path(value).resolve() for value in api_identity['modules'].values())
    return [dict(absolute_path=str(path), bytes=path.stat().st_size, sha256=c.file_hash(path))
            for path in sorted(paths)]


def prepare():
    from tools.splendorust_search_scale_20261024.compression import runtime_contract
    c.require(not (c.OUT/'freeze.json').exists(), 'Already frozen; no replacement')
    c.OUT.mkdir(parents=True, exist_ok=True)
    sources = verify_runtime_sources()
    for previous in (DIAGNOSTIC, PROBE):
        c.require(c.read(previous/'completion.json')['status'] == 'completed_verified' and
                  c.read(previous/'execution.json')['status'] == 'completed',
                  'Verified diagnosis/probe and cleanup required')
    for name in ('prelaunch_tests.json', 'prelaunch_review.json'):
        receipt = c.read(c.OUT/name)
        c.require(receipt['status'] == 'passed', 'Prelaunch tests/review missing')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free >= 7*c.GiB, 'Launch requires7GiB free disk')
    diagnostic, probe = c.read(DIAGNOSTIC/'freeze.json'), c.read(PROBE/'freeze.json')
    plan = plan_rules.build_plan()
    c.save(c.OUT/'qualification_plan.json', plan)
    schedule = operational_schedule()
    inventory = seed_inventory(schedule)
    c.save(c.OUT/'seed_inventory.json', inventory)
    c.save(c.OUT/'operational_schedule.json', schedule)
    c.save(c.OUT/'model_free_qualification.json', frozen_q.model_free_checks(32))
    items = {item['path']: item for item in diagnostic['inputs']}
    for item in probe['inputs']:
        if item['path'] in items:
            c.require(items[item['path']] == item, 'Prior dependency inventories disagree')
        items[item['path']] = item
    paths = set()
    for directory in (DIAGNOSTIC, PROBE, c.OUT):
        paths.update(path for path in directory.rglob('*') if path.is_file())
    paths.update((c.ROOT/'tools/splendorust_timing_opt_20261024').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_timing_opt*.py'))
    paths.add(c.PROTOCOL)
    for path in paths:
        if '__pycache__' in path.parts:
            continue
        item = c.artifact(path)
        if item['path'] in items:
            c.require(items[item['path']] == item, 'Preserved source/evidence changed')
        items[item['path']] = item
    api_identity = snapshot_api_modules()
    runtime = {item['absolute_path']: item for item in probe['runtime']}
    for item in loaded_windows_runtime(api_identity):
        runtime[item['absolute_path']] = item
    freeze = dict(schema='splendorust_timing_optimization_qualification_v1', time=c.now(),
        inputs=sorted(items.values(), key=lambda item: item['path']),
        python=sys.version, python_executable_sha256=c.file_hash(sys.executable),
        torch_runtime=diagnostic['torch_runtime'], external_runtime=list(runtime.values()),
        sampler_api_modules=api_identity,
        compression_runtime=runtime_contract(), topology=processor_topology(),
        native_artifacts=diagnostic['native_artifacts'], source_guard=sources,
        job_seconds=c.JOB_SECONDS, stage_seconds=c.STAGES,
        collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS, automatic_retries=0,
        seconds_per_move=1., internal_seconds=.9, qualification_margin_seconds=.7,
        minimum_mean_improvement=.1, p95_may_regress=False,
        baseline='frozen_StudyWorker', candidate='BatchedStudyWorker', unchanged_child_module='tools.splendorust_search_scale_20261024.worker',
        max_rss_bytes=2*c.GiB, max_evidence_bytes=512*c.MiB, final_reserve_bytes=64*c.MiB,
        operational_games_maximum=12, operational_reproductions_maximum=24,
        strength_games=0, larger_data_deferred=True,
        authorization='One measured process-sampling optimization, qualification at32 then conditional64; future strength games require separate freeze')
    c.require(freeze['compression_runtime'] == diagnostic['compression_runtime'], 'Codec runtime changed')
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical(dict(prepared=True, inputs=len(items), freeze_sha256=c.file_hash(c.OUT/'freeze.json'), model_calls=0)), flush=True)


def make_worker(name, envelope, freeze, *, mode, quota, idle=False):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    c.require(mode in MODES and quota in (32, 64) and 'torch' not in sys.modules,
              'Valid mode/quota and model-free supervisor required')
    envelope.admit()
    folder = c.OUT/'workers'/name
    envelope.register_worker(folder)
    cls = c.BaselineStudyWorker if mode == 'baseline' else c.CandidateStudyWorker
    environment = {'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'STUDY_WORKER_DIR': str(folder),
        'STUDY_SEARCH_SIMULATIONS': str(quota), 'CUDA_VISIBLE_DEVICES': '',
        'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'}
    if idle:
        environment.update(PROBE_SELECTED_CPU=str(freeze['topology']['selected_logical_processor']),
            PROBE_NATIVE_ARTIFACTS=c.canonical(freeze['native_artifacts']),
            PROBE_MONOTONIC_DEADLINE=str(envelope.effective_deadline()))
    child = cls('tools.splendorust_sampler_probe_20261024.idle_worker' if idle else
        'tools.splendorust_search_scale_20261024.worker', folder,
        native_artifacts=freeze['native_artifacts'], aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']], initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(), max_message_bytes=4*c.MiB, max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB, max_rss_bytes=2*c.GiB, resource_guard=envelope.guard, env=environment)
    try:
        envelope.own_worker(child)
        child.study_after_close = lambda: envelope.close_worker(folder, child)
        telemetry = child._sample()
        metadata = child.initialization['result']
        if idle:
            live = {row['role']: row['pid'] for row in telemetry['process_tree']}
            c.require(metadata == dict(probe_only=True, model_calls=0, native_requests=0,
                sources=SOURCE_HASHES, pid=child.process.pid,
                native_pids=metadata.get('native_pids'), native_artifacts=freeze['native_artifacts'],
                affinity=child.affinity, model_loaded=False) and
                set(metadata['native_pids']) == {live['search'], live['reference']}, 'Idle worker identity differs')
        else:
            validate_ready_metadata(metadata, expected_sources=SOURCE_HASHES,
                worker_pid=child.process.pid, process_tree=telemetry['process_tree'], expected_quota=quota)
        child.runtime_mode, child.challenger_quota = mode, quota
        c.require(not child._failure, 'Worker process supervision failed at readiness')
        envelope.guard()
        return child
    except BaseException:
        child.close()
        raise


def qualify_sampler(envelope, freeze):
    records = []
    with make_worker('sampler_idle', envelope, freeze, mode='candidate', quota=32, idle=True) as child:
        for index in range(20):
            envelope.guard()
            telemetry, comparisons, error = None, [], None
            try:
                telemetry = child._sample()
                c.require(not child._failure, 'Batched sampler rejected stable owned tree')
                for row in telemetry['process_tree']:
                    process = psutil.Process(row['pid'])
                    created_before = process.create_time()
                    public = dict(pid=process.pid, created=created_before, parent_pid=process.ppid(),
                                  observed_threads=process.num_threads())
                    created_after = psutil.Process(row['pid']).create_time()
                    comparisons.append(dict(sampled=row, public=public, fresh_created_after=created_after))
                    c.require(created_after == created_before and all(public[key] == row[key] for key in public),
                              'Batched/public process identity or thread-count disagreement')
            except BaseException:
                error = traceback.format_exc()
            record = dict(index=index, status='passed' if error is None else 'failed', telemetry=telemetry,
                          independent_public_psutil=comparisons, error=error)
            c.save(c.OUT/'sampler_checks'/f'{index:02d}.json', record)
            records.append(record)
            c.require(error is None, 'Stable sampler qualification failed; exact poll evidence preserved: '+str(error))
    c.save(c.OUT/'sampler_qualification.json', dict(status='passed', polls=len(records), model_calls=0,
        backend='fresh Windows process snapshot, unchanged remaining authentication', evidence=[c.artifact(c.OUT/'sampler_checks'/f'{i:02d}.json') for i in range(20)]))


def actual_entrypoint(child, envelope, label):
    receipt = child.request({'operation': 'entrypoint_checks'}, envelope.timeout(300))
    c.save(c.OUT/'entrypoint_checks'/f'{label}.json', receipt)
    envelope.guard()
    c.require(receipt['status'] == 'ok' and receipt['result']['status'] == 'entrypoint_counterexamples_passed',
              'Actual corrected entrypoint failed without retry: '+str(receipt.get('detail')))


def decision(child, state, envelope, *, arm='search'):
    envelope.admit()
    observer, before = rules.observation_payload(state), rules.serialize_snapshot(state)
    ledger.check_observer(before, observer)
    legal = rules.legal_actions(state)
    serialized = [serialize_action(action) for action in legal]
    request_id = f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json', dict(player=arm, observer=observer, before=before))
    quota = child.challenger_quota if arm == 'search' else 16
    def validate(result):
        c.require(isinstance(result, dict) and result.get('action') in serialized, 'Illegal/malformed complete response')
        return audit.validate_result(result, observer, arm, expected_quota=quota)
    c.require(envelope.remaining() >= 1., 'Insufficient stage time for a complete one-second move')
    receipt = child.request({'player': arm, 'observer': observer}, envelope.timeout(1.), validate=validate)
    row = dict(mode=child.runtime_mode, quota=quota, arm=arm, seat=state.current_player,
        observer=observer, before=before, after=None, action=None, trace=None,
        completed=receipt['status'] == 'ok', clock=receipt,
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix())
    if row['completed']:
        result = receipt['result']
        c.require(result['observer_sha256'] == c.digest(observer) and result['legal_actions_sha256'] == legal_digest(legal)
                  and result['legal_count'] == len(legal), 'Worker observer/legal binding changed')
        row.update(action=result['action'], trace=result['diagnostics'])
        row['after'] = rules.serialize_snapshot(rules.CorrectedEnv().step(state, deserialize_action(row['action'])))
    return row


def save_and_check(row, path, envelope):
    c.save(path, row)  # Seal failures before any downstream assertion.
    envelope.guard()
    c.require(row['completed'], 'Move failed; retained without retries: '+str(row['clock'].get('detail')))
    audit.check_clock(row['clock'])
    audit.audit_decision(row['observer'], row['action'], row['trace'], row['arm'],
        before=row['before'], after=row['after'], expected_quota=row['quota'])


def qualify32(envelope, freeze, plan):
    rows = []
    # Qualification is completed on each exact child/parent chain before paired requests.
    with ExitStack() as stack:
        workers = {mode: stack.enter_context(make_worker('qualification32_'+mode, envelope, freeze,
                   mode=mode, quota=32)) for mode in MODES}
        for mode, child in workers.items():
            actual_entrypoint(child, envelope, '32_'+mode)
        for spec in plan['stage32_schedule']:
            state = decode_training_snapshot(plan_rules.snapshot_for_request(plan, spec))
            row = {**decision(workers[spec['mode']], state, envelope), **spec}
            save_and_check(row, c.OUT/'qualification32'/f"{spec['index']:04d}.json", envelope)
            rows.append(row)
            if len(rows) % 32 == 0:
                print(c.canonical(dict(stage='qual32', requests=len(rows), total=len(plan['stage32_schedule']))), flush=True)
    result = plan_rules.validate_32(rows, plan)
    c.save(c.OUT/'qualification32_summary.json', result)
    return result


def qualify64(envelope, freeze, plan):
    panel_rows, holdout_rows = [], []
    with make_worker('qualification64_candidate', envelope, freeze, mode='candidate', quota=64) as child:
        actual_entrypoint(child, envelope, '64_candidate')
        for spec in plan['stage64_panel']['timed_requests']:
            state = frozen_q.state_for_request(plan['stage64_panel'], spec)
            row = {**decision(child, state, envelope, arm=spec['arm']), **spec, 'challenger_simulations': 64}
            row['elapsed_seconds'] = row['clock']['receipt_elapsed_seconds']
            save_and_check(row, c.OUT/'qualification64_panel'/f"{spec['index']:03d}.json", envelope)
            panel_rows.append(row)
        for spec in plan['stage64_holdout_schedule']:
            state = decode_training_snapshot(plan_rules.snapshot_for_request(plan, spec))
            row = {**decision(child, state, envelope), **spec}
            save_and_check(row, c.OUT/'qualification64_holdout'/f"{spec['index']:03d}.json", envelope)
            holdout_rows.append(row)
    result = dict(panel=plan_rules.validate_64_panel(panel_rows, plan),
                  holdout=plan_rules.validate_64_holdout(holdout_rows, plan))
    result['qualified'] = gate_passed(result['panel']) and gate_passed(result['holdout'])
    c.save(c.OUT/'qualification64_summary.json', result)
    return result


def gate_passed(result):
    """Only explicit booleans from a complete checked stage admit subsequent work."""
    for key in ('qualified', 'timing_qualified', 'passed'):
        if key in result:
            c.require(type(result[key]) is bool, 'Qualification gate must be Boolean')
            return result[key]
    raise ValueError('Missing explicit qualification gate')


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def replay_game(initial, rows, ending, spec, *, boundary=lambda: None):
    """Independent dictionary-ledger replay, with both players using search profiles."""
    decode_training_snapshot(initial)
    current = deepcopy(initial)
    c.require(0 < len(rows) <= 2000 and ending['plies'] == len(rows), 'Bounded game move inventory')
    for ply, row in enumerate(rows):
        boundary()
        seat = current['state']['current_player']
        role = 'challenger' if seat == spec['challenger_seat'] else 'opponent'
        quota, mode = spec[role+'_quota'], spec[role+'_mode']
        c.require(not current['state']['terminal'] and row['ply'] == ply and row['before'] == current,
                  'Noncontiguous game or move after terminal')
        c.require(row['seat'] == seat and row['game_role'] == role and row['quota'] == quota and
                  row['mode'] == mode and row['arm'] == 'search', 'Operational player identity changed')
        c.require(all(row[key] == spec[key] for key in ('stage', 'pair_index', 'game_index')), 'Operational schedule changed')
        ledger.check_observer(current, row['observer'])
        audit.check_clock(row['clock'])
        c.require(row['completed'] and row['clock']['status'] == 'ok', 'Operational game includes failed move')
        audit.validate_result(row['clock']['result'], row['observer'], 'search', expected_quota=quota)
        c.require(row['action'] == row['clock']['result']['action'] and
                  row['trace'] == row['clock']['result']['diagnostics'], 'Operational result coupling changed')
        after = ledger.transition(current, row['action'])
        c.require(after == row['after'], 'Independent operational transition differs')
        current = after
    c.require(current['state']['terminal'] and ending['reason'] == 'natural' and
              ending['winner'] == current['state']['winner'] and ending['final'] == current,
              'Operational ending is not official natural termination')
    c.require(all(ending[key] == value for key, value in spec.items()), 'Operational ending schedule changed')
    return dict(status='passed', transitions=len(rows), final_sha256=c.digest(current),
                oracle='frozen independent dictionary ledger; no environment step')


def play_game(spec, envelope, freeze):
    env = rules.CorrectedEnv(seed=int(spec['seed_decimal']))
    state = env.initial_state()
    initial = rules.serialize_snapshot(state)
    folder = game_folder(spec)
    c.save(folder/'initial.json', dict(**spec, initial=initial))
    rows = []
    with ExitStack() as stack:
        workers = {role: stack.enter_context(make_worker(
            f"{spec['stage']}_g{spec['game_index']}_{role}", envelope, freeze,
            mode=spec[role+'_mode'], quota=spec[role+'_quota'])) for role in ('challenger', 'opponent')}
        for ply in range(2000):
            if state.terminal:
                break
            role = 'challenger' if state.current_player == spec['challenger_seat'] else 'opponent'
            row = decision(workers[role], state, envelope)
            row.update(ply=ply, game_role=role, **{key: spec[key] for key in ('stage', 'pair_index', 'game_index')})
            save_and_check(row, folder/'moves'/f'{ply:04d}.json', envelope)
            rows.append(row)
            state = decode_training_snapshot(row['after'])
            if not state.terminal and (row['after'] == row['before'] or row['action']['action_type'] == 'PASS' and
                len(rows) > 1 and rows[-2]['action']['action_type'] == 'PASS'):
                raise ValueError('Unresolved actual-game deadlock')
        c.require(state.terminal, 'Operational2,000-ply safety cap reached')
    ending = dict(**spec, reason='natural', winner=state.winner, final=rules.serialize_snapshot(state),
                  complete=True, plies=len(rows))
    replay = replay_game(initial, rows, ending, spec, boundary=envelope.guard)
    c.save(folder/'result.json', ending)
    c.save(folder/'online_replay.json', replay)
    return ending, rows


def operational(stage, envelope, freeze, schedule):
    specs = [spec for spec in schedule if spec['stage'] == stage]
    c.require(len(specs) == 6, 'Exactly six operational games required')
    rows, games = [], []
    for spec in specs:
        ending, moves = play_game(spec, envelope, freeze)
        games.append(ending)
        rows.extend(moves)
        print(c.canonical(dict(stage=stage, games=len(games), total=6)), flush=True)
    # Every candidate move counts. In ops64 both players use the candidate sampler.
    selected = [row for row in rows if row['mode'] == 'candidate']
    failures = [{key: row[key] for key in ('stage', 'pair_index', 'game_index', 'ply', 'quota')}
                | {'arrival_seconds': row['clock']['receipt_elapsed_seconds']}
                for row in selected if row['clock']['receipt_elapsed_seconds'] > .7]
    result = dict(qualified=not failures, games=6, complete_decisions=len(rows), candidate_decisions=len(selected),
        candidate_margin_failures=failures, margin_seconds=.7,
        maximum_candidate_arrival_seconds=max(row['clock']['receipt_elapsed_seconds'] for row in selected),
        outcomes_used_for_admission=False, interpretation='Operational correctness/timing only, not playing-strength evidence')
    c.save(c.OUT/f'{stage}_summary.json', result)
    return result


def verify_saved_decision(row):
    path = c.ROOT/row['worker_clock_file']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()), 'Clock path outside owned evidence')
    c.require(c.read(path) == row['clock'] and path.name == row['clock']['request_id']+'.json', 'Saved trusted receipt changed')
    c.require(c.read(path.parent/'inputs'/path.name) == dict(player=row['arm'], observer=row['observer'], before=row['before']),
              'Saved player input changed')
    audit.audit_decision(row['observer'], row['action'], row['trace'], row['arm'],
        before=row['before'], after=row['after'], expected_quota=row['quota'])


def final_checks(envelope, freeze, plan, schedule, summaries):
    decisions = 0
    for name in ('qualification32', 'qualification64_panel', 'qualification64_holdout'):
        for path in sorted((c.OUT/name).glob('*.json')):
            envelope.guard()
            verify_saved_decision(c.read(path))
            decisions += 1
    rows32 = [c.read(path) for path in sorted((c.OUT/'qualification32').glob('*.json'))]
    c.require(plan_rules.validate_32(rows32, plan) == summaries['qual32'], 'Repeated32 gate analysis differs')
    if 'qual64' in summaries:
        panel = [c.read(path) for path in sorted((c.OUT/'qualification64_panel').glob('*.json'))]
        holdout = [c.read(path) for path in sorted((c.OUT/'qualification64_holdout').glob('*.json'))]
        c.require(plan_rules.validate_64_panel(panel, plan) == summaries['qual64']['panel'] and
                  plan_rules.validate_64_holdout(holdout, plan) == summaries['qual64']['holdout'], 'Repeated64 gate analysis differs')
    games, samples = [], []
    for spec in schedule:
        if spec['stage'] not in summaries:
            continue
        folder = game_folder(spec)
        initial, ending = c.read(folder/'initial.json'), c.read(folder/'result.json')
        expected = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(initial == dict(**spec, initial=expected), 'Operational initial world changed')
        paths = sorted((folder/'moves').glob('*.json'))
        c.require([path.name for path in paths] == [f'{i:04d}.json' for i in range(ending['plies'])], 'Operational move-file inventory changed')
        rows = [c.read(path) for path in paths]
        for row in rows:
            envelope.guard()
            verify_saved_decision(row)
            decisions += 1
        games.append(replay_game(initial['initial'], rows, ending, spec, boundary=envelope.guard))
        # Outcome-independent first complete decision for each exact player.
        for role in ('challenger', 'opponent'):
            samples.append(next(row for row in rows if row['game_role'] == role))
    c.save(c.OUT/'all_game_replay.json', dict(status='passed', games=len(games), receipts=games))
    c.require(len(samples) <= 24, 'Operational reproduction bound')
    repetitions = []
    for index, original in enumerate(samples):
        with make_worker(f'reproduction_{index:02d}', envelope, freeze,
                mode=original['mode'], quota=original['quota']) as child:
            repeated = decision(child, decode_training_snapshot(original['before']), envelope)
            save_and_check(repeated, c.OUT/'repeated_decisions'/f'{index:02d}.json', envelope)
            c.require(repeated['action'] == original['action'] and repeated['trace'] == original['trace'],
                      'Operational complete decision reproduction differs')
        repetitions.append(dict(index=index, stage=original['stage'], game_index=original['game_index'],
                                ply=original['ply'], mode=original['mode'], quota=original['quota']))
    c.save(c.OUT/'repeated_decision_summary.json', dict(status='passed', requests=len(samples), receipts=repetitions,
        selection='first move per player per operational game; no outcome/timing selection', deadline_seconds=1., margin_gate=False))
    log_checks = []
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard()
        log_checks.append(verify_worker_logs(folder, boundary=envelope.guard, deadline=envelope.effective_deadline()))
    c.save(c.OUT/'all_worker_log_integrity.json', dict(status='passed', workers=len(log_checks), receipts=log_checks))
    verify_runtime(freeze, envelope)
    c.save(c.OUT/'verification_summary.json', dict(status='passed', decisions=decisions,
        operational_games=len(games), operational_reproductions=len(samples),
        maximum_sampled_combined_worker_rss_bytes=envelope.peak_combined_rss))


def run_stages(envelope, freeze, plan, schedule, *, sampler_started=False):
    summaries = {}
    if sampler_started:
        c.require(envelope.stage == 'sampler', 'Sampler startup stage missing')
    else:
        envelope.begin('sampler')
    qualify_sampler(envelope, freeze)
    envelope.finish()
    envelope.begin('qualification32')
    summaries['qual32'] = qualify32(envelope, freeze, plan)
    envelope.finish()
    if gate_passed(summaries['qual32']):
        envelope.begin('operational32')
        summaries['ops32'] = operational('ops32', envelope, freeze, schedule)
        envelope.finish()
        if gate_passed(summaries['ops32']):
            envelope.begin('qualification64')
            summaries['qual64'] = qualify64(envelope, freeze, plan)
            envelope.finish()
            if gate_passed(summaries['qual64']):
                envelope.begin('operational64')
                summaries['ops64'] = operational('ops64', envelope, freeze, schedule)
                envelope.finish()
    envelope.begin('final')
    final_checks(envelope, freeze, plan, schedule, summaries)
    envelope.finish()
    admitted64 = 'ops64' in summaries and gate_passed(summaries['ops64'])
    c.save(c.OUT/'qualification_outcome.json', dict(status='completed_verified',
        optimized32_qualified='ops32' in summaries and gate_passed(summaries['ops32']),
        optimized64_qualified=admitted64, next_strength_comparison='64versus32' if admitted64 else None,
        strength_games_launched=0, summaries=summaries, larger_data_deferred=True))
    return summaries


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Freeze identity changed')
    claim = c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256'] == expected_sha, 'Watchdog identity changed')
    freeze, envelope = c.read(c.OUT/'freeze.json'), Envelope(claim)
    envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json', dict(time=c.now(), pid=os.getpid(), created=psutil.Process().create_time(), freeze_sha256=expected_sha))
    status, failure = 'incomplete', None
    try:
        # Start the sampler stage before startup hashes/admission; no free work clock.
        envelope.begin('sampler')
        c.save(c.OUT/'supervisor_runtime_sources.json', dict(sources=verify_runtime_sources()))
        verify_runtime(freeze, envelope)
        envelope.admit()
        plan, schedule = c.read(c.OUT/'qualification_plan.json'), c.read(c.OUT/'operational_schedule.json')
        run_stages(envelope, freeze, plan, schedule, sampler_started=True)
        status = 'completed_verified'
    except BaseException:
        failure = traceback.format_exc()
        c.save(c.OUT/'failure.json', dict(time=c.now(), stage=envelope.stage, error=failure))
    c.save(c.OUT/'completion.json', dict(status=status, time=c.now(), failure=failure, stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start, wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha, automatic_retries=0, larger_data_deferred=True, strength_games=0,
        note='Acceptance also requires successful owning watchdog cleanup. No strength games run in this qualification.'))
    print(c.canonical(dict(status=status, stage=envelope.stage, failure=failure)), flush=True)
    return 0 if status == 'completed_verified' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'run'])
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    return prepare() if args.operation == 'prepare' else run(args.freeze_sha256)


if __name__ == '__main__':
    raise SystemExit(main())

