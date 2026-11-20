"""One isolated exact-inference-graph qualification; no strength benchmark branch."""
import argparse
import ast
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import time
import traceback

import psutil
from . import common as c, qualification as plan_rules
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit, qualification as frozen_q
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules

DOMAIN = 'splendorust_inference_graph_ops_20261025_v1'
MODES = ('baseline', 'candidate')
BACKEND = 'torchscript_frozen_exact_forward_v1'

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
    from importlib.metadata import version
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
    c.require(version('numpy')==freeze['numpy_version'] and
        version('torch')==freeze['torch_package_version'], 'Pinned numerical package version changed')


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
    c.require(ending.get('complete') is True and current['state']['terminal'] and ending['reason'] == 'natural' and
              ending['winner'] == current['state']['winner'] and ending['final'] == current,
              'Operational ending is not official natural termination')
    c.require(all(ending[key] == value for key, value in spec.items()), 'Operational ending schedule changed')
    return dict(status='passed', transitions=len(rows), final_sha256=c.digest(current),
                oracle='frozen independent dictionary ledger; no environment step')


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
    allowance = envelope.timeout(1.)
    c.require(allowance == 1., 'Stage guard shortened move allowance; no partial dispatch')
    receipt = child.request({'player': arm, 'observer': observer}, allowance, validate=validate)
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


def play_game(spec, envelope, freeze):
    c.require(spec in operational_schedule(), 'Unfrozen operational schedule item')
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


def verify_saved_decision(row):
    path = c.ROOT/row['worker_clock_file']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()), 'Clock path outside owned evidence')
    c.require(c.read(path) == row['clock'] and path.name == row['clock']['request_id']+'.json', 'Saved trusted receipt changed')
    c.require(c.read(path.parent/'inputs'/path.name) == dict(player=row['arm'], observer=row['observer'], before=row['before']),
              'Saved player input changed')
    audit.audit_decision(row['observer'], row['action'], row['trace'], row['arm'],
        before=row['before'], after=row['after'], expected_quota=row['quota'])


def operational_schedule():
    return plan_rules.operational_schedule()


def predecessor_acceptance():
    """Authenticate successful baseline qualification and the preserved failure."""
    names=('freeze.json','completion.json','execution.json','qualification_outcome.json',
           'qualification32_summary.json','qualification64_summary.json','ops64_summary.json',
           'verification_summary.json','all_game_replay.json','repeated_decision_summary.json',
           'all_worker_log_integrity.json')
    records={name:c.read(c.PRIOR/name) for name in names}
    complete,execution,outcome=(records[name] for name in ('completion.json','execution.json','qualification_outcome.json'))
    sha=c.file_hash(c.PRIOR/'freeze.json')
    c.require(complete['status']=='completed_verified' and complete['failure'] is None and
        execution['status']=='completed' and execution['exit_code']==0 and not execution['cleanup_errors'],
        'Previous optimized sampler qualification/cleanup must remain verified')
    c.require(complete['freeze_sha256']==execution['freeze_sha256']==sha and
        execution['charged_job_seconds']<records['freeze.json']['job_seconds'],
        'Previous qualification freeze identity or clock changed')
    c.require(outcome['optimized32_qualified'] is True and outcome['optimized64_qualified'] is True and
        outcome['strength_games_launched']==0,'Both unchanged baseline quotas must be admitted')
    c.require(all(records[name]['qualified'] is True for name in
        ('qualification32_summary.json','qualification64_summary.json','ops64_summary.json')),
        'Previous qualification gate failed')
    c.require(all(records[name]['status']=='passed' for name in
        ('verification_summary.json','all_game_replay.json','repeated_decision_summary.json','all_worker_log_integrity.json')),
        'Previous verification is incomplete')
    failed_complete,failed_execution=(c.read(c.FAILED/name) for name in ('completion.json','execution.json'))
    failed_sha=c.file_hash(c.FAILED/'freeze.json')
    c.require(failed_complete['status']=='incomplete' and failed_execution['status']=='runner_failed' and
        not failed_execution['cleanup_errors'] and
        failed_complete['freeze_sha256']==failed_execution['freeze_sha256']==failed_sha,
        'The failed operational attempt must remain preserved with confirmed cleanup')
    evidence=[c.artifact(c.PRIOR/name) for name in names]
    evidence.extend(c.artifact(c.FAILED/name) for name in ('freeze.json','completion.json','execution.json','failure.json'))
    evidence.append(c.artifact(c.POSTMORTEM/'analysis.json'))
    for folder, extra in ((c.BUFFERED, ('qualification_outcome.json','qualification_summary.json')),
                          (c.DIAGNOSTIC, ('overhead_summary.json','diagnostic_summary.json','component_analysis.json'))):
        names=('freeze.json','completion.json','execution.json','verification_summary.json',
               'all_worker_log_integrity.json')+extra
        saved={name:c.read(folder/name) for name in names}
        finish,outer=saved['completion.json'],saved['execution.json']
        previous_sha=c.file_hash(folder/'freeze.json')
        c.require(finish['status']=='completed_verified' and finish['failure'] is None and
            outer['status']=='completed' and outer['exit_code']==0 and not outer['cleanup_errors'],
            'Previous diagnostic/negative attempt must retain verified cleanup')
        c.require(finish['freeze_sha256']==outer['freeze_sha256']==previous_sha and
            outer['charged_job_seconds']<saved['freeze.json']['job_seconds'],
            'Previous diagnostic/negative freeze or clock changed')
        c.require(all(saved[name]['status']=='passed' for name in
            ('verification_summary.json','all_worker_log_integrity.json')), 'Previous verification missing')
        if folder==c.BUFFERED:
            outcome=saved['qualification_outcome.json']
            c.require(outcome['buffered_reader_qualified'] is False and outcome['saved_panel_qualified'] is False,
                      'Unsuccessful buffered reader must remain unpromoted')
        else:
            c.require(saved['verification_summary.json']['requests']==240 and
                saved['verification_summary.json']['science_and_transitions_verified'] is True,
                'Completed component evidence inventory changed')
        evidence.extend(c.artifact(folder/name) for name in names)
    return dict(status='passed',baseline_freeze_sha256=sha,preserved_failure_freeze_sha256=failed_sha,evidence=evidence)


def nested_artifacts(value):
    """Only explicit artifact triples, never an unbounded output-tree expansion."""
    if isinstance(value,dict):
        if {'path','sha256','bytes'}<=set(value):
            yield {key:value[key] for key in ('path','sha256','bytes')}
        for item in value.values():
            yield from nested_artifacts(item)
    elif isinstance(value,list):
        for item in value:
            yield from nested_artifacts(item)


def graph_runtime_inventory(previous):
    """Bind installed graph compiler sources without importing the model runtime."""
    c.require('torch' not in sys.modules, 'Model-free supervisor cannot import Torch')
    inherited={item['absolute_path']:deepcopy(item) for item in previous['torch_runtime']}
    roots=set()
    for item in inherited.values():
        root=Path(item['absolute_path'])
        for _ in Path(item['relative_path']).parts:
            root=root.parent
        roots.add(root.resolve())
    c.require(len(roots)==1, 'Pinned Torch runtime must have one package root')
    root=roots.pop()
    paths=set((root/'jit').rglob('*.py'))
    paths.update(root/name for name in ('_jit_internal.py','__init__.py','version.py','torch_version.py'))
    c.require(paths and (root/'jit/__init__.py') in paths, 'Installed Torch JIT sources missing')
    for path in sorted(paths):
        c.require(path.is_file() and path.resolve().is_relative_to(root), 'Invalid graph runtime source')
        row=dict(absolute_path=str(path.resolve()),relative_path=path.relative_to(root).as_posix(),
                 bytes=path.stat().st_size,sha256=c.file_hash(path))
        if row['absolute_path'] in inherited:
            c.require(inherited[row['absolute_path']]==row, 'Pinned Torch graph source changed')
        inherited[row['absolute_path']]=row
    return sorted(inherited.values(),key=lambda item:item['relative_path'])


def graph_external_runtime(previous):
    from importlib.util import find_spec
    from importlib.metadata import version
    spec=find_spec('numpy')
    c.require(spec is not None and spec.origin is not None, 'NumPy identity runtime unavailable')
    root=Path(spec.origin).resolve().parent
    paths={root/'__init__.py',root/'version.py',root/'core/__init__.py',root/'core/multiarray.py'}
    binaries=list(root.rglob('*multiarray_umath*.pyd'))
    c.require(len(binaries)==1, 'One installed NumPy multiarray binary required')
    paths.update(binaries)
    paths.update(root.glob('_core/_multiarray_umath.py'))
    inherited={item['absolute_path']:deepcopy(item) for item in previous['external_runtime']}
    for path in sorted(paths):
        c.require(path.is_file() and path.resolve().is_relative_to(root), 'NumPy identity source missing')
        row=dict(absolute_path=str(path.resolve()),bytes=path.stat().st_size,sha256=c.file_hash(path))
        if row['absolute_path'] in inherited:
            c.require(all(inherited[row['absolute_path']][key]==value for key,value in row.items()),
                      'Inherited NumPy runtime changed')
        else:
            inherited[row['absolute_path']]=row
    torch_spec=find_spec('torch')
    c.require(torch_spec is not None and torch_spec.origin is not None,'Torch package location missing')
    parsed=ast.parse((Path(torch_spec.origin).resolve().parent/'version.py').read_text(encoding='utf-8'))
    versions=[ast.literal_eval(node.value) for node in parsed.body if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id=='__version__' for target in node.targets)]
    c.require(len(versions)==1 and type(versions[0]) is str,'Installed Torch runtime version is not literal')
    return dict(numpy_version=version('numpy'),torch_package_version=version('torch'),torch_runtime_version=versions[0],
                files=sorted(inherited.values(),key=lambda item:item['absolute_path']))


def prepare():
    from tools.splendorust_search_scale_20261024.compression import runtime_contract
    c.require(not (c.OUT/'freeze.json').exists(),'Already frozen; no replacement')
    acceptance=predecessor_acceptance()
    for name in ('prelaunch_tests.json','prelaunch_review.json'):
        receipt=c.read(c.OUT/name)
        c.require(receipt['status']=='passed','Prelaunch model-free checks/review required')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free>=7*c.GiB,'Launch requires7GiB free disk')
    sources=verify_runtime_sources()
    previous=c.read(c.DIAGNOSTIC/'freeze.json')
    plan=plan_rules.build_plan()
    c.save(c.OUT/'qualification_plan.json',plan)
    parity_plan=plan_rules.build_parity_plan()
    c.save(c.OUT/'graph_parity_plan.json',parity_plan)
    c.save(c.OUT/'baseline_acceptance.json',acceptance)
    schedule=operational_schedule()
    c.save(c.OUT/'seed_inventory.json',seed_inventory(schedule))
    c.save(c.OUT/'operational_schedule.json',schedule)
    c.save(c.OUT/'model_free_qualification.json',frozen_q.model_free_checks(64))
    inherited={item['path']:item for item in previous['inputs']}
    # All inherited repository source remains bound. Output evidence is selected
    # explicitly through profiles, entrypoint references and the new panel.
    items={path:item for path,item in inherited.items() if not path.startswith('outputs/')}
    def bind(item):
        path=item['path']
        if path in inherited:
            c.require(inherited[path]==item,'Preserved inherited input changed: '+path)
        if path in items:
            c.require(items[path]==item,'Dependency inventory disagreement: '+path)
        items[path]=item
    for document in (acceptance,plan,parity_plan,frozen_q.build_plan(32),frozen_q.build_plan(64),
                     c.read(frozen_q.PARITY_RECEIPT)):
        for item in nested_artifacts(document): bind(item)
    bind(c.artifact(frozen_q.PARITY_RECEIPT))
    for quota in (16,32,64):
        bind(c.artifact(c.profile_path(quota)))
        for item in c.profile_for(quota)['artifacts']: bind(item)
    for item in previous['native_artifacts']: bind(item)
    paths=set((c.ROOT/'tools/splendorust_inference_opt_20261025').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_inference_opt*.py'))
    paths.update(path for path in c.OUT.rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    paths.add(c.PROTOCOL)
    for path in paths: bind(c.artifact(path))
    c.require(runtime_contract()==previous['compression_runtime'] and
        snapshot_api_modules()==previous['sampler_api_modules'],'Unchanged sampler/codec runtime required')
    external=graph_external_runtime(previous)
    freeze=dict(schema='splendorust_inference_graph_qualification_v1',time=c.now(),
        inputs=sorted(items.values(),key=lambda item:item['path']),baseline_acceptance=acceptance,
        python=sys.version,python_executable_sha256=c.file_hash(sys.executable),
        torch_runtime=graph_runtime_inventory(previous),external_runtime=external['files'],
        numpy_version=external['numpy_version'],
        torch_package_version=external['torch_package_version'],
        torch_runtime_version=external['torch_runtime_version'],
        sampler_api_modules=previous['sampler_api_modules'],compression_runtime=runtime_contract(),
        topology=processor_topology(),native_artifacts=previous['native_artifacts'],source_guard=sources,
        job_seconds=c.JOB_SECONDS,stage_seconds=c.STAGES,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        seconds_per_move=1.,internal_seconds=.9,qualification_margin_seconds=.7,minimum_mean_improvement=.05,
        p95_may_regress=False,baseline='qualified_raw_stdout_reader',candidate='same_weights_float32_one_thread_frozen_graph',
        baseline_child_module='tools.splendorust_search_scale_20261024.worker',
        candidate_child_module='tools.splendorust_inference_opt_20261025.worker',
        max_rss_bytes=2*c.GiB,max_evidence_bytes=512*c.MiB,final_reserve_bytes=64*c.MiB,
        paired_timing_requests=288,actual_entrypoint_workers=4,timed_panel_workers=4,operational_games_maximum=6,
        graph_parity_cases_per_candidate_worker=11,inference_backend=BACKEND,
        graph_parity_plan=c.artifact(c.OUT/'graph_parity_plan.json'),
        operational_reproductions_maximum=12,strength_games=0,automatic_retries=0,larger_data_deferred=True,
        input_scope='Inherited repository source/runtime plus explicit profile, entrypoint and selected historical evidence; no recursive prior-output expansion',
        authorization='One exact inference-graph change; corrected qualification and conditional six operational games only')
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json',freeze)
    print(c.canonical(dict(prepared=True,inputs=len(items),freeze_sha256=c.file_hash(c.OUT/'freeze.json'),model_calls=0)),flush=True)


def make_worker(name,envelope,freeze,*,mode,quota):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    c.require(mode in MODES and quota in (32,64) and 'torch' not in sys.modules,
              'Valid inference mode/quota and model-free supervisor required')
    envelope.admit()
    folder=c.OUT/'workers'/name
    envelope.register_worker(folder)
    cls=c.BaselineStudyWorker if mode=='baseline' else c.CandidateStudyWorker
    module='tools.splendorust_search_scale_20261024.worker' if mode=='baseline' else 'tools.splendorust_inference_opt_20261025.worker'
    child=cls(module,folder,
        native_artifacts=freeze['native_artifacts'],aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']],initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(),max_message_bytes=4*c.MiB,max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB,max_rss_bytes=2*c.GiB,resource_guard=envelope.guard,
        env={'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'STUDY_WORKER_DIR':str(folder),
             'STUDY_SEARCH_SIMULATIONS':str(quota),'CUDA_VISIBLE_DEVICES':'',
             'RAYON_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'1'})
    try:
        envelope.own_worker(child)
        child.study_after_close=lambda:envelope.close_worker(folder,child)
        telemetry=child._sample()
        metadata=deepcopy(child.initialization['result'])
        if mode=='candidate':
            c.require(metadata.pop('inference_backend')==BACKEND,'Candidate inference backend changed')
        validate_ready_metadata(metadata,expected_sources=SOURCE_HASHES,
            worker_pid=child.process.pid,process_tree=telemetry['process_tree'],expected_quota=quota)
        child.runtime_mode,child.challenger_quota=mode,quota
        c.require(not child._failure,'Worker supervision failed at readiness')
        if mode=='candidate':
            graph_parity(child,envelope,freeze)
        envelope.guard()
        return child
    except BaseException:
        child.close()
        raise


def graph_parity(child,envelope,freeze):
    """Every graph instance proves exact outputs before any scientific decision."""
    item=freeze['graph_parity_plan']
    c.verify([item])
    plan=c.read(c.ROOT/item['path'])
    fixtures=plan['fixtures']
    c.require(len(fixtures)==11, 'Fixed graph parity inventory')
    receipt=child.request(dict(operation='graph_parity',fixtures=fixtures),envelope.timeout(60))
    path=child.output_dir/'graph_parity_clock.json'
    c.save(path,receipt)
    envelope.guard()
    c.require(receipt['status']=='ok', 'Graph output parity failed before decisions; no fallback/retry')
    validate_graph_parity(receipt['result'],fixtures)
    c.require(c.read(child.output_dir/f"{receipt['request_id']}.json")==receipt,
              'Graph parity raw receipt changed')
    verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)


def validate_graph_parity(result,fixtures):
    c.require(type(result) is dict and result.get('schema')=='splendorust_inference_graph_parity_v1' and
        result.get('status')=='passed' and result.get('passed') is True and
        result.get('fixtures_count')==len(fixtures)==11 and result.get('fixture_payload_sha256')==c.digest(fixtures),
        'Exact graph parity gate failed')
    rows=result['rows']
    c.require(len(rows)==11 and len({row['fixture_id'] for row in rows})==11, 'Graph parity row inventory')
    for row,fixture in zip(rows,fixtures,strict=True):
        c.require(row['fixture_id']==fixture['fixture_id'] and row['batch']==fixture['batch'] and
            row['input_sha256']==c.digest(fixture['tokens']) and
            row['exact_float_equal'] is True and row['float32_bits_equal'] is True and
            row['eager_output_sha256']==row['candidate_output_sha256'] and
            row['eager_float32_sha256']==row['candidate_float32_sha256'],
            'Graph parity output/bit/fixture equality failed')


def exact_output_hashes(outputs,batch):
    """Independent scalar/output-byte verification; no Torch or NumPy import."""
    c.require(type(outputs) is list and len(outputs)==batch,'Graph output batch inventory')
    flat=[]
    for row in outputs:
        c.require(type(row) is list and len(row)==83 and
            all(type(value) is float and math.isfinite(value) for value in row), 'Graph output float32 inventory')
        flat.extend(row)
    bits=struct.pack('<'+'f'*len(flat),*flat)
    c.require(tuple(flat)==struct.unpack('<'+'f'*len(flat),bits),'Graph output is not exact float32')
    return c.digest(outputs),hashlib.sha256(bits).hexdigest()


def verify_graph_evidence(folder,freeze,*,boundary=lambda:None):
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    folder=Path(folder)
    boundary()
    plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
    fixtures=plan['fixtures']
    clock=c.read(folder/'graph_parity_clock.json')
    c.require(clock['status']=='ok' and clock['receipt_elapsed_seconds']<=clock['deadline_seconds']<=60. and
        c.read(folder/f"{clock['request_id']}.json")==clock,'Graph parity trusted receipt changed')
    result=clock['result']
    validate_graph_parity(result,fixtures)
    identity=c.read(folder/'graph_identity.json')
    execution=c.read(folder/'inference_execution_identity.json')
    init=c.read(folder/'initialization.json')
    c.require(identity['schema']=='splendorust_inference_graph_identity_v1' and
        identity['backend']==execution['backend']==BACKEND and
        result['graph_identity_sha256']==execution['graph_identity_sha256']==c.digest(identity) and
        execution['metadata']==init['result'],'Graph identity/readiness/parity binding changed')
    c.require(identity['checkpoint_sha256']==CHECKPOINT_SHA256 and
        identity['weights_before_sha256']==identity['weights_after_sha256'] and
        identity['parameter_count']==4780883 and identity['dtype']=='torch.float32' and identity['device']=='cpu' and
        identity['threads']==identity['interop_threads']==1 and
        identity['supported_batches']==[1,2,8] and identity['input_shape']==['batch',31,48] and
        identity['output_shape']==['batch',83] and identity['serialized_graph_saved'] is False and
        identity['warmup_is_parity_proof'] is False, 'Exact graph settings or checkpoint changed')
    expected_options=dict(trace_batch=1,trace_check_trace=False,trace_strict=True,
        freeze_optimize_numerics=False,warm_batches=[1,2,8],
        trace_and_warm_inputs='architecture_zero_float32_only',automatic_fallback=False,alternative_candidates=0)
    c.require(identity['options']==expected_options,'Graph optimization settings changed')
    c.require(identity['numpy_version']==freeze['numpy_version'] and
        identity['torch_version']==freeze['torch_runtime_version'],'Graph numerical runtime version changed')
    c.require(identity['graph_ir_artifact']=='graph_ir.txt' and
        0<identity['graph_ir_bytes']<=256*1024 and (folder/'graph_ir.txt').stat().st_size==identity['graph_ir_bytes'] and
        c.file_hash(folder/'graph_ir.txt')==identity['graph_ir_sha256'],'Saved graph source changed')
    evidence=c.read(folder/'graph_parity.json')
    c.require(evidence['schema']=='splendorust_inference_graph_parity_v1' and
        evidence['summary']==result and evidence['fixtures']==fixtures and len(evidence['evidence'])==11,
        'Full graph parity evidence changed')
    for fixture,row,observed in zip(fixtures,result['rows'],evidence['evidence'],strict=True):
        boundary()
        c.require(observed['fixture_id']==fixture['fixture_id'] and observed['batch']==fixture['batch'] and
            observed['comparison']==row and observed['eager_outputs']==observed['candidate_outputs'],
            'Full exact graph output evidence differs')
        provenance=plan['provenance_by_fixture'][fixture['fixture_id']]
        if 'historical_outputs' in provenance:
            c.require(observed['eager_outputs']==provenance['historical_outputs'] and
                row['eager_output_sha256']==provenance['historical_output_sha256'],
                'Recorded source-root output differs from unchanged eager model')
        for key in ('eager','candidate'):
            output_hash,bits_hash=exact_output_hashes(observed[key+'_outputs'],fixture['batch'])
            c.require(output_hash==row[key+'_output_sha256'] and bits_hash==row[key+'_float32_sha256'],
                      'Independent graph output/bit hash differs')
    return dict(status='passed',worker=folder.name,fixtures=11,graph_identity_sha256=c.digest(identity),
                weights_sha256=identity['weights_before_sha256'])


def actual_entrypoint(child,envelope,label):
    receipt=child.request({'operation':'entrypoint_checks'},envelope.timeout(300))
    c.save(c.OUT/'entrypoint_checks'/f'{label}.json',receipt)
    envelope.guard()
    c.require(receipt['status']=='ok' and receipt['result']['status']=='entrypoint_counterexamples_passed',
              'Actual corrected entrypoint failed without retry: '+str(receipt.get('detail')))
    result=receipt['result']
    c.require(result['challenger_quota']==child.challenger_quota and result['counts']['complete_decisions']==52,
              'Actual entrypoint quota/decision inventory differs')


def qualify_panel(envelope,freeze,plan):
    rows=[]
    schedule=plan['timed_requests']
    c.require(len(schedule)==288 and [row['index'] for row in schedule]==list(range(288)),
              'Fixed complete paired timing schedule required')
    c.require([row['quota'] for row in schedule]==[64]*144+[32]*144,'Two-resident64then32 schedule required')
    # Establish every actual corrected entrypoint before any panel inference.
    # Close each pair before opening the next so aggregate admission stays exact.
    for quota in (64,32):
        with ExitStack() as stack:
            workers={mode:stack.enter_context(make_worker(f'entrypoint_{quota}_{mode}',envelope,freeze,
                mode=mode,quota=quota)) for mode in MODES}
            for mode,child in workers.items(): actual_entrypoint(child,envelope,f'{quota}_{mode}')
    for quota in (64,32):
        with ExitStack() as stack:
            workers={mode:stack.enter_context(make_worker(f'qualification_{quota}_{mode}',envelope,freeze,
                mode=mode,quota=quota)) for mode in MODES}
            for spec in (item for item in schedule if item['quota']==quota):
                state=decode_training_snapshot(plan_rules.snapshot_for_request(plan,spec))
                row={**decision(workers[spec['mode']],state,envelope),**spec}
                save_and_check(row,c.OUT/'paired_timing'/f"{spec['index']:04d}.json",envelope)
                rows.append(row)
                if len(rows)%24==0:
                    print(c.canonical(dict(stage='qualification',requests=len(rows),total=288)),flush=True)
    result=plan_rules.validate_panel(rows,plan,boundary=envelope.guard)
    c.save(c.OUT/'qualification_summary.json',result)
    return result


def operational(envelope,freeze,schedule):
    c.require(schedule==operational_schedule(),'Operational schedule changed')
    rows=[]; games=[]
    for spec in schedule:
        ending,moves=play_game(spec,envelope,freeze)
        games.append(ending); rows.extend(moves)
        print(c.canonical(dict(stage='operational',games=len(games),total=6)),flush=True)
    result=summarize_operational(rows,len(games))
    c.save(c.OUT/'operational_summary.json',result)
    return result


def summarize_operational(rows,games):
    c.require(games==6 and rows and all(row['mode']=='candidate' and row['quota'] in (32,64) and
        row['completed'] and row['clock']['status']=='ok' and
        row['clock']['deadline_seconds']==1. and row['clock']['receipt_elapsed_seconds']<=1. for row in rows),
        'Operational summary requires six complete games and exact one-second candidate decisions')
    failures=[{key:row[key] for key in ('stage','pair_index','game_index','ply','quota')}|
        {'arrival_seconds':row['clock']['receipt_elapsed_seconds']} for row in rows
        if row['clock']['receipt_elapsed_seconds']>.7]
    return dict(qualified=not failures,games=6,complete_decisions=len(rows),margin_seconds=.7,
        margin_failures=failures,maximum_arrival_seconds=max(row['clock']['receipt_elapsed_seconds'] for row in rows),
        outcomes_used_for_admission=False,interpretation='Timing/correctness only; no strength estimate')


def final_checks(envelope,freeze,plan,schedule,summaries):
    entrypoints=[]
    for quota in (64,32):
        for mode in MODES:
            envelope.guard()
            receipt=c.read(c.OUT/'entrypoint_checks'/f'{quota}_{mode}.json')
            folder=c.OUT/'workers'/f'entrypoint_{quota}_{mode}'
            c.require(c.read(folder/f"{receipt['request_id']}.json")==receipt and receipt['status']=='ok' and
                receipt['result']['status']=='entrypoint_counterexamples_passed' and
                receipt['result']['challenger_quota']==quota and
                receipt['result']['counts']['complete_decisions']==52,'Saved actual entrypoint evidence changed')
            entrypoints.append(dict(mode=mode,quota=quota,complete_decisions=52))
    c.save(c.OUT/'all_entrypoint_checks.json',dict(status='passed',workers=4,receipts=entrypoints))
    paths=sorted((c.OUT/'paired_timing').glob('*.json'))
    c.require([path.name for path in paths]==[f'{i:04d}.json' for i in range(288)],'Paired timing inventory differs')
    rows=[c.read(path) for path in paths]
    for row in rows:
        envelope.guard(); verify_saved_decision(row)
    c.require(plan_rules.validate_panel(rows,plan,boundary=envelope.guard)==summaries['qualification'],
              'Independent repeated panel qualification differs')
    games=[]; samples=[]; game_rows=[]; decisions=len(rows)
    if 'operational' in summaries:
        c.require(schedule==operational_schedule(),'Final operational schedule differs')
        for spec in schedule:
            envelope.guard(); folder=game_folder(spec)
            initial,ending=c.read(folder/'initial.json'),c.read(folder/'result.json')
            expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
            c.require(initial==dict(**spec,initial=expected),'Operational initial world/identity changed')
            paths=sorted((folder/'moves').glob('*.json'))
            c.require([path.name for path in paths]==[f'{i:04d}.json' for i in range(ending['plies'])],
                      'Operational move-file inventory differs')
            moves=[c.read(path) for path in paths]
            for row in moves:
                envelope.guard(); verify_saved_decision(row); decisions+=1
            game_rows.extend(moves)
            games.append(replay_game(initial['initial'],moves,ending,spec,boundary=envelope.guard))
            for role in ('challenger','opponent'):
                samples.append(next(row for row in moves if row['game_role']==role))
        c.require(summarize_operational(game_rows,len(games))==summaries['operational']==c.read(c.OUT/'operational_summary.json'),
                  'Independent reread of operational timing gate differs')
    c.require(len(games)==(6 if 'operational' in summaries else 0) and len(samples)==2*len(games),
              'Complete six-game/twelve-reproduction inventory required')
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=len(games),receipts=games))
    repetitions=[]
    for index,original in enumerate(samples):
        with make_worker(f'reproduction_{index:02d}',envelope,freeze,mode='candidate',quota=original['quota']) as child:
            repeated=decision(child,decode_training_snapshot(original['before']),envelope)
            save_and_check(repeated,c.OUT/'repeated_decisions'/f'{index:02d}.json',envelope)
            c.require(repeated['action']==original['action'] and repeated['trace']==original['trace'],
                      'Full operational decision reproduction differs')
            verify_saved_decision(repeated)
        repetitions.append(dict(index=index,game_index=original['game_index'],ply=original['ply'],quota=original['quota']))
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(repetitions),receipts=repetitions,
        selection='First move by each exact player in each operational game; outcome/timing independent',
        deadline_seconds=1.,margin_gate=False))
    checks=[]; graph_checks=[]
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard()
        checks.append(verify_worker_logs(folder,boundary=envelope.guard,deadline=envelope.effective_deadline()))
        metadata=c.read(folder/'initialization.json')['result']
        if 'inference_backend' in metadata:
            graph_checks.append(verify_graph_evidence(folder,freeze,boundary=envelope.guard))
        else:
            c.require(not any(folder.glob('graph_*.json')),'Unexpected graph evidence in baseline worker')
    c.require(len(checks)==8+2*len(games)+len(samples),'Qualification/game/reproduction worker inventory differs')
    c.require(len(graph_checks)==4+2*len(games)+len(samples) and
        len({row['weights_sha256'] for row in graph_checks})==1,'Candidate graph/weight instance inventory differs')
    c.save(c.OUT/'all_graph_parity.json',dict(status='passed',workers=len(graph_checks),receipts=graph_checks,
        independent_full_outputs_and_float32_bits=True))
    c.save(c.OUT/'all_worker_log_integrity.json',dict(status='passed',workers=len(checks),receipts=checks))
    seal=[]
    for directory in ('paired_timing','games','workers','repeated_decisions'):
        for path in sorted((c.OUT/directory).rglob('*')):
            if path.is_file():
                envelope.guard(); seal.append(c.artifact(path))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:
        envelope.guard(); c.verify([item])
    verify_runtime(freeze,envelope)
    c.require(predecessor_acceptance()==freeze['baseline_acceptance'],'Preserved baseline/failure evidence changed')
    c.save(c.OUT/'verification_summary.json',dict(status='passed',decisions=decisions,
        operational_games=len(games),operational_reproductions=len(samples),
        maximum_sampled_combined_worker_rss_bytes=envelope.peak_combined_rss))


def run_stages(envelope,freeze,plan,schedule,*,qualification_started=False):
    if qualification_started:
        c.require(envelope.stage=='qualification','Startup qualification clock missing')
    else:
        envelope.begin('qualification')
    summaries={'qualification':qualify_panel(envelope,freeze,plan)}
    envelope.finish()
    if gate_passed(summaries['qualification']):
        envelope.begin('operational')
        summaries['operational']=operational(envelope,freeze,schedule)
        envelope.finish()
    envelope.begin('final')
    final_checks(envelope,freeze,plan,schedule,summaries)
    envelope.finish()
    admitted='operational' in summaries and gate_passed(summaries['operational'])
    c.save(c.OUT/'qualification_outcome.json',dict(status='completed_verified',inference_graph_qualified=admitted,
        saved_panel_qualified=gate_passed(summaries['qualification']),summaries=summaries,
        strength_games_launched=0,automatic_strength_launch=False,larger_data_deferred=True))
    return summaries


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Freeze identity changed')
    claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze identity changed')
    freeze,envelope=c.read(c.OUT/'freeze.json'),Envelope(claim)
    envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status,failure='incomplete',None
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope)
        envelope.admit()
        plan,schedule=c.read(c.OUT/'qualification_plan.json'),c.read(c.OUT/'operational_schedule.json')
        c.require(schedule==operational_schedule(),'Frozen operational schedule differs')
        run_stages(envelope,freeze,plan,schedule,qualification_started=True)
        status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,automatic_retries=0,larger_data_deferred=True,strength_games=0,
        note='Acceptance additionally requires owning-watchdog cleanup. No200-game benchmark is part of this attempt.'))
    print(c.canonical(dict(status=status,stage=envelope.stage,failure=failure)),flush=True)
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['prepare','run'])
    parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    return prepare() if args.operation=='prepare' else run(args.freeze_sha256)


if __name__=='__main__':
    raise SystemExit(main())
