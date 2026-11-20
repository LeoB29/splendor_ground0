"""One fresh interleaved admission attempt; no assessment execution path."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from importlib.metadata import version
import psutil
from . import common as c, placement
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.future_training_rules_20261021 import verify_runtime_sources, SOURCE_HASHES
from tools.local_player_qualification_128.clock import PersistentWorker, THREAD_ENV, _WindowsJob, _new_json, WorkerInitializationError
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules
from tools.splendorust_boundary_timing_20261025.runner import verify_instrumentation_identity, BACKEND
SIDECAR_WAIT_SECONDS = 1.
OBSERVER_MODULE = 'tools.splendorust_boundary_timing_20261025.worker'
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
        self.last_resource_guard = -float("inf")

    def enable_evidence_accounting(self):
        self.evidence = dict(baseline=bytes_used(), parent_bytes=0, workers={})
        self.watchdog_control_baseline = self.watchdog_control_bytes()
        self.log_baseline = {name: (c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                             for name in ('runner_stdout.log', 'runner_stderr.log')}
        c.WRITE_OBSERVER = self.note_write

    def watchdog_control_bytes(self):
        paths=list((c.OUT/'placement_acks').glob('*'))+list(c.OUT.glob('watchdog*.json'))+list(c.OUT.glob('guardian*.json'))
        return sum(path.stat().st_size for path in paths if path.is_file())

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
        return total + max(0,self.watchdog_control_bytes()-self.watchdog_control_baseline) + 65536

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
        current=time.monotonic()
        if current-self.last_resource_guard<1.:return
        self.last_resource_guard=current
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        cap = c.MAX_EVIDENCE_BYTES if self.stage == 'final' else c.PREFINAL_EVIDENCE_BYTES
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
            c.require(self.waited+current-begun < c.RAM_WAIT_SECONDS, 'Cumulative frozen RAM wait exceeded')
            if stable is not None and current-stable >= 60:
                break
            time.sleep(.5)
        elapsed = time.monotonic()-begun
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.require(self.waited < c.RAM_WAIT_SECONDS, 'Cumulative RAM admission bound')
        if not fast:
            c.save(c.OUT/'memory_admissions'/f'{self.admissions:06d}.json', dict(time=c.now(), wait_seconds=elapsed,
                cumulative_wait_seconds=self.waited, minimum_available_bytes=minimum,
                stable_seconds=60))

    def finish(self):
        self.last_resource_guard=-float("inf")
        self.guard()
        if self.evidence is not None:
            c.require(bytes_used() <= self.accounted_bytes(), 'Unaccounted study evidence')
        c.save(c.OUT/'stages'/f'{self.stage}_end.json', dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time()))


def runtime_placement_mode(freeze,envelope=None):
    c.require(freeze.get('placement_mode')=='isolated','Freeze must require isolated supervision')
    mode=getattr(envelope,'placement_mode',freeze['placement_mode'])
    c.require(mode=='isolated','Runtime or preparation supervisor must remain isolated')
    return mode


def verify_runtime(freeze, envelope=None):
    validate_freeze_contract(freeze)
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python runtime changed')
    placement.verify_host(freeze)
    placement.verify_runtime(freeze,runtime_placement_mode(freeze,envelope))
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
    c.require(version('numpy')==freeze['numpy_version'] and version('torch')==freeze['torch_package_version'],
              'Numerical runtime package changed')


def validate_freeze_contract(freeze):
    expected=dict(job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES,max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,max_rss_bytes=2*c.GiB,
        minimum_available_ram_bytes=2*c.GiB,initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,seconds_per_move=1.,internal_seconds=.9,
        research_seconds_per_move=1.,research_internal_seconds=.9,research_log_max_bytes=128*c.MiB,
        frozen_log_max_bytes=16*c.MiB,research_response_max_bytes=4*c.MiB,frozen_response_max_bytes=256*1024,
        request_max_bytes=c.MiB,placement_mode='isolated',max_workers=c.MAX_WORKERS,max_graph_builds=c.MAX_GRAPH_BUILDS,
        phase='qualification',qualification_decisions=194,operational_games=6,main_games=0,
        maximum_reproductions=48,automatic_retries=0,failed_decision_retries=0,training=False)
    c.require(all(freeze.get(key)==value for key,value in expected.items()),'Exact frozen clock/resource/workload contract')
    return expected


def measure_seal_cost(folders, boundary):
    """Measure both real final-seal reads over closed operational evidence."""
    begun=time.perf_counter(); artifacts=[]
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                boundary(); artifacts.append(c.artifact(path))
    for item in artifacts:
        boundary(); c.verify([item])
    return dict(seconds=time.perf_counter()-begun,passes=2,files=len(artifacts),
                bytes=sum(item['bytes'] for item in artifacts),inventory_sha256=c.digest(artifacts))


def switch_placement(envelope,freeze,mode):
    """Switch both owned supervisors only with no resident player trees."""
    c.require(mode=='isolated','This trial authorizes only isolated supervision')
    c.require(not envelope.resident,'Cannot switch placement while workers are resident')
    envelope.guard();before=time.monotonic();end=before+min(5.,envelope.remaining())
    path=c.OUT/'watchdog.json'
    while not path.exists() and time.monotonic()<end:
        envelope.guard();time.sleep(.005)
    c.require(path.exists(),'Owning watchdog identity unavailable')
    watchdog=c.read(path);claim=c.read(c.OUT/'watchdog_claim.json')
    identity=dict(pid=os.getpid(),created=psutil.Process().create_time())
    c.require((watchdog['runner_pid'],watchdog['runner_created'])==(identity['pid'],identity['created']) and
        (watchdog['watchdog_pid'],watchdog['watchdog_created'])==(claim['watchdog_pid'],claim['watchdog_created']) and
        watchdog['freeze_sha256']==claim['freeze_sha256'],'Watchdog ownership identity differs')
    owner=psutil.Process(watchdog['watchdog_pid'])
    c.require(owner.create_time()==watchdog['watchdog_created'] and owner.is_running(),'Owning watchdog absent/reused')
    receipt=placement.configure_execution(freeze,mode)
    sequence=getattr(envelope,'placement_sequence',0)+1
    request=dict(sequence=sequence,placement_mode=mode,freeze_sha256=claim['freeze_sha256'],
        runner=identity,watchdog=dict(pid=owner.pid,created=owner.create_time()),runner_receipt=receipt)
    c.atomic_save(c.OUT/'placement_requests'/f'{sequence:04d}.json',request)
    ackpath=c.OUT/'placement_acks'/f'{sequence:04d}.json'
    while not ackpath.exists() and time.monotonic()<end:
        envelope.guard();time.sleep(.005)
    c.require(ackpath.exists(),'Watchdog placement acknowledgement missing within5s')
    ack=c.read(ackpath);c.validate_exchange(request,ack,freeze,watchdog,identity)
    c.require(time.monotonic()<end,'Watchdog placement acknowledgement exceeded5s')
    c.require(owner.cpu_affinity()==freeze['placement']['role_masks'][mode] and
        int(owner.nice())==placement.NORMAL_PRIORITY,'Watchdog current placement differs from acknowledgement')
    envelope.placement_mode=mode;envelope.placement_sequence=sequence
    envelope.guard()


def verify_exchanges(envelope,freeze):
    requests=sorted((c.OUT/'placement_requests').glob('*.json'))
    acks=sorted((c.OUT/'placement_acks').glob('*.json'))
    c.require([p.name for p in requests]==[f'{i:04d}.json' for i in range(1,envelope.placement_sequence+1)] and
        [p.name for p in requests]==[p.name for p in acks],'Placement handshake inventory differs')
    watchdog=c.read(c.OUT/'watchdog.json');launch=c.read(c.OUT/'launch.json')
    identity=dict(pid=launch['pid'],created=launch['created']);timeline=[]
    for request,ack in zip(requests,acks,strict=True):
        envelope.guard();r,a=c.read(request),c.read(ack)
        c.require(r['placement_mode']=='isolated','Unexpected nonisolated controller placement')
        c.require(r['sequence']==int(request.stem),'Placement request sequence differs from saved filename')
        c.validate_exchange(r,a,freeze,watchdog,identity)
        timeline.append(dict(mode=r['placement_mode'],start=a['watchdog_receipt']['monotonic_ns'],
            transition_start=r['runner_receipt']['monotonic_ns']))
    for folder in (c.OUT/'workers').iterdir():
        admission=c.read(folder/'placement_admission.json');stamp=admission['supervisor']['monotonic_ns']
        intervals=[(i,x) for i,x in enumerate(timeline) if x['start']<=stamp]
        c.require(intervals,'Worker started before acknowledged supervisory placement')
        i,item=intervals[-1]
        c.require(item['mode']==admission['placement_mode'],'Worker admission differs from acknowledged placement')
        for path in (folder/'placement_requests').glob('*.json'):
            value=c.read(path);before=value['before']['monotonic_ns'];after=value['after']['monotonic_ns']
            c.require(item['start']<=before<=after and (i+1==len(timeline) or after<timeline[i+1]['transition_start']),
                'Worker request overlaps unacknowledged placement switch')
    return dict(status='passed',exchanges=len(timeline),owned_watchdog_and_runner=True)


class _PlacementConstructor(PersistentWorker):
    # Local constructor copy. The sole body substitution is independently
    # checked by AST regression; all trusted transport/timing methods inherit.
    def __init__(self, module, output_dir, *, module_args=(), affinity=None,
                 initialization_timeout=60.0, env=None, max_message_bytes=2 * 1024**2,
                 max_request_bytes=1024**2, max_log_bytes=32 * 1024**2,
                 max_rss_bytes=2 * 1024**3, absolute_deadline=None, resource_guard=None):
        if (not isinstance(module, str) or not module or
                not math.isfinite(initialization_timeout) or initialization_timeout <= 0):
            raise ValueError("Invalid worker initialization")
        if getattr(self, '_research_native_roles', None) is not None:
            c.require(set(self._natives) == set(self._research_native_roles), 'Research native inventory differs')
            self._natives = dict(self._research_native_roles)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.max_message_bytes, self.max_request_bytes = max_message_bytes, max_request_bytes
        self.max_log_bytes, self.max_rss_bytes = max_log_bytes, max_rss_bytes
        self.absolute_deadline = absolute_deadline  # Parent time.monotonic() clock.
        self.resource_guard = resource_guard
        self._last_guard_sample = None
        self._queue = queue.Queue(maxsize=128)
        self._failure = None
        self._failure_detail = None
        self._closed = False
        self._counter = 0
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._logs = {name: (self.output_dir / name).open("xb")
                      for name in ("stdout.log", "stderr.log", "events.jsonl")}
        self._log_sizes = dict.fromkeys(self._logs, 0)
        self._threads = []
        self._job = None
        self.process = None
        self._ps = None
        self._peak_rss = 0
        self._last_telemetry = {}
        try:
            self.topology = placement.admit_worker_topology(self._placement_freeze, self.placement_mode)
        except BaseException:
            self.close()
            raise
        self.affinity = list(affinity if affinity is not None else
                             [self.topology["selected_logical_processor"]])
        if len(self.affinity) != 1 or self.affinity[0] not in self.topology["allowed_logical_processors"]:
            self.close()
            raise ValueError("Worker must use one allowed logical processor")
        child_env = os.environ.copy()
        child_env.update(env or {})
        child_env.update({name: "1" for name in THREAD_ENV})
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-B", "-u", "-m", module, *map(str, module_args)]
        started = time.perf_counter_ns()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=child_env, bufsize=0,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self._ps = psutil.Process(self.process.pid)
            self._job = _WindowsJob(self.process.pid)
            self._ps.cpu_affinity(self.affinity)
            if os.name == "nt":
                self._ps.nice(psutil.NORMAL_PRIORITY_CLASS)
            self.identity = {"pid": self.process.pid, "created": self._ps.create_time()}
            for function in (self._stdout_reader, self._stderr_reader):
                thread = threading.Thread(target=function, daemon=True)
                thread.start()
                self._threads.append(thread)
            receipt = self._receive(started, initialization_timeout, "ready", None, None)
            self.initialization = receipt
            _new_json(self.output_dir / "initialization.json", {**receipt, "command": command,
                      "identity": self.identity, "affinity": self.affinity, "topology": self.topology,
                      "thread_environment": {name: child_env[name] for name in THREAD_ENV}})
            if receipt["status"] != "ok":
                self.close()
                raise WorkerInitializationError(receipt)
        except BaseException:
            self.close()
            raise


class PlacedProfiledStudyWorker(ProfiledStudyWorker, _PlacementConstructor):
    def __init__(self, *args, placement_freeze, placement_mode, native_roles=None, aggregate_sample_guard=None, **kwargs):
        self._research_native_roles = native_roles
        self._placement_freeze = placement_freeze
        self.placement_mode = placement_mode
        self.last_placement_evidence = None
        self.last_placement_error = None
        placement.verify_runtime(placement_freeze, placement_mode)
        def guard(worker, telemetry):
            if aggregate_sample_guard is not None:
                aggregate_sample_guard(worker, telemetry)
            placement.verify_runtime(placement_freeze, placement_mode, full=False)
        self._placement_constructing = True
        placement.register_worker(self)
        try:
            super().__init__(*args, aggregate_sample_guard=guard, **kwargs)
        finally:
            self._placement_constructing = False
        try:
            c.require(self._ps.cpu_affinity() == [0] and int(self._ps.nice()) == placement.NORMAL_PRIORITY,
                      'Player placement/priority changed at admission')
            proof = dict(schema=placement.SCHEMA, placement_mode=placement_mode,
                player_pid=self.identity['pid'], player_created=self.identity['created'],
                player_mask=[0], player_priority_class=placement.NORMAL_PRIORITY,
                supervisor=placement.verify_runtime(placement_freeze, placement_mode))
            c.save(self.output_dir/'placement_admission.json', proof)
            placement.verify_worker_placement(self.output_dir, placement_freeze, placement_mode)
        except BaseException:
            self.close()
            raise


def request(child, payload, timeout, validate=None):
    """Symmetric placement evidence outside the unchanged original move clock."""
    child.last_placement_evidence = None
    child.last_placement_error = None
    before = placement.verify_runtime(child._placement_freeze, child.placement_mode)
    receipt = child.request(payload, timeout, validate=validate)
    after, error = None, None
    try:
        after = placement.verify_runtime(child._placement_freeze, child.placement_mode)
    except BaseException as caught:
        error = repr(caught)
        child.last_placement_error = error
    evidence = dict(schema=placement.SCHEMA, placement_mode=child.placement_mode,
        request_id=receipt['request_id'], player_pid=child.process.pid,
        before=before, after=after, error=error,
        checks_outside_original_request_clock=True,
        sample_checks_inside_original_clock=True,
        original_clock_and_failure_receipt_preserved=True)
    child.last_placement_evidence = evidence
    try:
        c.save(child.output_dir/'placement_requests'/f"{receipt['request_id']}.json", evidence)
    except BaseException as caught:
        child.last_placement_error = repr(caught)
    return receipt


def collect_profiles(child,receipt,envelope):
    """Preserve available evidence without extending or resetting move clocks.

    A caught failure or killed worker may lack a child sidecar. That disclosed
    absence is not a second failure. Any malformed or unbound evidence is an
    integrity error for the caller to preserve before stopping the study.
    """
    result=dict(placement_mode=child.placement_mode,placement_evidence=deepcopy(child.last_placement_evidence),
                parent_profile=None,parent_profile_file=None,child_profile=None,
                child_profile_file=None,profile_capture_status='pending')
    if child.last_placement_error is not None:
        result['placement_error']=child.last_placement_error
    try:
        profile=deepcopy(child.last_parent_profile)
        c.require(type(profile) is dict and profile['request_id']==receipt['request_id'],
                  'Parent observer request identity')
        parent=child.output_dir/'parent_profiles'/f"{receipt['request_id']}.json"
        result.update(parent_profile=profile,parent_profile_file=parent.relative_to(c.ROOT).as_posix())
        c.save(parent,profile)
        path=child.output_dir/f"{receipt['request_id']}_child_profile.json"
        begun=time.monotonic();wait_end=begun+min(SIDECAR_WAIT_SECONDS,max(0.,envelope.remaining()))
        while not path.exists() and not getattr(child,'_closed',False) and time.monotonic()<wait_end:
            envelope.guard();time.sleep(min(.005,max(0.,wait_end-time.monotonic())))
        if path.exists():
            c.require(path.stat().st_size<=512*1024+1,'Bounded child observer sidecar')
            result.update(child_profile=c.read(path),child_profile_file=path.relative_to(c.ROOT).as_posix(),
                          profile_capture_status='complete')
        elif receipt['status']=='ok':
            raise ValueError('Completed request missing required child observer sidecar')
        else:
            result['profile_capture_status']='missing_child_after_failed_request'
        return result
    except BaseException as error:
        result.update(profile_capture_status='evidence_error',profile_capture_error=repr(error))
        return result


def _require_profiles(profiles):
    c.require('placement_error' not in profiles,'Supervisor placement evidence error: '+str(profiles.get('placement_error')))
    c.require('profile_capture_error' not in profiles,'Observer evidence integrity error: '+str(profiles.get('profile_capture_error')))


def make_worker(name, envelope, freeze, *, route):
    """One of at most two live trees; original clocks/readers/sampler retained."""
    from . import adapters
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    c.require(route in ('frozen', 'research'), 'Worker route allowlist')
    c.require(len(envelope.resident)<2, 'At most two resident live player trees are authorized')
    c.require('torch' not in sys.modules, 'Model-free supervisor required')
    c.require(len(envelope.evidence['workers']) < c.MAX_WORKERS, 'Worker lifecycle bound')
    envelope.admit()
    folder = c.OUT/'workers'/name
    envelope.register_worker(folder)
    research = route == 'research'
    module = 'tools.splendorust_pool6_admission_20261027_02.worker' if research else OBSERVER_MODULE
    artifacts = freeze['research_native_artifacts'] if research else freeze['native_artifacts']
    roles = None
    if research:
        mapping = freeze['research_native_roles']
        c.require(set(mapping) == {a['path'] for a in artifacts} and sorted(mapping.values()) == ['reference','search'],
                  'Authenticated two-native research role mapping')
        roles = {(c.ROOT/path).resolve(): role for path, role in mapping.items()}
    env = {'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'STUDY_WORKER_DIR': str(folder),
           'STUDY_SEARCH_SIMULATIONS': '64', 'STUDY_BOUNDARY_PROFILE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1', 'SPLENDOR_POOL6_ADMISSION_PHASE':c.PHASE}
    if research:
        env.update(STUDY_RESEARCH_FREEZE=str(c.OUT/'freeze.json'),
                   STUDY_RESEARCH_FREEZE_SHA256=c.file_hash(c.OUT/'freeze.json'))
    child = PlacedProfiledStudyWorker(module, folder, placement_freeze=freeze, placement_mode='isolated',
        native_roles=roles, native_artifacts=artifacts, aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[0], initialization_timeout=envelope.timeout(60), absolute_deadline=envelope.effective_deadline(),
        max_message_bytes=4*c.MiB, max_request_bytes=c.MiB, max_log_bytes=(128 if research else 16)*c.MiB,
        max_rss_bytes=2*c.GiB, resource_guard=envelope.guard, env=env)
    try:
        envelope.own_worker(child)
        child.study_after_close = lambda: envelope.close_worker(folder, child)
        child.route, child.runtime_mode, child.challenger_quota = route, 'candidate', 64
        child.observation_mode = 'profiled'
        telemetry = child._sample()
        metadata = deepcopy(child.initialization['result'])
        if research:
            adapters.validate_ready(metadata, child, freeze)
        else:
            c.require(metadata.pop('inference_backend') == BACKEND, 'Frozen graph backend changed')
            validate_ready_metadata(metadata, expected_sources=SOURCE_HASHES, worker_pid=child.process.pid,
                process_tree=telemetry['process_tree'], expected_quota=64)
            verify_instrumentation_identity(folder, freeze, 64, child.initialization['result'])
        c.require(not child._failure, 'Worker supervision failed at readiness')
        c.save(folder/'route_identity.json', dict(route=route, module=module, native_artifacts=artifacts,
            native_roles=None if roles is None else freeze['research_native_roles'], metadata=child.initialization['result']))
        adapters.graph_parity(child, envelope, freeze, request_fn=request, collect_fn=collect_profiles)
        clock=c.read(folder/'graph_parity_clock.json');profiles=c.read(folder/'graph_parity_profiles.json')
        payload=c.read(folder/'startup_inputs'/(clock['request_id']+'.json'))
        validate_profiles(profiles['parent_profile'],profiles['child_profile'],clock,payload,route,worker_pid=child.process.pid)
        verify_graph_evidence(folder,freeze,boundary=envelope.guard)
        envelope.guard()
        return child
    except BaseException:
        child.close()
        envelope.close_worker(folder, child)
        raise


def ordinary(child, spec, observer, envelope, freeze):
    """Exact two-field live payload; original full clock and rejected evidence."""
    from . import audit
    envelope.admit()
    c.require(spec.get('quota',64)==64 and spec.get('route',child.route)==child.route,'Fixed live route/quota')
    payload=dict(player='search',observer=observer)
    request_id=f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json',payload)
    identity=audit.profile_identity(freeze,64,6,boundary=envelope.guard) if child.route=='research' else None
    legal=[rules.serialize_action(a) for a in rules.legal_actions(rules.state_from_observation(observer))]
    def validate(result):
        if type(result) is not dict or 'action' not in result:raise ValueError('PLAYER_MALFORMED_RESPONSE: missing action')
        if result['action'] not in legal:raise ValueError('PLAYER_ILLEGAL_ACTION: response action')
        return audit.validate_result(result,observer,child.route,identity=identity)
    c.require(envelope.remaining()>=1.,'Insufficient stage time for an unchanged full request clock')
    allowance=envelope.timeout(1.)
    c.require(allowance==1.,'Do not shorten or reset the request allowance')
    receipt=request(child,payload,allowance,validate=validate)
    row=dict(spec=deepcopy(spec),observer=observer,completed=receipt['status']=='ok',clock=receipt,
        route=child.route,worker_folder=child.output_dir.relative_to(c.ROOT).as_posix(),
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix(),
        observation_mode='profiled',action=None,trace=None)
    row.update(collect_profiles(child,receipt,envelope))
    if row['completed']:row.update(action=receipt['result']['action'],trace=receipt['result']['diagnostics'])
    return row


def deadline_failure(receipt,route):
    """Only original clock or explicit internal-clock failures are diagnostic negatives."""
    from . import audit
    limit=1.
    audit.check_clock(receipt,limit=limit)
    if receipt['status']=='timeout':
        return (receipt['elapsed_seconds']>=limit and receipt['received_before_deadline'] is False and
                (receipt['receipt_elapsed_seconds'] is None or receipt['receipt_elapsed_seconds']>limit))
    marker='ONE_SECOND_INTERNAL_DEADLINE:' if route=='frozen' else 'SAMPLING_INTERNAL_DEADLINE:'
    return receipt['status']=='worker_error' and marker in str(receipt.get('detail',''))


def close_and_audit(child, envelope, freeze):
    begun=time.perf_counter()
    child.close()
    # Failed original requests close before the late observer and placement files.
    envelope.close_worker(child.output_dir, child)
    envelope.guard()
    proof = verify_closed_worker(child.output_dir, freeze, envelope)
    envelope.close_worker(child.output_dir)
    c.save(c.OUT/'closed_workers'/f'{child.output_dir.name}.json', proof)
    c.save(c.OUT/'closed_worker_costs'/f'{child.output_dir.name}.json',dict(seconds=time.perf_counter()-begun,
        bytes=sum(p.stat().st_size for p in child.output_dir.rglob('*') if p.is_file())))
    envelope.guard()
    return proof


from tools.splendorust_boundary_timing_20261025 import analyze as boundaries
from tools.splendorust_boundary_timing_20261025.analyze import CHILD_SCHEMA, GRAPH_IDENTITY, _int, _finite, _sha, _digest
from tools.splendorust_live_boundary_20261025.raw_transport import _packet_evidence, _bind_parent_wire, _bind_child_wire, REQUEST_ID, MAX_FAILED_LINES
from tools.splendorust_search_scale_20261024.compression import verify_log
require = c.require

def research_child_identity(child, parent, receipt, payload, quota, *, completed, startup=False, worker_pid=None):
    require(type(child) is dict and child.get("schema") == CHILD_SCHEMA, "child sidecar schema")
    require(len(json.dumps(child, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <=
            (512*1024 if startup else 8192), "bounded child sidecar")
    require(child.get("request_id") == receipt["request_id"] and child.get("quota") == quota, "child request/quota")
    require(child.get("backend") == "research_graph" and
            child.get("graph_identity_sha256") == GRAPH_IDENTITY, "child backend identity")
    require(child.get("pid") == parent["child_pid"] and (worker_pid is None or child["pid"] == worker_pid),
            "parent/child supervised PID binding")
    python_pids = {p["pid"] for p in receipt.get("telemetry", {}).get("process_tree", []) if p.get("role") == "python"}
    require(python_pids == {child["pid"]}, "sampled Python PID identity differs")
    require(child.get("clock") == "perf_counter_ns" and child.get("clock_implementation") == "QueryPerformanceCounter()",
            "shared cross-process QPC runtime")
    for field in ("times_are_diagnostic_not_trusted_deadline_receipts", "success_sidecar_written_after_response_flush",
                  "post_flush_may_follow_parent_receipt", "child_read_may_precede_parent_dispatch_flush_completion",
                  "killed_worker_sidecar_may_be_missing", "text_transport_may_expand_newlines"):
        require(child.get(field) is True, "child sidecar scope " + field)
    require(child.get("outcome") in ("response_flushed", "failed_before_successful_emission"), "child outcome")
    require(not completed or child["outcome"] == "response_flushed", "successful request has failed sidecar")
    require(child.get("payload_sha256") == _digest(payload), "child input content binding")
    require(parent["request_raw_complete"] and not parent["request_raw_overflow"] and
            child.get("request_raw_sha256") == parent["request_raw_sha256"], "actual transported request hash differs")
    _sha(child.get("result_sha256"), "scientific result", nullable=not completed)
    if completed:
        require(child["result_sha256"] == _digest(receipt["result"]), "child scientific result binding")
        require(child.get("error") is None, "successful child error")
    else:
        require(child.get("error") is None or type(child.get("error")) is str and len(child["error"]) <= 4096,
                "bounded child error")

def research_boundaries(child, parent, start, arrival, *, begin=None, end=None, startup=False):
    read = _int(child.get("request_read_complete_ns"), "child input complete")
    require(start <= read <= parent["capture_snapshot_ns"], "child input outside parent request")
    origin = _finite(child.get("request_started_seconds"), "internal clock origin")
    require(0 <= read-origin*1e9, "internal origin/sample order")
    metrics = {"parent_start_to_child_read_ns": read-start}
    if begin is not None:
        metrics.update(child_read_to_handle_start_ns=begin-read, handle_wall_ns=end-begin)
    pre, post = child.get("pre_write_ns"), child.get("post_flush_ns")
    if child["outcome"] == "response_flushed":
        _int(pre, "pre-write"); _int(post, "post-flush")
        require(read <= pre <= post and (end is None or end <= pre), "child emission order")
        require(arrival is None or pre <= arrival, "pre-write occurs after trusted complete response")
        _sha(child.get("serialized_packet_utf8_sha256"), "serialized response")
        size = _int(child.get("serialized_packet_utf8_bytes"), "serialized response bytes", positive=True)
        require(size <= 4*1024*1024, "serialized response size bound")
        if arrival is not None:
            raw_size = parent["response_raw_bytes"]
            require(raw_size-size in (0, 1), "only terminal LF/CRLF response size difference permitted")
            if raw_size == size:
                require(parent["response_raw_sha256"] == child["serialized_packet_utf8_sha256"],
                        "equal-length raw response hash differs")
        metrics["write_and_flush_wall_ns"] = post-pre
        if end is not None:
            metrics["handle_end_to_pre_write_ns"] = pre-end
        if arrival is not None:
            metrics["pre_write_to_trusted_arrival_ns"] = arrival-pre
            metrics["signed_post_flush_to_trusted_arrival_ns"] = arrival-post
            if end is not None:
                partition = {"parent_start_to_child_read_ns": read-start, "child_read_to_handle_start_ns": begin-read,
                             "child_handle_ns": end-begin, "handle_end_to_pre_write_ns": pre-end,
                             "pre_write_to_trusted_arrival_ns": arrival-pre}
                require(sum(partition.values()) == arrival-start, "nonoverlapping boundary partition")
                metrics["ordered_boundary_segments_ns"] = partition
    else:
        require(post is None and child.get("serialized_packet_utf8_sha256") is None and
                child.get("serialized_packet_utf8_bytes") is None, "failed-before-emission marker contradiction")
        # A write/flush exception can follow the marker but cannot certify a
        # complete response. Keep the attempted-write boundary without making
        # a completed delivery segment or packet binding out of partial I/O.
        if pre is not None:
            _int(pre, "partial pre-write")
            require(read <= pre and (end is None or end <= pre), "partial child emission order")
            metrics["partial_emission_attempt_recorded"] = True
            if end is not None:
                metrics["handle_end_to_partial_pre_write_ns"] = pre-end
    dispatch = parent.get("dispatch_complete_ns")
    if dispatch is not None:
        metrics["signed_dispatch_complete_to_child_read_ns"] = read-dispatch
    return metrics

def verify_raw_transport(folder,freeze,envelope):
    folder=Path(folder);cleanup=c.read(folder/'cleanup.json');init=c.read(folder/'initialization.json')
    metadata=init['result'];route=c.read(folder/'route_identity.json')['route'];quota=64;mode=route
    c.require(route in ('frozen','research'),'Only frozen64 and research routes')
    paths=sorted(path for path in folder.glob('request-*.json') if REQUEST_ID.fullmatch(path.stem))
    c.require(len(paths)<=2002,'Bounded request inventory')
    c.require([p.stem for p in paths]==[f'request-{i:08d}' for i in range(1,len(paths)+1)],
              'Contiguous original worker request inventory')
    receipts={path.stem:c.read(path) for path in paths};ids=list(receipts)
    failures=[rid for rid,receipt in receipts.items() if receipt['status']!='ok']
    c.require(not failures or failures==[ids[-1]],'At most one final failed request per worker')
    failed_id=failures[0] if failures else None
    packets={rid:[] for rid in ids};pending=bytearray();ready=False;index=0;partial=None
    def packet_line(raw):
        nonlocal ready,index
        envelope.guard();c.require(len(raw)<=4*c.MiB and raw.endswith(b'\n'),'Raw packet frame bound')
        item=_packet_evidence(raw);packet=item['packet']
        if not ready:
            c.require(packet=={'type':'ready','metadata':metadata},'Exact raw READY metadata changed')
            ready=True;return
        c.require(index<len(ids),'Unsolicited extra raw packet after successful final request')
        rid=ids[index];receipt=receipts[rid]
        if receipt['status']=='ok':
            c.require(type(packet) is dict and set(packet)=={'type','request_id','result'} and
                packet['type']=='response' and packet['request_id']==rid and packet['result']==receipt['result'],
                'Completed raw response differs from original trusted receipt')
            packets[rid].append(item);index+=1
        else:
            c.require(rid==failed_id and len(packets[rid])<MAX_FAILED_LINES,'Bounded failed raw packet suffix')
            packets[rid].append(item)
    def sink(chunk):
        pending.extend(chunk)
        while True:
            end=pending.find(b'\n')
            if end<0:break
            raw=bytes(pending[:end+1]);del pending[:end+1];packet_line(raw)
        c.require(len(pending)<=4*c.MiB,'Raw pending response frame bound');envelope.guard()
    verified=verify_log(folder/'stdout.log.gz',cleanup['log_compression']['stdout.log'],
        max_raw_bytes=(128 if route=='research' else 16)*c.MiB,max_compressed_bytes=(128 if route=='research' else 16)*c.MiB*16+128,
        deadline=envelope.effective_deadline(),sink=sink)
    c.require(ready,'Closed worker raw READY missing')
    if pending:
        c.require(failed_id is not None and index==len(ids)-1,'Partial output without final failed request')
        partial=dict(bytes=len(pending),sha256=hashlib.sha256(pending).hexdigest(),complete_packet=False)
    c.require(index==len(ids) if failed_id is None else index==len(ids)-1,
              'Missing completed raw response')
    input_ids=[p.stem for directory in ('inputs','startup_inputs') for p in (folder/directory).glob('*.json')]
    c.require(len(input_ids)==len(ids) and set(input_ids)==set(ids),'Exact input request inventory')
    expected_profiles={rid+'_child_profile.json' for rid in ids}
    c.require({p.name for p in folder.glob('request-*_child_profile.json')}<=expected_profiles,
              'Unexpected child sidecar inventory')
    c.require({p.stem for p in (folder/'parent_profiles').glob('*.json')}==set(ids),
              'Complete parent sidecar inventory required')
    startup_count=0;checks=[]
    for rid,receipt in receipts.items():
        envelope.guard();c.require(receipt['request_id']==rid,'Raw trusted receipt identity')
        startup=folder/'startup_inputs'/(rid+'.json');ordinary=folder/'inputs'/(rid+'.json')
        c.require(startup.is_file()!=ordinary.is_file(),'Exactly one saved request payload type required')
        if startup.is_file():
            payload=c.read(startup);startup_count+=1
            c.require(receipt['status']=='ok','Admission failure cannot be accepted as a live-player failure')
            if payload=={'operation':'entrypoint_checks'}:
                c.require(folder.name.startswith('qualification_'),'Unexpected entrypoint request during play')
            else:
                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
                c.require(quota==64 and payload=={'operation':'graph_parity','fixtures':plan['fixtures']},
                          'Exact original graph startup payload required')
        else:
            payload=c.read(ordinary)
            c.require(set(payload)=={'player','observer'} and payload['player']=='search','Exact fixed live payload')
            c.require(receipt['deadline_seconds']==1.,'Exact full one-second route allowance required')
            if route=='frozen':
                for packet in packets[rid]:
                    c.require(packet['raw_bytes']<=256*1024+1,'Frozen ordinary raw response exceeds unchanged256KiB cap')
        parent=c.read(folder/'parent_profiles'/(rid+'.json'))
        child_path=folder/(rid+'_child_profile.json')
        if child_path.exists():
            c.require(child_path.stat().st_size<=(512*1024+1 if startup.is_file() else 8193),
                      'Original bounded child sidecar size')
            child=c.read(child_path)
        else:
            c.require(receipt['status']!='ok','Completed child sidecar missing');child=None
        c.require(parent['request_id']==rid,'Parent sidecar request binding')
        parent_observed=_bind_parent_wire(parent,packets[rid],completed=receipt['status']=='ok')
        if child is not None:
            c.require(child['request_id']==rid and child['payload_sha256']==c.digest(payload) and
                parent['request_raw_sha256']==child['request_raw_sha256'] and
                type(parent['request_raw_bytes']) is int and 0<parent['request_raw_bytes']<=c.MiB and
                parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False,
                'Actual parent written bytes differ from child raw request')
        binding=_bind_child_wire(child,packets[rid],receipt)
        if receipt['status']=='illegal' or receipt.get('result') is not None and receipt['status']!='ok':
            # The original parent retained this decoded value before rejecting
            # it. Bind it to the actual trusted packet without accepting it.
            observed=[item for item in packets[rid] if item['raw_sha256']==parent['response_raw_sha256']]
            c.require(len(observed)==1,'Exactly one trusted rejected response packet')
            packet=observed[0]['packet']
            c.require(type(packet) is dict and set(packet)=={'type','request_id','result'} and
                packet['type']=='response' and packet['request_id']==rid and
                c.canonical(packet['result'])==c.canonical(receipt['result']),
                'Retained rejected result differs from original trusted raw response')
        validate_profiles(parent,child,receipt,payload,route,worker_pid=init['identity']['pid'])
        if receipt['status']=='worker_error':
            errors=[p['packet'] for p in packets[rid] if type(p['packet']) is dict and p['packet'].get('type')=='error']
            c.require(len(errors)==1 and errors[0].get('request_id')==rid and
                str(errors[0].get('error'))[:4096]==receipt['detail'],'Original worker error/raw packet binding')
        checks.append(dict(request_id=rid,status=receipt['status'],startup=startup.is_file(),
            parent_trusted_packet_observed=parent_observed,raw_packets=len(packets[rid]),**binding,
            failed_packets=[{key:item[key] for key in ('raw_sha256','raw_bytes','decode_error')}
                            for item in packets[rid]] if receipt['status']!='ok' else []))
    expected_startups=1+(1 if folder.name.startswith('qualification_') else 0)
    c.require(startup_count==expected_startups,'Original startup request inventory differs')
    return dict(status='passed',worker=folder.name,requests=len(ids),startup_requests=startup_count,
        observed=True,quota=quota,scientific_mode=mode,stdout_log=verified,exact_bytes_checked=True,
        failed_requests=len(failures),failed_request_id=failed_id,partial_failed_output=partial,requests_checked=checks,
        newline_normalization='terminal CRLF to LF only; JSON bytes never reserialized',
        failed_packets_remain_unaccepted=True,original_trusted_receipts_unchanged=True)


def validate_profiles(parent, child, receipt, payload, route, *, worker_pid=None):
    from . import audit
    startup = 'operation' in payload
    completed = receipt['status'] == 'ok'
    if not startup:
        audit.check_clock(receipt,limit=1.)
    returned_response = completed or receipt['status']=='illegal' or receipt.get('result') is not None
    if not completed and returned_response and child is not None:
        c.require(child['outcome']=='response_flushed' and child.get('error') is None and
            child['result_sha256']==c.digest(receipt['result']),
            'Retained rejected response/child scientific hash binding')
        c.require(startup or len(child['profiles'])==1 and child['profiles'][0]['handle_failed'] is False,
            'Rejected returned response requires completed worker handle')
    if route == 'frozen':
        if startup:
            return boundaries.validate_startup_profiles(parent,child,receipt,payload,64,worker_pid=worker_pid)
        return boundaries.validate_profile_row(dict(mode='profiled',quota=64,player_role='challenger',
            observer=payload['observer'],clock=receipt,completed=completed,parent_profile=parent,child_profile=child))
    c.require(route == 'research', 'Explicit research sidecar route')
    limit = (300. if payload.get('operation')=='entrypoint_checks' else 60.) if startup else 1.
    if startup:
        from tools.splendorust_search_scale_20261024 import audit as original_science
        original_science.check_clock(receipt,limit=limit)
    else:
        audit.check_clock(receipt, limit=limit)
    c.require(startup or receipt['deadline_seconds']==1., 'Sampling full1s clock required')
    c.require(parent is not None, 'Original parent boundary evidence required')
    start,arrival,metrics = boundaries._validate_parent(parent,receipt,completed=returned_response)
    if child is None:
        c.require(not completed, 'Successful research child sidecar absent')
        return dict(status='preserved_failed_request_missing_child',parent=metrics)
    quota = 64
    research_child_identity(child,parent,receipt,payload,quota,completed=returned_response,startup=startup,worker_pid=worker_pid)
    profiles = child['profiles']
    c.require(type(profiles) is list and len(profiles)<=(64 if startup else 1), 'Exact bounded handle inventory')
    begin=end=None
    if startup and completed and payload=={'operation':'entrypoint_checks'}:
        result=receipt['result']
        c.require(result['status']=='research_entrypoint_counterexamples_passed' and
            result['counts']['complete_decisions']==52 and len(profiles)==63, 'Actual research52 suite/63handles')
        previous=child['request_read_complete_ns']
        for index,profile in enumerate(profiles):
            c.require(profile['handle_failed'] is (index>=37), 'Exact research startup success/rejection order')
            _,left,right=boundaries._validate_handle(profile,child,completed=index<37,startup=True)
            c.require(previous<=left<=right<=child['pre_write_ns'], 'Research startup handle order')
            previous=right
    elif startup:
        c.require(payload.get('operation')=='graph_parity' and not profiles,'Graph startup handle inventory')
    elif profiles:
        _,begin,end=boundaries._validate_handle(profiles[0],child,completed=returned_response)
    else:
        c.require(not completed,'Completed ordinary research handle absent')
    result=research_boundaries(child,parent,start,arrival,begin=begin,end=end,startup=startup)
    return dict(status='passed',parent=metrics,child=result,request_deadline_unchanged=True,
                accepted_science=completed,rejected_response_retained=returned_response and not completed)


def verify_placement_request(folder, request_id, freeze, parent_profile):
    proof=placement.verify_worker_placement(folder,freeze,'isolated')
    value=c.read(folder/'placement_requests'/f'{request_id}.json')
    placement.verify_saved_request(value,freeze,'isolated',request_id=request_id,
        player_pid=proof['player_pid'],supervisor_identity=proof['supervisor'],parent_profile=parent_profile)
    launch=c.read(c.OUT/'launch.json')
    c.require((proof['supervisor']['pid'],proof['supervisor']['created'])==(launch['pid'],launch['created']),
              'Worker supervisor differs from owning runner')
    return value


def verify_saved_record(row, freeze):
    c.require(not row.get('profile_capture_error') and not row.get('placement_error'),
              'Observer/placement integrity failure remains incomplete')
    c.require(row['placement_mode']=='isolated' and row['observation_mode']=='profiled', 'Observed isolated route required')
    path=c.ROOT/row['worker_clock_file'];receipt=row['clock']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()) and path.name==receipt['request_id']+'.json' and
        c.read(path)==receipt and path.parent==c.ROOT/row['worker_folder'], 'Original receipt path/content binding')
    spec=row['spec'];payload=dict(player='search',observer=row['observer'])
    c.require(spec.get('route',row['route'])==row['route'],'Saved prospective worker route binding')
    c.require(c.read(path.parent/'inputs'/path.name)==payload, 'Exact lawful request payload')
    for key,expected in (('parent_profile',path.parent/'parent_profiles'/path.name),
                         ('child_profile',path.parent/(receipt['request_id']+'_child_profile.json'))):
        value=row.get(key+'_file')
        if value is None:
            c.require(row.get(key) is None and not row['completed'] and key=='child_profile','Missing parent/success sidecar')
        else:
            c.require((c.ROOT/value).resolve()==expected.resolve() and c.read(expected)==row[key],'Saved sidecar row binding')
    c.require(verify_placement_request(path.parent,receipt['request_id'],freeze,row['parent_profile'])==row['placement_evidence'],
              'Saved placement proof binding')
    c.require(row['completed']==(receipt['status']=='ok'), 'Original completion status')
    if row['completed']:
        c.require(row['action']==receipt['result']['action'] and row['trace']==receipt['result']['diagnostics'], 'Response science fields')
    else:
        c.require(row['action'] is None and row['trace'] is None, 'Failed result not accepted')
    validate_profiles(row['parent_profile'],row['child_profile'],receipt,payload,row['route'])
    return dict(status='passed',request_id=receipt['request_id'],route=row['route'])


def verify_worker_logs(folder,envelope,route):
    cap=(128 if route=='research' else 16)*c.MiB
    folder=Path(folder);path=folder/'cleanup.json'
    c.require(path.stat().st_size<=c.MiB,'Bounded cleanup manifest')
    cleanup=c.read(path);descriptors=cleanup['log_compression']
    names=('stdout.log','stderr.log','events.jsonl')
    c.require(cleanup['status']=='passed' and set(descriptors)==set(names),'Complete cleanup/stream inventory')
    c.require({p.name for p in folder.glob('*.gz')}=={n+'.gz' for n in names},'Exact compressed stream inventory')
    rows=[]
    for name in names:
        envelope.guard();target=folder/(name+'.gz');item=descriptors[name]
        c.require(not(folder/name).exists() and item['status']=='verified' and item['path']==target.name,'Original compressed log identity')
        rows.append(dict(name=name,**verify_log(target,item,max_raw_bytes=cap,max_compressed_bytes=cap*16+128,
            deadline=envelope.effective_deadline(),sink=lambda _:envelope.guard())))
    return dict(status='passed',streams=3,decoded_streams=rows,raw_bytes=sum(r['raw_bytes'] for r in rows),
        compressed_bytes=sum(r['compressed_bytes'] for r in rows),max_raw_bytes_per_stream=cap,cleanup_sha256=c.file_hash(path))


def verify_closed_worker(folder,freeze,envelope):
    from . import adapters
    folder=Path(folder);envelope.close_worker(folder);envelope.guard()
    identity=c.read(folder/'route_identity.json');route=identity['route']
    logs=verify_worker_logs(folder,envelope,route)
    raw=verify_raw_transport(folder,freeze,envelope)
    init=c.read(folder/'initialization.json');metadata=init['result']
    expected_artifacts=freeze['research_native_artifacts'] if route=='research' else freeze['native_artifacts']
    expected_roles=freeze['research_native_roles'] if route=='research' else None
    c.require(identity==dict(route=route,module='tools.splendorust_pool6_admission_20261027_02.worker' if route=='research' else OBSERVER_MODULE,
        native_artifacts=expected_artifacts,native_roles=expected_roles,metadata=metadata), 'Saved READY/role/module identity binding')
    if route=='frozen':
        from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
        copy=deepcopy(metadata);c.require(copy.pop('inference_backend')==BACKEND,'Frozen graph backend changed')
        validate_ready_metadata(copy,expected_sources=SOURCE_HASHES,worker_pid=init['identity']['pid'],
            process_tree=init['telemetry']['process_tree'],expected_quota=64)
        verify_instrumentation_identity(folder,freeze,64,metadata)
    else:adapters.verify_saved_ready(folder,freeze)
    cleanup=c.read(folder/'cleanup.json')
    c.require(not cleanup['alive'] and not cleanup['errors'] and cleanup['deadline_clamped_to_stage_and_global'] is True,
              'Owned worker cleanup errors remain failures')
    for owned in cleanup['confirmed']:
        envelope.guard()
        try:proc=psutil.Process(owned['pid'])
        except psutil.NoSuchProcess:continue
        c.require(proc.create_time()!=owned['created'],'Owned worker identity remains alive after cleanup')
    graph=verify_graph_evidence(folder,freeze,boundary=envelope.guard)
    parents=sorted((folder/'parent_profiles').glob('request-*.json'))
    c.require([p.name for p in parents]==[p.name for p in sorted((folder/'placement_requests').glob('request-*.json'))],
              'Complete per-request placement inventory')
    for path in parents:
        envelope.guard();verify_placement_request(folder,path.stem,freeze,c.read(path))
    envelope.close_worker(folder);envelope.guard()
    return dict(status='passed',worker=folder.name,route=route,logs=logs,raw=raw,graph=graph,
                placement_requests=len(parents),placement_mode='isolated')


from tools.splendorust_boundary_timing_20261025.runner import validate_graph_parity, exact_output_hashes
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
    c.require(c.digest(identity)==freeze['unchanged_graph_backend_identity']['graph_identity_sha256'] and
        identity['weights_before_sha256']==freeze['unchanged_graph_backend_identity']['weights_sha256'],
        'Previously verified graph or weight identity changed')
    route=c.read(folder/'route_identity.json')['route']
    c.require(execution['metadata'].get('route')=='research' if route=='research' else execution['metadata']['search_simulations']==64, 'Graph route identity differs')
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


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def decision(child,state,envelope,role,freeze):
    from . import audit
    c.require(role in audit.ROLES and child.route==audit.MODES[role], 'Exact live role/route')
    before,observer=rules.serialize_snapshot(state),rules.observation_payload(state)
    decode_training_snapshot(before);ledger.check_observer(before,observer)
    row=ordinary(child,dict(route=child.route,quota=64),observer,envelope,freeze)
    row.update(arm='search',player_role=role,quota=64,mode=child.route,seat=state.current_player,
               before=before,after=None)
    if row['completed']:
        try:row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,rules.deserialize_action(row['action'])))
        except BaseException:row['transition_error']=traceback.format_exc()
    return row


def verify_saved_decision(row,freeze):
    from . import audit
    verify_saved_record(row,freeze)
    c.require(row['mode']==row['route']==audit.MODES[row['player_role']] and row['arm']=='search' and
        row['quota']==64 and row['seat']==row['before']['state']['current_player'], 'Live scientific role binding')
    decode_training_snapshot(row['before']);ledger.check_observer(row['before'],row['observer'])
    if row['completed']:
        identity=audit.profile_identity(freeze,64,6) if row['route']=='research' else None
        audit.validate_result(row['clock']['result'],row['observer'],row['route'],identity=identity)
        c.require(ledger.transition(row['before'],row['action'])==row['after'],'Independent saved transition')
        decode_training_snapshot(row['after'])
    else:c.require(row['after'] is None,'Failed move has no invented successor')
    return dict(status='passed',request_id=row['clock']['request_id'])


def captured_fault(spec,row):
    from . import audit
    classification=audit.classify_failure(row['clock'])
    return dict(**{key:spec[key] for key in ('stage','pair_index','game_index')},ply=row['ply'],
        player_role=row['player_role'],seat=row['seat'],clock_status=row['clock']['status'],
        failure_classification=classification,request_id=row['clock']['request_id'],clock=row['clock'],
        worker_clock_file=row['worker_clock_file'],no_retry=True,
        collection_stopped=c.PHASE=='qualification',time=c.now())


def play_game(spec,envelope,freeze):
    from . import schedule as schedules, audit
    schedules.validate_spec(spec)
    state=rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state()
    initial=rules.serialize_snapshot(state);decode_training_snapshot(initial)
    folder=game_folder(spec);c.save(folder/'initial.json',dict(**spec,initial=initial))
    workers={};rows=[];fault=None;outcome=None;independent=0.;begun=time.perf_counter();original_error=None
    try:
        for role in audit.ROLES:
            workers[role]=make_worker(f"{spec['stage']}_g{spec['game_index']}_{role}",envelope,freeze,route=spec[role+'_mode'])
        for ply in range(2000):
            if state.terminal:break
            seat=state.current_player;role='challenger' if seat==spec['challenger_seat'] else 'opponent'
            row=decision(workers[role],state,envelope,role,freeze)
            row.update(ply=ply,**{key:spec[key] for key in ('stage','pair_index','game_index')})
            c.save(folder/'moves'/f'{ply:04d}.json',row);rows.append(row)
            c.require('transition_error' not in row,'Saved transition failure; original clock retained')
            envelope.guard();_require_profiles(row)
            started=time.perf_counter();verify_saved_decision(row,freeze);independent+=time.perf_counter()-started
            if not row['completed']:
                fault=captured_fault(spec,row)
                outcome=dict(reason='forfeit',winner=1-seat,loser=seat,failed_role=role,failed_seat=seat,
                             failure=fault['failure_classification'])
                c.save(folder/'captured_fault.json',fault)
                if c.PHASE=='qualification':c.save(c.OUT/'collection_stop.json',fault)
                break
            state=decode_training_snapshot(row['after'])
            if not state.terminal and (row['after']==row['before'] or row['action']['action_type']=='PASS' and
                len(rows)>1 and rows[-2]['action'] and rows[-2]['action']['action_type']=='PASS'):
                raise ValueError('Unresolved actual-game deadlock; no official result')
        if outcome is None:
            c.require(state.terminal,'2,000-ply safety cap; incomplete game')
            outcome=dict(reason='natural',winner=state.winner)
        ending=dict(**spec,**outcome,final=rules.serialize_snapshot(state),complete=True,plies=len(rows),
                    collection_seconds=time.perf_counter()-begun)
        c.save(folder/'result.json',ending)
    except BaseException:
        original_error=traceback.format_exc()
        raise
    finally:
        close_errors=[]
        for role,child in workers.items():
            try:
                started=time.perf_counter();proof=close_and_audit(child,envelope,freeze)
                independent+=time.perf_counter()-started
                c.save(folder/f'closed_{role}.json',proof)
            except BaseException:close_errors.append(dict(role=role,error=traceback.format_exc()))
        if close_errors:
            c.save(folder/'closure_failures.json',dict(original_error=original_error,errors=close_errors))
            if original_error is None:raise ValueError('Owned worker closure/integrity failed; preserved errors')
    started=time.perf_counter()
    checked=audit.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
    independent+=time.perf_counter()-started;c.save(folder/'online_replay.json',checked)
    hashing=None
    if spec['stage']=='operational':
        hashing=measure_seal_cost([folder]+[child.output_dir for child in workers.values()],envelope.guard)
        independent+=hashing['seconds'];c.save(folder/'hashing_measurement.json',hashing)
    c.save(folder/'cost.json',dict(independent_seconds=independent,collection_seconds=time.perf_counter()-begun,
        complete_decisions=sum(r['completed'] for r in rows),
        measured_two_pass_hash_seconds=None if hashing is None else hashing['seconds']))
    return ending,rows,independent,fault


def qualification_gate(rows):
    c.require(len(rows)<=90,'Bounded qualification requests')
    complete=len(rows)==90 and all(r['completed'] for r in rows)
    times=[r['clock']['receipt_elapsed_seconds'] for r in rows if r['completed']]
    return dict(passed=complete and all(t<.7 for t in times),complete=complete,
        requests=len(rows),completed=len(times),at_or_above_700ms=sum(t>=.7 for t in times),
        maximum_seconds=max(times,default=None),margin_seconds=.7,margin_is_not_timeout=True,outcomes_used=False)


def qualify(envelope,freeze,plan):
    from . import adapters, qualification
    rows=[];failure=None;workers={};first_results={};original_error=None
    try:
        for route in ('frozen','research'):
            child=make_worker('qualification_'+route,envelope,freeze,route=route);workers[route]=child
            receipt=adapters.actual_entrypoint(child,envelope,freeze,request_fn=request,collect_fn=collect_profiles)
            c.save(c.OUT/'entrypoint_checks'/f'{route}.json',receipt)
        frozen_entrypoint=c.read(c.OUT/'entrypoint_checks/frozen.json')
        for spec in plan['parity_requests']:
            child=workers[spec['route']]
            row=ordinary(child,spec,spec['observer'],envelope,freeze)
            c.save(c.OUT/'qualification_rows'/f"{spec['index']:04d}.json",row);rows.append(row)
            if not row['completed']:
                failure=dict(phase='qualification',index=spec['index'],clock=row['clock'],route=spec['route'],no_retry=True,
                             diagnostic_deadline_failure=deadline_failure(row['clock'],spec['route']))
                c.save(c.OUT/'qualification_stop.json',failure)
            verify_saved_record(row,freeze)
            if failure is not None:break
            qualification.validate_admission_row(row,spec,first_results,boundary=envelope.guard,
                                                  frozen_entrypoint=frozen_entrypoint)
    except BaseException:
        original_error=traceback.format_exc();raise
    finally:
        errors=[]
        for route,child in workers.items():
            try:close_and_audit(child,envelope,freeze)
            except BaseException:errors.append(dict(route=route,error=traceback.format_exc()))
        if errors:
            c.save(c.OUT/'qualification_closure_failures.json',dict(original_error=original_error,errors=errors))
            if original_error is None:raise ValueError('Qualification closure/integrity failure preserved')
    gate=qualification_gate(rows)
    summary=dict(qualified=gate['passed'],requests=len(rows),failure=failure,timing_gate=gate)
    if failure is None:summary['parity']=qualification.validate_parity_rows(rows,plan,boundary=envelope.guard)
    c.save(c.OUT/'qualification_summary.json',summary)
    return summary


def operational_gate(*,observed_seconds,observed_variable_bytes,independent_seconds,
                     fixed_final_seconds,final_evidence_bytes,maximum_move_seconds,all_natural,
                     current_evidence_bytes):
    values=(observed_seconds,observed_variable_bytes,independent_seconds,fixed_final_seconds,
            final_evidence_bytes,maximum_move_seconds,current_evidence_bytes)
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values),'Finite measured cost inputs')
    factor=1.5;final=independent_seconds*factor+fixed_final_seconds
    resource=current_evidence_bytes<c.PREFINAL_EVIDENCE_BYTES and final<c.STAGES['final'] and \
        final_evidence_bytes<c.FINAL_RESERVE_BYTES and current_evidence_bytes+final_evidence_bytes<c.MAX_EVIDENCE_BYTES
    return dict(passed=resource and all_natural and maximum_move_seconds<.7,resource_passed=resource,
        timing_margin_passed=maximum_move_seconds<.7,all_operational_games_natural=all_natural,
        observed_games=6,projected_games=0,conservative_factor=1.5,arithmetic_factor=factor,
        projected_prefinal_bytes=current_evidence_bytes,
        projected_final_seconds=final,projected_final_evidence_bytes=final_evidence_bytes,
        projected_total_evidence_bytes=current_evidence_bytes+final_evidence_bytes,
        final_seconds_cap=c.STAGES['final'],prefinal_bytes_cap=c.PREFINAL_EVIDENCE_BYTES,
        final_bytes_cap=c.FINAL_RESERVE_BYTES,total_bytes_cap=c.MAX_EVIDENCE_BYTES,maximum_move_seconds=maximum_move_seconds,
        margin_seconds=.7,margin_is_not_timeout=True,outcomes_used=False,benchmark_authorized_to_launch=False)


def measured_game_costs(schedule,boundary=lambda:None):
    measured=[]
    for spec in schedule:
        boundary();folder=game_folder(spec);cost=c.read(folder/'cost.json')
        paths=[folder]+[c.OUT/'workers'/f"operational_g{spec['game_index']}_{role}" for role in ('challenger','opponent')]
        size=sum(p.stat().st_size for d in paths for p in d.rglob('*') if p.is_file())
        size+=sum((c.OUT/'closed_workers'/f"operational_g{spec['game_index']}_{role}.json").stat().st_size+
                  (c.OUT/'closed_worker_costs'/f"operational_g{spec['game_index']}_{role}.json").stat().st_size
                  for role in ('challenger','opponent'))
        measured.append(dict(game_index=spec['game_index'],collection_seconds=cost['collection_seconds'],
            independent_seconds=cost['independent_seconds'],bytes=size))
    return measured


def reproduction_fixed_components(envelope):
    folders=[p for p in (c.OUT/'workers').iterdir() if not p.name.startswith('audit_')]
    return dict(initialization=max(c.read(p/'initialization.json')['elapsed_seconds'] for p in folders),
        graph=max(c.read(p/'graph_parity_clock.json')['elapsed_seconds'] for p in folders),
        closure=max(c.read(c.OUT/'closed_worker_costs'/f'{p.name}.json')['seconds'] for p in folders),
        runtime=envelope.source_runtime_seconds)


def admission_inputs(measured,components,final_evidence_bytes,maximum_move_seconds,current_evidence_bytes):
    c.require(len(measured)==6,'Fixed six measured game costs')
    fixed=48*1.5*(components['initialization']+components['graph']+components['closure']+2.)+2*components['runtime']+300.
    return dict(observed_seconds=6*max(m['collection_seconds'] for m in measured),
        observed_variable_bytes=6*max(m['bytes'] for m in measured),
        independent_seconds=6*max(m['independent_seconds'] for m in measured),fixed_final_seconds=fixed,
        current_evidence_bytes=current_evidence_bytes,final_evidence_bytes=final_evidence_bytes,
        maximum_move_seconds=maximum_move_seconds,all_natural=True)


def final_evidence_forecast(schedule,boundary=lambda:None):
    """Measured closure/graph/placement costs, conservatively bounded48 reproductions."""
    c.require(len(schedule)==6,'Six operational measurements')
    game_files=[];worker_files=[];largest_move=0;worker_bounds=[];largest_proof=0;largest_replay=0
    for spec in schedule:
        boundary();folder=game_folder(spec)
        paths=[p for p in folder.rglob('*') if p.is_file()];game_files+=paths
        largest_move=max(largest_move,max(p.stat().st_size for p in (folder/'moves').glob('*.json')))
        largest_replay=max(largest_replay,(folder/'online_replay.json').stat().st_size)
        for role in ('challenger','opponent'):
            worker=c.OUT/'workers'/f"operational_g{spec['game_index']}_{role}"
            paths=[p for p in worker.rglob('*') if p.is_file()];worker_files+=paths
            # An entire measured live tree bounds each one-decision reproduction,
            # including all startup graph files and placement/sidecar records.
            worker_bounds.append(dict(bytes=sum(p.stat().st_size for p in paths),files=len(paths)))
            largest_proof=max(largest_proof,(folder/f'closed_{role}.json').stat().st_size)
    bound=max(x['bytes'] for x in worker_bounds);files=max(x['files'] for x in worker_bounds)
    qualification_files=[p for folder in(c.OUT/'workers').glob('qualification_*') for p in folder.rglob('*') if p.is_file()]
    projected_files=math.ceil(1.5*(len(game_files)+len(worker_files)+len(qualification_files)))+48*files
    item_bytes=max(len(c.canonical(dict(path=p.relative_to(c.ROOT).as_posix(),sha256='0'*64,bytes=2**63-1)).encode())+129
                   for p in game_files+worker_files)
    for path in(c.OUT/'closed_workers').glob('qualification_*.json'):largest_proof=max(largest_proof,path.stat().st_size)
    raw=48*(bound+largest_move)+(62*largest_proof)+(6*largest_replay)+projected_files*item_bytes+32*c.MiB
    return dict(outcomes_used=False,conservative_factor=1.5,maximum_reproductions=48,
        measured_workers=12,entire_game_worker_byte_bound=bound,largest_move_bytes=largest_move,
        largest_closed_proof_bytes=largest_proof,largest_online_replay_bytes=largest_replay,
        projected_sealed_files=projected_files,artifact_item_bytes=item_bytes,
        projected_final_evidence_bytes=math.ceil(1.5*raw),reserve_bytes=c.GiB)


def prefinal_size_inventory(envelope):
    """Bind closed evidence bytes; reserve16MiB for ongoing owner/control logs."""
    excluded={'runner_stdout.log','runner_stderr.log','watchdog_stdout.log','watchdog_stderr.log'}
    items=[]
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file() and path.name not in excluded:
            envelope.guard();items.append(c.artifact(path))
    value=dict(artifacts=items,closed_bytes=sum(x['bytes'] for x in items),dynamic_control_reserve_bytes=16*c.MiB,
               scope='Every existing file except ongoing supervisor log streams')
    descriptor=c.save(c.OUT/'prefinal_size_inventory.json',value)
    return value['closed_bytes']+descriptor['bytes']+value['dynamic_control_reserve_bytes']


def operational(envelope,freeze,schedule,games=None):
    games=[] if games is None else games
    allrows=[];start=time.perf_counter();initial_bytes=bytes_used();independent=0.;stop=None
    for spec in schedule:
        game,rows,cost,stop=play_game(spec,envelope,freeze);games.append(game);allrows.extend(rows);independent+=cost
        print(c.canonical(dict(stage='operational',completed_games=len(games),total=6,fault=stop is not None)),flush=True)
        if stop is not None:break
    if len(games)==6 and stop is None:
        evidence=final_evidence_forecast(schedule,envelope.guard);c.save(c.OUT/'final_evidence_forecast.json',evidence)
        # 48 fresh graph workers, exact reproductions, closures and complete
        # final readback are charged; no future assessment is launched/priced.
        components=reproduction_fixed_components(envelope);measured=measured_game_costs(schedule,envelope.guard)
        current_bytes=prefinal_size_inventory(envelope)
        inputs=admission_inputs(measured,components,evidence['projected_final_evidence_bytes'],
            max(r['clock']['receipt_elapsed_seconds'] for r in allrows),current_bytes)
        c.save(c.OUT/'admission_gate.json',dict(**operational_gate(**inputs),inputs=inputs,
            measurement_basis='six times maximum observed independent cost,1.5margin,48fresh exact reproductions',
            measured_game_costs=measured,actual_operational_elapsed_seconds=time.perf_counter()-start,
            actual_operational_evidence_growth_bytes=bytes_used()-initial_bytes,
            reproduction_fixed_components=components))
    return games,stop


def reproduce_decision(original,index,envelope,freeze):
    c.require(original['completed'],'Failed decisions are never retried')
    child=make_worker(f'audit_{index:02d}',envelope,freeze,route=original['route'])
    try:
        repeated=decision(child,decode_training_snapshot(original['before']),envelope,original['player_role'],freeze)
        c.save(c.OUT/'repeated_decisions'/f'{index:02d}.json',repeated)
        envelope.guard();_require_profiles(repeated);verify_saved_decision(repeated,freeze)
        c.require(repeated['completed'] and all(repeated[k]==original[k] for k in ('action','trace','after')) and
            repeated['clock']['result']==original['clock']['result'],'Full exact scientific reproduction differs')
    finally:close_and_audit(child,envelope,freeze)
    return repeated


def load_inputs(freeze,envelope):
    from . import schedule as schedules
    c.verify([freeze['schedule']]);schedule=c.read(c.ROOT/freeze['schedule']['path'])
    c.require(schedule==schedules.make_schedule(),'Frozen schedule derivation changed')
    plan=None
    if c.PHASE=='qualification':
        from . import qualification
        c.verify([freeze['qualification_plan']]);plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
        c.require(plan==qualification.build_plan(),'Prospective actual entrypoint inventory differs')
    return schedule,plan


def verify_prior_preservation(freeze,envelope):
    """Bind both prior inventories to the just-completed full frozen-input hash pass."""
    source=freeze['prior_accepted_attempt']['preservation'];c.verify([source]);value=c.read(c.ROOT/source['path'])
    pins={x['path']:x for x in freeze['inputs']};receipts=[]
    scopes=[value]+value.get('additional_prior_attempts',[])
    c.require(len(scopes)==2,'Both accepted repeat and original negative inventories required')
    roots=set()
    for scope in scopes:
        envelope.guard();items=scope['prior_attempt_artifacts']
        c.require(len({x['path'] for x in items})==len(items) and all(pins.get(x['path'])==x for x in items),
                  'Preservation files absent from completed source hash pass')
        prior=(c.ROOT/scope['prior_freeze']['path']).parent.resolve()
        c.require(prior not in roots and prior.is_relative_to(c.ROOT.resolve()),'Distinct bounded predecessor roots')
        roots.add(prior)
        paths=set()
        for p in prior.rglob('*'):
            envelope.guard()
            if p.is_file():paths.add(p.relative_to(c.ROOT).as_posix())
        c.require(paths=={x['path'] for x in items},'Prior evidence tree gained or lost files')
        receipts.append(dict(prior_freeze=scope['prior_freeze'],prior_status=scope['prior_status'],
            prior_files=len(items),prior_bytes=sum(x['bytes'] for x in items)))
    result=dict(status='passed',prior_manifest=source,prior_attempts=receipts,
        prior_files=sum(x['prior_files'] for x in receipts),prior_bytes=sum(x['prior_bytes'] for x in receipts),
        hash_pass='completed verify_runtime over all freeze inputs',benchmark_authorized_to_launch=False)
    c.save(c.OUT/'prior_preservation_after.json',result)
    return c.artifact(c.OUT/'prior_preservation_after.json')


def verified_admission_gate(envelope,schedule,records):
    """Reconstruct the six-game resource and margin gate before fresh final inference."""
    gate=c.read(c.OUT/'admission_gate.json')
    measured=measured_game_costs(schedule,envelope.guard);components=reproduction_fixed_components(envelope)
    evidence=final_evidence_forecast(schedule,envelope.guard)
    c.require(evidence==c.read(c.OUT/'final_evidence_forecast.json'),'Independent final evidence forecast')
    inventory=c.read(c.OUT/'prefinal_size_inventory.json')
    for item in inventory['artifacts']:envelope.guard();c.verify([item])
    c.require(inventory['closed_bytes']==sum(x['bytes'] for x in inventory['artifacts']) and
        inventory['dynamic_control_reserve_bytes']==16*c.MiB,'Bounded closed/current evidence census')
    current_bytes=inventory['closed_bytes']+(c.OUT/'prefinal_size_inventory.json').stat().st_size+16*c.MiB
    expected_inputs=admission_inputs(measured,components,evidence['projected_final_evidence_bytes'],
        max(r['clock']['receipt_elapsed_seconds'] for r in records),current_bytes)
    c.require(gate['measured_game_costs']==measured and gate['reproduction_fixed_components']==components and
        gate['inputs']==expected_inputs,'Gate inputs bound to actual per-game measurements and startup/closure costs')
    c.require(all(gate[k]==v for k,v in operational_gate(**gate['inputs']).items()),'Saved operational gate arithmetic')
    return gate


def ensure_reproduction_resources(sample,records,envelope,admission_gate=None):
    """No final learned request starts unless its measured remaining bound fits."""
    count=len(sample['decisions']);c.require(0<=count<=48,'Bounded final sample')
    if count==0:return None
    envelope.guard();components=reproduction_fixed_components(envelope)
    folders=[p for p in (c.OUT/'workers').iterdir() if not p.name.startswith('audit_')]
    trees=[]
    for folder in folders:
        envelope.guard();paths=[p for p in folder.rglob('*') if p.is_file()]
        trees.append(dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths)))
    proof_bytes=max(p.stat().st_size for p in(c.OUT/'closed_workers').glob('*.json'))
    max_row=max(len(c.canonical(row).encode('utf-8')) for row in records)
    existing_files=sum(1 for p in c.OUT.rglob('*') if p.is_file())
    projected_files=existing_files+count*(max(t['files'] for t in trees)+8)
    # 1024 bytes per seal descriptor conservatively bounds the fixed workspace paths.
    item_bound=1024
    for p in c.OUT.rglob('*'):
        if p.is_file():
            envelope.guard()
            c.require(len(c.canonical(dict(path=p.relative_to(c.ROOT).as_posix(),sha256='0'*64,bytes=2**63-1)).encode())+129<item_bound,
                      'Seal descriptor forecast bound')
    final_bytes=math.ceil(1.5*(count*(max(t['bytes'] for t in trees)+max_row)+
        (len(trees)+count)*proof_bytes+projected_files*item_bound+32*c.MiB))
    remaining_wait=max(0.,c.RAM_WAIT_SECONDS-envelope.waited)
    seconds=1.5*(count*(components['initialization']+components['graph']+components['closure']+2.)+
        len(trees)*components['closure'])+2*components['runtime']+300.+remaining_wait
    if admission_gate is not None:
        seconds=max(seconds,admission_gate['projected_final_seconds'])
        final_bytes=max(final_bytes,admission_gate['projected_final_evidence_bytes'])
    current_bytes=bytes_used();remaining=envelope.remaining()
    passed=(admission_gate is None or admission_gate['resource_passed']) and seconds<remaining and         final_bytes<c.FINAL_RESERVE_BYTES and current_bytes+final_bytes<c.MAX_EVIDENCE_BYTES
    result=dict(passed=passed,selected_reproductions=count,closed_workers=len(trees),
        projected_remaining_seconds=seconds,available_seconds=remaining,remaining_ram_wait_seconds=remaining_wait,
        current_evidence_bytes=current_bytes,projected_final_bytes=final_bytes,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,total_evidence_cap=c.MAX_EVIDENCE_BYTES,
        projected_sealed_files=projected_files,artifact_item_bytes=item_bound,components=components,
        source='measured completed worker trees and closures; current workload only',
        engineering_margin_used=False,benchmark_authorized_to_launch=False)
    c.save(c.OUT/'reproduction_resource_gate.json',result)
    c.require(passed,'Final reproduction resources do not fit; no final inference started')
    return result



def final_checks(envelope,freeze,schedule,plan,games,stop,qualified):
    from . import audit, qualification, schedule as schedules
    c.require(schedule==schedules.make_schedule(),'Fixed schedule changed')
    qpaths=sorted((c.OUT/'qualification_rows').glob('*.json'));qrows=[c.read(p) for p in qpaths]
    expected_q=[] if plan is None else plan['parity_requests']
    c.require([r['spec'] for r in qrows]==expected_q[:len(qrows)] and len(qrows)<=len(expected_q),'Exact qualification prefix')
    c.require([p.name for p in qpaths]==[f"{s['index']:04d}.json" for s in expected_q[:len(qrows)]],'Qualification file inventory')
    if qualified is not None:
        c.require(qualified==c.read(c.OUT/'qualification_summary.json') and qualified['requests']==len(qrows) and
            qualified['timing_gate']==qualification_gate(qrows) and
            qualified['qualified']==qualified['timing_gate']['passed'],'Stored qualification summary and admission status')
    if qrows:
        c.require(all(r['completed'] for r in qrows[:-1]),'No work after failed qualification request')
        for row in qrows:
            envelope.guard();verify_saved_record(row,freeze)
            if row['completed']:
                identity=audit.profile_identity(freeze,64,6,boundary=envelope.guard) if row['route']=='research' else None
                audit.validate_result(row['clock']['result'],row['observer'],row['route'],identity=identity)
        parity=qualification.validate_parity_rows(qrows,plan,boundary=envelope.guard,allow_prefix=True)
        if len(qrows)==90 and qrows[-1]['completed']:
            c.require(qualified and qualified['parity']==parity and qualified['timing_gate']==qualification_gate(qrows),
                      'Independent qualification summary/gate differs')
        else:
            failure=c.read(c.OUT/'qualification_stop.json')
            c.require(not qrows[-1]['completed'] and failure['clock']==qrows[-1]['clock'] and
                failure['index']==qrows[-1]['spec']['index'] and qualified['failure']==failure,'Final failed qualification binding')
    if games and c.PHASE=='qualification':c.require(qualified and qualified['qualified'],'Play forbidden after negative qualification')
    coverage=audit.validate_stopped_prefix(schedule,games,stop)
    records=[];faults=[];replays=[]
    for spec,game in zip(schedule,games,strict=False):
        envelope.guard();folder=game_folder(spec);initial=c.read(folder/'initial.json')
        expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(initial==dict(**spec,initial=expected) and c.read(folder/'result.json')==game,'Initial seed/ending binding')
        paths=sorted((folder/'moves').glob('*.json'))
        c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(game['plies'])],'All actual move files')
        rows=[c.read(p) for p in paths]
        for row in rows:envelope.guard();verify_saved_decision(row,freeze)
        checked=audit.audit_game(expected,rows,game,spec,freeze=freeze,boundary=envelope.guard)
        c.require(checked==c.read(folder/'online_replay.json'),'Independent replay receipt differs')
        replays.append(checked);records.extend(rows)
        if game['reason']=='forfeit':
            saved=c.read(folder/'captured_fault.json');again=captured_fault(spec,rows[-1])
            c.require({k:v for k,v in saved.items() if k!='time'}=={k:v for k,v in again.items() if k!='time'},'Exact original fault attribution')
            faults.append(saved)
        else:c.require(not(folder/'captured_fault.json').exists(),'Unreported game fault')
    actual={p.parent.resolve() for p in (c.OUT/'games').rglob('initial.json')}
    c.require(actual=={game_folder(s).resolve() for s in schedule[:len(games)]},'No hidden additional or partial game')
    if stop is not None:c.require(faults and faults[-1]==stop==c.read(c.OUT/'collection_stop.json'),'Exact ops stop')
    else:c.require(not(c.OUT/'collection_stop.json').exists(),'Unreported collection stop')
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=len(games),receipts=replays))
    c.save(c.OUT/'all_player_faults.json',dict(faults=faults,count=len(faults),failed_move_retries=0))
    c.save(c.OUT/'performance.json',audit.performance(records,games))
    c.save(c.OUT/'boundary_analysis.json',audit.summarize_boundary_rows(records,boundary=envelope.guard))
    sample=audit.select_replay_sample(records,allow_operational=c.PHASE=='qualification')
    c.save(c.OUT/'replay_sample_manifest.json',sample);repeated=[]
    gate=verified_admission_gate(envelope,schedule,records) if len(games)==6 and not faults else None
    ensure_reproduction_resources(sample,records,envelope,admission_gate=gate)
    for index,item in enumerate(sample['decisions']):
        original=next(r for r in records if all(r[k]==item[k] for k in ('stage','pair_index','game_index','ply','player_role')))
        repeated.append(reproduce_decision(original,index,envelope,freeze))
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(repeated),identities=sample['decisions']))
    worker_names={p.name for p in (c.OUT/'workers').iterdir()};expected_workers=set()
    if c.PHASE=='qualification':
        expected_workers={f'qualification_{r}' for r in ('frozen','research') if (c.OUT/'entrypoint_checks'/f'{r}.json').exists()}
    expected_workers|={f"{s['stage']}_g{s['game_index']}_{role}" for s in schedule[:len(games)] for role in audit.ROLES}
    expected_workers|={f'audit_{i:02d}' for i in range(len(repeated))}
    c.require(worker_names==expected_workers and len(worker_names)<=c.MAX_WORKERS,'Exact complete worker inventory')
    represented={name:[] for name in worker_names}
    for row in qrows+records+repeated:represented[Path(row['worker_folder']).name].append(row['clock']['request_id'])
    proofs=[]
    for name in sorted(worker_names):
        folder=c.OUT/'workers'/name
        c.require(len(represented[name])==len(set(represented[name])) and set(represented[name])==
            {p.stem for p in (folder/'inputs').glob('*.json')},'Exact represented ordinary-request inventory')
        proof=verify_closed_worker(folder,freeze,envelope);proofs.append(proof)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json'),'Closed raw/graph/log proof changed')
        if name.startswith('qualification_'):
            receipt=c.read(c.OUT/'entrypoint_checks'/f"{name.removeprefix('qualification_')}.json")
            c.require(c.read(folder/(receipt['request_id']+'.json'))==receipt,'Actual entrypoint trusted receipt')
            qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=envelope.guard)
    c.require(len(proofs)<=c.MAX_GRAPH_BUILDS,'Every worker has its own11-case graph proof')
    c.save(c.OUT/'all_closed_workers.json',dict(status='passed',workers=len(proofs),proofs=proofs))
    c.save(c.OUT/'all_placement_integrity.json',verify_exchanges(envelope,freeze))
    admission=bool(gate and gate['passed'])
    coverage.update(phase=c.PHASE,admission_passed=admission if c.PHASE=='qualification' else None,
                    qualification_ordinary_requests=len(qrows),live_decisions=len(records),reproductions=len(repeated))
    c.save(c.OUT/'coverage.json',coverage)
    seal=[]
    for sub in ('workers','games','qualification_rows','entrypoint_checks','repeated_decisions','closed_workers',
                'closed_worker_costs','placement_requests','placement_acks'):
        for path in sorted((c.OUT/sub).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    for name in ('qualification_summary.json','qualification_stop.json','collection_stop.json','admission_gate.json',
                 'prefinal_size_inventory.json','reproduction_resource_gate.json',
                 'final_evidence_forecast.json','coverage.json','all_game_replay.json','all_player_faults.json','performance.json',
                 'boundary_analysis.json','replay_sample_manifest.json','repeated_decision_summary.json','all_closed_workers.json',
                 'all_placement_integrity.json'):
        if (c.OUT/name).exists():envelope.guard();seal.append(c.artifact(c.OUT/name))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope)
    preservation=verify_prior_preservation(freeze,envelope)
    c.save(c.OUT/'verification_summary.json',dict(status='passed',coverage=coverage,workers=len(proofs),
        source_inputs=len(freeze['inputs']),parent_requests=sum(p['raw']['requests'] for p in proofs),
        graph_cases=11*len(proofs),peak_combined_rss_bytes=envelope.peak_combined_rss,prior_preservation=preservation,
        all_raw_placement_logs_graph_science_sources_verified=True,training=False,benchmark_authorized_to_launch=False))
    return coverage


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Execution freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze changed')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json')
    envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha,phase=c.PHASE))
    status,failure,coverage,qualified='incomplete',None,None,None
    games=[];stop=None;ready=False
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        begun=time.perf_counter();verify_runtime(freeze,envelope);envelope.source_runtime_seconds=time.perf_counter()-begun
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        schedule,plan=load_inputs(freeze,envelope);ready=True
        qualified=qualify(envelope,freeze,plan);envelope.finish()
        if qualified['qualified']:
            envelope.begin('operational');games,stop=operational(envelope,freeze,schedule,games);envelope.finish()
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure,preserved_partial_evidence=True))
    if ready and not envelope.resident:
        try:
            envelope.begin('final');coverage=final_checks(envelope,freeze,schedule,plan,games,stop,qualified);envelope.finish()
            if qualified and qualified['failure'] and not qualified['failure']['diagnostic_deadline_failure']:
                failure=failure or 'Unclassified qualification failure; verified evidence is not scientific acceptance.'
            if failure is None:
                status='qualified_verified' if coverage['admission_passed'] else 'not_admitted_verified'
        except BaseException:
            final_error=traceback.format_exc();c.save(c.OUT/'finalization_failure.json',dict(time=c.now(),error=final_error))
            failure=(failure or '')+final_error
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,coverage=coverage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,phase=c.PHASE,automatic_retries=0,larger_data_deferred=True,training=False,
        requires_all_three_clean_actual_os_exits=True,benchmark_authorized_to_launch=False))
    return 1 if status=='incomplete' else 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run'])
    parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    return run(args.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())


