"""Fixed saved-position repeats with unchanged live players and request clocks."""
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

class MeasurementFailure(ValueError):
    """Preserved invalid observer measurement; never eligible for inference continuation."""

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
        stable = begun if available >= 4*c.GiB else None
        minimum = available
        fast = self.admitted and stable is not None
        while not fast:
            self.guard()
            current, available = time.monotonic(), psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 4*c.GiB:
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
                stable_seconds=60,required_headroom_bytes=4*c.GiB))
        return dict(admission_number=self.admissions,monotonic_ns=time.perf_counter_ns(),
            current_available_bytes=available,required_headroom_bytes=4*c.GiB,wait_seconds=elapsed,
            cumulative_wait_seconds=self.waited,stable_recovery_required=not fast,
            stable_seconds=60 if not fast else None)

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
    from . import host_observer
    c.require(host_observer.runtime_identity()==freeze['host_observer_runtime'],'Host observer runtime changed')
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
        minimum_available_ram_bytes=2*c.GiB,initial_ram_stability_seconds=60,diagnostic_available_ram_bytes=c.DIAGNOSTIC_AVAILABLE_RAM_BYTES,host_observer_max_bytes=c.HOST_OBSERVER_MAX_BYTES,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,seconds_per_move=1.,internal_seconds=.9,
        research_seconds_per_move=1.,research_internal_seconds=.9,research_log_max_bytes=128*c.MiB,
        frozen_log_max_bytes=16*c.MiB,research_response_max_bytes=4*c.MiB,frozen_response_max_bytes=256*1024,
        request_max_bytes=c.MiB,placement_mode='isolated',max_workers=c.MAX_WORKERS,max_graph_builds=c.MAX_GRAPH_BUILDS,
        phase=c.PHASE,qualification_decisions=104,total_requests=216,scientific_decisions=320,
        parent_requests=220,graph_cases=22,overhead_requests=72,diagnostic_requests=144,
        operational_games=0,main_games=0,maximum_reproductions=216,new_games=0,
        automatic_retries=0,failed_decision_retries=0,training=False)
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
    module = 'tools.splendorust_pool6_hostdiag_20261027_03.worker' if research else OBSERVER_MODULE
    artifacts = freeze['research_native_artifacts'] if research else freeze['native_artifacts']
    roles = None
    if research:
        mapping = freeze['research_native_roles']
        c.require(set(mapping) == {a['path'] for a in artifacts} and sorted(mapping.values()) == ['reference','search'],
                  'Authenticated two-native research role mapping')
        roles = {(c.ROOT/path).resolve(): role for path, role in mapping.items()}
    env = {'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'STUDY_WORKER_DIR': str(folder),
           'STUDY_SEARCH_SIMULATIONS': '64', 'STUDY_BOUNDARY_PROFILE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1', 'SPLENDOR_POOL6_HOSTDIAG03_PHASE':c.PHASE}
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
        c.save(folder/'resource_deadline_admission.json',dict(stage=envelope.stage,absolute_deadline=child.absolute_deadline,
            stage_deadline=envelope.deadline,global_deadline=envelope.end,request_counter=child._counter,
            player=child.identity,ordinary_deadline_seconds=1.,internal_seconds=.9))
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
    ram_admission=envelope.admit()
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
    binding=dict(stage=spec['stage'],sequence=spec['sequence'],route=child.route,request_id=request_id,active_player=child.identity)
    token=envelope.host_observer.start_window(spec['sequence'],spec['observer_enabled'],binding=binding,
        timeout_seconds=min(5.,envelope.remaining()))
    if token['status']!='ready':
        failed=envelope.host_observer.finish_window(token,timeout_seconds=max(0.,min(5.,envelope.remaining())))
        c.save(c.OUT/'host_observer_start_failures'/f"{spec['sequence']:04d}.json",failed)
        raise ValueError('Host observer start failed; no player request dispatched')
    c.require(envelope.remaining()>=1.,'Full original allowance after observer acknowledgement')
    receipt=request(child,payload,allowance,validate=validate)
    host_receipt=None;host_error=None
    try:host_receipt=envelope.host_observer.finish_window(token,timeout_seconds=max(0.,min(5.,envelope.remaining())))
    except BaseException:host_error=traceback.format_exc()
    row=dict(spec=deepcopy(spec),observer=observer,completed=receipt['status']=='ok',clock=receipt,
        route=child.route,worker_folder=child.output_dir.relative_to(c.ROOT).as_posix(),
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix(),
        observation_mode='profiled',action=None,trace=None)
    row.update(collect_profiles(child,receipt,envelope))
    row['ram_admission']=ram_admission
    row['host_observation']=host_receipt
    if host_error is not None:row['host_observer_error']=host_error
    if host_receipt is not None:
        try:row['host_observation_file']=c.save(c.OUT/'host_observations'/f"{spec['sequence']:04d}.json",host_receipt)
        except BaseException:row['host_observer_error']=traceback.format_exc()
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
    c.require(identity==dict(route=route,module='tools.splendorust_pool6_hostdiag_20261027_03.worker' if route=='research' else OBSERVER_MODULE,
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


def load_inputs(freeze,envelope):
    from . import design, qualification
    for key in ('panel','request_matrix','qualification_plan'):
        envelope.guard();c.verify([freeze[key]])
    panel=c.read(c.ROOT/freeze['panel']['path']);matrix=c.read(c.ROOT/freeze['request_matrix']['path'])
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(panel==design.build_panel() and matrix==design.build_matrix(panel),'Frozen fixed saved panel/schedule changed')
    c.require(plan==qualification.build_plan(),'Actual104-decision entrypoint inventory changed')
    return panel,matrix,plan


def qualify(envelope,freeze,workers):
    from . import adapters
    receipts={}
    for route in ('frozen','research'):
        child=make_worker('qualification_'+route,envelope,freeze,route=route);workers[route]=child
        receipt=adapters.actual_entrypoint(child,envelope,freeze,request_fn=request,collect_fn=collect_profiles)
        c.save(c.OUT/'entrypoint_checks'/f'{route}.json',receipt);receipts[route]=receipt
    summary=dict(status='passed',workers=2,complete_scientific_decisions=104,graph_cases=22,
        ordinary_requests=0,seven_hundred_ms_gate=False)
    c.save(c.OUT/'qualification_summary.json',summary)
    return summary


def bind_collection_deadline(envelope,workers,rows):
    """Change only stage resource supervision outside every original move clock."""
    c.require(envelope.stage in ('overhead','diagnostic') and set(workers)=={'frozen','research'},'Two qualified retained workers')
    c.require(len(envelope.resident)==2,'Two resident identities before collection')
    bindings=[];stamp=time.perf_counter_ns()
    for route in ('frozen','research'):
        child=workers[route];counter=2+sum(r['route']==route for r in rows)
        c.require(not child._lock.locked() and child._counter==counter and not child._closed,'No pending/failed request at stage boundary')
        previous=child.absolute_deadline;current=envelope.effective_deadline()
        c.require(current<=envelope.end and current<=envelope.start+c.COLLECTION_CUTOFF_SECONDS,'No global resource-clock extension')
        child.absolute_deadline=current
        bindings.append(dict(route=route,worker=child.output_dir.name,player=child.identity,request_counter=child._counter,
            previous_absolute_deadline=previous,collection_absolute_deadline=current))
    value=dict(status='passed',bindings=bindings,stage=envelope.stage,transition_ns=stamp,
        stage_deadline=envelope.deadline,global_deadline=envelope.end,
        ordinary_deadline_seconds=1.,internal_seconds=.9,request_clocks_reset=False)
    c.save(c.OUT/'resident_stage_transitions'/f'{envelope.stage}.json',value)
    return value


def overhead_gate(rows):
    c.require(len(rows)<=72 and all(r['spec']['stage']=='overhead' for r in rows),'Bounded overhead inventory')
    measurements_valid=all(not r.get('host_observer_error') and r.get('host_observation',{}).get('status') in ('observed','disabled')
        and not r.get('host_observation',{}).get('errors') for r in rows)
    complete=len(rows)==72 and all(r['completed'] for r in rows) and measurements_valid;groups={}
    def p95(values):return sorted(values)[math.ceil(.95*len(values))-1]
    for route in ('frozen','research'):
        role=[r for r in rows if r['route']==route]
        off=[r['clock']['receipt_elapsed_seconds'] for r in role if r['completed'] and not r['spec']['observer_enabled']]
        on=[r['clock']['receipt_elapsed_seconds'] for r in role if r['completed'] and r['spec']['observer_enabled']]
        ready=len(role)==36 and len(off)==len(on)==18 and all(0<=x<=1. for x in off+on)
        off_mean=sum(off)/len(off) if off else None;on_mean=sum(on)/len(on) if on else None
        passed=ready and on_mean<=1.05*off_mean and on_mean-off_mean<=.010 and p95(on)<=p95(off)+.025
        groups[route]=dict(passed=bool(passed),requests=len(role),off_count=len(off),on_count=len(on),
            off_mean_seconds=off_mean,on_mean_seconds=on_mean,off_p95_seconds=p95(off) if off else None,
            on_p95_seconds=p95(on) if on else None,all_complete_under_one_second=ready)
    return dict(passed=complete and all(g['passed'] for g in groups.values()),complete=complete,
        attempted=len(rows),completed=sum(r['completed'] for r in rows),measurements_valid=measurements_valid,per_player=groups,
        maximum_mean_ratio=1.05,maximum_mean_difference_seconds=.010,maximum_p95_difference_seconds=.025,
        p95='nearest_rank',trimming=False,automatic_retries=0,margin_700ms='descriptive_only')


def host_targets(workers):
    targets=[]
    for route in ('frozen','research'):
        child=workers[route];init=c.read(child.output_dir/'initialization.json')
        tree=init['telemetry']['process_tree']
        compute=[p for p in tree if p.get('role') in ('python','search','reference')]
        c.require({p['role'] for p in compute}=={'python','search','reference'} and len(compute)==3,'Exact observer target roles')
        targets.extend(dict(role=route+':'+p['role'],pid=p['pid'],created=p['created']) for p in compute)
    c.require(len({(t['pid'],t['created']) for t in targets})==6,'Distinct unchanged compute identities')
    return targets


def start_host_observer(workers,envelope,freeze):
    from . import host_observer
    targets=host_targets(workers)
    observer=host_observer.HostObserver(targets,interval_seconds=.1,max_buffer_bytes=c.HOST_OBSERVER_MAX_BYTES)
    envelope.host_observer=observer
    c.save(c.OUT/'host_observer_identity.json',dict(targets=targets,runner=dict(pid=os.getpid(),created=psutil.Process().create_time()),
        runtime=host_observer.runtime_identity(),contract=freeze['host_observer_contract']))
    return observer


def verify_host_row(row,targets,*,forensic=False):
    from . import host_observer
    c.require(not row.get('host_observer_error') and row.get('host_observation') is not None,'Host observer evidence error')
    item=row['host_observation_file'];c.verify([item]);value=c.read(c.ROOT/item['path'])
    c.require(value==row['host_observation'] and (c.ROOT/item['path']).resolve()==
        (c.OUT/'host_observations'/f"{row['spec']['sequence']:04d}.json").resolve(),'Saved host receipt binding')
    parent=row['parent_profile'];ram=row['ram_admission']
    c.require(ram['required_headroom_bytes']==4*c.GiB and ram['current_available_bytes']>=4*c.GiB and
        ram['monotonic_ns']<=parent['request_start_ns'] and ram['cumulative_wait_seconds']<c.RAM_WAIT_SECONDS and
        (not ram['stable_recovery_required'] or ram['stable_seconds']==60),'Per-request headroom evidence')
    owner=c.read(c.OUT/'host_observer_identity.json')['runner']
    c.require(value.get('placement_start') is not None,'Dispatched request requires recorded initial observer placement')
    for key in ('placement_start','placement_end'):
        if value.get(key) is not None:
            c.require(dict(pid=value[key]['pid'],created=value[key]['created'])==owner,'Observed sampler belongs to owning runner')
    active=next(t for t in targets if t['role']==row['route']+':python')
    expected=dict(stage=row['spec']['stage'],sequence=row['spec']['sequence'],route=row['route'],request_id=row['clock']['request_id'],
        active_player=dict(pid=active['pid'],created=active['created']))
    c.require(value['binding']==expected,'Host window original request identity')
    validator=host_observer.verify_forensic_receipt if forensic else host_observer.validate_receipt
    checked=validator(value,expected_sequence=row['spec']['sequence'],expected_enabled=row['spec']['observer_enabled'],
        process_identities=targets,clock_bounds=dict(request_start_ns=parent['request_start_ns'],receipt_arrival_ns=parent['receipt_arrival_ns']))
    accepted=checked.get('measurement_accepted',checked.get('accepted'))
    c.require(type(accepted) is bool,'Explicit observer measurement disposition')
    if not forensic and not accepted:raise MeasurementFailure('Host observer failed; original response retained')
    return {**checked,'measurement_accepted':accepted}


def verify_measurement_prefix(rows,checks):
    """Authenticate one terminal measurement failure without qualifying its window."""
    c.require(len(rows)==len(checks) and all(type(p.get('measurement_accepted')) is bool for p in checks),
              'Every saved observer window requires an explicit disposition')
    invalid=[i for i,p in enumerate(checks) if not p['measurement_accepted']]
    path=c.OUT/'measurement_stop.json'
    if not invalid:
        c.require(not path.exists(),'Unrepresented observer failure stop')
        return []
    c.require(invalid==[len(rows)-1],'Only one terminal observer failure may be preserved')
    row=rows[-1];proof=checks[-1];value=row['host_observation']
    c.require(value['status']=='observer_failed' and value['errors'],'Terminal measurement failure must retain its cause')
    expected=dict(sequence=row['spec']['sequence'],stage=row['spec']['stage'],route=row['route'],
        clock=row['clock'],worker_clock_file=row['worker_clock_file'],host_observation=row['host_observation_file'],
        measurement_accepted=False,overhead_qualified=False,no_retry=True)
    c.require(path.exists() and c.read(path)==expected,'Exact preserved observer failure stop')
    return [dict(sequence=row['spec']['sequence'],stage=row['spec']['stage'],route=row['route'],
        case_id=row['spec']['case_id'],original_player_clock_status=row['clock']['status'],
        measurement_accepted=False,original_errors=value['errors'],missing_fields=proof['missing_fields'])]


def collect(envelope,freeze,matrix,workers,rows,first_results):
    from . import audit, qualification
    targets=host_targets(workers)
    for spec in (s for s in matrix['requests'] if s['stage']==envelope.stage):
        child=workers[spec['route']]
        row=ordinary(child,spec,spec['observer'],envelope,freeze)
        row.update(before=spec['before_snapshot'],after=None,mode=spec['route'],quota=64,arm='search',
                   player_role='challenger' if spec['route']=='research' else 'opponent')
        if row['completed']:
            try:row['after']=ledger.transition(row['before'],row['action'])
            except BaseException:row['transition_error']=traceback.format_exc()
        c.save(c.OUT/'records'/f"{spec['sequence']:04d}.json",row);rows.append(row)
        if not row['completed']:
            stop=dict(sequence=spec['sequence'],case_id=spec['case_id'],route=spec['route'],clock=row['clock'],
                no_retry=True,diagnostic_deadline_failure=deadline_failure(row['clock'],spec['route']))
            c.save(c.OUT/'collection_stop.json',stop)
        c.require('transition_error' not in row,'Saved transition verification failure; original clock preserved')
        envelope.guard();verify_saved_record(row,freeze)
        try:verify_host_row(row,targets)
        except MeasurementFailure:
            c.save(c.OUT/'measurement_stop.json',dict(sequence=spec['sequence'],stage=spec['stage'],route=spec['route'],
                clock=row['clock'],worker_clock_file=row['worker_clock_file'],host_observation=row['host_observation_file'],
                measurement_accepted=False,overhead_qualified=False,no_retry=True))
            raise
        audit.validate_record(row,spec,freeze,boundary=envelope.guard)
        if not row['completed']:return
        qualification.validate_repeat_row(row,spec,first_results,boundary=envelope.guard)
        print(c.canonical(dict(stage=envelope.stage,requests=len(rows),total=216,round=spec['round'],
            at_or_above_700ms=row['clock']['receipt_elapsed_seconds']>=.7)),flush=True)


def collect_stages(envelope,freeze,matrix,workers,rows,first_results):
    """The full diagnostic is conditional solely on the prospectively fixed overhead gate."""
    c.require(envelope.stage=='overhead','Overhead stage must precede diagnostic')
    collect(envelope,freeze,matrix,workers,rows,first_results)
    gate=overhead_gate(rows);c.save(c.OUT/'overhead_gate.json',gate);envelope.finish()
    if gate['passed']:
        envelope.begin('diagnostic');bind_collection_deadline(envelope,workers,rows)
        collect(envelope,freeze,matrix,workers,rows,first_results)
    return gate


def verify_stage_transitions(envelope,workers,rows):
    receipts=[];previous={};earlier=[]
    for stage in ('overhead','diagnostic'):
        path=c.OUT/'resident_stage_transitions'/f'{stage}.json'
        stage_rows=[r for r in rows if r['spec']['stage']==stage]
        if not path.exists():
            c.require(not stage_rows,'No requests before resource stage binding');continue
        value=c.read(path);start=c.read(c.OUT/'stages'/f'{stage}_start.json')
        c.require(value['stage']==stage and value['stage_deadline']==start['monotonic_deadline'] and
            value['global_deadline']==envelope.end and value['ordinary_deadline_seconds']==1. and
            value['internal_seconds']==.9 and value['request_clocks_reset'] is False,'Resource stage contract')
        c.require([b['route'] for b in value['bindings']]==['frozen','research'],'Two explicit stage bindings')
        for item in value['bindings']:
            folder=c.OUT/'workers'/item['worker'];before=c.read(folder/'resource_deadline_admission.json');init=c.read(folder/'initialization.json')
            c.require(item['worker']=='qualification_'+item['route'] and item['player']==init['identity']==before['player'] and
                item['request_counter']==2+sum(r['route']==item['route'] for r in earlier) and
                item['previous_absolute_deadline']==previous.get(item['route'],before['absolute_deadline']) and
                before['stage']=='qualification' and item['collection_absolute_deadline']<=start['monotonic_deadline'] and
                item['collection_absolute_deadline']<=envelope.start+c.COLLECTION_CUTOFF_SECONDS,'Stage identity/resource deadline')
            previous[item['route']]=item['collection_absolute_deadline']
        for row in stage_rows:c.require(row['parent_profile']['request_start_ns']>=value['transition_ns'],'Request predates stage binding')
        receipts.append(value);earlier+=stage_rows
    return receipts


def verify_prior_preservation(freeze,envelope):
    source=freeze['prior_accepted_attempt']['preservation'];c.verify([source]);value=c.read(c.ROOT/source['path'])
    pins={x['path']:x for x in freeze['inputs']};items=value['prior_attempt_artifacts']
    c.require(all(pins.get(x['path'])==x for x in items),'Prior artifacts absent from completed source hash pass')
    prior=(c.ROOT/value['prior_freeze']['path']).parent
    paths={p.relative_to(c.ROOT).as_posix() for p in prior.rglob('*') if p.is_file()}
    c.require(paths=={x['path'] for x in items},'Prior incomplete attempt gained or lost files')
    result=dict(status='passed',prior_manifest=source,prior_files=len(items),prior_bytes=sum(x['bytes'] for x in items),
        hash_pass='completed verify_runtime over every frozen input',original_incomplete_preserved=True,prior_attempt_accepted=False)
    c.save(c.OUT/'prior_preservation_after.json',result)
    return c.artifact(c.OUT/'prior_preservation_after.json')


def final_checks(envelope,freeze,panel,matrix,plan,qualified):
    from . import audit, qualification, analysis, design
    c.require(panel==design.build_panel() and matrix==design.build_matrix(panel),'Frozen panel/schedule derivation')
    paths=sorted((c.OUT/'records').glob('*.json'));rows=[c.read(p) for p in paths]
    c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(len(rows))],'Exact attempted record inventory')
    for row in rows:envelope.guard();verify_saved_record(row,freeze)
    coverage=audit.audit_prefix(rows,matrix,freeze,boundary=envelope.guard)
    parity=qualification.validate_repeat_rows(rows,plan,allow_prefix=True,boundary=envelope.guard)
    if rows:c.require(qualified==c.read(c.OUT/'qualification_summary.json') and qualified['status']=='passed',
                      'No ordinary work before both actual suites passed')
    stop=c.read(c.OUT/'collection_stop.json') if(c.OUT/'collection_stop.json').exists() else None
    if rows and not rows[-1]['completed']:
        c.require(stop is not None and stop['sequence']==len(rows)-1 and stop['clock']==rows[-1]['clock'] and
            stop['route']==rows[-1]['route'] and stop['no_retry'] is True and
            stop['diagnostic_deadline_failure']==deadline_failure(stop['clock'],stop['route']),
            'Exact first preserved failure receipt and classification')
    else:c.require(stop is None,'Unreported or invented diagnostic stop')
    names={p.name for p in (c.OUT/'workers').iterdir()}
    c.require(names<={'qualification_frozen','qualification_research'},'No extra or restarted workers')
    if qualified:c.require(names=={'qualification_frozen','qualification_research'},'Two qualified retained trees')
    represented={name:[] for name in names}
    for row in rows:represented[Path(row['worker_folder']).name].append(row['clock']['request_id'])
    proofs=[]
    for name in sorted(names):
        folder=c.OUT/'workers'/name
        c.require(len(represented[name])==len(set(represented[name])) and set(represented[name])==
            {p.stem for p in (folder/'inputs').glob('*.json')},'All attempted original requests represented exactly once')
        proof=verify_closed_worker(folder,freeze,envelope);proofs.append(proof)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json'),'Closed raw/graph/placement/log proof changed')
        path=c.OUT/'entrypoint_checks'/f"{name.removeprefix('qualification_')}.json"
        receipt=c.read(path)
        c.require(receipt==c.read(folder/(receipt['request_id']+'.json')),'Original entrypoint clock binding')
        qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=envelope.guard)
    transition=verify_stage_transitions(envelope,names,rows)
    hostproof=None;hostchecks=[]
    identity_path=c.OUT/'host_observer_identity.json'
    if identity_path.exists():
        from . import host_observer
        from types import SimpleNamespace
        targets=host_targets({route:SimpleNamespace(output_dir=c.OUT/'workers'/('qualification_'+route)) for route in ('frozen','research')})
        identity=c.read(identity_path);launch=c.read(c.OUT/'launch.json')
        c.require(identity['targets']==targets and identity['runtime']==freeze['host_observer_runtime'] and
            identity['contract']==freeze['host_observer_contract'] and
            identity['runner']==dict(pid=launch['pid'],created=launch['created']),'Observer targets/runtime/owner binding')
        for row in rows:
            envelope.guard();hostchecks.append(verify_host_row(row,targets,forensic=True))
        hostproof=host_observer.validate_close(c.read(c.OUT/'host_observer_close.json'),process_identities=targets,expected_windows=len(rows))
        close=c.read(c.OUT/'host_observer_close.json')
        c.require(dict(pid=close['placement']['pid'],created=close['placement']['created'])==identity['runner'] and
            all(r['host_observation']['placement_start']==close['placement'] for r in rows),'One owned sampler thread throughout')
        c.require({p.name for p in(c.OUT/'host_observations').glob('*.json')}=={f"{r['spec']['sequence']:04d}.json" for r in rows},
                  'Every attempted host window represented exactly once')
    else:c.require(not rows,'No observed request without observer ownership')
    invalid=verify_measurement_prefix(rows,hostchecks)
    if hostproof is not None:
        prior_error=hostproof['prior_observer_error']
        c.require(prior_error in rows[-1]['host_observation']['errors'] if invalid else prior_error is None,
                  'Observer cleanup prior error must match the preserved terminal measurement failure')
    c.save(c.OUT/'host_observer_forensic_audit.json',dict(status='preserved_failed_prefix' if invalid else 'passed',
        attempted=len(rows),valid_windows=len(rows)-len(invalid),invalid_windows=invalid,checks=hostchecks))
    overhead_rows=[r for r in rows if r['spec']['stage']=='overhead'];gate=None
    if (c.OUT/'overhead_gate.json').exists():
        gate=overhead_gate(overhead_rows)
        c.require(gate==c.read(c.OUT/'overhead_gate.json'),'Independent observer overhead gate differs')
    c.require(not any(r['spec']['stage']=='diagnostic' for r in rows) or gate is not None and gate['passed'],
              'Diagnostic requests require qualified observer overhead')
    if len(rows)==72 and all(r['completed'] for r in rows) and not invalid:c.require(gate is not None,'Complete overhead gate required')
    coverage.update(complete_scientific_workload=coverage['complete'],complete=coverage['complete'] and not invalid,
        measurement_failure_verified=bool(invalid),invalid_observer_windows=len(invalid),
        overhead_gate_passed=None if gate is None else gate['passed'],
        overhead_qualified=False if invalid else (None if gate is None else gate['passed']),
        overhead_not_qualified=not invalid and bool(gate and gate['complete'] and not gate['passed']),
        overhead_requests=len(overhead_rows),diagnostic_requests=len(rows)-len(overhead_rows),host_observer_verified=hostproof is not None)
    c.save(c.OUT/'all_closed_workers.json',dict(status='passed',workers=len(proofs),proofs=proofs))
    c.save(c.OUT/'all_placement_integrity.json',verify_exchanges(envelope,freeze))
    c.save(c.OUT/'scientific_audit.json',dict(coverage=coverage,exact_repetitions=parity))
    result=analysis.analyze(rows,panel,matrix,coverage['complete'],boundary=envelope.guard)
    c.save(c.OUT/'analysis.json',result);c.save(c.OUT/'coverage.json',coverage)
    seal=[]
    for sub in ('workers','records','entrypoint_checks','closed_workers','closed_worker_costs','placement_requests','placement_acks','resident_stage_transitions','host_observations'):
        for path in sorted((c.OUT/sub).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    for name in ('qualification_summary.json','collection_stop.json','overhead_gate.json','scientific_audit.json',
                 'analysis.json','coverage.json','all_closed_workers.json','all_placement_integrity.json',
                 'host_observer_identity.json','host_observer_close.json','measurement_stop.json','host_observer_forensic_audit.json'):
        if(c.OUT/name).exists():envelope.guard();seal.append(c.artifact(c.OUT/name))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope);preservation=verify_prior_preservation(freeze,envelope)
    c.save(c.OUT/'verification_summary.json',dict(status='passed',coverage=coverage,workers=len(proofs),
        source_inputs=len(freeze['inputs']),parent_requests=sum(p['raw']['requests'] for p in proofs),
        graph_cases=11*len(proofs),qualification_scientific_decisions=104 if qualified else None,
        peak_combined_rss_bytes=envelope.peak_combined_rss,prior_preservation=preservation,
        all_raw_placement_logs_graph_science_sources_verified=True,host_observer_proof=hostproof,no_games_or_admission=True))
    return coverage


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Execution freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze changed')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json')
    envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha,phase=c.PHASE))
    status,failure,coverage,qualified='incomplete',None,None,None
    workers={};rows=[];ready=False;observer=None;first_results={};collection_measurement_failure=False
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);envelope.admit();switch_placement(envelope,freeze,'isolated')
        panel,matrix,plan=load_inputs(freeze,envelope);ready=True
        qualified=qualify(envelope,freeze,workers);envelope.finish()
        envelope.begin('overhead');bind_collection_deadline(envelope,workers,rows)
        observer=start_host_observer(workers,envelope,freeze)
        collect_stages(envelope,freeze,matrix,workers,rows,first_results)
    except BaseException as error:
        collection_measurement_failure=isinstance(error,MeasurementFailure)
        failure=traceback.format_exc();c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    finally:
        closure_errors=[]
        if observer is not None:
            try:c.save(c.OUT/'host_observer_close.json',observer.close(timeout_seconds=max(0.,min(5.,envelope.remaining()))))
            except BaseException:closure_errors.append(dict(role='host_observer',error=traceback.format_exc()))
        for route,child in workers.items():
            try:close_and_audit(child,envelope,freeze)
            except BaseException:closure_errors.append(dict(route=route,error=traceback.format_exc()))
        if closure_errors:
            collection_measurement_failure=False
            c.save(c.OUT/'closure_failures.json',dict(original_failure=failure,errors=closure_errors))
            failure=(failure or '')+'\nOwned closure/integrity failed: '+c.canonical(closure_errors)
    if ready and not envelope.resident:
        try:
            endpath=c.OUT/'stages'/f'{envelope.stage}_end.json'
            if not endpath.exists():
                try:envelope.finish()
                except BaseException:
                    collection_measurement_failure=False
                    boundary_error=traceback.format_exc()
                    c.save(c.OUT/'collection_end_failure.json',dict(time=c.now(),error=boundary_error))
                    failure=(failure or '')+boundary_error
            envelope.begin('final')
            coverage=final_checks(envelope,freeze,panel,matrix,plan,qualified);envelope.finish()
            stop=c.read(c.OUT/'collection_stop.json') if(c.OUT/'collection_stop.json').exists() else None
            if stop and not stop['diagnostic_deadline_failure']:
                failure=failure or 'Unclassified rejected request; verified evidence does not accept scientific/infrastructure failure.'
            if collection_measurement_failure and coverage['measurement_failure_verified'] and coverage['failed_requests']==0:
                status='incomplete_measurement_failure_verified'
            elif failure is None:
                status='completed_verified' if coverage['complete'] else ('overhead_not_qualified_verified' if coverage['overhead_not_qualified'] else 'verified_partial_diagnostic')
        except BaseException:
            error=traceback.format_exc();c.save(c.OUT/'finalization_failure.json',dict(time=c.now(),error=error));failure=(failure or '')+error
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,coverage=coverage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,phase=c.PHASE,automatic_retries=0,new_games=0,larger_data_deferred=True,training=False,
        no_admission_or_strength_claim=True,requires_all_three_clean_actual_os_exits=True,
        actual_os_exit_requirement_scope='successful_study_acceptance',incomplete_verified_requires_nonzero_actual_exits=True))
    return 1 if status in ('incomplete','incomplete_measurement_failure_verified') else 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run'])
    parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    return run(args.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())


