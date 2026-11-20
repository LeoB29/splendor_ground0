"""Separate supervisor placement; exact qualified scientific workers and clocks."""
import json
import math
import queue
import subprocess
import threading
import psutil
from tools.local_player_qualification_128.clock import (PersistentWorker, THREAD_ENV, _WindowsJob, _new_json, WorkerInitializationError)
from . import placement
from copy import deepcopy
import os
from pathlib import Path
import sys
import time

from . import common as c
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.splendorust_search_scale_20261024 import audit as scientific
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker
from tools.splendorust_boundary_timing_20261025 import analyze as observer_audit
from tools.splendorust_boundary_timing_20261025.runner import (
    verify_instrumentation_identity, validate_graph_parity, verify_graph_evidence, BACKEND)

OBSERVER_MODULE='tools.splendorust_boundary_timing_20261025.worker'
SIDECAR_WAIT_SECONDS=1.


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
    def __init__(self, *args, placement_freeze, placement_mode, aggregate_sample_guard=None, **kwargs):
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


def verify_worker_placement(folder, freeze, placement_mode):
    return placement.verify_worker_placement(folder, freeze, placement_mode)


def require_player(mode,quota):
    c.require(type(quota) is int and ((mode=='baseline' and quota==32) or
              (mode=='candidate' and quota==64)), 'Only unchanged eager32 and qualified graph64 are authorized')


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


def graph_parity(child,envelope,freeze):
    item=freeze['graph_parity_plan'];c.verify([item]);plan=c.read(c.ROOT/item['path']);fixtures=plan['fixtures']
    c.require(len(fixtures)==11,'Fixed graph parity inventory')
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request(child,payload,envelope.timeout(60))
    c.save(child.output_dir/'graph_parity_clock.json',receipt)
    profiles=collect_profiles(child,receipt,envelope)
    _require_profiles(profiles);envelope.guard()
    c.require(receipt['status']=='ok','Graph output parity failed before decisions; no fallback/retry')
    validate_graph_parity(receipt['result'],fixtures)
    observer_audit.validate_startup_profiles(profiles['parent_profile'],profiles['child_profile'],
        receipt,payload,child.challenger_quota,worker_pid=child.process.pid)
    c.require(c.read(child.output_dir/f"{receipt['request_id']}.json")==receipt,'Graph parity raw receipt changed')
    verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)
    return receipt


def make_worker(name,envelope,freeze,*,mode,quota,placement_mode='original'):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    require_player(mode,quota)
    c.require('torch' not in sys.modules,'Model-free supervisor required')
    envelope.admit();folder=c.OUT/'workers'/name;envelope.register_worker(folder)
    placement.verify_runtime(freeze,placement_mode)
    child=PlacedProfiledStudyWorker(OBSERVER_MODULE,folder,placement_freeze=freeze,placement_mode=placement_mode,
        native_artifacts=freeze['native_artifacts'],aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']],initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(),max_message_bytes=4*c.MiB,max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB,max_rss_bytes=2*c.GiB,resource_guard=envelope.guard,
        env={'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'STUDY_WORKER_DIR':str(folder),
             'STUDY_SEARCH_SIMULATIONS':str(quota),'STUDY_BOUNDARY_PROFILE':'1','CUDA_VISIBLE_DEVICES':'',
             'RAYON_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'1'})
    try:
        envelope.own_worker(child);child.study_after_close=lambda:envelope.close_worker(folder,child)
        telemetry=child._sample();metadata=deepcopy(child.initialization['result'])
        if mode=='candidate':c.require(metadata.pop('inference_backend')==BACKEND,'Candidate inference backend changed')
        validate_ready_metadata(metadata,expected_sources=SOURCE_HASHES,worker_pid=child.process.pid,
            process_tree=telemetry['process_tree'],expected_quota=quota)
        child.runtime_mode,child.challenger_quota,child.observation_mode=mode,quota,'profiled'
        c.require(not child._failure,'Worker supervision failed at readiness')
        verify_instrumentation_identity(folder,freeze,quota,child.initialization['result'])
        if mode=='candidate':graph_parity(child,envelope,freeze)
        envelope.guard();return child
    except BaseException:
        child.close();raise


def actual_entrypoint(child,envelope,label=None):
    """Return the original52-decision receipt; caller saves its summary path."""
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request(child,payload,envelope.timeout(300))
    profiles=collect_profiles(child,receipt,envelope)
    _require_profiles(profiles);envelope.guard()
    c.require(receipt['status']=='ok' and receipt['result']['status']=='entrypoint_counterexamples_passed' and
        receipt['result']['challenger_quota']==child.challenger_quota and
        receipt['result']['counts']['complete_decisions']==52,'Actual corrected-entrypoint counterexamples failed; no retry')
    observer_audit.validate_startup_profiles(profiles['parent_profile'],profiles['child_profile'],
        receipt,payload,child.challenger_quota,worker_pid=child.process.pid)
    return receipt


def decision(child,state,envelope,role):
    c.require(role in ('challenger','opponent') and child.challenger_quota==(64 if role=='challenger' else 32) and
        child.runtime_mode==('candidate' if role=='challenger' else 'baseline') and child.observation_mode=='profiled',
        'Role/quota/observation binding differs')
    require_player(child.runtime_mode,child.challenger_quota);envelope.admit()
    observer,before=rules.observation_payload(state),rules.serialize_snapshot(state)
    ledger.check_observer(before,observer);legal=rules.legal_actions(state)
    actions=[serialize_action(action) for action in legal];request_id=f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json',dict(player='search',observer=observer,before=before))
    def validate(result):
        if not isinstance(result,dict) or set(result)!=scientific.RESULT_FIELDS or not isinstance(result['diagnostics'],dict):
            raise ValueError('PLAYER_MALFORMED_RESPONSE: response envelope fields')
        if result['action'] not in actions:
            raise ValueError('PLAYER_ILLEGAL_ACTION: response action')
        return scientific.validate_result(result,observer,'search',expected_quota=child.challenger_quota)
    c.require(envelope.remaining()>=1.,'Insufficient stage time for a full one-second decision')
    allowance=envelope.timeout(1.);c.require(allowance==1.,'Stage guard shortened the next move allowance; do not dispatch')
    receipt=request(child,{'player':'search','observer':observer},allowance,validate=validate)
    row=dict(arm='search',player_role=role,quota=child.challenger_quota,mode=child.runtime_mode,
        observation_mode='profiled',seat=state.current_player,observer=observer,before=before,after=None,
        action=None,trace=None,completed=receipt['status']=='ok',clock=receipt,
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix())
    row.update(collect_profiles(child,receipt,envelope))
    if row['completed']:
        result=receipt['result']
        c.require(result['player']=='search' and result['observer_sha256']==c.digest(observer) and
            result['legal_actions_sha256']==legal_digest(legal) and result['legal_count']==len(legal),
            'Worker observation/legal-support binding differs')
        row.update(action=result['action'],trace=result['diagnostics'])
        row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,deserialize_action(row['action'])))
    # The caller must save this row before evaluating any observer-integrity
    # error or classifying the original failure. Never revise the clock/result.
    return row


def verify_raw_transport(folder,freeze,envelope):
    from tools.splendorust_live_boundary_20261025.raw_transport import verify_raw_transport as verify
    return verify(folder,freeze,envelope)
