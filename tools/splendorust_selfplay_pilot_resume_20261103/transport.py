"""Trusted transport with explicit256/512 identities and common evidence limits."""
from copy import deepcopy
import hashlib,json,math,os,queue,subprocess,sys,threading,time
from pathlib import Path
import psutil
from . import common as c,adapters,scientific as clock_policy
from tools.splendorust_search256_recovery_20261030_02 import placement
from tools.external_strength_screen_20261018 import rules
from tools.local_player_qualification_128.clock import PersistentWorker,THREAD_ENV,_WindowsJob,_new_json,WorkerInitializationError
from .parent_profile import ProfiledStudyWorker
from tools.splendorust_boundary_timing_20261025.runner import BACKEND,validate_graph_parity,exact_output_hashes
from . import boundary_adapter as boundaries
from tools.splendorust_boundary_timing_20261025.analyze import CHILD_SCHEMA,GRAPH_IDENTITY,_int,_finite,_sha,_digest
from tools.splendorust_live_boundary_20261025.raw_transport import _packet_evidence,_bind_parent_wire,_bind_child_wire,REQUEST_ID,MAX_FAILED_LINES
from tools.splendorust_search_scale_20261024.compression import verify_log
SIDECAR_WAIT_SECONDS=1.
OBSERVER_MODULE='tools.splendorust_selfplay_pilot_resume_20261103.worker'
require=c.require
check_clock=clock_policy.check_clock

def lifecycle_inventory():
    from .qualification_control import lifecycle_inventory as planned
    return planned()

def declared_lifecycle(folder,freeze,route):
    folder=Path(folder);expected=lifecycle_inventory()
    c.require(freeze['declared_worker_lifecycles']==expected,'Exact prospective worker lifecycle inventory')
    c.require(folder.resolve()==(c.OUT/'workers'/folder.name).resolve(),'Current worker evidence root')
    c.require(folder.name in expected and route==expected[folder.name]['route'],'Declared exact worker route')
    return dict(expected[folder.name])

def authenticated_lifecycle(folder,freeze,route):
    folder=Path(folder);value=declared_lifecycle(folder,freeze,route)
    payloads={p.stem:c.read(p).get('operation') for p in (folder/'startup_inputs').glob('*.json')}
    expected={value['graph_request_id']:'graph_parity'}
    if value['entrypoint_request_id'] is not None:expected[value['entrypoint_request_id']]='entrypoint_checks'
    c.require(payloads==expected,'Exactly declared graph/suite prefix; no injection or repetition')
    ordinary=sorted((folder/'inputs').glob('*.json'));first=int(value['ordinary_first_request_id'].split('-')[1])
    c.require(len(ordinary)<=value['ordinary_requests'] and
        [p.stem for p in ordinary]==[f'request-{i:08d}' for i in range(first,first+len(ordinary))],
        'Contiguous bounded ordinary suffix after declared startup')
    if value['entrypoint_request_id'] is not None:
        receipt=c.read(folder/(value['entrypoint_request_id']+'.json'))
        c.require(receipt==c.read(folder/'entrypoint_clock.json') and receipt['status']=='ok' and
            receipt['request_id']==value['entrypoint_request_id'] and receipt['result']['execution_route']==route and
            receipt['result']['counts']['complete_decisions']==67,'Original owning actual-suite identity/count')
    return value

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
        self.recovery_accounting.bind_child(self)
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
                      "absolute_deadline": self.absolute_deadline,
                      "thread_environment": {name: child_env[name] for name in THREAD_ENV}})
            if receipt["status"] != "ok":
                self.close()
                raise WorkerInitializationError(receipt)
        except BaseException:
            self.close()
            raise

class PlacedProfiledStudyWorker(ProfiledStudyWorker, _PlacementConstructor):
    def __init__(self, *args, placement_freeze, placement_mode, recovery_accounting, native_roles=None, aggregate_sample_guard=None, **kwargs):
        self.recovery_accounting = recovery_accounting
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
    accounting = child.recovery_accounting
    accounting.begin_request(f'request-{child._counter+1:08d}', payload,
        pid=child.process.pid, quota=child.challenger_quota,
        graph_identity_sha256=child._placement_freeze['unchanged_graph_backend_identity']['graph_identity_sha256'],
        boundary=child.recovery_envelope.time_guard,
        lock_timeout=min(1.,max(0.,child.recovery_envelope.remaining())))
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
    finally:
        try:
            child.recovery_accounting.finish_request(require_published=receipt['status']=='ok',
                boundary=envelope.time_guard,lock_timeout=min(1.,max(0.,envelope.remaining())))
            if getattr(child,'_closed',False):
                envelope.close_worker(child.output_dir,child)
        except BaseException as error:
            result.update(profile_capture_status='evidence_error',profile_capture_error=repr(error))

def _require_profiles(profiles):
    c.require('placement_error' not in profiles,'Supervisor placement evidence error: '+str(profiles.get('placement_error')))
    c.require('profile_capture_error' not in profiles,'Observer evidence integrity error: '+str(profiles.get('profile_capture_error')))

def make_worker(name, envelope, freeze, *, route):
    """One of at most two live trees; original clocks/readers/sampler retained."""
    from . import adapters
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    c.require(route in adapters.ROUTES, 'Worker route allowlist')
    c.require(len(envelope.resident)<2, 'At most two resident live player trees are authorized')
    c.require('torch' not in sys.modules, 'Model-free supervisor required')
    c.require(len(envelope.evidence['workers']) < c.MAX_WORKER_LIFECYCLES, 'Worker lifecycle bound')
    envelope.admit()
    folder = c.OUT/'workers'/name
    declared_lifecycle(folder,freeze,route)
    envelope.register_worker(folder)
    module = OBSERVER_MODULE
    artifacts=adapters.native_artifacts(route);mapping=adapters.native_roles(route)
    c.require(freeze['native_artifacts_by_route'][route]==artifacts and freeze['native_roles_by_route'][route]==mapping,
        'Exact prospective native role artifacts')
    c.require(set(mapping)=={a['path'] for a in artifacts} and sorted(mapping.values())==['reference','search'],
        'Authenticated two-native route mapping')
    roles={(c.ROOT/path).resolve():role for path,role in mapping.items()}
    env = {'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'STUDY_WORKER_DIR': str(folder),
           'STUDY_SEARCH_SIMULATIONS': str(adapters.route_quota(route)), 'STUDY_ROUTE': route, 'STUDY_BOUNDARY_PROFILE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1', 'SPLENDOR_SEARCH256_STRENGTH_PHASE':'strength','SPLENDOR_SEARCH512_STRENGTH_PHASE':'strength','STUDY_SELFPLAY_PHASE':c.PHASE,
           'SELFPLAY_JOB_SCOPE':c.SCOPE}
    env.update(STUDY_RESEARCH_FREEZE=str(c.OUT/'freeze.json'),
               STUDY_RESEARCH_FREEZE_SHA256=c.file_hash(c.OUT/'freeze.json'))
    child = PlacedProfiledStudyWorker(module, folder, placement_freeze=freeze, placement_mode='isolated',
        recovery_accounting=envelope.worker_accounting[folder],
        native_roles=roles, native_artifacts=artifacts, aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[0], initialization_timeout=envelope.timeout(60), absolute_deadline=envelope.effective_deadline(),
        max_message_bytes=c.MAX_RESPONSE_BYTES, max_request_bytes=c.MiB, max_log_bytes=c.MAX_LOG_BYTES,
        max_rss_bytes=2*c.GiB, resource_guard=envelope.guard, env=env)
    try:
        envelope.own_worker(child)
        child.recovery_accounting = envelope.worker_accounting[folder]
        child.recovery_envelope = envelope
        child.study_after_close = lambda: envelope.close_worker(folder, child)
        child.route, child.runtime_mode, child.challenger_quota = route, 'candidate', adapters.route_quota(route)
        child.observation_mode = 'profiled'
        telemetry = child._sample()
        metadata = deepcopy(child.initialization['result'])
        adapters.validate_ready(metadata,child,freeze,route=route)
        from .worker import verify_execution_identity
        verify_execution_identity(folder,freeze,metadata)
        c.require(not child._failure, 'Worker supervision failed at readiness')
        c.save(folder/'route_identity.json', dict(route=route, module=module, native_artifacts=artifacts,
            native_roles=mapping, metadata=child.initialization['result']))
        child.recovery_accounting.finish_initialization(boundary=envelope.time_guard,
            lock_timeout=min(1.,max(0.,envelope.remaining())))
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
    from . import scientific as audit
    envelope.admit()
    c.require(spec.get('quota',adapters.route_quota(child.route))==adapters.route_quota(child.route) and spec.get('route',child.route)==child.route,'Fixed live route/quota')
    payload=dict(player='search',observer=observer)
    request_id=f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json',payload)
    identity=audit.profile_identity(freeze,adapters.route_quota(child.route),3,route=child.route,boundary=envelope.guard)
    legal=[rules.serialize_action(a) for a in rules.legal_actions(rules.state_from_observation(observer))]
    def validate(result):
        if type(result) is not dict or 'action' not in result:raise ValueError('PLAYER_MALFORMED_RESPONSE: missing action')
        if result['action'] not in legal:raise ValueError('PLAYER_ILLEGAL_ACTION: response action')
        return adapters.validate_result(result,observer,child.route,identity=identity)
    c.require(envelope.remaining()>=c.SECONDS_PER_MOVE,'Insufficient stage time for full research request clock')
    allowance=envelope.timeout(c.SECONDS_PER_MOVE)
    c.require(allowance==c.SECONDS_PER_MOVE,'Do not shorten or reset the request allowance')
    receipt=request(child,payload,allowance,validate=validate)
    row=dict(spec=deepcopy(spec),observer=observer,completed=receipt['status']=='ok',clock=receipt,
        route=child.route,worker_folder=child.output_dir.relative_to(c.ROOT).as_posix(),
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix(),
        observation_mode='profiled',action=None,trace=None)
    row.update(collect_profiles(child,receipt,envelope))
    if row['completed']:row.update(action=receipt['result']['action'],trace=receipt['result']['diagnostics'])
    return row


def verify_capture_publication(folder,receipt,observer,freeze):
    """Capture is durably published before the existing boundary barrier."""
    folder=Path(folder);name=folder.name
    c.require(name in lifecycle_inventory(),'Frozen capture lifetime')
    path=folder/(receipt['request_id']+'_training_capture.json')
    c.require(path.is_file() and path.stat().st_size<=c.MiB,'Bounded complete root-capture publication')
    value=c.read(path);initial=c.read(folder/'initialization.json')
    expected={'schema','request_id','pid','observer_sha256','result_sha256','capture_targets','training_capture'}
    c.require(set(value)==expected and value['schema']=='selfplay_capture_publication_v1' and
        value['request_id']==receipt['request_id'] and value['pid']==initial['identity']['pid'] and
        value['observer_sha256']==c.digest(observer) and value['result_sha256']==c.digest(receipt['result']) and
        value['capture_targets'] is lifecycle_inventory()[name]['capture_targets'],
        'Exact root-target/public-observer/accepted scientific response binding')
    c.require(type(value['training_capture']) is dict,'Explicit target or mechanically excluded root')
    if value['capture_targets']:
        from .capture_adapter import validate_capture
        validate_capture(value['training_capture'],observer,expected_quota=512)
    else:
        c.require(value['training_capture']==dict(schema='selfplay_root_capture_disabled_v1',enabled=False,
                observer_sha256=c.digest(observer),quota=512,training_eligible=False),'Capture-OFF marker has no invented target')
    return value

def deadline_failure(receipt,route):
    from . import scientific as audit
    try:audit.classify_failure(receipt);return True
    except ValueError:return False

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
    envelope.release_verified_worker(child.output_dir)
    return proof

def research_child_identity(child, parent, receipt, payload, quota, *, route, completed, startup=False, worker_pid=None):
    require(type(child) is dict and child.get("schema") == CHILD_SCHEMA, "child sidecar schema")
    require(len(json.dumps(child, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <=
            (512*1024 if startup else 8192), "bounded child sidecar")
    require(child.get("request_id") == receipt["request_id"] and child.get("quota") == quota, "child request/quota")
    require(child.get("backend") == "qualified_graph64" and
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
        require(size <= c.MAX_RESPONSE_BYTES, "serialized response size bound")
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
    metadata=init['result'];route=c.read(folder/'route_identity.json')['route'];quota=adapters.route_quota(route);mode=route
    c.require(route in adapters.ROUTES,'Only frozen64 and research routes')
    lifecycle=authenticated_lifecycle(folder,freeze,route)
    paths=sorted(path for path in folder.glob('request-*.json') if REQUEST_ID.fullmatch(path.stem))
    c.require(len(paths)<=lifecycle['startup_requests']+lifecycle['ordinary_requests'],'Bounded declared request inventory')
    c.require([p.stem for p in paths]==[f'request-{i:08d}' for i in range(1,len(paths)+1)],
              'Contiguous original worker request inventory')
    receipts={path.stem:c.read(path) for path in paths};ids=list(receipts)
    failures=[rid for rid,receipt in receipts.items() if receipt['status']!='ok']
    c.require(not failures or failures==[ids[-1]],'At most one final failed request per worker')
    failed_id=failures[0] if failures else None
    packets={rid:[] for rid in ids};pending=bytearray();ready=False;index=0;partial=None
    def packet_line(raw):
        nonlocal ready,index
        envelope.guard();c.require(len(raw)<=c.MAX_RESPONSE_BYTES and raw.endswith(b'\n'),'Raw packet frame bound')
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
        c.require(len(pending)<=c.MAX_RESPONSE_BYTES,'Raw pending response frame bound');envelope.guard()
    verified=verify_log(folder/'stdout.log.gz',cleanup['log_compression']['stdout.log'],
        max_raw_bytes=c.MAX_LOG_BYTES,max_compressed_bytes=c.MAX_LOG_BYTES*16+128,
        deadline=envelope.effective_deadline(),sink=sink)
    c.require(ready,'Closed worker raw READY missing')
    if pending:
        c.require(failed_id is not None and index==len(ids)-1,'Partial output without final failed request')
        partial=dict(bytes=len(pending),sha256=hashlib.sha256(pending).hexdigest(),complete_packet=False)
    c.require(index==len(ids) if failed_id is None else index==len(ids)-1,
              'Missing completed raw response')
    input_ids=[p.stem for directory in ('inputs','startup_inputs') for p in (folder/directory).glob('*.json')]
    c.require(len(input_ids)==len(ids) and set(input_ids)==set(ids),'Exact input request inventory')
    ordinary_ids={p.stem for p in (folder/'inputs').glob('*.json')}
    c.require({p.name for p in folder.glob('request-*_training_capture.json')}=={rid+'_training_capture.json' for rid in ordinary_ids},'Exact root capture publication inventory')
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
                c.require(rid==lifecycle['entrypoint_request_id'],'Exact authenticated retained-worker entrypoint request')
            else:
                c.require(rid==lifecycle['graph_request_id'],'Exact graph request precedes ordinary play')
                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
                c.require(quota in (256,512) and payload=={'operation':'graph_parity','fixtures':plan['fixtures']},
                          'Exact original graph startup payload required')
        else:
            payload=c.read(ordinary)
            c.require(set(payload)=={'player','observer'} and payload['player']=='search','Exact fixed live payload')
            c.require(receipt['deadline_seconds']==c.SECONDS_PER_MOVE,'Exact full research route allowance required')
            if receipt['status']=='ok':verify_capture_publication(folder,receipt,payload['observer'],freeze)
            for packet in packets[rid]:
                c.require(packet['raw_bytes']<=c.MAX_RESPONSE_BYTES,'Common ordinary response exceeds8MiB cap')
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
    expected_startups=lifecycle['startup_requests']
    c.require(startup_count==expected_startups,'Original startup request inventory differs')
    return dict(status='passed',worker=folder.name,requests=len(ids),startup_requests=startup_count,
        observed=True,quota=quota,scientific_mode=mode,stdout_log=verified,exact_bytes_checked=True,
        failed_requests=len(failures),failed_request_id=failed_id,partial_failed_output=partial,requests_checked=checks,
        newline_normalization='terminal CRLF to LF only; JSON bytes never reserialized',
        failed_packets_remain_unaccepted=True,original_trusted_receipts_unchanged=True)

def validate_profiles(parent, child, receipt, payload, route, *, worker_pid=None):
    from . import scientific as audit
    startup = 'operation' in payload
    completed = receipt['status'] == 'ok'
    if not startup:
        audit.check_clock(receipt,limit=c.SECONDS_PER_MOVE)
    returned_response = completed or receipt['status']=='illegal' or receipt.get('result') is not None
    if not completed and returned_response and child is not None:
        c.require(child['outcome']=='response_flushed' and child.get('error') is None and
            child['result_sha256']==c.digest(receipt['result']),
            'Retained rejected response/child scientific hash binding')
        c.require(startup or len(child['profiles'])==1 and child['profiles'][0]['handle_failed'] is False,
            'Rejected returned response requires completed worker handle')
    c.require(route in adapters.ROUTES, 'Explicit separately versioned route')
    limit = (300. if payload.get('operation')=='entrypoint_checks' else 60.) if startup else c.SECONDS_PER_MOVE
    if startup:
        from tools.splendorust_search_scale_20261024 import audit as original_science
        original_science.check_clock(receipt,limit=limit)
    else:
        audit.check_clock(receipt, limit=limit)
    c.require(startup or receipt['deadline_seconds']==c.SECONDS_PER_MOVE, 'Exact research complete-response allowance required')
    c.require(parent is not None, 'Original parent boundary evidence required')
    start,arrival,metrics = boundaries._validate_parent(parent,receipt,completed=returned_response)
    if child is None:
        c.require(not completed, 'Successful research child sidecar absent')
        return dict(status='preserved_failed_request_missing_child',parent=metrics)
    quota = adapters.route_quota(route)
    research_child_identity(child,parent,receipt,payload,quota,route=route,completed=returned_response,startup=startup,worker_pid=worker_pid)
    profiles = child['profiles']
    c.require(type(profiles) is list and len(profiles)<=(64 if startup else 1), 'Exact bounded handle inventory')
    begin=end=None
    if startup and completed and payload=={'operation':'entrypoint_checks'}:
        result=receipt['result']
        c.require(result['status']=='entrypoint_counterexamples_passed' and
            result['counts']['complete_decisions']==67 and len(profiles)==63, 'Actual research52 suite/63handles')
        previous=child['request_read_complete_ns']
        for index,profile in enumerate(profiles):
            c.require(profile['handle_failed'] is (index>=37), 'Exact research startup success/rejection order')
            _,left,right=boundaries._validate_handle(profile,child,completed=index<37,startup=True)
            c.require(previous<=left<=right<=child['pre_write_ns'], 'Startup handle order')
            previous=right
    elif startup:
        c.require(payload.get('operation')=='graph_parity' and not profiles,'Graph startup handle inventory')
    elif profiles:
        _,begin,end=boundaries._validate_handle(profiles[0],child,completed=returned_response)
    else:
        c.require(not completed,'Completed ordinary research handle absent')
    result=research_boundaries(child,parent,start,arrival,begin=begin,end=end,startup=startup)
    return dict(status='passed',parent=metrics,child=result,original_receipt_unmodified=True,research_deadline_seconds=c.SECONDS_PER_MOVE,
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
    cap=c.MAX_LOG_BYTES
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
    expected_artifacts=freeze['native_artifacts_by_route'][route]
    expected_roles=freeze['native_roles_by_route'][route]
    c.require(identity==dict(route=route,module=OBSERVER_MODULE,
        native_artifacts=expected_artifacts,native_roles=expected_roles,metadata=metadata), 'Saved READY/role/module identity binding')
    adapters.verify_saved_ready(folder,freeze,route=route)
    from .worker import verify_execution_identity
    execution=verify_execution_identity(folder,freeze,metadata)
    cleanup=c.read(folder/'cleanup.json')
    c.require(not cleanup['alive'] and not cleanup['errors'] and cleanup['deadline_clamped_to_stage_and_global'] is True,
              'Owned worker cleanup errors remain failures')
    for owned in cleanup['confirmed']:
        envelope.guard()
        try:proc=psutil.Process(owned['pid'])
        except psutil.NoSuchProcess:continue
        c.require(proc.create_time()!=owned['created'],'Owned worker identity remains alive after cleanup')
    native_configuration=adapters.verify_native_configuration_receipts(folder,route,boundary=envelope.guard)
    graph=verify_graph_evidence(folder,freeze,boundary=envelope.guard)
    parents=sorted((folder/'parent_profiles').glob('request-*.json'))
    c.require([p.name for p in parents]==[p.name for p in sorted((folder/'placement_requests').glob('request-*.json'))],
              'Complete per-request placement inventory')
    for path in parents:
        envelope.guard();verify_placement_request(folder,path.stem,freeze,c.read(path))
    envelope.close_worker(folder);envelope.guard()
    return dict(status='passed',worker=folder.name,route=route,logs=logs,raw=raw,graph=graph,
                placement_requests=len(parents),placement_mode='isolated',native_configuration=native_configuration,execution_identity=execution)

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
    c.require(execution['metadata'].get('route')==route,'Graph route identity differs')
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
