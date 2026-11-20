"""One frozen qualification, operational and400-game run under one24-hour clock."""
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
from .transport import (_require_profiles, close_and_audit, collect_profiles, deadline_failure, make_worker, ordinary, request, research_boundaries, research_child_identity, validate_profiles, verify_closed_worker, verify_graph_evidence, verify_placement_request, verify_raw_transport, verify_saved_record, verify_worker_logs)
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
        c.require(len(self.resident)<2,'At most two resident worker trees')
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
        c.require(len([i for i in self.resident if i[0]!=child.process.pid])<2,'Constructing a third resident worker is forbidden')
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
        stable = begun if available >= c.MINIMUM_AVAILABLE_RAM_BYTES else None
        minimum = available
        fast = self.admitted and stable is not None
        while not fast:
            self.guard()
            current, available = time.monotonic(), psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < c.MINIMUM_AVAILABLE_RAM_BYTES:
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
        minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,seconds_per_move=c.SECONDS_PER_MOVE,internal_seconds=c.INTERNAL_SECONDS,
        research_seconds_per_move=c.SECONDS_PER_MOVE,research_internal_seconds=c.INTERNAL_SECONDS,research_log_max_bytes=128*c.MiB,
        frozen_log_max_bytes=128*c.MiB,research_response_max_bytes=4*c.MiB,frozen_response_max_bytes=4*c.MiB,
        request_max_bytes=c.MiB,placement_mode='isolated',max_workers=c.MAX_WORKERS,max_graph_builds=c.MAX_GRAPH_BUILDS,
        phase=c.PHASE,qualification_decisions=c.QUALIFICATION_DECISIONS,qualification_ordinary_requests=c.QUALIFICATION_ORDINARY_REQUESTS,independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        operational_games=6,main_games=400,
        startup_minimum_available_ram_bytes=c.STARTUP_MINIMUM_AVAILABLE_RAM_BYTES,maximum_reproductions=48,automatic_retries=0,failed_decision_retries=0,training=False)
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




















from tools.splendorust_boundary_timing_20261025 import analyze as boundaries
from tools.splendorust_boundary_timing_20261025.analyze import CHILD_SCHEMA, GRAPH_IDENTITY, _int, _finite, _sha, _digest
from tools.splendorust_live_boundary_20261025.raw_transport import _packet_evidence, _bind_parent_wire, _bind_child_wire, REQUEST_ID, MAX_FAILED_LINES
from tools.splendorust_search_scale_20261024.compression import verify_log
require = c.require















from tools.splendorust_boundary_timing_20261025.runner import validate_graph_parity, exact_output_hashes


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def decision(child,state,envelope,role,freeze):
    from . import audit
    c.require(role in audit.ROLES and child.route==audit.MODES[role], 'Exact live role/route')
    before,observer=rules.serialize_snapshot(state),rules.observation_payload(state)
    decode_training_snapshot(before);ledger.check_observer(before,observer)
    row=ordinary(child,dict(route=child.route,quota=c.ROUTE_QUOTAS[child.route]),observer,envelope,freeze)
    row.update(arm='search',player_role=role,quota=c.ROUTE_QUOTAS[child.route],mode=child.route,seat=state.current_player,
               before=before,after=None)
    if row['completed']:
        try:row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,rules.deserialize_action(row['action'])))
        except BaseException:row['transition_error']=traceback.format_exc()
    return row


def verify_saved_decision(row,freeze):
    from . import audit
    verify_saved_record(row,freeze)
    c.require(row['mode']==row['route']==audit.MODES[row['player_role']] and row['arm']=='search' and
        row['quota']==c.ROUTE_QUOTAS[row['route']] and row['seat']==row['before']['state']['current_player'], 'Live scientific role binding')
    decode_training_snapshot(row['before']);ledger.check_observer(row['before'],row['observer'])
    if row['completed']:
        identity=audit.profile_identity(freeze,c.ROUTE_QUOTAS[row['route']],3,route=row['route'])
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
        collection_stopped=spec['stage']=='operational',time=c.now(),administrative_forfeit=True,
                failure_policy='research_hard_guard_9s_internal_10s_complete_response',
                hard_guard_seconds=c.SECONDS_PER_MOVE,internal_guard_seconds=c.INTERNAL_SECONDS)


def play_game(spec,envelope,freeze):
    from . import schedule as schedules, audit
    schedules.validate_spec(spec)
    state=rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state()
    initial=rules.serialize_snapshot(state);decode_training_snapshot(initial)
    folder=game_folder(spec);c.save(folder/'initial.json',dict(**spec,initial=initial))
    workers={};rows=[];fault=None;outcome=None;independent=0.;begun=time.perf_counter();original_error=None
    first_admission=envelope.admissions
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
                             failure=fault['failure_classification'],administrative_forfeit=True,
                             failure_policy='research_hard_guard_9s_internal_10s_complete_response')
                c.save(folder/'captured_fault.json',fault)
                if spec['stage']=='operational':c.save(c.OUT/'collection_stop.json',fault)
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
    wait_files=[c.artifact(p) for p in sorted((c.OUT/'memory_admissions').glob('*.json'))
                if first_admission<int(p.stem)<=envelope.admissions]
    waited=sum(c.read(c.ROOT/item['path'])['wait_seconds'] for item in wait_files)
    gross=time.perf_counter()-begun
    c.require(0<=waited<=gross,'Recorded inter-request RAM wait fits game wall time')
    c.save(folder/'cost.json',dict(independent_seconds=independent,collection_seconds=gross,
        explicit_inter_request_ram_wait_seconds=waited,nonwait_collection_seconds=gross-waited,
        memory_admission_first=first_admission,memory_admission_last=envelope.admissions,memory_wait_artifacts=wait_files,
        complete_decisions=sum(r['completed'] for r in rows),
        measured_two_pass_hash_seconds=None if hashing is None else hashing['seconds']))
    return ending,rows,independent,fault


def qualification_gate(rows, *, expected_requests):
    from . import audit
    c.require(type(expected_requests) is int and expected_requests>0 and len(rows)<=expected_requests,'Bounded frozen qualification requests')
    for row in rows:
        audit.check_clock(row['clock'])
        c.require(row['completed'] is (row['clock']['status']=='ok'),'Qualification completion/raw clock disagreement')
    complete=len(rows)==expected_requests and all(r['completed'] for r in rows)
    times=[r['clock']['receipt_elapsed_seconds'] for r in rows if r['completed']]
    return dict(passed=complete,complete=complete,requests=len(rows),expected_requests=expected_requests,completed=len(times),
        completed_over_one_second=sum(t>1. for t in times),maximum_seconds=max(times,default=None),
        timing_used_for_admission=False,research_outer_seconds=c.SECONDS_PER_MOVE,
        research_internal_seconds=c.INTERNAL_SECONDS,outcomes_used=False)


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
    gate=qualification_gate(rows,expected_requests=len(plan['parity_requests']))
    summary=dict(qualified=gate['passed'],requests=len(rows),failure=failure,timing_gate=gate)
    if failure is None:summary['parity']=qualification.validate_parity_rows(rows,plan,boundary=envelope.guard)
    c.save(c.OUT/'qualification_summary.json',summary)
    return summary


def operational_gate(*,observed_seconds,observed_gross_seconds,observed_variable_bytes,independent_seconds,
                     fixed_final_seconds,final_evidence_bytes,maximum_move_seconds,all_natural,
                     current_evidence_bytes,current_final_evidence_bytes,available_collection_seconds,
                     available_global_seconds,remaining_ram_wait_seconds,fixed_collection_seconds=600.):
    values=(observed_seconds,observed_gross_seconds,observed_variable_bytes,independent_seconds,fixed_final_seconds,
            final_evidence_bytes,maximum_move_seconds,fixed_collection_seconds,current_evidence_bytes,current_final_evidence_bytes,
            available_collection_seconds,available_global_seconds,remaining_ram_wait_seconds)
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values),'Finite measured cost inputs')
    c.require(observed_gross_seconds>=observed_seconds and remaining_ram_wait_seconds<=c.RAM_WAIT_SECONDS,'Gross and bounded RAM costs')
    factor=400/6*1.5
    collection=observed_seconds*factor+fixed_collection_seconds+remaining_ram_wait_seconds
    prefinal=current_evidence_bytes+observed_variable_bytes*factor+64*c.MiB
    final=independent_seconds*(406/6)*1.5+fixed_final_seconds+c.INDEPENDENT_REVIEW_SECONDS
    current_final=independent_seconds*1.5+fixed_final_seconds+c.INDEPENDENT_REVIEW_SECONDS+remaining_ram_wait_seconds
    current_resource=(current_evidence_bytes<c.PREFINAL_EVIDENCE_BYTES and current_final<min(c.STAGES['final'],available_global_seconds) and
        current_final_evidence_bytes<c.FINAL_RESERVE_BYTES and current_evidence_bytes+current_final_evidence_bytes<c.MAX_EVIDENCE_BYTES)
    assessment_resource=(collection<available_collection_seconds and prefinal<c.PREFINAL_EVIDENCE_BYTES and final<c.STAGES['final'] and
        collection+final<available_global_seconds and final_evidence_bytes<c.FINAL_RESERVE_BYTES and prefinal+final_evidence_bytes<c.MAX_EVIDENCE_BYTES)
    return dict(passed=current_resource and assessment_resource and all_natural,
        resource_passed=current_resource and assessment_resource,current_resource_passed=current_resource,
        assessment_resource_passed=assessment_resource,all_operational_games_natural=all_natural,
        observed_games=6,projected_games=400,conservative_factor=1.5,arithmetic_factor=factor,
        projected_collection_seconds=collection,projected_prefinal_bytes=prefinal,
        projected_final_seconds=current_final,projected_final_evidence_bytes=current_final_evidence_bytes,
        assessment_projected_final_seconds=final,assessment_projected_final_evidence_bytes=final_evidence_bytes,
        assessment_projected_total_evidence_bytes=prefinal+final_evidence_bytes,
        gross_game_cost_projection_seconds=observed_gross_seconds*factor+fixed_collection_seconds,
        ram_wait_charged_once_seconds=remaining_ram_wait_seconds,move_timing_subtracted_seconds=0.,
        collection_seconds_cap=available_collection_seconds,final_seconds_cap=c.STAGES['final'],
        global_seconds_remaining=available_global_seconds,independent_review_reserved_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        prefinal_bytes_cap=c.PREFINAL_EVIDENCE_BYTES,final_bytes_cap=c.FINAL_RESERVE_BYTES,total_bytes_cap=c.MAX_EVIDENCE_BYTES,
        maximum_move_seconds=maximum_move_seconds,timing_used_for_admission=False,research_outer_seconds=c.SECONDS_PER_MOVE,
        research_internal_seconds=c.INTERNAL_SECONDS,outcomes_used=False)


def measured_game_costs(schedule,boundary=lambda:None):
    measured=[];seen_waits=set()
    for spec in schedule:
        boundary();folder=game_folder(spec);cost=c.read(folder/'cost.json')
        waits=cost['memory_wait_artifacts']
        c.require(len({x['path'] for x in waits})==len(waits),'Duplicate recorded RAM wait')
        for item in waits:
            boundary();c.verify([item]);c.require(item['path'] not in seen_waits,'RAM wait charged to multiple games')
            seen_waits.add(item['path'])
            path=c.ROOT/item['path']
            c.require(path.parent==c.OUT/'memory_admissions' and
                cost['memory_admission_first']<int(path.stem)<=cost['memory_admission_last'], 'RAM wait game ownership')
        expected={p.relative_to(c.ROOT).as_posix() for p in (c.OUT/'memory_admissions').glob('*.json')
                  if cost['memory_admission_first']<int(p.stem)<=cost['memory_admission_last']}
        c.require(expected=={x['path'] for x in waits},'Complete recorded RAM wait inventory')
        waited=sum(c.read(c.ROOT/item['path'])['wait_seconds'] for item in waits)
        c.require(0<=waited<=cost['collection_seconds'] and waited==cost['explicit_inter_request_ram_wait_seconds'] and
            cost['nonwait_collection_seconds']==cost['collection_seconds']-waited,'RAM wait cost accounting; no move-clock subtraction')
        paths=[folder]+[c.OUT/'workers'/f"operational_g{spec['game_index']}_{role}" for role in ('challenger','opponent')]
        size=sum(p.stat().st_size for d in paths for p in d.rglob('*') if p.is_file())
        size+=sum((c.OUT/'closed_workers'/f"operational_g{spec['game_index']}_{role}.json").stat().st_size+
                  (c.OUT/'closed_worker_costs'/f"operational_g{spec['game_index']}_{role}.json").stat().st_size
                  for role in ('challenger','opponent'))
        measured.append(dict(game_index=spec['game_index'],collection_seconds=cost['collection_seconds'],
            nonwait_collection_seconds=cost['nonwait_collection_seconds'],explicit_inter_request_ram_wait_seconds=waited,
            memory_wait_artifacts=waits,independent_seconds=cost['independent_seconds'],bytes=size))
    return measured


def reproduction_fixed_components(envelope, *, admission_only=False):
    folders=[p for p in (c.OUT/'workers').iterdir() if not p.name.startswith('audit_') and (not admission_only or p.name.startswith(('qualification_','operational_')))]
    return dict(initialization=max(c.read(p/'initialization.json')['elapsed_seconds'] for p in folders),
        graph=max(c.read(p/'graph_parity_clock.json')['elapsed_seconds'] for p in folders),
        closure=max(c.read(c.OUT/'closed_worker_costs'/f'{p.name}.json')['seconds'] for p in folders),
        runtime=envelope.source_runtime_seconds)


def assessment_inputs(measured,components,final_evidence_bytes,maximum_move_seconds,
                      current_evidence_bytes,current_final_evidence_bytes,clock_budget):
    c.require(len(measured)==6,'Fixed six measured game costs')
    fixed=48*1.5*(components['initialization']+components['graph']+components['closure']+c.SECONDS_PER_MOVE+1.)+2*components['runtime']+300.
    return dict(observed_seconds=6*max(m['nonwait_collection_seconds'] for m in measured),
        observed_gross_seconds=6*max(m['collection_seconds'] for m in measured),
        observed_variable_bytes=6*max(m['bytes'] for m in measured),
        independent_seconds=6*max(m['independent_seconds'] for m in measured),fixed_final_seconds=fixed,
        fixed_collection_seconds=600.+2*components['runtime'],final_evidence_bytes=final_evidence_bytes,
        maximum_move_seconds=maximum_move_seconds,all_natural=True,current_evidence_bytes=current_evidence_bytes,
        current_final_evidence_bytes=current_final_evidence_bytes,**clock_budget)


def final_evidence_forecast(schedule,boundary=lambda:None,*,projected_games=406,maximum_workers=862,reserve_bytes=None):
    """Measured closure/graph/placement costs, conservatively bounded48 reproductions."""
    c.require(len(schedule)==6 and projected_games in (6,406) and maximum_workers==(62 if projected_games==6 else 862),'Six measurements and fixed prospective inventory')
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
    qualification_files=[p for folder in (c.OUT/'workers').glob('qualification_*') for p in folder.rglob('*') if p.is_file()]
    projected_files=math.ceil(1.5*((len(game_files)+len(worker_files))*projected_games/6+len(qualification_files)))+48*files
    item_bytes=max(len(c.canonical(dict(path=p.relative_to(c.ROOT).as_posix(),sha256='0'*64,bytes=2**63-1)).encode())+129
                   for p in game_files+worker_files)
    raw=48*(bound+largest_move)+(maximum_workers*largest_proof)+(projected_games*largest_replay)+projected_files*item_bytes+32*c.MiB
    return dict(outcomes_used=False,conservative_factor=1.5,maximum_reproductions=48,
        measured_workers=12,entire_game_worker_byte_bound=bound,largest_move_bytes=largest_move,
        largest_closed_proof_bytes=largest_proof,largest_online_replay_bytes=largest_replay,
        projected_sealed_files=projected_files,artifact_item_bytes=item_bytes,
        projected_final_evidence_bytes=math.ceil(1.5*raw),reserve_bytes=reserve_bytes if reserve_bytes is not None else 4*c.GiB,
        projected_games=projected_games,maximum_workers=maximum_workers)


def operational(envelope,freeze,schedule,games=None):
    games=[] if games is None else games
    allrows=[];start=time.perf_counter();initial_bytes=bytes_used();stop=None
    for spec in schedule:
        game,rows,cost,stop=play_game(spec,envelope,freeze);games.append(game);allrows.extend(rows)
        print(c.canonical(dict(stage='operational',completed_games=len(games),total=6,fault=stop is not None)),flush=True)
        if stop is not None:break
    if len(games)==6 and stop is None:
        current=final_evidence_forecast(schedule,envelope.guard,projected_games=6,maximum_workers=62,reserve_bytes=c.FINAL_RESERVE_BYTES)
        evidence=final_evidence_forecast(schedule,envelope.guard)
        c.save(c.OUT/'current_final_evidence_forecast.json',current);c.save(c.OUT/'final_evidence_forecast.json',evidence)
        components=reproduction_fixed_components(envelope,admission_only=True);measured=measured_game_costs(schedule,envelope.guard)
        current_bytes=prefinal_size_inventory(envelope)
        budget=clock_budget(envelope)
        c.save(c.OUT/'assessment_clock_budget.json',budget)
        inputs=assessment_inputs(measured,components,evidence['projected_final_evidence_bytes'],
            max(r['clock']['receipt_elapsed_seconds'] for r in allrows),current_bytes,current['projected_final_evidence_bytes'],budget['available'])
        c.save(c.OUT/'assessment_gate.json',dict(**operational_gate(**inputs),inputs=inputs,
            measurement_basis='Maximum observed gross game costs, explicit RAM waits separated once;400future games with1.5 contingency;406game final checks and independent acceptance' ,
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
    from . import schedule as schedules, qualification
    c.verify([freeze['schedule'],freeze['qualification_plan']])
    schedule=c.read(c.ROOT/freeze['schedule']['path'])
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(schedule==schedules.make_schedule(),'Frozen full406 schedule derivation changed')
    c.require(plan==qualification.build_plan(),'Frozen actual-entrypoint qualification inventory changed')
    c.require(len([x for x in schedule if x['stage']=='operational'])==6 and len([x for x in schedule if x['stage']=='main'])==400,'Fixed6+400 workload')
    c.require(freeze['qualification_decisions']==plan['complete_decisions'] and freeze['qualification_ordinary_requests']==len(plan['parity_requests']),'Bound actual qualification counts')
    return schedule,plan


def verify_prior_preservation(freeze,envelope):
    source=freeze['prior_preservation'];c.verify([source]);manifest=c.read(c.ROOT/source['path'])
    c.require(manifest['scope']=='finite_referenced_dependencies','Explicit bounded historical preservation scope')
    inputs={x['path']:x for x in freeze['inputs']};seen=set()
    for item in manifest['artifacts']:
        envelope.guard();c.require(item['path'] not in seen,'Duplicate historical dependency')
        seen.add(item['path']);c.require(inputs.get(item['path'])==item,'Historical dependency must be frozen')
        c.verify([item])
    for binding in manifest.get('historical_statuses',[]):
        envelope.guard();c.require(binding['path'] in seen,'Historical status receipt not pinned')
        value=c.read(c.ROOT/binding['path'])[binding['field']]
        c.require(value==binding['expected'],'Historical disposition changed')
    for relative in manifest.get('required_absent',[]):
        envelope.guard();c.require(not(c.ROOT/relative).exists(),'An earlier failed attempt was relabelled accepted')
    c.require(manifest.get('earlier_failures_preserved') is True,'Explicit preservation of earlier failures')
    result=dict(status='passed',prior_manifest=source,scope=manifest['scope'],
        prior_files=len(seen),prior_bytes=sum(x['bytes'] for x in manifest['artifacts']),
        historical_statuses_changed=False,entire_unrelated_historical_corpora_reaudited=False,
        mutation_of_previous_sources_or_evidence=False)
    c.save(c.OUT/'prior_preservation_after.json',result)
    return c.artifact(c.OUT/'prior_preservation_after.json')


def collect_assessment(envelope,freeze,schedule,games=None):
    games=[] if games is None else games
    for spec in schedule:
        game,rows,cost,fault=play_game(spec,envelope,freeze);games.append(game)
        print(c.canonical(dict(stage='main',completed_games=sum(g['stage']=='main' for g in games),total=400,fault=fault is not None)),flush=True)
        # A new scheduled game is never a retry of the failed move or game.
        # Integrity/resource exceptions propagate and stop the fixed attempt.
    return games


def final_checks(envelope,freeze,schedule,plan,games,stop,qualified):
    from . import audit, qualification, schedule as schedules
    c.require(schedule==schedules.make_schedule(),'Fixed schedule changed')
    qpaths=sorted((c.OUT/'qualification_rows').glob('*.json'));qrows=[c.read(p) for p in qpaths]
    expected_q=[] if plan is None else plan['parity_requests']
    c.require([r['spec'] for r in qrows]==expected_q[:len(qrows)] and len(qrows)<=len(expected_q),'Exact qualification prefix')
    c.require([p.name for p in qpaths]==[f"{s['index']:04d}.json" for s in expected_q[:len(qrows)]],'Qualification file inventory')
    if qualified is not None:
        c.require(qualified==c.read(c.OUT/'qualification_summary.json') and qualified['requests']==len(qrows) and
            qualified['timing_gate']==qualification_gate(qrows,expected_requests=len(expected_q)) and
            qualified['qualified']==qualified['timing_gate']['passed'],'Stored qualification summary and admission status')
    if qrows:
        c.require(all(r['completed'] for r in qrows[:-1]),'No work after failed qualification request')
        for row in qrows:
            envelope.guard();verify_saved_record(row,freeze)
            if row['completed']:
                identity=audit.profile_identity(freeze,c.ROUTE_QUOTAS[row['route']],3,route=row['route'],boundary=envelope.guard)
                audit.validate_result(row['clock']['result'],row['observer'],row['route'],identity=identity)
        parity=qualification.validate_parity_rows(qrows,plan,boundary=envelope.guard,allow_prefix=True)
        if len(qrows)==len(expected_q) and qrows[-1]['completed']:
            c.require(qualified and qualified['parity']==parity and qualified['timing_gate']==qualification_gate(qrows,expected_requests=len(expected_q)),
                      'Independent qualification summary/gate differs')
        else:
            failure=c.read(c.OUT/'qualification_stop.json')
            c.require(not qrows[-1]['completed'] and failure['clock']==qrows[-1]['clock'] and
                failure['index']==qrows[-1]['spec']['index'] and qualified['failure']==failure,'Final failed qualification binding')
    if games:c.require(qualified and qualified['qualified'],'Play forbidden after negative qualification')
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
    main_complete=coverage['complete_planned_workload']
    sample_rows=[r for r in records if r['stage']=='main'] if main_complete else records
    sample=audit.select_replay_sample(sample_rows,allow_operational=not main_complete)
    c.save(c.OUT/'replay_sample_manifest.json',sample);repeated=[]
    ops_schedule=[s for s in schedule if s['stage']=='operational']
    ops_records=[r for r in records if r['stage']=='operational']
    ops_games=[g for g in games if g['stage']=='operational']
    admission_gate=verified_assessment_gate(envelope,ops_schedule,ops_records) if len(ops_games)==6 and all(g['reason']=='natural' for g in ops_games) else None
    if any(g['stage']=='main' for g in games):c.require(admission_gate and admission_gate['passed'],'Main collection without verified resource admission')
    ensure_reproduction_resources(sample,records,envelope,admission_gate=admission_gate)
    for index,item in enumerate(sample['decisions']):
        original=next(r for r in records if all(r[k]==item[k] for k in ('stage','pair_index','game_index','ply','player_role')))
        repeated.append(reproduce_decision(original,index,envelope,freeze))
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(repeated),identities=sample['decisions']))
    worker_names={p.name for p in (c.OUT/'workers').iterdir()};expected_workers=set()
    if plan is not None:
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
    admission=bool(admission_gate and admission_gate['passed'])
    coverage.update(phase=c.PHASE,admission_passed=admission,operational_games=len(ops_games),main_games=sum(g['stage']=='main' for g in games),
                    qualification_ordinary_requests=len(qrows),live_decisions=len(records),reproductions=len(repeated))
    c.save(c.OUT/'coverage.json',coverage)
    seal=[]
    for sub in ('workers','games','qualification_rows','entrypoint_checks','repeated_decisions','closed_workers',
                'closed_worker_costs','placement_requests','placement_acks'):
        for path in sorted((c.OUT/sub).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    for name in ('qualification_summary.json','qualification_stop.json','collection_stop.json','assessment_gate.json',
                 'current_final_evidence_forecast.json','prefinal_size_inventory.json','reproduction_resource_gate.json','assessment_clock_budget.json',
                 'final_evidence_forecast.json','coverage.json','all_game_replay.json','all_player_faults.json','performance.json',
                 'boundary_analysis.json','replay_sample_manifest.json','repeated_decision_summary.json','all_closed_workers.json',
                 'all_placement_integrity.json'):
        if (c.OUT/name).exists():envelope.guard();seal.append(c.artifact(c.OUT/name))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope)
    preservation=verify_prior_preservation(freeze,envelope)
    if coverage['complete_planned_workload']:
        from . import analysis
        main_games=[g for g in games if g['stage']=='main']
        result=analysis.analyze_games(main_games,True,boundary=envelope.guard)
        checked=analysis.verify_analysis(main_games,result,boundary=envelope.guard)
        c.save(c.OUT/'analysis.json',result);c.save(c.OUT/'analysis_verification.json',checked)
        analysis_artifacts=[c.artifact(c.OUT/n) for n in ('analysis.json','analysis_verification.json')]
        c.save(c.OUT/'analysis_seal.json',dict(artifacts=analysis_artifacts))
        for item in analysis_artifacts:envelope.guard();c.verify([item])
    c.save(c.OUT/'verification_summary.json',dict(status='passed',coverage=coverage,workers=len(proofs),
        source_inputs=len(freeze['inputs']),parent_requests=sum(p['raw']['requests'] for p in proofs),
        graph_cases=11*len(proofs),peak_combined_rss_bytes=envelope.peak_combined_rss,prior_preservation=preservation,
        all_raw_placement_logs_graph_science_sources_verified=True,training=False))
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
            operational_schedule=[x for x in schedule if x['stage']=='operational']
            envelope.begin('operational');games,stop=operational(envelope,freeze,operational_schedule,games);envelope.finish()
            gate_path=c.OUT/'assessment_gate.json'
            if stop is None and gate_path.exists() and c.read(gate_path)['passed']:
                envelope.begin('main')
                games=collect_assessment(envelope,freeze,[x for x in schedule if x['stage']=='main'],games)
                envelope.finish()
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure,preserved_partial_evidence=True))
    if ready and not envelope.resident:
        try:
            envelope.begin('final');coverage=final_checks(envelope,freeze,schedule,plan,games,stop,qualified);envelope.finish()
            if qualified and qualified['failure'] and not qualified['failure']['diagnostic_deadline_failure']:
                failure=failure or 'Unclassified qualification failure; verified evidence is not scientific acceptance.'
            if failure is None:
                status='completed_verified' if coverage['complete_planned_workload'] else 'not_admitted_verified'
        except BaseException:
            final_error=traceback.format_exc();c.save(c.OUT/'finalization_failure.json',dict(time=c.now(),error=final_error))
            failure=(failure or '')+final_error
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,coverage=coverage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,phase=c.PHASE,automatic_retries=0,larger_data_deferred=True,training=False,
        requires_all_three_clean_actual_os_exits=True,independent_acceptance_pending=True,
        independent_acceptance_deadline_monotonic=envelope.deadline,independent_acceptance_deadline_unix=envelope.wall_deadline,
        independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,shared_job_origin=claim))
    return 1 if status=='incomplete' else 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run'])
    parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    return run(args.freeze_sha256)







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
    seconds=c.INDEPENDENT_REVIEW_SECONDS+1.5*(count*(components['initialization']+components['graph']+components['closure']+c.SECONDS_PER_MOVE+1.)+
        len(trees)*components['closure'])+2*components['runtime']+300.+remaining_wait
    if admission_gate is not None:
        seconds=max(seconds,admission_gate['assessment_projected_final_seconds'] if len([r for r in records if r['stage']=='main']) else admission_gate['projected_final_seconds'])
        final_bytes=max(final_bytes,admission_gate['assessment_projected_final_evidence_bytes'] if any(r['stage']=='main' for r in records) else admission_gate['projected_final_evidence_bytes'])
    current_bytes=bytes_used();remaining=envelope.remaining()
    passed=(admission_gate is None or admission_gate['current_resource_passed']) and seconds<remaining and         final_bytes<c.FINAL_RESERVE_BYTES and current_bytes+final_bytes<c.MAX_EVIDENCE_BYTES
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


def verified_assessment_gate(envelope,schedule,records):
    """Reconstruct prospective and current resource bounds, without performance selection."""
    gate=c.read(c.OUT/'assessment_gate.json')
    measured=measured_game_costs(schedule,envelope.guard);components=reproduction_fixed_components(envelope,admission_only=True)
    evidence=final_evidence_forecast(schedule,envelope.guard)
    current=final_evidence_forecast(schedule,envelope.guard,projected_games=6,maximum_workers=62,reserve_bytes=c.FINAL_RESERVE_BYTES)
    c.require(evidence==c.read(c.OUT/'final_evidence_forecast.json') and
        current==c.read(c.OUT/'current_final_evidence_forecast.json'),'Independent evidence forecast binding')
    inventory=c.read(c.OUT/'prefinal_size_inventory.json')
    for item in inventory['artifacts']:envelope.guard();c.verify([item])
    c.require(inventory['closed_bytes']==sum(x['bytes'] for x in inventory['artifacts']) and
        inventory['dynamic_control_reserve_bytes']==16*c.MiB,'Bound closed evidence census')
    size=inventory['closed_bytes']+(c.OUT/'prefinal_size_inventory.json').stat().st_size+16*c.MiB
    inputs=assessment_inputs(measured,components,evidence['projected_final_evidence_bytes'],
        max(r['clock']['receipt_elapsed_seconds'] for r in records),size,current['projected_final_evidence_bytes'],verify_clock_budget(c.read(c.OUT/'assessment_clock_budget.json'),envelope))
    c.require(gate['measured_game_costs']==measured and gate['reproduction_fixed_components']==components and
        gate['inputs']==inputs,'Prospective cost gate must bind every measured game')
    c.require(all(gate[k]==v for k,v in operational_gate(**inputs).items()),'Prospective resource arithmetic')
    return gate





def clock_budget(envelope):
    mono,wall=time.monotonic(),time.time()
    receipt=dict(monotonic=mono,unix_time=wall,monotonic_origin=envelope.start,unix_origin=envelope.wall_start,ram_wait_used_seconds=envelope.waited)
    receipt['available']=verify_clock_budget(receipt,envelope,checking_new=True)
    return receipt


def verify_clock_budget(value,envelope,*,checking_new=False):
    c.require(value['monotonic_origin']==envelope.start and value['unix_origin']==envelope.wall_start,'Shared forecast clock identity')
    mono,wall=value['monotonic'],value['unix_time']
    c.require(envelope.start<=mono<=time.monotonic() and envelope.wall_start<=wall<=time.time(),'Forecast clock within original run')
    used=value['ram_wait_used_seconds'];c.require(0<=used<=envelope.waited<=c.RAM_WAIT_SECONDS,'Forecast RAM wait budget')
    result=dict(available_collection_seconds=max(0.,min(envelope.start+c.COLLECTION_CUTOFF_SECONDS-mono,envelope.wall_start+c.COLLECTION_CUTOFF_SECONDS-wall)),
        available_global_seconds=max(0.,min(envelope.end-mono,envelope.wall_start+c.JOB_SECONDS-wall)),
        remaining_ram_wait_seconds=max(0.,c.RAM_WAIT_SECONDS-used))
    if not checking_new:c.require(value['available']==result,'Original forecast clock arithmetic changed')
    return result


if __name__=='__main__':raise SystemExit(main())
