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
    return sum(path.stat().st_size for root in (c.OLD_OUT,c.OUT)
               for path in root.rglob('*') if path.is_file())


class Envelope:
    """One non-resettable job clock, aggregate resident-worker resource bound."""
    def __init__(self, claim):
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        c.require((self.start,self.wall_start)==(c.ORIGINAL_MONOTONIC,c.ORIGINAL_UNIX),
                  'Recovery cannot reset the original clocks')
        self.end = self.start + c.JOB_SECONDS
        self.stage = None
        self.deadline, self.wall_deadline = self.end, self.wall_start + c.JOB_SECONDS
        self.waited, self.admitted, self.admissions = c.PRIOR_RAM_WAIT_SECONDS, False, 0
        self.worker_accounting = {}
        self.prior_worker_count = len(list((c.OLD_OUT/'workers').iterdir()))
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
        # One frozen placement exchange plus fixed owner publications. Reserve
        # their declared capacity rather than stat a pending publication live.
        return 0

    def note_write(self, path, size):
        if not path.resolve().is_relative_to((c.OUT/'workers').resolve()):
            self.evidence['parent_bytes'] += size

    def register_worker(self, folder):
        from .accounting import WorkerAccounting
        c.require(self.prior_worker_count+len(self.worker_accounting)<c.MAX_WORKERS,
                  'Combined original/recovery worker bound')
        self.worker_accounting[folder] = WorkerAccounting(folder)
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
        receipt = self.worker_accounting[folder].close(boundary=self.time_guard,
            lock_timeout=min(1.,max(0.,self.remaining())))
        self.evidence['workers'][folder] = receipt['accounted_bytes']

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
        self.worker_accounting[child.output_dir].bind_child(child)
        self.aggregate_worker_rss(exclude_pid=child.process.pid, additional_rss=telemetry['rss_bytes'])

    def accounted_bytes(self):
        if self.evidence is None:
            return bytes_used()
        total = self.evidence['baseline'] + self.evidence['parent_bytes']
        for folder, closed in self.evidence['workers'].items():
            total += closed if closed is not None else self.worker_accounting[folder].snapshot(
                boundary=self.time_guard,lock_timeout=min(1.,max(0.,self.remaining())))['accounted_bytes']
        for name, initial in self.log_baseline.items():
            path = c.OUT/name
            total += max(0, (path.stat().st_size if path.exists() else 0)-initial)
        return total + 32*c.MiB + 65536

    def time_guard(self):
        c.require(self.remaining()>0,'Shared clock while accounting evidence')

    def begin(self, stage):
        mono, wall = time.monotonic(), time.time()
        self.stage = stage
        cap = c.STAGES[stage]
        if stage == 'final':
            cap -= c.PRIOR_FINAL_SECONDS
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












def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run'])
    parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    return run(args.freeze_sha256)


class OriginalEvidenceBoundary:
    """Read old proof functions without registering or writing original files."""
    def __init__(self,envelope):
        self.envelope=envelope
        self.start,self.wall_start,self.end=envelope.start,envelope.wall_start,envelope.end
        self.waited=envelope.waited
        self.placement_mode='isolated'
        self.placement_sequence=len(list((c.OLD_OUT/'placement_requests').glob('*.json')))
        self.source_runtime_seconds=c.read(c.OLD_OUT/'assessment_gate.json')['reproduction_fixed_components']['runtime']
    def guard(self):return self.envelope.guard()
    def effective_deadline(self):return self.envelope.effective_deadline()
    def close_worker(self,folder,child=None):
        c.require(Path(folder).resolve().is_relative_to((c.OLD_OUT/'workers').resolve()) and child is None,
                  'Old evidence is read-only; no live old worker')
    def remaining(self):return self.envelope.remaining()


def original_freeze():
    c.require(c.file_hash(c.OLD_OUT/'freeze.json')==c.OLD_FREEZE_SHA,'Original freeze unchanged')
    return c.read(c.OLD_OUT/'freeze.json')


def recovery_context(freeze,envelope):
    from . import recovery,schedule as schedules
    descriptor=freeze['recovery_manifest'];c.verify([descriptor]);manifest=c.read(c.ROOT/descriptor['path'])
    for key,value in dict(original_attempt_status='incomplete',completed_main_games=38,partial_game_index=38,
        continuation_ply=40,completed_prefix_plies=39,prior_final_seconds=c.PRIOR_FINAL_SECONDS,
        prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS).items():
        c.require(manifest[key]==value,'Prospective recovery boundary '+key)
    old=original_freeze();c.verify([manifest['old_freeze'],manifest['original_claim'],manifest['bridge']])
    c.require(manifest['old_freeze']==c.artifact(c.OLD_OUT/'freeze.json'),'Original source manifest')
    claim=c.read(c.ROOT/manifest['original_claim']['path'])
    c.require((claim['monotonic_origin'],claim['unix_origin'])==(c.ORIGINAL_MONOTONIC,c.ORIGINAL_UNIX),'Original clock provenance')
    saved=c.read(c.ROOT/manifest['bridge']['path']);proved=recovery.verify_bridge(saved['bridge'],old,boundary=envelope.guard)
    c.require(saved==proved,'Full saved bridge proof identity')
    schedule=c.read(c.ROOT/old['schedule']['path']);schedules.validate_schedule(schedule)
    c.require(c.read(c.ROOT/freeze['schedule']['path'])==schedule,'Recovery keeps every original scheduled world/seat')
    completed=schedule[:44]
    c.require(all((c.OLD_OUT/'games'/s['stage']/f"p{s['pair_index']:03d}_g{s['game_index']}"/'result.json').exists()
                  for s in completed),'All six operational and38 completed games retained')
    c.require(len(list((c.OLD_OUT/'games/main').glob('*/result.json')))==38 and
        len(list((c.OLD_OUT/'games/operational').glob('*/result.json')))==6,'No discarded or extra completed original games')
    game=c.OLD_OUT/'games/main/p019_g38'
    c.require(not(game/'result.json').exists() and sorted(p.name for p in (game/'moves').glob('*.json'))==
        [f'{i:04d}.json' for i in range(40)],'Exact interrupted original game')
    return dict(manifest=manifest,old_freeze=old,schedule=schedule,old_completed=completed,
        bridge=proved,partial_folder=game,prefix_rows=[c.read(game/'moves'/f'{i:04d}.json') for i in range(39)])


def verify_prior_preservation(freeze,envelope):
    descriptor=freeze['recovery_manifest'];c.verify([descriptor]);manifest=c.read(c.ROOT/descriptor['path'])
    artifacts=manifest['original_tree'];names={x['path'] for x in artifacts}
    c.require(len(names)==len(artifacts),'Unique original preservation inventory')
    actual={p.relative_to(c.ROOT).as_posix() for p in c.OLD_OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    c.require(actual==names,'Original incomplete evidence cannot gain, lose, or replace files')
    for item in artifacts:envelope.guard();c.verify([item])
    c.require(c.read(c.OLD_OUT/'completion.json')['status']=='incomplete','Original attempt remains incomplete')
    result=dict(status='passed',original_attempt_status='incomplete',original_files=len(artifacts),
        original_bytes=sum(x['bytes'] for x in artifacts),recovery_manifest=descriptor,
        original_rows_receipts_sources_rewritten=False,full_original_tree_unchanged=True)
    c.save(c.OUT/'prior_preservation_after.json',result)
    return c.artifact(c.OUT/'prior_preservation_after.json')


def verify_prior_qualification(context,envelope):
    from tools.splendorust_search128_strength_20261028 import runner as oldrunner,qualification as oldqualification,transport as oldtransport
    old=context['old_freeze'];plan=c.read(c.ROOT/old['qualification_plan']['path'])
    paths=sorted((c.OLD_OUT/'qualification_rows').glob('*.json'));rows=[c.read(p) for p in paths]
    c.require(len(rows)==84 and [p.name for p in paths]==[f'{i:04d}.json' for i in range(84)],'Original84 actual requests')
    for row in rows:envelope.guard();oldtransport.verify_saved_record(row,old)
    checked=oldqualification.validate_parity_rows(rows,plan,boundary=envelope.guard,
        frozen_entrypoint=c.read(c.OLD_OUT/'entrypoint_checks/frozen.json'))
    summary=c.read(c.OLD_OUT/'qualification_summary.json')
    c.require(summary['qualified'] is True and summary['parity']==checked and summary['failure'] is None,'Original qualification fully verified')
    suites=[]
    for route in ('frozen','research'):
        receipt=c.read(c.OLD_OUT/'entrypoint_checks'/f'{route}.json')
        suites.append(oldqualification.verify_entrypoint(c.OLD_OUT/'workers'/f'qualification_{route}',receipt,plan,old,boundary=envelope.guard))
    proxy=OriginalEvidenceBoundary(envelope)
    ops=[s for s in context['schedule'] if s['stage']=='operational'];records=[]
    for spec in ops:
        folder=c.OLD_OUT/'games/operational'/f"p{spec['pair_index']:03d}_g{spec['game_index']}"
        records.extend(c.read(p) for p in sorted((folder/'moves').glob('*.json')))
    gate=oldrunner.verified_assessment_gate(proxy,ops,records)
    c.require(gate['passed'] is True,'Original operational resource admission retained')
    result=dict(status='passed',ordinary_requests=84,scientific_decisions=203,original_suites=suites,
        old_freeze_sha256=c.OLD_FREEZE_SHA,prior_operational_gate=c.artifact(c.OLD_OUT/'assessment_gate.json'))
    save_or_verify(c.OUT/'prior_qualification_verification.json',result)
    return plan


def save_or_verify(path,value):
    if path.exists():c.require(c.read(path)==value,'Previously saved verification differs')
    else:c.save(path,value)


class SavedRows:
    """Reiterable immutable row references; no whole-study decoded-row cache."""
    def __init__(self, artifacts=(), *, boundary=lambda:None):
        self.artifacts=list(artifacts);self.boundary=boundary;self.identities={}
    def __len__(self):return len(self.artifacts)
    def __iter__(self):
        for item in self.artifacts:
            self.boundary();yield c.read(c.ROOT/item['path'])
    def add(self,item,row):
        key=tuple(row[k] for k in ('stage','pair_index','game_index','ply','player_role'))
        c.require(key not in self.identities,'Duplicate live row reference')
        self.artifacts.append(item);self.identities[key]=item
    def extend(self,other):
        c.require(not(self.identities.keys()&other.identities.keys()),'Duplicate live segment identity')
        self.artifacts.extend(other.artifacts);self.identities.update(other.identities)
    def find(self,item):
        key=tuple(item[k] for k in ('stage','pair_index','game_index','ply','player_role'))
        self.boundary();return c.read(c.ROOT/self.identities[key]['path'])
    @property
    def maximum_bytes(self):return max((a['bytes'] for a in self.artifacts),default=0)


def recovery_freeze_sha(context):
    if 'recovery_freeze_sha256' not in context:
        context['recovery_freeze_sha256']=c.file_hash(c.OUT/'freeze.json')
    return context['recovery_freeze_sha256']


def verify_live_row(row,context,freeze):
    from tools.splendorust_search128_strength_20261028 import runner as oldrunner
    folder=(c.ROOT/row['worker_folder']).resolve()
    if folder.is_relative_to((c.OLD_OUT/'workers').resolve()):return oldrunner.verify_saved_decision(row,context['old_freeze'])
    c.require(folder.is_relative_to((c.OUT/'workers').resolve()),'Unknown segment evidence root')
    return verify_saved_decision(row,freeze)


def audit_segmented_game(initial,events,ending,spec,context,freeze,boundary=lambda:None):
    """Replay the one saved-response bridge without making an accepted clock."""
    from . import audit
    current=deepcopy(initial);latency={r:[] for r in audit.ROLES};failure=None;bridges=0;transitions=0
    c.require(ending['complete'] is True and ending['plies']==len(events) and all(ending[k]==v for k,v in spec.items()),'Segmented ending inventory')
    for ply,event in enumerate(events):
        boundary();c.require(not current['state']['terminal'] and failure is None,'No event after ending')
        if event['kind']=='recovery_bridge':
            proved=context['bridge'];bridge=proved['bridge']
            c.require(spec['stage']=='main' and spec['game_index']==38 and ply==39 and bridges==0 and
                bridge['ply']==ply and current==proved['before'],'Unique exact recovery transition')
            ledger.check_observer(current,proved['source_row']['observer'])
            current=ledger.transition(current,proved['result']['action'])
            c.require(current==proved['after'],'Recovered saved action applied once')
            bridges+=1;transitions+=1;continue
        c.require(event['kind']=='ordinary','Known event kind')
        row=event['value'];verify_live_row(row,context,freeze)
        c.require(row['ply']==ply and row['before']==current and all(row[k]==spec[k] for k in ('stage','pair_index','game_index')),'Contiguous sourced ordinary move')
        seat=current['state']['current_player'];role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        c.require(row['seat']==seat and row['player_role']==role,'Segmented acting role')
        audit.validate_boundary_row(row)
        if row['completed']:
            current=ledger.transition(current,row['action']);c.require(current==row['after'],'Independent ordinary transition')
            latency[role].append(row['clock']['receipt_elapsed_seconds']);transitions+=1
        else:failure=dict(seat=seat,role=role,status=audit.classify_failure(row['clock']))
    c.require(bridges==1 and ending['final']==current,'One bridge and final state')
    if failure is None:
        c.require(ending['reason']=='natural' and current['state']['terminal'] and ending['winner']==current['state']['winner'] and
            not ending.get('administrative_forfeit',False) and ending.get('failure') is None,'Natural resumed ending')
    else:
        c.require(ending['reason']=='forfeit' and ending['administrative_forfeit'] is True and
            ending['failure_policy']=='research_hard_guard_9s_internal_10s_complete_response' and
            ending['failed_role']==failure['role'] and ending['failed_seat']==ending['loser']==failure['seat'] and
            ending['winner']==1-failure['seat'] and ending['failure']==failure['status'],'Resumed hard-guard loss')
    return dict(status='passed',transitions=transitions,failed_moves=int(failure is not None),
        final_sha256=c.digest(current),latencies=latency,recovered_transitions=1,
        original_failed_receipt_rewritten=False,recovered_move_in_ordinary_timing=False,
        transition_oracle='frozen independent dictionary ledger; explicit saved-response bridge')


def play_game(spec,envelope,freeze,*,context=None,workers=None):
    from . import audit,schedule
    schedule.validate_spec(spec);resume=context is not None
    initial=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
    state=decode_training_snapshot(context['bridge']['after'] if resume else initial)
    start_ply=40 if resume else 0
    folder=game_folder(spec);c.save(folder/'initial.json',dict(**spec,initial=initial))
    if resume:
        c.require(spec['stage']=='main' and spec['game_index']==38,'Only the declared interrupted game can resume')
        c.save(folder/'continuation.json',dict(start_ply=40,original_initial=c.artifact(context['partial_folder']/'initial.json'),
            original_failed_row=c.artifact(context['partial_folder']/'moves/0039.json'),
            bridge=c.artifact(c.OUT/'recovered_bridge.json'),initial_state_sha256=c.digest(rules.serialize_snapshot(state)),
            original_clock_restarted=False,failed_move_retried=False))
    workers={} if workers is None else workers;rows=[];outcome=None;fault=None;error=None;begun=time.perf_counter()
    try:
        for role in audit.ROLES:
            if role not in workers:workers[role]=make_worker(f"main_g{spec['game_index']}_{role}",envelope,freeze,route=spec[role+'_mode'])
        for ply in range(start_ply,2000):
            if state.terminal:break
            role='challenger' if state.current_player==spec['challenger_seat'] else 'opponent'
            row=decision(workers[role],state,envelope,role,freeze)
            row.update(ply=ply,**{k:spec[k] for k in ('stage','pair_index','game_index')})
            c.save(folder/'moves'/f'{ply:04d}.json',row);rows.append(row)
            c.require('transition_error' not in row,'Preserved transition failure')
            envelope.guard();_require_profiles(row);verify_saved_decision(row,freeze)
            if not row['completed']:
                if not deadline_failure(row['clock'],row['route']):
                    c.save(folder/'infrastructure_failure.json',dict(status='incomplete',row=c.artifact(folder/'moves'/f'{ply:04d}.json'),
                        original_clock=row['clock'],classified_as_player_loss=False,failed_move_retried=False))
                    raise ValueError('Preserved infrastructure failure; not an administrative player forfeit')
                fault=captured_fault(spec,row)
                outcome=dict(reason='forfeit',winner=1-row['seat'],loser=row['seat'],failed_role=role,failed_seat=row['seat'],
                    failure=fault['failure_classification'],administrative_forfeit=True,failure_policy='research_hard_guard_9s_internal_10s_complete_response')
                c.save(folder/'captured_fault.json',fault);break
            state=decode_training_snapshot(row['after'])
            previous_action=(rows[-2]['action'] if len(rows)>1 else context['bridge']['result']['action'] if resume else None)
            c.require(state.terminal or (row['after']!=row['before'] and not(row['action']['action_type']=='PASS' and
                previous_action and previous_action['action_type']=='PASS')),'Unresolved deadlock remains incomplete')
        if outcome is None:
            c.require(state.terminal,'2,000-ply cap; no result');outcome=dict(reason='natural',winner=state.winner)
        ending=dict(**spec,**outcome,final=rules.serialize_snapshot(state),complete=True,plies=start_ply+len(rows),
            collection_seconds=time.perf_counter()-begun,collection_seconds_scope='new_recovery_segment' if resume else 'whole_new_game')
        c.save(folder/'result.json',ending)
    except BaseException:
        error=traceback.format_exc()
        if not(folder/'infrastructure_failure.json').exists():
            c.save(folder/'infrastructure_failure.json',dict(status='incomplete',error=error,
                last_saved_row=None if not rows else c.artifact(folder/'moves'/f"{rows[-1]['ply']:04d}.json"),
                classified_as_player_loss=False,failed_move_retried=False))
        raise
    finally:
        errors=[]
        for role,child in workers.items():
            try:c.save(folder/f'closed_{role}.json',close_and_audit(child,envelope,freeze))
            except BaseException:errors.append(dict(role=role,error=traceback.format_exc()))
        if errors:
            c.save(folder/'closure_failures.json',dict(original_error=error,errors=errors))
            if error is None:raise ValueError('Closed-worker proof failed')
    if resume:
        events=[dict(kind='ordinary',value=r) for r in context['prefix_rows']]+[dict(kind='recovery_bridge')]+[dict(kind='ordinary',value=r) for r in rows]
        checked=audit_segmented_game(initial,events,ending,spec,context,freeze,envelope.guard)
    else:checked=audit.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
    c.save(folder/'online_replay.json',checked)
    c.save(folder/'cost.json',dict(collection_seconds=time.perf_counter()-begun,new_ordinary_requests=len(rows),
        recovered_saved_actions=int(resume),scope='new segment only; original wall time remains charged by shared origin'))
    return ending


def recovery_resource_gate(context,envelope):
    """Reuse outcome-blind measured costs, charge all original evidence/time."""
    old=c.read(c.OLD_OUT/'assessment_gate.json');measured=old['measured_game_costs'];remaining_games=362
    c.require(old['passed'] is True and len(measured)==6,'Original measured admission gate')
    remaining_wait=max(0.,c.RAM_WAIT_SECONDS-envelope.waited)
    collection=1.5*remaining_games*max(x['nonwait_collection_seconds'] for x in measured)+600.+2*envelope.source_runtime_seconds+remaining_wait
    final=old['assessment_projected_final_seconds']+300.+2*envelope.source_runtime_seconds
    current=envelope.accounted_bytes()
    growth=1.5*remaining_games*max(x['bytes'] for x in measured)+64*c.MiB
    final_bytes=old['assessment_projected_final_evidence_bytes']+64*c.MiB
    available_collection=min(envelope.start+c.COLLECTION_CUTOFF_SECONDS-time.monotonic(),envelope.wall_start+c.COLLECTION_CUTOFF_SECONDS-time.time())
    available_global=min(envelope.end-time.monotonic(),envelope.wall_start+c.JOB_SECONDS-time.time())
    passed=collection<available_collection and collection+final<available_global and final<c.STAGES['final']-c.PRIOR_FINAL_SECONDS and \
        current+growth<c.PREFINAL_EVIDENCE_BYTES and final_bytes<c.FINAL_RESERVE_BYTES and current+growth+final_bytes<c.MAX_EVIDENCE_BYTES
    value=dict(passed=passed,remaining_scheduled_games=remaining_games,projected_collection_seconds=collection,
        projected_final_seconds=final,available_collection_seconds=available_collection,available_global_seconds=available_global,
        aggregate_current_evidence_bytes=current,projected_additional_collection_bytes=growth,projected_final_bytes=final_bytes,
        prior_final_seconds_charged=c.PRIOR_FINAL_SECONDS,prior_ram_wait_allocation_seconds=c.PRIOR_RAM_WAIT_SECONDS,
        remaining_ram_wait_seconds=remaining_wait,original_gate=c.artifact(c.OLD_OUT/'assessment_gate.json'),
        source_runtime_seconds=envelope.source_runtime_seconds,outcomes_used=False,move_timing_subtracted_seconds=0,
        original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start,
        maximum_combined_workers=c.MAX_WORKERS,qualification_extra_scientific_decisions=119)
    c.save(c.OUT/'recovery_resource_gate.json',value)
    c.require(passed,'Remaining work does not fit the original shared resource envelope')
    return value


def game_manifest_entry(spec,context,freeze,envelope):
    from . import audit
    from tools.splendorust_search128_strength_20261028 import audit as oldaudit
    old=spec['stage']=='operational' or spec['game_index']<38
    base=c.OLD_OUT if old else c.OUT
    folder=base/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"
    initial=c.read(folder/'initial.json');ending=c.read(folder/'result.json')
    expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
    c.require(initial==dict(**spec,initial=expected),'Frozen seeded initial state')
    paths=sorted((folder/'moves').glob('*.json'));rows=[c.read(p) for p in paths]
    resumed=spec['stage']=='main' and spec['game_index']==38
    start=40 if resumed else 0
    c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(start,ending['plies'])],'Exact per-segment move filenames')
    events=[];ordinary=SavedRows(boundary=envelope.guard)
    if resumed:
        for i,row in enumerate(context['prefix_rows']):
            path=context['partial_folder']/'moves'/f'{i:04d}.json'
            item=c.artifact(path)
            events.append(dict(kind='ordinary',row=item,source_freeze_sha256=c.OLD_FREEZE_SHA));ordinary.add(item,row)
        events.append(dict(kind='recovery_bridge',bridge=c.artifact(c.OUT/'recovered_bridge.json'),source_freeze_sha256=c.OLD_FREEZE_SHA))
    for path,row in zip(paths,rows,strict=True):
        envelope.guard();verify_live_row(row,context,freeze);item=c.artifact(path);ordinary.add(item,row)
        events.append(dict(kind='ordinary',row=item,source_freeze_sha256=c.OLD_FREEZE_SHA if old else recovery_freeze_sha(context)))
    if resumed:
        values=[dict(kind='ordinary',value=c.read(c.ROOT/e['row']['path'])) if e['kind']=='ordinary' else dict(kind='recovery_bridge') for e in events]
        checked=audit_segmented_game(expected,values,ending,spec,context,freeze,envelope.guard)
    elif old:checked=oldaudit.audit_game(expected,rows,ending,spec,freeze=context['old_freeze'],boundary=envelope.guard)
    else:checked=audit.audit_game(expected,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
    c.require(checked==c.read(folder/'online_replay.json'),'Independently replayed ledger proof')
    if ending['reason']=='forfeit':
        saved=c.read(folder/'captured_fault.json');again=captured_fault(spec,rows[-1])
        c.require({k:v for k,v in saved.items() if k!='time'}=={k:v for k,v in again.items() if k!='time'},'Retained original forfeit')
    else:c.require(not(folder/'captured_fault.json').exists(),'No unreported game fault')
    return dict(spec=spec,initial=c.artifact(folder/'initial.json'),result=c.artifact(folder/'result.json'),
        online_replay=c.artifact(folder/'online_replay.json'),events=events),ending,ordinary


def ensure_reproduction_resources(sample,records,envelope,admission_gate=None):
    count=len(sample['decisions']);c.require(0<=count<=48,'Frozen reproduction bound')
    folders=[p for root in (c.OLD_OUT,c.OUT) for p in (root/'workers').iterdir() if not p.name.startswith('audit_')]
    c.require(folders,'Measured closed workers required')
    max_init=max(c.read(p/'initialization.json')['elapsed_seconds'] for p in folders)
    max_graph=max(c.read(p/'graph_parity_clock.json')['elapsed_seconds'] for p in folders)
    costs=[c.read(p) for root in (c.OLD_OUT,c.OUT) for p in (root/'closed_worker_costs').glob('*.json')]
    closure=max(x['seconds'] for x in costs);tree_bytes=max(x['bytes'] for x in costs)
    files=sum(1 for root in (c.OLD_OUT,c.OUT) for p in root.rglob('*') if p.is_file())
    row_bytes=records.maximum_bytes
    # Measured final seal descriptors, exact replay trees and explicit review reserve.
    predicted_bytes=math.ceil(1.5*(count*(tree_bytes+row_bytes)+files*2048+64*c.MiB))
    predicted_seconds=c.INDEPENDENT_REVIEW_SECONDS+1.5*(count*(max_init+max_graph+closure+11.)+len(folders)*closure)+2*envelope.source_runtime_seconds+300.+max(0.,c.RAM_WAIT_SECONDS-envelope.waited)
    current=bytes_used();remaining=envelope.remaining()
    passed=predicted_seconds<remaining and predicted_bytes<c.FINAL_RESERVE_BYTES and current+predicted_bytes<c.MAX_EVIDENCE_BYTES
    gate=dict(passed=passed,selected_reproductions=count,closed_workers=len(folders),projected_remaining_seconds=predicted_seconds,
        available_seconds=remaining,current_evidence_bytes=current,projected_final_bytes=predicted_bytes,
        prior_final_seconds_charged=c.PRIOR_FINAL_SECONDS,independent_review_reserved_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        maximum_measured_tree_bytes=tree_bytes,source_runtime_seconds=envelope.source_runtime_seconds)
    c.save(c.OUT/'reproduction_resource_gate.json',gate)
    c.require(passed,'Final reproductions cannot fit original remaining resources; no inference started')
    return gate


def final_checks(envelope,freeze,context):
    from . import audit,analysis,qualification,recovery
    from tools.splendorust_search128_strength_20261028 import runner as oldrunner,transport as oldtransport
    recovery.verify_bridge(c.read(c.OUT/'recovered_bridge.json')['bridge'],context['old_freeze'],boundary=envelope.guard)
    plan=verify_prior_qualification(context,envelope)
    oldrunner.verify_saved_decision(context['bridge']['source_row'],context['old_freeze'])
    entries=[];games=[];records=SavedRows(boundary=envelope.guard)
    for spec in context['schedule']:
        envelope.guard();entry,game,rows=game_manifest_entry(spec,context,freeze,envelope)
        entries.append(entry);games.append(game);records.extend(rows)
    c.require(len(entries)==406 and sum(e['kind']=='recovery_bridge' for g in entries for e in g['events'])==1,'Complete406 games and one adopted action')
    actual={p.parent.resolve() for p in (c.OUT/'games').rglob('initial.json')}
    expected={game_folder(s).resolve() for s in context['schedule'] if s['stage']=='main' and s['game_index']>=38}
    c.require(actual==expected,'No additional new game or replacement world')
    c.save(c.OUT/'game_evidence_manifest.json',dict(schema='search128_segmented_games_v1',games=entries,
        original_attempt_status='incomplete',recovered_bridge_events=1,original_failed_request_not_retried=True))
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=406,receipts=[e['online_replay'] for e in entries],recovered_transitions=1))
    faults=[c.read((c.ROOT/e['result']['path']).parent/'captured_fault.json') for e,g in zip(entries,games,strict=True) if g['reason']=='forfeit']
    c.save(c.OUT/'all_player_faults.json',dict(count=len(faults),faults=faults,failed_move_retries=0,original_infrastructure_failure_preserved=True))
    c.save(c.OUT/'performance.json',audit.performance(records,games))
    c.save(c.OUT/'boundary_analysis.json',audit.summarize_boundary_rows(records,boundary=envelope.guard))
    c.save(c.OUT/'recovery_timing.json',dict(ordinary_metrics_exclude_bridge=True,excluded_original_clock=context['bridge']['source_row']['clock'],
        recovered_raw_arrival_evidence=context['bridge']['bridge']['timing_evidence'],source_receipt_not_rewritten=True,
        ordinary_timed_live_requests=len(records),logical_live_events=len(records)+1))
    sample=audit.select_replay_sample(r for r in records if r['stage']=='main')
    c.save(c.OUT/'replay_sample_manifest.json',sample);ensure_reproduction_resources(sample,records,envelope)
    repeated=[]
    for i,item in enumerate(sample['decisions']):
        row=records.find(item)
        repeated.append(reproduce_decision(row,i,envelope,freeze))
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(repeated),identities=sample['decisions']))
    oldfolders={p.name:p for p in (c.OLD_OUT/'workers').iterdir()};newfolders={p.name:p for p in (c.OUT/'workers').iterdir()}
    expected_old={'qualification_frozen','qualification_research'}|{f"{s['stage']}_g{s['game_index']}_{r}" for s in context['schedule'][:45] for r in audit.ROLES}
    expected_new={f"main_g{i}_{r}" for i in range(38,400) for r in audit.ROLES}|{f'audit_{i:02d}' for i in range(len(repeated))}
    c.require(set(oldfolders)==expected_old and set(newfolders)==expected_new and len(oldfolders)+len(newfolders)<=c.MAX_WORKERS,'Complete combined864 tree inventory')
    qrows=[c.read(p) for p in sorted((c.OLD_OUT/'qualification_rows').glob('*.json'))]
    represented={p.resolve():[] for p in list(oldfolders.values())+list(newfolders.values())}
    from itertools import chain
    for row in chain(qrows,records,repeated,[context['bridge']['source_row']]):represented[(c.ROOT/row['worker_folder']).resolve()].append(row['clock']['request_id'])
    workers=[];proofs=[];proxy=OriginalEvidenceBoundary(envelope)
    recovered_suites=[]
    for old,folders in ((True,oldfolders),(False,newfolders)):
        for name,folder in sorted(folders.items()):
            envelope.guard();ids=represented[folder.resolve()]
            c.require(len(ids)==len(set(ids)) and set(ids)=={p.stem for p in (folder/'inputs').glob('*.json')},'Every actual request represented exactly once')
            proof=oldtransport.verify_closed_worker(folder,context['old_freeze'],proxy) if old else verify_closed_worker(folder,freeze,envelope)
            proofpath=(c.OLD_OUT if old else c.OUT)/'closed_workers'/f'{name}.json'
            c.require(c.read(proofpath)==proof,'Closed raw/graph/log/placement proof unchanged')
            if not old and name in ('main_g38_challenger','main_g38_opponent'):
                route='research' if name.endswith('challenger') else 'frozen';receipt=c.read(c.OUT/'recovery_entrypoint_checks'/f'{route}.json')
                c.require(c.read(folder/(receipt['request_id']+'.json'))==receipt,'New actual entrypoint raw binding')
                checked=qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=envelope.guard)
                recovered_suites.append(dict(route=route,worker_folder=folder.relative_to(c.ROOT).as_posix(),
                    clock=c.artifact(c.OUT/'recovery_entrypoint_checks'/f'{route}.json'),verification=checked))
            proofs.append(proof);workers.append(dict(folder=folder.relative_to(c.ROOT).as_posix(),
                source_freeze_sha256=c.OLD_FREEZE_SHA if old else recovery_freeze_sha(context),closed_proof=c.artifact(proofpath),ordinary_request_ids=ids))
    c.save(c.OUT/'all_closed_workers.json',dict(status='passed',workers=len(proofs),proofs=proofs))
    summary=c.read(c.OUT/'recovery_qualification_summary.json')
    c.require(summary==dict(status='passed',scientific_decisions=119,parent_requests=2,
        receipts=sorted(recovered_suites,key=lambda x:('research','frozen').index(x['route'])),
        original_scientific_decisions=203,combined_scientific_decisions=322,
        same_workers_retained_for_continuation=True,failed_game_decision_retried=False),'Recovery actual-entrypoint summary readback')
    exchanges=dict(status='passed',original=oldrunner.verify_exchanges(proxy,context['old_freeze']),recovery=verify_exchanges(envelope,freeze))
    c.save(c.OUT/'all_placement_integrity.json',exchanges)
    c.save(c.OUT/'segments_manifest.json',dict(schema='search128_recovery_segments_v1',original_freeze=c.artifact(c.OLD_OUT/'freeze.json'),
        recovery_freeze=c.artifact(c.OUT/'freeze.json'),workers=workers,
        original_failed_request=c.artifact(context['partial_folder']/'moves/0039.json'),bridge=c.artifact(c.OUT/'recovered_bridge.json'),
        ordinary_timing_excludes_bridge=True,original_attempt_status='incomplete',failed_decision_retries=0))
    coverage=dict(status='amended_recovered_full_workload_verified',complete_planned_workload=True,operational_games=6,main_games=400,
        planned_games=406,verified_games=406,unplayed_games=0,forfeits=len(faults),
        live_game_events=len(records)+1,logical_live_events=len(records)+1,live_decisions=len(records),recovered_events=1,recovered_transitions=1,reproductions=len(repeated),
        qualification_ordinary_requests=84,qualification_scientific_decisions=322,recovery_actual_entrypoint_decisions=119,
        original_infrastructure_failures_preserved=1,failed_decision_retries=0,original_attempt_status='incomplete')
    c.save(c.OUT/'coverage.json',coverage)
    result=analysis.analyze_games([g for g in games if g['stage']=='main'],True,boundary=envelope.guard)
    c.save(c.OUT/'analysis.json',result);c.save(c.OUT/'analysis_verification.json',analysis.verify_analysis([g for g in games if g['stage']=='main'],result,boundary=envelope.guard))
    c.save(c.OUT/'analysis_seal.json',dict(artifacts=[c.artifact(c.OUT/'analysis.json'),c.artifact(c.OUT/'analysis_verification.json')]))
    verify_runtime(freeze,envelope);preservation=verify_prior_preservation(freeze,envelope)
    seal=[]
    for sub in ('workers','games','closed_workers','closed_worker_costs','placement_requests','placement_acks','recovery_entrypoint_checks','repeated_decisions'):
        for path in sorted((c.OUT/sub).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    for name in ('game_evidence_manifest.json','segments_manifest.json','recovered_bridge.json','prior_qualification_verification.json',
        'recovery_qualification_summary.json','recovery_stage_transition.json','recovery_resource_gate.json','reproduction_resource_gate.json','all_game_replay.json',
        'all_player_faults.json','performance.json','boundary_analysis.json','recovery_timing.json','replay_sample_manifest.json',
        'repeated_decision_summary.json','all_closed_workers.json','all_placement_integrity.json','coverage.json','analysis.json','analysis_verification.json','analysis_seal.json','prior_preservation_after.json'):
        seal.append(c.artifact(c.OUT/name))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal,original_tree_manifest=freeze['recovery_manifest']))
    for item in seal:envelope.guard();c.verify([item])
    c.save(c.OUT/'verification_summary.json',dict(status='passed',coverage=coverage,workers=len(proofs),source_inputs=len(freeze['inputs']),
        parent_requests=sum(p['raw']['requests'] for p in proofs),graph_cases=11*len(proofs),
        peak_combined_rss_bytes=envelope.peak_combined_rss,prior_preservation=preservation,
        all_raw_placement_logs_graph_science_sources_verified=True,original_attempt_accepted=False,
        acceptance_scope='amended recovered benchmark only',training=False))
    return coverage


def qualify_resumed_workers(envelope,freeze,plan,workers):
    """Actual119 checks on the same two trees subsequently used for ply40."""
    from . import adapters,qualification,audit
    receipts=[]
    for role in audit.ROLES:
        route=audit.MODES[role]
        child=make_worker(f'main_g38_{role}',envelope,freeze,route=route)
        workers[role]=child
        receipt=adapters.actual_entrypoint(child,envelope,freeze,request_fn=request,collect_fn=collect_profiles)
        path=c.OUT/'recovery_entrypoint_checks'/f'{route}.json';c.save(path,receipt)
        checked=qualification.verify_entrypoint(child.output_dir,receipt,plan,freeze,boundary=envelope.guard)
        receipts.append(dict(route=route,worker_folder=child.output_dir.relative_to(c.ROOT).as_posix(),
            clock=c.artifact(path),verification=checked))
    result=dict(status='passed',scientific_decisions=119,parent_requests=2,receipts=receipts,
        original_scientific_decisions=203,combined_scientific_decisions=322,
        same_workers_retained_for_continuation=True,failed_game_decision_retried=False)
    c.save(c.OUT/'recovery_qualification_summary.json',result)
    return result


def rebind_retained_workers(workers,envelope):
    """Stage allowance update only while both trusted request locks are idle."""
    before=[]
    for role,child in workers.items():
        c.require(not child._closed and child.recovery_accounting.state=='idle' and
            not child._lock.locked(),'Retained worker must be idle outside every original request')
        before.append(dict(role=role,worker=child.output_dir.name,request_counter=child._counter,
            previous_absolute_deadline=child.absolute_deadline))
    deadline=envelope.effective_deadline()
    for child in workers.values():child.absolute_deadline=deadline
    c.save(c.OUT/'recovery_stage_transition.json',dict(status='passed',workers=before,
        new_absolute_deadline=deadline,stage=envelope.stage,original_monotonic_origin=envelope.start,
        original_unix_origin=envelope.wall_start,ordinary_clocks_reset=False))


def incomplete_checks(envelope,freeze,primary_failure):
    """Best-effort saved-prefix proofs never upgrade a failed recovery."""
    checks=[];errors=[]
    for folder in sorted((c.OUT/'workers').glob('*')):
        if not folder.is_dir():continue
        try:
            envelope.guard();proof=verify_closed_worker(folder,freeze,envelope)
            save_or_verify(c.OUT/'closed_workers'/f'{folder.name}.json',proof)
            checks.append(dict(worker=folder.name,status='passed'))
        except BaseException:errors.append(dict(worker=folder.name,error=traceback.format_exc()))
    for name,function in (('runtime',lambda:verify_runtime(freeze,envelope)),
                          ('original_preservation',lambda:verify_prior_preservation(freeze,envelope))):
        try:envelope.guard();function();checks.append(dict(component=name,status='passed'))
        except BaseException:errors.append(dict(component=name,error=traceback.format_exc()))
    value=dict(status='incomplete',primary_failure_preserved=primary_failure,checks=checks,
        remaining_verification_errors=errors,full_workload_accepted=False,original_attempt_accepted=False,
        new_model_calls_for_finalization=0,failed_decision_retries=0)
    c.save(c.OUT/'incomplete_verification.json',value)
    try:
        artifacts=[]
        for sub in ('workers','games','closed_workers','recovery_entrypoint_checks'):
            for path in sorted((c.OUT/sub).rglob('*')):
                if path.is_file():envelope.guard();artifacts.append(c.artifact(path))
        c.save(c.OUT/'incomplete_evidence_seal.json',dict(artifacts=artifacts,acceptance=False))
    except BaseException:
        c.save(c.OUT/'incomplete_seal_failure.json',dict(error=traceback.format_exc(),primary_failure_unchanged=True))
    return value


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Recovery execution freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze binding')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json')
    envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        freeze_sha256=expected_sha,phase=c.PHASE,original_freeze_sha256=c.OLD_FREEZE_SHA))
    status,failure,coverage='incomplete',None,None;workers={};context=None
    try:
        envelope.begin('recovery_qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        begun=time.perf_counter();verify_runtime(freeze,envelope);envelope.source_runtime_seconds=time.perf_counter()-begun
        context=recovery_context(freeze,envelope);context['recovery_freeze_sha256']=expected_sha
        c.save(c.OUT/'recovered_bridge.json',context['bridge'])
        plan=verify_prior_qualification(context,envelope)
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        qualify_resumed_workers(envelope,freeze,plan,workers)
        print(json.dumps(dict(event='recovery_qualification_passed',scientific_decisions=119,continuation_ply=40)),flush=True)
        envelope.finish();envelope.begin('main');rebind_retained_workers(workers,envelope)
        recovery_resource_gate(context,envelope)
        spec=next(s for s in context['schedule'] if s['stage']=='main' and s['game_index']==38)
        try:game=play_game(spec,envelope,freeze,context=context,workers=workers)
        finally:workers={}
        print(json.dumps(dict(event='main_game_completed',completed_games=39,total_games=400,forfeit=game['reason']=='forfeit')),flush=True)
        for spec in context['schedule']:
            if spec['stage']=='main' and spec['game_index']>38:
                game=play_game(spec,envelope,freeze)
                print(json.dumps(dict(event='main_game_completed',completed_games=spec['game_index']+1,total_games=400,forfeit=game['reason']=='forfeit')),flush=True)
        envelope.finish();envelope.begin('final')
        print(json.dumps(dict(event='final_verification_started',completed_games=400,total_games=400)),flush=True)
        coverage=final_checks(envelope,freeze,context);envelope.finish()
        status='amended_recovered_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure,
            original_failure_retained=True,preserved_partial_evidence=True,classified_as_player_loss=False))
        cleanup_errors=[]
        for role,child in workers.items():
            try:
                if not child._closed:child.close()
                envelope.close_worker(child.output_dir,child)
            except BaseException:cleanup_errors.append(dict(role=role,error=traceback.format_exc()))
        if cleanup_errors:c.save(c.OUT/'failure_cleanup.json',dict(errors=cleanup_errors,primary_failure_unchanged=True))
        try:
            if envelope.stage!='final':envelope.begin('final')
            incomplete_checks(envelope,freeze,failure)
        except BaseException:
            c.save(c.OUT/'finalization_failure.json',dict(time=c.now(),error=traceback.format_exc(),primary_failure_unchanged=True))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,coverage=coverage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,phase=c.PHASE,automatic_retries=0,failed_decision_retries=0,
        original_attempt_status='incomplete',original_attempt_accepted=False,amended_recovered_benchmark=True,
        prior_final_seconds_charged=c.PRIOR_FINAL_SECONDS,prior_ram_wait_allocation_seconds=c.PRIOR_RAM_WAIT_SECONDS,
        larger_data_deferred=True,training=False,requires_all_three_clean_actual_os_exits=True,
        independent_acceptance_pending=True,independent_acceptance_deadline_monotonic=envelope.deadline,
        independent_acceptance_deadline_unix=envelope.wall_deadline,
        independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,shared_job_origin=claim))
    return 0 if status=='amended_recovered_verified' else 1


if __name__=='__main__':raise SystemExit(main())




















