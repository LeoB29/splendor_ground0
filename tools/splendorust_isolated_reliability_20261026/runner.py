"""Bounded live reliability workload with the qualified minimal observer.

Fresh game lifecycles preserve the exact players. The first attributable player
fault ends collection, retains its loss, and admits only verified-prefix analysis.
"""
import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from importlib.metadata import version
import psutil
from . import common as c, schedule as schedules, audit, adapters, placement
from tools.external_strength_screen_20261018 import rules
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit as scientific
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules

def make_worker(name,envelope,freeze,**kwargs):
    """Constrain the copied adapter's broader two-placement default."""
    c.require(kwargs.get('placement_mode','isolated')=='isolated','Only isolated worker supervision is authorized')
    kwargs['placement_mode']='isolated'
    return adapters.make_worker(name,envelope,freeze,**kwargs)


decision = adapters.decision
verify_graph_evidence = adapters.verify_graph_evidence
verify_raw_transport = adapters.verify_raw_transport

def require_player(mode,quota):
    c.require(type(quota) is int and ((mode=='baseline' and quota==32) or
        (mode=='candidate' and quota==64)), 'Only unchanged eager32 and qualified graph64 are authorized')

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
        self.watchdog_control_baseline = self.watchdog_control_bytes()
        self.log_baseline = {name: (c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                             for name in ('runner_stdout.log', 'runner_stderr.log')}
        c.WRITE_OBSERVER = self.note_write

    def watchdog_control_bytes(self):
        paths=list((c.OUT/'placement_acks').glob('*'))+list(c.OUT.glob('watchdog*.json'))
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

def runtime_placement_mode(freeze,envelope=None):
    c.require(freeze.get('placement_mode')=='isolated','Freeze must require isolated supervision')
    mode=getattr(envelope,'placement_mode',freeze['placement_mode'])
    c.require(mode=='isolated','Runtime or preparation supervisor must remain isolated')
    return mode


def verify_runtime(freeze, envelope=None):
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


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


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



def final_evidence_forecast(schedule,boundary=lambda:None):
    """Conservative byte projection from closed files, without reading outcomes.

    A one-decision reproduction is charged an entire observed game's compressed
    worker logs, plus its largest request/input and an additional full move row.
    Final control files are projected separately; none consume an assumed-free
    portion of the fixed final evidence reserve.
    """
    ops=[spec for spec in schedule if spec['stage']=='operational']
    c.require(len(ops)==6,'Exactly six operational games required for byte forecast')
    game_paths=[]; worker_paths=[]; worker_bounds=[]; replay_bytes=0
    log_receipt_bytes=[]; graph_receipt_bytes=[]; raw_receipt_bytes=[]
    largest_move=0
    for spec in ops:
        boundary(); folder=game_folder(spec)
        paths=[path for path in folder.rglob('*') if path.is_file()]
        game_paths.extend(paths)
        moves=list((folder/'moves').glob('*.json'))
        c.require(moves,'Operational move files absent from byte forecast')
        largest_move=max(largest_move,max(path.stat().st_size for path in moves))
        replay_bytes+=(folder/'online_replay.json').stat().st_size
        graph_receipt_bytes.append((folder/'graph_integrity_challenger.json').stat().st_size)
        for role in ('challenger','opponent'):
            log_receipt_bytes.append((folder/f'log_integrity_{role}.json').stat().st_size)
            raw_receipt_bytes.append((folder/f'raw_transport_{role}.json').stat().st_size)
            worker=c.OUT/'workers'/f"{spec['stage']}_g{spec['game_index']}_{role}"
            files=[path for path in worker.rglob('*') if path.is_file()]
            inputs=list((worker/'inputs').glob('*.json'))
            c.require(inputs,'Operational worker inputs absent from byte forecast')
            receipts=[worker/path.name for path in inputs]
            c.require(all(path.is_file() for path in receipts),'Operational receipt inventory missing')
            logs=[path for path in files if path.suffix=='.gz']
            c.require(logs,'Compressed operational logs absent from byte forecast')
            variable=set(inputs+receipts+logs)
            static=[path for path in files if path not in variable]
            worker_paths.extend(files)
            worker_bounds.append(dict(
                static_bytes=sum(path.stat().st_size for path in static),
                entire_game_compressed_log_bytes=sum(path.stat().st_size for path in logs),
                largest_input_bytes=max(path.stat().st_size for path in inputs),
                largest_receipt_bytes=max(path.stat().st_size for path in receipts),
                reproduction_worker_files=len(static)+len(logs)+2))
    qualification_paths=[path for folder in (c.OUT/'workers').glob('qualification*')
                         for path in folder.rglob('*') if path.is_file()]
    observed_paths=game_paths+worker_paths+qualification_paths
    c.require(observed_paths,'No measured files for final seal forecast')
    # Reserve 128 extra path bytes for longer final audit/control names. JSON
    # bytes are measured using the actual canonical artifact representation.
    seal_item_bytes=max(len(c.canonical(dict(path=path.relative_to(c.ROOT).as_posix(),
        sha256='0'*64,bytes=2**63-1)).encode())+129 for path in observed_paths)
    single_worker=max(sum(item[key] for key in ('static_bytes','entire_game_compressed_log_bytes',
        'largest_input_bytes','largest_receipt_bytes')) for item in worker_bounds)
    reproduction_files=max(item['reproduction_worker_files'] for item in worker_bounds)
    reproduction_bytes=48*(single_worker+largest_move)
    projected_sealed_files=math.ceil((len(game_paths)+len(worker_paths))*200/6)+len(qualification_paths)+48*reproduction_files
    seal_bytes=projected_sealed_files*seal_item_bytes+1024
    replay_control_bytes=math.ceil(replay_bytes*200/6)
    log_control_bytes=(402+48)*max(log_receipt_bytes)
    graph_control_bytes=(201+48)*max(graph_receipt_bytes)
    raw_transport_control_bytes=(402+48)*max(raw_receipt_bytes)
    fixed_control_bytes=8*c.MiB
    raw_final=(reproduction_bytes+seal_bytes+replay_control_bytes+log_control_bytes+
        graph_control_bytes+raw_transport_control_bytes+fixed_control_bytes)
    return dict(outcomes_used=False,conservative_factor=1.35,reproductions=48,
        observed_workers=len(worker_bounds),observed_game_files=len(game_paths),
        observed_worker_files=len(worker_paths),observed_worker_bounds=worker_bounds,
        largest_move_row_bytes=largest_move,single_reproduction_worker_bound_bytes=single_worker,
        projected_reproduction_bytes=reproduction_bytes,projected_sealed_files=projected_sealed_files,
        artifact_item_bound_bytes=seal_item_bytes,projected_seal_bytes=seal_bytes,
        projected_replay_control_bytes=replay_control_bytes,projected_log_control_bytes=log_control_bytes,
        projected_graph_control_bytes=graph_control_bytes,
        projected_raw_transport_control_bytes=raw_transport_control_bytes,
        fixed_analysis_and_control_bytes=fixed_control_bytes,
        projected_final_evidence_bytes=math.ceil(1.35*raw_final),reserve_bytes=c.FINAL_RESERVE_BYTES)


def operational_gate(*,observed_seconds,observed_variable_bytes,current_bytes,independent_seconds,
                     fixed_final_seconds,final_evidence_bytes,remaining_collection_seconds):
    values=(observed_seconds,observed_variable_bytes,current_bytes,independent_seconds,fixed_final_seconds,final_evidence_bytes)
    c.require(all(type(value) in (int,float) and math.isfinite(value) and value>=0 for value in values) and
              type(remaining_collection_seconds) in (int,float) and math.isfinite(remaining_collection_seconds),
              'Resource forecast requires finite measured quantities')
    factor=194/6*1.35
    final_factor=200/6*1.35
    collection=observed_seconds*factor
    evidence=current_bytes+observed_variable_bytes*factor
    final=independent_seconds*final_factor+fixed_final_seconds
    total_evidence=evidence+final_evidence_bytes
    passed=(collection<min(c.STAGES['main'],remaining_collection_seconds) and evidence<c.PREFINAL_EVIDENCE_BYTES and
            final<c.STAGES['final'] and final_evidence_bytes<c.FINAL_RESERVE_BYTES and total_evidence<c.MAX_EVIDENCE_BYTES)
    return dict(passed=passed,operational_games=6,remaining_games=194,total_games=200,observed_seconds=observed_seconds,
        observed_variable_bytes=observed_variable_bytes,projected_collection_seconds=collection,
        remaining_collection_seconds=min(c.STAGES['main'],remaining_collection_seconds),projected_prefinal_bytes=evidence,
        projected_final_seconds=final,arithmetic_factor=factor,final_arithmetic_factor=final_factor,
        projected_final_evidence_bytes=final_evidence_bytes,projected_total_evidence_bytes=total_evidence,
        final_evidence_reserve_bytes=c.FINAL_RESERVE_BYTES,total_evidence_cap_bytes=c.MAX_EVIDENCE_BYTES,outcomes_used=False)




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


def check_row_integrity(row):
    c.require('profile_capture_error' not in row and 'placement_error' not in row,
        'Saved observer/placement evidence error; infrastructure stop')


def verify_placement_request(folder,request_id,freeze,mode,parent_profile):
    proof=adapters.verify_worker_placement(folder,freeze,mode)
    value=c.read(folder/'placement_requests'/f'{request_id}.json')
    placement.verify_saved_request(value,freeze,mode,request_id=request_id,
        player_pid=proof['player_pid'],supervisor_identity=proof['supervisor'],parent_profile=parent_profile)
    launch=c.read(c.OUT/'launch.json')
    c.require((proof['supervisor']['pid'],proof['supervisor']['created'])==(launch['pid'],launch['created']),
        'Worker admission supervisor differs from owning runner')
    return value


def verify_saved_decision(row,freeze):
    require_player(row['mode'],row['quota']);check_row_integrity(row)
    c.require(row['placement_mode']=='isolated','Every collected decision must use isolated supervision')
    c.require(row['mode']==('candidate' if row['player_role']=='challenger' else 'baseline') and
        row.get('observation_mode')=='profiled','Saved role/backend/observer binding changed')
    path=c.ROOT/row['worker_clock_file']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()),'Clock outside owned evidence')
    c.require(c.read(path)==row['clock'] and path.name==row['clock']['request_id']+'.json','Trusted clock record changed')
    c.require(c.read(path.parent/'inputs'/path.name)==dict(player='search',observer=row['observer'],before=row['before']),
        'Saved player input changed')
    for field,expected in (('parent_profile_file',path.parent/'parent_profiles'/path.name),
        ('child_profile_file',path.parent/(row['clock']['request_id']+'_child_profile.json'))):
        value=row.get(field);profile=row.get(field.removesuffix('_file'))
        if value is None:c.require(profile is None and not row['completed'],'Successful timing sidecar missing')
        else:
            actual=c.ROOT/value
            c.require(actual.resolve()==expected.resolve() and c.read(actual)==profile,'Saved timing sidecar binding changed')
    c.require(verify_placement_request(path.parent,row['clock']['request_id'],freeze,row['placement_mode'],row['parent_profile'])==
        row['placement_evidence'],'Saved placement receipt differs from move row')
    audit.validate_boundary_row(row)
    scientific.check_clock(row['clock'])
    c.require(row['clock']['deadline_seconds']==1.,'Original full one-second decision required')
    if row['completed']:
        scientific.validate_result(row['clock']['result'],row['observer'],'search',expected_quota=row['quota'])
        c.require(row['action']==row['clock']['result']['action'] and row['trace']==row['clock']['result']['diagnostics'],
            'Scientific response and saved decision differ')
        scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',
            before=row['before'],after=row['after'],expected_quota=row['quota'])
    else:
        c.require(row['action'] is row['trace'] is row['after'] is None,'Failed request invents completed science')
        audit.classify_failure(row['clock'])


def verify_closed_worker(folder,freeze,envelope):
    # A deadline can close the worker inside the trusted request, before the
    # caller persists its placement/profile sidecars. Refresh that early cache.
    envelope.close_worker(folder)
    envelope.guard()
    logs=verify_worker_logs(folder,boundary=envelope.guard,deadline=envelope.effective_deadline())
    raw=verify_raw_transport(folder,freeze,envelope)
    metadata=c.read(folder/'initialization.json')['result'];quota=metadata['search_simulations']
    adapters.verify_instrumentation_identity(folder,freeze,quota,metadata)
    proof=c.read(folder/'placement_admission.json');mode=proof['placement_mode']
    c.require(mode=='isolated','Every worker must use isolated supervision')
    adapters.verify_worker_placement(folder,freeze,mode)
    requests=sorted((folder/'parent_profiles').glob('request-*.json'))
    placement_paths=sorted((folder/'placement_requests').glob('request-*.json'))
    c.require([p.name for p in requests]==[p.name for p in placement_paths],'Per-request placement inventory differs')
    for path in requests:
        envelope.guard();verify_placement_request(folder,path.stem,freeze,mode,c.read(path))
    graph=verify_graph_evidence(folder,freeze,boundary=envelope.guard) if quota==64 else None
    if quota==32:c.require('inference_backend' not in metadata and not any(folder.glob('graph_*.json')),'Eager32 backend changed')
    envelope.close_worker(folder)
    envelope.guard()
    return dict(logs=logs,raw=raw,graph=graph,placement_mode=mode,quota=quota,placement_requests=len(requests))


def qualify(envelope,freeze):
    """Two actual 52-decision suites, fully closed before live collection."""
    from .preparation import acceptance
    accepted_started=time.perf_counter()
    c.require(acceptance(boundary=envelope.guard)==freeze['qualification_acceptance'],
              'Accepted observer qualification changed')
    acceptance_seconds=time.perf_counter()-accepted_started
    costs=[]
    for quota in (32,64):
        mode='candidate' if quota==64 else 'baseline'
        begun=time.perf_counter()
        with make_worker(f'qualification_{quota}',envelope,freeze,mode=mode,quota=quota,placement_mode='isolated') as child:
            startup_seconds=time.perf_counter()-begun
            started=time.perf_counter()
            receipt=adapters.actual_entrypoint(child,envelope)
            c.save(c.OUT/f'entrypoint_checks_{quota}.json',receipt)
            c.require(receipt['status']=='ok' and receipt['result']['status']=='entrypoint_counterexamples_passed' and
                receipt['result']['challenger_quota']==quota and receipt['result']['counts']['complete_decisions']==52,
                'Actual corrected entrypoint counterexamples failed; no retry')
            check_seconds=time.perf_counter()-started
        started=time.perf_counter()
        proof=verify_closed_worker(child.output_dir,freeze,envelope)
        logs=proof['logs']
        c.save(c.OUT/f'qualification_log_integrity_{quota}.json',logs)
        transport=proof['raw']
        c.save(c.OUT/f'qualification_raw_transport_{quota}.json',transport)
        log_seconds=time.perf_counter()-started
        graph_seconds=0.
        if quota==64:
            graph=proof['graph']
            c.save(c.OUT/'qualification_graph_integrity.json',graph)
        hashing=measure_seal_cost([child.output_dir],envelope.guard)
        costs.append(dict(quota=quota,mode=mode,initialization_seconds=child.initialization['elapsed_seconds'],
            complete_startup_and_parity_seconds=startup_seconds,counterexample_seconds=check_seconds,
            log_and_raw_transport_seconds=log_seconds,graph_proof_reread_seconds=graph_seconds,
            combined_verification_includes_placement_and_graph=True,
            closed_worker_two_pass_seal_seconds=hashing['seconds']))
    started=time.perf_counter();verify_runtime(freeze,envelope);source_seconds=time.perf_counter()-started
    initialization=max(row['complete_startup_and_parity_seconds'] for row in costs)
    logs=max(row['log_and_raw_transport_seconds'] for row in costs)
    graphs=max(row['graph_proof_reread_seconds'] for row in costs)
    seals=max(row['closed_worker_two_pass_seal_seconds'] for row in costs)
    validation=max(row['counterexample_seconds']/52 for row in costs)
    fixed=1.35*(source_seconds+acceptance_seconds+2*logs+graphs+2*seals+
        48*(initialization+1.+validation+logs+graphs+seals))+30.
    result=dict(status='passed',actual_workers=2,complete_decisions=104,costs=costs,
        observer_overhead_requalification=False,source_seconds=source_seconds,
        acceptance_seconds=acceptance_seconds,
        initialization_seconds=initialization,validation_allowance_seconds=validation,
        final_reproduction_maximum=48,forecast_seconds=fixed,
        no_strength_statistics=True)
    c.save(c.OUT/'qualification_summary.json',result)
    return result


def captured_fault(spec,row):
    """The established classifier must authorize the loss before collection stops."""
    classification=audit.classify_failure(row['clock'])
    return dict(**{key:spec[key] for key in ('stage','pair_index','game_index')},
        ply=row['ply'],player_role=row['player_role'],seat=row['seat'],
        clock_status=row['clock']['status'],failure_classification=classification,
        request_id=row['clock']['request_id'],clock=row['clock'],
        worker_clock_file=row['worker_clock_file'],
        no_retry=True,collection_stopped=True,time=c.now())


def play_game(spec,envelope,freeze):
    schedules.validate_spec(spec)
    env=rules.CorrectedEnv(seed=int(spec['seed_decimal']))
    state=env.initial_state();initial=rules.serialize_snapshot(state);decode_training_snapshot(initial)
    folder=game_folder(spec);c.save(folder/'initial.json',dict(**spec,initial=initial))
    rows=[];outcome=None;stop=None;independent_seconds=0.;begun=time.perf_counter()
    with ExitStack() as stack:
        workers={role:stack.enter_context(make_worker(
            f"{spec['stage']}_g{spec['game_index']}_{role}",envelope,freeze,
            mode=spec[role+'_mode'],quota=spec[role+'_quota'],placement_mode='isolated')) for role in ('challenger','opponent')}
        for ply in range(2000):
            if state.terminal:break
            seat=state.current_player
            role='challenger' if seat==spec['challenger_seat'] else 'opponent'
            row=decision(workers[role],state,envelope,role)
            row.update(ply=ply,**{key:spec[key] for key in ('stage','pair_index','game_index')})
            c.save(folder/'moves'/f'{ply:04d}.json',row);rows.append(row)
            # Save before any integrity, budget or failure-classification check.
            envelope.guard()
            check_row_integrity(row)
            started=time.perf_counter()
            verify_saved_decision(row,freeze)
            independent_seconds+=time.perf_counter()-started
            if row['completed']:
                state=decode_training_snapshot(row['after'])
            else:
                stop=captured_fault(spec,row)
                outcome=dict(reason='forfeit',winner=1-seat,loser=seat,failed_role=role,failed_seat=seat,
                    failure=stop['failure_classification'])
                c.save(folder/'captured_fault.json',stop)
                c.save(c.OUT/'collection_stop.json',stop)
                break
            if not state.terminal and (row['after']==row['before'] or row['action']['action_type']=='PASS' and
                len(rows)>1 and rows[-2]['action'] and rows[-2]['action']['action_type']=='PASS'):
                raise ValueError('Unresolved actual-game deadlock; no official result')
        if outcome is None:
            c.require(state.terminal,'2,000-ply safety cap reached; incomplete')
            outcome=dict(reason='natural',winner=state.winner)
    # Both original worker trees must close before a game is eligible for audit.
    ending=dict(**spec,**outcome,final=rules.serialize_snapshot(state),complete=True,plies=len(rows),
        collection_seconds=time.perf_counter()-begun)
    c.save(folder/'result.json',ending)
    for role,child in workers.items():
        started=time.perf_counter()
        proof=verify_closed_worker(child.output_dir,freeze,envelope)
        check=proof['logs']
        c.save(folder/f'log_integrity_{role}.json',check)
        raw=proof['raw']
        c.save(folder/f'raw_transport_{role}.json',raw)
        if role=='challenger':
            graph=proof['graph']
            c.save(folder/'graph_integrity_challenger.json',graph)
        independent_seconds+=time.perf_counter()-started
    started=time.perf_counter()
    checked=audit.audit_game(initial,rows,ending,spec,boundary=envelope.guard)
    independent_seconds+=time.perf_counter()-started
    c.save(folder/'online_replay.json',checked)
    hashing=None
    if spec['stage']=='operational':
        hashing=measure_seal_cost([folder]+[child.output_dir for child in workers.values()],envelope.guard)
        independent_seconds+=hashing['seconds'];c.save(folder/'hashing_measurement.json',hashing)
    c.save(folder/'cost.json',dict(independent_seconds=independent_seconds,
        collection_seconds=time.perf_counter()-begun,complete_decisions=sum(row['completed'] for row in rows),
        measured_two_pass_hash_seconds=None if hashing is None else hashing['seconds']))
    return ending,rows,independent_seconds,stop


def operational(envelope,freeze,schedule,qualification):
    start=time.perf_counter();initial_bytes=bytes_used();independent=0.;games=[]
    for spec in (row for row in schedule if row['stage']=='operational'):
        game,rows,cost,stop=play_game(spec,envelope,freeze);games.append(game);independent+=cost
        print(c.canonical(dict(stage='operational',completed_games=len(games),total=6,
            captured_fault=stop is not None)),flush=True)
        if stop is not None:return games,stop
    c.require(len(games)==6,'Operational schedule incomplete')
    final_bytes=final_evidence_forecast(schedule,envelope.guard)
    c.save(c.OUT/'final_evidence_forecast.json',final_bytes);used=bytes_used()
    gate=operational_gate(observed_seconds=time.perf_counter()-start,observed_variable_bytes=used-initial_bytes,
        current_bytes=used,independent_seconds=independent,fixed_final_seconds=qualification['forecast_seconds'],
        final_evidence_bytes=final_bytes['projected_final_evidence_bytes'],
        remaining_collection_seconds=min(envelope.start+c.COLLECTION_CUTOFF_SECONDS-time.monotonic(),
            envelope.wall_start+c.COLLECTION_CUTOFF_SECONDS-time.time()))
    c.save(c.OUT/'operational_gate.json',gate)
    c.require(gate['passed'],'Outcome-blind operational resource forecast failed; no expansion')
    return games,None


def reproduce_decision(original,index,envelope,freeze):
    c.require(original['completed'],'Cannot replay an incomplete decision')
    require_player(original['mode'],original['quota'])
    with make_worker(f'audit_{index:02d}',envelope,freeze,mode=original['mode'],quota=original['quota'],placement_mode='isolated') as child:
        repeated=decision(child,decode_training_snapshot(original['before']),envelope,original['player_role'])
        c.save(c.OUT/'repeated_decisions'/f'{index:02d}.json',repeated)
        envelope.guard()
        check_row_integrity(repeated)
        c.require(repeated['completed'] and repeated['action']==original['action'] and repeated['trace']==original['trace'] and
            repeated['after']==original['after'] and repeated['clock']['result']==original['clock']['result'],
            'Sampled full scientific decision or transition reproduction differs')
        verify_saved_decision(repeated,freeze)
    return repeated


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


def final_checks(envelope,freeze,schedule,games,stop):
    c.require(schedule==schedules.make_schedule(),'Frozen schedule derivation changed')
    coverage=audit.validate_stopped_prefix(schedule,games,stop)
    for quota in (32,64):
        envelope.guard();receipt=c.read(c.OUT/f'entrypoint_checks_{quota}.json')
        folder=c.OUT/'workers'/f'qualification_{quota}'
        c.require(receipt['status']=='ok' and c.read(folder/f"{receipt['request_id']}.json")==receipt and
            receipt['result']['status']=='entrypoint_counterexamples_passed' and
            receipt['result']['challenger_quota']==quota and receipt['result']['counts']['complete_decisions']==52,
            'Preserved actual entrypoint receipt or inventory changed')
    records=[];main_records=[];replays=[]
    for spec,game in zip(schedule,games,strict=False):
        envelope.guard();folder=game_folder(spec)
        initial=c.read(folder/'initial.json')
        c.require(c.read(folder/'result.json')==game,'Closed game result changed')
        expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(initial==dict(**spec,initial=expected),'Initial world/identity changed')
        paths=sorted((folder/'moves').glob('*.json'))
        c.require([path.name for path in paths]==[f'{i:04d}.json' for i in range(game['plies'])],
            'Operational/main move inventory changed')
        rows=[c.read(path) for path in paths]
        for row in rows:envelope.guard();verify_saved_decision(row,freeze)
        replays.append(audit.audit_game(initial['initial'],rows,game,spec,boundary=envelope.guard))
        records.extend(rows)
        if spec['stage']=='main':main_records.extend(rows)
    actual_folders={path.parent.resolve() for path in (c.OUT/'games').rglob('initial.json')}
    c.require(actual_folders=={game_folder(spec).resolve() for spec in schedule[:len(games)]},
        'Unexpected game beyond exact collected prefix')
    if stop is not None:
        failed=records[-1]
        c.require(not failed['completed'] and stop==c.read(c.OUT/'collection_stop.json') and
            stop==c.read(game_folder(schedule[len(games)-1])/'captured_fault.json') and
            stop['clock']==failed['clock'] and stop['worker_clock_file']==failed['worker_clock_file'] and
            all(stop[key]==failed[key] for key in ('stage','pair_index','game_index','ply','player_role','seat')),
            'Captured fault differs from original failed raw receipt')
    else:c.require(not (c.OUT/'collection_stop.json').exists(),'Unreported collection stop')
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=len(games),receipts=replays))
    c.save(c.OUT/'performance.json',audit.performance(records,games))
    c.save(c.OUT/'boundary_analysis.json',audit.summarize_boundary_rows(records,boundary=envelope.guard))
    sample_rows=main_records if stop is None else records
    sample=audit.select_replay_sample(sample_rows,maximum=48,allow_operational=stop is not None)
    c.save(c.OUT/'replay_sample_manifest.json',sample);selected=sample['decisions'];repetitions=[]
    c.require(len(selected)<=48 and all(sum(item['player_role']==role for item in selected)<=24
        for role in ('challenger','opponent')),'Complete-decision reproduction bound')
    for index,item in enumerate(selected):
        envelope.guard()
        original=next(row for row in sample_rows if all(row[key]==item[key]
            for key in ('stage','pair_index','game_index','ply','player_role')))
        c.require(original['completed'] and original['mode']==item['mode'] and original['quota']==item['quota'],
            'Sampled completed decision identity changed')
        reproduce_decision(original,index,envelope,freeze);repetitions.append(item)
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(repetitions),identities=repetitions,
        prefix_only=stop is not None))
    logs=[];graph_checks=[];transports=[];placements=[]
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard()
        proof=verify_closed_worker(folder,freeze,envelope)
        logs.append(proof['logs']);transports.append(proof['raw'])
        placements.append(dict(worker=folder.name,placement_mode=proof['placement_mode'],requests=proof['placement_requests']))
        if proof['graph'] is not None:graph_checks.append(proof['graph'])
    expected_workers=2+2*len(games)+len(repetitions)
    c.require(len(logs)==expected_workers<=450,'Two workers per game plus qualification/audit inventory required')
    c.require(len(graph_checks)==1+len(games)+sum(item['mode']=='candidate' for item in selected) and len(graph_checks)<=249,
        'Graph instance inventory differs from qualification/game/sample roles')
    c.save(c.OUT/'all_worker_log_integrity.json',dict(status='passed',workers=len(logs),receipts=logs))
    c.save(c.OUT/'all_graph_parity.json',dict(status='passed',workers=len(graph_checks),receipts=graph_checks,
        independent_full_outputs_and_float32_bits=True))
    c.save(c.OUT/'all_raw_transport.json',dict(status='passed',workers=len(transports),receipts=transports))
    exchange=verify_exchanges(envelope,freeze)
    watchdog=c.read(c.OUT/'watchdog.json')
    placement.verify_saved_runtime(c.read(c.OUT/'watchdog_initial_placement.json'),freeze,'isolated',role='watchdog',
        identity=dict(pid=watchdog['watchdog_pid'],created=watchdog['watchdog_created']))
    c.save(c.OUT/'all_placement_integrity.json',dict(status='passed',workers=len(placements),receipts=placements,
        controller_exchanges=exchange,all_workers_isolated=True))
    seal=[]
    for subdir in ('games','workers','repeated_decisions','placement_requests','placement_acks'):
        for path in sorted((c.OUT/subdir).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope)
    from .preparation import acceptance
    c.require(acceptance(boundary=envelope.guard)==freeze['qualification_acceptance'],
        'Preceding observer qualification changed')
    c.save(c.OUT/'coverage.json',coverage)
    c.save(c.OUT/'verification_summary.json',dict(status='passed',games=len(games),requests=len(records),
        workers=len(logs),graph_workers=len(graph_checks),reproductions=len(repetitions),
        complete_planned_workload=stop is None,captured_fault=stop is not None,
        maximum_sampled_combined_worker_rss_bytes=envelope.peak_combined_rss,
        all_scientific_outputs_transitions_and_raw_receipts_verified=True,all_workers_and_supervisors_isolated=True,
        no_strength_claim=True))
    return coverage


def run_stages(envelope,freeze,schedule,*,qualification_started=False):
    if qualification_started:c.require(envelope.stage=='qualification','Startup qualification clock missing')
    else:envelope.begin('qualification')
    qualified=qualify(envelope,freeze);envelope.finish()
    envelope.begin('operational');games,stop=operational(envelope,freeze,schedule,qualified);envelope.finish()
    if stop is None:
        envelope.begin('main');main=[spec for spec in schedule if spec['stage']=='main']
        c.require(len(main)==194,'Fixed194-game expansion plus6 initial games required')
        for index,spec in enumerate(main):
            game,rows,cost,stop=play_game(spec,envelope,freeze);games.append(game)
            print(c.canonical(dict(stage='main',completed_games=index+1,total=194,captured_fault=stop is not None)),flush=True)
            if stop is not None:break
        envelope.finish()
    envelope.begin('final');result=final_checks(envelope,freeze,schedule,games,stop);envelope.finish()
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Reliability freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze identity differs')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json');envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status,failure,coverage='incomplete',None,None
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);envelope.admit();switch_placement(envelope,freeze,'isolated');schedule=c.read(c.OUT/'schedule.json')
        c.require(schedule==schedules.make_schedule(),'Frozen schedule changed')
        coverage=run_stages(envelope,freeze,schedule,qualification_started=True)
        status='captured_fault_verified' if coverage['status']=='captured_fault_verified_prefix' else 'completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure,
            preserved_partial_evidence=True,partial_evidence_not_accepted=True))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,automatic_retries=0,larger_data_deferred=True,coverage=coverage,
        challenger_quota=64,opponent_quota=32,main_games_maximum=194,total_games_maximum=200,no_strength_claim=True,
        note='Acceptance requires successful owning watchdog cleanup; captured-fault prefix is not complete planned workload.'))
    print(c.canonical(dict(status=status,stage=envelope.stage,failure=failure)),flush=True)
    return 0 if status in ('completed_verified','captured_fault_verified') else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','run']);parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.operation=='prepare':
        from .preparation import prepare
        return prepare()
    return run(args.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
