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
from . import common as c, game_audit as audit, adapters, placement, panel, assessment
from . import preparation as schedules
from tools.external_strength_screen_20261018 import rules
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit as scientific
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules

make_worker = adapters.make_worker
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

def verify_runtime(freeze, envelope=None):
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python runtime changed')
    placement.verify_host(freeze)
    placement.verify_runtime(freeze, getattr(envelope, 'placement_mode', 'original'))
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





def switch_placement(envelope,freeze,mode):
    """Switch both owned supervisors only with no resident player trees."""
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
    from .preparation import acceptance
    c.require(acceptance(boundary=envelope.guard)==freeze['qualification_acceptance'],'Accepted observer qualification changed')
    rows=[]
    for mode in ('original','isolated'):
        switch_placement(envelope,freeze,mode)
        for quota in (64,32):
            begun=time.perf_counter()
            with make_worker(f'qualification_{mode}_{quota}',envelope,freeze,
                mode='candidate' if quota==64 else 'baseline',quota=quota,placement_mode=mode) as child:
                receipt=adapters.actual_entrypoint(child,envelope)
                c.save(c.OUT/f'entrypoint_checks_{mode}_{quota}.json',receipt)
            checks=verify_closed_worker(child.output_dir,freeze,envelope)
            c.save(c.OUT/f'qualification_integrity_{mode}_{quota}.json',checks)
            rows.append(dict(placement_mode=mode,quota=quota,seconds=time.perf_counter()-begun,
                complete_decisions=receipt['result']['counts']['complete_decisions']))
    c.require(sum(r['complete_decisions'] for r in rows)==208,'Four actual52-decision suites required')
    result=dict(status='passed',workers=4,actual_decisions=208,suites=rows)
    c.save(c.OUT/'qualification_summary.json',result);return result


def validate_timing_rows(rows,plan,*,boundary=lambda:None,allow_prefix=False):
    """Delegate exact science/coverage to the independently reviewed assessment."""
    return assessment.validate_timing_rows(rows,plan,boundary=boundary,allow_prefix=allow_prefix)


def timing_failure_record(row):
    """Bind a disclosed timing outcome to its unchanged request and scheduled index."""
    return dict(index=row['index'],fixture_id=row['fixture_id'],quota=row['quota'],
        placement_mode=row['placement_mode'],clock=row['clock'],
        classification=audit.classify_failure(row['clock']),worker_clock_file=row['worker_clock_file'],
        timing_worker_name=row['timing_worker_name'],timing_segment=row['timing_segment'],no_retry=True)


def collect_timing(envelope,freeze,plan):
    rows=[];stop=None;controls=[];segments=[]
    for block in range(8):
        specs=[s for s in plan['schedule'] if s['block_index']==block]
        first=specs[0];mode=first['placement_mode'];quota=first['quota']
        switch_placement(envelope,freeze,mode)
        cursor=0;segment=0
        while cursor<len(specs):
            name=f'timing_block_{block}_segment_{segment:02d}'
            started=cursor;failed=None;started_ns=time.perf_counter_ns()
            with make_worker(name,envelope,freeze,mode='candidate' if quota==64 else 'baseline',
                quota=quota,placement_mode=mode) as child:
                while cursor<len(specs):
                    spec=specs[cursor];fixture=plan['fixtures'][spec['fixture_index']]
                    row=decision(child,decode_training_snapshot(fixture['snapshot']),envelope,spec['player_role'])
                    row.update(spec);row.update(stage='timing',timing_worker_name=name,timing_segment=segment)
                    c.save(c.OUT/'timing_rows'/f"{spec['index']:04d}.json",row);rows.append(row);cursor+=1
                    # Original receipt/sidecars exist before any evidence check or adjudication.
                    envelope.guard();check_row_integrity(row);verify_saved_decision(row,freeze)
                    if not row['completed']:
                        failed=timing_failure_record(row)
                        if mode=='original':
                            c.require(assessment.control_timeout(row)==failed['classification'],
                                'Only an established original-control deadline permits continuation')
                            c.save(c.OUT/'control_timeouts'/f"{spec['index']:04d}.json",failed)
                            controls.append(failed)
                        else:
                            stop=failed;c.save(c.OUT/'timing_stop.json',stop)
                        break
            # Close and independently audit before constructing a continuation worker.
            proof=verify_closed_worker(child.output_dir,freeze,envelope)
            receipt=dict(block_index=block,segment=segment,worker_name=name,quota=quota,placement_mode=mode,
                scheduled_indices=[s['index'] for s in specs[started:cursor]],
                failed_index=None if failed is None else failed['index'],
                next_unattempted_index=specs[cursor]['index'] if cursor<len(specs) and stop is None else None,
                termination='isolated_failure' if stop is not None else 'control_timeout' if failed is not None else 'block_complete',
                started_ns=started_ns,closed_audit_completed_ns=time.perf_counter_ns(),
                closed_and_audited_before_continuation=True,integrity=proof)
            c.save(c.OUT/'timing_worker_segments'/f'{block}_{segment:02d}.json',receipt);segments.append(receipt)
            if stop is not None:break
            segment+=1
        print(c.canonical(dict(stage='timing',attempted=len(rows),total=192,control_timeouts=len(controls),
            stopped=stop is not None)),flush=True)
        if stop is not None:break
    validation=validate_timing_rows(rows,plan,boundary=envelope.guard,allow_prefix=stop is not None)
    gate=panel.timing_gate(rows)
    c.save(c.OUT/'control_timeout_manifest.json',dict(status='preserved',count=len(controls),failures=controls,
        continuation_workers=sum(s['segment']>0 for s in segments),failed_indices_retried=False,
        successful_control_timings_after_restart_are_descriptive=True))
    c.save(c.OUT/'timing_validation.json',validation);c.save(c.OUT/'timing_gate.json',gate)
    return rows,gate,stop


def verify_timing_lifecycle(rows,plan,freeze,envelope):
    """Reconstruct every initial/continuation worker and each failed control."""
    controls=[];expected=[];expected_paths=[];seen_names=set();previous_closed=None
    for block in sorted({row['block_index'] for row in rows}):
        block_rows=[row for row in rows if row['block_index']==block]
        cursor=0;segment=0
        while cursor<len(block_rows):
            start=cursor;name=f'timing_block_{block}_segment_{segment:02d}';first=block_rows[cursor]
            failed=None
            while cursor<len(block_rows):
                row=block_rows[cursor]
                c.require(row['timing_worker_name']==name and row['timing_segment']==segment,
                    'Timing continuation worker or segment differs from first-failure boundaries')
                folder=c.ROOT/row['worker_clock_file']
                c.require(folder.parent.resolve()==(c.OUT/'workers'/name).resolve(),
                    'Timing scheduled request bound to wrong worker')
                cursor+=1
                if not row['completed']:
                    failed=timing_failure_record(row)
                    if row['placement_mode']=='original':
                        assessment.control_timeout(row);controls.append(failed)
                    else:c.require(row is rows[-1],'Isolated failure did not end timing collection')
                    break
            spec=plan['schedule'][block_rows[cursor-1]['index']]
            next_index=(spec['index']+1 if spec['ordinal']<23 and failed is not None and
                failed['placement_mode']=='original' else None)
            selected=block_rows[start:cursor]
            path=c.OUT/'timing_worker_segments'/f'{block}_{segment:02d}.json'
            receipt=c.read(path);expected_paths.append(path.name)
            checks=dict(block_index=block,segment=segment,worker_name=name,quota=first['quota'],
                placement_mode=first['placement_mode'],scheduled_indices=[x['index'] for x in selected],
                failed_index=None if failed is None else failed['index'],next_unattempted_index=next_index,
                termination='isolated_failure' if failed and failed['placement_mode']=='isolated' else
                    'control_timeout' if failed else 'block_complete',closed_and_audited_before_continuation=True)
            c.require(all(receipt.get(k)==v for k,v in checks.items()),'Saved timing lifecycle differs')
            c.require(receipt['integrity']['quota']==first['quota'] and receipt['integrity']['placement_mode']==first['placement_mode'],
                'Segment audit role differs')
            folder=c.OUT/'workers'/name
            begun,closed=receipt['started_ns'],receipt['closed_audit_completed_ns']
            c.require(type(begun) is int and type(closed) is int and 0<begun<closed and
                (previous_closed is None or previous_closed<=begun),'Continuation starts before prior worker audit ends')
            admitted=c.read(folder/'placement_admission.json')['supervisor']['monotonic_ns']
            c.require(begun<=admitted<=closed and all(begun<=row['parent_profile']['request_start_ns']<=
                row['parent_profile']['capture_snapshot_ns']<=closed for row in selected),
                'Segment chronology does not enclose worker admission and original request clocks')
            previous_closed=closed
            inputs=sorted((folder/'inputs').glob('request-*.json'))
            c.require([p.name for p in inputs]==[Path(x['worker_clock_file']).name for x in selected],
                'Timing worker attempted an omitted, duplicate or retried request')
            expected.append(checks);seen_names.add(name);segment+=1
    c.require(sorted(expected_paths)==sorted(p.name for p in (c.OUT/'timing_worker_segments').glob('*.json')),
        'Timing segment manifest inventory differs')
    c.require(seen_names=={p.name for p in (c.OUT/'workers').glob('timing_block_*')},'Unexpected timing continuation worker')
    paths=sorted((c.OUT/'control_timeouts').glob('*.json'))
    c.require([p.name for p in paths]==[f"{x['index']:04d}.json" for x in controls] and
        [c.read(p) for p in paths]==controls,'Original-control timeout inventory changed')
    manifest=dict(status='preserved',count=len(controls),failures=controls,
        continuation_workers=sum(x['segment']>0 for x in expected),failed_indices_retried=False,
        successful_control_timings_after_restart_are_descriptive=True)
    c.require(c.read(c.OUT/'control_timeout_manifest.json')==manifest,'Control timeout summary differs')
    c.require(len(expected)<=100 and manifest['continuation_workers']<=92,'Prospective timing worker continuation bound')
    return dict(status='passed',workers=len(expected),graph_workers=sum(x['quota']==64 for x in expected),
        continuation_workers=manifest['continuation_workers'],control_timeouts=len(controls),segments=expected)


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
            verify_saved_decision(row,freeze)
            if row['completed']:
                started=time.perf_counter()
                scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',
                    before=row['before'],after=row['after'],expected_quota=row['quota'])
                audit.validate_boundary_row(row)
                independent_seconds+=time.perf_counter()-started
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
        envelope.close_worker(child.output_dir)
        envelope.guard()
        started=time.perf_counter()
        check=verify_worker_logs(child.output_dir,boundary=envelope.guard,deadline=envelope.effective_deadline())
        c.save(folder/f'log_integrity_{role}.json',check)
        raw=verify_raw_transport(child.output_dir,freeze,envelope)
        c.save(folder/f'raw_transport_{role}.json',raw)
        if role=='challenger':
            graph=verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)
            c.save(folder/'graph_integrity_challenger.json',graph)
        envelope.close_worker(child.output_dir)
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




def reproduce_decision(original,index,envelope,freeze):
    c.require(original['completed'],'Cannot replay an incomplete decision')
    with make_worker(f'audit_{index:02d}',envelope,freeze,mode=original['mode'],quota=original['quota'],placement_mode='isolated') as child:
        repeated=decision(child,decode_training_snapshot(original['before']),envelope,original['player_role'])
        c.save(c.OUT/'repeated_decisions'/f'{index:02d}.json',repeated)
        envelope.guard();check_row_integrity(repeated)
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


def final_checks(envelope,freeze,plan,timing_rows,gate,timing_stop,schedule,games,stop):
    c.require(schedule==schedules.operational_schedule(),'Frozen six-game schedule derivation changed')
    timing_paths=sorted((c.OUT/'timing_rows').glob('*.json'))
    c.require([p.name for p in timing_paths]==[f'{i:04d}.json' for i in range(len(timing_rows))] and
        [c.read(p) for p in timing_paths]==timing_rows,'Saved timing prefix inventory changed')
    for row in timing_rows:envelope.guard();verify_saved_decision(row,freeze)
    validation=validate_timing_rows(timing_rows,plan,boundary=envelope.guard,allow_prefix=timing_stop is not None)
    c.require(validation==c.read(c.OUT/'timing_validation.json'),'Timing independent validation differs')
    lifecycle=verify_timing_lifecycle(timing_rows,plan,freeze,envelope)
    c.save(c.OUT/'timing_lifecycle_verification.json',lifecycle)
    c.require(gate==panel.timing_gate(timing_rows),'Timing isolated/relative assessment changed')
    if timing_stop is None:
        c.require(gate==panel.timing_gate(timing_rows) and not (c.OUT/'timing_stop.json').exists(),'Timing gate changed')
    else:
        last=timing_rows[-1]
        c.require(timing_stop==c.read(c.OUT/'timing_stop.json') and timing_stop['clock']==last['clock'] and
            timing_stop['worker_clock_file']==last['worker_clock_file'] and not last['completed'],'Timing fault raw binding differs')
    c.require(gate==c.read(c.OUT/'timing_gate.json'),'Saved timing gate differs')
    for mode in ('original','isolated'):
        for quota in (64,32):
            envelope.guard();receipt=c.read(c.OUT/f'entrypoint_checks_{mode}_{quota}.json')
            folder=c.OUT/'workers'/f'qualification_{mode}_{quota}'
            c.require(receipt['status']=='ok' and c.read(folder/f"{receipt['request_id']}.json")==receipt and
                receipt['result']['status']=='entrypoint_counterexamples_passed' and
                receipt['result']['challenger_quota']==quota and receipt['result']['counts']['complete_decisions']==52,
                'Four preserved actual entrypoint receipts changed')
    records=[];replays=[]
    if gate['qualified']:
        coverage=audit.validate_stopped_prefix(schedule,games,stop)
    else:
        c.require(not games and stop is None,'Operational games cannot follow timing nonpass')
        coverage=dict(status='timing_failure_verified' if timing_stop else 'timing_gate_negative_verified',
            planned_games=6,verified_games=0,unplayed_games=6,complete_planned_workload=False,no_strength_claim=True)
    for spec,game in zip(schedule,games,strict=False):
        envelope.guard();folder=game_folder(spec);initial=c.read(folder/'initial.json')
        expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(initial==dict(**spec,initial=expected) and c.read(folder/'result.json')==game,'Saved initial/result identity changed')
        paths=sorted((folder/'moves').glob('*.json'))
        c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(game['plies'])],'Live move inventory changed')
        rows=[c.read(p) for p in paths]
        for row in rows:
            envelope.guard();c.require(row['placement_mode']=='isolated','Operational placement differs');verify_saved_decision(row,freeze)
        replays.append(audit.audit_game(initial['initial'],rows,game,spec,boundary=envelope.guard));records.extend(rows)
    folders={p.parent.resolve() for p in (c.OUT/'games').rglob('initial.json')}
    c.require(folders=={game_folder(spec).resolve() for spec in schedule[:len(games)]},'Unexpected live game beyond prefix')
    if stop is not None:
        failed=records[-1]
        c.require(not failed['completed'] and stop==c.read(c.OUT/'collection_stop.json') and
            stop==c.read(game_folder(schedule[len(games)-1])/'captured_fault.json') and stop['clock']==failed['clock'] and
            stop['worker_clock_file']==failed['worker_clock_file'] and
            all(stop[k]==failed[k] for k in ('stage','pair_index','game_index','ply','player_role','seat')),'Live fault raw binding differs')
    else:c.require(not (c.OUT/'collection_stop.json').exists(),'Unreported live collection stop')
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=len(games),receipts=replays))
    c.save(c.OUT/'performance.json',audit.performance(records,games))
    c.save(c.OUT/'boundary_analysis.json',audit.summarize_boundary_rows(records,boundary=envelope.guard))
    sample=audit.select_replay_sample(records,maximum=48,allow_operational=True)
    c.save(c.OUT/'replay_sample_manifest.json',sample);selected=sample['decisions']
    c.require(len(selected)<=48 and all(sum(x['player_role']==r for x in selected)<=24 for r in ('challenger','opponent')),
        'Stratified completed reproduction inventory')
    if selected:switch_placement(envelope,freeze,'isolated')
    for index,item in enumerate(selected):
        original=next(r for r in records if all(r[k]==item[k] for k in ('stage','pair_index','game_index','ply','player_role')))
        reproduce_decision(original,index,envelope,freeze)
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(selected),identities=selected,
        no_failed_decision_retries=True,selection_scope='completed operational cells only'))
    workers=[];graphs=0
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard();proof=verify_closed_worker(folder,freeze,envelope);workers.append(proof);graphs+=proof['graph'] is not None
    expected=4+lifecycle['workers']+2*len(games)+len(selected)
    expected_graphs=2+lifecycle['graph_workers']+len(games)+sum(x['quota']==64 for x in selected)
    c.require(len(workers)==expected<=164 and graphs==expected_graphs<=108,'Exact worker/graph inventory differs')
    c.save(c.OUT/'all_worker_integrity.json',dict(status='passed',workers=len(workers),receipts=workers))
    c.save(c.OUT/'all_placement_integrity.json',verify_exchanges(envelope,freeze))
    seal=[]
    for subdir in ('games','workers','timing_rows','timing_worker_segments','control_timeouts',
                   'repeated_decisions','placement_requests','placement_acks'):
        for path in sorted((c.OUT/subdir).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope)
    c.require(schedules.acceptance(boundary=envelope.guard)==freeze['qualification_acceptance'] and
        schedules.history_acceptance()==freeze['history_acceptance'] and
        schedules.previous_attempt_acceptance(boundary=envelope.guard)==freeze['previous_attempt_acceptance'],
        'Accepted prior evidence changed')
    operational_margin=(len(games)==6 and stop is None and all(r['completed'] and r['clock']['receipt_elapsed_seconds']<.7 for r in records))
    qualified=gate['qualified'] and operational_margin
    coverage.update(separation_qualified=qualified,isolated_qualified=qualified,operational_margin_passed=operational_margin,
        timing_complete=validation['complete'],timing_gate_passed=gate['qualified'],causal_proof=False,
        timing_control_timeouts=lifecycle['control_timeouts'],timing_continuation_workers=lifecycle['continuation_workers'],
        timing_all_attempted_decisions_completed=all(row['completed'] for row in timing_rows),
        original_control_failures_preserved=True)
    c.save(c.OUT/'coverage.json',coverage)
    c.save(c.OUT/'verification_summary.json',dict(status='passed',games=len(games),timing_requests=len(timing_rows),
        live_requests=len(records),workers=len(workers),graph_workers=graphs,reproductions=len(selected),
        separation_qualified=qualified,isolated_qualified=qualified,control_timeouts=lifecycle['control_timeouts'],
        timing_continuation_workers=lifecycle['continuation_workers'],
        maximum_sampled_combined_worker_rss_bytes=envelope.peak_combined_rss,
        all_scientific_outputs_transitions_raw_receipts_placement_verified=True,no_strength_claim=True))
    return coverage


def run_stages(envelope,freeze,plan,schedule,*,qualification_started=False):
    if qualification_started:c.require(envelope.stage=='qualification','Startup qualification clock missing')
    else:envelope.begin('qualification')
    qualify(envelope,freeze);envelope.finish()
    envelope.begin('timing');rows,gate,timing_stop=collect_timing(envelope,freeze,plan);envelope.finish()
    games=[];stop=None
    if gate['qualified']:
        envelope.begin('operational');switch_placement(envelope,freeze,'isolated')
        for spec in schedule:
            game,played,cost,stop=play_game(spec,envelope,freeze);games.append(game)
            print(c.canonical(dict(stage='operational',completed_games=len(games),total=6,stopped=stop is not None)),flush=True)
            if stop is not None:break
        envelope.finish()
    envelope.begin('final')
    result=final_checks(envelope,freeze,plan,rows,gate,timing_stop,schedule,games,stop);envelope.finish()
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Supervisor-separation freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze identity differs')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json');envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status,failure,coverage='incomplete',None,None
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);envelope.admit()
        plan=c.read(c.OUT/'panel.json');panel.validate_plan(plan)
        schedule=schedules.validate_schedule(c.read(c.OUT/'schedule.json'))
        coverage=run_stages(envelope,freeze,plan,schedule,qualification_started=True)
        if coverage['isolated_qualified']:status='isolated_qualified_verified'
        elif coverage['status']=='captured_fault_verified_prefix':status='captured_fault_verified'
        elif coverage['status'].startswith('timing_'):status=coverage['status']
        else:status='operational_margin_negative_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure,preserved_partial_evidence=True,
            partial_evidence_not_accepted=True))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,automatic_retries=0,larger_data_deferred=True,coverage=coverage,
        challenger_quota=64,opponent_quota=32,operational_games_maximum=6,no_strength_claim=True,
        note='Acceptance requires successful owning watchdog cleanup; no placement causality proof.'))
    print(c.canonical(dict(status=status,stage=envelope.stage,failure=failure)),flush=True)
    return 1 if status=='incomplete' else 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','run']);parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.operation=='prepare':return schedules.prepare()
    return run(args.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
