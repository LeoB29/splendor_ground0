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
from . import common as c, schedule as schedules, verification as audit, clock as adapters, placement
from tools.external_strength_screen_20261018 import rules
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit as scientific
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules

make_worker=adapters.make_worker
decision=adapters.decision
verify_saved_decision=audit.verify_saved_decision
verify_closed_worker=audit.verify_closed_worker
check_row_integrity=audit.check_row_integrity

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
            c.require(self.waited+current-begun < 600, 'Cumulative RAM wait exceeded ten minutes')
            if stable is not None and current-stable >= 60:
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
    from . import numerical_runtime
    numerical_runtime.verify_runtime(freeze['numpy_blas_runtime'],require_one_thread=True)


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
            player='breck' if role=='challenger' else 'control32')) for role in ('challenger','opponent')}
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
    gate.update(operational_700ms_is_descriptive_only=True,resource_forecast_passed=gate['passed'])
    c.save(c.OUT/'operational_gate.json',gate)
    return games,None


def reproduce_decision(original,index,envelope,freeze):
    c.require(original['completed'],'Cannot replay an incomplete decision')
    with make_worker(f'audit_{index:02d}',envelope,freeze,player=original['player']) as child:
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



def final_evidence_forecast(schedule,boundary=lambda:None):
    ops=[spec for spec in schedule if spec['stage']=='operational'];require=c.require
    require(len(ops)==6,'Six measured operational games required')
    bounds=[];paths=[];max_move=0
    for spec in ops:
        folder=game_folder(spec);files=[p for p in folder.rglob('*') if p.is_file()];paths+=files
        max_move=max(max_move,max(p.stat().st_size for p in (folder/'moves').glob('*.json')))
        for role in ('challenger','opponent'):
            worker=c.OUT/'workers'/f"{spec['stage']}_g{spec['game_index']}_{role}"
            items=[p for p in worker.rglob('*') if p.is_file()];paths+=items
            bounds.append(sum(p.stat().st_size for p in items))
        boundary()
    # Charge an entire measured closed game worker for each one-move reproduction.
    replay=48*(max(bounds)+max_move)
    allpaths=[p for p in c.OUT.rglob('*') if p.is_file()]
    item=max(len(c.canonical(c.artifact(p)).encode()) for p in allpaths)+128
    projected_files=math.ceil(len(paths)*66/6)+len(allpaths)+48*max(sum(1 for p in (c.OUT/'workers'/f"{spec['stage']}_g{spec['game_index']}_{role}").rglob('*') if p.is_file()) for spec in ops for role in ('challenger','opponent'))
    controls=16*c.MiB
    final=math.ceil(1.5*(replay+projected_files*item+controls))
    return dict(projected_final_evidence_bytes=final,projected_reproduction_bytes=replay,projected_sealed_files=projected_files,
        artifact_bound_bytes=item,fixed_control_bytes=controls,conservative_factor=1.5,outcomes_used=False)

def operational_gate(*,observed_seconds,observed_variable_bytes,current_bytes,independent_seconds,
                     fixed_final_seconds,final_evidence_bytes,remaining_collection_seconds):
    quantities=(observed_seconds,observed_variable_bytes,current_bytes,independent_seconds,fixed_final_seconds,final_evidence_bytes,remaining_collection_seconds)
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in quantities),'Invalid forecast input')
    collection=1.5*60/6*observed_seconds;prefinal=current_bytes+1.5*60/6*observed_variable_bytes
    final=1.5*66/6*independent_seconds+fixed_final_seconds
    passed=collection<min(c.STAGES['main'],remaining_collection_seconds) and final<c.STAGES['final'] and prefinal<c.PREFINAL_EVIDENCE_BYTES and final_evidence_bytes<c.FINAL_RESERVE_BYTES and prefinal+final_evidence_bytes<c.MAX_EVIDENCE_BYTES
    return dict(passed=passed,operational_games=6,remaining_games=60,total_games=66,conservative_factor=1.5,
        projected_collection_seconds=collection,projected_final_seconds=final,projected_prefinal_bytes=prefinal,
        projected_final_evidence_bytes=final_evidence_bytes,projected_total_evidence_bytes=prefinal+final_evidence_bytes,outcomes_used=False)

def verify_entrypoint(player,receipt,folder,plan,boundary):
    c.require(receipt['status']=='ok' and receipt['accepted'] and receipt['received_before_deadline'] and 0<receipt['deadline_seconds']<=300.,'Bundled startup clock/status differs')
    result=receipt['result']
    if player=='control32':
        c.require(result['status']=='entrypoint_counterexamples_passed' and result['challenger_quota']==32 and result['counts']['complete_decisions']==52,'Original52-decision startup inventory')
    else:
        from . import qualification
        qualification.validate_entrypoint_result(result,plan,folder/'breck_entrypoint_forensics.jsonl',boundary=boundary)


def qualify(envelope,freeze):
    from . import qualification
    plan=c.read(c.OUT/'qualification_plan.json');qualification.validate_plan(plan)
    c.require(plan==qualification.build_plan(),'Frozen qualification roots changed')
    costs=[];rows=[]
    for player in ('control32','breck'):
        begun=time.perf_counter()
        with make_worker(f'qualification_{player}',envelope,freeze,player=player) as child:
            startup=time.perf_counter()-begun
            receipt=adapters.startup_request(child,{'operation':'entrypoint_checks'},envelope)
            c.save(c.OUT/f'entrypoint_checks_{player}.json',receipt)
            verify_entrypoint(player,receipt,child.output_dir,plan,envelope.guard)
            if player=='breck':
                for spec in plan['requests']:
                    row=decision(child,decode_training_snapshot(spec['before']),envelope,'challenger',observer=spec['observer'])
                    row.update(spec)
                    c.save(c.OUT/'qualification_rows'/f"{spec['index']:04d}.json",row);rows.append(row)
                    check_row_integrity(row);verify_saved_decision(row,freeze)
                    if not row['completed']:break
        started=time.perf_counter();proof=verify_closed_worker(child.output_dir,freeze,envelope)
        c.save(c.OUT/f'qualification_closed_{player}.json',proof)
        audit_seconds=time.perf_counter()-started
        seal=measure_seal_cost([child.output_dir],envelope.guard)
        costs.append(dict(player=player,startup_seconds=startup,closed_verification_seconds=audit_seconds,seal_seconds=seal['seconds']))
    if len(rows)==256 and all(row['completed'] for row in rows):result=qualification.validate_qualification_rows(rows,plan)
    else:result=dict(status='request_failed',qualified=False,requests=len(rows),complete_inventory=False)
    begun=time.perf_counter();verify_runtime(freeze,envelope);runtime_seconds=time.perf_counter()-begun
    fixed=1.5*(runtime_seconds+48*(max(x['startup_seconds'] for x in costs)+1.+max(x['closed_verification_seconds']+x['seal_seconds'] for x in costs)))+60.
    result.update(costs=costs,forecast_seconds=fixed,source_seconds=runtime_seconds,actual_local_startup_decisions=52,external_startup_decisions=64)
    c.save(c.OUT/'qualification_summary.json',result);return result

def final_checks(envelope,freeze,schedule,games,stop,qualified):
    from . import qualification
    c.require(schedule==schedules.make_schedule(),'Frozen schedule derivation changed')
    plan=c.read(c.OUT/'qualification_plan.json');qpaths=sorted((c.OUT/'qualification_rows').glob('*.json'))
    c.require([p.name for p in qpaths]==[f'{i:04d}.json' for i in range(len(qpaths))],'Qualification row inventory')
    qrows=[c.read(p) for p in qpaths]
    for row,spec in zip(qrows,plan['requests'],strict=False):
        envelope.guard();c.require(all(row[k]==v for k,v in spec.items()),'Qualification prefix/order changed');verify_saved_decision(row,freeze)
    c.require(0<len(qrows)<=256 and len(qrows)==qualified['requests'],'Qualification attempted request count changed')
    if len(qrows)<256 or not all(row['completed'] for row in qrows):
        c.require(not qrows[-1]['completed'] and all(row['completed'] for row in qrows[:-1]) and not qualified['qualified'],'Qualification must stop at exactly the first failed request')
    if qualified['qualified']:c.require(qualification.validate_qualification_rows(qrows,plan)['qualified'],'Qualification no longer passes')
    elif len(qrows)==256 and all(row['completed'] for row in qrows):c.require(not qualification.validate_qualification_rows(qrows,plan)['qualified'],'Negative timing qualification changed')
    for player in ('control32','breck'):
        receipt=c.read(c.OUT/f'entrypoint_checks_{player}.json');folder=c.OUT/'workers'/f'qualification_{player}'
        c.require(c.read(folder/f"{receipt['request_id']}.json")==receipt and receipt['status']=='ok','Actual startup receipt binding')
        verify_entrypoint(player,receipt,folder,plan,envelope.guard)
    if games and stop is None and len(games)==6:
        gate=c.read(c.OUT/'operational_gate.json');c.require(not gate['passed'] and all(g['reason']=='natural' for g in games),'Unexplained operational-only prefix')
        coverage=dict(status='operational_not_admitted',complete_planned_workload=False,verified_games=6)
    else:coverage=audit.validate_stopped_prefix(schedule,games,stop) if games else dict(status='qualification_not_admitted',complete_planned_workload=False,verified_games=0)
    records=[];replays=[]
    for spec,game in zip(schedule,games,strict=False):
        envelope.guard();folder=game_folder(spec);initial=c.read(folder/'initial.json')
        expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(initial==dict(**spec,initial=expected) and c.read(folder/'result.json')==game,'Saved initial world/result changed')
        paths=sorted((folder/'moves').glob('*.json'));c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(game['plies'])],'Game move inventory changed')
        rows=[c.read(p) for p in paths]
        for row in rows:envelope.guard();verify_saved_decision(row,freeze)
        replays.append(audit.audit_game(expected,rows,game,spec,boundary=envelope.guard));records.extend(rows)
    c.require({p.parent.resolve() for p in (c.OUT/'games').rglob('initial.json')}=={game_folder(spec).resolve() for spec in schedule[:len(games)]},'Unexpected game outside played prefix')
    if stop is not None:
        row=records[-1];c.require(stop==c.read(c.OUT/'collection_stop.json')==c.read(game_folder(schedule[len(games)-1])/'captured_fault.json') and stop['clock']==row['clock'] and stop['worker_clock_file']==row['worker_clock_file'],'Original failed receipt binding')
    else:c.require(not (c.OUT/'collection_stop.json').exists(),'Unreported fault stop')
    prefix=stop is not None or len(games)<66
    sample_rows=records if prefix else [r for r in records if r['stage']=='main']
    sample=audit.select_replay_sample(sample_rows,allow_operational=prefix);c.save(c.OUT/'replay_sample_manifest.json',sample)
    for index,item in enumerate(sample['decisions']):
        original=next(r for r in sample_rows if all(r[k]==item[k] for k in ('stage','pair_index','game_index','ply','player_role')))
        reproduce_decision(original,index,envelope,freeze)
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(sample['decisions']),identities=sample['decisions']))
    expected_workers={'qualification_control32','qualification_breck'}|{f"{spec['stage']}_g{spec['game_index']}_{role}" for spec in schedule[:len(games)] for role in ('challenger','opponent')}|{f'audit_{i:02d}' for i in range(len(sample['decisions']))}
    c.require({p.name for p in (c.OUT/'workers').iterdir()}==expected_workers,'Exact named worker inventory differs')
    represented=qrows+records+[c.read(c.OUT/'repeated_decisions'/f'{i:02d}.json') for i in range(len(sample['decisions']))]
    inputs_by_worker={name:[] for name in expected_workers}
    for row in represented:
        path=c.ROOT/row['worker_clock_file'];inputs_by_worker[path.parent.name].append(path.stem)
    for name,ids in inputs_by_worker.items():
        c.require(len(ids)==len(set(ids)) and set(ids)=={p.stem for p in (c.OUT/'workers'/name/'inputs').glob('*.json')},'Unrepresented or duplicate ordinary worker request')
    proofs=[]
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard();proofs.append(verify_closed_worker(folder,freeze,envelope))
    c.require(len(proofs)==2+2*len(games)+len(sample['decisions'])<=182,'Complete worker lifecycle inventory')
    c.save(c.OUT/'all_closed_workers.json',dict(status='passed',workers=len(proofs),proofs=proofs))
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=len(games),receipts=replays))
    c.save(c.OUT/'all_placement_integrity.json',verify_exchanges(envelope,freeze))
    c.save(c.OUT/'performance.json',audit.performance(records,games));c.save(c.OUT/'boundary_analysis.json',audit.summarize_boundary_rows(records,boundary=envelope.guard))
    c.save(c.OUT/'screen_assessment.json',audit.screen_assessment(games))
    seal=[]
    for subdir in ('games','workers','qualification_rows','repeated_decisions','placement_requests','placement_acks'):
        for path in sorted((c.OUT/subdir).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope)
    c.save(c.OUT/'coverage.json',coverage)
    c.save(c.OUT/'verification_summary.json',dict(status='passed',games=len(games),requests=len(records),qualification_requests=len(qrows),
        workers=len(proofs),reproductions=len(sample['decisions']),all_science_ledger_raw_placement_logs_verified=True,
        source_inputs=len(freeze['inputs']),peak_combined_rss_bytes=envelope.peak_combined_rss,only_breck_and_unchanged32=True))
    return coverage

def run_stages(envelope,freeze,schedule):
    qualified=qualify(envelope,freeze);envelope.finish();games=[];stop=None
    if qualified['qualified']:
        envelope.begin('operational');games,stop=operational(envelope,freeze,schedule,qualified);envelope.finish()
        if stop is None and c.read(c.OUT/'operational_gate.json')['passed']:
            envelope.begin('main')
            for spec in (x for x in schedule if x['stage']=='main'):
                game,rows,cost,stop=play_game(spec,envelope,freeze);games.append(game)
                print(c.canonical(dict(stage='main',completed_games=sum(x['stage']=='main' for x in games),total=60,captured_fault=stop is not None)),flush=True)
                if stop is not None:break
            envelope.finish()
    envelope.begin('final');coverage=final_checks(envelope,freeze,schedule,games,stop,qualified);envelope.finish()
    status='completed_verified' if len(games)==66 and stop is None else 'captured_fault_verified' if stop else 'operational_not_admitted_verified' if games else 'qualification_not_admitted_verified'
    return status,coverage


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Execution freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze changed')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json');envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status,failure,coverage='incomplete',None,None
    try:
        envelope.begin('qualification');c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);envelope.admit();switch_placement(envelope,freeze,'isolated');schedule=c.read(c.OUT/'schedule.json')
        c.require(schedule==schedules.make_schedule(),'Frozen schedule changed')
        status,coverage=run_stages(envelope,freeze,schedule)
    except BaseException:
        failure=traceback.format_exc();c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure,preserved_partial_evidence=True))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,coverage=coverage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,freeze_sha256=expected_sha,
        automatic_retries=0,larger_data_deferred=True,confirmation_not_authorized=True,requires_runner_and_watchdog_clean_os_exit=True))
    return 1 if status=='incomplete' else 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    return run(args.freeze_sha256)

if __name__=='__main__':raise SystemExit(main())
