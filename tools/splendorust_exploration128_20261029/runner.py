"""Bounded fixed128 exploration calibration with independent held-out confirmation."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import math
from importlib.metadata import version
import psutil
from . import common as c,placement,transport
from tools.future_training_rules_20261021 import verify_runtime_sources, decode_training_snapshot
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules
from tools.external_strength_screen_20261018 import rules, audit as ledger
from .design import game_worker_name, replay_worker_name
from .transport import make_worker, ordinary, verify_saved_record, close_and_audit, _require_profiles
def bytes_used():
    return sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())


class Envelope:
    """One non-resettable job clock, aggregate resident-worker resource bound."""
    def __init__(self, claim):
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.end = self.start + c.JOB_SECONDS
        self.stage = None
        self.deadline, self.wall_deadline = self.end, self.wall_start + c.JOB_SECONDS
        self.waited, self.admitted, self.admissions = 0., False, 0
        self.worker_accounting = {}
        self.prior_worker_count = 0
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
        from tools.splendorust_search128_resume_20261029.accounting import WorkerAccounting
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
        if stage=='final':c.require(not self.resident,'Saved-only final admission requires no resident model trees')
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
        threshold = c.MINIMUM_AVAILABLE_RAM_BYTES if self.admitted else c.INITIAL_AVAILABLE_RAM_BYTES
        available = psutil.virtual_memory().available
        stable = begun if available >= threshold else None
        minimum = available
        fast = self.admitted and stable is not None
        while not fast:
            self.guard()
            current, available = time.monotonic(), psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < threshold:
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
    from . import analysis
    c.require(analysis.runtime_contract()==freeze['statistics_runtime'],'Pinned statistics numerical runtime changed')
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


def captured_fault(spec,row):
    from . import audit
    classification=audit.classify_failure(row['clock'])
    return dict(**{key:spec[key] for key in ('stage','candidate','pair_index','game_index')},ply=row['ply'],
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
            workers[role]=make_worker(game_worker_name(spec,role),envelope,freeze,route=spec[role+'_mode'])
        for ply in range(2000):
            if state.terminal:break
            seat=state.current_player;role='challenger' if seat==spec['challenger_seat'] else 'opponent'
            row=decision(workers[role],state,envelope,role,freeze)
            row.update(ply=ply,**{key:spec[key] for key in ('stage','candidate','pair_index','game_index')})
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



def validate_freeze_contract(freeze):
    from .preparation import contract
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()),'Exact exploration clock/resource/workload contract')
    c.require(freeze['schema']=='splendorust_exploration128_v1' and freeze['only_scientific_change']=='cpuct_0.2_or_0.8', 'Exploration identity')
    return expected


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/spec['candidate']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def decision(child,state,envelope,role,freeze):
    from . import audit
    c.require(role in audit.ROLES and (child.route=='frozen' if role=='opponent' else child.route in ('low','high')), 'Exact live role/route')
    before,observer=rules.serialize_snapshot(state),rules.observation_payload(state)
    decode_training_snapshot(before);ledger.check_observer(before,observer)
    row=ordinary(child,dict(route=child.route,quota=128),observer,envelope,freeze)
    row.update(arm='search',player_role=role,quota=128,mode=child.route,seat=state.current_player,before=before,after=None)
    if row['completed']:
        try:row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,rules.deserialize_action(row['action'])))
        except BaseException:row['transition_error']=traceback.format_exc()
    return row


def verify_saved_decision(row,freeze):
    from . import audit
    verify_saved_record(row,freeze)
    c.require(row['mode']==row['route'] and (row['mode']=='frozen' if row['player_role']=='opponent' else row['mode'] in ('low','high')) and row['arm']=='search' and row['quota']==128 and row['seat']==row['before']['state']['current_player'], 'Live role/route/quota binding')
    decode_training_snapshot(row['before']);ledger.check_observer(row['before'],row['observer'])
    if row['completed']:
        audit.validate_result(row['clock']['result'],row['observer'],row['route'],identity=audit.profile_identity(freeze,128,3,route=row['route']))
        c.require(ledger.transition(row['before'],row['action'])==row['after'],'Independent saved transition')
        decode_training_snapshot(row['after'])
    else:c.require(row['after'] is None,'Failed move cannot invent a successor')
    return dict(status='passed',request_id=row['clock']['request_id'])


def load_inputs(freeze,envelope):
    from . import design,qualification,schedule
    c.verify([freeze['schedule'],freeze['qualification_plan']])
    schedules=c.read(c.ROOT/freeze['schedule']['path']);plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    expected=dict(operational=schedule.make_schedule('operational'),development=schedule.make_schedule('development'),confirmation={arm:schedule.make_schedule('confirmation',arm) for arm in schedule.CANDIDATES})
    c.require(schedules==expected and plan==qualification.build_plan(),'Frozen schedule and actual-entrypoint inventory')
    c.require(freeze['declared_worker_lifecycles']==design.lifecycle_inventory(),'Prospective worker lifecycle inventory')
    return schedules,plan


def qualify(envelope,freeze,plan):
    from . import adapters,qualification
    receipts=[];rows=[]
    for route in ('frozen','parity','low','high'):
        child=make_worker('qualification_'+route,envelope,freeze,route=route)
        original_error=None
        try:
            receipt=adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
            pin=c.save(c.OUT/'entrypoint_checks'/f'{route}.json',receipt)
            proof=qualification.verify_entrypoint(child.output_dir,receipt,plan,freeze,boundary=envelope.guard)
            receipts.append(dict(route=route,clock=pin,verification=proof))
            for spec in [x for x in plan['panel_requests'] if x['route']==route]:
                row=ordinary(child,spec,spec['observer'],envelope,freeze)
                c.save(c.OUT/'qualification_rows'/f"{spec['index']:04d}.json",row);rows.append(row)
                verify_saved_record(row,freeze)
                c.require(row['completed'],'Qualification ordinary failure; original evidence retained')
        except BaseException:
            original_error=traceback.format_exc()
            c.save(c.OUT/'qualification_failure.json',dict(route=route,error=original_error,completed_rows=len(rows)))
            raise
        finally:
            try:close_and_audit(child,envelope,freeze)
            except BaseException:
                c.save(c.OUT/'qualification_closure_failure.json',dict(route=route,original_error=original_error,error=traceback.format_exc()))
                if original_error is None:raise
        print(c.canonical(dict(stage='qualification',route=route,status='passed')),flush=True)
    parity=qualification.compare_default_parity(c.read(c.OUT/'entrypoint_checks/frozen.json')['result'],c.read(c.OUT/'entrypoint_checks/parity.json')['result'])
    panel=qualification.validate_panel_rows(rows,plan,freeze,boundary=envelope.guard)
    summary=dict(status='passed',scientific_decisions=364,entrypoint_scientific_decisions=268,ordinary_requests=96,parent_requests=104,graph_cases=44,receipts=receipts,default_parity=parity,panel=panel)
    c.save(c.OUT/'qualification_summary.json',summary)
    return summary


def verify_qualification(envelope,freeze,plan):
    from . import qualification
    paths=sorted((c.OUT/'qualification_rows').glob('*.json'))
    c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(96)],'Exact96 qualification panel files')
    rows=[c.read(p) for p in paths];receipts=[]
    for row in rows:envelope.guard();verify_saved_record(row,freeze)
    for route in ('frozen','parity','low','high'):
        folder=c.OUT/'workers'/('qualification_'+route);receipt=c.read(c.OUT/'entrypoint_checks'/f'{route}.json')
        c.require(c.read(folder/(receipt['request_id']+'.json'))==receipt,'Actual entrypoint trusted receipt binding')
        proof=qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=envelope.guard)
        receipts.append(dict(route=route,clock=c.artifact(c.OUT/'entrypoint_checks'/f'{route}.json'),verification=proof))
        represented=[r['clock']['request_id'] for r in rows if r['route']==route]
        c.require(len(represented)==len(set(represented))==24 and set(represented)=={p.stem for p in(folder/'inputs').glob('*.json')},'Exact24 ordinary qualification requests perroute')
        closed=transport.verify_closed_worker(folder,freeze,envelope)
        c.require(closed==c.read(c.OUT/'closed_workers'/f'{folder.name}.json'),'Qualification full closure readback')
    parity=qualification.compare_default_parity(c.read(c.OUT/'entrypoint_checks/frozen.json')['result'],c.read(c.OUT/'entrypoint_checks/parity.json')['result'])
    panel=qualification.validate_panel_rows(rows,plan,freeze,boundary=envelope.guard)
    summary=dict(status='passed',scientific_decisions=364,entrypoint_scientific_decisions=268,ordinary_requests=96,parent_requests=104,graph_cases=44,receipts=receipts,default_parity=parity,panel=panel)
    c.require(c.read(c.OUT/'qualification_summary.json')==summary,'Fresh complete qualification evidence')
    return summary



def read_games(specs,envelope,freeze):
    """Fresh immutable-file replay; no live model or cached result shortcuts."""
    from . import audit
    games=[];record_paths=[];proofs=[]
    for spec in specs:
        envelope.guard();folder=game_folder(spec)
        expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(c.read(folder/'initial.json')==dict(**spec,initial=expected),'Prospective initial seed binding')
        game=c.read(folder/'result.json');paths=sorted((folder/'moves').glob('*.json'))
        c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(game['plies'])],'Exact contiguous game move inventory')
        rows=[c.read(p) for p in paths]
        for row in rows:envelope.guard();verify_saved_decision(row,freeze)
        proof=audit.audit_game(expected,rows,game,spec,freeze=freeze,boundary=envelope.guard)
        c.require(proof==c.read(folder/'online_replay.json'),'Saved complete ledger proof')
        if game['reason']=='forfeit':
            saved=c.read(folder/'captured_fault.json');recomputed=captured_fault(spec,rows[-1])
            c.require({k:v for k,v in saved.items() if k!='time'}=={k:v for k,v in recomputed.items() if k!='time'},'Preserved hard-guard forfeit attribution')
        else:c.require(not(folder/'captured_fault.json').exists(),'Hidden forfeit')
        for role in audit.ROLES:
            name=game_worker_name(spec,role);worker=c.OUT/'workers'/name
            ids=[r['clock']['request_id'] for r in rows if r['player_role']==role]
            c.require(len(ids)==len(set(ids)) and set(ids)=={p.stem for p in (worker/'inputs').glob('*.json')},'Complete game ordinary-request inventory')
            closure=transport.verify_closed_worker(worker,freeze,envelope)
            c.require(closure==c.read(c.OUT/'closed_workers'/f'{name}.json')==c.read(folder/f'closed_{role}.json'),'Independent full raw/log/graph/closure readback')
            proofs.append(closure)
        games.append(game);record_paths.extend(paths)
    return games,SavedRows(record_paths,envelope.guard),proofs


def collect_games(specs,envelope,freeze,games):
    for index,spec in enumerate(specs):
        game,rows,cost,fault=play_game(spec,envelope,freeze);games.append(game)
        print(c.canonical(dict(stage=spec['stage'],candidate=spec['candidate'],completed_games=index+1,total=len(specs),hard_guard_forfeit=fault is not None)),flush=True)
        if spec['stage']=='operational' and fault is not None:
            raise ValueError('Operational hard-guard failure; frozen attempt retained')
    return games


def reproduce_decision(original,index,stage,candidate,envelope,freeze):
    c.require(original['completed'],'Failed decisions are never retried')
    name=replay_worker_name(stage,candidate,index,original['route'])
    child=make_worker(name,envelope,freeze,route=original['route'])
    try:
        repeated=decision(child,decode_training_snapshot(original['before']),envelope,original['player_role'],freeze)
        repeated.update(original_identity={k:original[k] for k in ('stage','candidate','pair_index','game_index','ply','player_role')})
        c.save(c.OUT/'repeated_decisions'/stage/candidate/f'{index:02d}.json',repeated)
        envelope.guard();_require_profiles(repeated);verify_saved_decision(repeated,freeze)
        c.require(repeated['completed'] and all(repeated[k]==original[k] for k in ('action','trace','after')) and repeated['clock']['result']==original['clock']['result'],'Full exact scientific reproduction differs')
    finally:close_and_audit(child,envelope,freeze)
    return repeated


def verify_reproductions(stage,candidate,records,envelope,freeze,*,run_models):
    from . import audit
    sample=audit.select_replay_sample(records,maximum=48,stage=stage,candidate=candidate)
    directory=c.OUT/'replay_samples'/stage
    path=directory/f'{candidate}.json'
    if run_models:c.save(path,sample)
    else:c.require(c.read(path)==sample,'Outcome-independent replay sample derivation')
    repeated=[]
    for index,item in enumerate(sample['decisions']):
        original=c.read(game_folder(item)/'moves'/f"{item['ply']:04d}.json")
        c.require(all(original[k]==item[k] for k in ('stage','candidate','pair_index','game_index','ply','player_role')),'Exact selected saved reproduction source')
        if run_models:row=reproduce_decision(original,index,stage,candidate,envelope,freeze)
        else:row=c.read(c.OUT/'repeated_decisions'/stage/candidate/f'{index:02d}.json')
        verify_saved_decision(row,freeze)
        c.require(row['completed'] and row['original_identity']=={k:original[k] for k in ('stage','candidate','pair_index','game_index','ply','player_role')} and all(row[k]==original[k] for k in ('action','trace','after')) and row['clock']['result']==original['clock']['result'],'Exact reproduced complete scientific result')
        folder=c.OUT/'workers'/replay_worker_name(stage,candidate,index,row['route'])
        c.require(set(p.stem for p in (folder/'inputs').glob('*.json'))=={row['clock']['request_id']},'Exactly one reproduction request')
        proof=transport.verify_closed_worker(folder,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{folder.name}.json'),'Reproduction closure proof')
        repeated.append(row)
    actual=sorted((c.OUT/'repeated_decisions'/stage/candidate).glob('*.json'))
    c.require([p.name for p in actual]==[f'{i:02d}.json' for i in range(len(repeated))],'Exact reproduction file inventory')
    return dict(status='passed',requests=len(repeated),sample=c.artifact(path),workers=[Path(r['worker_folder']).name for r in repeated])


def evidence_seal(paths,envelope):
    artifacts=[];seen=set()
    for root in paths:
        entries=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
        for path in entries:
            envelope.guard();relative=path.relative_to(c.ROOT).as_posix()
            if relative in seen:continue
            seen.add(relative);artifacts.append(c.artifact(path))
    for item in artifacts:envelope.guard();c.verify([item])
    return sorted(artifacts,key=lambda x:x['path'])


def verify_phase(stage,specs,envelope,freeze,*,run_models):
    from . import analysis
    games,records,proofs=read_games(specs,envelope,freeze);reproductions=[]
    for candidate in ('low','high'):
        armrows=records.for_candidate(candidate)
        if armrows:reproductions.append(verify_reproductions(stage,candidate,armrows,envelope,freeze,run_models=run_models))
    verify_runtime(freeze,envelope)
    result=dict(status='passed',stage=stage,games=len(games),moves=len(records),closed_game_workers=len(proofs),reproductions=reproductions,
        full_ledger_raw_transport_logs_graph_clock_placement_and_source_checks=True)
    target=c.OUT/'phase_verification'/f'{stage}.json'
    if run_models:
        c.save(target,result)
        folders=[c.OUT/'games'/stage,c.OUT/'repeated_decisions'/stage,c.OUT/'replay_samples'/stage,target]
        names={game_worker_name(spec,role) for spec in specs for role in ('challenger','opponent')}
        names|={name for proof in reproductions for name in proof['workers']}
        folders += [c.OUT/'workers'/name for name in sorted(names)]
        folders += [c.OUT/'closed_workers'/f'{name}.json' for name in sorted(names)]
        c.save(c.OUT/'phase_seals'/f'{stage}.json',dict(artifacts=evidence_seal(folders,envelope)))
    else:
        c.require(c.read(target)==result,'Fresh phase-verification arithmetic')
        for item in c.read(c.OUT/'phase_seals'/f'{stage}.json')['artifacts']:envelope.guard();c.verify([item])
    return games,records,result


def forecast_resources(specs,envelope,freeze):
    """Conservative full remaining workload projection; no timing-admission gate."""
    game_costs=[];worker_costs=[];hashcosts=[];files=[]
    for spec in specs:
        folder=game_folder(spec);cost=c.read(folder/'cost.json');trees=[folder]+[c.OUT/'workers'/game_worker_name(spec,r) for r in ('challenger','opponent')]
        paths=[p for tree in trees for p in tree.rglob('*') if p.is_file()]
        game_costs.append(dict(spec=spec,cost=c.artifact(folder/'cost.json'),seconds=cost['nonwait_collection_seconds'],independent_seconds=cost['independent_seconds'],bytes=sum(p.stat().st_size for p in paths),files=len(paths)))
        hashcosts.append(c.read(folder/'hashing_measurement.json')['seconds'])
        for role in ('challenger','opponent'):
            tree=c.OUT/'workers'/game_worker_name(spec,role)
            worker_costs.append(dict(initialization=c.read(tree/'initialization.json')['elapsed_seconds'],graph=c.read(tree/'graph_parity_clock.json')['elapsed_seconds'],closure=c.read(c.OUT/'closed_worker_costs'/f'{tree.name}.json')['seconds']))
    now,wall=time.monotonic(),time.time()
    inputs=dict(game_costs=game_costs,worker_costs=worker_costs,hash_seconds=hashcosts,
      runtime_seconds=envelope.source_runtime_seconds,ram_wait_used=envelope.waited,current_bytes=bytes_used(),
      current_files=sum(1 for p in c.OUT.rglob('*') if p.is_file()),monotonic=now,unix_time=wall,
      monotonic_origin=envelope.start,unix_origin=envelope.wall_start)
    value=dict(resource_projection(inputs),inputs=inputs)
    c.save(c.OUT/'assessment_gate.json',value)
    c.require(value['passed'],'Measured whole-study resource forecast does not fit frozen limits')
    return value


class SavedRows:
    """Repeatable, bounded-memory view of already checked immutable move files."""
    def __init__(self,paths,boundary=lambda:None):self.paths=list(paths);self.boundary=boundary
    def __len__(self):return len(self.paths)
    def __iter__(self):
        for path in self.paths:
            self.boundary();yield c.read(path)
    def for_candidate(self,candidate):
        return SavedRows([p for p in self.paths if p.parent.parent.parent.name==candidate],self.boundary)


def save_or_verify(path,value,writing):
    if writing:return c.save(path,value)
    c.require(c.read(path)==value,'Saved acceptance arithmetic differs: '+str(path))
    return c.artifact(path)


def verify_selection_evidence(envelope,freeze,specs,*,writing):
    from . import analysis
    games,records,verification=verify_phase('development',specs,envelope,freeze,run_models=False)
    selection=analysis.select_candidate(games,True,boundary=envelope.guard)
    checked=analysis.verify_selection(games,selection,boundary=envelope.guard)
    save_or_verify(c.OUT/'selection.json',selection,writing)
    save_or_verify(c.OUT/'selection_verification.json',checked,writing)
    binding=dict(freeze=c.artifact(c.OUT/'freeze.json'),qualification=c.artifact(c.OUT/'qualification_summary.json'),development_verification=c.artifact(c.OUT/'phase_verification/development.json'),development_seal=c.artifact(c.OUT/'phase_seals/development.json'),selection=c.artifact(c.OUT/'selection.json'),analysis_verification=c.artifact(c.OUT/'selection_verification.json'))
    if writing:c.save(c.OUT/'selection_authorization.json',dict(bindings=binding,perf_counter_ns=time.perf_counter_ns(),monotonic=time.monotonic(),unix_time=time.time()))
    else:c.require(c.read(c.OUT/'selection_authorization.json')['bindings']==binding,'Immutable fully verified development evidence authorized selection')
    return selection


def verify_saved_attempt(envelope,freeze,*,writing):
    """Complete fresh saved readback. Never starts a model, native helper or game."""
    from . import analysis,audit,qualification,schedule
    schedules,plan=load_inputs(freeze,envelope)
    qproof=verify_qualification(envelope,freeze,plan)
    ops,oprows,opproofs=read_games(schedules['operational'],envelope,freeze)
    c.require(len(ops)==12 and all(g['reason']=='natural' for g in ops),'All operational games qualified')
    verify_resource_forecast(schedules['operational'],envelope)
    selection=verify_selection_evidence(envelope,freeze,schedules['development'],writing=False)
    selected=selection['selected_candidate'];specs=schedules['operational']+schedules['development'];games=ops
    dev=[c.read(game_folder(spec)/'result.json') for spec in schedules['development']];games+=dev
    if selected is not None:
        confirm,_,_=verify_phase('confirmation',schedules['confirmation'][selected],envelope,freeze,run_models=False)
        result=analysis.analyze_confirmation(confirm,selected,True,boundary=envelope.guard)
        check=analysis.verify_confirmation(confirm,selected,result,boundary=envelope.guard)
        save_or_verify(c.OUT/'confirmation_analysis.json',result,writing)
        save_or_verify(c.OUT/'confirmation_analysis_verification.json',check,writing)
        specs+=schedules['confirmation'][selected];games+=confirm
        verify_resource_forecast(schedules['operational'],envelope,scope='confirmation')
        permit=c.read(c.OUT/'confirmation_admission.json')
        c.require(permit['resource_forecast']==c.artifact(c.OUT/'confirmation_resource_gate.json'),'Conditional resource admission binding')
        c.require(permit['selected_candidate']==selected and permit['selection']==c.artifact(c.OUT/'selection.json') and permit['selection_verification']==c.artifact(c.OUT/'selection_verification.json'),'Immutable selection before confirmation')
        first=c.read(game_folder(schedules['confirmation'][selected][0])/'initial.json')
        c.require(first['candidate']==selected,'Only selected branch played')
        authority=c.read(c.OUT/'selection_authorization.json')
        c.require(permit['selection_authorization']==c.artifact(c.OUT/'selection_authorization.json') and authority['perf_counter_ns']<=permit['perf_counter_ns'],'Selection authority precedes confirmation admission')
        for spec in schedules['confirmation'][selected]:
            for role in ('challenger','opponent'):
                folder=c.OUT/'workers'/game_worker_name(spec,role)
                validate_confirmation_startup(permit,c.read(folder/'initialization.json'),c.read(folder/'placement_admission.json'))
            firstmove=c.read(game_folder(spec)/'moves/0000.json')
            c.require(firstmove['parent_profile']['request_start_ns']>permit['perf_counter_ns'],'Every confirmation game begins after frozen selection admission')
    else:
        c.require(not(c.OUT/'confirmation_admission.json').exists() and not(c.OUT/'confirmation_analysis.json').exists(),'No confirmation after negative development screen')
    schedule.validate_schedule(specs,selected)
    actual={p.parent.resolve() for p in (c.OUT/'games').rglob('initial.json')}
    c.require(actual=={game_folder(s).resolve() for s in specs},'No hidden additional or partial games')
    expected_phases={'development'}|({'confirmation'} if selected is not None else set())
    for directory in ('phase_verification','phase_seals'):
        c.require({p.stem for p in (c.OUT/directory).glob('*.json')}==expected_phases,'Exact completed verification phase inventory')
    expected_workers={'qualification_'+r for r in ('frozen','parity','low','high')}
    expected_workers|={game_worker_name(s,r) for s in specs for r in ('challenger','opponent')}
    for path in (c.OUT/'phase_verification').glob('*.json'):
        for receipt in c.read(path)['reproductions']:expected_workers.update(receipt['workers'])
    actual_workers={p.name for p in (c.OUT/'workers').iterdir() if p.is_dir()}
    c.require(actual_workers==expected_workers and len(expected_workers)<=c.MAX_WORKERS,'Exact complete owned worker inventory')
    rows=SavedRows([p for spec in specs for p in sorted((game_folder(spec)/'moves').glob('*.json'))],envelope.guard)
    save_or_verify(c.OUT/'performance.json',audit.performance(rows,games),writing)
    save_or_verify(c.OUT/'boundary_analysis.json',audit.summarize_boundary_rows(rows,boundary=envelope.guard),writing)
    faults=[c.read(game_folder(s)/'captured_fault.json') for s in specs if (game_folder(s)/'captured_fault.json').exists()]
    save_or_verify(c.OUT/'all_player_faults.json',dict(count=len(faults),faults=faults,failed_move_retries=0),writing)
    proofs=[];parent_requests=0
    for name in sorted(expected_workers):
        envelope.guard();proof=transport.verify_closed_worker(c.OUT/'workers'/name,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json'),'Final closed worker proof')
        parent_requests+=proof['raw']['requests'];proofs.append(c.artifact(c.OUT/'closed_workers'/f'{name}.json'))
    save_or_verify(c.OUT/'all_closed_workers.json',dict(status='passed',workers=len(proofs),proofs=proofs),writing)
    save_or_verify(c.OUT/'all_placement_integrity.json',verify_exchanges(envelope,freeze),writing)
    expected_stages=['qualification','operational','development','development_verification']
    if selected is not None:expected_stages+=['confirmation','confirmation_verification']
    for stage in expected_stages:
        begin=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        c.require(begin['monotonic']<=end['monotonic']<begin['monotonic_deadline']<=envelope.start+c.COLLECTION_CUTOFF_SECONDS and begin['unix_time']<=end['unix_time']<begin['unix_deadline'],'Real stage clocks and all model work before collection cutoff')
    replaycount=sum(r['requests'] for p in (c.OUT/'phase_verification').glob('*.json') for r in c.read(p)['reproductions'])
    coverage=dict(complete=True,operational_games=12,development_games=200,confirmation_games=400 if selected else 0,selected_candidate=selected,
      actual_games=len(games),ordinary_game_decisions=len(rows),qualification_scientific_decisions=364,qualification_parent_requests=104,
      model_reproductions=replaycount,workers=len(expected_workers),graph_cases=11*len(expected_workers),parent_requests=parent_requests,
      hard_guard_forfeits=len(faults),failed_decision_retries=0,training=False,unplayed_unselected_confirmation_games=400)
    c.require(parent_requests==len(rows)+96+4+len(expected_workers)+replaycount,'Exact graph/suite/ordinary parent-request reconciliation')
    save_or_verify(c.OUT/'coverage.json',coverage,writing)
    verify_runtime(freeze,envelope)
    return coverage


def final_checks(envelope,freeze):
    coverage=verify_saved_attempt(envelope,freeze,writing=True)
    c.save(c.OUT/'verification_summary.json',dict(status='passed',coverage=coverage,source_inputs=len(freeze['inputs']),
      peak_combined_rss_bytes=envelope.peak_combined_rss,all_raw_placement_logs_graph_science_sources_verified=True))
    directories=('workers','games','entrypoint_checks','qualification_rows','closed_workers','closed_worker_costs','phase_verification','phase_seals','repeated_decisions','replay_samples','placement_requests','placement_acks','memory_admissions')
    paths=[c.OUT/name for name in directories]
    names=('qualification_summary.json','source_runtime_measurement.json','assessment_gate.json','selection.json','selection_verification.json','selection_authorization.json','confirmation_resource_gate.json','confirmation_admission.json','confirmation_analysis.json','confirmation_analysis_verification.json','performance.json','boundary_analysis.json','all_player_faults.json','all_closed_workers.json','all_placement_integrity.json','coverage.json','verification_summary.json')
    paths += [c.OUT/name for name in names if(c.OUT/name).exists()]
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=evidence_seal(paths,envelope)))
    return coverage


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact execution freeze')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Owning watchdog freeze')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json')
    envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha,phase=c.PHASE))
    games=[];coverage=None;failure=None;status='incomplete'
    try:
        envelope.begin('qualification');c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        begun=time.perf_counter();verify_runtime(freeze,envelope);envelope.source_runtime_seconds=time.perf_counter()-begun
        c.save(c.OUT/'source_runtime_measurement.json',dict(seconds=envelope.source_runtime_seconds))
        envelope.admit();switch_placement(envelope,freeze,'isolated');schedules,plan=load_inputs(freeze,envelope)
        qualify(envelope,freeze,plan);envelope.finish()
        envelope.begin('operational');collect_games(schedules['operational'],envelope,freeze,games)
        forecast_resources(schedules['operational'],envelope,freeze);envelope.finish()
        envelope.begin('development');collect_games(schedules['development'],envelope,freeze,games);envelope.finish()
        envelope.begin('development_verification');verify_phase('development',schedules['development'],envelope,freeze,run_models=True)
        selection=verify_selection_evidence(envelope,freeze,schedules['development'],writing=True);envelope.finish()
        selected=selection['selected_candidate']
        print(c.canonical(dict(stage='development_verification',selected_candidate=selected,status='verified')),flush=True)
        if selected is not None:
            envelope.begin('confirmation')
            confirmation_forecast(schedules['operational'],envelope)
            c.save(c.OUT/'confirmation_admission.json',dict(selected_candidate=selected,selection=c.artifact(c.OUT/'selection.json'),selection_verification=c.artifact(c.OUT/'selection_verification.json'),selection_authorization=c.artifact(c.OUT/'selection_authorization.json'),resource_forecast=c.artifact(c.OUT/'confirmation_resource_gate.json'),perf_counter_ns=time.perf_counter_ns(),time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),sample_size=400,no_optional_stopping=True))
            collect_games(schedules['confirmation'][selected],envelope,freeze,games);envelope.finish()
            envelope.begin('confirmation_verification');verify_phase('confirmation',schedules['confirmation'][selected],envelope,freeze,run_models=True);envelope.finish()
        envelope.begin('final');coverage=final_checks(envelope,freeze);envelope.finish();status='completed_verified'
    except BaseException:
        failure=traceback.format_exc();c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
        print(c.canonical(dict(stage=envelope.stage,status='failed_preserved',error=failure)),flush=True)
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,coverage=coverage,freeze_sha256=expected_sha,
      phase=c.PHASE,elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,maximum_seconds=c.JOB_SECONDS,
      actual_games=len(games),automatic_retries=0,training=False,requires_clean_actual_os_exits=['runner','watchdog','independent_reader','guardian']))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True)
    return run(parser.parse_args().freeze_sha256)


def resource_projection(inputs):
    games=inputs['game_costs'];workers=inputs['worker_costs'];hashes=inputs['hash_seconds']
    c.require(len(games)==12 and len(workers)==24 and len(hashes)==12,'Twelve operational games measured')
    numerical=[inputs[k] for k in ('runtime_seconds','ram_wait_used','current_bytes','current_files','monotonic','unix_time','monotonic_origin','unix_origin')]
    numerical += [r[k] for r in games for k in ('seconds','independent_seconds','bytes','files')]+[r[k] for r in workers for k in ('initialization','graph','closure')]+hashes
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in numerical),'Finite nonnegative forecast values')
    c.require(inputs['ram_wait_used']<=c.RAM_WAIT_SECONDS,'Bound original RAM wait budget')
    maximum=lambda key:max(x[key] for x in games)
    startup=max(x['initialization']+x['graph']+x['closure'] for x in workers)
    scope=inputs.get('scope','initial');c.require(scope in ('initial','confirmation'),'Resource forecast scope')
    games_left,replays_left,verify_passes=(600,144,800) if scope=='initial' else (400,48,400)
    wait=c.RAM_WAIT_SECONDS-inputs['ram_wait_used'];replay=1.5*replays_left*(startup+c.SECONDS_PER_MOVE+1.)
    collection=1.5*games_left*maximum('seconds')+replay+1.5*verify_passes*maximum('independent_seconds')+wait+1800.
    final=1.5*612*(maximum('independent_seconds')+max(hashes))+2*inputs['runtime_seconds']+c.INDEPENDENT_REVIEW_SECONDS+600.
    prefinal=inputs['current_bytes']+math.ceil(1.5*games_left*maximum('bytes')+1.5*replays_left*maximum('bytes')/2)
    files=inputs['current_files']+math.ceil(1.5*(games_left+replays_left/2)*maximum('files'));final_bytes=math.ceil(files*2048+128*c.MiB)
    available_collection=min(inputs['monotonic_origin']+c.COLLECTION_CUTOFF_SECONDS-inputs['monotonic'],inputs['unix_origin']+c.COLLECTION_CUTOFF_SECONDS-inputs['unix_time'])
    available_global=min(inputs['monotonic_origin']+c.JOB_SECONDS-inputs['monotonic'],inputs['unix_origin']+c.JOB_SECONDS-inputs['unix_time'])
    phases=dict(development=1.5*200*maximum('seconds'),confirmation=1.5*400*maximum('seconds'),development_verification=1.5*400*maximum('independent_seconds')+replay*2/3+600.,confirmation_verification=1.5*400*maximum('independent_seconds')+replay/3+600.)
    if scope=='confirmation':phases={key:value for key,value in phases.items() if key.startswith('confirmation')};phases['confirmation_verification']=1.5*400*maximum('independent_seconds')+replay+600.
    passed=all(value<c.STAGES[key] for key,value in phases.items()) and collection<available_collection and final<c.STAGES['final'] and collection+final<available_global and prefinal<c.PREFINAL_EVIDENCE_BYTES and final_bytes<c.FINAL_RESERVE_BYTES and prefinal+final_bytes<c.MAX_EVIDENCE_BYTES
    return dict(passed=passed,measurement='Maximum of12 fresh operational game costs,1.5 contingency',scope=scope,operational_games=12,remaining_games=games_left,replay_decisions_maximum=replays_left,
      phase_projected_seconds=phases,projected_collection_seconds=collection,projected_saved_final_seconds=final,
      projected_prefinal_bytes=prefinal,projected_final_bytes=final_bytes,projected_files=files,
      available_collection_seconds=available_collection,available_global_seconds=available_global,remaining_ram_wait_seconds=wait,
      timing_used_for_admission=False,outcomes_used=False)


def verify_resource_forecast(specs,envelope,scope='initial'):
    value=c.read(c.OUT/('assessment_gate.json' if scope=='initial' else 'confirmation_resource_gate.json'));inputs=value['inputs']
    c.require(inputs.get('scope','initial')==scope,'Conditional forecast scope')
    c.require(inputs['monotonic_origin']==envelope.start and inputs['unix_origin']==envelope.wall_start,'Original nonresettable forecast clocks')
    phase='operational' if scope=='initial' else 'confirmation'
    stage=c.read(c.OUT/'stages'/f'{phase}_start.json');end=c.read(c.OUT/'stages'/f'{phase}_end.json')
    c.require(stage['monotonic']<=inputs['monotonic']<end['monotonic'] and stage['unix_time']<=inputs['unix_time']<end['unix_time'],'Forecast made inside operational stage')
    c.require(inputs['runtime_seconds']==c.read(c.OUT/'source_runtime_measurement.json')['seconds'],'Bound measured source closure runtime')
    c.require(len(inputs['game_costs'])==len(specs)==12,'Complete fixed operational cost inventory')
    worker_costs=[];hashes=[]
    for spec,item in zip(specs,inputs['game_costs'],strict=True):
        envelope.guard();folder=game_folder(spec);cost=c.read(folder/'cost.json')
        c.require(item['spec']==spec and item['cost']==c.artifact(folder/'cost.json') and item['seconds']==cost['nonwait_collection_seconds'] and item['independent_seconds']==cost['independent_seconds'],'Bound original measured game costs')
        trees=[folder]+[c.OUT/'workers'/game_worker_name(spec,r) for r in ('challenger','opponent')]
        paths=[p for tree in trees for p in tree.rglob('*') if p.is_file()]
        c.require(item['bytes']==sum(p.stat().st_size for p in paths) and item['files']==len(paths),'Measured immutable game evidence bytes/files')
        hashes.append(c.read(folder/'hashing_measurement.json')['seconds'])
        for role in ('challenger','opponent'):
            tree=c.OUT/'workers'/game_worker_name(spec,role)
            worker_costs.append(dict(initialization=c.read(tree/'initialization.json')['elapsed_seconds'],graph=c.read(tree/'graph_parity_clock.json')['elapsed_seconds'],closure=c.read(c.OUT/'closed_worker_costs'/f'{tree.name}.json')['seconds']))
    c.require(inputs['worker_costs']==worker_costs and inputs['hash_seconds']==hashes,'Original graph/initialization/closure/hash cost measurements')
    c.require(inputs['current_bytes']>=sum(x['bytes'] for x in inputs['game_costs']) and inputs['current_files']>=sum(x['files'] for x in inputs['game_costs']),'Original evidence census covers all operational trees')
    c.require(value==dict(resource_projection(inputs),inputs=inputs) and value['passed'] is True,'Independent whole-study and each-stage resource admission arithmetic')
    return value


def confirmation_forecast(specs,envelope):
    inputs=dict(c.read(c.OUT/'assessment_gate.json')['inputs'])
    inputs.update(scope='confirmation',monotonic=time.monotonic(),unix_time=time.time(),ram_wait_used=envelope.waited,
      current_bytes=bytes_used(),current_files=sum(1 for p in c.OUT.rglob('*') if p.is_file()))
    result=dict(resource_projection(inputs),inputs=inputs)
    c.save(c.OUT/'confirmation_resource_gate.json',result)
    c.require(result['passed'],'Remaining confirmation, replay and final resources do not fit; no confirmation game launched')
    return result


def validate_confirmation_startup(permit,initialization,admission):
    # OS creation precedes all graph construction, warmup and READY, whereas the
    # saved placement admission is captured only after initialization returns.
    c.require(initialization['identity']['created']>permit['unix_time'],'Confirmation process creation and every model initialization must follow selected admission')
    c.require(admission['supervisor']['monotonic_ns']>permit['perf_counter_ns'],'Confirmation retained placement must follow selected admission')
    return True

if __name__=='__main__':raise SystemExit(main())
