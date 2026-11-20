"""One disclosed interrupted-move retry and unchanged fixed-sample continuation.

Original attempts remain immutable. New workers and every new receipt belong to
this namespace; old accepted decisions are referenced and independently checked.
"""
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
from . import common as c,placement,transport,adapters
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
        validate_original_origins(claim)
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.recovery_owner_start=claim.get('owner_started_monotonic')
        self.recovery_owner_wall_start=claim.get('owner_started_unix_time')
        self.end = self.start + c.JOB_SECONDS
        self.stage = None
        self.deadline, self.wall_deadline = self.end, self.wall_start + c.JOB_SECONDS
        self.waited, self.admitted, self.admissions = c.PRIOR_RAM_WAIT_SECONDS, False, 0
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
        from tools.splendorust_search256_accounting_20261029 import WorkerAccounting
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
        if stage in ('recovery_qualification','main'):
            self.deadline=min(self.deadline,c.ORIGINAL_MAIN_MONOTONIC_DEADLINE)
            self.wall_deadline=min(self.wall_deadline,c.ORIGINAL_MAIN_UNIX_DEADLINE)
        if stage=='recovery_qualification':
            c.require(self.recovery_owner_start is not None and self.recovery_owner_wall_start is not None,
                'Qualification must bind the actual recovery guardian start')
            self.deadline=min(self.deadline,self.recovery_owner_start+cap)
            self.wall_deadline=min(self.wall_deadline,self.recovery_owner_wall_start+cap)
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


def validate_original_origins(claim):
    c.require(claim['monotonic_origin']==c.PRIOR_ORIGIN and claim['unix_origin']==c.PRIOR_WALL_START,
        'Amendment must retain both original nonresettable owner clocks')
    return True


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


def decision(child,state,envelope,role,freeze):
    from . import audit
    c.require(role in audit.ROLES and (child.route=='frozen' if role=='opponent' else child.route=='research'), 'Exact live role/route')
    before,observer=rules.serialize_snapshot(state),rules.observation_payload(state)
    decode_training_snapshot(before);ledger.check_observer(before,observer)
    quota=adapters.route_quota(child.route)
    row=ordinary(child,dict(route=child.route,quota=quota),observer,envelope,freeze)
    row.update(arm='search',player_role=role,quota=quota,mode=child.route,seat=state.current_player,before=before,after=None)
    if row['completed']:
        try:row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,rules.deserialize_action(row['action'])))
        except BaseException:row['transition_error']=traceback.format_exc()
    return row


def verify_saved_decision(row,freeze):
    from . import audit
    verify_saved_record(row,freeze)
    c.require(row['player_role'] in audit.ROLES and row['mode']==row['route'] and (row['mode']=='frozen' if row['player_role']=='opponent' else row['mode']=='research') and row['arm']=='search' and row['quota']==adapters.route_quota(row['route']) and row['seat']==row['before']['state']['current_player'], 'Live role/route/quota binding')
    decode_training_snapshot(row['before']);ledger.check_observer(row['before'],row['observer'])
    if row['completed']:
        audit.validate_result(row['clock']['result'],row['observer'],row['route'],identity=audit.profile_identity(freeze,adapters.route_quota(row['route']),3,route=row['route']))
        c.require(ledger.transition(row['before'],row['action'])==row['after'],'Independent saved transition')
        decode_training_snapshot(row['after'])
    else:c.require(row['after'] is None,'Failed move cannot invent a successor')
    return dict(status='passed',request_id=row['clock']['request_id'])


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


def validate_freeze_contract(freeze):
    from .preparation import contract,SCHEMA
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()),'Exact disclosed recovery contract')
    c.require(freeze.get('schema')==SCHEMA,'Separately disclosed recovery schema required')
    c.require(freeze['only_scientific_change']=='simulations_128_to_256','Scientific comparison remains unchanged')
    return expected


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/spec['candidate']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def load_inputs(freeze,envelope):
    from . import design,qualification,schedule
    c.verify([freeze['schedule'],freeze['qualification_plan']])
    schedules=c.read(c.ROOT/freeze['schedule']['path']);plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(schedules=={s:schedule.make_schedule(s) for s in ('operational','main')},'Exact unchanged original schedule')
    c.require(plan==qualification.build_plan(),'Unchanged actual qualification cases')
    c.require(freeze['declared_worker_lifecycles']==design.lifecycle_inventory(),'Exact recovery-only worker allowlist')
    return schedules,plan


def prefix_manifest(prefix):
    """Immutable references, never replacement copies of the original moves."""
    return dict(schema='search256_saved_game_prefix_v1',
        original_initial=c.artifact(c.OLD_OUT/'games/main/research/p117_g235/initial.json'),
        original_moves=prefix['move_artifacts'],original_last_ply=29,continuation_first_ply=30,
        ambiguous_original_request_id='request-00000017',
        original_worker_folder=(c.OLD_OUT/'workers/main_research_g235_challenger').relative_to(c.ROOT).as_posix(),
        only_authorized_retry=True)


def verify_retry_boundary(spec,rows,state,child,freeze):
    """Authorize exactly the unresolved input, with no claim about old dispatch."""
    c.require(spec['stage']=='main' and spec['candidate']=='research' and spec['game_index']==235 and
        spec['pair_index']==117 and spec['challenger_seat']==0,'Only original interrupted game may resume')
    c.require(len(rows)==30 and [r['ply'] for r in rows]==list(range(30)) and all(r['completed'] for r in rows),
        'Exactly30 accepted original moves precede the retry')
    snapshot=rules.serialize_snapshot(state)
    c.require(snapshot==rows[-1]['after'] and snapshot['state']['turn_index']==30 and
        snapshot['state']['current_player']==0 and not snapshot['state']['terminal'],'Exact saved continuation state')
    payload=c.read(c.OLD_OUT/'workers/main_research_g235_challenger/inputs/request-00000017.json')
    observer=rules.observation_payload(state)
    c.require(payload==dict(player='search',observer=observer),'Retry uses exactly the original lawful input')
    c.require(child.route=='research' and child._counter==2 and child.output_dir==c.OUT/'workers/main_research_g235_challenger',
        'Exactly the retained qualified256 worker and first ordinary request3')
    c.require(not (c.OUT/'retry_authorization_consumed.json').exists() and not (c.OUT/'disclosed_retry.json').exists(),
        'Single-use retry; no restart or second attempt')
    return dict(schema='search256_single_disclosed_retry_v1',game_index=235,ply=30,
        original_request_id='request-00000017',new_request_id='request-00000003',
        original_input=c.artifact(c.OLD_OUT/'workers/main_research_g235_challenger/inputs/request-00000017.json'),
        original_last_move=c.artifact(c.OLD_OUT/'games/main/research/p117_g235/moves/0029.json'),
        new_worker_folder=child.output_dir.relative_to(c.ROOT).as_posix(),
        observer_sha256=c.digest(observer),original_dispatch_status='unresolved',original_attempt_status='incomplete',
        fresh_complete_response_seconds=c.SECONDS_PER_MOVE,fresh_internal_seconds=c.INTERNAL_SECONDS,
        original_global_clocks_unchanged=True,maximum_disclosed_retries=1,automatic_retries=0,
        freeze=c.artifact(c.OUT/'freeze.json'),time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),
        perf_counter_ns=time.perf_counter_ns())


def consume_retry_authorization(spec,rows,state,child,freeze):
    """Exclusive durable commit before dispatch; an exception never unconsumes it."""
    return c.save(c.OUT/'retry_authorization_consumed.json',verify_retry_boundary(spec,rows,state,child,freeze))


def verify_disclosed_retry(context,freeze,envelope,row):
    """Saved-only proof of the one explicitly changed move clock."""
    envelope.guard();path=c.OUT/'retry_authorization_consumed.json';marker=c.read(path)
    old_input=c.OLD_OUT/'workers/main_research_g235_challenger/inputs/request-00000017.json'
    old_last=c.OLD_OUT/'games/main/research/p117_g235/moves/0029.json'
    new_move=game_folder(row)/'moves/0030.json'
    expected=dict(schema='search256_single_disclosed_retry_v1',game_index=235,ply=30,
        original_request_id='request-00000017',new_request_id='request-00000003',
        original_input=c.artifact(old_input),original_last_move=c.artifact(old_last),
        new_worker_folder=(c.OUT/'workers/main_research_g235_challenger').relative_to(c.ROOT).as_posix(),
        observer_sha256=c.digest(row['observer']),original_dispatch_status='unresolved',original_attempt_status='incomplete',
        fresh_complete_response_seconds=c.SECONDS_PER_MOVE,fresh_internal_seconds=c.INTERNAL_SECONDS,
        original_global_clocks_unchanged=True,maximum_disclosed_retries=1,automatic_retries=0,
        freeze=c.artifact(c.OUT/'freeze.json'))
    c.require(set(marker)==set(expected)|{'time','monotonic','unix_time','perf_counter_ns'} and
        all(marker[k]==v for k,v in expected.items()),'Exact exclusive retry commitment and immutable inputs')
    c.require(row['stage']=='main' and row['candidate']=='research' and row['game_index']==235 and
        row['pair_index']==117 and row['ply']==30 and row['seat']==0 and row['player_role']=='challenger' and
        row['route']=='research' and row['quota']==256 and row['clock']['request_id']=='request-00000003',
        'Only declared replacement move can use the fresh clock')
    c.require(c.read(old_input)==dict(player='search',observer=row['observer']) and
        c.read(old_last)['after']==row['before'] and c.read(new_move)==row,'Exact retry input/state/output binding')
    permit=c.read(c.OUT/'recovery_main_admission.json')
    c.require(permit['perf_counter_ns']<=marker['perf_counter_ns']<row['parent_profile']['request_start_ns'] and
        permit['monotonic']<=marker['monotonic']<c.ORIGINAL_MAIN_MONOTONIC_DEADLINE and
        permit['unix_time']<=marker['unix_time']<c.ORIGINAL_MAIN_UNIX_DEADLINE,
        'Passed qualified admission then durable consumption precede the replacement clock')
    c.require(c.read(c.OUT/'disclosed_retry.json')==dict(status='recorded',authorization=c.artifact(path),
        new_move=c.artifact(new_move),completed=row['completed'],clock_status=row['clock']['status'],
        disclosed_retry_count=1,automatic_retries=0,original_request_not_classified_as_timeout=True),
        'Exactly one disclosed replacement outcome; no invented old timeout')
    return dict(status='passed',disclosed_retries=1,automatic_retries=0,marker=c.artifact(path),move=c.artifact(new_move))


def qualify_retained_workers(spec,envelope,freeze,plan,workers):
    """Two replacement trees, each checked then retained for the saved game."""
    from . import qualification
    receipts=[]
    for role in ('challenger','opponent'):
        route=spec[role+'_mode'];child=make_worker(game_worker_name(spec,role),envelope,freeze,route=route)
        workers[role]=child
        receipt=adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
        pin=c.save(c.OUT/'entrypoint_checks'/f'{route}.json',receipt)
        proof=qualification.verify_entrypoint(child.output_dir,receipt,plan,freeze,boundary=envelope.guard)
        old=c.read(c.OLD_OUT/'entrypoint_checks'/f'{route}.json')
        c.require(old['status']=='ok' and c.digest(receipt['result'])==c.digest(old['result']),
            'Complete67 unchanged scientific suite outputs match original qualified route')
        receipts.append(dict(route=route,clock=pin,verification=proof,old_clock=c.artifact(c.OLD_OUT/'entrypoint_checks'/f'{route}.json')))
        c.require(child._counter==2,'Graph and one actual entrypoint suite, no hidden ordinary request')
    result=dict(status='passed',scientific_decisions=134,parent_requests=4,graph_cases=22,
        worker_trees=2,retained_for_game235=True,original_qualification_reused=True,
        repeats_of_failed_decisions=0,receipts=receipts)
    c.save(c.OUT/'recovery_qualification_summary.json',result)
    return result


def transition_retained_workers(workers,envelope):
    c.require(envelope.stage=='main' and set(workers)=={'challenger','opponent'},'Qualified idle pair enters original main stage')
    transitions=[]
    for role,child in workers.items():
        c.require(child._counter==2 and not child._closed and child.recovery_accounting.state=='idle',
            'No active request while changing retained-worker phase ceiling')
        before=child.absolute_deadline
        child.absolute_deadline=envelope.effective_deadline()
        c.require(child.absolute_deadline<=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE,'Retained worker keeps original main deadline')
        transitions.append(dict(role=role,worker=child.output_dir.name,pid=child.process.pid,
            previous_absolute_deadline=before,absolute_deadline=child.absolute_deadline,
            request_counter=child._counter,active_request=False))
    return c.save(c.OUT/'retained_worker_phase_transition.json',dict(stage='main',time=c.now(),
        monotonic=time.monotonic(),unix_time=time.time(),original_main_deadline=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE,
        original_main_unix_deadline=c.ORIGINAL_MAIN_UNIX_DEADLINE,workers=transitions))


def play_game(spec,envelope,freeze,context,*,prefix=None,retained_workers=None):
    from . import schedule,verification,audit
    schedule.validate_spec(spec)
    c.require(spec['stage']=='main' and 235<=spec['game_index']<400,'Only unchanged remaining main games')
    resumed=spec['game_index']==235
    c.require(resumed==(prefix is not None)==(retained_workers is not None),'Exactly one segmented game')
    if resumed:
        rows=list(prefix['rows']);initial=prefix['initial'];state=decode_training_snapshot(prefix['next_snapshot'])
        c.require(prefix['spec']==spec and len(rows)==30,'Exact qualified saved prefix')
    else:
        rows=[];state=rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state()
        initial=rules.serialize_snapshot(state)
    decode_training_snapshot(initial);folder=game_folder(spec)
    c.save(folder/'initial.json',dict(**spec,initial=initial))
    if resumed:c.save(folder/'prefix_manifest.json',prefix_manifest(prefix))
    workers=retained_workers if retained_workers is not None else {}
    fault=None;outcome=None;independent=0.;begun=time.perf_counter();original_error=None
    first_admission=envelope.admissions;start_ply=len(rows)
    try:
        if not resumed:
            for role in ('challenger','opponent'):
                workers[role]=make_worker(game_worker_name(spec,role),envelope,freeze,route=spec[role+'_mode'])
        for ply in range(start_ply,2000):
            if state.terminal:break
            seat=state.current_player;role='challenger' if seat==spec['challenger_seat'] else 'opponent'
            retry=None
            if resumed and ply==30:
                retry=consume_retry_authorization(spec,rows,state,workers[role],freeze)
            row=decision(workers[role],state,envelope,role,freeze)
            row.update(ply=ply,**{key:spec[key] for key in ('stage','candidate','pair_index','game_index')})
            pin=c.save(folder/'moves'/f'{ply:04d}.json',row);rows.append(row)
            if retry is not None:
                c.require(row['clock']['request_id']=='request-00000003','Authorized replacement request identity')
                c.save(c.OUT/'disclosed_retry.json',dict(status='recorded',authorization=retry,new_move=pin,
                    completed=row['completed'],clock_status=row['clock']['status'],disclosed_retry_count=1,
                    automatic_retries=0,original_request_not_classified_as_timeout=True))
            c.require('transition_error' not in row,'Saved transition failure; original clock retained')
            envelope.guard();_require_profiles(row)
            started=time.perf_counter();verify_saved_decision(row,freeze);independent+=time.perf_counter()-started
            if not row['completed']:
                fault=captured_fault(spec,row)
                outcome=dict(reason='forfeit',winner=1-seat,loser=seat,failed_role=role,failed_seat=seat,
                    failure=fault['failure_classification'],administrative_forfeit=True,
                    failure_policy='research_hard_guard_9s_internal_10s_complete_response')
                c.save(folder/'captured_fault.json',fault);break
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
        original_error=traceback.format_exc();raise
    finally:
        errors=[]
        for role,child in workers.items():
            try:
                started=time.perf_counter();proof=close_and_audit(child,envelope,freeze)
                independent+=time.perf_counter()-started;c.save(folder/f'closed_{role}.json',proof)
            except BaseException:errors.append(dict(role=role,error=traceback.format_exc()))
        if errors:
            c.save(folder/'closure_failures.json',dict(original_error=original_error,errors=errors))
            if original_error is None:raise ValueError('Owned worker closure/integrity failed; all errors preserved')
    started=time.perf_counter()
    proof=(verification.audit_merged_game(initial,rows,ending,spec,context,freeze,boundary=envelope.guard) if resumed else
        audit.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard))
    independent+=time.perf_counter()-started;c.save(folder/'online_replay.json',proof)
    waits=[c.artifact(p) for p in sorted((c.OUT/'memory_admissions').glob('*.json'))
        if first_admission<int(p.stem)<=envelope.admissions]
    waited=sum(c.read(c.ROOT/p['path'])['wait_seconds'] for p in waits);gross=time.perf_counter()-begun
    c.require(0<=waited<=gross,'Recorded RAM wait fits new-segment wall time')
    c.save(folder/'cost.json',dict(independent_seconds=independent,collection_seconds=gross,
        explicit_inter_request_ram_wait_seconds=waited,nonwait_collection_seconds=gross-waited,
        memory_admission_first=first_admission,memory_admission_last=envelope.admissions,memory_wait_artifacts=waits,
        complete_decisions=sum(r['completed'] for r in rows[start_ply:]),retained_prior_decisions=start_ply,
        measured_two_pass_hash_seconds=None,segmented_game=resumed))
    return ending,rows,independent,fault


def verify_reproductions(stage,candidate,records,envelope,freeze,context,*,run_models):
    from . import audit,verification
    sample=audit.select_replay_sample(records,maximum=48,stage=stage,candidate=candidate)
    path=c.OUT/'replay_samples'/stage/f'{candidate}.json'
    save_or_verify(path,sample,run_models);repeated=[]
    for index,item in enumerate(sample['decisions']):
        source=verification.source_move_path(item,item['ply'],context);original=c.read(source)
        c.require(all(original[k]==item[k] for k in ('stage','candidate','pair_index','game_index','ply','player_role')),
            'Exact selected immutable original or resumed decision')
        row=(reproduce_decision(original,index,stage,candidate,envelope,freeze) if run_models else
            c.read(c.OUT/'repeated_decisions'/stage/candidate/f'{index:02d}.json'))
        verify_saved_decision(row,freeze)
        c.require(row['completed'] and row['original_identity']=={k:original[k] for k in ('stage','candidate','pair_index','game_index','ply','player_role')} and
            all(row[k]==original[k] for k in ('action','trace','after')) and row['clock']['result']==original['clock']['result'],
            'Exact complete scientific reproduction, regardless of source segment')
        folder=c.OUT/'workers'/replay_worker_name(stage,candidate,index,row['route'])
        c.require({p.stem for p in (folder/'inputs').glob('*.json')}=={row['clock']['request_id']},'Exactly one reproduction request')
        proof=transport.verify_closed_worker(folder,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{folder.name}.json'),'Reproduction full closure readback')
        repeated.append(row)
    c.require([p.name for p in sorted((c.OUT/'repeated_decisions'/stage/candidate).glob('*.json'))]==
        [f'{i:02d}.json' for i in range(len(repeated))],'Exact reproduction file inventory')
    return dict(status='passed',requests=len(repeated),sample=c.artifact(path),workers=[Path(r['worker_folder']).name for r in repeated])


def verify_phase(stage,specs,envelope,freeze,context,*,run_models):
    from . import verification
    c.require(stage=='main','Only the unchanged main assessment is resumed')
    games,rows,proofs=verification.read_games(specs,envelope,freeze,context)
    repeats=verify_reproductions(stage,'research',rows,envelope,freeze,context,run_models=run_models)
    verify_runtime(freeze,envelope)
    result=dict(status='passed',stage=stage,games=len(games),moves=len(rows),closed_game_workers=len(proofs),
        reproductions=[repeats],full_ledger_raw_transport_logs_graph_clock_placement_and_source_checks=True,
        original_attempt_remains_incomplete=True,one_disclosed_retry=True)
    target=c.OUT/'phase_verification/main.json';save_or_verify(target,result,run_models)
    if run_models:
        paths=[c.OUT/'games/main',c.OUT/'repeated_decisions/main',c.OUT/'replay_samples/main',target,
            c.ROOT/freeze['recovery_manifest']['path']]
        names={game_worker_name(s,r) for s in specs if s['game_index']>=235 for r in ('challenger','opponent')}
        names.update(repeats['workers'])
        paths += [c.OUT/'workers'/name for name in sorted(names)]
        paths += [c.OUT/'closed_workers'/f'{name}.json' for name in sorted(names)]
        c.save(c.OUT/'phase_seals/main.json',dict(artifacts=evidence_seal(paths,envelope),
            immutable_prior_inventory=context['manifest']['original_tree']))
    else:
        seal=c.read(c.OUT/'phase_seals/main.json')
        c.require(seal['immutable_prior_inventory']==context['manifest']['original_tree'],'Original segment preservation binding')
        for item in seal['artifacts']:envelope.guard();c.verify([item])
    return games,rows,result


def recovery_projection(inputs):
    """Outcome-blind whole-work forecast on the unchanged original clocks."""
    validate_original_origins(inputs)
    games=inputs['game_costs'];workers=inputs['worker_costs'];hashes=inputs['hash_seconds']
    c.require(len(games)==6 and len(workers)==12 and len(hashes)==6,'Six frozen operational cost measurements')
    numbers=[inputs[k] for k in ('runtime_seconds','ram_wait_used','current_bytes','current_files','monotonic','unix_time','monotonic_origin','unix_origin')]
    numbers += [g[k] for g in games for k in ('seconds','independent_seconds','bytes','files')]+hashes
    numbers += [w[k] for w in workers for k in ('initialization','graph','closure')]
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in numbers),'Finite nonnegative cost observations')
    c.require(type(inputs['prior_files']) is int and inputs['prior_files']>=0,'Bounded prior-file finalization cost')
    c.require(c.PRIOR_RAM_WAIT_SECONDS<=inputs['ram_wait_used']<=c.RAM_WAIT_SECONDS,'Prior cumulative RAM wait is charged')
    maximum=lambda key:max(g[key] for g in games)
    startup=max(w['initialization']+w['graph']+w['closure'] for w in workers)
    wait=c.RAM_WAIT_SECONDS-inputs['ram_wait_used']
    replay=1.5*48*(startup+c.SECONDS_PER_MOVE+1.)
    main=1.5*165*maximum('seconds')
    verification=1.5*400*maximum('independent_seconds')+replay+600.
    # Four bounded startup requests are separately charged, even when this is
    # recomputed after qualification; this conservative reuse cannot relax admission.
    qualification=2*(60.+60.+300.)  # initialization, graph parity, actual suite
    collection=main+verification+wait+qualification+1200.
    final=1.5*406*(maximum('independent_seconds')+max(hashes))+2*inputs['runtime_seconds']+c.INDEPENDENT_REVIEW_SECONDS+600.
    prefinal=inputs['current_bytes']+math.ceil(1.5*165*maximum('bytes')+1.5*24*maximum('bytes'))
    files=inputs['current_files']+math.ceil(1.5*(165+24)*maximum('files'))
    final_bytes=math.ceil((files+inputs['prior_files'])*2048+128*c.MiB)
    available_main=min(c.ORIGINAL_MAIN_MONOTONIC_DEADLINE-inputs['monotonic'],c.ORIGINAL_MAIN_UNIX_DEADLINE-inputs['unix_time'])
    available_model=min(inputs['monotonic_origin']+c.COLLECTION_CUTOFF_SECONDS-inputs['monotonic'],inputs['unix_origin']+c.COLLECTION_CUTOFF_SECONDS-inputs['unix_time'])
    available_global=min(inputs['monotonic_origin']+c.JOB_SECONDS-inputs['monotonic'],inputs['unix_origin']+c.JOB_SECONDS-inputs['unix_time'])
    passed=(main+wait+qualification<available_main and verification<c.STAGES['main_verification'] and
        collection<available_model and final<c.STAGES['final'] and collection+final<available_global and
        prefinal<c.PREFINAL_EVIDENCE_BYTES and final_bytes<c.FINAL_RESERVE_BYTES and prefinal+final_bytes<c.MAX_EVIDENCE_BYTES)
    return dict(passed=passed,measurement='Maximum of6 original operational games;1.5 contingency;165 full game equivalents',
        remaining_games=165,retained_completed_games=235,replay_decisions_maximum=48,all_main_saved_readbacks=400,
        projected_main_seconds=main,projected_verification_seconds=verification,projected_collection_seconds=collection,
        projected_saved_final_seconds=final,projected_prefinal_bytes=prefinal,projected_final_bytes=final_bytes,
        projected_new_files=files,available_main_seconds=available_main,available_model_seconds=available_model,
        available_global_seconds=available_global,remaining_ram_wait_seconds=wait,
        main_deadline_not_reset=True,timing_admission_thresholds_used=False,outcomes_used=False)


def forecast_recovery(envelope,freeze,context,*,name):
    c.require(name in ('preflight','qualified'),'Exactly two prospective recovery forecasts')
    old=c.read(c.OLD_OUT/'assessment_gate.json')['inputs']
    prior=c.read(c.ROOT/context['manifest']['original_tree']['path'])
    inputs=dict(game_costs=old['game_costs'],worker_costs=old['worker_costs'],hash_seconds=old['hash_seconds'],
        runtime_seconds=envelope.source_runtime_seconds,ram_wait_used=envelope.waited,
        current_bytes=envelope.accounted_bytes(),current_files=sum(1 for p in c.OUT.rglob('*') if p.is_file()),
        prior_files=prior['file_count'],monotonic=time.monotonic(),unix_time=time.time(),
        monotonic_origin=envelope.start,unix_origin=envelope.wall_start,
        old_assessment_gate=c.artifact(c.OLD_OUT/'assessment_gate.json'))
    value=dict(recovery_projection(inputs),inputs=inputs)
    pin=c.save(c.OUT/'recovery_forecasts'/f'{name}.json',value)
    c.require(value['passed'],'Remaining full recovery workload does not fit original clocks/storage')
    return pin


def run(expected_sha):
    from . import verification
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact separately frozen recovery')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Owning watchdog freeze')
    owner=c.read(c.OUT/'guardian_claim.json');validate_original_origins(owner)
    c.require(owner['freeze_sha256']==expected_sha,'Actual guardian freeze identity')
    claim=dict(claim,owner_started_monotonic=owner['owner_started_monotonic'],owner_started_unix_time=owner['owner_started_unix_time'])
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json')
    envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha,phase=c.PHASE))
    games=[];coverage=None;failure=None;status='incomplete';workers={};context=None
    try:
        envelope.begin('recovery_qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        begun=time.perf_counter();verify_runtime(freeze,envelope);envelope.source_runtime_seconds=time.perf_counter()-begun
        c.save(c.OUT/'source_runtime_measurement.json',dict(seconds=envelope.source_runtime_seconds))
        schedules,plan=load_inputs(freeze,envelope);context=verification.load_context(freeze,envelope)
        verification.verify_prior_admission(context,envelope);prefix=verification.verify_prefix(context,envelope)
        forecast_recovery(envelope,freeze,context,name='preflight')
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        spec=schedules['main'][235];qualify_retained_workers(spec,envelope,freeze,plan,workers)
        envelope.finish();envelope.begin('main');transition_retained_workers(workers,envelope)
        gate=forecast_recovery(envelope,freeze,context,name='qualified')
        c.save(c.OUT/'recovery_main_admission.json',dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),
            perf_counter_ns=time.perf_counter_ns(),freeze=c.artifact(c.OUT/'freeze.json'),
            qualification=c.artifact(c.OUT/'recovery_qualification_summary.json'),forecast=gate,
            original_main_admission=c.artifact(c.OLD_OUT/'main_admission.json'),
            retained_completed_games=235,retained_prefix_moves=30,remaining_games=165,
            maximum_disclosed_retries=1,automatic_retries=0,no_optional_stopping=True))
        outcome=play_game(spec,envelope,freeze,context,prefix=prefix,retained_workers=workers)
        workers={};games.append(outcome[0])
        print(c.canonical(dict(stage='main',completed_games=236,total=400,one_disclosed_retry_recorded=True)),flush=True)
        for spec in schedules['main'][236:]:
            game,rows,cost,fault=play_game(spec,envelope,freeze,context);games.append(game)
            print(c.canonical(dict(stage='main',completed_games=235+len(games),total=400,hard_guard_forfeit=fault is not None)),flush=True)
        envelope.finish();envelope.begin('main_verification')
        verify_phase('main',schedules['main'],envelope,freeze,context,run_models=True);envelope.finish()
        print(c.canonical(dict(stage='main_verification',status='verified')),flush=True)
        envelope.begin('final');coverage=verification.final_checks(envelope,freeze,context=context)
        envelope.finish();status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        for role,child in workers.items():
            if child._closed:continue
            try:close_and_audit(child,envelope,freeze)
            except BaseException:
                c.save(c.OUT/'recovery_cleanup_failures'/f'{role}.json',dict(time=c.now(),error=traceback.format_exc()))
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
        print(c.canonical(dict(stage=envelope.stage,status='failed_preserved',error=failure)),flush=True)
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,coverage=coverage,freeze_sha256=expected_sha,
        phase=c.PHASE,elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        maximum_seconds=c.JOB_SECONDS,newly_completed_games=len(games),retained_completed_main_games=235,
        automatic_retries=0,disclosed_retry_recorded=(c.OUT/'disclosed_retry.json').exists(),training=False,
        amended_recovery=True,prior_attempt_status='incomplete',
        inherited_original_monotonic_origin=envelope.start,inherited_original_unix_origin=envelope.wall_start,
        requires_clean_actual_os_exits=['runner','watchdog','independent_reader','guardian']))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True)
    return run(parser.parse_args().freeze_sha256)


if __name__=='__main__':raise SystemExit(main())

