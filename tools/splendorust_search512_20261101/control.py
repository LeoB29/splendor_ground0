"""One-clock fixed512-versus256 study with bounded block ownership and saved verification."""
import argparse,os,sys,time,traceback,shutil,math,re
from copy import deepcopy
from pathlib import Path
from importlib.metadata import version
import psutil
from . import common as c,transport,adapters
from tools.splendorust_search256_recovery_20261030_02 import placement
from . import qualification,scientific
from . import analysis,design
from tools.splendorust_search256_efficiency_20261031.common import validate_exchange
from tools.future_training_rules_20261021 import verify_runtime_sources,decode_training_snapshot
from tools.external_strength_screen_20261018 import rules,audit as ledger
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules

def bytes_used():
    return sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())


class Envelope:
    """One non-resettable job clock, aggregate resident-worker resource bound."""
    def __init__(self, claim):
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.recovery_owner_start=claim.get('owner_started_monotonic')
        self.recovery_owner_wall_start=claim.get('owner_started_unix_time')
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
        from tools.splendorust_search512_accounting_20261101 import WorkerAccounting
        c.require(self.prior_worker_count+len(self.worker_accounting)<c.MAX_WORKER_LIFECYCLES,
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
        if self.worker_accounting[folder] is None:
            c.require(child is None and self.evidence['workers'][folder] is not None,
                'Only a fully verified closed lifecycle may use its compact tombstone')
            return
        receipt = self.worker_accounting[folder].close(boundary=self.time_guard,
            lock_timeout=min(1.,max(0.,self.remaining())))
        self.evidence['workers'][folder] = receipt['accounted_bytes']

    def release_verified_worker(self,folder):
        c.require(self.evidence['workers'][folder] is not None and
            (c.OUT/'closed_workers'/f'{folder.name}.json').is_file(),
            'Release accounting memory only after complete verified owned closure')
        c.require(not any(p.pid==self.worker_accounting[folder]._child.process.pid for p in self.resident.values()),
            'Cannot release a resident worker')
        self.worker_accounting[folder]=None

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
        c.atomic_save(c.OUT/'stages'/f'{stage}_start.json', dict(time=c.now(), monotonic=mono, unix_time=wall,
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
        if not self.admitted:
            c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial frozen free-disk reserve')
        begun,wall_begun,wait_before = time.monotonic(),time.time(),self.waited
        threshold = c.MINIMUM_AVAILABLE_RAM_BYTES if self.admitted else c.INITIAL_AVAILABLE_RAM_BYTES
        minimum=available=None
        try:
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
                    cumulative_wait_seconds=self.waited, minimum_available_bytes=minimum,stable_seconds=60))
        except BaseException as original:
            ended,wall_ended=time.monotonic(),time.time()
            self.waited=max(self.waited,wait_before+ended-begun)
            failure=dict(time=c.now(),stage=self.stage,error=traceback.format_exc(),
                monotonic_start=begun,monotonic_end=ended,unix_start=wall_begun,unix_end=wall_ended,
                wait_seconds=ended-begun,prior_cumulative_wait_seconds=wait_before,
                cumulative_wait_seconds=self.waited,maximum_wait_seconds=c.RAM_WAIT_SECONDS,
                minimum_available_bytes=minimum,last_available_bytes=available,required_available_bytes=threshold,
                decision_input_not_published=True,decision_clock_not_started=True)
            try:c.save(c.OUT/'memory_wait_failure.json',failure)
            except BaseException as preservation_error:
                original.add_note('Could not save bounded RAM-wait failure receipt: '+repr(preservation_error))
            raise

    def finish(self):
        self.last_resource_guard=-float("inf")
        self.guard()
        if self.evidence is not None:
            c.require(bytes_used() <= self.accounted_bytes(), 'Unaccounted study evidence')
        c.atomic_save(c.OUT/'stages'/f'{self.stage}_end.json', dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time()))


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
    ack=c.read(ackpath);validate_exchange(request,ack,freeze,watchdog,identity)
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
        validate_exchange(r,a,freeze,watchdog,identity)
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


def verify_runtime(freeze, envelope=None):
    validate_freeze_contract(freeze)
    from tools.matched_strength_20261020 import statistics as numerical_statistics
    c.require(numerical_statistics.runtime_contract()==freeze['statistics_runtime'],'Pinned statistics numerical runtime changed')
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python runtime changed')
    placement.verify_host(freeze)
    placement.verify_runtime(freeze,'isolated')
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


def save_or_verify(path,value,writing):
    if writing:return c.save(path,value)
    c.require(c.read(path)==value,'Independent saved result differs: '+str(path));return c.artifact(path)


def lifecycle_inventory():
    inventory={}
    for stage,count in [('operational',c.OPERATIONAL_GAMES),('main',c.MAIN_GAMES)]:
        for index in range(count):
            remaining=(count-index if stage=='operational' else min(c.GAMES_PER_BLOCK-index%c.GAMES_PER_BLOCK,count-index))
            for route in c.VARIANTS:
                inventory[f'{stage}_g{index:03d}_{route}']=dict(route=route,quota=c.ROUTE_QUOTAS[route],
                    graph_request_id='request-00000001',entrypoint_request_id='request-00000002',
                    ordinary_first_request_id='request-00000003',ordinary_requests=remaining*c.MAX_PLIES//2,startup_requests=2)
    for route in c.VARIANTS:
        inventory['replay_'+route]=dict(route=route,quota=c.ROUTE_QUOTAS[route],graph_request_id='request-00000001',
            entrypoint_request_id='request-00000002',ordinary_first_request_id='request-00000003',
            ordinary_requests=c.MAX_REPLAY_DECISIONS,startup_requests=2)
    return inventory


def contract():
    return dict(schema='splendorust_search512_strength_v1',phase=c.PHASE,
        job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,stage_seconds=c.STAGES,
        independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,automatic_retries=0,failed_decision_retries=0,
        seconds_per_move=c.SECONDS_PER_MOVE,internal_seconds=c.INTERNAL_SECONDS,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,max_rss_bytes=c.MAX_RSS_BYTES,max_workers=c.MAX_WORKER_LIFECYCLES,
        initial_available_ram_bytes=c.INITIAL_AVAILABLE_RAM_BYTES,minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,
        initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=c.MINIMUM_FREE_DISK_BYTES,
        placement_mode='isolated',max_resident_workers=2,routes=dict(c.ROUTE_QUOTAS),
        declared_worker_lifecycles=lifecycle_inventory(),qualification_scientific_decisions_maximum=c.MAX_SCIENTIFIC_QUALIFICATION,
        graph_cases_maximum=c.MAX_GRAPH_CASES,maximum_live_decisions=c.MAX_LIVE_DECISIONS,
        maximum_replay_decisions=c.MAX_REPLAY_DECISIONS,maximum_plies_per_game=c.MAX_PLIES,
        parent_requests_maximum=c.MAX_LIVE_DECISIONS+c.MAX_REPLAY_DECISIONS+2*c.MAX_WORKER_LIFECYCLES,
        games_per_main_block=c.GAMES_PER_BLOCK,operational_games=c.OPERATIONAL_GAMES,main_games=c.MAIN_GAMES,
        main_pairs=200,new_games=c.MAX_GAMES,training=False,one_second_gate=False,seven_hundred_ms_gate=False,
        only_scientific_change='simulations_256_to_512',network_execution='unchanged_original_full_token_graph',
        world_pool=3,cpuct=.4,search_depth=64,max_response_bytes=c.MAX_RESPONSE_BYTES,max_raw_log_bytes=c.MAX_LOG_BYTES,
        source_checkpoint_unchanged=True,numerical_tolerances_used=False,graph_exact_outputs_required=True,
        main_classified_hard_guard_is_loss=True,continue_fixed_main_sample_after_guard_loss=True,
        stop_operational_on_any_failure=True,stop_on_infrastructure_failure=True,game_replacements=0,
        worker_restart_only_before_next_unplayed_game=True,full_qualification_each_lifecycle=True,
        forecast_factor=1.5,forecast_nominal_main_worker_pairs=40,forecast_remaining_ram_wait_reserved=True,
        forecast_saved_final_reserve_seconds=c.STAGES['final'],
        replay_full_result_exact_within_route=True,strength_promotion=False)


def validate_freeze_contract(freeze):
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()),'Exact frozen512 research contract')
    c.require(set(freeze['profiles'])==set(c.VARIANTS),'Only unchanged256 and fixed512 profiles')
    return expected


def load_inputs(freeze,envelope):
    c.verify([freeze[k] for k in ('schedule','qualification_plan','replay_rule','initial_states')])
    schedule=c.read(c.ROOT/freeze['schedule']['path']);design.validate_schedule(schedule)
    c.require(c.read(c.ROOT/freeze['replay_rule']['path'])==design.replay_rule(),'Exact blind reproduction selection rule')
    initials=c.read(c.ROOT/freeze['initial_states']['path'])
    c.require(initials==[dict(**s,initial=design.initial_snapshot(s)) for s in schedule],'Corrected paired initial states')
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(plan==qualification.build_plan(),'Actual corrected512/256 entrypoint plan unchanged')
    return schedule,plan


def route_for(name):
    c.require(name in lifecycle_inventory(),'Declared exact lifecycle')
    return lifecycle_inventory()[name]['route']


def qualification_evidence(name,envelope,freeze,plan,*,writing):
    folder=c.OUT/'workers'/name;route=route_for(name)
    graph=transport.verify_graph_evidence(folder,freeze,boundary=envelope.guard)
    clock=c.read(folder/'entrypoint_clock.json');profiles=c.read(folder/'entrypoint_profiles.json')
    transport._require_profiles(profiles)
    transport.validate_profiles(profiles['parent_profile'],profiles['child_profile'],clock,{'operation':'entrypoint_checks'},route)
    suite=qualification.verify_entrypoint(folder,clock,plan,freeze,boundary=envelope.guard)
    historical=qualification.compare_control_parity(clock['result'],plan,boundary=envelope.guard) if route=='control' else None
    first='operational_g000_'+route
    if name!=first:
        c.require(clock['result']==c.read(c.OUT/'workers'/first/'entrypoint_clock.json')['result'],
            'Every lifecycle repeats the same route full scientific suite exactly')
    result=dict(status='qualified',lifecycle=name,route=route,quota=c.ROUTE_QUOTAS[route],graph=graph,suite=suite,
        control_historical_parity=historical,graph_cases=11,scientific_decisions=67,parent_requests=2)
    save_or_verify(c.OUT/'qualification'/f'{name}.json',result,writing);return result


def qualify(name,workers,envelope,freeze,plan):
    begun=time.monotonic();wait_before=envelope.waited;bytes_before=envelope.accounted_bytes()
    child=transport.make_worker(name,envelope,freeze,route=route_for(name));workers[name]=child
    c.require(child._counter==1 and envelope.remaining()>=300.,'Full actual suite after graph qualification')
    adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
    proof=qualification_evidence(name,envelope,freeze,plan,writing=True)
    elapsed=time.monotonic()-begun
    c.save(c.OUT/'startup_costs'/f'{name}.json',dict(seconds=elapsed,nonwait_seconds=elapsed-(envelope.waited-wait_before),
        accounted_growth_bytes=envelope.accounted_bytes()-bytes_before,raw_stream_bytes=dict(child._log_sizes)))
    return proof


def qualify_pair(stage,start,workers,envelope,freeze,plan):
    c.require(not workers,'Close preceding pair before creating a new block')
    names=[f'{stage}_g{start:03d}_{route}' for route in c.VARIANTS]
    for name in names:qualify(name,workers,envelope,freeze,plan)
    return names


def advance_idle_worker(child,envelope):
    from tools.splendorust_network_practical_handoff_20261031 import live_worker_identity
    c.require(not child._closed and child._counter==2 and child.process.poll() is None and
        not child._failure and not child._lock.locked(),'Only idle fully qualified two-request startup may hand off')
    old=child.absolute_deadline
    c.require(old<=envelope.start+c.COLLECTION_CUTOFF_SECONDS and envelope.deadline<=envelope.start+c.COLLECTION_CUTOFF_SECONDS,
        'Idle stage handoff retains original model cutoff')
    identity=live_worker_identity(child,common=c)
    receipt=dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),perf_counter_ns=time.perf_counter_ns(),
        lifecycle=child.output_dir.name,worker_identity=identity,completed_startup_requests=2,next_request_id='request-00000003',
        old_absolute_deadline=old,new_absolute_deadline=envelope.deadline,original_monotonic_origin=envelope.start,
        original_unix_origin=envelope.wall_start,active_request=False,request_allowance_unchanged=10.)
    result=c.save(c.OUT/'worker_phase_transitions'/f'{child.output_dir.name}.json',receipt)
    child.absolute_deadline=envelope.deadline;return result


def actual_after(before,action):
    state=decode_training_snapshot(before);environment=rules.CorrectedEnv();chosen=rules.deserialize_action(action)
    c.require(chosen in environment.legal_actions(state),'Corrected complete action legality')
    after=rules.serialize_snapshot(environment.step(state,chosen))
    c.require(after==ledger.transition(before,action),'Actual corrected transition agrees with independent ledger')
    decode_training_snapshot(after);return after


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/f"g{spec['game_index']:03d}"


def source_move_path(selection):
    return c.OUT/'games'/selection['stage']/f"g{selection['game_index']:03d}"/'moves'/f"{selection['ply']:06d}.json"


def decision(child,spec,before,envelope,freeze,*,ply):
    state=decode_training_snapshot(before);observer=rules.observation_payload(state)
    role='challenger' if state.current_player==spec['challenger_seat'] else 'opponent';route=spec[role+'_mode']
    c.require(child.route==route and route_for(child.output_dir.name)==route,'Exact current player route')
    row=transport.ordinary(child,spec,observer,envelope,freeze)
    row.update(**spec,ply=ply,variant=route,player_role=role,arm='search',quota=c.ROUTE_QUOTAS[route],mode=route,
        seat=state.current_player,before=before,after=None)
    if row['completed']:
        try:row['after']=actual_after(before,row['action'])
        except BaseException:row['transition_error']=traceback.format_exc()
    return row


def validate_row(row,spec,freeze,envelope,*,worker_name):
    envelope.guard();design.validate_spec(row);c.require(row['spec']==spec,'Exact game request specification')
    role='challenger' if row['seat']==spec['challenger_seat'] else 'opponent';route=spec[role+'_mode']
    c.require(row['variant']==row['route']==row['mode']==route==route_for(worker_name) and row['player_role']==role and
        row['arm']=='search' and row['quota']==c.ROUTE_QUOTAS[route] and row['seat']==row['before']['state']['current_player'] and
        row['worker_folder']==(c.OUT/'workers'/worker_name).relative_to(c.ROOT).as_posix(),'Exact role,quota,lifecycle and acting seat')
    transport._require_profiles(row);transport.verify_saved_record(row,freeze)
    ledger.check_observer(row['before'],row['observer'])
    c.require(row['observer']==rules.observation_payload(decode_training_snapshot(row['before'])),
        'Actual legal observer and canonical observer-derived random seed')
    c.require('transition_error' not in row,'Failed actual transition remains incomplete')
    if row['completed']:
        c.require(actual_after(row['before'],row['action'])==row['after'],'Actual chosen successor independently reconstructed')
        scientific.validate_result(row['clock']['result'],row['observer'],'search',expected_quota=row['quota'])
        scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',before=row['before'],after=row['after'],expected_quota=row['quota'])
        return None
    c.require(row['after'] is None,'A failed move cannot invent an afterstate')
    return scientific.classify_failure(row['clock'])


class SavedRows:
    """Reiterable disk stream; retain at most one full move, never all406 games."""
    def __init__(self,games,boundary=lambda:None):self.games,self.boundary=games,boundary
    def __iter__(self):
        for game in self.games:
            for ply in range(game['plies']):
                self.boundary()
                yield c.read(game_folder(game)/'moves'/f'{ply:06d}.json')


def current_worker_names(workers):
    c.require(len(workers)==2,'One complete resident pair')
    return {route:next(name for name in workers if route_for(name)==route) for route in c.VARIANTS}


def play_game(spec,workers,envelope,freeze):
    design.validate_spec(spec);folder=game_folder(spec);initial=design.initial_snapshot(spec);names=current_worker_names(workers)
    c.save(folder/'spec.json',spec);c.save(folder/'initial.json',dict(**spec,initial=initial));c.save(folder/'workers.json',names)
    state=decode_training_snapshot(initial);rows=[];begun=time.monotonic();wait_before=envelope.waited
    bytes_before=envelope.accounted_bytes();audit_seconds=0.;outcome=None
    streams_before={name:dict(child._log_sizes) for name,child in workers.items()}
    for ply in range(c.MAX_PLIES):
        if state.terminal:break
        role='challenger' if state.current_player==spec['challenger_seat'] else 'opponent';route=spec[role+'_mode']
        row=decision(workers[names[route]],spec,rules.serialize_snapshot(state),envelope,freeze,ply=ply)
        c.save(folder/'moves'/f'{ply:06d}.json',row);rows.append(row);started=time.monotonic()
        failure=validate_row(row,spec,freeze,envelope,worker_name=names[route]);audit_seconds+=time.monotonic()-started
        if failure:
            outcome=dict(reason='forfeit',winner=1-row['seat'],loser=row['seat'],failed_seat=row['seat'],failed_role=role,
                failure=failure,administrative_forfeit=True,failure_policy='research_hard_guard_9s_internal_10s_complete_response')
            c.save(folder/'captured_failure.json',dict(**spec,ply=ply,clock=c.artifact(c.ROOT/row['worker_clock_file']),**outcome))
            break
        state=decode_training_snapshot(row['after'])
        c.require(state.terminal or not(row['after']==row['before'] or row['action']['action_type']=='PASS' and
            len(rows)>1 and rows[-2]['action']['action_type']=='PASS'),'Unresolved actual-game deadlock remains incomplete')
    if outcome is None:
        c.require(state.terminal,'Historical2000-ply safety cap; no fabricated terminal result')
        outcome=dict(reason='natural',winner=state.winner)
    ending=dict(**spec,**outcome,complete=True,plies=len(rows),final=rules.serialize_snapshot(state))
    c.save(folder/'result.json',ending);started=time.monotonic()
    proof=analysis.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
    audit_seconds+=time.monotonic()-started;c.save(folder/'online_replay.json',proof)
    elapsed=time.monotonic()-begun;waited=envelope.waited-wait_before
    c.require(0<=waited<=elapsed,'Charged intermove RAM waits belong to current game')
    growth=envelope.accounted_bytes()-bytes_before
    c.require(growth>0,'Conservative game evidence growth accounting')
    cost=dict(seconds=elapsed,nonwait_seconds=elapsed-waited,ram_wait_seconds=waited,independent_seconds=audit_seconds,
        accounted_growth_bytes=growth,plies=len(rows),completed_decisions=sum(r['completed'] for r in rows),
        maximum_receipt_seconds=max((r['clock']['receipt_elapsed_seconds'] or 0) for r in rows))
    cost['raw_stream_growth']={route:{key:workers[name]._log_sizes[key]-streams_before[name][key]
        for key in streams_before[name]} for route,name in names.items()}
    c.save(folder/'cost.json',cost)
    print(c.canonical(dict(stage=spec['stage'],game=spec['game_index']+1,total=c.OPERATIONAL_GAMES if spec['stage']=='operational' else c.MAIN_GAMES,
        plies=len(rows),ending=outcome['reason'])),flush=True)
    return ending


def expected_worker_plan(games):
    """Fixed block boundaries, with new trees only after a recorded guard forfeit."""
    result={};start=0;previous=None
    for game in games:
        design.validate_spec(game);stage=game['stage'];index=game['game_index']
        if stage=='operational':start=0
        elif index%c.GAMES_PER_BLOCK==0 or previous is not None and previous['stage']==stage and previous['reason']=='forfeit':start=index
        names={route:f'{stage}_g{start:03d}_{route}' for route in c.VARIANTS}
        result[(stage,index)]=names;previous=game
    return result


def read_games(schedule,envelope,freeze):
    games=[];worker_counts={};previous_by_stage={};expected_start={}
    for spec in schedule:
        envelope.guard();folder=game_folder(spec);initial=design.initial_snapshot(spec);stage=spec['stage'];index=spec['game_index']
        c.require(c.read(folder/'spec.json')==spec and c.read(folder/'initial.json')==dict(**spec,initial=initial),'Exact fresh fixed setup')
        ending=c.read(folder/'result.json');count=ending['plies']
        c.require(type(count)is int and 0<count<=c.MAX_PLIES and
            [p.name for p in sorted((folder/'moves').glob('*.json'))]==[f'{i:06d}.json' for i in range(count)],'Exact contiguous move inventory')
        previous=previous_by_stage.get(stage)
        if stage=='operational':start=0
        elif index%c.GAMES_PER_BLOCK==0 or previous is not None and previous['reason']=='forfeit':start=index
        else:start=expected_start[stage]
        expected_start[stage]=start;names={route:f'{stage}_g{start:03d}_{route}' for route in c.VARIANTS}
        c.require(c.read(folder/'workers.json')==names,'Worker restart only at fixed block/next unplayed game after guard loss')
        rows=[]
        for ply in range(count):
            row=c.read(folder/'moves'/f'{ply:06d}.json');c.require(row['ply']==ply,'Contiguous saved ply')
            name=names[row['variant']];validate_row(row,spec,freeze,envelope,worker_name=name)
            expected=worker_counts.get(name,0)+3;c.require(row['clock']['request_id']==f'request-{expected:08d}','Exact persistent worker input sequence')
            worker_counts[name]=expected-2;rows.append(row)
        proof=analysis.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
        c.require(proof==c.read(folder/'online_replay.json'),'Independent full-game ledger replay')
        if stage=='operational':c.require(ending['reason']=='natural','All six operational games must finish naturally')
        games.append(ending);previous_by_stage[stage]=ending
    c.require(sum(g['plies'] for g in games)<=c.MAX_LIVE_DECISIONS,'Aggregate live decision ceiling')
    return games,SavedRows(games,envelope.guard),worker_counts



def replay_selected(selection,workers,envelope,freeze,schedule):
    specs={(s['stage'],s['game_index']):s for s in schedule}
    for index,item in enumerate(selection['decisions']):
        envelope.guard();path=source_move_path(item);original=c.read(path);spec=specs[(item['stage'],item['game_index'])]
        name='replay_'+item['variant'];row=decision(workers[name],spec,original['before'],envelope,freeze,ply=item['ply'])
        row.update(source_move=c.artifact(path),selection=deepcopy(item))
        folder=c.OUT/'replays'/f'r{index:03d}';c.save(folder/'selection.json',item);c.save(folder/'decision.json',row)
        c.require(validate_row(row,spec,freeze,envelope,worker_name=name) is None,'Any failed reproduction remains incomplete')
        proof=analysis.verify_replay(original,row,item);c.save(folder/'verification.json',proof)
        print(c.canonical(dict(stage='replay',completed=index+1,total=len(selection['decisions']))),flush=True)


def verify_replays(selection,envelope,freeze,schedule):
    specs={(s['stage'],s['game_index']):s for s in schedule};counts={};decisions=selection['decisions']
    c.require(0<len(decisions)<=c.MAX_REPLAY_DECISIONS and
        {p.name for p in(c.OUT/'replays').iterdir() if p.is_dir()}=={f'r{i:03d}' for i in range(len(decisions))},
        'Exact outcome-blind selected reproduction inventory')
    for index,item in enumerate(decisions):
        envelope.guard();folder=c.OUT/'replays'/f'r{index:03d}';row=c.read(folder/'decision.json');path=source_move_path(item)
        c.require(c.read(folder/'selection.json')==item and row['selection']==item and row['source_move']==c.artifact(path),
            'Reproduction binds immutable exact source decision')
        name='replay_'+item['variant'];expected=counts.get(name,0)+3
        c.require(row['clock']['request_id']==f'request-{expected:08d}','Exact replay request sequence')
        c.require(validate_row(row,specs[(item['stage'],item['game_index'])],freeze,envelope,worker_name=name) is None,
            'Failed reproduction cannot qualify the study')
        proof=analysis.verify_replay(c.read(path),row,item)
        c.require(proof==c.read(folder/'verification.json'),'Independent exact fresh-process reproduction')
        counts[name]=expected-2
    return counts


def phase_admission(envelope,name,required_seconds,**extra):
    c.require(envelope.remaining()>required_seconds,'Remaining original phase/model allowance for bounded admission')
    return c.save(c.OUT/(name+'_admission.json'),dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),
        freeze_sha256=c.file_hash(c.OUT/'freeze.json'),original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start,
        effective_monotonic_deadline=envelope.deadline,effective_unix_deadline=envelope.wall_deadline,
        required_seconds=required_seconds,automatic_retries=0,**extra))


def verify_admission(envelope,name,required_seconds,**extra):
    value=c.read(c.OUT/(name+'_admission.json'));start=c.read(c.OUT/'stages'/f'{name}_start.json')
    c.require(value['freeze_sha256']==c.file_hash(c.OUT/'freeze.json') and
        value['original_monotonic_origin']==envelope.start and value['original_unix_origin']==envelope.wall_start and
        value['required_seconds']==required_seconds and value['automatic_retries']==0 and
        all(value.get(k)==v for k,v in extra.items()),'Original phase admission identity')
    for key,deadline,origin in [('monotonic','monotonic_deadline',envelope.start),('unix_time','unix_deadline',envelope.wall_start)]:
        c.require(start[key]<=value[key] and value[key]+required_seconds<value['effective_'+deadline]==start[deadline] and
            start[deadline]<=origin+c.COLLECTION_CUTOFF_SECONDS,'Complete original-clock remaining allowance recomputes')
    return value


def forecast_projection(costs,startups,closures,snapshot):
    """Outcome-blind prospective accounting. Late moves never affect a strength gate."""
    c.require(len(costs)==6 and set(startups)==set(closures)==set(c.VARIANTS),'Complete operational cost basis')
    for value in costs:
        c.require(0<=value['ram_wait_seconds']<=value['seconds'] and value['nonwait_seconds']>0 and
            abs(value['seconds']-value['nonwait_seconds']-value['ram_wait_seconds'])<1e-5 and
            value['accounted_growth_bytes']>0 and value['completed_decisions']==value['plies']>0,
            'Complete natural-game timing and evidence costs')
    for route in c.VARIANTS:
        c.require(startups[route]['nonwait_seconds']>0 and startups[route]['accounted_growth_bytes']>0 and closures[route]['seconds']>0,
            'Full startup and actual closure costs')
    factor=1.5;pair_start=sum(v['nonwait_seconds'] for v in startups.values());pair_close=sum(v['seconds'] for v in closures.values())
    waits=max(0.,c.RAM_WAIT_SECONDS-snapshot['charged_ram_wait_seconds'])
    c.require(0<=snapshot['charged_ram_wait_seconds']<c.RAM_WAIT_SECONDS,'Charged original RAM wait')
    main_nonwait=factor*(c.MAIN_GAMES*max(v['nonwait_seconds'] for v in costs)+40*(pair_start+pair_close))
    main=main_nonwait+waits
    replay=factor*(pair_start+pair_close)+c.MAX_REPLAY_DECISIONS*c.SECONDS_PER_MOVE+300.
    # Reserve the entire saved-only stage, including the bounded independent reader.
    final=c.STAGES['final']
    model=main+replay;total=model+final
    pair_bytes=sum(v['accounted_growth_bytes'] for v in startups.values())
    per_move=max(v['accounted_growth_bytes']/v['completed_decisions'] for v in costs)
    additional=factor*(c.MAIN_GAMES*max(v['accounted_growth_bytes'] for v in costs)+40*pair_bytes+
        pair_bytes+c.MAX_REPLAY_DECISIONS*per_move)
    raw={};replay_raw={};checks={}
    for route in c.VARIANTS:
        raw[route]={}
        replay_raw[route]={}
        for stream,initial in startups[route]['raw_stream_bytes'].items():
            projected=initial+factor*c.GAMES_PER_BLOCK*max(v['raw_stream_growth'][route][stream] for v in costs)
            raw[route][stream]=projected;checks[f'raw_{route}_{stream}']=projected<c.MAX_LOG_BYTES
            replay_projected=initial+factor*c.MAX_REPLAY_DECISIONS*max(
                v['raw_stream_growth'][route][stream]/max(1,v['completed_decisions']//2) for v in costs)
            replay_raw[route][stream]=replay_projected
            checks[f'replay_raw_{route}_{stream}']=replay_projected<c.MAX_LOG_BYTES
    checks.update(main_stage=main<c.STAGES['main'],replay_stage=replay<c.STAGES['replay'],
        final_stage=final<=c.STAGES['final'],main_remaining=main<snapshot['main_remaining_seconds'],
        model_remaining=model<snapshot['model_remaining_seconds'],global_remaining=total<snapshot['global_remaining_seconds'],
        prefinal_evidence=snapshot['accounted_bytes']+additional<c.PREFINAL_EVIDENCE_BYTES,
        total_evidence=snapshot['accounted_bytes']+additional+c.FINAL_RESERVE_BYTES<c.MAX_EVIDENCE_BYTES,
        free_disk=additional+c.FINAL_RESERVE_BYTES+c.MINIMUM_FREE_DISK_BYTES<snapshot['free_disk_bytes'])
    return dict(passed=all(checks.values()),checks=checks,main_seconds=main,main_nonwait_seconds=main_nonwait,
        remaining_ram_reserve_seconds=waits,replay_seconds=replay,saved_final_seconds=final,
        model_seconds=model,total_seconds=total,additional_prefinal_bytes=additional,raw_stream_projections=raw,
        replay_raw_stream_projections=replay_raw,
        games=400,normal_main_worker_pairs=40,reproduction_workers=2,maximum_reproductions=96,
        growth_factor=factor,forfeit_replacements_charged_to_unchanged_actual_caps=True)


def assessment_gate(envelope,*,writing):
    cost_paths=[game_folder(s)/'cost.json' for s in design.make_schedule('operational')]
    startup_paths={v:c.OUT/'startup_costs'/f'operational_g000_{v}.json' for v in c.VARIANTS}
    closure_paths={v:c.OUT/'closed_worker_costs'/f'operational_g000_{v}.json' for v in c.VARIANTS}
    inputs=[c.artifact(p) for p in cost_paths+list(startup_paths.values())+list(closure_paths.values())]
    if writing:
        mono,wall=time.monotonic(),time.time()
        snapshot=dict(monotonic=mono,unix_time=wall,original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start,
            main_remaining_seconds=envelope.remaining(),
            model_remaining_seconds=min(envelope.start+c.COLLECTION_CUTOFF_SECONDS-mono,envelope.wall_start+c.COLLECTION_CUTOFF_SECONDS-wall),
            global_remaining_seconds=min(envelope.start+c.JOB_SECONDS-mono,envelope.wall_start+c.JOB_SECONDS-wall),
            charged_ram_wait_seconds=envelope.waited,accounted_bytes=envelope.accounted_bytes(),free_disk_bytes=shutil.disk_usage(c.ROOT).free)
    else:
        saved=c.read(c.OUT/'assessment_gate.json');snapshot=saved['snapshot'];c.require(saved['inputs']==inputs,'Sealed operational forecast basis')
        c.require(snapshot['original_monotonic_origin']==envelope.start and snapshot['original_unix_origin']==envelope.wall_start,
            'Forecast retains original owner clocks')
        stage=c.read(c.OUT/'stages/main_start.json')
        for key,cap in [('model_remaining_seconds',c.COLLECTION_CUTOFF_SECONDS),('global_remaining_seconds',c.JOB_SECONDS)]:
            expected=min(envelope.start+cap-snapshot['monotonic'],envelope.wall_start+cap-snapshot['unix_time'])
            c.require(abs(expected-snapshot[key])<.1,'Forecast remaining original-clock arithmetic')
        expected=min(stage['monotonic_deadline']-snapshot['monotonic'],stage['unix_deadline']-snapshot['unix_time'])
        c.require(abs(expected-snapshot['main_remaining_seconds'])<.1,'Forecast frozen main-stage arithmetic')
    projection=forecast_projection([c.read(p) for p in cost_paths],{k:c.read(p) for k,p in startup_paths.items()},
        {k:c.read(p) for k,p in closure_paths.items()},snapshot)
    result=dict(schema='search512_operational_forecast_v1',inputs=inputs,snapshot=snapshot,projection=projection,
        outcome_blind=True,no_timing_strength_selection=True)
    save_or_verify(c.OUT/'assessment_gate.json',result,writing);return result


def begin_block(spec,workers,envelope,freeze,plan):
    c.require(spec['stage']=='main' and not workers,'New main block only with no resident previous pair')
    index=spec['game_index'];previous=None if index==0 else c.artifact(game_folder(dict(stage='main',game_index=index-1))/'result.json')
    if index%c.GAMES_PER_BLOCK:
        c.require(c.read(c.ROOT/previous['path'])['reason']=='forfeit','Only a classified prior guard loss opens a midblock lifecycle')
    record=dict(stage='main',start_game_index=index,monotonic=time.monotonic(),unix_time=time.time(),
        original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start,
        effective_monotonic_deadline=envelope.deadline,effective_unix_deadline=envelope.wall_deadline,
        previous_game=previous,reason='fixed_block' if index%c.GAMES_PER_BLOCK==0 else 'next_unplayed_after_guard_loss',
        assessment_gate=c.artifact(c.OUT/'assessment_gate.json'),no_retried_decision=True,
        previous_closed_workers=prior_pair_closures('main',index))
    c.save(c.OUT/'block_admissions'/f'g{index:03d}.json',record)
    return qualify_pair('main',index,workers,envelope,freeze,plan)


def prior_pair_closures(stage,index):
    prior=dict(stage='operational' if stage=='main' and index==0 else 'main',
        game_index=5 if stage=='main' and index==0 else index-1)
    names=c.read(game_folder(prior)/'workers.json')
    return [c.artifact(c.OUT/'workers'/names[route]/'cleanup.json') for route in c.VARIANTS]


def verify_closed_before(descriptors,wall):
    from datetime import datetime
    c.require(len(descriptors)==2 and len({d['path'] for d in descriptors})==2,'Exact prior pair cleanup receipts')
    for item in descriptors:
        c.verify([item]);value=c.read(c.ROOT/item['path'])
        c.require(not value['alive'] and not value['errors'] and
            datetime.fromisoformat(value['time']).timestamp()<=wall,'Both preceding trees closed before next process admission')


def verify_phase_transition(name,envelope):
    transition=c.read(c.OUT/'worker_phase_transitions'/f'{name}.json');folder=c.OUT/'workers'/name
    before=c.read(c.OUT/'stages/qualification_start.json');after=c.read(c.OUT/'stages/operational_start.json')
    admit=c.read(c.OUT/'operational_admission.json');init=c.read(folder/'initialization.json')
    suite=c.read(folder/'parent_profiles/request-00000002.json');first=c.read(folder/'parent_profiles/request-00000003.json')
    c.require(transition['lifecycle']==name and transition['worker_identity']==init['identity'] and
        transition['completed_startup_requests']==2 and transition['next_request_id']=='request-00000003' and
        transition['active_request'] is False and transition['request_allowance_unchanged']==10. and
        transition['original_monotonic_origin']==envelope.start and transition['original_unix_origin']==envelope.wall_start,
        'Idle phase transition identity/allowance')
    c.require(abs(transition['old_absolute_deadline']-before['monotonic_deadline'])<.1 and
        transition['new_absolute_deadline']==after['monotonic_deadline']<=envelope.start+c.COLLECTION_CUTOFF_SECONDS,
        'Retained worker deadline clamped to original stage/model clock')
    c.require(admit['monotonic']<=transition['monotonic'] and
        suite['capture_snapshot_ns']<=transition['perf_counter_ns']<=first['request_start_ns'],
        'Completed suite before idle transition before first ordinary clock')
    return transition


def verify_blocks(games,envelope):
    expected=expected_worker_plan(games);starts=sorted({int(n.split('_')[1][1:]) for key,names in expected.items()
        if key[0]=='main' for n in names.values()});stage=c.read(c.OUT/'stages/main_start.json');admit=c.read(c.OUT/'main_admission.json')
    c.require({p.name for p in(c.OUT/'block_admissions').glob('*.json')}=={f'g{i:03d}.json' for i in starts},'Exact block admission inventory')
    for index in starts:
        proof=c.read(c.OUT/'block_admissions'/f'g{index:03d}.json')
        previous=None if index==0 else c.artifact(game_folder(dict(stage='main',game_index=index-1))/'result.json')
        c.require(proof['stage']=='main' and proof['start_game_index']==index and proof['previous_game']==previous and
            proof['reason']==('fixed_block' if index%c.GAMES_PER_BLOCK==0 else 'next_unplayed_after_guard_loss') and
            proof['no_retried_decision'] is True and proof['assessment_gate']==c.artifact(c.OUT/'assessment_gate.json'),
            'Exact unplayed block identity and outcome-blind forecast binding')
        c.require(proof['previous_closed_workers']==prior_pair_closures('main',index),'Previous assigned pair closure bindings')
        verify_closed_before(proof['previous_closed_workers'],proof['unix_time'])
        for key,deadline,origin in [('monotonic','monotonic_deadline',envelope.start),('unix_time','unix_deadline',envelope.wall_start)]:
            c.require(admit[key]<=proof[key]<proof['effective_'+deadline]==stage[deadline] and
                proof['original_'+('monotonic_origin' if key=='monotonic' else 'unix_origin')]==origin,
                'Every startup shares original main/model clocks')
        for route in c.VARIANTS:
            init=c.read(c.OUT/'workers'/f'main_g{index:03d}_{route}'/'initialization.json')
            c.require(init['identity']['created']>=proof['unix_time'],'Immutable block admission before any fresh model startup')
    return expected


def verify_saved(envelope,freeze,*,writing):
    verify_runtime(freeze,envelope);schedule,plan=load_inputs(freeze,envelope)
    c.require({p.name for p in(c.OUT/'games').iterdir()}=={'operational','main'},'Exact two game phases')
    for stage,count in [('operational',6),('main',400)]:
        c.require({p.name for p in(c.OUT/'games'/stage).iterdir()}=={f'g{i:03d}' for i in range(count)},'Exact fixed game inventory')
    games,rows,worker_counts=read_games(schedule,envelope,freeze)
    operational=[g for g in games if g['stage']=='operational'];main_games=[g for g in games if g['stage']=='main']
    op_summary=analysis.summarize_operational(operational,SavedRows(operational),schedule=design.make_schedule('operational'),boundary=envelope.guard)
    c.require(op_summary==c.read(c.OUT/'operational_analysis.json'),'Complete independently verified operational screen')
    gate=assessment_gate(envelope,writing=False);c.require(gate['projection']['passed'],'All prospective resource gates passed before main')
    verify_admission(envelope,'operational',300.,qualification_summary=c.artifact(c.OUT/'qualification_summary.json'))
    verify_admission(envelope,'main',gate['projection']['main_seconds'],assessment_gate=c.artifact(c.OUT/'assessment_gate.json'))
    expected=verify_blocks(games,envelope)
    names=sorted({n for v in expected.values() for n in v.values()})
    for name in ['operational_g000_'+v for v in c.VARIANTS]:verify_phase_transition(name,envelope)
    c.require({p.stem for p in(c.OUT/'worker_phase_transitions').glob('*.json')}=={'operational_g000_'+v for v in c.VARIANTS},
        'Only two initial idle phase transitions')
    selection=design.select_replay_sample(rows,maximum=c.MAX_REPLAY_DECISIONS)
    c.require(selection==c.read(c.OUT/'replay_selection.json'),'Independent outcome/time-blind exact reproduction selection')
    required=2*(60.+300.)+len(selection['decisions'])*c.SECONDS_PER_MOVE+120.
    replay_admit=verify_admission(envelope,'replay',required,replay_selection=c.artifact(c.OUT/'replay_selection.json'),
        previous_closed_workers=prior_pair_closures('replay',400))
    verify_closed_before(replay_admit['previous_closed_workers'],replay_admit['unix_time'])
    replay_counts=verify_replays(selection,envelope,freeze,schedule)
    for name in ['replay_'+v for v in c.VARIANTS]:
        init=c.read(c.OUT/'workers'/name/'initialization.json')
        c.require(init['identity']['created']>=replay_admit['unix_time'],'Fresh reproductions follow immutable selection')
        names.append(name)
    names=sorted(names);c.require(len(names)<=c.MAX_WORKER_LIFECYCLES,'Finite actual lifecycle ceiling')
    for directory in ('workers','qualification','closed_workers','closed_worker_costs','startup_costs'):
        c.require({p.name if p.is_dir() else p.stem for p in(c.OUT/directory).iterdir()}==set(names),'Exact lifecycle inventory: '+directory)
    qualification_results={};failures={}
    for game in main_games:
        if game['reason']=='forfeit':
            name=expected[('main',game['game_index'])][game['failed_role']=='challenger' and 'candidate' or 'control']
            c.require(name not in failures,'A failed worker is never reused');failures[name]=game
    for name in names:
        qualification_results[name]=qualification_evidence(name,envelope,freeze,plan,writing=False)
        proof=transport.verify_closed_worker(c.OUT/'workers'/name,freeze,envelope)
        ordinary=replay_counts.get(name,0) if name.startswith('replay_') else worker_counts.get(name,0)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json') and
            proof['raw']['failed_requests']==int(name in failures) and proof['raw']['requests']==2+ordinary,
            'Full raw request history, exact single failed suffix and actual owned cleanup')
        if name in failures:
            game=failures[name];last=c.read(game_folder(game)/'moves'/f"{game['plies']-1:06d}.json")
            c.require(last['clock']['request_id']==f'request-{ordinary+2:08d}' and
                scientific.classify_failure(last['clock'])==game['failure'],'Only classified terminal request adjudicated loss')
    initial_qualification={n:qualification_results[n] for n in ['operational_g000_'+v for v in c.VARIANTS]}
    c.require(c.read(c.OUT/'qualification_summary.json')==dict(accepted=True,workers=initial_qualification),
        'Initial both-route qualification summary')
    previous=None
    for stage in ('qualification','operational','main','replay'):
        start=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        c.require(start['cap_seconds']==c.STAGES[stage],'Frozen phase duration')
        for key,origin in [('monotonic',envelope.start),('unix_time',envelope.wall_start)]:
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            c.require(start[key]<=end[key]<start[deadline]==min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS),
                'Original dual-clock stage/model limits')
            if previous:c.require(previous[key]<=start[key],'No reordered stage clocks')
        previous=end
    c.require({p.stem[:-6] for p in(c.OUT/'stages').glob('*_start.json')}==set(c.STAGES),'Exact phase inventory')
    verify_exchanges(envelope,freeze)
    stats=analysis.analyze_main(main_games,True,boundary=envelope.guard)
    stats_proof=analysis.verify_main(main_games,stats,boundary=envelope.guard)
    timing=analysis.summarize_timing(rows,boundary=envelope.guard)
    save_or_verify(c.OUT/'main_analysis.json',stats,writing);save_or_verify(c.OUT/'main_analysis_verification.json',stats_proof,writing)
    save_or_verify(c.OUT/'timing_analysis.json',timing,writing)
    count=sum(g['plies'] for g in games);replays=len(selection['decisions'])
    coverage=dict(status='verified_fixed512_vs256_comparison',workers=len(names),graph_cases=11*len(names),
        scientific_qualification_decisions=67*len(names),completed_games=len(games),operational_games=6,main_games=400,
        main_pairs=200,live_decisions=count,exact_reproductions=replays,parent_requests=2*len(names)+count+replays,
        classified_main_guard_losses=len(failures),strength_and_timing_separate=True,training=False,
        automatic_retries=0,game_replacements=0,strength_promotion=False,all_raw_science_and_clocks_verified=True)
    save_or_verify(c.OUT/'verification_summary.json',coverage,writing);verify_runtime(freeze,envelope);return coverage


SEALED_NAMES=('workers','games','replays','qualification','qualification_summary.json','startup_costs','block_admissions',
    'closed_workers','closed_worker_costs','operational_admission.json','main_admission.json','replay_admission.json',
    'assessment_gate.json','worker_phase_transitions','operational_analysis.json','main_analysis.json','main_analysis_verification.json',
    'timing_analysis.json','verification_summary.json','replay_selection.json','memory_admissions','placement_requests','placement_acks','progress')


def final_checks(envelope,freeze):
    result=verify_saved(envelope,freeze,writing=True)
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=evidence_seal([c.OUT/n for n in SEALED_NAMES if(c.OUT/n).exists()],envelope)))
    return result


def close_workers(workers,envelope,freeze):
    for name in list(workers):
        transport.close_and_audit(workers[name],envelope,freeze);workers.pop(name)


def collect_main(schedule,workers,envelope,freeze,plan):
    games=[]
    for spec in schedule:
        if not workers:begin_block(spec,workers,envelope,freeze,plan)
        game=play_game(spec,workers,envelope,freeze);games.append(game)
        if game['reason']=='forfeit' or (spec['game_index']+1)%c.GAMES_PER_BLOCK==0 or spec==schedule[-1]:
            close_workers(workers,envelope,freeze)
    return games


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact fixed512 comparison freeze')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Original owning watchdog freeze')
    envelope=Envelope(claim);envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    workers={};coverage=None;failure=None;status='incomplete'
    try:
        envelope.begin('qualification');placement.configure_execution(freeze,'isolated')
        c.save(c.OUT/'runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);schedule,plan=load_inputs(freeze,envelope)
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        names=qualify_pair('operational',0,workers,envelope,freeze,plan)
        results={name:c.read(c.OUT/'qualification'/f'{name}.json') for name in names}
        c.save(c.OUT/'qualification_summary.json',dict(accepted=True,workers=results));envelope.finish()
        envelope.begin('operational');phase_admission(envelope,'operational',300.,qualification_summary=c.artifact(c.OUT/'qualification_summary.json'))
        for child in workers.values():advance_idle_worker(child,envelope)
        operational=[]
        for spec in design.make_schedule('operational'):
            game=play_game(spec,workers,envelope,freeze);operational.append(game)
            c.require(game['reason']=='natural','Any operational hard guard stops this attempt; no replacement game')
        close_workers(workers,envelope,freeze)
        op_summary=analysis.summarize_operational(operational,SavedRows(operational,envelope.guard),schedule=design.make_schedule('operational'),boundary=envelope.guard)
        c.save(c.OUT/'operational_analysis.json',op_summary);envelope.finish()
        envelope.begin('main');gate=assessment_gate(envelope,writing=True)
        c.require(gate['projection']['passed'],'Operational resource forecast failed; main games never start')
        phase_admission(envelope,'main',gate['projection']['main_seconds'],assessment_gate=c.artifact(c.OUT/'assessment_gate.json'))
        games=operational+collect_main(design.make_schedule('main'),workers,envelope,freeze,plan);envelope.finish()
        envelope.begin('replay');selection=design.select_replay_sample(SavedRows(games,envelope.guard),maximum=c.MAX_REPLAY_DECISIONS)
        c.save(c.OUT/'replay_selection.json',selection)
        required=2*(60.+300.)+len(selection['decisions'])*c.SECONDS_PER_MOVE+120.
        phase_admission(envelope,'replay',required,replay_selection=c.artifact(c.OUT/'replay_selection.json'),
            previous_closed_workers=prior_pair_closures('replay',400))
        for name in ['replay_'+v for v in c.VARIANTS]:qualify(name,workers,envelope,freeze,plan)
        replay_selected(selection,workers,envelope,freeze,schedule);close_workers(workers,envelope,freeze);envelope.finish()
        envelope.begin('final');coverage=final_checks(envelope,freeze);envelope.finish();status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        for name,child in workers.items():
            if child._closed:continue
            try:transport.close_and_audit(child,envelope,freeze)
            except BaseException:c.save(c.OUT/'cleanup_failures'/f'{name}.json',dict(error=traceback.format_exc()))
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,coverage=coverage,freeze_sha256=expected_sha,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        completed_games=0 if coverage is None else coverage['completed_games'],training=False,automatic_retries=0,
        requires_actual_reader_and_guardian_exit=True))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True)
    return run(parser.parse_args().freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
