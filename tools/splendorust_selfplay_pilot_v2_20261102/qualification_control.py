"""One-clock corrected self-play pilot and the same entrypoint's qualification."""
import argparse,os,sys,time,traceback,shutil,math,re
from copy import deepcopy
from pathlib import Path
from importlib.metadata import version
import psutil
from . import common as c,transport,adapters
from tools.splendorust_search256_recovery_20261030_02 import placement
from . import qualification,scientific
from . import data,design,clock_contract as clocks
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
        resolved=path.resolve()
        if not any(resolved.is_relative_to(folder.resolve()) for folder in self.worker_accounting):
            self.evidence['parent_bytes'] += size

    def register_worker(self, folder):
        from .accounting import WorkerAccounting
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
            (folder.parent.parent/'closed_workers'/f'{folder.name}.json').is_file(),
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
        c.require(total <= 2*c.GiB, 'Combined inference workers/helpers RSS exceeded 2GiB')
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
        fit_folder=c.OUT/'fit'
        fit_bytes=sum(p.stat().st_size for p in fit_folder.rglob('*') if p.is_file()) if fit_folder.exists() else 0
        return total + fit_bytes + 32*c.MiB + 65536

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
    c.verify_failed_attempt_preservation(freeze['failed_attempt_preservation'])
    if envelope:
        envelope.guard()
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.verify_external_runtime(freeze['external_runtime_inputs'])
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
    return design.source_lifecycle_inventory()


def build_plan():
    cases=data.build_qualification_cases()
    c.require(len(cases['cases'])==15,'Frozen fifteen structural capture cases')
    count=len(lifecycle_inventory());replays=3*len(design.replay_game_indices())
    return dict(schema='selfplay_pilot_entrypoint_plan_v1',scope=c.SCOPE,suite=qualification.build_plan(),
        loader_cases=cases,fixture_orders=['capture_off','capture_on'],ordinary_per_fixture_worker=16,
        actual_loader_ordinary_requests=32,graph_cases=11*count,scientific_decisions=67*count,
        fresh_replay_decisions=replays,maximum_fixture_decisions=32+67*count,
        replay_rule=dict(kind='one_per_selected_game_progress_third',game_indices=design.replay_game_indices(),thirds=3,
            priority_domain='selfplay_pilot_replay_20261102_'+c.SCOPE,outcomes_and_timings_unused=True),
        new_games=len(design.schedule()),optimizer_steps='one' if c.SCOPE=='qualification' else 'two_complete_epochs',
        learned_adapter_decisions=6,learned_adapter_simulations=128,
        qualification_only=c.SCOPE=='qualification',pilot_data=c.SCOPE=='pilot')


def contract():
    return dict(schema=c.SCHEMA,phase=c.PHASE,job_seconds=c.JOB_SECONDS,clock_contract=clocks.runtime_contract(),
        collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,stage_seconds=c.STAGES,
        independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        automatic_retries=0,failed_decision_retries=0,game_replacements=0,
        seconds_per_move=c.SECONDS_PER_MOVE,internal_seconds=c.INTERNAL_SECONDS,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,max_rss_bytes=c.MAX_INFERENCE_RSS_BYTES,
        fitter_max_rss_bytes=c.MAX_FITTER_RSS_BYTES,max_workers=c.MAX_WORKER_LIFECYCLES,max_resident_workers=2,
        initial_available_ram_bytes=c.INITIAL_AVAILABLE_RAM_BYTES,
        minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,initial_ram_stability_seconds=60,
        maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,
        minimum_free_disk_bytes=c.MINIMUM_FREE_DISK_BYTES,placement_mode='isolated',
        declared_worker_lifecycles=lifecycle_inventory(),maximum_live_decisions=c.MAX_LIVE_DECISIONS,
        qualification_scientific_decisions_maximum=67*len(lifecycle_inventory()),graph_cases_maximum=11*len(lifecycle_inventory()),
        maximum_replay_decisions=3*len(design.replay_game_indices()),maximum_plies_per_game=c.MAX_PLIES,
        max_response_bytes=c.MAX_RESPONSE_BYTES,max_raw_log_bytes=c.MAX_LOG_BYTES,
        teacher_simulations=512,world_pool=3,cpuct=.4,search_depth=64,
        original_full_network=True,one_second_gate=False,seven_hundred_ms_gate=False,
        source_games=len(design.schedule()),optimizer_steps='one' if c.SCOPE=='qualification' else 'two_complete_epochs',
        training_sample_reuse=False,strength_promotion=False,learned_adapter_decisions=6,learned_adapter_simulations=128,
        qualification_only=c.SCOPE=='qualification',pilot_data=c.SCOPE=='pilot',forecast_factor=1.5)


def validate_freeze_contract(freeze):
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()),'Exact frozen engineering contract')
    c.require(set(freeze['profiles'])=={'candidate'} and set(freeze['profiles']['candidate'])=={'16','512'},
        'Literal accepted512 and qualification16 profiles')
    return expected


def load_inputs(freeze,envelope):
    c.verify([freeze[k] for k in ('game_schedule','initial_states','qualification_plan','training_plan','graph_parity_plan','fit_plan','fit_parity_fixtures')])
    schedule=c.read(c.ROOT/freeze['game_schedule']['path'])
    c.require(schedule==design.schedule(),'Exact immutable source-game schedule')
    c.require(c.read(c.ROOT/freeze['initial_states']['path'])==design.initial_states(),'Fresh corrected initial states')
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(plan==build_plan(),'Exact collector/loader qualification plan')
    return schedule,plan


def qualification_evidence(name,envelope,freeze,plan,*,writing):
    folder=c.OUT/'workers'/name
    graph=transport.verify_graph_evidence(folder,freeze,boundary=envelope.guard)
    clock=c.read(folder/'entrypoint_clock.json');profiles=c.read(folder/'entrypoint_profiles.json')
    transport._require_profiles(profiles)
    transport.validate_profiles(profiles['parent_profile'],profiles['child_profile'],clock,{'operation':'entrypoint_checks'},'candidate')
    suite=qualification.verify_entrypoint(folder,clock,plan['suite'],freeze,boundary=envelope.guard)
    historical=qualification.compare_control_parity(clock['result'],plan['suite'],boundary=envelope.guard)
    result=dict(status='qualified',lifecycle=name,route='candidate',quota=512,graph=graph,suite=suite,
        historical_parity=historical,graph_cases=11,scientific_decisions=67,parent_requests=2)
    save_or_verify(c.OUT/'qualification'/f'{name}.json',result,writing);return result


def qualify(name,workers,envelope,freeze,plan):
    c.require(not workers,'Qualification lifecycles are sequential')
    begun=time.monotonic();wait_before=envelope.waited;bytes_before=envelope.accounted_bytes()
    child=transport.make_worker(name,envelope,freeze,route='candidate');workers[name]=child
    c.require(envelope.remaining()>=300.,'Full67-suite allowance remains')
    adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
    proof=qualification_evidence(name,envelope,freeze,plan,writing=True)
    elapsed=time.monotonic()-begun
    c.save(c.OUT/'startup_costs'/f'{name}.json',dict(seconds=elapsed,nonwait_seconds=elapsed-(envelope.waited-wait_before),
        accounted_growth_bytes=envelope.accounted_bytes()-bytes_before,raw_stream_bytes=dict(child._log_sizes)))
    return proof


def actual_after(before,action):
    state=decode_training_snapshot(before);env=rules.CorrectedEnv();chosen=rules.deserialize_action(action)
    c.require(chosen in env.legal_actions(state),'Corrected complete action')
    after=rules.serialize_snapshot(env.step(state,chosen))
    c.require(after==ledger.transition(before,action),'Actual environment and independent ledger agree')
    decode_training_snapshot(after);return after


def decision(child,spec,before,envelope,freeze,ply):
    observer=rules.observation_payload(decode_training_snapshot(before))
    row=transport.ordinary(child,spec,observer,envelope,freeze)
    row.update(ply=ply,before=before,after=None,training_capture=None)
    if row['completed']:
        capture=transport.verify_capture_publication(child.output_dir,row['clock'],observer,freeze)
        row['training_capture']=capture['training_capture']
        row['capture_publication']=c.artifact(child.output_dir/(row['clock']['request_id']+'_training_capture.json'))
        row['after']=actual_after(before,row['action'])
    return row


def validate_row(row,freeze,envelope,*,capture_required=True):
    envelope.guard();transport._require_profiles(row);transport.verify_saved_record(row,freeze)
    c.require(row['completed'] and row['clock']['status']=='ok','Any failed research decision remains incomplete')
    ledger.check_observer(row['before'],row['observer'])
    c.require(actual_after(row['before'],row['action'])==row['after'],'Full lawful state transition')
    scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',before=row['before'],after=row['after'],expected_quota=512)
    publication=transport.verify_capture_publication(c.ROOT/row['worker_folder'],row['clock'],row['observer'],freeze)
    c.require(publication['training_capture']==row['training_capture'],'Bound actual root capture')
    c.verify([row['capture_publication']])
    if capture_required:data.decode_row(row)
    return True


def fixture_rows(name,envelope=None):
    return [c.read(c.OUT/'capture_fixtures'/name/f'{i:03d}.json') for i in range(16)]


def collect_fixtures(child,envelope,freeze,plan):
    cases=plan['loader_cases'];ordered=cases['cases']+[cases['private_variant']]
    rows=[]
    for index,case in enumerate(ordered):
        spec=dict(kind='capture_counterexample',index=index,case_id=case.get('id','private_hidden_repeat'),route='candidate',quota=512)
        row=decision(child,spec,case['snapshot'],envelope,freeze,index)
        c.save(c.OUT/'capture_fixtures'/child.output_dir.name/f'{index:03d}.json',row)
        validate_row(row,freeze,envelope,capture_required=child.output_dir.name!='capture_off');rows.append(row)
    return rows


def verify_capture_fixtures(envelope,freeze,plan,*,writing):
    off,on=fixture_rows('capture_off'),fixture_rows('capture_on')
    for index,(left,right) in enumerate(zip(off,on,strict=True)):
        validate_row(left,freeze,envelope,capture_required=False);validate_row(right,freeze,envelope)
        for key in ('spec','before','observer','action','trace','after'):
            c.require(c.canonical(left[key])==c.canonical(right[key]),'OFF/ON exact science '+key)
        c.require(left['clock']['result']==right['clock']['result'],'Entire capture-OFF/ON scientific result')
    loader=data.qualify_captured_rows(plan['loader_cases'],on,boundary=envelope.guard)
    proof=dict(status='passed',actual_loader=loader,ordinary_decisions=32,full_scientific_parity_pairs=16,
        privacy='same lawful observer under inaccessible-card permutation',no_game_data_generated_yet=True)
    save_or_verify(c.OUT/'capture_qualification.json',proof,writing);return proof


def forecast_snapshot(envelope,completed_games,*,prospective_collection=False):
    mono,wall=time.monotonic(),time.time()
    end=min(mono+c.STAGES['collection'],envelope.start+c.COLLECTION_CUTOFF_SECONDS) if prospective_collection else envelope.deadline
    wall_end=min(wall+c.STAGES['collection'],envelope.wall_start+c.COLLECTION_CUTOFF_SECONDS) if prospective_collection else envelope.wall_deadline
    return dict(monotonic_now=mono,unix_now=wall,original_monotonic_origin=envelope.start,
        original_unix_origin=envelope.wall_start,stage_monotonic_deadline=end,stage_unix_deadline=wall_end,
        accounted_bytes=envelope.accounted_bytes(),cumulative_ram_wait_seconds=envelope.waited,
        completed_games=completed_games,qualified_source_workers=len(list((c.OUT/'qualification').glob('*.json'))))


def capture_forecast(envelope,freeze,*,writing):
    path=c.OUT/'collection_forecast.json'
    snapshot=forecast_snapshot(envelope,0,prospective_collection=True) if writing else c.read(path)['snapshot']
    c.require(snapshot['original_monotonic_origin']==envelope.start and
        snapshot['original_unix_origin']==envelope.wall_start and snapshot['completed_games']==0 and
        snapshot['qualified_source_workers']==2,'Forecast must bind original owner and exact precollection prefix')
    for now,origin,end in [('monotonic_now','original_monotonic_origin','stage_monotonic_deadline'),
                            ('unix_now','original_unix_origin','stage_unix_deadline')]:
        c.require(snapshot[end]==min(snapshot[now]+c.STAGES['collection'],snapshot[origin]+c.COLLECTION_CUTOFF_SECONDS),
                  'Prospective forecast cannot invent a longer collection allowance')
    value=design.source_forecast(freeze,snapshot,source_costs=[])
    value.update(passed=value['affordable'],status='passed' if value['affordable'] else 'failed')
    save_or_verify(path,value,writing);return value


def verify_block_forecast(block,envelope,freeze,*,writing):
    path=c.OUT/'block_forecasts'/f"{block['block_index']:03d}.json"
    completed=max(block['game_indices'])+1
    costs=[c.read(game_folder({'game_index':i})/'cost.json') for i in range(completed)]
    snapshot=forecast_snapshot(envelope,completed) if writing else c.read(path)['snapshot']
    c.require(snapshot['original_monotonic_origin']==envelope.start and
        snapshot['original_unix_origin']==envelope.wall_start,'Every block keeps original job clocks')
    collection=c.read(c.OUT/'stages/collection_start.json')
    c.require(snapshot['stage_monotonic_deadline']==collection['monotonic_deadline'] and
        snapshot['stage_unix_deadline']==collection['unix_deadline'],'Block forecast keeps original collection clocks')
    c.require(snapshot['completed_games']==completed,'Forecast bound to exact contiguous completed prefix')
    expected_workers=2+block['block_index']
    c.require(snapshot['qualified_source_workers']==expected_workers,'Exact startup prefix before source replay')
    value=design.source_forecast(freeze,snapshot,source_costs=costs)
    value.update(block_index=block['block_index'],cost_inputs=[c.artifact(game_folder({'game_index':i})/'cost.json') for i in range(completed)])
    save_or_verify(path,value,writing)
    c.require(value['affordable'],'Remaining fixed source sample and downstream reserves are unaffordable; preserve incomplete')
    return value


def advance_idle_worker(child,envelope):
    from tools.splendorust_network_practical_handoff_20261031 import live_worker_identity
    c.require(not child._closed and child._counter==18 and child.process.poll() is None and not child._failure and not child._lock.locked(),
        'Idle collector after exact graph/suite and16 fixtures only')
    identity=live_worker_identity(child,common=c);old=child.absolute_deadline
    c.require(old<=envelope.start+c.COLLECTION_CUTOFF_SECONDS and envelope.deadline<=envelope.start+c.COLLECTION_CUTOFF_SECONDS,'Original model cutoff')
    receipt=dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),perf_counter_ns=time.perf_counter_ns(),worker_identity=identity,
        old_absolute_deadline=old,new_absolute_deadline=envelope.deadline,completed_requests=18,next_request_id='request-00000019',
        active_request=False,request_allowance_unchanged=10.,original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start)
    c.save(c.OUT/'worker_phase_transitions'/f'{child.output_dir.name}.json',receipt)
    child.absolute_deadline=envelope.deadline


def verify_phase_transition(envelope):
    receipt=c.read(c.OUT/'worker_phase_transitions/capture_on.json')
    init=c.read(c.OUT/'workers/capture_on/initialization.json')
    qualification_start=c.read(c.OUT/'stages/qualification_start.json')
    collection_start=c.read(c.OUT/'stages/collection_start.json')
    first=c.read(c.OUT/'workers/capture_on/parent_profiles/request-00000019.json')
    c.require(receipt['worker_identity']==init['identity'] and receipt['completed_requests']==18 and
        receipt['next_request_id']=='request-00000019' and receipt['active_request'] is False and
        receipt['request_allowance_unchanged']==10. and
        receipt['old_absolute_deadline']==init['absolute_deadline'] and
        qualification_start['monotonic']<init['absolute_deadline']<=qualification_start['monotonic_deadline'] and
        init['absolute_deadline']<=envelope.start+c.COLLECTION_CUTOFF_SECONDS and
        receipt['new_absolute_deadline']==collection_start['monotonic_deadline'] and
        receipt['original_monotonic_origin']==envelope.start and receipt['original_unix_origin']==envelope.wall_start and
        collection_start['monotonic']<=receipt['monotonic']<collection_start['monotonic_deadline'] and
        receipt['perf_counter_ns']<=first['request_start_ns'],'Saved idle handoff before first fresh game move')
    return True


def game_folder(spec):return c.OUT/'games'/f"g{spec['game_index']:03d}"


def play_game(child,spec,initial,envelope,freeze):
    design.validate_spec(spec);folder=game_folder(spec)
    c.save(folder/'initial.json',dict(spec=spec,initial=initial));before=initial;rows=[]
    begun=time.monotonic();wait=envelope.waited;used=envelope.accounted_bytes()
    for ply in range(c.MAX_PLIES):
        if decode_training_snapshot(before).terminal:break
        row=decision(child,spec,before,envelope,freeze,ply)
        c.save(folder/'moves'/f'{ply:06d}.json',row);validate_row(row,freeze,envelope);rows.append(row);before=row['after']
    winner=__import__('tools.future_training_rules_20261021',fromlist=['require_official_terminal_target']).require_official_terminal_target(before,'completed')
    document=dict(schema=data.GAME_SCHEMA,**data.IDENTITY,spec=spec,initial=initial,rows=rows,final=before,reason='completed',winner=winner)
    data.load_game(document,spec,boundary=envelope.guard)
    descriptor=c.save(folder/'game.json',document)
    c.save(folder/'cost.json',dict(seconds=time.monotonic()-begun,ram_wait_seconds=envelope.waited-wait,
        accounted_growth_bytes=envelope.accounted_bytes()-used,plies=len(rows),eligible=sum(r['training_capture']['eligible'] for r in rows)))
    c.save(c.OUT/'progress'/f'game_{spec["game_index"]:03d}.json',dict(completed=True,plies=len(rows),time=c.now()))
    return descriptor


def select_replays(game_descriptors,envelope):
    selected=[]
    for descriptor in game_descriptors:
        envelope.guard();c.verify([descriptor]);game=c.read(c.ROOT/descriptor['path']);n=len(game['rows'])
        if game['spec']['game_index'] not in design.replay_game_indices():continue
        c.require(n>=3,'Three disjoint progress strata per fixture game')
        for third in range(3):
            candidates=[r for r in game['rows'] if min(2,3*r['ply']//n)==third]
            priority=lambda row:c.digest(['selfplay_pilot_replay_20261102_'+c.SCOPE,game['spec']['game_index'],row['ply'],c.digest(row['observer'])])
            chosen=min(candidates,key=priority)
            selected.append(dict(game_index=game['spec']['game_index'],ply=chosen['ply'],third=third,selection_sha256=priority(chosen),
                source=c.artifact(game_folder(game['spec'])/'moves'/f"{chosen['ply']:06d}.json")))
    c.require(len(selected)==3*len(design.replay_game_indices()),'Exact prospective blind reproduction count');return selected


def reproduce(child,selection,envelope,freeze):
    for index,item in enumerate(selection):
        c.verify([item['source']]);old=c.read(c.ROOT/item['source']['path'])
        row=decision(child,old['spec'],old['before'],envelope,freeze,old['ply'])
        c.save(c.OUT/'replays'/f'{index:03d}.json',dict(selection=item,row=row))
        verify_reproduction(old,row,freeze,envelope)


def verify_reproduction(old,row,freeze,envelope):
    validate_row(row,freeze,envelope)
    for key in ('spec','ply','before','observer','action','trace','after','training_capture'):
        c.require(row[key]==old[key],'Exact fresh-process reproduction '+key)
    c.require(row['clock']['result']==old['clock']['result'],'Exact full fresh scientific result')
    return True


def close_workers(workers,envelope,freeze):
    for name in list(workers):
        transport.close_and_audit(workers[name],envelope,freeze);workers.pop(name)


def observe_learned_requests(folder,identity,envelope,freeze,observed):
    """External QPC receipts plus independent original monotonic job boundaries."""
    import hashlib,json
    directory=folder/'learned_entrypoint'
    starts=sorted(directory.glob('request_*.started.json')) if directory.exists() else []
    c.require(len(starts)<=6 and [p.name for p in starts]==[f'request_{i:03d}.started.json' for i in range(len(starts))],
        'At most six contiguous learned-adapter decisions')
    c.require(len(starts)<=len(observed)+1,'No next learned decision before owning receipt acknowledgement')
    index=len(observed)
    if index>=len(starts):return
    path=starts[index];start=start_item=response=reading=None
    try:
        c.require(path.stat().st_size<65536,'Bounded learned start marker')
        start=c.read(path);start_item=c.artifact(path)
        ids=freeze['learned_entrypoint']['case_ids'];expected=freeze['clock_contract']
        clocks.validate_start(start,expected_contract=expected)
        c.require(start['request_index']==index and start['case_id']==ids[index%3] and
            start['mode']==('eager' if index<3 else 'compiled'),'Fixed learned request identity')
        owner=c.read(folder/'owner.json')
        clocks.validate_admission(start,owner_start=owner['owner_monotonic_start'],
            owner_unix_start=owner['unix_time'],owner_deadline=envelope.deadline,
            unix_deadline=envelope.wall_deadline,expected_contract=expected)
        c.require(start['request_qpc_start']>=owner['request_qpc_start'],'Request QPC starts after owning fitter launch')
        if observed:clocks.validate_sequence(observed[-1],start)
        completed=directory/f'request_{index:03d}.completed.json'
        if not completed.exists():
            reading=clocks.sample()
            clocks.require_inherited_deadlines(envelope.deadline,envelope.wall_deadline,reading=reading)
            clocks.validate_pending(start,reading,expected_contract=expected)
            return
        info=completed.stat();c.require(0<info.st_size<=c.MAX_RESPONSE_BYTES,'Bounded learned complete response')
        raw=completed.read_bytes();reading=clocks.sample()
        clocks.require_inherited_deadlines(envelope.deadline,envelope.wall_deadline,reading=reading)
        after=completed.stat()
        c.require((info.st_ino,info.st_size,info.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),
            'Atomic immutable learned response')
        response=json.loads(raw)
        clocks.validate_received(start,response,observed=reading,start_artifact=start_item,request_index=index,
            case_id=ids[index%3],mode=('eager' if index<3 else 'compiled'),
            candidate_manifest=c.artifact(folder/'checkpoint_manifest.json'),expected_contract=expected)
        descriptor=dict(path=completed.relative_to(c.ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        receipt=dict(schema='selfplay_learned_parent_receipt_v2',clock_contract=expected,
            accepted=True,request_index=index,response=descriptor,start=start_item,fitter_identity=identity,
            observer=dict(pid=os.getpid(),created=psutil.Process().create_time()),
            parent_observed_qpc=reading['request_qpc'],parent_observed_owner_monotonic=reading['owner_monotonic'],
            parent_observed_unix=reading['unix'],receipt_elapsed_seconds=reading['request_qpc']-start['request_qpc_start'],
            deadline_seconds=10.,response_complete_before_deadline=True,original_request_clock_preserved=True)
        clocks.validate_ack(start,response,receipt,expected_contract=expected)
        c.atomic_save(directory/f'request_{index:03d}.parent_accepted.json',receipt);observed.append(receipt)
    except BaseException as error:
        # Persist the actual sampled clocks and precise failing branch before cleanup.
        rejection=directory/f'request_{index:03d}.parent_rejected.json'
        if not rejection.exists():
            c.atomic_save(rejection,dict(schema='selfplay_learned_parent_rejection_v2',accepted=False,
                request_index=index,fitter_identity=identity,start_artifact=start_item,start=start,
                observation=reading,branch=getattr(error,'branch',type(error).__name__),error=str(error),
                owner_monotonic_deadline=envelope.deadline,unix_deadline=envelope.wall_deadline,
                response_clock_fields=None if response is None else {k:response.get(k) for k in
                    ('schema','request_index','case_id','mode','clock_contract','finished_qpc','finished_owner_monotonic','finished_unix')},
                original_request_clock_preserved=True,retry=False))
        raise


def fit_command_environment(envelope,freeze,folder):
    paths=dict(manifest=c.OUT/'corpus_manifest.json',plan=c.ROOT/freeze['fit_plan']['path'],fixtures=c.ROOT/freeze['fit_parity_fixtures']['path'])
    command=[sys.executable,'-B','-u','-m','tools.splendorust_selfplay_pilot_v2_20261102.training','--mode',c.SCOPE]
    for flag,path in paths.items():command += ['--'+flag,str(path)]
    command+=['--out',str(folder)]
    env=os.environ.copy();env.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
        SELFPLAY_MONOTONIC_DEADLINE=str(envelope.deadline),SELFPLAY_UNIX_DEADLINE=str(envelope.wall_deadline),
        STUDY_RESEARCH_FREEZE=str(c.OUT/'freeze.json'),STUDY_RESEARCH_FREEZE_SHA256=c.file_hash(c.OUT/'freeze.json'),
        SELFPLAY_JOB_SCOPE=c.SCOPE)
    return command,env


def run_fit(envelope,freeze):
    import subprocess
    from tools.local_player_qualification_128.clock import _WindowsJob
    from .qualification_watchdog import cleanup_owned
    c.require(not envelope.resident and 'torch' not in sys.modules,'Fit is sequential with no model in supervisor')
    envelope.admit();folder=c.OUT/'fit';folder.mkdir(exist_ok=False)
    command,env=fit_command_environment(envelope,freeze,folder)
    child=job=stdout=stderr=None;identity=None;failure=None;peak=0;status='failed';observed=[]
    try:
        stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb')
        owner_reading=clocks.sample()
        child=subprocess.Popen(command,cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        identity=dict(pid=child.pid,created=psutil.Process(child.pid).create_time());job=_WindowsJob(child.pid)
        c.save(folder/'owner.json',dict(identity=identity,command=command,clock_contract=freeze['clock_contract'],
            monotonic=owner_reading['owner_monotonic'],owner_monotonic_start=owner_reading['owner_monotonic'],
            request_qpc_start=owner_reading['request_qpc'],unix_time=owner_reading['unix'],
            monotonic_deadline=envelope.deadline,unix_deadline=envelope.wall_deadline,max_rss_bytes=c.MAX_FITTER_RSS_BYTES))
        while child.poll() is None:
            envelope.guard();root=psutil.Process(child.pid);c.require(root.create_time()==identity['created'],'Fitter identity')
            if (folder/'placement.json').exists():
                placed=c.read(folder/'placement.json')
                c.require(placed['identity']==identity and placed['before_torch_import'] and
                    placed['cpu_affinity']==[0] and root.cpu_affinity()==[0] and
                    int(root.nice())==placement.NORMAL_PRIORITY,'Fitter stays on declared CPU0/NORMAL')
            observe_learned_requests(folder,identity,envelope,freeze,observed)
            processes=[root]+root.children(recursive=True);rss=0
            for process in processes:
                try:rss+=process.memory_info().rss
                except psutil.NoSuchProcess:continue
            peak=max(peak,rss);c.require(rss<=c.MAX_FITTER_RSS_BYTES,'Fitter aggregateRSS4GiB')
            c.require(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())<c.MAX_FIT_EVIDENCE_BYTES,'Bounded fitter artifact allowance')
            time.sleep(.1)
        c.require(child.returncode==0,'Single fitter attempt failed; no retry')
        c.require(len(observed)==6,'Exactly six owning learned decision receipts');envelope.guard();status='completed'
    except BaseException:failure=traceback.format_exc()
    finally:
        cleanup=cleanup_owned(child,job,(stdout,stderr),monotonic_origin=envelope.start,unix_origin=envelope.wall_start,
            root_identity=identity,extra_deadlines=(envelope.deadline,envelope.wall_deadline))
        if cleanup['errors']:status='failed'
        c.save(c.OUT/'fit_execution.json',dict(status=status,exit_code=None if child is None else child.returncode,
            identity=identity,error=failure,cleanup=cleanup,peak_rss_bytes=peak,learned_response_receipts=observed,
            monotonic_end=time.monotonic(),unix_end=time.time(),
            original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start))
    c.require(status=='completed' and failure is None,'Fitter failed or cleanup incomplete')


def verify_fit(envelope,freeze):
    from . import training
    result=c.read(c.OUT/'fit_execution.json');c.require(result['status']=='completed' and result['exit_code']==0 and not result['error'] and not result['cleanup']['errors'],'Actual successful fitter exit')
    identity=result['identity']
    placed=c.read(c.OUT/'fit/placement.json');owner=c.read(c.OUT/'fit/owner.json')
    fit_stage=c.read(c.OUT/'stages/fit_start.json')
    c.require(owner['monotonic_deadline']==fit_stage['monotonic_deadline'] and
        owner['unix_deadline']==fit_stage['unix_deadline'] and
        fit_stage['monotonic']<=owner['owner_monotonic_start'] and fit_stage['unix_time']<=owner['unix_time'],
        'Fitter inherits the unchanged original fit stage clocks')
    c.require(placed['identity']==identity==owner['identity'] and placed['cpu_affinity']==[0] and
        placed['priority']=='NORMAL_PRIORITY_CLASS' and placed['before_torch_import'] is True and
        placed['clock_contract']==owner['clock_contract']==freeze['clock_contract'] and
        owner['owner_monotonic_start']<=placed['owner_monotonic_time']<=result['monotonic_end'] and
        owner['request_qpc_start']<=placed['request_qpc_time'] and placed['unix_time']<result['unix_end'],
        'Actual fitter identity and CPU0/NORMAL placement before Torch/model work')
    try:process=psutil.Process(identity['pid'])
    except psutil.NoSuchProcess:process=None
    c.require(process is None or process.create_time()!=identity['created'],'Fitter absent after actual exit')
    c.require(len(result['learned_response_receipts'])==6,'Six trusted learned-response receipts')
    previous=None
    for index,receipt in enumerate(result['learned_response_receipts']):
        envelope.guard();directory=c.OUT/'fit/learned_entrypoint'
        c.require(receipt==c.read(directory/f'request_{index:03d}.parent_accepted.json'),'Original external learned receipt')
        start=c.read(directory/f'request_{index:03d}.started.json');c.verify([receipt['start'],receipt['response']])
        response=c.read(c.ROOT/receipt['response']['path'])
        c.require(receipt['accepted'] is True and receipt['request_index']==index and receipt['fitter_identity']==identity,
            'Six trusted learned decision identities')
        clocks.validate_ack(start,response,receipt,expected_contract=freeze['clock_contract'])
        c.require(start['request_qpc_start']>=owner['request_qpc_start'],
            'Saved request QPC starts after owning fitter launch')
        clocks.validate_admission(start,owner_start=owner['owner_monotonic_start'],
            owner_unix_start=owner['unix_time'],owner_deadline=owner['monotonic_deadline'],
            unix_deadline=owner['unix_deadline'],expected_contract=freeze['clock_contract'])
        clocks.require_inherited_deadlines(owner['monotonic_deadline'],owner['unix_deadline'],reading={
            'request_qpc':receipt['parent_observed_qpc'],'owner_monotonic':receipt['parent_observed_owner_monotonic'],
            'unix':receipt['parent_observed_unix']})
        if previous is not None:clocks.validate_sequence(previous,start)
        previous=receipt
    verifier=training.verify_qualification if c.SCOPE=='qualification' else training.verify_pilot_fit
    return verifier(c.OUT/'fit',c.read(c.OUT/'corpus_manifest.json'),
        c.read(c.ROOT/freeze['fit_plan']['path']),c.read(c.ROOT/freeze['fit_parity_fixtures']['path'])['fixtures'],
        expected_specs=design.schedule(),boundary=envelope.guard)


def verify_saved(envelope,freeze,*,writing):
    schedule,plan=load_inputs(freeze,envelope);manifest=c.read(c.OUT/'corpus_manifest.json')
    c.require(manifest==data.build_manifest([c.artifact(game_folder(s)/'game.json') for s in schedule],schedule,qualification_only=c.SCOPE=='qualification'),'Exact fixed completed corpus')
    count=0;ordinary_counts={name:0 for name in lifecycle_inventory()}
    ordinary_counts.update(capture_off=16,capture_on=16,replay_on=3*len(design.replay_game_indices()))
    by_game={index:block['worker'] for block in design.collection_blocks() for index in block['game_indices']}
    for spec,initial in zip(schedule,design.initial_states(),strict=True):
        game=c.read(game_folder(spec)/'game.json');c.require(game['initial']==initial['initial'],'Frozen source setup')
        c.require([p.name for p in sorted((game_folder(spec)/'moves').glob('*.json'))]==[f'{i:06d}.json' for i in range(len(game['rows']))],'Exact contiguous saved game ledger')
        for row in game['rows']:
            envelope.guard();c.require(row==c.read(game_folder(spec)/'moves'/f"{row['ply']:06d}.json"),'Full game/row binding')
            c.require(Path(row['worker_folder']).name==by_game[spec['game_index']],'Every game belongs to its immutable worker block')
            validate_row(row,freeze,envelope);count+=1;ordinary_counts[by_game[spec['game_index']]]+=1
    loaded=data.load_corpus(manifest,expected_specs=schedule,retain_examples=False,boundary=envelope.guard)
    # Capture proof already exists from precollection qualification. Never recreate it.
    verify_capture_fixtures(envelope,freeze,plan,writing=False)
    names=set(lifecycle_inventory());c.require({p.name for p in (c.OUT/'workers').iterdir() if p.is_dir()}==names,'Exactly the declared source inference lifecycles')
    for name in sorted(names):
        qualification_evidence(name,envelope,freeze,plan,writing=False)
        proof=transport.verify_closed_worker(c.OUT/'workers'/name,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json'),'Independent full raw/clock/cleanup closure')
        ordinary_count=len(list((c.OUT/'workers'/name/'inputs').glob('*.json')))
        c.require(ordinary_count==ordinary_counts[name],
            'Every ordinary request belongs to a declared fixture, game, or reproduction')
    selection=select_replays(manifest['games'],envelope)
    c.require(selection==c.read(c.OUT/'replay_selection.json') and len(list((c.OUT/'replays').glob('*.json')))==len(selection),'Outcome-blind exact source replay inventory')
    for index,item in enumerate(selection):
        observed=c.read(c.OUT/'replays'/f'{index:03d}.json');c.require(observed['selection']==item,'Replay source selection')
        verify_reproduction(c.read(c.ROOT/item['source']['path']),observed['row'],freeze,envelope)
    fit=verify_fit(envelope,freeze)
    fit_result=c.read(c.OUT/'fit/fit_result.json')
    gate=capture_forecast(envelope,freeze,writing=False);c.require(gate['passed'],'Passed whole-stage resource admission')
    for block in design.collection_blocks():
        verify_block_forecast(block,envelope,freeze,writing=False)
    verify_phase_transition(envelope)
    verify_exchanges(envelope,freeze)
    from .assessment import control as assessment
    assessment_result=assessment.verify_saved(envelope,freeze,writing=writing)
    expected_stages=['qualification','collection','fit','assessment_qualification','operational']
    if c.SCOPE=='pilot':expected_stages.append('assessment')
    expected_stages.append('reproduction')
    for stage in expected_stages:
        start=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        for key,origin in [('monotonic',envelope.start),('unix_time',envelope.wall_start)]:
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            c.require(start[key]<=end[key]<start[deadline]==min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS),'Original stage/model clocks')
    result=dict(status='verified_corrected_selfplay_'+c.SCOPE,completed_games=len(schedule),workers=len(names),
        graph_cases=11*len(names),scientific_qualification_decisions=67*len(names),capture_fixture_decisions=32,live_decisions=count,
        exact_reproductions=len(selection),parent_requests=2*len(names)+32+count+len(selection),optimizer_steps=fit_result['optimizer_steps'],
        fit_direct_forward_invocations=fit_result['direct_forward_invocations'],learned_adapter_decisions=6,learned_adapter_simulations=128,
        corpus_game_counts=loaded['game_counts'],corpus_example_counts=loaded['example_counts'],fit=fit,
        assessment=assessment_result,qualification_only=c.SCOPE=='qualification',pilot_data=c.SCOPE=='pilot',strength_promotion=False,automatic_retries=0,game_replacements=0)
    save_or_verify(c.OUT/'verification_summary.json',result,writing);verify_runtime(freeze,envelope);return result


SEALED_NAMES=('workers','capture_fixtures','games','fit','replays','qualification','closed_workers','closed_worker_costs',
    'startup_costs','memory_admissions','placement_requests','placement_acks','worker_phase_transitions','progress',
    'runtime_sources.json','capture_qualification.json','collection_forecast.json','corpus_manifest.json',
    'replay_selection.json','fit_execution.json','verification_summary.json','assessment','block_forecasts')


def final_checks(envelope,freeze):
    result=verify_saved(envelope,freeze,writing=True)
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=evidence_seal([c.OUT/n for n in SEALED_NAMES if(c.OUT/n).exists()],envelope)))
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact qualification freeze')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Original owner freeze')
    envelope=Envelope(claim);envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    workers={};coverage=None;failure=None;status='incomplete'
    try:
        envelope.begin('qualification');placement.configure_execution(freeze,'isolated')
        c.save(c.OUT/'runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);schedule,plan=load_inputs(freeze,envelope)
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        qualify('capture_off',workers,envelope,freeze,plan);collect_fixtures(workers['capture_off'],envelope,freeze,plan);close_workers(workers,envelope,freeze)
        qualify('capture_on',workers,envelope,freeze,plan);collect_fixtures(workers['capture_on'],envelope,freeze,plan)
        verify_capture_fixtures(envelope,freeze,plan,writing=True)
        gate=capture_forecast(envelope,freeze,writing=True);c.require(gate['passed'],'Observed capture resource forecast failed before collection')
        envelope.finish();envelope.begin('collection');advance_idle_worker(workers['capture_on'],envelope)
        games=[];initial_by_game={item['spec']['game_index']:item for item in design.initial_states()}
        for block in design.collection_blocks():
            name=block['worker']
            if name!='capture_on':qualify(name,workers,envelope,freeze,plan)
            for index in block['game_indices']:
                item=initial_by_game[index]
                games.append(play_game(workers[name],item['spec'],item['initial'],envelope,freeze))
            close_workers(workers,envelope,freeze)
            verify_block_forecast(block,envelope,freeze,writing=True)
        manifest=data.build_manifest(games,schedule,qualification_only=c.SCOPE=='qualification');c.save(c.OUT/'corpus_manifest.json',manifest)
        selection=select_replays(games,envelope);c.save(c.OUT/'replay_selection.json',selection)
        qualify('replay_on',workers,envelope,freeze,plan);reproduce(workers['replay_on'],selection,envelope,freeze);close_workers(workers,envelope,freeze)
        envelope.finish();envelope.begin('fit');run_fit(envelope,freeze);verify_fit(envelope,freeze);envelope.finish()
        from .assessment import control as assessment
        candidate=c.read(c.OUT/'fit/fit_result.json')['checkpoint_manifest']
        assessment.run(envelope,freeze,candidate)
        envelope.begin('final');coverage=final_checks(envelope,freeze);envelope.finish();status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        for name,child in workers.items():
            try:
                if not child._closed:transport.close_and_audit(child,envelope,freeze)
            except BaseException:c.save(c.OUT/'cleanup_failures'/f'{name}.json',dict(error=traceback.format_exc()))
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,coverage=coverage,freeze_sha256=expected_sha,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        completed_games=len(list((c.OUT/'progress').glob('game_*.json'))),qualification_only=c.SCOPE=='qualification',automatic_retries=0,
        requires_actual_reader_and_guardian_exit=True))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True)
    return run(parser.parse_args().freeze_sha256)


if __name__=='__main__':raise SystemExit(main())


