"""Bounded exact256 saved-position timing; observer admission before interpretation."""
import argparse,os,sys,time,traceback,shutil,math
from copy import deepcopy
from pathlib import Path
from importlib.metadata import version
import psutil
from . import common as c,transport,adapters,placement,panel,analysis,qualification,design
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


def contract():
    return dict(schema='splendorust_search256_efficiency_v1',phase=c.PHASE,
        job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,stage_seconds=c.STAGES,
        independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,automatic_retries=0,failed_decision_retries=0,
        seconds_per_move=c.SECONDS_PER_MOVE,internal_seconds=c.INTERNAL_SECONDS,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,max_rss_bytes=c.MAX_RSS_BYTES,max_workers=c.MAX_WORKERS,
        initial_available_ram_bytes=c.INITIAL_AVAILABLE_RAM_BYTES,minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,
        initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
        placement_mode='isolated',max_resident_workers=2,efficiency_observers=c.EFFICIENCY_OBSERVERS,
        declared_worker_lifecycles=design.lifecycle_inventory(),qualification_scientific_decisions=134,
        graph_cases=22,overhead_requests=48,diagnostic_requests=144,ordinary_requests_maximum=192,
        parent_requests_maximum=196,only_scientific_change='none_observation_only',
        new_games=0,training=False,one_second_gate=False,seven_hundred_ms_gate=False)


def validate_freeze_contract(freeze):
    expected=contract()
    c.require(all(freeze.get(key)==value for key,value in expected.items()),'Exact bounded efficiency contract')
    return expected


def load_inputs(freeze,envelope):
    c.verify([freeze['panel'],freeze['qualification_plan']])
    selected=c.read(c.ROOT/freeze['panel']['path']);plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(plan==qualification.build_plan(),'Preserved original actual-suite cases')
    for phase in ('overhead','diagnostic'):
        analysis.validate_layout(selected,panel.request_matrix(selected,phase),phase)
    for item in selected['inputs']:envelope.guard();c.verify([item])
    return selected,plan


def transition_workers(workers,envelope):
    c.require(set(workers)==set(c.OBSERVER_MODES),'Exactly two persistent observer modes')
    records=[]
    for mode,child in workers.items():
        c.require(not child._closed and child.recovery_accounting.state=='idle','Never change an active move clock')
        previous=child.absolute_deadline;child.absolute_deadline=envelope.effective_deadline()
        records.append(dict(mode=mode,worker=child.output_dir.name,previous_deadline=previous,
            deadline=child.absolute_deadline,request_counter=child._counter,active_request=False))
    return c.save(c.OUT/'worker_phase_transitions'/f'{envelope.stage}.json',dict(stage=envelope.stage,
        monotonic=time.monotonic(),unix_time=time.time(),workers=records))


def qualification_checks(workers,envelope,freeze,plan):
    original=c.read(c.OLD_OUT/'entrypoint_checks/research.json');c.require(original['status']=='ok','Accepted original256 suite')
    proofs=[]
    for mode in c.OBSERVER_MODES:
        child=transport.make_worker('observer_'+mode,envelope,freeze,route='research');workers[mode]=child
        clock=adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
        pin=c.save(c.OUT/'entrypoint_checks'/f'{mode}.json',clock)
        proof=qualification.verify_entrypoint(child.output_dir,clock,plan,freeze,boundary=envelope.guard)
        c.require(clock['result']==original['result'],'Entire unchanged256 scientific qualification result must reproduce')
        c.require(child._counter==2,'Only graph and complete suite before measurement')
        proofs.append(dict(mode=mode,clock=pin,verification=proof,original=c.artifact(c.OLD_OUT/'entrypoint_checks/research.json')))
    result=dict(status='passed',scientific_decisions=134,graph_cases=22,parent_requests=4,workers=2,receipts=proofs)
    c.save(c.OUT/'qualification_summary.json',result);return result


def verify_qualification(envelope,freeze,plan):
    original=c.read(c.OLD_OUT/'entrypoint_checks/research.json');proofs=[]
    for mode in c.OBSERVER_MODES:
        folder=c.OUT/'workers'/('observer_'+mode);clock=c.read(c.OUT/'entrypoint_checks'/f'{mode}.json')
        c.require(clock==c.read(folder/'entrypoint_clock.json') and clock['request_id']=='request-00000002',
            'Exact second-request complete qualification')
        proof=qualification.verify_entrypoint(folder,clock,plan,freeze,boundary=envelope.guard)
        c.require(clock['result']==original['result'],'Saved full256 qualification matches accepted reference')
        proofs.append(dict(mode=mode,clock=c.artifact(c.OUT/'entrypoint_checks'/f'{mode}.json'),verification=proof,
            original=c.artifact(c.OLD_OUT/'entrypoint_checks/research.json')))
    expected=dict(status='passed',scientific_decisions=134,graph_cases=22,parent_requests=4,workers=2,receipts=proofs)
    c.require(c.read(c.OUT/'qualification_summary.json')==expected,'Exact saved qualification inventory')
    return expected


def verify_row(row,spec,selected,freeze,envelope):
    envelope.guard();transport._require_profiles(row);transport.verify_saved_record(row,freeze)
    c.require(row['completed'] is True,'Failed diagnostic decisions remain failures; never retried')
    c.require(row['worker_folder']==(c.OUT/'workers'/('observer_'+spec['mode'])).relative_to(c.ROOT).as_posix(),
        'Frozen measurement mode has exact owning tree')
    decode_training_snapshot(row['before']);decode_training_snapshot(row['after'])
    ledger.check_observer(row['before'],row['observer'])
    c.require(ledger.transition(row['before'],row['action'])==row['after'],'Independent exact saved transition')
    return analysis.validate_row(row,spec,selected)


def collect_phase(phase,workers,envelope,freeze,selected):
    matrix=panel.request_matrix(selected,phase);roots={r['root_id']:r for r in selected['roots']};rows=[]
    for spec in matrix:
        envelope.guard();root=roots[spec['root_id']]
        row=transport.ordinary(workers[spec['mode']],spec,root['observer'],envelope,freeze)
        row.update(before=root['before'],after=root['after'] if row['completed'] else None,
            source_move=root['source_row'],seat=root['seat'],player_role='challenger',mode='research',quota=256,arm='search')
        c.save(c.OUT/'observations'/phase/f"{spec['index']:04d}.json",row)
        verify_row(row,spec,selected,freeze,envelope);rows.append(row)
        print(c.canonical(dict(phase=phase,completed=len(rows),total=len(matrix),mode=spec['mode'])),flush=True)
    return rows


def read_phase(phase,selected,freeze,envelope):
    matrix=panel.request_matrix(selected,phase);folder=c.OUT/'observations'/phase
    c.require([p.name for p in sorted(folder.glob('*.json'))]==[f"{x['index']:04d}.json" for x in matrix],
        'Complete exact ordered fixed phase; no missing or extra measurement')
    rows=[]
    for spec in matrix:
        row=c.read(folder/f"{spec['index']:04d}.json");verify_row(row,spec,selected,freeze,envelope);rows.append(row)
    return rows


def save_or_verify(path,value,writing):
    if writing:return c.save(path,value)
    c.require(c.read(path)==value,'Independent saved result differs: '+str(path));return c.artifact(path)


def verify_saved(envelope,freeze,*,writing):
    selected,plan=load_inputs(freeze,envelope);verify_runtime(freeze,envelope)
    verify_qualification(envelope,freeze,plan)
    overhead=read_phase('overhead',selected,freeze,envelope);gate=analysis.analyze_overhead(overhead,selected)
    c.require(c.read(c.OUT/'overhead_result.json')==gate,'Exact independently recomputed overhead admission')
    rows=list(overhead);phases=['qualification','overhead'];diagnostic=None
    if gate['passed']:
        admission=c.read(c.OUT/'diagnostic_admission.json')
        c.require(admission['overhead']==c.artifact(c.OUT/'overhead_result.json') and admission['freeze']==c.artifact(c.OUT/'freeze.json'),
            'Complete fixed overhead binds conditional diagnostic admission')
        diagnostic=read_phase('diagnostic',selected,freeze,envelope);rows.extend(diagnostic);phases.append('diagnostic')
        c.require(all(r['parent_profile']['request_start_ns']>admission['perf_counter_ns'] for r in diagnostic),
            'No conditional measurement precedes passed overhead gate')
        summary=analysis.analyze_diagnostic(diagnostic,selected)
        save_or_verify(c.OUT/'diagnostic_analysis.json',summary,writing)
    else:
        c.require(not(c.OUT/'diagnostic_admission.json').exists() and not(c.OUT/'observations/diagnostic').exists(),
            'Rejected observer cannot run or claim full-panel measurements')
    c.require({p.name for p in(c.OUT/'workers').iterdir() if p.is_dir()}==set(c.EFFICIENCY_OBSERVERS),'Exactly two declared persistent trees')
    closures=[]
    for mode in c.OBSERVER_MODES:
        folder=c.OUT/'workers'/('observer_'+mode);ids=[r['clock']['request_id'] for r in rows if r['spec']['mode']==mode]
        c.require(ids==[f'request-{i:08d}' for i in range(3,3+len(ids))] and
            {p.stem for p in(folder/'inputs').glob('*.json')}==set(ids),'Exact ordinary request count and no hidden warmup/retry')
        proof=transport.verify_closed_worker(folder,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{folder.name}.json'),'Exact complete worker closure')
        c.require(proof['raw']['requests']==len(ids)+2 and proof['raw']['failed_requests']==0,'No unreported request failure')
        closures.append(proof)
    previous=None
    for stage in phases:
        start=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        for key,origin in (('monotonic',envelope.start),('unix_time',envelope.wall_start)):
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            c.require(start[key]<=end[key]<start[deadline]<=min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS),
                'Real stage/model deadline preserved')
            if previous:c.require(previous[key]<=start[key],'Phases do not overlap or reorder')
        previous=end
    c.require({p.stem[:-6] for p in(c.OUT/'stages').glob('*_start.json')}==set(phases+['final']),
        'Exact attempted stage inventory')
    verify_exchanges(envelope,freeze)
    result=dict(status='verified_full_diagnostic' if gate['passed'] else 'verified_observer_overhead_rejected',
        overhead_passed=gate['passed'],overhead_requests=48,diagnostic_requests=144 if diagnostic is not None else 0,
        parent_requests=len(rows)+4,scientific_qualification_decisions=134,graph_cases=22,workers=2,
        exact_historical_decisions=len(rows),new_games=0,training=False,automatic_retries=0,
        all_raw_logs_graph_science_clock_placement_and_sidecars_verified=True)
    save_or_verify(c.OUT/'verification_summary.json',result,writing);verify_runtime(freeze,envelope)
    return result


def final_checks(envelope,freeze):
    result=verify_saved(envelope,freeze,writing=True)
    paths=[c.OUT/name for name in ('workers','observations','entrypoint_checks','closed_workers','closed_worker_costs',
        'qualification_summary.json','overhead_result.json','diagnostic_admission.json','diagnostic_analysis.json',
        'verification_summary.json','worker_phase_transitions','memory_admissions','placement_requests','placement_acks')]
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=evidence_seal([p for p in paths if p.exists()],envelope)))
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact frozen diagnostic')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Owning watchdog freeze')
    envelope=Envelope(claim);envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    workers={};failure=None;coverage=None;status='incomplete'
    try:
        envelope.begin('qualification');placement.configure_execution(freeze,'isolated')
        c.save(c.OUT/'runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);selected,plan=load_inputs(freeze,envelope)
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        qualification_checks(workers,envelope,freeze,plan);envelope.finish()
        envelope.begin('overhead');transition_workers(workers,envelope)
        overhead=collect_phase('overhead',workers,envelope,freeze,selected)
        gate=analysis.analyze_overhead(overhead,selected);c.save(c.OUT/'overhead_result.json',gate)
        if gate['passed']:
            envelope.finish();envelope.begin('diagnostic');transition_workers(workers,envelope)
            c.save(c.OUT/'diagnostic_admission.json',dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),
                perf_counter_ns=time.perf_counter_ns(),overhead=c.artifact(c.OUT/'overhead_result.json'),freeze=c.artifact(c.OUT/'freeze.json')))
            collect_phase('diagnostic',workers,envelope,freeze,selected)
        for child in workers.values():transport.close_and_audit(child,envelope,freeze)
        workers={};envelope.finish();envelope.begin('final');coverage=final_checks(envelope,freeze)
        envelope.finish();status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        for mode,child in workers.items():
            if child._closed:continue
            try:transport.close_and_audit(child,envelope,freeze)
            except BaseException:c.save(c.OUT/'cleanup_failures'/f'{mode}.json',dict(error=traceback.format_exc()))
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
        print(c.canonical(dict(status='failed_preserved',stage=envelope.stage,error=failure)),flush=True)
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,coverage=coverage,freeze_sha256=expected_sha,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        new_games=0,training=False,automatic_retries=0,requires_actual_reader_and_guardian_exit=True))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True)
    return run(parser.parse_args().freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
