"""One-clock practical execution qualification and fixed saved-decision assessment."""
import argparse,os,sys,time,traceback,shutil,math,re
from copy import deepcopy
from pathlib import Path
from importlib.metadata import version
import psutil
from . import common as c,transport,adapters
from tools.splendorust_search256_recovery_20261030_02 import placement,qualification
from . import analysis as panel_analysis
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
        from tools.splendorust_search256_accounting_20261029 import WorkerAccounting
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
    return {name:dict(route='research',graph_request_id='request-00000001',
        numerical_request_id='request-00000002',entrypoint_request_id='request-00000003',
        ordinary_first_request_id='request-00000004',ordinary_requests=72,startup_requests=3)
        for name in c.VARIANTS}


def contract():
    return dict(schema='splendorust_network_practical_v1',phase=c.PHASE,
        job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,stage_seconds=c.STAGES,
        independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,automatic_retries=0,failed_decision_retries=0,
        seconds_per_move=c.SECONDS_PER_MOVE,internal_seconds=c.INTERNAL_SECONDS,
        numerical_checks_external_seconds=60.,numerical_checks_internal_seconds=45.,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,max_rss_bytes=c.MAX_RSS_BYTES,max_workers=c.MAX_WORKER_LIFECYCLES,
        initial_available_ram_bytes=c.INITIAL_AVAILABLE_RAM_BYTES,minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,
        initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
        placement_mode='isolated',max_resident_workers=2,execution_variants=list(c.VARIANTS),
        declared_worker_lifecycles=lifecycle_inventory(),qualification_scientific_decisions_maximum=134,
        graph_cases_maximum=22,fresh_numerical_cases_per_worker=38,fresh_numerical_cases_maximum=76,
        profile_forward_calls=0,evaluation_requests=144,parent_requests_maximum=150,
        maximum_candidates=1,new_games=0,training=False,one_second_gate=False,seven_hundred_ms_gate=False,
        action_differences_stop_collection=False,candidate_numerical_rejection_is_valid_negative=True,
        candidate_repeat_self_equality_required=True,control_historical_equality_required=True,
        numerical_tolerances=dict(policy_atol=1e-5,policy_rtol=1e-6,value_atol=1e-6,value_rtol=0.0))


def validate_freeze_contract(freeze):
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()),'Exact bounded practical-execution contract')
    c.require(freeze['candidate_module']==c.artifact(c.CANDIDATE_SOURCE) and
        freeze['candidate_module']['sha256']==c.CANDIDATE_SOURCE_SHA and
        freeze['candidate_import_module']==c.CANDIDATE_MODULE and freeze['candidate_id']==c.CANDIDATE_ID and
        freeze['candidate_function']=='build_candidate','One unchanged held candidate implementation')
    return expected


def load_inputs(freeze,envelope):
    from . import analysis
    c.verify([freeze[k] for k in ('panel','evaluation_matrix','qualification_plan','numerical_plan')])
    selected=c.read(c.ROOT/freeze['panel']['path']);matrix=c.read(c.ROOT/freeze['evaluation_matrix']['path'])
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(matrix==analysis.evaluation_matrix(selected) and len(matrix)==144,'Exact fresh18 by4 paired matrix')
    c.require(plan['actual_suites']['256']==c.read(c.OLD_OUT/'qualification_plan.json')['actual_suites']['256'],
        'Unchanged actual corrected256 entrypoint suite')
    for item in selected['inputs']:envelope.guard();c.verify([item])
    return selected,matrix,plan


def qualification_evidence(name,envelope,freeze,plan,*,writing):
    folder=c.OUT/'workers'/name
    graph=transport.verify_graph_evidence(folder,freeze,boundary=envelope.guard)
    numeric=suite=None;rejected_at=None
    if not graph['accepted']:rejected_at='graph'
    else:
        numeric=transport.verify_numerical_evidence(folder,freeze,boundary=envelope.guard)
        if not numeric['accepted']:rejected_at='numerical'
        else:
            clock=c.read(folder/'entrypoint_clock.json');profiles=c.read(folder/'entrypoint_profiles.json')
            transport._require_profiles(profiles)
            transport.validate_profiles(profiles['parent_profile'],profiles['child_profile'],clock,{'operation':'entrypoint_checks'},'research')
            source=c.OLD_OUT/'entrypoint_checks/research.json'
            if name=='control':c.require(clock['result']==c.read(source)['result'],'Control full scientific suite historical equality')
            proof=qualification.verify_entrypoint(folder,clock,plan,freeze,boundary=envelope.guard)
            suite=dict(clock=c.artifact(folder/'entrypoint_clock.json'),verification=proof,
                control_historical_source=c.artifact(source) if name=='control' else None)
    c.require(name=='candidate' or rejected_at is None,'Unchanged control qualification must pass')
    result=dict(status='qualified' if rejected_at is None else 'valid_numerical_rejection',variant=name,
        accepted=rejected_at is None,rejected_at=rejected_at,graph=graph,numerical=numeric,suite=suite,
        graph_cases=11,numerical_cases=38 if numeric is not None else 0,
        scientific_decisions=67 if suite is not None else 0,parent_requests=1+int(numeric is not None)+int(suite is not None))
    save_or_verify(c.OUT/'qualification'/f'{name}.json',result,writing);return result


def qualify(name,workers,envelope,freeze,plan):
    child=transport.make_worker(name,envelope,freeze,route='research');workers[name]=child
    graph=transport.verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)
    if graph['accepted']:
        adapters.numerical_checks(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
        numerical=transport.verify_numerical_evidence(child.output_dir,freeze,boundary=envelope.guard)
        if numerical['accepted']:
            c.require(child._counter==2 and envelope.remaining()>=300.,'Full actual suite after graph/fresh numerical qualification')
            adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
    return qualification_evidence(name,envelope,freeze,plan,writing=True)


def qualification_outcome(results):
    c.require(set(results)==set(c.VARIANTS) and results['control']['accepted'],'Both fixed variants and passed control')
    accepted=results['candidate']['accepted']
    return dict(status='qualified' if accepted else 'valid_candidate_numerical_rejection',accepted=accepted,
        rejected_at=results['candidate']['rejected_at'],workers=2,
        graph_cases=sum(x['graph_cases'] for x in results.values()),
        numerical_cases=sum(x['numerical_cases'] for x in results.values()),
        scientific_decisions=sum(x['scientific_decisions'] for x in results.values()),
        startup_parent_requests=sum(x['parent_requests'] for x in results.values()))


def evaluation_admission(envelope,freeze):
    required=144*c.SECONDS_PER_MOVE+120.
    c.require(envelope.remaining()>required,'Original evaluation/model clock must cover144 full requests and120s closure')
    return c.save(c.OUT/'evaluation_admission.json',dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),
        freeze_sha256=c.file_hash(c.OUT/'freeze.json'),qualification_summary=c.artifact(c.OUT/'qualification_summary.json'),
        original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start,
        effective_monotonic_deadline=envelope.deadline,effective_unix_deadline=envelope.wall_deadline,
        required_seconds=required,automatic_retries=0))


def advance_idle_worker(child,envelope):
    from tools.splendorust_network_practical_handoff_20261031 import advance_idle_worker as corrected_handoff
    return corrected_handoff(child,envelope,common=c)


def actual_after(before,action):
    """Two independent corrected transitions, including a candidate-selected action."""
    state=decode_training_snapshot(before)
    environment=rules.CorrectedEnv()
    chosen=rules.deserialize_action(action)
    c.require(chosen in environment.legal_actions(state),'Actual complete action is legal in corrected environment')
    after=rules.serialize_snapshot(environment.step(state,chosen))
    c.require(after==ledger.transition(before,action),'Corrected environment and independent ledger transitions agree')
    decode_training_snapshot(after);return after


def validate_row(row,spec,selected,freeze,envelope):
    from . import analysis
    envelope.guard();transport._require_profiles(row);transport.verify_saved_record(row,freeze)
    c.require(row['completed'] is True,'Any failed request remains failed and is never replaced')
    c.require(row['variant']==spec['variant'] and row['root_id']==spec['root_id'] and
        row['worker_folder']==(c.OUT/'workers'/spec['variant']).relative_to(c.ROOT).as_posix(),'Exact scheduled fixed variant')
    ledger.check_observer(row['before'],row['observer'])
    c.require(actual_after(row['before'],row['action'])==row['after'],'Actual chosen-action transition independently reconstructed')
    from tools.splendorust_search256_recovery_20261030_02 import scientific
    scientific.validate_result(row['clock']['result'],row['observer'],'search',expected_quota=256)
    scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',before=row['before'],after=row['after'],expected_quota=256)
    return analysis.validate_row(row,spec,selected)


def evaluate(workers,envelope,freeze,selected,matrix):
    roots={r['root_id']:r for r in selected['roots']};rows=[]
    for spec in matrix:
        envelope.guard();root=roots[spec['root_id']]
        row=transport.ordinary(workers[spec['variant']],spec,root['observer'],envelope,freeze)
        row.update(variant=spec['variant'],root_id=spec['root_id'],before=root['before'],after=None,
            source_move=root['source_row'],seat=root['seat'])
        if row['completed']:
            try:row['after']=actual_after(root['before'],row['action'])
            except BaseException:
                row['transition_error']=traceback.format_exc()
                c.save(c.OUT/'observations'/f"{spec['index']:04d}.json",row);raise
        c.save(c.OUT/'observations'/f"{spec['index']:04d}.json",row)
        validate_row(row,spec,selected,freeze,envelope);rows.append(row)
        print(c.canonical(dict(phase='evaluation',complete=len(rows),total=144,variant=spec['variant'])),flush=True)
    return rows


def verify_evaluation_admission(envelope):
    value=c.read(c.OUT/'evaluation_admission.json');start=c.read(c.OUT/'stages/evaluation_start.json')
    required=144*c.SECONDS_PER_MOVE+120.
    c.require(value['freeze_sha256']==c.file_hash(c.OUT/'freeze.json') and
        value['qualification_summary']==c.artifact(c.OUT/'qualification_summary.json') and
        value['original_monotonic_origin']==envelope.start and value['original_unix_origin']==envelope.wall_start and
        value['required_seconds']==required and value['automatic_retries']==0,'Original evaluation admission identity')
    for key,deadline,origin in [('monotonic','monotonic_deadline',envelope.start),('unix_time','unix_deadline',envelope.wall_start)]:
        c.require(start[key]<=value[key] and value[key]+required<value['effective_'+deadline]==start[deadline] and
            start[deadline]<=origin+c.COLLECTION_CUTOFF_SECONDS,'Complete original-clock remaining allowance recomputes')
    return value


def verify_phase_transition(name,envelope,rows):
    transition=c.read(c.OUT/'worker_phase_transitions'/f'{name}.json');folder=c.OUT/'workers'/name
    before=c.read(c.OUT/'stages/qualification_start.json');after=c.read(c.OUT/'stages/evaluation_start.json')
    admit=c.read(c.OUT/'evaluation_admission.json');init=c.read(folder/'initialization.json')
    suite=c.read(folder/'parent_profiles/request-00000003.json');first=rows[0]['parent_profile']
    c.require(transition['variant']==name and transition['worker_pid']==init['identity']['pid'] and
        transition['worker_created']==init['identity']['created'] and transition['completed_startup_requests']==3 and
        transition['next_request_id']=='request-00000004' and transition['active_request'] is False and
        transition['request_allowance_unchanged']==10. and transition['original_monotonic_origin']==envelope.start and
        transition['original_unix_origin']==envelope.wall_start,'Idle phase transition identity/allowance')
    c.require(abs(transition['old_absolute_deadline']-before['monotonic_deadline'])<.1 and
        transition['new_absolute_deadline']==after['monotonic_deadline']<=envelope.start+c.COLLECTION_CUTOFF_SECONDS,
        'Retained worker deadline is clamped to original stage/model clock')
    c.require(admit['monotonic']<=transition['monotonic'] and
        suite['capture_snapshot_ns']<=transition['perf_counter_ns']<=first['request_start_ns'],
        'Completed suite before idle transition before first ordinary clock')
    return transition


def verify_saved(envelope,freeze,*,writing):
    from . import analysis
    verify_runtime(freeze,envelope);selected,matrix,plan=load_inputs(freeze,envelope)
    results={name:qualification_evidence(name,envelope,freeze,plan,writing=False) for name in c.VARIANTS}
    outcome=qualification_outcome(results)
    c.require(outcome==c.read(c.OUT/'qualification_summary.json'),'Independent qualification result/counts')
    rows=[];stages=['qualification']
    if outcome['accepted']:
        stages.append('evaluation');verify_evaluation_admission(envelope)
        c.require([p.name for p in sorted((c.OUT/'observations').glob('*.json'))]==[f'{i:04d}.json' for i in range(144)],
            'Exact complete144 measurements; no changed-action filtering')
        for spec in matrix:
            row=c.read(c.OUT/'observations'/f"{spec['index']:04d}.json");validate_row(row,spec,selected,freeze,envelope);rows.append(row)
        for name in c.VARIANTS:
            arm=[row for row in rows if row['variant']==name]
            c.require([r['clock']['request_id'] for r in arm]==[f'request-{i:08d}' for i in range(4,76)],'Exactly72 plain requests per variant')
            verify_phase_transition(name,envelope,arm)
        analysis_result=analysis.analyze_evaluation(rows,selected,matrix)
        save_or_verify(c.OUT/'evaluation_analysis.json',analysis_result,writing)
    else:
        c.require(not(c.OUT/'observations').exists() and not(c.OUT/'evaluation_admission.json').exists() and
            not(c.OUT/'worker_phase_transitions').exists() and not(c.OUT/'evaluation_analysis.json').exists(),
            'Early valid numerical rejection has no ordinary timing or implied agreement result')
    c.require({p.name for p in(c.OUT/'workers').iterdir() if p.is_dir()}==set(c.VARIANTS),'Exact two fixed worker lifecycles')
    c.require({p.stem for p in(c.OUT/'qualification').glob('*.json')}==set(c.VARIANTS),'Exact qualification receipts')
    c.require({p.stem for p in(c.OUT/'closed_workers').glob('*.json')}==set(c.VARIANTS),'Exact two closure receipts')
    if outcome['accepted']:
        c.require({p.stem for p in(c.OUT/'worker_phase_transitions').glob('*.json')}==set(c.VARIANTS),'Exact retained-worker phase transitions')
    for name in c.VARIANTS:
        proof=transport.verify_closed_worker(c.OUT/'workers'/name,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json') and proof['raw']['failed_requests']==0 and
            proof['raw']['requests']==results[name]['parent_requests']+(72 if outcome['accepted'] else 0),
            'Complete accepted transport/science prefix and actual cleanup')
    previous=None
    for stage in stages:
        start=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        c.require(start['cap_seconds']==c.STAGES[stage],'Frozen phase-specific duration')
        for key,origin in [('monotonic',envelope.start),('unix_time',envelope.wall_start)]:
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            c.require(start[key]<=end[key]<start[deadline]==min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS),
                'Original dual-clock phase/model limits')
            if previous:c.require(previous[key]<=start[key],'No reordered or overlapping phase clocks')
        previous=end
    c.require({p.stem[:-6] for p in(c.OUT/'stages').glob('*_start.json')}==set(stages+['final']),'Exact phase inventory')
    verify_exchanges(envelope,freeze)
    coverage=dict(status='verified_practical_evaluation' if outcome['accepted'] else 'verified_candidate_numerical_rejection',
        workers=2,graph_cases=outcome['graph_cases'],fresh_numerical_cases=outcome['numerical_cases'],
        scientific_qualification_decisions=outcome['scientific_decisions'],ordinary_measurements=len(rows),
        parent_requests=outcome['startup_parent_requests']+len(rows),early_numerical_rejection=not outcome['accepted'],
        rejection_stage=outcome['rejected_at'],new_games=0,training=False,candidates_attempted=1,
        automatic_retries=0,all_raw_science_and_clocks_verified=True)
    save_or_verify(c.OUT/'verification_summary.json',coverage,writing);verify_runtime(freeze,envelope);return coverage


SEALED_NAMES=('workers','observations','qualification','qualification_summary.json','closed_workers','closed_worker_costs',
    'evaluation_admission.json','worker_phase_transitions','evaluation_analysis.json','verification_summary.json',
    'memory_admissions','placement_requests','placement_acks')


def final_checks(envelope,freeze):
    result=verify_saved(envelope,freeze,writing=True)
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=evidence_seal([c.OUT/n for n in SEALED_NAMES if(c.OUT/n).exists()],envelope)))
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact practical follow-up freeze')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Original owning watchdog freeze')
    envelope=Envelope(claim);envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    workers={};coverage=None;failure=None;status='incomplete'
    try:
        envelope.begin('qualification');placement.configure_execution(freeze,'isolated')
        c.save(c.OUT/'runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);selected,matrix,plan=load_inputs(freeze,envelope)
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        results={name:qualify(name,workers,envelope,freeze,plan) for name in c.VARIANTS}
        outcome=qualification_outcome(results);c.save(c.OUT/'qualification_summary.json',outcome)
        if not outcome['accepted']:
            for child in workers.values():transport.close_and_audit(child,envelope,freeze)
            workers={}
        envelope.finish()
        if outcome['accepted']:
            envelope.begin('evaluation');evaluation_admission(envelope,freeze)
            for child in workers.values():advance_idle_worker(child,envelope)
            evaluate(workers,envelope,freeze,selected,matrix)
            for child in workers.values():transport.close_and_audit(child,envelope,freeze)
            workers={};envelope.finish()
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
        new_games=0,training=False,automatic_retries=0,requires_actual_reader_and_guardian_exit=True))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True)
    return run(parser.parse_args().freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
