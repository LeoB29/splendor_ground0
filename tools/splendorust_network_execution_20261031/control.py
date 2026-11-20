"""One-clock owner and exact-output network execution controller."""
import argparse,os,sys,time,traceback,shutil,math,re
from copy import deepcopy
from pathlib import Path
from importlib.metadata import version
import psutil
from . import common as c,transport,adapters
from tools.splendorust_search256_recovery_20261030_02 import placement,qualification
from tools.splendorust_search256_efficiency_20261031 import analysis as panel_analysis
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
    common=dict(route='research',graph_request_id='request-00000001',entrypoint_request_id='request-00000002')
    return dict(localization=dict(common,profile_request_id='request-00000003',ordinary_first_request_id='request-00000004',
        ordinary_requests=0,startup_requests=3),**{name:dict(common,profile_request_id=None,
        ordinary_first_request_id='request-00000003',ordinary_requests=72,startup_requests=2) for name in c.VARIANTS})


def contract():
    return dict(schema='splendorust_network_execution_v1',phase=c.PHASE,
        job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,stage_seconds=c.STAGES,
        independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,automatic_retries=0,failed_decision_retries=0,
        seconds_per_move=c.SECONDS_PER_MOVE,internal_seconds=c.INTERNAL_SECONDS,
        operator_profile_external_seconds=120.,operator_profile_internal_seconds=90.,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,max_rss_bytes=c.MAX_RSS_BYTES,max_workers=c.MAX_WORKER_LIFECYCLES,
        initial_available_ram_bytes=c.INITIAL_AVAILABLE_RAM_BYTES,minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,
        initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
        placement_mode='isolated',max_resident_workers=2,execution_variants=['localization','control','candidate'],
        declared_worker_lifecycles=lifecycle_inventory(),qualification_scientific_decisions_maximum=201,
        graph_cases_maximum=33,profile_forward_calls=60,evaluation_requests=144,parent_requests_maximum=151,
        maximum_candidate_supplements=1,new_games=0,training=False,one_second_gate=False,seven_hundred_ms_gate=False)


def validate_freeze_contract(freeze):
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()),'Exact bounded two-stage network-execution contract')
    return expected


def load_inputs(freeze,envelope):
    c.verify([freeze[k] for k in ('panel','evaluation_matrix','qualification_plan','profile_plan')])
    selected=c.read(c.ROOT/freeze['panel']['path']);matrix=c.read(c.ROOT/freeze['evaluation_matrix']['path'])
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    panel_analysis.panel_roots(selected)
    c.require(len(matrix)==144 and [s['index'] for s in matrix]==list(range(144)),'Fixed144 matrix order')
    roots={x['root_id']:x for x in selected['roots']};seen=set()
    for index in range(0,144,2):
        pair=matrix[index:index+2]
        c.require({s['variant'] for s in pair}==set(c.VARIANTS) and
            len({(s['root_id'],s['repeat'],s['pair_index']) for s in pair})==1,'Exact paired same-root input')
        for s in pair:
            c.require(s['root_id'] in roots and s['panel_id']==s['root_id'] and s['repeat'] in range(4) and
                s['group']==roots[s['root_id']]['group'],'Known fixed root/variant/repeat')
            key=(s['root_id'],s['repeat'],s['variant']);c.require(key not in seen,'No extra or duplicate measurement');seen.add(key)
    c.require(len(seen)==144,'Complete18 by4 by2 matrix')
    c.require(plan['actual_suites']['256']==c.read(c.OLD_OUT/'qualification_plan.json')['actual_suites']['256'],
        'Unchanged actual corrected256 entrypoint suite')
    for item in selected['inputs']:envelope.guard();c.verify([item])
    return selected,matrix,plan


def qualify(name,workers,envelope,freeze,plan):
    child=transport.make_worker(name,envelope,freeze,route='research');workers[name]=child
    clock=adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
    profiles=c.read(child.output_dir/'entrypoint_profiles.json');transport._require_profiles(profiles)
    transport.validate_profiles(profiles['parent_profile'],profiles['child_profile'],clock,{'operation':'entrypoint_checks'},'research')
    source=c.OLD_OUT/'entrypoint_checks/research.json'
    c.require(clock['result']==c.read(source)['result'],'All67 full scientific outputs exactly match original256')
    proof=qualification.verify_entrypoint(child.output_dir,clock,plan,freeze,boundary=envelope.guard)
    c.save(c.OUT/'qualification'/f'{name}.json',dict(clock=c.artifact(child.output_dir/'entrypoint_clock.json'),
        source=c.artifact(source),verification=proof,scientific_decisions=67,graph_cases=11))
    return child


def verify_qualification(name,envelope,freeze,plan):
    folder=c.OUT/'workers'/name;clock=c.read(folder/'entrypoint_clock.json');source=c.OLD_OUT/'entrypoint_checks/research.json'
    c.require(clock['result']==c.read(source)['result'],'Saved actual suite complete scientific equality')
    proof=qualification.verify_entrypoint(folder,clock,plan,freeze,boundary=envelope.guard)
    expected=dict(clock=c.artifact(folder/'entrypoint_clock.json'),source=c.artifact(source),verification=proof,
        scientific_decisions=67,graph_cases=11)
    c.require(c.read(c.OUT/'qualification'/f'{name}.json')==expected,'Exact saved qualification proof')
    return expected


def profile_request(child,envelope,freeze):
    payload={'operation':'operator_profile'}
    c.require(child._counter==2 and envelope.remaining()>=120.,'One complete profile request after graph and suite')
    c.save(child.output_dir/'startup_inputs/request-00000003.json',payload)
    receipt=transport.request(child,payload,120.)
    profiles=transport.collect_profiles(child,receipt,envelope)
    c.save(c.OUT/'localization_clock.json',receipt);c.save(c.OUT/'localization_profiles.json',profiles)
    transport._require_profiles(profiles)
    c.require(receipt['status']=='ok','Operator profile administrative request failed')
    transport.validate_profiles(profiles['parent_profile'],profiles['child_profile'],receipt,payload,'research')
    from .profile_worker import verify_operator_profile
    proof=verify_operator_profile(child.output_dir,freeze,receipt['result'],boundary=envelope.guard)
    c.save(c.OUT/'localization_result.json',receipt['result']);return proof


def verify_localization(envelope,freeze,plan,*,writing):
    from .profile_worker import verify_operator_profile
    folder=c.OUT/'workers/localization';verify_qualification('localization',envelope,freeze,plan)
    receipt=c.read(c.OUT/'localization_clock.json');profiles=c.read(c.OUT/'localization_profiles.json')
    c.require(receipt==c.read(folder/'request-00000003.json') and receipt['result']==c.read(c.OUT/'localization_result.json'),
        'Original sole profile request and saved summary')
    transport._require_profiles(profiles)
    transport.validate_profiles(profiles['parent_profile'],profiles['child_profile'],receipt,{'operation':'operator_profile'},'research')
    profile=verify_operator_profile(folder,freeze,receipt['result'],boundary=envelope.guard)
    closure=transport.verify_closed_worker(folder,freeze,envelope)
    c.require(closure==c.read(c.OUT/'closed_workers/localization.json') and closure['raw']['requests']==3 and
        closure['raw']['failed_requests']==0,'Exactly3 passed localization requests with complete closure')
    result=dict(status='verified_localization',profile=profile,closed_worker=c.artifact(c.OUT/'closed_workers/localization.json'),
        localization_result=c.artifact(c.OUT/'localization_result.json'),scientific_decisions=67,graph_cases=11,
        raw_forward_calls=60,parent_requests=3,no_new_games=True)
    save_or_verify(c.OUT/'localization_verification.json',result,writing);return result


def verify_supplement(supplement,freeze,envelope):
    c.require(supplement['schema']=='splendorust_network_execution_candidate_v1' and
        supplement['phase1_freeze_sha256']==c.file_hash(c.OUT/'freeze.json'),'Candidate tied to original phase1 freeze')
    for key in ('localization_result','localization_verification'):
        c.require(supplement[key]==c.artifact(c.OUT/(key+'.json')),'Candidate decision binds passed localization')
    c.require(c.read(c.OUT/'localization_verification.json')['status']=='verified_localization','Selection follows full verified localization')
    c.require(type(supplement['candidate_id']) is str and re.fullmatch('[a-z][a-z0-9_]{0,63}',supplement['candidate_id']),
        'One named candidate')
    inputs=supplement['inputs'];c.require(len({x['path'] for x in inputs})==len(inputs),'Unique supplemental source inventory')
    for item in inputs:envelope.guard();c.verify([item])
    module=supplement['candidate_module'];name=supplement['candidate_import_module']
    c.require(type(name) is str and re.fullmatch(r'tools\.splendorust_network_execution_20261031\.candidate_[a-z0-9_]+',name),
        'Candidate import confined to one new implementation')
    c.require(module in inputs and module['path']==name.replace('.','/')+'.py' and
        module['path'] not in {x['path'] for x in freeze['inputs']},'New pinned candidate module; no frozen implementation overwritten')
    c.require(supplement['candidate_function']=='build_candidate','Exact additive candidate factory')
    for key in ('decision','model_free_tests','independent_review'):
        c.require(supplement[key] in inputs,'Pinned candidate '+key)
    tests=c.read(c.ROOT/supplement['model_free_tests']['path']);review=c.read(c.ROOT/supplement['independent_review']['path'])
    c.require(tests['exit_code']==0 and tests['models']==tests['games']==0,'Candidate preparation tests cannot consume unseen model trials')
    c.verify(tests['inputs']+[tests['log']])
    c.require(review['approved'] is True and not review.get('blocking_findings',[]),'Independent candidate implementation admission')
    c.verify(review['inputs'])
    c.require(all(x in inputs for x in tests['inputs']+review['inputs']),'All checked candidate sources included in supplement')
    return dict(status='passed',candidate_id=supplement['candidate_id'],module=module)


def wait_for_selection(envelope,freeze):
    c.require(not envelope.resident,'Selection waiting cannot retain model trees')
    candidate=c.OUT/'candidate_supplement.json';stop=c.OUT/'candidate_stop.json'
    c.save(c.OUT/'awaiting_candidate.json',dict(time=c.now(),phase1_freeze_sha256=c.file_hash(c.OUT/'freeze.json'),
        localization_result=c.artifact(c.OUT/'localization_result.json'),
        localization_verification=c.artifact(c.OUT/'localization_verification.json'),
        monotonic_deadline=envelope.deadline,unix_deadline=envelope.wall_deadline))
    while True:
        envelope.guard();c.require(not(candidate.exists() and stop.exists()),'Candidate and stop decisions are mutually exclusive')
        if stop.exists():
            value=c.read(stop)
            c.require(value['phase1_freeze_sha256']==c.file_hash(c.OUT/'freeze.json') and type(value['reason']) is str and value['reason'],
                'Explicit no-candidate decision identity')
            for key in ('localization_result','localization_verification'):
                c.require(value[key]==c.artifact(c.OUT/(key+'.json')),'No-candidate conclusion binds measured evidence')
            return None
        if candidate.exists():
            value=c.read(candidate);verify_supplement(value,freeze,envelope)
            c.require(not(c.OUT/'candidate_admission.json').exists(),'Single candidate commitment; no automatic repeat')
            return value
        if envelope.remaining()<=2.:
            c.save(c.OUT/'selection_expired.json',dict(reason='No frozen justified candidate before original selection allowance',
                phase1_freeze_sha256=c.file_hash(c.OUT/'freeze.json'),time=c.now(),monotonic=time.monotonic(),unix_time=time.time()))
            return None
        time.sleep(min(.25,envelope.remaining()/2))


def admit_candidate(supplement,freeze,envelope):
    c.require(not envelope.resident,'No model overlaps candidate commitment')
    verify_supplement(supplement,freeze,envelope)
    # The clock is not renewed by waiting for implementation or freezing it.
    required=2*(60+60+300)+144*c.SECONDS_PER_MOVE+120
    c.require(envelope.remaining()>required,'Remaining original candidate/model allowance must cover all maximum request clocks and120s closure')
    return c.save(c.OUT/'candidate_admission.json',dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time(),
        perf_counter_ns=time.perf_counter_ns(),phase1_freeze_sha256=c.file_hash(c.OUT/'freeze.json'),
        supplement=c.artifact(c.OUT/'candidate_supplement.json'),localization_verification=c.artifact(c.OUT/'localization_verification.json'),
        candidate_id=supplement['candidate_id'],required_seconds=required,
        original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start,
        effective_monotonic_deadline=envelope.deadline,effective_unix_deadline=envelope.wall_deadline,
        commitment_consumed=True,automatic_retries=0))


def verify_candidate_admission(admission,supplement,envelope):
    start=c.read(c.OUT/'stages/candidate_start.json')
    required=2*(60+60+300)+144*c.SECONDS_PER_MOVE+120
    c.require(admission['phase1_freeze_sha256']==c.file_hash(c.OUT/'freeze.json') and
        admission['candidate_id']==supplement['candidate_id'] and
        admission['supplement']==c.artifact(c.OUT/'candidate_supplement.json') and
        admission['localization_verification']==c.artifact(c.OUT/'localization_verification.json') and
        admission['original_monotonic_origin']==envelope.start and admission['original_unix_origin']==envelope.wall_start and
        admission['commitment_consumed'] is True and admission['automatic_retries']==0 and
        admission['required_seconds']==required,'Single original-clock candidate commitment identities')
    for key,deadline_key,origin in (('monotonic','monotonic_deadline',envelope.start),('unix_time','unix_deadline',envelope.wall_start)):
        effective='effective_'+deadline_key
        c.require(admission[effective]==start[deadline_key] and start[key]<=admission[key] and
            admission[key]+required<admission[effective]<=origin+c.COLLECTION_CUTOFF_SECONDS,
            'Complete saved remaining-clock candidate feasibility recomputed')
    return dict(status='passed',required_seconds=required)


def validate_row(row,spec,selected,freeze,envelope):
    envelope.guard();transport._require_profiles(row);transport.verify_saved_record(row,freeze)
    c.require(row['completed'] is True,'Any failed scientific request stops and is never repeated')
    c.require(row['variant']==spec['variant'] and row['root_id']==spec['root_id'] and
        row['worker_folder']==(c.OUT/'workers'/spec['variant']).relative_to(c.ROOT).as_posix(),'Exact scheduled execution variant')
    decode_training_snapshot(row['before']);decode_training_snapshot(row['after'])
    ledger.check_observer(row['before'],row['observer'])
    c.require(ledger.transition(row['before'],row['action'])==row['after'],'Independent lawful saved transition')
    return panel_analysis.validate_row(row,spec,selected)


def evaluate(workers,envelope,freeze,selected,matrix):
    roots={x['root_id']:x for x in selected['roots']};rows=[]
    for spec in matrix:
        envelope.guard();root=roots[spec['root_id']]
        row=transport.ordinary(workers[spec['variant']],spec,root['observer'],envelope,freeze)
        row.update(variant=spec['variant'],root_id=spec['root_id'],before=root['before'],
            after=root['after'] if row['completed'] else None,source_move=root['source_row'],seat=root['seat'])
        c.save(c.OUT/'observations'/f"{spec['index']:04d}.json",row)
        validate_row(row,spec,selected,freeze,envelope);rows.append(row)
        print(c.canonical(dict(phase='candidate',complete=len(rows),total=144,variant=spec['variant'])),flush=True)
    return rows


def verify_saved(envelope,freeze,*,writing):
    from . import analysis
    verify_runtime(freeze,envelope);selected,matrix,plan=load_inputs(freeze,envelope)
    verify_localization(envelope,freeze,plan,writing=False)
    admitted=(c.OUT/'candidate_admission.json').exists();rows=[];stages=['localization','selection'];workers=['localization']
    if admitted:
        stages.append('candidate');workers+=list(c.VARIANTS)
        supplement=c.read(c.OUT/'candidate_supplement.json');verify_supplement(supplement,freeze,envelope)
        admission=c.read(c.OUT/'candidate_admission.json')
        verify_candidate_admission(admission,supplement,envelope)
        c.require(not(c.OUT/'candidate_stop.json').exists() and not(c.OUT/'selection_expired.json').exists(),'No candidate after no-candidate outcome')
        c.require([p.name for p in sorted((c.OUT/'observations').glob('*.json'))]==[f'{i:04d}.json' for i in range(144)],'Exact144 measurements')
        for spec in matrix:
            row=c.read(c.OUT/'observations'/f"{spec['index']:04d}.json");validate_row(row,spec,selected,freeze,envelope);rows.append(row)
        for name in c.VARIANTS:
            folder=c.OUT/'workers'/name;verify_qualification(name,envelope,freeze,plan)
            c.require(c.read(folder/'initialization.json')['identity']['created']>admission['unix_time'],
                'Candidate commitment precedes every evaluation model initialization')
            ids=[r['clock']['request_id'] for r in rows if r['variant']==name]
            c.require(ids==[f'request-{i:08d}' for i in range(3,75)],'Exactly72 scientific requests per evaluation tree')
            proof=transport.verify_closed_worker(folder,freeze,envelope)
            c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json') and proof['raw']['requests']==74 and proof['raw']['failed_requests']==0,
                'Complete no-failure raw/clock/placement/graph/scientific closure')
        result=analysis.analyze_evaluation(rows,selected,matrix)
        save_or_verify(c.OUT/'evaluation_analysis.json',result,writing)
    else:
        c.require(not(c.OUT/'observations').exists() and not(c.OUT/'candidate_supplement.json').exists() and
            ((c.OUT/'candidate_stop.json').exists()!=(c.OUT/'selection_expired.json').exists()),'Exactly one disclosed localization-only outcome')
        result=None
    c.require({p.name for p in(c.OUT/'workers').iterdir() if p.is_dir()}==set(workers),'Exact worker lifecycle inventory')
    previous=None
    for stage in stages:
        start=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        for key,origin in (('monotonic',envelope.start),('unix_time',envelope.wall_start)):
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            c.require(start[key]<=end[key]<start[deadline]<=min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS),
                'Real original phase/model deadlines')
            if previous:c.require(previous[key]<=start[key],'No overlapping or reordered phases')
        previous=end
    c.require({p.stem[:-6] for p in(c.OUT/'stages').glob('*_start.json')}==set(stages+['final']),'Exact phase inventory')
    verify_exchanges(envelope,freeze)
    coverage=dict(status='verified_candidate_evaluation' if admitted else 'verified_localization_only',
        workers=len(workers),graph_cases=11*len(workers),scientific_qualification_decisions=67*len(workers),
        profile_forward_calls=60,ordinary_measurements=len(rows),parent_requests=3+(148 if admitted else 0),
        new_games=0,training=False,candidates_evaluated=int(admitted),automatic_retries=0,all_raw_science_and_clocks_verified=True)
    save_or_verify(c.OUT/'verification_summary.json',coverage,writing);verify_runtime(freeze,envelope);return coverage


SEALED_NAMES=('workers','observations','qualification','closed_workers','closed_worker_costs','localization_clock.json',
    'localization_profiles.json','localization_result.json','localization_verification.json','candidate_supplement.json',
    'candidate_stop.json','selection_expired.json','candidate_admission.json','awaiting_candidate.json','evaluation_analysis.json',
    'verification_summary.json','memory_admissions','placement_requests','placement_acks')


def final_checks(envelope,freeze):
    result=verify_saved(envelope,freeze,writing=True)
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=evidence_seal([c.OUT/n for n in SEALED_NAMES if(c.OUT/n).exists()],envelope)))
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact phase1 frozen network-execution plan')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Owning original watchdog freeze')
    envelope=Envelope(claim);envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    workers={};coverage=None;failure=None;status='incomplete'
    try:
        envelope.begin('localization');placement.configure_execution(freeze,'isolated')
        c.save(c.OUT/'runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);selected,matrix,plan=load_inputs(freeze,envelope)
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        child=qualify('localization',workers,envelope,freeze,plan);profile_request(child,envelope,freeze)
        transport.close_and_audit(child,envelope,freeze);workers={}
        verify_localization(envelope,freeze,plan,writing=True);verify_runtime(freeze,envelope);envelope.finish()
        envelope.begin('selection');supplement=wait_for_selection(envelope,freeze);envelope.finish()
        if supplement is not None:
            envelope.begin('candidate');admit_candidate(supplement,freeze,envelope)
            for name in c.VARIANTS:qualify(name,workers,envelope,freeze,plan)
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


