"""Fresh-clock policy-head-only fit and fixed matched128 assessment; no collection."""
import argparse,os,sys,time,traceback,shutil,math,re
from pathlib import Path
from importlib.metadata import version
import psutil
from . import common as c,clock_contract as clocks
from tools.splendorust_search256_recovery_20261030_02 import placement
from tools.splendorust_search256_efficiency_20261031.common import validate_exchange
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules


def lifecycle_inventory():
    return {}


def build_plan():
    return dict(schema='policy_head_fit_only_plan_v1', source_games_reused=256,
        new_source_games=0, new_source_workers=0, new_source_reproductions=0,
        training_games=192, training_roots=10660, epochs=2, optimizer_steps=668,
        trainable_tensors=['policy.weight','policy.bias'], trainable_parameters=20817,
        frozen_groups=['shared','value'], learned_adapter_decisions=6,
        actual_corrected_tensor_and_optimizer_scope_qualification_inside_fitter=True,
        operational_games=6, assessment_games=400, maximum_assessment_reproductions=96,
        automatic_retries=0, reference_promoted=False)


def contract():
    return dict(schema=c.SCHEMA, phase=c.PHASE, job_seconds=c.JOB_SECONDS,
        clock_contract=clocks.runtime_contract(), collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES, independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        automatic_retries=0, failed_decision_retries=0, game_replacements=0,
        seconds_per_move=c.SECONDS_PER_MOVE, internal_seconds=c.INTERNAL_SECONDS,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES, prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES, max_rss_bytes=c.MAX_INFERENCE_RSS_BYTES,
        fitter_max_rss_bytes=c.MAX_FITTER_RSS_BYTES, max_workers=c.MAX_WORKER_LIFECYCLES,
        max_resident_workers=2, initial_available_ram_bytes=0, minimum_available_ram_bytes=0,
        initial_ram_stability_seconds=0, maximum_ram_wait_seconds=0,
        initial_free_disk_bytes=c.MINIMUM_FREE_DISK_BYTES, minimum_free_disk_bytes=c.MINIMUM_FREE_DISK_BYTES,
        placement_mode='isolated', declared_worker_lifecycles={}, maximum_live_decisions=c.MAX_LIVE_DECISIONS,
        maximum_replay_decisions=96, maximum_plies_per_game=c.MAX_PLIES,
        max_response_bytes=c.MAX_RESPONSE_BYTES, max_raw_log_bytes=c.MAX_LOG_BYTES,
        world_pool=3, cpuct=.4, search_depth=64, original_full_network=True,
        one_second_gate=False, seven_hundred_ms_gate=False, source_games=256,
        new_source_games=0, source_worker_lifecycles=0, source_replay_decisions=0,
        optimizer_steps=668, epochs=2, training_sample_reuse=True,
        strength_promotion=False, learned_adapter_decisions=6, learned_adapter_simulations=128,
        qualification_only=False, pilot_data=False, reused_corrected_pilot_data=True,
        resource_forecast_policy='advisory_only', resource_forecasts_may_stop=False,
        available_ram_worker_qualification_gate=False, fresh_owned_clock=True,
        trainable_tensors=['policy.weight','policy.bias'], frozen_parameter_groups=['shared','value'])


def validate_freeze_contract(freeze):
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()), 'Exact policy-head-only fit/assessment contract')
    c.require(c.SCOPE=='pilot' and c.JOB_SECONDS==72*3600 and c.COLLECTION_CUTOFF_SECONDS==60*3600,
        'Fresh72-hour job and60-hour model clock')
    return expected


def load_inputs(freeze,envelope):
    from . import preparation
    c.verify([freeze[k] for k in ('training_plan','fit_plan','fit_parity_fixtures','reused_corpus')])
    reused=preparation.verify_reused_corpus(freeze,boundary=envelope.guard)
    c.require(set(reused)=={'manifest','source_specs','proof'} and len(reused['source_specs'])==256,
        'Exactly the accepted256-game corrected corpus is reused')
    plan=c.read(c.ROOT/freeze['fit_plan']['path'])
    c.require(plan['source_specs']==reused['source_specs'] and
        plan['training_plan']==c.read(c.ROOT/freeze['training_plan']['path']), 'Exact fixed fit recipe and original corpus schedule')
    return reused

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
        self.prior_qualified_source_workers = 0
        self.evidence = None
        self.resident = {}
        self.peak_combined_rss = 0
        self.last_resource_guard = -float("inf")
        self.continuation = None  # No old clock, evidence or worker accounting is inherited.

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
                  'Fresh study worker lifecycle bound')
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
            c.require(shutil.disk_usage(c.ROOT).free >= c.MINIMUM_FREE_DISK_BYTES, 'Actual free-disk floor')
            c.save(c.OUT/'memory_admissions'/'000001.json', dict(time=c.now(),
                observed_available_bytes=psutil.virtual_memory().available,
                ram_reserve_policy='advisory_only', wait_seconds=0., cumulative_wait_seconds=0., stable_seconds=0.))
        self.admitted=True
        self.admissions+=1

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
    for folder in ((c.OUT/'workers').iterdir() if (c.OUT/'workers').exists() else []):
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
    paths=dict(manifest=c.ROOT/freeze['reused_corpus']['path'],plan=c.ROOT/freeze['fit_plan']['path'],fixtures=c.ROOT/freeze['fit_parity_fixtures']['path'])
    command=[sys.executable,'-B','-u','-m','tools.splendorust_policy_head_20261104.training','--mode',c.SCOPE]
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
    reused=load_inputs(freeze,envelope)
    return training.verify_pilot_fit(c.OUT/'fit',reused['manifest'],
        c.read(c.ROOT/freeze['fit_plan']['path']),c.read(c.ROOT/freeze['fit_parity_fixtures']['path'])['fixtures'],
        expected_specs=reused['source_specs'],boundary=envelope.guard)


def verify_stage_intervals(envelope):
    stages=['fit','assessment_qualification','operational','assessment','reproduction']
    previous=None
    for stage in stages:
        start=c.read(c.OUT/'stages'/f'{stage}_start.json')
        end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        for key,origin,deadline in [('monotonic',envelope.start,'monotonic_deadline'),
                                   ('unix_time',envelope.wall_start,'unix_deadline')]:
            expected=min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS,origin+c.JOB_SECONDS)
            c.require(origin<=start[key]<=end[key]<start[deadline]==expected,
                'Recorded stage intervals retain both original fresh owner deadlines')
            if previous:c.require(previous[key]<=start[key], 'No stage overlap or reordered model work')
        c.require(start['cap_seconds']==c.STAGES[stage], 'Exact stage allowance')
        previous=end
    c.require({p.name for p in (c.OUT/'stages').glob('*_start.json')}==
              {f'{name}_start.json' for name in stages+['final']}, 'Exact fit-only stage inventory')
    return stages


def verify_saved(envelope,freeze,*,writing):
    reused=load_inputs(freeze,envelope)
    c.require(not any((c.OUT/name).exists() for name in ('games','workers','capture_fixtures','replays','corpus_manifest.json')),
        'No new source games, collector, source replay or relabelled corpus')
    admission=c.read(c.OUT/'source_reuse_admission.json')
    c.require(admission['reused_corpus']==freeze['reused_corpus'] and
        admission['source_reuse_proof']==freeze['source_reuse_proof'] and admission['reused_source_games']==256 and
        admission['new_source_games']==0 and admission['automatic_retries']==0,
        'Previously accepted corpus is reused under its exact original identity')
    fit=verify_fit(envelope,freeze)
    fit_result=c.read(c.OUT/'fit/fit_result.json')
    c.require(fit_result['optimizer_steps']==668 and fit_result['training_roots']==10660 and
        fit_result['training_games']==192, 'Unchanged fixed training sample and two-epoch update count')
    verify_exchanges(envelope,freeze)
    from .assessment import control as assessment
    assessment_result=assessment.verify_saved(envelope,freeze,writing=writing)
    verify_stage_intervals(envelope)
    result=dict(status='verified_policy_head_fit_and_assessment', reused_source_games=256, new_source_games=0,
        source_worker_lifecycles=0, source_replay_decisions=0, optimizer_steps=668,
        training_games=192,training_roots=10660,fit_direct_forward_invocations=fit_result['direct_forward_invocations'],
        learned_adapter_decisions=6,learned_adapter_simulations=128,fit=fit,assessment=assessment_result,
        source_reuse_proof=freeze['source_reuse_proof'],reused_corpus=freeze['reused_corpus'],
        fixed_trainable_tensors=['policy.weight','policy.bias'], frozen_parameter_groups=['shared','value'],
        strength_promotion=False,automatic_retries=0,game_replacements=0,
        resource_forecasts_advisory=True,fresh_owned_clock=True)
    save_or_verify(c.OUT/'verification_summary.json',result,writing)
    verify_runtime(freeze,envelope)
    return result


SEALED_NAMES=('fit','assessment','memory_admissions','placement_requests','placement_acks',
    'runtime_sources.json','source_reuse_admission.json','fit_execution.json','verification_summary.json')


def final_checks(envelope,freeze):
    result=verify_saved(envelope,freeze,writing=True)
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=evidence_seal(
        [c.OUT/n for n in SEALED_NAMES if (c.OUT/n).exists()],envelope)))
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact new policy-head study freeze')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Fresh owning clock/freeze identity')
    envelope=Envelope(claim);envelope.placement_mode='isolated'
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    coverage=None;failure=None;status='incomplete'
    try:
        placement.configure_execution(freeze,'isolated')
        c.save(c.OUT/'runtime_sources.json',dict(sources=verify_runtime_sources()))
        envelope.enable_evidence_accounting();envelope.begin('fit')
        verify_runtime(freeze,envelope);load_inputs(freeze,envelope)
        envelope.admit();switch_placement(envelope,freeze,'isolated')
        c.save(c.OUT/'source_reuse_admission.json',dict(time=c.now(),
            reused_corpus=freeze['reused_corpus'],source_reuse_proof=freeze['source_reuse_proof'],
            reused_source_games=256,new_source_games=0,automatic_retries=0))
        run_fit(envelope,freeze);verify_fit(envelope,freeze);envelope.finish()
        from .assessment import control as assessment
        candidate=c.read(c.OUT/'fit/fit_result.json')['checkpoint_manifest']
        assessment.run(envelope,freeze,candidate)
        envelope.begin('final');coverage=final_checks(envelope,freeze);envelope.finish();status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,coverage=coverage,freeze_sha256=expected_sha,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        reused_source_games=256,new_source_games=0,automatic_retries=0,
        requires_actual_reader_and_guardian_exit=True))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run']);parser.add_argument('--freeze-sha256',required=True)
    return run(parser.parse_args().freeze_sha256)


if __name__=='__main__':raise SystemExit(main())


