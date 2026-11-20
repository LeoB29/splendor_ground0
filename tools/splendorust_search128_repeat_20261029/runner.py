"""Two unchanged player trees;120 saved decisions under one bounded clock."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from importlib.metadata import version
import psutil
from . import common as c,placement,transport
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules

def validate_freeze_contract(freeze):
    from .preparation import contract
    expected=contract()
    c.require(all(freeze.get(k)==v for k,v in expected.items()),'Exact repeat clock/resource/workload contract')
    c.require(freeze['schema']=='splendorust_search128_repeat_v1' and
        freeze['same_workers_retained_for_repeats'] is True and freeze['numerical_player_change'] is False,
        'Repeat identity and unchanged numerical players')
    return expected

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


def load_inputs(freeze,envelope):
    from . import design,qualification
    for key in ('panel','request_matrix','qualification_plan'):c.verify([freeze[key]])
    panel=c.read(c.ROOT/freeze['panel']['path']);matrix=c.read(c.ROOT/freeze['request_matrix']['path'])
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(matrix==design.build_matrix(panel) and plan==qualification.build_plan(),'Frozen repeat inventory')
    c.require(len(panel['roots'])==20 and len(matrix)==120,'Fixed20 roots/120 requests')
    # Source observers/results were authenticated during preparation; every source
    # descriptor is also in the begin/end full finite input pass.
    pins={p['path']:p for p in freeze['inputs']}
    c.require(all(pins.get(p['path'])==p for p in panel['inputs']),'Saved panel input authority')
    return panel,matrix,plan


def qualify(envelope,freeze,plan,workers):
    from . import adapters,qualification
    receipts=[]
    for route in ('research','frozen'):
        child=transport.make_worker('qualification_repeat_'+route,envelope,freeze,route=route)
        workers[route]=child
        receipt=adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,
                                          collect_fn=transport.collect_profiles)
        clock=c.save(c.OUT/'entrypoint_checks'/f'{route}.json',receipt)
        proof=qualification.verify_entrypoint(child.output_dir,receipt,plan,freeze,boundary=envelope.guard)
        receipts.append(dict(route=route,worker_folder=child.output_dir.relative_to(c.ROOT).as_posix(),
                             clock=clock,verification=proof))
    summary=dict(status='passed',scientific_decisions=119,parent_requests=2,graph_cases=22,
        same_workers_retained_for_repeats=True,receipts=receipts)
    c.save(c.OUT/'qualification_summary.json',summary)
    print(c.canonical(dict(stage='qualification',scientific_decisions=119,graph_cases=22,status='passed')),flush=True)
    return summary


def bind_diagnostic_deadline(envelope,workers):
    c.require(envelope.stage=='diagnostic' and set(workers)=={'frozen','research'} and
              len(envelope.resident)==2,'Two qualified retained trees')
    bindings=[];stamp=time.perf_counter_ns()
    for route in ('research','frozen'):
        child=workers[route]
        c.require(not child._lock.locked() and child._counter==2 and not child._closed,
                  'Only idle qualified trees may cross the stage boundary')
        old=child.absolute_deadline;new=envelope.effective_deadline()
        c.require(new<=envelope.start+c.COLLECTION_CUTOFF_SECONDS and new<=envelope.end,
                  'No global/collection resource-clock extension')
        child.absolute_deadline=new
        bindings.append(dict(route=route,worker=child.output_dir.name,player=child.identity,
            request_counter=2,previous_absolute_deadline=old,diagnostic_absolute_deadline=new))
    value=dict(status='passed',bindings=bindings,transition_ns=stamp,stage='diagnostic',
        stage_deadline=envelope.deadline,global_deadline=envelope.end,
        ordinary_deadline_seconds=10.,internal_seconds=9.,request_clocks_reset=False)
    c.save(c.OUT/'resident_stage_transition.json',value)
    return value


def collect(envelope,freeze,panel,matrix,workers,rows):
    from . import analysis
    roots={r['root_id']:r for r in panel['roots']};cache={}
    for spec in matrix:
        root=roots[spec['root_id']]
        row=transport.ordinary(workers[spec['route']],spec,root['observer'],envelope,freeze)
        row['quota']=spec['quota']
        row.update(before=root['before'],after=None,transition_proof=None)
        if row['completed']:
            try:row['after'],row['transition_proof']=validate_transition(row,root)
            except BaseException:row['transition_error']=traceback.format_exc()
        c.save(c.OUT/'records'/f"{spec['sequence']:04d}.json",row);rows.append(row)
        if not row['completed']:
            c.save(c.OUT/'collection_stop.json',dict(sequence=spec['sequence'],spec=spec,clock=row['clock'],
                no_retry=True,deadline_failure=transport.deadline_failure(row['clock'],spec['route'])))
        envelope.guard();transport.verify_saved_record(row,freeze)
        if not row['completed']:return
        c.require('transition_error' not in row,'Saved transition failed; original receipt retained')
        analysis.validate_repeat_row(row,spec,cache,panel,boundary=envelope.guard)
        print(c.canonical(dict(stage='diagnostic',completed_requests=len(rows),total=120,
            root_id=spec['root_id'],repeat=spec['repeat'],quota=spec['quota'])),flush=True)


def validate_transition(row,root):
    """One corrected saved-root step checked by the independent dictionary ledger."""
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from tools.external_strength_screen_20261018 import rules, audit as ledger
    before=root['before'];state=decode_training_snapshot(before)
    ledger.check_observer(before,row['observer'])
    legal=[rules.serialize_action(a) for a in rules.legal_actions(state)]
    c.require(row['action'] in legal,'Saved-root corrected legal action')
    after=rules.serialize_snapshot(rules.CorrectedEnv(seed=0).step(state,rules.deserialize_action(row['action'])))
    ledger.check_step(before,row['action'],after)
    if row['quota']==128:c.require(after==root['after'],'Historical128 saved successor equality')
    return after,dict(status='passed',before_sha256=c.digest(before),observer_sha256=c.digest(row['observer']),
        action_sha256=c.digest(row['action']),after_sha256=c.digest(after),legal_count=len(legal),
        corrected_engine_and_independent_ledger=True,model_calls=0,new_games=0)


def verify_stage_transition(envelope,rows):
    path=c.OUT/'resident_stage_transition.json'
    if not path.exists():
        c.require(not rows,'No ordinary work before resource stage binding');return None
    value=c.read(path);stage=c.read(c.OUT/'stages/diagnostic_start.json')
    c.require(value['stage_deadline']==stage['monotonic_deadline'] and value['global_deadline']==envelope.end and
        value['ordinary_deadline_seconds']==10. and value['internal_seconds']==9. and
        value['request_clocks_reset'] is False,'Saved resource-only stage transition')
    c.require([b['route'] for b in value['bindings']]==['research','frozen'],'Both retained trees bound')
    for b in value['bindings']:
        c.require(b['worker']=='qualification_repeat_'+b['route'] and b['request_counter']==2 and
            b['player']==c.read(c.OUT/'workers'/b['worker']/'initialization.json')['identity'] and
            b['diagnostic_absolute_deadline']<=stage['monotonic_deadline'] and
            b['diagnostic_absolute_deadline']<=envelope.start+c.COLLECTION_CUTOFF_SECONDS,
            'Saved retained tree identity/deadline')
    for row in rows:c.require(row['parent_profile']['request_start_ns']>=value['transition_ns'],
                             'Ordinary request before stage binding')
    return value


def final_checks(envelope,freeze,panel,matrix,plan,qualified):
    from . import analysis,qualification
    paths=sorted((c.OUT/'records').glob('*.json'))
    c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(len(paths))],'Exact prefix file inventory')
    rows=[]
    roots={r['root_id']:r for r in panel['roots']}
    for path in paths:
        envelope.guard();row=c.read(path);transport.verify_saved_record(row,freeze)
        root=roots[row['spec']['root_id']]
        c.require(row['before']==root['before'],'Saved-root snapshot binding')
        if row['completed']:
            after,proof=validate_transition(row,root)
            c.require(row['after']==after and row['transition_proof']==proof and 'transition_error' not in row,
                      'Saved transition proof exact readback')
        else:c.require(row['after'] is None and row['transition_proof'] is None,'No successor for rejected move')
        rows.append(row)
    complete=len(rows)==120 and all(r['completed'] for r in rows)
    exact=analysis.verify_records(rows,panel,matrix,complete=complete,boundary=envelope.guard)
    c.require(qualified==c.read(c.OUT/'qualification_summary.json'),'Both actual suites before ordinary repeats')
    stop=c.read(c.OUT/'collection_stop.json') if (c.OUT/'collection_stop.json').exists() else None
    if rows and not rows[-1]['completed']:
        last=rows[-1]
        c.require(stop and stop['sequence']==len(rows)-1 and stop['clock']==last['clock'] and
            stop['spec']==last['spec'] and stop['no_retry'] is True and
            stop['deadline_failure']==transport.deadline_failure(last['clock'],last['route']),
            'Original terminal failure retained exactly')
    else:c.require(stop is None,'Invented/unrepresented failed request')
    folders=sorted((c.OUT/'workers').iterdir())
    c.require({f.name for f in folders}==set(transport.lifecycle_inventory()),'Exactly2 owned trees; no restart')
    represented={f.name:[] for f in folders};proofs=[]
    for row in rows:represented[Path(row['worker_folder']).name].append(row['clock']['request_id'])
    for folder in folders:
        ids=represented[folder.name]
        c.require(len(ids)==len(set(ids)) and set(ids)=={p.stem for p in (folder/'inputs').glob('*.json')},
                  'Every attempted ordinary request represented exactly once')
        proof=transport.verify_closed_worker(folder,freeze,envelope)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{folder.name}.json'),'Closure evidence unchanged')
        route=c.read(folder/'route_identity.json')['route'];receipt=c.read(c.OUT/'entrypoint_checks'/f'{route}.json')
        qproof=qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=envelope.guard)
        c.require(qproof==next(x['verification'] for x in qualified['receipts'] if x['route']==route),
                  'Actual entrypoint forensic proof unchanged')
        proofs.append(proof)
    parents=sum(p['raw']['requests'] for p in proofs)
    c.require(parents==len(rows)+4,'Startup plus attempted ordinary raw request inventory')
    verify_stage_transition(envelope,rows)
    coverage=dict(complete=complete,planned_requests=120,attempted_requests=len(rows),
        completed_requests=sum(r['completed'] for r in rows),failed_requests=sum(not r['completed'] for r in rows),
        unattempted_requests=120-len(rows),qualification_scientific_decisions=119,
        scientific_decisions=119+sum(r['completed'] for r in rows),graph_cases=22,
        workers=2,parent_requests=parents,retries=0,new_games=0,exact_comparisons=exact)
    c.save(c.OUT/'all_closed_workers.json',dict(status='passed',workers=2,proofs=proofs))
    c.save(c.OUT/'all_placement_integrity.json',verify_exchanges(envelope,freeze))
    result=analysis.analyze(rows,panel,matrix,complete=complete,boundary=envelope.guard)
    c.save(c.OUT/'analysis.json',result);c.save(c.OUT/'coverage.json',coverage)
    verify_runtime(freeze,envelope)
    summary=dict(status='passed',coverage=coverage,source_inputs=len(freeze['inputs']),
        parent_requests=parents,graph_cases=22,workers=2,peak_combined_rss_bytes=envelope.peak_combined_rss,
        all_raw_placement_logs_graph_science_sources_verified=True,no_games_or_admission=True)
    c.save(c.OUT/'verification_summary.json',summary)
    seal=[]
    for sub in ('workers','records','entrypoint_checks','closed_workers','closed_worker_costs',
                'placement_requests','placement_acks','stages','memory_admissions'):
        for path in sorted((c.OUT/sub).rglob('*')):
            if path.is_file():envelope.guard();seal.append(c.artifact(path))
    for name in ('qualification_summary.json','collection_stop.json','resident_stage_transition.json',
        'analysis.json','coverage.json','all_closed_workers.json','all_placement_integrity.json','verification_summary.json',
        'failure.json','closure_failures.json'):
        if(c.OUT/name).exists():seal.append(c.artifact(c.OUT/name))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:envelope.guard();c.verify([item])
    return coverage


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact execution freeze')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Owning watchdog freeze')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json')
    envelope.placement_mode='isolated';envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        freeze_sha256=expected_sha,phase=c.PHASE))
    workers={};rows=[];qualified=None;ready=False;coverage=None;failure=None;status='incomplete'
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);envelope.admit();switch_placement(envelope,freeze,'isolated')
        panel,matrix,plan=load_inputs(freeze,envelope);ready=True
        qualified=qualify(envelope,freeze,plan,workers);envelope.finish()
        envelope.begin('diagnostic');bind_diagnostic_deadline(envelope,workers)
        collect(envelope,freeze,panel,matrix,workers,rows);envelope.finish()
    except BaseException:
        failure=traceback.format_exc();c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    finally:
        errors=[]
        try:envelope.begin('final')
        except BaseException:errors.append(dict(scope='final stage admission',error=traceback.format_exc()))
        for route,child in workers.items():
            try:transport.close_and_audit(child,envelope,freeze)
            except BaseException:errors.append(dict(route=route,error=traceback.format_exc()))
        # Initialization failures may leave a constructor-cleaned folder without a
        # returned child. Strictly refresh its accounting; never invent a pass.
        for folder in envelope.worker_accounting:
            if folder not in [child.output_dir for child in workers.values()]:
                try:envelope.close_worker(folder)
                except BaseException:errors.append(dict(scope=str(folder),error=traceback.format_exc()))
        if errors:
            c.save(c.OUT/'closure_failures.json',dict(primary_failure=failure,errors=errors))
            failure=(failure or '')+'\nClosure/final admission: '+c.canonical(errors)
    if ready and qualified and not envelope.resident:
        try:
            coverage=final_checks(envelope,freeze,panel,matrix,plan,qualified);envelope.finish()
            if failure is None:status='completed_verified' if coverage['complete'] else 'incomplete_verified_prefix'
        except BaseException:
            error=traceback.format_exc();c.save(c.OUT/'finalization_failure.json',dict(time=c.now(),error=error))
            failure=(failure or '')+error
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,coverage=coverage,
        freeze_sha256=expected_sha,phase=c.PHASE,elapsed_seconds=time.monotonic()-envelope.start,
        wall_elapsed_seconds=time.time()-envelope.wall_start,maximum_seconds=c.JOB_SECONDS,
        automatic_retries=0,new_games=0,training=False,no_admission_or_strength_claim=True,
        requires_clean_actual_os_exits=['runner','watchdog','independent_reader','guardian'],
        actual_os_exit_requirement_scope='successful_study_acceptance'))
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run'])
    parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    return run(args.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
