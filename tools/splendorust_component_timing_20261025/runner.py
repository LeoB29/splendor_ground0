"""Bounded component timing, with overhead admission before the saved diagnostic."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import psutil
from . import common as c, qualification as plan_rules, analyze
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit, qualification as frozen_q
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules
MODES=('baseline','profiled')
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

    def enable_evidence_accounting(self):
        self.evidence = dict(baseline=bytes_used(), parent_bytes=0, workers={})
        self.log_baseline = {name: (c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                             for name in ('runner_stdout.log', 'runner_stderr.log')}
        c.WRITE_OBSERVER = self.note_write

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
        return total + 65536

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
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        cap = (512 if self.stage == 'final' else 448)*c.MiB
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
            if stable is not None and (self.admitted or current-stable >= 60):
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
                stable_seconds=60 if self.admissions == 1 else 0))

    def finish(self):
        self.guard()
        if self.evidence is not None:
            c.require(bytes_used() <= self.accounted_bytes(), 'Unaccounted study evidence')
        c.save(c.OUT/'stages'/f'{self.stage}_end.json', dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time()))

def verify_runtime(freeze, envelope=None):
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python runtime changed')
    c.require(processor_topology() == freeze['topology'], 'Processor topology changed')
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

def nested_artifacts(value):
    """Only explicit artifact triples, never an unbounded output-tree expansion."""
    if isinstance(value,dict):
        if {'path','sha256','bytes'}<=set(value):
            yield {key:value[key] for key in ('path','sha256','bytes')}
        for item in value.values():
            yield from nested_artifacts(item)
    elif isinstance(value,list):
        for item in value:
            yield from nested_artifacts(item)


def predecessor_acceptance():
    evidence=[]
    for folder, qualified in ((c.QUALIFIED, True), (c.PRIOR, False)):
        names=('freeze.json','completion.json','execution.json','qualification_outcome.json',
               'verification_summary.json','all_worker_log_integrity.json')
        saved={name:c.read(folder/name) for name in names}
        finish,outer=saved['completion.json'],saved['execution.json']
        sha=c.file_hash(folder/'freeze.json')
        c.require(finish['status']=='completed_verified' and finish['failure'] is None and
            outer['status']=='completed' and outer['exit_code']==0 and not outer['cleanup_errors'],
            'Preceding verified evidence/cleanup changed')
        c.require(finish['freeze_sha256']==outer['freeze_sha256']==sha and
            outer['charged_job_seconds']<saved['freeze.json']['job_seconds'],'Preceding freeze/clock changed')
        c.require(all(saved[name]['status']=='passed' for name in ('verification_summary.json','all_worker_log_integrity.json')),
                  'Preceding verification missing')
        outcome=saved['qualification_outcome.json']
        if qualified:
            c.require(outcome['optimized32_qualified'] is True and outcome['optimized64_qualified'] is True,
                      'Raw-reader optimized sampler must remain qualified')
        else:
            c.require(outcome['buffered_reader_qualified'] is False and outcome['saved_panel_qualified'] is False,
                      'Buffered-reader negative result must remain preserved')
        evidence.extend(c.artifact(folder/name) for name in names)
    return dict(status='passed',evidence=evidence,baseline='verified_raw_reader_with_optimized_sampler')


def prepare():
    c.require(not (c.OUT/'freeze.json').exists(),'No replacement freeze')
    accepted=predecessor_acceptance()
    for name in ('prelaunch_tests.json','prelaunch_review.json'):
        receipt=c.read(c.OUT/name)
        c.require(receipt['status']=='passed','Prelaunch checks/review required')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free>=7*c.GiB,'Launch requires7GiB free disk')
    sources=verify_runtime_sources()
    prior=c.read(c.PRIOR/'freeze.json')
    plan=plan_rules.build_plan()
    c.save(c.OUT/'qualification_plan.json',plan)
    c.save(c.OUT/'baseline_acceptance.json',accepted)
    inherited={item['path']:item for item in prior['inputs']}
    items={path:item for path,item in inherited.items() if not path.startswith('outputs/')}
    def bind(item):
        if item['path'] in inherited:
            c.require(inherited[item['path']]==item,'Changed preserved input')
        if item['path'] in items:
            c.require(items[item['path']]==item,'Conflicting source inventory')
        items[item['path']]=item
    for value in (accepted,plan,frozen_q.build_plan(32),frozen_q.build_plan(64),c.read(frozen_q.PARITY_RECEIPT)):
        for item in nested_artifacts(value): bind(item)
    bind(c.artifact(frozen_q.PARITY_RECEIPT))
    for quota in (16,32,64):
        bind(c.artifact(c.profile_path(quota)))
        for item in c.profile_for(quota)['artifacts']: bind(item)
    for item in prior['native_artifacts']: bind(item)
    paths=set((c.ROOT/'tools/splendorust_component_timing_20261025').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_component_timing*.py'))
    paths.update(path for path in c.OUT.rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    paths.add(c.PROTOCOL)
    for path in paths: bind(c.artifact(path))
    freeze=dict(schema='splendorust_component_timing_execution_v1',time=c.now(),
        inputs=sorted(items.values(),key=lambda item:item['path']),baseline_acceptance=accepted,
        python=sys.version,python_executable_sha256=c.file_hash(sys.executable),
        torch_runtime=prior['torch_runtime'],external_runtime=prior['external_runtime'],
        sampler_api_modules=prior['sampler_api_modules'],compression_runtime=prior['compression_runtime'],
        topology=processor_topology(),native_artifacts=prior['native_artifacts'],source_guard=sources,
        job_seconds=c.JOB_SECONDS,stage_seconds=c.STAGES,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        automatic_retries=0,seconds_per_move=1.,internal_seconds=.9,margin_seconds=.7,
        margin_is_diagnostic_only=True,overhead_requests=48,diagnostic_requests_maximum=192,
        modes=list(MODES),quotas=[64,32],max_rss_bytes=2*c.GiB,max_evidence_bytes=512*c.MiB,
        final_reserve_bytes=64*c.MiB,fresh_games=0,strength_games=0,larger_data_deferred=True)
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json',freeze)
    print(c.canonical(dict(prepared=True,model_calls=0,inputs=len(items),freeze_sha256=c.file_hash(c.OUT/'freeze.json'))),flush=True)


def make_worker(name,envelope,freeze,*,mode,quota):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    c.require(mode in MODES and type(quota) is int and quota in (64,32) and 'torch' not in sys.modules,
              'Fixed mode/quota and model-free parent required')
    envelope.admit()
    folder=c.OUT/'workers'/name
    envelope.register_worker(folder)
    cls=c.BaselineStudyWorker if mode=='baseline' else c.ProfiledStudyWorker
    module='tools.splendorust_search_scale_20261024.worker' if mode=='baseline' else 'tools.splendorust_component_timing_20261025.worker'
    child=cls(module,folder,native_artifacts=freeze['native_artifacts'],
        aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']],initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(),max_message_bytes=4*c.MiB,max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB,max_rss_bytes=2*c.GiB,resource_guard=envelope.guard,
        env={'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'STUDY_WORKER_DIR':str(folder),
             'STUDY_SEARCH_SIMULATIONS':str(quota),'STUDY_COMPONENT_PROFILE':'1' if mode=='profiled' else '0',
             'CUDA_VISIBLE_DEVICES':'','RAYON_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'1'})
    try:
        envelope.own_worker(child)
        child.study_after_close=lambda:envelope.close_worker(folder,child)
        telemetry=child._sample()
        metadata=deepcopy(child.initialization['result'])
        if mode=='profiled':
            c.require(metadata.pop('component_timing_mode')=='profiled','Child timing mode changed')
        validate_ready_metadata(metadata,expected_sources=SOURCE_HASHES,worker_pid=child.process.pid,
            process_tree=telemetry['process_tree'],expected_quota=quota)
        child.runtime_mode,child.challenger_quota=mode,quota
        c.require(not child._failure,'Worker readiness supervision failed')
        envelope.guard()
        return child
    except BaseException:
        child.close()
        raise


def save_profile(child,receipt):
    if child.runtime_mode=='baseline':
        return None,None
    profile=deepcopy(child.last_parent_profile)
    c.require(profile['request_id']==receipt['request_id'],'Parent profile request identity')
    path=child.output_dir/'parent_profiles'/f"{receipt['request_id']}.json"
    c.save(path,profile)
    return profile,path.relative_to(c.ROOT).as_posix()


def science_result(result,mode):
    c.require(type(result) is dict,'Scientific response object')
    copied=deepcopy(result)
    if mode=='profiled':
        c.require(type(copied.pop('timing_profile')) is dict,'Child component profile absent')
    else:
        c.require('timing_profile' not in copied,'Baseline must remain uninstrumented')
    return copied


def actual_entrypoint(child,envelope,label):
    receipt=child.request({'operation':'entrypoint_checks'},envelope.timeout(300))
    c.save(c.OUT/'entrypoint_checks'/f'{label}.json',receipt)
    save_profile(child,receipt)
    envelope.guard()
    c.require(receipt['status']=='ok' and receipt['result']['status']=='entrypoint_counterexamples_passed' and
        receipt['result']['challenger_quota']==child.challenger_quota and
        receipt['result']['counts']['complete_decisions']==52,'Actual corrected entrypoint failed; no retry')


def decision(child,state,envelope):
    envelope.admit()
    observer,before=rules.observation_payload(state),rules.serialize_snapshot(state)
    ledger.check_observer(before,observer)
    legal=rules.legal_actions(state)
    actions=[serialize_action(action) for action in legal]
    request_id=f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json',dict(player='search',observer=observer,before=before))
    def validate(raw):
        result=science_result(raw,child.runtime_mode)
        c.require(result.get('action') in actions,'Illegal scientific action')
        return audit.validate_result(result,observer,'search',expected_quota=child.challenger_quota)
    c.require(envelope.remaining()>=1.,'Insufficient full move allowance')
    allowance=envelope.timeout(1.)
    c.require(allowance==1.,'No stage-clamped move dispatch')
    receipt=child.request({'player':'search','observer':observer},allowance,validate=validate)
    profile,profile_file=save_profile(child,receipt)
    row=dict(arm='search',mode=child.runtime_mode,quota=child.challenger_quota,seat=state.current_player,
        before=before,observer=observer,after=None,action=None,trace=None,completed=receipt['status']=='ok',clock=receipt,
        parent_profile=profile,parent_profile_file=profile_file,
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix())
    if row['completed']:
        result=science_result(receipt['result'],child.runtime_mode)
        c.require(result['observer_sha256']==c.digest(observer) and result['legal_actions_sha256']==legal_digest(legal) and
            result['legal_count']==len(legal),'Scientific observer/legal binding changed')
        row.update(action=result['action'],trace=result['diagnostics'])
        row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,deserialize_action(row['action'])))
    return row


def run_panel(envelope,freeze,plan,stage):
    schedule=plan[stage+'_schedule']
    expected=48 if stage=='overhead' else 192
    c.require(len(schedule)==expected and [row['index'] for row in schedule]==list(range(expected)),
              'Fixed complete timing schedule required')
    rows=[]
    for quota in (64,32):
        with ExitStack() as stack:
            workers={mode:stack.enter_context(make_worker(f'{stage}_{quota}_{mode}',envelope,freeze,mode=mode,quota=quota)) for mode in MODES}
            if stage=='overhead':
                for mode,child in workers.items(): actual_entrypoint(child,envelope,f'{quota}_{mode}')
            for spec in (item for item in schedule if item['quota']==quota):
                row={**decision(workers[spec['mode']],decode_training_snapshot(plan_rules.snapshot_for_request(plan,spec)),envelope),**spec}
                c.save(c.OUT/stage/f"{spec['index']:04d}.json",row)
                envelope.guard()
                c.require(row['completed'],'Hard move failure preserved; no retry')
                audit.check_clock(row['clock'])
                if len(rows)%24==0:
                    print(c.canonical(dict(stage=stage,requests=len(rows)+1,total=expected)),flush=True)
                rows.append(row)
    summary=plan_rules.validate_panel(rows,plan,stage=stage,boundary=envelope.guard)
    c.save(c.OUT/f'{stage}_summary.json',summary)
    return summary


def verify_saved_row(row):
    path=c.ROOT/row['worker_clock_file']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()),'Unowned trusted receipt')
    c.require(c.read(path)==row['clock'] and path.name==row['clock']['request_id']+'.json','Raw trusted receipt changed')
    c.require(c.read(path.parent/'inputs'/path.name)==dict(player='search',observer=row['observer'],before=row['before']),
              'Saved input changed')
    if row['mode']=='profiled':
        profile_path=c.ROOT/row['parent_profile_file']
        c.require(profile_path.resolve().is_relative_to((path.parent/'parent_profiles').resolve()) and
            profile_path.name==path.name and c.read(profile_path)==row['parent_profile'],'Parent timing sidecar changed')
    else:
        c.require(row['parent_profile'] is None and row['parent_profile_file'] is None,'Baseline profile unexpectedly present')


def final_checks(envelope,freeze,plan,summaries):
    collected=[]
    for stage,summary in summaries.items():
        paths=sorted((c.OUT/stage).glob('*.json'))
        expected=48 if stage=='overhead' else 192
        c.require([path.name for path in paths]==[f'{i:04d}.json' for i in range(expected)],'Timing inventory differs')
        rows=[c.read(path) for path in paths]
        for row in rows:
            envelope.guard(); verify_saved_row(row)
        c.require(plan_rules.validate_panel(rows,plan,stage=stage,boundary=envelope.guard)==summary,
                  'Independent panel reread differs')
        collected.extend(rows)
    logs=[]
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard(); logs.append(verify_worker_logs(folder,boundary=envelope.guard,deadline=envelope.effective_deadline()))
    c.require(len(logs)==4*len(summaries),'Worker inventory differs')
    c.save(c.OUT/'all_worker_log_integrity.json',dict(status='passed',workers=len(logs),receipts=logs))
    report=analyze.summarize(collected,boundary=envelope.guard)
    c.save(c.OUT/'component_analysis.json',report)
    seal=[]
    for directory in ('overhead','diagnostic','workers'):
        for path in sorted((c.OUT/directory).rglob('*')):
            if path.is_file():
                envelope.guard(); seal.append(c.artifact(path))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal:
        envelope.guard(); c.verify([item])
    verify_runtime(freeze,envelope)
    c.require(predecessor_acceptance()==freeze['baseline_acceptance'],'Preserved earlier evidence changed')
    c.save(c.OUT/'verification_summary.json',dict(status='passed',requests=len(collected),workers=len(logs),
        maximum_sampled_combined_worker_rss_bytes=envelope.peak_combined_rss,science_and_transitions_verified=True))


def run_stages(envelope,freeze,plan,*,overhead_started=False):
    if overhead_started:
        c.require(envelope.stage=='overhead','Startup overhead clock missing')
    else:
        envelope.begin('overhead')
    summaries={'overhead':run_panel(envelope,freeze,plan,'overhead')}
    envelope.finish()
    c.require(type(summaries['overhead']['qualified']) is bool,'Overhead admission must be Boolean')
    if summaries['overhead']['qualified']:
        envelope.begin('diagnostic')
        summaries['diagnostic']=run_panel(envelope,freeze,plan,'diagnostic')
        envelope.finish()
    envelope.begin('final')
    final_checks(envelope,freeze,plan,summaries)
    envelope.finish()
    c.save(c.OUT/'diagnostic_outcome.json',dict(status='completed_verified',
        instrumentation_qualified=summaries['overhead']['qualified'],diagnostic_completed='diagnostic' in summaries,
        summaries=summaries,fresh_games=0,strength_games=0,larger_data_deferred=True))
    return summaries


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Execution freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze identity differs')
    freeze,envelope=c.read(c.OUT/'freeze.json'),Envelope(claim)
    envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status,failure='incomplete',None
    try:
        envelope.begin('overhead')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope)
        envelope.admit()
        run_stages(envelope,freeze,c.read(c.OUT/'qualification_plan.json'),overhead_started=True)
        status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,automatic_retries=0,fresh_games=0,strength_games=0,larger_data_deferred=True,
        note='Acceptance also requires owning watchdog cleanup. Timing measurements do not change any clock.'))
    print(c.canonical(dict(status=status,stage=envelope.stage,failure=failure)),flush=True)
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['prepare','run'])
    parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    return prepare() if args.operation=='prepare' else run(args.freeze_sha256)


if __name__=='__main__':
    raise SystemExit(main())
