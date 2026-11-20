"""One bounded attempt, with qualification before any fresh comparison games."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil
from . import common as c
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology

PLAYER_FAILURES={'timeout','worker_exit','worker_error','malformed','wrong_request','illegal','unsolicited','write_error'}


def bytes_used():
    return sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())


class Envelope:
    def __init__(self, claim):
        self.start,self.wall_start=claim['monotonic_origin'],claim['unix_origin']
        self.end=self.start+c.JOB_SECONDS
        self.stage=None; self.deadline=self.end; self.wall_deadline=self.wall_start+c.JOB_SECONDS
        self.waited=0.; self.admitted=False; self.admissions=0
        self.evidence=None

    def enable_evidence_accounting(self):
        # Only the current worker tree remains mutable. Closed trees are counted
        # once, avoiding a quadratic full-history scan on every audited move.
        self.evidence=dict(baseline=bytes_used(),parent_bytes=0,workers={})
        self.log_baseline={name:(c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                           for name in ('runner_stdout.log','runner_stderr.log')}
        c.WRITE_OBSERVER=self.note_write

    def note_write(self,path,size):
        if not path.resolve().is_relative_to((c.OUT/'workers').resolve()):
            self.evidence['parent_bytes']+=size

    def register_worker(self,folder):
        self.evidence['workers'][folder]=None

    def close_worker(self,folder):
        self.evidence['workers'][folder]=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())

    def accounted_bytes(self):
        if self.evidence is None: return bytes_used()
        total=self.evidence['baseline']+self.evidence['parent_bytes']
        for folder,closed_size in self.evidence['workers'].items():
            total+=closed_size if closed_size is not None else sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        for name,start_size in self.log_baseline.items():
            path=c.OUT/name
            total+=max(0,(path.stat().st_size if path.exists() else 0)-start_size)
        # Reserve the small owning-watchdog control/final records.
        return total+65536

    def begin(self, stage):
        mono,wall=time.monotonic(),time.time()
        cap=c.STAGES[stage]
        self.stage=stage
        cutoff=19800 if stage!='final' else c.JOB_SECONDS
        self.deadline=min(mono+cap,self.start+cutoff,self.end)
        self.wall_deadline=min(wall+cap,self.wall_start+cutoff,self.wall_start+c.JOB_SECONDS)
        self.guard()
        c.save(c.OUT/'stages'/f'{stage}_start.json',dict(time=c.now(),monotonic=mono,unix_time=wall,
            monotonic_deadline=self.deadline,unix_deadline=self.wall_deadline,cap_seconds=cap))

    def remaining(self):
        return min(self.deadline-time.monotonic(),self.wall_deadline-time.time(),
                   self.end-time.monotonic(),self.wall_start+c.JOB_SECONDS-time.time())

    def effective_deadline(self):
        return time.monotonic()+self.remaining()

    def guard(self):
        c.require(self.remaining()>0,'Stage/global real-clock deadline')
        c.require(shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Free disk reserve')
        cap=(512 if self.stage=='final' else 448)*c.MiB
        c.require(self.accounted_bytes()<cap,'Study evidence ceiling')

    def timeout(self, maximum):
        self.guard()
        return min(maximum,self.remaining())

    def admit(self):
        self.guard(); begun=time.monotonic(); available=psutil.virtual_memory().available
        stable=begun if available>=2*c.GiB else None; minimum=available
        fast=self.admitted and stable is not None
        while not fast:
            self.guard(); current=time.monotonic(); available=psutil.virtual_memory().available
            minimum=min(minimum,available)
            if available<2*c.GiB: stable=None
            elif stable is None: stable=current
            c.require(self.waited+current-begun<600,'Cumulative RAM wait exceeded ten minutes')
            if stable is not None and (self.admitted or current-stable>=60): break
            time.sleep(.5)
        elapsed=time.monotonic()-begun; self.waited+=elapsed; self.admitted=True; self.admissions+=1
        c.require(self.waited<600,'Cumulative RAM admission bound')
        # Log waits and initial stability, not a separate disk file for every move.
        if not fast:
            c.save(c.OUT/'memory_admissions'/f'{self.admissions:06d}.json',dict(time=c.now(),wait_seconds=elapsed,
                cumulative_wait_seconds=self.waited,minimum_available_bytes=minimum,stable_seconds=60 if self.admissions==1 else 0))

    def finish(self):
        self.guard()
        if self.evidence is not None:
            actual=bytes_used()
            c.require(actual<=self.accounted_bytes(),'Unaccounted study evidence')
        c.save(c.OUT/'stages'/f'{self.stage}_end.json',dict(time=c.now(),monotonic=time.monotonic(),unix_time=time.time()))


def seed_inventory():
    target={int(r['seed_decimal']) for r in c.make_schedule()}
    found=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json','-g','*schedule*.json',
        '-g','*seed_registry*.json'],cwd=c.ROOT,capture_output=True,text=True,check=True)
    checked=[]; skipped=[]; excluded=[]; collisions=[]; fields=0
    def walk(value,path):
        nonlocal fields
        if type(value) is int or (isinstance(value,str) and value.isdecimal()):
            fields+=1
            if int(value) in target: collisions.append(path)
        elif isinstance(value,list):
            for v in value: walk(v,path)
        elif isinstance(value,dict):
            for v in value.values(): walk(v,path)
    for name in sorted(found.stdout.splitlines()):
        path=(c.ROOT/name).resolve()
        if path.is_relative_to(c.OUT): excluded.append(c.artifact(path)); continue
        if path.stat().st_size>8*c.MiB: skipped.append(c.artifact(path)); continue
        walk(c.read(path),path.relative_to(c.ROOT).as_posix()); checked.append(c.artifact(path))
    c.require(not collisions,'Fresh schedule collides with registered prior evidence: '+str(collisions))
    return dict(status='passed',checked=checked,skipped_oversized=skipped,excluded_current=excluded,
                collisions=collisions,numeric_fields=fields,scope='registered JSON names up to8MiB; not exhaustive logs')


def preserve_prior_attempts():
    path=c.OUT/'prior_attempts_preservation.json'
    if not path.exists():
        attempts=[]
        for directory,sha in c.PRIOR_ATTEMPTS:
            c.require(c.file_hash(directory/'freeze.json')==sha,'Prior freeze changed')
            c.verify(c.read(directory/'freeze.json')['inputs'])
            evidence=[c.artifact(p) for p in sorted(directory.rglob('*')) if p.is_file()]
            attempts.append(dict(freeze=c.artifact(directory/'freeze.json'),evidence=evidence))
        c.save(path,dict(status='preserved',attempts=attempts))
    preserved=c.read(path)
    for (directory,sha),attempt in zip(c.PRIOR_ATTEMPTS,preserved['attempts'],strict=True):
        c.require(c.file_hash(directory/'freeze.json')==sha,'Prior freeze changed')
        c.verify([attempt['freeze']]+attempt['evidence'])
        c.verify(c.read(directory/'freeze.json')['inputs'])
        expected={x['path'] for x in attempt['evidence']}
        actual={p.relative_to(c.ROOT).as_posix() for p in directory.rglob('*') if p.is_file()}
        c.require(actual==expected,'Prior evidence inventory changed')
    c.require(c.read(c.PRIOR/'completion.json')['status']=='completed_verified','Prior comparison not verified')
    return preserved


def verify_runtime(freeze, envelope=None):
    for item in freeze['inputs']:
        if envelope: envelope.guard()
        c.verify([item])
    c.require(sys.version==freeze['python'] and c.file_hash(sys.executable)==freeze['python_executable_sha256'], 'Python runtime changed')
    c.require(processor_topology()==freeze['topology'],'Processor topology changed')
    for item in freeze['torch_runtime']:
        p=Path(item['absolute_path'])
        c.require(p.stat().st_size==item['bytes'] and c.file_hash(p)==item['sha256'],'Pinned Torch runtime changed')
    from tools.matched_strength_20261020.statistics import runtime_contract
    c.require(runtime_contract()==c.read(c.OUT/'statistics_runtime.json'),'Numerical runtime changed')
    from .compression import runtime_contract as compression_runtime
    c.require(compression_runtime()==freeze['compression_runtime'],'Compression runtime changed')



def prepare():
    from . import qualification as q
    from .compression import runtime_contract as compression_runtime
    c.require(not (c.OUT/'freeze.json').exists(),'Already frozen; no replacement')
    c.OUT.mkdir(parents=True,exist_ok=True)
    sources=verify_runtime_sources()
    preserved=preserve_prior_attempts()
    for name in ('prelaunch_tests.json','prelaunch_review.json'):
        receipt=c.read(c.OUT/name); c.require(receipt['status']=='passed','Prelaunch tests/review missing')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free>=7*c.GiB,'Launch requires7GiB free disk')
    c.require(c.file_hash(c.ROOT/'tools/splendorust_one_second_v3_20261024/compression.py')==
              c.file_hash(c.ROOT/'tools/splendorust_search_scale_20261024/compression.py'),'Compression codec must be byte-identical')
    prior_compression=c.read(c.PRIOR/'freeze.json')['compression_runtime']
    c.require(compression_runtime()==prior_compression,'Qualified compression runtime changed')
    c.save(c.OUT/'schedule.json',c.make_schedule())
    c.save(c.OUT/'seed_collision_check.json',seed_inventory())
    for quota in c.QUOTAS:
        c.save(c.OUT/f'qualification_plan_{quota}.json',q.build_plan(quota))
        c.save(c.OUT/f'model_free_qualification_{quota}.json',q.model_free_checks(quota))
    from tools.matched_strength_20261020.statistics import runtime_contract
    c.save(c.OUT/'statistics_runtime.json',runtime_contract())
    profile=c.read(c.PROFILE)
    c.require(c.file_hash(c.PROFILE)==c.PROFILE_SHA and profile['iterations']==16,'Exact PUCT16 profile')
    original_binary=c.ROOT/profile['native_binary']
    c.require(c.file_hash(c.NATIVE_COPY)==c.file_hash(original_binary),'Native copy changed bytes')
    for quota in c.QUOTAS:
        derived=c.profile_for(quota); expected=deepcopy(profile)
        expected.update(iterations=quota,profile_id=c.profile_id(quota),native_binary=c.NATIVE_COPY.relative_to(c.ROOT).as_posix())
        for item in expected['artifacts']:
            if item['path']==profile['native_binary']: item['path']=expected['native_binary']
        c.require(derived==expected,'Challenger changes more than declared quota/identity/path')
    items={}
    for prior in preserved['attempts']:
        for item in c.read(c.ROOT/prior['freeze']['path'])['inputs']+prior['evidence']:
            if item['path'] in items: c.require(items[item['path']]==item,'Conflicting prior binding')
            items[item['path']]=item
    paths=set()
    for folder in ('tools','src'):
        paths.update((c.ROOT/folder).rglob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_search_scale*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_one_second*.py'))
    paths.add(c.ROOT/'tests/test_local_player_qualification_clock.py')
    paths.update([c.PROFILE,c.PROTOCOL,c.ROOT/'docs/engineering/future_training_rules_20261021.md'])
    paths.update(p for p in c.OUT.rglob('*') if p.is_file())
    for path in paths:
        if '__pycache__' not in path.parts:
            item=c.artifact(path)
            if item['path'] in items: c.require(item==items[item['path']],'Prior dependency changed')
            items[item['path']]=item
    oldfreeze=c.read(c.PRIOR/'freeze.json')
    freeze=dict(schema='splendorust_search_scale_execution_v1',time=c.now(),
        inputs=sorted(items.values(),key=lambda a:a['path']),python=sys.version,
        python_executable_sha256=c.file_hash(sys.executable),torch_runtime=oldfreeze['torch_runtime'],
        compression_runtime=compression_runtime(),topology=processor_topology(),
        native_artifacts=[c.artifact(original_binary),c.artifact(c.NATIVE_COPY)],
        job_seconds=c.JOB_SECONDS,stage_seconds=c.STAGES,collection_cutoff_seconds=19800,
        automatic_retries=0,seconds_per_move=1,internal_seconds=.9,margin_seconds=.7,
        calibration_quotas=list(c.QUOTAS),selection='64 if qualified else32 if qualified else stop; timing/checks only',
        profiles={str(q):c.artifact(c.profile_path(q)) for q in (16,32,64)},
        shared_model='One immutable Entity model and two byte-identical native transports; fresh native agents per request',
        max_rss_bytes=2*c.GiB,max_evidence_bytes=512*c.MiB,larger_data_deferred=True,
        prior_attempts_preserved=True,source_guard=sources,
        operational_domain=c.OPERATIONAL_DOMAIN,main_domain=c.DOMAIN,
        authorization='User authorized32/64timing checks then frozen200game comparison if one qualifies; no larger-data work')
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json',freeze)
    print(c.canonical(dict(prepared=True,freeze_sha256=c.file_hash(c.OUT/'freeze.json'),inputs=len(items),model_calls=0)),flush=True)


def make_worker(name,envelope,freeze,quota):
    c.require('torch' not in sys.modules,'Referee must remain model-free')
    envelope.admit(); folder=c.OUT/'workers'/name
    envelope.register_worker(folder)
    child=c.StudyWorker('tools.splendorust_search_scale_20261024.worker',folder,native_artifacts=freeze['native_artifacts'],
        affinity=[freeze['topology']['selected_logical_processor']],initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(),max_message_bytes=4*c.MiB,max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB,max_rss_bytes=2*c.GiB,resource_guard=envelope.guard,
        env={'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'STUDY_WORKER_DIR':str(folder),
             'STUDY_SEARCH_SIMULATIONS':str(quota),'CUDA_VISIBLE_DEVICES':'','RAYON_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'1'})
    try:
        from .adapters import validate_ready_metadata
        from tools.future_training_rules_20261021 import SOURCE_HASHES
        telemetry=child._sample()
        validate_ready_metadata(child.initialization['result'],expected_sources=SOURCE_HASHES,
            worker_pid=child.process.pid,process_tree=telemetry['process_tree'],expected_quota=quota)
        child.study_after_close=lambda:envelope.close_worker(folder)
        child.challenger_quota=quota
        c.require(not child._failure,'Worker process supervision failed at initialization')
        envelope.guard()
        return child
    except BaseException:
        child.close()
        raise


def decision(child,state,arm,envelope):
    from .audit import validate_result
    envelope.admit()
    observer=rules.observation_payload(state); before=rules.serialize_snapshot(state)
    ledger.check_observer(before,observer)
    legal=rules.legal_actions(state); actions=[serialize_action(a) for a in legal]
    request_id=f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json',dict(player=arm,observer=observer,before=before))
    def response_validator(result):
        from .audit import RESULT_FIELDS
        if not isinstance(result,dict) or set(result)!=RESULT_FIELDS or not isinstance(result['diagnostics'],dict):
            raise ValueError('PLAYER_MALFORMED_RESPONSE: response envelope fields')
        if result['action'] not in actions:
            raise ValueError('PLAYER_ILLEGAL_ACTION: response action')
        return validate_result(result,observer,arm,expected_quota=child.challenger_quota if arm=='search' else 16)
    receipt=child.request({'player':arm,'observer':observer},envelope.timeout(1),validate=response_validator)
    # Infrastructure boundary wins over a move timeout caused by clamping.
    envelope.guard()
    result=receipt.get('result')
    row=dict(arm=arm,challenger_simulations=child.challenger_quota,seat=state.current_player,observer=observer,before=before,after=None,action=None,trace=None,
             completed=receipt['status']=='ok',clock=receipt)
    if receipt['status']=='ok':
        c.require(result['player']==arm and result['observer_sha256']==c.digest(observer) and
            result['legal_actions_sha256']==legal_digest(legal) and result['legal_count']==len(legal) and
            result['ruleset']==rules.RULESET_ID and result['catalogue_sha256']==rules.CATALOGUE_SHA256,'Worker binding differs')
        row.update(action=result['action'],trace=result['diagnostics'])
    return row


def timing_failure(receipt, *, entrypoint=False):
    """Only explicit player clock exhaustion disqualifies one quota and continues."""
    detail=str(receipt.get('detail','')).lower()
    if entrypoint:
        # The enclosing batch has300s; only its inner0.9s decision guard is
        # a candidate timing failure. External batch exhaustion is infrastructure.
        return (receipt.get('status')=='worker_error' and 'one_second_internal_deadline' in detail
                and not any(x in detail for x in ('resource','source changed','adapter_prerequisite','catalogue')))
    from .audit import classify_failure
    classification=classify_failure(receipt)
    return classification=='timeout' or 'one_second_internal_deadline' in detail


def qualify_one(envelope,freeze,quota):
    from . import qualification as q, audit
    from .log_audit import verify_worker_logs
    plan=c.read(c.OUT/f'qualification_plan_{quota}.json'); rows=[]; timed_failure=None
    with make_worker(f'qualification_{quota}',envelope,freeze,quota) as child:
        receipt=child.request({'operation':'entrypoint_checks'},envelope.timeout(300))
        c.save(c.OUT/f'entrypoint_checks_{quota}.json',receipt)
        if receipt['status']!='ok':
            envelope.guard()
            if timing_failure(receipt,entrypoint=True): timed_failure=receipt
            else: raise ValueError('Actual entrypoint checks failed: '+str(receipt.get('detail')))
        if timed_failure is None:
            for spec in plan['timed_requests']:
                state=q.state_for_request(plan,spec)
                row=decision(child,state,spec['arm'],envelope); row.update(spec)
                if row['completed']:
                    after=rules.CorrectedEnv().step(state,deserialize_action(row['action']))
                    row['after']=rules.serialize_snapshot(after)
                row['elapsed_seconds']=row['clock']['receipt_elapsed_seconds']
                c.save(c.OUT/f'qualification_{quota}'/f'{len(rows):03d}.json',row); rows.append(row)
                if row['completed']:
                    begun=time.perf_counter()
                    audit.audit_decision(row['observer'],row['action'],row['trace'],row['arm'],
                        before=row['before'],after=row['after'],expected_quota=quota if row['arm']=='search' else 16)
                    row['parent_validation_seconds']=time.perf_counter()-begun
                elif timing_failure(row['clock']):
                    timed_failure=row['clock']; break
                else: raise ValueError('Calibration non-timing failure: '+str(row['clock']))
                if len(rows)%16==0:
                    print(c.canonical(dict(stage='qualification',quota=quota,requests=len(rows),total=184)),flush=True)
    begun=time.perf_counter()
    log_check=verify_worker_logs(child.output_dir,boundary=envelope.guard,deadline=envelope.effective_deadline())
    log_seconds=time.perf_counter()-begun
    c.save(c.OUT/f'qualification_log_integrity_{quota}.json',log_check)
    if timed_failure is None:
        summary=q.validate_timed_panel(rows,plan)
    else:
        summary=dict(status='timing_not_qualified',timing_qualified=False,requests=len(rows),failure=timed_failure)
    summary.update(challenger_quota=quota,initialization_seconds=child.initialization['elapsed_seconds'],
        qualification_log_verify_seconds=log_seconds,
        maximum_parent_validation_seconds=max((r.get('parent_validation_seconds',0) for r in rows),default=0))
    c.save(c.OUT/f'qualification_summary_{quota}.json',summary)
    print(c.canonical(dict(stage='qualification_complete',quota=quota,status=summary['status'],
        timing_qualified=summary['timing_qualified'])),flush=True)
    return summary


def select_quota(summaries):
    c.require(type(summaries) is list and len(summaries)==2,'Both predetermined quotas must be assessed')
    for summary,quota in zip(summaries,c.QUOTAS,strict=True):
        c.require(type(summary) is dict and type(summary.get('challenger_quota')) is int
                  and summary['challenger_quota']==quota,'Ordered exact integer calibration quota')
        c.require(summary.get('status') in ('passed','timing_not_qualified')
                  and type(summary.get('timing_qualified')) is bool
                  and summary['timing_qualified']==(summary['status']=='passed'),'Inconsistent qualification status')
    eligible=[s['challenger_quota'] for s in summaries if s['status']=='passed' and s['timing_qualified'] is True]
    return max(eligible) if eligible else None


def qualify(envelope,freeze):
    from . import statistics
    summaries=[qualify_one(envelope,freeze,quota) for quota in c.QUOTAS]
    quota=select_quota(summaries)
    c.save(c.OUT/'timing_selection.json',dict(selected_simulations=quota,summaries=summaries,outcomes_used=False))
    if quota is None: return None,None
    # Freeze the exact challenger and all calibration evidence before any games.
    evidence=[]
    for path in sorted(c.OUT.rglob('*')):
        if not path.is_file(): continue
        relative=path.relative_to(c.OUT)
        if relative.parts[0] in ('workers','qualification_32','qualification_64') or relative.name.startswith(('entrypoint_checks_','qualification_summary_','qualification_log_integrity_')) or relative.name=='timing_selection.json':
            envelope.guard(); evidence.append(c.artifact(path))
    selection=dict(schema='splendorust_search_scale_selected_comparison_v1',time=c.now(),
        source_freeze=c.artifact(c.OUT/'freeze.json'),challenger_simulations=quota,
        reference_simulations=16,challenger_profile=c.artifact(c.profile_path(quota)),
        reference_profile=c.artifact(c.PROFILE),schedule=c.artifact(c.OUT/'schedule.json'),
        calibration_evidence=evidence,automatic_fallback=False,clocks_reset=False)
    envelope.comparison_freeze=c.save(c.OUT/'comparison_freeze.json',selection); c.verify(evidence)
    begun=time.perf_counter(); verify_runtime(freeze,envelope); source_seconds=time.perf_counter()-begun
    synthetic=[dict(pair_index=p,game_index=2*p+g,search_seat=(p+g)%2,winner=None,reason='natural',challenger_simulations=quota) for p in range(100) for g in range(2)]
    begun=time.perf_counter()
    result=statistics.analyze_games(synthetic,evidence_gates_passed=True,boundary=envelope.guard)
    statistics.verify_analysis(synthetic,result,boundary=envelope.guard)
    statistic_seconds=time.perf_counter()-begun
    initialization=max(s['initialization_seconds'] for s in summaries)
    validation=max(s['maximum_parent_validation_seconds'] for s in summaries)
    logs=max(s['qualification_log_verify_seconds'] for s in summaries)
    overhead=1.35*(source_seconds+statistic_seconds+2*logs+48*(initialization+1+validation+logs))+30
    c.save(c.OUT/'fixed_overhead_measurement.json',dict(source_seconds=source_seconds,
        statistics_seconds=statistic_seconds,initialization_seconds=initialization,
        maximum_parent_validation_seconds=validation,qualification_log_verify_seconds=logs,
        sample_replay_maximum=48,forecast_seconds=overhead,statistical_measurement='200syntheticdraws; discarded'))
    return quota,overhead


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def classify_failure(receipt):
    from .audit import classify_failure as qualified_classification
    return qualified_classification(receipt)


def play_game(spec,envelope,freeze):
    from . import audit
    from .log_audit import verify_worker_logs
    env=rules.CorrectedEnv(seed=int(spec['seed_decimal'])); state=env.initial_state()
    initial=rules.serialize_snapshot(state); decode_training_snapshot(initial)
    folder=game_folder(spec); c.save(folder/'initial.json',dict(**spec,initial=initial))
    rows=[]; independent_seconds=0.; outcome=None; begun=time.perf_counter()
    name=f"{spec['stage']}_p{spec['pair_index']:03d}_g{spec['game_index']}"
    with make_worker(name,envelope,freeze,spec['challenger_simulations']) as child:
        for ply in range(2000):
            if state.terminal: break
            seat=state.current_player; arm='search' if seat==spec['search_seat'] else 'reference'
            row=decision(child,state,arm,envelope)
            row.update(ply=ply,pair_index=spec['pair_index'],game_index=spec['game_index'],stage=spec['stage'])
            row['worker_clock_file']=(child.output_dir/f"{row['clock']['request_id']}.json").relative_to(c.ROOT).as_posix()
            if row['completed']:
                state=env.step(state,deserialize_action(row['action'])); row['after']=rules.serialize_snapshot(state)
            c.save(folder/'moves'/f'{ply:04d}.json',row); rows.append(row)
            if row['completed']:
                t=time.perf_counter()
                audit.audit_decision(row['observer'],row['action'],row['trace'],arm,before=row['before'],after=row['after'],expected_quota=spec['challenger_simulations'] if arm=='search' else 16)
                independent_seconds+=time.perf_counter()-t
            if not row['completed']:
                failure=classify_failure(row['clock'])
                outcome=dict(reason='forfeit',winner=1-seat,loser=seat,failure=failure); break
            if not state.terminal and (row['after']==row['before'] or (row['action']['action_type']=='PASS' and
                len(rows)>1 and rows[-2]['action'] and rows[-2]['action']['action_type']=='PASS')):
                raise ValueError('Unresolved actual-game deadlock; no official result')
        if outcome is None:
            c.require(state.terminal,'2,000-ply safety cap reached; incomplete')
            outcome=dict(reason='natural',winner=state.winner)
    t=time.perf_counter()
    log_check=verify_worker_logs(child.output_dir,boundary=envelope.guard,deadline=envelope.effective_deadline())
    independent_seconds+=time.perf_counter()-t
    c.save(folder/'log_integrity.json',log_check)
    ending={**spec,**outcome,'final':rules.serialize_snapshot(state),'complete':True,'plies':len(rows),
            'collection_seconds':time.perf_counter()-begun}
    t=time.perf_counter()
    replay=audit.audit_game(initial,rows,ending,search_seat=spec['search_seat'],boundary=envelope.guard,
        expected_identity={k:spec[k] for k in ('stage','pair_index','game_index','challenger_simulations')})
    independent_seconds+=time.perf_counter()-t
    c.save(folder/'result.json',ending); c.save(folder/'online_replay.json',replay)
    c.save(folder/'cost.json',dict(independent_seconds=independent_seconds,
        maximum_arrival_seconds=max((r['clock']['receipt_elapsed_seconds'] or 0) for r in rows),
        collection_seconds=time.perf_counter()-begun,completed=len(rows)))
    return ending,rows,independent_seconds


def operational(envelope,freeze,schedule,overhead):
    start=time.perf_counter(); baseline=bytes_used(); independent=0.; games=[]
    for spec in (r for r in schedule if r['stage']=='operational'):
        game,rows,cost=play_game(spec,envelope,freeze); games.append(game); independent+=cost
        c.require(game['reason']=='natural' and all(r['clock']['receipt_elapsed_seconds']<=.7 for r in rows),
                  'Operational completion or 0.700-second margin failed')
        print(c.canonical(dict(stage='operational',completed_games=len(games),total=6,checks='passed')),flush=True)
    elapsed=time.perf_counter()-start; used=bytes_used(); factor=200/6*1.35
    collection=elapsed*factor; variable=used-baseline; evidence=used+variable*factor
    final=independent*factor+overhead
    remaining=min(14400,envelope.start+19800-time.monotonic(),envelope.wall_start+19800-time.time())
    passed=collection<remaining and evidence<448*c.MiB and final<1800
    gate=dict(passed=passed,operational_games=6,observed_seconds=elapsed,observed_variable_bytes=variable,
        projected_collection_seconds=collection,remaining_collection_seconds=remaining,projected_prefinal_bytes=evidence,
        projected_final_seconds=final,arithmetic_factor=factor,outcomes_used=False)
    c.save(c.OUT/'operational_gate.json',gate)
    c.require(passed,'Outcome-blind operational resource forecast failed')


def final_checks(envelope,freeze,schedule):
    from . import audit,statistics
    from .log_audit import verify_worker_logs
    records=[]; main_games=[]; replay_receipts=[]
    c.verify([envelope.comparison_freeze])
    selection=c.read(c.OUT/'comparison_freeze.json')
    quota=selection['challenger_simulations']
    c.verify([selection['source_freeze'],selection['challenger_profile'],selection['reference_profile'],selection['schedule']]+selection['calibration_evidence'])
    c.require(schedule==[{**r,'challenger_simulations':quota} for r in c.make_schedule()],'Schedule derivation changed')
    for spec in schedule:
        envelope.guard(); folder=game_folder(spec)
        initial=c.read(folder/'initial.json'); game=c.read(folder/'result.json')
        c.require(all(initial[k]==game[k]==v for k,v in spec.items()),'Saved game identity differs')
        expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(expected==initial['initial'],'Initial world differs')
        rows=[c.read(p) for p in sorted((folder/'moves').glob('*.json'))]
        c.require(len(rows)==game['plies'],'Move inventory differs')
        for row in rows:
            expected_worker=c.OUT/'workers'/f"{spec['stage']}_p{spec['pair_index']:03d}_g{spec['game_index']}"
            request_id=f"request-{row['ply']+1:08d}"
            expected_clock=expected_worker/f'{request_id}.json'
            c.require(row['worker_clock_file']==expected_clock.relative_to(c.ROOT).as_posix() and
                row['clock']['request_id']==request_id and row['clock']==c.read(expected_clock),'Trusted clock identity/copy differs')
            c.require(c.read(expected_worker/'inputs'/f'{request_id}.json')==
                dict(player=row['arm'],observer=row['observer'],before=row['before']),'Saved player input differs')
        replay_receipts.append(audit.audit_game(initial['initial'],rows,game,search_seat=spec['search_seat'],
            boundary=envelope.guard,expected_identity={k:spec[k] for k in ('stage','pair_index','game_index','challenger_simulations')}))
        if spec['stage']=='main': records.extend(rows); main_games.append(game)
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=206,receipts=replay_receipts))
    c.save(c.OUT/'performance.json',audit.summarise_performance(records,main_games))
    sample=audit.select_replay_sample(records)
    c.save(c.OUT/'replay_sample_manifest.json',sample)
    selected=sample['decisions'] if isinstance(sample,dict) else sample
    repetitions=[]
    # Separate bounded workers keep sample ordering from inheriting game state.
    for index,item in enumerate(selected):
        envelope.guard()
        row=next(r for r in records if all(r[k]==item[k] for k in ('pair_index','game_index','ply','arm')))
        with make_worker(f'audit_{index:02d}',envelope,freeze,quota) as child:
            again=decision(child,decode_training_snapshot(row['before']),row['arm'],envelope)
            c.save(c.OUT/'repeated_decisions'/f'{index:02d}.json',again)
            c.require(again['completed'] and again['action']==row['action'] and again['trace']==row['trace'],
                      'Sampled complete decision reproduction differs')
        repetitions.append({k:item[k] for k in ('pair_index','game_index','ply','arm')})
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(repetitions),identities=repetitions))
    log_checks=[]
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard()
        c.require(folder.is_dir(),'Unexpected worker inventory entry')
        log_checks.append(verify_worker_logs(folder,boundary=envelope.guard,deadline=envelope.effective_deadline()))
    c.require(len(log_checks)==208+len(repetitions),'Worker log inventory incomplete')
    c.save(c.OUT/'all_worker_log_integrity.json',dict(status='passed',workers=len(log_checks),receipts=log_checks))
    # Seal all completed game evidence before independent arithmetic.
    seal=[]
    for subdir in ('games','workers'):
        for p in sorted((c.OUT/subdir).rglob('*')):
            if p.is_file():
                envelope.guard(); seal.append(c.artifact(p))
    c.save(c.OUT/'completed_game_seal.json',dict(artifacts=seal))
    analysis=statistics.analyze_games(main_games,evidence_gates_passed=True,boundary=envelope.guard)
    independent=statistics.verify_analysis(main_games,analysis,boundary=envelope.guard)
    c.save(c.OUT/'independent_statistics.json',independent)
    c.save(c.OUT/'analysis.json',analysis)
    c.verify(seal); verify_runtime(freeze,envelope)
    return analysis


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Freeze identity changed')
    claim=c.read(c.OUT/'watchdog_claim.json'); c.require(claim['freeze_sha256']==expected_sha,'Watchdog identity')
    envelope=Envelope(claim); freeze=c.read(c.OUT/'freeze.json')
    envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status='incomplete'; failure=None; analysis=None
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope); envelope.admit()
        quota,overhead=qualify(envelope,freeze); envelope.finish()
        if quota is None:
            status='no_qualifying_challenger'
        else:
            schedule=[{**r,'challenger_simulations':quota} for r in c.read(c.OUT/'schedule.json')]
            c.verify([envelope.comparison_freeze])
            envelope.begin('operational'); operational(envelope,freeze,schedule,overhead); envelope.finish()
            c.verify([envelope.comparison_freeze])
            envelope.begin('main')
            for index,spec in enumerate(r for r in schedule if r['stage']=='main'):
                play_game(spec,envelope,freeze)
                print(c.canonical(dict(stage='main',completed_games=index+1,total=200)),flush=True)
            envelope.finish(); envelope.begin('final')
            analysis=final_checks(envelope,freeze,schedule); envelope.finish(); status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,larger_data_deferred=True,automatic_retries=0,
        note='Acceptance also requires owning watchdog cleanup receipt with no errors.'))
    print(c.canonical(dict(status=status,stage=envelope.stage,failure=failure)),flush=True)
    return 0 if status in ('completed_verified','no_qualifying_challenger') else 1


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('operation',choices=['prepare','run'])
    parser.add_argument('--freeze-sha256'); args=parser.parse_args()
    return prepare() if args.operation=='prepare' else run(args.freeze_sha256)


if __name__=='__main__':
    raise SystemExit(main())
