"""Separate collection-only owner; scientific training is not released here."""
import argparse,os,shutil,subprocess,sys,time,traceback
from copy import deepcopy
from pathlib import Path
import psutil
from . import common as c, clocks
from tools.splendorust_gpu_probe_20261105.control import require_absent,finish_owned_process


def child_environment(repetition=None):
    env = os.environ.copy()
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(SELFPLAY_JOB_SCOPE='pilot', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='', RAYON_NUM_THREADS='1')
    if repetition is not None:
        c.require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
        env['INDEPENDENT_DATA_REPETITION'] = str(repetition)
    return env

def guard(claim, phase):
    mono,wall=time.monotonic(),time.time()
    clocks.check(claim,mono=mono,wall=wall,model=phase['stage']!='final')
    clocks.validate_phase(claim,phase)
    c.require(phase['monotonic']<=mono<phase['monotonic_deadline'] and
              phase['unix_time']<=wall<phase['unix_deadline'],'Original phase/global clock expired')
    c.require(shutil.disk_usage(c.OUT).free>=c.MIN_DISK,'Actual free disk floor')


def begin_phase(claim,name):
    value=clocks.phase_limits(claim,name)
    value['claim']=c.artifact(c.OUT/'claim.json')
    guard(claim,value)
    return c.save(c.OUT/'stages'/f'{name}_start.json',value)


def end_phase(claim, descriptor):
    phase = c.read(c.resolve(descriptor)); guard(claim,phase)
    return c.save(c.OUT/'stages'/f"{phase['stage']}_end.json", dict(stage=phase['stage'],
        start=descriptor, monotonic=time.monotonic(), unix_time=time.time()))

def observe_tree(identity, observed):
    total=0
    try:
        parent=psutil.Process(identity['pid'])
        c.require(parent.create_time()==identity['created'],'Exact owned process identity')
        for child in [parent]+parent.children(recursive=True):
            try:
                item=dict(pid=child.pid,created=child.create_time())
                observed[(item['pid'],item['created'])]=item
                total+=child.memory_info().rss
            except psutil.NoSuchProcess: pass
    except psutil.NoSuchProcess: pass
    return total

def cleanup_owned(proc,job,identity,observed,errors):
    if job:
        try: job.close()
        except BaseException as exc: errors.append(repr(exc))
    if proc:
        try: finish_owned_process(proc,identity)
        except BaseException as exc: errors.append(repr(exc))
    items=[*observed.values(),*([identity] if identity else [])]
    wait_for_absence(items)
    for item in items:
        try: require_absent(item)
        except BaseException as exc: errors.append(repr(exc))


def wait_for_absence(items,seconds=5.):
    """Windows job termination is asynchronous; still require exact absence."""
    end=time.monotonic()+seconds
    while True:
        alive=False
        for item in items:
            try:
                p=psutil.Process(item['pid'])
                alive|=p.create_time()==item['created'] and p.is_running()
            except psutil.NoSuchProcess:pass
        if not alive or time.monotonic()>=end:return
        time.sleep(.02)

def run_job(claim, phase_descriptor, expected, module, mode, container, *, seconds,
            rss, fit_id=None, extra_args=(), model=True, cell_deadline=None):
    """Fresh single-use process with actual exit/cleanup and clipped parent clocks."""
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_gpu_live_20261105_03.status_io import read_status_json
    container=Path(container); container.mkdir(parents=True,exist_ok=False)
    folder=container/'worker'; folder.mkdir()
    phase=c.read(c.resolve(phase_descriptor)); begun,wall=time.monotonic(),time.time()
    deadline=min(begun+seconds,phase['monotonic_deadline'])
    wall_deadline=min(wall+seconds,phase['unix_deadline'])
    if cell_deadline:
        deadline=min(deadline,cell_deadline['monotonic_deadline'])
        wall_deadline=min(wall_deadline,cell_deadline['unix_deadline'])
    args=[sys.executable,'-B','-u','-m',module,'--mode',mode,'--plan-sha256',expected,'--output',str(folder)]
    if fit_id is not None: args += ['--fit-id',fit_id]
    args+=list(extra_args)
    proc=job=identity=None
    observed,cleanup,error,status={},[],None,'failed'
    peak=0; registries=set(); deep_at=-float('inf')
    # Closed evidence is counted once. The active subtree is sampled; no repeated full-study walk per move.
    baseline=original_evidence_bytes()+output_bytes(c.OUT)
    cap=c.MAX_EVIDENCE_BYTES if phase['stage']=='final' else c.PREFINAL_EVIDENCE_BYTES
    c.status(c.OUT/'stage.json',dict(time=c.now(),stage=phase['stage'],mode=mode,fit_id=fit_id,
        container=container.relative_to(c.OUT).as_posix(),status='starting'))
    try:
        guard(claim,phase)
        c.require(baseline<cap,'Actual evidence bound before worker')
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(args,cwd=c.ROOT,env=child_environment(),stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            c.atomic_save(container/'owner.json',dict(time=c.now(),identity=identity,args=args,mode=mode,
                fit_id=fit_id,global_claim=c.artifact(c.OUT/'claim.json'),phase_clock=phase_descriptor,
                monotonic_origin=begun,unix_origin=wall,monotonic_deadline=deadline,unix_deadline=wall_deadline,
                maximum_seconds=seconds,maximum_rss_bytes=rss,windows_job_assigned=True,plan_sha256=expected,
                model_work=model))
            while proc.poll() is None:
                guard(claim,phase)
                c.require(time.monotonic()<deadline and time.time()<wall_deadline,'Actual clipped worker clock')
                used=observe_tree(identity,observed);peak=max(peak,used)
                c.require(used<=rss,'Actual aggregate owned worker RSS cap')
                if module==c.PACKAGE+'.collection.worker' and mode in ('recovery_qualification','recovery_games','games'):
                    from .collection.control import check_registry, active_registry_paths
                    runtime_name='cpu_runtime' if mode=='recovery_qualification' else 'runtime'
                    paths={path for path in [folder/runtime_name/'active_decisions.json'] if path.exists()}
                    c.require(registries<=paths,'Active decision registry cannot disappear')
                    registries|=paths
                    for path in sorted(registries):
                        check_registry(read_status_json(path),identity,time.time(),time.perf_counter())
                if time.monotonic()-deep_at>=5:
                    deep_at=time.monotonic()
                    c.require(baseline+output_bytes(container)<cap,'Actual evidence cap')
                time.sleep(.25)
            guard(claim,phase)
            c.require(time.monotonic()<deadline and time.time()<wall_deadline,'Actual worker clock at exit')
            c.require(proc.returncode==0,'Worker failure preserved without retry')
            result=c.read(folder/'result.json')
            c.require(result['status']=='passed' and result['mode']==mode,'Completed exact worker role')
            c.require(result['identity']==identity,'Completed exact worker identity')
            wait_for_absence([*observed.values(),identity])
            for item in [*observed.values(),identity]: require_absent(item)
            status='passed'
    except BaseException: error=traceback.format_exc()
    finally: cleanup_owned(proc,job,identity,observed,cleanup)
    if cleanup: status='failed'
    receipt=c.save(container/'execution.json',dict(time=c.now(),status=status,mode=mode,fit_id=fit_id,
        identity=identity,worker_exit_code=None if proc is None else proc.returncode,
        observed_owned_identities=list(observed.values()),cleanup_errors=cleanup,error=error,
        monotonic_origin=begun,unix_origin=wall,monotonic_end=time.monotonic(),unix_end=time.time(),
        maximum_seconds=seconds,monotonic_deadline=deadline,unix_deadline=wall_deadline,
        peak_rss_bytes=peak,plan_sha256=expected,retries=0))
    c.require(status=='passed','Failed worker preserved: '+str(container))
    return receipt


def output_bytes(folder):
    total=0;pending=[Path(folder)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):total+=entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:c.require(entry.name in ('stage.tmp','progress.tmp'),'Only mutable status rename may vanish')
    return total


_ORIGINAL_BYTES = None


def original_evidence_bytes():
    global _ORIGINAL_BYTES
    if _ORIGINAL_BYTES is None:
        _ORIGINAL_BYTES=output_bytes(c.ORIGINAL_OUT)+output_bytes(c.PREVIOUS_OUT)
    return _ORIGINAL_BYTES


def contract():
    """Scientific contract remains the exact previously frozen collector."""
    from tools.splendorust_collection4096_20261115.owner import contract as original_contract
    return original_contract()


def recovery_contract():
    return dict(schema='collection4096_timeout_resume_v2',original_freeze_sha256=c.ORIGINAL_FREEZE_SHA,
        previous_recovery_freeze_sha256=c.PREVIOUS_FREEZE_SHA,
        original_attempt_status='incomplete',previous_attempt_status='incomplete',
        verification_only=False,gameplay_retries=8,replacement_games=0,
        successful_prefix_cells=135,completed_source_games=2168,partial_cell='c135',
        preserved_completed_replays=6480,previous_verification_retry=dict(game_id='g0666',ply=40),
        specific_gameplay_retries=[dict(game_id='g'+str(g),ply=p) for g,p in
            [(2168,27),(2169,23),(2170,22),(2171,22),(2172,22),(2173,21),(2174,20),(2175,17)]],
        new_verification_retries=0,verification_retry_limit=1,unstarted_selected_replays_maximum=48,
        supplemental_seconds=7200,original_cell_deadline_not_reset=True,
        supplement_is_separate_partial_cell_clock=True,
        original_qualification_receipts_reused=True,new_qualification_fixture_decisions=116,
        new_qualification_games=0,new_optimizer_updates=0,remaining_cell_indices=list(range(136,208)),
        original_collection_unix_deadline=1795174250.1671505,
        original_model_seconds=144*3600,original_total_seconds=168*3600,
        downtime_charged=True,automatic_retries=0,maximum_selected_replay_attempts=9985,
        corrected_worker_startup_guard_required=True,all_original_evidence_preserved=True,
        all_previous_recovery_evidence_preserved=True,completed_game_replacements=0,
        completed_moves_retried=0,per_decision_internal_seconds=60.,per_decision_outer_seconds=75.)


def validate_plan(plan):
    c.require(c.sha(c.ORIGINAL_OUT/'freeze.json')==clocks.ORIGINAL_FREEZE_SHA,'Exact source freeze')
    original=c.read(c.ORIGINAL_OUT/'freeze.json')
    c.require(plan['original_freeze']==c.artifact(c.ORIGINAL_OUT/'freeze.json') and
              plan['recovery']==recovery_contract(),'Exact separately frozen disclosed partial-game recovery')
    c.require({k:plan.get(k) for k in contract()}=={k:original.get(k) for k in contract()},
              'Scientific players/sample/rules/resources and original limits unchanged')
    c.require(plan['corrected_source_receipt']==original['corrected_source_receipt'] and
              plan['existing_training_corpus']==original['existing_training_corpus'],
              'Corrected rules and prior accepted source unchanged')
    collection=c.read(c.OUT/'collection/plan.json')
    c.require(collection==c.read(c.ORIGINAL_OUT/'collection/plan.json') and
              plan['collection_plan']==c.artifact(c.OUT/'collection/plan.json'),
              'Byte-equivalent scientific collection plan in new output context')
    c.require(plan['previous_freeze']==c.artifact(c.PREVIOUS_OUT/'freeze.json') and
              plan['saved_partial135']==plan['saved_boundary'],'Exact prior recovery and partial-cell boundary')
    c.require(c.sha(c.PREVIOUS_OUT/'freeze.json')==c.PREVIOUS_FREEZE_SHA,'Previous freeze unchanged')
    c.verify([plan['saved_boundary']])
    return plan


def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    sources=python_dependencies((c.ROOT/c.PACKAGE.replace('.','/')).rglob('*.py'))
    return [c.artifact(p) for p in sorted(set(sources))]+[c.artifact(c.PROTOCOL)]


def launch_readiness():
    """Bind successful original qualification and require every old owner absent."""
    c.require(not (c.OUT/'claim.json').exists(),'No duplicate recovery root')
    c.require(c.sha(c.ORIGINAL_OUT/'freeze.json')==clocks.ORIGINAL_FREEZE_SHA,'Unchanged original freeze')
    claim=c.read(c.ORIGINAL_OUT/'claim.json');rootexec=c.read(c.ORIGINAL_OUT/'execution.json')
    c.require(rootexec['status']=='incomplete' and c.read(c.ORIGINAL_OUT/'failure.json')['status']=='incomplete',
              'Original incomplete attempt retained')
    identities=[dict(pid=claim['pid'],created=claim['created'])]
    for rel in ('launcher_claim.json','launch/owner.json'):
        identities.append(c.read(c.ORIGINAL_OUT/rel)['identity'])
    executions=list(rootexec['segments'])
    executions.append(c.artifact(c.ORIGINAL_OUT/'collection/cells/c041/execution.json'))
    for item in executions:
        c.verify([item]);row=c.read(c.resolve(item))
        identities.extend(row.get('observed_owned_identities',[]))
        if row.get('identity'):identities.append(row['identity'])
    for item in {(x['pid'],x['created']):x for x in identities}.values():require_absent(item)
    gate=c.read(c.ORIGINAL_OUT/'qualification_gate.json')
    c.require(gate['status']=='passed' and gate['plan']==c.artifact(c.ORIGINAL_OUT/'freeze.json') and
              gate['qualification_optimizer_calls']==2 and gate['excluded_engineering_games']==2 and
              len(gate['actual_exits'])==3,'Reuse exact completed original three-part qualification')
    for item in gate['actual_exits']:
        c.verify([item]);row=c.read(c.resolve(item))
        c.require(row['status']=='passed' and row['worker_exit_code']==0 and not row['cleanup_errors']
                  and row['error'] is None,'Original qualifications actually exited zero with cleanup')
    c.require(c.sha(c.PREVIOUS_OUT/'freeze.json')==c.PREVIOUS_FREEZE_SHA,'Exact previous recovery freeze')
    previous=c.read(c.PREVIOUS_OUT/'execution.json')
    c.require(previous['status']=='incomplete' and c.read(c.PREVIOUS_OUT/'failure.json')['status']=='incomplete',
              'Both failed attempts remain incomplete')
    identities=[]
    priorclaim=c.read(c.PREVIOUS_OUT/'claim.json')
    identities.append(dict(pid=priorclaim['pid'],created=priorclaim['created']))
    identities.append(c.read(c.PREVIOUS_OUT/'launcher_claim.json')['identity'])
    paths=[c.PREVIOUS_OUT/'launch/execution.json',c.PREVIOUS_OUT/'collection/cells/c135/execution.json']
    paths += [c.resolve(item) for item in previous['segments']]
    for path in paths:
        row=c.read(path);identities.extend(row.get('observed_owned_identities',[]))
        if row.get('identity'):identities.append(row['identity'])
    for item in {(x['pid'],x['created']):x for x in identities}.values():require_absent(item)
    clocks.original_deadlines()


def run(expected):
    launch_readiness();c.require(c.sha(c.PLAN)==expected,'Exact prepared recovery freeze')
    plan=validate_plan(c.read(c.PLAN));c.verify(plan['inputs'])
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(verify_runtime_sources()==plan['corrected_source_receipt'],'Root corrected source guard')
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed=placement.configure_execution(plan,'isolated')
    started=time.monotonic()
    while not (c.OUT/'launch/release.json').exists():
        c.require(time.monotonic()-started<10,'Bounded external launcher release');time.sleep(.01)
    launcher=c.read(c.OUT/'launcher_claim.json');owned=c.read(c.OUT/'launch/owner.json')
    release=c.read(c.OUT/'launch/release.json');me=dict(pid=os.getpid(),created=psutil.Process().create_time())
    c.require(owned['identity']==me and owned['parent']==launcher['identity'] and
        owned['freeze_sha256']==expected and owned['windows_job_assigned'] is True and
        release['identity']==me and release['freeze_sha256']==expected,
        'Exact released external recovery owner and Windows job')
    parent=psutil.Process(launcher['identity']['pid'])
    c.require(parent.create_time()==launcher['identity']['created'],'Exact live external launcher')
    clock_keys=('monotonic_origin','unix_origin','monotonic_deadline','unix_deadline','recovery_anchor',
        'recovery_anchor_sha256','original_deadlines','original_deadlines_sha256','downtime_charged','per_decision_clocks_rebased')
    claim=dict(time=c.now(),pid=me['pid'],created=me['created'],freeze=c.artifact(c.PLAN),
        plan_sha256=expected,total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,placement=placed,
        automatic_retries=0,recovery=plan['recovery'],original_claim=c.artifact(c.ORIGINAL_OUT/'claim.json'),
        **{k:launcher[k] for k in clock_keys})
    clocks.check(claim,model=True)
    c.require(claim['original_deadlines']==clocks.original_deadlines(),'Exact original clocks at launch')
    c.save(c.OUT/'claim.json',claim);receipts=[];error=None
    try:
        module=c.PACKAGE+'.collection.worker';phase=begin_phase(claim,'supplemental')
        for mode,container,model in (
            ('saved_boundary','recovery_boundary',False),
            ('recovery_qualification','recovery_qualification',True),
            ('audit_recovery_qualification','recovery_qualification_audit',False),
            ('recovery_games','recovery_games',True),
            ('supplemental_audit','recovery_audit',False)):
            receipts.append(run_job(claim,phase,expected,module,mode,c.OUT/'collection'/container,
                seconds=clocks.SUPPLEMENTAL_SECONDS,rss=c.COLLECTION_RSS,model=model))
        end_phase(claim,phase)
        c.save(c.OUT/'recovery_gate.json',dict(status='passed',plan=c.artifact(c.PLAN),phase=phase,
            actual_exits=list(receipts),original_qualification=c.artifact(c.ORIGINAL_OUT/'qualification_gate.json'),
            original_collection_qualification=c.artifact(c.ORIGINAL_OUT/'collection/qualification_gate.json'),
            recovered_cell_audit=c.artifact(c.OUT/'collection/recovery_audit/worker/saved_audit.json'),
            original_attempt_status='incomplete',previous_attempt_status='incomplete',
            previous_recovery_freeze=c.artifact(c.PREVIOUS_OUT/'freeze.json'),
            new_games_released=0,gameplay_retries=8,resumed_games=8,preserved_completed_games=2168,
            verification_retries=1,new_verification_retries=0,new_fixture_decisions=116))
        phase=begin_phase(claim,'collection');collection=c.read(c.OUT/'collection/plan.json')
        for cell in collection['cells'][136:]:
            guard(claim,c.read(c.resolve(phase)));mono,wall=time.monotonic(),time.time()
            clock=dict(monotonic_deadline=mono+4*3600,unix_deadline=wall+4*3600);ident=cell['cell_id']
            c.save(c.OUT/'collection/cell_clocks'/f'{ident}.json',dict(**clock,monotonic_origin=mono,
                unix_origin=wall,phase=phase,cell_id=ident))
            for mode,container in (('games','cells'),('audit_cell','cell_audits')):
                receipts.append(run_job(claim,phase,expected,module,mode,c.OUT/'collection'/container/ident,
                    seconds=4*3600,rss=c.COLLECTION_RSS,extra_args=('--cell-id',ident),
                    model=mode=='games',cell_deadline=clock))
        end_phase(claim,phase);phase=begin_phase(claim,'corpus')
        receipts.append(run_job(claim,phase,expected,module,'corpus',c.OUT/'collection/corpus_audit',
            seconds=24*3600,rss=c.COLLECTION_RSS,model=False))
        receipts.append(run_job(claim,phase,expected,c.PACKAGE+'.audit','selection',c.OUT/'union_build',
            seconds=4*3600,rss=c.FIT_RSS,model=False))
        end_phase(claim,phase);phase=begin_phase(claim,'final')
        receipts.append(run_job(claim,phase,expected,c.PACKAGE+'.audit','final',c.OUT/'final_audit',
            seconds=24*3600,rss=c.COLLECTION_RSS,model=False))
        c.verify(plan['inputs']);end_phase(claim,phase)
    except BaseException:error=traceback.format_exc()
    status='incomplete' if error else 'verified_pending_actual_owner_exit'
    execution=c.save(c.OUT/'execution.json',dict(status=status,freeze_sha256=expected,segments=receipts,
        error=error,original_claim=c.artifact(c.OUT/'claim.json'),time=c.now(),scientific_optimizer_calls=0,
        original_attempt_status='incomplete',recovered_verification_separate=True))
    c.status(c.OUT/'stage.json',dict(status=status,stage=status,time=c.now()))
    if error:c.save(c.OUT/'failure.json',dict(status=status,error=error,execution=execution));return 1
    c.save(c.OUT/'completion.json',dict(status=status,execution=execution,
        independent_acceptance=c.artifact(c.OUT/'final_audit/worker/result.json'),no_automatic_followup=True,
        original_attempt_status='incomplete',recovered_verification_separate=True));return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('run',))
    p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    return run(a.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
