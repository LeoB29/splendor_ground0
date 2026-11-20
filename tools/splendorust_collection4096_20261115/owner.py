"""Separate collection-only owner; scientific training is not released here."""
import argparse,os,shutil,subprocess,sys,time,traceback
from copy import deepcopy
from pathlib import Path
import psutil
from . import common as c
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
    mono, wall = time.monotonic(), time.time()
    c.require(claim['monotonic_origin'] <= mono < phase['monotonic_deadline']
              and claim['unix_origin'] <= wall < phase['unix_deadline'], 'Original phase/global clock expired')
    c.require(shutil.disk_usage(c.OUT).free >= c.MIN_DISK, 'Actual free disk floor')

def begin_phase(claim, name):
    c.require(name in ('qualification','collection','corpus','final'),
              'Fixed study phase')
    mono, wall = time.monotonic(), time.time()
    cap = c.STAGES[name]
    cutoff = c.TOTAL_SECONDS if name == 'final' else c.MODEL_SECONDS
    value = dict(stage=name, claim=c.artifact(c.OUT/'claim.json'), monotonic=mono,
        unix_time=wall, cap_seconds=cap,
        monotonic_deadline=min(mono+cap, claim['monotonic_origin']+cutoff),
        unix_deadline=min(wall+cap, claim['unix_origin']+cutoff))
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
    baseline=output_bytes(c.OUT)
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
                if module==c.PACKAGE+'.collection.worker' and mode in ('qualification','games'):
                    from .collection.control import check_registry, active_registry_paths
                    paths={path for path in active_registry_paths(folder,mode) if path.exists()}
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


def contract():
    return dict(schema=c.SCHEMA,scope='collection_only',new_games=c.NEW_GAMES,
        partition_counts=c.NEW_GAME_COUNTS,combined_training_games=4096,existing_training_games=1024,
        fresh_assessment_games=256,collection_cells=c.CELLS,games_per_cell=16,
        excluded_qualification_games=2,teacher_checkpoint_sha256=c.PARENT_CHECKPOINT_SHA,
        teacher_simulations=512,worlds=3,cpuct=.4,depth=64,precision='float32',collection_concurrency=8,
        collection_internal_seconds=60.,collection_complete_seconds=75.,
        total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,stage_seconds=c.STAGES,
        cell_seconds=4*3600,qualification_optimizer_calls=2,scientific_optimizer_calls=0,
        maximum_model_replays=3*c.NEW_GAMES,main_strength_games=0,
        maximum_evidence=c.MAX_EVIDENCE_BYTES,prefinal_evidence=c.PREFINAL_EVIDENCE_BYTES,
        actual_disk_floor=c.MIN_DISK,collection_audit_rss=c.COLLECTION_RSS,fit_qualification_rss=c.FIT_RSS,
        available_ram_gate=False,resource_forecasts_may_stop=False,
        automatic_retries=0,replacement_games=0,automatic_promotion=False,automatic_followup=False,
        no_sixworld_outcomes_used_for_readiness=True)


def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    sources=python_dependencies((c.ROOT/c.PACKAGE.replace('.','/')).rglob('*.py'))
    tests=list((c.ROOT/'tests').glob('*independent_data*20261107*.py'))
    return [c.artifact(p) for p in sorted(set(sources+tests))]+[c.artifact(c.PROTOCOL)]


def prepare():
    from . import schedule
    from .collection import control as collector
    c.OUT.mkdir(parents=True,exist_ok=True)
    reservation=schedule.prepare();c.save(c.OUT/'reservation.json',reservation)
    item=collector.prepare(reservation)
    c.require('torch' not in sys.modules,'Model-free prospective preparation')
    return dict(status='prepared',collection_plan=item,source_inputs=source_inputs(),models=0,games=0)


def freeze(tests,reviews,probe_path):
    from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS
    c.require(not c.PLAN.exists(),'Exclusive separate execution freeze')
    proof=c.read(tests);summary=proof['suite_summary'];code=source_inputs()
    c.require(proof['exit_code']==0 and proof['inputs']==code and summary['tests_passed']==summary['tests_collected']>=300
        and summary['failures']==summary['skipped']==0 and not summary['tensor_runtime_loaded']
        and summary['models']==summary['games']==0,'Complete source-bound model-free regression suite')
    c.verify(proof['inputs']+[proof['log']]);authors=set();covered=set()
    for p in reviews:
        review=c.read(p);c.require(review['approved'] and not review['blocking_findings'],'Independent readiness review')
        c.verify(review['inputs']);authors.add(review['reviewer']);covered.update(i['path'] for i in review['inputs'])
    c.require(len(authors)>=3,'Three distinct independent reviewers')
    own={i['path'] for i in code if i['path'].startswith(c.PACKAGE.replace('.','/')+'/') and not Path(i['path']).name.startswith('tests_')}
    c.require(own<=covered,'All new production sources independently reviewed')
    probe_path=Path(probe_path).resolve()
    c.require(probe_path.is_relative_to(c.OUT/'prelaunch'),'Only separate prelaunch fixture evidence')
    probe=c.read(probe_path)
    c.require(probe['status']=='passed' and probe['actual_probe_owner_exit']==0
        and probe['actual_probe_owner_absent'] and probe['detached'] and probe['error'] is None
        and probe['models']==probe['games']==0,'Actual detached lifecycle qualification')
    c.verify(probe['inputs'])
    c.require(c.artifact(__file__) in probe['inputs'],'Current owner covered by detached fixture')
    old=c.read(c.PRIOR_OUT/'freeze.json')
    audit=c.read(c.PREPARATION_OUT/'pipeline_audit.json')
    c.require(audit['schema']=='corrected_collection_pipeline_audit_20261115_v1'
        and audit['qualification']['actual_production_loader_and_collator_exercised']
        and audit['qualification']['all_three_qualification_processes_observed_exit_zero_and_cleanup']
        and not audit['qualification']['historical_loader_used']
        and audit['accepted_corpus']['games']==1152
        and audit['accepted_corpus']['all_raw_semantics_transitions_callbacks_and_selected_exact_replays_verified']
        and all(x['frozen_binding_found'] and x['matches_frozen_binding'] for x in audit['source_bindings']),
        'Authenticated predecessor pipeline audit')
    c.verify([x['artifact'] for x in audit['source_bindings']])
    c.verify(list(c.nested_artifacts(audit['receipts'])))
    prior_acceptance=c.artifact(c.PRIOR_OUT/'collection/corpus_acceptance.json')
    accepted=c.read(c.resolve(prior_acceptance))
    c.require(accepted['status']=='accepted' and accepted['passed'] and accepted['game_count']==1152
        and accepted['partition_counts']==dict(train=1024,validation=64,offline=64),'Complete corrected prior corpus')
    sources=verify_runtime_sources();c.require(sources==old['corrected_source_receipt'],'Exact corrected rule source identity')
    value={key:deepcopy(old[key]) for key in RUNTIME_KEYS}
    value.update(contract());value.update(protocol=c.artifact(c.PROTOCOL),time=c.now(),
        user_authorization='Prepare larger three-world collection; actual launch is a separate user action.',
        original_pipeline_freeze=c.artifact(c.PRIOR_OUT/'freeze.json'),pipeline_audit=c.artifact(c.PREPARATION_OUT/'pipeline_audit.json'),
        reservation=c.read(c.OUT/'reservation.json'),collection_plan=c.artifact(c.OUT/'collection/plan.json'),
        corrected_source_receipt=sources,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        existing_training_corpus=dict(acceptance=prior_acceptance,catalogue=accepted['catalogue']),
        historical_parent_checkpoint=old['historical_parent_checkpoint'],unchanged_model_definition=old['unchanged_model_definition'],
        external_runtime_inputs=old['external_runtime_inputs'],tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        detached_lifecycle_qualification=c.artifact(probe_path),
        prelaunch_evidence=[c.artifact(p) for p in sorted((c.OUT/'prelaunch').rglob('*')) if p.is_file()],
        administrative_stop_freeze=c.artifact(c.ROOT/'outputs/splendorust_administrative_stop_20261115_01/freeze.json'))
    runtime=[dict(path=x['absolute_path'],bytes=x['bytes'],sha256=x['sha256']) for x in old['torch_runtime']]
    collection=c.read(c.OUT/'collection/plan.json')
    value['worker_inputs']=code+runtime+[c.artifact(c.PARENT_CHECKPOINT)]+collection['inputs']
    all_inputs=value['worker_inputs']+list(c.nested_artifacts(value))+list(c.nested_artifacts(collection))
    unique={}
    for item in all_inputs:c.require(unique.setdefault(item['path'],item)==item,'Consistent frozen descriptor')
    value['inputs']=sorted(unique.values(),key=lambda i:i['path']);c.verify(value['inputs'])
    c.require('torch' not in sys.modules,'Freeze runs no learned model')
    return c.save(c.PLAN,value)


def launch_readiness():
    """An explicit later launch cannot overlap the old six-world model owner."""
    active=c.ROOT/'outputs/splendorust_sixworld512_20261114_02'
    for rel in ('launcher_claim.json','launch/owner.json'):
        row=c.read(active/rel);require_absent(row['identity'])
    for path in (active/'jobs').glob('*/owner.json'):require_absent(c.read(path)['identity'])
    administrative=c.ROOT/'outputs/splendorust_administrative_stop_20261115_01'
    c.require(c.sha(administrative/'freeze.json')=='a7cbc874b504c8c854d758b4b24264a8dd25c196d44959871f1070ce78f2ffa4',
        'Exact separately qualified administrative stop')
    ended=c.read(administrative/'cancellation.json')
    c.verify([ended['claim'],ended['intent']])
    stop_claim=c.read(c.resolve(ended['claim']))
    c.require(stop_claim['freeze']==c.artifact(administrative/'freeze.json')
        and ended['status']=='administratively_cancelled' and ended['exact_owned_absence'],
        'Original owner and every recorded service fully cleaned up')
    for item in ended['observed_processes']:require_absent(item)
    c.require(not (c.OUT/'claim.json').exists(),'No duplicate collection claim')


def run(expected):
    launch_readiness();c.require(c.sha(c.PLAN)==expected,'Exact prepared execution freeze')
    plan=c.read(c.PLAN);c.require({k:plan.get(k) for k in contract()}==contract(),'Exact collection-only contract')
    c.verify(plan['inputs'])
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(verify_runtime_sources()==plan['corrected_source_receipt'],'Root corrected source guard')
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed=placement.configure_execution(plan,'isolated')
    started=time.monotonic()
    while not (c.OUT/'launch/release.json').exists():
        c.require(time.monotonic()-started<10,'Bounded external launcher release')
        time.sleep(.01)
    launcher=c.read(c.OUT/'launcher_claim.json');owned=c.read(c.OUT/'launch/owner.json')
    release=c.read(c.OUT/'launch/release.json')
    me=dict(pid=os.getpid(),created=psutil.Process().create_time())
    c.require(owned['identity']==me and owned['parent']==launcher['identity']
        and owned['freeze_sha256']==expected and owned['windows_job_assigned'] is True
        and release['identity']==me and release['freeze_sha256']==expected,
        'Exact released external owner and frozen job')
    parent=psutil.Process(launcher['identity']['pid'])
    c.require(parent.create_time()==launcher['identity']['created'],'Exact live external launcher')
    claim=dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        monotonic_origin=launcher['monotonic_origin'],unix_origin=launcher['unix_origin'],freeze=c.artifact(c.PLAN),
        plan_sha256=expected,total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,placement=placed,automatic_retries=0)
    c.save(c.OUT/'claim.json',claim);receipts=[];error=None
    try:
        module=c.PACKAGE+'.collection.worker';phase=begin_phase(claim,'qualification')
        for mode,container,worker,rss in (
            ('qualification','collection/qualification',module,c.COLLECTION_RSS),
            ('audit_qualification','collection/qualification_audit',module,c.COLLECTION_RSS),
            ('qualification','training_qualification',c.PACKAGE+'.fit',c.FIT_RSS)):
            receipts.append(run_job(claim,phase,expected,worker,mode,c.OUT/container,
                seconds=4*3600,rss=rss,model=mode!='audit_qualification'))
        end_phase(claim,phase)
        c.save(c.OUT/'qualification_gate.json',dict(status='passed',plan=c.artifact(c.PLAN),phase=phase,
            actual_exits=receipts[-3:],scientific_games_released=0,qualification_optimizer_calls=2,excluded_engineering_games=2))
        phase=begin_phase(claim,'collection');collection=c.read(c.OUT/'collection/plan.json')
        for cell in collection['cells']:
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
        error=error,original_claim=c.artifact(c.OUT/'claim.json'),time=c.now(),scientific_optimizer_calls=0))
    c.status(c.OUT/'stage.json',dict(status=status,stage=status,time=c.now()))
    if error:c.save(c.OUT/'failure.json',dict(status=status,error=error,execution=execution));return 1
    c.save(c.OUT/'completion.json',dict(status=status,execution=execution,
        independent_acceptance=c.artifact(c.OUT/'final_audit/worker/result.json'),no_automatic_followup=True));return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare','freeze','run'))
    p.add_argument('--tests');p.add_argument('--reviews',nargs='*');p.add_argument('--probe');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.operation=='prepare':print(c.canonical(prepare()));return 0
    if a.operation=='freeze':print(c.canonical(freeze(a.tests,a.reviews,a.probe)));return 0
    return run(a.freeze_sha256)

if __name__=='__main__':raise SystemExit(main())
