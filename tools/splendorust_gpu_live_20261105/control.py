"""One immutable live GPU attempt, supervised by real clocks and exact ownership."""
import argparse,os,sys,time,subprocess,traceback,shutil
from pathlib import Path
import psutil
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent,finish_owned_process,native_fallback_messages

def prior_complete():
    close=c.PRIOR/'final_closeout.json'
    c.require(c.sha(close)==c.PRIOR_CLOSEOUT_SHA,'Accepted prior inference probe')
    prior=c.read(close)
    c.require(prior['status']=='accepted_completed_verified_inference_only','Prior accepted status')
    c.verify([prior['plan'],prior['completion'],prior['independent_audit']])
    c.require(c.read(c.resolve(prior['independent_audit']))['status']=='saved_evidence_audit_passed','Prior independent acceptance')
    claim=c.read(c.PRIOR/'claim.json');execution=c.read(c.PRIOR/'execution.json')
    require_absent(dict(pid=claim['pid'],created=claim['created']));require_absent(execution['identity'])
    return c.artifact(close)

def check_active(value,identity,wall,perf):
    c.require(value['pid']==identity['pid'] and type(value['active']) is list,'Owned active-clock registry')
    for row in value['active']:
        c.require(abs(row['deadline_unix']-row['started_unix']-10.)<1e-5 and
            abs(row['deadline_perf']-row['started_perf']-10.)<1e-5,'Unextended whole-response deadline')
        c.require(wall<row['deadline_unix'] and perf<row['deadline_perf'],
            'Actual outer10s deadline: '+str(row['client_id']))

def build_plan():
    from . import qualification,games
    from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    old=c.read(c.PRIOR/'plan.json')
    return dict(schema='splendorust_gpu_live_fixed512_v1',time=c.now(),authorization='User accepted proposed GPU integration and small throughput study',
        prior=prior_complete(),checkpoint=old['checkpoint'],model_source=old['model_source'],prior_panel=old['panel'],
        torch_version_expected=old['torch_version_expected'],directml_version_expected=old['directml_version_expected'],
        device_name_contains=old['device_name_contains'],profiles=qualification.profile_bindings(),
        qualification=qualification.build_plan(),games=games.build_plan(),numerical=old['numerical'],
        gpu_implementation='full_transformer_primitives_v1',gpu_physical_batch=8,client_slots=8,live_callback_batch=1,
        maximum_batch_wait_seconds=.001,worker_affinity=list(range(19)),supervisor_affinity=[19],
        internal_seconds=9.,outer_seconds=10.,deployment_target_seconds=1.,descriptive_margin_seconds=.7,
        precision='float32',training_updates=0,automatic_retries=0,automatic_promotion=False,larger_data_deferred=True,
        CPU_control='Concurrent direct original compiled forward; one Torch numerical thread per call; shared immutable weights',
        source_guard=verify_runtime_sources(),ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        total_seconds=c.TOTAL_SECONDS,worker_seconds=c.WORKER_SECONDS,maximum_worker_rss_bytes=c.MAX_RSS,
        maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,available_ram_reserve_bytes=0,
        protocol=c.artifact(c.PROTOCOL),within_gpu_exact_replay_required=True,cross_backend_actions_descriptive=True,
        comparison_is_strength_evidence=False,operational_games_are_training_corpus=False)

def freeze(tests_path,review_paths):
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    c.require(not c.PLAN.exists(),'One prospective new attempt')
    tests=c.read(tests_path);c.require(tests['exit_code']==0 and tests['models']==tests['games']==0,'Passed model-free checks')
    c.verify(tests['inputs']+[tests['log']]);c.require(bool(review_paths),'Independent review required')
    for path in review_paths:
        review=c.read(path);c.require(review['approved'] and not review['blocking_findings'],'Independent implementation approval');c.verify(review['inputs'])
    plan=build_plan();plan['tests']=c.artifact(tests_path);plan['reviews']=[c.artifact(p) for p in review_paths]
    files=python_dependencies((c.ROOT/'tools/splendorust_gpu_live_20261105').glob('*.py'))
    files+=sorted((c.ROOT/'tests').glob('test_splendorust_gpu_live*20261105*.py'))
    entries=[c.artifact(p) for p in files]+list(c.nested_artifacts(plan))+c.read(c.PRIOR/'plan.json')['inputs']
    for path in (tests_path,*review_paths):entries+=list(c.nested_artifacts(c.read(path)))
    for item in plan['profiles'].values():entries+=list(c.nested_artifacts(c.read(c.resolve(item))))
    unique={}
    for item in entries:
        previous=unique.setdefault(item['path'],item);c.require(previous==item,'Consistent source identity')
    plan['inputs']=sorted(unique.values(),key=lambda x:x['path']);c.verify(plan['inputs'])
    c.require('torch' not in sys.modules,'Prospective freeze has no model execution')
    return c.save(c.PLAN,plan)

def run(expected):
    from tools.local_player_qualification_128.clock import _WindowsJob
    c.require(c.sha(c.PLAN)==expected,'Frozen live plan identity');plan=c.read(c.PLAN);c.verify(plan['inputs']);prior_complete()
    c.require(plan['total_seconds']==c.TOTAL_SECONDS and plan['worker_seconds']==c.WORKER_SECONDS,'Unchanged run bounds')
    for name in ('claim.json','execution.json','completion.json','failure.json'):c.require(not(c.OUT/name).exists(),'No implicit retry or duplicate run')
    psutil.Process().cpu_affinity(plan['supervisor_affinity'])
    mono,wall=time.monotonic(),time.time();proc=job=None;identity=None;error=None;cleanup=[];seen={};peak=0;status='failed'
    c.save(c.OUT/'claim.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),plan_sha256=expected,
        monotonic_origin=mono,unix_origin=wall,total_seconds=c.TOTAL_SECONDS,automatic_retries=0))
    folder=c.OUT/'worker';folder.mkdir(exist_ok=False)
    def guard(deep=False):
        c.require(time.monotonic()-mono<c.TOTAL_SECONDS and time.time()-wall<c.TOTAL_SECONDS,'Original total dual-clock bound')
        c.require(shutil.disk_usage(c.ROOT).free>=c.MIN_DISK,'Actual free disk floor')
        if deep:c.require(sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())<c.MAX_OUTPUT,'Actual evidence size bound')
        if identity:
            for route in ('cpu','gpu'):
                path=folder/route/'active_decisions.json'
                if path.exists():check_active(c.read(path),identity,time.time(),time.perf_counter())
    try:
        guard(True);env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        args=[sys.executable,'-B','-u','-m','tools.splendorust_gpu_live_20261105.worker','--plan',str(c.PLAN),'--plan-sha256',expected,'--output',str(folder)]
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            begun,begun_wall=time.monotonic(),time.time()
            proc=subprocess.Popen(args,cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            c.save(folder/'owner.json',dict(time=c.now(),identity=identity,args=args,monotonic_origin=begun,unix_origin=begun_wall))
            last_deep=0
            while proc.poll() is None:
                deep=time.monotonic()-last_deep>=2
                guard(deep)
                if deep:last_deep=time.monotonic()
                c.require(time.monotonic()-begun<c.WORKER_SECONDS and time.time()-begun_wall<c.WORKER_SECONDS,'Actual worker dual-clock bound')
                try:
                    parent=psutil.Process(proc.pid);c.require(parent.create_time()==identity['created'],'Exact worker identity')
                    members=[parent]+parent.children(recursive=True);rss=0
                    for member in members:
                        try:
                            key=(member.pid,member.create_time());seen[key]=dict(pid=key[0],created=key[1]);rss+=member.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss);c.require(rss<c.MAX_RSS,'Actual worker-tree RSS bound')
                except psutil.NoSuchProcess:pass
                time.sleep(.25)
            c.require(time.monotonic()-begun<c.WORKER_SECONDS and time.time()-begun_wall<c.WORKER_SECONDS,'Worker deadline at exit')
            c.require(proc.returncode==0,'Worker failure preserved; no retry')
        guard(True);status='completed'
    except BaseException:error=traceback.format_exc()
    finally:
        if job:
            try:job.close()
            except BaseException as e:cleanup.append(repr(e))
        if proc:
            try:finish_owned_process(proc,identity)
            except BaseException as e:cleanup.append(repr(e))
        for item in seen.values():
            try:require_absent(item)
            except BaseException as e:cleanup.append(repr(e))
        if cleanup:status='failed'
        execution=dict(time=c.now(),status=status,error=error,cleanup_errors=cleanup,identity=identity,
            observed_owned_identities=list(seen.values()),exit_code=None if proc is None else proc.returncode,
            unix_end=time.time(),monotonic_end=time.monotonic(),peak_rss_bytes=peak,plan_sha256=expected,automatic_retries=0)
        c.save(c.OUT/'execution.json',execution)
    if status!='completed':c.save(c.OUT/'failure.json',execution);return 1
    try:
        guard(True);c.verify(plan['inputs'])
        fallback=native_fallback_messages(folder/'stderr.log')
        c.save(c.OUT/'native_stderr_check.json',dict(stderr=c.artifact(folder/'stderr.log'),fallback_lines=fallback))
        c.require(not fallback,'Native CPU fallback/unsupported GPU execution')
        result=c.read(folder/'worker_result.json');c.require(result['status']=='completed_live_qualification_and_games','Complete qualified study only')
        c.save(c.OUT/'completion.json',dict(time=c.now(),status='worker_completed_inputs_preserved',plan=c.artifact(c.PLAN),
            execution=c.artifact(c.OUT/'execution.json'),result=c.artifact(folder/'worker_result.json'),inputs_unchanged=True))
        return 0
    except BaseException:
        c.save(c.OUT/'failure.json',dict(time=c.now(),status='postrun_check_failed',error=traceback.format_exc()));return 1

def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['freeze','run']);p.add_argument('--tests');p.add_argument('--reviews',nargs='*');p.add_argument('--plan-sha256');a=p.parse_args()
    if a.operation=='freeze':print(c.canonical(freeze(Path(a.tests),[Path(x) for x in a.reviews])));return 0
    return run(a.plan_sha256)
if __name__=='__main__':raise SystemExit(main())
