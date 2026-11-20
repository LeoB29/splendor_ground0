"""Freeze once, supervise one GPU worker, preserve every outcome."""
import argparse,os,sys,time,subprocess,traceback,shutil
from pathlib import Path
import psutil
from . import common as c

def require_absent(identity):
    try:c.require(psutil.Process(identity['pid']).create_time()!=identity['created'],'Exact previous process still alive')
    except psutil.NoSuchProcess:pass

def finish_owned_process(proc,identity):
    """Fallback cleanup targets only this launch's exact process identity."""
    if proc is None:return
    if proc.poll() is None:
        p=psutil.Process(proc.pid)
        c.require(identity is not None and p.create_time()==identity['created'],'Exact owned cleanup identity')
        children=p.children(recursive=True)
        for child in reversed(children):
            try:child.kill()
            except psutil.NoSuchProcess:pass
        try:p.kill()
        except psutil.NoSuchProcess:pass
    proc.wait(timeout=5.)
    if identity is not None:require_absent(identity)

def native_fallback_messages(path):
    from .worker import is_backend_fallback
    return [line for line in Path(path).read_text(encoding='utf-8',errors='replace').splitlines()
            if is_backend_fallback(line)]

def previous_complete():
    folder=c.ROOT/'outputs/splendorust_policy_head_20261104_01'
    owner=c.ROOT/'outputs/splendorust_policy_head_owner_20261104_01'
    close=folder/'final_closeout.json'
    c.require(c.sha(close)=='a188767b6479b861e78f05258609e33bf998a488d22d52ca8d5f4a916ff6ca21','Accepted completed predecessor')
    value=c.read(close)
    c.verify([value['freeze'],value['completion'],value['independent_review']])
    c.require(value['status']=='accepted_completed_verified' and c.read(c.resolve(value['independent_review']))['approved'],'Independent predecessor acceptance')
    c.require(c.read(owner/'completion.json')['study_closeout']==c.artifact(close),'Retained predecessor owner completed')
    for item in value['exact_owner_identities_absent']:require_absent(item)
    claim=c.read(owner/'claim.json');require_absent(dict(pid=claim['pid'],created=claim['created']))
    return c.artifact(close)

def source_binding():
    old=c.read(c.ROOT/'outputs/splendorust_policy_head_20261104_01/freeze.json')
    return old['historical_parent_checkpoint'],old['unchanged_model_definition'],old['external_runtime_inputs']

def external_inputs(inherited):
    site=Path(sys.prefix)/'Lib/site-packages'
    extra=[Path(sys.executable),site/'torch/version.py',site/'torch_directml_native.cp312-win_amd64.pyd',
        site/'torch_directml/DirectML.dll',*sorted((site/'torch_directml').glob('*.py'))]
    return list({x['path']:x for x in inherited+[c.artifact(p) for p in extra]}.values())

def freeze(test_path,reviews):
    from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    from . import panel
    c.require(not c.PLAN.exists(),'One prospective fixed probe')
    tests=c.read(test_path)
    c.require(tests['exit_code']==0 and tests['models']==tests['games']==0,'Passed model-free source-bound tests')
    c.verify(tests['inputs']+[tests['log']])
    c.require(bool(reviews),'Independent implementation review')
    for path in reviews:
        value=c.read(path);c.require(value['approved'] and not value['blocking_findings'],'Independent approval without blockers');c.verify(value['inputs'])
    previous=previous_complete();saved=c.read(c.PANEL);panel.validate_panel(saved)
    checkpoint,model_source,inherited=source_binding()
    value=dict(schema='splendorust_gpu_inference_probe_v1',time=c.now(),authorization='User: ok let\'s do the gpu test',
        predecessor=previous,panel=c.artifact(c.PANEL),checkpoint=checkpoint,model_source=model_source,
        protocol=c.artifact(c.PROTOCOL),host_inventory=c.artifact(c.OUT/'prelaunch/host_inventory.json'),
        torch_version_expected='2.4.1+cpu',directml_version_expected='0.2.5.dev240914',
        device_name_contains='AMD Radeon RX 7800 XT',cpu_affinity=[0],supervisor_affinity=[2],
        batches=[1,2,8,32,64],gpu_implementation='full_transformer_primitives_v1',cpu_primitive_bridge_check=True,
        numerical=dict(atol=1e-4,rtol=1e-4,value_abs_atol=1e-4,policy_probability_atol=1e-5,
            robust_argmax_gap=1e-3,require_legal_argmax_same=False,require_robust_argmax_same=True),
        timing=dict(warmups=5,repetitions=30,order='paired_alternating_cpu_gpu',
            gpu_scope='CPU input tensor to GPU, full forward, output to CPU and readback',
            cpu_scope='CPU input tensor, accepted compiled full forward, CPU output readback',
            input_list_conversion_excluded_equally=True),
        source_guard=verify_runtime_sources(),ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        total_seconds=c.TOTAL_SECONDS,worker_seconds=c.WORKER_SECONDS,maximum_worker_rss_bytes=c.MAX_RSS,
        maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,available_ram_reserve_bytes=0,
        precision='float32',new_games=0,training_updates=0,automatic_retries=0,automatic_promotion=False,
        larger_data_deferred=True,live_player_qualification=False,full_game_speedup_established=False,
        fallback_policy='Any CPU fallback or unsupported GPU operator invalidates GPU qualification; no alternate route',
        model_free_tests=c.artifact(test_path),reviews=[c.artifact(p) for p in reviews])
    files=python_dependencies((c.ROOT/'tools/splendorust_gpu_probe_20261105').glob('*.py'))
    files+=sorted((c.ROOT/'tests').glob('test_splendorust_gpu_probe*20261105*.py'))
    entries=[c.artifact(p) for p in files]+list(c.nested_artifacts(value))+list(c.nested_artifacts(saved))+external_inputs(inherited)
    for path in [test_path,*reviews]:entries+=list(c.nested_artifacts(c.read(path)))
    unique={}
    for item in entries:
        prev=unique.setdefault(item['path'],item);c.require(prev==item,'Conflicting frozen source pin')
    value['inputs']=sorted(unique.values(),key=lambda x:x['path']);c.verify(value['inputs'])
    c.require('torch' not in sys.modules,'Freeze is model-free')
    return c.save(c.PLAN,value)

def verify_plan(expected):
    c.require(c.sha(c.PLAN)==expected,'Exact prospective GPU plan')
    plan=c.read(c.PLAN);c.verify(plan['inputs'])
    c.require(plan['total_seconds']==c.TOTAL_SECONDS and plan['worker_seconds']==c.WORKER_SECONDS and
        plan['new_games']==plan['training_updates']==plan['automatic_retries']==0,'Fixed finite inference-only scope')
    previous_complete()
    for name in ('claim.json','completion.json','failure.json','execution.json'):
        c.require(not (c.OUT/name).exists(),'No duplicate/implicit retry')
    return plan

def run(expected):
    from tools.local_player_qualification_128.clock import _WindowsJob
    plan=verify_plan(expected);psutil.Process().cpu_affinity(plan['supervisor_affinity'])
    mono,wall=time.monotonic(),time.time();proc=job=None;identity=None;error=None;cleanup_errors=[];peak=0;status='failed'
    c.save(c.OUT/'claim.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),time=c.now(),
        monotonic_origin=mono,unix_origin=wall,plan_sha256=expected,total_seconds=c.TOTAL_SECONDS,automatic_retries=0))
    def guard():
        c.require(time.monotonic()-mono<c.TOTAL_SECONDS and time.time()-wall<c.TOTAL_SECONDS,'Original dual-clock probe bound')
        c.require(shutil.disk_usage(c.ROOT).free>=c.MIN_DISK,'Actual free-disk floor')
        c.require(sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())<c.MAX_OUTPUT,'Actual new evidence bound')
    folder=c.OUT/'worker';folder.mkdir(exist_ok=False)
    try:
        guard();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        args=[sys.executable,'-B','-u','-m','tools.splendorust_gpu_probe_20261105.worker','--plan',str(c.PLAN),
            '--panel',str(c.PANEL),'--output',str(folder/'results'),'--plan-sha256',expected,'--panel-sha256',plan['panel']['sha256']]
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            begun,begun_wall=time.monotonic(),time.time()
            proc=subprocess.Popen(args,cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            c.save(folder/'owner.json',dict(identity=identity,time=c.now(),monotonic_start=begun,unix_start=begun_wall,args=args))
            while proc.poll() is None:
                guard();c.require(time.monotonic()-begun<c.WORKER_SECONDS and time.time()-begun_wall<c.WORKER_SECONDS,'Real worker deadline')
                try:
                    p=psutil.Process(proc.pid);c.require(p.create_time()==identity['created'],'Exact live worker identity')
                    rss=sum(x.memory_info().rss for x in [p]+p.children(recursive=True));peak=max(peak,rss)
                    c.require(rss<c.MAX_RSS,'Actual worker RSS bound')
                except psutil.NoSuchProcess:pass
                time.sleep(1.)
            c.require(time.monotonic()-begun<c.WORKER_SECONDS and time.time()-begun_wall<c.WORKER_SECONDS,'Real worker deadline at exit')
            c.require(proc.returncode==0,'Worker failed; retain diagnostics, no automatic retry')
        guard();status='completed'
    except BaseException:error=traceback.format_exc()
    finally:
        if job is not None:
            try:job.close()
            except BaseException as e:cleanup_errors.append(repr(e))
        if proc is not None:
            try:finish_owned_process(proc,identity)
            except BaseException as e:cleanup_errors.append(repr(e))
        if identity is not None:
            try:require_absent(identity)
            except BaseException as e:cleanup_errors.append(repr(e))
        if cleanup_errors:status='failed'
        execution=dict(status=status,error=error,cleanup_errors=cleanup_errors,identity=identity,
            exit_code=None if proc is None else proc.returncode,time=c.now(),monotonic_end=time.monotonic(),unix_end=time.time(),
            peak_rss_bytes=peak,plan_sha256=expected,automatic_retries=0)
        c.save(c.OUT/'execution.json',execution)
    if status!='completed':c.save(c.OUT/'failure.json',execution);return 1
    try:
        guard();c.verify(plan['inputs'])
        result=c.read(folder/'results/worker_result.json')
        c.require(result.get('status') is not None,'Explicit scientific outcome')
        fallback=native_fallback_messages(folder/'stderr.log')
        c.save(c.OUT/'native_stderr_check.json',dict(time=c.now(),stderr=c.artifact(folder/'stderr.log'),
            fallback_lines=fallback,qualified=result['status']=='qualified_inference_only' and not fallback))
        c.require(not fallback or result['status']=='qualification_failed','Native fallback invalidates GPU qualification')
        c.save(c.OUT/'completion.json',dict(status='worker_completed_inputs_preserved',time=c.now(),plan=c.artifact(c.PLAN),
            result=c.artifact(folder/'results/worker_result.json'),execution=c.artifact(c.OUT/'execution.json'),
            worker_status=result['status'],games=0,training_updates=0,automatic_retries=0,inputs_unchanged=True))
        return 0
    except BaseException:
        c.save(c.OUT/'failure.json',dict(status='postrun_verification_failed',time=c.now(),error=traceback.format_exc()));return 1

def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['freeze','run']);p.add_argument('--tests');p.add_argument('--reviews',nargs='*');p.add_argument('--plan-sha256');a=p.parse_args()
    if a.operation=='freeze':print(c.canonical(freeze(Path(a.tests),[Path(x) for x in a.reviews])));return 0
    return run(a.plan_sha256)

if __name__=='__main__':raise SystemExit(main())
