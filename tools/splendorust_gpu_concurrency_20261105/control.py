"""Frozen independent workers, absolute move clocks and exact process ownership."""
import argparse,os,sys,time,subprocess,traceback,shutil
from pathlib import Path
import psutil
from . import common as c,design
from tools.splendorust_gpu_probe_20261105.control import require_absent,finish_owned_process,native_fallback_messages


class DecisionOuterDeadline(TimeoutError):
    def __init__(self,row):
        super().__init__('Actual original10s ordinary decision deadline')
        self.row=row


def check_registry(value,identity,wall,perf):
    c.require(value['pid']==identity['pid'] and type(value['active']) is list,'Owned active registry')
    for row in value['active']:
        c.require(abs(row['deadline_unix']-row['started_unix']-10.)<1e-5 and
                  abs(row['deadline_perf']-row['started_perf']-10.)<1e-5,'Unextended outer deadline')
        if wall>=row['deadline_unix'] or perf>=row['deadline_perf']:
            if row['kind']=='ordinary':raise DecisionOuterDeadline(dict(row,observed_unix=wall,observed_perf=perf))
            raise TimeoutError('Startup/warmup/nondecision outer deadline')


def freeze(tests_path,review_paths):
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    c.require(not c.PLAN.exists(),'One new prospective freeze')
    tests=c.read(tests_path);c.require(tests['exit_code']==0 and tests['models']==tests['games']==0,'Passed model-free tests')
    c.verify(tests['inputs']+[tests['log']]);c.require(bool(review_paths),'Independent review required')
    for path in review_paths:
        review=c.read(path);c.require(review['approved'] and not review['blocking_findings'],'Independent review approval')
        c.verify(review['inputs'])
    plan=design.build_plan();plan['tests']=c.artifact(tests_path);plan['reviews']=[c.artifact(p) for p in review_paths]
    files=python_dependencies((c.ROOT/'tools/splendorust_gpu_concurrency_20261105').glob('*.py'))
    files+=sorted((c.ROOT/'tests').glob('test_splendorust_gpu_concurrency*20261105*.py'))
    entries=[c.artifact(p) for p in files]+list(c.nested_artifacts(plan))
    for path in (tests_path,*review_paths):entries+=list(c.nested_artifacts(c.read(path)))
    unique={}
    for item in entries:
        previous=unique.setdefault(item['path'],item);c.require(previous==item,'Consistent frozen source identity')
    plan['inputs']=sorted(unique.values(),key=lambda x:x['path']);c.verify(plan['inputs'])
    c.require('torch' not in sys.modules,'Freeze loads no model')
    return c.save(c.PLAN,plan)


def run_stage(plan,expected,mode,cell,container,global_guard):
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_gpu_live_20261105_03.status_io import read_status_json
    container.mkdir(parents=True,exist_ok=False);folder=container/'worker';folder.mkdir()
    stage_id='qualification' if cell is None else cell['cell_id']
    seconds=plan['worker_limits']['qualification_seconds' if mode=='qualification' else 'cell_seconds']
    args=[sys.executable,'-B','-u','-m','tools.splendorust_gpu_concurrency_20261105.worker',
          '--mode',mode,'--plan',str(c.PLAN),'--plan-sha256',expected,'--output',str(folder)]
    if cell is not None:args+=['--cell-id',cell['cell_id']]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    proc=job=None;identity=None;seen={};registries=set();error=None;cleanup=[];outer=None;status='fatal';peak=0
    begun,wall=time.monotonic(),time.time()
    try:
        global_guard(True)
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(args,cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            c.save(container/'owner.json',dict(time=c.now(),identity=identity,args=args,mode=mode,stage_id=stage_id,
                monotonic_origin=begun,unix_origin=wall,maximum_seconds=seconds,plan_sha256=expected))
            deep_at=0
            while proc.poll() is None:
                deep=time.monotonic()-deep_at>=2;global_guard(deep)
                if deep:deep_at=time.monotonic()
                c.require(time.monotonic()-begun<seconds and time.time()-wall<seconds,'Actual worker dual-clock cap')
                paths=set(folder.rglob('active_decisions.json'));c.require(registries<=paths,'Active registry cannot disappear')
                registries|=paths
                for path in sorted(registries):check_registry(read_status_json(path),identity,time.time(),time.perf_counter())
                try:
                    parent=psutil.Process(proc.pid);c.require(parent.create_time()==identity['created'],'Exact worker identity')
                    members=[parent]+parent.children(recursive=True);rss=0
                    for member in members:
                        try:
                            key=(member.pid,member.create_time());seen[key]=dict(pid=key[0],created=key[1]);rss+=member.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss);c.require(rss<c.MAX_RSS,'Actual worker-tree RSS cap')
                except psutil.NoSuchProcess:pass
                time.sleep(.25)
            c.require(time.monotonic()-begun<seconds and time.time()-wall<seconds,'Worker cap at exit')
            if proc.returncode==0:status='completed'
            elif mode=='cell' and proc.returncode==20:status='deadline_failed'
            else:raise RuntimeError('Worker integrity/qualification failure; no continuation')
    except DecisionOuterDeadline as failure:
        outer=failure.row;error=traceback.format_exc();status='deadline_failed' if mode=='cell' else 'fatal'
    except BaseException:error=traceback.format_exc();status='fatal'
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
        if identity:
            try:require_absent(identity)
            except BaseException as e:cleanup.append(repr(e))
        if cleanup:status='fatal'
    fallback=native_fallback_messages(folder/'stderr.log') if (folder/'stderr.log').exists() else []
    if fallback:status='fatal';error=(error or '')+'\nUnsupported native fallback: '+repr(fallback)
    result=dict(time=c.now(),stage_id=stage_id,mode=mode,cell_id=None if cell is None else cell['cell_id'],
        status=status,worker_exit_code=None if proc is None else proc.returncode,identity=identity,
        observed_owned_identities=list(seen.values()),cleanup_errors=cleanup,error=error,outer_deadline_observed=outer,
        monotonic_origin=begun,unix_origin=wall,monotonic_end=time.monotonic(),unix_end=time.time(),
        maximum_seconds=seconds,peak_rss_bytes=peak,plan_sha256=expected,native_fallback_lines=fallback,retries=0)
    c.save(container/'execution.json',result)
    return result


def run(expected):
    from . import audit
    c.require(c.sha(c.PLAN)==expected,'Frozen plan identity');plan=c.read(c.PLAN);c.verify(plan['inputs'])
    for name in ('claim.json','execution.json','completion.json','failure.json'):
        c.require(not(c.OUT/name).exists(),'No duplicate launch or implicit retry')
    psutil.Process().cpu_affinity(plan['supervisor_affinity'])
    mono,wall=time.monotonic(),time.time();seen=[];stages=[];error=None;status='failed'
    c.save(c.OUT/'claim.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        monotonic_origin=mono,unix_origin=wall,total_seconds=plan['total_seconds'],plan_sha256=expected,retries=0))
    def guard(deep=False):
        c.require(time.monotonic()-mono<plan['total_seconds'] and time.time()-wall<plan['total_seconds'],'Actual total dual-clock cap')
        c.require(shutil.disk_usage(c.ROOT).free>=c.MIN_DISK,'Actual free disk floor')
        if deep:c.require(sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())<c.MAX_OUTPUT,'Actual new evidence cap')
    def stage(name):guard(True);c.atomic_json(c.OUT/'stage.json',dict(time=c.now(),stage=name,pid=os.getpid()))
    try:
        stage('qualification')
        q=run_stage(plan,expected,'qualification',None,c.OUT/'qualification',guard);stages.append(q)
        seen.extend(q['observed_owned_identities']);c.require(q['status']=='completed','Qualification worker must pass')
        proof=audit.audit_qualification(plan,c.OUT/'qualification');c.save(c.OUT/'qualification/saved_audit.json',proof)
        baselines={'cpu':{},'gpu':{}}
        for cell in plan['cells']:
            stage(cell['cell_id']);container=c.OUT/'cells'/cell['cell_id']
            execution=run_stage(plan,expected,'cell',cell,container,guard);stages.append(execution)
            seen.extend(execution['observed_owned_identities']);c.require(execution['status']!='fatal','Technical/integrity cell failure stops study')
            proof=audit.audit_cell(plan,cell,container/'worker',baselines=baselines[cell['route']])
            c.save(container/'saved_audit.json',proof)
            if cell['concurrency']==1:
                baselines[cell['route']]=(proof['completed_results'] if proof['status']=='passed' and
                    proof['runtime_closure']['final_weights_verified'] else {})
            audit.audit_cross_baselines(plan,baselines)
        guard(True);c.verify(plan['inputs']);status='completed_cells_pending_independent_acceptance'
    except BaseException:error=traceback.format_exc()
    cleanup=[]
    for identity in seen:
        try:require_absent(identity)
        except BaseException as failure:cleanup.append(repr(failure))
    if cleanup:status='failed'
    result=dict(time=c.now(),status=status,error=error,stages=[dict(stage_id=x['stage_id'],status=x['status'],worker_exit_code=x['worker_exit_code']) for x in stages],
        observed_owned_identities=seen,monotonic_origin=mono,unix_origin=wall,monotonic_end=time.monotonic(),unix_end=time.time(),
        plan_sha256=expected,cleanup_errors=cleanup,retries=0,new_games=0,training_updates=0)
    c.save(c.OUT/'execution.json',result)
    if status=='failed':c.save(c.OUT/'failure.json',result);return 1
    c.save(c.OUT/'completion.json',dict(time=c.now(),status=status,execution=c.artifact(c.OUT/'execution.json'),
        plan=c.artifact(c.PLAN),decision_cells=len(stages)-1,qualified=True,inputs_unchanged=True))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','run'])
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.operation=='freeze':print(c.canonical(freeze(Path(args.tests),[Path(p) for p in args.reviews])));return 0
    return run(args.plan_sha256)

if __name__=='__main__':raise SystemExit(main())
