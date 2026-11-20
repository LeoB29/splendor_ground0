"""Frozen independent workers, absolute move clocks and exact process ownership."""
import argparse,os,sys,time,subprocess,traceback,shutil
from pathlib import Path
import psutil
from . import common as c,design
from tools.splendorust_gpu_probe_20261105.control import require_absent,finish_owned_process,native_fallback_messages


class DecisionOuterDeadline(TimeoutError):
    def __init__(self,row):
        super().__init__('Actual original75s ordinary decision safeguard')
        self.row=row


def check_registry(value,identity,wall,perf):
    c.require(value['pid']==identity['pid'] and type(value['active']) is list,'Owned active registry')
    for row in value['active']:
        c.require(row['execution_policy_id']==c.EXECUTION_POLICY_ID and
                  abs(row['deadline_unix']-row['started_unix']-c.OUTER_SECONDS)<1e-5 and
                  abs(row['deadline_perf']-row['started_perf']-c.OUTER_SECONDS)<1e-5,'Unextended outer deadline')
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
    files=python_dependencies((c.ROOT/'tools/splendorust_gpu_compiled_20261105').glob('*.py'))
    files+=sorted((c.ROOT/'tests').glob('test_splendorust_gpu_compiled*20261105*.py'))
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
    c.require(mode in ('qualification','cell'), 'No new games in scheduling diagnostic')
    stage_id='qualification' if mode=='qualification' else cell['cell_id']
    seconds=plan['worker_limits'][mode+'_seconds']
    args=[sys.executable,'-B','-u','-m','tools.splendorust_gpu_compiled_20261105.worker',
          '--mode',mode,'--plan',str(c.PLAN),'--plan-sha256',expected,'--output',str(folder),
          '--claim-sha256',c.sha(c.OUT/'claim.json')]
    if mode=='cell':args+=['--cell-id',cell['cell_id']]
    if mode=='cell':
        admission='support_admission.json' if cell['phase']=='decision_qualification' else 'qualification_gate.json'
        option='--support-sha256' if cell['phase']=='decision_qualification' else '--qualification-sha256'
        args += [option,c.sha(c.OUT/admission)]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    proc=job=None;identity=None;seen={};registries=set();error=None;cleanup=[];outer=None;status='fatal';peak=0
    begun,wall=time.monotonic(),time.time()
    try:
        global_guard(True,model=True)
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(args,cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            c.save(container/'owner.json',dict(time=c.now(),identity=identity,args=args,mode=mode,stage_id=stage_id,
                monotonic_origin=begun,unix_origin=wall,maximum_seconds=seconds,plan_sha256=expected,
                route=None if cell is None else cell['route'],global_claim=c.artifact(c.OUT/'claim.json')))
            deep_at=0
            while proc.poll() is None:
                deep=time.monotonic()-deep_at>=2;global_guard(deep,model=True)
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
            elif mode == 'cell' and proc.returncode==20:status='deadline_failed'
            elif mode == 'cell' and proc.returncode==21:status='exact_mismatch'
            else:raise RuntimeError('Worker integrity/qualification failure; no continuation')
    except DecisionOuterDeadline as failure:
        outer=failure.row;error=traceback.format_exc();status='deadline_failed' if mode == 'cell' else 'fatal'
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
    # A preserved scientific rejection may carry its causal unsupported-op
    # warning into independent audit. Successful paths still forbid fallback.
    rejected=False
    result_path=folder/'worker_result.json'
    if mode=='qualification' and status=='completed' and result_path.exists():
        rejected=c.read(result_path)['status']=='qualification_rejected'
    if fallback and not rejected:
        status='fatal';error=(error or '')+'\nUnsupported native fallback: '+repr(fallback)
    result=dict(time=c.now(),stage_id=stage_id,mode=mode,cell_id=cell['cell_id'] if mode=='cell' else None,
        route=cell['route'] if cell is not None else None,
        status=status,worker_exit_code=None if proc is None else proc.returncode,identity=identity,
        observed_owned_identities=list(seen.values()),cleanup_errors=cleanup,error=error,outer_deadline_observed=outer,
        monotonic_origin=begun,unix_origin=wall,monotonic_end=time.monotonic(),unix_end=time.time(),
        maximum_seconds=seconds,peak_rss_bytes=peak,plan_sha256=expected,native_fallback_lines=fallback,retries=0)
    c.save(container/'execution.json',result)
    return result


def run(expected):
    from . import audit
    c.require(c.sha(c.PLAN)==expected,'Frozen plan identity'); plan=c.read(c.PLAN);c.verify(plan['inputs'])
    for name in ('claim.json','execution.json','completion.json','failure.json'):
        c.require(not(c.OUT/name).exists(),'No duplicate launch or implicit retry')
    psutil.Process().cpu_affinity(plan['supervisor_affinity'])
    mono,wall=time.monotonic(),time.time();seen=[];stages=[];proofs=[];error=None;status='failed'
    rejection_stage=None;qualified=False
    c.save(c.OUT/'claim.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        monotonic_origin=mono,unix_origin=wall,total_seconds=plan['total_seconds'],model_seconds=plan['model_seconds'],
        plan_sha256=expected,retries=0,execution_policy_id=c.EXECUTION_POLICY_ID))
    def guard(deep=False,model=False):
        c.require(time.monotonic()-mono<plan['total_seconds'] and time.time()-wall<plan['total_seconds'],'Actual total dual-clock cap')
        if model:
            c.require(time.monotonic()-mono<plan['model_seconds'] and time.time()-wall<plan['model_seconds'],
                      'Actual original global model cutoff')
        c.require(shutil.disk_usage(c.ROOT).free>=c.MIN_DISK,'Actual free disk floor')
        if deep:c.require(sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())<c.MAX_OUTPUT,'Actual new evidence cap')
    def stage(name):
        guard(True);c.atomic_json(c.OUT/'stage.json',dict(time=c.now(),stage=name,pid=os.getpid()))
    try:
        stage('qualification')
        execution=run_stage(plan,expected,'qualification',None,c.OUT/'qualification',guard);stages.append(execution)
        seen.extend(execution['observed_owned_identities']);c.require(execution['status']=='completed','Qualification must pass')
        proof=audit.audit_qualification(plan,c.OUT/'qualification')
        support_proof=proof
        support_artifact=c.save(c.OUT/'qualification/saved_audit.json',proof);proofs.append(support_artifact)
        support=dict(audit.support_admission(proof),plan_sha256=expected,source_audits=[support_artifact])
        c.save(c.OUT/'support_admission.json',support)
        if not support['passed']:
            rejection_stage='qualification'
        else:
            spec=plan['qualification_cell'];stage(spec['cell_id']);container=c.OUT/'cells'/spec['cell_id']
            execution=run_stage(plan,expected,'cell',spec,container,guard);stages.append(execution)
            seen.extend(execution['observed_owned_identities'])
            c.require(execution['status']!='fatal','Technical decision qualification failure stops study')
            decision_proof=audit.audit_cell(plan,spec,container/'worker')
            decision_artifact=c.save(container/'saved_audit.json',decision_proof);proofs.append(decision_artifact)
            gate=dict(audit.qualification_gate(support_proof,decision_proof),
                      plan_sha256=expected,source_audits=[support_artifact,decision_artifact])
            c.save(c.OUT/'qualification_gate.json',gate);qualified=gate['passed']
            if not qualified:rejection_stage=spec['cell_id']
            else:
                timing=[]
                for spec in plan['cells']:
                    stage(spec['cell_id']);container=c.OUT/'cells'/spec['cell_id']
                    execution=run_stage(plan,expected,'cell',spec,container,guard);stages.append(execution)
                    seen.extend(execution['observed_owned_identities'])
                    c.require(execution['status']!='fatal','Technical/integrity timing cell failure stops study')
                    proof=audit.audit_cell(plan,spec,container/'worker');timing.append(proof)
                    proofs.append(c.save(container/'saved_audit.json',proof))
                    if execution['status']!='completed':
                        rejection_stage=spec['cell_id'];break
                if rejection_stage is None:
                    c.save(c.OUT/'comparison.json',audit.timing_summary(timing))
        guard(True);c.verify(plan['inputs'])
        status=('compiled_study_negative_pending_independent_acceptance' if rejection_stage is not None
                else 'completed_compiled_comparison_pending_independent_acceptance')
    except BaseException:
        error=traceback.format_exc();status='failed'
    cleanup=[]
    for identity in seen:
        try:require_absent(identity)
        except BaseException as failure:cleanup.append(repr(failure))
    if cleanup:status='failed'
    result=dict(time=c.now(),status=status,error=error,
        stages=[dict(stage_id=x['stage_id'],status=x['status'],worker_exit_code=x['worker_exit_code']) for x in stages],
        observed_owned_identities=seen,monotonic_origin=mono,unix_origin=wall,monotonic_end=time.monotonic(),unix_end=time.time(),
        plan_sha256=expected,cleanup_errors=cleanup,retries=0,model_seconds=plan['model_seconds'],
        game_routes_started=0,new_games=0,training_updates=0,execution_policy_id=c.EXECUTION_POLICY_ID,
        rejection_stage=rejection_stage,qualified=qualified)
    c.save(c.OUT/'execution.json',result)
    if status=='failed':c.save(c.OUT/'failure.json',result);return 1
    c.save(c.OUT/'completion.json',dict(time=c.now(),status=status,execution=c.artifact(c.OUT/'execution.json'),
        plan=c.artifact(c.PLAN),audited_stages=len(proofs),qualified=qualified,stage_audits=proofs,
        rejection_stage=rejection_stage,
        new_games=0,training_updates=0,inputs_unchanged=True))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','run'])
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.operation=='freeze':
        print(c.canonical(freeze(Path(args.tests),[Path(p) for p in args.reviews])));return 0
    return run(args.plan_sha256)

if __name__=='__main__':raise SystemExit(main())
