"""Reproduce saved analysis and render in a fresh process that never imports Torch."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import psutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'outputs/splendorust_learning_diagnostic_20261106_01'
OUT = ROOT/'outputs/splendorust_learning_finalization_20261106_01'
PLAN = OUT/'plan.json'
PROTOCOL = ROOT/'docs/engineering/splendorust_learning_finalization_20261106.md'
SOURCE_PLAN_SHA = 'ab8a82f8e0f0f8d0dbd06335ed598ccc1e1b762fe63dc0a3c238aa3b2af1d185'
ANALYSIS_SOURCE = ROOT/'tools/splendorust_learning_diagnostic_20261106/analysis.py'
CAP = 3600
MAX_RSS = 4*1024**3
MAX_NEW_BYTES = 512*1024**2
MIN_DISK = 1024**3


def require(value, message):
    if not value: raise ValueError('LEARNING_FINALIZATION: '+message)


def now(): return datetime.now(timezone.utc).isoformat()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def artifact(path):
    path=Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),bytes=path.stat().st_size,sha256=sha(path))
def resolve(item): return (ROOT/item['path']).resolve()
def verify(items,boundary=lambda:None):
    for item in items:
        boundary();path=resolve(item)
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Changed preserved artifact: '+item['path'])
def save(path,value):
    path=Path(path).resolve();require(path.is_relative_to(OUT),'New output root only')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    return artifact(path)
def nested(value):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=set(value):yield {k:value[k] for k in ('path','bytes','sha256')}
        else:
            for v in value.values():yield from nested(v)
    elif isinstance(value,list):
        for v in value:yield from nested(v)
def unique(items):
    result={}
    for item in items:
        key=str(resolve(item)).casefold();old=result.setdefault(key,item)
        require((old['bytes'],old['sha256'])==(item['bytes'],item['sha256']),'Conflicting original binding')
    return sorted(result.values(),key=lambda x:x['path'])
def absent(identity):
    try:require(psutil.Process(identity['pid']).create_time()!=identity['created'],'Original exact process remains alive')
    except psutil.NoSuchProcess:pass


def scientific_receipts():
    require(sha(SOURCE/'plan.json')==SOURCE_PLAN_SHA,'Exact original frozen study')
    plan=read(SOURCE/'plan.json');audit=read(SOURCE/'audit/worker/saved_evidence.json')
    failed=read(SOURCE/'execution.json');audit_exit=read(SOURCE/'audit/execution.json');claim=read(SOURCE/'claim.json')
    require(failed['status']=='failed' and failed['retries']==0 and audit_exit['worker_exit_code']==3 and
            audit_exit['status']=='failed' and not audit_exit['cleanup_errors'],'Original finalization failure retained')
    require(audit['status']=='saved_evidence_verified_pending_independent_acceptance' and
            audit['plan']==artifact(SOURCE/'plan.json') and audit['inputs']==plan['inputs'] and
            len(audit['stages'])==17 and len(audit['fits'])==len(audit['source_bindings'])==8 and
            audit['total_verified_optimizer_calls']==5354 and audit['verified_per_loss_gradient_probes']==144 and
            audit['new_model_instances']==audit['network_forward_calls']==audit['optimizer_calls']==audit['new_games']==0,
            'Completed original scientific verification receipt')
    require({x['fit_id'] for x in audit['fits']}=={x['fit_id'] for x in plan['fits']} and
            all(x['status']=='fit_saved_verified' and x['optimizer_calls']==668 and x['replay_optimizer_calls']==1 for x in audit['fits']),
            'All eight fixed fits and exact final-step reproductions')
    require(len(failed['stages'])==17 and sum(x['mode']=='fit' for x in failed['stages'])==8 and
            sum(x['mode']=='diagnostics' for x in failed['stages'])==8 and
            all(x['status']=='passed' and x['worker_exit_code']==0 and not x['cleanup_errors']
                and x['plan_sha256']==SOURCE_PLAN_SHA for x in failed['stages']), 'All scientific worker exits passed')
    for stage in audit['stages']:
        verify([stage['owner'],stage['execution'],stage['result']])
        actual=read(resolve(stage['execution']));result=read(resolve(stage['result']))
        require(actual['status']==result['status']=='passed' and actual['worker_exit_code']==0 and
                actual['plan_sha256']==result['plan_sha256']==SOURCE_PLAN_SHA,'Passed receipts retain exact original identity')
    identities=[dict(pid=claim['pid'],created=claim['created'])]
    for stage in [*failed['stages'],audit_exit]:
        identities += [stage['identity'],*stage['observed_owned_identities']]
    identities=list({(x['pid'],x['created']):x for x in identities}.values())
    require(len(identities)==37,'Complete original identity inventory')
    for item in identities:absent(item)
    return plan,audit,identities


def evidence_closure(seeds):
    """Follow current-study receipts only; historical inputs are verified, not expanded."""
    pending=list(seeds); found={}
    while pending:
        item=pending.pop();key=str(resolve(item)).casefold()
        if key in found:
            require(found[key]['sha256']==item['sha256'] and found[key]['bytes']==item['bytes'],'Conflicting receipt link')
            continue
        verify([item]);found[key]=item;path=resolve(item)
        if path.is_relative_to(SOURCE) and path.suffix=='.json':pending.extend(nested(read(path)))
    return sorted(found.values(),key=lambda x:x['path'])


def freeze(test_path,review_paths):
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    require(not PLAN.exists(),'One separate finalization freeze')
    original,audit,identities=scientific_receipts()
    tests=read(test_path);require(tests['exit_code']==0 and tests['models']==tests['games']==0,'Passed model-free recovery tests')
    verify(tests['inputs']+[tests['log']]);require(len(review_paths)>=2,'Independent scientific and implementation review')
    for path in review_paths:
        review=read(path);require(review['approved'] and not review['blocking_findings'],'Approved bounded recovery');verify(review['inputs'])
    seeds=[artifact(SOURCE/name) for name in ('plan.json','claim.json','execution.json','failure.json','audit/owner.json',
        'audit/execution.json','audit/worker/stderr.log','audit/worker/saved_evidence.json','report/analysis.json')]
    linked=evidence_closure(seeds)
    # Preserve unreferenced failures, stdout/stderr and mutable status as observed after exit.
    inventory=[artifact(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()]
    code=[artifact(p) for p in python_dependencies((ROOT/'tools/splendorust_learning_finalization_20261106').glob('*.py'))]
    code += [artifact(ANALYSIS_SOURCE),artifact(ROOT/'tools/splendorust_learning_diagnostic_20261106/common.py')]
    code += [artifact(p) for p in sorted((ROOT/'tests').glob('test_splendorust_learning_finalization_20261106.py'))]
    inputs=code+linked+inventory+[artifact(PROTOCOL),artifact(test_path),*[artifact(p) for p in review_paths]]
    for path in [test_path,*review_paths]:inputs.extend(nested(read(path)))
    require('torch' not in sys.modules,'No tensor runtime in finalization preparation')
    claim=read(SOURCE/'claim.json');owner=read(SOURCE/'audit/owner.json')
    require(original['total_seconds']==43200 and original['audit_seconds']==owner['maximum_seconds']==14400,
            'Original overall and audit windows are unchanged')
    plan=dict(schema='saved_learning_finalization_v1',time=now(),protocol=artifact(PROTOCOL),original_plan=artifact(SOURCE/'plan.json'),
        original_saved_audit=artifact(SOURCE/'audit/worker/saved_evidence.json'),original_analysis=artifact(SOURCE/'report/analysis.json'),
        original_failure=artifact(SOURCE/'failure.json'),source=artifact(ANALYSIS_SOURCE),
        source_plan_sha256=SOURCE_PLAN_SHA,original_owner_identities=identities,
        original_deadline_monotonic=claim['monotonic_origin']+original['total_seconds'],
        original_deadline_unix=claim['unix_origin']+original['total_seconds'],
        original_audit_deadline_monotonic=owner['monotonic_origin']+owner['maximum_seconds'],
        original_audit_deadline_unix=owner['unix_origin']+owner['maximum_seconds'],
        maximum_seconds=CAP,maximum_rss_bytes=MAX_RSS,maximum_new_evidence_bytes=MAX_NEW_BYTES,
        minimum_free_disk_bytes=MIN_DISK,available_ram_reserve_bytes=0,resource_forecasts_may_stop=False,
        new_models=0,new_games=0,new_optimizer_calls=0,automatic_retries=0,scientific_changes=False,
        private_module_change='Only explicit output facade; exact frozen analysis code and numerical protocol retained',
        no_torch_import=True,no_duplicate_openmp_override=True,exact_saved_analysis_required=True,
        scientific_verification_reused=True,original_attempt_remains_failed=True,
        original_input_count=len(original['inputs']),receipt_closure_count=len(linked),original_inventory_count=len(inventory),
        inputs=unique(inputs),code_inputs=code,tests=artifact(test_path),reviews=[artifact(p) for p in review_paths])
    return save(PLAN,plan)


def check_deadlines(plan,claim,mono,wall):
    require(claim['monotonic_origin']<=mono<min(claim['monotonic_origin']+CAP,plan['original_deadline_monotonic'],plan['original_audit_deadline_monotonic']) and
            claim['unix_origin']<=wall<min(claim['unix_origin']+CAP,plan['original_deadline_unix'],plan['original_audit_deadline_unix']),
            'Actual new and unextended original deadlines')


def no_torch():
    require(not any(x=='torch' or x.startswith('torch.') for x in sys.modules),'Finalization must not import Torch')
    require(os.environ.get('KMP_DUPLICATE_LIB_OK','').lower() not in ('true','1','yes'),'No duplicate-runtime bypass')


def private_analysis(original_value):
    verify([read(PLAN)['source']])
    name='tools.splendorust_learning_diagnostic_20261106._saved_finalization_analysis'
    spec=importlib.util.spec_from_file_location(name,ANALYSIS_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_common=module.c
    def checked_save(path,value):
        require(Path(path).resolve()==(OUT/'report/analysis.json').resolve(),'Only the expected analysis JSON write')
        require(value==original_value,'Independent saved prediction analysis exactly equals original')
        return save(path,value)
    module.c=SimpleNamespace(ROOT=ROOT,OUT=OUT,read=read,resolve=resolve,verify=verify,artifact=artifact,
        save=checked_save,digest=digest,canonical=canonical,fits=original_common.fits)
    require(original_common.OUT==SOURCE,'Original output facade remains unchanged')
    return module


def worker(expected):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    require(sha(PLAN)==expected,'Exact recovery freeze');plan=read(PLAN);claim=read(OUT/'claim.json')
    handshake=time.monotonic();owner_path=OUT/'worker_owner.json'
    while not owner_path.exists():
        require(time.monotonic()-handshake<10,'Bounded job handshake');time.sleep(.01)
    owner=read(owner_path);me=dict(pid=os.getpid(),created=psutil.Process().create_time())
    require(owner['identity']==me and owner['windows_job_assigned'] and owner['claim']==artifact(OUT/'claim.json') and
            claim['plan']==artifact(PLAN),'Exact worker owner and original recovery claim')
    calls=0;last=0.
    def boundary():
        nonlocal calls,last
        calls+=1;mono=time.monotonic();check_deadlines(plan,claim,mono,time.time());no_torch()
        if mono-last>5:
            last=mono;p=psutil.Process(claim['pid']);require(p.create_time()==claim['created'],'Exact retained owner alive')
            require(psutil.Process().memory_info().rss<MAX_RSS,'Actual worker RSS cap')
            require(shutil.disk_usage(ROOT).free>=MIN_DISK,'Actual disk floor')
    status='failed';result=None;error=None
    try:
        boundary();save(OUT/'worker/runtime_sources.json',verify_runtime_sources());verify(plan['inputs'],boundary)
        original,_,identities=scientific_receipts()
        original_value=read(resolve(plan['original_analysis']))
        module=private_analysis(original_value)
        result=module.run_analysis(original,{s['fit_id']:SOURCE/'diagnostics'/s['fit_id']/'worker' for s in original['fits']},OUT/'report',boundary=boundary)
        require(read(OUT/'report/analysis.json')==original_value,'Exact saved analysis reproduced and preserved')
        no_torch();verify(plan['inputs'],boundary)
        for identity in identities:absent(identity)
        status='recovered_finalization_passed_pending_independent_acceptance'
    except BaseException:error=traceback.format_exc()
    receipt=dict(status=status,time=now(),result=result,error=error,plan=artifact(PLAN),identity=me,
        boundary_calls=calls,original_attempt_remains_failed=True,original_analysis_unchanged=True if result else None,
        no_torch_import=not any(x=='torch' or x.startswith('torch.') for x in sys.modules),new_models=0,new_games=0,new_optimizer_calls=0)
    save(OUT/'worker/result.json',receipt)
    if status=='failed':save(OUT/'worker/failure.json',receipt);print(error,flush=True);return 1
    print(canonical({'status':status,'new_models':0,'new_games':0}),flush=True);return 0


def run(expected):
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_gpu_probe_20261105.control import finish_owned_process
    require(sha(PLAN)==expected and not (OUT/'claim.json').exists(),'One retained recovery launch')
    plan=read(PLAN);verify(plan['code_inputs']);scientific_receipts()
    psutil.Process().cpu_affinity([2])
    claim=dict(time=now(),pid=os.getpid(),created=psutil.Process().create_time(),plan=artifact(PLAN),
        monotonic_origin=time.monotonic(),unix_origin=time.time(),automatic_retries=0)
    check_deadlines(plan,claim,time.monotonic(),time.time());save(OUT/'claim.json',claim)
    args=[sys.executable,'-B','-u','-m','tools.splendorust_learning_finalization_20261106.recovery','worker','--plan-sha256',expected]
    env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    proc=job=identity=None;cleanup=[];seen={};error=None;status='failed';peak=0;last=0.
    (OUT/'worker').mkdir(exist_ok=False)
    try:
        with (OUT/'worker/stdout.log').open('xb') as stdout,(OUT/'worker/stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            save(OUT/'worker_owner.json',dict(identity=identity,args=args,claim=artifact(OUT/'claim.json'),windows_job_assigned=True))
            while proc.poll() is None:
                mono=time.monotonic();check_deadlines(plan,claim,mono,time.time())
                if mono-last>2:
                    last=mono;rss=psutil.Process().memory_info().rss
                    try:
                        parent=psutil.Process(proc.pid);require(parent.create_time()==identity['created'],'Exact worker identity')
                        for member in [parent,*parent.children(recursive=True)]:
                            try:
                                key=(member.pid,member.create_time());seen[key]=dict(pid=key[0],created=key[1]);rss+=member.memory_info().rss
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    peak=max(peak,rss);require(rss<MAX_RSS,'Actual combined RSS cap')
                    require(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<MAX_NEW_BYTES,'Actual output cap')
                    require(shutil.disk_usage(ROOT).free>=MIN_DISK,'Actual disk floor')
                time.sleep(.25)
            check_deadlines(plan,claim,time.monotonic(),time.time())
            require(proc.returncode==0,'Finalization worker failed; preserve without retry')
            require(read(OUT/'worker/result.json')['status']=='recovered_finalization_passed_pending_independent_acceptance','Full recovery result')
            status='recovered_finalization_passed_pending_independent_acceptance'
    except BaseException:error=traceback.format_exc()
    finally:
        if job:
            try:job.close()
            except BaseException as exc:cleanup.append(repr(exc))
        if proc:
            try:finish_owned_process(proc,identity)
            except BaseException as exc:cleanup.append(repr(exc))
        for item in [*seen.values(),*([identity] if identity else [])]:
            try:absent(item)
            except BaseException as exc:cleanup.append(repr(exc))
        if cleanup:status='failed'
    receipt=dict(time=now(),status=status,error=error,cleanup_errors=cleanup,identity=identity,observed_owned_identities=list(seen.values()),
        worker_exit_code=None if proc is None else proc.returncode,plan=artifact(PLAN),
        monotonic_origin=claim['monotonic_origin'],unix_origin=claim['unix_origin'],monotonic_end=time.monotonic(),unix_end=time.time(),
        peak_rss_bytes=peak,original_attempt_remains_failed=True,new_models=0,new_games=0,new_optimizer_calls=0)
    save(OUT/'execution.json',receipt)
    if status=='failed':save(OUT/'failure.json',receipt);return 1
    save(OUT/'completion.json',dict(status=status,time=now(),execution=artifact(OUT/'execution.json'),result=artifact(OUT/'worker/result.json')))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['freeze','run','worker'])
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.operation=='freeze':print(canonical(freeze(Path(args.tests),[Path(x) for x in args.reviews])));return 0
    return run(args.plan_sha256) if args.operation=='run' else worker(args.plan_sha256)


if __name__=='__main__':raise SystemExit(main())
