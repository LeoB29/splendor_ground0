"""Explicit continuation of immutable science after a pre-decision controller failure."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback
import psutil
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent


def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    sources=python_dependencies((c.ROOT/'tools/splendorust_independent_data_recovery_20261109').rglob('*.py'))
    tests=sorted(set((c.ROOT/'tests').glob('*independent_data*20261107*.py')) |
                 set((c.ROOT/'tests').glob('*independent_data*recovery*20261109*.py')))
    return [c.artifact(p) for p in sources+tests]


def validate_test_receipt(proof):
    from .preflight import complete_summary,test_paths
    c.require(proof.get('schema')=='independent_data_recovery_complete_pytest_v1' and
              proof.get('full_suite') is True,'Complete recovery pytest proof')
    s=proof['suite_summary']
    complete_summary(s)
    c.require(type(proof.get('observed_process_exit_code')) is int and proof['observed_process_exit_code']==0,
              'Observed complete test process exit')
    c.require(all(type(proof[k]) is int and proof[k]==0 for k in ('models','games')),
              'Strict outer model-free accounting')
    c.require(type(proof['exit_code']) is int and proof['exit_code']==0 and
              type(s['tests_passed']) is int and type(s['tests_collected']) is int and
              s['tests_passed']==s['tests_collected']>=284 and s['failures']==s['skipped']==0 and
              s['tensor_runtime_loaded'] is False and s['models']==s['games']==proof['models']==proof['games']==0,
              'All source-bound model-free tests pass without skipped cases')
    c.require(proof['runner']==c.artifact(c.ROOT/'tools/splendorust_independent_data_recovery_20261109/preflight.py'),
              'Exact recovery runner')
    c.require(any(x['path']=='tests/test_independent_data_recovery_20261109.py' for x in proof['inputs']),
              'Recovery counterexamples included')
    c.require(proof['test_files']==[c.artifact(p) for p in test_paths()],'Exact complete test file list')
    return proof


def expected_source_executions():
    names=['collection/qualification','collection/qualification_audit','training_qualification']
    for index in range(72):names += [f'collection/cells/c{index:03d}',f'collection/cell_audits/c{index:03d}']
    names += ['collection/corpus_audit','selection_build']
    names += ['fits/'+x['fit_id'] for x in c.fits()]
    names += ['commit','offline']
    return [c.SOURCE_OUT/name/'execution.json' for name in names]


def validate_reuse(value, *, boundary=lambda:None):
    c.require(value['schema']=='independent_data_preserved_prefix_v1' and value['original_freeze']==c.artifact(c.SOURCE_PLAN)
              and value['original_freeze']['sha256']==c.SOURCE_FREEZE_SHA,'Exact original frozen attempt')
    c.verify([value['original_claim'],value['original_strength_phase'],value['failed_execution'],
              value['original_execution'],value['old_inventory'],*value['passed_executions']],boundary=boundary)
    old=c.read(c.resolve(value['original_claim']))
    c.require(value['original_claim']==c.artifact(c.SOURCE_OUT/'claim.json') and old['freeze']==value['original_freeze'],
              'Original owner and freeze binding')
    require_absent(dict(pid=old['pid'],created=old['created']))
    execution=c.read(c.resolve(value['original_execution']))
    c.require(execution['status']=='failed' and execution['segments']==value['passed_executions'] and
              len(value['passed_executions'])==155,'Exact completed prefix without dropping stages')
    c.require([c.resolve(x) for x in value['passed_executions']]==
              expected_source_executions(),'Exact ordered original stage roles')
    identities={}
    for item in value['passed_executions']:
        boundary();r=c.read(c.resolve(item))
        c.require(r['status']=='passed' and r['worker_exit_code']==0 and r['error'] is None and r['cleanup_errors']==[],
                  'Every reused stage actually exited cleanly')
        for ident in r['observed_owned_identities']+[r['identity']]: identities[(ident['pid'],ident['created'])]=ident
    failed=c.read(c.resolve(value['failed_execution']))
    c.require(value['failed_execution']==c.artifact(c.SOURCE_OUT/'segments/qualification_r0/execution.json') and
              failed['status']=='failed' and failed['worker_exit_code']==1 and failed['cleanup_errors']==[],
              'Preserved failed qualification exit')
    for ident in failed['observed_owned_identities']+[failed['identity']]:identities[(ident['pid'],ident['created'])]=ident
    for ident in identities.values():boundary();require_absent(ident)
    c.require(not list((c.SOURCE_OUT/'strength').glob('r*/workers/*')) and
              not list((c.SOURCE_OUT/'strength').glob('r*/games/*/g*')) and
              not list((c.SOURCE_OUT/'strength').glob('r*/segments/*/start.json')),
              'No strength decision, active clock or game can be retried')
    clock=c.read(c.resolve(value['original_strength_phase']))
    c.require(value['original_strength_phase']==c.artifact(c.SOURCE_OUT/'stages/strength_start.json') and
              clock['stage']=='strength' and clock['claim']==value['original_claim'] and
              clock['cap_seconds']==c.STAGES['strength'],'Exact original strength phase')
    for origin,begin,end in [('monotonic_origin','monotonic','monotonic_deadline'),('unix_origin','unix_time','unix_deadline')]:
        c.require(clock[end]==min(clock[begin]+c.STAGES['strength'],old[origin]+c.MODEL_SECONDS),
                  'Original strength deadline never enlarged')
    return value


def validate_recovery_claim(claim):
    reuse=validate_reuse(c.read(c.OUT/'reuse.json'))
    old=c.read(c.resolve(reuse['original_claim']))
    c.require(claim['original_claim']==reuse['original_claim'] and claim['reuse']==c.artifact(c.OUT/'reuse.json') and
              claim['freeze']==c.artifact(c.PLAN) and claim['plan_sha256']==c.sha(c.PLAN), 'Authenticated recovery lineage')
    c.require(claim['pid']!=old['pid'] or claim['created']!=old['created'],'New exact owner, never fabricated old liveness')
    for key in ('monotonic_origin','unix_origin','total_seconds','model_seconds'):
        c.require(type(claim[key]) is type(old[key]) and claim[key]==old[key],'Inherited unchanged origin/allowance: '+key)
    c.require(claim['automatic_retries']==0,'No retry of a scientific decision')
    return claim


def recover_phase(claim):
    validate_recovery_claim(claim)
    reuse=c.read(c.OUT/'reuse.json');original=c.read(c.resolve(reuse['original_strength_phase']))
    phase=deepcopy(original)
    phase.update(claim=c.artifact(c.OUT/'claim.json'),original_phase=reuse['original_strength_phase'])
    from .owner import guard
    guard(claim,phase)
    return c.save(c.OUT/'stages/strength_start.json',phase)


def prepare():
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(not (c.OUT/'reuse.json').exists(),'Exclusive recovery preparation')
    c.require(c.sha(c.SOURCE_PLAN)==c.SOURCE_FREEZE_SHA,'Original freeze preserved')
    origin=c.read(c.SOURCE_OUT/'claim.json');require_absent(dict(pid=origin['pid'],created=origin['created']))
    oldexec=c.read(c.SOURCE_OUT/'execution.json')
    # Physical inventory records every original byte; nothing in the old attempt is edited.
    inventory=c.OUT/'original_inventory.jsonl';count=total=0
    deadline=min(origin['unix_origin']+c.MODEL_SECONDS,c.read(c.SOURCE_OUT/'stages/strength_start.json')['unix_deadline'])
    source=c.SOURCE_OUT.resolve();paths=[]
    for directory,dirs,files in os.walk(source,followlinks=False):
        for name in dirs+files:
            path=Path(directory)/name
            c.require(not (path.lstat().st_file_attributes & 0x400),'No preservation reparse point')
        paths.extend(Path(directory)/name for name in files)
    paths.sort()
    def descriptor(p):
        c.require(time.time()<deadline,'Original remaining clock during preservation')
        before=p.stat()
        with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        after=p.stat()
        c.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Immutable old evidence')
        return dict(path=p.relative_to(c.ROOT).as_posix(),bytes=before.st_size,sha256=sha)
    # Bound queued read work; avoid resolving the same canonical Windows path repeatedly.
    with inventory.open('xb') as stream, ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(paths),256):
            c.require(psutil.Process().memory_info().rss<c.COLLECTION_RSS,'Preservation memory bound')
            for item in pool.map(descriptor,paths[start:start+256]):
                stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
            c.status(c.OUT/'preservation/progress.json',dict(time=c.now(),files=count,total_files=len(paths),bytes=total))
        stream.flush();os.fsync(stream.fileno())
    value=dict(schema='independent_data_preserved_prefix_v1',original_freeze=c.artifact(c.SOURCE_PLAN),
        original_claim=c.artifact(c.SOURCE_OUT/'claim.json'),original_strength_phase=c.artifact(c.SOURCE_OUT/'stages/strength_start.json'),
        original_execution=c.artifact(c.SOURCE_OUT/'execution.json'),
        failed_execution=c.artifact(c.SOURCE_OUT/'segments/qualification_r0/execution.json'),
        passed_executions=oldexec['segments'],old_inventory=c.artifact(inventory),preserved_files=count,preserved_bytes=total,
        corpus_acceptance=c.artifact(c.SOURCE_OUT/'collection/corpus_acceptance.json'),
        all_fits_committed=c.artifact(c.SOURCE_OUT/'all_fits_committed.json'),
        offline_evaluation=c.artifact(c.SOURCE_OUT/'offline/worker/evaluation_result.json'),
        source_selection=c.artifact(c.SELECTION),new_model_calls=0,new_games=0,time=c.now())
    validate_reuse(value)
    return c.save(c.OUT/'reuse.json',value)


def freeze(tests,reviews):
    from .owner import contract
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(not c.PLAN.exists(),'Exclusive recovery freeze')
    reused=validate_reuse(c.read(c.OUT/'reuse.json'))
    proof=validate_test_receipt(c.read(tests));c.verify(proof['inputs']+[proof['log']])
    c.require(len(reviews)>=3,'Three independent recovery reviews')
    authors=set()
    for path in reviews:
        review=c.read(path);c.require(review['approved'] is True and not review['blocking_findings'],'Review blockers resolved')
        c.verify(review['inputs']);authors.add(review['reviewer'])
    c.require(len(authors)>=3,'Distinct review scopes')
    old=c.read(c.SOURCE_PLAN);value=deepcopy(old)
    c.require(c.canonical({k:old.get(k) for k in contract()})==c.canonical(contract()),'Scientific contract unchanged')
    code=source_inputs()
    value.update(time=c.now(),protocol=c.artifact(c.PROTOCOL),tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        recovery=dict(schema='independent_data_predecision_recovery_v1',reuse=c.artifact(c.OUT/'reuse.json'),
            original_freeze=reused['original_freeze'],original_claim=reused['original_claim'],
            original_strength_phase=reused['original_strength_phase'],original_failed_attempt_preserved=True,
            scientific_decisions_retried=0,models_refitted=0,predictions_regenerated=0),
        user_authorization='Implement and qualify separate recovery; continue saved models under original deadlines automatically.',
        corrected_source_receipt=verify_runtime_sources())
    entries=old['inputs']+code+list(c.nested_artifacts(reused))+[c.artifact(c.PROTOCOL),c.artifact(c.OUT/'reuse.json')]
    for path in [tests,*reviews]:entries += [c.artifact(path),*c.nested_artifacts(c.read(path))]
    unique={}
    for item in entries:
        key=str(c.resolve(item)).casefold();prior=unique.setdefault(key,item)
        c.require(prior['bytes']==item['bytes'] and prior['sha256']==item['sha256'],'Conflicting freeze input')
    value['inputs']=sorted(unique.values(),key=lambda x:x['path'])
    value['worker_inputs']=value['inputs']
    c.verify(value['inputs']);c.require('torch' not in sys.modules,'Model-free freeze')
    return c.save(c.PLAN,value)


def run(expected):
    from . import owner as o
    c.require(c.sha(c.PLAN)==expected and not (c.OUT/'claim.json').exists(),'Exclusive frozen recovery launch')
    plan=c.read(c.PLAN);c.verify(plan['inputs']);reuse=validate_reuse(c.read(c.OUT/'reuse.json'))
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(verify_runtime_sources()==plan['corrected_source_receipt'],'Root corrected sources')
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed=placement.configure_execution(plan,'isolated');old=c.read(c.resolve(reuse['original_claim']))
    claim=dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        **{k:old[k] for k in ('monotonic_origin','unix_origin','total_seconds','model_seconds')},
        freeze=c.artifact(c.PLAN),plan_sha256=expected,automatic_retries=0,placement=placed,
        original_claim=reuse['original_claim'],reuse=c.artifact(c.OUT/'reuse.json'),
        recovery_started_monotonic=time.monotonic(),recovery_started_unix=time.time())
    c.save(c.OUT/'claim.json',claim);validate_recovery_claim(claim)
    receipts=[];status='failed';error=None
    try:
        phase=recover_phase(claim)
        # Original immutable first strength clock is already running; it is never restarted.
        for mode in ('qualification','operational'):
            for rep in (0,1):receipts.append(o.run_segment(claim,phase,expected,mode,rep))
            c.save(c.OUT/f'strength_{mode}_gate.json',dict(status='passed',both_repetitions=True,phase=phase,actual_exits=receipts[-2:]))
        for block in range(0,200,10):
            for rep in (0,1):receipts.append(o.run_segment(claim,phase,expected,'main_block',rep,block))
        o.end_phase(claim,phase);phase=o.begin_phase(claim,'strength_replay')
        for rep in (0,1):receipts.append(o.run_segment(claim,phase,expected,'replay',rep))
        o.end_phase(claim,phase);phase=o.begin_phase(claim,'final')
        for rep in (0,1):receipts.append(o.run_segment(claim,phase,expected,'saved',rep))
        receipts.append(o.run_segment(claim,phase,expected,'aggregate',0))
        for mode,folder in [('audit','training_audit'),('analysis','offline_analysis')]:
            receipts.append(o.run_job(claim,phase,expected,c.PACKAGE+'.saved_worker',mode,c.OUT/folder,
                seconds=12*3600,rss=c.COLLECTION_RSS if mode=='audit' else c.FIT_RSS,model=False))
        final=c.read(c.resolve(phase));mono,wall=time.monotonic(),time.time()
        reader=c.save(c.OUT/'reader_clock.json',dict(monotonic=mono,unix_time=wall,cap_seconds=c.INDEPENDENT_REVIEW_SECONDS,
            monotonic_deadline=min(mono+c.INDEPENDENT_REVIEW_SECONDS,final['monotonic_deadline']),
            unix_deadline=min(wall+c.INDEPENDENT_REVIEW_SECONDS,final['unix_deadline']),original_phase=phase,claim=c.artifact(c.OUT/'claim.json')))
        for rep in (0,1):receipts.append(o.run_segment(claim,phase,expected,'independent',rep,reader_clock=reader))
        receipts.append(o.run_segment(claim,phase,expected,'reader',0,reader_clock=reader))
        receipts.append(o.run_job(claim,phase,expected,c.PACKAGE+'.saved_worker','final',c.OUT/'final_audit',
            seconds=12*3600,rss=c.COLLECTION_RSS,model=False,cell_deadline=c.read(c.resolve(reader))))
        c.verify(plan['inputs']);o.end_phase(claim,phase);status='verified_pending_actual_owner_exit'
    except BaseException:error=traceback.format_exc()
    execution=c.save(c.OUT/'execution.json',dict(status=status,freeze_sha256=expected,segments=receipts,error=error,
        original_claim=reuse['original_claim'],recovery_claim=c.artifact(c.OUT/'claim.json'),reused_stages=reuse['passed_executions'],
        ended_monotonic=time.monotonic(),ended_unix=time.time(),automatic_promotion=False))
    c.status(c.OUT/'stage.json',dict(time=c.now(),stage=status,status=status,completed_stages=len(receipts),completed_main_games=o.completed_games()))
    if error:c.save(c.OUT/'failure.json',dict(status='incomplete',execution=execution,error=error));return 1
    c.save(c.OUT/'completion.json',dict(status=status,execution=execution,
        independent_reader=c.artifact(c.OUT/'final_audit/worker/result.json'),owner_exit_still_requires_observation=True,
        original_failed_attempt_preserved=True,no_automatic_followup=True))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','freeze','run'))
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.operation=='prepare':print(c.canonical(prepare()));return 0
    if args.operation=='freeze':print(c.canonical(freeze(Path(args.tests),[Path(p) for p in args.reviews])));return 0
    return run(args.freeze_sha256)
