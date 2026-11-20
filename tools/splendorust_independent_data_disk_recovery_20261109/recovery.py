"""Continue only untouched decisions after an authenticated disk-stop boundary."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import psutil
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent

BOUNDARY = c.ROOT/'outputs/splendorust_independent_data_disk_recovery_preparation_20261109_01/boundary_diagnostic.json'
BOUNDARY_SHA = '3b5f8d4ffda835d7be7c7a7b72763ff81a041d51c69551af57af08f5cf31ddb9'


def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    sources=python_dependencies((c.ROOT/c.PACKAGE.replace('.','/')).rglob('*.py'))
    from .preflight import test_paths
    return [c.artifact(p) for p in sources+test_paths()]


def validate_test_receipt(proof):
    from .preflight import complete_summary,test_paths
    c.require(proof.get('schema')=='independent_data_disk_recovery_complete_pytest_v1' and proof.get('full_suite') is True,
              'Complete separate disk-recovery suite')
    complete_summary(proof['suite_summary'])
    for key in ('exit_code','observed_process_exit_code','models','games'):
        c.require(type(proof[key]) is int and proof[key]==0,'Strict successful model-free process accounting')
    c.require(proof['runner']==c.artifact(c.ROOT/c.PACKAGE.replace('.','/')/'preflight.py') and
              proof['test_files']==[c.artifact(p) for p in test_paths()], 'Exact complete test entry point and files')
    return proof


def expected_source_executions():
    names=[f'{mode}_r{r}' for mode in ('qualification','operational') for r in (0,1)]
    names += [f'main_block_r{r}_g{b:03d}' for b in range(0,150,10) for r in (0,1)]
    names += ['main_block_r0_g150']
    return [c.LIVE_SOURCE_OUT/'segments'/name/'execution.json' for name in names]


def check_source_boundary(boundary=lambda:None):
    from tools.splendorust_independent_data_recovery_20261109.recovery import validate_reuse as original_reuse
    original_reuse(c.read(c.LIVE_SOURCE_OUT/'reuse.json'),boundary=boundary)
    c.require(c.sha(c.LIVE_SOURCE_PLAN)==c.LIVE_SOURCE_SHA and c.sha(BOUNDARY)==BOUNDARY_SHA,
              'Authenticated prior frozen continuation and saved boundary diagnostic')
    old=c.read(c.LIVE_SOURCE_OUT/'claim.json');require_absent(dict(pid=old['pid'],created=old['created']))
    source=c.read(c.LIVE_SOURCE_OUT/'execution.json')
    c.require(source['recovery_claim']==c.artifact(c.LIVE_SOURCE_OUT/'claim.json') and
              source['original_claim']==c.artifact(c.SOURCE_OUT/'claim.json') and old['freeze']==c.artifact(c.LIVE_SOURCE_PLAN),
              'Prior actual execution authenticates its owner and original study')
    original=c.read(c.SOURCE_OUT/'claim.json')
    for key in ('monotonic_origin','unix_origin','total_seconds','model_seconds'):
        c.require(type(old[key]) is type(original[key]) and old[key]==original[key], 'Prior owner inherited original '+key)
    c.require(source['status']=='failed' and source['segments']==[c.artifact(p) for p in expected_source_executions()],
              'Exactly35 passed live segments, with failure kept separate')
    failed=c.read(c.LIVE_SOURCE_OUT/'segments/main_block_r1_g150/execution.json')
    c.require(failed['status']=='failed' and failed['worker_exit_code']==1 and failed['cleanup_errors']==[],
              'Preserved actual failed segment exit and successful ownership cleanup')
    failure=c.read(c.LIVE_SOURCE_OUT/'segments/main_block_r1_g150/result.json')
    c.require('Free disk reserve' in failure['error'],'Exact saved resource failure, never a failed decision retry')
    identities={}
    for value in [*[c.read(p) for p in expected_source_executions()],failed]:
        boundary()
        if value is not failed:
            c.require(value['status']=='passed' and value['worker_exit_code']==0 and not value['cleanup_errors'] and value['error'] is None,
                      'Prior accepted segment actual successful exit')
        for item in value['observed_owned_identities']+[value['identity']]:identities[(item['pid'],item['created'])]=item
    for item in identities.values():boundary();require_absent(item)
    diagnostic=c.read(BOUNDARY);c.verify(list(c.nested_artifacts(diagnostic)),boundary=boundary)
    c.require(diagnostic['status']=='saved_boundary_verified' and diagnostic['checked_moves']==26 and
              diagnostic['next_unplayed_ply']==26 and diagnostic['in_flight_decision'] is False and
              diagnostic['missing_completed_decision'] is False and diagnostic['last_request_accepted'] is True and
              diagnostic['afterstate_nonterminal'] is True and diagnostic['model_calls']==diagnostic['games']==0,
              'No missing, failed or active decision at the saved continuation boundary')
    for rep,limit in ((0,160),(1,159)):
        folder=c.LIVE_SOURCE_OUT/'strength'/f'r{rep}'/'games/main'
        c.require({p.parent.name for p in folder.glob('g*/result.json')}=={f'g{i:03d}' for i in range(limit)},
                  'Exact immutable completed main-game prefix')
        c.require({p.name for p in folder.iterdir()}=={f'g{i:03d}' for i in range(160)},'No hidden additional source game')
    partial=c.LIVE_SOURCE_OUT/'strength/r1/games/main/g159'
    c.require({p.name for p in (partial/'moves').iterdir()}=={f'{i:06d}.json' for i in range(26)} and
              not (partial/'result.json').exists(),'Exactly26 saved prefix moves, no invented game outcome')
    last=c.read(c.resolve(diagnostic['last_saved_move']))
    c.require(last['completed'] is True and c.digest(last['after'])==diagnostic['last_afterstate_sha256'],
              'Exact already-completed afterstate, next ply only')
    return dict(previous_claim=c.artifact(c.LIVE_SOURCE_OUT/'claim.json'),
        previous_freeze=c.artifact(c.LIVE_SOURCE_PLAN),previous_execution=c.artifact(c.LIVE_SOURCE_OUT/'execution.json'),
        previous_failure=c.artifact(c.LIVE_SOURCE_OUT/'failure.json'),
        failed_execution=c.artifact(c.LIVE_SOURCE_OUT/'segments/main_block_r1_g150/execution.json'),
        passed_executions=source['segments'],boundary=c.artifact(BOUNDARY),source_identity_count=len(identities))


def dual_deadline():
    old=c.read(c.SOURCE_OUT/'claim.json');phase=c.read(c.SOURCE_OUT/'stages/strength_start.json')
    c.require(old['monotonic_origin']<=time.monotonic()<phase['monotonic_deadline'] and
              old['unix_origin']<=time.time()<phase['unix_deadline'],'Original remaining strength clocks')


def inventory_tree(base,path,boundary=lambda:None):
    paths=[];base=Path(base).resolve()
    for directory,dirs,files in os.walk(base,followlinks=False):
        boundary()
        for name in dirs+files:
            p=Path(directory)/name
            c.require(not (p.lstat().st_file_attributes&0x400),'No reparse point in preservation inventory')
        paths.extend(Path(directory)/name for name in files)
    paths.sort();count=total=0
    def descriptor(p):
        before=p.stat()
        with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        after=p.stat()
        c.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Immutable logical evidence')
        return dict(path=p.relative_to(c.ROOT).as_posix(),bytes=before.st_size,sha256=sha)
    with Path(path).open('xb') as stream,ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(paths),256):
            boundary();c.require(psutil.Process().memory_info().rss<c.COLLECTION_RSS,'Preservation RSS ceiling')
            for item in pool.map(descriptor,paths[start:start+256]):
                stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
            c.status(c.OUT/'preservation/progress.json',dict(time=c.now(),files=count,total_files=len(paths),bytes=total))
        stream.flush();os.fsync(stream.fileno())
    return dict(root=base.relative_to(c.ROOT).as_posix(),inventory=c.artifact(path),files=count,bytes=total)


def prepare():
    c.OUT.mkdir(parents=True,exist_ok=True);c.require(not (c.OUT/'reuse.json').exists(),'Exclusive preparation')
    dual_deadline();prior=check_source_boundary(boundary=dual_deadline)
    c.verify(c.read(c.LIVE_SOURCE_PLAN)['inputs'],boundary=dual_deadline)
    physical=inventory_tree(c.LIVE_SOURCE_OUT,c.OUT/'live_source_inventory.jsonl',dual_deadline)
    original=c.read(c.LIVE_SOURCE_OUT/'reuse.json')
    value=dict(schema='independent_data_disk_continuation_reuse_v1',**prior,
        source_collection_reuse=c.artifact(c.LIVE_SOURCE_OUT/'reuse.json'),live_inventory=physical,
        original_inventory=original['old_inventory'],original_preserved_files=original['preserved_files'],
        original_preserved_bytes=original['preserved_bytes'],preserved_bytes=original['preserved_bytes']+physical['bytes'],
        source_games=1152,completed_strength_games=319,operational_games=12,partial_repetition=1,
        partial_game=159,next_unplayed_ply=26,new_models=0,new_games=0,time=c.now())
    return c.save(c.OUT/'reuse.json',value)


def validate_reuse(value,boundary=lambda:None):
    c.require(value['schema']=='independent_data_disk_continuation_reuse_v1','Exact continuation reuse schema')
    checked=check_source_boundary(boundary)
    c.require(all(value[k]==v for k,v in checked.items()),'Exact unchanged source boundary')
    c.verify(list(c.nested_artifacts(value)),boundary=boundary)
    original=c.read(c.LIVE_SOURCE_OUT/'reuse.json')
    c.require(value['original_preserved_bytes']==original['preserved_bytes'] and
              value['original_inventory']==original['old_inventory'] and
              value['preserved_bytes']==value['original_preserved_bytes']+value['live_inventory']['bytes'],
              'Combined original logical evidence budget, no compression reset')
    return value


def validate_recovery_claim(claim):
    reuse=validate_reuse(c.read(c.OUT/'reuse.json'));old=c.read(c.SOURCE_OUT/'claim.json')
    c.require(claim['reuse']==c.artifact(c.OUT/'reuse.json') and claim['original_claim']==c.artifact(c.SOURCE_OUT/'claim.json')
              and claim['previous_claim']==reuse['previous_claim'] and claim['freeze']==c.artifact(c.PLAN)
              and claim['plan_sha256']==c.sha(c.PLAN),'Exact new owner and prior lineage')
    for key in ('monotonic_origin','unix_origin','total_seconds','model_seconds'):
        c.require(type(claim[key]) is type(old[key]) and claim[key]==old[key],'Original unextended '+key)
    c.require(claim['automatic_retries']==0,'No scientific retry')
    previous=c.read(c.resolve(reuse['previous_claim']))
    for historical in (old,previous):
        c.require((claim['pid'],claim['created'])!=(historical['pid'],historical['created']), 'Fresh exact owner identity')
    return claim


def recover_phase(claim):
    validate_recovery_claim(claim)
    descriptor=c.artifact(c.SOURCE_OUT/'stages/strength_start.json');phase=deepcopy(c.read(c.resolve(descriptor)))
    phase.update(claim=c.artifact(c.OUT/'claim.json'),original_phase=descriptor)
    from .owner import guard
    guard(claim,phase)
    return c.save(c.OUT/'stages/strength_start.json',phase)


def freeze(tests,reviews,storage):
    from .owner import contract
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(not c.PLAN.exists(),'Exclusive new freeze');reuse=validate_reuse(c.read(c.OUT/'reuse.json'))
    proof=validate_test_receipt(c.read(tests));c.verify(proof['inputs']+[proof['log']])
    authors=set()
    for path in reviews:
        review=c.read(path);c.require(review['approved'] is True and not review['blocking_findings'],'All review blockers resolved')
        c.verify(review['inputs']);authors.add(review['reviewer'])
    c.require(len(authors)>=3,'Three distinct independent review scopes')
    c.verify([c.artifact(storage)])
    evidence=c.read(storage)
    c.require(evidence.get('status')=='passed' and evidence.get('all_original_bytes_and_paths_preserved') is True,
              'Qualified logical-byte-preserving storage relief')
    c.require(type(evidence.get('observed_exit_code')) is int and evidence['observed_exit_code']==0 and
              evidence.get('identity_absent') is True,'Actual storage job exit and cleanup observed')
    old=c.read(c.LIVE_SOURCE_PLAN);value=deepcopy(old)
    c.require(c.canonical({k:old.get(k) for k in contract()})==c.canonical(contract()),'Unchanged scientific contract')
    value.update(time=c.now(),protocol=c.artifact(c.PROTOCOL),tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        corrected_source_receipt=verify_runtime_sources(),
        recovery=dict(schema='independent_data_disk_next_unplayed_move_v1',reuse=c.artifact(c.OUT/'reuse.json'),
            storage=c.artifact(storage),completed_games_reused=319,partial_game=159,partial_repetition=1,next_unplayed_ply=26,
            scientific_decisions_retried=0,replacement_games=0,original_failed_attempts_preserved=True),
        user_authorization='Resume automatically after tested separate technical recovery; saved models and original deadlines unchanged.')
    entries=old['inputs']+source_inputs()+list(c.nested_artifacts(reuse))+[c.artifact(c.LIVE_SOURCE_PLAN),
        c.artifact(c.PROTOCOL),c.artifact(c.OUT/'reuse.json'),c.artifact(storage),*c.nested_artifacts(evidence),
        *[c.artifact(c.LIVE_SOURCE_OUT/'strength'/f'r{rep}'/'freeze.json') for rep in (0,1)]]
    for path in [tests,*reviews]:entries += [c.artifact(path),*c.nested_artifacts(c.read(path))]
    unique={}
    for item in entries:
        key=str(c.resolve(item)).casefold();previous=unique.setdefault(key,item)
        c.require(previous==item,'No conflicting source binding')
    value['inputs']=sorted(unique.values(),key=lambda x:x['path']);value['worker_inputs']=value['inputs']
    c.verify(value['inputs']);c.require('torch' not in sys.modules,'Model-free freeze')
    return c.save(c.PLAN,value)


def run(expected):
    from . import owner as o
    c.require(c.sha(c.PLAN)==expected and not (c.OUT/'claim.json').exists(),'Exclusive frozen continuation')
    plan=c.read(c.PLAN);c.verify(plan['inputs']);reuse=validate_reuse(c.read(c.OUT/'reuse.json'));dual_deadline()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(verify_runtime_sources()==plan['corrected_source_receipt'],'Corrected root sources')
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed=placement.configure_execution(plan,'isolated');old=c.read(c.SOURCE_OUT/'claim.json')
    claim=dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        **{k:old[k] for k in ('monotonic_origin','unix_origin','total_seconds','model_seconds')},
        freeze=c.artifact(c.PLAN),plan_sha256=expected,automatic_retries=0,placement=placed,
        original_claim=c.artifact(c.SOURCE_OUT/'claim.json'),previous_claim=reuse['previous_claim'],reuse=c.artifact(c.OUT/'reuse.json'),
        recovery_started_monotonic=time.monotonic(),recovery_started_unix=time.time())
    c.save(c.OUT/'claim.json',claim);validate_recovery_claim(claim)
    receipts=[];status='failed';error=None
    try:
        phase=recover_phase(claim)
        for rep in (0,1):receipts.append(o.run_segment(claim,phase,expected,'prepare',rep))
        receipts.append(o.run_segment(claim,phase,expected,'resume_partial',1))
        for block in range(160,200,10):
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
        original_claim=claim['original_claim'],previous_claim=reuse['previous_claim'],recovery_claim=c.artifact(c.OUT/'claim.json'),
        reused_stages=reuse['passed_executions'],ended_monotonic=time.monotonic(),ended_unix=time.time(),automatic_promotion=False))
    c.status(c.OUT/'stage.json',dict(time=c.now(),stage=status,status=status,completed_stages=len(receipts),completed_main_games=o.completed_games()))
    if error:c.save(c.OUT/'failure.json',dict(status='incomplete',execution=execution,error=error));return 1
    c.save(c.OUT/'completion.json',dict(status=status,execution=execution,
        independent_reader=c.artifact(c.OUT/'final_audit/worker/result.json'),owner_exit_still_requires_observation=True,
        original_failed_attempts_preserved=True,no_automatic_followup=True))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','freeze','run'))
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--storage');parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.operation=='prepare':print(c.canonical(prepare()));return 0
    if args.operation=='freeze':print(c.canonical(freeze(Path(args.tests),[Path(p) for p in args.reviews],Path(args.storage))));return 0
    return run(args.freeze_sha256)
