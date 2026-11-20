"""Final saved acceptance for the original corpus and two disclosed recoveries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent


def expected_executions():
    from .owner import segment_id
    roles=[('prepare',r,None) for r in (0,1)]+[('resume_partial',1,None)]
    roles += [('main_block',r,b) for b in range(160,200,10) for r in (0,1)]
    roles += [(mode,r,None) for mode in ('replay','saved','independent') for r in (0,1)]
    roles += [('aggregate',0,None),('reader',0,None)]
    return [c.OUT/'segments'/segment_id(*role) for role in roles]+[c.OUT/'training_audit',c.OUT/'offline_analysis']


def verify_inventory(base,descriptor,expected_files,expected_bytes,boundary):
    """Stream exact logical hashes, then independently match the complete path set."""
    c.verify([descriptor],boundary=boundary);base=Path(base).resolve();prefix=base.relative_to(c.ROOT).as_posix()+'/'
    seen=set();count=total=0
    def checked(item):
        c.require(type(item) is dict and set(item)=={'path','bytes','sha256'},'Exact inventory descriptor')
        relative=item['path']
        c.require(type(relative) is str and relative.startswith(prefix) and '\\' not in relative and
                  all(x not in ('','..','.') for x in relative.split('/')),'Canonical bounded preserved path')
        path=c.ROOT/relative;before=path.stat()
        c.require(not(path.lstat().st_file_attributes&0x400),'No preserved reparse point')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        after=path.stat()
        c.require(type(item['bytes']) is int and before.st_size==after.st_size==item['bytes'] and
                  before.st_mtime_ns==after.st_mtime_ns and digest==item['sha256'],'Every preserved logical byte unchanged')
        return relative,item['bytes']
    with c.resolve(descriptor).open(encoding='utf-8') as stream,ThreadPoolExecutor(max_workers=8) as pool:
        batch=[]
        def flush(items):
            nonlocal count,total
            boundary()
            for relative,size in pool.map(checked,items):
                c.require(relative not in seen,'No duplicate preserved artifact');seen.add(relative);count+=1;total+=size
        for line in stream:
            batch.append(json.loads(line,object_pairs_hook=c._unique))
            if len(batch)==256:flush(batch);batch=[]
        if batch:flush(batch)
    actual=set()
    for directory,dirs,files in os.walk(base,followlinks=False):
        boundary()
        for name in dirs+files:
            path=Path(directory)/name;c.require(not(path.lstat().st_file_attributes&0x400),'No new reparse point')
        actual.update((Path(directory)/name).relative_to(c.ROOT).as_posix() for name in files)
    c.require(actual==seen and count==expected_files and total==expected_bytes,'Exact preserved file set and logical bytes')
    return dict(root=base.relative_to(c.ROOT).as_posix(),files=count,bytes=total,inventory=descriptor,
                all_original_bytes_preserved=True)


def final(plan,folder,boundary):
    from .recovery import validate_reuse,validate_recovery_claim
    c.verify(plan['inputs'],boundary=boundary);reuse=validate_reuse(c.read(c.OUT/'reuse.json'),boundary=boundary)
    validate_recovery_claim(c.read(c.OUT/'claim.json'))
    expected=expected_executions();expected_paths={p/'execution.json' for p in expected};actual=set()
    for path in c.OUT.rglob('execution.json'):
        if path.is_relative_to(folder.parent):continue
        boundary();value=c.read(path)
        if ('worker_exit_code' in value and 'observed_owned_identities' in value and 'plan_sha256' in value) or path.parent.parent==c.OUT/'segments':actual.add(path)
    c.require(actual==expected_paths,'Exact complete disk-continuation stage inventory')
    original=c.read(c.LIVE_SOURCE_OUT/'reuse.json');receipts=[];identities={}
    for item in original['passed_executions']+reuse['passed_executions']+[c.artifact(p/'execution.json') for p in expected]:
        boundary();value=c.read(c.resolve(item))
        c.require(value['status']=='passed' and value['worker_exit_code']==0 and value['cleanup_errors']==[] and value['error'] is None,
                  'Every accepted predecessor and new stage actually passed')
        for identity in value['observed_owned_identities']+[value['identity']]:
            require_absent(identity);identities[(identity['pid'],identity['created'])]=identity
        receipts.append(item)
    c.require(not [p for p in c.OUT.rglob('failure.json') if not p.is_relative_to(folder)],'No concealed new failure')
    corpus=c.read(c.SOURCE_OUT/'collection/corpus_acceptance.json')
    c.require(corpus['status']=='accepted' and corpus['game_count']==1152 and corpus['independent_setup_count']==1152,'Complete original corpus')
    from tools.splendorust_independent_data_20261107 import fit
    fit.validate_commitment(c.read(c.SELECTION),c.artifact(c.SOURCE_OUT/'all_fits_committed.json'),boundary=boundary)
    accepted=c.read(c.OUT/'strength/report/independent_acceptance.json')
    c.require(accepted['approved'] is True and not accepted['blocking_findings'] and accepted['fixed_main_games']==400
              and accepted['operational_games']==12,'Complete source-aware independent strength acceptance')
    training=c.read(c.OUT/'training_audit/worker/training_audit.json')
    c.require(training['status']=='all_training_saved_verified' and training['new_optimizer_calls']==0,'Saved fitting verification only')
    offline=c.read(c.OUT/'offline_analysis/worker/arithmetic_verification.json')
    c.require(offline['status']=='independent_saved_arithmetic_verified' and offline['contrasts']==42 and
              offline['model_calls']==offline['optimizer_calls']==0,'Full independent saved offline arithmetic')
    for base,names in ((c.SOURCE_OUT,('qualification','collection','corpus','fit','offline')),(c.OUT,('strength','strength_replay'))):
        for name in names:
            end=c.read(base/'stages'/f'{name}_end.json');start=c.read(c.resolve(end['start']))
            c.require(start['monotonic']<=end['monotonic']<start['monotonic_deadline'] and
                      start['unix_time']<=end['unix_time']<start['unix_deadline'],'Every complete phase inside inherited clocks')
    phase=c.read(c.OUT/'stages/strength_start.json');old=c.read(c.SOURCE_OUT/'stages/strength_start.json')
    c.require({k:phase[k] for k in old if k!='claim'}=={k:old[k] for k in old if k!='claim'},'Original strength phase never restarted')
    preserved=[verify_inventory(c.SOURCE_OUT,reuse['original_inventory'],reuse['original_preserved_files'],reuse['original_preserved_bytes'],boundary)]
    live=reuse['live_inventory'];preserved.append(verify_inventory(c.LIVE_SOURCE_OUT,live['inventory'],live['files'],live['bytes'],boundary))
    inventory=folder/'recovery_evidence.jsonl';count=total=0
    excluded={c.OUT/'stage.json',c.OUT/'stage.tmp',c.OUT/'runner_stdout.log',c.OUT/'runner_stderr.log'}
    with inventory.open('xb') as stream:
        for p in sorted(c.OUT.rglob('*')):
            if not p.is_file() or p.is_relative_to(folder.parent) or p in excluded:continue
            boundary();before=p.stat();item=c.artifact(p);after=p.stat()
            c.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable new saved evidence')
            stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    c.require(total+reuse['preserved_bytes']<c.MAX_EVIDENCE_BYTES,'Unchanged total logical evidence cap')
    value=dict(status='passed',scope='separately_recovered_disk_continuation',all_failed_attempts_preserved=True,
        recovery_freeze=c.artifact(c.PLAN),preserved_attempts=preserved,successful_actual_exits=receipts,
        original_stages=155,previous_live_passed_stages=35,new_completed_stages=len(expected),
        exact_owned_identities_absent=list(identities.values()),physical_inventory=c.artifact(inventory),
        recovery_files=count,recovery_bytes=total,source_games=1152,main_strength_games=400,operational_strength_games=12,
        strength_acceptance=c.artifact(c.OUT/'strength/report/independent_acceptance.json'),
        no_automatic_promotion=True,reference='original_ef8a',new_fits=0,new_predictions=0,
        scientific_decisions_retried=0,replacement_games=0,owner_exit_still_requires_observation=True)
    c.save(folder/'independent_acceptance.json',value);return value
