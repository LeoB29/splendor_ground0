"""Independent complete recovery acceptance, preserving both original and new evidence."""
import os
import json
from pathlib import Path
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent


def expected_executions():
    from .owner import segment_id
    paths=[c.OUT/'training_audit',c.OUT/'offline_analysis']
    for mode in ('qualification','operational','replay','saved','independent'):
        paths.extend(c.OUT/'segments'/segment_id(mode,rep) for rep in (0,1))
    paths.extend(c.OUT/'segments'/segment_id('main_block',rep,block) for block in range(0,200,10) for rep in (0,1))
    paths.extend(c.OUT/'segments'/segment_id(mode,0) for mode in ('aggregate','reader'))
    return paths


def verify_preserved(reuse,boundary=lambda:None):
    c.verify([reuse['old_inventory']],boundary=boundary)
    seen=set();total=0
    with c.resolve(reuse['old_inventory']).open(encoding='utf-8') as stream:
        for line in stream:
            boundary();item=json.loads(line,object_pairs_hook=c._unique)
            path=c.resolve(item)
            c.require(path.is_relative_to(c.SOURCE_OUT) and path not in seen,'Unique original preserved path')
            c.verify([item],boundary=boundary);seen.add(path);total+=item['bytes']
    actual={p.resolve() for p in c.SOURCE_OUT.rglob('*') if p.is_file()}
    c.require(actual==seen and len(seen)==reuse['preserved_files'] and total==reuse['preserved_bytes'],
              'Complete original attempt unchanged, including failure and partial exports')
    return dict(files=len(seen),bytes=total,inventory=reuse['old_inventory'],all_original_bytes_preserved=True)


def final(plan,folder,boundary):
    from .recovery import validate_reuse,validate_recovery_claim
    c.verify(plan['inputs'],boundary=boundary)
    reuse=validate_reuse(c.read(c.OUT/'reuse.json'),boundary=boundary)
    claim=validate_recovery_claim(c.read(c.OUT/'claim.json'))
    expected=expected_executions();expected_paths={p/'execution.json' for p in expected}
    actual=set()
    for p in c.OUT.rglob('execution.json'):
        if p.is_relative_to(folder.parent):continue
        boundary();v=c.read(p)
        if ('worker_exit_code' in v and 'observed_owned_identities' in v and 'plan_sha256' in v) or p.parent.parent==c.OUT/'segments':actual.add(p)
    c.require(actual==expected_paths,'All and only recovery stage executions')
    receipts=[];identities={}
    for item in reuse['passed_executions']+[c.artifact(p/'execution.json') for p in expected]:
        boundary();v=c.read(c.resolve(item))
        c.require(v['status']=='passed' and v['worker_exit_code']==0 and v['cleanup_errors']==[] and v['error'] is None,
                  'Every reused and new completed stage exited cleanly')
        for ident in v['observed_owned_identities']+[v['identity']]:
            require_absent(ident);identities[(ident['pid'],ident['created'])]=ident
        receipts.append(item)
    c.require(not [p for p in c.OUT.rglob('failure.json') if not p.is_relative_to(folder)],'No hidden recovery failure')
    corpus=c.read(c.SOURCE_OUT/'collection/corpus_acceptance.json')
    c.require(corpus['status']=='accepted' and corpus['game_count']==1152 and corpus['independent_setup_count']==1152,
              'Original accepted full independent corpus')
    from tools.splendorust_independent_data_20261107 import fit
    fit.validate_commitment(c.read(c.SELECTION),c.artifact(c.SOURCE_OUT/'all_fits_committed.json'),boundary=boundary)
    accepted=c.read(c.OUT/'strength/report/independent_acceptance.json')
    c.require(accepted['approved'] is True and accepted['blocking_findings']==[] and
              accepted['fixed_main_games']==400 and accepted['operational_games']==12,'Full independent strength acceptance')
    training=c.read(c.OUT/'training_audit/worker/training_audit.json')
    c.require(training['status']=='all_training_saved_verified' and training['new_optimizer_calls']==0,'Complete saved training verification')
    offline=c.read(c.OUT/'offline_analysis/worker/arithmetic_verification.json')
    c.require(offline['status']=='independent_saved_arithmetic_verified' and offline['contrasts']==42 and
              offline['model_calls']==offline['optimizer_calls']==0,'Independent full offline arithmetic')
    for base,names in [(c.SOURCE_OUT,('qualification','collection','corpus','fit','offline')),
                       (c.OUT,('strength','strength_replay'))]:
        for name in names:
            ended=c.read(base/'stages'/f'{name}_end.json');started=c.read(c.resolve(ended['start']))
            c.require(started['monotonic']<=ended['monotonic']<started['monotonic_deadline'] and
                      started['unix_time']<=ended['unix_time']<started['unix_deadline'],'Every phase within original dual clocks')
    phase=c.read(c.OUT/'stages/strength_start.json');old=c.read(c.SOURCE_OUT/'stages/strength_start.json')
    c.require({k:phase[k] for k in old if k!='claim'}=={k:old[k] for k in old if k!='claim'} and
              phase['original_phase']==reuse['original_strength_phase'],'Original strength time never reset')
    preserved=verify_preserved(reuse,boundary)
    inventory=folder/'recovery_evidence.jsonl';count=total=0
    excluded={c.OUT/'stage.json',c.OUT/'stage.tmp',c.OUT/'runner_stdout.log',c.OUT/'runner_stderr.log'}
    with inventory.open('xb') as stream:
        for p in sorted(c.OUT.rglob('*')):
            if not p.is_file() or p.is_relative_to(folder.parent) or p in excluded:continue
            boundary();before=p.stat();item=c.artifact(p);after=p.stat()
            c.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable recovery evidence')
            stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    c.require(total+preserved['bytes']<c.MAX_EVIDENCE_BYTES,'Combined original and recovery evidence ceiling')
    result=dict(status='passed',scope='separately_recovered_verification',original_failed_attempt_preserved=True,
        original_preservation=preserved,recovery_freeze=c.artifact(c.PLAN),successful_actual_exits=receipts,
        reused_stages=155,new_completed_stages=len(expected),exact_owned_identities_absent=list(identities.values()),
        physical_inventory=c.artifact(inventory),recovery_files=count,recovery_bytes=total,
        corpus_acceptance=c.artifact(c.SOURCE_OUT/'collection/corpus_acceptance.json'),
        all_fits_committed=c.artifact(c.SOURCE_OUT/'all_fits_committed.json'),
        strength_acceptance=c.artifact(c.OUT/'strength/report/independent_acceptance.json'),
        source_games=1152,main_strength_games=400,operational_strength_games=12,
        no_automatic_promotion=True,reference='original_ef8a',new_fits=0,new_predictions=0,
        owner_exit_still_requires_observation=True)
    c.save(folder/'independent_acceptance.json',result);return result
