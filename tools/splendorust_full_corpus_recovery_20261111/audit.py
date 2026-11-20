"""Saved-only corpus selection and independent whole-study preservation audit."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import traceback
import psutil
from . import common as c


def expected_executions():
    from .recovery import expected_live_segments
    from .owner import segment_id
    names=list(expected_live_segments())
    for matchup in c.MATCHUPS:
        names += [segment_id(mode,rep,matchup=matchup) for mode in ('replay','saved','independent') for rep in (0,1)]
        names += [segment_id(mode,0,matchup=matchup) for mode in ('aggregate','reader')]
    return [c.OUT/'segments'/name for name in names]+[c.OUT/'training_audit',c.OUT/'offline_analysis']


def final(plan,folder,boundary):
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    from .recovery import validate_reuse,validate_recovery_claim,original_inventory_verify
    from .boundary import check_source_boundary,verify_retry_consumption
    c.verify(plan['inputs'],boundary=boundary)
    reuse=validate_reuse(c.read(c.OUT/'reuse.json'),boundary=boundary)
    validate_recovery_claim(c.read(c.OUT/'claim.json'))
    source_boundary=check_source_boundary(boundary=boundary,deep=True)
    retry=verify_retry_consumption(boundary=boundary)
    expected=expected_executions();expected_paths={p/'execution.json' for p in expected};actual=set()
    for path in c.OUT.rglob('execution.json'):
        if path.is_relative_to(folder.parent):continue
        boundary();value=c.read(path)
        if ('worker_exit_code' in value and 'observed_owned_identities' in value and 'plan_sha256' in value) or path.parent.parent==c.OUT/'segments':actual.add(path)
    c.require(actual==expected_paths,'All and only planned completed amended execution stages')
    original=c.read(c.ORIGINAL_OUT/'execution.json');receipts=[];identities={}
    c.require(original['status']=='failed' and original['error'] is not None,'Original failed attempt remains incomplete')
    for item in original['segments']+[c.artifact(p/'execution.json') for p in expected]:
        boundary();value=c.read(c.resolve(item))
        c.require(value['status']=='passed' and value['worker_exit_code']==0 and value['cleanup_errors']==[] and value['error'] is None,
                  'Every accepted original predecessor and new stage actually passed')
        for identity in value['observed_owned_identities']+[value['identity']]:
            require_absent(identity);identities[(identity['pid'],identity['created'])]=identity
        receipts.append(item)
    c.require(not [p for p in c.OUT.rglob('failure.json') if not p.is_relative_to(folder)],'No concealed new failure')
    corpus=c.read(c.OLD_OUT/'collection/corpus_acceptance.json')
    c.require(corpus.get('status') in ('passed','accepted','verified') and
              c.read(c.SOURCE_ACCEPTANCE)['approved'] is True and c.read(c.SOURCE_CLOSEOUT)['status']=='completed_verified',
              'Original corrected corpus and recovered collection acceptance retained')
    from tools.splendorust_full_corpus_20261110 import fit
    fit.validate_commitment(c.read(c.SELECTION),c.artifact(c.ORIGINAL_OUT/'all_fits_committed.json'),boundary=boundary)
    training=c.read(c.OUT/'training_audit/worker/training_audit.json')
    c.require(training['status']=='all_training_saved_verified' and training['new_optimizer_calls']==training['recovery_model_calls']==0 and
              training['scientific_optimizer_calls']==7128 and training['optimizer_calls']==7134,
              'Saved fitting verification only; all original optimization accounting retained')
    offline=c.read(c.OUT/'offline_analysis/worker/arithmetic_verification.json')
    c.require(offline['status']=='independent_saved_arithmetic_verified' and offline['model_calls']==offline['optimizer_calls']==0,
              'Independent model-free saved prediction arithmetic')
    from . import family
    family_result=family.finalize(folder/'family',boundary=boundary)
    strength_acceptances=[];retry_total=0
    for matchup in c.MATCHUPS:
        path=c.OUT/'strength'/matchup/'report/independent_acceptance.json';accepted=c.read(path)
        c.require(accepted['approved'] is True and accepted['blocking_findings']==[] and
                  accepted['fixed_main_games']==400 and accepted['operational_games']==12 and
                  accepted['amended_recovered_study'] is True and accepted['original_attempt_status']=='incomplete' and
                  accepted['original_failed_attempt_preserved'] is True,'Complete source-aware independent matchup acceptance')
        retry_total+=accepted['authorized_potentially_interrupted_retry_count'];strength_acceptances.append(c.artifact(path))
    c.require(retry_total==1,'Exactly one disclosed authorized retry across all1200 games')
    for base,names in ((c.ORIGINAL_OUT,('qualification','corpus','fit','offline')),(c.OUT,('strength','strength_replay'))):
        for name in names:
            ended=c.read(base/'stages'/f'{name}_end.json');started=c.read(c.resolve(ended['start']))
            c.require(started['monotonic']<=ended['monotonic']<started['monotonic_deadline'] and
                      started['unix_time']<=ended['unix_time']<started['unix_deadline'],'Every complete phase inside original dual clocks')
    phase=c.read(c.OUT/'stages/strength_start.json');old=c.read(c.ORIGINAL_OUT/'stages/strength_start.json')
    c.require({k:phase[k] for k in old if k!='claim'}=={k:old[k] for k in old if k!='claim'},'Original strength phase never restarted')
    preserved=original_inventory_verify(boundary=boundary)
    inventory=folder/'recovery_evidence.jsonl';count=total=0
    excluded={c.OUT/'stage.json',c.OUT/'stage.tmp',c.OUT/'runner_stdout.log',c.OUT/'runner_stderr.log'}
    with inventory.open('xb') as stream:
        for path in sorted(c.OUT.rglob('*')):
            if not path.is_file() or path.is_relative_to(folder.parent) or path in excluded:continue
            boundary();before=path.stat();item=c.artifact(path);after=path.stat()
            c.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable new saved evidence')
            stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    c.require(total+reuse['preserved_bytes']<c.MAX_EVIDENCE_BYTES,'Unchanged combined logical evidence cap')
    result=dict(status='passed',scope='amended_recovered_full_corpus_study',amended_recovered_study=True,
        original_attempt_status='incomplete',original_failed_attempt_preserved=True,
        recovery_freeze=c.artifact(c.PLAN),original_freeze=c.artifact(c.ORIGINAL_PLAN),preserved_attempt=preserved,
        successful_actual_exits=receipts,original_passed_stages=len(original['segments']),new_completed_stages=len(expected),
        exact_owned_identities_absent=list(identities.values()),physical_inventory=c.artifact(inventory),
        recovery_files=count,recovery_bytes=total,preserved_original_bytes=reuse['preserved_bytes'],
        corpus_acceptance=c.artifact(c.OLD_OUT/'collection/corpus_acceptance.json'),selection=c.artifact(c.SELECTION),
        all_fits_committed=c.artifact(c.ORIGINAL_OUT/'all_fits_committed.json'),strength_acceptances=strength_acceptances,
        family_result=family_result,cache_acceptance=c.artifact(c.CACHE_ACCEPTANCE),source_boundary=source_boundary,
        retry_consumption=retry,authorized_potentially_interrupted_retry_count=1,completed_decisions_retried=0,
        replacement_games=0,source_games=1152,main_strength_games=1200,operational_strength_games=36,
        no_automatic_promotion=True,reference='original_ef8a',new_fits=0,new_predictions=0,
        owner_exit_still_requires_observation=True,
        inventory_exclusions=['active final-audit subtree','mutable root stage and tmp','retained owner stdout/stderr'])
    c.save(folder/'independent_acceptance.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('final',),required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.output).resolve()
    from .runtime import bootstrap
    plan,boundary,sources=bootstrap(folder,args.plan_sha256,args.mode,model=False)
    status,error,result='failed',None,None
    try:
        result=final(plan,folder,boundary)
        boundary();c.require('torch' not in sys.modules,'Saved-only worker loads no model runtime')
        status='passed'
    except BaseException:error=traceback.format_exc()
    c.save(folder/'result.json',dict(status=status,mode=args.mode,
        identity=dict(pid=os.getpid(),created=psutil.Process().create_time()),time=c.now(),
        result=result,error=error,plan_sha256=args.plan_sha256,boundary_calls=boundary.calls))
    if error:print(error,flush=True)
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
