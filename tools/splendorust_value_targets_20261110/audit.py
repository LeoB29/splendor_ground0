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
    generic=[c.OUT/'training_qualification']
    generic += [c.OUT/'fits'/spec['fit_id'] for spec in c.fits()]
    generic += [c.OUT/'commit',c.OUT/'offline',c.OUT/'training_audit',c.OUT/'offline_analysis']
    from .owner import segment_id
    strength=[]
    for mode in ('qualification','operational','replay','saved','independent'):
        strength.extend(c.OUT/'segments'/segment_id(mode,rep) for rep in (0,1))
    strength.extend(c.OUT/'segments'/segment_id('main_block',rep,block)
                    for block in range(0,200,10) for rep in (0,1))
    strength.extend(c.OUT/'segments'/segment_id(mode,0) for mode in ('aggregate','reader'))
    return generic+strength


def final(plan,folder,boundary):
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    c.verify(plan['inputs'],boundary=boundary)
    expected=expected_executions()
    excluded=folder.parent/'execution.json'
    actual={p.resolve() for p in c.OUT.rglob('execution.json') if p.resolve()!=excluded.resolve()}
    # Raw nested worker lifecycles use their own schema and are independently audited by their scientific reader.
    stage_actual=set()
    for path in actual:
        value=c.read(path)
        if 'worker_exit_code' in value and 'observed_owned_identities' in value and 'plan_sha256' in value:
            stage_actual.add(path)
        elif path.parent.parent==c.OUT/'segments': stage_actual.add(path)
    expected_paths={p/'execution.json' for p in expected}
    c.require(stage_actual==expected_paths,'All and only planned completed stage executions')
    receipts=[];identities={}
    for container in expected:
        boundary();path=container/'execution.json';value=c.read(path)
        c.require(value['status']=='passed' and value['worker_exit_code']==0
                  and not value['cleanup_errors'] and value['error'] is None,
                  'Every predecessor actually exited successfully and cleaned up')
        for item in value['observed_owned_identities']+[value['identity']]:
            require_absent(item);identities[(item['pid'],item['created'])]=item
        receipts.append(c.artifact(path))
    failures=[p for p in c.OUT.rglob('failure.json') if not p.is_relative_to(folder)]
    c.require(not failures,'No silently discarded failed stage')
    acceptance=c.read(c.OLD_OUT/'collection/corpus_acceptance.json')
    c.require(acceptance.get('status') in ('passed','accepted','verified'),'Accepted complete scientific corpus')
    c.require(c.read(c.SOURCE_ACCEPTANCE)['approved'] is True and
              c.read(c.SOURCE_CLOSEOUT)['status']=='completed_verified', 'Source study remains accepted')
    from . import fit
    selected=c.read(c.SELECTION)
    fit.validate_commitment(selected,c.artifact(c.OUT/'all_fits_committed.json'),boundary=boundary)
    strength_acceptance=c.OUT/'strength/report/independent_acceptance.json'
    c.require(strength_acceptance.exists(),'Independent complete strength acceptance required')
    accepted_strength=c.read(strength_acceptance)
    c.require(accepted_strength['approved'] is True and accepted_strength['blocking_findings']==[]
              and accepted_strength['fixed_main_games']==400 and accepted_strength['operational_games']==12,
              'Exact complete independent strength acceptance')
    for name in ('qualification','fit','offline','strength','strength_replay'):
        ended=c.read(c.OUT/'stages'/f'{name}_end.json')
        started=c.read(c.resolve(ended['start']))
        c.require(started['monotonic']<=ended['monotonic']<started['monotonic_deadline']
                  and started['unix_time']<=ended['unix_time']<started['unix_deadline'],
                  'Every phase completed within both original clocks')
    # Stream a physical-byte inventory instead of loading hundreds of thousands of records in memory.
    inventory=folder/'preserved_evidence.jsonl'
    count,total=0,0
    with inventory.open('xb') as stream:
        for path in sorted(c.OUT.rglob('*')):
            if not path.is_file() or path.is_relative_to(folder.parent): continue
            if path in (c.OUT/'stage.json',c.OUT/'runner_stdout.log',c.OUT/'runner_stderr.log'): continue
            boundary()
            before=path.stat();item=c.artifact(path);after=path.stat()
            c.require(before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,
                      'Closed evidence changed during independent preservation')
            stream.write((c.canonical(item)+'\n').encode())
            count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    c.require(total<c.MAX_EVIDENCE_BYTES,'Actual final evidence size cap')
    result=dict(status='passed',original_freeze=c.artifact(c.PLAN),planned_predecessors=len(expected),
        successful_actual_exits=receipts,exact_owned_identities_absent=list(identities.values()),
        physical_inventory=c.artifact(inventory),preserved_file_count=count,preserved_bytes=total,
        corpus_acceptance=c.artifact(c.OLD_OUT/'collection/corpus_acceptance.json'),
        selection=c.artifact(c.SELECTION),all_fits_committed=c.artifact(c.OUT/'all_fits_committed.json'),
        strength_acceptance=c.artifact(strength_acceptance),
        no_automatic_promotion=True,reference='original_ef8a',scientific_source_games=1152,
        new_collection_games=0,reused_engineering_games=2,main_strength_games=400,operational_strength_games=12,
        owner_exit_still_requires_observation=True,
        inventory_exclusions=['this active final-audit subtree','mutable root stage','retained owner stdout/stderr'])
    c.save(folder/'independent_acceptance.json',result)
    return result


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
