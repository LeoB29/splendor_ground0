"""Saved-only matchup analysis and independent final reader."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import psutil
from .. import common as c
from . import common as a
REPORT = a.MATCHUP_OUT/'report'

def root_segment(mode,rep,start=None):
    from ..owner import segment_id
    return c.OUT/'segments'/segment_id(mode,rep,start,matchup=a.MATCHUP)



def main_games():
    result = {}
    for rep in (0, 1):
        folder = a.MATCHUP_OUT/f'r{rep}'
        paths = sorted((folder/'games/main').glob('g*/result.json'))
        c.require(len(paths) == 200, 'Both complete fixed main cohorts required')
        result[rep] = [c.read(path) for path in paths]
    return result


def aggregate(envelope):
    from . import analysis
    games=main_games()
    for rep in (0,1):
        receipt=c.read(root_segment('saved',rep)/'execution.json')
        c.require(receipt['status']=='passed' and receipt['worker_exit_code']==0 and not receipt['cleanup_errors'], 'Saved science audits actually passed')
    value=analysis.analyze_pooled(games,True,boundary=envelope.guard)
    proof=analysis.verify_pooled(games,value,boundary=envelope.guard)
    result=c.save(REPORT/'analysis.json',value)
    c.save(REPORT/'arithmetic_verification.json',proof)
    summary=c.save(REPORT/'summary.json',dict(matchup=a.MATCHUP,status='complete',main_games=400,operational_games=12,
        analysis=result,primary_target_contrast=a.MATCHUP=='mixed_terminal',
        parent_family_claim_uses_97_5_interval=a.MATCHUP!='mixed_terminal',
        automatic_promotion=False,reference_arm=a.ROUTE_ARMS['control'],reference_checkpoints=[a.fit_id('control',r) for r in (0,1)],charts='Combined family summary writes one result chart'))
    return dict(status='aggregate_complete',analysis=result,summary=summary)


def independent_reader(envelope, freeze):
    from . import analysis
    from ..owner import segment_id
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    expected = [('qualification',rep,None) for rep in (0,1)]
    expected += [('operational',rep,None) for rep in (0,1)]
    expected += [('main_block',rep,start) for start in range(0,200,10) for rep in (0,1)]
    expected += [('replay',rep,None) for rep in (0,1)]
    expected += [('saved',rep,None) for rep in (0,1)] + [('aggregate',0,None)]
    expected += [('independent',rep,None) for rep in (0,1)]
    prefixes = tuple(mode + '_' + a.MATCHUP + '_r' for mode, _, _ in expected)
    actual = sorted(p.parent.name for p in (c.OUT/'segments').glob('*/execution.json')
                    if p.parent.name.startswith(prefixes))
    c.require(actual == sorted(segment_id(*item,matchup=a.MATCHUP) for item in expected), 'Exact completed segment inventory')
    previous_end = None
    for mode, rep, start in expected:
        envelope.guard()
        folder = root_segment(mode,rep,start)
        receipt, owner = c.read(folder/'execution.json'), c.read(folder/'owner.json')
        c.require(receipt['status'] == 'passed' and receipt['worker_exit_code'] == 0 and
                  receipt['cleanup_errors'] == [] and receipt['error'] is None, 'All actual segment exits passed')
        c.require(receipt['identity'] == owner['identity'], 'Exact actual owner')
        if previous_end is not None:
            c.require(owner['created_monotonic'] >= previous_end, 'No overlapping segment controllers')
        previous_end = receipt['ended_monotonic']
        for item in receipt['observed_owned_identities']: require_absent(item)
        require_absent(receipt['identity'])
        segment = c.read(folder/'claim.json')
        phase = c.read(c.ROOT/segment['phase_clock']['path'])
        c.require(receipt['ended_monotonic'] < phase['monotonic_deadline'] and
                  receipt['ended_unix'] < phase['unix_deadline'], 'Actual exit inside inherited original clocks')
    value = c.read(REPORT/'analysis.json')
    arithmetic = analysis.verify_pooled(main_games(), value, boundary=envelope.guard)
    for item in freeze['inputs']:
        envelope.guard(); c.verify([item])
    inventory=[]
    # Seal finished scientific evidence. Exclude currently open owner/reader logs.
    for base in (a.MATCHUP_OUT/'r0',a.MATCHUP_OUT/'r1',REPORT):
        for path in sorted(base.rglob('*')):
            if path.is_file(): envelope.guard(); inventory.append(c.artifact(path))
    inventory_item = c.save(a.MATCHUP_OUT/'final_inventory.json',dict(schema='full_corpus_strength_completed_inventory_v1',
                         artifacts=inventory, files=len(inventory), bytes=sum(x['bytes'] for x in inventory)))
    for item in inventory: envelope.guard(); c.verify([item])
    workers = list(a.MATCHUP_OUT.glob('r*/workers/*/initialization.json'))
    c.require(len(workers) <= 812 <= freeze['maximum_live_lifecycles'], 'Bounded complete lifecycle accounting')
    for path in workers:
        init = c.read(path); require_absent(init['identity'])
    value = dict(schema='full_corpus_strength_independent_acceptance_v1', approved=True, blocking_findings=[],
        matchup=a.MATCHUP,candidate_arm=a.ROUTE_ARMS['candidate'],reference_arm=a.ROUTE_ARMS['control'],analysis=c.artifact(REPORT/'analysis.json'),fixed_main_games=400, operational_games=12, fixed_candidate_checkpoints=2, fixed_control_checkpoints=1 if a.ROUTE_ARMS['control']=='parent' else 2, distinct_compared_checkpoints=3 if a.ROUTE_ARMS['control']=='parent' else 4, route_arms=dict(a.ROUTE_ARMS),
        independent_per_repetition_saved_reviews=[c.artifact(root_segment('independent',rep)/'result.json') for rep in (0,1)],
        independent_pooled_arithmetic=arithmetic, completed_segments=len(expected),
        exact_worker_lifecycles=len(workers), all_completed_segment_exits_observed=True,
        saved_inventory=inventory_item, frozen_inputs_unchanged=len(freeze['inputs']),
        original_reference_retained=True, parent_is_opponent=a.ROUTE_ARMS['control']=='parent', automatic_promotion=False, actual_outer_owner_exit_pending=True,
        charts_require_final_visual_inspection=True, model_calls=0, new_games=0)
    return c.save(REPORT/'independent_acceptance.json', value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('aggregate','reader'),required=True)
    parser.add_argument('--segment',required=True);parser.add_argument('--freeze-sha256',required=True)
    args=parser.parse_args()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources()
    folder=Path(args.segment).resolve()
    c.require(folder.parent==c.OUT/'segments','Exact saved-only segment')
    c.require(c.file_hash(c.OUT/'freeze.json')==args.freeze_sha256,'Frozen saved-only reader')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(folder/'claim.json')
    from ..owner import validate_segment_claim
    validate_segment_claim(claim,folder,args.mode,0,None,args.freeze_sha256,matchup=a.MATCHUP)
    c.require(claim['mode']==args.mode and claim['repetition']==0 and claim['matchup']==a.MATCHUP,'Saved-only fixed owner role')
    c.require(claim['freeze_sha256']==args.freeze_sha256 and
              c.read(c.ROOT/claim['claim']['path'])['freeze']==c.artifact(c.OUT/'freeze.json'),
              'Original owner and saved-only segment freeze binding')
    from .segment import owning_handshake
    owner_handshake=owning_handshake(folder,claim)
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placement.configure_execution(freeze,'isolated')
    c.save(folder/'runtime_sources.json',dict(runtime_sources=sources,mode=args.mode,matchup=a.MATCHUP,new_models=0,new_games=0,owner_handshake=owner_handshake))
    from .envelope import SegmentEnvelope
    envelope=SegmentEnvelope(claim['claim'],claim['phase_clock'],c.artifact(folder/'claim.json'))
    status,error,value='failed',None,None
    try:
        c.require('torch' not in sys.modules,'No models in saved-only report process')
        value=aggregate(envelope) if args.mode=='aggregate' else independent_reader(envelope,freeze)
        envelope.finish();status='passed'
    except BaseException: error=traceback.format_exc()
    c.save(folder/'result.json',dict(status=status,mode=args.mode,repetition=0,matchup=a.MATCHUP,result=value,error=error,
        pid=os.getpid(),created=psutil.Process().create_time(),new_games=0,model_calls=0))
    if error: print(error,file=sys.stderr)
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
