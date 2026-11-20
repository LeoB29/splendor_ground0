"""Model-free union construction and independent collection-only final audit."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import traceback
import psutil
from . import common as c


def expected_jobs():
    """Only the frozen collection route; no scientific fit or live assessment."""
    module=c.PACKAGE+'.collection.worker'
    jobs=[('collection/qualification','qualification','qualification',module,True,c.COLLECTION_RSS),
          ('collection/qualification_audit','audit_qualification','qualification',module,False,c.COLLECTION_RSS),
          ('training_qualification','qualification','qualification',c.PACKAGE+'.fit',True,c.FIT_RSS)]
    for index in range(c.CELLS):
        ident=f'c{index:03d}'
        jobs.extend([(f'collection/cells/{ident}','games','collection',module,True,c.COLLECTION_RSS),
                     (f'collection/cell_audits/{ident}','audit_cell','collection',module,False,c.COLLECTION_RSS)])
    jobs.extend([('collection/corpus_audit','corpus','corpus',module,False,c.COLLECTION_RSS),
                 ('union_build','selection','corpus',c.PACKAGE+'.audit',False,c.FIT_RSS)])
    return [dict(path=path,mode=mode,phase=phase,module=module,model=model,rss=rss,
                 seconds=(24 if mode=='corpus' else 4)*3600)
            for path,mode,phase,module,model,rss in jobs]


def expected_executions():
    return [c.OUT/job['path'] for job in expected_jobs()]


def closed_files(folder):
    """Bound memory by one directory instead of sorting the complete evidence tree."""
    pending=[Path(folder)]
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:rows=sorted(entries,key=lambda item:item.name)
        for entry in rows:
            c.require(not entry.is_symlink(),'Evidence contains no linked paths')
            if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):yield Path(entry.path).resolve()


def verify_phases(claim,*,boundary=lambda:None):
    """Rebuild the unchanged dual-clock caps and completed phase ordering."""
    freeze=c.artifact(c.PLAN);claim_item=c.artifact(c.OUT/'claim.json')
    c.require(claim['freeze']==freeze and claim['plan_sha256']==freeze['sha256'] and
        claim['total_seconds']==c.TOTAL_SECONDS and claim['model_seconds']==c.MODEL_SECONDS and
        claim['automatic_retries']==0,'Original collection-only owner claim')
    phases={};previous=(claim['monotonic_origin'],claim['unix_origin'])
    for name in ('qualification','collection','corpus','final'):
        boundary();path=c.OUT/'stages'/f'{name}_start.json';item=c.artifact(path);row=c.read(path)
        cap=c.STAGES[name];cutoff=c.TOTAL_SECONDS if name=='final' else c.MODEL_SECONDS
        c.require(row['stage']==name and row['claim']==claim_item and type(row['cap_seconds']) is int
            and row['cap_seconds']==cap,'Exact original stage identity and cap')
        c.require(row['monotonic_deadline']==min(row['monotonic']+cap,claim['monotonic_origin']+cutoff)
            and row['unix_deadline']==min(row['unix_time']+cap,claim['unix_origin']+cutoff)
            and previous[0]<=row['monotonic']<row['monotonic_deadline']
            and previous[1]<=row['unix_time']<row['unix_deadline'],'Original ordered clipped stage clocks')
        end=None
        if name!='final':
            end=c.read(c.OUT/'stages'/f'{name}_end.json')
            c.require(end['stage']==name and end['start']==item
                and row['monotonic']<=end['monotonic']<row['monotonic_deadline']
                and row['unix_time']<=end['unix_time']<row['unix_deadline'],'Complete phase inside original dual clocks')
            previous=(end['monotonic'],end['unix_time'])
        phases[name]=dict(start=row,start_item=item,end=end)
    return phases


def verify_execution(job,plan,claim,phases,*,boundary=lambda:None,absent=None):
    """Authenticate actual successful process exit, fixed role, bounds and cleanup."""
    if absent is None:
        from tools.splendorust_gpu_probe_20261105.control import require_absent
        absent=require_absent
    boundary();container=c.OUT/job['path'];owner=c.read(container/'owner.json')
    value=c.read(container/'execution.json');phase=phases[job['phase']]['start']
    phase_item=phases[job['phase']]['start_item'];claim_item=c.artifact(c.OUT/'claim.json')
    freeze=c.artifact(c.PLAN)
    c.require(value['status']=='passed' and type(value['worker_exit_code']) is int and value['worker_exit_code']==0
        and value['cleanup_errors']==[] and value['error'] is None and value['retries']==0,
        'Every predecessor actually exited successfully without retry and cleaned up')
    c.require(value['identity']==owner['identity'] and value['mode']==owner['mode']==job['mode']
        and value['fit_id'] is None and owner['fit_id'] is None
        and value['plan_sha256']==owner['plan_sha256']==freeze['sha256']
        and owner['windows_job_assigned'] is True and owner['model_work'] is job['model']
        and owner['global_claim']==claim_item and owner['phase_clock']==phase_item,
        'Exact original owner, worker role, freeze and phase')
    c.require(type(owner['maximum_seconds']) is int and owner['maximum_seconds']==value['maximum_seconds']==job['seconds']
        and owner['maximum_rss_bytes']==job['rss'] and 0<=value['peak_rss_bytes']<=job['rss'],
        'Exact worker time and actual RSS limits')
    expected_args=['-B','-u','-m',job['module'],'--mode',job['mode'],'--plan-sha256',freeze['sha256'],
                   '--output',str(container/'worker')]
    cell=None
    if job['phase']=='collection':
        cell_id=container.name;expected_args+=['--cell-id',cell_id]
        cell=c.read(c.OUT/'collection/cell_clocks'/f'{cell_id}.json')
        c.require(cell['cell_id']==cell_id and cell['phase']==phase_item
            and cell['monotonic_deadline']==cell['monotonic_origin']+4*3600
            and cell['unix_deadline']==cell['unix_origin']+4*3600,'Shared unchanged cell clock')
    c.require(type(owner['args']) is list and len(owner['args'])==len(expected_args)+1
        and owner['args'][1:]==expected_args,'Exact scheduled worker executable arguments')
    for origin,end,deadline,phase_start in (('monotonic_origin','monotonic_end','monotonic_deadline','monotonic'),
                                          ('unix_origin','unix_end','unix_deadline','unix_time')):
        bound=min(owner[origin]+job['seconds'],phase[deadline])
        if cell is not None:
            bound=min(bound,cell[deadline]);c.require(phase[phase_start]<=cell[origin]<=owner[origin],
                                                   'Cell begins inside original phase')
        c.require(value[origin]==owner[origin] and value[deadline]==owner[deadline]==bound
            and phase[phase_start]<=owner[origin]<=value[end]<bound
            and value[end]<=phases[job['phase']]['end'][phase_start],
            'Unextended original execution and phase clock chain')
        c.require(value[end]<claim[origin]+c.MODEL_SECONDS,'All collection work inside original model cutoff')
    identities=value['observed_owned_identities']+[value['identity']]
    for identity in identities:
        c.require(type(identity) is dict and set(identity)=={'pid','created'} and type(identity['pid']) is int
            and type(identity['created']) in (int,float),'Exact process identity receipt')
        absent(identity)
    startup=c.read(container/'worker/runtime_sources.json')
    c.require(startup['sources']==plan['corrected_source_receipt'] and startup['identity']==value['identity']
        and startup['original_claim']==claim_item and startup['original_phase']==phase_item
        and startup['mode']==job['mode'] and startup['fit_id'] is None,'Actual once-startup corrected source guard')
    result=c.read(container/'worker/result.json')
    c.require(result['status']=='passed' and result['mode']==job['mode'] and result['identity']==value['identity'],
        'Actual matching worker completion receipt')
    from tools.splendorust_gpu_probe_20261105.control import native_fallback_messages
    c.require(not native_fallback_messages(container/'worker/stderr.log'),'No unsupported native fallback')
    return dict(execution=c.artifact(container/'execution.json'),owner=c.artifact(container/'owner.json'),
                result=c.artifact(container/'worker/result.json'),identities=identities,
                monotonic_origin=value['monotonic_origin'],monotonic_end=value['monotonic_end'],
                unix_origin=value['unix_origin'],unix_end=value['unix_end'])


def verify_execution_inventory(folder,*,boundary=lambda:None):
    expected={p/'execution.json' for p in expected_executions()};actual=set()
    engineering=c.read(c.PLAN).get('prelaunch_evidence',[])
    c.verify(engineering,boundary=boundary)
    engineering_paths={c.resolve(item) for item in engineering}
    c.require(len(engineering_paths)==len(engineering) and
        all(path.is_relative_to(c.OUT/'prelaunch') for path in engineering_paths),
        'Only exact frozen prelaunch engineering exclusions')
    for path in closed_files(c.OUT):
        boundary()
        if path.is_relative_to(folder.parent):continue
        if path.is_relative_to(c.OUT/'prelaunch'):
            c.require(path in engineering_paths,'No new unbound prelaunch evidence after freeze')
            continue
        c.require(path.name not in ('failure.json','worker_failure.json'),'No silently discarded failed stage')
        if path.name!='execution.json':continue
        value=c.read(path)
        if path in expected or any(key in value for key in ('worker_exit_code','observed_owned_identities','plan_sha256')):
            actual.add(path)
    c.require(actual==expected,'All and only the 421 planned completed predecessor executions')
    return actual


def full_outputs(outputs):
    c.require(type(outputs) is list and len(outputs)==32 and all(type(row) is list and len(row)==83 for row in outputs),
              'Complete qualification 32 by 83 outputs')
    c.require(all(type(x) in (int,float) and math.isfinite(x) and struct.unpack('<f',struct.pack('<f',x))[0]==x
                  for row in outputs for x in row),'Finite exact float32 qualification outputs')
    return b''.join(struct.pack('<f',x) for row in outputs for x in row)


def verify_training_qualification(*,boundary=lambda:None):
    """Recheck saved numerical and manifest evidence; do not deserialize tensor files."""
    from .collection import data
    folder=c.OUT/'training_qualification/worker';item=c.artifact(folder/'qualification_result.json');value=c.read(c.resolve(item))
    data._legacy.accepted.verify_identity(value)
    c.require(value['schema']=='independent_data_qualification_v1' and value['status']=='qualification_passed'
        and value['study_id']=='splendorust_collection4096_20261115_v1'
        and value['optimizer_calls']==2 and value['forward_calls']==8 and value['backward_calls']==2
        and value['scientific_updates']==0 and [r['scope'] for r in value['scopes']]==['joint','policy'],
        'Exactly two discarded original qualification updates')
    c.require(c.read(folder/'result.json')['result']==value,'Worker binds complete qualification record')
    loader=value['data_qualification'];data._legacy.accepted.verify_identity(loader)
    c.require(loader==c.read(folder/'data/loader_qualification.json')
        and loader['status']=='actual_production_loader_qualified'
        and loader['qualification_examples']==32 and loader['actual_selected_loader_and_collator_exercised'] is True
        and loader['zero_scientific_source_games'] is True and loader['synthetic_value_targets'] is False
        and loader['structural_terminal_labels_fabricated'] is False
        and loader['collector_audit']==c.artifact(c.OUT/'collection/qualification_audit/worker/saved_audit.json'),
        'Actual corrected engineering loader, no scientific games or fabricated outcomes')
    c.require(all(row['rejected'] is True for row in loader['actual_target_parser_rejections'])
        and len(loader['actual_target_parser_rejections'])>=12,'Saved actual malformed-target counterexamples')
    c.verify(list(c.nested_artifacts(loader)),boundary=boundary)
    for row in value['scopes']:
        boundary();step=row['step'];scope=row['scope']
        c.require(row['discarded'] is True and row['optimizer_calls']==1 and row['forward_calls']==4
            and row['backward_calls']==1 and row['trainable_parameters']==(4780883 if scope=='joint' else 20817)
            and step['optimizer_calls']==step['forward_backward_calls']==1 and step['finite_gradients'] is True
            and step['scope']==scope and step['batch_examples']==step['batch_denominator']==32,
            'Fixed complete discarded training scope')
        c.require(full_outputs(row['before_outputs83'])==full_outputs(step['outputs83'])
            and full_outputs(row['after_outputs83'])==full_outputs(row['reloaded_outputs83']),
            'Saved exact forward and checkpoint-reload output parity')
        c.verify([row['checkpoint']],boundary=boundary);manifest=c.read(c.resolve(row['checkpoint']));meta=manifest['metadata']
        data._legacy.accepted.verify_identity(meta)
        c.require(manifest['schema']=='learning_diagnostic_checkpoint_manifest_v1'
            and meta['scope']==scope and meta['qualification_only'] is True and meta['checkpoint_step']==1
            and meta['fit_id']=='qualification_'+scope and meta['group_sha256']==step['after_group_sha256']
            and meta['historical_warm_start_sha256']==c.PARENT_CHECKPOINT_SHA and meta['historical_data_relabelled'] is False
            and meta['source_receipt_sha256']==c.digest(loader),'Bound discarded checkpoint metadata')
        c.verify([manifest['checkpoint'],manifest['tensor_manifest']],boundary=boundary)
        trees=c.read(c.resolve(manifest['tensor_manifest']))
        c.require(set(trees)=={'state_dict','optimizer','rng'} and
            {k:c.digest(v) for k,v in trees.items()}==manifest['tree_sha256'],'Preserved full tensor/optimizer/RNG manifest')
        if scope=='policy':
            c.require(all(row['before_group_sha256'][k]==step['after_group_sha256'][k] for k in ('shared','value')),
                      'Frozen shared and value qualification groups')
            before=b''.join(struct.pack('<f',x) for output in row['before_outputs83'] for x in output[81:])
            after=b''.join(struct.pack('<f',x) for output in row['after_outputs83'] for x in output[81:])
            c.require(before==after and row['value_preservation']==dict(rows=32,
                value_float32_sha256=hashlib.sha256(before).hexdigest(),equal=True),'Exact saved value parity')
    return dict(status='qualification_saved_verified_without_tensor_runtime',optimizer_calls=2,result=item,
                tensor_payloads_reloaded=False,actual_model_reload_was_qualified_before_collection=True)


def verify_collection_receipts(plan,*,boundary=lambda:None):
    """Reuse complete accepted semantic/replay audits and authenticate their bindings."""
    from .collection import common as cc
    from . import corpus
    collection=cc.read(cc.PLAN);c.require(plan['collection_plan']==c.artifact(cc.PLAN),'Frozen exact collection plan')
    gate=c.read(c.OUT/'qualification_gate.json');small=cc.read(cc.OUT/'qualification_gate.json')
    c.require(gate['status']=='passed' and gate['plan']==c.artifact(c.PLAN)
        and gate['qualification_optimizer_calls']==2 and gate['scientific_games_released']==0
        and gate['excluded_engineering_games']==2
        and gate['actual_exits']==[c.artifact(p/'execution.json') for p in expected_executions()[:3]],
        'All three qualifications completed before production')
    qual_item=c.artifact(cc.OUT/'qualification_audit/worker/saved_audit.json');qual=cc.read(c.resolve(qual_item))
    c.require(small==dict(passed=True,collection_plan=c.artifact(cc.PLAN),audit=qual_item)
        and qual['status']=='passed' and qual['passed'] is True and qual['complete_decisions']==116
        and qual['engineering_games']==2 and qual['excluded_from_production'] is True
        and qual['model_calls']==0 and qual['collection_plan']==c.artifact(cc.PLAN),'Saved complete actual collector qualification')
    c.require(cc.read(cc.OUT/'qualification_audit/worker/result.json')['result']==qual_item,
              'Qualification auditor returns exact accepted evidence')
    acceptance_item=c.artifact(cc.OUT/'corpus_acceptance.json');corpus._receipt(acceptance_item,boundary)
    acceptance=cc.read(c.resolve(acceptance_item))
    c.require(acceptance['engineering_games_excluded']==2 and acceptance['teacher_simulations']==512
        and acceptance['raw_semantics_and_all_transitions_verified'] is True
        and acceptance['all_original_callbacks_attributed'] is True
        and acceptance['selected_model_replays_verified'] is True and acceptance['models_run']==acceptance['games_run']==0,
        'Complete independently accepted corrected corpus')
    saved=cc.read(cc.OUT/'corpus_audit/worker/saved_audit.json')
    c.require(saved==acceptance and cc.read(cc.OUT/'corpus_audit/worker/result.json')['result']==
        c.artifact(cc.OUT/'corpus_audit/worker/saved_audit.json'),'Exited corpus worker binds exact acceptance')
    expected_audits=[c.artifact(cc.OUT/'corpus_cell_audits'/f'c{i:03d}.json') for i in range(c.CELLS)]
    c.require(acceptance['independent_cell_audits']==expected_audits and len(collection['cells'])==c.CELLS,
        'Every fixed cell independently rechecked before corpus release')
    for cell,item in zip(collection['cells'],expected_audits,strict=True):
        boundary();ident=cell['cell_id'];proof=cc.read(c.resolve(item));firstpath=cc.OUT/'cell_audits'/ident/'worker/saved_audit.json'
        c.require(proof==cc.read(firstpath) and proof['status']=='passed' and proof['cell_id']==ident
            and proof['full_cohort_verified'] is True and proof['games']==16 and proof['model_calls']==0
            and proof['training_updates']==0 and proof['route']=='cpu' and proof['concurrency']==8
            and proof['seed_decimals']==[s['seed_decimal'] for s in cell['specifications']],
            'Same full cohort, original replays and exact saved repeated semantic audit')
        c.require(cc.read(firstpath.parent/'result.json')['result']==c.artifact(firstpath),
            'Cell auditor returns its exact saved proof')
    training=verify_training_qualification(boundary=boundary)
    union=c.artifact(c.OUT/'training_union.json');union_proof=corpus.verify_union_inventory(union,plan,boundary=boundary)
    returned=c.read(c.OUT/'union_build/worker/result.json')['result']
    c.require(returned==dict(status='accepted',inventory=union,verification=union_proof),
        'Exited union worker binds independent old/new split reconstruction')
    return dict(corpus_acceptance=acceptance_item,qualification=qual_item,training_qualification=training,
                training_union=union,union_verification=union_proof,independent_cell_audits=expected_audits)


def selection(plan,boundary):
    from . import corpus
    execution=c.read(c.OUT/'collection/corpus_audit/execution.json')
    c.require(execution['status']=='passed' and type(execution['worker_exit_code']) is int
        and execution['worker_exit_code']==0 and execution['cleanup_errors']==[] and execution['error'] is None
        and execution['plan_sha256']==c.sha(c.PLAN) and execution['mode']=='corpus' and execution['retries']==0,
        'Actual completed corpus audit before union construction')
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    for identity in execution['observed_owned_identities']+[execution['identity']]:require_absent(identity)
    return corpus.union_inventory(plan,boundary=boundary)


def preserve_evidence(folder,*,boundary=lambda:None):
    inventory=folder/'preserved_evidence.jsonl';count,total=0,0
    exclusions={c.OUT/name for name in ('stage.json','stage.tmp','runner_stdout.log','runner_stderr.log',
                                       'launch/stdout.log','launch/stderr.log')}
    with inventory.open('xb') as stream:
        for path in closed_files(c.OUT):
            if path.is_relative_to(folder.parent) or path in exclusions:continue
            boundary();before=path.stat();item=c.artifact(path);after=path.stat()
            c.require(before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,
                'Closed evidence changed during independent preservation')
            stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    c.require(total+inventory.stat().st_size<c.MAX_EVIDENCE_BYTES,'Actual final evidence size bound')
    return dict(physical_inventory=c.artifact(inventory),preserved_file_count=count,preserved_bytes=total)


def final(plan,folder,boundary):
    c.require('torch' not in sys.modules,'Final audit starts without a tensor runtime')
    c.verify(plan['inputs'],boundary=boundary);claim=c.read(c.OUT/'claim.json')
    phases=verify_phases(claim,boundary=boundary);verify_execution_inventory(folder,boundary=boundary)
    receipts=[];identities={};previous=(claim['monotonic_origin'],claim['unix_origin'])
    for job in expected_jobs():
        proof=verify_execution(job,plan,claim,phases,boundary=boundary)
        c.require(previous[0]<=proof['monotonic_origin'] and previous[1]<=proof['unix_origin'],
            'Only one successive root-owned worker, no hidden overlapping segment')
        previous=(proof['monotonic_end'],proof['unix_end']);receipts.append(proof)
        identities.update({(i['pid'],i['created']):i for i in proof['identities']})
    accepted=verify_collection_receipts(plan,boundary=boundary)
    preserved=preserve_evidence(folder,boundary=boundary)
    c.verify(plan['inputs'],boundary=boundary);boundary()
    c.require('torch' not in sys.modules,'Independent final audit ran no model or tensor loader')
    result=dict(status='passed',schema='collection4096_independent_acceptance_v1',
        original_freeze=c.artifact(c.PLAN),planned_predecessors=len(receipts),successful_actual_exits=receipts,
        exact_owned_identities_absent=list(identities.values()),**preserved,**accepted,
        new_scientific_source_games=c.NEW_GAMES,combined_training_games=4096,fresh_assessment_games=256,
        excluded_engineering_games=2,qualification_optimizer_calls=2,scientific_optimizer_calls=0,
        main_strength_games=0,models_run=0,games_run=0,no_automatic_followup=True,reference='original_ef8a',
        owner_exit_still_requires_observation=True,
        inventory_exclusions=['active final-audit subtree','mutable root stage.json and stage.tmp',
                              'retained root/launcher stdout/stderr'],
        execution_inventory_exclusions='Only frozen hash-authenticated prelaunch engineering evidence; bytes preserved')
    c.save(folder/'independent_acceptance.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('selection','final'),required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.output).resolve()
    c.require(folder==(c.OUT/('union_build' if args.mode=='selection' else 'final_audit')/'worker').resolve(),
        'Exact new owned model-free audit folder')
    from .runtime import bootstrap
    plan,boundary,sources=bootstrap(folder,args.plan_sha256,args.mode,model=False)
    status,error,result='failed',None,None
    try:
        result=selection(plan,boundary) if args.mode=='selection' else final(plan,folder,boundary)
        boundary();c.require('torch' not in sys.modules,'Saved-only worker loads no model runtime');status='passed'
    except BaseException:error=traceback.format_exc()
    c.save(folder/'result.json',dict(status=status,mode=args.mode,
        identity=dict(pid=os.getpid(),created=psutil.Process().create_time()),time=c.now(),
        result=result,error=error,plan_sha256=args.plan_sha256,boundary_calls=boundary.calls))
    if error:print(error,flush=True)
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
