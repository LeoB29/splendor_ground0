"""Independent acceptance for an explicitly amended, segmented collection.

The failed original remains failed. Completed original games and successful
checks are reused with authenticated bytes; only the named unfinished replay is
repeated. The original global/model/collection deadlines include the downtime.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import psutil
from . import common as c, clocks


def expected_jobs():
    module=c.PACKAGE+'.collection.worker'
    jobs=[]
    for mode,path,model in (
        ('saved_boundary','recovery_boundary',False),
        ('recovery_qualification','recovery_qualification',True),
        ('audit_recovery_qualification','recovery_qualification_audit',False),
        ('recovery_games','recovery_games',True),
        ('supplemental_audit','recovery_audit',False)):
        jobs.append(dict(path='collection/'+path,mode=mode,phase='supplemental',
            module=module,model=model,rss=c.COLLECTION_RSS,seconds=2*3600))
    for index in range(136,c.CELLS):
        for kind,mode,model in (('cells','games',True),('cell_audits','audit_cell',False)):
            jobs.append(dict(path=f'collection/{kind}/c{index:03d}',mode=mode,phase='collection',
                module=module,model=model,rss=c.COLLECTION_RSS,seconds=4*3600))
    jobs.extend([
        dict(path='collection/corpus_audit',mode='corpus',phase='corpus',module=module,
            model=False,rss=c.COLLECTION_RSS,seconds=24*3600),
        dict(path='union_build',mode='selection',phase='corpus',module=c.PACKAGE+'.audit',
            model=False,rss=c.FIT_RSS,seconds=4*3600)])
    return jobs


def expected_executions():
    return [c.OUT/job['path'] for job in expected_jobs()]


def closed_files(folder):
    pending=[Path(folder)]
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:rows=sorted(entries,key=lambda item:item.name)
        for entry in rows:
            c.require(not entry.is_symlink(),'No linked evidence paths')
            if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):yield Path(entry.path).resolve()


def verify_phases(claim,*,boundary=lambda:None):
    c.require(claim['freeze']==c.artifact(c.PLAN) and claim['plan_sha256']==c.sha(c.PLAN)
        and claim['total_seconds']==c.TOTAL_SECONDS and claim['model_seconds']==c.MODEL_SECONDS
        and claim['automatic_retries']==0,'Original bounded recovery claim')
    clocks.validate_claim(claim)
    c.require(claim['original_deadlines']==clocks.original_deadlines(), 'Authentic original UTC deadlines')
    launcher=c.read(c.OUT/'launcher_claim.json')
    for key in ('monotonic_origin','unix_origin','monotonic_deadline','unix_deadline',
        'recovery_anchor','recovery_anchor_sha256','original_deadlines','original_deadlines_sha256',
        'downtime_charged','per_decision_clocks_rebased'):
        c.require(claim[key]==launcher[key], 'Launcher/root identical reboot-safe clock '+key)
    phases={};previous=(claim['recovery_anchor']['monotonic'],claim['recovery_anchor']['unix'])
    for name in ('supplemental','collection','corpus','final'):
        boundary();path=c.OUT/'stages'/f'{name}_start.json';item=c.artifact(path);row=c.read(path)
        c.require(row['stage']==name and row['claim']==c.artifact(c.OUT/'claim.json'), 'Recovery phase identity')
        clocks.validate_phase(claim,row)
        if name!='collection':
            c.require(previous[0]<=row['monotonic']<row['monotonic_deadline'] and
                previous[1]<=row['unix_time']<row['unix_deadline'], 'Successive bounded phase ordering')
        end=None
        if name!='final':
            end=c.read(c.OUT/'stages'/f'{name}_end.json')
            c.require(end['stage']==name and end['start']==item
                and previous[0]<=end['monotonic']<row['monotonic_deadline']
                and previous[1]<=end['unix_time']<row['unix_deadline'], 'Completed phase inside original bounds')
            previous=(end['monotonic'],end['unix_time'])
        phases[name]=dict(start=row,start_item=item,end=end)
    return phases


def verify_execution(job,plan,claim,phases,*,boundary=lambda:None,absent=None):
    if absent is None:
        from tools.splendorust_gpu_probe_20261105.control import require_absent
        absent=require_absent
    boundary();container=c.OUT/job['path'];owner=c.read(container/'owner.json')
    value=c.read(container/'execution.json');phase=phases[job['phase']]['start']
    phase_item=phases[job['phase']]['start_item'];freeze=c.artifact(c.PLAN)
    c.require(value['status']=='passed' and type(value['worker_exit_code']) is int and value['worker_exit_code']==0
        and value['cleanup_errors']==[] and value['error'] is None and value['retries']==0,
        'Every recovery worker actually exited successfully without automatic retry')
    c.require(value['identity']==owner['identity'] and value['mode']==owner['mode']==job['mode']
        and value['fit_id'] is None and owner['fit_id'] is None
        and value['plan_sha256']==owner['plan_sha256']==freeze['sha256']
        and owner['windows_job_assigned'] is True and owner['model_work'] is job['model']
        and owner['global_claim']==c.artifact(c.OUT/'claim.json') and owner['phase_clock']==phase_item,
        'Exact actual recovery owner, role and freeze')
    c.require(type(owner['maximum_seconds']) is int and owner['maximum_seconds']==value['maximum_seconds']==job['seconds']
        and owner['maximum_rss_bytes']==job['rss'] and 0<=value['peak_rss_bytes']<=job['rss'],
        'Fixed time and actual RSS bounds')
    args=['-B','-u','-m',job['module'],'--mode',job['mode'],'--plan-sha256',freeze['sha256'],
        '--output',str(container/'worker')]
    cell=None
    if job['phase']=='collection':
        ident=container.name;args+=['--cell-id',ident]
        cell=c.read(c.OUT/'collection/cell_clocks'/f'{ident}.json')
        c.require(cell['cell_id']==ident and cell['phase']==phase_item
            and cell['monotonic_deadline']==cell['monotonic_origin']+4*3600
            and cell['unix_deadline']==cell['unix_origin']+4*3600, 'Unchanged four-hour new-cell allowance')
    c.require(type(owner['args']) is list and len(owner['args'])==len(args)+1 and owner['args'][1:]==args,
        'Exact scheduled worker entry point and arguments')
    for origin,end,deadline,phase_start in (('monotonic_origin','monotonic_end','monotonic_deadline','monotonic'),
        ('unix_origin','unix_end','unix_deadline','unix_time')):
        bound=min(owner[origin]+job['seconds'],phase[deadline])
        if cell is not None:
            bound=min(bound,cell[deadline])
            c.require(phase[phase_start]<=cell[origin]<=owner[origin], 'New cell within original collection phase')
        c.require(value[origin]==owner[origin] and value[deadline]==owner[deadline]==bound
            and phase[phase_start]<=owner[origin]<=value[end]<bound
            and value[end]<=phases[job['phase']]['end'][phase_start], 'Unextended saved execution clock chain')
        c.require(value[end]<claim[origin]+c.MODEL_SECONDS, 'Every predecessor before original model cutoff')
    identities=value['observed_owned_identities']+[value['identity']]
    for identity in identities:
        c.require(type(identity) is dict and set(identity)=={'pid','created'} and type(identity['pid']) is int
            and type(identity['created']) in (int,float), 'Exact process identity')
        absent(identity)
    startup=c.read(container/'worker/runtime_sources.json')
    c.require(startup['sources']==plan['corrected_source_receipt'] and startup['identity']==value['identity']
        and startup['original_claim']==owner['global_claim'] and startup['original_phase']==phase_item
        and startup['mode']==job['mode'] and startup['fit_id'] is None, 'Actual once-startup corrected source guard')
    result=c.read(container/'worker/result.json')
    c.require(result['status']=='passed' and result['mode']==job['mode'] and result['identity']==value['identity'],
        'Actual worker completion receipt')
    from tools.splendorust_gpu_probe_20261105.control import native_fallback_messages
    c.require(not native_fallback_messages(container/'worker/stderr.log'), 'No unsupported native fallback')
    return dict(execution=c.artifact(container/'execution.json'),owner=c.artifact(container/'owner.json'),
        result=c.artifact(container/'worker/result.json'),identities=identities,
        monotonic_origin=value['monotonic_origin'],monotonic_end=value['monotonic_end'],
        unix_origin=value['unix_origin'],unix_end=value['unix_end'])


def verify_execution_inventory(folder,*,boundary=lambda:None):
    expected={p/'execution.json' for p in expected_executions()};actual=set()
    engineering=c.read(c.PLAN).get('prelaunch_evidence',[]);c.verify(engineering,boundary=boundary)
    engineering_paths={c.resolve(item) for item in engineering}
    engineering_roots=(c.OUT/'prelaunch',c.OUT/'test_artifacts')
    c.require(len(engineering_paths)==len(engineering) and
        all(any(path.is_relative_to(root) for root in engineering_roots) for path in engineering_paths),
        'Exact frozen preparation and preserved first-fixture exclusions only')
    for path in closed_files(c.OUT):
        boundary()
        if path.is_relative_to(folder.parent):continue
        if any(path.is_relative_to(root) for root in engineering_roots):
            c.require(path in engineering_paths, 'No unbound post-freeze prelaunch evidence');continue
        c.require(path.name not in ('failure.json','worker_failure.json'), 'No hidden new failure')
        if path.name!='execution.json':continue
        value=c.read(path)
        if path in expected or any(k in value for k in ('worker_exit_code','observed_owned_identities','plan_sha256')):
            actual.add(path)
    c.require(actual==expected, 'All and only151 planned successful new predecessor executions')
    return actual


def verify_original_prefix(plan,*,boundary=lambda:None):
    from tools.splendorust_collection4096_20261115 import audit as original_audit
    from tools.splendorust_collection4096_20261115.collection import audit as original_cell_audit
    from .collection import common as cc
    c.require(plan['original_freeze']==c.artifact(c.ORIGINAL_OUT/'freeze.json')
        and plan['original_freeze']['sha256']==c.ORIGINAL_FREEZE_SHA, 'Exact preserved original study')
    original_plan=cc.read(c.ORIGINAL_OUT/'collection/plan.json')
    items=[];last_end=None;identities={}
    jobs=original_audit.expected_jobs()[:85]
    c.require(len(jobs)==85 and jobs[-1]['path']=='collection/cell_audits/c040', 'Exact completed original prefix')
    for job in jobs:
        boundary();container=c.ORIGINAL_OUT/job['path']
        value,absent=original_cell_audit.stage_execution(original_plan,container,mode=job['mode'])
        owner=c.read(container/'owner.json')
        c.require(owner['maximum_seconds']==job['seconds'] and owner['maximum_rss_bytes']==job['rss']
            and owner['model_work'] is job['model'], 'Original qualification/prefix roles unchanged')
        args=['-B','-u','-m',job['module'],'--mode',job['mode'],'--plan-sha256',c.ORIGINAL_FREEZE_SHA,
            '--output',str(container/'worker')]
        if job['phase']=='collection':args+=['--cell-id',container.name]
        c.require(owner['args'][1:]==args, 'Exact original scheduled entry point')
        if last_end is not None:
            c.require(last_end[0]<=value['monotonic_origin'] and last_end[1]<=value['unix_origin'],
                'Original successful prefix serial ordering')
        last_end=(value['monotonic_end'],value['unix_end'])
        items.append(c.artifact(container/'execution.json'))
        identities.update({(i['pid'],i['created']):i for i in absent})
    failed=c.read(c.ORIGINAL_OUT/'collection/cells/c041/execution.json')
    c.require(failed['status']=='failed' and failed['worker_exit_code']==1073807364
        and failed['cleanup_errors']==[] and failed['retries']==0, 'Original c041 failure remains a failure')
    c.require(not (c.ORIGINAL_OUT/'completion.json').exists()
        and c.read(c.ORIGINAL_OUT/'stage.json')['status']=='incomplete'
        and (c.ORIGINAL_OUT/'failure.json').is_file(), 'Original incomplete study is never relabelled')
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    for identity in failed['observed_owned_identities']+[failed['identity']]:
        require_absent(identity);identities[(identity['pid'],identity['created'])]=identity
    observer=c.ROOT/'outputs/splendorust_collection4096_launch_20261115_01/observed_launcher_exit.json'
    c.require(c.artifact(observer) in plan['inputs'], 'Original actual launcher-exit receipt frozen')
    ended=c.read(observer)
    c.require(ended['status']=='failed' and ended['actual_launcher_exit_code']==1
        and ended['actual_launcher_exit_observed'] is True and ended['exact_launcher_identity_absent'] is True,
        'Original launcher actually exited after failure')
    training=original_audit.verify_training_qualification(boundary=boundary)
    return dict(successful_actual_exits=items,failed_execution=c.artifact(c.ORIGINAL_OUT/'collection/cells/c041/execution.json'),
        original_attempt_status='incomplete',actual_launcher_exit=c.artifact(observer),
        exact_owned_identities_absent=list(identities.values()),training_qualification=training)


def verify_previous_prefix(plan,*,boundary=lambda:None):
    """Reuse actual successful predecessor jobs, without repairing their failed owner."""
    from tools.splendorust_collection4096_recovery_20261116 import audit as old_audit
    from tools.splendorust_collection4096_recovery_20261116.collection import audit as old_cell_audit
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    from .collection import common as cc
    frozen=c.artifact(c.PREVIOUS_OUT/'freeze.json')
    c.require(plan['previous_freeze']==frozen and frozen['sha256']==c.PREVIOUS_FREEZE_SHA,
        'Exact preserved first recovery freeze')
    collection=cc.read(c.PREVIOUS_OUT/'collection/plan.json')
    c.require(collection==cc.read(cc.PLAN),'Original scientific collection plan unchanged')
    jobs=old_audit.expected_jobs()[:191]
    c.require(len(jobs)==191 and jobs[-1]['path']=='collection/cell_audits/c134',
        'Five technical jobs and93 complete prior recovery cells')
    items=[];identities={};previous=None
    for job in jobs:
        boundary();container=c.PREVIOUS_OUT/job['path']
        value,absent=old_cell_audit.stage_execution(collection,container,mode=job['mode'])
        owner=c.read(container/'owner.json')
        c.require(owner['maximum_seconds']==job['seconds'] and owner['maximum_rss_bytes']==job['rss']
            and owner['model_work'] is job['model'],'Prior job role and actual limits unchanged')
        args=['-B','-u','-m',job['module'],'--mode',job['mode'],'--plan-sha256',frozen['sha256'],
            '--output',str(container/'worker')]
        if job['phase']=='collection':args+=['--cell-id',container.name]
        c.require(owner['args'][1:]==args,'Prior exact owned entrypoint and arguments')
        if previous is not None:
            c.require(previous[0]<=value['monotonic_origin'] and previous[1]<=value['unix_origin'],
                'Prior successful jobs retain serial order')
        previous=(value['monotonic_end'],value['unix_end'])
        items.append(c.artifact(container/'execution.json'))
        identities.update({(i['pid'],i['created']):i for i in absent})
    failed=c.read(c.PREVIOUS_OUT/'collection/cells/c135/execution.json')
    c.require(failed['status']=='failed' and failed['worker_exit_code']==1 and failed['retries']==0
        and failed['error'] and failed['cleanup_errors']==[],
        'Recorded failed c135 process execution is preserved as failed')
    cleanup=c.read(c.PREVIOUS_OUT/'collection/cells/c135/worker/cleanup.json')
    c.require(cleanup['success'] is False and cleanup['errors'],
        'Incomplete original operation closure remains failed despite process cleanup')
    c.require(not (c.PREVIOUS_OUT/'completion.json').exists()
        and c.read(c.PREVIOUS_OUT/'stage.json')['status']=='incomplete'
        and (c.PREVIOUS_OUT/'failure.json').is_file() and (c.PREVIOUS_OUT/'launcher_failure.json').is_file(),
        'First recovery remains incomplete with original failures')
    for identity in failed['observed_owned_identities']+[failed['identity']]:
        require_absent(identity);identities[(identity['pid'],identity['created'])]=identity
    observer=c.ROOT/'outputs/splendorust_collection4096_recovery_launch_20261116_01/observed_launcher_exit.json'
    c.require(c.artifact(observer) in plan['inputs'],'Prior actual launcher exit frozen')
    ended=c.read(observer)
    c.require(ended['status']=='failed' and ended['actual_launcher_exit_code']==1
        and ended['actual_launcher_exit_observed'] is True and ended['exact_launcher_identity_absent'] is True,
        'Prior recovery launcher actually exited unsuccessfully')
    gate=c.read(c.PREVIOUS_OUT/'recovery_gate.json')
    c.require(gate['status']=='passed' and gate['actual_exits']==items[:5]
        and gate['gameplay_retries']==0 and gate['verification_retries']==1
        and gate['recovered_cell_audit']==c.artifact(c.PREVIOUS_OUT/'collection/recovery_audit/worker/saved_audit.json'),
        'Previous verification-only amendment remains separately accounted')
    return dict(successful_actual_exits=items,previous_freeze=frozen,
        failed_execution=c.artifact(c.PREVIOUS_OUT/'collection/cells/c135/execution.json'),
        failed_operation_cleanup=c.artifact(c.PREVIOUS_OUT/'collection/cells/c135/worker/cleanup.json'),
        previous_attempt_status='incomplete',actual_launcher_exit=c.artifact(observer),
        exact_owned_identities_absent=list(identities.values()),prior_recovery_gate=c.artifact(c.PREVIOUS_OUT/'recovery_gate.json'))


def verify_collection_receipts(plan,*,boundary=lambda:None):
    from .collection import common as cc
    from . import corpus
    collection=cc.read(cc.PLAN)
    c.require(plan['collection_plan']==c.artifact(cc.PLAN), 'Exact scientific plan')
    gate=c.read(c.OUT/'recovery_gate.json')
    c.require(gate['status']=='passed' and gate['plan']==c.artifact(c.PLAN)
        and gate['actual_exits']==[c.artifact(p/'execution.json') for p in expected_executions()[:5]]
        and gate['original_attempt_status']=='incomplete' and gate['new_games_released']==0
        and gate['gameplay_retries']==8 and gate['verification_retries']==1
        and gate['new_verification_retries']==0 and gate['resumed_games']==8
        and gate['preserved_completed_games']==2168 and gate['previous_attempt_status']=='incomplete'
        and gate['previous_recovery_freeze']==c.artifact(c.PREVIOUS_OUT/'freeze.json')
        and gate['new_fixture_decisions']==116,
        'Five actual technical qualification/recovery jobs before previously unstarted games')
    c.require(gate['original_qualification']==c.artifact(c.ORIGINAL_OUT/'qualification_gate.json')
        and gate['original_collection_qualification']==c.artifact(c.ORIGINAL_OUT/'collection/qualification_gate.json')
        and gate['recovered_cell_audit']==c.artifact(cc.OUT/'recovery_audit/worker/saved_audit.json'),
        'Original qualifications and separately resumed c135 audit are bound explicitly')
    qualifier=cc.read(cc.OUT/'recovery_qualification_audit/worker/saved_audit.json')
    c.require(qualifier['status']=='passed' and qualifier['passed'] is True and qualifier['complete_decisions']==116
        and qualifier['new_games']==qualifier['training_updates']==qualifier['model_calls']==0
        and qualifier['collection_plan']==c.artifact(cc.PLAN), 'Actual recovery entry-point qualification before replay')
    acceptance_item=c.artifact(cc.OUT/'corpus_acceptance.json');corpus._receipt(acceptance_item,boundary)
    acceptance=cc.read(c.resolve(acceptance_item))
    c.require(acceptance['engineering_games_excluded']==2 and acceptance['teacher_simulations']==512
        and acceptance['raw_semantics_and_all_transitions_verified'] is True
        and acceptance['all_original_callbacks_attributed'] is True and acceptance['selected_model_replays_verified'] is True
        and acceptance['models_run']==acceptance['games_run']==0, 'Complete recovered corrected corpus')
    c.require(cc.read(cc.OUT/'corpus_audit/worker/saved_audit.json')==acceptance
        and cc.read(cc.OUT/'corpus_audit/worker/result.json')['result']==
            c.artifact(cc.OUT/'corpus_audit/worker/saved_audit.json'), 'Actual corpus worker binds exact acceptance')
    audits=[c.artifact(cc.OUT/'corpus_cell_audits'/f'c{i:03d}.json') for i in range(c.CELLS)]
    c.require(acceptance['independent_cell_audits']==audits and len(collection['cells'])==c.CELLS,
        'All208 fixed cells independently authenticated before corpus release')
    for cell,item in zip(collection['cells'],audits,strict=True):
        boundary();proof=cc.read(c.resolve(item));first=corpus.first_audit_path(cell['cell_id'])
        c.require(proof==cc.read(first) and proof['status']=='passed' and proof['cell_id']==cell['cell_id']
            and proof['full_cohort_verified'] is True and proof['games']==16
            and proof['model_calls']==proof['training_updates']==0 and proof['route']=='cpu' and proof['concurrency']==8
            and proof['seed_decimals']==[s['seed_decimal'] for s in cell['specifications']], 'Full exact original cohort and checks')
        c.require(cc.read(first.parent/'result.json')['result']==c.artifact(first), 'Actual independent audit return')
        if cell['cell_id']=='c041':
            c.require(proof['recovered_verification'] is True, 'c041 retains prior amended verification')
        if cell['cell_id']=='c135':
            c.require(proof['resumed_gameplay'] is True and proof['gameplay_retries']==8,
                'c135 explicitly accounts for eight resumed failed or cancelled gameplay decisions')
    union=c.artifact(c.OUT/'training_union.json');union_proof=corpus.verify_union_inventory(union,plan,boundary=boundary)
    c.require(c.read(c.OUT/'union_build/worker/result.json')['result']==
        dict(status='accepted',inventory=union,verification=union_proof), 'Exited union worker binds original/new source reconstruction')
    return dict(corpus_acceptance=acceptance_item,recovery_gate=c.artifact(c.OUT/'recovery_gate.json'),
        training_union=union,union_verification=union_proof,independent_cell_audits=audits)


def _verify_preservation_inventory(item,root,*,boundary=lambda:None):
    c.verify([item],boundary=boundary);seen=set();total=0
    with c.resolve(item).open('r',encoding='utf-8') as stream:
        for line in stream:
            boundary();entry=json.loads(line);path=c.resolve(entry)
            c.require(path.is_relative_to(root) and path not in seen,
                'Unique predecessor-only preservation member')
            c.verify([entry],boundary=boundary);seen.add(path);total+=entry['bytes']
    actual=set(closed_files(root))
    c.require(seen and seen==actual,'Exact predecessor evidence file set remains byte-identical')
    return dict(inventory=item,preserved_files=len(seen),preserved_bytes=total,
        failed_attempt_preserved=True,all_original_bytes_unchanged=True)


def verify_original_preservation(plan,*,boundary=lambda:None):
    return _verify_preservation_inventory(plan['original_preservation_inventory'],c.ORIGINAL_OUT,boundary=boundary)


def verify_previous_preservation(plan,*,boundary=lambda:None):
    return _verify_preservation_inventory(plan['previous_preservation_inventory'],c.PREVIOUS_OUT,boundary=boundary)


def preserve_evidence(folder,plan,*,boundary=lambda:None):
    original=verify_original_preservation(plan,boundary=boundary)
    previous=verify_previous_preservation(plan,boundary=boundary)
    inventory=folder/'preserved_evidence.jsonl';count,total=0,0
    exclusions={c.OUT/name for name in ('stage.json','stage.tmp','runner_stdout.log','runner_stderr.log',
        'launch/stdout.log','launch/stderr.log')}
    with inventory.open('xb') as stream:
        for path in closed_files(c.OUT):
            if path.is_relative_to(folder.parent) or path in exclusions:continue
            boundary();before=path.stat();item=c.artifact(path);after=path.stat()
            c.require(before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,
                'Closed recovery evidence unchanged during final preservation')
            stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    c.require(original['preserved_bytes']+previous['preserved_bytes']+total+inventory.stat().st_size<c.MAX_EVIDENCE_BYTES,
        'Actual original-plus-both-recoveries evidence remains below original96GiB cap')
    return dict(original_preservation=original,previous_preservation=previous,physical_inventory=c.artifact(inventory),
        preserved_file_count=count,preserved_bytes=total,
        combined_preserved_bytes=original['preserved_bytes']+previous['preserved_bytes']+total)


def final(plan,folder,boundary):
    c.require('torch' not in sys.modules, 'Model-free independent final audit')
    c.verify(plan['inputs'],boundary=boundary);claim=c.read(c.OUT/'claim.json')
    original=verify_original_prefix(plan,boundary=boundary)
    previous_recovery=verify_previous_prefix(plan,boundary=boundary)
    phases=verify_phases(claim,boundary=boundary);verify_execution_inventory(folder,boundary=boundary)
    receipts=[];identities={};previous=(claim['recovery_anchor']['monotonic'],claim['recovery_anchor']['unix'])
    for job in expected_jobs():
        proof=verify_execution(job,plan,claim,phases,boundary=boundary)
        c.require(previous[0]<=proof['monotonic_origin'] and previous[1]<=proof['unix_origin'],
            'One successive owned recovery worker, no hidden overlap')
        previous=(proof['monotonic_end'],proof['unix_end']);receipts.append(proof)
        identities.update({(i['pid'],i['created']):i for i in proof['identities']})
    accepted=verify_collection_receipts(plan,boundary=boundary)
    preserved=preserve_evidence(folder,plan,boundary=boundary)
    c.verify(plan['inputs'],boundary=boundary);boundary()
    c.require('torch' not in sys.modules, 'No learned model or tensor loader during final audit')
    result=dict(status='passed',schema='collection4096_resumed_independent_acceptance_v1',
        original_freeze=plan['original_freeze'],recovery_freeze=c.artifact(c.PLAN),
        original_attempt_status='incomplete',amended_recovered_verification=True,
        original_prefix=original,previous_recovery_prefix=previous_recovery,
        previous_recovery_freeze=plan['previous_freeze'],previous_recovery_attempt_status='incomplete',
        planned_predecessors=len(receipts),successful_actual_exits=receipts,
        exact_owned_identities_absent=list(identities.values()),**preserved,**accepted,
        new_scientific_source_games=c.NEW_GAMES,combined_training_games=4096,fresh_assessment_games=256,
        original_completed_games=672,previous_completed_games=1496,preserved_completed_games=2168,
        resumed_games=8,fresh_unstarted_games=1152,newly_completed_games=1160,
        gameplay_retries=8,verification_retries=1,new_verification_retries=0,
        repeated_verification=dict(game_id='g0666',ply=40),prior_supplemental_verification_allowance_seconds=3600,
        partial_cell_recovery_allowance_seconds=2*3600,
        original_utc_deadlines_unchanged=True,downtime_charged=True,
        excluded_engineering_games=2,qualification_optimizer_calls=2,scientific_optimizer_calls=0,
        main_strength_games=0,models_run=0,games_run=0,no_automatic_followup=True,reference='original_ef8a',
        owner_exit_still_requires_observation=True,
        inventory_exclusions=['active final-audit subtree','mutable recovery root stage.json and stage.tmp',
            'retained recovery root/launcher stdout/stderr'],
        execution_inventory_exclusions='Exact frozen prelaunch technical evidence only; original failure separately preserved')
    c.save(folder/'independent_acceptance.json',result);return result


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
