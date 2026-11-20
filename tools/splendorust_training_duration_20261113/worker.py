"""One guarded fixed role per fresh owned worker."""
import argparse
import time
import traceback
from . import common as c
from . import control

def passed_job(role):
    folder=c.OUT/'jobs'/role
    item=c.artifact(folder/'actual_exit.json');execution=c.read(c.resolve(item))
    result=c.read(folder/'worker/result.json')
    c.require(execution['result_artifact']==c.artifact(folder/'worker/result.json'),'Prior actual exit binds exact result artifact')
    c.require(execution['status']=='passed' and type(execution['exit_code']) is int and execution['exit_code']==0
        and execution['actual_exit_observed'] and execution['exact_owned_absence'] and not execution['cleanup_errors']
        and execution['error'] is None and execution['role']==role and execution['freeze_sha256']==c.sha(c.PLAN),
        'Actual clean prior execution: '+role)
    c.require(result['status']=='passed' and result['error'] is None and result['role']==role and result['identity']==execution['identity']
        and result['freeze_sha256']==c.sha(c.PLAN),'Passed exact prior result: '+role)
    c.require(not c.exact_alive(execution['identity']) and all(not c.exact_alive(i) for i in execution['observed_owned_identities']),
        'Prior owned processes absent')
    return item,result

def commit(folder,boundary):
    from . import fit
    selection=c.read(c.SELECTION);proofs=[];fits={}
    for rep in (0,1):
        execution,_=passed_job(f'fit_r{rep}')
        verified=fit.verify_fit(c.OUT/'jobs'/f'fit_r{rep}'/'worker',boundary=boundary)
        c.require(verified['repetition']==rep and verified['optimizer_calls']==5346 and verified['replay_optimizer_calls']==3,'Fixed continuation count')
        fits[rep]=verified
        proofs.append(dict(repetition=rep,execution=execution,verification=c.save(folder/f'verified_r{rep}.json',verified)))
    models=[]
    for passes in (2,4):
        for rep in (0,1):
            checkpoint=fits[rep]['checkpoints'][str(passes*1782)]
            meta=c.read(c.resolve(checkpoint))['metadata']
            models.append(dict(model_id=f'r{rep}_pass{passes}',repetition=rep,passes=passes,
                checkpoint=checkpoint,weights_sha256=meta['weights_sha256']))
    return dict(commitment=c.save(c.COMMITMENT,dict(status='all_duration_checkpoints_committed',study_id=c.STUDY,
        checkpoint_selection=False,selection_sha256=c.digest(selection),original_commitment=c.artifact(c.ORIGINAL_COMMITMENT),
        models=models,fits=proofs,scientific_new_updates=10692,replay_updates=6,**c.IDENTITY)))

def acceptance(folder,boundary,plan):
    from . import fit,offline
    executions=[passed_job(role)[0] for role in c.JOBS if role!='acceptance']
    qualification=fit.verify_qualification(c.OUT/'jobs/qualification/worker',boundary=boundary)
    fits=[fit.verify_fit(c.OUT/'jobs'/f'fit_r{rep}'/'worker',boundary=boundary) for rep in (0,1)]
    c.require(qualification['optimizer_calls']==6 and sum(r['optimizer_calls'] for r in fits)==10692 and
        sum(r['replay_optimizer_calls'] for r in fits)==6,'All actual optimizer calls counted')
    selection=c.read(c.SELECTION)
    c.require(passed_job('commit')[1]['result']['commitment']==c.artifact(c.COMMITMENT),'Accepted commitment artifact unchanged')
    c.require(passed_job('offline')[1]['result']['evaluation']==c.artifact(c.OUT/'jobs/offline/worker/evaluation_result.json'),
        'Accepted evaluation artifact unchanged')
    c.require(passed_job('analysis')[1]['result']['analysis']==c.artifact(c.OUT/'jobs/analysis/worker/analysis.json'),
        'Accepted analysis artifact unchanged')
    args=(selection,c.artifact(c.ORIGINAL_COMMITMENT),c.artifact(c.ORIGINAL_EVALUATION),c.artifact(c.COMMITMENT),
        c.artifact(c.OUT/'jobs/offline/worker/evaluation_result.json'))
    bundles,_=offline.load_predictions(*args,boundary=boundary)
    labels=offline.source_labels(selection,boundary=boundary)
    analysis=c.read(c.OUT/'jobs/analysis/worker/analysis.json')
    arithmetic=offline.verify_arithmetic(bundles,labels,analysis,boundary=boundary)
    c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
    commitment=c.read(c.COMMITMENT)
    for row in commitment['fits']:c.verify([row['execution'],row['verification']],boundary=boundary)
    charts=passed_job('analysis')[1]['result']['charts'];c.verify(charts,boundary=boundary)
    result=dict(approved=True,status='duration_independent_acceptance_passed',all_prior_actual_exits=executions,
        source_preservation=True,qualification=qualification,fits=fits,arithmetic=arithmetic,
        checkpoint_commitment=c.artifact(c.COMMITMENT),analysis=c.artifact(c.OUT/'jobs/analysis/worker/analysis.json'),
        charts=charts,new_games=0,scientific_updates=10692,qualification_updates=6,replay_updates=6,
        automatic_promotion=False,automatic_followup=False,time=c.now())
    return dict(acceptance=c.save(folder/'independent_acceptance.json',result))

def run(role,expected):
    folder=c.OUT/'jobs'/role/'worker';folder.mkdir(parents=True,exist_ok=False)
    error=None;result=None
    try:
        owner=control.handshake(folder.parent,expected,role);boundary=control.Boundary(folder,owner);boundary()
        c.require(c.sha(c.PLAN)==expected,'Exact worker freeze');plan=c.read(c.PLAN)
        c.require(plan['design']==c.design(),'Unchanged fixed design')
        c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
        from tools.future_training_rules_20261021 import verify_runtime_sources
        sources=verify_runtime_sources()
        c.require(sources==plan['corrected_source_receipt'],'Exact corrected runtime source receipt')
        c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),role=role,**c.IDENTITY))
        if role!='qualification':passed_job('qualification')
        if role=='qualification':
            from . import fit
            result=fit.run_qualification(folder,boundary=boundary,runtime_sources=sources)
        elif role in ('fit_r0','fit_r1'):
            from . import fit
            rep=int(role[-1]);schedule=fit.build_schedules(c.read(c.SELECTION),rep)
            c.require(c.digest(schedule)==plan['schedule_sha256'][str(rep)],'Prospective fixed full-pass schedules')
            result=fit.run_fit(rep,folder,boundary=boundary,runtime_sources=sources)
        elif role=='commit':result=commit(folder,boundary)
        elif role in ('offline','analysis'):
            from . import offline
            c.require(passed_job('commit')[1]['result']['commitment']==c.artifact(c.COMMITMENT),'Passed commit binds exact checkpoint inventory')
            args=(c.read(c.SELECTION),c.artifact(c.ORIGINAL_COMMITMENT),c.artifact(c.ORIGINAL_EVALUATION),c.artifact(c.COMMITMENT))
            if role=='offline':
                result=dict(evaluation=offline.run_evaluation(*args,folder,boundary=boundary,runtime_sources=sources))
            else:
                c.require(passed_job('offline')[1]['result']['evaluation']==c.artifact(c.OUT/'jobs/offline/worker/evaluation_result.json'),
                    'Passed evaluator binds exact saved prediction inventory')
                result=offline.run_analysis(*args,c.artifact(c.OUT/'jobs/offline/worker/evaluation_result.json'),folder,boundary=boundary)
        elif role=='acceptance':result=acceptance(folder,boundary,plan)
        else:raise ValueError('Unregistered role')
        boundary()
    except BaseException:error=traceback.format_exc()
    receipt=dict(status='passed' if error is None else 'failed',role=role,identity=c.identity(),
        freeze_sha256=expected,result=result,error=error,time=c.now())
    c.save(folder/'result.json',receipt)
    if error:
        c.save(folder/'failure.json',receipt);print(error,flush=True);return 1
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=c.JOBS,required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    raise SystemExit(run(a.role,a.freeze_sha256))
