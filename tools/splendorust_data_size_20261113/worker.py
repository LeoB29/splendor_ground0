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
    from . import fit,offline
    selected=c.read(c.SELECTION);proofs=[];fits={}
    for games in (256,512):
        for rep in (0,1):
            role=f'g{games}_r{rep}';execution,_=passed_job(role)
            verified=fit.verify_fit(c.OUT/'jobs'/role/'worker',boundary=boundary)
            c.require(verified['game_count']==games and verified['repetition']==rep and verified['optimizer_calls']==1782
                and verified['replay_optimizer_calls']==2,'Fixed fit and reproduction inventory')
            fits[games,rep]=verified
            proofs.append(dict(game_count=games,repetition=rep,execution=execution,
                verification=c.save(folder/f'verified_{role}.json',verified)))
    models=[]
    for name in offline.FRESH_IDS:
        game_text,rep_text,endpoint=name.split('_');games,rep=int(game_text[1:]),int(rep_text[1:])
        verified=fits[games,rep];step=verified['endpoint_steps'][endpoint]
        checkpoint=verified['checkpoints'][str(step)];meta=c.read(c.resolve(checkpoint))['metadata']
        models.append(dict(model_id=name,game_count=games,repetition=rep,endpoint=endpoint,
            checkpoint_step=step,checkpoint=checkpoint,weights_sha256=meta['weights_sha256']))
    return dict(commitment=c.save(c.COMMITMENT,dict(status='all_data_size_checkpoints_committed',study_id=c.STUDY,
        checkpoint_selection=False,selection_sha256=c.digest(selected),source_selection=c.artifact(c.SOURCE_SELECTION),
        original_commitment=c.artifact(c.ORIGINAL_COMMITMENT),models=models,fits=proofs,
        scientific_new_updates=7128,replay_updates=8,**c.IDENTITY)))


def acceptance(folder,boundary,plan):
    from . import fit,offline,placement
    executions=[passed_job(role)[0] for role in c.JOBS if role!='acceptance']
    qualification=fit.verify_qualification(c.OUT/'jobs/qualification/worker',boundary=boundary)
    fits=[fit.verify_fit(c.OUT/'jobs'/role/'worker',boundary=boundary) for role in c.FIT_IDS]
    c.require(qualification['optimizer_calls']==6 and sum(r['optimizer_calls'] for r in fits)==7128 and
        sum(r['replay_optimizer_calls'] for r in fits)==8,'All actual optimizer calls counted')
    selected=c.read(c.SELECTION)
    c.require(passed_job('commit')[1]['result']['commitment']==c.artifact(c.COMMITMENT),'Exact accepted endpoint commitment')
    evaluation=c.artifact(c.OUT/'jobs/offline/worker/evaluation_result.json')
    c.require(passed_job('offline')[1]['result']['evaluation']==evaluation,'Exact accepted prediction inventory')
    analysis_item=c.artifact(c.OUT/'jobs/analysis/worker/analysis.json')
    c.require(passed_job('analysis')[1]['result']['analysis']==analysis_item,'Exact accepted analysis artifact')
    args=(selected,c.artifact(c.ORIGINAL_COMMITMENT),c.artifact(c.ORIGINAL_EVALUATION),c.artifact(c.COMMITMENT),evaluation)
    bundles,_=offline.load_predictions(*args,boundary=boundary)
    labels=offline.source_labels(selected,boundary=boundary)
    arithmetic=offline.verify_arithmetic(bundles,labels,selected,c.read(c.resolve(analysis_item)),boundary=boundary)
    c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
    commitment=c.read(c.COMMITMENT)
    for row in commitment['fits']:c.verify([row['execution'],row['verification']],boundary=boundary)
    charts=passed_job('analysis')[1]['result']['charts'];c.verify(charts,boundary=boundary)
    protection=placement.preserve_calibration(boundary=boundary)
    result=dict(approved=True,status='data_size_independent_acceptance_passed',all_prior_actual_exits=executions,
        source_preservation=True,qualification=qualification,fits=fits,arithmetic=arithmetic,
        checkpoint_commitment=c.artifact(c.COMMITMENT),analysis=analysis_item,charts=charts,
        calibration_preservation=protection,new_games=0,scientific_updates=7128,qualification_updates=6,replay_updates=8,
        automatic_promotion=False,automatic_followup=False,time=c.now())
    return dict(acceptance=c.save(folder/'independent_acceptance.json',result))

def run(role,expected):
    folder=c.OUT/'jobs'/role/'worker';folder.mkdir(parents=True,exist_ok=False)
    error=None;result=None
    try:
        owner=control.handshake(folder.parent,expected,role)
        from . import placement
        placement.set_own(c.SUPERVISOR_CPU)
        boundary=control.Boundary(folder,owner);boundary()
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
            verification=fit.verify_qualification(folder,boundary=boundary)
            c.save(folder/'qualification_saved_verification.json',verification)
        elif role in c.FIT_IDS:
            from . import fit,selection
            games,rep=int(role.split('_')[0][1:]),int(role.split('_')[1][1:])
            schedule=selection.build_schedules(c.read(c.SELECTION),rep,games)
            c.require(schedule==c.read(c.resolve(plan['schedules'][role])),'Prospective exact fit schedule')
            result=fit.run_fit(games,rep,folder,boundary=boundary,runtime_sources=sources)
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
