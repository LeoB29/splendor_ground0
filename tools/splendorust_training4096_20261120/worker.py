"""Single-use corrected-rules workers with fixed dependencies and saved acceptance."""
import argparse,traceback
from . import common as c,control

def dependencies(role,boundary=lambda:None):
    c.require(role in c.JOBS,'Fixed worker role')
    return [control.passed_job(prior,boundary=boundary)[0] for prior in c.JOBS[:c.JOBS.index(role)]]

def acceptance(folder,boundary,plan):
    from . import fit,offline,seal,corpus
    executions=dependencies('acceptance',boundary)
    qualification=fit.verify_qualification(c.OUT/'jobs/qualification/worker',boundary=boundary)
    fits=[fit.verify_fit(c.OUT/'jobs'/role/'worker',boundary=boundary) for role in c.FIT_IDS]
    c.require(qualification['optimizer_calls']<=16 and sum(x['optimizer_calls'] for x in fits)==28404
        and sum(x['replay_optimizer_calls'] for x in fits)==4,'Exact bounded optimizer accounting')
    committed=seal.validate_commitment(boundary=boundary)
    evaluation=offline.verify_evaluation(boundary=boundary)
    analysis=offline.verify_analysis(boundary=boundary)
    preserved_sources=corpus.verify_consumed_sources(folder,boundary=boundary)
    c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
    for cache in (c.TRAIN_CACHE,c.ASSESS_CACHE):
        c.require((cache/'cache_acceptance.json').is_file(),'Both independent cache acceptances exist')
    result=dict(status='training4096_independent_acceptance_passed',approved=True,
        executions=executions,qualification=qualification,fits=fits,checkpoint_commitment=c.artifact(c.COMMITMENT),
        evaluation=evaluation,analysis=analysis,source_preservation=True,consumed_source_preservation=preserved_sources,new_games=0,
        scientific_updates=28404,replay_updates=4,qualification_updates=qualification['optimizer_calls'],
        automatic_promotion=False,automatic_followup=False,time=c.now())
    return dict(acceptance=c.save(folder/'independent_acceptance.json',result))

def dispatch(role,folder,boundary,sources,plan):
    from . import fit,selection,corpus,seal,offline
    if role=='qualification':
        result=fit.run_qualification(folder,boundary=boundary,runtime_sources=sources)
        c.save(folder/'qualification_saved_verification.json',fit.verify_qualification(folder,boundary=boundary))
        return result
    if role=='training_cache':return corpus.build_training(folder,boundary=boundary)
    if role=='training_cache_audit':return corpus.audit_training(folder,boundary=boundary)
    if role=='schedules':return selection.verify_schedules(folder,boundary=boundary)
    if role in c.FIT_IDS:
        fit.run_fit(role,folder,boundary=boundary,runtime_sources=sources)
        return dict(fit_result=c.artifact(folder/'fit_result.json'))
    if role=='commit':return seal.commit(folder,boundary=boundary)
    if role=='assessment_cache':return corpus.build_assessment(folder,boundary=boundary)
    if role=='assessment_cache_audit':return corpus.audit_assessment(folder,boundary=boundary)
    if role=='offline':return dict(evaluation=offline.run_evaluation(folder,boundary=boundary,runtime_sources=sources))
    if role=='analysis':return offline.run_analysis(folder,boundary=boundary)
    if role=='acceptance':return acceptance(folder,boundary,plan)
    raise ValueError('Unknown worker role')

def run(role,expected):
    folder=c.OUT/'jobs'/role/'worker';folder.mkdir(parents=True,exist_ok=False)
    error=None;result=None
    try:
        owner=control.handshake(folder.parent,expected,role)
        from . import placement
        placement.set_own(c.SUPERVISOR_CPU)
        boundary=control.Boundary(folder,owner);boundary()
        c.require(c.sha(c.PLAN)==expected,'Exact worker execution freeze');plan=c.read(c.PLAN)
        c.require(plan['design']==c.design(),'Unchanged fixed design')
        c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
        from tools.future_training_rules_20261021 import verify_runtime_sources
        sources=verify_runtime_sources()
        c.require(sources==plan['corrected_source_receipt'],'Exact corrected runtime source receipt')
        c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),role=role,**c.IDENTITY))
        dependencies(role,boundary)
        result=dispatch(role,folder,boundary,sources,plan);boundary()
    except BaseException:error=traceback.format_exc()
    receipt=dict(status='passed' if error is None else 'failed',role=role,identity=c.identity(),
        freeze_sha256=expected,result=result,error=error,time=c.now())
    c.save(folder/'result.json',receipt)
    if error:c.save(folder/'failure.json',receipt);print(error,flush=True);return 1
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=c.JOBS,required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    raise SystemExit(run(a.role,a.freeze_sha256))
