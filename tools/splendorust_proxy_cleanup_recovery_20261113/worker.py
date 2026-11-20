"""Two fresh guarded saved-evidence roles; no model or search entry points."""
import argparse,sys,traceback
from . import common as c,control,checks
from tools.splendorust_proxy_calibration_20261113 import common as oldc,worker as oldworker,panel,verifier,analysis

def passed_job(role):
    folder=c.OUT/'jobs'/role
    item=c.artifact(folder/'actual_exit.json');exit=c.read(c.resolve(item));result=c.read(folder/'worker/result.json')
    c.require(exit['role']==role and exit['freeze_sha256']==c.sha(c.PLAN) and exit['status']=='passed' and type(exit['exit_code']) is int and exit['exit_code']==0 and exit['actual_exit_observed'] is True and exit['exact_owned_absence'] is True and not exit['cleanup_errors'] and exit['error'] is None,'Recovered worker actual clean exit')
    c.require(exit['result_artifact']==c.artifact(folder/'worker/result.json') and result['identity']==exit['identity'] and result['status']=='passed' and result['role']==role and result['freeze_sha256']==c.sha(c.PLAN) and result['error'] is None,'Exact recovered result binding')
    c.require(not c.exact_alive(exit['identity']) and all(not c.exact_alive(x) for x in exit['observed_owned_identities']),'Recovered job exact absence')
    return item,result

def bind_analysis_evidence(job,proof,boundary):
    item=job['result']['verification']
    c.require(item==c.artifact(c.OUT/'jobs/analysis/worker/saved_evidence.json'),'Exact recovered analysis evidence path and hash')
    c.verify([item],boundary=boundary)
    c.require(c.read(c.resolve(item))==proof,'Independent acceptance reproduces complete analysis evidence')
    return item

def inputs(recovery,oldplan,boundary):
    qualification=oldworker.check_qualification(oldplan,boundary)
    saved=panel.validate_panel(c.read(oldc.PANEL));folder=oldc.OUT/'jobs/diagnostic/worker'
    _,job=oldworker.passed_job('diagnostic');c.require(job['result']['index']==c.artifact(folder/'index.json'),'Exact original diagnostic inventory')
    diagnostic=checks.check_service('diagnostic',oldplan,boundary,recovery)
    records,proof=verifier.verify_records(folder,saved,panel.request_matrix(saved),freeze=oldplan,boundary=boundary)
    repeat=oldc.OUT/'jobs/reproduction/worker';_,job=oldworker.passed_job('reproduction')
    c.require(job['result']['index']==c.artifact(repeat/'index.json'),'Exact original reproduction inventory')
    reproduction=checks.check_service('reproduction',oldplan,boundary,recovery)
    repeats=verifier.verify_reproductions(folder,repeat,saved,freeze=oldplan,boundary=boundary)
    return saved,records,dict(qualification=qualification,diagnostic_service=diagnostic,reproduction_service=reproduction,ordinary=proof,reproduction=repeats)

def run(role,expected):
    folder=c.OUT/'jobs'/role/'worker';folder.mkdir(parents=True,exist_ok=False);error=None;result=None
    try:
        control.set_own_cpu();owner=control.handshake(folder.parent,expected,role);boundary=control.Boundary(folder,owner);boundary()
        c.require(c.sha(c.PLAN)==expected,'Exact separately frozen recovery');recovery=c.read(c.PLAN)
        c.require(recovery['design']==c.design(),'Fixed saved-evidence-only recovery design')
        c.verify(recovery['worker_inputs'],boundary=boundary);c.verify(recovery['data_inputs'],boundary=boundary)
        c.require(c.sha(c.OLD_PLAN)==c.OLD_SHA,'Original frozen attempt unchanged');oldplan=c.read(c.OLD_PLAN)
        from tools.future_training_rules_20261021 import verify_runtime_sources
        sources=verify_runtime_sources();c.require(sources==oldplan['corrected_source_receipt'],'Original guarded corrected runtime')
        c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),role=role,**c.IDENTITY))
        saved,records,proof=inputs(recovery,oldplan,boundary)
        evidence=c.save(folder/'saved_evidence.json',proof)
        if role=='analysis':
            from . import charts
            report=analysis.analyze(records,saved,boundary=boundary)
            report_item=c.save(folder/'analysis.json',report)
            pictures=charts.render_charts(report,folder,c.read(c.resolve(oldplan['known_live_controls'])),boundary=boundary)
            result=dict(analysis=report_item,verification=evidence,charts=pictures,verification_label='separate_recovered_verification',original_attempt_remains_incomplete=True)
        elif role=='acceptance':
            exit,job=passed_job('analysis');reported=job['result']['analysis']
            original_analysis_evidence=bind_analysis_evidence(job,proof,boundary)
            c.require(reported==c.artifact(c.OUT/'jobs/analysis/worker/analysis.json'),'Exact recovered analysis path');c.verify([reported,*job['result']['charts']],boundary=boundary)
            arithmetic=analysis.verify_arithmetic(records,saved,c.read(c.resolve(reported)),boundary=boundary)
            c.verify(recovery['worker_inputs'],boundary=boundary);c.verify(recovery['data_inputs'],boundary=boundary)
            panel.validate_panel(saved,authenticate=True,boundary=boundary)
            accepted=dict(status='recovered_independent_acceptance_passed',approved=True,original_attempt_remains_incomplete=True,
                original_freeze=recovery['original_freeze'],original_failure=c.artifact(oldc.OUT/'failure.json'),
                cleanup_exception=recovery['cleanup_exception'],recovered_analysis_exit=exit,source_preservation=True,
                saved_evidence=evidence,analysis_saved_evidence=original_analysis_evidence,arithmetic=arithmetic,analysis=reported,charts=job['result']['charts'],
                new_models=0,search_calls=0,new_games=0,replay_decisions=0,automatic_followup=False,time=c.now())
            result=dict(acceptance=c.save(folder/'independent_acceptance.json',accepted))
        else:raise ValueError('Only saved-evidence roles')
        c.require('torch' not in sys.modules,'No tensor/model runtime during saved-evidence recovery');boundary()
    except BaseException:error=traceback.format_exc()
    receipt=dict(status='passed' if error is None else 'failed',role=role,identity=c.identity(),freeze_sha256=expected,result=result,error=error,time=c.now())
    c.save(folder/'result.json',receipt)
    if error:c.save(folder/'failure.json',receipt);print(error,flush=True);return 1
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=c.JOBS,required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    raise SystemExit(run(a.role,a.freeze_sha256))
