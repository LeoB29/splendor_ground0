"""Fresh owned role: corrected startup guard, finite work, retained failures."""
import argparse
import traceback
from . import common as c
from . import control


def passed_job(role):
    folder=c.OUT/'jobs'/role
    item=c.artifact(folder/'actual_exit.json');execution=c.read(c.resolve(item))
    result=c.read(folder/'worker/result.json')
    c.require(execution['result_artifact']==c.artifact(folder/'worker/result.json'),'Actual exit binds exact result')
    c.require(execution['status']=='passed' and type(execution['exit_code']) is int and execution['exit_code']==0
        and execution['actual_exit_observed'] and execution['exact_owned_absence'] and not execution['cleanup_errors']
        and execution['error'] is None and execution['role']==role and execution['freeze_sha256']==c.sha(c.PLAN),'Clean prior execution')
    c.require(result['status']=='passed' and result['error'] is None and result['role']==role and result['identity']==execution['identity']
        and result['freeze_sha256']==c.sha(c.PLAN),'Passed exact prior result')
    c.require(not c.exact_alive(execution['identity']) and all(not c.exact_alive(i) for i in execution['observed_owned_identities']),
        'Prior owned processes absent')
    return item,result


def check_service(role,plan,boundary):
    from . import qualification,ownership
    folder=c.OUT/'jobs'/role/'worker';_,job=passed_job(role)
    bound=job['result']['service_actual_exit'];c.verify([bound],boundary=boundary)
    c.require(bound==c.artifact(folder/'service_actual_exit.json'),'Job binds exact service receipt')
    actual=c.read(c.resolve(bound));receipt=c.read(folder/'service/service_exit.json')
    owners=job['result']['native_owners'];c.verify([owners],boundary=boundary)
    native=c.read(c.resolve(owners));c.verify([native['runtime_identity'],native['startup']],boundary=boundary)
    startup=c.read(c.resolve(native['startup']))
    c.require(startup['rows']==native['book']['startup'] and startup['service']==actual['identity'],
        'Exact original startup ownership')
    c.require(native['book']['all_owned_identities']==actual['owned_native_identities'],'All native and console ownership retained')
    runtime=c.read(c.resolve(native['runtime_identity']));graph_identity=runtime['graph_identity']
    c.require(native['book']['service']==actual['identity'] and native['book']['native_pids']==runtime['native_pids'],
        'Service book bound to exact runtime roles')
    expected_binaries={}
    for name in runtime['native_pids']:
        item=plan['control_profiles']['legacy_p3_q16'] if name=='control' else plan['research_profiles']['legacy_p3_q16' if name=='reference' else 'legacy_p3_q512']
        document=c.read(c.resolve(item))
        expected_binaries[name]=c.artifact(c.ROOT/document['native_binary'])
        c.require(expected_binaries[name]['sha256']==document['native_binary_sha256'],'Frozen native content binding')
    ownership.validate_book(native['book'],expected_binaries,plan['console_executable'])
    c.require(runtime['checkpoint_sha256']==graph_identity['checkpoint_sha256']==c.PARENT_CHECKPOINT_SHA and
        graph_identity['weights_before_sha256']==graph_identity['weights_after_sha256'],'Unchanged EF8A identity')
    qualified=c.read(c.OUT/'jobs/qualification/worker/service/variability_runtime_identity.json')['graph_identity']
    for field in ('checkpoint_sha256','weights_before_sha256','weights_after_sha256','torch_version','numpy_version','backend','options'):
        c.require(graph_identity[field]==qualified[field],'Identical qualified graph/runtime identity')
    c.require(actual['status']=='passed' and actual['exit_code']==0 and actual['exact_owned_absence'] and actual['error'] is None
        and receipt['status']=='passed' and receipt['error'] is None and actual['identity']==receipt['identity'],
        'Observed graceful service exit0')
    c.require(not c.exact_alive(actual['identity']) and all(not c.exact_alive(x) for x in actual['owned_native_identities']),
        'Exact service/native absence')
    sources=c.read(folder/'service/runtime_sources.json')
    c.require(sources['sources']==plan['corrected_source_receipt'],'Actual service guarded runtime')
    fixtures=c.read(c.resolve(plan['graph_parity_plan']))['fixtures']
    proof=qualification.verify_graph_evidence(folder/'service/graph_parity.json',fixtures)
    return dict(actual_exit=bound,graph=proof)


def check_qualification(plan,boundary):
    from . import qualification
    folder=c.OUT/'jobs/qualification/worker';_,job=passed_job('qualification')
    service=check_service('qualification',plan,boundary)
    qplan=c.read(c.QUALIFICATION);entry=[]
    c.require(len(job['result']['entrypoints'])==4,'All four qualified mode/pool routes')
    for saved,route in zip(job['result']['entrypoints'],qplan['entrypoint_routes'],strict=True):
        c.require(saved['route']==route,'Exact entrypoint route')
        item=saved['receipt'];c.verify([item],boundary=boundary)
        c.require(item==c.artifact(folder/f"entrypoint_{route['mode']}_p{route['pool']}_receipt.json"),'Exact entrypoint receipt')
        proof=qualification.verify_entrypoint(folder/'service',c.read(c.resolve(item)),qplan,plan,**route,boundary=boundary)
        c.require(proof==saved['verification'],'Entrypoint arithmetic binding');entry.append(proof)
    rows=[c.read(folder/f'parity_{i:02d}.json') for i in range(len(qplan['parity_requests']))]
    profiles={q:c.read(c.resolve(i)) for q,i in plan['research_profiles'].items()}
    controls={q:c.read(c.resolve(i)) for q,i in plan['control_profiles'].items()}
    parity=qualification.validate_parity_rows(rows,qplan,profiles,plan['research_profiles'],controls,plan['control_profiles'])
    c.require(parity==job['result']['parity'],'Qualification arithmetic binding')
    return dict(service=service,entrypoint=entry,parity=parity)


def analysis_inputs(plan,boundary):
    from . import panel,verifier
    saved=panel.validate_panel(c.read(c.PANEL));folder=c.OUT/'jobs/diagnostic/worker'
    _,job=passed_job('diagnostic');c.require(job['result']['index']==c.artifact(folder/'index.json'),'Exact diagnostic inventory')
    check_service('diagnostic',plan,boundary)
    records,proof=verifier.verify_records(folder,saved,panel.request_matrix(saved),freeze=plan,boundary=boundary)
    repeat=c.OUT/'jobs/reproduction/worker';_,job=passed_job('reproduction')
    c.require(job['result']['index']==c.artifact(repeat/'index.json'),'Exact reproduction inventory')
    check_service('reproduction',plan,boundary)
    repeats=verifier.verify_reproductions(folder,repeat,saved,freeze=plan,boundary=boundary)
    return saved,records,dict(ordinary=proof,reproduction=repeats)


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
        c.require(sources==plan['corrected_source_receipt'],'Exact corrected worker source guard')
        c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),role=role,**c.IDENTITY))
        if role!='qualification':check_qualification(plan,boundary)
        if role in c.MODEL_JOBS:
            from . import run_models
            if role=='reproduction':passed_job('diagnostic')
            result=run_models.run(role,folder,owner,plan,boundary)
        elif role=='analysis':
            from . import analysis,charts as chart_module
            saved,records,proof=analysis_inputs(plan,boundary)
            report=analysis.analyze(records,saved,boundary=boundary)
            item=c.save(folder/'analysis.json',report)
            boundary();charts=[c.artifact(p) for p in chart_module.render(report,folder)];boundary()
            result=dict(analysis=item,verification=c.save(folder/'saved_evidence.json',proof),charts=charts)
        elif role=='acceptance':
            from . import analysis,panel
            saved,records,proof=analysis_inputs(plan,boundary)
            _,job=passed_job('analysis');report_item=job['result']['analysis'];c.verify([report_item],boundary=boundary)
            c.require(report_item==c.artifact(c.OUT/'jobs/analysis/worker/analysis.json'),'Accepted analysis binding')
            report=c.read(c.resolve(report_item))
            arithmetic=analysis.verify_arithmetic(records,saved,report,boundary=boundary)
            # Verify all source hashes again; no model calls or replacement decisions.
            c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
            panel.validate_panel(saved,authenticate=True,boundary=boundary)
            c.verify(job['result']['charts'],boundary=boundary)
            acceptance=dict(status='independent_acceptance_passed',approved=True,source_preservation=True,
                prior_actual_exits=[passed_job(r)[0] for r in c.JOBS if r!='acceptance'],
                qualification=check_qualification(plan,boundary),saved_evidence=proof,arithmetic=arithmetic,
                analysis=report_item,charts=job['result']['charts'],new_games=0,weight_updates=0,
                automatic_promotion=False,automatic_followup=False,time=c.now())
            result=dict(acceptance=c.save(folder/'independent_acceptance.json',acceptance))
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
