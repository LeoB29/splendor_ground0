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
    from tools.splendorust_teacher_variability_20261113 import qualification,ownership
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
    qualified=c.read(c.ORIGINAL_OUT/'jobs/qualification/worker/service/variability_runtime_identity.json')['graph_identity']
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
    from tools.splendorust_teacher_variability_20261113 import worker as old_worker
    return old_worker.check_qualification(c.read(c.ORIGINAL_FREEZE),boundary)


def check_admission(plan,boundary):
    from tools.splendorust_teacher_variability_20261113 import verifier as old_verifier
    folder=c.OUT/'jobs/admission/worker';_,job=passed_job('admission')
    c.require(job['result']['index']==c.artifact(folder/'index.json'),'Exact admission inventory')
    service=check_service('admission',plan,boundary)
    specs=c.read(c.ADMISSION);rows,index=old_verifier._index(folder,specs,boundary=boundary)
    c.require(len(rows)==2 and {r['spec']['quota'] for r in rows}=={512,2048},'Two fixed continuity probes')
    proofs=[]
    for row in rows:
        receipt=old_verifier._read_bound(row['receipt']);source=old_verifier._read_bound(row['spec']['source_receipt'])
        old_verifier.clock_audit.check_clock(receipt,limit=120)
        c.require(receipt['accepted'] and receipt['status']=='ok' and receipt['received_before_deadline'],'Accepted admission clock')
        c.require(c.canonical(receipt['result'])==c.canonical(source['result']),'Exact unchanged complete result for continuity probe')
        proofs.append(dict(reproduced=row['receipt'],source=row['spec']['source_receipt'],exact=True))
    forensic=old_verifier.verify_forensics(folder,rows,boundary=boundary)
    return dict(service=service,index=index,probes=proofs,forensic=forensic)


def check_retry(plan,boundary):
    item=c.artifact(c.OUT/'ambiguous_retry_consumed.json');record=c.read(c.resolve(item))
    spec=c.read(c.MATRIX)[c.PREFIX_COUNT]
    c.require(record['request']==spec and record['original_request_index']==c.PREFIX_COUNT
        and record['new_allowance_seconds']==120 and record['original_attempt_remains_incomplete']
        and record['freeze_sha256']==c.sha(c.PLAN),
        'Only the disclosed possibly interrupted request consumes a fresh allowance')
    dispatch=c.read(c.OUT/'jobs/diagnostic/worker/dispatch/00000.json')
    owner=c.read(c.OUT/'jobs/diagnostic/owner.json')
    c.require(dispatch['spec']==spec and record['time']<=dispatch['time'] and record['identity']==owner['identity'],
        'Consumption saved by exact owner before first suffix dispatch')
    expected_specs=c.read(c.SUFFIX);folder=c.OUT/'jobs/diagnostic/worker/dispatch'
    c.require({p.name for p in folder.glob('*.json')}=={f'{i:05d}.json' for i in range(len(expected_specs))},'Exactly one dispatch record per suffix request')
    for i,s in enumerate(expected_specs):
        boundary();d=c.read(folder/f'{i:05d}.json')
        c.require(d['spec']==s and d['role']=='diagnostic' and d['deadline_seconds']==120
            and d['identity']==owner['identity'] and d['freeze_sha256']==c.sha(c.PLAN),'Exact remaining dispatch sequence')
    return dict(consumption=item,only_boundary_request_may_repeat=True,original_attempt_incomplete=True)


def analysis_inputs(plan,boundary):
    from tools.splendorust_teacher_variability_20261113 import panel
    from . import verifier
    saved=panel.validate_panel(c.read(c.PANEL));folder=c.OUT/'jobs/diagnostic/worker'
    old_folder=c.ORIGINAL_OUT/'jobs/diagnostic/worker';oldfreeze=c.read(c.ORIGINAL_FREEZE)
    _,job=passed_job('diagnostic');c.require(job['result']['index']==c.artifact(folder/'index.json'),'Exact recovered suffix inventory')
    check_service('diagnostic',plan,boundary)
    records,proof=verifier.verify_segments(old_folder,folder,saved,oldfreeze,c.read(c.BOUNDARY),boundary=boundary)
    repeat=c.OUT/'jobs/reproduction/worker';_,job=passed_job('reproduction')
    c.require(job['result']['index']==c.artifact(repeat/'index.json'),'Exact reproduction inventory')
    check_service('reproduction',plan,boundary)
    repeats=verifier.verify_reproductions(old_folder,folder,repeat,saved,oldfreeze,c.read(c.BOUNDARY),boundary=boundary)
    return saved,records,dict(ordinary=proof,reproduction=repeats,retry_disclosure=check_retry(plan,boundary),
        recovered_admission=check_admission(plan,boundary))


def run(role,expected):
    folder=c.OUT/'jobs'/role/'worker';folder.mkdir(parents=True,exist_ok=False)
    error=None;result=None
    try:
        owner=control.handshake(folder.parent,expected,role);boundary=control.Boundary(folder,owner);boundary()
        c.require(c.sha(c.PLAN)==expected,'Exact worker freeze');plan=c.read(c.PLAN)
        c.require(plan['design']==c.design(),'Unchanged fixed design')
        c.require_original_absence()
        c.require_original_inventory()
        c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
        from tools.future_training_rules_20261021 import verify_runtime_sources
        sources=verify_runtime_sources()
        c.require(sources==plan['corrected_source_receipt'],'Exact corrected worker source guard')
        c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),role=role,**c.IDENTITY))
        check_qualification(plan,boundary)
        if role!='admission':check_admission(plan,boundary)
        if role in c.MODEL_JOBS:
            from . import run_models
            if role=='reproduction':passed_job('diagnostic')
            result=run_models.run(role,folder,owner,plan,boundary)
        elif role=='analysis':
            from tools.splendorust_teacher_variability_20261113 import analysis,charts as chart_module
            saved,records,proof=analysis_inputs(plan,boundary)
            report=analysis.analyze(records,saved,boundary=boundary)
            item=c.save(folder/'analysis.json',report)
            boundary();charts=[c.artifact(p) for p in chart_module.render(report,folder)];boundary()
            result=dict(analysis=item,verification=c.save(folder/'saved_evidence.json',proof),charts=charts)
        elif role=='acceptance':
            from tools.splendorust_teacher_variability_20261113 import analysis,panel
            saved,records,proof=analysis_inputs(plan,boundary)
            _,job=passed_job('analysis');report_item=job['result']['analysis'];c.verify([report_item],boundary=boundary)
            c.require(report_item==c.artifact(c.OUT/'jobs/analysis/worker/analysis.json'),'Accepted analysis binding')
            report=c.read(c.resolve(report_item))
            arithmetic=analysis.verify_arithmetic(records,saved,report,boundary=boundary)
            # Verify all source hashes again; no model calls or replacement decisions.
            c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
            panel.validate_panel(saved,authenticate=True,boundary=boundary)
            c.verify(job['result']['charts'],boundary=boundary)
            acceptance=dict(status='amended_recovered_independent_acceptance_passed',approved=True,source_preservation=True,
                original_attempt_remains_incomplete=True,original_crash_boundary=c.artifact(c.BOUNDARY),
                preserved_original_inventory=c.artifact(c.PRESERVATION),
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
