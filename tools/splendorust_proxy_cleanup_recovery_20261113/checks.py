"""Saved-evidence-only recovery of one premature descendant-absence check.

All original scientific, source, runtime, ownership and exit-code checks remain.
Only the reproduction service's immediate absence flag may use its already-saved
later outer-owner cleanup proof. No receipt is rewritten and no decision reruns.
"""
from copy import deepcopy
from datetime import datetime, timedelta

from tools.splendorust_proxy_calibration_20261113 import common as c
from tools.splendorust_proxy_calibration_20261113 import worker as original

BASE = 'outputs/splendorust_proxy_calibration_20261113_01/'
PINS = {
    'freeze': dict(path=BASE+'freeze.json', bytes=252370, sha256='51340bd0cae293d84fbc181b8985195f76f1db61d4a30cc09fbaa0a012a18940'),
    'root_owner': dict(path=BASE+'launch/owner.json', bytes=608, sha256='0f377899c97d4ffbf4bea64a4b4fee47f1b237441ed41da9e6dfd906c69dd16d'),
    'worker_owner': dict(path=BASE+'jobs/reproduction/owner.json', bytes=637, sha256='5b5b4c4bd58cc47df57b503c8c0c546e5087e4c1f35b0ae2dca8817acce0481b'),
    'outer_exit': dict(path=BASE+'jobs/reproduction/actual_exit.json', bytes=1047, sha256='f444d49d78ac26dde1250619e9c6b5ab0ccc4090eaa443730a5f18efca65de26'),
    'worker_result': dict(path=BASE+'jobs/reproduction/worker/result.json', bytes=885, sha256='22d5707606cbbd3ceac4ce54a08f077e8b6644cb27cf54350700e7bf4c80a556'),
    'premature_exit': dict(path=BASE+'jobs/reproduction/worker/service_actual_exit.json', bytes=496, sha256='6eb966050cf7ca0f38852458272b39a9ff2b344fafbbe966408dca52ea69c7ad'),
    'service_exit': dict(path=BASE+'jobs/reproduction/worker/service/service_exit.json', bytes=150, sha256='60da6cb461608e0e24576a0ae6fa55bde261c353f5514757f1e5f143a6fae4df'),
    'native_owners': dict(path=BASE+'jobs/reproduction/worker/native_owners.json', bytes=6294, sha256='3ee857101000e937df6af6518f1568bd176ee1746caa732abea128ff66eed784'),
}


def exception_inputs():
    return sorted(deepcopy(list(PINS.values())), key=lambda x:x['path'])


def exception_contract():
    return dict(schema='proxy_saved_outer_cleanup_exception_v1', role='reproduction',
        original_inputs=deepcopy(PINS), new_decisions=0, replacements=0,
        original_failed_receipt_preserved=True,
        permitted_difference='Finalize only immediate descendant absence using bound later original outer-owner cleanup',
        saved_cleanup_interval_microseconds=233332,
        does_not_identify_the_lingering_descendant=True,
        no_claim_of_spontaneous_exit=True)


def _identity(item):
    c.require(type(item) is dict and set(item)=={'pid','created'} and type(item['pid']) is int
        and item['pid']>0 and type(item['created']) in (int,float) and item['created']>0,
        'Strict exact process identity')
    return item['pid'],item['created']


def _time(text):
    value=datetime.fromisoformat(text)
    c.require(value.tzinfo is not None,'Original receipt timestamp is timezone-aware')
    return value


def verify_cleanup_bundle(bundle, *, original_limits):
    """Pure semantic proof; caller also authenticates every exact artifact hash."""
    actual,receipt,outer,owner,job,native,root_owner=(bundle[k] for k in
        ('premature_exit','service_exit','outer_exit','worker_owner','worker_result','native_owners','root_owner'))
    freeze=PINS['freeze']['sha256']
    c.require(root_owner['role']=='root' and root_owner['freeze_sha256']==freeze and
        root_owner['windows_job_assigned'] is True and root_owner['limits']==original_limits,
        'Exact original root and original global clocks')
    c.require(owner['role']=='reproduction' and owner['freeze_sha256']==freeze and
        owner['windows_job_assigned'] is True and owner['parent']==root_owner['identity'],
        'Exact original owned reproduction role')
    c.require(actual['status']=='failed' and actual['error'] is None and type(actual['exit_code']) is int
        and actual['exit_code']==0 and actual['exact_owned_absence'] is False,
        'Only this preserved premature absence failure is recoverable')
    c.require(receipt['status']=='passed' and receipt['error'] is None and receipt['identity']==actual['identity']
        and type(receipt['guarded_decisions']) is int and receipt['guarded_decisions']==32,
        'Unchanged graceful service result and complete finite decision count')
    c.require(outer['status']=='passed' and type(outer['exit_code']) is int and outer['exit_code']==0
        and outer['actual_exit_observed'] is True and outer['exact_owned_absence'] is True
        and outer['error'] is None and outer['cleanup_errors']==[] and outer['role']=='reproduction'
        and outer['freeze_sha256']==freeze and outer['identity']==owner['identity'],
        'Later original owner actually observed successful complete cleanup')
    c.require(job['status']=='passed' and job['error'] is None and job['role']=='reproduction'
        and job['freeze_sha256']==freeze and job['identity']==outer['identity']
        and outer['result_artifact']==PINS['worker_result'],
        'Later cleanup binds exact original worker result')
    c.require(job['result']['service_actual_exit']==PINS['premature_exit'] and
        job['result']['native_owners']==PINS['native_owners'] and
        type(job['result']['requests']) is int and job['result']['requests']==32,
        'Exact original result binds failed inner receipt and ownership evidence')
    c.require(native['book']['service']==actual['identity'] and
        native['book']['all_owned_identities']==actual['owned_native_identities'],
        'All original service and descendant identities retained')
    targets=[actual['identity'],*actual['owned_native_identities']]
    wanted=[_identity(x) for x in targets];covered=[_identity(x) for x in outer['observed_owned_identities']]
    c.require(len(wanted)==8 and len(set(wanted))==len(wanted) and len(set(covered))==len(covered)
        and set(wanted)<=set(covered) and _identity(owner['identity']) in covered,
        'Later cleanup covers every exact original service/native/console identity')
    service_time,early_time,job_time,outer_time=map(_time,
        (receipt['time'],actual['time'],job['time'],outer['time']))
    c.require(service_time<=early_time<job_time<=outer_time and
        outer_time-early_time==timedelta(microseconds=233332),
        'Exact preserved ordering and233ms finalized cleanup observation')
    limits=owner['limits']
    c.require(limits['unix_origin']<=service_time.timestamp()<=outer_time.timestamp()<limits['unix_deadline']
        and original_limits['unix_origin']<=limits['unix_origin']<limits['unix_deadline']<=original_limits['unix_deadline'],
        'Both original cleanup observations are inside original stage and global clocks')
    c.require(limits['monotonic_origin']<limits['monotonic_deadline']<=original_limits['monotonic_deadline']
        and original_limits['monotonic_origin']<=limits['monotonic_origin'],
        'Original monotonic stage remains within original global cap')
    c.require(type(outer['elapsed_seconds']) in (int,float) and 0<outer['elapsed_seconds']<
        min(limits['unix_deadline']-limits['unix_origin'],limits['monotonic_deadline']-limits['monotonic_origin']),
        'Original cleanup completed within original dual-clock stage duration')
    return dict(status='saved_later_outer_cleanup_verified', recovered_cleanup=True,
        premature_receipt=deepcopy(PINS['premature_exit']),later_outer_exit=deepcopy(PINS['outer_exit']),
        bound_worker_result=deepcopy(PINS['worker_result']),graceful_service_exit=deepcopy(PINS['service_exit']),
        exact_owned_identities=targets,cleanup_interval_microseconds=233332,
        complete_decisions=32,exact_reproduction_comparisons_still_required=True,
        original_failed_receipt_preserved=True,new_decisions=0,replacements=0,
        current_absence_is_not_the_sole_proof=True,lingering_descendant_not_identified=True)


def _load_exception(recovery_plan, oldplan, boundary):
    c.require(recovery_plan['cleanup_exception']==exception_contract(), 'Fixed single cleanup exception')
    c.require(recovery_plan['original_freeze']==PINS['freeze'], 'Exact original study freeze')
    c.verify(exception_inputs(),boundary=boundary)
    c.require(c.artifact(c.PLAN)==PINS['freeze'] and c.read(c.PLAN)==oldplan,'Original source plan preserved')
    bundle={k:c.read(c.resolve(v)) for k,v in PINS.items() if k!='freeze'}
    c.require(recovery_plan['original_limits']==bundle['root_owner']['limits'],'No reset of original clocks')
    c.check_deadlines(recovery_plan['original_limits']);boundary()
    return bundle


def check_service(role, oldplan, boundary, recovery_plan):
    """Preserve original check except one separately authenticated cleanup proof."""
    if role!='reproduction':
        result=original.check_service(role,oldplan,boundary)
        return dict(result,recovered_cleanup=False)
    bundle=_load_exception(recovery_plan,oldplan,boundary)
    plan=oldplan
    from tools.splendorust_proxy_calibration_20261113 import qualification,ownership
    folder=c.OUT/'jobs'/role/'worker';_,job=original.passed_job(role)
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
        item=plan['control_profiles']['64'] if name=='control' else plan['research_profiles']['16' if name=='reference' else '64']
        document=c.read(c.resolve(item))
        expected_binaries[name]=c.artifact(c.ROOT/document['native_binary'])
        c.require(expected_binaries[name]['sha256']==document['native_binary_sha256'],'Frozen native content binding')
    ownership.validate_book(native['book'],expected_binaries,plan['console_executable'])
    c.require(runtime['checkpoint_sha256']==graph_identity['checkpoint_sha256']==c.PARENT_CHECKPOINT_SHA and
        graph_identity['weights_before_sha256']==graph_identity['weights_after_sha256'],'Unchanged EF8A identity')
    qualified=c.read(c.OUT/'jobs/qualification/worker/service/proxy_runtime_identity.json')['graph_identity']
    for field in ('checkpoint_sha256','weights_before_sha256','weights_after_sha256','torch_version','numpy_version','backend','options'):
        c.require(graph_identity[field]==qualified[field],'Identical qualified graph/runtime identity')
    c.require(actual==bundle['premature_exit'] and receipt==bundle['service_exit'] and job==bundle['worker_result']
        and native==bundle['native_owners'],'Exact authenticated cleanup bundle consumed by original service checker')
    cleanup=verify_cleanup_bundle(bundle,original_limits=recovery_plan['original_limits'])
    c.require(not c.exact_alive(actual['identity']) and all(not c.exact_alive(x) for x in actual['owned_native_identities']),
        'Exact service/native absence')
    sources=c.read(folder/'service/runtime_sources.json')
    c.require(sources['sources']==plan['corrected_source_receipt'],'Actual service guarded runtime')
    fixtures=c.read(c.resolve(plan['graph_parity_plan']))['fixtures']
    proof=qualification.verify_graph_evidence(folder/'service/graph_parity.json',fixtures)
    return dict(actual_exit=bound,graph=proof,recovered_cleanup=cleanup)
