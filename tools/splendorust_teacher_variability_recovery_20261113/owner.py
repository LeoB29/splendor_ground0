"""Separate amended continuation after host reboot; original attempt is immutable."""
import argparse
from pathlib import Path
import sys
import traceback
from . import common as c, control, clocks

BOUNDARY_SHA='a2eaa3ec8148bd502791acb1b427dca999b7040d487a657762b4e92c96fc8f9b'

def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    c.require(c.sha(c.ORIGINAL_FREEZE)==c.ORIGINAL_FREEZE_SHA,'Original freeze preserved')
    prior=c.read(c.ORIGINAL_FREEZE)['worker_inputs'];c.verify(prior)
    paths=list(Path(__file__).parent.glob('*.py'))
    added=[c.artifact(p) for p in python_dependencies(paths)]+[c.artifact(c.PROTOCOL)]
    book={x['path']:x for x in prior}
    for item in added:
        c.require(item['path'] not in book or book[item['path']]==item,'No changed inherited implementation')
        book[item['path']]=item
    return sorted(book.values(),key=lambda i:i['path'])

def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_teacher_variability_20261113 import worker as old_worker
    from . import verifier
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Prepare before new freeze and launch')
    c.require(c.sha(c.BOUNDARY)==BOUNDARY_SHA,'Independent preserved crash boundary')
    old=c.read(c.ORIGINAL_FREEZE);sources=verify_runtime_sources()
    c.require(sources==old['corrected_source_receipt'],'Unchanged corrected runtime')
    boundary=c.read(c.BOUNDARY)
    c.require_original_absence()
    c.require_original_inventory()
    c.require(boundary['prefix_count']==c.PREFIX_COUNT and boundary['dispatch_state']=='unknown',
        'Only disclosed ambiguous boundary is authorized')
    c.verify(old['worker_inputs']);c.verify(old['data_inputs'])
    c.verify(c.read(c.PRESERVATION)['artifacts'])
    original_qualification=old_worker.check_qualification(old,lambda:None)
    rows,prefix_proof=verifier.read_prefix(boundary)
    matrix=c.read(c.MATRIX);c.require(len(matrix)==3072 and len(rows)==c.PREFIX_COUNT,'Fixed total and prefix')
    suffix=c.save(c.SUFFIX,matrix[c.PREFIX_COUNT:])
    probes=[]
    for quota in (512,2048):
        row=min((r for r in rows if r['spec']['quota']==quota),key=lambda r:c.digest([c.STUDY,'continuity_probe',r['spec']['request_id']]))
        probes.append(dict(row['spec'],request_id='continuity_'+row['spec']['request_id'],
            original_request_id=row['spec']['request_id'],source_receipt=row['receipt']))
    admission=c.save(c.ADMISSION,probes)
    reused=c.save(c.OUT/'original_qualification_readback.json',original_qualification)
    prefix=c.save(c.OUT/'prefix_admission_readback.json',prefix_proof)
    items=old['data_inputs']+c.read(c.PRESERVATION)['artifacts']+[c.artifact(c.ORIGINAL_FREEZE),
        c.artifact(c.BOUNDARY),c.artifact(c.PRESERVATION),suffix,admission,reused,prefix]
    data=sorted({i['path']:i for i in items}.values(),key=lambda i:i['path'])
    deadlines=clocks.original_deadlines()
    c.verify(data)
    return c.save(c.OUT/'prepared.json',dict(design=c.design(),worker_inputs=source_inputs(),data_inputs=data,
        original_freeze=c.artifact(c.ORIGINAL_FREEZE),boundary=c.artifact(c.BOUNDARY),preservation=c.artifact(c.PRESERVATION),
        original_deadlines=deadlines,original_qualification=reused,prefix_readback=prefix,
        panel=old['panel'],request_matrix=old['request_matrix'],reproduction_matrix=old['reproduction_matrix'],
        admission_requests=admission,suffix_requests=suffix,
        research_profiles=old['research_profiles'],control_profiles=old['control_profiles'],
        graph_parity_plan=old['graph_parity_plan'],console_executable=old['console_executable'],
        corrected_source_receipt=sources,time=c.now()))

def freeze(tests,reviews):
    prepared=c.read(c.OUT/'prepared.json')
    c.require(prepared['design']==c.design() and prepared['worker_inputs']==source_inputs(),'Current recovery design/source')
    proof=c.read(tests)
    c.require(proof['status']=='passed' and proof['exit_code']==0 and proof['inputs']==prepared['worker_inputs']
        and proof['summary']['failed']==proof['summary']['skipped']==0
        and proof['summary']['passed']==proof['summary']['collected']>=20
        and not proof['summary']['tensor_runtime_loaded'],'Complete source-bound model-free tests')
    c.require(len(reviews)==3,'Three scoped independent reviews')
    names=set()
    for path in reviews:
        r=c.read(path);names.add(r['reviewer'])
        c.require(r['approved'] and not r['blocking_findings'] and r['inputs']==prepared['worker_inputs']
            and r['design']==c.design(),'Current scoped source-bound review')
    c.require(len(names)==3,'Distinct scoped reviewers')
    fixture=c.OUT/'prelaunch/detached_fixture_final2/actual_exit.json';f=c.read(fixture)
    c.require(f['status']=='passed' and f['exit_code']==0 and f['exact_owned_absence'] and f['actual_exit_observed']
        and f['error'] is None and not f['cleanup_errors'],'Detached recovery owner actual exit')
    c.verify(f['source_inputs']);c.verify(prepared['worker_inputs']);c.verify(prepared['data_inputs'])
    c.require(not c.exact_alive(c.read(c.OUT/'prelaunch/fixture_launcher_final2.json')['identity']),'Fixture launcher exited')
    return c.save(c.PLAN,dict(**prepared,tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        prepared_input=c.artifact(c.OUT/'prepared.json'),detached_fixture=c.artifact(fixture),frozen_at=c.now()))

def run(expected):
    owner=control.handshake(c.OUT/'launch',expected,'root');bounds=owner['limits']
    control.bounds(bounds);c.require(c.sha(c.PLAN)==expected,'Exact recovery freeze')
    plan=c.read(c.PLAN);c.verify(plan['worker_inputs']);c.verify(plan['data_inputs'])
    c.save(c.OUT/'claim.json',dict(identity=c.identity(),freeze=c.artifact(c.PLAN),limits=bounds,time=c.now()))
    receipts=[];error=None
    try:
        for role in c.JOBS:
            c.status(c.OUT/'stage.json',dict(time=c.now(),stage=role,status='running',completed_jobs=len(receipts)))
            limits=clocks.stage_limits(bounds,c.CAPS[role],role)
            folder=c.OUT/'jobs'/role
            command=[sys.executable,'-B','-u','-m',c.PACKAGE+'.worker','--role',role,'--freeze-sha256',expected]
            item=control.launch(command,folder,limits,dict(role=role,freeze_sha256=expected),
                rss=c.MODEL_RSS if role in c.MODEL_JOBS else c.AUDIT_RSS)
            result=c.read(folder/'worker/result.json')
            c.require(result['status']=='passed' and result['error'] is None and result['role']==role
                and result['freeze_sha256']==expected and result['identity']==c.read(c.resolve(item))['identity'],
                'Exact passed recovered worker result')
            receipts.append(item)
        c.verify(plan['worker_inputs']);c.verify(plan['data_inputs']);control.bounds(bounds)
    except BaseException:error=traceback.format_exc()
    execution=c.save(c.OUT/'execution.json',dict(status='passed' if error is None else 'failed',
        original_attempt_remains_incomplete=True,freeze_sha256=expected,receipts=receipts,error=error,time=c.now()))
    if error:
        c.save(c.OUT/'failure.json',dict(status='incomplete_recovery',execution=execution,error=error))
        c.status(c.OUT/'stage.json',dict(status='failed',stage='failed',time=c.now()));return 1
    c.save(c.OUT/'completion.json',dict(status='amended_verified_pending_actual_owner_exit',execution=execution,
        acceptance=c.artifact(c.OUT/'jobs/acceptance/worker/result.json'),original_attempt_remains_incomplete=True,
        automatic_followup=False))
    c.status(c.OUT/'stage.json',dict(status='amended_verified_pending_actual_owner_exit',stage='complete',time=c.now()))
    return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','freeze','run'))
    p.add_argument('--freeze-sha256');p.add_argument('--tests');p.add_argument('--reviews',nargs='*');a=p.parse_args()
    if a.mode=='prepare':print(c.canonical(prepare()));return 0
    if a.mode=='freeze':print(c.canonical(freeze(a.tests,a.reviews)));return 0
    try:return run(a.freeze_sha256)
    except BaseException:
        error=traceback.format_exc()
        if not (c.OUT/'failure.json').exists():c.save(c.OUT/'failure.json',dict(status='incomplete_recovery',error=error,time=c.now()))
        print(error,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
