"""Fixed sequential study; never launches collection, search or strength games."""
import argparse
from pathlib import Path
import sys
import time
import traceback
from . import common as c
from . import control

def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    from tools.splendorust_independent_data_20261107.collection import private
    paths=list((c.ROOT/'tools/splendorust_training_duration_20261113').glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('*training_duration*20261113*.py'))
    items=[c.artifact(p) for p in python_dependencies(paths)]+private.inputs()+[c.artifact(c.PROTOCOL)]
    unique={i['path']:i for i in items};return sorted(unique.values(),key=lambda x:x['path'])

def consumed_inputs():
    paths=[c.SELECTION,c.ORIGINAL_COMMITMENT,c.ORIGINAL_EVALUATION,c.CACHE_ACCEPTANCE,c.CACHE_MANIFEST,
        c.PARENT_CHECKPOINT,c.ORIGINAL_OUT/'freeze.json',
        c.ROOT/'outputs/splendorust_full_corpus_closeout_20261112_01/closeout_complete.json',
        c.ROOT/'outputs/splendorust_full_corpus_closeout_20261112_01/independent_closeout_check.json']
    items=[c.artifact(p) for p in paths]
    # Bind consumed cache arrays and index; existing independent acceptance supplies source-row provenance.
    manifest=c.read(c.CACHE_MANIFEST)
    for key in ('arrays','index','selection','source_receipt'):items+=list(c.nested_artifacts(manifest[key]))
    items+=list(c.nested_artifacts(c.read(c.CACHE_ACCEPTANCE)))
    for relative in ('corpus_build/worker/result.json','corpus_audit/worker/result.json',
                     'corpus_build/execution.json','corpus_audit/execution.json'):
        items.append(c.artifact(c.ORIGINAL_OUT/relative))
    from tools.splendorust_independent_data_20261107.collection import common as collector
    items.append(c.artifact(collector.OUT/'qualification_gate.json'))
    commitment=c.read(c.ORIGINAL_COMMITMENT)
    items+=list(c.nested_artifacts(commitment))
    for row in commitment['fits']:
        result=c.read(c.resolve(row['fit_result']));items+=list(c.nested_artifacts(result))
        for checkpoint in result['checkpoints'].values():items+=list(c.nested_artifacts(c.read(c.resolve(checkpoint))))
    evaluation=c.read(c.ORIGINAL_EVALUATION);items+=list(c.nested_artifacts(evaluation))
    qualification=c.ORIGINAL_OUT/'training_qualification/worker/qualification_result.json'
    items.append(c.artifact(qualification))
    items+=list(c.nested_artifacts(c.read(qualification)['data_qualification']))
    unique={i['path']:i for i in items};return sorted(unique.values(),key=lambda x:x['path'])

def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from . import fit
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Prepare before freeze/launch')
    c.require(c.read(c.ROOT/'outputs/splendorust_full_corpus_closeout_20261112_01/closeout_complete.json')['status']=='completed_verified','Accepted predecessor')
    selection=c.read(c.SELECTION)
    schedules={str(rep):fit.build_schedules(selection,rep) for rep in (0,1)}
    inputs=source_inputs();data=consumed_inputs();c.verify(data)
    return c.save(c.OUT/'prepared.json',dict(design=c.design(),worker_inputs=inputs,data_inputs=data,
        schedule_sha256={k:c.digest(v) for k,v in schedules.items()},corrected_source_receipt=verify_runtime_sources(),time=c.now()))

def freeze(tests,reviews):
    prepared=c.read(c.OUT/'prepared.json')
    c.require(prepared['design']==c.design() and prepared['worker_inputs']==source_inputs(),'Current prepared source/design')
    proof=c.read(tests)
    c.require(proof['status']=='passed' and type(proof['exit_code']) is int and proof['exit_code']==0 and proof['inputs']==prepared['worker_inputs'] and
        proof['summary']['failed']==proof['summary']['skipped']==0 and proof['summary']['passed']==proof['summary']['collected']>=20 and
        proof['summary']['tensor_runtime_loaded'] is False and proof['summary']['models']==proof['summary']['new_games']==0,'Full source-bound tests')
    c.require(len(reviews)==3,'Three independent reviews')
    reviewers=set()
    for path in reviews:
        review=c.read(path);reviewers.add(review['reviewer'])
        c.require(review['approved'] is True and not review['blocking_findings'] and
            review['inputs']==prepared['worker_inputs'] and review['design']==c.design(),'Current source-bound independent approval')
    c.require(len(reviewers)==3,'Distinct independent reviewers')
    fixture_path=c.OUT/'prelaunch/detached_fixture_05/actual_exit.json'
    fixture=c.read(fixture_path)
    c.require(fixture['status']=='passed' and fixture['exit_code']==0 and fixture['exact_owned_absence']
        and fixture['actual_exit_observed'] and fixture['models']==fixture['new_games']==0 and
        not fixture['cleanup_errors'] and fixture['error'] is None,'Durable detached fixture')
    c.verify(fixture['source_inputs'])
    fixture_launcher=c.read(c.OUT/'prelaunch/fixture_launcher_05.json')['identity']
    c.require(not c.exact_alive(fixture_launcher) and not c.exact_alive(fixture['identity']),'Detached fixture actual cleanup')
    c.verify(prepared['worker_inputs']);c.verify(prepared['data_inputs'])
    return c.save(c.PLAN,dict(**prepared,tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        detached_fixture=c.artifact(fixture_path),frozen_at=c.now()))

def run(expected):
    owner=control.handshake(c.OUT/'launch',expected,'root');bounds=owner['limits']
    control.bounds(bounds);c.require(c.sha(c.PLAN)==expected,'Exact frozen plan')
    plan=c.read(c.PLAN);c.verify(plan['worker_inputs']);c.verify(plan['data_inputs'])
    c.save(c.OUT/'claim.json',dict(identity=c.identity(),freeze=c.artifact(c.PLAN),limits=bounds,time=c.now()))
    receipts=[];error=None
    try:
        for role in c.JOBS:
            c.status(c.OUT/'stage.json',dict(time=c.now(),stage=role,status='running',completed_jobs=len(receipts)))
            limits=control.clipped_limits(bounds,c.CAPS[role],model=role in c.MODEL_JOBS)
            folder=c.OUT/'jobs'/role
            command=[sys.executable,'-B','-u','-m',c.PACKAGE+'.worker','--role',role,'--freeze-sha256',expected]
            receipt=control.launch(command,folder,limits,dict(role=role,freeze_sha256=expected),
                rss=c.MODEL_RSS if role in c.MODEL_JOBS else c.AUDIT_RSS)
            result=c.read(folder/'worker/result.json')
            c.require(result['status']=='passed' and result['role']==role and result['error'] is None and result['freeze_sha256']==expected
                and result['identity']==c.read(c.resolve(receipt))['identity'],'Exact passed worker result')
            receipts.append(receipt)
        c.verify(plan['worker_inputs']);c.verify(plan['data_inputs']);control.bounds(bounds)
    except BaseException:error=traceback.format_exc()
    execution=c.save(c.OUT/'execution.json',dict(status='passed' if error is None else 'failed',
        freeze_sha256=expected,receipts=receipts,error=error,time=c.now()))
    if error:
        c.save(c.OUT/'failure.json',dict(status='incomplete',execution=execution,error=error))
        c.status(c.OUT/'stage.json',dict(status='failed',stage='failed',time=c.now()));return 1
    c.save(c.OUT/'completion.json',dict(status='verified_pending_actual_owner_exit',execution=execution,
        acceptance=c.artifact(c.OUT/'jobs/acceptance/worker/result.json'),automatic_followup=False))
    c.status(c.OUT/'stage.json',dict(status='verified_pending_actual_owner_exit',stage='complete',time=c.now()))
    return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','freeze','run'))
    p.add_argument('--freeze-sha256');p.add_argument('--tests');p.add_argument('--reviews',nargs='*');a=p.parse_args()
    if a.mode=='prepare':print(c.canonical(prepare()));return 0
    if a.mode=='freeze':print(c.canonical(freeze(a.tests,a.reviews)));return 0
    try:return run(a.freeze_sha256)
    except BaseException:
        error=traceback.format_exc()
        if not (c.OUT/'failure.json').exists():
            c.save(c.OUT/'failure.json',dict(status='incomplete',stage='root_bootstrap',error=error,time=c.now()))
        print(error,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
