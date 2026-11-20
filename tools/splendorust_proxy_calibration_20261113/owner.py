"""Prospectively frozen saved-position study; no new games or training."""
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
    paths=list((c.ROOT/c.PACKAGE.replace('.', '/')).glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('*proxy_calibration*20261113*.py'))
    items=[c.artifact(p) for p in python_dependencies(paths)]+private.inputs()+[c.artifact(c.PROTOCOL)]
    prior_path=c.ROOT/'outputs/splendorust_full_corpus_20261110_01/freeze.json'
    prior=c.read(prior_path)
    items += [c.artifact(prior_path)] + prior['external_runtime_inputs']
    items += [dict(path=x['absolute_path'],bytes=x['bytes'],sha256=x['sha256']) for x in prior['torch_runtime']]
    import importlib.util,os
    runtime=[Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    runtime.append(Path(os.environ['SystemRoot'])/'System32/conhost.exe')
    for module in ('psutil','threadpoolctl'):
        origin=Path(importlib.util.find_spec(module).origin);runtime.append(origin)
        if module=='psutil':runtime+=list(origin.parent.glob('*.py'))+list(origin.parent.glob('*.pyd'))
    items += [c.artifact(p) for p in runtime]
    unique={i['path']:i for i in items};return sorted(unique.values(),key=lambda x:x['path'])

def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from . import panel,adapters,qualification
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Prepare before freeze/launch')
    sources=verify_runtime_sources()
    saved=panel.build_panel(runtime_sources=sources)
    live=panel.known_live_controls()
    panel_item=c.save(c.PANEL,saved)
    matrix=c.save(c.MATRIX,panel.request_matrix(saved));repeats=c.save(c.REPEATS,panel.reproduction_matrix(saved))
    live_item=c.save(c.OUT/'known_live_controls.json',live)
    # Root chosen solely by prospective observer identity, no action/value/outcome.
    root=min(saved['roots'],key=lambda r:c.digest(['qualification',r['root_id']]))
    qplan=c.save(c.QUALIFICATION,qualification.build_plan(root['observer'],root['root_id']))
    profiles={str(q):c.save(c.OUT/'profiles'/f'research{q}.json',adapters.profile(q)) for q in (16,*adapters.QUOTAS)}
    controls={str(q):c.save(c.OUT/'profiles'/f'control{q}.json',adapters.profile(q,control=True)) for q in adapters.CONTROL_QUOTAS}
    items=saved['source_bindings']+list(c.nested_artifacts(live))+[panel_item,matrix,repeats,live_item,qplan,c.artifact(c.GRAPH_PLAN)]
    for item in list(profiles.values())+list(controls.values()):
        items.append(item);document=c.read(c.resolve(item));items+=list(c.nested_artifacts(document))
        items.append(c.artifact(c.ROOT/document['native_binary']))
    items += [c.artifact(c.PARENT_CHECKPOINT)]
    unique={i['path']:i for i in items};data=sorted(unique.values(),key=lambda x:x['path'])
    c.verify(data)
    return c.save(c.OUT/'prepared.json',dict(design=c.design(),worker_inputs=source_inputs(),data_inputs=data,
        panel=panel_item,request_matrix=matrix,reproduction_matrix=repeats,known_live_controls=live_item,
        qualification_plan=qplan,research_profiles=profiles,control_profiles=controls,
        graph_parity_plan=c.artifact(c.GRAPH_PLAN),console_executable=c.artifact(Path(__import__('os').environ['SystemRoot'])/'System32/conhost.exe'),
        corrected_source_receipt=sources,time=c.now()))

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
    fixture_path=c.OUT/'prelaunch/detached_fixture_final/actual_exit.json'
    fixture=c.read(fixture_path)
    c.require(fixture['status']=='passed' and fixture['exit_code']==0 and fixture['exact_owned_absence']
        and fixture['actual_exit_observed'] and fixture['models']==fixture['new_games']==0 and
        not fixture['cleanup_errors'] and fixture['error'] is None,'Durable detached fixture')
    c.verify(fixture['source_inputs'])
    fixture_launcher=c.read(c.OUT/'prelaunch/fixture_launcher_final.json')['identity']
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
