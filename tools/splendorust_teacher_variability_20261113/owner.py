"""Prospectively frozen saved-position study; no new games or training."""
import argparse
from pathlib import Path
import sys
import time
import traceback
from . import common as c
from . import control

PREPARED_NAME='prepared02.json'

def immutable_native_inventory(receipt):
    from . import native_build
    mutable=(native_build.BUILD/'cargo_home/.global-cache').relative_to(c.ROOT).as_posix()
    inventory=c.read(c.resolve(receipt['source_inventory']))
    c.require(sum(i['path']==mutable for i in inventory)==1,'One identified Cargo bookkeeping file')
    # The retained inventory was written before compilation. Cargo updates this
    # SQLite access/cache bookkeeping while compiling; it is not compiler input.
    # Every crate, manifest, lockfile and tool/binary remains content-bound.
    return [i for i in inventory if i['path']!=mutable]

def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    from tools.splendorust_independent_data_20261107.collection import private
    paths=list((c.ROOT/c.PACKAGE.replace('.', '/')).glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('*teacher_variability*20261113*.py'))
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
    from . import native_build
    receipt_path=native_build.BUILD/'build_receipt.json'
    receipt=c.read(receipt_path)
    items += [c.artifact(receipt_path)] + list(c.nested_artifacts(receipt))
    # Bind the actual copied Rust sources and all preserved predecessors,
    # not merely the JSON inventory describing them.
    items += immutable_native_inventory(receipt)
    items += c.read(c.resolve(receipt['prior_source_preservation']))
    items += list(c.nested_artifacts(c.read(c.resolve(receipt['build_plan']))))
    before=native_build.BUILD/'source_preservation_before.json'
    if before.exists():items += [c.artifact(before)]+c.read(before)
    unique={i['path']:i for i in items};return sorted(unique.values(),key=lambda x:x['path'])

def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from . import panel,adapters,qualification
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Prepare before freeze/launch')
    sources=verify_runtime_sources()
    saved=panel.build_panel(runtime_sources=sources)
    panel_item=c.save(c.PANEL,saved)
    matrix=c.save(c.MATRIX,panel.request_matrix(saved));repeats=c.save(c.REPEATS,panel.reproduction_matrix(saved))
    # Root chosen solely by prospective observer identity, no action/value/outcome.
    root=min((r for r in saved['roots'] if r['root_id'] in saved['primary_root_ids']),key=lambda r:c.digest(['qualification',r['root_id']]))
    qplan=c.save(c.QUALIFICATION,qualification.build_plan(root['observer'],root['root_id']))
    profiles={adapters.profile_key(q,pool,mode):c.save(c.OUT/'profiles'/f'{adapters.profile_key(q,pool,mode)}.json',adapters.profile(q,pool,mode))
        for q in adapters.ALL_QUOTAS for pool in adapters.POOLS for mode in adapters.MODES}
    controls={adapters.profile_key(q):c.save(c.OUT/'profiles'/f'control_{q}.json',adapters.profile(q,control=True)) for q in adapters.ALL_QUOTAS}
    items=saved['source_bindings']+saved['prior_bindings']+[panel_item,matrix,repeats,qplan,c.artifact(c.GRAPH_PLAN)]
    for item in list(profiles.values())+list(controls.values()):
        items.append(item);document=c.read(c.resolve(item));items+=list(c.nested_artifacts(document))
        items.append(c.artifact(c.ROOT/document['native_binary']))
    items += [c.artifact(c.PARENT_CHECKPOINT)]
    unique={i['path']:i for i in items};data=sorted(unique.values(),key=lambda x:x['path'])
    c.verify(data)
    return c.save(c.OUT/'prepared.json',dict(design=c.design(),worker_inputs=source_inputs(),data_inputs=data,
        panel=panel_item,request_matrix=matrix,reproduction_matrix=repeats,
        qualification_plan=qplan,research_profiles=profiles,control_profiles=controls,
        graph_parity_plan=c.artifact(c.GRAPH_PLAN),console_executable=c.artifact(Path(__import__('os').environ['SystemRoot'])/'System32/conhost.exe'),
        corrected_source_receipt=sources,time=c.now()))

def freeze(tests,reviews):
    prepared=c.read(c.OUT/PREPARED_NAME)
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
    return c.save(c.PLAN,dict(**prepared,prepared_input=c.artifact(c.OUT/PREPARED_NAME),tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        detached_fixture=c.artifact(fixture_path),frozen_at=c.now()))

def refresh_preparation():
    c.require(not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Refresh before any freeze/launch')
    original=c.read(c.OUT/'prepared.json')
    c.require(original['design']==c.design(),'Scientific design unchanged by bookkeeping correction')
    c.verify(original['data_inputs'])
    preserved=[c.artifact(c.OUT/'prepared.json'),c.artifact(c.OUT/'prelaunch/tests01.json'),
        c.artifact(c.OUT/'prelaunch/tests01.log')]
    old_review=c.OUT/'prelaunch/review_design_integration.json'
    if old_review.exists():preserved.append(c.artifact(old_review))
    updated=dict(original,worker_inputs=source_inputs(),time=c.now(),
        data_inputs=original['data_inputs']+preserved,
        preparation_correction='Only Cargo .global-cache mutable bookkeeping excluded; full prebuild inventory and failed preparation retained')
    c.verify(updated['worker_inputs']);c.verify(updated['data_inputs'])
    return c.save(c.OUT/PREPARED_NAME,updated)

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
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','refresh','freeze','run'))
    p.add_argument('--freeze-sha256');p.add_argument('--tests');p.add_argument('--reviews',nargs='*');a=p.parse_args()
    if a.mode=='prepare':print(c.canonical(prepare()));return 0
    if a.mode=='refresh':print(c.canonical(refresh_preparation()));return 0
    if a.mode=='freeze':print(c.canonical(freeze(a.tests,a.reviews)));return 0
    try:return run(a.freeze_sha256)
    except BaseException:
        error=traceback.format_exc()
        if not (c.OUT/'failure.json').exists():
            c.save(c.OUT/'failure.json',dict(status='incomplete',stage='root_bootstrap',error=error,time=c.now()))
        print(error,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
