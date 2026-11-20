"""Frozen owned four-fit study; no live games or automatic subsequent experiment."""
import argparse
from pathlib import Path
import sys,time,traceback
from . import common as c,control

def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    from tools.splendorust_independent_data_20261107.collection import private
    paths=list((c.ROOT/c.PACKAGE.replace('.', '/')).rglob('*.py'))
    paths+=list((c.ROOT/'tests').glob('*training4096*20261120*.py'))
    items=[c.artifact(p) for p in python_dependencies(paths)]+private.inputs()+[c.artifact(c.PROTOCOL)]
    prior_path=c.ORIGINAL_OUT/'freeze.json';prior=c.read(prior_path)
    items += [c.artifact(prior_path)] + prior['external_runtime_inputs']
    items += [dict(path=x['absolute_path'],bytes=x['bytes'],sha256=x['sha256']) for x in prior['torch_runtime']]
    import importlib.util,os
    runtime=[Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(os.environ['SystemRoot'])/'System32/conhost.exe']
    for module in ('psutil','threadpoolctl'):
        origin=Path(importlib.util.find_spec(module).origin);runtime.append(origin)
        if module=='psutil':runtime+=list(origin.parent.rglob('*.py'))+list(origin.parent.glob('*.pyd'))
    items += [c.artifact(p) for p in runtime]
    unique={i['path']:i for i in items};return sorted(unique.values(),key=lambda x:x['path'])

def consumed_inputs():
    # Explicit metadata-only binding: never recursively flatten a fresh catalogue.
    items=[c.artifact(x) for x in c.PREP.iterdir() if x.is_file()]
    items += [c.artifact(c.UNION),c.artifact(c.SELECTION),c.artifact(c.ORIGINAL_COMMITMENT),
        c.artifact(c.PARENT_CHECKPOINT),c.artifact(c.CACHE_ACCEPTANCE),c.artifact(c.CACHE_MANIFEST)]
    union=c.read(c.UNION)
    for source in union['sources'].values():
        items += [source[k] for k in ('acceptance','catalogue','source_plan')]
    oldcache=c.read(c.CACHE_MANIFEST)
    items += list(oldcache['arrays'].values())+[oldcache['index']]
    for name in ('corpus_build/worker/result.json','corpus_audit/worker/result.json','corpus_build/execution.json','corpus_audit/execution.json'):
        items.append(c.artifact(c.ORIGINAL_OUT/name))
    from . import fit,corpus,offline
    for module in (fit,corpus,offline):
        if hasattr(module,'consumed_inputs'):items+=module.consumed_inputs()
    unique={i['path']:i for i in items}
    # The fresh assessment records must never appear in initial verification inputs.
    from . import selection
    if hasattr(selection,'validate_initial_inputs'):selection.validate_initial_inputs(list(unique.values()))
    return sorted(unique.values(),key=lambda x:x['path'])


def prepare(label):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from . import placement,selection
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(label.isalnum() and not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Prepare exclusive unfrozen attempt')
    sources=verify_runtime_sources();selection.prepare()
    inputs=source_inputs();data=consumed_inputs();c.verify(data)
    return c.save(c.OUT/'prelaunch'/(label+'.json'),dict(design=c.design(),worker_inputs=inputs,data_inputs=data,
        selection=c.artifact(c.SELECTION),placement=placement.snapshot(),corrected_source_receipt=sources,
        authorization=dict(user_authorized_implementation_qualification_freeze_and_launch=True,
            request='Implement and qualify the4096-game comparison, then freeze and run; EF8A reference; fresh assessment sealed until all four fixed'),
        time=c.now()))


def freeze(prepared_path,tests,reviews,probe):
    prepared=c.read(prepared_path);inputs=source_inputs()
    c.require(prepared['design']==c.design() and prepared['worker_inputs']==inputs,'Exact current source and approved design')
    proof=c.read(tests);summary=proof['summary']
    c.require(proof['status']=='passed' and proof['exit_code']==0 and proof['inputs']==inputs
        and summary['failed']==summary['skipped']==0 and summary['passed']==summary['collected']>=40
        and summary['tensor_runtime_loaded'] is False and summary['models']==summary['games']==0,'Complete model-free regression receipt')
    c.require(len(reviews)==3,'Three independent cross-reviews')
    coverage={x['path']:0 for x in inputs};reviewers=set()
    for path in reviews:
        review=c.read(path);reviewers.add(review['reviewer'])
        c.require(review['approved'] is True and not review['blocking_findings'] and review['design']==c.design(),'Independent approval of fixed design')
        excluded=set(review.get('excluded_authored_paths',[]))
        c.require(all(x['path'] not in excluded for x in review['inputs']),'No author approves own implementation')
        for item in review['inputs']:
            c.require(item in inputs,'Review binds exact current input');coverage[item['path']]+=1
    c.require(len(reviewers)==3 and all(v>=1 for v in coverage.values()),'Complete independent source coverage')
    observed=c.read(probe)
    c.require(observed['status']=='passed' and observed['models']==observed['games']==0 and observed['positive_exit']==0
        and observed['negative_exit']==7 and observed['cleanup_passed'] and observed['probe_owner_exit']==0,'Actual detached positive/negative process probe')
    c.verify(observed['source_inputs']);c.require(not c.exact_alive(observed['probe_owner_identity']),'Probe owner actually absent')
    c.verify(inputs);c.verify(prepared['data_inputs'])
    return c.save(c.PLAN,dict(**prepared,tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        detached_probe=c.artifact(probe),frozen_at=c.now()))


def run(expected):
    from . import placement
    owner=control.handshake(c.OUT/'launch',expected,'root');limits=owner['limits']
    placement.set_own(c.SUPERVISOR_CPU);control.bounds(limits)
    c.require(c.sha(c.PLAN)==expected,'Exact frozen plan');plan=c.read(c.PLAN)
    c.require(plan['design']==c.design() and placement.snapshot()==plan['placement'],'Fixed design and actual placement')
    c.verify(plan['worker_inputs']);c.verify(plan['data_inputs'])
    c.save(c.OUT/'claim.json',dict(identity=c.identity(),freeze=c.artifact(c.PLAN),limits=limits,time=c.now()))
    receipts=[];error=None
    try:
        for role in c.JOBS:
            c.status(c.OUT/'stage.json',dict(time=c.now(),stage=role,status='running',completed_jobs=len(receipts)))
            joblimits=control.clipped_limits(limits,c.CAPS[role],model=role in c.MODEL_JOBS)
            command=[sys.executable,'-B','-u','-m',c.PACKAGE+'.worker','--role',role,'--freeze-sha256',expected]
            receipt=control.launch(command,c.OUT/'jobs'/role,joblimits,dict(role=role,freeze_sha256=expected),
                rss=c.MODEL_RSS if role in c.MODEL_JOBS else c.AUDIT_RSS)
            control.passed_job(role);receipts.append(receipt)
        c.verify(plan['worker_inputs']);c.verify(plan['data_inputs']);control.bounds(limits)
    except BaseException:error=traceback.format_exc()
    execution=c.save(c.OUT/'execution.json',dict(status='passed' if error is None else 'failed',freeze_sha256=expected,
        receipts=receipts,error=error,time=c.now()))
    if error:
        c.save(c.OUT/'failure.json',dict(status='incomplete',execution=execution,error=error))
        c.status(c.OUT/'stage.json',dict(status='failed',stage='failed',time=c.now()));return 1
    c.save(c.OUT/'completion.json',dict(status='verified_pending_actual_owner_exit',execution=execution,
        acceptance=c.artifact(c.OUT/'jobs/acceptance/worker/result.json'),automatic_followup=False))
    c.status(c.OUT/'stage.json',dict(status='verified_pending_actual_owner_exit',stage='complete',time=c.now()));return 0


def main():
    from . import placement
    placement.set_own(c.SUPERVISOR_CPU)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','freeze','run'))
    parser.add_argument('--label',default='prepared01');parser.add_argument('--freeze-sha256')
    parser.add_argument('--prepared');parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--probe')
    a=parser.parse_args()
    if a.mode=='prepare':print(c.canonical(prepare(a.label)));return 0
    if a.mode=='freeze':print(c.canonical(freeze(a.prepared,a.tests,a.reviews,a.probe)));return 0
    try:return run(a.freeze_sha256)
    except BaseException:
        error=traceback.format_exc()
        if not (c.OUT/'failure.json').exists():c.save(c.OUT/'failure.json',dict(status='incomplete',stage='root_bootstrap',error=error,time=c.now()))
        print(error,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
