"""Prospective separate recovery freeze; original incomplete evidence stays immutable."""
import argparse,sys
from . import common as c,checks

def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    original=c.read(c.OLD_PLAN)
    paths=list((c.ROOT/c.PACKAGE.replace('.','/')).glob('*.py'))
    items=[c.artifact(p) for p in python_dependencies(paths)]+original['worker_inputs']+[c.artifact(c.PROTOCOL)]
    return sorted({i['path']:i for i in items}.values(),key=lambda i:i['path'])

def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from .control import set_own_cpu
    set_own_cpu();c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'One separate recovery before launch')
    sources=verify_runtime_sources();c.require(c.sha(c.OLD_PLAN)==c.OLD_SHA,'Unchanged original freeze')
    original=c.read(c.OLD_PLAN);exit=c.read(c.OLD_OUT/'launch/actual_exit.json')
    c.require(exit['exit_code']==1 and exit['exact_owned_absence'] and not exit['cleanup_errors'],'Preserved failed owner cleanly exited')
    ids=exit['observed_owned_identities']+[exit['identity'],c.read(c.OLD_OUT/'launcher_claim.json')['identity']]
    c.require(all(not c.exact_alive(i) for i in ids),'All original owners absent before recovery')
    c.verify(original['worker_inputs']);c.verify(original['data_inputs'])
    data=original['data_inputs']+[c.artifact(p) for p in c.OLD_OUT.rglob('*') if p.is_file()]+checks.exception_inputs()
    data=sorted({i['path']:i for i in data}.values(),key=lambda i:i['path']);c.verify(data)
    limits=c.recovery_limits();c.check_deadlines(limits)
    return c.save(c.OUT/'prepared_reviewed.json',dict(design=c.design(),worker_inputs=source_inputs(),data_inputs=data,
        original_freeze=c.artifact(c.OLD_PLAN),original_limits=c.read(c.OLD_OUT/'launch/owner.json')['limits'],
        original_analysis_limits=c.read(c.OLD_OUT/'jobs/analysis/owner.json')['limits'],
        cleanup_exception=checks.exception_contract(),corrected_source_receipt=sources,limits=limits,
        original_failure=c.artifact(c.OLD_OUT/'failure.json'),original_exact_owners_absent=ids,time=c.now()))

def freeze(tests,reviews):
    prepared=c.read(c.OUT/'prepared_reviewed.json')
    c.require(prepared['design']==c.design() and prepared['worker_inputs']==source_inputs(),'Source and design unchanged since preparation')
    proof=c.read(tests)
    c.require(proof['status']=='passed' and proof['exit_code']==0 and proof['inputs']==prepared['worker_inputs'] and
        proof['summary']['passed']==proof['summary']['collected']>=20 and proof['summary']['failed']==proof['summary']['skipped']==0 and
        proof['summary']['tensor_runtime_loaded'] is False and proof['summary']['models']==proof['summary']['new_games']==0,'Current full model-free qualification')
    c.require(len(reviews)==3,'Three scoped independent reviews')
    names=set()
    for path in reviews:
        review=c.read(path);names.add(review['reviewer'])
        c.require(review['approved'] is True and not review['blocking_findings'] and review['inputs']==prepared['worker_inputs'] and review['design']==c.design(),'Current scoped independent approval')
    c.require(len(names)==3,'Distinct reviewers')
    fixture=c.OUT/'prelaunch/detached_fixture_final/actual_exit.json';f=c.read(fixture)
    c.require(f['status']=='passed' and f['exit_code']==0 and f['exact_owned_absence'] and f['actual_exit_observed'] and not f['cleanup_errors'] and f['error'] is None,'Detached model-free ownership qualification')
    c.verify(f['source_inputs']);launcher=c.read(c.OUT/'prelaunch/fixture_launcher_final.json')['identity']
    c.require(not c.exact_alive(launcher) and not c.exact_alive(f['identity']),'Detached fixture launcher cleanup')
    c.verify(prepared['worker_inputs']);c.verify(prepared['data_inputs']);c.check_deadlines(prepared['limits'])
    return c.save(c.PLAN,dict(**prepared,tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],detached_fixture=c.artifact(fixture),frozen_at=c.now()))

if __name__=='__main__':
    from .control import set_own_cpu
    set_own_cpu();p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','freeze'));p.add_argument('--tests');p.add_argument('--reviews',nargs='*');a=p.parse_args()
    print(c.canonical(prepare() if a.mode=='prepare' else freeze(a.tests,a.reviews)))
