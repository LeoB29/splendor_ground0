"""Exclusive preparation and freeze of the disclosed verification-only recovery."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import common as c


def source_inputs():
    package=c.ROOT/c.PACKAGE.replace('.','/')
    return [c.artifact(p) for p in sorted(package.rglob('*.py'))]+[c.artifact(c.PROTOCOL)]


def original_files():
    pending=[c.ORIGINAL_OUT]
    while pending:
        folder=pending.pop()
        with os.scandir(folder) as entries:rows=sorted(entries,key=lambda x:x.name)
        for entry in rows:
            c.require(not entry.is_symlink(),'Original evidence has no symbolic links')
            if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):yield Path(entry.path)


def inventory():
    """Preserve the complete stopped tree without copying or editing evidence."""
    destination=c.OUT/'prelaunch/original_preservation_inventory.jsonl'
    destination.parent.mkdir(parents=True,exist_ok=True)
    count=total=0;began=time.monotonic()
    with destination.open('x',encoding='utf-8',newline='\n') as stream:
        for path in original_files():
            before=path.stat();item=c.artifact(path);after=path.stat()
            c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
                      (after.st_size,after.st_mtime_ns,after.st_ino),'Stopped evidence remained stable while hashing')
            stream.write(c.canonical(item)+'\n');count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    result=dict(status='preserved_inventory',inventory=c.artifact(destination),files=count,bytes=total,
        original_attempt_status='incomplete',elapsed_seconds=time.monotonic()-began,time=c.now(),models=0,games=0)
    c.save(c.OUT/'prelaunch/original_preservation.json',result)
    return result


def prepare():
    from . import boundary
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(c.sha(c.ORIGINAL_OUT/'freeze.json')==c.ORIGINAL_FREEZE_SHA,'Exact frozen original')
    old=c.read(c.ORIGINAL_OUT/'freeze.json');c.verify(old['inputs'])
    c.require(verify_runtime_sources()==old['corrected_source_receipt'],'Original corrected sources unchanged')
    report=boundary.inspect_boundary()
    c.save(c.OUT/'prelaunch/saved_boundary.json',report)
    preserved=inventory()
    destination=c.OUT/'collection/plan.json';destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as target:target.write((c.ORIGINAL_OUT/'collection/plan.json').read_bytes())
    c.require(c.sha(destination)==c.sha(c.ORIGINAL_OUT/'collection/plan.json'),'Original scientific contract retained byte for byte')
    authorization=dict(time=c.now(),user_request='my computer got turned off in the middle, can you check where you left off and resume, if possible',
        interpretation='Technical recovery of interrupted verification, with no retry or replacement of scientific gameplay.',
        original_attempt_remains_incomplete=True,completed_games_retained=672,original_verified_games=656,
        permitted_verification_retry=dict(game_id='g0666',ply=40,original_operation_id=935,maximum_repeats=1),
        unstarted_selected_replays=15,completed_selected_replays_retained=32,
        supplemental_seconds=3600,original_cell_clock_not_reset=True,original_global_model_collection_utc_deadlines_unchanged=True,
        scientific_game_retries=0,game_replacements=0,training_updates=0,automatic_retry_after_recovery_failure=False)
    c.save(c.OUT/'recovery_authorization.json',authorization)
    return dict(status='prepared',boundary=c.artifact(c.OUT/'prelaunch/saved_boundary.json'),
        preservation=preserved,models=0,games=0)


def suite():
    import pytest
    class Counts:
        collected=passed=failed=skipped=0
        def pytest_collection_finish(self,session):self.collected=len(session.items)
        def pytest_runtest_logreport(self,report):
            if report.failed:self.failed+=1
            if report.skipped:self.skipped+=1
            if report.when=='call' and report.passed:self.passed+=1
    counts=Counts();package=c.ROOT/c.PACKAGE.replace('.','/')
    names=('tests_boundary.py','tests_partial.py','tests_recovery_owner.py','tests_recovery_corpus.py','tests_worker_recovery.py')
    paths=[package/name for name in names]
    c.require(all(p.exists() for p in paths),'All recovery test areas present')
    # Retain the original suite as a source-preservation regression, without rebinding its modules.
    paths+=sorted((c.ROOT/'tools/splendorust_collection4096_20261115').rglob('tests_*.py'))
    paths+=sorted((c.ROOT/'tools/splendorust_collection4096_20261115').rglob('contract_tests.py'))
    result=pytest.main(['-q',*[str(p) for p in paths]],plugins=[counts])
    summary=dict(tests_collected=counts.collected,tests_passed=counts.passed,failures=counts.failed,
        skipped=counts.skipped,tensor_runtime_loaded='torch' in sys.modules,models=0,games=0)
    print(c.canonical(summary),flush=True)
    return 0 if result==0 and counts.passed==counts.collected and not counts.skipped and 'torch' not in sys.modules else 1


def record_tests(label):
    from .owner import child_environment
    c.require(label.isalnum(),'Exclusive simple test label')
    inputs=source_inputs();log=c.OUT/'prelaunch'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    env=child_environment();env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    with log.open('xb') as stream:
        proc=subprocess.run([sys.executable,'-B','-u','-m',c.PACKAGE+'.preparation','suite'],cwd=c.ROOT,
            env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    c.verify(inputs)
    lines=log.read_text(encoding='utf-8').splitlines()
    summary=json.loads(lines[-1]) if lines and lines[-1].startswith('{') else None
    return c.save(c.OUT/'prelaunch'/(label+'.json'),dict(status='passed' if proc.returncode==0 else 'failed',
        exit_code=proc.returncode,inputs=inputs,suite_summary=summary,log=c.artifact(log),time=c.now(),models=0,games=0))


def freeze(tests,reviews,probe):
    from . import boundary,owner
    c.require(not c.PLAN.exists() and not (c.OUT/'claim.json').exists(),'Exclusive separate recovery freeze')
    code=source_inputs();proof=c.read(tests);summary=proof['suite_summary']
    c.require(proof['status']=='passed' and proof['exit_code']==0 and proof['inputs']==code and summary and
        summary['tests_passed']==summary['tests_collected'] and summary['tests_passed']>=50 and
        summary['failures']==summary['skipped']==0 and not summary['tensor_runtime_loaded'],'Complete current-source model-free tests')
    c.verify(code+[proof['log']]);covered=set();authors=set()
    for path in reviews:
        review=c.read(path);c.require(review['approved'] and not review['blocking_findings'],'Independent recovery review approved')
        c.verify(review['inputs']);covered.update(x['path'] for x in review['inputs']);authors.add(review['reviewer'])
    production={x['path'] for x in code if not Path(x['path']).name.startswith('tests_')}
    c.require(len(authors)>=2 and production<=covered,'Two independent cross-reviews cover every recovery source')
    lifecycle=c.read(probe);c.require(lifecycle['status']=='passed' and lifecycle['actual_probe_owner_exit']==0
        and lifecycle['actual_probe_owner_absent'] and lifecycle['detached'] and lifecycle['error'] is None
        and lifecycle['models']==lifecycle['games']==0,'Actual detached lifecycle and preserved intentional failure qualified')
    c.verify(lifecycle['inputs'])
    c.require(c.artifact(owner.__file__) in lifecycle['inputs'],'Actual current recovery supervisor probed')
    boundary_item=c.artifact(c.OUT/'prelaunch/saved_boundary.json')
    boundary.validate_boundary(c.read(c.resolve(boundary_item)))
    old=c.read(c.ORIGINAL_OUT/'freeze.json');c.verify(old['inputs'])
    preserved=c.read(c.OUT/'prelaunch/original_preservation.json')
    plan=deepcopy(old)
    plan.update(recovery_schema=c.SCHEMA,original_freeze=c.artifact(c.ORIGINAL_OUT/'freeze.json'),
        original_observed_launcher_exit=c.artifact(c.ROOT/'outputs/splendorust_collection4096_launch_20261115_01/observed_launcher_exit.json'),
        original_preservation_inventory=preserved['inventory'],original_preservation=preserved,
        original_evidence_bytes=preserved['bytes'],saved_boundary=boundary_item,
        recovery=owner.recovery_contract(),recovery_authorization=c.artifact(c.OUT/'recovery_authorization.json'),
        protocol=c.artifact(c.PROTOCOL),collection_plan=c.artifact(c.OUT/'collection/plan.json'),
        tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        detached_lifecycle_qualification=c.artifact(probe),time=c.now(),
        prelaunch_evidence=[c.artifact(p) for folder in ('prelaunch','test_artifacts')
            for p in sorted((c.OUT/folder).rglob('*')) if p.is_file()])
    plan['worker_inputs']=old['worker_inputs']+code
    items=old['inputs']+code+list(c.nested_artifacts(plan))
    unique={}
    for item in items:c.require(unique.setdefault(item['path'],item)==item,'Consistent frozen recovery input')
    plan['inputs']=sorted(unique.values(),key=lambda x:x['path']);c.verify(plan['inputs'])
    owner.validate_plan(plan)
    c.require('torch' not in sys.modules,'Freeze loads no learned model')
    return c.save(c.PLAN,plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','suite','tests','freeze'))
    parser.add_argument('--label');parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--probe')
    args=parser.parse_args()
    if args.operation=='suite':raise SystemExit(suite())
    value=prepare() if args.operation=='prepare' else record_tests(args.label) if args.operation=='tests' else freeze(args.tests,args.reviews,args.probe)
    print(c.canonical(value),flush=True)
