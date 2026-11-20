"""Exclusive preparation and freeze of the disclosed partial-game recovery."""
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


def original_files(folder=None):
    pending=[c.ORIGINAL_OUT if folder is None else Path(folder)]
    while pending:
        folder=pending.pop()
        with os.scandir(folder) as entries:rows=sorted(entries,key=lambda x:x.name)
        for entry in rows:
            c.require(not entry.is_symlink(),'Original evidence has no symbolic links')
            if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):yield Path(entry.path)


def inventory(folder=None,label="original"):
    """Preserve the complete stopped tree without copying or editing evidence."""
    destination=c.OUT/'prelaunch'/f'{label}_preservation_inventory.jsonl'
    destination.parent.mkdir(parents=True,exist_ok=True)
    count=total=0;began=time.monotonic()
    with destination.open('x',encoding='utf-8',newline='\n') as stream:
        for path in original_files(folder):
            before=path.stat();item=c.artifact(path);after=path.stat()
            c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
                      (after.st_size,after.st_mtime_ns,after.st_ino),'Stopped evidence remained stable while hashing')
            stream.write(c.canonical(item)+'\n');count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    result=dict(status='preserved_inventory',inventory=c.artifact(destination),files=count,bytes=total,
        original_attempt_status='incomplete',elapsed_seconds=time.monotonic()-began,time=c.now(),models=0,games=0)
    c.save(c.OUT/'prelaunch'/f'{label}_preservation.json',result)
    return result


def prepare():
    from . import partial135
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(c.sha(c.ORIGINAL_OUT/'freeze.json')==c.ORIGINAL_FREEZE_SHA,'Exact frozen original')
    c.require(c.sha(c.PREVIOUS_OUT/'freeze.json')==c.PREVIOUS_FREEZE_SHA,'Exact previous recovery')
    old=c.read(c.ORIGINAL_OUT/'freeze.json');c.verify(old['inputs'])
    previous=c.read(c.PREVIOUS_OUT/'freeze.json');c.verify(previous['inputs'])
    c.require(verify_runtime_sources()==old['corrected_source_receipt'],'Original corrected sources unchanged')
    plan=c.read(c.ORIGINAL_OUT/'collection/plan.json')
    report=partial135.build_boundary(plan)
    c.save(c.OUT/'prelaunch/saved_boundary.json',report)
    # Original tree already has an authenticated whole-tree inventory in prior recovery.
    # Reuse those bytes; final acceptance re-verifies every original file.
    original=c.read(c.PREVIOUS_OUT/'prelaunch/original_preservation.json')
    c.verify([original['inventory']])
    destination=c.OUT/'prelaunch/original_preservation_inventory.jsonl'
    with destination.open('xb') as out:out.write(c.resolve(original['inventory']).read_bytes())
    c.require(c.sha(destination)==original['inventory']['sha256'],'Original inventory byte identity')
    preserved=dict(original,inventory=c.artifact(destination),reused_from=c.artifact(c.PREVIOUS_OUT/'prelaunch/original_preservation.json'))
    c.save(c.OUT/'prelaunch/original_preservation.json',preserved)
    c.require(c.sha(c.PRELAUNCH_REFUSED_OUT/'freeze.json')==c.PRELAUNCH_REFUSED_SHA,'Exact preserved prelaunch refusal')
    refused=c.read(c.PRELAUNCH_REFUSED_OUT/'freeze.json');c.verify(refused['inputs'])
    c.require(not any((c.PRELAUNCH_REFUSED_OUT/name).exists() for name in
        ('claim.json','launcher_claim.json','stage.json','completion.json')),'Earlier freeze refused before study or models launched')
    previous_preserved=c.read(c.PRELAUNCH_REFUSED_OUT/'prelaunch/previous_preservation.json')
    c.verify([previous_preserved['inventory']])
    c.save(c.OUT/'prelaunch/previous_preservation.json',dict(previous_preserved,
        reused_from=c.artifact(c.PRELAUNCH_REFUSED_OUT/'prelaunch/previous_preservation.json')))
    from . import preservation
    refused_preserved=preservation.inventory(c.PRELAUNCH_REFUSED_OUT,
        c.OUT/'prelaunch/refused_preparation_inventory.jsonl')
    c.save(c.OUT/'prelaunch/refused_preparation_preservation.json',refused_preserved)
    from . import historical_ownership
    excluded=historical_ownership.validate_proof(c.read(historical_ownership.PROOF),require_frozen=False)
    c.save(c.OUT/'prelaunch/launch_ownership_amendment.json',dict(status='qualified_historical_false_observations',
        exact_excluded_launch_observations=excluded,all_authentic_workers_still_required_absent=True,
        previous_launch_cleanup_remains_failed=True,unrelated_apps_untouched=True))
    destination=c.OUT/'collection/plan.json';destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as target:target.write((c.ORIGINAL_OUT/'collection/plan.json').read_bytes())
    c.require(c.sha(destination)==c.sha(c.ORIGINAL_OUT/'collection/plan.json'),'Original scientific contract retained byte for byte')
    authorization=dict(time=c.now(),user_request='diagnose the saved timeout, implement and qualify a separate recovery, then resume if checks pass. Keep the teacher and original deadlines unchanged and preserve all completed games and failures.',
        interpretation='One separately disclosed retry of each of eight unfinished decisions; no completed game or move replaced.',
        original_attempt_remains_incomplete=True,previous_recovery_remains_incomplete=True,
        completed_games_retained=2168,original_verified_games=2160,
        permitted_gameplay_retries=[dict(game_id='g'+str(g),ply=p,maximum_repeats=1) for g,p in
            [(2168,27),(2169,23),(2170,22),(2171,22),(2172,22),(2173,21),(2174,20),(2175,17)]],
        unstarted_selected_replays_maximum=48,completed_selected_replays_retained=6480,
        supplemental_seconds=7200,original_cell_clock_not_reset=True,original_global_model_collection_utc_deadlines_unchanged=True,
        scientific_game_retries=0,game_replacements=0,unfinished_decision_retries=8,training_updates=0,
        automatic_retry_after_recovery_failure=False)
    c.save(c.OUT/'recovery_authorization.json',authorization)
    return dict(status='prepared',boundary=c.artifact(c.OUT/'prelaunch/saved_boundary.json'),
        preservation=preserved,previous_preservation=c.artifact(c.OUT/'prelaunch/previous_preservation.json'),models=0,games=0)


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
    names=('tests_recovery_owner.py','tests_worker_recovery.py')
    paths=[package/name for name in names]+sorted((package/'tests').glob('test_*.py'))+sorted((package/'diagnostics').glob('test_*.py'))
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
    from . import partial135,owner
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
    partial135.validate_boundary(c.read(c.resolve(boundary_item)),c.read(c.OUT/'collection/plan.json'))
    old=c.read(c.ORIGINAL_OUT/'freeze.json');c.verify(old['inputs'])
    preserved=c.read(c.OUT/'prelaunch/original_preservation.json')
    previous_preserved=c.read(c.OUT/'prelaunch/previous_preservation.json')
    previous=c.read(c.PREVIOUS_OUT/'freeze.json');c.verify(previous['inputs'])
    refused=c.read(c.PRELAUNCH_REFUSED_OUT/'freeze.json');c.verify(refused['inputs'])
    refused_preserved=c.read(c.OUT/'prelaunch/refused_preparation_preservation.json')
    plan=deepcopy(old)
    plan.update(recovery_schema=c.SCHEMA,original_freeze=c.artifact(c.ORIGINAL_OUT/'freeze.json'),
        original_observed_launcher_exit=c.artifact(c.ROOT/'outputs/splendorust_collection4096_launch_20261115_01/observed_launcher_exit.json'),
        original_preservation_inventory=preserved['inventory'],original_preservation=preserved,
        original_evidence_bytes=preserved['bytes'],saved_boundary=boundary_item,saved_partial135=boundary_item,
        previous_freeze=c.artifact(c.PREVIOUS_OUT/'freeze.json'),
        previous_preservation_inventory=previous_preserved['inventory'],previous_preservation=previous_preserved,
        previous_evidence_bytes=previous_preserved['bytes'],
        prelaunch_refused_freeze=c.artifact(c.PRELAUNCH_REFUSED_OUT/'freeze.json'),
        prelaunch_refused_preservation=refused_preserved,
        prelaunch_refusal_observer=[c.artifact(c.ROOT/'outputs/splendorust_collection4096_resume_launch_20261118_01'/name)
            for name in ('launch_observer.py','observer_stdout.log','observer_stderr.log')],
        historical_ownership_proof=c.artifact(c.OUT/'prelaunch/historical_ownership01.json'),
        ownership_amendment=c.artifact(c.OUT/'prelaunch/launch_ownership_amendment.json'),
        previous_observed_launcher_exit=c.artifact(c.ROOT/'outputs/splendorust_collection4096_recovery_launch_20261116_01/observed_launcher_exit.json'),
        recovery=owner.recovery_contract(),recovery_authorization=c.artifact(c.OUT/'recovery_authorization.json'),
        protocol=c.artifact(c.PROTOCOL),collection_plan=c.artifact(c.OUT/'collection/plan.json'),
        tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],
        detached_lifecycle_qualification=c.artifact(probe),time=c.now(),
        prelaunch_evidence=[c.artifact(p) for folder in ('prelaunch','test_artifacts')
            for p in sorted((c.OUT/folder).rglob('*')) if p.is_file()])
    plan['worker_inputs']=previous['worker_inputs']+code
    items=refused['inputs']+code+list(c.nested_artifacts(plan))
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
