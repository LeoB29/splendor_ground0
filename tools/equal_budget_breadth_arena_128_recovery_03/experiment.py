"""Resume only the frozen independent audit, preserving all games and prior checks."""
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from . import common as c, verify
from tools.equal_budget_breadth_arena_128_recovery_02 import experiment as prior, arena
from tools.equal_budget_breadth_arena_128 import models
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter

ROOT, DATA, OUTPUT, old = c.ROOT, c.DATA, c.AUDIT_OUTPUT, c.old
ENTRY = ROOT/'run_equal_budget_breadth_arena_128_recovery_03.py'
PROTOCOL = ROOT/'docs/engineering/equal_budget_breadth_arena_128_recovery_03.md'
DIAGNOSIS = c.SOURCE/'recovery_diagnosis_03'
TESTS = ['tools/equal_budget_breadth_arena_128_recovery_03/test_verify.py',
         'tools/equal_budget_breadth_arena_128_recovery_03/test_experiment.py']
CANARIES = [13, 310]
REPORTER = None


def identity():
    c.require(prior.validate_freeze() == c.DATA_BINDING, 'R2 frozen implementation changed')
    paths = [*sorted((ROOT/'tools/equal_budget_breadth_arena_128_recovery_03').glob('*.py')), ENTRY, PROTOCOL]
    return {'source_binding':c.DATA_BINDING, 'original_binding':c.ORIGINAL_BINDING,
            'python':sys.version, 'dependencies':{p.relative_to(ROOT).as_posix():old.file_hash(p) for p in paths},
            'checks':TESTS, 'repair':'canonical_json_inherited_snapshot_comparison',
            'gameplay':False, 'resample':False, 'automatic_successor':False,
            'deadline_epoch':c.DEADLINE_EPOCH}


def check(folder):
    folder = Path(folder).resolve()
    c.require(not folder.is_relative_to(c.SOURCE.resolve()), 'Checks must be outside production evidence')
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    args = ['-q',*TESTS,'-p','no:cacheprovider','--basetemp',str(folder/'temp')]
    command = 'import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main('+repr(args)+'))'
    start = time.perf_counter()
    with (folder/'stdout.txt').open('x',encoding='utf-8') as out, (folder/'stderr.txt').open('x',encoding='utf-8') as err:
        try:
            result = subprocess.run([sys.executable,'-u','-c',command],cwd=ROOT,stdin=subprocess.DEVNULL,
                                    stdout=out,stderr=err,timeout=1200,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = -1
    c.require(identity() == before, 'Source changed during checks')
    old.save_new(folder/'checks.json', {'identity':before,'exit_code':code,'seconds':time.perf_counter()-start,
                                      'logs':[c.artifact(folder/n) for n in ('stdout.txt','stderr.txt')]})
    return code


def source_refs():
    names = ['freeze.json','manifest.json','inventory.json','launch.json','main_launch.json',
             'main_games_lock.json','main/audit_sample.json','main/audit_sample_seal.json',
             'main_audits_launch.json','failure.json','abort.json']
    return ([c.artifact(DATA/n) for n in names]
            +[c.artifact(DIAGNOSIS/n) for n in ('diagnose.py','report.json')]
            +[c.artifact(c.SOURCE/n) for n in ('operational_decision.json','models_lock.json','probe_complete.json')])


def validate_receipt(receipt, item, sample_sha):
    c.require(receipt['passed'] is True and receipt['manifest']==c.ORIGINAL_BINDING
              and receipt['recovery_binding']==c.DATA_BINDING, 'Audit provenance changed')
    c.require(receipt['game_artifact']==item and receipt['sample_sha256']==sample_sha,
              'Audit/game/sample linkage differs')
    c.require(receipt['game']['index']==item['index'] and receipt['full_inner_replay'] is False,
              'Audit identity or scope differs')


def build_inputs():
    c.require(prior.validate_freeze(full_inventory=True)==c.DATA_BINDING, 'Source freeze differs')
    models.validate(c.ORIGINAL_BINDING,c.SOURCE)
    sample = verify.validate_sample('main',c.ORIGINAL_BINDING,DATA)
    lock = old.read_json(DATA/'main_games_lock.json')
    games = lock['files']
    c.require([x['index'] for x in games]==list(range(402)), 'Full original panel required')
    for item,spec in zip(games,arena.specs('main'),strict=True):
        game = c.verified(item)
        c.require(game['spec']==spec and 'failure' not in game
                  and game['reason'] in old.VALID_NATURAL_TERMINATIONS, 'Invalid retained game')
    report = old.read_json(DIAGNOSIS/'report.json')
    c.require(report['all_canonical_comparisons_passed'] and report['errors']==[]
              and report['completed_games_checked']==402 and report['inherited_partial_boundaries_checked']==9,
              'Complete outer replay diagnosis required')
    for ref in report['sources']:
        c.verify_hash(ref)
    retained=[]
    for path in sorted((DATA/'main/audits').glob('game_*.json')):
        receipt=old.read_json(path);index=receipt['game']['index']
        validate_receipt(receipt,games[index],old.file_hash(DATA/'main/audit_sample.json'))
        c.require(path.name==f'game_{index:04d}.json', 'Unexpected retained audit filename')
        retained.append(c.artifact(path,index=index))
    indices=[x['index'] for x in retained]
    c.require(indices==[0,1,2,3,4,5,6,7,9,11,12], 'Unexpected prior audit set; inspect before freezing')
    return {'source_binding':c.DATA_BINDING,'refs':source_refs(),'games':games,'retained_audits':retained,
            'pending_indices':[x for x in range(402) if x not in indices],
            'canaries':CANARIES,'sample_sha256':old.file_hash(DATA/'main/audit_sample.json'),
            'sample_coverage':sample['coverage'],'original_deadline_epoch':c.DEADLINE_EPOCH}


def validate_inputs(saved):
    c.require(saved['source_binding']==c.DATA_BINDING and saved['original_deadline_epoch']==c.DEADLINE_EPOCH,
              'Source identity or original deadline differs')
    for ref in saved['refs']:
        c.verify_hash(ref)
    lock=old.read_json(DATA/'main_games_lock.json')
    c.require(lock['files']==saved['games'], 'Retained game panel changed')
    for item in saved['games']:
        c.verify_hash(item)
    for item in saved['retained_audits']:
        receipt=c.verified(item)
        validate_receipt(receipt,saved['games'][item['index']],saved['sample_sha256'])
    retained={x['index'] for x in saved['retained_audits']}
    c.require(len(retained)==len(saved['retained_audits']) and saved['pending_indices']==[i for i in range(402) if i not in retained],
              'Audit partition changed')
    c.require(saved['canaries']==CANARIES and set(CANARIES)<=set(saved['pending_indices']), 'Canary schedule differs')


def resource_decision(saved, now=None, free=None):
    now=time.time() if now is None else now
    free=shutil.disk_usage(ROOT).free if free is None else free
    gate=old.read_json(c.SOURCE/'operational_decision.json')
    # Preserve the original full-panel forecast; observed completed audits can only increase it.
    prior_seconds=gate['main_ordinary_audit_seconds_before_allowance']+gate['main_stress_seconds_before_allowance']
    receipts=[c.verified(r) for r in saved['retained_audits']]
    observed=sum(r['audit_timing']['wall'] for r in receipts)/len(receipts)*len(saved['pending_indices'])/c.CONFIG['workers']
    estimate=c.CONFIG['forecast_allowance']*max(prior_seconds,observed)+3600+c.CONFIG['finalization_allowance_seconds']
    projected=gate['projected_main_bytes']
    return {'forecast_seconds':estimate,'forecast_hours':estimate/3600,
            'original_probe_audit_seconds':prior_seconds,'observed_audit_seconds':observed,
            'deadline_epoch':c.DEADLINE_EPOCH,'elapsed_original_seconds':now-c.ORIGINAL_LAUNCH_EPOCH,
            'free_bytes':free,'projected_additional_bytes':projected,'remaining_games':0,
            'remaining_audits':len(saved['pending_indices']),'outcome_blind':True,
            'reasonable':estimate<=c.CONFIG['maximum_forecast_hours']*3600 and now+estimate<=c.DEADLINE_EPOCH
                         and free-projected>=c.CONFIG['disk_reserve_bytes']}


def locks():
    stack=ExitStack()
    for base in (c.SOURCE,c.PREVIOUS,DATA,OUTPUT):
        stack.enter_context(old.process_lock(base/'run.lock'))
    return stack


def prepare(checks_path):
    c.require(not (OUTPUT/'manifest.json').exists(), 'One-shot preparation already exists')
    checks=old.read_json(checks_path)
    c.require(checks['exit_code']==0 and checks['identity']==identity(), 'Current passing checks required')
    for ref in checks['logs']:c.verify_hash(ref)
    with locks():
        saved=build_inputs();validate_inputs(saved)
        gate=resource_decision(saved)
        c.require(gate['reasonable'], 'Original deadline/storage budget insufficient')
        old.save_new(OUTPUT/'inputs.json',saved)
        old.save_new(OUTPUT/'resource_gate.json',gate)
        manifest={'identity':identity(),'inputs':c.artifact(OUTPUT/'inputs.json'),
                  'resource_gate':c.artifact(OUTPUT/'resource_gate.json'),'checks':c.artifact(checks_path),
                  'authorization':'Owner explicitly requested fix and resume on2026-10-06.',
                  'scope':'Verification only; all402 games and original audit sample unchanged.',
                  'known_provisional_results':{'wins':198,'draws':4,'losses':200}}
        old.save_new(OUTPUT/'manifest.json',manifest)
        old.save_new(OUTPUT/'freeze.json',{'manifest':c.artifact(OUTPUT/'manifest.json'),
                                         'binding':old.digest(manifest),'time':old.now()})
        c.require(validate_freeze()==old.digest(manifest), 'New freeze mismatch')
    print({'prepared':True,'binding':old.digest(manifest),'forecast_hours':gate['forecast_hours']},flush=True)


def validate_freeze(full=False):
    seal=old.read_json(OUTPUT/'freeze.json');manifest=c.verified(seal['manifest'])
    c.require(old.digest(manifest)==seal['binding'] and manifest['identity']==identity(), 'Audit freeze changed')
    checks=c.verified(manifest['checks'])
    c.require(checks['exit_code']==0 and checks['identity']==manifest['identity'], 'Audit checks changed')
    for ref in [*checks['logs'],manifest['inputs'],manifest['resource_gate']]:c.verify_hash(ref)
    if full:validate_inputs(c.verified(manifest['inputs']))
    return seal['binding']


def event(stage,**fields):
    global REPORTER
    if REPORTER is None:REPORTER=StatusReporter(OUTPUT,{'original_binding':c.ORIGINAL_BINDING,'source_binding':c.DATA_BINDING})
    REPORTER.event(stage,**fields)


def abort(exc):
    if not (OUTPUT/'abort.json').exists():
        old.save_new(OUTPUT/'abort.json',{'time':old.now(),'type':type(exc).__name__,'reason':str(exc)})


def worker_initialize(binding):
    folder=OUTPUT/'workers';folder.mkdir(parents=True,exist_ok=True)
    health=c.windows.assert_windowless_worker(folder/f'{os.getpid()}_stdout.txt',folder/f'{os.getpid()}_stderr.txt')
    c.require(old.read_json(OUTPUT/'freeze.json')['binding']==binding, 'Worker audit identity differs')
    arena.initialize_models(base=DATA)
    c.require(torch.get_num_threads()==1 and torch.are_deterministic_algorithms_enabled(), 'Worker arithmetic differs')
    old.save_new(folder/f'{os.getpid()}_health.json',{'audit_recovery_binding':binding,**health})


def audit_job(item,binding):
    c.budget_boundary(DATA)
    try:
        value=verify.audit_game('main',item['index'],item['sha256'],c.ORIGINAL_BINDING,DATA)
        path=OUTPUT/'main/audits'/f"game_{item['index']:04d}.json"
        old.save_new(path,{'manifest':c.ORIGINAL_BINDING,'recovery_binding':c.DATA_BINDING,
                          'audit_recovery_binding':binding,'game_artifact':item,'passed':True,
                          'full_inner_replay':False,**value})
        return c.artifact(path,index=item['index'])
    except BaseException as exc:
        path=OUTPUT/'worker_failures'/f"game_{item['index']:04d}_{os.getpid()}.json"
        old.save_new(path,{'time':old.now(),'index':item['index'],'game':item,
                          'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()})
        raise


def run_pool(arguments,binding,stage,already):
    c.budget_boundary(DATA)
    c.require(stage in ('canary_audits','main_audits') and validate_freeze()==binding, 'Audit stage identity changed')
    launch=OUTPUT/f'{stage}_launch.json'
    old.save_new(launch,{'time':old.now(),'audit_recovery_binding':binding,
                        'indices':[a['index'] for a in arguments]})
    event(stage,completed=already,total=402,stage_completed=0,stage_total=len(arguments))
    runtime=c.windows.configure_spawn()
    iterator=iter(arguments);files=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(c.CONFIG['workers'],len(arguments)),mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_initialize,initargs=(binding,)) as pool:
        pending={}
        def next_job():
            item=next(iterator,None)
            if item is not None:pending[pool.submit(audit_job,item,binding)]=item['index']
        try:
            for _ in range(min(c.CONFIG['workers'],len(arguments))):next_job()
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                batch=[]
                for future in done:
                    pending.pop(future);batch.append(future.result())
                files.extend(batch)
                c.budget_boundary(DATA)
                event(stage,completed=already+len(files),total=402,stage_completed=len(files),stage_total=len(arguments))
                for _ in batch:next_job()
        except BaseException as exc:
            abort(exc)
            for future in pending:future.cancel()
            raise
    c.require(sorted(r['index'] for r in files)==sorted(a['index'] for a in arguments),'Incomplete audit stage')
    return sorted(files,key=lambda r:r['index']),{'seconds':time.perf_counter()-start,'runtime':runtime,'launch':c.artifact(launch)}


def receipt_generation(ref,receipt,saved,binding):
    retained={r['index']:r for r in saved['retained_audits']}
    if ref['index'] in retained:
        c.require(ref==retained[ref['index']] and 'audit_recovery_binding' not in receipt,'Prior audit attribution changed')
    else:
        c.require(receipt.get('audit_recovery_binding')==binding
                  and (ROOT/ref['path']).resolve()==(OUTPUT/'main/audits'/f"game_{ref['index']:04d}.json").resolve(),
                  'New audit attribution changed')


def finish(binding,saved,audit_refs):
    c.budget_boundary(DATA)
    c.require(validate_freeze(full=True)==binding,'Final audit freeze changed')
    c.require(prior.validate_freeze(full_inventory=True)==c.DATA_BINDING,'Preserved ancestral evidence changed')
    models.validate(c.ORIGINAL_BINDING,c.SOURCE)
    sample=verify.validate_sample('main',c.ORIGINAL_BINDING,DATA)
    refs=sorted(audit_refs,key=lambda r:r['index'])
    c.require([r['index'] for r in refs]==list(range(402)),'Exactly one audit for every game required')
    receipts=[];rows=[]
    for item,ref,spec in zip(saved['games'],refs,arena.specs('main'),strict=True):
        c.budget_boundary(DATA)
        game,receipt=c.verified(item),c.verified(ref)
        validate_receipt(receipt,item,saved['sample_sha256']);receipt_generation(ref,receipt,saved,binding)
        c.require(game['spec']==spec and 'failure' not in game and game['reason'] in old.VALID_NATURAL_TERMINATIONS,
                  'Final game identity or ending invalid')
        raw={'index':spec['index'],'pair':spec['pair'],'fit':spec['fit'],'seat':spec['candidate_seat'],
             'score':.5 if game['winner'] is None else int(game['winner']==spec['candidate_seat'])}
        c.require(raw=={k:receipt['game'][k] for k in raw},'Raw game and audited outcome disagree')
        c.require(len(game['trace'])==len(receipt['roots']),'Audited root count differs')
        for move,root in zip(game['trace'],receipt['roots'],strict=True):
            path=c.resolve_root('main',spec['index'],move['turn'],DATA)
            c.require(old.file_hash(path)==move['root_sha256']==root['sha256'],'Audited root changed')
        receipts.append(receipt);rows.append(raw)
    coverage=prior.checked_coverage(receipts,sample,saved['sample_sha256'])
    summary=arena.statistics(rows,'main')
    independent=verify.independent_statistics([r['game'] for r in receipts],'main')
    verify.compare(summary,independent)
    roots=[root for r in receipts for root in r['roots']]
    result={'manifest':binding,'original_binding':c.ORIGINAL_BINDING,'source_binding':c.DATA_BINDING,'time':old.now(),
            **summary,'rows':[r['game'] for r in receipts],'coverage':coverage,'all_prescribed_audits_pass':True,
            'full_inner_replay':False,'amendment':'coupled_repetition_original_order_v1',
            'verification_repair':'canonical_json_inherited_snapshot_comparison',
            'all_completed_games_reused':402,'prior_passed_audits_reused':len(saved['retained_audits']),
            'elapsed_hours_from_original_launch':(time.time()-c.ORIGINAL_LAUNCH_EPOCH)/3600,
            'continuations_by_role':{role:sum(r['continuations'] for r in roots if r['planner']==role) for role in ('candidate','reference')},
            'unresolved_by_role':{role:sum(r['unresolved_continuations'] for r in roots if r['planner']==role) for role in ('candidate','reference')},
            'decision_seconds':{role:{'count':sum(r['planner']==role for r in roots),
                'total':sum(r['timing']['wall']+r.get('inherited_timing',{}).get('wall',0) for r in roots if r['planner']==role),
                'inherited_failed_root_work':sum(r.get('inherited_timing',{}).get('wall',0) for r in roots if r['planner']==role)}
                for role in ('candidate','reference')},'automatic_successor':False}
    old.save_new(OUTPUT/'main_audits_lock.json',{'audit_recovery_binding':binding,'files':refs})
    old.save_new(OUTPUT/'result.json',result)
    old.save_new(OUTPUT/'final_audit.json',{'manifest':binding,'passed':True,'time':old.now(),
                  'independent_statistics':independent,'raw_game_endpoints_agree':True,'coverage':coverage,
                  'result':c.artifact(OUTPUT/'result.json'),'inputs':c.artifact(OUTPUT/'inputs.json'),
                  'audit_lock':c.artifact(OUTPUT/'main_audits_lock.json')})
    lines=['# Equal budget breadth arena: verification resumed','',f"Classification: {summary['classification']}.",'',
           f"{summary['wins']} wins, {summary['draws']} draws, {summary['losses']} losses across 402 games / 201 pairs.",
           f"Score {summary['share']:.4%}; paired 95% interval [{summary['ci95'][0]:.4%}, {summary['ci95'][1]:.4%}].",'',
           'All 402 collected games were preserved. Eleven passed audits were retained; 391 were completed with the corrected JSON checkpoint comparison. The original sealed ordinary/stress sample and all unresolved-cell checks were preserved.',
           'The repair changes representation comparison only. Both disclosed decision-certificate amendments remain part of the study. No score imputation, new games, changed model or sampling rule, or relaxed safeguard was introduced.',
           f"Recorded continuations: {coverage['recorded_continuations']}; unresolved: {coverage['unresolved_continuations']}; independently replayed: {coverage['replayed_continuations']}.",
           f"Coupled roots: {coverage['coupled_certified_roots']}; proof cells: {coverage['equivalence_proof_cells']}; unknown groups: {coverage['coupled_unknown_groups']}.",
           'The original observed-score >=55% and paired 95% lower >50% gate is retained. Full inner replay was not performed. No automatic successor.','']
    with (OUTPUT/'report.md').open('x',encoding='utf-8',newline='\n') as handle:handle.write('\n'.join(lines))
    c.budget_boundary(DATA)
    old.save_new(OUTPUT/'final_status.json',{'manifest':binding,'time':old.now(),'complete':True,
                  'all_prescribed_audits_pass':True,'classification':summary['classification'],
                  'result':c.artifact(OUTPUT/'result.json'),'audit':c.artifact(OUTPUT/'final_audit.json'),
                  'report':c.artifact(OUTPUT/'report.md'),'automatic_successor':False})
    event('complete',completed=402,total=402,score=summary['share'],classification=summary['classification'])


def run():
    global REPORTER
    runtime=OUTPUT/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    c.windows.assert_windowless_worker(runtime/'stdout.txt',runtime/'stderr.txt')
    with locks():
        c.require(not (OUTPUT/'launch.json').exists(),'One-shot audit recovery; preserve attempts')
        binding=validate_freeze(full=True)
        saved=old.read_json(OUTPUT/'inputs.json')
        REPORTER=StatusReporter(OUTPUT,{'manifest':binding,'source_binding':c.DATA_BINDING,'original_binding':c.ORIGINAL_BINDING})
        try:
            with prior.original.parent.inhibit_system_sleep() as awake:
                c.budget_boundary(DATA);gate=resource_decision(saved)
                c.require(gate['reasonable'],'Remaining original resource budget no longer fits')
                old.save_new(OUTPUT/'launch.json',{'time':old.now(),'pid':os.getpid(),'audit_recovery_binding':binding,
                    'original_launch_epoch':c.ORIGINAL_LAUNCH_EPOCH,'deadline_epoch':c.DEADLINE_EPOCH,
                    'windowless':c.windows.assert_windowless_worker(),'sleep_inhibition':awake,'resource_gate':gate})
                canaries,first=run_pool([saved['games'][i] for i in CANARIES],binding,'canary_audits',len(saved['retained_audits']))
                old.save_new(OUTPUT/'canaries_complete.json',{'audit_recovery_binding':binding,'time':old.now(),'files':canaries,**first})
                pending=[saved['games'][i] for i in saved['pending_indices'] if i not in CANARIES]
                audits,rest=run_pool(pending,binding,'main_audits',len(saved['retained_audits'])+len(canaries))
                old.save_new(OUTPUT/'audit_execution.json',{'canaries':first,'remaining':rest,'time':old.now()})
                event('finalizing',completed=402,total=402)
                finish(binding,saved,[*saved['retained_audits'],*canaries,*audits])
        except BaseException as exc:
            abort(exc)
            old.save_new(OUTPUT/'failure.json',{'manifest':binding,'time':old.now(),'type':type(exc).__name__,
                                               'reason':str(exc),'traceback':traceback.format_exc()})
            event('failed',failure='failure.json',reason=str(exc))
            raise


def launch():
    binding=validate_freeze();folder=OUTPUT/'runtime';folder.mkdir(parents=True,exist_ok=True)
    with old.process_lock(folder/'launcher.lock'):
        c.require(not (folder/'process.json').exists() and not (OUTPUT/'launch.json').exists(),'Already launched')
        process=c.windows.detached_launch([sys.executable,'-u',ENTRY,'run'],folder/'stdout.txt',folder/'stderr.txt',ROOT)
        old.save_new(folder/'process.json',{**process,'audit_recovery_binding':binding,'time':old.now()})
        deadline=time.monotonic()+600
        while not (OUTPUT/'launch.json').exists():
            c.require(not (OUTPUT/'failure.json').exists(),'Audit failed before acknowledgement; inspect preserved logs')
            c.require(time.monotonic()<deadline,'No acknowledgement; do not automatically retry')
            time.sleep(.05)
        receipt=old.read_json(OUTPUT/'launch.json')
        c.require(receipt['pid']==process['pid'] and receipt['audit_recovery_binding']==binding,'Launch acknowledgement mismatch')
        old.save_new(folder/'acknowledged.json',{'time':old.now(),'pid':process['pid'],'launch':c.artifact(OUTPUT/'launch.json')})
    print({'launched':True,'pid':process['pid'],'binding':binding},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','prepare','launch','run'))
    parser.add_argument('--checks',type=Path)
    args=parser.parse_args()
    if args.command=='check':
        c.require(args.checks is not None,'New check folder required');return check(args.checks)
    if args.command=='prepare':
        c.require(args.checks is not None,'Passing checks receipt required');prepare(args.checks)
    elif args.command=='launch':launch()
    else:run()
    return 0
