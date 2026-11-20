"""Hidden, lineage-preserving recovery with exact decision certificates."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from . import common as c, inventory, arena, verify
from tools.equal_budget_breadth_arena_128 import experiment as original
from tools.equal_budget_breadth_arena_128 import models
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter

ROOT, OUTPUT, SOURCE, old = c.ROOT, c.OUTPUT, c.SOURCE, c.old
ENTRY = ROOT / 'run_equal_budget_breadth_arena_128_recovery_01.py'
PROTOCOL = ROOT / 'docs/engineering/equal_budget_breadth_arena_128_recovery_01.md'
CHECK_TARGETS = [f'tools/equal_budget_breadth_arena_128_recovery_01/{name}' for name in
                 ('test_inventory.py','test_arena.py','test_verify.py','test_experiment.py','test_worker.py')]
REPORTER = None


def identity():
    c.require(original.validate_freeze() == c.ORIGINAL_BINDING, 'Original freeze changed')
    paths = sorted((ROOT/'tools/equal_budget_breadth_arena_128_recovery_01').glob('*.py'))
    paths += [ENTRY, PROTOCOL]
    return {'original_binding':c.ORIGINAL_BINDING, 'python':sys.version,
            'dependencies':{p.relative_to(ROOT).as_posix():old.file_hash(p) for p in paths},
            'checks':CHECK_TARGETS,'amendment':'repetition_cutoff_interval_certificate_v1',
            'original_launch_epoch':c.ORIGINAL_LAUNCH_EPOCH,
            'automatic_retry':False,'automatic_successor':False}


def check(folder):
    folder = Path(folder)
    c.require(not folder.resolve().is_relative_to(SOURCE.resolve()), 'Fixtures must be outside original/recovery production outputs')
    folder.mkdir(parents=True,exist_ok=False)
    before = identity()
    args = ['-q',*CHECK_TARGETS,'-p','no:cacheprovider','--basetemp',str(folder/'temp')]
    code = 'import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main('+repr(args)+'))'
    start = time.perf_counter()
    with (folder/'stdout.txt').open('x',encoding='utf-8') as out, (folder/'stderr.txt').open('x',encoding='utf-8') as err:
        try:
            run = subprocess.run([sys.executable,'-u','-c',code],cwd=ROOT,stdin=subprocess.DEVNULL,
                                 stdout=out,stderr=err,timeout=1800,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            exit_code,timed_out = run.returncode,False
        except subprocess.TimeoutExpired:
            exit_code,timed_out = -1,True
    c.require(identity() == before,'Recovery changed during checks')
    old.save_new(folder/'checks.json',{'identity':before,'exit_code':exit_code,'timeout':timed_out,
                 'seconds':time.perf_counter()-start,'logs':[c.artifact(folder/n) for n in ('stdout.txt','stderr.txt')]})
    return exit_code


def resource_decision(prior, elapsed, free_bytes):
    remaining = prior['remaining_forecast_seconds']
    projected = prior['projected_main_bytes']
    c.require(prior['reasonable'] is True and prior['outcome_blind'] is True
              and prior['manifest'] == c.ORIGINAL_BINDING,'Original cost gate not accepted')
    c.require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0
                  for v in (remaining,projected,elapsed,free_bytes))
              and remaining > 0 and projected > 0,'Invalid resource evidence')
    return {'remaining_forecast_hours':remaining/3600,'remaining_forecast_seconds':remaining,
            'projected_additional_bytes':projected,'free_bytes':free_bytes,'elapsed_original_seconds':elapsed,
            'deadline_epoch':c.ORIGINAL_LAUNCH_EPOCH+c.CONFIG['maximum_runtime_hours']*3600,
            'deduct_completed_work':False,'outcome_blind':True,
            'reasonable':remaining <= c.CONFIG['maximum_forecast_hours']*3600
                         and elapsed+remaining <= c.CONFIG['maximum_runtime_hours']*3600
                         and free_bytes-projected >= c.CONFIG['disk_reserve_bytes']}


def prepare(checks_path):
    c.require(not (OUTPUT/'manifest.json').exists(),'Recovery already prepared')
    checks = old.read_json(checks_path)
    c.require(checks['exit_code'] == 0 and checks['identity'] == identity(),'Current passing checks required')
    for item in checks['logs']:
        c.verify_hash(item)
    with old.process_lock(SOURCE/'run.lock'):
        saved = inventory.build()
        inventory.validate(saved)
        decision = resource_decision(old.read_json(SOURCE/'operational_decision.json'),
                                    time.time()-c.ORIGINAL_LAUNCH_EPOCH,shutil.disk_usage(ROOT).free)
        c.require(decision['reasonable'],'Original remaining time or storage budget is insufficient')
        old.save_new(OUTPUT/'inventory.json',saved)
        old.save_new(OUTPUT/'resource_gate.json',decision)
        manifest = {'identity':identity(),'original_binding':c.ORIGINAL_BINDING,
                    'inventory':c.artifact(OUTPUT/'inventory.json'),'resource_gate':c.artifact(OUTPUT/'resource_gate.json'),
                    'checks':c.artifact(checks_path),'owner_authorized':True,
                    'authorization':'Owner requested diagnosis and valid recovery on2026-10-04; no new training or successor.',
                    'scope':'Same schedule/models; conservative interval extension for repetition cutoffs only.'}
        old.save_new(OUTPUT/'manifest.json',manifest)
        old.save_new(OUTPUT/'freeze.json',{'manifest':c.artifact(OUTPUT/'manifest.json'),
                     'binding':old.digest(manifest),'time':old.now()})
        c.require(validate_freeze() == old.digest(manifest),'Recovery freeze mismatch')
    print({'prepared':True,'binding':old.digest(manifest),'remaining_forecast_hours':decision['remaining_forecast_hours']},flush=True)


def validate_freeze(full_inventory=False):
    freeze = old.read_json(OUTPUT/'freeze.json')
    value = c.verified(freeze['manifest'])
    c.require(old.digest(value) == freeze['binding'] and value['owner_authorized'] is True,'Recovery freeze changed')
    c.require(value['identity'] == identity(),'Recovery source changed')
    checks = c.verified(value['checks'])
    c.require(checks['exit_code'] == 0 and checks['identity'] == identity(),'Recovery checks changed')
    for item in [*checks['logs'],value['inventory'],value['resource_gate']]:
        c.verify_hash(item)
    if full_inventory:
        inventory.validate(c.verified(value['inventory']))
    return freeze['binding']


def event(stage,**fields):
    global REPORTER
    if REPORTER is None:
        REPORTER = StatusReporter(OUTPUT,{'original_binding':c.ORIGINAL_BINDING})
    return REPORTER.event(stage,**fields)


def worker_initialize(kind,binding,base):
    base = Path(base)
    c.require(kind in ('games','audits'),'Unknown recovery worker role')
    folder = base/'workers'/kind
    folder.mkdir(parents=True,exist_ok=True)
    health = c.windows.assert_windowless_worker(folder/f'{os.getpid()}_stdout.txt',folder/f'{os.getpid()}_stderr.txt')
    c.require(c.recovery_binding(base) == binding,'Worker recovery binding changed')
    arena.initialize_models(base=base)
    c.require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(),'Worker arithmetic changed')
    old.save_new(folder/f'{os.getpid()}_health.json',{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,'kind':kind,**health})


def collect_job(spec,binding,base):
    c.budget_boundary(base)
    arena.play_game('main',spec,c.ORIGINAL_BINDING,base,recovery_binding=binding)
    path = c.resolve_game('main',spec['index'],base)
    c.require(path is not None,'Missing completed game')
    return c.artifact(path,index=spec['index'])


def audit_job(item,binding,base):
    c.budget_boundary(base)
    value = verify.audit_game('main',item['index'],item['sha256'],c.ORIGINAL_BINDING,base)
    path = Path(base)/'main/audits'/f"game_{item['index']:04d}.json"
    old.save_new(path,{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,'game_artifact':item,
                      'passed':True,'full_inner_replay':False,**value})
    return c.artifact(path,index=item['index'])


def signal_abort(base,exc):
    path = Path(base)/'abort.json'
    if not path.exists():
        old.save_new(path,{'time':old.now(),'type':type(exc).__name__,'reason':str(exc)})


def run_pool(kind,arguments,binding):
    c.budget_boundary(OUTPUT)
    c.require(validate_freeze() == binding,'Frozen recovery changed')
    launch_path = OUTPUT/f'main_{kind}_launch.json'
    old.save_new(launch_path,{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,'time':old.now(),
                            'kind':kind,'indices':[a['index'] for a in arguments]})
    start = time.perf_counter()
    event('main_'+kind,completed=0,total=len(arguments),retained_games=25 if kind == 'games' else 0)
    runtime = c.windows.configure_spawn()
    fn = collect_job if kind == 'games' else audit_job
    iterator,files = iter(arguments),[]
    with ProcessPoolExecutor(max_workers=c.CONFIG['workers'],mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_initialize,initargs=(kind,binding,OUTPUT)) as pool:
        pending = {}
        def submit_next():
            arg = next(iterator,None)
            if arg is not None:
                pending[pool.submit(fn,arg,binding,OUTPUT)] = arg['index']
        try:
            for _ in range(min(c.CONFIG['workers'],len(arguments))):
                submit_next()
            while pending:
                done,_ = wait(pending,return_when=FIRST_COMPLETED)
                completed = []
                for future in done:
                    pending.pop(future)
                    completed.append(future.result())
                files.extend(completed)
                c.budget_boundary(OUTPUT)
                event('main_'+kind,completed=len(files),total=len(arguments),retained_games=25 if kind == 'games' else 0)
                for _ in completed:
                    submit_next()
        except BaseException as exc:
            signal_abort(OUTPUT,exc)
            for future in pending:
                future.cancel()
            raise
    c.require(len(files) == len(arguments),'Incomplete recovery stage')
    return sorted(files,key=lambda r:r['index']),{'elapsed_seconds':time.perf_counter()-start,'runtime':runtime,
                                                'launch':c.artifact(launch_path)}


def checked_coverage(receipts,sample,sample_sha):
    baseline = copy.deepcopy(receipts)
    terminal=unknown=certified=extras=extra_worlds=0
    for original_receipt,adjusted_receipt in zip(receipts,baseline,strict=True):
        c.require(original_receipt['passed'] is True,'Failed audit receipt')
        for root,adjusted in zip(original_receipt['roots'],adjusted_receipt['roots'],strict=True):
            u,t = root['unresolved_continuations'],root['terminal_continuations']
            c.require(type(u) is int and type(t) is int and 0 <= u <= root['continuations']
                      and t+u == root['continuations'],'Bad terminal/unresolved accounting')
            c.require(root['certified_roots'] == int(u>0) and root['censored_replay_cells'] == u,'Missing cutoff certificate/replay')
            cells = root['extra_censored_cells']
            c.require(len(cells) == root['extra_censored_replay_cells']
                      and len({tuple(cell) for cell in cells}) == len(cells),'Bad extra cutoff replay accounting')
            for wi,j in cells:
                c.require(type(wi) is int and type(j) is int and 0 <= j < root['support']
                          and 0 <= wi < root['world_counts'][j] and wi not in root['replay_world_indices'],
                          'Invalid extra replay cell')
            more_worlds = sorted({wi for wi,j in cells})
            c.require(root['extra_censored_replay_world_indices'] == more_worlds
                      and root['replayed_worlds'] == len(root['replay_world_indices'])+len(more_worlds),
                      'Bad replay world union')
            adjusted['replayed_branches'] -= len(cells)
            terminal += t;unknown += u;certified += int(u>0);extras += len(cells);extra_worlds += len(more_worlds)
    coverage = original.checked_audit_coverage(baseline,sample,sample_sha)
    c.require(terminal+unknown == coverage['recorded_continuations'],'Global continuation mismatch')
    return {**coverage,'prescribed_replayed_continuations':coverage['replayed_continuations'],
            'prescribed_replayed_worlds':coverage['replayed_worlds'],
            'replayed_continuations':coverage['replayed_continuations']+extras,
            'replayed_worlds':coverage['replayed_worlds']+extra_worlds,
            'terminal_continuations':terminal,'unresolved_continuations':unknown,
            'certified_roots':certified,'extra_censored_replay_cells':extras,
            'extra_censored_replay_worlds':extra_worlds}


def finish(binding):
    c.budget_boundary(OUTPUT)
    c.require(validate_freeze(full_inventory=True) == binding,'Final recovery identity changed')
    models.validate(c.ORIGINAL_BINDING,SOURCE)
    sample = verify.validate_sample('main',c.ORIGINAL_BINDING,OUTPUT)
    games = old.read_json(OUTPUT/'main_games_lock.json')
    audits = old.read_json(OUTPUT/'main_audits_lock.json')
    c.require(games['recovery_binding'] == audits['recovery_binding'] == binding,'Stage binding changed')
    specs = arena.specs('main')
    c.require([x['index'] for x in games['files']] == [x['index'] for x in audits['files']] == list(range(402)),
              'Incomplete final schedule')
    receipts,rows = [],[]
    for game_item,audit_item,spec in zip(games['files'],audits['files'],specs,strict=True):
        c.budget_boundary(OUTPUT)
        game,receipt = c.verified(game_item),c.verified(audit_item)
        c.require(game['spec'] == spec and 'failure' not in game and game['reason'] in old.VALID_NATURAL_TERMINATIONS,
                  'Invalid final game')
        c.require(receipt['manifest'] == c.ORIGINAL_BINDING and receipt['recovery_binding'] == binding
                  and receipt['game_artifact'] == game_item and receipt['passed'],'Final audit linkage')
        raw = {'index':spec['index'],'pair':spec['pair'],'fit':spec['fit'],'seat':spec['candidate_seat'],
               'score':.5 if game['winner'] is None else int(game['winner'] == spec['candidate_seat'])}
        c.require(raw == {k:receipt['game'][k] for k in raw},'Raw game/audit outcome disagreement')
        c.require(len(game['trace']) == len(receipt['roots']),'Final root inventory mismatch')
        for move,root in zip(game['trace'],receipt['roots'],strict=True):
            path = c.resolve_root('main',spec['index'],move['turn'],OUTPUT)
            c.require(path is not None and old.file_hash(path) == move['root_sha256'] == root['sha256'], 'Root mutated')
        rows.append(raw);receipts.append(receipt)
    coverage = checked_coverage(receipts,sample,old.file_hash(OUTPUT/'main/audit_sample.json'))
    summary = arena.statistics(rows,'main')
    independent = verify.independent_statistics([r['game'] for r in receipts],'main')
    verify.compare(summary,independent)
    roots = [r for receipt in receipts for r in receipt['roots']]
    result = {'manifest':binding,'original_binding':c.ORIGINAL_BINDING,'time':old.now(),**summary,
              'rows':[r['game'] for r in receipts],'coverage':coverage,'all_prescribed_audits_pass':True,
              'full_inner_replay':False,'amendment':'repetition_cutoff_interval_certificate_v1',
              'original_completed_games_reused':25,'elapsed_hours_from_original_launch':(time.time()-c.ORIGINAL_LAUNCH_EPOCH)/3600,
              'continuations_by_role':{role:sum(r['continuations'] for r in roots if r['planner']==role)
                                       for role in ('candidate','reference')},
              'unresolved_by_role':{role:sum(r['unresolved_continuations'] for r in roots if r['planner']==role)
                                    for role in ('candidate','reference')},
              'decision_seconds':{role:{'count':sum(r['planner']==role for r in roots),
                  'total':sum(r['timing']['wall']+r.get('inherited_timing',{}).get('wall',0)
                              for r in roots if r['planner']==role),
                  'inherited_failed_root_work':sum(r.get('inherited_timing',{}).get('wall',0)
                                                   for r in roots if r['planner']==role)}
                  for role in ('candidate','reference')},
              'scope':'Same fixed full-game comparison under disclosed conservative cutoff certification; no imputed scores, new worlds or continuation-policy change.',
              'automatic_successor':False}
    old.save_new(OUTPUT/'main_complete.json',{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,'time':old.now(),
                 'games':402,'all_prescribed_audits_pass':True,'coverage':coverage,
                 'game_lock':c.artifact(OUTPUT/'main_games_lock.json'),'audit_lock':c.artifact(OUTPUT/'main_audits_lock.json')})
    old.save_new(OUTPUT/'result.json',result)
    old.save_new(OUTPUT/'final_audit.json',{'manifest':binding,'original_binding':c.ORIGINAL_BINDING,'passed':True,
                 'result':c.artifact(OUTPUT/'result.json'),'independent_statistics':independent,'raw_game_endpoints_agree':True,
                 'coverage':coverage,'original_probe':c.artifact(SOURCE/'probe_complete.json'),
                 'main':c.artifact(OUTPUT/'main_complete.json'),'inventory':c.artifact(OUTPUT/'inventory.json')})
    lines = ['# Equal budget breadth arena with certified recovery','',f"Classification: {summary['classification']}.",'',
             f"{summary['wins']} wins, {summary['draws']} draws, {summary['losses']} losses across402 games/201 seat pairs.",
             f"Score {summary['share']:.4%}; paired95% interval [{summary['ci95'][0]:.4%}, {summary['ci95'][1]:.4%}].",'',
             'This is the explicitly amended recovery result. Original failed artifacts remain intact. Models, seeds, candidate budgets and natural outer-game scoring are unchanged.',
             f"Recorded continuations: {coverage['recorded_continuations']}; terminal {coverage['terminal_continuations']}; unresolved repetition cutoffs {coverage['unresolved_continuations']}. Exact certificates made {coverage['certified_roots']} decisions independent of every possible missing WDL value. No missing result was counted as a draw.",
             f"All actual games/decisions were checked; {coverage['replayed_continuations']} branches replayed, including every unresolved cell and the unchanged ordinary/stress samples. Whole inner replay was not performed.",'',
             'The original55% observed-score and95% lower>50% gate is retained. Equal per-state continuation quotas do not establish equal wall time or total game work. No broader-opponent or superhuman claim. No automatic successor.','']
    with (OUTPUT/'report.md').open('x',encoding='utf-8',newline='\n') as handle:
        handle.write('\n'.join(lines))
    c.budget_boundary(OUTPUT)
    old.save_new(OUTPUT/'final_status.json',{'manifest':binding,'original_binding':c.ORIGINAL_BINDING,'time':old.now(),
                 'complete':True,'all_prescribed_audits_pass':True,'classification':summary['classification'],
                 'result':c.artifact(OUTPUT/'result.json'),'audit':c.artifact(OUTPUT/'final_audit.json'),
                 'report':c.artifact(OUTPUT/'report.md'),'automatic_successor':False})
    event('complete',score=summary['share'],classification=summary['classification'])


def certify_failed_root(binding,saved):
    failed = c.verified(saved['failed_root'])
    state = old.deserialize_state_snapshot(failed['snapshot'])
    spec = arena.specs('main')[13]
    path = arena.root_path('main',13,17,OUTPUT)
    event('certifying_failed_root',index=13,turn=17)
    arena.choose_root(state,spec['search_stream'],path,
                      c.ORIGINAL_BINDING,failed['proposal']['model_id'],['main',13,17],True,OUTPUT,
                      recovery_binding=binding)
    root = old.read_json(path)
    verify.certified_selection(root,root['counts'])
    event('failed_root_certified',index=13,turn=17,choice=root['choice'],
          unresolved=sum(root['certificate']['unresolved_counts']))


def run():
    global REPORTER
    runtime = OUTPUT/'runtime'
    runtime.mkdir(parents=True,exist_ok=True)
    c.windows.assert_windowless_worker(runtime/'stdout.txt',runtime/'stderr.txt')
    with old.process_lock(SOURCE/'run.lock'), old.process_lock(OUTPUT/'run.lock'):
        c.require(not (OUTPUT/'launch.json').exists(),'One-shot recovery; preserve attempts')
        binding = validate_freeze(full_inventory=True)
        REPORTER = StatusReporter(OUTPUT,{'manifest':binding,'original_binding':c.ORIGINAL_BINDING})
        try:
            with original.parent.inhibit_system_sleep() as awake:
                c.budget_boundary(OUTPUT)
                gate = resource_decision(old.read_json(SOURCE/'operational_decision.json'),
                                         time.time()-c.ORIGINAL_LAUNCH_EPOCH,shutil.disk_usage(ROOT).free)
                c.require(gate['reasonable'],'Original remaining resource budget no longer fits')
                old.save_new(OUTPUT/'launch.json',{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,
                    'time':old.now(),'pid':os.getpid(),'recovery_epoch_seconds':time.time(),
                    'original_launch_epoch':c.ORIGINAL_LAUNCH_EPOCH,'sleep_inhibition':awake,
                    'windowless':c.windows.assert_windowless_worker(),'resource_gate':gate})
                old.save_new(OUTPUT/'main_launch.json',{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,
                             'time':old.now(),'schedule':arena.specs('main')})
                models.validate(c.ORIGINAL_BINDING,SOURCE)
                arena.initialize_models(base=OUTPUT)
                saved = c.load_inventory(OUTPUT)
                certify_failed_root(binding,saved)
                pending = [arena.specs('main')[i] for i in saved['pending_main_indices']]
                generated,game_metadata = run_pool('games',pending,binding)
                games = sorted([*saved['retained_main_games'],*generated],key=lambda x:x['index'])
                c.require([x['index'] for x in games] == list(range(402)),'Merged game inventory incomplete')
                old.save_new(OUTPUT/'main_games_lock.json',{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,
                             'files':games,**game_metadata})
                verify.seal_sample('main',games,c.ORIGINAL_BINDING,OUTPUT)
                verify.validate_sample('main',c.ORIGINAL_BINDING,OUTPUT)
                audits,audit_metadata = run_pool('audits',games,binding)
                old.save_new(OUTPUT/'main_audits_lock.json',{'manifest':c.ORIGINAL_BINDING,'recovery_binding':binding,
                             'files':audits,**audit_metadata})
                finish(binding)
        except (arena.AmbiguousDecision,c.BudgetExceeded) as exc:
            signal_abort(OUTPUT,exc)
            old.save_new(OUTPUT/'operational_hold.json',{'manifest':binding,'original_binding':c.ORIGINAL_BINDING,
                         'time':old.now(),'reason':str(exc),'type':type(exc).__name__})
            event('operational_hold',reason=str(exc),type=type(exc).__name__)
        except BaseException as exc:
            signal_abort(OUTPUT,exc)
            old.save_new(OUTPUT/'failure.json',{'manifest':binding,'original_binding':c.ORIGINAL_BINDING,
                         'time':old.now(),'traceback':traceback.format_exc()})
            event('failed',failure='failure.json')
            raise


def launch():
    binding = validate_freeze(full_inventory=True)
    folder = OUTPUT/'runtime'
    folder.mkdir(parents=True,exist_ok=True)
    with old.process_lock(folder/'launcher.lock'):
        c.require(not (folder/'process.json').exists() and not (OUTPUT/'launch.json').exists(),'Already launched')
        process = c.windows.detached_launch([sys.executable,'-u',ENTRY,'run'],folder/'stdout.txt',folder/'stderr.txt',ROOT)
        old.save_new(folder/'process.json',{**process,'recovery_binding':binding,'time':old.now()})
        deadline = time.monotonic()+60
        while not (OUTPUT/'launch.json').exists():
            c.require(not (OUTPUT/'failure.json').exists(),'Recovery failed before acknowledgement; inspect preserved logs')
            c.require(time.monotonic() < deadline,'No acknowledgement; do not automatically retry')
            time.sleep(.05)
        receipt = old.read_json(OUTPUT/'launch.json')
        c.require(receipt['pid'] == process['pid'] and receipt['recovery_binding'] == binding,'Recovery acknowledgement mismatch')
        old.save_new(folder/'acknowledged.json',{'time':old.now(),'pid':process['pid'],'launch':c.artifact(OUTPUT/'launch.json')})
    print({'launched':True,'pid':process['pid'],'binding':binding},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','prepare','launch','run'))
    parser.add_argument('--checks',type=Path)
    args = parser.parse_args()
    if args.command == 'check':
        c.require(args.checks is not None,'New check folder required')
        return check(args.checks)
    if args.command == 'prepare':
        c.require(args.checks is not None,'Passing checks receipt required')
        prepare(args.checks)
    elif args.command == 'launch':
        launch()
    else:
        run()
    return 0
