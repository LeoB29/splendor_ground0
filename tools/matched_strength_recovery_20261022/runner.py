"""One explicitly amended retry; original study remains incomplete."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import traceback
import math
import psutil

from tools.matched_strength_recovery_20261022 import common as c
from tools.external_strength_screen_20261018 import rules, audit
from tools.external_strength_screen_20261018_02.runner import hidden_variant
from tools.splendorust_120_qualification_20261018.common import ProcessTreeWorker, load_profile
from tools.splendorust_120_qualification_20261018.runner import check_result as check_opponent
from tools.corrected_planner_qualification_20261018.clock import PlannerWorker
from tools.local_player_qualification_128.clock import processor_topology, WorkerInitializationError
from tools.matched_strength_20261020.runner import response_has_recognized_illegal_action

PLAYER_FAILURES = {'timeout','worker_exit','malformed','wrong_request','illegal','unsolicited','write_error'}
TESTS = ['tests/test_matched_strength_recovery_20261022.py',
         'tests/test_matched_strength_recovery_analyzer_20261022.py',
         'tests/test_matched_strength_recovery_completion_20261022.py',
         'tests/test_matched_strength_recovery_launch_20261022.py',
         'tests/test_matched_strength_runner_20261020.py',
         'tests/test_corrected_planner_adapter_20261018.py',
         'tests/test_future_training_rules_20261021.py']
PROTOCOL = 'docs/engineering/matched_strength_recovery_20261022.md'
REVIEW = 'docs/engineering/matched_strength_recovery_20261022_review.md'
ANALYZER = 'tools/analyze_matched_strength_recovery_20261022.py'


def save_runtime_sources(path):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.save_new(path, {'schema': 'corrected_runtime_sources_v1', 'ruleset_id': c.RULESET_ID,
                     'catalogue_sha256': c.CATALOGUE_SHA256, 'sources': verify_runtime_sources()})


def check_source_closed():
    launch = c.read(c.SOURCE/'launch.json'); outer = c.read(c.SOURCE/'runner_process.json')
    init = c.read(c.SOURCE/'workers'/c.RETRY_IDENTITY/'initialization.json')
    identities = [launch['identity'], {'pid': outer['supervisor_pid'], 'created': outer['supervisor_created']}]
    identities.extend({'pid': x['pid'], 'created': x['created']} for x in init['telemetry']['process_tree'])
    for value in identities:
        try:
            process = psutil.Process(value['pid'])
            c.require(process.create_time() != value['created'] or not process.is_running(),
                      'Authenticated original process is still running')
        except psutil.NoSuchProcess:
            pass


def check_interrupted_request():
    folder = c.SOURCE/'workers'/c.RETRY_IDENTITY
    c.require(not (folder/'request-00000001.json').exists() and
              not (c.SOURCE/'decision_attempts'/(c.RETRY_IDENTITY+'.json')).exists() and
              not (c.SOURCE/'sealed/games/059/moves/0043.json').exists(), 'Retry would replace accepted/attributed evidence')
    c.require((folder/'events.jsonl').stat().st_size == 0 and
              (folder/'continuations.jsonl.gz').is_file(), 'Interrupted request evidence changed')
    import json
    packets = [json.loads(line) for line in (folder/'stdout.log').read_bytes().splitlines()]
    c.require(len(packets) == 1 and packets[0]['type'] == 'ready', 'Original request now has a response')
    return [c.artifact(p) for p in sorted(folder.iterdir()) if p.is_file()]


def prior_memory_state():
    items, total = [], 0.
    for path in sorted((c.SOURCE/'memory_admissions').rglob('*.json')):
        value = c.read(path); charge = value['charged_seconds']
        c.require(value['status'] == 'passed' and type(charge) in (int, float) and
                  math.isfinite(charge) and charge >= 0, 'Prior RAM admission incomplete')
        items.append(c.artifact(path)); total += charge
    c.require(items and total < c.config()['cumulative_memory_wait_seconds'], 'Original RAM allowance exhausted')
    return {'schema': 'matched_strength_prior_memory_v1', 'admission_count': len(items),
            'charged_seconds': total, 'inputs': items}


def prepare():
    base = c.OUTPUT; base.mkdir(parents=True, exist_ok=False)
    try:
        c.require('torch' not in sys.modules, 'Preparation must remain model-free')
        c.require(c.file_hash(c.SOURCE/'freeze.json') == c.SOURCE_FREEZE_SHA256, 'Original freeze changed')
        frozen = c.read(c.SOURCE/'freeze.json'); topology = processor_topology()
        c.require(topology == frozen['topology'], 'Original full supervisor affinity required')
        check_source_closed()
        review = c.read(c.SOURCE_REVIEW/'review.json')
        c.require(review['status'] == 'inspection_complete' and review['original_files_modified'] == 0 and
                  review['matchup_outcomes_opened'] is False and review['interruption']['id'] == c.RETRY_IDENTITY and
                  review['interruption']['classification'] == 'unresolved_active_request_infrastructure_interruption',
                  'Bounded source inspection required')
        inventory = c.read(c.SOURCE_REVIEW/'source_inventory.json')
        c.verify(frozen['inputs']); c.verify_source_inventory(inventory)
        c.require(shutil.disk_usage(c.ROOT).free >= c.config()['evidence_cap_bytes']-inventory['total_bytes']+c.MIN_FREE,
                  'Remaining combined evidence allowance and disk reserve')
        from tools.matched_strength_recovery_20261022.completion import contract
        c.require(c.read(c.PREPARATION/'completion_contract.json') == contract(), 'Completion contract differs')
        schedule = c.read(c.SOURCE/'sealed/schedule.json')
        c.require(tuple(x['game'] for x in schedule) == c.game_order() and len(schedule) == 60 and
                  tuple(x['game'] for x in schedule[:-1]) == c.COMPLETED_GAMES and schedule[-1]['game'] == 59,
                  'Exact original schedule required')
        specs = {x['game']: x for x in schedule}
        c.require({p.name for p in (c.SOURCE/'sealed/games').iterdir() if p.is_dir()} == {f'{i:03d}' for i in range(60)},
                  'Unexpected original game directories')
        validation = c.read(c.SOURCE/'adapter_validation.json')
        c.require(validation['status'] == 'passed' and validation['requests'] == 12 and validation['all_full_replays'] is True,
                  'Twelve original validation replays required')
        c.require(not (c.SOURCE/'audit_manifest.json').exists(), 'Unexpected completed original game audit manifest')
        sources = {c.SOURCE/name for name in ('config.json', 'local_profile.json', 'validation_cases.json',
                   'seed_collision_check.json', 'adapter_validation.json', 'sealed/schedule.json',
                   'sealed/referee_seed_material.json')}
        for directory in ('validation/rows', 'decision_attempts/validation', 'audit_rows/validation'):
            sources.update((c.SOURCE/directory).glob('*.json'))
        failures = []; partial = None
        for game in c.game_order():
            complete = game != c.PARTIAL_GAME
            recovered = reconstruct_game(specs[game], c.SOURCE, complete=complete,
                                         next_ply=None if complete else c.NEXT_PLY)
            folder = c.SOURCE/'sealed/games'/f'{game:03d}'; sources.add(folder/'initial.json')
            if complete: sources.add(folder/'result.json')
            else: partial = recovered
            if recovered['failure'] is not None: failures.append((game, recovered['failure'][0], recovered['failure'][2]))
            for path, row in zip(recovered['move_paths'], recovered['rows'], strict=True):
                sources.add(path); sources.add(c.SOURCE/'decision_attempts'/(row['id']+'.json'))
        c.require(failures == [(39, 'forfeit_timeout', 'local')], 'Original genuine timeout attribution changed')
        preserved = check_interrupted_request()
        c.require(partial['state'].current_player == specs[59]['local_seat'], 'Explicit retry is not the local player')
        imports = [copy_new(path, base/path.relative_to(c.SOURCE)) for path in sorted(sources)]
        c.save_new(base/'prior_memory_state.json', prior_memory_state())
        plan = {'schema': 'matched_strength_amended_recovery_plan_v1',
            'source_base': c.SOURCE.relative_to(c.ROOT).as_posix(), 'source_freeze': c.artifact(c.SOURCE/'freeze.json'),
            'source_launch': c.artifact(c.SOURCE/'launch.json'),
            'source_inventory': c.artifact(c.SOURCE_REVIEW/'source_inventory.json'),
            'source_review': c.artifact(c.SOURCE_REVIEW/'review.json'), 'imports': imports,
            'completed_games': list(c.COMPLETED_GAMES),
            'partial_game': {'game': 59, 'next_ply': 43, 'state_sha256': partial['state_sha256']},
            'preserved_interrupted_request': preserved, 'preserved_timeout': {'game': 39, 'ply': 7, 'status': 'forfeit_timeout'},
            'amendment': c.AMENDMENT, 'validation_reused': True, 'new_validation_requests': 0,
            'original_primary_complete': False, 'outcomes_closed': True}
        c.save_new(base/'recovery_plan.json', plan)
        c.require(all((c.ROOT/name).is_file() for name in (*TESTS, PROTOCOL, REVIEW, ANALYZER)), 'Recovery prerequisites missing')
        code = 'import psutil,pytest; psutil.Process().cpu_affinity([2]); raise SystemExit(pytest.main('+repr(['-q',*TESTS,'-p','no:cacheprovider'])+'))'
        test = subprocess.run([sys.executable,'-B','-c',code], cwd=c.ROOT,
            env={**os.environ,'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'PYTHONDONTWRITEBYTECODE':'1'},
            capture_output=True,text=True,timeout=300,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        c.save_new(base/'preflight_tests.json', {'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr,
                                               'cpu':2,'models_or_games_executed':False})
        c.require(test.returncode == 0, 'Preflight failed; preserve preparation')
        paths = {x['path'] for x in frozen['inputs']} | {x['path'] for x in inventory['files']}
        for directory in (c.SOURCE_REVIEW, c.PREPARATION, c.ROOT/'tools/matched_strength_recovery_20261022', base):
            paths.update(p.relative_to(c.ROOT).as_posix() for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        paths.update((*TESTS,PROTOCOL,REVIEW,ANALYZER,'tools/future_training_rules_20261021.py',
                      'docs/engineering/future_training_rules_20261021.md',
                      'docs/engineering/matched_strength_interruption_recovery_options_20261022.md'))
        c.verify(frozen['inputs']); c.verify_source_inventory(inventory)
        c.require(processor_topology() == topology, 'Preparation topology changed')
        c.save_new(base/'freeze.json', {**{k:v for k,v in frozen.items() if k not in ('time','inputs')},
            'schema':'matched_strength_amended_recovery_freeze_v1','time':c.now(),
            'inputs':[c.artifact(c.ROOT/p) for p in sorted(paths)],'source_freeze':plan['source_freeze'],
            'source_inventory':plan['source_inventory'],'recovery_plan':c.artifact(base/'recovery_plan.json'),
            'recovery_max_seconds':c.MAX_SECONDS,'recovery_post_game_reserve_seconds':c.POST_GAME_RESERVE,
            'original_post_game_reserve_seconds':c.ORIGINAL_POST_GAME_RESERVE,
            'imported_completed_games':59,'new_validation_requests':0,'validation_reused':True,
            'amendment':c.AMENDMENT,'original_primary_complete':False,'topology':topology,'executable':sys.executable})
        print(c.canonical({'prepared':True,'imported_completed_games':59,'retry_identity':c.RETRY_IDENTITY,
                          'execution_started':False,'original_primary_complete':False}),flush=True)
    except BaseException:
        c.save_new(base/'preparation_failure.json', {'time':c.now(),'failure':traceback.format_exc(),'outcomes_closed':True})
        raise

def copy_new(source, destination):
    source,destination=Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src,dst,1024**2)
        dst.flush();os.fsync(dst.fileno())
    before,after=c.artifact(source),c.artifact(destination)
    c.require((before['sha256'],before['bytes'])==(after['sha256'],after['bytes']),'Imported bytes differ')
    return {'source':before,'destination':after}

def bound_read(descriptor, expected_path=None):
    path=(c.ROOT/descriptor['path']).resolve()
    c.require(path.is_relative_to(c.ROOT.resolve()) and c.artifact(path)==descriptor,'Changed referenced evidence')
    if expected_path is not None:c.require(path==Path(expected_path).resolve(),'Unexpected source evidence path')
    return c.read(path)

def receipt_origin(game, ply, source_base):
    return Path(source_base)


def read_saved_move(descriptor, path, source_base):
    return bound_read(descriptor, path)


def reconstruct_game(spec, source_base, *, complete, next_ply=None):
    """Model-free state recovery: canonical initial seed plus every accepted action."""
    source_base=Path(source_base)
    folder=source_base/'sealed/games'/f"{spec['game']:03d}"
    env=rules.CorrectedEnv(seed=int(spec['seed']));state=env.initial_state()
    initial=rules.serialize_snapshot(state);audit.validate_ledger(initial)
    c.require(c.read(folder/'initial.json')=={**spec,'initial':initial,'initial_sha256':c.digest(initial)},
              'Recovered initial world differs from exact frozen seed/schedule')
    paths=sorted((folder/'moves').glob('*.json'))
    c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(len(paths))],'Noncontiguous saved prefix')
    c.require(0<len(paths)<=c.MAX_PLIES,'Empty or excessive saved prefix')
    if next_ply is not None:c.require(len(paths)==next_ply,'Unexpected saved-prefix length')
    game=c.read(folder/'result.json') if complete else None
    if complete:
        c.require(all(game.get(k)==v for k,v in spec.items()) and game['initial_sha256']==c.digest(initial) and
                  len(game['moves'])==len(paths),'Completed source game identity/count')
    else:c.require(not (folder/'result.json').exists(),'Partial game already completed')
    failure=None; rows=[]
    profile=None
    for ply,path in enumerate(paths):
        c.require(not state.terminal and failure is None,'Saved moves continued after game ending')
        row=read_saved_move(game['moves'][ply],path,source_base) if complete else c.read(path)
        before=rules.serialize_snapshot(state);actor=state.current_player
        player='local' if actor==spec['local_seat'] else 'opponent'
        name=f"games/g{spec['game']:03d}/t{ply:04d}"
        c.require([row[k] for k in ('game','pair','fit','ply','actor','before_sha256','id','player')]==
                  [spec['game'],spec['pair'],spec['fit'],ply,actor,c.digest(before),name,player],
                  'Saved move identity/state chain')
        c.require(c.read(source_base/'decision_attempts'/(name+'.json'))==
                  {k:v for k,v in row.items() if k not in ('game','pair','fit','ply','actor','before_sha256')},
                  'Saved move differs from original decision attempt')
        observer=(c.make_observer_payload(before,c.operational_context(spec['game'],ply))
                  if player=='local' else rules.observation_payload(state))
        c.require(row['observer']==observer,'Recovered observer differs from actual lawful input')
        audit.check_observer(before,c.opponent_observer(observer) if player=='local' else observer)
        origin=receipt_origin(spec['game'],ply,source_base)
        receipt=bound_read(row['receipt'],origin/'workers'/name/'request-00000001.json')
        c.require(without_result(receipt)==row['clock'],'Saved request clock copy differs')
        if row['status']=='completed':
            c.require(receipt['status']=='ok' and receipt['accepted'] is True and row['correct'] is True,
                      'Completed source action lacks accepted receipt')
            result=receipt['result'];c.require(c.digest(result)==row['result_sha256'],'Saved result binding')
            legal=[c.serialize_action(a) for a in rules.legal_actions(state)]
            if player=='local':
                trace=(c.ROOT/row['trace']['path']).resolve()
                c.require(trace==(origin/'workers'/name/'continuations.jsonl.gz').resolve() and
                          c.artifact(trace)==row['trace'],'Saved continuation trace changed or wrong origin')
                c.check_wrapper(result,observer,legal,trace);action=result['decision']['action']
            else:
                if profile is None:profile=load_profile(c.EXTERNAL_PROFILE_PATH,c.EXTERNAL_PROFILE_SHA256)
                expected=check_opponent(result,observer,profile,c.EXTERNAL_PROFILE_SHA256,before)
                c.require(expected is not None,'Completed opponent receipt exhausts budget')
                action=result['action']
            c.require(action in legal,'Saved action no longer legal')
            state=env.step(state,c.deserialize_action(action));after=rules.serialize_snapshot(state)
            audit.check_step(before,action,after)
            c.require(c.digest(after)==row['after_sha256'],'Recovered successor differs')
        elif row['status'].startswith('forfeit_'):
            category=row['status'][len('forfeit_'):]
            if category=='internal_budget_exhausted':
                c.require(player=='opponent' and receipt['status']=='ok','Invalid internal-budget forfeit')
                if profile is None:profile=load_profile(c.EXTERNAL_PROFILE_PATH,c.EXTERNAL_PROFILE_SHA256)
                c.require(check_opponent(receipt['result'],observer,profile,c.EXTERNAL_PROFILE_SHA256,before) is None,
                          'Internal-budget forfeit payload differs')
            else:c.require(category in PLAYER_FAILURES and receipt['status']==category and receipt['accepted'] is False,
                           'Invalid preserved player forfeit')
            failure=(row['status'],actor,player)
        else:raise ValueError('Unresolved source attempt cannot be imported as a completed move')
        rows.append(row)
    snapshot=rules.serialize_snapshot(state)
    if complete:
        c.require(game['complete'] is True and game['final']==snapshot,'Completed source final state differs')
        if failure:
            c.require(game['reason']==failure[0] and game['winner']==1-failure[1] and game['failing_player']==failure[2],
                      'Preserved forfeit ending differs')
        else:c.require(state.terminal and game['reason']=='natural' and game['winner']==state.winner,
                       'Preserved natural ending differs')
    else:c.require(failure is None and not state.terminal and state.turn_index==len(paths),'Partial prefix is not a live continuation')
    return {'env':env,'state':state,'snapshot':snapshot,'state_sha256':c.digest(snapshot),'rows':rows,'move_paths':paths,
            'game':game,'failure':failure}

def without_result(value):
    return {k: v for k, v in value.items() if k != 'result'}

def completed_result(row):
    receipt = c.read(c.ROOT/row['receipt']['path'])
    c.require(c.artifact(c.ROOT/row['receipt']['path']) == row['receipt'], 'Receipt artifact changed')
    return receipt['result']

def decision_attempt(name, player, snapshot, observer, deadline, cpu, *, reserve):
    from tools.matched_strength_recovery_20261022.memory import wait_for_memory
    admission = wait_for_memory(c.OUTPUT, deadline, name=name, reserve=reserve, request_seconds=c.MOVE_SECONDS)
    if deadline-time.monotonic() < c.INIT_SECONDS+c.MOVE_SECONDS+reserve:
        raise c.ResourceHold('Insufficient move reserve')
    folder = c.OUTPUT/'workers'/name
    trace = folder/'continuations.jsonl.gz'
    row = {'id': name, 'player': player, 'observer': observer, 'status': 'incomplete', 'correct': False,
           'memory_admission': c.artifact(c.OUTPUT/'memory_admissions'/(name+'.json'))}
    started = time.monotonic()
    try:
        audit.check_observer(snapshot, c.opponent_observer(observer) if player == 'local' else observer)
        objects = rules.legal_actions(rules.deserialize_snapshot(snapshot))
        legal = [c.serialize_action(a) for a in objects]
        worker_class = PlannerWorker if player == 'local' else ProcessTreeWorker
        module = ('tools.matched_strength_recovery_20261022.worker' if player == 'local'
                  else 'tools.matched_strength_recovery_20261022.opponent_worker')
        payload = {'observer': observer} if player == 'local' else {'player': 'opponent', 'observer': observer}
        with worker_class(module, folder, **worker_options(deadline, cpu, reserve=reserve, mode=player, trace=trace, runtime=folder/'runtime_sources.json')) as child:
            row['initialization'] = without_result(child.initialization)
            receipt = child.request(payload, c.MOVE_SECONDS, validate=lambda value:
                not response_has_recognized_illegal_action(value, player, observer, legal, trace))
            row['clock'] = without_result(receipt)
        row['runtime_sources'] = c.artifact(folder/'runtime_sources.json')
        row['initialization_artifact'] = c.artifact(folder/'initialization.json')
        row['receipt'] = c.artifact(folder/'request-00000001.json')
        if receipt['status'] != 'ok':
            if receipt['status'] not in PLAYER_FAILURES:
                raise ValueError('Hard worker or resource failure: '+receipt['status'])
            if receipt['status'] == 'illegal':
                c.require(response_has_recognized_illegal_action(receipt.get('result'), player, observer, legal, trace),
                          'Unrecognized illegal-action receipt; scientific hold')
                row['result_sha256'] = c.digest(receipt['result'])
                if player == 'local': row['trace'] = c.artifact(trace)
            row['status'] = 'forfeit_'+receipt['status']
            return row
        result = receipt['result']; row['result_sha256'] = c.digest(result)
        if player == 'local':
            c.check_wrapper(result, observer, legal, trace)
            row['trace'] = c.artifact(trace)
            action = result['decision']['action']
            after = rules.serialize_snapshot(rules.CorrectedEnv().step(rules.deserialize_snapshot(snapshot), c.deserialize_action(action)))
            audit.check_step(snapshot, action, after)
        else:
            after = check_opponent(result, observer, load_profile(c.EXTERNAL_PROFILE_PATH, c.EXTERNAL_PROFILE_SHA256),
                                   c.EXTERNAL_PROFILE_SHA256, snapshot)
            if after is None:
                row['status'] = 'forfeit_internal_budget_exhausted'; return row
        row.update(status='completed', correct=True, after_sha256=c.digest(after))
        return row
    except WorkerInitializationError as error:
        row['initialization'] = without_result(error.receipt)
        row['status'] = 'initialization_failure'
        raise
    except BaseException as error:
        row['failure_category'] = type(error).__name__
        raise
    finally:
        row['total_seconds'] = time.monotonic()-started
        c.save_new(c.OUTPUT/'decision_attempts'/(name+'.json'), row)

def audit_attempt(name, row, mode, deadline, cpu):
    from tools.matched_strength_recovery_20261022.memory import wait_for_memory
    admission = wait_for_memory(c.OUTPUT, deadline, name='audits/'+name, reserve=c.ANALYSIS_SECONDS+c.FINAL_RESERVE, request_seconds=c.AUDIT_SECONDS)
    if deadline-time.monotonic() < c.INIT_SECONDS+c.AUDIT_SECONDS+c.ANALYSIS_SECONDS+c.FINAL_RESERVE:
        raise c.ResourceHold('Insufficient audit/finalization reserve')
    result = completed_result(row); observer = row['observer']
    folder = c.OUTPUT/'audits'/name
    record = {'id': name, 'mode': mode, 'observer_sha256': c.digest(observer),
              'result_sha256': c.digest(result), 'correct': False,
              'memory_admission': c.artifact(c.OUTPUT/'memory_admissions/audits'/(name+'.json'))}
    started = time.monotonic()
    try:
        with PlannerWorker('tools.matched_strength_recovery_20261022.worker', folder,
                **worker_options(deadline, cpu, reserve=c.ANALYSIS_SECONDS+c.FINAL_RESERVE, mode='audit', runtime=folder/'runtime_sources.json')) as child:
            record['initialization'] = without_result(child.initialization)
            receipt = child.request({'observer': observer, 'result': result,
                'traces_path': str((c.ROOT/row['trace']['path']).resolve()), 'mode': mode}, c.AUDIT_SECONDS)
            record['clock'] = without_result(receipt)
        record['runtime_sources'] = c.artifact(folder/'runtime_sources.json')
        record['initialization_artifact'] = c.artifact(folder/'initialization.json')
        record['receipt'] = c.artifact(folder/'request-00000001.json')
        c.require(receipt['status'] == 'ok', 'Independent continuation audit did not complete')
        c.check_audit_proof(receipt['result'], observer, result, mode)
        record.update(verification=receipt['result'], correct=True)
        return record
    finally:
        record['total_seconds'] = time.monotonic()-started
        c.save_new(c.OUTPUT/'audit_rows'/(name+'.json'), record)

def make_audit_manifest():
    entries = []
    for game in range(2*c.config()['pair_count']):
        folder = c.OUTPUT/'sealed'/'games'/f'{game:03d}'
        for path in sorted((folder/'moves').glob('*.json')):
            row = c.read(path)
            if row['player'] != 'local' or row['status'] != 'completed': continue
            result = completed_result(row); identity = row['observer']['context']['identity']
            support = len(result['decision']['record']['candidates']['actions'])
            entries.append({'id': f"g{game:03d}-t{row['ply']:04d}", 'game': game, 'ply': row['ply'],
                'fit': row['fit'], 'identity': identity, 'observer_sha256': c.digest(row['observer']),
                'move': c.artifact(path), 'receipt': row['receipt'], 'trace': row['trace'],
                'candidate_support': support, 'mode': 'sample', 'world_indices': c.sample_worlds(identity),
                'sample_world_indices': c.sample_worlds(identity),
                'sample_hashes': [{'world_index': wi, 'sha256': c.digest([c.AUDIT_DOMAIN, 'world', c.AUDIT_SEED, identity, wi])}
                                  for wi in c.sample_worlds(identity)],
                'stress_priority_sha256': c.stress_priority(identity),
                'stress_cells': c.stress_cells(row['observer'], support)})
    cells = {}
    for cell in c.expected_stress_cells():
        eligible = [e for e in entries if cell in e['stress_cells']]
        chosen = min(eligible, key=lambda e: (c.stress_priority(e['identity']), e['game'], e['ply'])) if eligible else None
        cells[cell] = {'eligible': [e['id'] for e in eligible], 'selected': chosen['id'] if chosen else None}
    full = sorted({cell['selected'] for cell in cells.values() if cell['selected'] is not None})
    for entry in entries:
        if entry['id'] in full: entry.update(mode='full', world_indices=list(range(128)))
    return {'schema': 'operational_continuation_manifest_v1', 'entries': entries,
            'cells': cells, 'full_stress_ids': full}

def worker_options(deadline, cpu, *, reserve, mode, trace=None, runtime=None):
    c.require('torch' not in sys.modules, 'Supervisor must remain model-free')
    admission_started = time.monotonic()
    admission = c.boundary(c.OUTPUT, deadline)
    request_seconds = c.AUDIT_SECONDS if mode == 'audit' else c.MOVE_SECONDS
    if c.work_budget(admission, reserve) < c.INIT_SECONDS+request_seconds:
        raise c.ResourceHold('Insufficient dual-clock worker admission reserve')
    effective_deadline = min(deadline-reserve, admission_started+c.work_budget(admission, reserve))
    def live_guard():
        receipt=c.boundary(c.OUTPUT,deadline,memory_headroom=False)
        if c.work_budget(receipt, reserve) <= 0:
            raise c.ResourceHold('Dual-clock protected reserve reached')
        return receipt
    environment = {'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'CUDA_VISIBLE_DEVICES': '',
        'LIVE_WORKER_MODE': mode, 'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'}
    if trace is not None: environment['LIVE_TRACE_PATH'] = str(trace.resolve())
    if runtime is not None: environment['LIVE_RUNTIME_SOURCE_PATH'] = str(runtime.resolve())
    if mode == 'opponent':
        environment.update(SPLENDOR_OPPONENT_PROFILE=str(c.EXTERNAL_PROFILE_PATH.resolve()),
                           SPLENDOR_OPPONENT_PROFILE_SHA256=c.EXTERNAL_PROFILE_SHA256)
    return {'affinity': [cpu], 'initialization_timeout': c.INIT_SECONDS,
        'absolute_deadline': effective_deadline, 'env': environment,
        'max_message_bytes': 8*1024**2,
        'max_request_bytes': (16*1024**2 if mode == 'audit' else 1024**2),
        'max_log_bytes': 2*1024**2,
        'max_rss_bytes': c.MAX_WORKER_RSS,
        'resource_guard': live_guard}


def play_final_game(spec, deadline, cpu):
    c.require(spec['game'] == c.PARTIAL_GAME, 'Only the last frozen game may continue')
    recovered = reconstruct_game(spec, c.SOURCE, complete=False, next_ply=c.NEXT_PLY)
    plan = c.read(c.OUTPUT/'recovery_plan.json')
    c.require(recovered['state_sha256'] == plan['partial_game']['state_sha256'], 'Recovered boundary state changed')
    env, state = recovered['env'], recovered['state']
    folder = c.OUTPUT/'sealed/games/059'
    c.require(not (folder/'result.json').exists() and
              [p.name for p in sorted((folder/'moves').glob('*.json'))] == [f'{i:04d}.json' for i in range(c.NEXT_PLY)],
              'Recovery game prefix changed or already continued')
    for path in recovered['move_paths']:
        c.require(c.file_hash(folder/'moves'/path.name) == c.file_hash(path), 'Imported prefix copy changed')
    game = {**spec, 'initial_sha256':c.read(folder/'initial.json')['initial_sha256'],
            'moves':[c.artifact(path) for path in recovered['move_paths']], 'complete':False}
    for ply in range(c.NEXT_PLY, c.MAX_PLIES):
        if state.terminal: break
        before = rules.serialize_snapshot(state); actor = state.current_player
        player = 'local' if actor == spec['local_seat'] else 'opponent'
        observer = (c.make_observer_payload(before,c.operational_context(spec['game'],state.turn_index))
                    if player == 'local' else rules.observation_payload(state))
        alternate = rules.serialize_snapshot(hidden_variant(state))
        alternate_observer = (c.make_observer_payload(alternate,observer['context']) if player == 'local'
                              else rules.observation_payload(rules.deserialize_snapshot(alternate)))
        c.require(observer == alternate_observer, 'Actual-game observation privacy mismatch')
        name = f'games/g059/t{ply:04d}'
        if ply == c.NEXT_PLY:
            c.require(name == c.RETRY_IDENTITY and player == 'local', 'Explicit amendment identity mismatch')
            c.save_new(c.OUTPUT/'amended_retry.json', {'time':c.now(),'amendment':c.AMENDMENT,
                'observer_sha256':c.digest(observer),'before_sha256':c.digest(before),
                'preserved_original_request':plan['preserved_interrupted_request'],
                'original_primary_complete':False,'outcomes_closed':True})
        row = {'game':spec['game'],'pair':spec['pair'],'fit':spec['fit'],'ply':ply,'actor':actor,
               'before_sha256':c.digest(before),
               **decision_attempt(name,player,before,observer,deadline,cpu,reserve=c.POST_GAME_RESERVE)}
        if row['status'] == 'completed':
            result = completed_result(row)
            action = result['decision']['action'] if player == 'local' else result['action']
            state = env.step(state,c.deserialize_action(action)); after = rules.serialize_snapshot(state)
            c.require(c.digest(after) == row['after_sha256'], 'Actual referee successor mismatch')
            audit.check_step(before,action,after)
        else:
            c.require(row['status'].startswith('forfeit_'), 'Unscored hard failure')
            game.update(complete=True,reason=row['status'],winner=1-actor,failing_player=player)
        path = folder/'moves'/f'{ply:04d}.json'; c.save_new(path,row); game['moves'].append(c.artifact(path))
        print(c.canonical({'stage':'amended_final_game','game':59,'ply':ply,'player':player,
            'decision_status':row['status'],'move_seconds':row['clock'].get('receipt_elapsed_seconds'),
            'outcomes_closed':True,'original_primary_complete':False}),flush=True)
        if game['complete']: break
    if not game['complete']:
        if not state.terminal:
            c.save_new(folder/'unresolved.json',{**game,'final':rules.serialize_snapshot(state),'reason':'ply_ceiling'})
            raise c.ResourceHold('Nonterminal maximum-ply ceiling; no invented result')
        game.update(complete=True,reason='natural',winner=state.winner)
    game['final'] = rules.serialize_snapshot(state); c.save_new(folder/'result.json',game)
    return game


def clock_guard(launch, original):
    mono, epoch = time.monotonic(), time.time()
    c.require(mono >= original['monotonic_start'] and mono >= launch['monotonic_start'], 'Monotonic continuity lost')
    if (mono >= launch['monotonic_start']+c.MAX_SECONDS or epoch >= launch['epoch_seconds']+c.MAX_SECONDS or
            mono >= original['monotonic_start']+original['maximum_seconds'] or
            epoch >= original['epoch_seconds']+original['maximum_seconds']):
        raise c.ResourceHold('Original/recovery clock ceiling during authentication')


def verify_execution_inputs(freeze, launch):
    original = c.read(c.SOURCE/'launch.json')
    for item in freeze['inputs']:
        clock_guard(launch,original)
        c.require(c.artifact(c.ROOT/item['path']) == item, 'Changed frozen input: '+item['path'])
    inventory = c.read(c.SOURCE_REVIEW/'source_inventory.json')
    c.require({p.relative_to(c.ROOT).as_posix() for p in c.SOURCE.rglob('*') if p.is_file()} ==
              {x['path'] for x in inventory['files']}, 'Original source inventory changed')
    clock_guard(launch,original)


def run(expected_freeze_sha256):
    base = c.OUTPUT
    c.require(c.file_hash(base/'freeze.json') == expected_freeze_sha256, 'Expected reviewed recovery freeze')
    freeze = c.read(base/'freeze.json')
    c.require(not (base/'launch.json').exists(), 'No second recovery execution or retry')
    started, epoch = time.monotonic(), time.time(); deadline = started+c.MAX_SECONDS
    launch = {'time':c.now(),'epoch_seconds':epoch,'monotonic_start':started,'maximum_seconds':c.MAX_SECONDS,
        'pid':os.getpid(),'identity':{'pid':os.getpid(),'created':psutil.Process().create_time()},
        'freeze':c.artifact(base/'freeze.json'),'source_launch':c.artifact(c.SOURCE/'launch.json'),
        'recovery_plan':c.artifact(base/'recovery_plan.json'),
        'torch_loaded_in_supervisor':'torch' in sys.modules,'imported_completed_games':59,
        'original_primary_complete':False,'outcomes_closed':True,'amendment':c.AMENDMENT}
    c.save_new(base/'launch.json',launch)
    completed, failure = 59, None
    try:
        c.require('torch' not in sys.modules and Path(sys.executable).resolve() == Path(freeze['executable']).resolve(),
                  'Qualified model-free supervisor required')
        c.require(processor_topology() == freeze['topology'], 'Qualified full supervisor affinity required')
        save_runtime_sources(base/'runtime_sources.json')
        check_source_closed(); verify_execution_inputs(freeze,launch); check_interrupted_request()
        from tools.matched_strength_20261020.statistics import runtime_contract
        c.require(c.read(c.original.PREPARATION/'statistics_runtime.json') == runtime_contract(), 'Statistical runtime changed')
        from tools.matched_strength_recovery_20261022.memory import wait_for_memory
        wait_for_memory(base,deadline,name='initial',reserve=c.FINAL_RESERVE,request_seconds=0,initial=True)
        cpu = freeze['topology']['selected_logical_processor']
        schedule = c.read(base/'sealed/schedule.json'); spec = next(x for x in schedule if x['game'] == 59)
        c.save_new(base/'games_admission.json', {'time':c.now(),'validation_reused':True,
            'adapter_validation':c.artifact(base/'adapter_validation.json'),'outcomes_closed':True,
            'remaining_seconds':c.boundary(base,deadline,memory_headroom=False)['remaining_seconds']})
        play_final_game(spec,deadline,cpu); completed = 60
        manifest = make_audit_manifest(); c.save_new(base/'audit_manifest.json',manifest)
        for index, entry in enumerate(manifest['entries']):
            row = c.read(c.ROOT/entry['move']['path'])
            audit_attempt(entry['id'],row,entry['mode'],deadline,cpu)
            print(c.canonical({'stage':'amended_continuation_verification','completed_roots':index+1,
                'total_roots':len(manifest['entries']),'mode':entry['mode'],'outcomes_closed':True}),flush=True)
        c.boundary(base,deadline,memory_headroom=False)
    except BaseException as error:
        failure = {'category':type(error).__name__,'detail_path':'sealed/failure.json'}
        c.save_new(base/'sealed/failure.json',{'time':c.now(),'traceback':traceback.format_exc()})
    try:
        verify_execution_inputs(freeze,launch)
        c.save_new(base/'final_input_check.json',{'time':c.now(),'all_hashes_and_sizes_match':True,
                                               'files':len(freeze['inputs'])})
        c.boundary(base,deadline,memory_headroom=False)
    except BaseException as error:
        failure = failure or {'category':type(error).__name__,'detail_path':'sealed/final_check_failure.json'}
        c.save_new(base/'sealed/final_check_failure.json',{'time':c.now(),'traceback':traceback.format_exc()})
    result = {'time':c.now(),'status':'completed_pending_independent_analysis' if failure is None else 'preserved_hold',
        'failure':failure,'validation_passed':True,'validation_reused':True,'new_validation_requests':0,
        'completed_games':completed,'imported_completed_games':59,'elapsed_seconds':time.monotonic()-started,
        'outcomes_closed':True,'strength_analysis_games':0,'larger_data_deferred':True,
        'amendment':c.AMENDMENT,'original_primary_complete':False,'study_label':'amended_recovered_benchmark'}
    c.save_new(base/'operational_progress.json',result)
    c.save_new(base/'completion.json',{**result,'worker_processes_closed':True,
                                     'progress':c.artifact(base/'operational_progress.json')})
    print(c.canonical(result),flush=True)
    if failure is not None: return 1
    from tools.matched_strength_recovery_20261022.completion import complete
    return complete(deadline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--freeze-sha256'); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        if args.freeze_sha256 is None: parser.error('run requires --freeze-sha256')
        with c.execution_lease(): raise SystemExit(run(args.freeze_sha256))

