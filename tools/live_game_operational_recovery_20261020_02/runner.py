"""Second explicitly authorized recovery; preserve completed games and exact streams."""
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

from tools.live_game_operational_recovery_20261020_02 import common as c
from tools.external_strength_screen_20261018 import rules, audit
from tools.external_strength_screen_20261018_02.runner import hidden_variant
from tools.splendorust_120_qualification_20261018.common import ProcessTreeWorker, load_profile
from tools.splendorust_120_qualification_20261018.runner import check_result as check_opponent
from tools.corrected_planner_qualification_20261018.clock import PlannerWorker
from tools.local_player_qualification_128.clock import processor_topology, WorkerInitializationError

PLAYER_FAILURES = {'timeout','worker_exit','malformed','wrong_request','illegal','unsolicited','write_error'}
TESTS = ['tests/test_live_game_operational_recovery_20261020_02.py',
         'tests/test_live_game_recovery_memory_20261020_02.py',
         'tests/test_live_game_operational_recovery_analyzer_20261020_02.py',
         'tests/test_operational_recovery_completion_20261020_02.py']
PROTOCOL = 'docs/engineering/live_game_operational_recovery_20261020_02.md'
ANALYZER = 'tools/analyze_live_game_operational_recovery_20261020_02.py'
COORDINATOR = 'tools/operational_recovery_completion_20261020_02.py'
REVIEW = 'docs/engineering/live_game_operational_recovery_20261020_02_review.md'


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
    """Fixed first-attempt prefix; all later imported work belongs to recovery01."""
    c.require(type(game) is int and game in c.GAME_ORDER and type(ply) is int and 0 <= ply < c.MAX_PLIES,
              'Invalid imported decision identity')
    return c.ORIGINAL if game in (6,8,10,7,9) or game==11 and ply<15 else Path(source_base)


def read_saved_move(descriptor, path, source_base):
    """Accept an older descriptor only through the immediate source's sealed copy map."""
    value=bound_read(descriptor)
    target=(c.ROOT/descriptor['path']).resolve()
    if target!=Path(path).resolve():
        plan=c.read(Path(source_base)/'recovery_plan.json')
        c.require(plan['schema']=='live_operational_recovery_plan_v1','Immediate source copy-map schema')
        matches=[item for item in plan['imports']
                 if (c.ROOT/item['destination']['path']).resolve()==Path(path).resolve()]
        c.require(len(matches)==1 and matches[0]['source']==descriptor and
                  matches[0]['destination']==c.artifact(path),'Unbound transitive saved-move alias')
    c.require(value==c.read(path),'Saved move copy differs')
    return value


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


def check_unstarted_partial_attempt(recovered):
    name=f'games/g{c.PARTIAL_GAME:03d}/t{c.NEXT_PLY:04d}'
    path=c.SOURCE/'decision_attempts'/(name+'.json');row=c.read(path)
    c.require(row['id']==name and row['status']=='incomplete' and row['correct'] is False and
              row.get('failure_category')=='ResourceHold' and
              not any(k in row for k in ('initialization','clock','receipt','initialization_artifact','result_sha256','trace')),
              'Interrupted decision had started or produced a result; no automatic repeat allowed')
    c.require(not (c.SOURCE/'workers'/name).exists() and
              not (c.SOURCE/'sealed/games'/f'{c.PARTIAL_GAME:03d}'/'moves'/f'{c.NEXT_PLY:04d}.json').exists(),
              'Interrupted move already has worker or accepted move evidence')
    snapshot=recovered['snapshot'];state=recovered['state']
    spec=next(x for x in c.make_schedule() if x['game']==c.PARTIAL_GAME)
    player='local' if state.current_player==spec['local_seat'] else 'opponent'
    observer=(c.make_observer_payload(snapshot,c.operational_context(c.PARTIAL_GAME,c.NEXT_PLY))
              if player=='local' else rules.observation_payload(state))
    c.require(row['player']==player and row['observer']==observer,'Interrupted lawful input differs from reconstructed state')
    return path


def validate_closed_source(base, freeze, expected_games):
    completion=c.read(base/'completion.json');execution=c.read(base/'runner_execution.json')
    launch=c.read(base/'launch.json');preserved=c.read(base/'final_input_check.json')
    c.require(completion['status']=='preserved_hold' and completion['completed_games']==expected_games and
              completion['validation_passed'] is True and completion['failure']['category']=='ResourceHold' and
              completion['outcomes_closed'] is True and completion['worker_processes_closed'] is True and
              execution['exit_code']==1 and execution['runner_identity']['pid']==launch['pid'],
              'Unexpected preserved source ending')
    c.require(preserved['all_hashes_and_sizes_match'] is True and preserved['files']==len(freeze['inputs']),
              'Source final preservation check missing')


def inherited_freeze_contract(source):
    """Retain player/rule contracts while recording the newly authorized shorter cap."""
    return {**{k:v for k,v in source.items() if k not in ('time','inputs')},
            'execution_cap_seconds':c.MAX_SECONDS,
            'post_game_reserve_seconds':c.POST_GAME_RESERVE,'final_reserve_seconds':c.FINAL_RESERVE}


def prepare():
    base=c.OUTPUT;base.mkdir(parents=True,exist_ok=False)
    try:
        c.require('torch' not in sys.modules,'Preparation must remain model-free')
        c.require(shutil.disk_usage(c.ROOT).free>=c.MAX_BYTES+c.MIN_FREE,'Preparation disk headroom')
        c.require(c.file_hash(c.SOURCE/'freeze.json')==c.SOURCE_FREEZE_SHA256 and
                  c.file_hash(c.ORIGINAL/'freeze.json')==c.ORIGINAL_FREEZE_SHA256,'Preserved frozen identity differs')
        source_freeze=c.read(c.SOURCE/'freeze.json');original_freeze=c.read(c.ORIGINAL/'freeze.json')
        topology=processor_topology()
        c.require(topology==source_freeze['topology']==original_freeze['topology'],
                  'Preparation must inherit original unrestricted topology and selected core')
        c.verify(source_freeze['inputs'])
        source_inputs={item['path']:item for item in source_freeze['inputs']}
        c.require(all(source_inputs.get(item['path'])==item for item in original_freeze['inputs']),
                  'Original frozen inputs missing from immediate source lineage')
        validate_closed_source(c.ORIGINAL,original_freeze,5)
        validate_closed_source(c.SOURCE,source_freeze,len(c.COMPLETED_GAMES))
        old_plan=c.read(c.SOURCE/'recovery_plan.json')
        c.require(old_plan['schema']=='live_operational_recovery_plan_v1' and
                  old_plan['source_freeze']==c.artifact(c.ORIGINAL/'freeze.json') and
                  old_plan['source_snapshot']==c.artifact(c.SOURCE/'source_snapshot.json'),
                  'Immediate source lineage identity')
        old_snapshot=c.read(c.SOURCE/'source_snapshot.json')
        c.require({item['path'] for item in old_snapshot['files']}==
                  {p.relative_to(c.ROOT).as_posix() for p in c.ORIGINAL.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts},'Original first-attempt inventory changed')
        c.require(all(source_inputs.get(item['path'])==item for item in old_snapshot['files']),
                  'Original snapshot not immediate-source-frozen')
        validation=c.read(c.SOURCE/'adapter_validation.json')
        c.require(validation['status']=='passed' and validation['requests']==12 and validation['all_full_replays'] is True,
                  'Original full validation is incomplete')
        schedule=c.read(c.SOURCE/'schedule.json');c.require(schedule==c.make_schedule(),'Recovery may not alter schedule')
        specs={row['game']:row for row in schedule};count=len(c.COMPLETED_GAMES)
        c.require(tuple(row['game'] for row in schedule[:count])==c.COMPLETED_GAMES and
                  schedule[count]['game']==c.PARTIAL_GAME,'Recovery schedule boundary differs')
        c.require({p.name for p in (c.SOURCE/'sealed/games').iterdir() if p.is_dir()}==
                  {f'{g:03d}' for g in (*c.COMPLETED_GAMES,c.PARTIAL_GAME)},'Unexpected source game inventory')
        recovered={g:reconstruct_game(specs[g],c.SOURCE,complete=True) for g in c.COMPLETED_GAMES}
        partial=reconstruct_game(specs[c.PARTIAL_GAME],c.SOURCE,complete=False,next_ply=c.NEXT_PLY)
        failed=check_unstarted_partial_attempt(partial)
        c.require(sum(value['failure'] is not None for value in recovered.values())==4 and
                  all(value['failure'] is None or value['failure'][0]=='forfeit_timeout' and value['failure'][2]=='local'
                      for value in recovered.values()),'Expected four retained local timeout forfeits')
        c.require(not (c.SOURCE/'audit_manifest.json').exists() and
                  not any((c.SOURCE/'audit_rows').glob('g*.json')),'Unexpected completed source game audits')
        snapshot={'schema':'operational_recovery01_snapshot_v1','source_base':c.SOURCE.relative_to(c.ROOT).as_posix(),
                  'files':[c.artifact(p) for p in sorted(c.SOURCE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
                  'outcomes_closed':True}
        c.save_new(base/'source_snapshot.json',snapshot)
        sources=set()
        for name in ('schedule.json','local_profile.json','validation_cases.json','seed_collision_check.json','adapter_validation.json'):
            sources.add(c.SOURCE/name)
        for directory in ('validation/rows','decision_attempts/validation','audit_rows/validation'):
            sources.update((c.SOURCE/directory).glob('*.json'))
        for game,value in [*recovered.items(),(c.PARTIAL_GAME,partial)]:
            folder=c.SOURCE/'sealed/games'/f'{game:03d}';sources.add(folder/'initial.json')
            if game in recovered:sources.add(folder/'result.json')
            for path,row in zip(value['move_paths'],value['rows'],strict=True):
                sources.add(path);sources.add(c.SOURCE/'decision_attempts'/(row['id']+'.json'))
        sources.update(p for p in (c.SOURCE/'preserved').rglob('*') if p.is_file())
        imports=[copy_new(path,base/path.relative_to(c.SOURCE)) for path in sorted(sources)]
        imports.append(copy_new(failed,base/'preserved/incomplete_g021_t0011.json'))
        imports.append(copy_new(c.SOURCE/'sealed/failure.json',base/'preserved/recovery01_failure.json'))
        preserved=[]
        for origin,game,ply in ((c.ORIGINAL,11,15),(c.SOURCE,21,11)):
            preserved.extend(c.artifact(origin/name) for name in (
                f'decision_attempts/games/g{game:03d}/t{ply:04d}.json','sealed/failure.json','completion.json',
                'coordinator_hold.json','operational_progress.json','runner_execution.json'))
        plan={'schema':'live_operational_recovery02_plan_v1','source_base':c.SOURCE.relative_to(c.ROOT).as_posix(),
              'source_freeze':c.artifact(c.SOURCE/'freeze.json'),'original_freeze':c.artifact(c.ORIGINAL/'freeze.json'),
              'source_launch':c.artifact(c.SOURCE/'launch.json'),'source_completion':c.artifact(c.SOURCE/'completion.json'),
              'source_runner_execution':c.artifact(c.SOURCE/'runner_execution.json'),
              'source_snapshot':c.artifact(base/'source_snapshot.json'),'imports':imports,
              'completed_games':list(c.COMPLETED_GAMES),
              'partial_game':{'game':c.PARTIAL_GAME,'next_ply':c.NEXT_PLY,'state_sha256':partial['state_sha256']},
              'preserved_failure':preserved,'validation_reused':True,'outcomes_closed':True,'new_validation_requests':0}
        c.save_new(base/'recovery_plan.json',plan)
        c.require(all((c.ROOT/name).is_file() for name in (*TESTS,PROTOCOL,ANALYZER,COORDINATOR,REVIEW)),
                  'Recovery code/test/protocol/review/coordinator incomplete')
        test_command=[sys.executable,'-B','-c',
            "import psutil,pytest,sys; psutil.Process().cpu_affinity([2]); raise SystemExit(pytest.main(sys.argv[1:]))",
            '-q',*TESTS,'-p','no:cacheprovider']
        test=subprocess.run(test_command,cwd=c.ROOT,
            env={**os.environ,'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'PYTHONDONTWRITEBYTECODE':'1'},
            text=True,capture_output=True,timeout=300,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        c.save_new(base/'preflight_tests.json',{'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr,
                                             'command':test_command})
        c.require(test.returncode==0,'Recovery preflight failed; preserve attempt')
        paths={item['path'] for item in source_freeze['inputs']}|{item['path'] for item in snapshot['files']}
        for directory in (c.ROOT/'tools/live_game_operational_recovery_20261020_02',base,
                          c.ROOT/'outputs/live_game_operational_recovery_20261020_02_preparation'):
            paths.update(p.relative_to(c.ROOT).as_posix() for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        paths.update((*TESTS,PROTOCOL,ANALYZER,COORDINATOR,REVIEW))
        c.verify(source_freeze['inputs']);c.verify(snapshot['files'])
        c.require(processor_topology()==topology,'Preparation topology changed')
        c.save_new(base/'freeze.json',{**inherited_freeze_contract(source_freeze),
            'schema':'live_operational_recovery02_freeze_v1','time':c.now(),'inputs':[c.artifact(c.ROOT/p) for p in sorted(paths)],
            'source_freeze':c.artifact(c.SOURCE/'freeze.json'),'original_freeze':c.artifact(c.ORIGINAL/'freeze.json'),
            'source_snapshot':c.artifact(base/'source_snapshot.json'),'recovery_plan':c.artifact(base/'recovery_plan.json'),
            'topology':topology,'executable':sys.executable,
            'validation_reused':True,'new_validation_requests':0,'imported_completed_games':count})
        print(c.canonical({'prepared':True,'imported_completed_games':count,'resumed_game':c.PARTIAL_GAME,
                           'next_ply':c.NEXT_PLY,'new_validation_requests':0,'outcomes_closed':True}),flush=True)
    except BaseException:
        c.save_new(base/'preparation_failure.json',{'time':c.now(),'failure':traceback.format_exc(),'outcomes_closed':True})
        raise



def without_result(value):
    return {k: v for k, v in value.items() if k != 'result'}


def worker_options(deadline, cpu, *, reserve, mode, trace=None):
    c.require('torch' not in sys.modules, 'Supervisor must remain model-free')
    admission_started = time.monotonic()
    admission = c.boundary(c.OUTPUT, deadline, memory_headroom=False)
    request_seconds = c.AUDIT_SECONDS if mode == 'audit' else c.MOVE_SECONDS
    if admission['remaining_seconds'] < c.INIT_SECONDS+request_seconds+reserve:
        raise c.ResourceHold('Insufficient dual-clock worker admission reserve')
    effective_deadline = min(deadline-reserve, admission_started+admission['remaining_seconds']-reserve)
    def live_guard():
        receipt=c.boundary(c.OUTPUT,deadline,memory_headroom=False)
        if receipt['remaining_seconds'] <= reserve:
            raise c.ResourceHold('Dual-clock protected reserve reached')
        return receipt
    environment = {'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'CUDA_VISIBLE_DEVICES': '',
        'LIVE_WORKER_MODE': mode, 'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'}
    if trace is not None: environment['LIVE_TRACE_PATH'] = str(trace.resolve())
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


def completed_result(row):
    receipt = c.read(c.ROOT/row['receipt']['path'])
    c.require(c.artifact(c.ROOT/row['receipt']['path']) == row['receipt'], 'Receipt artifact changed')
    return receipt['result']


def memory_gate(name, deadline, *, reserve, request_seconds, initial=False):
    from tools.live_game_operational_recovery_20261020_02.memory import wait_for_memory
    value=wait_for_memory(c.OUTPUT,deadline,name=name,reserve=reserve,
                          request_seconds=request_seconds,initial=initial)
    path=c.OUTPUT/'memory_admissions'/(name+'.json')
    c.require(c.read(path)==value,'Memory admission saved receipt differs')
    return c.artifact(path)


def decision_attempt(name, player, snapshot, observer, deadline, cpu, *, reserve):
    if deadline-time.monotonic() < c.INIT_SECONDS+c.MOVE_SECONDS+reserve:
        raise c.ResourceHold('Insufficient move reserve')
    folder = c.OUTPUT/'workers'/name
    trace = folder/'continuations.jsonl.gz'
    row = {'id': name, 'player': player, 'observer': observer, 'status': 'incomplete', 'correct': False}
    started = time.monotonic()
    try:
        row['memory_admission']=memory_gate(name,deadline,reserve=reserve,request_seconds=c.MOVE_SECONDS)
        audit.check_observer(snapshot, c.opponent_observer(observer) if player == 'local' else observer)
        objects = rules.legal_actions(rules.deserialize_snapshot(snapshot))
        legal = [c.serialize_action(a) for a in objects]
        worker_class = PlannerWorker if player == 'local' else ProcessTreeWorker
        module = ('tools.live_game_operational_recovery_20261020_02.worker' if player == 'local'
                  else 'tools.splendorust_120_qualification_20261018.worker')
        payload = {'observer': observer} if player == 'local' else {'player': 'opponent', 'observer': observer}
        with worker_class(module, folder, **worker_options(deadline, cpu, reserve=reserve, mode=player, trace=trace)) as child:
            row['initialization'] = without_result(child.initialization)
            receipt = child.request(payload, c.MOVE_SECONDS)
            row['clock'] = without_result(receipt)
        row['initialization_artifact'] = c.artifact(folder/'initialization.json')
        row['receipt'] = c.artifact(folder/'request-00000001.json')
        if receipt['status'] != 'ok':
            if receipt['status'] not in PLAYER_FAILURES:
                raise ValueError('Hard worker or resource failure: '+receipt['status'])
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
    if deadline-time.monotonic() < c.INIT_SECONDS+c.AUDIT_SECONDS+c.FINAL_RESERVE:
        raise c.ResourceHold('Insufficient audit/finalization reserve')
    result = completed_result(row); observer = row['observer']
    trace_path=(c.ROOT/row['trace']['path']).resolve()
    c.require(c.artifact(trace_path)==row['trace'],'Audit trace source hash/size mismatch')
    if any(trace_path.is_relative_to(base.resolve()) for base in (c.ORIGINAL,c.SOURCE)):
        c.require(c.authenticated_source_artifact(trace_path)==row['trace'],'Imported trace not recovery-frozen')
    else:c.require(trace_path.is_relative_to(c.OUTPUT.resolve()),'Audit trace outside approved evidence')
    folder = c.OUTPUT/'audits'/name
    record = {'id': name, 'mode': mode, 'observer_sha256': c.digest(observer),
              'result_sha256': c.digest(result), 'correct': False}
    started = time.monotonic()
    try:
        record['memory_admission']=memory_gate('audits/'+row['id'],deadline,
                                               reserve=c.FINAL_RESERVE,request_seconds=c.AUDIT_SECONDS)
        with PlannerWorker('tools.live_game_operational_recovery_20261020_02.worker', folder,
                **worker_options(deadline, cpu, reserve=c.FINAL_RESERVE, mode='audit')) as child:
            record['initialization'] = without_result(child.initialization)
            receipt = child.request({'observer': observer, 'result': result,
                'traces_path': str((c.ROOT/row['trace']['path']).resolve()), 'mode': mode}, c.AUDIT_SECONDS)
            record['clock'] = without_result(receipt)
        record['initialization_artifact'] = c.artifact(folder/'initialization.json')
        record['receipt'] = c.artifact(folder/'request-00000001.json')
        c.require(receipt['status'] == 'ok', 'Independent continuation audit did not complete')
        c.check_audit_proof(receipt['result'], observer, result, mode)
        record.update(verification=receipt['result'], correct=True)
        return record
    finally:
        record['total_seconds'] = time.monotonic()-started
        c.save_new(c.OUTPUT/'audit_rows'/(name+'.json'), record)


def scientific_projection(decision):
    value = deepcopy(decision)
    value['record'].pop('profile')
    return value


def make_audit_manifest():
    entries = []
    for game in range(6, 24):
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


def play_game(spec, deadline, cpu, *, recovered=None):
    folder = c.OUTPUT/'sealed'/'games'/f"{spec['game']:03d}"
    env = recovered['env'] if recovered is not None else rules.CorrectedEnv(seed=int(spec['seed']))
    if recovered is None:
        state=env.initial_state();initial=rules.serialize_snapshot(state);audit.validate_ledger(initial)
        c.save_new(folder/'initial.json',{**spec,'initial':initial,'initial_sha256':c.digest(initial)})
        game={**spec,'initial_sha256':c.digest(initial),'moves':[],'complete':False}
        start_ply=0
    else:
        c.require(spec['game']==c.PARTIAL_GAME and len(recovered['rows'])==c.NEXT_PLY,'Wrong recovered game boundary')
        state=recovered['state'];initial=c.read(folder/'initial.json')
        game={**spec,'initial_sha256':initial['initial_sha256'],
              'moves':[c.artifact(folder/'moves'/f'{ply:04d}.json') for ply in range(c.NEXT_PLY)],'complete':False}
        start_ply=c.NEXT_PLY
    for ply in range(start_ply,c.MAX_PLIES):
        if state.terminal: break
        before = rules.serialize_snapshot(state); actor = state.current_player
        player = 'local' if actor == spec['local_seat'] else 'opponent'
        observer = (c.make_observer_payload(before, c.operational_context(spec['game'], state.turn_index))
                    if player == 'local' else rules.observation_payload(state))
        # Per-root lawful redaction remains checked independently of played outcomes.
        alternate = rules.serialize_snapshot(hidden_variant(state))
        alternate_observer = (c.make_observer_payload(alternate, observer['context']) if player == 'local'
                              else rules.observation_payload(rules.deserialize_snapshot(alternate)))
        c.require(observer == alternate_observer, 'Actual-game observation privacy mismatch')
        name = f"games/g{spec['game']:03d}/t{ply:04d}"
        row = {'game': spec['game'], 'pair': spec['pair'], 'fit': spec['fit'], 'ply': ply,
               'actor': actor, 'before_sha256': c.digest(before),
               **decision_attempt(name, player, before, observer, deadline, cpu, reserve=c.POST_GAME_RESERVE)}
        if row['status'] == 'completed':
            result = completed_result(row)
            action = result['decision']['action'] if player == 'local' else result['action']
            state = env.step(state, c.deserialize_action(action))
            after = rules.serialize_snapshot(state)
            c.require(c.digest(after) == row['after_sha256'], 'Actual referee successor mismatch')
            audit.check_step(before, action, after)
        else:
            c.require(row['status'].startswith('forfeit_'), 'Unscored hard failure')
            game.update(complete=True, reason=row['status'], winner=1-actor, failing_player=player)
        path = folder/'moves'/f'{ply:04d}.json'; c.save_new(path, row); game['moves'].append(c.artifact(path))
        print(c.canonical({'stage': 'operational_games', 'game': spec['game'], 'ply': ply,
            'player': player, 'decision_status': row['status'],
            'move_seconds': row['clock'].get('receipt_elapsed_seconds'), 'outcomes_closed': True}), flush=True)
        if game['complete']: break
    if not game['complete']:
        if not state.terminal:
            c.save_new(folder/'unresolved.json', {**game, 'final': rules.serialize_snapshot(state), 'reason': 'ply_ceiling'})
            raise c.ResourceHold('Nonterminal maximum-ply ceiling; no invented game result')
        game.update(complete=True, reason='natural', winner=state.winner)
    game['final'] = rules.serialize_snapshot(state)
    c.save_new(folder/'result.json', game)
    return game


def run():
    base=c.OUTPUT;freeze=c.read(base/'freeze.json');c.verify(freeze['inputs'])
    c.require(freeze['schema']=='live_operational_recovery02_freeze_v1' and not (base/'launch.json').exists(),
              'Never restart a launched recovery attempt')
    c.require(processor_topology()==freeze['topology'],'Production supervisor must inherit original full topology')
    plan=c.read(base/'recovery_plan.json');c.require(plan['schema']=='live_operational_recovery02_plan_v1','Recovery plan schema')
    started=time.monotonic();deadline=started+c.MAX_SECONDS
    c.save_new(base/'launch.json',{'time':c.now(),'epoch_seconds':time.time(),'monotonic_start':started,'pid':os.getpid(),
        'freeze':c.artifact(base/'freeze.json'),'maximum_seconds':c.MAX_SECONDS,'torch_loaded_in_supervisor':'torch' in sys.modules,
        'recovery_plan':c.artifact(base/'recovery_plan.json'),'source_launch':plan['source_launch'],
        'validation_reused':True,'imported_completed_games':len(c.COMPLETED_GAMES),'outcomes_closed':True})
    games=[c.read(base/'sealed/games'/f'{g:03d}'/'result.json') for g in c.COMPLETED_GAMES]
    failure=None
    try:
        initial_admission=memory_gate('initial',deadline,reserve=c.POST_GAME_RESERVE,
                                     request_seconds=c.MOVE_SECONDS,initial=True)
        schedule=c.read(base/'schedule.json');spec=next(x for x in schedule if x['game']==c.PARTIAL_GAME)
        recovered=reconstruct_game(spec,c.SOURCE,complete=False,next_ply=c.NEXT_PLY)
        check_unstarted_partial_attempt(recovered)
        c.require(recovered['state_sha256']==plan['partial_game']['state_sha256'],'Recovery prefix changed since freeze')
        c.save_new(base/'resume_check.json',{'time':c.now(),'game':c.PARTIAL_GAME,'next_ply':c.NEXT_PLY,
            'state_sha256':recovered['state_sha256'],'prefix_transitions':c.NEXT_PLY,'no_completed_request_repeated':True,
            'outcomes_closed':True})
        c.save_new(base/'games_admission.json',{'time':c.now(),'adapter_validation':c.artifact(base/'adapter_validation.json'),
            'memory_admission':initial_admission,'validation_reused':True,
            'remaining_seconds':c.boundary(base,deadline,memory_headroom=False)['remaining_seconds'],'outcomes_closed':True})
        cpu=freeze['topology']['selected_logical_processor']
        for spec in schedule[len(c.COMPLETED_GAMES):]:
            games.append(play_game(spec,deadline,cpu,recovered=recovered if spec['game']==c.PARTIAL_GAME else None))
            print(c.canonical({'stage':'operational_games','completed_games':len(games),'total':18,
                               'imported_completed_games':len(c.COMPLETED_GAMES),'outcomes_closed':True}),flush=True)
        manifest=make_audit_manifest();c.save_new(base/'audit_manifest.json',manifest)
        for index,entry in enumerate(manifest['entries']):
            row=c.read(c.ROOT/entry['move']['path'])
            audit_attempt(entry['id'],row,entry['mode'],deadline,cpu)
            print(c.canonical({'stage':'continuation_verification','completed_roots':index+1,
                'total_roots':len(manifest['entries']),'mode':entry['mode'],'outcomes_closed':True}),flush=True)
        c.boundary(base,deadline,memory_headroom=False)
    except BaseException as error:
        failure={'category':type(error).__name__,'detail_path':'sealed/failure.json'}
        c.save_new(base/'sealed/failure.json',{'time':c.now(),'traceback':traceback.format_exc(),'outcomes_closed':True})
    try:
        c.verify(freeze['inputs'])
        c.save_new(base/'final_input_check.json',{'time':c.now(),'all_hashes_and_sizes_match':True,'files':len(freeze['inputs'])})
        c.boundary(base,deadline,memory_headroom=False)
    except BaseException as error:
        failure=failure or {'category':type(error).__name__,'detail_path':'sealed/final_check_failure.json'}
        c.save_new(base/'sealed/final_check_failure.json',{'time':c.now(),'traceback':traceback.format_exc(),'outcomes_closed':True})
    result={'time':c.now(),'status':'completed_pending_independent_analysis' if failure is None else 'preserved_hold',
        'failure':failure,'validation_passed':True,'validation_reused':True,'new_validation_requests':0,
        'imported_completed_games':len(c.COMPLETED_GAMES),'completed_games':len(games),'elapsed_seconds':time.monotonic()-started,
        'outcomes_closed':True,'strength_analysis_games':0,'larger_data_deferred':True}
    c.save_new(base/'operational_progress.json',result)
    c.save_new(base/'completion.json',{**result,'worker_processes_closed':True,'progress':c.artifact(base/'operational_progress.json')})
    print(c.canonical(result),flush=True)
    return 0 if failure is None else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:
        with c.execution_lease():raise SystemExit(run())

