"""Bounded actual self-play collector with complete official target evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib
import queue
import struct
import threading
import time
import traceback

from . import common as c, data
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.splendorust_gpu_throughput_20261105 import qualification as q, games as old_games
from tools.splendorust_gpu_throughput_20261105.cell import classify_decision_error,error_chain

MAX_PLIES=2000
SEED_DOMAIN='splendorust_gpu_collector_20261105_fresh_matched32_v1'


def build_plan():
    specs=[dict(game_id=f'g{i:03d}',game_index=i,
                seed_decimal=str(int.from_bytes(hashlib.sha256(c.canonical([SEED_DOMAIN,i]).encode()).digest(),'big')),
                start_player=i%2,quota=512,world_pool=3,cpuct=.4) for i in range(32)]
    seeds={x['seed_decimal'] for x in specs}; prior=[]
    for relative in old_games.PRIOR_SCHEDULES:
        records=c.read(c.ROOT/relative)
        c.require(not seeds & {str(x['seed_decimal']) for x in records},'new independent setup seeds')
        prior.append(c.artifact(c.ROOT/relative))
    for relative in (old_games.PRIOR_GAME_PLAN,'outputs/splendorust_gpu_throughput_20261105_01/plan.json'):
        records=c.read(c.ROOT/relative)['games']['specifications']
        c.require(not seeds & {x['seed_decimal'] for x in records},'new seeds disjoint earlier GPU games')
        prior.append(c.artifact(c.ROOT/relative))
    return dict(schema='gpu_collector_games_v1',seed_domain=SEED_DOMAIN,specifications=specs,
        initial_states=[dict(game_id=x['game_id'],initial=data.initial_snapshot(x)) for x in specs],
        prior_schedules=prior,independent_setups=32,matched_trajectories=64,games_per_cell=16,
        maximum_plies=MAX_PLIES,dynamic_queue='fixed ascending game index; next game assigned only to idle persistent worker',
        both_seats_same_backend=True,targets='actual full legal Main visit distribution; official actor-perspective terminal value only',
        exclusions=['forced_pass','single_choice','late_heuristic'],retries=0,replacements=0,training_updates=0,
        model_replay_selection='first neural decision per game in bands0..19,20..39,40+',
        model_replay_slot='(original assigned slot+1)%8; no other active game clients',model_replay_maximum_per_game=3,
        throughput_includes='first release through durable complete target manifest and actual corpus loader validation',
        throughput_excludes='runtime/native startup, five service warmups, selected model replays and independent saved audit')


def warmup_runtime(plan,runtime,folder):
    source=plan['cases'][0]['expected_results'][runtime.route]
    prior=data.read_artifact(source)
    root=prior['result']['diagnostics']['phases'][0]['inference'][0]
    c.require(root['tokens']==plan['cases'][0]['expected_root_tokens'],'warmup uses accepted same-route root input')
    accepted_output=[root['output83']]
    accepted_bits=struct.pack('<83f',*accepted_output[0])
    records=[];runtime.service.register('warmup',0)
    try:
        expected=None
        for index in range(5):
            def operation(started):
                with runtime.operation('warmup',started=started,kind='service_warmup'):
                    return runtime.service.infer('warmup',plan['cases'][0]['expected_root_tokens'],1,
                                                 deadline=started+c.INTERNAL_SECONDS)
            (output,receipt),clock=q.timed_call(operation,abort=runtime.abort,
                on_receipt=lambda clock,i=index:c.save(folder/'warmup_clocks'/f'{i:02d}.json',clock))
            packed=b''.join(struct.pack('<83f',*row) for row in output)
            c.require(output==accepted_output and packed==accepted_bits,'warmup bit-exact accepted same-route root output')
            if expected is None:expected=(output,packed)
            c.require(expected==(output,packed),'same-route exact service warmups')
            row=dict(index=index,output=output,receipt=receipt,clock=clock)
            c.save(folder/'warmup_raw'/f'{index:02d}.json',row);records.append(row)
        c.save(folder/'warmup.json',dict(status='passed',calls=records,inside_cell_timing=False,
            accepted_source=source,exact_accepted_root_bits=True))
    finally:runtime.service.unregister('warmup')


def decide_snapshot(runtime,adapter,before,spec,ply,folder,*,kind='collector'):
    """The single actual decision/transition/target path, also used by qualification."""
    folder=Path(folder);state=guard.decode_training_snapshot(before)
    c.require(not state.terminal,'no postterminal collection')
    observer=rules.observation_payload(state);request=dict(player='search',observer=observer)
    client=adapter.client_id; name=f'{ply:06d}.json'
    def operation(started):
        c.save(folder/'requests'/name,request)
        c.save(folder/'attempts'/name,dict(game_id=spec['game_id'],ply=ply,client_id=client,slot=adapter.slot,
            kind=kind,started_perf_counter=started,observed_unix=time.time(),request_sha256=c.digest(request),
            before_sha256=c.digest(before),no_retry=True))
        return q.ordinary(runtime,{'search':adapter},request,started=started,client_ids={'search':client},
                          evidence_callback=lambda raw:c.save(folder/'raw_results'/name,raw))
    try:
        result,clock=q.timed_call(operation,abort=runtime.abort,
            on_receipt=lambda clock:c.save(folder/'clocks'/name,clock))
    finally:c.save(folder/'service_calls'/name,list(adapter.last_service_calls))
    capture=deepcopy(adapter.last_training_capture)
    c.save(folder/'captures'/name,capture)
    after=old_games.actual_after(before,result['action'])
    row=dict(game_id=spec['game_id'],spec=spec,route=runtime.route,ply=ply,actor=state.current_player,
             client_id=client,slot=adapter.slot,before=before,observer=observer,result=result,
             action=result['action'],trace=result['diagnostics'],clock=clock,after=after,training_capture=capture)
    c.save(folder/'moves'/name,row)
    data.validate_row(row,spec,before,ply)
    return row


class InvalidGame(ValueError):
    pass


def _play_game(plan,runtime,adapter,spec,folder,stopped,boundary):
    before=data.initial_snapshot(spec);rows=[];previous_pass=False;ply=0
    c.save(folder/'spec.json',spec);c.save(folder/'initial.json',before)
    c.save(folder/'started.json',dict(game_id=spec['game_id'],client_id=adapter.client_id,slot=adapter.slot,
        started_perf_counter=time.perf_counter(),started_unix=time.time()))
    try:
        for ply in range(MAX_PLIES):
            boundary()
            if guard.decode_training_snapshot(before).terminal:break
            if stopped.is_set():
                return dict(game_id=spec['game_id'],status='cancelled_between_decisions',completed_moves=len(rows),
                            next_snapshot_sha256=c.digest(before),no_retry=True)
            row=decide_snapshot(runtime,adapter,before,spec,ply,folder)
            rows.append(row);is_pass=row['action']['action_type']=='PASS'
            if row['after']==before or (previous_pass and is_pass):raise InvalidGame('Consecutive passes/deadlock; no targets')
            before=row['after'];previous_pass=is_pass
        if not guard.decode_training_snapshot(before).terminal or not rows:
            raise InvalidGame('Ply cap reached without official terminal; no targets')
        winner=guard.require_official_terminal_target(before,'completed')
        document=dict(schema=data.GAME_SCHEMA,**data.IDENTITY,route=runtime.route,spec=spec,reason='completed',
            initial=data.initial_snapshot(spec),final=before,winner=winner,
            rows=[c.artifact(folder/'moves'/f'{i:06d}.json') for i in range(len(rows))])
        c.save(folder/'target_game.json',document)
        summary=dict(status='naturally_completed',route=runtime.route,game_id=spec['game_id'],specification=spec,
            client_id=adapter.client_id,slot=adapter.slot,plies=len(rows),winner=winner,final=before,
            model_replay_selection=old_games.replay_selection(rows),target_game=c.artifact(folder/'target_game.json'))
        c.save(folder/'result.json',summary)
        finished=dict(game_id=spec['game_id'],ended_perf_counter=time.perf_counter(),ended_unix=time.time(),
                      scope='observed after durable official game and target source document')
        c.save(folder/'finished.json',finished)
        return dict(game_id=spec['game_id'],status='completed',completed_moves=len(rows),finished=finished,no_retry=True)
    except BaseException as error:
        stopped.set();path=folder/'clocks'/f'{ply:06d}.json';clock=c.read(path) if path.exists() else None
        status=classify_decision_error(error)
        if clock is None or clock.get('status')=='ok' or clock.get('completed') is True:status='integrity_failed'
        if isinstance(error,InvalidGame):status='invalid_game'
        failure=dict(game_id=spec['game_id'],status=status,failed_ply=ply,completed_moves=len(rows),
            next_snapshot_sha256=c.digest(before),clock=clock,error_chain=error_chain(error),error=traceback.format_exc(),no_retry=True)
        c.save(folder/'failure.json',failure)
        return failure


def run_replays(plan,spec,runtime,folder,boundary):
    started=time.perf_counter();receipts=[]
    for game in spec['specifications']:
        ending=c.read(folder/'games'/game['game_id']/'result.json');slot=(ending['slot']+1)%8
        adapter=runtime.make_adapter('replay_'+game['game_id'],slot,512)
        try:
            for selected in ending['model_replay_selection']:
                boundary();ply=selected['ply'];source=c.read(folder/'games'/game['game_id']/'moves'/f'{ply:06d}.json')
                destination=folder/'model_replays'/game['game_id']/f'{ply:06d}'
                row=decide_snapshot(runtime,adapter,source['before'],game,ply,destination,kind='selected_model_replay')
                exact=row['result']==source['result'] and row['training_capture']==source['training_capture'] and row['after']==source['after']
                receipt=dict(game_id=game['game_id'],ply=ply,source=c.artifact(folder/'games'/game['game_id']/'moves'/f'{ply:06d}.json'),
                    replay=c.artifact(destination/'moves'/f'{ply:06d}.json'),original_slot=ending['slot'],replay_slot=slot,
                    other_clients_inactive=True,exact_full_result_capture_and_after=exact)
                c.save(destination/'comparison.json',receipt)
                c.require(exact,'selected replay original result, visit targets and afterstate exact')
                receipts.append(receipt)
        finally:adapter.close()
    summary=dict(status='exact_replays_passed',count=len(receipts),receipts=receipts,
                 wall_seconds=time.perf_counter()-started,model_calls=len(receipts),inside_collection_timing=False)
    c.require(0<len(receipts)<=48,'fixed bounded selected replay inventory')
    c.save(folder/'model_replay_summary.json',summary)
    return summary


def run_cell(plan,spec,runtime,folder,boundary=lambda:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    c.require(plan['games']==build_plan(),'frozen fresh game plan')
    selected=[plan['games']['specifications'][i] for i in spec['game_indices']]
    c.require(spec['specifications']==selected and len(selected)==16 and spec['concurrency'] in (8,16)
              and runtime.route==spec['route'],'fixed matched whole-game cell')
    q.validate_runtime(runtime);warmup_runtime(plan,runtime,folder)
    adapters={};stopped=threading.Event();assign_lock=threading.Lock();next_index=0;outcomes={};assignments=[]
    start={};errors=[];threads=[]
    try:
        for index in range(spec['concurrency']):
            boundary();adapters[index]=runtime.make_adapter(f'worker_{index}',index%8,512)
        def release():
            start.update(started_perf_counter=time.perf_counter(),started_unix=time.time())
            c.save(folder/'started.json',dict(cell_id=spec['cell_id'],**start))
        barrier=threading.Barrier(spec['concurrency']+1,action=release)
        def run_worker(index):
            nonlocal next_index
            try:
                barrier.wait(timeout=c.OUTER_SECONDS)
                while not stopped.is_set():
                    boundary()
                    with assign_lock:
                        if stopped.is_set() or next_index>=len(selected):return
                        order=next_index;next_index+=1;game=selected[order]
                        assignment=dict(queue_index=order,game_id=game['game_id'],worker_index=index,client_id=f'worker_{index}',
                            slot=index%8,assigned_perf_counter=time.perf_counter(),assigned_unix=time.time())
                        c.save(folder/'assignments'/f'{order:02d}.json',assignment);assignments.append(assignment)
                    row=_play_game(plan,runtime,adapters[index],game,folder/'games'/game['game_id'],stopped,boundary)
                    c.save(folder/'games'/game['game_id']/'outcome.json',row)
                    with assign_lock:outcomes[game['game_id']]=row
                    if row['status']!='completed':stopped.set();return
            except BaseException:
                stopped.set()
                with assign_lock:errors.append(dict(worker_index=index,error=traceback.format_exc()))
        for index in adapters:
            thread=threading.Thread(target=run_worker,args=(index,),name=f'collector-worker-{index}',daemon=True)
            threads.append(thread);thread.start()
        barrier.wait(timeout=c.OUTER_SECONDS)
        while any(thread.is_alive() for thread in threads):
            boundary()
            for thread in threads:thread.join(timeout=.05)
        gameplay_end=time.perf_counter();gameplay_end_wall=time.time()
        c.save(folder/'queue_outcomes.json',dict(assignments=assignments,outcomes=outcomes,errors=errors,
            never_started=[x['game_id'] for x in selected if x['game_id'] not in outcomes]))
        c.require(not errors and not any(x['status']=='integrity_failed' for x in outcomes.values()),'collector technical failure')
        status='completed' if len(outcomes)==16 and all(x['status']=='completed' for x in outcomes.values()) else (
            'invalid_game' if any(x['status']=='invalid_game' for x in outcomes.values()) else 'deadline_failed')
        if status!='completed':
            summary=dict(status=status,cell_id=spec['cell_id'],outcomes=outcomes,full_cohort_completed=False,
                corpus_produced=False,retries=0,replacements=0,**start)
            c.save(folder/'cohort.json',summary)
            return summary
        documents=[c.artifact(folder/'games'/game['game_id']/'target_game.json') for game in selected]
        manifest=data.build_manifest(documents,selected,route=runtime.route,cell_id=spec['cell_id'])
        c.save(folder/'corpus_manifest.json',manifest);target_artifacts=[]
        def sink(game,examples):
            target_artifacts.append(c.save(folder/'targets'/(game['game_id']+'.json'),dict(
                schema='gpu_collector_examples_v1',**data.IDENTITY,spec=game,examples=examples,
                official_terminal_only=True,teacher_simulations=512,training_updates=0)))
        loaded=data.load_corpus(manifest,expected_specs=selected,boundary=boundary,sink=sink)
        c.save(folder/'loader_result.json',loaded)
        c.save(folder/'target_manifest.json',dict(schema='gpu_collector_loaded_targets_v1',**data.IDENTITY,
            corpus=c.artifact(folder/'corpus_manifest.json'),loader=c.artifact(folder/'loader_result.json'),targets=target_artifacts,
            count=loaded['eligible_roots'],training_updates=0))
        end=time.perf_counter();end_wall=time.time();elapsed=end-start['started_perf_counter']
        summary=dict(status='completed',cell_id=spec['cell_id'],route=runtime.route,concurrency=spec['concurrency'],
            **start,ended_perf_counter=end,ended_unix=end_wall,wall_seconds=elapsed,
            gameplay_end_perf_counter=gameplay_end,gameplay_end_unix=gameplay_end_wall,
            gameplay_seconds=gameplay_end-start['started_perf_counter'],target_loader_seconds=end-gameplay_end,
            games=16,plies=sum(x['plies'] for x in loaded['games']),eligible_roots=loaded['eligible_roots'],
            games_per_hour=16*3600/elapsed,eligible_roots_per_hour=loaded['eligible_roots']*3600/elapsed,
            full_cohort_completed=True,corpus_produced=True,retries=0,replacements=0,training_updates=0)
        c.save(folder/'cohort.json',summary)
    except BaseException:
        stopped.set();runtime.abort()
        cleanup_deadline=time.monotonic()+c.OUTER_SECONDS+5.
        for thread in threads:
            thread.join(timeout=max(0.,cleanup_deadline-time.monotonic()))
        c.require(not any(thread.is_alive() for thread in threads),'all collector threads returned after abort')
        c.save(folder/'failure.json',dict(error=traceback.format_exc(),no_retry=True));raise
    finally:
        for adapter in adapters.values():adapter.close()
    replay=run_replays(plan,spec,runtime,folder,boundary)
    return dict(summary,model_replay=replay)
