"""Fixed production schedules around unchanged accepted collection decisions."""
from pathlib import Path
import threading
import time
from . import common as c, data, private
from .. import schedule

_legacy = private.load('collector')


def build_plan():
    specs=schedule.collection_schedule()
    return dict(schema='independent_data_collection_games_20261115_v1', specifications=specs,
        independent_setups=schedule.COLLECTION_GAMES, matched_trajectories=schedule.COLLECTION_GAMES,
        games_per_cell=schedule.GAMES_PER_CELL, maximum_plies=2000,
        dynamic_queue='fixed ascending game index; next game assigned only to idle persistent worker',
        both_seats_same_backend=True, targets='actual full legal Main visit distribution; official actor-perspective terminal value only',
        exclusions=['forced_pass','single_choice','late_heuristic'],retries=0,replacements=0,training_updates=0,
        model_replay_selection='first neural decision per game in bands0..19,20..39,40+',
        model_replay_slot='(original assigned slot+1)%8; no other active game clients',model_replay_maximum_per_game=3,
        throughput_includes='first release through durable complete target manifest and actual corpus loader validation',
        throughput_excludes='runtime/native startup, five service warmups, selected model replays and independent saved audit')


def cells():
    specs=schedule.collection_schedule()
    return [dict(cell_id=f'c{i:03d}', block=i+1,route='cpu',concurrency=8,scheduler='original8',implementation='eager',
                 game_indices=list(range(i*16,(i+1)*16)),specifications=specs[i*16:(i+1)*16])
            for i in range(schedule.CELL_COUNT)]


_legacy.build_plan=build_plan
decide_snapshot, warmup_runtime = _legacy.decide_snapshot, _legacy.warmup_runtime
run_cell, run_replays = _legacy.run_cell, _legacy.run_replays


def run_engineering(plan,runtime,folder,boundary):
    """Two fixed excluded complete games, with the actual saving and loader paths."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    specs=schedule.qualification_schedule()
    c.require(c.canonical(specs)==c.canonical(plan['engineering_games']), 'Fixed excluded engineering games')
    _legacy.q.validate_runtime(runtime);warmup_runtime(plan,runtime,folder)
    documents=[];outcomes=[];stopped=threading.Event()
    for index,spec in enumerate(specs):
        boundary();adapter=runtime.make_adapter('engineering_'+str(index),index,512)
        try:
            c.save(folder/'assignments'/f'{index:02d}.json',dict(queue_index=index,game_id=spec['game_id'],
                worker_index=index,client_id=adapter.client_id,slot=index,
                assigned_perf_counter=time.perf_counter(),assigned_unix=time.time()))
            result=_legacy._play_game(plan,runtime,adapter,spec,folder/'games'/spec['game_id'],stopped,boundary)
            c.save(folder/'games'/spec['game_id']/'outcome.json',result)
            c.require(result['status']=='completed','Excluded qualification game must complete naturally; no replacement')
            documents.append(c.artifact(folder/'games'/spec['game_id']/'target_game.json'));outcomes.append(result)
        finally:adapter.close()
    manifest=data.build_manifest(documents,specs,route='cpu',cell_id='engineering')
    c.save(folder/'corpus_manifest.json',manifest);targets=[]
    def sink(spec,examples):
        targets.append(c.save(folder/'targets'/(spec['game_id']+'.json'),dict(schema='gpu_collector_examples_v1',
            **data.IDENTITY,spec=spec,examples=examples,official_terminal_only=True,teacher_simulations=512,training_updates=0)))
    loaded=data.load_corpus(manifest,expected_specs=specs,boundary=boundary,sink=sink)
    c.require(loaded['eligible_roots']>=32,'At least32 genuine engineering roots for actual trainer qualification')
    c.save(folder/'loader_result.json',loaded)
    c.save(folder/'cohort.json',dict(ended_perf_counter=time.perf_counter(),ended_unix=time.time()))
    replay=run_replays(plan,dict(specifications=specs),runtime,folder,boundary)
    result=dict(status='passed',games=2,excluded_from_production=True,corpus=c.artifact(folder/'corpus_manifest.json'),
        loader=c.artifact(folder/'loader_result.json'),targets=targets,model_replay=replay,outcomes=outcomes)
    c.save(folder/'engineering_result.json',result)
    return result
