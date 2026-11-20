"""Actual accepted capture path with separate process/queue orchestration."""
from pathlib import Path
import time
from . import common as c, data, private
from .queueing import SharedStop, take_next, run_clients

_legacy = private.load('collector')
decide_snapshot = _legacy.decide_snapshot
warmup_runtime = _legacy.warmup_runtime
run_replays = _legacy.run_replays


def packing_contract():
    from .backend import source_equivalence
    return dict(route='cpu', native_batch=1, physical_batch=1,
        implementation='original_frozen_torchscript', dtype='float32',
        dispatch='one synchronous CPU callback per caller; never aggregate clients',
        control_clients=8, candidate_processes=4, candidate_clients_per_process=2,
        numerical_tolerance=None, exact_float32_required=True,
        source=c.artifact(c.ROOT/'tools/splendorust_gpu_collector_20261105/backend.py'))


class Session:
    """A single independently pinned model process; parent owns its lifecycle."""
    def __init__(self, plan, runtime, folder, *, process_index, clients, cell_folder, queue_path, stop_path, boundary):
        c.require(runtime.route=='cpu' and clients in (2,8), 'Unchanged CPU direct model route')
        self.plan,self.runtime,self.folder,self.boundary=plan,runtime,Path(folder),boundary
        self.process_index,self.clients=process_index,clients
        self.cell_folder,self.queue_path=Path(cell_folder),Path(queue_path)
        self.stopped=SharedStop(stop_path);self.adapters={}

    def run(self):
        _legacy.q.validate_runtime(self.runtime)
        warmup_runtime(self.plan,self.runtime,self.folder/'warmup')
        try:
            for index in range(self.clients):
                self.boundary()
                self.adapters[index]=self.runtime.make_adapter(f'worker_{index}',index,512)
            c.save(self.folder/'ready.json',dict(process_index=self.process_index,clients=self.clients,
                ready_perf_counter=time.perf_counter(),ready_unix=time.time(),warmup_complete=True))
            release=self.cell_folder/'release.json'
            while not release.exists():
                self.boundary()
                c.require(not self.stopped.is_set(),'Stopped before dispatch; no replacement')
                time.sleep(.025)
            released=c.read(release)
            c.require(released['services']==(1 if self.clients==8 else 4) and released['clients']==8,'All independent services ready before first dispatch')
            def take(index):
                item=take_next(self.queue_path,self.stopped,process_index=self.process_index,client_index=index,clients=self.clients)
                if item is not None:
                    c.save(self.cell_folder/'assignments'/f'{item[1]["queue_index"]:02d}.json',item[1])
                return item
            def play(index,spec,assignment):
                c.require(spec.get('purpose')=='excluded_parallel_throughput_engineering_only', 'Engineering-only setup')
                destination=self.cell_folder/'games'/spec['game_id']
                result=_legacy._play_game(self.plan,self.runtime,self.adapters[index],spec,destination,self.stopped,self.boundary)
                c.save(destination/'outcome.json',result)
                return result
            result=run_clients(self.clients,take,play,self.stopped,self.boundary,self.runtime.abort)
            c.save(self.folder/'session_result.json',result)
            return result
        finally:
            for adapter in self.adapters.values():
                adapter.close()


def assemble_cell(plan,spec,folder,*,started,gameplay_end,boundary=lambda:None):
    """Actual accepted manifest, per-game loader and complete target-record sink."""
    folder=Path(folder);selected=spec['specifications']
    c.require(len(selected)==16 and len({x['game_id'] for x in selected})==16,'Whole fixed engineering cell')
    c.require(not (folder/'stop.json').exists(),'Stopped cell cannot produce target acceptance')
    outcomes=[c.read(folder/'games'/x['game_id']/'outcome.json') for x in selected]
    c.require(all(x['status']=='completed' for x in outcomes),'No incomplete game labels or replacements')
    from .queueing import assignments
    assigned=assignments(folder/'queue.sqlite')
    c.save(folder/'queue_outcomes.json',dict(assignments=assigned,outcomes={x['game_id']:x for x in outcomes},errors=[],never_started=[]))
    documents=[c.artifact(folder/'games'/x['game_id']/'target_game.json') for x in selected]
    manifest=data.build_manifest(documents,selected,route='cpu',cell_id=spec['cell_id'])
    c.save(folder/'corpus_manifest.json',manifest);targets=[]
    def sink(game,examples):
        targets.append(c.save(folder/'targets'/(game['game_id']+'.json'),dict(schema='gpu_collector_examples_v1',
            **data.IDENTITY,spec=game,examples=examples,official_terminal_only=True,teacher_simulations=512,training_updates=0)))
    loaded=data.load_corpus(manifest,expected_specs=selected,boundary=boundary,sink=sink)
    c.save(folder/'loader_result.json',loaded)
    c.save(folder/'target_manifest.json',dict(schema='gpu_collector_loaded_targets_v1',**data.IDENTITY,
        corpus=c.artifact(folder/'corpus_manifest.json'),loader=c.artifact(folder/'loader_result.json'),
        targets=targets,count=loaded['eligible_roots'],training_updates=0))
    ended=time.perf_counter();elapsed=ended-started['started_perf_counter']
    gameplay_wall=max(x['finished']['ended_unix'] for x in outcomes)
    summary=dict(status='completed',cell_id=spec['cell_id'],route='cpu',concurrency=8,
        **started,ended_perf_counter=ended,ended_unix=time.time(),wall_seconds=ended-started['started_perf_counter'],
        gameplay_end_perf_counter=gameplay_end,gameplay_end_unix=gameplay_wall,gameplay_seconds=gameplay_end-started['started_perf_counter'],
        target_loader_seconds=ended-gameplay_end,games=16,plies=sum(x['plies'] for x in loaded['games']),
        eligible_roots=loaded['eligible_roots'],games_per_hour=16*3600/elapsed,
        eligible_roots_per_hour=loaded['eligible_roots']*3600/elapsed,full_cohort_completed=True,corpus_produced=False,
        engineering_targets_only=True,excluded_from_every_training_corpus=True,retries=0,replacements=0,training_updates=0)
    c.save(folder/'cohort.json',summary)
    return summary


def run_engineering(plan,runtime,folder,boundary):
    """The two accepted actual complete-game/loader engineering checks."""
    import threading
    folder=Path(folder);specs=plan['engineering_games'];stopped=threading.Event();documents=[];outcomes=[]
    c.require(len(specs)==2,'Exactly two excluded complete-game qualification trajectories')
    warmup_runtime(plan,runtime,folder)
    for index,spec in enumerate(specs):
        boundary();adapter=runtime.make_adapter('engineering_'+str(index),index,512)
        try:
            c.save(folder/'assignments'/f'{index:02d}.json',dict(queue_index=index,game_id=spec['game_id'],
                worker_index=index,client_id=adapter.client_id,slot=index,
                assigned_perf_counter=time.perf_counter(),assigned_unix=time.time()))
            result=_legacy._play_game(plan,runtime,adapter,spec,folder/'games'/spec['game_id'],stopped,boundary)
            c.save(folder/'games'/spec['game_id']/'outcome.json',result)
            c.require(result['status']=='completed','Excluded engineering qualification cannot retry or replace')
            documents.append(c.artifact(folder/'games'/spec['game_id']/'target_game.json'));outcomes.append(result)
        finally:adapter.close()
    manifest=data.build_manifest(documents,specs,route='cpu',cell_id='engineering')
    c.save(folder/'corpus_manifest.json',manifest);targets=[]
    def sink(spec,examples):
        targets.append(c.save(folder/'targets'/(spec['game_id']+'.json'),dict(schema='gpu_collector_examples_v1',
            **data.IDENTITY,spec=spec,examples=examples,official_terminal_only=True,teacher_simulations=512,training_updates=0)))
    loaded=data.load_corpus(manifest,expected_specs=specs,boundary=boundary,sink=sink)
    c.require(loaded['eligible_roots']>=32,'Actual full-game loader includes at least32 retained targets')
    c.save(folder/'loader_result.json',loaded)
    c.save(folder/'cohort.json',dict(ended_perf_counter=time.perf_counter(),ended_unix=time.time()))
    replay=run_replays(plan,dict(specifications=specs),runtime,folder,boundary)
    result=dict(status='passed',games=2,excluded_from_production=True,corpus=c.artifact(folder/'corpus_manifest.json'),
        loader=c.artifact(folder/'loader_result.json'),targets=targets,model_replay=replay,outcomes=outcomes)
    c.save(folder/'engineering_result.json',result)
    return result
