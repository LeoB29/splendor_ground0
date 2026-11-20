"""Owned lane runtime; unchanged scientific service and independent saved audits."""
from pathlib import Path
import time,traceback
from . import common as c,queue

def selected_replays(games):
    """Two hash-fixed completed moves per game, covering early/late if present."""
    selected=[]
    for game in games:
        rows=game['replay_candidates']
        key=lambda r:c.digest(['parallel_live_replay_v1',r['spec']['game_index'],r['ply']])
        early=sorted((r for r in rows if r['ply']<24),key=key)
        late=sorted((r for r in rows if r['ply']>=24),key=key)
        choices=([early[0]] if early else [])+([late[0]] if late else [])
        if len(choices)<2:choices+=sorted((r for r in rows if r not in choices),key=key)[:2-len(choices)]
        c.require(len(choices)==min(2,len(rows)) and bool(choices),'Every game supplies fixed exact replay')
        selected.extend(dict(r,replay_id='replay_'+key(r)[:24]) for r in choices)
    return selected

def run(folder,plan,owner,boundary,request):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from .. import topology
    import psutil
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    sources=verify_runtime_sources()
    c.require(sources==plan['corrected_source_receipt'],'Actual lane/evaluator corrected source guard')
    c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),**c.IDENTITY))
    topology.verify_startup(plan['topology'],plan['placement'])
    placement=plan['placement']
    c.require(request['model_cpu'] in placement['model_cpus'] and
        request['controller_cpus']==placement['controller_cpus'] and request['original_allowed']==placement['original_allowed'],
        'Lane pins match frozen actual topology')
    psutil.Process().cpu_affinity(request['controller_cpus'])
    # The trusted live transport uses a nested limits object; it receives the
    # exact original owner clocks, never a newly calculated allowance.
    transported_owner=dict(owner,limits={k:owner[k] for k in
        ('monotonic_origin','monotonic_deadline','unix_origin','unix_deadline')})
    return run_lane(folder,plan,transported_owner,boundary,request)

def run_lane(folder,plan,owner,boundary,request,*,service_factory=None):
    """Same runtime used by actual workers and model-free service fixtures."""
    from . import transport,qualification,arena,verifier
    service_factory=transport.Services if service_factory is None else service_factory
    folder=Path(folder);kind=request['kind'];c.require(kind in ('qualification','lane'),'Fixed live worker kind')
    pin={k:request[k] for k in ('model_cpu','controller_cpus','original_allowed')}
    queue_item=request.get('queue');games=[];claims=[];services=[];replay_services=[];replayed=[]
    def checked_boundary():
        boundary()
        if queue_item is not None:queue.check_running(queue_item)
    started=time.monotonic();manager=service_factory(folder/'live',plan,owner,checked_boundary,
        'qualification' if kind=='qualification' else 'main',1,**pin)
    try:
        manager.start()
        if kind=='qualification':
            result=qualification.run_live_qualification(manager.request,plan,service_folder=manager.service_folder,boundary=checked_boundary)
            services=manager.finish()
            proof=transport.verify_services(services,plan,checked_boundary)
            return dict(qualification=c.save(folder/'qualification.json',result),services=services,
                        service_audit=c.save(folder/'services_audit.json',proof))
        c.require(queue_item is not None,'Every live lane uses the shared frozen queue')
        arena_folder=folder/'arena'
        c.save(arena_folder/'controller.json',dict(identity=c.identity(),plan_sha256=c.digest(plan),time=c.now()))
        initials=verifier.read_bound(plan['initials'])
        while True:
            checked_boundary();leased=queue.lease(queue_item,request['lane'])
            if leased is None:break
            spec,claim=leased
            game=arena.play_game(arena_folder/'games'/f"g{spec['game_index']:04d}",plan,spec,manager.decision,
                checked_boundary,initials=initials)
            games.append(game);claims.append(claim)
            c.require(game['failed_decisions']==0,'Preserved deadline loss fails engineering qualification')
            c.status(folder/'progress.json',dict(completed_games=len(games),last_game=spec['game_index'],time=c.now()))
        services=manager.finish()
        # One fresh inference service per lane checks selected original-slot moves.
        selection=selected_replays(games)
        if selection:
            replay=service_factory(folder/'replay',plan,owner,checked_boundary,'replay',1,**pin)
            try:
                replay.start()
                for row in selection:
                    checked_boundary();value=replay.decision(row['observer'],3,context=dict(replay_id=row['replay_id']))
                    saved=dict(replay_id=row['replay_id'],source=value['source']);replayed.append(saved)
                    c.save(folder/'replay_rows'/f"{row['replay_id']}.json",saved)
                    c.require(value['clock']['status']=='ok','Fresh replay failed; no retry')
                replay_services=replay.finish()
            finally:
                if replay.child is not None:replay.close()
        proof=verifier.verify_replays(selection,replayed,plan,boundary=checked_boundary)
        service_proof=transport.verify_services(services+replay_services,plan,checked_boundary)
        rows=[dict(lease=claim,game=g['game'],spec=g['spec']) for claim,g in zip(claims,games,strict=True)]
        result=dict(status='passed',identity=c.identity(),lane=request['lane'],queue=queue_item,games=rows,
            replay_selection=selection,replayed=replayed,replay_proof=c.save(folder/'replay_proof.json',proof),
            services=services,replay_services=replay_services,service_audit=c.save(folder/'services_audit.json',service_proof),
            elapsed_seconds=time.monotonic()-started)
        result['receipt']=c.save(folder/'lane_receipt.json',result)
        return result
    except BaseException:
        error=traceback.format_exc()
        if queue_item is not None:queue.fail(queue_item,request['lane'],error)
        c.save(folder/'failure.json',dict(error=error,identity=c.identity(),no_retry=True,time=c.now()))
        raise
    finally:
        if manager.child is not None:manager.close()
