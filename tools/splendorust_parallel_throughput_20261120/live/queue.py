"""Exclusive once-only queue leases; a failed or interrupted game is never retried."""
from pathlib import Path
from . import common as c,schedule

def prepare(folder,specs,arm,phase):
    c.require(arm==schedule.phase_arm(phase) and specs==schedule.phase_specs(phase),'Frozen four-phase order and setups')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    return c.save(folder/'queue.json',dict(schema='parallel_live_queue_v1',arm=arm,phase=phase,specs=specs))

def load(item):
    c.verify([item]);value=c.read(c.resolve(item))
    c.require(set(value)=={'schema','arm','phase','specs'} and value['schema']=='parallel_live_queue_v1','Exact shared queue')
    c.require(value['arm']==schedule.phase_arm(value['phase']) and value['specs']==schedule.phase_specs(value['phase']),
              'Fixed original queue inventory')
    return value

def check_running(item):
    c.require(not (c.resolve(item).parent/'abort.json').exists(),'A lane failed: no further dispatch permitted')

def lease(item,lane):
    value=load(item);check_running(item)
    lanes=1 if value['arm']=='A1' else 4
    c.require(type(lane) is int and 0<=lane<lanes,'Declared process lane')
    folder=c.resolve(item).parent
    for spec in value['specs']:
        check_running(item)
        try:
            claim=c.save(folder/'leases'/f"g{spec['game_index']:02d}.json",dict(
                spec=spec,lane=lane,identity=c.identity(),queue=item,time=c.now()))
        except FileExistsError:continue
        # A racing failure can leave a preserved lease, never an unrecorded retry.
        check_running(item)
        return spec,claim
    return None

def fail(item,lane,error):
    path=c.resolve(item).parent/'abort.json'
    try:return c.save(path,dict(lane=lane,error=error,identity=c.identity(),no_retry=True,time=c.now()))
    except FileExistsError:return c.artifact(path)

def verify(item,lanes):
    value=load(item);check_running(item);folder=c.resolve(item).parent
    expected={s['game_index']:s for s in value['specs']};seen={}
    for lane in lanes:
        for row in lane['games']:
            claim=row['lease'];c.verify([claim]);data=c.read(c.resolve(claim));index=data['spec']['game_index']
            c.require(data['queue']==item and data['identity']==lane['identity'] and data['lane']==lane['lane'],
                      'Exact lane identity owns its once-only claim')
            c.require(index in expected and index not in seen and data['spec']==expected[index],
                      'No missing, extra or duplicate queue setup')
            c.require(c.resolve(claim)==folder/'leases'/f'g{index:02d}.json','Canonical queue claim path')
            seen[index]=row
    c.require(set(seen)==set(expected) and {p.name for p in (folder/'leases').glob('*.json')}==
              {f'g{i:02d}.json' for i in expected},'Complete queue and claim inventory')
    return [seen[i] for i in sorted(seen)]
