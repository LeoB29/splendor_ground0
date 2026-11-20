"""Full saved-state audit and exact cross-arm scientific trajectories."""
from pathlib import Path
import struct
from . import common as c,verifier,transport

def execution_payload(item,request):
    c.verify([item]);execution=c.read(c.resolve(item))
    c.require(execution['status']=='passed' and execution['exit_code']==0 and not execution['error'] and
      not execution['cleanup_errors'] and execution['request']==request and execution['retries']==0,
      'Actual successful unchanged owned execution')
    c.require(all(not c.exact_alive(i) for i in [execution['identity'],*execution['observed_owned_identities']]),
              'Every exact worker and descendant has exited')
    c.verify([execution['result']]);result=c.read(c.resolve(execution['result']))
    c.require(result['identity']==execution['identity'] and result['status']=='passed' and result['error'] is None
      and result['request']==request and result['module']==execution['module']==c.PACKAGE+'.worker',
      'Worker result binds exact executed role and process')
    return result['payload']

def verify_lane(result,plan,boundary=lambda:None):
    c.verify([result['receipt']]);saved=c.read(c.resolve(result['receipt']))
    c.require(saved=={k:v for k,v in result.items() if k!='receipt'},'Exact complete lane receipt')
    folder=c.resolve(result['receipt']).parent;games=[];sources=set()
    game_dir=folder/'arena/games'
    expected={f"g{row['spec']['game_index']:04d}" for row in result['games']}
    actual={p.name for p in game_dir.iterdir() if p.is_dir()} if game_dir.exists() else set()
    c.require(actual==expected,'Exact lane game-directory inventory; no extra or missing game')
    for row in result['games']:
        boundary();c.verify([row['game'],row['lease']])
        c.require(c.resolve(row['game'])==game_dir/f"g{row['spec']['game_index']:04d}"/'result.json',
                  'Every game result belongs to its exact lane and setup')
        game=verifier.verify_game(c.resolve(row['game']).parent,plan,row['spec'],boundary=boundary)
        c.require(game['failed_decisions']==0,'No engineering deadline loss can pass')
        c.require(not sources.intersection(game['decision_sources']),'Raw response cannot be reused across lane games')
        sources.update(game['decision_sources'])
        games.append(game)
    from .worker import selected_replays
    c.require(result['replay_selection']==selected_replays(games),'Reconstructed prospective replay selection')
    replay=verifier.verify_replays(result['replay_selection'],result['replayed'],plan,boundary=boundary)
    c.require(replay==verifier.read_bound(result['replay_proof']),'Saved exact replay proof')
    services=transport.verify_services(result['services']+result['replay_services'],plan,boundary)
    c.require(services==verifier.read_bound(result['service_audit']),'Full wire/forensic/actual-exit audit unchanged')
    return dict(status='passed',games=len(games),decisions=sum(g['plies'] for g in games),
        timings=[t for g in games for t in g['timings']['3']],replays=replay['repetitions'],
        game_artifacts=[g['game'] for g in games],decision_sources=sorted(sources))

def science(result):
    """Full packet plus explicit root scalar bytes; nothing scientific omitted."""
    phases=result['research']['phases']
    roots=[]
    for phase in phases:
        telemetry=phase.get('telemetry')
        if telemetry is None:continue
        roots.append(dict(visits=telemetry['visits'],root_value_f64_hex=struct.pack('!d',telemetry['root_value']).hex()))
    return dict(action=result['action'],roots=roots,complete_result_sha256=c.digest(result))

def trajectory(game_item):
    ending=verifier.read_bound(game_item);rows=[]
    for event in verifier.journal_rows(ending['journal']):
        if event['kind']!='response':continue
        clock=verifier.read_decision_source(event['data']['source'])
        c.require(clock['status']=='ok','Failed move cannot be an exact accepted trajectory')
        rows.append(science(clock['result']))
    c.require(len(rows)==ending['plies'],'Every completed game decision participates in identity')
    return dict(spec=ending['spec'],initial=ending['initial'],moves=rows,final=ending['final'],winner=ending['winner'],
        reason=ending['reason'],failure=ending['failure'])

def compare_subset(a1,a4,expected_indices,boundary=lambda:None):
    expected=list(expected_indices)
    c.require(expected in (list(range(8)),list(range(16))) and len(a1)==len(a4)==len(expected),
              'Only the prospective first matched eight or complete sixteen')
    def indexed(items):
        out={}
        for item in items:
            data=verifier.read_bound(item);index=data['spec']['game_index']
            c.require(index not in out,'One game per setup per arm');out[index]=item
        c.require(set(out)==set(expected),'All prospective live setups, no replacements')
        return out
    left,right=indexed(a1),indexed(a4);decisions=0;proof=[]
    for index in expected:
        boundary();x,y=trajectory(left[index]),trajectory(right[index])
        c.require(c.canonical(x)==c.canonical(y),'Cross-arm science/trajectory/terminal mismatch at setup '+str(index))
        decisions+=len(x['moves']);proof.append(dict(setup=index,a1=left[index],a4=right[index],
            exact=True,decisions=len(x['moves']),trajectory_sha256=c.digest(x)))
    return dict(status='passed',setups=len(expected),exact_decisions=decisions,checks=proof,
                actions_root_visits_root_value_bytes_and_full_results=True,terminal_outcomes_exact=True)

def compare_arms(a1,a4,boundary=lambda:None):
    return compare_subset(a1,a4,list(range(16)),boundary)
