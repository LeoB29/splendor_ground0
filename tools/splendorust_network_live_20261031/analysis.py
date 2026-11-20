"""Independent corrected ledger, same-player reproduction and operational timing.

No model, new game simulation, strength interval or promotion is performed.
Raw request/response, process and execution-variant identities are separately
authenticated by the controller and independent saved reader.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import math
import statistics
from tools.external_strength_screen_20261018 import audit as ledger
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.splendorust_search256_recovery_20261030_02 import scientific
from . import common as c,design


def require(value,message):
    if not value:raise ValueError('NETWORK_LIVE_ANALYSIS: '+message)


def same(a,b):return c.canonical(a)==c.canonical(b)


def validate_completed(row):
    received=scientific.check_clock(row['clock'],limit=10.)
    require(row.get('completed') is True and row['clock']['status']=='ok' and
        row['clock']['deadline_seconds']==10. and received is not None,'complete original research response')
    result=row['clock']['result']
    require(same(row.get('action'),result['action']) and same(row.get('trace'),result['diagnostics']),
        'saved action/trace binds actual response')
    ledger.check_observer(row['before'],row['observer'])
    scientific.validate_result(result,row['observer'],'search',expected_quota=256)
    scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',before=row['before'],after=row['after'],expected_quota=256)
    require(same(ledger.transition(row['before'],row['action']),row['after']),'independent corrected ledger successor')
    decode_training_snapshot(row['after'])
    return received


def audit_game(initial,rows,ending,spec,*,freeze=None,boundary=lambda:None):
    """Authenticate every move and official ending independently of env.step."""
    spec=design.validate_spec(spec)
    require(same(initial,design.initial_snapshot(spec)),'exact corrected seed-bound initial state')
    require(type(rows) is list and 0<len(rows)<=c.MAX_PLIES,'bounded nonempty game rows')
    require(type(ending) is dict and all(k in ending and same(ending[k],v) for k,v in spec.items()),'ending specification')
    require(ending.get('complete') is True and type(ending.get('plies')) is int and ending['plies']==len(rows),
        'exact complete game move count')
    current=deepcopy(initial);decode_training_snapshot(current);failure=None;transitions=0
    latency={v:[] for v in c.VARIANTS}
    for ply,row in enumerate(rows):
        boundary();require(failure is None and not current['state']['terminal'],'no move after terminal/failure')
        require(design.validate_spec(row)==spec and type(row.get('ply')) is int and row['ply']==ply and
            same(row.get('before'),current),'contiguous full game identity and state')
        seat=current['state']['current_player'];role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        variant=spec[role+'_variant']
        require(type(row.get('seat')) is int and row['seat']==seat and row.get('player_role')==role and
            row.get('variant')==variant and row.get('arm')=='search' and row.get('quota')==256 and
            row.get('route')==row.get('mode')=='research','actual acting seat/variant/scientific identity')
        require(row.get('placement_mode')=='isolated' and not any(k in row for k in
            ('transition_error','placement_error','profile_capture_error')),'no hidden infrastructure failure')
        ledger.check_observer(current,row['observer'])
        require(row['clock']['deadline_seconds']==10.,'unshortened research deadline')
        if row['clock']['status']=='ok':
            elapsed=validate_completed(row);current=row['after'];latency[variant].append(elapsed);transitions+=1
            if not current['state']['terminal']:
                require(not same(row['before'],row['after']),'unresolved unchanged-state deadlock')
                require(not(ply and row['action']['action_type']=='PASS' and rows[ply-1]['action'] and
                    rows[ply-1]['action']['action_type']=='PASS'),'unresolved consecutive-pass deadlock')
        else:
            require(row.get('completed') is False and row.get('action') is None and row.get('trace') is None and
                row.get('after') is None,'failed request cannot invent a move')
            failure=dict(seat=seat,role=role,variant=variant,status=scientific.classify_failure(row['clock']))
    boundary();require(same(ending.get('final'),current),'final snapshot binds ledger')
    if failure is None:
        require(ending.get('reason')=='natural' and current['state']['terminal'] and
            same(ending.get('winner'),current['state']['winner']) and ending.get('loser') is None and
            ending.get('failed_role') is None and ending.get('failed_seat') is None and
            ending.get('failure') is None and ending.get('failure_policy') is None and
            ending.get('administrative_forfeit',False) is False,'true natural terminal and winner')
    else:
        require(ending.get('reason')=='forfeit' and ending.get('administrative_forfeit') is True and
            ending.get('failure_policy')=='research_hard_guard_9s_internal_10s_complete_response' and
            not current['state']['terminal'] and type(ending.get('winner')) is int and ending['winner']==1-failure['seat'] and
            type(ending.get('loser')) is int and ending['loser']==failure['seat'] and
            ending.get('failed_role')==failure['role'] and ending.get('failed_seat')==failure['seat'],
            'explicit hard-guard forfeit attribution')
        detail=ending.get('failure')
        require(detail==failure['status'] or type(detail) is dict and detail.get('status')==failure['status'],
            'exact preserved failure classification')
    return dict(status='passed',game_index=spec['game_index'],transitions=transitions,failed_moves=int(failure is not None),
        final_sha256=c.digest(current),latencies=latency,natural_terminal=failure is None,
        transition_oracle='independent corrected dictionary ledger, never env.step',
        trust_boundary='shared pinned corrected catalogue; raw transport/variant identity and process closure checked separately')


def verify_replay(original,replayed,selection):
    """A fresh worker must reproduce its own variant exactly, not its opponent."""
    require(original.get('completed') is True,'only completed played moves can be reproduced')
    expected={k:original[k] for k in ('game_index','pair_index','ply','variant','player_role','seat')}
    require(all(selection.get(k)==v for k,v in expected.items()) and
        selection.get('observer_sha256')==c.digest(original['observer']),'source selection binding')
    require(replayed.get('variant')==original['variant'] and replayed.get('route')=='research',
        'reproduction same fixed execution variant')
    for key in ('observer','before','action','trace','after'):
        require(same(original[key],replayed.get(key)),'exact same-variant replay '+key)
    require(same(original['clock']['result'],replayed['clock']['result']),'full scientific replay output equality')
    validate_completed(original);validate_completed(replayed)
    seed=int(c.digest(original['observer'])[:16],16)
    require(original['trace']['observer_seed']==replayed['trace']['observer_seed']==seed,'unchanged observer-derived world/search seed')
    return dict(status='passed',game_index=original['game_index'],ply=original['ply'],variant=original['variant'],
        observer_sha256=c.digest(original['observer']),result_sha256=c.digest(original['clock']['result']),
        full_scientific_output_exact=True,cross_variant_equality_required=False)


def distribution(values):
    values=sorted(values)
    require(all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=10 for x in values),'valid completed response times')
    if not values:return dict(count=0,minimum=None,maximum=None,mean=None,median=None,p95=None,
        completed_over_700ms=0,completed_over_900ms=0,completed_over_one_second=0)
    q=(len(values)-1)*.95;lo=math.floor(q);hi=math.ceil(q)
    return dict(count=len(values),minimum=values[0],maximum=values[-1],mean=statistics.fmean(values),median=statistics.median(values),
        p95=values[lo]+(values[hi]-values[lo])*(q-lo),completed_over_700ms=sum(x>.7 for x in values),
        completed_over_900ms=sum(x>.9 for x in values),completed_over_one_second=sum(x>1 for x in values))


def summarize_games(games,rows,schedule=None):
    schedule=design.make_schedule() if schedule is None else design.validate_schedule(schedule)
    require(type(games) is list and len(games)<=6,'fixed complete-game prefix')
    records=list(rows);by_game=defaultdict(list)
    for row in records:design.validate_spec(row);by_game[row['game_index']].append(row)
    require(set(by_game)=={g['game_index'] for g in games},'no hidden or missing game decision inventory')
    proofs=[];wins=Counter();timing={v:[] for v in c.VARIANTS};failures=[];by_game_timing=[]
    for index,game in enumerate(games):
        spec=design.validate_spec(game);require(spec==schedule[index],'fixed original game order')
        if index:require(games[index-1]['reason']=='natural','no continuation after operational failure')
        selected=by_game[index];proof=audit_game(design.initial_snapshot(spec),selected,game,spec)
        proofs.append(proof)
        winner=game['winner'];require(winner is None or type(winner) is int and winner in (0,1),'winner encoding')
        winner_variant='draw' if winner is None else 'candidate' if winner==spec['challenger_seat'] else 'control'
        wins[winner_variant]+=1
        if game['reason']=='forfeit':failures.append(dict(game_index=index,failed_role=game['failed_role'],failed_seat=game['failed_seat'],failure=game['failure']))
        for variant in c.VARIANTS:timing[variant].extend(proof['latencies'][variant])
        by_game_timing.append(dict(game_index=index,timings={v:distribution(proof['latencies'][v]) for v in c.VARIANTS}))
    complete=len(games)==6;passed=complete and not failures
    return dict(schema='network_live_operational_summary_v1',status='operational_games_passed' if passed else 'verified_stopped_prefix',
        complete_planned_games=complete,operational_games_passed=passed,planned_games=6,completed_games=len(games),
        completed_pairs=len(games)//2,decisions=len(records),completed_decisions=sum(len(x) for x in timing.values()),
        natural_games=sum(g['reason']=='natural' for g in games),administrative_forfeits=len(failures),failures=failures,
        descriptive_wins={v:wins[v] for v in ('control','candidate','draw')},
        timings={v:distribution(timing[v]) for v in c.VARIANTS},by_game=by_game_timing,game_proofs=proofs,
        strength_test=False,strength_equivalence_established=False,reference_promoted=False,
        one_second_is_descriptive=True,seven_hundred_ms_is_descriptive=True,
        limits=['Three paired setups are operational qualification, not a strength test.',
            'Live states can diverge; unpaired aggregate variant times do not isolate an implementation speed ratio.',
            'Late completed responses are retained under the unchanged9/10-second research guards.',
            'Model reproductions, raw transport, source preservation and actual owner exits remain separate required checks.'])
