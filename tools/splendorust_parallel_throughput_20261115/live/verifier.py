"""Saved game audit with an independent dictionary transition oracle.

No learned model or native search executes here. Compressed response packets
remain authenticated at their original transport paths and are decoded exactly.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from . import common as c, schedule
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.splendorust_decision_quality_20261027_v2.audit import check_clock

SCHEMA='sixworld512_saved_game_audit_v1'
MAX_JOURNAL_LINE=2*1024**2


def read_bound(item):
    c.verify([item])
    return c.read(c.resolve(item))


def read_decision_source(item):
    from .adapters import unpack_result
    path=c.resolve(item)
    c.require(path.is_relative_to(c.OUT.resolve()),'Response source stays in this separate attempt')
    raw=read_bound(item)
    c.require(type(raw) is dict and 'result' in raw,'Original trusted transport clock')
    decoded=deepcopy(raw)
    if raw['result'] is not None:decoded['result']=unpack_result(raw['result'])
    check_clock(decoded,limit=120.)
    return decoded


def classify_deadline(clock,*,timeout_evidence=None,source=None):
    from .adapters import INTERNAL_DEADLINE_MARKER
    check_clock(clock,limit=120.)
    c.require(clock['status']!='ok' and clock['result'] is None,'A rejected model response is not a timeout loss')
    if clock['status']=='timeout':
        c.require(clock['elapsed_seconds']>=120 and not clock['received_before_deadline'] and
                  (clock['receipt_elapsed_seconds'] is None or clock['receipt_elapsed_seconds']>120),
                  'Genuine full external research-clock timeout')
        return 'external_120_second_guard'
    if clock['status']=='worker_exit' and timeout_evidence is not None:
        c.require(source is not None and clock['elapsed_seconds']>=120,'Watchdog exit must consume the real allowance')
        proof=read_bound(timeout_evidence)
        initialization=c.read(c.resolve(source).parent/'initialization.json')['result']
        c.require(proof['request_id']==clock['request_id'] and proof['identity']==initialization['identity'] and
                  proof['freeze_sha256']==initialization['freeze_sha256']==c.sha(c.PLAN) and
                  proof['reason']=='sixworld512_decision_watchdog_120s' and proof['deadline_seconds']==120 and
                  type(proof['elapsed_seconds']) in (int,float) and proof['elapsed_seconds']>=120,
                  'Exact service/request/freeze and elapsed durable watchdog evidence')
        return 'service_120_second_watchdog'
    detail=str(clock.get('detail',''))
    c.require(clock['status']=='worker_error' and INTERNAL_DEADLINE_MARKER in detail and clock['elapsed_seconds']>=119 and
              not any(x in detail.lower() for x in ('source changed','ruleset','catalogue','resource','prerequisite')),
              'Only an explicit internal119 deadline is a loss; other failures stop incomplete')
    return 'internal_119_second_guard'


def journal_rows(item):
    """One fsynced complete gzip member per event; partial tails fail closed."""
    c.verify([item]);previous=None;count=0
    with gzip.open(c.resolve(item),'rb') as stream:
        while True:
            raw=stream.readline(MAX_JOURNAL_LINE+1)
            if not raw:break
            c.require(len(raw)<=MAX_JOURNAL_LINE and raw.endswith(b'\n'),'Bounded complete journal record')
            # c.read rejects nonfinite and duplicate keys; apply the same strictness in a stream.
            def unique(pairs):
                out={}
                for key,value in pairs:
                    c.require(key not in out,'Duplicate journal field');out[key]=value
                return out
            row=json.loads(raw,object_pairs_hook=unique,
                parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite journal field')))
            c.require(set(row)=={'sequence','previous_sha256','kind','data'} and
                      type(row['sequence']) is int and row['sequence']==count and row['previous_sha256']==previous,
                      'Ordered unbroken durable journal chain')
            previous=c.digest(row);count+=1
            c.require(count<=6001,'Finite game journal')
            yield row


def required_event(events):
    try:return next(events)
    except StopIteration as error:raise ValueError('Incomplete durable decision sequence') from error


def initial_for(spec,plan,initials=None):
    schedule.validate_spec(spec)
    initials=read_bound(plan['initials']) if initials is None else initials
    c.require(initials['schema']=='parallel_throughput_live_initials_v1' and
              initials['domain']==schedule.DOMAIN,'Frozen corrected initial setup inventory')
    item=initials['snapshots'][f"{spec['stage']}:{spec['pair_index']}"]
    snapshot=read_bound(item['artifact']);decode_training_snapshot(snapshot)
    c.require(item['seed_decimal']==spec['seed_decimal'] and c.digest(snapshot)==item['snapshot_sha256'],
              'Frozen seed and complete initial snapshot binding')
    # Reconstruct only the initial environment; every played step uses the independent ledger.
    rebuilt=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
    c.require(rebuilt==snapshot,'Fresh seed reproduces the entire initial setup')
    return snapshot,item['artifact']


def verify_game(folder,plan,spec,*,boundary=lambda:None,initials=None):
    from .qualification import validate_live_result
    folder=Path(folder);schedule.validate_spec(spec)
    ending=c.read(folder/'result.json')
    c.require(ending['spec']==spec and ending['complete'] is True,'Complete fixed game identity')
    controller=c.read(folder.parent.parent/'controller.json')
    c.require(controller['plan_sha256']==c.digest(plan),'Original game controller binds this exact freeze')
    expected_initial,item=initial_for(spec,plan,initials)
    c.require(ending['initial']==item,'Ending binds original frozen setup')
    events=iter(journal_rows(ending['journal']));current=expected_initial
    failure=None;count=0;last_action=None;seen_sources=set();timings={3:[],6:[]};candidates=[]
    while True:
        boundary()
        try:dispatch=next(events)
        except StopIteration:break
        c.require(count<2000 and failure is None and not current['state']['terminal'],
                  'No decision after terminal, forfeit or game safety cap')
        c.require(dispatch['kind']=='dispatch','Every decision starts with durable dispatch')
        d=dispatch['data'];seat=current['state']['current_player'];pool=3
        c.require(d['spec']==spec and type(d['ply']) is int and d['ply']==count and
                  d['seat']==seat and d['pool']==pool and d['before']==current and
                  d['deadline_seconds']==120 and d['attempt']==1,'Exact contiguous game dispatch')
        c.require(d['identity']==controller['identity'],'Original dispatch belongs to the recorded controller')
        c.require(type(d['dispatched_monotonic_ns']) is int and d['dispatched_monotonic_ns']>0,
                  'Original dispatch clock retained')
        ledger.check_observer(current,d['observer'])
        response=required_event(events);r=response['data']
        c.require(response['kind']=='response' and r['ply']==count and
                  r['dispatch_sha256']==c.digest(dispatch),'Raw response follows its exact dispatch')
        source=r['source'];c.require(source['path'] not in seen_sources,'A saved response cannot play twice')
        seen_sources.add(source['path']);clock=read_decision_source(source)
        transition=required_event(events);t=transition['data']
        c.require(transition['kind']=='transition' and t['ply']==count and t['source']==source,
                  'Transition binds the persisted response')
        if clock['status']=='ok':
            validate_live_result(clock['result'],d['observer'],pool,plan)
            action=clock['result']['action'];c.require(t['completed'] is True and t['action']==action and t['failure'] is None,
                  'Completed action is exactly the accepted scientific response')
            after=ledger.transition(current,action)
            c.require(t['after']==after,'Independent dictionary transition matches saved successor')
            decode_training_snapshot(after)
            c.require(after['state']['terminal'] or (after!=current and not
                (action['action_type']=='PASS' and last_action=='PASS')),
                'No invented outcome for an actual deadlock')
            current=after;last_action=action['action_type'];timings[pool].append(clock['receipt_elapsed_seconds'])
            candidates.append(dict(spec=spec,ply=count,pool=pool,source=source,
                observer=d['observer'],result_sha256=c.digest(clock['result'])))
        else:
            failure=classify_deadline(clock,timeout_evidence=r.get('timeout_evidence'),source=source)
            c.require(t['completed'] is False and t['action'] is None and t['after'] is None and t['failure']==failure,
                      'Failed decision never receives an invented action/successor')
            failure=dict(kind=failure,seat=seat,pool=pool,source=source)
        count+=1
    c.require(0<count<=2000 and type(ending['plies']) is int and ending['plies']==count and ending['final']==current,
              'Exact terminal inventory and state')
    c.require(ending['winner'] is None or (type(ending['winner']) is int and ending['winner'] in (0,1)),
              'Strict winner seat or official draw')
    if failure is None:
        c.require(current['state']['terminal'] and ending['reason']=='natural' and
                  ending['winner']==current['state']['winner'] and ending['failure'] is None,
                  'Official natural outcome from the independent referee')
    else:
        c.require(not current['state']['terminal'] and ending['reason']=='forfeit' and
                  ending['winner']==1-failure['seat'] and ending['failure']==failure,
                  'Only a genuine research deadline produces the recorded forfeit')
    points2=1 if ending['winner'] is None else 2 if ending['winner']==spec['candidate_seat'] else 0
    c.require(type(ending['candidate_points2']) is int and ending['candidate_points2']==points2,'Exact candidate score credit')
    return dict(status='passed',spec=spec,game=c.artifact(folder/'result.json'),plies=count,
        candidate_points2=points2,reason=ending['reason'],failed_decisions=int(failure is not None),
        timings={str(k):v for k,v in timings.items()},replay_candidates=candidates,decision_sources=sorted(seen_sources),
        independent_transition_oracle='corrected dictionary ledger; no env.step during replay')


def verify_replays(selection,replayed,plan,*,boundary=lambda:None):
    from .qualification import validate_live_result
    c.require(type(replayed) is list and len(replayed)==len(selection),'Exact replay inventory')
    checks=[];sources=set()
    for expected,observed in zip(selection,replayed,strict=True):
        boundary();c.require(observed['replay_id']==expected['replay_id'],'Prospectively ordered exact replay')
        source=observed['source'];c.require(source['path'] not in sources and source!=expected['source'],
                    'Fresh distinct replay request source');sources.add(source['path'])
        original=read_decision_source(expected['source']);reproduced=read_decision_source(source)
        c.require(original['status']==reproduced['status']=='ok','Successful original and replay clocks')
        validate_live_result(reproduced['result'],expected['observer'],expected['pool'],plan)
        c.require(c.digest(original['result'])==expected['result_sha256'] and
                  c.canonical(original['result'])==c.canonical(reproduced['result']),
                  'Complete scientific result reproduced exactly, including actions/callbacks/world samples')
        checks.append(dict(replay_id=expected['replay_id'],original=expected['source'],reproduced=source,exact=True))
    return dict(status='passed',repetitions=len(checks),checks=checks,outer_clocks_excluded=True,
                scientific_fields_omitted=False)
