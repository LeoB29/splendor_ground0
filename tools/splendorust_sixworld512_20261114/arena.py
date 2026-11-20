"""Serial corrected-rule games with durable pre-dispatch custody.

One complete gzip member is fsynced for each journal event. An interrupted
decision remains recorded as dispatched and can never be silently replayed.
Full model packets live once in the immutable compressed transport source.
"""
from pathlib import Path
import gzip
import os
import time
import traceback
from . import common as c, schedule, verifier
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.future_training_rules_20261021 import decode_training_snapshot


def output_path(path):
    path=Path(path).resolve()
    c.require(path.is_relative_to(c.OUT.resolve()),'All new game evidence stays in the separate attempt')
    return path


class Journal:
    def __init__(self,path):
        self.path=output_path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('xb');self.sequence=0;self.previous=None
    def append(self,kind,data):
        row=dict(sequence=self.sequence,previous_sha256=self.previous,kind=kind,data=data)
        raw=(c.canonical(row)+'\n').encode()
        c.require(len(raw)<=verifier.MAX_JOURNAL_LINE,'Finite durable game event')
        self.stream.write(gzip.compress(raw,compresslevel=3,mtime=0));self.stream.flush();os.fsync(self.stream.fileno())
        self.sequence+=1;self.previous=c.digest(row)
        return row
    def close(self):self.stream.close()


def play_game(folder,plan,spec,call_decision,boundary,*,initials=None):
    from .qualification import validate_live_result
    folder=output_path(folder);schedule.validate_spec(spec);folder.mkdir(parents=True,exist_ok=False)
    before,initial_item=verifier.initial_for(spec,plan,initials)
    state=decode_training_snapshot(before);journal=Journal(folder/'events.jsonl.gz')
    failure=None;reason=None;winner=None;last_action=None;started=time.monotonic();plies=0
    try:
        for ply in range(2000):
            boundary()
            if state.terminal:reason='natural';winner=state.winner;break
            before=rules.serialize_snapshot(state);decode_training_snapshot(before)
            observer=rules.observation_payload(state);ledger.check_observer(before,observer)
            seat=state.current_player;pool=6 if seat==spec['candidate_seat'] else 3
            dispatch=journal.append('dispatch',dict(spec=spec,ply=ply,seat=seat,pool=pool,before=before,
                observer=observer,deadline_seconds=120,attempt=1,time=c.now(),
                dispatched_monotonic_ns=time.perf_counter_ns(),identity=c.identity()))
            context=dict(stage=spec['stage'],pair_index=spec['pair_index'],game_index=spec['game_index'],ply=ply)
            packet=call_decision(observer,pool,context=context)
            # The immutable raw packet exists before any model/output validation.
            source=packet['source']
            journal.append('response',dict(ply=ply,source=source,dispatch_sha256=c.digest(dispatch),
                timeout_evidence=packet.get('timeout_evidence'),time=c.now()))
            clock=verifier.read_decision_source(source)
            c.require(c.canonical(packet['clock'])==c.canonical(clock),'Returned clock matches its original persisted compressed source')
            if clock['status']=='ok':
                validate_live_result(clock['result'],observer,pool,plan)
                action=clock['result']['action'];decoded=rules.deserialize_action(action)
                legal=rules.CorrectedEnv().legal_actions(state)
                c.require(decoded in legal,'Accepted completed action belongs to corrected legal support')
                after=rules.serialize_snapshot(rules.CorrectedEnv().step(state,decoded))
                journal.append('transition',dict(ply=ply,source=source,completed=True,action=action,after=after,failure=None))
                ledger.check_step(before,action,after);decode_training_snapshot(after)
                c.require(after['state']['terminal'] or (before!=after and not
                          (action['action_type']=='PASS' and last_action=='PASS')),
                          'Unresolved actual game deadlock has no invented result')
                state=decode_training_snapshot(after);last_action=action['action_type'];plies=ply+1
                if state.terminal:reason='natural';winner=state.winner;break
            else:
                kind=verifier.classify_deadline(clock,timeout_evidence=packet.get('timeout_evidence'),source=source)
                failure=dict(kind=kind,seat=seat,pool=pool,source=source)
                journal.append('transition',dict(ply=ply,source=source,completed=False,action=None,after=None,failure=kind))
                c.save(folder/'deadline_failure.json',dict(spec=spec,ply=ply,failure=failure,no_retry=True,time=c.now()))
                reason='forfeit';winner=1-seat;plies=ply+1;break
        c.require(reason in ('natural','forfeit'),'2,000-ply safety cap: game incomplete')
    except BaseException:
        error=traceback.format_exc()
        c.save(folder/'failure.json',dict(status='technical_incomplete',spec=spec,error=error,time=c.now(),no_retry=True))
        raise
    finally:journal.close()
    points2=1 if winner is None else 2 if winner==spec['candidate_seat'] else 0
    ending=dict(schema='sixworld512_game_result_v1',spec=spec,initial=initial_item,journal=c.artifact(journal.path),
        complete=True,reason=reason,winner=winner,failure=failure,plies=plies,candidate_points2=points2,
        final=rules.serialize_snapshot(state),elapsed_seconds=time.monotonic()-started,time=c.now())
    c.save(folder/'result.json',ending)
    return verifier.verify_game(folder,plan,spec,boundary=boundary,initials=initials)


def run_block(folder,plan,specs,call_decision,boundary):
    folder=output_path(folder);stage=schedule.validate_block(specs)
    folder.mkdir(parents=True,exist_ok=True)
    c.require(not (folder/'result.json').exists() and not (folder/'games').exists(),'Exclusive block, no repeated or replaced games')
    c.save(folder/'controller.json',dict(identity=c.identity(),plan_sha256=c.digest(plan),time=c.now()))
    initials=verifier.read_bound(plan['initials']);games=[];started=time.monotonic()
    for spec in specs:
        boundary()
        game=play_game(folder/'games'/f"g{spec['game_index']:04d}",plan,spec,call_decision,boundary,initials=initials)
        games.append({k:v for k,v in game.items() if k not in ('replay_candidates','decision_sources')})
        c.status(folder/'progress.json',dict(stage=stage,games_completed=len(games),games_total=len(specs),
            last_game_index=spec['game_index'],elapsed_seconds=time.monotonic()-started,time=c.now()))
        if stage=='operational':c.require(game['failed_decisions']==0,'Operational deadline failure retained; qualification stops')
    # Independent pass rereads the saved immutable transport/journal and transitions.
    audit=verifier.verify_block(folder,plan,specs,boundary=boundary)
    result=dict(schema='sixworld512_block_result_v1',status='passed',stage=stage,
        games=[g['game'] for g in audit['games']],pair_scores=audit['pair_scores'],
        replay_selection=audit['replay_selection'],failed_decisions=audit['failed_decisions'],
        verified_game_audit=c.save_gzip(folder/'saved_audit.json.gz',audit),
        elapsed_seconds=time.monotonic()-started,time=c.now())
    item=c.save(folder/'result.json',result)
    return dict(result=result,artifact=item)
