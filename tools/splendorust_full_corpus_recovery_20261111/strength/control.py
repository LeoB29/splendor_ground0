"""Bounded per-repetition segments under one retained study clock.

Full transport, independent game replay and exact reproductions reuse the accepted
joint-assessment mechanics. Segment boundaries never restart a move or study clock.
"""
from copy import deepcopy
from pathlib import Path
import os
import time
import traceback
import psutil
from . import common as c, transport, adapters, qualification, scientific, analysis, design, lineage, profile_parity
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import rules, audit as ledger

def bytes_used():
    return sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())


def evidence_seal(paths,envelope):
    artifacts=[];seen=set()
    for root in paths:
        entries=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
        for path in entries:
            envelope.guard();relative=path.relative_to(c.ROOT).as_posix()
            if relative in seen:continue
            seen.add(relative);artifacts.append(c.artifact(path))
    for item in artifacts:envelope.guard();c.verify([item])
    return sorted(artifacts,key=lambda x:x['path'])


def save_or_verify(path,value,writing):
    if writing and not path.exists():return c.save(path,value)
    c.require(c.read(path)==value,'Independent saved result differs: '+str(path));return c.artifact(path)


def load_inputs(freeze,envelope):
    c.verify([freeze[k] for k in ('schedule','qualification_plan','replay_rule','initial_states')])
    schedule=c.read(c.ROOT/freeze['schedule']['path']);design.validate_schedule(schedule)
    c.require(c.read(c.ROOT/freeze['replay_rule']['path'])==design.replay_rule(),'Exact blind reproduction selection rule')
    initials=c.read(c.ROOT/freeze['initial_states']['path'])
    c.require(initials==[dict(**s,initial=design.initial_snapshot(s)) for s in schedule],'Corrected paired initial states')
    plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    c.require(plan==qualification.build_plan(),'Actual same128 corrected entrypoint plan unchanged')
    return schedule,plan


def route_for(name):
    c.require(name in lifecycle_inventory(),'Declared exact lifecycle')
    return lifecycle_inventory()[name]['route']


def qualification_evidence(name,envelope,freeze,plan,*,writing):
    folder=c.OUT/'workers'/name;route=route_for(name)
    graph=transport.verify_graph_evidence(folder,freeze,boundary=envelope.guard)
    clock=c.read(folder/'entrypoint_clock.json');profiles=c.read(folder/'entrypoint_profiles.json')
    transport._require_profiles(profiles)
    transport.validate_profiles(profiles['parent_profile'],profiles['child_profile'],clock,{'operation':'entrypoint_checks'},route,graph_identity_sha256=c.read(folder/'initialization.json')['result']['graph_identity_sha256'])
    suite=qualification.verify_entrypoint(folder,clock,plan,freeze,boundary=envelope.guard)
    historical=qualification.compare_control_parity(clock['result'],plan,boundary=envelope.guard) if c.is_parent_route(route) else None
    first='qualification_'+route
    relocation=profile_parity.compare_suite(
        c.read(c.LIVE_SOURCE_OUT/'workers'/first/'entrypoint_clock.json')['result'],
        clock['result'],lineage.old_freeze(),freeze,route,boundary=envelope.guard)
    result=dict(status='qualified',lifecycle=name,route=route,quota=c.ROUTE_QUOTAS[route],graph=graph,suite=suite,
        control_historical_parity=historical,authenticated_profile_relocation=relocation,
        graph_cases=11,scientific_decisions=67,parent_requests=2)
    save_or_verify(c.OUT/'qualification'/f'{name}.json',result,writing);return result


def qualify(name,workers,envelope,freeze,plan):
    begun=time.monotonic();wait_before=envelope.waited;bytes_before=envelope.accounted_bytes()
    child=transport.make_worker(name,envelope,freeze,route=route_for(name));workers[name]=child
    c.require(child._counter==1,'Full actual suite after graph qualification')
    adapters.actual_entrypoint(child,envelope,freeze,request_fn=transport.request,collect_fn=transport.collect_profiles)
    proof=qualification_evidence(name,envelope,freeze,plan,writing=True)
    elapsed=time.monotonic()-begun
    c.save(c.OUT/'startup_costs'/f'{name}.json',dict(seconds=elapsed,nonwait_seconds=elapsed-(envelope.waited-wait_before),
        accounted_growth_bytes=envelope.accounted_bytes()-bytes_before,raw_stream_bytes=dict(child._log_sizes)))
    return proof


def qualify_pair(stage,start,workers,envelope,freeze,plan):
    c.require(not workers,'Close preceding pair before creating a new block')
    names=[f'{stage}_g{start:03d}_{route}' for route in c.VARIANTS]
    for name in names:qualify(name,workers,envelope,freeze,plan)
    return names


def actual_after(before,action):
    state=decode_training_snapshot(before);environment=rules.CorrectedEnv();chosen=rules.deserialize_action(action)
    c.require(chosen in environment.legal_actions(state),'Corrected complete action legality')
    after=rules.serialize_snapshot(environment.step(state,chosen))
    c.require(after==ledger.transition(before,action),'Actual corrected transition agrees with independent ledger')
    decode_training_snapshot(after);return after


def game_folder(spec):
    return lineage.game_folder(spec)


def source_move_path(selection):
    return lineage.source_move_path(selection)


def decision(child,spec,before,envelope,freeze,*,ply):
    state=decode_training_snapshot(before);observer=rules.observation_payload(state)
    role='challenger' if state.current_player==spec['challenger_seat'] else 'opponent';route=spec[role+'_mode']
    c.require(child.route==route and route_for(child.output_dir.name)==route,'Exact current player route')
    # Only live ordinary execution may consume the single disclosed retry. The
    # irreversible receipt precedes dispatch, so any further failure is closed.
    if lineage.partial(spec) and ply==lineage.PREFIX_PLIES and not child.output_dir.name.startswith('replay_'):
        from ..boundary import consume_authorized_retry
        consume_authorized_retry(c.STUDY_OUT/'authorized_retry_consumed.json',before,
            matchup=c.MATCHUP,repetition=c.REPETITION,stage=spec['stage'],game_index=spec['game_index'],
            ply=ply,route=route,worker_name=child.output_dir.name,
            request_id=f'request-{child._counter+1:08d}')
    row=transport.ordinary(child,spec,observer,envelope,freeze)
    row.update(**spec,ply=ply,variant=route,player_role=role,arm='search',quota=c.ROUTE_QUOTAS[route],mode=route,
        seat=state.current_player,before=before,after=None)
    if row['completed']:
        try:row['after']=actual_after(before,row['action'])
        except BaseException:row['transition_error']=traceback.format_exc()
    return row


def validate_row(row,spec,freeze,envelope,*,worker_name):
    if lineage.source_of(row)=='old':
        return lineage.old_validate_row(row,spec,envelope,worker_name)
    envelope.guard();design.validate_spec(row);c.require(row['spec']==spec,'Exact game request specification')
    role='challenger' if row['seat']==spec['challenger_seat'] else 'opponent';route=spec[role+'_mode']
    c.require(row['variant']==row['route']==row['mode']==route==route_for(worker_name) and row['player_role']==role and
        row['arm']=='search' and row['quota']==c.ROUTE_QUOTAS[route] and row['seat']==row['before']['state']['current_player'] and
        row['worker_folder']==(c.OUT/'workers'/worker_name).relative_to(c.ROOT).as_posix(),'Exact role,quota,lifecycle and acting seat')
    transport._require_profiles(row);transport.verify_saved_record(row,freeze)
    ledger.check_observer(row['before'],row['observer'])
    c.require(row['observer']==rules.observation_payload(decode_training_snapshot(row['before'])),
        'Actual legal observer and canonical observer-derived random seed')
    c.require('transition_error' not in row,'Failed actual transition remains incomplete')
    if row['completed']:
        c.require(actual_after(row['before'],row['action'])==row['after'],'Actual chosen successor independently reconstructed')
        scientific.validate_result(row['clock']['result'],row['observer'],'search',expected_quota=row['quota'])
        scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',before=row['before'],after=row['after'],expected_quota=row['quota'])
        return None
    c.require(row['after'] is None,'A failed move cannot invent an afterstate')
    return scientific.classify_failure(row['clock'])


class SavedRows:
    """Reiterable disk stream; retain at most one full move, never all406 games."""
    def __init__(self,games,boundary=lambda:None):self.games,self.boundary=games,boundary
    def __iter__(self):
        for game in self.games:
            for ply in range(game['plies']):
                self.boundary()
                yield c.read(lineage.move_path(game,ply))


def current_worker_names(workers):
    c.require(len(workers)==2,'One complete resident pair')
    return {route:next(name for name in workers if route_for(name)==route) for route in c.VARIANTS}


def play_game(spec,workers,envelope,freeze):
    design.validate_spec(spec);folder=game_folder(spec);initial=design.initial_snapshot(spec);names=current_worker_names(workers)
    c.save(folder/'spec.json',spec);c.save(folder/'initial.json',dict(**spec,initial=initial));c.save(folder/'workers.json',names)
    state=decode_training_snapshot(initial);rows=[];begun=time.monotonic();wait_before=envelope.waited
    bytes_before=envelope.accounted_bytes();audit_seconds=0.;outcome=None
    streams_before={name:dict(child._log_sizes) for name,child in workers.items()}
    for ply in range(c.MAX_PLIES):
        if state.terminal:break
        role='challenger' if state.current_player==spec['challenger_seat'] else 'opponent';route=spec[role+'_mode']
        row=decision(workers[names[route]],spec,rules.serialize_snapshot(state),envelope,freeze,ply=ply)
        c.save(folder/'moves'/f'{ply:06d}.json',row);rows.append(row);started=time.monotonic()
        failure=validate_row(row,spec,freeze,envelope,worker_name=names[route]);audit_seconds+=time.monotonic()-started
        if failure:
            outcome=dict(reason='forfeit',winner=1-row['seat'],loser=row['seat'],failed_seat=row['seat'],failed_role=role,
                failure=failure,administrative_forfeit=True,failure_policy='research_hard_guard_9s_internal_10s_complete_response')
            c.save(folder/'captured_failure.json',dict(**spec,ply=ply,clock=c.artifact(c.ROOT/row['worker_clock_file']),**outcome))
            break
        state=decode_training_snapshot(row['after'])
        c.require(state.terminal or not(row['after']==row['before'] or row['action']['action_type']=='PASS' and
            len(rows)>1 and rows[-2]['action']['action_type']=='PASS'),'Unresolved actual-game deadlock remains incomplete')
    if outcome is None:
        c.require(state.terminal,'Historical2000-ply safety cap; no fabricated terminal result')
        outcome=dict(reason='natural',winner=state.winner)
    ending=dict(**spec,**outcome,complete=True,plies=len(rows),final=rules.serialize_snapshot(state))
    c.save(folder/'result.json',ending);started=time.monotonic()
    proof=profile_parity.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
    audit_seconds+=time.monotonic()-started;c.save(folder/'online_replay.json',proof)
    elapsed=time.monotonic()-begun;waited=envelope.waited-wait_before
    c.require(0<=waited<=elapsed,'Charged intermove RAM waits belong to current game')
    growth=envelope.accounted_bytes()-bytes_before
    c.require(growth>0,'Conservative game evidence growth accounting')
    cost=dict(seconds=elapsed,nonwait_seconds=elapsed-waited,ram_wait_seconds=waited,independent_seconds=audit_seconds,
        accounted_growth_bytes=growth,plies=len(rows),completed_decisions=sum(r['completed'] for r in rows),
        maximum_receipt_seconds=max((r['clock']['receipt_elapsed_seconds'] or 0) for r in rows))
    cost['raw_stream_growth']={route:{key:workers[name]._log_sizes[key]-streams_before[name][key]
        for key in streams_before[name]} for route,name in names.items()}
    c.save(folder/'cost.json',cost)
    print(c.canonical(dict(stage=spec['stage'],game=spec['game_index']+1,total=c.OPERATIONAL_GAMES if spec['stage']=='operational' else c.MAIN_GAMES,
        plies=len(rows),ending=outcome['reason'])),flush=True)
    return ending


def expected_worker_plan(games):
    """Fixed block boundaries, with new trees only after a recorded guard forfeit."""
    result={};start=0;previous=None
    for game in games:
        design.validate_spec(game);stage=game['stage'];index=game['game_index']
        if stage=='operational':start=0
        elif lineage.partial(game) or index%c.GAMES_PER_BLOCK==0 or previous is not None and previous['stage']==stage and previous['reason']=='forfeit':start=index
        names={route:f'{stage}_g{start:03d}_{route}' for route in c.VARIANTS}
        result[(stage,index)]=names;previous=game
    return result


def read_games(schedule,envelope,freeze):
    from .recovered_audit import read_games as merged_read_games
    return merged_read_games(schedule,envelope,freeze)


def replay_selected(selection,workers,envelope,freeze,schedule):
    specs={(s['stage'],s['game_index']):s for s in schedule}
    for index,item in enumerate(selection['decisions']):
        envelope.guard();path=source_move_path(item);original=c.read(path);spec=specs[(item['stage'],item['game_index'])]
        name='replay_'+item['variant'];row=decision(workers[name],spec,original['before'],envelope,freeze,ply=item['ply'])
        row.update(source_move=c.artifact(path),selection=deepcopy(item))
        folder=c.OUT/'replays'/f'r{index:03d}';c.save(folder/'selection.json',item);c.save(folder/'decision.json',row)
        c.require(validate_row(row,spec,freeze,envelope,worker_name=name) is None,'Any failed reproduction remains incomplete')
        proof=profile_parity.verify_replay(original,row,item,freeze=freeze,boundary=envelope.guard);c.save(folder/'verification.json',proof)
        print(c.canonical(dict(stage='replay',completed=index+1,total=len(selection['decisions']))),flush=True)


def verify_replays(selection,envelope,freeze,schedule):
    specs={(s['stage'],s['game_index']):s for s in schedule};counts={};decisions=selection['decisions']
    c.require(0<len(decisions)<=c.MAX_REPLAY_DECISIONS and
        {p.name for p in(c.OUT/'replays').iterdir() if p.is_dir()}=={f'r{i:03d}' for i in range(len(decisions))},
        'Exact outcome-blind selected reproduction inventory')
    for index,item in enumerate(decisions):
        envelope.guard();folder=c.OUT/'replays'/f'r{index:03d}';row=c.read(folder/'decision.json');path=source_move_path(item)
        c.require(c.read(folder/'selection.json')==item and row['selection']==item and row['source_move']==c.artifact(path),
            'Reproduction binds immutable exact source decision')
        name='replay_'+item['variant'];expected=counts.get(name,0)+3
        c.require(row['clock']['request_id']==f'request-{expected:08d}','Exact replay request sequence')
        c.require(validate_row(row,specs[(item['stage'],item['game_index'])],freeze,envelope,worker_name=name) is None,
            'Failed reproduction cannot qualify the study')
        proof=profile_parity.verify_replay(c.read(path),row,item,freeze=freeze,boundary=envelope.guard)
        c.require(proof==c.read(folder/'verification.json'),'Independent exact fresh-process reproduction')
        counts[name]=expected-2
    return counts


def verify_closed_before(descriptors,wall):
    from datetime import datetime
    c.require(len(descriptors)==2 and len({d['path'] for d in descriptors})==2,'Exact prior pair cleanup receipts')
    for item in descriptors:
        c.verify([item]);value=c.read(c.ROOT/item['path'])
        c.require(not value['alive'] and not value['errors'] and
            datetime.fromisoformat(value['time']).timestamp()<=wall,'Both preceding trees closed before next process admission')


def close_workers(workers,envelope,freeze):
    for name in list(workers):
        transport.close_and_audit(workers[name],envelope,freeze);workers.pop(name)


def collect_main(schedule,workers,envelope,freeze,plan):
    games=[]
    for spec in schedule:
        if not workers:begin_block(spec,workers,envelope,freeze,plan)
        game=play_game(spec,workers,envelope,freeze);games.append(game)
        if game['reason']=='forfeit' or (spec['game_index']+1)%c.GAMES_PER_BLOCK==0 or spec==schedule[-1]:
            close_workers(workers,envelope,freeze)
    return games


def resume_partial(workers,envelope,freeze,plan):
    """Keep22 completed decisions; retry only the user-authorized uncertain ply22."""
    c.require(lineage.partial_cohort() and not workers,'One fixed interrupted game only')
    spec=design.make_schedule('main')[lineage.PARTIAL_GAME]
    source=lineage.old_game_folder(spec);folder=game_folder(spec)
    c.require(not folder.exists() and not (source/'result.json').exists(),'Exclusive unfinished source game')
    c.require([p.name for p in sorted((source/'moves').glob('*.json'))]==
              [f'{i:06d}.json' for i in range(lineage.PREFIX_PLIES)],'Exact immutable22-move prefix')
    initial=design.initial_snapshot(spec)
    c.require(c.read(source/'spec.json')==spec and c.read(source/'initial.json')==dict(**spec,initial=initial),
              'Original fixed interrupted setup')
    old_names=c.read(source/'workers.json');rows=[];current=initial
    for ply in range(lineage.PREFIX_PLIES):
        row=c.read(lineage.move_path(spec,ply))
        c.require(row['ply']==ply and row['before']==current and row['completed'] is True,
                  'Contiguous completed prefix; no completed decision is repeated')
        c.require(validate_row(row,spec,freeze,envelope,worker_name=old_names[row['variant']]) is None,
                  'Saved prefix science and original clocks pass')
        rows.append(row);current=row['after']
    c.require(not current['state']['terminal'],'Exactly the recorded unfinished game remains')
    from ..boundary import validate_retry_authorization
    validate_retry_authorization()
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    original_execution=c.read(c.parent.LIVE_SOURCE_OUT/'segments/main_block_terminal_parent_r1_g030/execution.json')
    for ident in original_execution['observed_owned_identities']+[original_execution['identity']]:require_absent(ident)
    absence=lineage.orphan_absence_descriptor();stamp=time.time()
    start_item=c.artifact(c.OUT/'segments/resume_partial/start.json')
    admission=dict(matchup=c.MATCHUP,repetition=1,game_index=32,start_ply=22,
        original_owner_absence=absence,segment_start=start_item,both_operational_complete=_both_operational_passed(),
        source_initial=c.artifact(source/'initial.json'),source_spec=c.artifact(source/'spec.json'),
        prefix=[c.artifact(lineage.move_path(spec,i)) for i in range(22)],
        before_sha256=c.digest(current),unix_time=stamp,monotonic=time.monotonic(),
        original_monotonic_origin=envelope.start,original_unix_origin=envelope.wall_start,
        effective_monotonic_deadline=envelope.deadline,effective_unix_deadline=envelope.wall_deadline,
        original_active_decision_uncertain=True,authorized_potentially_interrupted_retries=1,
        retry_identity=lineage.authorized_retry_identity(),completed_decisions_retried=0,game_replaced=False)
    c.save(c.OUT/'resume_admission.json',admission)
    qualify_pair('main',32,workers,envelope,freeze,plan)
    names=current_worker_names(workers)
    c.save(folder/'spec.json',spec);c.save(folder/'initial.json',dict(**spec,initial=initial));c.save(folder/'workers.json',names)
    c.save(folder/'continuation.json',dict(source_initial=c.artifact(source/'initial.json'),
        source_workers=c.artifact(source/'workers.json'),source_freeze=c.artifact(c.LIVE_SOURCE_OUT/'freeze.json'),
        prefix=[c.artifact(lineage.move_path(spec,i)) for i in range(22)],start_ply=22,
        admission=c.artifact(c.OUT/'resume_admission.json'),original_rows_unchanged=True,
        authorized_potentially_interrupted_retries=1,completed_decisions_retried=0,game_replacements=0))
    state=decode_training_snapshot(current);begun=time.monotonic();wait_before=envelope.waited
    bytes_before=envelope.accounted_bytes();audit_seconds=0.;outcome=None;new_rows=[]
    streams_before={name:dict(child._log_sizes) for name,child in workers.items()}
    for ply in range(22,c.MAX_PLIES):
        if state.terminal:break
        role='challenger' if state.current_player==spec['challenger_seat'] else 'opponent';route=spec[role+'_mode']
        row=decision(workers[names[route]],spec,rules.serialize_snapshot(state),envelope,freeze,ply=ply)
        c.save(folder/'moves'/f'{ply:06d}.json',row);rows.append(row);new_rows.append(row);started=time.monotonic()
        failure=validate_row(row,spec,freeze,envelope,worker_name=names[route]);audit_seconds+=time.monotonic()-started
        if failure:
            outcome=dict(reason='forfeit',winner=1-row['seat'],loser=row['seat'],failed_seat=row['seat'],failed_role=role,
                failure=failure,administrative_forfeit=True,failure_policy='research_hard_guard_9s_internal_10s_complete_response')
            c.save(folder/'captured_failure.json',dict(**spec,ply=ply,clock=c.artifact(c.ROOT/row['worker_clock_file']),**outcome))
            break
        state=decode_training_snapshot(row['after'])
        c.require(state.terminal or not(row['after']==row['before'] or row['action']['action_type']=='PASS' and
            rows[-2]['action']['action_type']=='PASS'),'Unresolved actual-game deadlock remains incomplete')
    if outcome is None:
        c.require(state.terminal,'Historical2000-ply cap; no invented ending');outcome=dict(reason='natural',winner=state.winner)
    ending=dict(**spec,**outcome,complete=True,plies=len(rows),final=rules.serialize_snapshot(state))
    c.save(folder/'result.json',ending);started=time.monotonic()
    proof=profile_parity.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
    audit_seconds+=time.monotonic()-started;c.save(folder/'online_replay.json',proof)
    elapsed=time.monotonic()-begun;waited=envelope.waited-wait_before;growth=envelope.accounted_bytes()-bytes_before
    c.require(0<=waited<=elapsed and growth>0,'Bounded suffix costs')
    cost=dict(seconds=elapsed,nonwait_seconds=elapsed-waited,ram_wait_seconds=waited,independent_seconds=audit_seconds,
        accounted_growth_bytes=growth,plies=len(rows),saved_prefix_plies=22,new_plies=len(new_rows),
        completed_decisions=sum(r['completed'] for r in new_rows),scope='new suffix only; original prefix timing retained',
        maximum_receipt_seconds=max((r['clock']['receipt_elapsed_seconds'] or 0) for r in new_rows))
    cost['raw_stream_growth']={route:{key:workers[name]._log_sizes[key]-streams_before[name][key]
        for key in streams_before[name]} for route,name in names.items()}
    c.save(folder/'cost.json',cost)
    print(c.canonical(dict(stage='main',game=33,total=200,plies=len(rows),ending=outcome['reason'],
        disclosed_recovery=True)),flush=True)
    # Resident pair remains for33..39; only a genuine hard-guard forfeit closes it.
    if outcome['reason']=='forfeit':close_workers(workers,envelope,freeze)
    return ending


def replay_required_seconds(selection):
    # Observed full same-route startup prices, generously padded; clocks still enforce actual caps.
    startups=[c.read(c.LIVE_SOURCE_OUT/'startup_costs'/f'operational_g000_{v}.json') for v in c.VARIANTS]
    return 1.5*sum(x['nonwait_seconds'] for x in startups)+len(selection['decisions'])*c.SECONDS_PER_MOVE+60.


class AssessmentEnvelope:
    """Per-repetition evidence view; all actual guards remain global."""
    def __init__(self, root):
        self.root = root

    def __getattr__(self, name):
        return getattr(self.root, name)

    @property
    def evidence(self):
        value = self.root.evidence
        return {**value, 'workers': {p: v for p, v in value['workers'].items()
                                   if Path(p).resolve().is_relative_to((c.OUT / 'workers').resolve())}}


def lifecycle_inventory():
    inventory = {}
    for route in c.VARIANTS:
        inventory['qualification_' + route] = dict(route=route, quota=128,
            graph_request_id='request-00000001', entrypoint_request_id='request-00000002',
            ordinary_first_request_id='request-00000003', ordinary_requests=0, startup_requests=2)
    for stage, count in (('operational', 6), ('main', 200)):
        for index in (range(count) if stage == 'main' else range(1)):
            remaining = count if stage == 'operational' else min(10 - index % 10, count - index)
            for route in c.VARIANTS:
                inventory[f'{stage}_g{index:03d}_{route}'] = dict(route=route, quota=128,
                    graph_request_id='request-00000001', entrypoint_request_id='request-00000002',
                    ordinary_first_request_id='request-00000003', ordinary_requests=remaining * c.MAX_PLIES // 2,
                    startup_requests=2)
    for route in c.VARIANTS:
        inventory['replay_' + route] = dict(route=route, quota=128,
            graph_request_id='request-00000001', entrypoint_request_id='request-00000002',
            ordinary_first_request_id='request-00000003', ordinary_requests=c.MAX_REPLAY_DECISIONS,
            startup_requests=2)
    c.require(len(inventory) == c.MAX_WORKER_LIFECYCLES == 406, 'Exact per-repetition lifecycle ceiling')
    return inventory


def contract():
    return dict(schema='splendorust_full_corpus_amended_assessment_v1', phase='pilot', repetition=c.REPETITION, matchup=c.MATCHUP, route_arms=dict(c.ROUTE_ARMS),
        job_seconds=c.JOB_SECONDS, collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES, independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        automatic_retries=0, failed_decision_retries=0,
        authorized_potentially_interrupted_retries=1 if lineage.partial_cohort() else 0,
        retry_identity=lineage.authorized_retry_identity() if lineage.partial_cohort() else None,
        original_attempt_remains_incomplete=True, seconds_per_move=10., internal_seconds=9.,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES, prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES, max_rss_bytes=c.MAX_RSS_BYTES,
        max_workers=c.MAX_WORKER_LIFECYCLES, max_resident_workers=2,
        initial_available_ram_bytes=0, minimum_available_ram_bytes=0,
        initial_ram_stability_seconds=0, maximum_ram_wait_seconds=0,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES, minimum_free_disk_bytes=c.MINIMUM_FREE_DISK_BYTES,
        placement_mode='isolated', routes=dict(c.ROUTE_QUOTAS),
        declared_worker_lifecycles=lifecycle_inventory(),
        qualification_scientific_decisions_maximum=c.MAX_SCIENTIFIC_QUALIFICATION,
        graph_cases_maximum=c.MAX_GRAPH_CASES, maximum_live_decisions=c.MAX_LIVE_DECISIONS,
        maximum_replay_decisions=96, maximum_plies_per_game=c.MAX_PLIES,
        parent_requests_maximum=c.MAX_LIVE_DECISIONS + 96 + 2 * c.MAX_WORKER_LIFECYCLES,
        games_per_main_block=10, operational_games=6, main_games=200, main_pairs=100, new_games=206,
        training=False, one_second_gate=False, seven_hundred_ms_gate=False,
        only_scientific_change='prospectively_fixed_full_corpus_checkpoint_weights_at_matched128',
        network_execution='unchanged_original_full_token_graph', world_pool=3, cpuct=.4, search_depth=64,
        max_response_bytes=c.MAX_RESPONSE_BYTES, max_raw_log_bytes=c.MAX_LOG_BYTES,
        parent_checkpoint_unchanged=True, candidate_checkpoint_selected_by_assessment=False,
        new_numerical_tolerances_used=False, native_host_parity_max_abs_delta=.0005,
        graph_exact_outputs_required=True, main_classified_hard_guard_is_loss=True,
        continue_fixed_main_sample_after_guard_loss=True, stop_operational_on_any_failure=True,
        stop_on_infrastructure_failure=True, game_replacements=0,
        worker_restart_only_before_next_unplayed_game=not lineage.partial_cohort(), full_qualification_each_lifecycle=True,
        resource_forecast_policy='advisory_only', resource_forecasts_may_stop=False,
        phase_estimates_advisory=True, replay_full_result_exact_within_route=True,
        standalone_qualification_workers=2, normal_lifecycles=46, strength_promotion=False)


def validate_freeze_contract(freeze):
    expected = contract()
    c.require(type(freeze.get('repetition')) is int and freeze['repetition'] == c.REPETITION,
              'Strict fixed repetition in local freeze')
    c.require(c.canonical({k: freeze.get(k) for k in expected}) == c.canonical(expected),
              'Exact typed independent-data assessment contract')
    c.require(set(freeze['profiles']) == set(c.VARIANTS), 'Exactly two weight-specific profiles')
    return expected


def build_plan():
    return dict(schema='full_corpus_assessment_plan_v1', repetition=c.REPETITION, contract=contract(),
        qualification=qualification.build_plan(), schedule=design.make_schedule(),
        checkpoint_binding='accepted fixed step1782 source; exact state-only import for every trained route; no selection',
        original_profiles={str(q): c.artifact(adapters.ORIGINAL_PROFILE_ROOT/f'puct{q}.json') for q in (16, 128)},
        native_artifacts={r: adapters.native_artifacts(r) for r in c.VARIANTS})


def prepare(*, boundary=lambda: None):
    """Model-free imports and fixed plan, called once per repetition before freeze."""
    from . import checkpoint
    prepared = design.prepare()
    plan = build_plan()
    plan_item = c.save(c.OUT / 'assessment_plan.json', plan)
    qplan = c.save(c.OUT / 'qualification_plan.json', plan['qualification'])
    imported = checkpoint.prepare_candidate(boundary=boundary)
    profiles = adapters.prepare_profiles(imported['trained_manifests'], imported['checkpoint_bindings'])
    return dict(assessment_plan=plan_item, qualification_plan=qplan, **prepared, **imported, **profiles)


def _prepared(rootfreeze):
    from . import checkpoint
    from ..schedule import MATCHUP_ARMS
    c.require(c.canonical(rootfreeze['matchups'])==c.canonical(MATCHUP_ARMS)
              and rootfreeze['updates_per_fit']==1782, 'Fixed prospective arm map and full-corpus final step')
    descriptor = c.artifact(c.OUT / 'prepared.json')
    doc = c.read(c.OUT / 'prepared.json')
    c.require(doc['status'] == 'prepared' and type(doc['repetition']) is int and
              doc['repetition'] == c.REPETITION and doc['matchup']==c.MATCHUP and doc['root_freeze'] == c.artifact(c.STUDY_OUT / 'freeze.json') and
              c.canonical(doc['all_fits_committed']) == c.canonical(checkpoint.commit_descriptor()),
              'Later preparation bound to original recipe and all committed fits')
    c.verify(list(c.parent.nested_artifacts(doc['prepared'])))
    return {**doc['prepared'], 'prepared_receipt': descriptor,
            'all_fits_committed': doc['all_fits_committed']}


def bind_checkpoint(rootfreeze, candidate_manifest, envelope):
    c.require(not (c.OUT / 'freeze.json').exists(), 'One immutable local binding; no replacement')
    prepared = _prepared(rootfreeze)
    c.require(prepared['candidate_manifest'] == candidate_manifest, 'Accepted fixed checkpoint only')
    c.verify(list(c.parent.nested_artifacts(prepared)))
    c.require(c.read(c.ROOT / prepared['assessment_plan']['path']) == build_plan(), 'Prospective fixed plan')
    value = deepcopy(rootfreeze)
    value.update(contract())
    value.update(prepared)
    value.update(root_freeze=c.artifact(c.parent.OUT / 'freeze.json'),
                 binding_created_unix=time.time(), binding_created_monotonic=time.monotonic())
    c.save(c.OUT / 'freeze.json', value)
    verify_binding(value, boundary=envelope.guard)
    return value


def verify_binding(freeze, *, boundary=lambda: None):
    from . import checkpoint
    boundary()
    c.verify([freeze['root_freeze'], freeze['candidate_manifest']])
    c.require(freeze['root_freeze'] == c.artifact(c.parent.OUT / 'freeze.json'), 'Original root freeze')
    rootfreeze = c.read(c.ROOT / freeze['root_freeze']['path'])
    prepared = _prepared(rootfreeze)
    c.require(c.canonical({k: freeze.get(k) for k in prepared}) == c.canonical(prepared), 'Every prepared repetition binding and exact JSON type')
    c.require(c.read(c.ROOT / freeze['assessment_plan']['path']) == build_plan(), 'All assessment choices fixed')
    c.require(set(freeze['trained_manifests'])=={r for r in c.VARIANTS if not c.is_parent_route(r)}, 'Exact trained-player inventory')
    for route in c.VARIANTS:
        binding=freeze['checkpoint_bindings'][route]
        if c.is_parent_route(route):
            checkpoint.verify_parent_export(boundary=boundary)
            c.require(binding['checkpoint']==c.artifact(checkpoint.architecture.PARENT_CHECKPOINT)
                      and binding['native_reference']==c.artifact(checkpoint.MODEL_BIN),'Original parent route byte identity')
        else:
            descriptor=freeze['trained_manifests'][route]
            manifest,_=checkpoint.candidate_sources(descriptor,boundary=boundary)
            c.require(manifest['route']==route and manifest['metadata']['qualification_only'] is False
                      and binding['checkpoint']==manifest['checkpoint']
                      and binding['weights_sha256']==manifest['metadata']['weights_sha256'], 'Exact route-specific accepted final weights')
            checkpoint.verify_candidate_export(descriptor,binding['native_reference'],boundary=boundary)
    for route in c.VARIANTS:
        for quota in (16, 128):
            adapters.profile_identity(freeze, route, quota, boundary=boundary)
    return dict(status='passed', repetition=c.REPETITION, same_search128=True,
                candidate_manifest=freeze['candidate_manifest'])


def verify_runtime(freeze, envelope=None):
    validate_freeze_contract(freeze)
    boundary = envelope.guard if envelope else lambda: None
    boundary()
    rootfreeze = c.read(c.ROOT / freeze['root_freeze']['path'])
    for item in rootfreeze['inputs']:
        boundary()
        c.verify([item])
    verify_binding(freeze, boundary=boundary)


def segment_id(phase, block_start=None):
    c.require(type(phase) is str and phase in ('prepare', 'qualification', 'resume_partial', 'main_block', 'replay', 'saved'),
              'Known fixed segment phase')
    if phase == 'main_block':
        c.require(type(block_start) is int and block_start in range(0, 200, 10), 'Fixed main block start')
        return f'main_{block_start:03d}'
    c.require(block_start is None, 'No block index on another phase')
    if phase=='resume_partial':c.require(lineage.partial_cohort(),'One fixed interrupted cohort only')
    return phase


def _segment_start(envelope, freeze, phase, block_start):
    name = segment_id(phase, block_start)
    c.require(not (c.OUT / 'segments' / name).exists(), 'A segment is never retried or overwritten')
    phase_clock = envelope.phase_receipt
    claim = envelope.claim
    c.verify([phase_clock, claim])
    clock = c.read(c.ROOT / phase_clock['path'])
    local_phase = 'main' if phase in ('main_block','prepare','resume_partial') else 'final' if phase == 'saved' else phase
    expected_phase = c.STAGE_MAP[local_phase]
    c.require(clock['stage'] == envelope.stage == expected_phase and clock['claim'] == claim,
              'Segment cannot run under a different root phase or claim')
    c.require(clock['monotonic_deadline'] == envelope.deadline and
              clock['unix_deadline'] == envelope.wall_deadline, 'Original root phase clocks')
    for key, origin in (('monotonic', envelope.start), ('unix_time', envelope.wall_start)):
        deadline = 'monotonic_deadline' if key == 'monotonic' else 'unix_deadline'
        cap = c.JOB_SECONDS if phase == 'saved' else c.COLLECTION_CUTOFF_SECONDS
        c.require(clock[key] <= (time.monotonic() if key == 'monotonic' else time.time()) < clock[deadline]
                  and clock[deadline] <= origin + cap, 'No new study or model clock')
    value = dict(schema='full_corpus_assessment_segment_start_v1', segment=name, phase=phase,
        repetition=c.REPETITION, matchup=c.MATCHUP, block_start=block_start, owner=dict(pid=os.getpid(), created=psutil.Process().create_time()),
        time=c.now(), monotonic=time.monotonic(), unix_time=time.time(), original_monotonic_origin=envelope.start,
        original_unix_origin=envelope.wall_start, monotonic_deadline=envelope.deadline, unix_deadline=envelope.wall_deadline,
        root_claim=claim, root_phase_clock=phase_clock, local_freeze=c.artifact(c.OUT / 'freeze.json'),
        segment_claim=getattr(envelope, 'segment_claim', None), active_decision=False, deadline_reset=False)
    if value['segment_claim'] is not None:
        c.verify([value['segment_claim']])
    return c.save(c.OUT / 'segments' / name / 'start.json', value)


def _prior_complete(name):
    path = c.OUT/'segments/prepare/complete.json' if name=='prepare' else lineage.prior_segment(name)
    value = c.read(path)
    c.require(value['status'] == 'segment_complete' and value['repetition'] == c.REPETITION and
              value['resident_workers_after'] == 0, 'Prior segment closed successfully')
    return c.artifact(path)


def _both_operational_passed():
    results = []
    for rep in (0, 1):
        folder = c.parent.LIVE_SOURCE_OUT / 'strength' / c.MATCHUP / f'r{rep}'
        receipt = c.read(folder / 'segments' / 'operational' / 'complete.json')
        c.require(receipt['status'] == 'segment_complete' and receipt['repetition'] == rep and receipt.get('matchup')==c.MATCHUP and
                  receipt['games'] == 6 and receipt['resident_workers_after'] == 0,
                  'Both six-game operational cohorts completed before any main game')
        results.append(c.artifact(folder / 'segments' / 'operational' / 'complete.json'))
    return results


def prior_pair_closures(stage, index):
    prior = dict(stage='operational' if stage == 'main' and index == 0 else 'main',
                 game_index=5 if stage == 'main' and index == 0 else index - 1)
    names = c.read(game_folder(prior) / 'workers.json')
    return [c.artifact(lineage.worker_folder(names[route],lineage.game_source(prior))/'cleanup.json') for route in c.VARIANTS]


def begin_block(spec, workers, envelope, freeze, plan):
    c.require(spec['stage'] == 'main' and not workers, 'New block requires closed previous workers')
    index = spec['game_index']
    previous = None if index == 0 else c.artifact(game_folder(dict(stage='main', game_index=index - 1)) / 'result.json')
    if index % 10:
        c.require(c.read(c.ROOT / previous['path'])['reason'] == 'forfeit', 'Midblock restart only after guard loss')
    record = dict(stage='main', repetition=c.REPETITION, start_game_index=index,
        monotonic=time.monotonic(), unix_time=time.time(), original_monotonic_origin=envelope.start,
        original_unix_origin=envelope.wall_start, effective_monotonic_deadline=envelope.deadline,
        effective_unix_deadline=envelope.wall_deadline, previous_game=previous,
        reason='fixed_block' if index % 10 == 0 else 'next_unplayed_after_guard_loss',
        previous_closed_workers=prior_pair_closures('main', index), no_retried_decision=True,
        segment_start=c.artifact(c.OUT / 'segments' / ('resume_partial' if lineage.partial_cohort() and 32<index<40 else f'main_{index // 10 * 10:03d}') / 'start.json'),
        both_operational_complete=_both_operational_passed())
    verify_closed_before(record['previous_closed_workers'], record['unix_time'])
    c.save(c.OUT / 'block_admissions' / f'g{index:03d}.json', record)
    return qualify_pair('main', index, workers, envelope, freeze, plan)


def _finish_segment(start_item, envelope, phase, block_start, *, games=0, replays=0, saved=None):
    envelope.guard()
    c.require(not envelope.resident, 'All resident trees closed and verified before segment completion')
    start = c.read(c.ROOT / start_item['path'])
    result = dict(status='segment_complete', repetition=c.REPETITION, matchup=c.MATCHUP, phase=phase, block_start=block_start,
        segment=start['segment'], start=start_item, owner=start['owner'], monotonic=time.monotonic(),
        unix_time=time.time(), games=games, replays=replays, resident_workers_after=0,
        failed_decision_retries=0, authorized_potentially_interrupted_retries=1 if phase=='resume_partial' else 0,
        game_replacements=0, saved_verification=saved)
    item = c.save(c.OUT / 'segments' / start['segment'] / 'complete.json', result)
    return {**result, 'segment_receipt': item}


def run_segment(root_envelope, rootfreeze, phase, block_start=None):
    """Root dispatches fixed segments; this function never starts or extends a phase."""
    envelope = AssessmentEnvelope(root_envelope)
    name = segment_id(phase, block_start)
    workers = {}
    if phase == 'prepare':
        from . import checkpoint
        c.require(not (c.OUT / 'prepared.json').exists(), 'No repeat of checkpoint preparation')
        envelope.guard()
        prepared = prepare(boundary=envelope.guard)
        c.save(c.OUT / 'prepared.json', dict(status='prepared', repetition=c.REPETITION, matchup=c.MATCHUP,
            prepared=prepared, root_freeze=c.artifact(c.STUDY_OUT / 'freeze.json'),
            all_fits_committed=checkpoint.commit_descriptor(), models=0, games=0))
        freeze = bind_checkpoint(rootfreeze, _prepared(rootfreeze)['candidate_manifest'], envelope)
    else:
        freeze = c.read(c.OUT / 'freeze.json')
    verify_runtime(freeze, envelope)
    specs, plan = load_inputs(freeze, envelope)
    start_item = _segment_start(envelope, freeze, phase, block_start)
    try:
        if phase == 'prepare':
            old=lineage.old_freeze()
            c.require(all(freeze['checkpoint_bindings'][r]['weights_sha256']==old['checkpoint_bindings'][r]['weights_sha256']
                          for r in c.VARIANTS),'Original exact player weights')
            return _finish_segment(start_item,envelope,phase,block_start)
        if phase == 'qualification':
            _prior_complete('prepare')
            names = ['qualification_' + route for route in c.VARIANTS]
            for worker_name in names:
                qualify(worker_name, workers, envelope, freeze, plan)
            summary = dict(accepted=True, workers={n: c.read(c.OUT / 'qualification' / f'{n}.json') for n in names})
            c.save(c.OUT / 'qualification_summary.json', summary)
            close_workers(workers, envelope, freeze)
            return _finish_segment(start_item, envelope, phase, block_start)
        if phase == 'resume_partial':
            _prior_complete('prepare');_both_operational_passed()
            resume_partial(workers,envelope,freeze,plan)
            block=design.make_schedule('main')[33:40]
            games=collect_main(block,workers,envelope,freeze,plan)
            return _finish_segment(start_item,envelope,phase,block_start,games=1+len(games))
        if phase == 'main_block':
            c.require(block_start >= (40 if lineage.partial_cohort() else c.SOURCE_MAIN_COUNT),
                      'Only untouched fixed main blocks')
            _both_operational_passed()
            _prior_complete('operational' if block_start == 0 else f'main_{block_start - 10:03d}')
            block = design.make_schedule('main')[block_start:block_start + 10]
            c.require(len(block) == 10 and [s['game_index'] for s in block] == list(range(block_start, block_start + 10)),
                      'Exactly the next fixed ten-game block')
            games = collect_main(block, workers, envelope, freeze, plan)
            return _finish_segment(start_item, envelope, phase, block_start, games=len(games))
        if phase == 'replay':
            _prior_complete('main_190')
            games, rows, _ = read_games(specs, envelope, freeze)
            selection = design.select_replay_sample(rows, maximum=96)
            c.save(c.OUT / 'replay_selection.json', selection)
            previous = prior_pair_closures('replay', c.MAIN_GAMES)
            c.save(c.OUT / 'replay_admission.json', dict(replay_selection=c.artifact(c.OUT / 'replay_selection.json'),
                previous_closed_workers=previous, segment_start=start_item,
                required_seconds=replay_required_seconds(selection), forecast_policy='advisory_only', may_stop=False))
            verify_closed_before(previous, time.time())
            for worker_name in ['replay_' + route for route in c.VARIANTS]:
                qualify(worker_name, workers, envelope, freeze, plan)
            replay_selected(selection, workers, envelope, freeze, specs)
            close_workers(workers, envelope, freeze)
            result = _finish_segment(start_item, envelope, phase, block_start, replays=len(selection['decisions']))
            c.save(c.OUT / 'collection_completion.json', dict(status='assessment_collected_and_reproduced',
                games=len(games), replays=len(selection['decisions']), repetition=c.REPETITION,
                freeze=c.artifact(c.OUT / 'freeze.json'), replay_segment=result['segment_receipt']))
            return result
        c.require(phase == 'saved', 'Only saved phase remains')
        coverage = verify_saved(envelope, rootfreeze, writing=True)
        return _finish_segment(start_item, envelope, phase, block_start, saved=coverage)
    except BaseException:
        error = traceback.format_exc()
        for worker_name, child in list(workers.items()):
            try:
                if not child._closed:
                    transport.close_and_audit(child, envelope, freeze)
            except BaseException:
                path = c.OUT / 'cleanup_failures' / f'{worker_name}.json'
                if not path.exists():
                    c.save(path, dict(error=traceback.format_exc()))
        c.save(c.OUT / 'segments' / name / 'failure.json', dict(time=c.now(), repetition=c.REPETITION,
            phase=phase, block_start=block_start, error=error, source_start=start_item))
        raise


def _expected_segments():
    return [('qualification', None), ('operational', None)] + [('main_block', start) for start in range(0, 200, 10)] + [('replay', None)]


def _absent(identity):
    c.require(type(identity) is dict and set(identity) == {'pid', 'created'} and
              type(identity['pid']) is int and type(identity['created']) in (float, int), 'Exact process identity')
    try:
        process = psutil.Process(identity['pid'])
    except psutil.NoSuchProcess:
        return True
    return process.create_time() != identity['created']


def verify_segments(envelope, freeze):
    """Validate original clocks and all closed local segment owners, saved only."""
    result = {}
    previous = None
    clock_by_phase = {}
    expected_names = {segment_id(phase, start) for phase, start in _expected_segments()}
    observed_names = {p.name for p in (c.OUT / 'segments').iterdir() if p.is_dir()}
    c.require(observed_names in (expected_names, expected_names | {'saved'}), 'Exact predetermined segment inventory')
    for phase, block_start in _expected_segments():
        envelope.guard()
        name = segment_id(phase, block_start)
        folder = c.OUT / 'segments' / name
        c.require({p.name for p in folder.iterdir()} == {'start.json', 'complete.json'}, 'Passed segment has no failed or extra receipts')
        start, end = c.read(folder / 'start.json'), c.read(folder / 'complete.json')
        c.require(start['segment'] == end['segment'] == name and start['phase'] == end['phase'] == phase and
                  type(start['repetition']) is int and start['repetition'] == end['repetition'] == c.REPETITION and
                  start.get('matchup')==end.get('matchup')==c.MATCHUP and
                  start['block_start'] == end['block_start'] == block_start and
                  start['local_freeze'] == c.artifact(c.OUT / 'freeze.json') and
                  end['start'] == c.artifact(folder / 'start.json') and end['owner'] == start['owner'],
                  'Exact immutable segment binding')
        c.require(end['status'] == 'segment_complete' and end['resident_workers_after'] == 0 and
                  end['failed_decision_retries'] == end['game_replacements'] == 0 and
                  start['active_decision'] is False and start['deadline_reset'] is False,
                  'No retry, replacement or clock reset')
        c.require(end['games'] == (6 if phase == 'operational' else 10 if phase == 'main_block' else 0),
                  'Exact segment game count')
        if phase != 'replay':
            c.require(end['replays'] == 0, 'No hidden replay work outside replay phase')
        c.verify([start['root_claim'], start['root_phase_clock']])
        c.require(start['root_claim'] == envelope.claim and
                  start['original_monotonic_origin'] == envelope.start and start['original_unix_origin'] == envelope.wall_start,
                  'Same original whole-study origins')
        if start['segment_claim'] is not None:
            c.verify([start['segment_claim']])
        clock = c.read(c.ROOT / start['root_phase_clock']['path'])
        phase_name = c.STAGE_MAP['main' if phase == 'main_block' else phase]
        c.require(clock['stage'] == phase_name and clock['claim'] == envelope.claim and
                  clock['cap_seconds'] == c.parent.STAGES[phase_name], 'Frozen aggregate root phase')
        if phase_name in clock_by_phase:
            c.require(clock_by_phase[phase_name] == start['root_phase_clock'], 'Qualification, operational and main blocks share one root phase clock')
        clock_by_phase[phase_name] = start['root_phase_clock']
        for key, origin in (('monotonic', envelope.start), ('unix_time', envelope.wall_start)):
            deadline = 'monotonic_deadline' if key == 'monotonic' else 'unix_deadline'
            c.require(clock[deadline] == min(clock[key] + clock['cap_seconds'], origin + c.COLLECTION_CUTOFF_SECONDS)
                      and clock[key] <= start[key] <= end[key] < clock[deadline] == start[deadline],
                      'Segment lies inside original nonreset root phase and model clocks')
            if previous is not None:
                c.require(previous[key] <= start[key], 'Local segment order is fixed')
        c.require(_absent(start['owner']), 'Prior segment controller actual identity still alive')
        previous = end
        result[name] = (start, end)
    return result


def _worker_segment(name):
    if lineage.partial_cohort() and name.startswith('main_g') and 32<=int(name.split('_')[1][1:])<40:
        return 'resume_partial'
    if name.startswith('qualification_'):
        return 'qualification'
    if name.startswith('operational_'):
        return 'operational'
    if name.startswith('replay_'):
        return 'replay'
    c.require(name.startswith('main_g'), 'Known live lifecycle')
    return f"main_{int(name.split('_')[1][1:]) // 10 * 10:03d}"


def verify_blocks(games, envelope, segments):
    expected = expected_worker_plan(games)
    starts = sorted({int(name.split('_')[1][1:]) for (stage, _), names in expected.items()
                     if stage == 'main' for name in names.values()})
    c.require({p.name for p in (c.OUT / 'block_admissions').glob('*.json')} == {f'g{i:03d}.json' for i in starts},
              'Exact startup or post-forfeit main admissions')
    both = _both_operational_passed()
    for index in starts:
        envelope.guard()
        proof = c.read(c.OUT / 'block_admissions' / f'g{index:03d}.json')
        start, end = segments[f'main_{index // 10 * 10:03d}']
        previous = None if index == 0 else c.artifact(game_folder(dict(stage='main', game_index=index - 1)) / 'result.json')
        c.require(proof['stage'] == 'main' and proof['repetition'] == c.REPETITION and
                  proof['start_game_index'] == index and proof['previous_game'] == previous and
                  proof['reason'] == ('fixed_block' if index % 10 == 0 else 'next_unplayed_after_guard_loss') and
                  proof['no_retried_decision'] is True and proof['both_operational_complete'] == both and
                  proof['segment_start'] == c.artifact(c.OUT / 'segments' / start['segment'] / 'start.json'),
                  'Exact fresh block binding and both operational gates')
        c.require(proof['previous_closed_workers'] == prior_pair_closures('main', index), 'Prior assigned pair closures')
        verify_closed_before(proof['previous_closed_workers'], proof['unix_time'])
        for key, origin in (('monotonic', envelope.start), ('unix_time', envelope.wall_start)):
            deadline = 'monotonic_deadline' if key == 'monotonic' else 'unix_deadline'
            c.require(start[key] <= proof[key] <= end[key] < proof['effective_' + deadline] == start[deadline] and
                      proof['original_' + ('monotonic_origin' if key == 'monotonic' else 'unix_origin')] == origin,
                      'Every block keeps the same inherited phase clock')
        for route in c.VARIANTS:
            init = c.read(c.OUT / 'workers' / f'main_g{index:03d}_{route}' / 'initialization.json')
            c.require(init['identity']['created'] >= proof['unix_time'], 'Admission precedes fresh worker creation')
    return expected


def verify_saved(root_envelope, rootfreeze, *, writing=False):
    from .recovered_audit import verify_saved as verify_merged_saved
    return verify_merged_saved(root_envelope,rootfreeze,writing=writing)
