"""Preserve c135 prefixes and resume only its eight disclosed unfinished moves.

All old artifacts retain their original paths. The new game documents contain
old-prefix and new-suffix row descriptors; no old receipt is rewritten or given
a successful runtime closure. Saved and new journals are audited separately.
"""
from collections import Counter
from pathlib import Path
import threading
import time
import traceback

from . import common as root
from .collection import common as c, audit, collector, data
from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.splendorust_gpu_probe_20261105.control import require_absent, native_fallback_messages

PREVIOUS = root.ROOT/'outputs/splendorust_collection4096_recovery_20261116_01'
CONTAINER = PREVIOUS/'collection/cells/c135'
WORKER, CELL = CONTAINER/'worker', CONTAINER/'worker/cell'
PREVIOUS_FREEZE = 'd1074484d6dbf522288d9ff9354550d7e1c0759925512d976f9582443c301615'
COMPLETED = tuple(f'g{i:04d}' for i in range(2160,2168))
UNFINISHED = tuple(f'g{i:04d}' for i in range(2168,2176))
FAILED_PLIES = dict(zip(UNFINISHED, (27,23,22,22,22,21,20,17)))
EVIDENCE_DIRS = ('attempts','clocks','requests','raw_results','service_calls','captures','moves')
a = audit._legacy


def _spec(plan):
    selected = [x for x in plan['cells'] if x['cell_id']=='c135']
    c.require(len(selected)==1 and [x['game_id'] for x in selected[0]['specifications']]==list(COMPLETED+UNFINISHED),
              'Exactly the original c135 sixteen specifications')
    return selected[0]


def _binding():
    frozen = root.read(root.PLAN)
    item = frozen['saved_partial135']
    c.verify([item])
    return c.read(c.resolve(item))


def _prefix_inventory(folder, count, *, failed):
    names = [f'{i:06d}.json' for i in range(count)]
    result = []
    for directory in EVIDENCE_DIRS:
        files = sorted((folder/directory).glob('*.json'))
        expected = names + ([f'{count:06d}.json'] if failed and directory in ('attempts','clocks','requests','service_calls') else [])
        c.require([x.name for x in files]==expected, 'Exact successful-prefix and unfinished evidence '+directory)
        result.extend(c.artifact(p) for p in files)
    return result


def build_boundary(plan):
    """Model-free preparation: bind the exact saved frontier and every move file."""
    spec = _spec(plan)
    c.require(c.sha(PREVIOUS/'freeze.json')==PREVIOUS_FREEZE, 'Exact preceding frozen recovery')
    queue = c.read(CELL/'queue_outcomes.json')
    assignments = [c.read(CELL/'assignments'/f'{i:02d}.json') for i in range(16)]
    c.require(queue['assignments']==assignments and queue['errors']==queue['never_started']==[],
              'All sixteen assignments exist without queue error or unstarted game')
    c.require(set(queue['outcomes'])==set(COMPLETED+UNFINISHED), 'No extra or replaced game')
    games=[]
    for index,(game,assigned) in enumerate(zip(spec['specifications'],assignments,strict=True)):
        gid=game['game_id'];folder=CELL/'games'/gid;outcome=c.read(folder/'outcome.json')
        c.require(outcome==queue['outcomes'][gid] and assigned['game_id']==gid and assigned['queue_index']==index and
            assigned['client_id']==f"worker_{assigned['worker_index']}" and assigned['slot']==assigned['worker_index'] and
            0<=assigned['slot']<8, 'Exact original game assignment/outcome')
        c.require(c.read(folder/'spec.json')==game and c.read(folder/'initial.json')==data.initial_snapshot(game),
                  'Exact original seed, initial setup and specification')
        count=outcome['completed_moves'];complete=gid in COMPLETED
        c.require(type(count) is int and 0<count<2000 and
            (outcome['status']=='completed' if complete else outcome['status']==('integrity_failed' if gid=='g2174' else 'cancelled')),
            'Frozen saved completion/failure classification')
        inventory=_prefix_inventory(folder,count,failed=not complete)
        before=data.initial_snapshot(game);refs=[];previous=None
        for ply in range(count):
            path=folder/'moves'/f'{ply:06d}.json';row=c.read(path)
            data.validate_row(row,game,before,ply)
            c.require(row['client_id']==assigned['client_id'] and row['slot']==assigned['slot'] and
                (previous is None or previous['clock']['ended_perf_counter']<=row['clock']['started_perf_counter']),
                'Original completed decisions remain ordered in their original slot')
            before=row['after'];previous=row;refs.append(c.artifact(path))
        extra=['spec.json','initial.json','started.json','outcome.json']
        item=dict(game_id=gid,spec=game,assignment=assigned,completed=complete,prefix_count=count,
                  prefix_rows=refs,next_snapshot=before,next_snapshot_sha256=c.digest(before))
        if complete:
            guard.require_official_terminal_target(before,'completed')
            document=c.read(folder/'target_game.json')
            c.require(document['rows']==refs and document['final']==before and document['spec']==game,
                      'Completed source game retains every original row')
            item['document']=c.artifact(folder/'target_game.json')
            extra+=['result.json','target_game.json','finished.json']
        else:
            c.require(count==FAILED_PLIES[gid] and outcome['failed_ply']==count and not guard.decode_training_snapshot(before).terminal and
                outcome['next_snapshot_sha256']==c.digest(before), 'Exact unchanged unfinished ply and snapshot')
            name=f'{count:06d}.json';attempt=c.read(folder/'attempts'/name);clock=c.read(folder/'clocks'/name)
            request=dict(player='search',observer=rules.observation_payload(guard.decode_training_snapshot(before)))
            c.require(attempt['game_id']==gid and attempt['ply']==count and attempt['client_id']==assigned['client_id'] and
                attempt['slot']==assigned['slot'] and attempt['before_sha256']==c.digest(before) and
                attempt['request_sha256']==c.digest(request) and c.read(folder/'requests'/name)==request and
                attempt['started_perf_counter']==clock['started_perf_counter'] and attempt['no_retry'] is True and
                clock['completed'] is False and clock['internal_seconds']==60. and clock['outer_seconds']==75. and
                clock['execution_policy_id']==c.EXECUTION_POLICY_ID and outcome['clock']==clock,
                'Original failed decision request, identity and unmodified60/75 clock')
            c.require(clock['status']==('timeout' if gid=='g2174' else 'error') and
                (clock['elapsed_seconds']>=75. if gid=='g2174' else clock['elapsed_seconds']<75.),
                'One original deadline failure and seven propagated cancellations')
            c.require(c.read(folder/'failure.json')==outcome,'Original failure remains exact')
            item.update(failed_attempt=c.artifact(folder/'attempts'/name),failed_clock=c.artifact(folder/'clocks'/name),
                        failed_outcome=c.artifact(folder/'outcome.json'),retry_count_authorized=1)
            extra+=['failure.json']
        item['evidence']=inventory+[c.artifact(folder/p) for p in extra]
        games.append(item)
    sources=[CONTAINER/p for p in ('owner.json','execution.json')]+[WORKER/p for p in (
        'runtime_sources.json','worker_identity.json','qualification_gate_binding.json','cleanup.json','worker_failure.json',
        'runtime/runtime_identity.json','runtime/runtime_final.json','runtime/active_decisions.json','runtime/inference.jsonl')]
    sources += [CELL/p for p in ('queue_outcomes.json','started.json','warmup.json','failure.json')]
    sources += sorted((CELL/'assignments').glob('*.json'))
    sources += sorted((CELL/'warmup_clocks').glob('*.json'))+sorted((CELL/'warmup_raw').glob('*.json'))
    return dict(schema='c135_disclosed_gameplay_resume_boundary_v1',cell_id='c135',previous_freeze=c.artifact(PREVIOUS/'freeze.json'),
        original_scientific_plan=c.artifact(root.ORIGINAL_OUT/'collection/plan.json'),
        saved_sources=[c.artifact(p) for p in sources],games=games,completed_games=8,unfinished_games=8,
        completed_moves=sum(x['prefix_count'] for x in games),disclosed_decision_retries=8,
        original_attempt_incomplete=True,completed_games_replayed=0,completed_moves_repeated=0,
        decision_internal_seconds=60.,decision_outer_seconds=75.,original_deadlines_unchanged=True)


def validate_boundary(binding,plan,boundary=lambda:None):
    c.require(binding['schema']=='c135_disclosed_gameplay_resume_boundary_v1' and binding['cell_id']=='c135' and
        binding['previous_freeze']['sha256']==PREVIOUS_FREEZE and binding['completed_games']==binding['unfinished_games']==8 and
        binding['disclosed_decision_retries']==8 and binding['completed_games_replayed']==binding['completed_moves_repeated']==0 and
        binding['original_attempt_incomplete'] is binding['original_deadlines_unchanged'] is True and
        binding['decision_internal_seconds']==60. and binding['decision_outer_seconds']==75., 'Exact disclosed recovery boundary')
    c.verify([binding['previous_freeze'],binding['original_scientific_plan']]+binding['saved_sources'],boundary)
    c.require(binding['original_scientific_plan']==c.artifact(root.ORIGINAL_OUT/'collection/plan.json') and
        c.canonical(plan)==c.canonical(c.read(root.ORIGINAL_OUT/'collection/plan.json')), 'Unchanged scientific collection plan')
    c.require([x['game_id'] for x in binding['games']]==list(COMPLETED+UNFINISHED), 'Exact saved frontier inventory')
    for game,item in zip(_spec(plan)['specifications'],binding['games'],strict=True):
        boundary();c.require(item['spec']==game,'Original game specification retained')
        c.verify(item['evidence'],boundary)
        c.require(item['next_snapshot_sha256']==c.digest(item['next_snapshot']) and
            len(item['prefix_rows'])==item['prefix_count'] and 0<item['prefix_count']<2000 and
            item['completed']==(game['game_id'] in COMPLETED), 'Saved game boundary consistency')
        c.require(data.read_artifact(item['prefix_rows'][-1])['after']==item['next_snapshot'] and
            item['assignment']==c.read(CELL/'assignments'/f"{game['game_index']-2160:02d}.json"),
            'Saved frontier and assignment come from actual preserved records')
        if not item['completed']:
            c.require(item['prefix_count']==FAILED_PLIES[game['game_id']] and item['retry_count_authorized']==1 and
                item['failed_attempt']==c.artifact(CELL/'games'/game['game_id']/'attempts'/f"{item['prefix_count']:06d}.json"),
                'Only the exact one unfinished decision per game can receive a new attempt')
        current=_prefix_inventory(CELL/'games'/game['game_id'],item['prefix_count'],failed=not item['completed'])
        c.require(current==item['evidence'][:len(current)], 'No prefix evidence added, removed or replaced')
        c.require(item['prefix_rows']==[c.artifact(CELL/'games'/game['game_id']/'moves'/f'{i:06d}.json')
            for i in range(item['prefix_count'])], 'No prefix row substituted')
    c.require(binding['completed_moves']==sum(x['prefix_count'] for x in binding['games']), 'Full completed-prefix count')


def _bind(boundary):
    audit.set_boundary(boundary);a.guard_audit_clock=lambda claim=None:boundary()


def _saved_execution(plan):
    execution=c.read(CONTAINER/'execution.json');owner=c.read(CONTAINER/'owner.json')
    c.require(execution['status']=='failed' and execution['worker_exit_code']==1 and execution['retries']==0 and
        execution['identity']==owner['identity'] and execution['mode']==owner['mode']=='games' and
        execution['plan_sha256']==owner['plan_sha256']==PREVIOUS_FREEZE and owner['windows_job_assigned'] is True and
        not execution['cleanup_errors'] and execution['error'] and
        execution['peak_rss_bytes']<=owner['maximum_rss_bytes']<=c.MAX_RSS, 'Exact failed previous owned worker')
    c.require(owner['global_claim']==c.artifact(PREVIOUS/'claim.json'), 'Previous owner claim binding')
    c.verify([owner['phase_clock']]);phase=c.read(c.resolve(owner['phase_clock']));claim=c.read(PREVIOUS/'claim.json')
    for origin,end,deadline,start,limit in [('monotonic_origin','monotonic_end','monotonic_deadline','monotonic','monotonic_deadline'),
        ('unix_origin','unix_end','unix_deadline','unix_time','unix_deadline')]:
        c.require(owner[origin]==execution[origin] and owner[deadline]==execution[deadline] and
            claim[origin]<=phase[start]<=owner[origin]<=execution[end]<owner[deadline]<=phase[limit], 'Original failure dual-clock chain')
    identities={(x['pid'],x['created']):x for x in execution['observed_owned_identities']+[execution['identity']]}
    for identity in identities.values():require_absent(identity)
    startup=c.read(WORKER/'runtime_sources.json')
    c.require(startup['sources']==plan['source_guard'] and startup['identity']==execution['identity'] and
        startup['original_claim']==owner['global_claim'] and startup['original_phase']==owner['phase_clock'], 'Prior once-startup actual source guard')
    c.require(not native_fallback_messages(WORKER/'stderr.log'),'No original native fallback')
    gate=c.read(root.ORIGINAL_OUT/'collection/qualification_gate.json');c.verify([gate['audit']]);qualification=c.read(c.resolve(gate['audit']))
    c.require(gate['passed'] is True and gate['collection_plan']==c.artifact(root.ORIGINAL_OUT/'collection/plan.json') and
        qualification['status']=='passed' and qualification['complete_decisions']==116 and qualification['engineering_games']==2 and
        c.read(WORKER/'qualification_gate_binding.json')==gate, 'Previous production entrypoint had actual complete corrected-rule qualification')
    identity=c.read(WORKER/'runtime/runtime_identity.json');final=c.read(WORKER/'runtime/runtime_final.json')
    c.require(identity['route']=='cpu' and identity['weights_sha256']==a.prior.WEIGHTS and
        identity['checkpoint_sha256']==plan['runtime']['checkpoint']['sha256'] and identity['runtime_sources']==plan['source_guard'] and
        identity['threads']==identity['interop_threads']==1 and identity['scheduler']=='original8' and
        identity['maximum_logical_clients']==identity['physical_batch_unchanged']==8 and identity['implementation']=='eager' and
        identity['execution_policy_id']==c.EXECUTION_POLICY_ID and not identity['warnings'] and
        final['unchanged_weights_sha256']==identity['weights_sha256'] and final['service_failure'] and not final['warnings'],
        'Original CPU8 teacher identity preserved despite unsuccessful service closure')
    c.require(all(identity[k]==data.IDENTITY[k] for k in ('ruleset_id','catalogue_sha256','turn_contract')),'Corrected previous rules binding')
    c.require(c.read(WORKER/'cleanup.json')['success'] is False and not (WORKER/'runtime/final_identity.json').exists(),
              'Never fabricate successful original runtime closure')
    return execution,list(identities.values())


def _audit_prefix(item,journal,boundary):
    game=item['spec'];gid=game['game_id'];folder=CELL/'games'/gid;before=data.initial_snapshot(game);rows=[];callbacks=0
    started=c.read(folder/'started.json');assigned=item['assignment'];previous_pass=False
    c.require(started['game_id']==gid and started['client_id']==assigned['client_id'] and started['slot']==assigned['slot'] and
        assigned['assigned_perf_counter']<=started['started_perf_counter'], 'Saved game started after original assignment')
    for ply in range(item['prefix_count']):
        boundary();row,_,count=a.audit_decision(folder,game,ply,journal,before=before,owner=f'game:{gid}:{ply}',
            client=assigned['client_id'],slot=assigned['slot'])
        c.require(started['started_perf_counter']<=row['clock']['started_perf_counter'] and
            (not rows or rows[-1]['clock']['ended_perf_counter']<=row['clock']['started_perf_counter']), 'Original sequential complete move clocks')
        is_pass=row['action']['action_type']=='PASS'
        c.require(row['after']!=before and not(previous_pass and is_pass),'No prefix deadlock or fabricated transition')
        rows.append(row);before=row['after'];previous_pass=is_pass;callbacks+=count
    c.require(before==item['next_snapshot'],'Audited complete prefix ends exactly at saved boundary')
    return rows,callbacks


def _attribute_failed(item,journal,execution):
    folder=CELL/'games'/item['game_id'];ply=item['prefix_count'];name=f'{ply:06d}.json'
    attempt=c.read(folder/'attempts'/name);clock=c.read(folder/'clocks'/name)
    callbacks=journal.decision(attempt['client_id'],attempt['started_perf_counter'],
        owner=f"preserved_failed:{item['game_id']}:{ply}",slot=attempt['slot'])
    receipts=c.read(folder/'service_calls'/name)
    c.require(len(receipts)<=len(callbacks),'No returned callbacks missing from original raw journal')
    for receipt,entry in zip(receipts,callbacks):
        c.require({k:v for k,v in receipt.items() if k!='client_received_perf'}==entry and
            entry['complete_readback_perf']<=receipt['client_received_perf']<entry['deadline_perf'],
            'Successful callbacks of failed move retain exact original request/output and receipt clock')
    c.require(all(x['complete_readback_perf']<=execution['monotonic_end'] for x in callbacks), 'Partial callbacks existed before original failure exit')
    return dict(game_id=item['game_id'],ply=ply,status='original_failed_decision_preserved',attempt=item['failed_attempt'],
        clock=item['failed_clock'],outcome=item['failed_outcome'],completed_callbacks=len(callbacks),returned_callbacks=len(receipts),
        completed_decision=False,original_clock_reset=False,disclosed_new_attempts_allowed=1)


def audit_saved(plan,boundary=lambda:None,*,binding=None,binding_artifact=None):
    _bind(boundary)
    if binding is None:
        binding=_binding();binding_artifact=root.read(root.PLAN)['saved_partial135']
    else:
        c.require(binding_artifact is not None,'Explicit prelaunch saved boundary artifact required')
        c.verify([binding_artifact]);c.require(c.read(c.resolve(binding_artifact))==binding,'Prelaunch boundary exact saved bytes')
    validate_boundary(binding,plan,boundary)
    execution,absent=_saved_execution(plan);games=[];failed=[]
    Journal=audit.JournalIndex.__bases__[0]
    with Journal(WORKER/'runtime/inference.jsonl','cpu','original8',incomplete=True) as journal:
        warmup=a.audit_warmup(plan,CELL,journal)
        c.require(warmup['calls'][-1]['clock']['ended_perf_counter']<c.read(CELL/'started.json')['started_perf_counter'],
                  'Saved games follow complete service warmup')
        worker_last={};last_assignment=None
        for item in binding['games']:
            assignment=item['assignment'];client=assignment['client_id']
            c.require((last_assignment is None or last_assignment<=assignment['assigned_perf_counter']) and
                (client not in worker_last or worker_last[client]<=assignment['assigned_perf_counter']), 'Original ascending idle-client queue')
            rows,callbacks=_audit_prefix(item,journal,boundary);last_assignment=assignment['assigned_perf_counter']
            if item['completed']:
                _verify_document(item['spec'],rows,item['document'])
                finished=c.read(CELL/'games'/item['game_id']/'finished.json')
                c.require(rows[-1]['clock']['ended_perf_counter']<=finished['ended_perf_counter'],'Original completed game endpoint')
                worker_last[client]=finished['ended_perf_counter']
            else:failed.append(_attribute_failed(item,journal,execution))
            games.append(dict(game_id=item['game_id'],prefix_count=len(rows),callback_count=callbacks,
                prefix_rows=item['prefix_rows'],next_snapshot_sha256=c.digest(item['next_snapshot']),completed=item['completed']))
        accounting=journal.complete_attribution();metrics=journal.metrics('game:')
        initialization=a.audit_initialization(CONTAINER,WORKER/'runtime','cpu',warmup['calls'][0]['clock']['started_perf_counter'])
    validate_boundary(binding,plan,boundary)
    return dict(status='saved_c135_prefixes_verified',cell_id='c135',games=games,completed_games=8,unfinished_games=8,
        completed_moves=sum(x['prefix_count'] for x in games),failed_decisions=failed,journal=accounting,backend_metrics=metrics,
        initialization=initialization,owned_processes_absent=absent,model_calls=0,training_updates=0,
        boundary=binding_artifact,original_execution=c.artifact(CONTAINER/'execution.json'),
        original_runtime_closure=dict(success=False,final_weights_observed_unchanged=True,all_operations_closed=False,
            original_attempt_incomplete=True))


def _verify_document(game,rows,item):
    winner=guard.require_official_terminal_target(rows[-1]['after'],'completed');document=data.read_artifact(item)
    expected=dict(schema=data.GAME_SCHEMA,**data.IDENTITY,route='cpu',spec=game,reason='completed',
        initial=data.initial_snapshot(game),final=rows[-1]['after'],winner=winner,rows=document['rows'])
    c.require(document==expected and len(document['rows'])==len(rows) and
        all(data.read_artifact(ref)==row for ref,row in zip(document['rows'],rows,strict=True)), 'Complete original-plus-suffix target game document')
    return document,winner


def _finish_game(game,rows,refs,folder,assignment,*,resumed):
    winner=guard.require_official_terminal_target(rows[-1]['after'],'completed')
    document=dict(schema=data.GAME_SCHEMA,**data.IDENTITY,route='cpu',spec=game,reason='completed',
        initial=data.initial_snapshot(game),final=rows[-1]['after'],winner=winner,rows=refs)
    c.save(folder/'target_game.json',document)
    result=dict(status='naturally_completed',route='cpu',game_id=game['game_id'],specification=game,
        client_id=assignment['client_id'],slot=assignment['slot'],plies=len(rows),winner=winner,final=rows[-1]['after'],
        model_replay_selection=collector._legacy.old_games.replay_selection(rows),target_game=c.artifact(folder/'target_game.json'))
    c.save(folder/'result.json',result)
    finished=dict(game_id=game['game_id'],ended_perf_counter=time.perf_counter(),ended_unix=time.time(),
                  scope='amended game completed after preserved prefix and one disclosed unfinished-decision retry')
    c.save(folder/'finished.json',finished)
    outcome=dict(game_id=game['game_id'],status='completed',completed_moves=len(rows),finished=finished,no_retry=True,
                 disclosed_boundary_retry=1,completed_prefix_preserved=True)
    c.save(folder/'outcome.json',outcome)
    return outcome


def _continue_game(plan,runtime,adapter,item,folder,stopped,boundary):
    game=item['spec'];rows=[data.read_artifact(x) for x in item['prefix_rows']];refs=list(item['prefix_rows'])
    before=item['next_snapshot'];previous_pass=rows[-1]['action']['action_type']=='PASS';ply=item['prefix_count']
    c.save(folder/'resume.json',dict(game_id=game['game_id'],source_boundary=root.read(root.PLAN)['saved_partial135'],
        original_assignment=item['assignment'],prefix_rows=refs,first_new_ply=ply,before_sha256=c.digest(before),
        preserved_failed_attempt=item['failed_attempt'],disclosed_boundary_retry=1,original_clock_reset=False,
        new_attempt_uses_unchanged60_75_guard=True))
    try:
        for ply in range(item['prefix_count'],2000):
            boundary()
            if guard.decode_training_snapshot(before).terminal:break
            c.require(not stopped.is_set(),'Other resumed client failed; no further decisions')
            row=collector.decide_snapshot(runtime,adapter,before,game,ply,folder)
            is_pass=row['action']['action_type']=='PASS'
            c.require(row['after']!=before and not(previous_pass and is_pass),'No resumed deadlock or fabricated terminal')
            rows.append(row);refs.append(c.artifact(folder/'moves'/f'{ply:06d}.json'))
            before=row['after'];previous_pass=is_pass
        return _finish_game(game,rows,refs,folder,item['assignment'],resumed=True)
    except BaseException:
        stopped.set()
        failure=dict(game_id=game['game_id'],status='failed_amended_continuation',failed_ply=ply,
            preserved_prefix_moves=item['prefix_count'],completed_moves=len(rows),next_snapshot_sha256=c.digest(before),
            error=traceback.format_exc(),no_further_retry=True)
        c.save(folder/'failure.json',failure);raise


def _combined_documents(binding,folder):
    return [item['document'] if item['completed'] else c.artifact(folder/'games'/item['game_id']/'target_game.json')
            for item in binding['games']]


def _run_replays(plan,spec,runtime,folder,documents,boundary):
    receipts=[];started=time.perf_counter()
    for game,docref,item in zip(spec['specifications'],documents,_binding()['games'],strict=True):
        document=data.read_artifact(docref);rows=[data.read_artifact(x) for x in document['rows']]
        slot=(item['assignment']['slot']+1)%8;adapter=runtime.make_adapter('replay_'+game['game_id'],slot,512)
        try:
            for chosen in collector._legacy.old_games.replay_selection(rows):
                boundary();ply=chosen['ply'];source=rows[ply];destination=folder/'model_replays'/game['game_id']/f'{ply:06d}'
                row=collector.decide_snapshot(runtime,adapter,source['before'],game,ply,destination,kind='selected_model_replay')
                exact=row['result']==source['result'] and row['training_capture']==source['training_capture'] and row['after']==source['after']
                receipt=dict(game_id=game['game_id'],ply=ply,source=document['rows'][ply],replay=c.artifact(destination/'moves'/f'{ply:06d}.json'),
                    original_slot=item['assignment']['slot'],replay_slot=slot,other_clients_inactive=True,exact_full_result_capture_and_after=exact)
                c.save(destination/'comparison.json',receipt);c.require(exact,'Original selected full-output/target/transition exact replay')
                receipts.append(receipt)
        finally:adapter.close()
    c.require(0<len(receipts)<=48,'Exactly the original first-neural-band selection, at most three per game')
    summary=dict(status='exact_replays_passed',count=len(receipts),receipts=receipts,model_calls=len(receipts),wall_seconds=time.perf_counter()-started,inside_collection_timing=False)
    c.save(folder/'model_replay_summary.json',summary);return summary


def run_continuation(plan,runtime,folder,boundary=lambda:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    binding=_binding();validate_boundary(binding,plan,boundary);spec=_spec(plan)
    proof_path=c.OUT/'recovery_boundary/worker/saved_audit.json';proof=c.read(proof_path)
    c.require(proof['status']=='saved_c135_prefixes_verified' and proof['boundary']==root.read(root.PLAN)['saved_partial135'],
              'Actual independent saved-prefix qualification before continuation')
    collector._legacy.q.validate_runtime(runtime);collector.warmup_runtime(plan,runtime,folder)
    started=time.perf_counter();startwall=time.time();c.save(folder/'started.json',dict(cell_id='c135',started_perf_counter=started,started_unix=startwall,
        amended_partial_cell=True,original_cell_clock_reset=False))
    stopped=threading.Event();adapters={};threads=[];outcomes={};errors=[];lock=threading.Lock()
    try:
        for item in binding['games'][8:]:
            boundary();assigned=item['assignment']
            adapters[item['game_id']]=runtime.make_adapter(assigned['client_id'],assigned['slot'],512)
        def run_one(item):
            try:
                result=_continue_game(plan,runtime,adapters[item['game_id']],item,folder/'games'/item['game_id'],stopped,boundary)
                with lock:outcomes[item['game_id']]=result
            except BaseException:
                stopped.set()
                with lock:errors.append(dict(game_id=item['game_id'],error=traceback.format_exc()))
        for item in binding['games'][8:]:
            thread=threading.Thread(target=run_one,args=(item,),daemon=True,name='resume-'+item['game_id']);threads.append(thread);thread.start()
        while any(thread.is_alive() for thread in threads):
            boundary()
            for thread in threads:thread.join(timeout=.05)
        c.save(folder/'resume_outcomes.json',dict(outcomes=outcomes,errors=errors,original_failed_games_preserved=True))
        c.require(not errors and set(outcomes)==set(UNFINISHED),'All eight continuations complete; no further retry')
    except BaseException:
        stopped.set();runtime.abort();limit=time.monotonic()+80.
        for thread in threads:thread.join(timeout=max(0.,limit-time.monotonic()))
        c.require(not any(thread.is_alive() for thread in threads),'All continuation threads returned after failure')
        c.save(folder/'failure.json',dict(error=traceback.format_exc(),no_further_retry=True));raise
    finally:
        for adapter in adapters.values():adapter.close()
    gameplay_end=time.perf_counter();gameplay_end_wall=time.time();documents=_combined_documents(binding,folder)
    manifest=data.build_manifest(documents,spec['specifications'],route='cpu',cell_id='c135')
    c.save(folder/'corpus_manifest.json',manifest);targets=[]
    def sink(game,examples):
        targets.append(c.save(folder/'targets'/(game['game_id']+'.json'),dict(schema='gpu_collector_examples_v1',**data.IDENTITY,
            spec=game,examples=examples,official_terminal_only=True,teacher_simulations=512,training_updates=0)))
    loaded=data.load_corpus(manifest,expected_specs=spec['specifications'],boundary=boundary,sink=sink)
    c.save(folder/'loader_result.json',loaded)
    c.save(folder/'target_manifest.json',dict(schema='gpu_collector_loaded_targets_v1',**data.IDENTITY,
        corpus=c.artifact(folder/'corpus_manifest.json'),loader=c.artifact(folder/'loader_result.json'),targets=targets,count=loaded['eligible_roots'],training_updates=0))
    ended=time.perf_counter();summary=dict(status='completed',cell_id='c135',route='cpu',concurrency=8,games=16,
        plies=sum(x['plies'] for x in loaded['games']),eligible_roots=loaded['eligible_roots'],full_cohort_completed=True,
        corpus_produced=True,preserved_complete_games=8,preserved_prefix_moves=binding['completed_moves'],
        resumed_games=8,disclosed_boundary_retries=8,replacements=0,training_updates=0,original_attempt_incomplete=True,
        started_perf_counter=started,started_unix=startwall,ended_perf_counter=ended,ended_unix=time.time(),
        gameplay_end_perf_counter=gameplay_end,gameplay_end_unix=gameplay_end_wall,
        wall_seconds=ended-started,gameplay_seconds=gameplay_end-started,target_loader_seconds=ended-gameplay_end,
        original_cell_clock_reset=False,throughput_comparable=False)
    c.save(folder/'cohort.json',summary)
    replay=_run_replays(plan,spec,runtime,folder,documents,boundary)
    validate_boundary(binding,plan,boundary)
    return dict(summary,model_replay=replay)


def _game_proof(item,document_ref,rows,folder,callback_count):
    game=item['spec'];gid=game['game_id'];document,winner=_verify_document(game,rows,document_ref)
    examples=[];exclusions=Counter()
    for row in rows:
        example=a.independent_example(row,winner=winner,terminal=True)
        if example['eligible']:examples.append(example)
        else:exclusions[example['reason']]+=1
    c.require(examples,'Each completed game has usable search targets')
    target=dict(schema='gpu_collector_examples_v1',**data.IDENTITY,spec=game,examples=examples,official_terminal_only=True,teacher_simulations=512,training_updates=0)
    target_path=folder/'targets'/(gid+'.json');c.require(c.read(target_path)==target,'Every preserved and new target independently reconstructed')
    source=CELL/'games'/gid if item['completed'] else folder/'games'/gid
    ending=dict(status='naturally_completed',route='cpu',game_id=gid,specification=game,client_id=item['assignment']['client_id'],
        slot=item['assignment']['slot'],plies=len(rows),winner=winner,final=rows[-1]['after'],
        model_replay_selection=collector._legacy.old_games.replay_selection(rows),target_game=document_ref)
    c.require(c.read(source/'result.json')==ending,'Exact full amended game summary')
    finished=c.read(source/'finished.json');outcome=c.read(source/'outcome.json')
    c.require(outcome['status']=='completed' and outcome['completed_moves']==len(rows) and outcome['finished']==finished and
        finished['game_id']==gid and rows[-1]['clock']['ended_perf_counter']<=finished['ended_perf_counter'], 'Durable game completion after last successful decision')
    loaded=dict(spec=game,route='cpu',plies=len(rows),eligible_roots=len(examples),exclusions=dict(exclusions),winner=winner,
        batches=[dict(start=i,count=min(32,len(examples)-i),shape=[min(32,len(examples)-i),31,48]) for i in range(0,len(examples),32)],
        initial_sha256=c.digest(document['initial']),final_sha256=c.digest(document['final']),game_sha256=c.digest(document))
    return dict(game_id=gid,loaded=loaded,ending=ending,outcome=outcome,finished=finished,target=c.artifact(target_path),document=document_ref,
        trajectory_sha256=c.digest(dict(initial=document['initial'],actions=[row['action'] for row in rows],final=document['final'])),
        descriptive_latency=a.prior.descriptive_latency([x['clock']['elapsed_seconds'] for x in rows]),callback_count=callback_count)


def validate_combined_lifecycle(started,cohort,outcomes,games,execution,warmup_end):
    """Independent lifecycle accounting; amended throughput is never comparable."""
    c.require(started==dict(cell_id='c135',started_perf_counter=cohort['started_perf_counter'],
        started_unix=cohort['started_unix'],amended_partial_cell=True,original_cell_clock_reset=False),
        'Exact amended release receipt and cohort origin')
    c.require(execution['monotonic_origin']<=warmup_end<cohort['started_perf_counter']<=
        cohort['gameplay_end_perf_counter']<=cohort['ended_perf_counter']<=execution['monotonic_end'] and
        execution['unix_origin']<=cohort['started_unix']<=cohort['gameplay_end_unix']<=cohort['ended_unix']<=execution['unix_end'],
        'All amended release/gameplay/loader clocks stay inside actual owned execution')
    c.require(cohort['wall_seconds']==cohort['ended_perf_counter']-cohort['started_perf_counter']>0 and
        cohort['gameplay_seconds']==cohort['gameplay_end_perf_counter']-cohort['started_perf_counter']>0 and
        cohort['target_loader_seconds']==cohort['ended_perf_counter']-cohort['gameplay_end_perf_counter']>=0,
        'Exact complete amended elapsed/gameplay/loader arithmetic')
    expected={game['game_id']:game['outcome'] for game in games if game['game_id'] in UNFINISHED}
    c.require(set(expected)==set(UNFINISHED) and outcomes==dict(outcomes=expected,errors=[],original_failed_games_preserved=True),
        'Exactly eight durable successful continuation outcomes with original failures retained')
    for game in games:
        if game['game_id'] not in UNFINISHED:continue
        outcome=game['outcome'];finished=game['finished']
        c.require(outcome==dict(game_id=game['game_id'],status='completed',completed_moves=game['loaded']['plies'],
            finished=finished,no_retry=True,disclosed_boundary_retry=1,completed_prefix_preserved=True),
            'Exact amended completed game outcome, one disclosed retry')
        c.require(cohort['started_perf_counter']<=finished['ended_perf_counter']<=cohort['gameplay_end_perf_counter'] and
            cohort['started_unix']<=finished['ended_unix']<=cohort['gameplay_end_unix'], 'Every new durable game endpoint precedes loader')


def audit_combined(plan,container=None,boundary=lambda:None):
    _bind(boundary);binding=_binding();validate_boundary(binding,plan,boundary)
    container=Path(container) if container is not None else c.OUT/'recovery_games';folder=container/'worker/cell';spec=_spec(plan)
    saved_path=c.OUT/'recovery_boundary/worker/saved_audit.json';saved=c.read(saved_path)
    audit.stage_execution(plan,c.OUT/'recovery_boundary',mode='saved_boundary')
    c.require(saved['status']=='saved_c135_prefixes_verified' and saved['boundary']==root.read(root.PLAN)['saved_partial135'] and
        c.read(c.OUT/'recovery_boundary/worker/result.json')['result']==c.artifact(saved_path), 'Authenticated actual saved-prefix auditor')
    execution,absent=audit.stage_execution(plan,container,mode='recovery_games')
    c.require(c.read(container/'worker/cleanup.json')['success'] is True,'New continuation runtime closed successfully')
    documents=_combined_documents(binding,folder);games=[];receipts=[];new_callbacks=0
    with audit.JournalIndex(container/'worker/runtime/inference.jsonl','cpu','original8') as journal:
        warmup=a.audit_warmup(plan,folder,journal);cohort=c.read(folder/'cohort.json')
        for item,docref,saved_game in zip(binding['games'],documents,saved['games'],strict=True):
            boundary();rows=[data.read_artifact(x) for x in item['prefix_rows']];callbacks=saved_game['callback_count']
            c.require(saved_game['game_id']==item['game_id'] and saved_game['prefix_rows']==item['prefix_rows'] and
                saved_game['prefix_count']==len(rows),'Exact previously independently audited prefix')
            if not item['completed']:
                destination=folder/'games'/item['game_id'];doc=data.read_artifact(docref);refs=doc['rows']
                c.require(refs[:item['prefix_count']]==item['prefix_rows'] and len(refs)>item['prefix_count'], 'Preserve every original row and add a nonempty suffix')
                names=[f'{i:06d}.json' for i in range(item['prefix_count'],len(refs))]
                for key in EVIDENCE_DIRS:c.require(sorted(p.name for p in (destination/key).glob('*.json'))==names,'Only new suffix evidence '+key)
                resume=c.read(destination/'resume.json')
                c.require(resume==dict(game_id=item['game_id'],source_boundary=root.read(root.PLAN)['saved_partial135'],
                    original_assignment=item['assignment'],prefix_rows=item['prefix_rows'],first_new_ply=item['prefix_count'],
                    before_sha256=c.digest(item['next_snapshot']),preserved_failed_attempt=item['failed_attempt'],
                    disclosed_boundary_retry=1,original_clock_reset=False,new_attempt_uses_unchanged60_75_guard=True), 'Explicit single disclosed boundary retry')
                before=item['next_snapshot'];previous_pass=rows[-1]['action']['action_type']=='PASS'
                for ply in range(item['prefix_count'],len(refs)):
                    boundary();row,_,count=a.audit_decision(destination,item['spec'],ply,journal,before=before,
                        owner=f"game:{item['game_id']}:{ply}",client=item['assignment']['client_id'],slot=item['assignment']['slot'])
                    c.require(refs[ply]==c.artifact(destination/'moves'/f'{ply:06d}.json') and
                        cohort['started_perf_counter']<=row['clock']['started_perf_counter'] and
                        (ply==item['prefix_count'] or rows[-1]['clock']['ended_perf_counter']<=row['clock']['started_perf_counter']),
                        'New suffix original60/75 decisions follow new supplement release and sequential trajectory')
                    is_pass=row['action']['action_type']=='PASS';c.require(row['after']!=before and not(previous_pass and is_pass),'No suffix deadlock')
                    rows.append(row);before=row['after'];previous_pass=is_pass;callbacks+=count;new_callbacks+=count
            games.append(_game_proof(item,docref,rows,folder,callbacks))
        c.require({p.name for p in (folder/'games').iterdir() if p.is_dir()}==set(UNFINISHED), 'No completed game regenerated or replacement game')
        validate_combined_lifecycle(c.read(folder/'started.json'),cohort,c.read(folder/'resume_outcomes.json'),games,
            execution,warmup['calls'][-1]['clock']['ended_perf_counter'])
        manifest=data.build_manifest(documents,spec['specifications'],route='cpu',cell_id='c135')
        c.require(c.read(folder/'corpus_manifest.json')==manifest, 'Exact mixed-provenance corpus')
        loaded=dict(status='actual_collector_loader_passed',manifest_sha256=c.digest(manifest),games=[x['loaded'] for x in games],
            game_count=16,eligible_roots=sum(x['loaded']['eligible_roots'] for x in games),training_updates=0,
            actual_training_loader_used=True,historical_training_loader_used=False,**data.IDENTITY)
        c.require(c.read(folder/'loader_result.json')==loaded,'Independent all16game loader agreement')
        c.require(c.read(folder/'target_manifest.json')==dict(schema='gpu_collector_loaded_targets_v1',**data.IDENTITY,
            corpus=c.artifact(folder/'corpus_manifest.json'),loader=c.artifact(folder/'loader_result.json'),targets=[x['target'] for x in games],
            count=loaded['eligible_roots'],training_updates=0),'All combined target records')
        last=cohort['ended_perf_counter'];first=None
        for item,proof in zip(binding['games'],games,strict=True):
            document=data.read_artifact(proof['document']);slot=(item['assignment']['slot']+1)%8
            for chosen in proof['ending']['model_replay_selection']:
                ply=chosen['ply'];source=data.read_artifact(document['rows'][ply]);destination=folder/'model_replays'/item['game_id']/f'{ply:06d}'
                row,_,_=a.audit_decision(destination,item['spec'],ply,journal,before=source['before'],
                    owner=f"replay:{item['game_id']}:{ply}",client='replay_'+item['game_id'],slot=slot)
                c.require(row['result']==source['result'] and row['training_capture']==source['training_capture'] and row['after']==source['after'] and
                    last<=row['clock']['started_perf_counter'],'Exact isolated full model replay after collection')
                first=row['clock']['started_perf_counter'] if first is None else first;last=row['clock']['ended_perf_counter']
                receipt=dict(game_id=item['game_id'],ply=ply,source=document['rows'][ply],replay=c.artifact(destination/'moves'/f'{ply:06d}.json'),
                    original_slot=item['assignment']['slot'],replay_slot=slot,other_clients_inactive=True,exact_full_result_capture_and_after=True)
                c.require(c.read(destination/'comparison.json')==receipt,'Exact combined replay receipt');receipts.append(receipt)
        replay=c.read(folder/'model_replay_summary.json')
        c.require(0<len(receipts)<=48 and replay['status']=='exact_replays_passed' and replay['count']==replay['model_calls']==len(receipts) and
            replay['receipts']==receipts and replay['inside_collection_timing'] is False and replay['wall_seconds']>=last-first and
            len(list((folder/'model_replays').rglob('comparison.json')))==len(receipts),'Exactly all original selected model reproductions')
        closure=a.runtime_closure(plan,container/'worker/runtime','cpu',journal)
        initialization=a.audit_initialization(container,container/'worker/runtime','cpu',warmup['calls'][0]['clock']['started_perf_counter'])
        accounting=journal.complete_attribution();metrics=journal.metrics('game:')
    worker=c.read(container/'worker/worker_result.json')
    c.require(worker['status']=='passed' and worker['mode']=='recovery_games' and worker['result']==dict(cohort,model_replay=replay) and
        worker['reference_weights_unchanged'] is True, 'Actual completed continuation worker')
    c.require(cohort['status']=='completed' and cohort['games']==16 and cohort['preserved_complete_games']==8 and
        cohort['resumed_games']==cohort['disclosed_boundary_retries']==8 and cohort['preserved_prefix_moves']==binding['completed_moves'] and
        cohort['original_attempt_incomplete'] is True and cohort['original_cell_clock_reset'] is False and cohort['throughput_comparable'] is False and
        cohort['replacements']==cohort['training_updates']==0 and cohort['plies']==sum(x['loaded']['plies'] for x in games) and
        cohort['eligible_roots']==loaded['eligible_roots'], 'Amended scientific accounting never relabels original attempt')
    validate_boundary(binding,plan,boundary)
    return dict(status='passed',cell_id='c135',route='cpu',block=spec['block'],concurrency=8,full_cohort_verified=True,
        games=16,plies=cohort['plies'],eligible_roots=loaded['eligible_roots'],wall_seconds=None,games_per_hour=None,eligible_roots_per_hour=None,
        gameplay_seconds=None,target_loader_seconds=cohort['target_loader_seconds'],replay_seconds=replay['wall_seconds'],
        seed_decimals=[x['seed_decimal'] for x in spec['specifications']],trajectory_hashes=[x['trajectory_sha256'] for x in games],
        game_proofs=games,replay=replay,backend_metrics=metrics,journal=accounting,runtime_closure=closure,initialization=initialization,
        owned_processes_absent=absent,model_calls=0,training_updates=0,recovered_gameplay=True,resumed_gameplay=True,gameplay_retries=8,throughput_comparable=False,
        saved_prefix_audit=c.artifact(saved_path),saved_journal=saved['journal'],original_failed_runtime=saved['original_runtime_closure'],
        disclosed_boundary_retries=8,preserved_complete_games=8,preserved_prefix_moves=binding['completed_moves'],original_attempt_incomplete=True,
        supplemental_execution=c.artifact(container/'execution.json'),supplemental_wall_seconds=cohort['wall_seconds'])
