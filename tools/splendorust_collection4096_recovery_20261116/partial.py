"""Saved interrupted-cell science and separately identified verification recovery."""
from pathlib import Path
import time
from . import common as root, boundary as saved
from .collection import common as c, audit, collector, data, qualification
from tools.splendorust_gpu_probe_20261105.control import require_absent, native_fallback_messages

a = audit._legacy


def _bind(boundary):
    audit.set_boundary(boundary)
    # This private auditor belongs only to this additive recovery package.
    # Historical monotonic clocks are checked as saved evidence, never against a rebooted host.
    a.guard_audit_clock = lambda claim=None: boundary()


def _saved_execution(plan):
    execution = c.read(saved.CONTAINER/'execution.json'); owner = c.read(saved.CONTAINER/'owner.json')
    claim = c.read(saved.ORIGINAL/'claim.json'); phase = c.read(c.resolve(owner['phase_clock']))
    c.require(owner['identity'] == execution['identity'] and owner['mode'] == execution['mode'] == 'games' and
        owner['windows_job_assigned'] is True and owner['plan_sha256'] == execution['plan_sha256'] ==
        c.artifact(saved.ORIGINAL/'freeze.json')['sha256'], 'Exact original failed owned worker binding')
    c.require(owner['global_claim'] == c.artifact(saved.ORIGINAL/'claim.json') and phase['claim'] == owner['global_claim'],
              'Original claim and phase binding')
    for origin,end,deadline,start,limit in [('monotonic_origin','monotonic_end','monotonic_deadline','monotonic','monotonic_deadline'),
            ('unix_origin','unix_end','unix_deadline','unix_time','unix_deadline')]:
        c.require(execution[origin] == owner[origin] and execution[deadline] == owner[deadline] and
            claim[origin] <= phase[start] <= owner[origin] <= execution[end] < owner[deadline] <= phase[limit] and
            execution[end] < claim[origin]+144*3600, 'Preserved original dual-clock chain before shutdown')
    c.require(execution['status'] == 'failed' and execution['worker_exit_code'] == 1073807364 and
        execution['retries'] == 0 and not execution['cleanup_errors'] and execution['error'] and
        execution['peak_rss_bytes'] <= owner['maximum_rss_bytes'] <= c.MAX_RSS, 'Original interrupted execution remains failed')
    identities = { (x['pid'],x['created']):x for x in execution['observed_owned_identities']+[execution['identity']] }
    for identity in identities.values(): require_absent(identity)
    c.require(not native_fallback_messages(saved.WORKER/'stderr.log'), 'No unsupported original native fallback')
    startup = c.read(saved.WORKER/'runtime_sources.json')
    c.require(startup['sources'] == plan['source_guard'] and startup['identity'] == execution['identity'] and
        startup['original_claim'] == owner['global_claim'] and startup['original_phase'] == owner['phase_clock'],
        'Actual original once-startup corrected source guard')
    gate = c.read(saved.ORIGINAL/'collection/qualification_gate.json')
    c.verify([gate['audit']]); proof = c.read(c.resolve(gate['audit']))
    c.require(gate['passed'] is True and gate['collection_plan'] == c.artifact(saved.ORIGINAL_PLAN) and
        proof['status'] == 'passed' and proof['complete_decisions'] == 116 and proof['engineering_games'] == 2 and
        c.read(saved.WORKER/'qualification_gate_binding.json') == gate, 'Original qualified production admission')
    identity = c.read(saved.WORKER/'runtime/runtime_identity.json')
    expected = plan['runtime']
    c.require(identity['route'] == 'cpu' and identity['weights_sha256'] == a.prior.WEIGHTS and
        identity['checkpoint_sha256'] == expected['checkpoint']['sha256'] and
        identity['runtime_sources'] == plan['source_guard'] and identity['threads'] == identity['interop_threads'] == 1 and
        identity['execution_policy_id'] == c.EXECUTION_POLICY_ID and identity['scheduler'] == 'original8' and
        identity['maximum_logical_clients'] == identity['physical_batch_unchanged'] == 8 and
        identity['implementation'] == 'eager' and identity['compilation_fingerprint'] is None and
        identity['compilation_receipt'] is None and not identity['warnings'], 'Original unchanged CPU8 runtime identity')
    c.require(all(identity[key] == data.IDENTITY[key] for key in ('ruleset_id','catalogue_sha256','turn_contract')),
              'Exact original corrected rules and catalogue identity')
    return execution, list(identities.values())


def _replay_receipt(item, destination):
    return dict(game_id=item['game_id'], ply=item['ply'], source=item['source'],
        replay=c.artifact(Path(destination)/'moves'/f'{item["ply"]:06d}.json'),
        original_slot=item['original_slot'], replay_slot=item['replay_slot'],
        other_clients_inactive=True, exact_full_result_capture_and_after=True)


def _audit_replay(plan, item, destination, journal, *, previous=None):
    game = next(x for x in plan['cells'][41]['specifications'] if x['game_id'] == item['game_id'])
    source = c.read(c.resolve(item['source']))
    row, _, _ = a.audit_decision(destination, game, item['ply'], journal, before=source['before'],
        owner=f'replay:{item["game_id"]}:{item["ply"]}', client='replay_'+item['game_id'], slot=item['replay_slot'])
    c.require(row['result'] == source['result'] and row['training_capture'] == source['training_capture'] and
        row['after'] == source['after'], 'Entire original result, true visits and afterstate exact')
    if previous is not None:
        c.require(previous <= row['clock']['started_perf_counter'], 'Isolated replay decisions never overlap')
    receipt = _replay_receipt(item, destination)
    c.require(c.read(Path(destination)/'comparison.json') == receipt, 'Exact saved replay comparison')
    return receipt, row['clock']['ended_perf_counter']


def attribute_interrupted(journal, item, operation, execution):
    """Attribute only the authenticated incomplete suffix; never declare it successful."""
    destination = saved.CELL/'model_replays'/item['game_id']/f'{item["ply"]:06d}'
    attempt = c.read(destination/'attempts'/f'{item["ply"]:06d}.json')
    source = c.read(c.resolve(item['source']))
    from tools.external_strength_screen_20261018 import rules
    from tools import future_training_rules_20261021 as guard
    request = dict(player='search', observer=rules.observation_payload(guard.decode_training_snapshot(source['before'])))
    c.require(c.read(destination/'requests'/f'{item["ply"]:06d}.json') == request and
        attempt['request_sha256'] == c.digest(request) and attempt['before_sha256'] == c.digest(source['before']) and
        attempt['game_id'] == item['game_id'] and attempt['ply'] == item['ply'] and attempt['kind'] == 'selected_model_replay' and
        attempt['client_id'] == operation['client_id'] == 'replay_'+item['game_id'] and
        attempt['slot'] == item['replay_slot'] and attempt['no_retry'] is True and
        attempt['started_perf_counter'] == operation['started_perf'], 'Authenticated interrupted verification request')
    callbacks = journal.decision(operation['client_id'], operation['started_perf'],
        owner='interrupted_verification:g0666:40', slot=item['replay_slot'])
    expected_hashes = a.prior.callback_hashes(source['result'])
    actual_hashes = [c.digest([x['input_sha256'],x['output_sha256'],1]) for x in callbacks]
    c.require(callbacks and len(callbacks) <= len(expected_hashes) and actual_hashes == expected_hashes[:len(callbacks)] and
        all(x['complete_readback_perf'] <= execution['monotonic_end'] for x in callbacks),
        'Every partial callback is an exact prefix of the original verification target')
    return dict(status='interrupted_verification_preserved', active_operation=operation,
        attempt=c.artifact(destination/'attempts'/f'{item["ply"]:06d}.json'),
        partial_callbacks=len(callbacks), expected_complete_callbacks=len(expected_hashes),
        output_missing=True, completed_decision=False, original_clock_reset=False, scientific_games_retried=0)


def audit_partial(plan, boundary):
    """Check all 878 original moves/targets and every saved callback without models."""
    _bind(boundary); boundary()
    binding = root.read(root.resolve(root.read(root.PLAN)['saved_boundary']))
    saved.validate_boundary(binding)
    c.require(plan == c.read(saved.ORIGINAL_PLAN), 'Partial audit uses exact original scientific collection plan')
    execution, absent = _saved_execution(plan)
    spec = next(x for x in plan['cells'] if x['cell_id'] == 'c041')
    # The accepted scanner supports incomplete journals; raw source remains untouched.
    Journal = audit.JournalIndex.__bases__[0]
    with Journal(saved.WORKER/'runtime/inference.jsonl', 'cpu', 'original8', incomplete=True) as journal:
        warmup = a.audit_warmup(plan, saved.CELL, journal)
        started = c.read(saved.CELL/'started.json'); cohort = c.read(saved.CELL/'cohort.json')
        c.require(started['cell_id'] == spec['cell_id'] and cohort['started_perf_counter'] == started['started_perf_counter'] and
            cohort['started_unix'] == started['started_unix'] and
            warmup['calls'][-1]['clock']['ended_perf_counter'] < started['started_perf_counter'], 'Original gameplay clock follows warmups')
        files = sorted((saved.CELL/'assignments').glob('*.json'))
        c.require([p.name for p in files] == [f'{i:02d}.json' for i in range(16)], 'All original queue assignments')
        assignments = [c.read(p) for p in files]; games = []; worker_last = {}
        for i, (game, assigned) in enumerate(zip(spec['specifications'], assignments, strict=True)):
            boundary(); worker = assigned['worker_index']
            c.require(type(worker) is int and 0 <= worker < 8 and assigned['queue_index'] == i and
                assigned['game_id'] == game['game_id'] and assigned['client_id'] == f'worker_{worker}' and assigned['slot'] == worker and
                started['started_perf_counter'] <= assigned['assigned_perf_counter'] and
                (i == 0 or assignments[i-1]['assigned_perf_counter'] <= assigned['assigned_perf_counter']) and
                (worker not in worker_last or worker_last[worker] <= assigned['assigned_perf_counter']), 'Original idle-worker ascending queue')
            proof = a.audit_game(plan, spec, game, saved.CELL/'games'/game['game_id'], journal, assigned)
            games.append(proof); worker_last[worker] = proof['finished']['ended_perf_counter']
        c.require({p.name for p in (saved.CELL/'games').iterdir() if p.is_dir()} == {x['game_id'] for x in spec['specifications']}, 'No additional or replaced games')
        c.require(c.read(saved.CELL/'queue_outcomes.json') == dict(assignments=assignments,
            outcomes={x['game_id']:x['outcome'] for x in games}, errors=[], never_started=[]), 'All original queue outcomes')
        manifest = data.build_manifest([x['document'] for x in games], spec['specifications'], route='cpu', cell_id='c041')
        c.require(c.read(saved.CELL/'corpus_manifest.json') == manifest, 'Every original corpus source')
        targets = sum(x['loaded']['eligible_roots'] for x in games); plies = sum(x['loaded']['plies'] for x in games)
        loaded = dict(status='actual_collector_loader_passed',manifest_sha256=c.digest(manifest),games=[x['loaded'] for x in games],
            game_count=16,eligible_roots=targets,training_updates=0,actual_training_loader_used=True,historical_training_loader_used=False,**data.IDENTITY)
        c.require(c.read(saved.CELL/'loader_result.json') == loaded and targets == plies == 878, 'All 878 targets and loader results reconstructed')
        c.require(c.read(saved.CELL/'target_manifest.json') == dict(schema='gpu_collector_loaded_targets_v1', **data.IDENTITY,
            corpus=c.artifact(saved.CELL/'corpus_manifest.json'),loader=c.artifact(saved.CELL/'loader_result.json'),
            targets=[x['target'] for x in games],count=targets,training_updates=0), 'Full corrected target manifest')
        c.require(cohort['status'] == 'completed' and cohort['cell_id'] == 'c041' and cohort['route'] == 'cpu' and
            cohort['concurrency'] == 8 and cohort['games'] == 16 and cohort['plies'] == plies and cohort['eligible_roots'] == targets and
            cohort['full_cohort_completed'] is cohort['corpus_produced'] is True and
            cohort['retries'] == cohort['replacements'] == cohort['training_updates'] == 0, 'Completed original gameplay, zero retries')
        elapsed = cohort['ended_perf_counter']-started['started_perf_counter']
        c.require(elapsed == cohort['wall_seconds'] > 0 and cohort['games_per_hour'] == 16*3600/elapsed and
            cohort['eligible_roots_per_hour'] == targets*3600/elapsed and
            max(x['finished']['ended_perf_counter'] for x in games) <= cohort['gameplay_end_perf_counter'] <= cohort['ended_perf_counter'] and
            cohort['gameplay_seconds'] == cohort['gameplay_end_perf_counter']-started['started_perf_counter'] and
            cohort['target_loader_seconds'] == cohort['ended_perf_counter']-cohort['gameplay_end_perf_counter'] and
            execution['unix_origin'] <= started['started_unix'] <= cohort['gameplay_end_unix'] <= cohort['ended_unix'] <= execution['unix_end'],
            'Original complete gameplay and loader timing')
        receipts = []; last = cohort['ended_perf_counter']
        for item in binding['selected_replays'][:32]:
            boundary(); destination = saved.CELL/'model_replays'/item['game_id']/f'{item["ply"]:06d}'
            receipt,last = _audit_replay(plan,item,destination,journal,previous=last);receipts.append(receipt)
        c.require(last <= binding['active_operation']['started_perf'], 'Interrupted replay follows completed prefix')
        interrupted = attribute_interrupted(journal,binding['selected_replays'][32],binding['active_operation'],execution)
        initialization = a.audit_initialization(saved.CONTAINER,saved.WORKER/'runtime','cpu',warmup['calls'][0]['clock']['started_perf_counter'])
        metrics = journal.metrics('game:'); accounting = journal.complete_attribution()
    boundary(); saved.validate_boundary(binding)
    return dict(status='saved_interrupted_cell_science_verified', cell_id='c041',route='cpu',block=spec['block'],concurrency=8,
        full_cohort_verified=False,games=16,plies=plies,eligible_roots=targets,wall_seconds=elapsed,games_per_hour=16*3600/elapsed,
        eligible_roots_per_hour=targets*3600/elapsed,gameplay_seconds=cohort['gameplay_seconds'],target_loader_seconds=cohort['target_loader_seconds'],
        seed_decimals=[x['seed_decimal'] for x in spec['specifications']],trajectory_hashes=[x['trajectory_sha256'] for x in games],
        game_proofs=games,replay_prefix=receipts,interrupted_verification=interrupted,backend_metrics=metrics,journal=accounting,
        runtime_closure=dict(initial_identity_verified=True,final_weights_verified=False,original_runtime_interrupted=True,
            reason='Host shutdown during verification; original final runtime and successful-worker receipts do not exist'),
        initialization=initialization,owned_processes_absent=absent,model_calls=0,training_updates=0,
        saved_boundary=root.artifact(root.resolve(root.read(root.PLAN)['saved_boundary'])),original_failed_execution=c.artifact(saved.CONTAINER/'execution.json'))


def run_supplement(plan, runtime, folder, boundary):
    """Exactly one disclosed repeated verification plus the 15 unstarted selections."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    binding = root.read(root.resolve(root.read(root.PLAN)['saved_boundary'])); saved.validate_boundary(binding)
    collector._legacy.q.validate_runtime(runtime)
    collector.warmup_runtime(plan,runtime,folder)
    started = time.perf_counter(); receipts=[]; adapter=None; current=None
    try:
        for item in binding['selected_replays'][32:]:
            boundary()
            if item['game_id'] != current:
                if adapter is not None:adapter.close()
                current=item['game_id'];adapter=runtime.make_adapter('replay_'+current,item['replay_slot'],512)
            game = next(x for x in plan['cells'][41]['specifications'] if x['game_id']==current)
            source=c.read(c.resolve(item['source'])); destination=folder/'model_replays'/current/f'{item["ply"]:06d}'
            row=collector.decide_snapshot(runtime,adapter,source['before'],game,item['ply'],destination,kind='selected_model_replay')
            c.require(row['result']==source['result'] and row['training_capture']==source['training_capture'] and row['after']==source['after'],
                'Supplemental verification exactly reproduces original decision, targets and transition')
            receipt=_replay_receipt(item,destination);c.save(destination/'comparison.json',receipt);receipts.append(receipt)
    finally:
        if adapter is not None:adapter.close()
    summary=dict(status='exact_supplemental_replays_passed',count=16,receipts=receipts,model_calls=16,
        wall_seconds=time.perf_counter()-started,inside_collection_timing=False,
        repeated_interrupted_verification=1,unstarted_verification_decisions=15,scientific_games_retried=0,
        original_attempt_incomplete=True,original_clocks_reset=False)
    c.require(len(receipts)==16,'Exactly the remaining verification inventory');c.save(folder/'model_replay_summary.json',summary)
    return summary


def audit_supplement(plan, container=None, boundary=lambda:None):
    _bind(boundary);boundary();container=Path(container) if container is not None else c.OUT/'recovery_replays'
    folder=container/'worker';cell=folder/'cell'
    binding=root.read(root.resolve(root.read(root.PLAN)['saved_boundary']));saved.validate_boundary(binding)
    partial_path=c.OUT/'recovery_boundary/worker/saved_audit.json';partial=c.read(partial_path)
    audit.stage_execution(plan,c.OUT/'recovery_boundary',mode='saved_boundary')
    c.require(c.read(c.OUT/'recovery_boundary/worker/result.json')['result']==c.artifact(partial_path),
              'Actual completed independent partial auditor binds its saved proof')
    c.require(partial['status']=='saved_interrupted_cell_science_verified' and partial['games']==16 and
        partial['plies']==partial['eligible_roots']==878 and len(partial['replay_prefix'])==32 and
        partial['saved_boundary']==root.artifact(root.resolve(root.read(root.PLAN)['saved_boundary'])), 'Authenticated saved science acceptance')
    execution,absent=audit.stage_execution(plan,container,mode='supplemental_replays')
    c.require(c.read(folder/'cleanup.json')['success'] is True,'Supplemental runtime cleanup passed')
    with audit.JournalIndex(folder/'runtime/inference.jsonl','cpu','original8') as journal:
        warmup=a.audit_warmup(plan,cell,journal);receipts=[];first=None;last=warmup['calls'][-1]['clock']['ended_perf_counter']
        for item in binding['selected_replays'][32:]:
            boundary();destination=cell/'model_replays'/item['game_id']/f'{item["ply"]:06d}'
            receipt,last=_audit_replay(plan,item,destination,journal,previous=last);receipts.append(receipt)
            if first is None:first=c.read(destination/'clocks'/f'{item["ply"]:06d}.json')['started_perf_counter']
        expected={str((cell/'model_replays'/x['game_id']/f'{x["ply"]:06d}'/'comparison.json').resolve()) for x in binding['selected_replays'][32:]}
        c.require({str(p.resolve()) for p in (cell/'model_replays').rglob('comparison.json')}==expected,'Only16 supplemental decisions')
        summary=c.read(cell/'model_replay_summary.json')
        c.require(summary['status']=='exact_supplemental_replays_passed' and summary['count']==summary['model_calls']==16 and
            summary['receipts']==receipts and summary['repeated_interrupted_verification']==1 and
            summary['unstarted_verification_decisions']==15 and summary['scientific_games_retried']==0 and
            summary['original_attempt_incomplete'] is True and summary['original_clocks_reset'] is False and
            summary['inside_collection_timing'] is False and summary['wall_seconds']>=last-first>0,'Exact disclosed supplemental accounting')
        closure=a.runtime_closure(plan,folder/'runtime','cpu',journal)
        initialization=a.audit_initialization(container,folder/'runtime','cpu',warmup['calls'][0]['clock']['started_perf_counter'])
        accounting=journal.complete_attribution()
    worker=c.read(folder/'worker_result.json')
    c.require(worker['status']=='passed' and worker['mode']=='supplemental_replays' and worker['exit_code']==0 and
        worker['result']==summary and worker['reference_weights_unchanged'] is True,'Actual complete supplemental worker result')
    proof={k:v for k,v in partial.items() if k not in ('status','full_cohort_verified','replay_prefix')}
    proof.update(status='passed',full_cohort_verified=True,recovered_verification=True,partial_audit=c.artifact(partial_path),
        replay=dict(status='exact_replays_passed_with_disclosed_verification_recovery',count=48,
            receipts=partial['replay_prefix']+receipts,reused_original_replays=32,new_completed_replays=16,
            repeated_interrupted_verification=1,original_attempt_incomplete=True),
        replay_seconds=None,supplemental_replay_seconds=summary['wall_seconds'],
        supplemental_execution=c.artifact(container/'execution.json'),supplemental_journal=accounting,
        supplemental_runtime_closure=closure,supplemental_initialization=initialization,supplemental_owned_processes_absent=absent)
    boundary();saved.validate_boundary(binding);return proof
