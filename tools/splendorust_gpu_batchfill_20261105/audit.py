"""Independent saved-evidence audit of the fixed-position batch-filling study.

No models, native searches or games are created here. Unchanged, authenticated
scientific and clock validators are reused; scheduling and 32-case accounting
are reconstructed independently from raw receipts.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json
import math

from . import common as c
from tools.splendorust_gpu_throughput_20261105 import audit as prior
from tools.splendorust_gpu_throughput_20261105 import qualification as q
from tools.splendorust_gpu_throughput_20261105 import scientific

WEIGHTS = prior.WEIGHTS
read, canonical, digest, bits = prior.read, prior.canonical, prior.digest, prior.bits
timing, descriptive_latency = prior.timing, prior.descriptive_latency


def require(value, message):
    if not value:
        raise ValueError("BATCHFILL_AUDIT: " + message)


def validate_cases(cases):
    """Reconstruct all requests from bound, corrected, actually played positions."""
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules, audit as ledger
    from tools.splendorust_policy_head_20261104.capture_adapter import public_root_tokens, main_support
    require(len(cases) == 32 and [x['index'] for x in cases] == list(range(32)), "fixed32 source cases")
    source_ids = []
    for case in cases:
        i = case['index']
        require(case['case_id'] == f'p{i:03d}' and case['slot'] == i % 8, "fixed case/physical row")
        c.verify([case['source_before'], *case['expected_results'].values()])
        source = read(c.resolve(case['source_before']))
        require(case['snapshot'] == source['before'], "saved position uses actual corrected beforestate")
        state = guard.decode_training_snapshot(case['snapshot'])
        require(not state.terminal, "saved diagnostic position is unfinished")
        observer = rules.observation_payload(state)
        require(case['observer'] == observer and case['observer_sha256'] == digest(observer), "lawful exact observer")
        require(case['request'] == dict(player='search', observer=observer) and
                case['request_sha256'] == digest(case['request']), "exact original request")
        require(case['expected_root_tokens'] == public_root_tokens(observer), "actual corrected root tokens")
        support = main_support(observer)[2]
        require(case['legal_policy_indices'] == [j for j,v in enumerate(support) if v], "actual corrected legal support")
        for route in ('cpu','gpu'):
            move = read(c.resolve(case['expected_results'][route]))
            require(move['before'] == case['snapshot'], "paired source routes share exact beforestate")
            result = move['result']
            prior.root_output(case, result)
            require(move['after'] == ledger.transition(case['snapshot'], result['action']), "source legal transition")
            guard.check_training_transition(case['snapshot'], result['action'], move['after'])
            require(result['action'] == source['result']['action'] and move['after'] == source['after'], "paired exact source trajectory")
        path = Path(case['source_before']['path'])
        expected_ply = (0,12,24,40)[i//8]
        require(path.name == f'{expected_ply:06d}.json' and path.parent.name == 'moves' and
                path.parent.parent.name == f'g{i%8:03d}', "predetermined ply-first game order")
        source_ids.append(case['source_before']['path'])
    require(len(set(source_ids)) == 32, "32 distinct original positions")
    return {'positions':32, 'plies':[0,12,24,40], 'source_games':8, 'model_calls':0}


def read_journal(folder, route):
    events, scheduling, requests = [], [], {}
    path = Path(folder)/'inference.jsonl'
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                if event['kind'] == 'scheduler_dispatch':
                    scheduling.append(event)
                    continue
                require(event['kind'] == 'inference_batch' and event['route'] == route, "actual inference journal kind/route")
                rows = event['requests']; physical = 8 if route == 'gpu' else 1
                require(event['physical_batch'] == physical and 1 <= len(rows) <= physical, "unchanged physical batch shape")
                occupied = sorted(x['slot'] for x in rows)
                require(len(set(occupied)) == len(occupied) and all(type(x) is int and 0 <= x < 8 for x in occupied), "unique occupied physical rows")
                for row in rows:
                    require(row['route'] == route and row['batch_id'] == event['batch_id'] and row['native_batch'] == 1 and
                            row['physical_batch'] == physical and row['occupied_slots'] == occupied, "per-request actual batch identity")
                    require(row['submitted_perf'] <= row['dispatch_perf'] < row['complete_readback_perf'] < row['deadline_perf'] and
                            row['queue_seconds'] == row['dispatch_perf']-row['submitted_perf'] and
                            row['forward_roundtrip_seconds'] == row['complete_readback_perf']-row['dispatch_perf'], "raw inference clocks/deadlines")
                    require(row['request_id'] not in requests, "each request completes once")
                    requests[row['request_id']] = row
                events.append(event)
    require(sorted(e['batch_id'] for e in events) == list(range(1,len(events)+1)), "complete physical batch IDs")
    return events, requests, scheduling


def audit_scheduler(events, requests, scheduling, scheduler, *, incomplete=False):
    """Rebuild oldest-per-row selection and the retained FIFO prefix."""
    require(scheduler in ('original8','row_fifo32'), "declared scheduler")
    if scheduler == 'original8':
        require(not scheduling, "unchanged control has no candidate scheduling")
        return {'scheduler':scheduler,'dispatches':len(events),'fifo_applicable':False}
    require(len(scheduling) == len(events) or incomplete and len(scheduling) == len(events)+1,
            "every candidate dispatch is paired with its physical completion")
    by_batch = {e['batch_id']:e for e in events}
    deferred, seen, selected_all = [], {}, set()
    last_new_ticket = 0
    for index, event in enumerate(scheduling, 1):
        require(event['scheduler'] == scheduler and event['scheduler_dispatch_index'] == index, "candidate dispatch sequence")
        ready = event['ready']
        require(ready and ready[:len(deferred)] == deferred, "older deferred requests stay ahead of new arrivals")
        require(len(ready)<=32 and len({x['client_id'] for x in ready})==len(ready), "bounded backlog, one outstanding request per client")
        ids = [x['request_id'] for x in ready]
        tickets = [x['enqueue_sequence'] for x in ready]
        require(len(set(ids)) == len(ids) and tickets == sorted(tickets) and len(set(tickets)) == len(tickets), "ordered unique ready queue")
        for item in ready[len(deferred):]:
            ticket = item['enqueue_sequence']
            require(type(ticket) is int and ticket == last_new_ticket+1 and item['request_id'] not in seen,
                    "every newly enqueued request appears once in ticket order")
            expiry_allowed = incomplete and index == len(scheduling) and index not in by_batch
            require(item['submitted_perf'] <= event['observed_perf'] and item['submitted_perf'] < item['deadline_perf'] and
                    (expiry_allowed or event['observed_perf'] < item['deadline_perf']), "ready request original deadline")
            last_new_ticket = ticket; seen[item['request_id']] = item
        rows, selected, keep = set(), [], []
        for item in ready:
            require(type(item['slot']) is int and 0 <= item['slot'] < 8 and item['request_id'] not in selected_all,
                    "valid pending row, never redispatch completed request")
            if item['slot'] in rows:
                keep.append(item)
            else:
                rows.add(item['slot']); selected.append(item)
        require(event['selected_request_ids'] == [x['request_id'] for x in selected] and
                event['deferred_request_ids'] == [x['request_id'] for x in keep], "exact earliest-per-row scheduling")
        selected_ids = {x['request_id'] for x in selected}
        selected_all.update(selected_ids)
        if index in by_batch:
            batch = by_batch[index]
            require([x['request_id'] for x in batch['requests']] == [x['request_id'] for x in selected], "scheduled request order equals actual physical batch")
            for row,item in zip(batch['requests'],selected):
                require(all(row[k] == item[k] for k in ('request_id','client_id','slot','submitted_perf','deadline_perf')) and
                        event['observed_perf'] <= row['dispatch_perf'], "scheduled identity/clock binds actual forward")
        deferred = keep
    if not incomplete:
        require(not deferred and selected_all == set(requests) and set(seen) == set(requests), "all enqueued requests complete; none dropped")
    else:
        require(set(requests) <= selected_all <= set(seen), "failed candidate retains a valid completion prefix")
    return dict(scheduler=scheduler,dispatches=len(scheduling),enqueued=len(seen),completed=len(requests),
                remaining_deferred=len(deferred),fifo_applicable=True,exact_oldest_per_row=True)


def validate_client_sequences(requests):
    groups = {}
    for row in requests.values(): groups.setdefault(row['client_id'],[]).append(row)
    for client, values in groups.items():
        rows = sorted(values,key=lambda x:int(x['request_id'].rsplit(':',1)[1]))
        require([x['request_id'] for x in rows] == [client+':'+str(i+1) for i in range(len(rows))], "whole-client callback sequence has no gaps/retries")
        require(all(a['complete_readback_perf'] <= b['submitted_perf'] for a,b in zip(rows,rows[1:])), "one outstanding callback per search client")
        require(len({x['slot'] for x in rows}) == 1, "client retains physical row across waves")
    return groups


def case_requests(case, cell, requests, started):
    client = f"worker_{case['index'] % cell['concurrency']}"
    result = sorted((r for r in requests.values() if r['client_id'] == client and r['deadline_perf'] == started+60.),
                    key=lambda x:int(x['request_id'].rsplit(':',1)[1]))
    require(all(r['slot'] == case['slot'] and r['submitted_perf'] >= started for r in result), "case callback row and original decision origin")
    return client,result


def qualified_rate(status, rows, seconds, expected=32):
    eligible = status == 'passed' and len(rows) == expected and all(x['completed'] and x['baseline_exact'] for x in rows)
    if not eligible:
        return dict(eligible=False,completed_decisions_per_second=None)
    require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0, "positive complete-workload time")
    return dict(eligible=True,completed_decisions_per_second=expected/seconds)


def aggregate_cells(cells):
    names = ('cpu8','gpu8','gpu16','gpu32')
    result = {}
    for name in names:
        rows = [x for x in cells if x['configuration'] == name]
        complete = len(rows) == 2 and {x['block'] for x in rows} == {1,2} and all(x['eligible'] for x in rows)
        seconds = sum(x['wall_seconds'] for x in rows) if complete else None
        result[name] = dict(complete_both_blocks=complete,completed_decisions=64 if complete else None,
                           wall_seconds=seconds,decisions_per_second=64/seconds if complete else None)
    for name in ('gpu16','gpu32'):
        for control in ('cpu8','gpu8'):
            a,b = result[name]['decisions_per_second'],result[control]['decisions_per_second']
            result[name][f'ratio_to_{control}'] = a/b if a is not None and b is not None else None
    return result


def audit_stage_execution(plan, container, *, mode, cell=None):
    container = Path(container)
    owner, execution = read(container/'owner.json'), read(container/'execution.json')
    stage_id = 'qualification' if mode == 'qualification' else cell['cell_id']
    seconds = plan['worker_limits'][mode+'_seconds']
    require(owner['identity'] == execution['identity'] and owner['stage_id'] == execution['stage_id'] == stage_id and
            owner['mode'] == execution['mode'] == mode and owner['plan_sha256'] == execution['plan_sha256'] == c.sha(c.PLAN),
            "exact frozen stage ownership")
    require(owner['maximum_seconds'] == execution['maximum_seconds'] == seconds and
            owner['monotonic_origin'] == execution['monotonic_origin'] and owner['unix_origin'] == execution['unix_origin'] and
            0 <= execution['monotonic_end']-execution['monotonic_origin'] < seconds and
            0 <= execution['unix_end']-execution['unix_origin'] < seconds, "actual unextended stage dual-clock cap")
    require(execution['retries'] == 0 and not execution['cleanup_errors'] and not execution['native_fallback_lines'],
            "no retry, clean exact process cleanup and no native fallback")
    require(0 <= execution['peak_rss_bytes'] < plan['maximum_worker_rss_bytes'], "observed worker tree stayed inside actual RSS bound")
    absent = prior.require_processes_absent(execution['observed_owned_identities']+[execution['identity']])
    worker = read(container/'worker/worker_identity.json')
    c.verify([owner['global_claim']]); claim = read(c.resolve(owner['global_claim']))
    require(worker['global_claim'] == owner['global_claim'] and claim['plan_sha256'] == execution['plan_sha256'] and
            claim['model_seconds'] == plan['model_seconds'] == c.MODEL_SECONDS and
            claim['execution_policy_id'] == worker['execution_policy_id'] == c.EXECUTION_POLICY_ID and
            claim['monotonic_origin'] <= execution['monotonic_origin'] <= execution['monotonic_end'] < claim['monotonic_origin']+c.MODEL_SECONDS and
            claim['unix_origin'] <= execution['unix_origin'] <= execution['unix_end'] < claim['unix_origin']+c.MODEL_SECONDS,
            "one original global model cutoff in every worker")
    require(dict(pid=worker['pid'],created=worker['created']) == execution['identity'] and worker['mode'] == mode and
            worker['cell_id'] == (None if cell is None else cell['cell_id']) and worker['worker_affinity'] == plan['worker_affinity'],
            "actual stage worker identity and affinity")
    lib = prior.load_saved_library(container)
    stderr = (container/'worker/stderr.log').read_text(encoding='utf-8',errors='replace')
    require(not any(lib.fallback(line) for line in stderr.splitlines()), "independent native fallback scan")
    return execution, absent


def audit_runtime(plan, folder, route, events, requests, scheduling, scheduler, *, incomplete=False, schedule_incomplete=False):
    closure = prior.audit_runtime(plan, folder, route, events, requests, incomplete=incomplete)
    identity = read(Path(folder)/'runtime_identity.json')
    require(identity['scheduler'] == scheduler and identity['maximum_logical_clients']==(32 if scheduler=='row_fifo32' else 8) and
            identity['physical_batch_unchanged']==8, "actual runtime uses frozen scheduler and unchanged physical shape")
    schedule = audit_scheduler(events,requests,scheduling,scheduler,incomplete=incomplete or schedule_incomplete)
    validate_client_sequences(requests)
    return dict(**closure,scheduling=schedule)


def audit_case(plan, case, cell, folder, requests, *, killed=False):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import audit as ledger
    folder = Path(folder)
    require(read(folder/'case.json') == case and read(folder/'request.json') == case['request'], "prospective case/request retained")
    if not (folder/'attempt.json').exists():
        require(not (folder/'clock.json').exists() and not (folder/'result.json').exists(), "unstarted case has no clock or result")
        return dict(case_id=case['case_id'],attempted=False,completed=False,callbacks=0,baseline_exact=False)
    attempt = read(folder/'attempt.json'); started = attempt['started_perf_counter']
    require(attempt['case_id'] == case['case_id'] and attempt['request_sha256'] == case['request_sha256'] and
            case['index'] in cell['groups'][attempt['group_index']], "original durable attempt/request/group")
    client, rows = case_requests(case,cell,requests,started)
    clock = read(folder/'clock.json') if (folder/'clock.json').exists() else None
    if clock is not None:
        success = prior.validate_clock(clock)
        require(clock['started_perf_counter'] == started, "original complete-response clock origin")
    else:
        require(killed, "missing clock only after authenticated supervisor stop")
        success = False
    if rows:
        require(rows[0]['input_sha256'] == digest(case['expected_root_tokens']), "first actual callback uses corrected saved root")
    returned = read(folder/'service_calls.json') if (folder/'service_calls.json').exists() else []
    require(len(returned) <= len(rows), "returned receipts form completed callback prefix")
    for item,row in zip(returned,rows):
        require({k:v for k,v in item.items() if k != 'client_received_perf'} == row and
                row['complete_readback_perf'] <= item['client_received_perf'] < row['deadline_perf'], "exact returned callback receipt")
    result_path = folder/'result.json'
    if result_path.exists() and (folder/'after.json').exists():
        require(success, "accepted result has successful original clock")
        result = read(result_path)
        output = prior.root_output(case,result)
        expected = read(c.resolve(case['expected_results'][cell['route']]))
        require(prior.exact_baseline(case,result,expected['result']), "full same-route saved scientific result reproduced")
        require(result == expected['result'], "unchanged raw scientific result; no extra normalization")
        require(read(folder/'root.json') == q.validate_fixed_result(case,result), "saved root metadata rederived")
        raw = read(folder/'raw_result.json')
        require(raw['result'] == result and raw['observer'] == case['observer'] and raw['action'] == result['action'] and
                raw['diagnostics'] == result['diagnostics'], "prevalidation actual result bound")
        after = read(folder/'after.json')
        require(after == ledger.transition(case['snapshot'],result['action']) == expected['after'], "independent corrected afterstate and saved reference")
        guard.check_training_transition(case['snapshot'],result['action'],after)
        require(len(returned) == len(rows) and prior.callback_hashes(result) ==
                [digest([r['input_sha256'],r['output_sha256'],1]) for r in rows], "all scientific callbacks bind actual ordered service responses")
        require(all(row['complete_readback_perf'] <= clock['ended_perf_counter'] for row in rows), "all callbacks finish inside original decision")
        return dict(case_id=case['case_id'],client_id=client,attempted=True,completed=True,callbacks=len(rows),baseline_exact=True,
                    elapsed_seconds=clock['elapsed_seconds'],result=result,root_output83=output,clock=clock)
    if not killed:
        require(not success and (folder/'failure.json').exists(), "failed request preserves failure and unsuccessful clock")
    if (folder/'raw_result.json').exists():
        raw = read(folder/'raw_result.json')
        require(raw['observer'] == case['observer'], "failed raw result observer")
        prior.root_output(case,raw['result'])
        prior.exact_baseline(case,raw['result'],read(c.resolve(case['expected_results'][cell['route']]))['result'])
        require(prior.callback_hashes(raw['result']) == [digest([r['input_sha256'],r['output_sha256'],1]) for r in rows],
                "failed unaccepted raw result binds complete callback evidence")
    return dict(case_id=case['case_id'],client_id=client,attempted=True,completed=False,callbacks=len(rows),baseline_exact=False,clock=clock)


def audit_cell(plan, cell, worker_folder):
    folder = Path(worker_folder)
    execution, absent = audit_stage_execution(plan,folder.parent,mode='cell',cell=cell)
    require(execution['status'] in ('completed','deadline_failed'), "only valid clock failures isolate a cell")
    require(read(folder/'runtime_sources.json')['sources'] == plan['source_guard'], "actual startup source guard")
    validate_cases(plan['cases'])
    events,requests,scheduling = read_journal(folder/'runtime',cell['route'])
    warmup = prior.audit_warmup(plan,folder,requests)
    killed = execution['outer_deadline_observed'] is not None
    closure = audit_runtime(plan,folder/'runtime',cell['route'],events,requests,scheduling,cell['scheduler'],incomplete=killed,
                            schedule_incomplete=execution['status']!='completed')
    rows,outcomes = [],{}
    for case in plan['cases']:
        path = folder/'cell'/case['case_id']
        row = audit_case(plan,case,cell,path,requests,killed=killed)
        if (path/'outcome.json').exists():
            outcome = read(path/'outcome.json')
            require(outcome['case_id'] == case['case_id'] and outcome['index'] == case['index'] and outcome['client_id'] == row['client_id'] and
                    outcome['decision_started'] is True and outcome['clock'] == row['clock'] and
                    outcome['request_sha256'] == case['request_sha256'] and outcome['no_retry'] is True, "case outcome binds original request, client and clock")
            if row['completed']:
                require(outcome['status'] == 'completed' and outcome['result_sha256'] == digest(row['result']) and
                        outcome['after_sha256'] == digest(read(path/'after.json')), "successful outcome hashes")
            outcomes[case['index']] = outcome
        elif row['attempted']:
            require(killed, "all ordinary returned cases have durable outcomes")
        row.pop('result',None); rows.append(row)
    require(len(requests) == 5+sum(x['callbacks'] for x in rows), "every backend callback belongs to warmup or one case")
    attempts=[]
    for case,row in zip(plan['cases'],rows):
        if row['attempted']:
            attempt=read(folder/'cell'/case['case_id']/'attempt.json')
            attempts.append((f"worker_{case['index']%cell['concurrency']}",case['slot'],attempt['started_perf_counter']))
    for event in scheduling:
        for item in event['ready']:
            if item['client_id']=='warmup':
                matching=[x['clock']['started_perf_counter'] for x in warmup['calls'] if
                          item['deadline_perf']==x['clock']['started_perf_counter']+60.]
            else:
                matching=[started for client,slot,started in attempts if item['client_id']==client and item['slot']==slot and
                          item['deadline_perf']==started+60. and item['submitted_perf']>=started]
            require(len(matching)==1, "including unfinished queued requests, every scheduler deadline binds one original operation")
    if killed:
        return audit_killed_cell(plan,cell,folder,execution,absent,closure,rows,requests,events)
    require(execution['worker_exit_code'] == (0 if execution['status']=='completed' else 20), "actual exit convention")
    require(read(folder/'cleanup.json')['success'] is True, "returned worker cleanup")
    summary = read(folder/'cell/cell_summary.json'); worker = read(folder/'worker_result.json')
    require(worker['mode'] == 'cell' and worker['cell_id'] == cell['cell_id'] and worker['result'] == summary and
            worker['exit_code'] == execution['worker_exit_code'] and worker['reference_weights_unchanged'] is True and worker['no_retry'] is True,
            "worker summary binds actual cell outcome")
    require(summary['cell_id'] == cell['cell_id'] and summary['route'] == cell['route'] and
            summary['concurrency'] == cell['concurrency'] and summary['planned_case_count'] == 32 and
            summary['status'] == ('passed' if execution['status']=='completed' else 'deadline_failed') and
            summary['no_retry'] is True and summary['game_continuations'] == 0, "fixed cell identity and no new games")
    groups = summary['groups']; expected_attempts=[]
    require(groups and len(groups) <= len(cell['groups']), "fixed group prefix")
    for group_index,group in enumerate(groups):
        indexes=cell['groups'][group_index]; expected_attempts.extend(indexes)
        path=folder/'cell/groups'/f'{group_index:03d}'
        require(group == read(path/'summary.json') and group['group_index'] == group_index and group['case_indexes'] == indexes and
                group['case_ids'] == [plan['cases'][i]['case_id'] for i in indexes] and
                group['outcomes'] == [outcomes[i] for i in indexes], "all frozen group members and outcomes")
        failures={plan['cases'][i]['case_id']:read(folder/'cell'/plan['cases'][i]['case_id']/'failure.json') for i in indexes if not rows[i]['completed']}
        prior.validate_group_failures(group['outcomes'],failures)
        require(read(path/'start.json') == dict(group_index=group_index,case_indexes=indexes,
                started_perf_counter=group['started_perf_counter'],started_unix=group['started_unix']), "original group barrier receipt")
        require(group['wall_seconds'] == group['ended_perf_counter']-group['started_perf_counter'] > 0 and
                group['attempted'] == len(indexes) and group['completed'] == sum(rows[i]['completed'] for i in indexes), "raw group counts and clock arithmetic")
        for i in indexes:
            clock=outcomes[i]['clock']
            require(group['started_perf_counter'] <= clock['started_perf_counter'] and clock['ended_perf_counter'] <= group['ended_perf_counter'],
                    "decision wholly inside scheduled group")
        if group_index: require(groups[group_index-1]['ended_perf_counter'] <= group['started_perf_counter'], "nonoverlapping ordered waves")
        require(group['status'] == ('passed' if all(rows[i]['completed'] for i in indexes) else 'deadline_failed') and
                (group_index == len(groups)-1 or group['status']=='passed'), "cell stops after first failed group")
    require(set(outcomes) == set(expected_attempts) and summary['attempted_case_ids'] == [x['case_id'] for x in rows if x['attempted']] and
            summary['completed_case_ids'] == [x['case_id'] for x in rows if x['completed']] and
            summary['never_started_case_ids'] == [x['case_id'] for x in rows if not x['attempted']], "attempted/completed/unstarted partition")
    require(summary['started_perf_counter'] == groups[0]['started_perf_counter'] and summary['started_unix'] == groups[0]['started_unix'] and
            summary['ended_perf_counter'] >= groups[-1]['ended_perf_counter'] and
            summary['wall_seconds'] == summary['ended_perf_counter']-summary['started_perf_counter'] > 0 and
            warmup['calls'][-1]['clock']['ended_perf_counter'] < summary['started_perf_counter'], "complete primary interval includes waves and excludes warmup")
    ready = read(folder/'cell/adapters_ready.json')
    require(ready['count'] == cell['concurrency'] and ready['prepared_before_panel'] is True and
            ready['observed_perf_counter'] < summary['started_perf_counter'] and
            ready['clients'] == [dict(index=i,client_id=f'worker_{i}',slot=i%8) for i in range(cell['concurrency'])],
            "all fixed clients prepared before measured panel and reused")
    require(summary['intentional_intergroup_gap_seconds'] == 0. and summary['full_cell_completed'] is
            (summary['status']=='passed' and len(outcomes)==32), "complete32 requirement and no added gap")
    if summary['status']=='passed': require(closure['service_failure'] is None, "no latent backend failure in successful cell")
    panel_events=[e for e in events if e['requests'][0]['client_id']!='warmup']
    panel_requests=[r for e in panel_events for r in e['requests']]
    require(len(panel_requests) == len(requests)-5 and all(r['client_id'].startswith('worker_') and
            summary['started_perf_counter'] <= r['submitted_perf'] and r['complete_readback_perf'] <= summary['ended_perf_counter']
            for r in panel_requests), "actual panel callbacks all inside primary interval")
    rate=qualified_rate(summary['status'],rows,summary['wall_seconds'])
    return dict(status=summary['status'],cell_id=cell['cell_id'],configuration=cell['configuration'],block=cell['block'],
        route=cell['route'],concurrency=cell['concurrency'],scheduler=cell['scheduler'],**rate,case_rows=rows,
        wall_seconds=summary['wall_seconds'],runtime_closure=closure,callback_count=len(panel_requests),warmup_calls=5,
        backend_metrics=prior.backend_metrics(panel_events,panel_requests),owned_processes_absent=absent,
        complete_response_seconds=timing([x['elapsed_seconds'] for x in rows if x['completed']]) if any(x['completed'] for x in rows) else None,
        descriptive_latency=descriptive_latency([x['elapsed_seconds'] for x in rows if x['completed']]),model_calls=0,new_games=0)


def audit_killed_cell(plan,cell,folder,execution,absent,closure,rows,requests,events):
    outer=execution['outer_deadline_observed']
    require(execution['status']=='deadline_failed' and execution['worker_exit_code'] != 0 and outer['kind']=='ordinary' and
            abs(outer['deadline_perf']-outer['started_perf']-75.) < 1e-5 and
            abs(outer['deadline_unix']-outer['started_unix']-75.) < 1e-5 and
            (outer['observed_perf'] >= outer['deadline_perf'] or outer['observed_unix'] >= outer['deadline_unix']), "authenticated original75s expiry")
    matches=[]
    for case,row in zip(plan['cases'],rows):
        if row['attempted']:
            path=folder/'cell'/case['case_id']; attempt=read(path/'attempt.json')
            group=read(folder/'cell/groups'/f"{attempt['group_index']:03d}"/'start.json')
            require(group['case_indexes']==cell['groups'][attempt['group_index']] and
                    group['started_perf_counter'] <= attempt['started_perf_counter'], "killed attempt follows its original barrier")
            if row['client_id']==outer['client_id'] and attempt['started_perf_counter']==outer['started_perf']:
                matches.append(case['case_id'])
    require(len(matches)==1, "supervisor stopped exactly one identified original active case")
    return dict(status='deadline_failed',cell_id=cell['cell_id'],configuration=cell['configuration'],block=cell['block'],
        route=cell['route'],concurrency=cell['concurrency'],scheduler=cell['scheduler'],eligible=False,
        completed_decisions_per_second=None,wall_seconds=None,case_rows=rows,runtime_closure=closure,
        supervisor_expiry=outer,owned_processes_absent=absent,callback_count=len(requests)-5,warmup_calls=5,
        unfinished_callback_and_final_weight_receipts_unavailable=True,model_calls=0,new_games=0)


def audit_collisions(plan, folder, requests):
    folder=Path(folder); summary=read(folder/'summary.json'); total=0; groups=[]; used=set()
    require(summary['status']=='actual_collision_service_qualified' and summary['route']=='gpu' and
            summary['scheduler']=='row_fifo32' and summary['exact_source_outputs'] is True and
            summary['search_decisions']==summary['games']==summary['training_updates']==0 and summary['no_retry'] is True,
            "actual candidate service qualification without search/games")
    for count in (16,32):
        group_folder=folder/f'clients_{count}'; group=read(group_folder/'summary.json')
        require(group['clients']==count and group['callbacks']==count*2 and group['status']=='passed' and
                group['source_case_ids']==[plan['cases'][i]['case_id'] for i in range(count)] and
                group['repetitions']==2 and group['same_rows_collide'] is True and group['no_retry'] is True,
                "fixed16/32 collision cohort")
        require(read(group_folder/'start.json')==dict(clients=count,started_perf_counter=group['started_perf_counter'],started_unix=group['started_unix']),
                "original collision barrier")
        for index in range(count):
            client=f'collision{count}_{index:02d}'; case=plan['cases'][index]
            expected=prior.root_output(case,read(c.resolve(case['expected_results']['gpu']))['result'])
            for repetition in range(2):
                path=group_folder/client; value=read(path/f'result_{repetition}.json'); clock=read(path/f'clock_{repetition}.json')
                require(value['clock']==clock and prior.validate_clock(clock) and
                        group['started_perf_counter'] <= clock['started_perf_counter'] <= clock['ended_perf_counter'] <= group['ended_perf_counter'],
                        "actual bounded collision call inside its group")
                require(value['client_id']==client and value['client_index']==index and value['slot']==index%8 and
                        value['repetition']==repetition and value['case_id']==case['case_id'] and value['inputs']==case['expected_root_tokens'] and
                        value['expected_result']==case['expected_results']['gpu'] and value['outputs']==[expected] and
                        bits(value['outputs'][0])==bits(expected), "collision response exact accepted root bits and ownership")
                receipt=value['receipt']; key=client+':'+str(repetition+1)
                require(key in requests and key not in used, "each collision callback exactly once")
                row=requests[key]; used.add(key)
                require({k:v for k,v in receipt.items() if k!='client_received_perf'}==row and
                        row['slot']==index%8 and row['client_id']==client and row['deadline_perf']==clock['started_perf_counter']+60. and
                        row['input_sha256']==digest(value['inputs']) and row['output_sha256']==digest(value['outputs']) and
                        row['submitted_perf']>=clock['started_perf_counter'] and
                        row['complete_readback_perf'] <= receipt['client_received_perf'] <= clock['ended_perf_counter'] < clock['started_perf_counter']+75.,
                        "collision service journal matches original whole-call clock")
                total+=1
        groups.append(group)
    require(summary['groups']==groups and summary['callbacks']==total==96, "complete96 service callbacks")
    return dict(status='saved_collision_service_verified',concurrencies=[16,32],callbacks=96,exact_source_outputs=True,
                model_calls=0,request_ids=sorted(used))


def audit_qualification(plan, container):
    from . import backend
    container=Path(container); folder=container/'worker'
    execution,absent=audit_stage_execution(plan,container,mode='qualification')
    require(execution['status']=='completed' and execution['worker_exit_code']==0, "fresh qualifier completed normally")
    require(read(folder/'runtime_sources.json')['sources']==plan['source_guard'], "actual qualifier startup guard")
    selected=plan['runtime']['qualification']
    require(selected==q.build_plan() and plan['scheduler_source_equivalence']==backend.source_equivalence(),
            "authenticated reused mathematical qualification and new source equivalence")
    c.verify(selected['reused_evidence'])
    accepted=read(next(c.resolve(x) for x in selected['reused_evidence'] if x['path'].endswith('/qualification/saved_audit.json')))
    require(accepted['status']=='saved_fresh_qualification_verified' and accepted['stability']==
            dict(cpu_exact_cases=35,gpu_anchor_cases=840,service_calls=80,failures=[]), "accepted physical packing/stability reused")
    proofs,qualified,callbacks,closures={}, {}, {}, {}
    lib=prior.load_saved_library(container); collision=None
    for route in ('cpu','gpu'):
        path=folder/('qualification_'+route)
        proof=q.verify_qualification(plan['runtime'],path)
        require(proof==read(path/'saved_qualification_audit.json'), "fresh67 actual entrypoint proof independently reconstructed")
        proofs[route]=proof
        qualified[route]=dict(summary=read(path/'qualification.json'),suite=read(path/'entrypoint_suite.json'),fixed_roots=[])
        events,requests,scheduling=read_journal(folder/route,route)
        scheduler='original8' if route=='cpu' else 'row_fifo32'
        closures[route]=audit_runtime(plan,folder/route,route,events,requests,scheduling,scheduler)
        require(closures[route]['service_failure'] is None, "qualification service never poisoned")
        forensic,_=q.read_forensics(path/'entrypoint_forensics.jsonl')
        hashes={'search':[],'reference':[]}
        clocks=[e for e in forensic if e.get('operation') in ('outer_clock','outer_ordinary_clock')]
        for clock in clocks: prior.validate_clock(clock)
        for event in forensic:
            if event.get('event')=='result' and event.get('operation') in ('ordinary_worker_request','reference_choose','direct_same_quota_choose','exchange'):
                hashes[event['arm']].extend(lib.result_callback_hashes(event['result']))
        for arm,slot in (('search',0),('reference',1)):
            rows=lib.bind_client_callbacks('qualification_'+arm,hashes[arm],requests,slot=slot)
            for row in rows:
                matching=[clock for clock in clocks if row['deadline_perf']==clock['started_perf_counter']+60. and
                          clock['started_perf_counter'] <= row['submitted_perf'] and
                          row['complete_readback_perf'] <= clock['started_perf_counter']+clock['elapsed_seconds']]
                require(len(matching)==1 and matching[0]['status']=='ok', "every qualification callback original60s clock")
        callbacks[route]={arm:len(values) for arm,values in hashes.items()}
        collision_count=0
        if route=='gpu':
            collision=audit_collisions(plan,folder/'collisions',requests); collision_count=collision['callbacks']
        require(sum(callbacks[route].values())+collision_count==len(requests), "all qualifier backend callbacks accounted")
    comparison=dict(q.compare_qualification(qualified['cpu'],qualified['gpu'],plan['runtime']),
        fresh_cross_route_numerics='32 bound saved roots; every diagnostic result reproduced exactly within route',new_C1_cells=False)
    require(read(folder/'decision_comparison.json')==comparison, "accurate new qualification scope metadata")
    worker=read(folder/'worker_result.json')
    require(read(folder/'cleanup.json')['success'] is True and worker['mode']=='qualification' and worker['exit_code']==0 and
            worker['reference_weights_unchanged'] is True and worker['result']['decision_comparison']==comparison and
            worker['result']['collisions']==read(folder/'collisions/summary.json'),
            "qualifier successful result, weights and cleanup")
    return dict(status='saved_entrypoints_and_collision_service_verified',routes=proofs,callbacks=callbacks,
        collision_qualification=collision,runtime_closure=closures,comparison=comparison,
        reused_evidence=selected['reused_evidence'],reused_stability=accepted['stability'],
        scheduler_source_equivalence=plan['scheduler_source_equivalence'],owned_processes_absent=absent,model_calls=0,new_games=0)


def validate_plan(plan):
    from . import backend
    configs=(('cpu8','cpu',8,'original8'),('gpu8','gpu',8,'original8'),
             ('gpu16','gpu',16,'row_fifo32'),('gpu32','gpu',32,'row_fifo32'))
    expected=[]
    for block,order in enumerate((configs,tuple(reversed(configs))),1):
        for name,route,count,scheduler in order:
            expected.append(dict(cell_id=f'b{block}_{name}',configuration=name,block=block,route=route,
                concurrency=count,scheduler=scheduler,groups=[list(range(i,i+count)) for i in range(0,32,count)]))
    require(plan['schema']=='splendorust_gpu_batchfill_v1' and plan['cells']==expected, "exact two reversed blocks and fixed four configurations")
    require(plan['worker_limits']==dict(qualification_seconds=2700,cell_seconds=900) and
            plan['total_seconds']==c.TOTAL_SECONDS==14400 and plan['model_seconds']==c.MODEL_SECONDS==10800,
            "fixed total/model/stage clocks")
    require(plan['internal_seconds']==60. and plan['outer_seconds']==75. and plan['execution_policy_id']==c.EXECUTION_POLICY_ID and
            plan['late_completed_responses_accepted'] is True and plan['descriptive_latency_threshold_seconds']==[1.,9.,10.], "unchanged decision guards; old latency descriptive")
    require(plan['maximum_worker_rss_bytes']==c.MAX_RSS and plan['maximum_output_bytes']==c.MAX_OUTPUT and
            plan['minimum_free_disk_bytes']==c.MIN_DISK and plan['available_ram_reserve_bytes']==0 and plan['forecast_admission_gate'] is False,
            "fixed actual resource bounds, no reserve/forecast admission")
    require(plan['worker_affinity']==list(range(19)) and plan['supervisor_affinity']==[19] and plan['group_gap_seconds']==0., "unchanged host placement and zero group gap")
    require(plan['maximum_diagnostic_decisions']==256 and plan['qualification_decisions_per_route']==67 and
            plan['retries']==plan['new_games']==plan['training_updates']==0 and
            plan['larger_data_deferred'] is True and plan['automatic_followup_games'] is False and plan['automatic_followup_training'] is False,
            "bounded fixed-position scope only")
    require(plan['scheduler_source_equivalence']==backend.source_equivalence(), "frozen independently checkable implementation equivalence")
    require(plan['runtime']['qualification']==q.build_plan(), "authenticated unchanged scientific qualification")
    validate_cases(plan['cases'])
    return True


def cross_route_sources(plan):
    compare=prior.load_saved_library(c.OUT).independent_compare(); results=[]
    for case in plan['cases']:
        outputs={r:prior.root_output(case,read(c.resolve(case['expected_results'][r]))['result']) for r in ('cpu','gpu')}
        value=compare([outputs['cpu']],[outputs['gpu']],[case['legal_policy_indices']],plan['runtime']['numerical'])
        require(value['passed'], "32 bound same-state cross-route root numerical gates")
        results.append(dict(case_id=case['case_id'],numerical=value))
    return dict(status='32_bound_source_roots_numerically_qualified',rows=results,models_launched=0)


def useful_candidates(cells):
    result={}
    for name in ('gpu16','gpu32'):
        blocks=[]
        for block in (1,2):
            candidate=next((x for x in cells if x['configuration']==name and x['block']==block),None)
            control=next((x for x in cells if x['configuration']=='gpu8' and x['block']==block),None)
            ratio=(candidate['completed_decisions_per_second']/control['completed_decisions_per_second']
                   if candidate and control and candidate['eligible'] and control['eligible'] else None)
            blocks.append(dict(block=block,ratio_to_gpu8=ratio,at_least_10_percent_faster=ratio is not None and ratio>=1.10))
        result[name]=dict(useful_scheduling_signal=all(x['at_least_10_percent_faster'] for x in blocks),blocks=blocks,
                          automatic_followup=False,strength_claim=False)
    return result


def audit_study(plan, output, *, expected_plan_sha, controller_session, controller_exit_code, controller_chunk, allow_failure=False):
    output=Path(output)
    require(c.sha(c.PLAN)==expected_plan_sha, "exact frozen plan SHA")
    validate_plan(plan); c.verify(plan['inputs'])
    require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) < plan['maximum_output_bytes'], "actual retained evidence cap")
    claim,execution=read(output/'claim.json'),read(output/'execution.json')
    require(claim['plan_sha256']==execution['plan_sha256']==expected_plan_sha and
            claim['total_seconds']==plan['total_seconds'] and claim['model_seconds']==execution['model_seconds']==plan['model_seconds'] and
            claim['monotonic_origin']==execution['monotonic_origin'] and claim['unix_origin']==execution['unix_origin'] and
            0 <= execution['monotonic_end']-execution['monotonic_origin'] < plan['total_seconds'] and
            0 <= execution['unix_end']-execution['unix_origin'] < plan['total_seconds'], "single actual original global clock and bound")
    require(execution['retries']==0 and execution['new_games']==execution['game_routes_started']==execution['training_updates']==0 and
            not execution['cleanup_errors'] and execution['execution_policy_id']==c.EXECUTION_POLICY_ID, "no new games/training/retries and exact cleanup")
    absent=prior.require_processes_absent(execution['observed_owned_identities']+[dict(pid=claim['pid'],created=claim['created'])])
    require(type(controller_session) is int and controller_session>0 and bool(controller_chunk), "actual retained controller exit attribution")
    attribution=dict(session_id=controller_session,exit_code=controller_exit_code,tool_chunk=controller_chunk,
                     source='Actual retained execution supplied by root; exact owner absence independently checked')
    base=dict(plan_sha256=expected_plan_sha,auditor=c.artifact(Path(__file__)),frozen_inputs_unchanged=len(plan['inputs']),
              exact_owned_processes_absent=absent,retained_controller_exit=attribution,auditor_models_launched=0,
              auditor_games_launched=0,training_updates=0,new_games=0)
    if execution['status']=='failed':
        require(allow_failure and controller_exit_code!=0 and read(output/'failure.json')==execution and execution['error'],
                "preserved technical failure requires explicit failure-audit scope")
        return dict(**base,status='preserved_failed_attempt_verified',failure=execution['error'],
                    eligible=False,scope='Saved failure preservation/owner exit only; no completed-workload or numerical acceptance claim')
    require(controller_exit_code==0 and execution['status']=='completed_batchfill_diagnostic_pending_independent_acceptance' and
            execution['error'] is None and not (output/'failure.json').exists(), "completed controller with actual successful exit")
    qualification=audit_qualification(plan,output/'qualification')
    require(read(output/'qualification/saved_audit.json')==qualification, "saved qualification matches independent reconstruction")
    source_numerics=cross_route_sources(plan)
    cells=[]
    for spec in plan['cells']:
        container=output/'cells'/spec['cell_id']
        value=audit_cell(plan,spec,container/'worker')
        require(read(container/'saved_audit.json')==value, "saved cell audit exactly reproduced")
        cells.append(value)
    expected_stages=[dict(stage_id='qualification',status='completed',worker_exit_code=0)] + [
        dict(stage_id=x['cell_id'],status='completed' if x['status']=='passed' else 'deadline_failed',
             worker_exit_code=read(output/'cells'/x['cell_id']/'execution.json')['worker_exit_code']) for x in cells]
    require(execution['stages']==expected_stages, "all nine predetermined stages, no hidden replacement")
    completion=read(output/'completion.json')
    require(completion['execution']==c.artifact(output/'execution.json') and completion['plan']==c.artifact(c.PLAN) and
            completion['decision_cells']==8 and completion['qualified'] is True and completion['new_games']==completion['training_updates']==0 and
            completion['inputs_unchanged'] is True and completion['cell_audits']==[
                c.artifact(output/'cells'/x['cell_id']/'saved_audit.json') for x in cells], "completion binds every actual stage and saved result")
    c.verify(plan['inputs'])
    return dict(**base,status='saved_batchfill_diagnostic_verified',qualification=qualification,cross_route_source_roots=source_numerics,
        cells=cells,aggregate=aggregate_cells(cells),useful_candidate_screen=useful_candidates(cells),
        attempted_decisions=sum(sum(x['attempted'] for x in cell['case_rows']) for cell in cells),
        completed_decisions=sum(sum(x['completed'] for x in cell['case_rows']) for cell in cells),
        scope='Fixed saved-position throughput and scheduling only; no games, no strength, no sustained-production or training-collector qualification')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected-plan-sha',required=True)
    parser.add_argument('--controller-session',required=True,type=int)
    parser.add_argument('--controller-exit-code',required=True,type=int)
    parser.add_argument('--controller-chunk',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--allow-failure',action='store_true')
    args=parser.parse_args()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources()  # Once before saved independent corrected replays.
    plan=read(c.PLAN); require(plan['source_guard']==sources, "saved auditor corrected-source startup guard")
    result=audit_study(plan,c.OUT,expected_plan_sha=args.expected_plan_sha,controller_session=args.controller_session,
        controller_exit_code=args.controller_exit_code,controller_chunk=args.controller_chunk,allow_failure=args.allow_failure)
    destination=Path(args.output).resolve(); destination.parent.mkdir(parents=True,exist_ok=True)
    require(not destination.is_relative_to(c.OUT), "final audit lives outside frozen experiment evidence")
    with destination.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(c.canonical(dict(status=result['status'],output=c.artifact(destination),models=0,games=0)))
    return 0


if __name__=='__main__': raise SystemExit(main())
