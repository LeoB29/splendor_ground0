"""Model-free, bounded diagnosis of preserved c135 evidence; never mutates it."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import time

from tools.splendorust_collection4096_20261115.collection.storage import parse_bytes

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'outputs/splendorust_collection4096_recovery_20261116_01/collection/cells/c135/worker'
MAX_JOURNAL_BYTES = 256 * 1024**2


def require(value, text):
    if not value:
        raise ValueError(text)


def read(path):
    require(path.stat().st_size <= 32 * 1024**2, 'Bounded evidence document')
    return parse_bytes(path.read_bytes())


def artifact(path):
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def summarize_receipts(rows, started):
    require(isinstance(rows, list), 'Receipt list')
    if not rows:
        return dict(count=0)
    ids = [r['request_id'] for r in rows]
    require(len(set(ids)) == len(ids), 'Unique callback identities')
    for r in rows:
        require(started <= r['submitted_perf'] <= r['dispatch_perf'] <= r['complete_readback_perf'] <= r['client_received_perf'],
                'Chronological successful callback receipt')
        require(r['client_received_perf'] < r['deadline_perf'], 'Successful receipt before its unchanged deadline')
    forwards = [r['forward_roundtrip_seconds'] for r in rows]
    postforward = [r['client_received_perf'] - r['complete_readback_perf'] for r in rows]
    return dict(count=len(rows), first_request_id=ids[0], last_request_id=ids[-1],
                first_return_seconds=rows[0]['client_received_perf'] - started,
                last_return_seconds=rows[-1]['client_received_perf'] - started,
                maximum_forward_seconds=max(forwards), total_forward_seconds=sum(forwards),
                maximum_postforward_seconds=max(postforward), last_postforward_seconds=postforward[-1],
                last_forward_seconds=forwards[-1])


def classify_outcomes(outcomes):
    statuses = Counter(v['status'] for v in outcomes.values())
    triggers = []
    cancellations = []
    for game, value in outcomes.items():
        if value['status'] == 'integrity_failed':
            require(value['clock']['status'] == 'timeout' and not value['clock']['completed'], 'Saved timeout classification')
            triggers.append(game)
        elif value['status'] == 'cancelled':
            require(value['clock']['status'] == 'error' and not value['clock']['completed'], 'Cascaded error classification')
            cancellations.append(game)
        else:
            require(value['status'] == 'completed', 'No invented outcome category')
    return dict(status_counts=dict(statuses), triggering_recorded_timeout_games=triggers,
                cancelled_games=cancellations,
                underlying_cause='unresolved: successful network execution timing does not measure OS scheduling or evidence I/O independently',
                strength_inference=None)


def diagnose(source=SOURCE):
    started = time.perf_counter()
    queue_path = source / 'cell/queue_outcomes.json'
    queue = read(queue_path)
    require(not queue['errors'] and not queue['never_started'], 'All sixteen fixed games assigned')
    outcomes = queue['outcomes']; classification = classify_outcomes(outcomes)
    require(classification['status_counts'] == {'completed': 8, 'cancelled': 7, 'integrity_failed': 1}, 'Exact preserved failed-cell boundary')
    require(classification['triggering_recorded_timeout_games'] == ['g2174'], 'Exact triggering recorded failure')
    partials = []; returned = {}; journal_by_deadline = {}; bindings = [artifact(queue_path)]
    for game, result in sorted(outcomes.items()):
        if result['status'] == 'completed':
            continue
        folder = source / 'cell/games' / game
        ply = result['failed_ply']; name = f'{ply:06d}.json'
        moves = sorted((folder / 'moves').glob('*.json'))
        require([p.name for p in moves] == [f'{n:06d}.json' for n in range(ply)], 'Contiguous completed move prefix')
        require(result['completed_moves'] == ply, 'Saved completed move count')
        last = read(moves[-1]); attempt = read(folder / 'attempts' / name)
        require(hashlib.sha256(json.dumps(last['after'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                == result['next_snapshot_sha256'] == attempt['before_sha256'], 'Exact last saved state for continuation')
        require(not any((folder / key / name).exists() for key in ['moves', 'raw_results', 'captures']), 'No completed failed response to overwrite')
        calls_path = folder / 'service_calls' / name
        rows = read(calls_path)
        start = result['clock']['started_perf_counter']; summary = summarize_receipts(rows, start)
        require(attempt['started_perf_counter'] == start and result['clock']['internal_seconds'] == 60
                and result['clock']['outer_seconds'] == 75, 'Original unchanged decision clocks')
        for row in rows:
            require(row['request_id'] not in returned, 'No successful request attributed twice')
            returned[row['request_id']] = row
        key = (attempt['client_id'], start + 60.)
        require(key not in journal_by_deadline, 'Distinct partial operation deadline')
        journal_by_deadline[key] = []
        partials.append(dict(game_id=game, failed_ply=ply, status=result['status'], client_id=attempt['client_id'],
                             original_slot=attempt['slot'], before_sha256=attempt['before_sha256'],
                             clock=result['clock'], saved_successful_callbacks=summary,
                             attempt_observed_seconds=attempt['observed_unix'] - result['clock']['started_unix'],
                             last_completed_move=artifact(moves[-1]), original_attempt=artifact(folder / 'attempts' / name),
                             original_request=artifact(folder / 'requests' / name), service_calls=artifact(calls_path),
                             original_failure=artifact(folder / 'failure.json')))
    journal = source / 'runtime/inference.jsonl'
    require(journal.stat().st_size <= MAX_JOURNAL_BYTES, 'Bounded raw journal scan')
    digest = hashlib.sha256(); count = 0; last = deque(maxlen=24); seen = set(); matched = 0
    forward_window_max = 0.; window = min(p['clock']['started_perf_counter'] for p in partials)
    with journal.open('rb') as stream:
        for line in stream:
            require(time.perf_counter()-started < 120., 'Two-minute read-only diagnosis bound')
            require(len(line) < 64*1024, 'Bounded journal row')
            digest.update(line); row = json.loads(line); count += 1
            require(row['kind'] == 'inference_batch' and row['route'] == 'cpu', 'Saved CPU-only inference batch')
            for request in row['requests']:
                identity = request['request_id']; require(identity not in seen, 'Unique raw journal callback'); seen.add(identity)
                if request['dispatch_perf'] >= window:
                    forward_window_max = max(forward_window_max, request['forward_roundtrip_seconds'])
                    last.append(request)
                key = (request['client_id'], request['deadline_perf'])
                if key in journal_by_deadline:
                    journal_by_deadline[key].append(request)
                if identity in returned:
                    expected = dict(returned[identity]); expected.pop('client_received_perf')
                    require(request == expected, 'Every successful partial callback matches raw evidence exactly')
                    matched += 1
    require(matched == len(returned), 'All partial successful callbacks independently attributed')
    for p in partials:
        rows = journal_by_deadline[(p['client_id'], p['clock']['started_perf_counter'] + 60.)]
        p['raw_journal_callback_count'] = len(rows)
        p['journal_callbacks_without_successful_return'] = [r for r in rows if r['request_id'] not in returned]
    final = read(source / 'runtime/runtime_final.json')
    require(count == final['service_batches'] == final['service_requests'], 'Exact complete raw journal count')
    return dict(schema='collection4096_saved_timeout_diagnosis_v1', status='diagnosis_completed', model_calls=0,
                games_started=0, source_mutations=0, source_cell='c135', classification=classification,
                partial_decisions=partials, raw_journal=dict(path=journal.relative_to(ROOT).as_posix(),
                    bytes=journal.stat().st_size, sha256=digest.hexdigest(), batches=count),
                successful_partial_callbacks_matched=matched, final_window_maximum_forward_seconds=forward_window_max,
                final_journal_callbacks=list(last), worker_cleanup=read(source/'cleanup.json'),
                service_final=final, bindings=bindings, elapsed_seconds=time.perf_counter()-started,
                interpretation=['One preserved 75-second response timeout; seven propagated cancellations.',
                    'Long waiting before first callback and after network execution; no direct OS or I/O measurements establish cause.',
                    'Synchronous shared-journal write and flush occurs after complete_readback_perf and before client_received_perf.',
                    'Do not change teacher, numerical execution, deadlines or runtime on an unproven causal hypothesis.',
                    'Recovery must separately disclose eight retried unfinished decisions, retain every completed move/game and every failed attempt.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    result = diagnose(); result['diagnostic_source'] = artifact(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], callbacks=result['raw_journal']['batches'],
                         partial_callbacks=result['successful_partial_callbacks_matched'],
                         partial_decisions=len(result['partial_decisions']), elapsed_seconds=result['elapsed_seconds'])))
