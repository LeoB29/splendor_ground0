"""Offline diagnosis only: read sealed records, never import or execute a player.

This is a post-hoc analysis, not a fresh timing trial or a resumed benchmark.
All original inputs are read-only; derived evidence goes to a new directory.
"""
from pathlib import Path
import collections
import datetime
import gzip
import hashlib
import json
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/splendorust_live_boundary_20261025_01'
OUT = ROOT / 'outputs/splendorust_saved_failure_diagnosis_20261025_01'
START = time.monotonic()
READ_CAP = 1024 ** 3
SECONDS = 300
read_bytes = 0
inputs = {}
seal_index = {}
source_index = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def guard():
    require(time.monotonic() - START < SECONDS, 'Offline analysis time cap')
    require(read_bytes <= READ_CAP, 'Offline analysis read cap')


def raw(path, verify=True):
    global read_bytes
    guard()
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Input outside repository')
    size = path.stat().st_size
    require(size <= 64 * 1024 ** 2, 'Per-file read cap')
    data = path.read_bytes()
    read_bytes += len(data)
    guard()
    rel = path.relative_to(ROOT).as_posix()
    item = dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if rel in inputs:
        require(inputs[rel] == item, 'Input changed during diagnosis: ' + rel)
    inputs[rel] = item
    if verify:
        expected = seal_index.get(rel) or source_index.get(rel)
        require(expected is not None, 'Input absent from original seal/freeze: ' + rel)
        require(item == expected, 'Original sealed input differs: ' + rel)
    return data


def read(path, verify=True):
    return json.loads(raw(path, verify))


def save(name, value):
    guard()
    path = OUT / name
    with path.open('x', encoding='utf-8', newline='\n') as target:
        json.dump(value, target, sort_keys=True, indent=2, allow_nan=False)
        target.write('\n')
    require(path.stat().st_size < 8 * 1024 ** 2, 'Derived file size cap')


def interval_summary(rows):
    result = {}
    for role in ('challenger', 'opponent'):
        group = [r for r in rows if r['role'] == role and r['completed']]
        result[role] = dict(count=len(group),
            mean_handler_ms=statistics.mean(r['handler_ms'] for r in group),
            mean_arrival_ms=statistics.mean(r['arrival_ms'] for r in group),
            mean_sampled_request_cpu_ms=statistics.mean(r['sampled_request_cpu_ms'] for r in group),
            callback_counts=sorted(set(r['callbacks'] for r in group)),
            simulation_counts=sorted(set(r['simulations'] for r in group)))
    return result


def main():
    require(not OUT.exists(), 'Never overwrite a diagnosis attempt')
    OUT.mkdir()
    save('scope.json', dict(schema='splendorust_saved_failure_diagnosis_v1',
        purpose='Post-hoc offline saved-failure diagnosis only', seconds_cap=SECONDS,
        read_cap_bytes=READ_CAP, original_attempt=OLD.relative_to(ROOT).as_posix(),
        model_calls=0, simulated_moves=0, failed_move_retries=0,
        players_changed=False, deadlines_changed=False, original_verification_changed=False,
        windows_selected_after_inspection=True, source_code=Path(__file__).relative_to(ROOT).as_posix(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    freeze = read(OLD / 'freeze.json', False)
    seal = read(OLD / 'completed_evidence_seal.json', False)
    seal_index.update({x['path']: x for x in seal['artifacts']})
    source_index.update({x['path']: x for x in freeze['inputs']})
    require(len(seal_index) == len(seal['artifacts']), 'Duplicate seal path')
    require(inputs[(OLD/'freeze.json').relative_to(ROOT).as_posix()]['sha256'] ==
        '50ca8363a364b06dbf452778b90aa27c5ea763d8f2d76d2c4efab8bca582ec6b', 'Original freeze')
    # Final lifecycle receipts postdate the evidence seal. Record their hashes,
    # but do not misrepresent them as members of that seal.
    completion = read(OLD/'completion.json', False)
    execution = read(OLD/'execution.json', False)
    verification = read(OLD/'verification_summary.json', False)
    performance = read(OLD/'performance.json')
    boundary = read(OLD/'boundary_analysis.json')
    read(OLD/'collection_stop.json')
    require(completion['status'] == 'captured_fault_verified', 'Original completion status')
    require(execution['exit_code'] == 0 and not execution['cleanup_errors'], 'Original cleanup')
    require(verification['status'] == 'passed' and verification['reproductions'] == 48,
        'Original verification receipt')
    rows = []
    focus = []
    for path in sorted((OLD/'games').glob('*/*/moves/*.json')):
        move = read(path)
        clock, child, parent = move['clock'], move['child_profile'], move['parent_profile']
        require(len(child['profiles']) == 1, 'One live handler interval')
        profile = child['profiles'][0]
        trace = move['trace'] if move['completed'] else None
        py = next(x for x in clock['telemetry']['process_tree'] if x['role'] == 'python')
        row = dict(source=path.relative_to(ROOT).as_posix(), stage=move['stage'],
            game=move['game_index'], ply=move['ply'], role=move['player_role'],
            completed=move['completed'], status=clock['status'], request_id=clock['request_id'],
            arrival_ms=clock['receipt_elapsed_seconds']*1000,
            handler_ms=(profile['handle_end_ns']-profile['handle_start_ns'])/1e6,
            input_delay_ms=(profile['request_read_complete_ns']-parent['request_start_ns'])/1e6,
            parse_ms=(profile['handle_start_ns']-profile['request_read_complete_ns'])/1e6,
            sampled_request_cpu_ms=clock['telemetry']['request_cpu_seconds']*1000,
            python_rss_bytes=py['rss_bytes'], python_threads=py['observed_threads'],
            affinity=py['affinity'], callbacks=trace['inference_calls'] if trace else None,
            simulations=sum(x['simulations'] for x in trace['phases']) if trace else None,
            phase_count=len(trace['phases']) if trace else None)
        require(clock['deadline_seconds'] == 1.0, 'Original one-second deadline changed')
        require(row['affinity'] == [0], 'Original player affinity changed')
        require(move['completed'] == (clock['status'] == 'ok'), 'Completion binding')
        rows.append(row)
        if (move['stage']=='main' and move['game_index']==72) or (
            move['completed'] and row['arrival_ms'] > 700):
            # Cross-bind embedded copies to the original independent receipts.
            for key, value in (('worker_clock_file',clock), ('child_profile_file',child),
                               ('parent_profile_file',parent)):
                require(read(ROOT/move[key]) == value, 'Independent receipt mismatch: '+key)
            focus.append(row)
    require(len(rows) == 4371, 'Live prefix request count')
    failed = [r for r in rows if not r['completed']]
    require(len(failed)==1 and failed[0]['game']==72 and failed[0]['ply']==28,
        'Exactly the original failed decision')
    slow = [r for r in rows if r['completed'] and r['arrival_ms']>700]
    require(len(slow)==10 and all(r['role']=='challenger' and r['callbacks']==65 and
        r['simulations']==64 for r in slow), 'Slow-move work counts')
    game = [r for r in rows if r['stage']=='main' and r['game']==72]
    require(len(game)==29, 'Final game length')
    for role in ('challenger','opponent'):
        group = [r for r in rows if r['role']==role and r['completed']]
        recorded = performance['players'][role]
        require(len(group)==recorded['completed_decisions'], 'Completed count')
        require(abs(statistics.mean(r['arrival_ms'] for r in group)/1000-
            recorded['completed_arrival_seconds']['mean']) < 1e-12, 'Recomputed mean')
        require(max(r['arrival_ms'] for r in group)/1000 ==
            recorded['completed_arrival_seconds']['max'], 'Recomputed maximum')
    failed_dir = OLD/'workers/main_g72_challenger'
    error = read(failed_dir/'worker_error.json')
    require('line 134, in _infer' in error['traceback'], 'Saved failing boundary location')
    stdout = gzip.decompress(raw(failed_dir/'stdout.log.gz'))
    require(len(stdout)<8*1024**2, 'Decoded stdout cap')
    packets=[json.loads(line) for line in stdout.splitlines() if line.strip()]
    last=[p for p in packets if p.get('request_id')=='request-00000016']
    require(len(last)==1 and last[0]['type']=='error', 'Raw failed request has error only')
    for role in ('challenger','opponent'):
        require(gzip.decompress(raw(OLD/f'workers/main_g72_{role}/stderr.log.gz'))==b'',
            'Final game stderr')
    sources = [
        'tools/local_player_qualification_128/clock.py',
        'tools/splendorust_search_scale_20261024/adapters.py',
        'tools/splendorust_timing_opt_20261024/sampler.py',
        'tools/splendorust_live_boundary_20261025/runner.py',
        'tools/splendorust_live_boundary_20261025/launch.py',
        'tools/splendorust_boundary_timing_20261025/worker.py',
        'tools/splendorust_inference_opt_20261025/worker.py',
        'tools/external_strength_screen_20261018_02/splendorust_adapter.py']
    for name in sources:
        raw(ROOT/name)
    windows = dict(early_plies_0_through_9=interval_summary([r for r in game if r['ply']<10]),
        later_plies_14_through_27=interval_summary([r for r in game if 14<=r['ply']<=27]))
    memory = {}
    for role in ('challenger','opponent'):
        selected=[r for r in game if r['role']==role]
        memory[role]=dict(min_python_rss_bytes=min(r['python_rss_bytes'] for r in selected),
            max_python_rss_bytes=max(r['python_rss_bytes'] for r in selected),
            observed_python_thread_counts=sorted(set(r['python_threads'] for r in selected)))
    result=dict(schema='splendorust_saved_failure_diagnosis_results_v1', status='passed',
        original_completion_status=completion['status'], rows=len(rows), completed=len(rows)-1,
        failed_decision=failed[0], successful_slow_moves=slow, final_game_windows=windows,
        final_game_memory=memory, final_game_all_completed_callback_counts={role:
            sorted(set(r['callbacks'] for r in game if r['role']==role and r['completed']))
            for role in ('challenger','opponent')}, original_topology=freeze['topology'],
        original_verification=verification, failure_boundary=boundary['failures'][0],
        original_error_packet_count=len(last), failed_move_has_no_partial_callback_count=True,
        model_calls=0, new_games=0, original_files_written=0,
        causal_component_proven=False, supervisor_contention_proven=False,
        cpu_caveat='Sampled cumulative request CPU includes non-handler work, excludes parent, and is coarse; not exact inference CPU or exact waiting time.',
        windows_caveat='Intervals chosen after inspecting this failure; descriptive only, not independent significance or a timeout-rate estimate.')
    save('diagnosis.json',result)
    save('final_game_timeline.json',game)
    save('all_decision_metrics.json',rows)
    # Reread every consumed original input. This is a scoped preservation check,
    # not a new claim that the entire historical evidence seal was re-audited.
    for rel in list(inputs):
        raw(ROOT/rel, False)
    save('input_manifest.json',dict(inputs=sorted(inputs.values(),key=lambda x:x['path']),
        original_seal_members_checked=sum(p in seal_index for p in inputs),
        original_frozen_source_members_checked=sum(p in source_index for p in inputs),
        unchanged_after_analysis=True, whole_original_seal_reaudit=False))
    save('completion.json',dict(status='completed', elapsed_seconds=time.monotonic()-START,
        original_files_unchanged=True, bytes_read=read_bytes, inputs=len(inputs),
        model_calls=0, fresh_games=0, failed_move_retries=0,
        time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(status='completed', decisions=len(rows), slow=len(slow),
        windows=windows, elapsed_seconds=time.monotonic()-START)))


if __name__ == '__main__':
    main()
