"""Bounded, read-only evidence review; never launches or scores a game."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import time
import zlib

import psutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/matched_strength_20261020_01'
OUTPUT = ROOT / 'outputs/matched_strength_interruption_review_20261022_01'
FREEZE = '41244628326e23ea6a8685752f367bd4ca5e13809cfb47d983d0cd6f139b248e'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    OUTPUT.mkdir(exist_ok=False)
    started, epoch = time.monotonic(), time.time()
    launch = read(SOURCE / 'launch.json')
    end_epoch = min(epoch + 600, launch['epoch_seconds'] + launch['maximum_seconds'])

    def guard():
        if time.monotonic() - started >= 600 or time.time() >= end_epoch:
            raise RuntimeError('Read-only inspection time bound reached')

    def artifact(path):
        guard()
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while data := stream.read(4 * 1024**2):
                guard()
                digest.update(data)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Evidence changed during inspection: ' + str(path))
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size,
                'sha256': digest.hexdigest()}

    try:
        freeze_artifact = artifact(SOURCE / 'freeze.json')
        assert freeze_artifact['sha256'] == FREEZE
        freeze = read(SOURCE / 'freeze.json')
        process_receipt = read(SOURCE / 'runner_process.json')
        interrupted = SOURCE / 'workers/games/g059/t0043'
        initialization = read(interrupted / 'initialization.json')
        identities = [launch['identity'],
            {'pid': process_receipt['supervisor_pid'], 'created': process_receipt['supervisor_created']}]
        identities += [{'pid': p['pid'], 'created': p['created']}
                       for p in initialization['telemetry']['process_tree']]
        for identity in identities:
            try:
                process = psutil.Process(identity['pid'])
                assert process.create_time() != identity['created'] or not process.is_running()
            except psutil.NoSuchProcess:
                pass
        checked = []
        for item in freeze['inputs']:
            actual = artifact(ROOT / item['path'])
            if actual != item:
                raise RuntimeError('Changed frozen input: ' + item['path'])
            checked.append(actual)
        save(OUTPUT / 'frozen_input_check.json', {'status': 'passed', 'freeze': freeze_artifact,
            'files': len(checked), 'bytes': sum(x['bytes'] for x in checked),
            'all_hashes_and_sizes_match': True, 'elapsed_seconds': time.monotonic() - started})
        print(json.dumps({'stage': 'frozen_inputs_verified', 'files': len(checked)}), flush=True)
        paths = sorted(p for p in SOURCE.rglob('*') if p.is_file())
        inventory = []
        for index, path in enumerate(paths):
            inventory.append(artifact(path))
            if (index + 1) % 5000 == 0:
                print(json.dumps({'stage': 'preserved_inventory', 'files': index + 1,
                                  'total': len(paths)}), flush=True)
        assert paths == sorted(p for p in SOURCE.rglob('*') if p.is_file())
        save(OUTPUT / 'source_inventory.json', {'schema': 'preserved_interrupted_benchmark_inventory_v1',
            'source': SOURCE.relative_to(ROOT).as_posix(), 'files': inventory,
            'total_bytes': sum(x['bytes'] for x in inventory), 'content_used_only_for_hashing': True})
        timeout = SOURCE / 'workers/games/g039/t0007'
        receipt = read(timeout / 'request-00000001.json')
        row = read(SOURCE / 'decision_attempts/games/g039/t0007.json')
        assert artifact(timeout / 'request-00000001.json') == row['receipt']
        assert row['status'] == 'forfeit_timeout' and receipt['status'] == 'timeout'
        assert receipt['deadline_seconds'] == 120 and receipt['receipt_elapsed_seconds'] > 120
        assert receipt['accepted'] is False and receipt['received_before_deadline'] is False
        packets = [json.loads(line) for line in (timeout / 'stdout.log').read_bytes().splitlines()]
        assert [x['type'] for x in packets] == ['ready', 'response']
        assert packets[1]['request_id'] == receipt['request_id']
        trace = artifact(timeout / 'continuations.jsonl.gz')
        assert all(trace[k] == packets[1]['result']['trace'][k] for k in ('sha256', 'bytes'))
        timeout_review = {'id': 'games/g039/t0007', 'status': receipt['status'],
            'deadline_seconds': receipt['deadline_seconds'],
            'trusted_complete_response_seconds': receipt['receipt_elapsed_seconds'],
            'late_by_seconds': receipt['receipt_elapsed_seconds'] - 120,
            'supervisor_elapsed_seconds': receipt['elapsed_seconds'],
            'sampled_request_cpu_seconds': receipt['telemetry']['request_cpu_seconds'],
            'peak_sampled_tree_rss_bytes': receipt['telemetry']['peak_sampled_rss_bytes'],
            'late_response_retained_not_accepted': True, 'trace_identity_matches_late_payload': True,
            'recorded_classification_supported': True,
            'cause_of_wall_cpu_gap_established': False}
        stdout = (interrupted / 'stdout.log').read_bytes()
        assert len(stdout.splitlines()) == 1 and json.loads(stdout)['type'] == 'ready'
        assert (interrupted / 'events.jsonl').stat().st_size == 0
        assert not (interrupted / 'request-00000001.json').exists()
        assert not (SOURCE / 'decision_attempts/games/g059/t0043.json').exists()
        raw = (interrupted / 'continuations.jsonl.gz').read_bytes()
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        prefix = decoder.decompress(raw)
        assert decoder.eof is False
        interruption_review = {'id': 'games/g059/t0043', 'initialization_status': initialization['status'],
            'worker_identity': initialization['identity'], 'all_known_process_identities_closed': True,
            'stdout_only_ready': True, 'events_bytes': 0, 'request_receipt_present': False,
            'trusted_request_start_persisted': False, 'completed_response_present': False,
            'trace': artifact(interrupted / 'continuations.jsonl.gz'),
            'gzip_complete': decoder.eof, 'complete_trace_lines': prefix.count(b'\n'),
            'trace_contents_parsed': False, 'cause_confirmed': False,
            'classification': 'unresolved_active_request_infrastructure_interruption',
            'timeout_forfeit_supported': False, 'unchanged_clock_resumption_supported': False}
        result = {'schema': 'matched_strength_interruption_review_v1', 'status': 'inspection_complete',
            'time': dt.datetime.now(dt.timezone.utc).isoformat(), 'freeze': freeze_artifact,
            'original_deadline_epoch_seconds': launch['epoch_seconds'] + launch['maximum_seconds'],
            'inspection_maximum_seconds': 600, 'elapsed_seconds': time.monotonic() - started,
            'original_files': len(inventory), 'original_bytes': sum(x['bytes'] for x in inventory),
            'frozen_inputs_verified': len(checked), 'original_files_modified': 0,
            'new_model_or_game_requests': 0, 'matchup_outcomes_opened': False,
            'timeout': timeout_review, 'interruption': interruption_review,
            'original_primary_acceptance': 'blocked_by_unresolved_active_request',
            'required_before_new_120_second_request': 'explicit_disclosed_protocol_amendment'}
        guard()
        save(OUTPUT / 'review.json', result)
        print(json.dumps({'status': result['status'], 'original_files': len(inventory),
                          'elapsed_seconds': result['elapsed_seconds'], 'no_execution_resumed': True}), flush=True)
    except BaseException as error:
        save(OUTPUT / 'inspection_failure.json', {'category': type(error).__name__, 'detail': str(error),
            'elapsed_seconds': time.monotonic() - started, 'original_files_modified': 0})
        raise


if __name__ == '__main__':
    main()
