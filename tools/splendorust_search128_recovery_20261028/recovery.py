"""Forensic adoption of one already emitted response; never retries a decision.

The original resource_guard receipt stays rejected.  A separate recovery bridge
binds the unique captured reader event to the unique complete raw response and
checks its science and dictionary transition.  This is an amendment prerequisite,
not acceptance of the original attempt or authorization to restart any clock.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'outputs/splendorust_search128_strength_20261028_01'
FREEZE_SHA256 = '9026f33a3a6dcc449bb28f025493825230f04d91298f21c2c5bb2382b69587c9'
WORKER = 'main_g38_challenger'
REQUEST_ID = 'request-00000021'
GAME = 'games/main/p019_g38'
SCHEMA = 'splendorust_search128_saved_response_recovery_v1'
SOURCE_ROW_SHA256 = 'f04bc4686b2623a1bcbf4f2c2d96c6fcace49c08372a6fcf1751ef0da49a468f'
REQUIRED_SOURCES = (
    'tools/local_player_qualification_128/clock.py',
    'tools/splendorust_boundary_timing_20261025/parent_profile.py',
    'tools/splendorust_search128_strength_20261028/transport.py',
    'tools/splendorust_search128_strength_20261028/adapters.py',
    'tools/splendorust_search128_strength_20261028/scientific.py',
    'tools/splendorust_search128_strength_20261028/placement.py',
    'tools/splendorust_search128_strength_20261028/common.py',
    'tools/splendorust_search128_strength_20261028/worker.py',
    'tools/splendorust_search128_strength_20261028/schedule.py',
    'tools/splendorust_search_scale_20261024/adapters.py',
    'tools/external_strength_screen_20261018_02/splendorust_adapter.py',
    'tools/external_strength_screen_20261018/audit.py',
    'tools/external_strength_screen_20261018/rules.py',
    'tools/external_strength_screen_20261018_02/bridge.py',
    'tools/future_training_rules_20261021.py',
)


def require(condition, message):
    if not condition:
        raise ValueError('SAVED_RESPONSE_RECOVERY: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _int(value, label):
    require(type(value) is int and value >= 0, label)
    return value


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class Reader:
    """Small bounded immutable byte readback; no files are written."""
    def __init__(self, boundary):
        self.boundary, self.inputs = boundary, {}

    def raw(self, path, maximum=8*1024*1024):
        self.boundary()
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT.resolve()), 'read outside project')
        before = path.stat()
        require(before.st_size <= maximum, 'bounded source/evidence read')
        raw = path.read_bytes()
        after = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after), 'file changed while reading')
        item = dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=sha(raw))
        prior = self.inputs.get(item['path'])
        require(prior is None or prior == item, 'contradictory repeated read')
        self.inputs[item['path']] = item
        return raw

    def read(self, path):
        return _json(self.raw(path))

    def pin(self, item):
        raw = self.raw(ROOT / item['path'])
        require(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'changed frozen source ' + item['path'])

    def finish(self):
        for item in list(self.inputs.values()):
            self.pin(item)
        return sorted(self.inputs.values(), key=lambda x: x['path'])


def validate_emitted_response(row, receipt, parent, child, payload, raw_line, *, worker_pid):
    """Pure check of the separate bridge; never changes an original receipt."""
    require(row['clock'] == receipt and row['parent_profile'] == parent and row['child_profile'] == child,
            'saved row/receipt/sidecar equality')
    require(row.get('completed') is False and all(row.get(k) is None for k in ('action', 'trace', 'after')),
            'original rejected row remains unplayed')
    require(receipt['status'] == 'resource_guard' and receipt['accepted'] is False and receipt['result'] is None and
            receipt['receipt_elapsed_seconds'] is None and receipt['received_before_deadline'] is False,
            'only this unconsumed infrastructure receipt is eligible')
    require('FileNotFoundError: [WinError 2]' in receipt['detail'] and
            REQUEST_ID + '_child_profile.pending' in receipt['detail'], 'exact pending-rename failure mechanism')
    require(receipt['request_id'] == parent['request_id'] == child['request_id'] == REQUEST_ID,
            'exact failed request identity')
    require(type(receipt['deadline_seconds']) in (int, float) and receipt['deadline_seconds'] == 10.,
            'unchanged original ten-second deadline')
    require(type(receipt['elapsed_seconds']) in (int, float) and math.isfinite(receipt['elapsed_seconds']) and
            0 < receipt['elapsed_seconds'] < 10., 'original failure inside complete deadline')
    require(all(x['clock'] == 'perf_counter_ns' and x['clock_implementation'] == 'QueryPerformanceCounter()'
                for x in (receipt, parent, child)), 'same original clock domain')
    require(parent['child_pid'] == child['pid'] == worker_pid and type(worker_pid) is int,
            'same owned child process')
    start = _int(parent['request_start_ns'], 'original start')
    require(type(parent['raw_arrivals_ns']) is list and len(parent['raw_arrivals_ns']) == 1 and
            parent['dropped_arrival_events'] == 0, 'one unambiguous original reader event without drops')
    arrival = _int(parent['raw_arrivals_ns'][0], 'observed reader arrival')
    require(start <= arrival <= start + 10_000_000_000 and
            arrival <= start + round(receipt['elapsed_seconds']*1e9) <= parent['capture_snapshot_ns'],
            'original event within original deadline and failed request interval')
    require(all(parent.get(k) is None for k in ('receipt_arrival_ns', 'receipt_reconstructed_arrival_ns',
            'response_raw_sha256', 'response_raw_bytes', 'response_capture_overflow')),
            'do not invent a parent-consumed packet/hash')
    require(parent['dispatch_write_calls'] >= 1 and parent['dispatch_flush_calls'] == 1 and
            all(parent[k] == 0 for k in ('dispatch_write_failures', 'dispatch_flush_failures', 'pending_writes', 'pending_flushes')),
            'completed unambiguous dispatch')
    require(parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False and
            set(payload) == {'player', 'observer'} and payload['player'] == 'search' and payload['observer'] == row['observer'],
            'complete lawful original request')
    # Saved JSON changes key ordering. Do not invent original wire bytes by reserializing it.
    # The child hashes its actually received raw line and its decoded payload separately.
    require(type(parent['request_raw_bytes']) is int and 0 < parent['request_raw_bytes'] <= 1024*1024 and
            type(parent['request_raw_sha256']) is str and len(parent['request_raw_sha256']) == 64 and
            parent['request_raw_sha256'] == child['request_raw_sha256'] and digest(payload) == child['payload_sha256'],
            'actual parent/child raw-request hash plus canonical decoded-payload binding')
    require(type(raw_line) is bytes and raw_line.endswith(b'\n') and raw_line.count(b'\n') == 1 and
            0 < len(raw_line) <= 4*1024*1024, 'one bounded complete raw response line')
    normalized = raw_line[:-2] + b'\n' if raw_line.endswith(b'\r\n') else raw_line
    require(len(normalized) == child['serialized_packet_utf8_bytes'] and sha(normalized) == child['serialized_packet_utf8_sha256'],
            'child serialized packet equals raw wire, terminal CRLF normalization only')
    packet = _json(raw_line)
    require(type(packet) is dict and set(packet) == {'type', 'request_id', 'result'} and
            packet['type'] == 'response' and packet['request_id'] == REQUEST_ID, 'exact successful response envelope')
    result = packet['result']
    require(digest(result) == child['result_sha256'], 'unchanged saved full scientific result')
    require(child['outcome'] == 'response_flushed' and child['error'] is None and
            child['success_sidecar_written_after_response_flush'] is True and child['quota'] == 128,
            'successful original handler/emission, not reconstructed output')
    require(len(child['profiles']) == 1, 'one original ordinary handler')
    handle = child['profiles'][0]
    require(handle['handle_failed'] is False and handle['pid'] == worker_pid and
            handle['clock'] == 'perf_counter_ns', 'same successful handler')
    read, begin, end, pre, post = [_int(x, 'child boundary') for x in (
        child['request_read_complete_ns'], handle['handle_start_ns'], handle['handle_end_ns'],
        child['pre_write_ns'], child['post_flush_ns'])]
    require(start <= read <= begin <= end <= pre <= post <= arrival <= parent['capture_snapshot_ns'],
            'complete original saved boundary chronology')
    require(type(child['request_started_seconds']) in (int, float) and math.isfinite(child['request_started_seconds']) and
            0 <= end/1e9-child['request_started_seconds'] < 9., 'original nine-second handler guard')
    return dict(result=result, original_start_ns=start, observed_reader_arrival_ns=arrival,
                observed_complete_response_seconds=(arrival-start)/1e9,
                raw_wire_bytes=len(raw_line), raw_wire_sha256=sha(raw_line),
                normalized_packet_bytes=len(normalized), normalized_packet_sha256=sha(normalized),
                original_handler_seconds=(end-begin)/1e9)


def _log(reader, folder, name, descriptor):
    packed = reader.raw(folder/(name+'.gz'))
    require(len(packed) == descriptor['compressed_bytes'] and sha(packed) == descriptor['compressed_sha256'] and
            descriptor['status'] == 'verified', 'saved compressed log binding')
    require(type(descriptor['raw_bytes']) is int and 0 <= descriptor['raw_bytes'] < 8*1024*1024,
            'bounded raw log')
    raw = gzip.decompress(packed)
    require(len(raw) == descriptor['raw_bytes'] and sha(raw) == descriptor['raw_sha256'], 'decoded exact log bytes')
    return raw


def validate_wire_inventory(stdout, events, initialization, receipts, parents):
    """Unique packet/order proof compensates for absent consumed-response hash."""
    ids = [f'request-{i:08d}' for i in range(1, 22)]
    require(list(receipts) == list(parents) == ids, 'exact contiguous request inventory')
    lines = stdout.splitlines(keepends=True)
    require(len(lines) == 22 and all(x.endswith(b'\n') for x in lines), 'READY plus exactly21 complete responses')
    packets = [_json(x) for x in lines]
    require(packets[0] == dict(type='ready', metadata=initialization['result']), 'original READY binding')
    event_lines = events.splitlines(keepends=True)
    require(len(event_lines) == 21 and all(x.endswith(b'\n') for x in event_lines), 'exact21 event receipts')
    event_rows = [_json(x) for x in event_lines]
    previous_end = None
    for index, rid in enumerate(ids):
        receipt, parent, packet, raw = receipts[rid], parents[rid], packets[index+1], lines[index+1]
        require(packet.keys() == {'type', 'request_id', 'result'} and packet['type'] == 'response' and
                packet['request_id'] == receipt['request_id'] == parent['request_id'] == rid,
                'ordered unique response/request identities')
        require(event_rows[index] == {k:v for k,v in receipt.items() if k != 'result'}, 'event/original receipt equality')
        if previous_end is not None:
            require(previous_end < parent['request_start_ns'], 'nonoverlapping serial request contexts')
        previous_end = parent['capture_snapshot_ns']
        if index < 20:
            require(receipt['status'] == 'ok' and receipt['accepted'] is True and receipt['result'] == packet['result'] and
                    parent['response_raw_sha256'] == sha(raw) and parent['response_raw_bytes'] == len(raw) and
                    parent['response_capture_overflow'] is False and parent['dropped_arrival_events'] == 0 and
                    parent['raw_arrivals_ns'] == [parent['receipt_arrival_ns']] and
                    parent['receipt_arrival_ns'] == parent['receipt_reconstructed_arrival_ns'] and
                    parent['receipt_arrival_ns'] == parent['request_start_ns'] + round(receipt['receipt_elapsed_seconds']*1e9),
                    'every preceding line already consumed and uniquely hash-bound')
    return lines[-1]


def build_bridge(original_freeze, boundary=lambda: None):
    """Return bridge/result/source_row/before/after without modifying anything."""
    reader = Reader(boundary)
    frozen_raw = reader.raw(ORIGINAL/'freeze.json')
    require(sha(frozen_raw) == FREEZE_SHA256 and _json(frozen_raw) == original_freeze, 'exact original freeze')
    pins = {str(Path(x['path'])).replace('\\','/'):x for x in original_freeze['inputs']}
    for name in REQUIRED_SOURCES:
        require(name in pins, 'scientific/clock source missing from original freeze')
        reader.pin(pins[name])
    from tools.future_training_rules_20261021 import verify_runtime_sources, decode_training_snapshot
    from tools.external_strength_screen_20261018 import audit as ledger
    from tools.splendorust_search128_strength_20261028 import adapters, transport
    verify_runtime_sources()
    folder, game = ORIGINAL/'workers'/WORKER, ORIGINAL/GAME
    row_raw = reader.raw(game/'moves/0039.json')
    require(sha(row_raw) == SOURCE_ROW_SHA256, 'exact preserved failed source row')
    row = _json(row_raw)
    require((row['stage'],row['pair_index'],row['game_index'],row['ply'],row['player_role'],row['seat'],row['quota'],row['route']) ==
            ('main',19,38,39,'challenger',1,128,'research'), 'exact authorized bridge position')
    cleanup, initial = reader.read(folder/'cleanup.json'), reader.read(folder/'initialization.json')
    require(cleanup['status'] == 'passed' and not cleanup['errors'] and not cleanup['alive'], 'owned failed worker cleaned up')
    require(initial['status'] == 'ok' and initial['accepted'] is True, 'original worker admission')
    ids = [f'request-{i:08d}' for i in range(1,22)]
    require(sorted(p.name for p in (folder/'parent_profiles').glob('*.json')) == [x+'.json' for x in ids],
            'exact parent event inventory')
    receipts = {rid:reader.read(folder/(rid+'.json')) for rid in ids}
    parents = {rid:reader.read(folder/'parent_profiles'/(rid+'.json')) for rid in ids}
    payload = reader.read(folder/'inputs'/(REQUEST_ID+'.json'))
    child = reader.read(folder/(REQUEST_ID+'_child_profile.json'))
    stdout = _log(reader,folder,'stdout.log',cleanup['log_compression']['stdout.log'])
    events = _log(reader,folder,'events.jsonl',cleanup['log_compression']['events.jsonl'])
    line = validate_wire_inventory(stdout,events,initial,receipts,parents)
    observed = validate_emitted_response(row,receipts[REQUEST_ID],parents[REQUEST_ID],child,payload,line,
                                         worker_pid=initial['identity']['pid'])
    # Retain the original validator's failure semantics. No fake accepted clock is passed to it.
    for path in (folder/'placement_admission.json',folder/'placement_requests'/(REQUEST_ID+'.json'), ORIGINAL/'launch.json'):
        reader.read(path)
    transport.verify_saved_record(row,original_freeze)
    require(reader.read(folder/'placement_requests'/(REQUEST_ID+'.json')) == row['placement_evidence'], 'placement bytes bound')
    identity = adapters.profile_identity(original_freeze,'research',128,boundary=boundary)
    reader.pin(original_freeze['research_profiles']['128'])
    result = observed.pop('result')
    adapters.validate_result(result,row['observer'],'research',identity=identity)
    saved_initial = reader.read(game/'initial.json')
    current = saved_initial['initial']
    require(sorted(p.name for p in (game/'moves').glob('*.json')) == [f'{i:04d}.json' for i in range(40)],
            'original game ends exactly at the failed row; no hidden continuation')
    decode_training_snapshot(current)
    for ply in range(39):
        boundary(); prior = reader.read(game/'moves'/f'{ply:04d}.json')
        require(prior['ply'] == ply and prior['before'] == current and prior['completed'] is True and
                prior['clock']['status'] == 'ok' and prior['clock']['accepted'] is True and
                prior['action'] == prior['clock']['result']['action'], 'contiguous completed saved game prefix')
        ledger.check_observer(current,prior['observer'])
        current = ledger.transition(current,prior['action'])
        require(current == prior['after'], 'saved prefix ledger transition')
    require(current == row['before'] and not current['state']['terminal'], 'exact unplayed failed position')
    ledger.check_observer(current,row['observer'])
    after = ledger.transition(current,result['action']); decode_training_snapshot(after)
    # The caller authenticates the complete original freeze. This finite bridge
    # binds its explicit clock/science sources independently and deterministically.
    consumed = reader.finish()
    bridge = dict(schema=SCHEMA, disposition='eligible_saved_response_adoption_under_explicit_amendment',
        original_freeze_sha256=FREEZE_SHA256, original_attempt_status='incomplete',
        original_clock_status='resource_guard', original_clock_accepted=False,
        original_receipt_rewritten=False, original_row_rewritten=False, request_id=REQUEST_ID, worker=WORKER,
        stage='main',pair_index=19,game_index=38,ply=39,resume_ply=40,route='research',quota=128,
        source_row_sha256=SOURCE_ROW_SHA256, original_clock_sha256=digest(row['clock']),
        result_sha256=digest(result),action_sha256=digest(result['action']),before_sha256=digest(current),after_sha256=digest(after),
        packet_binding='unique final original-reader event plus exact serial raw packet inventory; absent parent consumed-packet hash is not invented',
        timing_evidence=observed, original_parent_response_hash_available=False,
        scientific_validation='original quota128 full-result validator and independent corrected dictionary ledger',
        original_game_prefix_transitions=39,recovered_transition_count=1,
        new_model_calls=0,new_search_calls=0,decision_clock_restarted=False,
        original_attempt_accepted=False,continuation_authorized_by_this_proof=False,
        consumed_inputs=consumed)
    return dict(bridge=bridge,result=result,source_row=deepcopy(row),before=deepcopy(current),after=after)


def prove_saved_response(original_freeze, *, boundary=lambda: None):
    return build_bridge(original_freeze,boundary)


def verify_bridge(saved_bridge, original_freeze, boundary=lambda: None):
    proved = build_bridge(original_freeze,boundary)
    require(canonical(saved_bridge) == canonical(proved['bridge']), 'saved recovery bridge differs from independent reconstruction')
    return proved
