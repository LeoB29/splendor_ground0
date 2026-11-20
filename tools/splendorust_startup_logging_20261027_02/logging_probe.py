"""Time the existing synchronous forensic writer, never buffer its records.

OFF delegates the immutable recorder. ON performs the same JSON/write/flush/
fsync/chain-digest operations, with six clock boundaries. Only timing metadata
is held in memory. Success snapshots follow response flush. Failed-handler
snapshots precede error packets, since the parent closes failed workers.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

from tools.external_strength_screen_20261018.common import digest
from tools.splendorust_search_scale_20261024.worker import _forensic_recorder

SCHEMA = 'splendorust_startup_logging_probe_v1'
MAX_RECORDS = 1024
MAX_LOG_BYTES = 32 * 1024**2
MAX_PROFILE_BYTES = 2 * 1024**2
MARKERS = ('start_ns', 'serialized_ns', 'written_ns', 'flushed_ns', 'synced_ns', 'digested_ns')
STAGES = ('serialization', 'write', 'flush', 'fsync', 'chain_digest')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _failure(error):
    return dict(type=type(error).__name__, detail=str(error)[:4096])


class Recorder:
    def __init__(self, path, instrumented):
        require(type(instrumented) is bool, 'Literal instrumentation mode required')
        self.path, self.instrumented = Path(path), instrumented
        self.entries, self.sequence, self.previous, self.closed = [], 0, None, False
        if instrumented:
            self.stream = self.path.open('x', encoding='utf-8')
            self.off_record = None
        else:
            self.off_record, self.stream = _forensic_recorder(self.path)

    def record(self, event):
        if not self.instrumented:
            # No timer or per-record bookkeeping in the OFF callback. Rebuild
            # its entries from the persisted chain after packet emission.
            return self.off_record(event)
        require(not self.closed and len(self.entries) < MAX_RECORDS, 'Forensic probe record bound/closure')
        entry = dict(sequence=self.sequence, event=event.get('event'), completed=False,
                     timestamps={name: None for name in MARKERS}, error=None)
        self.entries.append(entry)
        stamps = entry['timestamps']
        row = dict(sequence=self.sequence, previous_sha256=self.previous, event=event)
        try:
            stamps['start_ns'] = time.perf_counter_ns()
            encoded = json.dumps(row, separators=(',', ':'), sort_keys=True, allow_nan=False)
            stamps['serialized_ns'] = time.perf_counter_ns()
            self.stream.write(encoded + '\n')
            stamps['written_ns'] = time.perf_counter_ns()
            self.stream.flush()
            stamps['flushed_ns'] = time.perf_counter_ns()
            os.fsync(self.stream.fileno())
            stamps['synced_ns'] = time.perf_counter_ns()
            self.previous = digest(row)
            stamps['digested_ns'] = time.perf_counter_ns()
            self.sequence += 1
            entry['completed'] = True
        except BaseException as error:
            entry['error'] = _failure(error)
            raise

    def close(self):
        require(not self.closed, 'Forensic recorder already closed')
        self.stream.close()
        self.closed = True

    def snapshot(self, request_id, start_record, child_profile, emission, binding=None):
        """Read durable bytes after success, or before an already-failed error packet."""
        require(self.path.stat().st_size <= MAX_LOG_BYTES, 'Forensic log byte bound')
        raw = self.path.read_bytes()
        require(len(raw) <= MAX_LOG_BYTES, 'Forensic log byte bound')
        if not self.instrumented:
            lines = raw.splitlines(keepends=True)
            require(all(line.endswith(b'\n') for line in lines), 'Partial OFF forensic bytes')
            rows = [json.loads(line.decode('utf-8'), object_pairs_hook=_unique) for line in lines]
            require(len(rows) <= MAX_RECORDS, 'OFF forensic record bound')
            self.entries = [dict(sequence=row['sequence'], event=row['event']['event'], completed=True,
                timestamps={name: None for name in MARKERS}, error=None) for row in rows]
        require(type(start_record) is int and 0 <= start_record <= len(self.entries), 'Forensic record slice')
        entries = deepcopy(self.entries[start_record:])
        before_error = emission.get('kind') == 'before_error'
        value = dict(schema=SCHEMA, instrumented=self.instrumented, request_id=request_id,
            binding=deepcopy(binding), log=self.path.name, start_record=start_record,
            end_record=len(self.entries), prefix_bytes=len(raw), prefix_sha256=hashlib.sha256(raw).hexdigest(),
            records=entries, emission=deepcopy(emission),
            child_profile_sha256=digest(child_profile),
            record_operations_unchanged=True, synchronous_fsync_retained=True,
            timing_metadata_written_after_response_or_error_emission=not before_error,
            persistence_disposition='after_failure_before_error_emission' if before_error else 'after_successful_response_flush',
            status='log_failure' if any(not e['completed'] for e in entries) else 'recorded')
        value['summary'] = validate_profile(value, expected_enabled=self.instrumented,
                                             child_profile=child_profile)
        return value


def make_recorder(path, instrumented):
    return Recorder(path, instrumented)


def publish_probe(folder, request_id, value):
    """Atomic Windows publication of metadata only; preserve pending failures."""
    require(type(request_id) is str and re.fullmatch(r'request-[0-9]{8}', request_id) is not None,
            'Probe sidecar request ID')
    folder = Path(folder); folder.mkdir(exist_ok=True)
    target, temporary = folder/(request_id+'.json'), folder/(request_id+'.pending')
    encoded = json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()+b'\n'
    require(len(encoded) <= MAX_PROFILE_BYTES, 'Probe metadata byte bound')
    if target.exists(): raise FileExistsError(str(target))
    with temporary.open('xb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    # This study supports Windows: rename is atomic and refuses overwrite.
    os.rename(temporary,target)
    return target


def _scope(entry, profiles):
    start, end = entry['timestamps']['start_ns'], entry['timestamps']['digested_ns']
    if start is None or end is None:
        return dict(kind='unmeasured' if start is None else 'partial', handler_index=None)
    found = [i for i, p in enumerate(profiles)
             if p['handle_start_ns'] <= start <= end <= p['handle_end_ns']]
    require(len(found) <= 1, 'Overlapping original handler profiles')
    return dict(kind='inside_original_handler' if found else 'outside_original_handler',
                handler_index=found[0] if found else None)


def validate_profile(value, *, expected_enabled, child_profile=None):
    """Pure span validation; byte/chain authentication is validate_probe_sidecar."""
    require(type(expected_enabled) is bool and value['schema'] == SCHEMA and
            value['instrumented'] is expected_enabled, 'Logging probe schema/mode')
    require(type(value['start_record']) is int and type(value['end_record']) is int and
            0 <= value['start_record'] <= value['end_record'] <= MAX_RECORDS and
            len(value['records']) == value['end_record']-value['start_record'], 'Probe record inventory')
    require(value['record_operations_unchanged'] is True and value['synchronous_fsync_retained'] is True,
            'Probe scope changed')
    disposition = value['persistence_disposition']
    require(disposition in ('after_successful_response_flush','after_failure_before_error_emission') and
            value['timing_metadata_written_after_response_or_error_emission'] is
            (disposition == 'after_successful_response_flush'), 'Probe persistence scope changed')
    if child_profile is not None:
        require(value['child_profile_sha256'] == digest(child_profile), 'Probe child profile binding')
    profiles = [] if child_profile is None else child_profile['profiles']
    total = {name: 0 for name in STAGES} if expected_enabled else None
    rows, previous_end, failures = [], None, 0
    for entry in value['records']:
        require(type(entry['sequence']) is int and entry['sequence'] >= 0 and
                type(entry['event']) is str and type(entry['completed']) is bool and
                set(entry['timestamps']) == set(MARKERS), 'Probe record shape')
        times = [entry['timestamps'][name] for name in MARKERS]
        if not expected_enabled:
            require(all(t is None for t in times), 'Timer OFF fabricated timestamps')
            durations = None
        else:
            present = [t for t in times if t is not None]
            require(all(type(t) is int and t >= 0 for t in present) and
                    times == present+[None]*(len(times)-len(present)) and
                    present == sorted(present), 'Probe timestamp type/order/partial prefix')
            require(not present or previous_end is None or present[0] >= previous_end, 'Probe records overlap')
            if present: previous_end = present[-1]
            require(not entry['completed'] or len(present) == len(MARKERS), 'Completed record lacks stage timestamps')
            durations = {name: (times[i+1]-times[i] if times[i+1] is not None else None)
                         for i, name in enumerate(STAGES)}
            for name, duration in durations.items():
                if duration is not None: total[name] += duration
        if entry['completed']:
            require(entry['error'] is None, 'Completed forensic record claims error')
        else:
            failures += 1
            require(type(entry['error']) is dict and set(entry['error']) == {'type','detail'} and
                    all(type(v) is str for v in entry['error'].values()), 'Failed record missing original error')
        rows.append(dict(sequence=entry['sequence'], event=entry['event'], completed=entry['completed'],
            durations_ns=durations, scope=_scope(entry, profiles) if child_profile is not None else None))
    require(value['status'] == ('log_failure' if failures else 'recorded'), 'Probe status contradicts records')
    return dict(status='timing_evidence_valid' if not failures else 'partial_logging_failure_preserved',
        measurement_accepted=not failures, instrumented=expected_enabled, record_count=len(rows),
        failed_record_count=failures, durations_ns=total, records=rows,
        scope='forensic_operations_only_not_search_time_or_deadline_acceptance')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate forensic JSON key')
        result[key] = value
    return result


def validate_probe_sidecar(sidecar, logpath, expected_mode, request_id, child_profile):
    summary = validate_profile(sidecar, expected_enabled=expected_mode, child_profile=child_profile)
    require(sidecar['request_id'] == request_id and sidecar['log'] == Path(logpath).name,
            'Probe request/log identity')
    require(type(sidecar['prefix_bytes']) is int and 0 <= sidecar['prefix_bytes'] <= MAX_LOG_BYTES,
            'Probe prefix byte bound')
    require(Path(logpath).stat().st_size <= MAX_LOG_BYTES, 'Probe log oversized')
    raw = Path(logpath).read_bytes()
    require(len(raw) <= MAX_LOG_BYTES and len(raw) >= sidecar['prefix_bytes'], 'Probe log truncated/oversized')
    prefix = raw[:sidecar['prefix_bytes']]
    require(hashlib.sha256(prefix).hexdigest() == sidecar['prefix_sha256'], 'Probe log prefix hash')
    rows, previous = [], None
    for line in prefix.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            require(sidecar['status'] == 'log_failure', 'Incomplete successful forensic line')
            break
        row = json.loads(line.decode('utf-8'), object_pairs_hook=_unique)
        require(set(row) == {'sequence','previous_sha256','event'} and
                type(row['sequence']) is int and row['sequence'] == len(rows) and
                row['previous_sha256'] == previous, 'Forensic byte-chain sequence/digest')
        encoded = json.dumps(row, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()
        require(line in (encoded+b'\n', encoded+b'\r\n'), 'Forensic serialization bytes changed')
        previous = digest(row); rows.append(row)
    require(len(rows) <= MAX_RECORDS, 'Forensic row bound')
    for entry in sidecar['records']:
        if entry['completed']:
            require(entry['sequence'] < len(rows) and rows[entry['sequence']]['event']['event'] == entry['event'],
                    'Probe entry does not bind persisted event')
    if sidecar['status'] == 'recorded':
        require(len(rows) == sidecar['end_record'] and
                [e['sequence'] for e in sidecar['records']] ==
                list(range(sidecar['start_record'], sidecar['end_record'])), 'Probe complete record range differs')
    require(sidecar['summary'] == summary, 'Probe derived summary differs')
    emission = sidecar['emission']
    require(type(emission) is dict, 'Probe persistence markers missing')
    last = max((e['timestamps']['digested_ns'] or e['timestamps']['start_ns'] or 0
                for e in sidecar['records']), default=0)
    if sidecar['persistence_disposition'] == 'after_failure_before_error_emission':
        require(child_profile.get('outcome') == 'failed_before_successful_emission' and
                emission.get('kind') == 'before_error' and emission.get('pre_write_ns') is None and
                emission.get('post_flush_ns') is None and type(emission.get('prepared_ns')) is int and
                last <= emission['prepared_ns'], 'Failed probe must precede error emission without invented flush')
    else:
        require(type(emission.get('post_flush_ns')) is int and last <= emission['post_flush_ns'],
                'Probe span follows packet flush')
    return dict(**summary, prefix_verified=True, verified_bytes=len(prefix),
                persisted_rows=len(rows), original_player_outcome_not_inferred=True)
