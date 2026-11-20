"""Fresh worker; unchanged pilot delivery contract and new output boundary.

Source: tools/live_game_operational_20261019/worker.py.
Only common/output plumbing and authenticated read-only old trace admission differ.
"""
import gzip
import json
import os
from pathlib import Path
import sys

from tools.local_player_qualification_128.clock import serve
from tools.live_game_operational_recovery_20261020 import common as c


def initialize():
    mode = os.environ['LIVE_WORKER_MODE']
    if mode not in ('local', 'audit'):
        raise ValueError('Unsupported live worker mode')
    from tools.live_game_operational_20261019 import local
    metadata = local.initialize()
    return {'mode': mode, 'local': local, 'metadata': metadata, 'used': False}


def trace_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(c.OUTPUT.resolve()) or path.name != 'continuations.jsonl.gz':
        raise ValueError('Trace outside approved live output')
    return path


def audit_trace_path(value):
    """Only audits may read a completed trace from the preserved source tree."""
    path=Path(value).resolve()
    if path.is_relative_to(c.OUTPUT.resolve()):
        return trace_path(path)
    if (not path.is_relative_to((c.SOURCE/'workers/games').resolve()) or
            path.name!='continuations.jsonl.gz'):
        raise ValueError('Audit trace outside recovery or preserved game-worker evidence')
    c.authenticated_source_artifact(path)
    return path


def handle(context, payload):
    if context['used']:
        raise ValueError('Exactly one request is allowed per fresh worker')
    context['used'] = True
    calls = 0
    def progress():
        nonlocal calls
        calls += 1
        if calls <= 2 or (calls - 2) % 16 == 0:
            sys.stderr.write(json.dumps({'event': 'progress', 'boundary_calls': calls}) + '\n')
            sys.stderr.flush()
    if context['mode'] == 'audit':
        if (not isinstance(payload, dict) or set(payload) != {'observer', 'result', 'traces_path', 'mode'} or
                payload['mode'] not in ('sample', 'full')):
            raise ValueError('Unexpected audit request')
        from tools.live_game_operational_20261019.verify import audit_decision
        path = audit_trace_path(payload['traces_path'])
        return audit_decision(payload['observer'], payload['result'], path,
                              mode=payload['mode'], boundary=progress)
    if not isinstance(payload, dict) or set(payload) != {'observer'}:
        raise ValueError('Unexpected local request')
    path = trace_path(os.environ['LIVE_TRACE_PATH'])
    count = 0
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
            def trace_sink(row):
                nonlocal count
                compressed.write((c.canonical(row) + '\n').encode())
                count += 1
            decision = context['local'].choose(payload['observer'], boundary=progress, trace_sink=trace_sink)
        raw.flush()
        os.fsync(raw.fileno())
    return {'decision': decision, 'trace': {'sha256': c.file_hash(path), 'bytes': path.stat().st_size,
                                           'records': count},
            'ruleset_id': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
            'profile_id': c.LOCAL_PROFILE_ID, 'observer_sha256': c.digest(payload['observer'])}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle, max_request_bytes=16 * 1024**2))
