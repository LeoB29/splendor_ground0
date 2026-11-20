"""Fresh worker; unchanged pilot delivery contract and new output boundary.

Source: tools/live_game_operational_20261018/worker.py.
"""
import gzip
import json
import os
from pathlib import Path
import sys

from tools.local_player_qualification_128.clock import serve
from tools.matched_strength_recovery_20261022 import common as c


def authenticate_runtime_sources():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    path = Path(os.environ['LIVE_RUNTIME_SOURCE_PATH']).resolve()
    if not path.is_relative_to(c.OUTPUT.resolve()) or path.name != 'runtime_sources.json':
        raise ValueError('Runtime source receipt outside recovery output')
    c.save_new(path, {'schema': 'corrected_runtime_sources_v1', 'ruleset_id': c.RULESET_ID,
                     'catalogue_sha256': c.CATALOGUE_SHA256, 'sources': verify_runtime_sources()})


def initialize():
    mode = os.environ['LIVE_WORKER_MODE']
    if mode not in ('local', 'audit'):
        raise ValueError('Unsupported live worker mode')
    authenticate_runtime_sources()
    from tools.matched_strength_20261020 import local
    metadata = local.initialize()
    return {'mode': mode, 'local': local, 'metadata': metadata, 'used': False}


def trace_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(c.OUTPUT.resolve()) or path.name != 'continuations.jsonl.gz':
        raise ValueError('Trace outside approved live output')
    return path


def audit_trace_path(value):
    path = Path(value).resolve()
    if path.name != 'continuations.jsonl.gz' or not any(path.is_relative_to(base.resolve())
            for base in (c.OUTPUT, c.SOURCE/'workers')):
        raise ValueError('Audit trace outside authenticated source/recovery boundaries')
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
        from tools.matched_strength_20261020.verify import audit_decision
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
