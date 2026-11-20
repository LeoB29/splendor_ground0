"""Timed planner receives only the lawful observer and writes sealed trace evidence."""
import gzip
import json
import os
from pathlib import Path
import sys

from tools.local_player_qualification_128.clock import serve
from tools.corrected_planner_qualification_20261018 import common as c


def initialize():
    from tools.corrected_planner_qualification_20261018 import local
    metadata = local.initialize()
    return local, metadata


def handle(context, payload):
    local, _ = context
    if set(payload) != {'observer', 'breadth'} or payload['breadth'] is not False:
        raise ValueError('Unexpected request fields or player')
    path = Path(os.environ['CORRECTED_PLANNER_TRACE_PATH']).resolve()
    if not path.is_relative_to(c.OUTPUT.resolve()) or path.name != 'continuations.jsonl.gz':
        raise ValueError('Trace path outside new qualification output')
    calls, count = 0, 0

    def progress():
        nonlocal calls
        calls += 1
        if calls <= 2 or (calls - 2) % 16 == 0:
            sys.stderr.write(json.dumps({'event': 'progress', 'boundary_calls': calls}) + '\n')
            sys.stderr.flush()

    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
            def trace_sink(row):
                nonlocal count
                compressed.write((c.canonical(row) + '\n').encode())
                count += 1
            decision = local.choose(payload['observer'], False, boundary=progress, trace_sink=trace_sink)
        raw.flush()
        os.fsync(raw.fileno())
    return {'decision': decision, 'trace': {'sha256': c.file_hash(path), 'bytes': path.stat().st_size,
                                           'records': count},
            'ruleset_id': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
            'profile_id': c.PROFILE_ID, 'observer_sha256': c.digest(payload['observer'])}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle))
