"""Separate verification allowance, outside the player's thinking-time measurement."""
import json
from pathlib import Path
import sys

from tools.local_player_qualification_128.clock import serve
from tools.corrected_planner_qualification_20261018 import common as c


def initialize():
    from tools.corrected_planner_qualification_20261018 import local
    metadata = local.initialize()
    return local, metadata


def handle(context, payload):
    from tools.corrected_planner_qualification_20261018.verify import audit_decision
    if set(payload) != {'observer', 'result', 'traces_path'}:
        raise ValueError('Unexpected audit request')
    path = Path(payload['traces_path']).resolve()
    if not path.is_relative_to(c.OUTPUT.resolve()) or path.name != 'continuations.jsonl.gz':
        raise ValueError('Audit trace outside current qualification')
    calls = 0
    def progress():
        nonlocal calls
        calls += 1
        if calls == 1 or calls % 16 == 0:
            sys.stderr.write(json.dumps({'event': 'audit_progress', 'boundary_calls': calls}) + '\n')
            sys.stderr.flush()
    return audit_decision(payload['observer'], payload['result'], path, boundary=progress)


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle, max_request_bytes=4 * 1024**2))
