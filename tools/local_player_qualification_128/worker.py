"""The child never reads saved roots or expected decisions."""
import sys
import json

from tools.local_player_qualification_128.clock import serve


def initialize():
    from tools.local_player_qualification_128 import local
    metadata = local.initialize()
    return local, metadata


def handle(context, payload):
    local, _ = context
    if set(payload) != {"observer", "breadth"}:
        raise ValueError("Unexpected request fields")
    calls = 0

    def progress():
        nonlocal calls
        calls += 1
        if calls == 1 or calls == 2 or (calls - 2) % 16 == 0:
            sys.stderr.write(json.dumps({"event": "progress", "completed_worlds": max(0, calls - 2),
                                         "worlds": 128}) + "\n")
            sys.stderr.flush()

    return local.choose(payload["observer"], payload["breadth"], boundary=progress)


if __name__ == "__main__":
    raise SystemExit(serve(initialize, handle))
