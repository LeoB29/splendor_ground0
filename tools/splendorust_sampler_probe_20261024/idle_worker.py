"""Authenticated Python/native process shape, without loading a neural model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_search_scale_20261024 import common as frozen


def parse_echo_request(raw):
    frozen.require(len(raw) <= 4096 and raw.endswith(b'\n'), 'Echo request size')
    request = json.loads(raw)
    frozen.require(type(request) is dict and set(request) == {'type', 'request_id', 'payload'} and
        request['type'] == 'request' and type(request['request_id']) is str, 'Echo envelope')
    payload = request['payload']
    frozen.require(type(payload) is dict and set(payload) == {'operation', 'bytes'} and
        payload['operation'] == 'synthetic_echo' and type(payload['bytes']) is int and
        payload['bytes'] in (1024, 16384, 65536), 'Fixed synthetic echo allowlist')
    return request['request_id'], payload['bytes']


def main():
    sources = verify_runtime_sources()
    selected = int(os.environ['PROBE_SELECTED_CPU'])
    psutil.Process().cpu_affinity([selected])
    expected = json.loads(os.environ['PROBE_NATIVE_ARTIFACTS'])
    frozen.verify(expected)
    deadline = float(os.environ['PROBE_MONOTONIC_DEADLINE'])
    children = []
    try:
        for artifact in expected:
            frozen.require(time.monotonic() < deadline, 'Probe startup deadline')
            child = subprocess.Popen([str(frozen.ROOT/artifact['path'])], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            children.append(child)
            # The outer authenticated worker owns a 30-second startup clock.
            # No request is sent: this native bridge only emits readiness and
            # waits for stdin, without loading weights or invoking inference.
            raw = child.stdout.readline(16385)
            frozen.require(len(raw) <= 16384 and raw.endswith(b'\n'), 'Native readiness size')
            packet = json.loads(raw)
            frozen.require(packet == {'kind': 'ready', 'bridge': 'observer_only_fixed_work_v1',
                'source_pin': '1828507b8d2f0e02bc03d102b9c6d29a54fa8d57'}, 'Native readiness identity')
        frozen.require('torch' not in sys.modules, 'Model-free helper imported Torch')
        metadata = dict(probe_only=True, model_calls=0, native_requests=0, sources=sources,
            pid=os.getpid(), native_pids=[child.pid for child in children], native_artifacts=expected,
            affinity=[selected], model_loaded=False)
        print(json.dumps({'type': 'ready', 'metadata': metadata}), flush=True)
        # Only a fixed synthetic string can be requested. No observer or model
        # payload is accepted or forwarded to either native executable.
        while True:
            raw = sys.stdin.buffer.readline(4097)
            if not raw:
                break
            request_id = None
            try:
                request_id, size = parse_echo_request(raw)
                frozen.require(time.monotonic() < deadline and 'torch' not in sys.modules, 'Echo bound/model-free gate')
                result = dict(operation='synthetic_echo', bytes=size, payload='x'*size,
                              model_calls=0, native_requests=0)
                print(json.dumps({'type': 'response', 'request_id': request_id, 'result': result},
                                 separators=(',', ':')), flush=True)
            except BaseException as error:
                print(json.dumps({'type': 'error', 'request_id': request_id, 'error': repr(error)}), flush=True)
                return 1
        return 0
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.kill()
            try:
                child.wait(timeout=max(0., min(1., deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                pass  # The outer Windows job is the mandatory final owner.


if __name__ == '__main__':
    raise SystemExit(main())
