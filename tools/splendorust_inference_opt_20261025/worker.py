"""Separate exact-science worker using the single frozen tensor-forward graph."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

from tools.local_player_qualification_128.clock import THREAD_ENV, _decode
from tools.splendorust_search_scale_20261024 import worker as frozen
from tools.splendorust_search_scale_20261024.adapters import INTERNAL_SECONDS, require
from . import graph


_ORIGINAL_INITIALIZE = frozen.initialize
_ORIGINAL_HANDLE = frozen.handle


def initialize():
    # This unchanged initializer authenticates corrected runtime sources once,
    # sets both numerical thread limits, and loads the same eager checkpoint.
    context = _ORIGINAL_INITIALIZE()
    context['inference_binding'] = None
    context['graph_parity_passed'] = False
    context['graph_parity_attempted'] = False
    try:
        adapters = context['adapters']
        require(bool(adapters) and len({id(adapter.model) for adapter in adapters.values()}) == 1,
                'Graph initialization requires one shared unchanged model')
        forward = graph.build_graph(next(iter(adapters.values())).model,
                                    context['folder'], frozen._save)
        context['frozen_forward'] = forward
        context['inference_binding'] = graph.InferenceBinding().install(adapters, forward)
        context['metadata'] = dict(context['metadata'], inference_backend=graph.BACKEND)
        frozen._save(context['folder'] / 'inference_execution_identity.json',
                     {'backend': graph.BACKEND, 'graph_identity_sha256': graph.digest(forward.identity),
                      'metadata': context['metadata']})
        return context
    except BaseException as error:
        try:
            frozen._save(context['folder'] / 'graph_initialization_error.json',
                         {'error': repr(error), 'traceback': traceback.format_exc()})
        finally:
            close_context(context)
        raise


def graph_parity(context, request):
    require(not context['graph_parity_attempted'], 'Graph parity cannot be retried')
    context['graph_parity_attempted'] = True
    rows, evidence = [], []
    fixtures = request.get('fixtures')
    forward = context['frozen_forward']
    payload_sha = graph.digest(fixtures)
    base = {'schema': graph.PARITY_SCHEMA,
            'graph_identity_sha256': graph.digest(forward.identity),
            'fixture_payload_sha256': payload_sha}
    try:
        require(set(request) == {'operation', 'fixtures'}, 'Graph parity request allowlist')
        graph.validate_parity_fixtures(fixtures)
        binding = context['inference_binding']
        adapter = context['adapters']['search']
        for fixture in fixtures:
            item = {'fixture_id': fixture['fixture_id'], 'batch': fixture['batch'],
                    'eager_outputs': None, 'candidate_outputs': None}
            evidence.append(item)
            # These eager shadow calls are exclusively an explicit bounded
            # startup qualification operation, never a timed playing decision.
            item['eager_outputs'] = binding.original(adapter, fixture['tokens'], fixture['batch'])
            item['candidate_outputs'] = forward.infer(adapter, fixture['tokens'], fixture['batch'])
            row = graph.compare_outputs(fixture, item['eager_outputs'], item['candidate_outputs'])
            item['comparison'] = row
            rows.append(row)
        passed = all(row['exact_float_equal'] and row['float32_bits_equal'] for row in rows)
        result = dict(base, status='passed' if passed else 'failed', passed=passed,
                      fixtures_count=len(rows), rows=rows)
        # All actual values survive a negative result. Set the admission flag
        # only after exclusive, flushed evidence persistence succeeds.
        frozen._save(context['folder'] / 'graph_parity.json',
                     {'schema': graph.PARITY_SCHEMA, 'summary': result,
                      'fixtures': fixtures, 'evidence': evidence})
        context['graph_parity_passed'] = passed
        return result
    except BaseException as error:
        if not (context['folder'] / 'graph_parity.json').exists():
            frozen._save(context['folder'] / 'graph_parity.json',
                         {'schema': graph.PARITY_SCHEMA, 'summary': dict(base, status='error', passed=False,
                          fixtures_count=len(rows), rows=rows), 'fixtures': fixtures,
                          'evidence': evidence, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def handle(context, request):
    if type(request) is dict and request.get('operation') == 'graph_parity':
        return graph_parity(context, request)
    require(context['graph_parity_passed'], 'Candidate decisions require exact graph parity first')
    if request == {'operation': 'entrypoint_checks'}:
        # Counterexample callbacks resolve the frozen module's handle. Route
        # each of its 52 ordinary requests through this admitted candidate
        # handler, preserving the original checker and its scientific fields.
        previous = frozen.handle
        frozen.handle = handle
        try:
            return _ORIGINAL_HANDLE(context, request)
        finally:
            frozen.handle = previous
    return _ORIGINAL_HANDLE(context, request)


def close_context(context):
    binding = context.get('inference_binding')
    try:
        if binding is not None:
            binding.close()
    finally:
        for adapter in context['adapters'].values():
            adapter.close()


def serve(*, max_request_bytes=1024 * 1024):
    """Keep the raw-request clock origin and the original 0.9 s emission gate."""
    for name in THREAD_ENV:
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits

    def emit(value, deadline=None):
        encoded = json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n'
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: response serialization exceeded 0.900s')
        sys.stdout.write(encoded)
        sys.stdout.flush()

    context = None
    with threadpool_limits(limits=1):
        try:
            context = initialize()
            emit({'type': 'ready', 'metadata': context['metadata']})
            while True:
                raw = sys.stdin.buffer.readline(max_request_bytes + 1)
                if not raw:
                    return 0
                context['request_started'] = time.perf_counter()
                request_id = None
                try:
                    require(len(raw) <= max_request_bytes and raw.endswith(b'\n'), 'request size/termination')
                    request = _decode(raw)
                    require(set(request) == {'type', 'request_id', 'payload'} and request['type'] == 'request'
                            and type(request['request_id']) is str, 'transport request envelope')
                    request_id = request['request_id']
                    payload = request['payload']
                    result = handle(context, payload)
                    startup_operation = (payload == {'operation': 'entrypoint_checks'} or
                        type(payload) is dict and set(payload) == {'operation', 'fixtures'} and
                        payload['operation'] == 'graph_parity')
                    deadline = None if startup_operation else context['request_started'] + INTERNAL_SECONDS
                    emit({'type': 'response', 'request_id': request_id, 'result': result}, deadline)
                except BaseException as error:
                    frozen._save(context['folder'] / 'worker_error.json', {'request_id': request_id,
                        'error': repr(error), 'traceback': traceback.format_exc()})
                    emit({'type': 'error', 'request_id': request_id, 'error': repr(error)[:4096]})
                    return 1
        except BaseException as error:
            emit({'type': 'error', 'error': repr(error)[:4096]})
            return 1
        finally:
            if context is not None:
                close_context(context)


if __name__ == '__main__':
    raise SystemExit(serve())
