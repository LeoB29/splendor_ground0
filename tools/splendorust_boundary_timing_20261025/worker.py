"""Diagnostic-only envelope around the exact graph64 and eager32 workers.

Scientific stdout stays identical. Successful timing evidence is published only
after response flush; caught failures persist partial evidence before error
emission. A killed process may leave no sidecar, which is never inferred passed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

from tools.local_player_qualification_128.clock import THREAD_ENV, _decode
from tools.splendorust_search_scale_20261024 import worker as frozen
from tools.splendorust_search_scale_20261024.adapters import INTERNAL_SECONDS, require
from tools.splendorust_inference_opt_20261025 import worker as graph_worker
from .child_profile import (
    SCHEMA, MAX_BATCH_PROFILES,
    digest, publish_profile,
)

_EAGER_INITIALIZE = frozen.initialize
_EAGER_HANDLE = frozen.handle
_GRAPH_INITIALIZE = graph_worker.initialize
_GRAPH_HANDLE = graph_worker.handle


def initialize():
    require(os.environ.get('STUDY_BOUNDARY_PROFILE') == '1', 'Profiled worker requires explicit opt-in')
    quota = os.environ['STUDY_SEARCH_SIMULATIONS']
    require(quota in ('32', '64'), 'Only unchanged eager32 and qualified graph64 are allowed')
    context = (_GRAPH_INITIALIZE if quota == '64' else _EAGER_INITIALIZE)()
    context['boundary_handle_active'] = False
    context['component_backend'] = 'qualified_graph64' if quota == '64' else 'unchanged_eager32'
    context['component_backend_identity'] = (digest(context['frozen_forward'].identity)
                                             if quota == '64' else None)
    context['component_profiles'] = []
    try:
        frozen._save(context['folder'] / 'boundary_timing_identity.json', dict(
            schema=SCHEMA, backend=context['component_backend'], quota=int(quota),
            graph_identity_sha256=context['component_backend_identity'],
            metadata=context['metadata'], scientific_response_fields_unchanged=True,
            clock='perf_counter_ns', clock_implementation=time.get_clock_info('perf_counter').implementation,
            evidence='atomic exclusive request sidecars; successful sidecar after stdout flush'))
        return context
    except BaseException:
        close_context(context)
        raise


def handle(context, request):
    if type(request) is dict and request.get('operation') == 'graph_parity':
        require(context['quota'] == 64, 'Eager32 has no graph parity operation')
        return _GRAPH_HANDLE(context, request)
    if context['quota'] == 64:
        require(context['graph_parity_passed'], 'Candidate decisions require exact graph parity first')
    if request == {'operation': 'entrypoint_checks'}:
        # Calling graph_worker.handle here would replace this routing with its
        # own wrapper. Its parity gate above is retained; the exact original
        # checker instead resolves every ordinary callback to this handle.
        previous = frozen.handle
        frozen.handle = handle
        try:
            return _EAGER_HANDLE(context, request)
        finally:
            frozen.handle = previous
    require(not context['boundary_handle_active'], 'Nested timing handle')
    require(len(context['component_profiles']) < MAX_BATCH_PROFILES, 'Timing profile count bound')
    context['boundary_handle_active'] = True
    failed = True
    # Only these two clock reads wrap the entire scientific handle. No tensor,
    # model, callback, native-pipe, encoder, hash or CPU hooks are installed.
    started = time.perf_counter_ns()
    try:
        result = (_GRAPH_HANDLE if context['quota'] == 64 else _EAGER_HANDLE)(context, request)
        failed = False
        return result
    finally:
        ended = time.perf_counter_ns()
        context['boundary_handle_active'] = False
        context['component_profiles'].append(dict(schema=SCHEMA, pid=os.getpid(),
            clock='perf_counter_ns', measurement_units='nanoseconds',
            request_read_complete_ns=(context.get('request_read_complete_ns')
                if context.get('request_started') is not None else None),
            handle_start_ns=started, handle_end_ns=ended, handle_failed=failed,
            times_are_diagnostic_not_trusted_deadline_receipts=True))


def close_context(context):
    if context['quota'] == 64:
        graph_worker.close_context(context)
    else:
        for adapter in context['adapters'].values():
            adapter.close()


def make_evidence(context, request_id, raw, payload, result, *, outcome,
                  pre_write_ns=None, post_flush_ns=None, encoded=None, error=None):
    """Called after success emission, or before error emission on failure."""
    return dict(schema=SCHEMA, request_id=request_id, pid=os.getpid(),
        backend=context['component_backend'], quota=context['quota'],
        graph_identity_sha256=context['component_backend_identity'],
        clock='perf_counter_ns', clock_implementation=time.get_clock_info('perf_counter').implementation,
        request_read_complete_ns=context['request_read_complete_ns'],
        request_started_seconds=context['request_started'],
        pre_write_ns=pre_write_ns, post_flush_ns=post_flush_ns,
        outcome=outcome, error=error, profiles=list(context['component_profiles']),
        request_raw_sha256=hashlib.sha256(raw).hexdigest(),
        payload_sha256=None if payload is None else digest(payload),
        result_sha256=None if result is None else digest(result),
        serialized_packet_utf8_sha256=None if encoded is None else hashlib.sha256(encoded.encode()).hexdigest(),
        serialized_packet_utf8_bytes=None if encoded is None else len(encoded.encode()),
        text_transport_may_expand_newlines=True,
        success_sidecar_written_after_response_flush=True,
        times_are_diagnostic_not_trusted_deadline_receipts=True,
        post_flush_may_follow_parent_receipt=True,
        child_read_may_precede_parent_dispatch_flush_completion=True,
        killed_worker_sidecar_may_be_missing=True)


def save_sidecar_error(context, request_id, error, *, primary_error=None):
    # Disk failure cannot turn one successfully emitted packet into a second
    # packet, or replace an already caught scientific/deadline exception.
    try:
        frozen._save(context['folder'] / 'boundary_sidecar_error.json', dict(
            request_id=request_id, error=repr(error), primary_error=primary_error,
            traceback=traceback.format_exc()))
    except BaseException:
        pass  # The missing required sidecar is itself a collector failure.


def serve(*, max_request_bytes=1024 * 1024):
    for name in THREAD_ENV:
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits

    def emit(value, deadline=None, markers=None):
        encoded = json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n'
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: response serialization exceeded 0.900s')
        pre_write = time.perf_counter_ns()
        if markers is not None:
            markers['pre_write_ns'] = pre_write
        sys.stdout.write(encoded)
        sys.stdout.flush()
        post_flush = time.perf_counter_ns()
        if markers is not None:
            markers['post_flush_ns'] = post_flush
        return pre_write, post_flush, encoded

    context = None
    with threadpool_limits(limits=1):
        try:
            context = initialize()
            emit({'type': 'ready', 'metadata': context['metadata']})
            while True:
                raw = sys.stdin.buffer.readline(max_request_bytes + 1)
                if not raw:
                    return 0
                # Preserve the exact original float clock origin and ordering.
                context['request_started'] = time.perf_counter()
                context['request_read_complete_ns'] = time.perf_counter_ns()
                context['component_profiles'] = []
                request_id, payload, result = None, None, None
                emission_markers = {}
                try:
                    require(len(raw) <= max_request_bytes and raw.endswith(b'\n'), 'request size/termination')
                    request = _decode(raw)
                    require(set(request) == {'type', 'request_id', 'payload'} and request['type'] == 'request'
                            and type(request['request_id']) is str, 'transport request envelope')
                    request_id, payload = request['request_id'], request['payload']
                    result = handle(context, payload)
                    startup = (payload == {'operation': 'entrypoint_checks'} or
                        type(payload) is dict and set(payload) == {'operation', 'fixtures'} and
                        payload['operation'] == 'graph_parity')
                    deadline = None if startup else context['request_started'] + INTERNAL_SECONDS
                    pre_write, post_flush, encoded = emit(
                        {'type': 'response', 'request_id': request_id, 'result': result}, deadline, emission_markers)
                except BaseException as error:
                    frozen._save(context['folder'] / 'worker_error.json', dict(
                        request_id=request_id, error=repr(error), traceback=traceback.format_exc()))
                    if request_id is not None:
                        try:
                            evidence = make_evidence(context, request_id, raw, payload, result,
                                outcome='failed_before_successful_emission', error=repr(error)[:4096],
                                pre_write_ns=emission_markers.get('pre_write_ns'),
                                post_flush_ns=emission_markers.get('post_flush_ns'))
                            publish_profile(context['folder'], request_id, evidence,
                                            batch=len(context['component_profiles']) > 1)
                        except BaseException as sidecar_error:
                            save_sidecar_error(context, request_id, sidecar_error,
                                               primary_error=repr(error)[:4096])
                    emit({'type': 'error', 'request_id': request_id, 'error': repr(error)[:4096]})
                    return 1
                # Sidecar I/O cannot make an already emitted scientific
                # response into a second error packet. Missing evidence is a
                # separate diagnostic integrity failure, never a player retry.
                try:
                    evidence = make_evidence(context, request_id, raw, payload, result,
                        outcome='response_flushed', pre_write_ns=pre_write,
                        post_flush_ns=post_flush, encoded=encoded)
                    publish_profile(context['folder'], request_id, evidence, batch=startup)
                except BaseException as error:
                    save_sidecar_error(context, request_id, error)
                    return 1
        except BaseException as error:
            emit({'type': 'error', 'error': repr(error)[:4096]})
            return 1
        finally:
            if context is not None:
                close_context(context)


if __name__ == '__main__':
    raise SystemExit(serve())
