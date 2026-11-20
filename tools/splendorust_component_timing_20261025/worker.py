"""Thin diagnostic envelope around the unchanged corrected scientific worker."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

from tools.local_player_qualification_128.clock import THREAD_ENV, _decode
from tools.splendorust_search_scale_20261024 import worker as frozen
from tools.splendorust_search_scale_20261024.adapters import INTERNAL_SECONDS, require
from .child_profile import ChildProfiler, InstrumentedBindings, SCHEMA, validate_payload_size


_ORIGINAL_INITIALIZE = frozen.initialize
_ORIGINAL_HANDLE = frozen.handle


def initialize():
    mode = os.environ['STUDY_COMPONENT_PROFILE']
    require(mode in ('0', '1'), 'component instrumentation mode')
    # This is the exact corrected initializer: it calls verify_runtime_sources
    # once, sets both numerical thread limits, and loads the frozen adapters.
    context = _ORIGINAL_INITIALIZE()
    context['component_profile_enabled'] = mode == '1'
    context['component_bindings'] = None
    try:
        if mode == '1':
            context['component_bindings'] = InstrumentedBindings().install(context['adapters'])
        context['metadata'] = dict(context['metadata'],
            component_timing_mode='profiled' if mode == '1' else 'baseline')
        frozen._save(context['folder']/'component_timing_identity.json',
                     dict(schema=SCHEMA, metadata=context['metadata']))
        return context
    except BaseException:
        close_context(context)
        raise


def handle(context, request):
    context.pop('_failed_component_profile', None)
    if request == {'operation': 'entrypoint_checks'}:
        # Frozen entrypoint callbacks resolve `handle` in the frozen module.
        # Redirect only that process-local binding for the batch, then restore
        # it even on failure. Every ordinary decision reaches this wrapper.
        previous = frozen.handle
        frozen.handle = qualification_request
        try:
            return _ORIGINAL_HANDLE(context, request)
        finally:
            frozen.handle = previous
    if not context.get('component_profile_enabled'):
        return _ORIGINAL_HANDLE(context, request)
    bindings = context['component_bindings']
    require(bindings.active is None, 'nested component profiling request')
    started = context.get('request_started')
    deadline = (time.perf_counter() if started is None else started)+INTERNAL_SECONDS
    profiler = ChildProfiler(request_read_complete_ns=(
        context.get('request_read_complete_ns') if started is not None else None))
    bindings.active = profiler
    failed = True
    try:
        result = _ORIGINAL_HANDLE(context, request)
        failed = False
    finally:
        bindings.active = None
        profiler.finish(failed=failed)
        if failed:
            context['_failed_component_profile'] = profiler.snapshot()
    profile = profiler.snapshot()
    context['_failed_component_profile'] = profile
    validate_payload_size(profile)
    # The unchanged internal deadline also charges constructing the extra
    # diagnostic envelope, including counterexample decisions with no emit.
    if time.perf_counter() >= deadline:
        raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: component envelope exceeded0.900s')
    result['timing_profile'] = profile
    context.pop('_failed_component_profile', None)
    return result


def qualification_request(context, payload):
    result = handle(context, payload)
    profile = result.pop('timing_profile', None)
    if profile is not None:
        context['evidence_callback']({'event': 'qualification_timing_profile', 'profile': profile})
    return result


def close_context(context):
    bindings = context.get('component_bindings')
    try:
        if bindings is not None:
            bindings.close()
    finally:
        for adapter in context['adapters'].values():
            adapter.close()


def serve(*, max_request_bytes=1024*1024):
    """Preserve raw-request clock origin, protocol and final emission deadline."""
    for name in THREAD_ENV:
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits

    def emit(value, deadline=None):
        profile = value.get('result', {}).get('timing_profile') if isinstance(value.get('result'), dict) else None
        if profile is not None:
            # This is deliberately BEFORE JSON serialization. The following
            # gap is not described as pure pipe time or serialized-byte time.
            profile['before_emit_ns'] = time.perf_counter_ns()
        encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: response serialization exceeded0.900s')
        sys.stdout.write(encoded)
        sys.stdout.flush()

    context = None
    with threadpool_limits(limits=1):
        try:
            context = initialize()
            emit({'type':'ready', 'metadata':context['metadata']})
            while True:
                raw = sys.stdin.buffer.readline(max_request_bytes+1)
                if not raw:
                    return 0
                # Exact original clock origin: immediately after the complete
                # raw request, before JSON parsing or any model work.
                context['request_started'] = time.perf_counter()
                if context['component_profile_enabled']:
                    context['request_read_complete_ns'] = time.perf_counter_ns()
                request_id, result = None, None
                context.pop('_failed_component_profile', None)
                try:
                    require(len(raw)<=max_request_bytes and raw.endswith(b'\n'), 'request size/termination')
                    request = _decode(raw)
                    require(set(request)=={'type','request_id','payload'} and request['type']=='request' and
                            type(request['request_id']) is str, 'transport request envelope')
                    request_id = request['request_id']
                    result = handle(context, request['payload'])
                    deadline = None if request['payload']=={'operation':'entrypoint_checks'} else context['request_started']+INTERNAL_SECONDS
                    emit({'type':'response','request_id':request_id,'result':result}, deadline)
                except BaseException as error:
                    frozen._save(context['folder']/'worker_error.json', dict(request_id=request_id,
                        error=repr(error),traceback=traceback.format_exc(),
                        failed_timing_profile=(result.get('timing_profile') if isinstance(result,dict)
                                               else context.get('_failed_component_profile'))))
                    emit({'type':'error','request_id':request_id,'error':repr(error)[:4096]})
                    return 1
        except BaseException as error:
            emit({'type':'error','error':repr(error)[:4096]})
            return 1
        finally:
            if context is not None:
                close_context(context)


if __name__=='__main__':
    raise SystemExit(serve())
