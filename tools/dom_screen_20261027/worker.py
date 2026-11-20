"""Lawful-observer-only Dom policy worker with unchanged boundary transport."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from tools.local_player_qualification_128.clock import THREAD_ENV, _decode
from tools.splendorust_boundary_timing_20261025.child_profile import SCHEMA, digest, publish_profile
from tools.future_training_rules_20261021 import RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT, verify_runtime_sources
from tools.external_strength_screen_20261018 import rules
from . import common as c


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_worker(path, value):
    return c.save(path, value)


def initialize():
    sources = verify_runtime_sources()
    folder = Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder.is_relative_to(c.OUT.resolve()), 'Dom worker output boundary')
    require(os.environ.get('STUDY_BOUNDARY_PROFILE') == '1', 'Boundary evidence required')
    from .network import FrozenPolicy, WEIGHT_SHA256
    from . import adapter, numerical_runtime
    numerical=numerical_runtime.verify_runtime(c.read(c.OUT/'freeze.json')['numpy_blas_runtime'],require_one_thread=True)
    save_worker(folder/'numerical_runtime.json',numerical)
    policy = FrozenPolicy()
    profile = c.read(c.OUT/'opponent_profile.json')
    metadata = dict(player='dom', profile_id=adapter.PROFILE_ID,
        profile_sha256=c.file_hash(c.OUT/'opponent_profile.json'), pid=os.getpid(), threads=1,
        sources=sources, ruleset_id=RULESET_ID, catalogue_sha256=CATALOGUE_SHA256,
        turn_contract=TURN_CONTRACT, weight_sha256=WEIGHT_SHA256,
        no_search=True, no_training=True, historical_weights=True,
        numpy_blas_runtime_sha256=c.digest(numerical['contract']))
    save_worker(folder/'runtime_sources.json', metadata)
    save_worker(folder/'player_identity.json', metadata)
    save_worker(folder/'boundary_timing_identity.json', dict(schema=SCHEMA,
        backend='dom_numpy_policy', quota=None, graph_identity_sha256=None,
        metadata=metadata, scientific_response_fields_unchanged=True,
        clock='perf_counter_ns', clock_implementation=time.get_clock_info('perf_counter').implementation,
        evidence='atomic exclusive request sidecars; successful sidecar after stdout flush'))
    return dict(policy=policy, metadata=metadata, profile=profile, folder=folder,
        qualification_plan=c.read(c.OUT/'qualification_plan.json'),
        component_backend='dom_numpy_policy', component_backend_identity=None,
        component_profiles=[], quota=None, request_started=None, diagnostic_capture=False)


def handle(context, request):
    if request == {'operation': 'entrypoint_checks'}:
        from .qualification import run_entrypoint_checks
        import psutil
        def boundary():
            require(psutil.virtual_memory().available >= 2*c.GiB,
                    'STUDY_RESOURCE: qualification RAM reserve')
        require(not context['diagnostic_capture'], 'Nested qualification request')
        context['diagnostic_capture'] = True
        try:
            return run_entrypoint_checks(context, lambda payload: handle(context, payload), boundary=boundary)
        finally:
            context['diagnostic_capture'] = False
    require(type(request) is dict and set(request) == {'player','observer'}
            and request['player'] == 'dom', 'Dom request allowlist')
    started = time.perf_counter_ns()
    failed = True
    try:
        from . import adapter
        observer = request['observer']
        state = rules.state_from_observation(observer)
        require(rules.observation_payload(state) == observer, 'Lawful observer roundtrip')
        require(not state.terminal, 'Terminal position cannot request a decision')
        legal = rules.CorrectedEnv().legal_actions(state)
        chosen = adapter.decide(observer, context['policy'].predict)
        action = chosen['action']
        if isinstance(action, dict):
            action = rules.deserialize_action(action)
        require(action in legal, 'PLAYER_ILLEGAL_ACTION: Dom translated action')
        require(chosen['observer_sha256'] == digest(observer), 'Adapter observer binding')
        require(chosen['legal_actions_sha256'] == digest([rules.serialize_action(a) for a in legal]),
                'Adapter corrected legal-support binding')
        result = chosen
        json.dumps(result, separators=(',', ':'), allow_nan=False)
        failed = False
        return result
    finally:
        ended = time.perf_counter_ns()
        if not context['diagnostic_capture']:
            context['component_profiles'].append(dict(schema=SCHEMA, pid=os.getpid(),
                clock='perf_counter_ns', measurement_units='nanoseconds',
                request_read_complete_ns=context.get('request_read_complete_ns'),
                handle_start_ns=started, handle_end_ns=ended, handle_failed=failed,
                times_are_diagnostic_not_trusted_deadline_receipts=True))


def close_context(context):
    pass


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
        save_worker(context['folder'] / 'boundary_sidecar_error.json', dict(
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
                    deadline = None  # The unchanged parent clock enforces the complete 1.000-second response.
                    pre_write, post_flush, encoded = emit(
                        {'type': 'response', 'request_id': request_id, 'result': result}, deadline, emission_markers)
                except BaseException as error:
                    save_worker(context['folder'] / 'worker_error.json', dict(
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
