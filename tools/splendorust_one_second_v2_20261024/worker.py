"""Supervised observer-only worker for the frozen one-second comparison.

The parent owns actual state, clocks, stage limits and independent replay. This
worker receives no referee seeds, hidden deck order, or opponent private cards.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.future_training_rules_20261021 import (
    RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT, verify_runtime_sources,
)
from tools.local_player_qualification_128.clock import THREAD_ENV, _decode
from .adapters import (
    GREEDY_PROFILE_ID, GREEDY_WRAPPER_ID, INTERNAL_SECONDS, PROFILE_SHA256,
    SEARCH_PROFILE_ID, SEARCH_WRAPPER_ID, initialize_adapters, require,
)


def _save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _forensic_recorder(path):
    """Append and sync each completed fixture before any checker can reject it."""
    stream = Path(path).open('x', encoding='utf-8')
    sequence = 0
    previous = None
    def record(event):
        nonlocal sequence, previous
        row = {'sequence': sequence, 'previous_sha256': previous, 'event': event}
        encoded = json.dumps(row, separators=(',', ':'), sort_keys=True, allow_nan=False)
        stream.write(encoded + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        previous = digest(row)
        sequence += 1
    return record, stream


def _forensic_action(action):
    try:
        return rules.serialize_action(action)
    except BaseException as error:
        return {'unserializable_action': repr(action), 'serialization_error': repr(error)}


def initialize():
    # Every new model worker authenticates the actual corrected runtime once.
    sources = verify_runtime_sources()
    folder = Path(os.environ['STUDY_WORKER_DIR']).resolve()
    require(folder.is_relative_to((ROOT / 'outputs').resolve()), 'worker evidence boundary')
    folder.mkdir(parents=True, exist_ok=True)
    runtime = {'sources': sources, 'ruleset_id': RULESET_ID,
               'catalogue_sha256': CATALOGUE_SHA256, 'turn_contract': TURN_CONTRACT}
    _save(folder / 'runtime_sources.json', runtime)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    adapters, identity = initialize_adapters(folder)
    metadata = {**identity, 'pid': os.getpid(), 'threads': 1, **runtime}
    _save(folder / 'player_identity.json', metadata)
    return {'adapters': adapters, 'metadata': metadata, 'folder': folder,
            'diagnostic_capture': False, 'request_started': None}


def decode_observer(observer):
    # Deliberately no legacy, snapshot, context-bearing, or fallback decoder.
    require(type(observer) is dict, 'observer must be an object')
    state = rules.state_from_observation(observer)
    require(rules.observation_payload(state) == observer, 'lawful observer roundtrip')
    require(not state.terminal, 'terminal position is not a player request')
    return state


def handle(context, request):
    if request == {'operation': 'entrypoint_checks'}:
        from .qualification import run_entrypoint_checks
        import psutil
        def admission_boundary():
            if psutil.virtual_memory().available < 2 * 1024**3:
                raise RuntimeError('STUDY_RESOURCE: qualification batch available RAM below 2 GiB')
        require(not context['diagnostic_capture'], 'nested qualification request')
        context['diagnostic_capture'] = True
        outer_started = context.get('request_started')
        context['request_started'] = None
        for adapter in context['adapters'].values():
            adapter.admission_guard = admission_boundary
        record, forensic_stream = _forensic_recorder(context['folder'] / 'entrypoint_forensics.jsonl')
        context['evidence_callback'] = record
        try:
            admission_boundary()
            return run_entrypoint_checks(context['adapters'], boundary=admission_boundary,
                                        request_callback=lambda payload: handle(context, payload),
                                        evidence_callback=record)
        except BaseException as error:
            record({'event': 'entrypoint_failed', 'error': repr(error),
                    'traceback': traceback.format_exc()})
            raise
        finally:
            forensic_stream.close()
            context.pop('evidence_callback', None)
            for adapter in context['adapters'].values():
                adapter.admission_guard = None
            context['diagnostic_capture'] = False
            context['request_started'] = outer_started

    started = context.get('request_started')
    if started is None:
        started = time.perf_counter()
    deadline = started + INTERNAL_SECONDS
    require(type(request) is dict and set(request) == {'player', 'observer'} and
            request['player'] in ('search', 'greedy'), 'request allowlist')
    observer, player = request['observer'], request['player']
    state = decode_observer(observer)
    legal = rules.CorrectedEnv().legal_actions(state)
    require(bool(legal), 'empty complete-action support')
    adapter = context['adapters'][player]
    adapter.request_deadline = deadline
    try:
        action, diagnostics = adapter.choose(observer,
            diagnostic_capture=context.get('diagnostic_capture', False))
        if context.get('evidence_callback') is not None:
            context['evidence_callback']({'event': 'complete_result_before_worker_validation',
                'arm': player, 'observer': observer, 'action': _forensic_action(action),
                'diagnostics': diagnostics})
        if action not in legal:
            raise ValueError('PLAYER_ILLEGAL_ACTION: returned action is not legal')
        profile_id = SEARCH_PROFILE_ID if player == 'search' else GREEDY_PROFILE_ID
        require(diagnostics['profile_id'] == profile_id, 'scientific profile identity')
        result = {'player': player, 'action': rules.serialize_action(action),
            'diagnostics': diagnostics, 'profile_id': profile_id,
            'profile_sha256': PROFILE_SHA256 if player == 'search' else None,
            'wrapper_id': SEARCH_WRAPPER_ID if player == 'search' else GREEDY_WRAPPER_ID,
            'observer_sha256': digest(observer),
            'legal_actions_sha256': digest([rules.serialize_action(a) for a in legal]),
            'legal_count': len(legal), 'ruleset': RULESET_ID,
            'catalogue_sha256': CATALOGUE_SHA256, 'turn_contract': TURN_CONTRACT}
        # Evidence JSON creation is charged before returning to the transport.
        json.dumps(result, separators=(',', ':'), allow_nan=False)
        if time.perf_counter() >= deadline:
            raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: complete response exceeded 0.900s')
        return result
    except BaseException as error:
        if context.get('evidence_callback') is not None:
            context['evidence_callback']({'event': 'worker_decision_failed', 'arm': player,
                'observer': observer, 'error': repr(error),
                'last_phase_callbacks': getattr(adapter, 'callbacks', None)})
        raise
    finally:
        adapter.request_deadline = None


def serve(*, max_request_bytes=1024 * 1024):
    """Same authenticated JSONL envelope as PersistentWorker, with ready identity."""
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
                    require(set(request) == {'type', 'request_id', 'payload'} and
                            request['type'] == 'request' and type(request['request_id']) is str,
                            'transport request envelope')
                    request_id = request['request_id']
                    result = handle(context, request['payload'])
                    deadline = (None if request['payload'] == {'operation': 'entrypoint_checks'}
                                else context['request_started'] + INTERNAL_SECONDS)
                    emit({'type': 'response', 'request_id': request_id, 'result': result}, deadline)
                except BaseException as error:
                    _save(context['folder'] / 'worker_error.json',
                        {'request_id': request_id, 'error': repr(error),
                         'traceback': traceback.format_exc()})
                    emit({'type': 'error', 'request_id': request_id, 'error': repr(error)[:4096]})
                    return 1
        except BaseException as error:
            emit({'type': 'error', 'error': repr(error)[:4096]})
            return 1
        finally:
            if context is not None:
                for adapter in context['adapters'].values():
                    adapter.close()


if __name__ == '__main__':
    raise SystemExit(serve())
