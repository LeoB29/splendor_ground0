"""Saved-only protocol integration; synthetic bulk packets are not model evidence.

No worker constructor, native executable, network forward or game collection is
invoked. The actual parent capture and child publication implementations remain
in the path; only the inherited process send/receive endpoint is a test double.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_search512_20261101 import (
    common as c, control, parent_profile, transport, worker,
)
from tools.splendorust_search512_accounting_20261101 import WorkerAccounting, LOG_NAMES
from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile


def captured_exchange(monkeypatch, bulk_bytes, *, crlf=False, enforce_packet=True):
    """Exercise the real capture wrapper with a deterministic opaque wire probe."""
    rid = 'request-00000001'
    payload = {'operation': 'graph_parity', 'fixtures': []}
    # This opaque size fixture intentionally makes no scientific claim. The
    # separate saved-graph test below validates authentic scientific evidence.
    result = {'fixture_kind': 'synthetic_transport_only', 'bulk': 'x' * bulk_bytes}
    response = dict(type='response', request_id=rid, result=result)
    encoded = (worker.response_packet(response) if enforce_packet else
               json.dumps(response, separators=(',', ':')) + '\n')
    raw_response = (encoded[:-1] + '\r\n' if crlf else encoded).encode()
    raw_request = (json.dumps(dict(type='request', request_id=rid, payload=payload),
                              separators=(',', ':')) + '\n').encode()
    capture = object.__new__(parent_profile.ProfiledStudyWorker)
    capture._boundary_context = None
    capture._boundary_request_lock = threading.Lock()
    capture.last_parent_profile = None
    capture.process = SimpleNamespace(pid=os.getpid(), stdin=io.BytesIO())
    capture.process.stdin = parent_profile._BoundaryWriteCapture(capture, capture.process.stdin)
    observed = {}

    def fake_request(self, *args, **kwargs):
        start = time.perf_counter_ns()
        self._boundary_context['request_start_ns'] = start
        self.process.stdin.write(raw_request)
        self.process.stdin.flush()
        origin = time.perf_counter()
        read = time.perf_counter_ns()
        pre = time.perf_counter_ns()
        post = time.perf_counter_ns()
        arrival = time.perf_counter_ns()
        self._put(('line', arrival, raw_response))
        observed.update(origin=origin, read=read, pre=pre, post=post)
        return dict(request_id=rid, clock='perf_counter_ns', status='ok',
                    accepted=True, received_before_deadline=True, deadline_seconds=60.,
                    receipt_elapsed_seconds=(arrival-start)/1e9,
                    elapsed_seconds=(time.perf_counter_ns()-start)/1e9,
                    result=result, telemetry={'process_tree': [{'role': 'python', 'pid': os.getpid()}]})

    monkeypatch.setattr(parent_profile.CandidateStudyWorker, 'request', fake_request)
    monkeypatch.setattr(parent_profile.CandidateStudyWorker, '_put', lambda self, event: None)
    receipt = capture.request(payload)
    context = dict(component_backend='qualified_graph64', quota=512,
                   component_backend_identity=transport.GRAPH_IDENTITY,
                   request_read_complete_ns=observed['read'], request_started=observed['origin'],
                   component_profiles=[])
    child = worker.make_evidence(context, rid, raw_request, payload, result,
                                 outcome='response_flushed', pre_write_ns=observed['pre'],
                                 post_flush_ns=observed['post'], encoded=encoded)
    return payload, receipt, capture.last_parent_profile, child, raw_response


@pytest.mark.parametrize('crlf', [False, True])
def test_large512_capture_boundary_and_real_publication(monkeypatch, tmp_path, crlf):
    payload, receipt, parent, child, raw = captured_exchange(monkeypatch, 5*c.MiB, crlf=crlf)
    assert 4*c.MiB < len(raw) <= c.MAX_RESPONSE_BYTES == 8*c.MiB
    assert parent['response_raw_bytes'] == len(raw)
    assert parent['response_raw_sha256'] == hashlib.sha256(raw).hexdigest()
    assert parent['response_capture_overflow'] is False
    original = deepcopy(receipt)
    proof = transport.validate_profiles(parent, child, receipt, payload, 'candidate', worker_pid=os.getpid())
    assert proof['status'] == 'passed' and proof['original_receipt_unmodified'] is True
    assert receipt == original

    # Real lifecycle accounting and atomic sidecar publication catch quota or
    # stream-cap mismatches before any future scientific request can launch.
    folder = tmp_path/'workers'/'candidate_wire_fixture'
    accounting = WorkerAccounting(folder)
    folder.mkdir(parents=True)
    process_double = SimpleNamespace(output_dir=folder, _log_lock=threading.Lock(),
                                    _logs={name: None for name in LOG_NAMES},
                                    max_log_bytes=c.MAX_LOG_BYTES, _closed=False)
    assert c.MAX_LOG_BYTES == 256*c.MiB
    accounting.bind_child(process_double)
    assert accounting.finish_initialization()['state'] == 'idle'
    request = accounting.begin_request(receipt['request_id'], payload, pid=os.getpid(), quota=512,
                                       graph_identity_sha256=transport.GRAPH_IDENTITY)
    assert request['publication']['expected']['quota'] == 512
    target = publish_profile(folder, receipt['request_id'], child, batch=True)
    finished = accounting.finish_request(require_published=True)
    assert finished['state'] == 'idle' and finished['metadata_bytes'] == target.stat().st_size
    assert finished['reserved_bytes'] == 0 and not list(folder.glob('*.pending'))
    assert c.read(target) == child
    process_double._closed = True
    assert accounting.close()['state'] == 'closed'

    tampered = deepcopy(child)
    tampered['serialized_packet_utf8_sha256'] = '0'*64
    if not crlf:  # Equal lengths require the actual byte hash, not JSON equality.
        with pytest.raises(ValueError, match='raw response hash'):
            transport.validate_profiles(parent, tampered, receipt, payload, 'candidate')
    tampered['request_raw_sha256'] = '0'*64
    with pytest.raises(ValueError, match='request hash'):
        transport.validate_profiles(parent, tampered, receipt, payload, 'candidate')


def test_above8mib_rejected_by_serializer_capture_and_child_boundary(monkeypatch):
    payload, receipt, parent, child, raw = captured_exchange(monkeypatch, 8*c.MiB, enforce_packet=False)
    assert len(raw) > c.MAX_RESPONSE_BYTES
    assert parent['response_raw_bytes'] == c.MAX_RESPONSE_BYTES
    assert parent['response_capture_overflow'] is True
    with pytest.raises(ValueError, match='8MiB'):
        worker.response_packet(dict(type='response', request_id=receipt['request_id'], result=receipt['result']))
    with pytest.raises(ValueError, match='capture overflow'):
        transport.validate_profiles(parent, child, receipt, payload, 'candidate')
    # Check the child-side gate independently; parent rejection must not hide a
    # stale smaller cap or an absent bound in research_boundaries.
    with pytest.raises(ValueError, match='serialized response size bound'):
        transport.research_boundaries(child, parent, parent['request_start_ns'],
                                      parent['receipt_arrival_ns'], startup=True)


def test_saved_original256_graph_and_move_cross_new_transport_and_control():
    folder = c.OLD_OUT/'workers'/'qualification_research'
    receipt = c.read(folder/'graph_parity_clock.json')
    rid = receipt['request_id']
    payload = c.read(folder/'startup_inputs'/f'{rid}.json')
    parent = c.read(folder/'parent_profiles'/f'{rid}.json')
    child = c.read(folder/f'{rid}_child_profile.json')
    original_parent = deepcopy(parent)
    # Explicit prospective transport-cap adaptation only. All original request,
    # result, raw hashes and clock markers stay byte-for-byte in memory.
    parent['response_capture_maximum_bytes'] = c.MAX_RESPONSE_BYTES
    assert transport.validate_profiles(parent, child, receipt, payload, 'control')['status'] == 'passed'
    assert worker.verify_graph_admission(folder, c.read(c.OLD_OUT/'freeze.json'), receipt['result'])['accepted'] is True
    assert c.read(folder/'parent_profiles'/f'{rid}.json') == original_parent

    source = c.ROOT/'outputs/splendorust_network_live_20261031_01/games/g000/moves/000001.json'
    saved = c.read(source)
    assert saved['variant'] == 'control' and saved['quota'] == 256
    parent = deepcopy(saved['parent_profile'])
    parent['response_capture_maximum_bytes'] = c.MAX_RESPONSE_BYTES
    payload = dict(player='search', observer=saved['observer'])
    proof = transport.validate_profiles(parent, saved['child_profile'], saved['clock'], payload, 'control')
    assert proof['status'] == 'passed'
    assert control.actual_after(saved['before'], saved['action']) == saved['after']
    assert c.read(source) == saved
