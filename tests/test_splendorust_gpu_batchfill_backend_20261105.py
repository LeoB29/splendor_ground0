"""Model-free scheduler safety, fairness, routing and unchanged-math checks."""
from copy import deepcopy
import queue
import threading
import time

import pytest

from tools.splendorust_gpu_batchfill_20261105 import backend as b


def request(index, *, slot=None, deadline=None):
    return b._Request(f"c{index}", f"c{index}:1", index % 8 if slot is None else slot,
                      [float(index + 1)] * b.WIDTH, time.perf_counter(),
                      time.perf_counter() + 5 if deadline is None else deadline,
                      enqueue_sequence=index + 1)


def rows_from_tokens(tokens, batch):
    return [[tokens[i * b.WIDTH]] * 83 for i in range(batch)]


def wait_for(predicate, seconds=3):
    until = time.perf_counter() + seconds
    while time.perf_counter() < until:
        if predicate():
            return
        time.sleep(.002)
    assert predicate(), "bounded test synchronization did not complete"


def test_source_equivalence_covers_exact_model_and_control_scheduler():
    proof = b.source_equivalence()
    assert proof["models_loaded"] == 0 and proof["physical_batch"] == 8
    assert "complete_model_factory_excluding_scheduler_plumbing" in proof["unchanged_ast_sha256"]
    assert "InferenceService._dispatch" in proof["unchanged_ast_sha256"]
    assert "original8_scheduler" in proof["unchanged_ast_sha256"]


@pytest.mark.parametrize("count", [16, 32])
def test_fifo_matching_keeps_first_request_of_every_row_and_retains_skips(count):
    ready = [request(i) for i in range(count)]
    original = list(ready)
    selected, deferred = b.select_oldest_per_row(ready)
    assert selected == ready[:8] and deferred == ready[8:]
    assert ready == original
    served = selected[:]
    while deferred:
        selected, deferred = b.select_oldest_per_row(deferred)
        assert len({item.slot for item in selected}) == len(selected)
        served += selected
    assert served == ready


def test_old_skipped_row_is_not_pushed_behind_new_arrivals():
    first = [request(0), request(8), request(1)]
    selected, deferred = b.select_oldest_per_row(first)
    assert [item.client_id for item in selected] == ["c0", "c1"]
    assert [item.client_id for item in deferred] == ["c8"]
    selected, deferred = b.select_oldest_per_row(deferred + [request(16), request(2)])
    assert [item.client_id for item in selected] == ["c8", "c2"]
    assert [item.client_id for item in deferred] == ["c16"]


def test_all_same_row_makes_progress_one_per_dispatch():
    remaining = [request(i, slot=3) for i in range(32)]
    served = []
    while remaining:
        selected, remaining = b.select_oldest_per_row(remaining)
        assert len(selected) == 1
        served += selected
    assert [item.client_id for item in served] == [f"c{i}" for i in range(32)]


def test_successful_ticket_queue_puts_have_gapless_fifo_sequence():
    q = b._TicketQueue(2)
    first, second, third = request(100), request(101), request(102)
    q.put_nowait(first); q.put_nowait(second)
    with pytest.raises(queue.Full):
        q.put_nowait(third)
    assert (first.enqueue_sequence, second.enqueue_sequence, q.enqueue_sequence) == (1, 2, 2)
    assert q.get_nowait() is first
    q.put_nowait(third)
    assert third.enqueue_sequence == 3
    assert [q.get_nowait(), q.get_nowait()] == [second, third]


@pytest.mark.parametrize("route", ["cpu", "gpu"])
def test_original8_keeps_unique_registration_and_capacity(route):
    service = b.InferenceService(rows_from_tokens, route)
    try:
        service.register("first", 0)
        with pytest.raises(ValueError, match="unique slot"):
            service.register("duplicate", 0)
        for i in range(1, 8):
            service.register(f"c{i}", i)
        with pytest.raises(ValueError, match="capacity"):
            service.register("ninth", 0)
    finally:
        service.close()


def test_candidate_accepts32_logical_clients_with_four_per_row_but_not33():
    service = b.InferenceService(rows_from_tokens, "gpu", scheduler="row_fifo32")
    try:
        for i in range(32):
            service.register(f"c{i}", i % 8)
        with pytest.raises(ValueError, match="capacity"):
            service.register("c32", 0)
        with pytest.raises(ValueError):
            service.register("c0", 0)
        assert len(service.clients) == 32
    finally:
        service.close()


def test_candidate_not_admitted_as_cpu_or_with_gather_wait():
    with pytest.raises(ValueError, match="unchanged CPU"):
        b.InferenceService(rows_from_tokens, "cpu", scheduler="row_fifo32")
    with pytest.raises(ValueError, match="zero intentional gather wait"):
        b.InferenceService(rows_from_tokens, "gpu", scheduler="row_fifo32", max_wait_seconds=.001)


@pytest.mark.parametrize("count", [16, 32])
def test_actual_service_routes_colliding_clients_and_repeated_callbacks_exactly(count):
    entered, release = threading.Event(), threading.Event()
    evidence, outputs, errors = [], {}, []
    counter = [0]
    def forward(tokens, batch):
        counter[0] += 1
        if counter[0] == 1:
            entered.set()
            assert release.wait(3)
        assert batch == 8
        return rows_from_tokens(tokens, batch)
    service = b.InferenceService(forward, "gpu", scheduler="row_fifo32", evidence=evidence.append)
    threads = []
    def run(i):
        try:
            for repeat in range(2):
                value = float(i + repeat * 100 + 1)
                result, receipt = service.infer(f"c{i}", [value] * b.WIDTH,
                                                deadline=time.perf_counter() + 5)
                outputs[i, repeat] = (result, receipt)
        except BaseException as error:
            errors.append(error)
    try:
        for i in range(count):
            service.register(f"c{i}", i % 8)
        for i in range(count):
            thread = threading.Thread(target=run, args=(i,)); thread.start(); threads.append(thread)
        assert entered.wait(3)
        wait_for(lambda: len(service.pending) == count)
        release.set()
        for thread in threads:
            thread.join(5)
        assert not any(thread.is_alive() for thread in threads)
        assert not errors
        assert service.failure is None and not service.pending
        assert service.completed_requests == count * 2
        for (i, repeat), (result, receipt) in outputs.items():
            assert result == [[float(i + repeat * 100 + 1)] * 83]
            assert receipt["request_id"] == f"c{i}:{repeat + 1}"
            assert receipt["slot"] == i % 8
        schedule = [event for event in evidence if event["kind"] == "scheduler_dispatch"]
        batches = [event for event in evidence if event["kind"] == "inference_batch"]
        assert len(schedule) == len(batches)
        deferred = []; seen_dispatches = []
        for event, batch in zip(schedule, batches):
            ready = event["ready"]
            ids = [item["request_id"] for item in ready]
            assert ids[:len(deferred)] == deferred
            sequences = [item["enqueue_sequence"] for item in ready]
            assert sequences == sorted(sequences) and len(sequences) == len(set(sequences))
            first_by_row = {}
            for item in ready:
                first_by_row.setdefault(item["slot"], item["request_id"])
            assert event["selected_request_ids"] == list(first_by_row.values())
            deferred = [item for item in ids if item not in event["selected_request_ids"]]
            assert event["deferred_request_ids"] == deferred
            assert [item["request_id"] for item in batch["requests"]] == event["selected_request_ids"]
            assert len({item["slot"] for item in batch["requests"]}) == len(batch["requests"])
            seen_dispatches += event["selected_request_ids"]
        assert not deferred and len(seen_dispatches) == count * 2
        assert len(set(seen_dispatches)) == count * 2
    finally:
        release.set()
        service.close()
        for thread in threads:
            thread.join(5)


def test_original8_has_unchanged_dispatch_evidence_without_candidate_schedule_events():
    evidence = []
    service = b.InferenceService(rows_from_tokens, "gpu", evidence=evidence.append)
    try:
        service.register("a", 0)
        out, _ = service.infer("a", [3.] * b.WIDTH, deadline=time.perf_counter() + 2)
        assert out == [[3.] * 83]
        assert [event["kind"] for event in evidence] == ["inference_batch"]
    finally:
        service.close()


def test_second_pending_callback_and_slot_release_are_rejected():
    entered, release = threading.Event(), threading.Event()
    def forward(tokens, batch):
        entered.set(); assert release.wait(3)
        return rows_from_tokens(tokens, batch)
    service = b.InferenceService(forward, "gpu", scheduler="row_fifo32")
    service.register("a", 0); errors = []
    def run():
        try:
            service.infer("a", [1.] * b.WIDTH, deadline=time.perf_counter() + 5)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run); thread.start()
    try:
        assert entered.wait(3)
        with pytest.raises(ValueError, match="one outstanding"):
            service.infer("a", [2.] * b.WIDTH, deadline=time.perf_counter() + 5)
        with pytest.raises(ValueError, match="outstanding"):
            service.unregister("a")
    finally:
        release.set(); thread.join(5); service.close()
    assert not errors


def test_expired_queue_request_poison_is_preserved_and_no_retry_is_possible():
    entered, release = threading.Event(), threading.Event()
    def forward(tokens, batch):
        entered.set(); assert release.wait(3)
        return rows_from_tokens(tokens, batch)
    service = b.InferenceService(forward, "gpu", scheduler="row_fifo32")
    errors = []
    def run(client, delay):
        try:
            service.infer(client, [1.] * b.WIDTH, deadline=time.perf_counter() + delay)
        except BaseException as error:
            errors.append(error)
    first = threading.Thread(target=run, args=("a", 2))
    second = threading.Thread(target=run, args=("b", .05))
    try:
        service.register("a", 0); service.register("b", 0)
        first.start(); assert entered.wait(3); second.start(); second.join(2)
        assert not second.is_alive()
        assert isinstance(service.failure, TimeoutError)
        with pytest.raises(RuntimeError, match="no retry"):
            service.infer("b", [2.] * b.WIDTH, deadline=time.perf_counter() + 2)
    finally:
        release.set(); first.join(3); second.join(3); service.close()
    assert len(errors) == 2
    assert service.completed_requests == 0


def test_close_unblocks_pending_clients_and_joins_owned_service():
    entered, release = threading.Event(), threading.Event()
    def forward(tokens, batch):
        entered.set(); assert release.wait(3)
        return rows_from_tokens(tokens, batch)
    service = b.InferenceService(forward, "gpu", scheduler="row_fifo32")
    service.register("a", 0); errors = []
    def infer():
        try:
            service.infer("a", [1.] * b.WIDTH, deadline=time.perf_counter() + 5)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=infer); thread.start(); assert entered.wait(3)
    closer = threading.Thread(target=service.close); closer.start()
    try:
        thread.join(2)
        assert not thread.is_alive() and errors
        assert "Service closed with outstanding inference" in str(errors[0].__cause__)
    finally:
        release.set(); closer.join(5); thread.join(5)
    assert service.closed and not service.thread.is_alive()


def test_candidate_physical_qualification_keeps_full8_rows_and_excludes_clients():
    service = b.InferenceService(rows_from_tokens, "gpu", scheduler="row_fifo32")
    try:
        tokens = [float(row + 1) for row in range(8) for _ in range(b.WIDTH)]
        result = service.physical_forward(tokens, deadline=time.perf_counter() + 2)
        assert result == [[float(row + 1)] * 83 for row in range(8)]
        service.register("a", 0)
        with pytest.raises(ValueError, match="no clients"):
            service.physical_forward(tokens, deadline=time.perf_counter() + 2)
    finally:
        service.close()
