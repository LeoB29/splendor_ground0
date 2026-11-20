"""Independent, model-free concurrency contracts for the live GPU bridge."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
import time

import pytest

from tools.splendorust_gpu_live_20261105.backend import InferenceService


WIDTH = 31 * 48


def tokens(marker):
    return [marker] + [0.0] * (WIDTH - 1)


def marker_forward(flat, batch):
    return [[flat[i * WIDTH]] * 83 for i in range(batch)]


def test_gpu_keeps_registered_slots_and_zero_padding_under_concurrent_arrivals():
    captured, lock = [], Lock()

    def forward(flat, batch):
        with lock:
            captured.append((batch, [flat[i * WIDTH] for i in range(batch)]))
        return marker_forward(flat, batch)

    service = InferenceService(forward, "gpu")
    for slot in range(8):
        service.register(f"slot{slot}", slot)
    start = Barrier(8)

    def call(slot):
        start.wait(timeout=5)
        return service.infer(f"slot{slot}", tokens((slot + 1) / 8), deadline=time.perf_counter() + 5)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(call, range(8)))
        for slot, (output, receipt) in enumerate(results):
            assert output == [[(slot + 1) / 8] * 83]
        assert captured
        for batch, markers in captured:
            assert batch == 8
            assert len(markers) == 8
            assert all(value in (0.0, (slot + 1) / 8) for slot, value in enumerate(markers))
    finally:
        service.close()


def test_gpu_single_client_never_waits_for_inactive_slots():
    service = InferenceService(marker_forward, "gpu")
    service.register("only", 7)
    try:
        output, _ = service.infer("only", tokens(.75), deadline=time.perf_counter() + 2)
        assert output == [[.75] * 83]
    finally:
        service.close()


def test_already_waiting_requests_are_batched_even_after_gather_window_expired():
    entered, release = Event(), Event()
    calls = []

    def forward(flat, batch):
        calls.append([flat[i * WIDTH] for i in range(batch)])
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=5)
        return marker_forward(flat, batch)

    service = InferenceService(forward, "gpu")
    for slot in range(8):
        service.register(f"slot{slot}", slot)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            first = pool.submit(service.infer, "slot0", tokens(.125), deadline=time.perf_counter() + 5)
            assert entered.wait(timeout=2)
            pending = [pool.submit(service.infer, f"slot{slot}", tokens((slot + 1) / 8),
                                   deadline=time.perf_counter() + 5) for slot in range(1, 8)]
            until = time.perf_counter() + 2
            while service.queue.qsize() < 7 and time.perf_counter() < until:
                time.sleep(.001)
            assert service.queue.qsize() == 7
            time.sleep(.01)  # All seven have already exceeded the 1ms gather window.
            release.set()
            assert first.result(timeout=5)[0] == [[.125] * 83]
            for slot, future in enumerate(pending, start=1):
                assert future.result(timeout=5)[0] == [[(slot + 1) / 8] * 83]
        assert len(calls) == 2
        assert calls[1] == [0.0] + [(slot + 1) / 8 for slot in range(1, 8)]
    finally:
        release.set()
        service.close()


def test_cpu_forward_calls_really_overlap_instead_of_sharing_a_serial_queue():
    rendezvous = Barrier(2)
    observed, lock = [], Lock()

    def forward(flat, batch):
        with lock:
            observed.append(batch)
        rendezvous.wait(timeout=3)
        return marker_forward(flat, batch)

    service = InferenceService(forward, "cpu")
    service.register("a", 0)
    service.register("b", 1)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(service.infer, "a", tokens(.25), deadline=time.perf_counter() + 5)
            b = pool.submit(service.infer, "b", tokens(.5), deadline=time.perf_counter() + 5)
            assert a.result(timeout=6)[0] == [[.25] * 83]
            assert b.result(timeout=6)[0] == [[.5] * 83]
        assert observed == [1, 1]
    finally:
        service.close()


def test_a_second_outstanding_request_cannot_reuse_a_slot():
    entered, release = Event(), Event()

    def forward(flat, batch):
        entered.set()
        assert release.wait(timeout=5)
        return marker_forward(flat, batch)

    service = InferenceService(forward, "gpu")
    service.register("a", 0)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(service.infer, "a", tokens(.25), deadline=time.perf_counter() + 5)
            assert entered.wait(timeout=2)
            try:
                with pytest.raises(Exception, match="outstanding|pending|slot|client|busy"):
                    service.infer("a", tokens(.5), deadline=time.perf_counter() + 1)
            finally:
                release.set()
            assert first.result(timeout=5)[0] == [[.25] * 83]
    finally:
        release.set()
        service.close()


@pytest.mark.parametrize("bad", ["short_rows", "short_columns", "nonfinite"])
def test_malformed_physical_output_never_reaches_a_client(bad):
    def forward(flat, batch):
        result = marker_forward(flat, batch)
        if bad == "short_rows":
            result.pop()
        elif bad == "short_columns":
            result[0].pop()
        else:
            result[0][82] = float("nan")
        return result

    service = InferenceService(forward, "gpu")
    service.register("a", 0)
    try:
        with pytest.raises(Exception):
            service.infer("a", tokens(.25), deadline=time.perf_counter() + 2)
    finally:
        service.close()


def test_backend_failure_wakes_all_clients_and_prevents_later_inference():
    def forward(flat, batch):
        raise RuntimeError("independent injected device failure")

    service = InferenceService(forward, "gpu")
    for slot in range(8):
        service.register(f"slot{slot}", slot)
    start = Barrier(8)

    def call(slot):
        start.wait(timeout=5)
        with pytest.raises(Exception):
            service.infer(f"slot{slot}", tokens(.25), deadline=time.perf_counter() + 3)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(call, range(8)))
        with pytest.raises(Exception):
            service.infer("slot0", tokens(.25), deadline=time.perf_counter() + 1)
    finally:
        service.close()


def test_expired_request_cannot_execute_a_forward():
    calls = []

    def forward(flat, batch):
        calls.append(True)
        return marker_forward(flat, batch)

    service = InferenceService(forward, "gpu")
    service.register("a", 0)
    try:
        with pytest.raises(Exception):
            service.infer("a", tokens(.25), deadline=time.perf_counter() - 1)
        assert calls == []
    finally:
        service.close()


def test_late_forward_completion_is_rejected_without_resetting_the_deadline():
    def forward(flat, batch):
        time.sleep(.08)
        return marker_forward(flat, batch)

    service = InferenceService(forward, "gpu")
    service.register("a", 0)
    try:
        with pytest.raises(Exception):
            service.infer("a", tokens(.25), deadline=time.perf_counter() + .03)
    finally:
        service.close()


@pytest.mark.parametrize("batch", [0, 2, 8, True])
def test_only_original_batch_one_callbacks_are_accepted(batch):
    service = InferenceService(marker_forward, "gpu")
    service.register("a", 0)
    try:
        with pytest.raises(Exception):
            service.infer("a", tokens(.25), batch=batch, deadline=time.perf_counter() + 2)
    finally:
        service.close()


def test_closed_service_does_not_accept_more_work():
    service = InferenceService(marker_forward, "gpu")
    service.register("a", 0)
    service.close()
    with pytest.raises(Exception):
        service.infer("a", tokens(.25), deadline=time.perf_counter() + 1)


def _saved_stability_inputs():
    from tools.splendorust_gpu_live_20261105 import common as c, stability
    panel_path = c.PRIOR / "panel.json"
    panel = c.read(panel_path)
    roots = stability.roots_from_panel(panel)
    plan = {"prior_panel": c.artifact(panel_path),
            "numerical": c.read(c.PRIOR / "plan.json")["numerical"]}
    expected = {c.digest(root["tokens"]): root["historical_output83"] for root in roots}
    return plan, expected


def test_first_cpu_mismatch_is_saved_before_stability_gate_rejects(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.splendorust_gpu_live_20261105 import common as c, stability
    plan, expected = _saved_stability_inputs()
    monkeypatch.setattr(c, "OUT", tmp_path)
    folder = tmp_path / "cpu_mismatch"

    def wrong_cpu(flat, batch):
        rows = [list(expected[c.digest(flat)])]
        rows[0][0] += .25
        return rows

    with pytest.raises(ValueError, match="historical exact"):
        stability.qualify(plan, SimpleNamespace(infer_raw=wrong_cpu), None, folder, lambda: None)
    retained = c.read(folder / "cpu_root_00.json")
    assert retained["exact"] is False
    assert retained["output"] != retained["expected"]


def test_gpu_slot_drift_is_saved_and_rejected_even_inside_numeric_tolerance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.splendorust_gpu_live_20261105 import common as c, stability
    plan, expected = _saved_stability_inputs()
    monkeypatch.setattr(c, "OUT", tmp_path)
    folder = tmp_path / "gpu_slot_drift"

    def cpu(flat, batch):
        return [list(expected[c.digest(flat)])]

    def gpu(flat):
        rows = [list(expected.get(c.digest(flat[i * WIDTH:(i + 1) * WIDTH]), [0.] * 83))
                for i in range(8)]
        rows[7][81] += 5e-6
        return rows

    with pytest.raises(ValueError, match="packing stability"):
        stability.qualify(plan, SimpleNamespace(infer_raw=cpu), SimpleNamespace(fixed_batch_infer=gpu),
                          folder, lambda: None)
    retained = c.read(folder / "gpu_root_00.json")
    last = retained["cases"][-1]
    assert retained["status"] == "failed"
    assert last["slot"] == 7
    assert last["comparison"]["passed"] is True
    assert last["exact_within_gpu"] is False
