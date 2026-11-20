"""Model-free contracts for the bounded original-weight inference worker."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import types
import warnings

import pytest

from tools.splendorust_gpu_probe_20261105 import worker as w


def outputs():
    return [[0.0] * 81 + [.25, -.5]]


def test_numerical_gate_allows_roundoff_not_bit_identity():
    baseline = outputs(); candidate = copy.deepcopy(baseline)
    candidate[0][0] += 1e-5
    result = w.compare_outputs(baseline, candidate, [[0]])
    assert result["passed"] and not result["float32_bits_equal"]


def test_near_tie_flip_is_reported_but_not_rejected():
    baseline = outputs(); candidate = copy.deepcopy(baseline)
    baseline[0][0] = 1e-6; candidate[0][1] = 1e-6
    result = w.compare_outputs(baseline, candidate, [[0, 1]])
    assert result["passed"] and len(result["legal_argmax_disagreements"]) == 1
    assert not result["robust_legal_argmax_disagreements"]


def test_robust_argmax_flip_fails_even_under_looser_logit_tolerance():
    baseline = outputs(); candidate = copy.deepcopy(baseline)
    baseline[0][0] = .002; candidate[0][1] = .002
    tolerances = dict(w.NUMERICAL, atol=.01, policy_probability_atol=.01)
    result = w.compare_outputs(baseline, candidate, [[0, 1]], tolerances)
    assert not result["passed"] and result["max_tolerance_fraction"] < 1
    assert len(result["robust_legal_argmax_disagreements"]) == 1


def test_value_absolute_tolerance_has_independent_gate():
    baseline = outputs(); candidate = copy.deepcopy(baseline)
    candidate[0][82] += .00012
    result = w.compare_outputs(baseline, candidate, None)
    assert result["max_tolerance_fraction"] < 1 and not result["passed"]


def test_legal_softmax_gate_is_independent():
    baseline = outputs(); candidate = copy.deepcopy(baseline)
    candidate[0][0] += .00008
    result = w.compare_outputs(baseline, candidate, [[0, 1]])
    assert result["max_tolerance_fraction"] < 1 and not result["passed"]
    assert result["max_legal_probability_abs_error"] > 1e-5


def test_unmasked_controls_have_no_move_claim():
    result = w.compare_outputs(outputs(), outputs(), None)
    assert result["passed"] and result["legal_rows"] == 0
    assert result["max_legal_probability_abs_error"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_output_is_rejected(bad):
    candidate = outputs(); candidate[0][0] = bad
    with pytest.raises(ValueError, match="finite"):
        w.compare_outputs(outputs(), candidate, None)


@pytest.mark.parametrize("legal", [[[81]], [[]], [[0, 0]], [[True]], []])
def test_invalid_legal_columns_are_rejected(legal):
    with pytest.raises(ValueError):
        w.compare_outputs(outputs(), outputs(), legal)


class FakeProvider:
    def __init__(self, names):
        self.names = names; self.selected = None

    def is_available(self):
        return bool(self.names)

    def device_count(self):
        return len(self.names)

    def device_name(self, index):
        return self.names[index]

    def device(self, index):
        self.selected = index
        return "privateuseone:" + str(index)


def test_gpu_selection_never_assumes_adapter_zero():
    provider = FakeProvider(["Intel UHD", "AMD Radeon RX 7800 XT"])
    device, identity = w.select_device(provider, "AMD Radeon RX 7800 XT")
    assert device == "privateuseone:1" and provider.selected == 1
    assert identity["selected"]["index"] == 1


@pytest.mark.parametrize("names", [[], ["Intel UHD"], ["AMD Radeon RX 7800 XT"] * 2])
def test_missing_or_ambiguous_gpu_does_not_fallback(names):
    provider = FakeProvider(names)
    with pytest.raises(w.QualificationFailure):
        w.select_device(provider, "AMD Radeon RX 7800 XT")
    assert provider.selected is None


def test_backend_fallback_warning_is_retained_and_stops(tmp_path):
    journal = w.WarningJournal(tmp_path)
    def call():
        warnings.warn("Operator is not currently supported on the DML backend and will fall back to CPU")
        return 42
    with pytest.raises(w.QualificationFailure, match="fallback"):
        journal.call("directml_numeric_fixture", call)
    assert len(journal.records) == 1
    assert list(tmp_path.glob("warnings_*.json"))
    progress = [json.loads(line) for line in journal.progress.read_text().splitlines()]
    assert [row["state"] for row in progress] == ["started"]


def test_unsupported_backend_exception_is_negative_not_retry(tmp_path):
    journal = w.WarningJournal(tmp_path)
    def call():
        raise NotImplementedError("Operator unsupported on DML")
    with pytest.raises(w.QualificationFailure, match="unsupported"):
        journal.call("directml_numeric_fixture", call)
    assert "exception" in journal.progress.read_text()


def test_unrelated_exception_is_not_relabelled_qualification(tmp_path):
    journal = w.WarningJournal(tmp_path)
    with pytest.raises(RuntimeError, match="out of memory"):
        journal.call("directml_numeric_fixture", lambda: (_ for _ in ()).throw(RuntimeError("out of memory")))


def test_host_readback_is_blocking_and_complete():
    events = []
    class FakeTensor:
        dtype = "float32"
        shape = (2, 83)
        def detach(self):
            events.append("detach"); return self
        def to(self, device, non_blocking):
            assert device == "cpu" and non_blocking is False
            events.append("blocking_cpu_transfer"); return self
        def contiguous(self):
            events.append("contiguous"); return self
        def numpy(self):
            events.append("host_numpy_view"); return self
        def copy(self):
            events.append("complete_host_copy"); return "host_result"
    assert w.output_to_host(FakeTensor(), 2, types.SimpleNamespace(float32="float32")) == "host_result"
    assert events == ["detach", "blocking_cpu_transfer", "contiguous", "host_numpy_view", "complete_host_copy"]


def test_gpu_actual_device_is_checked_before_forward():
    gpu_device = types.SimpleNamespace(type="privateuseone", index=1)
    class Input:
        device = types.SimpleNamespace(type="cpu")
        dtype = "float32"
        def to(self, device, non_blocking):
            return self  # Simulates an accidental provider/adapter CPU route.
    with pytest.raises(w.QualificationFailure, match="input device"):
        w.gpu_roundtrip(lambda x: pytest.fail("must not infer on CPU"), Input(), 1,
                        gpu_device, types.SimpleNamespace(float32="float32"))


def test_timing_uses_complete_batch_latency():
    result = w.summarize_timings([.02] * 30, 8)
    assert result["positions_per_second"] == 400
    assert result["mean_seconds"] == result["p95_seconds"] == .02
    with pytest.raises(ValueError):
        w.summarize_timings([.02] * 29, 8)


def test_original_bindings_reject_wrong_bytes_before_loading(tmp_path):
    path = tmp_path / "model.bin"; path.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="identity"):
        w.verify_binding({"path": str(path), "bytes": path.stat().st_size, "sha256": "0" * 64})


def test_plan_prospectively_pins_implementation_and_tolerances():
    plan = {"batches": w.BATCHES, "numerical": w.NUMERICAL,
            "timing": w.TIMING,
            "gpu_implementation": w.GPU_IMPLEMENTATION, "cpu_primitive_bridge_check": True,
            "device_name_contains": "AMD Radeon RX 7800 XT", "cpu_affinity": [0]}
    w.validate_plan(plan)
    for key, changed in [("numerical", dict(w.NUMERICAL, atol=.1)),
                         ("gpu_implementation", "auto_retry"), ("cpu_affinity", [1]),
                         ("batches", [1, 2, 8]),
                         ("timing", dict(w.TIMING, repetitions=31)),
                         ("timing", dict(w.TIMING, order="CPU_then_GPU")),
                         ("timing", dict(w.TIMING, gpu_scope="async enqueue")),
                         ("timing", dict(w.TIMING, cpu_scope="full player")),
                         ("timing", dict(w.TIMING, input_list_conversion_excluded_equally=False))]:
        with pytest.raises(ValueError):
            w.validate_plan(dict(plan, **{key: changed}))


def test_worker_source_has_no_training_or_game_entrypoints():
    text = Path(w.__file__).read_text()
    for forbidden in (".backward(", "torch.optim", ".step(", "Opponent(", "subprocess."):
        assert forbidden not in text
