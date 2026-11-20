"""Independent model-free regression checks for GPU feasibility acceptance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_probe_20261105 import worker


def outputs():
    return [[0.0] * 83]


def test_numerical_acceptance_does_not_require_identical_float_bits():
    reference, changed = outputs(), outputs()
    changed[0][0] = 1e-6
    result = worker.compare_outputs(reference, changed, [[0, 1]])
    assert result["passed"] is True
    assert result["float32_bits_equal"] is False


def test_near_tie_flip_is_disclosed_without_failing_numerical_qualification():
    reference, changed = outputs(), outputs()
    reference[0][0] = 1e-6
    changed[0][1] = 1e-6
    result = worker.compare_outputs(reference, changed, [[0, 1]])
    assert result["passed"] is True
    assert len(result["legal_argmax_disagreements"]) == 1
    assert result["robust_legal_argmax_disagreements"] == []


def test_robust_argmax_gate_operates_separately_from_numeric_thresholds():
    reference, changed = outputs(), outputs()
    reference[0][0] = .002
    changed[0][1] = .003
    relaxed = dict(worker.NUMERICAL, atol=.01, rtol=.01, policy_probability_atol=.01)
    result = worker.compare_outputs(reference, changed, [[0, 1]], relaxed)
    assert result["max_tolerance_fraction"] < 1
    assert result["max_legal_probability_abs_error"] < relaxed["policy_probability_atol"]
    assert len(result["robust_legal_argmax_disagreements"]) == 1
    assert result["passed"] is False


def test_absolute_value_limit_is_not_diluted_by_relative_tolerance():
    reference, changed = outputs(), outputs()
    reference[0][81] = .8
    changed[0][81] = .80015
    result = worker.compare_outputs(reference, changed, [[0, 1]])
    assert result["max_tolerance_fraction"] < 1
    assert result["max_value_abs_error"] > worker.NUMERICAL["value_abs_atol"]
    assert result["passed"] is False


def test_legal_probability_limit_is_independent_of_logit_limit():
    reference, changed = outputs(), outputs()
    changed[0][0], changed[0][1] = .00009, -.00009
    result = worker.compare_outputs(reference, changed, [[0, 1]])
    assert result["max_tolerance_fraction"] < 1
    assert result["max_legal_probability_abs_error"] > worker.NUMERICAL["policy_probability_atol"]
    assert result["passed"] is False


def test_illegal_largest_logit_cannot_change_reported_legal_argmax():
    reference, changed = outputs(), outputs()
    reference[0][80] = changed[0][80] = 1000.0
    reference[0][0] = changed[0][0] = 2.0
    reference[0][1] = changed[0][1] = 1.0
    result = worker.compare_outputs(reference, changed, [[0, 1]])
    assert result["passed"] is True
    assert result["legal_argmax_disagreements"] == []


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_output_is_rejected_before_statistics(invalid):
    changed = outputs()
    changed[0][82] = invalid
    with pytest.raises(ValueError, match="finite"):
        worker.compare_outputs(outputs(), changed, [[0, 1]])


@pytest.mark.parametrize("mask", [[], [0, 0], [-1], [81], [True]])
def test_invalid_legal_support_is_rejected(mask):
    with pytest.raises(ValueError, match="policy columns"):
        worker.compare_outputs(outputs(), outputs(), [mask])


def test_full_host_read_is_blocking_and_copies_all_output():
    events = []

    class HostArray:
        def copy(self):
            events.append("copy_all")
            return "full_host_output"

    class Output:
        dtype = "float32"
        shape = (2, 83)

        def detach(self):
            events.append("detach")
            return self

        def to(self, device, *, non_blocking):
            events.append(("to", device, non_blocking))
            return self

        def contiguous(self):
            events.append("contiguous")
            return self

        def numpy(self):
            events.append("host_array")
            return HostArray()

    assert worker.output_to_host(Output(), 2, SimpleNamespace(float32="float32")) == "full_host_output"
    assert events == ["detach", ("to", "cpu", False), "contiguous", "host_array", "copy_all"]


@pytest.mark.parametrize("count", [0, 29, 31])
def test_timing_rejects_missing_or_extra_samples(count):
    with pytest.raises(ValueError, match="thirty"):
        worker.summarize_timings([.1] * count, 8)


def test_timing_throughput_uses_whole_batch_and_complete_elapsed_time():
    result = worker.summarize_timings([.25] * 30, 8)
    assert result["positions_per_second"] == 32
    assert result["median_seconds"] == result["p95_seconds"] == .25


@pytest.mark.parametrize("message", ["Operator will fall back to run on CPU", "unsupported operator", "kernel not currently supported"])
def test_backend_fallback_warning_fails_closed_and_is_retained(tmp_path, message):
    import warnings
    journal = worker.WarningJournal(tmp_path)
    with pytest.raises(worker.QualificationFailure):
        journal.call("directml_test", lambda: warnings.warn(message, UserWarning))
    assert journal.records[0]["message"] == message
    assert list(tmp_path.glob("warnings_*.json"))


def test_adapter_selection_rejects_ambiguous_hardware_names():
    provider = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                               device_name=lambda index: "AMD Radeon RX 7800 XT")
    with pytest.raises(ValueError, match="exactly one"):
        worker.select_device(provider, "AMD Radeon RX 7800 XT")


def test_fixed_plan_rejects_numerical_relaxation():
    plan = dict(batches=worker.BATCHES, numerical=deepcopy(worker.NUMERICAL),
                timing=deepcopy(worker.TIMING),
                gpu_implementation=worker.GPU_IMPLEMENTATION, cpu_primitive_bridge_check=True,
                device_name_contains="AMD Radeon RX 7800 XT", cpu_affinity=[0])
    worker.validate_plan(plan)
    plan["numerical"]["atol"] *= 2
    with pytest.raises(ValueError, match="tolerances"):
        worker.validate_plan(plan)
