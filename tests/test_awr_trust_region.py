from __future__ import annotations

import torch

from splendor_ai.training.run_awr_trust_region import interpolate_state_dict


def test_parameter_interpolation_preserves_endpoints_and_integer_buffers() -> None:
    parent = {
        "weight": torch.tensor([0.0, 2.0]),
        "counter": torch.tensor([3], dtype=torch.long),
    }
    proposal = {
        "weight": torch.tensor([4.0, 6.0]),
        "counter": torch.tensor([9], dtype=torch.long),
    }

    midpoint = interpolate_state_dict(parent, proposal, 0.25)

    torch.testing.assert_close(midpoint["weight"], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(midpoint["counter"], parent["counter"])
    torch.testing.assert_close(interpolate_state_dict(parent, proposal, 0.0)["weight"], parent["weight"])
    torch.testing.assert_close(interpolate_state_dict(parent, proposal, 1.0)["weight"], proposal["weight"])


def test_parameter_interpolation_rejects_mismatched_state() -> None:
    try:
        interpolate_state_dict({"a": torch.zeros(1)}, {"b": torch.zeros(1)}, 0.5)
    except ValueError as error:
        assert "identical keys" in str(error)
    else:
        raise AssertionError("Expected mismatched state dictionaries to fail.")

