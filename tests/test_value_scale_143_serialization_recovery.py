import json
import runpy

import numpy as np
import pytest


def test_numpy_scalar_conversion_preserves_result_values_and_types():
    convert = runpy.run_path("scripts/analysis/recover_value_generalization_scale_143_serialization.py")["json_safe"]
    source = {"passed": np.bool_(True), "n": np.int64(512), "means": [np.float64(.12345678901234567)]}
    safe = convert(source)
    assert type(safe["passed"]) is bool and type(safe["n"]) is int
    assert safe["means"][0] == source["means"][0]
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe


def test_conversion_does_not_hide_nonfinite_results():
    convert = runpy.run_path("scripts/analysis/recover_value_generalization_scale_143_serialization.py")["json_safe"]
    with pytest.raises(ValueError):
        json.dumps(convert({"mse": np.float64(np.nan)}), allow_nan=False)
