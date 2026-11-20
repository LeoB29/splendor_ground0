"""Model-free contracts for the saved GPU feasibility input route."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.splendorust_gpu_probe_20261105 import panel as p


@pytest.fixture(scope="module")
def saved_panel():
    return p.prepare_panel()


def test_actual_saved_counterexamples_and_native_callback_bindings(saved_panel):
    assert saved_panel["distinct_input_rows"] == 35
    assert saved_panel["saved_root_count"] == 36
    assert len(saved_panel["fixtures"]) == 51
    checks = saved_panel["corrected_input_checks"]
    assert checks["status"] == "actual_loader_counterexamples_passed"
    assert checks["six_corrected_faces"] == ["T1_05", "T1_07", "T2_10", "T2_12", "T2_13", "T2_14"]
    assert len(checks["rows"]) == 16
    assert len(checks["actual_loader_rejections"]) == 13
    assert all(x["rejected"] for x in checks["actual_loader_rejections"])
    assert saved_panel["checkpoint"]["sha256"] == p.CHECKPOINT_SHA256
    assert saved_panel["new_games"] == saved_panel["training_updates"] == saved_panel["model_calls_in_preparation"] == 0


def test_timing_batch32_distinct_and_batch64_cycling_disclosed(saved_panel):
    fixtures = {x["fixture_id"]: x for x in saved_panel["fixtures"]}
    for batch in p.BATCHES:
        fixture = fixtures[saved_panel["timing_fixtures"][str(batch)]]
        assert fixture["timing_fixture"] is True
        assert fixture["unique_input_rows"] == min(batch, 35)
        assert fixture["repeated_input_rows"] == max(0, batch-35)
        assert fixture["legal_policy_indices"] is not None
        assert fixture["purpose"] == "fixed_saved_input_timing"
    assert fixtures["timing_saved_cycle_batch64"]["root_ids"][:35] == fixtures["timing_saved_cycle_batch64"]["root_ids"][35:] + fixtures["timing_saved_cycle_batch64"]["root_ids"][29:35]


def test_historical_cpu_cases_preserve_actual_batch_composition(saved_panel):
    old = [x for x in saved_panel["fixtures"] if x["purpose"] == "historical_cpu_exact_graph_parity"]
    assert sorted(x["batch"] for x in old) == [1]*7 + [2]*2 + [8]*2
    assert all("historical_expected_cpu_outputs" in x for x in old)
    assert sum(x["legal_policy_indices"] is None for x in old) == 3
    assert all(not x["timing_fixture"] for x in old)


def test_validation_repeats_actual_corrected_pipeline(saved_panel):
    receipt = p.validate_panel(saved_panel)
    assert receipt["status"] == "actual_saved_input_pipeline_qualified"
    assert receipt["historical_cpu_cases"] == 46
    assert receipt["model_calls"] == 0


@pytest.mark.parametrize("change", ["token", "mask", "expected", "snapshot", "source_hash"])
def test_changed_evidence_rejected_by_actual_reconstruction(saved_panel, change):
    bad = deepcopy(saved_panel)
    if change == "token":
        bad["fixtures"][0]["tokens"][0] += .25
    elif change == "mask":
        bad["fixtures"][0]["legal_policy_indices"][0] = [80]
    elif change == "expected":
        bad["fixtures"][0]["historical_expected_cpu_outputs"][0][0] += .125
    elif change == "snapshot":
        bad["roots"][0]["snapshot"]["ruleset_id"] = "historical"
    else:
        bad["source_inputs"][0]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        p.validate_panel(bad)


def test_duplicate_disclosure_and_zero_mask_scope_rejected(saved_panel):
    bad = deepcopy(saved_panel)
    bad["fixtures"][-1]["repeated_input_rows"] = 0
    with pytest.raises(ValueError, match="duplicate disclosure"):
        p.validate_structure(bad)
    bad = deepcopy(saved_panel)
    bad["fixtures"][0]["legal_policy_indices"] = None
    with pytest.raises(ValueError, match="only zero"):
        p.validate_structure(bad)


def test_source_reader_rejects_hash_and_path_traversal(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}", encoding="utf-8")
    source = p.Sources(tmp_path)
    with pytest.raises(ValueError, match="fixed source identity"):
        source.read("source.json", expected_hash="0"*64)
    with pytest.raises(ValueError, match="relative source"):
        source.read("../source.json")


def test_preparation_never_imports_torch_or_directml():
    code = "from tools.splendorust_gpu_probe_20261105.panel import prepare_panel; prepare_panel(); import sys; assert 'torch' not in sys.modules; assert 'torch_directml' not in sys.modules"
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=p.ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_exclusive_output_preserves_existing_file(tmp_path):
    destination = tmp_path / "panel.json"
    destination.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        p.prepare_panel(destination)
    assert destination.read_text(encoding="utf-8") == "preserved"
