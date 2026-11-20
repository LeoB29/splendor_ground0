"""Authentication of temporary synthetic saved-screen chains, without rollouts."""
from copy import deepcopy

import pytest

from tools.saved_adaptive_breadth_128 import common as c


def prior_chain(tmp_path, monkeypatch, alteration=None):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "PRIOR", tmp_path / "prior")
    monkeypatch.setattr(c, "CONFIG", {**c.CONFIG, "roots": 2})

    def write(name, value):
        path = tmp_path / name
        c.save(path, value)
        return c.reference(path)

    original_roots = [{**write(f"original/root_{i}.json", {"synthetic": i}), "index": i}
                      for i in range(2)]
    specs = [{"index": i, "pair": 100+i, "game_index": 100+i,
              "fit": 161001, "seat": i, "phase": 0} for i in range(2)]
    source = {"source_binding": c.SOURCE_BINDING,
              "inventory": {ref["path"]: ref["sha256"] for ref in original_roots},
              "roots": original_roots, "specs": specs, "root_count": 2, "cell_count": 2,
              "matrix_totals": {}, "inherited_coverage": {"states": 2, "replayed_continuations": 0},
              "inherited_full_inner_replay": False, "new_trajectory_replays": 0,
              "source_audit": {"synthetic": "audit"}, "source_result": {"synthetic": "result"}}
    monkeypatch.setattr(c, "collect_original_source", lambda: deepcopy(source))
    dummy_code = write("prior/code_identity.json", {"synthetic": "arithmetic module"})
    identity = {"files": {dummy_code["path"]: dummy_code["sha256"]}, "config": c.CONFIG.copy()}
    log_refs = [write(f"prior/{name}.json", {"synthetic": name}) for name in ("stdout", "stderr")]
    checks = write("prior/checks.json", {"passed": True, "returncode": 0, "identity": identity,
                                          "stdout": log_refs[0], "stderr": log_refs[1]})
    manifest = {"inputs": deepcopy(source), "identity": identity, "checks": checks}
    binding = c.digest(manifest)
    monkeypatch.setattr(c, "PRIOR_BINDING", binding)
    manifest_ref = write("prior/manifest.json", manifest)
    write("prior/freeze.json", {"owner_authorized": True, "binding": binding, "manifest": manifest_ref})
    rows, receipts = [], []
    for index, spec in enumerate(specs):
        root_ref = {key: original_roots[index][key] for key in ("path", "sha256")}
        row = {"manifest": binding, "source": root_ref,
               "result": {"spec": spec, "canonical_support": 8,
                          "partitions": [{"partition": j} for j in range(64)],
                          "mean_difference_fraction": [0, 1], "mean_score_difference": 0.0}}
        row_ref = write(f"prior/analysis_{index}.json", row)
        evidence = {"passed": True, "index": index, "partitions": 64,
                    "independent_selection_and_assessment_reconstruction": True,
                    "whole_world_vectors_preserved": True, "exact_rational_comparisons": True,
                    "equal_per_arm_selection_budget": True,
                    "mean_difference_fraction": [0, 1], "mean_score_difference": 0.0}
        envelope = {"manifest": binding, "source": root_ref, "analysis": row_ref, "receipt": evidence}
        if index == 0:
            if alteration == "root_link":
                envelope["source"] = {key: original_roots[1][key] for key in ("path", "sha256")}
            elif alteration == "receipt_index":
                evidence["index"] = 1
            elif alteration == "false_assurance":
                evidence["whole_world_vectors_preserved"] = False
            elif alteration == "different_mean":
                evidence["mean_difference_fraction"] = [1, 2]
        rows.append(row_ref)
        receipts.append(write(f"prior/verification_{index}.json", envelope))
    statistics = {"mean_difference_fraction": [0, 1], "ci95": [0.0, 0.0], "gate_pass": False}
    analysis = write("prior/analysis_complete.json", {"manifest": binding, "roots": rows,
                                                        "statistics": statistics, "verification_pending": True})
    result = write("prior/result.json", {"manifest": binding, "all_prescribed_checks_pass": True,
                                            "source_binding": c.SOURCE_BINDING,
                                            "analysis": analysis, "statistics": statistics})
    stats_audit = {"passed": True, "independent_bootstrap": True, "resampling_unit": "root", **statistics}
    if alteration == "false_statistics":
        stats_audit["passed"] = False
    elif alteration == "statistics_mismatch":
        stats_audit["ci95"] = [0.0, 0.1]
    audit = write("prior/final_audit.json", {"manifest": binding, "passed": True, "result": result,
                  "inherited_coverage": source["inherited_coverage"], "root_receipts": receipts,
                  "source_inventory_count": len(source["inventory"]), "independent_statistics": stats_audit})
    report = write("prior/report.json", {"synthetic": "report"})
    monkeypatch.setattr(c, "PRIOR_RESULT", result["sha256"])
    monkeypatch.setattr(c, "PRIOR_AUDIT", audit["sha256"])
    monkeypatch.setattr(c, "PRIOR_REPORT", report["sha256"])
    write("prior/final_status.json", {"complete": True, "all_prescribed_checks_pass": True,
                                        "manifest": binding, "result": result, "audit": audit, "report": report})
    return source, rows, receipts


def test_closed_prior_chain_binds_every_row_receipt_and_input(tmp_path, monkeypatch):
    source, rows, receipts = prior_chain(tmp_path, monkeypatch)
    inputs = c.collect_source()
    assert inputs["roots"] == source["roots"] and inputs["specs"] == source["specs"]
    assert inputs["prior_roots"] == rows and inputs["prior_receipts"] == receipts
    for ref in [*source["roots"], *rows, *receipts]:
        assert inputs["inventory"][ref["path"]] == ref["sha256"]
    assert inputs["prior_scores_reused"] is False
    c.check_inventory(inputs["inventory"])


@pytest.mark.parametrize("alteration", ["root_link", "receipt_index", "false_assurance",
                                        "different_mean", "false_statistics", "statistics_mismatch"])
def test_hash_valid_semantically_false_prior_closure_is_rejected(tmp_path, monkeypatch, alteration):
    prior_chain(tmp_path, monkeypatch, alteration)
    with pytest.raises(ValueError):
        c.collect_source()


def test_mutating_a_predecessor_row_cannot_be_hidden_by_a_passed_status(tmp_path, monkeypatch):
    _, rows, _ = prior_chain(tmp_path, monkeypatch)
    path = c.local(rows[0]["path"])
    path.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Changed artifact"):
        c.collect_source()
