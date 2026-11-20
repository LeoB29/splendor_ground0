"""Small real-policy fits plus production-size, inference-free exposure checks."""
from collections import Counter
import copy
import time

import pytest
import torch

from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from tools.independent_game_coverage_128_recovery_01.test_learning import row
from . import learning as h


def fixture(tmp_path):
    original = [row(i, support=(2, 3, 4, 5, 6, 7, 8, 5)[i]) for i in range(8)]
    added = []
    for i in range(8):
        value = row(11950+i, "added", (3, 4, 6, 8)[i % 4])
        value.update(source_panel="fixture", source_game=i, source_fit=h.FITS[0], seat=i % 2, phase=1)
        value["cell"] = [value["source_panel"], value["source_fit"], value["seat"], value["phase"]]
        added.append(value)
    torch.manual_seed(719)
    model = LegalActionAttentionScorerModel(LegalActionAttentionScorerConfig(
        observation_size=6, action_feature_size=4, state_hidden_size=8, action_hidden_size=8,
        scorer_hidden_size=8, state_depth=1, action_depth=1, attention_layers=1,
        attention_heads=2, attention_feedforward_size=16))
    path = tmp_path/"initial.pt"
    h.save_model(path, model, 0, {"fixture": True})
    settings = {**copy.deepcopy(h.CONFIG), "updates": 16, "reference_updates": 4,
        "batch_size": 4, "train_rows": 8, "added_rows": 8,
        "sizes": {"small": 2, "medium": 4, "large": 8}, "diagnostic_updates": [0, 4, 8, 12, 16],
        "resume_checkpoint_interval": 3, "curve_old_rows": 3, "curve_added_rows_per_cell": 1}
    return {"rows": {"old": original, "added": added}, "initializer": h.c.artifact(path),
            "config": settings, "partitions": {"small": [0, 1], "medium": list(range(4)), "large": list(range(8))}}


def schedule_pools():
    original = [{"id": f"old:{i}", "features": torch.empty(1+i % 13, 0)} for i in range(9461)]
    added = []
    cells = [(fit, seat, phase) for fit in h.FITS for seat in (0, 1) for phase in range(3)]
    # Nested blocks preserve the original720 prefix and balanced18-cell counts.
    for copies in (40, 120, 480):
        for fit, seat, phase in cells:
            for _ in range(copies):
                i = len(added)
                added.append({"id": f"added:{i}", "source_panel": "fixture", "source_fit": fit,
                    "source_game": i, "seat": seat, "phase": phase, "features": torch.empty(1+i % 17, 0)})
    subsets = {name: list(range(count)) for name, count in h.CONFIG["sizes"].items()}
    return {"old": original, "added": added}, subsets


def test_production_exposures_and_all_three_independent_schedules():
    pools, subsets = schedule_pools()
    for fit in h.FITS:
        planned = h.batch_plan(pools, subsets, fit)
        independent = h.independent_plan(pools, subsets, fit)
        assert planned == independent
        paired = h._paired_receipt(planned, pools, h.CONFIG, fixture=False)
        for arm in h.ARMS:
            receipt = paired["arms"][arm]
            assert receipt["row_exposures"] == 456176
            assert receipt["old_exposures"] == receipt["added_exposures"] == 228088
            assert Counter(receipt["batch_lengths"]) == {64: 7088, 53: 48}
            assert len(receipt["per_added_game_exposures"]) == h.CONFIG["sizes"][arm]
            assert max(receipt["per_added_cell_exposures"].values())-min(receipt["per_added_cell_exposures"].values()) <= 1
        assert len({paired["arms"][arm]["batch_sequence_sha256"] for arm in h.ARMS}) == 3


def test_long_lengths_repeat_exact_predecessor_dose_four_times():
    pools, _ = schedule_pools()
    for fit in h.FITS:
        expected = h.prior._reference_lengths(pools["old"], fit, {**h.CONFIG, "updates": 1784})
        assert h._reference_lengths(pools["old"], fit, h.CONFIG) == expected*4


def test_independent_schedule_uses_no_collector_helpers(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    pools, subsets, settings = values["rows"], values["partitions"], values["config"]
    expected = h.batch_plan(pools, subsets, h.FITS[0], settings)
    for name in ("_reference_lengths", "_cycling_indices", "_cell", "batch_plan"):
        monkeypatch.setattr(h, name, lambda *a, **k: pytest.fail("Collector helper called"))
    assert h.independent_plan(pools, subsets, h.FITS[0], settings) == expected


@pytest.mark.parametrize("damage", ["duplicate", "unnested", "cell", "game", "validation", "assessment", "overlap", "temperature", "512", "curve"])
def test_training_population_firewall(tmp_path, damage):
    values = fixture(tmp_path)
    pools, subsets, settings = values["rows"], values["partitions"], values["config"]
    if damage == "duplicate":
        subsets["small"] = [0, 0]
    elif damage == "unnested":
        subsets["small"] = [6, 7]
    elif damage == "cell":
        pools["added"][0]["phase"] = 2
    elif damage == "game":
        pools["added"][1]["source_game"] = 0
    elif damage == "validation":
        pools["old"][0]["split"] = "validation"
    elif damage == "assessment":
        pools["added"][0]["assessment_outcomes"] = []
    elif damage == "overlap":
        pools["added"][0]["id"] = pools["old"][0]["id"]
    elif damage == "temperature":
        settings["target_temperature"] = .1
    elif damage == "512":
        pools["added"][0]["teacher_worlds"] = 512
    else:
        settings["diagnostic_updates"] = [0, 16]
    with pytest.raises(ValueError):
        h._inputs(h.FITS[0], "small", "fixture", tmp_path, pools, values["initializer"], settings, subsets)


def test_fixture_cannot_override_production(tmp_path):
    values = fixture(tmp_path)
    with pytest.raises(ValueError, match="Fixture cannot target production"):
        h._inputs(h.FITS[0], "small", "fixture", h.OUTPUT, values["rows"], values["initializer"],
                  values["config"], values["partitions"])


def test_short_supports_are_differentiable_zero_and_audit_matches():
    rows = [row(i, support=n) for i, n in enumerate((1, 2, 3, 4, 5, 8))]
    logits = torch.arange(6*11, dtype=torch.float32).reshape(6, 11).requires_grad_()
    losses, choices = h.objective(logits, rows)
    independent, independent_choices = h.audit_objective(logits, rows)
    assert torch.equal(losses, independent) and choices == independent_choices
    assert torch.equal(losses[:4], torch.zeros(4))
    losses.mean().backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[:4].count_nonzero()


def _aged_launch(tmp_path, hours):
    launch = tmp_path/"launch.json"
    h.old.save_new(launch, {"manifest": "fixture", "epoch_seconds": time.time()-hours*3600})
    h.old.save_new(tmp_path/"launch_seal.json", {"manifest": "fixture", "receipt": h.c.artifact(launch)})


def test_report_uses_current_budget_after_60_hours_without_ancestor_mutation(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    _aged_launch(tmp_path, 60)
    prior_config = copy.deepcopy(h.prior.c.CONFIG)
    precision_config = copy.deepcopy(h.checkpoint_helpers.c.CONFIG)
    prior_report = h.prior.training_report
    model, *_ = h._model(values["initializer"], "small", 42, values["config"])
    fresh = tmp_path/"fresh"
    fresh.mkdir()
    reference = prior_report(model, values["rows"]["old"], fresh, values["config"])
    normal = h.training_report(model, values["rows"]["old"], tmp_path, values["config"])
    audited = h.training_report(model, values["rows"]["old"], tmp_path, values["config"], audit=True)
    h._compare(normal, reference)
    h._compare(normal, audited)
    assert h.prior.c.CONFIG == prior_config and h.checkpoint_helpers.c.CONFIG == precision_config
    assert h.prior.training_report is prior_report
    monkeypatch.setattr(h.prior.c, "budget_boundary", lambda *a: pytest.fail("Inherited8h boundary reached"))
    h.training_report(model, values["rows"]["old"], tmp_path, values["config"], audit=True)


@pytest.mark.parametrize("mode", ["abort", "168h"])
def test_current_report_budget_still_enforces_abort_and_ceiling(tmp_path, mode):
    values = fixture(tmp_path)
    if mode == "abort":
        h.old.save_new(tmp_path/"abort.json", {"reason": "fixture"})
    else:
        _aged_launch(tmp_path, 169)
    model, *_ = h._model(values["initializer"], "large", 42, values["config"])
    for audit in (False, True):
        with pytest.raises(h.c.BudgetExceeded):
            h.training_report(model, values["rows"]["old"], tmp_path, values["config"], audit=audit)


def test_all_nine_real_fits_independently_reproduced_and_registry_sealed_at_60h(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    _aged_launch(tmp_path, 60)
    fits, audits = [], []
    monkeypatch.setattr(h.prior.c, "budget_boundary", lambda *a: pytest.fail("Inherited8h boundary reached"))
    monkeypatch.setattr(h.checkpoint_helpers.c, "budget_boundary", lambda *a: pytest.fail("Inherited48h boundary reached"))
    for fit_id in h.FITS:
        for arm in h.ARMS:
            ref = h.fit(fit_id, arm, "fixture", tmp_path, **values)
            fit = h.c.verified(ref)
            assert fit["selected_update"] == 16
            assert [point["update"] for point in fit["checkpoints"]] == [0, 4, 8, 12, 16]
            for point in fit["checkpoints"]:
                report = h.c.verified(point["training"])
                assert (report["full"] is not None) == (point["update"] in (0, 16))
                assert len(report["probe"]["old"]["rows"]) == 3
                assert len(report["probe"]["added"]["rows"]) == 2
            audit = h.audit_fit(fit_id, arm, ref, "fixture", tmp_path, **values)
            assert all(h.c.verified(audit)[key] for key in h.AUDIT_FLAGS)
            assert h.fit(fit_id, arm, "fixture", tmp_path, **values) == ref
            fits.append(ref)
            audits.append(audit)
    controls = {f"{arm}_{fit}": values["initializer"] for arm in ("anchor", "control") for fit in h.FITS}
    registry = h.c.verified(h.models_lock(fits, audits, "fixture", tmp_path, controls=controls))
    assert len(registry["models"]) == 15 and registry["all_new_fits_independently_reproduced"]
    with pytest.raises(ValueError, match="registry order"):
        h.models_lock(list(reversed(fits)), audits, "fixture", tmp_path/"bad", controls=controls)


def test_interrupted_fit_resumes_exact_trajectory_and_keeps_completed_points(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    original = h._save_progress
    def interrupt(folder, model, optimizer, update, norms, identity):
        original(folder, model, optimizer, update, norms, identity)
        if update == 8:
            raise h.c.BudgetExceeded("fixture interruption")
    monkeypatch.setattr(h, "_save_progress", interrupt)
    with pytest.raises(h.c.BudgetExceeded):
        h.fit(h.FITS[0], "large", "fixture", tmp_path, **values)
    folder = tmp_path/"fits"/f"fit_{h.FITS[0]}"/"large"
    point = h.c.artifact(folder/"point_0008.json")
    monkeypatch.setattr(h, "_save_progress", original)
    ref = h.fit(h.FITS[0], "large", "fixture", tmp_path, **values)
    assert h.c.artifact(folder/"point_0008.json") == point
    assert h.c.verified(h.audit_fit(h.FITS[0], "large", ref, "fixture", tmp_path, **values))["passed"]


def test_resume_rejects_changed_target_identity(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    original = h._save_progress
    def interrupt(*args):
        original(*args)
        raise h.c.BudgetExceeded("fixture interruption")
    monkeypatch.setattr(h, "_save_progress", interrupt)
    with pytest.raises(h.c.BudgetExceeded):
        h.fit(h.FITS[0], "small", "fixture", tmp_path, **values)
    values["rows"]["old"][0]["old_sums_wdl"][0] += 1
    monkeypatch.setattr(h, "_save_progress", original)
    with pytest.raises(ValueError):
        h.fit(h.FITS[0], "small", "fixture", tmp_path, **values)


def test_changed_midpoint_report_is_detected_even_with_updated_receipt_hash(tmp_path):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "medium", "fixture", tmp_path, **values)
    fitted = h.c.verified(ref)
    point = fitted["checkpoints"][2]
    report = h.c.verified(point["training"])
    report["probe"]["old"]["metrics"]["all_rows"]["first_winner_count"] += 1
    altered = tmp_path/"changed_report.json"
    h.old.save_new(altered, report)
    point["training"] = h.c.artifact(altered)
    receipt = tmp_path/"changed_receipt.json"
    h.old.save_new(receipt, {key: value for key, value in point.items() if key != "receipt"})
    point["receipt"] = h.c.artifact(receipt)
    result = tmp_path/"changed_result.json"
    h.old.save_new(result, fitted)
    with pytest.raises(ValueError):
        h.audit_fit(h.FITS[0], "medium", h.c.artifact(result), "fixture", tmp_path, **values)
