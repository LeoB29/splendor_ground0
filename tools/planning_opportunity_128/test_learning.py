"""Bounded weighted reproduction, control matching and frozen-value tests."""
import copy
import json
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from tools.independent_game_coverage_128_recovery_01.test_learning import row, schedule_pools
from . import learning as h


def fixture(tmp_path, fit=None, updates=4):
    fit = h.FITS[0] if fit is None else fit
    pools = {"old": [row(i, support=n) for i, n in enumerate((3, 4, 5, 8, 6))], "added": []}
    for i in range(8):
        item = row(11950+i, "added", (3, 4, 5, 8)[i % 4])
        item.update(source_panel=f"panel{i//4}", source_game=i, environment_seed=97000+i,
                    source_fit=h.FITS[i//4], seat=0, phase=1)
        item["cell"] = [item["source_panel"], item["source_fit"], 0, 1]
        pools["added"].append(item)
    torch.manual_seed(719)
    model_config = LegalActionAttentionScorerConfig(observation_size=6, action_feature_size=4,
        state_hidden_size=8, action_hidden_size=8, scorer_hidden_size=8, state_depth=1, action_depth=1,
        attention_layers=1, attention_heads=2, attention_feedforward_size=16)
    model = LegalActionAttentionScorerModel(model_config)
    path = tmp_path/"initial.pt"
    h.save_model(path, model, 0, {"fixture": True})
    initial = h.c.artifact(path, source_fit=fit)
    config = {**copy.deepcopy(h.CONFIG), "updates": updates, "batch_size": 4, "train_rows": 5,
        "added_rows": 8, "narrow_rows_per_fit": 2}
    values = {"rows": pools, "initializer": initial, "config": config, "narrow": [0, 4]}
    values["control"] = control(tmp_path/"prior", fit, values)
    return values


def control(base, fit, values):
    options = {key: values[key] for key in ("rows", "initializer", "config", "narrow")}
    ref = h.prior.fit(fit, "prior-fixture", base, **options)
    audit = h.prior.audit_fit(fit, ref, "prior-fixture", base, **options)
    return {"result": ref, "audit": audit}


def test_exact_all_three_production_broad_schedules_without_training():
    pools = schedule_pools()
    for j, fit in enumerate(h.FITS):
        narrow = [i for i in range(720) if 5*j <= i % 20 < 5*(j+1)]
        expected = h.prior.batch_plan(pools, narrow, fit, h.prior.CONFIG)["broad"]
        assert h.batch_plan(pools, narrow, fit) == expected == h.independent_plan(pools, narrow, fit)
        receipt = h.prior._plan_receipt(expected, pools)
        assert receipt["row_exposures"] == 114044
        assert receipt["old_exposures"] == receipt["added_exposures"] == 57022
        assert receipt["updates"] == 1784 and Counter(receipt["batch_lengths"]) == {64: 1772, 53: 12}
        assert len(receipt["per_added_game_exposures"]) == 720


def test_weighted_actual_fit_full_audit_and_frozen_value_head(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "new-fixture", tmp_path/"head", **values)
    result = h.c.verified(ref)
    assert ref["index"] == 0 and result["arm"] == "weighted"
    assert result["row_exposures"] == 10 and result["old_exposures"] == result["added_exposures"] == 5
    assert [r["update"] for r in result["checkpoints"]] == [0, 4]
    initial = torch.load(h.c.verify_hash(values["initializer"]), weights_only=True)["model_state_dict"]
    final = torch.load(h.c.verify_hash(result["selected_model"]), weights_only=True)["model_state_dict"]
    assert all(torch.equal(initial[k], final[k]) for k in initial if k.startswith("value_head."))
    assert any(not torch.equal(initial[k], final[k]) for k in initial if not k.startswith("value_head."))
    assert any(not torch.equal(initial[k], final[k]) for k in initial if k.startswith("state_encoder."))
    report = h.c.verified(result["selected_training"])
    assert report["added"]["metrics_by_slice"]["optimizer_exposed"]["metrics"]["all_rows"]["rows"] == 8
    assert report["added"]["metrics_by_slice"]["not_optimizer_exposed"]["metrics"]["all_rows"]["rows"] == 0
    assert not report["assessment_read"] and not report["validation_read"]
    monkeypatch.setattr(h, "fit", lambda *a, **k: pytest.fail("Collector refit used"))
    monkeypatch.setattr(h, "batch_plan", lambda *a, **k: pytest.fail("Collector batch planner used"))
    monkeypatch.setattr(h, "objective", lambda *a, **k: pytest.fail("Collector objective used"))
    monkeypatch.setattr(h, "weight_plan", lambda *a, **k: pytest.fail("Collector weighting used"))
    monkeypatch.setattr(h, "opportunity_units", lambda *a, **k: pytest.fail("Collector opportunity used"))
    audit = h.c.verified(h.audit_fit(h.FITS[0], ref, "new-fixture", tmp_path/"head", **values))
    assert all(audit[key] for key in h.AUDIT_FLAGS)
    assert audit["matched_broad_control"] == result["matched_broad_control"]
    assert audit["selected_model"] == result["selected_model"]


@pytest.mark.parametrize("damage", ["recipe", "stream", "corpus", "partition", "initializer", "control_audit"])
def test_changed_matched_control_rejected_before_fit(tmp_path, damage):
    values = fixture(tmp_path)
    if damage == "recipe":
        values["config"]["lr"] *= 2
    elif damage == "stream":
        values["config"]["train_seeds"][str(h.FITS[0])] += 1
    elif damage == "corpus":
        values["rows"]["old"][0]["observation"][0] += 1
    elif damage == "partition":
        values["narrow"] = [1, 5]
    elif damage == "initializer":
        values["initializer"] = {**values["initializer"], "source_fit": h.FITS[1]}
    else:
        checked = h.c.verified(values["control"]["audit"])
        checked["arms"]["broad"]["complete_fit_bitwise_identical"] = False
        path = tmp_path/"bad_audit.json"
        h.old.save_new(path, checked)
        values["control"]["audit"] = h.c.artifact(path)
    with pytest.raises(ValueError):
        h.fit(h.FITS[0], "bad", tmp_path/"head", **values)
    assert not (tmp_path/"head/fits").exists()


@pytest.mark.parametrize("damage", ["frozen", "head", "metric", "membership", "selected", "gradient", "control_receipt", "normalizer", "raw_opportunity", "concentration", "marginal_metric"])
def test_independent_audit_rejects_corruption(tmp_path, damage):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "bad", tmp_path/"head", **values)
    result = h.c.verified(ref)
    if damage in ("frozen", "head"):
        point = result["checkpoints"][-1]
        payload = torch.load(h.c.verify_hash(point["checkpoint"]), weights_only=True)
        key = next(k for k in payload["model_state_dict"] if not k.startswith("value_head.") == (damage == "head"))
        payload["model_state_dict"][key].flatten()[0] += 1
        path = tmp_path/"corrupt.pt"
        torch.save(payload, path)
        point["checkpoint"] = h.c.artifact(path, source_fit=h.FITS[0], arm="weighted")
    elif damage == "metric":
        report = h.c.verified(result["selected_training"])
        report["added"]["metrics"]["all_rows"]["mean_target_kl"] += .1
        path = tmp_path/"corrupt_report.json"
        h.old.save_new(path, report)
        result["checkpoints"][-1]["training"] = h.c.artifact(path)
    elif damage == "membership":
        result["trainable_parameter_names"].append("value_head.0.weight")
    elif damage == "selected":
        result["selected_model"] = result["checkpoints"][0]["checkpoint"]
    elif damage == "gradient":
        result["optimization_stability"]["preclip_gradient_norms"][0] += .1
    elif damage == "control_receipt":
        result["matched_broad_control"]["batch_receipt"]["batch_sequence_sha256"] = "wrong"
    elif damage == "normalizer":
        result["opportunity_weighting"]["pools"]["old"]["normalizer_opportunity_units_numerator"] += 1
    elif damage == "raw_opportunity":
        result["opportunity_weighting"]["pools"]["added"]["per_row"][0]["raw_opportunity_units"] += 1
    elif damage == "concentration":
        result["opportunity_weighting"]["pools"]["old"]["game_or_pair_effective_sample_size"] += 1
    else:
        report = h.c.verified(result["selected_training"])
        report["added"]["marginal_shortlist"]["regret_units_sum"] += 1
        path = tmp_path/"corrupt_marginal.json"
        h.old.save_new(path, report)
        result["checkpoints"][-1]["training"] = h.c.artifact(path)
    path = tmp_path/"corrupt_result.json"
    h.old.save_new(path, result)
    with pytest.raises(ValueError):
        h.audit_fit(h.FITS[0], h.c.artifact(path), "bad", tmp_path/"head", **values)


def test_fixture_cannot_enter_production_and_defaults_are_dynamic(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    with pytest.raises(ValueError, match="Fixture cannot target production"):
        h._inputs(h.FITS[0], "bad", h.OUTPUT, **values)
    target = tmp_path/"dynamic"
    monkeypatch.setattr(h, "OUTPUT", target)
    assert h._base(None) == target
    assert not target.exists()


def test_training_entry_loads_only_two_training_pools(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    calls = []
    fake = SimpleNamespace(load_pools=lambda binding, base: calls.append("TRAIN") or values["rows"],
        initializer=lambda fit, base: values["initializer"], narrow_indices=lambda fit, binding, base: values["narrow"],
        broad_fit=lambda fit, binding, base: values["control"]["result"],
        broad_audit=lambda fit, binding, base: values["control"]["audit"])
    monkeypatch.setattr(sys.modules[h.__package__], "source", fake, raising=False)
    # Production counts reject this fixture before any fit or validation access.
    with pytest.raises(ValueError, match="Production population"):
        h._inputs(h.FITS[0], "test", tmp_path/"head", None, None, None, None, None)
    assert calls == ["TRAIN"]


def test_diagnostics_use_new_budget_and_match_legacy_without_mutation(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    model = h.StrongPolicyInference(str(h.c.verify_hash(values["initializer"]))).model
    old_budget = h.prior.c.budget_boundary
    before = {key: value.clone() for key, value in model.state_dict().items()}
    calls = []
    monkeypatch.setattr(h.c, "budget_boundary", lambda base: calls.append(base))
    actual = h.training_report(model, values["rows"]["old"], tmp_path, values["config"])
    expected = h.prior.training_report(model, values["rows"]["old"], tmp_path, values["config"])
    assert h._legacy_report(actual) == expected and len(calls) == 4
    assert h.prior.c.budget_boundary is old_budget
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    monkeypatch.setattr(h.c, "budget_boundary", lambda base: (_ for _ in ()).throw(ValueError("new ceiling")))
    with pytest.raises(ValueError, match="new ceiling"):
        h.training_report(model, values["rows"]["old"], tmp_path, values["config"])


def test_original_model_full_policy_actual_two_step_reproduction(tmp_path):
    from tools.full_teacher_fit_128 import source as original
    h.old.initialize_worker()
    spec = next(s for s in h.old.read_json(original.c.SOURCE/"schedule.json")["main"] if s["split"] == "train")
    initial = h.old.read_json(original.c.SOURCE/"models_lock.json")["models"][f"anchor_{spec['fit']}"]
    payload = torch.load(h.c.verify_hash(initial), map_location="cpu", weights_only=True)
    real = original.reconstruct_row(spec, LegalActionAttentionScorerConfig(**payload["model_config"]))
    # Lawful real dimensions with an explicit synthetic target to exercise a
    # positive opportunity; this isolated fixture is not a scientific sample.
    real["old_sums_wdl"] = [-128, -64, 0]+[128-i for i in range(len(real["members"])-3)]
    real["outsider_teacher"] = 3 if len(real["members"]) > 3 else -1
    added = []
    for i in range(4):
        item = copy.deepcopy(real)
        item.update(id=f"fixture-added:{i}", index=12000+i, pair=f"fixture:{i}", source_panel="fixture",
                    source_game=i, environment_seed=97000+i, source_fit=spec["fit"], seat=0, phase=0,
                    cell=["fixture", spec["fit"], 0, 0])
        added.append(item)
    config = {**copy.deepcopy(h.CONFIG), "updates": 2, "batch_size": 2}
    values = {"rows": {"old": [real], "added": added}, "initializer": initial, "config": config, "narrow": [0]}
    values["control"] = control(tmp_path/"prior", spec["fit"], values)
    ref = h.fit(spec["fit"], "real-fixture", tmp_path/"head", **values)
    audit = h.c.verified(h.audit_fit(spec["fit"], ref, "real-fixture", tmp_path/"head", **values))
    assert audit["trainable_parameters"] > 98817 and audit["complete_fit_bitwise_identical"]
    assert audit["row_exposures"] == 2


def test_zero_outsider_loss_keeps_full_policy_adamw_steps(tmp_path):
    values = fixture(tmp_path)
    for kind, items in values["rows"].items():
        for i, original in enumerate(items):
            replacement = row(original["index"], kind, 3+i % 2)
            for key in ("id", "pair", "source_fit", "seat", "phase", "cell", "source_panel", "source_game", "environment_seed"):
                if key in original:
                    replacement[key] = original[key]
            items[i] = replacement
    values["config"].update(lr=.01, weight_decay=.1)
    values["control"] = control(tmp_path/"zero_prior", h.FITS[0], values)
    ref = h.fit(h.FITS[0], "zero", tmp_path/"zero_head", **values)
    result = h.c.verified(ref)
    assert result["optimization_stability"]["preclip_gradient_norms"] == [0.]*4
    initial = torch.load(h.c.verify_hash(values["initializer"]), weights_only=True)["model_state_dict"]
    final = torch.load(h.c.verify_hash(result["selected_model"]), weights_only=True)["model_state_dict"]
    assert all(torch.equal(initial[k], final[k]) for k in initial if k.startswith("value_head."))
    assert any(not torch.equal(initial[k], final[k]) for k in initial if not k.startswith("value_head."))
    assert h.c.verified(h.audit_fit(h.FITS[0], ref, "zero", tmp_path/"zero_head", **values))["passed"]


def test_all_nine_models_audited_before_sealing(tmp_path, monkeypatch):
    fits, audits, controls = [], [], {}
    for fit in h.FITS:
        folder = tmp_path/f"inputs{fit}"
        folder.mkdir()
        values = fixture(folder, fit)
        trained = h.c.verified(values["control"]["result"])
        controls[f"anchor_{fit}"] = values["initializer"]
        controls[f"broad_{fit}"] = trained["arms"]["broad"]["selected_model"]
        ref = h.fit(fit, "registry", tmp_path/"head", **values)
        fits.append(ref)
        audits.append(h.audit_fit(fit, ref, "registry", tmp_path/"head", **values))
    monkeypatch.setattr(sys.modules[h.__package__], "source", SimpleNamespace(control_models=lambda binding, base: controls), raising=False)
    ref = h.models_lock(fits, audits, "registry", tmp_path/"head")
    registry = h.validate_models("registry", tmp_path/"head")
    assert registry == h.c.verified(ref) and len(registry["models"]) == 9
    registry["models"][f"weighted_{h.FITS[0]}"] = controls[f"broad_{h.FITS[0]}"]
    (tmp_path/"head/models_lock.json").write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="registry differs"):
        h.validate_models("registry", tmp_path/"head")


def target_row(index, sums, prefix="old"):
    item = row(index, prefix, len(sums))
    item["old_sums_wdl"] = sums
    item["outsider_teacher"] = max(range(3, len(sums)), key=lambda i: sums[i]) if len(sums) > 3 else -1
    if prefix == "added":
        item.update(source_panel="fixture", source_game=index)
    return item


@pytest.mark.parametrize("sums,expected", [([1, 5, 3], 0), ([1, 5, 3, 4, 2], 0),
    ([1, 5, 3, 5, 2], 0), ([1, 5, 3, 6], 1), ([1, 5, 3, 6, 7], 2),
    ([-128, -128, -128, 128, -128], 256)])
def test_exact_integer_opportunity_including_ties_and_singletons(sums, expected):
    item = target_row(0, sums)
    assert h.opportunity_units(item) == h.audit_opportunity_units(item) == expected


def weight_fixture():
    pools = {"old": [target_row(0, [0, 0, 0, 2, 1]), target_row(1, [0, 0, 0, 4]),
                      target_row(2, [8, 0, 0, 7, 1])],
             "added": [target_row(3, [0, 0, 0, 10, 1], "added"),
                       target_row(4, [0, 0, 0, 0, -1], "added")]}
    plan = [[("old", 0), ("old", 0), ("old", 0), ("old", 1), ("old", 2), ("old", 2),
             ("added", 0), ("added", 1), ("added", 1), ("added", 1)]]
    return pools, plan


def test_separate_exposure_normalizers_and_concentration_match_hand_calculation():
    pools, plan = weight_fixture()
    weights, receipt = h.weight_plan(pools, plan)
    assert h.independent_weights(pools, plan) == (weights, receipt)
    assert weights == {"old:0": 1.2, "old:1": 2.4, "old:2": 0., "added:3": 4., "added:4": 0.}
    old, added = receipt["pools"]["old"], receipt["pools"]["added"]
    assert old["normalizer_exposures_denominator"] == 6 and old["normalizer_opportunity_units_numerator"] == 10
    assert added["normalizer_exposures_denominator"] == 4 and added["normalizer_opportunity_units_numerator"] == 10
    assert old["exposure_effective_sample_size"] == 100/28
    assert old["row_effective_sample_size"] == 100/52
    assert old["game_or_pair_effective_sample_size"] == 1.
    assert old["zero_opportunity_exposures"] == 2 and old["zero_loss_exposures"] == 3
    assert old["multiple_outsider_opportunity_mass_fraction"] == .6
    assert old["maximum_row_weight_mass_fraction"] == .6
    assert old["maximum_game_or_pair_weight_mass_fraction"] == 1.
    for pool, block in receipt["pools"].items():
        assert sum(item["normalized_weight"]*item["exposures"] for item in block["per_row"])/block["exposures"] == pytest.approx(1.)


def test_independent_normalization_detects_shared_collector_error(monkeypatch):
    pools, plan = weight_fixture()
    original = h._weight_summary
    def damaged(*args, **kwargs):
        mapping, report = original(*args, **kwargs)
        return {key: value*2 for key, value in mapping.items()}, report
    monkeypatch.setattr(h, "_weight_summary", damaged)
    with pytest.raises(ValueError, match="Independent opportunity normalization differs"):
        h.independent_weights(pools, plan)


def test_zero_opportunity_pool_is_undefined_without_pruning():
    pools, plan = weight_fixture()
    pools["added"][0]["old_sums_wdl"] = [10, 0, 0, 9, 1]
    with pytest.raises(ValueError, match="positive exposed planning opportunity"):
        h.weight_plan(pools, plan)
    with pytest.raises(ValueError, match="positive exposed planning opportunity"):
        h.independent_weights(pools, plan)


def test_weighted_soft_loss_keeps_targets_zero_rows_and_full_denominator():
    rows = [target_row(0, [0, 0, 0, 2, 1]), target_row(1, [0, 0, 0, 4]),
            target_row(2, [8, 0, 0, 7, 1])]
    weights = {rows[0]["id"]: 2., rows[1]["id"]: 3., rows[2]["id"]: 0.}
    logits = torch.arange(24, dtype=torch.float32).reshape(3, 8).div(10).requires_grad_()
    losses, choices = h.objective(logits, rows, weights)
    ordinary, original_choices = h.unweighted_objective(logits, rows)
    independent, independent_choices = h.audit_objective(logits, rows, weights)
    assert choices == original_choices == independent_choices
    assert torch.equal(losses, independent) and torch.equal(losses, ordinary*torch.tensor([2., 3., 0.]))
    assert losses[1] == losses[2] == 0
    losses.mean().backward()
    gradient = logits.grad.clone()
    logits.grad.zero_()
    (ordinary[0]*2/3).backward()
    assert torch.equal(gradient, logits.grad)


@pytest.mark.parametrize("sums,choice,expected", [([0, 5, 1], -1, 0), ([10, 5, 1, 9, 0], 4, 0),
    ([0, 5, 1, 9, 0], 4, 4), ([0, 5, 1, 9, 7], 4, 2), ([0, 5, 1, 9], 3, 0)])
def test_marginal_regret_measures_actual_shortlist_improvement(sums, choice, expected):
    item = target_row(0, sums)
    assert h.marginal_regret_units(item, choice) == h.marginal_regret_units(item, choice, audit=True) == expected
