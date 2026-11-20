"""Bounded head-only reproduction, control matching and frozen-feature tests."""
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
        "added_rows": 8, "narrow_rows_per_fit": 2, "trainable_parameters": sum(p.numel() for p in model.score_head.parameters())}
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


def test_headonly_actual_fit_full_audit_and_frozen_tensors(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "new-fixture", tmp_path/"head", **values)
    result = h.c.verified(ref)
    assert ref["index"] == 0 and result["arm"] == "headonly"
    assert result["row_exposures"] == 10 and result["old_exposures"] == result["added_exposures"] == 5
    assert [r["update"] for r in result["checkpoints"]] == [0, 4]
    initial = torch.load(h.c.verify_hash(values["initializer"]), weights_only=True)["model_state_dict"]
    final = torch.load(h.c.verify_hash(result["selected_model"]), weights_only=True)["model_state_dict"]
    assert all(torch.equal(initial[k], final[k]) for k in initial if not k.startswith("score_head."))
    assert any(not torch.equal(initial[k], final[k]) for k in initial if k.startswith("score_head."))
    report = h.c.verified(result["selected_training"])
    assert report["added"]["metrics_by_slice"]["optimizer_exposed"]["metrics"]["all_rows"]["rows"] == 8
    assert report["added"]["metrics_by_slice"]["not_optimizer_exposed"]["metrics"]["all_rows"]["rows"] == 0
    assert not report["assessment_read"] and not report["validation_read"]
    monkeypatch.setattr(h, "fit", lambda *a, **k: pytest.fail("Collector refit used"))
    monkeypatch.setattr(h, "batch_plan", lambda *a, **k: pytest.fail("Collector batch planner used"))
    monkeypatch.setattr(h, "objective", lambda *a, **k: pytest.fail("Collector objective used"))
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


@pytest.mark.parametrize("damage", ["frozen", "head", "metric", "membership", "selected", "gradient", "control_receipt"])
def test_independent_audit_rejects_corruption(tmp_path, damage):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "bad", tmp_path/"head", **values)
    result = h.c.verified(ref)
    if damage in ("frozen", "head"):
        point = result["checkpoints"][-1]
        payload = torch.load(h.c.verify_hash(point["checkpoint"]), weights_only=True)
        key = next(k for k in payload["model_state_dict"] if k.startswith("score_head.") == (damage == "head"))
        payload["model_state_dict"][key].flatten()[0] += 1
        path = tmp_path/"corrupt.pt"
        torch.save(payload, path)
        point["checkpoint"] = h.c.artifact(path, source_fit=h.FITS[0], arm="headonly")
    elif damage == "metric":
        report = h.c.verified(result["selected_training"])
        report["added"]["metrics"]["all_rows"]["mean_target_kl"] += .1
        path = tmp_path/"corrupt_report.json"
        h.old.save_new(path, report)
        result["checkpoints"][-1]["training"] = h.c.artifact(path)
    elif damage == "membership":
        result["trainable_parameter_names"].append("state_encoder.0.weight")
    elif damage == "selected":
        result["selected_model"] = result["checkpoints"][0]["checkpoint"]
    elif damage == "gradient":
        result["optimization_stability"]["preclip_gradient_norms"][0] += .1
    else:
        result["matched_broad_control"]["batch_receipt"]["batch_sequence_sha256"] = "wrong"
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
    assert actual == expected and len(calls) == 4
    assert h.prior.c.budget_boundary is old_budget
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    monkeypatch.setattr(h.c, "budget_boundary", lambda base: (_ for _ in ()).throw(ValueError("new ceiling")))
    with pytest.raises(ValueError, match="new ceiling"):
        h.training_report(model, values["rows"]["old"], tmp_path, values["config"])


def test_original_model_98817_head_actual_two_step_reproduction(tmp_path):
    from tools.full_teacher_fit_128 import source as original
    h.old.initialize_worker()
    spec = next(s for s in h.old.read_json(original.c.SOURCE/"schedule.json")["main"] if s["split"] == "train")
    initial = h.old.read_json(original.c.SOURCE/"models_lock.json")["models"][f"anchor_{spec['fit']}"]
    payload = torch.load(h.c.verify_hash(initial), map_location="cpu", weights_only=True)
    real = original.reconstruct_row(spec, LegalActionAttentionScorerConfig(**payload["model_config"]))
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
    assert audit["trainable_parameters"] == 98817 and audit["complete_fit_bitwise_identical"]
    assert audit["row_exposures"] == 2


def test_zero_outsider_loss_only_head_decays(tmp_path):
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
    assert all(torch.equal(initial[k], final[k]) for k in initial if not k.startswith("score_head."))
    assert any(not torch.equal(initial[k], final[k]) for k in initial if k.startswith("score_head."))
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
    registry["models"][f"headonly_{h.FITS[0]}"] = controls[f"broad_{h.FITS[0]}"]
    (tmp_path/"head/models_lock.json").write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="registry differs"):
        h.validate_models("registry", tmp_path/"head")
