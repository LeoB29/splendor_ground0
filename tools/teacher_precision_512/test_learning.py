"""Exact target arithmetic, full-policy matching, reproduction and resume tests."""
import copy
import json
from collections import Counter

import pytest
import torch

from splendor_ai.alphazero.strong_policy import save_model as save_base
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from tools.independent_game_coverage_128_recovery_01.test_learning import row as old_row, schedule_pools
from . import learning as h


row = old_row


def fixture(tmp_path, fit=None, updates=4):
    from tools.independent_game_coverage_128_recovery_01.test_learning import fixture as old_fixture
    fit = h.FITS[0] if fit is None else fit
    pools, initial, settings, narrow = old_fixture(tmp_path, updates)
    initial["source_fit"] = fit
    settings = {**copy.deepcopy(h.CONFIG), **{k: settings[k] for k in ("updates", "batch_size", "train_rows", "added_rows", "narrow_rows_per_fit")}}
    values = {"rows": pools, "initializer": initial, "config": settings, "narrow": narrow}
    ref = h.prior.fit(fit, "prior-fixture", tmp_path/"prior", **values)
    checked = h.prior.audit_fit(fit, ref, "prior-fixture", tmp_path/"prior", **values)
    values["control"] = {"result": ref, "audit": checked}
    for item in pools["added"]:
        sums = [4*v for v in item["old_sums_wdl"]]
        if len(sums) > 4:
            sums[3], sums[4] = 401, 79
        item.update(teacher_sums_wdl=sums, teacher_worlds=512,
                    teacher_outsider=max(range(3, len(sums)), key=lambda j: sums[j]) if len(sums) > 3 else -1)
    return values


def test_exact_all_three_production_broad_schedules_without_training():
    pools = schedule_pools()
    for j, fit in enumerate(h.FITS):
        narrow = [i for i in range(720) if 5*j <= i % 20 < 5*(j+1)]
        expected = h.prior.batch_plan(pools, narrow, fit, h.prior.CONFIG)["broad"]
        assert h.batch_plan(pools, narrow, fit) == expected == h.independent_plan(pools, narrow, fit)
        receipt = h.prior._plan_receipt(expected, pools)
        assert receipt["updates"] == 1784 and receipt["row_exposures"] == 114044
        assert receipt["old_exposures"] == receipt["added_exposures"] == 57022
        assert Counter(receipt["batch_lengths"]) == {64: 1772, 53: 12}


def test_full_policy_exact_replay_and_frozen_value(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "fixture", tmp_path/"new", **values)
    result = h.c.verified(ref)
    base = torch.load(h.c.verify_hash(values["initializer"]), weights_only=True)["model_state_dict"]
    assert set(result["arms"]) == set(h.ARMS)
    for arm, reference in result["arms"].items():
        trained = h.c.verified(reference)
        assert trained["trainable_parameters"] == sum(v.numel() for k,v in base.items() if not k.startswith("value_head.")) and trained["selected_update"] == 4
        assert [point["update"] for point in trained["checkpoints"]] == [0, 4]
        final = h.StrongPolicyInference(str(h.c.verify_hash(trained["selected_model"]))).model
        assert all(torch.equal(v, base[k]) for k,v in final.state_dict().items() if k.startswith("value_head."))
        first = torch.load(h.c.verify_hash(trained["checkpoints"][0]["checkpoint"]), weights_only=True)["model_state_dict"]
        assert any(not torch.equal(tensor, first[name]) for name, tensor in final.state_dict().items() if not name.startswith("value_head."))
        assert not trained["assessment_read"] and not trained["validation_read"]
    monkeypatch.setattr(h, "fit", lambda *a, **k: pytest.fail("Collector fit used in audit"))
    monkeypatch.setattr(h, "batch_plan", lambda *a, **k: pytest.fail("Collector schedule used in audit"))
    monkeypatch.setattr(h, "objective", lambda *a, **k: pytest.fail("Collector objective used in audit"))
    checked = h.c.verified(h.audit_fit(h.FITS[0], ref, "fixture", tmp_path/"new", **values))
    assert checked["passed"]
    assert all(all(arm[key] for key in h.AUDIT_FLAGS) for arm in checked["arms"].values())


def test_resume_reconstructs_exact_uninterrupted_trajectory(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    baseline = h.c.verified(h.fit(h.FITS[0], "fixture", tmp_path/"baseline", **values))
    original = h.objective
    calls = []
    def interrupt(*args, **kwargs):
        if torch.is_grad_enabled():
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(h, "objective", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        h.fit(h.FITS[0], "fixture", tmp_path/"resume", **values)
    pointer = tmp_path/f"resume/fits/fit_{h.FITS[0]}/precision/resume.json"
    assert h.old.read_json(pointer)["update"] == 1
    monkeypatch.setattr(h, "objective", original)
    ref = h.fit(h.FITS[0], "fixture", tmp_path/"resume", **values)
    result = h.c.verified(ref)
    for arm in h.ARMS:
        before, after = h.c.verified(baseline["arms"][arm]), h.c.verified(result["arms"][arm])
        assert before["optimization_stability"] == after["optimization_stability"]
        a = torch.load(h.c.verify_hash(before["selected_model"]), weights_only=True)
        b = torch.load(h.c.verify_hash(after["selected_model"]), weights_only=True)
        assert h._equal_torch(a, b)
    assert h.c.verified(h.audit_fit(h.FITS[0], ref, "fixture", tmp_path/"resume", **values))["passed"]
    assert h.fit(h.FITS[0], "fixture", tmp_path/"resume", **values) == ref


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
        h.fit(h.FITS[0], "bad", tmp_path/"new", **values)
    assert not (tmp_path/"new/fits").exists()


@pytest.mark.parametrize("damage", ["frozen", "policy", "arm", "generation", "metric", "membership", "selected", "gradient", "control_receipt"])
def test_independent_audit_rejects_corruption(tmp_path, damage):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "bad", tmp_path/"new", **values)
    result = h.c.verified(ref)
    trained = h.c.verified(result["arms"]["precision"])
    if damage in ("frozen", "policy", "arm", "generation"):
        point = trained["checkpoints"][-1]
        payload = torch.load(h.c.verify_hash(point["checkpoint"]), weights_only=True)
        if damage == "arm":
            payload["metadata"]["arm"] = "wrong"
        elif damage == "generation":
            payload["generation"] += 1
        else:
            key = next(k for k in payload["model_state_dict"] if k.startswith("value_head.") == (damage == "frozen"))
            payload["model_state_dict"][key].flatten()[0] += 1
        path = tmp_path/"corrupt.pt"
        torch.save(payload, path)
        point["checkpoint"] = h.c.artifact(path, source_fit=h.FITS[0], arm="precision")
    elif damage == "metric":
        report = h.c.verified(trained["selected_training"])
        report["added"]["metrics"]["all_rows"]["mean_target_kl"] += .1
        path = tmp_path/"corrupt_report.json"
        h.old.save_new(path, report)
        trained["checkpoints"][-1]["training"] = h.c.artifact(path)
    elif damage == "membership":
        trained["trainable_parameter_names"].append("value_head.0.weight")
    elif damage == "selected":
        trained["selected_model"] = trained["checkpoints"][0]["checkpoint"]
    elif damage == "gradient":
        trained["optimization_stability"]["preclip_gradient_norms"][0] += .1
    else:
        trained["identity"]["matched_broad_control"]["batch_receipt"]["batch_sequence_sha256"] = "wrong"
    path = tmp_path/"corrupt_arm.json"
    h.old.save_new(path, trained)
    result["arms"]["precision"] = h.c.artifact(path, arm="precision", fit=h.FITS[0])
    path = tmp_path/"corrupt_result.json"
    h.old.save_new(path, result)
    with pytest.raises(ValueError):
        h.audit_fit(h.FITS[0], h.c.artifact(path), "bad", tmp_path/"new", **values)


def test_resume_identity_and_bytes_tamper_rejected(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    real_save = h._save_progress
    def stop(*args, **kwargs):
        real_save(*args, **kwargs)
        raise RuntimeError("interrupted after durable checkpoint")
    monkeypatch.setattr(h, "_save_progress", stop)
    with pytest.raises(RuntimeError):
        h.fit(h.FITS[0], "fixture", tmp_path/"resume", **values)
    monkeypatch.setattr(h, "_save_progress", real_save)
    path = tmp_path/f"resume/fits/fit_{h.FITS[0]}/precision/resume.json"
    original = h.old.read_json(path)
    path.write_text(json.dumps({**original, "identity_sha256": "0"*64}))
    with pytest.raises(ValueError, match="Resume identity"):
        h.fit(h.FITS[0], "fixture", tmp_path/"resume", **values)
    path.write_text(json.dumps(original))
    checkpoint = h.c.verify_hash(original["checkpoint"])
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError):
        h.fit(h.FITS[0], "fixture", tmp_path/"resume", **values)


def test_fixtures_cannot_enter_production(tmp_path):
    values = fixture(tmp_path)
    with pytest.raises(ValueError, match="Fixture cannot target production"):
        h._inputs(h.FITS[0], "bad", h.OUTPUT, **values)


def test_all_three_models_reproduced_before_seal(tmp_path, monkeypatch):
    from . import training_source
    fits, audits, controls = [], [], {}
    for fit in h.FITS:
        folder = tmp_path/f"inputs{fit}"
        folder.mkdir()
        values = fixture(folder, fit, updates=4)
        controls[f"anchor_{fit}"] = values["initializer"]
        controls[f"broad_{fit}"] = h.c.verified(values["control"]["result"])["arms"]["broad"]["selected_model"]
        ref = h.fit(fit, "registry", tmp_path/"new", **values)
        fits.append(ref)
        audits.append(h.audit_fit(fit, ref, "registry", tmp_path/"new", **values))
    monkeypatch.setattr(training_source, "control_models", lambda binding, base: controls)
    ref = h.models_lock(fits, audits, "registry", tmp_path/"new")
    registry = h.validate_models("registry", tmp_path/"new")
    assert registry == h.c.verified(ref) and len(registry["models"]) == 9
    with pytest.raises(ValueError, match="All three paired fits"):
        h.models_lock(fits[:2], audits[:2], "registry", tmp_path/"incomplete")
    registry["models"][f"precision_{h.FITS[0]}"] = controls[f"anchor_{h.FITS[0]}"]
    (tmp_path/"new/models_lock.json").write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="registry differs"):
        h.validate_models("registry", tmp_path/"new")


@pytest.mark.parametrize("support", [1,3,4,5,8])
def test512_soft_targets_and_independent_gradients(support):
    value = row(0, support=support)
    sums = [min(512, 4*v+1) for v in value["old_sums_wdl"]]
    value.update(teacher_sums_wdl=sums, teacher_worlds=512,
                 teacher_outsider=max(range(3,support), key=lambda j:sums[j]) if support > 3 else -1)
    h.validate_target(value, added=True)
    assert torch.equal(h.soft_targets(value), h.audit_soft_targets(value))
    z = torch.linspace(-1,1,len(value["features"]))[None].requires_grad_()
    a, ac = h.objective(z,[value]); a.mean().backward(); grad = z.grad.clone()
    z.grad = None
    b, bc = h.audit_objective(z,[value]); b.mean().backward()
    assert torch.equal(a,b) and ac == bc and torch.equal(grad,z.grad)
    if support < 5:
        assert float(a[0]) == 0 and not bool(grad.any())
    else:
        expected = torch.softmax(torch.tensor([.5+v/1024 for v in sums[3:]])/.05,dim=0)
        assert torch.equal(expected,h.soft_targets(value))


@pytest.mark.parametrize("change",["denominator","sum","label","missing","old"])
def test_precision_target_contract_rejects_changes(change):
    value = row(11950,"added",6)
    sums = [4*v for v in value["old_sums_wdl"]]
    value.update(teacher_sums_wdl=sums,teacher_worlds=512,teacher_outsider=4)
    if change == "denominator": value["teacher_worlds"] = 128
    elif change == "sum": value["teacher_sums_wdl"][0] = 513
    elif change == "label": value["teacher_outsider"] = 5
    elif change == "missing": del value["teacher_outsider"]
    with pytest.raises(ValueError): h.validate_target(value,added=change != "old")


def test512_report_keeps_fractional_regret_without_truncation(tmp_path):
    values = fixture(tmp_path)
    model,_,_,_,_ = h._model(values["initializer"],"precision",19,values["config"])
    report = h.training_report(model,values["rows"]["added"],tmp_path,values["config"],precision=True)
    checked = h.training_report(model,values["rows"]["added"],tmp_path,values["config"],precision=True,audit=True)
    h._compare(report,checked)
    assert report["target_worlds"] == 512
    units = sum(r["old_regret_units"] for r in report["rows"])
    assert all(type(r["old_regret_units"]) is int for r in report["rows"])
    assert report["metrics"]["all_rows"]["mean_old_score_regret"] == units/(1024*8)
    assert any(r["old_regret_units"] %4 for r in report["rows"])
