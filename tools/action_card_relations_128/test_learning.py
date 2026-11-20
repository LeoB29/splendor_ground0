"""Bounded matched residual, corruption, exact replay and interruption tests."""
import copy
import json
from collections import Counter

import pytest
import torch

from splendor_ai.alphazero.strong_policy import save_model as save_base
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from tools.independent_game_coverage_128_recovery_01.test_learning import row as old_row, schedule_pools
from . import learning as h


def row(index, prefix="old", support=6):
    value = old_row(index, prefix, support)
    observation = torch.zeros(295)
    for slot in range(12):
        card = observation[33+13*slot:46+13*slot]
        card[0] = 1.
        card[1+slot//4] = 1.
        card[3+slot % 5] = 1.
        card[8:13] = torch.tensor([(slot+j) % 5 for j in range(5)])/7.
    observation[189:195] = torch.tensor([1., 2., 1., 0., 2., 1.])/10.
    observation[195:200] = torch.tensor([1., 0., 0., 1., 0.])/7.
    actions = torch.zeros(len(value["features"]), 79)
    for i in range(len(actions)):
        color = max(0, i-1) % 5
        actions[i, 1] = 1.
        actions[i, 36+color] = .5
        actions[i, 54+color] = .5
    value.update(observation=observation, features=actions)
    return value


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
    config = LegalActionAttentionScorerConfig(observation_size=295, action_feature_size=79,
        state_hidden_size=8, action_hidden_size=8, scorer_hidden_size=8, state_depth=1, action_depth=1,
        attention_layers=1, attention_heads=2, attention_feedforward_size=16)
    model = LegalActionAttentionScorerModel(config)
    path = tmp_path/"initial.pt"
    save_base(path, model, 0, {"fixture": True})
    settings = {**copy.deepcopy(h.CONFIG), "updates": updates, "batch_size": 4,
                "train_rows": 5, "added_rows": 8, "narrow_rows_per_fit": 2}
    values = {"rows": pools, "initializer": h.c.artifact(path, source_fit=fit), "config": settings, "narrow": [0, 4]}
    ref = h.prior.fit(fit, "prior-fixture", tmp_path/"prior", **values)
    checked = h.prior.audit_fit(fit, ref, "prior-fixture", tmp_path/"prior", **values)
    values["control"] = {"result": ref, "audit": checked}
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


def test_both_arms_exact_replay_and_intact_base(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "fixture", tmp_path/"new", **values)
    result = h.c.verified(ref)
    base = torch.load(h.c.verify_hash(values["initializer"]), weights_only=True)["model_state_dict"]
    assert set(result["arms"]) == set(h.ARMS)
    for arm, reference in result["arms"].items():
        trained = h.c.verified(reference)
        assert trained["trainable_parameters"] == 9441 and trained["selected_update"] == 4
        assert [point["update"] for point in trained["checkpoints"]] == [0, 4]
        final = h.residual.RelationInference(h.c.verify_hash(trained["selected_model"])).model
        assert h._equal_torch(final.base.state_dict(), base)
        first = torch.load(h.c.verify_hash(trained["checkpoints"][0]["checkpoint"]), weights_only=True)["model_state_dict"]
        assert any(not torch.equal(tensor, first[name]) for name, tensor in final.state_dict().items() if not name.startswith("base."))
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
    pointer = tmp_path/f"resume/fits/fit_{h.FITS[0]}/relations/resume.json"
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


@pytest.mark.parametrize("damage", ["frozen", "residual", "arm", "seed", "metric", "membership", "selected", "gradient", "control_receipt"])
def test_independent_audit_rejects_corruption(tmp_path, damage):
    values = fixture(tmp_path)
    ref = h.fit(h.FITS[0], "bad", tmp_path/"new", **values)
    result = h.c.verified(ref)
    trained = h.c.verified(result["arms"]["relations"])
    if damage in ("frozen", "residual", "arm", "seed"):
        point = trained["checkpoints"][-1]
        payload = torch.load(h.c.verify_hash(point["checkpoint"]), weights_only=True)
        if damage == "arm":
            payload["arm"] = "raw"
        elif damage == "seed":
            payload["seed"] += 1
        else:
            key = next(k for k in payload["model_state_dict"] if k.startswith("base.") == (damage == "frozen"))
            payload["model_state_dict"][key].flatten()[0] += 1
        path = tmp_path/"corrupt.pt"
        torch.save(payload, path)
        point["checkpoint"] = h.c.artifact(path, source_fit=h.FITS[0], arm="relations")
    elif damage == "metric":
        report = h.c.verified(trained["selected_training"])
        report["added"]["metrics"]["all_rows"]["mean_target_kl"] += .1
        path = tmp_path/"corrupt_report.json"
        h.old.save_new(path, report)
        trained["checkpoints"][-1]["training"] = h.c.artifact(path)
    elif damage == "membership":
        trained["trainable_parameter_names"].append("base.state_encoder.0.weight")
    elif damage == "selected":
        trained["selected_model"] = trained["checkpoints"][0]["checkpoint"]
    elif damage == "gradient":
        trained["optimization_stability"]["preclip_gradient_norms"][0] += .1
    else:
        trained["identity"]["matched_broad_control"]["batch_receipt"]["batch_sequence_sha256"] = "wrong"
    path = tmp_path/"corrupt_arm.json"
    h.old.save_new(path, trained)
    result["arms"]["relations"] = h.c.artifact(path, arm="relations", fit=h.FITS[0])
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
    path = tmp_path/f"resume/fits/fit_{h.FITS[0]}/relations/resume.json"
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


def test_all_six_models_reproduced_before_seal(tmp_path, monkeypatch):
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
    registry["models"][f"relations_{h.FITS[0]}"] = controls[f"anchor_{h.FITS[0]}"]
    (tmp_path/"new/models_lock.json").write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="registry differs"):
        h.validate_models("registry", tmp_path/"new")
