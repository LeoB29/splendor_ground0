"""Bounded exact paired-mixture schedules and independently reproduced fits."""
import copy
import json
import math
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from . import learning as h


def row(index, prefix="old", support=6):
    count = support+3
    groups = torch.tensor([0, 0]+list(range(1, count-1)), dtype=torch.long)
    sums = [-128, 0, 16, 20, 100, 100, -30, 1][:support]
    target = max(range(3, support), key=lambda j: sums[j]) if support > 3 else -1
    return {"id": f"{prefix}:{index}", "pair": f"{prefix}:pair:{index//2}", "index": index,
        "source_fit": h.FITS[index % 3], "split": "train", "seat": index % 2, "phase": index % 3,
        "observation": torch.arange(6, dtype=torch.float32)/10+(index % 5)/20,
        "features": torch.arange(count*4, dtype=torch.float32).reshape(count, 4)/100+(index % 5)/20,
        "groups": groups, "group_count": count-1,
        "members": [torch.where(groups == group)[0] for group in range(support)],
        "old_sums_wdl": sums, "outsider_start": min(3, support), "outsider_support": max(0, support-3),
        "outsider_teacher": target}


def fixture(tmp_path, updates=4):
    old = [row(i, support=n) for i, n in enumerate((3, 4, 5, 8, 6))]
    added = []
    for i in range(8):
        item = row(11950+i, "added", (3, 4, 5, 8)[i % 4])
        item.update(source_panel="panel%d" % (i//4), source_game=i, environment_seed=97000+i,
                    source_fit=h.FITS[i//4], seat=0, phase=1)
        item["cell"] = [item["source_panel"], item["source_fit"], item["seat"], item["phase"]]
        added.append(item)
    torch.manual_seed(719)
    model_config = LegalActionAttentionScorerConfig(observation_size=6, action_feature_size=4,
        state_hidden_size=8, action_hidden_size=8, scorer_hidden_size=8, state_depth=1, action_depth=1,
        attention_layers=1, attention_heads=2, attention_feedforward_size=16)
    model = LegalActionAttentionScorerModel(model_config)
    path = tmp_path/"initial.pt"
    h.save_model(path, model, 0, {"fixture": True})
    settings = {**copy.deepcopy(h.CONFIG), "updates": updates, "batch_size": 4,
                "train_rows": 5, "added_rows": 8, "narrow_rows_per_fit": 2}
    return {"old": old, "added": added}, h.c.artifact(path), settings, [0, 4]


def kwargs(values):
    pools, initial, settings, selected = values
    return {"rows": pools, "initializer": initial, "config": settings, "narrow": selected}


def schedule_pools():
    old = [{"id": f"old:{i}", "features": torch.empty(1+i % 13, 0)} for i in range(9461)]
    added = []
    for panel in ("panel0", "panel1"):
        for fit in h.FITS:
            for seat in (0, 1):
                for phase in range(3):
                    cell = [panel, fit, seat, phase]
                    for _ in range(20):
                        i = len(added)
                        added.append({"id": f"added:{i}", "source_panel": panel, "source_fit": fit,
                            "source_game": i, "seat": seat, "phase": phase, "cell": cell,
                            "features": torch.empty(1+i % 17, 0)})
    return {"old": old, "added": added}


def test_full_production_exposure_and_all_three_independent_schedules():
    pools = schedule_pools()
    for fi, fit in enumerate(h.FITS):
        narrow = [i for i in range(720) if fi*5 <= i % 20 < (fi+1)*5]
        actual = h.batch_plan(pools, narrow, fit)
        independent = h.independent_plan(pools, narrow, fit)
        assert actual == independent
        receipt = h._paired_receipt(actual, pools, h.CONFIG, fixture=False)
        for arm, games in (("narrow", 180), ("broad", 720)):
            summary = receipt["arms"][arm]
            assert Counter(summary["batch_lengths"]) == {64: 1772, 53: 12}
            assert summary["old_exposures"] == summary["added_exposures"] == 57022
            assert summary["row_exposures"] == 114044 and summary["updates"] == 1784
            assert len(summary["per_added_game_exposures"]) == games
            assert max(summary["per_added_cell_exposures"].values())-min(summary["per_added_cell_exposures"].values()) <= 1
            assert sum(summary["per_added_game_exposures"].values()) == 57022
            assert summary["attention_padding_proxy"] > 0
        assert receipt["arms"]["narrow"]["old_batch_stream_sha256"] == receipt["arms"]["broad"]["old_batch_stream_sha256"]
        assert receipt["arms"]["narrow"]["added_cell_batch_stream_sha256"] == receipt["arms"]["broad"]["added_cell_batch_stream_sha256"]
        assert receipt["arms"]["narrow"]["batch_sequence_sha256"] != receipt["arms"]["broad"]["batch_sequence_sha256"]


def test_reference_lengths_match_frozen_predecessor_contract():
    from tools.soft_teacher_targets_128 import learning as previous
    pools = schedule_pools()
    for fit in h.FITS:
        expected = []
        for epoch in range(13):
            expected.extend(len(batch) for batch in previous.batch_plan(pools["old"], fit, epoch))
        assert h._reference_lengths(pools["old"], fit, h.CONFIG) == expected[:1784]


def test_new_cell_and_pool_schedules_are_seeded_without_collector_helpers(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    pools, _, settings, selected = values
    expected = h.batch_plan(pools, selected, h.FITS[0], settings)
    monkeypatch.setattr(h, "_reference_lengths", lambda *a: pytest.fail("Collector length reconstruction used"))
    monkeypatch.setattr(h, "_cycling_indices", lambda *a: pytest.fail("Collector sampler used"))
    monkeypatch.setattr(h, "_cell", lambda *a: pytest.fail("Collector cell reconstruction used"))
    assert h.independent_plan(pools, selected, h.FITS[0], settings) == expected
    assert h.independent_plan(pools, selected, h.FITS[1], settings) != expected


@pytest.mark.parametrize("damage", ["duplicate", "imbalance", "cell", "game", "validation", "assessment", "overlap", "temperature"])
def test_training_population_firewall(tmp_path, damage):
    values = fixture(tmp_path)
    pools, _, settings, selected = values
    if damage == "duplicate":
        selected[:] = [0, 0]
    elif damage == "imbalance":
        selected[:] = [0, 1]
    elif damage == "cell":
        pools["added"][0]["cell"][2] = 1
    elif damage == "game":
        pools["added"][1]["source_game"] = pools["added"][0]["source_game"]
    elif damage == "validation":
        pools["old"][0]["split"] = "validation"
    elif damage == "assessment":
        pools["added"][0]["assessment"] = []
    elif damage == "overlap":
        pools["added"][0]["id"] = pools["old"][0]["id"]
    else:
        settings["target_temperature"] = .1
    with pytest.raises(ValueError):
        h._inputs(h.FITS[0], "fixture", tmp_path, pools, values[1], settings, selected)


def test_production_overrides_forbidden(tmp_path):
    pools, initial, settings, selected = fixture(tmp_path)
    with pytest.raises(ValueError, match="Fixture cannot target production"):
        h._inputs(h.FITS[0], "fixture", h.OUTPUT, pools, initial, settings, selected)


def test_public_training_entry_never_loads_validation(tmp_path, monkeypatch):
    pools = schedule_pools()
    calls = []
    fake = SimpleNamespace(load_rows=lambda binding, name, base: calls.append(name) or pools[name],
                           initializer=lambda fit, base: {}, narrow_indices=lambda fit, binding, base: list(range(180)))
    monkeypatch.setattr(sys.modules[h.__package__], "training_source", fake, raising=False)
    monkeypatch.setattr(h, "_validate_rows", lambda *a: None)
    # The deliberately malformed narrow selection fails before opening anything else.
    with pytest.raises(ValueError, match="imbalance"):
        h._inputs(h.FITS[0], "production-shape", tmp_path, None, None, None, None)
    assert calls == ["old", "added"]


@pytest.mark.parametrize("sums", [[0, 0], [0, 1], [-128, 128], [-128, -127, 0, 127, 128]])
def test_soft_targets_and_canonical_gradient_are_independent(sums):
    item = row(1, support=3+len(sums))
    item["old_sums_wdl"][3:] = sums
    item["outsider_teacher"] = 3+max(range(len(sums)), key=lambda j: sums[j])
    weights = h.soft_targets(item, dtype=torch.float64)
    assert torch.equal(weights, h.audit_soft_targets(item, dtype=torch.float64))
    expected = [math.exp((value-max(sums))/256/.05) for value in sums]
    assert weights.tolist() == pytest.approx([x/math.fsum(expected) for x in expected], abs=1e-15)
    logits = torch.linspace(-1, 1, len(item["features"]))[None].requires_grad_()
    actual, choices = h.objective(logits, [item])
    independent, other = h.audit_objective(logits, [item])
    assert torch.equal(actual, independent) and choices == other
    assert torch.equal(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                       torch.autograd.grad(independent.sum(), logits)[0])


def test_actual_paired_miniature_training_and_complete_independent_audit(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    base = tmp_path/"run"
    fit_ref = h.fit(h.FITS[0], "fixture", base, **kwargs(values))
    result = h.c.verified(fit_ref)
    assert fit_ref["index"] == 0 and set(result["arms"]) == {"narrow", "broad"}
    assert result["complete"] and not result["validation_read"] and not result["assessment_read"]
    for arm in h.ARMS:
        trained = result["arms"][arm]
        assert trained["updates"] == trained["selected_update"] == 4
        assert trained["old_exposures"] == trained["added_exposures"] == 5
        assert trained["row_exposures"] == 10
        assert [entry["update"] for entry in trained["checkpoints"]] == [0, 4]
        assert trained["selected_model"]["source_fit"] == h.FITS[0]
        diagnostics = h.c.verified(trained["selected_training"])
        assert diagnostics["old"]["metrics"]["all_rows"]["rows"] == 5
        assert diagnostics["added"]["metrics"]["all_rows"]["rows"] == 8
        assert diagnostics["added"]["metrics_by_slice"]["optimizer_exposed"]["metrics"]["all_rows"]["rows"] == (2 if arm == "narrow" else 8)
        assert diagnostics["added"]["metrics_by_slice"]["not_optimizer_exposed"]["metrics"]["all_rows"]["rows"] == (6 if arm == "narrow" else 0)
    monkeypatch.setattr(h, "_fit_arm", lambda *a: pytest.fail("Collector fit used by audit"))
    monkeypatch.setattr(h, "batch_plan", lambda *a: pytest.fail("Collector plan used by audit"))
    monkeypatch.setattr(h, "objective", lambda *a: pytest.fail("Collector objective used by audit"))
    monkeypatch.setattr(h, "soft_targets", lambda *a, **k: pytest.fail("Collector target used by audit"))
    audit_ref = h.audit_fit(h.FITS[0], fit_ref, "fixture", base, **kwargs(values))
    audit = h.c.verified(audit_ref)
    assert audit["passed"] and audit["complete_fit_bitwise_identical"]
    assert audit["soft_target_arithmetic_independently_reproduced"] and audit["exact_between_arm_old_and_cell_streams"]
    for arm in h.ARMS:
        assert audit["arms"][arm]["selected_model"] == result["arms"][arm]["selected_model"]
        assert audit["arms"][arm]["optimization_stability_independently_reproduced"]


def test_original_model_real_lawful_row_both_arm_two_step_reproduction(tmp_path):
    from tools.full_teacher_fit_128 import source as inherited
    h.old.initialize_worker()
    schedule = h.old.read_json(inherited.c.SOURCE/"schedule.json")["main"]
    spec = next(item for item in schedule if item["split"] == "train")
    registry = h.old.read_json(inherited.c.SOURCE/"models_lock.json")["models"]
    initial = registry[f"anchor_{spec['fit']}"]
    payload = torch.load(h.c.verify_hash(initial), map_location="cpu", weights_only=True)
    config = LegalActionAttentionScorerConfig(**payload["model_config"])
    real = inherited.reconstruct_row(spec, config)
    # Copies exercise real model/tensor dimensions; these are synthetic source
    # identities inside an isolated fixture, never a scientific data sample.
    added = []
    for i in range(4):
        item = copy.deepcopy(real)
        item.update(id=f"fixture-added:{i}", index=12000+i, pair=f"fixture:{i}",
                    source_panel="fixture", source_game=i, environment_seed=97000+i,
                    source_fit=spec["fit"], seat=0, phase=0)
        item["cell"] = ["fixture", spec["fit"], 0, 0]
        added.append(item)
    settings = {**copy.deepcopy(h.CONFIG), "updates": 2, "batch_size": 2}
    values = ({"old": [real], "added": added}, initial, settings, [0])
    base = tmp_path/"real_model"
    fit_ref = h.fit(spec["fit"], "real_fixture", base, **kwargs(values))
    audit_ref = h.audit_fit(spec["fit"], fit_ref, "real_fixture", base, **kwargs(values))
    assert h.c.verified(audit_ref)["complete_fit_bitwise_identical"]
    result = h.c.verified(fit_ref)
    for arm in h.ARMS:
        assert result["arms"][arm]["row_exposures"] == 2
        assert result["arms"][arm]["old_exposures"] == result["arms"][arm]["added_exposures"] == 1


@pytest.mark.parametrize("damage", ["old_stream", "cell_stream", "subset", "selected", "model", "metric", "gradient"])
def test_corrupt_paired_evidence_rejected(tmp_path, damage):
    values = fixture(tmp_path)
    base = tmp_path/"run"
    fit_ref = h.fit(h.FITS[0], "fixture", base, **kwargs(values))
    result = h.c.verified(fit_ref)
    if damage in ("old_stream", "cell_stream"):
        key = "old_batch_stream_sha256" if damage == "old_stream" else "added_cell_batch_stream_sha256"
        result["paired_schedule"]["arms"]["broad"][key] = "wrong"
    elif damage == "subset":
        result["narrow_indices"] = [0, 1]
    elif damage == "selected":
        result["arms"]["broad"]["selected_model"] = result["arms"]["broad"]["checkpoints"][0]["checkpoint"]
    elif damage == "gradient":
        result["arms"]["narrow"]["optimization_stability"]["preclip_gradient_norms"][0] += .1
    elif damage == "model":
        point = result["arms"]["broad"]["checkpoints"][-1]
        payload = torch.load(h.c.verify_hash(point["checkpoint"]), weights_only=True)
        next(iter(payload["model_state_dict"].values())).flatten()[0] += 1
        path = tmp_path/"damaged.pt"
        torch.save(payload, path)
        point["checkpoint"] = h.c.artifact(path)
    else:
        point = result["arms"]["broad"]["checkpoints"][-1]
        report = h.c.verified(point["training"])
        report["added"]["metrics"]["all_rows"]["mean_target_kl"] += .1
        path = tmp_path/"damaged_metric.json"
        h.old.save_new(path, report)
        point["training"] = h.c.artifact(path)
    path = tmp_path/"damaged_result.json"
    h.old.save_new(path, result)
    with pytest.raises(ValueError):
        h.audit_fit(h.FITS[0], h.c.artifact(path), "fixture", base, **kwargs(values))


def test_real_both_zero_loss_arms_still_execute_adamw(tmp_path):
    values = fixture(tmp_path)
    pools, initial, settings, selected = values
    for name, items in pools.items():
        for i, previous in enumerate(items):
            replacement = row(previous["index"], name, 3+i % 2)
            for key in ("id", "pair", "source_fit", "seat", "phase", "cell", "source_panel", "source_game", "environment_seed"):
                if key in previous:
                    replacement[key] = previous[key]
            items[i] = replacement
    settings.update(lr=.01, weight_decay=.1)
    fit_ref = h.fit(h.FITS[0], "zero", tmp_path/"zero", **kwargs(values))
    result = h.c.verified(fit_ref)
    before = torch.load(h.c.verify_hash(initial), weights_only=True)["model_state_dict"]
    for arm in h.ARMS:
        after = torch.load(h.c.verify_hash(result["arms"][arm]["selected_model"]), weights_only=True)["model_state_dict"]
        assert any(not torch.equal(before[key], after[key]) for key in before if not key.startswith("value_head."))
        assert result["arms"][arm]["optimization_stability"]["preclip_gradient_norms"] == [0.]*4
    assert h.c.verified(h.audit_fit(h.FITS[0], fit_ref, "zero", tmp_path/"zero", **kwargs(values)))["passed"]


def test_model_registry_has_all_fifteen_models_and_exact_source_assignment(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    pools, initial, settings, selected = values
    base = tmp_path/"registry"
    fits = [h.fit(fit, "registry", base, **kwargs(values)) for fit in h.FITS]
    audits = [h.audit_fit(fit, ref, "registry", base, **kwargs(values)) for fit, ref in zip(h.FITS, fits)]
    controls = {f"{arm}_{fit}": {**initial, "source_fit": fit} for arm in ("anchor", "lowrate", "soft") for fit in h.FITS}
    monkeypatch.setattr(sys.modules[h.__package__], "training_source", SimpleNamespace(control_models=lambda binding, base: controls), raising=False)
    lock_ref = h.models_lock(fits, audits, "registry", base)
    registry = h.validate_models("registry", base)
    assert registry == h.c.verified(lock_ref) and len(registry["models"]) == 15
    assert all(ref["source_fit"] == int(key.rsplit("_", 1)[1]) for key, ref in registry["models"].items())
    damaged = copy.deepcopy(registry)
    damaged["models"][f"broad_{h.FITS[0]}"] = damaged["models"][f"narrow_{h.FITS[0]}"]
    path = base/"models_lock.json"
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="final linkage"):
        h.validate_models("registry", base)
