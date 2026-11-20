"""Independent miniature fitting, exact exposure, support and data firewall tests."""
import copy
from dataclasses import asdict
import sys
from types import SimpleNamespace

import pytest
import torch

from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from . import learning as h


def rows_fixture():
    rows = []
    for index, support in enumerate((3, 4, 5, 8, 6)):
        length = support+3
        groups = torch.tensor([0, 0]+list(range(1, length-1)), dtype=torch.long)
        sums = [-128, 0, 16, 20, 100, 100, -30, 1][:support]
        start = min(3, support)
        target = max(range(start, support), key=lambda i: sums[i]) if support > start else -1
        rows.append({"id": f"breadth128:fixture:{index}", "pair": f"breadth128:fixture:{index//2}",
            "index": index, "source_fit": h.FITS[index % 3], "split": "train",
            "observation": torch.arange(6, dtype=torch.float32)/10+index/20,
            "features": torch.arange(length*4, dtype=torch.float32).reshape(length, 4)/100+index/20,
            "groups": groups, "group_count": length-1,
            "members": [torch.where(groups == group)[0] for group in range(support)],
            "old_sums_wdl": sums, "outsider_start": start, "outsider_support": support-start,
            "outsider_teacher": target})
    return rows


def fixture_inputs(tmp_path, updates=3):
    torch.manual_seed(771)
    config = LegalActionAttentionScorerConfig(observation_size=6, action_feature_size=4,
        state_hidden_size=8, action_hidden_size=8, scorer_hidden_size=8,
        state_depth=1, action_depth=1, attention_layers=1, attention_heads=2,
        attention_feedforward_size=16)
    model = LegalActionAttentionScorerModel(config)
    initial = tmp_path/"initial.pt"
    h.save_model(initial, model, 0, {"fixture": True})
    settings = {**copy.deepcopy(h.CONFIG), "updates": updates, "batch_size": 2,
                "checkpoints": list(range(updates+1))}
    return rows_fixture(), h.c.artifact(initial), settings


def test_exact_batch_stream_short_batches_and_epoch_boundaries():
    rows = rows_fixture()
    config = {**h.CONFIG, "batch_size": 2}
    for fit in h.FITS:
        for epoch in (0, 1, 512):
            actual = h.batch_plan(rows, fit, epoch, config)
            assert actual == h.independent_plan(rows, fit, epoch, config)
            assert sorted(len(batch) for batch in actual) == [1, 2, 2]
            assert sorted(j for batch in actual for j in batch) == list(range(5))
    assert h.batch_plan(rows, h.FITS[0], 0, config) != h.batch_plan(rows, h.FITS[0], 1, config)


def test_independent_loss_canonical_max_and_unoffered_gradients():
    rows = rows_fixture()
    logits = torch.linspace(-1, 1, 11)[None].repeat(5, 1).requires_grad_()
    logits.data[0, len(rows[0]["features"]):] = -torch.inf
    left, choices = h.objective(logits, rows)
    right, alternate = h.audit_objective(logits, rows)
    assert torch.equal(left, right) and choices == alternate
    grad = torch.autograd.grad(left.sum(), logits, retain_graph=True)[0]
    independent = torch.autograd.grad(right.sum(), logits)[0]
    assert torch.equal(grad, independent)
    assert choices[:2] == [-1, 3] and torch.equal(left[:2], torch.zeros(2))
    for i, row in enumerate(rows):
        offered = set(int(member) for group in row["members"][3:] for member in group)
        unused = [j for j in range(logits.shape[1]) if j not in offered]
        assert torch.equal(grad[i, unused], torch.zeros(len(unused)))


def test_training_targets_ties_and_alias_corruption_rejected():
    rows = rows_fixture()
    h._validate_rows(rows)
    changed = copy.deepcopy(rows)
    changed[3]["outsider_teacher"] = 5  # Tied, but not the first maximum.
    with pytest.raises(ValueError):
        h._validate_rows(changed)
    changed = copy.deepcopy(rows)
    changed[0]["members"][0] = changed[0]["members"][0][:1]
    with pytest.raises(ValueError):
        h._validate_rows(changed)


def test_alias_max_changes_target_score_and_shared_tie_gradient():
    row = rows_fixture()[3]
    row["groups"][-1] = 4
    row["members"][4] = torch.where(row["groups"] == 4)[0]
    h._validate_rows([row])
    logits = torch.zeros((1, len(row["features"])), requires_grad=True)
    target_aliases = row["members"][4]
    logits.data[0, target_aliases] = 2
    left, chosen = h.objective(logits, [row])
    right, alternate = h.audit_objective(logits, [row])
    assert chosen == alternate == [4] and torch.equal(left, right)
    direct = torch.autograd.grad(left.sum(), logits, retain_graph=True)[0]
    audited = torch.autograd.grad(right.sum(), logits)[0]
    assert torch.equal(direct, audited)
    assert direct[0, target_aliases[0]] == direct[0, target_aliases[1]] < 0


@pytest.mark.parametrize("damage", ["validation", "assessment", "fresh_outcomes", "duplicate", "sums"])
def test_training_data_firewall(damage):
    rows = rows_fixture()
    if damage == "validation":
        rows[0]["split"] = "validation"
    elif damage in ("assessment", "fresh_outcomes"):
        rows[0][damage] = [1]
    elif damage == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    else:
        rows[0]["old_sums_wdl"][0] = -129
    with pytest.raises(ValueError):
        h._validate_rows(rows)


def test_production_fixture_override_forbidden(tmp_path):
    rows, initial, settings = fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="Fixture cannot target production"):
        h._inputs(h.FITS[0], "fixture", h.OUTPUT, rows, initial, settings)


def test_zero_and_singleton_rows_preserve_real_adamw_steps(tmp_path):
    rows, initial, settings = fixture_inputs(tmp_path, 2)
    settings.update(lr=.01, weight_decay=.1)  # Make fixture-only decay representable in float32.
    subset = rows[:2]
    item = h.fit(h.FITS[0], "zero", tmp_path/"zero", rows=subset, initializer=initial, config=settings)
    result = h.c.verified(item)
    assert result["updates"] == 2 and result["row_exposures"] == 4
    assert all(value == 2 for value in result["per_row_exposures"].values())
    before = torch.load(h.c.verify_hash(initial), weights_only=True)["model_state_dict"]
    after = torch.load(h.c.verify_hash(result["selected_model"]), weights_only=True)["model_state_dict"]
    # Zero losses still construct gradients and execute AdamW's weight decay.
    assert any(not torch.equal(before[key], after[key]) for key in before if not key.startswith("value_head."))
    checked = h.audit_fit(h.FITS[0], item, "zero", tmp_path/"zero", rows=subset, initializer=initial, config=settings)
    assert h.c.verified(checked)["passed"]


def test_actual_miniature_full_optimizer_and_all_checkpoint_reproduction(tmp_path, monkeypatch):
    rows, initial, settings = fixture_inputs(tmp_path, 4)
    settings["regret_capture_guide"] = -100.
    item = h.fit(h.FITS[0], "miniature", tmp_path/"run", rows=rows, initializer=initial, config=settings)
    result = h.c.verified(item)
    assert result["updates"] == 4 and result["epochs_entered"] == 2
    assert result["selected_model"] == result["checkpoints"][-1]["checkpoint"]
    assert result["selected_update"] == 4 and result["final_always_selected"]
    assert h.c.verified(result["checkpoints"][0]["training"])["regret_capture_guide_met"]
    assert sum(result["per_row_exposures"].values()) == result["row_exposures"]
    assert not result["assessment_read"] and not result["validation_read"]
    monkeypatch.setattr(h, "objective", lambda *args: pytest.fail("Collector objective called in independent reproduction"))
    monkeypatch.setattr(h, "batch_plan", lambda *args: pytest.fail("Collector batch plan called in independent reproduction"))
    checked = h.audit_fit(h.FITS[0], item, "miniature", tmp_path/"run", rows=rows, initializer=initial, config=settings)
    audit = h.c.verified(checked)
    assert audit["passed"] and audit["complete_fit_bitwise_identical"]
    assert audit["all_training_metrics_independently_reproduced"] and audit["value_head_frozen"]
    assert audit["selected_model"] == result["selected_model"]


def test_actual_original_model_and_reconstructed_source_row_two_step_reproduction(tmp_path):
    from . import source
    h.old.initialize_worker()
    schedule = h.old.read_json(h.c.SOURCE/"schedule.json")["main"]
    spec = next(item for item in schedule if item["split"] == "train")
    registry = h.old.read_json(h.c.SOURCE/"models_lock.json")["models"]
    initial = registry[f"anchor_{spec['fit']}"]
    payload = torch.load(h.c.verify_hash(initial), map_location="cpu", weights_only=True)
    model_config = LegalActionAttentionScorerConfig(**payload["model_config"])
    row = source.reconstruct_row(spec, model_config)
    settings = {**copy.deepcopy(h.CONFIG), "updates": 2, "checkpoints": [0, 1, 2]}
    base = tmp_path/"actual_model_fixture"
    item = h.fit(spec["fit"], "actual_source_fixture", base,
                 rows=[row], initializer=initial, config=settings)
    checked = h.audit_fit(spec["fit"], item, "actual_source_fixture", base,
                         rows=[row], initializer=initial, config=settings)
    result, audit = h.c.verified(item), h.c.verified(checked)
    assert result["row_exposures"] == 2 and result["train_rows"] == 1
    assert result["initializer"] == initial and result["selected_update"] == 2
    assert audit["passed"] and audit["complete_fit_bitwise_identical"]
    assert audit["value_head_frozen"] and audit["all_training_metrics_independently_reproduced"]


def test_public_input_path_only_reads_train_not_validation(tmp_path, monkeypatch):
    rows, initial, settings = fixture_inputs(tmp_path)
    settings["train_rows"] = len(rows)
    calls = []
    def load_rows(binding, split, base):
        calls.append(split)
        assert split == "train"
        return rows
    fake = SimpleNamespace(load_rows=load_rows, initializer=lambda fit, base: initial)
    monkeypatch.setattr(sys.modules[h.__package__], "source", fake, raising=False)
    monkeypatch.setattr(h, "CONFIG", settings)
    actual, artifact, config = h._inputs(h.FITS[0], "read", tmp_path, None, None, None)
    assert actual is rows and artifact == initial and config == settings and calls == ["train"]


def test_guide_zero_denominator_stays_undefined():
    rows = rows_fixture()[:2]
    records = [{"outsider_support": r["outsider_support"], "loss": 0., "first_winner": True if r["outsider_support"] else None,
                "any_old_max": True if r["outsider_support"] else None, "old_regret_units": 0} for r in rows]
    actual = h._metrics(records)
    independent = h._metrics(records, audit=True)
    assert actual == independent
    assert actual["all_rows"]["accuracy_denominator"] == 1
    assert actual["multiple_outsiders"]["mean_loss"] is None


@pytest.mark.parametrize("damage", ["selected", "exposure", "checkpoint", "metric"])
def test_independent_reproduction_rejects_corrupted_evidence(tmp_path, damage):
    rows, initial, settings = fixture_inputs(tmp_path, 2)
    base = tmp_path/"run"
    item = h.fit(h.FITS[0], "corruption", base, rows=rows, initializer=initial, config=settings)
    result = h.c.verified(item)
    if damage == "selected":
        result["selected_model"] = result["checkpoints"][0]["checkpoint"]
    elif damage == "exposure":
        result["per_row_exposures"][rows[0]["id"]] += 1
    elif damage == "checkpoint":
        payload = torch.load(h.c.verify_hash(result["checkpoints"][1]["checkpoint"]), weights_only=True)
        key = next(iter(payload["model_state_dict"]))
        payload["model_state_dict"][key].flatten()[0] += 1
        path = tmp_path/"damaged.pt"; torch.save(payload, path)
        result["checkpoints"][1]["checkpoint"] = h.c.artifact(path)
    else:
        report = h.c.verified(result["checkpoints"][1]["training"])
        report["metrics"]["all_rows"]["mean_loss"] += .1
        path = tmp_path/"damaged.json"; h.old.save_new(path, report)
        result["checkpoints"][1]["training"] = h.c.artifact(path)
    path = tmp_path/"changed_result.json"; h.old.save_new(path, result)
    with pytest.raises(ValueError):
        h.audit_fit(h.FITS[0], h.c.artifact(path), "corruption", base, rows=rows, initializer=initial, config=settings)
