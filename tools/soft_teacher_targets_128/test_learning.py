"""Independent miniature fitting, exact exposure, support and data firewall tests."""
import copy
from dataclasses import asdict
import math
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


def test_full_corpus_budget_is_twelve_passes_plus_eight_batches():
    # Equal lengths isolate exact dose accounting without constructing full
    # production tensors or performing an optimization step.
    rows = [{"features": torch.empty(1, 0)} for _ in range(9461)]
    assert h.CONFIG["train_rows"] == 9461 and h.CONFIG["validation_rows"] == 2489
    assert h.CONFIG["updates"] == 1784 and h.CONFIG["batch_size"] == 64
    for fit in h.FITS:
        plans = [h.batch_plan(rows, fit, epoch) for epoch in range(13)]
        assert len(plans[0]) == 148
        assert sorted(len(batch) for batch in plans[0]) == [53]+[64]*147
        final_partial = plans[12][:8]
        assert sum(len(batch) for plan in plans[:12] for batch in plan) == 9461*12
        assert sum(len(batch) for batch in final_partial) == 512
        assert sum(len(batch) for plan in plans[:12] for batch in plan)+512 == 114044
        assert len([batch for plan in plans[:12] for batch in plan])+len(final_partial) == 1784


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


@pytest.mark.parametrize("sums", [[0, 0], [0, 1], [-128, 128], [-128, -127, 0, 127, 128], [17, 17, 17, 17, 17]])
def test_finite_target_independent_score_units_ties_near_ties_and_extremes(sums):
    row = rows_fixture()[2 if len(sums) == 2 else 3]
    row["old_sums_wdl"][3:] = sums
    targets = h.soft_targets(row, dtype=torch.float64)
    independent = h.audit_soft_targets(row, dtype=torch.float64)
    assert torch.equal(targets, independent)
    scaled = [(128+value)/256/.05 for value in sums]
    weights = [math.exp(value-max(scaled)) for value in scaled]
    expected = [value/math.fsum(weights) for value in weights]
    assert targets.tolist() == pytest.approx(expected, rel=0, abs=1e-15)
    assert bool((targets > 0).all()) and float(targets.sum()) == pytest.approx(1.)
    for i, left in enumerate(sums):
        for j, right in enumerate(sums):
            assert float(targets[i]/targets[j]) == pytest.approx(math.exp((left-right)/256/.05), rel=1e-13)
            if left == right:
                assert targets[i] == targets[j]


def test_independent_optimizer_target_does_not_call_collector_target(monkeypatch):
    rows = rows_fixture()
    logits = torch.linspace(-1, 1, 11)[None].repeat(5, 1)
    expected = h.objective(logits, rows)
    monkeypatch.setattr(h, "soft_targets", lambda *a, **k: pytest.fail("Collector target used by audit"))
    actual = h.audit_objective(logits, rows)
    assert torch.equal(expected[0], actual[0]) and expected[1] == actual[1]


def test_finite_target_has_zero_gradient_at_matching_gaps():
    row = rows_fixture()[3]
    logits = torch.zeros((1, len(row["features"])), dtype=torch.float64, requires_grad=True)
    for j, members in enumerate(row["members"]):
        logits.data[0, members] = (128+row["old_sums_wdl"][j])/256/.05
    loss, _ = h.objective(logits, [row])
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    assert gradient.abs().max() < 1e-14
    terms = h.diagnostic_terms(logits, [row])[0]
    independent = h.diagnostic_terms(logits, [row], audit=True)[0]
    h._compare(terms, independent)
    assert abs(terms["target_kl"]) < 1e-14
    assert terms["soft_loss"] == pytest.approx(terms["target_entropy"], abs=1e-14)


def test_diagnostics_preserve_original_hard_loss_and_expose_finite_metrics():
    from tools.higher_teacher_rate_128 import learning as previous
    rows = rows_fixture()
    logits = torch.linspace(-4, 3, 11)[None].repeat(5, 1)
    hard, _ = previous.objective(logits, rows)
    terms = h.diagnostic_terms(logits, rows)
    independent = h.diagnostic_terms(logits, rows, audit=True)
    h._compare(terms, independent)
    assert [entry["hard_loss"] for entry in terms] == hard.tolist()
    assert any(abs(entry["hard_loss"]-entry["soft_loss"]) > 1e-3 for entry in terms)
    for entry in terms:
        assert entry["soft_loss"] == pytest.approx(entry["target_entropy"]+entry["target_kl"], abs=1e-12)
        assert entry["target_kl"] >= -1e-12 and entry["policy_entropy"] >= 0
    for entry in terms[:2]:
        assert all(entry[name] == 0 for name in ("hard_loss", "soft_loss", "target_entropy", "target_kl", "policy_entropy"))


def test_initial_policy_kl_independent_direction_and_short_support():
    for left, right in (([], []), ([1.], [-5.]), ([0., 1.], [0., 1.]), ([0., 0.], [0., 2.])):
        actual = h.initial_policy_kl(left, right)
        audited = h.initial_policy_kl(left, right, audit=True)
        assert actual == pytest.approx(audited, abs=1e-12)
    expected = math.log((1+math.exp(2))/2)-1
    assert h.initial_policy_kl([0., 0.], [0., 2.]) == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError, match="support changed"):
        h.initial_policy_kl([0.], [0., 1.])


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
    logits.data[0, target_aliases] = .5
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
    assert audit["soft_target_arithmetic_independently_reproduced"]
    assert audit["selected_model"] == result["selected_model"]
    assert audit["optimization_stability_independently_reproduced"]
    assert result["optimization_stability"]["updates"] == 4
    assert len(result["optimization_stability"]["preclip_gradient_norms"]) == 4
    h._compare(result["optimization_stability"], audit["optimization_stability"])


def test_production_settings_change_only_prescribed_target():
    from tools.higher_teacher_rate_128 import common as previous
    h._control_recipe(h.CONFIG, previous.CONFIG)
    assert h.CONFIG["lr"] == previous.CONFIG["lr"] == 1e-4
    assert h.CONFIG["target_temperature"] == .05
    rows = rows_fixture()
    for fit in h.FITS:
        for epoch in (0, 1, 12):
            assert h.batch_plan(rows, fit, epoch, h.CONFIG) == h.independent_plan(rows, fit, epoch, previous.CONFIG)


@pytest.mark.parametrize("changed", ["lr", "weight_decay", "gradient_norm", "batch_size", "updates", "batch_tag", "checkpoints", "target_temperature", "objective"])
def test_second_recipe_changes_are_rejected(changed):
    from tools.higher_teacher_rate_128 import common as previous
    settings = copy.deepcopy(h.CONFIG)
    value = settings[changed]
    settings[changed] = value[:-1] if isinstance(value, list) else value+"wrong" if isinstance(value, str) else value*2
    with pytest.raises(ValueError):
        h._control_recipe(settings, previous.CONFIG)


def test_actual_matched_fixture_changes_target_not_training_stream(tmp_path):
    from tools.higher_teacher_rate_128 import learning as previous
    rows, initial, settings = fixture_inputs(tmp_path, 3)
    hard_settings = {**copy.deepcopy(settings),
        "objective": previous.CONFIG["objective"], "comparison_control": previous.CONFIG["comparison_control"]}
    del hard_settings["target_temperature"], hard_settings["required_control_endpoint"]
    # This constructs a small test-only control, not a replay of an old fit.
    hard_ref = previous.fit(h.FITS[0], "hard_fixture", tmp_path/"hard_fixture",
                           rows=rows, initializer=initial, config=hard_settings)
    soft_ref = h.fit(h.FITS[0], "soft_fixture", tmp_path/"soft_fixture", rows=rows,
                     initializer=initial, config=settings, high_rate_control=hard_ref)
    high, low = h.c.verified(soft_ref), h.c.verified(hard_ref)
    matched = high["matched_high_rate_control"]
    assert matched["result"] == hard_ref and matched["only_target_changes_fitting_recipe"]
    assert matched["same_seed_batch_stream_and_per_row_exposures"] and not matched["old_fit_replayed"]
    assert high["batch_sequence_sha256"] == low["batch_sequence_sha256"]
    assert high["per_row_exposures"] == low["per_row_exposures"]
    assert high["training_rows_sha256"] == low["training_rows_sha256"]
    high_tensors = torch.load(h.c.verify_hash(high["selected_model"]), weights_only=True)["model_state_dict"]
    low_tensors = torch.load(h.c.verify_hash(low["selected_model"]), weights_only=True)["model_state_dict"]
    assert any(not torch.equal(high_tensors[name], low_tensors[name]) for name in high_tensors if not name.startswith("value_head."))
    checked = h.audit_fit(h.FITS[0], soft_ref, "soft_fixture", tmp_path/"soft_fixture", rows=rows,
                          initializer=initial, config=settings, high_rate_control=hard_ref)
    audit = h.c.verified(checked)
    assert audit["matched_high_rate_control_verified"] and audit["matched_high_rate_control"] == matched
    for field in ("batch_sequence_sha256", "row_exposures", "training_rows_sha256", "initializer", "selected_update"):
        damaged = copy.deepcopy(high)
        damaged[field] = "wrong"
        with pytest.raises(ValueError, match="stream or dose"):
            h._match_control(damaged, hard_ref, low)


def test_stability_summary_and_nonfinite_guard():
    values = [0., .25, 1., 2., 4.]
    expected = h._stability(values, h.CONFIG)
    h._compare(expected, h._stability(values, h.CONFIG, audit=True))
    assert expected["updates"] == 5 and expected["preclip_gradient_norm_max"] == 4.
    assert expected["preclip_gradient_norm_mean"] == 1.45
    assert expected["norm_above_clip_limit_steps"] == 2 and expected["norm_above_clip_limit_fraction"] == .4
    for values in ([float("nan")], [float("inf")], [-1.]):
        with pytest.raises(ValueError, match="gradient norm"):
            h._stability(values, h.CONFIG)


def test_intermediate_probe_and_full_endpoint_slices_reproduce(tmp_path):
    rows, initial, settings = fixture_inputs(tmp_path, 4)
    probe = {rows[0]["id"], rows[2]["id"]}
    settings["training_probe_ids"] = sorted(probe)
    base = tmp_path/"scoped_diagnostics"
    item = h.fit(h.FITS[0], "scopes", base, rows=rows, initializer=initial, config=settings)
    result = h.c.verified(item)
    reports = [h.c.verified(curve["training"]) for curve in result["checkpoints"]]
    assert [report["metrics"]["all_rows"]["rows"] for report in reports] == [5, 2, 2, 2, 5]
    assert [report["diagnostic_scope"] for report in reports] == [
        "full_train", "original_168_train_probe", "original_168_train_probe", "original_168_train_probe", "full_train"]
    assert reports[0]["metrics_by_slice"]["original_168_train"]["metrics"]["all_rows"]["rows"] == 2
    assert reports[-1]["metrics_by_slice"]["added_9293_train"]["metrics"]["all_rows"]["rows"] == 3
    assert reports[1]["initial_old_regret_units"] == reports[0]["metrics_by_slice"]["original_168_train"]["initial_old_regret_units"]
    for report in reports:
        assert not report["validation_read"] and not report["assessment_read"]
        assert report["data_scope"] == "train"
        assert report["loss_semantics"] == "hard_first_winner_cross_entropy_common_to_controls"
        assert report["initial_policy_kl_scope"] == "recorded_outsiders_only_KL(initial||current)"
    assert reports[0]["metrics"]["all_rows"]["mean_initial_policy_kl"] is None
    assert reports[-1]["metrics"]["all_rows"]["initial_policy_kl_rows"] == len(rows)
    assert reports[-1]["metrics"]["all_rows"]["mean_initial_policy_kl"] >= -1e-12
    for row in reports[-1]["rows"]:
        if row["outsider_support"] < 2:
            assert row["initial_policy_kl"] == 0
    checked = h.audit_fit(h.FITS[0], item, "scopes", base, rows=rows, initializer=initial, config=settings)
    assert h.c.verified(checked)["complete_fit_bitwise_identical"]


def test_probe_resolver_never_opens_validation(tmp_path, monkeypatch):
    rows = [{"id": f"row:{index}"} for index in range(9461)]
    probe = {row["id"] for row in rows[:168]}
    calls = []
    def train_probe_ids(binding, base):
        calls.append((binding, base))
        return probe
    fake = SimpleNamespace(train_probe_ids=train_probe_ids,
                           load_rows=lambda *args: pytest.fail("Probe opened a corpus tensor file"))
    monkeypatch.setattr(sys.modules[h.__package__], "source", fake, raising=False)
    assert h._probe_ids("probe", tmp_path, rows, h.CONFIG, fixture=False) == probe
    assert calls == [("probe", tmp_path)]
    probe.pop()
    with pytest.raises(ValueError, match="168"):
        h._probe_ids("probe", tmp_path, rows, h.CONFIG, fixture=False)


def test_explicit_validation_report_is_truthfully_labelled(tmp_path):
    rows, initial, settings = fixture_inputs(tmp_path)
    model, _, _ = h._model(initial, 71, settings)
    for row in rows:
        row["split"] = "validation"
    with pytest.raises(ValueError, match="Mixed diagnostic"):
        h.training_report(model, rows, tmp_path, settings)
    report = h.training_report(model, rows, tmp_path, settings, data_scope="validation",
                               diagnostic_scope="full_validation_matched_source_fit")
    independent = h.training_report(model, rows, tmp_path, settings, data_scope="validation",
                                    diagnostic_scope="full_validation_matched_source_fit", audit=True)
    h._compare(report, independent)
    assert report["validation_read"] and not report["assessment_read"]
    assert report["data_scope"] == "validation" and report["metrics"]["all_rows"]["rows"] == 5


def test_diagnostic_reports_preserve_rows_model_tensors_and_random_state(tmp_path):
    rows, initial, settings = fixture_inputs(tmp_path)
    model, _, _ = h._model(initial, 71, settings)
    model.train()
    tensors = {name: value.clone() for name, value in model.state_dict().items()}
    identity, order = h._rows_identity(rows), [row["id"] for row in rows]
    random_state = torch.get_rng_state().clone()
    h.training_report(model, rows, tmp_path, settings)
    h.training_report(model, rows, tmp_path, settings, audit=True)
    assert [row["id"] for row in rows] == order and h._rows_identity(rows) == identity
    assert all(torch.equal(value, tensors[name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), random_state)


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
    records = [{"outsider_support": r["outsider_support"], "loss": 0., "soft_loss": 0., "target_entropy": 0.,
                "target_kl": 0., "policy_entropy": 0., "initial_policy_kl": None,
                "first_winner": True if r["outsider_support"] else None,
                "any_old_max": True if r["outsider_support"] else None, "old_regret_units": 0} for r in rows]
    actual = h._metrics(records)
    independent = h._metrics(records, audit=True)
    assert actual == independent
    assert actual["all_rows"]["accuracy_denominator"] == 1
    assert actual["multiple_outsiders"]["mean_loss"] is None


@pytest.mark.parametrize("damage", ["selected", "exposure", "checkpoint", "metric", "soft_metric", "entropy", "movement", "gradient_norm", "matched_control"])
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
    elif damage in ("metric", "soft_metric", "entropy", "movement"):
        report = h.c.verified(result["checkpoints"][1]["training"])
        key = {"metric": "mean_loss", "soft_metric": "mean_target_kl", "entropy": "mean_target_entropy",
               "movement": "mean_initial_policy_kl"}[damage]
        report["metrics"]["all_rows"][key] += .1
        path = tmp_path/"damaged.json"; h.old.save_new(path, report)
        result["checkpoints"][1]["training"] = h.c.artifact(path)
    elif damage == "gradient_norm":
        result["optimization_stability"]["preclip_gradient_norms"][0] += .1
    else:
        result["matched_high_rate_control"] = {"wrong": True}
    path = tmp_path/"changed_result.json"; h.old.save_new(path, result)
    with pytest.raises(ValueError):
        h.audit_fit(h.FITS[0], h.c.artifact(path), "corruption", base, rows=rows, initializer=initial, config=settings)
