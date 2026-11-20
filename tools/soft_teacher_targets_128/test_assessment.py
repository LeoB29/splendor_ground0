"""Fixed-final assessment firewall, independent inference, bounds and bootstrap."""
import copy
from pathlib import Path

import pytest
import torch

from . import assessment as a, statistics as st, source
from splendor_ai.alphazero.strong_policy import encode_position, save_model
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel
from tools.teacher_signal_capture_128.test_engine import worlds, outcome, proposals, private_state
from tools.teacher_signal_capture_128.test_verify import specs
from tools.teacher_signal_capture_128 import verify as original_verify


def old_seal(n, values=None):
    old_worlds = worlds([0]*n if values is None else values)
    predictions = proposals(n)
    selected = original_verify.independent_selection(old_worlds, predictions, n)
    return {"identity": {"candidates": {"keys": [str(i) for i in range(n)]}, "proposals": predictions},
            "old_selection": {"worlds": old_worlds}, "selection_lock": {"choices": selected["choices"]}}, selected


@pytest.mark.parametrize("n", range(1, 9))
def test_ties_and_all_short_supports_are_exact(n):
    seal, selected = old_seal(n)
    scores = [0.]*n
    actual = a._choices(selected["sums_wdl"], selected["unknown_counts"], scores, selected["choices"], scores, scores, scores)
    independent = a._independent_choices(seal, {"offered_group_scores": scores}, {"offered_group_scores": scores}, {"offered_group_scores": scores}, {"offered_group_scores": scores})
    assert actual == independent
    assert actual["focused_outsider"] == (3 if n > 3 else None)
    assert actual["focused_supported"] == 0
    sp = specs()[0]
    row = a._row(sp, actual, worlds([0]*n))
    assert row == a._independent_row(sp, independent, worlds([0]*n))
    assert all(value == [0., 0.] for value in row["delta_bounds"].values())
    if n <= 3:
        assert row["arm_scores"]["focused_outsider"] is None


def test_old_selector_and_model_ties_use_original_order():
    seal, selected = old_seal(6, [0, 0, 0, 1, 1, 0])
    for scores, expected in (([0, 0, 0, 4, 4, 3], 3), ([0, 0, 0, 4, 5, 3], 4)):
        actual = a._choices(selected["sums_wdl"], selected["unknown_counts"], scores, selected["choices"], scores, scores, scores)
        assert actual == a._independent_choices(seal, {"offered_group_scores": scores}, {"offered_group_scores": scores}, {"offered_group_scores": scores}, {"offered_group_scores": scores})
        assert actual["focused_outsider"] == actual["focused_supported"] == expected
        assert actual["teacher"] == actual["teacher_outsider"] == 3


def test_old_ambiguity_holds_and_fresh_unknowns_are_bounded():
    seal, selected = old_seal(5, [0, 0, 0, 0, 1])
    chosen = a._choices(selected["sums_wdl"], selected["unknown_counts"], [0, 0, 0, 1, 2], selected["choices"], [0, 0, 0, 2, 1], [0, 0, 0, 2, 1], [0, 0, 0, 2, 1])
    fresh = worlds([0]*5)
    fresh[0]["outcomes"][4] = outcome(None)
    row = a._row(specs()[0], chosen, fresh)
    assert row == a._independent_row(specs()[0], chosen, fresh)
    assert row["delta_bounds"]["focused_outsider_minus_anchor_outsider"] == [-1/256, 1/256]
    assert row["delta_bounds"]["teacher_outsider_minus_focused_outsider"] == [0., 0.]
    assert row["arm_scores"]["focused_outsider"] is None
    with pytest.raises(ValueError):
        a._best([0, 0], [1, 0], [0, 1])
    with pytest.raises(ValueError):
        a._independent_best([0, 0], [1, 0], [0, 1])
    fresh[0]["outcomes"][4]["wdl"] = 0
    with pytest.raises(ValueError):
        a._columns(fresh, 5)
    with pytest.raises(ValueError):
        a._independent_columns(fresh, 5)


def population_rows():
    seal, selected = old_seal(5, [0, 0, 0, 0, 1])
    chosen = a._choices(selected["sums_wdl"], selected["unknown_counts"], [0, 0, 0, 0, 1], selected["choices"], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0])
    variants = []
    for j in range(7):
        fresh = worlds([0]*5)
        for i in range(j):
            fresh[i]["outcomes"][4] = outcome(1)
        if j == 6:
            fresh[-1]["outcomes"][4] = outcome(None)
        variants.append(fresh)
    return [a._row(spec, chosen, variants[spec["pair"] % 7]) for spec in specs()]


def test_whole_pair_shared_bootstrap_independent_histogram_and_unknown_bounds(monkeypatch):
    rows = population_rows()
    expected = st.summarize(rows)
    monkeypatch.setattr(st, "_arrays", lambda *args: pytest.fail("Auditor reused collection bootstrap"))
    monkeypatch.setattr(a, "_row", lambda *args: pytest.fail("Auditor reused arithmetic"))
    checked = a.independent_statistics(rows)
    a.compare(expected, checked, rel_tol=0., abs_tol=1e-12)
    assert checked["bootstrap_samples"] == 50000 and checked["bootstrap_seed"] == a.c.CONFIG["bootstrap_seed"]
    assert checked["resampling_unit"] == "whole_source_pair_four_roots"
    assert checked["groups"]["train"]["contrasts"]["teacher_outsider_minus_focused_outsider"]["ci95"] == [0., 0.]
    assert checked["groups"]["train"]["contrasts"]["focused_outsider_minus_anchor_outsider"]["mean"] is None
    assert checked["assessment_reused_development"] and not checked["promotion"]


@pytest.mark.parametrize("damage", ["missing", "split", "fit", "phase", "leg", "scale", "unknown_arm",
                                    "escaping_specialist", "wrong_outsider"])
def test_bootstrap_refuses_changed_population(damage):
    rows = population_rows()
    if damage == "missing":
        rows.pop()
    elif damage == "split":
        rows[0]["split"] = "validation"
    elif damage == "fit":
        rows[0]["fit"] = 161003
    elif damage == "phase":
        rows[0]["phase"] = rows[1]["phase"]
    elif damage == "leg":
        rows[0]["game_index"] += 4
    elif damage == "scale":
        rows[0]["delta_bounds"][st.NAMES[0]] = [.001, .001]
    elif damage == "unknown_arm":
        rows[0]["delta_bounds"]["invented"] = [0., 0.]
    elif damage == "escaping_specialist":
        rows[0]["choices"]["specialist_supported"] = 4
    else:
        rows[0]["choices"]["specialist_outsider"] = 2
    with pytest.raises((ValueError, AssertionError)):
        st.summarize(rows)
    with pytest.raises((ValueError, AssertionError)):
        a.independent_statistics(rows)


def fixture_panel(tmp_path, monkeypatch):
    torch.manual_seed(932)
    torch.set_num_threads(1)
    config = LegalActionAttentionScorerConfig(state_hidden_size=8, action_hidden_size=8,
        scorer_hidden_size=8, state_depth=1, action_depth=1, attention_layers=1,
        attention_heads=2, attention_feedforward_size=16)
    model = LegalActionAttentionScorerModel(config)
    model.eval()
    registry = {"manifest": "fixture", "models": {}, "fits": [], "audits": [],
                "fixed_final": True, "all_fit_audits_pass": True}
    for fit in a.c.FITS:
        checkpoint = tmp_path/f"model_{fit}.pt"
        save_model(checkpoint, model, 1784, {"fit": fit, "manifest": "fixture"})
        ref = a.c.artifact(checkpoint)
        registry["models"][f"focused_{fit}"] = {**ref, "source_fit": fit, "selected_update": 1784}
        path = tmp_path/f"fit_{fit}.json"
        a.old.save_new(path, {"manifest": "fixture", "fit": fit, "selected_update": 1784,
                              "selected_model": ref, "final_always_selected": True})
        fit_ref = a.c.artifact(path)
        registry["fits"].append(fit_ref)
        path = tmp_path/f"fit_audit_{fit}.json"
        a.old.save_new(path, {"manifest": "fixture", "result": fit_ref, "passed": True,
            "complete_fit_bitwise_identical": True, "all_training_metrics_independently_reproduced": True,
            "value_head_frozen": True, "selected_model": ref})
        registry["audits"].append(a.c.artifact(path))
    a.old.save_new(tmp_path/"models_lock.json", registry)
    state = private_state()
    encoded, canonical_keys, legal = encode_position(state, config)
    representatives = a.old.actions_by_canonical_successor_key(state, legal)
    rows, inputs, descriptors = [], [], []
    for index, n in enumerate((8, 3)):
        keys = [repr(k) for k in canonical_keys[:n]]
        candidates = {"keys": keys, "actions": [a.old.serialize_action(representatives[k]) for k in canonical_keys[:n]]}
        row = {**encoded, "id": f"fixture:{index}", "index": index, "source_fit": 161001+index,
               "split": "train" if index == 0 else "validation",
               "members": [torch.where(encoded["groups"] == j)[0] for j in range(n)]}
        spec = {"index": index, "fit": row["source_fit"], "source_rowid": row["id"], "split": row["split"],
                "pair": index, "game_index": 2*index, "seat": 0, "phase": 1,
                "privacy": {"own": 1, "opponent": 1}}
        row.update(pair=index, old_sums_wdl=[0, 0, 0, 0, 128, -128, 0, 0][:n],
                   outsider_start=min(3, n), outsider_support=max(0, n-3),
                   outsider_teacher=4 if n>3 else -1, outsider_label=1 if n>3 else -1)
        snapshot = a.old.serialize_state_snapshot(state)
        old_worlds = worlds(([0, 0, 0, 0, 1, -1, 0, 0])[:n])
        predicted = proposals(n)
        selected = original_verify.independent_selection(old_worlds, predicted, n)
        identity = {"spec": spec, "snapshot": snapshot, "candidates": candidates, "proposals": predicted}
        block = {"worlds": old_worlds}
        lock = {"choices": selected["choices"], "identity_digest": a.old.digest(identity),
                "old_selection_sha256": a.old.digest(block), "selection_statistics": selected}
        path = tmp_path/f"selection_{index}.json"
        a.old.save_new(path, {"manifest": a.c.SOURCE_BINDING, "stage": "main", "index": index,
                             "identity": identity, "selection_lock": lock, "old_selection": block})
        selection_ref = a.c.artifact(path)
        item = {"index": index, "spec": spec, "snapshot": snapshot, "candidates": candidates,
                "old_sums_wdl": selected["sums_wdl"], "old_choices": selected["choices"], "selection_artifact": selection_ref}
        fresh = worlds([1]+[0]*(n-1))
        for j, world in enumerate(fresh):
            world["seed"] = 2211000000+index*512+j
        root = {"manifest": a.c.SOURCE_BINDING, "stage": "main", "spec": spec,
                "candidates": candidates, "selection_lock": lock, "selection_artifact": selection_ref,
                "assessment": {"candidate_indices": list(range(n)), "world_seeds": [w["seed"] for w in fresh], "worlds": fresh}}
        root["summary"] = original_verify.independent_summary(fresh, selected["choices"])
        path = tmp_path/f"source_root_{index}.json"
        a.old.save_new(path, root)
        root_ref = a.c.artifact(path)
        path = tmp_path/f"source_audit_{index}.json"
        a.old.save_new(path, {"manifest": a.c.SOURCE_BINDING, "root": root_ref, "passed": True,
                             "row": {"delta_bounds": root["summary"]["delta_bounds"]}})
        descriptors.append({"index": index, "spec": spec, "root": root_ref,
                            "audit": a.c.artifact(path), "selection": selection_ref})
        rows.append(row)
        inputs.append(item)
    monkeypatch.setattr(source, "load_panel_rows", lambda binding, base: rows, raising=False)
    monkeypatch.setattr(source, "control_models", lambda binding, base: {
        f"{family}_{fit}": registry["models"][f"focused_{fit}"]
        for family in ("anchor", "specialist", "lowrate", "highrate") for fit in a.c.FITS}, raising=False)
    monkeypatch.setattr(source, "load_rows", lambda binding, split, base: [r for r in rows if r["split"] == split])
    monkeypatch.setattr(source, "old_inputs", lambda binding, base: inputs)
    monkeypatch.setattr(source, "evaluation_sources", lambda binding, base: descriptors)
    cached_parents = {}
    def rate_evidence(binding, base, family):
        if family not in cached_parents:
            cached_parent = cached_parents[family] = {}
            from tools.full_teacher_fit_128 import statistics as low_stats
            from tools.higher_teacher_rate_128 import statistics as high_stats
            prior_stats = low_stats if family == "lowrate" else high_stats
            locked = a.old.read_json(tmp_path/"predictions_lock.json")
            validation = a.c.verified(locked["validation_old_targets"])
            previous_rows, previous_predictions = [], []
            for prediction, descriptor in zip(locked["predictions"], descriptors):
                choices = copy.deepcopy(prediction["choices"])
                for role in ("outsider", "supported"):
                    choices[f"focused_{role}"] = choices[f"{family}_{role}"]
                root = a.c.verified(descriptor["root"])
                previous = a._row(descriptor["spec"], choices, root["assessment"]["worlds"])
                for field in ("choices", "arm_scores", "arm_score_bounds"):
                    previous[field] = {k:v for k,v in previous[field].items() if k in prior_stats.ARMS}
                previous["delta_bounds"] = {k:v for k,v in previous["delta_bounds"].items() if k in prior_stats.NAMES}
                previous_rows.append(previous)
                item = {k:copy.deepcopy(prediction[k]) for k in ("index", "fit", "row_id", "spec", "source_selection")}
                item.update(model=prediction[family+"_model"], proposal=prediction[family+"_proposal"], choices=previous["choices"])
                previous_predictions.append(item)
            prior_validation = copy.deepcopy(validation)
            prior_validation["manifest"] = family+"_fixture"
            prior_validation["families"]["focused"] = copy.deepcopy(prior_validation["families"][family])
            prior_validation["families"] = {k:v for k,v in prior_validation["families"].items()
                if k in (("anchor", "focused", "specialist") if family == "lowrate" else ("anchor", "focused", "specialist", "lowrate"))}
            # Frozen controls have only the legacy hard-CE diagnostic schema.
            record_keys = {"id","pair","index","source_fit","support","outsider_support","choice","target",
                           "first_winner","any_old_max","old_regret_units","loss"}
            metric_keys = {"rows","accuracy_denominator","mean_loss","first_winner_count","any_old_max_count",
                           "first_winner_accuracy","any_old_max_accuracy","old_regret_units_sum","mean_old_score_regret"}
            for report in prior_validation["families"].values():
                report["rows"] = [{k:v for k,v in r.items() if k in record_keys} for r in report["rows"]]
                report["metrics"] = {group:{k:v for k,v in m.items() if k in metric_keys} for group,m in report["metrics"].items()}
                report["pair_weighted"] = {k:v for k,v in report["pair_weighted"].items() if k in ("pairs","mean_loss","mean_old_score_regret")}
            path = tmp_path/f"{family}_prior_validation.json"
            a.old.save_new(path, prior_validation)
            cached_parent["validation_old_targets"] = a.c.artifact(path)
            path = tmp_path/f"{family}_prior_predictions.json"
            a.old.save_new(path, {"manifest": family+"_fixture", "predictions": previous_predictions})
            cached_parent["predictions"] = a.c.artifact(path)
            cached_parent["models"] = a.c.artifact(tmp_path/"models_lock.json")
            groups = {}
            for split in ("train", "validation"):
                selected = [row for row in previous_rows if row["split"] == split]
                groups[split] = {"contrasts": {"focused_"+suffix: {"mean_bounds": [
                    sum(row["delta_bounds"]["focused_"+suffix][j] for row in selected)/len(selected) for j in (0,1)]}
                    for suffix in ("supported_minus_anchor", "outsider_minus_anchor_outsider")}}
            path = tmp_path/f"{family}_prior_assessment.json"
            a.old.save_new(path, {"manifest": family+"_fixture", "rows": previous_rows, "statistics": {"groups": groups},
                "models": cached_parent["models"], "predictions": cached_parent["predictions"],
                "validation_old_targets": cached_parent["validation_old_targets"]})
            cached_parent["assessment"] = a.c.artifact(path)
        return cached_parents[family]
    monkeypatch.setattr(source, "lowrate_evidence", lambda binding,base:rate_evidence(binding,base,"lowrate"), raising=False)
    monkeypatch.setattr(source, "highrate_evidence", lambda binding,base:rate_evidence(binding,base,"highrate"), raising=False)
    # Population bootstrap is separately exercised on all 336 synthetic rows above.
    monkeypatch.setattr(st, "summarize", lambda rows: {"fixture_states": len(rows), "groups": {"validation": {"contrasts": {name: {"mean_bounds": [0., 0.], "ci95": [0., 0.]} for name in st.PRIMARY}}}})
    monkeypatch.setattr(a, "independent_statistics", lambda rows: {"fixture_states": len(rows), "groups": {"validation": {"contrasts": {name: {"mean_bounds": [0., 0.], "ci95": [0., 0.]} for name in st.PRIMARY}}}})
    return rows, inputs, descriptors


def test_all_predictions_sealed_before_assessment_and_complete_independent_audit(tmp_path, monkeypatch):
    rows, inputs, descriptors = fixture_panel(tmp_path, monkeypatch)
    reader, observed = a.c.verified, []
    fresh_paths = {value["root"]["path"] for value in descriptors}
    def guarded_read(ref):
        if ref["path"] in fresh_paths:
            sealed = a.old.read_json(tmp_path/"predictions_lock.json")
            assert sealed["complete"] and len(sealed["predictions"]) == 2
            observed.append(ref["path"])
        return reader(ref)
    monkeypatch.setattr(a.c, "verified", guarded_read)
    result_ref = a.assess("fixture", tmp_path)
    result = reader(result_ref)
    assert len(observed) == 6 and result["new_simulations"] == 0
    assert result["rows"][0]["choices"]["teacher"] == 4  # Fresh outcomes favor zero.
    for name in ("_predict", "_choices", "_row", "_open_assessment", "_old_input"):
        monkeypatch.setattr(a, name, lambda *args, **kwargs: pytest.fail("Auditor reused collection implementation"))
    checked = reader(a.audit("fixture", result_ref, tmp_path))
    assert checked["passed"] and checked["lawful_inputs_and_private_invariance_pass"]
    assert checked["highrate_inference_independently_reproduced"] and checked["highrate_control_identity_pass"]
    assert checked["new_trajectory_replays"] == 0 and checked["unknown_outcomes"] == 0
    with pytest.raises(ValueError, match="Preserve"):
        a.assess("fixture", tmp_path)


@pytest.mark.parametrize("damage", ["missing_seal", "unaudited", "wrong_fit", "early_checkpoint", "corrupt_model"])
def test_fixed_final_model_seal_required(tmp_path, monkeypatch, damage):
    fixture_panel(tmp_path, monkeypatch)
    path = tmp_path/"models_lock.json"
    registry = a.old.read_json(path)
    if damage == "missing_seal":
        path.rename(tmp_path/"preserved_models_lock.json")
    else:
        if damage == "unaudited":
            registry["all_fit_audits_pass"] = False
        elif damage == "wrong_fit":
            registry["models"]["focused_161001"]["source_fit"] = 161002
        elif damage == "early_checkpoint":
            registry["models"]["focused_161001"]["selected_update"] = 1024
        else:
            registry["models"]["focused_161001"]["sha256"] = "0"*64
        path.write_text(__import__("json").dumps(registry))
    with pytest.raises((ValueError, FileNotFoundError)):
        a.assess("fixture", tmp_path)
    assert not (tmp_path/"predictions_lock.json").exists()


@pytest.mark.parametrize("damage", ["fit", "old_sum", "offered_mapping", "early_access"])
def test_old_inputs_and_outcome_firewall_reject_tamper(tmp_path, monkeypatch, damage):
    rows, inputs, descriptors = fixture_panel(tmp_path, monkeypatch)
    if damage == "fit":
        rows[0]["source_fit"] = 161003
    elif damage == "old_sum":
        inputs[0]["old_sums_wdl"][0] += 1
    elif damage == "offered_mapping":
        # Collection is later caught by independent lawful encoding, even when old hash is intact.
        rows[0]["members"][0] = rows[0]["members"][1]
        ref = a.assess("fixture", tmp_path)
        with pytest.raises(ValueError, match="membership"):
            a.audit("fixture", ref, tmp_path)
        return
    else:
        with pytest.raises(ValueError, match="sealed"):
            a._open_assessment(descriptors[0], {}, {"path": "absent", "sha256": "none"}, tmp_path)
        return
    with pytest.raises(ValueError):
        a.assess("fixture", tmp_path)


def test_independent_lawful_encoder_private_invariance_and_group_max(tmp_path, monkeypatch):
    rows, inputs, _ = fixture_panel(tmp_path, monkeypatch)
    _, _, models = a._models("fixture", tmp_path)
    for row, item in zip(rows, inputs):
        model = models[row["source_fit"]]
        expected = a._predict(row, model)
        actual = a._independent_proposal(row, item, model)
        a.compare(expected, actual)
        state = a.old.deserialize_state_snapshot(item["snapshot"])
        alternate = a.input_audit.hidden_alternative(state)
        a.compare(actual, a._independent_proposal(row, item, model, alternate))
    assert inputs[0]["spec"]["privacy"]["opponent"] > 0


def test_actual_old_source_inputs_and_original_anchor_specialist_match_lawful_inference(monkeypatch):
    """One previously audited row per fit; no current assessment is opened."""
    freeze = a.old.read_json(a.c.SOURCE/"freeze.json")
    manifest = a.c.verified(freeze["manifest"])
    assert a.old.digest(manifest) == freeze["binding"] == a.c.SOURCE_BINDING
    schedule = a.c.verified(manifest["schedule"])["main"]
    reader = a.c.verified
    def old_only(ref):
        path = str(ref["path"]).replace("\\", "/")
        assert "teacher_signal_capture_128/main/roots/" not in path
        assert "teacher_signal_capture_128/result.json" not in path
        return reader(ref)
    monkeypatch.setattr(a.c, "verified", old_only)
    monkeypatch.setattr(a.old, "trajectory", lambda *args, **kwargs: pytest.fail("No new simulation in input integration"))
    specialist = old_only(a.c.artifact(source.SPECIALIST/"models_lock.json"))
    assert specialist["manifest"] == source.SPECIALIST_BINDING
    lowrate = old_only(a.c.artifact(source.LOWRATE/"models_lock.json"))
    assert lowrate["manifest"] == source.LOWRATE_BINDING
    highrate = old_only(a.c.artifact(source.HIGHRATE/"models_lock.json"))
    assert highrate["manifest"] == source.HIGHRATE_BINDING
    for fit in a.c.FITS:
        spec = next(row for row in schedule if row["fit"] == fit and row["split"] == "train")
        original = old_only(spec["source_root"])
        checkpoint = manifest["model_parent"]["models"][f"anchor_{fit}"]
        model = a.StrongPolicyInference(a.c.verify_hash(checkpoint)).model
        data = torch.load(a.c.verify_hash(spec["source_export"]), map_location="cpu", weights_only=True)
        row = next(value for value in data if value["id"] == spec["source_rowid"])
        item = {"snapshot": original["snapshot"], "candidates": original["candidates"]}
        predicted = a._predict(row, model)
        independently = a._independent_proposal(row, item, model)
        a.compare(predicted, independently)
        state = a.old.deserialize_state_snapshot(item["snapshot"])
        alternate = a.input_audit.hidden_alternative(state)
        a.compare(independently, a._independent_proposal(row, item, model, alternate))
        ref = specialist["models"][f"specialist_{fit}"]
        assert ref["source_fit"] == fit and ref["sha256"] == source.SPECIALIST_MODELS[fit]
        control = a.StrongPolicyInference(a.c.verify_hash(ref)).model
        a.compare(a._predict(row, control), a._independent_proposal(row, item, control))
        a.compare(a._independent_proposal(row, item, control), a._independent_proposal(row, item, control, alternate))
        ref = lowrate["models"][f"focused_{fit}"]
        assert ref["source_fit"] == fit and ref["sha256"] == source.LOWRATE_MODELS[fit]
        control = a.StrongPolicyInference(a.c.verify_hash(ref)).model
        a.compare(a._predict(row, control), a._independent_proposal(row, item, control))
        a.compare(a._independent_proposal(row, item, control), a._independent_proposal(row, item, control, alternate))
        ref = highrate["models"][f"focused_{fit}"]
        assert ref["source_fit"] == fit and ref["sha256"] == source.HIGHRATE_MODELS[fit]
        control = a.StrongPolicyInference(a.c.verify_hash(ref)).model
        a.compare(a._predict(row, control), a._independent_proposal(row, item, control))
        a.compare(a._independent_proposal(row, item, control), a._independent_proposal(row, item, control, alternate))


def test_statistical_comparison_is_stricter_than_inference_tolerance():
    a.compare({"mean": .01+1e-8}, {"mean": .01})
    with pytest.raises(ValueError):
        a.compare({"mean": .01+1e-8}, {"mean": .01}, rel_tol=0., abs_tol=1e-12)


def test_specialist_is_a_separate_fixed_candidate_control():
    seal, selected = old_seal(6, [0, 0, 0, 0, 1, -1])
    focused, specialist = [0, 0, 0, 0, 4, 1], [0, 0, 0, 2, 1, 4]
    chosen = a._choices(selected["sums_wdl"], selected["unknown_counts"], focused, selected["choices"], specialist, specialist, specialist)
    checked = a._independent_choices(seal, {"offered_group_scores": focused}, {"offered_group_scores": specialist}, {"offered_group_scores": specialist}, {"offered_group_scores": specialist})
    assert chosen == checked
    assert chosen["focused_outsider"] == chosen["focused_supported"] == 4
    assert chosen["specialist_outsider"] == 5 and chosen["specialist_supported"] == 0
    row = a._row(specs()[0], chosen, worlds([0, 0, 0, 0, 1, -1]))
    assert row == a._independent_row(specs()[0], checked, worlds([0, 0, 0, 0, 1, -1]))
    assert row["delta_bounds"]["focused_outsider_minus_specialist_outsider"] == [1., 1.]
    assert row["delta_bounds"]["focused_supported_minus_specialist_supported"] == [.5, .5]


@pytest.mark.parametrize("family", ["specialist", "lowrate", "highrate"])
def test_wrong_control_source_fit_is_rejected_before_predictions(tmp_path, monkeypatch, family):
    fixture_panel(tmp_path, monkeypatch)
    controls = copy.deepcopy(source.control_models("fixture", tmp_path))
    controls[f"{family}_161001"]["source_fit"] = 161002
    monkeypatch.setattr(source, "control_models", lambda *args: controls)
    with pytest.raises(ValueError, match="control fit"):
        a.assess("fixture", tmp_path)
    assert not (tmp_path/"predictions_lock.json").exists()


def test_full_validation_is_separate_from_panel_and_independently_reproduced(tmp_path, monkeypatch):
    from . import learning
    panel, _, descriptors = fixture_panel(tmp_path, monkeypatch)
    validation = []
    for i in range(5):
        row = copy.deepcopy(panel[0])
        row.update(id=f"val:{i}", index=100+i, source_fit=a.c.FITS[i % 3], split="validation", pair=f"pair:{i//2}")
        validation.append(row)
    def load_validation(binding, split, base):
        assert split == "validation" and (base/"models_lock.json").exists()
        return validation
    monkeypatch.setattr(source, "load_rows", load_validation)
    reader = a.c.verified
    fresh = {item["root"]["path"] for item in descriptors}
    def guarded(ref):
        if ref["path"] in fresh:
            lock = a.old.read_json(tmp_path/"predictions_lock.json")
            assert reader(lock["validation_old_targets"])["states"] == 5
            assert set(lock["controls"]) == {f"{family}_{fit}" for family in ("anchor", "specialist", "lowrate", "highrate") for fit in a.c.FITS}
            assert all("specialist_proposal" in row and "specialist_model" in row
                       and "highrate_proposal" in row and "highrate_model" in row for row in lock["predictions"])
        return reader(ref)
    monkeypatch.setattr(a.c, "verified", guarded)
    ref = a.assess("fixture", tmp_path)
    result = reader(ref)
    report = reader(result["validation_old_targets"])
    assert len(result["rows"]) == 2 and report["states"] == 5
    for family in ("anchor", "focused", "specialist", "lowrate", "highrate"):
        item = report["families"][family]
        assert item["metrics"]["multiple_outsiders"]["rows"] == 5
        assert item["pair_weighted"]["pairs"] == 3
        assert [row["id"] for row in item["rows"]] == [f"val:{i}" for i in range(5)]
        assert sum(model["states"] for model in item["models"].values()) == 5
    monkeypatch.setattr(learning, "objective", lambda *args: pytest.fail("Auditor reused collection objective"))
    monkeypatch.setattr(learning, "collate", lambda *args: pytest.fail("Auditor reused collection tensorizer"))
    audit = reader(a.audit("fixture", ref, tmp_path))
    assert audit["full_validation_old_target_metrics_independently_reproduced"]
    assert audit["specialist_inference_independently_reproduced"]


@pytest.mark.parametrize("mean,lower,control_lower,expected", [
    (.005,.00001,.00001,True),(.005,0.,.001,False),(.0049,.001,.001,False),
    (.01,.001,0.,False),(.01,.001,-.001,False),(-.001,-.002,.001,False)])
def test_anchor_and_lowrate_both_drive_prespecified_development_screen(mean, lower, control_lower, expected):
    values = {"groups": {"validation": {"contrasts": {
        "focused_supported_minus_anchor": {"mean_bounds": [mean,mean], "ci95": [lower,.1]},
        "focused_supported_minus_lowrate_supported": {"mean_bounds": [.003,.003], "ci95": [control_lower,.1]},
        "focused_supported_minus_highrate_supported": {"mean_bounds": [-.2,-.2], "ci95": [-.3,-.1]},
        "focused_supported_minus_specialist_supported": {"mean_bounds": [-.2,-.2], "ci95": [-.3,-.1]}}}}}
    screened = st.development_screen(values)
    assert screened["passed"] == expected and not screened["specialist_attribution_is_additional_gate"]
    assert not screened["highrate_attribution_is_additional_gate"]
    assert screened["control_lower_strictly_positive"] == (control_lower > 0)
    assert screened["required_control_endpoint"] == "focused_supported_minus_lowrate_supported"
    assert not screened["promotion"] and not screened["automatic_successor"]
    assert screened["fresh_full_legal_confirmation_required"]
    assert st.PRIMARY == ("focused_supported_minus_anchor", "focused_supported_minus_lowrate_supported")


def test_lowrate_is_separate_from_specialist_and_new_policy():
    seal, selected = old_seal(6, [0, 0, 0, 0, 1, -1])
    vectors = [[0,0,0,1,4,2], [0,0,0,1,2,4], [0,0,0,4,2,1], [0,0,0,1,2,4]]
    choices = a._choices(selected["sums_wdl"], selected["unknown_counts"], vectors[0], selected["choices"], *vectors[1:])
    checked = a._independent_choices(seal, *[{"offered_group_scores": values} for values in vectors])
    assert choices == checked
    assert [choices[f"{family}_outsider"] for family in ("focused", "specialist", "lowrate")] == [4,5,3]
    row = a._row(specs()[0], choices, worlds([0,0,0,0,1,-1]))
    assert row == a._independent_row(specs()[0], checked, worlds([0,0,0,0,1,-1]))
    assert row["delta_bounds"]["focused_outsider_minus_lowrate_outsider"] == [.5,.5]
    assert row["delta_bounds"]["focused_supported_minus_lowrate_supported"] == [.5,.5]


@pytest.mark.parametrize("family", ["lowrate", "highrate"])
@pytest.mark.parametrize("damage", ["choice", "score", "mean", "validation", "logits", "model"])
def test_rate_parent_identity_rejects_changed_prior_evidence(tmp_path, monkeypatch, damage, family):
    import json
    fixture_panel(tmp_path, monkeypatch)
    result = a.c.verified(a.assess("fixture", tmp_path))
    lock = a.c.verified(result["predictions"])
    validation = a.c.verified(result["validation_old_targets"])
    refs = getattr(source, family+"_evidence")("fixture", tmp_path)
    field = "predictions" if damage in ("choice", "logits", "model") else "validation_old_targets" if damage == "validation" else "assessment"
    value = a.c.verified(refs[field])
    if damage == "choice":
        value["predictions"][0]["choices"]["focused_outsider"] = 7
    elif damage == "logits":
        value["predictions"][0]["proposal"]["legal_logits"][0] += .2
    elif damage == "model":
        value["predictions"][0]["model"]["sha256"] = "0"*64
    elif damage == "score":
        value["rows"][0]["arm_score_bounds"]["focused_supported"][0] += 1/256
    elif damage == "mean":
        value["statistics"]["groups"]["train"]["contrasts"]["focused_supported_minus_anchor"]["mean_bounds"][0] += 1/256
    else:
        value["families"]["focused"]["metrics"]["all_rows"]["mean_loss"] += .2
    path = a.c.verify_hash(refs[field])
    path.write_text(json.dumps(value))
    refs[field] = a.c.artifact(path)
    if field != "assessment":
        previous = a.c.verified(refs["assessment"])
        previous[field] = refs[field]
        path = a.c.verify_hash(refs["assessment"])
        path.write_text(json.dumps(previous))
        refs["assessment"] = a.c.artifact(path)
    with pytest.raises(ValueError):
        getattr(a, "_"+family+"_identity")("fixture", lock["predictions"], result["rows"], validation, tmp_path)


@pytest.mark.parametrize("family", ["lowrate", "highrate"])
def test_rate_score_identity_cannot_open_before_global_seal(tmp_path, monkeypatch, family):
    fixture_panel(tmp_path, monkeypatch)
    monkeypatch.setattr(source, family+"_evidence", lambda *args: pytest.fail("Prior scores opened before global seal"))
    with pytest.raises(FileNotFoundError):
        getattr(a, "_"+family+"_identity")("fixture", [], [], {}, tmp_path)


def test_highrate_is_distinct_and_uses_shared_action_cancellation():
    seal, selected = old_seal(7, [0,0,0,0,1,-1,0])
    vectors = [[0,0,0,0,5,1,2], [0,0,0,1,0,5,2], [0,0,0,5,1,2,0], [0,0,0,1,2,0,5]]
    chosen = a._choices(selected["sums_wdl"], selected["unknown_counts"], vectors[0], selected["choices"], *vectors[1:])
    independent = a._independent_choices(seal, *[{"offered_group_scores": v} for v in vectors])
    assert chosen == independent
    assert [chosen[f"{family}_outsider"] for family in ("focused","specialist","lowrate","highrate")] == [4,5,3,6]
    fresh = worlds([0,0,0,0,1,-1,0])
    fresh[0]["outcomes"][6] = outcome(None)
    row = a._row(specs()[0], chosen, fresh)
    assert row == a._independent_row(specs()[0], independent, fresh)
    assert row["delta_bounds"]["focused_outsider_minus_highrate_outsider"] == [127/256,129/256]
    assert row["delta_bounds"]["highrate_supported_minus_anchor"] == [0.,0.]


def test_soft_validation_preserves_frozen_hard_metrics_and_adds_independent_soft_diagnostics(tmp_path, monkeypatch):
    from tools.higher_teacher_rate_128 import learning as historical
    from . import learning
    panel, _, _ = fixture_panel(tmp_path, monkeypatch)
    rows = []
    for i in range(5):
        row = copy.deepcopy(panel[0])
        row.update(id=f"val:{i}", index=900+i, split="validation", source_fit=a.c.FITS[i%3], pair=i//2)
        # Deliberately nontrivial graded targets and exact ties.
        row["old_sums_wdl"] = [0,0,0,3,5,5,-5,1]
        row["outsider_teacher"], row["outsider_label"] = 4,1
        rows.append(row)
    monkeypatch.setattr(source, "load_rows", lambda *args:rows)
    _, registry, models = a._models("fixture", tmp_path)
    refs, controls = a._control_models("fixture", tmp_path)
    current = a._validation_metrics("fixture", registry, models, refs, controls, tmp_path)
    independent = a._validation_metrics("fixture", registry, models, refs, controls, tmp_path, audit=True)
    a.compare(current, independent, rel_tol=0., abs_tol=1e-12)
    for family in ("anchor","focused","specialist","lowrate","highrate"):
        actual = current["families"][family]
        expected_rows = []
        for fit in a.c.FITS:
            selected = [r for r in rows if r["source_fit"] == fit]
            model = models[fit] if family == "focused" else controls[f"{family}_{fit}"]
            old = historical.training_report(model, selected, tmp_path, historical.c.CONFIG,
                data_scope="validation", diagnostic_scope="full_validation_matched_source_fit")
            expected_rows.extend(old["rows"])
        expected_rows.sort(key=lambda r:r["id"])
        a.compare(a._legacy_projection(actual["rows"], expected_rows), expected_rows, rel_tol=0., abs_tol=1e-12)
        expected_metrics = historical._metrics(expected_rows)
        a.compare(a._legacy_projection(actual["metrics"], expected_metrics), expected_metrics, rel_tol=0., abs_tol=1e-12)
        metrics = actual["metrics"]["all_rows"]
        assert abs(metrics["mean_soft_loss"] - metrics["mean_loss"]) > 1e-5
        assert abs(metrics["mean_target_kl"] - (metrics["mean_soft_loss"]-metrics["mean_target_entropy"])) < 1e-12
        assert actual["pair_weighted"]["pairs"] == 3
        assert "mean_soft_loss" in actual["pair_weighted"]
    assert current["mean_loss_definition"] == "legacy_hard_first_winner_cross_entropy"
    assert current["target_temperature"] == .05


def test_auditor_rejects_control_gate_interval_drift_below_inference_tolerance(tmp_path, monkeypatch):
    import json
    fixture_panel(tmp_path, monkeypatch)
    ref = a.assess("fixture", tmp_path)
    result = a.c.verified(ref)
    result["development_screen"]["control_ci95"][0] += 1e-8
    path = a.c.verify_hash(ref)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        a.audit("fixture", a.c.artifact(path), tmp_path)
