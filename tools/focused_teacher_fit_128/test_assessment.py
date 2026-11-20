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
    actual = a._choices(selected["sums_wdl"], selected["unknown_counts"], scores, selected["choices"])
    independent = a._independent_choices(seal, {"offered_group_scores": scores})
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
        actual = a._choices(selected["sums_wdl"], selected["unknown_counts"], scores, selected["choices"])
        assert actual == a._independent_choices(seal, {"offered_group_scores": scores})
        assert actual["focused_outsider"] == actual["focused_supported"] == expected
        assert actual["teacher"] == actual["teacher_outsider"] == 3


def test_old_ambiguity_holds_and_fresh_unknowns_are_bounded():
    seal, selected = old_seal(5, [0, 0, 0, 0, 1])
    chosen = a._choices(selected["sums_wdl"], selected["unknown_counts"], [0, 0, 0, 1, 2], selected["choices"])
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
    chosen = a._choices(selected["sums_wdl"], selected["unknown_counts"], [0, 0, 0, 0, 1], selected["choices"])
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
    assert checked["bootstrap_samples"] == 50000 and checked["bootstrap_seed"] == 2233000000
    assert checked["resampling_unit"] == "whole_source_pair_four_roots"
    assert checked["groups"]["train"]["contrasts"]["teacher_outsider_minus_focused_outsider"]["ci95"] == [0., 0.]
    assert checked["groups"]["train"]["contrasts"]["focused_outsider_minus_anchor_outsider"]["mean"] is None
    assert checked["assessment_reused_development"] and not checked["promotion"]


@pytest.mark.parametrize("damage", ["missing", "split", "fit", "phase", "leg", "scale", "unknown_arm"])
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
    else:
        rows[0]["delta_bounds"]["invented"] = [0., 0.]
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
    monkeypatch.setattr(source, "load_rows", lambda binding, split, base: [r for r in rows if r["split"] == split])
    monkeypatch.setattr(source, "old_inputs", lambda binding, base: inputs)
    monkeypatch.setattr(source, "evaluation_sources", lambda binding, base: descriptors)
    # Population bootstrap is separately exercised on all 336 synthetic rows above.
    monkeypatch.setattr(st, "summarize", lambda rows: {"fixture_states": len(rows)})
    monkeypatch.setattr(a, "independent_statistics", lambda rows: {"fixture_states": len(rows)})
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
    assert len(observed) == 2 and result["new_simulations"] == 0
    assert result["rows"][0]["choices"]["teacher"] == 4  # Fresh outcomes favor zero.
    for name in ("_predict", "_choices", "_row", "_open_assessment", "_old_input"):
        monkeypatch.setattr(a, name, lambda *args, **kwargs: pytest.fail("Auditor reused collection implementation"))
    checked = reader(a.audit("fixture", result_ref, tmp_path))
    assert checked["passed"] and checked["lawful_inputs_and_private_invariance_pass"]
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


def test_actual_old_source_inputs_and_original_anchors_match_lawful_inference(monkeypatch):
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


def test_statistical_comparison_is_stricter_than_inference_tolerance():
    a.compare({"mean": .01+1e-8}, {"mean": .01})
    with pytest.raises(ValueError):
        a.compare({"mean": .01+1e-8}, {"mean": .01}, rel_tol=0., abs_tol=1e-12)
