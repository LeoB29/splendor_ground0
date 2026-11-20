"""Fixed-final predictions are sealed before reused outcome blocks are opened.

This module never trains, generates worlds or runs continuations. The auditor
independently encodes inputs, runs private model instances, reconstructs choices
and reduces outcomes. Historical trajectory and model evidence is inherited.
"""
from pathlib import Path
import math
import time

import numpy as np
import torch

from . import common as c
from . import statistics as st
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit
from splendor_ai.alphazero.strong_policy import StrongPolicyInference
from splendor_ai.encoding import (
    encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector,
)

old, OUTPUT = c.old, c.OUTPUT


def _same_ref(left, right):
    return all(left[k] == right[k] for k in ("path", "sha256"))


def compare(actual, expected, *, rel_tol=1e-6, abs_tol=1e-7):
    """Only inference/quantile rounding receives a small absolute tolerance."""
    if isinstance(expected, dict):
        c.require(isinstance(actual, dict) and set(actual) == set(expected), "Dictionary evidence differs")
        for key in expected:
            compare(actual[key], expected[key], rel_tol=rel_tol, abs_tol=abs_tol)
    elif isinstance(expected, (list, tuple)):
        c.require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), "Sequence evidence differs")
        for a, b in zip(actual, expected):
            compare(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    elif type(expected) is float:
        c.require(type(actual) in (int, float) and math.isfinite(actual)
                  and math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), "Numerical evidence differs")
    else:
        c.require(actual == expected, "Evidence differs")


def _models(binding, base):
    registry_ref = c.artifact(Path(base)/"models_lock.json")
    registry = c.verified(registry_ref)
    c.require(registry["manifest"] == binding and registry["fixed_final"] is True
              and registry["all_fit_audits_pass"] is True, "Unaudited model seal")
    c.require(set(registry["models"]) == {f"focused_{f}" for f in c.FITS}, "Missing focused model")
    fits = [c.verified(ref) for ref in registry["fits"]]
    audits = [c.verified(ref) for ref in registry["audits"]]
    c.require(len(fits) == len(audits) == 3 and {fit["fit"] for fit in fits} == set(c.FITS), "Incomplete fit closure")
    models = {}
    for fit in c.FITS:
        result = next(value for value in fits if value["fit"] == fit)
        fit_ref = registry["fits"][fits.index(result)]
        matching = [value for value in audits if _same_ref(value["result"], fit_ref)]
        c.require(len(matching) == 1, "Missing or duplicated fit audit")
        receipt = matching[0]
        c.require(receipt["manifest"] == binding and receipt["passed"] is True
                  and receipt["complete_fit_bitwise_identical"] is True
                  and receipt["all_training_metrics_independently_reproduced"] is True
                  and receipt["value_head_frozen"] is True, "Incomplete model reproduction")
        item = registry["models"][f"focused_{fit}"]
        c.require(item["source_fit"] == fit and item["selected_update"] == 1784
                  and result["manifest"] == binding and result["selected_update"] == 1784
                  and result["final_always_selected"] is True
                  and _same_ref(item, result["selected_model"])
                  and _same_ref(item, receipt["selected_model"]), "Wrong fixed-final model")
        model = StrongPolicyInference(c.verify_hash(item)).model
        model.eval()
        models[fit] = model
    return registry_ref, registry, models


def _panel(binding, base):
    from . import source
    rows = source.load_panel_rows(binding, base)
    rows = sorted(rows, key=lambda r: r.get("index", r.get("globalindex")))
    inputs = source.old_inputs(binding, base)
    sources = source.evaluation_sources(binding, base)
    c.require(len(rows) == len(inputs) == len(sources) > 0, "Incomplete old-only input panel")
    if Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        c.require(len(rows) == 336, "Production assessment requires every fixed position")
    for index, (row, item, descriptor) in enumerate(zip(rows, inputs, sources)):
        c.require(row.get("index", row.get("globalindex")) == item["index"] == descriptor["index"] == index,
                  "Wrong panel ordering")
        c.require(old.digest(item["spec"]) == old.digest(descriptor["spec"])
                  and row["source_fit"] == item["spec"]["fit"]
                  and row["split"] == item["spec"]["split"]
                  and row["id"] == item["spec"]["source_rowid"], "Wrong fit or source row")
    return rows, inputs, sources


def _control_models(binding, base):
    from . import source
    refs = source.control_models(binding, base)
    expected = {f"{family}_{fit}" for family in ("anchor", "specialist", "lowrate", "highrate") for fit in c.FITS}
    c.require(set(refs) == expected, "Missing frozen specialist or anchor control")
    c.require(all(refs[f"{family}_{fit}"]["source_fit"] == fit
                  for family in ("anchor", "specialist", "lowrate", "highrate") for fit in c.FITS), "Wrong control fit identity")
    models = {name: StrongPolicyInference(c.verify_hash(ref)).model for name, ref in refs.items()}
    for model in models.values():
        model.eval()
    return refs, models


def _validation_metrics(binding, registry, focused, control_refs, controls, base, *, audit=False):
    """Old targets only; called after the three audited final models are sealed."""
    from . import source, learning
    rows = source.load_rows(binding, "validation", base)
    c.require(rows and all(row["split"] == "validation" and row["source_fit"] in c.FITS
                          and not {"assessment", "assessment_outcomes", "fresh_outcomes", "fresh_means"} & row.keys()
                          for row in rows)
              and len({row["id"] for row in rows}) == len(rows), "Invalid full validation corpus")
    if Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        c.require(len(rows) == c.CONFIG["validation_rows"], "Incomplete full validation corpus")
    result = {"manifest": binding, "states": len(rows), "data_scope": "validation",
              "old_targets_only": True, "after_fixed_model_seal": True, "matched_source_fit": True,
              "assessment_read": False, "checkpoint_selection": False,
              "target_temperature": c.CONFIG["target_temperature"],
              "mean_loss_definition": "legacy_hard_first_winner_cross_entropy",
              "soft_diagnostics_are_old_targets_only": True, "families": {}}
    for family in ("anchor", "focused", "specialist", "lowrate", "highrate"):
        records, assignments = [], {}
        for fit in c.FITS:
            selected = [row for row in rows if row["source_fit"] == fit]
            if not selected:
                continue  # Small fixtures may omit a source-fit stratum.
            model = focused[fit] if family == "focused" else controls[f"{family}_{fit}"]
            ref = registry["models"][f"focused_{fit}"] if family == "focused" else control_refs[f"{family}_{fit}"]
            report = learning.training_report(model, selected, base, c.CONFIG, audit=audit,
                data_scope="validation", diagnostic_scope="full_validation_matched_source_fit")
            c.require(report["validation_read"] is True and report["assessment_read"] is False,
                      "Validation diagnostic scope changed")
            records.extend(report["rows"])
            assignments[str(fit)] = {"model": ref, "states": len(selected)}
        records.sort(key=lambda row: row["id"])
        c.require(len(records) == len(rows), "Missing matched validation predictions")
        pair_ids = sorted({record["pair"] for record in records})
        reducer = math.fsum if audit else sum
        pair_loss, pair_regret = [], []
        for pair in pair_ids:
            items = [record for record in records if record["pair"] == pair]
            pair_loss.append(reducer(record["loss"] for record in items)/len(items))
            pair_regret.append(reducer(record["old_regret_units"] for record in items)/(256*len(items)))
        result["families"][family] = {"rows": records, "models": assignments,
            "metrics": learning._metrics(records, audit=audit), "pair_weighted": {
                "pairs": len(pair_ids), "mean_loss": reducer(pair_loss)/len(pair_ids),
                "mean_old_score_regret": reducer(pair_regret)/len(pair_ids),
                **{f"mean_{field}": reducer(
                    reducer(record[field] for record in records if record["pair"] == pair)
                    /sum(record["pair"] == pair for record in records) for pair in pair_ids)/len(pair_ids)
                   for field in ("soft_loss", "target_entropy", "target_kl", "policy_entropy")}}}
    return result


def _columns(worlds, support):
    c.require(len(worlds) == 128 and all(len(w["outcomes"]) == support for w in worlds), "Incomplete outcome block")
    totals, missing = [0]*support, [0]*support
    for world in worlds:
        for i, outcome in enumerate(world["outcomes"]):
            if outcome["reason"] == "repetition_cutoff":
                c.require(outcome["wdl"] is None and outcome["winner"] is None, "Invented unresolved outcome")
                missing[i] += 1
            else:
                c.require(outcome["reason"] in ("completed", "forced_pass_deadlock")
                          and type(outcome["wdl"]) in (float, int) and outcome["wdl"] in (-1, 0, 1), "Invalid terminal outcome")
                totals[i] += int(outcome["wdl"])
    return totals, missing


def _best(totals, missing, support):
    if not support:
        return None
    certified = []
    for position, candidate in enumerate(support):
        if all(totals[candidate]-totals[other]-missing[candidate]-missing[other]
               >= (1 if j < position else 0) for j, other in enumerate(support) if other != candidate):
            certified.append(candidate)
    c.require(len(certified) == 1, "Old outcomes cannot certify a fixed decision")
    return certified[0]


def _old_input(item):
    seal = c.verified(item["selection_artifact"])
    c.require(seal["manifest"] == c.SOURCE_BINDING and seal["stage"] == "main"
              and seal["index"] == item["index"], "Wrong predecessor selection seal")
    identity, lock = seal["identity"], seal["selection_lock"]
    for name in ("spec", "snapshot", "candidates"):
        c.require(old.digest(identity[name]) == old.digest(item[name]), "Old input differs from pre-assessment seal")
    c.require(lock["identity_digest"] == old.digest(identity)
              and lock["old_selection_sha256"] == old.digest(seal["old_selection"])
              and lock["choices"] == item["old_choices"], "Corrupted old-only choice evidence")
    totals, missing = _columns(seal["old_selection"]["worlds"], len(identity["candidates"]["keys"]))
    c.require(totals == item["old_sums_wdl"] == lock["selection_statistics"]["sums_wdl"]
              and missing == lock["selection_statistics"]["unknown_counts"], "Old outcomes differ from sealed targets")
    c.require(item.get("old_unknown_counts", missing) == missing, "Changed old unknown counts")
    return seal, totals, missing


def _choices(totals, missing, scores, inherited, specialist_scores, lowrate_scores, highrate_scores):
    n = len(totals)
    c.require(1 <= n <= 8 and len(scores) == n and len(missing) == n
              and all(math.isfinite(x) for x in scores), "Invalid offered logits")
    choices = dict(inherited)
    c.require(set(choices) == set(st.ARMS)-st.NEW_ARMS, "Incomplete inherited choices")
    c.require(choices["anchor"] == _best(totals, missing, list(range(min(n, 4))))
              and choices["teacher"] == _best(totals, missing, list(range(n)))
              and choices["teacher_outsider"] == _best(totals, missing, list(range(3, n)))
              and choices["anchor_outsider"] == choices["original_fourth"] == (3 if n > 3 else None)
              and choices["anchor_supported"] == choices["anchor"], "Changed predecessor controls")
    for family, values in (("focused", scores), ("specialist", specialist_scores), ("lowrate", lowrate_scores), ("highrate", highrate_scores)):
        c.require(len(values) == n and all(math.isfinite(v) for v in values), "Invalid new control logits")
        outsider = max(range(3, n), key=lambda j: values[j]) if n > 3 else None
        candidates = list(range(min(n, 3))) + ([] if outsider is None else [outsider])
        choices[f"{family}_outsider"] = outsider
        choices[f"{family}_supported"] = _best(totals, missing, candidates)
    teacher_support = list(range(min(n, 3))) + ([] if n <= 3 else [choices["teacher_outsider"]])
    c.require(_best(totals, missing, teacher_support) == choices["teacher"], "Teacher outsider fails full-teacher identity")
    return choices


def _predict(row, model):
    features = row["features"]
    with torch.no_grad():
        logits, _ = model(row["observation"][None], features[None], torch.ones((1, len(features)), dtype=torch.bool))
    c.require(logits.shape == (1, len(features)) and torch.isfinite(logits).all().item(), "Invalid focused predictions")
    full = logits[0].tolist()
    offered = [max(full[int(i)] for i in members) for members in row["members"]]
    return {"legal_logits": full, "offered_group_scores": offered}


def _row(spec, choices, worlds):
    n = len(worlds[0]["outcomes"])
    totals, missing = _columns(worlds, n)
    bounds = {name: None if index is None else [(128+totals[index]-missing[index])/256,
                                               (128+totals[index]+missing[index])/256]
              for name, index in choices.items()}
    scores = {name: value[0] if value is not None and value[0] == value[1] else None for name, value in bounds.items()}
    deltas = {}
    for name, left, right in st.CONTRASTS:
        if choices[left] == choices[right]:
            deltas[name] = [0., 0.]
        else:
            c.require(bounds[left] is not None and bounds[right] is not None, "Asymmetric absent outsider")
            deltas[name] = [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
    return {**{key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "split")},
            "privacy": spec["privacy"], "support": n, "outsider_available": n > 3,
            "choices": choices, "arm_scores": scores, "arm_score_bounds": bounds,
            "delta_bounds": deltas, "unknown_counts": missing}


def _open_assessment(descriptor, prediction, lock_ref, base):
    # This guard is intentionally at the sole collection-side outcome access point.
    c.require(Path(base, "predictions_lock.json").exists(), "Predictions must be sealed before assessment access")
    lock = c.verified(lock_ref)
    c.require(lock["complete"] is True and lock["predictions"][descriptor["index"]] == prediction,
              "Assessment attempted with an unsealed prediction")
    root, audit = c.verified(descriptor["root"]), c.verified(descriptor["audit"])
    c.require(root["manifest"] == audit["manifest"] == c.SOURCE_BINDING
              and audit["passed"] is True and _same_ref(audit["root"], descriptor["root"])
              and "failure" not in root and root["stage"] == "main"
              and old.digest(root["spec"]) == old.digest(descriptor["spec"])
              and _same_ref(root["selection_artifact"], prediction["source_selection"]), "Invalid inherited assessment")
    assessment = root["assessment"]
    expected = list(range(2211000000+512*descriptor["index"], 2211000000+512*descriptor["index"]+128))
    c.require(assessment["candidate_indices"] == list(range(len(root["candidates"]["keys"])))
              and assessment["world_seeds"] == [w["seed"] for w in assessment["worlds"]] == expected,
              "Changed inherited assessment mapping")
    c.require(root["selection_lock"]["choices"] == {k: v for k, v in prediction["choices"].items()
                                                     if k not in st.NEW_ARMS}, "Changed inherited choices")
    return root


def _context_controls(rows, base):
    if not Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        return {"fixture": True}
    status = old.read_json(c.SOURCE/"final_status.json")
    c.require(status["manifest"] == c.SOURCE_BINDING and status["complete"] is True, "Predecessor not closed")
    result = c.verified(status["result"])
    controls = {}
    for split in st.SPLITS:
        selected = [row for row in rows if row["split"] == split]
        controls[split] = {}
        for name in st.CONTEXT:
            measured = [sum(row["delta_bounds"][name][i] for row in selected)/len(selected) for i in (0, 1)]
            c.require(measured == result["statistics"]["groups"][split]["contrasts"][name]["mean_bounds"],
                      "Inherited positive-control point mean changed")
            controls[split][name] = measured
    return {"source_result": status["result"], "point_means_exact": True, "groups": controls,
            "new_seed_intervals_need_not_match_predecessor": True}


def _legacy_projection(current, previous):
    """Preserve every historical hard-label field while allowing new soft diagnostics."""
    if isinstance(previous, dict):
        c.require(isinstance(current, dict) and set(previous) <= set(current), "Missing historical diagnostic field")
        return {key: _legacy_projection(current[key], value) for key, value in previous.items()}
    if isinstance(previous, list):
        c.require(isinstance(current, list) and len(current) == len(previous), "Changed historical diagnostic rows")
        return [_legacy_projection(a, b) for a, b in zip(current, previous, strict=True)]
    return current


def _rate_identity(binding, predictions, rows, validation, base, family):
    """Check a saved rate experiment only after the new global seal."""
    from . import source
    lock = old.read_json(Path(base)/"predictions_lock.json")
    c.require(lock["manifest"] == binding and lock["complete"] is True
              and lock["predictions"] == predictions, "Seal predictions before opening fixed-rate assessment")
    c.require(family in ("lowrate", "highrate"), "Unknown rate control")
    refs = getattr(source, family+"_evidence")(binding, base)
    previous = c.verified(refs["assessment"])
    previous_predictions = c.verified(refs["predictions"])
    previous_validation = c.verified(refs["validation_old_targets"])
    c.require(previous["manifest"] == previous_predictions["manifest"] == previous_validation["manifest"]
              and previous["manifest"] != binding and _same_ref(previous["predictions"], refs["predictions"])
              and _same_ref(previous["validation_old_targets"], refs["validation_old_targets"])
              and _same_ref(previous["models"], refs["models"]), "Frozen-rate predecessor linkage changed")
    c.require(len(rows) == len(previous["rows"]) == len(predictions) == len(previous_predictions["predictions"]),
              "Frozen-rate panel population changed")
    rename = lambda name: family+"_"+name[len("focused_"):] if name.startswith("focused_") else name
    for current, saved, prediction, old_prediction in zip(rows, previous["rows"], predictions, previous_predictions["predictions"], strict=True):
        c.require(current["index"] == saved["index"] == prediction["index"] == old_prediction["index"]
                  and current["fit"] == saved["fit"] == prediction["fit"] == old_prediction["fit"]
                  and prediction["row_id"] == old_prediction["row_id"]
                  and prediction["spec"] == old_prediction["spec"]
                  and _same_ref(prediction["source_selection"], old_prediction["source_selection"])
                  and _same_ref(prediction[family+"_model"], old_prediction["model"]), "Frozen-rate prediction identity changed")
        compare(prediction[family+"_proposal"], old_prediction["proposal"])
        for field in ("pair", "game_index", "seat", "phase", "split", "privacy", "support", "outsider_available", "unknown_counts"):
            c.require(current[field] == saved[field], "Frozen-rate source population differs")
        for arm in saved["choices"]:
            mapped = rename(arm)
            c.require(prediction["choices"][mapped] == old_prediction["choices"][arm], "Frozen-rate old-world choice changed")
            for field in ("choices", "arm_scores", "arm_score_bounds"):
                c.require(current[field][mapped] == saved[field][arm], "Frozen-rate assessed choice or score changed")
    means = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        means[split] = {}
        for suffix in ("supported_minus_anchor", "outsider_minus_anchor_outsider"):
            value = [sum(row["delta_bounds"][family+"_"+suffix][j] for row in selected)/len(selected) for j in (0, 1)]
            c.require(value == previous["statistics"]["groups"][split]["contrasts"]["focused_"+suffix]["mean_bounds"],
                      "Frozen-rate point means changed")
            means[split][family+"_"+suffix] = value
    validation_families = [("anchor", "anchor"), ("specialist", "specialist"), (family, "focused")]
    if family == "highrate":
        validation_families.append(("lowrate", "lowrate"))
    for current_family, previous_family in validation_families:
        current, saved = validation["families"][current_family], previous_validation["families"][previous_family]
        for field in ("rows", "metrics", "pair_weighted"):
            compare(_legacy_projection(current[field], saved[field]), saved[field], rel_tol=0., abs_tol=1e-12)
        c.require(set(current["models"]) == set(saved["models"]), "Frozen-rate validation fit coverage changed")
        for fit, assignment in current["models"].items():
            c.require(assignment["states"] == saved["models"][fit]["states"]
                      and _same_ref(assignment["model"], saved["models"][fit]["model"]), "Frozen-rate validation assignment changed")
    return {"evidence": refs, "prediction_choices_exact": True, "assessment_scores_exact": True,
            "point_means_exact": True, "old_validation_metrics_reproduced": True, "groups": means}


def _lowrate_identity(binding, predictions, rows, validation, base):
    return _rate_identity(binding, predictions, rows, validation, base, "lowrate")


def _highrate_identity(binding, predictions, rows, validation, base):
    return _rate_identity(binding, predictions, rows, validation, base, "highrate")


def assess(binding, base=OUTPUT):
    base = Path(base)
    c.budget_boundary(base)
    c.require(not (base/"predictions_lock.json").exists() and not (base/"assessment_result.json").exists(),
              "Preserve prior prediction and outcome evidence")
    registry_ref, registry, models = _models(binding, base)
    control_refs, controls = _control_models(binding, base)
    rows, inputs, descriptors = _panel(binding, base)
    predictions = []
    for row, item in zip(rows, inputs):
        c.budget_boundary(base)
        _, totals, missing = _old_input(item)
        fit = item["spec"]["fit"]
        proposal = _predict(row, models[fit])
        specialist = _predict(row, controls[f"specialist_{fit}"])
        lowrate = _predict(row, controls[f"lowrate_{fit}"])
        highrate = _predict(row, controls[f"highrate_{fit}"])
        choices = _choices(totals, missing, proposal["offered_group_scores"], item["old_choices"],
                           specialist["offered_group_scores"], lowrate["offered_group_scores"], highrate["offered_group_scores"])
        predictions.append({"index": item["index"], "fit": fit, "row_id": row["id"], "spec": item["spec"],
                            "model": registry["models"][f"focused_{fit}"], "source_selection": item["selection_artifact"],
                            "proposal": proposal, "specialist_proposal": specialist,
                            "specialist_model": control_refs[f"specialist_{fit}"],
                            "lowrate_proposal": lowrate, "lowrate_model": control_refs[f"lowrate_{fit}"],
                            "highrate_proposal": highrate, "highrate_model": control_refs[f"highrate_{fit}"], "choices": choices})
    validation = _validation_metrics(binding, registry, models, control_refs, controls, base)
    validation_path = base/"validation_old_targets.json"
    old.save_new(validation_path, validation)
    validation_ref = c.artifact(validation_path)
    lock = {"manifest": binding, "models": registry_ref, "complete": True, "states": len(predictions),
            "controls": control_refs, "validation_old_targets": validation_ref,
            "predictions": predictions, "created_at_ns": time.time_ns(), "time": old.now(),
            "selection_used_assessment": False, "all_predictions_precede_assessment_open": True}
    lock_path = base/"predictions_lock.json"
    old.save_new(lock_path, lock)
    lock_ref = c.artifact(lock_path)
    assessment_opened_at_ns = time.time_ns()
    scored, inherited = [], []
    for descriptor, prediction in zip(descriptors, predictions):
        c.budget_boundary(base)
        root = _open_assessment(descriptor, prediction, lock_ref, base)
        row = _row(descriptor["spec"], prediction["choices"], root["assessment"]["worlds"])
        for name in st.CONTEXT:
            c.require(row["delta_bounds"][name] == root["summary"]["delta_bounds"][name], "Changed baseline arithmetic")
        scored.append(row)
        inherited.append({"index": descriptor["index"], "root": descriptor["root"], "audit": descriptor["audit"]})
    context = _context_controls(scored, base)
    lowrate_identity = _lowrate_identity(binding, predictions, scored, validation, base)
    highrate_identity = _highrate_identity(binding, predictions, scored, validation, base)
    statistics = st.summarize(scored)
    result = {"manifest": binding, "models": registry_ref, "predictions": lock_ref, "rows": scored,
              "statistics": statistics, "development_screen": st.development_screen(statistics),
              "validation_old_targets": validation_ref,
              "lowrate_control_identity": lowrate_identity, "highrate_control_identity": highrate_identity,
              "context_controls": context, "inherited_assessments": inherited,
              "assessment_opened_at_ns": assessment_opened_at_ns, "new_simulations": 0,
              "assessment_used_for_selection": False, "reused_development": True,
              "classification": "diagnostic_only", "promotion": False, "time": old.now()}
    path = base/"assessment_result.json"
    old.save_new(path, result)
    return c.artifact(path)


def _independent_proposal(row, item, model, state=None):
    state = old.deserialize_state_snapshot(item["snapshot"]) if state is None else state
    legal = old.SplendorEnv().legal_actions(state)
    actions = [old.decode_action(value) for value in item["candidates"]["actions"]]
    keys = item["candidates"]["keys"]
    c.require(all(action in legal for action in actions)
              and [repr(old.canonical_successor_action_key(state, action)) for action in actions] == keys,
              "Unlawful offered action identity")
    groups, members = [], {}
    for index, action in enumerate(legal):
        key = repr(old.canonical_successor_action_key(state, action))
        if key not in members:
            groups.append(key)
            members[key] = []
        members[key].append(index)
    observation = encode_public_observation_tensor(state, state.current_player)
    feature_rows = encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)
    obs = torch.tensor(fit_feature_vector(observation.vector, model.config.observation_size), dtype=torch.float32)
    features = torch.tensor([fit_feature_vector(value.vector, model.config.action_feature_size) for value in feature_rows], dtype=torch.float32)
    c.require(torch.equal(obs, row["observation"]) and torch.equal(features, row["features"]), "Export differs from independent lawful encoding")
    c.require([members[key] for key in keys] == [value.tolist() for value in row["members"]], "Wrong canonical maximum membership")
    c.require(row["groups"].tolist() == [groups.index(repr(old.canonical_successor_action_key(state, action))) for action in legal],
              "Wrong full-legal grouping")
    with torch.no_grad():
        logits, _ = model(obs.unsqueeze(0), features.unsqueeze(0), torch.ones(1, len(legal), dtype=torch.bool))
    full = [float(value) for value in logits[0]]
    c.require(all(math.isfinite(value) for value in full), "Nonfinite audited predictions")
    return {"legal_logits": full, "offered_group_scores": [max(full[i] for i in members[key]) for key in keys]}


def _independent_columns(worlds, count):
    c.require(len(worlds) == 128, "Incomplete independent outcome block")
    totals, missing = [], []
    for column in range(count):
        observed, censored = [], 0
        for world in worlds:
            c.require(len(world["outcomes"]) == count, "Wrong independent support")
            outcome = world["outcomes"][column]
            if outcome["wdl"] is None:
                c.require(outcome["reason"] == "repetition_cutoff" and outcome["winner"] is None, "Invalid missing outcome")
                censored += 1
            else:
                c.require(outcome["reason"] in ("completed", "forced_pass_deadlock")
                          and type(outcome["wdl"]) in (float, int) and outcome["wdl"] in (-1, 0, 1), "Invalid audited outcome")
                observed.append(int(outcome["wdl"]))
        totals.append(sum(observed))
        missing.append(censored)
    return totals, missing


def _independent_best(totals, missing, indices):
    winners = [candidate for candidate in indices if all(
        (totals[candidate]-missing[candidate] > totals[other]+missing[other] if other < candidate
         else totals[candidate]-missing[candidate] >= totals[other]+missing[other])
        for other in indices if candidate != other)]
    c.require(len(winners) == (1 if indices else 0), "Uncertified independently selected old label")
    return winners[0] if winners else None


def _independent_choices(seal, proposal, specialist_proposal, lowrate_proposal, highrate_proposal):
    n = len(seal["identity"]["candidates"]["keys"])
    totals, missing = _independent_columns(seal["old_selection"]["worlds"], n)
    choices = {"anchor": _independent_best(totals, missing, list(range(min(n, 4)))),
               "teacher": _independent_best(totals, missing, list(range(n))),
               "teacher_outsider": _independent_best(totals, missing, list(range(3, n))),
               "original_fourth": 3 if n > 3 else None}
    for family in ("anchor", "broad", "narrow", "focused", "specialist", "lowrate", "highrate"):
        additions = {"focused": proposal, "specialist": specialist_proposal, "lowrate": lowrate_proposal, "highrate": highrate_proposal}
        logits = additions[family]["offered_group_scores"] if family in additions else seal["identity"]["proposals"][family]["offered_group_scores"]
        c.require(len(logits) == n and all(math.isfinite(v) for v in logits), "Invalid predecessor logits")
        outside = min(range(3, n), key=lambda j: (-logits[j], j)) if n > 3 else None
        choices[f"{family}_outsider"] = outside
        if family not in ("focused", "specialist", "lowrate", "highrate"):
            choices[f"{family}_direct"] = min(range(n), key=lambda j: (-logits[j], j))
        choices[f"{family}_supported"] = _independent_best(totals, missing, list(range(min(n, 3))) + ([] if outside is None else [outside]))
    c.require({k: v for k, v in choices.items() if k not in st.NEW_ARMS} == seal["selection_lock"]["choices"],
              "Inherited choices fail independent reconstruction")
    c.require(choices["anchor_outsider"] == choices["original_fourth"] and choices["anchor_supported"] == choices["anchor"],
              "Anchor proposal identity changed")
    support = list(range(min(n, 3))) + ([] if n <= 3 else [choices["teacher_outsider"]])
    c.require(_independent_best(totals, missing, support) == choices["teacher"], "Teacher support identity failed")
    return choices


def _independent_row(spec, choices, worlds):
    support = len(worlds[0]["outcomes"])
    totals, unknown = _independent_columns(worlds, support)
    bounds, scores = {}, {}
    for arm in st.ARMS:
        choice = choices[arm]
        c.require((choice is None) == (support <= 3 and (arm.endswith("_outsider") or arm == "original_fourth")), "Invalid absent action")
        if choice is None:
            bounds[arm] = scores[arm] = None
        else:
            c.require(type(choice) is int and 0 <= choice < support, "Wrong chosen action index")
            bounds[arm] = [(128+totals[choice]+sign*unknown[choice])/256 for sign in (-1, 1)]
            scores[arm] = bounds[arm][0] if not unknown[choice] else None
    differences = {}
    for name, left, right in st.CONTRASTS:
        a, b = choices[left], choices[right]
        differences[name] = [0., 0.] if a == b else [
            (totals[a]-totals[b]+sign*(unknown[a]+unknown[b]))/256 for sign in (-1, 1)]
    return {**{key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "split")},
            "privacy": spec["privacy"], "support": support, "outsider_available": support > 3,
            "choices": choices, "arm_scores": scores, "arm_score_bounds": bounds,
            "delta_bounds": differences, "unknown_counts": unknown}


def audit(binding, result_ref, base=OUTPUT):
    base = Path(base)
    c.budget_boundary(base)
    c.require(not (base/"assessment_audit.json").exists(), "Preserve assessment audit")
    result = c.verified(result_ref)
    c.require(result["manifest"] == binding and result["new_simulations"] == 0
              and result["assessment_used_for_selection"] is False and result["reused_development"] is True
              and result["classification"] == "diagnostic_only" and result["promotion"] is False, "Changed assessment scope")
    lock = c.verified(result["predictions"])
    registry_ref, registry, models = _models(binding, base)
    control_refs, controls = _control_models(binding, base)
    c.require(result["models"] == lock["models"] == registry_ref and lock["manifest"] == binding
              and lock["controls"] == control_refs and lock["validation_old_targets"] == result["validation_old_targets"]
              and lock["complete"] is True and lock["selection_used_assessment"] is False
              and lock["all_predictions_precede_assessment_open"] is True
              and c.verify_hash(registry_ref).stat().st_mtime_ns
              <= c.verify_hash(result["validation_old_targets"]).stat().st_mtime_ns <= lock["created_at_ns"]
              and lock["created_at_ns"] <= c.verify_hash(result["predictions"]).stat().st_mtime_ns
              <= result["assessment_opened_at_ns"], "Prediction seal timing or identity failed")
    rows, inputs, descriptors = _panel(binding, base)
    c.require(lock["states"] == len(rows) == len(lock["predictions"]) == len(result["rows"]), "Missing focused predictions")
    c.require(result["inherited_assessments"] == [
        {"index": item["index"], "root": item["root"], "audit": item["audit"]} for item in descriptors],
        "Changed inherited assessment receipts")
    independent_rows = []
    for row, item, descriptor, saved in zip(rows, inputs, descriptors, lock["predictions"]):
        c.budget_boundary(base)
        fit = descriptor["spec"]["fit"]
        c.require(saved["index"] == item["index"] and saved["fit"] == row["source_fit"] == fit
                  and saved["row_id"] == row["id"] and old.digest(saved["spec"]) == old.digest(item["spec"])
                  and saved["model"] == registry["models"][f"focused_{fit}"]
                  and saved["specialist_model"] == control_refs[f"specialist_{fit}"]
                  and saved["lowrate_model"] == control_refs[f"lowrate_{fit}"]
                  and saved["highrate_model"] == control_refs[f"highrate_{fit}"]
                  and saved["source_selection"] == item["selection_artifact"], "Wrong frozen fit assignment")
        seal = c.verified(item["selection_artifact"])
        c.require(seal["manifest"] == c.SOURCE_BINDING and seal["index"] == item["index"]
                  and seal["selection_lock"]["identity_digest"] == old.digest(seal["identity"])
                  and seal["selection_lock"]["old_selection_sha256"] == old.digest(seal["old_selection"]), "Changed old evidence")
        for key in ("spec", "snapshot", "candidates"):
            c.require(old.digest(seal["identity"][key]) == old.digest(item[key]), "Changed old input")
        expected = _independent_proposal(row, item, models[fit])
        compare(saved["proposal"], expected)
        state = old.deserialize_state_snapshot(item["snapshot"])
        alternate = input_audit.hidden_alternative(state)
        compare(expected, _independent_proposal(row, item, models[fit], state=alternate))
        specialist = _independent_proposal(row, item, controls[f"specialist_{fit}"])
        compare(saved["specialist_proposal"], specialist)
        compare(specialist, _independent_proposal(row, item, controls[f"specialist_{fit}"], state=alternate))
        lowrate = _independent_proposal(row, item, controls[f"lowrate_{fit}"])
        compare(saved["lowrate_proposal"], lowrate)
        compare(lowrate, _independent_proposal(row, item, controls[f"lowrate_{fit}"], state=alternate))
        highrate = _independent_proposal(row, item, controls[f"highrate_{fit}"])
        compare(saved["highrate_proposal"], highrate)
        compare(highrate, _independent_proposal(row, item, controls[f"highrate_{fit}"], state=alternate))
        choices = _independent_choices(seal, expected, specialist, lowrate, highrate)
        c.require(saved["choices"] == choices, "Incorrect frozen focused choices")
        # Independent path intentionally does not call the collector's outcome accessor.
        root, receipt = c.verified(descriptor["root"]), c.verified(descriptor["audit"])
        c.require(root["manifest"] == receipt["manifest"] == c.SOURCE_BINDING and receipt["passed"] is True
                  and _same_ref(receipt["root"], descriptor["root"]) and "failure" not in root
                  and old.digest(root["spec"]) == old.digest(item["spec"])
                  and root["selection_lock"] == seal["selection_lock"]
                  and _same_ref(root["selection_artifact"], item["selection_artifact"]), "Wrong inherited outcome identity")
        count = len(item["candidates"]["keys"])
        block = root["assessment"]
        seeds = [2211000000+512*item["index"]+j for j in range(128)]
        c.require(block["candidate_indices"] == list(range(count))
                  and block["world_seeds"] == [world["seed"] for world in block["worlds"]] == seeds,
                  "Wrong inherited world mapping")
        independent = _independent_row(item["spec"], choices, block["worlds"])
        c.require(independent == result["rows"][item["index"]], "New assessment arithmetic differs")
        for name in st.CONTEXT:
            c.require(independent["delta_bounds"][name] == receipt["row"]["delta_bounds"][name], "Historical context arithmetic differs")
        independent_rows.append(independent)
    independent = independent_statistics(independent_rows)
    compare(result["statistics"], independent, rel_tol=0., abs_tol=1e-12)
    endpoint = independent["groups"]["validation"]["contrasts"]["focused_supported_minus_anchor"]
    control = independent["groups"]["validation"]["contrasts"]["focused_supported_minus_lowrate_supported"]
    passes = (endpoint["mean_bounds"][0] >= c.CONFIG["minimum_progress"] and endpoint["ci95"][0] > 0
              and control["ci95"][0] > 0)
    compare(result["development_screen"], {
        "primary_endpoint": "focused_supported_minus_anchor", "split": "validation",
        "minimum_mean": c.CONFIG["minimum_progress"], "positive_lower_ci_required": True,
        "required_control_endpoint": "focused_supported_minus_lowrate_supported",
        "control_ci95": control["ci95"], "control_lower_strictly_positive": control["ci95"][0] > 0,
        "highrate_attribution_is_additional_gate": False,
        "passed": passes, "classification": "fresh_confirmation_justified" if passes else "progress_not_established",
        "specialist_attribution_is_additional_gate": False, "fresh_full_legal_confirmation_required": True,
        "reused_development": True, "automatic_successor": False, "promotion": False}, rel_tol=0., abs_tol=1e-12)
    validation = _validation_metrics(binding, registry, models, control_refs, controls, base, audit=True)
    compare(c.verified(result["validation_old_targets"]), validation, rel_tol=0., abs_tol=1e-12)
    compare(result["lowrate_control_identity"], _lowrate_identity(binding, lock["predictions"], independent_rows, validation, base),
            rel_tol=0., abs_tol=1e-12)
    compare(result["highrate_control_identity"], _highrate_identity(binding, lock["predictions"], independent_rows, validation, base),
            rel_tol=0., abs_tol=1e-12)
    compare(result["context_controls"], _context_controls(independent_rows, base), rel_tol=0., abs_tol=1e-12)
    receipt = {"manifest": binding, "result": result_ref, "predictions": result["predictions"], "models": registry_ref,
               "passed": True, "states": len(independent_rows), "new_focused_inference_independently_reproduced": True,
               "lawful_inputs_and_private_invariance_pass": True, "old_choices_independently_reconstructed": True,
               "all_new_arithmetic_pass": True, "independent_statistics": independent,
               "specialist_inference_independently_reproduced": True,
               "lowrate_inference_independently_reproduced": True,
               "lowrate_control_identity_pass": True,
               "highrate_inference_independently_reproduced": True, "highrate_control_identity_pass": True,
               "full_validation_old_target_metrics_independently_reproduced": True,
               "development_screen_independently_reproduced": True,
               "unknown_outcomes": sum(sum(row["unknown_counts"]) for row in independent_rows),
               "new_trajectory_replays": 0, "inherited_assessment_audits": True,
               "full_inner_replay": False, "time": old.now()}
    path = base/"assessment_audit.json"
    old.save_new(path, receipt)
    return c.artifact(path)


def independent_statistics(rows):
    """Independent histogram bootstrap with explicit integer endpoint arithmetic."""
    assert len(rows) == 336 and [r["index"] for r in rows] == list(range(336))
    pair_ids = sorted({r["pair"] for r in rows})
    assert len(pair_ids) == 84
    pair_values, pair_cells = {}, {}
    for pair in pair_ids:
        records = [r for r in rows if r["pair"] == pair]
        assert len(records) == 4 and {r["game_index"] for r in records} == {2*pair, 2*pair+1}
        assert {r["seat"] for r in records} == {0, 1}
        assert all((r["fit"], r["split"]) == (records[0]["fit"], records[0]["split"]) for r in records)
        for game in (2*pair, 2*pair+1):
            leg = [r for r in records if r["game_index"] == game]
            assert len(leg) == 2 and len({r["phase"] for r in leg}) == 2
        assert {r["phase"] for r in records if r["game_index"] == 2*pair} == {r["phase"] for r in records if r["game_index"] == 2*pair+1}
        pair_cells[pair] = (records[0]["split"], records[0]["fit"])
        values = np.zeros(len(st.CONTRASTS)*2, dtype=np.int64)
        for row in records:
            assert set(row["choices"]) == set(row["arm_scores"]) == set(row["arm_score_bounds"]) == set(st.ARMS)
            assert type(row["support"]) is int and 1 <= row["support"] <= 8
            assert row["outsider_available"] == (row["support"] > 3)
            for arm in st.ARMS:
                choice = row["choices"][arm]
                bound = row["arm_score_bounds"][arm]
                if choice is None:
                    assert row["support"] <= 3 and (arm.endswith("_outsider") or arm == "original_fourth")
                    assert bound is row["arm_scores"][arm] is None
                else:
                    assert type(choice) is int and 0 <= choice < row["support"]
                    if arm.endswith("_outsider") or arm == "original_fourth":
                        assert choice >= 3
                    assert len(bound) == 2 and 0 <= bound[0] <= bound[1] <= 1
                    assert all(type(x) in (int, float) and math.isfinite(x) and x*256 == int(x*256) for x in bound)
                    assert row["arm_scores"][arm] == (bound[0] if bound[0] == bound[1] else None)
            assert row["choices"]["anchor_outsider"] == row["choices"]["original_fourth"]
            assert row["choices"]["anchor_supported"] == row["choices"]["anchor"]
            assert row["choices"]["original_fourth"] == (3 if row["support"] > 3 else None)
            for family in ("anchor", "broad", "narrow", "focused", "specialist", "lowrate", "highrate"):
                assert row["choices"][f"{family}_supported"] in (
                    list(range(min(row["support"], 3))) + ([] if row["support"] <= 3 else [row["choices"][f"{family}_outsider"]]))
            assert set(row["delta_bounds"]) == set(st.NAMES)
            for j, (name, left, right) in enumerate(st.CONTRASTS):
                bounds = row["delta_bounds"][name]
                expected = [0., 0.] if row["choices"][left] == row["choices"][right] else [
                    row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][1],
                    row["arm_score_bounds"][left][1]-row["arm_score_bounds"][right][0]]
                assert bounds == expected and len(bounds) == 2 and -1 <= bounds[0] <= bounds[1] <= 1
                for endpoint, value in enumerate(bounds):
                    assert type(value) in (int, float) and math.isfinite(value) and value*256 == int(value*256)
                    values[2*j+endpoint] += int(value*256)
        pair_values[pair] = values
    strata = []
    for split in ("train", "validation"):
        for fit in c.FITS:
            members = [p for p in pair_ids if pair_cells[p] == (split, fit)]
            assert len(members) == 14
            strata.append(np.stack([pair_values[p] for p in members]))
    totals = [sum((a.sum(axis=0) for a in strata[i:i+3]), np.zeros(2*len(st.CONTRASTS), dtype=np.int64)) for i in (0, 3)]
    rng = np.random.default_rng(c.CONFIG["bootstrap_seed"])
    boot = np.zeros((c.CONFIG["bootstrap_samples"], 2, len(st.CONTRASTS)*2), dtype=np.int64)
    for start in range(0, len(boot), c.CONFIG["bootstrap_chunk"]):
        count = min(c.CONFIG["bootstrap_chunk"], len(boot)-start)
        for si, matrix in enumerate(strata):
            indices = rng.integers(0, 14, size=(count, 14))
            frequencies = np.stack([np.bincount(draw, minlength=14) for draw in indices])
            boot[start:start+count, si//3] += frequencies @ matrix
    denominator = 168*256
    def quantile(column, fraction):
        ordered = np.sort(column)
        position = (len(ordered)-1)*fraction
        lower, upper = math.floor(position), math.ceil(position)
        return (int(ordered[lower])+(int(ordered[upper])-int(ordered[lower]))*(position-lower))/denominator
    def entry(lo, hi, low_samples, high_samples):
        bounds = [int(lo)/denominator, int(hi)/denominator]
        interval = [quantile(low_samples, .025), quantile(high_samples, .975)]
        material = bool(int(lo)*200 >= denominator and interval[0] > 0)
        return {"mean": bounds[0] if lo == hi else None, "mean_bounds": bounds, "ci95": interval,
                "material_signal": material, "material_target_excluded": bool(interval[1] < .005),
                "positive_signal": bool(interval[0] > 0), "negative_signal": bool(interval[1] < 0),
                "classification": "material_signal" if material else "target_not_established"}
    groups = {}
    for si, split in enumerate(("train", "validation")):
        groups[split] = {"states": 168, "pairs": 42, "strata": 3, "contrasts": {
            name: entry(totals[si][2*j], totals[si][2*j+1], boot[:, si, 2*j], boot[:, si, 2*j+1])
            for j, (name, _, _) in enumerate(st.CONTRASTS)}}
    groups["validation_minus_train"] = {"contrasts": {
        name: entry(totals[1][2*j]-totals[0][2*j+1], totals[1][2*j+1]-totals[0][2*j],
                    boot[:, 1, 2*j]-boot[:, 0, 2*j+1], boot[:, 1, 2*j+1]-boot[:, 0, 2*j])
        for j, (name, _, _) in enumerate(st.CONTRASTS)}}
    return {"states": 336, "pairs": 84, "strata": 6, "bootstrap_samples": c.CONFIG["bootstrap_samples"], "bootstrap_seed": c.CONFIG["bootstrap_seed"],
            "resampling_unit": "whole_source_pair_four_roots", "groups": groups,
            "primary_contrasts": list(st.PRIMARY), "unknown_outcomes_imputed": False,
            "classification": "diagnostic_only", "promotion": False,
            "train_validation_difference_is_causal": False,
            "validation_previously_used_for_checkpoint_selection": True,
            "assessment_reused_development": True, "assessment_used_for_selection": False}
