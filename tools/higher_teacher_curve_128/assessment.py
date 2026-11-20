"""Seal the entire saved-checkpoint curve before opening reused outcomes.

New inference and arithmetic are independently reproduced; no optimizer or
trajectory is rerun. Every checkpoint and source-fit assignment is retained.
"""
from pathlib import Path
import math
import time

import torch

from . import common as c
from . import source, statistics
from tools.higher_teacher_rate_128 import assessment as prior
from tools.higher_teacher_rate_128 import learning as inherited_learning

old, OUTPUT = c.old, c.OUTPUT


def _progress(callback, stage, **values):
    if callback is not None:
        callback(stage, **values)


def _checkpoint_model(parent, update, fit):
    c.require(update in c.CHECKPOINTS and fit in c.FITS, "Unscheduled model identity")
    ref = parent["models"][str(update)][str(fit)]
    path = c.verify_hash(ref)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    c.require(payload["generation"] == update and payload["metadata"]["fit"] == fit
              and payload["metadata"]["manifest"] == c.PARENT_BINDING, "Wrong saved checkpoint or fit")
    model = prior.StrongPolicyInference(path).model
    model.eval()
    return model


def _panel(parent, base):
    rows, inputs, descriptors = source.panel(parent)
    count = len(rows)
    c.require(count == len(inputs) == len(descriptors) > 0, "Incomplete saved-curve panel")
    if Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        c.require(count == 336, "All fixed positions are required")
    c.require(set(parent["models"]) == {str(update) for update in c.CHECKPOINTS}, "Missing checkpoint history")
    for update in c.CHECKPOINTS:
        c.require(set(parent["models"][str(update)]) == {str(fit) for fit in c.FITS}, "Missing fit history")
    for i, (row, item, descriptor) in enumerate(zip(rows, inputs, descriptors)):
        c.require(row["index"] == item["index"] == descriptor["index"] == i,
                  "Wrong saved-curve ordering")
        c.require(old.digest(item["spec"]) == old.digest(descriptor["spec"])
                  and row["source_fit"] == item["spec"]["fit"] in c.FITS
                  and row["split"] == item["spec"]["split"]
                  and row["id"] == item["spec"]["source_rowid"], "Wrong source-fit mapping")
    return rows, inputs, descriptors


def _prediction(item, row, parent, update, proposal, choices):
    return {"index": item["index"], "fit": item["spec"]["fit"], "row_id": row["id"],
            "update": update, "model": parent["models"][str(update)][str(item["spec"]["fit"])],
            "source_selection": item["selection_artifact"], "proposal": proposal, "choices": choices}


def _initial_endpoint(seal, prediction):
    original = seal["identity"]["proposals"]["anchor"]
    prior.compare(prediction["proposal"], {"legal_logits": original["learned_logits"],
                                         "offered_group_scores": original["offered_group_scores"]})
    c.require(prediction["choices"]["focused_outsider"] == prediction["choices"]["anchor_outsider"]
              and prediction["choices"]["focused_supported"] == prediction["choices"]["anchor"],
              "Initial endpoint does not reproduce the original anchor")


def _label_record(row, totals, proposal, choices):
    outside = list(range(3, len(totals)))
    target = max(outside, key=lambda j: totals[j]) if outside else None
    choice = choices["focused_outsider"]
    c.require((choice is None) == (not outside), "Invalid absent outsider metric")
    values = proposal["offered_group_scores"]
    loss = 0.
    if len(outside) > 1:
        maximum = max(values[j] for j in outside)
        loss = maximum+math.log(sum(math.exp(values[j]-maximum) for j in outside))-values[target]
    regret = totals[target]-totals[choice] if outside else 0
    c.require(regret >= 0 and math.isfinite(loss), "Invalid old-label metric")
    return {"index": row["index"], "id": row["id"], "fit": row["source_fit"], "split": row["split"],
            "outsider_support": len(outside), "choice": choice, "target": target, "loss": loss,
            "first_winner": choice == target if outside else None,
            "any_old_max": regret == 0 if outside else None, "old_regret_units": regret}


def _label_metrics(records):
    groups = {}
    for split in ("train", "validation"):
        selected = [row for row in records if row["split"] == split]
        metrics = {}
        for label, minimum in (("all_rows", 0), ("nonempty_outsiders", 1), ("multiple_outsiders", 2)):
            rows = [row for row in selected if row["outsider_support"] >= minimum]
            eligible = [row for row in rows if row["outsider_support"] > 0]
            first = sum(row["first_winner"] for row in eligible)
            anymax = sum(row["any_old_max"] for row in eligible)
            regret = sum(row["old_regret_units"] for row in rows)
            metrics[label] = {"rows": len(rows), "accuracy_denominator": len(eligible),
                "mean_cross_entropy": sum(row["loss"] for row in rows)/len(rows) if rows else None,
                "first_winner_count": first, "tied_maximum_count": anymax,
                "first_winner_accuracy": first/len(eligible) if eligible else None,
                "tied_maximum_accuracy": anymax/len(eligible) if eligible else None,
                "old_regret_units_sum": regret, "mean_old_score_regret": regret/(256*len(rows)) if rows else None}
        groups[split] = {"matched_source_fit": True, "states": len(selected), "metrics": metrics,
                         "outsider_support_counts": {"empty": sum(r["outsider_support"] == 0 for r in selected),
                             "singleton": sum(r["outsider_support"] == 1 for r in selected),
                             "multiple": sum(r["outsider_support"] >= 2 for r in selected)}}
    return {"rows": records, "groups": groups, "old_targets_only": True,
            "no_checkpoint_selection": True, "not_a_generalization_or_utility_estimate": True}


def _open_outcomes(descriptor, inputs, lock_ref, base):
    # The sole collection-side accessor refuses any read before the global seal.
    c.require(Path(base, "predictions_lock.json").exists(), "Seal every checkpoint before assessment access")
    locked = c.verified(lock_ref)
    c.require(locked["complete"] is True and locked["all_predictions_precede_assessment_open"] is True
              and locked["assessment_used_for_selection"] is False
              and locked["checkpoints"] == list(c.CHECKPOINTS)
              and set(locked["predictions"]) == {str(u) for u in c.CHECKPOINTS}
              and all(len(values) == locked["states_per_checkpoint"] for values in locked["predictions"].values())
              and all(locked["predictions"][str(u)][descriptor["index"]]["index"] == descriptor["index"] for u in c.CHECKPOINTS),
              "Incomplete global prediction seal before outcome access")
    root, receipt = c.verified(descriptor["root"]), c.verified(descriptor["audit"])
    c.require(root["manifest"] == receipt["manifest"] == prior.c.SOURCE_BINDING and receipt["passed"] is True
              and prior._same_ref(receipt["root"], descriptor["root"])
              and root["stage"] == "main" and "failure" not in root
              and old.digest(root["spec"]) == old.digest(descriptor["spec"])
              and prior._same_ref(root["selection_artifact"], inputs["selection_artifact"]), "Invalid inherited outcome identity")
    block = root["assessment"]
    seeds = list(range(2211000000+512*descriptor["index"], 2211000000+512*descriptor["index"]+128))
    c.require(block["candidate_indices"] == list(range(len(inputs["candidates"]["keys"])))
              and block["world_seeds"] == [world["seed"] for world in block["worlds"]] == seeds,
              "Invalid saved outcome mapping")
    c.require(root["selection_lock"]["choices"] == inputs["old_choices"], "Changed inherited choices")
    return root


def _controls(parent):
    refs = source.control_models(parent)
    c.require(refs == parent["controls"] and set(refs) == {
        f"{family}_{fit}" for family in ("anchor", "specialist", "lowrate") for fit in c.FITS}, "Incomplete fixed controls")
    c.require(all(refs[f"{family}_{fit}"]["source_fit"] == fit
        for family in ("anchor", "specialist", "lowrate") for fit in c.FITS), "Wrong fixed-control fit")
    models = {key: prior.StrongPolicyInference(c.verify_hash(ref)).model for key, ref in refs.items()}
    for model in models.values():
        model.eval()
    return models


def _choices(totals, missing, proposal, inherited, fixed):
    choices = prior._choices(totals, missing, proposal["offered_group_scores"], inherited,
        fixed["specialist"]["offered_group_scores"], fixed["lowrate"]["offered_group_scores"])
    n = len(totals)
    values = fixed["finalrate"]["offered_group_scores"]
    c.require(len(values) == n and all(math.isfinite(v) for v in values), "Invalid fixed-final logits")
    outside = max(range(3, n), key=lambda j: values[j]) if n > 3 else None
    choices["finalrate_outsider"] = outside
    choices["finalrate_supported"] = prior._best(totals, missing, list(range(min(n, 3)))+([] if outside is None else [outside]))
    return choices


def _independent_choices(seal, proposal, fixed):
    choices = prior._independent_choices(seal, proposal, fixed["specialist"], fixed["lowrate"])
    n = len(seal["identity"]["candidates"]["keys"])
    totals, missing = prior._independent_columns(seal["old_selection"]["worlds"], n)
    values = fixed["finalrate"]["offered_group_scores"]
    outside = min(range(3, n), key=lambda j: (-values[j], j)) if n > 3 else None
    choices["finalrate_outsider"] = outside
    choices["finalrate_supported"] = prior._independent_best(totals, missing,
        list(range(min(n, 3)))+([] if outside is None else [outside]))
    return choices


def _row(spec, choices, worlds):
    value = prior._row(spec, choices, worlds)
    for name, left, right in statistics.CONTRASTS:
        if name not in value["delta_bounds"]:
            bounds = value["arm_score_bounds"]
            value["delta_bounds"][name] = [0., 0.] if choices[left] == choices[right] else [
                bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
    return value


def _independent_row(spec, choices, worlds):
    n = len(worlds[0]["outcomes"])
    totals, missing = prior._independent_columns(worlds, n)
    bounds, scores = {}, {}
    c.require(set(choices) == set(statistics.ARMS), "Missing independently reconstructed arm")
    for arm, chosen in choices.items():
        absent = n <= 3 and (arm.endswith("_outsider") or arm == "original_fourth")
        c.require((chosen is None) == absent, "Invalid independent absent arm")
        if chosen is None:
            bounds[arm] = scores[arm] = None
        else:
            c.require(type(chosen) is int and 0 <= chosen < n, "Invalid independent choice")
            bounds[arm] = [(128+totals[chosen]+sign*missing[chosen])/256 for sign in (-1, 1)]
            scores[arm] = bounds[arm][0] if not missing[chosen] else None
    differences = {}
    for name, left, right in statistics.CONTRASTS:
        a, b = choices[left], choices[right]
        differences[name] = [0., 0.] if a == b else [
            (totals[a]-totals[b]+sign*(missing[a]+missing[b]))/256 for sign in (-1, 1)]
    return {**{key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "split")},
        "privacy": spec["privacy"], "support": n, "outsider_available": n > 3, "choices": choices,
        "arm_scores": scores, "arm_score_bounds": bounds, "delta_bounds": differences, "unknown_counts": missing}


def _validation_report(rows, models, refs, base, *, audit=False):
    c.require(rows and all(row["split"] == "validation" and row["source_fit"] in c.FITS
        and not {"assessment", "assessment_outcomes", "fresh_outcomes", "fresh_means"} & row.keys() for row in rows)
        and len({row["id"] for row in rows}) == len(rows), "Invalid full old-target validation corpus")
    if Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        c.require(len(rows) == c.CONFIG["full_validation_rows"], "Incomplete full validation corpus")
    records, assignments = [], {}
    for fit in c.FITS:
        selected = [row for row in rows if row["source_fit"] == fit]
        if not selected:
            continue
        assignments[str(fit)] = {"model": refs[str(fit)], "states": len(selected)}
        ordered = sorted(selected, key=lambda row: len(row["features"]))
        model = models[fit]
        model.eval()
        with torch.no_grad():
            for start in range(0, len(ordered), 64):
                c.budget_boundary(base)
                batch = ordered[start:start+64]
                tensors = inherited_learning.audit_batch_tensors(batch) if audit else inherited_learning.collate(batch)
                logits, _ = model(*tensors)
                losses, choices = (inherited_learning.audit_objective if audit else inherited_learning.objective)(logits, batch)
                for row, loss, choice in zip(batch, losses.tolist(), choices, strict=True):
                    sums = row["old_sums_wdl"]
                    support = len(sums)-min(3, len(sums))
                    target = min(range(3, len(sums)), key=lambda j: (-sums[j], j)) if support else -1
                    c.require(target == row["outsider_teacher"] and math.isfinite(loss), "Invalid old validation target/loss")
                    regret = (max(sums[3:])-sums[choice] if audit else sums[target]-sums[choice]) if support else 0
                    records.append({"id": row["id"], "pair": row["pair"], "index": row["index"],
                        "source_fit": row["source_fit"], "support": len(sums), "outsider_support": support,
                        "choice": choice, "target": target, "first_winner": choice == target if support else None,
                        "any_old_max": regret == 0 if support else None, "old_regret_units": regret, "loss": loss})
    records.sort(key=lambda row: row["id"])
    pair_ids = sorted({row["pair"] for row in records})
    reduce = math.fsum if audit else sum
    pair_loss, pair_regret = [], []
    for pair in pair_ids:
        values = [row for row in records if row["pair"] == pair]
        pair_loss.append(reduce(row["loss"] for row in values)/len(values))
        pair_regret.append(reduce(row["old_regret_units"] for row in values)/(256*len(values)))
    return {"states": len(records), "rows": records, "matched_source_fit": True, "models": assignments,
        "metrics": inherited_learning._metrics(records, audit=audit),
        "outsider_support_counts": {"empty": sum(row["outsider_support"] == 0 for row in records),
            "singleton": sum(row["outsider_support"] == 1 for row in records), "multiple": sum(row["outsider_support"] > 1 for row in records)},
        "pair_weighted": {"pairs": len(pair_ids), "mean_loss": reduce(pair_loss)/len(pair_ids),
            "mean_old_score_regret": reduce(pair_regret)/len(pair_ids)},
        "old_targets_only": True, "checkpoint_selection": False}


def _full_summary(reports):
    return {key: {name: value for name, value in report.items() if name != "rows"} for key, report in reports.items()}


def _parent_records(parent):
    # Called only after every new checkpoint and full-validation report is sealed.
    previous, predictions = c.verified(parent["assessment"]), c.verified(parent["predictions"])
    validation = c.verified(parent["validation_old_targets"])
    lowrate = {key: c.verified(parent["lowrate_evidence"][key]) for key in ("assessment", "predictions", "validation_old_targets")}
    c.require(previous["manifest"] == predictions["manifest"] == validation["manifest"] == c.PARENT_BINDING,
              "Wrong completed higher-rate endpoint")
    c.require(lowrate["assessment"]["manifest"] == lowrate["predictions"]["manifest"]
              == lowrate["validation_old_targets"]["manifest"] != c.PARENT_BINDING, "Wrong completed low-rate endpoint")
    return previous, predictions, validation, lowrate


def _check_fixed(row, fixed, previous, previous_prediction, old_lowrate, old_low_prediction):
    c.require(row["index"] == previous["index"] == old_lowrate["index"]
              and row["fit"] == previous["fit"] == old_lowrate["fit"], "Fixed control source assignment changed")
    for family, old_family, reference, prediction, proposal_name in (
            ("finalrate", "focused", previous, previous_prediction, "proposal"),
            ("specialist", "specialist", previous, previous_prediction, "specialist_proposal"),
            ("lowrate", "focused", old_lowrate, old_low_prediction, "proposal")):
        prior.compare(fixed[family], prediction[proposal_name])
        for role in ("outsider", "supported"):
            c.require(prediction["choices"][f"{old_family}_{role}"] == reference["choices"][f"{old_family}_{role}"],
                      "Completed control prediction and score choices disagree")
            for field in ("choices", "arm_scores", "arm_score_bounds"):
                c.require(row[field][f"{family}_{role}"] == reference[field][f"{old_family}_{role}"],
                          "Fixed control choices or assessed scores changed")


def _check_final(row, previous):
    for key in previous:
        if key in ("choices", "arm_scores", "arm_score_bounds", "delta_bounds"):
            c.require({name: row[key][name] for name in previous[key]} == previous[key], "Final endpoint score/choice changed")
        else:
            c.require(row[key] == previous[key], "Final endpoint source metadata changed")


def _check_full_endpoints(reports, previous, lowrate):
    for update, family in (("0", "anchor"), ("1784", "focused")):
        for name in ("rows", "metrics", "pair_weighted"):
            prior.compare(reports[update][name], previous["families"][family][name], rel_tol=0., abs_tol=1e-12)
    for name in ("rows", "metrics", "pair_weighted"):
        prior.compare(previous["families"]["lowrate"][name], lowrate["families"]["focused"][name], rel_tol=0., abs_tol=1e-12)


def assess(binding, parent, base=OUTPUT, progress=None):
    base = Path(base)
    c.budget_boundary(base)
    c.require(not (base/"predictions_lock.json").exists() and not (base/"assessment_result.json").exists(), "Preserve saved-curve evidence")
    rows, inputs, descriptors = _panel(parent, base)
    validation_rows = source.validation_rows(parent)
    controls = _controls(parent)
    final_models = {fit: _checkpoint_model(parent, 1784, fit) for fit in c.FITS}
    old_data, fixed_predictions = [], []
    for row, item in zip(rows, inputs):
        c.budget_boundary(base)
        seal, totals, missing = prior._old_input(item)
        c.require(not any(missing), "Unresolved old labels")
        old_data.append((seal, totals, missing))
        fit = row["source_fit"]
        fixed_predictions.append({family: prior._predict(row, final_models[fit] if family == "finalrate" else controls[f"{family}_{fit}"])
            for family in ("specialist", "lowrate", "finalrate")})
    predictions, labels, full_reports = {}, {}, {}
    for update in c.CHECKPOINTS:
        models = {fit: _checkpoint_model(parent, update, fit) for fit in c.FITS}
        saved, records = [], []
        for row, item, (seal, totals, missing), fixed in zip(rows, inputs, old_data, fixed_predictions):
            c.budget_boundary(base)
            proposal = prior._predict(row, models[row["source_fit"]])
            choices = _choices(totals, missing, proposal, item["old_choices"], fixed)
            prediction = _prediction(item, row, parent, update, proposal, choices)
            if update == 0:
                _initial_endpoint(seal, prediction)
            saved.append(prediction)
            records.append(_label_record(row, totals, proposal, choices))
        predictions[str(update)], labels[str(update)] = saved, _label_metrics(records)
        full_reports[str(update)] = _validation_report(validation_rows, models, parent["models"][str(update)], base)
        _progress(progress, "checkpoint_predictions", update=update, states=len(saved), full_validation_states=len(validation_rows), total_checkpoints=len(c.CHECKPOINTS))
    full_path = base/"full_validation_old_labels.json"
    old.save_new(full_path, {"manifest": binding, "reports": full_reports, "complete": True})
    full_ref = c.artifact(full_path)
    lock = {"manifest": binding, "parent_binding": c.PARENT_BINDING, "parent_digest": old.digest(parent),
        "models": parent["models"], "controls": parent["controls"], "fixed_predictions": fixed_predictions,
        "full_validation_old_labels_artifact": full_ref,
        "checkpoints": list(c.CHECKPOINTS), "states_per_checkpoint": len(rows), "predictions": predictions,
        "old_label_metrics": labels, "complete": True, "assessment_used_for_selection": False,
        "all_predictions_precede_assessment_open": True, "created_at_ns": time.time_ns(), "time": old.now()}
    path = base/"predictions_lock.json"
    old.save_new(path, lock)
    lock_ref = c.artifact(path)
    assessment_opened_at_ns = time.time_ns()
    previous, previous_predictions, previous_validation, lowrate = _parent_records(parent)
    _check_full_endpoints(full_reports, previous_validation, lowrate["validation_old_targets"])
    curves = {str(update): [] for update in c.CHECKPOINTS}
    for i, (item, descriptor) in enumerate(zip(inputs, descriptors)):
        c.budget_boundary(base)
        root = _open_outcomes(descriptor, item, lock_ref, base)
        for update in c.CHECKPOINTS:
            scored = _row(item["spec"], predictions[str(update)][i]["choices"], root["assessment"]["worlds"])
            _check_fixed(scored, fixed_predictions[i], previous["rows"][i], previous_predictions["predictions"][i],
                         lowrate["assessment"]["rows"][i], lowrate["predictions"]["predictions"][i])
            for name in prior.st.CONTEXT:
                c.require(scored["delta_bounds"][name] == root["summary"]["delta_bounds"][name], "Changed inherited control")
            if update == 1784:
                _check_final(scored, previous["rows"][i])
            curves[str(update)].append(scored)
        if (i+1) % 64 == 0 or i+1 == len(inputs):
            _progress(progress, "saved_outcome_arithmetic", states=i+1, total=len(inputs))
    result = {"manifest": binding, "parent_binding": c.PARENT_BINDING, "parent_digest": old.digest(parent),
        "predictions": lock_ref, "curves": curves, "old_label_metrics": labels,
        "full_validation_old_labels": _full_summary(full_reports), "full_validation_old_labels_artifact": full_ref,
        "statistics": statistics.summarize(curves, boundary=lambda: c.budget_boundary(base)),
        "assessment_opened_at_ns": assessment_opened_at_ns, "states_per_checkpoint": len(rows),
        "endpoint_zero_reproduced": True, "endpoint_final_reproduced": True, "fixed_controls_reproduced": True,
        "full_validation_endpoints_reproduced": True, "old_context_controls_reproduced": True,
        "new_simulations": 0, "new_training": False, "checkpoint_selection": False, "best_fit_selection": False,
        "promotion": False, "assessment_reused_development": True, "assessment_used_for_selection": False, "time": old.now()}
    path = base/"assessment_result.json"
    old.save_new(path, result)
    return c.artifact(path)

def _independent_label_record(row, seal, proposal, choices):
    count = len(seal["identity"]["candidates"]["keys"])
    totals, missing = prior._independent_columns(seal["old_selection"]["worlds"], count)
    c.require(not any(missing), "Unresolved old label in independent curve metrics")
    support = count-min(3, count)
    target = min(range(3, count), key=lambda j: (-totals[j], j)) if support else None
    chosen = choices["focused_outsider"]
    c.require((chosen is None) == (support == 0), "Incorrect independent metric denominator")
    if support < 2:
        loss = 0.
    else:
        values = proposal["offered_group_scores"][3:]
        target_logit = proposal["offered_group_scores"][target]
        # Separate log-sum-exp calculation in double precision relative to target.
        differences = [value-target_logit for value in values]
        offset = max(differences)
        loss = math.log(math.fsum(math.exp(value-offset) for value in differences))+offset
    regret = max(totals[3:])-totals[chosen] if support else 0
    return {"index": row["index"], "id": row["id"], "fit": row["source_fit"], "split": row["split"],
            "outsider_support": support, "choice": chosen, "target": target, "loss": loss,
            "first_winner": chosen == target if support else None,
            "any_old_max": totals[chosen] == totals[target] if support else None, "old_regret_units": regret}


def _independent_label_metrics(records):
    groups = {}
    for split in ("train", "validation"):
        selected = [record for record in records if record["split"] == split]
        metrics = {}
        for label, lower in (("all_rows", 0), ("nonempty_outsiders", 1), ("multiple_outsiders", 2)):
            indices = [i for i, record in enumerate(selected) if record["outsider_support"] >= lower]
            denominator = sum(selected[i]["outsider_support"] > 0 for i in indices)
            first = sum(selected[i]["first_winner"] is True for i in indices)
            tied = sum(selected[i]["any_old_max"] is True for i in indices)
            regrets = sum(int(selected[i]["old_regret_units"]) for i in indices)
            metrics[label] = {"rows": len(indices), "accuracy_denominator": denominator,
                "mean_cross_entropy": math.fsum(selected[i]["loss"] for i in indices)/len(indices) if indices else None,
                "first_winner_count": first, "tied_maximum_count": tied,
                "first_winner_accuracy": first/denominator if denominator else None,
                "tied_maximum_accuracy": tied/denominator if denominator else None,
                "old_regret_units_sum": regrets, "mean_old_score_regret": regrets/(256*len(indices)) if indices else None}
        empty = sum(row["outsider_support"] == 0 for row in selected)
        singleton = sum(row["outsider_support"] == 1 for row in selected)
        groups[split] = {"matched_source_fit": True, "states": len(selected), "metrics": metrics,
                         "outsider_support_counts": {"empty": empty, "singleton": singleton,
                                                     "multiple": len(selected)-empty-singleton}}
    return {"rows": records, "groups": groups, "old_targets_only": True,
            "no_checkpoint_selection": True, "not_a_generalization_or_utility_estimate": True}


def audit(binding, parent, result_ref, base=OUTPUT, progress=None):
    base = Path(base)
    c.budget_boundary(base)
    c.require(not (base/"assessment_audit.json").exists(), "Preserve independent curve audit")
    result = c.verified(result_ref)
    c.require(result["manifest"] == binding and result["parent_binding"] == c.PARENT_BINDING
              and result["parent_digest"] == old.digest(parent), "Wrong curve identity")
    for key, expected in (("new_simulations", 0), ("new_training", False), ("checkpoint_selection", False),
        ("best_fit_selection", False), ("promotion", False), ("assessment_reused_development", True),
        ("assessment_used_for_selection", False), ("endpoint_zero_reproduced", True),
        ("endpoint_final_reproduced", True), ("old_context_controls_reproduced", True),
        ("fixed_controls_reproduced", True), ("full_validation_endpoints_reproduced", True)):
        c.require(result[key] == expected, "Changed curve scope")
    lock = c.verified(result["predictions"])
    full = c.verified(result["full_validation_old_labels_artifact"])
    c.require(full["manifest"] == binding and full["complete"] is True
              and lock["full_validation_old_labels_artifact"] == result["full_validation_old_labels_artifact"], "Full validation seal changed")
    c.require(lock["manifest"] == binding and lock["parent_binding"] == c.PARENT_BINDING
        and lock["parent_digest"] == old.digest(parent) and lock["models"] == parent["models"]
        and lock["controls"] == parent["controls"] and lock["checkpoints"] == list(c.CHECKPOINTS)
        and lock["complete"] is True and lock["assessment_used_for_selection"] is False
        and lock["all_predictions_precede_assessment_open"] is True
        and c.verify_hash(result["full_validation_old_labels_artifact"]).stat().st_mtime_ns <= lock["created_at_ns"]
        <= c.verify_hash(result["predictions"]).stat().st_mtime_ns <= result["assessment_opened_at_ns"], "Global prediction firewall failed")
    rows, inputs, descriptors = _panel(parent, base)
    validation_rows = source.validation_rows(parent)
    keys = {str(update) for update in c.CHECKPOINTS}
    c.require(lock["states_per_checkpoint"] == result["states_per_checkpoint"] == len(rows)
        and len(lock["fixed_predictions"]) == len(rows), "Changed population size")
    c.require(set(lock["predictions"]) == set(result["curves"]) == set(result["old_label_metrics"])
        == set(lock["old_label_metrics"]) == set(full["reports"]) == set(result["full_validation_old_labels"]) == keys,
        "Incomplete checkpoint or validation population")
    c.require(all(len(lock["predictions"][key]) == len(result["curves"][key]) == len(rows) for key in keys), "Missing checkpoint predictions")
    models = {}
    for update in c.CHECKPOINTS:
        c.budget_boundary(base)
        models[update] = {fit: _checkpoint_model(parent, update, fit) for fit in c.FITS}
    controls = _controls(parent)
    previous, previous_predictions, previous_validation, lowrate = _parent_records(parent)
    curves, labels = ({str(update): [] for update in c.CHECKPOINTS} for _ in range(2))
    for i, (row, item, descriptor) in enumerate(zip(rows, inputs, descriptors)):
        c.budget_boundary(base)
        seal = c.verified(item["selection_artifact"])
        c.require(seal["manifest"] == prior.c.SOURCE_BINDING and seal["index"] == i
            and seal["selection_lock"]["identity_digest"] == old.digest(seal["identity"])
            and seal["selection_lock"]["old_selection_sha256"] == old.digest(seal["old_selection"]), "Changed old selection")
        for key in ("spec", "snapshot", "candidates"):
            c.require(old.digest(seal["identity"][key]) == old.digest(item[key]), "Changed old input")
        state = old.deserialize_state_snapshot(item["snapshot"])
        alternate = prior.input_audit.hidden_alternative(state)
        fixed = {}
        for family in ("specialist", "lowrate", "finalrate"):
            model = models[1784][row["source_fit"]] if family == "finalrate" else controls[f"{family}_{row['source_fit']}"]
            fixed[family] = prior._independent_proposal(row, item, model, state=state)
            prior.compare(fixed[family], prior._independent_proposal(row, item, model, state=alternate))
        prior.compare(lock["fixed_predictions"][i], fixed)
        root, receipt = c.verified(descriptor["root"]), c.verified(descriptor["audit"])
        c.require(root["manifest"] == receipt["manifest"] == prior.c.SOURCE_BINDING and receipt["passed"] is True
            and prior._same_ref(receipt["root"], descriptor["root"]) and "failure" not in root
            and root["stage"] == "main" and old.digest(root["spec"]) == old.digest(item["spec"])
            and root["selection_lock"] == seal["selection_lock"]
            and prior._same_ref(root["selection_artifact"], item["selection_artifact"]), "Invalid independent assessment source")
        block = root["assessment"]
        seeds = [2211000000+512*i+j for j in range(128)]
        c.require(block["candidate_indices"] == list(range(len(item["candidates"]["keys"])))
            and block["world_seeds"] == [world["seed"] for world in block["worlds"]] == seeds, "Wrong independent world mapping")
        for update in c.CHECKPOINTS:
            c.budget_boundary(base)
            saved = lock["predictions"][str(update)][i]
            c.require(saved["index"] == i and saved["fit"] == row["source_fit"] and saved["update"] == update
                and saved["row_id"] == row["id"] and saved["source_selection"] == item["selection_artifact"]
                and saved["model"] == parent["models"][str(update)][str(row["source_fit"])], "Wrong checkpoint mapping")
            model = models[update][row["source_fit"]]
            proposal = prior._independent_proposal(row, item, model, state=state)
            prior.compare(saved["proposal"], proposal)
            prior.compare(proposal, prior._independent_proposal(row, item, model, state=alternate))
            choices = _independent_choices(seal, proposal, fixed)
            c.require(choices == saved["choices"], "Changed independently reconstructed choice")
            independent = _independent_row(item["spec"], choices, block["worlds"])
            c.require(independent == result["curves"][str(update)][i], "Curve arithmetic differs")
            _check_fixed(independent, fixed, previous["rows"][i], previous_predictions["predictions"][i],
                lowrate["assessment"]["rows"][i], lowrate["predictions"]["predictions"][i])
            for name in prior.st.CONTEXT:
                c.require(independent["delta_bounds"][name] == receipt["row"]["delta_bounds"][name], "Changed inherited context")
            if update == 0:
                original = seal["identity"]["proposals"]["anchor"]
                prior.compare(proposal["legal_logits"], original["learned_logits"])
                prior.compare(proposal["offered_group_scores"], original["offered_group_scores"])
                c.require(choices["focused_outsider"] == choices["anchor_outsider"]
                    and choices["focused_supported"] == choices["anchor"], "Initializer endpoint differs")
            if update == 1784:
                prior.compare(proposal, previous_predictions["predictions"][i]["proposal"])
                _check_final(independent, previous["rows"][i])
            curves[str(update)].append(independent)
            labels[str(update)].append(_independent_label_record(row, seal, proposal, choices))
        if (i+1) % 32 == 0 or i+1 == len(rows):
            _progress(progress, "independent_curve_inference", states=i+1, total=len(rows), checkpoints=len(c.CHECKPOINTS))
    label_metrics = {key: _independent_label_metrics(records) for key, records in labels.items()}
    prior.compare(lock["old_label_metrics"], label_metrics, rel_tol=0., abs_tol=1e-12)
    c.require(result["old_label_metrics"] == lock["old_label_metrics"], "Postseal label changes")
    full_reports = {}
    for update in c.CHECKPOINTS:
        full_reports[str(update)] = _validation_report(validation_rows, models[update], parent["models"][str(update)], base, audit=True)
        prior.compare(full_reports[str(update)], full["reports"][str(update)], rel_tol=0., abs_tol=1e-12)
        _progress(progress, "independent_full_validation", update=update, states=len(validation_rows))
    prior.compare(result["full_validation_old_labels"], _full_summary(full_reports), rel_tol=0., abs_tol=1e-12)
    _check_full_endpoints(full_reports, previous_validation, lowrate["validation_old_targets"])
    independent_statistics = statistics.independent_summarize(curves, boundary=lambda: c.budget_boundary(base))
    prior.compare(result["statistics"], independent_statistics, rel_tol=0., abs_tol=1e-12)
    receipt = {"manifest": binding, "parent_digest": old.digest(parent), "result": result_ref,
        "predictions": result["predictions"], "passed": True, "checkpoints": list(c.CHECKPOINTS),
        "states_per_checkpoint": len(rows), "new_inference_independently_reproduced": True,
        "lawful_inputs_and_private_invariance_pass": True, "old_choices_independently_reconstructed": True,
        "old_label_metrics_independently_reproduced": True, "all_new_arithmetic_pass": True,
        "full_validation_old_label_metrics_independently_reproduced": True, "fixed_controls_reproduced": True,
        "endpoint_zero_reproduced": True, "endpoint_final_reproduced": True,
        "independent_statistics": independent_statistics, "unknown_outcomes": sum(sum(row["unknown_counts"]) for row in curves["0"]),
        "new_training": False, "new_simulations": 0, "new_trajectory_replays": 0,
        "historical_training_and_simulation_audits_inherited": True, "time": old.now()}
    path = base/"assessment_audit.json"
    old.save_new(path, receipt)
    return c.artifact(path)
