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
from tools.focused_teacher_fit_128 import assessment as prior

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


def assess(binding, parent, base=OUTPUT, progress=None):
    base = Path(base)
    c.budget_boundary(base)
    c.require(not (base/"predictions_lock.json").exists() and not (base/"assessment_result.json").exists(),
              "Preserve saved-curve evidence")
    rows, inputs, descriptors = _panel(parent, base)
    old_data = []
    for item in inputs:
        c.budget_boundary(base)
        seal, totals, missing = prior._old_input(item)
        c.require(sum(missing) == 0, "Old-target curve requires resolved source labels")
        old_data.append((seal, totals, missing))
    predictions, labels = {}, {}
    for update in c.CHECKPOINTS:
        c.budget_boundary(base)
        models = {fit: _checkpoint_model(parent, update, fit) for fit in c.FITS}
        saved, records = [], []
        for row, item, (seal, totals, missing) in zip(rows, inputs, old_data):
            c.budget_boundary(base)
            proposal = prior._predict(row, models[row["source_fit"]])
            choices = prior._choices(totals, missing, proposal["offered_group_scores"], item["old_choices"])
            prediction = _prediction(item, row, parent, update, proposal, choices)
            if update == 0:
                _initial_endpoint(seal, prediction)
            saved.append(prediction)
            records.append(_label_record(row, totals, proposal, choices))
        predictions[str(update)], labels[str(update)] = saved, _label_metrics(records)
        _progress(progress, "checkpoint_predictions", update=update, states=len(saved), total_checkpoints=len(c.CHECKPOINTS))
    lock = {"manifest": binding, "parent_binding": c.PARENT_BINDING, "parent_digest": old.digest(parent),
            "models": parent["models"], "checkpoints": list(c.CHECKPOINTS), "states_per_checkpoint": len(rows),
            "predictions": predictions, "old_label_metrics": labels, "complete": True,
            "assessment_used_for_selection": False, "all_predictions_precede_assessment_open": True,
            "created_at_ns": time.time_ns(), "time": old.now()}
    path = base/"predictions_lock.json"
    old.save_new(path, lock)
    lock_ref = c.artifact(path)
    # Authenticate the complete lock once; individual access guards rehash it.
    sealed = c.verified(lock_ref)
    c.require(sealed["complete"] and set(sealed["predictions"]) == {str(u) for u in c.CHECKPOINTS}
              and all(len(v) == len(rows) for v in sealed["predictions"].values()), "Incomplete global prediction seal")
    assessment_opened_at_ns = time.time_ns()
    previous = c.verified(parent["assessment"])
    previous_predictions = c.verified(parent["predictions"])
    c.require(previous["manifest"] == previous_predictions["manifest"] == c.PARENT_BINDING,
              "Wrong previous fixed-final endpoint")
    curves = {str(update): [] for update in c.CHECKPOINTS}
    for i, (item, descriptor) in enumerate(zip(inputs, descriptors)):
        c.budget_boundary(base)
        root = _open_outcomes(descriptor, item, lock_ref, base)
        prior.compare(predictions["1784"][i]["proposal"], previous_predictions["predictions"][i]["proposal"])
        c.require(predictions["1784"][i]["choices"] == previous_predictions["predictions"][i]["choices"],
                  "Final endpoint changed saved choices")
        for update in c.CHECKPOINTS:
            scored = prior._row(item["spec"], predictions[str(update)][i]["choices"], root["assessment"]["worlds"])
            for name in prior.st.CONTEXT:
                c.require(scored["delta_bounds"][name] == root["summary"]["delta_bounds"][name], "Changed inherited control")
            if update == 0:
                c.require(all(scored["delta_bounds"][name] == [0., 0.] for name in c.CONFIG["primary_contrasts"]), "Nonzero initializer effect")
            if update == 1784:
                c.require(scored == previous["rows"][i], "Final endpoint differs from previous result")
            curves[str(update)].append(scored)
        if (i+1) % 64 == 0 or i+1 == len(inputs):
            _progress(progress, "saved_outcome_arithmetic", states=i+1, total=len(inputs))
    result = {"manifest": binding, "parent_binding": c.PARENT_BINDING, "parent_digest": old.digest(parent),
              "predictions": lock_ref, "curves": curves, "old_label_metrics": labels,
              "statistics": statistics.summarize(curves, boundary=lambda: c.budget_boundary(base)),
              "assessment_opened_at_ns": assessment_opened_at_ns, "states_per_checkpoint": len(rows),
              "endpoint_zero_reproduced": True, "endpoint_final_reproduced": True,
              "old_context_controls_reproduced": True, "new_simulations": 0, "new_training": False,
              "checkpoint_selection": False, "best_fit_selection": False, "promotion": False,
              "assessment_reused_development": True, "assessment_used_for_selection": False,
              "time": old.now()}
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
              and result["parent_digest"] == old.digest(parent), "Wrong curve closure identity")
    for key, expected in (("new_simulations", 0), ("new_training", False), ("checkpoint_selection", False),
                          ("best_fit_selection", False), ("promotion", False), ("assessment_reused_development", True),
                          ("assessment_used_for_selection", False), ("endpoint_zero_reproduced", True),
                          ("endpoint_final_reproduced", True), ("old_context_controls_reproduced", True)):
        c.require(result[key] == expected, "Changed saved-curve scope")
    lock = c.verified(result["predictions"])
    c.require(lock["manifest"] == binding and lock["parent_binding"] == c.PARENT_BINDING
              and lock["parent_digest"] == old.digest(parent) and lock["models"] == parent["models"]
              and lock["checkpoints"] == list(c.CHECKPOINTS) and lock["complete"] is True
              and lock["assessment_used_for_selection"] is False and lock["all_predictions_precede_assessment_open"] is True
              and lock["created_at_ns"] <= c.verify_hash(result["predictions"]).stat().st_mtime_ns
              <= result["assessment_opened_at_ns"], "Global prediction firewall failed")
    rows, inputs, descriptors = _panel(parent, base)
    c.require(lock["states_per_checkpoint"] == result["states_per_checkpoint"] == len(rows), "Changed assessment denominator")
    updates = {str(update) for update in c.CHECKPOINTS}
    c.require(set(lock["predictions"]) == set(result["curves"]) == set(result["old_label_metrics"])
              == set(lock["old_label_metrics"]) == updates, "Missing or extra checkpoint")
    c.require(all(len(lock["predictions"][key]) == len(result["curves"][key]) == len(rows) for key in updates),
              "Incomplete checkpoint population")
    # Different model instances from collection, retained so each outcome file is opened once.
    models = {}
    for update in c.CHECKPOINTS:
        c.budget_boundary(base)
        models[update] = {fit: _checkpoint_model(parent, update, fit) for fit in c.FITS}
    previous = c.verified(parent["assessment"])
    previous_predictions = c.verified(parent["predictions"])
    c.require(previous["manifest"] == previous_predictions["manifest"] == c.PARENT_BINDING, "Wrong audited previous endpoint")
    curves, labels = ({str(update): [] for update in c.CHECKPOINTS} for _ in range(2))
    for i, (row, item, descriptor) in enumerate(zip(rows, inputs, descriptors)):
        c.budget_boundary(base)
        seal = c.verified(item["selection_artifact"])
        c.require(seal["manifest"] == prior.c.SOURCE_BINDING and seal["index"] == i
                  and seal["selection_lock"]["identity_digest"] == old.digest(seal["identity"])
                  and seal["selection_lock"]["old_selection_sha256"] == old.digest(seal["old_selection"]),
                  "Invalid inherited pre-assessment seal")
        for key in ("spec", "snapshot", "candidates"):
            c.require(old.digest(seal["identity"][key]) == old.digest(item[key]), "Changed old inputs")
        root, receipt = c.verified(descriptor["root"]), c.verified(descriptor["audit"])
        c.require(root["manifest"] == receipt["manifest"] == prior.c.SOURCE_BINDING and receipt["passed"] is True
                  and prior._same_ref(receipt["root"], descriptor["root"]) and "failure" not in root
                  and root["stage"] == "main" and old.digest(root["spec"]) == old.digest(item["spec"])
                  and root["selection_lock"] == seal["selection_lock"]
                  and prior._same_ref(root["selection_artifact"], item["selection_artifact"]), "Invalid audited outcome source")
        block = root["assessment"]
        seeds = [2211000000+512*i+j for j in range(128)]
        c.require(block["candidate_indices"] == list(range(len(item["candidates"]["keys"])))
                  and block["world_seeds"] == [world["seed"] for world in block["worlds"]] == seeds,
                  "Changed independent outcome mapping")
        state = old.deserialize_state_snapshot(item["snapshot"])
        alternate = prior.input_audit.hidden_alternative(state)
        for update in c.CHECKPOINTS:
            c.budget_boundary(base)
            saved = lock["predictions"][str(update)][i]
            c.require(saved["index"] == i and saved["fit"] == row["source_fit"] and saved["update"] == update
                      and saved["row_id"] == row["id"] and saved["source_selection"] == item["selection_artifact"]
                      and saved["model"] == parent["models"][str(update)][str(row["source_fit"])], "Wrong locked checkpoint mapping")
            model = models[update][row["source_fit"]]
            proposal = prior._independent_proposal(row, item, model, state=state)
            prior.compare(saved["proposal"], proposal)
            prior.compare(proposal, prior._independent_proposal(row, item, model, state=alternate))
            choices = prior._independent_choices(seal, proposal)
            c.require(choices == saved["choices"], "Changed independently reconstructed choice")
            independent = prior._independent_row(item["spec"], choices, block["worlds"])
            c.require(independent == result["curves"][str(update)][i], "Saved-curve raw arithmetic differs")
            for name in prior.st.CONTEXT:
                c.require(independent["delta_bounds"][name] == receipt["row"]["delta_bounds"][name], "Changed audited context score")
            if update == 0:
                original = seal["identity"]["proposals"]["anchor"]
                prior.compare(proposal["legal_logits"], original["learned_logits"])
                prior.compare(proposal["offered_group_scores"], original["offered_group_scores"])
                c.require(choices["focused_outsider"] == choices["anchor_outsider"]
                          and choices["focused_supported"] == choices["anchor"]
                          and all(independent["delta_bounds"][name] == [0., 0.] for name in c.CONFIG["primary_contrasts"]),
                          "Initializer endpoint identity failed")
            if update == 1784:
                prior.compare(proposal, previous_predictions["predictions"][i]["proposal"])
                c.require(choices == previous_predictions["predictions"][i]["choices"]
                          and independent == previous["rows"][i], "Final endpoint identity failed")
            curves[str(update)].append(independent)
            labels[str(update)].append(_independent_label_record(row, seal, proposal, choices))
        if (i+1) % 32 == 0 or i+1 == len(rows):
            _progress(progress, "independent_curve_inference", states=i+1, total=len(rows), checkpoints=len(c.CHECKPOINTS))
    label_metrics = {key: _independent_label_metrics(records) for key, records in labels.items()}
    prior.compare(lock["old_label_metrics"], label_metrics, rel_tol=1e-6, abs_tol=1e-7)
    c.require(result["old_label_metrics"] == lock["old_label_metrics"], "Post-seal old-label metric changes")
    independent_statistics = statistics.independent_summarize(curves, boundary=lambda: c.budget_boundary(base))
    prior.compare(result["statistics"], independent_statistics, rel_tol=0., abs_tol=1e-12)
    receipt = {"manifest": binding, "parent_digest": old.digest(parent), "result": result_ref,
               "predictions": result["predictions"], "passed": True, "checkpoints": list(c.CHECKPOINTS),
               "states_per_checkpoint": len(rows), "new_inference_independently_reproduced": True,
               "lawful_inputs_and_private_invariance_pass": True, "old_choices_independently_reconstructed": True,
               "old_label_metrics_independently_reproduced": True, "all_new_arithmetic_pass": True,
               "endpoint_zero_reproduced": True, "endpoint_final_reproduced": True,
               "independent_statistics": independent_statistics, "unknown_outcomes": sum(sum(r["unknown_counts"]) for r in curves["0"]),
               "new_training": False, "new_simulations": 0, "new_trajectory_replays": 0,
               "historical_training_and_simulation_audits_inherited": True, "time": old.now()}
    path = base/"assessment_audit.json"
    old.save_new(path, receipt)
    return c.artifact(path)
