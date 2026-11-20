"""Full-corpus outsider fitting with TRAIN-only diagnostic checkpoints.

The independent path reconstructs batches, canonical membership, loss, metrics
and complete optimizer trajectories while sharing mature Torch primitives.
The report helper also supports explicitly labelled post-seal old-target
validation diagnostics; the fitting entry points never load validation rows.
"""
from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from splendor_ai.alphazero.strong_policy import StrongPolicyInference, save_model, MODEL_KIND
from splendor_ai.alphazero.fast_selector_157 import collate
from scripts.analysis.verify_fast_selector_transfer_157 import audit_batch_tensors
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from . import common as c

FITS, CONFIG, OUTPUT, old = c.FITS, c.CONFIG, c.OUTPUT, c.old


def _plain(value):
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _rows_identity(rows):
    # Bound peak conversion memory to one row when authenticating the full
    # corpus in three concurrent fitting/reproduction workers.
    return old.digest({"scheme": "ordered_row_json_sha256_v1",
                       "rows": [old.digest(_plain(row)) for row in rows]})


def _validate_rows(rows):
    c.require(bool(rows), "Empty focused training corpus")
    c.require(len({row["id"] for row in rows}) == len(rows), "Duplicate training row")
    c.require(len({row["index"] for row in rows}) == len(rows), "Duplicate panel index")
    for row in rows:
        c.require(row["split"] == "train", "Only TRAIN may enter fitting")
        c.require(not {"assessment", "assessment_outcomes", "fresh_outcomes", "fresh_means"} & row.keys(),
                  "Assessment fields forbidden in training input")
        members, groups = row["members"], row["groups"]
        n = len(members)
        c.require(1 <= n <= 8 and len(row["old_sums_wdl"]) == n, "Invalid offered support")
        c.require(all(isinstance(x, int) and -128 <= x <= 128 for x in row["old_sums_wdl"]), "Invalid old sums")
        group_ids = []
        for member in members:
            c.require(len(member) > 0, "Empty canonical member")
            group = int(groups[member[0]])
            c.require(torch.equal(member, torch.where(groups == group)[0]), "Incorrect canonical membership")
            group_ids.append(group)
        c.require(len(set(group_ids)) == n, "Repeated canonical group")
        start = min(3, n)
        target = max(range(start, n), key=lambda j: row["old_sums_wdl"][j]) if start < n else -1
        c.require(row["outsider_start"] == start and row["outsider_support"] == n-start,
                  "Outsider support differs from exact first three")
        c.require(row["outsider_teacher"] == target, "Incorrect first-maximum outsider label")
        for name in ("features", "observation"):
            c.require(row[name].dtype == torch.float32 and bool(torch.isfinite(row[name]).all()), "Invalid lawful tensor")
        c.require(len(groups) == len(row["features"]), "Action group/feature mismatch")


def _inputs(fit_id, binding, base, rows, initializer, config):
    c.require(fit_id in FITS, "Unknown fit")
    fixture = any(value is not None for value in (rows, initializer, config))
    if fixture:
        c.require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture cannot target production")
        c.require(all(value is not None for value in (rows, initializer, config)), "Complete fixture inputs required")
        values, initial, settings = rows, initializer, dict(config)
    else:
        from . import source
        values = source.load_rows(binding, "train", base)
        initial, settings = source.initializer(fit_id, base), dict(CONFIG)
        c.require(len(values) == CONFIG["train_rows"], "Incomplete production TRAIN corpus")
        c.require({row["source_fit"] for row in values} == set(FITS), "Training must pool all source fits")
    _validate_rows(values)
    points = settings["checkpoints"]
    c.require(points == sorted(set(points)) and points[0] == 0 and points[-1] == settings["updates"], "Invalid checkpoints")
    c.require(settings["updates"] > 0 and settings["batch_size"] > 0, "Invalid fixed dose")
    c.verify_hash(initial)
    return values, initial, settings


def batch_plan(rows, fit_id, epoch, config=CONFIG):
    c.require(epoch >= 0 and fit_id in FITS, "Invalid batch identity")
    ordered = list(range(len(rows)))
    generator = random.Random(f"{config['batch_tag']}|{config['train_seeds'][str(fit_id)]}|{epoch}")
    generator.shuffle(ordered)
    ordered.sort(key=lambda i: len(rows[i]["features"]))
    size = config["batch_size"]
    batches = [ordered[start:start+size] for start in range(0, len(ordered), size)]
    generator.shuffle(batches)
    return batches


def independent_plan(rows, fit_id, epoch, config=CONFIG):
    indices = list(range(len(rows)))
    randomizer = random.Random("%s|%s|%s" % (config["batch_tag"], config["train_seeds"][str(fit_id)], epoch))
    randomizer.shuffle(indices)
    ordered = sorted(indices, key=lambda j: rows[j]["features"].shape[0])
    blocks = []
    for lo in range(0, len(ordered), config["batch_size"]):
        blocks.append(ordered[lo:min(len(ordered), lo+config["batch_size"])])
    randomizer.shuffle(blocks)
    return blocks


def _probe_ids(binding, base, rows, config, *, fixture):
    """Resolve the original TRAIN panel without opening any validation tensors."""
    if fixture:
        ids = set(config.get("training_probe_ids", [row["id"] for row in rows]))
    else:
        from . import source
        ids = set(source.train_probe_ids(binding, base))
        c.require(len(ids) == 168, "Original TRAIN probe must retain all 168 positions")
    c.require(bool(ids) and ids <= {row["id"] for row in rows}, "TRAIN probe is not a nonempty corpus subset")
    return ids


def objective(logits, rows):
    losses, choices = [], []
    for i, row in enumerate(rows):
        start, width, target = row["outsider_start"], row["outsider_support"], row["outsider_teacher"]
        if width < 2:
            losses.append(logits[i, row["members"][0][0]]*0.)
            choices.append(target)
        else:
            scores = torch.stack([torch.amax(logits[i, member]) for member in row["members"][start:]])
            losses.append(F.cross_entropy(scores[None], torch.tensor([target-start])))
            choices.append(start+int(scores.detach().argmax()))
    return torch.stack(losses), choices


def audit_objective(logits, rows):
    losses, choices = [], []
    for i, row in enumerate(rows):
        offered = list(range(min(3, len(row["members"])), len(row["members"])))
        target = max(offered, key=lambda j: row["old_sums_wdl"][j]) if offered else -1
        if len(offered) <= 1:
            losses.append((logits[i, :len(row["features"])]*0.).sum())
            choices.append(target)
        else:
            ids = [int(row["groups"][row["members"][j][0]]) for j in offered]
            scores = torch.stack([torch.amax(logits[i, torch.where(row["groups"] == group)[0]]) for group in ids])
            losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([offered.index(target)])))
            choices.append(offered[int(scores.detach().argmax())])
    return torch.stack(losses), choices


def _metrics(records, audit=False):
    result = {}
    for label, minimum in (("all_rows", 0), ("nonempty_outsiders", 1), ("multiple_outsiders", 2)):
        selected = [row for row in records if row["outsider_support"] >= minimum]
        eligible = [row for row in selected if row["outsider_support"] > 0]
        if audit:
            loss = math.fsum(row["loss"] for row in selected)/len(selected) if selected else None
            first = sum(1 for row in eligible if row["first_winner"])
            anymax = sum(1 for row in eligible if row["any_old_max"])
            regret = sum(int(row["old_regret_units"]) for row in selected)
        else:
            loss = sum(row["loss"] for row in selected)/len(selected) if selected else None
            first = sum(row["first_winner"] for row in eligible)
            anymax = sum(row["any_old_max"] for row in eligible)
            regret = sum(row["old_regret_units"] for row in selected)
        result[label] = {"rows": len(selected), "accuracy_denominator": len(eligible), "mean_loss": loss,
            "first_winner_count": first, "any_old_max_count": anymax,
            "first_winner_accuracy": first/len(eligible) if eligible else None,
            "any_old_max_accuracy": anymax/len(eligible) if eligible else None,
            "old_regret_units_sum": regret,
            "mean_old_score_regret": regret/(256*len(selected)) if selected else None}
    return result


def _report_summary(records, config, initial_records=None, *, audit=False):
    metrics = _metrics(records, audit)
    initial_units = (metrics["all_rows"]["old_regret_units_sum"] if initial_records is None
                     else _metrics(initial_records, audit)["all_rows"]["old_regret_units_sum"])
    current_units = metrics["all_rows"]["old_regret_units_sum"]
    capture = (initial_units-current_units)/initial_units if initial_units else None
    return {"metrics": metrics, "initial_old_regret_units": initial_units,
        "outsider_support_counts": {"empty": sum(row["outsider_support"] == 0 for row in records),
                                    "singleton": sum(row["outsider_support"] == 1 for row in records),
                                    "multiple": sum(row["outsider_support"] >= 2 for row in records)},
        "regret_capture_vs_initializer": capture,
        "regret_capture_guide_met": capture is not None and capture >= config["regret_capture_guide"]}


def training_report(model, rows, base, config, initial=None, *, audit=False,
                    data_scope="train", diagnostic_scope="full_train", slice_ids=None):
    c.require(data_scope in ("train", "validation") and bool(rows), "Invalid diagnostic split")
    c.require(all(row["split"] == data_scope for row in rows), "Mixed diagnostic splits")
    c.require(all(not {"assessment", "assessment_outcomes", "fresh_outcomes", "fresh_means"} & row.keys()
                  for row in rows), "Assessment outcomes forbidden in old-target diagnostics")
    model.eval()
    records = []
    tensorizer = audit_batch_tensors if audit else collate
    criterion = audit_objective if audit else objective
    # Stable action-length ordering greatly reduces attention padding in the
    # large corpus. This is diagnostic batching only, never an optimizer stream.
    ordered = sorted(rows, key=lambda row: len(row["features"]))
    with torch.no_grad():
        for lo in range(0, len(ordered), config["batch_size"]):
            c.budget_boundary(base)
            batch = ordered[lo:lo+config["batch_size"]]
            logits, _ = model(*tensorizer(batch))
            losses, choices = criterion(logits, batch)
            for row, loss, choice in zip(batch, losses.tolist(), choices, strict=True):
                target, sums = row["outsider_teacher"], row["old_sums_wdl"]
                c.require(math.isfinite(loss), "Nonfinite training metric")
                if audit:
                    support = len(row["members"])-min(3, len(row["members"]))
                    regret = max(sums[3:])-sums[choice] if support else 0
                    correct = choice == next((j for j in range(3, len(sums)) if sums[j] == max(sums[3:])), -1)
                else:
                    support = row["outsider_support"]
                    regret = sums[target]-sums[choice] if support else 0
                    correct = choice == target
                records.append({"id": row["id"], "pair": row["pair"], "index": row["index"],
                    "source_fit": row["source_fit"], "support": len(sums), "outsider_support": support,
                    "choice": choice, "target": target, "first_winner": correct if support else None,
                    "any_old_max": regret == 0 if support else None, "old_regret_units": regret, "loss": loss})
    records.sort(key=lambda row: row["id"])
    ids = {row["id"] for row in records}
    initial_records = None if initial is None else [row for row in initial["rows"] if row["id"] in ids]
    c.require(initial_records is None or {row["id"] for row in initial_records} == ids,
              "Initializer diagnostics do not cover the current scope")
    slices = {}
    for name, members in (slice_ids or {}).items():
        members = set(members)
        c.require(members <= ids, "Diagnostic slice leaves the evaluated corpus")
        current_slice = [row for row in records if row["id"] in members]
        initial_slice = None if initial_records is None else [row for row in initial_records if row["id"] in members]
        slices[name] = {"row_ids_sha256": old.digest(sorted(members)),
            **_report_summary(current_slice, config, initial_slice, audit=audit)}
    return {"rows": records, **_report_summary(records, config, initial_records, audit=audit),
        "metrics_by_slice": slices, "data_scope": data_scope, "diagnostic_scope": diagnostic_scope,
        "diagnostic_batching": "stable_action_length_ascending",
        "guide_is_stopping_rule": False, "assessment_read": False,
        "validation_read": data_scope == "validation"}


def _training_diagnostics(model, rows, probe_ids, update, base, config, initial, *, audit=False):
    full = update in (0, config["updates"])
    selected = rows if full else [row for row in rows if row["id"] in probe_ids]
    slices = {"original_168_train": probe_ids}
    if full:
        slices["full_train"] = {row["id"] for row in rows}
        slices["added_9293_train"] = slices["full_train"]-probe_ids
    return training_report(model, selected, base, config, initial, audit=audit,
                           diagnostic_scope="full_train" if full else "original_168_train_probe",
                           slice_ids=slices)


def _model(initializer, seed, config):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = StrongPolicyInference(str(c.verify_hash(initializer))).model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_head."))
    parameters = [p for name, p in model.named_parameters() if not name.startswith("value_head.")]
    c.require(all(p.dtype == torch.float32 for p in parameters), "Expected float32 policy")
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    c.require(not optimizer.state, "Fresh optimizer required")
    return model, parameters, optimizer


def _metadata(fit_id, binding):
    return {"experiment": "full_teacher_fit_128", "fit": fit_id, "manifest": binding,
            "checkpoint_selection": "fixed_final", "training_split": "train", "assessment_used": False}


def _compare(left, right):
    if isinstance(left, dict):
        c.require(isinstance(right, dict) and left.keys() == right.keys(), "Metric keys changed")
        for key in left:
            _compare(left[key], right[key])
    elif isinstance(left, list):
        c.require(isinstance(right, list) and len(left) == len(right), "Metric count changed")
        for a, b in zip(left, right, strict=True):
            _compare(a, b)
    elif isinstance(left, float):
        c.require(isinstance(right, (int, float)) and math.isfinite(right) and abs(left-right) <= 1e-12, "Metric arithmetic differs")
    else:
        c.require(left == right, "Training evidence differs")


def fit(fit_id, binding, base=OUTPUT, *, rows=None, initializer=None, config=None):
    c.budget_boundary(base)
    values, initial_ref, settings = _inputs(fit_id, binding, base, rows, initializer, config)
    probe_ids = _probe_ids(binding, base, values, settings, fixture=rows is not None)
    folder = Path(base)/"fits"/f"fit_{fit_id}"
    folder.mkdir(parents=True, exist_ok=False)
    reporter = StatusReporter(folder, {"manifest": binding, "fit": fit_id})
    started = time.perf_counter()
    seed = settings["train_seeds"][str(fit_id)]
    model, parameters, optimizer = _model(initial_ref, seed, settings)
    initial_tensors = {name: value.clone() for name, value in model.state_dict().items()}
    update, epoch, stream, curves, counts, initial_report = 0, 0, [], [], Counter(), None

    def observe():
        nonlocal initial_report
        c.budget_boundary(base)
        c.require(all(torch.equal(initial_tensors[name], value) for name, value in model.state_dict().items()
                      if name.startswith("value_head.")), "Value head changed")
        report = _training_diagnostics(model, values, probe_ids, update, base, settings, initial_report)
        if initial_report is None:
            initial_report = report
        checkpoint, metric = folder/f"checkpoint_{update:04d}.pt", folder/f"training_{update:04d}.json"
        save_model(checkpoint, model, update, _metadata(fit_id, binding))
        old.save_new(metric, report)
        curves.append({"update": update, "checkpoint": c.artifact(checkpoint), "training": c.artifact(metric)})
        reporter.event("full_teacher_fit", update=update, total_updates=settings["updates"],
                       diagnostic_scope=report["diagnostic_scope"])

    observe()
    while update < settings["updates"]:
        for indices in batch_plan(values, fit_id, epoch, settings):
            if update == settings["updates"]:
                break
            c.budget_boundary(base)
            batch = [values[i] for i in indices]
            ids = [row["id"] for row in batch]
            stream.append(ids); counts.update(ids)
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*collate(batch))
            loss = objective(logits, batch)[0].mean()
            c.require(bool(torch.isfinite(loss)), "Nonfinite focused loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
            c.require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()), "Nonfinite model")
            update += 1
            if update in settings["checkpoints"]:
                observe()
        epoch += 1
    selected = curves[-1]
    result = {"manifest": binding, "fit": fit_id, "config": settings, "seed": seed,
        "initializer": initial_ref, "training_rows_sha256": _rows_identity(values), "train_rows": len(values),
        "training_probe_rows": len(probe_ids), "training_probe_ids_sha256": old.digest(sorted(probe_ids)),
        "updates": update, "epochs_entered": epoch, "row_exposures": sum(counts.values()),
        "per_row_exposures": {row["id"]: counts[row["id"]] for row in values},
        "batch_sequence_sha256": old.digest(stream), "checkpoints": curves,
        "selected_update": update, "selected_model": selected["checkpoint"], "selected_training": selected["training"],
        "final_always_selected": True, "assessment_read": False, "validation_read": False,
        "elapsed_seconds": time.perf_counter()-started}
    path = folder/"result.json"; old.save_new(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def audit_fit(fit_id, fit_item, binding, base=OUTPUT, *, rows=None, initializer=None, config=None):
    c.budget_boundary(base)
    values, initial_ref, settings = _inputs(fit_id, binding, base, rows, initializer, config)
    probe_ids = _probe_ids(binding, base, values, settings, fixture=rows is not None)
    expected = c.verified(fit_item)
    c.require(expected["manifest"] == binding and expected["fit"] == fit_id, "Wrong fit identity")
    c.require(expected["config"] == settings and expected["initializer"] == initial_ref, "Wrong fitting recipe")
    c.require(expected["training_rows_sha256"] == _rows_identity(values) and expected["train_rows"] == len(values), "Wrong TRAIN corpus")
    c.require(expected["training_probe_rows"] == len(probe_ids)
              and expected["training_probe_ids_sha256"] == old.digest(sorted(probe_ids)), "Wrong original TRAIN probe")
    seed = settings["train_seeds"][str(fit_id)]
    c.require(expected["seed"] == seed, "Wrong training stream")
    curves = expected["checkpoints"]
    c.require([curve["update"] for curve in curves] == settings["checkpoints"], "Incomplete checkpoints")
    started = time.perf_counter()
    model, parameters, optimizer = _model(initial_ref, seed, settings)
    initial_tensors = {name: value.clone() for name, value in model.state_dict().items()}
    reporter = StatusReporter(Path(base)/"fits"/f"fit_{fit_id}"/"audit_status", {"manifest": binding, "fit": fit_id})
    update, epoch, stream, counts, initial_report = 0, 0, [], Counter(), None

    def inspect():
        nonlocal initial_report
        c.budget_boundary(base)
        curve = curves[settings["checkpoints"].index(update)]
        payload = torch.load(c.verify_hash(curve["checkpoint"]), map_location="cpu", weights_only=True)
        c.require(payload["model_kind"] == MODEL_KIND and payload["model_config"] == asdict(model.config), "Wrong saved architecture")
        c.require(payload["generation"] == update and payload["metadata"] == _metadata(fit_id, binding), "Wrong checkpoint identity")
        c.require(payload["model_state_dict"].keys() == model.state_dict().keys(), "Model tensor keys changed")
        for name, tensor in model.state_dict().items():
            c.require(bool(torch.isfinite(tensor).all()) and torch.equal(tensor, payload["model_state_dict"][name]), "Optimizer reproduction differs: "+name)
            if name.startswith("value_head."):
                c.require(torch.equal(tensor, initial_tensors[name]), "Value head changed")
        report = _training_diagnostics(model, values, probe_ids, update, base, settings, initial_report, audit=True)
        if initial_report is None:
            initial_report = report
        _compare(report, c.verified(curve["training"]))
        reporter.event("full_teacher_fit_reproduction", update=update, total_updates=settings["updates"],
                       diagnostic_scope=report["diagnostic_scope"])

    inspect()
    while update < settings["updates"]:
        for indices in independent_plan(values, fit_id, epoch, settings):
            if update >= settings["updates"]:
                break
            c.budget_boundary(base)
            batch = [values[index] for index in indices]
            ids = [row["id"] for row in batch]
            stream.append(ids); counts.update(ids)
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*audit_batch_tensors(batch))
            loss = audit_objective(logits, batch)[0].mean()
            c.require(bool(torch.isfinite(loss)), "Nonfinite independently fitted loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
            c.require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()), "Nonfinite independently fitted model")
            update += 1
            if update in settings["checkpoints"]:
                inspect()
        epoch += 1
    c.require(expected["updates"] == update == settings["updates"] and expected["epochs_entered"] == epoch, "Wrong exposure duration")
    c.require(expected["row_exposures"] == sum(counts.values()), "Wrong total exposures")
    c.require(expected["per_row_exposures"] == {row["id"]: counts[row["id"]] for row in values}, "Wrong row exposures")
    c.require(expected["batch_sequence_sha256"] == old.digest(stream), "Batch stream differs")
    c.require(expected["selected_update"] == update and expected["selected_model"] == curves[-1]["checkpoint"], "Final model was not selected")
    c.require(expected["selected_training"] == curves[-1]["training"] and expected["final_always_selected"], "Wrong final metric linkage")
    c.require(not expected["assessment_read"] and not expected["validation_read"], "Forbidden fitting data use")
    path = Path(base)/"fits"/f"fit_{fit_id}"/"audit.json"
    old.save_new(path, {"manifest": binding, "fit": fit_id, "result": c.artifact(c.verify_hash(fit_item)),
        "passed": True, "complete_fit_bitwise_identical": True, "all_training_metrics_independently_reproduced": True,
        "value_head_frozen": True, "batch_sequence_sha256": old.digest(stream), "row_exposures": sum(counts.values()),
        "per_row_exposures": {row["id"]: counts[row["id"]] for row in values},
        "training_probe_rows": len(probe_ids), "training_probe_ids_sha256": old.digest(sorted(probe_ids)),
        "selected_update": update, "selected_model": curves[-1]["checkpoint"],
        "selected_training": curves[-1]["training"], "final_always_selected": True,
        "assessment_read": False, "validation_read": False, "elapsed_seconds": time.perf_counter()-started})
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))
