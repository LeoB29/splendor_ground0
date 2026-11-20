"""One fixed outsider-ranking intervention and fairly reselected saved CE control.

Training and independent reproduction are only called by the frozen experiment.
Parent checkpoints and exports are read-only; no predecessor globals are changed.
"""
from collections import defaultdict
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from scripts.analysis import verify_fast_selector_transfer_157 as independent
from tools.proposal_iteration_128 import learning as previous
from . import common as c
from . import learning_data as data

old, f = c.old, previous.f
ROOT, OUTPUT, SOURCE, FITS = c.ROOT, c.OUTPUT, c.SOURCE, c.FITS
CONFIG = {
    "epochs": 4, "batch_size": 64, "lr": 1e-5, "weight_decay": 1e-4,
    "gradient_norm": 1., "torch_threads": 1, "device": "cpu", "dtype": "float32",
    "fresh_optimizer": True, "value_head_frozen": True,
    "objective": "abs(delta)/M * softplus(-sign(delta)*(groupmax_slot2-groupmax_slot3))",
    "delta": "all128_score_slot2_minus_slot3;short_support_zero",
    "M": "mean_abs_delta_over_all_training_rows_including_short_zero",
    "validation": "mean_of_whole_pair_mean_ranking_loss",
    "selection": "earliest_strict_minimum", "initial_checkpoint_eligible": True,
    "training_seeds": {str(key): value for key, value in previous.TRAIN_SEEDS.items()},
    "unoffered_actions_are_negative": False, "saved_ce_control": "all_treatment_checkpoints",
}
prepare_data = data.prepare_data


def _same_file(left, right):
    return c.verify_hash(left) == c.verify_hash(right) and left["sha256"] == right["sha256"]


def source_evidence():
    """Read-only prefreeze inventory, using sealed receipts for source-root hashes."""
    found = {}

    def add(item):
        c.verify_hash(item)
        key = item["path"]
        if key in found:
            assert found[key]["sha256"] == item["sha256"]
        found[key] = {"path": key, "sha256": item["sha256"]}

    lock_item = c.artifact(SOURCE / "data_lock.json")
    add(lock_item)
    lock = c.verified(lock_item)
    assert lock["manifest"] == c.SOURCE_BINDING and lock["config"] == previous.CONFIG
    add(lock["audit"])
    audit = c.verified(lock["audit"])
    assert audit["passed"] and audit["manifest"] == c.SOURCE_BINDING
    for item in lock["source_dependencies"]:
        add(item)
    for kind in ("old", "new"):
        for item, receipt_item in zip(lock[f"{kind}_exports"], audit[f"{kind}_export_audits"], strict=True):
            add(item)
            add(receipt_item)
            receipt = c.verified(receipt_item)
            assert receipt["passed"] and receipt["export"] == item
            source = receipt["source"]
            if kind == "old":
                add({"path": source["path"], "sha256": source["sha256"]})
            else:
                add(source["game"])
                add(source["audit"])
    for fit_id in FITS:
        add(previous.initializer(fit_id))
        folder = SOURCE / "fits" / f"treatment_{fit_id}"
        result_item, audit_item = c.artifact(folder / "result.json"), c.artifact(folder / "audit.json")
        add(result_item)
        add(audit_item)
        result, checked = c.verified(result_item), c.verified(audit_item)
        assert checked["result"] == result_item and checked["complete_fit_bitwise_identical"]
        assert result["manifest"] == checked["manifest"] == c.SOURCE_BINDING
        for curve in result["curves"]:
            for stem, key, ext in (("checkpoint", "checkpoint_sha256", "pt"), ("validation", "validation_sha256", "json")):
                item = c.artifact(folder / f"{stem}_{curve['update']:04d}.{ext}")
                assert item["sha256"] == curve[key]
                add(item)
    return [found[key] for key in sorted(found)]


def objective(logits, rows, normalizer):
    assert math.isfinite(normalizer) and normalizer > 0
    losses = []
    for i, row in enumerate(rows):
        gap = row["ranking_delta"]
        assert math.isfinite(gap) and -1 <= gap <= 1
        if len(row["members"]) != 4 or gap == 0:
            assert gap == 0
            losses.append(logits[i, 0] * 0.)
        else:
            scores = f.restricted_scores(logits[i], row)
            losses.append(F.softplus(-math.copysign(1., gap)*(scores[2]-scores[3]))*(abs(gap)/normalizer))
    return torch.stack(losses)


def audit_objective(logits, rows, normalizer):
    assert math.isfinite(normalizer) and normalizer > 0
    losses = []
    for i, row in enumerate(rows):
        difference = row["ranking_delta"]
        assert math.isfinite(difference) and abs(difference) <= 1
        if len(row["members"]) < 4 or difference == 0:
            assert 1 <= len(row["members"]) <= 4 and difference == 0
            losses.append(logits[i, 0] * 0.)
            continue
        assert len(row["members"]) == 4
        group2, group3 = (int(row["groups"][row["members"][j][0]]) for j in (2, 3))
        second = torch.amax(logits[i, torch.where(row["groups"] == group2)[0]])
        third = torch.amax(logits[i, torch.where(row["groups"] == group3)[0]])
        signed = third-second if difference > 0 else second-third
        losses.append(F.softplus(signed)*(abs(difference)/normalizer))
    return torch.stack(losses)


def pair_loss(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["pair"]].append(row["loss"])
    assert groups
    return sum(sum(groups[key])/len(groups[key]) for key in sorted(groups))/len(groups)


def validation(model, rows, normalizer, base=OUTPUT, audit=False):
    model.eval()
    detail = []
    batches = independent.audit_batches if audit else f.batches
    collate = independent.audit_batch_tensors if audit else f.collate
    loss_fn = audit_objective if audit else objective
    with torch.no_grad():
        for indices in batches(rows, 157000, -1):
            c.budget_boundary(base)
            batch = [rows[i] for i in indices]
            logits, _ = model(*collate(batch))
            losses = loss_fn(logits, batch, normalizer)
            for row, loss in zip(batch, losses.tolist(), strict=True):
                assert math.isfinite(loss)
                detail.append({"id": row["id"], "pair": row["pair"], "loss": loss,
                               "ranking_delta": row["ranking_delta"], "support": len(row["members"])})
    detail.sort(key=lambda row: row["id"])
    if audit:
        means = []
        for key in sorted({row["pair"] for row in detail}):
            values = [row["loss"] for row in detail if row["pair"] == key]
            means.append(sum(values)/len(values))
        score = sum(means)/len(means)
    else:
        score = pair_loss(detail)
    return {"pair_loss": score, "normalizer_M": normalizer, "rows": detail}


def _inputs(binding, base, rows):
    if rows is None:
        return data.load_data(binding, base)
    location = Path(base).resolve()
    assert location != OUTPUT.resolve() and not location.is_relative_to(OUTPUT.resolve())
    assert location != SOURCE.resolve() and not location.is_relative_to(SOURCE.resolve())
    assert rows and len({r["id"] for r in rows}) == len(rows)
    training = [r for r in rows if r["split"] == "train"]
    assert training and any(r["split"] == "validation" for r in rows)
    normalizer = sum(abs(r["ranking_delta"]) for r in training)/len(training)
    assert normalizer > 0
    return rows, normalizer


def _plan(training, fit_id, audit=False):
    plan = previous.independent_plan if audit else previous.batch_plan
    return [[shared for shared, _ in plan(training, previous.TRAIN_SEEDS[fit_id], epoch)] for epoch in range(4)]


def _control(fit_id, training, valid, base, rows, ce_base=None):
    if ce_base is not None:
        assert rows is not None
        location = Path(ce_base).resolve()
        assert location != SOURCE.resolve() and not location.is_relative_to(SOURCE.resolve())
    else:
        assert rows is None
        location = SOURCE
    folder = location / "fits" / f"treatment_{fit_id}"
    result_item, audit_item = c.artifact(folder / "result.json"), c.artifact(folder / "audit.json")
    result, checked = c.verified(result_item), c.verified(audit_item)
    assert result["manifest"] == checked["manifest"]
    if rows is None:
        assert result["manifest"] == c.SOURCE_BINDING
    assert result["arm"] == checked["arm"] == "treatment" and result["fit"] == checked["fit"] == fit_id
    assert _same_file(checked["result"], result_item) and checked["complete_fit_bitwise_identical"] and checked["value_head_frozen"]
    assert result["initializer"] == previous.initializer(fit_id) and result["seed"] == previous.TRAIN_SEEDS[fit_id]
    assert result["train_rows"] == len(training) and result["validation_rows"] == len(valid)
    assert result["row_exposures"] == 4*len(training)
    points = f.checkpoint_updates(len(training))
    assert result["updates"] == points[-1]
    assert [curve["update"] for curve in result["curves"]] == checked["checkpoints"] == points
    stream = [[training[i]["id"] for i in batch] for epoch in _plan(training, fit_id, audit=True) for batch in epoch]
    assert old.digest(stream) == result["batch_sequence_sha256"] == result["slot_sequence_sha256"]
    if rows is None:
        assert result["data_lock"] == c.artifact(SOURCE / "data_lock.json")
        assert c.verified(result["data_lock"])["config"] == previous.CONFIG
    else:
        assert result["data_lock"] is None
    checkpoints = {}
    for curve in result["curves"]:
        c.budget_boundary(base)
        checkpoint = c.artifact(folder / f"checkpoint_{curve['update']:04d}.pt")
        original_validation = c.artifact(folder / f"validation_{curve['update']:04d}.json")
        assert checkpoint["sha256"] == curve["checkpoint_sha256"]
        assert original_validation["sha256"] == curve["validation_sha256"]
        checkpoints[curve["update"]] = checkpoint
    return {"result": result_item, "audit": audit_item, "binding": result["manifest"], "checkpoints": checkpoints}


def _model(fit_id):
    model = f.StrongPolicyInference(str(c.verify_hash(previous.initializer(fit_id)))).model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_head."))
    parameters = [value for name, value in model.named_parameters() if not name.startswith("value_head.")]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4)
    assert not optimizer.state
    return model, parameters, optimizer


def _ce_model(item, fit_id, update, binding, initial):
    payload = torch.load(c.verify_hash(item), map_location="cpu", weights_only=True)
    assert payload["generation"] == update
    assert payload["metadata"] == {"experiment": "proposal_iteration_128", "arm": "treatment", "fit": fit_id,
                                   "seed": previous.TRAIN_SEEDS[fit_id], "manifest": binding}
    state = payload["model_state_dict"]
    assert state.keys() == initial.keys()
    assert all(torch.isfinite(value).all() for value in state.values())
    assert all(torch.equal(state[name], value) for name, value in initial.items() if update == 0 or name.startswith("value_head."))
    return f.StrongPolicyInference(str(c.verify_hash(item))).model


def _metadata(fit_id, binding):
    return {"experiment": "proposal_ranking_128", "arm": "ranking", "fit": fit_id,
            "seed": previous.TRAIN_SEEDS[fit_id], "manifest": binding}


def fit(fit_id, binding, base=OUTPUT, *, rows=None, ce_base=None):
    assert fit_id in FITS
    c.budget_boundary(base)
    old.initialize_worker()
    values, normalizer = _inputs(binding, base, rows)
    training, valid = ([r for r in values if r["split"] == kind] for kind in ("train", "validation"))
    control = _control(fit_id, training, valid, base, rows, ce_base)
    model, parameters, optimizer = _model(fit_id)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    folder = Path(base) / "fits" / f"ranking_{fit_id}"
    folder.mkdir(parents=True, exist_ok=False)
    points = f.checkpoint_updates(len(training))
    update, curves, stream = 0, [], []
    best = {"ranking": math.inf, "ce": math.inf}
    selected = {"ranking": None, "ce": None}
    started = time.perf_counter()

    def observe():
        c.budget_boundary(base)
        reports = {"ranking": validation(model, valid, normalizer, base)}
        ce_item = control["checkpoints"][update]
        ce_model = _ce_model(ce_item, fit_id, update, control["binding"], initial)
        reports["ce"] = validation(ce_model, valid, normalizer, base)
        checkpoint = folder / f"checkpoint_{update:04d}.pt"
        f.save_model(checkpoint, model, update, _metadata(fit_id, binding))
        curve = {"update": update, "ranking_checkpoint": c.artifact(checkpoint), "ce_checkpoint": ce_item}
        for arm, report in reports.items():
            path = folder / f"{arm}_validation_{update:04d}.json"
            old.save_new(path, report)
            curve[f"{arm}_validation"] = c.artifact(path)
            curve[f"{arm}_pair_loss"] = report["pair_loss"]
            if report["pair_loss"] < best[arm]:
                best[arm], selected[arm] = report["pair_loss"], update
        curves.append(curve)
        print({"ranking_fit": fit_id, "update": update, "total_updates": points[-1], "selected": selected}, flush=True)

    observe()
    for epoch in _plan(training, fit_id):
        for indices in epoch:
            c.budget_boundary(base)
            batch = [training[i] for i in indices]
            stream.append([row["id"] for row in batch])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*f.collate(batch))
            loss = objective(logits, batch, normalizer).mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.isfinite(t).all() for t in model.state_dict().values())
            update += 1
            if update in points:
                observe()
    assert update == points[-1]
    assert old.digest(stream) == c.verified(control["result"])["batch_sequence_sha256"]
    result = {"manifest": binding, "fit": fit_id, "seed": previous.TRAIN_SEEDS[fit_id], "config": CONFIG,
              "initializer": previous.initializer(fit_id), "normalizer_M": normalizer,
              "data_lock": c.artifact(Path(base) / "data_lock.json") if rows is None else None,
              "ce_control": {key: control[key] for key in ("result", "audit", "binding")},
              "selected_updates": selected, "curves": curves, "updates": update,
              "train_rows": len(training), "validation_rows": len(valid), "row_exposures": 4*len(training),
              "batch_sequence_sha256": old.digest(stream), "slot_sequence_sha256": old.digest(stream),
              "elapsed_seconds": time.perf_counter()-started}
    old.save_new(folder / "result.json", result)
    return c.artifact(folder / "result.json", fit=fit_id, index=FITS.index(fit_id))


def audit_fit(fit_id, fit_item, binding, base=OUTPUT, *, rows=None, ce_base=None):
    assert fit_id in FITS
    c.budget_boundary(base)
    old.initialize_worker()
    values, normalizer = _inputs(binding, base, rows)
    training, valid = ([r for r in values if r["split"] == kind] for kind in ("train", "validation"))
    expected = c.verified(fit_item)
    assert expected["manifest"] == binding and expected["fit"] == fit_id and expected["config"] == CONFIG
    assert expected["initializer"] == previous.initializer(fit_id) and expected["seed"] == 207001+FITS.index(fit_id)
    assert expected["normalizer_M"] == normalizer
    assert expected["data_lock"] == (c.artifact(Path(base) / "data_lock.json") if rows is None else None)
    control = _control(fit_id, training, valid, base, rows, ce_base)
    assert expected["ce_control"] == {key: control[key] for key in ("result", "audit", "binding")}
    model, parameters, optimizer = _model(fit_id)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    steps = (len(training)+63)//64
    points = sorted({x for x in (0, 1, 2, 4, 8, 16, 32, 64) if x <= 4*steps} |
                    {epoch*steps+math.ceil(q*steps/4) for epoch in range(4) for q in (1, 2, 3, 4)})
    assert [r["update"] for r in expected["curves"]] == points
    update, stream = 0, []
    selected, best = {"ranking": None, "ce": None}, {"ranking": math.inf, "ce": math.inf}
    started = time.perf_counter()

    def inspect():
        c.budget_boundary(base)
        curve = expected["curves"][points.index(update)]
        assert curve["ce_checkpoint"] == control["checkpoints"][update]
        payload = torch.load(c.verify_hash(curve["ranking_checkpoint"]), map_location="cpu", weights_only=True)
        assert payload["generation"] == update and payload["metadata"] == _metadata(fit_id, binding)
        assert payload["model_state_dict"].keys() == model.state_dict().keys()
        for name, value in model.state_dict().items():
            assert torch.isfinite(value).all() and torch.equal(value, payload["model_state_dict"][name]), (fit_id, update, name)
            if name.startswith("value_head."):
                assert torch.equal(value, initial[name])
        ce_model = _ce_model(curve["ce_checkpoint"], fit_id, update, control["binding"], initial)
        for arm, current in (("ranking", model), ("ce", ce_model)):
            report = validation(current, valid, normalizer, base, audit=True)
            assert report == c.verified(curve[f"{arm}_validation"])
            assert report["pair_loss"] == curve[f"{arm}_pair_loss"]
            if report["pair_loss"] < best[arm]:
                best[arm], selected[arm] = report["pair_loss"], update
        print({"ranking_reproduction": fit_id, "update": update, "total_updates": 4*steps}, flush=True)

    inspect()
    for epoch in _plan(training, fit_id, audit=True):
        for indices in epoch:
            c.budget_boundary(base)
            batch = [training[i] for i in indices]
            stream.append([row["id"] for row in batch])
            optimizer.zero_grad(set_to_none=True)
            model.train()
            logits, _ = model(*independent.audit_batch_tensors(batch))
            loss = audit_objective(logits, batch, normalizer).mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.isfinite(value).all() for value in model.state_dict().values())
            update += 1
            if update in points:
                inspect()
    assert update == expected["updates"] == 4*steps and selected == expected["selected_updates"]
    assert expected["row_exposures"] == 4*len(training)
    assert expected["train_rows"] == len(training) and expected["validation_rows"] == len(valid)
    assert old.digest(stream) == expected["batch_sequence_sha256"] == expected["slot_sequence_sha256"]
    unchanged = {}
    for arm in ("ranking", "ce"):
        curve = next(row for row in expected["curves"] if row["update"] == selected[arm])
        state = torch.load(c.verify_hash(curve[f"{arm}_checkpoint"]), weights_only=True, map_location="cpu")["model_state_dict"]
        unchanged[arm] = all(torch.equal(value, initial[name]) for name, value in state.items())
    result = {"manifest": binding, "fit": fit_id, "result": c.artifact(c.verify_hash(fit_item)),
              "selected_updates": selected, "checkpoints": points, "complete_fit_bitwise_identical": True,
              "ce_selection_independently_reproduced": True, "value_head_frozen": True,
              "unchanged_from_anchor": unchanged, "elapsed_seconds": time.perf_counter()-started}
    path = Path(base) / "fits" / f"ranking_{fit_id}" / "audit.json"
    old.save_new(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def _models(fits, audits, binding, base):
    data.load_data(binding, base)
    assert len(fits) == len(audits) == 3
    models = {}
    for fit_id, item, checked in zip(FITS, fits, audits, strict=True):
        result, audit = c.verified(item), c.verified(checked)
        assert result["manifest"] == audit["manifest"] == binding
        assert result["fit"] == audit["fit"] == fit_id and result["config"] == CONFIG
        assert audit["result"] == c.artifact(c.verify_hash(item))
        assert audit["complete_fit_bitwise_identical"] and audit["ce_selection_independently_reproduced"] and audit["value_head_frozen"]
        assert result["data_lock"] == c.artifact(Path(base) / "data_lock.json")
        assert result["initializer"] == previous.initializer(fit_id)
        assert result["selected_updates"] == audit["selected_updates"]
        c.verify_hash(result["ce_control"]["result"])
        c.verify_hash(result["ce_control"]["audit"])
        for curve in result["curves"]:
            c.budget_boundary(base)
            for arm in ("ranking", "ce"):
                c.verify_hash(curve[f"{arm}_checkpoint"])
                report = c.verified(curve[f"{arm}_validation"])
                assert report["pair_loss"] == curve[f"{arm}_pair_loss"]
        for arm in ("ranking", "ce"):
            earliest = min(result["curves"], key=lambda row: row[f"{arm}_pair_loss"])
            selected = result["selected_updates"][arm]
            assert earliest["update"] == selected
            models[f"{arm}_{fit_id}"] = {**earliest[f"{arm}_checkpoint"], "source_fit": fit_id,
                     "selected_update": selected, "unchanged_from_anchor": audit["unchanged_from_anchor"][arm]}
        models[f"anchor_{fit_id}"] = {**previous.initializer(fit_id), "unchanged_from_anchor": True}
    return models


def models_lock(fits, audits, binding, base=OUTPUT):
    fits = sorted(fits, key=lambda item: item["fit"])
    audits = sorted(audits, key=lambda item: item["fit"])
    models = _models(fits, audits, binding, base)
    result = {"manifest": binding, "models": models, "data_lock": c.artifact(Path(base) / "data_lock.json"),
              "fits": fits, "audits": audits, "training_config": CONFIG,
              "all_ranking_unchanged": all(models[f"ranking_{fit_id}"]["unchanged_from_anchor"] for fit_id in FITS),
              "matched_ce_dose_and_batches": True, "common_ranking_validation_selection": True}
    path = Path(base) / "models_lock.json"
    old.save_new(path, result)
    return c.artifact(path)


def validate_models(binding, base=OUTPUT):
    lock = old.read_json(Path(base) / "models_lock.json")
    assert lock["manifest"] == binding and lock["training_config"] == CONFIG
    assert lock["data_lock"] == c.artifact(Path(base) / "data_lock.json")
    assert lock["models"] == _models(lock["fits"], lock["audits"], binding, base)
    assert lock["all_ranking_unchanged"] == all(lock["models"][f"ranking_{fit_id}"]["unchanged_from_anchor"] for fit_id in FITS)
    assert lock["matched_ce_dose_and_batches"] and lock["common_ranking_validation_selection"]
    return lock
