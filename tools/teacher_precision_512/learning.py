"""Same-state teacher precision, fixed endpoints and independent optimizer replay.

Only authenticated TRAIN pools are opened. The full policy trains; only the
value head is frozen. The exact completed BROAD batch stream is reused.
Atomic optimizer checkpoints resume interrupted fitting; independent audits
reconstruct each complete trajectory from its original initialization.
"""
from pathlib import Path
from dataclasses import asdict
import os
import math
import time
import uuid

import torch

from splendor_ai.alphazero.fast_selector_157 import collate
from scripts.analysis.verify_fast_selector_transfer_157 import audit_batch_tensors
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.independent_game_coverage_128_recovery_01 import learning as prior
from tools.frozen_representation_128 import learning as matched_prior
from . import common as c
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, save_model, MODEL_KIND

FITS, OUTPUT, old = c.FITS, c.OUTPUT, c.old
ARMS = ("precision",)
CONFIG = {**prior.CONFIG, **c.CONFIG}
RECIPE_KEYS = matched_prior.RECIPE_KEYS

_compare = prior._compare
diagnostic_terms, initial_policy_kl, _report_summary = prior.diagnostic_terms, prior.initial_policy_kl, prior._report_summary


def _base(base):
    return Path(OUTPUT if base is None else base)


class _BudgetModel(torch.nn.Module):
    def __init__(self, model, base):
        super().__init__()
        self.model, self.base = model, base

    def forward(self, *args):
        c.budget_boundary(self.base)
        return self.model(*args)


TARGET_FIELDS = ("teacher_sums_wdl", "teacher_worlds", "teacher_outsider")


def original_pools(pools):
    return {pool: [{k: v for k, v in row.items() if k not in TARGET_FIELDS} for row in rows]
            for pool, rows in pools.items()}


def validate_target(row, *, added):
    fields = set(TARGET_FIELDS) & row.keys()
    c.require(fields == (set(TARGET_FIELDS) if added else set()), "Precision fields only on added TRAIN rows")
    if added:
        values = row["teacher_sums_wdl"]
        c.require(row["teacher_worlds"] == 512 and len(values) == len(row["members"])
                  and all(type(v) is int and -512 <= v <= 512 for v in values), "Invalid512 target support or sums")
        target = max(range(3, len(values)), key=lambda j: values[j]) if len(values) > 3 else -1
        c.require(row["teacher_outsider"] == target, "Incorrect512 outsider first maximum")


def _target(row):
    return ((row["teacher_sums_wdl"], row["teacher_worlds"], row["teacher_outsider"])
            if "teacher_worlds" in row else (row["old_sums_wdl"], 128, row["outsider_teacher"]))


def soft_targets(row, *, dtype=torch.float32, device=None, temperature=.05):
    c.require(temperature == .05, "Fixed score-temperature required")
    values, worlds, _ = _target(row)
    q = torch.tensor([(worlds + value)/(2*worlds) for value in values[row["outsider_start"]:]],
                     dtype=dtype, device=device)
    return torch.softmax(q/temperature, dim=0)


def audit_soft_targets(row, *, dtype=torch.float32, device=None, temperature=.05):
    c.require(temperature == .05, "Fixed score-temperature required")
    refined = "teacher_worlds" in row
    n = 512 if refined else 128
    sums = row["teacher_sums_wdl"] if refined else row["old_sums_wdl"]
    if refined:
        c.require(row["teacher_worlds"] == n, "Independent precision denominator differs")
    q = torch.tensor([.5 + float(sums[j])/(2*n) for j in range(3, len(row["members"]))],
                     dtype=dtype, device=device)
    return torch.nn.functional.softmax(q/temperature, dim=0)


def objective(logits, rows, temperature=.05):
    losses, choices = [], []
    for i, row in enumerate(rows):
        _, _, target = _target(row)
        if row["outsider_support"] < 2:
            losses.append(logits[i, row["members"][0][0]]*0.)
            choices.append(target)
        else:
            scores = torch.stack([torch.amax(logits[i, member]) for member in row["members"][3:]])
            weights = soft_targets(row, dtype=scores.dtype, device=scores.device, temperature=temperature)
            losses.append(-(weights*torch.log_softmax(scores, dim=0)).sum())
            choices.append(3+int(scores.detach().argmax()))
    return torch.stack(losses), choices


def audit_objective(logits, rows, temperature=.05):
    losses, choices = [], []
    for i, row in enumerate(rows):
        offered = list(range(min(3, len(row["members"])), len(row["members"])))
        values = row.get("teacher_sums_wdl", row["old_sums_wdl"])
        target = max(offered, key=lambda j: values[j]) if offered else -1
        if len(offered) <= 1:
            losses.append((logits[i, :len(row["features"])]*0.).sum())
            choices.append(target)
        else:
            groups = [int(row["groups"][row["members"][j][0]]) for j in offered]
            scores = torch.stack([torch.amax(logits[i, torch.where(row["groups"] == group)[0]]) for group in groups])
            weights = audit_soft_targets(row, dtype=scores.dtype, device=scores.device, temperature=temperature)
            losses.append(-torch.sum(weights*torch.nn.functional.log_softmax(scores, dim=0)))
            choices.append(offered[int(scores.detach().argmax())])
    return torch.stack(losses), choices


def _diagnostic_view(rows, *, precision):
    # All denominators are powers of two. Rescaling sums to128-world units is
    # exact binary arithmetic and permits reuse of independently tested score
    # diagnostics; reports explicitly identify these as normalized units.
    result = []
    for row in rows:
        value = {k: v for k, v in row.items() if k not in TARGET_FIELDS}
        if precision and "teacher_worlds" in row:
            value["old_sums_wdl"] = [x/4 for x in row["teacher_sums_wdl"]]
            value["outsider_teacher"] = row["teacher_outsider"]
        result.append(value)
    return result


def training_report(model, rows, base, config, initial=None, *, audit=False, precision=False,
                    data_scope="train", diagnostic_scope="full_train", **kwargs):
    view = _diagnostic_view(rows, precision=precision)
    report = prior.training_report(_BudgetModel(model, base), view, base, config, initial,
        audit=audit, data_scope=data_scope, diagnostic_scope=diagnostic_scope, **kwargs)
    if precision:
        # The inherited diagnostic reducer assumes integer128-world units.
        # Convert exactly back to integer512-world units and independently
        # rebuild every regret reduction; never truncate fractional quarters.
        for record in report["rows"]:
            raw = record["old_regret_units"]*4
            c.require(raw == int(raw), "Nonintegral512 regret units")
            record["old_regret_units"] = int(raw)
        for key, minimum in (("all_rows", 0), ("nonempty_outsiders", 1), ("multiple_outsiders", 2)):
            eligible = [r for r in report["rows"] if r["outsider_support"] >= minimum]
            units = sum(int(r["old_regret_units"]) for r in eligible) if audit else sum(r["old_regret_units"] for r in eligible)
            report["metrics"][key]["old_regret_units_sum"] = units
            report["metrics"][key]["mean_old_score_regret"] = units/(1024*len(eligible)) if eligible else None
        units = report["metrics"]["all_rows"]["old_regret_units_sum"]
        initial_units = units if initial is None else sum(r["old_regret_units"] for r in initial["rows"])
        capture = (initial_units-units)/initial_units if initial_units else None
        report.update(initial_old_regret_units=initial_units, regret_capture_vs_initializer=capture,
                      regret_capture_guide_met=capture is not None and capture >= config["regret_capture_guide"],
                      target_worlds=512, target_semantics="own512_added_teacher_targets",
                      regret_units="integer512_world_WDL_units_divide1024_for_score",
                      finite_targets_not_ground_truth=True)
    return report


def batch_plan(pools, narrow, fit_id, config=CONFIG):
    return prior.batch_plan(pools, narrow, fit_id, config)["broad"]


def independent_plan(pools, narrow, fit_id, config=CONFIG):
    return prior.independent_plan(pools, narrow, fit_id, config)["broad"]


def _inputs(fit_id, binding, base, rows, initializer, config, narrow, control):
    c.require(fit_id in FITS, "Unknown fit")
    supplied = (rows, initializer, config, narrow, control)
    fixture = any(value is not None for value in supplied)
    if fixture:
        c.require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Fixture cannot target production")
        c.require(all(value is not None for value in supplied), "Complete fixture inputs required")
        pools, initial, settings, selected, refs = rows, initializer, dict(config), list(narrow), control
    else:
        from . import training_source as source
        source.validate_targets(binding, base)
        pools, initial = source.load_pools(binding, base), source.initializer(fit_id, base)
        selected, settings = source.narrow_indices(fit_id, binding, base), dict(CONFIG)
        refs = {"result": source.broad_fit(fit_id, binding, base), "audit": source.broad_audit(fit_id, binding, base)}
        c.require(len(pools["old"]) == 9461 and len(pools["added"]) == 720 and len(selected) == 180,
                  "Production population changed")
    pools, initial, settings, selected, _ = prior._inputs(fit_id, binding, base, pools, initial, settings, selected)
    for pool, group in pools.items():
        for row in group:
            validate_target(row, added=pool == "added")
    c.require(set(refs) == {"result", "audit"}, "Both completed control references required")
    c.require(settings["checkpoint_selection"] == "fixed_final_update_1784"
              and not settings["selection_uses_intermediate_checkpoints"], "Checkpoint selection forbidden")
    return pools, initial, settings, selected, refs, fixture


def _matched_control(fit_id, pools, initial, settings, selected, refs, plan, *, fixture):
    receipt = matched_prior._matched_control(fit_id, original_pools(pools), initial, settings,
                                            selected, refs, plan, fixture=fixture)
    receipt.pop("only_trainable_parameter_mask_changes")
    receipt.update(only_added_teacher_precision_changes=True, same_full_policy_trainability=True,
                   original_broad_fit_replayed=False, old_teacher_worlds=128, added_teacher_worlds=512)
    return receipt


def _model(initial, arm, seed, settings, *, audit=False):
    c.require(arm == "precision", "Unknown learner arm")
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = StrongPolicyInference(str(c.verify_hash(initial))).model
    c.require(model.config.dropout == 0., "Matched dropout-free policy required")
    frozen_ids = {id(parameter) for parameter in model.value_head.parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(id(parameter) not in frozen_ids if audit else not name.startswith("value_head."))
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = [name for name, parameter in model.named_parameters() if id(parameter) not in frozen_ids]
    c.require(names == expected and names, "All policy and only policy parameters must train")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    c.require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), "Wrong fitting dtype")
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    c.require(not optimizer.state, "Fresh optimizer required")
    model.eval()
    frozen = {name: tensor.clone() for name, tensor in model.state_dict().items() if name.startswith("value_head.")}
    return model, parameters, optimizer, frozen, names


def _check_frozen(model, frozen):
    current = model.state_dict()
    c.require(set(frozen) == {k for k in current if k.startswith("value_head.")}
              and all(torch.equal(current[k], v) for k, v in frozen.items()), "Value head tensor changed")
    c.require(all(not module.training for module in model.modules()), "Model left evaluation mode")


def _metadata(fit_id, arm, binding, update):
    return {"experiment": "teacher_precision_512", "fit": fit_id, "arm": arm, "manifest": binding,
            "update": update, "checkpoint_selection": "fixed_final", "assessment_used": False,
            "value_head_frozen": True}


def _diagnostics(model, pools, base, settings, initial=None, *, audit=False):
    scopes = (("old", "old", False), ("added", "added", True), ("added128", "added", False))
    return {name: training_report(model, pools[pool], base, settings,
        None if initial is None else initial[name], audit=audit, precision=refined,
        diagnostic_scope=f"full_{name}_train") for name, pool, refined in scopes}


def _equal_torch(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal_torch(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(_equal_torch(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _new_json_or_equal(path, value):
    if path.exists():
        _compare(value, old.read_json(path))
    else:
        old.save_new(path, value)


def _new_torch_or_equal(path, value):
    if path.exists():
        c.require(_equal_torch(value, torch.load(path, map_location="cpu", weights_only=True)),
                  "Existing optimizer checkpoint differs")
        return
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _checkpoint_model(path, model, metadata):
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        c.require(saved["metadata"] == metadata and saved["generation"] == metadata["update"]
                  and saved["model_kind"] == MODEL_KIND and saved["model_config"] == asdict(model.config)
                  and _equal_torch(saved["model_state_dict"], model.state_dict()), "Existing final model differs")
        return
    save_model(path, model, metadata["update"], metadata)


def _identity(fit_id, arm, binding, initial, settings, matched, pools):
    return {"manifest": binding, "fit": fit_id, "arm": arm, "initializer": initial,
            "config": settings, "matched_broad_control": matched,
            "old_rows_sha256": prior._rows_identity(pools["old"]),
            "added_rows_sha256": prior._rows_identity(pools["added"])}


def _save_progress(folder, model, optimizer, update, norms, identity):
    path = folder/f"optimizer_{update:04d}.pt"
    payload = {"identity": identity, "update": update, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "preclip_gradient_norms": list(norms)}
    _new_torch_or_equal(path, payload)
    value = {"identity_sha256": old.digest(identity), "update": update, "checkpoint": c.artifact(path)}
    pointer = folder/"resume.json"
    temporary = folder/("resume."+uuid.uuid4().hex+".tmp")
    old.save_new(temporary, value)
    os.replace(temporary, pointer)


def _resume(folder, identity, model, optimizer, frozen, settings):
    if not (folder/"resume.json").exists():
        return 0, []
    receipt = old.read_json(folder/"resume.json")
    c.require(receipt["identity_sha256"] == old.digest(identity), "Resume identity changed")
    payload = torch.load(c.verify_hash(receipt["checkpoint"]), map_location="cpu", weights_only=True)
    c.require(payload["identity"] == identity and payload["update"] == receipt["update"]
              and type(payload["update"]) is int and 0 <= payload["update"] <= settings["updates"]
              and len(payload["preclip_gradient_norms"]) == payload["update"], "Resume receipt changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    c.require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["lr"] == settings["lr"]
              and optimizer.param_groups[0]["weight_decay"] == settings["weight_decay"], "Resume optimizer recipe changed")
    _check_frozen(model, frozen)
    c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite resume tensor")
    return payload["update"], list(payload["preclip_gradient_norms"])


def _fit_arm(fit_id, arm, binding, base, pools, initial_ref, settings, matched, plan):
    folder = base/"fits"/f"fit_{fit_id}"/arm
    folder.mkdir(parents=True, exist_ok=True)
    identity = _identity(fit_id, arm, binding, initial_ref, settings, matched, pools)
    seed = settings["train_seeds"][str(fit_id)]
    model, parameters, optimizer, frozen, names = _model(initial_ref, arm, seed, settings)
    identity["trainable_parameter_names"] = names
    identity["trainable_parameters"] = sum(p.numel() for p in parameters)
    result_path = folder/"result.json"
    if result_path.exists():
        result = old.read_json(result_path)
        c.require(result["identity"] == identity and result["complete"] is True, "Completed arm identity changed")
        for point in result["checkpoints"]:
            c.verify_hash(point["checkpoint"])
            c.verify_hash(point["training"])
        return c.artifact(result_path, arm=arm, fit=fit_id)
    initial = _diagnostics(model, pools, base, settings)
    # Exact initial equivalence includes actual batched logits and inherited
    # training metrics, not merely a zero-weight assertion.
    previous = c.verified(c.verified(matched["result"])["arms"]["broad"]["checkpoints"][0]["training"])
    for kind in ("old", "added"):
        current = initial["added128" if kind == "added" else kind]
        previous_scope = previous[kind]
        for key in ("rows", "metrics", "initial_old_regret_units", "regret_capture_vs_initializer"):
            _compare(current[key], previous_scope[key])
    checkpoints = []

    def save(update, diagnostics):
        _check_frozen(model, frozen)
        checkpoint, report = folder/f"checkpoint_{update:04d}.pt", folder/f"training_{update:04d}.json"
        _checkpoint_model(checkpoint, model, _metadata(fit_id, arm, binding, update))
        _new_json_or_equal(report, diagnostics)
        checkpoints.append({"update": update, "checkpoint": c.artifact(checkpoint, source_fit=fit_id, arm=arm),
                            "training": c.artifact(report)})
    save(0, initial)
    start, norms = _resume(folder, identity, model, optimizer, frozen, settings)
    reporter = StatusReporter(folder, {"manifest": binding, "fit": fit_id, "arm": arm})
    for step in range(start, len(plan)):
        c.budget_boundary(base)
        model.eval()
        batch = [pools[kind][index] for kind, index in plan[step]]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*collate(batch))
        loss = objective(logits, batch, settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite precision loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)))
        optimizer.step()
        _check_frozen(model, frozen)
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite fitted tensor")
        update = step+1
        if update == 1 or update % 64 == 0 or update == settings["updates"]:
            _save_progress(folder, model, optimizer, update, norms, identity)
            reporter.event("teacher_precision_fit", update=update, total_updates=settings["updates"], resumed_at_update=start)
    c.require(len(plan) == len(norms) == settings["updates"], "Wrong final update")
    save(settings["updates"], _diagnostics(model, pools, base, settings, initial))
    result = {"identity": identity, "manifest": binding, "fit": fit_id, "arm": arm, "seed": seed,
        "checkpoints": checkpoints, "selected_update": settings["updates"], "selected_model": checkpoints[-1]["checkpoint"],
        "selected_training": checkpoints[-1]["training"], "value_head_frozen": True,
        "initial_logits_and_common128_metrics_equal_anchor": True, "model_always_eval": True,
        "trainable_parameter_names": names, "trainable_parameters": sum(p.numel() for p in parameters),
        "optimization_stability": prior._stability(norms, settings), "final_always_selected": True,
        "complete": True, "assessment_read": False, "validation_read": False, **matched["batch_receipt"]}
    old.save_new(result_path, result)
    return c.artifact(result_path, arm=arm, fit=fit_id)


def fit(fit_id, binding, base=None, *, rows=None, initializer=None, config=None, narrow=None, control=None):
    base = _base(base)
    c.budget_boundary(base)
    pools, initial, settings, selected, refs, fixture = _inputs(fit_id, binding, base, rows, initializer, config, narrow, control)
    plan = batch_plan(pools, selected, fit_id, settings)
    matched = _matched_control(fit_id, pools, initial, settings, selected, refs, plan, fixture=fixture)
    arms = {arm: _fit_arm(fit_id, arm, binding, base, pools, initial, settings, matched, plan) for arm in ARMS}
    counts = {c.verified(reference)["trainable_parameters"] for reference in arms.values()}
    c.require(len(counts) == 1, "Matched arms have unequal trainable capacity")
    result = {"manifest": binding, "fit": fit_id, "config": settings, "initializer": initial,
        "matched_broad_control": matched, "arms": arms, "complete": True,
        "assessment_read": False, "validation_read": False, "identical_between_arm_batch_stream": True}
    path = base/"fits"/f"fit_{fit_id}"/"result.json"
    _new_json_or_equal(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def _audit_arm(fit_id, arm, arm_ref, binding, base, pools, initial_ref, settings, matched, plan):
    expected = c.verified(arm_ref)
    identity = _identity(fit_id, arm, binding, initial_ref, settings, matched, pools)
    seed = settings["train_seeds"][str(fit_id)]
    model, parameters, optimizer, frozen, names = _model(initial_ref, arm, seed, settings, audit=True)
    identity.update(trainable_parameter_names=names, trainable_parameters=sum(p.numel() for p in parameters))
    c.require(expected["identity"] == identity and expected["manifest"] == binding and expected["fit"] == fit_id
              and expected["arm"] == arm and expected["seed"] == seed and expected["complete"], "Precision fit identity changed")
    c.require(expected["trainable_parameter_names"] == names
              and expected["trainable_parameters"] == sum(p.numel() for p in parameters), "Trainable membership changed")
    for key, value in matched["batch_receipt"].items():
        _compare(value, expected[key])
    checkpoints = expected["checkpoints"]
    c.require([p["update"] for p in checkpoints] == [0, settings["updates"]], "Unplanned checkpoints")
    initial = None

    def inspect(point):
        nonlocal initial
        payload = torch.load(c.verify_hash(point["checkpoint"]), map_location="cpu", weights_only=True)
        c.require(payload["metadata"] == _metadata(fit_id, arm, binding, point["update"])
                  and payload["model_kind"] == MODEL_KIND and payload["model_config"] == asdict(model.config)
                  and payload["generation"] == point["update"]
                  and _equal_torch(payload["model_state_dict"], model.state_dict()), "Independent optimizer trajectory differs")
        loaded = StrongPolicyInference(str(c.verify_hash(point["checkpoint"]))).model
        c.require(loaded.config == model.config and _equal_torch(loaded.state_dict(), model.state_dict()),
                  "Saved inference checkpoint differs")
        _check_frozen(model, frozen)
        report = _diagnostics(model, pools, base, settings, initial, audit=True)
        _compare(report, c.verified(point["training"]))
        if initial is None:
            initial = report
    inspect(checkpoints[0])
    norms = []
    reporter = StatusReporter(base/"fits"/f"fit_{fit_id}"/arm/"audit_status", {"manifest": binding, "fit": fit_id, "arm": arm})
    for step, indices in enumerate(plan):
        c.budget_boundary(base)
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        batch = []
        for kind, index in indices:
            batch.append(pools[kind][index])
        logits, _ = model(*audit_batch_tensors(batch))
        losses, _ = audit_objective(logits, batch, settings["target_temperature"])
        loss = losses.mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite audit loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
        _check_frozen(model, frozen)
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite audit tensor")
        if step == 0 or (step+1) % 64 == 0 or step+1 == settings["updates"]:
            reporter.event("teacher_precision_fit_reproduction", update=step+1, total_updates=settings["updates"])
    inspect(checkpoints[-1])
    stability = prior._stability(norms, settings, audit=True)
    _compare(stability, expected["optimization_stability"])
    c.require(expected["selected_update"] == settings["updates"]
              and expected["selected_model"] == checkpoints[-1]["checkpoint"]
              and expected["selected_training"] == checkpoints[-1]["training"]
              and not expected["assessment_read"] and not expected["validation_read"]
              and all(expected[k] is True for k in ("value_head_frozen",
                  "initial_logits_and_common128_metrics_equal_anchor", "model_always_eval", "final_always_selected")),
              "Residual final selection or frozen-state claim changed")
    return {"arm": arm, "result": arm_ref, "passed": True, "complete_fit_bitwise_identical": True,
        "all_training_metrics_independently_reproduced": True, "optimization_stability_independently_reproduced": True,
        "soft_target_arithmetic_independently_reproduced": True, "value_head_frozen": True,
        "trainable_membership_independently_reconstructed": True, "model_always_eval": True, "final_always_selected": True,
        "trainable_parameters": sum(p.numel() for p in parameters), "selected_update": settings["updates"],
        "selected_model": expected["selected_model"], "selected_training": expected["selected_training"],
        "optimization_stability": stability, **matched["batch_receipt"]}


AUDIT_FLAGS = ("passed", "complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
    "optimization_stability_independently_reproduced", "soft_target_arithmetic_independently_reproduced",
    "value_head_frozen", "trainable_membership_independently_reconstructed",
    "model_always_eval", "final_always_selected")


def audit_fit(fit_id, fit_ref, binding, base=None, *, rows=None, initializer=None, config=None, narrow=None, control=None):
    base = _base(base)
    pools, initial, settings, selected, refs, fixture = _inputs(fit_id, binding, base, rows, initializer, config, narrow, control)
    plan = independent_plan(pools, selected, fit_id, settings)
    matched = _matched_control(fit_id, pools, initial, settings, selected, refs, plan, fixture=fixture)
    expected = c.verified(fit_ref)
    c.require(expected["manifest"] == binding and expected["fit"] == fit_id and expected["complete"]
              and expected["config"] == settings and expected["initializer"] == initial
              and expected["matched_broad_control"] == matched and set(expected["arms"]) == set(ARMS)
              and expected["identical_between_arm_batch_stream"]
              and not expected["assessment_read"] and not expected["validation_read"], "Paired fit identity changed")
    arms = {arm: _audit_arm(fit_id, arm, expected["arms"][arm], binding, base, pools, initial, settings, matched, plan)
            for arm in ARMS}
    c.require(len({value["trainable_parameters"] for value in arms.values()}) == 1, "Unequal matched capacities")
    result = {"manifest": binding, "fit": fit_id, "result": fit_ref, "passed": True, "arms": arms,
        "matched_broad_control": matched, "complete_fit_bitwise_identical": True,
        "all_training_metrics_independently_reproduced": True, "identical_between_arm_batch_stream": True,
        "all_value_heads_frozen": True, "assessment_read": False, "validation_read": False}
    path = base/"fits"/f"fit_{fit_id}"/"audit.json"
    _new_json_or_equal(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def _verified_fits(fits, audits, binding, base, controls):
    c.require(len(fits) == len(audits) == len(FITS), "All three paired fits and audits required")
    models = {f"anchor_{fit}": controls[f"anchor_{fit}"] for fit in FITS}
    models.update({f"control_{fit}": controls[f"broad_{fit}"] for fit in FITS})
    for fit_id, fit_ref, audit_ref in zip(FITS, fits, audits, strict=True):
        fitted, checked = c.verified(fit_ref), c.verified(audit_ref)
        c.require(fit_ref["fit"] == audit_ref["fit"] == fitted["fit"] == checked["fit"] == fit_id
                  and fit_ref["index"] == audit_ref["index"] == FITS.index(fit_id), "Fit registry order differs")
        c.require(fitted["manifest"] == checked["manifest"] == binding and fitted["complete"]
                  and checked["passed"] and checked["result"] == fit_ref
                  and checked["complete_fit_bitwise_identical"] and checked["all_value_heads_frozen"]
                  and fitted["initializer"] == models[f"anchor_{fit_id}"]
                  and fitted["matched_broad_control"] == checked["matched_broad_control"], "Incomplete paired fitting audit")
        c.require(not any(value[key] for value in (fitted, checked) for key in ("validation_read", "assessment_read")),
                  "Held-out fitting information")
        if base.resolve().is_relative_to(OUTPUT.resolve()):
            c.require(fitted["config"] == CONFIG, "Production precision recipe changed")
        initial_payload = torch.load(c.verify_hash(models[f"anchor_{fit_id}"]), map_location="cpu", weights_only=True)
        for arm in ARMS:
            trained, audit = c.verified(fitted["arms"][arm]), checked["arms"][arm]
            c.require(audit["result"] == fitted["arms"][arm] and all(audit[key] for key in AUDIT_FLAGS), "Incomplete arm audit")
            c.require(trained["identity"]["initializer"] == fitted["initializer"]
                      and trained["identity"]["config"] == fitted["config"]
                      and trained["selected_update"] == audit["selected_update"] == fitted["config"]["updates"]
                      and trained["selected_model"] == audit["selected_model"] == trained["checkpoints"][-1]["checkpoint"]
                      and trained["selected_training"] == audit["selected_training"] == trained["checkpoints"][-1]["training"],
                      "Fixed final model linkage changed")
            for key, value in fitted["matched_broad_control"]["batch_receipt"].items():
                _compare(value, trained[key])
                _compare(value, audit[key])
            _compare(trained["optimization_stability"], audit["optimization_stability"])
            reference = trained["selected_model"]
            payload = torch.load(c.verify_hash(reference), map_location="cpu", weights_only=True)
            c.require(payload["metadata"] == _metadata(fit_id, arm, binding, fitted["config"]["updates"])
                      and payload["model_config"] == initial_payload["model_config"]
                      and payload["model_kind"] == MODEL_KIND
                      and payload["generation"] == fitted["config"]["updates"], "Final model metadata differs")
            loaded = StrongPolicyInference(str(c.verify_hash(reference))).model
            frozen = {k: v for k, v in initial_payload["model_state_dict"].items() if k.startswith("value_head.")}
            _check_frozen(loaded, frozen)
            c.verify_hash(trained["selected_training"])
            models[f"{arm}_{fit_id}"] = reference
    for key, reference in models.items():
        c.require(reference["source_fit"] == int(key.rsplit("_", 1)[1]), "Model source assignment differs")
        c.verify_hash(reference)
    return models


def models_lock(fits, audits, binding, base=None):
    from . import training_source as source
    base = _base(base)
    models = _verified_fits(fits, audits, binding, base, source.control_models(binding, base))
    result = {"manifest": binding, "models": models, "fits": fits, "audits": audits, "complete": True,
        "all_new_fits_independently_reproduced": True, "selection": "all_three_precision_fits_fixed_final",
        "assessment_used_for_selection": False, "all_value_heads_frozen": True}
    path = base/"models_lock.json"
    _new_json_or_equal(path, result)
    return c.artifact(path)


def validate_models(binding, base=None):
    from . import training_source as source
    base = _base(base)
    registry = old.read_json(base/"models_lock.json")
    c.require(registry["manifest"] == binding and registry["complete"] and registry["all_new_fits_independently_reproduced"]
              and registry["all_value_heads_frozen"]
              and registry["selection"] == "all_three_precision_fits_fixed_final" and not registry["assessment_used_for_selection"],
              "Invalid precision model seal")
    expected = _verified_fits(registry["fits"], registry["audits"], binding, base, source.control_models(binding, base))
    c.require(registry["models"] == expected, "Model registry differs from audited final models")
    return registry
