"""Matched residual corrections, fixed final endpoints and exact replay.

Only the two authenticated TRAIN pools are opened. Both arms preserve all
original policy/value tensors and receive the completed BROAD batch stream.
Atomic optimizer checkpoints resume interrupted fitting; independent audits
reconstruct each complete trajectory from its original initialization.
"""
from pathlib import Path
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
from . import model as residual

FITS, OUTPUT, old = c.FITS, c.OUTPUT, c.old
ARMS = ("relations", "raw")
CONFIG = {**prior.CONFIG, **c.CONFIG}
RECIPE_KEYS = matched_prior.RECIPE_KEYS
soft_targets, audit_soft_targets = prior.soft_targets, prior.audit_soft_targets
objective, audit_objective = prior.objective, prior.audit_objective
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


def training_report(model, rows, base, config, initial=None, *, audit=False,
                    data_scope="train", diagnostic_scope="full_train", slice_ids=None):
    c.require(data_scope in ("train", "validation") and bool(rows), "Invalid diagnostic split")
    c.require(all(row["split"] == data_scope for row in rows), "Mixed diagnostic splits")
    c.require(all(not {"assessment", "assessment_outcomes", "fresh_outcomes", "fresh_means"} & row.keys()
                  for row in rows), "Assessment outcomes forbidden in old-target diagnostics")
    model.eval()
    records = []
    initial_by_id = None if initial is None else {row["id"]: row for row in initial["rows"]}
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
            diagnostics = diagnostic_terms(logits, batch, audit=audit)
            for row, loss, choice, terms in zip(batch, losses.tolist(), choices, diagnostics, strict=True):
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
                movement = None
                if initial_by_id is not None:
                    c.require(row["id"] in initial_by_id, "Initializer diagnostics do not cover the current scope")
                    movement = initial_policy_kl(initial_by_id[row["id"]]["outsider_logits"], terms["outsider_logits"], audit=audit)
                hard_loss = terms.pop("hard_loss")
                records.append({"id": row["id"], "pair": row["pair"], "index": row["index"],
                    "source_fit": row["source_fit"], "support": len(sums), "outsider_support": support,
                    "choice": choice, "target": target, "first_winner": correct if support else None,
                    "any_old_max": regret == 0 if support else None, "old_regret_units": regret,
                    "loss": hard_loss, **terms, "initial_policy_kl": movement})
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
    c.budget_boundary(base)
    return {"rows": records, **_report_summary(records, config, initial_records, audit=audit),
        "metrics_by_slice": slices, "data_scope": data_scope, "diagnostic_scope": diagnostic_scope,
        "diagnostic_batching": "stable_action_length_ascending",
        "loss_semantics": "hard_first_winner_cross_entropy_common_to_controls",
        "soft_metrics_dtype": "float64", "fitting_dtype": "float32",
        "target_temperature": .05,
        "initial_policy_kl_scope": "recorded_outsiders_only_KL(initial||current)",
        "initial_policy_reference": "provided_initial_report" if initial is not None else "not_provided",
        "guide_is_stopping_rule": False, "assessment_read": False,
        "validation_read": data_scope == "validation"}


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
        source.validate_features(binding, base)
        pools, initial = source.load_pools(binding, base), source.initializer(fit_id, base)
        selected, settings = source.narrow_indices(fit_id, binding, base), dict(CONFIG)
        refs = {"result": source.broad_fit(fit_id, binding, base), "audit": source.broad_audit(fit_id, binding, base)}
        c.require(len(pools["old"]) == 9461 and len(pools["added"]) == 720 and len(selected) == 180,
                  "Production population changed")
    pools, initial, settings, selected, _ = prior._inputs(fit_id, binding, base, pools, initial, settings, selected)
    c.require(set(refs) == {"result", "audit"}, "Both completed control references required")
    c.require(settings["checkpoint_selection"] == "fixed_final_update_1784"
              and not settings["selection_uses_intermediate_checkpoints"], "Checkpoint selection forbidden")
    return pools, initial, settings, selected, refs, fixture


def _matched_control(fit_id, pools, initial, settings, selected, refs, plan, *, fixture):
    receipt = matched_prior._matched_control(fit_id, pools, initial, settings, selected, refs, plan, fixture=fixture)
    # The older head-only study changed solely the parameter mask. This study
    # adds a branch, while preserving all data, optimizer and exposure choices.
    receipt.pop("only_trainable_parameter_mask_changes")
    receipt["same_unweighted_soft_targets_and_optimizer"] = True
    receipt["original_broad_fit_replayed"] = False
    return receipt


def _model(initial, arm, seed, settings, *, audit=False):
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    model = residual.create_model(c.verify_hash(initial), arm, seed)
    c.require(model.base.config.dropout == 0., "Original dropout-free base required")
    frozen_ids = {id(parameter) for parameter in model.base.parameters()}
    expected = [name for name, parameter in model.named_parameters() if id(parameter) not in frozen_ids]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    c.require(names == expected and names, "Only residual parameters may train")
    c.require(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
              == residual.TRAINABLE_PARAMETERS == 9441, "Residual capacity changed")
    if audit:
        # Independent membership uses object identity, not a name predicate.
        parameters = [parameter for parameter in model.parameters() if id(parameter) not in frozen_ids]
    else:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    c.require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), "Wrong fitting dtype")
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    c.require(not optimizer.state, "Fresh optimizer required")
    model.eval()
    frozen = {name: tensor.clone() for name, tensor in model.base.state_dict().items()}
    return model, parameters, optimizer, frozen, names


def _check_frozen(model, frozen):
    current = model.base.state_dict()
    c.require(current.keys() == frozen.keys() and all(torch.equal(current[k], v) for k, v in frozen.items()),
              "Original policy/value tensor changed")
    c.require(all(not module.training for module in model.base.modules()), "Frozen base left evaluation mode")


def _metadata(fit_id, arm, binding, update):
    return {"experiment": "action_card_relations_128", "fit": fit_id, "arm": arm, "manifest": binding,
            "update": update, "checkpoint_selection": "fixed_final", "assessment_used": False,
            "complete_original_policy_and_value_frozen": True}


def _diagnostics(model, pools, base, settings, initial=None, *, audit=False):
    return {kind: training_report(model, pools[kind], base, settings,
                None if initial is None else initial[kind], audit=audit,
                diagnostic_scope="full_"+kind+"_train") for kind in ("old", "added")}


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
        c.require(saved["metadata"] == metadata and saved["arm"] == model.arm and saved["seed"] == model.seed
                  and saved["model_kind"] == residual.MODEL_KIND and saved["residual_config"] == residual.MODEL_CONFIG
                  and _equal_torch(saved["model_state_dict"], model.state_dict()),
                  "Existing final model differs")
        return
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    residual.save_model(model, temporary, metadata)
    os.replace(temporary, path)


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
        current = initial[kind]
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
        c.require(bool(torch.isfinite(loss)), "Nonfinite residual loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)))
        optimizer.step()
        _check_frozen(model, frozen)
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite fitted tensor")
        update = step+1
        if update == 1 or update % 64 == 0 or update == settings["updates"]:
            _save_progress(folder, model, optimizer, update, norms, identity)
            reporter.event("residual_fit", update=update, total_updates=settings["updates"], resumed_at_update=start)
    c.require(len(plan) == len(norms) == settings["updates"], "Wrong final update")
    save(settings["updates"], _diagnostics(model, pools, base, settings, initial))
    result = {"identity": identity, "manifest": binding, "fit": fit_id, "arm": arm, "seed": seed,
        "checkpoints": checkpoints, "selected_update": settings["updates"], "selected_model": checkpoints[-1]["checkpoint"],
        "selected_training": checkpoints[-1]["training"], "complete_original_policy_and_value_frozen": True,
        "initial_logits_and_training_metrics_equal_anchor": True, "base_always_eval": True,
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
              and expected["arm"] == arm and expected["seed"] == seed and expected["complete"], "Residual fit identity changed")
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
                  and payload["arm"] == arm and payload["seed"] == seed
                  and payload["model_kind"] == residual.MODEL_KIND and payload["residual_config"] == residual.MODEL_CONFIG
                  and payload["generation"] == point["update"]
                  and _equal_torch(payload["model_state_dict"], model.state_dict()), "Independent optimizer trajectory differs")
        loaded = residual.RelationInference(c.verify_hash(point["checkpoint"])).model
        c.require(loaded.arm == arm and loaded.seed == seed and loaded.config == model.config
                  and _equal_torch(loaded.state_dict(), model.state_dict()), "Saved inference checkpoint differs")
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
        for name, value in model.base.state_dict().items():
            c.require(torch.equal(value, frozen[name]), "Original tensor changed during independent fit")
        c.require(all(not module.training for module in model.base.modules()), "Frozen base mode changed")
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite audit tensor")
        if step == 0 or (step+1) % 64 == 0 or step+1 == settings["updates"]:
            reporter.event("residual_fit_reproduction", update=step+1, total_updates=settings["updates"])
    inspect(checkpoints[-1])
    stability = prior._stability(norms, settings, audit=True)
    _compare(stability, expected["optimization_stability"])
    c.require(expected["selected_update"] == settings["updates"]
              and expected["selected_model"] == checkpoints[-1]["checkpoint"]
              and expected["selected_training"] == checkpoints[-1]["training"]
              and not expected["assessment_read"] and not expected["validation_read"]
              and all(expected[k] is True for k in ("complete_original_policy_and_value_frozen",
                  "initial_logits_and_training_metrics_equal_anchor", "base_always_eval", "final_always_selected")),
              "Residual final selection or frozen-state claim changed")
    return {"arm": arm, "result": arm_ref, "passed": True, "complete_fit_bitwise_identical": True,
        "all_training_metrics_independently_reproduced": True, "optimization_stability_independently_reproduced": True,
        "soft_target_arithmetic_independently_reproduced": True, "complete_original_policy_and_value_frozen": True,
        "trainable_membership_independently_reconstructed": True, "base_always_eval": True, "final_always_selected": True,
        "trainable_parameters": sum(p.numel() for p in parameters), "selected_update": settings["updates"],
        "selected_model": expected["selected_model"], "selected_training": expected["selected_training"],
        "optimization_stability": stability, **matched["batch_receipt"]}


AUDIT_FLAGS = ("passed", "complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
    "optimization_stability_independently_reproduced", "soft_target_arithmetic_independently_reproduced",
    "complete_original_policy_and_value_frozen", "trainable_membership_independently_reconstructed",
    "base_always_eval", "final_always_selected")


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
        "all_original_policy_and_value_tensors_frozen": True, "assessment_read": False, "validation_read": False}
    path = base/"fits"/f"fit_{fit_id}"/"audit.json"
    _new_json_or_equal(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def _verified_fits(fits, audits, binding, base, controls):
    c.require(len(fits) == len(audits) == len(FITS), "All three paired fits and audits required")
    models = {f"anchor_{fit}": controls[f"anchor_{fit}"] for fit in FITS}
    for fit_id, fit_ref, audit_ref in zip(FITS, fits, audits, strict=True):
        fitted, checked = c.verified(fit_ref), c.verified(audit_ref)
        c.require(fit_ref["fit"] == audit_ref["fit"] == fitted["fit"] == checked["fit"] == fit_id
                  and fit_ref["index"] == audit_ref["index"] == FITS.index(fit_id), "Fit registry order differs")
        c.require(fitted["manifest"] == checked["manifest"] == binding and fitted["complete"]
                  and checked["passed"] and checked["result"] == fit_ref
                  and checked["complete_fit_bitwise_identical"] and checked["all_original_policy_and_value_tensors_frozen"]
                  and fitted["initializer"] == models[f"anchor_{fit_id}"]
                  and fitted["matched_broad_control"] == checked["matched_broad_control"], "Incomplete paired fitting audit")
        c.require(not any(value[key] for value in (fitted, checked) for key in ("validation_read", "assessment_read")),
                  "Held-out fitting information")
        if base.resolve().is_relative_to(OUTPUT.resolve()):
            c.require(fitted["config"] == CONFIG, "Production residual recipe changed")
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
                      and payload["arm"] == arm and payload["seed"] == fitted["config"]["train_seeds"][str(fit_id)]
                      and payload["model_config"] == initial_payload["model_config"]
                      and payload["model_kind"] == residual.MODEL_KIND and payload["residual_config"] == residual.MODEL_CONFIG
                      and payload["generation"] == fitted["config"]["updates"], "Final model metadata differs")
            loaded = residual.RelationInference(c.verify_hash(reference)).model
            c.require(_equal_torch(loaded.base.state_dict(), initial_payload["model_state_dict"]), "Original final base differs")
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
        "all_new_fits_independently_reproduced": True, "selection": "all_six_residual_fits_fixed_final",
        "assessment_used_for_selection": False, "all_original_policy_and_value_tensors_frozen": True}
    path = base/"models_lock.json"
    _new_json_or_equal(path, result)
    return c.artifact(path)


def validate_models(binding, base=None):
    from . import training_source as source
    base = _base(base)
    registry = old.read_json(base/"models_lock.json")
    c.require(registry["manifest"] == binding and registry["complete"] and registry["all_new_fits_independently_reproduced"]
              and registry["all_original_policy_and_value_tensors_frozen"]
              and registry["selection"] == "all_six_residual_fits_fixed_final" and not registry["assessment_used_for_selection"],
              "Invalid residual model seal")
    expected = _verified_fits(registry["fits"], registry["audits"], binding, base, source.control_models(binding, base))
    c.require(registry["models"] == expected, "Model registry differs from audited final models")
    return registry
