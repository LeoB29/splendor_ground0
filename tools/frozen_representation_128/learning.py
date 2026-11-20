"""Existing-head-only intervention on the exact completed R1 BROAD streams.

Mature pure target, reduction and two independent schedule implementations are
reused without mutating predecessor modules. Every new optimization trajectory
is independently reconstructed; frozen tensors are checked after every update.
"""
from dataclasses import asdict
from pathlib import Path
import time

import torch

from splendor_ai.alphazero.strong_policy import StrongPolicyInference, save_model, MODEL_KIND
from splendor_ai.alphazero.fast_selector_157 import collate
from scripts.analysis.verify_fast_selector_transfer_157 import audit_batch_tensors
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.independent_game_coverage_128_recovery_01 import learning as prior
from . import common as c

FITS, OUTPUT, old = c.FITS, c.OUTPUT, c.old
CONFIG = {**prior.CONFIG, **c.CONFIG}
RECIPE_KEYS = (
    "train_rows", "added_rows", "narrow_rows_per_fit", "old_fraction", "updates", "batch_size",
    "row_exposures_per_fit", "lr", "weight_decay", "gradient_norm", "value_head_frozen",
    "fresh_optimizer", "target_temperature", "objective", "train_seeds", "batch_tag",
    "reference_batch_tag", "training_schedule", "checkpoint_selection",
    "selection_uses_intermediate_checkpoints", "torch_threads", "device", "dtype",
)
soft_targets, audit_soft_targets = prior.soft_targets, prior.audit_soft_targets
objective, audit_objective = prior.objective, prior.audit_objective
_metrics, _compare = prior._metrics, prior._compare


def _base(base):
    return Path(OUTPUT if base is None else base)


class _BudgetModel(torch.nn.Module):
    """Apply this study's ceiling to each unchanged diagnostic forward call."""
    def __init__(self, model, base):
        super().__init__()
        self.model, self.base = model, base

    def forward(self, *args):
        c.budget_boundary(self.base)
        return self.model(*args)


def training_report(model, rows, base, config, initial=None, **kwargs):
    base = _base(base)
    c.budget_boundary(base)
    report = prior.training_report(_BudgetModel(model, base), rows, base, config, initial, **kwargs)
    c.budget_boundary(base)
    return report


def _inputs(fit_id, binding, base, rows, initializer, config, narrow, control):
    c.require(fit_id in FITS, "Unknown fit")
    supplied = (rows, initializer, config, narrow, control)
    fixture = any(value is not None for value in supplied)
    if fixture:
        c.require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Fixture cannot target production")
        c.require(all(value is not None for value in supplied), "Complete fixture inputs required")
        pools, initial, settings, selected, refs = rows, initializer, dict(config), list(narrow), control
    else:
        from . import source
        pools, initial = source.load_pools(binding, base), source.initializer(fit_id, base)
        selected, settings = source.narrow_indices(fit_id, binding, base), dict(CONFIG)
        refs = {"result": source.broad_fit(fit_id, binding, base), "audit": source.broad_audit(fit_id, binding, base)}
        c.require(len(pools["old"]) == 9461 and len(pools["added"]) == 720 and len(selected) == 180,
                  "Production population changed")
    # Explicit overrides use only the mature population checks, never its source
    # accessors or fit machinery. New production/fixture boundaries are above.
    pools, initial, settings, selected, _ = prior._inputs(
        fit_id, binding, base, pools, initial, settings, selected)
    c.require(settings["trainable_prefix"] == "score_head." and settings["frozen_representation"] is True,
              "Intervention must train only the existing score head")
    if not fixture:
        c.require(settings["trainable_parameters"] == 98817, "Production head size changed")
    c.require(set(refs) == {"result", "audit"}, "Both completed control references required")
    return pools, initial, settings, selected, refs, fixture


def batch_plan(pools, narrow, fit_id, config=CONFIG):
    return prior.batch_plan(pools, narrow, fit_id, config)["broad"]


def independent_plan(pools, narrow, fit_id, config=CONFIG):
    return prior.independent_plan(pools, narrow, fit_id, config)["broad"]


def _matched_control(fit_id, pools, initial, settings, selected, refs, plan, *, fixture):
    trained, checked = c.verified(refs["result"]), c.verified(refs["audit"])
    c.require(trained["fit"] == checked["fit"] == fit_id and trained["complete"] and checked["passed"]
              and checked["result"] == refs["result"] and trained["manifest"] == checked["manifest"]
              and checked["complete_fit_bitwise_identical"]
              and checked["soft_target_arithmetic_independently_reproduced"]
              and checked["exact_between_arm_old_and_cell_streams"], "Incomplete matched BROAD control")
    c.require(trained["initializer"] == initial and trained["narrow_indices"] == selected
              and trained["narrow_row_ids"] == [pools["added"][i]["id"] for i in selected],
              "Matched control initializer or pairing partition differs")
    c.require(trained["old_rows_sha256"] == prior._rows_identity(pools["old"])
              and trained["added_rows_sha256"] == prior._rows_identity(pools["added"]),
              "Matched control corpus differs")
    recipe = {key: settings[key] for key in RECIPE_KEYS}
    c.require(recipe == {key: trained["config"][key] for key in RECIPE_KEYS}, "Matched control fitting recipe differs")
    arm, audit = trained["arms"]["broad"], checked["arms"]["broad"]
    c.require(arm["selected_update"] == audit["selected_update"] == settings["updates"]
              and arm["selected_model"] == audit["selected_model"]
              and arm["selected_training"] == audit["selected_training"]
              and arm["selected_model"] == arm["checkpoints"][-1]["checkpoint"]
              and arm["final_always_selected"] and audit["final_always_selected"], "Matched control final selection changed")
    c.require(all(audit[key] for key in ("complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
        "soft_target_arithmetic_independently_reproduced", "optimization_stability_independently_reproduced",
        "value_head_frozen")), "Incomplete matched control arm audit")
    c.require(not any(value[flag] for value in (trained, checked, arm) for flag in ("assessment_read", "validation_read")),
              "Control fit used held-out outcomes")
    receipt = prior._plan_receipt(plan, pools)
    for expected in (trained["paired_schedule"]["arms"]["broad"], checked["paired_schedule"]["arms"]["broad"]):
        _compare(receipt, expected)
    for key, value in receipt.items():
        _compare(value, arm[key])
        _compare(value, audit[key])
    if not fixture:
        c.require(receipt["updates"] == 1784 and receipt["row_exposures"] == 114044
                  and receipt["old_exposures"] == receipt["added_exposures"] == 57022
                  and len(receipt["per_added_game_exposures"]) == 720, "Fixed production dose changed")
    c.verify_hash(arm["selected_model"])
    return {"result": refs["result"], "audit": refs["audit"], "selected_model": arm["selected_model"],
            "initializer": initial, "recipe": recipe, "batch_receipt": receipt,
            "same_training_rows": True, "same_batch_stream_and_per_row_exposures": True,
            "same_initializer_and_fixed_final": True, "only_trainable_parameter_mask_changes": True}


def _model(initial, seed, settings, *, audit=False):
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = StrongPolicyInference(str(c.verify_hash(initial))).model
    c.require(model.config.dropout == 0., "Frozen representation requires the original dropout-free model")
    if audit:
        # Determine membership independently from the existing score_head object,
        # rather than the collector's parameter-name predicate.
        allowed = {id(parameter) for parameter in model.score_head.parameters()}
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in allowed)
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(settings["trainable_prefix"]))
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for p in model.parameters() if p.requires_grad]
    c.require(names == ["score_head.0.weight", "score_head.0.bias", "score_head.2.weight", "score_head.2.bias"],
              "Trainable head membership changed")
    c.require(sum(p.numel() for p in parameters) == settings["trainable_parameters"]
              and all(p.dtype == torch.float32 for p in model.parameters()), "Wrong trainable count or dtype")
    model.eval()
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    c.require(not optimizer.state, "Fresh optimizer required")
    frozen = {name: tensor.clone() for name, tensor in model.state_dict().items() if name not in set(names)}
    return model, parameters, optimizer, frozen, names


def _check_frozen(model, frozen):
    current = model.state_dict()
    c.require(all(torch.equal(current[name], value) for name, value in frozen.items()),
              "Non-score-head tensor changed")
    c.require(all(not module.training for name, module in model.named_modules() if not name.startswith("score_head")),
              "Frozen feature module left evaluation mode")


def _metadata(fit_id, binding):
    return {"experiment": "frozen_representation_128", "fit": fit_id, "arm": "headonly", "manifest": binding,
            "checkpoint_selection": "fixed_final", "assessment_used": False, "frozen_representation": True}


def _diagnostics(model, pools, base, settings, initial=None, *, audit=False):
    ids = {row["id"] for row in pools["added"]}
    return {"old": training_report(model, pools["old"], base, settings,
                None if initial is None else initial["old"], audit=audit, diagnostic_scope="full_old_train"),
            "added": training_report(model, pools["added"], base, settings,
                None if initial is None else initial["added"], audit=audit,
                diagnostic_scope="full_added_train_pool_with_exposure_slices",
                slice_ids={"optimizer_exposed": ids, "not_optimizer_exposed": set()}),
            "all_added_rows_retired_from_future_tests": True,
            "assessment_read": False, "validation_read": False}


def fit(fit_id, binding, base=None, *, rows=None, initializer=None, config=None, narrow=None, control=None):
    base = _base(base)
    c.budget_boundary(base)
    started = time.perf_counter()
    pools, initial_ref, settings, selected, refs, fixture = _inputs(
        fit_id, binding, base, rows, initializer, config, narrow, control)
    plan = batch_plan(pools, selected, fit_id, settings)
    matched = _matched_control(fit_id, pools, initial_ref, settings, selected, refs, plan, fixture=fixture)
    folder = base/"fits"/f"fit_{fit_id}"
    folder.mkdir(parents=True, exist_ok=False)
    reporter = StatusReporter(folder, {"manifest": binding, "fit": fit_id, "arm": "headonly"})
    seed = settings["train_seeds"][str(fit_id)]
    model, parameters, optimizer, frozen, names = _model(initial_ref, seed, settings)
    initial, checkpoints, norms = None, [], []

    def save(update):
        nonlocal initial
        c.budget_boundary(base)
        _check_frozen(model, frozen)
        report = _diagnostics(model, pools, base, settings, initial)
        if initial is None:
            # Both arms begin with the very same deployed policy, including
            # diagnostic batching and canonical alias reduction numerics.
            old_point = c.verified(refs["result"])["arms"]["broad"]["checkpoints"][0]
            c.require(old_point["update"] == 0, "Matched control lacks its initializer")
            old_report = c.verified(old_point["training"])
            _compare(report["old"], old_report["old"])
            _compare(report["added"], old_report["added"])
            initial = report
        checkpoint, metrics = folder/f"checkpoint_{update:04d}.pt", folder/f"training_{update:04d}.json"
        save_model(checkpoint, model, update, _metadata(fit_id, binding))
        old.save_new(metrics, report)
        checkpoints.append({"update": update, "checkpoint": c.artifact(checkpoint, source_fit=fit_id, arm="headonly"),
                            "training": c.artifact(metrics)})
    save(0)
    for update, indices in enumerate(plan, 1):
        c.budget_boundary(base)
        model.eval()
        model.score_head.train()
        batch = [pools[kind][index] for kind, index in indices]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*collate(batch))
        loss = objective(logits, batch, settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite head-only loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)))
        optimizer.step()
        _check_frozen(model, frozen)
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite fitted model")
        if update == 1 or update % 64 == 0 or update == settings["updates"]:
            reporter.event("headonly_fit", update=update, total_updates=settings["updates"])
    c.require(len(plan) == settings["updates"], "Wrong fixed final update")
    save(settings["updates"])
    result = {"manifest": binding, "fit": fit_id, "arm": "headonly", "seed": seed,
        "config": settings, "initializer": initial_ref, "matched_broad_control": matched,
        "old_rows_sha256": prior._rows_identity(pools["old"]), "added_rows_sha256": prior._rows_identity(pools["added"]),
        "narrow_indices_for_original_bucketing": selected, **matched["batch_receipt"],
        "checkpoints": checkpoints, "selected_update": settings["updates"],
        "selected_model": checkpoints[-1]["checkpoint"], "selected_training": checkpoints[-1]["training"],
        "trainable_parameter_names": names, "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_tensor_names": sorted(frozen), "every_non_score_head_tensor_frozen": True,
        "frozen_features_always_eval": True, "value_head_frozen": True,
        "optimization_stability": prior._stability(norms, settings), "final_always_selected": True,
        "complete": True, "assessment_read": False, "validation_read": False,
        "elapsed_seconds": time.perf_counter()-started}
    path = folder/"result.json"
    old.save_new(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def audit_fit(fit_id, fit_ref, binding, base=None, *, rows=None, initializer=None, config=None, narrow=None, control=None):
    base = _base(base)
    c.budget_boundary(base)
    started = time.perf_counter()
    pools, initial_ref, settings, selected, refs, fixture = _inputs(
        fit_id, binding, base, rows, initializer, config, narrow, control)
    expected = c.verified(fit_ref)
    c.require(expected["manifest"] == binding and expected["fit"] == fit_id and expected["arm"] == "headonly"
              and expected["complete"] and expected["config"] == settings and expected["initializer"] == initial_ref,
              "Wrong head-only fit identity")
    c.require(expected["old_rows_sha256"] == prior._rows_identity(pools["old"])
              and expected["added_rows_sha256"] == prior._rows_identity(pools["added"])
              and expected["narrow_indices_for_original_bucketing"] == selected, "Head-only corpus or partition changed")
    plan = independent_plan(pools, selected, fit_id, settings)
    matched = _matched_control(fit_id, pools, initial_ref, settings, selected, refs, plan, fixture=fixture)
    _compare(matched, expected["matched_broad_control"])
    for key, value in matched["batch_receipt"].items():
        _compare(value, expected[key])
    seed = settings["train_seeds"][str(fit_id)]
    c.require(expected["seed"] == seed, "Changed head-only seed")
    model, parameters, optimizer, frozen, names = _model(initial_ref, seed, settings, audit=True)
    c.require(expected["trainable_parameter_names"] == names and expected["frozen_tensor_names"] == sorted(frozen)
              and expected["trainable_parameters"] == sum(p.numel() for p in parameters)
              and all(expected[key] for key in ("every_non_score_head_tensor_frozen", "frozen_features_always_eval",
                                                "value_head_frozen", "final_always_selected")), "Changed trainable/frozen membership")
    checkpoints = expected["checkpoints"]
    c.require([point["update"] for point in checkpoints] == [0, settings["updates"]], "Unplanned checkpoints")
    reporter = StatusReporter(base/"fits"/f"fit_{fit_id}"/"audit_status", {"manifest": binding, "fit": fit_id})
    initial, norms = None, []

    def inspect(point):
        nonlocal initial
        c.budget_boundary(base)
        payload = torch.load(c.verify_hash(point["checkpoint"]), map_location="cpu", weights_only=True)
        c.require(payload["model_kind"] == MODEL_KIND and payload["model_config"] == asdict(model.config)
                  and payload["generation"] == point["update"] and payload["metadata"] == _metadata(fit_id, binding),
                  "Head-only checkpoint identity changed")
        c.require(payload["model_state_dict"].keys() == model.state_dict().keys(), "Checkpoint tensor set changed")
        for name, tensor in model.state_dict().items():
            c.require(torch.equal(tensor, payload["model_state_dict"][name]), "Independent head-only trajectory differs: "+name)
            if name in frozen:
                c.require(torch.equal(tensor, frozen[name]), "Independently reproduced frozen tensor moved")
        diagnostics = _diagnostics(model, pools, base, settings, initial, audit=True)
        _compare(diagnostics, c.verified(point["training"]))
        if initial is None:
            initial = diagnostics
    inspect(checkpoints[0])
    for step in range(len(plan)):
        c.budget_boundary(base)
        model.eval()
        model.score_head.train()
        optimizer.zero_grad(set_to_none=True)
        batch = []
        for kind, index in plan[step]:
            batch.append(pools[kind][index])
        logits, _ = model(*audit_batch_tensors(batch))
        losses, _ = audit_objective(logits, batch, settings["target_temperature"])
        loss = losses.mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite independent head-only loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
        # Independently enumerate all non-head tensors every optimization step.
        for name, value in model.state_dict().items():
            c.require(bool(torch.isfinite(value).all()), "Nonfinite reproduced tensor")
            if name not in names:
                c.require(torch.equal(value, frozen[name]), "Non-head tensor changed in independent fit")
        c.require(not model.training and all(not module.training for name, module in model.named_modules()
                  if name.split('.')[0] != "score_head"), "Independent frozen feature mode changed")
        if step == 0 or (step+1) % 64 == 0 or step+1 == settings["updates"]:
            reporter.event("headonly_fit_reproduction", update=step+1, total_updates=settings["updates"])
    inspect(checkpoints[-1])
    c.require(expected["selected_update"] == settings["updates"]
              and expected["selected_model"] == checkpoints[-1]["checkpoint"]
              and expected["selected_training"] == checkpoints[-1]["training"], "Changed fixed final linkage")
    c.require(not expected["assessment_read"] and not expected["validation_read"], "Forbidden fitting information")
    stability = prior._stability(norms, settings, audit=True)
    _compare(stability, expected["optimization_stability"])
    result = {"manifest": binding, "fit": fit_id, "result": fit_ref, "passed": True,
        "complete_fit_bitwise_identical": True, "all_training_metrics_independently_reproduced": True,
        "soft_target_arithmetic_independently_reproduced": True, "matched_broad_control_verified": True,
        "matched_broad_control": matched, "optimization_stability_independently_reproduced": True,
        "optimization_stability": stability, "every_non_score_head_tensor_frozen": True,
        "trainable_membership_independently_reconstructed": True, "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_features_always_eval": True, "value_head_frozen": True, "final_always_selected": True,
        "selected_update": settings["updates"], "selected_model": expected["selected_model"],
        "selected_training": expected["selected_training"], **matched["batch_receipt"],
        "assessment_read": False, "validation_read": False, "elapsed_seconds": time.perf_counter()-started}
    path = base/"fits"/f"fit_{fit_id}"/"audit.json"
    old.save_new(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


AUDIT_FLAGS = ("passed", "complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
    "soft_target_arithmetic_independently_reproduced", "matched_broad_control_verified",
    "optimization_stability_independently_reproduced", "every_non_score_head_tensor_frozen",
    "trainable_membership_independently_reconstructed", "frozen_features_always_eval", "value_head_frozen", "final_always_selected")


def _verified_fits(fits, audits, binding, base, controls):
    c.require(len(fits) == len(audits) == len(FITS), "All three fits and audits required")
    models = dict(controls)
    c.require(set(models) == {f"{arm}_{fit}" for arm in ("anchor", "broad") for fit in FITS}, "Control registry differs")
    for fit_id, fit_ref, audit_ref in zip(FITS, fits, audits, strict=True):
        fitted, checked = c.verified(fit_ref), c.verified(audit_ref)
        c.require(fit_ref["fit"] == audit_ref["fit"] == fitted["fit"] == checked["fit"] == fit_id
                  and fit_ref["index"] == audit_ref["index"] == FITS.index(fit_id), "Fit registry order differs")
        c.require(fitted["manifest"] == checked["manifest"] == binding and fitted["complete"]
                  and checked["result"] == fit_ref and all(checked[key] for key in AUDIT_FLAGS), "Incomplete head-only audit")
        c.require(not any(value[key] for value in (fitted, checked) for key in ("validation_read", "assessment_read")),
                  "Held-out fitting information")
        if base.resolve().is_relative_to(OUTPUT.resolve()):
            c.require(fitted["config"] == CONFIG and fitted["trainable_parameters"] == checked["trainable_parameters"] == 98817,
                      "Production head-only recipe changed")
        matched = fitted["matched_broad_control"]
        _compare(matched, checked["matched_broad_control"])
        c.require(matched["selected_model"] == controls[f"broad_{fit_id}"]
                  and matched["initializer"] == fitted["initializer"] == controls[f"anchor_{fit_id}"], "Control model linkage changed")
        for key, value in matched["batch_receipt"].items():
            _compare(value, fitted[key])
            _compare(value, checked[key])
        _compare(fitted["optimization_stability"], checked["optimization_stability"])
        c.require(fitted["selected_update"] == checked["selected_update"] == fitted["config"]["updates"]
                  and fitted["selected_model"] == checked["selected_model"] == fitted["checkpoints"][-1]["checkpoint"]
                  and fitted["selected_training"] == checked["selected_training"] == fitted["checkpoints"][-1]["training"],
                  "Fixed final model linkage changed")
        reference = fitted["selected_model"]
        payload = torch.load(c.verify_hash(reference), map_location="cpu", weights_only=True)
        original = torch.load(c.verify_hash(fitted["initializer"]), map_location="cpu", weights_only=True)
        c.require(payload["model_kind"] == MODEL_KIND and payload["model_config"] == original["model_config"]
                  and payload["generation"] == fitted["config"]["updates"]
                  and payload["metadata"] == _metadata(fit_id, binding), "Final checkpoint identity differs")
        c.require(payload["model_state_dict"].keys() == original["model_state_dict"].keys(), "Final tensor set changed")
        for name, tensor in original["model_state_dict"].items():
            if not name.startswith("score_head."):
                c.require(torch.equal(tensor, payload["model_state_dict"][name]), "Frozen final representation differs")
        c.verify_hash(fitted["selected_training"])
        models[f"headonly_{fit_id}"] = reference
    for key, ref in models.items():
        c.require(ref["source_fit"] == int(key.rsplit("_", 1)[1]), "Model source assignment differs")
        c.verify_hash(ref)
    return models


def models_lock(fits, audits, binding, base=None):
    from . import source
    base = _base(base)
    models = _verified_fits(fits, audits, binding, base, source.control_models(binding, base))
    result = {"manifest": binding, "models": models, "fits": fits, "audits": audits, "complete": True,
        "all_new_fits_independently_reproduced": True, "selection": "all_three_fits_fixed_final_headonly",
        "assessment_used_for_selection": False, "frozen_representations_verified": True}
    path = base/"models_lock.json"
    old.save_new(path, result)
    return c.artifact(path)


def validate_models(binding, base=None):
    from . import source
    base = _base(base)
    registry = old.read_json(base/"models_lock.json")
    c.require(registry["manifest"] == binding and registry["complete"]
              and registry["all_new_fits_independently_reproduced"] and registry["frozen_representations_verified"]
              and registry["selection"] == "all_three_fits_fixed_final_headonly"
              and not registry["assessment_used_for_selection"], "Invalid head-only model seal")
    expected = _verified_fits(registry["fits"], registry["audits"], binding, base, source.control_models(binding, base))
    c.require(registry["models"] == expected, "Model registry differs from audited final models")
    return registry
