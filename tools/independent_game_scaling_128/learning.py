"""Fixed-dose nested game coverage, with independently reproduced trajectories.

The loss and diagnostic mathematics come from the authenticated coverage
experiment. Only a private report namespace is rebound to this experiment's
resource boundary; importing this module never changes an ancestor module.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
import random

import torch

from splendor_ai.alphazero.fast_selector_157 import collate
from scripts.analysis.verify_fast_selector_transfer_157 import audit_batch_tensors
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, save_model, MODEL_KIND
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.independent_game_coverage_128_recovery_01 import learning as prior
from tools.teacher_precision_512 import learning as checkpoint_helpers
from . import common as c

FITS, OUTPUT, old = c.FITS, c.OUTPUT, c.old
ARMS = c.LEARNING_ARMS
CONFIG = {**prior.CONFIG, **c.CONFIG,
    "reference_updates": 1784, "schedule_repeats": 4,
    "curve_old_rows": 256, "curve_added_rows_per_cell": 8,
    "curve_tag": "independent-game-scaling-128-identity-probes-v1",
    "training_schedule": "three_nested_pools_shared_old_and_18cell_cycles_paired_length_buckets_v1",
    "diagnostic_checkpoints": "full_initial_final_and_five_fixed_small_train_probes",
    "added_diagnostics": "each_arms_own_optimizer_exposed_train_pool",
}


def _clone(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


# These functions do no inference and contain no execution-budget callbacks.
_compare, _rows_identity, _validate_rows = prior._compare, prior._rows_identity, prior._validate_rows
soft_targets, audit_soft_targets = prior.soft_targets, prior.audit_soft_targets
objective, audit_objective = prior.objective, prior.audit_objective
_equal_torch = checkpoint_helpers._equal_torch
_new_json_or_equal = checkpoint_helpers._new_json_or_equal
_checkpoint_model = checkpoint_helpers._checkpoint_model
_save_progress, _resume = checkpoint_helpers._save_progress, checkpoint_helpers._resume
_check_frozen = checkpoint_helpers._check_frozen
training_report = _clone(prior.training_report, {"c": c})


def _base(base):
    return Path(OUTPUT if base is None else base)


def _cell(row):
    return tuple(row[key] for key in ("source_fit", "seat", "phase"))


def _inputs(fit_id, arm, binding, base, rows, initializer, config, partitions):
    c.require(fit_id in FITS and arm in ARMS, "Unknown fit or arm")
    supplied = (rows, initializer, config, partitions)
    fixture = any(value is not None for value in supplied)
    if fixture:
        c.require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Fixture cannot target production")
        c.require(all(value is not None for value in supplied), "Complete fixture inputs required")
        pools, initial, settings, subsets = rows, initializer, dict(config), partitions
    else:
        from . import training_source as source
        pools, initial = source.load_pools(binding, base), source.initializer(fit_id, base)
        settings, subsets = dict(CONFIG), source.partitions(binding, base)
    c.require(set(pools) == {"old", "added"} and set(subsets) == set(ARMS), "Unexpected training pools")
    for values in pools.values():
        _validate_rows(values)
        c.require(all(not {"teacher_worlds", "teacher_sums_wdl", "teacher_outsider"} & row.keys()
                      for row in values), "Only original128 targets may enter this experiment")
    c.require(not {r["id"] for r in pools["old"]} & {r["id"] for r in pools["added"]}, "Overlapping pools")
    games = [(r["source_panel"], r["source_game"]) for r in pools["added"]]
    c.require(len(games) == len(set(games)), "Added games repeated")
    cells = Counter(_cell(row) for row in pools["added"])
    for name in ARMS:
        indices = subsets[name]
        c.require(len(indices) == settings["sizes"][name] and len(indices) == len(set(indices))
                  and all(type(i) is int and 0 <= i < len(pools["added"]) for i in indices), "Invalid partition")
        counts = Counter(_cell(pools["added"][i]) for i in indices)
        c.require(counts.keys() == cells.keys() and len(set(counts.values())) == 1, "Unbalanced source cells")
    c.require(set(subsets["small"]) < set(subsets["medium"]) < set(subsets["large"])
              and subsets["large"] == list(range(len(pools["added"]))), "Partitions must be strictly nested")
    c.require(settings["target_temperature"] == .05 and settings["old_fraction"] == .5
              and settings["objective"] == "recorded_outside_first_three_soft_score_cross_entropy"
              and settings["value_head_frozen"] is True and settings["fresh_optimizer"] is True
              and not settings["selection_uses_intermediate_checkpoints"]
              and settings["updates"] == settings["reference_updates"] * settings["schedule_repeats"]
              and settings["schedule_repeats"] == 4 and settings["batch_size"] > 1,
              "Fixed learning recipe changed")
    c.require(settings["diagnostic_updates"] == [i*settings["reference_updates"] for i in range(5)],
              "Diagnostic checkpoints changed")
    if not fixture:
        c.require(len(pools["old"]) == 9461 and len(pools["added"]) == 11520
                  and len(cells) == 18 and set(cells.values()) == {640}, "Production population changed")
    c.verify_hash(initial)
    return pools, initial, settings, {name: list(subsets[name]) for name in ARMS}, fixture


def _model(initial, arm, seed, settings, *, audit=False):
    c.require(arm in ARMS, "Unknown learner arm")
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = StrongPolicyInference(str(c.verify_hash(initial))).model
    c.require(model.config.dropout == 0., "Dropout-free policy required")
    frozen_ids = {id(parameter) for parameter in model.value_head.parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(id(parameter) not in frozen_ids if audit else not name.startswith("value_head."))
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = [name for name, parameter in model.named_parameters() if id(parameter) not in frozen_ids]
    c.require(names == expected and names, "All policy and only policy parameters must train")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    c.require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), "Wrong dtype")
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    c.require(not optimizer.state, "Fresh optimizer required")
    model.eval()
    frozen = {name: value.clone() for name, value in model.state_dict().items() if name.startswith("value_head.")}
    return model, parameters, optimizer, frozen, names


def _reference_lengths(rows, fit_id, config):
    reference = prior._reference_lengths(rows, fit_id, {**config, "updates": config["reference_updates"]})
    return reference * config["schedule_repeats"]


_cycling_indices = prior._cycling_indices


def _probe_indices(pools, selected, config):
    def rank(pool, indices):
        return sorted(indices, key=lambda i: (old.digest([config["curve_tag"], pool, pools[pool][i]["id"]]),
                                             pools[pool][i]["id"]))
    old_indices = rank("old", range(len(pools["old"])))[:config["curve_old_rows"]]
    added = []
    for cell in sorted({_cell(pools["added"][i]) for i in selected}):
        domain = [i for i in selected if _cell(pools["added"][i]) == cell]
        added.extend(rank("added", domain)[:config["curve_added_rows_per_cell"]])
    return {"old": old_indices, "added": added}


def _diagnostics(model, pools, selected, base, settings, initial=None, *, audit=False, probe=False):
    indices = _probe_indices(pools, selected, settings) if probe else {
        "old": list(range(len(pools["old"]))), "added": list(selected)}
    return {pool: training_report(model, [pools[pool][i] for i in indices[pool]], base, settings,
        None if initial is None else initial[pool], audit=audit,
        diagnostic_scope=("fixed_identity_probe_" if probe else "full_own_train_")+pool)
        for pool in ("old", "added")}


def _metadata(fit_id, arm, binding, update):
    return {"experiment": "independent_game_scaling_128", "fit": fit_id, "arm": arm, "manifest": binding,
            "update": update, "checkpoint_selection": "fixed_final", "assessment_used": False,
            "value_head_frozen": True}


def _identity(fit_id, arm, binding, initial, settings, subsets, pools, paired):
    return {"manifest": binding, "fit": fit_id, "arm": arm, "initializer": initial, "config": settings,
            "partitions": subsets, "paired_batch_receipt": paired,
            "old_rows_sha256": _rows_identity(pools["old"]),
            "added_rows_sha256": _rows_identity(pools["added"])}


def _job_index(fit_id, arm):
    return FITS.index(fit_id)*len(ARMS)+ARMS.index(arm)


def fit(fit_id, arm, binding, base=None, *, rows=None, initializer=None, config=None, partitions=None):
    base = _base(base)
    c.budget_boundary(base)
    pools, initial_ref, settings, subsets, fixture = _inputs(
        fit_id, arm, binding, base, rows, initializer, config, partitions)
    plans = batch_plan(pools, subsets, fit_id, settings)
    paired = _paired_receipt(plans, pools, settings, fixture=fixture)
    plan = plans[arm]
    identity = _identity(fit_id, arm, binding, initial_ref, settings, subsets, pools, paired)
    seed = settings["train_seeds"][str(fit_id)]
    model, parameters, optimizer, frozen, names = _model(initial_ref, arm, seed, settings)
    identity.update(trainable_parameter_names=names, trainable_parameters=sum(p.numel() for p in parameters))
    folder = base/"fits"/f"fit_{fit_id}"/arm
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/"result.json"
    if result_path.exists():
        result = old.read_json(result_path)
        c.require(result["identity"] == identity and result["complete"] is True, "Completed fit identity changed")
        for point in result["checkpoints"]:
            for key in ("checkpoint", "training", "receipt"):
                c.verify_hash(point[key])
        return c.artifact(result_path, arm=arm, fit=fit_id, index=_job_index(fit_id, arm))
    selected = subsets[arm]
    initial_full = _diagnostics(model, pools, selected, base, settings)
    initial_probe = _diagnostics(model, pools, selected, base, settings, probe=True)

    def save(update, full=None, probe=None):
        _check_frozen(model, frozen)
        checkpoint, report = folder/f"checkpoint_{update:04d}.pt", folder/f"training_{update:04d}.json"
        _checkpoint_model(checkpoint, model, _metadata(fit_id, arm, binding, update))
        _new_json_or_equal(report, {"full": full, "probe": probe, "update": update,
                                  "assessment_read": False, "validation_read": False})
        point = {"update": update, "checkpoint": c.artifact(checkpoint, source_fit=fit_id, arm=arm),
                 "training": c.artifact(report), "identity_sha256": old.digest(identity)}
        receipt = folder/f"point_{update:04d}.json"
        _new_json_or_equal(receipt, point)
        return {**point, "receipt": c.artifact(receipt)}

    save(0, initial_full, initial_probe)
    start, norms = _resume(folder, identity, model, optimizer, frozen, settings)
    reporter = StatusReporter(folder, {"manifest": binding, "fit": fit_id, "arm": arm})
    for step in range(start, len(plan)):
        c.budget_boundary(base)
        model.eval()
        batch = [pools[kind][index] for kind, index in plan[step]]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*collate(batch))
        loss = objective(logits, batch, settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite fitting loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)))
        optimizer.step()
        _check_frozen(model, frozen)
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite fitted tensor")
        update = step+1
        if update in settings["diagnostic_updates"]:
            full = _diagnostics(model, pools, selected, base, settings, initial_full) if update == settings["updates"] else None
            save(update, full, _diagnostics(model, pools, selected, base, settings, initial_probe, probe=True))
        # A diagnostic point is durable before any recovery checkpoint can skip it.
        if update == 1 or update % settings["resume_checkpoint_interval"] == 0 or update in settings["diagnostic_updates"]:
            _save_progress(folder, model, optimizer, update, norms, identity)
            reporter.event("independent_game_scaling_fit", update=update, total_updates=settings["updates"], resumed_at_update=start)
    c.require(len(plan) == len(norms) == settings["updates"], "Wrong final update")
    points = []
    for update in settings["diagnostic_updates"]:
        ref = c.artifact(folder/f"point_{update:04d}.json")
        point = c.verified(ref)
        c.require(point["identity_sha256"] == old.digest(identity) and point["update"] == update, "Point identity changed")
        c.verify_hash(point["checkpoint"])
        c.verify_hash(point["training"])
        points.append({**point, "receipt": ref})
    result = {"identity": identity, "manifest": binding, "fit": fit_id, "arm": arm, "seed": seed,
        "initializer": initial_ref, "config": settings, "checkpoints": points,
        "selected_update": settings["updates"], "selected_model": points[-1]["checkpoint"],
        "selected_training": points[-1]["training"], "value_head_frozen": True, "model_always_eval": True,
        "trainable_parameter_names": names, "trainable_parameters": sum(p.numel() for p in parameters),
        "optimization_stability": prior._stability(norms, settings), "final_always_selected": True,
        "complete": True, "assessment_read": False, "validation_read": False, **paired["arms"][arm]}
    old.save_new(result_path, result)
    return c.artifact(result_path, arm=arm, fit=fit_id, index=_job_index(fit_id, arm))


AUDIT_FLAGS = ("passed", "complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
    "optimization_stability_independently_reproduced", "soft_target_arithmetic_independently_reproduced",
    "value_head_frozen", "trainable_membership_independently_reconstructed", "model_always_eval", "final_always_selected")


def audit_fit(fit_id, arm, fit_ref, binding, base=None, *, rows=None, initializer=None, config=None, partitions=None):
    base = _base(base)
    c.budget_boundary(base)
    pools, initial_ref, settings, subsets, fixture = _inputs(
        fit_id, arm, binding, base, rows, initializer, config, partitions)
    plans = independent_plan(pools, subsets, fit_id, settings)
    paired = _paired_receipt(plans, pools, settings, fixture=fixture)
    identity = _identity(fit_id, arm, binding, initial_ref, settings, subsets, pools, paired)
    seed = settings["train_seeds"][str(fit_id)]
    model, parameters, optimizer, frozen, names = _model(initial_ref, arm, seed, settings, audit=True)
    identity.update(trainable_parameter_names=names, trainable_parameters=sum(p.numel() for p in parameters))
    expected = c.verified(fit_ref)
    c.require(expected["identity"] == identity and expected["manifest"] == binding and expected["fit"] == fit_id
              and expected["arm"] == arm and expected["seed"] == seed and expected["complete"], "Fit identity changed")
    c.require(expected["config"] == settings and expected["initializer"] == initial_ref
              and expected["trainable_parameter_names"] == names
              and expected["trainable_parameters"] == sum(p.numel() for p in parameters), "Fit membership changed")
    for key, value in paired["arms"][arm].items():
        _compare(value, expected[key])
    points = expected["checkpoints"]
    c.require([p["update"] for p in points] == settings["diagnostic_updates"], "Unplanned checkpoints")
    initial_full = initial_probe = None

    def inspect(point):
        nonlocal initial_full, initial_probe
        receipt = c.verified(point["receipt"])
        c.require(receipt == {key: value for key, value in point.items() if key != "receipt"}
                  and point["identity_sha256"] == old.digest(identity), "Diagnostic point receipt changed")
        payload = torch.load(c.verify_hash(point["checkpoint"]), map_location="cpu", weights_only=True)
        c.require(payload["metadata"] == _metadata(fit_id, arm, binding, point["update"])
                  and payload["generation"] == point["update"] and payload["model_kind"] == MODEL_KIND
                  and payload["model_config"] == asdict(model.config)
                  and _equal_torch(payload["model_state_dict"], model.state_dict()), "Independent optimizer trajectory differs")
        loaded = StrongPolicyInference(str(c.verify_hash(point["checkpoint"]))).model
        c.require(_equal_torch(loaded.state_dict(), model.state_dict()), "Saved inference model differs")
        _check_frozen(model, frozen)
        endpoint = point["update"] in (0, settings["updates"])
        full = _diagnostics(model, pools, subsets[arm], base, settings, initial_full, audit=True) if endpoint else None
        probe = _diagnostics(model, pools, subsets[arm], base, settings, initial_probe, audit=True, probe=True)
        _compare({"full": full, "probe": probe, "update": point["update"],
                  "assessment_read": False, "validation_read": False}, c.verified(point["training"]))
        if point["update"] == 0:
            initial_full, initial_probe = full, probe
    inspect(points[0])
    points_by_update = {point["update"]: point for point in points[1:]}
    norms = []
    reporter = StatusReporter(base/"fits"/f"fit_{fit_id}"/arm/"audit_status", {"manifest": binding, "fit": fit_id, "arm": arm})
    for step, indices in enumerate(plans[arm]):
        c.budget_boundary(base)
        model.eval()
        batch = [pools[kind][index] for kind, index in indices]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*audit_batch_tensors(batch))
        loss = audit_objective(logits, batch, settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite audit loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)))
        optimizer.step()
        _check_frozen(model, frozen)
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite audit tensor")
        if step+1 in points_by_update:
            inspect(points_by_update[step+1])
        if step == 0 or (step+1) % settings["resume_checkpoint_interval"] == 0 or step+1 == settings["updates"]:
            reporter.event("independent_game_scaling_reproduction", update=step+1, total_updates=settings["updates"])
    stability = prior._stability(norms, settings, audit=True)
    _compare(stability, expected["optimization_stability"])
    c.require(expected["selected_update"] == settings["updates"]
              and expected["selected_model"] == points[-1]["checkpoint"]
              and expected["selected_training"] == points[-1]["training"]
              and not expected["assessment_read"] and not expected["validation_read"]
              and all(expected[key] is True for key in ("value_head_frozen", "model_always_eval", "final_always_selected")),
              "Final selection or frozen-state claim changed")
    result = {"manifest": binding, "fit": fit_id, "arm": arm, "result": fit_ref,
        **{flag: True for flag in AUDIT_FLAGS}, "assessment_read": False, "validation_read": False,
        "selected_update": settings["updates"], "selected_model": expected["selected_model"],
        "selected_training": expected["selected_training"], "trainable_parameters": sum(p.numel() for p in parameters),
        "paired_batch_receipt": paired, "optimization_stability": stability}
    path = base/"fits"/f"fit_{fit_id}"/arm/"audit.json"
    _new_json_or_equal(path, result)
    return c.artifact(path, arm=arm, fit=fit_id, index=_job_index(fit_id, arm))


def _verified_fits(fits, audits, binding, base, controls):
    jobs = [(fit_id, arm) for fit_id in FITS for arm in ARMS]
    c.require(len(fits) == len(audits) == len(jobs), "All nine fits and audits required")
    c.require(set(controls) == {f"{arm}_{fit_id}" for arm in ("anchor", "control") for fit_id in FITS},
              "Original anchor and saved short-dose controls required")
    models, by_fit = dict(controls), {}
    for index, ((fit_id, arm), fit_ref, audit_ref) in enumerate(zip(jobs, fits, audits, strict=True)):
        fitted, checked = c.verified(fit_ref), c.verified(audit_ref)
        c.require(fit_ref["index"] == audit_ref["index"] == index
                  and all(v["fit"] == fit_id and v["arm"] == arm for v in (fit_ref, audit_ref, fitted, checked)),
                  "Fit registry order differs")
        c.require(fitted["manifest"] == checked["manifest"] == binding and fitted["complete"]
                  and checked["result"] == fit_ref and all(checked[key] is True for key in AUDIT_FLAGS)
                  and fitted["initializer"] == models[f"anchor_{fit_id}"]
                  and checked["selected_model"] == fitted["selected_model"]
                  and checked["selected_training"] == fitted["selected_training"]
                  and checked["paired_batch_receipt"] == fitted["identity"]["paired_batch_receipt"],
                  "Incomplete nine-fit reproduction evidence")
        c.require(not any(v[key] for v in (fitted, checked) for key in ("assessment_read", "validation_read")),
                  "Held-out fitting information")
        if base.resolve().is_relative_to(OUTPUT.resolve()):
            c.require(fitted["config"] == CONFIG and fitted["selected_update"] == 7136, "Production recipe changed")
        summary = (fitted["identity"]["paired_batch_receipt"], fitted["trainable_parameter_names"], fitted["trainable_parameters"])
        if fit_id in by_fit:
            c.require(summary == by_fit[fit_id], "Between-arm schedule or trainable capacity differs")
        by_fit[fit_id] = summary
        model_ref = fitted["selected_model"]
        payload = torch.load(c.verify_hash(model_ref), map_location="cpu", weights_only=True)
        c.require(payload["metadata"] == _metadata(fit_id, arm, binding, fitted["config"]["updates"])
                  and payload["model_kind"] == MODEL_KIND, "Final model metadata changed")
        models[f"{arm}_{fit_id}"] = model_ref
    for reference in models.values():
        c.verify_hash(reference)
    return models


def models_lock(fits, audits, binding, base=None, *, controls=None):
    base = _base(base)
    if controls is None:
        from . import training_source as source
        controls = source.control_models(binding, base)
    else:
        c.require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Fixture controls forbidden in production")
    models = _verified_fits(fits, audits, binding, base, controls)
    result = {"manifest": binding, "models": models, "fits": fits, "audits": audits, "complete": True,
        "all_new_fits_independently_reproduced": True, "selection": "all_nine_fits_fixed_final",
        "assessment_used_for_selection": False, "all_value_heads_frozen": True}
    path = base/"models_lock.json"
    _new_json_or_equal(path, result)
    return c.artifact(path)


def validate_models(binding, base=None):
    from . import training_source as source
    base = _base(base)
    registry = old.read_json(base/"models_lock.json")
    c.require(registry["manifest"] == binding and registry["complete"]
              and registry["all_new_fits_independently_reproduced"] and registry["all_value_heads_frozen"]
              and registry["selection"] == "all_nine_fits_fixed_final"
              and not registry["assessment_used_for_selection"], "Invalid scaling model seal")
    expected = _verified_fits(registry["fits"], registry["audits"], binding, base, source.control_models(binding, base))
    c.require(registry["models"] == expected, "Registry differs from independently reproduced models")
    return registry


# Three-arm generalizations of the separately implemented coverage samplers.
def batch_plan(pools, subsets, fit_id, config=CONFIG):
    """Shared old IDs and cell slots; new row identity is the intervention."""
    tag = f"{config['batch_tag']}|{config['train_seeds'][str(fit_id)]}"
    lengths = _reference_lengths(pools["old"], fit_id, config)
    c.require(sum(lengths) % 2 == 0, "Balanced mixture needs even total exposure")
    old_stream = _cycling_indices(range(len(pools["old"])), tag+"|old")
    cells = sorted({_cell(row) for row in pools["added"]})
    cell_stream = _cycling_indices(cells, tag+"|cells")
    streams = {}
    for arm, chosen in subsets.items():
        streams[arm] = {cell: _cycling_indices([i for i in chosen if _cell(pools["added"][i]) == cell],
                                               tag+"|pool|"+old.digest(list(cell))) for cell in cells}
    plan = {arm: [] for arm in ARMS}
    odd = 0
    for update, length in enumerate(lengths):
        old_count = length//2 + (length % 2 and odd % 2 == 0)
        odd += length % 2
        old_indices = [next(old_stream) for _ in range(old_count)]
        new_cells = [next(cell_stream) for _ in range(length-old_count)]
        positions = [("old", i) for i in range(old_count)]+[("added", i) for i in range(length-old_count)]
        random.Random(f"{tag}|mixed|{update}").shuffle(positions)
        for arm in ARMS:
            new_indices = [next(streams[arm][cell]) for cell in new_cells]
            plan[arm].append([(kind, old_indices[i] if kind == "old" else new_indices[i]) for kind, i in positions])
    # Reorder paired exposures using legal tensor lengths only. This avoids
    # paying maximum action-set padding on every old/new mixture minibatch.
    old_draws = [i for batch in plan["small"] for kind, i in batch if kind == "old"]
    paired_new = [tuple(plan[arm][bi][position][1] for arm in ARMS)
                  for bi, batch in enumerate(plan["small"])
                  for position, (kind, _) in enumerate(batch) if kind == "added"]
    old_draws.sort(key=lambda i: len(pools["old"][i]["features"]))
    paired_new.sort(key=lambda pair: max(len(pools["added"][i]["features"]) for i in pair))
    order = list(range(len(lengths)))
    random.Random(tag+"|length-batches").shuffle(order)
    result = {arm: [None]*len(lengths) for arm in ARMS}
    old_at = new_at = 0
    for update in order:
        for arm_index, arm in enumerate(ARMS):
            oi, ni = old_at, new_at
            batch = []
            for kind, _ in plan[arm][update]:
                index = old_draws[oi] if kind == "old" else paired_new[ni][arm_index]
                batch.append((kind, index))
                oi += kind == "old"
                ni += kind == "added"
            result[arm][update] = batch
        old_at, new_at = oi, ni
    c.require(old_at == len(old_draws) and new_at == len(paired_new), "Length bucketing lost exposure")
    return result


def independent_plan(pools, subsets, fit_id, config=CONFIG):
    """Reconstruct lengths and all three sampling layers without collector helpers."""
    seed = config["train_seeds"][str(fit_id)]
    tag = "%s|%s" % (config["batch_tag"], seed)
    lengths = []
    epoch = 0
    while len(lengths) < config["reference_updates"]:
        rng = random.Random("%s|%s|%s" % (config["reference_batch_tag"], seed, epoch))
        rows = list(range(len(pools["old"])))
        rng.shuffle(rows)
        ordered = sorted(rows, key=lambda i: pools["old"][i]["features"].shape[0])
        blocks = []
        for start in range(0, len(ordered), config["batch_size"]):
            blocks.append(ordered[start:min(start+config["batch_size"], len(ordered))])
        rng.shuffle(blocks)
        lengths += [len(block) for block in blocks]
        epoch += 1
    lengths = lengths[:config["reference_updates"]] * config["schedule_repeats"]
    c.require(sum(lengths) % 2 == 0, "Balanced mixture needs even total exposure")
    cells = sorted({(row["source_fit"], row["seat"], row["phase"]) for row in pools["added"]})
    cache, offsets, cycles = {}, {}, {}
    def take(key, domain, namespace):
        position = offsets.get(key, 0)
        if key not in cache or position == len(cache[key]):
            cycle = cycles.get(key, 0)
            values = list(domain)
            random.Random("%s|%s" % (namespace, cycle)).shuffle(values)
            cache[key], offsets[key], cycles[key] = values, 0, cycle+1
        value = cache[key][offsets[key]]
        offsets[key] += 1
        return value
    domains = {}
    for arm in ARMS:
        indices = list(subsets[arm])
        domains[arm] = {cell: [i for i in indices if tuple(pools["added"][i][key] for key in
                            ("source_fit", "seat", "phase")) == cell] for cell in cells}
    out = {arm: [] for arm in ARMS}
    next_extra_old = True
    for update, length in enumerate(lengths):
        counts = {"old": length//2, "added": length//2}
        if length % 2:
            counts["old" if next_extra_old else "added"] += 1
            next_extra_old = not next_extra_old
        old_rows = [take("old", range(len(pools["old"])), tag+"|old") for _ in range(counts["old"])]
        new_cells = [take("cells", cells, tag+"|cells") for _ in range(counts["added"])]
        order = [(kind, i) for kind in ("old", "added") for i in range(counts[kind])]
        random.Random("%s|mixed|%s" % (tag, update)).shuffle(order)
        for arm in ARMS:
            additions = []
            for cell in new_cells:
                namespace = tag+"|pool|"+old.digest(list(cell))
                additions.append(take((arm, cell), domains[arm][cell], namespace))
            out[arm].append([(kind, old_rows[i] if kind == "old" else additions[i]) for kind, i in order])
    old_draws, new_draws = [], []
    for batch_index in range(len(lengths)):
        for position, (kind, row_index) in enumerate(out["small"][batch_index]):
            if kind == "old":
                old_draws.append(row_index)
            else:
                new_draws.append(tuple(out[arm][batch_index][position][1] for arm in ARMS))
    old_draws = sorted(old_draws, key=lambda i: pools["old"][i]["features"].shape[0])
    new_draws = sorted(new_draws, key=lambda indices: max(pools["added"][i]["features"].shape[0] for i in indices))
    placement = list(range(len(lengths)))
    random.Random("%s|length-batches" % tag).shuffle(placement)
    rebuilt = {arm: [None for _ in lengths] for arm in ARMS}
    old_position = added_position = 0
    for block in placement:
        selected_old = sum(kind == "old" for kind, _ in out["small"][block])
        for arm_index in range(len(ARMS)):
            consumed = {"old": 0, "added": 0}
            batch = []
            for kind, _ in out[ARMS[arm_index]][block]:
                index = (old_draws[old_position+consumed[kind]] if kind == "old"
                         else new_draws[added_position+consumed[kind]][arm_index])
                batch.append((kind, index))
                consumed[kind] += 1
            rebuilt[ARMS[arm_index]][block] = batch
        old_position += selected_old
        added_position += lengths[block]-selected_old
    c.require(old_position == len(old_draws) and added_position == len(new_draws), "Audit bucket exposure differs")
    return rebuilt


def _plan_receipt(plan, pools):
    ids, old_ids, cells, counts, games, old_games = [], [], [], Counter(), Counter(), Counter()
    kind_counts = Counter()
    for batch in plan:
        ids.append([])
        old_ids.append([])
        cells.append([])
        for kind, index in batch:
            row = pools[kind][index]
            ids[-1].append([kind, row["id"]])
            counts[row["id"]] += 1
            kind_counts[kind] += 1
            if kind == "old":
                old_ids[-1].append(row["id"])
                old_games[row["id"].rsplit(":", 1)[0]] += 1
            else:
                cells[-1].append(list(_cell(row)))
                games[f"{row['source_panel']}:{row['source_game']}"] += 1
    cell_counts = Counter(tuple(cell) for batch in cells for cell in batch)
    padded_lengths = [max(len(pools[kind][i]["features"]) for kind, i in batch) for batch in plan]
    return {"batch_lengths": [len(batch) for batch in plan], "updates": len(plan),
        "row_exposures": sum(kind_counts.values()), "old_exposures": kind_counts["old"],
        "added_exposures": kind_counts["added"], "per_row_exposures": dict(sorted(counts.items())),
        "per_added_game_exposures": dict(sorted(games.items())),
        "per_old_game_exposures": dict(sorted(old_games.items())),
        "per_added_cell_exposures": {old.digest(list(cell)): cell_counts[cell] for cell in sorted(cell_counts)},
        "added_cells": [{"cell": list(cell), "exposures": cell_counts[cell]} for cell in sorted(cell_counts)],
        "batch_sequence_sha256": old.digest(ids), "old_batch_stream_sha256": old.digest(old_ids),
        "added_cell_batch_stream_sha256": old.digest(cells),
        "maximum_actions_per_batch": padded_lengths,
        "attention_padding_proxy": sum(len(batch)*length*length for batch, length in zip(plan, padded_lengths, strict=True))}


def _paired_receipt(plans, pools, config, *, fixture):
    receipts = {arm: _plan_receipt(plans[arm], pools) for arm in ARMS}
    reference = receipts[ARMS[0]]
    equal = ("batch_lengths", "updates", "row_exposures", "old_exposures", "added_exposures",
             "per_old_game_exposures", "per_added_cell_exposures", "added_cells",
             "old_batch_stream_sha256", "added_cell_batch_stream_sha256")
    c.require(all(reference[key] == receipt[key] for receipt in receipts.values() for key in equal),
              "Between-arm old or cell exposure mismatch")
    c.require(reference["old_exposures"] == reference["added_exposures"], "Mixture must be exactly half old")
    for batches in zip(*(plans[arm] for arm in ARMS), strict=True):
        c.require(len({len(batch) for batch in batches}) == 1, "Paired batch lengths changed")
        for slots in zip(*batches, strict=True):
            kind, index = slots[0]
            c.require(all(other_kind == kind and (other_index == index if kind == "old" else
                _cell(pools["added"][other_index]) == _cell(pools["added"][index]))
                for other_kind, other_index in slots), "Paired batch slots differ")
        count = sum(kind == "old" for kind, _ in batches[0])
        c.require(abs(2*count-len(batches[0])) <= 1, "Batch violates half-old mixture")
    c.require(reference["updates"] == config["updates"], "Wrong fixed update count")
    for arm in ARMS:
        c.require(len(receipts[arm]["per_added_game_exposures"]) == config["sizes"][arm], "Some added games were not exposed")
    if not fixture:
        c.require(reference["row_exposures"] == 456176 and reference["old_exposures"] == 228088
                  and Counter(reference["batch_lengths"]) == {64: 7088, 53: 48}, "Production dose changed")
    return {"arms": receipts, "same_batch_lengths": True, "same_old_rows_in_every_batch": True,
            "same_new_cell_in_every_batch_slot": True, "equal_total_old_and_added_exposures": True}
