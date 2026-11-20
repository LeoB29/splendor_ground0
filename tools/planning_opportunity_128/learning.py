"""Planning-opportunity-weighted intervention on the exact completed R1 BROAD streams.

Mature pure target, reduction and two independent schedule implementations are
reused without mutating predecessor modules. Every new optimization trajectory
is independently reconstructed; frozen tensors are checked after every update.
"""
from dataclasses import asdict
from collections import Counter
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
unweighted_objective, unweighted_audit_objective = prior.objective, prior.audit_objective
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
    by_id = {row["id"]: row for row in rows}
    audit = kwargs.get("audit", False)
    for record in report["rows"]:
        record["marginal_shortlist_regret_units"] = marginal_regret_units(by_id[record["id"]], record["choice"], audit=audit)
    units = sum(record["marginal_shortlist_regret_units"] for record in report["rows"])
    if initial is None:
        initial_units = units
    else:
        before = {record["id"]: record for record in initial["rows"]}
        c.require(set(before) == set(by_id), "Marginal initial scope differs")
        initial_units = sum(marginal_regret_units(by_id[key], before[key]["choice"], audit=audit) for key in by_id)
    report["marginal_shortlist"] = {"rows": len(rows), "regret_units_sum": units,
        "initial_regret_units_sum": initial_units, "mean_score_regret": units/(256*len(rows)),
        "regret_capture_vs_initializer": (initial_units-units)/initial_units if initial_units else None,
        "value_scale": "expected_score_integer_wdl_sum_units_divided_by_256"}
    c.budget_boundary(base)
    return report


def marginal_regret_units(row, choice, *, audit=False):
    sums = row["old_sums_wdl"]
    if len(sums) <= 3:
        c.require(choice == -1, "Absent outsider choice changed")
        return 0
    c.require(type(choice) is int and 3 <= choice < len(sums), "Marginal outsider leaves recorded support")
    if audit:
        oracle = sorted(sums, reverse=True)[0]
        selected = sorted([sums[0], sums[1], sums[2], sums[choice]], reverse=True)[0]
        return oracle-selected
    prefix = max(sums[:3])
    return max(prefix, max(sums[3:]))-max(prefix, sums[choice])


def _legacy_report(report):
    """Project unchanged diagnostics for exact initial-policy control checks."""
    return {key: [{k: v for k, v in row.items() if k != "marginal_shortlist_regret_units"} for row in value]
            if key == "rows" else value for key, value in report.items() if key != "marginal_shortlist"}


def opportunity_units(row):
    values = row["old_sums_wdl"]
    c.require(1 <= len(values) <= 8 and all(type(v) is int and -128 <= v <= 128 for v in values),
              "Invalid raw opportunity values")
    return max(0, max(values[3:])-max(values[:3])) if len(values) > 3 else 0


def audit_opportunity_units(row):
    c.require(len(row["members"]) == len(row["old_sums_wdl"]), "Independent target support differs")
    first, alternatives = [], []
    for index, value in enumerate(row["old_sums_wdl"]):
        c.require(type(value) is int and -128 <= value <= 128, "Invalid independent opportunity value")
        (first if index < 3 else alternatives).append(value)
    c.require(first, "Empty recorded action support")
    if not alternatives:
        return 0
    difference = sorted(alternatives)[-1]-sorted(first)[-1]
    return difference if difference > 0 else 0


def _group(row, pool):
    # Old source-pair clustering includes both reversed-seat game legs. Added
    # data contain exactly one position per independent source game.
    return str(row["pair"]) if pool == "old" else f"{row['source_panel']}:{row['source_game']}"


def _weight_summary(rows, raw, counts, pool, *, audit=False):
    n = sum(counts)
    numerator = sum(count*value for count, value in zip(counts, raw, strict=True))
    c.require(n > 0 and numerator > 0, "Each TRAIN pool needs positive exposed planning opportunity")
    weights = [value*n/numerator for value in raw]
    masses = [count*value for count, value in zip(counts, raw, strict=True)]
    groups = {}
    if audit:
        names = sorted({_group(row, pool) for row in rows})
        for name in names:
            groups[name] = sum(masses[index] for index, row in enumerate(rows) if _group(row, pool) == name)
    else:
        for row, mass in zip(rows, masses, strict=True):
            name = _group(row, pool)
            groups[name] = groups.get(name, 0)+mass
    row_squares = sum(mass*mass for mass in masses)
    game_squares = sum(value*value for value in groups.values())
    exposure_squares = sum(count*value*value for count, value in zip(counts, raw, strict=True))
    zero_exposures = sum(count for count, value in zip(counts, raw, strict=True) if value == 0)
    inactive_exposures = sum(count for row, count, value in zip(rows, counts, raw, strict=True)
                             if value == 0 or len(row["members"]) < 5)
    active_mass = sum(mass for row, mass in zip(rows, masses, strict=True) if len(row["members"]) >= 5)
    summary = {"rows": len(rows), "exposures": n, "normalizer_opportunity_units_numerator": numerator,
        "normalizer_exposures_denominator": n, "mean_raw_score_opportunity": numerator/(256*n),
        "normalized_exposure_weight_mean": 1., "maximum_normalized_exposed_weight": max(
            weight for count, weight in zip(counts, weights, strict=True) if count),
        "zero_opportunity_rows": sum(value == 0 for value in raw),
        "zero_opportunity_row_fraction": sum(value == 0 for value in raw)/len(rows),
        "zero_opportunity_exposures": zero_exposures, "zero_opportunity_exposure_fraction": zero_exposures/n,
        "zero_loss_exposures": inactive_exposures, "zero_loss_exposure_fraction": inactive_exposures/n,
        "multiple_outsider_opportunity_mass_fraction": active_mass/numerator,
        "exposure_effective_sample_size": numerator*numerator/exposure_squares,
        "row_effective_sample_size": numerator*numerator/row_squares,
        "game_or_pair_effective_sample_size": numerator*numerator/game_squares,
        "maximum_row_weight_mass_fraction": max(masses)/numerator,
        "maximum_game_or_pair_weight_mass_fraction": max(groups.values())/numerator,
        "grouping": "whole_source_pair" if pool == "old" else "independent_source_game",
        "per_game_or_pair_opportunity_mass_units": dict(sorted(groups.items())),
        "per_row": sorted([{"id": row["id"], "index": row["index"], "group": _group(row, pool),
            "raw_opportunity_units": value, "exposures": count, "normalized_weight": weight,
            "exposure_opportunity_mass_units": mass} for row, value, count, weight, mass
            in zip(rows, raw, counts, weights, masses, strict=True)], key=lambda item: item["id"])}
    return {row["id"]: weight for row, weight in zip(rows, weights, strict=True)}, summary


def weight_plan(pools, plan):
    counts = {pool: [0]*len(rows) for pool, rows in pools.items()}
    for batch in plan:
        for pool, index in batch:
            counts[pool][index] += 1
    weights, summaries = {}, {}
    for pool, rows in pools.items():
        mapping, summaries[pool] = _weight_summary(rows, [opportunity_units(row) for row in rows], counts[pool], pool)
        weights.update(mapping)
    return weights, {"scheme": "positive_outsider_minus_first_three_max_separate_exposure_means_v1",
        "unit_score_scale": 256, "normalization_uses_train_only": True, "pools": summaries,
        "rows_pruned": 0, "batch_denominators_unchanged": True, "targets_unchanged": True}


def independent_weights(pools, plan):
    exposure = Counter((pool, index) for batch in plan for pool, index in batch)
    mapping, blocks = {}, {}
    for name in ("old", "added"):
        rows = pools[name]
        counts = [exposure[(name, index)] for index in range(len(rows))]
        raw = [audit_opportunity_units(row) for row in rows]
        # Reconstruct the scientific normalization without the collector's
        # reduction helper. The helper below supplies descriptive concentration
        # fields only; its actual weights and normalization are cross-checked.
        total, opportunity = 0, 0
        for (pool, index), count in exposure.items():
            if pool == name:
                total += count
                opportunity += count*raw[index]
        c.require(total > 0 and opportunity > 0,
                  "Each independently checked TRAIN pool needs positive exposed planning opportunity")
        values = {row["id"]: raw[index]*total/opportunity for index, row in enumerate(rows)}
        described, blocks[name] = _weight_summary(rows, raw, counts, name, audit=True)
        c.require(values == described and blocks[name]["normalizer_exposures_denominator"] == total
                  and blocks[name]["normalizer_opportunity_units_numerator"] == opportunity,
                  "Independent opportunity normalization differs")
        c.require(all(item["normalized_weight"] == values[item["id"]] for item in blocks[name]["per_row"]),
                  "Independent recorded opportunity weights differ")
        mapping.update(values)
    return mapping, {"scheme": "positive_outsider_minus_first_three_max_separate_exposure_means_v1",
        "unit_score_scale": 256, "normalization_uses_train_only": True, "pools": blocks,
        "rows_pruned": 0, "batch_denominators_unchanged": True, "targets_unchanged": True}


def objective(logits, rows, weights, temperature=.05):
    losses, choices = unweighted_objective(logits, rows, temperature)
    values = torch.tensor([weights[row["id"]] for row in rows], dtype=losses.dtype, device=losses.device)
    return losses*values, choices


def audit_objective(logits, rows, weights, temperature=.05):
    losses, choices = unweighted_audit_objective(logits, rows, temperature)
    values = losses.new_tensor([weights[rows[index]["id"]] for index in range(len(rows))])
    return torch.mul(losses, values), choices


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
    c.require(settings["value_head_frozen"] is True and settings["frozen_representation"] is False
              and settings["trainable_parameters_rule"] == "all_policy_except_value_head"
              and settings["weighting"] == "positive_best_outsider_minus_best_actual_first_three"
              and settings["weight_normalization"] == "exact_train_exposure_mean_separately_old_added"
              and settings["weight_normalization_reads_validation"] is False
              and settings["soft_targets_unchanged"] and settings["zero_weight_rows_retained"]
              and settings["zero_loss_steps_retained"] and settings["batch_denominator"] == "all_prescribed_rows",
              "Changed fixed opportunity intervention")
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
            "same_initializer_and_fixed_final": True, "only_per_row_opportunity_weight_changes": True}


def _model(initial, seed, settings, *, audit=False):
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = StrongPolicyInference(str(c.verify_hash(initial))).model
    if audit:
        excluded = {id(parameter) for parameter in model.value_head.parameters()}
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) not in excluded)
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not name.startswith("value_head."))
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for p in model.parameters() if p.requires_grad]
    c.require(parameters and all(p.dtype == torch.float32 for p in model.parameters()), "Wrong policy dtype")
    c.require(set(names) == {name for name, _ in model.named_parameters() if not name.startswith("value_head.")},
              "Only the value head may remain frozen")
    optimizer = torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])
    c.require(not optimizer.state, "Fresh optimizer required")
    frozen = {name: tensor.clone() for name, tensor in model.state_dict().items() if name not in set(names)}
    return model, parameters, optimizer, frozen, names


def _check_frozen(model, frozen):
    current = model.state_dict()
    c.require(all(torch.equal(current[name], value) for name, value in frozen.items()), "Frozen value head changed")


def _metadata(fit_id, binding):
    return {"experiment": "planning_opportunity_128", "fit": fit_id, "arm": "weighted", "manifest": binding,
            "checkpoint_selection": "fixed_final", "assessment_used": False, "opportunity_weighted": True}


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
    weights, weighting = weight_plan(pools, plan)
    folder = base/"fits"/f"fit_{fit_id}"
    folder.mkdir(parents=True, exist_ok=False)
    reporter = StatusReporter(folder, {"manifest": binding, "fit": fit_id, "arm": "weighted"})
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
            _compare(_legacy_report(report["old"]), old_report["old"])
            _compare(_legacy_report(report["added"]), old_report["added"])
            initial = report
        checkpoint, metrics = folder/f"checkpoint_{update:04d}.pt", folder/f"training_{update:04d}.json"
        save_model(checkpoint, model, update, _metadata(fit_id, binding))
        old.save_new(metrics, report)
        checkpoints.append({"update": update, "checkpoint": c.artifact(checkpoint, source_fit=fit_id, arm="weighted"),
                            "training": c.artifact(metrics)})
    save(0)
    for update, indices in enumerate(plan, 1):
        c.budget_boundary(base)
        model.train()
        batch = [pools[kind][index] for kind, index in indices]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*collate(batch))
        loss = objective(logits, batch, weights, settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite weighted loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)))
        optimizer.step()
        _check_frozen(model, frozen)
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite fitted model")
        if update == 1 or update % 64 == 0 or update == settings["updates"]:
            reporter.event("weighted_fit", update=update, total_updates=settings["updates"])
    c.require(len(plan) == settings["updates"], "Wrong fixed final update")
    save(settings["updates"])
    result = {"manifest": binding, "fit": fit_id, "arm": "weighted", "seed": seed,
        "config": settings, "initializer": initial_ref, "matched_broad_control": matched,
        "opportunity_weighting": weighting,
        "old_rows_sha256": prior._rows_identity(pools["old"]), "added_rows_sha256": prior._rows_identity(pools["added"]),
        "narrow_indices_for_original_bucketing": selected, **matched["batch_receipt"],
        "checkpoints": checkpoints, "selected_update": settings["updates"],
        "selected_model": checkpoints[-1]["checkpoint"], "selected_training": checkpoints[-1]["training"],
        "trainable_parameter_names": names, "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_tensor_names": sorted(frozen), "value_head_frozen": True,
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
    c.require(expected["manifest"] == binding and expected["fit"] == fit_id and expected["arm"] == "weighted"
              and expected["complete"] and expected["config"] == settings and expected["initializer"] == initial_ref,
              "Wrong weighted fit identity")
    c.require(expected["old_rows_sha256"] == prior._rows_identity(pools["old"])
              and expected["added_rows_sha256"] == prior._rows_identity(pools["added"])
              and expected["narrow_indices_for_original_bucketing"] == selected, "Weighted corpus or partition changed")
    plan = independent_plan(pools, selected, fit_id, settings)
    matched = _matched_control(fit_id, pools, initial_ref, settings, selected, refs, plan, fixture=fixture)
    _compare(matched, expected["matched_broad_control"])
    weights, weighting = independent_weights(pools, plan)
    _compare(weighting, expected["opportunity_weighting"])
    for key, value in matched["batch_receipt"].items():
        _compare(value, expected[key])
    seed = settings["train_seeds"][str(fit_id)]
    c.require(expected["seed"] == seed, "Changed weighted seed")
    model, parameters, optimizer, frozen, names = _model(initial_ref, seed, settings, audit=True)
    c.require(expected["trainable_parameter_names"] == names and expected["frozen_tensor_names"] == sorted(frozen)
              and expected["trainable_parameters"] == sum(p.numel() for p in parameters)
              and all(expected[key] for key in ("value_head_frozen", "final_always_selected")), "Changed trainable/frozen membership")
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
                  "Weighted checkpoint identity changed")
        c.require(payload["model_state_dict"].keys() == model.state_dict().keys(), "Checkpoint tensor set changed")
        for name, tensor in model.state_dict().items():
            c.require(torch.equal(tensor, payload["model_state_dict"][name]), "Independent weighted trajectory differs: "+name)
            if name in frozen:
                c.require(torch.equal(tensor, frozen[name]), "Independently reproduced frozen tensor moved")
        diagnostics = _diagnostics(model, pools, base, settings, initial, audit=True)
        _compare(diagnostics, c.verified(point["training"]))
        if initial is None:
            initial = diagnostics
    inspect(checkpoints[0])
    for step in range(len(plan)):
        c.budget_boundary(base)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = []
        for kind, index in plan[step]:
            batch.append(pools[kind][index])
        logits, _ = model(*audit_batch_tensors(batch))
        losses, _ = audit_objective(logits, batch, weights, settings["target_temperature"])
        loss = losses.mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite independent weighted loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
        # Independently enumerate frozen value tensors every optimization step.
        for name, value in model.state_dict().items():
            c.require(bool(torch.isfinite(value).all()), "Nonfinite reproduced tensor")
            if name not in names:
                c.require(torch.equal(value, frozen[name]), "Value tensor changed in independent fit")
        if step == 0 or (step+1) % 64 == 0 or step+1 == settings["updates"]:
            reporter.event("weighted_fit_reproduction", update=step+1, total_updates=settings["updates"])
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
        "opportunity_weights_independently_reproduced": True, "opportunity_weighting": weighting,
        "marginal_shortlist_metrics_independently_reproduced": True,
        "matched_broad_control": matched, "optimization_stability_independently_reproduced": True,
        "optimization_stability": stability,
        "trainable_membership_independently_reconstructed": True, "trainable_parameters": sum(p.numel() for p in parameters),
        "value_head_frozen": True, "final_always_selected": True,
        "selected_update": settings["updates"], "selected_model": expected["selected_model"],
        "selected_training": expected["selected_training"], **matched["batch_receipt"],
        "assessment_read": False, "validation_read": False, "elapsed_seconds": time.perf_counter()-started}
    path = base/"fits"/f"fit_{fit_id}"/"audit.json"
    old.save_new(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


AUDIT_FLAGS = ("passed", "complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
    "soft_target_arithmetic_independently_reproduced", "matched_broad_control_verified",
    "optimization_stability_independently_reproduced", "opportunity_weights_independently_reproduced",
    "marginal_shortlist_metrics_independently_reproduced",
    "trainable_membership_independently_reconstructed", "value_head_frozen", "final_always_selected")


def _verified_fits(fits, audits, binding, base, controls):
    c.require(len(fits) == len(audits) == len(FITS), "All three fits and audits required")
    models = dict(controls)
    c.require(set(models) == {f"{arm}_{fit}" for arm in ("anchor", "broad") for fit in FITS}, "Control registry differs")
    for fit_id, fit_ref, audit_ref in zip(FITS, fits, audits, strict=True):
        fitted, checked = c.verified(fit_ref), c.verified(audit_ref)
        c.require(fit_ref["fit"] == audit_ref["fit"] == fitted["fit"] == checked["fit"] == fit_id
                  and fit_ref["index"] == audit_ref["index"] == FITS.index(fit_id), "Fit registry order differs")
        c.require(fitted["manifest"] == checked["manifest"] == binding and fitted["complete"]
                  and checked["result"] == fit_ref and all(checked[key] for key in AUDIT_FLAGS), "Incomplete weighted audit")
        c.require(not any(value[key] for value in (fitted, checked) for key in ("validation_read", "assessment_read")),
                  "Held-out fitting information")
        if base.resolve().is_relative_to(OUTPUT.resolve()):
            c.require(fitted["config"] == CONFIG and fitted["trainable_parameters"] == checked["trainable_parameters"],
                      "Production weighted recipe changed")
        matched = fitted["matched_broad_control"]
        _compare(matched, checked["matched_broad_control"])
        c.require(matched["selected_model"] == controls[f"broad_{fit_id}"]
                  and matched["initializer"] == fitted["initializer"] == controls[f"anchor_{fit_id}"], "Control model linkage changed")
        for key, value in matched["batch_receipt"].items():
            _compare(value, fitted[key])
            _compare(value, checked[key])
        _compare(fitted["optimization_stability"], checked["optimization_stability"])
        _compare(fitted["opportunity_weighting"], checked["opportunity_weighting"])
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
            if name.startswith("value_head."):
                c.require(torch.equal(tensor, payload["model_state_dict"][name]), "Frozen final value head differs")
        c.verify_hash(fitted["selected_training"])
        models[f"weighted_{fit_id}"] = reference
    for key, ref in models.items():
        c.require(ref["source_fit"] == int(key.rsplit("_", 1)[1]), "Model source assignment differs")
        c.verify_hash(ref)
    return models


def models_lock(fits, audits, binding, base=None):
    from . import source
    base = _base(base)
    models = _verified_fits(fits, audits, binding, base, source.control_models(binding, base))
    result = {"manifest": binding, "models": models, "fits": fits, "audits": audits, "complete": True,
        "all_new_fits_independently_reproduced": True, "selection": "all_three_fits_fixed_final_weighted",
        "assessment_used_for_selection": False, "opportunity_weighting_verified": True}
    path = base/"models_lock.json"
    old.save_new(path, result)
    return c.artifact(path)


def validate_models(binding, base=None):
    from . import source
    base = _base(base)
    registry = old.read_json(base/"models_lock.json")
    c.require(registry["manifest"] == binding and registry["complete"]
              and registry["all_new_fits_independently_reproduced"] and registry["opportunity_weighting_verified"]
              and registry["selection"] == "all_three_fits_fixed_final_weighted"
              and not registry["assessment_used_for_selection"], "Invalid weighted model seal")
    expected = _verified_fits(registry["fits"], registry["audits"], binding, base, source.control_models(binding, base))
    c.require(registry["models"] == expected, "Model registry differs from audited final models")
    return registry
