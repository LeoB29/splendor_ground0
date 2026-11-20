"""Paired narrow/broad learning with exactly matched old and new-cell exposure.

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

FITS, OUTPUT, old = c.FITS, c.OUTPUT, c.old
ARMS = ("narrow", "broad")
CONFIG = {**c.CONFIG,
    "reference_batch_tag": "focused-teacher-fit-128-training-v1",
    "regret_capture_guide": .90,
    "training_schedule": "shared_old_and_new_cell_cycles_paired_length_buckets_v1",
    "diagnostic_checkpoints": "initial_and_fixed_final_only",
    "added_diagnostics": "all720_with_optimizer_exposed_and_unexposed_slices",
}


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


def _stability(norms, config, *, audit=False):
    c.require(all(math.isfinite(value) and value >= 0 for value in norms), "Invalid pre-clip gradient norm")
    exceeded = (sum(1 for value in norms if value > config["gradient_norm"]) if audit
                else sum(value > config["gradient_norm"] for value in norms))
    # Canonical stable reduction permits exact receipt linkage even for large
    # finite gradients. The audited norms come from a separately reproduced
    # optimizer trajectory, not the collector's saved vector.
    total = math.fsum(norms)
    return {"updates": len(norms), "gradient_clip_limit": config["gradient_norm"],
            "preclip_gradient_norms": norms,
            "preclip_gradient_norm_mean": total/len(norms) if norms else None,
            "preclip_gradient_norm_max": max(norms) if norms else None,
            "norm_above_clip_limit_steps": exceeded,
            "norm_above_clip_limit_fraction": exceeded/len(norms) if norms else None,
            "all_preclip_norms_finite": True, "all_postupdate_model_tensors_finite": True}


def soft_targets(row, *, dtype=torch.float32, device=None, temperature=.05):
    """Finite score-gap preferences; these are not calibrated confidence."""
    c.require(temperature == .05, "Target temperature must remain fixed at .05")
    sums = row["old_sums_wdl"][row["outsider_start"]:]
    q = torch.tensor([(128+value)/256 for value in sums], dtype=dtype, device=device)
    return torch.softmax(q/temperature, dim=0)


def audit_soft_targets(row, *, dtype=torch.float32, device=None, temperature=.05):
    c.require(temperature == .05, "Target temperature must remain fixed at .05")
    # Independently reconstruct the support and WDL -> expected-score map.
    values = [float(row["old_sums_wdl"][j])/256+.5 for j in range(3, len(row["members"]))]
    scaled = torch.tensor(values, dtype=dtype, device=device)/temperature
    # Sharing this Torch primitive keeps the independent optimizer replay bitwise.
    return F.softmax(scaled, dim=0)


def objective(logits, rows, temperature=.05):
    losses, choices = [], []
    for i, row in enumerate(rows):
        start, width, target = row["outsider_start"], row["outsider_support"], row["outsider_teacher"]
        if width < 2:
            losses.append(logits[i, row["members"][0][0]]*0.)
            choices.append(target)
        else:
            scores = torch.stack([torch.amax(logits[i, member]) for member in row["members"][start:]])
            probabilities = soft_targets(row, dtype=scores.dtype, device=scores.device, temperature=temperature)
            losses.append(-(probabilities*F.log_softmax(scores, dim=0)).sum())
            choices.append(start+int(scores.detach().argmax()))
    return torch.stack(losses), choices


def audit_objective(logits, rows, temperature=.05):
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
            weights = audit_soft_targets(row, dtype=scores.dtype, device=scores.device, temperature=temperature)
            losses.append(-torch.sum(weights*torch.log_softmax(scores, dim=0)))
            choices.append(offered[int(scores.detach().argmax())])
    return torch.stack(losses), choices


def diagnostic_terms(logits, rows, *, audit=False):
    """Common hard CE plus float64 finite-target and confidence diagnostics."""
    result = []
    for i, row in enumerate(rows):
        offered = list(range(3, len(row["members"])))
        if audit:
            groups = [int(row["groups"][row["members"][j][0]]) for j in offered]
            scores = torch.stack([torch.amax(logits[i, torch.where(row["groups"] == group)[0]])
                                  for group in groups]) if offered else logits.new_empty(0)
            target = max(offered, key=lambda j: row["old_sums_wdl"][j]) if offered else -1
        else:
            scores = torch.stack([torch.amax(logits[i, member]) for member in row["members"][3:]]) if offered else logits.new_empty(0)
            target = row["outsider_teacher"]
        values = scores.detach().tolist()
        if len(offered) < 2:
            result.append({"hard_loss": 0., "soft_loss": 0., "target_entropy": 0.,
                           "target_kl": 0., "policy_entropy": 0., "outsider_logits": values})
            continue
        hard = float(F.cross_entropy(scores[None], torch.tensor([target-3], device=scores.device)))
        if audit:
            # Independent scalar normalization and reduction, not collector terms.
            raw = [(row["old_sums_wdl"][j]/256+.5)/.05 for j in offered]
            maximum = max(raw)
            total = math.fsum(math.exp(value-maximum) for value in raw)
            log_t = [value-maximum-math.log(total) for value in raw]
            t = [math.exp(value) for value in log_t]
            peak = max(values)
            denominator = math.fsum(math.exp(value-peak) for value in values)
            log_p = [value-peak-math.log(denominator) for value in values]
            soft = -math.fsum(weight*value for weight, value in zip(t, log_p))
            entropy = -math.fsum(weight*value for weight, value in zip(t, log_t))
            kl = math.fsum(weight*(left-right) for weight, left, right in zip(t, log_t, log_p))
            policy_entropy = -math.fsum(math.exp(value)*value for value in log_p)
        else:
            q = torch.tensor([(128+row["old_sums_wdl"][j])/256 for j in offered], dtype=torch.float64)
            log_t = torch.log_softmax(q/.05, dim=0)
            t = log_t.exp()
            log_p = torch.log_softmax(scores.detach().double(), dim=0)
            soft = float(-(t*log_p).sum())
            entropy = float(-(t*log_t).sum())
            kl = float((t*(log_t-log_p)).sum())
            policy_entropy = float(-(log_p.exp()*log_p).sum())
        c.require(all(math.isfinite(value) for value in (hard, soft, entropy, kl, policy_entropy)), "Nonfinite target metric")
        result.append({"hard_loss": hard, "soft_loss": soft, "target_entropy": entropy,
                       "target_kl": kl, "policy_entropy": policy_entropy, "outsider_logits": values})
    return result


def initial_policy_kl(initial_logits, current_logits, *, audit=False):
    c.require(len(initial_logits) == len(current_logits), "Initial policy support changed")
    if len(initial_logits) < 2:
        return 0.
    if audit:
        def log_prob(values):
            peak = max(values)
            log_sum = math.log(math.fsum(math.exp(value-peak) for value in values))
            return [value-peak-log_sum for value in values]
        before, after = log_prob(initial_logits), log_prob(current_logits)
        return math.fsum(math.exp(left)*(left-right) for left, right in zip(before, after))
    before = torch.log_softmax(torch.tensor(initial_logits, dtype=torch.float64), dim=0)
    after = torch.log_softmax(torch.tensor(current_logits, dtype=torch.float64), dim=0)
    return float((before.exp()*(before-after)).sum())


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
        for key in ("soft_loss", "target_entropy", "target_kl", "policy_entropy"):
            values = [row[key] for row in selected]
            result[label]["mean_"+key] = (math.fsum(values) if audit else sum(values))/len(values) if values else None
        movement = [row["initial_policy_kl"] for row in selected if row["initial_policy_kl"] is not None]
        result[label]["mean_initial_policy_kl"] = (math.fsum(movement) if audit else sum(movement))/len(movement) if movement else None
        result[label]["initial_policy_kl_rows"] = len(movement)
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


def _cell(row):
    expected = [row["source_panel"], row["source_fit"], row["seat"], row["phase"]]
    c.require(row["cell"] == expected, "Added cell metadata differs")
    return tuple(expected)


def _inputs(fit_id, binding, base, rows, initializer, config, narrow):
    c.require(fit_id in FITS, "Unknown fit")
    fixture = any(value is not None for value in (rows, initializer, config, narrow))
    if fixture:
        c.require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture cannot target production")
        c.require(all(value is not None for value in (rows, initializer, config, narrow)), "Complete fixture inputs required")
        pools, initial, settings, selected = rows, initializer, dict(config), list(narrow)
    else:
        from . import training_source as source
        pools = {name: source.load_rows(binding, name, base) for name in ("old", "added")}
        initial, settings = source.initializer(fit_id, base), dict(CONFIG)
        selected = source.narrow_indices(fit_id, binding, base)
        c.require(len(pools["old"]) == 9461 and len(pools["added"]) == 720 and len(selected) == 180,
                  "Production training population changed")
    c.require(set(pools) == {"old", "added"}, "Unexpected training pool")
    for values in pools.values():
        _validate_rows(values)
    c.require(not {r["id"] for r in pools["old"]} & {r["id"] for r in pools["added"]}, "Old and added rows overlap")
    c.require(len(selected) == len(set(selected)) and all(type(i) is int and 0 <= i < len(pools["added"]) for i in selected),
              "Invalid narrow subset")
    c.require(len(selected)*4 == len(pools["added"]), "Narrow must contain one quarter of added rows")
    full_cells = Counter(_cell(row) for row in pools["added"])
    narrow_cells = Counter(_cell(pools["added"][i]) for i in selected)
    c.require(full_cells and full_cells.keys() == narrow_cells.keys()
              and all(full_cells[key] == 4*narrow_cells[key] for key in full_cells), "Narrow subset cell imbalance")
    games = [(row["source_panel"], row["source_game"]) for row in pools["added"]]
    c.require(len(games) == len(set(games)), "Added source game repeated")
    if not fixture:
        c.require(len(full_cells) == 36 and set(full_cells.values()) == {20}
                  and set(narrow_cells.values()) == {5}, "Production source-cell coverage changed")
        c.require({row["source_fit"] for row in pools["old"]} == set(FITS)
                  and {row["source_fit"] for row in pools["added"]} == set(FITS), "Training must pool all source fits")
    c.require(settings["target_temperature"] == .05
              and settings["objective"] == "recorded_outside_first_three_soft_score_cross_entropy"
              and settings["old_fraction"] == .5 and settings["updates"] > 0 and settings["batch_size"] > 1,
              "Changed fixed target or mixture")
    c.verify_hash(initial)
    return pools, initial, settings, sorted(selected), fixture


def _reference_lengths(rows, fit_id, config):
    """Retain predecessor batch lengths, not its all-old row exposure stream."""
    lengths, epoch = [], 0
    while len(lengths) < config["updates"]:
        order = list(range(len(rows)))
        generator = random.Random(f"{config['reference_batch_tag']}|{config['train_seeds'][str(fit_id)]}|{epoch}")
        generator.shuffle(order)
        order.sort(key=lambda i: len(rows[i]["features"]))
        size = config["batch_size"]
        batches = [order[j:j+size] for j in range(0, len(order), size)]
        generator.shuffle(batches)
        lengths.extend(map(len, batches))
        epoch += 1
    return lengths[:config["updates"]]


def _cycling_indices(indices, seed):
    cycle = 0
    while True:
        order = list(indices)
        random.Random(f"{seed}|{cycle}").shuffle(order)
        yield from order
        cycle += 1


def batch_plan(pools, narrow, fit_id, config=CONFIG):
    """Shared old IDs and cell slots; new row identity is the intervention."""
    tag = f"{config['batch_tag']}|{config['train_seeds'][str(fit_id)]}"
    lengths = _reference_lengths(pools["old"], fit_id, config)
    c.require(sum(lengths) % 2 == 0, "Balanced mixture needs even total exposure")
    old_stream = _cycling_indices(range(len(pools["old"])), tag+"|old")
    cells = sorted({_cell(row) for row in pools["added"]})
    cell_stream = _cycling_indices(cells, tag+"|cells")
    streams = {}
    for arm, chosen in (("narrow", narrow), ("broad", range(len(pools["added"])))):
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
    old_draws = [i for batch in plan["narrow"] for kind, i in batch if kind == "old"]
    paired_new = [(i, j) for left, right in zip(plan["narrow"], plan["broad"], strict=True)
                  for (kind, i), (_, j) in zip(left, right, strict=True) if kind == "added"]
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


def independent_plan(pools, narrow, fit_id, config=CONFIG):
    """Reconstruct lengths and all three sampling layers without collector helpers."""
    seed = config["train_seeds"][str(fit_id)]
    tag = "%s|%s" % (config["batch_tag"], seed)
    lengths = []
    epoch = 0
    while len(lengths) < config["updates"]:
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
    lengths = lengths[:config["updates"]]
    c.require(sum(lengths) % 2 == 0, "Balanced mixture needs even total exposure")
    cells = sorted({(row["source_panel"], row["source_fit"], row["seat"], row["phase"]) for row in pools["added"]})
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
        indices = list(narrow) if arm == "narrow" else list(range(len(pools["added"])))
        domains[arm] = {cell: [i for i in indices if tuple(pools["added"][i][key] for key in
                            ("source_panel", "source_fit", "seat", "phase")) == cell] for cell in cells}
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
        for position, (kind, row_index) in enumerate(out["narrow"][batch_index]):
            if kind == "old":
                old_draws.append(row_index)
            else:
                new_draws.append((row_index, out["broad"][batch_index][position][1]))
    old_draws = sorted(old_draws, key=lambda i: pools["old"][i]["features"].shape[0])
    new_draws = sorted(new_draws, key=lambda indices: max(pools["added"][indices[0]]["features"].shape[0],
                                                       pools["added"][indices[1]]["features"].shape[0]))
    placement = list(range(len(lengths)))
    random.Random("%s|length-batches" % tag).shuffle(placement)
    rebuilt = {arm: [None for _ in lengths] for arm in ARMS}
    old_position = added_position = 0
    for block in placement:
        selected_old = sum(kind == "old" for kind, _ in out["narrow"][block])
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
    left, right = (receipts[arm] for arm in ARMS)
    equal = ("batch_lengths", "updates", "row_exposures", "old_exposures", "added_exposures",
             "per_old_game_exposures", "per_added_cell_exposures", "added_cells",
             "old_batch_stream_sha256", "added_cell_batch_stream_sha256")
    c.require(all(left[key] == right[key] for key in equal), "Between-arm old or cell exposure mismatch")
    c.require(left["old_exposures"] == left["added_exposures"], "Mixture must be globally half old and half added")
    for narrow_batch, broad_batch in zip(plans["narrow"], plans["broad"], strict=True):
        c.require(len(narrow_batch) == len(broad_batch), "Paired batch lengths changed")
        for (kind, i), (other_kind, j) in zip(narrow_batch, broad_batch, strict=True):
            c.require(kind == other_kind and (i == j if kind == "old" else _cell(pools[kind][i]) == _cell(pools[kind][j])),
                      "Paired batch slots changed old row or new cell")
        count = sum(kind == "old" for kind, _ in narrow_batch)
        c.require(abs(2*count-len(narrow_batch)) <= 1, "A batch violates the half-old mixture")
    c.require(left["updates"] == config["updates"], "Wrong fixed update count")
    if not fixture:
        c.require(left["row_exposures"] == 114044 and left["old_exposures"] == 57022
                  and Counter(left["batch_lengths"]) == {64: 1772, 53: 12}, "Production dose changed")
        c.require(len(left["per_added_game_exposures"]) == 180
                  and len(right["per_added_game_exposures"]) == 720, "Some added games were not exposed")
    return {"arms": receipts, "same_batch_lengths": True, "same_old_rows_in_every_batch": True,
            "same_new_cell_in_every_batch_slot": True, "equal_total_old_and_added_exposures": True}


def _metadata(fit_id, arm, binding):
    return {"experiment": "independent_game_coverage_128", "fit": fit_id, "arm": arm,
            "manifest": binding, "checkpoint_selection": "fixed_final", "assessment_used": False}


def _diagnostics(model, pools, selected, arm, base, settings, initial, *, audit=False):
    train_indices = selected if arm == "narrow" else range(len(pools["added"]))
    ids = {pools["added"][i]["id"] for i in train_indices}
    all_ids = {row["id"] for row in pools["added"]}
    return {
        "old": training_report(model, pools["old"], base, settings,
            None if initial is None else initial["old"], audit=audit, diagnostic_scope="full_old_train"),
        "added": training_report(model, pools["added"], base, settings,
            None if initial is None else initial["added"], audit=audit,
            diagnostic_scope="full_added_train_pool_with_exposure_slices",
            slice_ids={"optimizer_exposed": ids, "not_optimizer_exposed": all_ids-ids}),
        "narrow_unexposed_rows_are_development_only": True,
        "all_added_rows_retired_from_future_tests": True,
        "assessment_read": False, "validation_read": False,
    }


def _fit_arm(fit_id, arm, binding, base, pools, initial_ref, settings, selected, plan, receipt):
    folder = Path(base)/"fits"/f"fit_{fit_id}"/arm
    folder.mkdir(parents=True, exist_ok=False)
    reporter = StatusReporter(folder, {"manifest": binding, "fit": fit_id, "arm": arm})
    seed = settings["train_seeds"][str(fit_id)]
    model, parameters, optimizer = _model(initial_ref, seed, settings)
    frozen = {key: value.clone() for key, value in model.state_dict().items() if key.startswith("value_head.")}
    checkpoints, initial, norms = [], None, []
    started = time.perf_counter()

    def save(update):
        nonlocal initial
        c.budget_boundary(base)
        c.require(all(torch.equal(model.state_dict()[key], value) for key, value in frozen.items()), "Value head changed")
        report = _diagnostics(model, pools, selected, arm, base, settings, initial)
        if initial is None:
            initial = report
        model_path = folder/f"checkpoint_{update:04d}.pt"
        metrics_path = folder/f"training_{update:04d}.json"
        save_model(model_path, model, update, _metadata(fit_id, arm, binding))
        old.save_new(metrics_path, report)
        checkpoints.append({"update": update, "checkpoint": c.artifact(model_path, source_fit=fit_id, arm=arm),
                            "training": c.artifact(metrics_path)})
    save(0)
    for update, indices in enumerate(plan, 1):
        c.budget_boundary(base)
        batch = [pools[kind][index] for kind, index in indices]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*collate(batch))
        loss = objective(logits, batch, settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite paired soft loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)))
        optimizer.step()
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite model")
        if update == 1 or update % 64 == 0 or update == settings["updates"]:
            reporter.event("coverage_fit", update=update, total_updates=settings["updates"], arm=arm)
    c.require(len(plan) == settings["updates"], "Wrong final update")
    save(settings["updates"])
    return {"arm": arm, "seed": seed, "initializer": initial_ref, "checkpoints": checkpoints,
        "old_rows": len(pools["old"]), "added_rows_exposed": len(selected) if arm == "narrow" else len(pools["added"]),
        **receipt, "selected_update": settings["updates"], "selected_model": checkpoints[-1]["checkpoint"],
        "selected_training": checkpoints[-1]["training"], "final_always_selected": True,
        "optimization_stability": _stability(norms, settings), "value_head_frozen": True,
        "assessment_read": False, "validation_read": False, "elapsed_seconds": time.perf_counter()-started}


def fit(fit_id, binding, base=OUTPUT, *, rows=None, initializer=None, config=None, narrow=None):
    c.budget_boundary(base)
    pools, initial, settings, selected, fixture = _inputs(fit_id, binding, base, rows, initializer, config, narrow)
    started = time.perf_counter()
    plans = batch_plan(pools, selected, fit_id, settings)
    paired = _paired_receipt(plans, pools, settings, fixture=fixture)
    arms = {arm: _fit_arm(fit_id, arm, binding, base, pools, initial, settings, selected, plans[arm], paired["arms"][arm])
            for arm in ARMS}
    result = {"manifest": binding, "fit": fit_id, "config": settings, "initializer": initial,
        "old_rows_sha256": _rows_identity(pools["old"]), "added_rows_sha256": _rows_identity(pools["added"]),
        "narrow_indices": selected, "narrow_row_ids": [pools["added"][i]["id"] for i in selected],
        "paired_schedule": paired, "arms": arms, "complete": True,
        "assessment_read": False, "validation_read": False, "elapsed_seconds": time.perf_counter()-started}
    path = Path(base)/"fits"/f"fit_{fit_id}"/"result.json"
    old.save_new(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def _audit_arm(fit_id, arm, binding, base, pools, initial_ref, settings, selected, plan, receipt, expected):
    seed = settings["train_seeds"][str(fit_id)]
    c.require(expected["arm"] == arm and expected["seed"] == seed and expected["initializer"] == initial_ref,
              "Wrong arm initializer or seed")
    c.require(expected["old_rows"] == len(pools["old"])
              and expected["added_rows_exposed"] == (len(selected) if arm == "narrow" else len(pools["added"])),
              "Wrong arm population")
    for key, value in receipt.items():
        _compare(value, expected[key])
    checkpoints = expected["checkpoints"]
    c.require([row["update"] for row in checkpoints] == [0, settings["updates"]], "Only initial/final checkpoints are permitted")
    model, parameters, optimizer = _model(initial_ref, seed, settings)
    frozen = {name: tensor.clone() for name, tensor in model.state_dict().items() if name.startswith("value_head.")}
    reporter = StatusReporter(Path(base)/"fits"/f"fit_{fit_id}"/arm/"audit_status",
                              {"manifest": binding, "fit": fit_id, "arm": arm})
    initial, norms = None, []

    def inspect(checkpoint):
        nonlocal initial
        c.budget_boundary(base)
        payload = torch.load(c.verify_hash(checkpoint["checkpoint"]), map_location="cpu", weights_only=True)
        c.require(payload["model_kind"] == MODEL_KIND and payload["model_config"] == asdict(model.config)
                  and payload["generation"] == checkpoint["update"]
                  and payload["metadata"] == _metadata(fit_id, arm, binding), "Wrong checkpoint identity")
        c.require(payload["model_state_dict"].keys() == model.state_dict().keys(), "Model tensor keys changed")
        for name, tensor in model.state_dict().items():
            c.require(bool(torch.isfinite(tensor).all()) and torch.equal(tensor, payload["model_state_dict"][name]),
                      "Independent optimizer reproduction differs: "+name)
            if name in frozen:
                c.require(torch.equal(tensor, frozen[name]), "Frozen value head moved")
        diagnostics = _diagnostics(model, pools, selected, arm, base, settings, initial, audit=True)
        _compare(diagnostics, c.verified(checkpoint["training"]))
        if initial is None:
            initial = diagnostics
    inspect(checkpoints[0])
    for index in range(len(plan)):
        c.budget_boundary(base)
        batch = []
        for name, item in plan[index]:
            batch.append(pools[name][item])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*audit_batch_tensors(batch))
        losses, _ = audit_objective(logits, batch, settings["target_temperature"])
        loss = losses.mean()
        c.require(bool(torch.isfinite(loss)), "Nonfinite independent paired loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm"], error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
        c.require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "Nonfinite independently fitted model")
        update = index+1
        if update == 1 or update % 64 == 0 or update == settings["updates"]:
            reporter.event("coverage_fit_reproduction", update=update, total_updates=settings["updates"], arm=arm)
    inspect(checkpoints[-1])
    c.require(expected["selected_update"] == settings["updates"]
              and expected["selected_model"] == checkpoints[-1]["checkpoint"]
              and expected["selected_training"] == checkpoints[-1]["training"]
              and expected["final_always_selected"] and expected["value_head_frozen"], "Changed fixed final selection")
    c.require(not expected["assessment_read"] and not expected["validation_read"], "Forbidden fitting information")
    stability = _stability(norms, settings, audit=True)
    _compare(stability, expected["optimization_stability"])
    return {"arm": arm, "complete_fit_bitwise_identical": True,
        "all_training_metrics_independently_reproduced": True,
        "soft_target_arithmetic_independently_reproduced": True,
        "optimization_stability_independently_reproduced": True, "optimization_stability": stability,
        "value_head_frozen": True, "selected_update": settings["updates"],
        "selected_model": expected["selected_model"], "selected_training": expected["selected_training"],
        "final_always_selected": True, **receipt}


def audit_fit(fit_id, fit_ref, binding, base=OUTPUT, *, rows=None, initializer=None, config=None, narrow=None):
    c.budget_boundary(base)
    started = time.perf_counter()
    pools, initial, settings, selected, fixture = _inputs(fit_id, binding, base, rows, initializer, config, narrow)
    expected = c.verified(fit_ref)
    c.require(expected["manifest"] == binding and expected["fit"] == fit_id and expected["config"] == settings
              and expected["initializer"] == initial and expected["complete"], "Wrong paired fit identity")
    c.require(expected["old_rows_sha256"] == _rows_identity(pools["old"])
              and expected["added_rows_sha256"] == _rows_identity(pools["added"]), "Wrong paired corpus identity")
    c.require(expected["narrow_indices"] == selected
              and expected["narrow_row_ids"] == [pools["added"][i]["id"] for i in selected], "Wrong narrow subset")
    plans = independent_plan(pools, selected, fit_id, settings)
    paired = _paired_receipt(plans, pools, settings, fixture=fixture)
    _compare(paired, expected["paired_schedule"])
    c.require(set(expected["arms"]) == set(ARMS), "Missing paired arm")
    arms = {arm: _audit_arm(fit_id, arm, binding, base, pools, initial, settings, selected,
                           plans[arm], paired["arms"][arm], expected["arms"][arm]) for arm in ARMS}
    c.require(not expected["assessment_read"] and not expected["validation_read"], "Forbidden paired fit information")
    result = {"manifest": binding, "fit": fit_id, "result": fit_ref, "passed": True,
        "arms": arms, "paired_schedule": paired, "complete_fit_bitwise_identical": True,
        "soft_target_arithmetic_independently_reproduced": True,
        "exact_between_arm_old_and_cell_streams": True,
        "assessment_read": False, "validation_read": False, "elapsed_seconds": time.perf_counter()-started}
    path = Path(base)/"fits"/f"fit_{fit_id}"/"audit.json"
    old.save_new(path, result)
    return c.artifact(path, fit=fit_id, index=FITS.index(fit_id))


def models_lock(fits, audits, binding, base=OUTPUT):
    from . import training_source as source
    c.require(len(fits) == len(audits) == len(FITS), "All three paired fits and audits are required")
    models = dict(source.control_models(binding, base))
    c.require(set(models) == {f"{arm}_{fit}" for arm in ("anchor", "lowrate", "soft") for fit in FITS}, "Wrong frozen controls")
    for fit_id, fit_ref, audit_ref in zip(FITS, fits, audits, strict=True):
        fitted, checked = c.verified(fit_ref), c.verified(audit_ref)
        c.require(fit_ref["fit"] == audit_ref["fit"] == fitted["fit"] == checked["fit"] == fit_id
                  and fit_ref["index"] == audit_ref["index"] == FITS.index(fit_id), "Fit registry order changed")
        c.require(fitted["manifest"] == checked["manifest"] == binding and fitted["complete"]
                  and checked["result"] == fit_ref and checked["passed"]
                  and checked["complete_fit_bitwise_identical"] and checked["soft_target_arithmetic_independently_reproduced"]
                  and checked["exact_between_arm_old_and_cell_streams"], "Unverified paired fit")
        _compare(fitted["paired_schedule"], checked["paired_schedule"])
        for arm in ARMS:
            trained, audited = fitted["arms"][arm], checked["arms"][arm]
            c.require(all(audited[key] for key in ("complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
                "soft_target_arithmetic_independently_reproduced", "optimization_stability_independently_reproduced", "value_head_frozen")),
                "Incomplete arm audit")
            c.require(trained["selected_update"] == audited["selected_update"] == fitted["config"]["updates"]
                      and trained["selected_model"] == audited["selected_model"]
                      and trained["selected_training"] == audited["selected_training"]
                      and trained["final_always_selected"] and audited["final_always_selected"], "Fixed final linkage changed")
            _compare(trained["optimization_stability"], audited["optimization_stability"])
            models[f"{arm}_{fit_id}"] = trained["selected_model"]
    for key, reference in models.items():
        c.require(reference["source_fit"] == int(key.rsplit("_", 1)[1]), "Model source assignment changed")
        c.verify_hash(reference)
    result = {"manifest": binding, "models": models, "fits": fits, "audits": audits,
              "complete": True, "all_new_fits_independently_reproduced": True,
              "selection": "all_three_fits_fixed_final_both_arms", "assessment_used_for_selection": False}
    path = Path(base)/"models_lock.json"
    old.save_new(path, result)
    return c.artifact(path)


def validate_models(binding, base=OUTPUT):
    from . import training_source as source
    registry = old.read_json(Path(base)/"models_lock.json")
    c.require(registry["manifest"] == binding and registry["complete"]
              and registry["all_new_fits_independently_reproduced"] and not registry["assessment_used_for_selection"]
              and registry["selection"] == "all_three_fits_fixed_final_both_arms", "Invalid model seal")
    expected = {f"{arm}_{fit}" for arm in ("anchor", "narrow", "broad", "lowrate", "soft") for fit in FITS}
    c.require(set(registry["models"]) == expected, "Model seal coverage changed")
    for key, ref in source.control_models(binding, base).items():
        c.require(registry["models"][key] == ref, "Frozen control model changed")
    for fit_id, fit_ref, audit_ref in zip(FITS, registry["fits"], registry["audits"], strict=True):
        fit, audit = c.verified(fit_ref), c.verified(audit_ref)
        c.require(fit["manifest"] == audit["manifest"] == binding and fit["fit"] == audit["fit"] == fit_id
                  and audit["result"] == fit_ref and audit["passed"] and audit["complete_fit_bitwise_identical"]
                  and audit["soft_target_arithmetic_independently_reproduced"] and audit["exact_between_arm_old_and_cell_streams"],
                  "Model fit/audit linkage changed")
        _compare(fit["paired_schedule"], audit["paired_schedule"])
        for arm in ARMS:
            reference = registry["models"][f"{arm}_{fit_id}"]
            c.require(reference == fit["arms"][arm]["selected_model"] == audit["arms"][arm]["selected_model"]
                      and fit["arms"][arm]["selected_update"] == audit["arms"][arm]["selected_update"] == fit["config"]["updates"],
                      "Model final linkage changed")
            payload = torch.load(c.verify_hash(reference), map_location="cpu", weights_only=True)
            c.require(payload["model_kind"] == MODEL_KIND and payload["generation"] == fit["config"]["updates"]
                      and payload["metadata"] == _metadata(fit_id, arm, binding), "Model checkpoint identity changed")
    for key, reference in registry["models"].items():
        c.require(reference["source_fit"] == int(key.rsplit("_", 1)[1]), "Model source fit changed")
        c.verify_hash(reference)
    return registry


