"""Fixed duration diagnostics on reused games; no playing-strength inference.

The parent and one-pass outputs are authenticated accepted evidence. Only the
prospectively committed two/four-pass checkpoints receive fresh predictions.
All statistics weight games equally, never treating positions as independent.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import sys

from tools.splendorust_full_corpus_20261110 import offline as previous
from tools.splendorust_full_corpus_20261110 import fit as previous_fit
from tools.splendorust_full_corpus_20261110 import sampling, corpus, targets
from tools.splendorust_policy_head_20261104 import training as accepted
from . import common as c

PASSES = (1, 2, 4)
FRESH_IDS = tuple(f"r{rep}_pass{passes}" for passes in (2, 4) for rep in (0, 1))
MODEL_IDS = ("parent",) + tuple(f"r{rep}_pass{passes}" for passes in PASSES for rep in (0, 1))
SPLITS = ("train", "validation", "offline")
HELDOUT = ("validation", "offline")
DRAW_COUNT = 20000
BOOTSTRAP_SEED = 2026111357022
GAP_FLOOR = 1e-8
METRICS = ("policy_ce", "teacher_entropy", "policy_kl", "outcome_mse", "search_value_mse",
           "mixed_value_mse", "policy_teacher_agreement", "policy_choice_changed", "value_drift")
OLD_IDS = {"parent": "parent", "r0_pass1": "r0_mixed_joint", "r1_pass1": "r1_mixed_joint"}
require = c.require


def _read(item, boundary=lambda: None):
    c.verify([item], boundary=boundary)
    return c.read(c.resolve(item))


def fixed_bindings(selection, original_commitment, duration_commitment, *, boundary=lambda: None):
    """Reject missing/endogenous checkpoint choices before model construction."""
    original = previous_fit.validate_commitment(selection, original_commitment, boundary=boundary)
    old = previous.prediction_bindings(original, boundary=boundary)
    committed = _read(duration_commitment, boundary)
    require(committed.get("status") == "all_duration_checkpoints_committed" and
            committed.get("checkpoint_selection") is False and
            committed.get("selection_sha256") == c.digest(selection) and
            committed.get("original_commitment") == original_commitment,
            "all fixed duration endpoints committed against original corpus")
    require([r["model_id"] for r in committed["models"]] == list(FRESH_IDS),
            "all and only two/four-pass paired checkpoints, in fixed order")
    out = {new: old[old_id] for new, old_id in OLD_IDS.items()}
    from . import fit
    for row in committed["models"]:
        name = row["model_id"]
        expected = dict(repetition=int(name[1]), passes=int(name[-1]))
        require(all(type(row[k]) is int and row[k] == v for k, v in expected.items()),
                "fixed duration model identity")
        manifest = _read(row["checkpoint"], boundary)
        fit.validate_checkpoint_metadata(manifest["metadata"])
        meta = manifest["metadata"]
        require(meta["checkpoint_step"] == row["passes"] * 1782 and
                meta["repetition"] == row["repetition"] and
                meta["qualification_only"] is False and
                meta["weights_sha256"] == row["weights_sha256"], "committed exact endpoint weights")
        c.verify([manifest["checkpoint"], manifest["tensor_manifest"]], boundary=boundary)
        out[name] = dict(checkpoint=row["checkpoint"], weights_sha256=row["weights_sha256"])
    return out


def predict_outputs(model, examples):
    """Actual input/forward route, including structural rule fixtures without outcomes."""
    import torch
    tokens = sampling.collate_inputs(examples, [1.] * len(examples))["tokens"]
    with torch.no_grad():
        outputs = accepted.tensor_forward(model, tokens).tolist()
    return outputs


def predict_batch(model, examples):
    """Actual evaluator route, also exercised by genuine-outcome qualification rows."""
    outputs = predict_outputs(model, examples)
    rows = [previous_fit.prediction_row(x, y) for x, y in zip(examples, outputs, strict=True)]
    for row in rows:
        previous.strict_prediction(row)
    return rows


def example_groups(selection, original_commitment, *, boundary=lambda: None):
    heldout = sampling.load_heldout(selection, original_commitment, boundary=boundary)
    training = sampling.load_diagnostic_train(selection, boundary=boundary)
    return {"train": training, **{split: [x for x in heldout if x["split"] == split] for split in HELDOUT}}


def run_evaluation(selection, original_commitment, original_evaluation, duration_commitment, folder,
                   *, boundary=lambda: None, runtime_sources=None):
    from . import fit
    folder = Path(folder)
    selection_sha = c.digest(selection)
    bindings = fixed_bindings(selection, original_commitment, duration_commitment, boundary=boundary)
    # Saved evidence is fully checked, including source rows, masks and exact float32 outputs.
    old_bundles, old_result = previous.load_predictions(selection, original_commitment,
                                                       original_evaluation, boundary=boundary)
    del old_bundles
    fit.engine(folder, runtime_sources=runtime_sources)
    groups = example_groups(selection, original_commitment, boundary=boundary)
    results = []
    for model_id in FRESH_IDS:
        boundary()
        binding = bindings[model_id]
        model, _, _ = fit.load_checkpoint(binding["checkpoint"], boundary=boundary)
        model.eval()
        require(accepted.state_digest(model) == binding["weights_sha256"], "actual fixed evaluation weights")
        batches = []
        for split, examples in groups.items():
            for index, offset in enumerate(range(0, len(examples), 32)):
                boundary()
                rows = predict_batch(model, examples[offset:offset + 32])
                batches.append(c.save(folder / "predictions" / model_id / split / f"{index:06d}.json",
                    dict(schema="duration_prediction_batch_v1", model_id=model_id, split=split,
                         batch_index=index, **binding, selection_sha256=selection_sha,
                         duration_commitment=duration_commitment, rows=rows)))
        require(accepted.state_digest(model) == binding["weights_sha256"], "evaluation leaves weights unchanged")
        results.append(dict(model_id=model_id, **binding, batches=batches,
                            root_counts={s: len(x) for s, x in groups.items()}, forward_calls=len(batches)))
        del model
    value = dict(schema="duration_evaluation_v1", status="all_fixed_predictions_complete",
                 selection_sha256=selection_sha, original_commitment=original_commitment,
                 original_evaluation=original_evaluation, duration_commitment=duration_commitment,
                 reused_model_ids=OLD_IDS, models=results, optimizer_calls=0,
                 forward_calls=sum(x["forward_calls"] for x in results),
                 original_forward_calls_reexecuted=0, all_endpoints_committed_before_prediction=True,
                 checkpoint_selection=False, heldout_games=128, heldout_reused_development_only=True,
                 reference_promoted=False, **sampling.IDENTITY)
    item = c.save(folder / "evaluation_result.json", value)
    boundary()
    return item


def load_predictions(selection, original_commitment, original_evaluation, duration_commitment, evaluation,
                     *, boundary=lambda: None):
    selection_sha = c.digest(selection)
    bindings = fixed_bindings(selection, original_commitment, duration_commitment, boundary=boundary)
    old, old_result = previous.load_predictions(selection, original_commitment, original_evaluation, boundary=boundary)
    bundles = {new: old[old_id] for new, old_id in OLD_IDS.items()}
    value = _read(evaluation, boundary)
    require(value["schema"] == "duration_evaluation_v1" and value["status"] == "all_fixed_predictions_complete" and
            value["selection_sha256"] == selection_sha and value["original_commitment"] == original_commitment and
            value["original_evaluation"] == original_evaluation and value["duration_commitment"] == duration_commitment and
            value["reused_model_ids"] == OLD_IDS and value["optimizer_calls"] == 0 and
            value["all_endpoints_committed_before_prediction"] is True and
            value["original_forward_calls_reexecuted"] == 0 and
            [x["model_id"] for x in value["models"]] == list(FRESH_IDS), "complete fixed evaluation identity")
    base = c.resolve(evaluation).parent
    paths = []
    for entry in value["models"]:
        model_id = entry["model_id"]
        require(bindings[model_id] == {k: entry[k] for k in ("checkpoint", "weights_sha256")}, "fixed saved predictor")
        groups = {s: [] for s in SPLITS}
        for artifact in entry["batches"]:
            boundary()
            batch = _read(artifact, boundary)
            split = batch["split"]
            require(split in SPLITS, "known diagnostic split")
            index = len(groups[split]) // 32
            require(batch["schema"] == "duration_prediction_batch_v1" and batch["model_id"] == model_id and
                    type(batch["batch_index"]) is int and batch["batch_index"] == index and
                    batch["duration_commitment"] == duration_commitment and
                    batch["selection_sha256"] == selection_sha and
                    all(batch[k] == entry[k] for k in ("checkpoint", "weights_sha256")) and
                    c.resolve(artifact) == base / "predictions" / model_id / split / f"{index:06d}.json",
                    "canonical ordered saved batch with exact predictor binding")
            expected = bundles["parent"][split][index * 32:index * 32 + 32]
            require(len(batch["rows"]) == len(expected) > 0, "complete fixed batch")
            for row, parent in zip(batch["rows"], expected, strict=True):
                previous.strict_prediction(row)
                require(previous.target_identity(row) == previous.target_identity(parent), "same authentic input and targets")
            groups[split].extend(batch["rows"])
            paths.append(c.resolve(artifact))
        require(entry["forward_calls"] == len(entry["batches"]), "exact fresh forward inventory")
        for split in SPLITS:
            require(len(groups[split]) == len(bundles["parent"][split]) == entry["root_counts"][split],
                    "all diagnostic roots predicted")
        bundles[model_id] = groups
    require(len(paths) == len(set(paths)) == value["forward_calls"] and
            set(base.glob("predictions/*/*/*.json")) == set(paths), "all and only fixed prediction batches")
    return bundles, value


def source_labels(selection, *, boundary=lambda: None):
    cache = corpus.load_cache(selection, boundary=boundary)
    labels = {}
    for row in cache["index"]:
        key = previous.identity(row)
        require(key not in labels, "unique source target inventory")
        labels[key] = row
    return labels


def row_metrics(row, parent, label):
    losses = previous.strict_prediction(row)
    require(previous.target_identity(row) == previous.target_identity(parent), "paired exact policy/value target")
    require(previous.identity(row) == previous.identity(label) and
            row["row_sha256"] == label["row_sha256"] and
            row["root_input_sha256"] == label["root_input_sha256"] and
            row["value_target2"] == label["terminal_value"], "authenticated actor-oriented source labels")
    q = label["root_q_signed"]
    require(type(q) in (int, float) and math.isfinite(q) and -1 <= q <= 1 and
            q == 2. * label["root_q_win_credit"] - 1., "exact signed source search value")
    entropy = -math.fsum(t * math.log(t) for t in row["policy_target81"] if t)
    kl = losses["policy"] - entropy
    require(kl >= -1e-12, "nonnegative teacher KL up to float64 rounding")
    legal = [i for i, ok in enumerate(row["legal_mask81"]) if ok]
    choose = lambda output: max(legal, key=lambda i: (output[i], -i))
    chosen, old = choose(row["output83"]), choose(parent["output83"])
    teacher = max(legal, key=lambda i: (row["policy_target81"][i], -i))
    mixed = targets.mixture(row["value_target2"], label["root_q_win_credit"])
    return dict(policy_ce=losses["policy"], teacher_entropy=entropy, policy_kl=max(0., kl),
                outcome_mse=losses["value"],
                search_value_mse=math.fsum((a - b) ** 2 for a, b in zip(row["output83"][81:], [q, -q])) / 2.,
                mixed_value_mse=math.fsum((a - targets.f32(b)) ** 2 for a, b in zip(row["output83"][81:], mixed)) / 2.,
                policy_teacher_agreement=float(chosen == teacher), policy_choice_changed=float(chosen != old),
                value_drift=math.fsum(abs(a - b) for a, b in zip(row["output83"][81:], parent["output83"][81:])) / 2.)


def game_metrics(rows, parent, labels):
    grouped = defaultdict(list)
    require(len(rows) == len(parent) > 0, "complete matched diagnostic rows")
    require(len({previous.identity(x) for x in rows}) == len(rows), "no duplicate diagnostic roots")
    for row, original in zip(rows, parent, strict=True):
        grouped[row["game_index"]].append(row_metrics(row, original, labels[previous.identity(row)]))
    return [dict(game_index=g, roots=len(values),
                 **{m: math.fsum(x[m] for x in values) / len(values) for m in METRICS})
            for g, values in sorted(grouped.items())]


def gap_closure(parent_ce, candidate_ce, entropy):
    """Ratio of game-weighted means, not mean of unstable per-position ratios."""
    denominator = parent_ce - entropy
    require(denominator >= -1e-12, "nonnegative baseline policy gap")
    return None if denominator <= GAP_FLOOR else (parent_ce - candidate_ce) / denominator


def _definitions():
    definitions = {"parent": {"parent": 1.}}
    for passes in PASSES:
        for rep in (0, 1):
            name = f"r{rep}_pass{passes}"
            definitions[name] = {name: 1.}
        definitions[f"mean_pass{passes}"] = {f"r{rep}_pass{passes}": .5 for rep in (0, 1)}
    return definitions


def _contrast_definitions():
    result = []
    for label, before, after in (("one_to_two", 1, 2), ("two_to_four", 2, 4), ("one_to_four", 1, 4)):
        for prefix in ("r0", "r1", "mean"):
            result.append((label, prefix, f"{prefix}_pass{before}", f"{prefix}_pass{after}"))
    for passes in PASSES:
        for prefix in ("r0", "r1", "mean"):
            result.append((f"parent_to_pass{passes}", prefix, "parent", f"{prefix}_pass{passes}"))
    return result


def analyze_predictions(bundles, labels, *, draws=DRAW_COUNT, seed=BOOTSTRAP_SEED,
                        strict=True, boundary=lambda: None):
    import numpy as np
    require(type(draws) is int and draws > 0 and type(seed) is int, "fixed valid bootstrap counts")
    if strict:
        require(draws == DRAW_COUNT and seed == BOOTSTRAP_SEED, "prospective bootstrap settings")
    require(set(bundles) == set(MODEL_IDS), "all fixed models without selective omission")
    summaries, matrices = {}, {}
    for model in MODEL_IDS:
        require(set(bundles[model]) == set(SPLITS), "all diagnostic splits")
        summaries[model] = {}
        for split in SPLITS:
            boundary()
            rows = bundles[model][split]
            require(all(x["split"] == split for x in rows), "split identity")
            summaries[model][split] = game_metrics(rows, bundles["parent"][split], labels)
            expected = summaries["parent"][split]
            require([(x["game_index"], x["roots"]) for x in summaries[model][split]] ==
                    [(x["game_index"], x["roots"]) for x in expected], "paired whole-game inventory")
    if strict:
        require([len(bundles["parent"][s]) for s in SPLITS] == [3072, 3533, 3553] and
                [len(summaries["parent"][s]) for s in SPLITS] == [975, 64, 64], "exact accepted diagnostic panel")
    sets = [set(x["game_index"] for x in summaries["parent"][s]) for s in SPLITS]
    require(all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "disjoint game partitions")
    for split in SPLITS:
        matrices[split] = np.asarray([[[x[m] for m in METRICS] for x in summaries[name][split]]
                                      for name in MODEL_IDS], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled, index_receipts = {}, {}
    for split in HELDOUT:
        n = len(summaries["parent"][split])
        require(n > 1, "nondegenerate held-out game sample")
        indices = rng.integers(0, n, size=(draws, n), dtype=np.int64)
        index_receipts[split] = dict(games=n, game_ids=[x["game_index"] for x in summaries["parent"][split]],
                                    index_sha256=hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest())
        # Bounded chunks avoid a models x draws x games x metrics temporary.
        sampled[split] = np.empty((len(MODEL_IDS), draws, len(METRICS)), dtype=np.float64)
        for start in range(0, draws, 128):
            boundary()
            sampled[split][:, start:start + 128] = matrices[split][:, indices[start:start + 128], :].mean(axis=2)
    matrices["pooled"] = np.concatenate([matrices[s] for s in HELDOUT], axis=1)
    total = matrices["pooled"].shape[1]
    sampled["pooled"] = sum(sampled[s] * matrices[s].shape[1] for s in HELDOUT) / total
    definitions = _definitions()
    estimates = {}
    distributions = {}
    for name, coefficients in definitions.items():
        estimates[name] = {}
        distributions[name] = {}
        for split in (*SPLITS, "pooled"):
            point = sum(weight * matrices[split][MODEL_IDS.index(model)].mean(axis=0)
                        for model, weight in coefficients.items())
            dist = None if split == "train" else sum(weight * sampled[split][MODEL_IDS.index(model)]
                        for model, weight in coefficients.items())
            distributions[name][split] = dist
            record = {metric: dict(mean=float(point[j]), ci95=None if dist is None else
                       [float(x) for x in np.quantile(dist[:, j], [.025, .975], method="linear")])
                      for j, metric in enumerate(METRICS)}
            parent_mean = matrices[split][0].mean(axis=0)
            closure = gap_closure(float(parent_mean[0]), float(point[0]), float(parent_mean[1]))
            interval, valid = None, 0
            if dist is not None:
                denominator = sampled[split][0, :, 0] - sampled[split][0, :, 1]
                valid = int(np.sum(denominator > GAP_FLOOR))
                # Do not silently condition intervals on only favorable valid draws.
                if valid == draws:
                    ratios = (sampled[split][0, :, 0] - dist[:, 0]) / denominator
                    interval = [float(x) for x in np.quantile(ratios, [.025, .975], method="linear")]
            record["teacher_gap_closed"] = dict(mean=closure, ci95=interval,
                    baseline_gap=float(parent_mean[0] - parent_mean[1]), denominator_floor=GAP_FLOOR,
                    valid_draws=valid if dist is not None else None, ratio_of_equal_game_means=True)
            estimates[name][split] = record
    contrasts = []
    for label, prefix, left, right in _contrast_definitions():
        for split in (*SPLITS, "pooled"):
            for j, metric in enumerate(METRICS):
                diff = estimates[right][split][metric]["mean"] - estimates[left][split][metric]["mean"]
                dist = None if split == "train" else distributions[right][split][:, j] - distributions[left][split][:, j]
                contrasts.append(dict(contrast=label, seed=prefix, split=split, metric=metric,
                    difference=diff, ci95=None if dist is None else [float(x) for x in np.quantile(dist, [.025, .975], method="linear")],
                    descriptive=True, lower_is_better=metric.endswith("mse") or metric in ("policy_ce", "policy_kl")))
    return dict(schema="duration_analysis_v1", status="fixed_duration_diagnostics_analyzed",
        game_summaries=summaries, estimates=estimates, contrasts=contrasts,
        bootstrap=dict(seed=seed, draws=draws, generator="numpy.PCG64", unit="whole_game", strata=index_receipts,
                       paired_same_draws_all_models=True, pooled="partition_size_weighted_same_draws",
                       training_panel="descriptive_no_bootstrap", intervals="nominal95_descriptive_no_selection"),
        checkpoint_selection=False, strength_improvement_established=False, reference_promoted=False,
        limitations=["Both held-out partitions have been inspected repeatedly and are reused development diagnostics.",
          "Two fixed seeds share the same games; these intervals do not estimate arbitrary training-seed variation.",
          "Teacher-gap closure measures agreement with this saved teacher, not optimal play or all learnable information.",
          "Train/held-out divergence suggests diminishing generalization, not proof that game quantity is the sole bottleneck.",
          "Better prediction losses do not establish stronger searched play; no live test or new collection launches here."])


def verify_arithmetic(bundles, labels, reported, *, strict=True, boundary=lambda: None):
    """Independent array loss, game reduction, paired draw and ratio calculation."""
    import numpy as np
    require(reported["schema"] == "duration_analysis_v1" and
            reported["status"] == "fixed_duration_diagnostics_analyzed" and
            reported["checkpoint_selection"] is False and reported["strength_improvement_established"] is False and
            reported["reference_promoted"] is False, "fixed descriptive diagnostic status")
    if strict:
        require(reported["bootstrap"]["draws"] == DRAW_COUNT and reported["bootstrap"]["seed"] == BOOTSTRAP_SEED,
                "production analysis uses fixed draws and seed")
    require(reported["bootstrap"]["generator"] == "numpy.PCG64" and
            reported["bootstrap"]["unit"] == "whole_game" and
            reported["bootstrap"]["paired_same_draws_all_models"] is True,
            "fixed paired whole-game uncertainty recipe")
    matrices = {}
    for split in SPLITS:
        rows_by_model = []
        for model in MODEL_IDS:
            per_game = defaultdict(list)
            for row, parent in zip(bundles[model][split], bundles["parent"][split], strict=True):
                x = np.asarray(row["output83"], dtype=np.float64)
                p = np.asarray(row["policy_target81"], dtype=np.float64)
                legal = np.asarray(row["legal_mask81"], dtype=bool)
                z = np.asarray(row["value_target2"], dtype=np.float64)
                old = np.asarray(parent["output83"], dtype=np.float64)
                source = labels[previous.identity(row)]
                q = 2. * source["root_q_win_credit"] - 1.
                mixed = np.asarray([(z[0] + .837 * q) / 1.837, -(z[0] + .837 * q) / 1.837], dtype=np.float32).astype(np.float64)
                norm = np.logaddexp.reduce(x[:81][legal])
                ce = float(np.dot(p, norm - x[:81]))
                nonzero = p > 0
                entropy = float(-np.dot(p[nonzero], np.log(p[nonzero])))
                indices = np.flatnonzero(legal)
                choice = indices[np.argmax(x[:81][legal])]
                parent_choice = indices[np.argmax(old[:81][legal])]
                teacher = indices[np.argmax(p[legal])]
                per_game[row["game_index"]].append([ce, entropy, max(0., ce - entropy),
                    float(np.mean((x[81:] - z) ** 2)), float(np.mean((x[81:] - np.array([q, -q])) ** 2)),
                    float(np.mean((x[81:] - mixed) ** 2)), float(choice == teacher), float(choice != parent_choice),
                    float(np.mean(np.abs(x[81:] - old[81:])))])
            actual = np.asarray([np.asarray(per_game[g]).mean(axis=0) for g in sorted(per_game)])
            saved = reported["game_summaries"][model][split]
            require([r["game_index"] for r in saved] == sorted(per_game) and
                    [r["roots"] for r in saved] == [len(per_game[g]) for g in sorted(per_game)] and
                    np.allclose(actual, [[r[m] for m in METRICS] for r in saved], rtol=1e-11, atol=1e-12),
                    "independent row loss and equal-game reductions")
            rows_by_model.append(actual)
            boundary()
        matrices[split] = np.stack(rows_by_model)
    counts = reported["bootstrap"]
    draws, seed = counts["draws"], counts["seed"]
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = {}
    for split in HELDOUT:
        n = matrices[split].shape[1]
        indices = np.empty((draws, n), dtype=np.int64)
        for draw in range(draws):
            if draw % 500 == 0:
                boundary()
            indices[draw] = rng.integers(0, n, size=n, dtype=np.int64)
        require(hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest() ==
                counts["strata"][split]["index_sha256"], "independent paired draw stream")
        # Independent count-weight reduction instead of indexed-position means.
        sampled[split] = np.empty((len(MODEL_IDS), draws, len(METRICS)))
        for start in range(0, draws, 128):
            weights = np.asarray([np.bincount(ids, minlength=n) / n for ids in indices[start:start + 128]])
            sampled[split][:, start:start + 128, :] = np.einsum('dg,mgk->mdk', weights, matrices[split])
            boundary()
    matrices["pooled"] = np.concatenate([matrices[s] for s in HELDOUT], axis=1)
    sampled["pooled"] = sum(sampled[s] * matrices[s].shape[1] for s in HELDOUT) / matrices["pooled"].shape[1]
    definitions = _definitions()
    require(set(reported["estimates"]) == set(definitions), "complete fixed model/seed mean inventory")
    combined = {}
    for name, coefficients in definitions.items():
        combined[name] = {}
        for split in (*SPLITS, "pooled"):
            vector = sum(w * matrices[split][MODEL_IDS.index(m)].mean(axis=0) for m, w in coefficients.items())
            dist = None if split == "train" else sum(w * sampled[split][MODEL_IDS.index(m)] for m, w in coefficients.items())
            combined[name][split] = dist
            for j, metric in enumerate(METRICS):
                entry = reported["estimates"][name][split][metric]
                require(math.isclose(entry["mean"], vector[j], rel_tol=1e-10, abs_tol=1e-12), "independent metric mean")
                expected = None if dist is None else np.percentile(dist[:, j], [2.5, 97.5], method="linear")
                require(entry["ci95"] is None if expected is None else
                        np.allclose(entry["ci95"], expected, rtol=1e-10, atol=1e-12), "independent metric interval")
            original = matrices[split][0].mean(axis=0)
            denom = float(original[0] - original[1])
            expected_ratio = None if denom <= GAP_FLOOR else float((original[0] - vector[0]) / denom)
            entry = reported["estimates"][name][split]["teacher_gap_closed"]
            require(math.isclose(entry['baseline_gap'],denom,rel_tol=1e-10,abs_tol=1e-12) and
                    entry['denominator_floor']==GAP_FLOOR and entry['ratio_of_equal_game_means'] is True,
                    'Independent teacher-gap denominator metadata')
            require(entry["mean"] is None if expected_ratio is None else
                    math.isclose(entry["mean"], expected_ratio, rel_tol=1e-10, abs_tol=1e-12), "independent ratio of game means")
            if dist is not None:
                denominators = sampled[split][0, :, 0] - sampled[split][0, :, 1]
                valid = int(np.count_nonzero(denominators > GAP_FLOOR))
                require(entry["valid_draws"] == valid, "independent small-gap guard")
                expected = None if valid != draws else np.percentile((sampled[split][0, :, 0] - dist[:, 0]) / denominators,
                                                                      [2.5, 97.5], method="linear")
                require(entry["ci95"] is None if expected is None else
                        np.allclose(entry["ci95"], expected, rtol=1e-10, atol=1e-12), "independent gap-closure interval")
    expected_keys = {(label, rep, split, metric) for label in ("one_to_two", "two_to_four", "one_to_four",
                     "parent_to_pass1", "parent_to_pass2", "parent_to_pass4")
                     for rep in ("r0", "r1", "mean") for split in (*SPLITS, "pooled") for metric in METRICS}
    require(len(reported["contrasts"]) == len(expected_keys) and
            {(r["contrast"], r["seed"], r["split"], r["metric"]) for r in reported["contrasts"]} == expected_keys,
            "complete predeclared duration contrasts")
    for row in reported["contrasts"]:
        if row["contrast"].startswith("parent_to_pass"):
            a, b = "parent", f'{row["seed"]}_pass{int(row["contrast"][-1])}'
        else:
            first, last = {"one_to_two": (1, 2), "two_to_four": (2, 4), "one_to_four": (1, 4)}[row["contrast"]]
            a, b = f'{row["seed"]}_pass{first}', f'{row["seed"]}_pass{last}'
        split, metric = row["split"], row["metric"]
        expected = reported["estimates"][b][split][metric]["mean"] - reported["estimates"][a][split][metric]["mean"]
        require(math.isclose(expected, row["difference"], rel_tol=1e-10, abs_tol=1e-12), "independent paired duration difference")
        interval = None if split == "train" else np.percentile((combined[b][split] - combined[a][split])[:, METRICS.index(metric)],
                                                               [2.5, 97.5], method="linear")
        require(row["ci95"] is None if interval is None else np.allclose(row["ci95"], interval, rtol=1e-10, atol=1e-12),
                "independent paired duration interval")
    return dict(status="independent_duration_arithmetic_verified", model_calls=0, optimizer_calls=0,
                independent_loss_formula=True, independent_count_weight_bootstrap=True,
                same_input_teacher_entropy=True, gap_ratio_of_game_means=True,
                draws=draws, seed=seed, contrasts=len(expected_keys), fixed_models=len(MODEL_IDS))


def render_charts(result, output, *, boundary=lambda: None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    folder = Path(output)
    colors = {"train": "#0f766e", "pooled": "#c2410c"}
    artifacts = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, metric, title in zip(axes, ("policy_ce", "outcome_mse"),
                                ("Policy: error predicting teacher choices", "Value: error predicting game outcomes")):
        for split, label in (("train", "Training panel"), ("pooled", "Reused held-out games")):
            baseline = result["estimates"]["parent"][split][metric]["mean"]
            for rep in (0, 1):
                values = [result["estimates"][f"r{rep}_pass{p}"][split][metric]["mean"] - baseline for p in PASSES]
                ax.plot(PASSES, values, color=colors[split], alpha=.28, lw=1)
            values = [result["estimates"][f"mean_pass{p}"][split][metric]["mean"] - baseline for p in PASSES]
            ax.plot(PASSES, values, marker="o", color=colors[split], lw=2.5, label=label)
            if split == "pooled":
                changes = [next(row for row in result["contrasts"] if row["contrast"] == f"parent_to_pass{p}"
                                and row["seed"] == "mean" and row["split"] == split and row["metric"] == metric)
                           for p in PASSES]
                ax.fill_between(PASSES, [row["ci95"][0] for row in changes], [row["ci95"][1] for row in changes],
                                color=colors[split], alpha=.12)
        ax.axhline(0, color="#64748b", ls="--", lw=1)
        ax.set_xticks(PASSES)
        ax.set_xlabel("Full passes through the same 57,022 training positions")
        ax.set_ylabel("Change from original model (lower is better)")
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Does additional training generalize beyond its training games?", fontsize=14)
    fig.text(.5, .025, "Bold: fixed mean of two seeds. Thin: individual seeds. Training panel: 3,072 positions / 975 games.\n"
             "Held-out: 128 reused games; shaded nominal paired 95% intervals. Prediction error is not playing strength.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .93))
    path = folder / "duration_learning_curves.png"
    require(path.resolve().is_relative_to(c.OUT) and not path.exists(), "new exclusive chart")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(c.artifact(path))
    boundary()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for split, label in (("train", "Training panel"), ("pooled", "Reused held-out games")):
        for rep in (0, 1):
            values = [result["estimates"][f"r{rep}_pass{p}"][split]["teacher_gap_closed"]["mean"] for p in PASSES]
            if all(v is not None for v in values):
                ax.plot(PASSES, [100 * v for v in values], color=colors[split], alpha=.25, lw=1)
        records = [result["estimates"][f"mean_pass{p}"][split]["teacher_gap_closed"] for p in PASSES]
        if all(r["mean"] is not None for r in records):
            ax.plot(PASSES, [100 * r["mean"] for r in records], marker="o", color=colors[split], lw=2.5, label=label)
            if split == "pooled" and all(r["ci95"] is not None for r in records):
                ax.fill_between(PASSES, [100 * r["ci95"][0] for r in records],
                                [100 * r["ci95"][1] for r in records], color=colors[split], alpha=.14,
                                label="Held-out paired 95% interval")
    ax.axhline(0, color="#64748b", ls="--", lw=1)
    ax.set_xticks(PASSES)
    ax.set_xlabel("Full training passes")
    ax.set_ylabel("Initial policy gap to saved teacher closed (%)")
    ax.set_title("How much closer do the policy predictions get to the teacher?")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.text(.5, .025, "Gap = policy cross-entropy minus the teacher distribution's entropy. Ratio of equal-game means.\n"
             "Two fixed seeds; nominal whole-game intervals. Teacher agreement does not establish stronger play.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    path = folder / "duration_teacher_gap.png"
    require(path.resolve().is_relative_to(c.OUT) and not path.exists(), "new exclusive chart")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(c.artifact(path))
    boundary()
    return artifacts


def run_analysis(selection, original_commitment, original_evaluation, duration_commitment, evaluation, output,
                 *, boundary=lambda: None):
    require(not any(x == "torch" or x.startswith("torch.") for x in sys.modules), "fresh model-free analysis worker")
    bundles, source = load_predictions(selection, original_commitment, original_evaluation, duration_commitment,
                                       evaluation, boundary=boundary)
    labels = source_labels(selection, boundary=boundary)
    result = analyze_predictions(bundles, labels, boundary=boundary)
    check = verify_arithmetic(bundles, labels, result, boundary=boundary)
    check_item = c.save(Path(output) / "arithmetic_verification.json", check)
    result.update(selection_sha256=c.digest(selection), original_commitment=original_commitment,
                  original_evaluation=original_evaluation, duration_commitment=duration_commitment,
                  evaluation=evaluation, verified_forward_calls=source["forward_calls"],
                  analysis_model_calls=0, analysis_optimizer_calls=0, independent_arithmetic=check_item)
    item = c.save(Path(output) / "analysis.json", result)
    charts = render_charts(result, output, boundary=boundary)
    require(not any(x == "torch" or x.startswith("torch.") for x in sys.modules), "analysis remained model-free")
    return dict(status="duration_saved_analysis_verified", analysis=item, arithmetic=check_item, charts=charts,
                checkpoint_selection=False, models=0, optimizer_calls=0)


def run_saved_smoke(folder, *, boundary=lambda: None, runtime_sources):
    """Actual-source preflight: parse and score reused evidence, without models."""
    require(not any(x == "torch" or x.startswith("torch.") for x in sys.modules), "model-free saved smoke")
    selection = c.read(c.SELECTION)
    original_commitment = c.artifact(c.ORIGINAL_COMMITMENT)
    original_evaluation = c.artifact(c.ORIGINAL_EVALUATION)
    bundles, source = previous.load_predictions(selection, original_commitment, original_evaluation, boundary=boundary)
    labels = source_labels(selection, boundary=boundary)
    rows = []
    for model, old_id in OLD_IDS.items():
        for split in SPLITS:
            boundary()
            metrics = game_metrics(bundles[old_id][split], bundles["parent"][split], labels)
            require(len(metrics) == (975 if split == "train" else 64), "actual accepted game-panel size")
            rows.append(dict(model_id=model, split=split, roots=len(bundles[old_id][split]), games=len(metrics),
                             saved_metric_sha256=c.digest(metrics)))
    require(sum(row["roots"] for row in rows) == 3 * (3072 + 3533 + 3553), "all reused roots exercised")
    result = dict(status="saved_metric_route_qualified", runtime_sources=runtime_sources,
                  inputs=[c.artifact(c.SELECTION), original_commitment, original_evaluation,
                          c.artifact(c.CACHE_ACCEPTANCE)], panels=rows, model_calls=0, optimizer_calls=0,
                  all_actual_saved_target_types_accepted=True, source_labels=len(labels), **sampling.IDENTITY)
    boundary()
    item = c.save(Path(folder) / "metrics_saved_smoke01.json", result)
    require(not any(x == "torch" or x.startswith("torch.") for x in sys.modules), "saved smoke remained model-free")
    return item
