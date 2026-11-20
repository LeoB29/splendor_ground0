"""Fixed data-size diagnostics on reused games; no playing-strength inference.

Original parent and1024-game one-pass predictions remain authenticated reuse.
Eight new endpoints cover256/512games, two seeds, one-pass/matched updates.
All statistics pair and equally weight games; no checkpoint selection.
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

SIZES = (256, 512, 1024)
GROUPS = ((256, 'onepass'), (256, 'matched'), (512, 'onepass'), (512, 'matched'), (1024, 'onepass'))
FRESH_IDS = tuple(f'g{size}_r{rep}_{endpoint}' for size, endpoint in GROUPS if size != 1024 for rep in (0, 1))
MODEL_IDS = ('parent',) + tuple(f'g{size}_r{rep}_{endpoint}' for size, endpoint in GROUPS for rep in (0, 1))
SPLITS = ('train', 'validation', 'offline')
HELDOUT = ('validation', 'offline')
DRAW_COUNT = 20000
BOOTSTRAP_SEED = 20261113256512
GAP_FLOOR = 1e-8
METRICS = ('policy_ce', 'teacher_entropy', 'policy_kl', 'outcome_mse', 'search_value_mse',
           'mixed_value_mse', 'policy_teacher_agreement', 'policy_choice_changed', 'value_drift')
OLD_IDS = {'parent': 'parent', 'g1024_r0_onepass': 'r0_mixed_joint', 'g1024_r1_onepass': 'r1_mixed_joint'}
require = c.require

def _read(item, boundary=lambda: None):
    c.verify([item], boundary=boundary)
    return c.read(c.resolve(item))


def source_selection():
    return c.read(c.SOURCE_SELECTION)


def _expected_fit_bindings(selection, *, boundary=lambda: None):
    """Derive subset and target identities through the real read-only loader."""
    from . import fit, selection as subset_selection
    require(c.digest(c.read(c.SELECTION)) == c.digest(selection), 'Exact on-disk prospective selection')
    expected = {}
    for size in (256, 512):
        loaded = subset_selection.load_selected(selection, 0, size, boundary=boundary)
        for rep in (0, 1):
            schedule = subset_selection.build_schedules(selection, rep, size)
            expected[size, rep] = (fit.metadata(size, rep, loaded, schedule), schedule)
        del loaded
    return expected


def _committed_fit_endpoints(committed, *, boundary=lambda: None):
    """Bind commitment rows to verified fits and their exact original result files."""
    expected_order = [(size, rep) for size in (256, 512) for rep in (0, 1)]
    require([(row['game_count'], row['repetition']) for row in committed['fits']] == expected_order,
            'All four verified fits in prospective order')
    endpoints = {}
    for row in committed['fits']:
        size, rep = row['game_count'], row['repetition']
        role = f'g{size}_r{rep}'
        proof = _read(row['verification'], boundary)
        require(proof['status'] == 'data_size_fit_saved_verified' and proof['game_count'] == size and
                proof['repetition'] == rep and proof['optimizer_calls'] == 1782 and proof['replay_optimizer_calls'] == 2,
                'Exact verified fit proof')
        result = _read(proof['result'], boundary)
        require(proof['result'] == c.artifact(c.OUT/'jobs'/role/'worker/fit_result.json') and
                result['status'] == 'fit_complete' and result['game_count'] == size and result['repetition'] == rep and
                proof['endpoints'] == result['endpoints'] and proof['checkpoints'] == result['checkpoints'] and
                proof['endpoint_steps'] == result['endpoint_steps'], 'Verified endpoints bind exact actual fit result')
        execution = _read(row['execution'], boundary)
        require(row['execution'] == c.artifact(c.OUT/'jobs'/role/'actual_exit.json') and
                execution['role'] == role and execution['status'] == 'passed' and type(execution['exit_code']) is int and
                execution['exit_code'] == 0 and execution['actual_exit_observed'] is True and
                execution['exact_owned_absence'] is True and not execution['cleanup_errors'] and execution['error'] is None,
                'Committed fit has a clean actual exit')
        for endpoint, item in proof['endpoints'].items():
            endpoints[f'g{size}_r{rep}_{endpoint}'] = item
    require(set(endpoints) == set(FRESH_IDS), 'All and only eight verified assessment endpoints')
    return endpoints


def fixed_bindings(selection, original_commitment, study_commitment, *, boundary=lambda: None):
    """All eight prospective endpoints committed before any new predictions."""
    from . import fit
    original_selection = source_selection()
    original = previous_fit.validate_commitment(original_selection, original_commitment, boundary=boundary)
    old = previous.prediction_bindings(original, boundary=boundary)
    committed = _read(study_commitment, boundary)
    require(committed.get('status') == 'all_data_size_checkpoints_committed' and
        committed.get('checkpoint_selection') is False and committed.get('selection_sha256') == c.digest(selection) and
        committed.get('source_selection') == c.artifact(c.SOURCE_SELECTION) and
        committed.get('original_commitment') == original_commitment, 'All fixed data-size endpoints against exact source')
    require([r['model_id'] for r in committed['models']] == list(FRESH_IDS), 'Exact eight new endpoints in prospective order')
    expected = _expected_fit_bindings(selection, boundary=boundary)
    endpoints = _committed_fit_endpoints(committed, boundary=boundary)
    out = {new: old[old_id] for new, old_id in OLD_IDS.items()}
    for row in committed['models']:
        name = row['model_id']; size_text, rep_text, endpoint = name.split('_')
        size, rep = int(size_text[1:]), int(rep_text[1:])
        require(type(row['game_count']) is int and row['game_count'] == size and
            type(row['repetition']) is int and row['repetition'] == rep and row['endpoint'] == endpoint,
            'Fixed size/seed/endpoint identity')
        manifest = _read(row['checkpoint'], boundary)
        fit.validate_checkpoint_metadata(manifest['metadata'])
        meta = manifest['metadata']
        expected_meta, schedule = expected[size, rep]
        require(row['checkpoint'] == endpoints[name] and all(meta.get(key) == value for key, value in expected_meta.items()),
                'Exact selected subset, schedule, targets and verified fit endpoint')
        step = selection['subsets'][str(size)]['one_pass_steps'] if endpoint == 'onepass' else 1782
        require(type(row['checkpoint_step']) is int and row['checkpoint_step'] == step and
            meta['checkpoint_step'] == step and meta['game_count'] == size and meta['repetition'] == rep and
            meta['endpoint'] == endpoint and
            meta['qualification_only'] is False and meta['weights_sha256'] == row['weights_sha256'],
            'Committed exact endpoint step and weights')
        require(meta['record_presentations'] == sum(map(len, schedule['batches'][:step])),
                'Exact selected-schedule example presentations')
        c.verify([manifest['checkpoint'], manifest['tensor_manifest']], boundary=boundary)
        out[name] = dict(checkpoint=row['checkpoint'], weights_sha256=row['weights_sha256'])
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


def run_evaluation(selection, original_commitment, original_evaluation, study_commitment, folder,
                   *, boundary=lambda: None, runtime_sources=None):
    from . import fit
    folder = Path(folder)
    selection_sha = c.digest(selection)
    bindings = fixed_bindings(selection, original_commitment, study_commitment, boundary=boundary)
    # Saved evidence is fully checked, including source rows, masks and exact float32 outputs.
    old_bundles, old_result = previous.load_predictions(source_selection(), original_commitment,
                                                       original_evaluation, boundary=boundary)
    del old_bundles
    fit.engine(folder, runtime_sources=runtime_sources)
    groups = example_groups(source_selection(), original_commitment, boundary=boundary)
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
                    dict(schema="data_size_prediction_batch_v1", model_id=model_id, split=split,
                         batch_index=index, **binding, selection_sha256=selection_sha,
                         study_commitment=study_commitment, rows=rows)))
        require(accepted.state_digest(model) == binding["weights_sha256"], "evaluation leaves weights unchanged")
        results.append(dict(model_id=model_id, **binding, batches=batches,
                            root_counts={s: len(x) for s, x in groups.items()}, forward_calls=len(batches)))
        del model
    value = dict(schema="data_size_evaluation_v1", status="all_fixed_predictions_complete",
                 selection_sha256=selection_sha, original_commitment=original_commitment,
                 original_evaluation=original_evaluation, study_commitment=study_commitment,
                 reused_model_ids=OLD_IDS, models=results, optimizer_calls=0,
                 forward_calls=sum(x["forward_calls"] for x in results),
                 original_forward_calls_reexecuted=0, all_endpoints_committed_before_prediction=True,
                 checkpoint_selection=False, heldout_games=128, heldout_reused_development_only=True,
                 reference_promoted=False, **sampling.IDENTITY)
    item = c.save(folder / "evaluation_result.json", value)
    boundary()
    return item


def load_predictions(selection, original_commitment, original_evaluation, study_commitment, evaluation,
                     *, boundary=lambda: None):
    selection_sha = c.digest(selection)
    bindings = fixed_bindings(selection, original_commitment, study_commitment, boundary=boundary)
    old, old_result = previous.load_predictions(source_selection(), original_commitment, original_evaluation, boundary=boundary)
    bundles = {new: old[old_id] for new, old_id in OLD_IDS.items()}
    value = _read(evaluation, boundary)
    require(value["schema"] == "data_size_evaluation_v1" and value["status"] == "all_fixed_predictions_complete" and
            value["selection_sha256"] == selection_sha and value["original_commitment"] == original_commitment and
            value["original_evaluation"] == original_evaluation and value["study_commitment"] == study_commitment and
            value["reused_model_ids"] == OLD_IDS and value["optimizer_calls"] == 0 and
            value["all_endpoints_committed_before_prediction"] is True and
            value["original_forward_calls_reexecuted"] == 0 and
            [x["model_id"] for x in value["models"]] == list(FRESH_IDS), "complete fixed evaluation identity")
    require(all(value.get(k)==v for k,v in sampling.IDENTITY.items()), 'Corrected evaluation provenance identity')
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
            require(batch["schema"] == "data_size_prediction_batch_v1" and batch["model_id"] == model_id and
                    type(batch["batch_index"]) is int and batch["batch_index"] == index and
                    batch["study_commitment"] == study_commitment and
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
    cache = corpus.load_cache(source_selection(), boundary=boundary)
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
    definitions = {'parent': {'parent': 1.}}
    for size, endpoint in GROUPS:
        for rep in (0, 1):
            name = f'g{size}_r{rep}_{endpoint}'
            definitions[name] = {name: 1.}
        definitions[f'g{size}_mean_{endpoint}'] = {f'g{size}_r{rep}_{endpoint}': .5 for rep in (0, 1)}
    return definitions


def _contrast_definitions():
    result = []
    for prefix in ('r0', 'r1', 'mean'):
        for mode in ('onepass', 'matched'):
            for small, large in ((256, 512), (512, 1024), (256, 1024)):
                left = f'g{small}_{prefix}_{mode}'
                right = f'g{large}_{prefix}_{"onepass" if large == 1024 else mode}'
                result.append((f'{mode}_{small}_to_{large}', prefix, left, right))
        for size in (256, 512):
            result.append((f'extra_training_g{size}', prefix, f'g{size}_{prefix}_onepass', f'g{size}_{prefix}_matched'))
        for size, endpoint in GROUPS:
            result.append((f'parent_to_g{size}_{endpoint}', prefix, 'parent', f'g{size}_{prefix}_{endpoint}'))
    return result


def _membership_sets(selection, parent_rows):
    available = {r['game_index'] for r in parent_rows}
    result = {}
    for size in (256, 512):
        ids = set(selection['subsets'][str(size)]['game_ids'])
        result[str(size)] = dict(selected=sorted(available & ids), outside=sorted(available - ids))
    return result


def membership_summaries(game_summaries, selection):
    """Descriptive equal-game scores on identical membership masks for every model."""
    parent = game_summaries['parent']['train']
    masks = _membership_sets(selection, parent)
    out = {}
    for size, groups in masks.items():
        out[size] = {}
        for part, ids in groups.items():
            out[size][part] = dict(game_ids=ids, games=len(ids), models={})
            for name, coefficients in _definitions().items():
                rows_by_model = {model:[r for r in game_summaries[model]['train'] if r['game_index'] in set(ids)]
                                 for model in coefficients}
                representative = next(iter(rows_by_model.values()))
                out[size][part]['models'][name] = dict(roots=sum(r['roots'] for r in representative),
                    metrics={m: sum(weight*math.fsum(r[m] for r in rows_by_model[model])/len(representative)
                                    for model,weight in coefficients.items()) if representative else None for m in METRICS})
    return out

def analyze_predictions(bundles, labels, selection, *, draws=DRAW_COUNT, seed=BOOTSTRAP_SEED,
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
    return dict(schema="data_size_analysis_v1", status="fixed_data_size_diagnostics_analyzed",
        training_membership=membership_summaries(summaries, selection),
        comparison_design=dict(game_counts=list(SIZES),optimization_seeds=[0,1],matched_optimizer_updates=1782,
            onepass_is_not_matched_compute=True,partial_batches_retained=True,
            subset_selection_uncertainty_estimated=False,training_membership_descriptive=True),
        game_summaries=summaries, estimates=estimates, contrasts=contrasts,
        bootstrap=dict(seed=seed, draws=draws, generator="numpy.PCG64", unit="whole_game", strata=index_receipts,
                       paired_same_draws_all_models=True, pooled="partition_size_weighted_same_draws",
                       training_panel="descriptive_no_bootstrap", intervals="nominal95_descriptive_no_selection"),
        checkpoint_selection=False, strength_improvement_established=False, reference_promoted=False,
        limitations=["Both held-out partitions have been inspected repeatedly and are reused development diagnostics.",
          "Two fixed seeds share the same games; these intervals do not estimate arbitrary training-seed variation.",
          "Teacher-gap closure measures agreement with this saved teacher, not optimal play or all learnable information.",
          "One-pass comparisons change both distinct data and optimizer effort; only matched-update endpoints fix optimizer step count.",
          "Matched 1782 updates have slightly different exposure counts because genuine partial end-of-pass batches are retained.",
          "The same nested subsets are used for both seeds; intervals do not measure training-subset selection uncertainty.",
          "The common training panel includes games outside smaller fits; membership summaries use identical masks for every model.",
          "A data-size trend is conditional on this corpus, recipe and reused development panel; it is not a scaling-law forecast.",
          "Better prediction losses do not establish stronger searched play; no live test or new collection launches here."])


def verify_arithmetic(bundles, labels, selection, reported, *, strict=True, boundary=lambda: None):
    """Independent array loss, game reduction, paired draw and ratio calculation."""
    import numpy as np
    require(reported["schema"] == "data_size_analysis_v1" and
            reported["status"] == "fixed_data_size_diagnostics_analyzed" and
            reported["checkpoint_selection"] is False and reported["strength_improvement_established"] is False and
            reported["reference_promoted"] is False, "fixed descriptive diagnostic status")
    if strict:
        require(reported["bootstrap"]["draws"] == DRAW_COUNT and reported["bootstrap"]["seed"] == BOOTSTRAP_SEED,
                "production analysis uses fixed draws and seed")
    require(reported["bootstrap"]["generator"] == "numpy.PCG64" and
            reported["bootstrap"]["unit"] == "whole_game" and
            reported["bootstrap"]["paired_same_draws_all_models"] is True,
            "fixed paired whole-game uncertainty recipe")
    require(reported['comparison_design']==dict(game_counts=list(SIZES),optimization_seeds=[0,1],matched_optimizer_updates=1782,
        onepass_is_not_matched_compute=True,partial_batches_retained=True,
        subset_selection_uncertainty_estimated=False,training_membership_descriptive=True), 'Fixed data-size comparison interpretation')
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
    contrast_map = {(label, rep):(left, right) for label, rep, left, right in _contrast_definitions()}
    expected_keys = {(label, rep, split, metric) for label, rep in contrast_map
                     for split in (*SPLITS, 'pooled') for metric in METRICS}
    require(len(reported['contrasts']) == len(expected_keys) and
        {(r['contrast'], r['seed'], r['split'], r['metric']) for r in reported['contrasts']} == expected_keys,
        'Complete prospective data-size contrasts')
    for row in reported['contrasts']:
        a, b = contrast_map[row['contrast'], row['seed']]
        split, metric = row["split"], row["metric"]
        expected = reported["estimates"][b][split][metric]["mean"] - reported["estimates"][a][split][metric]["mean"]
        require(math.isclose(expected, row["difference"], rel_tol=1e-10, abs_tol=1e-12), "independent paired data-size difference")
        interval = None if split == "train" else np.percentile((combined[b][split] - combined[a][split])[:, METRICS.index(metric)],
                                                               [2.5, 97.5], method="linear")
        require(row["ci95"] is None if interval is None else np.allclose(row["ci95"], interval, rtol=1e-10, atol=1e-12),
                "independent paired data-size interval")
    # Rebuild membership using independent numerical matrices and source game IDs.
    membership=reported['training_membership']
    require(set(membership)=={'256','512'}, 'Both fixed subset membership views')
    train_games=[r['game_index'] for r in reported['game_summaries']['parent']['train']]
    for size in (256,512):
        selected=set(selection['subsets'][str(size)]['game_ids'])
        require(set(membership[str(size)])=={'selected','outside'}, 'Selected/outside descriptive masks')
        for part in ('selected','outside'):
            indices=[i for i,g in enumerate(train_games) if (g in selected)==(part=='selected')]
            record=membership[str(size)][part]
            require(record['game_ids']==[train_games[i] for i in indices] and record['games']==len(indices)
                and set(record['models'])==set(definitions), 'Same game membership mask for every predictor')
            roots=sum(reported['game_summaries']['parent']['train'][i]['roots'] for i in indices)
            for name,coefficients in definitions.items():
                saved=record['models'][name]
                require(saved['roots']==roots and set(saved['metrics'])==set(METRICS), 'Membership root and metric inventory')
                expected=None if not indices else sum(weight*matrices['train'][MODEL_IDS.index(model),indices,:].mean(axis=0)
                    for model,weight in coefficients.items())
                for j,metric in enumerate(METRICS):
                    require(saved['metrics'][metric] is None if expected is None else
                        math.isclose(saved['metrics'][metric],expected[j],rel_tol=1e-10,abs_tol=1e-12),
                        'Independent membership-matched descriptive metric')
    return dict(status="independent_data_size_arithmetic_verified", model_calls=0, optimizer_calls=0,
                independent_loss_formula=True, independent_count_weight_bootstrap=True,
                same_input_teacher_entropy=True, gap_ratio_of_game_means=True,
                draws=draws, seed=seed, contrasts=len(expected_keys), fixed_models=len(MODEL_IDS))


def render_charts(result, output, *, boundary=lambda:None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    folder=Path(output);artifacts=[]
    colors={'onepass':'#0f766e','matched':'#c2410c'}
    def records(mode,metric,prefix='mean'):
        return [result['estimates'][f'g{size}_{prefix}_{"onepass" if size==1024 else mode}']['pooled'][metric]
                for size in SIZES]
    fig,ax=plt.subplots(figsize=(9,5.1))
    for mode,label in (('onepass','One pass: data and update count both increase'),
                       ('matched','Matched training: 1,782 optimizer updates')):
        for rep in (0,1):
            values=[r['mean'] for r in records(mode,'teacher_gap_closed',f'r{rep}')]
            if all(v is not None for v in values):ax.plot(SIZES,[100*v for v in values],color=colors[mode],alpha=.25,lw=1)
        rows=records(mode,'teacher_gap_closed')
        if all(r['mean'] is not None for r in rows):
            ax.plot(SIZES,[100*r['mean'] for r in rows],marker='o',color=colors[mode],lw=2.5,label=label)
            if all(r['ci95'] is not None for r in rows):
                ax.fill_between(SIZES,[100*r['ci95'][0] for r in rows],[100*r['ci95'][1] for r in rows],
                    color=colors[mode],alpha=.12)
    ax.axhline(0,color='#64748b',ls='--',lw=1)
    ax.set_xticks(SIZES);ax.set_xlabel('Distinct training games')
    ax.set_ylabel('Initial policy gap to saved teacher closed (%)')
    ax.set_title('Does more independent training data improve teacher imitation?')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.025,'Bold: fixed two-seed mean. Thin: seeds. Shading: nominal paired whole-game 95% intervals.\n'
        'All points use the same 128 reused development games. The 1,024-game point is reused. Not a strength test.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,1))
    path=folder/'data_size_teacher_gap.png';require(path.resolve().is_relative_to(c.OUT) and not path.exists(),'Exclusive new chart')
    fig.savefig(path,dpi=160);plt.close(fig);artifacts.append(c.artifact(path));boundary()
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,title in zip(axes,('policy_ce','mixed_value_mse'),('Policy imitation error','Mixed value-target error')):
        baseline=result['estimates']['parent']['pooled'][metric]['mean']
        for mode,label in (('onepass','One pass'),('matched','Matched 1,782 updates')):
            for rep in (0,1):
                ax.plot(SIZES,[r['mean']-baseline for r in records(mode,metric,f'r{rep}')],color=colors[mode],alpha=.25,lw=1)
            values=[r['mean']-baseline for r in records(mode,metric)]
            ax.plot(SIZES,values,color=colors[mode],marker='o',lw=2.5,label=label)
            contrasts=[next(r for r in result['contrasts'] if r['contrast']==f'parent_to_g{size}_{"onepass" if size==1024 else mode}'
                and r['seed']=='mean' and r['split']=='pooled' and r['metric']==metric) for size in SIZES]
            ax.fill_between(SIZES,[r['ci95'][0] for r in contrasts],[r['ci95'][1] for r in contrasts],color=colors[mode],alpha=.12)
        ax.axhline(0,color='#64748b',ls='--',lw=1);ax.set_xticks(SIZES);ax.set_xlabel('Distinct training games')
        ax.set_ylabel('Change from original model (lower is better)');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18)
    axes[0].legend(frameon=False)
    fig.suptitle('More games versus more repetitions of fewer games',fontsize=14)
    fig.text(.5,.025,'Same architecture, original initialization, targets and learning rate. Two fixed optimization seeds.\n'
        'Matched updates retain genuine partial batches; example exposures differ slightly. Prediction gains do not prove stronger play.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.93))
    path=folder/'data_size_prediction_errors.png';require(path.resolve().is_relative_to(c.OUT) and not path.exists(),'Exclusive new chart')
    fig.savefig(path,dpi=160);plt.close(fig);artifacts.append(c.artifact(path));boundary()
    return artifacts

def run_analysis(selection, original_commitment, original_evaluation, study_commitment, evaluation, output,
                 *, boundary=lambda: None):
    require(not any(x == "torch" or x.startswith("torch.") for x in sys.modules), "fresh model-free analysis worker")
    bundles, source = load_predictions(selection, original_commitment, original_evaluation, study_commitment,
                                       evaluation, boundary=boundary)
    labels = source_labels(selection, boundary=boundary)
    result = analyze_predictions(bundles, labels, selection, boundary=boundary)
    check = verify_arithmetic(bundles, labels, selection, result, boundary=boundary)
    check_item = c.save(Path(output) / "arithmetic_verification.json", check)
    result.update(selection_sha256=c.digest(selection), original_commitment=original_commitment,
                  original_evaluation=original_evaluation, study_commitment=study_commitment,
                  evaluation=evaluation, verified_forward_calls=source["forward_calls"],
                  analysis_model_calls=0, analysis_optimizer_calls=0, independent_arithmetic=check_item)
    item = c.save(Path(output) / "analysis.json", result)
    charts = render_charts(result, output, boundary=boundary)
    require(not any(x == "torch" or x.startswith("torch.") for x in sys.modules), "analysis remained model-free")
    return dict(status="data_size_saved_analysis_verified", analysis=item, arithmetic=check_item, charts=charts,
                checkpoint_selection=False, models=0, optimizer_calls=0)


def run_saved_smoke(folder, *, boundary=lambda: None, runtime_sources):
    """Actual-source preflight: parse and score reused evidence, without models."""
    require(not any(x == "torch" or x.startswith("torch.") for x in sys.modules), "model-free saved smoke")
    selection = c.read(c.SELECTION)
    original_commitment = c.artifact(c.ORIGINAL_COMMITMENT)
    original_evaluation = c.artifact(c.ORIGINAL_EVALUATION)
    bundles, source = previous.load_predictions(source_selection(), original_commitment, original_evaluation, boundary=boundary)
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
