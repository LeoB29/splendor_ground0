"""Descriptive, paired whole-game analysis of already-opened saved predictions."""
import hashlib
import math
import re

import numpy as np

from .common import Context, require

METRICS = ("policy", "value", "combined")
SPLITS = ("validation", "offline")
EXPECTED_GAMES = {"validation": list(range(192, 224)), "offline": list(range(224, 256))}
EXPECTED_ROOTS = {"validation": 1802, "offline": 1755}
ROW_KEYS = {"game_index", "ply", "split", "root_input_sha256", "legal_mask81",
            "policy_target81", "value_target2", "output83"}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def prediction_losses(row):
    """Independent float64 masked CE and two-slot signed-target MSE."""
    require(set(row) == ROW_KEYS, "Exact saved prediction row schema")
    require(type(row["game_index"]) is int and type(row["ply"]) is int and row["ply"] >= 0,
            "Integer saved game/ply identity")
    require(row["split"] in SPLITS and re.fullmatch(r"[0-9a-f]{64}", row["root_input_sha256"]),
            "Declared split and canonical tensor digest")
    mask, target, output, value = (row[k] for k in
        ("legal_mask81", "policy_target81", "output83", "value_target2"))
    require(len(mask) == len(target) == 81 and all(type(x) is bool for x in mask) and any(mask),
            "Complete nonempty corrected legal mask")
    require(all(finite_number(x) and x >= 0 for x in target)
            and abs(math.fsum(target)-1.0) <= 1e-12
            and all(ok or x == 0 for ok, x in zip(mask, target)), "Normalized legal policy target")
    require(len(value) == 2 and all(finite_number(x) and x in (-1., 0., 1.) for x in value)
            and sum(value) == 0, "Official signed two-slot terminal targets")
    require(len(output) == 83 and all(finite_number(x) for x in output), "Finite saved output83")
    require(np.array_equal(np.asarray(output, dtype=np.float64),
                           np.asarray(output, dtype=np.float32).astype(np.float64)),
            "Actual float32 output values")
    legal = [x for x, ok in zip(output[:81], mask) if ok]
    maximum = max(legal)
    log_z = maximum + math.log(math.fsum(math.exp(x-maximum) for x in legal))
    policy = math.fsum(t*(log_z-x) for t, x in zip(target, output[:81]) if t)
    value_loss = math.fsum((x-y)**2 for x, y in zip(output[81:], value))/2
    require(math.isfinite(policy) and policy >= -1e-12 and math.isfinite(value_loss), "Finite losses")
    policy = max(0., policy)
    return {"policy": policy, "value": value_loss, "combined": policy+value_loss}


def paired_rows(parent, candidate):
    require(len(parent) == len(candidate) and len(parent) > 0, "Complete paired saved root inventory")
    seen = set()
    for a, b in zip(parent, candidate, strict=True):
        prediction_losses(a)
        prediction_losses(b)
        require({k: v for k, v in a.items() if k != "output83"} ==
                {k: v for k, v in b.items() if k != "output83"}, "Identical input, mask, target and row identity")
        identity = (a["game_index"], a["ply"])
        require(identity not in seen, "No duplicated saved root")
        seen.add(identity)
    require([(r["game_index"], r["ply"]) for r in parent] == sorted(seen), "Canonical game/ply order")


def game_summary(rows, expected_games):
    grouped = {g: [] for g in expected_games}
    for row in rows:
        require(row["game_index"] in grouped, "Every root belongs to declared held-out games")
        grouped[row["game_index"]].append(prediction_losses(row))
    require(all(grouped.values()), "Every declared held-out game has roots")
    games = [{"game_index": g, "roots": len(values),
              **{k: math.fsum(v[k] for v in values)/len(values) for k in METRICS}}
             for g, values in sorted(grouped.items())]
    return {"games": games, "game_count": len(games), "root_count": len(rows),
            "mean_game_losses": {k: math.fsum(g[k] for g in games)/len(games) for k in METRICS}}


def load_paired_rows(context):
    source = context.source
    result = context.read(source/"fit/heldout_diagnostics/result.json")
    manifest = context.read(source/"fit/checkpoint_manifest.json")
    fit = context.read(source/"fit/fit_result.json")
    opening = context.read(source/"fit/heldout_opening.json")
    item = context.bind(source/"fit/checkpoint_manifest.json")
    require(result["schema"] == "selfplay_heldout_diagnostics_v1"
            and result["status"] == "fixed_checkpoint_diagnostics_complete"
            and result["candidate_manifest"] == item and result["all_eligible_roots"] is True
            and result["checkpoint_selection"] is False and result["strength_claim"] is False,
            "Fixed final checkpoint and complete descriptive evidence")
    require(opening["checkpoint_fixed_before_heldout_opening"] is True
            and opening["checkpoint_manifest"] == item and opening["training_splits_opened"] == ["train"],
            "Held-out examples opened after final checkpoint")
    weights = {"parent": fit["parent_weights_sha256"], "candidate": fit["candidate_weights_sha256"]}
    require(weights["candidate"] == manifest["metadata"]["weights_sha256"], "Fixed learned weights identity")
    require(weights["parent"] == "32cdd9466befc527d265a356be65200c3666a4fa8a82cfefdc40b327b2f529db",
            "Original warm-start parent weights")
    descriptors = result["batches"]
    require(len(descriptors) == len({x["path"] for x in descriptors}) == result["forward_calls"] == 224,
            "Exact complete saved batch count")
    by_path = {x["path"]: x for x in descriptors}
    used, data, reconstructed = [], {}, {}
    for model in ("parent", "candidate"):
        data[model], reconstructed[model] = {}, {}
        for split in SPLITS:
            folder = source/"fit/heldout_diagnostics"/model/split
            count = math.ceil(EXPECTED_ROOTS[split]/32)
            expected_paths = [folder/f"{i:06d}.json" for i in range(count)]
            require(sorted(folder.glob("*.json")) == expected_paths, "Exact saved batch file inventory")
            rows = []
            for i, path in enumerate(expected_paths):
                context.check()
                descriptor = context.bind(path)
                require(by_path.get(descriptor["path"]) == descriptor, "Authenticated saved batch bytes")
                batch = context.read(path)
                require(set(batch) == {"model", "split", "batch_index", "candidate_manifest_fixed", "weights_sha256", "rows"}
                        and batch["model"] == model and batch["split"] == split
                        and batch["batch_index"] == i and batch["candidate_manifest_fixed"] == item
                        and batch["weights_sha256"] == weights[model], "Exact batch/checkpoint identities")
                require(len(batch["rows"]) == min(32, EXPECTED_ROOTS[split]-32*i)
                        and all(r["split"] == split for r in batch["rows"]), "Exact complete batch rows")
                rows.extend(batch["rows"])
                used.append(descriptor)
            reconstructed[model][split] = game_summary(rows, EXPECTED_GAMES[split])
            data[model][split] = rows
    require(used == descriptors and reconstructed == result["summaries"], "Independent exact game-loss reproduction")
    deltas = {split: {k: reconstructed["candidate"][split]["mean_game_losses"][k]-
                     reconstructed["parent"][split]["mean_game_losses"][k] for k in METRICS} for split in SPLITS}
    require(deltas == result["candidate_minus_parent_mean_game_losses"], "Independent exact saved mean differences")
    for split in SPLITS:
        paired_rows(data["parent"][split], data["candidate"][split])
    return result, data


def bootstrap_game_means(deltas, rng, replicates, check=lambda: None):
    values = np.asarray(deltas, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] == 3
            and np.isfinite(values).all(), "Finite whole-game three-loss matrix")
    require(type(replicates) is int and replicates > 0, "Positive fixed bootstrap count")
    chunks, digest = [], hashlib.sha256()
    for start in range(0, replicates, 512):
        check()
        indices = rng.integers(0, len(values), size=(min(512, replicates-start), len(values)), dtype=np.int64)
        digest.update(indices.astype("<i8", copy=False).tobytes())
        chunks.append(values[indices].mean(axis=1))
    return np.concatenate(chunks), digest.hexdigest()


def describe_game_differences(parent_games, candidate_games, sampled_means):
    require([(g["game_index"], g["roots"]) for g in parent_games] ==
            [(g["game_index"], g["roots"]) for g in candidate_games], "Game identities and root counts match")
    require(len(parent_games) >= 2, "At least two complete games")
    draws = np.asarray(sampled_means, dtype=float)
    require(draws.ndim == 2 and draws.shape[1] == 3 and np.isfinite(draws).all(), "Finite bootstrap means")
    result = {}
    n, roots = len(parent_games), sum(g["roots"] for g in parent_games)
    for j, k in enumerate(METRICS):
        delta = [b[k]-a[k] for a, b in zip(parent_games, candidate_games)]
        average = math.fsum(delta)/n
        loo = [(math.fsum(delta)-x)/(n-1) for x in delta]
        parent_mean = math.fsum(a[k] for a in parent_games)/n
        candidate_mean = math.fsum(a[k] for a in candidate_games)/n
        intervals = np.quantile(draws[:, j], [0.025, 0.975], method="linear").tolist()
        result[k] = {"parent_game_mean": parent_mean, "candidate_game_mean": candidate_mean,
                     "candidate_minus_parent_game_mean": average,
                     "relative_change_percent": 100*(candidate_mean/parent_mean-1) if parent_mean else None,
                     "nominal_paired_game_bootstrap_ci95": intervals,
                     "median_game_delta": float(np.median(delta)),
                     "games_improved": sum(x < 0 for x in delta), "games_equal": sum(x == 0 for x in delta),
                     "games_worsened": sum(x > 0 for x in delta),
                     "leave_one_game_out_mean_range": [min(loo), max(loo)],
                     "root_weighted_mean_delta": math.fsum(x*g["roots"] for x, g in zip(delta, parent_games))/roots,
                     "bootstrap_degenerate": bool(np.all(draws[:, j] == draws[0, j])),
                     "per_game": [{"game_index": g["game_index"], "roots": g["roots"], "delta": x}
                                  for g, x in zip(parent_games, delta)]}
    return {"game_count": n, "root_count": roots, "losses": result}


def value_bias(parent, candidate):
    """Condition on output slot and target sign; first average within each game."""
    paired_rows(parent, candidate)
    result = {}
    for slot in range(2):
        result[str(slot)] = {}
        for sign in (-1, 0, 1):
            by_game = {}
            for a, b in zip(parent, candidate):
                if a["value_target2"][slot] == sign:
                    by_game.setdefault(a["game_index"], []).append(
                        (a["output83"][81+slot]-sign, b["output83"][81+slot]-sign))
            n = len(by_game)
            means = {variant: [math.fsum(x[j] for x in rows)/len(rows) for rows in by_game.values()]
                     for j, variant in enumerate(("parent", "candidate"))}
            result[str(slot)][str(sign)] = {"games_contributing": n,
                "roots": sum(len(x) for x in by_game.values()),
                "parent_mean_game_signed_residual": math.fsum(means["parent"])/n if n else None,
                "candidate_mean_game_signed_residual": math.fsum(means["candidate"])/n if n else None,
                "candidate_minus_parent_mean_game_signed_residual":
                    math.fsum(b-a for a, b in zip(means["parent"], means["candidate"]))/n if n else None}
    return {"conditioning": "canonical value output slot and target sign; not player seat",
            "weighting": "equal weight per contributing game within each stratum", "strata": result}


def analyze(context):
    config = context.plan["bootstrap"]
    require(config == {"interval": .95, "pooled64": "descriptive_only", "replicates": 20000,
                       "seed": 20261104256, "separate_splits": True,
                       "unit": "whole_game_paired_parent_candidate"}, "Exact frozen analysis choices")
    saved, rows = load_paired_rows(context)
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    split_results, draws, index_hashes = {}, {}, {}
    for split in SPLITS:
        a, b = (saved["summaries"][v][split]["games"] for v in ("parent", "candidate"))
        differences = [[y[k]-x[k] for k in METRICS] for x, y in zip(a, b)]
        draws[split], index_hashes[split] = bootstrap_game_means(differences, rng, config["replicates"], context.check)
        split_results[split] = describe_game_differences(a, b, draws[split])
        split_results[split]["value_bias"] = value_bias(rows["parent"][split], rows["candidate"][split])
    a = [g for split in SPLITS for g in saved["summaries"]["parent"][split]["games"]]
    b = [g for split in SPLITS for g in saved["summaries"]["candidate"][split]["games"]]
    pooled = describe_game_differences(a, b, (draws["validation"]+draws["offline"])/2)
    pooled["interpretation"] = "Descriptive pooled64 with independent whole-game resampling within each32-game split."
    return {"status": "heldout_saved_losses_reproduced", "exact_source_summary_reproduction": True,
            "paired_roots": sum(EXPECTED_ROOTS.values()), "paired_games": 64, "saved_batches": 224,
            "splits": split_results, "pooled64_descriptive": pooled,
            "bootstrap": {**config, "generator": "numpy.PCG64", "numpy_version": np.__version__,
                          "quantile_method": "linear", "draw_order": "validation then offline, replicate-major int64",
                          "same_indices_across_losses": True, "indices_sha256": index_hashes},
            "limitations": ["Already-opened data; nominal intervals are descriptive and not multiplicity-adjusted.",
                "Resampling games preserves within-game dependence but does not prove independent outcomes at every root.",
                "Lower loss is better; a loss decrease is not a playing-strength measurement.",
                "Opposite split signs do not establish overfitting, value harm or policy/value causal interference.",
                "No new checkpoint, training recipe selection or assessment sample extension."]}


def main():
    context = Context()
    print(context.finish("heldout.json", analyze(context)))


if __name__ == "__main__":
    main()
