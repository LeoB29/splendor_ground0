"""Frozen, game-paired variability analysis; no model or environment imports.

Pair disagreement measures repeatability, not correctness or a noise floor.
The crossed streams expose sensitivity, not an additive variance decomposition.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import hashlib
import math
import statistics

DRAW_COUNT = 20000
BOOTSTRAP_SEED = 202611133072
PRIMARY_ROOTS = 128
DEEP_ROOTS = 64
POOLS = (3, 6)
QUOTAS = (512, 2048)
LEGACY = tuple((x, x) for x in "ABCD")
SPLIT = tuple((w, e) for w in "AB" for e in "AB")
SCOPES = ("legacy_total", "split_total", "split_change_world",
          "split_change_encoding", "split_change_both")
METRICS = ("jsd", "tv", "action_disagreement", "root_value_pair_rms",
           "q_contrast_pair_rms", "visit_entropy", "largest_visit_share")


def require(condition, message):
    if not condition:
        raise ValueError("TEACHER_VARIABILITY_ANALYSIS: " + message)


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _metadata(panel, *, strict=True):
    require(type(panel) is dict and type(panel.get("roots")) is list, "Original panel roots")
    original = {}
    for row in panel["roots"]:
        rid = row["root_id"]
        require(type(rid) is str and rid and rid not in original, "Unique original roots")
        host = panel.get("host_metadata", {}).get(rid, row)
        g, s = host["game_index"], host["split"]
        require(type(g) is int and s in ("validation", "offline"), "Source game and split")
        original[rid] = dict(root_id=rid, game_index=g, split=s)
    primary, deep, probes = (panel[k] for k in ("primary_root_ids", "deep_root_ids", "probe_actions67"))
    require(type(primary) is list and type(deep) is list and len(set(primary)) == len(primary)
            and len(set(deep)) == len(deep), "Unique frozen selection lists")
    require(set(primary) <= set(original) and set(deep) <= set(primary), "Nested existing source roots")
    require(type(probes) is dict and set(probes) == set(primary), "Exact frozen Q probe inventory")
    meta = {rid: dict(original[rid], deep=rid in set(deep), probe_actions67=probes[rid]) for rid in primary}
    for r in meta.values():
        p = r["probe_actions67"]
        require(type(p) is list and len(p) == 2 and p[0] != p[1]
                and all(type(a) is int and 0 <= a < 67 for a in p), "Two distinct frozen Main probes")
    games = {r["game_index"] for r in meta.values()}
    require(len(games) == len(meta), "Exactly one selected root per independent source game")
    if strict:
        by_game = defaultdict(list)
        for r in original.values(): by_game[r["game_index"]].append(r)
        require(len(original) == 512 and len(by_game) == 128 and all(len(v) == 4 for v in by_game.values()),
                "Original panel has four roots in each of 128 games")
        require(len(meta) == PRIMARY_ROOTS and len(deep) == DEEP_ROOTS
                and games == set(range(1024, 1152)), "Frozen 128 primary and 64 nested deep roots")
        require(all(r["split"] == ("validation" if r["game_index"] < 1088 else "offline")
                    for r in original.values()), "Exact original source split assignments")
        require(all(sum(meta[r]["split"] == s for r in deep) == 32 for s in ("validation", "offline")),
                "Deep subset contains 32 games per original split")
    return meta


def _conditions():
    return tuple((p, mode, w, e) for p in POOLS for mode, seeds in
                 (("legacy", LEGACY), ("split", SPLIT)) for w, e in seeds)


def validate_record(row, meta):
    require(type(row) is dict and row.get("root_id") == meta["root_id"], "Exact record root")
    require(type(row.get("game_index")) is int and row["game_index"] == meta["game_index"]
            and row.get("split") == meta["split"], "Record game/split binding")
    q, p, mode = row.get("quota"), row.get("pool"), row.get("mode")
    w, e = row.get("world_seed_label"), row.get("encoding_seed_label")
    require(type(q) is int and q in QUOTAS and (q != 2048 or meta["deep"]), "Planned quota")
    require(type(p) is int and (p, mode, w, e) in _conditions(), "Frozen pool and stream condition")
    require(row.get("replicate") == w + "_" + e, "Replicate binds both seed labels")
    action, visits, values = (row.get(k) for k in ("main_action_index67", "visits67", "q67"))
    require(type(action) is int and 0 <= action < 67, "Actual canonical Main action")
    require(type(visits) is list and type(values) is list and len(visits) == len(values) == 67,
            "Complete canonical 67-channel evidence")
    require(all(type(n) is int and n >= 0 for n in visits) and sum(visits) == q, "Exact fixed visit quota")
    for n, v in zip(visits, values, strict=True):
        require((v is None) if n == 0 else _finite(v) and 0 <= v <= 1,
                "Visited Q is win credit; unvisited Q remains missing")
    require(visits[action] > 0 and visits[action] == max(visits), "Actual decision attains maximum visits")
    value = row.get("root_value")
    require(_finite(value) and 0 <= value <= 1, "Root value uses win-credit units")
    require(abs(value - math.fsum(n * v for n, v in zip(visits, values) if n) / q) <= 2e-6,
            "Root value binds the visit-weighted root edges")
    require(_finite(row.get("elapsed_seconds")) and row["elapsed_seconds"] >= 0, "Finite actual response time")
    if "root_network_value" in row:
        require(_finite(row["root_network_value"]) and 0 <= row["root_network_value"] <= 1,
                "Optional root-network value uses win-credit units")
    return q, p, mode, w, e


def _inventory(rows, panel, *, strict=True, boundary=lambda: None):
    meta = _metadata(panel, strict=strict)
    require(type(rows) is list, "Decision rows list")
    book = {rid: {} for rid in meta}
    for i, row in enumerate(rows):
        if i % 64 == 0: boundary()
        rid = row.get("root_id")
        require(rid in meta, "No unplanned root")
        key = validate_record(row, meta[rid])
        require(key not in book[rid], "No duplicate decision cell")
        book[rid][key] = row
    for rid, r in meta.items():
        expected = {(q, *c) for q in ((512, 2048) if r["deep"] else (512,)) for c in _conditions()}
        require(set(book[rid]) == expected, "Complete planned decision cells for " + rid)
    return meta, book


def _pairs(scope):
    seeds = LEGACY if scope == "legacy_total" else SPLIT
    pairs = list(combinations(seeds, 2))
    if scope == "split_change_world": return [(a, b) for a, b in pairs if a[0] != b[0] and a[1] == b[1]]
    if scope == "split_change_encoding": return [(a, b) for a, b in pairs if a[0] == b[0] and a[1] != b[1]]
    if scope == "split_change_both": return [(a, b) for a, b in pairs if a[0] != b[0] and a[1] != b[1]]
    return pairs


def distribution_pair(a, b):
    """Natural-log JSD and TV on normalized counts; zero mass is handled exactly."""
    p = [n / sum(a) for n in a]
    q = [n / sum(b) for n in b]
    middle = [(x + y) / 2 for x, y in zip(p, q, strict=True)]
    js = .5 * math.fsum(x * math.log(x / m) for x, m in zip(p, middle) if x)
    js += .5 * math.fsum(x * math.log(x / m) for x, m in zip(q, middle) if x)
    return max(0., js), .5 * math.fsum(abs(x - y) for x, y in zip(p, q))


def _root_rows(meta, book, *, independent=False, boundary=lambda: None):
    out = []
    if independent: import numpy as np
    for rid in sorted(meta):
        boundary()
        host, cells = meta[rid], book[rid]
        for quota in ((512, 2048) if host["deep"] else (512,)):
            relevant = [r for k, r in cells.items() if k[0] == quota]
            probe = host["probe_actions67"]
            supported = all(r["visits67"][a] > 0 for r in relevant for a in probe)
            missing = [dict(pool=r["pool"], mode=r["mode"], replicate=r["replicate"],
                            actions=[a for a in probe if r["visits67"][a] == 0])
                       for r in relevant if any(r["visits67"][a] == 0 for a in probe)]
            for scope in SCOPES:
                mode = "legacy" if scope == "legacy_total" else "split"
                pool_values = {}
                for pool in POOLS:
                    selected = {s: cells[(quota, pool, mode, *s)] for s in (LEGACY if mode == "legacy" else SPLIT)}
                    pairs = _pairs(scope)
                    if independent:
                        probabilities = {s: np.asarray(r["visits67"], dtype=np.float64) / quota for s, r in selected.items()}
                        js, tv, disagree, value2, contrast2 = [], [], [], [], []
                        for sa, sb in pairs:
                            a, b = selected[sa], selected[sb]
                            x, y = probabilities[sa], probabilities[sb]
                            m = (x + y) / 2
                            entropy = lambda v: -float(np.sum(v[v > 0] * np.log(v[v > 0])))
                            js.append(max(0., entropy(m) - .5 * entropy(x) - .5 * entropy(y)))
                            tv.append(float(np.abs(x - y).sum()) / 2)
                            disagree.append(a["main_action_index67"] != b["main_action_index67"])
                            value2.append((a["root_value"] - b["root_value"]) ** 2 / 2)
                            if supported:
                                contrast2.append(((a["q67"][probe[0]] - a["q67"][probe[1]]) -
                                                  (b["q67"][probe[0]] - b["q67"][probe[1]])) ** 2 / 2)
                        values = dict(jsd=float(np.mean(js)), tv=float(np.mean(tv)), action_disagreement=float(np.mean(disagree)),
                                      root_value_pair_rms=float(np.sqrt(np.mean(value2))),
                                      q_contrast_pair_rms=float(np.sqrt(np.mean(contrast2))) if supported else None,
                                      visit_entropy=float(np.mean([entropy(p) for p in probabilities.values()])),
                                      largest_visit_share=float(np.mean([p.max() for p in probabilities.values()])))
                    else:
                        js, tv, disagree, value2, contrast2 = [], [], [], [], []
                        for sa, sb in pairs:
                            a, b = selected[sa], selected[sb]
                            j, t = distribution_pair(a["visits67"], b["visits67"])
                            js.append(j); tv.append(t)
                            disagree.append(float(a["main_action_index67"] != b["main_action_index67"]))
                            value2.append((a["root_value"] - b["root_value"]) ** 2 / 2)
                            if supported:
                                ca = a["q67"][probe[0]] - a["q67"][probe[1]]
                                cb = b["q67"][probe[0]] - b["q67"][probe[1]]
                                contrast2.append((ca - cb) ** 2 / 2)
                        probs = [[n / quota for n in r["visits67"]] for r in selected.values()]
                        values = dict(jsd=statistics.fmean(js), tv=statistics.fmean(tv),
                                      action_disagreement=statistics.fmean(disagree),
                                      root_value_pair_rms=math.sqrt(statistics.fmean(value2)),
                                      q_contrast_pair_rms=math.sqrt(statistics.fmean(contrast2)) if supported else None,
                                      visit_entropy=statistics.fmean(-math.fsum(p * math.log(p) for p in v if p) for v in probs),
                                      largest_visit_share=statistics.fmean(max(p) for p in probs))
                    pool_values[str(pool)] = values
                out.append(dict(root_id=rid, game_index=host["game_index"], split=host["split"], quota=quota,
                                scope=scope, pairs=len(_pairs(scope)), probe_actions67=probe,
                                q_joint_supported=supported, missing_q_probes=missing, pools=pool_values))
    return out


def _bootstrap(matrix, splits, *, draws, seed, independent=False, boundary=lambda: None):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed))
    sample = np.zeros((draws, 2), dtype=np.float64)
    strata = {}
    for split in ("validation", "offline"):
        which = [i for i, s in enumerate(splits) if s == split]
        if not which: continue
        values = np.asarray([matrix[i] for i in which], dtype=np.float64)
        n = len(values)
        indices = rng.integers(0, n, size=(draws, n), dtype=np.int64)
        strata[split] = dict(games=n, index_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())
        for start in range(0, draws, 256):
            boundary()
            part = indices[start:start + 256]
            if independent:
                counts = np.asarray([np.bincount(row, minlength=n) for row in part])
                sample[start:start + len(part)] += counts @ values
            else:
                sample[start:start + len(part)] += values[part].sum(axis=1)
    return sample / len(matrix), strata


def _reduce(root_rows, *, draws, seed, independent=False, boundary=lambda: None):
    import numpy as np
    cells = []
    for quota in QUOTAS:
        for scope in SCOPES:
            rows = [r for r in root_rows if r["quota"] == quota and r["scope"] == scope]
            metrics = {}
            for metric in METRICS:
                grouped = defaultdict(list); splits = {}
                for row in rows:
                    vals = [row["pools"][str(p)][metric] for p in POOLS]
                    if any(v is None for v in vals): continue
                    grouped[row["game_index"]].append(vals); splits[row["game_index"]] = row["split"]
                games = []
                for g in sorted(grouped):
                    vals = grouped[g]
                    mean = np.asarray(vals).mean(axis=0).tolist() if independent else [statistics.fmean(v[i] for v in vals) for i in range(2)]
                    games.append(dict(game_index=g, split=splits[g], roots=len(vals), values=mean))
                if games:
                    matrix = [g["values"] for g in games]
                    sample, strata = _bootstrap(matrix, [g["split"] for g in games], draws=draws, seed=seed,
                                                 independent=independent, boundary=boundary)
                    means = np.asarray(matrix).mean(axis=0).tolist() if independent else [statistics.fmean(v[i] for v in matrix) for i in range(2)]
                    cis = np.quantile(sample, [.025, .975], axis=0, method="linear")
                    diff = float(np.mean(np.asarray(matrix)[:, 1] - np.asarray(matrix)[:, 0]))
                    ci = np.quantile(sample[:, 1] - sample[:, 0], [.025, .975], method="linear").tolist()
                    estimates = {str(p): dict(mean=means[i], ci95=cis[:, i].tolist()) for i, p in enumerate(POOLS)}
                else:
                    estimates = {str(p): dict(mean=None, ci95=None) for p in POOLS}
                    diff = ci = None; strata = {}
                metrics[metric] = dict(estimates=estimates, difference_6_minus_3=diff, difference_ci95=ci,
                                       games=len(games), roots=sum(g["roots"] for g in games),
                                       game_summaries=games, bootstrap_strata=strata)
            cells.append(dict(quota=quota, scope=scope, planned_roots=len(rows),
                              q_joint_supported_roots=sum(r["q_joint_supported"] for r in rows), metrics=metrics))
    return cells


def _timing(rows):
    import numpy as np
    out = []
    for quota in QUOTAS:
        for pool in POOLS:
            for mode in ("legacy", "split"):
                values = [r["elapsed_seconds"] for r in rows if (r["quota"], r["pool"], r["mode"]) == (quota, pool, mode)]
                out.append(dict(quota=quota, pool=pool, mode=mode, decisions=len(values),
                                mean=statistics.fmean(values), median=statistics.median(values),
                                p95=float(np.quantile(values, .95)), maximum=max(values),
                                over_one_second=sum(v > 1. for v in values)))
    return out


def _matched_budget(root_rows, *, draws, seed, independent=False, boundary=lambda: None):
    """Secondary budget differences on the same deep-subset source games.

Q contrasts require support at both budgets and both pools, rather than
silently comparing differently supported populations.
"""
    import numpy as np
    lookup = {(r["root_id"], r["quota"], r["scope"]): r for r in root_rows}
    deep = sorted({r["root_id"] for r in root_rows if r["quota"] == 2048})
    out = []
    for pool in POOLS:
        for scope in SCOPES:
            metrics = {}
            for metric in METRICS:
                grouped = defaultdict(list); splits = {}
                for rid in deep:
                    pair = [lookup[rid, q, scope] for q in QUOTAS]
                    values = [r["pools"][str(pool)][metric] for r in pair]
                    if any(v is None for v in values): continue
                    game = pair[0]["game_index"]
                    require(all(r["game_index"] == game and r["split"] == pair[0]["split"] for r in pair),
                            "Matched budget source-game identity")
                    grouped[game].append(values); splits[game] = pair[0]["split"]
                games = []
                for g in sorted(grouped):
                    values = grouped[g]
                    mean = (np.asarray(values).mean(axis=0).tolist() if independent else
                            [statistics.fmean(v[i] for v in values) for i in range(2)])
                    games.append(dict(game_index=g, split=splits[g], roots=len(values), values=mean))
                if games:
                    matrix = [g["values"] for g in games]
                    sample, strata = _bootstrap(matrix, [g["split"] for g in games], draws=draws, seed=seed,
                                                 independent=independent, boundary=boundary)
                    means = (np.asarray(matrix).mean(axis=0).tolist() if independent else
                             [statistics.fmean(v[i] for v in matrix) for i in range(2)])
                    bounds = np.quantile(sample, [.025, .975], axis=0, method="linear")
                    estimates = {str(q): dict(mean=means[i], ci95=bounds[:, i].tolist()) for i, q in enumerate(QUOTAS)}
                    difference = float(np.mean(np.asarray(matrix)[:, 1] - np.asarray(matrix)[:, 0]))
                    interval = np.quantile(sample[:, 1] - sample[:, 0], [.025, .975], method="linear").tolist()
                else:
                    estimates = {str(q): dict(mean=None, ci95=None) for q in QUOTAS}
                    difference = interval = None; strata = {}
                metrics[metric] = dict(estimates=estimates, difference_2048_minus_512=difference,
                                      difference_ci95=interval, games=len(games),
                                      roots=sum(g["roots"] for g in games), game_summaries=games, bootstrap_strata=strata)
            out.append(dict(pool=pool, scope=scope, planned_roots=len(deep),
                            comparison="2048_minus_512_on_identical_deep_subset", exploratory=True, metrics=metrics))
    return out


def _classification(primary):
    ci = primary["difference_ci95"]
    if ci[1] < 0: status = "six_worlds_lower_legacy_distribution_variability"
    elif ci[0] > 0: status = "six_worlds_higher_legacy_distribution_variability"
    else: status = "legacy_distribution_variability_difference_unresolved"
    return dict(status=status, metric="legacy_total_jsd", quota=512, contrast="pool6_minus_pool3",
                strength_inference=False, noise_floor_estimated=False, followup_launched=False)


def analyze(rows, panel, *, draws=DRAW_COUNT, seed=BOOTSTRAP_SEED, boundary=lambda: None):
    require(type(draws) is int and draws == DRAW_COUNT and type(seed) is int and seed == BOOTSTRAP_SEED,
            "Frozen uncertainty settings")
    meta, book = _inventory(rows, panel, boundary=boundary)
    roots = _root_rows(meta, book, boundary=boundary)
    cells = _reduce(roots, draws=draws, seed=seed, boundary=boundary)
    matched = _matched_budget(roots, draws=draws, seed=seed, boundary=boundary)
    primary = next(c for c in cells if c["quota"] == 512 and c["scope"] == "legacy_total")["metrics"]["jsd"]
    return dict(schema="teacher_variability_analysis_v1", status="complete_saved_root_analysis", decisions=len(rows),
                primary_roots=len(meta), deep_roots=sum(r["deep"] for r in meta.values()), roots=roots, cells=cells,
                primary=primary, matched_budget_contrasts=matched,
                classification=_classification(primary), timing=_timing(rows),
                bootstrap=dict(draws=draws, seed=seed, generator="numpy.PCG64", unit="whole_game",
                               stratification="original_validation_offline", weighting="equal_game_mean_of_root_metrics",
                               primary_interval="two_sided_95_percent_paired_percentile",
                               other_intervals="exploratory_not_multiplicity_adjusted"),
                limitations=[
                    "All roots are reused development evidence from the same 128 source games, with one root selected per game.",
                    "Four repeats estimate finite-seed repeatability; pairwise comparisons are not independent observations.",
                    "Budget effects use only the same 64 deep-subset roots; the full 512 and 2048 panels have different populations.",
                    "JSD is a distribution discrepancy, not an irreducible noise floor or a fraction of learnable training signal.",
                    "Actual Main-action agreement is not an accuracy ceiling; auxiliary decisions are outside these summaries.",
                    "Changing a stream measures conditional sensitivity, not an additive or exclusive cause of variability.",
                    "Split-stream results describe a separate diagnostic implementation; legacy-stream results are primary.",
                    "Q-contrast variation uses only fixed probes supported jointly by every compared repeat and pool; no missing Q is imputed.",
                    "Pair RMS equals sample SD for all-pairs summaries; conditional two-pair summaries are RMS sensitivities, not fitted variance components.",
                    "Lower variability can preserve systematic errors and does not establish stronger play, a better teacher, or a model-promotion decision."],
                playing_strength_established=False, noise_floor_estimated=False, new_games=0, followup_launched=False)


def _close(actual, expected, label="arithmetic"):
    if expected is None:
        require(actual is None, label); return
    if type(expected) in (int, bool, str):
        require(type(actual) is type(expected) and actual == expected, label); return
    if type(expected) is float:
        require(_finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label); return
    if type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), label)
        for a, b in zip(actual, expected): _close(a, b, label)
        return
    require(type(actual) is dict and type(expected) is dict and set(actual) == set(expected), label)
    for key in expected: _close(actual[key], expected[key], label + " " + key)


def verify_arithmetic(rows, panel, reported, *, boundary=lambda: None):
    """Recompute distributions by entropy differences and bootstrap by counts.

Inventory/field validation is shared; arithmetic is separate from the primary
KL-ratio and indexed-sum paths. No model calls or saved evidence mutation.
"""
    meta, book = _inventory(rows, panel, boundary=boundary)
    require(reported["schema"] == "teacher_variability_analysis_v1" and
            reported["status"] == "complete_saved_root_analysis", "Analysis identity")
    b = reported["bootstrap"]
    require(b == dict(draws=DRAW_COUNT, seed=BOOTSTRAP_SEED, generator="numpy.PCG64", unit="whole_game",
                      stratification="original_validation_offline", weighting="equal_game_mean_of_root_metrics",
                      primary_interval="two_sided_95_percent_paired_percentile",
                      other_intervals="exploratory_not_multiplicity_adjusted"), "Frozen bootstrap contract")
    roots = _root_rows(meta, book, independent=True, boundary=boundary)
    cells = _reduce(roots, draws=DRAW_COUNT, seed=BOOTSTRAP_SEED, independent=True, boundary=boundary)
    matched = _matched_budget(roots, draws=DRAW_COUNT, seed=BOOTSTRAP_SEED, independent=True, boundary=boundary)
    _close(reported["roots"], roots, "Independent root arithmetic")
    _close(reported["cells"], cells, "Independent equal-game/count-bootstrap arithmetic")
    _close(reported["matched_budget_contrasts"], matched, "Independent paired same-root budget arithmetic")
    primary = next(c for c in cells if c["quota"] == 512 and c["scope"] == "legacy_total")["metrics"]["jsd"]
    _close(reported["primary"], primary, "Independent primary contrast")
    _close(reported["classification"], _classification(primary), "Classification")
    _close(reported["timing"], _timing(rows), "Timing inventory")
    require(reported["decisions"] == len(rows) and reported["primary_roots"] == len(meta)
            and reported["deep_roots"] == sum(r["deep"] for r in meta.values()), "Exact counts")
    require(reported["playing_strength_established"] is False and reported["noise_floor_estimated"] is False
            and reported["new_games"] == 0 and reported["followup_launched"] is False, "Scope retained")
    return dict(schema="teacher_variability_arithmetic_acceptance_v1", status="independent_arithmetic_verified",
                decisions=len(rows), primary_roots=len(meta), deep_roots=sum(r["deep"] for r in meta.values()),
                entropy_identity=True, multinomial_count_bootstrap=True, no_models_executed=True)
