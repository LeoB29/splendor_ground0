"""Exploratory, saved-outcome-only budget curves after experiment 159.

Reads immutable predecessor artifacts, verifies discovery/selection hashes, and
creates new diagnostic files exclusively. Never imports game or inference code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/planner_preference_159/review_sampling_curve_20260915"
BUDGETS = [1, 2, 4, 8, 16]
PARTITIONS = 512
BOOTSTRAPS = 20_000


def read(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def saved_curve(experiment: int):
    if experiment == 159:
        base = ROOT / "outputs/planner_preference_159"
        stage, lock_name, split_seed, bootstrap_seed = (
            "selection", "selection_lock.json", 159515, 159516)
        policies = ["GG", "LL"]
    else:
        assert experiment == 154
        base = ROOT / "outputs/rollout_selector_154"
        stage, lock_name, split_seed, bootstrap_seed = (
            "targets/discovery", "discovery_targets_lock.json", 154515, 154516)
        policies = ["GG"]
    panel, panel_sha = read(base / "panel.json")
    lock, lock_sha = read(base / lock_name)
    samples = panel["samples"]
    expected = 180 if experiment == 159 else 768
    assert len(samples) == expected
    independent_ids = [s["pair" if experiment == 159 else "source_index"] for s in samples]
    assert len(set(independent_ids)) == expected
    rng = np.random.default_rng(split_seed)
    permutations = np.array([rng.permutation(32) for _ in range(PARTITIONS)])
    values = {policy: [] for policy in policies}
    roots = []
    for sample in samples:
        index = sample["index"]
        name = f"root_{index:03d}.json" if experiment == 159 else f"state_{index:04d}.json"
        path = base / stage / name
        record, sha = read(path)
        expected_sha = lock["files"][str(index)]
        if experiment == 154:
            expected_sha = expected_sha["sha256"]
        assert sha == expected_sha, path
        assert record["index"] == index and len(record["worlds"]) == 32
        root_row = {
            "index": index, "independent_source_id": independent_ids[len(roots)],
            "cell": sample["cell"], "turn": sample["turn"], "actor": sample["actor"],
            "source_path": path.relative_to(ROOT).as_posix(), "source_sha256": sha,
            "gain_score_units": {},
        }
        for policy in policies:
            outcomes = [w["outcomes"][policy] if experiment == 159 else w["outcomes"]
                        for w in record["worlds"]]
            scores = (np.array([[a["wdl"] for a in row] for row in outcomes], dtype=float) + 1) / 2
            assert np.isin(scores, [0, 0.5, 1]).all()
            held_out = scores[permutations[:, 16:]].mean(axis=1)
            gains = []
            for budget in BUDGETS:
                chosen = scores[permutations[:, :budget]].mean(axis=1).argmax(axis=1)
                gains.append(float((held_out[np.arange(PARTITIONS), chosen] - held_out[:, 0]).mean()))
            values[policy].append(gains)
            root_row["gain_score_units"][policy] = gains
        roots.append(root_row)
    cells = np.array([s["cell"] for s in samples])
    strata = [np.where(cells == cell)[0] for cell in sorted(set(cells))]
    assert len({len(ix) for ix in strata}) == 1
    assert all(len(ix) > 1 for ix in strata)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    draws = np.concatenate([bootstrap_rng.choice(ix, (BOOTSTRAPS, len(ix)), replace=True)
                            for ix in strata], axis=1)
    result = {}
    for policy, raw_values in values.items():
        observed = np.asarray(raw_values)
        corrected = observed.copy()
        for ix in strata:
            mean = observed[ix].mean(axis=0)
            corrected[ix] = mean + np.sqrt(len(ix)/(len(ix)-1)) * (observed[ix] - mean)
        bootstrap_means = corrected[draws].mean(axis=1)
        increments = np.diff(observed.mean(axis=0))
        bootstrap_increments = np.diff(bootstrap_means, axis=1)
        phases = np.array([min(s["turn"] // 20, 2) for s in samples])
        result[policy] = {
            "mean_gain_pp": (100 * observed.mean(axis=0)).tolist(),
            "pointwise_exploratory_bootstrap_95_intervals_pp":
                (100 * np.quantile(bootstrap_means, [0.025, 0.975], axis=0).T).tolist(),
            "successive_doubling_increment_pp": (100 * increments).tolist(),
            "pointwise_exploratory_increment_95_intervals_pp":
                (100 * np.quantile(bootstrap_increments, [0.025, 0.975], axis=0).T).tolist(),
            "phase_mean_gain_pp": {str(phase): (100 * observed[phases == phase].mean(axis=0)).tolist()
                                   for phase in range(3)},
        }
    summary = {
        "experiment": experiment, "independent_sources": expected,
        "panel_path": (base / "panel.json").relative_to(ROOT).as_posix(), "panel_sha256": panel_sha,
        "lock_path": (base / lock_name).relative_to(ROOT).as_posix(), "lock_sha256": lock_sha,
        "all_source_record_hashes_verified": True,
        "split_seed": split_seed, "bootstrap_seed": bootstrap_seed,
        "stratum_sizes": [len(ix) for ix in strata], "policies": result,
    }
    return summary, roots


def save_exclusive(path: Path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    # Refuse to overwrite or silently reuse an earlier diagnostic run.
    OUTPUT.mkdir(exist_ok=False)
    summaries, roots = {}, {}
    for experiment in [159, 154]:
        summaries[str(experiment)], roots[str(experiment)] = saved_curve(experiment)
    root_path = OUTPUT / "roots.json"
    save_exclusive(root_path, roots)
    summary = {
        "status": "exploratory_saved_data_review_not_new_experiment",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "budgets": BUDGETS,
        "partitions_per_root": PARTITIONS, "evaluation_worlds_per_partition": 16,
        "bootstrap_resamples": BOOTSTRAPS,
        "methodology": [
            "Read all four (or smaller lawful support) action outcomes from each immutable 32-world selection/discovery record; verify every byte hash against its original lock.",
            "Draw 512 permutations of 32 world indices. The same permutations are applied to all roots and both policies within an experiment. For each partition, nested first1/2/4/8/16 worlds choose first-argmax in original candidate order.",
            "Evaluate that chosen action minus candidate0 on the last16 disjoint worlds under the same continuation policy. Convert WDL to win1/draw0.5/loss0 before averaging.",
            "Average the512 partition results within each source root before population estimation. Partitions and simulated worlds are not extra independent observations.",
            "Use complete distinct source pairs (159) or source games (154) as the statistical unit, stratifying on the original36 or6 cells. All cells and roots retain their original equal weights.",
            "For exploratory bootstrap intervals, scale each root's deviation from its stratum mean by sqrt(n/(n-1)); n=5 for159 gives sqrt(5/4), matching the finite-stratum variance correction. Resample corrected complete root vectors within strata, preserving paired budget differences.",
            "Intervals are pointwise exploratory percentile intervals, not simultaneous confirmatory gates. No model fitting, neural inference, new simulated trajectory or gameplay was performed.",
        ],
        "limitations": [
            "Repeated partitions reuse the same32-world dataset. This estimates resampled lower-budget held-out behavior; it is not fresh32-world or128-world validation.",
            "These are expected terminal-score differences under fixed continuation policies, not actual repeated-replanning match win rates or a ground-truth action-quality oracle.",
            "Positive lower-budget scaling does not establish32-to128 scaling, its size, lack of saturation, or its causal role in158's result.",
            "Intervals reflect root-level exploratory uncertainty conditional on these fixed source populations and saved outcomes. No post-hoc budget, phase, or policy is selected for deployment.",
        ],
        "roots_path": root_path.relative_to(ROOT).as_posix(),
        "roots_sha256": hashlib.sha256(root_path.read_bytes()).hexdigest(),
        "experiments": summaries,
    }
    save_exclusive(OUTPUT / "summary.json", summary)
    print(json.dumps({key: value["policies"] for key, value in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
