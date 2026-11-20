"""Synthetic cross-implementation contracts; production matrices are never read."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from tools.saved_adaptive_breadth_128 import analysis, common, verify
from tools.saved_equal_budget_breadth import analysis as prior_analysis


def synthetic_root(support, index=0):
    m, n = min(support, 4), min(support, 8)
    extension = 32 * (n-m)
    keys = [f"action-{j}" for j in range(n)]

    def block(width, count, offset):
        seeds = list(range(offset, offset+count))
        return {"candidate_indices": list(range(width)), "world_seeds": seeds,
                "worlds": [{"seed": seed, "outcomes": [
                    {"wdl": ((wi*(j+3)+j*5+offset) % 7) % 3-1}
                    for j in range(width)]} for wi, seed in enumerate(seeds)]}

    return {"spec": {"index": index, "pair": index+1000, "game_index": index+1000,
                     "fit": 161001, "seat": 0, "phase": 1},
            "budgets": {"canonical_support": support, "anchor_candidates": m,
                        "breadth_candidates": n, "extension_worlds": extension},
            "union": {"keys": keys},
            "candidates": {arm: {"indices": list(range(size)), "keys": keys[:size]}
                           for arm, size in (("anchor", m), ("breadth", n), ("precision", m))},
            "selection": block(n, 128, 10000), "extension": block(m, extension, 20000)}


@pytest.mark.parametrize("support", [1, 2, 3, 4, 5, 6, 7, 8, 9, 31])
def test_all_64_partitions_preserve_published_uniform_anchor_choices(support):
    record = synthetic_root(support, support)
    old = prior_analysis.analyze_root(record, common.CONFIG)
    result = analysis.analyze_root(record, common.CONFIG)
    assert verify.verify_root(record, result, common.CONFIG)["passed"] is True
    preserved = common.validate_preserved_choices(old, result)
    assert preserved == {"passed": True, "partitions": 64,
                         "prior_choices_preserved": True, "prior_scores_reused": False}
    for previous, current in zip(old["partitions"], result["partitions"], strict=True):
        assert all(value == 128*min(support, 4) for value in current["selection_cell_counts"].values())
        assert current["uniform_sums"] == previous["broad_sums"]
        assert current["anchor_sums"] == previous["reference_sums"]
        if support >= 8:
            assert current["assessment_indices"] == previous["common_order"][96:]
            assert current["assessment_count"] == 32
        else:
            assert current["assessment_indices"] == previous["assessment_indices"]
            assert current["choices"]["adaptive"] == current["choices"]["uniform"]
        if support <= 4:
            assert current["assessment_sums"] == {arm: None for arm in analysis.ARMS}
            assert all(value == [0, 1] for value in current["difference_fractions"].values())


@pytest.mark.parametrize("field", ["common_order", "extension_order", "remainder_order", "uniform", "anchor"])
def test_preservation_check_rejects_changed_partition_or_choice(field):
    record = synthetic_root(8)
    old = prior_analysis.analyze_root(record, common.CONFIG)
    result = analysis.analyze_root(record, common.CONFIG)
    part = result["partitions"][0]
    if field in ("uniform", "anchor"):
        part["choices"][field] = (part["choices"][field]+1) % 4
    else:
        part[field][0], part[field][1] = part[field][1], part[field][0]
    with pytest.raises(ValueError, match="Prior"):
        common.validate_preserved_choices(old, result)


def test_full_support_assessment_cannot_change_any_selector():
    config = {**common.CONFIG, "partitions": 1}
    record = synthetic_root(8, 13)
    before = analysis.analyze_root(record, config)
    changed = deepcopy(record)
    for wi in before["partitions"][0]["assessment_indices"]:
        for outcome in changed["selection"]["worlds"][wi]["outcomes"]:
            outcome["wdl"] = 1
    after = analysis.analyze_root(changed, config)
    assert verify.verify_root(changed, after, config)["passed"] is True
    first, second = before["partitions"][0], after["partitions"][0]
    for field in ("choices", "adaptive", "uniform_sums", "anchor_sums",
                  "selection_common_indices", "selection_extension_indices"):
        assert first[field] == second[field]
    assert second["assessment_sums"] == {arm: 64 for arm in analysis.ARMS}


def test_correlated_candidates_share_whole_worlds_in_both_stages():
    record = synthetic_root(8)
    for wi, world in enumerate(record["selection"]["worlds"]):
        shared = 1 if wi % 3 else -1
        for j, outcome in enumerate(world["outcomes"]):
            outcome["wdl"] = shared if j in (0, 4) else -1
    result = analysis.analyze_root(record, common.CONFIG)
    assert verify.verify_root(record, result, common.CONFIG)["passed"] is True
    for part in result["partitions"]:
        assert part["uniform_sums"][0] == part["uniform_sums"][4]
        assert part["adaptive"]["stage1_sums"][0] == part["adaptive"]["stage1_sums"][4]
        assert part["choices"]["uniform"] == part["choices"]["adaptive"] == 0


def test_all_source_outcomes_leave_permutation_and_remainder_identity_fixed():
    record = synthetic_root(7, 11)
    before = analysis.analyze_root(record, common.CONFIG)
    changed = deepcopy(record)
    for block in ("selection", "extension"):
        for world in changed[block]["worlds"]:
            for outcome in world["outcomes"]:
                outcome["wdl"] = -outcome["wdl"]
    after = analysis.analyze_root(changed, common.CONFIG)
    for first, second in zip(before["partitions"], after["partitions"], strict=True):
        for field in ("common_order", "extension_order", "remainder_order", "remainder_candidates", "assessment_indices"):
            assert first[field] == second[field]


def test_verifier_never_calls_analysis_helpers(monkeypatch):
    record = synthetic_root(8, 23)
    result = analysis.analyze_root(record, common.CONFIG)

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent verifier called the analysis implementation")

    for name in ("analyze_root", "_order", "_winner", "_matrix"):
        monkeypatch.setattr(analysis, name, forbidden)
    assert verify.verify_root(record, result, common.CONFIG)["passed"] is True


def test_statistics_pair_three_arms_across_cells_and_partial_bootstrap_chunk():
    config = {**common.CONFIG, "partitions": 2, "roots": 36, "cell_size": 2,
              "bootstrap_resamples": 501}
    rows = []
    for fit in config["fits"]:
        for seat in (0, 1):
            for phase in (0, 1, 2):
                for _ in range(2):
                    index = len(rows)
                    record = synthetic_root(5+index % 4, index)
                    record["spec"].update(fit=fit, seat=seat, phase=phase)
                    rows.append(analysis.analyze_root(record, config))
    stats = analysis.summarize(rows, config)
    receipt = verify.verify_statistics(rows, stats, config)
    assert receipt["passed"] is True and receipt["independent_bootstrap"] is True
    assert receipt["resampling_unit"] == "root"
    assert stats["roots"] == 36 and stats["root_partitions"] == 72
    assert stats["bootstrap_integer_denominator"] > 1
    assert set(stats["contrasts"]) == {name for name, _, _ in analysis.CONTRASTS}


def test_production_modules_have_no_game_model_or_old_analysis_imports():
    folder = Path(__file__).parent
    files = [folder/name for name in ("common.py", "analysis.py", "verify.py")]
    files.append(folder.parents[1] / "run_saved_adaptive_breadth_128.py")
    allowed = {"__future__", "argparse", "collections", "ctypes", "datetime", "fractions",
               "hashlib", "json", "math", "numpy", "os", "pathlib", "shutil", "subprocess",
               "sys", "time", "traceback"}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                assert module.split(".")[0] in allowed or module.startswith("tools.saved_adaptive_breadth_128"), (path, module)
