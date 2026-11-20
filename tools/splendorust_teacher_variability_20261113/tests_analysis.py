"""Model-free counterexamples for finite-seed, paired-game variability claims."""
from copy import deepcopy
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from . import analysis as a


def fixture():
    panel = dict(roots=[], host_metadata={}, primary_root_ids=[], deep_root_ids=[], probe_actions67={})
    rows = []
    for g in range(1024, 1152):
        split = "validation" if g < 1088 else "offline"
        chosen = f"g{g}_p000000"
        for j in range(4):
            rid = f"g{g}_p{j:06d}"
            panel["roots"].append(dict(root_id=rid))
            panel["host_metadata"][rid] = dict(game_index=g, split=split)
        panel["primary_root_ids"].append(chosen)
        deep = (g - 1024) % 64 < 32
        if deep: panel["deep_root_ids"].append(chosen)
        panel["probe_actions67"][chosen] = [0, 1]
        for quota in ((512, 2048) if deep else (512,)):
            for pool, mode, w, e in a._conditions():
                offset = (-3, -1, 1, 3)["ABCD".index(w)] if mode == "legacy" else (-2 if w == "A" else 2) + (-1 if e == "A" else 1)
                scale = (12 + g % 5) * (2 if pool == 3 else 1)
                count = quota // 2 + offset * scale
                visits = [count, quota - count] + [0] * 65
                qs = [.42 + .004 * "ABCD".index(w), .58 - .003 * "ABCD".index(e)] + [None] * 65
                value = sum(n * q for n, q in zip(visits, qs) if n) / quota
                rows.append(dict(root_id=chosen, game_index=g, split=split, quota=quota, pool=pool, mode=mode,
                                 replicate=w + "_" + e, world_seed_label=w, encoding_seed_label=e,
                                 main_action_index67=0 if count >= quota / 2 else 1, visits67=visits, q67=qs,
                                 root_value=value, root_network_value=.5, elapsed_seconds=quota / 1000 + pool / 100))
    return rows, panel


class VariabilityAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.panel = fixture()
        cls.report = a.analyze(cls.rows, cls.panel)

    def test_complete_inventory_and_independent_reader(self):
        receipt = a.verify_arithmetic(self.rows, self.panel, self.report)
        self.assertEqual(receipt["decisions"], 3072)
        self.assertEqual(receipt["primary_roots"], 128)
        self.assertEqual(receipt["deep_roots"], 64)
        self.assertEqual(self.report["classification"]["status"], "six_worlds_lower_legacy_distribution_variability")
        self.assertLess(self.report["primary"]["difference_ci95"][1], 0)
        self.assertFalse(self.report["playing_strength_established"])
        self.assertFalse(self.report["noise_floor_estimated"])

    def test_distribution_endpoints_and_zero_entries(self):
        self.assertEqual(a.distribution_pair([5, 0], [10, 0]), (0., 0.))
        js, tv = a.distribution_pair([1, 0], [0, 1])
        self.assertAlmostEqual(js, math.log(2))
        self.assertEqual(tv, 1.)
        self.assertEqual(a.distribution_pair([2, 1], [1, 2]), a.distribution_pair([1, 2], [2, 1]))

    def test_fixed_pair_families(self):
        self.assertEqual([len(a._pairs(s)) for s in a.SCOPES], [6, 6, 2, 2, 2])
        for x, y in a._pairs("split_change_world"):
            self.assertNotEqual(x[0], y[0]); self.assertEqual(x[1], y[1])
        for x, y in a._pairs("split_change_encoding"):
            self.assertEqual(x[0], y[0]); self.assertNotEqual(x[1], y[1])
        for x, y in a._pairs("split_change_both"):
            self.assertNotEqual(x[0], y[0]); self.assertNotEqual(x[1], y[1])

    def test_inventory_rejects_missing_duplicate_and_unplanned_decisions(self):
        with self.assertRaises(ValueError): a._inventory(self.rows[:-1], self.panel)
        with self.assertRaises(ValueError): a._inventory(self.rows + [self.rows[0]], self.panel)
        rows = self.rows.copy(); rows[-1] = dict(rows[-1], root_id="unknown")
        with self.assertRaises(ValueError): a._inventory(rows, self.panel)

    def test_panel_game_independence_and_selection_binding(self):
        for change in ("duplicate", "two_same_game", "wrong_deep", "wrong_split", "missing_probe", "same_probe"):
            panel = deepcopy(self.panel)
            if change == "duplicate": panel["primary_root_ids"][-1] = panel["primary_root_ids"][0]
            elif change == "two_same_game": panel["primary_root_ids"][-1] = "g1024_p000001"
            elif change == "wrong_deep": panel["deep_root_ids"][-1] = "g1024_p000001"
            elif change == "wrong_split": panel["host_metadata"]["g1024_p000000"]["split"] = "offline"
            elif change == "missing_probe": del panel["probe_actions67"]["g1024_p000000"]
            else: panel["probe_actions67"]["g1024_p000000"] = [0, 0]
            with self.subTest(change=change), self.assertRaises(ValueError): a._metadata(panel)

    def test_rejects_malformed_and_wrong_units(self):
        meta = a._metadata(self.panel)[self.rows[0]["root_id"]]
        updates = [dict(quota=True), dict(pool=True), dict(main_action_index67=True),
                   dict(main_action_index67=67), dict(mode="unknown"), dict(replicate="wrong"),
                   dict(world_seed_label=None), dict(encoding_seed_label="C"), dict(game_index=1025),
                   dict(elapsed_seconds=math.nan), dict(root_value=-.1), dict(root_value=.99),
                   dict(root_network_value=2.), dict(visits67=[512] * 81), dict(q67=[None] * 81)]
        for update in updates:
            row = deepcopy(self.rows[0]); row.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError): a.validate_record(row, meta)
        for field, value in (("visits67", True), ("visits67", -1), ("q67", None), ("q67", math.nan)):
            row = deepcopy(self.rows[0]); row[field][0] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): a.validate_record(row, meta)
        row = deepcopy(self.rows[0]); row["q67"][2] = 0.
        with self.assertRaises(ValueError): a.validate_record(row, meta)

    def root_case(self):
        rid = self.rows[0]["root_id"]
        rows = [deepcopy(r) for r in self.rows if r["root_id"] == rid]
        panel = dict(roots=[dict(root_id=rid)], host_metadata={rid: self.panel["host_metadata"][rid]},
                     primary_root_ids=[rid], deep_root_ids=[rid], probe_actions67={rid: [0, 1]})
        return rows, panel

    def test_actual_tie_decisions_not_first_argmax(self):
        rows, panel = self.root_case()
        for r in rows:
            r["visits67"] = [r["quota"] // 2] * 2 + [0] * 65
            r["root_value"] = statistics.fmean(r["q67"][:2])
            r["main_action_index67"] = int(r["world_seed_label"] in "CD")
        meta, book = a._inventory(rows, panel, strict=False)
        root = a._root_rows(meta, book)[0]
        self.assertEqual(root["pools"]["3"]["jsd"], 0.)
        self.assertAlmostEqual(root["pools"]["3"]["action_disagreement"], 4 / 6)

    def test_q_missingness_uses_complete_common_support(self):
        rows, panel = self.root_case()
        r = rows[0]
        r["visits67"] = [0, r["quota"]] + [0] * 65
        r["q67"][0] = None; r["main_action_index67"] = 1; r["root_value"] = r["q67"][1]
        meta, book = a._inventory(rows, panel, strict=False)
        roots = a._root_rows(meta, book)
        for row in roots:
            if row["quota"] == 512:
                self.assertFalse(row["q_joint_supported"])
                self.assertIsNone(row["pools"]["3"]["q_contrast_pair_rms"])
                self.assertIsNone(row["pools"]["6"]["q_contrast_pair_rms"])
            else: self.assertTrue(row["q_joint_supported"])

    def test_all_pairs_value_rms_is_sample_standard_deviation(self):
        meta, book = a._inventory(self.rows, self.panel)
        root = self.report["roots"][0]
        vals = [book[root["root_id"]][(512, 3, "legacy", *s)]["root_value"] for s in a.LEGACY]
        self.assertAlmostEqual(root["pools"]["3"]["root_value_pair_rms"], statistics.stdev(vals))

    def test_whole_game_paired_bootstrap_and_strata(self):
        primary = self.report["primary"]
        self.assertEqual(primary["games"], 128)
        self.assertEqual(primary["bootstrap_strata"]["validation"]["games"], 64)
        self.assertEqual(primary["bootstrap_strata"]["offline"]["games"], 64)
        for cell in self.report["cells"]:
            count = 128 if cell["quota"] == 512 else 64
            self.assertEqual(cell["planned_roots"], count)
            self.assertEqual(cell["metrics"]["jsd"]["games"], count)
        matrix = [[0., .2], [.4, .6], [.8, 1.]]
        sample, _ = a._bootstrap(matrix, ["validation", "validation", "offline"], draws=100,
                                 seed=a.BOOTSTRAP_SEED)
        for v in sample: self.assertAlmostEqual(v[1] - v[0], .2)

    def test_rejects_unfrozen_uncertainty_and_tampered_reports(self):
        with self.assertRaises(ValueError): a.analyze(self.rows, self.panel, draws=100)
        report = deepcopy(self.report); report["bootstrap"]["seed"] += 1
        with self.assertRaises(ValueError): a.verify_arithmetic(self.rows, self.panel, report)
        with self.assertRaises(ValueError): a._close([1.], [2.])
        with self.assertRaises(ValueError): a._close(dict(a=False), dict(a=0))

    def test_unresolved_is_not_no_effect_or_negative_result(self):
        result = a._classification(dict(difference_ci95=[-.1, .1]))
        self.assertEqual(result["status"], "legacy_distribution_variability_difference_unresolved")
        self.assertFalse(result["strength_inference"])
        result = a._classification(dict(difference_ci95=[.01, .02]))
        self.assertEqual(result["status"], "six_worlds_higher_legacy_distribution_variability")

    def test_budget_comparison_uses_only_identical_deep_roots(self):
        deep = set(self.panel["deep_root_ids"])
        games = {self.panel["host_metadata"][rid]["game_index"] for rid in deep}
        for cell in self.report["matched_budget_contrasts"]:
            self.assertEqual(cell["planned_roots"], 64)
            self.assertTrue(cell["exploratory"])
            result = cell["metrics"]["jsd"]
            self.assertEqual(result["games"], 64)
            self.assertEqual({r["game_index"] for r in result["game_summaries"]}, games)
            self.assertEqual(result["bootstrap_strata"]["validation"]["games"], 32)
            self.assertEqual(result["bootstrap_strata"]["offline"]["games"], 32)
            self.assertAlmostEqual(result["difference_2048_minus_512"],
                                   result["estimates"]["2048"]["mean"] - result["estimates"]["512"]["mean"])

    def test_matched_budget_q_coverage_is_intersection(self):
        roots = deepcopy(self.report["roots"])
        rid = self.panel["deep_root_ids"][0]
        for row in roots:
            if row["root_id"] == rid and row["quota"] == 2048:
                for pool in ("3", "6"): row["pools"][pool]["q_contrast_pair_rms"] = None
        result = a._matched_budget(roots, draws=30, seed=a.BOOTSTRAP_SEED)
        for cell in result:
            self.assertEqual(cell["metrics"]["q_contrast_pair_rms"]["games"], 63)
            self.assertEqual(cell["metrics"]["jsd"]["games"], 64)

    def test_chart_renders_constant_and_off_center_percentile_intervals(self):
        from .charts import render
        report = deepcopy(self.report)
        # A percentile interval can exclude the point estimate; preserve it,
        # rather than manufacturing nonnegative asymmetric error magnitudes.
        estimate = report["cells"][0]["metrics"]["jsd"]["estimates"]["3"]
        estimate["ci95"] = [estimate["mean"] + .001, estimate["mean"] + .002]
        with tempfile.TemporaryDirectory(prefix="variability_chart_test_") as folder:
            paths = render(report, folder)
            self.assertEqual(len(paths), 2)
            for path in paths: self.assertEqual(Path(path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
