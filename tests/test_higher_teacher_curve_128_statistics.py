"""Synthetic-only validation of paired learning-curve inference and scan bands."""
import copy
import math
import numpy as np
import pytest

from tools.higher_teacher_curve_128 import statistics as st


@pytest.fixture(autouse=True)
def bounded_bootstrap(monkeypatch):
    monkeypatch.setitem(st.c.CONFIG, "bootstrap_samples", 79)
    monkeypatch.setitem(st.c.CONFIG, "bootstrap_chunk", 17)


def synthetic_curves(*, uncertain=False, same_after_initial=False):
    curves = {str(update): [] for update in st.CHECKPOINTS}
    for si, split in enumerate(st.SPLITS):
        for fi, fit in enumerate(st.FITS):
            for rank in range(14):
                pair = si*42+fi*14+rank
                omitted = (rank+fi) % 3
                for seat in (0, 1):
                    for phase in range(3):
                        if phase == omitted:
                            continue
                        index = len(curves["0"])
                        n = 3 if index % 53 == 0 else 8
                        scores = [128, 160, 100, 120, 112+2*rank+fi, 110, 170, 100+rank][:n]
                        missing = [0]*n
                        if uncertain and n == 8 and rank % 4 == 0:
                            missing[4] = 1
                        for ck, update in enumerate(st.CHECKPOINTS):
                            choices = {"anchor": 0, "teacher": 1, "teacher_outsider": 6 if n > 3 else None,
                                "original_fourth": 3 if n > 3 else None,
                                "anchor_direct": 0, "anchor_outsider": 3 if n > 3 else None, "anchor_supported": 0,
                                "broad_direct": 1, "broad_outsider": 4 if n > 3 else None, "broad_supported": 4 if n > 3 else 0,
                                "narrow_direct": 2, "narrow_outsider": 5 if n > 3 else None, "narrow_supported": 5 if n > 3 else 0,
                                "specialist_outsider": 6 if n > 3 else None, "specialist_supported": 6 if n > 3 else 0,
                                "lowrate_outsider": 7 if n > 3 else None, "lowrate_supported": 7 if n > 3 else 0}
                            final = 4 if same_after_initial else 4+(len(st.CHECKPOINTS)-1+rank) % 4
                            choices["finalrate_outsider"] = final if n > 3 else None
                            choices["finalrate_supported"] = final if n > 3 else 0
                            proposed = 3 if ck == 0 else (4 if same_after_initial else 4+(ck+rank) % 4)
                            choices["focused_outsider"] = proposed if n > 3 else None
                            choices["focused_supported"] = 0 if ck == 0 or n <= 3 else proposed
                            bounds = {name: None if j is None else [(scores[j]-missing[j])/256, (scores[j]+missing[j])/256]
                                      for name, j in choices.items()}
                            deltas = {}
                            for name, left, right in st.CONTRASTS:
                                deltas[name] = [0., 0.] if choices[left] == choices[right] else [
                                    bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
                            curves[str(update)].append({"index": index, "pair": pair, "game_index": 2*pair+seat,
                                "fit": fit, "seat": seat, "phase": phase, "split": split, "privacy": {"own": 0, "opponent": 0},
                                "support": n, "outsider_available": n > 3, "choices": choices,
                                "arm_score_bounds": bounds,
                                "arm_scores": {name: None if value is None or value[0] != value[1] else value[0] for name, value in bounds.items()},
                                "delta_bounds": deltas, "unknown_counts": missing.copy()})
    return curves


def assert_close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for name in a:
            assert_close(a[name], b[name])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_close(x, y)
    elif isinstance(a, float):
        assert math.isfinite(b) and abs(a-b) <= 1e-12
    else:
        assert a == b


@pytest.mark.parametrize("uncertain", [False, True])
def test_independent_histogram_and_raw_arithmetic_agree(uncertain):
    curves = synthetic_curves(uncertain=uncertain)
    direct = st.summarize(curves)
    independent = st.independent_summarize(curves)
    assert_close(direct, independent)
    assert direct["classification"] == "diagnostic_only"
    assert not direct["checkpoint_selection"] and not direct["promotion"]
    assert direct["shared_draws_across_checkpoints_and_endpoints"]
    assert direct["assessment_reused_development"]


def test_point_means_and_fit_summary_are_raw_equal_root_arithmetic():
    curves = synthetic_curves()
    result = st.summarize(curves)
    name = st.PRIMARY[0]
    for update in (0, 16, 512, 1784):
        panel = curves[str(update)]
        for split in st.SPLITS:
            expected = sum(r["delta_bounds"][name][0] for r in panel if r["split"] == split)/168
            assert result["curves"][str(update)]["groups"][split]["contrasts"][name]["mean"] == expected
            for fit in st.FITS:
                value = sum(r["delta_bounds"][name][0] for r in panel if r["split"] == split and r["fit"] == fit)/56
                assert result["curves"][str(update)]["per_fit"][str(fit)][split]["contrasts"][name]["mean"] == value
        val = result["curves"][str(update)]["groups"]["validation"]["contrasts"][name]["mean"]
        train = result["curves"][str(update)]["groups"]["train"]["contrasts"][name]["mean"]
        difference = result["curves"][str(update)]["groups"]["validation_minus_train"]["contrasts"][name]["mean"]
        assert abs(difference-(val-train)) < 1e-12
    assert result["states"] == 336 and result["pairs"] == 84


def test_identical_checkpoint_functions_receive_identical_paired_draws():
    curves = synthetic_curves(same_after_initial=True)
    arrays = st._arrays(curves)
    boot = st._bootstrap(arrays)
    for k in range(2, len(st.CHECKPOINTS)):
        assert np.array_equal(boot[:, :, 1], boot[:, :, k])
    # Only the two checkpoint0 anchor contrasts cancel. The other controls
    # are genuinely trained, so their initial differences must remain visible.
    for name in st.ANCHOR_CONTRASTS:
        assert np.array_equal(boot[:, :, 0, st.NAMES.index(name)], np.zeros((79, 2, 2)))
    assert np.any(boot[:, :, 0, st.NAMES.index("focused_outsider_minus_lowrate_outsider")] != 0)
    for name in st.FINAL_CONTRASTS:
        assert np.array_equal(boot[:, :, -1, st.NAMES.index(name)], np.zeros((79, 2, 2)))
    result = st.summarize(curves)
    for update in st.CHECKPOINTS[2:]:
        assert result["curves"][str(update)] == result["curves"]["1"]


def test_simultaneous_family_is_72_and_uses_both_unknown_bound_sides():
    curves = synthetic_curves(uncertain=True)
    arrays = st._arrays(curves)
    boot = st._bootstrap(arrays)
    point = np.stack([sum((a.sum(axis=0) for a in arrays[start:start+3]), np.zeros_like(arrays[0][0])) for start in (0, 3)])
    scales = st._standard_errors(arrays)
    critical = st._simultaneous_critical_value(boot, point, scales)
    per_cell = []
    for k in range(1, len(st.CHECKPOINTS)):
        for name in st.PRIMARY:
            for endpoint in (0, 1):
                col = st.NAMES.index(name)
                errors = np.abs(boot[:, 1, k, col, endpoint]-point[1, k, col, endpoint])/st.DENOMINATOR
                errors = errors/scales[k, col, endpoint] if scales[k, col, endpoint] else errors
                per_cell.append(np.quantile(errors, .95))
    assert critical >= max(per_cell)-1e-12
    result = st.summarize(curves)["simultaneous_validation"]
    assert result["family_size"] == 72 and len(result["family"]) == 72
    assert len(st.NAMES) == 20 and len(st.ARMS) == 21 and len(st.PRIMARY) == 6
    assert st.PRIMARY == tuple(st.c.CONFIG["primary_contrasts"])
    assert "0" not in result["groups"] and all(item["update"] != 0 for item in result["family"])
    assert result["centered_bound_endpoints_both_in_maximum"]
    for bands in result["groups"].values():
        for item in bands.values():
            lo, hi = item["standard_error_bounds"]
            assert item["ci95"] == [item["mean_bounds"][0]-critical*lo, item["mean_bounds"][1]+critical*hi]


def test_initial_and_nonprimary_large_deviations_do_not_enter_scan_family():
    boot = np.zeros((20, 2, len(st.CHECKPOINTS), len(st.NAMES), 2), dtype=np.int64)
    point = np.zeros_like(boot[0])
    scales = np.ones_like(point[1], dtype=np.float64)
    boot[:, 1, 0] = 100000
    boot[:, 0, 1:] = 200000
    boot[:, 1, 1:, st.NAMES.index(st.NAMES[1])] = 300000
    assert st._simultaneous_critical_value(boot, point, scales) == 0
    boot[:, 1, 1, st.NAMES.index(st.PRIMARY[0]), 1] = np.arange(20)*10
    assert st._simultaneous_critical_value(boot, point, scales) == np.quantile(np.arange(20)*10, .95)/st.DENOMINATOR


def test_fixed_standard_errors_match_hand_computed_stratified_pair_variance():
    arrays = st._arrays(synthetic_curves())
    scales = st._standard_errors(arrays)
    for k in (0, 4, 12):
        for name in st.PRIMARY:
            j = st.NAMES.index(name)
            means = [[int(value)/(4*256) for value in matrix[:, k, j, 0]] for matrix in arrays[3:]]
            variance = sum(sum((value-sum(cell)/14)**2 for value in cell)/13 for cell in means)/(9*14)
            assert abs(scales[k, j, 0]-math.sqrt(variance)) < 1e-12


def test_standardized_bands_are_equivariant_to_endpoint_scale():
    arrays = st._arrays(synthetic_curves())
    boot = st._bootstrap(arrays)
    point = np.stack([sum((a.sum(axis=0) for a in arrays[start:start+3]), np.zeros_like(arrays[0][0])) for start in (0, 3)])
    scales = st._standard_errors(arrays)
    critical = st._simultaneous_critical_value(boot, point, scales)
    original = st._simultaneous(point, scales, critical)
    # Magnify only the outsider endpoint. Its standard error and interval should
    # scale, without widening the separate supported-planning endpoint's band.
    j = st.NAMES.index(st.PRIMARY[0])
    enlarged = [array.copy() for array in arrays]
    for array in enlarged:
        array[:, :, j] *= 7
    bigger_boot = st._bootstrap(enlarged)
    bigger_point = point.copy(); bigger_point[:, :, j] *= 7
    bigger_scales = st._standard_errors(enlarged)
    bigger_critical = st._simultaneous_critical_value(bigger_boot, bigger_point, bigger_scales)
    changed = st._simultaneous(bigger_point, bigger_scales, bigger_critical)
    assert abs(critical-bigger_critical) < 1e-12
    for update in st.CHECKPOINTS[1:]:
        for name in st.PRIMARY:
            ratio = 7 if name == st.PRIMARY[0] else 1
            left = original["groups"][str(update)][name]["ci95"]
            right = changed["groups"][str(update)][name]["ci95"]
            assert np.allclose(np.array(left)*ratio, right, rtol=0, atol=1e-12)


def test_zero_se_cells_have_zero_deviation_or_stop_without_arbitrary_floor():
    boot = np.zeros((20, 2, len(st.CHECKPOINTS), len(st.NAMES), 2), dtype=np.int64)
    point = np.zeros_like(boot[0])
    scales = np.zeros_like(point[1], dtype=np.float64)
    assert st._simultaneous_critical_value(boot, point, scales) == 0
    bands = st._simultaneous(point, scales, 0)
    assert all(item["ci95"] == [0., 0.] and item["standard_error_bounds"] == [0., 0.]
               for group in bands["groups"].values() for item in group.values())
    boot[0, 1, 1, st.NAMES.index(st.PRIMARY[0]), 1] = 1
    with pytest.raises(ValueError, match="Zero-SE"):
        st._simultaneous_critical_value(boot, point, scales)


def test_both_full_paths_handle_an_entire_deterministic_zero_family():
    curves = synthetic_curves()
    for rows in curves.values():
        for row in rows:
            for new, old in ((family+"_"+role, "anchor_"+role)
                             for family in ("focused", "lowrate", "finalrate")
                             for role in ("outsider", "supported")):
                for key in ("choices", "arm_score_bounds", "arm_scores"):
                    row[key][new] = copy.deepcopy(row[key][old])
            for name, left, right in st.CONTRASTS:
                a, b = row["arm_score_bounds"][left], row["arm_score_bounds"][right]
                row["delta_bounds"][name] = [0., 0.] if row["choices"][left] == row["choices"][right] else [a[0]-b[1], a[1]-b[0]]
    direct, independent = st.summarize(curves), st.independent_summarize(curves)
    assert_close(direct, independent)
    assert direct["simultaneous_validation"]["critical_value"] == 0
    for group in direct["simultaneous_validation"]["groups"].values():
        for entry in group.values():
            assert entry["ci95"] == entry["standard_error_bounds"] == [0., 0.]


def test_final_reference_identity_produces_exact_zero_bands_even_with_unknowns():
    curves = synthetic_curves(uncertain=True)
    for function in (st.summarize, st.independent_summarize):
        result = function(curves)
        for name in st.FINAL_CONTRASTS:
            band = result["simultaneous_validation"]["groups"][str(st.CHECKPOINTS[-1])][name]
            assert band["ci95"] == band["standard_error_bounds"] == band["mean_bounds"] == [0., 0.]
            for split in st.SPLITS:
                point = result["curves"][str(st.CHECKPOINTS[-1])]["groups"][split]["contrasts"][name]
                assert point["pointwise_ci95"] == point["mean_bounds"] == [0., 0.]


@pytest.mark.parametrize("family", ["lowrate", "specialist", "finalrate"])
def test_every_new_fixed_control_must_be_invariant_across_checkpoints(family):
    curves = synthetic_curves()
    row = curves["64"][1]
    for role in ("outsider", "supported"):
        for key in ("choices", "arm_score_bounds", "arm_scores"):
            row[key][family+"_"+role] = copy.deepcopy(row[key]["anchor_"+role])
    for name, left, right in st.CONTRASTS:
        a, b = row["arm_score_bounds"][left], row["arm_score_bounds"][right]
        row["delta_bounds"][name] = [0., 0.] if row["choices"][left] == row["choices"][right] else [a[0]-b[1], a[1]-b[0]]
    for function in (st.summarize, st.independent_summarize):
        with pytest.raises(ValueError, match="[Ff]ixed control"):
            function(curves)


def test_invariant_final_control_still_must_match_actual_final_checkpoint():
    curves = synthetic_curves()
    # Change the final reference consistently at every stage: ordinary fixed
    # control checks pass, but its asserted checkpoint-1784 identity must fail.
    for rows in curves.values():
        row = rows[1]
        for role in ("outsider", "supported"):
            for key in ("choices", "arm_score_bounds", "arm_scores"):
                row[key]["finalrate_"+role] = copy.deepcopy(row[key]["anchor_"+role])
        for name, left, right in st.CONTRASTS:
            a, b = row["arm_score_bounds"][left], row["arm_score_bounds"][right]
            row["delta_bounds"][name] = [0., 0.] if row["choices"][left] == row["choices"][right] else [a[0]-b[1], a[1]-b[0]]
    for function in (st.summarize, st.independent_summarize):
        with pytest.raises(ValueError, match="[Ff]inal.*identity"):
            function(curves)


def test_whole_pair_draws_not_independent_root_or_checkpoint_resampling():
    curves = synthetic_curves()
    arrays = st._arrays(curves)
    boot = st._bootstrap(arrays)
    rng = np.random.default_rng(st.c.CONFIG["bootstrap_seed"])
    manual = np.zeros_like(boot[:17])
    for i, matrix in enumerate(arrays):
        for_draws = rng.integers(0, 14, size=(17, 14))
        for replicate, selected in enumerate(for_draws):
            for pair in selected:
                manual[replicate, i//3] += matrix[pair]
    assert np.array_equal(boot[:17], manual)


@pytest.mark.parametrize("damage", ["missing_checkpoint", "missing_root", "split", "pair", "phase", "support", "choice", "delta", "unknown", "fixed_control"])
def test_both_paths_reject_incomplete_or_tampered_panel(damage):
    curves = synthetic_curves()
    if damage == "missing_checkpoint":
        del curves["64"]
    elif damage == "missing_root":
        curves["0"].pop()
    else:
        for rows in curves.values():
            row = rows[1]
            if damage == "split": row["split"] = "validation"
            elif damage == "pair": row["pair"] = 999
            elif damage == "phase": row["phase"] = rows[0]["phase"]
            elif damage == "support": row["support"] = 9
            elif damage == "choice": row["choices"]["focused_outsider"] = 2
            elif damage == "delta": row["delta_bounds"][st.PRIMARY[0]][0] += 1/256
            elif damage == "unknown": row["unknown_counts"][3] = 1
        if damage == "fixed_control":
            curves["16"][1]["arm_scores"]["teacher"] += 1/256
    for function in (st.summarize, st.independent_summarize):
        with pytest.raises((ValueError, AssertionError)):
            function(curves)


def test_budget_callback_interrupts_bootstrap_and_summary():
    curves = synthetic_curves()
    class Budget(Exception):
        pass
    for function in (st.summarize, st.independent_summarize):
        seen = []
        def boundary():
            seen.append(1)
            if len(seen) == 15:
                raise Budget()
        with pytest.raises(Budget):
            function(curves, boundary=boundary)
        assert len(seen) == 15


def test_independent_path_does_not_call_collector_arithmetic(monkeypatch):
    curves = synthetic_curves()
    expected = st.summarize(curves)
    def forbidden(*args, **kwargs):
        pytest.fail("Collector statistical implementation reused")
    for name in ("_arrays", "_checked_row", "_units", "_bootstrap", "_entry", "_simultaneous", "_standard_errors", "_simultaneous_critical_value"):
        monkeypatch.setattr(st, name, forbidden)
    assert_close(st.independent_summarize(curves), expected)
