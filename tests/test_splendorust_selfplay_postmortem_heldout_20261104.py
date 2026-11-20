"""Model-free counterexamples for paired saved-prediction analysis."""
from copy import deepcopy
import math

import numpy as np
import pytest

from tools.splendorust_selfplay_postmortem_20261104 import heldout as h


def row(game=192, ply=0, *, target=(1., -1.), output=(.5, -.5)):
    return {"game_index": game, "ply": ply, "split": "validation", "root_input_sha256": "a"*64,
            "legal_mask81": [True, True]+[False]*79,
            "policy_target81": [.75, .25]+[0.]*79,
            "value_target2": list(target), "output83": [0., 0.]+[1000.]*79+list(output)}


def test_masked_two_action_and_signed_two_head_losses():
    loss = h.prediction_losses(row())
    assert loss == {"policy": math.log(2), "value": .25, "combined": math.log(2)+.25}


def test_policy_improves_when_probability_moves_toward_teacher():
    a, b = row(), row()
    b["output83"][0] = 1.
    assert h.prediction_losses(b)["policy"] < h.prediction_losses(a)["policy"]
    assert h.prediction_losses(b)["value"] == h.prediction_losses(a)["value"]


def test_canonical_slot_permutation_preserves_value_mse():
    a = row(output=(.75, -.25)); b = row(target=(-1., 1.), output=(-.25, .75))
    assert h.prediction_losses(a)["value"] == h.prediction_losses(b)["value"]
    b["value_target2"] = [1., -1.]
    assert h.prediction_losses(b)["value"] > h.prediction_losses(a)["value"]


@pytest.mark.parametrize("mutation", [
    lambda r:r["legal_mask81"].__setitem__(0, 1),
    lambda r:r.__setitem__("legal_mask81", [False]*81),
    lambda r:r["policy_target81"].__setitem__(2, .1),
    lambda r:r["policy_target81"].__setitem__(0, .7),
    lambda r:r["policy_target81"].__setitem__(0, float("nan")),
    lambda r:r.__setitem__("value_target2", [1., 1.]),
    lambda r:r.__setitem__("value_target2", [True, -1.]),
    lambda r:r["output83"].__setitem__(82, float("inf")),
    lambda r:r["output83"].__setitem__(81, .1),
    lambda r:r.__setitem__("root_input_sha256", "unbound"),
    lambda r:r.__setitem__("game_index", True),
    lambda r:r.__setitem__("invented", 1),
])
def test_reject_malformed_prediction(mutation):
    r = row(); mutation(r)
    with pytest.raises(ValueError): h.prediction_losses(r)


@pytest.mark.parametrize("field,value", [
    ("game_index", 193), ("ply", 1), ("root_input_sha256", "b"*64),
    ("policy_target81", [.25, .75]+[0.]*79), ("value_target2", [-1., 1.]),
    ("legal_mask81", [True]*81), ("split", "offline"),
])
def test_pair_requires_same_inputs_and_targets(field, value):
    a, b = row(), row(); b[field] = value
    with pytest.raises(ValueError): h.paired_rows([a], [b])


def test_reject_duplicate_or_reordered_rows():
    a, b = row(ply=1), row(ply=2)
    for rows in [[a, a], [b, a]]:
        with pytest.raises(ValueError): h.paired_rows(rows, deepcopy(rows))


def test_independent_game_summary_keeps_equal_game_weight():
    low = row(output=(1., -1.)); high = row(game=193, output=(-1., 1.))
    rows = [low]+[dict(high, ply=i) for i in range(3)]
    summary = h.game_summary(rows, [192, 193])
    assert summary["mean_game_losses"]["value"] == 2.
    assert summary["root_count"] == 4
    with pytest.raises(ValueError): h.game_summary(rows, [192, 193, 194])


def test_resampling_unit_is_whole_game_and_metrics_stay_correlated():
    data = np.array([[-1., 1., 0.], [3., -3., 0.]])
    actual, digest = h.bootstrap_game_means(data, np.random.Generator(np.random.PCG64(77)), 1025)
    independent_rng = np.random.Generator(np.random.PCG64(77))
    indices = independent_rng.integers(0, 2, size=(1025, 2), dtype=np.int64)
    expected = np.array([[math.fsum(data[i, j] for i in ix)/2 for j in range(3)] for ix in indices])
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[:, 0], -actual[:, 1])
    assert np.all(actual[:, 2] == 0)
    assert len(digest) == 64


def test_weighting_signs_and_leave_one_out_are_explicit():
    a = [{"game_index": 0, "roots": 1, "policy": 2., "value": 2., "combined": 4.},
         {"game_index": 1, "roots": 100, "policy": 2., "value": 2., "combined": 4.}]
    b = [dict(a[0], policy=1., combined=3.), dict(a[1], policy=5., combined=7.)]
    draws, _ = h.bootstrap_game_means([[-1., 0., -1.], [3., 0., 3.]],
                                      np.random.Generator(np.random.PCG64(4)), 1000)
    report = h.describe_game_differences(a, b, draws)["losses"]["policy"]
    assert report["candidate_minus_parent_game_mean"] == 1.
    assert report["median_game_delta"] == 1.
    assert report["games_improved"] == report["games_worsened"] == 1
    assert report["games_equal"] == 0
    assert report["root_weighted_mean_delta"] == 299/101
    assert report["leave_one_game_out_mean_range"] == [-1., 3.]


def test_degenerate_interval_is_disclosed_and_absent_sign_is_null():
    a = [{"game_index": i, "roots": 1, "policy": 1., "value": 1., "combined": 2.} for i in range(3)]
    draws, _ = h.bootstrap_game_means(np.zeros((3, 3)), np.random.Generator(np.random.PCG64(2)), 10)
    result = h.describe_game_differences(a, deepcopy(a), draws)["losses"]["value"]
    assert result["nominal_paired_game_bootstrap_ci95"] == [0., 0.]
    assert result["bootstrap_degenerate"] is True
    assert result["games_equal"] == 3
    bias = h.value_bias([row()], [row()])["strata"]
    assert bias["0"]["0"]["games_contributing"] == 0
    assert bias["0"]["0"]["parent_mean_game_signed_residual"] is None
    assert bias["0"]["1"]["parent_mean_game_signed_residual"] == -.5
    assert bias["1"]["-1"]["parent_mean_game_signed_residual"] == .5


def test_value_bias_conditions_within_game_before_equal_game_average():
    a = [row(output=(0., 0.))]+[row(game=193, ply=i, output=(1., -1.)) for i in range(4)]
    b = [row(output=(.5, -.5))]+[row(game=193, ply=i, output=(1., -1.)) for i in range(4)]
    stratum = h.value_bias(a, b)["strata"]["0"]["1"]
    assert stratum["games_contributing"] == 2 and stratum["roots"] == 5
    assert stratum["parent_mean_game_signed_residual"] == -.5
    assert stratum["candidate_mean_game_signed_residual"] == -.25
    assert stratum["candidate_minus_parent_mean_game_signed_residual"] == .25


@pytest.mark.parametrize("data", [np.zeros((1, 3)), np.zeros((2, 2)), [[1., 2., float('nan')]]*2])
def test_bootstrap_rejects_missing_games_metrics_or_nonfinite(data):
    with pytest.raises(ValueError): h.bootstrap_game_means(data, np.random.Generator(np.random.PCG64(1)), 5)
