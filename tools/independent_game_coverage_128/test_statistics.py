"""Independent statistics, coupled bounds, fixed units and separate claims."""
import copy
import pytest
from . import statistics as s, verify as v
from .test_verify import statistic_rows


@pytest.mark.parametrize("unknown", [0, 1, 3])
def test_two_independent_implementations_match_exact_and_censored_rows(unknown):
    rows = statistic_rows(unknown=unknown)
    first, second = s.summarize(rows), v.statistics(rows)
    for name, value in first["contrasts"].items():
        assert value["mean_score_difference"] == second["contrasts"][name]["mean_score_difference"]
        assert value["mean_bounds"] == second["contrasts"][name]["mean_bounds"]
        assert value["ci95"] == pytest.approx(second["contrasts"][name]["ci95"], abs=1e-12)
    for key in ("anchor_practical_gain", "anchor_superiority", "lowrate_superiority", "breadth_attribution",
                "gate_pass", "classification", "unknown_outcomes_imputed"):
        assert first[key] == second[key]
    assert not first["unknown_outcomes_imputed"]


def test_shared_package_improvement_does_not_establish_breadth():
    result = s.summarize(statistic_rows(broad=3, narrow=3, vary=False))
    assert result["gate_pass"] and not result["breadth_attribution"]
    assert result["contrasts"]["broad_minus_narrow"]["ci95"] == [0., 0.]


def test_breadth_without_practical_progress_is_not_a_pass():
    result = s.summarize(statistic_rows(broad=1, narrow=-2, vary=False))
    assert result["breadth_attribution"] and not result["anchor_practical_gain"]
    assert not result["gate_pass"]


@pytest.mark.parametrize("fault", ["duplicate_game", "missing_row", "wrong_cell", "nonquantized", "uncoupled", "same_choice_different_bound"])
def test_wrong_statistical_units_or_bounds_are_rejected(fault):
    rows = copy.deepcopy(statistic_rows())
    if fault == "duplicate_game": rows[0]["game_index"] = rows[1]["game_index"]
    elif fault == "missing_row": rows.pop()
    elif fault == "wrong_cell": rows[0]["fit"] = 161002
    elif fault == "nonquantized": rows[0]["arm_score_bounds"]["broad"][1] = .511
    elif fault == "uncoupled": rows[0]["delta_bounds"]["broad_minus_anchor"] = [0., 0.]
    else: rows[0]["choices"]["broad"] = rows[0]["choices"]["anchor"]
    with pytest.raises(ValueError): s.summarize(rows)
