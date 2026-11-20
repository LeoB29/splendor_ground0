"""Joint gate, independent reductions, independent units and honest unknowns."""
import copy
import pytest
from . import common as c, statistics as s, verify as v


def rows(large=3, small=0, control=0, unknown=False, vary=True):
    result = []
    for fit in c.FITS:
        for seat in (0, 1):
            for phase in (0, 1, 2):
                for j in range(40):
                    i = len(result)
                    anchor = 128 + (j % 3 - 1 if vary else 0)
                    values = {"anchor": anchor, "small": anchor+small, "medium": anchor+1,
                              "large": anchor+large, "control": anchor+control, "teacher": anchor+6}
                    bounds = {a: [x/256, (x+int(unknown and j%7 == 0))/256] for a,x in values.items()}
                    delta = {name: [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
                             for name in c.CONTRASTS for left,right in [name.split("_minus_")]}
                    result.append({"index":i, "game_index":i, "pair":i, "fit":fit, "seat":seat, "phase":phase,
                                   "arm_score_bounds":bounds, "delta_bounds":delta,
                                   "arm_scores":{a:pair[0] if pair[0] == pair[1] else None for a,pair in bounds.items()},
                                   "choices":dict(zip(c.ARMS,range(6))), "privacy":"neither"})
    return result


@pytest.mark.parametrize("unknown", [False, True])
def test_independent_arithmetic_agrees_for_complete_and_unknown_outcomes(unknown):
    data = rows(unknown=unknown)
    left,right = s.summarize(data),v.statistics(data)
    for name, value in left["contrasts"].items():
        for key in ("mean_score_difference", "mean_bounds", "ci95"):
            v.compare(value[key],right["contrasts"][name][key])
    for key in ("anchor_practical_gain", "anchor_superiority", "small_superiority",
                "coverage_attribution", "gate_pass", "classification", "unknown_outcomes_imputed", "per_fit"):
        v.compare(left[key],right[key])
    assert left["gate_pass"] and not left["unknown_outcomes_imputed"]
    assert set(left["per_fit"]) == set(map(str,c.FITS))


def test_shared_training_gain_does_not_establish_large():
    result = s.summarize(rows(large=3,small=3,vary=False))
    assert result["anchor_practical_gain"] and result["anchor_superiority"]
    assert not result["small_superiority"] and not result["coverage_attribution"] and not result["gate_pass"]
    assert result["contrasts"]["large_minus_small"]["ci95"] == [0.,0.]


def test_small_positive_gain_over_both_controls_is_not_material_pass():
    result = s.summarize(rows(large=1,small=-2,vary=False))
    assert result["coverage_attribution"] and result["anchor_superiority"]
    assert not result["anchor_practical_gain"] and not result["gate_pass"]


def test_saved_short_dose_control_is_descriptive_not_a_primary_gate(monkeypatch):
    monkeypatch.setitem(c.CONFIG,"bootstrap_samples",37)
    data = rows(large=3,small=0,control=20,vary=False)
    direct,independent = s.summarize(data),v.statistics(data)
    assert direct["gate_pass"] and independent["gate_pass"]
    assert direct["contrasts"]["large_minus_control"]["ci95"][1] < 0
    assert direct["contrasts"]["large_minus_small"]["ci95"][0] > 0


@pytest.mark.parametrize("fault", ["duplicate_game", "missing", "wrong_cell", "fraction", "uncoupled", "same_choice"])
def test_reject_invalid_units_and_inconsistent_bounds(fault):
    data = copy.deepcopy(rows())
    if fault == "duplicate_game": data[0]["game_index"] = data[1]["game_index"]
    elif fault == "missing": data.pop()
    elif fault == "wrong_cell": data[0]["fit"] = 161002
    elif fault == "fraction": data[0]["arm_score_bounds"]["large"][1] = .511
    elif fault == "uncoupled": data[0]["delta_bounds"]["large_minus_anchor"] = [0.,0.]
    else: data[0]["choices"]["large"] = data[0]["choices"]["anchor"]
    with pytest.raises(ValueError): s.summarize(data)
