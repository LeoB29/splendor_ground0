"""Synthetic independent-audit regressions; never read production root outcomes."""
import copy
import itertools

import numpy as np
import pytest

from . import analysis, common as c, verify as v


def _outcome(value, actor=0):
    return {"wdl": value, "reason": "repetition_cutoff" if value is None else "completed",
            "winner": None if value in (None, 0) else actor if value == 1 else 1-actor}


def _block(totals, first, actor=0):
    return {"candidate_indices": list(range(len(totals))), "world_seeds": list(range(first, first+128)),
            "worlds": [{"seed": first+w, "state_sha256": "synthetic-state",
                        "outcomes": [_outcome((1 if total > 0 else -1) if w < abs(total) else 0, actor)
                                     for total in totals]} for w in range(128)]}


def _seal_fixture(record):
    """Construct source metadata for synthetic input, without the new analyzers."""
    width = len(record["union"]["keys"])
    columns = [[world["outcomes"][j]["wdl"] for world in record["selection"]["worlds"]] for j in range(width)]
    totals = [sum(value for value in col if value is not None) for col in columns]
    missing = [col.count(None) for col in columns]
    unknown = [{"world": w, "candidate": j} for w in range(128) for j in range(width) if columns[j][w] is None]
    choices = {arm: max(data["indices"], key=lambda j: totals[j]) for arm, data in record["candidates"].items()}
    certs = {}
    for arm, choice in choices.items():
        indices = record["candidates"][arm]["indices"]
        margins = [0 if other == choice else totals[choice]-totals[other]-missing[choice]-missing[other] for other in indices]
        certs[arm] = {"indices": indices, "choice": choice, "pairwise_lower_numerators": margins}
    selection_stats = {"means": [None if missing[j] else totals[j]/128 for j in range(width)],
                       "means_bounds": [[(totals[j]-missing[j])/128, (totals[j]+missing[j])/128] for j in range(width)],
                       "sums_wdl": totals, "unknown_counts": missing, "unresolved_cells": unknown,
                       "choices": choices,
                       "certificate": {"rule": "bounded_repetition_original_order_v1",
                                       "unknown_groups": [{"context_sha256": None, "cells": [cell]} for cell in unknown],
                                       "arm_certificates": certs}}
    lock = record["selection_lock"]
    lock.update(choices=choices, selection_statistics=selection_stats, selection_sha256=c.digest(record["selection"]),
                assessment_indices=sorted(set(choices.values())))
    assessed = record["assessment"]["candidate_indices"]
    scores = {}
    for j, candidate in enumerate(assessed):
        values = [world["outcomes"][j]["wdl"] for world in record["assessment"]["worlds"]]
        total, absent = sum(value for value in values if value is not None), values.count(None)
        scores[candidate] = [(128+total-absent)/256, (128+total+absent)/256]
    arm_scores = {arm: scores[choice] for arm, choice in choices.items()}
    deltas = {name: [0., 0.] if choices[left] == choices[right] else [
        arm_scores[left][0]-arm_scores[right][1], arm_scores[left][1]-arm_scores[right][0]]
              for name, left, right in v.CONTRASTS[4:]}
    record["summary"] = {"choices": choices, "arm_score_bounds": arm_scores, "delta_bounds": deltas}


def fixture(selection_totals=None, assessment_totals=None, size=10, actor=0):
    totals = selection_totals if selection_totals is not None else [-64, -32, -48, 32, 64, 48, 16, 0, -8, -16]
    offers = ({arm: list(range(size)) for arm in v.ORIGINAL} if size <= 4 else
              {"anchor": [0, 1, 2, 3], "broad": [0, 1, 2, 4], "narrow": [0, 1, 2, 5],
               "teacher": [0, 1, 2, 3, 6, 7, 8, 9]})
    choices = {arm: max(ids, key=lambda i: totals[i]) for arm, ids in offers.items()}
    assessed = sorted(set(choices.values()))
    keys = [f"synthetic-{i}" for i in range(size)]
    actions = [{"fixture_action": i} for i in range(size)]
    selection = _block(totals, 2181000000, actor)
    assessment = _block(assessment_totals if assessment_totals is not None else [0]*len(assessed), 2181000256, actor)
    assessment["candidate_indices"] = assessed
    result = {"manifest": c.SOURCE_BINDING, "stage": "main",
              "spec": {"index": 0, "pair": 0, "game_index": actor, "fit": 161001, "seat": actor,
                       "phase": 0, "source_arm": "candidate", "canonical_support": size},
              "union": {"keys": keys, "actions": actions},
              "candidates": {arm: {"indices": ids, "keys": [keys[i] for i in ids], "actions": [actions[i] for i in ids]}
                             for arm, ids in offers.items()},
              "selection": selection, "assessment": assessment,
              "selection_lock": {"manifest": c.SOURCE_BINDING, "stage": "main", "index": 0,
                                 "proofs_sha256": c.digest([]), "value_scale": "terminal_wdl", "created_at_ns": 1},
              "assessment_started_at_ns": 2,
              "proposals": {arm: {"timing": {"learned_extra_wall": seconds}}
                            for arm, seconds in zip(("anchor", "broad", "narrow"), (.01, .02, .03))}}
    _seal_fixture(result)
    return result


def test_independent_exact_selection_sparse_assessment_and_cost_receipt():
    record = fixture(assessment_totals=[64, -64, -32])
    reported = analysis.analyze_root(record)
    receipt = v.verify_root(record, reported)
    assert receipt["passed"] and receipt["row"] == reported
    assert reported["choices"] == {"anchor": 3, "broad": 4, "narrow": 5, "additive_broad": 4, "additive_narrow": 5, "teacher": 3}
    assert reported["delta_bounds"]["additive_broad_minus_anchor"] == [-.5, -.5]
    assert reported["delta_bounds"]["additive_broad_minus_broad"] == [0., 0.]
    assert reported["costs"]["selection_continuations"]["additive_broad"] == 640
    assert reported["costs"]["additional_learned_extra_wall"] == {"additive_broad": .02, "additive_narrow": .03}


def test_exact_tie_preserves_anchor_even_when_assessment_favors_alternate():
    record = fixture([-64, -32, -48, 64, 64, 48, 16, 0, -8, -16], [-64, 64, 0])
    result = v.verify_root(record, analysis.analyze_root(record))["row"]
    assert result["choices"]["additive_broad"] == 3 and result["arm_score_bounds"]["additive_broad"] == [.25, .25]
    assert result["arm_score_bounds"]["broad"] == [.75, .75]


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_short_or_duplicate_support_is_charged_exactly_once(size):
    record = fixture([0]*size, [0], size=size)
    row = v.verify_root(record, analysis.analyze_root(record))["row"]
    assert set(row["choices"].values()) == {0}
    assert set(row["costs"]["candidate_counts"].values()) == {size}
    assert set(row["costs"]["extra_selection_continuations"].values()) == {0}
    assert row["costs"]["additional_learned_extra_wall"]["additive_broad"] == .02


def test_opposite_actor_is_scored_from_saved_actor():
    record = fixture(assessment_totals=[64, -64, 32], actor=1)
    row = v.verify_root(record, analysis.analyze_root(record))["row"]
    assert row["delta_bounds"]["additive_broad_minus_anchor"] == [-.5, -.5]


def test_safe_unknown_selection_and_unknown_assessment_keep_bounds_and_cancellation():
    record = fixture([-64, -32, -48, 96, 64, 48, 16, 0, -8, -16], [64, -64, -32])
    record["selection"]["worlds"][0]["outcomes"][3] = _outcome(None)
    record["assessment"]["worlds"][0]["outcomes"][0] = _outcome(None)
    _seal_fixture(record)
    row = v.verify_root(record, analysis.analyze_root(record))["row"]
    assert row["arm_score_bounds"]["anchor"] == [190/256, 192/256]
    assert row["delta_bounds"]["additive_broad_minus_anchor"] == [0., 0.]
    assert row["delta_bounds"]["additive_broad_minus_broad"] == [126/256, 128/256]


def test_unknown_union_ambiguity_is_not_rescued_by_certified_constituents():
    record = fixture([-128, -128, -128, 0, 0, -32, -64, -64, -64, -64])
    record["selection"]["worlds"][0]["outcomes"][4] = _outcome(None)
    _seal_fixture(record)
    with pytest.raises(AssertionError, match="Unresolved"):
        v.verify_root(record, {})


def test_bound_certificate_agrees_with_exhaustive_unknown_completions():
    # All +/-1 endpoints suffice for this linear ordered-argmax certificate.
    for known in itertools.product((-1, 0, 1), repeat=3):
        for unknown in ((0, 0, 0), (0, 1, 0), (1, 0, 1)):
            completions = itertools.product(*[(-count, count) if count else (0,) for count in unknown])
            winners = {max(range(3), key=lambda j: known[j]+values[j]) for values in completions}
            if len(winners) == 1:
                assert v._certified_choice([0, 1, 2], known, unknown)[0] == winners.pop()
            else:
                with pytest.raises(AssertionError):
                    v._certified_choice([0, 1, 2], known, unknown)


@pytest.mark.parametrize("field", ["choice", "score", "delta", "count", "extra", "time", "changed", "metadata"])
def test_independent_row_comparison_rejects_changed_scientific_or_cost_fields(field):
    record = fixture()
    row = analysis.analyze_root(record)
    if field == "choice": row["choices"]["additive_broad"] = 3
    if field == "score": row["arm_score_bounds"]["additive_broad"] = [0., 0.]
    if field == "delta": row["delta_bounds"]["additive_broad_minus_anchor"] = [1., 1.]
    if field == "count": row["costs"]["candidate_counts"]["additive_broad"] = 4
    if field == "extra": row["costs"]["extra_selection_continuations"]["additive_broad"] = 0
    if field == "time": row["costs"]["additional_learned_extra_wall"]["additive_broad"] = 0
    if field == "changed": row["costs"]["changed_choices"]["additive_broad"] = False
    if field == "metadata": row["pair"] = 9
    with pytest.raises(AssertionError): v.verify_root(record, row)


@pytest.mark.parametrize("change", ["proof", "seal", "certificate", "summary", "namespace", "timestamp", "selection_width",
                                   "assessment_support", "invented_draw", "wrong_actor", "reason", "missing_world", "timing", "representative"])
def test_source_tampering_and_unsupported_uncertainty_fail_closed(change):
    record = fixture()
    row = analysis.analyze_root(record)
    if change == "proof": record["equivalence_proofs"] = [{"fixture": "unsupported"}]
    if change == "seal": record["selection_lock"]["selection_sha256"] = "changed"
    if change == "certificate": record["selection_lock"]["selection_statistics"]["sums_wdl"][4] += 1
    if change == "summary": record["summary"]["arm_score_bounds"]["anchor"] = [0., 0.]
    if change == "namespace": record["assessment"]["world_seeds"] = record["selection"]["world_seeds"]
    if change == "timestamp": record["assessment_started_at_ns"] = 1
    if change == "selection_width": record["selection"]["candidate_indices"] = list(range(9))
    if change == "assessment_support": record["assessment"]["candidate_indices"] = [3, 4]
    if change == "invented_draw": record["selection"]["worlds"][0]["outcomes"][4] = {"wdl": 0, "winner": None, "reason": "repetition_cutoff"}
    if change == "wrong_actor": record["selection"]["worlds"][0]["outcomes"][4]["winner"] = 1
    if change == "reason": record["selection"]["worlds"][0]["outcomes"][4]["reason"] = "unknown_cutoff"
    if change == "missing_world": record["selection"]["worlds"].pop()
    if change == "timing": record["proposals"]["broad"]["timing"]["learned_extra_wall"] = float("nan")
    if change == "representative": record["candidates"]["broad"]["keys"][3] = "changed"
    if change in ("selection_width", "invented_draw", "wrong_actor", "reason", "missing_world"):
        record["selection_lock"]["selection_sha256"] = c.digest(record["selection"])
    with pytest.raises(AssertionError): v.verify_root(record, row)


def _refresh_deltas(row):
    scores, choices = row["arm_score_bounds"], row["choices"]
    row["delta_bounds"] = {name: [0., 0.] if choices[left] == choices[right] else [
        scores[left][0]-scores[right][1], scores[left][1]-scores[right][0]] for name, left, right in v.CONTRASTS}
    row["costs"]["changed_choices"] = {arm: choices[arm] != choices["anchor"] for arm in v.ARMS}


def panel(gap_units=4, matched=False, variable=False, unknown=False):
    rows = []
    for cell, (fit, phase, source) in enumerate(itertools.product((161001, 161002, 161003), range(3), ("candidate", "reference"))):
        for number in range(10):
            pair = cell*10+number
            gap = gap_units+(number if variable else 0)
            for seat in (0, 1):
                choices = {"anchor": 0, "broad": 1, "narrow": 1 if matched else 0,
                           "additive_broad": seat, "additive_narrow": seat if matched else 0, "teacher": 0}
                scores = {0: [(128+(gap if seat == 0 else 0))/256]*2,
                          1: [(128+(gap if seat == 1 else 0))/256]*2}
                if unknown: scores[1] = [.5, .5+8/256]
                counts = {"anchor": 4, "broad": 4, "narrow": 4, "additive_broad": 5, "additive_narrow": 5, "teacher": 8}
                costs = {"candidate_counts": counts, "selection_continuations": {arm: 128*n for arm, n in counts.items()},
                         "extra_candidates": {"additive_broad": 1, "additive_narrow": 1},
                         "extra_selection_continuations": {"additive_broad": 128, "additive_narrow": 128},
                         "learned_extra_wall": {"anchor": .01, "broad": .02, "narrow": .03},
                         "additional_learned_extra_wall": {"additive_broad": .02, "additive_narrow": .03}}
                row = {"index": len(rows), "pair": pair, "game_index": 2*pair+seat, "fit": fit, "seat": seat,
                       "phase": phase, "source_arm": source, "choices": choices,
                       "arm_score_bounds": {arm: list(scores[choice]) for arm, choice in choices.items()}, "costs": costs}
                _refresh_deltas(row)
                rows.append(row)
    return rows


@pytest.mark.parametrize("gap,matched,expected", [(4, False, "additive_broad_practical_and_attribution_signal"),
                                                 (4, True, "additive_broad_practical_signal_only"),
                                                 (2, False, "additive_broad_not_established"),
                                                 (0, False, "additive_broad_not_established")])
def test_independent_statistics_gates_and_pair_closed_form(gap, matched, expected):
    rows = panel(gap, matched)
    reported = analysis.summarize(rows)
    result = v.verify_statistics(rows, reported)["statistics"]
    assert result["classification"] == expected
    assert result["pairs"] == 180 and result["strata"] == 18
    for name, _, _ in v.CONTRASTS[:2]:
        assert result["contrasts"][name] == {"mean": gap/512, "mean_bounds": [gap/512]*2, "ci95": [gap/512]*2}
    assert result["aggregate_costs"]["changed_choices"]["additive_broad"] == 180
    assert result["aggregate_costs"]["selection_cost_ratio_to_anchor"]["additive_broad"] == 1.25
    assert result["aggregate_costs"]["selection_continuations"]["teacher"] == 360*1024


def test_shared_draw_multiplicity_bootstrap_matches_primary_on_nonconstant_pairs():
    rows = panel(variable=True)
    expected = analysis.summarize(rows)
    actual = v.verify_statistics(list(reversed(rows)), expected)["statistics"]
    interval = actual["contrasts"]["additive_broad_minus_anchor"]["ci95"]
    assert interval[0] < 8.5/512 < interval[1]


def test_statistics_unknown_bounds_do_not_acquire_point_means():
    rows = panel(gap_units=0, unknown=True)
    result = v.verify_statistics(rows, analysis.summarize(rows))["statistics"]
    assert result["contrasts"]["additive_broad_minus_anchor"] == {"mean": None, "mean_bounds": [0., 4/256], "ci95": [0., 4/256]}
    assert not result["practical_pass"] and result["unknown_outcomes_imputed"] is False


@pytest.mark.parametrize("change", ["roots", "duplicate_index", "duplicate_game", "pair", "seat", "stratum", "contrast", "score",
                                   "same_action", "ancestry", "candidate_cost", "extra_cost", "elapsed", "changed_bool"])
def test_statistics_reject_population_score_or_cost_corruption_before_resampling(change):
    rows = panel()
    row = rows[-1]
    if change == "roots": rows.pop()
    if change == "duplicate_index": row["index"] = 0
    if change == "duplicate_game": row["game_index"] = 0
    if change == "pair": row["pair"] = 0
    if change == "seat": row["seat"] = 0
    if change == "stratum": row["phase"] = 0
    if change == "contrast": row["delta_bounds"]["additive_broad_minus_anchor"] = [1., 1.]
    if change == "score": row["arm_score_bounds"]["broad"] = [.501, .501]
    if change == "same_action": row["arm_score_bounds"]["teacher"] = [0., 0.]
    if change == "ancestry": row["choices"]["additive_broad"] = 9
    if change == "candidate_cost": row["costs"]["candidate_counts"]["broad"] = 3
    if change == "extra_cost": row["costs"]["extra_selection_continuations"]["additive_broad"] = 0
    if change == "elapsed": row["costs"]["additional_learned_extra_wall"]["additive_broad"] = 0
    if change == "changed_bool": row["costs"]["changed_choices"]["anchor"] = 0
    with pytest.raises(AssertionError): v.independent_statistics(rows)


def test_final_comparison_rejects_a_changed_classification_or_cost():
    rows = panel()
    reported = analysis.summarize(rows)
    reported["classification"] = "additive_broad_not_established"
    with pytest.raises(AssertionError, match="classification"):
        v.verify_statistics(rows, reported)
    reported = analysis.summarize(rows)
    reported["aggregate_costs"]["selection_cost_ratio_to_anchor"]["additive_broad"] = 1
    with pytest.raises(AssertionError, match="selection_cost_ratio"):
        v.verify_statistics(rows, reported)


def test_budget_can_interrupt_independent_bootstrap():
    calls = 0
    def budget():
        nonlocal calls
        calls += 1
        if calls == 3: raise RuntimeError("synthetic cap")
    with pytest.raises(RuntimeError, match="synthetic cap"):
        v.verify_statistics(panel(), {}, budget)
    assert calls == 3
