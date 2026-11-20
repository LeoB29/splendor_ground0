"""Synthetic-only exact union, held-out scoring, cost and paired-bootstrap tests."""
import copy

import pytest

from . import analysis as p
from . import common as c


def block(sums, first_seed, actor=0):
    worlds = []
    for wi in range(128):
        outcomes = []
        for total in sums:
            value = (1 if total > 0 else -1) if wi < abs(total) else 0
            winner = None if value == 0 else actor if value == 1 else 1-actor
            outcomes.append({"reason": "completed", "winner": winner, "wdl": value})
        worlds.append({"seed": first_seed+wi, "state_sha256": f"fixture-{wi}", "outcomes": outcomes})
    return {"world_seeds": list(range(first_seed, first_seed+128)), "worlds": worlds,
            "candidate_indices": list(range(len(sums)))}


def source_root(selection_sums=None, assessment_sums=None, offers=None):
    sums = selection_sums or [-64, -32, -48, 32, 64, 48, 16, 0, -8, -16]
    support = len(sums)
    if offers is None:
        offers = {"anchor": [0, 1, 2, 3], "broad": [0, 1, 2, 4], "narrow": [0, 1, 2, 5],
                  "teacher": [0, 1, 2, 3, 6, 7, 8, 9]}
    choices = {arm: max(indices, key=lambda i: sums[i]) for arm, indices in offers.items()}
    indices = sorted(set(choices.values()))
    keys, actions = [f"k{i}" for i in range(support)], [{"fixture_action": i} for i in range(support)]
    selection = block(sums, 2181000000)
    assessment = block(assessment_sums if assessment_sums is not None else [0]*len(indices), 2181000256)
    assessment["candidate_indices"] = indices
    return {"spec": {"index": 0, "pair": 0, "game_index": 0, "fit": 161001, "seat": 0, "phase": 0, "source_arm": "candidate"},
            "union": {"keys": keys, "actions": actions},
            "candidates": {arm: {"indices": v, "keys": [keys[i] for i in v], "actions": [actions[i] for i in v]}
                           for arm, v in offers.items()},
            "selection": selection, "assessment": assessment,
            "selection_lock": {"choices": choices, "selection_statistics": {"choices": choices},
                               "selection_sha256": c.digest(selection), "proofs_sha256": c.digest([]),
                               "assessment_indices": indices, "created_at_ns": 1},
            "assessment_started_at_ns": 2,
            "proposals": {arm: {"timing": {"learned_extra_wall": .01*(i+1)}}
                          for i, arm in enumerate(("anchor", "broad", "narrow"))}}


def seal_selection(record):
    record["selection_lock"]["selection_sha256"] = c.digest(record["selection"])


def test_append_keeps_old_winner_and_scores_only_held_out_worlds():
    record = source_root([-64, -32, -48, 96, 64, 48, 16, 0, -8, -16], [64, -64, -32])
    row = p.analyze_root(record)
    assert row["choices"]["additive_broad"] == row["choices"]["anchor"] == 3
    assert row["choices"]["broad"] == 4
    assert row["delta_bounds"]["additive_broad_minus_anchor"] == [0., 0.]
    assert row["delta_bounds"]["additive_broad_minus_broad"] == [.5, .5]
    assert row["costs"]["extra_candidates"] == {"additive_broad": 1, "additive_narrow": 1}
    assert row["costs"]["selection_continuations"]["additive_broad"] == 640
    assert row["costs"]["additional_learned_extra_wall"] == {"additive_broad": .02, "additive_narrow": .03}


def test_union_can_select_new_action_and_lose_on_independent_assessment():
    record = source_root(assessment_sums=[64, -64, 0])
    row = p.analyze_root(record)
    assert row["choices"]["additive_broad"] == row["choices"]["broad"] == 4
    assert row["delta_bounds"]["additive_broad_minus_anchor"] == [-.5, -.5]
    changed = copy.deepcopy(record)
    for world in changed["assessment"]["worlds"]:
        for outcome in world["outcomes"]:
            outcome["wdl"] *= -1
            if outcome["winner"] is not None: outcome["winner"] = 1-outcome["winner"]
    assert p.analyze_root(changed)["choices"] == row["choices"]
    assert p.analyze_root(changed)["delta_bounds"]["additive_broad_minus_anchor"] == [.5, .5]


def test_exact_mean_tie_retains_anchor_instead_of_assessment_favorite():
    record = source_root([-64, -32, -48, 64, 64, 48, 16, 0, -8, -16], [-64, 64, 0])
    row = p.analyze_root(record)
    assert row["choices"]["additive_broad"] == 3
    assert row["arm_score_bounds"]["additive_broad"] == [.25, .25]
    assert row["arm_score_bounds"]["broad"] == [.75, .75]


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_duplicate_proposals_and_short_support_add_no_cost_or_effect(size):
    record = source_root([0]*size, [0], {arm: list(range(size)) for arm in p.ORIGINAL})
    row = p.analyze_root(record)
    assert set(row["choices"].values()) == {0}
    assert all(bound == [0., 0.] for bound in row["delta_bounds"].values())
    assert row["costs"]["extra_selection_continuations"] == {"additive_broad": 0, "additive_narrow": 0}


def test_unresolved_selection_and_assessment_are_bounded_without_imputation():
    record = source_root([-64, -32, -48, 96, 64, 48, 16, 0, -8, -16], [64, -64, -32])
    record["selection"]["worlds"][0]["outcomes"][3] = {"reason": "repetition_cutoff", "winner": None, "wdl": None}
    record["assessment"]["worlds"][0]["outcomes"][0] = {"reason": "repetition_cutoff", "winner": None, "wdl": None}
    seal_selection(record)
    row = p.analyze_root(record)
    assert row["arm_score_bounds"]["anchor"] == [190/256, 192/256]
    assert row["delta_bounds"]["additive_broad_minus_anchor"] == [0., 0.]
    assert row["delta_bounds"]["additive_broad_minus_broad"] == [126/256, 128/256]


def test_individually_certified_arms_can_still_have_ambiguous_union():
    record = source_root([-128, -128, -128, 0, 0, -32, -64, -64, -64, -64])
    record["selection"]["worlds"][0]["outcomes"][4] = {"reason": "repetition_cutoff", "winner": None, "wdl": None}
    seal_selection(record)
    with pytest.raises(ValueError, match="can change"):
        p.analyze_root(record)


@pytest.mark.parametrize("change", ["missing_choice", "wrong_original", "selection_hash", "overlap", "reordered_worlds",
                                   "wrong_actor", "invented_draw", "other_cutoff", "missing_world", "missing_outcome",
                                   "changed_representative", "shared_three", "coupled_proof", "negative_time"])
def test_corruption_and_unsupported_proofs_fail_closed(change):
    record = source_root()
    if change == "missing_choice": record["assessment"]["candidate_indices"] = [3, 5]
    if change == "wrong_original": record["selection_lock"]["choices"]["broad"] = 3
    if change == "selection_hash": record["selection_lock"]["selection_sha256"] = "changed"
    if change == "overlap": record["assessment"]["world_seeds"] = record["selection"]["world_seeds"]
    if change == "reordered_worlds": record["selection"]["worlds"].reverse(); seal_selection(record)
    if change == "wrong_actor": record["selection"]["worlds"][0]["outcomes"][3]["winner"] = 1; seal_selection(record)
    if change == "invented_draw": record["selection"]["worlds"][0]["outcomes"][3] = {"reason": "repetition_cutoff", "winner": None, "wdl": 0}; seal_selection(record)
    if change == "other_cutoff": record["selection"]["worlds"][0]["outcomes"][3]["reason"] = "no_progress_cutoff"; seal_selection(record)
    if change == "missing_world": record["selection"]["worlds"].pop(); seal_selection(record)
    if change == "missing_outcome": record["selection"]["worlds"][0]["outcomes"].pop(); seal_selection(record)
    if change == "changed_representative": record["candidates"]["broad"]["actions"][3] = {"fixture_action": 99}
    if change == "shared_three": record["candidates"]["broad"] = copy.deepcopy(record["candidates"]["teacher"])
    if change == "coupled_proof": record["equivalence_proofs"] = [{"fixture": "unsupported"}]
    if change == "negative_time": record["proposals"]["broad"]["timing"]["learned_extra_wall"] = -1
    with pytest.raises(ValueError):
        p.analyze_root(record)


def panel_rows(gap=4/256):
    base = p.analyze_root(source_root())
    rows, pair = [], 0
    for fit in c.CONFIG["fits"]:
        for phase in range(3):
            for source_arm in ("candidate", "reference"):
                for _ in range(10):
                    for seat in (0, 1):
                        row = copy.deepcopy(base)
                        row.update(index=len(rows), pair=pair, game_index=pair*2+seat, fit=fit, phase=phase, source_arm=source_arm, seat=seat)
                        row["choices"] = {"anchor": 0, "broad": 1, "narrow": 0, "additive_broad": seat,
                                          "additive_narrow": 0, "teacher": 0}
                        values = {0: .5+(gap if seat == 0 else 0), 1: .5+(gap if seat == 1 else 0)}
                        row["arm_score_bounds"] = {arm: [values[index]]*2 for arm, index in row["choices"].items()}
                        row["delta_bounds"] = {name: [values[row["choices"][left]]-values[row["choices"][right]]]*2
                                               for name, (left, right) in p.CONTRASTS.items()}
                        row["costs"]["changed_choices"] = {arm: index != 0 for arm, index in row["choices"].items()}
                        rows.append(row)
                    pair += 1
    return rows


def test_both_original_comparators_and_attribution_gates_use_exact_pair_arithmetic():
    rows = panel_rows()
    value = p.summarize(rows)
    assert value["practical_pass"] and value["attribution_pass"]
    for name in list(p.CONTRASTS)[:3]:
        assert value["contrasts"][name] == {"mean": 2/256, "mean_bounds": [2/256]*2, "ci95": [2/256]*2}
    assert value["aggregate_costs"]["extra_selection_continuations"]["additive_broad"] == 360*128
    assert value["aggregate_costs"]["changed_choices"]["additive_broad"] == 180
    assert value["aggregate_costs"]["selection_cost_ratio_to_anchor"]["additive_broad"] == 1.25
    assert p.summarize(list(reversed(rows))) == value


def test_fixed_practical_threshold_and_replacement_gate_cannot_be_rescued_by_teacher():
    assert not p.summarize(panel_rows(gap=2/256))["practical_pass"]
    rows = panel_rows()
    for row in rows:
        row["choices"]["broad"] = row["choices"]["additive_broad"]
        row["arm_score_bounds"]["broad"] = row["arm_score_bounds"]["additive_broad"]
        row["costs"]["changed_choices"]["broad"] = row["choices"]["broad"] != row["choices"]["anchor"]
        for name, (left, right) in p.CONTRASTS.items():
            row["delta_bounds"][name] = [row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][0]]*2
    value = p.summarize(rows)
    assert value["gates"]["anchor_superiority"] and not value["gates"]["replacement_superiority"]
    assert not value["practical_pass"]


def test_unknown_score_bounds_propagate_without_midpoint_and_same_choice_cancels():
    rows = panel_rows()
    for row in rows:
        for arm in p.ARMS:
            row["arm_score_bounds"][arm] = [.5, .5+8/256] if row["choices"][arm] == 1 else [.5, .5]
        for name, (left, right) in p.CONTRASTS.items():
            row["delta_bounds"][name] = [0., 0.] if row["choices"][left] == row["choices"][right] else [
                row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][1],
                row["arm_score_bounds"][left][1]-row["arm_score_bounds"][right][0]]
    value = p.summarize(rows)
    assert value["contrasts"]["additive_broad_minus_anchor"] == {"mean": None, "mean_bounds": [0., 4/256], "ci95": [0., 4/256]}
    assert value["unknown_outcomes_imputed"] is False and not value["practical_pass"]


@pytest.mark.parametrize("change", ["duplicate_game", "wrong_pair", "wrong_cell", "noninteger_score", "mismatched_shared_action", "wrong_cost", "extra_contrast"])
def test_statistics_reject_invalid_population_or_costs(change):
    rows = panel_rows()
    row = rows[-1]
    if change == "duplicate_game": row["game_index"] = 0
    if change == "wrong_pair": row["pair"] = 0
    if change == "wrong_cell": row["phase"] = 0
    if change == "noninteger_score": row["arm_score_bounds"]["broad"] = [.501, .501]
    if change == "mismatched_shared_action": row["arm_score_bounds"]["teacher"] = [0., 0.]
    if change == "wrong_cost": row["costs"]["extra_selection_continuations"]["additive_broad"] = 1
    if change == "extra_contrast": row["delta_bounds"]["unplanned"] = [0., 0.]
    with pytest.raises(ValueError):
        p.summarize(rows)


def test_budget_checked_during_resampling():
    calls = 0
    def budget():
        nonlocal calls
        calls += 1
        if calls == 3: raise RuntimeError("fixture time limit")
    with pytest.raises(RuntimeError, match="fixture time limit"):
        p.summarize(panel_rows(), budget)
    assert calls == 3
