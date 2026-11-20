"""Focused certificate, censor-replay and recovery-prefix regression checks."""
import copy
from fractions import Fraction
from itertools import product

import pytest

from tools.equal_budget_breadth_arena_128_recovery_01 import arena as p, common as c, verify as a
from tools.equal_budget_breadth_arena_128 import verify as prior
from tools.equal_budget_breadth_arena_128.test_arena import fixture_registry, private_state

RECOVERY = "isolated-recovery-audit-fixture"


def worlds(columns):
    rows = []
    for wi in range(max(map(len, columns))):
        active = [j for j, column in enumerate(columns) if wi < len(column)]
        outcomes = []
        for j in active:
            value = columns[j][wi]
            outcomes.append({"reason": "repetition_cutoff" if value is None else "completed",
                             "winner": None if value is None or value == 0 else int(value < 0),
                             "wdl": value, "turns": 4,
                             "trace_sha256": "a"*64, "final_state_sha256": "b"*64})
        rows.append({"candidate_indices": active, "outcomes": outcomes})
    return rows


def test_exhaustive_small_unequal_certificates_match_every_possible_completion():
    # Exhaustive discrete extrema/interiors, not a second interval implementation.
    for values in product((-1, 0, 1, None), repeat=5):
        columns = [list(values[:2]), list(values[2:])]
        saved = worlds(columns)
        result = a.independent_selection(saved, [2, 3])
        unknown = [i for i, value in enumerate(values) if value is None]
        winners = set()
        for completions in product((-1, 0, 1), repeat=len(unknown)):
            filled = list(values)
            for i, value in zip(unknown, completions):
                filled[i] = value
            means = [Fraction(sum(filled[:2]), 2), Fraction(sum(filled[2:]), 3)]
            winners.add(0 if means[0] >= means[1] else 1)
        assert result["choice"] == (next(iter(winners)) if len(winners) == 1 else None)
        try:
            collected = p.select(saved, [2, 3])
        except p.AmbiguousDecision as exc:
            collected = exc.selection
        assert collected == result


@pytest.mark.parametrize("columns,winner", [
    ([[1], [None]], 0), ([[None], [1]], None),
    ([[None]], 0), ([[0, 1], [-1, None]], 0),
    ([[-1, None], [1, 1]], 1),
])
def test_original_ties_and_single_candidate(columns, winner):
    result = a.independent_selection(worlds(columns), list(map(len, columns)))
    assert result["choice"] == winner
    if any(None in column for column in columns):
        assert "certificate" in result
        assert all(mean is None for column, mean in zip(columns, result["means"]) if None in column)


@pytest.mark.parametrize("counts", [[103,103,102,102,102], [86,86,85,85,85,85], [74,73,73,73,73,73,73], [64]*8])
def test_ragged_exact_denominators_with_unresolved_losing_candidate(counts):
    columns = [[1]*count for count in counts]
    columns[-1][-1] = None
    result = a.independent_selection(worlds(columns), counts)
    assert result == p.select(worlds(columns), counts)
    assert result["choice"] == 0 and result["counts"] == counts and sum(counts) == 512
    assert result["certificate"]["known_counts"][-1] == counts[-1]-1


@pytest.mark.parametrize("damage", ["reason", "winner", "wdl", "bool", "missing", "extra", "reorder"])
def test_only_valid_repetition_unknowns_and_exact_sparse_delivery(damage):
    rows = worlds([[1,1,1], [-1,None]])
    if damage == "reason": rows[1]["outcomes"][1]["reason"] = "max_turns"
    elif damage == "winner": rows[1]["outcomes"][1]["winner"] = 0
    elif damage == "wdl": rows[1]["outcomes"][1]["wdl"] = 0
    elif damage == "bool": rows[0]["outcomes"][0]["wdl"] = True
    elif damage == "missing": rows[-1]["outcomes"].clear()
    elif damage == "extra": rows[-1]["candidate_indices"].append(1); rows[-1]["outcomes"].append(rows[0]["outcomes"][1])
    else: rows[0]["candidate_indices"].reverse()
    with pytest.raises((AssertionError, ValueError)):
        a.independent_selection(rows, [3,2])


@pytest.mark.parametrize("reason", ["max_turns", "no_progress_cutoff", "stalled", "illegal_or_missing_action"])
def test_other_cutoffs_remain_fatal(reason):
    outcome = worlds([[None]])[0]["outcomes"][0]
    outcome["reason"] = reason
    with pytest.raises(AssertionError): a._check_outcome(outcome, 0, 0)


@pytest.mark.parametrize("damage", ["bound", "known", "unknown", "cells", "rule", "tie", "bool", "missing", "means", "denominator"])
def test_saved_certificate_tampering_is_rejected(damage):
    rows = worlds([[1,1], [None,-1]])
    root = {"worlds": rows, **a.independent_selection(rows, [2,2])}
    if damage == "bound": root["certificate"]["lower_sums"][1] += 1
    elif damage == "known": root["certificate"]["known_counts"][1] += 1
    elif damage == "unknown": root["certificate"]["unresolved_counts"][1] = 0
    elif damage == "cells": root["certificate"]["unresolved_cells"][0]["world"] = 1
    elif damage == "rule": root["certificate"]["rule"] = "draw_fill"
    elif damage == "tie": root["certificate"]["choice"] = 1
    elif damage == "bool": root["certificate"]["unresolved_counts"][1] = True
    elif damage == "missing": root.pop("certificate")
    elif damage == "means": root["means"][1] = -.5
    else: root["counts"][1] = 1
    with pytest.raises(AssertionError): a.certified_selection(root, [2,2])


def test_ambiguous_root_cannot_be_accepted_and_natural_roots_stay_identical():
    rows = worlds([[None], [0]])
    with pytest.raises(AssertionError, match="change the selected move"):
        a.certified_selection({"worlds": rows, **a.independent_selection(rows, [1,1])}, [1,1])
    natural = worlds([[1,0,-1], [0,1,0]])
    assert a.independent_selection(natural, [3,3]) == prior.independent_selection(natural, [3,3])


def test_censor_replays_added_without_changing_sample_or_double_counting():
    rows = worlds([[1,None,1], [None,1]])
    plan = a.independent_replay_cells(rows, [0,2])
    assert plan["sampled"] == {(0,0),(0,1),(2,0)}
    assert plan["unresolved"] == {(0,1),(1,0)}
    assert plan["extra"] == {(1,0)} and len(plan["all"]) == 4
    assert not a.independent_replay_cells(rows, [0,1,2])["extra"]
    for count in (64,74,86,103,128):
        assert a.ordinary_indices("main", 13, 17, count) == prior.ordinary_indices("main", 13, 17, count)


def prepare_fixture(base):
    p.old.save_new(base/"freeze.json", {"binding": RECOVERY})
    p.old.save_new(base/"inventory.json", {"original_binding": c.ORIGINAL_BINDING,
        "games": {}, "roots": {}, "original_stage_launches": {}})
    p.old.save_new(base/"test_launch.json", {"manifest": c.ORIGINAL_BINDING, "recovery_binding": RECOVERY})


@pytest.fixture(scope="module")
def actual_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("recovery_audit_root")
    prepare_fixture(base)
    p.initialize_models(base, fixture_registry())
    state, spec = private_state(), p.specs("test")[0]
    path = p.root_path("test", 0, state.turn_index, base)
    action, root = p.choose_root(state, spec["search_stream"], path, c.ORIGINAL_BINDING,
        f"anchor_{spec['fit']}", ["test",0,state.turn_index], True, base, RECOVERY)
    move = {"turn": state.turn_index, "actor": state.current_player,
            "state_sha256": p.old.digest(p.old.serialize_state_snapshot(state)),
            "root_sha256": p.old.file_hash(path), "action": p.old.serialize_action(action)}
    ordinary = a.ordinary_indices("test",0,state.turn_index,root["allocation"]["world_count"])
    selection = {"root_id": a.root_id(0,state.turn_index), "ordinary_world_indices": ordinary,
                 "replay_world_indices": ordinary, "stress": False}
    return base, state, spec, root, move, selection


def test_actual_natural_root_preserves_sample_counts_and_zero_censor_accounting(actual_root):
    base, state, spec, root, move, selection = actual_root
    receipt = a.audit_root("test",0,state,move,spec,c.ORIGINAL_BINDING,base,selection)
    assert receipt["terminal_continuations"] == receipt["continuations"] == 512
    assert receipt["unresolved_continuations"] == receipt["certified_roots"] == receipt["extra_censored_replay_cells"] == 0
    assert receipt["replayed_worlds"] == len(selection["replay_world_indices"])
    assert receipt["timing"] == root["timing"]
    assert receipt["inherited_timing"] == {"wall":0., "cpu":0.}


def test_forged_repetition_outside_sample_is_always_replayed(actual_root, tmp_path, monkeypatch):
    _, state, spec, original, original_move, selection = actual_root
    prepare_fixture(tmp_path)
    root, move = copy.deepcopy(original), copy.deepcopy(original_move)
    wi = next(i for i in range(len(root["worlds"])) if i not in selection["replay_world_indices"])
    root["worlds"][wi]["outcomes"][0].update(reason="repetition_cutoff", winner=None, wdl=None)
    path = p.root_path("test",0,state.turn_index,tmp_path)
    p.old.save_new(path,root)
    move["root_sha256"] = p.old.file_hash(path)
    def forbidden(*args): raise RuntimeError("selection reached before forged repetition was rejected")
    monkeypatch.setattr(a,"certified_selection",forbidden)
    with pytest.raises(ValueError, match="repetition_cutoff"):
        a.audit_root("test",0,state,move,spec,c.ORIGINAL_BINDING,tmp_path,selection)


def test_new_cell_accounting_tamper_is_rejected(actual_root, tmp_path):
    _, state, spec, original, original_move, selection = actual_root
    prepare_fixture(tmp_path)
    root, move = copy.deepcopy(original), copy.deepcopy(original_move)
    root["new_cells"] -= 1
    path = p.root_path("test",0,state.turn_index,tmp_path)
    p.old.save_new(path,root)
    move["root_sha256"] = p.old.file_hash(path)
    with pytest.raises(AssertionError):
        a.audit_root("test",0,state,move,spec,c.ORIGINAL_BINDING,tmp_path,selection)


def test_censored_root_prefix_must_preserve_every_original_outcome(tmp_path, monkeypatch):
    original = {key: {"fixture": key} for key in ("snapshot","privacy","root_identity","planner",
                "schedule","candidates","proposal","allocation","scores")}
    original.update(worlds=worlds([[1,None],[0,-1]]), timing={"wall":1.,"cpu":1.}, failure="original failure")
    source = tmp_path/"original_failed.json"
    p.old.save_new(source,original)
    ref = c.artifact(source)
    monkeypatch.setattr(c,"load_inventory",lambda base: {"roots":{"main":{"13:17":ref}},"failed_root":ref})
    monkeypatch.setattr(c,"validate_origin",lambda *args: True)
    root = {key: copy.deepcopy(value) for key,value in original.items() if key != "failure"}
    root.update(recovery_origin=ref, inherited_timing=original["timing"], inherited_cells=4)
    a._check_root_prefix(root,"main",13,17,c.ORIGINAL_BINDING,tmp_path)
    root["timing"] = {"wall":.5,"cpu":.25}
    assert a._timing_fields(root) == {"timing":{"wall":.5,"cpu":.25},
                                      "inherited_timing":original["timing"]}
    altered = copy.deepcopy(root)
    altered["inherited_timing"]["wall"] += 1
    with pytest.raises(AssertionError): a._check_root_prefix(altered,"main",13,17,c.ORIGINAL_BINDING,tmp_path)
    root["worlds"][0]["outcomes"][0]["wdl"] = 0
    with pytest.raises(AssertionError): a._check_root_prefix(root,"main",13,17,c.ORIGINAL_BINDING,tmp_path)


def test_recovered_outer_prefix_cannot_drop_or_change_original_decisions(tmp_path, monkeypatch):
    original = {"spec":{"index":13},"initial_state_sha256":"a"*64,"trace":[{"turn":0,"action":"unchanged"}],
                "timing":{"wall":1.,"cpu":1.},"failure":"original failure"}
    source = tmp_path/"original_game.json"
    p.old.save_new(source,original)
    monkeypatch.setattr(c,"resolve_partial_game",lambda *args: source)
    monkeypatch.setattr(c,"validate_origin",lambda *args: True)
    game = {key:copy.deepcopy(value) for key,value in original.items() if key != "failure"}
    game.update(recovery_origin=c.artifact(source), inherited_timing=original["timing"], inherited_turns=1)
    assert a._check_game_prefix(game,"main",13,c.ORIGINAL_BINDING,tmp_path) == original
    assert a._timing_fields(game)["inherited_timing"] == original["timing"]
    game["trace"].clear()
    with pytest.raises(AssertionError): a._check_game_prefix(game,"main",13,c.ORIGINAL_BINDING,tmp_path)
