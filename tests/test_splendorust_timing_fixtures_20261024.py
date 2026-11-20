"""Selection integrity checks; no neural model or native worker is used."""
from copy import deepcopy
import pytest

from tools.splendorust_timing_20261024 import fixtures as f


def pool(per_cell=12):
    rows = []
    for cell, (phase, seat) in enumerate(f.CELLS):
        for index in range(per_cell):
            pair = cell * per_cell + index
            ply = (0 if cell < 2 else 20 if cell < 4 else 40) + seat
            rows.append({"fixture_id": f"test/{pair}", "pair_index": pair,
                         "game_index": 2 * pair, "ply": ply, "seat": seat,
                         "phase": phase, "observer_sha256": f.digest([pair])})
    return rows


def test_selection_is_permutation_timing_and_outcome_invariant():
    rows = pool()
    expected = f.select_stratified(rows, domain="test", per_cell=2)
    changed = deepcopy(rows)
    for index, row in enumerate(changed):
        row.update(receipt_elapsed_seconds=index * .1, winner=index % 2,
                   action={"bad": index}, trace={"work": index})
    observed = f.select_stratified(list(reversed(changed)), domain="test", per_cell=2)
    assert [row["fixture_id"] for row in observed] == [row["fixture_id"] for row in expected]
    assert len({row["pair_index"] for row in observed}) == 12


def test_validation_excludes_whole_development_and_replay_pairs_and_observers():
    rows = pool()
    initial = f.select_stratified(rows, domain="test", per_cell=1)
    excluded_pairs = {row["pair_index"] for row in initial[:3]}
    excluded_observers = {row["observer_sha256"] for row in initial[3:]}
    selected = f.select_stratified(rows, domain="test", per_cell=2,
                                   excluded_pairs=excluded_pairs, excluded_observers=excluded_observers)
    assert not {row["pair_index"] for row in selected} & excluded_pairs
    assert not {row["observer_sha256"] for row in selected} & excluded_observers
    assert all(sum(row["phase"] == phase and row["seat"] == seat for row in selected) == 2
               for phase, seat in f.CELLS)


def test_strict_cell_shortage_stops_instead_of_substituting():
    with pytest.raises(ValueError, match="insufficient"):
        f.select_stratified(pool(1), domain="test", per_cell=2)


def test_interleaved_schedule_reverses_order_and_retains_every_root():
    roots = [{"fixture_id": str(i)} for i in range(3)]
    result = f.repeated_schedule(roots)
    assert [(row["mode"], row["fixture_index"]) for row in result[:6]] == [
        ("baseline", 0), ("profiled", 0), ("profiled", 1),
        ("baseline", 1), ("baseline", 2), ("profiled", 2)]
    assert [(row["mode"], row["fixture_index"]) for row in result[6:]] == [
        ("profiled", 2), ("baseline", 2), ("baseline", 1),
        ("profiled", 1), ("profiled", 0), ("baseline", 0)]
    assert [row["index"] for row in result] == list(range(12))


def test_exact_failed_variant_is_requested_in_both_modes_and_passes():
    roots = [{"fixture_id": "a", "snapshot": {"state": "original"},
              "preserved_variants": [{"snapshot": {"state": "hidden"}}]}]
    schedule = f.repeated_schedule(roots)
    assert len(schedule) == 8
    failed = [row for row in schedule if row["variant"] == "preserved/0"]
    assert {(row["pass_index"], row["mode"]) for row in failed} == {
        (0, "baseline"), (0, "profiled"), (1, "baseline"), (1, "profiled")}
    assert all(f.snapshot_for_request(roots, row) == {"state": "hidden"} for row in failed)


@pytest.mark.parametrize("ply,phase", [(0,"0..19"),(19,"0..19"),(20,"20..39"),(39,"20..39"),(40,"40+")])
def test_phase_boundaries(ply, phase):
    assert f.phase_for(ply) == phase


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_invalid_ply_rejected(value):
    with pytest.raises(ValueError):
        f.phase_for(value)
