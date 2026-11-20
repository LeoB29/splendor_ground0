"""Hand-checkable contracts for the isolated saved-outcome sampling study."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


BASE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / "scripts/analysis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = load("saved_halving_screen")
audit = load("verify_saved_halving_screen")
ORDER = list(range(128))


def zeros(support=4):
    return [[0] * support for _ in ORDER]


def tie_fixture():
    values = zeros()
    for world in range(16):
        values[world][3] = 2
        values[world][1] = 1
    for world in range(16, 48):
        values[world][1] = 1
    for world in range(16, 24):
        values[world][3] = 2
    for world in range(48, 128):
        values[world][1] = 1
        values[world][3] = 2
    return values


@pytest.mark.parametrize("support", [1, 2, 3])
def test_short_support_remains_identical(support):
    values = [[(world + action) % 3 for action in range(support)] for world in ORDER]
    row = screen.analyze_partition(values, ORDER)
    assert row["uniform_choice"] == row["adaptive_choice"]
    assert row["uniform_access"] == row["adaptive_access"]
    assert len(row["uniform_access"]) == 32 * support
    assert row["uniform_heldout_sum2"] == row["adaptive_heldout_sum2"]
    assert row == audit.expected_partition(values, ORDER)


def test_first_round_ties_keep_original_first_two():
    row = screen.analyze_partition(zeros(), ORDER)
    assert row["survivors"] == [0, 1]
    assert row["uniform_choice"] == row["adaptive_choice"] == 0
    assert len(row["adaptive_access"]) == len(row["uniform_access"]) == 128


def test_final_tie_uses_original_index_not_initial_ranking():
    row = screen.analyze_partition(tie_fixture(), ORDER)
    assert row["survivors"] == [1, 3]
    assert row["adaptive_choice"] == 1  # both totals48, despite first-round3 leading
    assert row["uniform_choice"] == 3
    assert row["uniform_heldout_sum2"] == 160
    assert row["adaptive_heldout_sum2"] == 80
    assert (row["adaptive_heldout_sum2"] - row["uniform_heldout_sum2"]) / 160 == -.5
    assert row == audit.expected_partition(tie_fixture(), ORDER)


def test_final_estimate_includes_initial_sixteen():
    values = tie_fixture()
    for world in range(16, 48):
        values[world][1] = 2 if world < 28 else 0  #24 extra vs16extra for3
    row = screen.analyze_partition(values, ORDER)
    assert row["adaptive_choice"] == 3  #48 total beats40, despite worse extra batch
    assert row == audit.expected_partition(values, ORDER)


def test_holdout_changes_cannot_change_decisions_or_accesses():
    values = tie_fixture()
    before = screen.analyze_partition(values, ORDER)
    for world in range(48, 128):
        values[world] = [2, 2, 2, 0]
    after = screen.analyze_partition(values, ORDER)
    for key in ("uniform_choice", "adaptive_choice", "survivors", "uniform_access", "adaptive_access"):
        assert before[key] == after[key]
    assert after["adaptive_heldout_sum2"] == 160
    assert after["uniform_heldout_sum2"] == 0


def test_unallocated_loser_cells_cannot_change_adaptive_choice():
    values = tie_fixture()
    before = screen.analyze_partition(values, ORDER)
    for world in range(16, 48):
        values[world][0] = values[world][2] = 2
    after = screen.analyze_partition(values, ORDER)
    assert before["adaptive_choice"] == after["adaptive_choice"] == 1
    assert before["adaptive_access"] == after["adaptive_access"]
    assert all(action in (1, 3) for world, action in after["adaptive_access"][64:])


def test_permutation_moves_whole_world_and_preserves_result():
    values = tie_fixture()
    order = np.random.default_rng(900).permutation(128).tolist()
    shuffled = zeros()
    for old, new in enumerate(order):
        shuffled[new] = values[old]
    original = screen.analyze_partition(values, ORDER)
    changed = screen.analyze_partition(shuffled, order)
    for key in ("uniform_choice", "adaptive_choice", "survivors", "uniform_heldout_sum2", "adaptive_heldout_sum2"):
        assert original[key] == changed[key]
    assert changed == audit.expected_partition(shuffled, order)
    for key in ("uniform_access", "adaptive_access"):
        assert not ({world for world, _ in changed[key]} & set(order[48:]))
        assert len(set(map(tuple, changed[key]))) == 128


@pytest.mark.parametrize("bad", [list(range(127)), [0] * 128, list(range(1, 129))])
def test_bad_world_partitions_are_rejected(bad):
    with pytest.raises(ValueError):
        screen.analyze_partition(zeros(), bad)
    with pytest.raises(ValueError):
        audit.expected_partition(zeros(), bad)


def test_reader_refuses_holdout_and_duplicate_cell_access():
    reader = screen.AllocatedCells(zeros(), range(48))
    with pytest.raises(ValueError, match="Evaluation"):
        reader.consume([48], [0])
    reader.consume([0], [0])
    with pytest.raises(ValueError, match="Duplicate"):
        reader.consume([0], [0])


@pytest.mark.parametrize("bad", [3, float("nan"), True])
def test_bad_scores_rejected(bad):
    values = zeros()
    values[60][0] = bad
    with pytest.raises(ValueError):
        screen.analyze_partition(values, ORDER)


def test_pair_aggregation_keeps_three_phases_together():
    rows = [{"pair": pair, "phase": phase, "uniform_score": .2 + phase * .001,
             "adaptive_score": .2 + phase * .001 + pair / 1000}
            for pair in range(200) for phase in range(3)]
    pairs = screen.aggregate_roots(rows)
    assert len(pairs) == 200
    assert pairs[100]["uniform_score"] == pytest.approx(.201)
    assert pairs[100]["delta_score"] == pytest.approx(.1)
    assert sum(row["delta_score"] for row in pairs) / 200 == pytest.approx(.0995)
    rows[1]["phase"] = 0
    with pytest.raises(ValueError):
        screen.aggregate_roots(rows)


def test_bootstrap_constant_pair_difference_is_exact():
    pairs = [{"pair": pair, "delta_score": .125} for pair in range(200)]
    primary = screen.primary_statistics(pairs)
    assert primary == {"delta_pp": 12.5, "ci95_pp": [12.5, 12.5]}
    assert primary == audit.bootstrap_primary([.125] * 200)
    with pytest.raises(ValueError):
        screen.primary_statistics(pairs * 3)


def test_bootstrap_two_independent_implementations_agree():
    pairs = [{"pair": pair, "delta_score": (pair % 9 - 4) / 100}
             for pair in range(200)]
    assert screen.primary_statistics(pairs) == audit.bootstrap_primary([p["delta_score"] for p in pairs])


def test_write_once_and_changed_frozen_file_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "OUTPUT", tmp_path)
    (tmp_path / "code.py").write_text("original\n")
    (tmp_path / "input.json").write_text("[]\n")
    value = {"config": screen.CONFIG, "numpy": np.__version__, "python": sys.version,
             "files": {"code.py": screen.sha(tmp_path / "code.py")},
             "input_files": {"input.json": screen.sha(tmp_path / "input.json")}}
    value["binding"] = screen.digest(value)
    screen.save(tmp_path / "freeze.json", value)
    assert screen.frozen() == value
    with pytest.raises(FileExistsError):
        screen.save(tmp_path / "freeze.json", value)
    (tmp_path / "input.json").write_text("[1]\n")
    with pytest.raises(ValueError, match="Bound artifact changed"):
        screen.frozen()
    with pytest.raises(ValueError, match="outside"):
        screen.local("../escape")


def test_unfrozen_analysis_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "OUTPUT", tmp_path)
    with pytest.raises(FileNotFoundError):
        screen.run()
    assert not (tmp_path / "run_started.json").exists()
