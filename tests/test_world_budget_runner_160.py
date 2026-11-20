"""160 scheduling, inference gates and durable stage orchestration contracts."""
from __future__ import annotations

import pytest

from splendor_ai.alphazero import run_world_budget as p


def panel(scores):
    return [{"index": i, "pair": i // 2, "seat": i % 2,
             "score": score, "margin": int(4 * score - 2), "turns": 60} for i, score in enumerate(scores)]


def test_schedule_is_complete_paired_and_offpanel():
    main, probe = p.specs("main"), p.specs("probe")
    assert len(main) == 400 and len(probe) == 8
    assert not {s["environment_seed"] for s in main} & {s["environment_seed"] for s in probe}
    for stage, rows, seed in (("main", main, 1600000000), ("probe", probe, 1609000000)):
        for pair in range(len(rows) // 2):
            a, b = rows[2 * pair:2 * pair + 2]
            assert (a["candidate_seat"], b["candidate_seat"]) == (0, 1)
            assert a["environment_seed"] == b["environment_seed"] == seed + pair
            assert a["search_stream"] == b["search_stream"] == p.old.digest(["160-root-stream-v1", stage, pair])
    with pytest.raises(ValueError):
        p.specs("test")


def test_statistics_resamples_pairs_not_separate_games():
    # Every pair has exactly one win and one loss. Correct pair resampling has
    # zero score variation; independent-game resampling would invent variance.
    result = p.statistics(panel([1, 0] * 200))
    assert result["ci95"] == [.5, .5]
    assert result["score_share"] == .5 and result["classification"] == "below_material_target"
    assert result["seats"]["0"]["score_share"] == 1
    assert result["seats"]["1"]["score_share"] == 0


def test_statistics_point_boundary_and_independent_implementation():
    rows = panel([1, 0] * 160 + [1, .5] * 40)
    result = p.statistics(rows)
    assert result["score_share"] == .55 and result["ci95"][0] > .5
    assert result["classification"] == "positive"
    p.auditor().a158.compare(result, p.auditor().independent_statistics(rows))
    assert p.statistics(panel([1, 1] * 110 + [0, 0] * 90))["classification"] == "inconclusive"


def test_statistics_rejects_missing_or_misaligned_pairs():
    with pytest.raises(ValueError):
        p.statistics(panel([1, 0, 1]))
    rows = panel([1, 0, 0, 1])
    rows[-1]["pair"] = 3
    with pytest.raises(AssertionError):
        p.statistics(rows)


def test_preflight_uses_storage_and_timing_without_score(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "stage_evidence", lambda *_: (
        [{"do_not_inspect_game_score": object()}],
        [{"worlds": w, "timing": {"wall": float(w)}} for w in (32, 128)],
        {"collect_seconds": 100, "audit_seconds": 20}))
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / "artifact").write_bytes(b"a" * 100)
    for phase in ("collect", "audit"):
        p.old.save_new(tmp_path / f"probe_{phase}_lock.json", {})
    monkeypatch.setattr(p.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 100 * 1024**3})())
    p.preflight("fixture")
    saved = p.old.read_json(tmp_path / "preflight.json")
    assert saved["passed"] and not saved["candidate_score_inspected_for_gate"]
    assert saved["estimated_main_seconds"] == 6000
    assert saved["estimated_main_bytes"] == 5000


def test_preflight_preserves_failure_without_launching_main(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "stage_evidence", lambda *_: ([],
        [{"worlds": w, "timing": {"wall": float(w)}} for w in (32, 128)],
        {"collect_seconds": 1, "audit_seconds": 1}))
    (tmp_path / "probe").mkdir()
    for phase in ("collect", "audit"):
        p.old.save_new(tmp_path / f"probe_{phase}_lock.json", {})
    monkeypatch.setattr(p.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    with pytest.raises(AssertionError, match="storage"):
        p.preflight("fixture")
    assert not p.old.read_json(tmp_path / "preflight.json")["passed"]
    assert not (tmp_path / "main").exists()
