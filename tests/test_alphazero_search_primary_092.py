from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from splendor_ai.alphazero.arena import run_paired_arena
from splendor_ai.alphazero.search_primary import (
    candidate_discovery_eligibility,
    paired_score_share_bootstrap,
    searched_confirmation_acceptance,
    select_discovery_candidate,
)


def _arena_rows(cluster_points: list[tuple[float, float]]) -> list[dict[str, object]]:
    rows = []
    for seed, points in enumerate(cluster_points, start=100):
        for seat, point in enumerate(points):
            rows.append(
                {
                    "seed": seed,
                    "candidate_seat": seat,
                    "candidate_points": point,
                    "candidate_margin": 1 if point == 1.0 else -1,
                    "termination_reason": "completed",
                }
            )
    return rows


def _arena_payload(
    cluster_points: list[tuple[float, float]],
    *,
    margin: float = 1.0,
) -> dict[str, object]:
    rows = _arena_rows(cluster_points)
    return {
        "games": len(rows),
        "rows": rows,
        "score_share": sum(float(row["candidate_points"]) for row in rows)
        / len(rows),
        "mean_margin": margin,
        "termination_reasons": {"completed": len(rows)},
    }


def test_paired_bootstrap_is_deterministic_and_clustered() -> None:
    arena = _arena_payload([(1.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.0, 0.0)])
    first = paired_score_share_bootstrap(arena, samples=500, seed=92)
    second = paired_score_share_bootstrap(arena, samples=500, seed=92)
    assert first == second
    assert first["paired_seeds"] == 4
    assert first["mean"] == pytest.approx(arena["score_share"])


def test_search_confirmation_requires_both_seats() -> None:
    arena = _arena_payload([(1.0, 0.0)] * 16, margin=1.0)
    bootstrap = {"ci95_low": 0.51}
    result = searched_confirmation_acceptance(arena, bootstrap)
    assert result["accepted"] is False
    assert result["checks"]["seat_zero_nonlosing"] is True
    assert result["checks"]["seat_one_nonlosing"] is False


def test_candidate_discovery_and_selection_use_comparable_loss_tiebreak() -> None:
    arena = _arena_payload([(1.0, 0.0)] * 8, margin=0.0)
    assert candidate_discovery_eligibility(arena)["eligible"] is True
    rows = [
        {
            "name": "first",
            "order": 0,
            "arena": arena,
            "eligibility": {"eligible": True},
            "unweighted_validation_total_loss": 3.5,
        },
        {
            "name": "second",
            "order": 1,
            "arena": arena,
            "eligibility": {"eligible": True},
            "unweighted_validation_total_loss": 3.4,
        },
    ]
    assert select_discovery_candidate(rows)["name"] == "second"


def test_arena_progress_resumes_exact_schedule_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls = {"candidate": 0, "baseline": 0}

    def candidate_factory(_seed: int, seat: int) -> dict[str, object]:
        factory_calls["candidate"] += 1
        return {"candidate": True, "seat": seat}

    def baseline_factory(_seed: int, seat: int) -> dict[str, object]:
        factory_calls["baseline"] += 1
        return {"candidate": False, "seat": seat}

    def fake_game(bot_0, bot_1, *, seed, max_turns, include_state_snapshots):
        del seed, max_turns, include_state_snapshots
        candidate_seat = 0 if bot_0["candidate"] else 1
        scores = (15, 10) if candidate_seat == 0 else (10, 15)
        return SimpleNamespace(
            winner=candidate_seat,
            final_scores=scores,
            turns=60,
            termination_reason="completed",
        )

    monkeypatch.setattr(
        "splendor_ai.alphazero.arena.collect_game_replay",
        fake_game,
    )
    progress = tmp_path / "arena.json"
    first = run_paired_arena(
        name="resume_test",
        candidate_factory=candidate_factory,
        baseline_factory=baseline_factory,
        seed_start=10,
        paired_seeds=2,
        progress_path=progress,
    )
    assert first.candidate_wins == 4
    payload = json.loads(progress.read_text(encoding="utf-8"))
    payload["rows"] = payload["rows"][:2]
    payload["completed_games"] = 2
    payload["completed"] = False
    progress.write_text(json.dumps(payload), encoding="utf-8")
    factory_calls.update(candidate=0, baseline=0)

    resumed = run_paired_arena(
        name="resume_test",
        candidate_factory=candidate_factory,
        baseline_factory=baseline_factory,
        seed_start=10,
        paired_seeds=2,
        progress_path=progress,
        resume=True,
    )
    assert resumed.candidate_wins == 4
    assert factory_calls == {"candidate": 2, "baseline": 2}
    assert json.loads(progress.read_text(encoding="utf-8"))["completed"] is True
