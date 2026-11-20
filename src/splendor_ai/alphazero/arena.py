"""Small paired development arenas for the programme-090 sandbox."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Callable

from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.training.replay import collect_game_replay

from .network import AlphaZeroInference
from .search import AlphaZeroRawBot, AlphaZeroSearchBot, SearchConfig


@dataclass(frozen=True, slots=True)
class ArenaResult:
    name: str
    paired_seeds: int
    games: int
    candidate_wins: int
    candidate_losses: int
    draws: int
    score_share: float
    mean_margin: float
    termination_reasons: dict[str, int]
    rows: tuple[dict[str, object], ...]


BotFactory = Callable[[int, int], object]


def alpha_zero_raw_factory(checkpoint: str | Path, device: str) -> BotFactory:
    inference = AlphaZeroInference(checkpoint, device=device)
    return lambda _seed, _seat: AlphaZeroRawBot(inference=inference)


def alpha_zero_search_factory(
    checkpoint: str | Path,
    device: str,
    config: SearchConfig,
) -> BotFactory:
    evaluation = replace(config, root_noise_fraction=0.0)
    inference = AlphaZeroInference(checkpoint, device=device)

    def factory(seed: int, seat: int) -> AlphaZeroSearchBot:
        return AlphaZeroSearchBot(
            config=replace(evaluation, seed=seed * 2 + seat),
            self_play=False,
            inference=inference,
        )

    return factory


def policy_035_factory(checkpoint: str | Path, device: str) -> BotFactory:
    return lambda _seed, _seat: LegalActionAttentionScorerBot(checkpoint, device=device)


def run_paired_arena(
    *,
    name: str,
    candidate_factory: BotFactory,
    baseline_factory: BotFactory,
    seed_start: int,
    paired_seeds: int,
    max_turns: int = 200,
    progress_path: str | Path | None = None,
    resume: bool = False,
) -> ArenaResult:
    if paired_seeds <= 0:
        raise ValueError("paired_seeds must be positive.")
    identity = {
        "name": name,
        "seed_start": seed_start,
        "paired_seeds": paired_seeds,
        "max_turns": max_turns,
    }
    schedule = [
        (seed_start + offset, candidate_seat)
        for offset in range(paired_seeds)
        for candidate_seat in (0, 1)
    ]
    output = Path(progress_path) if progress_path is not None else None
    rows: list[dict[str, object]] = []
    if output is not None and resume and output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("Arena progress identity does not match the requested run.")
        rows = list(payload.get("rows", []))
        completed_schedule = [
            (int(row["seed"]), int(row["candidate_seat"])) for row in rows
        ]
        if completed_schedule != schedule[: len(rows)]:
            raise ValueError("Arena progress rows are not the exact schedule prefix.")
    if output is not None:
        _write_arena_progress(output, identity, rows, completed=False)
    for task_index, (seed, candidate_seat) in enumerate(schedule):
        if task_index < len(rows):
            continue
        candidate = candidate_factory(seed, candidate_seat)
        baseline = baseline_factory(seed, 1 - candidate_seat)
        bots = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
        game = collect_game_replay(
            bots[0],
            bots[1],
            seed=seed,
            max_turns=max_turns,
            include_state_snapshots=False,
        )
        if game.winner is None:
            points = 0.5
        elif game.winner == candidate_seat:
            points = 1.0
        else:
            points = 0.0
        margin = game.final_scores[candidate_seat] - game.final_scores[1 - candidate_seat]
        rows.append(
            {
                "seed": seed,
                "candidate_seat": candidate_seat,
                "winner": game.winner,
                "candidate_points": points,
                "candidate_margin": margin,
                "final_scores": list(game.final_scores),
                "turns": game.turns,
                "termination_reason": game.termination_reason,
            }
        )
        if output is not None:
            _write_arena_progress(output, identity, rows, completed=False)
    result = _summarize_arena(name, paired_seeds, rows)
    if output is not None:
        _write_arena_progress(output, identity, rows, completed=True)
    return result


def _summarize_arena(
    name: str,
    paired_seeds: int,
    rows: list[dict[str, object]],
) -> ArenaResult:
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    margin_sum = sum(float(row["candidate_margin"]) for row in rows)
    terminations: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        terminations[reason] = terminations.get(reason, 0) + 1
    games = len(rows)
    if games != paired_seeds * 2:
        raise ValueError("Arena summary requires the complete paired schedule.")
    return ArenaResult(
        name=name,
        paired_seeds=paired_seeds,
        games=games,
        candidate_wins=wins,
        candidate_losses=losses,
        draws=draws,
        score_share=(wins + 0.5 * draws) / games,
        mean_margin=margin_sum / games,
        termination_reasons=dict(sorted(terminations.items())),
        rows=tuple(rows),
    )


def _write_arena_progress(
    path: Path,
    identity: dict[str, object],
    rows: list[dict[str, object]],
    *,
    completed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity,
        "target_games": int(identity["paired_seeds"]) * 2,
        "completed_games": len(rows),
        "completed": completed,
        "rows": rows,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))
