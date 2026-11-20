"""Seeded AlphaZero self-play with information-legal replay targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
import time
from typing import Any

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.termination import (
    adjudicate_scores as _adjudicate_scores,
    is_forced_pass_turn,
)

from .network import AlphaZeroInference
from .replay import (
    AlphaZeroReplayRow,
    append_replay_jsonl,
    read_replay_jsonl,
    replay_row_from_search,
)
from .search import AlphaZeroSearch, SearchConfig


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    generation: int
    game_count: int
    seed_start: int
    search: SearchConfig
    max_turns: int = 200
    repetition_limit: int = 4
    no_progress_limit: int = 60

    def validate(self) -> None:
        if self.game_count <= 0:
            raise ValueError("game_count must be positive.")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        self.search.validate()


@dataclass(frozen=True, slots=True)
class SelfPlayGameSummary:
    seed: int
    winner: int | None
    final_scores: tuple[int, int]
    turns: int
    termination_reason: str
    positions: int
    action_family_counts: dict[str, int]
    reserve_deck_legal: int
    reserve_deck_selected: int
    search_seconds: float
    simulations: int
    expansions: int


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    rows: tuple[AlphaZeroReplayRow, ...]
    games: tuple[SelfPlayGameSummary, ...]
    replay_path: Path | None


def generate_self_play(
    *,
    checkpoint_path: str | Path,
    device: str,
    config: SelfPlayConfig,
    replay_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    resume: bool = False,
) -> SelfPlayResult:
    config.validate()
    inference = AlphaZeroInference(checkpoint_path, device=device)
    if inference.checkpoint.generation != config.generation:
        raise ValueError(
            "Self-play generation does not match checkpoint generation: "
            f"{config.generation} != {inference.checkpoint.generation}"
        )
    search = AlphaZeroSearch(inference, config.search)
    all_rows: list[AlphaZeroReplayRow] = []
    game_summaries: list[SelfPlayGameSummary] = []
    output_path = Path(replay_path) if replay_path is not None else None
    progress_output = Path(progress_path) if progress_path is not None else None
    identity = _selfplay_identity(checkpoint_path, config)
    if resume and output_path is None:
        raise ValueError("Resumable self-play requires a replay path.")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and output_path.exists():
            if progress_output is None or not progress_output.exists():
                if output_path.stat().st_size != 0:
                    raise ValueError(
                        "Nonempty resumable replay is missing its identity progress file."
                    )
            else:
                progress_payload = json.loads(
                    progress_output.read_text(encoding="utf-8")
                )
                if progress_payload.get("identity") != identity:
                    raise ValueError("Self-play progress identity does not match the run.")
            all_rows = read_replay_jsonl(output_path)
            game_summaries = _reconstruct_completed_games(all_rows, config)
        else:
            output_path.write_text("", encoding="utf-8")
    if progress_output is not None:
        _write_selfplay_progress(
            progress_output,
            identity=identity,
            config=config,
            rows=all_rows,
            summaries=game_summaries,
        )
    for game_offset in range(len(game_summaries), config.game_count):
        seed = config.seed_start + game_offset
        rows, summary = _play_one_game(
            checkpoint_path=str(checkpoint_path),
            checkpoint_generation=inference.checkpoint.generation,
            search=search,
            config=config,
            seed=seed,
        )
        all_rows.extend(rows)
        game_summaries.append(summary)
        if output_path is not None:
            append_replay_jsonl(output_path, rows)
        if progress_output is not None:
            _write_selfplay_progress(
                progress_output,
                identity=identity,
                config=config,
                rows=all_rows,
                summaries=game_summaries,
            )
    return SelfPlayResult(
        rows=tuple(all_rows),
        games=tuple(game_summaries),
        replay_path=output_path,
    )


def _selfplay_identity(
    checkpoint_path: str | Path,
    config: SelfPlayConfig,
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "generation": config.generation,
        "game_count": config.game_count,
        "seed_start": config.seed_start,
        "search": asdict(config.search),
        "max_turns": config.max_turns,
        "repetition_limit": config.repetition_limit,
        "no_progress_limit": config.no_progress_limit,
    }


def _reconstruct_completed_games(
    rows: list[AlphaZeroReplayRow],
    config: SelfPlayConfig,
) -> list[SelfPlayGameSummary]:
    grouped: dict[int, list[AlphaZeroReplayRow]] = {}
    order: list[int] = []
    for row in rows:
        if row.game_seed not in grouped:
            grouped[row.game_seed] = []
            order.append(row.game_seed)
        grouped[row.game_seed].append(row)
    expected = [config.seed_start + offset for offset in range(len(order))]
    if order != expected or len(order) > config.game_count:
        raise ValueError("Resumable replay is not the exact self-play seed prefix.")
    summaries: list[SelfPlayGameSummary] = []
    for seed in order:
        game_rows = grouped[seed]
        if any(
            row.generation != config.generation
            or row.value_target is None
            or row.final_scores is None
            or row.termination_reason is None
            for row in game_rows
        ):
            raise ValueError("Resumable replay contains an incomplete game.")
        if [row.turn_index for row in game_rows] != list(range(len(game_rows))):
            raise ValueError("Resumable replay game is not an exact turn prefix.")
        winners = {row.winner for row in game_rows}
        scores = {row.final_scores for row in game_rows}
        reasons = {row.termination_reason for row in game_rows}
        if len(winners) != 1 or len(scores) != 1 or len(reasons) != 1:
            raise ValueError("Resumable replay has inconsistent game outcomes.")
        families: Counter[str] = Counter()
        reserve_legal = 0
        reserve_selected = 0
        for row in game_rows:
            payloads = row.action_payloads
            reserve_legal += int(
                any(payload.get("action_type") == "RESERVE_DECK" for payload in payloads)
            )
            selected = payloads[row.selected_action_row]
            family = str(selected.get("action_type"))
            families[family] += 1
            reserve_selected += int(family == "RESERVE_DECK")
        summaries.append(
            SelfPlayGameSummary(
                seed=seed,
                winner=next(iter(winners)),
                final_scores=next(iter(scores)),  # type: ignore[arg-type]
                turns=len(game_rows),
                termination_reason=str(next(iter(reasons))),
                positions=len(game_rows),
                action_family_counts=dict(sorted(families.items())),
                reserve_deck_legal=reserve_legal,
                reserve_deck_selected=reserve_selected,
                search_seconds=sum(
                    float(row.search_telemetry["elapsed_seconds"])
                    for row in game_rows
                ),
                simulations=sum(
                    int(row.search_telemetry["simulations"]) for row in game_rows
                ),
                expansions=sum(
                    int(row.search_telemetry["expansions"]) for row in game_rows
                ),
            )
        )
    return summaries


def _write_selfplay_progress(
    path: Path,
    *,
    identity: dict[str, Any],
    config: SelfPlayConfig,
    rows: list[AlphaZeroReplayRow],
    summaries: list[SelfPlayGameSummary],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity,
        "generation": config.generation,
        "target_games": config.game_count,
        "completed_games": len(summaries),
        "completed_positions": len(rows),
        "games": [asdict(game) for game in summaries],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def summarize_self_play(result: SelfPlayResult) -> dict[str, Any]:
    terminations = Counter(game.termination_reason for game in result.games)
    families: Counter[str] = Counter()
    for game in result.games:
        families.update(game.action_family_counts)
    positions = len(result.rows)
    visit_prior_jsd = [
        _jensen_shannon(row.policy_target, row.root_network_priors)
        for row in result.rows
    ]
    visit_prior_tv = [
        0.5
        * sum(
            abs(visit - prior)
            for visit, prior in zip(
                row.policy_target,
                row.root_network_priors,
                strict=True,
            )
        )
        for row in result.rows
    ]
    visit_entropies = [_entropy(row.policy_target) for row in result.rows]
    prior_entropies = [_entropy(row.root_network_priors) for row in result.rows]
    return {
        "games": len(result.games),
        "positions": positions,
        "termination_reasons": dict(sorted(terminations.items())),
        "mean_turns": (
            sum(game.turns for game in result.games) / len(result.games)
            if result.games
            else 0.0
        ),
        "action_family_counts": dict(sorted(families.items())),
        "reserve_deck_legal": sum(game.reserve_deck_legal for game in result.games),
        "reserve_deck_selected": sum(
            game.reserve_deck_selected for game in result.games
        ),
        "sampled_action_changed_raw_actions": sum(
            row.search_changed_raw_action for row in result.rows
        ),
        "sampled_action_changed_raw_fraction": (
            sum(row.search_changed_raw_action for row in result.rows) / positions
            if positions
            else 0.0
        ),
        "visit_argmax_changed_raw_actions": sum(
            row.visit_argmax_changed_raw_action for row in result.rows
        ),
        "visit_argmax_changed_raw_fraction": (
            sum(row.visit_argmax_changed_raw_action for row in result.rows) / positions
            if positions
            else 0.0
        ),
        "mean_search_minus_raw_value": (
            sum(row.search_root_value - row.raw_network_value for row in result.rows)
            / positions
            if positions
            else 0.0
        ),
        "mean_visit_to_network_prior_jsd": (
            sum(visit_prior_jsd) / positions if positions else 0.0
        ),
        "mean_visit_to_network_prior_tv": (
            sum(visit_prior_tv) / positions if positions else 0.0
        ),
        "mean_visit_entropy": (
            sum(visit_entropies) / positions if positions else 0.0
        ),
        "mean_network_prior_entropy": (
            sum(prior_entropies) / positions if positions else 0.0
        ),
        "search_seconds": sum(game.search_seconds for game in result.games),
        "simulations": sum(game.simulations for game in result.games),
        "expansions": sum(game.expansions for game in result.games),
        "games_detail": [asdict(game) for game in result.games],
    }


def _play_one_game(
    *,
    checkpoint_path: str,
    checkpoint_generation: int,
    search: AlphaZeroSearch,
    config: SelfPlayConfig,
    seed: int,
) -> tuple[list[AlphaZeroReplayRow], SelfPlayGameSummary]:
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    pending: list[AlphaZeroReplayRow] = []
    visits: dict[tuple[object, ...], int] = {}
    no_progress = 0
    forced_passes = 0
    reason = "completed"
    action_families: Counter[str] = Counter()
    reserve_deck_legal = 0
    reserve_deck_selected = 0
    total_search_seconds = 0.0
    simulations = 0
    expansions = 0

    while not state.terminal:
        signature = state_signature(state)
        visits[signature] = visits.get(signature, 0) + 1
        if config.repetition_limit > 0 and visits[signature] >= config.repetition_limit:
            reason = "repetition_cutoff"
            break
        if state.turn_index >= config.max_turns:
            reason = "max_turns"
            break
        legal_actions = env.legal_actions(state)
        if not legal_actions:
            reason = "stalled"
            break
        reserve_deck_legal += int(
            any(action.action_type is ActionType.RESERVE_DECK for action in legal_actions)
        )
        result = search.run(
            env,
            state,
            legal_actions,
            self_play=True,
            seed=(
                config.search.seed
                + seed * 1_000_003
                + state.turn_index * 97_409
            ),
        )
        pending.append(
            replay_row_from_search(
                generation=config.generation,
                game_seed=seed,
                state=state,
                result=result,
                checkpoint_path=checkpoint_path,
                checkpoint_generation=checkpoint_generation,
                search_config=asdict(config.search),
            )
        )
        action = result.action
        action_families[action.action_type.name] += 1
        reserve_deck_selected += int(action.action_type is ActionType.RESERVE_DECK)
        total_search_seconds += result.elapsed_seconds
        simulations += result.simulations
        expansions += result.expansions
        next_state = env.step(state, action)
        forced_pass = is_forced_pass_turn(legal_actions)
        forced_passes = forced_passes + 1 if forced_pass else 0
        if forced_passes >= 2:
            state = next_state
            reason = "forced_pass_deadlock"
            break
        if is_progress_transition(state, next_state, action):
            no_progress = 0
        else:
            no_progress += 1
        state = next_state
        if config.no_progress_limit > 0 and no_progress >= config.no_progress_limit:
            reason = "no_progress_cutoff"
            break

    winner = state.winner if state.terminal else _adjudicate_scores(state)
    final_scores = (state.players[0].score, state.players[1].score)
    completed = [
        row.with_outcome(
            winner=winner,
            final_scores=final_scores,
            termination_reason=reason,
        )
        for row in pending
    ]
    summary = SelfPlayGameSummary(
        seed=seed,
        winner=winner,
        final_scores=final_scores,
        turns=state.turn_index,
        termination_reason=reason,
        positions=len(completed),
        action_family_counts=dict(sorted(action_families.items())),
        reserve_deck_legal=reserve_deck_legal,
        reserve_deck_selected=reserve_deck_selected,
        search_seconds=total_search_seconds,
        simulations=simulations,
        expansions=expansions,
    )
    return completed, summary


def _entropy(probabilities: tuple[float, ...]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _jensen_shannon(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    if len(left) != len(right):
        raise ValueError("JSD distributions must have equal support.")
    midpoint = tuple((a + b) / 2.0 for a, b in zip(left, right, strict=True))

    def kl(source: tuple[float, ...], target: tuple[float, ...]) -> float:
        return sum(
            value * math.log(value / other)
            for value, other in zip(source, target, strict=True)
            if value > 0.0 and other > 0.0
        )

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)
