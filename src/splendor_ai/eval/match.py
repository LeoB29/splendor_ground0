"""Match and tournament configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from splendor_ai.bots.base import Bot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState


@dataclass(frozen=True, slots=True)
class MatchConfig:
    games: int = 100
    seconds_per_move: float = 1.0
    swap_seats: bool = True
    max_turns_per_game: int = 400
    repetition_limit: int = 0
    no_progress_limit: int = 0


@dataclass(frozen=True, slots=True)
class GameResult:
    seed: int
    turns: int
    winner: int | None
    final_scores: tuple[int, int]
    bot_seats: tuple[str, str]
    stalled: bool = False
    timed_out: bool = False
    termination_reason: str = "completed"
    repetition_count: int = 0
    no_progress_streak: int = 0
    loop_fallback_triggers_by_seat: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class MatchResult:
    games: tuple[GameResult, ...]

    @property
    def total_games(self) -> int:
        return len(self.games)

    @property
    def wins_by_seat(self) -> tuple[int, int]:
        seat0 = sum(1 for game in self.games if game.winner == 0)
        seat1 = sum(1 for game in self.games if game.winner == 1)
        return seat0, seat1

    @property
    def draws(self) -> int:
        return sum(1 for game in self.games if game.winner is None)


BotFactory = Callable[[], Bot]


def play_game(
    bot_seat_0: Bot,
    bot_seat_1: Bot,
    seed: int = 0,
    max_turns: int = 400,
    repetition_limit: int = 0,
    no_progress_limit: int = 0,
) -> GameResult:
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    bots = (bot_seat_0, bot_seat_1)
    state_visit_counts: dict[tuple[object, ...], int] = {}
    no_progress_streak = 0

    while not state.terminal:
        signature = state_signature(state)
        seen_count = state_visit_counts.get(signature, 0) + 1
        state_visit_counts[signature] = seen_count
        if repetition_limit > 0 and seen_count >= repetition_limit:
            return GameResult(
                seed=seed,
                turns=state.turn_index,
                winner=_adjudicate_scores(state),
                final_scores=(state.players[0].score, state.players[1].score),
                bot_seats=(type(bot_seat_0).__name__, type(bot_seat_1).__name__),
                timed_out=True,
                termination_reason="repetition_cutoff",
                repetition_count=seen_count,
                no_progress_streak=no_progress_streak,
                loop_fallback_triggers_by_seat=_loop_fallback_triggers(bots),
            )

        if state.turn_index >= max_turns:
            return GameResult(
                seed=seed,
                turns=state.turn_index,
                winner=_adjudicate_scores(state),
                final_scores=(state.players[0].score, state.players[1].score),
                bot_seats=(type(bot_seat_0).__name__, type(bot_seat_1).__name__),
                timed_out=True,
                termination_reason="max_turns",
                repetition_count=seen_count,
                no_progress_streak=no_progress_streak,
                loop_fallback_triggers_by_seat=_loop_fallback_triggers(bots),
            )

        legal_actions = env.legal_actions(state)
        if not legal_actions:
            return GameResult(
                seed=seed,
                turns=state.turn_index,
                winner=_adjudicate_scores(state),
                final_scores=(state.players[0].score, state.players[1].score),
                bot_seats=(type(bot_seat_0).__name__, type(bot_seat_1).__name__),
                stalled=True,
                termination_reason="stalled",
                repetition_count=seen_count,
                no_progress_streak=no_progress_streak,
                loop_fallback_triggers_by_seat=_loop_fallback_triggers(bots),
            )

        actor = bots[state.current_player]
        chosen_action = actor.choose_action(env, state, legal_actions)
        if chosen_action is None:
            raise RuntimeError("Bot returned no action in a non-terminal state.")
        next_state = env.step(state, chosen_action)
        if is_progress_transition(state, next_state, chosen_action):
            no_progress_streak = 0
        else:
            no_progress_streak += 1
            if no_progress_limit > 0 and no_progress_streak >= no_progress_limit:
                state = next_state
                return GameResult(
                    seed=seed,
                    turns=state.turn_index,
                    winner=_adjudicate_scores(state),
                    final_scores=(state.players[0].score, state.players[1].score),
                    bot_seats=(type(bot_seat_0).__name__, type(bot_seat_1).__name__),
                    timed_out=True,
                    termination_reason="no_progress_cutoff",
                    repetition_count=seen_count,
                    no_progress_streak=no_progress_streak,
                    loop_fallback_triggers_by_seat=_loop_fallback_triggers(bots),
                )
        state = next_state

    return GameResult(
        seed=seed,
        turns=state.turn_index,
        winner=state.winner,
        final_scores=(state.players[0].score, state.players[1].score),
        bot_seats=(type(bot_seat_0).__name__, type(bot_seat_1).__name__),
        termination_reason="completed",
        repetition_count=state_visit_counts.get(state_signature(state), 0),
        no_progress_streak=no_progress_streak,
        loop_fallback_triggers_by_seat=_loop_fallback_triggers(bots),
    )


def play_match(
    bot_a_factory: BotFactory,
    bot_b_factory: BotFactory,
    config: MatchConfig | None = None,
    seed_start: int = 0,
) -> MatchResult:
    cfg = config or MatchConfig()
    results: list[GameResult] = []

    for game_index in range(cfg.games):
        swap = cfg.swap_seats and (game_index % 2 == 1)
        seat0 = bot_b_factory() if swap else bot_a_factory()
        seat1 = bot_a_factory() if swap else bot_b_factory()
        results.append(
            play_game(
                bot_seat_0=seat0,
                bot_seat_1=seat1,
                seed=seed_start + game_index,
                max_turns=cfg.max_turns_per_game,
                repetition_limit=cfg.repetition_limit,
                no_progress_limit=cfg.no_progress_limit,
            )
        )

    return MatchResult(games=tuple(results))


def _adjudicate_scores(state: SplendorState) -> int | None:
    scores = [player.score for player in state.players]
    best_score = max(scores)
    contenders = [idx for idx, score in enumerate(scores) if score == best_score]
    if len(contenders) == 1:
        return contenders[0]

    fewest_cards = min(len(state.players[idx].purchased_cards) for idx in contenders)
    fewest_card_contenders = [
        idx for idx in contenders if len(state.players[idx].purchased_cards) == fewest_cards
    ]
    return fewest_card_contenders[0] if len(fewest_card_contenders) == 1 else None


def _loop_fallback_triggers(bots: tuple[Bot, Bot]) -> tuple[int, int]:
    return (
        int(getattr(bots[0], "loop_fallback_triggers", 0)),
        int(getattr(bots[1], "loop_fallback_triggers", 0)),
    )
