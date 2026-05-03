"""Replay collection and export utilities for warm-start data generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.bots.base import Bot
from splendor_ai.encoding import ActionCodec, encode_public_observation_tensor
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState


@dataclass(frozen=True, slots=True)
class ReplayStep:
    seed: int
    turn_index: int
    player_id: int
    action_index: int
    action_payload: dict[str, Any]
    legal_action_indices: tuple[int, ...]
    observation_vector: tuple[float, ...]
    final_value: float
    winner: int | None


@dataclass(frozen=True, slots=True)
class ReplayGame:
    seed: int
    winner: int | None
    final_scores: tuple[int, int]
    turns: int
    stalled: bool
    timed_out: bool
    termination_reason: str
    bot_seats: tuple[str, str]
    final_state_snapshot: dict[str, Any]
    steps: tuple[ReplayStep, ...]


def collect_game_replay(
    bot_seat_0: Bot,
    bot_seat_1: Bot,
    seed: int = 0,
    max_turns: int = 400,
    repetition_limit: int = 4,
    no_progress_limit: int = 60,
    codec: ActionCodec | None = None,
) -> ReplayGame:
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    bots = (bot_seat_0, bot_seat_1)
    action_codec = codec or ActionCodec()
    pending_steps: list[dict[str, object]] = []
    stalled = False
    timed_out = False
    termination_reason = "completed"
    state_visit_counts: dict[tuple[object, ...], int] = {}
    no_progress_streak = 0

    while not state.terminal:
        signature = state_signature(state)
        seen_count = state_visit_counts.get(signature, 0) + 1
        state_visit_counts[signature] = seen_count
        if repetition_limit > 0 and seen_count >= repetition_limit:
            timed_out = True
            termination_reason = "repetition_cutoff"
            winner = _adjudicate_scores(state)
            break

        if state.turn_index >= max_turns:
            timed_out = True
            termination_reason = "max_turns"
            winner = _adjudicate_scores(state)
            break

        legal_actions = env.legal_actions(state)
        if not legal_actions:
            stalled = True
            termination_reason = "stalled"
            winner = _adjudicate_scores(state)
            break

        player_id = state.current_player
        observation = encode_public_observation_tensor(state, player_id)
        legal_action_indices = tuple(action_codec.legal_action_indices(state, legal_actions))
        chosen_action = bots[player_id].choose_action(env, state, legal_actions)
        if chosen_action is None:
            raise RuntimeError("Bot returned no action in a non-terminal state.")
        action_index = action_codec.encode(state, chosen_action)
        pending_steps.append(
            {
                "seed": seed,
                "turn_index": state.turn_index,
                "player_id": player_id,
                "action_index": action_index,
                "action_payload": _serialize_action(chosen_action),
                "legal_action_indices": legal_action_indices,
                "observation_vector": observation.vector,
            }
        )
        next_state = env.step(state, chosen_action)
        if is_progress_transition(state, next_state, chosen_action):
            no_progress_streak = 0
        else:
            no_progress_streak += 1
            if no_progress_limit > 0 and no_progress_streak >= no_progress_limit:
                timed_out = True
                termination_reason = "no_progress_cutoff"
                state = next_state
                winner = _adjudicate_scores(state)
                break
        state = next_state
    else:
        winner = state.winner

    steps = tuple(
        ReplayStep(
            seed=int(step["seed"]),
            turn_index=int(step["turn_index"]),
            player_id=int(step["player_id"]),
            action_index=int(step["action_index"]),
            action_payload=dict(step["action_payload"]),
            legal_action_indices=tuple(step["legal_action_indices"]),
            observation_vector=tuple(step["observation_vector"]),
            final_value=_final_value_for_player(winner, int(step["player_id"])),
            winner=winner,
        )
        for step in pending_steps
    )

    return ReplayGame(
        seed=seed,
        winner=winner,
        final_scores=(state.players[0].score, state.players[1].score),
        turns=state.turn_index,
        stalled=stalled,
        timed_out=timed_out,
        termination_reason=termination_reason,
        bot_seats=(type(bot_seat_0).__name__, type(bot_seat_1).__name__),
        final_state_snapshot=_serialize_state_for_diagnostics(state),
        steps=steps,
    )


def export_replay_games_jsonl(path: str | Path, games: list[ReplayGame] | tuple[ReplayGame, ...]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for game in games:
            for step in game.steps:
                payload = {
                    "game_seed": game.seed,
                    "game_turns": game.turns,
                    "game_winner": game.winner,
                    "game_final_scores": list(game.final_scores),
                    "game_stalled": game.stalled,
                    "game_timed_out": game.timed_out,
                    "game_termination_reason": game.termination_reason,
                    "game_bot_seats": list(game.bot_seats),
                    **asdict(step),
                    "legal_action_indices": list(step.legal_action_indices),
                    "observation_vector": list(step.observation_vector),
                }
                handle.write(json.dumps(payload))
                handle.write("\n")


def export_stalled_traces_jsonl(
    path: str | Path,
    games: list[ReplayGame] | tuple[ReplayGame, ...],
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stalled_games = [game for game in games if game.stalled]
    with output_path.open("w", encoding="utf-8") as handle:
        for game in stalled_games:
            payload = {
                "game_seed": game.seed,
                "game_turns": game.turns,
                "game_winner": game.winner,
                "game_final_scores": list(game.final_scores),
                "game_termination_reason": game.termination_reason,
                "game_bot_seats": list(game.bot_seats),
                "final_state_snapshot": game.final_state_snapshot,
                "steps": [
                    {
                        "turn_index": step.turn_index,
                        "player_id": step.player_id,
                        "action_index": step.action_index,
                        "action_payload": step.action_payload,
                        "legal_action_count": len(step.legal_action_indices),
                    }
                    for step in game.steps
                ],
            }
            handle.write(json.dumps(payload))
            handle.write("\n")
    return len(stalled_games)


def export_timed_out_traces_jsonl(
    path: str | Path,
    games: list[ReplayGame] | tuple[ReplayGame, ...],
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timed_out_games = [game for game in games if game.timed_out]
    with output_path.open("w", encoding="utf-8") as handle:
        for game in timed_out_games:
            payload = {
                "game_seed": game.seed,
                "game_turns": game.turns,
                "game_winner": game.winner,
                "game_final_scores": list(game.final_scores),
                "game_termination_reason": game.termination_reason,
                "game_bot_seats": list(game.bot_seats),
                "final_state_snapshot": game.final_state_snapshot,
                "steps": [
                    {
                        "turn_index": step.turn_index,
                        "player_id": step.player_id,
                        "action_index": step.action_index,
                        "action_payload": step.action_payload,
                        "legal_action_count": len(step.legal_action_indices),
                    }
                    for step in game.steps
                ],
            }
            handle.write(json.dumps(payload))
            handle.write("\n")
    return len(timed_out_games)


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


def _final_value_for_player(winner: int | None, player_id: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player_id else -1.0


def _serialize_action(action: Action) -> dict[str, Any]:
    return {
        "action_type": action.action_type.name,
        "tier": action.tier,
        "market_index": action.market_index,
        "reserved_index": action.reserved_index,
        "take_tokens": list(action.take_tokens),
        "return_tokens": list(action.return_tokens),
        "spend_tokens": list(action.spend_tokens),
        "noble_id": action.noble_id,
        "metadata": [list(item) for item in action.metadata],
    }


def _serialize_state_for_diagnostics(state: SplendorState) -> dict[str, Any]:
    return {
        "current_player": state.current_player,
        "turn_index": state.turn_index,
        "pending_round_end": state.pending_round_end,
        "terminal": state.terminal,
        "winner": state.winner,
        "bank_tokens": dict(state.bank_tokens),
        "deck_counts": dict(state.deck_counts),
        "visible_market": {
            str(tier): [card.card_id for card in cards]
            for tier, cards in state.visible_tier_cards.items()
        },
        "available_nobles": [noble.noble_id for noble in state.nobles],
        "players": [
            {
                "player_id": player.player_id,
                "score": player.score,
                "tokens": dict(player.tokens),
                "bonuses": dict(player.bonuses),
                "reserved_card_ids": [card.card_id for card in player.reserved_cards],
                "purchased_card_ids": [card.card_id for card in player.purchased_cards],
                "noble_ids": [noble.noble_id for noble in player.nobles],
            }
            for player in state.players
        ],
    }
