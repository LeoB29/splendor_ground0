"""Generate the preregistered counterfactual target/action corpus for 081."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import torch

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.card_target_executor import target_executor_action
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, SplendorState
from splendor_ai.training.card_option_q import enumerate_option_action_pairs
from splendor_ai.training.replay import serialize_state_snapshot, deserialize_state_snapshot
from splendor_ai.training.run_action_value_training import _spearman
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer
from splendor_ai.training.run_rollout_action_credit import _sample_scored_action


@dataclass(frozen=True, slots=True)
class CardOptionCorpusConfig:
    checkpoint: Path
    output_path: Path
    progress_path: Path
    source_seed_start: int = 781_500
    source_games: int = 1_200
    selected_states: int = 1_000
    train_states: int = 600
    validation_states: int = 200
    confirmation_states: int = 200
    source_epsilon: float = 0.25
    determinizations: int = 2
    continuations_per_determinization: int = 4
    continuation_epsilon: float = 0.10
    continuation_temperature: float = 0.50
    continuation_top_k: int = 3
    max_additional_turns: int = 120
    bootstrap_samples: int = 20_000
    split_seed: int = 781_081
    workers: int = 10
    device: str = "cpu"


_WORKER_SCORER: _LegalActionScorer | None = None


def build_card_option_corpus(config: CardOptionCorpusConfig) -> dict[str, Any]:
    _validate_config(config)
    contract = _config_payload(config)
    header, completed = _load_progress(config.progress_path)
    if header is None:
        scorer = _LegalActionScorer(config.checkpoint, config.device, "legal_action_attention_scorer")
        selected, source = _collect_source_states(config, scorer)
        header = {"contract": contract, "selected_states": selected, "source": source}
        config.progress_path.parent.mkdir(parents=True, exist_ok=True)
        config.progress_path.write_text(json.dumps({"header": header}) + "\n", encoding="utf-8")
    elif header.get("contract") != contract:
        raise ValueError("Existing 081 corpus progress has a different contract.")
    selected = header["selected_states"]
    missing = [(index, row) for index, row in enumerate(selected) if index not in completed]
    print(f"[081:corpus] selected={len(selected)} completed={len(completed)} missing={len(missing)}", flush=True)
    if missing:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_worker,
            initargs=(str(config.checkpoint), config.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_saved_state, item, config): item[0]
                for item in missing
            }
            with config.progress_path.open("a", encoding="utf-8") as handle:
                for done, future in enumerate(as_completed(futures), 1):
                    state_index, row = future.result()
                    completed[state_index] = row
                    handle.write(json.dumps({"state_index": state_index, "row": row}) + "\n")
                    handle.flush()
                    total = len(completed)
                    if done % 5 == 0 or total == len(selected):
                        print(f"[081:corpus] completed={total}/{len(selected)}", flush=True)
    rows = [completed[index] for index in range(len(selected))]
    summary = _summarize(rows, config)
    exact_coverage = all(
        row["pair_count"] == row["target_count"] * row["action_count"]
        and all(pair["total_continuations"] == config.determinizations * config.continuations_per_determinization for pair in row["pairs"])
        for row in rows
    )
    source = header["source"]
    split_counts = Counter(row["split"] for row in rows)
    phase_counts = Counter(row["phase"] for row in rows)
    seat_counts = Counter(str(row["actor_seat"]) for row in rows)
    gates = {
        "exact_state_split_phase_seat_coverage": (
            len(rows) == config.selected_states
            and split_counts == {"train": config.train_states, "validation": config.validation_states, "confirmation": config.confirmation_states}
            and phase_counts == {"early": 334, "mid": 333, "late": 333}
            and seat_counts == {"0": 500, "1": 500}
            and source["selected_games"] == config.selected_states
        ),
        "exact_target_action_determinization_coverage": exact_coverage,
        "zero_illegal_null_errors": summary["errors"] == 0,
        "continuation_cutoff_fraction_safe": summary["cutoff_fraction"] <= 0.01,
        "split_half_rank_signal_positive": summary["split_spearman_ci95"][0] > 0.0,
    }
    payload = {
        "corpus": "card_option_q_counterfactual_081_v1",
        "contract": contract,
        "source": source,
        "split_counts": dict(split_counts),
        "phase_counts": dict(phase_counts),
        "seat_counts": dict(seat_counts),
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "human_data_used": False,
        "states": rows,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _collect_source_states(
    config: CardOptionCorpusConfig,
    scorer: _LegalActionScorer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    desired = (
        [("early", 0)] * 167 + [("early", 1)] * 167
        + [("mid", 0)] * 167 + [("mid", 1)] * 166
        + [("late", 0)] * 166 + [("late", 1)] * 167
    )
    remaining = Counter(desired)
    rng = random.Random(config.split_seed + 17)
    selected: list[dict[str, Any]] = []
    clean_games = 0
    for offset in range(config.source_games):
        seed = config.source_seed_start + offset
        env = SplendorEnv(seed=seed)
        state = env.initial_state()
        candidates: dict[tuple[str, int], list[SplendorState]] = {}
        seen: Counter[tuple[object, ...]] = Counter()
        no_progress = 0
        clean = True
        while not state.terminal and state.turn_index < 400:
            signature = state_signature(state)
            seen[signature] += 1
            if seen[signature] >= 4 or no_progress >= 60:
                clean = False
                break
            key = (_phase(state.turn_index), state.current_player)
            candidates.setdefault(key, []).append(state)
            actions = list(actions_by_canonical_successor_key(state, env.legal_actions(state)).values())
            scores = scorer.score_actions(state, actions)
            action_rng = random.Random(config.source_seed_start * 1_000_003 + seed * 10_007 + state.turn_index)
            if action_rng.random() < config.source_epsilon:
                action = actions[action_rng.randrange(len(actions))]
            else:
                action = scores[0].action
            next_state = env.step(state, action)
            no_progress = 0 if is_progress_transition(state, next_state, action) else no_progress + 1
            state = next_state
        clean_games += int(clean and state.terminal)
        available = [key for key, count in remaining.items() if count > 0 and candidates.get(key)]
        if available:
            available.sort()
            key = available[rng.randrange(len(available))]
            rows = candidates[key]
            chosen = rows[rng.randrange(len(rows))]
            selected.append(
                {
                    "source_game_seed": seed,
                    "phase": key[0],
                    "actor_seat": key[1],
                    "state_snapshot": serialize_state_snapshot(chosen),
                }
            )
            remaining[key] -= 1
        if len(selected) == config.selected_states:
            break
    if len(selected) != config.selected_states or any(remaining.values()):
        raise RuntimeError(f"Source games could not fill frozen phase/seat cells: {dict(remaining)}")
    indices = list(range(len(selected)))
    random.Random(config.split_seed).shuffle(indices)
    split_by_index: dict[int, str] = {}
    boundaries = (
        ("train", config.train_states),
        ("validation", config.validation_states),
        ("confirmation", config.confirmation_states),
    )
    cursor = 0
    for split, count in boundaries:
        for index in indices[cursor : cursor + count]:
            split_by_index[index] = split
        cursor += count
    for index, row in enumerate(selected):
        row["split"] = split_by_index[index]
    return selected, {
        "source_games_requested": config.source_games,
        "source_games_consumed": offset + 1,
        "clean_source_games": clean_games,
        "selected_games": len(selected),
        "outcomes_stored_or_used": False,
    }


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(Path(checkpoint), device, "legal_action_attention_scorer")


def _evaluate_saved_state(
    item: tuple[int, dict[str, Any]],
    config: CardOptionCorpusConfig,
) -> tuple[int, dict[str, Any]]:
    state_index, selected = item
    if _WORKER_SCORER is None:
        _initialize_worker(str(config.checkpoint), config.device)
    assert _WORKER_SCORER is not None
    state = deserialize_state_snapshot(selected["state_snapshot"])
    env = SplendorEnv()
    pairs = enumerate_option_action_pairs(state, env.legal_actions(state))
    codec = ActionCodec()
    baseline_action = _WORKER_SCORER.score_actions(
        state,
        list(actions_by_canonical_successor_key(state, env.legal_actions(state)).values()),
    )[0]
    pair_rows = []
    errors = 0
    for pair_index, pair in enumerate(pairs):
        first: list[float] = []
        second: list[float] = []
        terminal = 0
        for det_index in range(config.determinizations):
            det = determinize_for_observer(
                state,
                state.current_player,
                random.Random(config.split_seed + state_index * 1_000_003 + det_index),
            )
            det_actions = list(actions_by_canonical_successor_key(det, env.legal_actions(det)).values())
            action_by_index = {codec.encode(det, action): action for action in det_actions}
            root_action = action_by_index.get(pair.action_index)
            if root_action is None:
                errors += 1
                continue
            for continuation_index in range(config.continuations_per_determinization):
                value, complete = _rollout_option_share(
                    env,
                    _WORKER_SCORER,
                    det,
                    root_action,
                    pair.target.key,
                    state.current_player,
                    seed=(
                        config.split_seed
                        + state_index * 10_000_019
                        + det_index * 100_003
                        + continuation_index
                    ),
                    config=config,
                )
                (first if continuation_index < 2 else second).append(value)
                terminal += int(complete)
        values = first + second
        pair_rows.append(
            {
                "target_key": pair.target.key,
                "target_source": pair.target.source,
                "action_index": pair.action_index,
                "first_half_mean": statistics.fmean(first) if first else 0.0,
                "second_half_mean": statistics.fmean(second) if second else 0.0,
                "full_mean": statistics.fmean(values) if values else 0.0,
                "terminal_continuations": terminal,
                "total_continuations": len(values),
                "baseline": pair.target.key == "CLEAR" and pair.action_index == baseline_action.action_index,
            }
        )
    first_values = [row["first_half_mean"] for row in pair_rows]
    second_values = [row["second_half_mean"] for row in pair_rows]
    return state_index, {
        **selected,
        "target_count": len({row["target_key"] for row in pair_rows}),
        "action_count": len({row["action_index"] for row in pair_rows}),
        "pair_count": len(pair_rows),
        "split_spearman": _spearman(first_values, second_values),
        "errors": errors,
        "pairs": pair_rows,
    }


def _rollout_option_share(
    env: SplendorEnv,
    scorer: _LegalActionScorer,
    state: SplendorState,
    root_action: Action,
    target_key: str,
    root_player: int,
    seed: int,
    config: CardOptionCorpusConfig,
) -> tuple[float, bool]:
    rng = random.Random(seed)
    rollout = env.step(state, root_action)
    seen: Counter[tuple[object, ...]] = Counter()
    no_progress = 0
    end_turn = rollout.turn_index + config.max_additional_turns
    while not rollout.terminal and rollout.turn_index < end_turn:
        signature = state_signature(rollout)
        seen[signature] += 1
        if seen[signature] >= 4 or no_progress >= 60:
            break
        actions = env.legal_actions(rollout)
        scores = scorer.score_actions(rollout, actions)
        live_target = _find_target(rollout, root_player, target_key)
        if rollout.current_player == root_player and live_target is not None:
            action = _target_executor_action(rollout, actions, scores, live_target)
        else:
            action = _sample_scored_action(
                scores,
                rng=rng,
                epsilon=config.continuation_epsilon,
                temperature=config.continuation_temperature,
                top_k=config.continuation_top_k,
            )
        next_state = env.step(rollout, action)
        no_progress = 0 if is_progress_transition(rollout, next_state, action) else no_progress + 1
        rollout = next_state
    winner = rollout.winner if rollout.terminal else _adjudicate(rollout)
    share = 0.5 if winner is None else float(winner == root_player)
    return share, rollout.terminal


def _find_target(state: SplendorState, player_id: int, key: str) -> Card | None:
    if key == "CLEAR":
        return None
    player = state.players[player_id]
    if any(card.card_id == key for card in player.purchased_cards):
        return None
    for card in player.reserved_cards:
        if card.card_id == key:
            return card
    for cards in state.visible_tier_cards.values():
        for card in cards:
            if card.card_id == key:
                return card
    return None


def _target_executor_action(
    state: SplendorState,
    actions: list[Action],
    scores: tuple[Any, ...],
    target: Card,
) -> Action:
    return target_executor_action(state, actions, scores, target)


def _action_card(state: SplendorState, action: Action) -> Card | None:
    if action.action_type in (ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE):
        return state.visible_tier_cards[action.tier][action.market_index]
    if action.action_type is ActionType.BUY_RESERVED:
        return state.players[state.current_player].reserved_cards[action.reserved_index]
    return None


def _colour_shortfall(card: Card, tokens: dict[str, int], bonuses: dict[str, int]) -> int:
    return sum(max(0, card.cost.get(color, 0) - bonuses.get(color, 0) - tokens.get(color, 0)) for color in card.cost)


def _adjudicate(state: SplendorState) -> int | None:
    best = max(player.score for player in state.players)
    contenders = [index for index, player in enumerate(state.players) if player.score == best]
    if len(contenders) == 1:
        return contenders[0]
    fewest = min(len(state.players[index].purchased_cards) for index in contenders)
    tied = [index for index in contenders if len(state.players[index].purchased_cards) == fewest]
    return tied[0] if len(tied) == 1 else None


def _summarize(rows: list[dict[str, Any]], config: CardOptionCorpusConfig) -> dict[str, Any]:
    spearmans = [float(row["split_spearman"]) for row in rows]
    total = sum(pair["total_continuations"] for row in rows for pair in row["pairs"])
    terminal = sum(pair["terminal_continuations"] for row in rows for pair in row["pairs"])
    return {
        "states": len(rows),
        "pairs": sum(row["pair_count"] for row in rows),
        "continuations": total,
        "cutoffs": total - terminal,
        "cutoff_fraction": (total - terminal) / max(total, 1),
        "errors": sum(row["errors"] for row in rows),
        "mean_split_spearman": statistics.fmean(spearmans),
        "split_spearman_ci95": _bootstrap_ci(spearmans, config.bootstrap_samples, config.split_seed + 901),
    }


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def _load_progress(path: Path) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    if not path.exists():
        return None, {}
    header = None
    completed: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "header" in payload:
                header = payload["header"]
            elif "state_index" in payload:
                completed[int(payload["state_index"])] = payload["row"]
    return header, completed


def _phase(turn: int) -> str:
    return "early" if turn < 20 else "mid" if turn < 40 else "late"


def _config_payload(config: CardOptionCorpusConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _validate_config(config: CardOptionCorpusConfig) -> None:
    if config.selected_states != config.train_states + config.validation_states + config.confirmation_states:
        raise ValueError("Split counts must sum to selected_states.")
    if config.selected_states != 1_000 or config.source_games != 1_200:
        raise ValueError("081 requires exactly 1,000 states from 1,200 source games.")
    if (config.determinizations, config.continuations_per_determinization) != (2, 4):
        raise ValueError("081 requires two determinizations and four continuations each.")
    if config.workers <= 0:
        raise ValueError("workers must be positive.")
