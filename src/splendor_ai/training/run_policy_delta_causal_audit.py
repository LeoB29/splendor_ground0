"""Fresh action-level causal audit of a candidate policy against its parent."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import statistics
from typing import Any

import torch

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.encoding import ActionCodec, canonical_successor_action_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import (
    collect_game_replay,
    deserialize_state_snapshot,
    export_replay_games_jsonl,
    serialize_action,
)
from splendor_ai.training.run_entity_counterfactual_credit import (
    _bootstrap_correlation_ci,
    _bootstrap_mean_ci,
    _pearson,
    _rollout_score_margin,
    _standard_error,
)
from splendor_ai.training.run_league_generation import (
    OPPONENTS,
    _build_opponent,
    _sha256,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


PHASES = ("early", "mid", "late")
_EVALUATION_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class PolicyDeltaCausalAuditConfig:
    output_dir: Path
    data_dir: Path
    parent_checkpoint: Path
    candidate_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    games: int = 640
    states_per_opponent_seat: int = 10
    collection_seed_start: int = 360_000
    continuation_seed_start: int = 34_000_000
    determinizations: int = 3
    continuations_per_determinization: int = 8
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_additional_turns: int = 120
    score_margin_scale: float = 20.0
    minimum_effect_magnitude: float = 0.01
    resolved_effect_magnitude: float = 0.025
    maximum_continuation_cutoff_fraction: float = 0.01
    minimum_split_sign_states: int = 20
    minimum_split_sign_agreement: float = 0.65
    minimum_resolved_fraction: float = 0.10
    bootstrap_samples: int = 20_000
    workers: int = 4
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    device: str = "cpu"


def run_policy_delta_causal_audit(
    config: PolicyDeltaCausalAuditConfig,
) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing causal-audit protocol does not match this run.")
    else:
        _atomic_write(protocol_path, protocol)

    collection = collect_fresh_source_games(config)
    if not collection["passed"]:
        return _finalize(
            config,
            decision="collection_integrity_failed",
            collection=collection,
        )

    selection = select_fresh_disagreements(config)
    if not selection["passed"]:
        return _finalize(
            config,
            decision="insufficient_balanced_disagreements",
            collection=collection,
            selection=selection,
        )

    evaluation = evaluate_selected_disagreements(config, selection)
    return _finalize(
        config,
        decision=evaluation["diagnosis"],
        collection=collection,
        selection=selection,
        evaluation=evaluation,
    )


def collect_fresh_source_games(
    config: PolicyDeltaCausalAuditConfig,
) -> dict[str, Any]:
    summaries: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    for game_index in range(config.games):
        summary_path = _source_summary_path(config, game_index)
        replay_path = _source_replay_path(config, game_index)
        if summary_path.exists() and replay_path.exists():
            summaries[game_index] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        else:
            pending.append(game_index)
    print(
        f"[causal-source] games={config.games} complete={len(summaries)} "
        f"pending={len(pending)} workers={config.workers}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_collect_source_game, config, game_index): game_index
                for game_index in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                game_index = futures[future]
                summaries[game_index] = future.result()
                if completed == 1 or completed % 16 == 0 or completed == len(pending):
                    print(
                        f"[causal-source] complete={len(summaries)}/{config.games}",
                        flush=True,
                    )
    ordered = [summaries[index] for index in range(config.games)]
    by_cell: dict[str, int] = {}
    for row in ordered:
        key = _stratum_cell(str(row["opponent"]), int(row["actor_seat"]))
        by_cell[key] = by_cell.get(key, 0) + 1
    termination_reasons: dict[str, int] = {}
    for row in ordered:
        reason = str(row["termination_reason"])
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    expected_per_cell = config.games // (len(OPPONENTS) * 2)
    payload = {
        "format": "policy_delta_causal_source_v1",
        "games": len(ordered),
        "by_opponent_seat": by_cell,
        "actor_turns": sum(int(row["actor_turns"]) for row in ordered),
        "stalled_games": sum(bool(row["stalled"]) for row in ordered),
        "timed_out_games": sum(bool(row["timed_out"]) for row in ordered),
        "termination_reasons": termination_reasons,
        "loop_fallback_triggers": sum(
            sum(int(value) for value in row["loop_fallback_triggers_by_seat"])
            for row in ordered
        ),
    }
    payload["gates"] = {
        "complete": len(ordered) == config.games,
        "exact_opponent_seat_balance": (
            len(by_cell) == len(OPPONENTS) * 2
            and set(by_cell.values()) == {expected_per_cell}
        ),
        "zero_stalls": payload["stalled_games"] == 0,
        "zero_timeouts": payload["timed_out_games"] == 0,
    }
    payload["passed"] = all(payload["gates"].values())
    _atomic_write(config.output_dir / "collection_summary.json", payload)
    return payload


def select_fresh_disagreements(
    config: PolicyDeltaCausalAuditConfig,
) -> dict[str, Any]:
    selection_path = config.output_dir / "selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    parent = _LegalActionScorer(
        config.parent_checkpoint,
        config.device,
        "legal_action_attention_scorer",
    )
    candidate = _LegalActionScorer(
        config.candidate_checkpoint,
        config.device,
        "legal_action_attention_scorer",
    )
    env = SplendorEnv()
    candidates: list[dict[str, Any]] = []
    rows_scanned = 0
    equivalent_successor_changes = 0
    for game_index in range(config.games):
        summary = json.loads(
            _source_summary_path(config, game_index).read_text(encoding="utf-8")
        )
        actor_seat = int(summary["actor_seat"])
        opponent = str(summary["opponent"])
        with _source_replay_path(config, game_index).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                observation = outcome_blind_replay_projection(json.loads(line))
                if int(observation["player_id"]) != actor_seat:
                    continue
                rows_scanned += 1
                state = deserialize_state_snapshot(observation["state_snapshot"])
                legal_actions = env.legal_actions(state)
                parent_top = parent.score_actions(state, legal_actions)[0]
                candidate_top = candidate.score_actions(state, legal_actions)[0]
                if parent_top.action_index == candidate_top.action_index:
                    continue
                parent_key = canonical_successor_action_key(state, parent_top.action)
                candidate_key = canonical_successor_action_key(state, candidate_top.action)
                if parent_key == candidate_key:
                    equivalent_successor_changes += 1
                    continue
                candidates.append(
                    {
                        "source_game_index": game_index,
                        "source_game_seed": int(observation["game_seed"]),
                        "turn_index": int(observation["turn_index"]),
                        "phase": _phase(int(observation["turn_index"])),
                        "actor_seat": actor_seat,
                        "opponent": opponent,
                        "source_cell": _stratum_cell(opponent, actor_seat),
                        "stratum": _stratum(
                            opponent,
                            actor_seat,
                            _phase(int(observation["turn_index"])),
                        ),
                        "parent_action_index": parent_top.action_index,
                        "parent_action": serialize_action(parent_top.action),
                        "parent_top_score": parent_top.score,
                        "candidate_action_index": candidate_top.action_index,
                        "candidate_action": serialize_action(candidate_top.action),
                        "candidate_top_score": candidate_top.score,
                        "state_snapshot": observation["state_snapshot"],
                    }
                )
    selected, balance = select_balanced_unique_game_disagreements(
        candidates,
        states_per_opponent_seat=config.states_per_opponent_seat,
        seed=config.collection_seed_start + 701,
    )
    expected_cells = {
        _stratum_cell(opponent, seat)
        for opponent in OPPONENTS
        for seat in (0, 1)
    }
    selected_by_cell = {
        cell: sum(row["source_cell"] == cell for row in selected)
        for cell in sorted(expected_cells)
    }
    selected_by_phase = {
        phase: sum(row["phase"] == phase for row in selected) for phase in PHASES
    }
    expected_states = len(expected_cells) * config.states_per_opponent_seat
    payload = {
        "format": "policy_delta_outcome_blind_selection_v1",
        "rows_scanned": rows_scanned,
        "canonical_successor_disagreements": len(candidates),
        "equivalent_successor_changes_excluded": equivalent_successor_changes,
        "eligible_by_opponent_seat": balance["eligible_by_opponent_seat"],
        "eligible_unique_games_by_opponent_seat": balance[
            "eligible_unique_games_by_opponent_seat"
        ],
        "selected_states": len(selected),
        "selected_unique_games": len(
            {int(row["source_game_index"]) for row in selected}
        ),
        "selected_by_opponent_seat": selected_by_cell,
        "selected_by_phase": selected_by_phase,
        "selection_seed": config.collection_seed_start + 701,
        "outcomes_read_for_selection": False,
        "selected": selected,
    }
    payload["gates"] = {
        "enough_balanced_disagreements": len(selected) == expected_states,
        "one_state_per_source_game": payload["selected_unique_games"] == len(selected),
        "exact_opponent_seat_balance": (
            set(selected_by_cell) == expected_cells
            and set(selected_by_cell.values())
            == {config.states_per_opponent_seat}
        ),
        "outcome_blind_selection": not payload["outcomes_read_for_selection"],
    }
    payload["passed"] = all(payload["gates"].values())
    _atomic_write(selection_path, payload)
    return payload


def outcome_blind_replay_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only fields permitted during disagreement-state selection."""
    snapshot = payload.get("state_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("Causal source replay row is missing its state snapshot.")
    return {
        "game_seed": int(payload["game_seed"]),
        "turn_index": int(payload["turn_index"]),
        "player_id": int(payload["player_id"]),
        "state_snapshot": snapshot,
    }


def select_balanced_unique_game_disagreements(
    rows: list[dict[str, Any]],
    *,
    states_per_opponent_seat: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if states_per_opponent_seat <= 0:
        raise ValueError("states_per_opponent_seat must be positive.")
    expected_cells = [
        _stratum_cell(opponent, seat)
        for opponent in OPPONENTS
        for seat in (0, 1)
    ]
    by_cell_game: dict[str, dict[int, list[dict[str, Any]]]] = {
        cell: {} for cell in expected_cells
    }
    for row in rows:
        cell = str(row["source_cell"])
        if cell not in by_cell_game:
            continue
        game_index = int(row["source_game_index"])
        by_cell_game[cell].setdefault(game_index, []).append(row)
    rng = random.Random(seed)
    representative: dict[tuple[str, int], dict[str, Any]] = {}
    adjacency: dict[str, list[int]] = {}
    for cell in expected_cells:
        games = sorted(by_cell_game[cell])
        rng.shuffle(games)
        adjacency[cell] = games
        for game_index in games:
            candidates = sorted(
                by_cell_game[cell][game_index],
                key=lambda row: (
                    PHASES.index(str(row["phase"])),
                    int(row["turn_index"]),
                    int(row["parent_action_index"]),
                    int(row["candidate_action_index"]),
                ),
            )
            representative[(cell, game_index)] = candidates[
                rng.randrange(len(candidates))
            ]
    slots = [
        f"{cell}#slot{slot}"
        for cell in sorted(expected_cells, key=lambda value: len(adjacency[value]))
        for slot in range(states_per_opponent_seat)
    ]
    slot_cell = {slot: slot.rsplit("#slot", 1)[0] for slot in slots}
    game_to_slot: dict[int, str] = {}
    slot_to_game: dict[str, int] = {}

    def assign(slot: str, seen_games: set[int]) -> bool:
        cell = slot_cell[slot]
        for game_index in adjacency[cell]:
            if game_index in seen_games:
                continue
            seen_games.add(game_index)
            previous = game_to_slot.get(game_index)
            if previous is None or assign(previous, seen_games):
                game_to_slot[game_index] = slot
                slot_to_game[slot] = game_index
                return True
        return False

    for slot in slots:
        assign(slot, set())
    selected = [
        representative[(slot_cell[slot], slot_to_game[slot])]
        for slot in slots
        if slot in slot_to_game
    ]
    selected.sort(
        key=lambda row: (
            str(row["stratum"]),
            int(row["source_game_seed"]),
            int(row["turn_index"]),
        )
    )
    return selected, {
        "eligible_by_opponent_seat": {
            cell: sum(str(row["source_cell"]) == cell for row in rows)
            for cell in sorted(expected_cells)
        },
        "eligible_unique_games_by_opponent_seat": {
            cell: len(by_cell_game[cell])
            for cell in sorted(expected_cells)
        },
    }


def evaluate_selected_disagreements(
    config: PolicyDeltaCausalAuditConfig,
    selection: dict[str, Any],
) -> dict[str, Any]:
    progress_path = config.output_dir / "evaluation.progress.jsonl"
    header, completed = _load_progress(progress_path)
    selected = list(selection["selected"])
    expected_header = {
        "format": "policy_delta_causal_evaluation_progress_v1",
        "protocol_sha256": _sha256(config.output_dir / "protocol.json"),
        "selection_sha256": _sha256(config.output_dir / "selection.json"),
        "states": len(selected),
    }
    if header is None:
        progress_path.write_text(
            json.dumps({"header": expected_header}) + "\n", encoding="utf-8"
        )
    elif header != expected_header:
        raise ValueError("Existing causal-evaluation progress contract does not match.")
    missing = [
        (state_index, selected[state_index])
        for state_index in range(len(selected))
        if state_index not in completed
    ]
    print(
        f"[causal-eval] states={len(selected)} complete={len(completed)} "
        f"pending={len(missing)} workers={config.workers}",
        flush=True,
    )
    if missing:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_evaluation_worker,
            initargs=(str(config.parent_checkpoint), config.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_disagreement_state, item, config): item[0]
                for item in missing
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    state_index, row = future.result()
                    completed[state_index] = row
                    handle.write(
                        json.dumps({"state_index": state_index, "row": row}) + "\n"
                    )
                    handle.flush()
                    if len(completed) % 5 == 0 or len(completed) == len(selected):
                        print(
                            f"[causal-eval] complete={len(completed)}/{len(selected)}",
                            flush=True,
                        )
    rows = [completed[index] for index in range(len(selected))]
    summary = summarize_policy_delta_states(rows, config)
    diagnosis = _diagnose(summary)
    payload = {
        "format": "policy_delta_causal_audit_v1",
        "summary": summary,
        "diagnosis": diagnosis,
        "training_allowed": False,
        "promotion_allowed": False,
        "states": rows,
    }
    _atomic_write(config.output_dir / "evaluation.json", payload)
    return payload


def summarize_policy_delta_states(
    rows: list[dict[str, Any]],
    config: PolicyDeltaCausalAuditConfig,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero policy-delta states.")
    full = [float(row["candidate_minus_parent_full"]) for row in rows]
    first = [float(row["candidate_minus_parent_first"]) for row in rows]
    second = [float(row["candidate_minus_parent_second"]) for row in rows]
    eligible_signs = [
        (left, right)
        for left, right in zip(first, second, strict=True)
        if abs(left) >= config.minimum_effect_magnitude
        and abs(right) >= config.minimum_effect_magnitude
    ]
    total = sum(int(row["total_continuations"]) for row in rows)
    terminal = sum(int(row["terminal_continuations"]) for row in rows)
    mean_ci = _bootstrap_mean_ci(
        full,
        config.bootstrap_samples,
        config.continuation_seed_start + 101,
    )
    resolved = [row for row in rows if row["resolved"]]
    overall = {
        "states": len(rows),
        "mean_candidate_minus_parent_return": statistics.fmean(full),
        "mean_candidate_minus_parent_score_margin_points": (
            statistics.fmean(full) * config.score_margin_scale
        ),
        "mean_ci95": mean_ci,
        "mean_score_margin_points_ci95": [
            bound * config.score_margin_scale for bound in mean_ci
        ],
        "split_half_correlation": _pearson(first, second),
        "split_half_correlation_ci95": _bootstrap_correlation_ci(
            first,
            second,
            config.bootstrap_samples,
            config.continuation_seed_start + 103,
        ),
        "split_sign_eligible_states": len(eligible_signs),
        "split_sign_agreement": (
            statistics.fmean((left > 0.0) == (right > 0.0) for left, right in eligible_signs)
            if eligible_signs
            else 0.0
        ),
        "resolved_states": len(resolved),
        "resolved_fraction": len(resolved) / len(rows),
        "resolved_candidate_better": sum(
            float(row["candidate_minus_parent_full"]) > 0.0 for row in resolved
        ),
        "resolved_parent_better": sum(
            float(row["candidate_minus_parent_full"]) < 0.0 for row in resolved
        ),
        "candidate_better_fraction": statistics.fmean(value > 0.0 for value in full),
        "parent_better_fraction": statistics.fmean(value < 0.0 for value in full),
        "tied_fraction": statistics.fmean(value == 0.0 for value in full),
        "terminal_continuations": terminal,
        "total_continuations": total,
        "continuation_cutoffs": total - terminal,
        "continuation_cutoff_fraction": (total - terminal) / total,
    }
    breakdowns: dict[str, dict[str, Any]] = {}
    definitions = {
        "phase": lambda row: str(row["phase"]),
        "opponent": lambda row: str(row["opponent"]),
        "seat": lambda row: f"seat{int(row['actor_seat'])}",
        "opponent_seat_phase": lambda row: str(row["stratum"]),
    }
    for category_index, (name, key_fn) in enumerate(definitions.items()):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(key_fn(row), []).append(row)
        breakdowns[name] = {
            key: _summarize_bucket(
                bucket,
                config,
                seed=config.continuation_seed_start
                + 1_000
                + category_index * 100
                + bucket_index,
            )
            for bucket_index, (key, bucket) in enumerate(sorted(buckets.items()))
        }
    stability = {
        "continuations_complete": (
            overall["continuation_cutoff_fraction"]
            <= config.maximum_continuation_cutoff_fraction
        ),
        "split_correlation_positive": (
            overall["split_half_correlation_ci95"][0] > 0.0
        ),
        "split_signs_replicate": (
            overall["split_sign_eligible_states"] >= config.minimum_split_sign_states
            and overall["split_sign_agreement"] >= config.minimum_split_sign_agreement
        ),
        "resolved_frequency": (
            overall["resolved_fraction"] >= config.minimum_resolved_fraction
        ),
    }
    stability["stable_action_signal"] = all(stability.values())
    return {
        "overall": overall,
        "breakdowns": breakdowns,
        "stability": stability,
    }


def _summarize_bucket(
    rows: list[dict[str, Any]],
    config: PolicyDeltaCausalAuditConfig,
    seed: int,
) -> dict[str, Any]:
    values = [float(row["candidate_minus_parent_full"]) for row in rows]
    ci = _bootstrap_mean_ci(values, config.bootstrap_samples, seed)
    return {
        "states": len(rows),
        "mean_candidate_minus_parent_return": statistics.fmean(values),
        "mean_candidate_minus_parent_score_margin_points": (
            statistics.fmean(values) * config.score_margin_scale
        ),
        "mean_ci95": ci,
        "resolved_states": sum(bool(row["resolved"]) for row in rows),
        "candidate_better_fraction": statistics.fmean(value > 0.0 for value in values),
    }


def _diagnose(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    stable = bool(summary["stability"]["stable_action_signal"])
    low, high = (float(value) for value in overall["mean_ci95"])
    if stable and high < 0.0:
        return "stable_local_harm_close_awr"
    if stable and low > 0.0:
        return "stable_local_benefit_test_compounding"
    return "local_effect_unresolved_close_awr"


def _collect_source_game(
    config: PolicyDeltaCausalAuditConfig,
    game_index: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    opponent, actor_seat = _source_assignment(game_index)
    actor = LegalActionAttentionScorerBot(config.parent_checkpoint, device=config.device)
    opponent_bot = _build_opponent(config, opponent)
    bots = (actor, opponent_bot) if actor_seat == 0 else (opponent_bot, actor)
    replay = collect_game_replay(
        bots[0],
        bots[1],
        seed=config.collection_seed_start + game_index,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
        include_state_snapshots=True,
    )
    replay_path = _source_replay_path(config, game_index)
    temporary = replay_path.with_suffix(replay_path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    export_replay_games_jsonl(temporary, [replay])
    os.replace(temporary, replay_path)
    payload = {
        "game_index": game_index,
        "seed": config.collection_seed_start + game_index,
        "opponent": opponent,
        "actor_seat": actor_seat,
        "actor_turns": sum(step.player_id == actor_seat for step in replay.steps),
        "winner": replay.winner,
        "final_scores": list(replay.final_scores),
        "turns": replay.turns,
        "stalled": replay.stalled,
        "timed_out": replay.timed_out,
        "termination_reason": replay.termination_reason,
        "loop_fallback_triggers_by_seat": list(replay.loop_fallback_triggers_by_seat),
    }
    _atomic_write(_source_summary_path(config, game_index), payload)
    return payload


def _initialize_evaluation_worker(parent_checkpoint: str, device: str) -> None:
    global _EVALUATION_SCORER
    torch.set_num_threads(1)
    _EVALUATION_SCORER = _LegalActionScorer(
        Path(parent_checkpoint),
        device,
        "legal_action_attention_scorer",
    )


def _evaluate_disagreement_state(
    item: tuple[int, dict[str, Any]],
    config: PolicyDeltaCausalAuditConfig,
) -> tuple[int, dict[str, Any]]:
    if _EVALUATION_SCORER is None:
        raise RuntimeError("Causal-audit worker scorer was not initialized.")
    state_index, payload = item
    state = deserialize_state_snapshot(payload["state_snapshot"])
    env = SplendorEnv()
    codec = ActionCodec()
    parent_index = int(payload["parent_action_index"])
    candidate_index = int(payload["candidate_action_index"])
    first_deltas: list[float] = []
    second_deltas: list[float] = []
    terminal = 0
    total = 0
    half = config.continuations_per_determinization // 2
    for determinization_index in range(config.determinizations):
        determinized = determinize_for_observer(
            state,
            int(payload["actor_seat"]),
            random.Random(
                config.continuation_seed_start
                + state_index * 1_000_003
                + determinization_index
            ),
        )
        action_by_index = {
            codec.encode(determinized, action): action
            for action in env.legal_actions(determinized)
        }
        if parent_index not in action_by_index or candidate_index not in action_by_index:
            raise RuntimeError("Selected root action is absent after legal determinization.")
        for continuation_index in range(config.continuations_per_determinization):
            seed = (
                config.continuation_seed_start
                + state_index * 10_000_019
                + determinization_index * 100_003
                + continuation_index
            )
            parent_result = _rollout_score_margin(
                env,
                _EVALUATION_SCORER,
                determinized,
                action_by_index[parent_index],
                int(payload["actor_seat"]),
                seed,
                config.continuation_epsilon,
                config.continuation_temperature,
                config.continuation_top_k,
                config.continuation_max_additional_turns,
                config.score_margin_scale,
            )
            candidate_result = _rollout_score_margin(
                env,
                _EVALUATION_SCORER,
                determinized,
                action_by_index[candidate_index],
                int(payload["actor_seat"]),
                seed,
                config.continuation_epsilon,
                config.continuation_temperature,
                config.continuation_top_k,
                config.continuation_max_additional_turns,
                config.score_margin_scale,
            )
            delta = candidate_result.value - parent_result.value
            (first_deltas if continuation_index < half else second_deltas).append(delta)
            terminal += int(parent_result.terminal) + int(candidate_result.terminal)
            total += 2
    full_deltas = first_deltas + second_deltas
    full_mean = statistics.fmean(full_deltas)
    se = _standard_error(full_deltas)
    row = {
        **{
            key: payload[key]
            for key in (
                "source_game_index",
                "source_game_seed",
                "turn_index",
                "phase",
                "actor_seat",
                "opponent",
                "source_cell",
                "stratum",
                "parent_action_index",
                "parent_action",
                "candidate_action_index",
                "candidate_action",
            )
        },
        "candidate_minus_parent_first": statistics.fmean(first_deltas),
        "candidate_minus_parent_second": statistics.fmean(second_deltas),
        "candidate_minus_parent_full": full_mean,
        "paired_standard_error": se,
        "resolved": (
            abs(full_mean) >= config.resolved_effect_magnitude
            and abs(full_mean) > 1.96 * se
        ),
        "terminal_continuations": terminal,
        "total_continuations": total,
    }
    return state_index, row


def _source_assignment(game_index: int) -> tuple[str, int]:
    return OPPONENTS[game_index % len(OPPONENTS)], (
        game_index // len(OPPONENTS)
    ) % 2


def _phase(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 40:
        return "mid"
    return "late"


def _stratum_cell(opponent: str, actor_seat: int) -> str:
    return f"{opponent}:seat{actor_seat}"


def _stratum(opponent: str, actor_seat: int, phase: str) -> str:
    return f"{_stratum_cell(opponent, actor_seat)}:{phase}"


def _source_replay_path(config: PolicyDeltaCausalAuditConfig, game_index: int) -> Path:
    return config.data_dir / "raw" / f"game_{game_index:04d}.jsonl"


def _source_summary_path(config: PolicyDeltaCausalAuditConfig, game_index: int) -> Path:
    return config.data_dir / "raw" / f"game_{game_index:04d}.summary.json"


def _load_progress(
    path: Path,
) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    if not path.exists():
        return None, {}
    header = None
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "header" in payload:
                header = payload["header"]
            elif "state_index" in payload:
                rows[int(payload["state_index"])] = payload["row"]
    if header is None:
        raise ValueError("Evaluation progress file has no header.")
    return header, rows


def _protocol(config: PolicyDeltaCausalAuditConfig) -> dict[str, Any]:
    return {
        "experiment": "policy_delta_causal_audit_034",
        "config": _config_payload(config),
        "checkpoint_hashes": {
            "035_parent": _sha256(config.parent_checkpoint),
            "033_candidate": _sha256(config.candidate_checkpoint),
            "016": _sha256(config.checkpoint_016),
            "023": _sha256(config.checkpoint_023),
        },
        "opponents": list(OPPONENTS),
        "selection": (
            "canonical-successor top-action disagreements only; outcomes are not read; "
            "unique source games and exact opponent-seat balance; phase is measured, "
            "not forced"
        ),
        "causal_comparison": (
            "033 top action minus 035 top action under identical observer-relative "
            "determinizations and matched 035 continuation seeds"
        ),
        "statistical_unit": "one selected state from one unique source game",
        "training_allowed": False,
        "promotion_allowed": False,
        "stage_order": ["collection", "outcome_blind_selection", "matched_evaluation"],
    }


def _config_payload(config: PolicyDeltaCausalAuditConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _finalize(
    config: PolicyDeltaCausalAuditConfig,
    *,
    decision: str,
    **stages: Any,
) -> dict[str, Any]:
    payload = {
        "gate": "policy_delta_causal_audit_034",
        "decision": decision,
        "diagnostic_only": True,
        "training_allowed": False,
        "promotion_allowed": False,
        **stages,
    }
    _atomic_write(config.output_dir / "audit_final.json", payload)
    print(
        json.dumps(
            {
                "decision": decision,
                "training_allowed": False,
                "promotion_allowed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _validate(config: PolicyDeltaCausalAuditConfig) -> None:
    if config.games <= 0 or config.games % (len(OPPONENTS) * 2):
        raise ValueError("games must balance exactly across opponent and seat cells.")
    if config.states_per_opponent_seat <= 0:
        raise ValueError("states_per_opponent_seat must be positive.")
    if config.determinizations <= 0:
        raise ValueError("determinizations must be positive.")
    if (
        config.continuations_per_determinization < 2
        or config.continuations_per_determinization % 2
    ):
        raise ValueError("continuations_per_determinization must be positive and even.")
    if min(config.workers, config.bootstrap_samples) <= 0:
        raise ValueError("workers and bootstrap_samples must be positive.")
    if not 0.0 <= config.continuation_epsilon <= 1.0:
        raise ValueError("continuation_epsilon must be in [0, 1].")
    if config.continuation_temperature <= 0.0 or config.continuation_top_k <= 0:
        raise ValueError("continuation temperature and top-k must be positive.")
    if config.minimum_split_sign_states <= 0:
        raise ValueError("minimum_split_sign_states must be positive.")
    if not 0.0 <= config.minimum_split_sign_agreement <= 1.0:
        raise ValueError("minimum_split_sign_agreement must be in [0, 1].")
    for path in (
        config.parent_checkpoint,
        config.candidate_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
