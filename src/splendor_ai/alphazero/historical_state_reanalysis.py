"""Population-anchored historical-state reanalysis pilot 129."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Iterator

import torch

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import encode_canonical_state_entities
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import serialize_action

from .arena import ArenaResult, alpha_zero_search_factory, run_paired_arena
from .high_budget_distillation import compression_assessment, distillation_training_config
from .indexed_training import (
    IndexedReplayDataset,
    index_replay_paths,
    make_indexed_loader,
    train_parent_safe_indexed_successor,
)
from .network import AlphaZeroInference, load_checkpoint
from .replay import AlphaZeroReplayRow, _row_from_payload, replay_row_from_search
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearch, SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .selfplay import _adjudicate_scores
from .training import _run_epoch


@dataclass(frozen=True, slots=True)
class HistoricalStateReanalysisConfig:
    output_dir: Path
    result_128: Path
    generation_11_checkpoint: Path
    terminal_control_checkpoint: Path
    installed_checkpoint: Path
    source_replays: tuple[Path, Path, Path, Path]
    ordinary_holdout: Path
    device: str = "cpu"
    torch_threads: int = 1
    collection_workers: int = 4
    arena_workers: int = 4
    max_turns: int = 200


PROTOCOL_SHA256 = "1ed643ce84b788eebf555f3d0679fea145d181e01ea526d719f3d61b6b63e5a4"
SOURCE_GENERATIONS = (2, 5, 8, 11)
EXPECTED_HASHES = {
    "result_128": "21e506ea5792543330da90a7cd2880c0e85b1c7d46a721ade1ac57028a5c9170",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "terminal_control_checkpoint": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "source_generation_2": "855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56",
    "source_generation_5": "0f9393fb020cf01c4d06f829c03e7426913dcc29d6afde49091cb9def57772e1",
    "source_generation_8": "96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10",
    "source_generation_11": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "ordinary_holdout": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
}
PHASES = ("early", "middle", "late")
PHASE_FRACTIONS = {"early": 0.25, "middle": 0.50, "late": 0.75}
TRAIN_ACCEPTED_PER_SOURCE = 64
TRAIN_ATTEMPTS_PER_SOURCE = 80
HOLDOUT_ORDINAL_START = 96
HOLDOUT_GAMES_PER_SOURCE = 16
HISTORICAL_HOLDOUT_STATES = 64
SELFPLAY_SIMULATIONS = 1_024
SELFPLAY_SEARCH_SEED = 1_129_000
CONTROL_ROWS = 13_960
ORDINARY_ARENA_SEED_START = 2_720_000
ORDINARY_PAIRED_SEEDS = 200
ARENA_SHARDS = 4
BOOTSTRAP_SAMPLES = 50_000


def reanalysis_training_config():
    return replace(distillation_training_config(), seed=1_129_000)


def phase_for_slot(slot: int) -> str:
    return PHASES[slot % len(PHASES)]


def nearest_eligible_index(pending_round_end: list[bool], phase: str) -> int:
    if phase not in PHASE_FRACTIONS or not pending_round_end:
        raise ValueError("Unknown phase or empty trajectory.")
    eligible = [index for index, pending in enumerate(pending_round_end) if not pending]
    if not eligible:
        raise ValueError("Historical game has no eligible starting state.")
    target = PHASE_FRACTIONS[phase] * (len(pending_round_end) - 1)
    return min(eligible, key=lambda index: (abs(index - target), index))


def select_anchor_game_count(game_lengths: list[int], historical_rows: int) -> dict[str, int]:
    target = CONTROL_ROWS - historical_rows
    if target <= 0:
        raise ValueError("Historical replay alone exceeds the matched control size.")
    cumulative = 0
    selected = 0
    for length in game_lengths:
        before = cumulative
        after = cumulative + int(length)
        if after >= target:
            if abs(after - target) < abs(before - target):
                cumulative = after
                selected += 1
            break
        cumulative = after
        selected += 1
    if selected == 0:
        raise ValueError("Matched anchor selection chose no ordinary game.")
    return {
        "selected_games": selected,
        "ordinary_rows": cumulative,
        "historical_rows": historical_rows,
        "combined_rows": historical_rows + cumulative,
        "control_rows": CONTROL_ROWS,
        "absolute_row_difference": abs(historical_rows + cumulative - CONTROL_ROWS),
        "maximum_selected_game_rows": max(game_lengths[:selected]),
    }


def reanalysis_offline_qualification(
    *, training: dict[str, Any], metrics: dict[str, dict[str, float]]
) -> dict[str, Any]:
    parent = metrics["generation_11"]
    control = metrics["terminal_control_114"]
    candidate = metrics["historical_reanalysis_candidate"]
    finite = all(
        math.isfinite(float(value))
        for model_metrics in metrics.values()
        for value in model_metrics.values()
        if isinstance(value, int | float)
    )
    checks = {
        "nonzero_epoch_selected": int(training.get("selected_epoch", 0)) > 0,
        "all_metrics_finite": finite,
        "policy_improves_parent": candidate["policy_loss"] < parent["policy_loss"],
        "objective_improves_parent": candidate["objective_loss"] < parent["objective_loss"],
        "value_within_0_025_of_parent": candidate["value_loss"] <= parent["value_loss"] + 0.025,
        "policy_within_0_01_of_control": candidate["policy_loss"] <= control["policy_loss"] + 0.01,
        "objective_within_0_02_of_control": candidate["objective_loss"] <= control["objective_loss"] + 0.02,
        "value_within_0_025_of_control": candidate["value_loss"] <= control["value_loss"] + 0.025,
    }
    return {"qualified": all(checks.values()), "checks": checks}


def reanalysis_decision(
    *, offline_passed: bool, historical_passed: bool | None, ordinary_passed: bool | None
) -> str:
    if not offline_passed:
        return "historical_reanalysis_offline_failed_no_gameplay_no_install"
    if historical_passed is not True:
        return "historical_reanalysis_failed_diverse_state_gameplay_no_install"
    if ordinary_passed is not True:
        return "historical_reanalysis_diverse_gain_not_ordinary_robust_no_install"
    return "historical_reanalysis_transfers_authorize_population_loop_design_no_install"


def run_historical_state_reanalysis_129(
    config: HistoricalStateReanalysisConfig, *, launch: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = _preflight(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "preflight.json", preflight)
    if not launch:
        return preflight
    sealed = config.output_dir / "result.json"
    if sealed.exists():
        payload = json.loads(sealed.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 129 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_historical_state_reanalysis_129",
        "status": "running",
        "active_stage": "historical_continuation_collection",
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight": preflight,
        "historical_collection": None,
        "mixed_replay": None,
        "training": None,
        "ordinary_holdout": None,
        "panels": {},
        "external_035_arena": None,
        "promotion_authorized": False,
        "automatic_continuation": False,
        "installed_checkpoint_unchanged": True,
        "decision": "running",
    }
    _write_progress(config.output_dir, result, started)
    collection = _run_or_load_historical_collection(config)
    result["historical_collection"] = collection
    result["active_stage"] = "matched_mixed_replay"
    _write_progress(config.output_dir, result, started)
    mixed = _run_or_load_mixed_replay(config, collection)
    result["mixed_replay"] = mixed
    result["active_stage"] = "parent_safe_mixed_training"
    _write_progress(config.output_dir, result, started)
    training = _run_or_load_training(config, mixed)
    result["training"] = training
    result["active_stage"] = "ordinary_holdout_gate"
    _write_progress(config.output_dir, result, started)
    holdout = _run_or_load_holdout(config, training)
    result["ordinary_holdout"] = holdout
    _write_progress(config.output_dir, result, started)
    if not holdout["qualification"]["qualified"]:
        return _seal(config, result, started, reanalysis_decision(
            offline_passed=False, historical_passed=None, ordinary_passed=None
        ))
    candidate = Path(training["child_checkpoint"])
    result["active_stage"] = "historical_start_gameplay"
    _write_progress(config.output_dir, result, started)
    historical = _run_or_load_historical_panel(config, candidate)
    result["panels"]["historical_start"] = historical
    _write_progress(config.output_dir, result, started)
    if not historical["assessment"]["passed"]:
        return _seal(config, result, started, reanalysis_decision(
            offline_passed=True, historical_passed=False, ordinary_passed=None
        ))
    result["active_stage"] = "ordinary_initial_state_gameplay"
    _write_progress(config.output_dir, result, started)
    ordinary = _run_or_load_ordinary_panel(config, candidate)
    result["panels"]["ordinary_initial_state"] = ordinary
    _write_progress(config.output_dir, result, started)
    return _seal(config, result, started, reanalysis_decision(
        offline_passed=True,
        historical_passed=True,
        ordinary_passed=bool(ordinary["assessment"]["passed"]),
    ))


def _preflight(config: HistoricalStateReanalysisConfig) -> dict[str, Any]:
    if len(config.source_replays) != 4:
        raise ValueError("129 requires four exact source replays.")
    observed = {
        "result_128": _sha256(config.result_128),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "terminal_control_checkpoint": _sha256(config.terminal_control_checkpoint),
        "installed_checkpoint": _sha256(config.installed_checkpoint),
        **{
            f"source_generation_{generation}": _sha256(path)
            for generation, path in zip(SOURCE_GENERATIONS, config.source_replays, strict=True)
        },
        "ordinary_holdout": _sha256(config.ordinary_holdout),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"129 frozen input mismatch: {observed}")
    result_128 = json.loads(config.result_128.read_text(encoding="utf-8"))
    if result_128.get("decision") != "search_value_blend_offline_failed_no_gameplay_no_install":
        raise ValueError("129 does not reproduce finalized experiment 128.")
    _, parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    _, control = load_checkpoint(config.terminal_control_checkpoint, device=config.device)
    if (
        parent.generation != 11
        or control.generation != 12
        or control.metadata.get("parent_generation") != 11
        or config.torch_threads != 1
        or config.collection_workers != 4
        or config.arena_workers != 4
        or config.max_turns != 200
    ):
        raise ValueError("129 checkpoint lineage or execution contract mismatch.")
    source_summaries = []
    holdout_schedule = []
    smoke_state: tuple[SplendorEnv, SplendorState, dict[str, Any]] | None = None
    for source_index, (generation, path) in enumerate(
        zip(SOURCE_GENERATIONS, config.source_replays, strict=True)
    ):
        summary, schedule, first_state = _preflight_source(path, generation, source_index)
        source_summaries.append(summary)
        holdout_schedule.extend(schedule)
        if smoke_state is None:
            smoke_state = first_state
    if len(holdout_schedule) != HISTORICAL_HOLDOUT_STATES or smoke_state is None:
        raise ValueError("129 preflight did not construct its exact state schedules.")
    env, state, identity = smoke_state
    inference = AlphaZeroInference(config.generation_11_checkpoint, device=config.device)
    search = AlphaZeroSearch(inference, config=replace(_selfplay_search(), simulations=8))
    result = search.run(env, state, env.legal_actions(state), self_play=True, seed=12_900_001)
    row = replay_row_from_search(
        generation=11,
        game_seed=2_799_999,
        state=state,
        result=result,
        checkpoint_path=str(config.generation_11_checkpoint),
        checkpoint_generation=11,
        search_config=asdict(replace(_selfplay_search(), simulations=8)),
    )
    if len(row.root_visits) == 0 or sum(row.root_visits) != 8 or row.value_target is not None:
        raise ValueError("129 temporary continuation target construction failed.")
    return {
        "kind": "alphazero_historical_state_reanalysis_129_preflight",
        "status": "preflight_passed",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": observed,
        "source_summaries": source_summaries,
        "historical_holdout_schedule": holdout_schedule,
        "temporary_target_identity": identity,
        "temporary_target_simulations": sum(row.root_visits),
        "temporary_target_persisted": False,
        "training_config": asdict(reanalysis_training_config()),
        "no_optimizer_instantiated": True,
        "no_durable_replay": True,
        "no_training": True,
        "no_strength_gameplay": True,
    }


def _preflight_source(
    path: Path, generation: int, source_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[SplendorEnv, SplendorState, dict[str, Any]]]:
    games = 0
    rows = 0
    holdout_schedule = []
    first_state = None
    for ordinal, game_rows in enumerate(_iter_source_games(path)):
        games += 1
        rows += len(game_rows)
        if any(
            row.generation != generation
            or row.checkpoint_generation != generation
            or row.termination_reason != "completed"
            for row in game_rows
        ):
            raise ValueError(f"129 generation-{generation} source identity mismatch.")
        if ordinal < TRAIN_ATTEMPTS_PER_SOURCE:
            phase = phase_for_slot(ordinal % TRAIN_ACCEPTED_PER_SOURCE)
            env, state, identity = _reconstruct_start(game_rows, phase, generation, source_index, ordinal)
            if first_state is None:
                first_state = (env, state, identity)
        if HOLDOUT_ORDINAL_START <= ordinal < HOLDOUT_ORDINAL_START + HOLDOUT_GAMES_PER_SOURCE:
            phase = phase_for_slot(ordinal - HOLDOUT_ORDINAL_START)
            _env, _state, identity = _reconstruct_start(game_rows, phase, generation, source_index, ordinal)
            holdout_schedule.append(identity)
    expected_games = 224 if generation == 11 else 256
    if games != expected_games or first_state is None:
        raise ValueError(
            f"129 generation-{generation} source does not contain {expected_games} games."
        )
    return (
        {
            "source_index": source_index,
            "generation": generation,
            "path": str(path),
            "sha256": _sha256(path),
            "games_scanned": games,
            "rows_scanned": rows,
            "training_attempt_games": TRAIN_ATTEMPTS_PER_SOURCE,
            "holdout_games": HOLDOUT_GAMES_PER_SOURCE,
        },
        holdout_schedule,
        first_state,
    )


def _iter_source_games(path: Path, *, limit: int | None = None) -> Iterator[list[AlphaZeroReplayRow]]:
    current_seed = None
    rows: list[AlphaZeroReplayRow] = []
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = _row_from_payload(json.loads(line))
            if current_seed is None:
                current_seed = row.game_seed
            if row.game_seed != current_seed:
                yield rows
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
                rows = []
                current_seed = row.game_seed
            rows.append(row)
    if rows and (limit is None or yielded < limit):
        yield rows


def _reconstruct_start(
    game_rows: list[AlphaZeroReplayRow], phase: str, generation: int, source_index: int, source_ordinal: int
) -> tuple[SplendorEnv, SplendorState, dict[str, Any]]:
    if not game_rows:
        raise ValueError("Cannot reconstruct an empty historical game.")
    game_rows = sorted(game_rows, key=lambda row: row.turn_index)
    source_seed = game_rows[0].game_seed
    env = SplendorEnv(seed=source_seed)
    state = env.initial_state()
    states: list[SplendorState] = []
    pending: list[bool] = []
    for expected_turn, row in enumerate(game_rows):
        if row.turn_index != expected_turn or row.player_id != state.current_player or state.turn_index != expected_turn:
            raise ValueError("129 historical replay turn/player reconstruction mismatch.")
        encoded = encode_canonical_state_entities(state, state.current_player)
        if (
            encoded.entity_type_ids != row.entity_type_ids
            or encoded.card_ids != row.entity_card_ids
            or encoded.noble_ids != row.entity_noble_ids
            or encoded.numeric_features != row.entity_numeric_features
        ):
            raise ValueError("129 historical replay tensor reconstruction mismatch.")
        states.append(state.copy_shallow())
        pending.append(bool(state.pending_round_end))
        selected_payload = row.action_payloads[row.selected_action_row]
        matches = [action for action in env.legal_actions(state) if serialize_action(action) == selected_payload]
        if len(matches) != 1:
            raise ValueError("129 historical selected action is not uniquely legal.")
        state = env.step(state, matches[0])
    last = game_rows[-1]
    if (
        not state.terminal
        or last.termination_reason != "completed"
        or state.winner != last.winner
        or tuple(player.score for player in state.players) != last.final_scores
    ):
        raise ValueError("129 historical game did not reconstruct exactly to terminal.")
    selected_index = nearest_eligible_index(pending, phase)
    selected_state = states[selected_index]
    identity = {
        "source_generation": generation,
        "source_index": source_index,
        "source_game_ordinal": source_ordinal,
        "source_game_seed": source_seed,
        "source_turn_index": selected_state.turn_index,
        "source_player": selected_state.current_player,
        "source_phase": phase,
        "source_progress": selected_index / max(len(states) - 1, 1),
        "pending_round_end": bool(selected_state.pending_round_end),
    }
    return SplendorEnv(seed=source_seed), selected_state, identity


def _run_or_load_historical_collection(config: HistoricalStateReanalysisConfig) -> dict[str, Any]:
    output = config.output_dir / "historical_collection"
    manifest = output / "collection.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_collection_manifest(config, payload)
        return payload
    tasks = [
        {
            "source_index": index,
            "source_generation": generation,
            "source_path": str(path),
            "source_hash": EXPECTED_HASHES[f"source_generation_{generation}"],
            "output_dir": str(output),
            "checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
        }
        for index, (generation, path) in enumerate(zip(SOURCE_GENERATIONS, config.source_replays, strict=True))
    ]
    shards = {}
    with ProcessPoolExecutor(max_workers=config.collection_workers) as executor:
        futures = {executor.submit(_collection_worker, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            shards[int(record["source_index"])] = record
    if set(shards) != set(range(4)):
        raise ValueError("129 historical collection requires four exact source shards.")
    combined_path = output / "historical_reanalysis.jsonl"
    temporary = combined_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as destination:
        for index in range(4):
            with Path(shards[index]["sanitized_path"]).open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(combined_path)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "shards": {str(index): shards[index] for index in range(4)},
        "games": sum(int(shards[index]["games"]) for index in range(4)),
        "positions": sum(int(shards[index]["positions"]) for index in range(4)),
        "combined_path": str(combined_path),
        "combined_sha256": _sha256(combined_path),
    }
    if payload["games"] != 256:
        raise ValueError("129 did not seal exactly 256 historical continuations.")
    _write_json(manifest, payload)
    return payload


def _collection_worker(task: dict[str, Any]) -> dict[str, Any]:
    _limit_worker_threads()
    source_index = int(task["source_index"])
    generation = int(task["source_generation"])
    source_path = Path(task["source_path"])
    shard_dir = Path(task["output_dir"]) / f"source_generation_{generation}"
    progress_path = shard_dir / "progress.json"
    identity_path = shard_dir / "identity.json"
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_index": source_index,
        "source_generation": generation,
        "source_path": str(source_path),
        "source_sha256": task["source_hash"],
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "accepted_games": TRAIN_ACCEPTED_PER_SOURCE,
        "attempt_games": TRAIN_ATTEMPTS_PER_SOURCE,
        "search": asdict(_selfplay_search()),
        "max_turns": int(task["max_turns"]),
    }
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("129 collection shard identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError("129 collection progress lacks identity.")
        _write_json(identity_path, identity)
    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("129 collection progress identity mismatch.")
        records = list(progress.get("records", []))
        accepted = [record for record in records if record["accepted"]]
    inference = AlphaZeroInference(task["checkpoint"], device=task["device"])
    search = AlphaZeroSearch(inference, config=_selfplay_search())
    for ordinal, game_rows in enumerate(_iter_source_games(source_path, limit=TRAIN_ATTEMPTS_PER_SOURCE)):
        if ordinal < len(records):
            continue
        if len(accepted) >= TRAIN_ACCEPTED_PER_SOURCE:
            break
        slot = len(accepted)
        phase = phase_for_slot(slot)
        env, state, source_identity = _reconstruct_start(game_rows, phase, generation, source_index, ordinal)
        synthetic_seed = 2_700_000 + source_index * 10_000 + ordinal
        rows, summary = _play_continuation(
            env=env,
            initial_state=state,
            search=search,
            checkpoint_path=str(task["checkpoint"]),
            synthetic_seed=synthetic_seed,
            source_identity=source_identity,
            max_turns=int(task["max_turns"]),
        )
        game_path = shard_dir / "games" / f"attempt_{ordinal:03d}.jsonl"
        _write_continuation_rows(game_path, rows, source_identity)
        accepted_flag = summary["termination_reason"] == "completed"
        record = {
            "source_ordinal": ordinal,
            "accepted_slot": slot if accepted_flag else None,
            "accepted": accepted_flag,
            "phase": phase,
            "synthetic_seed": synthetic_seed,
            "path": str(game_path),
            "sha256": _sha256(game_path),
            "source_identity": source_identity,
            **summary,
        }
        records.append(record)
        if accepted_flag:
            accepted.append(record)
        _write_json(progress_path, {
            "identity": identity,
            "attempted": len(records),
            "accepted": len(accepted),
            "complete": False,
            "records": records,
        })
        del rows
        gc.collect()
    if len(accepted) != TRAIN_ACCEPTED_PER_SOURCE:
        raise RuntimeError(f"129 generation-{generation} could not seal 64 clean continuations.")
    sanitized = shard_dir / "sanitized.jsonl"
    temporary = sanitized.with_suffix(".jsonl.tmp")
    with temporary.open("wb") as destination:
        for record in accepted:
            with Path(record["path"]).open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(sanitized)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_index": source_index,
        "source_generation": generation,
        "source_sha256": task["source_hash"],
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "attempted_games": len(records),
        "games": len(accepted),
        "positions": sum(int(record["positions"]) for record in accepted),
        "phase_counts": dict(sorted(Counter(record["phase"] for record in accepted).items())),
        "abnormal_attempts": [record for record in records if not record["accepted"]],
        "selected_source_ordinals": [record["source_ordinal"] for record in accepted],
        "sanitized_path": str(sanitized),
        "sanitized_sha256": _sha256(sanitized),
    }
    _write_json(progress_path, {"identity": identity, "complete": True, "record": payload, "records": records})
    return payload


def _play_continuation(
    *, env: SplendorEnv, initial_state: SplendorState, search: AlphaZeroSearch,
    checkpoint_path: str, synthetic_seed: int, source_identity: dict[str, Any], max_turns: int
) -> tuple[list[AlphaZeroReplayRow], dict[str, Any]]:
    state = initial_state.copy_shallow()
    pending: list[AlphaZeroReplayRow] = []
    visits: dict[tuple[object, ...], int] = {}
    no_progress = 0
    forced_passes = 0
    reason = "completed"
    while not state.terminal:
        signature = state_signature(state)
        visits[signature] = visits.get(signature, 0) + 1
        if visits[signature] >= 4:
            reason = "repetition_cutoff"
            break
        if state.turn_index >= max_turns:
            reason = "max_turns"
            break
        legal = env.legal_actions(state)
        if not legal:
            reason = "stalled"
            break
        result = search.run(
            env,
            state,
            legal,
            self_play=True,
            seed=SELFPLAY_SEARCH_SEED + synthetic_seed * 1_000_003 + state.turn_index * 97_409,
        )
        pending.append(replay_row_from_search(
            generation=11,
            game_seed=synthetic_seed,
            state=state,
            result=result,
            checkpoint_path=checkpoint_path,
            checkpoint_generation=11,
            search_config=asdict(_selfplay_search()),
        ))
        action = result.action
        next_state = env.step(state, action)
        forced = len(legal) == 1 and legal[0].action_type is ActionType.PASS
        forced_passes = forced_passes + 1 if forced else 0
        if forced_passes >= 2:
            state = next_state
            reason = "forced_pass_deadlock"
            break
        no_progress = 0 if is_progress_transition(state, next_state, action) else no_progress + 1
        state = next_state
        if no_progress >= 60:
            reason = "no_progress_cutoff"
            break
    winner = state.winner if state.terminal else _adjudicate_scores(state)
    final_scores = (state.players[0].score, state.players[1].score)
    completed = [row.with_outcome(winner=winner, final_scores=final_scores, termination_reason=reason) for row in pending]
    return completed, {
        "winner": winner,
        "final_scores": list(final_scores),
        "turns": state.turn_index,
        "positions": len(completed),
        "termination_reason": reason,
        "source_turn_index": source_identity["source_turn_index"],
    }


def _write_continuation_rows(path: Path, rows: list[AlphaZeroReplayRow], source_identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row)
            payload["historical_reanalysis_source"] = source_identity
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _validate_collection_manifest(config: HistoricalStateReanalysisConfig, payload: dict[str, Any]) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("games") != 256
        or payload.get("combined_sha256") != _sha256(payload["combined_path"])
        or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
    ):
        raise ValueError("129 historical collection manifest mismatch.")


def _run_or_load_mixed_replay(config: HistoricalStateReanalysisConfig, collection: dict[str, Any]) -> dict[str, Any]:
    output = config.output_dir / "mixed_replay"
    manifest = output / "mixed.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("historical_sha256") != collection["combined_sha256"]
            or payload.get("ordinary_anchor_sha256") != _sha256(payload["ordinary_anchor_path"])
        ):
            raise ValueError("129 mixed replay identity mismatch.")
        return payload
    ordinary_path = config.source_replays[3]
    game_lengths = [len(rows) for rows in _iter_source_games(ordinary_path)]
    selection = select_anchor_game_count(game_lengths, int(collection["positions"]))
    if selection["absolute_row_difference"] > selection["maximum_selected_game_rows"]:
        raise ValueError("129 matched replay exceeds its one-game row tolerance.")
    anchor_path = output / "ordinary_anchor.jsonl"
    temporary = anchor_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    selected_games = int(selection["selected_games"])
    with temporary.open("w", encoding="utf-8") as handle:
        for ordinal, rows in enumerate(_iter_source_games(ordinary_path, limit=selected_games)):
            if ordinal >= selected_games:
                break
            for row in rows:
                handle.write(json.dumps(asdict(row), sort_keys=True))
                handle.write("\n")
    temporary.replace(anchor_path)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "historical_path": collection["combined_path"],
        "historical_sha256": collection["combined_sha256"],
        "ordinary_source_sha256": EXPECTED_HASHES["source_generation_11"],
        "ordinary_anchor_path": str(anchor_path),
        "ordinary_anchor_sha256": _sha256(anchor_path),
        **selection,
    }
    _write_json(manifest, payload)
    return payload


def _run_or_load_training(config: HistoricalStateReanalysisConfig, mixed: dict[str, Any]) -> dict[str, Any]:
    stage = config.output_dir / "training" / "training.stage.json"
    checkpoint = config.output_dir / "checkpoints" / "historical_reanalysis_generation_012.pt"
    metrics = config.output_dir / "training" / "metrics.json"
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("child_checkpoint_sha256") != _sha256(payload["child_checkpoint"])
            or payload.get("metrics_sha256") != _sha256(payload["metrics_path"])
        ):
            raise ValueError("129 training stage identity mismatch.")
        return payload
    training = train_parent_safe_indexed_successor(
        parent_checkpoint=config.generation_11_checkpoint,
        replay_paths=[mixed["historical_path"], mixed["ordinary_anchor_path"]],
        output_checkpoint=checkpoint,
        metrics_path=metrics,
        device=config.device,
        config=reanalysis_training_config(),
    )
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "child_checkpoint": str(checkpoint),
        "child_checkpoint_sha256": _sha256(checkpoint),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
        **training,
    }
    _write_json(stage, payload)
    return payload


def _run_or_load_holdout(config: HistoricalStateReanalysisConfig, training: dict[str, Any]) -> dict[str, Any]:
    path = config.output_dir / "holdout" / "ordinary_holdout.json"
    candidate = Path(training["child_checkpoint"])
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("candidate_checkpoint_sha256") != _sha256(candidate)
            or payload.get("holdout_sha256") != EXPECTED_HASHES["ordinary_holdout"]
        ):
            raise ValueError("129 ordinary holdout identity mismatch.")
        return payload
    paths, entries = index_replay_paths([config.ordinary_holdout])
    loader = make_indexed_loader(paths, entries, config=reanalysis_training_config(), shuffle=False)
    checkpoints = {
        "generation_11": config.generation_11_checkpoint,
        "terminal_control_114": config.terminal_control_checkpoint,
        "historical_reanalysis_candidate": candidate,
    }
    metrics_out = {}
    for name, checkpoint in checkpoints.items():
        model, _identity = load_checkpoint(checkpoint, device=config.device)
        model.eval()
        metrics_out[name] = _run_epoch(model, loader, config.device, None, reanalysis_training_config())
    qualification = reanalysis_offline_qualification(training=training, metrics=metrics_out)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_checkpoint_sha256": _sha256(candidate),
        "holdout_sha256": EXPECTED_HASHES["ordinary_holdout"],
        "rows": len(entries),
        "games": len({entry.game_seed for entry in entries}),
        "metrics": metrics_out,
        "qualification": qualification,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    if isinstance(loader.dataset, IndexedReplayDataset):
        loader.dataset.close()
    return payload


def _load_historical_holdout_states(config: HistoricalStateReanalysisConfig):
    states = []
    for source_index, (generation, path) in enumerate(zip(SOURCE_GENERATIONS, config.source_replays, strict=True)):
        for ordinal, game_rows in enumerate(_iter_source_games(path, limit=HOLDOUT_ORDINAL_START + HOLDOUT_GAMES_PER_SOURCE)):
            if ordinal < HOLDOUT_ORDINAL_START:
                continue
            local = ordinal - HOLDOUT_ORDINAL_START
            phase = phase_for_slot(local)
            _env, state, identity = _reconstruct_start(game_rows, phase, generation, source_index, ordinal)
            identity = {**identity, "panel_index": len(states), "paired_seed": 2_710_000 + len(states)}
            states.append((state, identity))
    if len(states) != HISTORICAL_HOLDOUT_STATES:
        raise ValueError("129 historical holdout reconstruction count mismatch.")
    return states


def _run_or_load_historical_panel(config: HistoricalStateReanalysisConfig, candidate: Path) -> dict[str, Any]:
    output = config.output_dir / "arenas" / "historical_start"
    manifest = output / "panel.stage.json"
    candidate_hash = _sha256(candidate)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("candidate_checkpoint_sha256") != candidate_hash
            or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
        ):
            raise ValueError("129 historical panel identity mismatch.")
        arena = _combine_historical_shards({int(key): value for key, value in payload["shards"].items()})
    else:
        states = _load_historical_holdout_states(config)
        tasks = [
            {
                "shard": shard,
                "states": states[shard * 16 : (shard + 1) * 16],
                "output_dir": str(output),
                "candidate": str(candidate),
                "candidate_hash": candidate_hash,
                "baseline": str(config.terminal_control_checkpoint),
                "device": config.device,
                "max_turns": config.max_turns,
            }
            for shard in range(4)
        ]
        shards = {}
        with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
            futures = {executor.submit(_historical_panel_worker, task): task for task in tasks}
            for future in as_completed(futures):
                shard, rows = future.result()
                shards[shard] = rows
        arena = _combine_historical_shards(shards)
        _write_json(manifest, {
            "protocol_sha256": PROTOCOL_SHA256,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": EXPECTED_HASHES["terminal_control_checkpoint"],
            "shards": {str(key): shards[key] for key in sorted(shards)},
        })
    bootstrap = paired_score_share_bootstrap(arena, samples=BOOTSTRAP_SAMPLES, seed=1_129_300)
    return {"arena": arena, "paired_bootstrap": bootstrap, "assessment": compression_assessment(arena, bootstrap)}


def _historical_panel_worker(task: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    _limit_worker_threads()
    shard = int(task["shard"])
    output = Path(task["output_dir"])
    progress = output / f"shard_{shard}.json"
    identities = [identity for _state, identity in task["states"]]
    sidecar = {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard": shard,
        "candidate_checkpoint_sha256": task["candidate_hash"],
        "baseline_checkpoint_sha256": EXPECTED_HASHES["terminal_control_checkpoint"],
        "search": asdict(_arena_search()),
        "states": identities,
        "max_turns": int(task["max_turns"]),
    }
    sidecar_path = output / f"shard_{shard}.identity.json"
    if sidecar_path.exists():
        if json.loads(sidecar_path.read_text(encoding="utf-8")) != sidecar:
            raise ValueError("129 historical arena shard identity mismatch.")
    else:
        if progress.exists():
            raise ValueError("129 historical arena progress lacks identity.")
        _write_json(sidecar_path, sidecar)
    rows = []
    if progress.exists():
        payload = json.loads(progress.read_text(encoding="utf-8"))
        if payload.get("identity") != sidecar:
            raise ValueError("129 historical arena progress identity mismatch.")
        rows = list(payload.get("rows", []))
    candidate_inference = AlphaZeroInference(task["candidate"], device=task["device"])
    baseline_inference = AlphaZeroInference(task["baseline"], device=task["device"])
    schedule = [(state, identity, seat) for state, identity in task["states"] for seat in (0, 1)]
    observed = [(row["identity"], row["candidate_seat"]) for row in rows]
    expected = [(identity, seat) for _state, identity, seat in schedule[: len(rows)]]
    if observed != expected:
        raise ValueError("129 historical arena progress is not an exact prefix.")
    for index, (state, identity, candidate_seat) in enumerate(schedule):
        if index < len(rows):
            continue
        rows.append(_play_historical_arena_game(
            state=state,
            identity=identity,
            candidate_seat=candidate_seat,
            candidate_inference=candidate_inference,
            baseline_inference=baseline_inference,
            max_turns=int(task["max_turns"]),
        ))
        _write_json(progress, {"identity": sidecar, "complete": False, "rows": rows})
    _write_json(progress, {"identity": sidecar, "complete": True, "rows": rows})
    return shard, rows


def _play_historical_arena_game(
    *, state: SplendorState, identity: dict[str, Any], candidate_seat: int,
    candidate_inference: AlphaZeroInference, baseline_inference: AlphaZeroInference, max_turns: int
) -> dict[str, Any]:
    env = SplendorEnv(seed=int(identity["source_game_seed"]))
    state = state.copy_shallow()
    searches = (
        AlphaZeroSearch(candidate_inference if candidate_seat == 0 else baseline_inference, config=_arena_search()),
        AlphaZeroSearch(candidate_inference if candidate_seat == 1 else baseline_inference, config=_arena_search()),
    )
    visits: dict[tuple[object, ...], int] = {}
    no_progress = 0
    forced_passes = 0
    reason = "completed"
    while not state.terminal:
        signature = state_signature(state)
        visits[signature] = visits.get(signature, 0) + 1
        if visits[signature] >= 4:
            reason = "repetition_cutoff"
            break
        if state.turn_index >= max_turns:
            reason = "max_turns"
            break
        legal = env.legal_actions(state)
        if not legal:
            reason = "stalled"
            break
        search = searches[state.current_player]
        result = search.run(
            env,
            state,
            legal,
            self_play=False,
            seed=int(identity["paired_seed"]) * 2_000_003 + candidate_seat * 101 + state.turn_index * 97_409,
        )
        next_state = env.step(state, result.action)
        forced = len(legal) == 1 and legal[0].action_type is ActionType.PASS
        forced_passes = forced_passes + 1 if forced else 0
        if forced_passes >= 2:
            state = next_state
            reason = "forced_pass_deadlock"
            break
        no_progress = 0 if is_progress_transition(state, next_state, result.action) else no_progress + 1
        state = next_state
        if no_progress >= 60:
            reason = "no_progress_cutoff"
            break
    winner = state.winner if state.terminal else _adjudicate_scores(state)
    scores = (state.players[0].score, state.players[1].score)
    points = 0.5 if winner is None else 1.0 if winner == candidate_seat else 0.0
    return {
        "seed": int(identity["paired_seed"]),
        "identity": identity,
        "candidate_seat": candidate_seat,
        "winner": winner,
        "candidate_points": points,
        "candidate_margin": scores[candidate_seat] - scores[1 - candidate_seat],
        "final_scores": list(scores),
        "turns": state.turn_index,
        "termination_reason": reason,
    }


def _combine_historical_shards(shards: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    if set(shards) != set(range(4)):
        raise ValueError("129 historical panel requires four exact shards.")
    rows = [row for shard in range(4) for row in shards[shard]]
    if len(rows) != 128:
        raise ValueError("129 historical panel requires exactly 128 games.")
    expected = [(2_710_000 + index, seat) for index in range(64) for seat in (0, 1)]
    observed = [(int(row["seed"]), int(row["candidate_seat"])) for row in rows]
    if observed != expected:
        raise ValueError("129 historical panel schedule mismatch.")
    return _arena_from_rows("historical_start_candidate_vs_terminal_control", 64, rows)


def _run_or_load_ordinary_panel(config: HistoricalStateReanalysisConfig, candidate: Path) -> dict[str, Any]:
    output = config.output_dir / "arenas" / "ordinary_initial_state"
    manifest = output / "panel.stage.json"
    candidate_hash = _sha256(candidate)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("candidate_checkpoint_sha256") != candidate_hash
            or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
        ):
            raise ValueError("129 ordinary arena manifest mismatch.")
        shards = {int(key): value for key, value in payload["shards"].items()}
    else:
        tasks = [{
            "shard": shard,
            "output_dir": str(output),
            "candidate": str(candidate),
            "candidate_hash": candidate_hash,
            "baseline": str(config.terminal_control_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
        } for shard in range(4)]
        shards = {}
        with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
            futures = {executor.submit(_ordinary_panel_worker, task): task for task in tasks}
            for future in as_completed(futures):
                shard, arena = future.result()
                shards[shard] = arena
        _write_json(manifest, {
            "protocol_sha256": PROTOCOL_SHA256,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": EXPECTED_HASHES["terminal_control_checkpoint"],
            "shards": {str(key): shards[key] for key in sorted(shards)},
        })
    arena = _combine_ordinary_shards(shards)
    bootstrap = paired_score_share_bootstrap(arena, samples=BOOTSTRAP_SAMPLES, seed=1_129_400)
    return {"arena": arena, "paired_bootstrap": bootstrap, "assessment": compression_assessment(arena, bootstrap)}


def _ordinary_panel_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    _limit_worker_threads()
    shard = int(task["shard"])
    per_shard = ORDINARY_PAIRED_SEEDS // 4
    seed_start = ORDINARY_ARENA_SEED_START + shard * per_shard
    output = Path(task["output_dir"])
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard": shard,
        "candidate_checkpoint_sha256": task["candidate_hash"],
        "baseline_checkpoint_sha256": EXPECTED_HASHES["terminal_control_checkpoint"],
        "seed_start": seed_start,
        "paired_seeds": per_shard,
        "search": asdict(_arena_search()),
        "max_turns": int(task["max_turns"]),
    }
    identity_path = output / f"shard_{shard}.identity.json"
    progress = output / f"shard_{shard}.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("129 ordinary arena identity mismatch.")
    else:
        if progress.exists():
            raise ValueError("129 ordinary arena progress lacks identity.")
        _write_json(identity_path, identity)
    arena = _arena_dict(run_paired_arena(
        name=f"ordinary_initial_state_shard_{shard}",
        candidate_factory=alpha_zero_search_factory(task["candidate"], task["device"], _arena_search()),
        baseline_factory=alpha_zero_search_factory(task["baseline"], task["device"], _arena_search()),
        seed_start=seed_start,
        paired_seeds=per_shard,
        max_turns=int(task["max_turns"]),
        progress_path=progress,
        resume=True,
    ))
    return shard, arena


def _combine_ordinary_shards(shards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(shards) != set(range(4)):
        raise ValueError("129 ordinary panel requires four exact shards.")
    rows = []
    for shard in range(4):
        start = ORDINARY_ARENA_SEED_START + shard * 50
        expected = [(start + offset, seat) for offset in range(50) for seat in (0, 1)]
        shard_rows = list(shards[shard]["rows"])
        observed = [(int(row["seed"]), int(row["candidate_seat"])) for row in shard_rows]
        if observed != expected:
            raise ValueError("129 ordinary panel schedule mismatch.")
        rows.extend(shard_rows)
    return _arena_from_rows("ordinary_initial_state_candidate_vs_terminal_control", 200, rows)


def _arena_from_rows(name: str, paired_seeds: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    reasons = Counter(str(row["termination_reason"]) for row in rows)
    return _arena_dict(ArenaResult(
        name=name,
        paired_seeds=paired_seeds,
        games=len(rows),
        candidate_wins=wins,
        candidate_losses=losses,
        draws=draws,
        score_share=(wins + 0.5 * draws) / len(rows),
        mean_margin=sum(float(row["candidate_margin"]) for row in rows) / len(rows),
        termination_reasons=dict(sorted(reasons.items())),
        rows=tuple(rows),
    ))


def _selfplay_search() -> SearchConfig:
    return SearchConfig(
        simulations=SELFPLAY_SIMULATIONS,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.25,
        seed=SELFPLAY_SEARCH_SEED,
    )


def _arena_search() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_129_200,
    )


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})


def _seal(config: HistoricalStateReanalysisConfig, result: dict[str, Any], started: float, decision: str) -> dict[str, Any]:
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = decision
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(config.output_dir / "result.json", result)
    return result
