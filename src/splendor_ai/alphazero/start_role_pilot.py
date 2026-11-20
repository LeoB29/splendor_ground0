"""Bounded start-role joint distillation pilot 118."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.encoding import (
    actions_by_canonical_successor_key,
    canonical_public_belief_key,
    collate_entity_action_batch,
    encode_canonical_state_entities,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import serialize_action

from .capacity_network import load_capacity_checkpoint
from .network import AlphaZeroInference, load_checkpoint
from .replay import AlphaZeroReplayRow, read_replay_jsonl
from .role_network import (
    create_identity_role_model,
    load_role_checkpoint,
    role_parameter_counts,
    save_role_checkpoint,
)
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearch, SearchConfig
from .start_role_audit import phase_for_turn


@dataclass(frozen=True, slots=True)
class StartRolePilotConfig:
    output_dir: Path
    result_117: Path
    generation_11_checkpoint: Path
    candidate_114: Path
    candidate_116: Path
    partition_manifest: Path
    train_replay: Path
    holdout_replay: Path
    installed_checkpoint: Path
    device: str = "cpu"
    torch_threads: int = 1
    workers: int = 4
    arena_workers: int = 4
    max_turns: int = 200


PROTOCOL_SHA256 = "b7b92bf53ab1801bef96fbdb4f9544a81943889c82a243c20b4349cb4b2bf186"
EXPECTED_HASHES = {
    "result_117": "a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "candidate_114": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
    "candidate_116": "bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565",
    "partition_manifest": "61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc",
    "train_replay": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "holdout_replay": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
}

TRAIN_GAMES = 224
HOLDOUT_GAMES = 32
STATES_PER_GAME = 2
TRAIN_PAIRS = 448
HOLDOUT_PAIRS = 64
TARGET_ROWS = 896
TRAIN_SPLIT_GAMES = 179
VALIDATION_SPLIT_GAMES = 45
TRAIN_SPLIT_PAIRS = 358
VALIDATION_SPLIT_PAIRS = 90
SIMULATIONS = 1_024
TARGET_SEED_BASE = 1_118_000
SPLIT_SEED = 1_118_100
SHUFFLE_SEED = 1_118_118
BOOTSTRAP_SAMPLES = 50_000
TARGET_SHARDS = 4
TARGET_PAIRS_PER_SHARD = 112
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
VALUE_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class _TrainingRow:
    game_seed: int
    panel_index: int
    state: SplendorState
    actions: tuple[Action, ...]
    observer_is_start_player: bool
    policy_target: tuple[float, ...]
    value_target: float


class _RoleTargetDataset(Dataset[_TrainingRow]):
    def __init__(self, rows: list[_TrainingRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> _TrainingRow:
        return self.rows[index]


def preflight_start_role_pilot_118(config: StartRolePilotConfig) -> dict[str, Any]:
    """Validate every frozen input and epoch-zero contract without evidence."""

    _limit_worker_threads()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    train_rows = read_replay_jsonl(config.train_replay)
    holdout_rows = read_replay_jsonl(config.holdout_replay)
    train_states, train_reconstruction = reconstruct_role_panel(
        train_rows,
        expected_games=TRAIN_GAMES,
        label="118 training",
    )
    holdout_states, holdout_reconstruction = reconstruct_role_panel(
        holdout_rows,
        expected_games=HOLDOUT_GAMES,
        label="118 confirmation",
    )
    split = split_pair_games(train_states)

    model, parent_identity = create_identity_role_model(
        config.generation_11_checkpoint,
        device=config.device,
    )
    counts = role_parameter_counts(model)
    if counts != {
        "parent": 356_450,
        "role_embedding": 192,
        "total": 356_642,
        "trainable": 192,
    }:
        raise ValueError(f"118 role model has wrong parameter counts: {counts}")
    identity = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=model,
        states=train_states + holdout_states,
        device=config.device,
    )
    if identity["max_logit_abs_diff"] > 1e-7 or identity["max_value_abs_diff"] > 1e-7:
        raise ValueError("118 epoch-zero role bridge is not exact gen11.")

    epoch_zero = config.output_dir / "checkpoints" / "epoch_zero_identity.pt"
    save_role_checkpoint(
        epoch_zero,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "parent_generation": parent_identity.generation,
            "selected_epoch": 0,
            "preflight_only": True,
        },
    )
    roundtrip, roundtrip_identity = load_role_checkpoint(
        epoch_zero,
        device=config.device,
    )
    roundtrip_audit = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=roundtrip,
        states=train_states[:4] + holdout_states[:4],
        device=config.device,
    )
    if (
        roundtrip_identity.generation != 12
        or roundtrip_identity.metadata.get("selected_epoch") != 0
        or roundtrip_audit["max_logit_abs_diff"] > 1e-7
        or roundtrip_audit["max_value_abs_diff"] > 1e-7
    ):
        raise ValueError("118 epoch-zero checkpoint failed round-trip identity.")

    payload = {
        "kind": "alphazero_start_role_joint_pilot_118_preflight",
        "status": "preflight_complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": dict(EXPECTED_HASHES),
        "config": _jsonable_config(config),
        "train_reconstruction": train_reconstruction,
        "confirmation_reconstruction": holdout_reconstruction,
        "split": split,
        "parameter_counts": counts,
        "parent_generation": parent_identity.generation,
        "identity_audit": identity,
        "roundtrip_identity_audit": roundtrip_audit,
        "epoch_zero_checkpoint": str(epoch_zero),
        "epoch_zero_checkpoint_sha256": _sha256(epoch_zero),
        "teacher_targets_generated": False,
        "training_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
    }
    _write_json(config.output_dir / "preflight.json", payload)
    return payload


def run_start_role_pilot_118(
    config: StartRolePilotConfig,
    *,
    launch: bool = False,
) -> dict[str, Any]:
    preflight = preflight_start_role_pilot_118(config)
    if not launch:
        return preflight
    started = time.perf_counter()
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 118 result is not a valid completed result.")

    train_rows = read_replay_jsonl(config.train_replay)
    holdout_rows = read_replay_jsonl(config.holdout_replay)
    train_states, _ = reconstruct_role_panel(train_rows, expected_games=TRAIN_GAMES, label="118 training")
    holdout_states, _ = reconstruct_role_panel(holdout_rows, expected_games=HOLDOUT_GAMES, label="118 confirmation")
    result: dict[str, Any] = {
        "kind": "alphazero_start_role_joint_pilot_118",
        "status": "running",
        "active_stage": "paired_teacher_targets",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": dict(EXPECTED_HASHES),
        "preflight": preflight,
        "teacher_targets_generated": False,
        "training_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
    }
    _write_progress(config.output_dir, result, started)

    targets = run_or_load_target_panel(config, train_states, confirmation=False)
    result["teacher_targets_generated"] = True
    result["target_summary"] = summarize_target_records(targets)
    result["active_stage"] = "role_only_training"
    _write_progress(config.output_dir, result, started)

    training = run_or_load_role_training(config, train_states, targets)
    result["training_performed"] = True
    result["training"] = training
    if not training["selection_passed"]:
        result["decision"] = "minimal_role_bridge_failed_direct_uptake_review_richer_interaction_no_gameplay"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "heldout_role_confirmation"
    _write_progress(config.output_dir, result, started)
    confirmation_targets = run_or_load_target_panel(
        config,
        holdout_states,
        confirmation=True,
    )
    confirmation = evaluate_role_confirmation(
        config,
        holdout_states,
        confirmation_targets,
        candidate_checkpoint=Path(training["checkpoint"]),
    )
    result["role_confirmation"] = confirmation
    if not confirmation["passed"]:
        result["decision"] = "minimal_role_bridge_failed_direct_uptake_review_richer_interaction_no_gameplay"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "reachable_safety"
    _write_progress(config.output_dir, result, started)
    safety = evaluate_reachable_safety(
        config,
        holdout_rows,
        candidate_checkpoint=Path(training["checkpoint"]),
    )
    result["reachable_safety"] = safety
    if not safety["passed"]:
        result["decision"] = "role_bridge_uptake_with_reachable_regression_no_gameplay"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "ordinary_budget_arena"
    _write_progress(config.output_dir, result, started)
    arena = run_or_load_arena(config, Path(training["checkpoint"]), training["checkpoint_sha256"])
    result["gameplay_arena"] = arena
    result["decision"] = (
        "minimal_start_role_joint_bridge_confirmed_review_short_continuation_no_install"
        if arena["assessment"]["passed"]
        else "role_omission_real_but_not_sufficient_close_simple_role_repair_no_install"
    )
    return _seal(config.output_dir, result, started)


def reconstruct_role_panel(
    rows: list[AlphaZeroReplayRow],
    *,
    expected_games: int,
    label: str,
) -> tuple[list[tuple[int, SplendorState]], dict[str, Any]]:
    grouped: dict[int, list[AlphaZeroReplayRow]] = {}
    for row in rows:
        grouped.setdefault(row.game_seed, []).append(row)
    if len(grouped) != expected_games:
        raise ValueError(f"{label} requires exactly {expected_games} games.")
    selected: list[tuple[int, SplendorState]] = []
    schedule: list[dict[str, Any]] = []
    for game_seed in sorted(grouped):
        game_rows = sorted(grouped[game_seed], key=lambda row: row.turn_index)
        env = SplendorEnv(seed=game_seed)
        state = env.initial_state()
        eligible: list[SplendorState] = []
        for expected_turn, row in enumerate(game_rows):
            if (
                row.turn_index != expected_turn
                or row.player_id != state.current_player
                or row.player_id != row.turn_index % 2
                or state.turn_index != row.turn_index
            ):
                raise ValueError(f"{label} reconstructed turn/player mismatch.")
            encoded = encode_canonical_state_entities(state, state.current_player)
            if (
                encoded.entity_type_ids != row.entity_type_ids
                or encoded.card_ids != row.entity_card_ids
                or encoded.noble_ids != row.entity_noble_ids
                or encoded.numeric_features != row.entity_numeric_features
            ):
                raise ValueError(f"{label} reconstructed entity tensor mismatch.")
            if not state.pending_round_end:
                eligible.append(state.copy_shallow())
            payload = row.action_payloads[row.selected_action_row]
            matches = [
                action
                for action in env.legal_actions(state)
                if serialize_action(action) == payload
            ]
            if len(matches) != 1:
                raise ValueError(f"{label} selected action is not uniquely legal.")
            state = env.step(state, matches[0])
        last = game_rows[-1]
        if (
            not state.terminal
            or state.winner != last.winner
            or tuple(player.score for player in state.players) != last.final_scores
            or last.termination_reason != "completed"
        ):
            raise ValueError(f"{label} reconstructed terminal mismatch.")
        if len(eligible) < STATES_PER_GAME:
            raise ValueError(f"{label} game lacks two pre-round-end states.")
        for state_item in eligible[-STATES_PER_GAME:]:
            selected.append((game_seed, state_item))
            schedule.append(_state_identity(game_seed, state_item))
    expected_pairs = expected_games * STATES_PER_GAME
    if len(selected) != expected_pairs:
        raise ValueError(f"{label} did not yield {expected_pairs} states.")
    roles = {item["observer_is_start_player"] for item in schedule}
    if roles != {False, True}:
        raise ValueError(f"{label} lacks both role classes.")
    return selected, {
        "games": expected_games,
        "pairs": len(selected),
        "states_per_game": STATES_PER_GAME,
        "role_counts": {
            str(role).lower(): sum(item["observer_is_start_player"] is role for item in schedule)
            for role in (False, True)
        },
        "schedule": schedule,
    }


def split_pair_games(states: list[tuple[int, SplendorState]]) -> dict[str, Any]:
    games = sorted({game_seed for game_seed, _state in states})
    if len(games) != TRAIN_GAMES:
        raise ValueError("118 split requires 224 games.")
    shuffled = list(games)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    validation = set(shuffled[:VALIDATION_SPLIT_GAMES])
    train = set(shuffled[VALIDATION_SPLIT_GAMES:])
    train_pairs = sum(game_seed in train for game_seed, _state in states)
    validation_pairs = sum(game_seed in validation for game_seed, _state in states)
    if (
        len(train) != TRAIN_SPLIT_GAMES
        or len(validation) != VALIDATION_SPLIT_GAMES
        or train_pairs != TRAIN_SPLIT_PAIRS
        or validation_pairs != VALIDATION_SPLIT_PAIRS
        or train & validation
    ):
        raise ValueError("118 whole-game split has wrong counts.")
    return {
        "seed": SPLIT_SEED,
        "train_games": sorted(train),
        "validation_games": sorted(validation),
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "train_rows": 2 * train_pairs,
        "validation_rows": 2 * validation_pairs,
    }


def identity_audit(
    *,
    parent_checkpoint: Path,
    role_model,
    states: list[tuple[int, SplendorState]],
    device: str,
) -> dict[str, Any]:
    parent = load_checkpoint(parent_checkpoint, device=device)[0]
    max_logit = 0.0
    max_value = 0.0
    rows = 0
    role_model.eval()
    parent.eval()
    oriented: list[tuple[SplendorState, list[Action], int]] = []
    for game_seed, original in states:
        for state in (original, _flipped_state(original)):
            actions = _ordered_actions(SplendorEnv(seed=game_seed), state)
            oriented.append((state, actions, int(state.current_player == state.start_player)))
    with torch.no_grad():
        for start in range(0, len(oriented), 128):
            chunk = oriented[start : start + 128]
            batch = collate_entity_action_batch(
                [(state, actions) for state, actions, _role in chunk],
                device=device,
            )
            parent_logits, parent_values = parent(**batch)
            role_logits, role_values = role_model(
                **batch,
                observer_is_start_player=torch.tensor(
                    [role for _state, _actions, role in chunk],
                    dtype=torch.long,
                    device=device,
                ),
            )
            for index, (_state, actions, _role) in enumerate(chunk):
                width = len(actions)
                max_logit = max(
                    max_logit,
                    float(torch.abs(parent_logits[index, :width] - role_logits[index, :width]).max().item()),
                )
            max_value = max(max_value, float(torch.abs(parent_values - role_values).max().item()))
            rows += len(chunk)
    return {
        "rows": rows,
        "max_logit_abs_diff": max_logit,
        "max_value_abs_diff": max_value,
    }


def run_or_load_target_panel(
    config: StartRolePilotConfig,
    states: list[tuple[int, SplendorState]],
    *,
    confirmation: bool,
    protocol_sha256: str = PROTOCOL_SHA256,
    experiment_label: str = "118",
) -> list[dict[str, Any]]:
    expected_pairs = HOLDOUT_PAIRS if confirmation else TRAIN_PAIRS
    if len(states) != expected_pairs:
        raise ValueError("118 target panel has wrong pair count.")
    panel_name = "confirmation" if confirmation else "training"
    directory = config.output_dir / "targets" / panel_name
    directory.mkdir(parents=True, exist_ok=True)
    shard_count = 4
    per_shard = expected_pairs // shard_count
    tasks = []
    for shard in range(shard_count):
        start = shard * per_shard
        tasks.append(
            {
                "shard": shard,
                "states": states[start : start + per_shard],
                "panel_start": start,
                "output_dir": str(directory),
                "checkpoint": str(config.generation_11_checkpoint),
                "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
                "device": config.device,
                "confirmation": confirmation,
                "protocol_sha256": protocol_sha256,
                "experiment_label": experiment_label,
            }
        )
    shards: dict[int, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(_target_worker, task): task for task in tasks}
        for future in as_completed(futures):
            shard, rows = future.result()
            shards[shard] = rows
    if set(shards) != set(range(shard_count)):
        raise ValueError("118 target generation requires every exact shard.")
    rows = [row for shard in range(shard_count) for row in shards[shard]]
    if [row["identity"] for row in rows] != [
        _state_identity(game_seed, state) for game_seed, state in states
    ]:
        raise ValueError("118 combined target schedule mismatch.")
    return rows


def _target_worker(task: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    _limit_worker_threads()
    shard = int(task["shard"])
    states = list(task["states"])
    directory = Path(task["output_dir"])
    progress = directory / f"shard_{shard}.json"
    sidecar = directory / f"shard_{shard}.identity.json"
    confirmation = bool(task["confirmation"])
    protocol_sha256 = str(task.get("protocol_sha256", PROTOCOL_SHA256))
    experiment_label = str(task.get("experiment_label", "118"))
    identity = {
        "protocol_sha256": protocol_sha256,
        "shard": shard,
        "panel": "confirmation" if confirmation else "training",
        "panel_start": int(task["panel_start"]),
        "pairs": len(states),
        "checkpoint_sha256": task["checkpoint_sha256"],
        "simulations": SIMULATIONS,
        "c_puct": 2.25,
        "max_depth": 96,
        "root_noise_fraction": 0.0,
        "seed_formula": (
            "1117000 + 100 * panel_index"
            if confirmation
            else "1118000 + 100 * panel_index"
        ),
        "schedule": [_state_identity(game_seed, state) for game_seed, state in states],
    }
    if sidecar.exists():
        if json.loads(sidecar.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"{experiment_label} target shard {shard} identity mismatch.")
    else:
        if progress.exists():
            raise ValueError(
                f"{experiment_label} target shard {shard} progress exists without identity."
            )
        _write_json(sidecar, identity)
    completed: list[dict[str, Any]] = []
    if progress.exists():
        payload = json.loads(progress.read_text(encoding="utf-8"))
        completed = list(payload.get("rows", []))
        expected_prefix = identity["schedule"][: len(completed)]
        if [row["identity"] for row in completed] != expected_prefix:
            raise ValueError(
                f"{experiment_label} target shard {shard} resume prefix mismatch."
            )
    inference = AlphaZeroInference(task["checkpoint"], device=task["device"])
    for local_index in range(len(completed), len(states)):
        game_seed, state = states[local_index]
        panel_index = int(task["panel_start"]) + local_index
        completed.append(
            _run_target_pair(
                inference,
                game_seed,
                state,
                panel_index,
                confirmation=confirmation,
            )
        )
        _write_json(
            progress,
            {
                "identity": identity,
                "completed_pairs": len(completed),
                "target_pairs": len(states),
                "completed": len(completed) == len(states),
                "rows": completed,
            },
        )
    return shard, completed


def _run_target_pair(
    inference: AlphaZeroInference,
    game_seed: int,
    original: SplendorState,
    panel_index: int,
    *,
    confirmation: bool,
) -> dict[str, Any]:
    flipped = _flipped_state(original)
    _validate_role_alias(game_seed, original, flipped)
    seed = (1_117_000 if confirmation else TARGET_SEED_BASE) + 100 * panel_index
    orientations = {}
    for name, state in (("original", original), ("flipped", flipped)):
        env = SplendorEnv(seed=game_seed)
        actions = env.legal_actions(state)
        result = AlphaZeroSearch(inference, _search_config(seed)).run(
            env,
            state,
            actions,
            self_play=False,
            seed=seed,
        )
        by_key = {_action_key(edge.action): edge for edge in result.root_edges}
        keys = sorted(by_key)
        orientations[name] = {
            "observer_is_start_player": state.current_player == state.start_player,
            "action_keys": keys,
            "policy_target": [float(by_key[key].probability) for key in keys],
            "raw_network_priors": [float(by_key[key].network_prior) for key in keys],
            "search_root_value": float(result.search_root_value),
            "raw_network_value": float(result.raw_network_value),
            "visit_argmax_action_key": _action_key(result.root_edges[result.visit_argmax_row].action),
            "simulations": int(result.simulations),
            "max_depth_reached": int(result.max_depth_reached),
            "inference_calls": int(result.inference_calls),
            "elapsed_seconds": float(result.elapsed_seconds),
        }
    if orientations["original"]["action_keys"] != orientations["flipped"]["action_keys"]:
        raise ValueError("118 paired search returned different semantic action sets.")
    return {
        "identity": _state_identity(game_seed, original),
        "panel_index": panel_index,
        "seed": seed,
        "original": orientations["original"],
        "flipped": orientations["flipped"],
    }


def summarize_target_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    tvs = []
    value_deltas = []
    top_changes = 0
    for row in records:
        original = row["original"]
        flipped = row["flipped"]
        tvs.append(
            0.5
            * sum(
                abs(float(a) - float(b))
                for a, b in zip(original["policy_target"], flipped["policy_target"], strict=True)
            )
        )
        value_deltas.append(abs(float(original["search_root_value"]) - float(flipped["search_root_value"])))
        top_changes += original["visit_argmax_action_key"] != flipped["visit_argmax_action_key"]
    return {
        "pairs": len(records),
        "games": len({row["identity"]["game_seed"] for row in records}),
        "oriented_rows": 2 * len(records),
        "mean_visit_tv": sum(tvs) / len(tvs),
        "top_action_change_rate": top_changes / len(records),
        "mean_abs_root_value_delta": sum(value_deltas) / len(value_deltas),
        "all_exact_1024": all(
            orientation["simulations"] == SIMULATIONS
            for row in records
            for orientation in (row["original"], row["flipped"])
        ),
    }


def run_or_load_role_training(
    config: StartRolePilotConfig,
    states: list[tuple[int, SplendorState]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    stage = config.output_dir / "training" / "role_only.stage.json"
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("target_manifest_sha256")
            != _target_manifest_hash(config.output_dir / "targets" / "training")
            or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        ):
            raise ValueError("118 training stage identity mismatch.")
        return payload

    split = split_pair_games(states)
    validation_games = set(split["validation_games"])
    rows = _training_rows_from_records(states, records)
    train_rows = [row for row in rows if row.game_seed not in validation_games]
    validation_rows = [row for row in rows if row.game_seed in validation_games]
    if len(train_rows) != 716 or len(validation_rows) != 180:
        raise ValueError("118 oriented train/validation rows have wrong counts.")

    model, parent_identity = create_identity_role_model(
        config.generation_11_checkpoint,
        device=config.device,
    )
    optimizer = AdamW(
        model.start_role_embedding.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    train_loader = DataLoader(
        _RoleTargetDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SHUFFLE_SEED),
        collate_fn=_collate_role_target_rows,
    )
    validation_loader = DataLoader(
        _RoleTargetDataset(validation_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate_role_target_rows,
    )
    epoch_zero = _evaluate_target_loader(model, validation_loader, config.device)
    best_epoch = 0
    best_objective = float(epoch_zero["objective_loss"])
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = [
        {"epoch": 0, "train": None, "validation": epoch_zero}
    ]
    for epoch in range(1, EPOCHS + 1):
        train_metrics = _train_target_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
        )
        validation_metrics = _evaluate_target_loader(
            model,
            validation_loader,
            config.device,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        objective = float(validation_metrics["objective_loss"])
        if objective < best_objective:
            best_epoch = epoch
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected_train = _evaluate_target_loader(
        model,
        DataLoader(
            _RoleTargetDataset(train_rows),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=_collate_role_target_rows,
        ),
        config.device,
    )
    selected_validation = _evaluate_target_loader(
        model,
        validation_loader,
        config.device,
    )
    relative_improvement = (
        float(epoch_zero["objective_loss"])
        - float(selected_validation["objective_loss"])
    ) / float(epoch_zero["objective_loss"])
    selection_passed = best_epoch > 0 and relative_improvement >= 0.005
    checkpoint = config.output_dir / "checkpoints" / "start_role_generation_012.pt"
    save_role_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "parent_generation": parent_identity.generation,
            "selected_epoch": best_epoch,
            "train_games": split["train_games"],
            "validation_games": split["validation_games"],
            "role_only_training": True,
        },
    )
    roundtrip, identity = load_role_checkpoint(checkpoint, device=config.device)
    if (
        identity.generation != 12
        or identity.metadata.get("selected_epoch") != best_epoch
        or role_parameter_counts(roundtrip)["trainable"] != 192
    ):
        raise RuntimeError("118 trained checkpoint failed round-trip validation.")
    weights = roundtrip.start_role_embedding.weight.detach().cpu()
    metrics = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "gradient_clip": 5.0,
            "shuffle_seed": SHUFFLE_SEED,
            "policy_weight": 1.0,
            "value_weight": VALUE_WEIGHT,
        },
        "split": split,
        "selected_epoch": best_epoch,
        "epoch_zero_validation": epoch_zero,
        "selected_train": selected_train,
        "selected_validation": selected_validation,
        "relative_validation_objective_improvement": relative_improvement,
        "selection_passed": selection_passed,
        "embedding_row_norms": [float(torch.linalg.vector_norm(row).item()) for row in weights],
        "embedding_row_distance": float(torch.linalg.vector_norm(weights[0] - weights[1]).item()),
        "history": history,
        "parameter_counts": role_parameter_counts(roundtrip),
        "parent_frozen": all(not parameter.requires_grad for parameter in roundtrip.parent.parameters()),
    }
    metrics_path = config.output_dir / "training" / "role_only.json"
    _write_json(metrics_path, metrics)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "target_manifest_sha256": _target_manifest_hash(config.output_dir / "targets" / "training"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "selection_passed": selection_passed,
        "training": metrics,
    }
    _write_json(stage, payload)
    return payload


def _training_rows_from_records(
    states: list[tuple[int, SplendorState]],
    records: list[dict[str, Any]],
) -> list[_TrainingRow]:
    if len(states) != len(records):
        raise ValueError("118 states and target records differ in length.")
    rows: list[_TrainingRow] = []
    for (game_seed, original), record in zip(states, records, strict=True):
        if record["identity"] != _state_identity(game_seed, original):
            raise ValueError("118 target record state identity mismatch.")
        for name, state in (("original", original), ("flipped", _flipped_state(original))):
            actions = tuple(_ordered_actions(SplendorEnv(seed=game_seed), state))
            action_keys = [_action_key(action) for action in actions]
            target = record[name]
            if action_keys != target["action_keys"]:
                raise ValueError("118 target semantic action order mismatch.")
            policy = tuple(float(value) for value in target["policy_target"])
            if abs(sum(policy) - 1.0) > 1e-8:
                raise ValueError("118 policy target does not sum to one.")
            rows.append(
                _TrainingRow(
                    game_seed=game_seed,
                    panel_index=int(record["panel_index"]),
                    state=state,
                    actions=actions,
                    observer_is_start_player=bool(target["observer_is_start_player"]),
                    policy_target=policy,
                    value_target=float(target["search_root_value"]),
                )
            )
    return rows


def _collate_role_target_rows(rows: list[_TrainingRow]) -> dict[str, Any]:
    batch = collate_entity_action_batch(
        [(row.state, list(row.actions)) for row in rows]
    )
    width = int(batch["action_mask"].shape[1])
    policy = torch.zeros((len(rows), width), dtype=torch.float32)
    for index, row in enumerate(rows):
        policy[index, : len(row.policy_target)] = torch.tensor(row.policy_target)
    batch.update(
        observer_is_start_player=torch.tensor(
            [row.observer_is_start_player for row in rows], dtype=torch.long
        ),
        policy_target=policy,
        value_target=torch.tensor([row.value_target for row in rows], dtype=torch.float32),
        game_seed=torch.tensor([row.game_seed for row in rows], dtype=torch.long),
    )
    return batch


def _target_losses(
    logits: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
    policy_rows = -(policy_target * torch.log_softmax(masked, dim=1)).sum(dim=1)
    value_rows = torch.square(values - value_target)
    objective_rows = policy_rows + VALUE_WEIGHT * value_rows
    return policy_rows, value_rows, objective_rows


def _train_target_epoch(model, loader, optimizer, device: str) -> dict[str, float]:
    model.train()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "objective_loss": 0.0}
    count = 0
    for batch in loader:
        moved = {key: value.to(device) for key, value in batch.items()}
        inputs = _target_model_inputs(moved)
        logits, values = model(**inputs)
        policy, value, objective = _target_losses(
            logits,
            values,
            moved["action_mask"],
            moved["policy_target"],
            moved["value_target"],
        )
        loss = objective.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.start_role_embedding.parameters(), 5.0)
        optimizer.step()
        batch_count = int(values.shape[0])
        count += batch_count
        totals["policy_loss"] += float(policy.sum().item())
        totals["value_loss"] += float(value.sum().item())
        totals["objective_loss"] += float(objective.sum().item())
    model.eval()
    return {**{key: value / count for key, value in totals.items()}, "rows": count}


def _evaluate_target_loader(model, loader, device: str) -> dict[str, float]:
    model.eval()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "objective_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            logits, values = model(**_target_model_inputs(moved))
            policy, value, objective = _target_losses(
                logits,
                values,
                moved["action_mask"],
                moved["policy_target"],
                moved["value_target"],
            )
            count += int(values.shape[0])
            totals["policy_loss"] += float(policy.sum().item())
            totals["value_loss"] += float(value.sum().item())
            totals["objective_loss"] += float(objective.sum().item())
    return {**{key: value / count for key, value in totals.items()}, "rows": count}


def evaluate_role_confirmation(
    config: StartRolePilotConfig,
    states: list[tuple[int, SplendorState]],
    records: list[dict[str, Any]],
    *,
    candidate_checkpoint: Path,
) -> dict[str, Any]:
    summary = summarize_target_records(records)
    sealed_117 = json.loads(config.result_117.read_text(encoding="utf-8"))["counterfactual_teacher"]
    reproduction = {
        key: abs(float(summary[key]) - float(sealed_117[key]))
        for key in (
            "mean_visit_tv",
            "top_action_change_rate",
            "mean_abs_root_value_delta",
        )
    }
    if any(delta > 1e-9 for delta in reproduction.values()):
        raise ValueError("118 confirmation failed to reproduce sealed 117 teacher metrics.")

    checkpoints = {
        "generation_11": config.generation_11_checkpoint,
        "candidate_114": config.candidate_114,
        "candidate_116": config.candidate_116,
        "candidate_118": candidate_checkpoint,
    }
    predictions = {
        name: _confirmation_prediction_rows(path, states, records, config.device)
        for name, path in checkpoints.items()
    }
    means = {
        name: _loss_means(rows)
        for name, rows in predictions.items()
    }
    blind_names = ("generation_11", "candidate_114", "candidate_116")
    selected_baselines = {
        endpoint: min(blind_names, key=lambda name: means[name][endpoint])
        for endpoint in ("policy_loss", "value_loss", "objective_loss")
    }
    candidate = predictions["candidate_118"]
    endpoint_results: dict[str, Any] = {}
    endpoint_seeds = {
        "policy_loss": 1_118_120,
        "value_loss": 1_118_121,
        "objective_loss": 1_118_122,
    }
    for endpoint, baseline_name in selected_baselines.items():
        baseline = predictions[baseline_name]
        deltas = [
            {
                "game_seed": row["game_seed"],
                "value": float(row[endpoint]) - float(base[endpoint]),
            }
            for row, base in zip(candidate, baseline, strict=True)
        ]
        by_role = {}
        for role in (False, True):
            role_candidate = [row for row in candidate if row["observer_is_start_player"] is role]
            role_baseline = [row for row in baseline if row["observer_is_start_player"] is role]
            by_role[str(role).lower()] = {
                "candidate": sum(float(row[endpoint]) for row in role_candidate) / len(role_candidate),
                "baseline": sum(float(row[endpoint]) for row in role_baseline) / len(role_baseline),
            }
            by_role[str(role).lower()]["delta"] = (
                by_role[str(role).lower()]["candidate"] - by_role[str(role).lower()]["baseline"]
            )
        endpoint_results[endpoint] = {
            "baseline": baseline_name,
            "candidate": means["candidate_118"][endpoint],
            "baseline_loss": means[baseline_name][endpoint],
            "delta": means["candidate_118"][endpoint] - means[baseline_name][endpoint],
            "by_role": by_role,
            "bootstrap": _cluster_bootstrap_mean(deltas, samples=BOOTSTRAP_SAMPLES, seed=endpoint_seeds[endpoint]),
        }

    contrasts = _candidate_contrasts(candidate)
    policy_bootstrap = _cluster_bootstrap_mean(
        [{"game_seed": row["game_seed"], "value": row["policy_tv"]} for row in contrasts],
        samples=BOOTSTRAP_SAMPLES,
        seed=1_118_123,
    )
    value_bootstrap = _cluster_bootstrap_mean(
        [{"game_seed": row["game_seed"], "value": row["abs_value_delta"]} for row in contrasts],
        samples=BOOTSTRAP_SAMPLES,
        seed=1_118_124,
    )
    role_model, _identity = load_role_checkpoint(candidate_checkpoint, device=config.device)
    weights = role_model.start_role_embedding.weight.detach().cpu()
    checks = {
        "teacher_reproduction_within_1e_9": all(delta <= 1e-9 for delta in reproduction.values()),
        "policy_overall_improves": endpoint_results["policy_loss"]["delta"] < 0.0,
        "policy_both_roles_improve": all(item["delta"] < 0.0 for item in endpoint_results["policy_loss"]["by_role"].values()),
        "policy_ci_high_below_zero": endpoint_results["policy_loss"]["bootstrap"]["ci95_high"] < 0.0,
        "value_overall_improves": endpoint_results["value_loss"]["delta"] < 0.0,
        "value_both_roles_improve": all(item["delta"] < 0.0 for item in endpoint_results["value_loss"]["by_role"].values()),
        "value_ci_high_below_zero": endpoint_results["value_loss"]["bootstrap"]["ci95_high"] < 0.0,
        "objective_ci_high_below_zero": endpoint_results["objective_loss"]["bootstrap"]["ci95_high"] < 0.0,
        "raw_policy_tv_at_least_0_025": policy_bootstrap["mean"] >= 0.025,
        "raw_policy_tv_ci_low_above_0_01": policy_bootstrap["ci95_low"] > 0.01,
        "raw_value_delta_at_least_0_05": value_bootstrap["mean"] >= 0.05,
        "raw_value_delta_ci_low_above_0_01": value_bootstrap["ci95_low"] > 0.01,
        "both_embedding_rows_nonzero": all(float(torch.linalg.vector_norm(row).item()) > 0.0 for row in weights),
        "metrics_finite": _all_finite({"means": means, "endpoints": endpoint_results, "policy": policy_bootstrap, "value": value_bootstrap}),
    }
    return {
        "teacher_summary": summary,
        "teacher_reproduction_abs_error": reproduction,
        "loss_means": means,
        "selected_role_blind_baselines": selected_baselines,
        "endpoint_results": endpoint_results,
        "candidate_contrasts": {
            "mean_policy_tv": policy_bootstrap["mean"],
            "policy_tv_bootstrap": policy_bootstrap,
            "mean_abs_value_delta": value_bootstrap["mean"],
            "abs_value_delta_bootstrap": value_bootstrap,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _confirmation_prediction_rows(
    checkpoint: Path,
    states: list[tuple[int, SplendorState]],
    records: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    inference = AlphaZeroInference(checkpoint, device=device)
    rows: list[dict[str, Any]] = []
    for (game_seed, original), record in zip(states, records, strict=True):
        pair_rows = []
        for name, state in (("original", original), ("flipped", _flipped_state(original))):
            actions = _ordered_actions(SplendorEnv(seed=game_seed), state)
            target = record[name]
            if [_action_key(action) for action in actions] != target["action_keys"]:
                raise ValueError("118 confirmation action order mismatch.")
            prediction = inference.predict_one(state, actions)
            probabilities = torch.softmax(torch.tensor(prediction.logits), dim=0).tolist()
            policy_loss = -sum(
                float(target_value) * math.log(max(float(probability), 1e-30))
                for target_value, probability in zip(target["policy_target"], probabilities, strict=True)
            )
            value_loss = (float(prediction.value) - float(target["search_root_value"])) ** 2
            pair_rows.append(
                {
                    "game_seed": game_seed,
                    "panel_index": int(record["panel_index"]),
                    "orientation": name,
                    "observer_is_start_player": bool(target["observer_is_start_player"]),
                    "probabilities": probabilities,
                    "value": float(prediction.value),
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "objective_loss": policy_loss + VALUE_WEIGHT * value_loss,
                }
            )
        rows.extend(pair_rows)
    return rows


def _candidate_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) % 2:
        raise ValueError("118 candidate contrast rows are not paired.")
    contrasts = []
    for index in range(0, len(rows), 2):
        original, flipped = rows[index : index + 2]
        if original["orientation"] != "original" or flipped["orientation"] != "flipped":
            raise ValueError("118 candidate contrast orientation order mismatch.")
        contrasts.append(
            {
                "game_seed": original["game_seed"],
                "panel_index": original["panel_index"],
                "policy_tv": 0.5 * sum(
                    abs(float(a) - float(b))
                    for a, b in zip(original["probabilities"], flipped["probabilities"], strict=True)
                ),
                "abs_value_delta": abs(float(original["value"]) - float(flipped["value"])),
            }
        )
    return contrasts


def evaluate_reachable_safety(
    config: StartRolePilotConfig,
    rows: list[AlphaZeroReplayRow],
    *,
    candidate_checkpoint: Path,
) -> dict[str, Any]:
    groups: dict[str, list[AlphaZeroReplayRow]] = {"all": list(rows)}
    for role_name, role_id in (("start", 0), ("nonstart", 1)):
        role_rows = [row for row in rows if row.player_id == role_id]
        groups[f"role:{role_name}"] = role_rows
        for phase in ("early", "mid", "late"):
            groups[f"role:{role_name}|phase:{phase}"] = [
                row for row in role_rows if phase_for_turn(row.turn_index) == phase
            ]
    for phase in ("early", "mid", "late"):
        groups[f"phase:{phase}"] = [row for row in rows if phase_for_turn(row.turn_index) == phase]
    parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)[0]
    candidate = load_role_checkpoint(candidate_checkpoint, device=config.device)[0]
    metrics = {
        group_name: {
            "generation_11": _evaluate_replay_rows(parent, group_rows, config.device, role_aware=False),
            "candidate_118": _evaluate_replay_rows(candidate, group_rows, config.device, role_aware=True),
        }
        for group_name, group_rows in groups.items()
    }
    deltas = {
        name: {
            key: float(values["candidate_118"][key]) - float(values["generation_11"][key])
            for key in ("policy_loss", "value_loss", "objective_loss", "policy_entropy", "value_mae")
        }
        for name, values in metrics.items()
    }
    checks = {
        "all_objective_nonregression": deltas["all"]["objective_loss"] <= 0.005,
        "start_objective_nonregression": deltas["role:start"]["objective_loss"] <= 0.005,
        "nonstart_objective_nonregression": deltas["role:nonstart"]["objective_loss"] <= 0.005,
        "all_metrics_finite": _all_finite({"metrics": metrics, "deltas": deltas}),
    }
    return {
        "rows": len(rows),
        "games": len({row.game_seed for row in rows}),
        "metrics": metrics,
        "deltas_candidate_minus_gen11": deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _evaluate_replay_rows(model, rows: list[AlphaZeroReplayRow], device: str, *, role_aware: bool) -> dict[str, float]:
    from .training import collate_replay_rows

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "objective_loss": 0.0,
        "policy_entropy": 0.0,
        "value_mae": 0.0,
    }
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start : start + 256]
            batch = collate_replay_rows(chunk)
            moved = {key: value.to(device) for key, value in batch.items()}
            inputs = _base_model_inputs(moved)
            if role_aware:
                inputs["observer_is_start_player"] = torch.tensor(
                    [row.player_id == 0 for row in chunk], dtype=torch.long, device=device
                )
            logits, values = model(**inputs)
            policy, value, objective = _target_losses(
                logits,
                values,
                moved["action_mask"],
                moved["policy_target"],
                moved["value_target"],
            )
            masked = logits.masked_fill(~moved["action_mask"], torch.finfo(logits.dtype).min)
            probabilities = torch.softmax(masked, dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(dim=1)
            totals["policy_loss"] += float(policy.sum().item())
            totals["value_loss"] += float(value.sum().item())
            totals["objective_loss"] += float(objective.sum().item())
            totals["policy_entropy"] += float(entropy.sum().item())
            totals["value_mae"] += float(torch.abs(values - moved["value_target"]).sum().item())
            count += len(chunk)
    return {**{key: value / count for key, value in totals.items()}, "rows": count}


def run_or_load_arena(
    config: StartRolePilotConfig,
    candidate_checkpoint: Path,
    candidate_hash: str,
    *,
    protocol_sha256: str = PROTOCOL_SHA256,
    experiment_label: str = "118",
    seed_base: int = 2_180_000,
    bootstrap_seed: int = 1_118_200,
    arena_name: str = "start_role_joint_primary_256",
) -> dict[str, Any]:
    from .arena import ArenaResult
    from .search_primary import _arena_dict, paired_score_share_bootstrap

    stage = config.output_dir / "arenas" / "primary_256.stage.json"
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != protocol_sha256
            or payload.get("candidate_checkpoint_sha256") != candidate_hash
        ):
            raise ValueError(f"{experiment_label} arena stage identity mismatch.")
        return payload["combined"]
    tasks = [
        {
            "index": index,
            "output_dir": str(config.output_dir / "arenas"),
            "candidate_checkpoint": str(candidate_checkpoint),
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
            "protocol_sha256": protocol_sha256,
            "experiment_label": experiment_label,
            "seed_base": seed_base,
            "bootstrap_seed": bootstrap_seed,
            "arena_name": arena_name,
        }
        for index in range(4)
    ]
    shards: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
        futures = {executor.submit(_arena_worker, task): task for task in tasks}
        for future in as_completed(futures):
            index, arena = future.result()
            shards[index] = arena
    if set(shards) != {0, 1, 2, 3}:
        raise ValueError(f"{experiment_label} arena requires all four shards.")
    rows: list[dict[str, Any]] = []
    for index in range(4):
        shard = shards[index]
        start = seed_base + 50 * index
        expected = [(start + offset, seat) for offset in range(50) for seat in (0, 1)]
        observed = [(int(row["seed"]), int(row["candidate_seat"])) for row in shard["rows"]]
        if observed != expected or shard["games"] != 100:
            raise ValueError(f"{experiment_label} arena shard {index} schedule mismatch.")
        rows.extend(shard["rows"])
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    terminations: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        terminations[reason] = terminations.get(reason, 0) + 1
    combined_arena = _arena_dict(
        ArenaResult(
            name=arena_name,
            paired_seeds=200,
            games=400,
            candidate_wins=wins,
            candidate_losses=losses,
            draws=draws,
            score_share=(wins + 0.5 * draws) / 400.0,
            mean_margin=sum(float(row["candidate_margin"]) for row in rows) / 400.0,
            termination_reasons=dict(sorted(terminations.items())),
            rows=tuple(rows),
        )
    )
    bootstrap = paired_score_share_bootstrap(
        combined_arena,
        samples=BOOTSTRAP_SAMPLES,
        seed=bootstrap_seed,
    )
    seat_shares = {
        str(seat): sum(
            float(row["candidate_points"])
            for row in rows
            if int(row["candidate_seat"]) == seat
        ) / 200.0
        for seat in (0, 1)
    }
    checks = {
        "score_share_at_least_55pct": combined_arena["score_share"] >= 0.55,
        "positive_margin": combined_arena["mean_margin"] > 0.0,
        "both_seats_at_least_50pct": min(seat_shares.values()) >= 0.50,
        "all_normal_terminations": combined_arena["termination_reasons"] == {"completed": 400},
        "paired_ci95_low_strictly_above_50pct": bootstrap["ci95_low"] > 0.50,
    }
    combined = {
        "arena": combined_arena,
        "paired_bootstrap": bootstrap,
        "seat_score_shares": seat_shares,
        "assessment": {"checks": checks, "passed": all(checks.values())},
    }
    _write_json(
        stage,
        {
            "protocol_sha256": protocol_sha256,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "shards": {str(index): shards[index] for index in sorted(shards)},
            "combined": combined,
        },
    )
    return combined


def _arena_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    from .arena import alpha_zero_search_factory, run_paired_arena
    from .search_primary import _arena_dict

    _limit_worker_threads()
    index = int(task["index"])
    protocol_sha256 = str(task.get("protocol_sha256", PROTOCOL_SHA256))
    experiment_label = str(task.get("experiment_label", "118"))
    seed_base = int(task.get("seed_base", 2_180_000))
    bootstrap_seed = int(task.get("bootstrap_seed", 1_118_200))
    arena_name = str(task.get("arena_name", "start_role_joint_primary_256"))
    search = SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=bootstrap_seed,
    )
    candidate = alpha_zero_search_factory(task["candidate_checkpoint"], task["device"], search)
    baseline = alpha_zero_search_factory(task["baseline_checkpoint"], task["device"], search)
    arena_dir = Path(task["output_dir"])
    arena_dir.mkdir(parents=True, exist_ok=True)
    progress = arena_dir / f"primary_256_shard_{index}.json"
    sidecar = arena_dir / f"primary_256_shard_{index}.identity.json"
    identity = {
        "protocol_sha256": protocol_sha256,
        "shard": index,
        "candidate_checkpoint_sha256": task["candidate_checkpoint_sha256"],
        "baseline_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "seed_start": seed_base + 50 * index,
        "paired_seeds": 50,
        "simulations": 256,
        "c_puct": 2.25,
        "root_noise_fraction": 0.0,
        "operative_search_seed": "2 * game_seed + seat",
        "max_turns": int(task["max_turns"]),
    }
    if sidecar.exists():
        if json.loads(sidecar.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"{experiment_label} arena shard {index} identity mismatch.")
    else:
        if progress.exists():
            raise ValueError(
                f"{experiment_label} arena shard {index} progress exists without identity."
            )
        _write_json(sidecar, identity)
    arena = run_paired_arena(
        name=f"{arena_name}_shard_{index}",
        candidate_factory=candidate,
        baseline_factory=baseline,
        seed_start=seed_base + 50 * index,
        paired_seeds=50,
        max_turns=int(task["max_turns"]),
        progress_path=progress,
        resume=True,
    )
    return index, _arena_dict(arena)


def _verify_inputs(config: StartRolePilotConfig) -> None:
    paths = {
        "result_117": config.result_117,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "candidate_114": config.candidate_114,
        "candidate_116": config.candidate_116,
        "partition_manifest": config.partition_manifest,
        "train_replay": config.train_replay,
        "holdout_replay": config.holdout_replay,
        "installed_checkpoint": config.installed_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 118 input hash mismatch for {name}: {actual}")
    result_117 = json.loads(config.result_117.read_text(encoding="utf-8"))
    gate = result_117.get("interpretation_gate", {})
    if (
        result_117.get("status") != "complete"
        or result_117.get("decision")
        != "teacher_role_sensitivity_confirmed_review_identity_role_aware_pilot_no_training"
        or gate.get("integrity_passed") is not True
        or gate.get("policy_role_sensitivity") is not True
        or gate.get("value_role_sensitivity") is not True
        or result_117.get("promotion_authorized") is not False
    ):
        raise ValueError("Frozen 117 result does not authorize pilot 118 review.")
    manifest = json.loads(config.partition_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("train_sha256") != EXPECTED_HASHES["train_replay"]
        or manifest.get("holdout_sha256") != EXPECTED_HASHES["holdout_replay"]
        or len(manifest.get("train_seeds", [])) != TRAIN_GAMES
        or len(manifest.get("holdout_seeds", [])) != HOLDOUT_GAMES
        or set(manifest["train_seeds"]) & set(manifest["holdout_seeds"])
    ):
        raise ValueError("Frozen 118 partition is not exact and disjoint.")
    parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)[1]
    candidate_114 = load_checkpoint(config.candidate_114, device=config.device)[1]
    candidate_116 = load_capacity_checkpoint(config.candidate_116, device=config.device)[1]
    if (
        parent.generation != 11
        or candidate_114.metadata.get("parent_generation") != 11
        or candidate_116.metadata.get("parent_generation") != 11
    ):
        raise ValueError("Frozen 118 checkpoint lineage mismatch.")
    if (
        config.torch_threads != 1
        or config.workers != 4
        or config.arena_workers != 4
        or config.max_turns != 200
    ):
        raise ValueError("118 requires its frozen thread, worker, and turn settings.")


def _validate_role_alias(game_seed: int, original: SplendorState, flipped: SplendorState) -> None:
    if canonical_public_belief_key(original, original.current_player) == canonical_public_belief_key(flipped, flipped.current_player):
        raise ValueError("118 role flip did not change public-belief key.")
    if encode_canonical_state_entities(original, original.current_player) != encode_canonical_state_entities(flipped, flipped.current_player):
        raise ValueError("118 v1 tensor unexpectedly exposes start role.")
    original_actions = {_action_key(action) for action in SplendorEnv(seed=game_seed).legal_actions(original)}
    flipped_actions = {_action_key(action) for action in SplendorEnv(seed=game_seed).legal_actions(flipped)}
    if original_actions != flipped_actions:
        raise ValueError("118 role flip changed semantic legal actions.")


def _flipped_state(state: SplendorState) -> SplendorState:
    flipped = state.copy_shallow()
    flipped.start_player = 1 - state.start_player
    return flipped


def _ordered_actions(env: SplendorEnv, state: SplendorState) -> list[Action]:
    # AlphaZeroSearch expands one representative per canonical successor, so
    # direct teacher targets are indexed by that collapsed action set rather
    # than by every syntactically legal action.
    actions = actions_by_canonical_successor_key(state, env.legal_actions(state))
    return sorted(actions.values(), key=_action_key)


def _action_key(action: Action) -> str:
    return json.dumps(serialize_action(action), sort_keys=True, separators=(",", ":"))


def _state_identity(game_seed: int, state: SplendorState) -> dict[str, Any]:
    return {
        "game_seed": int(game_seed),
        "turn_index": int(state.turn_index),
        "current_player": int(state.current_player),
        "start_player": int(state.start_player),
        "observer_is_start_player": state.current_player == state.start_player,
        "scores": [int(player.score) for player in state.players],
    }


def _search_config(seed: int) -> SearchConfig:
    return SearchConfig(
        simulations=SIMULATIONS,
        c_puct=2.25,
        max_depth=96,
        mode="real",
        root_noise_fraction=0.0,
        seed=seed,
    )


def _base_model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in (
            "entity_type_ids",
            "entity_card_ids",
            "entity_noble_ids",
            "entity_numeric_features",
            "entity_mask",
            "action_type_ids",
            "action_card_ids",
            "action_noble_ids",
            "action_numeric_features",
            "action_mask",
        )
    }


def _target_model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        **_base_model_inputs(batch),
        "observer_is_start_player": batch["observer_is_start_player"],
    }


def _loss_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        endpoint: sum(float(row[endpoint]) for row in rows) / len(rows)
        for endpoint in ("policy_loss", "value_loss", "objective_loss")
    }


def _cluster_bootstrap_mean(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    clusters: dict[int, list[float]] = {}
    for row in rows:
        clusters.setdefault(int(row["game_seed"]), []).append(float(row["value"]))
    cluster_means = [sum(clusters[key]) / len(clusters[key]) for key in sorted(clusters)]
    if not cluster_means:
        raise ValueError("118 cluster bootstrap requires rows.")
    rng = random.Random(seed)
    draws = [
        sum(rng.choice(cluster_means) for _ in cluster_means) / len(cluster_means)
        for _ in range(samples)
    ]
    draws.sort()
    return {
        "games": len(cluster_means),
        "samples": samples,
        "seed": seed,
        "mean": sum(cluster_means) / len(cluster_means),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def _percentile(values: list[float], fraction: float) -> float:
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _target_manifest_hash(directory: Path) -> str:
    paths = [directory / f"shard_{index}.json" for index in range(TARGET_SHARDS)]
    if any(not path.exists() for path in paths):
        raise ValueError("118 training target manifest requires four shards.")
    payload = [(path.name, _sha256(path)) for path in paths]
    import hashlib

    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _all_finite(payload: Any) -> bool:
    if isinstance(payload, bool) or payload is None or isinstance(payload, str):
        return True
    if isinstance(payload, (int, float)):
        return math.isfinite(float(payload))
    if isinstance(payload, dict):
        return all(_all_finite(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(_all_finite(value) for value in payload)
    return True


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config: StartRolePilotConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})


def _seal(output: Path, result: dict[str, Any], started: float) -> dict[str, Any]:
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result
