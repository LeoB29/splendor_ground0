"""Guarded search-value distillation pilot 128."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from .arena import ArenaResult, alpha_zero_search_factory, run_paired_arena
from .high_budget_distillation import compression_assessment
from .indexed_training import (
    IndexedReplayDataset,
    ReplayIndexEntry,
    index_replay_paths,
    split_index_by_game,
)
from .network import load_checkpoint, save_checkpoint
from .replay import AlphaZeroReplayRow, _row_from_payload
from .scaled_continuation import _sha256, _write_json
from .search import SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .training import TrainingConfig, _compare_policies, _run_epoch, collate_replay_rows


@dataclass(frozen=True, slots=True)
class SearchValueDistillationConfig:
    output_dir: Path
    result_114: Path
    result_127: Path
    train_replay: Path
    holdout_replay: Path
    generation_11_checkpoint: Path
    terminal_control_checkpoint: Path
    installed_checkpoint: Path
    device: str = "cpu"
    torch_threads: int = 1
    arena_workers: int = 4
    max_turns: int = 200


PROTOCOL_SHA256 = "916e3dd79d83ff30808c414e31f2a74e27f5db4b99a874d2e66c1d811bb12b5d"
EXPECTED_HASHES = {
    "result_114": "dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5",
    "result_127": "ba7b76de3b201a2222d43e59c997d8b4ae8d743ff3bc59f1a0fd9f995fcda44d",
    "train_replay": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "holdout_replay": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "terminal_control_checkpoint": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
}
BLEND_WEIGHT = 0.5
TRAINING_SEED = 1_128_000
SIMULATIONS = 256
PAIRED_SEEDS = 200
ARENA_SHARDS = 4
BOOTSTRAP_SAMPLES = 50_000
PANEL_SPECS = {
    "primary_vs_terminal_control": {"seed_start": 2_600_000, "bootstrap_seed": 1_128_300},
    "secondary_vs_generation_11": {"seed_start": 2_610_000, "bootstrap_seed": 1_128_400},
}


def search_value_training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=12,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_fraction=0.2,
        seed=TRAINING_SEED,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        bucket_batches=True,
        policy_divergence_weight=0.0,
        visit_argmax_aux_weight=0.0,
    )


def blended_value_target(row: AlphaZeroReplayRow) -> float:
    if row.value_target is None:
        raise ValueError("Search-value distillation requires terminal WDL.")
    value = (1.0 - BLEND_WEIGHT) * float(row.value_target) + BLEND_WEIGHT * float(
        row.search_root_value
    )
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("Blended value target is invalid.")
    return value


class SearchValueReplayDataset(IndexedReplayDataset):
    def __getitem__(self, index: int) -> AlphaZeroReplayRow:
        row = super().__getitem__(index)
        return replace(row, value_target=blended_value_target(row))


def make_search_value_loader(
    paths: list[Path],
    entries: list[ReplayIndexEntry],
    *,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = SearchValueReplayDataset(paths, entries)
    actual_batch_size = min(config.batch_size, len(entries))
    indices = list(range(len(entries)))
    if config.bucket_batches:
        indices.sort(
            key=lambda index: (
                entries[index].action_count // 4,
                entries[index].entity_count // 8,
                entries[index].action_count,
                entries[index].entity_count,
            )
        )
    batches = [
        indices[start : start + actual_batch_size]
        for start in range(0, len(indices), actual_batch_size)
    ]
    if shuffle:
        rng = random.Random(config.seed)
        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=partial(collate_replay_rows, include_corrective_fields=True),
    )


def audit_search_value_signal(path: Path) -> dict[str, Any]:
    rows = 0
    games: set[int] = set()
    correction_sum = 0.0
    material_corrections = 0
    raw_squared_error = 0.0
    search_squared_error = 0.0
    checks = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = _row_from_payload(json.loads(line))
            rows += 1
            games.add(row.game_seed)
            values = (
                row.raw_network_value,
                row.search_root_value,
                float(row.value_target) if row.value_target is not None else math.nan,
                *row.root_q_values,
            )
            checks["finite_bounded_values"] += int(
                all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values)
            )
            checks["aligned_root_vectors"] += int(
                len(row.root_visits) == len(row.root_q_values) == len(row.policy_target)
            )
            total_visits = sum(row.root_visits)
            checks["exact_root_visits"] += int(total_visits == 1_024)
            weighted_q = (
                sum(q * visits for q, visits in zip(row.root_q_values, row.root_visits, strict=True))
                / total_visits
                if total_visits
                else math.nan
            )
            checks["root_value_reproduced"] += int(
                math.isfinite(weighted_q) and abs(weighted_q - row.search_root_value) <= 1e-9
            )
            checks["search_contract"] += int(
                row.termination_reason == "completed"
                and row.checkpoint_generation == 11
                and int(row.search_config.get("simulations", -1)) == 1_024
                and int(row.search_telemetry.get("simulations", -1)) == 1_024
            )
            correction = abs(row.search_root_value - row.raw_network_value)
            correction_sum += correction
            material_corrections += int(correction >= 0.05)
            target = float(row.value_target) if row.value_target is not None else math.nan
            raw_squared_error += (row.raw_network_value - target) ** 2
            search_squared_error += (row.search_root_value - target) ** 2
            blended_value_target(row)
    if rows == 0:
        raise ValueError("128 signal audit received no rows.")
    mean_correction = correction_sum / rows
    material_fraction = material_corrections / rows
    raw_mse = raw_squared_error / rows
    search_mse = search_squared_error / rows
    gate_checks = {
        "exact_224_games": len(games) == 224,
        "exact_13960_rows": rows == 13_960,
        "all_values_finite_and_bounded": checks["finite_bounded_values"] == rows,
        "all_root_vectors_aligned": checks["aligned_root_vectors"] == rows,
        "all_roots_have_1024_visits": checks["exact_root_visits"] == rows,
        "all_root_values_reproduced": checks["root_value_reproduced"] == rows,
        "all_rows_match_search_contract": checks["search_contract"] == rows,
        "mean_absolute_search_correction_at_least_0_02": mean_correction >= 0.02,
        "material_correction_fraction_at_least_0_10": material_fraction >= 0.10,
        "search_terminal_mse_not_grossly_worse": search_mse <= raw_mse + 0.05,
    }
    return {
        "rows": rows,
        "games": len(games),
        "blend_weight": BLEND_WEIGHT,
        "mean_absolute_search_correction": mean_correction,
        "material_correction_fraction": material_fraction,
        "raw_network_terminal_mse": raw_mse,
        "search_root_terminal_mse": search_mse,
        "checks": gate_checks,
        "qualified": all(gate_checks.values()),
    }


def search_value_offline_qualification(
    *, training: dict[str, Any], holdout: dict[str, dict[str, float]]
) -> dict[str, Any]:
    parent = holdout["generation_11"]
    control = holdout["terminal_control_114"]
    candidate = holdout["search_value_candidate"]
    finite = all(
        math.isfinite(float(value))
        for metrics in holdout.values()
        for value in metrics.values()
        if isinstance(value, (int, float))
    )
    checks = {
        "nonzero_epoch_selected": int(training.get("selected_epoch", 0)) > 0,
        "all_metrics_finite": finite,
        "policy_ce_improves_parent": candidate["policy_ce"] < parent["policy_ce"],
        "search_value_mse_improves_parent": candidate["search_value_mse"] < parent["search_value_mse"],
        "blended_value_mse_improves_parent": candidate["blended_value_mse"] < parent["blended_value_mse"],
        "blended_value_mse_improves_terminal_control": candidate["blended_value_mse"] < control["blended_value_mse"],
        "policy_ce_within_0_005_of_terminal_control": candidate["policy_ce"] <= control["policy_ce"] + 0.005,
        "terminal_mse_within_0_025_of_parent": candidate["terminal_value_mse"] <= parent["terminal_value_mse"] + 0.025,
        "terminal_mse_within_0_025_of_terminal_control": candidate["terminal_value_mse"] <= control["terminal_value_mse"] + 0.025,
    }
    return {"qualified": all(checks.values()), "checks": checks}


def search_value_decision(
    *, audit_passed: bool, offline_passed: bool | None, primary_passed: bool | None, secondary_passed: bool | None
) -> str:
    if not audit_passed:
        return "search_value_signal_audit_failed_no_training_no_install"
    if offline_passed is not True:
        return "search_value_blend_offline_failed_no_gameplay_no_install"
    if primary_passed is not True:
        return "search_value_blend_failed_direct_control_close_fixed_blend_no_install"
    if secondary_passed is not True:
        return "search_value_blend_beats_terminal_control_not_parent_robust_no_install"
    return "search_value_blend_transfers_deep_search_authorize_population_pilot_no_install"


def run_search_value_distillation_128(
    config: SearchValueDistillationConfig, *, launch: bool = False
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
        raise ValueError("Existing 128 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_search_value_distillation_128",
        "status": "running",
        "active_stage": "training_replay_search_value_audit",
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight": preflight,
        "signal_audit": None,
        "training": None,
        "holdout_audit": None,
        "panels": {},
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "automatic_continuation": False,
        "installed_checkpoint_unchanged": True,
        "decision": "running",
    }
    _write_progress(config.output_dir, result, started)
    signal = audit_search_value_signal(config.train_replay)
    result["signal_audit"] = signal
    _write_progress(config.output_dir, result, started)
    if not signal["qualified"]:
        return _seal(config, result, started, search_value_decision(
            audit_passed=False, offline_passed=None, primary_passed=None, secondary_passed=None
        ))
    result["active_stage"] = "parent_safe_blended_value_training"
    _write_progress(config.output_dir, result, started)
    training = _run_or_load_training(config)
    result["training"] = training
    result["checkpoint_trained"] = True
    result["active_stage"] = "sealed_holdout_audit"
    _write_progress(config.output_dir, result, started)
    holdout = _run_or_load_holdout_audit(config, training)
    result["holdout_audit"] = holdout
    _write_progress(config.output_dir, result, started)
    if not holdout["qualification"]["qualified"]:
        return _seal(config, result, started, search_value_decision(
            audit_passed=True, offline_passed=False, primary_passed=None, secondary_passed=None
        ))
    candidate = Path(training["output_checkpoint"])
    result["active_stage"] = "primary_vs_terminal_control"
    _write_progress(config.output_dir, result, started)
    primary = _run_or_load_panel(config, "primary_vs_terminal_control", candidate)
    result["panels"]["primary_vs_terminal_control"] = primary
    _write_progress(config.output_dir, result, started)
    if not primary["assessment"]["passed"]:
        return _seal(config, result, started, search_value_decision(
            audit_passed=True, offline_passed=True, primary_passed=False, secondary_passed=None
        ))
    result["active_stage"] = "secondary_vs_generation_11"
    _write_progress(config.output_dir, result, started)
    secondary = _run_or_load_panel(config, "secondary_vs_generation_11", candidate)
    result["panels"]["secondary_vs_generation_11"] = secondary
    _write_progress(config.output_dir, result, started)
    return _seal(config, result, started, search_value_decision(
        audit_passed=True,
        offline_passed=True,
        primary_passed=True,
        secondary_passed=bool(secondary["assessment"]["passed"]),
    ))


def _preflight(config: SearchValueDistillationConfig) -> dict[str, Any]:
    observed = {
        "result_114": _sha256(config.result_114),
        "result_127": _sha256(config.result_127),
        "train_replay": _sha256(config.train_replay),
        "holdout_replay": _sha256(config.holdout_replay),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "terminal_control_checkpoint": _sha256(config.terminal_control_checkpoint),
        "installed_checkpoint": _sha256(config.installed_checkpoint),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"128 frozen input mismatch: {observed}")
    result_114 = json.loads(config.result_114.read_text(encoding="utf-8"))
    result_127 = json.loads(config.result_127.read_text(encoding="utf-8"))
    if (
        result_114.get("status") != "complete"
        or result_114.get("training", {}).get("checkpoint_sha256") != EXPECTED_HASHES["terminal_control_checkpoint"]
        or result_127.get("status") != "complete"
        or result_127.get("decision") != "deep_search_refresh_endpoint_failed_close_current_distillation_recipe_no_install"
        or result_127.get("promotion_authorized") is not False
    ):
        raise ValueError("128 does not reproduce finalized 114/127 evidence.")
    _, parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    _, control = load_checkpoint(config.terminal_control_checkpoint, device=config.device)
    if (
        parent.generation != 11
        or control.generation != 12
        or control.metadata.get("parent_generation") != 11
        or config.torch_threads != 1
        or config.arena_workers != 4
        or config.max_turns != 200
    ):
        raise ValueError("128 lineage or execution configuration mismatch.")
    train = _stream_replay_identity(config.train_replay, expected_games=224, expected_rows=13_960)
    holdout = _stream_replay_identity(config.holdout_replay, expected_games=32, expected_rows=1_956)
    with config.train_replay.open("r", encoding="utf-8") as handle:
        sample = [_row_from_payload(json.loads(handle.readline())) for _ in range(2)]
    batch = collate_replay_rows(
        [replace(row, value_target=blended_value_target(row)) for row in sample],
        include_corrective_fields=True,
    )
    expected = torch.tensor([blended_value_target(row) for row in sample], dtype=torch.float32)
    if not torch.equal(batch["value_target"], expected):
        raise ValueError("128 blended-target batch construction is not exact.")
    sealed = config.output_dir / "result.json"
    if sealed.exists():
        payload = json.loads(sealed.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("128 output path contains a non-final incompatible result.")
    return {
        "kind": "alphazero_search_value_distillation_128_preflight",
        "status": "preflight_passed",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": observed,
        "train_replay": train,
        "holdout_replay": holdout,
        "blend_weight": BLEND_WEIGHT,
        "training_config": asdict(search_value_training_config()),
        "no_optimizer_instantiated": True,
        "no_training": True,
        "no_gameplay": True,
        "no_selfplay": True,
    }


def _stream_replay_identity(path: Path, *, expected_games: int, expected_rows: int) -> dict[str, Any]:
    rows = 0
    games: set[int] = set()
    terminations = Counter()
    for line in path.open("r", encoding="utf-8"):
        row = _row_from_payload(json.loads(line))
        rows += 1
        games.add(row.game_seed)
        terminations[str(row.termination_reason)] += 1
        if (
            row.generation != 11
            or row.checkpoint_generation != 11
            or row.value_target is None
            or int(row.search_config.get("simulations", -1)) != 1_024
            or int(row.search_telemetry.get("simulations", -1)) != 1_024
        ):
            raise ValueError("128 replay row violates frozen identity.")
    if rows != expected_rows or len(games) != expected_games or terminations != {"completed": rows}:
        raise ValueError("128 replay count or termination identity mismatch.")
    return {"path": str(path), "sha256": _sha256(path), "rows": rows, "games": len(games)}


def _run_or_load_training(config: SearchValueDistillationConfig) -> dict[str, Any]:
    metrics_path = config.output_dir / "training" / "metrics.json"
    checkpoint_path = config.output_dir / "checkpoints" / "search_value_generation_012.pt"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("output_checkpoint_sha256") != _sha256(checkpoint_path)
        ):
            raise ValueError("128 saved training stage identity mismatch.")
        return payload
    training = _train_parent_safe_search_value_successor(
        parent_checkpoint=config.generation_11_checkpoint,
        replay_path=config.train_replay,
        output_checkpoint=checkpoint_path,
        device=config.device,
        config=search_value_training_config(),
    )
    training["protocol_sha256"] = PROTOCOL_SHA256
    training["output_checkpoint_sha256"] = _sha256(checkpoint_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(metrics_path, training)
    return training


def _train_parent_safe_search_value_successor(
    *, parent_checkpoint: Path, replay_path: Path, output_checkpoint: Path, device: str, config: TrainingConfig
) -> dict[str, Any]:
    config.validate()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    paths, entries = index_replay_paths([replay_path])
    train_entries, validation_entries = split_index_by_game(
        entries, validation_fraction=config.validation_fraction, seed=config.seed
    )
    model, parent = load_checkpoint(parent_checkpoint, device=device)
    parent_state = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_search_value_loader(paths, train_entries, config=config, shuffle=True)
    validation_loader = make_search_value_loader(paths, validation_entries, config=config, shuffle=False)
    model.eval()
    parent_train_metrics = _run_epoch(model, train_loader, device, None, config)
    parent_validation_metrics = _run_epoch(model, validation_loader, device, None, config)
    best_state = deepcopy(parent_state)
    best_validation = float(parent_validation_metrics["objective_loss"])
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_metrics = _run_epoch(model, train_loader, device, optimizer, config)
        model.eval()
        validation_metrics = _run_epoch(model, validation_loader, device, None, config)
        if validation_metrics["objective_loss"] < best_validation - config.early_stopping_min_delta:
            best_validation = float(validation_metrics["objective_loss"])
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if config.early_stopping_patience is not None and without_improvement >= config.early_stopping_patience:
            stopped_early = True
            break
    model.load_state_dict(best_state)
    model.eval()
    child_train_metrics = _run_epoch(model, train_loader, device, None, config)
    child_validation_metrics = _run_epoch(model, validation_loader, device, None, config)
    parent_model = deepcopy(model)
    parent_model.load_state_dict(parent_state)
    parent_model.eval()
    comparison = _compare_policies(parent_model, model, validation_loader, device)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_checkpoint,
        model=model,
        generation=12,
        metadata={
            "parent_checkpoint": str(parent_checkpoint),
            "parent_generation": 11,
            "replay_paths": [str(replay_path)],
            "value_target_contract": "0.5_terminal_wdl_plus_0.5_search_root_value",
            "blend_weight": BLEND_WEIGHT,
            "training_config": asdict(config),
            "parent_safe_selection": True,
            "selected_epoch": best_epoch,
        },
    )
    result = {
        "parent_checkpoint": str(parent_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "parent_generation": parent.generation,
        "child_generation": 12,
        "replay_path": str(replay_path),
        "replay_sha256": _sha256(replay_path),
        "rows": len(entries),
        "train_rows": len(train_entries),
        "validation_rows": len(validation_entries),
        "train_games": len({entry.game_seed for entry in train_entries}),
        "validation_games": len({entry.game_seed for entry in validation_entries}),
        "blend_weight": BLEND_WEIGHT,
        "selection_metric": "blended_objective_with_epoch_zero_parent_eligible",
        "selected_epoch": best_epoch,
        "selected_parent_baseline": best_epoch == 0,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "parent_train_metrics": parent_train_metrics,
        "parent_validation_metrics": parent_validation_metrics,
        "child_train_metrics": child_train_metrics,
        "child_validation_metrics": child_validation_metrics,
        "validation_loss_delta_child_minus_parent": {
            key: child_validation_metrics[key] - parent_validation_metrics[key]
            for key in ("policy_loss", "value_loss", "total_loss", "objective_loss")
        },
        "child_parent_policy_comparison": comparison,
        "history": history,
        "config": asdict(config),
        "indexed_loading": True,
        "parent_safe_selection": True,
    }
    for loader in (train_loader, validation_loader):
        if isinstance(loader.dataset, IndexedReplayDataset):
            loader.dataset.close()
    return result


def _collate_holdout_rows(rows: list[AlphaZeroReplayRow]) -> dict[str, torch.Tensor]:
    batch = collate_replay_rows(rows, include_corrective_fields=True)
    terminal = torch.tensor([float(row.value_target) for row in rows], dtype=torch.float32)
    search = torch.tensor([float(row.search_root_value) for row in rows], dtype=torch.float32)
    batch["terminal_value_target"] = terminal
    batch["search_value_target"] = search
    batch["blended_value_target"] = (1.0 - BLEND_WEIGHT) * terminal + BLEND_WEIGHT * search
    return batch


def _evaluate_holdout_model(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    totals = Counter()
    rows = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            logits, values = model(
                entity_type_ids=moved["entity_type_ids"], entity_card_ids=moved["entity_card_ids"],
                entity_noble_ids=moved["entity_noble_ids"], entity_numeric_features=moved["entity_numeric_features"],
                entity_mask=moved["entity_mask"], action_type_ids=moved["action_type_ids"],
                action_card_ids=moved["action_card_ids"], action_noble_ids=moved["action_noble_ids"],
                action_numeric_features=moved["action_numeric_features"], action_mask=moved["action_mask"],
            )
            mask = moved["action_mask"]
            log_policy = torch.log_softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=1)
            count = int(values.shape[0])
            rows += count
            totals["policy_ce"] += float((-(moved["policy_target"] * log_policy).sum(dim=1)).sum().item())
            totals["policy_top_agreement"] += float(
                (torch.argmax(log_policy, dim=1) == torch.argmax(moved["policy_target"], dim=1)).sum().item()
            )
            for label, key in (
                ("terminal_value", "terminal_value_target"),
                ("search_value", "search_value_target"),
                ("blended_value", "blended_value_target"),
            ):
                error = values - moved[key]
                totals[f"{label}_mse"] += float((error ** 2).sum().item())
                totals[f"{label}_mae"] += float(torch.abs(error).sum().item())
    return {key: value / rows for key, value in totals.items()} | {"rows": rows}


def _run_or_load_holdout_audit(config: SearchValueDistillationConfig, training: dict[str, Any]) -> dict[str, Any]:
    path = config.output_dir / "holdout" / "audit.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("holdout_sha256") != EXPECTED_HASHES["holdout_replay"]
            or payload.get("candidate_checkpoint_sha256") != _sha256(training["output_checkpoint"])
        ):
            raise ValueError("128 holdout audit identity mismatch.")
        return payload
    paths, entries = index_replay_paths([config.holdout_replay])
    dataset = IndexedReplayDataset(paths, entries)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=_collate_holdout_rows)
    checkpoints = {
        "generation_11": config.generation_11_checkpoint,
        "terminal_control_114": config.terminal_control_checkpoint,
        "search_value_candidate": Path(training["output_checkpoint"]),
    }
    metrics = {}
    for name, checkpoint in checkpoints.items():
        model, _identity = load_checkpoint(checkpoint, device=config.device)
        metrics[name] = _evaluate_holdout_model(model, loader, config.device)
    qualification = search_value_offline_qualification(training=training, holdout=metrics)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "holdout_sha256": EXPECTED_HASHES["holdout_replay"],
        "candidate_checkpoint_sha256": _sha256(training["output_checkpoint"]),
        "checkpoint_hashes": {name: _sha256(path) for name, path in checkpoints.items()},
        "rows": len(entries),
        "games": len({entry.game_seed for entry in entries}),
        "metrics": metrics,
        "qualification": qualification,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    dataset.close()
    return payload


def _panel_baseline(config: SearchValueDistillationConfig, panel: str) -> Path:
    if panel == "primary_vs_terminal_control":
        return config.terminal_control_checkpoint
    if panel == "secondary_vs_generation_11":
        return config.generation_11_checkpoint
    raise ValueError(f"Unknown 128 panel: {panel}")


def _run_or_load_panel(config: SearchValueDistillationConfig, panel: str, candidate: Path) -> dict[str, Any]:
    panel_dir = config.output_dir / "arenas" / panel
    manifest = panel_dir / "panel.stage.json"
    candidate_hash = _sha256(candidate)
    baseline = _panel_baseline(config, panel)
    baseline_hash = _sha256(baseline)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("candidate_checkpoint_sha256") != candidate_hash
            or payload.get("baseline_checkpoint_sha256") != baseline_hash
            or set(payload.get("shards", {})) != {"0", "1", "2", "3"}
        ):
            raise ValueError("128 arena manifest identity mismatch.")
        combined = _combine_panel(panel, {int(key): value for key, value in payload["shards"].items()})
    else:
        tasks = [
            {
                "panel": panel, "shard": shard, "panel_dir": str(panel_dir),
                "candidate": str(candidate), "candidate_hash": candidate_hash,
                "baseline": str(baseline), "baseline_hash": baseline_hash,
                "device": config.device, "max_turns": config.max_turns,
            }
            for shard in range(ARENA_SHARDS)
        ]
        shards: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
            futures = {executor.submit(_panel_shard_worker, task): task for task in tasks}
            for future in as_completed(futures):
                shard, arena = future.result()
                shards[shard] = arena
        combined = _combine_panel(panel, shards)
        _write_json(manifest, {
            "protocol_sha256": PROTOCOL_SHA256,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": baseline_hash,
            "shards": {str(key): shards[key] for key in sorted(shards)},
        })
    bootstrap = paired_score_share_bootstrap(
        combined, samples=BOOTSTRAP_SAMPLES, seed=int(PANEL_SPECS[panel]["bootstrap_seed"])
    )
    return {
        "arena": combined,
        "paired_bootstrap": bootstrap,
        "assessment": compression_assessment(combined, bootstrap),
    }


def _panel_shard_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    _limit_worker_threads()
    panel = str(task["panel"])
    shard = int(task["shard"])
    per_shard = PAIRED_SEEDS // ARENA_SHARDS
    seed_start = int(PANEL_SPECS[panel]["seed_start"]) + shard * per_shard
    panel_dir = Path(task["panel_dir"])
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "panel": panel,
        "shard": shard,
        "candidate_checkpoint_sha256": task["candidate_hash"],
        "baseline_checkpoint_sha256": task["baseline_hash"],
        "seed_start": seed_start,
        "paired_seeds": per_shard,
        "search": asdict(_arena_search()),
        "max_turns": int(task["max_turns"]),
    }
    identity_path = panel_dir / f"shard_{shard}.identity.json"
    progress_path = panel_dir / f"shard_{shard}.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("128 arena shard identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError("128 arena progress lacks identity.")
        _write_json(identity_path, identity)
    arena = _arena_dict(run_paired_arena(
        name=f"{panel}_shard_{shard}",
        candidate_factory=alpha_zero_search_factory(task["candidate"], task["device"], _arena_search()),
        baseline_factory=alpha_zero_search_factory(task["baseline"], task["device"], _arena_search()),
        seed_start=seed_start,
        paired_seeds=per_shard,
        max_turns=int(task["max_turns"]),
        progress_path=progress_path,
        resume=True,
    ))
    return shard, arena


def _combine_panel(panel: str, shards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(shards) != set(range(ARENA_SHARDS)):
        raise ValueError("128 panel requires four exact shards.")
    per_shard = PAIRED_SEEDS // ARENA_SHARDS
    rows: list[dict[str, Any]] = []
    for shard in range(ARENA_SHARDS):
        start = int(PANEL_SPECS[panel]["seed_start"]) + shard * per_shard
        expected = [(start + offset, seat) for offset in range(per_shard) for seat in (0, 1)]
        shard_rows = list(shards[shard]["rows"])
        observed = [(int(row["seed"]), int(row["candidate_seat"])) for row in shard_rows]
        if observed != expected:
            raise ValueError("128 arena shard schedule mismatch.")
        rows.extend(shard_rows)
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    reasons = Counter(str(row["termination_reason"]) for row in rows)
    return _arena_dict(ArenaResult(
        name=panel,
        paired_seeds=PAIRED_SEEDS,
        games=len(rows),
        candidate_wins=wins,
        candidate_losses=losses,
        draws=draws,
        score_share=(wins + 0.5 * draws) / len(rows),
        mean_margin=sum(float(row["candidate_margin"]) for row in rows) / len(rows),
        termination_reasons=dict(sorted(reasons.items())),
        rows=tuple(rows),
    ))


def _arena_search() -> SearchConfig:
    return SearchConfig(
        simulations=SIMULATIONS,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_128_200,
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


def _seal(config: SearchValueDistillationConfig, result: dict[str, Any], started: float, decision: str) -> dict[str, Any]:
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = decision
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(config.output_dir / "result.json", result)
    return result
