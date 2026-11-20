"""Fresh-dose value/gradient/head-swap attribution experiment 111."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch

from splendor_ai.encoding.entity_encoding import ENTITY_OPPONENT, ENTITY_SELF

from .arena import alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .head_attribution import paired_arena_difference_bootstrap
from .indexed_training import IndexedReplayDataset, index_replay_paths
from .network import HybridAlphaZeroInference, load_checkpoint
from .replay import AlphaZeroReplayRow, _row_from_payload
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearchBot, SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .training import collate_replay_rows


@dataclass(frozen=True, slots=True)
class FreshHeadAttributionConfig:
    output_dir: Path
    result_110: Path
    installed_checkpoint: Path
    generation_11_checkpoint: Path
    high_dose_checkpoint: Path
    high_dose_replay: Path
    common_holdout: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None
    arena_workers: int = 3


PROTOCOL_SHA256 = "01fa9c87d5c34be688eed6f9a1275fea339b56832b5172a172be46ebc80af715"
EXPECTED_HASHES = {
    "result_110": "df35d3893f4c184ee24906b884aef33542c954380bf03c0b45f14a6d829f34c2",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "high_dose_checkpoint": "a9553311c5e49df4c809ad9238e9f769aa2307f562816c78e0efcadeb282fa74",
    "high_dose_replay": "8cc5a04f7fca7a76e946605f99b270ae7a3e9e7440c9b97e9ee9479255ea0fdc",
    "common_holdout": "bdaf7d8015b0d5295904724191e04c2a3081cdb6e200203452ea5c00b1e6c717",
}
ARENA_SEED_START = 2_060_000
PAIRED_SEEDS = 200
BOOTSTRAP_SAMPLES = 50_000
GRADIENT_SAMPLE_SEED = 1_110_111
GRADIENT_ROWS = 2_048
GRADIENT_BATCH_SIZE = 128
PANEL_VARIANTS = (
    "full_high_dose",
    "restore_parent_value",
    "restore_parent_policy",
)
_SHARED_PREFIXES = (
    "entity_type_embedding.",
    "card_embedding.",
    "noble_embedding.",
    "entity_input.",
    "entity_encoder.",
)


def head_attribution_decision(
    *, value_restore_difference: dict[str, Any], policy_restore_difference: dict[str, Any]
) -> str:
    value_positive = float(value_restore_difference["ci95_low"]) > 0.0
    policy_positive = float(policy_restore_difference["ci95_low"]) > 0.0
    if value_positive and not policy_positive:
        return "high_dose_value_update_primary_blocker"
    if policy_positive and not value_positive:
        return "high_dose_policy_update_primary_blocker"
    if value_positive and policy_positive:
        return "both_high_dose_updates_or_joint_incompatibility"
    return "no_clear_head_attribution_review_representation_capacity_or_teacher"


def run_fresh_head_attribution_111(config: FreshHeadAttributionConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 111 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_fresh_head_attribution_111",
        "status": "running",
        "active_stage": "streaming_value_audit",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "value_audit": None,
        "gradient_conflict_audit": None,
        "panels": {},
        "paired_head_differences": None,
        "attribution": None,
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    result["value_audit"] = _run_or_load_value_audit(config)
    result["active_stage"] = "shared_gradient_conflict_audit"
    _write_progress(output, result, started)
    result["gradient_conflict_audit"] = _run_or_load_gradient_audit(config)
    result["active_stage"] = "parallel_same_seed_head_gameplay"
    _write_progress(output, result, started)
    arenas = _run_or_load_parallel_panels(config)
    for index, name in enumerate(PANEL_VARIANTS):
        arena = arenas[name]
        result["panels"][name] = {
            "arena": arena,
            "paired_bootstrap": paired_score_share_bootstrap(
                arena, samples=BOOTSTRAP_SAMPLES, seed=1_111_300 + index
            ),
            "assessment": practical_assessment(arena),
        }
    full = result["panels"]["full_high_dose"]["arena"]
    value_restore = paired_arena_difference_bootstrap(
        result["panels"]["restore_parent_value"]["arena"],
        full,
        samples=BOOTSTRAP_SAMPLES,
        seed=1_111_400,
    )
    policy_restore = paired_arena_difference_bootstrap(
        result["panels"]["restore_parent_policy"]["arena"],
        full,
        samples=BOOTSTRAP_SAMPLES,
        seed=1_111_401,
    )
    result["paired_head_differences"] = {
        "restore_parent_value_minus_full_high_dose": value_restore,
        "restore_parent_policy_minus_full_high_dose": policy_restore,
    }
    decision = head_attribution_decision(
        value_restore_difference=value_restore,
        policy_restore_difference=policy_restore,
    )
    result["attribution"] = {
        "decision": decision,
        "primary_endpoint": "restore_parent_value_minus_full_high_dose_paired_score_share",
        "rubric": "strict_paired_bootstrap_ci95_lower_bound_above_zero",
        "direct_practical_passes": {
            name: bool(result["panels"][name]["assessment"]["passed"])
            for name in PANEL_VARIANTS
        },
    }
    result["decision"] = decision + "_no_install"
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def audit_holdout_values(
    *,
    parent_checkpoint: Path,
    high_checkpoint: Path,
    replay_path: Path,
    training_replay_path: Path,
    device: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    models = {
        "parent": load_checkpoint(parent_checkpoint, device=device)[0],
        "high_dose": load_checkpoint(high_checkpoint, device=device)[0],
    }
    for model in models.values():
        model.eval()
    score_tempo = _fit_score_tempo_baseline(training_replay_path)
    accumulators = {
        phase: _new_value_accumulator(models)
        for phase in ("overall", "early_lt20", "middle_20_49", "late_ge50")
    }
    games: set[int] = set()
    rows_seen = 0
    for chunk in _replay_chunks(replay_path, batch_size):
        batch = collate_replay_rows(chunk)
        moved = {key: value.to(device) for key, value in batch.items()}
        inputs = {
            key: moved[key]
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
        predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for name, model in models.items():
                predictions[name] = model(**inputs)
        floor = torch.finfo(next(iter(predictions.values()))[0].dtype).min
        probabilities = {
            name: torch.softmax(logits.masked_fill(~moved["action_mask"], floor), dim=1)
            for name, (logits, _values) in predictions.items()
        }
        for local, row in enumerate(chunk):
            games.add(row.game_seed)
            rows_seen += 1
            targets = [float(value) for value in row.policy_target]
            phase = _phase(row.turn_index)
            for accumulator_name in ("overall", phase):
                accumulator = accumulators[accumulator_name]
                _accumulate_zero(accumulator["zero_predictor"], float(row.value_target))
                _accumulate_baseline_value(
                    accumulator["score_tempo_baseline"],
                    predicted=_score_tempo_prediction(row, score_tempo["coefficients"]),
                    target=float(row.value_target),
                )
                for model_name in models:
                    count = len(targets)
                    probs = [
                        float(value)
                        for value in probabilities[model_name][local, :count].tolist()
                    ]
                    value = float(predictions[model_name][1][local].item())
                    _accumulate_model_value(
                        accumulator[model_name],
                        predicted=value,
                        target=float(row.value_target),
                        policy_target=targets,
                        policy_probs=probs,
                    )
        del batch, moved, inputs, predictions, probabilities
    if not rows_seen:
        raise ValueError("111 common holdout is empty.")
    if rows_seen != 7_986 or len(games) != 128:
        raise ValueError("111 common holdout row/game coverage changed.")
    return {
        "rows": rows_seen,
        "games": len(games),
        "score_tempo_baseline": score_tempo,
        "phase_definition": {
            "early_lt20": "turn 0-19",
            "middle_20_49": "turn 20-49",
            "late_ge50": "turn 50+",
        },
        "slices": {
            name: _finalize_value_accumulator(accumulator)
            for name, accumulator in accumulators.items()
        },
    }


def audit_shared_gradient_conflict(
    *,
    parent_checkpoint: Path,
    high_checkpoint: Path,
    replay_path: Path,
    device: str,
) -> dict[str, Any]:
    paths, entries = index_replay_paths([replay_path])
    if len(entries) < GRADIENT_ROWS:
        raise ValueError("111 high-dose replay is too small for the frozen gradient sample.")
    if len(entries) != 55_806:
        raise ValueError("111 high-dose replay row coverage changed.")
    sampled_indices = random.Random(GRADIENT_SAMPLE_SEED).sample(
        range(len(entries)), GRADIENT_ROWS
    )
    sampled_entries = [entries[index] for index in sampled_indices]
    result: dict[str, Any] = {
        "source_rows": len(entries),
        "sample_rows": len(sampled_entries),
        "sample_seed": GRADIENT_SAMPLE_SEED,
        "batch_size": GRADIENT_BATCH_SIZE,
        "batches": math.ceil(len(sampled_entries) / GRADIENT_BATCH_SIZE),
        "shared_parameter_prefixes": list(_SHARED_PREFIXES),
        "models": {},
    }
    for name, checkpoint in (
        ("parent", parent_checkpoint),
        ("high_dose", high_checkpoint),
    ):
        model = load_checkpoint(checkpoint, device=device)[0]
        model.eval()
        shared = _shared_parameters(model)
        dataset = IndexedReplayDataset(paths, sampled_entries)
        batches: list[dict[str, float]] = []
        try:
            for start in range(0, len(dataset), GRADIENT_BATCH_SIZE):
                rows = [
                    dataset[index]
                    for index in range(start, min(start + GRADIENT_BATCH_SIZE, len(dataset)))
                ]
                batches.append(_gradient_batch_metrics(model, shared, rows, device))
                del rows
        finally:
            dataset.close()
        result["models"][name] = _summarize_gradient_batches(batches)
        del model, shared
        gc.collect()
    return result


def _run_or_load_value_audit(config: FreshHeadAttributionConfig) -> dict[str, Any]:
    path = config.output_dir / "audit" / "value_audit.stage.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_stage_identity(payload, "value_audit")
        return payload["audit"]
    audit = audit_holdout_values(
        parent_checkpoint=config.generation_11_checkpoint,
        high_checkpoint=config.high_dose_checkpoint,
        replay_path=config.common_holdout,
        training_replay_path=config.high_dose_replay,
        device=config.device,
    )
    _write_json(path, _stage_payload("value_audit", audit))
    return audit


def _run_or_load_gradient_audit(config: FreshHeadAttributionConfig) -> dict[str, Any]:
    path = config.output_dir / "audit" / "gradient_conflict.stage.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_stage_identity(payload, "gradient_conflict")
        return payload["audit"]
    audit = audit_shared_gradient_conflict(
        parent_checkpoint=config.generation_11_checkpoint,
        high_checkpoint=config.high_dose_checkpoint,
        replay_path=config.high_dose_replay,
        device=config.device,
    )
    _write_json(path, _stage_payload("gradient_conflict", audit))
    return audit


def _run_or_load_parallel_panels(
    config: FreshHeadAttributionConfig,
) -> dict[str, dict[str, Any]]:
    manifest = config.output_dir / "arenas" / "parallel_panels.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_stage_identity(payload, "parallel_panels")
        if set(payload.get("arenas", {})) != set(PANEL_VARIANTS):
            raise ValueError("111 parallel panel manifest is incomplete.")
        return payload["arenas"]
    tasks = [
        {
            "variant": variant,
            "output_dir": str(config.output_dir / "arenas"),
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "high_checkpoint": str(config.high_dose_checkpoint),
            "device": config.device,
            "max_turns": config.max_turns,
        }
        for variant in PANEL_VARIANTS
    ]
    arenas: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(config.arena_workers, len(tasks))) as executor:
        futures = {executor.submit(_panel_worker, task): task for task in tasks}
        for future in as_completed(futures):
            name, arena = future.result()
            arenas[name] = arena
    if set(arenas) != set(PANEL_VARIANTS):
        raise RuntimeError("111 did not complete every frozen gameplay panel.")
    _write_json(
        manifest,
        {
            **_stage_payload("parallel_panels", None),
            "arena_workers": min(config.arena_workers, len(tasks)),
            "arenas": arenas,
        },
    )
    return arenas


def _panel_worker(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _limit_worker_threads()
    variant = str(task["variant"])
    parent = Path(task["parent_checkpoint"])
    high = Path(task["high_checkpoint"])
    search = _evaluation_search()
    if variant == "full_high_dose":
        candidate = alpha_zero_search_factory(high, task["device"], search)
    elif variant == "restore_parent_value":
        candidate = _hybrid_search_factory(high, parent, task["device"], search)
    elif variant == "restore_parent_policy":
        candidate = _hybrid_search_factory(parent, high, task["device"], search)
    else:
        raise ValueError(f"Unknown 111 panel variant: {variant}")
    baseline = alpha_zero_search_factory(parent, task["device"], search)
    progress_path = Path(task["output_dir"]) / f"{variant}.json"
    identity_path = Path(task["output_dir"]) / f"{variant}.identity.json"
    identity = _panel_identity(task)
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"111 {variant} panel sidecar identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError(f"111 {variant} progress exists without an identity sidecar.")
        _write_json(identity_path, identity)
    arena = _arena_dict(
        run_paired_arena(
            name=variant,
            candidate_factory=candidate,
            baseline_factory=baseline,
            seed_start=ARENA_SEED_START,
            paired_seeds=PAIRED_SEEDS,
            max_turns=int(task["max_turns"]),
            progress_path=progress_path,
            resume=True,
        )
    )
    return variant, arena


def _panel_identity(task: dict[str, Any]) -> dict[str, Any]:
    variant = str(task["variant"])
    if variant == "full_high_dose":
        policy_hash = EXPECTED_HASHES["high_dose_checkpoint"]
        value_hash = EXPECTED_HASHES["high_dose_checkpoint"]
    elif variant == "restore_parent_value":
        policy_hash = EXPECTED_HASHES["high_dose_checkpoint"]
        value_hash = EXPECTED_HASHES["generation_11_checkpoint"]
    elif variant == "restore_parent_policy":
        policy_hash = EXPECTED_HASHES["generation_11_checkpoint"]
        value_hash = EXPECTED_HASHES["high_dose_checkpoint"]
    else:
        raise ValueError(f"Unknown 111 panel variant: {variant}")
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "variant": variant,
        "policy_checkpoint_sha256": policy_hash,
        "value_checkpoint_sha256": value_hash,
        "baseline_policy_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "baseline_value_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "search_config": asdict(_evaluation_search()),
        "operative_search_seed": "2 * game_seed + seat",
        "seed_start": ARENA_SEED_START,
        "paired_seeds": PAIRED_SEEDS,
        "max_turns": int(task["max_turns"]),
    }


def _hybrid_search_factory(
    policy_checkpoint: Path,
    value_checkpoint: Path,
    device: str,
    config: SearchConfig,
):
    inference = HybridAlphaZeroInference(policy_checkpoint, value_checkpoint, device=device)
    evaluation = replace(config, root_noise_fraction=0.0)

    def factory(seed: int, seat: int) -> AlphaZeroSearchBot:
        return AlphaZeroSearchBot(
            config=replace(evaluation, seed=seed * 2 + seat),
            self_play=False,
            inference=inference,  # type: ignore[arg-type]
        )

    return factory


def _new_value_accumulator(models: dict[str, torch.nn.Module]) -> dict[str, Any]:
    return {
        "zero_predictor": {"count": 0, "absolute_error": 0.0, "squared_error": 0.0},
        "score_tempo_baseline": {
            "count": 0,
            "prediction_sum": 0.0,
            "error_sum": 0.0,
            "absolute_error": 0.0,
            "squared_error": 0.0,
            "sign_correct": 0,
            "wins": [],
            "losses": [],
            "calibration": [
                {"count": 0, "prediction_sum": 0.0, "target_sum": 0.0}
                for _ in range(10)
            ],
        },
        **{
            name: {
                "count": 0,
                "prediction_sum": 0.0,
                "error_sum": 0.0,
                "absolute_error": 0.0,
                "squared_error": 0.0,
                "sign_correct": 0,
                "wins": [],
                "losses": [],
                "policy_cross_entropy": 0.0,
                "policy_top_correct": 0,
                "calibration": [
                    {"count": 0, "prediction_sum": 0.0, "target_sum": 0.0}
                    for _ in range(10)
                ],
            }
            for name in models
        },
    }


def _accumulate_zero(accumulator: dict[str, Any], target: float) -> None:
    accumulator["count"] += 1
    accumulator["absolute_error"] += abs(target)
    accumulator["squared_error"] += target * target


def _accumulate_baseline_value(
    accumulator: dict[str, Any], *, predicted: float, target: float
) -> None:
    error = predicted - target
    accumulator["count"] += 1
    accumulator["prediction_sum"] += predicted
    accumulator["error_sum"] += error
    accumulator["absolute_error"] += abs(error)
    accumulator["squared_error"] += error * error
    accumulator["sign_correct"] += _sign(predicted) == _sign(target)
    if target > 0:
        accumulator["wins"].append(predicted)
    elif target < 0:
        accumulator["losses"].append(predicted)
    bin_index = min(9, max(0, int(((predicted + 1.0) / 2.0) * 10.0)))
    calibration = accumulator["calibration"][bin_index]
    calibration["count"] += 1
    calibration["prediction_sum"] += predicted
    calibration["target_sum"] += target


def _accumulate_model_value(
    accumulator: dict[str, Any],
    *,
    predicted: float,
    target: float,
    policy_target: list[float],
    policy_probs: list[float],
) -> None:
    error = predicted - target
    accumulator["count"] += 1
    accumulator["prediction_sum"] += predicted
    accumulator["error_sum"] += error
    accumulator["absolute_error"] += abs(error)
    accumulator["squared_error"] += error * error
    accumulator["sign_correct"] += _sign(predicted) == _sign(target)
    if target > 0:
        accumulator["wins"].append(predicted)
    elif target < 0:
        accumulator["losses"].append(predicted)
    accumulator["policy_cross_entropy"] += -sum(
        target_value * math.log(max(probability, 1e-12))
        for target_value, probability in zip(policy_target, policy_probs, strict=True)
    )
    accumulator["policy_top_correct"] += _argmax(policy_target) == _argmax(policy_probs)
    bin_index = min(9, max(0, int(((predicted + 1.0) / 2.0) * 10.0)))
    calibration = accumulator["calibration"][bin_index]
    calibration["count"] += 1
    calibration["prediction_sum"] += predicted
    calibration["target_sum"] += target


def _finalize_value_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = int(accumulator["zero_predictor"]["count"])
    if count == 0:
        return {"count": 0}
    result: dict[str, Any] = {
        "count": count,
        "zero_predictor": {
            "value_mae": accumulator["zero_predictor"]["absolute_error"] / count,
            "value_mse": accumulator["zero_predictor"]["squared_error"] / count,
        },
    }
    baseline = accumulator["score_tempo_baseline"]
    result["score_tempo_baseline"] = {
        "value_mean": baseline["prediction_sum"] / count,
        "value_bias": baseline["error_sum"] / count,
        "value_mae": baseline["absolute_error"] / count,
        "value_mse": baseline["squared_error"] / count,
        "value_sign_accuracy": baseline["sign_correct"] / count,
        "decisive_value_roc_auc": _roc_auc(baseline["wins"], baseline["losses"]),
        "decisive_wins": len(baseline["wins"]),
        "decisive_losses": len(baseline["losses"]),
        "calibration": _finalize_calibration(baseline["calibration"]),
    }
    for name in ("parent", "high_dose"):
        payload = accumulator[name]
        result[name] = {
            "value_mean": payload["prediction_sum"] / count,
            "value_bias": payload["error_sum"] / count,
            "value_mae": payload["absolute_error"] / count,
            "value_mse": payload["squared_error"] / count,
            "value_sign_accuracy": payload["sign_correct"] / count,
            "decisive_value_roc_auc": _roc_auc(payload["wins"], payload["losses"]),
            "decisive_wins": len(payload["wins"]),
            "decisive_losses": len(payload["losses"]),
            "policy_cross_entropy": payload["policy_cross_entropy"] / count,
            "policy_target_top_agreement": payload["policy_top_correct"] / count,
            "calibration": _finalize_calibration(payload["calibration"]),
        }
    result["delta_high_dose_minus_parent"] = {
        key: result["high_dose"][key] - result["parent"][key]
        for key in (
            "value_bias",
            "value_mae",
            "value_mse",
            "value_sign_accuracy",
            "decisive_value_roc_auc",
            "policy_cross_entropy",
            "policy_target_top_agreement",
        )
        if result["high_dose"][key] is not None and result["parent"][key] is not None
    }
    return result


def _finalize_calibration(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bin": index,
            "range": [(-1.0 + 0.2 * index), (-0.8 + 0.2 * index)],
            "count": item["count"],
            "mean_prediction": (
                item["prediction_sum"] / item["count"] if item["count"] else None
            ),
            "mean_target": item["target_sum"] / item["count"] if item["count"] else None,
        }
        for index, item in enumerate(items)
    ]


def _fit_score_tempo_baseline(path: Path) -> dict[str, Any]:
    feature_count = 5
    xtx = torch.zeros((feature_count, feature_count), dtype=torch.float64)
    xty = torch.zeros(feature_count, dtype=torch.float64)
    rows = 0
    games: set[int] = set()
    for chunk in _replay_chunks(path, 512):
        features = torch.tensor(
            [_score_tempo_features(row) for row in chunk], dtype=torch.float64
        )
        targets = torch.tensor([float(row.value_target) for row in chunk], dtype=torch.float64)
        xtx += features.T @ features
        xty += features.T @ targets
        rows += len(chunk)
        games.update(row.game_seed for row in chunk)
    if rows != 55_806 or len(games) != 896:
        raise ValueError("111 score/tempo fit replay coverage changed.")
    ridge_lambda = 0.001
    penalty = torch.eye(feature_count, dtype=torch.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(xtx + penalty, xty)
    return {
        "fit_rows": rows,
        "fit_games": len(games),
        "ridge_lambda": ridge_lambda,
        "features": [
            "intercept",
            "self_minus_opponent_score_div_15",
            "self_minus_opponent_purchased_cards_div_15",
            "self_minus_opponent_total_bonuses_div_15",
            "turn_index_div_100",
        ],
        "coefficients": [float(value) for value in coefficients.tolist()],
        "prediction_clip": [-1.0, 1.0],
    }


def _score_tempo_features(row: AlphaZeroReplayRow) -> list[float]:
    player_rows = {
        entity_type: features
        for entity_type, features in zip(
            row.entity_type_ids, row.entity_numeric_features, strict=True
        )
        if entity_type in (ENTITY_SELF, ENTITY_OPPONENT)
    }
    if set(player_rows) != {ENTITY_SELF, ENTITY_OPPONENT}:
        raise ValueError("Replay row lacks exact self/opponent entities.")
    own = player_rows[ENTITY_SELF]
    opponent = player_rows[ENTITY_OPPONENT]
    return [
        1.0,
        ((own[31] - opponent[31]) * 20.0) / 15.0,
        ((own[33] - opponent[33]) * 20.0) / 15.0,
        (sum(own[26:31]) - sum(opponent[26:31])) * 7.0 / 15.0,
        float(row.turn_index) / 100.0,
    ]


def _score_tempo_prediction(row: AlphaZeroReplayRow, coefficients: list[float]) -> float:
    raw = sum(
        coefficient * feature
        for coefficient, feature in zip(
            coefficients, _score_tempo_features(row), strict=True
        )
    )
    return max(-1.0, min(1.0, raw))


def _gradient_batch_metrics(
    model: torch.nn.Module,
    shared: list[torch.nn.Parameter],
    rows: list[AlphaZeroReplayRow],
    device: str,
) -> dict[str, float]:
    batch = collate_replay_rows(rows)
    moved = {key: value.to(device) for key, value in batch.items()}
    logits, values = model(
        entity_type_ids=moved["entity_type_ids"],
        entity_card_ids=moved["entity_card_ids"],
        entity_noble_ids=moved["entity_noble_ids"],
        entity_numeric_features=moved["entity_numeric_features"],
        entity_mask=moved["entity_mask"],
        action_type_ids=moved["action_type_ids"],
        action_card_ids=moved["action_card_ids"],
        action_noble_ids=moved["action_noble_ids"],
        action_numeric_features=moved["action_numeric_features"],
        action_mask=moved["action_mask"],
    )
    floor = torch.finfo(logits.dtype).min
    log_policy = torch.log_softmax(logits.masked_fill(~moved["action_mask"], floor), dim=1)
    policy_loss = -(moved["policy_target"] * log_policy).sum(dim=1).mean()
    value_loss = torch.mean((values - moved["value_target"]) ** 2)
    policy_gradients = torch.autograd.grad(policy_loss, shared, retain_graph=True)
    value_gradients = torch.autograd.grad(value_loss, shared)
    dot = sum(float(torch.sum(left * right).item()) for left, right in zip(policy_gradients, value_gradients, strict=True))
    policy_sq = sum(float(torch.sum(value * value).item()) for value in policy_gradients)
    value_sq = sum(float(torch.sum(value * value).item()) for value in value_gradients)
    policy_norm = math.sqrt(policy_sq)
    value_norm = math.sqrt(value_sq)
    cosine = dot / (policy_norm * value_norm) if policy_norm and value_norm else 0.0
    del batch, moved, logits, values, log_policy, policy_gradients, value_gradients
    return {
        "rows": float(len(rows)),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "dot_product": dot,
        "policy_gradient_norm": policy_norm,
        "value_gradient_norm": value_norm,
        "cosine_similarity": cosine,
        "raw_value_to_policy_norm_ratio": value_norm / policy_norm if policy_norm else math.inf,
        "weighted_value_to_policy_norm_ratio": 0.5 * value_norm / policy_norm if policy_norm else math.inf,
    }


def _summarize_gradient_batches(batches: list[dict[str, float]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("Gradient conflict audit produced no batches.")
    policy_sq = sum(item["policy_gradient_norm"] ** 2 for item in batches)
    value_sq = sum(item["value_gradient_norm"] ** 2 for item in batches)
    dot = sum(item["dot_product"] for item in batches)
    aggregate_cosine = dot / math.sqrt(policy_sq * value_sq) if policy_sq and value_sq else 0.0
    return {
        "batch_records": batches,
        "aggregate_cosine_similarity": aggregate_cosine,
        "mean_batch_cosine_similarity": sum(item["cosine_similarity"] for item in batches) / len(batches),
        "median_batch_cosine_similarity": _median([item["cosine_similarity"] for item in batches]),
        "negative_cosine_batches": sum(item["cosine_similarity"] < 0.0 for item in batches),
        "negative_cosine_fraction": sum(item["cosine_similarity"] < 0.0 for item in batches) / len(batches),
        "mean_policy_gradient_norm": sum(item["policy_gradient_norm"] for item in batches) / len(batches),
        "mean_value_gradient_norm": sum(item["value_gradient_norm"] for item in batches) / len(batches),
        "mean_raw_value_to_policy_norm_ratio": sum(item["raw_value_to_policy_norm_ratio"] for item in batches) / len(batches),
        "mean_weighted_value_to_policy_norm_ratio": sum(item["weighted_value_to_policy_norm_ratio"] for item in batches) / len(batches),
    }


def _shared_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    selected = [
        parameter
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in _SHARED_PREFIXES)
    ]
    if not selected:
        raise ValueError("111 could not identify any shared policy/value parameters.")
    return selected


def _replay_chunks(path: Path, batch_size: int) -> Iterable[list[AlphaZeroReplayRow]]:
    chunk: list[AlphaZeroReplayRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = _row_from_payload(json.loads(line))
            if row.value_target is None or row.termination_reason != "completed":
                raise ValueError("111 requires complete normal replay rows.")
            chunk.append(row)
            if len(chunk) == batch_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _roc_auc(wins: list[float], losses: list[float]) -> float | None:
    if not wins or not losses:
        return None
    values = [(value, 1) for value in wins] + [(value, 0) for value in losses]
    values.sort(key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _value, label in values[index:end])
        index = end
    positives = len(wins)
    negatives = len(losses)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _phase(turn: int) -> str:
    return "early_lt20" if turn < 20 else ("middle_20_49" if turn < 50 else "late_ge50")


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _evaluation_search() -> SearchConfig:
    return SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=GRADIENT_SAMPLE_SEED,
    )


def _stage_payload(stage: str, audit: Any) -> dict[str, Any]:
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "stage": stage,
        "input_hashes": dict(EXPECTED_HASHES),
        "audit": audit,
    }


def _validate_stage_identity(payload: dict[str, Any], stage: str) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("stage") != stage
        or payload.get("input_hashes") != EXPECTED_HASHES
    ):
        raise ValueError(f"111 {stage} stage identity mismatch.")


def _verify_inputs(config: FreshHeadAttributionConfig) -> None:
    paths = {
        "result_110": config.result_110,
        "installed_checkpoint": config.installed_checkpoint,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "high_dose_checkpoint": config.high_dose_checkpoint,
        "high_dose_replay": config.high_dose_replay,
        "common_holdout": config.common_holdout,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 111 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_110.read_text(encoding="utf-8"))
    if (
        prior.get("status") != "complete"
        or prior.get("decision") != "high_dose_fresh_selfplay_failed_internal_gate_review_capacity_no_install"
        or prior.get("qualified_arms") != ["low", "high"]
        or prior.get("training", {}).get("high", {}).get("checkpoint_sha256")
        != EXPECTED_HASHES["high_dose_checkpoint"]
        or prior.get("partition", {}).get("high_sha256") != EXPECTED_HASHES["high_dose_replay"]
        or prior.get("partition", {}).get("holdout_sha256") != EXPECTED_HASHES["common_holdout"]
        or prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]
    ):
        raise ValueError("Frozen 110 result lacks its exact completed contract.")
    _, parent_identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    _, high_identity = load_checkpoint(config.high_dose_checkpoint, device=config.device)
    if parent_identity.generation != 11 or parent_identity.metadata.get("parent_generation") != 10:
        raise ValueError("Frozen 111 parent is not exact generation 11.")
    if (
        high_identity.generation != 12
        or high_identity.metadata.get("parent_generation") != 11
        or high_identity.metadata.get("selected_epoch") != 1
        or high_identity.metadata.get("parent_safe_selection") is not True
    ):
        raise ValueError("Frozen 111 high-dose checkpoint has invalid lineage.")
    if not 1 <= config.arena_workers <= 3:
        raise ValueError("111 arena workers exceed the frozen maximum of three.")


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config: FreshHeadAttributionConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})
