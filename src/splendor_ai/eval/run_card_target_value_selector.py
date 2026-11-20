"""Run the preregistered card-target value selector experiment 082."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.bots import (
    CardOptionTelemetry,
    CardTargetValueBot,
    LegalActionAttentionScorerBot,
)
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_card_option_q_controller import (
    RAW035_SHA256,
    _deterministic_payload,
    _game_payload,
    _integrity,
    _merge_telemetry,
    _rates,
    _sha,
    _strength,
)
from splendor_ai.training.run_card_target_corpus import (
    CardTargetCorpusConfig,
    build_card_target_corpus,
)
from splendor_ai.training.run_card_target_training import (
    CardTargetTrainingConfig,
    extract_historical_target_rows,
    train_card_target_value,
)


PROTOCOL = "docs/experiments/card_target_value_selector_082.md"
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class CardTargetExperimentConfig:
    checkpoint_035: Path
    historical_corpus_path: Path
    output_dir: Path
    workers: int = 10
    mechanics_pairs: int = 40
    mechanics_seed_start: int = 787_000
    strength_pairs: int = 200
    strength_seed_start: int = 788_000
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 788_082
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    device: str = "cpu"


def run_card_target_value_selector(
    config: CardTargetExperimentConfig,
) -> dict[str, Any]:
    began = time.perf_counter()
    _validate_experiment_config(config)
    raw_hash = _sha(config.checkpoint_035)
    base = _base_result(config, raw_hash)
    extraction_config = CardTargetTrainingConfig(
        historical_corpus_path=config.historical_corpus_path,
        fresh_corpus_path=config.output_dir / "fresh_corpus.json",
        checkpoint_035=config.checkpoint_035,
        output_dir=config.output_dir / "training",
        device=config.device,
    )
    historical_rows, historical_summary = extract_historical_target_rows(
        extraction_config
    )
    del historical_rows
    base["stage_a_historical"] = {
        "path": str(config.historical_corpus_path),
        "sha256": historical_summary["source_sha256"],
        "summary": historical_summary,
        "all_gates_passed": (
            historical_summary["states"] == 1_000
            and historical_summary["exact_unique_matches"]
            and raw_hash == RAW035_SHA256
        ),
    }
    if not base["stage_a_historical"]["all_gates_passed"]:
        base["decision"] = "card_target_value_082_invalid_no_strength_decision"
        return _finish(base, config.output_dir, began)

    corpus_path = config.output_dir / "fresh_corpus.json"
    progress_path = config.output_dir / "fresh_corpus.progress.jsonl"
    corpus = build_card_target_corpus(
        CardTargetCorpusConfig(
            checkpoint=config.checkpoint_035,
            output_path=corpus_path,
            progress_path=progress_path,
            workers=config.workers,
            device=config.device,
        )
    )
    base["stage_b_fresh_corpus"] = {
        "path": str(corpus_path),
        "sha256": _sha(corpus_path),
        "source": corpus["source"],
        "summary": corpus["summary"],
        "gates": corpus["gates"],
        "all_gates_passed": corpus["all_gates_passed"],
    }
    if not corpus["all_gates_passed"]:
        signal_key = "split_half_target_rank_signal_positive"
        signal_only = (
            not corpus["gates"][signal_key]
            and all(value for key, value in corpus["gates"].items() if key != signal_key)
        )
        base["decision"] = (
            "card_target_value_082_counterfactual_signal_failed_close_recipe"
            if signal_only
            else "card_target_value_082_invalid_no_strength_decision"
        )
        return _finish(base, config.output_dir, began)

    training_dir = config.output_dir / "training"
    training = train_card_target_value(
        CardTargetTrainingConfig(
            historical_corpus_path=config.historical_corpus_path,
            fresh_corpus_path=corpus_path,
            checkpoint_035=config.checkpoint_035,
            output_dir=training_dir,
            device=config.device,
        )
    )
    training_path = training_dir / "training_result.json"
    model_path = Path(training["checkpoint_path"])
    base["stage_c_training"] = {
        "path": str(training_path),
        "sha256": _sha(training_path),
        "historical": training["historical"],
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": _sha(model_path),
        "best_epoch": training["best_epoch"],
        "validation": training["validation"],
        "confirmation": training["confirmation"],
        "gates": training["gates"],
        "all_gates_passed": training["all_gates_passed"],
    }
    if not training["all_gates_passed"]:
        base["decision"] = "card_target_value_082_offline_failed_close_recipe"
        return _finish(base, config.output_dir, began)

    mechanics_rows, mechanics_elapsed = _panel(
        config,
        model_path,
        config.mechanics_seed_start,
        config.mechanics_pairs,
        "mechanics",
        False,
    )
    _initialize_eval(
        str(config.checkpoint_035),
        str(model_path),
        (config.max_turns, config.repetition_limit, config.no_progress_limit),
        False,
    )
    deterministic = (
        _deterministic_payload(_pair(config.mechanics_seed_start))
        == _deterministic_payload(mechanics_rows[0])
    )
    mechanics_telemetry = _merge_telemetry(mechanics_rows)
    mechanics_rates = _rates(mechanics_telemetry, config.mechanics_pairs * 2)
    mechanics_integrity = _integrity(
        mechanics_rows,
        config.mechanics_pairs,
        config.mechanics_seed_start,
        outcomes=False,
    )
    mechanics_integrity.update(
        {
            "deterministic": deterministic,
            "raw035_hash": raw_hash == RAW035_SHA256,
            "model_hash": (
                _sha(model_path)
                == base["stage_c_training"]["checkpoint_sha256"]
            ),
        }
    )
    projected = mechanics_elapsed * config.strength_pairs / config.mechanics_pairs
    operational = {
        "inference_p95_at_most_150ms": (
            mechanics_rates["inference_latency_p95_ms"] <= 150.0
        ),
        "mechanics_under_2_hours": mechanics_elapsed <= 2 * 60 * 60,
        "projected_strength_under_10_hours": projected <= 10 * 60 * 60,
    }
    mechanics = {
        "movement_at_least_015": mechanics_rates["movement"] >= 0.15,
        "target_active_at_least_020": mechanics_rates["target_decision_rate"] >= 0.20,
        "target_completion_at_least_050": (
            mechanics_rates["target_completion_rate"] >= 0.50
        ),
        "purchases_at_least_5": mechanics_rates["mean_purchases"] >= 5.0,
    }
    base["stage_d_mechanics"] = {
        "integrity": mechanics_integrity,
        "operational": operational,
        "mechanics": mechanics,
        "rates": mechanics_rates,
        "telemetry": mechanics_telemetry.as_dict(),
        "elapsed_seconds": mechanics_elapsed,
        "projected_strength_seconds": projected,
        "outcomes_used_or_exposed": False,
        "clusters": mechanics_rows,
    }
    if not all(mechanics_integrity.values()) or not all(operational.values()):
        base["decision"] = "card_target_value_082_invalid_no_strength_decision"
        return _finish(base, config.output_dir, began)
    if not all(mechanics.values()):
        base["decision"] = "card_target_value_082_mechanics_failed_close_candidate"
        return _finish(base, config.output_dir, began)

    strength_rows, strength_elapsed = _panel(
        config,
        model_path,
        config.strength_seed_start,
        config.strength_pairs,
        "strength",
        True,
    )
    strength_telemetry = _merge_telemetry(strength_rows)
    strength_rates = _rates(strength_telemetry, config.strength_pairs * 2)
    strength_integrity = _integrity(
        strength_rows,
        config.strength_pairs,
        config.strength_seed_start,
        outcomes=True,
    )
    strength_integrity.update(
        {
            "raw035_hash": _sha(config.checkpoint_035) == RAW035_SHA256,
            "model_hash": (
                _sha(model_path)
                == base["stage_c_training"]["checkpoint_sha256"]
            ),
            "movement_at_least_015": strength_rates["movement"] >= 0.15,
        }
    )
    strength = _strength(strength_rows, config)
    if not all(strength_integrity.values()):
        decision = "card_target_value_082_invalid_no_strength_decision"
    elif (
        strength["score_share"]["mean"] >= 0.55
        and strength["score_share"]["ci95_low"] > 0.50
    ):
        decision = "card_target_value_082_stronger_promote_fast_and_research_champion"
    elif strength["score_share"]["ci95_high"] < 0.50:
        decision = "card_target_value_082_weaker_close_candidate"
    else:
        decision = "card_target_value_082_not_decisively_stronger_close_candidate"
    base["stage_e_strength"] = {
        "integrity": strength_integrity,
        "rates": strength_rates,
        "telemetry": strength_telemetry.as_dict(),
        "strength": strength,
        "elapsed_seconds": strength_elapsed,
        "clusters": strength_rows,
    }
    base["decision"] = decision
    return _finish(base, config.output_dir, began)


def _initialize_eval(
    checkpoint_035: str,
    model_path: str,
    limits: tuple[int, int, int],
    outcomes: bool,
) -> None:
    torch.set_num_threads(1)
    _WORKER["raw"] = LegalActionAttentionScorerBot(checkpoint_035, device="cpu")
    _WORKER["candidate"] = CardTargetValueBot(
        model_path, checkpoint_035, device="cpu"
    )
    _WORKER["limits"] = limits
    _WORKER["outcomes"] = outcomes


def _pair(seed: int) -> dict[str, Any]:
    raw = _WORKER["raw"]
    candidate: CardTargetValueBot = _WORKER["candidate"]
    max_turns, repetition, no_progress = _WORKER["limits"]
    outcomes = bool(_WORKER["outcomes"])
    first = play_game(candidate, raw, seed, max_turns, repetition, no_progress)
    telemetry = CardOptionTelemetry()
    telemetry.merge(candidate.telemetry.as_dict())
    second = play_game(raw, candidate, seed, max_turns, repetition, no_progress)
    telemetry.merge(candidate.telemetry.as_dict())
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [_game_payload(first, outcomes), _game_payload(second, outcomes)],
        "telemetry": telemetry.as_dict(),
    }


def _panel(
    config: CardTargetExperimentConfig,
    model_path: Path,
    start: int,
    pairs: int,
    label: str,
    outcomes: bool,
) -> tuple[list[dict[str, Any]], float]:
    began = time.perf_counter()
    rows: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_initialize_eval,
        initargs=(
            str(config.checkpoint_035),
            str(model_path),
            (config.max_turns, config.repetition_limit, config.no_progress_limit),
            outcomes,
        ),
    ) as executor:
        futures = {
            executor.submit(_pair, seed): seed for seed in range(start, start + pairs)
        }
        for done, future in enumerate(as_completed(futures), 1):
            rows[futures[future]] = future.result()
            if done % 10 == 0 or done == pairs:
                elapsed = time.perf_counter() - began
                rate = done / elapsed
                print(
                    f"[082:{label}] clusters={done}/{pairs} rate={rate:.3f}/s "
                    f"eta={(pairs - done) / rate:.0f}s",
                    flush=True,
                )
    return (
        [rows[seed] for seed in range(start, start + pairs)],
        time.perf_counter() - began,
    )


def _base_result(
    config: CardTargetExperimentConfig,
    raw_hash: str,
) -> dict[str, Any]:
    return {
        "experiment": "082",
        "protocol": PROTOCOL,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "raw035_sha256": raw_hash,
        "stage_a_historical": None,
        "stage_b_fresh_corpus": None,
        "stage_c_training": None,
        "stage_d_mechanics": None,
        "stage_e_strength": None,
        "decision": None,
        "human_data_used": False,
        "passed": False,
        "fast_champion_before": "035",
        "research_champion_before": "035",
    }


def _finish(
    payload: dict[str, Any],
    output_dir: Path,
    began: float,
) -> dict[str, Any]:
    payload["elapsed_seconds"] = time.perf_counter() - began
    payload["passed"] = (
        payload["decision"]
        == "card_target_value_082_stronger_promote_fast_and_research_champion"
    )
    payload["fast_champion_after"] = "082" if payload["passed"] else "035"
    payload["research_champion_after"] = "082" if payload["passed"] else "035"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[082] decision={payload['decision']}", flush=True)
    return payload


def _validate_experiment_config(config: CardTargetExperimentConfig) -> None:
    if (config.mechanics_pairs, config.mechanics_seed_start) != (40, 787_000):
        raise ValueError("082 mechanics panel contract is frozen.")
    if (
        config.strength_pairs,
        config.strength_seed_start,
        config.bootstrap_seed,
    ) != (200, 788_000, 788_082):
        raise ValueError("082 strength panel contract is frozen.")
    if config.workers <= 0:
        raise ValueError("workers must be positive.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-035",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--historical-corpus",
        default="outputs/card_option_q_controller_081/corpus.json",
    )
    parser.add_argument(
        "--output-dir", default="outputs/card_target_value_selector_082"
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run_card_target_value_selector(
        CardTargetExperimentConfig(
            checkpoint_035=Path(args.checkpoint_035),
            historical_corpus_path=Path(args.historical_corpus),
            output_dir=Path(args.output_dir),
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
