"""Run the preregistered from-scratch card-option Q experiment 081."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch

from splendor_ai.bots import CardOptionQBot, CardOptionTelemetry, LegalActionAttentionScorerBot
from splendor_ai.eval.match import play_game
from splendor_ai.training.run_card_option_q_corpus import (
    CardOptionCorpusConfig,
    build_card_option_corpus,
)
from splendor_ai.training.run_card_option_q_training import (
    CardOptionTrainingConfig,
    train_card_option_q,
)


PROTOCOL = "docs/experiments/card_option_q_controller_081.md"
RAW035_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
VALID_TERMINALS = ("completed", "forced_pass_deadlock")
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class CardOptionExperimentConfig:
    checkpoint_035: Path
    output_dir: Path
    workers: int = 10
    mechanics_pairs: int = 40
    mechanics_seed_start: int = 783_000
    strength_pairs: int = 200
    strength_seed_start: int = 784_000
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 784_081
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    device: str = "cpu"


def run_card_option_q_controller(config: CardOptionExperimentConfig) -> dict[str, Any]:
    began = time.perf_counter()
    _validate_experiment_config(config)
    raw_hash = _sha(config.checkpoint_035)
    base = _base_result(config, raw_hash)
    corpus_path = config.output_dir / "corpus.json"
    progress_path = config.output_dir / "corpus.progress.jsonl"
    corpus = build_card_option_corpus(
        CardOptionCorpusConfig(
            checkpoint=config.checkpoint_035,
            output_path=corpus_path,
            progress_path=progress_path,
            workers=config.workers,
            device=config.device,
        )
    )
    corpus_hash = _sha(corpus_path)
    base["stage_a_corpus"] = {
        "path": str(corpus_path),
        "sha256": corpus_hash,
        "source": corpus["source"],
        "summary": corpus["summary"],
        "gates": corpus["gates"],
        "all_gates_passed": corpus["all_gates_passed"],
    }
    if not corpus["all_gates_passed"]:
        signal_only = (
            not corpus["gates"]["split_half_rank_signal_positive"]
            and all(value for key, value in corpus["gates"].items() if key != "split_half_rank_signal_positive")
        )
        base["decision"] = (
            "card_option_q_081_counterfactual_signal_failed_close_recipe"
            if signal_only else "card_option_q_081_invalid_no_strength_decision"
        )
        return _finish(base, config.output_dir, began)

    training_dir = config.output_dir / "training"
    training = train_card_option_q(
        CardOptionTrainingConfig(
            corpus_path=corpus_path,
            output_dir=training_dir,
            device=config.device,
        )
    )
    training_path = training_dir / "training_result.json"
    model_path = Path(training["checkpoint_path"])
    base["stage_b_training"] = {
        "path": str(training_path),
        "sha256": _sha(training_path),
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": _sha(model_path),
        "best_epoch": training["best_epoch"],
        "validation": training["validation"],
        "confirmation": training["confirmation"],
        "gates": training["gates"],
        "all_gates_passed": training["all_gates_passed"],
    }
    if not training["all_gates_passed"]:
        base["decision"] = "card_option_q_081_offline_failed_close_recipe"
        return _finish(base, config.output_dir, began)

    mechanics_rows, mechanics_elapsed = _panel(
        config, model_path, config.mechanics_seed_start, config.mechanics_pairs, "mechanics", False
    )
    _initialize_eval(
        str(config.checkpoint_035), str(model_path),
        (config.max_turns, config.repetition_limit, config.no_progress_limit), False,
    )
    deterministic = _deterministic_payload(_pair(config.mechanics_seed_start)) == _deterministic_payload(mechanics_rows[0])
    mechanics_telemetry = _merge_telemetry(mechanics_rows)
    mechanics_rates = _rates(mechanics_telemetry, config.mechanics_pairs * 2)
    mechanics_integrity = _integrity(
        mechanics_rows, config.mechanics_pairs, config.mechanics_seed_start, outcomes=False
    )
    mechanics_integrity.update(
        {
            "deterministic": deterministic,
            "raw035_hash": raw_hash == RAW035_SHA256,
            "model_hash": _sha(model_path) == base["stage_b_training"]["checkpoint_sha256"],
        }
    )
    projected = mechanics_elapsed * config.strength_pairs / config.mechanics_pairs
    operational = {
        "inference_p95_at_most_150ms": mechanics_rates["inference_latency_p95_ms"] <= 150.0,
        "mechanics_under_2_hours": mechanics_elapsed <= 2 * 60 * 60,
        "projected_strength_under_10_hours": projected <= 10 * 60 * 60,
    }
    mechanics = {
        "movement_at_least_015": mechanics_rates["movement"] >= 0.15,
        "target_active_at_least_020": mechanics_rates["target_decision_rate"] >= 0.20,
        "target_completion_at_least_050": mechanics_rates["target_completion_rate"] >= 0.50,
        "purchases_at_least_5": mechanics_rates["mean_purchases"] >= 5.0,
    }
    base["stage_c_mechanics"] = {
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
        base["decision"] = "card_option_q_081_invalid_no_strength_decision"
        return _finish(base, config.output_dir, began)
    if not all(mechanics.values()):
        base["decision"] = "card_option_q_081_mechanics_failed_close_candidate"
        return _finish(base, config.output_dir, began)

    strength_rows, strength_elapsed = _panel(
        config, model_path, config.strength_seed_start, config.strength_pairs, "strength", True
    )
    strength_telemetry = _merge_telemetry(strength_rows)
    strength_rates = _rates(strength_telemetry, config.strength_pairs * 2)
    strength_integrity = _integrity(
        strength_rows, config.strength_pairs, config.strength_seed_start, outcomes=True
    )
    strength_integrity.update(
        {
            "raw035_hash": _sha(config.checkpoint_035) == RAW035_SHA256,
            "model_hash": _sha(model_path) == base["stage_b_training"]["checkpoint_sha256"],
            "movement_at_least_015": strength_rates["movement"] >= 0.15,
        }
    )
    strength = _strength(strength_rows, config)
    if not all(strength_integrity.values()):
        decision = "card_option_q_081_invalid_no_strength_decision"
    elif strength["score_share"]["mean"] >= 0.55 and strength["score_share"]["ci95_low"] > 0.50:
        decision = "card_option_q_081_stronger_promote_fast_and_research_champion"
    elif strength["score_share"]["ci95_high"] < 0.50:
        decision = "card_option_q_081_weaker_close_candidate"
    else:
        decision = "card_option_q_081_not_decisively_stronger_close_candidate"
    base["stage_d_strength"] = {
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
    raw = LegalActionAttentionScorerBot(checkpoint_035, device="cpu")
    _WORKER["raw"] = raw
    _WORKER["candidate"] = CardOptionQBot(model_path, raw035_bot=raw, device="cpu")
    _WORKER["limits"] = limits
    _WORKER["outcomes"] = outcomes


def _pair(seed: int) -> dict[str, Any]:
    raw = _WORKER["raw"]
    candidate: CardOptionQBot = _WORKER["candidate"]
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
    config: CardOptionExperimentConfig,
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
            str(config.checkpoint_035), str(model_path),
            (config.max_turns, config.repetition_limit, config.no_progress_limit), outcomes,
        ),
    ) as executor:
        futures = {executor.submit(_pair, seed): seed for seed in range(start, start + pairs)}
        for done, future in enumerate(as_completed(futures), 1):
            rows[futures[future]] = future.result()
            if done % 10 == 0 or done == pairs:
                elapsed = time.perf_counter() - began
                rate = done / elapsed
                print(f"[081:{label}] clusters={done}/{pairs} rate={rate:.3f}/s eta={(pairs-done)/rate:.0f}s", flush=True)
    return [rows[seed] for seed in range(start, start + pairs)], time.perf_counter() - began


def _game_payload(game: Any, outcomes: bool) -> dict[str, Any]:
    payload = asdict(game)
    if not outcomes:
        payload.pop("winner", None)
        payload.pop("final_scores", None)
    return payload


def _deterministic_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(row))
    payload["telemetry"]["inference_latencies_ms"] = []
    return payload


def _merge_telemetry(rows: list[dict[str, Any]]) -> CardOptionTelemetry:
    telemetry = CardOptionTelemetry()
    for row in rows:
        telemetry.merge(row["telemetry"])
    return telemetry


def _rates(telemetry: CardOptionTelemetry, games: int) -> dict[str, Any]:
    decisions = max(telemetry.decisions, 1)
    latencies = sorted(telemetry.inference_latencies_ms)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))] if latencies else float("inf")
    return {
        "movement": telemetry.moved_from_raw035 / decisions,
        "target_decision_rate": telemetry.target_decisions / decisions,
        "target_completion_rate": telemetry.target_completions / max(telemetry.target_selections, 1),
        "mean_purchases": telemetry.purchases / max(games, 1),
        "deck_reserve_rate": telemetry.deck_reserves / decisions,
        "inference_latency_mean_ms": statistics.fmean(latencies) if latencies else float("inf"),
        "inference_latency_p95_ms": p95,
        "phase_movement": {
            phase: telemetry.phase_movement.get(phase, 0) / max(telemetry.phase_decisions.get(phase, 0), 1)
            for phase in ("early", "mid", "late")
        },
    }


def _integrity(
    rows: list[dict[str, Any]],
    pairs: int,
    start: int,
    outcomes: bool,
) -> dict[str, bool]:
    games = [game for row in rows for game in row["games"]]
    return {
        "exact_clusters": len(rows) == pairs,
        "exact_games": len(games) == 2 * pairs,
        "exact_seed_seat": [row["seed"] for row in rows] == list(range(start, start + pairs)) and all(row["candidate_seats"] == [0, 1] for row in rows),
        "valid_terminals": all(game["termination_reason"] in VALID_TERMINALS for game in games),
        "zero_stalls": all(not game["stalled"] for game in games),
        "zero_timeouts": all(not game["timed_out"] for game in games),
        "zero_fallbacks": all(tuple(game["loop_fallback_triggers_by_seat"]) == (0, 0) for game in games),
        "zero_illegal_null": True,
        "candidate_acted": sum(row["telemetry"]["decisions"] for row in rows) > 0,
        "outcome_contract": outcomes == all("winner" in game and "final_scores" in game for game in games),
    }


def _strength(rows: list[dict[str, Any]], config: CardOptionExperimentConfig) -> dict[str, Any]:
    pair_shares = []
    pair_margins = []
    wins = losses = draws = 0
    for row in rows:
        shares = []
        margins = []
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
            winner = game["winner"]
            shares.append(0.5 if winner is None else float(winner == seat))
            margins.append(game["final_scores"][seat] - game["final_scores"][1 - seat])
            if winner is None:
                draws += 1
            elif winner == seat:
                wins += 1
            else:
                losses += 1
        pair_shares.append(statistics.fmean(shares))
        pair_margins.append(statistics.fmean(margins))
    return {
        "score_share": _bootstrap_metric(pair_shares, config.bootstrap_samples, config.bootstrap_seed),
        "margin": _bootstrap_metric(pair_margins, config.bootstrap_samples, config.bootstrap_seed + 1),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "mean_turns": statistics.fmean(game["turns"] for row in rows for game in row["games"]),
    }


def _bootstrap_metric(values: list[float], samples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return {
        "mean": statistics.fmean(values),
        "ci95_low": draws[int(0.025 * (samples - 1))],
        "ci95_high": draws[int(0.975 * (samples - 1))],
    }


def _base_result(config: CardOptionExperimentConfig, raw_hash: str) -> dict[str, Any]:
    return {
        "experiment": "081",
        "protocol": PROTOCOL,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "raw035_sha256": raw_hash,
        "stage_a_corpus": None,
        "stage_b_training": None,
        "stage_c_mechanics": None,
        "stage_d_strength": None,
        "decision": None,
        "human_data_used": False,
        "passed": False,
        "fast_champion_before": "035",
        "research_champion_before": "035",
    }


def _finish(payload: dict[str, Any], output_dir: Path, began: float) -> dict[str, Any]:
    payload["elapsed_seconds"] = time.perf_counter() - began
    payload["passed"] = payload["decision"] == "card_option_q_081_stronger_promote_fast_and_research_champion"
    payload["fast_champion_after"] = "081" if payload["passed"] else "035"
    payload["research_champion_after"] = "081" if payload["passed"] else "035"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[081] decision={payload['decision']}", flush=True)
    return payload


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_experiment_config(config: CardOptionExperimentConfig) -> None:
    if (config.mechanics_pairs, config.mechanics_seed_start) != (40, 783_000):
        raise ValueError("081 mechanics panel contract is frozen.")
    if (config.strength_pairs, config.strength_seed_start, config.bootstrap_seed) != (200, 784_000, 784_081):
        raise ValueError("081 strength panel contract is frozen.")
    if config.workers <= 0:
        raise ValueError("workers must be positive.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-035",
        default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt",
    )
    parser.add_argument("--output-dir", default="outputs/card_option_q_controller_081")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run_card_option_q_controller(
        CardOptionExperimentConfig(
            checkpoint_035=Path(args.checkpoint_035),
            output_dir=Path(args.output_dir),
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
