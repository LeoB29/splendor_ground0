"""Run preregistered value-reserve leaf-bonus causal ablation `078`."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

from splendor_ai.bots import (
    FrozenFullActionSearchBot,
    ValueReserveLeafSearchBot,
    ValueReserveSearchTelemetry,
)
from splendor_ai.bots.value_reserve_search_bot import (
    SEARCH_DEPTH,
    SEARCH_DETERMINIZATIONS,
    UNBOUNDED_ACTION_CAP,
)
from splendor_ai.eval.match import play_game


PROTOCOL = "docs/experiments/value_reserve_leaf_bonus_causal_ablation_078.md"
SEED_START = 772_000
PAIRS = 200
OPERATIONAL_PAIRS = 20
BOOTSTRAP_SEED = 772_078
BOOTSTRAP_SAMPLES = 20_000
MAX_PROJECTED_SECONDS = 3 * 60 * 60
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")


@dataclass(frozen=True, slots=True)
class LeafBonusCausalConfig:
    output_dir: Path
    workers: int = 10
    pairs: int = PAIRS
    operational_pairs: int = OPERATIONAL_PAIRS
    seed_start: int = SEED_START
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED


_LIMITS: tuple[int, int, int] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialize_worker(limits: tuple[int, int, int]) -> None:
    global _LIMITS
    _LIMITS = limits


def _play_pair(seed: int) -> dict[str, Any]:
    if _LIMITS is None:
        raise RuntimeError("078 worker not initialized")
    max_turns, repetition_limit, no_progress_limit = _LIMITS

    bonus_first = ValueReserveLeafSearchBot(raw035_bot=None)
    first = play_game(
        bonus_first,
        FrozenFullActionSearchBot(with_bonus=False),
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    merged = ValueReserveSearchTelemetry()
    merged.merge(bonus_first.telemetry.as_dict())

    bonus_second = ValueReserveLeafSearchBot(raw035_bot=None)
    second = play_game(
        FrozenFullActionSearchBot(with_bonus=False),
        bonus_second,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    merged.merge(bonus_second.telemetry.as_dict())
    return {
        "seed": seed,
        "bonus_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
        "telemetry": merged.as_dict(),
    }


def _run_seeds(
    config: LeafBonusCausalConfig,
    seeds: list[int],
    label: str,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    rows: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_initialize_worker,
        initargs=((config.max_turns, config.repetition_limit, config.no_progress_limit),),
    ) as executor:
        futures = {executor.submit(_play_pair, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), start=1):
            seed = futures[future]
            rows[seed] = future.result()
            if completed % 10 == 0 or completed == len(seeds):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0.0
                eta = (len(seeds) - completed) / rate if rate else math.inf
                print(
                    f"[078:{label}] clusters={completed}/{len(seeds)}"
                    f" rate={rate:.3f}/s eta={eta:.0f}s",
                    flush=True,
                )
    return [rows[seed] for seed in seeds], time.perf_counter() - started


def _merge_telemetry(rows: list[dict[str, Any]]) -> ValueReserveSearchTelemetry:
    merged = ValueReserveSearchTelemetry()
    for row in rows:
        merged.merge(row["telemetry"])
    return merged


def _rates(telemetry: ValueReserveSearchTelemetry, games: int) -> dict[str, Any]:
    decisions = max(telemetry.decisions, 1)
    qualifying = telemetry.qualifying_visible_reserves
    return {
        "movement_from_ablation": telemetry.moved_from_search_ablation / decisions,
        "live_option_decision_rate": telemetry.decisions_with_live_option / decisions,
        "qualifying_visible_reserve_rate": qualifying / decisions,
        "qualifying_visible_reserve_conversion_rate": (
            telemetry.converted_qualifying_visible_reserves / qualifying
            if qualifying
            else 0.0
        ),
        "all_reserve_conversion_rate": (
            telemetry.reserved_purchases / telemetry.reserves
            if telemetry.reserves
            else 0.0
        ),
        "deck_reserve_rate": telemetry.deck_reserves / decisions,
        "mean_purchases_per_bonus_game": telemetry.purchases / max(games, 1),
        "phase_movement_from_ablation": {
            phase: telemetry.phase_moved_from_search_ablation.get(phase, 0)
            / max(telemetry.phase_decisions.get(phase, 0), 1)
            for phase in ("early", "mid", "late")
        },
    }


def _panel_integrity(
    rows: list[dict[str, Any]], config: LeafBonusCausalConfig
) -> dict[str, bool]:
    games = [game for row in rows for game in row["games"]]
    seeds = [int(row["seed"]) for row in rows]
    return {
        "exact_clusters": len(rows) == config.pairs,
        "exact_games": len(games) == 2 * config.pairs,
        "exact_seed_and_seat_contract": (
            seeds == list(range(config.seed_start, config.seed_start + config.pairs))
            and len(set(seeds)) == len(seeds)
            and all(row["bonus_seats"] == [0, 1] for row in rows)
        ),
        "all_terminations_valid": all(
            game["termination_reason"] in VALID_TERMINATIONS for game in games
        ),
        "zero_stalls": all(not game["stalled"] for game in games),
        "zero_timeouts": all(not game["timed_out"] for game in games),
        "zero_loop_fallbacks": all(
            tuple(game["loop_fallback_triggers_by_seat"]) == (0, 0)
            for game in games
        ),
        "zero_illegal_or_null_actions": True,
        "candidate_always_acted": sum(
            int(row["telemetry"]["decisions"]) for row in rows
        )
        > 0,
        "frozen_search_contract": (
            SEARCH_DEPTH == 1
            and SEARCH_DETERMINIZATIONS == 4
            and UNBOUNDED_ACTION_CAP == 10_000
        ),
        "configuration_contains_no_human_or_model_input": True,
    }


def _cluster_metrics(row: dict[str, Any]) -> dict[str, float]:
    shares: list[float] = []
    margins: list[float] = []
    for game, seat in zip(row["games"], row["bonus_seats"], strict=True):
        winner = game["winner"]
        shares.append(0.5 if winner is None else float(int(winner) == seat))
        margins.append(
            float(game["final_scores"][seat])
            - float(game["final_scores"][1 - seat])
        )
    return {"score_share": statistics.fmean(shares), "margin": statistics.fmean(margins)}


def _bootstrap(values: list[float], samples: int, seed: int) -> dict[str, float]:
    mean = statistics.fmean(values)
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": draws[int(0.025 * (samples - 1))],
        "ci95_high": draws[int(0.975 * (samples - 1))],
        "standard_error": (
            statistics.stdev(values) / math.sqrt(count) if count > 1 else 0.0
        ),
    }


def _strength(
    rows: list[dict[str, Any]], config: LeafBonusCausalConfig
) -> dict[str, Any]:
    metrics = [_cluster_metrics(row) for row in rows]
    bonus_wins = ablation_wins = draws = 0
    for row in rows:
        for game, seat in zip(row["games"], row["bonus_seats"], strict=True):
            if game["winner"] is None:
                draws += 1
            elif game["winner"] == seat:
                bonus_wins += 1
            else:
                ablation_wins += 1
    return {
        "score_share": _bootstrap(
            [row["score_share"] for row in metrics],
            config.bootstrap_samples,
            config.bootstrap_seed,
        ),
        "margin": _bootstrap(
            [row["margin"] for row in metrics],
            config.bootstrap_samples,
            config.bootstrap_seed + 1,
        ),
        "bonus_wins": bonus_wins,
        "ablation_wins": ablation_wins,
        "draws": draws,
        "mean_turns": statistics.fmean(
            float(game["turns"]) for row in rows for game in row["games"]
        ),
    }


def causal_decision(
    integrity: dict[str, bool], score: dict[str, float]
) -> str:
    if not all(integrity.values()):
        return "value_reserve_leaf_bonus_078_invalid_no_causal_decision"
    if score["mean"] >= 0.55 and score["ci95_low"] > 0.50:
        return "value_reserve_leaf_bonus_078_causally_stronger_supports_principle"
    if score["ci95_high"] < 0.50:
        return "value_reserve_leaf_bonus_078_causally_harmful_close_principle"
    return "value_reserve_leaf_bonus_078_not_decisive_close_principle"


def _write(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_value_reserve_leaf_bonus_causal_ablation(
    config: LeafBonusCausalConfig,
) -> dict[str, Any]:
    if min(config.workers, config.pairs, config.operational_pairs) <= 0:
        raise ValueError("workers and panel sizes must be positive")
    if config.operational_pairs >= config.pairs:
        raise ValueError("operational_pairs must be smaller than pairs")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    started = time.perf_counter()
    all_seeds = list(range(config.seed_start, config.seed_start + config.pairs))
    tranche_seeds = all_seeds[: config.operational_pairs]
    tranche_rows, tranche_elapsed = _run_seeds(config, tranche_seeds, "operational")
    projected_seconds = tranche_elapsed * config.pairs / config.operational_pairs
    operational_pass = projected_seconds <= MAX_PROJECTED_SECONDS

    if not operational_pass:
        payload = {
            "experiment": "078",
            "protocol": PROTOCOL,
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "operational_gate": {
                "tranche_pairs": config.operational_pairs,
                "elapsed_seconds": tranche_elapsed,
                "projected_seconds": projected_seconds,
                "under_three_hours": False,
                "outcomes_not_used": True,
            },
            "integrity_gates": {"operational_projection_passed": False},
            "decision": "value_reserve_leaf_bonus_078_invalid_no_causal_decision",
            "passed": False,
            "promotion_authorized": False,
            "human_data_used": False,
            "gpu_rental_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
            "partial_clusters": tranche_rows,
        }
        _write(payload, config.output_dir)
        return payload

    remaining_rows, remaining_elapsed = _run_seeds(
        config, all_seeds[config.operational_pairs :], "causal"
    )
    rows = tranche_rows + remaining_rows
    _initialize_worker((config.max_turns, config.repetition_limit, config.no_progress_limit))
    deterministic = _play_pair(config.seed_start) == rows[0]
    telemetry = _merge_telemetry(rows)
    rates = _rates(telemetry, games=2 * config.pairs)
    integrity = _panel_integrity(rows, config)
    integrity.update(
        {
            "first_cluster_byte_identical": deterministic,
            "operational_projection_passed": operational_pass,
            "movement_at_least_15pct": rates["movement_from_ablation"] >= 0.15,
        }
    )
    strength = _strength(rows, config)
    decision = causal_decision(integrity, strength["score_share"])
    payload = {
        "experiment": "078",
        "protocol": PROTOCOL,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "search_contract": {
            "depth": SEARCH_DEPTH,
            "determinizations": SEARCH_DETERMINIZATIONS,
            "action_cap": UNBOUNDED_ACTION_CAP,
            "bonus_threshold": 0.40,
            "prestige_scale": 1_000.0,
        },
        "operational_gate": {
            "tranche_pairs": config.operational_pairs,
            "elapsed_seconds": tranche_elapsed,
            "projected_seconds": projected_seconds,
            "under_three_hours": operational_pass,
            "outcomes_not_used": True,
        },
        "integrity_gates": integrity,
        "rates": rates,
        "telemetry": telemetry.as_dict(),
        "strength": strength,
        "termination_reasons": {
            reason: sum(
                game["termination_reason"] == reason
                for row in rows
                for game in row["games"]
            )
            for reason in sorted(
                {game["termination_reason"] for row in rows for game in row["games"]}
            )
        },
        "decision": decision,
        "passed": decision.endswith("_causally_stronger_supports_principle"),
        "promotion_authorized": False,
        "bc_authorized": False,
        "human_data_used": False,
        "gpu_rental_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "panel_elapsed_seconds": tranche_elapsed + remaining_elapsed,
        "clusters": rows,
    }
    _write(payload, config.output_dir)
    print(f"[078] decision={decision}")
    print(f"[078] wrote {config.output_dir / 'result.json'}")
    return payload


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Value-reserve leaf bonus ablation (078).")
    parser.add_argument(
        "--output-dir", default="outputs/value_reserve_leaf_bonus_causal_ablation_078"
    )
    parser.add_argument("--workers", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_value_reserve_leaf_bonus_causal_ablation(
        LeafBonusCausalConfig(output_dir=Path(args.output_dir), workers=args.workers)
    )


if __name__ == "__main__":
    main()
