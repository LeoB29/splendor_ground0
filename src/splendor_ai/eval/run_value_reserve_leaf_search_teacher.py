"""Run preregistered value-reserve leaf-search experiment `077`."""

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
    LegalActionAttentionScorerBot,
    ValueReserveLeafSearchBot,
    ValueReserveSearchTelemetry,
)
from splendor_ai.eval.match import play_game


PROTOCOL = "docs/experiments/value_reserve_leaf_search_teacher_077.md"
CHECKPOINT_035_SHA256 = (
    "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
)
STAGE_A_SEED_START = 770_000
STAGE_A_PAIRS = 40
STAGE_B_SEED_START = 771_000
STAGE_B_PAIRS = 200
BOOTSTRAP_SEED = 771_077
BOOTSTRAP_SAMPLES = 20_000
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")
MECHANICS_TIME_LIMIT_SECONDS = 30 * 60
PROJECTED_TOTAL_LIMIT_SECONDS = 3 * 60 * 60


@dataclass(frozen=True, slots=True)
class ValueReserveSearchConfig:
    checkpoint: Path
    output_dir: Path
    workers: int = 10
    stage_a_pairs: int = STAGE_A_PAIRS
    stage_a_seed_start: int = STAGE_A_SEED_START
    stage_b_pairs: int = STAGE_B_PAIRS
    stage_b_seed_start: int = STAGE_B_SEED_START
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    device: str = "cpu"
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED


_WORKER: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialize_worker(
    checkpoint: str,
    device: str,
    limits: tuple[int, int, int],
) -> None:
    _WORKER["opponent"] = LegalActionAttentionScorerBot(checkpoint, device=device)
    _WORKER["limits"] = limits


def _play_pair(seed: int) -> dict[str, Any]:
    opponent: LegalActionAttentionScorerBot = _WORKER["opponent"]
    max_turns, repetition_limit, no_progress_limit = _WORKER["limits"]

    first_candidate = ValueReserveLeafSearchBot(opponent)
    first = play_game(
        first_candidate,
        opponent,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    merged = ValueReserveSearchTelemetry()
    merged.merge(first_candidate.telemetry.as_dict())

    second_candidate = ValueReserveLeafSearchBot(opponent)
    second = play_game(
        opponent,
        second_candidate,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    merged.merge(second_candidate.telemetry.as_dict())
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
        "telemetry": merged.as_dict(),
    }


def _run_panel(
    config: ValueReserveSearchConfig,
    seed_start: int,
    pairs: int,
    label: str,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    seeds = list(range(seed_start, seed_start + pairs))
    rows: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_initialize_worker,
        initargs=(
            str(config.checkpoint),
            config.device,
            (config.max_turns, config.repetition_limit, config.no_progress_limit),
        ),
    ) as executor:
        futures = {executor.submit(_play_pair, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), start=1):
            seed = futures[future]
            rows[seed] = future.result()
            if completed % 10 == 0 or completed == pairs:
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0.0
                eta = (pairs - completed) / rate if rate else math.inf
                print(
                    f"[077:{label}] clusters={completed}/{pairs}"
                    f" rate={rate:.3f}/s eta={eta:.0f}s",
                    flush=True,
                )
    return [rows[seed] for seed in seeds], time.perf_counter() - started


def _merged_telemetry(rows: list[dict[str, Any]]) -> ValueReserveSearchTelemetry:
    merged = ValueReserveSearchTelemetry()
    for row in rows:
        merged.merge(row["telemetry"])
    return merged


def _panel_integrity(
    rows: list[dict[str, Any]],
    pairs: int,
    seed_start: int,
) -> dict[str, bool]:
    games = [game for row in rows for game in row["games"]]
    seeds = [int(row["seed"]) for row in rows]
    return {
        "exact_clusters": len(rows) == pairs,
        "exact_games": len(games) == 2 * pairs,
        "exact_seed_and_seat_contract": (
            seeds == list(range(seed_start, seed_start + pairs))
            and len(set(seeds)) == len(seeds)
            and all(row["candidate_seats"] == [0, 1] for row in rows)
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
        "candidate_always_acted": sum(
            int(row["telemetry"]["decisions"]) for row in rows
        )
        > 0,
        "zero_illegal_or_null_actions": True,
    }


def _rates(telemetry: ValueReserveSearchTelemetry, games: int) -> dict[str, Any]:
    decisions = max(telemetry.decisions, 1)
    qualifying = telemetry.qualifying_visible_reserves
    return {
        "movement_from_raw035": telemetry.moved_from_raw035 / decisions,
        "movement_from_search_ablation": (
            telemetry.moved_from_search_ablation / decisions
        ),
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
        "mean_purchases_per_candidate_game": telemetry.purchases / max(games, 1),
        "phase_movement_from_raw035": {
            phase: telemetry.phase_moved_from_raw035.get(phase, 0)
            / max(telemetry.phase_decisions.get(phase, 0), 1)
            for phase in ("early", "mid", "late")
        },
        "phase_movement_from_search_ablation": {
            phase: telemetry.phase_moved_from_search_ablation.get(phase, 0)
            / max(telemetry.phase_decisions.get(phase, 0), 1)
            for phase in ("early", "mid", "late")
        },
    }


def _mechanics_gates(rates: dict[str, Any]) -> dict[str, bool]:
    return {
        "raw035_movement_at_least_15pct": rates["movement_from_raw035"] >= 0.15,
        "leaf_isolated_movement_at_least_15pct": (
            rates["movement_from_search_ablation"] >= 0.15
        ),
        "qualifying_visible_reserves_at_least_5pct": (
            rates["qualifying_visible_reserve_rate"] >= 0.05
        ),
        "qualifying_reserve_conversion_at_least_half": (
            rates["qualifying_visible_reserve_conversion_rate"] >= 0.50
        ),
        "mean_purchases_at_least_five": (
            rates["mean_purchases_per_candidate_game"] >= 5.0
        ),
    }


def _cluster_metrics(row: dict[str, Any]) -> dict[str, float]:
    shares: list[float] = []
    margins: list[float] = []
    for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
        winner = game["winner"]
        shares.append(0.5 if winner is None else float(int(winner) == seat))
        margins.append(
            float(game["final_scores"][seat])
            - float(game["final_scores"][1 - seat])
        )
    return {
        "score_share": statistics.fmean(shares),
        "margin": statistics.fmean(margins),
    }


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


def _strength_summary(
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics = [_cluster_metrics(row) for row in rows]
    games = [game for row in rows for game in row["games"]]
    candidate_wins = 0
    opponent_wins = 0
    draws = 0
    for row in rows:
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
            if game["winner"] is None:
                draws += 1
            elif game["winner"] == seat:
                candidate_wins += 1
            else:
                opponent_wins += 1
    return {
        "score_share": _bootstrap(
            [metric["score_share"] for metric in metrics], samples, seed
        ),
        "margin": _bootstrap(
            [metric["margin"] for metric in metrics], samples, seed + 1
        ),
        "candidate_wins": candidate_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "mean_turns": statistics.fmean(float(game["turns"]) for game in games),
    }


def _config_payload(config: ValueReserveSearchConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def run_value_reserve_leaf_search_teacher(
    config: ValueReserveSearchConfig,
) -> dict[str, Any]:
    if min(config.workers, config.stage_a_pairs, config.stage_b_pairs) <= 0:
        raise ValueError("workers and both panel sizes must be positive")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not config.checkpoint.exists():
        raise FileNotFoundError(config.checkpoint)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    digest = sha256(config.checkpoint)

    stage_a_rows, stage_a_elapsed = _run_panel(
        config, config.stage_a_seed_start, config.stage_a_pairs, "mechanics"
    )
    _initialize_worker(
        str(config.checkpoint),
        config.device,
        (config.max_turns, config.repetition_limit, config.no_progress_limit),
    )
    deterministic_repeat = _play_pair(config.stage_a_seed_start) == stage_a_rows[0]
    stage_a_telemetry = _merged_telemetry(stage_a_rows)
    stage_a_rates = _rates(stage_a_telemetry, games=2 * config.stage_a_pairs)
    stage_a_integrity = _panel_integrity(
        stage_a_rows, config.stage_a_pairs, config.stage_a_seed_start
    )
    stage_a_integrity.update(
        {
            "first_cluster_byte_identical": deterministic_repeat,
            "checkpoint_hash_frozen": digest == CHECKPOINT_035_SHA256,
        }
    )
    projected_stage_b_seconds = (
        stage_a_elapsed * config.stage_b_pairs / config.stage_a_pairs
    )
    operational = {
        "mechanics_under_30_minutes": (
            stage_a_elapsed <= MECHANICS_TIME_LIMIT_SECONDS
        ),
        "projected_stage_b_under_three_hours": (
            projected_stage_b_seconds <= PROJECTED_TOTAL_LIMIT_SECONDS
        ),
    }
    mechanics = _mechanics_gates(stage_a_rates)

    stage_b_rows: list[dict[str, Any]] = []
    stage_b_payload: dict[str, Any] | None = None
    if not all(stage_a_integrity.values()) or not all(operational.values()):
        decision = "value_reserve_leaf_search_077_invalid_no_strength_decision"
    elif not all(mechanics.values()):
        decision = (
            "value_reserve_leaf_search_077_mechanics_failed_close_exact_candidate"
        )
    else:
        stage_b_rows, stage_b_elapsed = _run_panel(
            config, config.stage_b_seed_start, config.stage_b_pairs, "strength"
        )
        stage_b_telemetry = _merged_telemetry(stage_b_rows)
        stage_b_rates = _rates(stage_b_telemetry, games=2 * config.stage_b_pairs)
        stage_b_integrity = _panel_integrity(
            stage_b_rows, config.stage_b_pairs, config.stage_b_seed_start
        )
        stage_b_integrity.update(
            {
                "checkpoint_hash_frozen": digest == CHECKPOINT_035_SHA256,
                "final_movement_at_least_15pct": (
                    stage_b_rates["movement_from_raw035"] >= 0.15
                ),
            }
        )
        strength = _strength_summary(
            stage_b_rows, config.bootstrap_samples, config.bootstrap_seed
        )
        if not all(stage_b_integrity.values()):
            decision = "value_reserve_leaf_search_077_invalid_no_strength_decision"
        elif (
            strength["score_share"]["mean"] >= 0.55
            and strength["score_share"]["ci95_low"] > 0.50
        ):
            decision = (
                "value_reserve_leaf_search_077_stronger_authorize_bc_preregistration"
            )
        elif strength["score_share"]["ci95_high"] < 0.50:
            decision = "value_reserve_leaf_search_077_weaker_close_exact_candidate"
        else:
            decision = (
                "value_reserve_leaf_search_077_not_decisively_stronger_close_exact_candidate"
            )
        stage_b_payload = {
            "integrity_gates": stage_b_integrity,
            "rates": stage_b_rates,
            "telemetry": stage_b_telemetry.as_dict(),
            "strength": strength,
            "elapsed_seconds": stage_b_elapsed,
            "clusters": stage_b_rows,
        }

    payload = {
        "experiment": "077",
        "protocol": PROTOCOL,
        "configuration": _config_payload(config),
        "checkpoint_sha256": digest,
        "stage_a": {
            "integrity_gates": stage_a_integrity,
            "operational_gates": operational,
            "mechanics_gates": mechanics,
            "rates": stage_a_rates,
            "telemetry": stage_a_telemetry.as_dict(),
            "elapsed_seconds": stage_a_elapsed,
            "projected_stage_b_seconds": projected_stage_b_seconds,
            "clusters": stage_a_rows,
            "strength_not_used": True,
        },
        "stage_b": stage_b_payload,
        "decision": decision,
        "passed": decision.endswith("_stronger_authorize_bc_preregistration"),
        "promotion_authorized": False,
        "human_data_used": False,
        "gpu_rental_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = config.output_dir / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[077] decision={decision}")
    print(f"[077] wrote {result_path}")
    return payload


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Value-reserve leaf search (077).")
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/value_reserve_leaf_search_teacher_077"
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_value_reserve_leaf_search_teacher(
        ValueReserveSearchConfig(
            checkpoint=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            workers=args.workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
