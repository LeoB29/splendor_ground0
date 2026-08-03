"""Run the preregistered scripted value-reserve teacher panel (`072`).

Protocol: `docs/experiments/scripted_value_teacher_072.md`.

Deterministic rule-based teacher versus raw legal `035` on a frozen paired panel.
No training, no checkpoint for the teacher, no randomness. The faithfulness
diagnostics are reported and gated on nothing.
"""

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
    ValueReserveTeacherBot,
    ValueReserveTelemetry,
)
from splendor_ai.bots.value_reserve_bot import ValueReserveRules
from splendor_ai.eval.match import play_game


PROTOCOL = "docs/experiments/scripted_value_teacher_072.md"
CHECKPOINT_035_SHA256 = (
    "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
)
SEED_START = 750_000
PAIRS = 200
BOOTSTRAP_SEED = 750_072
BOOTSTRAP_SAMPLES = 20_000
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")

#: Expert action mix from the ten held-out human games, for the faithfulness
#: diagnostic only. Never a gate and never a tuning target.
EXPERT_MIX = {
    "TAKE_TOKENS": 0.513,
    "RESERVE_VISIBLE": 0.233,
    "BUY_RESERVED": 0.188,
    "BUY_VISIBLE": 0.066,
    "RESERVE_DECK": 0.0,
}
EXPERT_RESERVES_BY_TIER = {"1": 14, "2": 21, "3": 28}


@dataclass(frozen=True, slots=True)
class TeacherPanelConfig:
    checkpoint: Path
    output_dir: Path
    pairs: int = PAIRS
    seed_start: int = SEED_START
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    device: str = "cpu"
    # `073` carries the two elicited rule changes; both default off so `072`
    # remains bit-reproducible from this same runner.
    experiment: str = "072"
    protocol: str = PROTOCOL
    bootstrap_seed: int = BOOTSTRAP_SEED
    protect_gold: bool = False
    reserve_and_clear: bool = False


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
    rules: ValueReserveRules,
) -> None:
    _WORKER["opponent"] = LegalActionAttentionScorerBot(checkpoint, device=device)
    _WORKER["limits"] = limits
    _WORKER["rules"] = rules


def _play_pair(seed: int) -> dict[str, Any]:
    opponent: LegalActionAttentionScorerBot = _WORKER["opponent"]
    max_turns, repetition_limit, no_progress_limit = _WORKER["limits"]
    rules: ValueReserveRules = _WORKER["rules"]

    teacher = ValueReserveTeacherBot(rules)
    first = play_game(
        teacher,
        opponent,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    telemetry = dict(teacher.telemetry.as_dict())

    teacher_second = ValueReserveTeacherBot(rules)
    second = play_game(
        opponent,
        teacher_second,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    merged = ValueReserveTelemetry()
    merged.merge(telemetry)
    merged.merge(teacher_second.telemetry.as_dict())

    return {
        "seed": seed,
        "teacher_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
        "telemetry": merged.as_dict(),
        "nobles": [
            list(first.final_scores),
            list(second.final_scores),
        ],
    }


def _cluster_metrics(row: dict[str, Any]) -> dict[str, float]:
    shares, margins = [], []
    for game, seat in zip(row["games"], row["teacher_seats"], strict=True):
        winner = game["winner"]
        scores = game["final_scores"]
        if winner is None:
            shares.append(0.5)
        else:
            shares.append(1.0 if int(winner) == seat else 0.0)
        margins.append(float(scores[seat]) - float(scores[1 - seat]))
    return {
        "score_share": statistics.fmean(shares),
        "margin": statistics.fmean(margins),
    }


def _bootstrap(values: list[float], seed: int) -> dict[str, float]:
    mean = statistics.fmean(values)
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "mean": mean,
        "ci95_low": draws[int(0.025 * (BOOTSTRAP_SAMPLES - 1))],
        "ci95_high": draws[int(0.975 * (BOOTSTRAP_SAMPLES - 1))],
        "standard_error": (
            statistics.stdev(values) / math.sqrt(count) if count > 1 else 0.0
        ),
    }


def _integrity(
    config: TeacherPanelConfig,
    rows: list[dict[str, Any]],
    telemetry: ValueReserveTelemetry,
    digest: str,
    deterministic: bool,
) -> dict[str, bool]:
    seeds = [int(row["seed"]) for row in rows]
    games = [game for row in rows for game in row["games"]]
    reasons = {str(game["termination_reason"]) for game in games}
    extra: dict[str, bool] = {}
    if config.protect_gold:
        extra["gold_discipline_active"] = telemetry.gold_discipline_skips > 0
    if config.reserve_and_clear:
        extra["reserve_and_clear_active"] = (
            telemetry.multi_target_hand_decisions == 0
        )
    return {
        **extra,
        "exact_clusters": len(rows) == config.pairs,
        "exact_games": len(games) == 2 * config.pairs,
        "exact_reserved_seeds": seeds
        == list(range(config.seed_start, config.seed_start + config.pairs))
        and len(set(seeds)) == len(seeds),
        "all_terminations_valid": reasons.issubset(set(VALID_TERMINATIONS)),
        "zero_stalls": all(not game["stalled"] for game in games),
        "zero_timeouts": all(not game["timed_out"] for game in games),
        "teacher_always_acted": telemetry.decisions > 0,
        "zero_deck_reserves": telemetry.deck_reserves == 0,
        "teacher_deterministic": deterministic,
        "checkpoint_hash_frozen": digest == CHECKPOINT_035_SHA256,
    }


def _faithfulness(telemetry: ValueReserveTelemetry) -> dict[str, Any]:
    total = max(telemetry.decisions, 1)
    mix = {
        name: telemetry.action_counts.get(name, 0) / total for name in EXPERT_MIX
    }
    reserves = max(telemetry.reserves, 1)
    tier_share = {
        tier: telemetry.reserves_by_tier.get(int(tier), 0) / reserves
        for tier in EXPERT_RESERVES_BY_TIER
    }
    expert_reserve_total = sum(EXPERT_RESERVES_BY_TIER.values())
    return {
        "teacher_action_mix": mix,
        "expert_action_mix": EXPERT_MIX,
        "action_mix_absolute_deviation": {
            name: abs(mix[name] - EXPERT_MIX[name]) for name in EXPERT_MIX
        },
        "teacher_reserve_tier_share": tier_share,
        "expert_reserve_tier_share": {
            tier: count / expert_reserve_total
            for tier, count in EXPERT_RESERVES_BY_TIER.items()
        },
        "teacher_reserve_conversion_rate": (
            telemetry.reserved_purchases / reserves if telemetry.reserves else 0.0
        ),
        "expert_reserve_conversion_rate": 51 / 63,
        "note": (
            "Reported only. Not a gate and not a tuning target. A mix far from the"
            " expert's indicates the rules did not encode the described strategy,"
            " whatever the score."
        ),
    }


def run_scripted_value_teacher(config: TeacherPanelConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256(config.checkpoint)

    seeds = list(range(config.seed_start, config.seed_start + config.pairs))
    rows: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_initialize_worker,
        initargs=(
            str(config.checkpoint),
            config.device,
            (config.max_turns, config.repetition_limit, config.no_progress_limit),
            ValueReserveRules(
                protect_gold=config.protect_gold,
                reserve_and_clear=config.reserve_and_clear,
            ),
        ),
    ) as executor:
        futures = {executor.submit(_play_pair, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows[futures[future]] = future.result()
            if completed % 25 == 0 or completed == len(seeds):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                remaining = (len(seeds) - completed) / rate if rate else math.inf
                print(
                    f"[{config.experiment}] clusters={completed}/{len(seeds)}"
                    f" rate={rate:.2f}/s eta={remaining:.0f}s",
                    flush=True,
                )

    ordered = [rows[seed] for seed in seeds]

    # Integrity gate 5: replay the first cluster and require an exact match.
    _initialize_worker(
        str(config.checkpoint),
        config.device,
        (config.max_turns, config.repetition_limit, config.no_progress_limit),
        ValueReserveRules(
            protect_gold=config.protect_gold,
            reserve_and_clear=config.reserve_and_clear,
        ),
    )
    repeat = _play_pair(seeds[0])
    deterministic = repeat == ordered[0]

    telemetry = ValueReserveTelemetry()
    for row in ordered:
        telemetry.merge(row["telemetry"])

    metrics = [_cluster_metrics(row) for row in ordered]
    score = _bootstrap([m["score_share"] for m in metrics], config.bootstrap_seed)
    margin = _bootstrap([m["margin"] for m in metrics], config.bootstrap_seed + 1)

    integrity = _integrity(config, ordered, telemetry, digest, deterministic)

    tag = "" if config.experiment == "072" else f"_{config.experiment}"
    if not all(integrity.values()):
        decision = f"scripted_value_teacher{tag}_invalid_no_strength_decision"
    elif score["ci95_low"] > 0.50:
        decision = (
            f"scripted_value_teacher{tag}_stronger"
            "_authorize_bc_teacher_preregistration"
        )
    elif score["ci95_high"] < 0.50:
        decision = (
            f"scripted_value_teacher{tag}_weaker_close_scripted_strategy_line"
            + ("_permanently" if config.experiment != "072" else "")
        )
    else:
        decision = (
            f"scripted_value_teacher{tag}_competitive_close_line_report_only"
            if config.experiment != "072"
            else "scripted_value_teacher_competitive_inconclusive"
        )

    payload = {
        "experiment": config.experiment,
        "protocol": config.protocol,
        "configuration": {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(config).items()
            },
            "bootstrap_seed": config.bootstrap_seed,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "checkpoint_sha256": digest,
        "integrity_gates": integrity,
        "strength": {
            "score_share": score,
            "margin": margin,
            "teacher_wins": sum(
                1
                for row in ordered
                for game, seat in zip(row["games"], row["teacher_seats"], strict=True)
                if game["winner"] == seat
            ),
            "opponent_wins": sum(
                1
                for row in ordered
                for game, seat in zip(row["games"], row["teacher_seats"], strict=True)
                if game["winner"] is not None and game["winner"] != seat
            ),
            "draws": sum(
                1 for row in ordered for game in row["games"] if game["winner"] is None
            ),
            "mean_turns": statistics.fmean(
                float(game["turns"]) for row in ordered for game in row["games"]
            ),
        },
        "teacher_telemetry": telemetry.as_dict(),
        "faithfulness_diagnostics": _faithfulness(telemetry),
        "termination_reasons": {
            reason: sum(
                1
                for row in ordered
                for game in row["games"]
                if game["termination_reason"] == reason
            )
            for reason in sorted(
                {
                    game["termination_reason"]
                    for row in ordered
                    for game in row["games"]
                }
            )
        },
        "decision": decision,
        "passed": decision.endswith(
            "_stronger_authorize_bc_teacher_preregistration"
        ),
        "promotion_authorized": False,
        "gpu_rental_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "clusters": ordered,
    }

    result_path = config.output_dir / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[{config.experiment}] decision={decision}")
    print(f"[{config.experiment}] wrote {result_path}")
    return payload


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Scripted value-reserve teacher (072).")
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/scripted_value_teacher_072")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_scripted_value_teacher(
        TeacherPanelConfig(
            checkpoint=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            pairs=args.pairs,
            seed_start=args.seed_start,
            workers=args.workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
