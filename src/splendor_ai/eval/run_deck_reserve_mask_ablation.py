"""Run the preregistered deck-reserve mask ablation (`071`).

Protocol: `docs/experiments/deck_reserve_mask_ablation_071.md`.

Candidate and baseline load the identical frozen `035` checkpoint.  The only
difference is that the candidate removes `RESERVE_DECK` from its greedy
candidate set at inference time.  The runner reports the measured dose (the
fraction of candidate decisions on which the mask changed the action) as a
primary quantity alongside paired strength, because the protocol's decision
rules are keyed on both.
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
    DeckReserveMaskTelemetry,
    DeckReserveMaskedAttentionScorerBot,
    LegalActionAttentionScorerBot,
)
from splendor_ai.eval.match import play_game


PROTOCOL = "docs/experiments/deck_reserve_mask_ablation_071.md"
CHECKPOINT_035_SHA256 = (
    "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
)
SEED_START = 740_000
PAIRS = 1_000
BOOTSTRAP_SEED = 740_071
BOOTSTRAP_SAMPLES = 20_000
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")
MINIMUM_DOSE = 0.010


@dataclass(frozen=True, slots=True)
class AblationConfig:
    checkpoint: Path
    output_dir: Path
    pairs: int = PAIRS
    seed_start: int = SEED_START
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    device: str = "cpu"


_WORKER: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialize_worker(checkpoint: str, device: str, limits: tuple[int, int, int]) -> None:
    _WORKER["candidate"] = DeckReserveMaskedAttentionScorerBot(checkpoint, device=device)
    _WORKER["baseline"] = LegalActionAttentionScorerBot(checkpoint, device=device)
    _WORKER["limits"] = limits


def _play_pair(seed: int) -> dict[str, Any]:
    candidate: DeckReserveMaskedAttentionScorerBot = _WORKER["candidate"]
    baseline: LegalActionAttentionScorerBot = _WORKER["baseline"]
    max_turns, repetition_limit, no_progress_limit = _WORKER["limits"]
    candidate.telemetry.reset()

    first = play_game(
        candidate,
        baseline,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    second = play_game(
        baseline,
        candidate,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
        "telemetry": candidate.telemetry.as_dict(),
    }


def _cluster_metrics(row: dict[str, Any]) -> dict[str, float]:
    shares = []
    margins = []
    for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
        scores = game["final_scores"]
        winner = game["winner"]
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
    config: AblationConfig,
    rows: list[dict[str, Any]],
    telemetry: DeckReserveMaskTelemetry,
    checkpoint_digest: str,
) -> dict[str, bool]:
    seeds = [int(row["seed"]) for row in rows]
    games = [game for row in rows for game in row["games"]]
    reasons = {str(game["termination_reason"]) for game in games}
    return {
        "exact_clusters": len(rows) == config.pairs,
        "exact_games": len(games) == 2 * config.pairs,
        "exact_reserved_seeds": seeds
        == list(range(config.seed_start, config.seed_start + config.pairs))
        and len(set(seeds)) == len(seeds),
        "all_terminations_valid": reasons.issubset(set(VALID_TERMINATIONS)),
        "zero_stalls": all(not game["stalled"] for game in games),
        "zero_timeouts": all(not game["timed_out"] for game in games),
        "candidate_always_acted": telemetry.decisions > 0,
        # Protocol gate 5: the unmasked greedy policy -- i.e. baseline behaviour on
        # the candidate's own states -- must actually pick RESERVE_DECK, otherwise
        # the ablation never binds and the panel is vacuous.
        "mask_bound_at_least_once": telemetry.unmasked_choice_was_deck_reserve > 0,
        "checkpoint_hash_frozen": checkpoint_digest == CHECKPOINT_035_SHA256,
    }


def run_deck_reserve_mask_ablation(config: AblationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_digest = sha256(config.checkpoint)

    seeds = list(range(config.seed_start, config.seed_start + config.pairs))
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
            rows[futures[future]] = future.result()
            if completed % 50 == 0 or completed == len(seeds):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                remaining = (len(seeds) - completed) / rate if rate > 0 else math.inf
                print(
                    f"[071] clusters={completed}/{len(seeds)}"
                    f" rate={rate:.2f}/s eta={remaining:.0f}s",
                    flush=True,
                )

    ordered = [rows[seed] for seed in seeds]
    telemetry = DeckReserveMaskTelemetry()
    for row in ordered:
        telemetry.merge(row["telemetry"])

    metrics = [_cluster_metrics(row) for row in ordered]
    score = _bootstrap([m["score_share"] for m in metrics], BOOTSTRAP_SEED)
    margin = _bootstrap([m["margin"] for m in metrics], BOOTSTRAP_SEED + 1)

    seat0 = statistics.fmean(
        1.0
        if row["games"][0]["winner"] == 0
        else 0.5
        if row["games"][0]["winner"] is None
        else 0.0
        for row in ordered
    )
    seat1 = statistics.fmean(
        1.0
        if row["games"][1]["winner"] == 1
        else 0.5
        if row["games"][1]["winner"] is None
        else 0.0
        for row in ordered
    )

    integrity = _integrity(config, ordered, telemetry, checkpoint_digest)
    dose = telemetry.switch_fraction
    dose_sufficient = dose >= MINIMUM_DOSE

    if not all(integrity.values()):
        decision = "deck_reserve_mask_ablation_invalid_no_strength_decision"
    elif not dose_sufficient:
        decision = "deck_reserve_mask_dose_too_small_inconclusive"
    elif score["ci95_low"] > 0.50:
        decision = "deck_reserve_mask_improves_authorize_confirmation_preregistration"
    elif score["ci95_high"] < 0.50:
        decision = "deck_reserve_class_is_load_bearing_close_mask_direction"
    else:
        decision = "deck_reserve_mask_neutral_at_measured_dose"

    payload = {
        "experiment": "071",
        "protocol": PROTOCOL,
        "configuration": {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(config).items()
            },
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "minimum_dose": MINIMUM_DOSE,
        },
        "checkpoint_sha256": checkpoint_digest,
        "integrity_gates": integrity,
        "dose": {
            **telemetry.as_dict(),
            "dose_at_least_minimum": dose_sufficient,
        },
        "strength": {
            "score_share": score,
            "margin": margin,
            "candidate_seat0_share": seat0,
            "candidate_seat1_share": seat1,
        },
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
        "passed": decision
        == "deck_reserve_mask_improves_authorize_confirmation_preregistration",
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
    print(f"[071] decision={decision}")
    print(f"[071] wrote {result_path}")
    return payload


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Deck-reserve mask ablation (071).")
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/deck_reserve_mask_ablation_071")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_deck_reserve_mask_ablation(
        AblationConfig(
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
