"""Power analysis of experiment 069's final broad-gate conjunction.

Supports the claim in `docs/project_review_response_070.md` section 2.3 that the
`069` final gate had roughly 43% power at its own declared +0.025 minimum
detectable effect, with the hard per-stratum sub-gates as the dominant power sink.

Method: calibrate the null sampling distribution from the 200 real paired clusters
in experiment `066` -- the largest fresh balanced recurrent panel carrying per-seat
cluster records -- resample them to `069`'s planned panel shape (100 clusters per
each of 4 opponents), add a constant true improvement, and evaluate the exact
preregistered conjunction. Only sampling-noise conditions are simulated;
integrity and the policy-movement condition are treated as passing.

Run from the repository root:

    python scripts/analysis/gate_power_069.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics


RESULT_066 = Path("outputs/recurrent_broad_confirmation_066/result.json")
OPPONENTS = ("035", "016", "023", "shallow")
CLUSTERS_PER_OPPONENT = 100
# Score-share-to-margin conversion calibrated from resolver 050, where a +0.19
# score share came with a +3.595 mean final-score margin.
MARGIN_PER_SHARE = 3.595 / 0.19
TRIALS = 4000
SEED = 690_069


def load_clusters() -> dict[str, list[tuple[float, float, float, float]]]:
    payload = json.loads(RESULT_066.read_text(encoding="utf-8"))
    by_opponent: dict[str, list[tuple[float, float, float, float]]] = {
        label: [] for label in OPPONENTS
    }
    for row in payload["paired_seed_records"]:
        by_opponent[row["opponent"]].append(
            (
                float(row["score_share_delta"]),
                float(row["margin_delta"]),
                float(row["seat0_score_delta"]),
                float(row["seat1_score_delta"]),
            )
        )
    return by_opponent


def stratified_mean(panel, index: int) -> float:
    return statistics.fmean(row[index] for rows in panel.values() for row in rows)


def stratified_se(panel, index: int) -> float:
    total = 0.0
    strata = 0
    for rows in panel.values():
        values = [row[index] for row in rows]
        if len(values) < 2:
            continue
        total += statistics.variance(values) / len(values)
        strata += 1
    return math.sqrt(total) / strata if strata else float("inf")


def evaluate(panel) -> dict[str, bool]:
    """The nine sampling-noise sub-gates of the 069 final broad conjunction."""
    aggregate_score = stratified_mean(panel, 0)
    aggregate_margin = stratified_mean(panel, 1)
    score_low = aggregate_score - 1.96 * stratified_se(panel, 0)
    margin_low = aggregate_margin - 1.96 * stratified_se(panel, 1)

    opponent_scores: dict[str, float] = {}
    opponent_ci_high: dict[str, float] = {}
    for label, rows in panel.items():
        values = [row[0] for row in rows]
        mean = statistics.fmean(values)
        se = statistics.stdev(values) / math.sqrt(len(values))
        opponent_scores[label] = mean
        opponent_ci_high[label] = mean + 1.96 * se

    flat = [row for rows in panel.values() for row in rows]
    seat0 = statistics.fmean(row[2] for row in flat)
    seat1 = statistics.fmean(row[3] for row in flat)
    even = statistics.fmean(row[0] for i, row in enumerate(flat) if i % 2 == 0)
    odd = statistics.fmean(row[0] for i, row in enumerate(flat) if i % 2 == 1)

    return {
        "g2_aggregate_score_at_least_0.025": aggregate_score >= 0.025,
        "g3_score_ci_low_at_least_-0.025": score_low >= -0.025,
        "g4_margin_strictly_positive": aggregate_margin > 0.0,
        "g5_margin_ci_low_at_least_-0.50": margin_low >= -0.50,
        "g6_three_of_four_opponents_nonnegative": sum(
            1 for value in opponent_scores.values() if value >= 0.0
        )
        >= 3,
        "g7_no_opponent_below_-0.05_and_no_ci_high_below_0": all(
            value >= -0.05 for value in opponent_scores.values()
        )
        and all(value >= 0.0 for value in opponent_ci_high.values()),
        "g8_both_seats_at_least_-0.025": seat0 >= -0.025 and seat1 >= -0.025,
        "g9_both_halves_at_least_-0.025": even >= -0.025 and odd >= -0.025,
    }


def simulate(by_opponent, delta_share: float, trials: int = TRIALS):
    rng = random.Random(SEED)
    observed = [row for rows in by_opponent.values() for row in rows]
    centres = [statistics.fmean(row[index] for row in observed) for index in range(4)]
    shifts = (
        delta_share,
        delta_share * MARGIN_PER_SHARE,
        delta_share,
        delta_share,
    )
    pools = {
        label: [
            tuple(row[i] - centres[i] + shifts[i] for i in range(4)) for row in rows
        ]
        for label, rows in by_opponent.items()
    }
    tallies: dict[str, int] = {}
    passes = 0
    for _ in range(trials):
        panel = {
            label: [rng.choice(pool) for _ in range(CLUSTERS_PER_OPPONENT)]
            for label, pool in pools.items()
        }
        outcome = evaluate(panel)
        for key, value in outcome.items():
            tallies[key] = tallies.get(key, 0) + int(value)
        passes += int(all(outcome.values()))
    return passes / trials, {k: v / trials for k, v in tallies.items()}


def main() -> None:
    by_opponent = load_clusters()
    observed = [row for rows in by_opponent.values() for row in rows]
    print(f"observed 066 clusters: {len(observed)}")
    print(
        "observed aggregate score delta:"
        f" {statistics.fmean(r[0] for r in observed):+.5f}"
    )
    print(
        "observed cluster score-delta sd:"
        f" {statistics.stdev(r[0] for r in observed):.4f}"
    )
    print()
    print("069 final broad-gate conjunction pass probability")
    print(f"{'true delta':>11} {'conjunction':>12}")
    for delta in (0.0, 0.01, 0.02, 0.025, 0.03, 0.05, 0.08, 0.12):
        power, marginals = simulate(by_opponent, delta)
        print(f"{delta:>11.3f} {power:>12.3f}")
        if delta == 0.025:
            for key in sorted(marginals):
                print(f"{'':>11}   {marginals[key]:.3f}  {key}")


if __name__ == "__main__":
    main()
