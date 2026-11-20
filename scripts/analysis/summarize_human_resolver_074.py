"""Summarize the preregistered human--resolver calibration (experiment 074)."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import beta


SUMMARY_PATH = Path("outputs/human_games/games_summary.jsonl")
REPLAY_PATH = Path("outputs/human_games/replays.jsonl")
OUTPUT_PATH = Path("outputs/human_resolver_ceiling_calibration_074/result.json")
EXPECTED_SEEDS = tuple(range(800_000, 800_010))
CHECKPOINT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
EXPECTED_CONTRACT = {
    "determinizations": 3,
    "root_actions": 4,
    "opponent_actions": 4,
    "discovery_continuations": 4,
    "continuation_epsilon": 0.1,
    "continuation_temperature": 0.5,
    "continuation_top_k": 3,
    "score_margin_scale": 20.0,
    "expected_branches_per_search": 192,
}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clopper_pearson(wins: int, games: int, alpha: float = 0.05) -> tuple[float, float]:
    low = 0.0 if wins == 0 else float(beta.ppf(alpha / 2, wins, games - wins + 1))
    high = 1.0 if wins == games else float(beta.ppf(1 - alpha / 2, wins + 1, games - wins))
    return low, high


def _binomial_upper_tail(wins: int, games: int, probability: float = 0.5) -> float:
    return sum(
        math.comb(games, value)
        * probability**value
        * (1.0 - probability) ** (games - value)
        for value in range(wins, games + 1)
    )


def main() -> None:
    all_summaries = _read_jsonl(SUMMARY_PATH)
    games = [row for row in all_summaries if row.get("opponent_kind") == "resolver"]
    by_seed = {int(row["seed"]): row for row in games}
    if len(games) != len(EXPECTED_SEEDS) or tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise RuntimeError("Resolver summaries do not match the ten frozen seeds exactly.")

    all_replays = _read_jsonl(REPLAY_PATH)
    replay_rows = [row for row in all_replays if int(row.get("game_seed", -1)) in EXPECTED_SEEDS]
    replay_by_seed: dict[int, list[dict]] = {seed: [] for seed in EXPECTED_SEEDS}
    for row in replay_rows:
        replay_by_seed[int(row["game_seed"])].append(row)

    integrity = {
        "exact_seed_schedule": True,
        "exact_alternating_seats": True,
        "summaries_match_replays": True,
        "contiguous_complete_replays": True,
        "frozen_checkpoint_and_contract": True,
        "exact_branch_accounting": True,
        "all_continuations_resolved": True,
        "zero_hard_caps": True,
        "zero_loop_adjudications": True,
        "zero_rollout_fallback_triggers": True,
    }

    margins = []
    wins = draws = losses = 0
    resolver_totals: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    phase_totals = {phase: Counter() for phase in ("early", "mid", "late")}
    per_game = []

    for offset, seed in enumerate(EXPECTED_SEEDS):
        summary = by_seed[seed]
        expected_human_seat = offset % 2
        rows = replay_by_seed[seed]
        integrity["exact_alternating_seats"] &= int(summary["human_seat"]) == expected_human_seat
        integrity["contiguous_complete_replays"] &= (
            len(rows) == int(summary["turns"])
            and sorted(int(row["turn_index"]) for row in rows) == list(range(int(summary["turns"])))
            and all(row["game_termination_reason"] == "completed" for row in rows)
            and not any(bool(row["game_stalled"]) or bool(row["game_timed_out"]) for row in rows)
        )
        first = rows[0]
        integrity["summaries_match_replays"] &= (
            int(first["human_player_id"]) == int(summary["human_seat"])
            and list(first["game_final_scores"]) == list(summary["final_scores"])
            and first["game_winner"] == summary["winner"]
            and int(first["game_turns"]) == int(summary["turns"])
        )
        opponent_seat = 1 - expected_human_seat
        metadata = first["game_bot_metadata"][opponent_seat]
        telemetry = summary["opponent_telemetry"]
        integrity["summaries_match_replays"] &= metadata["final_telemetry"] == telemetry
        integrity["frozen_checkpoint_and_contract"] &= (
            metadata["bot_class"] == "FrozenPublicBeliefResolverBot"
            and metadata["gui_opponent_kind"] == "resolver"
            and metadata["checkpoint_sha256"] == CHECKPOINT_SHA256
            and int(metadata["continuation_workers"]) == 10
            and metadata["resolver_contract"] == EXPECTED_CONTRACT
        )
        integrity["exact_branch_accounting"] &= (
            int(telemetry["continuations"])
            == int(telemetry["completed_searches"]) * EXPECTED_CONTRACT["expected_branches_per_search"]
        )
        integrity["all_continuations_resolved"] &= (
            int(telemetry["resolved_continuations"]) == int(telemetry["continuations"])
        )
        integrity["zero_hard_caps"] &= int(telemetry["hard_caps"]) == 0
        integrity["zero_loop_adjudications"] &= int(telemetry["loop_adjudications"]) == 0
        integrity["zero_rollout_fallback_triggers"] &= int(telemetry["rollout_fallback_triggers"]) == 0

        human_won = bool(summary["human_won"])
        winner = summary["winner"]
        if winner is None:
            draws += 1
        elif human_won:
            wins += 1
        else:
            losses += 1
        scores = summary["final_scores"]
        margin = int(scores[expected_human_seat]) - int(scores[opponent_seat])
        margins.append(margin)

        for key in (
            "decisions",
            "completed_searches",
            "switches",
            "fallbacks",
            "continuations",
            "resolved_continuations",
            "hard_caps",
            "loop_adjudications",
            "rollout_fallback_triggers",
        ):
            resolver_totals[key] += int(telemetry[key])
        fallback_reasons.update(telemetry["fallback_reasons"])
        for phase in phase_totals:
            phase_totals[phase].update(telemetry["by_phase"][phase])
        per_game.append(
            {
                "seed": seed,
                "human_seat": expected_human_seat,
                "human_won": human_won,
                "final_scores": scores,
                "human_margin": margin,
                "turns": int(summary["turns"]),
                "resolver_telemetry": telemetry,
            }
        )

    score_share = (wins + 0.5 * draws) / len(games)
    interval = _clopper_pearson(wins, wins + losses) if draws == 0 else None
    if score_share >= 0.80:
        interpretation = "supports_resolver_ceiling_concern_deprioritize_cloning_as_endpoint"
    elif score_share <= 0.40:
        interpretation = "falsifies_working_claim_resolver_ceiling_below_human"
    else:
        interpretation = "inconclusive_no_ceiling_or_branch_decision"

    phase_payload = {}
    for phase, counts in phase_totals.items():
        decisions = int(counts["decisions"])
        phase_payload[phase] = {
            "decisions": decisions,
            "completed_searches": int(counts["completed_searches"]),
            "switches": int(counts["switches"]),
            "fallbacks": int(counts["fallbacks"]),
            "switch_dose_per_decision": counts["switches"] / decisions if decisions else None,
        }

    result = {
        "experiment": "074_human_resolver_ceiling_calibration",
        "protocol": "docs/experiments/human_resolver_ceiling_calibration_074.md",
        "data_sources": {
            "summaries": str(SUMMARY_PATH),
            "summaries_sha256": _sha256(SUMMARY_PATH),
            "replays": str(REPLAY_PATH),
            "replays_sha256": _sha256(REPLAY_PATH),
        },
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "primary_endpoint": {
            "games": len(games),
            "human_wins": wins,
            "draws": draws,
            "human_losses": losses,
            "human_score_share": score_share,
            "clopper_pearson_95": list(interval) if interval is not None else None,
            "one_sided_exact_p_vs_0.5": _binomial_upper_tail(wins, wins + losses),
            "preregistered_interpretation": interpretation,
        },
        "human_margin": {
            "mean": statistics.fmean(margins),
            "median": statistics.median(margins),
            "minimum": min(margins),
            "maximum": max(margins),
            "values_by_seed": margins,
        },
        "seat_breakdown": {
            f"human_seat_{seat}": {
                "games": sum(int(game["human_seat"] == seat) for game in per_game),
                "wins": sum(int(game["human_seat"] == seat and game["human_won"]) for game in per_game),
            }
            for seat in (0, 1)
        },
        "resolver_dose": {
            **dict(resolver_totals),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "switch_dose_per_decision": resolver_totals["switches"] / resolver_totals["decisions"],
            "switch_fraction_of_completed_searches": resolver_totals["switches"]
            / resolver_totals["completed_searches"],
            "by_phase": phase_payload,
        },
        "per_game": per_game,
        "decision": interpretation if all(integrity.values()) else "integrity_failure_no_interpretation",
        "training_or_promotion_authorized": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps(result["primary_endpoint"], indent=2, sort_keys=True))
    print(json.dumps(result["resolver_dose"], indent=2, sort_keys=True))
    print(f"integrity_passed={result['integrity_passed']}")
    print(f"result_sha256={_sha256(OUTPUT_PATH)}")
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
