"""Run preregistered MCCFR average-policy repair experiment 046."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from splendor_ai.cfr import (
    CorrectedExternalSamplingMCCFRSolver,
    ExactCFRSolver,
    ReducedSplendorCFRGame,
    profile_exploitability,
)
from splendor_ai.cfr.tabular import expected_utility


EXACT_HASH_044 = "b8184ec7c9dbfa3611d014f98c674efbeae27728429b2fa90c7939e1bbe043ab"
DEFAULT_SEEDS = (440046, 440047, 440048, 440049, 440050)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _compact_exploitability(analysis: dict[str, Any]) -> dict[str, Any]:
    response_contract: dict[str, Any] = {}
    legal = True
    for player, response in analysis["best_responses"].items():
        player_legal = (
            response["selected_information_sets"] == response["expected_information_sets"]
            and response["missing_selections"] == 0
            and response["unstable_action_sets"] == 0
            and all(
                selection["selected_action"] in selection["actions"]
                for selection in response["selections"]
            )
        )
        legal = legal and player_legal
        response_contract[player] = {
            "selected_information_sets": response["selected_information_sets"],
            "expected_information_sets": response["expected_information_sets"],
            "missing_selections": response["missing_selections"],
            "unstable_action_sets": response["unstable_action_sets"],
            "legal": player_legal,
        }
    return {
        key: value for key, value in analysis.items() if key != "best_responses"
    } | {
        "best_response_contract": response_contract,
        "best_response_integrity": legal,
    }


def _observation_bands(exact_keys, solver) -> dict[str, int]:
    bands = {"missing_or_zero": 0, "1_9": 0, "10_99": 0, "100_999": 0, "at_least_1000": 0}
    for key in exact_keys:
        node = solver.nodes.get(key)
        observations = 0 if node is None else node.average_observations
        if observations <= 0:
            bands["missing_or_zero"] += 1
        elif observations < 10:
            bands["1_9"] += 1
        elif observations < 100:
            bands["10_99"] += 1
        elif observations < 1000:
            bands["100_999"] += 1
        else:
            bands["at_least_1000"] += 1
    return bands


def _analyze_seed(game, exact_solver, exact_analysis, solver, payload, seed: int) -> dict[str, Any]:
    exact_policy = exact_solver.average_policy()
    corrected_policy = solver.average_policy()
    current_policy = {key: tuple(node.strategy()) for key, node in solver.nodes.items()}

    average_full = profile_exploitability(game, corrected_policy, corrected_policy)
    current_full = profile_exploitability(game, current_policy, current_policy)

    missing_filled = dict(exact_policy)
    missing_filled.update(corrected_policy)
    missing_full = profile_exploitability(game, missing_filled, missing_filled)

    sub100_filled = dict(exact_policy)
    for key, node in solver.nodes.items():
        if node.average_observations >= 100:
            sub100_filled[key] = corrected_policy[key]
    sub100_full = profile_exploitability(game, sub100_filled, sub100_filled)

    average = _compact_exploitability(average_full)
    current = _compact_exploitability(current_full)
    missing = _compact_exploitability(missing_full)
    sub100 = _compact_exploitability(sub100_full)
    exact_value = exact_analysis["profile_value_p0"]
    cross0 = expected_utility(game, corrected_policy, exact_policy)
    cross1 = expected_utility(game, exact_policy, corrected_policy)
    exact_keys = set(exact_solver.nodes)
    coverage = len(exact_keys & solver.nodes.keys()) / len(exact_keys)
    return {
        "seed": seed,
        "payload_hash": _canonical_hash(payload),
        "information_sets": len(solver.nodes),
        "exact_information_set_coverage": coverage,
        "observation_bands_over_exact_information_sets": _observation_bands(exact_keys, solver),
        "normalized_positive_regret": payload["normalized_positive_regret"],
        "illegal_actions": payload["illegal_actions"],
        "unstable_action_sets": payload["unstable_action_sets"],
        "average_policy": average,
        "current_regret_matched_policy": current,
        "missing_exact_substitution": {
            "policy": missing,
            "absolute_exploitability_change": abs(
                missing["exploitability"] - average["exploitability"]
            ),
        },
        "sub100_exact_substitution": {
            "policy": sub100,
            "absolute_exploitability_change": abs(
                sub100["exploitability"] - average["exploitability"]
            ),
        },
        "cross_play": {
            "corrected0_vs_exact1_value_p0": cross0,
            "exact0_vs_corrected1_value_p0": cross1,
            "corrected0_absolute_delta_from_exact_self": abs(cross0 - exact_value),
            "corrected1_absolute_delta_from_exact_self": abs(cross1 - exact_value),
        },
    }


def run_experiment(
    *,
    exact_iterations: int = 10_000,
    external_iterations: int = 50_000,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    output_dir: Path = Path("outputs/mccfr_average_policy_repair_046"),
    enforce_exact_hash: bool = True,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one external seed is required.")
    print("046 reconstructing exact CFR reference", flush=True)
    game = ReducedSplendorCFRGame()
    exact_solver = ExactCFRSolver(game)
    exact_payload = exact_solver.run(exact_iterations, snapshot_iterations=(100,))
    exact_hash = _canonical_hash(exact_payload)
    exact_analysis = _compact_exploitability(
        profile_exploitability(game, exact_solver.average_policy(), exact_solver.average_policy())
    )

    seed_results = []
    first_payload = None
    for index, seed in enumerate(seeds, start=1):
        print(f"046 corrected external MCCFR seed {seed} ({index}/{len(seeds)})", flush=True)
        solver = CorrectedExternalSamplingMCCFRSolver(game, seed)
        payload = solver.run(external_iterations, snapshot_iterations=(100,))
        if index == 1:
            first_payload = payload
        seed_results.append(
            _analyze_seed(game, exact_solver, exact_analysis, solver, payload, seed)
        )

    print(f"046 deterministic repeat of seed {seeds[0]}", flush=True)
    repeat_solver = CorrectedExternalSamplingMCCFRSolver(game, seeds[0])
    repeat_payload = repeat_solver.run(external_iterations, snapshot_iterations=(100,))
    assert first_payload is not None

    exploitabilities = [row["average_policy"]["exploitability"] for row in seed_results]
    exact_exploitability = exact_analysis["exploitability"]
    per_seed_integrity = all(
        row["illegal_actions"] == 0
        and row["unstable_action_sets"] == 0
        and row["exact_information_set_coverage"] >= 0.95
        and all(value < 0.05 for value in row["normalized_positive_regret"].values())
        and row["average_policy"]["best_response_integrity"]
        and _finite(row)
        for row in seed_results
    )
    gates = {
        "exact_payload_hash_matches_044": exact_hash == EXACT_HASH_044
        if enforce_exact_hash
        else True,
        "first_seed_byte_deterministic": first_payload == repeat_payload,
        "per_seed_integrity_regret_and_coverage": per_seed_integrity,
        "every_average_exploitability_at_most_0_020": all(
            value <= 0.020 for value in exploitabilities
        ),
        "median_average_exploitability_at_most_0_010": median(exploitabilities) <= 0.010,
        "every_excess_over_exact_at_most_0_015": all(
            value - exact_exploitability <= 0.015 for value in exploitabilities
        ),
        "every_player_deviation_incentive_at_most_0_040": all(
            row["average_policy"]["deviation_incentive_p0"] <= 0.040
            and row["average_policy"]["deviation_incentive_p1"] <= 0.040
            for row in seed_results
        ),
        "every_cross_play_delta_at_most_0_020": all(
            row["cross_play"]["corrected0_absolute_delta_from_exact_self"] <= 0.020
            and row["cross_play"]["corrected1_absolute_delta_from_exact_self"] <= 0.020
            for row in seed_results
        ),
        "missing_substitution_change_at_most_0_001": all(
            row["missing_exact_substitution"]["absolute_exploitability_change"] <= 0.001
            for row in seed_results
        ),
        "sub100_substitution_change_at_most_0_005": all(
            row["sub100_exact_substitution"]["absolute_exploitability_change"] <= 0.005
            for row in seed_results
        ),
    }
    passed = all(gates.values())
    result = {
        "experiment": "046_mccfr_average_policy_repair",
        "protocol": "docs/experiments/mccfr_average_policy_repair_046.md",
        "configuration": {
            "exact_iterations": exact_iterations,
            "external_iterations_per_seed": external_iterations,
            "external_seeds": list(seeds),
            "average_policy_semantics": "two_player_external_simple_opponent_node_v1",
        },
        "exact_reference": {
            "payload_hash": exact_hash,
            "expected_044_hash": EXACT_HASH_044,
            "strategic_analysis": exact_analysis,
        },
        "first_seed_repeat_hashes": [
            _canonical_hash(first_payload),
            _canonical_hash(repeat_payload),
        ],
        "seed_results": seed_results,
        "aggregate": {
            "min_exploitability": min(exploitabilities),
            "median_exploitability": median(exploitabilities),
            "max_exploitability": max(exploitabilities),
        },
        "gates": gates,
        "passed": passed,
        "decision": (
            "Authorize a separately preregistered full-game neural Deep CFR feasibility pilot."
            if passed
            else "Keep full-game neural Deep CFR and GPU rental locked; diagnose the failed repair gate."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"046 result: {'PASS' if passed else 'FAIL'} -> {output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-iterations", type=int, default=10_000)
    parser.add_argument("--external-iterations", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mccfr_average_policy_repair_046"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(
        exact_iterations=args.exact_iterations,
        external_iterations=args.external_iterations,
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
