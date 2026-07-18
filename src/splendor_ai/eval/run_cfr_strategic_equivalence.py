"""Run preregistered CFR strategic-equivalence diagnostic 045."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from splendor_ai.cfr import (
    ExactCFRSolver,
    ExternalSamplingMCCFRSolver,
    ReducedSplendorCFRGame,
    discrepant_action_value_analysis,
    profile_exploitability,
)
from splendor_ai.cfr.tabular import expected_utility


EXACT_HASH_044 = "b8184ec7c9dbfa3611d014f98c674efbeae27728429b2fa90c7939e1bbe043ab"
EXTERNAL_HASH_044 = "3c6644db39998800690a2adfa6efc8b64c7f69cd55be99d678ca2ea7ebadbf78"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _best_response_integrity(analysis: dict[str, Any]) -> bool:
    for profile_name in ("exact_profile", "external_profile"):
        for response in analysis[profile_name]["best_responses"].values():
            if response["selected_information_sets"] != response["expected_information_sets"]:
                return False
            if response["missing_selections"] != 0 or response["unstable_action_sets"] != 0:
                return False
            for selection in response["selections"]:
                if selection["selected_action"] not in selection["actions"]:
                    return False
    return True


def _zero_sum_and_nonnegative(analysis: dict[str, Any]) -> bool:
    for profile_name in ("exact_profile", "external_profile"):
        profile = analysis[profile_name]
        if abs(profile["profile_value_p0"] + profile["profile_value_p1"]) > 1e-12:
            return False
        if profile["deviation_incentive_p0"] < -1e-12:
            return False
        if profile["deviation_incentive_p1"] < -1e-12:
            return False
    return True


def run_experiment(
    *,
    exact_iterations: int = 10_000,
    external_iterations: int = 50_000,
    seed: int = 440_044,
    output_dir: Path = Path("outputs/cfr_strategic_equivalence_045"),
    enforce_044_hashes: bool = True,
) -> dict[str, Any]:
    print("045 reconstructing frozen exact CFR policy", flush=True)
    game = ReducedSplendorCFRGame()
    exact_solver = ExactCFRSolver(game)
    exact_payload = exact_solver.run(exact_iterations, snapshot_iterations=(100,))
    print("045 reconstructing frozen external MCCFR policy", flush=True)
    external_solver = ExternalSamplingMCCFRSolver(game, seed)
    external_payload = external_solver.run(external_iterations, snapshot_iterations=(100,))
    hashes = {
        "exact": _canonical_hash(exact_payload),
        "external": _canonical_hash(external_payload),
        "expected_exact_044": EXACT_HASH_044,
        "expected_external_044": EXTERNAL_HASH_044,
    }

    exact_policy = exact_solver.average_policy()
    external_policy = external_solver.average_policy()
    print("045 computing information-set-consistent best responses", flush=True)

    def analyze() -> dict[str, Any]:
        exact_profile = profile_exploitability(game, exact_policy, exact_policy)
        external_profile = profile_exploitability(game, external_policy, external_policy)
        action_values = discrepant_action_value_analysis(
            game,
            exact_policy=exact_policy,
            external_policy=external_policy,
            exact_nodes=exact_solver.nodes,
            external_nodes=external_solver.nodes,
        )
        exact_self_value = exact_profile["profile_value_p0"]
        external0_vs_exact1 = expected_utility(game, external_policy, exact_policy)
        exact0_vs_external1 = expected_utility(game, exact_policy, external_policy)
        return {
            "exact_profile": exact_profile,
            "external_profile": external_profile,
            "action_value_equivalence": action_values,
            "cross_play": {
                "exact_self_play_value_p0": exact_self_value,
                "external0_vs_exact1_value_p0": external0_vs_exact1,
                "exact0_vs_external1_value_p0": exact0_vs_external1,
                "external0_absolute_delta_from_exact_self": abs(
                    external0_vs_exact1 - exact_self_value
                ),
                "external1_absolute_delta_from_exact_self": abs(
                    exact0_vs_external1 - exact_self_value
                ),
            },
        }

    analysis = analyze()
    print("045 repeating strategic analysis for determinism", flush=True)
    analysis_replica = analyze()
    hashes_match = hashes["exact"] == EXACT_HASH_044 and hashes["external"] == EXTERNAL_HASH_044
    if not enforce_044_hashes:
        hashes_match = True
    external_exploitability = analysis["external_profile"]["exploitability"]
    exact_exploitability = analysis["exact_profile"]["exploitability"]
    weighted_loss = analysis["action_value_equivalence"][
        "reach_weighted_positive_loss_by_player"
    ]
    cross_play = analysis["cross_play"]
    gates = {
        "reconstructed_policy_hashes_match_044": hashes_match,
        "analysis_deterministic": analysis == analysis_replica,
        "best_responses_information_set_consistent_and_legal": _best_response_integrity(
            analysis
        ),
        "exact_exploitability_at_most_0_005": exact_exploitability <= 0.005,
        "external_exploitability_at_most_0_020": external_exploitability <= 0.020,
        "external_minus_exact_exploitability_at_most_0_015": (
            external_exploitability - exact_exploitability <= 0.015
        ),
        "all_high_tv_discrepancies_resolved": analysis["action_value_equivalence"][
            "all_resolved"
        ],
        "reach_weighted_loss_each_player_at_most_0_002": all(
            value <= 0.002 for value in weighted_loss.values()
        ),
        "cross_play_each_within_0_02_of_exact_self": (
            cross_play["external0_absolute_delta_from_exact_self"] <= 0.02
            and cross_play["external1_absolute_delta_from_exact_self"] <= 0.02
        ),
        "finite_nonnegative_and_zero_sum": _all_finite(analysis)
        and _zero_sum_and_nonnegative(analysis),
    }
    passed = all(gates.values())
    result = {
        "experiment": "045_cfr_strategic_equivalence",
        "protocol": "docs/experiments/cfr_strategic_equivalence_045.md",
        "configuration": {
            "exact_iterations": exact_iterations,
            "external_iterations": external_iterations,
            "external_seed": seed,
            "discrepancy_observation_floor": 100,
            "discrepancy_raw_tv_floor": 0.30,
            "unreachable_floor": 0.0001,
            "per_information_set_loss_ceiling": 0.02,
        },
        "policy_hashes": hashes,
        "analysis": analysis,
        "analysis_hashes": [_canonical_hash(analysis), _canonical_hash(analysis_replica)],
        "gates": gates,
        "passed": passed,
        "decision": (
            "Authorize a separately preregistered full-game neural Deep CFR feasibility pilot."
            if passed
            else "Keep full-game neural Deep CFR and GPU rental locked; diagnose the failed strategic gate."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"045 result: {'PASS' if passed else 'FAIL'} -> {output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-iterations", type=int, default=10_000)
    parser.add_argument("--external-iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=440_044)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cfr_strategic_equivalence_045"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(
        exact_iterations=args.exact_iterations,
        external_iterations=args.external_iterations,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
