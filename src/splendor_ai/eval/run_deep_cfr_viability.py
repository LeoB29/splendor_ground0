"""Run preregistered reduced-Splendor CFR viability experiment 044."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from splendor_ai.cfr.reduced_splendor import ReducedSplendorCFRGame
from splendor_ai.cfr.tabular import (
    ExactCFRSolver,
    ExternalSamplingMCCFRSolver,
    percentile,
    strategy_total_variation,
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _privacy_audit(game: ReducedSplendorCFRGame) -> dict[str, Any]:
    roots = [game.apply_chance(game.root(), outcome) for outcome in (0, 1)]
    root_keys = [game.information_state_key(node, 0) for node in roots]
    reserve_keys = [
        next(key for key in game.legal_action_keys(node) if key[0] == "RESERVE_DECK")
        for node in roots
    ]
    children = [game.apply_action(node, key) for node, key in zip(roots, reserve_keys, strict=True)]
    owner_keys = [game.information_state_key(node, 0) for node in children]
    opponent_keys = [game.information_state_key(node, 1) for node in children]
    histories = [node.public_history for node in children]
    checks = {
        "hidden_order_masked_at_root": root_keys[0] == root_keys[1],
        "blind_reserve_identity_visible_to_owner": owner_keys[0] != owner_keys[1],
        "blind_reserve_identity_masked_from_opponent": opponent_keys[0] == opponent_keys[1],
        "blind_reserve_public_history_masks_identity": histories[0] == histories[1],
    }
    return {"checks": checks, "passed": all(checks.values())}


def _node_map(solver: ExactCFRSolver | ExternalSamplingMCCFRSolver) -> dict[object, Any]:
    return solver.nodes


def _regret_decline_gate(payload: dict[str, Any]) -> dict[str, bool]:
    start = payload["regret_snapshots"]["100"]
    final = payload["normalized_positive_regret"]
    return {
        player: final[player] <= 0.25 * start[player]
        for player in ("0", "1")
    }


def run_experiment(
    *,
    exact_iterations: int = 10_000,
    external_iterations: int = 50_000,
    seed: int = 440_044,
    output_dir: Path = Path("outputs/deep_cfr_viability_044"),
) -> dict[str, Any]:
    if exact_iterations < 100 or external_iterations < 100:
        raise ValueError("Both runs require at least 100 iterations for the frozen snapshot.")

    game = ReducedSplendorCFRGame()
    mechanics = game.enumerate_tree_contract()
    privacy = _privacy_audit(game)

    print(f"044 exact CFR: {exact_iterations} iterations, deterministic run 1/2", flush=True)
    exact_solver = ExactCFRSolver(game)
    exact = exact_solver.run(exact_iterations, snapshot_iterations=(100,))
    print("044 exact CFR: deterministic run 2/2", flush=True)
    exact_replica_solver = ExactCFRSolver(game)
    exact_replica = exact_replica_solver.run(exact_iterations, snapshot_iterations=(100,))

    print(
        f"044 external-sampling MCCFR: {external_iterations} iterations, run 1/2",
        flush=True,
    )
    external_solver = ExternalSamplingMCCFRSolver(game, seed)
    external = external_solver.run(external_iterations, snapshot_iterations=(100,))
    print("044 external-sampling MCCFR: deterministic run 2/2", flush=True)
    external_replica_solver = ExternalSamplingMCCFRSolver(game, seed)
    external_replica = external_replica_solver.run(
        external_iterations,
        snapshot_iterations=(100,),
    )

    exact_nodes = _node_map(exact_solver)
    external_nodes = _node_map(external_solver)
    root_key = exact_solver.root_key()
    root_tv = strategy_total_variation(
        exact_nodes[root_key].average_strategy(),
        external_nodes[root_key].average_strategy(),
    )
    shared_observed = sorted(
        (
            key
            for key in exact_nodes.keys() & external_nodes.keys()
            if external_nodes[key].average_observations >= 100
        ),
        key=repr,
    )
    shared_tvs = [
        strategy_total_variation(
            exact_nodes[key].average_strategy(),
            external_nodes[key].average_strategy(),
        )
        for key in shared_observed
    ]
    coverage = len(exact_nodes.keys() & external_nodes.keys()) / max(len(exact_nodes), 1)
    comparison = {
        "root_strategy_total_variation": root_tv,
        "shared_information_sets_observed_at_least_100": len(shared_observed),
        "shared_mean_total_variation": sum(shared_tvs) / max(len(shared_tvs), 1),
        "shared_p90_total_variation": percentile(shared_tvs, 0.9),
        "external_coverage_of_exact_information_sets": coverage,
    }

    exact_decline = _regret_decline_gate(exact)
    gates = {
        "tree_finishes_at_four_actions": mechanics["max_player_actions"] == 4
        and mechanics["terminal_paths"] > 0,
        "privacy_contract": privacy["passed"],
        "exact_deterministic": exact == exact_replica,
        "exact_stable_and_legal": exact["illegal_actions"] == 0
        and exact["unstable_action_sets"] == 0,
        "exact_final_regret_below_0_02": all(
            value < 0.02 for value in exact["normalized_positive_regret"].values()
        ),
        "exact_regret_declines_75_percent": all(exact_decline.values()),
        "external_deterministic": external == external_replica,
        "external_stable_and_legal": external["illegal_actions"] == 0
        and external["unstable_action_sets"] == 0,
        "external_final_regret_below_0_05": all(
            value < 0.05 for value in external["normalized_positive_regret"].values()
        ),
        "root_strategy_tv_at_most_0_10": root_tv <= 0.10,
        "shared_mean_tv_at_most_0_15": comparison["shared_mean_total_variation"] <= 0.15,
        "shared_p90_tv_at_most_0_30": comparison["shared_p90_total_variation"] <= 0.30,
        "external_coverage_at_least_75_percent": coverage >= 0.75,
    }
    passed = all(gates.values())
    result = {
        "experiment": "044_reduced_splendor_cfr_viability",
        "protocol": "docs/experiments/deep_cfr_viability_044.md",
        "configuration": {
            "exact_iterations": exact_iterations,
            "external_iterations": external_iterations,
            "external_seed": seed,
        },
        "mechanics": mechanics,
        "privacy": privacy,
        "exact": exact,
        "external": external,
        "determinism_hashes": {
            "exact": [_canonical_hash(exact), _canonical_hash(exact_replica)],
            "external": [_canonical_hash(external), _canonical_hash(external_replica)],
        },
        "comparison": comparison,
        "exact_regret_decline_checks": exact_decline,
        "gates": gates,
        "passed": passed,
        "decision": (
            "Authorize a separately preregistered full-game neural Deep CFR pilot; "
            "do not promote a checkpoint."
            if passed
            else "Stop before GPU rental and diagnose the CFR/game interface."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"044 result: {'PASS' if passed else 'FAIL'} -> {output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-iterations", type=int, default=10_000)
    parser.add_argument("--external-iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=440_044)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/deep_cfr_viability_044"),
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
