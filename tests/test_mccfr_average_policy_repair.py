import json

from splendor_ai.cfr import (
    CorrectedExternalSamplingMCCFRSolver,
    ExternalSamplingMCCFRSolver,
    ReducedSplendorCFRGame,
    profile_exploitability,
)
from splendor_ai.eval.run_mccfr_average_policy_repair import run_experiment


def test_corrected_average_preserves_regret_updates_but_changes_averaging() -> None:
    game = ReducedSplendorCFRGame()
    legacy = ExternalSamplingMCCFRSolver(game, 440046)
    legacy.run(500, snapshot_iterations=(100,))
    corrected = CorrectedExternalSamplingMCCFRSolver(game, 440046)
    corrected.run(500, snapshot_iterations=(100,))

    assert legacy.nodes.keys() == corrected.nodes.keys()
    assert all(legacy.nodes[key].regrets == corrected.nodes[key].regrets for key in legacy.nodes)
    assert all(
        legacy.nodes[key].strategy() == corrected.nodes[key].strategy() for key in legacy.nodes
    )
    assert any(
        legacy.nodes[key].average_strategy() != corrected.average_policy()[key]
        for key in legacy.nodes
    )


def test_corrected_solver_is_seed_deterministic_and_uses_uniform_unobserved_fallback() -> None:
    game = ReducedSplendorCFRGame()
    first = CorrectedExternalSamplingMCCFRSolver(game, 440046)
    first_payload = first.run(200, snapshot_iterations=(100,))
    second = CorrectedExternalSamplingMCCFRSolver(game, 440046)
    second_payload = second.run(200, snapshot_iterations=(100,))

    assert first_payload == second_payload
    assert first_payload["average_policy_semantics"] == (
        "two_player_external_simple_opponent_node_v1"
    )
    for key, node in first.nodes.items():
        if node.average_observations == 0:
            assert first.average_policy()[key] == tuple(
                [1.0 / len(node.actions)] * len(node.actions)
            )


def test_corrected_average_reduces_known_legacy_exploitability() -> None:
    game = ReducedSplendorCFRGame()
    legacy = ExternalSamplingMCCFRSolver(game, 440046)
    legacy.run(1_000, snapshot_iterations=(100,))
    corrected = CorrectedExternalSamplingMCCFRSolver(game, 440046)
    corrected.run(1_000, snapshot_iterations=(100,))

    legacy_result = profile_exploitability(
        game, legacy.average_policy(), legacy.average_policy()
    )
    corrected_result = profile_exploitability(
        game, corrected.average_policy(), corrected.average_policy()
    )
    assert corrected_result["exploitability"] < legacy_result["exploitability"]


def test_046_runner_writes_deterministic_auditable_smoke_result(tmp_path) -> None:
    result = run_experiment(
        exact_iterations=100,
        external_iterations=200,
        seeds=(440046, 440047),
        output_dir=tmp_path,
        enforce_exact_hash=False,
    )

    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved == result
    assert result["first_seed_repeat_hashes"][0] == result["first_seed_repeat_hashes"][1]
    assert result["gates"]["first_seed_byte_deterministic"] is True
    assert result["configuration"]["average_policy_semantics"] == (
        "two_player_external_simple_opponent_node_v1"
    )
