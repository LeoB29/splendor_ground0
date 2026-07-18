import json

from splendor_ai.cfr import (
    ExactCFRSolver,
    ExternalSamplingMCCFRSolver,
    ReducedSplendorCFRGame,
)
from splendor_ai.eval.run_deep_cfr_viability import run_experiment


def test_reduced_game_exhaustive_contract() -> None:
    contract = ReducedSplendorCFRGame().enumerate_tree_contract()

    assert contract == {
        "nodes": 985,
        "terminal_paths": 750,
        "information_sets": 206,
        "max_player_actions": 4,
    }


def test_exact_cfr_is_deterministic_and_legal() -> None:
    game = ReducedSplendorCFRGame()
    first = ExactCFRSolver(game).run(100, snapshot_iterations=(100,))
    second = ExactCFRSolver(game).run(100, snapshot_iterations=(100,))

    assert first == second
    assert first["information_sets"] == 206
    assert first["illegal_actions"] == 0
    assert first["unstable_action_sets"] == 0


def test_external_sampling_is_seed_deterministic_and_legal() -> None:
    game = ReducedSplendorCFRGame()
    first = ExternalSamplingMCCFRSolver(game, 440044).run(200, snapshot_iterations=(100,))
    second = ExternalSamplingMCCFRSolver(game, 440044).run(200, snapshot_iterations=(100,))

    assert first == second
    assert first["illegal_actions"] == 0
    assert first["unstable_action_sets"] == 0


def test_viability_runner_writes_auditable_result(tmp_path) -> None:
    result = run_experiment(
        exact_iterations=100,
        external_iterations=100,
        output_dir=tmp_path,
    )

    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved == result
    assert result["privacy"]["passed"] is True
    assert result["determinism_hashes"]["exact"][0] == result["determinism_hashes"]["exact"][1]
    assert result["determinism_hashes"]["external"][0] == result["determinism_hashes"]["external"][1]
