import json

from splendor_ai.cfr import (
    ExactCFRSolver,
    ExternalSamplingMCCFRSolver,
    ReducedSplendorCFRGame,
    discrepant_action_value_analysis,
    exact_best_response,
    profile_exploitability,
)
from splendor_ai.eval.run_cfr_strategic_equivalence import run_experiment


def _small_policies():
    game = ReducedSplendorCFRGame()
    exact = ExactCFRSolver(game)
    exact.run(100, snapshot_iterations=(100,))
    external = ExternalSamplingMCCFRSolver(game, 440044)
    external.run(200, snapshot_iterations=(100,))
    return game, exact, external


def test_best_response_is_deterministic_legal_and_information_set_consistent() -> None:
    game, exact, _external = _small_policies()
    policy = exact.average_policy()

    first = exact_best_response(game, player=0, opponent_policy=policy)
    second = exact_best_response(game, player=0, opponent_policy=policy)

    assert first.value == second.value
    assert first.payload == second.payload
    assert first.payload["selected_information_sets"] == first.payload["expected_information_sets"]
    assert first.payload["missing_selections"] == 0
    for selection in first.payload["selections"]:
        assert selection["selected_action"] in selection["actions"]
    for strategy in first.policy.values():
        assert sum(strategy) == 1.0
        assert set(strategy) <= {0.0, 1.0}


def test_profile_best_responses_dominate_the_profile_value() -> None:
    game, exact, _external = _small_policies()
    analysis = profile_exploitability(game, exact.average_policy(), exact.average_policy())

    assert analysis["deviation_incentive_p0"] >= -1e-12
    assert analysis["deviation_incentive_p1"] >= -1e-12
    assert analysis["nash_conv"] >= -1e-12
    assert analysis["exploitability"] == analysis["nash_conv"] / 2.0


def test_discrepancy_analysis_uses_reach_and_acting_player_value() -> None:
    game, exact, external = _small_policies()
    result = discrepant_action_value_analysis(
        game,
        exact_policy=exact.average_policy(),
        external_policy=external.average_policy(),
        exact_nodes=exact.nodes,
        external_nodes=external.nodes,
    )

    assert result["discrepancy_count"] == len(result["records"])
    for record in result["records"]:
        assert record["raw_total_variation"] > 0.30
        assert record["conservative_strategic_weight"] == max(
            record["exact_profile_reach"], record["external_profile_reach"]
        )
        assert record["reach_weighted_positive_loss"] == (
            record["conservative_strategic_weight"] * record["positive_external_loss"]
        )
        assert record["action_value_range"] >= 0.0


def test_045_runner_writes_deterministic_auditable_smoke_result(tmp_path) -> None:
    result = run_experiment(
        exact_iterations=100,
        external_iterations=200,
        output_dir=tmp_path,
        enforce_044_hashes=False,
    )

    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved == result
    assert result["analysis_hashes"][0] == result["analysis_hashes"][1]
    assert result["gates"]["analysis_deterministic"] is True
    assert result["gates"]["best_responses_information_set_consistent_and_legal"] is True
