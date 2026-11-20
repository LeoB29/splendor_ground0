"""Fixed-sample, score, RNG-order and prospective interpretation regressions."""
from copy import deepcopy
import pytest
from tools.splendorust_one_second_20261024 import statistics as stats


def games(outcome):
    rows = []
    for pair in range(100):
        for game in (0, 1):
            seat = (pair + game) % 2
            halfpoints = outcome(pair, game, seat)
            winner = None if halfpoints == 1 else seat if halfpoints == 2 else 1-seat
            rows.append({"pair_index": pair, "game_index": 2*pair + game, "search_seat": seat,
                         "winner": winner, "reason": "natural", "stage": "main",
                         "seed_decimal": stats.expected_seed("main", pair)})
    return rows


def test_symmetric_extreme_results_and_nondegenerate_sensitivity_bound():
    search = stats.analyze_games(games(lambda *args: 2), True)
    assert search["search_mean_score"] == 1 and search["numerator"] == 400
    assert search["paired_ci95"] == [1., 1.] and search["bootstrap_degenerate"]
    assert search["bounded_pair_hoeffding_95"][0] < 1
    assert search["classification"] == "useful_evidence_for_cheap_search"
    greedy = stats.analyze_games(games(lambda *args: 0), True)
    assert greedy["classification"] == "useful_evidence_for_greedy"
    assert greedy["bounded_pair_hoeffding_95"][1] > 0


def test_draws_half_credit_inconclusive_not_equivalence():
    result = stats.analyze_games(games(lambda *args: 1), True)
    assert result["draws"] == 200 and result["pair_numerators"] == [2] * 100
    assert result["numerator"] == 200 and result["search_mean_score"] == .5
    assert result["classification"] == "inconclusive_or_small"


def test_one_seat_weakness_blocks_promotion():
    result = stats.analyze_games(games(lambda p, g, seat: 2 if seat == 0 or p < 40 else 0), True)
    assert result["search_mean_score"] == .7 and result["paired_ci95"][0] > .5
    assert result["by_search_seat"] == {"0": 1., "1": .4}
    assert result["classification"] == "inconclusive_or_small"


def test_independent_rng_stream_integer_arithmetic_and_quantiles():
    rows = games(lambda p, g, s: (p + g) % 3)
    result = stats.analyze_games(rows, True)
    assert stats.verify_analysis(rows, result)["status"] == "passed"
    result["bootstrap"]["indices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stream"):
        stats.verify_analysis(rows, result)


def test_fixed_sample_schedule_and_gate_rejections():
    rows = games(lambda *args: 2)
    with pytest.raises(ValueError, match="gates"):
        stats.analyze_games(rows, False)
    with pytest.raises(ValueError, match="200"):
        stats.validated_games(rows[:-1])
    for field, value in [("game_index", True), ("search_seat", 1), ("stage", "operational"),
                         ("seed_decimal", "1"), ("winner", True), ("reason", "max_turns")]:
        bad = deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError):
            stats.validated_games(bad)
    with pytest.raises(ValueError, match="duplicate"):
        stats.validated_games(rows[:-1] + rows[:1])


def test_forfeit_is_loss_without_invented_natural_terminal():
    rows = games(lambda *args: 2)
    rows[0].update(reason="forfeit", loser=0, winner=1)
    assert stats.validated_games(rows)[0]["search_score_halfpoints"] == 0
    rows[0]["winner"] = 0
    with pytest.raises(ValueError, match="forfeit"):
        stats.validated_games(rows)


def test_real_prepared_schedule_uses_stage_global_game_indices():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    prepared = json.loads((root / "outputs/splendorust_roadmap_20261024_preparation_01/prospective_schedule.json").read_text())
    rows = [{"stage": g["stage"], "pair_index": g["pair_index"], "game_index": g["game_index"],
             "search_seat": g["puct16_seat"], "winner": None, "reason": "natural",
             "seed_decimal": g["seed_decimal"]} for g in prepared["games"] if g["stage"] == "main"]
    assert [g["game_index"] for g in stats.validated_games(rows)] == list(range(200))
    assert rows[-1]["pair_index"] == 99 and rows[-1]["game_index"] == 199
    rows[2]["game_index"] = 0
    with pytest.raises(ValueError, match="schedule"):
        stats.validated_games(rows)
