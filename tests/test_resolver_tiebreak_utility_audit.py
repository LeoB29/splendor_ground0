import pytest

from splendor_ai.cfr import ResolvedLeaf, resolve_two_ply
from splendor_ai.eval.run_resolver_tiebreak_utility_audit import (
    clopper_pearson,
    resolve_two_ply_tolerant,
    select_outcome_blind_rows,
    summarize_branch_records,
)


def _leaf(root: str, action: str, value: float) -> ResolvedLeaf:
    return ResolvedLeaf(
        determinization=0,
        root_action=(root,),
        opponent_information_set=(root, "info"),
        opponent_action=(action,),
        discovery_payoff=value,
        evaluation_payoff=value,
    )


def test_tolerant_root_selection_restores_earlier_root_on_numerical_dust() -> None:
    roots = (("parent",), ("later",))
    leaves = [_leaf("parent", "reply", 0.0), _leaf("later", "reply", 1e-17)]
    exact = resolve_two_ply(leaves, roots, 1, payoff="discovery_payoff")
    tolerant = resolve_two_ply_tolerant(
        leaves, roots, 1, payoff="discovery_payoff", tolerance=1e-12
    )
    assert exact.best_index == 1
    assert tolerant.best_index == 0


def test_tolerant_opponent_selection_uses_semantic_order_within_tolerance() -> None:
    roots = (("root",),)
    leaves = [_leaf("root", "z", 0.0), _leaf("root", "a", 5e-13)]
    exact = resolve_two_ply(leaves, roots, 1, payoff="discovery_payoff")
    tolerant = resolve_two_ply_tolerant(
        leaves, roots, 1, payoff="discovery_payoff", tolerance=1e-12
    )
    assert exact.opponent_choices[0]["selected_action"] == repr(("z",))
    assert tolerant.opponent_choices[0]["selected_action"] == repr(("a",))


def test_outcome_blind_selection_is_balanced_and_uses_upper_middle_row() -> None:
    rows = []
    for game in range(6):
        for phase in ("early", "mid", "late"):
            for decision in (0, 2):
                rows.append(
                    {
                        "game_seed": 100 + game,
                        "phase": phase,
                        "decision_index": decision,
                        "turn_index": decision * 2,
                        "target_available": True,
                        "forbidden_outcome": "ignored",
                    }
                )
    selected = select_outcome_blind_rows(rows)
    assert [row["audit_assigned_phase"] for row in selected] == [
        "early",
        "mid",
        "late",
        "early",
        "mid",
        "late",
    ]
    assert all(row["decision_index"] == 2 for row in selected)


def test_branch_summary_distinguishes_tiebreak_results_from_true_draws() -> None:
    base = {
        "terminal": True,
        "resolved": True,
        "hard_cap_reached": False,
        "loop_adjudicated": False,
        "fallback_triggers": 0,
        "score_margin": 0,
        "original_value": 0.0,
    }
    records = [
        {**base, "winner": 0, "tiebreak_value": 0.025},
        {**base, "winner": 1, "tiebreak_value": -0.025},
        {**base, "winner": None, "tiebreak_value": 0.0},
        {
            **base,
            "score_margin": 1,
            "winner": 0,
            "original_value": 0.05,
            "tiebreak_value": 0.05,
        },
    ]
    summary = summarize_branch_records(records, actor_seat=0)
    assert summary["equal_prestige_terminal"] == 3
    assert summary["equal_prestige_root_wins"] == 1
    assert summary["equal_prestige_opponent_wins"] == 1
    assert summary["equal_prestige_true_draws"] == 1
    assert summary["utility_changed_branches"] == 2
    assert summary["nonzero_margin_values_identical"]


def test_clopper_pearson_handles_boundaries() -> None:
    low, high = clopper_pearson(0, 120)
    assert low == 0.0
    assert high == pytest.approx(0.03027, abs=1e-5)
    low, high = clopper_pearson(120, 120)
    assert low == pytest.approx(0.96973, abs=1e-5)
    assert high == 1.0
