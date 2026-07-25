from __future__ import annotations

from collections import Counter

import pytest

from splendor_ai.cfr import ResolvedLeaf
from splendor_ai.eval.run_early_resolver_variance_decomposition import (
    EarlyVarianceConfig,
    _attribution_gates,
    _validate_config,
    belief_root_index,
    continuation_seed,
    select_early_states,
    smooth_min_root_values,
    two_way_components,
)


def test_058_selects_all_balanced_early_states() -> None:
    cfg = EarlyVarianceConfig()
    _validate_config(cfg)
    selected = select_early_states(cfg)

    assert len(selected) == 24
    assert all(row["phase"] == "early" for row in selected)
    counts = Counter(
        f"seat{row['actor_seat']}|{row['gap_stratum']}" for row in selected
    )
    assert len(counts) == 6
    assert set(counts.values()) == {4}
    assert len({row["game_seed"] for row in selected}) >= 12


def test_seed_contract_pairs_continuations_across_beliefs() -> None:
    cfg = EarlyVarianceConfig()

    assert belief_root_index(cfg, 7, 0) != belief_root_index(cfg, 7, 1)
    seed = continuation_seed(cfg, 7, 2, 1, 0)
    assert seed == (
        cfg.continuation_seed_base
        + 7_000_000_000
        + 2_000_000
        + 10_000
    )


def test_smooth_min_is_between_minimum_and_mean() -> None:
    root = ("root",)
    info = ("info",)
    actions = (("response0",), ("response1",))
    leaves = [
        ResolvedLeaf(
            determinization=det,
            root_action=root,
            opponent_information_set=info,
            opponent_action=action,
            discovery_payoff=value,
            evaluation_payoff=value,
        )
        for det in range(3)
        for action, value in zip(actions, (0.0, 0.4), strict=True)
    ]

    value = smooth_min_root_values(leaves, (root,), 3, 0.05)[0]

    assert 0.0 < value < 0.2


def test_smooth_min_preserves_equal_response_value() -> None:
    root = ("root",)
    info = ("info",)
    leaves = [
        ResolvedLeaf(
            determinization=det,
            root_action=root,
            opponent_information_set=info,
            opponent_action=(f"response{response}",),
            discovery_payoff=0.3,
            evaluation_payoff=0.3,
        )
        for det in range(3)
        for response in range(4)
    ]

    assert smooth_min_root_values(leaves, (root,), 3, 0.05) == pytest.approx(
        [0.3]
    )


def test_two_way_components_identify_pure_main_effects() -> None:
    belief = two_way_components(
        [[float(b)] * 4 for b in range(4)]
    )
    continuation = two_way_components(
        [[float(r) for r in range(4)] for _ in range(4)]
    )

    assert belief["belief_ss"] > 0.0
    assert belief["continuation_ss"] == pytest.approx(0.0)
    assert belief["interaction_ss"] == pytest.approx(0.0)
    assert continuation["belief_ss"] == pytest.approx(0.0)
    assert continuation["continuation_ss"] > 0.0
    assert continuation["interaction_ss"] == pytest.approx(0.0)


def _summary(
    *,
    belief_fraction: float,
    continuation_fraction: float,
    belief_regret: float,
    continuation_regret: float,
    switch_rate: float = 0.0,
    exact_gain: float = 0.0,
    regret_ratio: float = 1.0,
) -> dict[str, object]:
    return {
        "anova": {
            "hard": {
                "aggregate": {
                    "belief_fraction": belief_fraction,
                    "continuation_fraction": continuation_fraction,
                }
            }
        },
        "splits": {
            "hard": {
                "belief": {"mean_symmetric_cross_regret": belief_regret},
                "continuation": {
                    "mean_symmetric_cross_regret": continuation_regret
                },
            }
        },
        "smooth_minus_hard": {
            "average": {
                "exact_best_agreement_gain": exact_gain,
                "regret_ratio": regret_ratio,
            }
        },
        "opponent_switching": {
            "mean_modal_response_deviation": switch_rate
        },
    }


def test_attribution_gates_separate_three_primary_sources() -> None:
    cfg = EarlyVarianceConfig()

    belief = _attribution_gates(
        cfg,
        _summary(
            belief_fraction=0.60,
            continuation_fraction=0.20,
            belief_regret=0.05,
            continuation_regret=0.02,
        ),
    )
    assert all(belief["belief_primary"].values())
    assert not all(belief["continuation_primary"].values())

    continuation = _attribution_gates(
        cfg,
        _summary(
            belief_fraction=0.20,
            continuation_fraction=0.60,
            belief_regret=0.02,
            continuation_regret=0.05,
        ),
    )
    assert all(continuation["continuation_primary"].values())

    minimizer = _attribution_gates(
        cfg,
        _summary(
            belief_fraction=0.30,
            continuation_fraction=0.30,
            belief_regret=0.03,
            continuation_regret=0.03,
            switch_rate=0.25,
            exact_gain=0.12,
            regret_ratio=0.70,
        ),
    )
    assert all(minimizer["minimizer_primary"].values())
