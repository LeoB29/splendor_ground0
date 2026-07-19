from __future__ import annotations

from collections import Counter

import pytest

from splendor_ai.cfr import (
    ResolvedLeaf,
    deterministic_belief_range,
    resolve_two_ply,
)
from splendor_ai.diagnostics import state_signature
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_bounded_public_belief_resolving import assigned_stratum


def test_resolver_shares_opponent_response_across_indistinguishable_histories() -> None:
    root = ("root",)
    info = ("same-opponent-information",)
    action_x = ("x",)
    action_y = ("y",)
    leaves = [
        ResolvedLeaf(0, root, info, action_x, 0.0, 1.0),
        ResolvedLeaf(0, root, info, action_y, 4.0, 3.0),
        ResolvedLeaf(1, root, info, action_x, 10.0, 9.0),
        ResolvedLeaf(1, root, info, action_y, 4.0, 3.0),
    ]

    resolution = resolve_two_ply(
        leaves,
        (root,),
        determinization_count=2,
        payoff="discovery_payoff",
    )

    # A hidden-state-aware opponent would choose x in det0 and y in det1 for
    # value 2.0. The legal shared response is y in both histories, value 4.0.
    assert resolution.action_values == pytest.approx((4.0,))
    assert resolution.opponent_choices[0]["selected_action"] == repr(action_y)


def test_resolver_rejects_unstable_actions_inside_opponent_information_set() -> None:
    root = ("root",)
    info = ("same",)
    leaves = [
        ResolvedLeaf(0, root, info, ("x",), 0.0, 0.0),
        ResolvedLeaf(1, root, info, ("y",), 0.0, 0.0),
    ]

    with pytest.raises(ValueError, match="unstable semantic actions"):
        resolve_two_ply(leaves, (root,), 2, payoff="discovery_payoff")


def test_belief_range_ignores_incidental_hidden_deck_order() -> None:
    state = SplendorEnv(seed=49).initial_state()
    permuted = state.copy_shallow()
    for tier in (1, 2, 3):
        permuted.hidden_tier_decks[tier].reverse()

    first = deterministic_belief_range(state, 0, root_index=7)
    second = deterministic_belief_range(permuted, 0, root_index=7)

    assert [state_signature(item) for item in first] == [state_signature(item) for item in second]


def test_source_assignment_is_exactly_balanced_before_collection() -> None:
    counts = Counter(assigned_stratum(index) for index in range(192))

    assert len(counts) == 6
    assert set(counts.values()) == {32}
