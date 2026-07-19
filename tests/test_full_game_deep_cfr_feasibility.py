from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from splendor_ai.cfr.full_game import (
    DeepCFRSchemaRow,
    FullGameTraversalProbe,
    ReservoirMemory,
    build_schema_row,
    collate_deep_cfr_rows,
    encode_public_history,
    traversal_determinism_payload,
)
from splendor_ai.encoding.entity_encoding import ENTITY_HIDDEN_DECK_CARD
from splendor_ai.encoding.information_state import public_action_event
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_full_game_deep_cfr_feasibility import (
    run_privacy_and_invariance_audits,
)


def test_public_history_hides_blind_reserve_then_reveals_purchase() -> None:
    env = SplendorEnv(seed=470090)
    state = env.initial_state()
    reserve = next(
        action
        for action in env.legal_actions(state)
        if action.action_type == ActionType.RESERVE_DECK and action.tier == 1
    )
    reserve_event = public_action_event(state, reserve)
    after = env.step(state, reserve)
    after.current_player = 0
    buy_event = public_action_event(after, Action(ActionType.BUY_RESERVED, reserved_index=0))

    encoded = encode_public_history((reserve_event, buy_event), observer=1)

    assert reserve_event[4] is None
    assert encoded.card_ids[0] == 0
    assert buy_event[4] == after.players[0].reserved_cards[0].card_id
    assert encoded.card_ids[1] != 0


def test_schema_rows_collate_exact_variable_action_targets() -> None:
    env = SplendorEnv(seed=47)
    state = env.initial_state()
    row = build_schema_row(state, (), env.legal_actions(state))
    batch = collate_deep_cfr_rows([row, row])

    assert len(row.target) == len(row.actions.action_type_ids)
    assert sum(row.target) == pytest.approx(1.0)
    assert batch["targets"].shape == batch["action_mask"].shape
    assert batch["history_mask"].shape == (2, 1)
    assert not batch["history_mask"].any()
    assert torch.isfinite(batch["targets"]).all()


def test_schema_row_rejects_privileged_entity_type() -> None:
    env = SplendorEnv(seed=47)
    state = env.initial_state()
    row = build_schema_row(state, (), env.legal_actions(state))
    privileged_state = replace(
        row.state,
        entity_type_ids=(ENTITY_HIDDEN_DECK_CARD,) + row.state.entity_type_ids[1:],
    )

    with pytest.raises(ValueError, match="Privileged entities"):
        replace(row, state=privileged_state)


def test_algorithm_r_reservoir_is_bounded_and_reproducible() -> None:
    first = ReservoirMemory[int](capacity=8, seed=470047)
    second = ReservoirMemory[int](capacity=8, seed=470047)
    for value in range(100):
        first.add(value)
        second.add(value)

    assert first.seen == second.seen == 100
    assert len(first.rows) == len(second.rows) == 8
    assert first.payload() == second.payload()


def test_partial_full_game_traversal_emits_no_eligible_rows_and_repeats() -> None:
    first = FullGameTraversalProbe(root_seed=470100, traverser=0, node_budget=64).run()
    second = FullGameTraversalProbe(root_seed=470100, traverser=0, node_budget=64).run()

    assert not first["complete"]
    assert first["cutoff_reason"] == "node_budget"
    assert first["nodes"] == 64
    assert first["eligible_advantage_rows"] == 0
    assert first["eligible_strategy_rows"] == 0
    assert traversal_determinism_payload(first) == traversal_determinism_payload(second)


def test_frozen_privacy_and_permutation_audits_pass() -> None:
    payload = run_privacy_and_invariance_audits()

    assert payload["passed"], payload["checks"]
