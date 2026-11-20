from __future__ import annotations

import hashlib
from pathlib import Path

from splendor_ai.alphazero.start_role_audit import (
    PAIR_COUNT,
    PROTOCOL_SHA256,
    SHARD_COUNT,
    SIMULATIONS,
    _action_key,
    audit_decision,
    interpretation_gate,
    phase_for_turn,
    summarize_pairs,
)
from splendor_ai.encoding import (
    canonical_public_belief_key,
    encode_canonical_state_entities,
)
from splendor_ai.engine.env import SplendorEnv


def _rows(*, visit_tv: float, value_delta: float, top_rate_stride: int):
    rows = []
    for game in range(32):
        for local in range(2):
            index = game * 2 + local
            rows.append(
                {
                    "identity": {
                        "game_seed": 3_000_000 + game,
                        "turn_index": 50 + local,
                    },
                    "observer_is_start_player": local == 0,
                    "alias_checks": {
                        "public_key_changed": True,
                        "entity_tensor_identical": True,
                        "action_keys_identical": True,
                        "max_raw_logit_abs_diff": 0.0,
                        "raw_value_abs_diff": 0.0,
                    },
                    "visit_tv": visit_tv,
                    "top_action_changed": index % top_rate_stride == 0,
                    "signed_root_value_delta_original_minus_flipped": value_delta,
                    "abs_root_value_delta": abs(value_delta),
                    "original": {"simulations": 1024},
                    "flipped": {"simulations": 1024},
                }
            )
    return rows


def test_protocol_hash_and_frozen_schedule() -> None:
    path = Path("docs/engineering/alphazero_start_role_representation_audit_117.md")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert (PAIR_COUNT, SHARD_COUNT, SIMULATIONS) == (64, 4, 1024)


def test_phase_contract() -> None:
    assert phase_for_turn(0) == phase_for_turn(19) == "early"
    assert phase_for_turn(20) == phase_for_turn(39) == "mid"
    assert phase_for_turn(40) == "late"


def test_v1_tensor_aliases_start_role_while_public_key_and_actions_do_not() -> None:
    env = SplendorEnv(seed=117)
    original = env.initial_state()
    flipped = original.copy_shallow()
    flipped.start_player = 1
    assert canonical_public_belief_key(original, 0) != canonical_public_belief_key(flipped, 0)
    assert encode_canonical_state_entities(original, 0) == encode_canonical_state_entities(flipped, 0)
    assert {_action_key(action) for action in env.legal_actions(original)} == {
        _action_key(action) for action in env.legal_actions(flipped)
    }


def test_summary_and_gate_authorize_material_role_sensitivity() -> None:
    summary = summarize_pairs(_rows(visit_tv=0.08, value_delta=0.06, top_rate_stride=5))
    gate = interpretation_gate(summary)
    assert gate["integrity_passed"]
    assert gate["policy_role_sensitivity"]
    assert gate["value_role_sensitivity"]
    assert gate["start_role_repair_authorized_for_review"]
    assert "role_aware_pilot" in audit_decision(gate)


def test_gate_rejects_immaterial_teacher_response() -> None:
    summary = summarize_pairs(_rows(visit_tv=0.005, value_delta=0.005, top_rate_stride=20))
    gate = interpretation_gate(summary)
    assert gate["integrity_passed"]
    assert not gate["policy_role_sensitivity"]
    assert not gate["value_role_sensitivity"]
    assert not gate["start_role_repair_authorized_for_review"]
    assert "multigeneration" in audit_decision(gate)
