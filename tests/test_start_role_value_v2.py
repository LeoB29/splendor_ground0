from __future__ import annotations

from dataclasses import asdict

import torch

from splendor_ai.bots.search_bot import CheckpointValueEvaluator
from splendor_ai.encoding import (
    canonical_public_belief_key,
    collate_canonical_state_batch,
    encode_canonical_state_entities,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    CanonicalEntityDistributionalValueModel,
    CanonicalEntityDistributionalValueV2Model,
    CanonicalEntityValueConfig,
)
from splendor_ai.training.run_start_role_value_v2 import (
    StartRoleBalancedDataset,
    _seat_search_gate,
)


def _config():
    return CanonicalEntityValueConfig(
        hidden_size=16,
        layers=1,
        attention_heads=1,
        attention_feedforward_size=16,
        id_embedding_size=4,
    )


def test_v1_entity_encoding_collides_on_start_role_but_public_key_does_not() -> None:
    first = SplendorEnv(seed=3).initial_state()
    second = first.copy_shallow()
    second.start_player = 1

    assert canonical_public_belief_key(first, 0) != canonical_public_belief_key(second, 0)
    assert encode_canonical_state_entities(first, 0) == encode_canonical_state_entities(second, 0)


def test_zero_initialized_v2_is_exactly_equal_to_v1() -> None:
    torch.manual_seed(4)
    v1 = CanonicalEntityDistributionalValueModel(_config(), margin_bins=41).eval()
    v2 = CanonicalEntityDistributionalValueV2Model(_config(), margin_bins=41).eval()
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert missing == ["start_role_embedding.weight"]
    assert unexpected == []
    state = SplendorEnv(seed=7).initial_state()
    batch = collate_canonical_state_batch([(state, 0), (state, 1)])
    with torch.no_grad():
        margin_v1, wdl_v1 = v1(**batch)
        margin_v2, wdl_v2 = v2(
            **batch,
            observer_is_start_player=torch.tensor([1, 0]),
        )
    torch.testing.assert_close(margin_v1, margin_v2)
    torch.testing.assert_close(wdl_v1, wdl_v2)


def test_v2_checkpoint_receives_explicit_start_role(tmp_path) -> None:
    model = CanonicalEntityDistributionalValueV2Model(_config(), margin_bins=41)
    with torch.no_grad():
        model.start_role_embedding.weight[1].fill_(1.0)
    path = tmp_path / "v2.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_reliability_blend_v2",
            "model_config": asdict(_config()),
            "margin_bins": 41,
            "phase_learned_weights": [1.0, 1.0, 1.0, 1.0],
            "wdl_confidence_floor": 0.0,
            "search_wdl_blend_weight": 0.5,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    first = SplendorEnv(seed=5).initial_state()
    second = first.copy_shallow()
    second.start_player = 1
    evaluator = CheckpointValueEvaluator(path, value_scale=1.0)

    assert evaluator.evaluate_state(first, 0) != evaluator.evaluate_state(second, 0)


def test_balanced_v2_dataset_derives_both_start_roles() -> None:
    rows = []
    for source in range(2):
        for opponent in range(4):
            for seat in range(2):
                for phase in range(4):
                    rows.append((source, opponent, seat, phase))
    count = len(rows)
    payload = {
        "entity_type_ids": torch.zeros(count, 1, dtype=torch.int16),
        "entity_card_ids": torch.zeros(count, 1, dtype=torch.int16),
        "entity_noble_ids": torch.zeros(count, 1, dtype=torch.int16),
        "entity_numeric_features": torch.zeros(count, 1, 34, dtype=torch.float16),
        "entity_mask": torch.ones(count, 1, dtype=torch.bool),
        "margin_class": torch.zeros(count, dtype=torch.int8),
        "wdl_class": torch.zeros(count, dtype=torch.int8),
        "source_id": torch.tensor([row[0] for row in rows], dtype=torch.int8),
        "opponent_id": torch.tensor([row[1] for row in rows], dtype=torch.int8),
        "actor_seat": torch.tensor([row[2] for row in rows], dtype=torch.int8),
        "phase_id": torch.tensor([row[3] for row in rows], dtype=torch.int8),
    }
    dataset = StartRoleBalancedDataset(payload)
    assert {int(dataset[index]["observer_is_start_player"]) for index in range(len(dataset))} == {0, 1}


def test_search_gate_requires_both_candidate_seats() -> None:
    result = {
        "gates": {"existing": True},
        "pair_rows": [
            {
                "candidate_seats": [0, 1],
                "games": [{"winner": 0}, {"winner": 0}],
            }
        ],
    }
    gated = _seat_search_gate(result)
    assert gated["seat_score_share"] == {"seat0": 1.0, "seat1": 0.0}
    assert not gated["passed"]
