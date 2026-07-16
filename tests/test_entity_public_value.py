from __future__ import annotations

import json

import pytest
import torch

from splendor_ai.encoding import collate_canonical_state_batch
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    CanonicalEntityValueConfig,
    CanonicalEntityValueModel,
)
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_entity_public_value import (
    BalancedValueStrataDataset,
    PROFILES,
    PublicValueConfig,
    PublicValueReplayDataset,
    _offline_gates,
    _ordering_summary,
    initialize_from_public_actor_encoder,
    phase_for_turn,
)


def _actor() -> CanonicalEntityActionScorerModel:
    return CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=16,
            action_hidden_size=16,
            scorer_hidden_size=16,
            entity_layers=1,
            action_layers=1,
            attention_heads=1,
            attention_feedforward_size=16,
            id_embedding_size=4,
        )
    )


def _value() -> CanonicalEntityValueModel:
    return CanonicalEntityValueModel(
        CanonicalEntityValueConfig(
            hidden_size=16,
            layers=1,
            attention_heads=1,
            attention_feedforward_size=16,
            id_embedding_size=4,
        )
    )


def _write_rows(tmp_path):
    path = tmp_path / "replays.jsonl"
    state = SplendorEnv(seed=5).initial_state()
    rows = []
    for profile_index, profile in enumerate(PROFILES):
        for seat in (0, 1):
            rows.append(
                {
                    "state_snapshot": serialize_state_snapshot(state),
                    "game_final_scores": [15, 10],
                    "rollout_profile": profile,
                    "player_id": seat,
                    "game_seed": 100 + 2 * profile_index + seat,
                    "turn_index": 5 + profile_index,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_public_value_model_ignores_hidden_deck_order() -> None:
    state_a = SplendorEnv(seed=4).initial_state()
    state_b = state_a.copy_shallow()
    state_b.hidden_tier_decks[2].reverse()
    model = _value().eval()

    with torch.no_grad():
        value_a = model(**collate_canonical_state_batch([(state_a, 0)]))
        value_b = model(**collate_canonical_state_batch([(state_b, 0)]))

    torch.testing.assert_close(value_a, value_b)


def test_public_value_dataset_targets_acting_player_margin_and_balances_strata(tmp_path) -> None:
    dataset = PublicValueReplayDataset(_write_rows(tmp_path), score_margin_scale=20.0)
    balanced = BalancedValueStrataDataset(dataset, range(len(dataset)))

    assert dataset[0].target == pytest.approx(0.25)
    assert dataset[1].target == pytest.approx(-0.25)
    assert len(balanced) == 8
    assert {balanced[index].profile for index in range(len(balanced))} == set(PROFILES)
    assert {balanced[index].actor_seat for index in range(len(balanced))} == {0, 1}


def test_public_value_encoder_initialization_excludes_value_head() -> None:
    torch.manual_seed(8)
    actor = _actor()
    value = _value()
    head_before = {name: tensor.clone() for name, tensor in value.value_head.state_dict().items()}

    transferred = initialize_from_public_actor_encoder(value, actor.state_dict())

    assert transferred
    assert all("value_head" not in name for name in transferred)
    for name, tensor in value.value_head.state_dict().items():
        torch.testing.assert_close(tensor, head_before[name])
    torch.testing.assert_close(
        value.entity_type_embedding.weight,
        actor.entity_type_embedding.weight,
    )


def test_ordering_summary_uses_game_phase_units() -> None:
    records = []
    for seed, target in enumerate((-0.5, -0.1, 0.2, 0.6), start=10):
        for turn in (2, 7):
            records.append(
                {
                    "game_seed": seed,
                    "profile": "parent",
                    "actor_seat": 0,
                    "phase": phase_for_turn(turn),
                    "target": target,
                    "candidate": target,
                    "heuristic": -target,
                }
            )

    candidate = _ordering_summary(records, "candidate")
    heuristic = _ordering_summary(records, "heuristic")

    assert candidate["concordance"] == 1.0
    assert heuristic["concordance"] == 0.0


def test_offline_gates_require_error_ordering_and_breakdown_retention(tmp_path) -> None:
    config = PublicValueConfig(
        source_dir=tmp_path,
        parent_checkpoint=tmp_path / "parent.pt",
        output_dir=tmp_path / "output",
    )
    candidate_breakdowns = {}
    heuristic_breakdowns = {}
    for label in [
        *(f"profile:{profile}" for profile in PROFILES),
        *(f"phase:{phase}" for phase in ("0-14", "15-29", "30-44", "45+")),
    ]:
        candidate_breakdowns[label] = {"mae": 0.20}
        heuristic_breakdowns[label] = {"mae": 0.30}
    result = {
        "candidate": {"macro_profile_seat_mae": 0.20},
        "heuristic": {"macro_profile_seat_mae": 0.30},
        "candidate_ordering": {
            "concordance": 0.65,
            "macro_stratum_correlation": 0.5,
        },
        "heuristic_ordering": {
            "concordance": 0.55,
            "macro_stratum_correlation": 0.3,
        },
        "candidate_breakdowns": candidate_breakdowns,
        "heuristic_breakdowns": heuristic_breakdowns,
        "paired_game_error": {"ci95_high": -0.01},
    }

    assert all(_offline_gates(config, result).values())
