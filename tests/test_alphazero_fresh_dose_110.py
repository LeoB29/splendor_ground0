from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from splendor_ai.alphazero.fresh_dose import (
    SHARD_SEED_STARTS,
    fresh_dose_decision,
    partition_seed_schedule,
    qualification_assessment,
)
from splendor_ai.alphazero.indexed_training import (
    IndexedReplayDataset,
    index_replay_paths,
    split_index_by_game,
)
from splendor_ai.alphazero.replay import AlphaZeroReplayRow, write_replay_jsonl


def _row(*, game_seed: int, turn_index: int) -> AlphaZeroReplayRow:
    return AlphaZeroReplayRow(
        generation=11,
        game_seed=game_seed,
        turn_index=turn_index,
        player_id=turn_index % 2,
        entity_type_ids=(1,),
        entity_card_ids=(0,),
        entity_noble_ids=(0,),
        entity_numeric_features=((0.0,) * 36,),
        action_type_ids=(1, 2),
        action_card_ids=(0, 0),
        action_noble_ids=(0, 0),
        action_numeric_features=((0.0,) * 25, (0.0,) * 25),
        action_payloads=({"action_type": "take_tokens"}, {"action_type": "pass"}),
        root_visits=(3, 1),
        policy_target=(0.75, 0.25),
        root_network_priors=(0.6, 0.4),
        root_search_priors=(0.6, 0.4),
        root_q_values=(0.1, -0.1),
        selected_action_row=0,
        raw_policy_top_row=0,
        search_changed_raw_action=False,
        visit_argmax_row=0,
        visit_argmax_changed_raw_action=False,
        raw_network_value=0.0,
        search_root_value=0.1,
        value_target=1.0 if turn_index % 2 == 0 else -1.0,
        winner=0,
        final_scores=(15, 10),
        termination_reason="completed",
        checkpoint_generation=11,
        checkpoint_path="generation_011.pt",
        search_config={"simulations": 256},
        search_telemetry={"simulations": 256},
    )


def test_frozen_seed_partition_is_nested_disjoint_and_exact() -> None:
    partition = partition_seed_schedule(
        [list(range(start, start + 256)) for start in SHARD_SEED_STARTS]
    )
    low = set(partition["low"])
    high = set(partition["high"])
    holdout = set(partition["holdout"])
    assert (len(low), len(high), len(holdout)) == (256, 896, 128)
    assert low < high
    assert high.isdisjoint(holdout)


def test_indexed_replay_roundtrip_and_whole_game_split(tmp_path: Path) -> None:
    rows = [_row(game_seed=seed, turn_index=turn) for seed in range(20, 25) for turn in range(2)]
    path = write_replay_jsonl(tmp_path / "replay.jsonl", rows)
    paths, entries = index_replay_paths([path])
    dataset = IndexedReplayDataset(paths, entries)
    try:
        assert [dataset[index] for index in range(len(dataset))] == rows
    finally:
        dataset.close()
    train, validation = split_index_by_game(entries, validation_fraction=0.2, seed=1100110)
    train_games = {(entry.generation, entry.game_seed) for entry in train}
    validation_games = {(entry.generation, entry.game_seed) for entry in validation}
    assert len(validation_games) == 1
    assert train_games.isdisjoint(validation_games)
    assert len(train) + len(validation) == len(rows)


def test_index_detects_row_mutation_after_sealing(tmp_path: Path) -> None:
    path = write_replay_jsonl(tmp_path / "replay.jsonl", [_row(game_seed=20, turn_index=0)])
    paths, entries = index_replay_paths([path])
    payload = asdict(_row(game_seed=20, turn_index=1))
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    dataset = IndexedReplayDataset(paths, entries)
    try:
        try:
            dataset[0]
        except ValueError as error:
            assert "identity changed" in str(error)
        else:
            raise AssertionError("Mutated indexed replay row was accepted.")
    finally:
        dataset.close()


def test_offline_qualification_requires_update_improvement_and_finiteness() -> None:
    parent = {"policy_loss": 2.0, "value_loss": 0.5, "objective_loss": 2.25}
    child = {"policy_loss": 1.9, "value_loss": 0.5, "objective_loss": 2.15}
    assert qualification_assessment(
        training={"selected_epoch": 2}, parent_metrics=parent, candidate_metrics=child
    )["qualified"]
    assert not qualification_assessment(
        training={"selected_epoch": 0}, parent_metrics=parent, candidate_metrics=child
    )["qualified"]
    assert not qualification_assessment(
        training={"selected_epoch": 2},
        parent_metrics=parent,
        candidate_metrics={**child, "objective_loss": float("nan")},
    )["qualified"]


def test_decision_requires_high_dose_internal_pass_and_specificity() -> None:
    passed_high = {"passed": True, "score_share": 0.58}
    passed_low = {"passed": True, "score_share": 0.54}
    assert "review_next_generation" in fresh_dose_decision(
        high_qualified=True, high_assessment=passed_high, low_assessment=passed_low
    )
    assert "not_dose_specific" in fresh_dose_decision(
        high_qualified=True,
        high_assessment={"passed": True, "score_share": 0.53},
        low_assessment=passed_low,
    )
    assert "failed_offline_gate" in fresh_dose_decision(
        high_qualified=False, high_assessment=None, low_assessment=None
    )
