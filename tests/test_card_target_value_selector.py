from __future__ import annotations

from pathlib import Path

import torch

from splendor_ai.bots import CardTargetValueBot, LegalActionAttentionScorerBot
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_card_target_value_selector import (
    CardTargetExperimentConfig,
    _validate_experiment_config,
)
from splendor_ai.training.card_option_q import enumerate_option_targets
from splendor_ai.training.card_target_value import (
    CardTargetValueConfig,
    CardTargetValueModel,
    collate_card_target_batch,
    load_card_target_model,
    save_card_target_checkpoint,
)
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_card_target_corpus import (
    CardTargetCorpusConfig,
    _desired_cells,
    _validate_config as validate_corpus_config,
)
from splendor_ai.training.run_card_target_training import (
    CardTargetDataset,
    collate_card_target_samples,
)


RAW035 = Path(
    "outputs/legal_action_scorer_035_legal_observation/"
    "legal_action_attention_scorer_best.pt"
)


def _small_model() -> CardTargetValueModel:
    torch.manual_seed(82)
    return CardTargetValueModel(
        CardTargetValueConfig(
            entity_hidden_size=32,
            target_hidden_size=32,
            scorer_hidden_size=48,
            entity_layers=1,
            attention_heads=4,
            attention_feedforward_size=64,
            id_embedding_size=8,
        )
    ).eval()


def test_target_scorer_is_equivariant_to_target_order() -> None:
    state = SplendorEnv(seed=782_000).initial_state()
    targets = enumerate_option_targets(state, 0)
    model = _small_model()
    normal = model(**collate_card_target_batch([(state, targets)]))[0]
    reversed_scores = model(
        **collate_card_target_batch([(state, list(reversed(targets)))])
    )[0]
    assert torch.allclose(normal, torch.flip(reversed_scores, dims=(0,)), atol=1e-6)


def test_target_scorer_does_not_observe_hidden_deck_order() -> None:
    import random

    state = SplendorEnv(seed=782_001).initial_state()
    changed = determinize_for_observer(state, 0, random.Random(82))
    targets = enumerate_option_targets(state, 0)
    changed_targets = enumerate_option_targets(changed, 0)
    model = _small_model()
    first = model(**collate_card_target_batch([(state, targets)]))[0]
    second = model(**collate_card_target_batch([(changed, changed_targets)]))[0]
    assert [target.key for target in targets] == [target.key for target in changed_targets]
    assert torch.allclose(first, second, atol=1e-6)


def test_target_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _small_model()
    path = tmp_path / "target.pt"
    save_card_target_checkpoint(path, model, {"best_epoch": 3})
    loaded = load_card_target_model(path)
    assert loaded.config == model.config
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_target_dataset_and_collation_contract() -> None:
    state = SplendorEnv(seed=782_002).initial_state()
    targets = enumerate_option_targets(state, 0)
    row = {
        "state_snapshot": serialize_state_snapshot(state),
        "targets": [
            {
                "target_key": target.key,
                "target_source": target.source,
                "executor_action_index": 100 + index,
                "full_mean": index / len(targets),
                "baseline": index == 0,
            }
            for index, target in enumerate(targets)
        ],
    }
    sample = CardTargetDataset([row])[0]
    batch = collate_card_target_samples([sample])
    assert batch["target_mask"].sum().item() == len(targets)
    assert batch["baseline_indices"].item() == 0
    assert batch["executor_action_indices"][0, 1].item() == 101


def test_public_raw_score_method_matches_greedy_choice() -> None:
    env = SplendorEnv(seed=782_003)
    state = env.initial_state()
    actions = env.legal_actions(state)
    bot = LegalActionAttentionScorerBot(RAW035)
    values = bot.score_actions(state, actions)
    assert bot.choose_action(env, state, actions) == actions[max(range(len(values)), key=values.__getitem__)]


def test_bot_can_select_and_execute_a_real_target(tmp_path: Path) -> None:
    path = tmp_path / "target.pt"
    save_card_target_checkpoint(path, _small_model(), {})
    env = SplendorEnv(seed=782_004)
    state = env.initial_state()
    targets = enumerate_option_targets(state, 0)
    bot = CardTargetValueBot(path, RAW035)
    forced = torch.zeros(len(targets))
    forced[1] = 1.0
    bot.score_targets = lambda _state: (targets, forced)  # type: ignore[method-assign]
    actions = env.legal_actions(state)
    chosen = bot.choose_action(env, state, actions)
    assert chosen in actions
    assert bot.current_target_id == targets[1].key
    assert bot.telemetry.target_selections == 1
    assert bot.telemetry.target_decisions == 1


def test_fresh_corpus_cells_are_exactly_balanced() -> None:
    cells = _desired_cells()
    assert sum(cells.values()) == 400
    for split in ("validation", "confirmation"):
        assert sum(count for (name, _phase, _seat), count in cells.items() if name == split) == 200
        assert sum(count for (name, _phase, seat), count in cells.items() if name == split and seat == 0) == 100
        assert sum(count for (name, _phase, seat), count in cells.items() if name == split and seat == 1) == 100


def test_frozen_082_contracts_reject_seed_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "raw.pt"
    checkpoint.write_bytes(b"x")
    _validate_experiment_config(
        CardTargetExperimentConfig(checkpoint, tmp_path / "old.json", tmp_path / "out")
    )
    bad_experiment = CardTargetExperimentConfig(
        checkpoint,
        tmp_path / "old.json",
        tmp_path / "out",
        mechanics_seed_start=787_001,
    )
    try:
        _validate_experiment_config(bad_experiment)
    except ValueError as error:
        assert "mechanics" in str(error)
    else:
        raise AssertionError("Changed mechanics seed should fail the frozen contract.")
    valid_corpus = CardTargetCorpusConfig(
        checkpoint, tmp_path / "fresh.json", tmp_path / "progress.jsonl"
    )
    validate_corpus_config(valid_corpus)
    bad_corpus = CardTargetCorpusConfig(
        checkpoint,
        tmp_path / "fresh.json",
        tmp_path / "progress.jsonl",
        source_games=599,
    )
    try:
        validate_corpus_config(bad_corpus)
    except ValueError as error:
        assert "source" in str(error)
    else:
        raise AssertionError("Changed fresh source count should fail the frozen contract.")
