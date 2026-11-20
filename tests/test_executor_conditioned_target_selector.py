from __future__ import annotations

from pathlib import Path

import torch

from splendor_ai.bots import ExecutorConditionedTargetBot
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_executor_conditioned_target_selector import (
    ExecutorConditionedExperimentConfig,
    _validate_experiment_config,
)
from splendor_ai.training.card_option_q import enumerate_option_targets
from splendor_ai.training.executor_conditioned_target import (
    ExecutorConditionedTargetConfig,
    ExecutorConditionedTargetModel,
    ExecutorTargetCandidate,
    collate_executor_target_batch,
    load_executor_conditioned_model,
    reconstruct_executor_candidates,
    save_executor_conditioned_checkpoint,
)
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_executor_conditioned_target_corpus import (
    ExecutorConditionedCorpusConfig,
    _validate_corpus_config,
)
from splendor_ai.training.run_executor_conditioned_target_training import (
    ExecutorConditionedDataset,
    ExecutorConditionedTrainingConfig,
    _validate_training_config,
    collate_executor_conditioned_samples,
)


RAW035 = Path(
    "outputs/legal_action_scorer_035_legal_observation/"
    "legal_action_attention_scorer_best.pt"
)


def _small_model() -> ExecutorConditionedTargetModel:
    torch.manual_seed(83)
    return ExecutorConditionedTargetModel(
        ExecutorConditionedTargetConfig(
            entity_hidden_size=32,
            item_hidden_size=32,
            scorer_hidden_size=48,
            entity_layers=1,
            attention_heads=4,
            attention_feedforward_size=64,
            id_embedding_size=8,
        )
    ).eval()


def _candidates(state):
    env = SplendorEnv()
    codec = ActionCodec()
    actions = list(
        actions_by_canonical_successor_key(state, env.legal_actions(state)).values()
    )
    return [
        ExecutorTargetCandidate(target, actions[index % len(actions)], codec.encode(state, actions[index % len(actions)]))
        for index, target in enumerate(enumerate_option_targets(state, state.current_player))
    ]


def test_clear_advantage_is_exactly_zero() -> None:
    state = SplendorEnv(seed=783_000).initial_state()
    candidates = _candidates(state)
    scores = _small_model()(**collate_executor_target_batch([(state, candidates)]))[0]
    assert scores.shape[0] == len(candidates)
    assert scores[0].item() == 0.0


def test_nonclear_candidate_order_is_equivariant() -> None:
    state = SplendorEnv(seed=783_001).initial_state()
    candidates = _candidates(state)
    reordered = [candidates[0], *reversed(candidates[1:])]
    model = _small_model()
    first = model(**collate_executor_target_batch([(state, candidates)]))[0]
    second = model(**collate_executor_target_batch([(state, reordered)]))[0]
    assert second[0].item() == 0.0
    assert torch.allclose(first[1:], torch.flip(second[1:], dims=(0,)), atol=1e-6)


def test_model_does_not_observe_hidden_deck_order() -> None:
    import random

    state = SplendorEnv(seed=783_002).initial_state()
    changed = determinize_for_observer(state, 0, random.Random(83))
    first_candidates = _candidates(state)
    second_candidates = _candidates(changed)
    model = _small_model()
    first = model(**collate_executor_target_batch([(state, first_candidates)]))[0]
    second = model(**collate_executor_target_batch([(changed, second_candidates)]))[0]
    assert [(c.target.key, c.action_index) for c in first_candidates] == [
        (c.target.key, c.action_index) for c in second_candidates
    ]
    assert torch.allclose(first, second, atol=1e-6)


def test_executor_conditioned_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _small_model()
    path = tmp_path / "executor.pt"
    save_executor_conditioned_checkpoint(path, model, {"best_epoch": 4})
    loaded = load_executor_conditioned_model(path)
    assert loaded.config == model.config
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_reconstruction_and_dataset_preserve_executor_action() -> None:
    state = SplendorEnv(seed=783_003).initial_state()
    candidates = _candidates(state)
    rows = [
        {
            "target_key": candidate.target.key,
            "target_source": candidate.target.source,
            "executor_action_index": candidate.action_index,
            "full_mean": index / len(candidates),
            "baseline": index == 0,
        }
        for index, candidate in enumerate(candidates)
    ]
    reconstructed = reconstruct_executor_candidates(
        state, rows, SplendorEnv().legal_actions(state)
    )
    assert [candidate.action_index for candidate in reconstructed] == [
        candidate.action_index for candidate in candidates
    ]
    sample = ExecutorConditionedDataset(
        [{"state_snapshot": serialize_state_snapshot(state), "targets": rows}]
    )[0]
    batch = collate_executor_conditioned_samples([sample])
    assert batch["label_advantages"][0, 0].item() == 0.0
    assert batch["executor_action_indices"][0, 1].item() == candidates[1].action_index


def test_bot_can_select_executor_conditioned_real_target(tmp_path: Path) -> None:
    path = tmp_path / "executor.pt"
    save_executor_conditioned_checkpoint(path, _small_model(), {})
    env = SplendorEnv(seed=783_004)
    state = env.initial_state()
    bot = ExecutorConditionedTargetBot(path, RAW035)

    def force_second(_state, candidates):
        scores = torch.zeros(len(candidates))
        scores[1] = 1.0
        return scores

    bot.score_candidates = force_second  # type: ignore[method-assign]
    actions = env.legal_actions(state)
    chosen = bot.choose_action(env, state, actions)
    targets = enumerate_option_targets(state, 0)
    assert chosen in actions
    assert bot.current_target_id == targets[1].key
    assert bot.telemetry.target_selections == 1


def test_frozen_083_contracts_reject_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "raw.pt"
    checkpoint.write_bytes(b"x")
    _validate_experiment_config(
        ExecutorConditionedExperimentConfig(
            checkpoint, tmp_path / "historical.json", tmp_path / "out"
        )
    )
    try:
        _validate_experiment_config(
            ExecutorConditionedExperimentConfig(
                checkpoint,
                tmp_path / "historical.json",
                tmp_path / "out",
                strength_seed_start=792_001,
            )
        )
    except ValueError as error:
        assert "strength" in str(error)
    else:
        raise AssertionError("Changed strength seed should fail.")
    _validate_corpus_config(
        ExecutorConditionedCorpusConfig(
            checkpoint, tmp_path / "fresh.json", tmp_path / "progress.jsonl"
        )
    )
    try:
        _validate_corpus_config(
            ExecutorConditionedCorpusConfig(
                checkpoint,
                tmp_path / "fresh.json",
                tmp_path / "progress.jsonl",
                source_games=599,
            )
        )
    except ValueError as error:
        assert "source" in str(error)
    else:
        raise AssertionError("Changed corpus source count should fail.")
    _validate_training_config(
        ExecutorConditionedTrainingConfig(tmp_path / "historical.json", tmp_path / "out")
    )
    try:
        _validate_training_config(
            ExecutorConditionedTrainingConfig(
                tmp_path / "historical.json", tmp_path / "out", epochs=49
            )
        )
    except ValueError as error:
        assert "epoch" in str(error)
    else:
        raise AssertionError("Changed training epochs should fail.")
