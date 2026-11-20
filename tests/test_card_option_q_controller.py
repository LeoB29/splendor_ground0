from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from splendor_ai.bots import CardOptionQBot
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.card_option_q import (
    CardOptionQConfig,
    CardOptionQModel,
    collate_card_option_batch,
    enumerate_option_action_pairs,
    enumerate_option_targets,
    load_card_option_model,
    save_card_option_checkpoint,
)
from splendor_ai.training.run_card_option_q_corpus import (
    CardOptionCorpusConfig,
    _target_executor_action,
)
from splendor_ai.eval.run_card_option_q_controller import (
    CardOptionExperimentConfig,
    _game_payload,
    _validate_experiment_config,
)
from splendor_ai.eval.match import GameResult


def _small_model() -> CardOptionQModel:
    torch.manual_seed(81)
    return CardOptionQModel(
        CardOptionQConfig(
            entity_hidden_size=32,
            item_hidden_size=32,
            scorer_hidden_size=48,
            entity_layers=1,
            attention_heads=4,
            attention_feedforward_size=64,
            id_embedding_size=8,
        )
    ).eval()


def test_initial_target_action_cartesian_contract() -> None:
    env = SplendorEnv(seed=781_000)
    state = env.initial_state()
    targets = enumerate_option_targets(state, 0)
    pairs = enumerate_option_action_pairs(state, env.legal_actions(state))
    assert targets[0].key == "CLEAR"
    assert len(targets) == 13
    action_indices = {pair.action_index for pair in pairs}
    assert len(pairs) == len(targets) * len(action_indices)
    assert [target.key for target in targets[1:]] == sorted(target.key for target in targets[1:])


def test_pair_scorer_is_equivariant_to_pair_order() -> None:
    env = SplendorEnv(seed=781_001)
    state = env.initial_state()
    pairs = enumerate_option_action_pairs(state, env.legal_actions(state))
    model = _small_model()
    normal = model(**collate_card_option_batch([(state, pairs)]))[0]
    reversed_pairs = list(reversed(pairs))
    reversed_scores = model(**collate_card_option_batch([(state, reversed_pairs)]))[0]
    assert torch.allclose(normal, torch.flip(reversed_scores, dims=(0,)), atol=1e-6)


def test_pair_scorer_does_not_observe_hidden_deck_order() -> None:
    env = SplendorEnv(seed=781_002)
    state = env.initial_state()
    changed = determinize_for_observer(state, 0, __import__("random").Random(9))
    pairs = enumerate_option_action_pairs(state, env.legal_actions(state))
    changed_pairs = enumerate_option_action_pairs(changed, env.legal_actions(changed))
    model = _small_model()
    first = model(**collate_card_option_batch([(state, pairs)]))[0]
    second = model(**collate_card_option_batch([(changed, changed_pairs)]))[0]
    assert [(p.target.key, p.action_index) for p in pairs] == [
        (p.target.key, p.action_index) for p in changed_pairs
    ]
    assert torch.allclose(first, second, atol=1e-6)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _small_model()
    path = tmp_path / "option.pt"
    save_card_option_checkpoint(path, model, {"best_epoch": 2})
    loaded = load_card_option_model(path)
    assert loaded.config == model.config
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_target_executor_buys_target_when_affordable() -> None:
    env = SplendorEnv(seed=781_003)
    state = env.initial_state()
    target = state.visible_tier_cards[1][0]
    player = state.players[0]
    for color, amount in target.cost.items():
        player.tokens[color] = amount
    actions = env.legal_actions(state)
    scored = tuple(
        SimpleNamespace(action=action, score=float(index), action_index=index)
        for index, action in enumerate(actions)
    )
    chosen = _target_executor_action(state, actions, scored, target)
    assert chosen.action_type is ActionType.BUY_VISIBLE
    assert state.visible_tier_cards[chosen.tier][chosen.market_index].card_id == target.card_id


def test_bot_selects_a_legal_pair_and_records_latency(tmp_path: Path) -> None:
    model = _small_model()
    path = tmp_path / "option.pt"
    save_card_option_checkpoint(path, model, {})
    env = SplendorEnv(seed=781_004)
    state = env.initial_state()
    actions = env.legal_actions(state)
    bot = CardOptionQBot(path)
    chosen = bot.choose_action(env, state, actions)
    assert chosen in actions
    assert bot.telemetry.decisions == 1
    assert len(bot.telemetry.inference_latencies_ms) == 1


def test_mechanics_payload_hides_outcomes() -> None:
    game = GameResult(
        seed=1, turns=2, winner=0, final_scores=(15, 4), bot_seats=("a", "b"),
        stalled=False, timed_out=False, termination_reason="completed",
        repetition_count=0, no_progress_streak=0, loop_fallback_triggers_by_seat=(0, 0),
    )
    hidden = _game_payload(game, outcomes=False)
    shown = _game_payload(game, outcomes=True)
    assert "winner" not in hidden and "final_scores" not in hidden
    assert shown["winner"] == 0 and shown["final_scores"] == (15, 4)


def test_frozen_experiment_contract_rejects_seed_change(tmp_path: Path) -> None:
    checkpoint = tmp_path / "raw.pt"
    checkpoint.write_bytes(b"x")
    _validate_experiment_config(CardOptionExperimentConfig(checkpoint, tmp_path / "out"))
    bad = CardOptionExperimentConfig(
        checkpoint,
        tmp_path / "out",
        mechanics_seed_start=783_001,
    )
    try:
        _validate_experiment_config(bad)
    except ValueError as error:
        assert "mechanics" in str(error)
    else:
        raise AssertionError("Changed mechanics seed should fail the frozen contract.")


def test_corpus_contract_rejects_smaller_evidence_panel(tmp_path: Path) -> None:
    config = CardOptionCorpusConfig(
        checkpoint=tmp_path / "raw.pt",
        output_path=tmp_path / "corpus.json",
        progress_path=tmp_path / "progress.jsonl",
        selected_states=10,
        train_states=6,
        validation_states=2,
        confirmation_states=2,
    )
    from splendor_ai.training.run_card_option_q_corpus import _validate_config

    try:
        _validate_config(config)
    except ValueError as error:
        assert "1,000" in str(error)
    else:
        raise AssertionError("A reduced post-registration corpus must be rejected.")
