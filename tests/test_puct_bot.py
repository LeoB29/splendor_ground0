from dataclasses import asdict
import random

import pytest
import torch

from splendor_ai.bots import (
    CheckpointLegalActionPriorScorer,
    CheckpointValueEvaluator,
    InformationSetPuctBot,
    PuctConfig,
)
from splendor_ai.bots.puct_bot import (
    _combine_n_step_leaf_value,
    _mix_root_priors,
    determinize_for_observer,
    information_state_key,
    legal_heuristic_value,
)
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import (
    Card,
    RESERVE_SOURCE_DECK,
)
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    ValueMLP,
    ValueModelConfig,
)


def _write_attention_checkpoint(tmp_path):
    model = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=16,
            state_depth=1,
            action_depth=1,
            attention_layers=1,
            attention_heads=1,
            attention_feedforward_size=16,
            dropout=0.0,
        )
    )
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    path = tmp_path / "attention.pt"
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        path,
    )
    return path


def _write_value_checkpoint(tmp_path):
    model = ValueMLP(ValueModelConfig(hidden_size=8, depth=1))
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    path = tmp_path / "value.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        path,
    )
    return path


def _write_canonical_entity_checkpoint(tmp_path):
    model = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(
            entity_hidden_size=16,
            action_hidden_size=16,
            scorer_hidden_size=16,
            entity_layers=1,
            action_layers=1,
            attention_heads=1,
            attention_feedforward_size=16,
            id_embedding_size=4,
            dropout=0.0,
        )
    )
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    path = tmp_path / "canonical_entity.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        path,
    )
    return path


def test_determinization_is_invariant_to_true_deck_order() -> None:
    env = SplendorEnv(seed=1)
    state_a = env.initial_state()
    private_card = state_a.hidden_tier_decks[1].pop(0)
    state_a.players[1].reserve_card(private_card, RESERVE_SOURCE_DECK)
    state_a.deck_counts[1] -= 1
    state_b = state_a.copy_shallow()
    state_b.hidden_tier_decks[1].reverse()

    determinized_a = determinize_for_observer(
        state_a,
        observer_id=0,
        rng=random.Random(9),
    )
    determinized_b = determinize_for_observer(
        state_b,
        observer_id=0,
        rng=random.Random(9),
    )

    assert [
        card.card_id for card in determinized_a.hidden_tier_decks[1]
    ] == [
        card.card_id for card in determinized_b.hidden_tier_decks[1]
    ]
    assert (
        determinized_a.players[1].reserved_cards[0].card_id
        == determinized_b.players[1].reserved_cards[0].card_id
    )


def test_information_key_hides_opponent_private_reserved_identity() -> None:
    env = SplendorEnv(seed=2)
    state_a = env.initial_state()
    state_b = env.initial_state()
    state_a.players[1].reserve_card(
        Card("private-a", 1, "white", 0, {"blue": 1}),
        RESERVE_SOURCE_DECK,
    )
    state_b.players[1].reserve_card(
        Card("private-b", 1, "black", 5, {"white": 7}),
        RESERVE_SOURCE_DECK,
    )
    codec = ActionCodec()

    key_a = information_state_key(
        state_a,
        0,
        env.legal_actions(state_a),
        codec,
    )
    key_b = information_state_key(
        state_b,
        0,
        env.legal_actions(state_b),
        codec,
    )

    assert key_a == key_b


def test_legal_heuristic_is_bounded() -> None:
    state = SplendorEnv(seed=3).initial_state()
    state.players[0].score = 100

    value = legal_heuristic_value(state, player_id=0, scale=4_000.0)

    assert 0.0 < value <= 1.0


def test_root_prior_mix_interpolates_only_the_root_distribution() -> None:
    assert _mix_root_priors([0.8, 0.2], 1.0) == pytest.approx([0.8, 0.2])
    assert _mix_root_priors([0.8, 0.2], 0.0) == pytest.approx([0.5, 0.5])
    assert _mix_root_priors([0.8, 0.2], 0.5) == pytest.approx([0.65, 0.35])


def test_puct_selects_immediate_winning_buy(tmp_path) -> None:
    env = SplendorEnv(seed=4)
    state = env.initial_state()
    state.start_player = 1
    state.players[0].score = 14
    state.visible_tier_cards[1][0] = Card(
        card_id="win-now",
        tier=1,
        bonus_color="white",
        points=1,
        cost={},
    )
    checkpoint = _write_attention_checkpoint(tmp_path)
    bot = InformationSetPuctBot(
        checkpoint,
        config=PuctConfig(
            simulations=96,
            c_puct=2.0,
            fpu_reduction=0.0,
            max_depth=8,
            seed=5,
        ),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert action.action_type == ActionType.BUY_VISIBLE
    assert action.market_index == 0
    assert bot.last_search_stats.simulations == 96
    assert bot.last_search_stats.selected_action_index is not None
    assert bot.last_search_stats.selected_action_index in bot.last_search_stats.root_edge_stats
    assert set(bot.last_search_stats.root_edge_stats) == {
        ActionCodec().encode(state, legal_action)
        for legal_action in env.legal_actions(state)
    }
    metadata = bot.last_action_metadata
    visits = metadata["root_visit_distribution"]
    assert visits
    assert sum(item["probability"] for item in visits) == pytest.approx(1.0)
    assert metadata["root_prior_effective_actions"] >= 1.0
    assert metadata["root_visit_effective_actions"] >= 1.0
    assert metadata["root_visited_actions"] >= 1


def test_puct_can_use_checkpoint_value_for_leaf_evaluation(tmp_path) -> None:
    env = SplendorEnv(seed=6)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(simulations=4, seed=7),
        value_checkpoint_path=_write_value_checkpoint(tmp_path),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert bot.last_search_stats.value_calls > 0


def test_puct_can_gate_checkpoint_value_by_turn(tmp_path) -> None:
    env = SplendorEnv(seed=18)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=19,
            value_min_turn_index=40,
        ),
        value_checkpoint_path=_write_value_checkpoint(tmp_path),
    )

    bot.choose_action(env, state, env.legal_actions(state))
    assert bot.last_search_stats.value_calls == 0

    state.turn_index = 40
    bot.choose_action(env, state, env.legal_actions(state))
    assert bot.last_search_stats.value_calls > 0


def test_puct_can_use_rollout_leaf_evaluation(tmp_path) -> None:
    env = SplendorEnv(seed=20)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=21,
            rollout_leaf_count=2,
            rollout_leaf_max_depth=1,
            rollout_leaf_policy="random",
        ),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert bot.last_search_stats.rollout_leaf_evaluations > 0
    assert bot.last_search_stats.value_calls == 0


def test_puct_can_use_n_step_leaf_evaluation(tmp_path) -> None:
    env = SplendorEnv(seed=24)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=25,
            n_step_leaf_depth=2,
            n_step_leaf_width=1,
            n_step_leaf_candidate_limit=4,
        ),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert bot.last_search_stats.n_step_leaf_evaluations > 0
    assert bot.last_search_stats.rollout_leaf_evaluations == 0
    assert bot.last_search_stats.value_calls == 0


def test_puct_accepts_blended_n_step_leaf_evaluation(tmp_path) -> None:
    env = SplendorEnv(seed=32)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=33,
            n_step_leaf_depth=1,
            n_step_leaf_weight=0.5,
        ),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert bot.last_search_stats.n_step_leaf_evaluations > 0


def test_n_step_leaf_delta_clip_preserves_only_bounded_correction() -> None:
    positive = _combine_n_step_leaf_value(
        heuristic_value=0.1,
        n_step_value=0.6,
        weight=1.0,
        delta_clip=0.2,
    )
    negative_blended = _combine_n_step_leaf_value(
        heuristic_value=0.1,
        n_step_value=-0.5,
        weight=0.5,
        delta_clip=0.2,
    )

    assert positive == pytest.approx((0.3, 0.5, 0.2))
    assert negative_blended == pytest.approx((0.0, -0.6, -0.2))


def test_n_step_leaf_takes_precedence_over_rollout_leaf(tmp_path) -> None:
    env = SplendorEnv(seed=26)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=27,
            n_step_leaf_depth=1,
            rollout_leaf_count=2,
            rollout_leaf_max_depth=1,
        ),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert bot.last_search_stats.n_step_leaf_evaluations > 0
    assert bot.last_search_stats.rollout_leaf_evaluations == 0


def test_puct_can_gate_n_step_leaf_by_turn(tmp_path) -> None:
    env = SplendorEnv(seed=28)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=29,
            n_step_leaf_depth=1,
            n_step_leaf_min_turn_index=40,
        ),
    )

    bot.choose_action(env, state, env.legal_actions(state))
    assert bot.last_search_stats.n_step_leaf_evaluations == 0

    state.turn_index = 40
    bot.choose_action(env, state, env.legal_actions(state))
    assert bot.last_search_stats.n_step_leaf_evaluations > 0


def test_puct_can_gate_n_step_leaf_by_score(tmp_path) -> None:
    env = SplendorEnv(seed=30)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=31,
            n_step_leaf_depth=1,
            n_step_leaf_min_max_score=10,
        ),
    )

    bot.choose_action(env, state, env.legal_actions(state))
    assert bot.last_search_stats.n_step_leaf_evaluations == 0

    state.players[0].score = 10
    bot.choose_action(env, state, env.legal_actions(state))
    assert bot.last_search_stats.n_step_leaf_evaluations > 0


def test_checkpoint_value_takes_precedence_over_rollout_leaf(tmp_path) -> None:
    env = SplendorEnv(seed=22)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=4,
            seed=23,
            n_step_leaf_depth=1,
            rollout_leaf_count=2,
            rollout_leaf_max_depth=1,
            rollout_leaf_policy="random",
        ),
        value_checkpoint_path=_write_value_checkpoint(tmp_path),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action is not None
    assert bot.last_search_stats.value_calls > 0
    assert bot.last_search_stats.n_step_leaf_evaluations == 0
    assert bot.last_search_stats.rollout_leaf_evaluations == 0


def test_checkpoint_wrappers_batch_matches_single_state_calls(tmp_path) -> None:
    env = SplendorEnv(seed=8)
    state_a = env.initial_state()
    state_b = env.step(state_a, env.legal_actions(state_a)[0])
    actions_a = env.legal_actions(state_a)
    actions_b = env.legal_actions(state_b)
    scorer = CheckpointLegalActionPriorScorer(
        _write_attention_checkpoint(tmp_path)
    )
    evaluator = CheckpointValueEvaluator(
        _write_value_checkpoint(tmp_path),
        value_scale=1.0,
    )

    score_batches = scorer.score_action_batches(
        [(state_a, actions_a), (state_b, actions_b)]
    )
    value_batch = evaluator.evaluate_states([(state_a, 0), (state_b, 1)])

    assert score_batches[0] == scorer.score_actions(state_a, actions_a)
    assert score_batches[1] == scorer.score_actions(state_b, actions_b)
    assert value_batch == [
        evaluator.evaluate_state(state_a, 0),
        evaluator.evaluate_state(state_b, 1),
    ]


def test_canonical_entity_prior_batches_match_single_state_calls(tmp_path) -> None:
    env = SplendorEnv(seed=34)
    state_a = env.initial_state()
    state_b = env.step(state_a, env.legal_actions(state_a)[0])
    actions_a = env.legal_actions(state_a)
    actions_b = env.legal_actions(state_b)
    scorer = CheckpointLegalActionPriorScorer(
        _write_canonical_entity_checkpoint(tmp_path)
    )

    score_batches = scorer.score_action_batches(
        [(state_a, actions_a), (state_b, actions_b)]
    )

    assert score_batches[0] == scorer.score_actions(state_a, actions_a)
    assert score_batches[1] == scorer.score_actions(state_b, actions_b)
    assert set(score_batches[0]) == set(actions_a)
    assert set(score_batches[1]) == set(actions_b)


def test_puct_accepts_canonical_entity_prior_checkpoint(tmp_path) -> None:
    env = SplendorEnv(seed=35)
    state = env.initial_state()
    bot = InformationSetPuctBot(
        _write_canonical_entity_checkpoint(tmp_path),
        config=PuctConfig(simulations=4, seed=36),
    )

    action = bot.choose_action(env, state, env.legal_actions(state))

    assert action in env.legal_actions(state)
    assert bot.last_search_stats.neural_calls > 0


def test_puct_reuses_information_state_tree_and_resets_for_new_game(
    tmp_path,
) -> None:
    env = SplendorEnv(seed=9)
    initial = env.initial_state()
    bot = InformationSetPuctBot(
        _write_attention_checkpoint(tmp_path),
        config=PuctConfig(
            simulations=96,
            inference_batch_size=8,
            reuse_tree=True,
            seed=10,
        ),
    )

    bot.choose_action(env, initial, env.legal_actions(initial))
    retained_after_first_move = len(bot._tree)
    reusable_state = None
    for root_action in env.legal_actions(initial):
        opponent_state = env.step(initial, root_action)
        for opponent_action in env.legal_actions(opponent_state):
            candidate = env.step(opponent_state, opponent_action)
            key = information_state_key(
                candidate,
                candidate.current_player,
                env.legal_actions(candidate),
                ActionCodec(),
            )
            if key in bot._tree:
                reusable_state = candidate
                break
        if reusable_state is not None:
            break

    assert retained_after_first_move > 0
    assert reusable_state is not None
    bot.choose_action(
        env,
        reusable_state,
        env.legal_actions(reusable_state),
    )
    assert bot.last_search_stats.root_reused

    fresh_game = SplendorEnv(seed=11)
    fresh_state = fresh_game.initial_state()
    bot.choose_action(
        fresh_game,
        fresh_state,
        fresh_game.legal_actions(fresh_state),
    )
    assert not bot.last_search_stats.root_reused
