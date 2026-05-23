from dataclasses import asdict

import torch

from splendor_ai.bots import CheckpointPolicyBot, LoopFallbackConfig
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import PolicyValueMLP, PolicyValueModelConfig


def _write_checkpoint(tmp_path, biased_action_indices: dict[int, float]):
    model = PolicyValueMLP(
        PolicyValueModelConfig(
            trunk_hidden_size=8,
            trunk_depth=1,
            dropout=0.0,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for action_index, bias in biased_action_indices.items():
            model.policy_head.bias[action_index] = bias

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _legal_token_take(legal_actions):
    return next(
        action
        for action in legal_actions
        if action.action_type == ActionType.TAKE_TOKENS
        and action.take_tokens == ("white", "blue", "green")
    )


def _state_with_legal_buy():
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[state.current_player]
    for card in state.visible_tier_cards[1][:2]:
        for color, cost in card.cost.items():
            player.tokens[color] = max(player.tokens[color], cost)
    return env, state, env.legal_actions(state)


def test_checkpoint_policy_bot_loads_and_chooses_masked_legal_action(tmp_path) -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    target_action = _legal_token_take(legal_actions)
    codec = ActionCodec()
    target_index = codec.encode(state, target_action)

    checkpoint_path = _write_checkpoint(tmp_path, {target_index: 5.0})

    bot = CheckpointPolicyBot(checkpoint_path, device="cpu")
    chosen_action = bot.choose_action(env, state, legal_actions)

    assert chosen_action == target_action


def test_loop_fallback_keeps_normal_argmax_before_loop_evidence(tmp_path) -> None:
    env, state, legal_actions = _state_with_legal_buy()
    codec = ActionCodec()
    take_action = _legal_token_take(legal_actions)
    buy_action = next(action for action in legal_actions if action.action_type == ActionType.BUY_VISIBLE)
    checkpoint_path = _write_checkpoint(
        tmp_path,
        {
            codec.encode(state, take_action): 5.0,
            codec.encode(state, buy_action): 4.5,
        },
    )

    bot = CheckpointPolicyBot(checkpoint_path, device="cpu")

    assert bot.choose_action(env, state, legal_actions) == take_action
    assert bot.loop_fallback_triggers == 0


def test_loop_fallback_uses_best_close_buy_after_repeated_state(tmp_path) -> None:
    env, state, legal_actions = _state_with_legal_buy()
    codec = ActionCodec()
    take_action = _legal_token_take(legal_actions)
    buy_actions = [action for action in legal_actions if action.action_type == ActionType.BUY_VISIBLE]
    lower_buy, better_buy = buy_actions[:2]
    checkpoint_path = _write_checkpoint(
        tmp_path,
        {
            codec.encode(state, take_action): 5.0,
            codec.encode(state, lower_buy): 4.0,
            codec.encode(state, better_buy): 4.5,
        },
    )

    bot = CheckpointPolicyBot(checkpoint_path, device="cpu")

    assert bot.choose_action(env, state, legal_actions) == take_action
    assert bot.choose_action(env, state, legal_actions) == better_buy
    assert bot.loop_fallback_triggers == 1


def test_loop_fallback_respects_buy_logit_gap(tmp_path) -> None:
    env, state, legal_actions = _state_with_legal_buy()
    codec = ActionCodec()
    take_action = _legal_token_take(legal_actions)
    buy_action = next(action for action in legal_actions if action.action_type == ActionType.BUY_VISIBLE)
    checkpoint_path = _write_checkpoint(
        tmp_path,
        {
            codec.encode(state, take_action): 20.0,
            codec.encode(state, buy_action): 1.0,
        },
    )

    bot = CheckpointPolicyBot(
        checkpoint_path,
        device="cpu",
        loop_fallback=LoopFallbackConfig(max_buy_logit_gap=8.0),
    )

    assert bot.choose_action(env, state, legal_actions) == take_action
    assert bot.choose_action(env, state, legal_actions) == take_action
    assert bot.loop_fallback_triggers == 0
