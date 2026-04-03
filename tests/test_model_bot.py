from dataclasses import asdict

import torch

from splendor_ai.bots import CheckpointPolicyBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.model import PolicyValueMLP, PolicyValueModelConfig


def test_checkpoint_policy_bot_loads_and_chooses_masked_legal_action(tmp_path) -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    target_action = next(
        action
        for action in legal_actions
        if action.action_type == ActionType.TAKE_TOKENS
        and action.take_tokens == ("white", "blue", "green")
    )
    codec = ActionCodec()
    target_index = codec.encode(state, target_action)

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
        model.policy_head.bias[target_index] = 5.0

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )

    bot = CheckpointPolicyBot(checkpoint_path, device="cpu")
    chosen_action = bot.choose_action(env, state, legal_actions)

    assert chosen_action == target_action
