"""Policy/value checkpoint bot with legal-action masking."""

from __future__ import annotations

from pathlib import Path

import torch

from splendor_ai.encoding import ActionCodec, encode_public_observation_tensor
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import PolicyValueMLP, PolicyValueModelConfig, masked_policy_logits


class CheckpointPolicyBot:
    """Greedy policy bot backed by a saved supervised checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        checkpoint = torch.load(self._checkpoint_path, map_location=device, weights_only=True)
        config_payload = checkpoint.get("model_config", {})
        self._model = PolicyValueMLP(PolicyValueModelConfig(**config_payload))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()
        self._codec = ActionCodec()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        observation = encode_public_observation_tensor(state, state.current_player)
        observation_tensor = torch.tensor(
            observation.vector,
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)
        mask = torch.zeros(
            (1, self._codec.action_space_size),
            dtype=torch.bool,
            device=self._device,
        )
        action_to_index = {}
        for action in actions:
            index = self._codec.encode(state, action)
            mask[0, index] = True
            action_to_index[action] = index

        with torch.no_grad():
            policy_logits, _value = self._model(observation_tensor)
            masked_logits = masked_policy_logits(policy_logits, mask)
            best_index = int(torch.argmax(masked_logits, dim=1).item())

        for action, index in action_to_index.items():
            if index == best_index:
                return action
        raise RuntimeError("Best legal action index did not map back to a legal action.")
