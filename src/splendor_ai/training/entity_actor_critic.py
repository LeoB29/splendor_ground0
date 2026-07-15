"""Clean on-policy actor-critic training for canonical entity policies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.optim import AdamW

from splendor_ai.bots.base import Bot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import (
    actions_by_canonical_successor_key,
    collate_entity_action_batch,
    collate_privileged_state_batch,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
    PrivilegedEntityCritic,
    PrivilegedEntityCriticConfig,
)

OpponentFactory = Callable[[], Bot]


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    rollout_games: int = 10
    seed_start: int = 80_000
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    behavior_temperature: float = 1.0
    uniform_mix: float = 0.05
    gamma: float = 1.0
    gae_lambda: float = 0.95
    actor_epochs: int = 1
    critic_warmup_epochs: int = 3
    critic_epochs: int = 1
    batch_size: int = 64
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_ratio: float = 0.15
    entropy_weight: float = 0.01
    device: str = "cpu"


@dataclass(slots=True)
class ActorStep:
    state: SplendorState
    legal_actions: tuple[Action, ...]
    action_row: int
    behavior_probabilities: tuple[float, ...]
    behavior_log_probability: float
    behavior_entropy: float
    actor_seat: int
    game_index: int
    final_return: float = 0.0
    advantage: float = 0.0
    value_target: float = 0.0


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    games: int
    steps: int
    actor_wins: int
    opponent_wins: int
    draws: int
    timed_out_games: int
    stalled_games: int
    non_greedy_actions: int
    action_type_count: int
    minimum_behavior_probability: float
    mean_normalized_entropy: float
    opponents: dict[str, int]


@dataclass(frozen=True, slots=True)
class UpdateSummary:
    behavior_log_prob_max_error: float
    mean_ratio: float
    max_ratio: float
    mean_kl: float
    mean_normalized_entropy: float
    policy_loss: float
    critic_loss: float
    advantage_mean: float
    advantage_std: float


@dataclass(frozen=True, slots=True)
class ActorCriticArtifacts:
    actor: CanonicalEntityActionScorerModel
    critic: PrivilegedEntityCritic
    rollout: RolloutSummary
    update: UpdateSummary
    steps: tuple[ActorStep, ...] = field(repr=False)


def load_entity_actor(checkpoint_path: str | Path, device: str = "cpu") -> CanonicalEntityActionScorerModel:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    actor = CanonicalEntityActionScorerModel(
        CanonicalEntityActionScorerConfig(**checkpoint.get("model_config", {}))
    )
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.to(device).eval()
    return actor


def full_support_probabilities(
    scores: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    uniform_mix: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= uniform_mix < 1.0:
        raise ValueError("uniform_mix must be in [0, 1).")
    masked_scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked_scores / temperature, dim=1)
    legal_count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(probabilities.dtype)
    uniform = mask.to(probabilities.dtype) / legal_count
    return (1.0 - uniform_mix) * probabilities + uniform_mix * uniform


def collect_league_rollouts(
    actor: CanonicalEntityActionScorerModel,
    opponent_factories: list[tuple[str, OpponentFactory]],
    config: ActorCriticConfig,
) -> tuple[list[ActorStep], RolloutSummary]:
    if not opponent_factories:
        raise ValueError("At least one league opponent is required.")
    steps: list[ActorStep] = []
    actor_wins = opponent_wins = draws = timed_out = stalled = non_greedy = 0
    opponent_counts: dict[str, int] = {}
    action_types: set[str] = set()
    minimum_probability = 1.0
    entropy_values: list[float] = []
    actor.eval()
    for game_index in range(config.rollout_games):
        label, factory = opponent_factories[game_index % len(opponent_factories)]
        opponent_counts[label] = opponent_counts.get(label, 0) + 1
        opponent = factory()
        actor_seat = game_index % 2
        env = SplendorEnv(seed=config.seed_start + game_index)
        state = env.initial_state()
        game_steps: list[ActorStep] = []
        visits: dict[tuple[object, ...], int] = {}
        no_progress = 0
        game_timed_out = False
        game_stalled = False
        while not state.terminal:
            signature = state_signature(state)
            visits[signature] = visits.get(signature, 0) + 1
            if config.repetition_limit > 0 and visits[signature] >= config.repetition_limit:
                game_timed_out = True
                break
            if state.turn_index >= config.max_turns:
                game_timed_out = True
                break
            raw_actions = env.legal_actions(state)
            if not raw_actions:
                game_stalled = True
                break
            if state.current_player == actor_seat:
                actions = list(actions_by_canonical_successor_key(state, raw_actions).values())
                batch = collate_entity_action_batch([(state, actions)], device=config.device)
                with torch.no_grad():
                    scores, _ = actor(**batch)
                    probabilities = full_support_probabilities(
                        scores,
                        batch["action_mask"],
                        config.behavior_temperature,
                        config.uniform_mix,
                    )[0, : len(actions)]
                generator = torch.Generator(device=probabilities.device)
                generator.manual_seed(config.seed_start * 1_000_003 + game_index * 10_007 + len(game_steps))
                action_row = int(torch.multinomial(probabilities, 1, generator=generator).item())
                greedy_row = int(torch.argmax(probabilities).item())
                non_greedy += action_row != greedy_row
                chosen = actions[action_row]
                probability_values = tuple(float(value) for value in probabilities.cpu().tolist())
                entropy = -sum(value * math.log(value) for value in probability_values if value > 0.0)
                normalized_entropy = entropy / math.log(len(actions)) if len(actions) > 1 else 0.0
                entropy_values.append(normalized_entropy)
                minimum_probability = min(minimum_probability, min(probability_values))
                action_types.add(chosen.action_type.name)
                game_steps.append(
                    ActorStep(
                        state=state,
                        legal_actions=tuple(actions),
                        action_row=action_row,
                        behavior_probabilities=probability_values,
                        behavior_log_probability=math.log(probability_values[action_row]),
                        behavior_entropy=entropy,
                        actor_seat=actor_seat,
                        game_index=game_index,
                    )
                )
            else:
                chosen = opponent.choose_action(env, state, raw_actions)
                if chosen is None:
                    game_stalled = True
                    break
            next_state = env.step(state, chosen)
            if is_progress_transition(state, next_state, chosen):
                no_progress = 0
            else:
                no_progress += 1
                if config.no_progress_limit > 0 and no_progress >= config.no_progress_limit:
                    game_timed_out = True
                    state = next_state
                    break
            state = next_state
        winner = state.winner if state.terminal else _score_winner(state)
        outcome = 0.0 if winner is None else (1.0 if winner == actor_seat else -1.0)
        for step in game_steps:
            step.final_return = outcome
        steps.extend(game_steps)
        actor_wins += winner == actor_seat
        opponent_wins += winner is not None and winner != actor_seat
        draws += winner is None
        timed_out += game_timed_out
        stalled += game_stalled
    return steps, RolloutSummary(
        games=config.rollout_games,
        steps=len(steps),
        actor_wins=actor_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        timed_out_games=timed_out,
        stalled_games=stalled,
        non_greedy_actions=non_greedy,
        action_type_count=len(action_types),
        minimum_behavior_probability=minimum_probability,
        mean_normalized_entropy=sum(entropy_values) / len(entropy_values),
        opponents=opponent_counts,
    )


def train_actor_critic_generation(
    parent_checkpoint: str | Path,
    opponent_factories: list[tuple[str, OpponentFactory]],
    config: ActorCriticConfig,
    critic_config: PrivilegedEntityCriticConfig | None = None,
) -> ActorCriticArtifacts:
    actor = load_entity_actor(parent_checkpoint, config.device)
    steps, rollout = collect_league_rollouts(actor, opponent_factories, config)
    if not steps:
        raise RuntimeError("League rollout produced no actor steps.")
    critic = PrivilegedEntityCritic(critic_config).to(config.device)
    critic_optimizer = AdamW(
        critic.parameters(), lr=config.critic_learning_rate, weight_decay=config.weight_decay
    )
    indices = list(range(len(steps)))
    for _ in range(config.critic_warmup_epochs):
        random.Random(config.seed_start + _).shuffle(indices)
        for batch_indices in _batches(indices, config.batch_size):
            targets = torch.tensor(
                [steps[index].final_return for index in batch_indices],
                dtype=torch.float32,
                device=config.device,
            )
            values = critic(**collate_privileged_state_batch(
                [(steps[index].state, steps[index].actor_seat) for index in batch_indices],
                device=config.device,
            ))
            loss = torch.nn.functional.mse_loss(values, targets)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            critic_optimizer.step()
    _assign_gae(steps, critic, config)
    advantages = torch.tensor([step.advantage for step in steps], dtype=torch.float32)
    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std(unbiased=False).clamp_min(1e-6))
    for step in steps:
        step.advantage = (step.advantage - advantage_mean) / advantage_std

    reconstruction_error = _behavior_reconstruction_error(actor, steps, config)
    actor_optimizer = AdamW(
        actor.parameters(), lr=config.actor_learning_rate, weight_decay=config.weight_decay
    )
    policy_losses: list[float] = []
    critic_losses: list[float] = []
    for epoch in range(config.actor_epochs):
        random.Random(config.seed_start + 100 + epoch).shuffle(indices)
        for batch_indices in _batches(indices, config.batch_size):
            batch_steps = [steps[index] for index in batch_indices]
            entity_batch = collate_entity_action_batch(
                [(step.state, list(step.legal_actions)) for step in batch_steps],
                device=config.device,
            )
            scores, _ = actor(**entity_batch)
            probabilities = full_support_probabilities(
                scores,
                entity_batch["action_mask"],
                config.behavior_temperature,
                config.uniform_mix,
            )
            row = torch.arange(len(batch_steps), device=config.device)
            action_rows = torch.tensor([step.action_row for step in batch_steps], device=config.device)
            old_log = torch.tensor(
                [step.behavior_log_probability for step in batch_steps],
                dtype=torch.float32,
                device=config.device,
            )
            new_log = probabilities[row, action_rows].clamp_min(1e-12).log()
            ratio = torch.exp(new_log - old_log)
            advantage = torch.tensor(
                [step.advantage for step in batch_steps], dtype=torch.float32, device=config.device
            )
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            loss = policy_loss - config.entropy_weight * entropy.mean()
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
    for _ in range(config.critic_epochs):
        for batch_indices in _batches(indices, config.batch_size):
            values = critic(**collate_privileged_state_batch(
                [(steps[index].state, steps[index].actor_seat) for index in batch_indices],
                device=config.device,
            ))
            targets = torch.tensor(
                [steps[index].value_target for index in batch_indices],
                dtype=torch.float32,
                device=config.device,
            )
            loss = torch.nn.functional.mse_loss(values, targets)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            critic_optimizer.step()
            critic_losses.append(float(loss.detach().cpu()))
    ratios, kls, entropies = _post_update_diagnostics(actor, steps, config)
    return ActorCriticArtifacts(
        actor=actor,
        critic=critic,
        rollout=rollout,
        update=UpdateSummary(
            behavior_log_prob_max_error=reconstruction_error,
            mean_ratio=sum(ratios) / len(ratios),
            max_ratio=max(ratios),
            mean_kl=sum(kls) / len(kls),
            mean_normalized_entropy=sum(entropies) / len(entropies),
            policy_loss=sum(policy_losses) / len(policy_losses),
            critic_loss=sum(critic_losses) / len(critic_losses) if critic_losses else 0.0,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
        ),
        steps=tuple(steps),
    )


def _assign_gae(steps: list[ActorStep], critic: PrivilegedEntityCritic, config: ActorCriticConfig) -> None:
    critic.eval()
    with torch.no_grad():
        values = critic(**collate_privileged_state_batch(
            [(step.state, step.actor_seat) for step in steps], device=config.device
        )).cpu().tolist()
    by_game: dict[int, list[int]] = {}
    for index, step in enumerate(steps):
        by_game.setdefault(step.game_index, []).append(index)
    for game_indices in by_game.values():
        running = 0.0
        for offset in range(len(game_indices) - 1, -1, -1):
            index = game_indices[offset]
            next_value = values[game_indices[offset + 1]] if offset + 1 < len(game_indices) else 0.0
            reward = steps[index].final_return if offset == len(game_indices) - 1 else 0.0
            delta = reward + config.gamma * next_value - values[index]
            running = delta + config.gamma * config.gae_lambda * running
            steps[index].advantage = running
            steps[index].value_target = running + values[index]


def _behavior_reconstruction_error(
    actor: CanonicalEntityActionScorerModel,
    steps: list[ActorStep],
    config: ActorCriticConfig,
) -> float:
    errors = []
    with torch.no_grad():
        for batch_indices in _batches(list(range(len(steps))), config.batch_size):
            batch_steps = [steps[index] for index in batch_indices]
            batch = collate_entity_action_batch(
                [(step.state, list(step.legal_actions)) for step in batch_steps],
                device=config.device,
            )
            scores, _ = actor(**batch)
            probabilities = full_support_probabilities(
                scores, batch["action_mask"], config.behavior_temperature, config.uniform_mix
            )
            for row, step in enumerate(batch_steps):
                reconstructed = math.log(float(probabilities[row, step.action_row].cpu()))
                errors.append(abs(reconstructed - step.behavior_log_probability))
    return max(errors)


def _post_update_diagnostics(
    actor: CanonicalEntityActionScorerModel,
    steps: list[ActorStep],
    config: ActorCriticConfig,
) -> tuple[list[float], list[float], list[float]]:
    ratios: list[float] = []
    kls: list[float] = []
    entropies: list[float] = []
    actor.eval()
    with torch.no_grad():
        for batch_indices in _batches(list(range(len(steps))), config.batch_size):
            batch_steps = [steps[index] for index in batch_indices]
            batch = collate_entity_action_batch(
                [(step.state, list(step.legal_actions)) for step in batch_steps], device=config.device
            )
            scores, _ = actor(**batch)
            probabilities = full_support_probabilities(
                scores, batch["action_mask"], config.behavior_temperature, config.uniform_mix
            )
            for row, step in enumerate(batch_steps):
                new = probabilities[row, : len(step.legal_actions)].cpu()
                old = torch.tensor(step.behavior_probabilities)
                ratios.append(float(new[step.action_row] / old[step.action_row]))
                kls.append(float((old * (old.clamp_min(1e-12).log() - new.clamp_min(1e-12).log())).sum()))
                entropy = float(-(new * new.clamp_min(1e-12).log()).sum())
                entropies.append(entropy / math.log(len(new)) if len(new) > 1 else 0.0)
    return ratios, kls, entropies


def _batches(indices: list[int], batch_size: int) -> list[list[int]]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _score_winner(state: SplendorState) -> int | None:
    scores = [player.score for player in state.players]
    if scores[0] != scores[1]:
        return 0 if scores[0] > scores[1] else 1
    cards = [len(player.purchased_cards) for player in state.players]
    if cards[0] != cards[1]:
        return 0 if cards[0] < cards[1] else 1
    return None
