"""Clean on-policy actor-critic training for canonical entity policies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
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
    score_margin_scale: float = 20.0
    return_mode: str = "outcome"
    gamma: float = 1.0
    gae_lambda: float = 0.95
    actor_epochs: int = 1
    critic_warmup_epochs: int = 3
    critic_epochs: int = 1
    critic_validation_modulus: int = 0
    batch_size: int = 64
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_ratio: float = 0.15
    entropy_weight: float = 0.01
    target_kl: float | None = None
    max_grad_norm: float = 1.0
    device: str = "cpu"


@dataclass(slots=True)
class ActorStep:
    state: SplendorState
    legal_actions: tuple[Action, ...]
    action_row: int
    behavior_scores: tuple[float, ...]
    behavior_probabilities: tuple[float, ...]
    behavior_log_probability: float
    behavior_entropy: float
    actor_seat: int
    game_index: int
    opponent_label: str
    final_return: float = 0.0
    final_score_margin: float = 0.0
    score_margin_return: float = 0.0
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
    opponent_seats: dict[str, dict[str, int]]
    opponent_results: dict[str, dict[str, int]]
    actor_seat_results: dict[str, dict[str, int]]
    minimum_stratum_games: int
    maximum_stratum_games: int


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
    actor_epochs_completed: int
    stopped_for_target_kl: bool
    epoch_mean_kls: tuple[float, ...]
    first_batch_gradient_norm: float
    mean_gradient_norm: float
    max_gradient_norm: float
    parameter_l2_norm_before: float
    parameter_delta_l2: float
    relative_parameter_delta: float
    mean_centered_logit_absolute_delta: float
    max_centered_logit_absolute_delta: float
    mean_probability_total_variation: float
    max_probability_total_variation: float
    argmax_change_fraction: float
    advantage_log_probability_covariance: float
    advantage_log_probability_correlation: float
    action_type_advantage_means: dict[str, float]
    nonzero_temporal_reward_fraction: float
    mean_absolute_temporal_reward: float
    critic_validation_steps: int
    critic_validation_mse: float
    critic_validation_baseline_mse: float
    critic_validation_skill: float
    critic_validation_correlation: float


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


def normalized_score_margin(score_margin: float, scale: float = 20.0) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    return max(-1.0, min(1.0, score_margin / scale))


def collect_league_rollouts(
    actor: CanonicalEntityActionScorerModel,
    opponent_factories: list[tuple[str, OpponentFactory]],
    config: ActorCriticConfig,
) -> tuple[list[ActorStep], RolloutSummary]:
    if not opponent_factories:
        raise ValueError("At least one league opponent is required.")
    opponent_labels = [label for label, _ in opponent_factories]
    if len(set(opponent_labels)) != len(opponent_labels):
        raise ValueError("League opponent labels must be unique.")
    steps: list[ActorStep] = []
    actor_wins = opponent_wins = draws = timed_out = stalled = non_greedy = 0
    opponent_counts = {label: 0 for label in opponent_labels}
    opponent_seats = {
        label: {"seat0": 0, "seat1": 0} for label in opponent_labels
    }
    opponent_results = {
        label: {"actor_wins": 0, "opponent_wins": 0, "draws": 0}
        for label in opponent_labels
    }
    actor_seat_results = {
        "seat0": {"actor_wins": 0, "opponent_wins": 0, "draws": 0},
        "seat1": {"actor_wins": 0, "opponent_wins": 0, "draws": 0},
    }
    action_types: set[str] = set()
    minimum_probability = 1.0
    entropy_values: list[float] = []
    actor.eval()
    for game_index in range(config.rollout_games):
        schedule_index = game_index % (2 * len(opponent_factories))
        opponent_index = schedule_index // 2
        actor_seat = schedule_index % 2
        label, factory = opponent_factories[opponent_index]
        opponent_counts[label] += 1
        opponent_seats[label][f"seat{actor_seat}"] += 1
        opponent = factory()
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
                score_values = tuple(float(value) for value in scores[0, : len(actions)].cpu().tolist())
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
                        behavior_scores=score_values,
                        behavior_probabilities=probability_values,
                        behavior_log_probability=math.log(probability_values[action_row]),
                        behavior_entropy=entropy,
                        actor_seat=actor_seat,
                        game_index=game_index,
                        opponent_label=label,
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
        score_margin = float(
            state.players[actor_seat].score - state.players[1 - actor_seat].score
        )
        for step in game_steps:
            step.final_return = outcome
            step.final_score_margin = score_margin
            step.score_margin_return = normalized_score_margin(
                score_margin, config.score_margin_scale
            )
        steps.extend(game_steps)
        actor_wins += winner == actor_seat
        opponent_wins += winner is not None and winner != actor_seat
        draws += winner is None
        if winner is None:
            result_key = "draws"
        elif winner == actor_seat:
            result_key = "actor_wins"
        else:
            result_key = "opponent_wins"
        opponent_results[label][result_key] += 1
        actor_seat_results[f"seat{actor_seat}"][result_key] += 1
        timed_out += game_timed_out
        stalled += game_stalled
    stratum_counts = [
        seat_count
        for seat_counts in opponent_seats.values()
        for seat_count in seat_counts.values()
    ]
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
        opponent_seats=opponent_seats,
        opponent_results=opponent_results,
        actor_seat_results=actor_seat_results,
        minimum_stratum_games=min(stratum_counts),
        maximum_stratum_games=max(stratum_counts),
    )


def train_actor_critic_generation(
    parent_checkpoint: str | Path,
    opponent_factories: list[tuple[str, OpponentFactory]],
    config: ActorCriticConfig,
    critic_config: PrivilegedEntityCriticConfig | None = None,
) -> ActorCriticArtifacts:
    actor = load_entity_actor(parent_checkpoint, config.device)
    steps, rollout = collect_league_rollouts(actor, opponent_factories, config)
    return train_actor_critic_from_steps(
        parent_checkpoint,
        steps,
        rollout,
        config,
        critic_config=critic_config,
    )


def train_actor_critic_from_steps(
    parent_checkpoint: str | Path,
    source_steps: list[ActorStep] | tuple[ActorStep, ...],
    rollout: RolloutSummary,
    config: ActorCriticConfig,
    critic_config: PrivilegedEntityCriticConfig | None = None,
) -> ActorCriticArtifacts:
    if config.actor_epochs <= 0:
        raise ValueError("actor_epochs must be positive.")
    if config.target_kl is not None and config.target_kl <= 0.0:
        raise ValueError("target_kl must be positive when configured.")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive.")
    if config.return_mode not in {"outcome", "score_margin", "score_margin_delta"}:
        raise ValueError(
            "return_mode must be 'outcome', 'score_margin', or 'score_margin_delta'."
        )
    if config.critic_validation_modulus == 1 or config.critic_validation_modulus < 0:
        raise ValueError("critic_validation_modulus must be 0 or at least 2.")
    steps = [replace(step, advantage=0.0, value_target=0.0) for step in source_steps]
    if not steps:
        raise RuntimeError("League rollout produced no actor steps.")
    torch.manual_seed(config.seed_start)
    actor = load_entity_actor(parent_checkpoint, config.device)
    critic = PrivilegedEntityCritic(critic_config).to(config.device)
    critic_optimizer = AdamW(
        critic.parameters(), lr=config.critic_learning_rate, weight_decay=config.weight_decay
    )
    indices = list(range(len(steps)))
    if config.critic_validation_modulus >= 2:
        critic_validation_indices = [
            index
            for index, step in enumerate(steps)
            if step.game_index % config.critic_validation_modulus
            == config.critic_validation_modulus - 1
        ]
        critic_validation_index_set = set(critic_validation_indices)
        critic_train_indices = [
            index for index in indices if index not in critic_validation_index_set
        ]
    else:
        critic_validation_indices = []
        critic_train_indices = indices
    for epoch in range(config.critic_warmup_epochs):
        epoch_indices = list(critic_train_indices)
        random.Random(config.seed_start + epoch).shuffle(epoch_indices)
        for batch_indices in _batches(epoch_indices, config.batch_size):
            targets = torch.tensor(
                [_terminal_return(steps[index], config) for index in batch_indices],
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
    critic_validation = _critic_validation_diagnostics(
        critic,
        steps,
        critic_train_indices,
        critic_validation_indices,
        config,
    )
    temporal_rewards = _temporal_rewards(steps, config)
    _assign_gae(steps, critic, config)
    advantages = torch.tensor([step.advantage for step in steps], dtype=torch.float32)
    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std(unbiased=False).clamp_min(1e-6))
    advantage_covariance, advantage_correlation, action_type_advantage_means = (
        _advantage_action_diagnostics(steps)
    )
    for step in steps:
        step.advantage = (step.advantage - advantage_mean) / advantage_std

    reconstruction_error = _behavior_reconstruction_error(actor, steps, config)
    parameter_snapshot = {
        name: parameter.detach().cpu().clone() for name, parameter in actor.named_parameters()
    }
    parameter_l2_norm_before = math.sqrt(
        sum(float((parameter.double() ** 2).sum()) for parameter in parameter_snapshot.values())
    )
    actor_optimizer = AdamW(
        actor.parameters(), lr=config.actor_learning_rate, weight_decay=config.weight_decay
    )
    policy_losses: list[float] = []
    critic_losses: list[float] = []
    gradient_norms: list[float] = []
    epoch_mean_kls: list[float] = []
    stopped_for_target_kl = False
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
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), config.max_grad_norm
            )
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            actor_optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
        _, epoch_kls, _ = _post_update_diagnostics(actor, steps, config)
        epoch_mean_kl = sum(epoch_kls) / len(epoch_kls)
        epoch_mean_kls.append(epoch_mean_kl)
        if config.target_kl is not None and epoch_mean_kl >= config.target_kl:
            stopped_for_target_kl = True
            break
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
    movement = _movement_diagnostics(actor, steps, parameter_snapshot, config)
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
            actor_epochs_completed=len(epoch_mean_kls),
            stopped_for_target_kl=stopped_for_target_kl,
            epoch_mean_kls=tuple(epoch_mean_kls),
            first_batch_gradient_norm=gradient_norms[0],
            mean_gradient_norm=sum(gradient_norms) / len(gradient_norms),
            max_gradient_norm=max(gradient_norms),
            parameter_l2_norm_before=parameter_l2_norm_before,
            parameter_delta_l2=movement[0],
            relative_parameter_delta=movement[1],
            mean_centered_logit_absolute_delta=movement[2],
            max_centered_logit_absolute_delta=movement[3],
            mean_probability_total_variation=movement[4],
            max_probability_total_variation=movement[5],
            argmax_change_fraction=movement[6],
            advantage_log_probability_covariance=advantage_covariance,
            advantage_log_probability_correlation=advantage_correlation,
            action_type_advantage_means=action_type_advantage_means,
            nonzero_temporal_reward_fraction=(
                sum(abs(reward) > 1e-12 for reward in temporal_rewards)
                / len(temporal_rewards)
            ),
            mean_absolute_temporal_reward=(
                sum(abs(reward) for reward in temporal_rewards) / len(temporal_rewards)
            ),
            critic_validation_steps=critic_validation[0],
            critic_validation_mse=critic_validation[1],
            critic_validation_baseline_mse=critic_validation[2],
            critic_validation_skill=critic_validation[3],
            critic_validation_correlation=critic_validation[4],
        ),
        steps=tuple(steps),
    )


def _terminal_return(step: ActorStep, config: ActorCriticConfig) -> float:
    if config.return_mode == "score_margin":
        return normalized_score_margin(step.final_score_margin, config.score_margin_scale)
    if config.return_mode == "score_margin_delta":
        return (
            normalized_score_margin(step.final_score_margin, config.score_margin_scale)
            - _current_score_margin_return(step, config)
        )
    return step.final_return


def score_margin_delta_rewards(
    game_steps: list[ActorStep] | tuple[ActorStep, ...],
    scale: float = 20.0,
) -> list[float]:
    """Return causal actor-turn margin changes that telescope to remaining margin."""
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    rewards: list[float] = []
    for offset, step in enumerate(game_steps):
        current = normalized_score_margin(
            float(
                step.state.players[step.actor_seat].score
                - step.state.players[1 - step.actor_seat].score
            ),
            scale,
        )
        if offset + 1 < len(game_steps):
            next_step = game_steps[offset + 1]
            if next_step.game_index != step.game_index:
                raise ValueError("score_margin_delta_rewards requires steps from one game.")
            if next_step.actor_seat != step.actor_seat:
                raise ValueError("score_margin_delta_rewards requires one actor perspective.")
            following = normalized_score_margin(
                float(
                    next_step.state.players[next_step.actor_seat].score
                    - next_step.state.players[1 - next_step.actor_seat].score
                ),
                scale,
            )
        else:
            following = normalized_score_margin(step.final_score_margin, scale)
        rewards.append(following - current)
    return rewards


def _current_score_margin_return(step: ActorStep, config: ActorCriticConfig) -> float:
    return normalized_score_margin(
        float(
            step.state.players[step.actor_seat].score
            - step.state.players[1 - step.actor_seat].score
        ),
        config.score_margin_scale,
    )


def _temporal_rewards(steps: list[ActorStep], config: ActorCriticConfig) -> list[float]:
    by_game: dict[int, list[ActorStep]] = {}
    for step in steps:
        by_game.setdefault(step.game_index, []).append(step)
    rewards: list[float] = []
    for game_steps in by_game.values():
        if config.return_mode == "score_margin_delta":
            rewards.extend(score_margin_delta_rewards(game_steps, config.score_margin_scale))
        else:
            rewards.extend(
                [0.0] * (len(game_steps) - 1) + [_terminal_return(game_steps[-1], config)]
            )
    return rewards


def _critic_validation_diagnostics(
    critic: PrivilegedEntityCritic,
    steps: list[ActorStep],
    train_indices: list[int],
    validation_indices: list[int],
    config: ActorCriticConfig,
) -> tuple[int, float, float, float, float]:
    if not validation_indices:
        return 0, 0.0, 0.0, 0.0, 0.0
    train_mean = sum(_terminal_return(steps[index], config) for index in train_indices) / len(
        train_indices
    )
    predictions: list[float] = []
    targets: list[float] = []
    critic.eval()
    with torch.no_grad():
        for batch_indices in _batches(validation_indices, config.batch_size):
            values = critic(
                **collate_privileged_state_batch(
                    [(steps[index].state, steps[index].actor_seat) for index in batch_indices],
                    device=config.device,
                )
            )
            predictions.extend(float(value) for value in values.cpu().tolist())
            targets.extend(_terminal_return(steps[index], config) for index in batch_indices)
    mse = sum((prediction - target) ** 2 for prediction, target in zip(predictions, targets)) / len(
        targets
    )
    baseline_mse = sum((target - train_mean) ** 2 for target in targets) / len(targets)
    skill = 1.0 - mse / max(baseline_mse, 1e-12)
    prediction_tensor = torch.tensor(predictions, dtype=torch.float64)
    target_tensor = torch.tensor(targets, dtype=torch.float64)
    centered_predictions = prediction_tensor - prediction_tensor.mean()
    centered_targets = target_tensor - target_tensor.mean()
    denominator = float(
        centered_predictions.square().sum().sqrt() * centered_targets.square().sum().sqrt()
    )
    correlation = (
        float((centered_predictions * centered_targets).sum()) / denominator
        if denominator > 0.0
        else 0.0
    )
    return len(validation_indices), mse, baseline_mse, skill, correlation


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
        if config.return_mode == "score_margin_delta":
            game_rewards = score_margin_delta_rewards(
                [steps[index] for index in game_indices], config.score_margin_scale
            )
        else:
            game_rewards = [0.0] * (len(game_indices) - 1) + [
                _terminal_return(steps[game_indices[-1]], config)
            ]
        running = 0.0
        for offset in range(len(game_indices) - 1, -1, -1):
            index = game_indices[offset]
            next_value = values[game_indices[offset + 1]] if offset + 1 < len(game_indices) else 0.0
            reward = game_rewards[offset]
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


def _advantage_action_diagnostics(
    steps: list[ActorStep],
) -> tuple[float, float, dict[str, float]]:
    advantages = torch.tensor([step.advantage for step in steps], dtype=torch.float64)
    log_probabilities = torch.tensor(
        [step.behavior_log_probability for step in steps], dtype=torch.float64
    )
    centered_advantages = advantages - advantages.mean()
    centered_log_probabilities = log_probabilities - log_probabilities.mean()
    covariance = float((centered_advantages * centered_log_probabilities).mean())
    denominator = float(
        centered_advantages.square().mean().sqrt()
        * centered_log_probabilities.square().mean().sqrt()
    )
    correlation = covariance / denominator if denominator > 0.0 else 0.0
    by_action_type: dict[str, list[float]] = {}
    for step in steps:
        action_type = step.legal_actions[step.action_row].action_type.name
        by_action_type.setdefault(action_type, []).append(step.advantage)
    action_type_means = {
        action_type: sum(values) / len(values)
        for action_type, values in sorted(by_action_type.items())
    }
    return covariance, correlation, action_type_means


def _movement_diagnostics(
    actor: CanonicalEntityActionScorerModel,
    steps: list[ActorStep],
    parameter_snapshot: dict[str, torch.Tensor],
    config: ActorCriticConfig,
) -> tuple[float, float, float, float, float, float, float]:
    parameter_delta_squared = 0.0
    parameter_norm_squared = 0.0
    for name, parameter in actor.named_parameters():
        before = parameter_snapshot[name].double()
        after = parameter.detach().cpu().double()
        parameter_delta_squared += float(((after - before) ** 2).sum())
        parameter_norm_squared += float((before ** 2).sum())
    parameter_delta = math.sqrt(parameter_delta_squared)
    relative_parameter_delta = parameter_delta / max(math.sqrt(parameter_norm_squared), 1e-12)

    absolute_logit_deltas: list[float] = []
    total_variations: list[float] = []
    argmax_changes = 0
    actor.eval()
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
                action_count = len(step.legal_actions)
                old_scores = torch.tensor(step.behavior_scores, dtype=torch.float64)
                new_scores = scores[row, :action_count].detach().cpu().double()
                old_centered = old_scores - old_scores.mean()
                new_centered = new_scores - new_scores.mean()
                absolute_logit_deltas.extend(
                    (new_centered - old_centered).abs().tolist()
                )
                old_probabilities = torch.tensor(
                    step.behavior_probabilities, dtype=torch.float64
                )
                new_probabilities = probabilities[row, :action_count].detach().cpu().double()
                total_variations.append(
                    float(0.5 * (new_probabilities - old_probabilities).abs().sum())
                )
                argmax_changes += int(
                    int(torch.argmax(old_probabilities)) != int(torch.argmax(new_probabilities))
                )
    return (
        parameter_delta,
        relative_parameter_delta,
        sum(absolute_logit_deltas) / len(absolute_logit_deltas),
        max(absolute_logit_deltas),
        sum(total_variations) / len(total_variations),
        max(total_variations),
        argmax_changes / len(steps),
    )


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
