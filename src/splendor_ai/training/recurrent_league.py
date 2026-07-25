"""Bounded recurrent population-self-play collection and policy updates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Callable

import torch
from torch.optim import AdamW

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.bots.base import Bot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import GameResult, play_game
from splendor_ai.training.recurrent_policy import (
    PublicHistoryTracker,
    RecurrentLegalActionAttentionScorerModel,
    RecurrentPolicyInput,
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
    save_recurrent_policy_checkpoint,
)


OpponentFactory = Callable[[], Bot]
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")
SHALLOW_CONTRACT = {
    "depth": 1,
    "max_branching": 6,
    "max_buy_actions": 4,
    "max_reserve_actions": 2,
    "max_take_actions": 2,
    "seed": 23,
    "information_legal": True,
    "determinizations": 1,
    "determinization_seed": 29,
}


@dataclass(frozen=True, slots=True)
class RecurrentLeagueConfig:
    games: int = 240
    maximum_attempts: int = 300
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    behavior_temperature: float = 1.0
    uniform_mix: float = 0.05
    score_margin_scale: float = 20.0
    outcome_weight: float = 0.5
    early_turn_weight: float = 1.0
    mid_turn_weight: float = 1.0
    late_turn_weight: float = 1.0
    validation_divisor: int = 5
    minimum_validation_games_per_cell: int = 4
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_ratio: float = 0.20
    entropy_weight: float = 0.002
    max_grad_norm: float = 1.0
    target_validation_kl: float = 0.008
    projection_steps: int = 24
    device: str = "cpu"


@dataclass(slots=True)
class RecurrentLeagueStep:
    policy_input: RecurrentPolicyInput
    action_row: int
    behavior_scores: tuple[float, ...]
    behavior_probabilities: tuple[float, ...]
    behavior_log_probability: float
    game_id: int
    opponent_label: str
    actor_seat: int
    action_type: str
    split: str = ""
    final_wdl: float = 0.0
    final_margin: float = 0.0
    game_return: float = 0.0
    advantage: float = 0.0
    game_weight: float = 0.0


@dataclass(slots=True)
class RecurrentLeagueGame:
    game_id: int
    seed: int
    opponent_label: str
    actor_seat: int
    cell_ordinal: int
    termination_reason: str
    turns: int
    winner: int | None
    final_scores: tuple[int, int]
    steps: list[RecurrentLeagueStep] = field(repr=False)
    split: str = ""
    final_wdl: float = 0.0
    final_margin: float = 0.0
    game_return: float = 0.0

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("steps")
        payload["step_count"] = len(self.steps)
        return payload


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    requested_games: int
    eligible_games: int
    attempted_games: int
    ineligible_games: int
    steps: int
    train_games: int
    validation_games: int
    train_steps: int
    validation_steps: int
    opponent_seat_games: dict[str, dict[str, int]]
    split_opponent_seat_games: dict[str, dict[str, dict[str, int]]]
    termination_counts: dict[str, int]
    eligible_termination_counts: dict[str, int]
    action_type_counts: dict[str, int]
    non_greedy_actions: int
    minimum_behavior_probability: float
    mean_normalized_entropy: float
    return_by_cell: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class UpdateSummary:
    behavior_reconstruction_max_error: float
    first_gradient_norm: float
    mean_gradient_norm: float
    epochs_completed: int
    epoch_validation_kl: tuple[float, ...]
    epoch_policy_loss: tuple[float, ...]
    projection_applied: bool
    projection_alpha: float
    validation_raw_kl: float
    validation_argmax_movement: float
    validation_surrogate: float
    validation_normalized_entropy: float
    validation_mean_total_variation: float
    checkpoint_reload_exact: bool


class InMemoryRecurrentBot:
    """Greedy bot sharing a read-only recurrent model with an owned transcript."""

    def __init__(self, model: RecurrentLegalActionAttentionScorerModel) -> None:
        self.model = model
        self.history = PublicHistoryTracker()

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def start_game(self, seed: int) -> None:
        del seed
        self.history.reset()

    def observe_action(
        self,
        before: SplendorState,
        action: Action,
        after: SplendorState,
    ) -> None:
        del after
        self.history.record(before, action)

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        row = build_recurrent_policy_input(state, actions, self.history.events)
        batch = collate_recurrent_policy_inputs([row])
        with torch.no_grad():
            scores, _value = forward_recurrent(self.model, batch)
        return actions[int(torch.argmax(scores[0, : len(actions)]).item())]


def forward_recurrent(model, batch):
    return model(
        batch["observation"],
        batch["legal_action_features"],
        batch["legal_action_mask"],
        batch["history_action_type_ids"],
        batch["history_card_ids"],
        batch["history_noble_ids"],
        batch["history_numeric_features"],
        batch["history_mask"],
    )


def behavior_probabilities(
    scores: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
    uniform_mix: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= uniform_mix < 1.0:
        raise ValueError("uniform_mix must lie in [0, 1).")
    masked = scores.masked_fill(~mask, -torch.inf)
    policy = torch.softmax(masked / temperature, dim=1)
    legal = mask.to(dtype=policy.dtype)
    uniform = legal / legal.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (1.0 - uniform_mix) * policy + uniform_mix * uniform


def collect_recurrent_league(
    actor: RecurrentLegalActionAttentionScorerModel,
    opponent_factories: list[tuple[str, OpponentFactory]],
    *,
    generation: int,
    seed_start: int,
    config: RecurrentLeagueConfig,
) -> tuple[list[RecurrentLeagueGame], list[RecurrentLeagueStep], CollectionSummary]:
    if not opponent_factories:
        raise ValueError("At least one opponent is required.")
    labels = [label for label, _factory in opponent_factories]
    if len(set(labels)) != len(labels):
        raise ValueError("Opponent labels must be unique.")
    cell_count = 2 * len(opponent_factories)
    if config.games % cell_count:
        raise ValueError("games must divide exactly across opponent/seat cells.")
    games_per_cell = config.games // cell_count
    validation_per_cell = max(
        config.minimum_validation_games_per_cell,
        games_per_cell // config.validation_divisor,
    )
    if validation_per_cell >= games_per_cell:
        raise ValueError("Every cell must retain training games.")

    actor.eval()
    games: list[RecurrentLeagueGame] = []
    all_steps: list[RecurrentLeagueStep] = []
    attempts = 0
    ineligible = 0
    termination_counts: dict[str, int] = defaultdict(int)
    action_type_counts: dict[str, int] = defaultdict(int)
    non_greedy = 0
    probability_minimum = 1.0
    entropies: list[float] = []
    for label, factory in opponent_factories:
        for actor_seat in (0, 1):
            accepted = 0
            while accepted < games_per_cell:
                if attempts >= config.maximum_attempts:
                    raise RuntimeError("Recurrent league exhausted its attempt budget.")
                seed = seed_start + attempts
                attempts += 1
                game = _collect_one_game(
                    actor,
                    factory,
                    generation=generation,
                    seed=seed,
                    actor_seat=actor_seat,
                    opponent_label=label,
                    game_id=len(games),
                    cell_ordinal=accepted,
                    config=config,
                )
                termination_counts[game.termination_reason] += 1
                if game.termination_reason not in VALID_TERMINATIONS:
                    ineligible += 1
                    continue
                accepted += 1
                split = (
                    "validation"
                    if game.cell_ordinal >= games_per_cell - validation_per_cell
                    else "train"
                )
                game.split = split
                for step in game.steps:
                    step.split = split
                    action_type_counts[step.action_type] += 1
                    probabilities = step.behavior_probabilities
                    probability_minimum = min(probability_minimum, min(probabilities))
                    greedy = max(range(len(probabilities)), key=probabilities.__getitem__)
                    non_greedy += step.action_row != greedy
                    entropy = -sum(
                        probability * math.log(probability)
                        for probability in probabilities
                        if probability > 0.0
                    )
                    entropies.append(
                        entropy / math.log(len(probabilities))
                        if len(probabilities) > 1
                        else 0.0
                    )
                games.append(game)
                all_steps.extend(game.steps)

    _assign_returns_and_advantages(games, config)
    counts: dict[str, dict[str, int]] = {
        label: {"seat0": 0, "seat1": 0} for label in labels
    }
    split_counts = {
        split: {
            label: {"seat0": 0, "seat1": 0}
            for label in labels
        }
        for split in ("train", "validation")
    }
    return_cells: dict[str, list[float]] = defaultdict(list)
    for game in games:
        seat = f"seat{game.actor_seat}"
        counts[game.opponent_label][seat] += 1
        split_counts[game.split][game.opponent_label][seat] += 1
        return_cells[f"{game.opponent_label}:{seat}:{game.split}"].append(
            game.game_return
        )
    return_summary = {
        key: {
            "games": len(values),
            "mean": sum(values) / len(values),
            "std": _population_std(values),
        }
        for key, values in sorted(return_cells.items())
    }
    train_games = sum(game.split == "train" for game in games)
    validation_games = len(games) - train_games
    train_steps = sum(step.split == "train" for step in all_steps)
    eligible_termination_counts: dict[str, int] = defaultdict(int)
    for game in games:
        eligible_termination_counts[game.termination_reason] += 1
    summary = CollectionSummary(
        requested_games=config.games,
        eligible_games=len(games),
        attempted_games=attempts,
        ineligible_games=ineligible,
        steps=len(all_steps),
        train_games=train_games,
        validation_games=validation_games,
        train_steps=train_steps,
        validation_steps=len(all_steps) - train_steps,
        opponent_seat_games=counts,
        split_opponent_seat_games=split_counts,
        termination_counts=dict(sorted(termination_counts.items())),
        eligible_termination_counts=dict(
            sorted(eligible_termination_counts.items())
        ),
        action_type_counts=dict(sorted(action_type_counts.items())),
        non_greedy_actions=non_greedy,
        minimum_behavior_probability=probability_minimum,
        mean_normalized_entropy=sum(entropies) / len(entropies),
        return_by_cell=return_summary,
    )
    return games, all_steps, summary


def _collect_one_game(
    actor: RecurrentLegalActionAttentionScorerModel,
    opponent_factory: OpponentFactory,
    *,
    generation: int,
    seed: int,
    actor_seat: int,
    opponent_label: str,
    game_id: int,
    cell_ordinal: int,
    config: RecurrentLeagueConfig,
) -> RecurrentLeagueGame:
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    opponent = opponent_factory()
    _start_bot(opponent, seed)
    history = PublicHistoryTracker()
    steps: list[RecurrentLeagueStep] = []
    visits: dict[tuple[object, ...], int] = {}
    no_progress = 0
    forced_pass_streak = 0
    termination = "completed"
    actor_decision = 0
    while not state.terminal:
        signature = state_signature(state)
        visits[signature] = visits.get(signature, 0) + 1
        if config.repetition_limit > 0 and visits[signature] >= config.repetition_limit:
            termination = "repetition_cutoff"
            break
        if state.turn_index >= config.max_turns:
            termination = "max_turns"
            break
        actions = env.legal_actions(state)
        if not actions:
            termination = "stalled"
            break
        if state.current_player == actor_seat:
            row = build_recurrent_policy_input(state, actions, history.events)
            batch = collate_recurrent_policy_inputs([row], config.device)
            with torch.no_grad():
                scores, _value = forward_recurrent(actor, batch)
                probabilities = behavior_probabilities(
                    scores,
                    batch["legal_action_mask"],
                    temperature=config.behavior_temperature,
                    uniform_mix=config.uniform_mix,
                )[0, : len(actions)]
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(
                generation * 10**15 + seed * 100_000 + actor_decision
            )
            action_row = int(
                torch.multinomial(probabilities, 1, generator=generator).item()
            )
            chosen = actions[action_row]
            probability_values = tuple(
                float(value)
                for value in probabilities.detach().cpu().tolist()
            )
            steps.append(
                RecurrentLeagueStep(
                    policy_input=row,
                    action_row=action_row,
                    behavior_scores=tuple(
                        float(value)
                        for value in scores[0, : len(actions)].detach().cpu().tolist()
                    ),
                    behavior_probabilities=probability_values,
                    behavior_log_probability=math.log(
                        probability_values[action_row]
                    ),
                    game_id=game_id,
                    opponent_label=opponent_label,
                    actor_seat=actor_seat,
                    action_type=chosen.action_type.name,
                )
            )
            actor_decision += 1
        else:
            chosen = opponent.choose_action(env, state, actions)
            if chosen is None:
                termination = "opponent_returned_none"
                break
        try:
            next_state = env.step(state, chosen)
        except ValueError:
            termination = "illegal_action"
            break
        _observe_bot(opponent, state, chosen, next_state)
        history.record(state, chosen)
        forced_pass = (
            len(actions) == 1 and actions[0].action_type == ActionType.PASS
        )
        forced_pass_streak = forced_pass_streak + 1 if forced_pass else 0
        if is_progress_transition(state, next_state, chosen):
            no_progress = 0
        else:
            no_progress += 1
        state = next_state
        if forced_pass_streak >= 2:
            termination = "forced_pass_deadlock"
            break
        if config.no_progress_limit > 0 and no_progress >= config.no_progress_limit:
            termination = "no_progress_cutoff"
            break
    winner = state.winner if state.terminal else _score_winner(state)
    return RecurrentLeagueGame(
        game_id=game_id,
        seed=seed,
        opponent_label=opponent_label,
        actor_seat=actor_seat,
        cell_ordinal=cell_ordinal,
        termination_reason=termination,
        turns=state.turn_index,
        winner=winner,
        final_scores=(state.players[0].score, state.players[1].score),
        steps=steps,
    )


def _assign_returns_and_advantages(
    games: list[RecurrentLeagueGame],
    config: RecurrentLeagueConfig,
) -> None:
    by_cell_split: dict[tuple[str, int, str], list[RecurrentLeagueGame]] = (
        defaultdict(list)
    )
    for game in games:
        game.final_wdl = (
            0.0
            if game.winner is None
            else 1.0
            if game.winner == game.actor_seat
            else -1.0
        )
        raw_margin = float(
            game.final_scores[game.actor_seat]
            - game.final_scores[1 - game.actor_seat]
        )
        game.final_margin = max(
            -1.0,
            min(1.0, raw_margin / config.score_margin_scale),
        )
        game.game_return = (
            config.outcome_weight * game.final_wdl
            + (1.0 - config.outcome_weight) * game.final_margin
        )
        by_cell_split[(game.opponent_label, game.actor_seat, game.split)].append(
            game
        )
    for cell_games in by_cell_split.values():
        returns = [game.game_return for game in cell_games]
        mean = sum(returns) / len(returns)
        std = max(_population_std(returns), 1e-6)
        for game in cell_games:
            advantage = (game.game_return - mean) / std
            phase_weights = [
                _decision_phase_weight(step.policy_input.turn, config)
                for step in game.steps
            ]
            phase_weight_total = sum(phase_weights)
            if phase_weight_total <= 0.0:
                raise ValueError("Every game needs positive decision weight.")
            for step, phase_weight in zip(game.steps, phase_weights):
                step.final_wdl = game.final_wdl
                step.final_margin = game.final_margin
                step.game_return = game.game_return
                step.advantage = advantage
                step.game_weight = phase_weight / phase_weight_total


def train_recurrent_generation(
    parent: RecurrentLegalActionAttentionScorerModel,
    steps: list[RecurrentLeagueStep],
    *,
    generation: int,
    config: RecurrentLeagueConfig,
) -> tuple[RecurrentLegalActionAttentionScorerModel, UpdateSummary]:
    train_steps = [step for step in steps if step.split == "train"]
    validation_steps = [step for step in steps if step.split == "validation"]
    if not train_steps or not validation_steps:
        raise ValueError("Both train and validation decisions are required.")
    proposal = copy.deepcopy(parent).to(config.device)
    proposal.train()
    parent = parent.to(config.device).eval()
    behavior_error = _behavior_reconstruction_error(parent, steps, config)
    parent_trainable = _trainable_snapshot(parent)
    optimizer = AdamW(
        [parameter for parameter in proposal.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    gradient_norms: list[float] = []
    epoch_kls: list[float] = []
    epoch_losses: list[float] = []
    indices = list(range(len(train_steps)))
    for epoch in range(config.epochs):
        random.Random(620_062 + generation * 100 + epoch).shuffle(indices)
        losses = []
        for batch_indices in _batches(indices, config.batch_size):
            chosen = [train_steps[index] for index in batch_indices]
            batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in chosen],
                config.device,
            )
            scores, _value = forward_recurrent(proposal, batch)
            probabilities = behavior_probabilities(
                scores,
                batch["legal_action_mask"],
                temperature=config.behavior_temperature,
                uniform_mix=config.uniform_mix,
            )
            row = torch.arange(len(chosen), device=config.device)
            action_rows = torch.tensor(
                [step.action_row for step in chosen],
                dtype=torch.long,
                device=config.device,
            )
            old_log = torch.tensor(
                [step.behavior_log_probability for step in chosen],
                dtype=torch.float32,
                device=config.device,
            )
            new_log = probabilities[row, action_rows].clamp_min(1e-12).log()
            ratio = torch.exp(new_log - old_log)
            advantage = torch.tensor(
                [step.advantage for step in chosen],
                dtype=torch.float32,
                device=config.device,
            )
            weight = torch.tensor(
                [step.game_weight for step in chosen],
                dtype=torch.float32,
                device=config.device,
            )
            clipped = ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            )
            objective = torch.minimum(ratio * advantage, clipped * advantage)
            policy_loss = -(weight * objective).sum() / weight.sum().clamp_min(1e-12)
            entropy = -(
                probabilities.clamp_min(1e-12).log() * probabilities
            ).sum(dim=1)
            entropy_loss = (weight * entropy).sum() / weight.sum().clamp_min(1e-12)
            loss = policy_loss - config.entropy_weight * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in proposal.parameters()
                    if parameter.requires_grad
                ],
                config.max_grad_norm,
            )
            gradient_norms.append(float(norm.detach().cpu()))
            optimizer.step()
            losses.append(float(policy_loss.detach().cpu()))
        diagnostics = policy_diagnostics(
            parent,
            proposal,
            validation_steps,
            config,
        )
        epoch_kls.append(diagnostics["raw_kl"])
        epoch_losses.append(sum(losses) / len(losses))
        if diagnostics["raw_kl"] >= config.target_validation_kl:
            break

    proposal.eval()
    diagnostics = policy_diagnostics(parent, proposal, validation_steps, config)
    projection_applied = diagnostics["raw_kl"] > config.target_validation_kl
    projection_alpha = 1.0
    if projection_applied:
        proposal_trainable = _trainable_snapshot(proposal)
        low = 0.0
        high = 1.0
        best = 0.0
        for _ in range(config.projection_steps):
            alpha = (low + high) / 2.0
            _interpolate_trainable(
                proposal,
                parent_trainable,
                proposal_trainable,
                alpha,
            )
            candidate = policy_diagnostics(
                parent,
                proposal,
                validation_steps,
                config,
            )
            if candidate["raw_kl"] <= config.target_validation_kl:
                best = alpha
                low = alpha
            else:
                high = alpha
        projection_alpha = best
        _interpolate_trainable(
            proposal,
            parent_trainable,
            proposal_trainable,
            projection_alpha,
        )
        diagnostics = policy_diagnostics(
            parent,
            proposal,
            validation_steps,
            config,
        )
    return proposal, UpdateSummary(
        behavior_reconstruction_max_error=behavior_error,
        first_gradient_norm=gradient_norms[0],
        mean_gradient_norm=sum(gradient_norms) / len(gradient_norms),
        epochs_completed=len(epoch_kls),
        epoch_validation_kl=tuple(epoch_kls),
        epoch_policy_loss=tuple(epoch_losses),
        projection_applied=projection_applied,
        projection_alpha=projection_alpha,
        validation_raw_kl=diagnostics["raw_kl"],
        validation_argmax_movement=diagnostics["argmax_movement"],
        validation_surrogate=diagnostics["surrogate"],
        validation_normalized_entropy=diagnostics["normalized_entropy"],
        validation_mean_total_variation=diagnostics["mean_total_variation"],
        checkpoint_reload_exact=False,
    )


def policy_diagnostics(
    parent: RecurrentLegalActionAttentionScorerModel,
    candidate: RecurrentLegalActionAttentionScorerModel,
    steps: list[RecurrentLeagueStep],
    config: RecurrentLeagueConfig,
) -> dict[str, float]:
    total_weight = 0.0
    kl_sum = 0.0
    surrogate_sum = 0.0
    entropy_sum = 0.0
    tv_sum = 0.0
    changed_weight = 0.0
    parent.eval()
    candidate.eval()
    with torch.no_grad():
        for start in range(0, len(steps), config.batch_size):
            chosen = steps[start : start + config.batch_size]
            batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in chosen],
                config.device,
            )
            parent_scores, _ = forward_recurrent(parent, batch)
            candidate_scores, _ = forward_recurrent(candidate, batch)
            mask = batch["legal_action_mask"]
            parent_probabilities = torch.softmax(
                parent_scores.masked_fill(~mask, -torch.inf),
                dim=1,
            )
            candidate_probabilities = torch.softmax(
                candidate_scores.masked_fill(~mask, -torch.inf),
                dim=1,
            )
            for row, step in enumerate(chosen):
                width = len(step.policy_input.legal_action_features)
                old = parent_probabilities[row, :width]
                new = candidate_probabilities[row, :width]
                weight = step.game_weight
                kl = float(
                    (
                        old
                        * (
                            old.clamp_min(1e-12).log()
                            - new.clamp_min(1e-12).log()
                        )
                    ).sum().cpu()
                )
                log_ratio = float(
                    (
                        new[step.action_row].clamp_min(1e-12).log()
                        - old[step.action_row].clamp_min(1e-12).log()
                    ).cpu()
                )
                entropy = float(
                    -(new * new.clamp_min(1e-12).log()).sum().cpu()
                )
                normalized_entropy = (
                    entropy / math.log(width) if width > 1 else 0.0
                )
                tv = float((0.5 * torch.abs(old - new).sum()).cpu())
                total_weight += weight
                kl_sum += weight * kl
                surrogate_sum += weight * step.advantage * log_ratio
                entropy_sum += weight * normalized_entropy
                tv_sum += weight * tv
                changed_weight += weight * (
                    int(torch.argmax(old).item()) != int(torch.argmax(new).item())
                )
    denominator = max(total_weight, 1e-12)
    return {
        "raw_kl": kl_sum / denominator,
        "argmax_movement": changed_weight / denominator,
        "surrogate": surrogate_sum / denominator,
        "normalized_entropy": entropy_sum / denominator,
        "mean_total_variation": tv_sum / denominator,
    }


def verify_checkpoint_reload(
    path: Path,
    model: RecurrentLegalActionAttentionScorerModel,
    validation_steps: list[RecurrentLeagueStep],
    config: RecurrentLeagueConfig,
) -> bool:
    reloaded = load_recurrent_policy_checkpoint(path, device=config.device).eval()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(validation_steps), config.batch_size):
            chosen = validation_steps[start : start + config.batch_size]
            batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in chosen],
                config.device,
            )
            expected = forward_recurrent(model, batch)
            actual = forward_recurrent(reloaded, batch)
            if not torch.equal(expected[0], actual[0]):
                return False
            if not torch.equal(expected[1], actual[1]):
                return False
    return True


def run_parent_gate(
    candidate: RecurrentLegalActionAttentionScorerModel,
    parent: RecurrentLegalActionAttentionScorerModel,
    *,
    seed_start: int,
    pairs: int,
    config: RecurrentLeagueConfig,
) -> dict[str, Any]:
    games: list[dict[str, Any]] = []
    points = 0.0
    paired_margins: list[float] = []
    for seed in range(seed_start, seed_start + pairs):
        first = play_game(
            InMemoryRecurrentBot(candidate),
            InMemoryRecurrentBot(parent),
            seed=seed,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
        )
        second = play_game(
            InMemoryRecurrentBot(parent),
            InMemoryRecurrentBot(candidate),
            seed=seed,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
        )
        margins = [
            first.final_scores[0] - first.final_scores[1],
            second.final_scores[1] - second.final_scores[0],
        ]
        paired_margins.append(sum(margins) / 2.0)
        for result, seat in ((first, 0), (second, 1)):
            points += (
                0.5 if result.winner is None else 1.0 if result.winner == seat else 0.0
            )
        games.append(
            {
                "seed": seed,
                "first": asdict(first),
                "second": asdict(second),
            }
        )
    valid = all(
        game[key]["termination_reason"] in VALID_TERMINATIONS
        for game in games
        for key in ("first", "second")
    )
    return {
        "pairs": pairs,
        "games": 2 * pairs,
        "valid": valid,
        "candidate_score_share": points / (2 * pairs),
        "mean_paired_score_margin": sum(paired_margins) / pairs,
        "rows": games,
    }


def run_static_population_gate(
    candidate: RecurrentLegalActionAttentionScorerModel,
    control: RecurrentLegalActionAttentionScorerModel,
    static_opponents: list[tuple[str, OpponentFactory]],
    *,
    seed_starts: dict[str, int],
    pairs: int,
    config: RecurrentLeagueConfig,
) -> dict[str, Any]:
    opponents: dict[str, Any] = {}
    total_candidate_points = 0.0
    total_control_points = 0.0
    total_games = 0
    valid = True
    for label, factory in static_opponents:
        candidate_points = 0.0
        control_points = 0.0
        rows = []
        for seed in range(seed_starts[label], seed_starts[label] + pairs):
            results = []
            for model in (candidate, control):
                first = play_game(
                    InMemoryRecurrentBot(model),
                    factory(),
                    seed=seed,
                    max_turns=config.max_turns,
                    repetition_limit=config.repetition_limit,
                    no_progress_limit=config.no_progress_limit,
                )
                second = play_game(
                    factory(),
                    InMemoryRecurrentBot(model),
                    seed=seed,
                    max_turns=config.max_turns,
                    repetition_limit=config.repetition_limit,
                    no_progress_limit=config.no_progress_limit,
                )
                results.append((first, second))
            for model_index, model_results in enumerate(results):
                model_points = 0.0
                for result, seat in zip(model_results, (0, 1)):
                    valid &= result.termination_reason in VALID_TERMINATIONS
                    model_points += (
                        0.5
                        if result.winner is None
                        else 1.0
                        if result.winner == seat
                        else 0.0
                    )
                if model_index == 0:
                    candidate_points += model_points
                else:
                    control_points += model_points
            rows.append(
                {
                    "seed": seed,
                    "candidate": [asdict(result) for result in results[0]],
                    "control": [asdict(result) for result in results[1]],
                }
            )
        game_count = 2 * pairs
        candidate_share = candidate_points / game_count
        control_share = control_points / game_count
        opponents[label] = {
            "candidate_score_share": candidate_share,
            "control_score_share": control_share,
            "delta": candidate_share - control_share,
            "rows": rows,
        }
        total_candidate_points += candidate_points
        total_control_points += control_points
        total_games += game_count
    return {
        "valid": valid,
        "games": total_games * 2,
        "aggregate_candidate_score_share": total_candidate_points / total_games,
        "aggregate_control_score_share": total_control_points / total_games,
        "aggregate_delta": (
            total_candidate_points - total_control_points
        )
        / total_games,
        "opponents": opponents,
    }


def recurrent_snapshot_factory(
    model: RecurrentLegalActionAttentionScorerModel,
) -> OpponentFactory:
    return lambda: InMemoryRecurrentBot(model)


def static_opponent_factories(
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
    *,
    device: str = "cpu",
) -> list[tuple[str, OpponentFactory]]:
    return [
        (
            "035",
            lambda: LegalActionAttentionScorerBot(checkpoint_035, device=device),
        ),
        ("016", lambda: LegalActionScorerBot(checkpoint_016, device=device)),
        (
            "023",
            lambda: CanonicalEntityActionScorerBot(checkpoint_023, device=device),
        ),
        ("shallow", lambda: ShallowSearchBot(**SHALLOW_CONTRACT)),
    ]


def save_generation_checkpoint(
    path: Path,
    model: RecurrentLegalActionAttentionScorerModel,
    *,
    generation: int,
    parent_checkpoint: Path,
    provenance: dict[str, Any],
) -> None:
    save_recurrent_policy_checkpoint(
        path,
        model,
        {
            **provenance,
            "experiment": "062",
            "generation": generation,
            "immediate_parent_checkpoint": str(parent_checkpoint).replace("\\", "/"),
            "immediate_parent_sha256": sha256(parent_checkpoint),
        },
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _behavior_reconstruction_error(
    model: RecurrentLegalActionAttentionScorerModel,
    steps: list[RecurrentLeagueStep],
    config: RecurrentLeagueConfig,
) -> float:
    errors: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(steps), config.batch_size):
            chosen = steps[start : start + config.batch_size]
            batch = collate_recurrent_policy_inputs(
                [step.policy_input for step in chosen],
                config.device,
            )
            scores, _value = forward_recurrent(model, batch)
            probabilities = behavior_probabilities(
                scores,
                batch["legal_action_mask"],
                temperature=config.behavior_temperature,
                uniform_mix=config.uniform_mix,
            )
            for row, step in enumerate(chosen):
                reconstructed = math.log(
                    float(probabilities[row, step.action_row].cpu())
                )
                errors.append(
                    abs(reconstructed - step.behavior_log_probability)
                )
    return max(errors, default=0.0)


def _trainable_snapshot(
    model: RecurrentLegalActionAttentionScorerModel,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _interpolate_trainable(
    model: RecurrentLegalActionAttentionScorerModel,
    parent: dict[str, torch.Tensor],
    proposal: dict[str, torch.Tensor],
    alpha: float,
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(
                    parent[name] + alpha * (proposal[name] - parent[name])
                )


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _decision_phase_weight(
    turn: int,
    config: RecurrentLeagueConfig,
) -> float:
    if min(
        config.early_turn_weight,
        config.mid_turn_weight,
        config.late_turn_weight,
    ) < 0.0:
        raise ValueError("Decision phase weights must be nonnegative.")
    if turn < 20:
        return config.early_turn_weight
    if turn < 50:
        return config.mid_turn_weight
    return config.late_turn_weight


def _batches(indices: list[int], size: int) -> list[list[int]]:
    return [
        indices[start : start + size]
        for start in range(0, len(indices), size)
    ]


def _start_bot(bot: Bot, seed: int) -> None:
    hook = getattr(bot, "start_game", None)
    if callable(hook):
        hook(seed)


def _observe_bot(
    bot: Bot,
    before: SplendorState,
    action: Action,
    after: SplendorState,
) -> None:
    hook = getattr(bot, "observe_action", None)
    if callable(hook):
        hook(before, action, after)


def _score_winner(state: SplendorState) -> int | None:
    scores = [player.score for player in state.players]
    if scores[0] != scores[1]:
        return 0 if scores[0] > scores[1] else 1
    cards = [len(player.purchased_cards) for player in state.players]
    if cards[0] != cards[1]:
        return 0 if cards[0] < cards[1] else 1
    return None
