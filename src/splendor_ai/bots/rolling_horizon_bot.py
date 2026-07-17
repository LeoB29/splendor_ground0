"""Information-legal branching-mutation rolling-horizon planning."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

from splendor_ai.encoding import (
    actions_by_canonical_successor_key,
    canonical_public_belief_key,
    canonical_successor_action_key,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.constants import TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE, SplendorState

from .hidden_information import determinize_for_observer
from .search_bot import ActionPriorScorer


EvaluatorMode = Literal["score_delta", "event_value_hc"]


@dataclass(frozen=True, slots=True)
class TransitionEvents:
    """The five hand-grouped event counts used by the published linear EF."""

    tokens_received: int = 0
    blind_reserves: int = 0
    visible_reserves: int = 0
    nobles_received: int = 0
    prestige_events: int = 0

    def signed_difference(self, other: "TransitionEvents") -> "TransitionEvents":
        return TransitionEvents(
            tokens_received=self.tokens_received - other.tokens_received,
            blind_reserves=self.blind_reserves - other.blind_reserves,
            visible_reserves=self.visible_reserves - other.visible_reserves,
            nobles_received=self.nobles_received - other.nobles_received,
            prestige_events=self.prestige_events - other.prestige_events,
        )


@dataclass(frozen=True, slots=True)
class RollingHorizonConfig:
    actor_horizon: int = 2
    sequence_evaluations: int = 32
    determinizations: int = 2
    max_actor_candidates: int = 4
    seed: int = 0
    determinization_seed: int = 0
    evaluator_mode: EvaluatorMode = "event_value_hc"

    def __post_init__(self) -> None:
        if self.actor_horizon < 1:
            raise ValueError("actor_horizon must be positive.")
        if self.determinizations < 1:
            raise ValueError("determinizations must be positive.")
        if self.sequence_evaluations < self.determinizations:
            raise ValueError("sequence_evaluations must cover every determinization.")
        if self.sequence_evaluations % self.determinizations:
            raise ValueError("sequence_evaluations must divide evenly across determinizations.")
        if self.max_actor_candidates < 1:
            raise ValueError("max_actor_candidates must be positive.")
        if self.evaluator_mode not in ("score_delta", "event_value_hc"):
            raise ValueError(f"Unknown evaluator_mode: {self.evaluator_mode}")


@dataclass(frozen=True, slots=True)
class RollingHorizonStats:
    sequence_evaluations: int
    genome_evaluations: int
    simulated_actions: int
    selected_root_rank: int
    selected_genome: tuple[int, ...]
    random_schedule: tuple[tuple[int, tuple[int, ...]], ...]


def transition_events(
    before: SplendorState,
    after: SplendorState,
    player_id: int,
) -> TransitionEvents:
    """Extract player-triggered event counts from one legal transition."""

    prior = before.players[player_id]
    current = after.players[player_id]
    token_gain = sum(
        max(current.tokens[color] - prior.tokens[color], 0)
        for color in (*TOKEN_COLORS, "gold")
    )

    prior_sources = list(prior.reserved_card_sources)
    current_sources = list(current.reserved_card_sources)
    added_sources = current_sources[len(prior_sources) :] if len(current_sources) > len(prior_sources) else []
    blind = sum(source == RESERVE_SOURCE_DECK for source in added_sources)
    visible = sum(source == RESERVE_SOURCE_VISIBLE for source in added_sources)

    prior_cards = {card.card_id for card in prior.purchased_cards}
    new_cards = [card for card in current.purchased_cards if card.card_id not in prior_cards]
    prior_nobles = {noble.noble_id for noble in prior.nobles}
    new_nobles = [noble for noble in current.nobles if noble.noble_id not in prior_nobles]
    prestige_events = sum(card.points > 0 for card in new_cards) + len(new_nobles)

    return TransitionEvents(
        tokens_received=token_gain,
        blind_reserves=blind,
        visible_reserves=visible,
        nobles_received=len(new_nobles),
        prestige_events=prestige_events,
    )


def event_value(events: TransitionEvents) -> float:
    """Published hand-grouped linear weights from Bravi and Lucas (2020)."""

    return (
        0.2 * events.tokens_received
        + 0.2 * events.blind_reserves
        - 0.4 * events.visible_reserves
        - 0.6 * events.nobles_received
        + 0.8 * events.prestige_events
    )


def transition_value(
    before: SplendorState,
    after: SplendorState,
    root_player: int,
    mode: EvaluatorMode,
) -> float:
    """Return a root-relative value for one simulated transition."""

    if after.terminal:
        if after.winner == root_player:
            return 100.0
        if after.winner is None:
            return 0.0
        return -100.0
    opponent = 1 - root_player
    if mode == "score_delta":
        root_delta = after.players[root_player].score - before.players[root_player].score
        opponent_delta = after.players[opponent].score - before.players[opponent].score
        return float(root_delta - opponent_delta)
    root_events = transition_events(before, after, root_player)
    opponent_events = transition_events(before, after, opponent)
    return event_value(root_events.signed_difference(opponent_events))


class BranchingMutationRollingHorizonBot:
    """Evolve short root-action rank genomes under a frozen policy model."""

    def __init__(
        self,
        prior: ActionPriorScorer,
        config: RollingHorizonConfig | None = None,
    ) -> None:
        self._prior = prior
        self.config = config or RollingHorizonConfig()
        self.last_stats: RollingHorizonStats | None = None
        self.total_planning_decisions = 0
        self.total_sequence_evaluations = 0
        self.total_simulated_actions = 0

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        root_player = state.current_player
        cache: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
        live_candidates = self._ranked_candidates(state, actions, cache)
        if len(live_candidates) == 1:
            self.last_stats = RollingHorizonStats(0, 0, 0, 0, (0,) * self.config.actor_horizon, ())
            return live_candidates[0]

        decision_seed = self.config.seed + state.turn_index * 1_000_003
        schedule_rng = random.Random(decision_seed)
        determinized_states = tuple(
            determinize_for_observer(
                state,
                observer_id=root_player,
                rng=random.Random(
                    self.config.determinization_seed
                    + state.turn_index * 10_000_019
                    + sample_index
                ),
            )
            for sample_index in range(self.config.determinizations)
        )
        genome_evaluations = self.config.sequence_evaluations // self.config.determinizations
        incumbent: tuple[int, ...] | None = None
        incumbent_value = float("-inf")
        simulated_actions = 0
        random_schedule: list[tuple[int, tuple[int, ...]]] = []

        for evaluation_index in range(genome_evaluations):
            if evaluation_index < len(live_candidates):
                mutation_point = 0
                suffix = (evaluation_index,) + tuple(
                    schedule_rng.randrange(self.config.max_actor_candidates)
                    for _ in range(1, self.config.actor_horizon)
                )
                genome = suffix
            else:
                mutation_point = schedule_rng.randrange(self.config.actor_horizon)
                suffix = tuple(
                    schedule_rng.randrange(self.config.max_actor_candidates)
                    for _ in range(self.config.actor_horizon - mutation_point)
                )
                prefix = () if incumbent is None else incumbent[:mutation_point]
                genome = prefix + suffix
            random_schedule.append((mutation_point, suffix))

            values = []
            for determinized in determinized_states:
                value, actions_used = self._evaluate_genome(
                    env,
                    determinized,
                    root_player,
                    genome,
                    cache,
                )
                values.append(value)
                simulated_actions += actions_used
            mean_value = sum(values) / len(values)
            if incumbent is None or mean_value > incumbent_value or (
                mean_value == incumbent_value and genome < incumbent
            ):
                incumbent = genome
                incumbent_value = mean_value

        assert incumbent is not None
        selected_rank = incumbent[0] % len(live_candidates)
        stats = RollingHorizonStats(
            sequence_evaluations=self.config.sequence_evaluations,
            genome_evaluations=genome_evaluations,
            simulated_actions=simulated_actions,
            selected_root_rank=selected_rank,
            selected_genome=incumbent,
            random_schedule=tuple(random_schedule),
        )
        self.last_stats = stats
        self.total_planning_decisions += 1
        self.total_sequence_evaluations += stats.sequence_evaluations
        self.total_simulated_actions += stats.simulated_actions
        return live_candidates[selected_rank]

    def _evaluate_genome(
        self,
        env: SplendorEnv,
        initial_state: SplendorState,
        root_player: int,
        genome: tuple[int, ...],
        cache: dict[tuple[object, ...], tuple[tuple[object, ...], ...]],
    ) -> tuple[float, int]:
        current = initial_state.copy_shallow()
        actor_decisions = 0
        value = 0.0
        simulated_actions = 0
        while not current.terminal and actor_decisions < self.config.actor_horizon:
            legal = env.legal_actions(current)
            if not legal:
                break
            candidates = self._ranked_candidates(current, legal, cache)
            if current.current_player == root_player:
                action = candidates[genome[actor_decisions] % len(candidates)]
                actor_decisions += 1
            else:
                action = candidates[0]
            next_state = env.step(current, action)
            value += transition_value(current, next_state, root_player, self.config.evaluator_mode)
            simulated_actions += 1
            current = next_state
        return value, simulated_actions

    def _ranked_candidates(
        self,
        state: SplendorState,
        legal_actions: list[Action],
        cache: dict[tuple[object, ...], tuple[tuple[object, ...], ...]],
    ) -> tuple[Action, ...]:
        key = canonical_public_belief_key(state, state.current_player)
        cached = cache.get(key)
        if cached is not None:
            by_key = actions_by_canonical_successor_key(state, legal_actions)
            rebound = tuple(by_key[action_key] for action_key in cached if action_key in by_key)
            if rebound:
                return rebound
        representatives = actions_by_canonical_successor_key(state, legal_actions)
        actions = list(representatives.values())
        scores = self._prior.score_actions(state, actions)
        ordered = tuple(
            sorted(
                actions,
                key=lambda action: (
                    -scores[action],
                    repr(canonical_successor_action_key(state, action)),
                ),
            )[: self.config.max_actor_candidates]
        )
        cache[key] = tuple(canonical_successor_action_key(state, action) for action in ordered)
        return ordered
