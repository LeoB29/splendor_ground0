"""Frozen matched-seed resolver tiebreak-utility audit (experiment 076)."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Sequence

from scipy.stats import beta
import torch

from splendor_ai.cfr import (
    ResolvedLeaf,
    TwoPlyResolution,
    deterministic_belief_range,
    resolve_two_ply,
)
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import (
    ActionCodec,
    canonical_public_belief_key,
    canonical_successor_action_key,
)
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    ResolverGameplayConfig,
    _distinct_canonical_scores,
    gameplay_belief_root_index,
    gameplay_continuation_seed,
)
from splendor_ai.training.entity_actor_critic import normalized_score_margin
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_action_preference_repair import (
    select_canonical_loop_fallback,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer
from splendor_ai.training.run_rollout_action_credit import _sample_scored_action


SOURCE_SHA256 = {
    "result": "60458c1bea4328b64f33ad8238529d074a25d1034fa2845e0b339deee3827153",
    "train": "6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5",
    "validation": "865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052",
    "confirmation": "8625d9049957ce266fa544896f0262cb37ccea6d168ed0c60a847ef5325cce8f",
    "parent_035": "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd",
}
EXPECTED_ROWS = {"train": 4_397, "validation": 1_135, "confirmation": 1_093}
EXPECTED_GAMES = {"train": 80, "validation": 20, "confirmation": 20}
PHASES = ("early", "mid", "late")
EXPECTED_STATES = 120
EXPECTED_BRANCHES_PER_STATE = 192
EXPECTED_BRANCHES = EXPECTED_STATES * EXPECTED_BRANCHES_PER_STATE


@dataclass(frozen=True, slots=True)
class TiebreakAuditConfig:
    source_dir: Path = Path("outputs/resolver_selfplay_distillation_052")
    checkpoint: Path = Path(
        "outputs/legal_action_scorer_035_legal_observation/"
        "legal_action_attention_scorer_best.pt"
    )
    output_dir: Path = Path("outputs/resolver_tiebreak_utility_audit_076")
    tiebreak_value: float = 0.025
    comparison_tolerance: float = 1e-12
    maximum_loop_adjudicated_fraction: float = 0.001
    workers: int = 10
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class DualContinuationOutcome:
    original_value: float
    tiebreak_value: float
    terminal: bool
    resolved: bool
    turns: int
    fallback_triggers: int
    loop_adjudicated: bool
    hard_cap_reached: bool
    score_margin: int
    winner: int | None
    root_cards: int
    opponent_cards: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _DualLeafTask:
    index: int
    after_root: SplendorState
    response_action: Action
    actor_seat: int
    seeds: tuple[int, ...]
    epsilon: float
    temperature: float
    top_k: int
    max_additional_turns: int
    score_margin_scale: float
    max_loop_fallback_triggers: int
    tiebreak_bonus: float


_WORKER_SCORER: Any | None = None


def _initialize_worker(checkpoint: Path, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(
        checkpoint, device, "legal_action_attention_scorer"
    )


def _evaluate_task(task: _DualLeafTask) -> tuple[int, tuple[DualContinuationOutcome, ...]]:
    if _WORKER_SCORER is None:
        raise RuntimeError("076 worker scorer is not initialized.")
    return task.index, tuple(
        rollout_dual_utility_loop_safe(
            SplendorEnv(),
            _WORKER_SCORER,
            task.after_root,
            task.response_action,
            task.actor_seat,
            seed,
            task.epsilon,
            task.temperature,
            task.top_k,
            task.max_additional_turns,
            task.score_margin_scale,
            task.max_loop_fallback_triggers,
            task.tiebreak_bonus,
        )
        for seed in task.seeds
    )


def rollout_dual_utility_loop_safe(
    env: SplendorEnv,
    scorer: Any,
    state: SplendorState,
    root_action: Action,
    root_player: int,
    seed: int,
    epsilon: float,
    temperature: float,
    top_k: int,
    max_additional_turns: int,
    score_margin_scale: float,
    max_loop_fallback_triggers: int,
    tiebreak_bonus: float,
) -> DualContinuationOutcome:
    rng = random.Random(seed)
    rollout_state = env.step(state, root_action)
    seen: dict[tuple[object, ...], int] = {}
    successor_counts: dict[tuple[object, ...], dict[tuple[object, ...], int]] = {}
    no_progress_turns = 0
    fallback_triggers = 0
    loop_adjudicated = False
    start_turn = rollout_state.turn_index
    end_turn = start_turn + max_additional_turns
    while not rollout_state.terminal and rollout_state.turn_index < end_turn:
        signature = state_signature(rollout_state)
        seen[signature] = seen.get(signature, 0) + 1
        scores = scorer.score_actions(rollout_state, env.legal_actions(rollout_state))
        if seen[signature] >= 4 or no_progress_turns >= 60:
            action = select_canonical_loop_fallback(
                rollout_state,
                scores,
                successor_counts.setdefault(signature, {}),
            )
            fallback_triggers += 1
        else:
            action = _sample_scored_action(
                scores,
                rng=rng,
                epsilon=epsilon,
                temperature=temperature,
                top_k=top_k,
            )
        successor_key = canonical_successor_action_key(rollout_state, action)
        counts = successor_counts.setdefault(signature, {})
        counts[successor_key] = counts.get(successor_key, 0) + 1
        next_state = env.step(rollout_state, action)
        no_progress_turns = (
            0
            if is_progress_transition(rollout_state, next_state, action)
            else no_progress_turns + 1
        )
        rollout_state = next_state
        if fallback_triggers >= max_loop_fallback_triggers:
            loop_adjudicated = True
            break
    margin = int(
        rollout_state.players[root_player].score
        - rollout_state.players[1 - root_player].score
    )
    original = normalized_score_margin(float(margin), score_margin_scale)
    corrected = original
    if rollout_state.terminal and margin == 0:
        if rollout_state.winner == root_player:
            corrected = tiebreak_bonus
        elif rollout_state.winner == 1 - root_player:
            corrected = -tiebreak_bonus
        else:
            corrected = 0.0
    return DualContinuationOutcome(
        original_value=original,
        tiebreak_value=corrected,
        terminal=rollout_state.terminal,
        resolved=rollout_state.terminal or loop_adjudicated,
        turns=rollout_state.turn_index - start_turn,
        fallback_triggers=fallback_triggers,
        loop_adjudicated=loop_adjudicated,
        hard_cap_reached=(
            not rollout_state.terminal and rollout_state.turn_index >= end_turn
        ),
        score_margin=margin,
        winner=rollout_state.winner,
        root_cards=len(rollout_state.players[root_player].purchased_cards),
        opponent_cards=len(rollout_state.players[1 - root_player].purchased_cards),
    )


def resolve_two_ply_tolerant(
    leaves: list[ResolvedLeaf],
    root_actions: tuple[tuple[object, ...], ...],
    determinization_count: int,
    *,
    payoff: str,
    tolerance: float,
) -> TwoPlyResolution:
    if payoff not in ("discovery_payoff", "evaluation_payoff"):
        raise ValueError("Unknown payoff table.")
    if determinization_count <= 0 or not root_actions or tolerance < 0.0:
        raise ValueError("Invalid tolerant resolver configuration.")
    values: list[float] = []
    choices: list[dict[str, Any]] = []
    for root_action in root_actions:
        root_leaves = [leaf for leaf in leaves if leaf.root_action == root_action]
        present = {leaf.determinization for leaf in root_leaves}
        if present != set(range(determinization_count)):
            raise ValueError("Every root action must exist in every determinization.")
        groups: dict[tuple[object, ...], list[ResolvedLeaf]] = defaultdict(list)
        for leaf in root_leaves:
            groups[leaf.opponent_information_set].append(leaf)
        root_total = 0.0
        for information_set, group in sorted(groups.items(), key=lambda item: repr(item[0])):
            determinizations = sorted({leaf.determinization for leaf in group})
            action_sets = {
                det: tuple(
                    sorted(
                        {
                            item.opponent_action
                            for item in group
                            if item.determinization == det
                        },
                        key=repr,
                    )
                )
                for det in determinizations
            }
            first_actions = action_sets[determinizations[0]]
            if any(actions != first_actions for actions in action_sets.values()):
                raise ValueError("Opponent information set has unstable semantic actions.")
            action_means = []
            for action in first_actions:
                records = [leaf for leaf in group if leaf.opponent_action == action]
                if len(records) != len(determinizations):
                    raise ValueError("Opponent action is missing from an information history.")
                action_means.append(
                    sum(float(getattr(leaf, payoff)) for leaf in records)
                    / len(records)
                )
            minimum = min(action_means)
            eligible = [
                index
                for index, value in enumerate(action_means)
                if value <= minimum + tolerance
            ]
            selected_index = min(eligible, key=lambda index: repr(first_actions[index]))
            selected_action = first_actions[selected_index]
            selected_records = [
                leaf for leaf in group if leaf.opponent_action == selected_action
            ]
            root_total += (
                sum(float(getattr(leaf, payoff)) for leaf in selected_records)
                / determinization_count
            )
            choices.append(
                {
                    "root_action": repr(root_action),
                    "opponent_information_set": repr(information_set),
                    "determinizations": determinizations,
                    "opponent_actions": [repr(action) for action in first_actions],
                    "action_means": action_means,
                    "selected_action": repr(selected_action),
                }
            )
        values.append(root_total)
    maximum = max(values)
    eligible_roots = [
        index for index, value in enumerate(values) if value >= maximum - tolerance
    ]
    best_index = min(eligible_roots)
    best_value = values[best_index]
    return TwoPlyResolution(
        root_actions=root_actions,
        action_values=tuple(values),
        regrets_from_best=tuple(value - best_value for value in values),
        best_index=best_index,
        opponent_choices=tuple(choices),
    )


def select_outcome_blind_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible_by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row["target_available"]):
            eligible_by_game[int(row["game_seed"])].append(row)
    games = sorted(eligible_by_game)
    if not games or len(games) % len(PHASES):
        raise ValueError("Game count must be positive and divisible by three.")
    selected = []
    for position, game_seed in enumerate(games):
        assigned_phase = PHASES[position % len(PHASES)]
        phase_rows = sorted(
            (
                row
                for row in eligible_by_game[game_seed]
                if str(row["phase"]) == assigned_phase
            ),
            key=lambda row: (int(row["decision_index"]), int(row["turn_index"])),
        )
        if not phase_rows:
            raise ValueError(
                f"Game {game_seed} has no completed-search row in {assigned_phase}."
            )
        chosen = phase_rows[len(phase_rows) // 2]
        selected.append({**chosen, "audit_assigned_phase": assigned_phase})
    return selected


def evaluate_selected_state(
    row: dict[str, Any],
    config: TiebreakAuditConfig,
    scorer: Any,
    executor: ProcessPoolExecutor | None,
) -> dict[str, Any]:
    resolver_config = ResolverGameplayConfig()
    state = deserialize_state_snapshot(row["state_snapshot"])
    env = SplendorEnv()
    codec = ActionCodec()
    legal_actions = env.legal_actions(state)
    legal_by_index = {codec.encode(state, action): action for action in legal_actions}
    stored_action_index = int(row["action_index"])
    parent_action_index = int(row["raw_parent_action_index"])
    if stored_action_index not in legal_by_index or parent_action_index not in legal_by_index:
        raise ValueError("Stored resolver or parent action is illegal.")
    ranked = _distinct_canonical_scores(state, scorer.score_actions(state, legal_actions))
    if len(ranked) < resolver_config.root_actions:
        raise ValueError("Selected completed-search row has fewer than four roots.")
    root_keys = tuple(
        canonical_successor_action_key(state, item.action)
        for item in ranked[: resolver_config.root_actions]
    )
    game_seed = int(row["game_seed"])
    actor_seat = int(row["player_id"])
    decision_index = int(row["decision_index"])
    root_index = gameplay_belief_root_index(
        resolver_config, game_seed, actor_seat, decision_index
    )
    determinizations = deterministic_belief_range(
        state,
        actor_seat,
        root_index=root_index,
        count=resolver_config.determinizations,
    )
    branches = []
    action_sets: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
    for det_index, determinized in enumerate(determinizations):
        root_action_map = actions_by_canonical_successor_key(
            determinized, env.legal_actions(determinized)
        )
        if any(key not in root_action_map for key in root_keys):
            raise ValueError("Root mapping changed in selected completed-search row.")
        for root_key in root_keys:
            after_root = env.step(determinized, root_action_map[root_key])
            if after_root.terminal:
                raise ValueError("Root action unexpectedly terminates selected row.")
            responses = _distinct_canonical_scores(
                after_root,
                scorer.score_actions(after_root, env.legal_actions(after_root)),
            )[: resolver_config.opponent_actions]
            if len(responses) != resolver_config.opponent_actions:
                raise ValueError("Selected row has fewer than four opponent responses.")
            response_keys = tuple(
                canonical_successor_action_key(after_root, item.action)
                for item in responses
            )
            information_set = (
                root_key,
                canonical_public_belief_key(after_root, after_root.current_player),
            )
            previous = action_sets.setdefault(information_set, response_keys)
            if previous != response_keys:
                raise ValueError("Opponent action set is unstable across determinizations.")
            branches.append(
                (det_index, root_key, after_root, information_set, responses)
            )
    leaf_specs = [
        (det_index, root_key, after_root, information_set, response)
        for det_index, root_key, after_root, information_set, responses in branches
        for response in responses
    ]
    tasks = [
        _DualLeafTask(
            index=index,
            after_root=after_root,
            response_action=response.action,
            actor_seat=actor_seat,
            seeds=tuple(
                gameplay_continuation_seed(
                    resolver_config,
                    game_seed,
                    actor_seat,
                    decision_index,
                    det_index,
                    continuation_index,
                )
                for continuation_index in range(
                    resolver_config.discovery_continuations
                )
            ),
            epsilon=resolver_config.continuation_epsilon,
            temperature=resolver_config.continuation_temperature,
            top_k=resolver_config.continuation_top_k,
            max_additional_turns=resolver_config.continuation_max_turns,
            score_margin_scale=resolver_config.score_margin_scale,
            max_loop_fallback_triggers=resolver_config.max_loop_fallback_triggers,
            tiebreak_bonus=config.tiebreak_value,
        )
        for index, (
            det_index,
            _root_key,
            after_root,
            _information_set,
            response,
        ) in enumerate(leaf_specs)
    ]
    if executor is None:
        ordered_outcomes = [
            tuple(
                rollout_dual_utility_loop_safe(
                    SplendorEnv(),
                    scorer,
                    task.after_root,
                    task.response_action,
                    task.actor_seat,
                    seed,
                    task.epsilon,
                    task.temperature,
                    task.top_k,
                    task.max_additional_turns,
                    task.score_margin_scale,
                    task.max_loop_fallback_triggers,
                    task.tiebreak_bonus,
                )
                for seed in task.seeds
            )
            for task in tasks
        ]
    else:
        indexed = executor.map(_evaluate_task, tasks)
        ordered: list[tuple[DualContinuationOutcome, ...] | None] = [None] * len(tasks)
        for index, outcomes in indexed:
            ordered[index] = outcomes
        if any(item is None for item in ordered):
            raise RuntimeError("076 parallel worker returned an incomplete leaf table.")
        ordered_outcomes = [item for item in ordered if item is not None]
    original_leaves = []
    tiebreak_leaves = []
    branch_records = []
    for leaf_spec, outcomes in zip(leaf_specs, ordered_outcomes, strict=True):
        det_index, root_key, after_root, information_set, response = leaf_spec
        original_value = statistics.fmean(item.original_value for item in outcomes)
        tiebreak_value = statistics.fmean(item.tiebreak_value for item in outcomes)
        response_key = canonical_successor_action_key(after_root, response.action)
        original_leaves.append(
            ResolvedLeaf(
                det_index,
                root_key,
                information_set,
                response_key,
                original_value,
                original_value,
            )
        )
        tiebreak_leaves.append(
            ResolvedLeaf(
                det_index,
                root_key,
                information_set,
                response_key,
                tiebreak_value,
                tiebreak_value,
            )
        )
        for continuation_index, outcome in enumerate(outcomes):
            branch_records.append(
                {
                    "determinization": det_index,
                    "root_action": repr(root_key),
                    "opponent_information_set": repr(information_set),
                    "opponent_action": repr(response_key),
                    "continuation_index": continuation_index,
                    **outcome.to_record(),
                }
            )
    original_exact = resolve_two_ply(
        original_leaves,
        root_keys,
        resolver_config.determinizations,
        payoff="discovery_payoff",
    )
    original_tolerant = resolve_two_ply_tolerant(
        original_leaves,
        root_keys,
        resolver_config.determinizations,
        payoff="discovery_payoff",
        tolerance=config.comparison_tolerance,
    )
    tiebreak_exact = resolve_two_ply(
        tiebreak_leaves,
        root_keys,
        resolver_config.determinizations,
        payoff="discovery_payoff",
    )
    combined = resolve_two_ply_tolerant(
        tiebreak_leaves,
        root_keys,
        resolver_config.determinizations,
        payoff="discovery_payoff",
        tolerance=config.comparison_tolerance,
    )
    resolutions = {
        "original_exact": original_exact.to_payload(),
        "original_tolerant": original_tolerant.to_payload(),
        "tiebreak_exact": tiebreak_exact.to_payload(),
        "combined": combined.to_payload(),
    }
    branch_bytes = _canonical_json_bytes(branch_records)
    original_action_map = actions_by_canonical_successor_key(state, legal_actions)
    original_action = original_action_map[root_keys[original_exact.best_index]]
    corrected_action = original_action_map[root_keys[combined.best_index]]
    original_after = env.step(state, original_action)
    corrected_after = env.step(state, corrected_action)
    actor = state.current_player
    original_score_gain = (
        original_after.players[actor].score - state.players[actor].score
    )
    corrected_score_gain = (
        corrected_after.players[actor].score - state.players[actor].score
    )
    original_card_gain = len(original_after.players[actor].purchased_cards) - len(
        state.players[actor].purchased_cards
    )
    corrected_card_gain = len(corrected_after.players[actor].purchased_cards) - len(
        state.players[actor].purchased_cards
    )
    stored_values = [
        float(item["value"]) for item in row.get("resolver_action_values", [])
    ]
    stored_roots = [
        str(item["canonical_successor"])
        for item in row.get("resolver_action_values", [])
    ]
    stored_selected_key = canonical_successor_action_key(
        state, legal_by_index[stored_action_index]
    )
    reproduction = {
        "root_actions_match": stored_roots == [repr(key) for key in root_keys],
        "action_values_match": len(stored_values) == len(original_exact.action_values)
        and all(
            abs(stored - observed) <= 1e-12
            for stored, observed in zip(
                stored_values, original_exact.action_values, strict=True
            )
        ),
        "selected_root_matches": stored_selected_key
        == root_keys[original_exact.best_index],
        "parent_root_matches": canonical_successor_action_key(
            state, legal_by_index[parent_action_index]
        )
        == root_keys[0],
    }
    stats = summarize_branch_records(branch_records, actor_seat)
    return {
        "source_split": str(row["source_split"]),
        "game_seed": game_seed,
        "turn_index": int(row["turn_index"]),
        "decision_index": decision_index,
        "player_id": actor_seat,
        "phase": str(row["audit_assigned_phase"]),
        "source_row_phase": str(row["phase"]),
        "root_actions": [repr(key) for key in root_keys],
        "resolutions": resolutions,
        "selected_indices": {
            "original_exact": original_exact.best_index,
            "original_tolerant": original_tolerant.best_index,
            "tiebreak_exact": tiebreak_exact.best_index,
            "combined": combined.best_index,
        },
        "changes": {
            "tolerance_only": original_tolerant.best_index
            != original_exact.best_index,
            "tiebreak_only": tiebreak_exact.best_index != original_exact.best_index,
            "combined": combined.best_index != original_exact.best_index,
            "combined_from_parent": combined.best_index != 0,
        },
        "action_diagnostic": {
            "transition": (
                f"{original_action.action_type.name}->"
                f"{corrected_action.action_type.name}"
            ),
            "original_score_gain": original_score_gain,
            "corrected_score_gain": corrected_score_gain,
            "original_card_gain": original_card_gain,
            "corrected_card_gain": corrected_card_gain,
            "corrected_more_immediate_prestige": corrected_score_gain
            > original_score_gain,
            "corrected_fewer_immediate_cards": corrected_card_gain
            < original_card_gain,
        },
        "branch_statistics": stats,
        "branch_records_sha256": hashlib.sha256(branch_bytes).hexdigest(),
        "reproduction": reproduction,
        "finite": all(
            math.isfinite(value)
            for resolution in (
                original_exact,
                original_tolerant,
                tiebreak_exact,
                combined,
            )
            for value in resolution.action_values
        ),
    }


def summarize_branch_records(
    records: Sequence[dict[str, Any]], actor_seat: int
) -> dict[str, Any]:
    tied = [record for record in records if record["terminal"] and record["score_margin"] == 0]
    root_wins = sum(record["winner"] == actor_seat for record in tied)
    opponent_wins = sum(record["winner"] == 1 - actor_seat for record in tied)
    true_draws = sum(record["winner"] is None for record in tied)
    return {
        "branches": len(records),
        "terminal": sum(bool(record["terminal"]) for record in records),
        "resolved": sum(bool(record["resolved"]) for record in records),
        "hard_caps": sum(bool(record["hard_cap_reached"]) for record in records),
        "loop_adjudications": sum(
            bool(record["loop_adjudicated"]) for record in records
        ),
        "fallback_triggers": sum(int(record["fallback_triggers"]) for record in records),
        "equal_prestige_terminal": len(tied),
        "equal_prestige_root_wins": root_wins,
        "equal_prestige_opponent_wins": opponent_wins,
        "equal_prestige_true_draws": true_draws,
        "utility_changed_branches": root_wins + opponent_wins,
        "nonzero_margin_values_identical": all(
            record["score_margin"] == 0
            or record["original_value"] == record["tiebreak_value"]
            for record in records
        ),
    }


def summarize_state_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = ("tolerance_only", "tiebreak_only", "combined")
    changes = {
        name: sum(bool(record["changes"][name]) for record in records)
        for name in variants
    }
    total = len(records)
    primary_interval = clopper_pearson(changes["combined"], total)
    by_phase = {}
    for phase in PHASES:
        phase_rows = [record for record in records if record["phase"] == phase]
        by_phase[phase] = {
            "states": len(phase_rows),
            **{
                f"{name}_changes": sum(
                    bool(record["changes"][name]) for record in phase_rows
                )
                for name in variants
            },
            **{
                f"{name}_fraction": statistics.fmean(
                    float(record["changes"][name]) for record in phase_rows
                )
                for name in variants
            },
        }
    transitions = Counter(
        record["action_diagnostic"]["transition"]
        for record in records
        if record["changes"]["combined"]
    )
    branch_totals = Counter()
    for record in records:
        for key, value in record["branch_statistics"].items():
            if isinstance(value, int):
                branch_totals[key] += value
    state_utility_changed_fractions = [
        record["branch_statistics"]["utility_changed_branches"]
        / record["branch_statistics"]["branches"]
        for record in records
    ]
    combined_changed = [
        record for record in records if record["changes"]["combined"]
    ]
    fraction = changes["combined"] / total
    classification = (
        "isolated_low_dose_correctness_repair"
        if fraction < 0.05
        else "nontrivial_subthreshold_correction"
        if fraction < 0.15
        else "material_resolver_intervention"
    )
    return {
        "states": total,
        "changes": changes,
        "fractions": {name: changes[name] / total for name in variants},
        "primary_combined_change": {
            "count": changes["combined"],
            "fraction": fraction,
            "clopper_pearson_95": list(primary_interval),
            "classification": classification,
        },
        "by_phase": by_phase,
        "combined_action_transitions": dict(sorted(transitions.items())),
        "combined_corrected_more_immediate_prestige": sum(
            bool(record["action_diagnostic"]["corrected_more_immediate_prestige"])
            for record in combined_changed
        ),
        "combined_corrected_fewer_immediate_cards": sum(
            bool(record["action_diagnostic"]["corrected_fewer_immediate_cards"])
            for record in combined_changed
        ),
        "branch_totals": dict(branch_totals),
        "branch_fractions": {
            key: branch_totals[key] / branch_totals["branches"]
            for key in (
                "equal_prestige_terminal",
                "equal_prestige_root_wins",
                "equal_prestige_opponent_wins",
                "equal_prestige_true_draws",
                "utility_changed_branches",
            )
        },
        "per_state_utility_changed_branch_fraction": {
            "mean": statistics.fmean(state_utility_changed_fractions),
            "median": statistics.median(state_utility_changed_fractions),
            "minimum": min(state_utility_changed_fractions),
            "maximum": max(state_utility_changed_fractions),
        },
    }


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Invalid binomial counts.")
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def run_audit(config: TiebreakAuditConfig = TiebreakAuditConfig()) -> dict[str, Any]:
    _validate_frozen_config(config)
    identity = _validate_identity(config)
    rows, source_counts = _load_rows(config)
    selected = select_outcome_blind_rows(rows)
    if len(selected) != EXPECTED_STATES:
        raise ValueError(f"076 requires {EXPECTED_STATES} selected states.")
    selection = [
        {
            "source_split": row["source_split"],
            "game_seed": int(row["game_seed"]),
            "phase": row["audit_assigned_phase"],
            "decision_index": int(row["decision_index"]),
            "turn_index": int(row["turn_index"]),
        }
        for row in selected
    ]
    selection_sha256 = hashlib.sha256(_canonical_json_bytes(selection)).hexdigest()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity or result.get("selection_sha256") != selection_sha256:
            raise ValueError("Existing 076 result does not match frozen identity.")
        return result
    progress_path = config.output_dir / "progress.jsonl"
    records = _load_progress(progress_path, selection_sha256)
    scorer = _LegalActionScorer(
        config.checkpoint, config.device, "legal_action_attention_scorer"
    )
    serial_repeat_identical = None
    verification_branches = 0
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_initialize_worker,
        initargs=(config.checkpoint, config.device),
    ) as executor:
        for index, row in enumerate(selected):
            if index < len(records):
                continue
            print(
                f"[076] state {index + 1}/{len(selected)} seed={row['game_seed']} "
                f"phase={row['audit_assigned_phase']}",
                flush=True,
            )
            record = evaluate_selected_state(row, config, scorer, executor)
            if index == 0:
                repeated = evaluate_selected_state(row, config, scorer, None)
                verification_branches = EXPECTED_BRANCHES_PER_STATE
                serial_repeat_identical = (
                    record["branch_records_sha256"]
                    == repeated["branch_records_sha256"]
                    and record["resolutions"] == repeated["resolutions"]
                )
            record["selection_sha256"] = selection_sha256
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            records.append(record)
    if serial_repeat_identical is None:
        # A resumed complete progress file has already passed per-state determinism;
        # repeat the first state once to re-establish this run-level gate.
        repeated = evaluate_selected_state(selected[0], config, scorer, None)
        verification_branches = EXPECTED_BRANCHES_PER_STATE
        serial_repeat_identical = (
            records[0]["branch_records_sha256"] == repeated["branch_records_sha256"]
            and records[0]["resolutions"] == repeated["resolutions"]
        )
    summary = summarize_state_records(records)
    totals = summary["branch_totals"]
    phase_counts = Counter(record["phase"] for record in records)
    integrity_checks = {
        "source_identity_and_collection_integrity": identity["collection_integrity_passed"],
        "exact_source_counts": source_counts
        == {
            "rows": EXPECTED_ROWS,
            "games": EXPECTED_GAMES,
        },
        "exact_selection_and_phase_balance": len(records) == EXPECTED_STATES
        and len({record["game_seed"] for record in records}) == EXPECTED_STATES
        and phase_counts == Counter({phase: 40 for phase in PHASES}),
        "declared_selection_only": True,
        "legal_canonical_reproduction": all(
            all(record["reproduction"].values()) for record in records
        ),
        "exact_main_branch_count": totals["branches"] == EXPECTED_BRANCHES,
        "all_branches_resolved": totals["resolved"] == EXPECTED_BRANCHES,
        "zero_hard_caps": totals["hard_caps"] == 0,
        "loop_adjudications_within_limit": totals["loop_adjudications"]
        / EXPECTED_BRANCHES
        <= config.maximum_loop_adjudicated_fraction,
        "finite_resolutions": all(record["finite"] for record in records),
        "same_branch_dual_utility": all(
            record["branch_statistics"]["nonzero_margin_values_identical"]
            for record in records
        ),
        "serial_parallel_first_state_identical": bool(serial_repeat_identical),
        "human_data_not_configured": "human" not in str(config.source_dir).lower()
        and "human" not in str(config.checkpoint).lower(),
    }
    integrity_passed = all(integrity_checks.values())
    result = {
        "experiment": "076_resolver_tiebreak_utility_audit",
        "protocol": "docs/experiments/resolver_tiebreak_utility_audit_076.md",
        "configuration": _config_payload(config),
        "identity": identity,
        "source_counts": source_counts,
        "selection": selection,
        "selection_sha256": selection_sha256,
        "state_records": records,
        "summary": summary,
        "verification_continuations": verification_branches,
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "decision": (
            summary["primary_combined_change"]["classification"]
            if integrity_passed
            else "invalid_no_correction_dose_conclusion"
        ),
        "production_change_authorized": False,
        "gameplay_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "human_data_used": False,
    }
    _atomic_write(result_path, result)
    print(
        json.dumps(
            {
                "integrity_passed": integrity_passed,
                "decision": result["decision"],
                "primary": summary["primary_combined_change"],
                "branch_fractions": summary["branch_fractions"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def _validate_frozen_config(config: TiebreakAuditConfig) -> None:
    expected = TiebreakAuditConfig(output_dir=config.output_dir)
    for name in (
        "source_dir",
        "checkpoint",
        "tiebreak_value",
        "comparison_tolerance",
        "maximum_loop_adjudicated_fraction",
        "workers",
        "device",
    ):
        if getattr(config, name) != getattr(expected, name):
            raise ValueError(f"076 frozen configuration mismatch for {name}.")


def _validate_identity(config: TiebreakAuditConfig) -> dict[str, Any]:
    paths = {
        "result": config.source_dir / "result.json",
        "train": config.source_dir / "train_targets.jsonl",
        "validation": config.source_dir / "validation_targets.jsonl",
        "confirmation": config.source_dir / "confirmation_targets.jsonl",
        "parent_035": config.checkpoint,
    }
    observed = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256_file(path)
        if observed[name] != SOURCE_SHA256[name]:
            raise ValueError(f"076 frozen identity mismatch for {name}.")
    source_result = json.loads(paths["result"].read_text(encoding="utf-8"))
    passed = all(
        bool(source_result[split]["passed"])
        and int(source_result[split]["integrity"]["invalid_games"]) == 0
        and int(source_result[split]["integrity"]["stalled_games"]) == 0
        and int(source_result[split]["integrity"]["timed_out_games"]) == 0
        for split in ("train", "validation", "confirmation")
    )
    if not passed:
        raise ValueError("Frozen 052 collection integrity did not pass.")
    return {
        "expected_sha256": SOURCE_SHA256,
        "observed_sha256": observed,
        "collection_integrity_passed": passed,
    }


def _load_rows(
    config: TiebreakAuditConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    rows = []
    row_counts = {}
    game_counts = {}
    for split in ("train", "validation", "confirmation"):
        split_rows = []
        with (config.source_dir / f"{split}_targets.jsonl").open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if bool(row["target_available"]):
                    row["source_split"] = split
                    split_rows.append(row)
        row_counts[split] = len(split_rows)
        game_counts[split] = len({int(row["game_seed"]) for row in split_rows})
        rows.extend(split_rows)
    return rows, {"rows": row_counts, "games": game_counts}


def _load_progress(path: Path, selection_sha256: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(record.get("selection_sha256") != selection_sha256 for record in records):
        raise ValueError("Existing 076 progress has the wrong selection identity.")
    if len(records) > EXPECTED_STATES:
        raise ValueError("Existing 076 progress has too many records.")
    return records


def _config_payload(config: TiebreakAuditConfig) -> dict[str, Any]:
    payload = asdict(config)
    for name in ("source_dir", "checkpoint", "output_dir"):
        payload[name] = str(payload[name])
    payload["resolver_contract"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(ResolverGameplayConfig()).items()
    }
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/resolver_tiebreak_utility_audit_076"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_audit(TiebreakAuditConfig(output_dir=Path(args.output_dir)))


if __name__ == "__main__":
    main()
