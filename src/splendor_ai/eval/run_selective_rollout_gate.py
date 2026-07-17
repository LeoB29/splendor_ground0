"""Gate a selective direct matched-rollout wrapper around frozen policy 035."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.encoding import (
    ActionCodec,
    canonical_public_belief_key,
    canonical_successor_action_key,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_shallow_operational_gate import summarize_pair_rows
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_league_generation import OPPONENTS, _sha256
from splendor_ai.training.run_policy_delta_causal_audit import (
    PHASES,
    PolicyDeltaCausalAuditConfig,
    _phase,
    _source_replay_path,
    _source_summary_path,
    _stratum,
    _stratum_cell,
    collect_fresh_source_games,
    outcome_blind_replay_projection,
)
from splendor_ai.training.run_policy_search_disagreement import (
    ModelActionScore,
    _LegalActionScorer,
)


_CONFIRMATION_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class SelectiveRolloutGateConfig:
    output_dir: Path
    data_dir: Path
    development_evaluation: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    threshold_grid: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10)
    routing_logit_gap: float = 0.5
    source_games: int = 960
    states_per_stratum: int = 20
    collection_seed_start: int = 372_000
    continuation_seed_start: int = 38_000_000
    gameplay_rollout_seed_start: int = 39_000_000
    direct_seed_start: int = 374_000
    determinizations: int = 3
    continuations_per_half: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_additional_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    minimum_switch_fraction: float = 0.05
    maximum_switch_fraction: float = 0.30
    minimum_switches: int = 20
    minimum_evaluation_positive_fraction: float = 0.60
    minimum_breakdown_gain: float = -0.025
    direct_pairs: int = 200
    minimum_direct_score_share: float = 0.55
    maximum_search_fraction: float = 0.10
    max_searches_per_game: int = 2
    maximum_loop_adjudicated_fraction: float = 0.001
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    device: str = "cpu"


def run_selective_rollout_gate(config: SelectiveRolloutGateConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing selective-rollout protocol does not match.")
    else:
        _atomic_write(protocol_path, protocol)

    calibration = calibrate_development_threshold(config)
    if not calibration["passed"]:
        return _finalize(config, "development_gate_failed", calibration=calibration)

    source_config = _source_config(config)
    collection = collect_fresh_source_games(source_config)
    if not collection["passed"]:
        return _finalize(
            config,
            "fresh_collection_failed",
            calibration=calibration,
            collection=collection,
        )
    selection = select_fresh_uncertain_states(config, source_config)
    if not selection["passed"]:
        return _finalize(
            config,
            "fresh_selection_failed",
            calibration=calibration,
            collection=collection,
            selection=selection,
        )
    confirmation = evaluate_fresh_action_quality(
        config,
        selection,
        float(calibration["selected_threshold"]),
    )
    if not confirmation["passed"]:
        return _finalize(
            config,
            "fresh_action_quality_failed",
            calibration=calibration,
            collection=collection,
            selection=selection,
            confirmation=confirmation,
        )
    direct = run_direct_gameplay_gate(
        config,
        float(calibration["selected_threshold"]),
    )
    return _finalize(
        config,
        "direct_gate_passed_stronger_teacher"
        if direct["passed"]
        else "direct_gameplay_failed",
        calibration=calibration,
        collection=collection,
        selection=selection,
        confirmation=confirmation,
        direct=direct,
    )


def threshold_metrics(
    rows: list[dict[str, Any]],
    threshold: float,
    bootstrap_samples: int,
    seed: int,
    *,
    symmetric: bool,
) -> dict[str, Any]:
    state_gains: list[float] = []
    switched_evaluation_values: list[float] = []
    switch_count = 0
    orientations = 2 if symmetric else 1
    for row in rows:
        first = float(row["candidate_minus_parent_first"])
        second = float(row["candidate_minus_parent_second"])
        pairs = ((first, second), (second, first)) if symmetric else ((first, second),)
        gains = []
        for discovery, evaluation in pairs:
            switched = discovery >= threshold
            gains.append(evaluation if switched else 0.0)
            if switched:
                switch_count += 1
                switched_evaluation_values.append(evaluation)
        state_gains.append(statistics.fmean(gains))
    opportunities = len(rows) * orientations
    gain_ci = _mean_ci(state_gains, bootstrap_samples, seed)
    switched_ci = (
        _mean_ci(switched_evaluation_values, bootstrap_samples, seed + 1)
        if switched_evaluation_values
        else {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    )
    return {
        "rows": len(rows),
        "orientations_per_row": orientations,
        "threshold": threshold,
        "switches": switch_count,
        "switch_fraction": switch_count / opportunities if opportunities else 0.0,
        "evaluation_gain": gain_ci,
        "switched_evaluation_gain": switched_ci,
        "evaluation_positive_fraction": (
            statistics.fmean(value > 0.0 for value in switched_evaluation_values)
            if switched_evaluation_values
            else 0.0
        ),
    }


def calibrate_development_threshold(config: SelectiveRolloutGateConfig) -> dict[str, Any]:
    path = config.output_dir / "development_calibration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    development_rows = _read_development_rows_without_confirmation(
        config.development_evaluation
    )
    train_rows = [row for row in development_rows if row["split"] == "train"]
    validation_rows = [row for row in development_rows if row["split"] == "validation"]
    candidates = [
        threshold_metrics(
            train_rows,
            threshold,
            config.bootstrap_samples,
            config.collection_seed_start + index,
            symmetric=True,
        )
        for index, threshold in enumerate(config.threshold_grid)
    ]
    eligible = [
        row
        for row in candidates
        if config.minimum_switch_fraction <= row["switch_fraction"]
        <= config.maximum_switch_fraction
        and int(row["switches"]) >= config.minimum_switches
        and float(row["evaluation_gain"]["mean"]) > 0.0
    ]
    if not eligible:
        result = {
            "format": "selective_rollout_development_calibration_v1",
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "candidates": candidates,
            "selected_threshold": None,
            "validation": None,
            "gates": {"eligible_train_threshold": False},
            "passed": False,
        }
        _atomic_write(path, result)
        return result
    selected = max(
        eligible,
        key=lambda row: (float(row["evaluation_gain"]["mean"]), float(row["threshold"])),
    )
    validation = threshold_metrics(
        validation_rows,
        float(selected["threshold"]),
        config.bootstrap_samples,
        config.collection_seed_start + 101,
        symmetric=True,
    )
    gates = {
        "eligible_train_threshold": True,
        "validation_rows_complete": len(validation_rows) == 384,
        "validation_switch_fraction": config.minimum_switch_fraction
        <= validation["switch_fraction"]
        <= config.maximum_switch_fraction,
        "validation_minimum_switches": int(validation["switches"])
        >= config.minimum_switches,
        "validation_gain_ci_above_zero": float(validation["evaluation_gain"]["ci95_low"])
        > 0.0,
        "validation_positive_fraction": float(validation["evaluation_positive_fraction"])
        >= config.minimum_evaluation_positive_fraction,
    }
    result = {
        "format": "selective_rollout_development_calibration_v1",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "candidates": candidates,
        "selected_threshold": float(selected["threshold"]),
        "validation": validation,
        "gates": gates,
        "passed": all(gates.values()),
        "confirmation_rows_used": False,
    }
    _atomic_write(path, result)
    return result


def _read_development_rows_without_confirmation(path: Path) -> list[dict[str, Any]]:
    """Stream the JSONL and reject confirmation lines before parsing outcomes."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if '"split": "confirmation"' in line or '"split":"confirmation"' in line:
                continue
            payload = json.loads(line)
            row = payload.get("row")
            if row is not None and row.get("split") in ("train", "validation"):
                rows.append(row)
    return rows


def select_fresh_uncertain_states(
    config: SelectiveRolloutGateConfig,
    source_config: PolicyDeltaCausalAuditConfig,
) -> dict[str, Any]:
    path = config.output_dir / "fresh_selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    scorer = _LegalActionScorer(
        config.parent_checkpoint,
        config.device,
        "legal_action_attention_scorer",
    )
    env = SplendorEnv()
    candidates: dict[str, list[dict[str, Any]]] = {
        _stratum(opponent, seat, phase): []
        for opponent in OPPONENTS
        for seat in (0, 1)
        for phase in PHASES
    }
    rows_scanned = 0
    for game_index in range(config.source_games):
        summary = json.loads(
            _source_summary_path(source_config, game_index).read_text(encoding="utf-8")
        )
        opponent = str(summary["opponent"])
        actor_seat = int(summary["actor_seat"])
        assigned_phase = PHASES[(game_index // (len(OPPONENTS) * 2)) % len(PHASES)]
        best: dict[str, Any] | None = None
        with _source_replay_path(source_config, game_index).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                observation = outcome_blind_replay_projection(json.loads(line))
                if int(observation["player_id"]) != actor_seat:
                    continue
                turn_index = int(observation["turn_index"])
                if _phase(turn_index) != assigned_phase:
                    continue
                rows_scanned += 1
                state = deserialize_state_snapshot(observation["state_snapshot"])
                ranked = distinct_canonical_scores(
                    state,
                    scorer.score_actions(state, env.legal_actions(state)),
                )
                if len(ranked) < 2:
                    continue
                parent, challenger = ranked[:2]
                gap = float(parent.score - challenger.score)
                if gap > config.routing_logit_gap:
                    continue
                row = {
                    "source_game_index": game_index,
                    "source_game_seed": int(observation["game_seed"]),
                    "turn_index": turn_index,
                    "phase": assigned_phase,
                    "actor_seat": actor_seat,
                    "opponent": opponent,
                    "source_cell": _stratum_cell(opponent, actor_seat),
                    "stratum": _stratum(opponent, actor_seat, assigned_phase),
                    "parent_action_index": parent.action_index,
                    "parent_action": serialize_action(parent.action),
                    "candidate_action_index": challenger.action_index,
                    "candidate_action": serialize_action(challenger.action),
                    "top_two_logit_gap": gap,
                    "state_snapshot": observation["state_snapshot"],
                }
                if best is None or (gap, turn_index) < (
                    float(best["top_two_logit_gap"]),
                    int(best["turn_index"]),
                ):
                    best = row
        if best is not None:
            candidates[str(best["stratum"])].append(best)
    selected = []
    for stratum in sorted(candidates):
        rows = sorted(
            candidates[stratum],
            key=lambda row: (
                float(row["top_two_logit_gap"]),
                int(row["source_game_seed"]),
            ),
        )
        selected.extend(rows[: config.states_per_stratum])
    selected.sort(
        key=lambda row: (
            str(row["stratum"]),
            int(row["source_game_seed"]),
        )
    )
    counts = {
        stratum: sum(row["stratum"] == stratum for row in selected)
        for stratum in sorted(candidates)
    }
    expected = len(OPPONENTS) * 2 * len(PHASES) * config.states_per_stratum
    gates = {
        "exact_state_count": len(selected) == expected,
        "unique_games": len({row["source_game_index"] for row in selected})
        == len(selected),
        "exact_stratum_balance": set(counts.values()) == {config.states_per_stratum},
        "all_within_routing_gap": all(
            float(row["top_two_logit_gap"]) <= config.routing_logit_gap
            for row in selected
        ),
        "outcome_blind_selection": True,
    }
    result = {
        "format": "selective_rollout_fresh_selection_v1",
        "rows_scanned": rows_scanned,
        "eligible_by_stratum": {key: len(value) for key, value in candidates.items()},
        "selected_states": len(selected),
        "selected_by_stratum": counts,
        "selected": selected,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_write(path, result)
    return result


def distinct_canonical_scores(
    state: SplendorState,
    scores: tuple[ModelActionScore, ...],
) -> tuple[ModelActionScore, ...]:
    rows = []
    seen = set()
    for row in scores:
        key = canonical_successor_action_key(state, row.action)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return tuple(rows)


def evaluate_fresh_action_quality(
    config: SelectiveRolloutGateConfig,
    selection: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    result_path = config.output_dir / "fresh_action_quality.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    progress_path = config.output_dir / "fresh_action_quality.progress.jsonl"
    header, completed = _load_progress(progress_path)
    selected = list(selection["selected"])
    expected_header = {
        "format": "selective_rollout_fresh_progress_v1",
        "protocol_sha256": _sha256(config.output_dir / "protocol.json"),
        "selection_sha256": _sha256(config.output_dir / "fresh_selection.json"),
        "threshold": threshold,
        "states": len(selected),
    }
    if header is None:
        progress_path.write_text(json.dumps({"header": expected_header}) + "\n", encoding="utf-8")
    elif header != expected_header:
        raise ValueError("Fresh action-quality progress does not match the protocol.")
    missing = [(index, selected[index]) for index in range(len(selected)) if index not in completed]
    print(
        f"[selective-rollout:fresh] states={len(selected)} complete={len(completed)} "
        f"pending={len(missing)} workers={config.workers}",
        flush=True,
    )
    if missing:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_confirmation_worker,
            initargs=(str(config.parent_checkpoint), config.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_confirmation_state, item, config, threshold): item[0]
                for item in missing
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    index, row = future.result()
                    completed[index] = row
                    handle.write(json.dumps({"state_index": index, "row": row}) + "\n")
                    handle.flush()
                    if len(completed) % 20 == 0 or len(completed) == len(selected):
                        print(
                            f"[selective-rollout:fresh] complete={len(completed)}/{len(selected)}",
                            flush=True,
                        )
    rows = [completed[index] for index in range(len(selected))]
    metrics = threshold_metrics(
        rows,
        threshold,
        config.bootstrap_samples,
        config.collection_seed_start + 301,
        symmetric=False,
    )
    breakdowns = _fresh_breakdowns(rows, threshold)
    total_continuations = sum(int(row["total_continuations"]) for row in rows)
    adjudicated = sum(int(row["loop_adjudicated_continuations"]) for row in rows)
    gates = {
        "complete": len(rows) == len(selected),
        "all_continuations_resolved": sum(int(row["resolved_continuations"]) for row in rows)
        == total_continuations,
        "zero_hard_caps": sum(int(row["hard_cap_continuations"]) for row in rows) == 0,
        "bounded_loop_adjudication": adjudicated / total_continuations
        <= config.maximum_loop_adjudicated_fraction,
        "switch_fraction": config.minimum_switch_fraction
        <= metrics["switch_fraction"]
        <= config.maximum_switch_fraction,
        "minimum_switches": int(metrics["switches"]) >= config.minimum_switches,
        "evaluation_gain_ci_above_zero": float(metrics["evaluation_gain"]["ci95_low"])
        > 0.0,
        "positive_fraction": float(metrics["evaluation_positive_fraction"])
        >= config.minimum_evaluation_positive_fraction,
        "breakdown_breadth": all(
            int(bucket["switches"]) >= 3
            and float(bucket["evaluation_gain_mean"])
            >= config.minimum_breakdown_gain
            for category in breakdowns.values()
            for bucket in category.values()
        ),
    }
    result = {
        "format": "selective_rollout_fresh_action_quality_v1",
        "threshold": threshold,
        "metrics": metrics,
        "breakdowns": breakdowns,
        "total_continuations": total_continuations,
        "fallback_triggers": sum(int(row["fallback_triggers"]) for row in rows),
        "loop_adjudicated_continuations": adjudicated,
        "hard_cap_continuations": sum(int(row["hard_cap_continuations"]) for row in rows),
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }
    _atomic_write(result_path, result)
    return result


def _initialize_confirmation_worker(parent_checkpoint: str, device: str) -> None:
    global _CONFIRMATION_SCORER
    torch.set_num_threads(1)
    _CONFIRMATION_SCORER = _LegalActionScorer(
        Path(parent_checkpoint),
        device,
        "legal_action_attention_scorer",
    )


def _evaluate_confirmation_state(
    item: tuple[int, dict[str, Any]],
    config: SelectiveRolloutGateConfig,
    threshold: float,
) -> tuple[int, dict[str, Any]]:
    if _CONFIRMATION_SCORER is None:
        raise RuntimeError("Selective-rollout worker was not initialized.")
    state_index, payload = item
    first, second, counters = evaluate_action_pair(
        deserialize_state_snapshot(payload["state_snapshot"]),
        int(payload["actor_seat"]),
        int(payload["parent_action_index"]),
        int(payload["candidate_action_index"]),
        _CONFIRMATION_SCORER,
        config,
        config.continuation_seed_start + state_index * 10_000_019,
    )
    return state_index, {
        **{
            key: payload[key]
            for key in (
                "source_game_index",
                "source_game_seed",
                "turn_index",
                "phase",
                "actor_seat",
                "opponent",
                "source_cell",
                "stratum",
                "top_two_logit_gap",
                "parent_action_index",
                "parent_action",
                "candidate_action_index",
                "candidate_action",
            )
        },
        "candidate_minus_parent_first": first,
        "candidate_minus_parent_second": second,
        "switched": first >= threshold,
        **counters,
    }


def evaluate_action_pair(
    state: SplendorState,
    actor_seat: int,
    parent_action_index: int,
    candidate_action_index: int,
    scorer: _LegalActionScorer,
    config: SelectiveRolloutGateConfig,
    seed_base: int,
) -> tuple[float, float, dict[str, int]]:
    env = SplendorEnv()
    codec = ActionCodec()
    halves: tuple[list[float], list[float]] = ([], [])
    counters = {
        "resolved_continuations": 0,
        "natural_terminal_continuations": 0,
        "total_continuations": 0,
        "fallback_triggers": 0,
        "loop_adjudicated_continuations": 0,
        "hard_cap_continuations": 0,
    }
    for determinization_index in range(config.determinizations):
        determinized = determinize_for_observer(
            state,
            actor_seat,
            random.Random(seed_base + determinization_index * 1_000_003),
        )
        action_by_index = {
            codec.encode(determinized, action): action
            for action in env.legal_actions(determinized)
        }
        if parent_action_index not in action_by_index or candidate_action_index not in action_by_index:
            raise RuntimeError("Selected action is absent after legal determinization.")
        for continuation_index in range(config.continuations_per_half * 2):
            seed = seed_base + determinization_index * 100_003 + continuation_index
            results = [
                rollout_score_margin_loop_safe(
                    env,
                    scorer,
                    determinized,
                    action_by_index[action_index],
                    actor_seat,
                    seed,
                    config.continuation_epsilon,
                    config.continuation_temperature,
                    config.continuation_top_k,
                    config.continuation_max_additional_turns,
                    config.score_margin_scale,
                    config.max_loop_fallback_triggers,
                )
                for action_index in (parent_action_index, candidate_action_index)
            ]
            half = int(continuation_index >= config.continuations_per_half)
            halves[half].append(results[1].value - results[0].value)
            counters["resolved_continuations"] += sum(int(result.resolved) for result in results)
            counters["natural_terminal_continuations"] += sum(int(result.terminal) for result in results)
            counters["total_continuations"] += 2
            counters["fallback_triggers"] += sum(result.fallback_triggers for result in results)
            counters["loop_adjudicated_continuations"] += sum(
                int(result.loop_adjudicated) for result in results
            )
            counters["hard_cap_continuations"] += sum(
                int(result.hard_cap_reached) for result in results
            )
    return statistics.fmean(halves[0]), statistics.fmean(halves[1]), counters


class SelectiveMatchedRolloutBot:
    """Use frozen 035 except for a bounded number of direct rollout comparisons."""

    def __init__(
        self,
        checkpoint: Path,
        config: SelectiveRolloutGateConfig,
        threshold: float,
        seed: int,
    ) -> None:
        self._scorer = _LegalActionScorer(
            checkpoint,
            config.device,
            "legal_action_attention_scorer",
        )
        self._config = config
        self._threshold = threshold
        self._seed = seed
        self._turns = 0
        self._searches = 0
        self._switches = 0
        self._fallbacks = 0
        self._adjudicated = 0
        self._hard_caps = 0
        self._search_seconds = 0.0

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    @property
    def telemetry(self) -> dict[str, Any]:
        return {
            "turns": self._turns,
            "searches": self._searches,
            "switches": self._switches,
            "search_fraction": self._searches / self._turns if self._turns else 0.0,
            "rollout_fallback_triggers": self._fallbacks,
            "rollout_loop_adjudications": self._adjudicated,
            "rollout_hard_caps": self._hard_caps,
            "search_seconds": self._search_seconds,
        }

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        self._turns += 1
        ranked = distinct_canonical_scores(state, self._scorer.score_actions(state, actions))
        parent = ranked[0]
        if (
            len(ranked) < 2
            or self._searches >= self._config.max_searches_per_game
            or parent.score - ranked[1].score > self._config.routing_logit_gap
        ):
            return parent.action
        challenger = ranked[1]
        started = time.perf_counter()
        discovery, counters = self._evaluate_discovery(
            state,
            parent.action_index,
            challenger.action_index,
        )
        self._search_seconds += time.perf_counter() - started
        self._searches += 1
        self._fallbacks += counters["fallback_triggers"]
        self._adjudicated += counters["loop_adjudicated_continuations"]
        self._hard_caps += counters["hard_cap_continuations"]
        if discovery >= self._threshold:
            self._switches += 1
            return challenger.action
        return parent.action

    def _evaluate_discovery(
        self,
        state: SplendorState,
        parent_action_index: int,
        candidate_action_index: int,
    ) -> tuple[float, dict[str, int]]:
        public_key = canonical_public_belief_key(state, state.current_player)
        digest = hashlib.sha256(repr(public_key).encode("utf-8")).digest()
        root_offset = int.from_bytes(digest[:8], "big") % 100_000_000
        seed_base = self._seed + root_offset + self._searches * 10_000_019
        # The gameplay decision uses exactly the discovery half. The separate
        # fresh action-quality runner owns the independent evaluation half.
        local = self._config
        env = SplendorEnv()
        codec = ActionCodec()
        deltas = []
        counters = {
            "fallback_triggers": 0,
            "loop_adjudicated_continuations": 0,
            "hard_cap_continuations": 0,
        }
        for determinization_index in range(local.determinizations):
            determinized = determinize_for_observer(
                state,
                state.current_player,
                random.Random(seed_base + determinization_index * 1_000_003),
            )
            action_by_index = {
                codec.encode(determinized, action): action
                for action in env.legal_actions(determinized)
            }
            for continuation_index in range(local.continuations_per_half):
                continuation_seed = (
                    seed_base + determinization_index * 100_003 + continuation_index
                )
                results = [
                    rollout_score_margin_loop_safe(
                        env,
                        self._scorer,
                        determinized,
                        action_by_index[action_index],
                        state.current_player,
                        continuation_seed,
                        local.continuation_epsilon,
                        local.continuation_temperature,
                        local.continuation_top_k,
                        local.continuation_max_additional_turns,
                        local.score_margin_scale,
                        local.max_loop_fallback_triggers,
                    )
                    for action_index in (parent_action_index, candidate_action_index)
                ]
                deltas.append(results[1].value - results[0].value)
                counters["fallback_triggers"] += sum(result.fallback_triggers for result in results)
                counters["loop_adjudicated_continuations"] += sum(
                    int(result.loop_adjudicated) for result in results
                )
                counters["hard_cap_continuations"] += sum(
                    int(result.hard_cap_reached) for result in results
                )
        return statistics.fmean(deltas), counters


def run_direct_gameplay_gate(
    config: SelectiveRolloutGateConfig,
    threshold: float,
) -> dict[str, Any]:
    output_path = config.output_dir / "direct_vs_035.json"
    progress_path = config.output_dir / "direct_vs_035.progress.json"
    identity = {
        "format": "selective_matched_rollout_direct_v1",
        "parent_sha256": _sha256(config.parent_checkpoint),
        "threshold": threshold,
        "pairs": config.direct_pairs,
        "seed_start": config.direct_seed_start,
        "routing_logit_gap": config.routing_logit_gap,
        "max_searches_per_game": config.max_searches_per_game,
        "determinizations": config.determinizations,
        "continuations_per_half": config.continuations_per_half,
        "gameplay_rollout_seed_start": config.gameplay_rollout_seed_start,
    }
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if payload["identity"] != identity:
            raise ValueError("Direct gameplay progress identity mismatch.")
        completed = {int(row["seed"]): row for row in payload["pairs"]}
    seeds = list(range(config.direct_seed_start, config.direct_seed_start + config.direct_pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(
        f"[selective-rollout:direct] pairs={len(seeds)} complete={len(completed)} "
        f"pending={len(pending)} workers={config.workers}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_play_direct_pair, config, threshold, seed): seed
                for seed in pending
            }
            for future in as_completed(futures):
                seed = futures[future]
                completed[seed] = future.result()
                _atomic_write(
                    progress_path,
                    {
                        "identity": identity,
                        "completed": False,
                        "pairs": [completed[value] for value in sorted(completed)],
                    },
                )
                if len(completed) % 10 == 0 or len(completed) == len(seeds):
                    print(
                        f"[selective-rollout:direct] complete={len(completed)}/{len(seeds)}",
                        flush=True,
                    )
    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, config.bootstrap_samples, config.direct_seed_start)
    telemetry = [item for row in rows for item in row["candidate_telemetry"]]
    total_turns = sum(int(row["turns"]) for row in telemetry)
    total_searches = sum(int(row["searches"]) for row in telemetry)
    total_rollout_branches = (
        total_searches * config.determinizations * config.continuations_per_half * 2
    )
    aggregate_telemetry = {
        "turns": total_turns,
        "searches": total_searches,
        "switches": sum(int(row["switches"]) for row in telemetry),
        "search_fraction": total_searches / total_turns if total_turns else 0.0,
        "rollout_branches": total_rollout_branches,
        "rollout_fallback_triggers": sum(
            int(row["rollout_fallback_triggers"]) for row in telemetry
        ),
        "rollout_loop_adjudications": sum(
            int(row["rollout_loop_adjudications"]) for row in telemetry
        ),
        "rollout_hard_caps": sum(int(row["rollout_hard_caps"]) for row in telemetry),
        "search_seconds": sum(float(row["search_seconds"]) for row in telemetry),
    }
    share = summary["score_share"]
    gates = {
        "integrity": int(summary["timed_out_games"]) == 0
        and int(summary["stalled_games"]) == 0,
        "score_share": float(share["mean"]) >= config.minimum_direct_score_share,
        "score_share_ci_above_half": float(share["ci95_low"]) > 0.50,
        "search_used": total_searches > 0,
        "actions_switched": aggregate_telemetry["switches"] > 0,
        "bounded_search_fraction": aggregate_telemetry["search_fraction"]
        <= config.maximum_search_fraction,
        "zero_rollout_hard_caps": aggregate_telemetry["rollout_hard_caps"] == 0,
        "bounded_rollout_adjudication": (
            aggregate_telemetry["rollout_loop_adjudications"] / total_rollout_branches
            if total_rollout_branches
            else 1.0
        )
        <= config.maximum_loop_adjudicated_fraction,
    }
    result = {
        "format": "selective_matched_rollout_direct_gate_v1",
        "identity": identity,
        "summary": summary,
        "telemetry": aggregate_telemetry,
        "gates": gates,
        "passed": all(gates.values()),
        "pairs": rows,
    }
    _atomic_write(output_path, result)
    _atomic_write(
        progress_path,
        {"identity": identity, "completed": True, "pairs": rows},
    )
    return result


def _play_direct_pair(
    config: SelectiveRolloutGateConfig,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    games = []
    telemetry = []
    for candidate_seat in (0, 1):
        candidate = SelectiveMatchedRolloutBot(
            config.parent_checkpoint,
            config,
            threshold,
            config.gameplay_rollout_seed_start + seed * 101 + candidate_seat,
        )
        baseline = LegalActionAttentionScorerBot(
            config.parent_checkpoint,
            device=config.device,
        )
        bots = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
        result = play_game(
            bots[0],
            bots[1],
            seed=seed,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
        )
        games.append(asdict(result))
        telemetry.append(candidate.telemetry)
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": games,
        "candidate_telemetry": telemetry,
    }


def _fresh_breakdowns(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, dict[str, dict[str, float | int]]]:
    definitions = {
        "phase": lambda row: str(row["phase"]),
        "opponent": lambda row: str(row["opponent"]),
        "seat": lambda row: f"seat{int(row['actor_seat'])}",
    }
    result = {}
    for category, key_fn in definitions.items():
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(key_fn(row), []).append(row)
        result[category] = {}
        for key, bucket in sorted(buckets.items()):
            switched = [
                float(row["candidate_minus_parent_second"])
                for row in bucket
                if float(row["candidate_minus_parent_first"]) >= threshold
            ]
            result[category][key] = {
                "rows": len(bucket),
                "switches": len(switched),
                "evaluation_gain_mean": statistics.fmean(switched) if switched else 0.0,
                "evaluation_positive_fraction": (
                    statistics.fmean(value > 0.0 for value in switched) if switched else 0.0
                ),
            }
    return result


def _mean_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = statistics.fmean(values)
    if samples <= 0 or len(values) == 1 or len(set(values)) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": estimates[int(0.025 * (samples - 1))],
        "ci95_high": estimates[int(0.975 * (samples - 1))],
    }


def _source_config(config: SelectiveRolloutGateConfig) -> PolicyDeltaCausalAuditConfig:
    return PolicyDeltaCausalAuditConfig(
        output_dir=config.output_dir,
        data_dir=config.data_dir,
        parent_checkpoint=config.parent_checkpoint,
        candidate_checkpoint=config.parent_checkpoint,
        checkpoint_016=config.checkpoint_016,
        checkpoint_023=config.checkpoint_023,
        games=config.source_games,
        states_per_opponent_seat=1,
        collection_seed_start=config.collection_seed_start,
        continuation_seed_start=config.continuation_seed_start,
        determinizations=config.determinizations,
        continuations_per_determinization=config.continuations_per_half * 2,
        continuation_epsilon=config.continuation_epsilon,
        continuation_temperature=config.continuation_temperature,
        continuation_top_k=config.continuation_top_k,
        continuation_max_additional_turns=config.continuation_max_additional_turns,
        score_margin_scale=config.score_margin_scale,
        workers=config.workers,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
        bootstrap_samples=config.bootstrap_samples,
        device=config.device,
    )


def _protocol(config: SelectiveRolloutGateConfig) -> dict[str, Any]:
    return {
        "format": "selective_matched_rollout_protocol_v1",
        "hypothesis": (
            "A bounded information-legal matched-terminal-rollout comparison can "
            "use the reproducible action signal directly and improve frozen 035."
        ),
        "development": (
            "Select one fixed delta threshold on Generation 002 train halves, then "
            "require independent validation-half gain. Confirmation rows are unused."
        ),
        "fresh_confirmation": (
            "Collect 960 disjoint balanced games, select 480 outcome-blind uncertain "
            "states, use one 3x4 half for decisions and the other for evaluation."
        ),
        "gameplay": (
            "Only a complete fresh action-quality pass unlocks 200 paired seeds/400 "
            "games against raw 035. At most two root searches are allowed per game."
        ),
        "non_learning": True,
        "no_model_promotion": True,
        "hashes": {
            "parent_035": _sha256(config.parent_checkpoint),
            "development_evaluation": _sha256(config.development_evaluation),
            "checkpoint_016": _sha256(config.checkpoint_016),
            "checkpoint_023": _sha256(config.checkpoint_023),
        },
        "config": _config_payload(config),
    }


def _finalize(config: SelectiveRolloutGateConfig, decision: str, **stages: Any) -> dict[str, Any]:
    payload = {
        "gate": "selective_matched_rollout_004",
        "decision": decision,
        "stronger_teacher": decision == "direct_gate_passed_stronger_teacher",
        **stages,
    }
    _atomic_write(config.output_dir / "gate_final.json", payload)
    print(json.dumps({"decision": decision, "stronger_teacher": payload["stronger_teacher"]}, indent=2))
    return payload


def _load_progress(path: Path) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    if not path.exists():
        return None, {}
    header = None
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "header" in payload:
                header = payload["header"]
            else:
                rows[int(payload["state_index"])] = payload["row"]
    return header, rows


def _config_payload(config: SelectiveRolloutGateConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _validate(config: SelectiveRolloutGateConfig) -> None:
    for path in (
        config.development_evaluation,
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if config.source_games % (len(OPPONENTS) * 2 * len(PHASES)):
        raise ValueError("source_games must balance every opponent/seat/phase cell.")
    if config.states_per_stratum <= 0 or config.workers <= 0:
        raise ValueError("states_per_stratum and workers must be positive.")
    if config.continuations_per_half <= 0 or config.determinizations <= 0:
        raise ValueError("Rollout counts must be positive.")
    if not all(value > 0.0 for value in config.threshold_grid):
        raise ValueError("Every threshold must be positive.")
    if not 0.0 < config.minimum_switch_fraction < config.maximum_switch_fraction < 1.0:
        raise ValueError("Switch-fraction bounds are invalid.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/selective_matched_rollout_004",
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/selective_matched_rollout_004/data",
    )
    parser.add_argument(
        "--development-evaluation",
        default=(
            "outputs/action_preference_generation_002_loop_repair/"
            "preference_evaluation_repaired.progress.jsonl"
        ),
    )
    parser.add_argument(
        "--parent-checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-016",
        default=(
            "outputs/legal_action_scorer_016_policy_guided_mix/"
            "legal_action_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-023",
        default=(
            "outputs/canonical_entity_coherent_023/"
            "canonical_entity_action_scorer_coherent_width64_best.pt"
        ),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_selective_rollout_gate(
        SelectiveRolloutGateConfig(
            output_dir=Path(args.output_dir),
            data_dir=Path(args.data_dir),
            development_evaluation=Path(args.development_evaluation),
            parent_checkpoint=Path(args.parent_checkpoint),
            checkpoint_016=Path(args.checkpoint_016),
            checkpoint_023=Path(args.checkpoint_023),
            workers=args.workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
