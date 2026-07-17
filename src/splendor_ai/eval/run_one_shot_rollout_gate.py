"""Gate a phase-scheduled one-shot matched-rollout wrapper around policy 035."""

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
from splendor_ai.encoding import ActionCodec, canonical_public_belief_key
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_selective_rollout_gate import (
    _atomic_write,
    _load_progress,
    distinct_canonical_scores,
    evaluate_action_pair,
    threshold_metrics,
)
from splendor_ai.eval.run_shallow_operational_gate import summarize_pair_rows
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_action_preference_repair import rollout_score_margin_loop_safe
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
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


_EVALUATION_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class OneShotRolloutGateConfig:
    output_dir: Path
    data_dir: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    switch_threshold: float = 0.075
    routing_logit_gap: float = 0.5
    source_games: int = 960
    states_per_stratum: int = 20
    collection_seed_start: int = 375_000
    continuation_seed_start: int = 40_000_000
    gameplay_rollout_seed_start: int = 41_000_000
    direct_seed_start: int = 377_000
    determinizations: int = 3
    continuations_per_half: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_additional_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    minimum_switches: int = 20
    minimum_evaluation_positive_fraction: float = 0.60
    minimum_breakdown_gain: float = -0.025
    maximum_search_turn_fraction: float = 0.05
    maximum_switch_turn_fraction: float = 0.03
    maximum_loop_adjudicated_fraction: float = 0.001
    direct_pairs: int = 200
    minimum_direct_score_share: float = 0.55
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    device: str = "cpu"


def run_one_shot_rollout_gate(config: OneShotRolloutGateConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing one-shot rollout protocol does not match.")
    else:
        _atomic_write(protocol_path, protocol)

    source_config = _source_config(config)
    collection = collect_fresh_source_games(source_config)
    if not collection["passed"]:
        return _finalize(config, "fresh_collection_failed", collection=collection)
    selection = select_first_phase_trigger_states(config, source_config, collection)
    if not selection["passed"]:
        return _finalize(
            config,
            "fresh_selection_failed",
            collection=collection,
            selection=selection,
        )
    confirmation = evaluate_one_shot_action_quality(config, selection)
    if not confirmation["passed"]:
        return _finalize(
            config,
            "fresh_action_quality_failed",
            collection=collection,
            selection=selection,
            confirmation=confirmation,
        )
    direct = run_one_shot_direct_gate(config)
    return _finalize(
        config,
        "direct_gate_passed_stronger_teacher"
        if direct["passed"]
        else "direct_gameplay_failed",
        collection=collection,
        selection=selection,
        confirmation=confirmation,
        direct=direct,
    )


def assigned_phase(seed: int) -> str:
    return PHASES[seed % len(PHASES)]


def operational_rates(
    eligible_triggers: int,
    actor_turns: int,
    trigger_switch_fraction: float,
) -> dict[str, float]:
    search_fraction = eligible_triggers / actor_turns if actor_turns else 0.0
    return {
        "search_turn_fraction": search_fraction,
        "projected_switch_turn_fraction": search_fraction * trigger_switch_fraction,
    }


def select_first_phase_trigger_states(
    config: OneShotRolloutGateConfig,
    source_config: PolicyDeltaCausalAuditConfig,
    collection: dict[str, Any],
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
        actor_seat = int(summary["actor_seat"])
        opponent = str(summary["opponent"])
        phase = assigned_phase(int(summary["seed"]))
        with _source_replay_path(source_config, game_index).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                observation = outcome_blind_replay_projection(json.loads(line))
                if int(observation["player_id"]) != actor_seat:
                    continue
                turn_index = int(observation["turn_index"])
                if _phase(turn_index) != phase:
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
                candidates[_stratum(opponent, actor_seat, phase)].append(
                    {
                        "source_game_index": game_index,
                        "source_game_seed": int(observation["game_seed"]),
                        "turn_index": turn_index,
                        "phase": phase,
                        "actor_seat": actor_seat,
                        "opponent": opponent,
                        "source_cell": _stratum_cell(opponent, actor_seat),
                        "stratum": _stratum(opponent, actor_seat, phase),
                        "parent_action_index": parent.action_index,
                        "parent_action": serialize_action(parent.action),
                        "candidate_action_index": challenger.action_index,
                        "candidate_action": serialize_action(challenger.action),
                        "top_two_logit_gap": gap,
                        "state_snapshot": observation["state_snapshot"],
                    }
                )
                break
    selected = []
    for stratum in sorted(candidates):
        selected.extend(
            sorted(
                candidates[stratum],
                key=lambda row: int(row["source_game_seed"]),
            )[: config.states_per_stratum]
        )
    selected.sort(
        key=lambda row: (str(row["stratum"]), int(row["source_game_seed"]))
    )
    selected_counts = {
        stratum: sum(row["stratum"] == stratum for row in selected)
        for stratum in sorted(candidates)
    }
    eligible_counts = {key: len(value) for key, value in candidates.items()}
    eligible_triggers = sum(eligible_counts.values())
    actor_turns = int(collection["actor_turns"])
    rates = operational_rates(eligible_triggers, actor_turns, 0.0)
    expected = len(OPPONENTS) * 2 * len(PHASES) * config.states_per_stratum
    gates = {
        "exact_state_count": len(selected) == expected,
        "unique_games": len({row["source_game_index"] for row in selected})
        == len(selected),
        "exact_stratum_balance": set(selected_counts.values())
        == {config.states_per_stratum},
        "all_first_online_triggers": True,
        "outcome_blind_selection": True,
        "bounded_source_search_fraction": 0.0 < rates["search_turn_fraction"]
        <= config.maximum_search_turn_fraction,
    }
    result = {
        "format": "one_shot_rollout_first_phase_trigger_selection_v1",
        "rows_scanned": rows_scanned,
        "actor_turns": actor_turns,
        "eligible_triggers": eligible_triggers,
        "source_search_turn_fraction": rates["search_turn_fraction"],
        "eligible_by_stratum": eligible_counts,
        "selected_states": len(selected),
        "selected_by_stratum": selected_counts,
        "selected": selected,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_write(path, result)
    return result


def evaluate_one_shot_action_quality(
    config: OneShotRolloutGateConfig,
    selection: dict[str, Any],
) -> dict[str, Any]:
    result_path = config.output_dir / "fresh_action_quality.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    progress_path = config.output_dir / "fresh_action_quality.progress.jsonl"
    header, completed = _load_progress(progress_path)
    selected = list(selection["selected"])
    expected_header = {
        "format": "one_shot_rollout_fresh_progress_v1",
        "protocol_sha256": _sha256(config.output_dir / "protocol.json"),
        "selection_sha256": _sha256(config.output_dir / "fresh_selection.json"),
        "states": len(selected),
    }
    if header is None:
        progress_path.write_text(json.dumps({"header": expected_header}) + "\n", encoding="utf-8")
    elif header != expected_header:
        raise ValueError("One-shot action-quality progress does not match.")
    missing = [(index, selected[index]) for index in range(len(selected)) if index not in completed]
    print(
        f"[one-shot:fresh] states={len(selected)} complete={len(completed)} "
        f"pending={len(missing)} workers={config.workers}",
        flush=True,
    )
    if missing:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_evaluation_worker,
            initargs=(str(config.parent_checkpoint), config.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_state, item, config): item[0]
                for item in missing
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    index, row = future.result()
                    completed[index] = row
                    handle.write(json.dumps({"state_index": index, "row": row}) + "\n")
                    handle.flush()
                    if len(completed) % 20 == 0 or len(completed) == len(selected):
                        print(f"[one-shot:fresh] complete={len(completed)}/{len(selected)}", flush=True)
    rows = [completed[index] for index in range(len(selected))]
    metrics = threshold_metrics(
        rows,
        config.switch_threshold,
        config.bootstrap_samples,
        config.collection_seed_start + 401,
        symmetric=False,
    )
    breakdowns = _breakdowns(rows, config.switch_threshold)
    rates = operational_rates(
        int(selection["eligible_triggers"]),
        int(selection["actor_turns"]),
        float(metrics["switch_fraction"]),
    )
    total = sum(int(row["total_continuations"]) for row in rows)
    adjudicated = sum(int(row["loop_adjudicated_continuations"]) for row in rows)
    gates = {
        "complete": len(rows) == len(selected),
        "all_continuations_resolved": sum(int(row["resolved_continuations"]) for row in rows)
        == total,
        "zero_hard_caps": sum(int(row["hard_cap_continuations"]) for row in rows) == 0,
        "bounded_loop_adjudication": adjudicated / total
        <= config.maximum_loop_adjudicated_fraction,
        "minimum_switches": int(metrics["switches"]) >= config.minimum_switches,
        "evaluation_gain_ci_above_zero": float(metrics["evaluation_gain"]["ci95_low"])
        > 0.0,
        "positive_fraction": float(metrics["evaluation_positive_fraction"])
        >= config.minimum_evaluation_positive_fraction,
        "bounded_search_turn_fraction": rates["search_turn_fraction"]
        <= config.maximum_search_turn_fraction,
        "bounded_projected_switch_turn_fraction": rates["projected_switch_turn_fraction"]
        <= config.maximum_switch_turn_fraction,
        "breakdown_breadth": all(
            int(bucket["switches"]) >= 3
            and float(bucket["evaluation_gain_mean"]) >= config.minimum_breakdown_gain
            for category in breakdowns.values()
            for bucket in category.values()
        ),
    }
    result = {
        "format": "one_shot_rollout_fresh_action_quality_v1",
        "threshold": config.switch_threshold,
        "metrics": metrics,
        "operational_rates": rates,
        "breakdowns": breakdowns,
        "total_continuations": total,
        "fallback_triggers": sum(int(row["fallback_triggers"]) for row in rows),
        "loop_adjudicated_continuations": adjudicated,
        "hard_cap_continuations": sum(int(row["hard_cap_continuations"]) for row in rows),
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }
    _atomic_write(result_path, result)
    return result


def _initialize_evaluation_worker(parent_checkpoint: str, device: str) -> None:
    global _EVALUATION_SCORER
    torch.set_num_threads(1)
    _EVALUATION_SCORER = _LegalActionScorer(
        Path(parent_checkpoint), device, "legal_action_attention_scorer"
    )


def _evaluate_state(
    item: tuple[int, dict[str, Any]],
    config: OneShotRolloutGateConfig,
) -> tuple[int, dict[str, Any]]:
    if _EVALUATION_SCORER is None:
        raise RuntimeError("One-shot evaluation worker was not initialized.")
    state_index, payload = item
    first, second, counters = evaluate_action_pair(
        deserialize_state_snapshot(payload["state_snapshot"]),
        int(payload["actor_seat"]),
        int(payload["parent_action_index"]),
        int(payload["candidate_action_index"]),
        _EVALUATION_SCORER,
        config,  # type: ignore[arg-type]
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
        "switched": first >= config.switch_threshold,
        **counters,
    }


class OneShotMatchedRolloutBot:
    def __init__(
        self,
        checkpoint: Path,
        config: OneShotRolloutGateConfig,
        phase: str,
        seed: int,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"Unknown phase: {phase}")
        self._scorer = _LegalActionScorer(
            checkpoint, config.device, "legal_action_attention_scorer"
        )
        self._config = config
        self._phase = phase
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
            "phase": self._phase,
            "turns": self._turns,
            "searches": self._searches,
            "switches": self._switches,
            "search_fraction": self._searches / self._turns if self._turns else 0.0,
            "switch_fraction": self._switches / self._turns if self._turns else 0.0,
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
            self._searches > 0
            or _phase(state.turn_index) != self._phase
            or len(ranked) < 2
            or parent.score - ranked[1].score > self._config.routing_logit_gap
        ):
            return parent.action
        challenger = ranked[1]
        started = time.perf_counter()
        discovery, counters = self._discovery(
            state, parent.action_index, challenger.action_index
        )
        self._search_seconds += time.perf_counter() - started
        self._searches = 1
        self._fallbacks += counters["fallback_triggers"]
        self._adjudicated += counters["loop_adjudicated_continuations"]
        self._hard_caps += counters["hard_cap_continuations"]
        if discovery >= self._config.switch_threshold:
            self._switches = 1
            return challenger.action
        return parent.action

    def _discovery(
        self,
        state: SplendorState,
        parent_action_index: int,
        candidate_action_index: int,
    ) -> tuple[float, dict[str, int]]:
        digest = hashlib.sha256(
            repr(canonical_public_belief_key(state, state.current_player)).encode("utf-8")
        ).digest()
        seed_base = self._seed + int.from_bytes(digest[:8], "big") % 100_000_000
        env = SplendorEnv()
        codec = ActionCodec()
        deltas = []
        counters = {
            "fallback_triggers": 0,
            "loop_adjudicated_continuations": 0,
            "hard_cap_continuations": 0,
        }
        for det_index in range(self._config.determinizations):
            determinized = determinize_for_observer(
                state,
                state.current_player,
                random.Random(seed_base + det_index * 1_000_003),
            )
            action_by_index = {
                codec.encode(determinized, action): action
                for action in env.legal_actions(determinized)
            }
            for continuation_index in range(self._config.continuations_per_half):
                continuation_seed = seed_base + det_index * 100_003 + continuation_index
                results = [
                    rollout_score_margin_loop_safe(
                        env,
                        self._scorer,
                        determinized,
                        action_by_index[action_index],
                        state.current_player,
                        continuation_seed,
                        self._config.continuation_epsilon,
                        self._config.continuation_temperature,
                        self._config.continuation_top_k,
                        self._config.continuation_max_additional_turns,
                        self._config.score_margin_scale,
                        self._config.max_loop_fallback_triggers,
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


def run_one_shot_direct_gate(config: OneShotRolloutGateConfig) -> dict[str, Any]:
    output_path = config.output_dir / "direct_vs_035.json"
    progress_path = config.output_dir / "direct_vs_035.progress.json"
    identity = {
        "format": "one_shot_rollout_direct_v1",
        "parent_sha256": _sha256(config.parent_checkpoint),
        "threshold": config.switch_threshold,
        "pairs": config.direct_pairs,
        "seed_start": config.direct_seed_start,
        "phase_assignment": "environment_seed_modulo_3",
        "routing_logit_gap": config.routing_logit_gap,
        "maximum_searches_per_game": 1,
        "determinizations": config.determinizations,
        "continuations_per_half": config.continuations_per_half,
        "gameplay_rollout_seed_start": config.gameplay_rollout_seed_start,
    }
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if payload["identity"] != identity:
            raise ValueError("One-shot direct progress identity mismatch.")
        completed = {int(row["seed"]): row for row in payload["pairs"]}
    seeds = list(range(config.direct_seed_start, config.direct_seed_start + config.direct_pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(
        f"[one-shot:direct] pairs={len(seeds)} complete={len(completed)} "
        f"pending={len(pending)} workers={config.workers}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_play_direct_pair, config, seed): seed for seed in pending
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
                    print(f"[one-shot:direct] complete={len(completed)}/{len(seeds)}", flush=True)
    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, config.bootstrap_samples, config.direct_seed_start)
    telemetry = [item for row in rows for item in row["candidate_telemetry"]]
    turns = sum(int(row["turns"]) for row in telemetry)
    searches = sum(int(row["searches"]) for row in telemetry)
    switches = sum(int(row["switches"]) for row in telemetry)
    branches = searches * config.determinizations * config.continuations_per_half * 2
    phase_rows = {
        phase: [row for row in telemetry if row["phase"] == phase] for phase in PHASES
    }
    aggregate = {
        "turns": turns,
        "searches": searches,
        "switches": switches,
        "search_fraction": searches / turns if turns else 0.0,
        "switch_fraction": switches / turns if turns else 0.0,
        "rollout_branches": branches,
        "rollout_fallback_triggers": sum(int(row["rollout_fallback_triggers"]) for row in telemetry),
        "rollout_loop_adjudications": sum(int(row["rollout_loop_adjudications"]) for row in telemetry),
        "rollout_hard_caps": sum(int(row["rollout_hard_caps"]) for row in telemetry),
        "search_seconds": sum(float(row["search_seconds"]) for row in telemetry),
        "by_phase": {
            phase: {
                "games": len(items),
                "searches": sum(int(row["searches"]) for row in items),
                "switches": sum(int(row["switches"]) for row in items),
            }
            for phase, items in phase_rows.items()
        },
    }
    share = summary["score_share"]
    gates = {
        "integrity": int(summary["timed_out_games"]) == 0
        and int(summary["stalled_games"]) == 0,
        "score_share": float(share["mean"]) >= config.minimum_direct_score_share,
        "score_share_ci_above_half": float(share["ci95_low"]) > 0.50,
        "search_used_every_phase": all(
            int(row["searches"]) > 0 for row in aggregate["by_phase"].values()
        ),
        "actions_switched": switches > 0,
        "bounded_search_fraction": aggregate["search_fraction"]
        <= config.maximum_search_turn_fraction,
        "bounded_switch_fraction": aggregate["switch_fraction"]
        <= config.maximum_switch_turn_fraction,
        "zero_rollout_hard_caps": aggregate["rollout_hard_caps"] == 0,
        "bounded_rollout_adjudication": (
            aggregate["rollout_loop_adjudications"] / branches if branches else 1.0
        )
        <= config.maximum_loop_adjudicated_fraction,
    }
    result = {
        "format": "one_shot_rollout_direct_gate_v1",
        "identity": identity,
        "summary": summary,
        "telemetry": aggregate,
        "gates": gates,
        "passed": all(gates.values()),
        "pairs": rows,
    }
    _atomic_write(output_path, result)
    _atomic_write(progress_path, {"identity": identity, "completed": True, "pairs": rows})
    return result


def _play_direct_pair(config: OneShotRolloutGateConfig, seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    games = []
    telemetry = []
    phase = assigned_phase(seed)
    for candidate_seat in (0, 1):
        candidate = OneShotMatchedRolloutBot(
            config.parent_checkpoint,
            config,
            phase,
            config.gameplay_rollout_seed_start + seed * 101 + candidate_seat,
        )
        baseline = LegalActionAttentionScorerBot(config.parent_checkpoint, device=config.device)
        bots = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
        game = play_game(
            bots[0],
            bots[1],
            seed=seed,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
        )
        games.append(asdict(game))
        telemetry.append(candidate.telemetry)
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": games,
        "candidate_telemetry": telemetry,
    }


def _breakdowns(
    rows: list[dict[str, Any]], threshold: float
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


def _source_config(config: OneShotRolloutGateConfig) -> PolicyDeltaCausalAuditConfig:
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


def _protocol(config: OneShotRolloutGateConfig) -> dict[str, Any]:
    return {
        "format": "one_shot_matched_rollout_protocol_v1",
        "hypothesis": (
            "The already validated 0.075 rollout switch rule improves 035 when "
            "capacity is bounded structurally to one online intervention per game."
        ),
        "not_a_threshold_repair": True,
        "switch_threshold_fixed_from_004_development": config.switch_threshold,
        "phase_schedule": "environment seed modulo early/mid/late",
        "selection": (
            "First online-eligible gap<=0.5 state in the assigned phase; no outcome "
            "or future-state uncertainty minimization."
        ),
        "fresh_confirmation": (
            "960 fresh balanced games and 480 balanced states; discovery 3x4 and "
            "independent evaluation 3x4 per action."
        ),
        "conditional_gameplay": "200 fresh paired seeds/400 games versus raw 035",
        "non_learning": True,
        "no_model_promotion": True,
        "hashes": {
            "parent_035": _sha256(config.parent_checkpoint),
            "checkpoint_016": _sha256(config.checkpoint_016),
            "checkpoint_023": _sha256(config.checkpoint_023),
        },
        "config": _config_payload(config),
    }


def _config_payload(config: OneShotRolloutGateConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _finalize(config: OneShotRolloutGateConfig, decision: str, **stages: Any) -> dict[str, Any]:
    payload = {
        "gate": "one_shot_matched_rollout_005",
        "decision": decision,
        "stronger_teacher": decision == "direct_gate_passed_stronger_teacher",
        **stages,
    }
    _atomic_write(config.output_dir / "gate_final.json", payload)
    print(json.dumps({"decision": decision, "stronger_teacher": payload["stronger_teacher"]}, indent=2))
    return payload


def _validate(config: OneShotRolloutGateConfig) -> None:
    for path in (config.parent_checkpoint, config.checkpoint_016, config.checkpoint_023):
        if not path.exists():
            raise FileNotFoundError(path)
    if config.source_games % (len(OPPONENTS) * 2 * len(PHASES)):
        raise ValueError("source_games must balance opponent/seat/phase assignments.")
    if config.switch_threshold != 0.075:
        raise ValueError("Experiment 005 freezes the already selected 0.075 threshold.")
    if config.states_per_stratum <= 0 or config.workers <= 0:
        raise ValueError("states_per_stratum and workers must be positive.")
    if config.determinizations <= 0 or config.continuations_per_half <= 0:
        raise ValueError("Rollout counts must be positive.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/one_shot_matched_rollout_005")
    parser.add_argument("--data-dir", default="outputs/one_shot_matched_rollout_005/data")
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
    run_one_shot_rollout_gate(
        OneShotRolloutGateConfig(
            output_dir=Path(args.output_dir),
            data_dir=Path(args.data_dir),
            parent_checkpoint=Path(args.parent_checkpoint),
            checkpoint_016=Path(args.checkpoint_016),
            checkpoint_023=Path(args.checkpoint_023),
            workers=args.workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
