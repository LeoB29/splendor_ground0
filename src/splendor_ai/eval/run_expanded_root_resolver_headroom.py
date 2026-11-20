"""Run preregistered experiment 085: expanded-root resolver headroom audit."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Sequence

import torch

from splendor_ai.cfr import ResolvedLeaf, deterministic_belief_range, resolve_two_ply
from splendor_ai.diagnostics import state_signature
from splendor_ai.encoding import canonical_public_belief_key, canonical_successor_action_key
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    _sha256_file,
)
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    _distinct_canonical_scores,
)
from splendor_ai.training.replay import collect_game_replay, deserialize_state_snapshot
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


PHASES = ("early", "mid", "late")
CELLS = tuple(f"seat{seat}|{phase}" for seat in (0, 1) for phase in PHASES)
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")
PROTOCOL = Path("docs/experiments/expanded_root_resolver_headroom_085.md")
PROTOCOL_SHA256 = "66ae0fdacd455dd69bbdb8cfef8fa254ee05320822d7fe419e8d7f4f10c940df"


@dataclass(frozen=True, slots=True)
class ExpandedRootHeadroomConfig:
    output_dir: Path = Path("outputs/expanded_root_resolver_headroom_085")
    checkpoint: Path = CHECKPOINT
    source_seed_start: int = 850_000
    source_games: int = 600
    states_per_cell: int = 50
    champion_roots: int = 4
    determinizations: int = 3
    opponent_actions: int = 4
    continuations_per_leaf: int = 8
    discovery_continuations: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_base: int = 850_000_000
    continuation_seed_base: int = 8_500_000_000_000_000_000
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 850_085
    minimum_mean_gain: float = 0.05
    minimum_change_fraction: float = 0.15
    phase_collapse_boundary: float = -0.05
    minimum_leaf_pearson: float = 0.60
    maximum_leaf_mae: float = 0.10
    minimum_leaf_sign_agreement: float = 0.70
    minimum_root_pearson: float = 0.60
    minimum_best_root_agreement: float = 0.60
    maximum_loop_adjudicated_fraction: float = 0.001
    maximum_evaluation_wall_seconds: float = 21_600.0
    device: str = "cpu"


class RootIneligible(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _GreedyScorerBot:
    def __init__(self, scorer: _LegalActionScorer) -> None:
        self._scorer = scorer

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        return self._scorer.score_actions(state, actions)[0].action


_WORKER_SCORER: _LegalActionScorer | None = None


def _initialize_worker(checkpoint: Path, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(
        checkpoint,
        device,
        "legal_action_attention_scorer",
    )


def run_experiment(
    config: ExpandedRootHeadroomConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ExpandedRootHeadroomConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 085 result does not match the frozen identity.")
        return result

    selection = _collect_and_select(cfg, identity)
    selection_gates = _selection_gates(cfg, selection)
    if not all(selection_gates.values()):
        result = _invalid_result(
            cfg,
            identity,
            selection,
            selection_gates,
            "source_or_selection_invalid",
        )
        _atomic_write(result_path, result)
        return result

    selected = list(selection["selected"])
    started = time.perf_counter()
    rows = _evaluate_parallel(cfg, identity, selected)
    evaluation_wall_seconds = time.perf_counter() - started
    if len(rows) != len(selected):
        result = _invalid_result(
            cfg,
            identity,
            selection,
            selection_gates,
            "incomplete_evaluation",
            rows=rows,
            evaluation_wall_seconds=evaluation_wall_seconds,
        )
        _atomic_write(result_path, result)
        return result

    scorer = _LegalActionScorer(
        cfg.checkpoint,
        cfg.device,
        "legal_action_attention_scorer",
    )
    repeated = _evaluate_state(selected[0], cfg, scorer)
    repeat_identical = _deterministic_projection(rows[0]) == _deterministic_projection(
        repeated
    )
    privacy = _privacy_audit(selected[0], cfg, scorer)

    summary = _summarize(rows, cfg, evaluation_wall_seconds)
    integrity = _integrity_gates(
        cfg,
        selection,
        rows,
        summary,
        privacy,
        repeat_identical,
        evaluation_wall_seconds,
    )
    signal = _signal_gates(cfg, summary)
    phase_guardrails = {
        phase: float(summary["phase_breakdown"][phase]["gain_ci95_high"])
        > cfg.phase_collapse_boundary
        for phase in PHASES
    }
    valid = all(selection_gates.values()) and all(integrity.values())
    passed = valid and all(signal.values()) and all(phase_guardrails.values())
    if not valid:
        decision = "expanded_root_085_invalid_no_headroom_decision"
    elif passed:
        decision = "expanded_root_085_headroom_pass_authorize_gameplay_preregistration"
    elif float(summary["change_fraction"]) < cfg.minimum_change_fraction:
        decision = "expanded_root_085_low_dose_close_exact_expansion"
    else:
        decision = "expanded_root_085_no_reliable_headroom_close_exact_expansion"

    result = {
        "experiment": "085_expanded_root_resolver_headroom",
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "identity": identity,
        "selection": {
            key: value for key, value in selection.items() if key != "selected"
        },
        "selection_gates": selection_gates,
        "summary": summary,
        "integrity_gates": integrity,
        "signal_gates": signal,
        "phase_guardrails": phase_guardrails,
        "privacy": privacy,
        "first_state_repeat_identical": repeat_identical,
        "valid": valid,
        "passed": passed,
        "decision": decision,
        "gameplay_preregistration_authorized": passed,
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "promotion_authorized": False,
        "fast_champion": "035",
        "research_champion": "public_belief_resolver_050_051",
        "rows": rows,
    }
    _atomic_write(result_path, result)
    print(f"085 decision={decision}", flush=True)
    return result


def _invalid_result(
    cfg: ExpandedRootHeadroomConfig,
    identity: dict[str, Any],
    selection: dict[str, Any],
    selection_gates: dict[str, bool],
    reason: str,
    *,
    rows: list[dict[str, Any]] | None = None,
    evaluation_wall_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "085_expanded_root_resolver_headroom",
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "identity": identity,
        "selection": {key: value for key, value in selection.items() if key != "selected"},
        "selection_gates": selection_gates,
        "summary": None,
        "integrity_gates": None,
        "signal_gates": None,
        "phase_guardrails": None,
        "valid": False,
        "passed": False,
        "decision": "expanded_root_085_invalid_no_headroom_decision",
        "invalid_reason": reason,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "completed_rows": len(rows or []),
        "gameplay_preregistration_authorized": False,
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "promotion_authorized": False,
    }


def _collect_and_select(
    cfg: ExpandedRootHeadroomConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    selection_path = cfg.output_dir / "selection.json"
    if selection_path.exists():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("Existing 085 selection does not match frozen identity.")
        return payload

    progress_path = cfg.output_dir / "source.progress.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 085 source progress identity mismatch.")
            completed[int(payload["seed"])] = payload["result"]

    seeds = list(range(cfg.source_seed_start, cfg.source_seed_start + cfg.source_games))
    pending = [seed for seed in seeds if seed not in completed]
    print(
        f"[resolver085:source] games={len(seeds)} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(
            max_workers=cfg.workers,
            initializer=_initialize_worker,
            initargs=(cfg.checkpoint, cfg.device),
        ) as executor:
            futures = {executor.submit(_screen_source_game_worker, seed, cfg): seed for seed in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                result = future.result()
                completed[seed] = result
                _append_jsonl(
                    progress_path,
                    {"identity": identity, "seed": seed, "result": result},
                )
                if index % 25 == 0 or index == len(pending):
                    print(
                        f"[resolver085:source] complete={len(completed)}/{len(seeds)}",
                        flush=True,
                    )

    games = [completed[seed] for seed in seeds]
    buckets: dict[str, list[dict[str, Any]]] = {cell: [] for cell in CELLS}
    for game in games:
        candidate = game.get("selection_candidate")
        if candidate is not None:
            buckets[str(game["cell"])].append(candidate)
    chosen = []
    for cell in CELLS:
        chosen.extend(buckets[cell][: cfg.states_per_cell])
    chosen.sort(key=lambda row: (int(row["source_seed"]), int(row["turn_index"])))
    selected = [{**row, "sample_index": index} for index, row in enumerate(chosen)]
    termination_reasons = Counter(str(game["termination_reason"]) for game in games)
    payload = {
        "format": "expanded_root_selection_085_v1",
        "identity": identity,
        "source_games": len(games),
        "source_seed_min": min(seeds),
        "source_seed_max": max(seeds),
        "source_termination_reasons": dict(sorted(termination_reasons.items())),
        "source_invalid_games": sum(
            str(game["termination_reason"]) not in VALID_TERMINATIONS
            or bool(game["stalled"])
            or bool(game["timed_out"])
            or sum(int(value) for value in game["loop_fallback_triggers_by_seat"]) != 0
            for game in games
        ),
        "eligible_games_by_cell": {cell: len(buckets[cell]) for cell in CELLS},
        "selected_by_cell": dict(
            sorted(Counter(str(row["selection_cell"]) for row in selected).items())
        ),
        "selected_count": len(selected),
        "unique_source_games": len({int(row["source_seed"]) for row in selected}),
        "unique_states": len(
            {(int(row["source_seed"]), int(row["turn_index"])) for row in selected}
        ),
        "outcome_blind": True,
        "selected": selected,
    }
    _atomic_write(selection_path, payload)
    return payload


def _screen_source_game_worker(
    seed: int,
    cfg: ExpandedRootHeadroomConfig,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("085 source worker scorer was not initialized.")
    return _screen_source_game(seed, cfg, _WORKER_SCORER)


def _screen_source_game(
    seed: int,
    cfg: ExpandedRootHeadroomConfig,
    scorer: _LegalActionScorer,
) -> dict[str, Any]:
    bot = _GreedyScorerBot(scorer)
    replay = collect_game_replay(
        bot,
        bot,
        seed=seed,
        max_turns=cfg.max_turns,
        repetition_limit=cfg.repetition_limit,
        no_progress_limit=cfg.no_progress_limit,
        include_state_snapshots=True,
    )
    seat, phase = assigned_cell(seed, cfg)
    matching = [
        step
        for step in replay.steps
        if int(step.player_id) == seat and phase_of_turn(int(step.turn_index)) == phase
    ]
    candidate = None
    if matching:
        median_turn = statistics.median(int(step.turn_index) for step in matching)
        ordered = sorted(
            matching,
            key=lambda step: (abs(int(step.turn_index) - median_turn), int(step.turn_index)),
        )
        for step in ordered:
            if step.state_snapshot is None:
                raise RuntimeError("085 source replay omitted a required state snapshot.")
            state = deserialize_state_snapshot(step.state_snapshot)
            structure = _prepare_structure(state, cfg, scorer, seed, int(step.turn_index))
            if not structure["champion_eligible"]:
                continue
            candidate = {
                "source_seed": seed,
                "turn_index": int(step.turn_index),
                "actor_seat": int(step.player_id),
                "phase": phase,
                "selection_cell": f"seat{seat}|{phase}",
                "state_snapshot": step.state_snapshot,
                "champion_root_keys": structure["champion_root_keys"],
                "expanded_root_keys": structure["expanded_root_keys"],
                "expanded_root_count": structure["expanded_root_count"],
                "excluded_root_reasons": structure["excluded_root_reasons"],
            }
            break
    return {
        "seed": seed,
        "cell": f"seat{seat}|{phase}",
        "turns": replay.turns,
        "termination_reason": replay.termination_reason,
        "stalled": replay.stalled,
        "timed_out": replay.timed_out,
        "loop_fallback_triggers_by_seat": list(replay.loop_fallback_triggers_by_seat),
        "selection_candidate": candidate,
    }


def assigned_cell(seed: int, cfg: ExpandedRootHeadroomConfig) -> tuple[int, str]:
    cell = (seed - cfg.source_seed_start) % 6
    return cell // 3, PHASES[cell % 3]


def phase_of_turn(turn: int) -> str:
    return "early" if turn <= 19 else "mid" if turn <= 49 else "late"


def belief_root_index(
    cfg: ExpandedRootHeadroomConfig,
    source_seed: int,
    turn_index: int,
) -> int:
    return (
        cfg.belief_root_base
        + (source_seed - cfg.source_seed_start) * 400
        + turn_index
    )


def continuation_seed(
    cfg: ExpandedRootHeadroomConfig,
    sample_index: int,
    determinization: int,
    continuation: int,
) -> int:
    return (
        cfg.continuation_seed_base
        + sample_index * 1_000_000_000
        + determinization * 10_000
        + continuation
    )


def _prepare_structure(
    state: SplendorState,
    cfg: ExpandedRootHeadroomConfig,
    scorer: Any,
    source_seed: int,
    turn_index: int,
) -> dict[str, Any]:
    env = SplendorEnv()
    legal = env.legal_actions(state)
    ranked = _distinct_canonical_scores(state, scorer.score_actions(state, legal))
    if len(ranked) < cfg.champion_roots:
        return {
            "champion_eligible": False,
            "champion_failure": "fewer_than_four_root_actions",
            "champion_root_keys": [],
            "expanded_root_keys": [],
            "expanded_root_count": 0,
            "excluded_root_reasons": {"fewer_than_four_root_actions": 1},
            "branches": [],
        }
    determinizations = deterministic_belief_range(
        state,
        state.current_player,
        root_index=belief_root_index(cfg, source_seed, turn_index),
        count=cfg.determinizations,
    )
    retained_keys: list[tuple[object, ...]] = []
    retained_branches: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    champion_failure = None
    for rank_index, row in enumerate(ranked):
        root_key = canonical_successor_action_key(state, row.action)
        if rank_index >= cfg.champion_roots and row.action.action_type == ActionType.RESERVE_DECK:
            excluded["omitted_reserve_deck"] += 1
            continue
        try:
            branches = _prepare_single_root(
                determinizations,
                root_key,
                cfg,
                scorer,
            )
        except RootIneligible as exc:
            excluded[exc.reason] += 1
            if rank_index < cfg.champion_roots:
                champion_failure = exc.reason
                break
            continue
        retained_keys.append(root_key)
        retained_branches.extend(branches)
    champion_eligible = champion_failure is None and len(retained_keys) >= cfg.champion_roots
    if not champion_eligible:
        return {
            "champion_eligible": False,
            "champion_failure": champion_failure or "incomplete_champion_roots",
            "champion_root_keys": [repr(key) for key in retained_keys[: cfg.champion_roots]],
            "expanded_root_keys": [],
            "expanded_root_count": 0,
            "excluded_root_reasons": dict(sorted(excluded.items())),
            "branches": [],
        }
    return {
        "champion_eligible": True,
        "champion_failure": None,
        "champion_root_keys": [repr(key) for key in retained_keys[: cfg.champion_roots]],
        "expanded_root_keys": [repr(key) for key in retained_keys],
        "expanded_root_count": len(retained_keys),
        "root_keys": tuple(retained_keys),
        "excluded_root_reasons": dict(sorted(excluded.items())),
        "branches": retained_branches,
    }


def _prepare_single_root(
    determinizations: Sequence[SplendorState],
    root_key: tuple[object, ...],
    cfg: ExpandedRootHeadroomConfig,
    scorer: Any,
) -> list[dict[str, Any]]:
    env = SplendorEnv()
    branches = []
    action_sets: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
    for det_index, determinized in enumerate(determinizations):
        root_map = actions_by_canonical_successor_key(
            determinized,
            env.legal_actions(determinized),
        )
        if root_key not in root_map:
            raise RootIneligible("root_mapping_changed")
        after_root = env.step(determinized, root_map[root_key])
        if after_root.terminal:
            raise RootIneligible("root_action_terminal")
        responses = _distinct_canonical_scores(
            after_root,
            scorer.score_actions(after_root, env.legal_actions(after_root)),
        )
        if len(responses) < cfg.opponent_actions:
            raise RootIneligible("fewer_than_four_opponent_actions")
        responses = responses[: cfg.opponent_actions]
        response_keys = tuple(
            canonical_successor_action_key(after_root, item.action) for item in responses
        )
        information_set = (
            root_key,
            canonical_public_belief_key(after_root, after_root.current_player),
        )
        previous = action_sets.setdefault(information_set, response_keys)
        if previous != response_keys:
            raise RootIneligible("opponent_action_set_unstable")
        branches.append(
            {
                "determinization": det_index,
                "root_key": root_key,
                "after_root": after_root,
                "information_set": information_set,
                "responses": responses,
            }
        )
    return branches


def _evaluate_parallel(
    cfg: ExpandedRootHeadroomConfig,
    identity: dict[str, Any],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    progress_path = cfg.output_dir / "evaluation.progress.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("identity") != identity:
                raise ValueError("Existing 085 evaluation progress identity mismatch.")
            completed[int(payload["sample_index"])] = payload["result"]
    pending = [row for row in selected if int(row["sample_index"]) not in completed]
    print(
        f"[resolver085:evaluate] states={len(selected)} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(
            max_workers=cfg.workers,
            initializer=_initialize_worker,
            initargs=(cfg.checkpoint, cfg.device),
        ) as executor:
            futures = {executor.submit(_evaluate_state_worker, row, cfg): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                sample = futures[future]
                result = future.result()
                sample_index = int(sample["sample_index"])
                completed[sample_index] = result
                _append_jsonl(
                    progress_path,
                    {"identity": identity, "sample_index": sample_index, "result": result},
                )
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver085:evaluate] complete={len(completed)}/{len(selected)} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )
    return [completed[index] for index in sorted(completed)]


def _evaluate_state_worker(
    sample: dict[str, Any],
    cfg: ExpandedRootHeadroomConfig,
) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("085 evaluation worker scorer was not initialized.")
    return _evaluate_state(sample, cfg, _WORKER_SCORER)


def _evaluate_state(
    sample: dict[str, Any],
    cfg: ExpandedRootHeadroomConfig,
    scorer: Any,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(sample["state_snapshot"])
    structure = _prepare_structure(
        state,
        cfg,
        scorer,
        int(sample["source_seed"]),
        int(sample["turn_index"]),
    )
    if not structure["champion_eligible"]:
        raise RuntimeError("085 selected champion root structure no longer reproduces.")
    if structure["champion_root_keys"] != sample["champion_root_keys"]:
        raise RuntimeError("085 champion root keys changed after selection.")
    if structure["expanded_root_keys"] != sample["expanded_root_keys"]:
        raise RuntimeError("085 expanded root keys changed after selection.")

    leaves: list[ResolvedLeaf] = []
    leaf_discovery: list[float] = []
    leaf_evaluation: list[float] = []
    counters: Counter[str] = Counter()
    env = SplendorEnv()
    for branch in structure["branches"]:
        for response in branch["responses"]:
            values = []
            for continuation in range(cfg.continuations_per_leaf):
                outcome = rollout_score_margin_loop_safe(
                    env,
                    scorer,
                    branch["after_root"],
                    response.action,
                    int(sample["actor_seat"]),
                    continuation_seed(
                        cfg,
                        int(sample["sample_index"]),
                        int(branch["determinization"]),
                        continuation,
                    ),
                    cfg.continuation_epsilon,
                    cfg.continuation_temperature,
                    cfg.continuation_top_k,
                    cfg.continuation_max_turns,
                    cfg.score_margin_scale,
                    cfg.max_loop_fallback_triggers,
                )
                if not math.isfinite(outcome.value):
                    raise RuntimeError("085 produced a nonfinite continuation value.")
                values.append(float(outcome.value))
                counters["continuations"] += 1
                counters["resolved_continuations"] += int(outcome.resolved)
                counters["hard_caps"] += int(outcome.hard_cap_reached)
                counters["loop_adjudications"] += int(outcome.loop_adjudicated)
                counters["rollout_fallback_triggers"] += int(outcome.fallback_triggers)
            discovery = statistics.fmean(values[: cfg.discovery_continuations])
            evaluation = statistics.fmean(values[cfg.discovery_continuations :])
            leaf_discovery.append(discovery)
            leaf_evaluation.append(evaluation)
            leaves.append(
                ResolvedLeaf(
                    determinization=int(branch["determinization"]),
                    root_action=branch["root_key"],
                    opponent_information_set=branch["information_set"],
                    opponent_action=canonical_successor_action_key(
                        branch["after_root"], response.action
                    ),
                    discovery_payoff=discovery,
                    evaluation_payoff=evaluation,
                )
            )

    root_keys = structure["root_keys"]
    champion_keys = root_keys[: cfg.champion_roots]
    champion_leaves = [leaf for leaf in leaves if leaf.root_action in champion_keys]
    champion_discovery = resolve_two_ply(
        champion_leaves,
        champion_keys,
        cfg.determinizations,
        payoff="discovery_payoff",
    )
    champion_evaluation = resolve_two_ply(
        champion_leaves,
        champion_keys,
        cfg.determinizations,
        payoff="evaluation_payoff",
    )
    expanded_discovery = resolve_two_ply(
        leaves,
        root_keys,
        cfg.determinizations,
        payoff="discovery_payoff",
    )
    expanded_evaluation = resolve_two_ply(
        leaves,
        root_keys,
        cfg.determinizations,
        payoff="evaluation_payoff",
    )
    champion_selected = int(champion_discovery.best_index)
    expanded_selected = int(expanded_discovery.best_index)
    gain = (
        float(expanded_evaluation.action_values[expanded_selected])
        - float(expanded_evaluation.action_values[champion_selected])
    )
    expected_continuations = (
        len(root_keys)
        * cfg.determinizations
        * cfg.opponent_actions
        * cfg.continuations_per_leaf
    )
    return {
        "sample_index": int(sample["sample_index"]),
        "source_seed": int(sample["source_seed"]),
        "turn_index": int(sample["turn_index"]),
        "actor_seat": int(sample["actor_seat"]),
        "phase": str(sample["phase"]),
        "selection_cell": str(sample["selection_cell"]),
        "champion_root_count": cfg.champion_roots,
        "expanded_root_count": len(root_keys),
        "added_root_count": len(root_keys) - cfg.champion_roots,
        "excluded_root_reasons": structure["excluded_root_reasons"],
        "champion_root_keys": structure["champion_root_keys"],
        "expanded_root_keys": structure["expanded_root_keys"],
        "champion_discovery_values": list(champion_discovery.action_values),
        "champion_evaluation_values": list(champion_evaluation.action_values),
        "expanded_discovery_values": list(expanded_discovery.action_values),
        "expanded_evaluation_values": list(expanded_evaluation.action_values),
        "champion_discovery_best_index": champion_selected,
        "champion_evaluation_best_index": int(champion_evaluation.best_index),
        "expanded_discovery_best_index": expanded_selected,
        "expanded_evaluation_best_index": int(expanded_evaluation.best_index),
        "changed": expanded_selected >= cfg.champion_roots,
        "evaluation_gain": gain,
        "discovery_gain": (
            float(expanded_discovery.action_values[expanded_selected])
            - float(expanded_discovery.action_values[champion_selected])
        ),
        "leaf_discovery": leaf_discovery,
        "leaf_evaluation": leaf_evaluation,
        "counters": {
            "continuations": counters["continuations"],
            "expected_continuations": expected_continuations,
            "resolved_continuations": counters["resolved_continuations"],
            "hard_caps": counters["hard_caps"],
            "loop_adjudications": counters["loop_adjudications"],
            "rollout_fallback_triggers": counters["rollout_fallback_triggers"],
        },
    }


def _summarize(
    rows: list[dict[str, Any]],
    cfg: ExpandedRootHeadroomConfig,
    evaluation_wall_seconds: float,
) -> dict[str, Any]:
    gains = [float(row["evaluation_gain"]) for row in rows]
    gain_ci = _bootstrap_mean_ci(gains, cfg.bootstrap_samples, cfg.bootstrap_seed)
    leaf_discovery = [value for row in rows for value in row["leaf_discovery"]]
    leaf_evaluation = [value for row in rows for value in row["leaf_evaluation"]]
    champion_discovery = [
        value for row in rows for value in row["champion_discovery_values"]
    ]
    champion_evaluation = [
        value for row in rows for value in row["champion_evaluation_values"]
    ]
    expanded_discovery = [
        value for row in rows for value in row["expanded_discovery_values"]
    ]
    expanded_evaluation = [
        value for row in rows for value in row["expanded_evaluation_values"]
    ]
    sign_indices = [
        index
        for index, (left, right) in enumerate(
            zip(leaf_discovery, leaf_evaluation, strict=True)
        )
        if abs(left) >= 0.025 and abs(right) >= 0.025
    ]
    phase_breakdown = {}
    for phase_index, phase in enumerate(PHASES):
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_gains = [float(row["evaluation_gain"]) for row in phase_rows]
        phase_ci = _bootstrap_mean_ci(
            phase_gains,
            cfg.bootstrap_samples,
            cfg.bootstrap_seed + phase_index + 1,
        )
        phase_breakdown[phase] = {
            "states": len(phase_rows),
            "changed": sum(bool(row["changed"]) for row in phase_rows),
            "change_fraction": sum(bool(row["changed"]) for row in phase_rows)
            / len(phase_rows),
            "mean_gain": statistics.fmean(phase_gains),
            "gain_ci95_low": phase_ci[0],
            "gain_ci95_high": phase_ci[1],
        }
    counters = Counter()
    excluded = Counter()
    for row in rows:
        counters.update({key: int(value) for key, value in row["counters"].items()})
        excluded.update(row["excluded_root_reasons"])
    continuations = counters["continuations"]
    return {
        "states": len(rows),
        "mean_evaluation_gain": statistics.fmean(gains),
        "evaluation_gain_ci95_low": gain_ci[0],
        "evaluation_gain_ci95_high": gain_ci[1],
        "changed_states": sum(bool(row["changed"]) for row in rows),
        "change_fraction": sum(bool(row["changed"]) for row in rows) / len(rows),
        "mean_expanded_root_count": statistics.fmean(
            int(row["expanded_root_count"]) for row in rows
        ),
        "minimum_expanded_root_count": min(int(row["expanded_root_count"]) for row in rows),
        "maximum_expanded_root_count": max(int(row["expanded_root_count"]) for row in rows),
        "excluded_root_reasons": dict(sorted(excluded.items())),
        "leaf_payoff_pearson": _pearson(leaf_discovery, leaf_evaluation),
        "leaf_payoff_mae": statistics.fmean(
            abs(left - right)
            for left, right in zip(leaf_discovery, leaf_evaluation, strict=True)
        ),
        "leaf_sign_eligible": len(sign_indices),
        "leaf_sign_agreement": (
            sum(
                leaf_discovery[index] * leaf_evaluation[index] > 0.0
                for index in sign_indices
            )
            / len(sign_indices)
            if sign_indices
            else 0.0
        ),
        "champion_root_value_pearson": _pearson(
            champion_discovery, champion_evaluation
        ),
        "expanded_root_value_pearson": _pearson(
            expanded_discovery, expanded_evaluation
        ),
        "champion_best_root_agreement": sum(
            int(row["champion_discovery_best_index"])
            == int(row["champion_evaluation_best_index"])
            for row in rows
        )
        / len(rows),
        "expanded_best_root_agreement": sum(
            int(row["expanded_discovery_best_index"])
            == int(row["expanded_evaluation_best_index"])
            for row in rows
        )
        / len(rows),
        "phase_breakdown": phase_breakdown,
        "continuations": continuations,
        "expected_continuations": counters["expected_continuations"],
        "resolved_continuations": counters["resolved_continuations"],
        "hard_caps": counters["hard_caps"],
        "loop_adjudications": counters["loop_adjudications"],
        "loop_adjudication_fraction": (
            counters["loop_adjudications"] / continuations if continuations else 1.0
        ),
        "rollout_fallback_triggers": counters["rollout_fallback_triggers"],
        "evaluation_wall_seconds": evaluation_wall_seconds,
    }


def _selection_gates(
    cfg: ExpandedRootHeadroomConfig,
    selection: dict[str, Any],
) -> dict[str, bool]:
    selected_counts = selection.get("selected_by_cell", {})
    return {
        "exact_600_clean_source_games": int(selection.get("source_games", 0))
        == cfg.source_games
        and int(selection.get("source_invalid_games", 1)) == 0,
        "exact_source_seed_range": int(selection.get("source_seed_min", -1))
        == cfg.source_seed_start
        and int(selection.get("source_seed_max", -1))
        == cfg.source_seed_start + cfg.source_games - 1,
        "exact_six_cell_balance": set(selected_counts) == set(CELLS)
        and all(int(selected_counts[cell]) == cfg.states_per_cell for cell in CELLS),
        "exact_300_unique_game_states": int(selection.get("selected_count", 0))
        == cfg.states_per_cell * len(CELLS)
        and int(selection.get("unique_source_games", 0))
        == cfg.states_per_cell * len(CELLS)
        and int(selection.get("unique_states", 0))
        == cfg.states_per_cell * len(CELLS),
        "outcome_blind_selection": bool(selection.get("outcome_blind")),
    }


def _integrity_gates(
    cfg: ExpandedRootHeadroomConfig,
    selection: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    privacy: dict[str, Any],
    repeat_identical: bool,
    evaluation_wall_seconds: float,
) -> dict[str, bool]:
    return {
        "frozen_protocol_checkpoint_and_configuration": _provenance_matches(cfg),
        "exact_selected_state_coverage": len(rows) == cfg.states_per_cell * len(CELLS)
        and {int(row["sample_index"]) for row in rows}
        == set(range(cfg.states_per_cell * len(CELLS))),
        "champion_and_expanded_root_contracts_reproduce": all(
            int(row["champion_root_count"]) == cfg.champion_roots
            and int(row["expanded_root_count"]) >= cfg.champion_roots
            for row in rows
        ),
        "exact_variable_branch_accounting": int(summary["continuations"])
        == int(summary["expected_continuations"])
        and all(
            int(row["counters"]["continuations"])
            == int(row["counters"]["expected_continuations"])
            for row in rows
        ),
        "all_continuations_resolved": int(summary["resolved_continuations"])
        == int(summary["continuations"]),
        "zero_hard_caps": int(summary["hard_caps"]) == 0,
        "zero_rollout_fallback_triggers": int(summary["rollout_fallback_triggers"]) == 0,
        "bounded_loop_adjudication": float(summary["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction,
        "reliable_leaf_halves": float(summary["leaf_payoff_pearson"])
        >= cfg.minimum_leaf_pearson
        and float(summary["leaf_payoff_mae"]) <= cfg.maximum_leaf_mae
        and float(summary["leaf_sign_agreement"]) >= cfg.minimum_leaf_sign_agreement,
        "reliable_champion_root_values": float(summary["champion_root_value_pearson"])
        >= cfg.minimum_root_pearson
        and float(summary["champion_best_root_agreement"])
        >= cfg.minimum_best_root_agreement,
        "reliable_expanded_root_values": float(summary["expanded_root_value_pearson"])
        >= cfg.minimum_root_pearson
        and float(summary["expanded_best_root_agreement"])
        >= cfg.minimum_best_root_agreement,
        "privacy_invariance": bool(privacy.get("passed")),
        "first_state_repeat_identical": repeat_identical,
        "within_six_hour_operational_ceiling": evaluation_wall_seconds
        <= cfg.maximum_evaluation_wall_seconds,
        "no_human_input": "human" not in json.dumps(_config_payload(cfg)).lower(),
    }


def _signal_gates(
    cfg: ExpandedRootHeadroomConfig,
    summary: dict[str, Any],
) -> dict[str, bool]:
    return {
        "mean_gain_at_least_0p05": float(summary["mean_evaluation_gain"])
        >= cfg.minimum_mean_gain,
        "gain_ci95_lower_above_zero": float(summary["evaluation_gain_ci95_low"]) > 0.0,
        "change_fraction_at_least_15pct": float(summary["change_fraction"])
        >= cfg.minimum_change_fraction,
        "at_least_one_change_every_phase": all(
            int(summary["phase_breakdown"][phase]["changed"]) > 0 for phase in PHASES
        ),
    }


def _privacy_audit(
    sample: dict[str, Any],
    cfg: ExpandedRootHeadroomConfig,
    scorer: Any,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(sample["state_snapshot"])
    permuted = state.copy_shallow()
    for tier in (1, 2, 3):
        permuted.hidden_tier_decks[tier].reverse()
    root_index = belief_root_index(
        cfg,
        int(sample["source_seed"]),
        int(sample["turn_index"]),
    )
    original_range = deterministic_belief_range(
        state,
        state.current_player,
        root_index=root_index,
        count=cfg.determinizations,
    )
    permuted_range = deterministic_belief_range(
        permuted,
        permuted.current_player,
        root_index=root_index,
        count=cfg.determinizations,
    )
    original_structure = _prepare_structure(
        state,
        cfg,
        scorer,
        int(sample["source_seed"]),
        int(sample["turn_index"]),
    )
    permuted_structure = _prepare_structure(
        permuted,
        cfg,
        scorer,
        int(sample["source_seed"]),
        int(sample["turn_index"]),
    )
    checks = {
        "public_key_invariant": canonical_public_belief_key(
            state, state.current_player
        )
        == canonical_public_belief_key(permuted, permuted.current_player),
        "deterministic_belief_range_invariant": all(
            state_signature(left) == state_signature(right)
            for left, right in zip(original_range, permuted_range, strict=True)
        ),
        "champion_roots_invariant": original_structure["champion_root_keys"]
        == permuted_structure["champion_root_keys"],
        "expanded_roots_invariant": original_structure["expanded_root_keys"]
        == permuted_structure["expanded_root_keys"],
        "excluded_reasons_invariant": original_structure["excluded_root_reasons"]
        == permuted_structure["excluded_root_reasons"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def _bootstrap_mean_ci(
    values: Sequence[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values or samples <= 0:
        raise ValueError("Bootstrap requires values and positive sample count.")
    rng = random.Random(seed)
    size = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    return draws[int(0.025 * samples)], draws[min(int(0.975 * samples), samples - 1)]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    covariance = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return covariance / denominator if denominator > 0.0 else 0.0


def _deterministic_projection(row: dict[str, Any]) -> dict[str, Any]:
    return row


def _provenance_matches(cfg: ExpandedRootHeadroomConfig) -> bool:
    return (
        cfg == ExpandedRootHeadroomConfig()
        and _sha256_file(PROTOCOL) == PROTOCOL_SHA256
        and _sha256_file(cfg.checkpoint) == CHECKPOINT_SHA256
    )


def _identity(cfg: ExpandedRootHeadroomConfig) -> dict[str, Any]:
    return {
        "configuration": _config_payload(cfg),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "checkpoint_sha256": _sha256_file(cfg.checkpoint),
        "root_contract": "top4_plus_all_stable_non_deck_public_roots_v1",
        "continuation_contract": "3x4x8_split4_discovery4_evaluation_v1",
    }


def _validate_config(cfg: ExpandedRootHeadroomConfig) -> None:
    if cfg != ExpandedRootHeadroomConfig():
        raise ValueError("Experiment 085 permits only its preregistered default configuration.")
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(cfg.checkpoint)
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 085 protocol, checkpoint, or configuration mismatch.")
    if cfg.discovery_continuations * 2 != cfg.continuations_per_leaf:
        raise ValueError("085 requires equal four-continuation discovery/evaluation halves.")
    if cfg.continuation_seed_base + cfg.states_per_cell * len(CELLS) * 1_000_000_000 >= 2**63:
        raise ValueError("085 continuation namespace exceeds signed 64-bit range.")


def _config_payload(cfg: ExpandedRootHeadroomConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
