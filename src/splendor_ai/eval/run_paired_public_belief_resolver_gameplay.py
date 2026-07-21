"""Run preregistered experiment 050: paired public-belief resolver gameplay."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.cfr import (
    ResolvedLeaf,
    TwoPlyResolution,
    deterministic_belief_range,
    resolve_two_ply,
)
from splendor_ai.encoding import canonical_public_belief_key, canonical_successor_action_key
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    ResolvingConfig,
    _sha256_file,
)
from splendor_ai.eval.run_shallow_operational_gate import summarize_pair_rows
from splendor_ai.training.run_action_preference_repair import (
    rollout_score_margin_loop_safe,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


PHASES = ("early", "mid", "late")
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")
EXPECTED_BRANCHES_PER_SEARCH = 192


@dataclass(frozen=True, slots=True)
class ResolverGameplayConfig:
    output_dir: Path = Path("outputs/paired_public_belief_resolver_gameplay_050")
    checkpoint: Path = CHECKPOINT
    smoke_seed: int = 500_000
    direct_seed_start: int = 500_100
    direct_pairs: int = 100
    determinizations: int = 3
    root_actions: int = 4
    opponent_actions: int = 4
    discovery_continuations: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_index_start: int = 1_000_000
    continuation_seed_start: int = 56_000_000
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    minimum_score_share: float = 0.55
    minimum_seat_score_share: float = 0.50
    minimum_switches: int = 8
    maximum_loop_adjudicated_fraction: float = 0.001
    device: str = "cpu"


class ResolverIneligible(RuntimeError):
    """A frozen structural condition requires raw-035 fallback for one turn."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FrozenPublicBeliefResolverBot:
    """Deploy the exact discovery half of the validated 049b resolver."""

    def __init__(
        self,
        config: ResolverGameplayConfig,
        engine_seed: int,
        actor_seat: int,
        *,
        scorer: Any | None = None,
    ) -> None:
        if actor_seat not in (0, 1):
            raise ValueError("actor_seat must be 0 or 1.")
        self._config = config
        self._engine_seed = engine_seed
        self._actor_seat = actor_seat
        self._scorer = scorer or _LegalActionScorer(
            config.checkpoint,
            config.device,
            "legal_action_attention_scorer",
        )
        self._decisions = 0
        self._searches = 0
        self._switches = 0
        self._fallback_reasons: Counter[str] = Counter()
        self._continuations = 0
        self._resolved_continuations = 0
        self._hard_caps = 0
        self._loop_adjudications = 0
        self._rollout_fallback_triggers = 0
        self._by_phase: dict[str, Counter[str]] = {
            phase: Counter() for phase in PHASES
        }

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    @property
    def telemetry(self) -> dict[str, Any]:
        return {
            "engine_seed": self._engine_seed,
            "actor_seat": self._actor_seat,
            "decisions": self._decisions,
            "completed_searches": self._searches,
            "switches": self._switches,
            "fallbacks": sum(self._fallback_reasons.values()),
            "fallback_reasons": dict(sorted(self._fallback_reasons.items())),
            "continuations": self._continuations,
            "resolved_continuations": self._resolved_continuations,
            "hard_caps": self._hard_caps,
            "loop_adjudications": self._loop_adjudications,
            "rollout_fallback_triggers": self._rollout_fallback_triggers,
            "by_phase": {
                phase: {
                    "decisions": counts["decisions"],
                    "completed_searches": counts["completed_searches"],
                    "switches": counts["switches"],
                    "fallbacks": counts["fallbacks"],
                }
                for phase, counts in self._by_phase.items()
            },
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
        if state.current_player != self._actor_seat:
            raise RuntimeError("Resolver bot was called for the wrong actor seat.")
        decision_index = self._decisions
        if decision_index >= 400:
            raise RuntimeError("Resolver decision index exceeds the frozen namespace.")
        self._decisions += 1
        phase = _phase(state.turn_index)
        self._by_phase[phase]["decisions"] += 1
        all_scores = self._scorer.score_actions(state, actions)
        ranked = _distinct_canonical_scores(state, all_scores)
        if not ranked:
            raise RuntimeError("035 scorer returned no ranked legal action.")
        parent = ranked[0].action
        try:
            selected, counters, resolution = self._resolve(
                state,
                actions,
                ranked,
                decision_index,
            )
        except ResolverIneligible as exc:
            self._fallback_reasons[exc.reason] += 1
            self._by_phase[phase]["fallbacks"] += 1
            self._on_fallback_decision(
                state=state,
                legal_actions=actions,
                all_scores=all_scores,
                parent=parent,
                decision_index=decision_index,
                reason=exc.reason,
            )
            return parent
        if selected not in actions:
            raise RuntimeError("Resolver selected an illegal concrete action.")
        self._searches += 1
        self._by_phase[phase]["completed_searches"] += 1
        self._continuations += counters["continuations"]
        self._resolved_continuations += counters["resolved_continuations"]
        self._hard_caps += counters["hard_caps"]
        self._loop_adjudications += counters["loop_adjudications"]
        self._rollout_fallback_triggers += counters["rollout_fallback_triggers"]
        if canonical_successor_action_key(state, selected) != canonical_successor_action_key(
            state, parent
        ):
            self._switches += 1
            self._by_phase[phase]["switches"] += 1
        self._on_resolved_decision(
            state=state,
            legal_actions=actions,
            all_scores=all_scores,
            parent=parent,
            selected=selected,
            decision_index=decision_index,
            resolution=resolution,
        )
        return selected

    def _on_resolved_decision(
        self,
        *,
        state: SplendorState,
        legal_actions: list[Action],
        all_scores: tuple[Any, ...],
        parent: Action,
        selected: Action,
        decision_index: int,
        resolution: TwoPlyResolution,
    ) -> None:
        """Optional observation-only hook; the frozen gameplay bot is a no-op."""

    def _on_fallback_decision(
        self,
        *,
        state: SplendorState,
        legal_actions: list[Action],
        all_scores: tuple[Any, ...],
        parent: Action,
        decision_index: int,
        reason: str,
    ) -> None:
        """Optional observation-only hook; the frozen gameplay bot is a no-op."""

    def _resolve(
        self,
        state: SplendorState,
        legal_actions: list[Action],
        ranked: tuple[Any, ...],
        decision_index: int,
    ) -> tuple[Action, Counter[str], TwoPlyResolution]:
        cfg = self._config
        if len(ranked) < cfg.root_actions:
            raise ResolverIneligible("fewer_than_four_root_actions")
        root_keys = tuple(
            canonical_successor_action_key(state, row.action)
            for row in ranked[: cfg.root_actions]
        )
        root_index = gameplay_belief_root_index(
            cfg,
            self._engine_seed,
            self._actor_seat,
            decision_index,
        )
        determinizations = deterministic_belief_range(
            state,
            state.current_player,
            root_index=root_index,
            count=cfg.determinizations,
        )
        branches = []
        action_sets: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
        env = SplendorEnv()
        for det_index, determinized in enumerate(determinizations):
            root_action_map = actions_by_canonical_successor_key(
                determinized,
                env.legal_actions(determinized),
            )
            if any(key not in root_action_map for key in root_keys):
                raise ResolverIneligible("root_mapping_changed")
            for root_key in root_keys:
                after_root = env.step(determinized, root_action_map[root_key])
                if after_root.terminal:
                    raise ResolverIneligible("root_action_terminal")
                opponent_ranked = _distinct_canonical_scores(
                    after_root,
                    self._scorer.score_actions(
                        after_root,
                        env.legal_actions(after_root),
                    ),
                )
                if len(opponent_ranked) < cfg.opponent_actions:
                    raise ResolverIneligible("fewer_than_four_opponent_actions")
                responses = opponent_ranked[: cfg.opponent_actions]
                response_keys = tuple(
                    canonical_successor_action_key(after_root, row.action)
                    for row in responses
                )
                information_set = (
                    root_key,
                    canonical_public_belief_key(after_root, after_root.current_player),
                )
                previous = action_sets.setdefault(information_set, response_keys)
                if previous != response_keys:
                    raise ResolverIneligible("opponent_action_set_unstable")
                branches.append(
                    (det_index, root_key, after_root, information_set, responses)
                )

        leaves: list[ResolvedLeaf] = []
        counters: Counter[str] = Counter()
        for det_index, root_key, after_root, information_set, responses in branches:
            for response in responses:
                response_key = canonical_successor_action_key(after_root, response.action)
                values = []
                for continuation_index in range(cfg.discovery_continuations):
                    seed = gameplay_continuation_seed(
                        cfg,
                        self._engine_seed,
                        self._actor_seat,
                        decision_index,
                        det_index,
                        continuation_index,
                    )
                    outcome = rollout_score_margin_loop_safe(
                        env,
                        self._scorer,
                        after_root,
                        response.action,
                        self._actor_seat,
                        seed,
                        cfg.continuation_epsilon,
                        cfg.continuation_temperature,
                        cfg.continuation_top_k,
                        cfg.continuation_max_turns,
                        cfg.score_margin_scale,
                        cfg.max_loop_fallback_triggers,
                    )
                    if not math.isfinite(outcome.value):
                        raise RuntimeError("Resolver produced a nonfinite leaf value.")
                    values.append(outcome.value)
                    counters["continuations"] += 1
                    counters["resolved_continuations"] += int(outcome.resolved)
                    counters["hard_caps"] += int(outcome.hard_cap_reached)
                    counters["loop_adjudications"] += int(outcome.loop_adjudicated)
                    counters["rollout_fallback_triggers"] += outcome.fallback_triggers
                value = statistics.fmean(values)
                leaves.append(
                    ResolvedLeaf(
                        determinization=det_index,
                        root_action=root_key,
                        opponent_information_set=information_set,
                        opponent_action=response_key,
                        discovery_payoff=value,
                        evaluation_payoff=value,
                    )
                )
        resolution = resolve_two_ply(
            leaves,
            root_keys,
            cfg.determinizations,
            payoff="discovery_payoff",
        )
        actual_map = actions_by_canonical_successor_key(state, legal_actions)
        selected_key = root_keys[resolution.best_index]
        if selected_key not in actual_map:
            raise RuntimeError("Resolved semantic action is absent from the actual state.")
        return actual_map[selected_key], counters, resolution


def run_gate(config: ResolverGameplayConfig | None = None) -> dict[str, Any]:
    cfg = config or ResolverGameplayConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    smoke = _run_smoke(cfg)
    smoke_passed = all(smoke["gates"].values())
    direct = _run_direct(cfg) if smoke_passed else None
    direct_gates = _direct_gates(cfg, direct) if direct is not None else None
    passed = bool(smoke_passed and direct_gates and all(direct_gates.values()))
    result = {
        "experiment": "050_paired_public_belief_resolver_gameplay",
        "protocol": "docs/experiments/paired_public_belief_resolver_gameplay_050.md",
        "configuration": _config_payload(cfg),
        "checkpoint_sha256": _sha256_file(cfg.checkpoint),
        "frozen_contract_matches_049b": _frozen_contract_matches_049(cfg),
        "smoke": smoke,
        "smoke_passed": smoke_passed,
        "direct": direct,
        "direct_gates": direct_gates,
        "passed": passed,
        "training_authorized": False,
        "gpu_rental_authorized": False,
        "decision": (
            "authorize_separately_preregistered_broad_resolver_confirmation"
            if passed
            else "reject_frozen_resolver_gameplay"
            if smoke_passed
            else "operational_smoke_failed_do_not_run_direct"
        ),
    }
    _atomic_write(cfg.output_dir / "result.json", result)
    print(f"050 result: {'PASS' if passed else 'FAIL'}", flush=True)
    return result


def gameplay_belief_root_index(
    cfg: ResolverGameplayConfig,
    engine_seed: int,
    actor_seat: int,
    decision_index: int,
) -> int:
    return (
        cfg.belief_root_index_start
        + (engine_seed - cfg.smoke_seed) * 1_000
        + actor_seat * 400
        + decision_index
    )


def gameplay_continuation_seed(
    cfg: ResolverGameplayConfig,
    engine_seed: int,
    actor_seat: int,
    decision_index: int,
    determinization_index: int,
    continuation_index: int,
) -> int:
    return (
        cfg.continuation_seed_start
        + (engine_seed - cfg.smoke_seed) * 1_000_000_000
        + actor_seat * 400_000_000
        + decision_index * 1_000_000
        + determinization_index * 10_000
        + continuation_index
    )


def _run_smoke(cfg: ResolverGameplayConfig) -> dict[str, Any]:
    path = cfg.output_dir / "smoke.json"
    identity = _identity(cfg, "smoke", cfg.smoke_seed, 1)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("Existing 050 smoke does not match the frozen protocol.")
        return payload
    print("[resolver050:smoke] running original and from-zero repeat", flush=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_play_pair, cfg, cfg.smoke_seed) for _ in range(2)]
        original, repeated = [future.result() for future in futures]
    rows = [original, repeated]
    telemetry = _summarize_telemetry(rows)
    integrity = _integrity(rows)
    repeat_identical = _deterministic_projection(original) == _deterministic_projection(repeated)
    gates = {
        "frozen_checkpoint_and_configuration": (
            cfg == ResolverGameplayConfig()
            and _sha256_file(cfg.checkpoint) == CHECKPOINT_SHA256
            and _frozen_contract_matches_049(cfg)
        ),
        "four_clean_games": integrity["games"] == 4 and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "resolver_exercised": telemetry["completed_searches"] > 0,
        "exact_branch_accounting": telemetry["exact_branch_accounting"],
        "all_continuations_resolved": telemetry["all_continuations_resolved"],
        "zero_hard_caps": telemetry["hard_caps"] == 0,
        "bounded_loop_adjudication": telemetry["loop_adjudication_fraction"]
        <= cfg.maximum_loop_adjudicated_fraction,
        "from_zero_repeat_identical": repeat_identical,
    }
    payload = {
        "format": "paired_public_belief_resolver_smoke_v1",
        "identity": identity,
        "integrity": integrity,
        "telemetry": telemetry,
        "repeat_identical": repeat_identical,
        "gates": gates,
        "passed": all(gates.values()),
        "original": original,
        "repeated": repeated,
    }
    _atomic_write(path, payload)
    return payload


def _run_direct(cfg: ResolverGameplayConfig) -> dict[str, Any]:
    path = cfg.output_dir / "direct_vs_035.json"
    progress_path = cfg.output_dir / "direct_vs_035.progress.json"
    identity = _identity(cfg, "direct_vs_035", cfg.direct_seed_start, cfg.direct_pairs)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("Existing 050 direct result does not match the frozen protocol.")
        return payload
    completed: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("Existing 050 direct progress does not match the frozen protocol.")
        completed = {int(row["seed"]): row for row in progress.get("pairs", [])}
    seeds = list(range(cfg.direct_seed_start, cfg.direct_seed_start + cfg.direct_pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(
        f"[resolver050:direct] pairs={cfg.direct_pairs} complete={len(completed)} "
        f"pending={len(pending)} workers={cfg.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(_play_pair, cfg, seed): seed for seed in pending}
            for index, future in enumerate(as_completed(futures), start=1):
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
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0.0 else 0.0
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver050:direct] complete={len(completed)}/{cfg.direct_pairs} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )
    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, cfg.bootstrap_samples, cfg.direct_seed_start)
    summary["seat_breakdown"] = _seat_breakdown(rows)
    payload = {
        "format": "paired_public_belief_resolver_direct_v1",
        "identity": identity,
        "summary": summary,
        "integrity": _integrity(rows),
        "telemetry": _summarize_telemetry(rows),
        "pairs": rows,
    }
    _atomic_write(path, payload)
    _atomic_write(
        progress_path,
        {"identity": identity, "completed": True, "pairs": rows},
    )
    return payload


def _play_pair(cfg: ResolverGameplayConfig, seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    games = []
    telemetry = []
    for candidate_seat in (0, 1):
        candidate = FrozenPublicBeliefResolverBot(cfg, seed, candidate_seat)
        baseline = LegalActionAttentionScorerBot(cfg.checkpoint, cfg.device)
        bots = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
        game = play_game(
            bots[0],
            bots[1],
            seed=seed,
            max_turns=cfg.max_turns,
            repetition_limit=cfg.repetition_limit,
            no_progress_limit=cfg.no_progress_limit,
        )
        games.append(asdict(game))
        telemetry.append(candidate.telemetry)
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": games,
        "candidate_telemetry": telemetry,
    }


def _summarize_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = [item for row in rows for item in row["candidate_telemetry"]]
    totals = Counter()
    fallback_reasons: Counter[str] = Counter()
    phase_totals = {phase: Counter() for phase in PHASES}
    for item in items:
        for key in (
            "decisions",
            "completed_searches",
            "switches",
            "fallbacks",
            "continuations",
            "resolved_continuations",
            "hard_caps",
            "loop_adjudications",
            "rollout_fallback_triggers",
        ):
            totals[key] += int(item[key])
        fallback_reasons.update(item["fallback_reasons"])
        for phase in PHASES:
            phase_totals[phase].update(item["by_phase"][phase])
    continuations = totals["continuations"]
    searches = totals["completed_searches"]
    return {
        **dict(totals),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "expected_branches_per_search": EXPECTED_BRANCHES_PER_SEARCH,
        "exact_branch_accounting": continuations
        == searches * EXPECTED_BRANCHES_PER_SEARCH,
        "all_continuations_resolved": totals["resolved_continuations"] == continuations,
        "loop_adjudication_fraction": totals["loop_adjudications"] / continuations
        if continuations
        else 1.0,
        "by_phase": {phase: dict(counts) for phase, counts in phase_totals.items()},
    }


def _integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = [game for row in rows for game in row["games"]]
    reasons = Counter(str(game["termination_reason"]) for game in games)
    invalid = [
        game
        for game in games
        if game["termination_reason"] not in VALID_TERMINATIONS
        or bool(game["timed_out"])
        or bool(game["stalled"])
    ]
    return {
        "games": len(games),
        "termination_reasons": dict(sorted(reasons.items())),
        "invalid_games": len(invalid),
        "timed_out_games": sum(bool(game["timed_out"]) for game in games),
        "stalled_games": sum(bool(game["stalled"]) for game in games),
    }


def _direct_gates(
    cfg: ResolverGameplayConfig,
    direct: dict[str, Any],
) -> dict[str, bool]:
    summary = direct["summary"]
    integrity = direct["integrity"]
    telemetry = direct["telemetry"]
    phase_active = all(
        int(telemetry["by_phase"][phase].get("completed_searches", 0)) > 0
        and int(telemetry["by_phase"][phase].get("switches", 0)) > 0
        for phase in PHASES
    )
    return {
        "frozen_checkpoint_and_configuration": (
            cfg == ResolverGameplayConfig()
            and _sha256_file(cfg.checkpoint) == CHECKPOINT_SHA256
            and _frozen_contract_matches_049(cfg)
        ),
        "exactly_200_clean_games": integrity["games"] == cfg.direct_pairs * 2
        and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "exact_branch_accounting": bool(telemetry["exact_branch_accounting"]),
        "all_continuations_resolved": bool(telemetry["all_continuations_resolved"]),
        "zero_hard_caps": int(telemetry["hard_caps"]) == 0,
        "bounded_loop_adjudication": float(telemetry["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction,
        "resolver_active_and_switches_all_phases": phase_active
        and int(telemetry["switches"]) >= cfg.minimum_switches,
        "score_share_at_least_55pct": float(summary["score_share"]["mean"])
        >= cfg.minimum_score_share,
        "score_share_ci95_lower_above_50pct": float(
            summary["score_share"]["ci95_low"]
        )
        > 0.50,
        "positive_mean_paired_margin": float(summary["final_score_margin"]["mean"])
        > 0.0,
        "no_seat_collapse": all(
            float(item["score_share"]) >= cfg.minimum_seat_score_share
            for item in summary["seat_breakdown"].values()
        ),
    }


def _seat_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[int, list[float]] = {0: [], 1: []}
    for row in rows:
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
            winner = game["winner"]
            points = 0.5 if winner is None else 1.0 if int(winner) == int(seat) else 0.0
            buckets[int(seat)].append(points)
    return {
        f"seat{seat}": {
            "games": len(points),
            "score_share": statistics.fmean(points),
        }
        for seat, points in buckets.items()
    }


def _deterministic_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": row["seed"],
        "candidate_seats": row["candidate_seats"],
        "games": row["games"],
        "candidate_telemetry": row["candidate_telemetry"],
    }


def _identity(
    cfg: ResolverGameplayConfig,
    name: str,
    seed_start: int,
    pairs: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "seed_start": seed_start,
        "pairs": pairs,
        "checkpoint": str(cfg.checkpoint),
        "checkpoint_sha256": _sha256_file(cfg.checkpoint),
        "configuration": _config_payload(cfg),
        "valid_terminations": list(VALID_TERMINATIONS),
        "expected_branches_per_search": EXPECTED_BRANCHES_PER_SEARCH,
    }


def _frozen_contract_matches_049(cfg: ResolverGameplayConfig) -> bool:
    original = ResolvingConfig()
    same_fields = (
        "checkpoint",
        "determinizations",
        "root_actions",
        "opponent_actions",
        "continuation_epsilon",
        "continuation_temperature",
        "continuation_top_k",
        "continuation_max_turns",
        "max_loop_fallback_triggers",
        "score_margin_scale",
        "workers",
        "bootstrap_samples",
        "device",
    )
    return (
        all(getattr(cfg, field) == getattr(original, field) for field in same_fields)
        and cfg.discovery_continuations == original.continuations_per_leaf // 2
        and EXPECTED_BRANCHES_PER_SEARCH
        == cfg.determinizations
        * cfg.root_actions
        * cfg.opponent_actions
        * cfg.discovery_continuations
    )


def _validate_config(cfg: ResolverGameplayConfig) -> None:
    if cfg != ResolverGameplayConfig():
        raise ValueError("Experiment 050 permits only its preregistered default configuration.")
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(cfg.checkpoint)
    if _sha256_file(cfg.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Raw 035 checkpoint hash mismatch.")
    if not _frozen_contract_matches_049(cfg):
        raise ValueError("Resolver or leaf contract differs from 049/049b.")


def _distinct_canonical_scores(
    state: SplendorState,
    scores: tuple[Any, ...],
) -> tuple[Any, ...]:
    rows = []
    seen = set()
    for row in scores:
        key = canonical_successor_action_key(state, row.action)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return tuple(rows)


def _phase(turn: int) -> str:
    return "early" if turn <= 19 else "mid" if turn <= 49 else "late"


def _config_payload(cfg: ResolverGameplayConfig) -> dict[str, Any]:
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


def main() -> None:
    run_gate()


if __name__ == "__main__":
    main()
