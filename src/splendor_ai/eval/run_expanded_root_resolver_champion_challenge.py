"""Run preregistered experiment 087: expanded-root resolver champion challenge."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import torch

from splendor_ai.cfr import ResolvedLeaf, TwoPlyResolution, deterministic_belief_range, resolve_two_ply
from splendor_ai.encoding import canonical_public_belief_key, canonical_successor_action_key
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_bounded_public_belief_resolving import CHECKPOINT, CHECKPOINT_SHA256, _sha256_file
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
    PHASES,
    VALID_TERMINATIONS,
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
    ResolverIneligible,
    _distinct_canonical_scores,
    _frozen_contract_matches_049,
    _phase,
    gameplay_belief_root_index,
    gameplay_continuation_seed,
)
from splendor_ai.eval.run_shallow_operational_gate import summarize_pair_rows
from splendor_ai.training.run_action_preference_repair import rollout_score_margin_loop_safe
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


PROTOCOL = Path("docs/experiments/expanded_root_resolver_champion_challenge_087.md")
PROTOCOL_SHA256 = "b9c65bee0597036ac17330b18480671e63d6a7193cc67003b6e7017b9c3cb649"
RESULT_084_SHA256 = "1ed98024072b92baf3c30bc896b9dae2484f7d80b727959307dfd83bbfe2cd0f"
RESULT_086_SHA256 = "8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5"
ROOT_CONTRACT = "top4_plus_all_stable_non_deck_public_roots_v1"


@dataclass(frozen=True, slots=True)
class ExpandedRootChampionChallengeConfig:
    output_dir: Path = Path("outputs/expanded_root_resolver_champion_challenge_087")
    checkpoint: Path = CHECKPOINT
    result_084: Path = Path("outputs/resolver_research_champion_confirmation_084/result.json")
    result_086: Path = Path("outputs/expanded_root_regret_stability_replication_086/result.json")
    smoke_seed: int = 870_000
    strength_seed_start: int = 870_100
    pairs: int = 200
    determinizations: int = 3
    champion_roots: int = 4
    opponent_actions: int = 4
    discovery_continuations: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    belief_root_index_start: int = 870_000_000
    continuation_seed_start: int = 8_700_000_000_000_000_000
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 870_100
    expected_score_share: float = 0.565
    minimum_score_share: float = 0.55
    minimum_seat_score_share: float = 0.50
    minimum_expansion_switch_rate: float = 0.15
    maximum_loop_adjudicated_fraction: float = 0.001
    maximum_strength_wall_seconds: float = 259_200.0
    device: str = "cpu"


class ExpandedRootPublicBeliefResolverBot(FrozenPublicBeliefResolverBot):
    """Deploy the exact four-continuation expanded selector validated by 086."""

    def __init__(
        self,
        config: ResolverGameplayConfig,
        engine_seed: int,
        actor_seat: int,
        *,
        scorer: Any | None = None,
        continuation_workers: int = 1,
    ) -> None:
        if continuation_workers != 1:
            raise ValueError("087 freezes serial continuation work within each game.")
        super().__init__(
            config,
            engine_seed,
            actor_seat,
            scorer=scorer,
            continuation_workers=continuation_workers,
        )
        self._expanded_expected_continuations = 0
        self._retained_roots = 0
        self._optional_roots = 0
        self._root_opportunities = 0
        self._minimum_roots: int | None = None
        self._maximum_roots = 0
        self._expansion_switches = 0
        self._optional_exclusions: Counter[str] = Counter()
        self._expanded_by_phase = {phase: Counter() for phase in PHASES}
        self._pending_expansion: dict[str, Any] | None = None

    @property
    def telemetry(self) -> dict[str, Any]:
        base = super().telemetry
        searches = int(base["completed_searches"])
        return {
            **base,
            "raw_parent_switches": int(base["switches"]),
            "expected_continuations": self._expanded_expected_continuations,
            "exact_branch_accounting": int(base["continuations"])
            == self._expanded_expected_continuations,
            "all_continuations_resolved": int(base["resolved_continuations"])
            == int(base["continuations"]),
            "retained_roots": self._retained_roots,
            "optional_roots_retained": self._optional_roots,
            "expanded_root_opportunities": self._root_opportunities,
            "minimum_retained_roots": self._minimum_roots or 0,
            "maximum_retained_roots": self._maximum_roots,
            "mean_retained_roots": self._retained_roots / searches if searches else 0.0,
            "expansion_switches": self._expansion_switches,
            "expansion_switch_rate": self._expansion_switches / searches if searches else 0.0,
            "optional_root_exclusions": dict(sorted(self._optional_exclusions.items())),
            "by_phase": {
                phase: {
                    **base["by_phase"][phase],
                    "retained_roots": self._expanded_by_phase[phase]["retained_roots"],
                    "optional_roots_retained": self._expanded_by_phase[phase][
                        "optional_roots_retained"
                    ],
                    "expanded_root_opportunities": self._expanded_by_phase[phase][
                        "expanded_root_opportunities"
                    ],
                    "expansion_switches": self._expanded_by_phase[phase][
                        "expansion_switches"
                    ],
                }
                for phase in PHASES
            },
        }

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
        del legal_actions, all_scores, parent, decision_index, resolution
        observation = self._pending_expansion
        self._pending_expansion = None
        if observation is None:
            raise RuntimeError("Expanded resolver lost its pending decision telemetry.")
        roots = int(observation["retained_roots"])
        optional = roots - self._config.root_actions
        phase = _phase(state.turn_index)
        self._expanded_expected_continuations += int(observation["expected_continuations"])
        self._retained_roots += roots
        self._optional_roots += optional
        self._minimum_roots = roots if self._minimum_roots is None else min(self._minimum_roots, roots)
        self._maximum_roots = max(self._maximum_roots, roots)
        self._optional_exclusions.update(observation["optional_root_exclusions"])
        self._expanded_by_phase[phase]["retained_roots"] += roots
        self._expanded_by_phase[phase]["optional_roots_retained"] += optional
        if optional > 0:
            self._root_opportunities += 1
            self._expanded_by_phase[phase]["expanded_root_opportunities"] += 1
        selected_key = canonical_successor_action_key(state, selected)
        if selected_key != observation["champion_selected_key"]:
            self._expansion_switches += 1
            self._expanded_by_phase[phase]["expansion_switches"] += 1

    def _resolve(
        self,
        state: SplendorState,
        legal_actions: list[Action],
        ranked: tuple[Any, ...],
        decision_index: int,
    ) -> tuple[Action, Counter[str], TwoPlyResolution]:
        self._pending_expansion = None
        structure = _prepare_expanded_structure(
            state,
            ranked,
            self._config,
            self._scorer,
            gameplay_belief_root_index(
                self._config,
                self._engine_seed,
                self._actor_seat,
                decision_index,
            ),
        )
        root_keys = structure["root_keys"]
        leaf_specs = [
            (det_index, root_key, after_root, information_set, response)
            for det_index, root_key, after_root, information_set, responses in structure[
                "branches"
            ]
            for response in responses
        ]
        env = SplendorEnv()
        leaves: list[ResolvedLeaf] = []
        counters: Counter[str] = Counter()
        for det_index, root_key, after_root, information_set, response in leaf_specs:
            values = []
            for continuation_index in range(self._config.discovery_continuations):
                seed = gameplay_continuation_seed(
                    self._config,
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
                    self._config.continuation_epsilon,
                    self._config.continuation_temperature,
                    self._config.continuation_top_k,
                    self._config.continuation_max_turns,
                    self._config.score_margin_scale,
                    self._config.max_loop_fallback_triggers,
                )
                if not math.isfinite(outcome.value):
                    raise RuntimeError("087 expanded resolver produced a nonfinite leaf value.")
                values.append(float(outcome.value))
                counters["continuations"] += 1
                counters["resolved_continuations"] += int(outcome.resolved)
                counters["hard_caps"] += int(outcome.hard_cap_reached)
                counters["loop_adjudications"] += int(outcome.loop_adjudicated)
                counters["rollout_fallback_triggers"] += int(outcome.fallback_triggers)
            value = statistics.fmean(values)
            leaves.append(
                ResolvedLeaf(
                    determinization=det_index,
                    root_action=root_key,
                    opponent_information_set=information_set,
                    opponent_action=canonical_successor_action_key(after_root, response.action),
                    discovery_payoff=value,
                    evaluation_payoff=value,
                )
            )

        champion_keys = root_keys[: self._config.root_actions]
        champion_leaves = [leaf for leaf in leaves if leaf.root_action in champion_keys]
        champion_resolution = resolve_two_ply(
            champion_leaves,
            champion_keys,
            self._config.determinizations,
            payoff="discovery_payoff",
        )
        expanded_resolution = resolve_two_ply(
            leaves,
            root_keys,
            self._config.determinizations,
            payoff="discovery_payoff",
        )
        expected = (
            len(root_keys)
            * self._config.determinizations
            * self._config.opponent_actions
            * self._config.discovery_continuations
        )
        counters["expected_continuations"] = expected
        if counters["continuations"] != expected:
            raise RuntimeError("087 expanded resolver branch accounting mismatch.")
        actual_map = actions_by_canonical_successor_key(state, legal_actions)
        selected_key = root_keys[expanded_resolution.best_index]
        if selected_key not in actual_map:
            raise RuntimeError("087 resolved semantic action is absent from the actual state.")
        self._pending_expansion = {
            "retained_roots": len(root_keys),
            "expected_continuations": expected,
            "champion_selected_key": champion_keys[champion_resolution.best_index],
            "optional_root_exclusions": structure["optional_root_exclusions"],
        }
        return actual_map[selected_key], counters, expanded_resolution


def _prepare_expanded_structure(
    state: SplendorState,
    ranked: Sequence[Any],
    cfg: ResolverGameplayConfig,
    scorer: Any,
    root_index: int,
) -> dict[str, Any]:
    if len(ranked) < cfg.root_actions:
        raise ResolverIneligible("fewer_than_four_root_actions")
    determinizations = deterministic_belief_range(
        state,
        state.current_player,
        root_index=root_index,
        count=cfg.determinizations,
    )
    retained_keys: list[tuple[object, ...]] = []
    retained_branches: list[tuple[Any, ...]] = []
    excluded: Counter[str] = Counter()
    for rank_index, row in enumerate(ranked):
        root_key = canonical_successor_action_key(state, row.action)
        if rank_index >= cfg.root_actions and row.action.action_type == ActionType.RESERVE_DECK:
            excluded["omitted_reserve_deck"] += 1
            continue
        try:
            branches = _prepare_root_branches(
                determinizations,
                root_key,
                cfg,
                scorer,
            )
        except ResolverIneligible as exc:
            if rank_index < cfg.root_actions:
                raise
            excluded[exc.reason] += 1
            continue
        retained_keys.append(root_key)
        retained_branches.extend(branches)
    if len(retained_keys) < cfg.root_actions:
        raise ResolverIneligible("incomplete_champion_roots")
    return {
        "root_keys": tuple(retained_keys),
        "champion_root_keys": tuple(retained_keys[: cfg.root_actions]),
        "branches": retained_branches,
        "optional_root_exclusions": dict(sorted(excluded.items())),
    }


def _prepare_root_branches(
    determinizations: Sequence[SplendorState],
    root_key: tuple[object, ...],
    cfg: ResolverGameplayConfig,
    scorer: Any,
) -> list[tuple[Any, ...]]:
    env = SplendorEnv()
    branches = []
    action_sets: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
    for det_index, determinized in enumerate(determinizations):
        root_map = actions_by_canonical_successor_key(
            determinized,
            env.legal_actions(determinized),
        )
        if root_key not in root_map:
            raise ResolverIneligible("root_mapping_changed")
        after_root = env.step(determinized, root_map[root_key])
        if after_root.terminal:
            raise ResolverIneligible("root_action_terminal")
        responses = _distinct_canonical_scores(
            after_root,
            scorer.score_actions(after_root, env.legal_actions(after_root)),
        )
        if len(responses) < cfg.opponent_actions:
            raise ResolverIneligible("fewer_than_four_opponent_actions")
        responses = responses[: cfg.opponent_actions]
        response_keys = tuple(
            canonical_successor_action_key(after_root, row.action) for row in responses
        )
        information_set = (
            root_key,
            canonical_public_belief_key(after_root, after_root.current_player),
        )
        previous = action_sets.setdefault(information_set, response_keys)
        if previous != response_keys:
            raise ResolverIneligible("opponent_action_set_unstable")
        branches.append((det_index, root_key, after_root, information_set, responses))
    return branches


def run_gate(config: ExpandedRootChampionChallengeConfig | None = None) -> dict[str, Any]:
    cfg = config or ExpandedRootChampionChallengeConfig()
    _validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    result_path = cfg.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing 087 result does not match the frozen protocol.")
        return result

    started = time.perf_counter()
    smoke = _run_smoke(cfg, identity)
    smoke_passed = bool(smoke["passed"])
    strength = _run_strength(cfg, identity) if smoke_passed else None
    strength_gates = _strength_gates(cfg, strength) if strength is not None else None
    operational_passed = bool(
        smoke_passed
        and strength_gates
        and all(strength_gates[name] for name in _operational_gate_names())
    )
    passed = bool(operational_passed and strength_gates and all(strength_gates.values()))
    valid = operational_passed
    decision = (
        "promote_expanded_root_public_belief_resolver_as_research_champion"
        if passed
        else "retain_050_051_research_champion_close_exact_expansion"
        if valid
        else "expanded_root_gameplay_087_invalid_no_strength_decision"
    )
    result = {
        "experiment": "087_expanded_root_resolver_champion_challenge",
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "protocol_sha256": PROTOCOL_SHA256,
        "identity": identity,
        "smoke": smoke,
        "smoke_passed": smoke_passed,
        "strength": strength,
        "strength_gates": strength_gates,
        "valid": valid,
        "passed": passed,
        "promotion_authorized": passed,
        "research_champion": (
            "expanded_root_public_belief_resolver_087"
            if passed
            else "public_belief_resolver_050_051"
        ),
        "fast_champion": "035",
        "training_authorized": False,
        "corpus_generation_authorized": False,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write(result_path, result)
    print(f"087 result: {'PASS' if passed else 'FAIL'} ({decision})", flush=True)
    return result


def _resolver_config(cfg: ExpandedRootChampionChallengeConfig) -> ResolverGameplayConfig:
    return ResolverGameplayConfig(
        output_dir=cfg.output_dir,
        checkpoint=cfg.checkpoint,
        smoke_seed=cfg.smoke_seed,
        direct_seed_start=cfg.strength_seed_start,
        direct_pairs=cfg.pairs,
        determinizations=cfg.determinizations,
        root_actions=cfg.champion_roots,
        opponent_actions=cfg.opponent_actions,
        discovery_continuations=cfg.discovery_continuations,
        continuation_epsilon=cfg.continuation_epsilon,
        continuation_temperature=cfg.continuation_temperature,
        continuation_top_k=cfg.continuation_top_k,
        continuation_max_turns=cfg.continuation_max_turns,
        max_loop_fallback_triggers=cfg.max_loop_fallback_triggers,
        score_margin_scale=cfg.score_margin_scale,
        belief_root_index_start=cfg.belief_root_index_start,
        continuation_seed_start=cfg.continuation_seed_start,
        workers=cfg.workers,
        max_turns=cfg.max_turns,
        repetition_limit=cfg.repetition_limit,
        no_progress_limit=cfg.no_progress_limit,
        bootstrap_samples=cfg.bootstrap_samples,
        minimum_score_share=cfg.minimum_score_share,
        minimum_seat_score_share=cfg.minimum_seat_score_share,
        maximum_loop_adjudicated_fraction=cfg.maximum_loop_adjudicated_fraction,
        device=cfg.device,
    )


def _play_pair(cfg: ExpandedRootChampionChallengeConfig, seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    resolver_cfg = _resolver_config(cfg)
    scorer = _LegalActionScorer(cfg.checkpoint, cfg.device, "legal_action_attention_scorer")
    games = []
    candidate_telemetry = []
    champion_telemetry = []
    for candidate_seat in (0, 1):
        champion_seat = 1 - candidate_seat
        candidate = ExpandedRootPublicBeliefResolverBot(
            resolver_cfg,
            seed,
            candidate_seat,
            scorer=scorer,
        )
        champion = FrozenPublicBeliefResolverBot(
            resolver_cfg,
            seed,
            champion_seat,
            scorer=scorer,
        )
        bots = (candidate, champion) if candidate_seat == 0 else (champion, candidate)
        game = play_game(
            bots[0],
            bots[1],
            seed=seed,
            max_turns=cfg.max_turns,
            repetition_limit=cfg.repetition_limit,
            no_progress_limit=cfg.no_progress_limit,
        )
        games.append(asdict(game))
        candidate_telemetry.append(candidate.telemetry)
        champion_telemetry.append(champion.telemetry)
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": games,
        "candidate_telemetry": candidate_telemetry,
        "champion_telemetry": champion_telemetry,
    }


def _run_smoke(
    cfg: ExpandedRootChampionChallengeConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    path = cfg.output_dir / "smoke.json"
    smoke_identity = {**identity, "block": "smoke", "seed": cfg.smoke_seed}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != smoke_identity:
            raise ValueError("Existing 087 smoke does not match the frozen protocol.")
        return payload
    print("[resolver087:smoke] running original and from-zero repeat", flush=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_play_pair, cfg, cfg.smoke_seed) for _ in range(2)]
        original, repeated = [future.result() for future in futures]
    rows = [original, repeated]
    integrity = _game_integrity(rows)
    candidate = _summarize_candidate_telemetry(rows)
    champion = _summarize_champion_telemetry(rows)
    repeat_identical = _deterministic_projection(original) == _deterministic_projection(repeated)
    gates = {
        "frozen_provenance_and_configuration": _provenance_matches(cfg),
        "four_clean_games": integrity["games"] == 4 and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "both_resolvers_exercised": candidate["completed_searches"] > 0
        and champion["completed_searches"] > 0,
        "candidate_exact_branch_accounting": bool(candidate["exact_branch_accounting"]),
        "champion_exact_branch_accounting": bool(champion["exact_branch_accounting"]),
        "all_continuations_resolved": bool(candidate["all_continuations_resolved"])
        and bool(champion["all_continuations_resolved"]),
        "zero_hard_caps": candidate["hard_caps"] == 0 and champion["hard_caps"] == 0,
        "zero_rollout_fallback_triggers": candidate["rollout_fallback_triggers"] == 0
        and champion["rollout_fallback_triggers"] == 0,
        "bounded_loop_adjudication": _bounded_loops(cfg, candidate, champion),
        "from_zero_repeat_identical": repeat_identical,
    }
    payload = {
        "format": "expanded_root_resolver_champion_smoke_087_v1",
        "identity": smoke_identity,
        "integrity": integrity,
        "candidate_telemetry": candidate,
        "champion_telemetry": champion,
        "repeat_identical": repeat_identical,
        "gates": gates,
        "passed": all(gates.values()),
        "original": original,
        "repeated": repeated,
    }
    _atomic_write(path, payload)
    return payload


def _run_strength(
    cfg: ExpandedRootChampionChallengeConfig,
    identity: dict[str, Any],
) -> dict[str, Any]:
    path = cfg.output_dir / "strength_vs_research_champion.json"
    progress_path = cfg.output_dir / "strength_vs_research_champion.progress.json"
    block_identity = {
        **identity,
        "block": "strength_vs_research_champion",
        "seed_start": cfg.strength_seed_start,
        "pairs": cfg.pairs,
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity") != block_identity:
            raise ValueError("Existing 087 strength result does not match the frozen protocol.")
        return payload

    completed: dict[int, dict[str, Any]] = {}
    prior_active_seconds = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != block_identity:
            raise ValueError("Existing 087 progress does not match the frozen protocol.")
        completed = {int(row["seed"]): row for row in progress.get("pairs", [])}
        prior_active_seconds = float(progress.get("active_wall_seconds", 0.0))

    seeds = list(range(cfg.strength_seed_start, cfg.strength_seed_start + cfg.pairs))
    pending = [seed for seed in seeds if seed not in completed]
    print(
        f"[resolver087:strength] pairs={cfg.pairs} complete={len(completed)} "
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
                active_seconds = prior_active_seconds + time.perf_counter() - started
                ordered = [completed[value] for value in sorted(completed)]
                _atomic_write(
                    progress_path,
                    {
                        "identity": block_identity,
                        "completed": False,
                        "active_wall_seconds": active_seconds,
                        "pairs": ordered,
                    },
                )
                rate = index / (time.perf_counter() - started)
                eta = (len(pending) - index) / rate if rate > 0.0 else math.inf
                print(
                    f"[resolver087:strength] complete={len(completed)}/{cfg.pairs} "
                    f"eta_seconds={eta:.0f}",
                    flush=True,
                )

    active_seconds = prior_active_seconds + time.perf_counter() - started
    rows = [completed[seed] for seed in seeds]
    summary = summarize_pair_rows(rows, cfg.bootstrap_samples, cfg.bootstrap_seed)
    summary["seat_breakdown"] = _seat_breakdown(rows)
    payload = {
        "format": "expanded_root_resolver_champion_strength_087_v1",
        "identity": block_identity,
        "active_wall_seconds": active_seconds,
        "summary": summary,
        "integrity": _game_integrity(rows),
        "candidate_telemetry": _summarize_candidate_telemetry(rows),
        "champion_telemetry": _summarize_champion_telemetry(rows),
        "pairs": rows,
    }
    _atomic_write(path, payload)
    _atomic_write(
        progress_path,
        {
            "identity": block_identity,
            "completed": True,
            "active_wall_seconds": active_seconds,
            "pairs": rows,
        },
    )
    return payload


def _strength_gates(
    cfg: ExpandedRootChampionChallengeConfig,
    strength: dict[str, Any],
) -> dict[str, bool]:
    summary = strength["summary"]
    integrity = strength["integrity"]
    candidate = strength["candidate_telemetry"]
    champion = strength["champion_telemetry"]
    candidate_phases = candidate["by_phase"]
    return {
        "frozen_provenance_and_configuration": _provenance_matches(cfg),
        "exactly_400_clean_games": integrity["games"] == cfg.pairs * 2
        and integrity["invalid_games"] == 0,
        "zero_timeouts_and_stalls": integrity["timed_out_games"] == 0
        and integrity["stalled_games"] == 0,
        "candidate_exact_variable_branch_accounting": bool(
            candidate["exact_branch_accounting"]
        ),
        "champion_exact_192_branch_accounting": bool(champion["exact_branch_accounting"]),
        "all_continuations_resolved": bool(candidate["all_continuations_resolved"])
        and bool(champion["all_continuations_resolved"]),
        "zero_hard_caps": candidate["hard_caps"] == 0 and champion["hard_caps"] == 0,
        "zero_rollout_fallback_triggers": candidate["rollout_fallback_triggers"] == 0
        and champion["rollout_fallback_triggers"] == 0,
        "bounded_loop_adjudication": _bounded_loops(cfg, candidate, champion),
        "within_72_active_wall_hours": float(strength["active_wall_seconds"])
        <= cfg.maximum_strength_wall_seconds,
        "expanded_roots_and_switches_every_phase": all(
            int(candidate_phases[phase].get("optional_roots_retained", 0)) > 0
            and int(candidate_phases[phase].get("expansion_switches", 0)) > 0
            for phase in PHASES
        ),
        "expansion_switch_rate_at_least_15pct": float(candidate["expansion_switch_rate"])
        >= cfg.minimum_expansion_switch_rate,
        "score_share_at_least_55pct": float(summary["score_share"]["mean"])
        >= cfg.minimum_score_share,
        "score_share_ci95_lower_above_50pct": float(summary["score_share"]["ci95_low"])
        > 0.50,
        "positive_mean_paired_margin": float(summary["final_score_margin"]["mean"]) > 0.0,
        "no_seat_collapse": all(
            float(item["score_share"]) >= cfg.minimum_seat_score_share
            for item in summary["seat_breakdown"].values()
        ),
    }


def _operational_gate_names() -> tuple[str, ...]:
    return (
        "frozen_provenance_and_configuration",
        "exactly_400_clean_games",
        "zero_timeouts_and_stalls",
        "candidate_exact_variable_branch_accounting",
        "champion_exact_192_branch_accounting",
        "all_continuations_resolved",
        "zero_hard_caps",
        "zero_rollout_fallback_triggers",
        "bounded_loop_adjudication",
        "within_72_active_wall_hours",
    )


def _summarize_candidate_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = [item for row in rows for item in row["candidate_telemetry"]]
    totals = Counter()
    fallbacks: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    phase_totals = {phase: Counter() for phase in PHASES}
    minima = []
    maxima = []
    for item in items:
        for key in (
            "decisions",
            "completed_searches",
            "raw_parent_switches",
            "fallbacks",
            "continuations",
            "expected_continuations",
            "resolved_continuations",
            "hard_caps",
            "loop_adjudications",
            "rollout_fallback_triggers",
            "retained_roots",
            "optional_roots_retained",
            "expanded_root_opportunities",
            "expansion_switches",
        ):
            totals[key] += int(item[key])
        fallbacks.update(item["fallback_reasons"])
        exclusions.update(item["optional_root_exclusions"])
        if int(item["completed_searches"]) > 0:
            minima.append(int(item["minimum_retained_roots"]))
            maxima.append(int(item["maximum_retained_roots"]))
        for phase in PHASES:
            phase_totals[phase].update(item["by_phase"][phase])
    continuations = totals["continuations"]
    searches = totals["completed_searches"]
    return {
        **dict(totals),
        "fallback_reasons": dict(sorted(fallbacks.items())),
        "optional_root_exclusions": dict(sorted(exclusions.items())),
        "minimum_retained_roots": min(minima) if minima else 0,
        "maximum_retained_roots": max(maxima) if maxima else 0,
        "mean_retained_roots": totals["retained_roots"] / searches if searches else 0.0,
        "expansion_switch_rate": totals["expansion_switches"] / searches if searches else 0.0,
        "exact_branch_accounting": continuations == totals["expected_continuations"],
        "all_continuations_resolved": totals["resolved_continuations"] == continuations,
        "loop_adjudication_fraction": totals["loop_adjudications"] / continuations
        if continuations
        else 1.0,
        "by_phase": {phase: dict(counts) for phase, counts in phase_totals.items()},
    }


def _summarize_champion_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = [item for row in rows for item in row["champion_telemetry"]]
    totals = Counter()
    fallbacks: Counter[str] = Counter()
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
        fallbacks.update(item["fallback_reasons"])
        for phase in PHASES:
            phase_totals[phase].update(item["by_phase"][phase])
    continuations = totals["continuations"]
    expected = totals["completed_searches"] * EXPECTED_BRANCHES_PER_SEARCH
    return {
        **dict(totals),
        "expected_continuations": expected,
        "fallback_reasons": dict(sorted(fallbacks.items())),
        "exact_branch_accounting": continuations == expected,
        "all_continuations_resolved": totals["resolved_continuations"] == continuations,
        "loop_adjudication_fraction": totals["loop_adjudications"] / continuations
        if continuations
        else 1.0,
        "by_phase": {phase: dict(counts) for phase, counts in phase_totals.items()},
    }


def _bounded_loops(
    cfg: ExpandedRootChampionChallengeConfig,
    candidate: dict[str, Any],
    champion: dict[str, Any],
) -> bool:
    total_continuations = int(candidate["continuations"]) + int(champion["continuations"])
    total_loops = int(candidate["loop_adjudications"]) + int(champion["loop_adjudications"])
    combined = total_loops / total_continuations if total_continuations else 1.0
    return (
        float(candidate["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction
        and float(champion["loop_adjudication_fraction"])
        <= cfg.maximum_loop_adjudicated_fraction
        and combined <= cfg.maximum_loop_adjudicated_fraction
    )


def _game_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _seat_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[int, list[float]] = {0: [], 1: []}
    for row in rows:
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
            winner = game["winner"]
            points = 0.5 if winner is None else 1.0 if int(winner) == int(seat) else 0.0
            buckets[int(seat)].append(points)
    return {
        f"seat{seat}": {"games": len(points), "score_share": statistics.fmean(points)}
        for seat, points in buckets.items()
    }


def _deterministic_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": row["seed"],
        "candidate_seats": row["candidate_seats"],
        "games": row["games"],
        "candidate_telemetry": row["candidate_telemetry"],
        "champion_telemetry": row["champion_telemetry"],
    }


def _candidate_contract_matches(cfg: ExpandedRootChampionChallengeConfig) -> bool:
    if not cfg.result_086.exists():
        return False
    result = json.loads(cfg.result_086.read_text(encoding="utf-8"))
    frozen = result.get("identity", {}).get("configuration", {})
    return (
        result.get("passed") is True
        and result.get("gameplay_preregistration_authorized") is True
        and result.get("identity", {}).get("root_contract") == ROOT_CONTRACT
        and int(frozen.get("champion_roots", -1)) == cfg.champion_roots
        and int(frozen.get("determinizations", -1)) == cfg.determinizations
        and int(frozen.get("opponent_actions", -1)) == cfg.opponent_actions
        and int(frozen.get("discovery_continuations", -1)) == cfg.discovery_continuations
        and float(frozen.get("continuation_epsilon", -1.0)) == cfg.continuation_epsilon
        and float(frozen.get("continuation_temperature", -1.0))
        == cfg.continuation_temperature
        and int(frozen.get("continuation_top_k", -1)) == cfg.continuation_top_k
        and int(frozen.get("continuation_max_turns", -1)) == cfg.continuation_max_turns
        and int(frozen.get("max_loop_fallback_triggers", -1))
        == cfg.max_loop_fallback_triggers
        and float(frozen.get("score_margin_scale", -1.0)) == cfg.score_margin_scale
    )


def _provenance_matches(cfg: ExpandedRootChampionChallengeConfig) -> bool:
    if cfg != ExpandedRootChampionChallengeConfig():
        return False
    if not all(path.exists() for path in (PROTOCOL, cfg.checkpoint, cfg.result_084, cfg.result_086)):
        return False
    result_084 = json.loads(cfg.result_084.read_text(encoding="utf-8"))
    resolver_cfg = _resolver_config(cfg)
    return (
        _sha256_file(PROTOCOL) == PROTOCOL_SHA256
        and _sha256_file(cfg.checkpoint) == CHECKPOINT_SHA256
        and _sha256_file(cfg.result_084) == RESULT_084_SHA256
        and _sha256_file(cfg.result_086) == RESULT_086_SHA256
        and result_084.get("passed") is True
        and result_084.get("research_champion") == "public_belief_resolver_050_051"
        and _frozen_contract_matches_049(resolver_cfg)
        and _candidate_contract_matches(cfg)
    )


def _identity(cfg: ExpandedRootChampionChallengeConfig) -> dict[str, Any]:
    resolver_cfg = _resolver_config(cfg)
    return {
        "configuration": _config_payload(cfg),
        "resolver_configuration": _config_payload(resolver_cfg),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "checkpoint_sha256": _sha256_file(cfg.checkpoint),
        "result_084_sha256": _sha256_file(cfg.result_084),
        "result_086_sha256": _sha256_file(cfg.result_086),
        "candidate_root_contract": ROOT_CONTRACT,
        "candidate_continuation_contract": "3x4x4_discovery_only_variable_roots_v1",
        "champion_continuation_contract": "3x4x4_discovery_only_top4_v1",
        "champion_expected_branches_per_search": EXPECTED_BRANCHES_PER_SEARCH,
        "matched_candidate_champion_namespaces": True,
        "valid_terminations": list(VALID_TERMINATIONS),
    }


def _validate_config(cfg: ExpandedRootChampionChallengeConfig) -> None:
    if cfg != ExpandedRootChampionChallengeConfig():
        raise ValueError("Experiment 087 permits only its preregistered default configuration.")
    for path in (PROTOCOL, cfg.checkpoint, cfg.result_084, cfg.result_086):
        if not path.exists():
            raise FileNotFoundError(path)
    if cfg.bootstrap_seed != cfg.strength_seed_start:
        raise ValueError("087 bootstrap seed must equal the frozen strength start.")
    if cfg.strength_seed_start != cfg.smoke_seed + 100 or cfg.pairs != 200:
        raise ValueError("087 fresh paired seed block differs from the preregistration.")
    if cfg.continuation_seed_start + 300 * 1_000_000_000 >= 2**63:
        raise ValueError("087 continuation namespace exceeds signed 64-bit range.")
    if not _provenance_matches(cfg):
        raise ValueError("Experiment 087 frozen provenance or resolver contract mismatch.")


def _config_payload(cfg: Any) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    run_gate()


if __name__ == "__main__":
    main()
