"""Run preregistered experiment 049b: eligibility-safe resolver replication."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_bounded_public_belief_resolving import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    PHASES,
    ResolvingConfig,
    _atomic_write,
    _evaluate_roots_parallel,
    _phase,
    _privacy_audit,
    _select_roots,
    _sha256_file,
    _summarize,
)
from splendor_ai.eval.run_selective_rollout_gate import distinct_canonical_scores
from splendor_ai.training.replay import serialize_state_snapshot
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


SOURCE_BANDS: dict[tuple[int, str], tuple[int, int]] = {
    (0, "early"): (491_000, 491_999),
    (0, "mid"): (492_000, 492_999),
    (0, "late"): (493_000, 493_999),
    (1, "early"): (494_000, 494_999),
    (1, "mid"): (495_000, 495_999),
    (1, "late"): (496_000, 496_999),
}

GATE_THRESHOLDS = {
    "max_loop_adjudication_fraction": 0.001,
    "min_leaf_pearson": 0.60,
    "max_leaf_mae": 0.10,
    "min_leaf_sign_agreement": 0.70,
    "min_action_value_pearson": 0.60,
    "min_best_action_agreement": 0.60,
    "min_effect_sign_eligible": 24,
    "min_effect_sign_agreement": 0.70,
    "min_changed_roots": 8,
    "min_resolved_fraction": 0.25,
    "min_stratum_changed": 1,
    "min_stratum_mean_gain": -0.025,
}


@dataclass(frozen=True, slots=True)
class ReplicationConfig:
    output_dir: Path = Path("outputs/bounded_public_belief_resolving_049b")
    checkpoint: Path = CHECKPOINT
    admitted_per_stratum: int = 32
    selected_per_stratum: int = 8
    determinizations: int = 3
    root_actions: int = 4
    opponent_actions: int = 4
    continuations_per_leaf: int = 8
    continuation_seed_start: int = 52_000_000
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    workers: int = 10
    bootstrap_samples: int = 20_000
    device: str = "cpu"


def run_experiment(config: ReplicationConfig | None = None) -> dict[str, Any]:
    cfg = config or ReplicationConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256_file(cfg.checkpoint)
    checkpoint_payload = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    print("049b collecting six eligibility-safe source bands", flush=True)
    source = _collect_source_games(cfg)
    scorer = _LegalActionScorer(cfg.checkpoint, cfg.device, "legal_action_attention_scorer")
    selection = _select_roots(cfg, source, scorer)
    _atomic_write(cfg.output_dir / "selection.json", selection)
    print("049b evaluating 48 frozen bounded public-belief subgames", flush=True)
    roots = selection["selected"]
    rows = _evaluate_roots_parallel(cfg, roots)
    print("049b repeating first eight roots from zero", flush=True)
    repeated = _evaluate_roots_parallel(cfg, roots[:8], progress_name=None)
    repeat_identical = rows[:8] == repeated

    metrics = _summarize(rows, cfg)
    privacy = _privacy_audit(roots[0], scorer) if roots else {"passed": False}
    source_gate = _source_collection_gate(cfg, source, selection)
    structural_gate = (
        selection["structural_failures"] >= 0
        and all(row["root_action_mapping_stable"] for row in rows)
        and all(row["opponent_information_sets_stable"] for row in rows)
        and all(row["illegal_actions"] == 0 for row in rows)
        and privacy["passed"]
    )
    continuation_gate = (
        metrics["continuations"] == 18_432
        and metrics["resolved_continuations"] == 18_432
        and metrics["hard_caps"] == 0
        and metrics["loop_adjudications"] / metrics["continuations"]
        <= GATE_THRESHOLDS["max_loop_adjudication_fraction"]
    )
    leaf_gate = (
        metrics["leaf_payoff_pearson"] >= GATE_THRESHOLDS["min_leaf_pearson"]
        and metrics["leaf_payoff_mae"] <= GATE_THRESHOLDS["max_leaf_mae"]
        and metrics["leaf_sign_eligible"] > 0
        and metrics["leaf_sign_agreement"] >= GATE_THRESHOLDS["min_leaf_sign_agreement"]
    )
    target_gate = (
        metrics["root_action_value_pearson"] >= GATE_THRESHOLDS["min_action_value_pearson"]
        and metrics["best_action_agreement"] >= GATE_THRESHOLDS["min_best_action_agreement"]
        and metrics["effect_sign_eligible"] >= GATE_THRESHOLDS["min_effect_sign_eligible"]
        and metrics["effect_sign_agreement"] >= GATE_THRESHOLDS["min_effect_sign_agreement"]
        and metrics["changed_roots"] >= GATE_THRESHOLDS["min_changed_roots"]
        and metrics["resolved_fraction"] >= GATE_THRESHOLDS["min_resolved_fraction"]
        and metrics["evaluation_gain_ci95"][0] > 0.0
        and all(
            item["changed"] >= GATE_THRESHOLDS["min_stratum_changed"]
            and item["mean_gain"] >= GATE_THRESHOLDS["min_stratum_mean_gain"]
            for item in metrics["seat_breakdown"].values()
        )
        and all(
            item["changed"] >= GATE_THRESHOLDS["min_stratum_changed"]
            and item["mean_gain"] >= GATE_THRESHOLDS["min_stratum_mean_gain"]
            for item in metrics["phase_breakdown"].values()
        )
    )
    gates = {
        "frozen_checkpoint_and_configuration": (
            checkpoint_hash == CHECKPOINT_SHA256
            and checkpoint_payload.get("model_kind") == "legal_action_attention_scorer"
            and cfg == ReplicationConfig()
            and _frozen_resolver_contract_matches_049(cfg)
        ),
        "eligibility_safe_source_and_selection": source_gate,
        "belief_privacy_and_action_structure": structural_gate,
        "continuation_integrity": continuation_gate,
        "leaf_payoff_consistency": leaf_gate,
        "resolved_target_quality": target_gate,
        "first_eight_repeat_deterministic": repeat_identical,
    }
    passed = all(gates.values())
    result = {
        "experiment": "049b_eligibility_safe_bounded_public_belief_resolving_replication",
        "protocol": "docs/experiments/bounded_public_belief_resolving_replication_049b.md",
        "configuration": _jsonable(asdict(cfg)),
        "source_bands": {f"seat{seat}_{phase}": [start, end] for (seat, phase), (start, end) in SOURCE_BANDS.items()},
        "gate_thresholds": GATE_THRESHOLDS,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_model_kind": checkpoint_payload.get("model_kind"),
        "source": {key: value for key, value in source.items() if key != "candidates"},
        "selection": {key: value for key, value in selection.items() if key != "selected"},
        "privacy": privacy,
        "metrics": metrics,
        "rows": rows,
        "first_eight_repeat_identical": repeat_identical,
        "gates": gates,
        "passed": passed,
        "gameplay_gate_authorized": passed,
        "neural_fit_authorized": False,
        "gpu_rental_authorized": False,
        "decision": (
            "authorize_separately_preregistered_paired_resolver_vs_035_gameplay_gate"
            if passed
            else "do_not_run_gameplay_or_train"
        ),
    }
    _atomic_write(cfg.output_dir / "result.json", result)
    print(f"049b result: {'PASS' if passed else 'FAIL'}", flush=True)
    return result


def _collect_source_games(cfg: ReplicationConfig) -> dict[str, Any]:
    cache = cfg.output_dir / "source_candidates.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    band_summaries: dict[str, dict[str, Any]] = {}
    game_index = 0
    for seat in (0, 1):
        for phase in PHASES:
            key = f"seat{seat}_{phase}"
            start, end = SOURCE_BANDS[(seat, phase)]
            admitted = 0
            attempted = 0
            invalid = 0
            missing_candidate = 0
            for seed in range(start, end + 1):
                if admitted >= cfg.admitted_per_stratum:
                    break
                attempted += 1
                if attempted == 1 or attempted % 8 == 0:
                    print(
                        f"049b {key}: admitted {admitted}/{cfg.admitted_per_stratum}, "
                        f"attempt {attempted}",
                        flush=True,
                    )
                summary, candidate = _play_source_game(cfg, game_index, seed, seat, phase)
                summaries.append(summary)
                game_index += 1
                if not summary["valid"]:
                    invalid += 1
                    continue
                if candidate is None:
                    missing_candidate += 1
                    continue
                candidate["admission_index"] = admitted
                candidates.append(candidate)
                admitted += 1
            band_summaries[key] = {
                "seed_start": start,
                "seed_end": end,
                "attempted_games": attempted,
                "admitted_games": admitted,
                "invalid_games_skipped": invalid,
                "valid_games_without_candidate_skipped": missing_candidate,
                "complete": admitted == cfg.admitted_per_stratum,
            }
    admitted_counts = Counter(row["stratum"] for row in candidates)
    payload = {
        "games": summaries,
        "band_summaries": band_summaries,
        "attempted_games": len(summaries),
        "valid_attempted_games": sum(row["valid"] for row in summaries),
        "invalid_attempted_games": sum(not row["valid"] for row in summaries),
        "admitted_games": len(candidates),
        "invalid_admitted_games": 0,
        "admitted_stratum_counts": dict(admitted_counts),
        "collection_complete": all(row["complete"] for row in band_summaries.values()),
        "candidates": candidates,
    }
    _atomic_write(cache, payload)
    return payload


def _play_source_game(
    cfg: ReplicationConfig,
    game_index: int,
    seed: int,
    seat: int,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    bots = (
        LegalActionAttentionScorerBot(cfg.checkpoint, cfg.device),
        LegalActionAttentionScorerBot(cfg.checkpoint, cfg.device),
    )
    selector = _LegalActionScorer(cfg.checkpoint, cfg.device, "legal_action_attention_scorer")
    visits: dict[tuple[object, ...], int] = {}
    no_progress = 0
    forced_passes = 0
    best_candidate: dict[str, Any] | None = None
    termination = "completed"
    while not state.terminal:
        signature = state_signature(state)
        visits[signature] = visits.get(signature, 0) + 1
        if visits[signature] >= 4:
            termination = "repetition_cutoff"
            break
        if state.turn_index >= 400:
            termination = "turn_limit"
            break
        legal = env.legal_actions(state)
        if not legal:
            termination = "stalled"
            break
        if state.current_player == seat and _phase(state.turn_index) == phase:
            ranked = distinct_canonical_scores(state, selector.score_actions(state, legal))
            if len(ranked) >= cfg.root_actions:
                gap = ranked[0].score - ranked[1].score
                if best_candidate is None or (gap, state.turn_index) < (
                    best_candidate["logit_gap"],
                    best_candidate["turn"],
                ):
                    best_candidate = {
                        "game_index": game_index,
                        "seed": seed,
                        "actor_seat": seat,
                        "phase": phase,
                        "stratum": f"seat{seat}_{phase}",
                        "turn": state.turn_index,
                        "logit_gap": gap,
                        "state_snapshot": serialize_state_snapshot(state),
                    }
        actor = bots[state.current_player]
        action = actor.choose_action(env, state, legal)
        if action is None:
            raise RuntimeError("035 returned no legal action.")
        next_state = env.step(state, action)
        forced = len(legal) == 1 and legal[0].action_type == ActionType.PASS
        forced_passes = forced_passes + 1 if forced else 0
        no_progress = 0 if is_progress_transition(state, next_state, action) else no_progress + 1
        state = next_state
        if forced_passes >= 2:
            termination = "forced_pass_deadlock"
            break
        if no_progress >= 60:
            termination = "no_progress_cutoff"
            break
    valid = termination in ("completed", "forced_pass_deadlock")
    summary = {
        "game_index": game_index,
        "seed": seed,
        "assigned_seat": seat,
        "assigned_phase": phase,
        "stratum": f"seat{seat}_{phase}",
        "turns": state.turn_index,
        "termination_reason": termination,
        "valid": valid,
        "candidate_found": best_candidate is not None,
        "admitted": valid and best_candidate is not None,
    }
    return summary, best_candidate if valid else None


def _source_collection_gate(
    cfg: ReplicationConfig,
    source: dict[str, Any],
    selection: dict[str, Any],
) -> bool:
    expected_admitted = cfg.admitted_per_stratum * len(SOURCE_BANDS)
    expected_selected = cfg.selected_per_stratum * len(SOURCE_BANDS)
    return bool(
        source["collection_complete"]
        and source["admitted_games"] == expected_admitted
        and source["invalid_admitted_games"] == 0
        and set(source["admitted_stratum_counts"]) == {
            f"seat{seat}_{phase}" for seat in (0, 1) for phase in PHASES
        }
        and all(
            value == cfg.admitted_per_stratum
            for value in source["admitted_stratum_counts"].values()
        )
        and selection["selected_count"] == expected_selected
        and selection["unique_source_games"] == expected_selected
        and len(selection["stratum_counts"]) == len(SOURCE_BANDS)
        and all(value == cfg.selected_per_stratum for value in selection["stratum_counts"].values())
    )


def _frozen_resolver_contract_matches_049(cfg: ReplicationConfig) -> bool:
    original = ResolvingConfig()
    frozen_fields = (
        "checkpoint",
        "selected_per_stratum",
        "determinizations",
        "root_actions",
        "opponent_actions",
        "continuations_per_leaf",
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
    return all(getattr(cfg, field) == getattr(original, field) for field in frozen_fields)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=ReplicationConfig().workers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != ReplicationConfig().workers:
        raise ValueError("049b preregistration freezes --workers at 10.")
    run_experiment()


if __name__ == "__main__":
    main()
