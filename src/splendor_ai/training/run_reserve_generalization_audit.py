"""Held-out reserve-choice audit on non-human policy035 self-play states."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from statistics import NormalDist
from typing import Any

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_human_game_forensic_audit import (
    HumanGameForensicAuditConfig,
    _action_map,
    _evaluate_suspicious_state,
    _result_key,
)
from splendor_ai.training.run_policy_drift import _ScorerConfig
from splendor_ai.training.run_policy_search_disagreement import _load_policy_scorer


TURN_BANDS = ((0, 4), (5, 9), (10, 14), (15, 19))
RESERVE_TYPES = {ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK}


@dataclass(frozen=True, slots=True)
class ReserveGeneralizationAuditConfig:
    replay_path: Path
    policy_checkpoint: Path
    output_path: Path
    report_path: Path
    progress_path: Path | None = None
    policy_marker: str = "legal_action_scorer_035_legal_observation"
    samples_per_band: int = 15
    selection_pool_multiplier: int = 10
    max_candidates_per_state: int = 8
    policy_top_k: int = 4
    rollout_determinizations: int = 3
    rollouts_per_determinization: int = 8
    rollout_epsilon: float = 0.1
    rollout_temperature: float = 0.5
    rollout_top_k: int = 3
    rollout_max_additional_turns: int = 120
    outcome_weight: float = 0.75
    score_margin_scale: float = 5.0
    min_resolved_delta: float = 0.10
    confidence_z: float = 1.96
    min_generalization_count: int = 6
    min_generalization_rate_lower_bound: float = 0.03
    min_generalization_bands: int = 2
    min_generalization_mean_delta: float = 0.20
    seed: int = 113
    device: str = "cpu"
    log_every_states: int = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit deterministic policy035 reserve choices on disjoint non-human "
            "early-game states."
        )
    )
    parser.add_argument("--replay-path", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument(
        "--policy-marker", default="legal_action_scorer_035_legal_observation"
    )
    parser.add_argument("--samples-per-band", type=int, default=15)
    parser.add_argument("--selection-pool-multiplier", type=int, default=10)
    parser.add_argument("--max-candidates-per-state", type=int, default=8)
    parser.add_argument("--policy-top-k", type=int, default=4)
    parser.add_argument("--rollout-determinizations", type=int, default=3)
    parser.add_argument("--rollouts-per-determinization", type=int, default=8)
    parser.add_argument("--rollout-epsilon", type=float, default=0.1)
    parser.add_argument("--rollout-temperature", type=float, default=0.5)
    parser.add_argument("--rollout-top-k", type=int, default=3)
    parser.add_argument("--rollout-max-additional-turns", type=int, default=120)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--min-resolved-delta", type=float, default=0.10)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--min-generalization-count", type=int, default=6)
    parser.add_argument(
        "--min-generalization-rate-lower-bound", type=float, default=0.03
    )
    parser.add_argument("--min-generalization-bands", type=int, default=2)
    parser.add_argument("--min-generalization-mean-delta", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-states", type=int, default=5)
    return parser


def run_reserve_generalization_audit(
    config: ReserveGeneralizationAuditConfig,
) -> dict[str, Any]:
    _validate(config)
    scorer = _load_policy_scorer(
        _ScorerConfig(
            checkpoint=config.policy_checkpoint,
            model_kind="legal_action_attention_scorer",
            device=config.device,
        )
    )
    env = SplendorEnv()
    codec = ActionCodec()
    progress_path = config.progress_path or config.output_path.with_suffix(".progress.json")
    cached_progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {}
    )
    cached_selected = cached_progress.get("selected_payloads")
    if cached_selected:
        selected = cached_selected
        selection = cached_progress["selection"]
    else:
        selected, selection = _select_policy_reserve_states(
            config=config,
            scorer=scorer,
            env=env,
            codec=codec,
        )
    selected_keys = [list(_result_key(row)) for row in selected]
    identity = _progress_identity(config, selected_keys)
    existing_results: dict[tuple[int, int, int], dict[str, Any]] = {}
    if cached_progress:
        if cached_progress.get("identity") != identity:
            raise ValueError(
                "Existing progress does not match this corpus, selection, or rollout config."
            )
        existing_results = {
            _result_key(row): row for row in cached_progress.get("rollout_results", [])
        }
    else:
        _write_progress(
            progress_path,
            identity=identity,
            selection=selection,
            selected_keys=selected_keys,
            selected_payloads=selected,
            rollout_results=[],
        )

    rollout_config = _rollout_config(config, progress_path)
    rollout_results: list[dict[str, Any]] = []
    for state_position, payload in enumerate(selected):
        key = _result_key(payload)
        if key in existing_results:
            rollout_results.append(existing_results[key])
            continue
        screen = {
            "game_seed": int(payload["game_seed"]),
            "turn_index": int(payload["turn_index"]),
            "player_id": int(payload["player_id"]),
            "chosen_action_index": int(payload["audit_action_index"]),
            "priority_alternative_action_indices": [],
            "categories": ["deterministic_policy_reserve"],
        }
        result = _evaluate_suspicious_state(
            payload=payload,
            screen=screen,
            state_position=state_position,
            scorer=scorer,
            env=env,
            codec=codec,
            config=rollout_config,
        )
        result["turn_band"] = _turn_band_label(int(payload["turn_index"]))
        result["recorded_action_index"] = int(payload["action_index"])
        result["recorded_action_matches_policy"] = bool(
            int(payload["action_index"]) == int(payload["audit_action_index"])
        )
        rollout_results.append(result)
        _write_progress(
            progress_path,
            identity=identity,
            selection=selection,
            selected_keys=selected_keys,
            selected_payloads=selected,
            rollout_results=rollout_results,
        )
        if config.log_every_states > 0 and (state_position + 1) % config.log_every_states == 0:
            resolved = sum(bool(row["resolved_mistake"]) for row in rollout_results)
            print(
                f"[reserve-generalization] states={state_position + 1}/{len(selected)}"
                f" resolved={resolved}"
            )

    summary = _summarize(config, selection, rollout_results)
    output = {
        "audit": "policy035_reserve_generalization_v1",
        "preregistered_gate": _gate_spec(config),
        "config": _config_payload(config, progress_path),
        "selection": selection,
        "selected_payloads": selected,
        "summary": summary,
        "rollout_results": rollout_results,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(_render_report(output), encoding="utf-8")
    _write_progress(
        progress_path,
        identity=identity,
        selection=selection,
        selected_keys=selected_keys,
        selected_payloads=selected,
        rollout_results=rollout_results,
        complete=True,
    )
    return output


def _select_policy_reserve_states(
    *,
    config: ReserveGeneralizationAuditConfig,
    scorer: Any,
    env: SplendorEnv,
    codec: ActionCodec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_limit = config.samples_per_band * config.selection_pool_multiplier
    pools: dict[str, list[dict[str, Any]]] = {
        _band_label(band): [] for band in TURN_BANDS
    }
    eligible_counts = Counter()
    early_policy_rows = 0
    rngs = {
        _band_label(band): random.Random(config.seed + index * 1_000_003)
        for index, band in enumerate(TURN_BANDS)
    }
    with config.replay_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            turn = int(payload.get("turn_index", -1))
            band = _turn_band(turn)
            if band is None or not _is_nonhuman_policy_row(payload, config.policy_marker):
                continue
            early_policy_rows += 1
            state = deserialize_state_snapshot(payload["state_snapshot"])
            action_map = _action_map(state, env, codec)
            scored = scorer.score_actions(state, list(action_map.values()))
            if not scored:
                continue
            top = max(scored, key=lambda item: (float(item.score), -int(item.action_index)))
            if top.action.action_type not in RESERVE_TYPES:
                continue
            label = _band_label(band)
            eligible_counts[label] += 1
            candidate = dict(payload)
            candidate["audit_action_index"] = int(top.action_index)
            candidate["audit_action_type"] = top.action.action_type.value
            candidate["audit_prior_score"] = float(top.score)
            _reservoir_add(
                pools[label],
                candidate,
                seen=eligible_counts[label],
                limit=pool_limit,
                rng=rngs[label],
            )

    selected_by_band: dict[str, list[dict[str, Any]]] = {}
    used_games: set[int] = set()
    for band in reversed(TURN_BANDS):
        label = _band_label(band)
        candidates = list(pools[label])
        random.Random(config.seed + band[0] * 10_007 + 71).shuffle(candidates)
        chosen = []
        for payload in candidates:
            game_seed = int(payload["game_seed"])
            if game_seed in used_games:
                continue
            chosen.append(payload)
            used_games.add(game_seed)
            if len(chosen) == config.samples_per_band:
                break
        if len(chosen) != config.samples_per_band:
            raise ValueError(
                f"Could select only {len(chosen)} unique games for turn band {label}; "
                f"increase selection_pool_multiplier."
            )
        selected_by_band[label] = chosen

    selected = [
        payload
        for band in TURN_BANDS
        for payload in selected_by_band[_band_label(band)]
    ]
    selected.sort(key=lambda row: (_band_order(int(row["turn_index"])), int(row["game_seed"])))
    selection = {
        "source_rows_screened": early_policy_rows,
        "eligible_deterministic_reserve_states_by_band": {
            _band_label(band): eligible_counts[_band_label(band)] for band in TURN_BANDS
        },
        "reservoir_pool_size_by_band": {
            _band_label(band): len(pools[_band_label(band)]) for band in TURN_BANDS
        },
        "selected_by_band": {
            _band_label(band): len(selected_by_band[_band_label(band)])
            for band in TURN_BANDS
        },
        "selected_states": len(selected),
        "unique_games": len(used_games),
        "human_rows_allowed": False,
        "selection_uses_rollout_outcomes": False,
        "selection_action_rule": "frozen deterministic policy035 argmax is a reserve",
        "selected_keys": [list(_result_key(row)) for row in selected],
    }
    return selected, selection


def _reservoir_add(
    pool: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if len(pool) < limit:
        pool.append(candidate)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        pool[replacement] = candidate


def _is_nonhuman_policy_row(payload: dict[str, Any], policy_marker: str) -> bool:
    if payload.get("human_player_id") is not None:
        return False
    seats = payload.get("game_bot_seats", [])
    if any("human" in str(seat).lower() for seat in seats):
        return False
    metadata = payload.get("game_bot_metadata", [])
    if any(
        isinstance(item, dict) and item.get("gui_opponent_kind") == "human"
        for item in metadata
    ):
        return False
    player = int(payload.get("player_id", -1))
    if not isinstance(metadata, list) or not 0 <= player < len(metadata):
        return False
    acting = metadata[player]
    if not isinstance(acting, dict):
        return False
    checkpoint = str(acting.get("checkpoint_path", "")).replace("\\", "/").lower()
    return policy_marker.replace("\\", "/").lower() in checkpoint


def _turn_band(turn: int) -> tuple[int, int] | None:
    return next((band for band in TURN_BANDS if band[0] <= turn <= band[1]), None)


def _turn_band_label(turn: int) -> str:
    band = _turn_band(turn)
    if band is None:
        raise ValueError(f"Turn {turn} is outside the preregistered early-game bands.")
    return _band_label(band)


def _band_label(band: tuple[int, int]) -> str:
    return f"{band[0]}-{band[1]}"


def _band_order(turn: int) -> int:
    band = _turn_band(turn)
    if band is None:
        return len(TURN_BANDS)
    return TURN_BANDS.index(band)


def _rollout_config(
    config: ReserveGeneralizationAuditConfig,
    progress_path: Path,
) -> HumanGameForensicAuditConfig:
    return HumanGameForensicAuditConfig(
        replay_path=config.replay_path,
        policy_checkpoint=config.policy_checkpoint,
        output_path=config.output_path,
        report_path=config.report_path,
        progress_path=progress_path,
        max_rollout_states=config.samples_per_band * len(TURN_BANDS),
        max_candidates_per_state=config.max_candidates_per_state,
        policy_top_k=config.policy_top_k,
        rollout_determinizations=config.rollout_determinizations,
        rollouts_per_determinization=config.rollouts_per_determinization,
        rollout_epsilon=config.rollout_epsilon,
        rollout_temperature=config.rollout_temperature,
        rollout_top_k=config.rollout_top_k,
        rollout_max_additional_turns=config.rollout_max_additional_turns,
        outcome_weight=config.outcome_weight,
        score_margin_scale=config.score_margin_scale,
        min_resolved_delta=config.min_resolved_delta,
        confidence_z=config.confidence_z,
        seed=config.seed,
        device=config.device,
        log_every_states=config.log_every_states,
        retain_rollout_returns=True,
    )


def _summarize(
    config: ReserveGeneralizationAuditConfig,
    selection: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved = [row for row in results if row["resolved_mistake"]]
    rate_interval = _wilson_interval(len(resolved), len(results))
    by_band = {}
    for band in TURN_BANDS:
        label = _band_label(band)
        rows = [row for row in results if row["turn_band"] == label]
        mistakes = [row for row in rows if row["resolved_mistake"]]
        by_band[label] = {
            "states": len(rows),
            "resolved_mistakes": len(mistakes),
            "resolved_rate": len(mistakes) / len(rows) if rows else 0.0,
            "mean_resolved_paired_delta": (
                statistics.fmean(float(row["paired_return_delta"]) for row in mistakes)
                if mistakes
                else 0.0
            ),
        }
    transitions = Counter()
    prior_gaps = []
    for row in resolved:
        actions = {int(item["action_index"]): item for item in row["actions"]}
        chosen = actions[int(row["chosen_action_index"])]
        best = actions[int(row["best_alternative_action_index"])]
        transitions[
            f"{chosen['action']['action_type']}->{best['action']['action_type']}"
        ] += 1
        prior_gaps.append(float(chosen["prior_score"]) - float(best["prior_score"]))
    multiplicity_robust = [
        row for row in resolved if _passes_bonferroni_robustness(row, config)
    ]
    expected_rollouts = config.rollout_determinizations * config.rollouts_per_determinization
    complete = bool(
        len(results) == selection["selected_states"]
        and all(
            row["candidate_count"] >= 2
            and all(int(action["rollout_count"]) == expected_rollouts for action in row["actions"])
            for row in results
        )
    )
    resolved_bands = sum(1 for row in by_band.values() if row["resolved_mistakes"] > 0)
    mean_delta = (
        statistics.fmean(float(row["paired_return_delta"]) for row in resolved)
        if resolved
        else 0.0
    )
    gates = {
        "complete_matched_rollouts": complete,
        "resolved_count": len(resolved) >= config.min_generalization_count,
        "resolved_rate_lower_bound": (
            rate_interval[0] >= config.min_generalization_rate_lower_bound
        ),
        "phase_breadth": resolved_bands >= config.min_generalization_bands,
        "mean_effect_size": mean_delta >= config.min_generalization_mean_delta,
    }
    return {
        "selected_states": len(results),
        "unique_games": selection["unique_games"],
        "resolved_mistakes": len(resolved),
        "resolved_rate": len(resolved) / len(results) if results else 0.0,
        "resolved_rate_wilson_95": list(rate_interval),
        "resolved_bands": resolved_bands,
        "mean_resolved_paired_delta": mean_delta,
        "mean_resolved_chosen_minus_best_prior": (
            statistics.fmean(prior_gaps) if prior_gaps else 0.0
        ),
        "bonferroni_robust_resolved_mistakes": len(multiplicity_robust),
        "bonferroni_robust_rate": (
            len(multiplicity_robust) / len(results) if results else 0.0
        ),
        "resolved_better_action_is_reserve": sum(
            count
            for transition, count in transitions.items()
            if transition.split("->", 1)[1] in {"RESERVE_VISIBLE", "RESERVE_DECK"}
        ),
        "resolved_better_action_is_nonreserve": sum(
            count
            for transition, count in transitions.items()
            if transition.split("->", 1)[1] not in {"RESERVE_VISIBLE", "RESERVE_DECK"}
        ),
        "recorded_action_match_rate": (
            sum(bool(row["recorded_action_matches_policy"]) for row in results) / len(results)
            if results
            else 0.0
        ),
        "by_turn_band": by_band,
        "resolved_action_transitions": dict(sorted(transitions.items())),
        "gates": gates,
        "generalizes": all(gates.values()),
    }


def _passes_bonferroni_robustness(
    result: dict[str, Any],
    config: ReserveGeneralizationAuditConfig,
    familywise_alpha: float = 0.05,
) -> bool:
    comparisons = max(int(result["candidate_count"]) - 1, 1)
    critical_z = NormalDist().inv_cdf(1.0 - familywise_alpha / (2.0 * comparisons))
    delta = float(result["paired_return_delta"])
    standard_error = float(result["paired_return_delta_standard_error"])
    det_deltas = [float(value) for value in result["determinization_deltas"]]
    return bool(
        delta >= config.min_resolved_delta
        and delta > critical_z * standard_error
        and det_deltas
        and min(det_deltas) > 0.0
    )


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _gate_spec(config: ReserveGeneralizationAuditConfig) -> dict[str, Any]:
    return {
        "sample": (
            f"{config.samples_per_band} states from each of four turn bands; "
            "one state per game; deterministic policy035 argmax must be a reserve"
        ),
        "individual_resolution": (
            f"paired delta >= {config.min_resolved_delta:.2f}, exceeds "
            f"{config.confidence_z:.2f} standard errors, and is positive in every "
            "determinization"
        ),
        "generalization": {
            "minimum_resolved_states": config.min_generalization_count,
            "minimum_wilson_95_lower_bound": config.min_generalization_rate_lower_bound,
            "minimum_bands_with_resolved_state": config.min_generalization_bands,
            "minimum_mean_resolved_paired_delta": config.min_generalization_mean_delta,
            "require_complete_matched_rollouts": True,
        },
    }


def _progress_identity(
    config: ReserveGeneralizationAuditConfig,
    selected_keys: list[list[int]],
) -> dict[str, Any]:
    return {
        "replay_path": str(config.replay_path.resolve()),
        "policy_checkpoint": str(config.policy_checkpoint.resolve()),
        "policy_marker": config.policy_marker,
        "samples_per_band": config.samples_per_band,
        "selection_pool_multiplier": config.selection_pool_multiplier,
        "selected_keys": selected_keys,
        "max_candidates_per_state": config.max_candidates_per_state,
        "policy_top_k": config.policy_top_k,
        "rollout_determinizations": config.rollout_determinizations,
        "rollouts_per_determinization": config.rollouts_per_determinization,
        "rollout_epsilon": config.rollout_epsilon,
        "rollout_temperature": config.rollout_temperature,
        "rollout_top_k": config.rollout_top_k,
        "rollout_max_additional_turns": config.rollout_max_additional_turns,
        "outcome_weight": config.outcome_weight,
        "score_margin_scale": config.score_margin_scale,
        "min_resolved_delta": config.min_resolved_delta,
        "confidence_z": config.confidence_z,
        "seed": config.seed,
    }


def _write_progress(
    path: Path,
    *,
    identity: dict[str, Any],
    selection: dict[str, Any],
    selected_keys: list[list[int]],
    selected_payloads: list[dict[str, Any]],
    rollout_results: list[dict[str, Any]],
    complete: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "audit": "policy035_reserve_generalization_progress_v1",
                "complete": complete,
                "identity": identity,
                "selection": selection,
                "selected_keys": selected_keys,
                "selected_payloads": selected_payloads,
                "rollout_results": rollout_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _config_payload(
    config: ReserveGeneralizationAuditConfig,
    progress_path: Path,
) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("replay_path", "policy_checkpoint", "output_path", "report_path"):
        payload[key] = str(payload[key])
    payload["progress_path"] = str(progress_path)
    return payload


def _render_report(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        "# Policy035 Reserve Generalization Audit",
        "",
        "## Design",
        "",
        "- Source: non-human legal-observation policy035 stochastic self-play.",
        "- Human games are excluded and remain held out.",
        "- Audited choice: frozen deterministic policy035 argmax, restricted to reserves.",
        "- Sampling: one state per game, balanced across turns 0-4, 5-9, 10-14, and 15-19.",
        "- Evaluation: common-seed terminal continuations across every candidate action.",
        "",
        "## Result",
        "",
        f"- States / games: {summary['selected_states']} / {summary['unique_games']}.",
        f"- Resolved mistakes: {summary['resolved_mistakes']} "
        f"({summary['resolved_rate']:.1%}; Wilson 95% CI "
        f"{summary['resolved_rate_wilson_95'][0]:.1%}-"
        f"{summary['resolved_rate_wilson_95'][1]:.1%}).",
        f"- Mean paired delta on resolved states: "
        f"{summary['mean_resolved_paired_delta']:+.3f}.",
        f"- Bonferroni-robust resolved mistakes: "
        f"{summary['bonferroni_robust_resolved_mistakes']} "
        f"({summary['bonferroni_robust_rate']:.1%}).",
        f"- Better action remains a reserve: "
        f"{summary['resolved_better_action_is_reserve']} / "
        f"{summary['resolved_mistakes']} resolved states.",
        f"- Generalization gate: **{'PASS' if summary['generalizes'] else 'FAIL'}**.",
        "",
        "## By Turn Band",
        "",
        "| Turns | States | Resolved | Rate |",
        "|---|---:|---:|---:|",
    ]
    for label, row in summary["by_turn_band"].items():
        lines.append(
            f"| {label} | {row['states']} | {row['resolved_mistakes']} | "
            f"{row['resolved_rate']:.1%} |"
        )
    lines.extend(["", "## Gate Checks", ""])
    for name, passed in summary["gates"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Resolved Action Transitions", ""])
    if summary["resolved_action_transitions"]:
        for name, count in summary["resolved_action_transitions"].items():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            (
                "A pass supports a reserve-focused iterative or league-learning line using "
                "these non-human matched outcomes. A failure keeps the human reserve pattern "
                "diagnostic-only and does not justify training on it."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _validate(config: ReserveGeneralizationAuditConfig) -> None:
    for name, value in (
        ("samples_per_band", config.samples_per_band),
        ("selection_pool_multiplier", config.selection_pool_multiplier),
        ("max_candidates_per_state", config.max_candidates_per_state),
        ("policy_top_k", config.policy_top_k),
        ("rollout_determinizations", config.rollout_determinizations),
        ("rollouts_per_determinization", config.rollouts_per_determinization),
        ("rollout_top_k", config.rollout_top_k),
        ("rollout_max_additional_turns", config.rollout_max_additional_turns),
        ("min_generalization_count", config.min_generalization_count),
        ("min_generalization_bands", config.min_generalization_bands),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.max_candidates_per_state < 2:
        raise ValueError("At least two rollout candidates are required.")
    if not config.replay_path.exists() or not config.policy_checkpoint.exists():
        raise FileNotFoundError("Replay and policy checkpoint must exist.")
    if not 0.0 <= config.rollout_epsilon <= 1.0:
        raise ValueError("rollout_epsilon must be in [0, 1].")
    if config.rollout_temperature <= 0.0 or config.score_margin_scale <= 0.0:
        raise ValueError("Rollout temperature and score scale must be positive.")
    if not 0.0 <= config.outcome_weight <= 1.0:
        raise ValueError("outcome_weight must be in [0, 1].")
    if config.min_resolved_delta < 0.0 or config.confidence_z < 0.0:
        raise ValueError("Resolution thresholds must be non-negative.")
    if not 0.0 <= config.min_generalization_rate_lower_bound <= 1.0:
        raise ValueError("Generalization rate lower bound must be in [0, 1].")
    if not 1 <= config.min_generalization_bands <= len(TURN_BANDS):
        raise ValueError("min_generalization_bands is outside the available turn bands.")


def main() -> None:
    args = build_parser().parse_args()
    output = run_reserve_generalization_audit(
        ReserveGeneralizationAuditConfig(
            replay_path=Path(args.replay_path),
            policy_checkpoint=Path(args.policy_checkpoint),
            output_path=Path(args.output_path),
            report_path=Path(args.report_path),
            progress_path=Path(args.progress_path) if args.progress_path else None,
            policy_marker=args.policy_marker,
            samples_per_band=args.samples_per_band,
            selection_pool_multiplier=args.selection_pool_multiplier,
            max_candidates_per_state=args.max_candidates_per_state,
            policy_top_k=args.policy_top_k,
            rollout_determinizations=args.rollout_determinizations,
            rollouts_per_determinization=args.rollouts_per_determinization,
            rollout_epsilon=args.rollout_epsilon,
            rollout_temperature=args.rollout_temperature,
            rollout_top_k=args.rollout_top_k,
            rollout_max_additional_turns=args.rollout_max_additional_turns,
            outcome_weight=args.outcome_weight,
            score_margin_scale=args.score_margin_scale,
            min_resolved_delta=args.min_resolved_delta,
            confidence_z=args.confidence_z,
            min_generalization_count=args.min_generalization_count,
            min_generalization_rate_lower_bound=args.min_generalization_rate_lower_bound,
            min_generalization_bands=args.min_generalization_bands,
            min_generalization_mean_delta=args.min_generalization_mean_delta,
            seed=args.seed,
            device=args.device,
            log_every_states=args.log_every_states,
        )
    )
    summary = output["summary"]
    print(
        "[reserve-generalization]"
        f" states={summary['selected_states']}"
        f" resolved={summary['resolved_mistakes']}"
        f" rate={summary['resolved_rate']:.3f}"
        f" generalizes={summary['generalizes']}"
    )


if __name__ == "__main__":
    main()
