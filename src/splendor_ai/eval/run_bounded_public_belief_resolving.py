"""Run preregistered experiment 049: bounded public-belief resolving."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import torch

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.cfr import (
    ResolvedLeaf,
    deterministic_belief_range,
    resolve_two_ply,
)
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import canonical_public_belief_key, canonical_successor_action_key
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_selective_rollout_gate import distinct_canonical_scores
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_state_snapshot
from splendor_ai.training.run_action_preference_repair import rollout_score_margin_loop_safe
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


CHECKPOINT = Path(
    "outputs/legal_action_scorer_035_legal_observation/"
    "legal_action_attention_scorer_best.pt"
)
CHECKPOINT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
PHASES = ("early", "mid", "late")
_WORKER_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class ResolvingConfig:
    output_dir: Path = Path("outputs/bounded_public_belief_resolving_049")
    checkpoint: Path = CHECKPOINT
    source_seed_start: int = 490_000
    source_games: int = 192
    selected_per_stratum: int = 8
    determinizations: int = 3
    root_actions: int = 4
    opponent_actions: int = 4
    continuations_per_leaf: int = 8
    continuation_seed_start: int = 50_000_000
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_turns: int = 400
    max_loop_fallback_triggers: int = 64
    score_margin_scale: float = 20.0
    workers: int = 10
    bootstrap_samples: int = 20_000
    device: str = "cpu"


def run_experiment(config: ResolvingConfig | None = None) -> dict[str, Any]:
    cfg = config or ResolvingConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256_file(cfg.checkpoint)
    checkpoint_payload = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    print("049 collecting 192 fresh outcome-blind roots", flush=True)
    source = _collect_source_games(cfg)
    scorer = _LegalActionScorer(cfg.checkpoint, cfg.device, "legal_action_attention_scorer")
    selection = _select_roots(cfg, source, scorer)
    _atomic_write(cfg.output_dir / "selection.json", selection)
    print("049 evaluating 48 bounded public-belief subgames", flush=True)
    roots = selection["selected"]
    rows = _evaluate_roots_parallel(cfg, roots)
    print("049 repeating first eight roots", flush=True)
    repeated = _evaluate_roots_parallel(cfg, roots[:8], progress_name=None)
    repeat_identical = rows[:8] == repeated

    metrics = _summarize(rows, cfg)
    privacy = _privacy_audit(roots[0], scorer) if roots else {"passed": False}
    source_gate = (
        source["valid_games"] == cfg.source_games
        and source["invalid_games"] == 0
        and source["missing_assigned_phase_games"] == 0
        and selection["selected_count"] == 48
        and selection["unique_source_games"] == 48
        and all(value == 8 for value in selection["stratum_counts"].values())
    )
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
        and metrics["loop_adjudications"] / metrics["continuations"] <= 0.001
    )
    leaf_gate = (
        metrics["leaf_payoff_pearson"] >= 0.60
        and metrics["leaf_payoff_mae"] <= 0.10
        and metrics["leaf_sign_eligible"] > 0
        and metrics["leaf_sign_agreement"] >= 0.70
    )
    target_gate = (
        metrics["root_action_value_pearson"] >= 0.60
        and metrics["best_action_agreement"] >= 0.60
        and metrics["effect_sign_eligible"] >= 24
        and metrics["effect_sign_agreement"] >= 0.70
        and metrics["changed_roots"] >= 8
        and metrics["resolved_fraction"] >= 0.25
        and metrics["evaluation_gain_ci95"][0] > 0.0
        and all(item["changed"] >= 1 and item["mean_gain"] >= -0.025 for item in metrics["seat_breakdown"].values())
        and all(item["changed"] >= 1 and item["mean_gain"] >= -0.025 for item in metrics["phase_breakdown"].values())
    )
    gates = {
        "frozen_checkpoint_and_configuration": (
            checkpoint_hash == CHECKPOINT_SHA256
            and checkpoint_payload.get("model_kind") == "legal_action_attention_scorer"
            and cfg == ResolvingConfig()
        ),
        "source_and_selection": source_gate,
        "belief_privacy_and_action_structure": structural_gate,
        "continuation_integrity": continuation_gate,
        "leaf_payoff_consistency": leaf_gate,
        "resolved_target_quality": target_gate,
        "first_eight_repeat_deterministic": repeat_identical,
    }
    passed = all(gates.values())
    result = {
        "experiment": "049_bounded_public_belief_resolving",
        "protocol": "docs/experiments/bounded_public_belief_resolving_049.md",
        "configuration": _jsonable(asdict(cfg)),
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
            "authorize_separately_preregistered_resolver_gameplay_gate"
            if passed
            else "do_not_run_gameplay_or_train"
        ),
    }
    _atomic_write(cfg.output_dir / "result.json", result)
    print(f"049 result: {'PASS' if passed else 'FAIL'}", flush=True)
    return result


def _collect_source_games(cfg: ResolvingConfig) -> dict[str, Any]:
    cache = cfg.output_dir / "source_candidates.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for game_index in range(cfg.source_games):
        if game_index % 24 == 0:
            print(f"049 source game {game_index + 1}/{cfg.source_games}", flush=True)
        seed = cfg.source_seed_start + game_index
        seat, phase = assigned_stratum(game_index)
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
        summaries.append(
            {
                "game_index": game_index,
                "seed": seed,
                "assigned_seat": seat,
                "assigned_phase": phase,
                "turns": state.turn_index,
                "termination_reason": termination,
                "valid": valid,
                "candidate_found": best_candidate is not None,
            }
        )
        if valid and best_candidate is not None:
            candidates.append(best_candidate)
    payload = {
        "games": summaries,
        "valid_games": sum(row["valid"] for row in summaries),
        "invalid_games": sum(not row["valid"] for row in summaries),
        "missing_assigned_phase_games": sum(not row["candidate_found"] for row in summaries),
        "candidates": candidates,
    }
    _atomic_write(cache, payload)
    return payload


def _select_roots(cfg: ResolvingConfig, source: dict[str, Any], scorer: _LegalActionScorer) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    structural_failures = 0
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source["candidates"]:
        by_stratum[row["stratum"]].append(row)
    for seat in (0, 1):
        for phase in PHASES:
            key = f"seat{seat}_{phase}"
            for candidate in sorted(by_stratum[key], key=lambda row: (row["logit_gap"], row["game_index"])):
                try:
                    _build_subgame_structure(
                        deserialize_state_snapshot(candidate["state_snapshot"]),
                        len(selected),
                        scorer,
                        cfg,
                    )
                except (ValueError, RuntimeError):
                    structural_failures += 1
                    continue
                selected.append({**candidate, "root_index": len(selected)})
                if sum(row["stratum"] == key for row in selected) == cfg.selected_per_stratum:
                    break
    counts = Counter(row["stratum"] for row in selected)
    return {
        "selected_count": len(selected),
        "unique_source_games": len({row["game_index"] for row in selected}),
        "stratum_counts": dict(counts),
        "structural_failures": structural_failures,
        "selected": selected,
    }


def _build_subgame_structure(state, root_index: int, scorer, cfg: ResolvingConfig):
    env = SplendorEnv()
    root_ranked = distinct_canonical_scores(state, scorer.score_actions(state, env.legal_actions(state)))
    if len(root_ranked) < cfg.root_actions:
        raise ValueError("Root has fewer than four semantic actions.")
    root_keys = tuple(canonical_successor_action_key(state, row.action) for row in root_ranked[: cfg.root_actions])
    determinizations = deterministic_belief_range(
        state, state.current_player, root_index=root_index, count=cfg.determinizations
    )
    branches = []
    action_sets: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
    for det_index, determinized in enumerate(determinizations):
        action_map = actions_by_canonical_successor_key(determinized, env.legal_actions(determinized))
        if any(key not in action_map for key in root_keys):
            raise ValueError("Root action mapping changed across determinizations.")
        for root_key in root_keys:
            after_root = env.step(determinized, action_map[root_key])
            if after_root.terminal:
                raise ValueError("Two-ply root action ended before opponent response.")
            opponent_ranked = distinct_canonical_scores(
                after_root, scorer.score_actions(after_root, env.legal_actions(after_root))
            )
            if len(opponent_ranked) < cfg.opponent_actions:
                raise ValueError("Opponent node has fewer than four semantic actions.")
            response_rows = opponent_ranked[: cfg.opponent_actions]
            response_keys = tuple(canonical_successor_action_key(after_root, row.action) for row in response_rows)
            info_key = (root_key, canonical_public_belief_key(after_root, after_root.current_player))
            previous = action_sets.setdefault(info_key, response_keys)
            if previous != response_keys:
                raise ValueError("Opponent information set action ranking is unstable.")
            branches.append((det_index, root_key, after_root, info_key, response_rows))
    return root_keys, branches


def _initialize_worker(checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = _LegalActionScorer(Path(checkpoint), device, "legal_action_attention_scorer")


def _evaluate_roots_parallel(
    cfg: ResolvingConfig,
    roots: list[dict[str, Any]],
    progress_name: str | None = "evaluation.progress.jsonl",
) -> list[dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    progress = cfg.output_dir / progress_name if progress_name else None
    if progress is not None and progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[int(row["root_index"])] = row["result"]
    missing = [row for row in roots if int(row["root_index"]) not in completed]
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(cfg.workers, len(missing)),
            initializer=_initialize_worker,
            initargs=(str(cfg.checkpoint), cfg.device),
        ) as executor:
            futures = {executor.submit(_evaluate_root_worker, row, cfg): int(row["root_index"]) for row in missing}
            handle = progress.open("a", encoding="utf-8") if progress is not None else None
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    completed[index] = future.result()
                    if handle is not None:
                        handle.write(json.dumps({"root_index": index, "result": completed[index]}) + "\n")
                        handle.flush()
                    print(f"049 evaluated roots {len(completed)}/{len(roots)}", flush=True)
            finally:
                if handle is not None:
                    handle.close()
    return [completed[int(row["root_index"])] for row in roots]


def _evaluate_root_worker(root: dict[str, Any], cfg: ResolvingConfig) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("Resolver worker scorer was not initialized.")
    return _evaluate_root(root, cfg, _WORKER_SCORER)


def _evaluate_root(root: dict[str, Any], cfg: ResolvingConfig, scorer) -> dict[str, Any]:
    state = deserialize_state_snapshot(root["state_snapshot"])
    root_index = int(root["root_index"])
    root_player = int(root["actor_seat"])
    root_keys, branches = _build_subgame_structure(state, root_index, scorer, cfg)
    leaves: list[ResolvedLeaf] = []
    leaf_halves: list[dict[str, Any]] = []
    counters = Counter()
    env = SplendorEnv()
    for det_index, root_key, after_root, info_key, response_rows in branches:
        for response in response_rows:
            response_key = canonical_successor_action_key(after_root, response.action)
            values = []
            for continuation_index in range(cfg.continuations_per_leaf):
                seed = (
                    cfg.continuation_seed_start
                    + root_index * 1_000_003
                    + det_index * 10_007
                    + continuation_index
                )
                outcome = rollout_score_margin_loop_safe(
                    env,
                    scorer,
                    after_root,
                    response.action,
                    root_player,
                    seed,
                    cfg.continuation_epsilon,
                    cfg.continuation_temperature,
                    cfg.continuation_top_k,
                    cfg.continuation_max_turns,
                    cfg.score_margin_scale,
                    cfg.max_loop_fallback_triggers,
                )
                values.append(outcome.value)
                counters["continuations"] += 1
                counters["resolved"] += int(outcome.resolved)
                counters["hard_caps"] += int(outcome.hard_cap_reached)
                counters["adjudications"] += int(outcome.loop_adjudicated)
                counters["fallbacks"] += outcome.fallback_triggers
            discovery = statistics.fmean(values[:4])
            evaluation = statistics.fmean(values[4:])
            leaves.append(
                ResolvedLeaf(
                    determinization=det_index,
                    root_action=root_key,
                    opponent_information_set=info_key,
                    opponent_action=response_key,
                    discovery_payoff=discovery,
                    evaluation_payoff=evaluation,
                )
            )
            leaf_halves.append({"discovery": discovery, "evaluation": evaluation})
    discovery_resolution = resolve_two_ply(
        leaves, root_keys, cfg.determinizations, payoff="discovery_payoff"
    )
    evaluation_resolution = resolve_two_ply(
        leaves, root_keys, cfg.determinizations, payoff="evaluation_payoff"
    )
    selected = discovery_resolution.best_index
    parent = 0
    discovery_effect = discovery_resolution.action_values[selected] - discovery_resolution.action_values[parent]
    evaluation_gain = evaluation_resolution.action_values[selected] - evaluation_resolution.action_values[parent]
    return {
        "root_index": root_index,
        "source_game_index": root["game_index"],
        "source_seed": root["seed"],
        "turn": root["turn"],
        "actor_seat": root_player,
        "phase": root["phase"],
        "logit_gap": root["logit_gap"],
        "root_action_mapping_stable": True,
        "opponent_information_sets_stable": True,
        "illegal_actions": 0,
        "continuations": counters["continuations"],
        "resolved_continuations": counters["resolved"],
        "hard_caps": counters["hard_caps"],
        "loop_adjudications": counters["adjudications"],
        "fallback_triggers": counters["fallbacks"],
        "leaf_halves": leaf_halves,
        "discovery": discovery_resolution.to_payload(),
        "evaluation": evaluation_resolution.to_payload(),
        "discovery_effect": discovery_effect,
        "evaluation_gain": evaluation_gain,
        "changed": selected != parent,
        "best_action_agrees": selected == evaluation_resolution.best_index,
    }


def _summarize(rows: list[dict[str, Any]], cfg: ResolvingConfig) -> dict[str, Any]:
    leaf_discovery = [leaf["discovery"] for row in rows for leaf in row["leaf_halves"]]
    leaf_evaluation = [leaf["evaluation"] for row in rows for leaf in row["leaf_halves"]]
    action_discovery = [value for row in rows for value in row["discovery"]["action_values"]]
    action_evaluation = [value for row in rows for value in row["evaluation"]["action_values"]]
    leaf_eligible = [
        index
        for index, (left, right) in enumerate(zip(leaf_discovery, leaf_evaluation, strict=True))
        if abs(left) >= 0.025 and abs(right) >= 0.025
    ]
    effect_eligible = [
        row
        for row in rows
        if abs(row["discovery_effect"]) >= 0.01 and abs(row["evaluation_gain"]) >= 0.01
    ]
    resolved = [
        row
        for row in rows
        if row["changed"]
        and row["discovery_effect"] * row["evaluation_gain"] > 0.0
        and abs(row["evaluation_gain"]) >= 0.025
    ]
    gains = [row["evaluation_gain"] for row in rows]
    return {
        "continuations": sum(row["continuations"] for row in rows),
        "resolved_continuations": sum(row["resolved_continuations"] for row in rows),
        "hard_caps": sum(row["hard_caps"] for row in rows),
        "loop_adjudications": sum(row["loop_adjudications"] for row in rows),
        "fallback_triggers": sum(row["fallback_triggers"] for row in rows),
        "leaves": len(leaf_discovery),
        "leaf_payoff_pearson": _pearson(leaf_discovery, leaf_evaluation),
        "leaf_payoff_mae": statistics.fmean(
            abs(left - right) for left, right in zip(leaf_discovery, leaf_evaluation, strict=True)
        ),
        "leaf_sign_eligible": len(leaf_eligible),
        "leaf_sign_agreement": sum(
            leaf_discovery[index] * leaf_evaluation[index] > 0.0 for index in leaf_eligible
        ) / max(len(leaf_eligible), 1),
        "root_action_value_pearson": _pearson(action_discovery, action_evaluation),
        "best_action_agreement": sum(row["best_action_agrees"] for row in rows) / len(rows),
        "effect_sign_eligible": len(effect_eligible),
        "effect_sign_agreement": sum(
            row["discovery_effect"] * row["evaluation_gain"] > 0.0 for row in effect_eligible
        ) / max(len(effect_eligible), 1),
        "changed_roots": sum(row["changed"] for row in rows),
        "resolved_roots": len(resolved),
        "resolved_fraction": len(resolved) / len(rows),
        "mean_evaluation_gain": statistics.fmean(gains),
        "evaluation_gain_ci95": _bootstrap_mean_ci(gains, cfg.bootstrap_samples, 49_000_049),
        "seat_breakdown": _breakdown(rows, "actor_seat"),
        "phase_breakdown": _breakdown(rows, "phase"),
    }


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows})
    result = {}
    for value in values:
        group = [row for row in rows if str(row[key]) == value]
        result[value] = {
            "states": len(group),
            "changed": sum(row["changed"] for row in group),
            "mean_gain": statistics.fmean(row["evaluation_gain"] for row in group),
        }
    return result


def _privacy_audit(root: dict[str, Any], scorer) -> dict[str, Any]:
    state = deserialize_state_snapshot(root["state_snapshot"])
    permuted = state.copy_shallow()
    for tier in (1, 2, 3):
        permuted.hidden_tier_decks[tier].reverse()
    first = deterministic_belief_range(state, state.current_player, root_index=0)
    second = deterministic_belief_range(permuted, permuted.current_player, root_index=0)
    hidden_order_invariant = all(
        state_signature(left) == state_signature(right)
        for left, right in zip(first, second, strict=True)
    )
    public_root_equal = canonical_public_belief_key(state, state.current_player) == canonical_public_belief_key(
        permuted, permuted.current_player
    )
    checks = {
        "hidden_order_determinization_invariant": hidden_order_invariant,
        "public_root_invariant": public_root_equal,
    }
    return {"checks": checks, "passed": all(checks.values())}


def assigned_stratum(game_index: int) -> tuple[int, str]:
    cell = game_index % 6
    return cell // 3, PHASES[cell % 3]


def _phase(turn: int) -> str:
    return "early" if turn <= 19 else "mid" if turn <= 49 else "late"


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[min(int(0.975 * samples), samples - 1)]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ResolvingConfig().output_dir)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(ResolvingConfig(output_dir=args.output_dir))


if __name__ == "__main__":
    main()
