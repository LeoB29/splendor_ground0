"""Loop-safe full reevaluation of the frozen action-preference corpus."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
import random
import statistics
from typing import Any

import torch

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import ActionCodec, canonical_successor_action_key
from splendor_ai.engine.actions import Action
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.parallel_paired import (
    PairedCheckpointConfig,
    run_paired_checkpoint_benchmark,
)
from splendor_ai.training.entity_actor_critic import normalized_score_margin
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_action_preference_corpus import (
    SPLITS,
    _split_quality,
    export_resolved_preference_rows,
)
from splendor_ai.training.run_action_preference_generation import (
    audit_preference_movement,
    _league_config,
)
from splendor_ai.training.run_action_preference_training import (
    ActionPreferenceTrainingConfig,
    train_and_gate_action_preferences,
)
from splendor_ai.training.run_entity_counterfactual_credit import _standard_error
from splendor_ai.training.run_league_generation import (
    evaluate_direct_gate,
    run_broad_gate,
    _sha256,
)
from splendor_ai.training.run_policy_delta_causal_audit import (
    summarize_policy_delta_states,
)
from splendor_ai.training.run_policy_search_disagreement import (
    ModelActionScore,
    _LegalActionScorer,
)
from splendor_ai.training.run_rollout_action_credit import _sample_scored_action


_REPAIR_SCORER: _LegalActionScorer | None = None


@dataclass(frozen=True, slots=True)
class LoopSafeContinuationResult:
    value: float
    terminal: bool
    resolved: bool
    turns: int
    fallback_triggers: int
    loop_adjudicated: bool
    hard_cap_reached: bool


@dataclass(frozen=True, slots=True)
class ActionPreferenceRepairConfig:
    output_dir: Path
    data_dir: Path
    source_output_dir: Path
    on_policy_replay: Path
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    broad_replay: Path
    continuation_seed_start: int = 36_000_000
    determinizations: int = 3
    continuations_per_determinization: int = 8
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_additional_turns: int = 400
    max_loop_fallback_triggers: int = 64
    maximum_loop_adjudicated_fraction: float = 0.001
    score_margin_scale: float = 20.0
    minimum_effect_magnitude: float = 0.01
    resolved_effect_magnitude: float = 0.025
    maximum_continuation_cutoff_fraction: float = 0.0
    minimum_split_sign_states: int = 20
    minimum_split_sign_agreement: float = 0.65
    minimum_resolved_fraction: float = 0.10
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    kl_weight: float = 1.0
    max_grad_norm: float = 1.0
    target_validation_kl: float = 0.004
    bisection_steps: int = 12
    direct_pairs: int = 200
    direct_seed_start: int = 368_000
    broad_pairs: int = 50
    broad_016_seed_start: int = 369_000
    broad_shallow_seed_start: int = 370_000
    broad_023_seed_start: int = 371_000
    workers: int = 10
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    device: str = "cpu"


def run_action_preference_repair(config: ActionPreferenceRepairConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    selection_path = config.source_output_dir / "preference_selection.json"
    failed_evaluation_path = config.source_output_dir / "preference_evaluation.json"
    protocol = {
        "format": "action_preference_loop_repair_protocol_v1",
        "source_generation": "action_preference_generation_001",
        "source_selection_sha256": _sha256(selection_path),
        "failed_evaluation_sha256": _sha256(failed_evaluation_path),
        "parent_checkpoint_sha256": _sha256(config.parent_checkpoint),
        "source_states_and_splits_are_immutable": True,
        "previous_continuation_outcomes_reused": False,
        "repair": (
            "At the exact repetition/no-progress boundary where the old evaluator "
            "would stop, choose the highest-scored least-used distinct canonical "
            "successor, prioritizing a canonical buy, and continue. If 64 fallback "
            "actions cannot escape a score-stable loop, adjudicate its limiting "
            "score margin rather than treating it as missing data."
        ),
        "fresh_continuation_seed_namespace": config.continuation_seed_start,
        "full_reevaluation_states": 2304,
        "hard_turn_cap": config.continuation_max_additional_turns,
        "training_and_promotion_order": [
            "repaired_corpus",
            "offline_pairwise",
            "movement",
            "direct_035",
            "broad",
        ],
        "config": _config_payload(config),
    }
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing action-preference repair protocol does not match.")
    else:
        _atomic_write(protocol_path, protocol)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    repaired = evaluate_repaired_preferences(config, selection)
    if not repaired["passed"]:
        return _finalize(config, "repaired_corpus_gate_failed", repaired=repaired)
    paths = {key: Path(value) for key, value in repaired["preference_paths"].items()}
    offline = train_and_gate_action_preferences(
        ActionPreferenceTrainingConfig(
            parent_checkpoint=config.parent_checkpoint,
            train_replay=paths["train"],
            validation_replay=paths["validation"],
            confirmation_replay=paths["confirmation"],
            output_dir=config.output_dir / "candidate",
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            kl_weight=config.kl_weight,
            max_grad_norm=config.max_grad_norm,
            target_validation_kl=config.target_validation_kl,
            bisection_steps=config.bisection_steps,
            bootstrap_samples=config.bootstrap_samples,
            device=config.device,
        )
    )
    if not offline["passed"]:
        return _finalize(config, "offline_gate_failed", repaired=repaired, offline=offline)
    candidate = Path(offline["candidate_checkpoint"])
    movement = audit_preference_movement(config, candidate)
    if not movement["passed"]:
        return _finalize(
            config,
            "movement_gate_failed",
            repaired=repaired,
            offline=offline,
            movement=movement,
        )
    direct = run_paired_checkpoint_benchmark(
        PairedCheckpointConfig(
            candidate_checkpoint=candidate,
            baseline_checkpoint=config.parent_checkpoint,
            output_path=config.output_dir / "direct_vs_035.json",
            checkpoint_path=config.output_dir / "direct_vs_035.progress.json",
            pairs=config.direct_pairs,
            seed_start=config.direct_seed_start,
            workers=config.workers,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
            bootstrap_samples=config.bootstrap_samples,
            device=config.device,
        )
    )
    direct_gates = evaluate_direct_gate(direct)
    broad = run_broad_gate(_league_config(config), candidate) if all(direct_gates.values()) else None
    decision = (
        "promote_action_preference_candidate"
        if all(direct_gates.values()) and broad is not None and broad["passed"]
        else "direct_gate_failed"
        if not all(direct_gates.values())
        else "broad_gate_failed"
    )
    return _finalize(
        config,
        decision,
        repaired=repaired,
        offline=offline,
        movement=movement,
        direct=direct,
        direct_gates=direct_gates,
        broad=broad,
    )


def rollout_score_margin_loop_safe(
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
    max_loop_fallback_triggers: int = 64,
) -> LoopSafeContinuationResult:
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
        at_loop_boundary = seen[signature] >= 4 or no_progress_turns >= 60
        if at_loop_boundary:
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
    margin = float(
        rollout_state.players[root_player].score
        - rollout_state.players[1 - root_player].score
    )
    return LoopSafeContinuationResult(
        value=normalized_score_margin(margin, score_margin_scale),
        terminal=rollout_state.terminal,
        resolved=rollout_state.terminal or loop_adjudicated,
        turns=rollout_state.turn_index - start_turn,
        fallback_triggers=fallback_triggers,
        loop_adjudicated=loop_adjudicated,
        hard_cap_reached=(
            not rollout_state.terminal and rollout_state.turn_index >= end_turn
        ),
    )


def select_canonical_loop_fallback(
    state: SplendorState,
    scores: tuple[ModelActionScore, ...],
    successor_counts: dict[tuple[object, ...], int],
) -> Action:
    if not scores:
        raise ValueError("Loop fallback requires at least one scored action.")
    representatives: dict[tuple[object, ...], ModelActionScore] = {}
    for score in scores:
        key = canonical_successor_action_key(state, score.action)
        current = representatives.get(key)
        if current is None or (score.score, -score.action_index) > (
            current.score,
            -current.action_index,
        ):
            representatives[key] = score
    buy_scores = [
        score
        for score in representatives.values()
        if score.action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
    ]
    if buy_scores:
        return max(
            buy_scores, key=lambda score: (score.score, -score.action_index)
        ).action
    least_uses = min(successor_counts.get(key, 0) for key in representatives)
    eligible = [
        score
        for key, score in representatives.items()
        if successor_counts.get(key, 0) == least_uses
    ]
    return max(eligible, key=lambda score: (score.score, -score.action_index)).action


def evaluate_repaired_preferences(config: ActionPreferenceRepairConfig, selection: dict[str, Any]):
    result_path = config.output_dir / "preference_evaluation_repaired.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    selected = list(selection["selected"])
    progress_path = config.output_dir / "preference_evaluation_repaired.progress.jsonl"
    header, completed = _load_progress(progress_path)
    expected_header = {
        "format": "action_preference_loop_repair_progress_v1",
        "protocol_sha256": _sha256(config.output_dir / "protocol.json"),
        "states": len(selected),
    }
    if header is None:
        progress_path.write_text(json.dumps({"header": expected_header}) + "\n", encoding="utf-8")
    elif header != expected_header:
        raise ValueError("Existing loop-repair progress does not match the protocol.")
    missing = [(index, selected[index]) for index in range(len(selected)) if index not in completed]
    print(
        f"[preference-repair] states={len(selected)} complete={len(completed)} "
        f"pending={len(missing)} workers={config.workers}",
        flush=True,
    )
    if missing:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_repair_worker,
            initargs=(str(config.parent_checkpoint), config.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_repaired_state, item, config): item[0]
                for item in missing
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    index, row = future.result()
                    row["split"] = selected[index]["split"]
                    completed[index] = row
                    handle.write(json.dumps({"state_index": index, "row": row}) + "\n")
                    handle.flush()
                    if len(completed) % 25 == 0 or len(completed) == len(selected):
                        print(
                            f"[preference-repair] complete={len(completed)}/{len(selected)}",
                            flush=True,
                        )
    rows = [completed[index] for index in range(len(selected))]
    overall = summarize_policy_delta_states(rows, config)
    split_quality = {
        split: _split_quality([row for row in rows if row["split"] == split])
        for split in SPLITS
    }
    total_fallbacks = sum(int(row["fallback_triggers"]) for row in rows)
    fallback_states = sum(int(row["fallback_triggers"]) > 0 for row in rows)
    hard_caps = sum(int(row["hard_cap_continuations"]) for row in rows)
    adjudicated = sum(int(row["loop_adjudicated_continuations"]) for row in rows)
    continuations = sum(int(row["total_continuations"]) for row in rows)
    gates = {
        "complete": len(rows) == 2304,
        "zero_continuation_cutoffs": overall["overall"]["continuation_cutoffs"] == 0,
        "zero_hard_caps": hard_caps == 0,
        "bounded_loop_adjudication": adjudicated / continuations
        <= config.maximum_loop_adjudicated_fraction,
        "stable_action_signal": overall["stability"]["stable_action_signal"],
        "train_resolved": split_quality["train"]["resolved_states"] >= 450,
        "validation_resolved": split_quality["validation"]["resolved_states"] >= 100,
        "confirmation_resolved": split_quality["confirmation"]["resolved_states"] >= 100,
        "both_preference_directions": all(
            0.20 <= split_quality[split]["challenger_preferred_fraction"] <= 0.80
            for split in SPLITS
        ),
    }
    paths = export_resolved_preference_rows(config, selected, rows)
    payload = {
        "format": "action_preference_loop_repair_evaluation_v1",
        "source_selection_sha256": _sha256(config.source_output_dir / "preference_selection.json"),
        "overall": overall,
        "split_quality": split_quality,
        "fallback_triggers": total_fallbacks,
        "fallback_states": fallback_states,
        "hard_cap_continuations": hard_caps,
        "loop_adjudicated_continuations": adjudicated,
        "loop_adjudicated_fraction": adjudicated / continuations,
        "gates": gates,
        "passed": all(gates.values()),
        "preference_paths": {key: str(value) for key, value in paths.items()},
        "rows": rows,
    }
    _atomic_write(result_path, payload)
    return payload


def _initialize_repair_worker(parent_checkpoint: str, device: str) -> None:
    global _REPAIR_SCORER
    torch.set_num_threads(1)
    _REPAIR_SCORER = _LegalActionScorer(
        Path(parent_checkpoint), device, "legal_action_attention_scorer"
    )


def _evaluate_repaired_state(item, config):
    if _REPAIR_SCORER is None:
        raise RuntimeError("Loop-repair worker scorer was not initialized.")
    state_index, payload = item
    state = deserialize_state_snapshot(payload["state_snapshot"])
    env = SplendorEnv()
    codec = ActionCodec()
    parent_index = int(payload["parent_action_index"])
    candidate_index = int(payload["candidate_action_index"])
    first_deltas: list[float] = []
    second_deltas: list[float] = []
    resolved = natural_terminal = total = fallbacks = hard_caps = adjudicated = 0
    half = config.continuations_per_determinization // 2
    for determinization_index in range(config.determinizations):
        determinized = determinize_for_observer(
            state,
            int(payload["actor_seat"]),
            random.Random(
                config.continuation_seed_start
                + state_index * 1_000_003
                + determinization_index
            ),
        )
        action_by_index = {
            codec.encode(determinized, action): action
            for action in env.legal_actions(determinized)
        }
        if parent_index not in action_by_index or candidate_index not in action_by_index:
            raise RuntimeError("Selected action is absent after legal determinization.")
        for continuation_index in range(config.continuations_per_determinization):
            seed = (
                config.continuation_seed_start
                + state_index * 10_000_019
                + determinization_index * 100_003
                + continuation_index
            )
            results = [
                rollout_score_margin_loop_safe(
                    env,
                    _REPAIR_SCORER,
                    determinized,
                    action_by_index[action_index],
                    int(payload["actor_seat"]),
                    seed,
                    config.continuation_epsilon,
                    config.continuation_temperature,
                    config.continuation_top_k,
                    config.continuation_max_additional_turns,
                    config.score_margin_scale,
                    config.max_loop_fallback_triggers,
                )
                for action_index in (parent_index, candidate_index)
            ]
            delta = results[1].value - results[0].value
            (first_deltas if continuation_index < half else second_deltas).append(delta)
            resolved += sum(int(result.resolved) for result in results)
            natural_terminal += sum(int(result.terminal) for result in results)
            total += 2
            fallbacks += sum(result.fallback_triggers for result in results)
            hard_caps += sum(int(result.hard_cap_reached) for result in results)
            adjudicated += sum(int(result.loop_adjudicated) for result in results)
    full = first_deltas + second_deltas
    mean = statistics.fmean(full)
    se = _standard_error(full)
    row = {
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
                "parent_action_index",
                "parent_action",
                "candidate_action_index",
                "candidate_action",
            )
        },
        "candidate_minus_parent_first": statistics.fmean(first_deltas),
        "candidate_minus_parent_second": statistics.fmean(second_deltas),
        "candidate_minus_parent_full": mean,
        "paired_standard_error": se,
        "resolved": (
            abs(mean) >= config.resolved_effect_magnitude and abs(mean) > 1.96 * se
        ),
        # Compatibility field consumed by the shared stability summary. In the
        # repaired protocol, a bounded score-stable loop adjudication is a
        # resolved continuation rather than a cutoff.
        "terminal_continuations": resolved,
        "natural_terminal_continuations": natural_terminal,
        "total_continuations": total,
        "fallback_triggers": fallbacks,
        "loop_adjudicated_continuations": adjudicated,
        "hard_cap_continuations": hard_caps,
    }
    return state_index, row


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/action_preference_generation_002_loop_repair")
    parser.add_argument("--data-dir", default="outputs/action_preference_generation_002_loop_repair/data")
    parser.add_argument("--source-output-dir", default="outputs/action_preference_generation_001")
    parser.add_argument(
        "--parent-checkpoint",
        default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt",
    )
    parser.add_argument(
        "--checkpoint-016",
        default="outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt",
    )
    parser.add_argument(
        "--checkpoint-023",
        default="outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt",
    )
    parser.add_argument(
        "--broad-replay", default="outputs/canonical_entity_supervised_scale_020/replays.jsonl"
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    args = build_parser().parse_args()
    source = Path(args.source_output_dir)
    run_action_preference_repair(
        ActionPreferenceRepairConfig(
            output_dir=Path(args.output_dir),
            data_dir=Path(args.data_dir),
            source_output_dir=source,
            on_policy_replay=source / "data" / "source_replays.jsonl",
            parent_checkpoint=Path(args.parent_checkpoint),
            checkpoint_016=Path(args.checkpoint_016),
            checkpoint_023=Path(args.checkpoint_023),
            broad_replay=Path(args.broad_replay),
            workers=args.workers,
            device=args.device,
        )
    )


def _load_progress(path):
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
            elif "state_index" in payload:
                rows[int(payload["state_index"])] = payload["row"]
    if header is None:
        raise ValueError("Loop-repair progress has no header.")
    return header, rows


def _finalize(config, decision, **stages):
    payload = {
        "gate": "action_preference_generation_002_loop_repair",
        "decision": decision,
        "promoted": decision == "promote_action_preference_candidate",
        **stages,
    }
    _atomic_write(config.output_dir / "gate_final.json", payload)
    print(json.dumps({"decision": decision, "promoted": payload["promoted"]}, indent=2))
    return payload


def _config_payload(config):
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _validate(config):
    for path in (
        config.source_output_dir / "preference_selection.json",
        config.source_output_dir / "preference_evaluation.json",
        config.on_policy_replay,
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
        config.broad_replay,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if config.continuation_seed_start == 35_000_000:
        raise ValueError("Repair must use a fresh continuation seed namespace.")
    if config.continuation_max_additional_turns < 400:
        raise ValueError("Repair hard cap must be at least 400 turns.")


def _atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
