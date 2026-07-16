"""Build a resumable game-disjoint late entity counterfactual corpus."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from splendor_ai.encoding import ActionCodec, actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.entity_actor_critic import (
    ActorCriticConfig,
    ActorStep,
    collect_league_rollouts,
    load_entity_actor,
)
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_state_snapshot
from splendor_ai.training.run_entity_counterfactual_credit import (
    EntityCounterfactualConfig,
    EntityPolicyScorer,
    _evaluate_state,
    select_game_disjoint_phase_states,
    summarize_counterfactual_states,
)
from splendor_ai.training.run_entity_return_ab import _full_league

_WORKER_SCORER: EntityPolicyScorer | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--progress-path")
    parser.add_argument("--rollout-games", type=int, default=240)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument("--late-states", type=int, default=240)
    parser.add_argument("--train-states", type=int, default=160)
    parser.add_argument("--validation-states", type=int, default=40)
    parser.add_argument("--confirmation-states", type=int, default=40)
    parser.add_argument("--policy-top-k", type=int, default=3)
    parser.add_argument("--reference-candidates", type=int, default=5)
    parser.add_argument("--determinizations", type=int, default=3)
    parser.add_argument("--continuations-per-determinization", type=int, default=8)
    parser.add_argument("--continuation-epsilon", type=float, default=0.1)
    parser.add_argument("--continuation-temperature", type=float, default=0.5)
    parser.add_argument("--continuation-top-k", type=int, default=3)
    parser.add_argument("--continuation-max-additional-turns", type=int, default=120)
    parser.add_argument("--minimum-action-range", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_path)
    progress_path = (
        Path(args.progress_path)
        if args.progress_path
        else output_path.with_suffix(".progress.jsonl")
    )
    config = EntityCounterfactualConfig(
        parent_checkpoint=Path(args.parent_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_035=Path(args.checkpoint_035),
        output_path=output_path,
        rollout_games=args.rollout_games,
        seed_start=args.seed_start,
        max_states=args.late_states,
        phase_targets=(0, 0, args.late_states),
        policy_top_k=args.policy_top_k,
        max_candidates=args.reference_candidates,
        baseline_mode="policy_reference",
        determinizations=args.determinizations,
        continuations_per_determinization=args.continuations_per_determinization,
        continuation_epsilon=args.continuation_epsilon,
        continuation_temperature=args.continuation_temperature,
        continuation_top_k=args.continuation_top_k,
        continuation_max_additional_turns=args.continuation_max_additional_turns,
        minimum_action_range=args.minimum_action_range,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )
    split_counts = {
        "train": args.train_states,
        "validation": args.validation_states,
        "confirmation": args.confirmation_states,
    }
    payload = build_late_counterfactual_corpus(
        config,
        split_counts,
        output_path,
        progress_path,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "states": len(payload["states"]),
                "splits": payload["split_counts"],
                "continuations": payload["summary"]["continuations"],
                "cutoff_fraction": payload["summary"]["continuation_cutoff_fraction"],
            },
            indent=2,
        )
    )


def build_late_counterfactual_corpus(
    config: EntityCounterfactualConfig,
    split_counts: dict[str, int],
    output_path: Path,
    progress_path: Path,
    workers: int,
) -> dict[str, Any]:
    if sum(split_counts.values()) != config.max_states:
        raise ValueError("Split counts must sum to max_states.")
    if any(count <= 0 for count in split_counts.values()):
        raise ValueError("Every corpus split must be non-empty.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    expected_contract = {
        "config": _config_payload(config),
        "split_counts": split_counts,
    }
    header, completed = _load_progress(progress_path)
    if header is None:
        header = _collect_source_header(config, split_counts)
        header["contract"] = expected_contract
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({"header": header}) + "\n", encoding="utf-8")
    elif header.get("contract") != expected_contract:
        raise ValueError("Existing progress contract does not match this corpus run.")
    selected = header["selected_states"]
    missing = [
        (state_index, state_payload)
        for state_index, state_payload in enumerate(selected)
        if state_index not in completed
    ]
    print(
        f"[late-counterfactual] selected={len(selected)} "
        f"completed={len(completed)} missing={len(missing)}",
        flush=True,
    )
    if missing:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(str(config.parent_checkpoint), config.device),
        ) as executor:
            futures = {
                executor.submit(_evaluate_saved_state, item, config): item[0]
                for item in missing
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    state_index, row = future.result()
                    row["split"] = selected[state_index]["split"]
                    row["source_game_seed"] = selected[state_index]["source_game_seed"]
                    row["state_snapshot"] = selected[state_index]["state_snapshot"]
                    completed[state_index] = row
                    handle.write(json.dumps({"state_index": state_index, "row": row}) + "\n")
                    handle.flush()
                    if len(completed) % 5 == 0 or len(completed) == len(selected):
                        print(
                            f"[late-counterfactual] completed={len(completed)}/{len(selected)}",
                            flush=True,
                        )
    rows = [completed[index] for index in range(len(selected))]
    summary = summarize_counterfactual_states(
        rows,
        minimum_action_range=config.minimum_action_range,
        minimum_advantage_magnitude=config.minimum_advantage_magnitude,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed_start + 809,
    )
    rollout = header["rollout"]
    selection = header["selection"]
    gates = {
        "clean_balanced_source_rollout": (
            rollout["timed_out_games"] == 0
            and rollout["stalled_games"] == 0
            and rollout["minimum_stratum_games"] == rollout["maximum_stratum_games"]
        ),
        "game_disjoint_late_states": (
            len(rows) == config.max_states
            and selection["unique_selected_games"] == config.max_states
            and selection["selected_by_phase"] == {
                "early": 0,
                "mid": 0,
                "late": config.max_states,
            }
        ),
        "continuation_cutoff_fraction_safe": (
            summary["continuation_cutoff_fraction"]
            <= config.maximum_continuation_cutoff_fraction
        ),
        "counterfactual_action_ranges_informative": (
            summary["informative_fraction"] >= config.minimum_informative_fraction
        ),
    }
    payload = {
        "corpus": "entity_late_counterfactual_q_v1",
        "contract": expected_contract,
        "rollout": header["rollout"],
        "selection": header["selection"],
        "split_counts": split_counts,
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "states": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _collect_source_header(
    config: EntityCounterfactualConfig,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    rollout_config = ActorCriticConfig(
        rollout_games=config.rollout_games,
        seed_start=config.seed_start,
        device=config.device,
    )
    league = _full_league(
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_035,
        config.device,
    )
    steps, rollout = collect_league_rollouts(
        load_entity_actor(config.parent_checkpoint, config.device),
        league,
        rollout_config,
    )
    selected, selection = select_game_disjoint_phase_states(
        steps,
        config.max_states,
        config.seed_start + 91,
        phase_targets=config.phase_targets,
    )
    if len(selected) != config.max_states:
        raise RuntimeError("Fresh source did not contain enough game-disjoint late states.")
    indices = list(range(len(selected)))
    random.Random(config.seed_start + 701).shuffle(indices)
    split_by_index: dict[int, str] = {}
    offset = 0
    for split in ("train", "validation", "confirmation"):
        count = split_counts[split]
        for index in indices[offset : offset + count]:
            split_by_index[index] = split
        offset += count
    codec = ActionCodec()
    saved_states = []
    for index, step in enumerate(selected):
        chosen = step.legal_actions[step.action_row]
        saved_states.append(
            {
                "state_snapshot": serialize_state_snapshot(step.state),
                "chosen_action_index": codec.encode(step.state, chosen),
                "game_index": step.game_index,
                "source_game_seed": config.seed_start + step.game_index,
                "actor_seat": step.actor_seat,
                "opponent_label": step.opponent_label,
                "split": split_by_index[index],
            }
        )
    return {
        "rollout": asdict(rollout),
        "selection": selection,
        "selected_states": saved_states,
    }


def _initialize_worker(parent_checkpoint: str, device: str) -> None:
    global _WORKER_SCORER
    torch.set_num_threads(1)
    _WORKER_SCORER = EntityPolicyScorer(Path(parent_checkpoint), device)


def _evaluate_saved_state(
    item: tuple[int, dict[str, Any]],
    config: EntityCounterfactualConfig,
) -> tuple[int, dict[str, Any]]:
    if _WORKER_SCORER is None:
        raise RuntimeError("Counterfactual worker scorer was not initialized.")
    state_index, payload = item
    state = deserialize_state_snapshot(payload["state_snapshot"])
    raw_actions = SplendorEnv().legal_actions(state)
    legal_actions = tuple(actions_by_canonical_successor_key(state, raw_actions).values())
    codec = ActionCodec()
    chosen_index = int(payload["chosen_action_index"])
    action_row = next(
        index
        for index, action in enumerate(legal_actions)
        if codec.encode(state, action) == chosen_index
    )
    step = ActorStep(
        state=state,
        legal_actions=legal_actions,
        action_row=action_row,
        behavior_scores=tuple(0.0 for _ in legal_actions),
        behavior_probabilities=tuple(0.0 for _ in legal_actions),
        behavior_log_probability=0.0,
        behavior_entropy=0.0,
        actor_seat=int(payload["actor_seat"]),
        game_index=int(payload["game_index"]),
        opponent_label=str(payload["opponent_label"]),
    )
    return state_index, _evaluate_state(step, state_index, _WORKER_SCORER, config)


def _load_progress(
    progress_path: Path,
) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    if not progress_path.exists():
        return None, {}
    header = None
    rows: dict[int, dict[str, Any]] = {}
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "header" in payload:
                header = payload["header"]
            elif "state_index" in payload:
                rows[int(payload["state_index"])] = payload["row"]
    if header is None:
        raise ValueError("Progress file has no header.")
    return header, rows


def _config_payload(config: EntityCounterfactualConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("parent_checkpoint", "checkpoint_016", "checkpoint_035", "output_path"):
        payload[key] = str(payload[key])
    if payload["phase_targets"] is not None:
        payload["phase_targets"] = list(payload["phase_targets"])
    return payload


if __name__ == "__main__":
    main()
