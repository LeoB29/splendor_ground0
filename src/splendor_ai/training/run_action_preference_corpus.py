"""Build a fresh game-disjoint top-two action-preference corpus."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from typing import Any

from splendor_ai.encoding import ActionCodec, canonical_successor_action_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_league_generation import OPPONENTS, _sha256
from splendor_ai.training.run_policy_delta_causal_audit import (
    PHASES,
    PolicyDeltaCausalAuditConfig,
    _evaluate_disagreement_state,
    _initialize_evaluation_worker,
    _source_replay_path,
    _source_summary_path,
    _stratum,
    _stratum_cell,
    collect_fresh_source_games,
    outcome_blind_replay_projection,
    summarize_policy_delta_states,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


SPLITS = ("train", "validation", "confirmation")
SPLIT_COUNTS_PER_STRATUM = {"train": 64, "validation": 16, "confirmation": 16}


def build_action_preference_corpus(config: Any) -> dict[str, Any]:
    source_config = PolicyDeltaCausalAuditConfig(
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
        continuations_per_determinization=config.continuations_per_determinization,
        continuation_epsilon=config.continuation_epsilon,
        continuation_temperature=config.continuation_temperature,
        continuation_top_k=config.continuation_top_k,
        continuation_max_additional_turns=config.continuation_max_additional_turns,
        score_margin_scale=config.score_margin_scale,
        minimum_effect_magnitude=config.minimum_effect_magnitude,
        resolved_effect_magnitude=config.resolved_effect_magnitude,
        maximum_continuation_cutoff_fraction=config.maximum_continuation_cutoff_fraction,
        minimum_split_sign_states=config.minimum_split_sign_states,
        minimum_split_sign_agreement=config.minimum_split_sign_agreement,
        minimum_resolved_fraction=config.minimum_resolved_fraction,
        bootstrap_samples=config.bootstrap_samples,
        workers=config.workers,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
        device=config.device,
    )
    collection = collect_fresh_source_games(source_config)
    if not collection["passed"]:
        return {"collection": collection, "passed": False, "decision": "source_failed"}
    combined_replay = combine_source_replays(config)
    selection = select_uncertain_top_two_states(config)
    if not selection["passed"]:
        return {
            "collection": collection,
            "combined_replay": str(combined_replay),
            "selection": selection,
            "passed": False,
            "decision": "selection_failed",
        }
    evaluation = evaluate_preference_pairs(config, selection)
    payload = {
        "format": "action_preference_corpus_v1",
        "collection": collection,
        "combined_replay": str(combined_replay),
        "selection": selection,
        "evaluation": evaluation,
        "preference_paths": evaluation.get("preference_paths"),
        "passed": bool(evaluation["passed"]),
        "decision": "corpus_passed" if evaluation["passed"] else "quality_failed",
    }
    _atomic_write(config.output_dir / "corpus_gate.json", payload)
    return payload


def combine_source_replays(config: Any) -> Path:
    path = config.data_dir / "source_replays.jsonl"
    if path.exists():
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as output:
        for game_index in range(config.source_games):
            with _source_replay_path(config, game_index).open("r", encoding="utf-8") as source:
                for line in source:
                    output.write(line)
    os.replace(temporary, path)
    return path


def assigned_phase_for_game(game_index: int) -> str:
    cell_ordinal = game_index // (len(OPPONENTS) * 2)
    return PHASES[cell_ordinal % len(PHASES)]


def select_uncertain_top_two_states(config: Any) -> dict[str, Any]:
    path = config.output_dir / "preference_selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    scorer = _LegalActionScorer(
        config.parent_checkpoint, config.device, "legal_action_attention_scorer"
    )
    env = SplendorEnv()
    candidates: dict[str, list[dict[str, Any]]] = {
        _stratum(opponent, seat, phase): []
        for opponent in OPPONENTS
        for seat in (0, 1)
        for phase in PHASES
    }
    rows_scanned = 0
    games_without_assigned_phase = 0
    for game_index in range(config.source_games):
        summary = json.loads(
            _source_summary_path(config, game_index).read_text(encoding="utf-8")
        )
        actor_seat = int(summary["actor_seat"])
        opponent = str(summary["opponent"])
        phase = assigned_phase_for_game(game_index)
        best: dict[str, Any] | None = None
        with _source_replay_path(config, game_index).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                observation = outcome_blind_replay_projection(json.loads(line))
                if int(observation["player_id"]) != actor_seat:
                    continue
                turn = int(observation["turn_index"])
                if _phase(turn) != phase:
                    continue
                rows_scanned += 1
                state = deserialize_state_snapshot(observation["state_snapshot"])
                ranked = scorer.score_actions(state, env.legal_actions(state))
                parent = ranked[0]
                parent_key = canonical_successor_action_key(state, parent.action)
                challenger = next(
                    (
                        row
                        for row in ranked[1:]
                        if canonical_successor_action_key(state, row.action) != parent_key
                    ),
                    None,
                )
                if challenger is None:
                    continue
                row = {
                    "source_game_index": game_index,
                    "source_game_seed": int(observation["game_seed"]),
                    "turn_index": turn,
                    "phase": phase,
                    "actor_seat": actor_seat,
                    "opponent": opponent,
                    "source_cell": _stratum_cell(opponent, actor_seat),
                    "stratum": _stratum(opponent, actor_seat, phase),
                    "parent_action_index": parent.action_index,
                    "parent_action": serialize_action(parent.action),
                    "candidate_action_index": challenger.action_index,
                    "candidate_action": serialize_action(challenger.action),
                    "parent_score": parent.score,
                    "challenger_score": challenger.score,
                    "top_two_logit_gap": parent.score - challenger.score,
                    "state_snapshot": observation["state_snapshot"],
                }
                if best is None or (
                    float(row["top_two_logit_gap"]), turn
                ) < (float(best["top_two_logit_gap"]), int(best["turn_index"])):
                    best = row
        if best is None:
            games_without_assigned_phase += 1
        else:
            candidates[str(best["stratum"])].append(best)

    selected: list[dict[str, Any]] = []
    eligible_by_stratum = {key: len(rows) for key, rows in candidates.items()}
    selection_seed = config.collection_seed_start + 701
    for stratum in sorted(candidates):
        rows = sorted(
            candidates[stratum],
            key=lambda row: (
                float(row["top_two_logit_gap"]),
                int(row["source_game_seed"]),
                int(row["turn_index"]),
            ),
        )
        required = sum(SPLIT_COUNTS_PER_STRATUM.values())
        chosen = rows[:required]
        random.Random(selection_seed + len(selected)).shuffle(chosen)
        offset = 0
        for split in SPLITS:
            count = SPLIT_COUNTS_PER_STRATUM[split]
            for row in chosen[offset : offset + count]:
                row["split"] = split
                selected.append(row)
            offset += count
    selected.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["stratum"]),
            int(row["source_game_seed"]),
        )
    )
    expected_strata = len(OPPONENTS) * 2 * len(PHASES)
    expected_total = expected_strata * sum(SPLIT_COUNTS_PER_STRATUM.values())
    split_counts = {split: sum(row["split"] == split for row in selected) for split in SPLITS}
    split_strata = {
        split: {
            stratum: sum(
                row["split"] == split and row["stratum"] == stratum for row in selected
            )
            for stratum in sorted(candidates)
        }
        for split in SPLITS
    }
    payload = {
        "format": "outcome_blind_uncertain_top_two_selection_v1",
        "rows_scanned": rows_scanned,
        "games_without_assigned_phase": games_without_assigned_phase,
        "eligible_by_stratum": eligible_by_stratum,
        "selected_states": len(selected),
        "selected_unique_games": len({row["source_game_index"] for row in selected}),
        "split_counts": split_counts,
        "split_strata": split_strata,
        "selection_seed": selection_seed,
        "outcomes_read_for_selection": False,
        "selected": selected,
    }
    payload["gates"] = {
        "enough_states": len(selected) == expected_total,
        "unique_source_games": payload["selected_unique_games"] == len(selected),
        "frozen_split_counts": split_counts
        == {"train": 1536, "validation": 384, "confirmation": 384},
        "exact_split_stratum_balance": all(
            set(split_strata[split].values()) == {SPLIT_COUNTS_PER_STRATUM[split]}
            for split in SPLITS
        ),
        "outcome_blind": True,
    }
    payload["passed"] = all(payload["gates"].values())
    _atomic_write(path, payload)
    return payload


def evaluate_preference_pairs(config: Any, selection: dict[str, Any]) -> dict[str, Any]:
    progress_path = config.output_dir / "preference_evaluation.progress.jsonl"
    header, completed = _load_progress(progress_path)
    selected = list(selection["selected"])
    expected_header = {
        "format": "action_preference_evaluation_progress_v1",
        "protocol_sha256": _sha256(config.output_dir / "protocol.json"),
        "selection_sha256": _sha256(config.output_dir / "preference_selection.json"),
        "states": len(selected),
    }
    if header is None:
        progress_path.write_text(json.dumps({"header": expected_header}) + "\n", encoding="utf-8")
    elif header != expected_header:
        raise ValueError("Existing preference evaluation progress does not match.")
    missing = [
        (index, selected[index]) for index in range(len(selected)) if index not in completed
    ]
    print(
        f"[preference-eval] states={len(selected)} complete={len(completed)} "
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
                executor.submit(_evaluate_disagreement_state, item, config): item[0]
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
                            f"[preference-eval] complete={len(completed)}/{len(selected)}",
                            flush=True,
                        )
    rows = [completed[index] for index in range(len(selected))]
    overall = summarize_policy_delta_states(rows, config)
    split_quality = {
        split: _split_quality([row for row in rows if row["split"] == split])
        for split in SPLITS
    }
    total = len(rows)
    gates = {
        "complete": total == 2304,
        "zero_continuation_cutoffs": overall["overall"]["continuation_cutoffs"] == 0,
        "stable_action_signal": overall["stability"]["stable_action_signal"],
        "train_resolved": split_quality["train"]["resolved_states"] >= 450,
        "validation_resolved": split_quality["validation"]["resolved_states"] >= 100,
        "confirmation_resolved": split_quality["confirmation"]["resolved_states"] >= 100,
        "both_preference_directions": all(
            0.20 <= split_quality[split]["challenger_preferred_fraction"] <= 0.80
            for split in SPLITS
        ),
    }
    preference_paths = export_resolved_preference_rows(config, selected, rows)
    payload = {
        "format": "action_preference_evaluation_v1",
        "overall": overall,
        "split_quality": split_quality,
        "gates": gates,
        "passed": all(gates.values()),
        "preference_paths": {key: str(value) for key, value in preference_paths.items()},
        "rows": rows,
    }
    _atomic_write(config.output_dir / "preference_evaluation.json", payload)
    return payload


def export_resolved_preference_rows(
    config: Any,
    selected: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {split: config.data_dir / f"preferences_{split}.jsonl" for split in SPLITS}
    handles = {}
    temporary_paths = {}
    try:
        for split, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary_paths[split] = temporary
            handles[split] = temporary.open("w", encoding="utf-8")
        env = SplendorEnv()
        codec = ActionCodec()
        for selection, result in zip(selected, evaluated, strict=True):
            if not result["resolved"]:
                continue
            challenger_better = float(result["candidate_minus_parent_full"]) > 0.0
            preferred = (
                int(selection["candidate_action_index"])
                if challenger_better
                else int(selection["parent_action_index"])
            )
            rejected = (
                int(selection["parent_action_index"])
                if challenger_better
                else int(selection["candidate_action_index"])
            )
            state = deserialize_state_snapshot(selection["state_snapshot"])
            legal_actions = env.legal_actions(state)
            action_by_index = {codec.encode(state, action): action for action in legal_actions}
            payload = {
                "game_seed": int(selection["source_game_seed"]),
                "turn_index": int(selection["turn_index"]),
                "player_id": int(selection["actor_seat"]),
                "action_index": preferred,
                "action_payload": serialize_action(action_by_index[preferred]),
                "model_top_action_index": rejected,
                "legal_action_indices": [codec.encode(state, action) for action in legal_actions],
                "state_snapshot": selection["state_snapshot"],
                "final_value": 0.0,
                "winner": None,
                "game_stalled": False,
                "game_timed_out": False,
                "game_termination_reason": "counterfactual_preference",
                "training_weight": 1.0,
                "preference_delta": abs(float(result["candidate_minus_parent_full"])),
                "preference_split": str(selection["split"]),
                "preference_challenger_better": challenger_better,
                "opponent_label": str(selection["opponent"]),
                "phase": str(selection["phase"]),
            }
            handles[str(selection["split"])].write(json.dumps(payload) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    for split, path in paths.items():
        os.replace(temporary_paths[split], path)
    return paths


def _split_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in rows if row["resolved"]]
    challenger = sum(float(row["candidate_minus_parent_full"]) > 0.0 for row in resolved)
    return {
        "states": len(rows),
        "resolved_states": len(resolved),
        "resolved_fraction": len(resolved) / len(rows),
        "challenger_preferred": challenger,
        "parent_preferred": len(resolved) - challenger,
        "challenger_preferred_fraction": challenger / len(resolved) if resolved else 0.0,
    }


def _phase(turn: int) -> str:
    return "early" if turn < 20 else "mid" if turn < 40 else "late"


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
            elif "state_index" in payload:
                rows[int(payload["state_index"])] = payload["row"]
    if header is None:
        raise ValueError("Preference progress file lacks a header.")
    return header, rows


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)

