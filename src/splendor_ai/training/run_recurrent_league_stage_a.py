"""Run the frozen CPU feasibility smoke for recurrent league Stage A (061)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.cfr.full_game import encode_public_history
from splendor_ai.encoding.public_belief import canonical_legal_action_descriptor
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.recurrent_policy import (
    PublicHistoryTracker,
    RecurrentLegalActionAttentionScorerModel,
    RecurrentPolicyInput,
    build_recurrent_policy_input,
    collate_recurrent_policy_inputs,
    load_recurrent_policy_checkpoint,
    load_recurrent_policy_from_parent,
    save_recurrent_policy_checkpoint,
)


PARENT_SHA256 = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
SEEDS = tuple(range(610_000, 610_008))
VALID_ENDINGS = ("completed", "forced_pass_deadlock")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _model_forward(model, batch):
    return model(
        batch["observation"],
        batch["legal_action_features"],
        batch["legal_action_mask"],
        batch["history_action_type_ids"],
        batch["history_card_ids"],
        batch["history_noble_ids"],
        batch["history_numeric_features"],
        batch["history_mask"],
    )


def _adjudicate(state: SplendorState) -> int | None:
    scores = [player.score for player in state.players]
    best = max(scores)
    contenders = [index for index, score in enumerate(scores) if score == best]
    if len(contenders) == 1:
        return contenders[0]
    fewest = min(len(state.players[index].purchased_cards) for index in contenders)
    remaining = [
        index
        for index in contenders
        if len(state.players[index].purchased_cards) == fewest
    ]
    return remaining[0] if len(remaining) == 1 else None


def _trace_digest(trace: list[tuple[object, ...]]) -> str:
    payload = json.dumps(trace, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_game(
    seed: int,
    model: RecurrentLegalActionAttentionScorerModel,
) -> tuple[dict[str, Any], list[RecurrentPolicyInput], list[float]]:
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    tracker = PublicHistoryTracker()
    rows: list[RecurrentPolicyInput] = []
    history_errors: list[float] = []
    trace: list[tuple[object, ...]] = []
    forced_pass_streak = 0
    parent_exact = True
    greedy_exact = True
    observer_legal = True
    blind_reserve_events = 0

    while not state.terminal and state.turn_index < 400:
        actions = env.legal_actions(state)
        if not actions:
            return (
                {
                    "seed": seed,
                    "termination_reason": "stalled",
                    "turns": state.turn_index,
                },
                rows,
                history_errors,
            )
        row = build_recurrent_policy_input(state, actions, tracker.events)
        batch = collate_recurrent_policy_inputs([row])
        with torch.no_grad():
            parent_scores, parent_value = model.parent(
                batch["observation"],
                batch["legal_action_features"],
                batch["legal_action_mask"],
            )
            scores, value = _model_forward(model, batch)
        parent_exact &= torch.equal(scores, parent_scores)
        parent_exact &= torch.equal(value, parent_value)
        parent_row = int(torch.argmax(parent_scores[0, : len(actions)]).item())
        recurrent_row = int(torch.argmax(scores[0, : len(actions)]).item())
        greedy_exact &= parent_row == recurrent_row
        action = actions[recurrent_row]
        trace.append(canonical_legal_action_descriptor(state, action))
        rows.append(row)

        if tracker.events:
            with torch.no_grad():
                full_context = model.encode_history(
                    batch["history_action_type_ids"],
                    batch["history_card_ids"],
                    batch["history_noble_ids"],
                    batch["history_numeric_features"],
                    batch["history_mask"],
                )
                hidden = None
                stepped = None
                for event_index in range(len(tracker.events)):
                    stepped, hidden = model.step_history(
                        batch["history_action_type_ids"][:, event_index],
                        batch["history_card_ids"][:, event_index],
                        batch["history_noble_ids"][:, event_index],
                        batch["history_numeric_features"][:, event_index],
                        hidden,
                    )
            assert stepped is not None
            history_errors.append(float(torch.max(torch.abs(full_context - stepped)).item()))

        before = state
        next_state = env.step(state, action)
        event = tracker.record(before, action)
        if action.action_type == ActionType.RESERVE_DECK:
            blind_reserve_events += 1
            observer_legal &= event[4] is None and event[5] == action.tier
        left = encode_public_history((event,), 0)
        right = encode_public_history((event,), 1)
        observer_legal &= left.action_type_ids == right.action_type_ids
        observer_legal &= left.card_ids == right.card_ids
        observer_legal &= left.noble_ids == right.noble_ids
        left_numeric = list(left.numeric_features[0])
        right_numeric = list(right.numeric_features[0])
        observer_legal &= left_numeric[32] == -right_numeric[32]
        left_numeric[32] = 0.0
        right_numeric[32] = 0.0
        observer_legal &= left_numeric == right_numeric

        forced_pass = (
            len(actions) == 1 and actions[0].action_type == ActionType.PASS
        )
        forced_pass_streak = forced_pass_streak + 1 if forced_pass else 0
        state = next_state
        if forced_pass_streak >= 2:
            return (
                {
                    "seed": seed,
                    "termination_reason": "forced_pass_deadlock",
                    "turns": state.turn_index,
                    "winner": _adjudicate(state),
                    "scores": [state.players[0].score, state.players[1].score],
                    "trace_sha256": _trace_digest(trace),
                    "parent_exact": parent_exact,
                    "greedy_exact": greedy_exact,
                    "observer_legal": observer_legal,
                    "blind_reserve_events": blind_reserve_events,
                },
                rows,
                history_errors,
            )

    reason = "completed" if state.terminal else "max_turns"
    return (
        {
            "seed": seed,
            "termination_reason": reason,
            "turns": state.turn_index,
            "winner": state.winner if state.terminal else _adjudicate(state),
            "scores": [state.players[0].score, state.players[1].score],
            "trace_sha256": _trace_digest(trace),
            "parent_exact": parent_exact,
            "greedy_exact": greedy_exact,
            "observer_legal": observer_legal,
            "blind_reserve_events": blind_reserve_events,
        },
        rows,
        history_errors,
    )


def _batched_contracts(
    rows: list[RecurrentPolicyInput],
    model: RecurrentLegalActionAttentionScorerModel,
    reloaded: RecurrentLegalActionAttentionScorerModel,
) -> dict[str, Any]:
    batch_greedy_equal = True
    reload_exact = True
    parent_batch_exact = True
    for start in range(0, len(rows), 64):
        chunk = rows[start : start + 64]
        batch = collate_recurrent_policy_inputs(chunk)
        with torch.no_grad():
            scores, value = _model_forward(model, batch)
            reload_scores, reload_value = _model_forward(reloaded, batch)
            parent_scores, parent_value = model.parent(
                batch["observation"],
                batch["legal_action_features"],
                batch["legal_action_mask"],
            )
        reload_exact &= torch.equal(scores, reload_scores)
        reload_exact &= torch.equal(value, reload_value)
        parent_batch_exact &= torch.equal(scores, parent_scores)
        parent_batch_exact &= torch.equal(value, parent_value)
        for index, row in enumerate(chunk):
            width = len(row.legal_action_features)
            batch_choice = int(torch.argmax(scores[index, :width]).item())
            single = collate_recurrent_policy_inputs([row])
            with torch.no_grad():
                single_scores, _single_value = _model_forward(model, single)
            single_choice = int(torch.argmax(single_scores[0, :width]).item())
            batch_greedy_equal &= batch_choice == single_choice
    return {
        "batch_single_greedy_equal": batch_greedy_equal,
        "checkpoint_reload_exact": reload_exact,
        "parent_batch_exact": parent_batch_exact,
    }


def _time_models(
    rows: list[RecurrentPolicyInput],
    model: RecurrentLegalActionAttentionScorerModel,
) -> dict[str, float]:
    batches = [
        collate_recurrent_policy_inputs(rows[start : start + 64])
        for start in range(0, len(rows), 64)
    ]
    with torch.no_grad():
        for batch in batches:
            model.parent(
                batch["observation"],
                batch["legal_action_features"],
                batch["legal_action_mask"],
            )
            _model_forward(model, batch)

    def measure(recurrent: bool) -> float:
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                for batch in batches:
                    if recurrent:
                        _model_forward(model, batch)
                    else:
                        model.parent(
                            batch["observation"],
                            batch["legal_action_features"],
                            batch["legal_action_mask"],
                        )
        return time.perf_counter() - started

    parent_seconds = measure(False)
    recurrent_seconds = measure(True)
    decisions = 3 * len(rows)
    return {
        "parent_seconds": parent_seconds,
        "recurrent_seconds": recurrent_seconds,
        "parent_decisions_per_second": decisions / parent_seconds,
        "recurrent_decisions_per_second": decisions / recurrent_seconds,
        "recurrent_to_parent_ratio": recurrent_seconds / parent_seconds,
    }


def run(parent_path: Path, output_dir: Path) -> dict[str, Any]:
    parent_hash = _sha256(parent_path)
    checkpoint_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    model = load_recurrent_policy_from_parent(parent_path).eval()
    checkpoint_path = output_dir / "recurrent_policy_epoch0.pt"
    provenance = {
        "experiment": "061",
        "parent_path": str(parent_path).replace("\\", "/"),
        "parent_sha256": parent_hash,
        "training_steps": 0,
    }
    save_recurrent_policy_checkpoint(checkpoint_path, model, provenance)
    reloaded = load_recurrent_policy_checkpoint(checkpoint_path).eval()

    runs: list[list[dict[str, Any]]] = []
    retained_rows: list[RecurrentPolicyInput] = []
    history_errors: list[float] = []
    for repetition in range(2):
        games = []
        for seed in SEEDS:
            game, rows, errors = _run_game(seed, model)
            games.append(game)
            if repetition == 0:
                retained_rows.extend(rows)
                history_errors.extend(errors)
        runs.append(games)

    batch_contracts = _batched_contracts(retained_rows, model, reloaded)
    timing = _time_models(retained_rows, model)
    traces_repeat = runs[0] == runs[1]
    valid_endings = all(
        game.get("termination_reason") in VALID_ENDINGS
        for games in runs
        for game in games
    )
    parent_exact = all(game.get("parent_exact") is True for game in runs[0])
    greedy_exact = all(game.get("greedy_exact") is True for game in runs[0])
    observer_legal = all(game.get("observer_legal") is True for game in runs[0])
    maximum_history_error = max(history_errors, default=0.0)
    parent_frozen = all(not parameter.requires_grad for parameter in model.parent.parameters())
    recurrent_trainable = any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("parent.")
    )
    residual_zero = (
        int(torch.count_nonzero(model.residual_output.weight).item()) == 0
        and int(torch.count_nonzero(model.residual_output.bias).item()) == 0
    )
    finite = all(
        math.isfinite(value)
        for value in (
            maximum_history_error,
            timing["parent_seconds"],
            timing["recurrent_seconds"],
            timing["recurrent_to_parent_ratio"],
        )
    )
    integrity = {
        "parent_hash_exact": parent_hash == PARENT_SHA256,
        "parent_kind_exact": checkpoint_payload.get("model_kind")
        == "legal_action_attention_scorer",
        "parent_frozen": parent_frozen,
        "recurrent_trainable": recurrent_trainable,
        "residual_output_zero": residual_zero,
        "parent_scores_values_exact": parent_exact
        and batch_contracts["parent_batch_exact"],
        "greedy_actions_exact": greedy_exact,
        "valid_endings": valid_endings,
        "two_runs_complete": len(runs[0]) == len(runs[1]) == len(SEEDS),
        "deterministic_repeats": traces_repeat,
        "observer_encoding_legal": observer_legal,
        "history_step_error_within_1e6": maximum_history_error <= 1e-6,
        "batch_single_greedy_equal": batch_contracts[
            "batch_single_greedy_equal"
        ],
        "checkpoint_reload_exact": batch_contracts["checkpoint_reload_exact"],
        "checkpoint_provenance_exact": provenance["parent_sha256"] == PARENT_SHA256,
        "throughput_ratio_within_3x": timing["recurrent_to_parent_ratio"] <= 3.0,
        "finite_metrics": finite,
        "no_training": provenance["training_steps"] == 0,
    }
    passed = all(integrity.values())
    decision = (
        "recurrent_stage_a_feasible_authorize_stage_b_preregistration"
        if passed
        else "recurrent_stage_a_infeasible_repair_engineering_before_scale"
    )
    payload = {
        "experiment": "061",
        "protocol": "docs/experiments/recurrent_league_stage_a_061.md",
        "parent": {
            "path": str(parent_path).replace("\\", "/"),
            "sha256": parent_hash,
            "checkpoint_kind": checkpoint_payload.get("model_kind"),
        },
        "checkpoint": {
            "path": str(checkpoint_path).replace("\\", "/"),
            "sha256": _sha256(checkpoint_path),
            "provenance": provenance,
        },
        "configuration": {
            "seeds": list(SEEDS),
            "repetitions": 2,
            "device": "cpu",
            "training_steps": 0,
            "model_config": {
                "history_hidden_size": model.config.history_hidden_size,
                "residual_hidden_size": model.config.residual_hidden_size,
            },
        },
        "summary": {
            "games_per_run": len(runs[0]),
            "retained_decisions": len(retained_rows),
            "maximum_full_step_history_error": maximum_history_error,
            "blind_reserve_events": sum(
                int(game.get("blind_reserve_events", 0)) for game in runs[0]
            ),
            "timing": timing,
        },
        "runs": runs,
        "integrity": integrity,
        "integrity_pass": passed,
        "decision": decision,
    }
    _atomic_json(output_dir / "result.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/recurrent_league_stage_a_061"),
    )
    args = parser.parse_args()
    result = run(args.parent, args.output_dir)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(result["decision"])
    if not result["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
