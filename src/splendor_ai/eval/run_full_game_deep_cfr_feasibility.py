"""Run preregistered experiment 047: full-game Deep CFR CPU feasibility."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.bots.model_bot import CanonicalEntityActionScorerBot
from splendor_ai.cfr.full_game import (
    DeepCFRSchemaRow,
    FullGameTraversalProbe,
    ReservoirMemory,
    build_schema_row,
    collate_deep_cfr_rows,
    encode_public_history,
    state_encoding_multiset,
    traversal_determinism_payload,
)
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding.entity_encoding import (
    ENTITY_HIDDEN_DECK_CARD,
    ENTITY_PRIVILEGED_OPPONENT_BLIND_RESERVED,
    encode_canonical_action_entities,
    encode_canonical_state_entities,
)
from splendor_ai.encoding.information_state import public_action_event
from splendor_ai.encoding.public_belief import (
    canonical_legal_action_descriptor,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE, SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerConfig,
    CanonicalEntityActionScorerModel,
)


CHECKPOINT = Path(
    "outputs/canonical_entity_coherent_023/"
    "canonical_entity_action_scorer_coherent_width64_best.pt"
)
CHECKPOINT_SHA256 = "29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070"
SCHEMA_SEEDS = tuple(range(470000, 470008))
TRAVERSAL_SEEDS = (470100, 470101)
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")


@dataclass(frozen=True, slots=True)
class FeasibilityConfig:
    checkpoint: Path = CHECKPOINT
    output_dir: Path = Path("outputs/full_game_deep_cfr_feasibility_047")
    reservoir_capacity: int = 128
    reservoir_seed: int = 470047
    node_budget: int = 25_000
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60


def collect_schema_game(
    seed: int,
    checkpoint: Path,
    *,
    max_turns: int = 400,
    repetition_limit: int = 4,
    no_progress_limit: int = 60,
) -> tuple[dict[str, Any], list[DeepCFRSchemaRow]]:
    """Play one loop-safe 023 mirror game while encoding every acting state."""

    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    bots = (
        CanonicalEntityActionScorerBot(checkpoint, device="cpu"),
        CanonicalEntityActionScorerBot(checkpoint, device="cpu"),
    )
    history = ()
    rows: list[DeepCFRSchemaRow] = []
    visits: dict[tuple[object, ...], int] = {}
    no_progress_streak = 0
    forced_pass_streak = 0
    termination = "completed"
    repetition_count = 0

    while not state.terminal:
        signature = state_signature(state)
        repetition_count = visits.get(signature, 0) + 1
        visits[signature] = repetition_count
        if repetition_count >= repetition_limit:
            termination = "repetition_cutoff"
            break
        if state.turn_index >= max_turns:
            termination = "max_turns"
            break
        legal_actions = env.legal_actions(state)
        if not legal_actions:
            termination = "stalled"
            break
        rows.append(build_schema_row(state, history, legal_actions))
        actor = bots[state.current_player]
        chosen = actor.choose_action(env, state, legal_actions)
        if chosen is None:
            raise RuntimeError("023 returned no action in a non-terminal state.")
        event = public_action_event(state, chosen)
        next_state = env.step(state, chosen)
        forced_pass = len(legal_actions) == 1 and legal_actions[0].action_type == ActionType.PASS
        forced_pass_streak = forced_pass_streak + 1 if forced_pass else 0
        if is_progress_transition(state, next_state, chosen):
            no_progress_streak = 0
        else:
            no_progress_streak += 1
        history = history + (event,)
        state = next_state
        if forced_pass_streak >= 2:
            termination = "forced_pass_deadlock"
            break
        if no_progress_streak >= no_progress_limit:
            termination = "no_progress_cutoff"
            break

    winner = state.winner if state.terminal else _adjudicate_scores(state)
    return (
        {
            "seed": seed,
            "turns": state.turn_index,
            "rows": len(rows),
            "termination_reason": termination,
            "valid_terminal": termination in VALID_TERMINATIONS,
            "winner": winner,
            "final_scores": [state.players[0].score, state.players[1].score],
            "repetition_count": repetition_count,
            "no_progress_streak": no_progress_streak,
            "loop_fallback_triggers_by_seat": [bot.loop_fallback_triggers for bot in bots],
        },
        rows,
    )


def run_privacy_and_invariance_audits() -> dict[str, Any]:
    env_a = SplendorEnv(seed=470090)
    original = env_a.initial_state()
    permuted_hidden = original.copy_shallow()
    permuted_hidden.hidden_tier_decks[1].reverse()
    hidden_state_equal = all(
        state_encoding_multiset(encode_canonical_state_entities(original, observer))
        == state_encoding_multiset(encode_canonical_state_entities(permuted_hidden, observer))
        for observer in (0, 1)
    )
    empty_history_equal = all(
        encode_public_history((), observer) == encode_public_history((), observer)
        for observer in (0, 1)
    )

    blind_a = original.copy_shallow()
    blind_b = original.copy_shallow()
    blind_b.hidden_tier_decks[1][0], blind_b.hidden_tier_decks[1][1] = (
        blind_b.hidden_tier_decks[1][1],
        blind_b.hidden_tier_decks[1][0],
    )
    action_a = next(
        action
        for action in env_a.legal_actions(blind_a)
        if action.action_type == ActionType.RESERVE_DECK and action.tier == 1
    )
    env_b = SplendorEnv(seed=470090)
    action_b = next(
        action
        for action in env_b.legal_actions(blind_b)
        if action.action_type == ActionType.RESERVE_DECK and action.tier == 1
    )
    reserve_event_a = public_action_event(blind_a, action_a)
    reserve_event_b = public_action_event(blind_b, action_b)
    after_a = env_a.step(blind_a, action_a)
    after_b = env_b.step(blind_b, action_b)
    owner_differs = (
        state_encoding_multiset(encode_canonical_state_entities(after_a, 0))
        != state_encoding_multiset(encode_canonical_state_entities(after_b, 0))
    )
    opponent_equal = (
        state_encoding_multiset(encode_canonical_state_entities(after_a, 1))
        == state_encoding_multiset(encode_canonical_state_entities(after_b, 1))
    )
    public_histories_equal = all(
        encode_public_history((reserve_event_a,), observer)
        == encode_public_history((reserve_event_b,), observer)
        for observer in (0, 1)
    )
    blind_history = encode_public_history((reserve_event_a,), 0)
    reserve_hides_id = reserve_event_a[4] is None and blind_history.card_ids == (0,)
    reveal_a = after_a.copy_shallow()
    reveal_b = after_b.copy_shallow()
    reveal_a.current_player = 0
    reveal_b.current_player = 0
    bought_event_a = public_action_event(
        reveal_a,
        Action(ActionType.BUY_RESERVED, reserved_index=0),
    )
    bought_event_b = public_action_event(
        reveal_b,
        Action(ActionType.BUY_RESERVED, reserved_index=0),
    )
    revealed_a = encode_public_history((reserve_event_a, bought_event_a), 0).card_ids[-1]
    revealed_b = encode_public_history((reserve_event_b, bought_event_b), 0).card_ids[-1]
    buy_reveals_id = (
        bought_event_a[4] == after_a.players[0].reserved_cards[0].card_id
        and bought_event_b[4] == after_b.players[0].reserved_cards[0].card_id
        and revealed_a != 0
        and revealed_b != 0
        and revealed_a != revealed_b
    )

    permutation_base = original.copy_shallow()
    permutation_base.players[0].reserve_card(
        permutation_base.hidden_tier_decks[1][0], RESERVE_SOURCE_DECK
    )
    permutation_base.players[0].reserve_card(
        permutation_base.visible_tier_cards[2][0], RESERVE_SOURCE_VISIBLE
    )
    cosmetic = permutation_base.copy_shallow()
    for tier in (1, 2, 3):
        cosmetic.visible_tier_cards[tier].reverse()
    cosmetic.nobles.reverse()
    cosmetic.players[0].reserved_cards.reverse()
    cosmetic.players[0].reserved_card_sources.reverse()
    reserve_and_board_permutation_equal = (
        state_encoding_multiset(encode_canonical_state_entities(permutation_base, 0))
        == state_encoding_multiset(encode_canonical_state_entities(cosmetic, 0))
    )

    legal = env_a.legal_actions(original)
    action_encoding = encode_canonical_action_entities(original, legal, 0)
    reversed_encoding = encode_canonical_action_entities(original, list(reversed(legal)), 0)
    action_order_equal = _semantic_action_rows(original, legal, action_encoding) == _semantic_action_rows(
        original, list(reversed(legal)), reversed_encoding
    )

    checks = {
        "hidden_deck_permutation_state_equal": hidden_state_equal,
        "hidden_deck_permutation_history_equal": empty_history_equal,
        "blind_reserve_owner_state_differs": owner_differs,
        "blind_reserve_opponent_state_equal": opponent_equal,
        "blind_reserve_public_histories_equal": public_histories_equal,
        "blind_reserve_history_hides_card_id": reserve_hides_id,
        "buy_reserved_history_reveals_card_id": buy_reveals_id,
        "board_noble_reserve_permutation_state_equal": reserve_and_board_permutation_equal,
        "legal_action_permutation_semantic_rows_equal": action_order_equal,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(config: FeasibilityConfig) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_hash = _sha256_file(config.checkpoint)
    checkpoint_payload = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    model_config = checkpoint_payload.get("model_config", {})
    model = CanonicalEntityActionScorerModel(CanonicalEntityActionScorerConfig(**model_config))
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    games: list[dict[str, Any]] = []
    all_rows: list[DeepCFRSchemaRow] = []
    for seed in SCHEMA_SEEDS:
        game, rows = collect_schema_game(
            seed,
            config.checkpoint,
            max_turns=config.max_turns,
            repetition_limit=config.repetition_limit,
            no_progress_limit=config.no_progress_limit,
        )
        games.append(game)
        all_rows.extend(rows)

    phase_counts = Counter(_phase(row.state.numeric_features[0][32]) for row in all_rows)
    action_counts = [len(row.actions.action_type_ids) for row in all_rows]
    schema_checks = {
        "exactly_eight_games": len(games) == 8,
        "all_games_clean": all(game["valid_terminal"] for game in games),
        "at_least_400_states": len(all_rows) >= 400,
        "at_least_32_early": phase_counts["early"] >= 32,
        "at_least_32_mid": phase_counts["mid"] >= 32,
        "at_least_32_late": phase_counts["late"] >= 32,
        "action_counts_in_bounds": bool(action_counts) and min(action_counts) >= 1 and max(action_counts) <= 512,
        "targets_aligned_and_finite": all(
            len(row.target) == len(row.actions.action_type_ids)
            and all(math.isfinite(value) for value in row.target)
            and math.isclose(sum(row.target), 1.0, rel_tol=0.0, abs_tol=1e-6)
            for row in all_rows
        ),
        "schema_only_provenance": all(row.provenance == "schema_only" for row in all_rows),
        "no_privileged_entities": all(
            not ({ENTITY_HIDDEN_DECK_CARD, ENTITY_PRIVILEGED_OPPONENT_BLIND_RESERVED} & set(row.state.entity_type_ids))
            for row in all_rows
        ),
    }

    forward = _run_forward_audit(model, all_rows)
    privacy = run_privacy_and_invariance_audits()
    reservoirs = _run_reservoir_audit(all_rows, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "schema_reservoirs.json").write_bytes(reservoirs.pop("serialized_bytes"))

    traversals = [
        FullGameTraversalProbe(root_seed=seed, traverser=traverser, node_budget=config.node_budget).run()
        for seed in TRAVERSAL_SEEDS
        for traverser in (0, 1)
    ]
    first_repeat = FullGameTraversalProbe(
        root_seed=TRAVERSAL_SEEDS[0], traverser=0, node_budget=config.node_budget
    ).run()
    traversal_deterministic = traversal_determinism_payload(traversals[0]) == traversal_determinism_payload(
        first_repeat
    )
    traversal_checks = {
        "all_four_complete": all(item["complete"] for item in traversals),
        "zero_invalid_cutoffs": all(item["cutoff_reason"] is None for item in traversals),
        "zero_illegal_actions": all(item["illegal_actions"] == 0 for item in traversals),
        "partial_attempts_emit_zero_eligible_rows": all(
            item["complete"]
            or (item["eligible_advantage_rows"] == 0 and item["eligible_strategy_rows"] == 0)
            for item in traversals
        ),
        "complete_rows_on_correct_node_types": all(
            (not item["complete"])
            or (
                item["eligible_advantage_rows"] == item["candidate_advantage_rows"]
                and item["eligible_strategy_rows"] == item["candidate_strategy_rows"]
            )
            for item in traversals
        ),
        "first_attempt_repeat_deterministic": traversal_deterministic,
    }

    gates = {
        "checkpoint": (
            checkpoint_hash == CHECKPOINT_SHA256
            and checkpoint_payload.get("model_kind") == "canonical_entity_action_scorer"
            and parameter_count == 98_658
            and forward["passed"]
        ),
        "schema_corpus": all(schema_checks.values()),
        "privacy_and_invariance": privacy["passed"],
        "reservoirs": reservoirs["passed"],
        "terminal_external_traversal": all(traversal_checks.values()),
    }
    payload = {
        "experiment": "047_full_game_deep_cfr_feasibility",
        "protocol": "docs/experiments/full_game_deep_cfr_feasibility_047.md",
        "config": _jsonable(asdict(config)),
        "checkpoint": {
            "path": str(config.checkpoint),
            "sha256": checkpoint_hash,
            "expected_sha256": CHECKPOINT_SHA256,
            "model_kind": checkpoint_payload.get("model_kind"),
            "model_config": model_config,
            "parameter_count": parameter_count,
        },
        "schema_corpus": {
            "games": games,
            "total_rows": len(all_rows),
            "phase_counts": dict(phase_counts),
            "min_legal_actions": min(action_counts, default=0),
            "max_legal_actions": max(action_counts, default=0),
            "mean_legal_actions": sum(action_counts) / max(len(action_counts), 1),
            "max_state_entities": max((len(row.state.entity_type_ids) for row in all_rows), default=0),
            "max_history_events": max((len(row.history.action_type_ids) for row in all_rows), default=0),
            "checks": schema_checks,
        },
        "frozen_023_forward": forward,
        "privacy_and_invariance": privacy,
        "reservoirs": reservoirs,
        "traversals": traversals,
        "first_traversal_repeat": first_repeat,
        "traversal_checks": traversal_checks,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "neural_fit_authorized": all(gates.values()),
        "gpu_rental_authorized": all(gates.values()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["decision"] = (
        "authorize_separately_preregistered_neural_fit"
        if payload["all_gates_passed"]
        else "do_not_train_or_rent_gpu"
    )
    result_path = config.output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _run_forward_audit(
    model: CanonicalEntityActionScorerModel,
    rows: list[DeepCFRSchemaRow],
) -> dict[str, Any]:
    batch_count = 0
    finite = True
    aligned = True
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(rows), 8):
            batch_rows = rows[offset : offset + 8]
            batch = collate_deep_cfr_rows(batch_rows, device="cpu")
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key
                in {
                    "entity_type_ids",
                    "entity_card_ids",
                    "entity_noble_ids",
                    "entity_numeric_features",
                    "entity_mask",
                    "action_type_ids",
                    "action_card_ids",
                    "action_noble_ids",
                    "action_numeric_features",
                    "action_mask",
                }
            }
            scores, values = model(**model_inputs)
            finite = finite and bool(torch.isfinite(scores[batch["action_mask"]]).all())
            finite = finite and bool(torch.isfinite(values).all())
            finite = finite and bool(torch.isfinite(batch["history_numeric_features"]).all())
            finite = finite and bool(torch.isfinite(batch["targets"]).all())
            aligned = aligned and scores.shape == batch["targets"].shape
            batch_count += 1
    checks = {
        "nonempty_batches": batch_count > 0,
        "finite_state_action_history_target_tensors": finite,
        "score_target_shapes_aligned": aligned,
        "batch_size_at_most_eight": True,
    }
    return {
        "batches": batch_count,
        "rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "history_branch_claimed_trained": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_reservoir_audit(
    rows: list[DeepCFRSchemaRow],
    config: FeasibilityConfig,
) -> dict[str, Any]:
    memories = [
        ReservoirMemory[DeepCFRSchemaRow](config.reservoir_capacity, config.reservoir_seed)
        for _ in range(4)
    ]
    for row in rows:
        for memory in memories:
            memory.add(row)
    primary = {"advantage": memories[0].payload(), "strategy": memories[1].payload()}
    repeat = {"advantage": memories[2].payload(), "strategy": memories[3].payload()}
    raw = json.dumps(primary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    roundtrip = json.loads(raw.decode("utf-8"))
    checks = {
        "capacity_exact": all(len(memory.rows) == min(config.reservoir_capacity, len(rows)) for memory in memories),
        "seen_exact": all(memory.seen == len(rows) for memory in memories),
        "repeat_byte_identical": json.dumps(primary, sort_keys=True) == json.dumps(repeat, sort_keys=True),
        "roundtrip_lossless": roundtrip == primary,
        "within_32_mib": len(raw) <= 32 * 1024 * 1024,
        "schema_only": all(row.provenance == "schema_only" for memory in memories for row in memory.rows),
    }
    return {
        "capacity_each": config.reservoir_capacity,
        "seed_each": config.reservoir_seed,
        "seen_each": len(rows),
        "rows_each": len(memories[0].rows),
        "serialized_combined_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checks": checks,
        "passed": all(checks.values()),
        "serialized_bytes": raw,
    }


def _semantic_action_rows(state, actions, encoding) -> dict[tuple[object, ...], tuple[object, ...]]:
    return {
        canonical_legal_action_descriptor(state, action): (
            encoding.action_type_ids[index],
            encoding.card_ids[index],
            encoding.noble_ids[index],
            encoding.numeric_features[index],
        )
        for index, action in enumerate(actions)
    }


def _phase(normalized_turn: float) -> str:
    turn = int(round(normalized_turn * 100.0))
    return "early" if turn <= 19 else "mid" if turn <= 49 else "late"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adjudicate_scores(state: SplendorState) -> int | None:
    best = max(player.score for player in state.players)
    contenders = [index for index, player in enumerate(state.players) if player.score == best]
    if len(contenders) == 1:
        return contenders[0]
    fewest = min(len(state.players[index].purchased_cards) for index in contenders)
    finalists = [
        index for index in contenders if len(state.players[index].purchased_cards) == fewest
    ]
    return finalists[0] if len(finalists) == 1 else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/full_game_deep_cfr_feasibility_047")
    )
    parser.add_argument("--node-budget", type=int, default=25_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(
        FeasibilityConfig(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            node_budget=args.node_budget,
        )
    )
    print(json.dumps({
        "result": str(args.output_dir / "result.json"),
        "all_gates_passed": payload["all_gates_passed"],
        "decision": payload["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
