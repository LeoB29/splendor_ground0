"""Existing-data plateau attribution and pooled replication for programme 105."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

import torch

from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import serialize_action

from .arena import alpha_zero_search_factory, run_paired_arena
from .endpoint_evaluation import practical_assessment
from .network import AlphaZeroInference, load_checkpoint
from .replay import AlphaZeroReplayRow, _row_from_payload
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearch, SearchConfig
from .search_primary import _arena_dict, paired_score_share_bootstrap
from .training import collate_replay_rows


@dataclass(frozen=True, slots=True)
class PlateauDiagnosisConfig:
    output_dir: Path
    result_104: Path
    installed_checkpoint: Path
    checkpoint_8: Path
    checkpoint_9: Path
    checkpoint_10: Path
    checkpoint_11: Path
    replay_7: Path
    replay_8: Path
    replay_9: Path
    replay_10: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "100ed8f8b2f2e1fdfc0ec580be11b92ff7487e77496c0bc01dc75907bcba022e"
EXPECTED_HASHES = {
    "result_104": "63a51a8b98d0850fbd39430f3fe8469046b8fcc2b5d3961f25bba69c4b3e90ad",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "checkpoint_8": "beea26beddf91c7b0ab8f531ba6bc2bfa104d4e635e12af8002d04f7fb9c9281",
    "checkpoint_9": "affd5670d0555a67094b19bd040c59bbcd1c8aa83502279110c26de6a424b8b7",
    "checkpoint_10": "690bd55e5b9c10e1b8436ee7f0b8b8c459ee9498999ac81f5eed7c1cf1819918",
    "checkpoint_11": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "replay_7": "06ee156543da64de3126eed00cb363ed5f2d4c0e49e093550024fffc2ace5a1c",
    "replay_8": "96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10",
    "replay_9": "1b13ef038ccc6b3f695d7c7860a2284272ae18c8e75d44028e4cbb6f69914f0c",
    "replay_10": "09279d9593acfd944a32571716a6527c0be40e41055b6fa1f5bc25c88f43a60c",
}
REPLAY_SEED_STARTS = {7: 1_840_000, 8: 1_880_000, 9: 1_890_000, 10: 1_900_000}
SAMPLED_SEED_OFFSETS = (3, 19, 27, 44, 58, 73, 89, 104, 121, 137, 152, 168, 183, 201, 223, 247)
TARGET_TURNS = (6, 10, 14, 18, 22, 28, 34, 40, 46, 49, 52, 54, 56, 58, 60, 62)
REPLICATION_SEED_START = 1_930_000
PAIRED_SEEDS = 100
_GAME_SEED_PATTERN = re.compile(r'"game_seed"\s*:\s*(\d+)')


def pooled_decision(assessment: dict[str, Any]) -> str:
    if assessment["passed"]:
        return "pooled_compounding_confirmed_review_035_no_install"
    if assessment["category"] == "practically_flat":
        return "pooled_practically_flat_offline_attribution_review"
    return "pooled_unhealthy_mechanism_diagnosis_required"


def pool_arena_rows(
    prior: dict[str, Any], replication: dict[str, Any]
) -> dict[str, Any]:
    prior_rows = list(prior["rows"])
    replication_rows = list(replication["rows"])
    prior_seeds = {int(row["seed"]) for row in prior_rows}
    replication_seeds = {int(row["seed"]) for row in replication_rows}
    if prior_seeds & replication_seeds:
        raise ValueError("Pooled arenas must use disjoint seed pairs.")
    rows = prior_rows + replication_rows
    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[int(row["seed"])].append(row)
    if not clusters or any(
        len(cluster) != 2
        or {int(row["candidate_seat"]) for row in cluster} != {0, 1}
        for cluster in clusters.values()
    ):
        raise ValueError("Pooled arena requires exact two-seat seed pairs.")
    games = len(rows)
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    losses = games - wins - draws
    reasons = Counter(str(row["termination_reason"]) for row in rows)
    return {
        "name": "searched_generation_11_vs_searched_generation_8_pooled_104_105",
        "paired_seeds": len(clusters),
        "games": games,
        "candidate_wins": wins,
        "candidate_losses": losses,
        "draws": draws,
        "score_share": (wins + 0.5 * draws) / games,
        "mean_margin": sum(float(row["candidate_margin"]) for row in rows) / games,
        "termination_reasons": dict(sorted(reasons.items())),
        "rows": rows,
    }


def run_plateau_diagnosis_105(config: PlateauDiagnosisConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed_path = output / "result.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sealed.get("status") == "complete" and sealed.get("protocol_sha256") == PROTOCOL_SHA256:
            return sealed
        raise ValueError("Existing 105 result is not a valid completed result.")

    result: dict[str, Any] = {
        "kind": "alphazero_plateau_diagnosis_105",
        "status": "running",
        "active_stage": "offline_forensic_audit",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "offline_audit": None,
        "searched_replication": None,
        "pooled_assessment": None,
        "selfplay_generated": False,
        "checkpoint_trained": False,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint": str(config.installed_checkpoint),
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(output, result, started)
    result["offline_audit"] = _run_or_load_offline_audit(config)
    _write_progress(output, result, started)

    result["active_stage"] = "searched_generation_11_vs_generation_8_replication"
    _write_progress(output, result, started)
    search = SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_050_105,
    )
    replication = _arena_dict(
        run_paired_arena(
            name="searched_generation_11_vs_searched_generation_8_replication_105",
            candidate_factory=alpha_zero_search_factory(
                config.checkpoint_11, config.device, search
            ),
            baseline_factory=alpha_zero_search_factory(
                config.checkpoint_8, config.device, search
            ),
            seed_start=REPLICATION_SEED_START,
            paired_seeds=PAIRED_SEEDS,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "searched_replication.json",
            resume=True,
        )
    )
    result["searched_replication"] = {
        "arena": replication,
        "paired_bootstrap": paired_score_share_bootstrap(
            replication, samples=50_000, seed=1_050_105
        ),
        "standalone_interpretation_prohibited": True,
    }
    prior_result = json.loads(config.result_104.read_text(encoding="utf-8"))
    prior_arena = prior_result["internal_endpoint"][
        "searched_generation_11_vs_searched_generation_8"
    ]["arena"]
    pooled = pool_arena_rows(prior_arena, replication)
    bootstrap = paired_score_share_bootstrap(
        pooled, samples=50_000, seed=1_050_205
    )
    assessment = practical_assessment(pooled)
    result["pooled_assessment"] = {
        "arena": pooled,
        "paired_bootstrap": bootstrap,
        "assessment": assessment,
        "prior_games": 200,
        "fresh_games": 200,
    }
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["decision"] = pooled_decision(assessment)
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed_path, result)
    return result


def _run_or_load_offline_audit(config: PlateauDiagnosisConfig) -> dict[str, Any]:
    path = config.output_dir / "offline_audit.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("Offline audit protocol hash mismatch.")
        if payload.get("input_hashes") != EXPECTED_HASHES:
            raise ValueError("Offline audit input hash mismatch.")
        return payload

    replay_paths = {
        7: config.replay_7,
        8: config.replay_8,
        9: config.replay_9,
        10: config.replay_10,
    }
    checkpoint_paths = {
        8: config.checkpoint_8,
        9: config.checkpoint_9,
        10: config.checkpoint_10,
        11: config.checkpoint_11,
    }
    selected: dict[int, dict[int, list[AlphaZeroReplayRow]]] = {}
    for generation, replay_path in replay_paths.items():
        selected[generation] = _load_selected_games(
            replay_path,
            replay_generation=generation,
            seed_start=REPLAY_SEED_STARTS[generation],
        )
    models = {
        generation: load_checkpoint(path, device=config.device)[0]
        for generation, path in checkpoint_paths.items()
    }
    network_matrix = {
        str(replay_generation): _network_matrix_for_rows(
            [row for game in games.values() for row in game],
            models=models,
            device=config.device,
        )
        for replay_generation, games in selected.items()
    }
    reconstructed: list[dict[str, Any]] = []
    reserve_trace: dict[str, Any] = {}
    for replay_generation, games in selected.items():
        snapshots, trace = _reconstruct_sampled_games(
            games,
            replay_generation=replay_generation,
            seed_start=REPLAY_SEED_STARTS[replay_generation],
        )
        reconstructed.extend(snapshots)
        reserve_trace[str(replay_generation)] = trace
    search_audit = _common_state_search_audit(
        reconstructed,
        checkpoint_paths=checkpoint_paths,
        device=config.device,
    )
    payload = {
        "kind": "alphazero_plateau_offline_audit_105",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": dict(EXPECTED_HASHES),
        "sampled_seed_offsets": list(SAMPLED_SEED_OFFSETS),
        "target_turns": list(TARGET_TURNS),
        "sample": {
            "replay_generations": list(replay_paths),
            "games_per_replay": len(SAMPLED_SEED_OFFSETS),
            "network_rows_by_replay": {
                str(generation): sum(len(rows) for rows in games.values())
                for generation, games in selected.items()
            },
            "common_search_states": len(reconstructed),
        },
        "network_checkpoint_by_corpus_matrix": network_matrix,
        "common_state_search_audit": search_audit,
        "reserve_deck_followthrough": reserve_trace,
        "descriptive_flags": _descriptive_flags(network_matrix, search_audit),
    }
    _write_json(path, payload)
    return payload


def _load_selected_games(
    path: Path, *, replay_generation: int, seed_start: int
) -> dict[int, list[AlphaZeroReplayRow]]:
    selected_seeds = {seed_start + offset for offset in SAMPLED_SEED_OFFSETS}
    games: dict[int, list[AlphaZeroReplayRow]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _GAME_SEED_PATTERN.search(line)
            if match is None:
                raise ValueError("Replay row is missing game_seed.")
            seed = int(match.group(1))
            if seed not in selected_seeds:
                continue
            row = _row_from_payload(json.loads(line))
            games[seed].append(row)
    if set(games) != selected_seeds:
        raise ValueError("Replay lacks an exact sampled game.")
    for seed, rows in games.items():
        rows.sort(key=lambda row: row.turn_index)
        if [row.turn_index for row in rows] != list(range(len(rows))):
            raise ValueError(f"Sampled replay game {seed} is not turn-contiguous.")
        if any(
            row.generation != replay_generation
            or row.checkpoint_generation != replay_generation
            or row.termination_reason != "completed"
            or row.value_target is None
            for row in rows
        ):
            raise ValueError(f"Sampled replay game {seed} violates clean identity.")
    return dict(sorted(games.items()))


def _network_matrix_for_rows(
    rows: list[AlphaZeroReplayRow],
    *,
    models: dict[int, torch.nn.Module],
    device: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    totals = {
        generation: {
            "rows": 0,
            "policy_cross_entropy": 0.0,
            "value_squared_error": 0.0,
            "value_absolute_error": 0.0,
            "policy_top_agreement": 0,
            "decisive_rows": 0,
            "decisive_top_agreement": 0,
            "raw_top_action_family_counts": Counter(),
        }
        for generation in models
    }
    pair_totals = {
        f"{left}_vs_{right}": {"rows": 0, "total_variation": 0.0, "top_changes": 0}
        for left, right in ((9, 8), (10, 8), (11, 8), (10, 9), (11, 10))
    }
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = collate_replay_rows(chunk)
            moved = {key: value.to(device) for key, value in batch.items()}
            inputs = {
                key: value
                for key, value in moved.items()
                if key not in {"policy_target", "value_target"}
            }
            probabilities: dict[int, torch.Tensor] = {}
            tops: dict[int, torch.Tensor] = {}
            for generation, model in models.items():
                logits, values = model(**inputs)
                masked = logits.masked_fill(
                    ~moved["action_mask"], torch.finfo(logits.dtype).min
                )
                probs = torch.softmax(masked, dim=1)
                probabilities[generation] = probs
                tops[generation] = torch.argmax(masked, dim=1)
                target = moved["policy_target"]
                policy_loss = -(target * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
                value_error = values - moved["value_target"]
                target_top = torch.argmax(target, dim=1)
                stats = totals[generation]
                stats["rows"] += len(chunk)
                stats["policy_cross_entropy"] += float(policy_loss.sum().item())
                stats["value_squared_error"] += float((value_error**2).sum().item())
                stats["value_absolute_error"] += float(value_error.abs().sum().item())
                stats["policy_top_agreement"] += int((tops[generation] == target_top).sum().item())
                for local, row in enumerate(chunk):
                    if row.visit_argmax_changed_raw_action:
                        stats["decisive_rows"] += 1
                        stats["decisive_top_agreement"] += int(
                            int(tops[generation][local].item()) == row.visit_argmax_row
                        )
                    family = str(
                        row.action_payloads[int(tops[generation][local].item())]["action_type"]
                    )
                    stats["raw_top_action_family_counts"][family] += 1
            for name, pair in pair_totals.items():
                left, right = (int(value) for value in name.split("_vs_"))
                pair["rows"] += len(chunk)
                pair["total_variation"] += float(
                    (0.5 * (probabilities[left] - probabilities[right]).abs().sum(dim=1))
                    .sum()
                    .item()
                )
                pair["top_changes"] += int((tops[left] != tops[right]).sum().item())
    checkpoint_metrics: dict[str, Any] = {}
    for generation, stats in totals.items():
        count = int(stats["rows"])
        decisive = int(stats["decisive_rows"])
        policy = float(stats["policy_cross_entropy"]) / count
        value = float(stats["value_squared_error"]) / count
        checkpoint_metrics[str(generation)] = {
            "rows": count,
            "policy_cross_entropy": policy,
            "value_mse": value,
            "value_mae": float(stats["value_absolute_error"]) / count,
            "objective_loss_policy_plus_half_value": policy + 0.5 * value,
            "policy_top_agreement": int(stats["policy_top_agreement"]) / count,
            "decisive_rows": decisive,
            "decisive_top_agreement": (
                int(stats["decisive_top_agreement"]) / decisive if decisive else 0.0
            ),
            "raw_top_action_family_counts": dict(
                sorted(stats["raw_top_action_family_counts"].items())
            ),
        }
    pairwise = {
        name: {
            "rows": int(stats["rows"]),
            "mean_total_variation": float(stats["total_variation"]) / int(stats["rows"]),
            "top_action_change_fraction": int(stats["top_changes"]) / int(stats["rows"]),
        }
        for name, stats in pair_totals.items()
    }
    return {
        "rows": len(rows),
        "games": len({row.game_seed for row in rows}),
        "checkpoint_metrics": checkpoint_metrics,
        "pairwise_policy_drift": pairwise,
    }


def _reconstruct_sampled_games(
    games: dict[int, list[AlphaZeroReplayRow]],
    *,
    replay_generation: int,
    seed_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    target_by_seed = {
        seed_start + offset: target
        for offset, target in zip(SAMPLED_SEED_OFFSETS, TARGET_TURNS, strict=True)
    }
    for seed, rows in games.items():
        selected_row = min(
            rows,
            key=lambda row: (abs(row.turn_index - target_by_seed[seed]), row.turn_index),
        )
        env = SplendorEnv(seed=seed)
        state = env.initial_state()
        open_deck_events: dict[tuple[int, str], dict[str, Any]] = {}
        for row in rows:
            if state.turn_index != row.turn_index or state.current_player != row.player_id:
                raise ValueError(f"Replay reconstruction diverged for seed {seed}.")
            if row.turn_index == selected_row.turn_index:
                snapshots.append(
                    {
                        "replay_generation": replay_generation,
                        "game_seed": seed,
                        "turn_index": row.turn_index,
                        "target_turn": target_by_seed[seed],
                        "state": state.copy_shallow(),
                    }
                )
            legal = env.legal_actions(state)
            payload = row.action_payloads[row.selected_action_row]
            matches = [action for action in legal if serialize_action(action) == payload]
            if len(matches) != 1:
                raise ValueError(f"Could not uniquely replay seed {seed} turn {row.turn_index}.")
            action = matches[0]
            reserving_player = state.current_player
            bought_card_id: str | None = None
            if action.action_type is ActionType.BUY_RESERVED:
                bought_card_id = state.players[reserving_player].reserved_cards[
                    int(action.reserved_index)
                ].card_id
            next_state = env.step(state, action)
            if action.action_type is ActionType.RESERVE_DECK:
                card = next_state.players[reserving_player].reserved_cards[-1]
                event = {
                    "replay_generation": replay_generation,
                    "game_seed": seed,
                    "player": reserving_player,
                    "reserve_turn": row.turn_index,
                    "card_id": card.card_id,
                    "card_tier": card.tier,
                    "card_points": card.points,
                    "purchased_later": False,
                    "purchase_turn": None,
                    "turns_to_purchase": None,
                    "turns_remaining": len(rows) - row.turn_index - 1,
                    "player_value_target": float(row.value_target),
                }
                events.append(event)
                open_deck_events[(reserving_player, card.card_id)] = event
            if bought_card_id is not None:
                event = open_deck_events.pop((reserving_player, bought_card_id), None)
                if event is not None:
                    event["purchased_later"] = True
                    event["purchase_turn"] = row.turn_index
                    event["turns_to_purchase"] = row.turn_index - int(event["reserve_turn"])
            state = next_state
        if not state.terminal:
            raise ValueError(f"Clean sampled game {seed} did not reconstruct to terminal.")
    purchased = [event for event in events if event["purchased_later"]]
    return snapshots, {
        "games": len(games),
        "events": len(events),
        "purchased_later": len(purchased),
        "purchase_fraction": len(purchased) / len(events) if events else 0.0,
        "mean_turns_to_purchase": (
            sum(int(event["turns_to_purchase"]) for event in purchased) / len(purchased)
            if purchased
            else None
        ),
        "mean_turns_remaining": (
            sum(int(event["turns_remaining"]) for event in events) / len(events)
            if events
            else None
        ),
        "mean_player_value_target": (
            sum(float(event["player_value_target"]) for event in events) / len(events)
            if events
            else None
        ),
        "event_rows": events,
    }


def _common_state_search_audit(
    snapshots: list[dict[str, Any]],
    *,
    checkpoint_paths: dict[int, Path],
    device: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for checkpoint_generation, checkpoint_path in checkpoint_paths.items():
        inference = AlphaZeroInference(checkpoint_path, device=device)
        search = AlphaZeroSearch(
            inference,
            SearchConfig(
                simulations=256,
                c_puct=2.25,
                mode="real",
                root_noise_fraction=0.0,
                seed=1_050_000 + checkpoint_generation,
            ),
        )
        records: list[dict[str, Any]] = []
        for snapshot in snapshots:
            state: SplendorState = snapshot["state"]
            env = SplendorEnv(seed=int(snapshot["game_seed"]))
            legal = env.legal_actions(state)
            result = search.run(
                env,
                state,
                legal,
                self_play=False,
                seed=(
                    1_050_000
                    + checkpoint_generation * 100_000
                    + int(snapshot["replay_generation"]) * 1_000
                    + int(snapshot["game_seed"]) % 1_000
                    + int(snapshot["turn_index"]) * 17
                ),
            )
            raw_top = result.root_edges[result.raw_policy_top_row]
            visit_top = result.root_edges[result.visit_argmax_row]
            reserve_edges = [
                edge for edge in result.root_edges if edge.action.action_type is ActionType.RESERVE_DECK
            ]
            nonreserve_edges = [
                edge for edge in result.root_edges if edge.action.action_type is not ActionType.RESERVE_DECK
            ]
            records.append(
                {
                    "replay_generation": int(snapshot["replay_generation"]),
                    "game_seed": int(snapshot["game_seed"]),
                    "turn_index": int(snapshot["turn_index"]),
                    "raw_top_family": raw_top.action.action_type.name,
                    "visit_top_family": visit_top.action.action_type.name,
                    "visit_changed_raw": result.visit_argmax_row != result.raw_policy_top_row,
                    "prior_visit_tv": 0.5
                    * sum(
                        abs(edge.probability - edge.network_prior)
                        for edge in result.root_edges
                    ),
                    "raw_network_value": result.raw_network_value,
                    "search_root_value": result.search_root_value,
                    "reserve_legal": bool(reserve_edges),
                    "reserve_prior_mass": sum(edge.network_prior for edge in reserve_edges),
                    "reserve_visit_mass": sum(edge.probability for edge in reserve_edges),
                    "reserve_best_q_minus_nonreserve_best_q": (
                        max(edge.q for edge in reserve_edges)
                        - max(edge.q for edge in nonreserve_edges)
                        if reserve_edges and nonreserve_edges
                        else None
                    ),
                }
            )
        output[str(checkpoint_generation)] = {
            "overall": _summarize_search_records(records),
            "by_replay_generation": {
                str(replay_generation): _summarize_search_records(
                    [row for row in records if row["replay_generation"] == replay_generation]
                )
                for replay_generation in sorted({row["replay_generation"] for row in records})
            },
            "records": records,
        }
    return output


def _summarize_search_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    reserve = [row for row in records if row["reserve_legal"]]
    q_rows = [
        row for row in reserve if row["reserve_best_q_minus_nonreserve_best_q"] is not None
    ]
    return {
        "states": len(records),
        "visit_changed_raw_fraction": sum(row["visit_changed_raw"] for row in records)
        / len(records),
        "mean_prior_visit_tv": sum(float(row["prior_visit_tv"]) for row in records)
        / len(records),
        "mean_search_minus_raw_value": sum(
            float(row["search_root_value"]) - float(row["raw_network_value"])
            for row in records
        )
        / len(records),
        "raw_top_action_family_counts": dict(
            sorted(Counter(str(row["raw_top_family"]) for row in records).items())
        ),
        "visit_top_action_family_counts": dict(
            sorted(Counter(str(row["visit_top_family"]) for row in records).items())
        ),
        "reserve_legal_states": len(reserve),
        "reserve_raw_top_fraction_when_legal": (
            sum(row["raw_top_family"] == "RESERVE_DECK" for row in reserve) / len(reserve)
            if reserve
            else 0.0
        ),
        "reserve_visit_top_fraction_when_legal": (
            sum(row["visit_top_family"] == "RESERVE_DECK" for row in reserve) / len(reserve)
            if reserve
            else 0.0
        ),
        "mean_reserve_prior_mass_when_legal": (
            sum(float(row["reserve_prior_mass"]) for row in reserve) / len(reserve)
            if reserve
            else 0.0
        ),
        "mean_reserve_visit_mass_when_legal": (
            sum(float(row["reserve_visit_mass"]) for row in reserve) / len(reserve)
            if reserve
            else 0.0
        ),
        "mean_reserve_best_q_minus_nonreserve_best_q": (
            sum(float(row["reserve_best_q_minus_nonreserve_best_q"]) for row in q_rows)
            / len(q_rows)
            if q_rows
            else None
        ),
    }


def _descriptive_flags(
    matrix: dict[str, Any], search_audit: dict[str, Any]
) -> dict[str, Any]:
    oldest = matrix["7"]["checkpoint_metrics"]
    newest = matrix["10"]["checkpoint_metrics"]
    old_delta = (
        oldest["11"]["objective_loss_policy_plus_half_value"]
        - oldest["8"]["objective_loss_policy_plus_half_value"]
    )
    new_delta = (
        newest["11"]["objective_loss_policy_plus_half_value"]
        - newest["8"]["objective_loss_policy_plus_half_value"]
    )
    final_search = search_audit["11"]["overall"]
    return {
        "gen11_minus_gen8_objective_on_replay7": old_delta,
        "gen11_minus_gen8_objective_on_replay10": new_delta,
        "forgetting_pattern_present": old_delta > 0.01 and new_delta < 0.0,
        "gen11_common_state_search_changes_raw_fraction": final_search[
            "visit_changed_raw_fraction"
        ],
        "gen11_common_state_search_materially_changes_raw": final_search[
            "visit_changed_raw_fraction"
        ]
        >= 0.20,
        "interpretation": "descriptive thresholds only; pooled gameplay controls the strength conclusion",
    }


def _verify_inputs(config: PlateauDiagnosisConfig) -> None:
    for name, path in _input_paths(config).items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 105 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_104.read_text(encoding="utf-8"))
    if prior.get("status") != "complete":
        raise ValueError("Frozen 104 result is not complete.")
    if prior.get("decision") != "generation_11_endpoint_not_compounded_035_skipped":
        raise ValueError("Frozen 104 result has the wrong decision.")
    if prior.get("internal_endpoint_decision") != "endpoint_practically_flat_diagnose_before_scaling":
        raise ValueError("Frozen 104 result has the wrong internal decision.")
    if prior.get("final_sandbox_checkpoint_sha256") != EXPECTED_HASHES["checkpoint_11"]:
        raise ValueError("Frozen 104 result does not end at exact gen11.")
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["installed_checkpoint"]:
        raise ValueError("Frozen 104 result did not retain exact gen2.")
    primary = prior["internal_endpoint"]["searched_generation_11_vs_searched_generation_8"]
    arena = primary["arena"]
    if (
        arena.get("candidate_wins") != 109
        or arena.get("candidate_losses") != 91
        or arena.get("draws") != 0
        or len(arena.get("rows", [])) != 200
        or primary["assessment"].get("passed")
    ):
        raise ValueError("Frozen 104 result lacks the exact 109-91 primary endpoint.")
    for generation, path in {
        8: config.checkpoint_8,
        9: config.checkpoint_9,
        10: config.checkpoint_10,
        11: config.checkpoint_11,
    }.items():
        _, identity = load_checkpoint(path, device=config.device)
        if identity.generation != generation:
            raise ValueError(f"Checkpoint {generation} has the wrong lineage identity.")


def _input_paths(config: PlateauDiagnosisConfig) -> dict[str, Path]:
    return {
        "result_104": config.result_104,
        "installed_checkpoint": config.installed_checkpoint,
        "checkpoint_8": config.checkpoint_8,
        "checkpoint_9": config.checkpoint_9,
        "checkpoint_10": config.checkpoint_10,
        "checkpoint_11": config.checkpoint_11,
        "replay_7": config.replay_7,
        "replay_8": config.replay_8,
        "replay_9": config.replay_9,
        "replay_10": config.replay_10,
    }


def _jsonable_config(config: PlateauDiagnosisConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )

