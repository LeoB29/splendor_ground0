"""Replay forensic and search-teacher health panel for programme 100."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

from .arena import alpha_zero_raw_factory, alpha_zero_search_factory, run_paired_arena
from .replay import AlphaZeroReplayRow, read_replay_jsonl
from .search import SearchConfig
from .search_primary import _arena_dict, _seat_score_shares


@dataclass(frozen=True, slots=True)
class SearchTeacherHealthConfig:
    output_dir: Path
    incumbent_checkpoint: Path
    child_checkpoint: Path
    replay_generation_2: Path
    replay_generation_3: Path
    result_099: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "ddf1853d839ba5737478def8bae2134b5a43f0a1d7ac2d01a675f828ddc0c7e1"
EXPECTED_HASHES = {
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "child_checkpoint": "4db34392dbf79cc91e9cce404dd6ba09400d19055d68bf61d3c4ad8ef9127c75",
    "replay_generation_2": "855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56",
    "replay_generation_3": "73afe6f16f784a1040dd5f215f15708a498b9c13582aa9969edb7766015a514e",
    "result_099": "37534c4d057d4dd8aef9e0d9973e8b9adf427130303aa322465de7d4701c1693",
}
DEADLOCK_SEED = 1_710_004


def value_calibration(records: Iterable[tuple[float, float, float]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"rows": 0}
    targets = [target for _raw, _search, target in rows]
    raw = [prediction for prediction, _search, _target in rows]
    search = [prediction for _raw, prediction, _target in rows]
    decisive = [index for index, target in enumerate(targets) if target != 0.0]

    def metrics(predictions: list[float]) -> dict[str, float | int]:
        errors = [prediction - target for prediction, target in zip(predictions, targets)]
        return {
            "mean_prediction": sum(predictions) / len(predictions),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "mse": sum(error * error for error in errors) / len(errors),
            "bias_prediction_minus_target": sum(errors) / len(errors),
            "decisive_sign_rows": len(decisive),
            "decisive_sign_accuracy": (
                sum(
                    (predictions[index] > 0.0) == (targets[index] > 0.0)
                    for index in decisive
                )
                / len(decisive)
                if decisive
                else 0.0
            ),
        }

    return {
        "rows": len(rows),
        "mean_target": sum(targets) / len(targets),
        "raw_network": metrics(raw),
        "search_root": metrics(search),
        "mean_search_minus_raw": sum(s - r for r, s in zip(raw, search)) / len(rows),
    }


def search_teacher_acceptance(arena: dict[str, Any]) -> dict[str, Any]:
    seats = _seat_score_shares(arena)
    checks = {
        "score_share": arena["score_share"] >= 0.55,
        "positive_margin": arena["mean_margin"] > 0.0,
        "seat_zero_nonlosing": seats[0] >= 0.5,
        "seat_one_nonlosing": seats[1] >= 0.5,
        "normal_terminations": arena["termination_reasons"]
        == {"completed": arena["games"]},
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "seat_score_shares": {str(key): value for key, value in seats.items()},
    }


def audit_replay(path: Path, *, expected_generation: int) -> dict[str, Any]:
    rows = read_replay_jsonl(path)
    if not rows or any(row.generation != expected_generation for row in rows):
        raise ValueError("Replay is empty or contains the wrong generation.")
    if any(row.value_target is None for row in rows):
        raise ValueError("Replay audit requires terminal value targets.")
    families: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "legal_states": 0,
            "network_prior_mass": 0.0,
            "search_visit_mass": 0.0,
            "raw_top": 0,
            "visit_top": 0,
            "sampled_selected": 0,
            "search_promotions": 0,
            "search_demotions": 0,
            "visited_q_sum": 0.0,
            "visited_q_actions": 0,
            "selected_outcome_sum": 0.0,
            "selected_outcome_rows": 0,
        }
    )
    strata: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    terminations: Counter[str] = Counter()
    game_rows: dict[int, list[AlphaZeroReplayRow]] = defaultdict(list)
    tv_sum = 0.0
    changed = 0
    selected_counts: Counter[str] = Counter()
    for row in rows:
        game_rows[row.game_seed].append(row)
        target = float(row.value_target)
        raw_value = float(row.raw_network_value)
        search_value = float(row.search_root_value)
        raw_top = int(row.raw_policy_top_row)
        visit_top = int(row.visit_argmax_row)
        selected = int(row.selected_action_row)
        action_families = [str(payload["action_type"]) for payload in row.action_payloads]
        raw_family = action_families[raw_top]
        visit_family = action_families[visit_top]
        selected_family = action_families[selected]
        selected_counts[selected_family] += 1
        changed_here = visit_top != raw_top
        changed += int(changed_here)
        tv_sum += 0.5 * sum(
            abs(visit - prior)
            for visit, prior in zip(
                row.policy_target, row.root_network_priors, strict=True
            )
        )
        labels = (
            "overall",
            f"phase:{_phase(row.turn_index)}",
            f"visit_family:{visit_family}",
            f"search_changed_raw:{str(changed_here).lower()}",
        )
        for label in labels:
            strata[label].append((raw_value, search_value, target))
        legal_families = set(action_families)
        for family in legal_families:
            families[family]["legal_states"] += 1
        for index, family in enumerate(action_families):
            families[family]["network_prior_mass"] += float(
                row.root_network_priors[index]
            )
            families[family]["search_visit_mass"] += float(row.policy_target[index])
            if int(row.root_visits[index]) > 0:
                families[family]["visited_q_sum"] += float(row.root_q_values[index])
                families[family]["visited_q_actions"] += 1
        families[raw_family]["raw_top"] += 1
        families[visit_family]["visit_top"] += 1
        families[selected_family]["sampled_selected"] += 1
        families[selected_family]["selected_outcome_sum"] += target
        families[selected_family]["selected_outcome_rows"] += 1
        if changed_here:
            families[visit_family]["search_promotions"] += 1
            families[raw_family]["search_demotions"] += 1
    for game in game_rows.values():
        reasons = {str(row.termination_reason) for row in game}
        if len(reasons) != 1:
            raise ValueError("Replay game has inconsistent termination reasons.")
        terminations[next(iter(reasons))] += 1
    positions = len(rows)
    family_result: dict[str, Any] = {}
    for family, metrics in sorted(families.items()):
        legal_states = int(metrics["legal_states"])
        q_actions = int(metrics["visited_q_actions"])
        outcome_rows = int(metrics["selected_outcome_rows"])
        family_result[family] = {
            **metrics,
            "legal_state_fraction": legal_states / positions,
            "mean_network_prior_mass_all_states": float(
                metrics["network_prior_mass"]
            )
            / positions,
            "mean_search_visit_mass_all_states": float(
                metrics["search_visit_mass"]
            )
            / positions,
            "raw_top_fraction": int(metrics["raw_top"]) / positions,
            "visit_top_fraction": int(metrics["visit_top"]) / positions,
            "sampled_selected_fraction": int(metrics["sampled_selected"]) / positions,
            "mean_visited_root_q": (
                float(metrics["visited_q_sum"]) / q_actions if q_actions else 0.0
            ),
            "mean_selected_outcome": (
                float(metrics["selected_outcome_sum"]) / outcome_rows
                if outcome_rows
                else 0.0
            ),
        }
    abnormal = {
        seed: str(game[0].termination_reason)
        for seed, game in sorted(game_rows.items())
        if game[0].termination_reason != "completed"
    }
    trace = _deadlock_trace(game_rows.get(DEADLOCK_SEED, []))
    result = {
        "generation": expected_generation,
        "games": len(game_rows),
        "positions": positions,
        "termination_reasons": dict(sorted(terminations.items())),
        "abnormal_games": abnormal,
        "mean_turns": positions / len(game_rows),
        "selected_action_family_counts": dict(sorted(selected_counts.items())),
        "visit_argmax_changed_raw_actions": changed,
        "visit_argmax_changed_raw_fraction": changed / positions,
        "mean_visit_to_network_prior_tv": tv_sum / positions,
        "action_families": family_result,
        "value_calibration": {
            label: value_calibration(records) for label, records in sorted(strata.items())
        },
        "deadlock_trace": trace,
    }
    del rows
    return result


def replay_delta(generation_2: dict[str, Any], generation_3: dict[str, Any]) -> dict[str, Any]:
    reserve_2 = generation_2["action_families"].get("RESERVE_DECK", {})
    reserve_3 = generation_3["action_families"].get("RESERVE_DECK", {})
    value_2 = generation_2["value_calibration"]["overall"]
    value_3 = generation_3["value_calibration"]["overall"]
    return {
        "generation_3_minus_generation_2": {
            "visit_argmax_changed_raw_fraction": generation_3[
                "visit_argmax_changed_raw_fraction"
            ]
            - generation_2["visit_argmax_changed_raw_fraction"],
            "mean_visit_to_network_prior_tv": generation_3[
                "mean_visit_to_network_prior_tv"
            ]
            - generation_2["mean_visit_to_network_prior_tv"],
            "reserve_deck": {
                key: float(reserve_3.get(key, 0.0)) - float(reserve_2.get(key, 0.0))
                for key in (
                    "raw_top_fraction",
                    "visit_top_fraction",
                    "sampled_selected_fraction",
                    "mean_network_prior_mass_all_states",
                    "mean_search_visit_mass_all_states",
                    "mean_visited_root_q",
                    "mean_selected_outcome",
                )
            },
            "raw_value_mae": value_3["raw_network"]["mae"]
            - value_2["raw_network"]["mae"],
            "search_value_mae": value_3["search_root"]["mae"]
            - value_2["search_root"]["mae"],
            "mean_search_minus_raw_value": value_3["mean_search_minus_raw"]
            - value_2["mean_search_minus_raw"],
        }
    }


def run_search_teacher_health_100(config: SearchTeacherHealthConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    result: dict[str, Any] = {
        "kind": "alphazero_search_teacher_health_100",
        "status": "running",
        "strength_claim": False,
        "checkpoint_trained": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "offline_audit": None,
        "search_vs_raw_panel": None,
        "installed_checkpoint": str(config.incumbent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.incumbent_checkpoint),
        "decision": "running",
    }
    _write_progress(output, result, started)
    generation_2 = audit_replay(config.replay_generation_2, expected_generation=2)
    _write_json(output / "offline_generation_002.json", generation_2)
    generation_3 = audit_replay(config.replay_generation_3, expected_generation=3)
    _write_json(output / "offline_generation_003.json", generation_3)
    result["offline_audit"] = {
        "generation_2": generation_2,
        "generation_3": generation_3,
        "comparison": replay_delta(generation_2, generation_3),
    }
    _write_progress(output, result, started)
    search = SearchConfig(
        simulations=256,
        c_puct=2.25,
        mode="real",
        root_noise_fraction=0.0,
        seed=1_000_100,
    )
    arena = _arena_dict(
        run_paired_arena(
            name="generation_3_search_vs_generation_3_raw_health_panel",
            candidate_factory=alpha_zero_search_factory(
                config.child_checkpoint, config.device, search
            ),
            baseline_factory=alpha_zero_raw_factory(
                config.child_checkpoint, config.device
            ),
            seed_start=1_790_000,
            paired_seeds=50,
            max_turns=config.max_turns,
            progress_path=output / "arenas" / "search_vs_raw.json",
            resume=True,
        )
    )
    acceptance = search_teacher_acceptance(arena)
    result["search_vs_raw_panel"] = {"arena": arena, "acceptance": acceptance}
    if acceptance["accepted"]:
        decision = "search_teacher_healthy_clean_resume_permitted"
    else:
        decision = "search_teacher_unhealthy_redesign_required"
    return _finish(output, result, started, decision)


def _deadlock_trace(rows: list[AlphaZeroReplayRow]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for row in rows:
        families = [str(payload["action_type"]) for payload in row.action_payloads]
        selected = int(row.selected_action_row)
        trace.append(
            {
                "turn_index": row.turn_index,
                "player_id": row.player_id,
                "legal_family_counts": dict(sorted(Counter(families).items())),
                "selected_family": families[selected],
                "raw_top_family": families[int(row.raw_policy_top_row)],
                "visit_top_family": families[int(row.visit_argmax_row)],
                "selected_visit_probability": float(row.policy_target[selected]),
                "selected_network_prior": float(row.root_network_priors[selected]),
                "selected_root_q": float(row.root_q_values[selected]),
                "assigned_value_target": float(row.value_target),
            }
        )
    return trace


def _phase(turn: int) -> str:
    if turn < 20:
        return "early_lt20"
    if turn < 50:
        return "middle_20_49"
    return "late_ge50"


def _verify_inputs(config: SearchTeacherHealthConfig) -> None:
    paths = {
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "child_checkpoint": config.child_checkpoint,
        "replay_generation_2": config.replay_generation_2,
        "replay_generation_3": config.replay_generation_3,
        "result_099": config.result_099,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 100 input hash mismatch for {name}: {actual}")
    prior = json.loads(config.result_099.read_text(encoding="utf-8"))
    abnormal = (prior.get("failed_selfplay") or {}).get("games_detail") or []
    abnormal = [row for row in abnormal if row.get("termination_reason") != "completed"]
    if prior.get("decision") != "invalid_selfplay_stopped_keep_generation_2":
        raise ValueError("Frozen 099 result has the wrong controlling decision.")
    if len(abnormal) != 1 or abnormal[0].get("seed") != DEADLOCK_SEED:
        raise ValueError("Frozen 099 result does not contain the exact single deadlock.")
    generations = prior.get("generations") or []
    if len(generations) != 1 or generations[0].get("child_checkpoint_sha256") != EXPECTED_HASHES["child_checkpoint"]:
        raise ValueError("Frozen 099 result has the wrong generation-3 child.")
    if prior.get("installed_checkpoint_sha256") != EXPECTED_HASHES["incumbent_checkpoint"]:
        raise ValueError("Frozen 099 result did not retain exact generation 2.")


def _jsonable_config(config: SearchTeacherHealthConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _finish(output: Path, result: dict[str, Any], started: float, decision: str) -> dict[str, Any]:
    result["status"] = "complete"
    result["decision"] = decision
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(output / "result.json", result)
    return result


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(
        output / "progress.json",
        {**result, "elapsed_seconds": time.perf_counter() - started},
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
