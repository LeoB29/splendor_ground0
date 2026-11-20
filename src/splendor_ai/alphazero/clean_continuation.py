"""Clean replay replacement and rolling-loop continuation for programme 101."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from .network import load_checkpoint
from .replay import AlphaZeroReplayRow, read_replay_jsonl, write_replay_jsonl
from .scaled_loop import _diagnostics, _training_config, _trajectory, training_validity
from .search import SearchConfig
from .selfplay import SelfPlayConfig, generate_self_play, summarize_self_play
from .training import train_successor


@dataclass(frozen=True, slots=True)
class CleanContinuationConfig:
    output_dir: Path
    incumbent_checkpoint: Path
    generation_3_checkpoint: Path
    replay_generation_2: Path
    contaminated_replay_generation_3: Path
    result_099: Path
    result_100: Path
    device: str = "cpu"
    max_turns: int = 200
    torch_threads: int | None = None


PROTOCOL_SHA256 = "9700955082bffab9adec50ea3d4069adece12dda9f10d161c2ea5e0e3a7f63f0"
EXPECTED_HASHES = {
    "incumbent_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "generation_3_checkpoint": "4db34392dbf79cc91e9cce404dd6ba09400d19055d68bf61d3c4ad8ef9127c75",
    "replay_generation_2": "855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56",
    "contaminated_replay_generation_3": "73afe6f16f784a1040dd5f215f15708a498b9c13582aa9969edb7766015a514e",
    "result_099": "37534c4d057d4dd8aef9e0d9973e8b9adf427130303aa322465de7d4701c1693",
    "result_100": "58bbaaee4079392537b8a4294d1a780811b711bac0f9edac9e8ac772b8161cc2",
}
TARGET_GAMES = 256
MAX_REPLACEMENT_ATTEMPTS = 16


def partition_game_terminations(
    rows: list[AlphaZeroReplayRow],
) -> tuple[list[int], dict[int, str]]:
    grouped: dict[int, list[AlphaZeroReplayRow]] = defaultdict(list)
    for row in rows:
        grouped[row.game_seed].append(row)
    clean: list[int] = []
    abnormal: dict[int, str] = {}
    for seed, game in sorted(grouped.items()):
        reasons = {str(row.termination_reason) for row in game}
        if len(reasons) != 1:
            raise ValueError("Replay game has inconsistent termination reasons.")
        reason = next(iter(reasons))
        if reason == "completed":
            clean.append(seed)
        else:
            abnormal[seed] = reason
    return clean, abnormal


def clean_replay_summary(
    rows: list[AlphaZeroReplayRow], *, expected_generation: int
) -> dict[str, Any]:
    clean_seeds, abnormal = partition_game_terminations(rows)
    grouped: dict[int, list[AlphaZeroReplayRow]] = defaultdict(list)
    for row in rows:
        grouped[row.game_seed].append(row)
    contiguous = all(
        [row.turn_index for row in game] == list(range(len(game)))
        for game in grouped.values()
    )
    exact_generation = all(
        row.generation == expected_generation
        and row.checkpoint_generation == expected_generation
        for row in rows
    )
    complete_targets = all(row.value_target is not None for row in rows)
    exact_dose = all(
        int(row.search_telemetry.get("simulations", -1)) == 256 for row in rows
    )
    checks = {
        "exact_game_count": len(grouped) == TARGET_GAMES,
        "unique_clean_games": len(clean_seeds) == TARGET_GAMES and not abnormal,
        "contiguous_turn_rows": contiguous,
        "exact_generation": exact_generation,
        "complete_targets": complete_targets,
        "exact_simulation_dose": exact_dose,
        "positive_positions": bool(rows),
    }
    selected = Counter(
        str(row.action_payloads[row.selected_action_row]["action_type"]) for row in rows
    )
    positions = len(rows)
    tv = sum(
        0.5
        * sum(
            abs(visit - prior)
            for visit, prior in zip(
                row.policy_target, row.root_network_priors, strict=True
            )
        )
        for row in rows
    )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "games": len(grouped),
        "positions": positions,
        "termination_reasons": {"completed": len(clean_seeds)} if not abnormal else {
            **dict(Counter(abnormal.values())),
            "completed": len(clean_seeds),
        },
        "mean_turns": positions / len(grouped) if grouped else 0.0,
        "selected_action_family_counts": dict(sorted(selected.items())),
        "reserve_deck_selected": selected.get("RESERVE_DECK", 0),
        "visit_argmax_changed_raw_fraction": (
            sum(row.visit_argmax_changed_raw_action for row in rows) / positions
            if positions
            else 0.0
        ),
        "sampled_action_changed_raw_fraction": (
            sum(row.search_changed_raw_action for row in rows) / positions
            if positions
            else 0.0
        ),
        "mean_visit_to_network_prior_tv": tv / positions if positions else 0.0,
    }


def sanitize_replay(
    *,
    source_path: Path,
    output_path: Path,
    checkpoint: Path,
    generation: int,
    original_seed_start: int,
    search_seed: int,
    config: CleanContinuationConfig,
) -> dict[str, Any]:
    rows = read_replay_jsonl(source_path)
    clean_seeds, abnormal = partition_game_terminations(rows)
    keep = set(clean_seeds)
    clean_rows = [row for row in rows if row.game_seed in keep]
    del rows
    attempts: list[dict[str, Any]] = []
    for attempt in range(MAX_REPLACEMENT_ATTEMPTS):
        if len(keep) >= TARGET_GAMES:
            break
        seed = original_seed_start + TARGET_GAMES + attempt
        replacement = generate_self_play(
            checkpoint_path=checkpoint,
            device=config.device,
            config=SelfPlayConfig(
                generation=generation,
                game_count=1,
                seed_start=seed,
                search=SearchConfig(
                    simulations=256,
                    c_puct=2.25,
                    mode="real",
                    root_noise_fraction=0.25,
                    seed=search_seed,
                ),
                max_turns=config.max_turns,
            ),
            replay_path=config.output_dir
            / "replacement_attempts"
            / f"generation_{generation:03d}_seed_{seed}.jsonl",
            progress_path=config.output_dir
            / "replacement_attempts"
            / f"generation_{generation:03d}_seed_{seed}.progress.json",
            resume=True,
        )
        summary = summarize_self_play(replacement)
        reason = replacement.games[0].termination_reason
        accepted = reason == "completed"
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": seed,
                "termination_reason": reason,
                "accepted": accepted,
                "winner": replacement.games[0].winner,
                "final_scores": list(replacement.games[0].final_scores),
                "turns": replacement.games[0].turns,
            }
        )
        if accepted:
            clean_rows.extend(replacement.rows)
            keep.add(seed)
        del replacement, summary
    if len(keep) != TARGET_GAMES:
        raise RuntimeError(
            f"Could not obtain {TARGET_GAMES} clean generation-{generation} games "
            f"within {MAX_REPLACEMENT_ATTEMPTS} attempts."
        )
    clean_rows.sort(key=lambda row: (row.game_seed, row.turn_index))
    write_replay_jsonl(output_path, clean_rows)
    summary = clean_replay_summary(clean_rows, expected_generation=generation)
    if not summary["valid"]:
        raise RuntimeError("Sanitized replay failed its frozen validity contract.")
    result = {
        "generation": generation,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "excluded_games": [
            {"seed": seed, "termination_reason": reason}
            for seed, reason in sorted(abnormal.items())
        ],
        "replacement_attempts": attempts,
        "accepted_replacement_seeds": [row["seed"] for row in attempts if row["accepted"]],
        "sanitized_path": str(output_path),
        "sanitized_sha256": _sha256(output_path),
        "summary": summary,
    }
    del clean_rows
    return result


def run_clean_continuation_101(config: CleanContinuationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prior_099, _prior_100 = _verify_inputs(config)
    inherited = prior_099["generations"][0]
    result: dict[str, Any] = {
        "kind": "alphazero_clean_continuation_101",
        "status": "running",
        "strength_claim": False,
        "sandbox_lineage": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "inherited_generation_3": inherited,
        "sanitation": {},
        "continued_generations": [],
        "active_stage": "sanitize_generation_3",
        "final_sandbox_checkpoint": None,
        "final_sandbox_checkpoint_sha256": None,
        "installed_checkpoint": str(config.incumbent_checkpoint),
        "installed_checkpoint_sha256": _sha256(config.incumbent_checkpoint),
        "decision": "running",
    }
    _write_progress(output, result, started)
    sanitized_3 = output / "selfplay" / "generation_003_sanitized.jsonl"
    sanitation_3 = sanitize_replay(
        source_path=config.contaminated_replay_generation_3,
        output_path=sanitized_3,
        checkpoint=config.generation_3_checkpoint,
        generation=3,
        original_seed_start=1_710_000,
        search_seed=990_100,
        config=config,
    )
    result["sanitation"]["generation_3"] = sanitation_3
    _write_progress(output, result, started)

    parent = config.generation_3_checkpoint
    previous_sanitized = sanitized_3
    previous_window_first = config.replay_generation_2
    for iteration, parent_generation in enumerate((3, 4), start=1):
        child_generation = parent_generation + 1
        result["active_stage"] = f"train_generation_{child_generation}"
        _write_progress(output, result, started)
        child = output / "checkpoints" / f"generation_{child_generation:03d}.pt"
        buffer_paths = [previous_window_first, previous_sanitized]
        training = train_successor(
            parent_checkpoint=parent,
            replay_paths=buffer_paths,
            output_checkpoint=child,
            metrics_path=output / "training" / f"generation_{child_generation:03d}.json",
            device=config.device,
            config=_training_config(iteration),
        )
        validity = training_validity(training, expected_parent=parent_generation)
        _, identity = load_checkpoint(child, device=config.device)
        validity["checks"]["checkpoint_roundtrip_generation"] = (
            identity.generation == child_generation
        )
        validity["valid"] = bool(
            validity["valid"] and validity["checks"]["checkpoint_roundtrip_generation"]
        )
        if not validity["valid"]:
            result["failed_training"] = training
            result["failed_training_validity"] = validity
            return _finish(
                output,
                result,
                started,
                "invalid_training_stopped_keep_generation_2",
            )
        result["active_stage"] = f"diagnostics_generation_{child_generation}"
        _write_progress(output, result, started)
        diagnostics = _diagnostics(
            output=output,
            config=config,  # type: ignore[arg-type]
            iteration=iteration,
            parent=parent,
            child=child,
        )
        selfplay_summary = sanitation_3["summary"] if parent_generation == 3 else result[
            "sanitation"
        ]["generation_4"]["summary"]
        generation_result = {
            "iteration": iteration + 1,
            "parent_generation": parent_generation,
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": _sha256(parent),
            "selfplay": selfplay_summary,
            "selfplay_validity": {"valid": True, "checks": selfplay_summary["checks"]},
            "replay_path": str(previous_sanitized),
            "replay_sha256": _sha256(previous_sanitized),
            "replay_window": [str(path) for path in buffer_paths],
            "child_generation": child_generation,
            "child_checkpoint": str(child),
            "child_checkpoint_sha256": _sha256(child),
            "training": training,
            "training_validity": validity,
            "diagnostics": diagnostics,
        }
        result["continued_generations"].append(generation_result)
        parent = child
        if parent_generation == 4:
            break
        result["active_stage"] = "selfplay_generation_4"
        _write_progress(output, result, started)
        raw_replay_4 = output / "selfplay" / "generation_004_raw.jsonl"
        raw_selfplay_4 = generate_self_play(
            checkpoint_path=parent,
            device=config.device,
            config=SelfPlayConfig(
                generation=4,
                game_count=TARGET_GAMES,
                seed_start=1_720_000,
                search=SearchConfig(
                    simulations=256,
                    c_puct=2.25,
                    mode="real",
                    root_noise_fraction=0.25,
                    seed=990_101,
                ),
                max_turns=config.max_turns,
            ),
            replay_path=raw_replay_4,
            progress_path=output / "selfplay" / "generation_004_raw.progress.json",
            resume=True,
        )
        result["raw_generation_4_selfplay"] = summarize_self_play(raw_selfplay_4)
        del raw_selfplay_4
        result["active_stage"] = "sanitize_generation_4"
        _write_progress(output, result, started)
        sanitized_4 = output / "selfplay" / "generation_004_sanitized.jsonl"
        sanitation_4 = sanitize_replay(
            source_path=raw_replay_4,
            output_path=sanitized_4,
            checkpoint=parent,
            generation=4,
            original_seed_start=1_720_000,
            search_seed=990_101,
            config=config,
        )
        result["sanitation"]["generation_4"] = sanitation_4
        previous_window_first = sanitized_3
        previous_sanitized = sanitized_4
        _write_progress(output, result, started)

    all_generations = [inherited, *result["continued_generations"]]
    result["active_stage"] = "complete"
    result["final_sandbox_checkpoint"] = str(parent)
    result["final_sandbox_checkpoint_sha256"] = _sha256(parent)
    result["trajectory"] = _trajectory(all_generations)
    return _finish(
        output,
        result,
        started,
        "clean_continuation_complete_review_trajectory",
    )


def _verify_inputs(
    config: CleanContinuationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "incumbent_checkpoint": config.incumbent_checkpoint,
        "generation_3_checkpoint": config.generation_3_checkpoint,
        "replay_generation_2": config.replay_generation_2,
        "contaminated_replay_generation_3": config.contaminated_replay_generation_3,
        "result_099": config.result_099,
        "result_100": config.result_100,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 101 input hash mismatch for {name}: {actual}")
    prior_099 = json.loads(config.result_099.read_text(encoding="utf-8"))
    prior_100 = json.loads(config.result_100.read_text(encoding="utf-8"))
    abnormal = (prior_099.get("failed_selfplay") or {}).get("games_detail") or []
    abnormal = [row for row in abnormal if row.get("termination_reason") != "completed"]
    if prior_099.get("decision") != "invalid_selfplay_stopped_keep_generation_2":
        raise ValueError("Frozen 099 result has the wrong controlling decision.")
    if len(abnormal) != 1 or abnormal[0].get("seed") != 1_710_004:
        raise ValueError("Frozen 099 result has the wrong abnormal game.")
    panel = (prior_100.get("search_vs_raw_panel") or {}).get("arena") or {}
    if prior_100.get("decision") != "search_teacher_healthy_clean_resume_permitted":
        raise ValueError("Frozen 100 result did not authorize clean continuation.")
    if (
        panel.get("candidate_wins") != 66
        or panel.get("draws") != 2
        or panel.get("candidate_losses") != 32
    ):
        raise ValueError("Frozen 100 result has the wrong health panel.")
    if prior_100.get("installed_checkpoint_sha256") != EXPECTED_HASHES["incumbent_checkpoint"]:
        raise ValueError("Frozen 100 result did not retain generation 2.")
    return prior_099, prior_100


def _jsonable_config(config: CleanContinuationConfig) -> dict[str, Any]:
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
